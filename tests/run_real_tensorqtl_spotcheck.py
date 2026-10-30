"""Opt-in, bounded real-data export and actual TensorQTL CLI compatibility check.

Run with the optional TensorQTL validation environment. This is intentionally
outside pytest discovery: it reads named production inputs and writes derived
artifacts only to --outdir. Matched donor SNPs are a complete-case exact
variant intersection, so association results are a technical check, without
population or biological interpretation. Separate simulated variants test
cis-window boundaries and are labeled in every filename and receipt.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from scipy.stats import linregress

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from mdb.grouping import load_group_index  # noqa: E402
from mdb.schema import TrackKey  # noqa: E402
from mdb.storage import load_cohort_index, load_cohort_manifest, load_view_columns, load_view_reader  # noqa: E402

KEY = TrackKey("5mC", "combined", "combined")


def regional_source(path: Path, chrom: str, start: int, end: int) -> dict:
    manifest = load_cohort_manifest(str(path))
    chroms, offsets, positions = load_cohort_index(str(path))
    ci = chroms.index(chrom)
    chrom_lo = int(offsets[ci])
    chrom_hi = int(offsets[ci + 1]) if ci + 1 < len(offsets) else len(positions)
    if manifest.get("representation") == "cpg_groups":
        groups = load_group_index(str(path))
        rows = np.flatnonzero((groups.start[chrom_lo:chrom_hi] < end) & (groups.end[chrom_lo:chrom_hi] > start)) + chrom_lo
        starts, ends = groups.start[rows], groups.end[rows]
        feature_ids = [f"{groups.method}_{i + 1}" for i in rows]
    else:
        lo = chrom_lo + int(np.searchsorted(positions[chrom_lo:chrom_hi], start))
        hi = chrom_lo + int(np.searchsorted(positions[chrom_lo:chrom_hi], end))
        rows = np.arange(lo, hi)
        starts, ends = positions[rows], positions[rows].astype(np.int64) + 1
        feature_ids = [f"{chrom}:{int(p) + 1}" for p in starts]
    if not len(rows):
        raise ValueError(f"No rows overlap the requested region: {path}")
    if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
        raise ValueError("This spot-check runner expects contiguous regional source rows")
    reader, _, _ = load_view_reader(str(path), KEY)
    try:
        values = reader.read_rows(slice(int(rows[0]), int(rows[-1] + 1)))
    finally:
        reader.close()
    return {
        "path": path, "manifest": manifest, "global_rows": rows, "start": starts,
        "end": ends, "ids": feature_ids, "values": values,
        "samples": load_view_columns(str(path), KEY)["sample_id"],
    }


def choose_samples(single: dict, grouped: dict, available_donors: set[str], count: int) -> tuple[list[str], dict]:
    common = sorted(set(single["samples"]) & set(grouped["samples"]))
    common = [x for x in common if x.split("-")[0] in available_donors]
    si = np.asarray([single["samples"].index(x) for x in common])
    gi = np.asarray([grouped["samples"].index(x) for x in common])
    sv, gv = single["values"][:, si], grouped["values"][:, gi]
    scores = np.minimum(np.isfinite(sv).mean(axis=0), np.isfinite(gv).mean(axis=0))
    candidates = []
    for anchor, group_values in enumerate(gv):
        by_donor = {}
        for j in np.flatnonzero(np.isfinite(group_values)):
            donor = common[j].split("-")[0]
            previous = by_donor.get(donor)
            if previous is None or (-scores[j], common[j]) < (-scores[previous], common[previous]):
                by_donor[donor] = int(j)
        ranked = sorted(by_donor.values(), key=lambda j: (-scores[j], common[j]))[:count]
        if len(ranked) != count:
            continue
        selected_groups = gv[:, ranked]
        group_complete = np.isfinite(selected_groups).all(axis=1)
        group_variable = np.ptp(selected_groups[group_complete], axis=1) > 0
        selected_single = sv[:, ranked]
        single_complete = np.isfinite(selected_single).all(axis=1)
        single_variable = np.ptp(selected_single[single_complete], axis=1) > 0
        names = [common[j] for j in ranked]
        rank = (-int(group_variable.sum()), -int(single_variable.sum()), -float(scores[ranked].sum()), tuple(names), anchor)
        candidates.append((rank, names, anchor, int(single_variable.sum()), int(group_variable.sum())))
    if not candidates:
        raise ValueError("No group has enough observed independent donors for a complete-case spot check; use another bounded region")
    _, names, anchor, single_count, group_count = min(candidates)
    return names, {
        "method": "deterministic anchor-group search; one sample per donor; maximize complete variable grouped rows then CpG rows; tie-break by minimum regional completeness then sample ID",
        "anchor_group_id": grouped["ids"][anchor],
        "single_complete_variable_rows": single_count, "grouped_complete_variable_rows": group_count,
        "regional_single_rows_scanned": len(single["ids"]), "regional_group_rows_scanned": len(grouped["ids"]),
        "all_view_samples_scanned": len(single["samples"]),
    }


def write_vcf(path: Path, donors: list[str], records: list[dict]) -> None:
    with path.open("w") as handle:
        handle.write('##fileformat=VCFv4.2\n##contig=<ID=chr20>\n##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(donors) + "\n")
        for record in records:
            gt = ["0/0" if x == 0 else "0/1" if x == 1 else "1/1" for x in record["dosages"]]
            handle.write("\t".join([record["chrom"], str(record["pos"]), record["id"], record["ref"], record["alt"], ".", "PASS", ".", "GT", *gt]) + "\n")


def matched_snps(vcf_paths: dict[str, Path], donors: list[str], chrom: str, lo: int, hi: int, limit: int) -> tuple[list[dict], dict]:
    calls = []
    scan_counts = {}
    for donor in donors:
        called = {}
        ambiguous = set()
        fetched = 0
        with pysam.VariantFile(str(vcf_paths[donor])) as handle:
            if donor not in handle.header.samples:
                raise ValueError(f"VCF sample ID {donor} is missing from {vcf_paths[donor]}")
            for record in handle.fetch(chrom, max(lo - 1, 0), hi):
                fetched += 1
                if len(record.ref) != 1 or not record.alts or len(record.alts) != 1 or len(record.alts[0]) != 1:
                    continue
                if record.ref not in "ACGT" or record.alts[0] not in "ACGT":
                    continue
                gt = record.samples[donor].get("GT")
                if gt is None or len(gt) != 2 or any(x not in (0, 1) for x in gt):
                    continue
                key = (record.chrom, record.pos, record.ref, record.alts[0])
                dosage = sum(gt)
                if key in called and called[key] != dosage:
                    ambiguous.add(key)
                called[key] = dosage
        for key in ambiguous:
            called.pop(key, None)
        calls.append(called)
        scan_counts[donor] = {"fetched_records": fetched, "called_biallelic_snv_keys": len(called), "ambiguous_keys_excluded": len(ambiguous)}
    common = sorted(set.intersection(*(set(x) for x in calls)))
    polymorphic = [key for key in common if len({x[key] for x in calls}) > 1]
    selected = polymorphic
    if len(selected) > limit:
        selected = [selected[i] for i in np.linspace(0, len(selected) - 1, limit, dtype=int)]
    if not selected:
        raise ValueError("No nonconstant exact called biallelic SNV intersection across the selected donors")
    records = [{"chrom": key[0], "pos": key[1], "ref": key[2], "alt": key[3], "id": f"REAL_{key[0]}_{key[1]}_{key[2]}_{key[3]}", "dosages": [x[key] for x in calls]} for key in selected]
    return records, {
        "source_vcfs": {d: str(vcf_paths[d]) for d in donors}, "scan_counts": scan_counts,
        "fetch_region_one_based_inclusive": [chrom, lo, hi], "exact_called_snv_intersection": len(common),
        "nonconstant_intersection": len(polymorphic), "variants_selected": len(selected),
        "maximum_selected_variants": limit,
        "selection": "evenly spaced sorted exact variant keys, after requiring a diploid 0/1 call in every donor and nonconstant sum(GT)",
        "absent_variants_inferred_homozygous_reference": False,
        "interpretation": "technical compatibility only; intersection ascertainment, small cohort, and no analysis covariates preclude biological inference",
        "vcf_phase": "source dosage preserved; phase intentionally not needed for nominal dosage regression",
    }


def boundary_records(frames: dict[str, pd.DataFrame], donors: list[str], window: int, seed: int) -> tuple[list[dict], list[dict]]:
    definitions = []
    all_lower, all_upper = [], []
    for name, frame in frames.items():
        first = frame.iloc[0]
        lower, upper = int(first["start"]) + 1 - window, int(first["end"]) + window
        for label, pos in [("left_outside", lower - 1), ("left_on", lower), ("left_inside", lower + 1), ("right_inside", upper - 1), ("right_on", upper), ("right_outside", upper + 1)]:
            definitions.append({"id": f"SIMULATED_{name}_{label}", "chrom": str(first["#chr"]), "pos": pos, "ref": "A", "alt": "C", "anchor_phenotype": str(first["phenotype_id"]), "expected_for_anchor": "outside" not in label})
        all_lower.append(int(frame["start"].min()) + 1 - window)
        all_upper.append(int(frame["end"].max()) + window)
    definitions += [
        {"id": "SIMULATED_global_left_excluded", "chrom": "chr20", "pos": min(all_lower) - 1, "ref": "A", "alt": "C"},
        {"id": "SIMULATED_global_right_excluded", "chrom": "chr20", "pos": max(all_upper) + 1, "ref": "A", "alt": "C"},
    ]
    rng = np.random.default_rng(seed)
    records = []
    for item in sorted(definitions, key=lambda x: (x["pos"], x["id"])):
        dosage = rng.binomial(2, 0.4, len(donors))
        dosage[0], dosage[-1] = 0, 2
        records.append({**item, "dosages": dosage.tolist()})
    return records, definitions


def source_verification(source: dict, frame: pd.DataFrame, samples: list[str], donors: list[str], output: Path) -> dict:
    columns = [source["samples"].index(x) for x in samples]
    values = source["values"][:, columns]
    complete = np.isfinite(values).all(axis=1)
    keep = complete & (np.ptp(values, axis=1) > 0)
    expected_ids = np.asarray(source["ids"])[keep].tolist()
    assert frame["phenotype_id"].tolist() == expected_ids
    assert frame.columns[4:].tolist() == donors
    np.testing.assert_array_equal(frame["start"], source["start"][keep])
    np.testing.assert_array_equal(frame["end"], source["end"][keep])
    observed = frame[donors].to_numpy()
    expected = values[keep]
    np.testing.assert_allclose(observed, expected, atol=1e-9, rtol=0)
    pd.DataFrame({"phenotype_id": expected_ids, "source_global_row": source["global_rows"][keep], "start": source["start"][keep], "end": source["end"][keep]}).to_csv(output, sep="\t", index=False)
    np.savez_compressed(output.with_suffix(".values.npz"), source_values=expected, exported_values=observed, donor_ids=np.asarray(donors), phenotype_ids=np.asarray(expected_ids))
    return {"source_rows_considered": len(values), "exported_rows_checked": int(keep.sum()), "source_cells_checked": int(expected.size), "max_absolute_beta_error": float(np.max(np.abs(observed - expected))), "source_check_table": str(output), "source_check_values": str(output.with_suffix(".values.npz")), "imputed_cells": 0}


def verify_nominal(frame: pd.DataFrame, pairs: pd.DataFrame, genotypes: pd.DataFrame, variants: pd.DataFrame, window: int) -> dict:
    expected = set()
    for row in frame.itertuples(index=False, name=None):
        chrom, start, end, pid = row[:4]
        keep = (variants["chrom"] == chrom) & (variants["pos"] >= int(start) + 1 - window) & (variants["pos"] <= int(end) + window)
        expected.update((pid, vid) for vid in variants.index[keep])
    observed = set(zip(pairs["phenotype_id"], pairs["variant_id"], strict=True))
    assert observed == expected, ("cis-window pair mismatch", len(observed), len(expected), list(observed ^ expected)[:5])
    assert len(observed) == len(pairs)
    phenotypes = frame.set_index("phenotype_id").iloc[:, 3:]
    assert np.isfinite(pairs[["pval_nominal", "slope"]].to_numpy()).all()
    max_slope, max_logp = 0.0, 0.0
    zero_correlation_standard_errors = []
    near_perfect_fits = []
    for row in pairs.itertuples(index=False):
        x = genotypes.loc[row.variant_id, phenotypes.columns].to_numpy(dtype=np.float32).astype(np.float64)
        y = phenotypes.loc[row.phenotype_id].to_numpy(dtype=np.float32).astype(np.float64)
        # TensorQTL actually regresses float32 input. Calculate the independent
        # reference in float64 on those same values, without an extra decimal
        # text-to-float64 discrepancy at exact zero correlation.
        reference = linregress(x, y)
        np.testing.assert_allclose(row.slope, reference.slope, atol=2e-6, rtol=2e-4)
        max_slope = max(max_slope, abs(row.slope - reference.slope))
        if abs(reference.rvalue) >= 1 - 1e-12:
            assert np.var(x) > 0 and np.var(y) > 0
            residual = y - (reference.intercept + reference.slope * x)
            residual_fraction = float(np.dot(residual, residual) / np.dot(y - y.mean(), y - y.mean()))
            assert residual_fraction <= 1e-10
            assert np.isfinite(row.slope_se) and 0 <= row.slope_se <= 0.01 * abs(reference.slope)
            assert 0 <= row.pval_nominal <= 1e-10
            log_error = None
            if row.pval_nominal > 0 and reference.pvalue > 0:
                log_error = float(abs(np.log10(row.pval_nominal) - np.log10(reference.pvalue)))
            near_perfect_fits.append({
                "phenotype_id": row.phenotype_id, "variant_id": row.variant_id,
                "scipy_correlation": float(reference.rvalue), "residual_sse_over_sst": residual_fraction,
                "tensorqtl_slope": float(row.slope), "scipy_slope": float(reference.slope),
                "tensorqtl_standard_error": float(row.slope_se), "scipy_standard_error": float(reference.stderr),
                "tensorqtl_pvalue": float(row.pval_nominal), "scipy_pvalue": float(reference.pvalue),
                "absolute_log10_pvalue_difference": log_error,
                "zero_pvalue_prevents_log_comparison": bool(row.pval_nominal == 0 or reference.pvalue == 0),
                "interpretation": "independently near-perfect fit; float32 correlation makes p-value and SE numerically unstable; raw outputs retained",
            })
            continue
        if np.isnan(row.slope_se):
            # TensorQTL 1.0.10 computes abs(slope)/sqrt(tstat2). Exact zero
            # correlation gives 0/0 even though p=1 and the true SE is finite.
            assert row.slope == 0 and row.pval_nominal == 1
            assert abs(reference.slope) < 1e-12 and abs(reference.rvalue) < 1e-12 and np.isfinite(reference.stderr)
            zero_correlation_standard_errors.append({"phenotype_id": row.phenotype_id, "variant_id": row.variant_id, "tensorqtl_pvalue": row.pval_nominal, "tensorqtl_slope": row.slope, "tensorqtl_standard_error": "NaN from upstream zero correlation 0/0", "scipy_standard_error": float(reference.stderr), "scipy_correlation": float(reference.rvalue)})
        else:
            assert np.isfinite(row.slope_se)
            np.testing.assert_allclose(row.slope_se, reference.stderr, atol=2e-6, rtol=2e-4)
        np.testing.assert_allclose(row.pval_nominal, reference.pvalue, atol=0, rtol=1e-3)
        max_logp = max(max_logp, abs(np.log10(row.pval_nominal) - np.log10(reference.pvalue)))
    return {"nominal_pairs": len(pairs), "pairs_checked_against_inclusive_cis_window": len(expected), "pairs_checked_against_independent_scipy_regression": len(pairs), "reference_input_precision": "float32 exactly as used by TensorQTL, followed by independent float64 SciPy calculation", "slopes_checked_for_all_pairs": len(pairs), "ordinary_pair_pvalues_checked": len(pairs) - len(near_perfect_fits), "max_absolute_slope_error": float(max_slope), "max_absolute_log10_pvalue_error": float(max_logp), "max_absolute_log10_pvalue_error_scope": "ordinary pairs excluding independently near-perfect fits", "all_pvalues_and_slopes_finite": True, "finite_standard_errors_checked": len(pairs) - len(zero_correlation_standard_errors) - len(near_perfect_fits), "upstream_zero_correlation_nan_standard_error_count": len(zero_correlation_standard_errors), "upstream_zero_correlation_nan_standard_errors": zero_correlation_standard_errors, "upstream_near_perfect_fit_precision_edge_count": len(near_perfect_fits), "upstream_near_perfect_fit_precision_edges": near_perfect_fits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpg-db", dest="single_store", type=Path, required=True)
    parser.add_argument("--group-db", dest="grouped_store", type=Path, required=True)
    parser.add_argument("--genotype-dir", dest="vcf_dir", type=Path, required=True)
    parser.add_argument("--outdir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--region", default="chr20:30000000-30100000")
    parser.add_argument("--window", type=int, default=100000)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-variants", type=int, default=128)
    parser.add_argument("--plink2", default=shutil.which("plink2"))
    args = parser.parse_args()
    if args.samples < 8 or args.samples > 24 or not 1 <= args.window <= 1_000_000 or args.max_variants < 1:
        raise ValueError("Use 8-24 donors, a cis window of 1-1000000 bp, and a positive variant limit")
    if args.plink2 is None:
        raise ValueError("PLINK2 must be on PATH or supplied with --plink2")
    output = args.output_dir.resolve()
    protected = [args.single_store.resolve(), args.grouped_store.resolve(), args.vcf_dir.resolve()]
    if any(output.is_relative_to(path) for path in protected):
        raise ValueError("Output directory must be outside both source databases and the genotype input directory")
    output.mkdir(parents=True, exist_ok=True)
    chrom, interval = args.region.split(":")
    start, end = map(int, interval.split("-"))
    if chrom != "chr20" or end - start > 1_000_000:
        raise ValueError("This named-input spot check is limited to at most 1 Mb on chr20")
    source_paths = {"single_cpg": args.single_store.resolve(), "denovo_grouped": args.grouped_store.resolve()}
    snapshots = {str(p / f): (p / f).stat().st_mtime_ns for p in source_paths.values() for f in ("manifest.json", "index.npz")}
    sources = {name: regional_source(path, chrom, start, end) for name, path in source_paths.items()}
    vcf_paths = {p.name.split("_")[0]: p.resolve() for p in args.vcf_dir.glob("SMHT*_dshybrid_202503.01_aouimpute_1_whatshap_2.8.vcf.gz")}
    samples, selection = choose_samples(sources["single_cpg"], sources["denovo_grouped"], set(vcf_paths), args.samples)
    donors = [x.split("-")[0] for x in samples]
    print(json.dumps({"stage": "samples selected", "donors": donors, "selection": selection}), flush=True)
    (output / "selected_source_samples.txt").write_text("\n".join(samples) + "\n")
    pd.DataFrame({"sample_id": samples, "genotype_id": donors}).to_csv(output / "sample_map.tsv", sep="\t", index=False)
    commands = []
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})

    def run(command: list[str], log: Path) -> None:
        command = list(map(str, command))
        commands.append(command)
        with log.open("w") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment, cwd=output, check=True)

    import tensorqtl
    import torch
    from tensorqtl.genotypeio import load_genotypes

    frames, export_results = {}, {}
    for name, source in sources.items():
        bed = output / f"{name}.methylation.bed.gz"
        run([sys.executable, "-m", "mdb.main", "export-tensorqtl", "-i", source["path"], "-o", bed, "--region", args.region, "--sample-file", output / "selected_source_samples.txt", "--sample-map", output / "sample_map.tsv", "--missing", "drop", "--min-frac-present", "1", "--batch-rows", "1024", "--overwrite"], output / f"{name}.export.log")
        frame = pd.read_csv(bed, sep="\t")
        actual_phenotypes, actual_positions = tensorqtl.read_phenotype_bed(str(bed))
        assert actual_phenotypes.columns.tolist() == donors
        assert len(actual_positions) == len(frame)
        export_results[name] = source_verification(source, frame, samples, donors, output / f"{name}.source_checks.tsv")
        export_results[name]["export_receipt"] = json.loads(Path(str(bed) + ".export.json").read_text())
        frames[name] = frame
        print(json.dumps({"stage": "export verified", "representation": name, "rows": len(frame)}), flush=True)

    lower = min(int(f["start"].min()) + 1 - args.window for f in frames.values())
    upper = max(int(f["end"].max()) + args.window for f in frames.values())
    real_records, real_metadata = matched_snps(vcf_paths, donors, chrom, lower, upper, args.max_variants)
    simulated, definitions = boundary_records(frames, donors, args.window, 20261030)
    genotype_cases = {"REAL_matched_donor_genotypes": real_records, "SIMULATED_boundary_genotypes": simulated}
    cli_results = {}
    for case, records in genotype_cases.items():
        prefix = output / case
        vcf = output / f"{case}.vcf"
        write_vcf(vcf, donors, records)
        run([args.plink2, "--vcf", vcf, "--make-bed", "--output-chr", "chrM", "--out", prefix, "--threads", "1", "--memory", "1024"], output / f"{case}.plink2.stdout.log")
        genotypes, variants = load_genotypes(str(prefix), select_samples=donors)
        assert genotypes.columns.tolist() == donors
        expected_dosage = pd.DataFrame([r["dosages"] for r in records], index=[r["id"] for r in records], columns=donors)
        np.testing.assert_array_equal(genotypes.loc[expected_dosage.index], expected_dosage)
        assert variants["chrom"].eq(chrom).all()
        print(json.dumps({"stage": "actual PLINK genotype files verified", "case": case, "variants": len(variants)}), flush=True)
        for name, frame in frames.items():
            label = f"{case}.{name}"
            run([sys.executable, "-m", "tensorqtl", prefix, output / f"{name}.methylation.bed.gz", label, "--mode", "cis_nominal", "--window", str(args.window), "--output_dir", output], output / f"{label}.cli.stdout.log")
            parquet = output / f"{label}.cis_qtl_pairs.{chrom}.parquet"
            pairs = pd.read_parquet(parquet)
            checked = verify_nominal(frame, pairs, genotypes, variants, args.window)
            checked.update({"genotype_case": case, "representation": name, "phenotypes": len(frame), "genotype_variants": len(variants), "tensorqtl_loaded_alt_dosages_equal_vcf_gt_sums": True, "intentional_allele_flip": False, "nominal_parquet": str(parquet), "cli_stdout_log": str(output / f"{label}.cli.stdout.log")})
            diagnostics = output / f"{label}.upstream_zero_correlation_se.tsv"
            pd.DataFrame(checked["upstream_zero_correlation_nan_standard_errors"], columns=["phenotype_id", "variant_id", "tensorqtl_pvalue", "tensorqtl_slope", "tensorqtl_standard_error", "scipy_standard_error", "scipy_correlation"]).to_csv(diagnostics, sep="\t", index=False)
            checked["upstream_zero_correlation_diagnostic_tsv"] = str(diagnostics)
            if case.startswith("SIMULATED"):
                pair_set = set(zip(pairs["phenotype_id"], pairs["variant_id"], strict=True))
                relevant = [d for d in definitions if "anchor_phenotype" in d and d["id"].startswith(f"SIMULATED_{name}_")]
                for boundary in relevant:
                    assert ((boundary["anchor_phenotype"], boundary["id"]) in pair_set) == boundary["expected_for_anchor"]
                assert not pairs["variant_id"].str.startswith("SIMULATED_global_").any()
                checked["explicit_one_bp_boundary_tests"] = relevant
                checked["globally_outside_variants_excluded"] = True
            cli_results[label] = checked
            print(json.dumps({"stage": "actual TensorQTL CLI verified", "case": label, "pairs": len(pairs)}), flush=True)
    for filename, mtime in snapshots.items():
        assert Path(filename).stat().st_mtime_ns == mtime, f"Production metadata changed: {filename}"
    script = output / "run_commands.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n# REAL donor genotype runs are technical checks. SIMULATED runs only test boundaries.\nexport PYTHONPATH=" + shlex.quote(environment["PYTHONPATH"]) + "\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n" + "\n".join(shlex.join(c) for c in commands) + "\n")
    script.chmod(0o755)
    receipt = {
        "status": "passed", "region_zero_based_half_open": args.region, "tensorqtl_cis_window_bp": args.window,
        "window_rule": "inclusive one-based variant positions in [BED start+1-window, BED end+window]",
        "phenotype": "real ONT 5mC combined haplotype/strand methylation beta fractions; original grouped means retained",
        "missing_policy": "drop", "imputed_cells": 0, "source_sample_ids": samples, "genotype_donor_ids": donors,
        "sample_selection": selection, "exports": export_results, "real_genotype_selection": real_metadata,
        "simulated_boundary_variants": definitions, "tensorqtl_cli_runs": cli_results,
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "versions": {x: importlib.metadata.version(x) for x in ["tensorqtl", "torch", "numpy", "pandas", "scipy", "pandas-plink", "pysam", "pyarrow", "zarr"]},
        "production_metadata_mtimes_unchanged": True, "replay_shell_script": str(script),
        "biological_inference": "none; a technical compatibility check with a small selected donor set, ascertained shared SNPs, and no covariates",
    }
    (output / "validation_summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "passed", "summary": str(output / "validation_summary.json"), "commands": str(script)}), flush=True)


if __name__ == "__main__":
    main()
