"""Opt-in PacBio export and actual TensorQTL CLI spot check on a bounded region.

This runner reads count-based and score-based production PacBio stores, makes
an exact raw-value regional snapshot of the count store, and runs the public
de novo grouping function on that snapshot. The grouped output is a new local
regional derivation, not a pre-existing whole-cohort grouping. Matched donor
SNPs are a called-site intersection and results are technical checks only.
All paths are supplied explicitly; production inputs are never modified.
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

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from mdb.grouping import group_cohort, load_group_index  # noqa: E402
from mdb.storage import create_cohort_store, create_view_store, load_view_columns, load_view_reader  # noqa: E402
from run_real_tensorqtl_spotcheck import (  # noqa: E402
    KEY, choose_samples, matched_snps, regional_source, source_verification,
    verify_nominal, write_vcf,
)


def regional_snapshot(source: dict, output: Path) -> tuple[np.ndarray, dict]:
    """Preserve exact offset-encoded values, missing sentinels, and metadata."""
    rows = source["global_rows"]
    reader, _, _ = load_view_reader(str(source["path"]), KEY)
    try:
        raw = reader._read_raw(slice(int(rows[0]), int(rows[-1] + 1)))
    finally:
        reader.close()
    columns = load_view_columns(str(source["path"]), KEY)
    assert set(columns["platform"]) == {"pacbio"}
    offsets = np.asarray([0], dtype=np.int64)
    create_cohort_store(str(output), ["chr20"], offsets, source["start"], backend="npy", block_size=64)
    writer = create_view_store(
        str(output), KEY, ["chr20"], offsets, len(rows), columns["sample_id"],
        columns["bundle_path"], columns["platform"], columns["source_path"], columns["input_tag"], 64,
    )
    try:
        writer.write_dense_chrom_block("chr20", 0, len(rows), 0, len(columns["sample_id"]), raw)
    finally:
        writer.close()
    check, _, _ = load_view_reader(str(output), KEY)
    try:
        np.testing.assert_array_equal(check._read_raw(slice(0, len(rows))), raw)
        np.testing.assert_array_equal(check.read_rows(slice(0, len(rows))), source["values"])
    finally:
        check.close()
    assert load_view_columns(str(output), KEY) == columns
    provenance = {
        "original_production_database": str(source["path"]),
        "regional_snapshot": str(output), "source_global_row_start": int(rows[0]),
        "source_global_row_end_exclusive": int(rows[-1] + 1),
        "cpg_rows": len(rows), "sample_columns": len(columns["sample_id"]),
        "raw_uint16_cells_checked": int(raw.size),
        "raw_missing_zero_cells": int((raw == 0).sum()),
        "raw_observed_unmethylated_one_cells": int((raw == 1).sum()),
        "observed_zero_methylation_is_distinct_from_missing": True,
        "source_column_metadata_preserved": True, "platform": "pacbio",
        "units": "beta fraction [0,1]; raw zero=missing, raw one=observed beta zero",
        "snapshot_raw_values_match_production_exactly": True,
    }
    (output / "snapshot_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return raw, provenance


def verify_grouped_means(grouped: Path, raw_source: np.ndarray, output: Path) -> dict:
    groups = load_group_index(str(grouped))
    reader, _, _ = load_view_reader(str(grouped), KEY)
    try:
        observed = reader._read_raw(slice(0, groups.n_groups))
    finally:
        reader.close()
    expected = np.zeros(observed.shape, dtype=np.uint16)
    for i, (lo, hi) in enumerate(zip(groups.source_row_start, groups.source_row_end, strict=True)):
        raw = raw_source[lo:hi].astype(np.int64)
        present = raw > 0
        counts = present.sum(axis=0)
        sums = np.where(present, raw - 1, 0).sum(axis=0, dtype=np.int64)
        valid = counts > 0
        expected[i, valid] = np.rint(sums[valid] / counts[valid]).astype(np.uint16) + 1
    np.testing.assert_array_equal(observed, expected)
    np.savez_compressed(output, expected_group_raw_uint16=expected, observed_group_raw_uint16=observed)
    return {
        "group_rows": groups.n_groups, "sample_columns": observed.shape[1],
        "raw_group_cells_checked": int(observed.size),
        "group_arithmetic_means_and_missing_values_match_original_pacbio_counts_exactly": True,
        "rounding": "mean of present original integer beta numerators; round to nearest 1/10000, then offset +1",
        "evidence_npz": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-db", type=Path, required=True)
    parser.add_argument("--score-db", type=Path, required=True)
    parser.add_argument("--genotype-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--region", default="chr20:30000000-30100000")
    parser.add_argument("--window", type=int, default=100000)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-variants", type=int, default=128)
    parser.add_argument("--plink2", default=shutil.which("plink2"))
    args = parser.parse_args()
    if not 8 <= args.samples <= 24 or not 1 <= args.window <= 1_000_000 or args.max_variants < 1:
        raise ValueError("Use 8-24 donors, a cis window of 1-1000000 bp, and a positive variant limit")
    if args.plink2 is None:
        raise ValueError("PLINK2 must be on PATH or supplied with --plink2")
    chrom, interval = args.region.split(":")
    start, end = map(int, interval.split("-"))
    if chrom != "chr20" or not 0 <= start < end or end - start > 1_000_000:
        raise ValueError("This spot check is limited to at most 1 Mb on chr20")
    output = args.outdir.resolve()
    protected = [args.count_db.resolve(), args.score_db.resolve(), args.genotype_dir.resolve()]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError("Output directory must be separate from both production databases and the genotype input directory; input and output directories must not contain one another")
    output.mkdir(parents=True, exist_ok=True)
    snapshots = {str(p / f): (p / f).stat().st_mtime_ns for p in protected[:2] for f in ("manifest.json", "index.npz")}
    count_source = regional_source(protected[0], chrom, start, end)
    score_source = regional_source(protected[1], chrom, start, end)
    assert count_source["samples"] == score_source["samples"]
    for source in (count_source, score_source):
        assert set(load_view_columns(str(source["path"]), KEY)["platform"]) == {"pacbio"}
        finite = source["values"][np.isfinite(source["values"])]
        assert ((finite >= 0) & (finite <= 1)).all()
    snapshot = output / "pacbio_count.regional_snapshot.mmdb"
    raw, provenance = regional_snapshot(count_source, snapshot)
    grouped_path = output / "pacbio_count.regional_denovo.mmdb"
    grouping = group_cohort(
        str(snapshot), str(grouped_path), "denovo", output_backend="npy", threads=1,
        denovo_min_correlation=0.8, denovo_max_gap_bp=200, denovo_min_shared_samples=3,
        min_group_cpgs=2, min_observed_cpgs=1, write_group_table=True,
        batch_rows=1024, batch_groups=1024,
    )
    grouped_source = regional_source(grouped_path, chrom, start, end)
    group_check = verify_grouped_means(grouped_path, raw, output / "pacbio_group_means_source_check.npz")
    vcf_paths = {p.name.split("_")[0]: p.resolve() for p in protected[2].glob("SMHT*_dshybrid_202503.01_aouimpute_1_whatshap_2.8.vcf.gz")}
    samples, selection = choose_samples(count_source, grouped_source, set(vcf_paths), args.samples)
    donors = [x.split("-")[0] for x in samples]
    print(json.dumps({"stage": "PacBio regional profile and sample selection", "donors": donors, "regional_count_rows": len(count_source["ids"]), "local_denovo_groups": len(grouped_source["ids"]), "selection": selection}), flush=True)
    (output / "selected_pacbio_source_samples.txt").write_text("\n".join(samples) + "\n")
    pd.DataFrame({"sample_id": samples, "genotype_id": donors}).to_csv(output / "sample_map.tsv", sep="\t", index=False)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    commands = []

    def run(command: list, log: Path) -> None:
        command = list(map(str, command))
        commands.append(command)
        with log.open("w") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment, cwd=output, check=True)

    import tensorqtl
    import torch
    from tensorqtl.genotypeio import load_genotypes

    sources = {"PACBIO_COUNT_single_cpg": count_source, "PACBIO_COUNT_local_denovo_grouped": grouped_source, "PACBIO_SCORE_single_cpg": score_source}
    frames, exports = {}, {}
    for name, source in sources.items():
        bed = output / f"{name}.methylation.bed.gz"
        run([sys.executable, "-m", "mdb.main", "export-tensorqtl", "-i", source["path"], "-o", bed, "--region", args.region, "--sample-file", output / "selected_pacbio_source_samples.txt", "--sample-map", output / "sample_map.tsv", "--missing", "drop", "--min-frac-present", "1", "--batch-rows", "1024", "--overwrite"], output / f"{name}.export.log")
        frame = pd.read_csv(bed, sep="\t")
        phenotype_df, positions = tensorqtl.read_phenotype_bed(str(bed))
        assert phenotype_df.columns.tolist() == donors and len(positions) == len(frame)
        exports[name] = source_verification(source, frame, samples, donors, output / f"{name}.source_checks.tsv")
        exports[name]["export_receipt"] = json.loads(Path(str(bed) + ".export.json").read_text())
        frames[name] = frame
        print(json.dumps({"stage": "PacBio export source values and IDs verified", "case": name, "phenotypes": len(frame)}), flush=True)
    lower = min(int(f["start"].min()) + 1 - args.window for f in frames.values())
    upper = max(int(f["end"].max()) + args.window for f in frames.values())
    real_records, real_metadata = matched_snps(vcf_paths, donors, chrom, lower, upper, args.max_variants)
    prefix = output / "REAL_PACBIO_matched_donor_genotypes"
    vcf = output / "REAL_PACBIO_matched_donor_genotypes.vcf"
    write_vcf(vcf, donors, real_records)
    run([args.plink2, "--vcf", vcf, "--make-bed", "--output-chr", "chrM", "--out", prefix, "--threads", "1", "--memory", "1024"], output / "REAL_PACBIO_matched_donor_genotypes.plink2.stdout.log")
    genotypes, variants = load_genotypes(str(prefix), select_samples=donors)
    expected_dosage = pd.DataFrame([r["dosages"] for r in real_records], index=[r["id"] for r in real_records], columns=donors)
    assert genotypes.columns.tolist() == donors and variants["chrom"].eq(chrom).all()
    np.testing.assert_array_equal(genotypes.loc[expected_dosage.index], expected_dosage)
    print(json.dumps({"stage": "real PacBio donor genotype intersection and ALT dosage verified", "variants": len(variants), "exact_called_snv_intersection": real_metadata["exact_called_snv_intersection"], "nonconstant_intersection": real_metadata["nonconstant_intersection"]}), flush=True)
    cli_results = {}
    for name, frame in frames.items():
        label = f"REAL_matched_genotypes.{name}"
        run([sys.executable, "-m", "tensorqtl", prefix, output / f"{name}.methylation.bed.gz", label, "--mode", "cis_nominal", "--window", str(args.window), "--output_dir", output], output / f"{label}.cli.stdout.log")
        parquet = output / f"{label}.cis_qtl_pairs.{chrom}.parquet"
        pairs = pd.read_parquet(parquet)
        result = verify_nominal(frame, pairs, genotypes, variants, args.window)
        diagnostics = output / f"{label}.upstream_zero_correlation_se.tsv"
        pd.DataFrame(result["upstream_zero_correlation_nan_standard_errors"], columns=["phenotype_id", "variant_id", "tensorqtl_pvalue", "tensorqtl_slope", "tensorqtl_standard_error", "scipy_standard_error", "scipy_correlation"]).to_csv(diagnostics, sep="\t", index=False)
        perfect_diagnostics = output / f"{label}.upstream_near_perfect_fit_precision.tsv"
        pd.DataFrame(result["upstream_near_perfect_fit_precision_edges"], columns=["phenotype_id", "variant_id", "scipy_correlation", "residual_sse_over_sst", "tensorqtl_slope", "scipy_slope", "tensorqtl_standard_error", "scipy_standard_error", "tensorqtl_pvalue", "scipy_pvalue", "absolute_log10_pvalue_difference", "zero_pvalue_prevents_log_comparison", "interpretation"]).to_csv(perfect_diagnostics, sep="\t", index=False)
        result.update({"phenotypes": len(frame), "genotype_variants": len(variants), "genotypes": "real donor VCF calls", "tensorqtl_loaded_alt_dosages_equal_original_vcf_gt_sums": True, "intentional_allele_flip": False, "nominal_parquet": str(parquet), "cli_stdout_log": str(output / f"{label}.cli.stdout.log"), "upstream_zero_correlation_diagnostic_tsv": str(diagnostics), "upstream_near_perfect_fit_diagnostic_tsv": str(perfect_diagnostics)})
        cli_results[name] = result
        print(json.dumps({"stage": "actual PacBio TensorQTL CLI validated", "case": name, "pairs": len(pairs), "upstream_null_SE_count": result["upstream_zero_correlation_nan_standard_error_count"], "upstream_near_perfect_precision_edge_count": result["upstream_near_perfect_fit_precision_edge_count"]}), flush=True)
    for filename, mtime in snapshots.items():
        assert Path(filename).stat().st_mtime_ns == mtime, f"Production metadata changed: {filename}"
    script = output / "run_commands.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n# Real PacBio phenotypes and real donor genotypes; technical checks only.\nexport PYTHONPATH=" + shlex.quote(environment["PYTHONPATH"]) + "\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n" + "\n".join(shlex.join(c) for c in commands) + "\n")
    script.chmod(0o755)
    receipt = {
        "status": "completed_with_upstream_numerical_caveats", "checks_passed": True,
        "platform": "PacBio", "region_zero_based_half_open": args.region,
        "tensorqtl_cis_window_bp": args.window,
        "window_rule": "inclusive one-based variant positions in [BED start+1-window, BED end+window]",
        "phenotype": "real PacBio 5mC combined haplotype/strand beta fractions; count-based CpGs, locally derived de novo grouped means, and original score-based CpGs",
        "source_sample_ids": samples, "genotype_donor_ids": donors,
        "missing_policy": "drop", "imputed_cells": 0, "sample_selection": selection,
        "regional_snapshot_provenance": provenance, "local_denovo_grouping_summary": grouping,
        "grouped_scope": "new local de novo grouping of the bounded PacBio count region across all source columns; no pre-existing production PacBio grouping used",
        "group_means_source_check": group_check, "exports": exports,
        "real_genotype_selection": real_metadata, "tensorqtl_cli_runs": cli_results,
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "versions": {x: importlib.metadata.version(x) for x in ["tensorqtl", "torch", "numpy", "pandas", "scipy", "pandas-plink", "pysam", "pyarrow", "zarr"]},
        "production_metadata_mtimes_unchanged": True, "replay_shell_script": str(script),
        "biological_inference": "none; small selected donor set, ascertained shared SNP intersection, and no analysis covariates",
    }
    (output / "validation_summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "checks_passed": True, "summary": str(output / "validation_summary.json"), "commands": str(script)}), flush=True)


if __name__ == "__main__":
    main()
