"""Opt-in real-BAM MethBat -> mdb validation; inputs remain read-only.

The three database columns are two imports of the same MethBat pileup and one
legacy pb-CpG-tools result. They are technical comparisons, not independent
biological samples. All subprocess commands, logs, and checksums are retained.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from mdb.grouping import load_group_index  # noqa: E402
from mdb.schema import TrackKey, VALUE_MISSING  # noqa: E402
from mdb.storage import list_tracks, load_view_columns, load_view_reader, read_track  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_rows(path):
    with gzip.open(path, "rt") as handle:
        yield from csv.DictReader((line for line in handle if not line.startswith("##")), delimiter="\t")


def sample_values(bundle, key):
    reader = read_track(str(bundle), key)
    try:
        return np.concatenate([np.asarray(reader.chrom_values(c, allow_missing=True)).copy() for c in reader.chroms])
    finally:
        reader.close()


def check_export(path, raw, positions, sample_ids):
    frame = pd.read_csv(path, sep="\t")
    eligible = np.all(raw != VALUE_MISSING, axis=1) & np.any(raw != raw[:, :1], axis=1)
    assert frame.columns[4:].tolist() == sample_ids
    np.testing.assert_array_equal(frame["start"], positions[eligible])
    np.testing.assert_allclose(frame.iloc[:, 4:], raw[eligible].astype(float) / 10000, atol=1e-7)
    return int(len(frame))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--legacy-prefix", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--pileup-prefix", type=Path, help="Reuse an already completed MethBat pileup")
    parser.add_argument("--methbat", default="methbat")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.outdir.resolve()
    legacy_beds = [Path(f"{args.legacy_prefix}.{tag}.bed.gz").resolve() for tag in ("combined", "hap1", "hap2")]
    protected = [args.bam.resolve(), Path(str(args.bam) + ".bai").resolve(), args.reference.resolve(), *legacy_beds]
    if args.pileup_prefix:
        protected.extend(Path(f"{args.pileup_prefix}.{assay}.bed.gz").resolve() for assay in ("5mC", "5hmC", "6mA"))
    for path in protected:
        if not path.is_file():
            raise FileNotFoundError(path)
        if output == path or output in path.parents:
            raise ValueError("Validation output must not contain source inputs")
    original_hashes = {str(p): sha256(p) for p in protected}
    output.mkdir(parents=True, exist_ok=False)
    (output / "input_sha256.before.json").write_text(json.dumps(original_hashes, indent=2) + "\n")
    commands = []
    env = dict(os.environ, PYTHONPATH=str(REPOSITORY), POLARS_MAX_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")

    def run(command, name, expected_failure=False):
        command = list(map(str, command))
        commands.append({"stage": name, "command": command, "expected_failure": expected_failure})
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(command, cwd=REPOSITORY, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / f"{name}.log").write_text(result.stdout)
        if expected_failure:
            assert result.returncode != 0 and "missing columns ['begin']" in result.stdout
        elif result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); see {output / (name + '.log')}")
        print(f"PASS {name}", flush=True)
        return result.stdout

    def cli(name, *argv, **kwargs):
        return run([sys.executable, "-m", "mdb.main", *argv], name, **kwargs)

    version = run([args.methbat, "--version"], "methbat_version").strip()
    prefix = args.pileup_prefix.resolve() if args.pileup_prefix else output / "pileup"
    if not args.pileup_prefix:
        run([args.methbat, "pileup", "--threads", args.threads, "--input-bam", args.bam,
             "--output-prefix", prefix], "methbat_pileup")
    index = output / "reference_index.npz"
    cli("index", "index", "-r", args.reference, "-o", index)
    with np.load(index, allow_pickle=True) as data:
        chroms, positions, offsets = list(data["chroms"]), data["pos0"], data["chrom_offsets"]
    assert chroms == ["chr20"], "This fixture runner expects the repository's chr20 test reference"
    lookup = {int(pos): row for row, pos in enumerate(positions)}
    pileup = Path(f"{prefix}.5mC.bed.gz")
    cli("old_parser_rejection", "create", "-p", "pacbio", "-n", index, "-b", pileup,
        "-o", output / "old_parser.smdb", expected_failure=True)

    sample_ids = ["methbat_prefix", "legacy_pbcpg", "methbat_direct"]
    bundles = [output / f"{s}.smdb" for s in sample_ids]
    for sample, bundle, source, flags in [
        (sample_ids[0], bundles[0], prefix, ["--input-format", "methbat"]),
        (sample_ids[1], bundles[1], args.legacy_prefix, []),
        (sample_ids[2], bundles[2], pileup, ["--input-format", "methbat"]),
    ]:
        cli(f"create_{sample}", "create", "-p", "pacbio", *flags, "-n", index,
            "-b", source, "-o", bundle, "--sample-id", sample, "-c", "5")

    expected = {hap: np.full(len(positions), VALUE_MISSING, dtype=np.uint16) for hap in ("combined", "hp1", "hp2")}
    coverage = {hap: values.copy() for hap, values in expected.items()}
    counts = {hap: {"source_rows": 0, "below_coverage": 0, "off_index": 0, "imported": 0} for hap in expected}
    for fields in source_rows(pileup):
        hap = {"Total": "combined", "hap1": "hp1", "hap2": "hp2"}[fields["type"]]
        counts[hap]["source_rows"] += 1
        cov = int(fields["cov"])
        if cov < 5:
            counts[hap]["below_coverage"] += 1
            continue
        position = int(fields["start"])
        if fields["#chrom"] != "chr20" or position not in lookup:
            counts[hap]["off_index"] += 1
            continue
        row_id = lookup[position]
        assert expected[hap][row_id] == VALUE_MISSING
        expected[hap][row_id] = round(float(fields["mod_score"]) * 100)
        coverage[hap][row_id] = cov
        counts[hap]["imported"] += 1

    for bundle in (bundles[0], bundles[2]):
        assert set(list_tracks(str(bundle))) == {TrackKey("5mC", h, "combined") for h in expected}
        for hap in expected:
            key = TrackKey("5mC", hap, "combined")
            track = read_track(str(bundle), key)
            try:
                np.testing.assert_array_equal(track.chrom_values("chr20"), expected[hap])
                np.testing.assert_array_equal(track.chrom_coverage("chr20"), coverage[hap])
            finally:
                track.close()
            first = int(np.flatnonzero(expected[hap] != VALUE_MISSING)[0])
            result = json.loads(cli(f"query_{bundle.stem}_{hap}", "query", "-i", bundle,
                                    "--haplotype", hap, "--locus", f"chr20:{positions[first]}"))
            assert result["value_percent"] == int(expected[hap][first]) / 100
            assert result["coverage"] == int(coverage[hap][first])
    print("PASS all MethBat values, coverage, and missing positions against independent BED reader", flush=True)

    matrices = {hap: np.column_stack([sample_values(bundle, TrackKey("5mC", hap, "combined")) for bundle in bundles]) for hap in expected}
    checks = {}
    for backend in ("npy", "zarr"):
        cohort = output / f"cohort_{backend}.mmdb"
        cli(f"merge_{backend}", "merge", "-i", *bundles[:2], "-o", cohort, "--cohort-backend", backend)
        cli(f"append_{backend}", "append", "-c", cohort, "-i", bundles[2])
        checks[backend] = {}
        for hap, matrix in matrices.items():
            key = TrackKey("5mC", hap, "combined")
            reader, _, _ = load_view_reader(str(cohort), key)
            try:
                observed = reader.read_rows(slice(0, len(positions)))
                beta = matrix.astype(float) / 10000
                beta[matrix == VALUE_MISSING] = np.nan
                np.testing.assert_allclose(observed, beta, atol=1e-7)
            finally:
                reader.close()
            assert load_view_columns(str(cohort), key)["sample_id"] == sample_ids
            bed = output / f"{backend}_{hap}.bed.gz"
            cli(f"export_{backend}_{hap}", "export-tensorqtl", "-i", cohort, "-o", bed, "--haplotype", hap)
            checks[backend][f"export_{hap}_rows"] = check_export(bed, matrix, positions, sample_ids)
        cli(f"cohort_query_{backend}", "query", "-i", cohort, "--sample-id", sample_ids[0],
            "--region", "chr20:1000000-1001000")
        grouped = output / f"groups_{backend}.mmdb"
        cli(f"group_{backend}", "group", "-i", cohort, "-o", grouped, "--grouping", "denovo")
        groups = load_group_index(str(grouped))
        assert groups.n_groups > 0
        for hap, matrix in matrices.items():
            means = np.full((groups.n_groups, len(bundles)), VALUE_MISSING, dtype=np.uint16)
            for row_id, (lo, hi) in enumerate(zip(groups.source_row_start, groups.source_row_end, strict=True)):
                block = matrix[lo:hi].astype(float)
                block[block == VALUE_MISSING] = np.nan
                valid = np.isfinite(block).any(axis=0)
                means[row_id, valid] = np.rint(np.nanmean(block[:, valid], axis=0)).astype(np.uint16)
            reader, _, _ = load_view_reader(str(grouped), TrackKey("5mC", hap, "combined"))
            try:
                expected_means = means.astype(float) / 10000
                expected_means[means == VALUE_MISSING] = np.nan
                np.testing.assert_allclose(reader.read_rows(slice(0, groups.n_groups)), expected_means, atol=1e-7)
            finally:
                reader.close()
            bed = output / f"groups_{backend}_{hap}.bed.gz"
            cli(f"export_groups_{backend}_{hap}", "export-tensorqtl", "-i", grouped, "-o", bed, "--haplotype", hap)
            checks[backend][f"group_export_{hap}_rows"] = check_export(bed, means, groups.start, sample_ids)
        checks[backend]["groups"] = groups.n_groups

    cohort = output / "cohort_zarr.mmdb"
    cli("stats", "stats", "-i", cohort, "-o", output / "stats")
    cli("pca", "pca", "-i", cohort, "-o", output / "pca", "--n_pcs", "2", "-pn", "2",
        "--pairplot-mode", "none", "--pairplot_diag_kind", "hist", "--frac_cpgs", "0.1")
    cli("viz", "viz", "-i", cohort, "-o", output / "viz", "--assay", "5mC", "--bin-length", "100000")
    cli("plot", "plot", "-i", cohort, "-o", output / "plot", "--assay", "5mC", "--region", "chr20:1000000-1010000")
    assert len(pd.read_csv(output / "pca/embedding.tsv", sep="\t")) == 3
    stats = pd.read_csv(output / "stats/sample_stats.tsv", sep="\t")
    assert len(stats) == 9
    final_hashes = {str(p): sha256(p) for p in protected}
    assert original_hashes == final_hashes
    receipt = {
        "status": "passed", "methbat_version": version, "reference_cpgs": len(positions),
        "methbat_tracks": counts, "database_checks": checks,
        "all_imported_values_and_coverage_match_source": True,
        "source_sha256_unchanged": final_hashes,
        "technical_columns_not_independent_samples": sample_ids,
        "real_data_assay_tested": "5mC; this BAM has no 5hmC or 6mA pileup rows",
        "additional_cli_checks": ["stats", "pca", "viz", "plot"],
    }
    (output / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
