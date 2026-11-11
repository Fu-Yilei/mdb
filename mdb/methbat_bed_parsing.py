"""Import MethBat pileups into CpG-indexed sample tracks."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import polars as pl

from mdb.schema import COVERAGE_MISSING, SampleTrack, TrackKey, encode_percent


HAPLOTYPES = {"Total": "combined", "hap1": "hp1", "hap2": "hp2"}
ASSAY_CODES = {"5mC": "m", "5hmC": "h"}
DTYPES = {
    "start": pl.Int64, "end": pl.Int64, "name": pl.String,
    "strand": pl.String, "mod_score": pl.Float64, "type": pl.String,
    "cov": pl.Int64,
}


def discover_methbat_beds(path: str) -> tuple[list[Path], list[Path]]:
    """Accept one BED or a MethBat output prefix; never guess a sample directory."""
    source = Path(path)
    if source.is_file():
        return [source.resolve()], []
    if source.is_dir():
        raise ValueError("For MethBat, supply one .bed(.gz) file or an output prefix, not a directory")
    found: dict[str, Path] = {}
    for assay in (*ASSAY_CODES, "6mA"):
        matches = [Path(f"{path}.{assay}.bed{suffix}") for suffix in (".gz", "")]
        matches = [p.resolve() for p in matches if p.is_file()]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous MethBat prefix: both compressed and plain {assay} files exist")
        if matches:
            found[assay] = matches[0]
    beds = [found[assay] for assay in ASSAY_CODES if assay in found]
    if not beds:
        raise ValueError(
            f"No MethBat 5mC/5hmC BED files found for {path}. "
            "6mA is not supported by the CpG index."
        )
    return beds, [found["6mA"]] if "6mA" in found else []


def _read_bed(path: Path) -> tuple[str, dict[str, str], pl.DataFrame]:
    metadata = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.startswith("##"):
                break
            key, sep, value = line[2:].strip().partition("=")
            if sep:
                metadata[key] = value
    assay = metadata.get("base_modification")
    if assay not in ASSAY_CODES:
        raise ValueError(
            f"{path}: MethBat ##base_modification must be 5mC or 5hmC; "
            f"found {assay!r}. 6mA is not supported by the CpG index."
        )
    lf = pl.scan_csv(
        path, separator="\t", has_header=True, comment_prefix="##",
        schema_overrides={"#chrom": pl.String, "chrom": pl.String, **DTYPES},
    )
    if "#chrom" in lf.collect_schema().names():
        lf = lf.rename({"#chrom": "chrom"})
    needed = ["chrom", *DTYPES]
    missing = set(needed) - set(lf.collect_schema().names())
    if missing:
        raise ValueError(f"{path}: MethBat file missing columns {sorted(missing)}")
    df = lf.select(needed).collect(engine="streaming")
    if any(df.null_count().row(0)):
        raise ValueError(f"{path}: missing values in required MethBat columns")
    if df.is_empty():
        return assay, metadata, df
    if set(df["name"].unique()) != {ASSAY_CODES[assay]}:
        raise ValueError(f"{path}: modification codes do not match {assay}")
    if set(df["type"].unique()) - HAPLOTYPES.keys():
        raise ValueError(f"{path}: unknown MethBat type; expected Total, hap1, or hap2")
    allowed_strands = {"."} if assay == "5mC" else {"+", "-"}
    if set(df["strand"].unique()) - allowed_strands:
        raise ValueError(f"{path}: invalid strand for {assay}; expected {sorted(allowed_strands)}")
    invalid = df.filter(
        (pl.col("start") < 0) | (pl.col("end") != pl.col("start") + 1)
        | ~pl.col("mod_score").is_finite()
        | (pl.col("mod_score") < 0) | (pl.col("mod_score") > 100)
        | (pl.col("cov") <= 0) | (pl.col("cov") >= int(COVERAGE_MISSING))
    )
    if invalid.height:
        raise ValueError(
            f"{path}: invalid MethBat coordinates, mod_score, or coverage; require "
            "0-based single-base intervals, finite percentages in [0,100], and cov in [1,65534]"
        )
    return assay, metadata, df


def methbat_bed_parsing(
    input_path: str,
    chrom_slices: dict[str, tuple[int, int]],
    cpg_pos0: np.ndarray,
    min_cov: int,
) -> tuple[dict[TrackKey, SampleTrack], dict]:
    """Keep Total/haplotypes separate, retain depth, and normalize reverse CpG 5hmC.

    Only reference-index CpGs are imported. Non-CpG and excluded-chromosome
    rows are counted in the receipt, not coerced onto a nearby CpG.
    """
    if min_cov < 0:
        raise ValueError("Minimum coverage must be nonnegative")
    paths, ignored = discover_methbat_beds(input_path)
    report = {
        "input_format": "methbat", "platform": "pacbio",
        "min_coverage": min_cov, "files": [],
        "ignored_6mA_files": [str(p) for p in ignored],
        "coordinates": "0-based reference CpG C; reverse 5hmC start is shifted by -1",
        "coverage": "actual MethBat cov, retained in sample tracks",
        "extra_count_and_probability_columns_retained": False,
    }
    tracks = {}
    for path in paths:
        assay, metadata, df = _read_bed(path)
        file_report = {
            "path": str(path), "assay": assay, "metadata": metadata,
            "rows_input": df.height, "tracks": {},
        }
        report["files"].append(file_report)
        for (track_type, strand), group in df.group_by(["type", "strand"], maintain_order=True):
            key = TrackKey(assay, HAPLOTYPES[track_type], {".": "combined", "+": "plus", "-": "minus"}[strand])
            if key in tracks:
                raise ValueError(f"Duplicate MethBat track {key.name()} across input files")
            retained = group.filter(pl.col("cov") >= min_cov)
            row_parts, value_parts, coverage_parts = [], [], []
            for (chrom,), sub in retained.group_by("chrom", maintain_order=True):
                if chrom not in chrom_slices:
                    continue
                lo, hi = chrom_slices[chrom]
                ref_pos = cpg_pos0[lo:hi]
                if ref_pos.size == 0:
                    continue
                starts = sub["start"].to_numpy() - (1 if strand == "-" else 0)
                positions = np.searchsorted(ref_pos, starts)
                in_range = (starts >= 0) & (positions < ref_pos.size)
                matches = np.zeros(len(starts), dtype=bool)
                matches[in_range] = ref_pos[positions[in_range]] == starts[in_range]
                row_parts.append((positions[matches] + lo).astype(np.uint32))
                value_parts.append(encode_percent(sub["mod_score"].to_numpy()[matches]))
                coverage_parts.append(sub["cov"].to_numpy()[matches].astype(np.uint16))
            row_ids = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.uint32)
            values = np.concatenate(value_parts) if value_parts else np.empty(0, dtype=np.uint16)
            coverage = np.concatenate(coverage_parts) if coverage_parts else np.empty(0, dtype=np.uint16)
            order = np.argsort(row_ids, kind="stable")
            row_ids, values, coverage = row_ids[order], values[order], coverage[order]
            if row_ids.size > 1 and np.any(row_ids[1:] == row_ids[:-1]):
                raise ValueError(f"{path}: duplicate CpG rows within MethBat track {key.name()}")
            tracks[key] = SampleTrack(
                key=key, row_ids=row_ids, values=values, coverage=coverage,
                platform="pacbio", input_tag=key.haplotype, source_path=str(path), min_coverage=min_cov,
            )
            file_report["tracks"][key.name()] = {
                "rows_input": group.height,
                "rows_below_min_coverage": group.height - retained.height,
                "rows_not_in_index": retained.height - len(row_ids),
                "rows_imported": len(row_ids),
            }
    if not any(track.row_ids.size for track in tracks.values()):
        raise ValueError("No MethBat CpG rows match the reference index and minimum coverage")
    return tracks, report
