# pacbio_bed_parsing.py

import os
import time
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

from scipy import stats
from tqdm.auto import tqdm

# ---------- PacBio (pb-cpg-tools cpg_scores)
PB_COL_CHROM = "chrom"
PB_COL_START = "begin"
PB_COL_COV = "cov"
PB_COL_DMS = "discretized_mod_score"  # standard cpg_scores schema, 0..100
PB_COL_MOD_SCORE = "mod_score"  # pileup-mode=count schema, 0..100


def discover_pacbio_beds(path: str) -> Dict[str, str]:
    """
    Return dict of {tag: filepath} for PacBio pb-cpg-tools cpg_scores inputs.

    PacBio naming (prefix-based):
      <prefix>.cpg_scores.combined.bed.gz
      <prefix>.cpg_scores.hap1.bed.gz
      <prefix>.cpg_scores.hap2.bed.gz

    Input handling:
      - If `path` is a file:
          * If it's a .bed or .bed.gz, treat as "combined".
          * Otherwise treat it as a PREFIX (file prefix) and look for the three expected files.
      - If `path` is a directory:
          * Require `prefix` to be the directory basename? (NO) -> instead:
            - If directory contains exactly one "*.cpg_scores.combined.bed.gz", use its prefix.
            - Otherwise error (ambiguous), to avoid guessing.
      - If `path` is neither file nor directory: treat as PREFIX string and resolve expected files.

    Tags: "combined", "hap1", "hap2"
    """
    def _from_prefix(prefix: str) -> Dict[str, str]:
        cand = {
            "combined": f"{prefix}.combined.bed.gz",
            "hp1": f"{prefix}.hap1.bed.gz",
            "hp2": f"{prefix}.hap2.bed.gz",
        }
        found = {k: v for k, v in cand.items() if os.path.exists(v)}

        # optional: accept uncompressed .bed
        for k, v in list(cand.items()):
            if k not in found:
                v2 = v[:-3]  # drop ".gz" -> ".bed"
                if os.path.exists(v2):
                    found[k] = v2

        if not found:
            raise FileNotFoundError(
                "No PacBio cpg_scores beds found for prefix.\n"
                f"Prefix tried: {prefix}\n"
                "Expected any of:\n"
                f"  {prefix}.cpg_scores.combined.bed(.gz)\n"
                f"  {prefix}.cpg_scores.hap1.bed(.gz)\n"
                f"  {prefix}.cpg_scores.hap2.bed(.gz)"
            )
        return found

    # ---- case 1: exact path is a file ----
    if os.path.isfile(path):
        # If user points directly to a bed(.gz), just take it as combined
        if path.endswith(".bed") or path.endswith(".bed.gz"):
            return {"combined": path}

        # Otherwise treat as a PREFIX-like string (rare but allowed)
        return _from_prefix(path)

    # ---- case 2: path is a directory ----
    if os.path.isdir(path):
        # Find combined bed.gz candidates to infer prefix
        cands = [
            os.path.join(path, fn)
            for fn in os.listdir(path)
            if fn.endswith(".cpg_scores.combined.bed.gz") or fn.endswith(".cpg_scores.combined.bed")
        ]
        if len(cands) == 0:
            raise FileNotFoundError(
                f"No '*.cpg_scores.combined.bed(.gz)' found in directory: {path}"
            )
        if len(cands) > 1:
            raise FileNotFoundError(
                f"Ambiguous PacBio directory (multiple combined beds found): {path}\n"
                "Provide a prefix (full path without the trailing '.cpg_scores.*') "
                "or point directly to a specific combined bed(.gz)."
            )

        combined_path = cands[0]
        # Strip suffix to get prefix
        if combined_path.endswith(".cpg_scores.combined.bed.gz"):
            prefix = combined_path[: -len(".cpg_scores.combined.bed.gz")]
        else:
            prefix = combined_path[: -len(".cpg_scores.combined.bed")]

        return _from_prefix(prefix)

    # ---- case 3: treat as PREFIX string (most common for your use case) ----
    return _from_prefix(path)

def map_positions_to_rows(
    chrom: str,
    starts_np: np.ndarray,
    *,
    pos0: np.ndarray,
    chrom_slices: Dict[str, Tuple[int, int]],
) -> np.ndarray:
    """Map 0-based CpG start positions on a given chrom to global row indices; -1 if not found."""
    if chrom not in chrom_slices:
        return np.full(starts_np.shape[0], -1, dtype=np.int64)

    s, e = chrom_slices[chrom]
    chrom_pos = pos0[s:e]  # sorted increasing within chrom
    j = np.searchsorted(chrom_pos, starts_np, side="left")
    ok = (j < chrom_pos.shape[0]) & (chrom_pos[j] == starts_np)

    out = np.full(starts_np.shape[0], -1, dtype=np.int64)
    out[ok] = (s + j[ok]).astype(np.int64)
    return out

def fill_one_sample_pacbio(
    bed_path: str,
    col_idx: int,
    M_5mC: np.ndarray,
    *,
    pos0: np.ndarray,
    chrom_slices: Dict[str, Tuple[int, int]],
    allowed_chroms: List[str],
    min_cov: int,
    ) -> Dict[str, int]:
    """
    PacBio pb-cpg-tools TSV:
    - comment lines start with ## and a header line starts with '#chrom ...'
    - standard cpg_scores uses discretized_mod_score (0..100)
    - pileup-mode=count uses mod_score (0..100)
    - 5hmC not available -> M_5hmC column stays NaN.

    This reader:
    - skips comment lines starting with "##"
    - treats the '#chrom ...' line as header (we strip the leading '#')
    """
    t0 = time.time()

    # We need to (1) skip lines starting with ##, and (2) parse the header line "#chrom ..."
    # Polars supports 'comment_prefix' for a single prefix; we use it to skip "##" lines.
    # Then we read with has_header=True; the first non-## line is the "#chrom ..." header.
    lf = pl.scan_csv(
        bed_path,
        separator="\t",
        has_header=True,
        comment_prefix="##",
    )

    # Header will create a column named "#chrom" (with leading '#').
    # Normalize it to "chrom" so the rest is consistent.
    # Also enforce the minimal needed columns exist.
    schema_names = lf.collect_schema().names()
    lf = lf.rename({"#chrom": PB_COL_CHROM}) if "#chrom" in schema_names else lf

    schema_names = lf.collect_schema().names()
    score_col = PB_COL_DMS if PB_COL_DMS in schema_names else PB_COL_MOD_SCORE if PB_COL_MOD_SCORE in schema_names else None

    # Keep required columns only
    needed = [PB_COL_CHROM, PB_COL_START, PB_COL_COV]
    missing = [c for c in needed if c not in schema_names]
    if missing or score_col is None:
        expected_score_cols = [PB_COL_DMS, PB_COL_MOD_SCORE]
        raise ValueError(
            f"PacBio file missing columns {missing} or a score column from {expected_score_cols}. "
            f"Found columns: {schema_names}"
        )

    lf = (
        lf.select(needed + [score_col])
          .filter(pl.col(PB_COL_CHROM).is_in(allowed_chroms))
    )
    if min_cov > 0:
        lf = lf.filter(pl.col(PB_COL_COV) >= min_cov)

    df = lf.collect(engine="streaming")

    stats = {
        "rows_after_filters": int(df.height),
        "mapped_m": 0,  # treat as 5mC
        "mapped_h": 0,  # always 0 for pacbio
    }

    if df.height == 0:
        return stats

    # Both supported PacBio schemas store methylation percent on a 0..100 scale.
    df = df.with_columns((pl.col(score_col) / 100.0).cast(pl.Float32).alias("val"))

    mapped = 0
    for chrom, sub in df.group_by(PB_COL_CHROM, maintain_order=True):
        chrom = chrom[0]
        starts = sub[PB_COL_START].to_numpy().astype(np.int32)
        rows = map_positions_to_rows(chrom, starts, pos0=pos0, chrom_slices=chrom_slices)
        ok = rows != -1
        if ok.any():
            vals = sub["val"].to_numpy().astype(np.float32)
            M_5mC[rows[ok], col_idx] = vals[ok]
            mapped += int(ok.sum())

    stats["mapped_m"] = mapped


    return stats

def pacbio_bed_parsing(
    input_path: str,
    chrom_slices: Dict[str, Tuple[int, int]],
    cpg_pos0: np.ndarray,
    allowed_chroms: List[str],
    min_cov: int,
):
    bed_map = discover_pacbio_beds(input_path)
    print(f"Discovering PacBio beds in: {bed_map}")
    m_5mC = np.full((cpg_pos0.shape[0], len(bed_map)), np.nan, dtype=np.float32)
    for index, (file_name, bed_path) in tqdm(
        enumerate(bed_map.items()),
        desc="Processing PacBio beds",
        unit="sample",
        total=len(bed_map)
    ):
        last_stats = fill_one_sample_pacbio(
            bed_path,
            index,
            m_5mC,
            pos0=cpg_pos0,
            chrom_slices=chrom_slices,
            allowed_chroms=allowed_chroms,
            min_cov=min_cov,
        )
        last_stats[file_name] = last_stats
    return last_stats, m_5mC, bed_map
