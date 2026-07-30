# ont_bed_parsing.py

import os
import time
from flask import logging
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

from tqdm.auto import tqdm


# ---------- ONT (modkit pileup bed, has_header=False => column_1..)
ONT_COL_CHROM  = "column_1"
ONT_COL_START  = "column_2"
ONT_COL_STRAND = "column_6"   # optional; only used if present
ONT_COL_CODE   = "column_4"   # "m" or "h"
ONT_COL_COV    = "column_10"  # Nvalid_cov
ONT_COL_PCT    = "column_11"  # percent modified


def discover_ont_beds(path: str) -> Dict[str, str]:
    """
    Return dict of {tag: filepath} for ONT inputs.

    If path is a file: tag="combined" and use it.
    If path is a folder: look for hp1/hp2/combined bed(.gz) with exact names.
    Accept .bed as fallback in addition to .bed.gz.

    Tags: "combined", "hp1", "hp2"
    """
    if os.path.isfile(path):
        return {"combined": path}

    if os.path.isdir(path):
        cand = {
            "combined": os.path.join(path, "combined.bed.gz"),
            "hp1": os.path.join(path, "hp1.bed.gz"),
            "hp2": os.path.join(path, "hp2.bed.gz"),
        }
        found = {k: v for k, v in cand.items() if os.path.exists(v)}

        # optionally also accept .bed (uncompressed)
        for k in list(cand.keys()):
            if k not in found:
                v2 = os.path.join(path, f"{k}.bed") if k != "combined" else os.path.join(path, "combined.bed")
                if os.path.exists(v2):
                    found[k] = v2
        if not found:
            raise FileNotFoundError(
                f"No ONT beds found in {path}. Expected any of: hp1.bed(.gz), hp2.bed(.gz), combined.bed(.gz)"
            )
        return found

    # Prefix mode: {path}.combined.bed.gz / .hap1.bed.gz / .hap2.bed.gz
    prefix_cand = {
        "combined": f"{path}.combined.bed.gz",
        "hp1":      f"{path}.hap1.bed.gz",
        "hp2":      f"{path}.hap2.bed.gz",
    }
    found = {k: v for k, v in prefix_cand.items() if os.path.exists(v)}
    if found:
        return found

    raise FileNotFoundError(f"ONT input is neither file, directory, nor valid prefix: {path}")


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

def detect_modkit_separate_strands(bed_path: str) -> bool:
    """
    Detect whether a modkit pileup BED reports separate strands.

    Heuristic:
      - The file must have column_6 available (ONT_COL_STRAND).
      - The observed values in column_6 must include BOTH '+' and '-'.

    Returns True if separate strands are present, else False.
    """
    lf = pl.scan_csv(
        bed_path,
        separator="\t",
        has_header=False,
        infer_schema_length=200,
    )

    names = lf.collect_schema().names()
    if ONT_COL_STRAND not in names:
        return False

    # Pull a small set of unique strand symbols (streaming-friendly)
    strand_vals = (
        lf.select(pl.col(ONT_COL_STRAND).cast(pl.Utf8))
          .unique()
          .limit(16)
          .collect(engine="streaming")[ONT_COL_STRAND]
          .to_list()
    )
    strand_set = {s for s in strand_vals if s is not None}
    return ("+" in strand_set) and ("-" in strand_set)


# --- minimally extend your existing fill_one_sample_ont (add a flag + outputs) ---

def fill_one_sample_ont(
    bed_path: str,
    col_idx: int,
    m_5mC: np.ndarray,
    m_5hmC: np.ndarray,
    *,
    pos0: np.ndarray,
    chrom_slices: Dict[str, Tuple[int, int]],
    allowed_chroms: List[str],
    min_cov: int,
    separate_strands: bool = False,
    m_5mC_plus: Optional[np.ndarray] = None,
    m_5mC_minus: Optional[np.ndarray] = None,
    m_5hmC_plus: Optional[np.ndarray] = None,
    m_5hmC_minus: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    """ONT modkit pileup BED: has_header=False, 'm' and 'h' rows."""
    t0 = time.time()

    select_cols = [ONT_COL_CHROM, ONT_COL_START, ONT_COL_CODE, ONT_COL_PCT, ONT_COL_COV]
    if separate_strands:
        select_cols.append(ONT_COL_STRAND)

    lf = (
        pl.scan_csv(bed_path, separator="\t", has_header=False)
          .select(select_cols)
          .filter(pl.col(ONT_COL_CHROM).is_in(allowed_chroms))
    )
    if min_cov > 0:
        lf = lf.filter(pl.col(ONT_COL_COV) >= min_cov)

    # If separate_strands=True, keep only +/- to avoid '.', '*', etc.
    if separate_strands:
        lf = lf.filter(pl.col(ONT_COL_STRAND).is_in(["+", "-"]))

    df = lf.collect(engine="streaming")

    stats = {
        "rows_after_filters": int(df.height),
        "rows_m": 0,
        "rows_h": 0,
        "mapped_m": 0,
        "mapped_h": 0,
        "mapped_m_plus": 0,
        "mapped_m_minus": 0,
        "mapped_h_plus": 0,
        "mapped_h_minus": 0,
    }

    if df.height == 0:
        return stats

    df = df.with_columns((pl.col(ONT_COL_PCT) / 100.0).cast(pl.Float32).alias("val"))

    df_m = df.filter(pl.col(ONT_COL_CODE) == "m").select(
        [ONT_COL_CHROM, ONT_COL_START, "val"] + ([ONT_COL_STRAND] if separate_strands else [])
    )
    df_h = df.filter(pl.col(ONT_COL_CODE) == "h").select(
        [ONT_COL_CHROM, ONT_COL_START, "val"] + ([ONT_COL_STRAND] if separate_strands else [])
    )
    stats["rows_m"] = int(df_m.height)
    stats["rows_h"] = int(df_h.height)

    def _apply_unstranded(df_sub: pl.DataFrame, mat: np.ndarray, tag: str) -> int:
        if df_sub.height == 0:
            stats[f"mapped_{tag}"] = 0
            return 0
        mapped = 0
        for chrom, sub in df_sub.group_by(ONT_COL_CHROM, maintain_order=True):
            chrom = chrom[0]
            starts = sub[ONT_COL_START].to_numpy().astype(np.int32)
            rows = map_positions_to_rows(chrom, starts, pos0=pos0, chrom_slices=chrom_slices)
            ok = rows != -1
            if ok.any():
                vals = sub["val"].to_numpy().astype(np.float32)
                mat[rows[ok], col_idx] = vals[ok]
                mapped += int(ok.sum())
        stats[f"mapped_{tag}"] = mapped
        return mapped

    def _apply_stranded(df_sub: pl.DataFrame, mat_plus: np.ndarray, mat_minus: np.ndarray, tag: str) -> None:
        if df_sub.height == 0:
            stats[f"mapped_{tag}_plus"] = 0
            stats[f"mapped_{tag}_minus"] = 0
            return

        # -----------------
        # PLUS strand
        # -----------------
        mapped_plus = 0
        df_plus = df_sub.filter(pl.col(ONT_COL_STRAND) == "+").select([ONT_COL_CHROM, ONT_COL_START, "val"])
        for chrom, sub in df_plus.group_by(ONT_COL_CHROM, maintain_order=True):
            chrom = chrom[0]
            starts = sub[ONT_COL_START].to_numpy().astype(np.int32)  # canonical already
            rows = map_positions_to_rows(chrom, starts, pos0=pos0, chrom_slices=chrom_slices)
            ok = rows != -1
            if ok.any():
                vals = sub["val"].to_numpy().astype(np.float32)
                mat_plus[rows[ok], col_idx] = vals[ok]
                mapped_plus += int(ok.sum())

        # -----------------
        # MINUS strand (shift coordinate)
        # -----------------
        mapped_minus = 0
        df_minus = df_sub.filter(pl.col(ONT_COL_STRAND) == "-").select([ONT_COL_CHROM, ONT_COL_START, "val"])
        for chrom, sub in df_minus.group_by(ONT_COL_CHROM, maintain_order=True):
            chrom = chrom[0]
            starts_raw = sub[ONT_COL_START].to_numpy().astype(np.int32)

            # Key fix: minus-strand cytosine is often reported at CpG+1 (plus-strand G position)
            starts = starts_raw - 1

            # guard against negative after shift
            starts = np.maximum(starts, 0)

            rows = map_positions_to_rows(chrom, starts, pos0=pos0, chrom_slices=chrom_slices)
            ok = rows != -1
            if ok.any():
                vals = sub["val"].to_numpy().astype(np.float32)
                mat_minus[rows[ok], col_idx] = vals[ok]
                mapped_minus += int(ok.sum())

        stats[f"mapped_{tag}_plus"] = mapped_plus
        stats[f"mapped_{tag}_minus"] = mapped_minus


    if not separate_strands:
        _apply_unstranded(df_m, m_5mC, "m")
        _apply_unstranded(df_h, m_5hmC, "h")
    else:
        if any(x is None for x in [m_5mC_plus, m_5mC_minus, m_5hmC_plus, m_5hmC_minus]):
            raise ValueError("separate_strands=True requires *_plus and *_minus matrices to be provided.")
        _apply_stranded(df_m, m_5mC_plus, m_5mC_minus, "m")
        _apply_stranded(df_h, m_5hmC_plus, m_5hmC_minus, "h")

    return stats


# --- update ont_bed_parsing to auto-detect and return 4 matrices when stranded ---

def ont_bed_parsing(
        input_path: str,
        chrom_slices: Dict[str, Tuple[int, int]],
        cpg_pos0: np.ndarray,
        allowed_chroms: List[str],
        min_cov: int,
):
    bed_map = discover_ont_beds(input_path)
    print(f"Discovered ONT beds: {bed_map}")

    n_samples = len(bed_map)
    n_cpg = cpg_pos0.shape[0]

    # Detect using the first discovered bed
    first_bed = next(iter(bed_map.values()))
    separate_strands = detect_modkit_separate_strands(first_bed)

    if not separate_strands:
        print("Detected modkit pileup BED without separate strands.")
        m_5mC = np.full((n_cpg, n_samples), np.nan, dtype=np.float32)
        m_5hmC = np.full((n_cpg, n_samples), np.nan, dtype=np.float32)

        last_stats = None
        for index, (file_name, bed_path) in tqdm(
            enumerate(bed_map.items()),
            total=n_samples,
            desc="Processing ONT beds",
            unit="sample",
        ):
            last_stats = fill_one_sample_ont(
                bed_path=bed_path,
                m_5mC=m_5mC,
                m_5hmC=m_5hmC,
                pos0=cpg_pos0,
                chrom_slices=chrom_slices,
                allowed_chroms=allowed_chroms,
                min_cov=min_cov,
                col_idx=index,
                separate_strands=False,
            )

        return last_stats, (m_5mC, m_5hmC), bed_map

    # stranded case: allocate 4 matrices
    print("Detected modkit pileup BED with separate strands.")
    m_5mC_plus  = np.full((n_cpg, n_samples), np.nan, dtype=np.float32)
    m_5mC_minus = np.full((n_cpg, n_samples), np.nan, dtype=np.float32)
    m_5hmC_plus  = np.full((n_cpg, n_samples), np.nan, dtype=np.float32)
    m_5hmC_minus = np.full((n_cpg, n_samples), np.nan, dtype=np.float32)
    last_stats = None
    for index, (file_name, bed_path) in tqdm(
        enumerate(bed_map.items()),
        total=n_samples,
        desc="Processing ONT beds (strand-separated)",
        unit="sample",
    ):
        last_stats = fill_one_sample_ont(
            bed_path=bed_path,
            m_5mC=np.empty((0, 0), dtype=np.float32),   # unused in stranded mode
            m_5hmC=np.empty((0, 0), dtype=np.float32),  # unused in stranded mode
            pos0=cpg_pos0,
            chrom_slices=chrom_slices,
            allowed_chroms=allowed_chroms,
            min_cov=min_cov,
            col_idx=index,
            separate_strands=True,
            m_5mC_plus=m_5mC_plus,
            m_5mC_minus=m_5mC_minus,
            m_5hmC_plus=m_5hmC_plus,
            m_5hmC_minus=m_5hmC_minus,
        )

    return last_stats, (m_5mC_plus, m_5mC_minus, m_5hmC_plus, m_5hmC_minus), bed_map


