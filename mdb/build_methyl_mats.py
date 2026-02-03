#!/usr/bin/env python3
"""
Build CpG-by-sample matrices for 5mC and 5hmC ONLY (no M_sum),
using a pre-built CpG index (NPZ) and methylation bed-like files.

Supports two input formats via --platform:
  - ont   (default): modkit pileup BED (no header; 18 columns; 'm' and 'h' rows)
  - pacbio: pb-cpg-tools cpg_scores TSV (has header lines starting with ## and a # header row;
            columns include: chrom, begin, end, ... discretized_mod_score)

Outputs:
  <out_prefix>.5mC.npy           float32 (fraction 0..1 unless --keep-percent)
  <out_prefix>.5hmC.npy          float32 (ONT only; for PacBio will remain NaN)

Notes:
- Full-genome RAM allocation can be huge; for 400 samples this likely won't fit in 50 GB.
  Use chromosome-sharded or memmap approach if needed.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Tuple, List, Optional, Callable

import numpy as np
import polars as pl
from tqdm import tqdm

# ---------- ONT (modkit pileup bed, has_header=False => column_1..)
ONT_COL_CHROM = "column_1"
ONT_COL_START = "column_2"
ONT_COL_CODE  = "column_4"    # "m" or "h"
ONT_COL_PCT   = "column_11"   # percent modified
ONT_COL_COV   = "column_10"   # Nvalid_cov

# ---------- PacBio (pb-cpg-tools cpg_scores, tab-separated with comments + header)
# Columns (example):
# chrom  begin  end  mod_score  type  cov  est_mod_count  est_unmod_count  discretized_mod_score
PB_COL_CHROM = "chrom"
PB_COL_START = "begin"                    # 0-based start
PB_COL_COV   = "cov"
PB_COL_DMS   = "discretized_mod_score"    # percent 0..100


def setup_logging(log_path: Optional[str], verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("build_5mc_5hmc")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def load_index(index_npz: str, logger: logging.Logger):
    idx = np.load(index_npz, allow_pickle=True)
    chroms = idx["chroms"]                 # object array
    chrom_offsets = idx["chrom_offsets"]   # int64
    pos0 = idx["pos0"]                     # int32

    n_cpg = int(pos0.shape[0])
    chrom_slices: Dict[str, Tuple[int, int]] = {}
    for i, c in enumerate(chroms):
        c = str(c)
        s = int(chrom_offsets[i])
        e = int(chrom_offsets[i + 1]) if i + 1 < len(chrom_offsets) else n_cpg
        chrom_slices[c] = (s, e)

    chrom_to_idx = {str(c): i for i, c in enumerate(chroms)}
    allowed_chroms = list(chrom_to_idx.keys())

    logger.info(f"Loaded index: {index_npz}")
    logger.info(f"Index CpGs: {n_cpg:,}")
    logger.info(f"Index chroms ({len(chroms)}): {allowed_chroms}")

    return chroms, chrom_offsets, pos0, chrom_slices, allowed_chroms


def read_beds_list(beds_list_path: str) -> Tuple[List[str], List[str]]:
    """
    Returns (sample_names, file_paths)
    Accepts either:
      - path
      - sample<TAB>path
    """
    names: List[str] = []
    paths: List[str] = []
    with open(beds_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                p = parts[0]
                names.append(os.path.basename(p))
                paths.append(p)
            else:
                names.append(parts[0])
                paths.append(parts[1])
    return names, paths


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


# -------------------- Platform-specific fill functions --------------------

def fill_one_sample_ont(
    bed_path: str,
    col_idx: int,
    M_5mC: np.ndarray,
    M_5hmC: np.ndarray,
    *,
    pos0: np.ndarray,
    chrom_slices: Dict[str, Tuple[int, int]],
    allowed_chroms: List[str],
    min_cov: int,
    keep_percent: bool,
    logger: logging.Logger,
) -> Dict[str, int]:
    """ONT modkit pileup BED: has_header=False, 'm' and 'h' rows."""
    t0 = time.time()

    lf = (
        pl.scan_csv(bed_path, separator="\t", has_header=False)
          .select([ONT_COL_CHROM, ONT_COL_START, ONT_COL_CODE, ONT_COL_PCT, ONT_COL_COV])
          .filter(pl.col(ONT_COL_CHROM).is_in(allowed_chroms))
    )
    if min_cov > 0:
        lf = lf.filter(pl.col(ONT_COL_COV) >= min_cov)

    df = lf.collect(engine="streaming")

    stats = {
        "rows_after_filters": int(df.height),
        "rows_m": 0,
        "rows_h": 0,
        "mapped_m": 0,
        "mapped_h": 0,
    }

    if df.height == 0:
        logger.warning(f"[{col_idx}] {os.path.basename(bed_path)}: no rows after filters")
        return stats

    # percent -> value
    if keep_percent:
        df = df.with_columns(pl.col(ONT_COL_PCT).cast(pl.Float32).alias("val"))
    else:
        df = df.with_columns((pl.col(ONT_COL_PCT) / 100.0).cast(pl.Float32).alias("val"))

    df_m = df.filter(pl.col(ONT_COL_CODE) == "m").select([ONT_COL_CHROM, ONT_COL_START, "val"])
    df_h = df.filter(pl.col(ONT_COL_CODE) == "h").select([ONT_COL_CHROM, ONT_COL_START, "val"])
    stats["rows_m"] = int(df_m.height)
    stats["rows_h"] = int(df_h.height)

    def _apply(df_sub: pl.DataFrame, mat: np.ndarray, tag: str):
        if df_sub.height == 0:
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

    mapped_m = _apply(df_m, M_5mC, "m")
    mapped_h = _apply(df_h, M_5hmC, "h")

    logger.info(
        f"[{col_idx}] {os.path.basename(bed_path)} "
        f"rows={stats['rows_after_filters']:,} "
        f"m={stats['rows_m']:,}(mapped {mapped_m:,}) "
        f"h={stats['rows_h']:,}(mapped {mapped_h:,}) "
        f"min_cov={min_cov}"
    )
    logger.debug(f"[{col_idx}] done in {time.time()-t0:.1f}s")
    return stats


def fill_one_sample_pacbio(
    bed_path: str,
    col_idx: int,
    M_5mC: np.ndarray,
    M_5hmC: np.ndarray,
    *,
    pos0: np.ndarray,
    chrom_slices: Dict[str, Tuple[int, int]],
    allowed_chroms: List[str],
    min_cov: int,
    keep_percent: bool,
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    PacBio pb-cpg-tools 'cpg_scores' TSV:
    - comment lines start with ## and a header line starts with '#chrom ...'
    - uses discretized_mod_score (0..100) as methylation percent for CpG.
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
    lf = lf.rename({"#chrom": PB_COL_CHROM}) if "#chrom" in lf.columns else lf

    # Keep required columns only
    needed = [PB_COL_CHROM, PB_COL_START, PB_COL_DMS, PB_COL_COV]
    missing = [c for c in needed if c not in lf.columns]
    if missing:
        raise ValueError(
            f"PacBio file missing columns {missing}. "
            f"Found columns: {lf.columns}"
        )

    lf = (
        lf.select(needed)
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
        logger.warning(f"[{col_idx}] {os.path.basename(bed_path)}: no rows after filters")
        return stats

    # discretized_mod_score is percent 0..100
    if keep_percent:
        df = df.with_columns(pl.col(PB_COL_DMS).cast(pl.Float32).alias("val"))
    else:
        df = df.with_columns((pl.col(PB_COL_DMS) / 100.0).cast(pl.Float32).alias("val"))

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

    logger.info(
        f"[{col_idx}] {os.path.basename(bed_path)} "
        f"rows={stats['rows_after_filters']:,} "
        f"pacbio(mapped {mapped:,}) "
        f"min_cov={min_cov} (5hmC=NaN)"
    )
    logger.debug(f"[{col_idx}] done in {time.time()-t0:.1f}s")
    return stats


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="CpG index NPZ")
    ap.add_argument("--beds-list", required=True, help="file list (path or sample<TAB>path)")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for .npy files")
    ap.add_argument("--min-cov", type=int, default=0, help="Minimum coverage filter (Nvalid_cov or cov)")
    ap.add_argument("--keep-percent", action="store_true", help="Store 0..100 instead of 0..1")
    ap.add_argument(
        "--platform",
        choices=["ont", "pacbio"],
        default="ont",
        help="Input format: ont(modkit) or pacbio(pb-cpg-tools cpg_scores)"
    )
    ap.add_argument("--log", default=None, help="Log file path")
    ap.add_argument("--verbose", action="store_true", help="Verbose console logs")
    args = ap.parse_args()

    logger = setup_logging(args.log, verbose=args.verbose)
    logger.info(f"polars {pl.__version__}")
    logger.info(f"platform={args.platform}")

    chroms, chrom_offsets, pos0, chrom_slices, allowed_chroms = load_index(args.index, logger)
    sample_names, file_paths = read_beds_list(args.beds_list)
    if not file_paths:
        logger.error("No files found in --beds-list")
        sys.exit(1)

    n_samples = len(file_paths)
    n_cpg = int(pos0.shape[0])

    logger.info(f"Samples: {n_samples}")
    logger.info(f"Allocating matrices: CpGs={n_cpg:,} x samples={n_samples}")
    logger.info(f"Value mode: {'percent(0..100)' if args.keep_percent else 'fraction(0..1)'}; min_cov={args.min_cov}")

    # Allocate RAM matrices
    M_5mC = np.full((n_cpg, n_samples), np.nan, dtype=np.float32)
    M_5hmC = np.full((n_cpg, n_samples), np.nan, dtype=np.float32)

    # Choose fill function
    if args.platform == "ont":
        fill_fn: Callable[..., Dict[str, int]] = fill_one_sample_ont
    else:
        fill_fn = fill_one_sample_pacbio

    # Fill
    for j, path in enumerate(tqdm(file_paths, desc="Processing files", unit="file")):
        if not os.path.exists(path):
            logger.warning(f"[{j}] Missing file: {path}")
            continue

        fill_fn(
            path,
            j,
            M_5mC,
            M_5hmC,
            pos0=pos0,
            chrom_slices=chrom_slices,
            allowed_chroms=allowed_chroms,
            min_cov=args.min_cov,
            keep_percent=args.keep_percent,
            logger=logger,
        )

    # Save outputs (stop here; no M_sum)
    out_prefix = args.out_prefix
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    # Always save 5mC
    np.save(out_prefix + ".5mC.npy", M_5mC)
    logger.info(f"Saved: {out_prefix}.5mC.npy")

    # Save 5hmC only for ONT
    if args.platform == "ont":
        np.save(out_prefix + ".5hmC.npy", M_5hmC)
        logger.info(f"Saved: {out_prefix}.5hmC.npy")

    # QC summary
    filled_5mc = int(np.isfinite(M_5mC).sum())
    logger.info(f"Filled entries: 5mC={filled_5mc:,}")

    if args.platform == "ont":
        filled_5hmc = int(np.isfinite(M_5hmC).sum())
        logger.info(f"Filled entries: 5hmC={filled_5hmc:,}")
    else:
        logger.info("PacBio mode: only 5mC matrix written (no 5hmC signal).")

    logger.info("Done.")


if __name__ == "__main__":
    main()

