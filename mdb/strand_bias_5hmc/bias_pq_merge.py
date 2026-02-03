#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import polars as pl


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("strand_bias_5hmc_merge")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


log = setup_logger("INFO")


def find_dyads_files(root: str, name: str | None, min_cov: int) -> list[str]:
    """
    If name provided, search root/*/name.
    Else default pattern: dyads.minCov{min_cov}.pq
    """
    if name:
        pat = os.path.join(root, "*", name)
    else:
        pat = os.path.join(root, "*", f"dyads.minCov{min_cov}.pq")
    return sorted(glob.glob(pat))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge per-sample dyads parquet files into one parquet.")
    p.add_argument("--root", required=True, help="Root containing sample dirs with per-sample dyads parquet")
    p.add_argument("--min-cov", type=int, default=3, help="Used if --name is not provided")
    p.add_argument("--name", default=None, help="Exact dyads parquet filename inside each sample dir (overrides --min-cov)")
    p.add_argument("--out", required=True, help="Output merged parquet path (e.g., merged_dyads.pq)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    return p


def main() -> int:
    args = build_argparser().parse_args()
    global log
    log = setup_logger(args.log_level)

    files = find_dyads_files(args.root, args.name, args.min_cov)
    if not files:
        log.error("No parquet files found to merge.")
        return 2

    log.info(f"Found {len(files):,} parquet files")

    lfs = []
    for fp in files:
        sample_id = Path(fp).parent.name
        lfs.append(pl.scan_parquet(fp).with_columns(pl.lit(sample_id).alias("sample_id")))

    merged = pl.concat(lfs, how="vertical_relaxed")

    outp = Path(args.out).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Writing merged parquet: {outp}")
    merged.sink_parquet(str(outp))

    log.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
