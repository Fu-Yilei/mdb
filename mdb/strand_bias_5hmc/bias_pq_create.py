#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path
import re

import polars as pl


# -----------------------------
# Logging
# -----------------------------
def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("strand_bias_5hmc")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


log = setup_logger("INFO")


def safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


# -----------------------------
# Discovery
# -----------------------------
def discover_samples(root_dir: str) -> list[tuple[str, str]]:
    """
    Return [(sample_id, sample_dir), ...] for root_dir/* with hp1.bed.gz and hp2.bed.gz.
    """
    out: list[tuple[str, str]] = []
    for d in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if not os.path.isdir(d):
            continue
        hp1 = os.path.join(d, "hp1.bed.gz")
        hp2 = os.path.join(d, "hp2.bed.gz")
        if os.path.exists(hp1) and os.path.exists(hp2):
            out.append((os.path.basename(d.rstrip("/")), d))
    return out


# -----------------------------
# Your dyad computation (as provided)
# -----------------------------
USE_COLS = [0, 1, 3, 5, 9, 11]  # 0-based
NEW_NAMES = ["chrom", "start", "mod", "strand", "coverage", "mod_count"]


def scan_hp(path: str, hap: str) -> pl.LazyFrame:
    log.info(f"Scanning {path} ({hap})")
    t0 = time.time()

    lf = (
        pl.scan_csv(
            path,
            separator="\t",
            has_header=False,
            infer_schema_length=0,
            ignore_errors=True,
        )
        .select([pl.col(f"column_{i+1}") for i in USE_COLS])
        .rename({f"column_{i+1}": n for i, n in zip(USE_COLS, NEW_NAMES)})
        .with_columns([
            pl.col("start").cast(pl.Int64),
            pl.col("coverage").cast(pl.Int32),
            pl.col("mod_count").cast(pl.Int32),
            pl.lit(hap).alias("hap"),
        ])
    )

    log.info(f"  lazy scan registered in {time.time() - t0:.2f}s")
    return lf


def compute_dyads_lazy(hp1_bed_gz: str, hp2_bed_gz: str, eps: float = 0.5) -> pl.LazyFrame:
    lf = pl.concat(
        [scan_hp(hp1_bed_gz, "hp1"), scan_hp(hp2_bed_gz, "hp2")],
        how="vertical",
    )

    plus = (
        lf.filter(pl.col("strand") == "+")
          .select(["chrom", "start", "mod", "hap", "coverage", "mod_count"])
          .rename({"coverage": "cov_plus", "mod_count": "mod_plus"})
    )

    minus = (
        lf.filter(pl.col("strand") == "-")
          .with_columns((pl.col("start") - 1).alias("start"))
          .select(["chrom", "start", "mod", "hap", "coverage", "mod_count"])
          .rename({"coverage": "cov_minus", "mod_count": "mod_minus"})
    )

    dyads_lf = (
        plus.join(minus, on=["chrom", "start", "mod", "hap"], how="inner")
            .with_columns([
                (((pl.col("mod_plus") + eps) / (pl.col("mod_minus") + eps)).log(2))
                    .alias("strand_bias_log2"),
                pl.when(pl.col("mod") == "m").then(pl.lit("5mC"))
                 .when(pl.col("mod") == "h").then(pl.lit("5hmC"))
                 .otherwise(pl.col("mod"))
                 .alias("mod_name"),
            ])
            .select([
                "chrom", "start", "hap", "mod_name",
                "cov_plus", "cov_minus",
                "mod_plus", "mod_minus",
                "strand_bias_log2",
            ])
    )
    return dyads_lf


def compute_summary(dyads_f: pl.DataFrame, sample_id: str, hp1_path: str, hp2_path: str) -> pl.DataFrame:
    return (
        dyads_f
        .group_by(["hap", "mod_name"])
        .agg([
            pl.col("strand_bias_log2").count().alias("n"),
            pl.col("strand_bias_log2").mean().alias("mean"),
            pl.col("strand_bias_log2").median().alias("median"),
            pl.col("strand_bias_log2").std().alias("sd"),
        ])
        .with_columns([
            pl.lit(sample_id).alias("sample_id"),
            pl.lit(hp1_path).alias("hp1_path"),
            pl.lit(hp2_path).alias("hp2_path"),
        ])
        .select(["sample_id", "hap", "mod_name", "n", "mean", "median", "sd", "hp1_path", "hp2_path"])
    )


# -----------------------------
# One-sample runner
# -----------------------------
def run_one_sample(
    sample_dir: str,
    min_cov: int,
    eps: float,
    out_name: str,
    write_summary: bool,
    skip_existing: bool,
    engine: str,
) -> int:
    sample_dir_p = Path(sample_dir).resolve()
    sample_id = sample_dir_p.name
    hp1 = sample_dir_p / "hp1.bed.gz"
    hp2 = sample_dir_p / "hp2.bed.gz"

    if not hp1.exists() or not hp2.exists():
        log.error(f"Missing hp1/hp2 in {sample_dir_p}")
        return 2

    dyads_pq = sample_dir_p / out_name
    summ_pq = sample_dir_p / f"summary.{out_name}"

    if skip_existing:
        if dyads_pq.exists() and (not write_summary or summ_pq.exists()):
            log.info(f"Skip existing outputs for {sample_id}")
            return 0

    log.info(f"Compute dyads: sample_id={sample_id}")

    dyads_lf = (
        compute_dyads_lazy(str(hp1), str(hp2), eps=eps)
        .filter((pl.col("cov_plus") >= min_cov) & (pl.col("cov_minus") >= min_cov))
    )

    # Write dyads: prefer sink (no collect) for scalability
    log.info(f"Writing dyads parquet: {dyads_pq}")
    dyads_lf.sink_parquet(str(dyads_pq))

    if write_summary:
        log.info(f"Collecting for summary (engine={engine})")
        dyads_df = dyads_lf.collect(engine=engine)
        summ = compute_summary(dyads_df, sample_id, str(hp1), str(hp2))
        log.info(f"Writing summary parquet: {summ_pq}")
        summ.write_parquet(str(summ_pq))

    log.info("Done")
    return 0


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create per-sample strand-bias dyads parquet (.pq) inside each hp1/hp2 directory."
    )
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--sample-dir", help="Process one sample directory containing hp1.bed.gz and hp2.bed.gz.")
    g.add_argument("--task-id", type=int, help="0-based index into discovered samples (use with --root).")

    p.add_argument(
        "--root",
        default="/stornext/snfs130/smaht/luis/analysis/donors/methylation_results/strand_5hmC",
        help="Root directory for discovery (used with --task-id / --list-samples).",
    )
    p.add_argument("--list-samples", action="store_true", help="List discovered samples (index, sample_id, path) and exit.")
    p.add_argument("--min-cov", type=int, default=3)
    p.add_argument("--eps", type=float, default=0.5)

    # output naming: goes INSIDE sample-dir
    p.add_argument(
        "--out-name",
        default="dyads.minCov3.pq",
        help="Output parquet filename to write inside each sample directory.",
    )
    p.add_argument("--write-summary", action="store_true", help="Also write summary parquet inside each sample directory.")
    p.add_argument("--skip-existing", action="store_true", help="Skip if output parquet already exists.")
    p.add_argument("--engine", default="streaming", choices=["streaming", "in_memory"], help="Polars collect engine (for summary).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    return p


def main() -> int:
    args = build_argparser().parse_args()
    global log
    log = setup_logger(args.log_level)

    # list mode
    if args.list_samples:
        samples = discover_samples(args.root)
        for i, (sid, sdir) in enumerate(samples):
            print(f"{i}\t{sid}\t{sdir}")
        return 0

    # choose sample_dir
    sample_dir = args.sample_dir
    if sample_dir is None:
        task_id = args.task_id
        if task_id is None:
            env_tid = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env_tid is not None:
                try:
                    task_id = int(env_tid)
                except ValueError:
                    log.error(f"Invalid SLURM_ARRAY_TASK_ID={env_tid!r}")
                    return 2

        if task_id is None:
            log.error("Provide --sample-dir OR --task-id (or set SLURM_ARRAY_TASK_ID).")
            return 2

        samples = discover_samples(args.root)
        if task_id < 0 or task_id >= len(samples):
            log.error(f"task-id {task_id} out of range [0, {len(samples)-1}] for root={args.root}")
            return 2

        sid, sample_dir = samples[task_id]
        log.info(f"Selected task-id={task_id}: sample_id={sid}")

    assert sample_dir is not None

    # if user left default out-name dyads.minCov3.pq, but min-cov differs, adjust automatically
    out_name = args.out_name
    if args.out_name == "dyads.minCov3.pq" and args.min_cov != 3:
        out_name = f"dyads.minCov{args.min_cov}.pq"

    return run_one_sample(
        sample_dir=sample_dir,
        min_cov=args.min_cov,
        eps=args.eps,
        out_name=out_name,
        write_summary=args.write_summary,
        skip_existing=args.skip_existing,
        engine=args.engine,
    )


if __name__ == "__main__":
    raise SystemExit(main())
