from __future__ import annotations

import time
import logging
import polars as pl

log = logging.getLogger("strand_bias")

# Column indices in modkit pileup BED (0-based)
USE_COLS = [0, 1, 3, 5, 9, 11]
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
    """
    Returns a LazyFrame of CpG dyads with strand_bias_log2 and mod_name.
    """
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
