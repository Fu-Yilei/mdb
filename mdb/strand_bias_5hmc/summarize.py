from __future__ import annotations
import polars as pl


def summarize_dyads(dyads: pl.DataFrame, sample_id: str, hp1_path: str, hp2_path: str) -> pl.DataFrame:
    return (
        dyads
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
