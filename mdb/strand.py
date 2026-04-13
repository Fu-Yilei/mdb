"""
mdb strand — genome-wide strand-specific DNA methylation hotspot detection.

Reads paired plus/minus tracks from a strand-aware cohort store (.mmdb) and:
  1. Detects loci where one strand consistently carries more methylation than
     the other (hotspot_score = |mean_diff| × sqrt(n_paired_samples)).
  2. Clusters adjacent same-direction CpGs into hotspot loci.
  3. Optionally stratifies hotspot calling per metadata category (--group-by).
  4. Emits ranked hotspot TSVs, BED files, per-sample bias metrics, and HTML
     reports.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm

from mdb.plotting import (
    apply_style_to_figure,
    build_color_styles,
    make_dropdown_scatter,
    maybe_merge_metadata,
    plotly_png_ok,
    resolve_plot_styles,
    write_plotly_image_safe,
)
from mdb.schema import TrackKey
from mdb.storage import (
    available_views,
    detect_store_kind,
    load_cohort_index,
    load_view_columns,
    load_view_reader,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class StrandConfig:
    assay: str
    haplotype: str
    min_paired_frac: float
    min_mean_total: float
    cluster_gap_bp: int
    min_cluster_cpgs: int
    top_n_hotspots: int
    batch_rows: int
    workers: int


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(outdir: str, verbose: bool) -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("mdb_strand")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(outdir, "strand.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ── Store / track helpers ─────────────────────────────────────────────────────

def _find_strand_pair(input_path: str, assay: str, haplotype: str) -> tuple[TrackKey, TrackKey]:
    views = set(available_views(input_path))
    plus_key = TrackKey(assay=assay, haplotype=haplotype, strand="plus")
    minus_key = TrackKey(assay=assay, haplotype=haplotype, strand="minus")
    if plus_key not in views or minus_key not in views:
        available = sorted(v.name() for v in views)
        raise ValueError(
            f"Strand pair not found for assay={assay!r} haplotype={haplotype!r}. "
            f"Available views: {available}"
        )
    return plus_key, minus_key


# ── Core scoring kernel (also called inside workers) ─────────────────────────

def _score_block(
    plus_block: np.ndarray,
    minus_block: np.ndarray,
    row_offset: int,
    min_paired: int,
    min_mean_total: float,
) -> pd.DataFrame:
    """Score a (rows × samples) block for strand bias.

    hotspot_score = |mean_diff| × sqrt(n_paired_samples)
    Only rows with ≥ min_paired paired observations and mean_total ≥ min_mean_total
    are returned.
    """
    paired = np.isfinite(plus_block) & np.isfinite(minus_block)
    paired_n = paired.sum(axis=1, dtype=np.int32)
    eligible = paired_n >= min_paired

    if not np.any(eligible):
        return pd.DataFrame()

    p = np.where(paired, plus_block, 0.0).astype(np.float64)
    m = np.where(paired, minus_block, 0.0).astype(np.float64)
    pn = np.maximum(paired_n, 1).astype(np.float64)

    plus_sum = p.sum(axis=1)
    minus_sum = m.sum(axis=1)
    diff_sum = (p - m).sum(axis=1)
    mean_total = (plus_sum + minus_sum) / (2.0 * pn)

    eligible &= mean_total >= min_mean_total
    if not np.any(eligible):
        return pd.DataFrame()

    idx = np.flatnonzero(eligible)
    pn_e = pn[eligible]
    mean_diff = diff_sum[eligible] / pn_e
    return pd.DataFrame({
        "row_id": (idx + row_offset).astype(np.int64),
        "paired_samples": paired_n[eligible].astype(np.int64),
        "mean_plus": plus_sum[eligible] / pn_e,
        "mean_minus": minus_sum[eligible] / pn_e,
        "mean_diff": mean_diff,
        "mean_abs_diff": np.abs(mean_diff),
        "mean_total": mean_total[eligible],
        "hotspot_score": np.abs(mean_diff) * np.sqrt(pn_e),
        "direction": np.where(mean_diff >= 0, "plus>minus", "minus>plus"),
    })


# ── Parallel chunk worker (top-level for pickle) ──────────────────────────────

def _chunk_worker(task: dict) -> dict:
    """
    Top-level worker: reads one row-range chunk, scores global + per-category.

    Returns
    -------
    dict with keys:
      "sample_paired_obs"      : np.ndarray (n_samples,) int64  — global
      "sample_paired_plus_sum" : np.ndarray (n_samples,) float64 — global
      "sample_paired_minus_sum": np.ndarray (n_samples,) float64 — global
      "global_candidates"      : pd.DataFrame
      "cat_candidates"         : dict[str, pd.DataFrame]
    """
    from mdb.schema import TrackKey
    from mdb.storage import load_view_reader

    input_path: str = task["input_path"]
    plus_key = TrackKey.from_name(task["plus_key"])
    minus_key = TrackKey.from_name(task["minus_key"])
    start_row: int = task["start_row"]
    end_row: int = task["end_row"]
    batch_rows: int = task["batch_rows"]
    min_mean_total: float = task["min_mean_total"]
    min_paired_frac: float = task["min_paired_frac"]
    group_indices: dict[str, list[int]] = task["group_indices"]
    top_keep: int = task["top_keep"]

    reader_plus, _, _ = load_view_reader(input_path, plus_key)
    reader_minus, _, _ = load_view_reader(input_path, minus_key)

    try:
        n_samples: int = reader_plus.shape[1]
        global_min_paired = max(1, int(math.ceil(n_samples * min_paired_frac)))
        cat_min_paired = {
            cat: max(1, int(math.ceil(len(idx) * min_paired_frac)))
            for cat, idx in group_indices.items()
        }
        cat_idx_arrays = {cat: np.array(idx, dtype=np.int64) for cat, idx in group_indices.items()}

        # Accumulators for per-sample global metrics
        sample_paired_obs = np.zeros(n_samples, dtype=np.int64)
        sample_paired_plus_sum = np.zeros(n_samples, dtype=np.float64)
        sample_paired_minus_sum = np.zeros(n_samples, dtype=np.float64)

        global_parts: list[pd.DataFrame] = []
        cat_parts: dict[str, list[pd.DataFrame]] = {cat: [] for cat in group_indices}

        row = start_row
        while row < end_row:
            chunk_end = min(row + batch_rows, end_row)
            plus_block = reader_plus.get_block(slice(row, chunk_end))   # float32, nan=missing
            minus_block = reader_minus.get_block(slice(row, chunk_end))

            # Accumulate per-sample global stats
            paired_mask = np.isfinite(plus_block) & np.isfinite(minus_block)
            sample_paired_obs += paired_mask.sum(axis=0, dtype=np.int64)
            p_clean = np.where(paired_mask, plus_block, 0.0).astype(np.float64)
            m_clean = np.where(paired_mask, minus_block, 0.0).astype(np.float64)
            sample_paired_plus_sum += p_clean.sum(axis=0)
            sample_paired_minus_sum += m_clean.sum(axis=0)

            # Global hotspot candidates
            scored = _score_block(plus_block, minus_block, row, global_min_paired, min_mean_total)
            if not scored.empty:
                global_parts.append(scored)

            # Per-category hotspot candidates
            for cat, idx_arr in cat_idx_arrays.items():
                scored_cat = _score_block(
                    plus_block[:, idx_arr],
                    minus_block[:, idx_arr],
                    row,
                    cat_min_paired[cat],
                    min_mean_total,
                )
                if not scored_cat.empty:
                    cat_parts[cat].append(scored_cat)

            row = chunk_end

        def _prune(parts: list[pd.DataFrame]) -> pd.DataFrame:
            if not parts:
                return pd.DataFrame()
            combined = pd.concat(parts, ignore_index=True)
            if len(combined) > top_keep:
                combined = combined.nlargest(top_keep, ["hotspot_score", "mean_abs_diff"])
            return combined.reset_index(drop=True)

        return {
            "sample_paired_obs": sample_paired_obs,
            "sample_paired_plus_sum": sample_paired_plus_sum,
            "sample_paired_minus_sum": sample_paired_minus_sum,
            "global_candidates": _prune(global_parts),
            "cat_candidates": {cat: _prune(parts) for cat, parts in cat_parts.items()},
        }
    finally:
        reader_plus.close()
        reader_minus.close()


# ── Scan orchestration ────────────────────────────────────────────────────────

def _run_scan(
    input_path: str,
    plus_key: TrackKey,
    minus_key: TrackKey,
    total_rows: int,
    group_indices: dict[str, list[int]],
    config: StrandConfig,
    logger: logging.Logger,
) -> dict:
    """Parallel full-genome scan. Returns merged candidates + per-sample stats."""
    n_workers = max(1, config.workers)
    # When requiring multi-CpG clusters, keep far more candidates so that
    # neighboring CpGs co-survive the per-chunk pruning and can form clusters.
    cluster_multiplier = max(5, config.min_cluster_cpgs * 20)
    top_keep = config.top_n_hotspots * cluster_multiplier

    # Divide genome into chunks — at least batch_rows each
    chunk_size = max(config.batch_rows, math.ceil(total_rows / n_workers))
    chunks: list[tuple[int, int]] = []
    row = 0
    while row < total_rows:
        chunks.append((row, min(row + chunk_size, total_rows)))
        row += chunk_size

    tasks = [
        {
            "input_path": input_path,
            "plus_key": plus_key.name(),
            "minus_key": minus_key.name(),
            "start_row": s,
            "end_row": e,
            "batch_rows": config.batch_rows,
            "min_mean_total": config.min_mean_total,
            "min_paired_frac": config.min_paired_frac,
            "group_indices": group_indices,
            "top_keep": top_keep,
        }
        for s, e in chunks
    ]

    logger.info(
        f"Scanning {total_rows:,} CpGs in {len(chunks)} chunk(s), "
        f"{n_workers} worker(s), {config.batch_rows:,} rows/batch"
    )

    results: list[dict] = []
    if n_workers == 1:
        for task in tqdm(tasks, desc="chunks", leave=False):
            results.append(_chunk_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_chunk_worker, t): i for i, t in enumerate(tasks)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="chunks", leave=False):
                results.append(fut.result())

    # Accumulate per-sample global stats
    reader_tmp, _, _ = load_view_reader(input_path, plus_key)
    n_smp = reader_tmp.shape[1]
    reader_tmp.close()
    del reader_tmp

    paired_obs = np.zeros(n_smp, dtype=np.int64)
    paired_plus = np.zeros(n_smp, dtype=np.float64)
    paired_minus = np.zeros(n_smp, dtype=np.float64)
    for r in results:
        paired_obs += r["sample_paired_obs"]
        paired_plus += r["sample_paired_plus_sum"]
        paired_minus += r["sample_paired_minus_sum"]

    def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full_like(a, np.nan, dtype=np.float64)
        valid = b > 0
        out[valid] = a[valid] / b[valid]
        return out

    sample_stats = {
        "paired_obs": paired_obs,
        "mean_plus": _safe_div(paired_plus, paired_obs),
        "mean_minus": _safe_div(paired_minus, paired_obs),
        "mean_diff": _safe_div(paired_plus - paired_minus, paired_obs),
        "balance_index": _safe_div(paired_plus - paired_minus, paired_plus + paired_minus),
    }

    # Merge hotspot candidates
    def _merge_candidates(frames: list[pd.DataFrame]) -> pd.DataFrame:
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return (
            combined.nlargest(top_keep, ["hotspot_score", "mean_abs_diff"])
            .drop_duplicates("row_id")
            .reset_index(drop=True)
        )

    global_cands = _merge_candidates([r["global_candidates"] for r in results])
    cat_cands = {
        cat: _merge_candidates([r["cat_candidates"].get(cat, pd.DataFrame()) for r in results])
        for cat in group_indices
    }

    return {
        "sample_stats": sample_stats,
        "global_candidates": global_cands,
        "cat_candidates": cat_cands,
    }


# ── Coordinate annotation ─────────────────────────────────────────────────────

def _annotate_rows(
    df: pd.DataFrame,
    chroms: list[str],
    chrom_offsets: np.ndarray,
    pos0: np.ndarray,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    boundaries = chrom_offsets[1:].astype(np.int64)
    row_ids = df["row_id"].to_numpy(dtype=np.int64)
    chrom_idx = np.searchsorted(boundaries, row_ids, side="right")
    out = df.copy()
    out["chrom"] = np.array(chroms, dtype=object)[chrom_idx]
    out["start"] = pos0[row_ids].astype(np.int64)
    out["end"] = out["start"] + 1
    out["coord"] = out["chrom"].astype(str) + ":" + (out["start"] + 1).map(lambda v: f"{v:,}")
    return out


# ── Clustering ────────────────────────────────────────────────────────────────

def _cluster_hotspots(candidates: pd.DataFrame, config: StrandConfig) -> pd.DataFrame:
    """Cluster CpG candidates into hotspot loci.

    New cluster boundary when: different chrom, gap > cluster_gap_bp, or
    different strand direction.
    """
    if candidates.empty:
        return pd.DataFrame()

    sorted_df = candidates.sort_values(["chrom", "start"]).reset_index(drop=True)

    chroms_arr = sorted_df["chrom"].to_numpy()
    starts_arr = sorted_df["start"].to_numpy(dtype=np.int64)
    dirs_arr = sorted_df["direction"].to_numpy()
    new_cluster = np.ones(len(sorted_df), dtype=bool)
    if len(sorted_df) > 1:
        new_cluster[1:] = (
            (chroms_arr[1:] != chroms_arr[:-1])
            | ((starts_arr[1:] - starts_arr[:-1]) > config.cluster_gap_bp)
            | (dirs_arr[1:] != dirs_arr[:-1])
        )
    sorted_df["_cid"] = np.cumsum(new_cluster) - 1
    # Vectorized cluster aggregation — fast even for tens of thousands of clusters
    g = sorted_df.groupby("_cid", sort=True)
    agg = g.agg(
        chrom=("chrom", "first"),
        start=("start", "min"),
        end=("start", "max"),
        n_cpgs=("row_id", "count"),
        max_hotspot_score=("hotspot_score", "max"),
        max_abs_mean_diff=("mean_abs_diff", "max"),
        direction=("direction", "first"),
    ).reset_index(drop=True)
    agg["end"] = agg["end"] + 1

    # Representative row: the one with highest hotspot_score per cluster
    rep_idx = sorted_df.groupby("_cid")["hotspot_score"].idxmax()
    rep_rows = sorted_df.loc[rep_idx.values, ["_cid", "row_id", "coord", "direction",
                                               "mean_diff", "mean_total", "paired_samples"]]
    rep_rows = rep_rows.set_index("_cid")
    agg["representative_row_id"] = rep_rows["row_id"].values
    agg["representative_coord"] = rep_rows["coord"].values
    agg["direction"] = rep_rows["direction"].values
    agg["rep_mean_diff"] = rep_rows["mean_diff"].values
    agg["rep_mean_total"] = rep_rows["mean_total"].values
    agg["rep_paired_samples"] = rep_rows["paired_samples"].values

    result = agg.drop(columns=["_cid"], errors="ignore")
    if config.min_cluster_cpgs > 1:
        result = result[result["n_cpgs"] >= config.min_cluster_cpgs]
    result = (
        result
        .nlargest(config.top_n_hotspots, ["max_hotspot_score", "max_abs_mean_diff", "n_cpgs"])
        .reset_index(drop=True)
    )
    result.insert(0, "hotspot_rank", np.arange(1, len(result) + 1, dtype=np.int64))
    result.insert(
        1,
        "hotspot_label",
        result.apply(
            lambda r: f"H{int(r.hotspot_rank)}: {r.chrom}:{int(r.start)+1:,}-{int(r.end):,}",
            axis=1,
        ),
    )
    return result


# ── Profile extraction ────────────────────────────────────────────────────────

def _extract_profiles(
    cluster_df: pd.DataFrame,
    input_path: str,
    plus_key: TrackKey,
    minus_key: TrackKey,
    sample_df: pd.DataFrame,
    group_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read methylation at hotspot representative CpGs; build sample + group profiles."""
    if cluster_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    row_ids = cluster_df["representative_row_id"].to_numpy(dtype=np.int64)

    reader_plus, _, _ = load_view_reader(input_path, plus_key)
    reader_minus, _, _ = load_view_reader(input_path, minus_key)
    try:
        sample_ids: list[str] = list(reader_plus.columns["sample_id"])
        plus_vals = reader_plus.read_rows(row_ids)    # (n_hotspots, n_samples)
        minus_vals = reader_minus.read_rows(row_ids)
    finally:
        reader_plus.close()
        reader_minus.close()

    paired = np.isfinite(plus_vals) & np.isfinite(minus_vals)
    diff_vals = np.where(paired, plus_vals - minus_vals, np.nan).astype(np.float32)

    n_hs, n_smp = len(row_ids), len(sample_ids)
    long = pd.DataFrame({
        "representative_row_id": np.repeat(row_ids, n_smp),
        "sample_id": np.tile(sample_ids, n_hs),
        "plus_value": plus_vals.ravel().astype(np.float32),
        "minus_value": minus_vals.ravel().astype(np.float32),
        "diff_value": diff_vals.ravel(),
        "is_paired": paired.ravel(),
    })

    long = long.merge(
        cluster_df[["representative_row_id", "hotspot_rank", "hotspot_label", "representative_coord"]],
        on="representative_row_id",
        how="left",
    )

    # Join sample metadata
    if sample_df is not None and not sample_df.empty:
        meta_cols = ["sample_id"] + [c for c in sample_df.columns if c != "sample_id"]
        long = long.merge(sample_df[meta_cols], on="sample_id", how="left")

    # Per-group aggregation
    if group_col and group_col in long.columns:
        group_profiles = (
            long[long["is_paired"]]
            .groupby(
                ["hotspot_rank", "hotspot_label", "representative_coord", group_col],
                as_index=False,
                observed=True,
            )
            .agg(
                n_samples=("sample_id", "nunique"),
                mean_plus=("plus_value", "mean"),
                mean_minus=("minus_value", "mean"),
                mean_diff=("diff_value", "mean"),
                median_diff=("diff_value", "median"),
            )
            .sort_values(["hotspot_rank", group_col])
            .reset_index(drop=True)
        )
    else:
        group_profiles = pd.DataFrame()

    return long, group_profiles


# ── BED output ────────────────────────────────────────────────────────────────

def _write_bed(cluster_df: pd.DataFrame, outpath: str) -> None:
    if cluster_df.empty:
        return
    bed = cluster_df[["chrom", "start", "end", "hotspot_label", "max_hotspot_score", "direction"]].copy()
    bed["start"] = bed["start"].astype(int)
    bed["end"] = bed["end"].astype(int)
    bed.to_csv(outpath, sep="\t", index=False, header=False)


# ── HTML: strand bias scatter ─────────────────────────────────────────────────

def _write_strand_bias_html(
    sample_df: pd.DataFrame,
    group_col: str | None,
    track_label: str,
    outdir: str,
    style_names: list[str],
    png_ok: bool,
    logger: logging.Logger,
) -> None:
    if "mean_diff_global" not in sample_df.columns or sample_df.empty:
        return

    plot_df = sample_df.copy()
    plot_df = plot_df.sort_values("mean_diff_global").reset_index(drop=True)
    plot_df["sample_rank"] = np.arange(1, len(plot_df) + 1)
    # Format hover values with more decimal places
    plot_df["mean_diff_fmt"] = plot_df["mean_diff_global"].map(lambda v: f"{v:.5f}")
    plot_df["balance_idx_fmt"] = plot_df["balance_index_global"].map(lambda v: f"{v:.5f}")

    color_cols: list[str] = []
    for c in ([group_col] if group_col else []) + [
        "tissue_broad", "tissue_name", "preservation", "sex", "technology", "center"
    ]:
        if c and c in plot_df.columns and c not in color_cols:
            if plot_df[c].nunique(dropna=False) > 1:
                color_cols.append(c)

    hover_cols = [c for c in [
        "sample_id", "donor", "tissue_name", "tissue_broad",
        "n_paired_global", "mean_diff_fmt", "balance_idx_fmt",
    ] if c in plot_df.columns]
    color_styles = build_color_styles(plot_df, color_cols)

    for i, style_name in enumerate(style_names):
        style = _style_preset(style_name)
        suffix = f"_{style_name}" if i > 0 else ""
        html_path = os.path.join(outdir, f"strand_bias{suffix}.html")
        fig = make_dropdown_scatter(
            plot_df,
            x="sample_rank",
            y="mean_diff_global",
            color_cols=color_cols,
            hover_cols=hover_cols,
            title=f"Per-sample strand bias — {track_label}",
            color_styles=color_styles,
            style_name=style_name,
            x_axis_label="Sample rank (sorted by global strand bias)",
            y_axis_label="Mean(plus − minus) across genome-wide paired CpGs",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        # Use scientific notation for the tiny y-axis values
        fig.update_yaxes(tickformat=".4f")
        fig.write_html(html_path, include_plotlyjs="cdn")
        logger.info(f"Wrote {html_path}")
        if i == 0 and png_ok:
            write_plotly_image_safe(fig, html_path.replace(".html", ".png"), logger)


# ── HTML: hotspot ranking scatter ─────────────────────────────────────────────

def _write_hotspot_ranking_html(
    cluster_df: pd.DataFrame,
    label: str,
    outdir: str,
    filename_stem: str,
    style_names: list[str],
    png_ok: bool,
    logger: logging.Logger,
) -> None:
    if cluster_df.empty:
        return

    for i, style_name in enumerate(style_names):
        style = _style_preset(style_name)
        suffix = f"_{style_name}" if i > 0 else ""
        html_path = os.path.join(outdir, f"{filename_stem}{suffix}.html")

        direction_colors = {"plus>minus": style["colorway"][0], "minus>plus": style["colorway"][2]}
        fig = px.scatter(
            cluster_df,
            x="hotspot_rank",
            y="max_abs_mean_diff",
            color="direction",
            size="n_cpgs",
            hover_data=["hotspot_label", "representative_coord", "rep_mean_diff",
                        "rep_paired_samples", "n_cpgs"],
            color_discrete_map=direction_colors,
            title=f"Hotspot effect-size ranking — {label}",
            size_max=22,
        )
        apply_style_to_figure(fig, style_name=style_name, with_dropdown=False)
        fig.update_xaxes(title_text="Hotspot rank")
        fig.update_yaxes(title_text="Max |mean(plus − minus)| in cluster")
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
        fig.write_html(html_path, include_plotlyjs="cdn")
        logger.info(f"Wrote {html_path}")
        if i == 0 and png_ok:
            write_plotly_image_safe(fig, html_path.replace(".html", ".png"), logger)


# ── HTML: category × hotspot profile heatmap ─────────────────────────────────

_HEATMAP_MAX_ROWS = 80   # cap for readability; top-ranked hotspots are most useful

def _write_hotspot_profiles_html(
    group_profiles: pd.DataFrame,
    group_col: str,
    outdir: str,
    style_names: list[str],
    png_ok: bool,
    logger: logging.Logger,
) -> None:
    if group_profiles.empty or "hotspot_label" not in group_profiles.columns:
        return

    # Pivot: rows = hotspot_label (ranked), columns = category
    pivot_full = group_profiles.pivot_table(
        index="hotspot_label",
        columns=group_col,
        values="mean_diff",
        aggfunc="mean",
    )
    # Preserve hotspot rank order
    ordered_labels = (
        group_profiles.drop_duplicates("hotspot_rank")
        .sort_values("hotspot_rank")["hotspot_label"]
        .tolist()
    )
    pivot_full = pivot_full.reindex([l for l in ordered_labels if l in pivot_full.index])
    # Cap rows for readability
    pivot = pivot_full.head(_HEATMAP_MAX_ROWS)
    n_shown = len(pivot)
    n_total = len(pivot_full)
    title_suffix = f" (top {n_shown} of {n_total})" if n_total > n_shown else ""

    for i, style_name in enumerate(style_names):
        style = _style_preset(style_name)
        suffix = f"_{style_name}" if i > 0 else ""
        html_path = os.path.join(outdir, f"hotspot_profiles{suffix}.html")

        abs_max = float(np.nanmax(np.abs(pivot.values))) if pivot.size > 0 else 0.1
        abs_max = max(abs_max, 0.01)

        # Compute label-length-aware left margin
        max_label_len = max((len(str(l)) for l in pivot.index), default=20)
        left_margin = max(200, min(380, max_label_len * 7 + 20))

        # Column width: enough for tissue names
        max_col_len = max((len(str(c)) for c in pivot.columns), default=8)
        col_width = max(80, max_col_len * 9)
        plot_width = max(700, col_width * pivot.shape[1] + left_margin + 160)

        # Row height: cap at 800px total
        row_h = max(8, min(18, 800 // n_shown))
        plot_height = row_h * n_shown + 220

        fig = px.imshow(
            pivot,
            color_continuous_scale="RdBu_r",
            zmin=-abs_max,
            zmax=abs_max,
            aspect="auto",
            title=f"Strand bias at hotspot loci by {group_col}{title_suffix}",
        )
        fig.update_coloraxes(
            colorbar_title="mean<br>(plus−minus)",
            colorbar_tickformat=".3f",
            colorbar_len=0.6,
        )
        fig.update_layout(
            template="none",
            width=plot_width,
            height=plot_height,
            paper_bgcolor=style["paper_bg"],
            plot_bgcolor=style["plot_bg"],
            font=dict(family="Source Sans Pro, Arial, sans-serif", size=12,
                      color=style["font_color"]),
            title=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                       font=dict(size=18, color=style["font_color"])),
            margin=dict(l=left_margin, r=130, t=80, b=120),
            xaxis=dict(
                tickangle=-35,
                tickfont=dict(size=11),
                title=dict(text=group_col, font=dict(size=13)),
                side="bottom",
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                title=dict(text="Hotspot locus (global rank)", font=dict(size=13)),
                autorange="reversed",
            ),
        )
        # Add subtle grid lines
        fig.update_traces(
            selector=dict(type="heatmap"),
            xgap=1,
            ygap=1,
        )
        fig.write_html(html_path, include_plotlyjs="cdn")
        logger.info(f"Wrote {html_path}")
        if i == 0 and png_ok:
            write_plotly_image_safe(fig, html_path.replace(".html", ".png"), logger)


# ── HTML: category strand summary bar chart ──────────────────────────────────

def _write_category_summary_html(
    cat_summary_df: pd.DataFrame,
    group_col: str,
    track_label: str,
    outdir: str,
    style_names: list[str],
    png_ok: bool,
    logger: logging.Logger,
) -> None:
    if cat_summary_df.empty or group_col not in cat_summary_df.columns:
        return

    plot_df = cat_summary_df.sort_values("median_mean_diff", ascending=True).reset_index(drop=True)
    plot_df["direction"] = np.where(plot_df["median_mean_diff"] >= 0, "plus > minus", "minus > plus")
    plot_df["label"] = plot_df.apply(
        lambda r: f"{r[group_col]} (n={int(r.n_samples)})", axis=1
    )

    for i, style_name in enumerate(style_names):
        style = _style_preset(style_name)
        suffix = f"_{style_name}" if i > 0 else ""
        html_path = os.path.join(outdir, f"category_strand_summary{suffix}.html")

        direction_colors = {
            "plus > minus": style["colorway"][0],
            "minus > plus": style["colorway"][2],
        }
        fig = px.bar(
            plot_df,
            x="median_mean_diff",
            y="label",
            color="direction",
            orientation="h",
            color_discrete_map=direction_colors,
            hover_data=["n_samples", "mean_mean_diff", "median_balance_index"],
            title=f"Genome-wide strand bias by {group_col} — {track_label}",
            barmode="relative",
        )
        # Auto-size left margin from label lengths
        max_lbl = max((len(str(v)) for v in plot_df["label"]), default=20)
        left_margin = max(160, min(340, max_lbl * 7 + 20))
        apply_style_to_figure(fig, style_name=style_name, with_dropdown=False)
        fig.update_layout(
            height=max(450, 45 * len(plot_df) + 150),
            width=max(900, left_margin + 620),
            margin=dict(l=left_margin, r=80, t=100, b=90),
            yaxis=dict(title="", tickfont=dict(size=12)),
            xaxis=dict(
                title="Median mean(plus − minus) across samples",
                tickformat=".5f",
            ),
            showlegend=True,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.6)
        fig.write_html(html_path, include_plotlyjs="cdn")
        logger.info(f"Wrote {html_path}")
        if i == 0 and png_ok:
            write_plotly_image_safe(fig, html_path.replace(".html", ".png"), logger)


# ── HTML: per-category hotspot comparison scatter ─────────────────────────────

def _write_category_comparison_html(
    all_clusters: dict[str, pd.DataFrame],
    group_col: str,
    outdir: str,
    style_names: list[str],
    png_ok: bool,
    logger: logging.Logger,
) -> None:
    """Scatter comparing hotspot score and n_cpgs across categories."""
    if not all_clusters:
        return

    frames = []
    for cat, df in all_clusters.items():
        if df.empty:
            continue
        tmp = df[["hotspot_rank", "hotspot_label", "chrom", "representative_coord",
                  "direction", "max_abs_mean_diff", "max_hotspot_score", "n_cpgs",
                  "rep_mean_diff", "rep_paired_samples"]].copy()
        tmp[group_col] = cat
        frames.append(tmp)
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    color_cols = [group_col]
    hover_cols = ["hotspot_label", "representative_coord", "direction",
                  "max_abs_mean_diff", "n_cpgs", "rep_paired_samples"]
    color_styles = build_color_styles(combined, color_cols)

    for i, style_name in enumerate(style_names):
        suffix = f"_{style_name}" if i > 0 else ""
        html_path = os.path.join(outdir, f"category_hotspot_comparison{suffix}.html")
        fig = make_dropdown_scatter(
            combined,
            x="hotspot_rank",
            y="max_abs_mean_diff",
            color_cols=color_cols,
            hover_cols=hover_cols,
            title=f"Per-{group_col} hotspot rankings",
            color_styles=color_styles,
            style_name=style_name,
            x_axis_label="Hotspot rank within category",
            y_axis_label="Max |mean(plus − minus)| in cluster",
        )
        fig.write_html(html_path, include_plotlyjs="cdn")
        logger.info(f"Wrote {html_path}")
        if i == 0 and png_ok:
            write_plotly_image_safe(fig, html_path.replace(".html", ".png"), logger)


# ── Style helper ──────────────────────────────────────────────────────────────

def _style_preset(name: str) -> dict:
    from mdb.plotting import PLOT_STYLE_PRESETS
    return PLOT_STYLE_PRESETS.get(name, PLOT_STYLE_PRESETS["studio"])


# ── Main entry point ──────────────────────────────────────────────────────────

def strand_main(args) -> None:
    outdir: str = args.outdir
    os.makedirs(outdir, exist_ok=True)
    logger = setup_logging(outdir, bool(getattr(args, "verbose", False)))

    t0 = time.time()
    logger.info("==== Run started ====")
    logger.info(f"Start time: {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {outdir}")

    # Validate store
    kind = detect_store_kind(args.input)
    if kind not in {"cohort_store_npy", "cohort_store_zarr"}:
        raise ValueError(
            f"mdb strand expects a cohort store (.mmdb); found {kind!r} at {args.input}"
        )

    config = StrandConfig(
        assay=args.assay,
        haplotype=args.haplotype,
        min_paired_frac=float(args.min_paired_frac),
        min_mean_total=float(args.min_mean_total),
        cluster_gap_bp=int(args.cluster_gap_bp),
        min_cluster_cpgs=int(args.min_cluster_cpgs),
        top_n_hotspots=int(args.top_n_hotspots),
        batch_rows=int(args.batch_rows),
        workers=int(getattr(args, "workers", 1)),
    )

    plus_key, minus_key = _find_strand_pair(args.input, config.assay, config.haplotype)
    track_label = f"{config.assay} / {config.haplotype}"
    logger.info(f"Track pair: {plus_key.name()} / {minus_key.name()}")

    # Cohort index
    chroms, chrom_offsets, pos0 = load_cohort_index(args.input)
    total_rows = int(pos0.shape[0])
    logger.info(f"Reference CpGs: {total_rows:,}")

    # Sample list
    columns = load_view_columns(args.input, plus_key)
    sample_ids: list[str] = list(columns["sample_id"])
    n_samples = len(sample_ids)
    sample_df = pd.DataFrame({"sample_id": sample_ids, "id": sample_ids})
    logger.info(f"Samples in store: {n_samples}")

    # Metadata
    metadata_path = getattr(args, "metadata", None)
    if metadata_path:
        sample_df, meta = maybe_merge_metadata(sample_df, metadata_path, logger=logger)
        logger.info(
            f"Metadata merged: {sample_df.shape[1] - 2} additional columns"
            f" ({'OK' if meta is not None else 'NOT aligned — ignored'})"
        )
    else:
        meta = None

    # Per-category groups
    group_col: str | None = getattr(args, "group_by", None)
    group_indices: dict[str, list[int]] = {}
    if group_col:
        if group_col not in sample_df.columns:
            logger.warning(
                f"--group-by column {group_col!r} not found in metadata; "
                "skipping per-category hotspot calling"
            )
            group_col = None
        else:
            for cat, sub in sample_df.groupby(group_col, sort=True, observed=True):
                group_indices[str(cat)] = sorted(sub.index.tolist())
            logger.info(
                f"Stratifying by {group_col!r}: {len(group_indices)} categories "
                f"({', '.join(list(group_indices)[:6])}"
                f"{'...' if len(group_indices) > 6 else ''})"
            )

    png_ok = plotly_png_ok()
    style_names = resolve_plot_styles(args)
    logger.info(f"Plot styles: {style_names}")
    logger.info(f"Plotly PNG: {'yes' if png_ok else 'no'}")

    # ── Full-genome scan ──────────────────────────────────────────────────────
    scan_result = _run_scan(
        args.input, plus_key, minus_key, total_rows,
        group_indices, config, logger,
    )

    # ── Per-sample global strand bias table ───────────────────────────────────
    ss = scan_result["sample_stats"]
    per_sample_df = sample_df.copy()
    per_sample_df["n_paired_global"] = ss["paired_obs"]
    per_sample_df["mean_plus_global"] = ss["mean_plus"]
    per_sample_df["mean_minus_global"] = ss["mean_minus"]
    per_sample_df["mean_diff_global"] = ss["mean_diff"]
    per_sample_df["balance_index_global"] = ss["balance_index"]
    per_sample_df = per_sample_df.drop(columns=["id"], errors="ignore")

    per_sample_path = os.path.join(outdir, "per_sample_metrics.tsv.gz")
    per_sample_df.to_csv(per_sample_path, sep="\t", index=False)
    logger.info(f"Wrote {per_sample_path}")

    # ── Global hotspots ───────────────────────────────────────────────────────
    global_cands = _annotate_rows(scan_result["global_candidates"], chroms, chrom_offsets, pos0)
    logger.info(f"Global candidates before clustering: {len(global_cands):,}")
    global_clusters = _cluster_hotspots(global_cands, config)
    logger.info(f"Global hotspot clusters: {len(global_clusters)}")

    global_hs_path = os.path.join(outdir, "hotspots_global.tsv")
    global_bed_path = os.path.join(outdir, "hotspots_global.bed")
    global_clusters.to_csv(global_hs_path, sep="\t", index=False)
    _write_bed(global_clusters, global_bed_path)
    logger.info(f"Wrote {global_hs_path}")
    logger.info(f"Wrote {global_bed_path}")

    # ── Global hotspot profiles ───────────────────────────────────────────────
    sample_profiles, group_profiles = _extract_profiles(
        global_clusters, args.input, plus_key, minus_key,
        per_sample_df, group_col,
    )
    if not sample_profiles.empty:
        sp_path = os.path.join(outdir, "hotspot_sample_profiles.tsv.gz")
        sample_profiles.to_csv(sp_path, sep="\t", index=False)
        logger.info(f"Wrote {sp_path}")
    if not group_profiles.empty:
        gp_path = os.path.join(outdir, f"hotspot_{group_col}_profiles.tsv")
        group_profiles.to_csv(gp_path, sep="\t", index=False)
        logger.info(f"Wrote {gp_path}")

    # ── Per-category hotspots ─────────────────────────────────────────────────
    cat_cluster_map: dict[str, pd.DataFrame] = {}
    for cat, cat_cands_df in scan_result["cat_candidates"].items():
        annotated = _annotate_rows(cat_cands_df, chroms, chrom_offsets, pos0)
        cat_clusters = _cluster_hotspots(annotated, config)
        cat_cluster_map[cat] = cat_clusters

        safe = cat.replace("/", "_").replace(" ", "_").replace(",", "")
        cat_tsv = os.path.join(outdir, f"hotspots_{group_col}_{safe}.tsv")
        cat_bed = os.path.join(outdir, f"hotspots_{group_col}_{safe}.bed")
        cat_clusters.to_csv(cat_tsv, sep="\t", index=False)
        _write_bed(cat_clusters, cat_bed)
        logger.info(
            f"  {group_col}={cat!r}: {len(cat_clusters)} hotspot clusters → {cat_tsv}"
        )

    # ── Category summary (global strand bias per group) ───────────────────────
    if group_col and group_col in per_sample_df.columns:
        cat_summary = (
            per_sample_df.groupby(group_col, observed=True, sort=True)
            .agg(
                n_samples=("sample_id", "nunique"),
                median_mean_diff=("mean_diff_global", "median"),
                mean_mean_diff=("mean_diff_global", "mean"),
                median_balance_index=("balance_index_global", "median"),
            )
            .reset_index()
        )
        cat_summary_path = os.path.join(outdir, f"{group_col}_strand_summary.tsv")
        cat_summary.to_csv(cat_summary_path, sep="\t", index=False)
        logger.info(f"Wrote {cat_summary_path}")

    # ── HTML plots ────────────────────────────────────────────────────────────
    _write_strand_bias_html(
        per_sample_df, group_col, track_label,
        outdir, style_names, png_ok, logger,
    )
    _write_hotspot_ranking_html(
        global_clusters, f"global — {track_label}",
        outdir, "hotspot_ranking_global",
        style_names, png_ok, logger,
    )
    if not group_profiles.empty:
        _write_hotspot_profiles_html(
            group_profiles, group_col,
            outdir, style_names, png_ok, logger,
        )
    if group_col and group_col in per_sample_df.columns:
        _write_category_summary_html(
            cat_summary, group_col, track_label,
            outdir, style_names, png_ok, logger,
        )
    if cat_cluster_map:
        _write_category_comparison_html(
            cat_cluster_map, group_col,
            outdir, style_names, png_ok, logger,
        )

    # ── params.json ───────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    params = {
        "input": str(args.input),
        "outdir": str(outdir),
        "assay": config.assay,
        "haplotype": config.haplotype,
        "min_paired_frac": config.min_paired_frac,
        "min_mean_total": config.min_mean_total,
        "cluster_gap_bp": config.cluster_gap_bp,
        "min_cluster_cpgs": config.min_cluster_cpgs,
        "top_n_hotspots": config.top_n_hotspots,
        "batch_rows": config.batch_rows,
        "workers": config.workers,
        "metadata": str(metadata_path) if metadata_path else None,
        "group_by": group_col,
        "n_samples": n_samples,
        "n_reference_cpgs": total_rows,
        "n_global_hotspot_clusters": int(len(global_clusters)),
        "n_categories": int(len(group_indices)),
        "runtime_seconds": elapsed,
    }
    with open(os.path.join(outdir, "params.json"), "w") as fh:
        json.dump(params, fh, indent=2)

    logger.info(f"==== Done in {elapsed:.1f}s ====")


# ── Public programmatic API ───────────────────────────────────────────────────

def run_strand(
    input_path: str,
    outdir: str,
    *,
    metadata: str | None = None,
    group_by: str | None = None,
    assay: str = "5hmC",
    haplotype: str = "combined",
    min_paired_frac: float = 0.8,
    min_mean_total: float = 0.005,
    cluster_gap_bp: int = 1_000,
    top_n_hotspots: int = 500,
    batch_rows: int = 65_536,
    workers: int = 1,
    plot_style: str = "studio",
    plot_style_variants: bool = False,
    verbose: bool = False,
) -> None:
    """Programmatic entry point for mdb strand analysis."""
    args = SimpleNamespace(
        input=input_path,
        outdir=outdir,
        metadata=metadata,
        group_by=group_by,
        assay=assay,
        haplotype=haplotype,
        min_paired_frac=min_paired_frac,
        min_mean_total=min_mean_total,
        cluster_gap_bp=cluster_gap_bp,
        top_n_hotspots=top_n_hotspots,
        batch_rows=batch_rows,
        workers=workers,
        plot_style=plot_style,
        plot_style_variants=plot_style_variants,
        verbose=verbose,
    )
    strand_main(args)
