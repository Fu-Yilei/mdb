#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime
from types import SimpleNamespace

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
    maybe_use_concise_sample_ids,
    plotly_png_ok,
    resolve_plot_styles,
    write_plotly_image_safe,
)
from mdb.schema import TrackKey, VALUE_MISSING
from mdb.storage import (
    available_views,
    detect_store_kind,
    load_cohort_index,
    load_track_manifest,
    load_view_columns,
    load_view_reader,
    read_track,
)


def setup_logging(outdir: str, verbose: bool) -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("mdb_stats")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(outdir, "stats.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def format_track_label(key: TrackKey) -> str:
    return f"{key.assay} / {key.haplotype} / {key.strand}"


def select_tracks(input_path: str, args) -> list[TrackKey]:
    views = available_views(input_path)
    assay = _parse_selector(getattr(args, "assay", None))
    haplotype = _parse_selector(getattr(args, "haplotype", None))
    strand = _parse_selector(getattr(args, "strand", None))

    selected = [
        key
        for key in views
        if (not assay or key.assay in assay)
        and (not haplotype or key.haplotype in haplotype)
        and (not strand or key.strand in strand)
    ]
    if not selected:
        raise ValueError(
            "No cohort views matched the requested filters: "
            f"assay={sorted(assay) if assay else 'ALL'}, "
            f"haplotype={sorted(haplotype) if haplotype else 'ALL'}, "
            f"strand={sorted(strand) if strand else 'ALL'}"
        )
    return selected


def _parse_selector(value: str | None) -> set[str] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "all":
        return None
    return {part.strip() for part in text.split(",") if part.strip()}


def count_view_observed_cpgs(input_path: str, key: TrackKey, batch_rows: int, logger: logging.Logger) -> np.ndarray:
    reader, _, _ = load_view_reader(input_path, key)
    total_blocks = int(math.ceil(float(reader.shape[0]) / float(batch_rows))) if reader.shape[0] else 0
    logger.info(f"Scanning cohort matrix for {key.name()} to recover observed-CpG counts")
    try:
        counts = np.zeros(int(reader.shape[1]), dtype=np.int64)
        for block in tqdm(reader.iter_blocks(batch_rows), total=total_blocks, desc=f"scan {key.name()}", leave=False):
            counts += np.count_nonzero(~np.isnan(np.asarray(block, dtype=np.float32)), axis=0).astype(np.int64, copy=False)
        return counts
    finally:
        reader.close()


def count_sample_track_observed_cpgs(bundle_path: str, key: TrackKey) -> int:
    track = read_track(bundle_path, key)
    try:
        total = 0
        for chrom in track.chroms_present:
            values = np.asarray(track.chrom_values(chrom, allow_missing=True), dtype=np.uint16)
            total += int(np.count_nonzero(values != VALUE_MISSING))
        return total
    finally:
        track.close()


def collect_view_sample_stats(
    input_path: str,
    key: TrackKey,
    *,
    n_reference_cpgs: int,
    batch_rows: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    columns = load_view_columns(input_path, key)
    n_samples = len(columns["sample_id"])
    n_obs_rows = np.full(n_samples, -1, dtype=np.int64)
    min_coverage = np.full(n_samples, np.nan, dtype=np.float64)
    n_chroms_present = np.full(n_samples, np.nan, dtype=np.float64)
    stats_source = np.full(n_samples, "track_manifest", dtype=object)

    missing_manifest = False
    missing_error = None
    for idx, bundle_path in enumerate(columns["bundle_path"]):
        try:
            meta = load_track_manifest(bundle_path, key)
        except Exception as exc:  # fallback path for derived or incomplete bundles
            missing_manifest = True
            missing_error = exc
            continue

        n_obs_value = int(meta.get("n_obs_rows", meta.get("n_rows", -1)))
        if n_obs_value < 0:
            try:
                n_obs_value = count_sample_track_observed_cpgs(bundle_path, key)
                stats_source[idx] = "sample_track_scan"
            except Exception as exc:
                missing_manifest = True
                missing_error = exc
                continue

        n_obs_rows[idx] = n_obs_value
        if "min_coverage" in meta:
            min_coverage[idx] = float(meta["min_coverage"])
        chroms_present = meta.get("chroms_present", [])
        if isinstance(chroms_present, list):
            n_chroms_present[idx] = float(len(chroms_present))

    if missing_manifest or np.any(n_obs_rows < 0):
        if missing_error is not None:
            logger.info(f"{key.name()}: per-sample track manifest stats incomplete; falling back to matrix scan ({missing_error})")
        n_obs_rows = count_view_observed_cpgs(input_path, key, batch_rows=batch_rows, logger=logger)
        stats_source = np.full(n_samples, "matrix_scan", dtype=object)

    out = pd.DataFrame(
        {
            "id": pd.Series(columns["sample_id"], dtype="string"),
            "sample_id": pd.Series(columns["sample_id"], dtype="string"),
            "bundle_path": pd.Series(columns["bundle_path"], dtype="string"),
            "platform": pd.Series(columns["platform"], dtype="string"),
            "source_path": pd.Series(columns["source_path"], dtype="string"),
            "input_tag": pd.Series(columns["input_tag"], dtype="string"),
            "track": key.name(),
            "track_display": format_track_label(key),
            "assay": key.assay,
            "haplotype": key.haplotype,
            "strand": key.strand,
            "n_reference_cpgs": int(n_reference_cpgs),
            "n_cpgs_observed": pd.Series(n_obs_rows, dtype="int64"),
            "frac_cpgs_observed": pd.Series(n_obs_rows / float(n_reference_cpgs), dtype="float64"),
            "min_coverage": pd.Series(min_coverage, dtype="float64"),
            "n_chroms_present": pd.Series(n_chroms_present, dtype="float64"),
            "stats_source": pd.Series(stats_source, dtype="string"),
        }
    )
    return out


def add_display_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sample_id" not in df.columns:
        return df

    candidate_cols = ["sample_name", "sample", "sample_id_y", "sample_id_x"]
    sample_df = df[[c for c in ["sample_id", "id"] + candidate_cols if c in df.columns]].drop_duplicates("sample_id").copy()
    sample_df["id"] = sample_df["sample_id"].astype(str)
    sample_df = maybe_use_concise_sample_ids(sample_df)
    display_map = sample_df.set_index("sample_id")["id"].astype(str).to_dict()

    out = df.copy()
    out["display_id"] = out["sample_id"].astype(str).map(display_map).fillna(out["sample_id"].astype(str))
    return out


def _is_reasonable_category(series: pd.Series, max_unique: int) -> bool:
    vals = series.dropna()
    if vals.empty:
        return False
    text = vals.astype(str).str.strip()
    text = text[text != ""]
    if text.empty:
        return False
    nunique = int(text.nunique(dropna=True))
    return 2 <= nunique <= max_unique


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def stats_plot_columns(meta: pd.DataFrame | None, df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    hover_priority = [
        "display_id",
        "sample_id",
        "sample_name",
        "donor",
        "tissue_name",
        "tissue_broad",
        "technology",
        "center",
        "platform",
        "input_tag",
        "track_display",
        "assay",
        "haplotype",
        "strand",
        "n_cpgs_observed",
        "frac_cpgs_observed",
        "stats_source",
    ]
    hover_cols = [col for col in hover_priority if col in df.columns]

    meta_cols: list[str] = []
    if meta is not None:
        for col in meta.columns:
            if col in df.columns and col != "id" and _is_reasonable_category(df[col], max_unique=24):
                meta_cols.append(col)

    builtins = [
        "track_display",
        "platform",
        "input_tag",
        "assay",
        "haplotype",
        "strand",
        "technology",
        "center",
    ]
    scatter_color_cols = [
        col
        for col in _dedupe_keep_order(builtins + meta_cols)
        if col in df.columns and _is_reasonable_category(df[col], max_unique=32)
    ]
    box_color_cols = [
        col
        for col in _dedupe_keep_order(meta_cols + ["platform", "input_tag", "assay", "haplotype", "strand", "track_display"])
        if col in df.columns and _is_reasonable_category(df[col], max_unique=24)
    ]
    return scatter_color_cols, box_color_cols, hover_cols


def build_symbol_map(df: pd.DataFrame, column: str | None) -> dict[str, str] | None:
    if not column or column not in df.columns:
        return None
    vals = [str(v) for v in pd.Series(df[column], dtype="string").fillna("NA").astype(str).unique().tolist()]
    if len(vals) <= 1:
        return None
    symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "star",
        "hexagram",
        "pentagon",
        "hexagon",
    ]
    return {value: symbols[idx % len(symbols)] for idx, value in enumerate(vals)}


def make_dropdown_boxplot(
    *,
    df: pd.DataFrame,
    x: str,
    y: str,
    color_cols: list[str],
    hover_cols: list[str],
    title: str,
    color_styles: dict[str, dict[str, object]] | None = None,
    style_name: str = "studio",
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
):
    if not color_cols:
        fig = px.box(df, x=x, y=y, hover_data=hover_cols, title=title, points="all")
        fig.update_layout(boxmode="group")
        apply_style_to_figure(fig, style_name=style_name, with_dropdown=False)
        fig.update_xaxes(title_text=x_axis_label or x)
        fig.update_yaxes(title_text=y_axis_label or y)
        fig.update_traces(selector=dict(type="box"), jitter=0.28, pointpos=0, marker=dict(size=4, opacity=0.45))
        return fig

    master = None
    groups: list[tuple[str, int, int]] = []
    initial_title = title
    for idx, col in enumerate(color_cols):
        if col not in df.columns:
            continue
        tmp_df = df.copy()
        tmp_df[col] = pd.Series(tmp_df[col], dtype="string").fillna("NA").astype(str)
        box_args: dict[str, object] = {}
        style = (color_styles or {}).get(col, {})
        if "ordered" in style:
            box_args["category_orders"] = {col: style["ordered"]}
        if "cmap" in style:
            box_args["color_discrete_map"] = style["cmap"]
        tmp = px.box(
            tmp_df,
            x=x,
            y=y,
            color=col,
            hover_data=hover_cols,
            title=f"{title} (color_by={col})",
            points="all",
            **box_args,
        )
        tmp.update_layout(boxmode="group")
        tmp.update_traces(selector=dict(type="box"), jitter=0.28, pointpos=0, marker=dict(size=4, opacity=0.45))

        if master is None:
            master = go.Figure(tmp)
            start = 0
            end = len(master.data)
            for tr in master.data:
                tr.visible = True
        else:
            start = len(master.data)
            for tr in tmp.data:
                tr.visible = False
                master.add_trace(tr)
            end = len(master.data)

        groups.append((col, start, end))
        if idx == 0:
            initial_title = f"{title} (color_by={col})"

    if master is None or not groups:
        fig = px.box(df, x=x, y=y, hover_data=hover_cols, title=title, points="all")
        fig.update_layout(boxmode="group")
        apply_style_to_figure(fig, style_name=style_name, with_dropdown=False)
        fig.update_xaxes(title_text=x_axis_label or x)
        fig.update_yaxes(title_text=y_axis_label or y)
        fig.update_traces(selector=dict(type="box"), jitter=0.28, pointpos=0, marker=dict(size=4, opacity=0.45))
        return fig

    buttons = []
    n_traces = len(master.data)
    for col, start, end in groups:
        visible = [False] * n_traces
        for trace_idx in range(start, end):
            visible[trace_idx] = True
        buttons.append(
            dict(
                label=col,
                method="update",
                args=[{"visible": visible}, {"title": f"{title} (color_by={col})"}],
            )
        )

    master.update_layout(
        title=initial_title,
        boxmode="group",
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.02, xanchor="left", y=1.0, yanchor="top")],
    )
    apply_style_to_figure(master, style_name=style_name, with_dropdown=True)
    master.update_xaxes(title_text=x_axis_label or x)
    master.update_yaxes(title_text=y_axis_label or y)
    master.update_traces(selector=dict(type="box"), jitter=0.28, pointpos=0, marker=dict(size=4, opacity=0.45))
    return master


def write_scatter_with_styles(
    *,
    df: pd.DataFrame,
    x: str,
    y: str,
    color_cols: list[str],
    hover_cols: list[str],
    title: str,
    color_styles: dict[str, dict[str, object]],
    out_html: str,
    args,
    logger: logging.Logger,
    png_ok: bool = False,
    out_png: str | None = None,
    symbol_col: str | None = None,
    symbol_map: dict[str, str] | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
) -> None:
    styles = resolve_plot_styles(args)
    html_root, html_ext = os.path.splitext(out_html)
    for idx, style_name in enumerate(styles):
        fig = make_dropdown_scatter(
            df=df,
            x=x,
            y=y,
            color_cols=color_cols,
            hover_cols=hover_cols,
            title=title,
            color_styles=color_styles,
            symbol_col=symbol_col,
            symbol_map=symbol_map,
            style_name=style_name,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
        )
        html_path = out_html if idx == 0 else f"{html_root}_{style_name}{html_ext}"
        fig.write_html(html_path, include_plotlyjs="cdn")
        if idx == 0 and png_ok and out_png:
            write_plotly_image_safe(fig, out_png, logger)


def write_boxplot_with_styles(
    *,
    df: pd.DataFrame,
    x: str,
    y: str,
    color_cols: list[str],
    hover_cols: list[str],
    title: str,
    color_styles: dict[str, dict[str, object]],
    out_html: str,
    args,
    logger: logging.Logger,
    png_ok: bool = False,
    out_png: str | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
) -> None:
    styles = resolve_plot_styles(args)
    html_root, html_ext = os.path.splitext(out_html)
    for idx, style_name in enumerate(styles):
        fig = make_dropdown_boxplot(
            df=df,
            x=x,
            y=y,
            color_cols=color_cols,
            hover_cols=hover_cols,
            title=title,
            color_styles=color_styles,
            style_name=style_name,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
        )
        html_path = out_html if idx == 0 else f"{html_root}_{style_name}{html_ext}"
        fig.write_html(html_path, include_plotlyjs="cdn")
        if idx == 0 and png_ok and out_png:
            write_plotly_image_safe(fig, out_png, logger)


def summarize_tracks(sample_stats: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sample_stats.groupby(["track", "track_display", "assay", "haplotype", "strand"], sort=False, observed=True)
        .agg(
            n_samples=("sample_id", "nunique"),
            n_samples_with_data=("n_cpgs_observed", lambda s: int((pd.Series(s) > 0).sum())),
            mean_n_cpgs_observed=("n_cpgs_observed", "mean"),
            median_n_cpgs_observed=("n_cpgs_observed", "median"),
            min_n_cpgs_observed=("n_cpgs_observed", "min"),
            max_n_cpgs_observed=("n_cpgs_observed", "max"),
            mean_frac_cpgs_observed=("frac_cpgs_observed", "mean"),
            median_frac_cpgs_observed=("frac_cpgs_observed", "median"),
        )
        .reset_index()
    )
    return grouped


def summarize_metadata_groups(sample_stats: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for col in group_cols:
        if col not in sample_stats.columns:
            continue
        subset_cols = _dedupe_keep_order(
            ["track", "track_display", "assay", "haplotype", "strand", "sample_id", "n_cpgs_observed", "frac_cpgs_observed", col]
        )
        subset = sample_stats[subset_cols].copy()
        subset[col] = pd.Series(subset[col], dtype="string").fillna("NA").astype(str).str.strip()
        subset = subset[subset[col] != ""]
        if subset.empty or subset[col].nunique(dropna=True) < 2:
            continue
        group_keys = _dedupe_keep_order(["track", "track_display", "assay", "haplotype", "strand", col])
        grouped = (
            subset.groupby(group_keys, sort=False, observed=True)
            .agg(
                n_samples=("sample_id", "nunique"),
                mean_n_cpgs_observed=("n_cpgs_observed", "mean"),
                median_n_cpgs_observed=("n_cpgs_observed", "median"),
                min_n_cpgs_observed=("n_cpgs_observed", "min"),
                max_n_cpgs_observed=("n_cpgs_observed", "max"),
                mean_frac_cpgs_observed=("frac_cpgs_observed", "mean"),
                median_frac_cpgs_observed=("frac_cpgs_observed", "median"),
            )
            .reset_index()
            .rename(columns={col: "metadata_value"})
        )
        grouped.insert(5, "metadata_field", col)
        frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def stats(
    input_path: str,
    outdir: str,
    *,
    metadata: str | None = None,
    assay: str | None = None,
    haplotype: str | None = None,
    strand: str | None = None,
    batch_rows: int = 65_536,
    plot_style: str = "studio",
    plot_style_variants: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    args = SimpleNamespace(
        input=input_path,
        outdir=outdir,
        metadata=metadata,
        assay=assay,
        haplotype=haplotype,
        strand=strand,
        batch_rows=batch_rows,
        plot_style=plot_style,
        plot_style_variants=plot_style_variants,
        verbose=verbose,
    )
    return stats_main(args)


def stats_main(args) -> pd.DataFrame:
    input_path = args.input
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    logger = setup_logging(outdir, bool(getattr(args, "verbose", False)))

    t0 = time.time()
    logger.info("==== Run started ====")
    logger.info(f"Start time: {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"Input: {input_path}")

    kind = detect_store_kind(input_path)
    if kind not in {"cohort_store_npy", "cohort_store_zarr"}:
        raise ValueError(f"mdb stats expects a cohort store (.mmdb); found store kind {kind!r} at {input_path}")

    selected_tracks = select_tracks(input_path, args)
    logger.info(f"Selected tracks ({len(selected_tracks)}): {', '.join(key.name() for key in selected_tracks)}")

    _, _, pos0 = load_cohort_index(input_path)
    n_reference_cpgs = int(pos0.shape[0])
    logger.info(f"Reference CpGs indexed in cohort: {n_reference_cpgs:,}")

    png_ok = plotly_png_ok()
    logger.info(f"Plotly PNG export: {'available' if png_ok else 'NOT available (HTML only)'}")

    frames: list[pd.DataFrame] = []
    for key in selected_tracks:
        frames.append(
            collect_view_sample_stats(
                input_path,
                key,
                n_reference_cpgs=n_reference_cpgs,
                batch_rows=int(getattr(args, "batch_rows", 400_000)),
                logger=logger,
            )
        )

    sample_stats = pd.concat(frames, ignore_index=True)
    track_order = [key.name() for key in selected_tracks]
    track_label_order = [format_track_label(key) for key in selected_tracks]
    sample_stats["track"] = pd.Categorical(sample_stats["track"], categories=track_order, ordered=True)
    sample_stats["track_display"] = pd.Categorical(sample_stats["track_display"], categories=track_label_order, ordered=True)

    sample_stats, meta = maybe_merge_metadata(sample_stats, getattr(args, "metadata", None), logger=logger)
    sample_stats = add_display_ids(sample_stats)
    sample_stats = sample_stats.sort_values(["track", "n_cpgs_observed", "sample_id"], ascending=[True, False, True]).reset_index(drop=True)
    sample_stats["sample_rank_in_track"] = sample_stats.groupby("track", sort=False, observed=True).cumcount() + 1

    track_stats = summarize_tracks(sample_stats)
    scatter_color_cols, box_color_cols, hover_cols = stats_plot_columns(meta, sample_stats)
    metadata_group_stats = summarize_metadata_groups(sample_stats, box_color_cols)

    sample_stats_path = os.path.join(outdir, "sample_stats.tsv")
    track_stats_path = os.path.join(outdir, "track_stats.tsv")
    metadata_group_path = os.path.join(outdir, "metadata_group_stats.tsv")
    sample_stats.to_csv(sample_stats_path, sep="\t", index=False)
    track_stats.to_csv(track_stats_path, sep="\t", index=False)
    if metadata_group_stats.empty:
        if os.path.exists(metadata_group_path):
            os.remove(metadata_group_path)
    else:
        metadata_group_stats.to_csv(metadata_group_path, sep="\t", index=False)

    color_styles = build_color_styles(sample_stats, _dedupe_keep_order(scatter_color_cols + box_color_cols))
    symbol_col = "track_display" if sample_stats["track_display"].nunique(dropna=True) > 1 else None
    symbol_map = build_symbol_map(sample_stats, symbol_col)

    write_scatter_with_styles(
        df=sample_stats,
        x="sample_rank_in_track",
        y="n_cpgs_observed",
        color_cols=scatter_color_cols,
        hover_cols=hover_cols,
        title="Observed CpGs per sample",
        color_styles=color_styles,
        out_html=os.path.join(outdir, "cpg_count_scatter.html"),
        args=args,
        logger=logger,
        png_ok=png_ok,
        out_png=os.path.join(outdir, "cpg_count_scatter.png"),
        symbol_col=symbol_col,
        symbol_map=symbol_map,
        x_axis_label="Sample rank within track (sorted by observed CpGs)",
        y_axis_label="Observed CpGs",
    )

    write_scatter_with_styles(
        df=sample_stats,
        x="sample_rank_in_track",
        y="frac_cpgs_observed",
        color_cols=scatter_color_cols,
        hover_cols=hover_cols,
        title="Observed CpG fraction per sample",
        color_styles=color_styles,
        out_html=os.path.join(outdir, "frac_cpg_scatter.html"),
        args=args,
        logger=logger,
        png_ok=png_ok,
        out_png=os.path.join(outdir, "frac_cpg_scatter.png"),
        symbol_col=symbol_col,
        symbol_map=symbol_map,
        x_axis_label="Sample rank within track (sorted by observed CpGs)",
        y_axis_label="Observed CpG fraction",
    )

    write_boxplot_with_styles(
        df=sample_stats,
        x="track_display",
        y="n_cpgs_observed",
        color_cols=box_color_cols,
        hover_cols=hover_cols,
        title="Observed CpGs stratified by metadata across tracks",
        color_styles=color_styles,
        out_html=os.path.join(outdir, "cpg_count_by_track.html"),
        args=args,
        logger=logger,
        png_ok=png_ok,
        out_png=os.path.join(outdir, "cpg_count_by_track.png"),
        x_axis_label="Track (assay / haplotype / strand)",
        y_axis_label="Observed CpGs",
    )

    write_boxplot_with_styles(
        df=sample_stats,
        x="track_display",
        y="frac_cpgs_observed",
        color_cols=box_color_cols,
        hover_cols=hover_cols,
        title="Observed CpG fraction stratified by metadata across tracks",
        color_styles=color_styles,
        out_html=os.path.join(outdir, "frac_cpg_by_track.html"),
        args=args,
        logger=logger,
        png_ok=png_ok,
        out_png=os.path.join(outdir, "frac_cpg_by_track.png"),
        x_axis_label="Track (assay / haplotype / strand)",
        y_axis_label="Observed CpG fraction",
    )

    params = {
        **vars(args),
        "input_store_kind": kind,
        "n_reference_cpgs": n_reference_cpgs,
        "n_selected_tracks": len(selected_tracks),
        "selected_tracks": [key.name() for key in selected_tracks],
        "n_output_rows": int(len(sample_stats)),
        "metadata_merged": bool(meta is not None),
        "plotly_png_supported": bool(png_ok),
        "scatter_color_columns": scatter_color_cols,
        "box_color_columns": box_color_cols,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    logger.info(f"Wrote sample stats: {sample_stats_path}")
    logger.info(f"Wrote track stats: {track_stats_path}")
    if not metadata_group_stats.empty:
        logger.info(f"Wrote metadata-group stats: {metadata_group_path}")
    logger.info(
        "Wrote plots: "
        f"{os.path.join(outdir, 'cpg_count_scatter.html')}, "
        f"{os.path.join(outdir, 'cpg_count_by_track.html')}, "
        f"{os.path.join(outdir, 'frac_cpg_scatter.html')}, "
        f"{os.path.join(outdir, 'frac_cpg_by_track.html')}"
    )
    logger.info(f"Completed in {(time.time() - t0):.2f}s")

    return sample_stats
