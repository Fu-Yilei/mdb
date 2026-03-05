#!/usr/bin/env python3
"""
PCA/UMAP on:
- modern cohort stores (.mmdb; npy/zarr backends, explicit track view), or
- legacy flat merged .npy folders.

Outputs:
  embedding.tsv, params.json, pca_umap.log
  pca.html (+ pca.png when plotly image engine exists)
  umap.html (+ umap.png when requested and available)
  pca_pairplot.png
  (optional) outlier_report.tsv, outliers_only.tsv,
             pca_with_outliers_marked.html, pca_no_outliers.html,
             pca_pairplot_no_outliers.png
"""

import os
import sys
import json
import time
import glob
import math
import logging
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.decomposition import IncrementalPCA
import umap

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2

from mdb.schema import TrackKey
from mdb.storage import detect_store_kind, load_view_reader


PLOT_STYLE_PRESETS: dict[str, dict[str, object]] = {
    "studio": {
        "paper_bg": "#f5f8fc",
        "plot_bg": "#ffffff",
        "font_color": "#0f2f4f",
        "grid": "#d7e2ee",
        "axis": "#8ea2b6",
        "menu_bg": "#ffffff",
        "menu_border": "#b8c6d6",
        "marker_line": "#ffffff",
        "marker_size": 10,
        "marker_opacity": 0.86,
        "accent_a": "rgba(27, 164, 176, 0.14)",
        "accent_b": "rgba(244, 133, 42, 0.10)",
        "colorway": [
            "#1f77b4",
            "#17a398",
            "#f28e2b",
            "#e15759",
            "#59a14f",
            "#edc949",
            "#4e79a7",
            "#76b7b2",
            "#ff9d52",
            "#9c755f",
        ],
    },
    "sunrise": {
        "paper_bg": "#fff8f1",
        "plot_bg": "#fffdf8",
        "font_color": "#4b2a1f",
        "grid": "#f0ddcc",
        "axis": "#c09c7f",
        "menu_bg": "#fffdf8",
        "menu_border": "#dfc0a6",
        "marker_line": "#fff8f1",
        "marker_size": 10,
        "marker_opacity": 0.84,
        "accent_a": "rgba(255, 167, 38, 0.16)",
        "accent_b": "rgba(0, 148, 136, 0.10)",
        "colorway": [
            "#e76f51",
            "#2a9d8f",
            "#f4a261",
            "#264653",
            "#8ab17d",
            "#e9c46a",
            "#4c956c",
            "#f08a5d",
            "#3d5a80",
            "#bc6c25",
        ],
    },
    "paper": {
        "paper_bg": "#fcfcfb",
        "plot_bg": "#ffffff",
        "font_color": "#222222",
        "grid": "#e1e3e5",
        "axis": "#a7adb3",
        "menu_bg": "#ffffff",
        "menu_border": "#c7ccd1",
        "marker_line": "#ffffff",
        "marker_size": 9,
        "marker_opacity": 0.82,
        "accent_a": "rgba(52, 152, 219, 0.10)",
        "accent_b": "rgba(39, 174, 96, 0.08)",
        "colorway": [
            "#1b6ca8",
            "#1f9d8a",
            "#e67e22",
            "#c0392b",
            "#2e7d32",
            "#c8a200",
            "#5d6d7e",
            "#2f855a",
            "#d35400",
            "#8d6e63",
        ],
    },
}


@dataclass
class InputContext:
    mode: str
    store_kind: str | None
    sample_ids: list[str]
    sample_paths: list[str]
    matrix_key: str
    matrix_path: str
    source: object
    out_columns: dict[str, list[str]]
    raw_shape: tuple[int, ...] | None = None


# ----------------------------
# Logging
# ----------------------------
def setup_logging(outdir: str, verbose: bool) -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("pca_umap")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(outdir, "pca_umap.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ----------------------------
# Basic IO
# ----------------------------
def read_lines(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def load_samples_from_merged_folder(merged_dir: str) -> tuple[list[str], list[str]]:
    tsv = os.path.join(merged_dir, "columns.tsv")
    txt = os.path.join(merged_dir, "columns.txt")

    if os.path.isfile(tsv):
        df = pd.read_csv(tsv, sep="\t").sort_values("col_idx")
        paths = df["mdb_dir"].astype(str).tolist()
    else:
        paths = read_lines(txt)

    ids = [os.path.basename(p.rstrip("/")) for p in paths]
    return paths, ids


def pick_matrix_path(merged_dir: str) -> tuple[str, str]:
    npys = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(merged_dir, "*.npy"))}
    for k in ("modifiedC", "5mC", "5hmC"):
        if k in npys:
            return k, npys[k]
    raise FileNotFoundError("No modifiedC.npy / 5mC.npy / 5hmC.npy found in merged folder")


class LegacyNpyMatrixSource:
    def __init__(self, matrix_path: str, n_samples: int):
        raw = np.load(matrix_path, mmap_mode="r")
        if raw.shape[1] == n_samples:
            X = raw
        elif raw.shape[0] == n_samples:
            X = raw.T
        else:
            raise ValueError(f"Sample count mismatch: n_samples={n_samples}, npy_shape={raw.shape}")
        self.raw = raw
        self.X = X
        self.shape = X.shape

    def iter_blocks(self, batch_rows: int):
        n_rows = int(self.shape[0])
        for start in range(0, n_rows, batch_rows):
            yield np.asarray(self.X[start : min(start + batch_rows, n_rows), :], dtype=np.float32)

    def read_rows(self, row_idx: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[row_idx, :], dtype=np.float32)

    def close(self) -> None:
        return None


class ReaderMatrixSource:
    def __init__(self, reader):
        self.reader = reader
        self.shape = reader.shape

    def iter_blocks(self, batch_rows: int):
        yield from self.reader.iter_blocks(batch_rows)

    def read_rows(self, row_idx: np.ndarray) -> np.ndarray:
        return self.reader.read_rows(row_idx)

    def close(self) -> None:
        self.reader.close()


def build_legacy_input_context(input_path: str) -> InputContext:
    sample_paths, sample_ids = load_samples_from_merged_folder(input_path)
    matrix_key, matrix_path = pick_matrix_path(input_path)
    source = LegacyNpyMatrixSource(matrix_path, n_samples=len(sample_ids))
    return InputContext(
        mode="legacy_npy",
        store_kind=None,
        sample_ids=sample_ids,
        sample_paths=sample_paths,
        matrix_key=matrix_key,
        matrix_path=matrix_path,
        source=source,
        out_columns={},
        raw_shape=tuple(source.raw.shape),
    )


def build_cohort_input_context(input_path: str, args) -> InputContext:
    key = TrackKey(assay=args.assay, haplotype=args.haplotype, strand=args.strand)
    reader, columns, matrix_ref = load_view_reader(input_path, key)
    source = ReaderMatrixSource(reader)

    out_columns = {
        "sample_id": list(columns.get("sample_id", [])),
        "bundle_path": list(columns.get("bundle_path", [])),
        "platform": list(columns.get("platform", [])),
        "source_path": list(columns.get("source_path", [])),
        "input_tag": list(columns.get("input_tag", [])),
    }
    sample_ids = out_columns["sample_id"]
    sample_paths = out_columns["bundle_path"]
    if not sample_ids:
        raise ValueError(f"No columns found for track {key.name()} in cohort store: {input_path}")

    return InputContext(
        mode="cohort_view",
        store_kind=detect_store_kind(input_path),
        sample_ids=sample_ids,
        sample_paths=sample_paths,
        matrix_key=key.name(),
        matrix_path=matrix_ref,
        source=source,
        out_columns=out_columns,
        raw_shape=None,
    )


def load_input_context(input_path: str, args) -> InputContext:
    try:
        kind = detect_store_kind(input_path)
    except Exception:
        kind = None

    if kind in {"cohort_store_npy", "cohort_store_zarr"}:
        return build_cohort_input_context(input_path, args)
    if kind == "sample_store_npy":
        raise ValueError("mdb pca expects a cohort store (.mmdb) or legacy merged .npy folder, not a sample bundle (.smdb).")
    return build_legacy_input_context(input_path)


# ----------------------------
# Streaming helpers (fast + simple)
# ----------------------------
def choose_rows_by_presence(source, min_frac_present=0.8, batch_rows=200_000) -> np.ndarray:
    n_cpgs = int(source.shape[0])
    if float(min_frac_present) <= 0:
        return np.arange(n_cpgs, dtype=np.int64)
    keep = np.zeros(n_cpgs, dtype=bool)
    cursor = 0
    for block in source.iter_blocks(batch_rows):
        n_rows = int(block.shape[0])
        keep[cursor : cursor + n_rows] = (np.mean(~np.isnan(block), axis=1) >= min_frac_present)
        cursor += n_rows
    if cursor != n_cpgs:
        raise RuntimeError(f"Row scan mismatch: scanned {cursor} rows, expected {n_cpgs}")
    return np.flatnonzero(keep)


def subsample_rows(rows: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return rows
    rng = np.random.default_rng(seed)
    k = max(int(len(rows) * frac), 1)
    return np.sort(rng.choice(rows, size=k, replace=False)).astype(np.int64)


def streaming_mean_std(source, row_idx: np.ndarray, batch_rows=200_000):
    """
    Welford per-sample mean/std over selected CpG rows, ignoring NaNs.
    """
    n_samples = int(source.shape[1])
    count = np.zeros(n_samples, dtype=np.float64)
    sum_vals = np.zeros(n_samples, dtype=np.float64)
    sum_sq = np.zeros(n_samples, dtype=np.float64)

    for start in tqdm(range(0, len(row_idx), batch_rows), desc="mean/std", leave=False):
        rows = row_idx[start : start + batch_rows]
        block = np.asarray(source.read_rows(rows), dtype=np.float32)
        mask = ~np.isnan(block)
        safe = np.where(mask, block, 0.0).astype(np.float64, copy=False)
        count += mask.sum(axis=0)
        sum_vals += safe.sum(axis=0)
        sum_sq += (safe * safe).sum(axis=0)

    mean = np.zeros(n_samples, dtype=np.float64)
    var = np.zeros(n_samples, dtype=np.float64)
    ok = count > 1
    mean[ok] = sum_vals[ok] / count[ok]
    var_num = sum_sq[ok] - (sum_vals[ok] * sum_vals[ok]) / count[ok]
    var[ok] = np.maximum(var_num / (count[ok] - 1.0), 0.0)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    mean[~ok] = 0.0
    std[~ok] = 1.0

    return mean.astype(np.float32), std.astype(np.float32), count.astype(np.int64)


def fit_ipca_sample_coords(
    source,
    n_components=50,
    frac_cpgs=0.1,
    min_frac_present=0.8,
    batch_rows=200_000,
    seed=1,
    logger=None,
):
    """
    Fit IPCA on source (n_cpgs x n_samples) and return SAMPLE coordinates.
    """
    n_rows = int(source.shape[0])
    if float(min_frac_present) <= 0 and float(frac_cpgs) < 1.0:
        rng = np.random.default_rng(seed)
        k = max(int(n_rows * float(frac_cpgs)), 1)
        row_idx = np.sort(rng.choice(n_rows, size=k, replace=False)).astype(np.int64)
        eligible_count = n_rows
    else:
        rows_ok = choose_rows_by_presence(source, min_frac_present=min_frac_present, batch_rows=batch_rows)
        row_idx = subsample_rows(rows_ok, frac=frac_cpgs, seed=seed)
        eligible_count = int(len(rows_ok))

    if logger:
        logger.info(f"Eligible CpGs: {eligible_count}; using for fit: {len(row_idx)} (frac_cpgs={frac_cpgs})")

    mean, std, obs_count = streaming_mean_std(source, row_idx, batch_rows=batch_rows)

    ipca = IncrementalPCA(n_components=n_components)
    for start in tqdm(range(0, len(row_idx), batch_rows), desc="ipca", leave=False):
        rows = row_idx[start : start + batch_rows]
        block = np.asarray(source.read_rows(rows), dtype=np.float32)
        block = (block - mean[None, :]) / std[None, :]
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        ipca.partial_fit(block)

    # sample coords: components^T scaled by singular values if present
    if hasattr(ipca, "singular_values_") and ipca.singular_values_ is not None:
        sample_coords = (ipca.components_.T * ipca.singular_values_[None, :]).astype(np.float32)
    else:
        sample_coords = (ipca.components_.T * np.sqrt(ipca.explained_variance_)[None, :]).astype(np.float32)

    return sample_coords, ipca, row_idx, obs_count


# ----------------------------
# Metadata + plotting
# ----------------------------
def maybe_merge_metadata(out: pd.DataFrame, metadata_path: str | None, logger=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not metadata_path or not os.path.isfile(metadata_path):
        return out, None

    meta = pd.read_csv(metadata_path, sep=None, engine="python")

    if "id" in meta.columns:
        out2 = out.copy()
        out2["id"] = out2["id"].astype(str)
        meta2 = meta.copy()
        meta2["id"] = meta2["id"].astype(str)
        out2 = out2.merge(meta2, on="id", how="left")
        return out2, meta

    if len(meta) == len(out):
        meta2 = meta.copy()
        meta2.index = out.index
        return pd.concat([out, meta2], axis=1), meta

    if logger:
        logger.info(f"Metadata not alignable; ignoring (meta_rows={len(meta)} vs n_samples={len(out)})")
    return out, None


def maybe_use_concise_sample_ids(out: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple sample-id-like columns exist (e.g. sample_id_x/sample_id_y),
    pick the shortest unique non-empty one for display id.
    """
    if out.empty or "id" not in out.columns:
        return out

    id_vals = out["id"].astype(str)
    candidate_cols: list[str] = []

    preferred = ["sample_id", "sample_id_y", "sample_name", "sample"]
    for col in preferred:
        if col in out.columns and col not in candidate_cols:
            candidate_cols.append(col)
    for col in out.columns:
        if col.startswith("sample_id") and col not in candidate_cols:
            candidate_cols.append(col)

    best_col = None
    best_mean_len = None
    for col in candidate_cols:
        raw = out[col]
        if raw.isna().any():
            continue
        vals = raw.astype(str).str.strip()
        if (vals == "").any():
            continue
        if vals.nunique(dropna=False) != len(out):
            continue
        if vals.equals(id_vals):
            continue
        mean_len = float(vals.str.len().mean())
        if best_col is None or mean_len < best_mean_len:
            best_col = col
            best_mean_len = mean_len

    if best_col is None:
        return out

    labels = out[best_col].astype(str).str.strip()
    out2 = out.copy()
    out2["id_original"] = out2["id"].astype(str)
    out2["id"] = labels
    if "sample_id" not in out2.columns:
        out2["sample_id"] = labels
    return out2


def plotly_color_options(meta: pd.DataFrame | None, out: pd.DataFrame, n_pcs: int, did_umap: bool) -> tuple[list[str], list[str]]:
    hover_priority = [
        "id",
        "sample_id",
        "donor",
        "tissue_name",
        "sex",
        "age_years",
        "center",
        "technology",
        "preservation",
        "core",
    ]
    hover_cols = [c for c in hover_priority if c in out.columns]
    if meta is None:
        return [], hover_cols

    banned = {"matrix_key", "matrix_path", "n_obs_cpgs_for_fit"} | {f"PC{i+1}" for i in range(n_pcs)}
    if did_umap:
        banned |= {"UMAP1", "UMAP2"}

    opts = [c for c in meta.columns if c in out.columns and c not in banned]
    return opts, hover_cols


def build_color_styles(df: pd.DataFrame, color_cols: list[str]) -> dict[str, dict[str, object]]:
    palette = (
        px.colors.qualitative.Safe
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Plotly
        + px.colors.qualitative.Dark24
    )
    styles: dict[str, dict[str, object]] = {}
    for col in color_cols:
        if col not in df.columns:
            continue
        vals = df[col].astype(str).fillna("NA")
        ordered = sorted(vals.unique().tolist())
        cmap = {v: palette[i % len(palette)] for i, v in enumerate(ordered)}
        styles[col] = {"ordered": ordered, "cmap": cmap}
    return styles


def resolve_plot_styles(args) -> list[str]:
    primary = str(getattr(args, "plot_style", "studio"))
    if primary not in PLOT_STYLE_PRESETS:
        primary = "studio"
    styles = [primary]
    if bool(getattr(args, "plot_style_variants", False)):
        for name in PLOT_STYLE_PRESETS:
            if name != primary:
                styles.append(name)
    return styles


def apply_style_to_figure(fig: go.Figure, style_name: str, with_dropdown: bool) -> None:
    style = PLOT_STYLE_PRESETS.get(style_name, PLOT_STYLE_PRESETS["studio"])
    right_margin = 285 if with_dropdown else 95
    top_margin = 136 if with_dropdown else 96
    fig.update_layout(
        template="none",
        width=1120,
        height=860,
        paper_bgcolor=style["paper_bg"],
        plot_bgcolor=style["plot_bg"],
        colorway=style["colorway"],
        font=dict(family="Source Sans Pro, Arial, sans-serif", size=14, color=style["font_color"]),
        title=dict(x=0.01, y=0.98, xanchor="left", yanchor="top", font=dict(size=22, color=style["font_color"])),
        margin=dict(l=72, r=right_margin, t=top_margin, b=70),
        legend=dict(
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor=style["menu_border"],
            borderwidth=1,
            x=1.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=11, color=style["font_color"]),
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.97)",
            bordercolor=style["menu_border"],
            font=dict(size=12, color=style["font_color"]),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=style["grid"],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=style["axis"],
        linewidth=1.2,
        ticks="outside",
        tickcolor=style["axis"],
        title=dict(font=dict(size=14, color=style["font_color"])),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=style["grid"],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=style["axis"],
        linewidth=1.2,
        ticks="outside",
        tickcolor=style["axis"],
        title=dict(font=dict(size=14, color=style["font_color"])),
    )
    fig.update_traces(
        selector=dict(type="scatter"),
        marker=dict(
            size=style["marker_size"],
            opacity=style["marker_opacity"],
            line=dict(color=style["marker_line"], width=0.7),
        ),
    )
    if fig.layout.updatemenus:
        styled_menus = []
        for menu in fig.layout.updatemenus:
            menu2 = menu.to_plotly_json()
            menu2["bgcolor"] = style["menu_bg"]
            menu2["bordercolor"] = style["menu_border"]
            menu2["borderwidth"] = 1
            menu2["font"] = dict(size=12, color=style["font_color"])
            menu2["pad"] = dict(l=6, r=6, t=6, b=6)
            if with_dropdown:
                menu2["x"] = 0.01
                menu2["xanchor"] = "left"
                menu2["y"] = 1.15
                menu2["yanchor"] = "top"
                menu2["direction"] = "down"
            styled_menus.append(menu2)
        fig.update_layout(updatemenus=styled_menus)
        if with_dropdown:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.17,
                text="<b>Color By</b>",
                showarrow=False,
                font=dict(size=12, color=style["font_color"]),
                align="left",
            )


def make_dropdown_scatter(
    df,
    x,
    y,
    color_cols,
    hover_cols,
    title,
    color_styles: dict[str, dict[str, object]] | None = None,
    symbol_col: str | None = None,
    symbol_map: dict[str, str] | None = None,
    style_name: str = "studio",
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
):
    scatter_kwargs: dict[str, object] = {}
    if symbol_col and symbol_col in df.columns:
        scatter_kwargs["symbol"] = symbol_col
        if symbol_map:
            scatter_kwargs["symbol_map"] = symbol_map

    if not color_cols:
        fig = px.scatter(df, x=x, y=y, hover_data=hover_cols, title=title, **scatter_kwargs)
        apply_style_to_figure(fig, style_name=style_name, with_dropdown=False)
        fig.update_xaxes(title_text=x_axis_label or x)
        fig.update_yaxes(title_text=y_axis_label or y)
        return fig

    master = go.Figure()
    groups = []
    for i, col in enumerate(color_cols):
        if col not in df.columns:
            continue
        tmp_df = df.copy()
        tmp_df[col] = tmp_df[col].astype(str)
        scatter_args: dict[str, object] = dict(scatter_kwargs)
        style = (color_styles or {}).get(col, {})
        if "ordered" in style:
            scatter_args["category_orders"] = {col: style["ordered"]}
        if "cmap" in style:
            scatter_args["color_discrete_map"] = style["cmap"]
        tmp = px.scatter(
            tmp_df,
            x=x,
            y=y,
            color=col,
            hover_data=hover_cols,
            title=f"{title} (color_by={col})",
            **scatter_args,
        )
        start = len(master.data)
        for tr in tmp.data:
            tr.visible = (i == 0)
            master.add_trace(tr)
        end = len(master.data)
        groups.append((col, start, end))

    if not groups:
        fig = px.scatter(df, x=x, y=y, hover_data=hover_cols, title=title, **scatter_kwargs)
        apply_style_to_figure(fig, style_name=style_name, with_dropdown=False)
        fig.update_xaxes(title_text=x_axis_label or x)
        fig.update_yaxes(title_text=y_axis_label or y)
        return fig

    buttons = []
    n_tr = len(master.data)
    for col, s, e in groups:
        vis = [False] * n_tr
        for j in range(s, e):
            vis[j] = True
        buttons.append(dict(label=col, method="update", args=[{"visible": vis}, {"title": f"{title} (color_by={col})"}]))

    master.update_layout(updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.02, xanchor="left", y=1.0, yanchor="top")])
    apply_style_to_figure(master, style_name=style_name, with_dropdown=True)
    master.update_xaxes(title_text=x_axis_label or x)
    master.update_yaxes(title_text=y_axis_label or y)
    return master


def write_pairplot_png(
    scores: np.ndarray,
    manifest: pd.DataFrame,
    out_png: str,
    pcs: tuple[str, ...],
    hue: str | None = None,
    diag_kind="kde",
    corner=False,
    pc_label_map: dict[str, str] | None = None,
    title: str | None = None,
):
    # Build a clean df: (manifest without PC*) + (PCs from scores)
    scores = np.asarray(scores)
    pc_cols_all = [f"PC{i+1}" for i in range(scores.shape[1])]
    man = manifest.reset_index(drop=True).copy()
    man = man.drop(columns=[c for c in man.columns if c in pc_cols_all], errors="ignore")
    pca_df = pd.concat([man, pd.DataFrame(scores, columns=pc_cols_all)], axis=1)

    cols = list(pcs) + ([hue] if hue else [])
    df_plot = pca_df[cols].copy()
    for pc in pcs:
        df_plot[pc] = pd.to_numeric(df_plot[pc], errors="coerce")
    vars_plot = list(pcs)
    if pc_label_map:
        rename_map = {pc: pc_label_map.get(pc, pc) for pc in pcs}
        df_plot = df_plot.rename(columns=rename_map)
        vars_plot = [rename_map[pc] for pc in pcs]

    hue_palette = None
    hue_order: list[str] = []
    if hue:
        hue_vals = df_plot[hue].astype(str).fillna("NA")
        hue_order = sorted(hue_vals.unique().tolist())
        df_plot[hue] = pd.Categorical(hue_vals, categories=hue_order, ordered=True)
        palette = sns.color_palette("tab20", n_colors=max(len(hue_order), 3))
        hue_palette = {cat: palette[i % len(palette)] for i, cat in enumerate(hue_order)}

    sns.set_theme(style="whitegrid", context="notebook")
    plot_kws = dict(s=16, alpha=0.72, linewidth=0.25, edgecolor="white")
    diag_kws = dict(bins=30, edgecolor="white", linewidth=0.4) if diag_kind == "hist" else {}
    g = sns.pairplot(
        df_plot,
        vars=vars_plot,
        hue=hue,
        palette=hue_palette,
        diag_kind=diag_kind,
        corner=corner,
        plot_kws=plot_kws,
        diag_kws=diag_kws,
    )
    side = max(8.5, 2.35 * len(vars_plot))
    g.fig.set_size_inches(side, side)
    for ax in g.fig.axes:
        if ax is None:
            continue
        ax.grid(True, color="#d9dee4", linewidth=0.7)
        ax.tick_params(labelsize=9)
    legend_obj = getattr(g, "_legend", None)
    legend_below = False
    legend_rows = 1
    if legend_obj is not None:
        n_levels = len(hue_order)
        legend_title = hue if hue else ""
        if n_levels > 12:
            legend_below = True
            ncol = min(6, max(3, math.ceil(n_levels / 5)))
            legend_rows = max(1, math.ceil(n_levels / ncol))
            try:
                sns.move_legend(
                    g,
                    "lower center",
                    bbox_to_anchor=(0.5, -0.03),
                    ncol=ncol,
                    frameon=False,
                    title=legend_title,
                )
            except Exception:
                legend_obj.set_bbox_to_anchor((0.5, -0.03))
                legend_obj._loc = 9  # upper center
                legend_obj.set_title(legend_title)
                legend_obj.set_frame_on(False)
                if hasattr(legend_obj, "set_ncols"):
                    legend_obj.set_ncols(ncol)
                else:
                    legend_obj._ncols = ncol
        else:
            try:
                sns.move_legend(
                    g,
                    "upper left",
                    bbox_to_anchor=(1.01, 0.99),
                    frameon=False,
                    title=legend_title,
                )
            except Exception:
                legend_obj.set_bbox_to_anchor((1.01, 0.99))
                legend_obj._loc = 2  # upper left
                legend_obj.set_title(legend_title)
                legend_obj.set_frame_on(False)
        legend_obj = getattr(g, "_legend", None)
        if legend_obj is not None:
            for txt in legend_obj.texts:
                txt.set_fontsize(8)
            if legend_obj.get_title() is not None:
                legend_obj.get_title().set_fontsize(9)
    if title:
        g.fig.suptitle(title, y=1.0, fontsize=12)
    if legend_below:
        bottom_pad = min(0.34, 0.05 + 0.032 * legend_rows)
        g.fig.tight_layout(rect=(0.0, bottom_pad, 1.0, 0.95))
    else:
        g.fig.tight_layout(rect=(0.0, 0.0, 0.84, 0.95))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    g.fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(g.fig)


def pick_pairplot_hue(out: pd.DataFrame, meta: pd.DataFrame | None, args) -> str | None:
    mode = getattr(args, "pairplot_mode", None)
    if mode is None:
        mode = "metadata" if meta is not None else "sample"
    if mode == "none":
        return None
    if mode == "sample":
        return "id"
    if mode == "metadata":
        if meta is None:
            return "id"
        preferred = getattr(args, "pairplot_hue", None)
        if preferred and preferred in out.columns:
            return preferred
        for c in meta.columns:
            if c in out.columns and c not in {"id", "path", "matrix_key", "matrix_path", "n_obs_cpgs_for_fit"}:
                return c
        return "id"
    return None


def plotly_png_ok() -> bool:
    try:
        import kaleido  # noqa: F401
        return True
    except Exception:
        return False


def write_plotly_image_safe(fig, path: str, logger: logging.Logger) -> bool:
    try:
        fig.write_image(path)
        return True
    except Exception as exc:
        logger.warning(f"Skipping plotly PNG export for {path}: {exc}")
        return False


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


def detect_outliers_mahalanobis(
    out: pd.DataFrame,
    outlier_n_pcs: int,
    outlier_alpha: float,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    alpha = float(outlier_alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"--outlier_alpha must be in (0,1); got {outlier_alpha}")

    n_use = max(int(outlier_n_pcs), 1)
    pc_cols = [f"PC{i+1}" for i in range(n_use) if f"PC{i+1}" in out.columns]
    if not pc_cols:
        out2 = out.copy()
        out2["mahalanobis_pc"] = np.nan
        out2["is_outlier"] = False
        out2["outlier_status"] = "inlier"
        info = {
            "enabled": True,
            "method": "mahalanobis",
            "alpha": alpha,
            "cutoff": None,
            "pc_cols": [],
            "outlier_count": 0,
            "reason": "no_pc_columns",
        }
        return out2, info

    out2 = out.copy()
    X = out2[pc_cols].to_numpy(dtype=np.float64, copy=True)
    finite_rows = np.isfinite(X).all(axis=1)

    out2["mahalanobis_pc"] = np.nan
    out2["is_outlier"] = False
    out2["outlier_status"] = "inlier"

    if int(finite_rows.sum()) < 3:
        info = {
            "enabled": True,
            "method": "mahalanobis",
            "alpha": alpha,
            "cutoff": None,
            "pc_cols": pc_cols,
            "outlier_count": 0,
            "reason": "insufficient_finite_rows",
        }
        if logger:
            logger.warning("Outlier detection skipped: insufficient finite rows")
        return out2, info

    Xf = X[finite_rows]
    mu = Xf.mean(axis=0)
    cov = np.atleast_2d(np.cov(Xf, rowvar=False))
    inv_cov = np.linalg.pinv(cov)
    d = Xf - mu
    md2 = np.einsum("ij,jk,ik->i", d, inv_cov, d)

    cutoff = float(chi2.ppf(alpha, df=len(pc_cols)))
    out_mask = md2 > cutoff

    md_full = np.full(shape=(len(out2),), fill_value=np.nan, dtype=np.float64)
    mask_full = np.zeros(shape=(len(out2),), dtype=bool)
    md_full[finite_rows] = md2
    mask_full[finite_rows] = out_mask

    out2["mahalanobis_pc"] = md_full
    out2["is_outlier"] = mask_full
    out2["outlier_status"] = np.where(mask_full, "outlier", "inlier")

    info = {
        "enabled": True,
        "method": "mahalanobis",
        "alpha": alpha,
        "cutoff": cutoff,
        "pc_cols": pc_cols,
        "outlier_count": int(mask_full.sum()),
        "finite_rows": int(finite_rows.sum()),
    }
    if logger:
        logger.info(
            f"Outlier detection: method=mahalanobis, alpha={alpha}, "
            f"pc_cols={len(pc_cols)}, outliers={int(mask_full.sum())}/{len(out2)}"
        )
    return out2, info


def write_outlier_artifacts(
    out: pd.DataFrame,
    sample_coords: np.ndarray,
    color_cols: list[str],
    hover_cols: list[str],
    color_styles: dict[str, dict[str, object]],
    hue: str | None,
    args,
    outdir: str,
    matrix_key: str,
    x_axis_label: str,
    y_axis_label: str,
    pairplot_pc_label_map: dict[str, str] | None,
    logger: logging.Logger,
    png_ok: bool,
) -> None:
    if "is_outlier" not in out.columns:
        return

    outlier_report = os.path.join(outdir, "outlier_report.tsv")
    outliers_only = os.path.join(outdir, "outliers_only.tsv")
    out.to_csv(outlier_report, sep="\t", index=False)
    out.loc[out["is_outlier"]].sort_values("mahalanobis_pc", ascending=False).to_csv(outliers_only, sep="\t", index=False)

    marked_hover = list(hover_cols)
    for c in ("outlier_status", "mahalanobis_pc", "is_outlier"):
        if c in out.columns and c not in marked_hover:
            marked_hover.append(c)

    marked_color = ["outlier_status"] + [c for c in color_cols if c != "outlier_status" and c in out.columns]
    marked_styles = dict(color_styles)
    marked_styles["outlier_status"] = {
        "ordered": ["inlier", "outlier"],
        "cmap": {"inlier": "#94a3b8", "outlier": "#dc2626"},
    }
    write_scatter_with_styles(
        df=out,
        x="PC1",
        y="PC2",
        color_cols=marked_color,
        hover_cols=marked_hover,
        title=f"PCA sample coords (PC1 vs PC2) [{matrix_key}] with outliers marked",
        color_styles=marked_styles,
        out_html=os.path.join(outdir, "pca_with_outliers_marked.html"),
        args=args,
        logger=logger,
        png_ok=png_ok,
        out_png=os.path.join(outdir, "pca_with_outliers_marked.png"),
        symbol_col="outlier_status",
        symbol_map={"inlier": "circle", "outlier": "x"},
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    inlier_mask = ~out["is_outlier"].to_numpy(dtype=bool)
    inlier_df = out.loc[inlier_mask].copy()
    if inlier_df.empty:
        logger.warning("Skipping no-outlier plots: no inlier samples")
        return
    inlier_color = [c for c in color_cols if c in inlier_df.columns]
    write_scatter_with_styles(
        df=inlier_df,
        x="PC1",
        y="PC2",
        color_cols=inlier_color,
        hover_cols=hover_cols,
        title=f"PCA sample coords (PC1 vs PC2) [{matrix_key}] no outliers",
        color_styles=color_styles,
        out_html=os.path.join(outdir, "pca_no_outliers.html"),
        args=args,
        logger=logger,
        png_ok=png_ok,
        out_png=os.path.join(outdir, "pca_no_outliers.png"),
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    n_pair = max(1, min(int(args.pairplot_pcs_n), int(sample_coords.shape[1])))
    pcs = tuple(f"PC{i}" for i in range(1, n_pair + 1))
    if int(inlier_mask.sum()) >= 2:
        manifest_no_outlier = inlier_df.drop(columns=[f"PC{i+1}" for i in range(int(sample_coords.shape[1]))], errors="ignore")
        hue2 = hue if hue in manifest_no_outlier.columns else None
        write_pairplot_png(
            scores=np.asarray(sample_coords[inlier_mask, :], dtype=np.float32),
            manifest=manifest_no_outlier,
            out_png=os.path.join(outdir, "pca_pairplot_no_outliers.png"),
            pcs=pcs,
            hue=hue2,
            diag_kind=getattr(args, "pairplot_diag_kind", "kde"),
            corner=bool(getattr(args, "pairplot_corner", False)),
            pc_label_map=pairplot_pc_label_map,
            title=f"PCA Pairplot [{matrix_key}] (outliers removed)",
        )
    else:
        logger.warning("Skipping pca_pairplot_no_outliers.png: fewer than 2 inlier samples")


# ----------------------------
# MAIN entry (submodule)
# ----------------------------
def pca_main(args):
    input_path = args.input
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    logger = setup_logging(outdir, args.verbose)

    t0 = time.time()
    logger.info("==== Run started ====")
    logger.info(f"Start time: {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"Input: {input_path}")

    png_ok = plotly_png_ok()
    logger.info(f"Plotly PNG export: {'available' if png_ok else 'NOT available (HTML only)'}")

    ctx = load_input_context(input_path, args)
    logger.info(
        f"Input mode: {ctx.mode} | samples={len(ctx.sample_ids)} "
        f"| matrix={ctx.matrix_key} | ref={ctx.matrix_path}"
    )
    logger.info(f"Matrix shape used: {ctx.source.shape} (n_cpgs x n_samples)")

    try:
        sample_coords, ipca, row_idx, obs_count = fit_ipca_sample_coords(
            ctx.source,
            n_components=args.n_pcs,
            frac_cpgs=args.frac_cpgs,
            min_frac_present=args.min_frac_present,
            batch_rows=args.batch_rows,
            seed=args.seed,
            logger=logger,
        )

        out = pd.DataFrame({"id": ctx.sample_ids, "path": ctx.sample_paths})
        for col_name, values in ctx.out_columns.items():
            if values and len(values) == len(out):
                out[col_name] = values
        out["matrix_key"] = ctx.matrix_key
        out["matrix_path"] = ctx.matrix_path
        out["n_obs_cpgs_for_fit"] = obs_count
        for i in range(args.n_pcs):
            out[f"PC{i+1}"] = sample_coords[:, i]

        did_umap = False
        if bool(getattr(args, "umap", False)):
            reducer = umap.UMAP(
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                random_state=args.seed,
            )
            emb = reducer.fit_transform(sample_coords).astype(np.float32)
            out["UMAP1"] = emb[:, 0]
            out["UMAP2"] = emb[:, 1]
            did_umap = True

        out, meta = maybe_merge_metadata(out, getattr(args, "metadata", None), logger=logger)
        out = maybe_use_concise_sample_ids(out)

        outlier_enabled = bool(getattr(args, "outlier_detect", False))
        outlier_info: dict[str, object] = {"enabled": outlier_enabled}
        if outlier_enabled:
            out, outlier_info = detect_outliers_mahalanobis(
                out=out,
                outlier_n_pcs=int(getattr(args, "outlier_n_pcs", 10)),
                outlier_alpha=float(getattr(args, "outlier_alpha", 0.999)),
                logger=logger,
            )

        out_tsv = os.path.join(outdir, "embedding.tsv")
        out.to_csv(out_tsv, sep="\t", index=False)

        params = vars(args).copy()
        params.update(
            dict(
                input=input_path,
                input_mode=ctx.mode,
                input_store_kind=ctx.store_kind,
                selected_matrix_key=ctx.matrix_key,
                selected_matrix_path=ctx.matrix_path,
                used_shape=tuple(ctx.source.shape),
                raw_shape=ctx.raw_shape,
                n_samples=int(len(ctx.sample_ids)),
                n_cpgs_used_for_fit=int(len(row_idx)),
                pca_semantics="IPCA on (n_cpgs x n_samples); PCs are sample coordinates",
                explained_variance_ratio=getattr(ipca, "explained_variance_ratio_", None).tolist()
                if getattr(ipca, "explained_variance_ratio_", None) is not None
                else None,
                umap_ran=bool(did_umap),
                plotly_png_supported=bool(png_ok),
                outlier_detection=outlier_info,
            )
        )
        with open(os.path.join(outdir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

        color_cols, hover_cols = plotly_color_options(meta, out, args.n_pcs, did_umap)
        color_styles = build_color_styles(out, color_cols)
        evr = getattr(ipca, "explained_variance_ratio_", None)
        if evr is not None and len(evr) >= 2:
            x_axis_label = f"PC1 ({float(evr[0]) * 100.0:.2f}% variance explained)"
            y_axis_label = f"PC2 ({float(evr[1]) * 100.0:.2f}% variance explained)"
        else:
            x_axis_label = "PC1"
            y_axis_label = "PC2"
        pairplot_pc_label_map: dict[str, str] = {}
        for i in range(int(args.pairplot_pcs_n)):
            pc = f"PC{i+1}"
            if evr is not None and i < len(evr):
                pairplot_pc_label_map[pc] = f"{pc} ({float(evr[i]) * 100.0:.2f}%)"
            else:
                pairplot_pc_label_map[pc] = pc

        write_scatter_with_styles(
            df=out,
            x="PC1",
            y="PC2",
            color_cols=color_cols,
            hover_cols=hover_cols,
            title=f"PCA sample coords (PC1 vs PC2) [{ctx.matrix_key}]",
            color_styles=color_styles,
            out_html=os.path.join(outdir, "pca.html"),
            args=args,
            logger=logger,
            png_ok=png_ok,
            out_png=os.path.join(outdir, "pca.png"),
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
        )

        if did_umap:
            write_scatter_with_styles(
                df=out,
                x="UMAP1",
                y="UMAP2",
                color_cols=color_cols,
                hover_cols=hover_cols,
                title=f"UMAP (on PCA coords) [{ctx.matrix_key}]",
                color_styles=color_styles,
                out_html=os.path.join(outdir, "umap.html"),
                args=args,
                logger=logger,
                png_ok=png_ok,
                out_png=os.path.join(outdir, "umap.png"),
                x_axis_label="UMAP1",
                y_axis_label="UMAP2",
            )

        # Pairplot
        n_pair = args.pairplot_pcs_n
        pcs = tuple(f"PC{i}" for i in range(1, n_pair + 1))
        hue = pick_pairplot_hue(out, meta, args)
        manifest_for_pairplot = out.drop(columns=[f"PC{i+1}" for i in range(args.n_pcs)], errors="ignore")
        write_pairplot_png(
            scores=sample_coords,
            manifest=manifest_for_pairplot,
            out_png=os.path.join(outdir, "pca_pairplot.png"),
            pcs=pcs,
            hue=hue,
            diag_kind=getattr(args, "pairplot_diag_kind", "kde"),
            corner=bool(getattr(args, "pairplot_corner", False)),
            pc_label_map=pairplot_pc_label_map,
            title=f"PCA Pairplot [{ctx.matrix_key}]",
        )

        if outlier_enabled:
            write_outlier_artifacts(
                out=out,
                sample_coords=sample_coords,
                color_cols=color_cols,
                hover_cols=hover_cols,
                color_styles=color_styles,
                hue=hue,
                args=args,
                outdir=outdir,
                matrix_key=ctx.matrix_key,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                pairplot_pc_label_map=pairplot_pc_label_map,
                logger=logger,
                png_ok=png_ok,
            )
    finally:
        ctx.source.close()

    logger.info(f"==== Done in {time.time() - t0:.2f}s ====")
