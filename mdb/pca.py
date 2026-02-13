#!/usr/bin/env python3
"""
pca_umap_from_npy.py (SUBMODULE, simplified)

- Load merged .npy matrix (mmap), pick: modifiedC > 5mC > 5hmC
- Load sample order from columns.tsv (preferred) or columns.txt
- Fit IPCA on X (n_cpgs x n_samples) using CpGs as observations, samples as features
- Output sample coordinates as PC1..PCn (dual/loadings coords; good for embedding samples)
- Optional UMAP on those coordinates
- Write:
    embedding.tsv, params.json, pca_umap.log
    pca.html (+ pca.png if plotly image engine exists)
    umap.html (+ umap.png if plotly image engine exists, and args.umap)
    pca_pairplot.png (always)

Minimal dependencies:
  numpy pandas scikit-learn umap-learn plotly tqdm seaborn matplotlib
"""

import os
import sys
import json
import time
import glob
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.decomposition import IncrementalPCA
import umap

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt


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


def load_mmap_matrix_oriented(matrix_path: str, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(matrix_path, mmap_mode="r")
    if X.shape[1] == n_samples:
        return X, X
    if X.shape[0] == n_samples:
        return X, X.T
    raise ValueError(f"Sample count mismatch: n_samples={n_samples}, npy_shape={X.shape}")


# ----------------------------
# Streaming helpers (fast + simple)
# ----------------------------
def choose_rows_by_presence(X, min_frac_present=0.8, batch_rows=200_000) -> np.ndarray:
    n_cpgs = X.shape[0]
    keep = np.zeros(n_cpgs, dtype=bool)
    for start in range(0, n_cpgs, batch_rows):
        rows = slice(start, min(start + batch_rows, n_cpgs))
        block = np.asarray(X[rows, :], dtype=np.float32)
        keep[rows] = (np.mean(~np.isnan(block), axis=1) >= min_frac_present)
    return np.where(keep)[0]


def subsample_rows(rows: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return rows
    rng = np.random.default_rng(seed)
    k = max(int(len(rows) * frac), 1)
    return np.sort(rng.choice(rows, size=k, replace=False)).astype(np.int64)


def streaming_mean_std(X, row_idx: np.ndarray, batch_rows=200_000):
    """
    Welford per-sample mean/std over selected CpG rows, ignoring NaNs.
    """
    n_samples = X.shape[1]
    count = np.zeros(n_samples, dtype=np.float64)
    mean = np.zeros(n_samples, dtype=np.float64)
    M2 = np.zeros(n_samples, dtype=np.float64)

    for start in tqdm(range(0, len(row_idx), batch_rows), desc="mean/std", leave=False):
        rows = row_idx[start : start + batch_rows]
        block = np.asarray(X[rows, :], dtype=np.float32)
        mask = ~np.isnan(block)

        for r in range(block.shape[0]):
            m = mask[r]
            if not np.any(m):
                continue
            cols = np.where(m)[0]
            x = block[r, m].astype(np.float64)
            count[cols] += 1.0
            delta = x - mean[cols]
            mean[cols] += delta / count[cols]
            M2[cols] += delta * (x - mean[cols])

    var = np.zeros(n_samples, dtype=np.float64)
    ok = count > 1
    var[ok] = M2[ok] / (count[ok] - 1.0)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    mean[~ok] = 0.0
    std[~ok] = 1.0

    return mean.astype(np.float32), std.astype(np.float32), count.astype(np.int64)


def fit_ipca_sample_coords(
    X,
    n_components=50,
    frac_cpgs=0.1,
    min_frac_present=0.8,
    batch_rows=200_000,
    seed=1,
    logger=None,
):
    """
    Fit IPCA on X (n_cpgs x n_samples) and return SAMPLE coordinates (dual/loadings coords).
    """
    rows_ok = choose_rows_by_presence(X, min_frac_present=min_frac_present, batch_rows=batch_rows)
    row_idx = subsample_rows(rows_ok, frac=frac_cpgs, seed=seed)

    if logger:
        logger.info(f"Eligible CpGs: {len(rows_ok)}; using for fit: {len(row_idx)} (frac_cpgs={frac_cpgs})")

    mean, std, obs_count = streaming_mean_std(X, row_idx, batch_rows=batch_rows)

    ipca = IncrementalPCA(n_components=n_components)
    for start in tqdm(range(0, len(row_idx), batch_rows), desc="ipca", leave=False):
        rows = row_idx[start : start + batch_rows]
        block = np.asarray(X[rows, :], dtype=np.float32)
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


def plotly_color_options(meta: pd.DataFrame | None, out: pd.DataFrame, n_pcs: int, did_umap: bool) -> tuple[list[str], list[str]]:
    hover_cols = [c for c in ["id", "path", "matrix_key"] if c in out.columns]
    if meta is None:
        return [], hover_cols

    banned = {"matrix_key", "matrix_path", "n_obs_cpgs_for_fit"} | {f"PC{i+1}" for i in range(n_pcs)}
    if did_umap:
        banned |= {"UMAP1", "UMAP2"}

    opts = [c for c in meta.columns if c in out.columns and c not in banned]
    hover_cols2 = [c for c in (hover_cols + opts) if c in out.columns]
    return opts, hover_cols2


def make_dropdown_scatter(df, x, y, color_cols, hover_cols, title):
    if not color_cols:
        fig = px.scatter(df, x=x, y=y, hover_data=hover_cols, title=title)
        fig.update_layout(template="plotly_white", width=1000, height=800, margin=dict(l=60, r=60, t=80, b=60))
        return fig

    master = go.Figure()
    groups = []
    for i, col in enumerate(color_cols):
        tmp = px.scatter(df, x=x, y=y, color=df[col].astype(str), hover_data=hover_cols, title=f"{title} (color_by={col})")
        start = len(master.data)
        for tr in tmp.data:
            tr.visible = (i == 0)
            master.add_trace(tr)
        end = len(master.data)
        groups.append((col, start, end))

    buttons = []
    n_tr = len(master.data)
    for col, s, e in groups:
        vis = [False] * n_tr
        for j in range(s, e):
            vis[j] = True
        buttons.append(dict(label=col, method="update", args=[{"visible": vis}, {"title": f"{title} (color_by={col})"}]))

    master.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.02, xanchor="left", y=1.0, yanchor="top")],
        template="plotly_white",
        width=1000,
        height=800,
        margin=dict(l=60, r=240, t=80, b=60),
    )
    return master


def write_pairplot_png(scores: np.ndarray, manifest: pd.DataFrame, out_png: str, pcs: tuple[str, ...], hue: str | None = None, diag_kind="kde", corner=False):
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
    if hue:
        df_plot[hue] = df_plot[hue].astype(str)

    g = sns.pairplot(df_plot, vars=list(pcs), hue=hue, diag_kind=diag_kind, corner=corner, plot_kws=dict(s=10, alpha=0.9, linewidth=0))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    g.fig.tight_layout()
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


# ----------------------------
# MAIN entry (submodule)
# ----------------------------
def pca_main(args):
    merged_dir = args.input
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    logger = setup_logging(outdir, args.verbose)

    t0 = time.time()
    logger.info("==== Run started ====")
    logger.info(f"Start time: {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"Input merged folder: {merged_dir}")

    png_ok = plotly_png_ok()
    logger.info(f"Plotly PNG export: {'available' if png_ok else 'NOT available (HTML only)'}")

    sample_paths, sample_ids = load_samples_from_merged_folder(merged_dir)
    key, matrix_path = pick_matrix_path(merged_dir)
    logger.info(f"Samples: {len(sample_ids)} | matrix={key} | {matrix_path}")

    X_raw, X = load_mmap_matrix_oriented(matrix_path, n_samples=len(sample_ids))
    logger.info(f"Matrix shape used: {X.shape} (n_cpgs x n_samples)")

    sample_coords, ipca, row_idx, obs_count = fit_ipca_sample_coords(
        X,
        n_components=args.n_pcs,
        frac_cpgs=args.frac_cpgs,
        min_frac_present=args.min_frac_present,
        batch_rows=args.batch_rows,
        seed=args.seed,
        logger=logger,
    )

    out = pd.DataFrame({"id": sample_ids, "path": sample_paths})
    out["matrix_key"] = key
    out["matrix_path"] = matrix_path
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

    out_tsv = os.path.join(outdir, "embedding.tsv")
    out.to_csv(out_tsv, sep="\t", index=False)

    params = vars(args).copy()
    params.update(
        dict(
            input=merged_dir,
            selected_matrix_key=key,
            selected_matrix_path=matrix_path,
            npy_shape=tuple(X_raw.shape),
            used_shape=tuple(X.shape),
            n_samples=int(len(sample_ids)),
            n_cpgs_used_for_fit=int(len(row_idx)),
            pca_semantics="IPCA on (n_cpgs x n_samples); PCs are SAMPLE coords (dual/loadings)",
            explained_variance_ratio=getattr(ipca, "explained_variance_ratio_", None).tolist()
            if getattr(ipca, "explained_variance_ratio_", None) is not None
            else None,
            umap_ran=bool(did_umap),
            plotly_png_supported=bool(png_ok),
        )
    )
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    color_cols, hover_cols = plotly_color_options(meta, out, args.n_pcs, did_umap)

    pca_fig = make_dropdown_scatter(out, "PC1", "PC2", color_cols, hover_cols, f"PCA sample coords (PC1 vs PC2) [{key}]")
    pca_html = os.path.join(outdir, "pca.html")
    pca_fig.write_html(pca_html, include_plotlyjs="cdn")
    if png_ok:
        pca_fig.write_image(os.path.join(outdir, "pca.png"))

    if did_umap:
        umap_fig = make_dropdown_scatter(out, "UMAP1", "UMAP2", color_cols, hover_cols, f"UMAP (on PCA coords) [{key}]")
        umap_html = os.path.join(outdir, "umap.html")
        umap_fig.write_html(umap_html, include_plotlyjs="cdn")
        if png_ok:
            umap_fig.write_image(os.path.join(outdir, "umap.png"))

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
    )

    logger.info(f"==== Done in {time.time() - t0:.2f}s ====")
