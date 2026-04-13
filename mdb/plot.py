#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

from mdb.plotting import PLOT_STYLE_PRESETS
from mdb.schema import TrackKey
from mdb.storage import detect_store_kind, load_cohort_index, load_view_columns, load_view_reader
from mdb.viz import align_metadata, select_tracks_for_cohort

REGION_RE = re.compile(r"^([^:]+):([\d,]+)-([\d,]+)$")


def setup_logging(outdir: str, verbose: bool) -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("mdb_plot")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(outdir, "plot.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _json_script(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


def _serialize_int32(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype=np.int32).tobytes(order="C")).decode("ascii")


def _serialize_float32(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype=np.float32).tobytes(order="C")).decode("ascii")


def _parse_region(region: str) -> tuple[str, int, int]:
    match = REGION_RE.match(str(region).strip())
    if not match:
        raise ValueError("Region must look like chrom:start-end")
    chrom = match.group(1).strip()
    start_pos0 = int(match.group(2).replace(",", ""))
    end_pos0 = int(match.group(3).replace(",", ""))
    if not chrom:
        raise ValueError("Region chromosome cannot be empty")
    if start_pos0 < 0 or end_pos0 < 0:
        raise ValueError("Region coordinates must be >= 0")
    if end_pos0 < start_pos0:
        raise ValueError("Region end must be >= region start")
    return chrom, start_pos0, end_pos0


def _region_rows(
    chroms: list[str],
    chrom_offsets: np.ndarray,
    pos0: np.ndarray,
    chrom: str,
    start_pos0: int,
    end_pos0: int,
) -> tuple[np.ndarray, np.ndarray]:
    if chrom not in chroms:
        raise ValueError(f"Chromosome {chrom!r} was not found in the cohort index")
    chrom_idx = chroms.index(chrom)
    chrom_start = int(chrom_offsets[chrom_idx])
    chrom_end = int(chrom_offsets[chrom_idx + 1]) if chrom_idx + 1 < len(chrom_offsets) else int(pos0.shape[0])
    chrom_pos = np.asarray(pos0[chrom_start:chrom_end], dtype=np.int64)
    lo = int(np.searchsorted(chrom_pos, np.uint32(start_pos0), side="left"))
    hi = int(np.searchsorted(chrom_pos, np.uint32(end_pos0), side="right"))
    row_ids = np.arange(chrom_start + lo, chrom_start + hi, dtype=np.int64)
    region_pos = chrom_pos[lo:hi]
    if row_ids.size == 0:
        raise ValueError(
            f"No indexed CpGs overlapped region {chrom}:{start_pos0:,}-{end_pos0:,} in the cohort store"
        )
    return row_ids, region_pos


def _format_track_label(key: TrackKey) -> str:
    return f"{key.assay} | {key.haplotype} | {key.strand}"


def _parse_multi_values(values: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        for token in str(raw).split(","):
            value = token.strip()
            if value and value not in seen:
                out.append(value)
                seen.add(value)
    return out


def _read_sample_file(path: str | None) -> list[str]:
    if not path:
        return []
    out: list[str] = []
    seen: set[str] = set()
    with open(path, "r") as fh:
        for line in fh:
            value = line.strip()
            if value and value not in seen:
                out.append(value)
                seen.add(value)
    return out


def _resolve_colorable_columns(metadata: pd.DataFrame) -> list[str]:
    preferred = ["display_id", "sample_name", "tissue_name", "tissue_broad", "platform", "sex", "preservation", "technology", "center"]
    out: list[str] = []
    seen: set[str] = set()
    for col in preferred:
        if col in metadata.columns and col not in seen:
            out.append(col)
            seen.add(col)
    for col in metadata.columns:
        if col in seen or col in {"sample_root"}:
            continue
        vals = pd.Series(metadata[col], dtype="string").fillna("").astype(str).str.strip()
        vals = vals[vals != ""]
        unique_n = int(vals.nunique(dropna=True))
        if 2 <= unique_n <= 40:
            out.append(col)
            seen.add(col)
    if "display_id" not in out and "sample_name" in metadata.columns:
        out.insert(0, "sample_name")
    return out


def _default_color_by(metadata: pd.DataFrame, requested: str | None) -> str:
    colorable = _resolve_colorable_columns(metadata)
    if requested:
        if requested not in metadata.columns:
            raise ValueError(f"--color-by column {requested!r} was not found in aligned metadata")
        return requested
    for col in colorable:
        if col not in {"display_id", "sample_name"}:
            return col
    if "display_id" in metadata.columns:
        return "display_id"
    return "sample_name"


def _default_series_mode(metadata: pd.DataFrame, default_color_by: str) -> str:
    if len(metadata) >= 60 and default_color_by not in {"display_id", "sample_name"}:
        vals = pd.Series(metadata[default_color_by], dtype="string").fillna("").astype(str).str.strip()
        vals = vals[vals != ""]
        if int(vals.nunique(dropna=True)) >= 2:
            return "groups"
    return "samples"


def _resolve_requested_samples(metadata: pd.DataFrame, requested_tokens: list[str]) -> list[str]:
    available = metadata["sample_name"].astype(str).tolist()
    available_set = set(available)
    if not requested_tokens:
        return available

    alias_map: dict[str, str] = {}
    ambiguous: set[str] = set()
    candidate_cols = [col for col in ["sample_name", "display_id", "sample_id", "id", "sample_label", "sample_root"] if col in metadata.columns]
    for col in candidate_cols:
        values = pd.Series(metadata[col], dtype="string").fillna("").astype(str).str.strip()
        for sample_name, alias in zip(available, values, strict=False):
            if not alias:
                continue
            if alias in alias_map and alias_map[alias] != sample_name:
                ambiguous.add(alias)
            else:
                alias_map[alias] = sample_name
    for alias in ambiguous:
        alias_map.pop(alias, None)

    resolved: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for token in requested_tokens:
        sample_name = token if token in available_set else alias_map.get(token)
        if sample_name is None:
            missing.append(token)
            continue
        if sample_name not in seen:
            resolved.append(sample_name)
            seen.add(sample_name)
    if missing:
        raise ValueError(
            "Requested samples were not found in the selected cohort tracks: " + ", ".join(missing)
        )
    return resolved


def _select_common_samples(
    input_path: str,
    selected_tracks: list[TrackKey],
    logger: logging.Logger,
) -> list[str]:
    if not selected_tracks:
        raise ValueError("No tracks were selected for plotting")
    ordered = list(load_view_columns(input_path, selected_tracks[0])["sample_id"])
    common = set(ordered)
    for key in selected_tracks[1:]:
        common &= set(load_view_columns(input_path, key)["sample_id"])
    sample_ids = [sample_id for sample_id in ordered if sample_id in common]
    if not sample_ids:
        raise ValueError("No common samples were present across the selected cohort tracks")
    dropped = len(ordered) - len(sample_ids)
    if dropped > 0:
        logger.info(
            "Retained %d common samples across %d track(s); %d sample(s) missing from at least one selected track were skipped",
            len(sample_ids),
            len(selected_tracks),
            dropped,
        )
    return sample_ids


def _read_track_matrix(
    input_path: str,
    key: TrackKey,
    row_ids: np.ndarray,
    sample_ids: list[str],
) -> np.ndarray:
    reader, columns, _ = load_view_reader(input_path, key)
    try:
        sample_index = {sample_id: idx for idx, sample_id in enumerate(columns["sample_id"])}
        col_idx = np.asarray([sample_index[sample_id] for sample_id in sample_ids], dtype=np.int64)
        matrix = np.asarray(reader.read_rows(row_ids)[:, col_idx], dtype=np.float32)
        return matrix.T
    finally:
        reader.close()


def _combine_track_matrices(matrices: dict[str, np.ndarray], method: str) -> np.ndarray:
    stack = np.stack([np.asarray(matrix, dtype=np.float32) for matrix in matrices.values()], axis=0)
    if method == "mean":
        with np.errstate(invalid="ignore"):
            return np.nanmean(stack, axis=0).astype(np.float32)
    if method == "sum":
        out = np.nansum(stack, axis=0).astype(np.float32)
        all_missing = np.all(np.isnan(stack), axis=0)
        out[all_missing] = np.nan
        return out
    raise ValueError(f"Unsupported combine method: {method}")


def _matrix_has_signal(matrix: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(matrix, dtype=np.float32)).any())


def _write_profile_npz(
    outdir: str,
    region_pos0: np.ndarray,
    matrices: dict[str, np.ndarray],
    metadata: pd.DataFrame,
) -> str:
    payload: dict[str, object] = {
        "positions_pos0": np.asarray(region_pos0, dtype=np.int32),
        "positions_pos1": np.asarray(region_pos0, dtype=np.int32) + 1,
        "sample_name": metadata["sample_name"].astype(str).to_numpy(dtype=object),
        "display_id": metadata["display_id"].astype(str).to_numpy(dtype=object),
    }
    for track_name, matrix in matrices.items():
        payload[f"track__{track_name}"] = np.asarray(matrix, dtype=np.float32)
    out_path = os.path.join(outdir, "region_profiles.npz")
    np.savez_compressed(out_path, **payload)
    return out_path


def sliding_mean(
    xs: np.ndarray,
    ys: np.ndarray,
    window_size: int,
    min_points_for_smooth: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if (
        x_arr.size == 0
        or window_size < 2
        or x_arr.size < window_size
        or x_arr.size < min_points_for_smooth
    ):
        return x_arr, y_arr

    csum_x = np.cumsum(x_arr, dtype=np.float64)
    csum_y = np.cumsum(y_arr, dtype=np.float64)
    out_x = csum_x[window_size - 1 :] - np.concatenate(([0.0], csum_x[:-window_size]))
    out_y = csum_y[window_size - 1 :] - np.concatenate(([0.0], csum_y[:-window_size]))
    return out_x / window_size, out_y / window_size


def _build_smoothed_profile_table(
    region_chrom: str,
    region_pos0: np.ndarray,
    metadata: pd.DataFrame,
    track_labels: dict[str, str],
    matrices: dict[str, np.ndarray],
    window_size: int,
    min_points_for_smooth: int,
) -> pd.DataFrame:
    keep_cols = [col for col in ["sample_name", "display_id", "sample_root", "sample_id", "id", "tissue_name", "tissue_broad", "platform", "sex", "preservation", "technology", "center"] if col in metadata.columns]
    frames: list[pd.DataFrame] = []
    positions = np.asarray(region_pos0, dtype=np.float64)
    meta_rows = metadata[keep_cols].reset_index(drop=True)

    for track_name, matrix in matrices.items():
        label = track_labels[track_name]
        for sample_idx in range(matrix.shape[0]):
            smooth_x, smooth_y = sliding_mean(positions, matrix[sample_idx, :], window_size, min_points_for_smooth)
            if smooth_x.size == 0:
                continue
            sample_meta = meta_rows.iloc[[sample_idx]].copy()
            sample_meta = pd.concat([sample_meta] * int(smooth_x.size), ignore_index=True)
            sample_meta.insert(0, "track", track_name)
            sample_meta.insert(1, "track_display", label)
            sample_meta.insert(2, "chrom", region_chrom)
            sample_meta["x_pos0_mean"] = np.asarray(smooth_x, dtype=np.float64)
            sample_meta["x_pos1_mean"] = sample_meta["x_pos0_mean"] + 1.0
            sample_meta["value_fraction"] = np.asarray(smooth_y, dtype=np.float64)
            sample_meta["value_percent"] = sample_meta["value_fraction"] * 100.0
            sample_meta["window_size"] = int(window_size)
            frames.append(sample_meta)

    if not frames:
        return pd.DataFrame(
            columns=[
                "track",
                "track_display",
                "chrom",
                "sample_name",
                "display_id",
                "x_pos0_mean",
                "x_pos1_mean",
                "value_fraction",
                "value_percent",
                "window_size",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _build_track_payloads(
    track_labels: dict[str, str],
    matrices: dict[str, np.ndarray],
    selected_tracks: list[TrackKey],
    combine_tracks: str,
) -> list[dict[str, object]]:
    source_track_names = [key.name() for key in selected_tracks]
    payloads: list[dict[str, object]] = []
    for key in selected_tracks:
        track_name = key.name()
        payloads.append(
            {
                "name": track_name,
                "label": track_labels[track_name],
                "kind": "track",
                "assay": key.assay,
                "haplotype": key.haplotype,
                "strand": key.strand,
                "combine_method": "none",
                "source_tracks": [track_name],
                "matrix_b64": _serialize_float32(matrices[track_name]),
            }
        )
    if combine_tracks != "none":
        combined_name = f"combined::{combine_tracks}"
        assays = sorted({key.assay for key in selected_tracks})
        haplotypes = sorted({key.haplotype for key in selected_tracks})
        strands = sorted({key.strand for key in selected_tracks})
        payloads.append(
            {
                "name": combined_name,
                "label": f"Combined ({combine_tracks})",
                "kind": "combined",
                "assay": assays[0] if len(assays) == 1 else "combined",
                "haplotype": haplotypes[0] if len(haplotypes) == 1 else "combined",
                "strand": strands[0] if len(strands) == 1 else "combined",
                "combine_method": combine_tracks,
                "source_tracks": source_track_names,
                "matrix_b64": _serialize_float32(matrices[combined_name]),
            }
        )
    return payloads


def write_html(
    out_html: str,
    *,
    region_chrom: str,
    start_pos0: int,
    end_pos0: int,
    region_pos0: np.ndarray,
    metadata: pd.DataFrame,
    track_payloads: list[dict[str, object]],
    initial_window_size: int,
    min_points_for_smooth: int,
    default_color_by: str,
    style_name: str,
) -> None:
    style = PLOT_STYLE_PRESETS.get(style_name, PLOT_STYLE_PRESETS["studio"])
    colorable_cols = _resolve_colorable_columns(metadata)
    if default_color_by in metadata.columns and default_color_by not in colorable_cols:
        colorable_cols = [default_color_by] + colorable_cols
    metadata_cols = [col for col in ["sample_name", "display_id", "sample_root"] + colorable_cols if col in metadata.columns]
    seen_cols: set[str] = set()
    ordered_metadata_cols: list[str] = []
    for col in metadata_cols:
        if col not in seen_cols:
            ordered_metadata_cols.append(col)
            seen_cols.add(col)
    metadata_records = (
        metadata[ordered_metadata_cols]
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    payload = {
        "title": f"Methylation Profile | {region_chrom}:{start_pos0 + 1:,}-{end_pos0 + 1:,}",
        "region": {
            "chrom": region_chrom,
            "start_pos0": int(start_pos0),
            "end_pos0": int(end_pos0),
            "n_cpgs": int(region_pos0.size),
        },
        "positions_b64": _serialize_int32(np.asarray(region_pos0, dtype=np.int32)),
        "n_points": int(region_pos0.size),
        "n_samples": int(len(metadata_records)),
        "initial_window_size": int(initial_window_size),
        "min_points_for_smooth": int(min_points_for_smooth),
        "default_color_by": default_color_by,
        "default_series_mode": _default_series_mode(metadata, default_color_by),
        "colorable_cols": colorable_cols,
        "metadata_records": metadata_records,
        "tracks": track_payloads,
        "style": {
            "paper_bg": style["paper_bg"],
            "plot_bg": style["plot_bg"],
            "font_color": style["font_color"],
            "grid": style["grid"],
            "axis": style["axis"],
            "menu_bg": style["menu_bg"],
            "menu_border": style["menu_border"],
            "colorway": list(style["colorway"]),
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>mdb plot</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --paper-bg: {style["paper_bg"]};
      --plot-bg: {style["plot_bg"]};
      --font-color: {style["font_color"]};
      --grid-color: {style["grid"]};
      --axis-color: {style["axis"]};
      --menu-bg: {style["menu_bg"]};
      --menu-border: {style["menu_border"]};
      --accent-a: {style["accent_a"]};
      --accent-b: {style["accent_b"]};
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Source Sans Pro", "Segoe UI", sans-serif;
      color: var(--font-color);
      background:
        radial-gradient(circle at top right, var(--accent-a), transparent 36%),
        radial-gradient(circle at bottom left, var(--accent-b), transparent 30%),
        linear-gradient(180deg, var(--paper-bg), color-mix(in srgb, var(--paper-bg) 86%, #ffffff 14%));
    }}
    .shell {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px 22px 36px;
    }}
    .hero {{
      padding: 20px 22px 18px;
      border: 1px solid color-mix(in srgb, var(--menu-border) 72%, #ffffff 28%);
      border-radius: 22px;
      background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(255,255,255,0.62));
      backdrop-filter: blur(10px);
      box-shadow: 0 20px 48px rgba(15, 47, 79, 0.08);
    }}
    .kicker {{
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      opacity: 0.72;
      margin-bottom: 8px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.02;
    }}
    .lede {{
      margin: 10px 0 0;
      max-width: 920px;
      font-size: 15px;
      line-height: 1.55;
      opacity: 0.92;
    }}
    .hero-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .chip {{
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid color-mix(in srgb, var(--menu-border) 72%, #ffffff 28%);
      background: rgba(255,255,255,0.72);
      font-size: 12px;
      letter-spacing: 0.02em;
    }}
    .panel {{
      margin-top: 18px;
      padding: 18px;
      border-radius: 20px;
      border: 1px solid color-mix(in srgb, var(--menu-border) 68%, #ffffff 32%);
      background: rgba(255,255,255,0.84);
      box-shadow: 0 16px 34px rgba(15, 47, 79, 0.06);
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
      align-items: end;
    }}
    .control-label {{
      display: block;
      margin-bottom: 7px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      opacity: 0.78;
    }}
    select,
    input {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--menu-border);
      background: var(--menu-bg);
      color: var(--font-color);
      font: inherit;
    }}
    .note {{
      font-size: 13px;
      line-height: 1.5;
      opacity: 0.82;
    }}
    #summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .summary-chip {{
      padding: 8px 10px;
      border-radius: 14px;
      background: color-mix(in srgb, var(--paper-bg) 82%, #ffffff 18%);
      border: 1px solid color-mix(in srgb, var(--menu-border) 72%, #ffffff 28%);
      font-size: 13px;
    }}
    #plot {{
      min-height: 760px;
    }}
    @media (max-width: 720px) {{
      .shell {{
        padding: 18px 14px 24px;
      }}
      .hero,
      .panel {{
        padding: 16px;
        border-radius: 18px;
      }}
      #plot {{
        min-height: 540px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="kicker">mdb plot</div>
      <h1>Locus-focused DNA methylation profiles</h1>
      <p class="lede">Switch assay and haplotype directly in the browser, show one strand or overlay plus and minus together, recolor by metadata, toggle grouped means, and change the smoothing window without rerunning the command. The underlying data come from CpG-level cohort values in the selected <code>.mmdb</code> region.</p>
      <div class="hero-meta">
        <div class="chip">Region: {region_chrom}:{start_pos0 + 1:,}-{end_pos0 + 1:,}</div>
        <div class="chip">Samples: {len(metadata_records):,}</div>
        <div class="chip">CpGs: {int(region_pos0.size):,}</div>
        <div class="chip">Initial window: {initial_window_size}</div>
      </div>
    </section>

    <section class="panel controls">
      <div>
        <label class="control-label" for="assaySelect">Assay</label>
        <select id="assaySelect"></select>
      </div>
      <div>
        <label class="control-label" for="haplotypeSelect">Haplotype</label>
        <select id="haplotypeSelect"></select>
      </div>
      <div>
        <label class="control-label" for="strandSelect">Strand</label>
        <select id="strandSelect"></select>
      </div>
      <div>
        <label class="control-label" for="colorSelect">Color By</label>
        <select id="colorSelect"></select>
      </div>
      <div>
        <label class="control-label" for="seriesSelect">Series Mode</label>
        <select id="seriesSelect">
          <option value="samples">Sample Lines</option>
          <option value="groups">Grouped Means</option>
        </select>
      </div>
      <div>
        <label class="control-label" for="windowInput">Window Size</label>
        <input id="windowInput" type="number" min="1" step="1" value="{initial_window_size}" />
      </div>
    </section>

    <section class="panel note">
      Group mode averages raw sample values within each selected metadata category before smoothing. In strand-aware cohorts, the Strand selector can show <code>plus</code>, <code>minus</code>, or <code>plus + minus</code> together on the same plot.
    </section>

    <section class="panel" id="summary"></section>
    <section class="panel"><div id="plot"></div></section>
  </div>

  <script>
    const DATA = __MDB_PLOT_PAYLOAD__;
    const assaySelect = document.getElementById("assaySelect");
    const haplotypeSelect = document.getElementById("haplotypeSelect");
    const strandSelect = document.getElementById("strandSelect");
    const colorSelect = document.getElementById("colorSelect");
    const seriesSelect = document.getElementById("seriesSelect");
    const windowInput = document.getElementById("windowInput");
    const plotNode = document.getElementById("plot");
    const summaryNode = document.getElementById("summary");
    const decodedMatrices = new Map();
    const TRACKS = new Map(DATA.tracks.map((track) => [track.name, track]));

    function decodeBase64ToInt32(b64) {{
      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {{
        bytes[i] = binary.charCodeAt(i);
      }}
      return new Int32Array(bytes.buffer);
    }}

    function decodeBase64ToFloat32(b64) {{
      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {{
        bytes[i] = binary.charCodeAt(i);
      }}
      return new Float32Array(bytes.buffer);
    }}

    const POSITIONS = decodeBase64ToInt32(DATA.positions_b64);

    function getMatrix(trackName) {{
      if (!decodedMatrices.has(trackName)) {{
        decodedMatrices.set(trackName, decodeBase64ToFloat32(TRACKS.get(trackName).matrix_b64));
      }}
      return decodedMatrices.get(trackName);
    }}

    function fillSelect(select, options, selectedValue) {{
      select.innerHTML = "";
      for (const option of options) {{
        const node = document.createElement("option");
        node.value = option.value;
        node.textContent = option.label;
        if (option.value === selectedValue) {{
          node.selected = true;
        }}
        select.appendChild(node);
      }}
    }}

    function metadataValue(record, column) {{
      const value = record[column];
      if (value === undefined || value === null || String(value).trim() === "") {{
        return "NA";
      }}
      return String(value);
    }}

    function sampleDisplayLabel(record) {{
      for (const column of ["display_id", "sample_name", "id"]) {{
        const value = record[column];
        if (value !== undefined && value !== null && String(value).trim() !== "") {{
          return String(value);
        }}
      }}
      return "sample";
    }}

    function buildColorMap(colorBy) {{
      const palette = DATA.style.colorway;
      const ordered = Array.from(new Set(DATA.metadata_records.map((record) => metadataValue(record, colorBy)))).sort();
      const out = new Map();
      ordered.forEach((value, idx) => out.set(value, palette[idx % palette.length]));
      return out;
    }}

    function uniqueTrackValues(key, filters = {{}}) {{
      const out = [];
      const seen = new Set();
      for (const track of DATA.tracks) {{
        let matches = true;
        for (const [filterKey, filterValue] of Object.entries(filters)) {{
          if (filterValue && track[filterKey] !== filterValue) {{
            matches = false;
            break;
          }}
        }}
        if (!matches || seen.has(track[key])) {{
          continue;
        }}
        seen.add(track[key]);
        out.push(track[key]);
      }}
      return out;
    }}

    function formatStrandSelection(value, strands) {{
      if (value !== "both") {{
        return value;
      }}
      return Array.from(new Set(strands)).join(" + ");
    }}

    function syncTrackSelectors() {{
      const assayOptions = uniqueTrackValues("assay");
      const assayValue = assayOptions.includes(assaySelect.value) ? assaySelect.value : assayOptions[0];
      fillSelect(
        assaySelect,
        assayOptions.map((value) => ({{ value, label: value }})),
        assayValue,
      );

      const haplotypeOptions = uniqueTrackValues("haplotype", {{ assay: assaySelect.value }});
      const haplotypeValue = haplotypeOptions.includes(haplotypeSelect.value) ? haplotypeSelect.value : haplotypeOptions[0];
      fillSelect(
        haplotypeSelect,
        haplotypeOptions.map((value) => ({{ value, label: value }})),
        haplotypeValue,
      );

      const rawStrandOptions = uniqueTrackValues("strand", {{
        assay: assaySelect.value,
        haplotype: haplotypeSelect.value,
      }});
      const strandOptions = rawStrandOptions.length > 1 ? [...rawStrandOptions, "both"] : rawStrandOptions;
      const strandValue = strandOptions.includes(strandSelect.value)
        ? strandSelect.value
        : (strandOptions.includes("both") ? "both" : strandOptions[0]);
      fillSelect(
        strandSelect,
        strandOptions.map((value) => ({{
          value,
          label: value === "both" ? rawStrandOptions.join(" + ") : value,
        }})),
        strandValue,
      );
    }}

    function currentTracks() {{
      const matching = DATA.tracks.filter((track) =>
        track.assay === assaySelect.value &&
        track.haplotype === haplotypeSelect.value &&
        (strandSelect.value === "both" || track.strand === strandSelect.value)
      );
      return matching.length > 0 ? matching : [DATA.tracks[0]];
    }}

    function strandLineDash(strand, multiStrand) {{
      if (!multiStrand) {{
        return "solid";
      }}
      if (strand === "minus") {{
        return "dash";
      }}
      if (strand === "plus") {{
        return "solid";
      }}
      return "dot";
    }}

    function smoothSeries(values, windowSize, minPoints) {{
      const xs = [];
      const ys = [];
      for (let idx = 0; idx < POSITIONS.length; idx += 1) {{
        const value = values[idx];
        if (Number.isFinite(value)) {{
          xs.push(POSITIONS[idx] + 1);
          ys.push(value * 100.0);
        }}
      }}
      if (xs.length === 0) {{
        return {{ x: [], y: [] }};
      }}
      if (windowSize < 2 || xs.length < windowSize || xs.length < minPoints) {{
        return {{ x: xs, y: ys }};
      }}
      const prefixX = new Array(xs.length + 1).fill(0);
      const prefixY = new Array(ys.length + 1).fill(0);
      for (let i = 0; i < xs.length; i += 1) {{
        prefixX[i + 1] = prefixX[i] + xs[i];
        prefixY[i + 1] = prefixY[i] + ys[i];
      }}
      const outX = [];
      const outY = [];
      for (let hi = windowSize; hi <= xs.length; hi += 1) {{
        const lo = hi - windowSize;
        outX.push((prefixX[hi] - prefixX[lo]) / windowSize);
        outY.push((prefixY[hi] - prefixY[lo]) / windowSize);
      }}
      return {{ x: outX, y: outY }};
    }}

    function sliceSample(matrix, sampleIdx) {{
      const start = sampleIdx * DATA.n_points;
      return matrix.subarray(start, start + DATA.n_points);
    }}

    function aggregateGroup(matrix, sampleIndices) {{
      const out = new Float32Array(DATA.n_points);
      for (let j = 0; j < DATA.n_points; j += 1) {{
        let sum = 0.0;
        let count = 0;
        for (const sampleIdx of sampleIndices) {{
          const value = matrix[(sampleIdx * DATA.n_points) + j];
          if (Number.isFinite(value)) {{
            sum += value;
            count += 1;
          }}
        }}
        out[j] = count > 0 ? (sum / count) : Number.NaN;
      }}
      return out;
    }}

    function formatTrackSource(track) {{
      if (!track.source_tracks || track.source_tracks.length <= 1) {{
        return track.label;
      }}
      return track.label + " ← " + track.source_tracks.join(", ");
    }}

    function renderSummary(activeTracks, rowMode, colorBy, windowSize, renderedCount) {{
      const strandLabel = formatStrandSelection(
        strandSelect.value,
        activeTracks.map((track) => track.strand),
      );
      const items = [
        "Assay: " + assaySelect.value,
        "Haplotype: " + haplotypeSelect.value,
        "Strand: " + strandLabel,
        "Native tracks: " + activeTracks.length,
        "Series mode: " + (rowMode === "groups" ? "Grouped means" : "Sample lines"),
        "Color by: " + colorBy,
        "Window size: " + windowSize,
        "Rendered lines: " + renderedCount,
      ];
      summaryNode.innerHTML = items.map((item) => '<div class="summary-chip">' + item + "</div>").join("");
    }}

    function render() {{
      const activeTracks = currentTracks();
      const colorBy = colorSelect.value;
      const rowMode = seriesSelect.value;
      const windowSize = Math.max(1, parseInt(windowInput.value || String(DATA.initial_window_size), 10) || DATA.initial_window_size);
      const colorMap = buildColorMap(colorBy);
      const traces = [];
      let maxY = 100.0;
      const multiStrand = new Set(activeTracks.map((track) => track.strand)).size > 1;
      const strandLabel = formatStrandSelection(
        strandSelect.value,
        activeTracks.map((track) => track.strand),
      );

      if (rowMode === "groups") {{
        const groups = new Map();
        DATA.metadata_records.forEach((record, idx) => {{
          const category = metadataValue(record, colorBy);
          if (!groups.has(category)) {{
            groups.set(category, []);
          }}
          groups.get(category).push(idx);
        }});
        Array.from(groups.keys()).sort().forEach((category) => {{
          const sampleIndices = groups.get(category);
          activeTracks.forEach((track) => {{
            const combined = aggregateGroup(getMatrix(track.name), sampleIndices);
            const smooth = smoothSeries(combined, windowSize, DATA.min_points_for_smooth);
            if (smooth.x.length === 0) {{
              return;
            }}
            for (const value of smooth.y) {{
              if (value > maxY) {{
                maxY = value;
              }}
            }}
            const trackSuffix = multiStrand ? " | " + track.strand : "";
            traces.push({{
              type: "scatter",
              mode: "lines",
              name: category + trackSuffix + " (n=" + sampleIndices.length + ")",
              x: smooth.x,
              y: smooth.y,
              legendgroup: category + "::" + track.strand,
              line: {{
                color: colorMap.get(category),
                width: 3.0,
                dash: strandLineDash(track.strand, multiStrand),
              }},
              hovertemplate:
                "<b>" + category + trackSuffix + " (n=" + sampleIndices.length + ")</b><br>" +
                "Assay: " + track.assay + "<br>" +
                "Strand: " + track.strand + "<br>" +
                "Position: %{{x:,}}<br>Methylation: %{{y:.2f}}%<extra></extra>",
            }});
          }});
        }});
      }} else {{
        const showLegend = DATA.n_samples <= 24;
        DATA.metadata_records.forEach((record, sampleIdx) => {{
          const category = metadataValue(record, colorBy);
          const sampleLabel = sampleDisplayLabel(record);
          activeTracks.forEach((track) => {{
            const smooth = smoothSeries(sliceSample(getMatrix(track.name), sampleIdx), windowSize, DATA.min_points_for_smooth);
            if (smooth.x.length === 0) {{
              return;
            }}
            for (const value of smooth.y) {{
              if (value > maxY) {{
                maxY = value;
              }}
            }}
            const trackSuffix = multiStrand ? " | " + track.strand : "";
            traces.push({{
              type: "scatter",
              mode: "lines",
              name: sampleLabel + trackSuffix,
              x: smooth.x,
              y: smooth.y,
              showlegend: showLegend,
              line: {{
                color: colorMap.get(category),
                width: 2.2,
                dash: strandLineDash(track.strand, multiStrand),
              }},
              legendgroup: category + "::" + track.strand,
              hovertemplate:
                "<b>" + sampleLabel + trackSuffix + "</b><br>" +
                "Assay: " + track.assay + "<br>" +
                "Strand: " + track.strand + "<br>" +
                "Position: %{{x:,}}<br>" +
                "Methylation: %{{y:.2f}}%<br>" +
                colorBy + ": " + category +
                "<extra></extra>",
            }});
          }});
        }});
      }}

      const layout = {{
        template: "none",
        paper_bgcolor: DATA.style.paper_bg,
        plot_bgcolor: DATA.style.plot_bg,
        width: 1180,
        height: 760,
        margin: {{ l: 74, r: 40, t: 86, b: 84 }},
        font: {{ family: "Source Sans Pro, Segoe UI, sans-serif", size: 14, color: DATA.style.font_color }},
        title: {{
          text:
            assaySelect.value + " | " +
            haplotypeSelect.value + " | " +
            strandLabel + " | " +
            DATA.region.chrom + ":" + (DATA.region.start_pos0 + 1).toLocaleString() + "-" + (DATA.region.end_pos0 + 1).toLocaleString(),
          x: 0.01,
          y: 0.98,
          xanchor: "left",
          yanchor: "top",
          font: {{ size: 22, color: DATA.style.font_color }},
        }},
        legend: {{
          bgcolor: "rgba(255,255,255,0.84)",
          bordercolor: DATA.style.menu_border,
          borderwidth: 1,
          orientation: "v",
          x: 1.01,
          xanchor: "left",
          y: 1.0,
          yanchor: "top",
          font: {{ size: 11, color: DATA.style.font_color }},
        }},
        xaxis: {{
          title: "Genomic position",
          tickformat: ",d",
          showgrid: true,
          gridcolor: DATA.style.grid,
          linecolor: DATA.style.axis,
          tickcolor: DATA.style.axis,
          showline: true,
          zeroline: false,
        }},
        yaxis: {{
          title: "Methylation (%)",
          range: [0, Math.max(100.0, maxY * 1.04)],
          showgrid: true,
          gridcolor: DATA.style.grid,
          linecolor: DATA.style.axis,
          tickcolor: DATA.style.axis,
          showline: true,
          zeroline: false,
        }},
        hoverlabel: {{
          bgcolor: "rgba(255,255,255,0.96)",
          bordercolor: DATA.style.menu_border,
          font: {{ size: 12, color: DATA.style.font_color }},
        }},
      }};

      Plotly.react(plotNode, traces, layout, {{ responsive: true, displaylogo: false }});
      renderSummary(activeTracks, rowMode, colorBy, windowSize, traces.length);
    }}

    fillSelect(
      seriesSelect,
      [
        {{ value: "samples", label: "Sample Lines" }},
        {{ value: "groups", label: "Grouped Means" }},
      ],
      DATA.default_series_mode,
    );
    syncTrackSelectors();
    fillSelect(
      colorSelect,
      DATA.colorable_cols.map((col) => ({{ value: col, label: col }})),
      DATA.default_color_by,
    );

    assaySelect.addEventListener("change", () => {{
      syncTrackSelectors();
      render();
    }});
    haplotypeSelect.addEventListener("change", () => {{
      syncTrackSelectors();
      render();
    }});
    strandSelect.addEventListener("change", render);
    colorSelect.addEventListener("change", render);
    seriesSelect.addEventListener("change", render);
    windowInput.addEventListener("change", render);
    windowInput.addEventListener("keyup", (event) => {{
      if (event.key === "Enter") {{
        render();
      }}
    }});
    window.addEventListener("resize", () => Plotly.Plots.resize(plotNode));
    render();
  </script>
</body>
</html>
"""

    html = html.replace("__MDB_PLOT_PAYLOAD__", _json_script(payload))
    with open(out_html, "w") as fh:
        fh.write(html)


def plot(
    input_path: str,
    outdir: str,
    *,
    region: str,
    metadata: str | None = None,
    sample_ids: list[str] | None = None,
    sample_file: str | None = None,
    assay: str = "5mC,5hmC",
    haplotype: str = "combined",
    strand: str = "combined",
    combine_tracks: str = "none",
    window_size: int = 20,
    min_points_for_smooth: int = 3,
    color_by: str | None = None,
    plot_style: str = "studio",
    verbose: bool = False,
) -> dict[str, str]:
    if int(window_size) <= 0:
        raise ValueError("window_size must be > 0")
    if int(min_points_for_smooth) <= 0:
        raise ValueError("min_points_for_smooth must be > 0")
    if combine_tracks not in {"none", "sum", "mean"}:
        raise ValueError("combine_tracks must be one of: none, sum, mean")

    os.makedirs(outdir, exist_ok=True)
    logger = setup_logging(outdir, verbose=verbose)
    started = time.time()

    kind = detect_store_kind(input_path)
    if kind not in {"cohort_store_npy", "cohort_store_zarr"}:
        raise ValueError(f"mdb plot expects a cohort store (.mmdb); found store kind {kind!r} at {input_path}")

    chrom, start_pos0, end_pos0 = _parse_region(region)
    selected_tracks = select_tracks_for_cohort(input_path, assay=assay, haplotype=haplotype, strand=strand)
    logger.info("Selected %d track(s): %s", len(selected_tracks), ", ".join(key.name() for key in selected_tracks))

    all_sample_ids = _select_common_samples(input_path, selected_tracks, logger)
    metadata_df = align_metadata(all_sample_ids, metadata, logger)

    requested_tokens = _parse_multi_values(sample_ids) + _read_sample_file(sample_file)
    chosen_sample_ids = _resolve_requested_samples(metadata_df, requested_tokens)
    metadata_df = metadata_df[metadata_df["sample_name"].astype(str).isin(chosen_sample_ids)].copy()
    metadata_df["__sample_order__"] = pd.Categorical(metadata_df["sample_name"].astype(str), categories=chosen_sample_ids, ordered=True)
    metadata_df = metadata_df.sort_values("__sample_order__").drop(columns="__sample_order__").reset_index(drop=True)
    if metadata_df.empty:
        raise ValueError("No samples remained after applying the requested sample filters")
    metadata_out = os.path.join(outdir, "sample_metadata_aligned.tsv")
    metadata_df.to_csv(metadata_out, sep="\t", index=False)
    logger.info("Selected %d sample(s) for plotting", len(metadata_df))

    chroms, chrom_offsets, pos0 = load_cohort_index(input_path)
    row_ids, region_pos0 = _region_rows(chroms, chrom_offsets, pos0, chrom, start_pos0, end_pos0)
    logger.info(
        "Resolved region %s:%d-%d to %d indexed CpGs",
        chrom,
        start_pos0,
        end_pos0,
        len(region_pos0),
    )

    chosen_sample_ids = metadata_df["sample_name"].astype(str).tolist()
    track_labels: dict[str, str] = {}
    matrices: dict[str, np.ndarray] = {}
    for key in selected_tracks:
        matrix = _read_track_matrix(input_path, key, row_ids, chosen_sample_ids)
        track_name = key.name()
        track_labels[track_name] = _format_track_label(key)
        matrices[track_name] = matrix
        logger.info(
            "Loaded %s matrix with shape samples=%d x cpgs=%d",
            track_name,
            matrix.shape[0],
            matrix.shape[1],
        )

    if not any(_matrix_has_signal(matrix) for matrix in matrices.values()):
        raise ValueError(
            "No observed methylation values were found in the selected region for the requested samples and tracks"
        )

    if combine_tracks != "none":
        combined_name = f"combined::{combine_tracks}"
        matrices[combined_name] = _combine_track_matrices(
            {key.name(): matrices[key.name()] for key in selected_tracks},
            combine_tracks,
        )
        track_labels[combined_name] = f"Combined ({combine_tracks})"
        logger.info("Added synthetic track %s across %d selected track(s)", combined_name, len(selected_tracks))

    payload_npz = _write_profile_npz(outdir, region_pos0, matrices, metadata_df)
    smoothed_df = _build_smoothed_profile_table(
        chrom,
        region_pos0,
        metadata_df,
        track_labels,
        matrices,
        window_size=int(window_size),
        min_points_for_smooth=int(min_points_for_smooth),
    )
    smoothed_out = os.path.join(outdir, "smoothed_profiles.tsv.gz")
    smoothed_df.to_csv(smoothed_out, sep="\t", index=False, compression="gzip")

    default_color = _default_color_by(metadata_df, color_by)
    html_out = os.path.join(outdir, "methylation_plot.html")
    track_payloads = _build_track_payloads(track_labels, matrices, selected_tracks, combine_tracks)
    write_html(
        html_out,
        region_chrom=chrom,
        start_pos0=start_pos0,
        end_pos0=end_pos0,
        region_pos0=region_pos0,
        metadata=metadata_df,
        track_payloads=track_payloads,
        initial_window_size=int(window_size),
        min_points_for_smooth=int(min_points_for_smooth),
        default_color_by=default_color,
        style_name=plot_style,
    )

    manifest = {
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": os.path.abspath(input_path),
        "region": {
            "chrom": chrom,
            "start_pos0": int(start_pos0),
            "end_pos0": int(end_pos0),
            "n_cpgs": int(region_pos0.size),
        },
        "metadata": metadata,
        "n_samples": int(len(metadata_df)),
        "samples": metadata_df["sample_name"].astype(str).tolist(),
        "selected_tracks": [key.name() for key in selected_tracks],
        "combine_tracks": combine_tracks,
        "window_size": int(window_size),
        "min_points_for_smooth": int(min_points_for_smooth),
        "default_color_by": default_color,
        "outputs": {
            "html": html_out,
            "npz": payload_npz,
            "smoothed_tsv_gz": smoothed_out,
            "metadata_tsv": metadata_out,
        },
        "runtime_seconds": round(time.time() - started, 3),
    }
    manifest_out = os.path.join(outdir, "plot_manifest.json")
    with open(manifest_out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote %s", html_out)
    logger.info("Completed mdb plot in %.2fs", time.time() - started)
    return manifest["outputs"]


def plot_main(args) -> None:
    plot(
        input_path=args.input,
        outdir=args.outdir,
        region=args.region,
        metadata=getattr(args, "metadata", None),
        sample_ids=list(getattr(args, "sample_id", []) or []),
        sample_file=getattr(args, "sample_file", None),
        assay=getattr(args, "assay", "5mC,5hmC"),
        haplotype=getattr(args, "haplotype", "combined"),
        strand=getattr(args, "strand", "combined"),
        combine_tracks=getattr(args, "combine_tracks", "none"),
        window_size=int(getattr(args, "window_size", 20)),
        min_points_for_smooth=int(getattr(args, "min_points_for_smooth", 3)),
        color_by=getattr(args, "color_by", None),
        plot_style=getattr(args, "plot_style", "studio"),
        verbose=bool(getattr(args, "verbose", False)),
    )


__all__ = ["plot", "plot_main"]
