#!/usr/bin/env python3
from __future__ import annotations

import base64
import glob
import json
import logging
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mdb.schema import COHORT_STORE_KINDS, SAMPLE_STORE_KIND, TrackKey, VALUE_MISSING
from mdb.storage import (
    available_views,
    list_tracks,
    load_cohort_index,
    load_reference_index,
    load_view_columns,
    read_track,
    sample_manifest,
)

MATRIX_MISSING = np.uint16(65535)
SAMPLE_ROOT_RE = re.compile(r"^(SMHT[^-]+(?:-[^-]+){5})")


@dataclass(frozen=True)
class BinLayout:
    index_path: str
    chroms: list[str]
    chrom_offsets: np.ndarray
    pos0: np.ndarray
    chrom_bin_counts: dict[str, int]
    chrom_bin_offsets: dict[str, int]
    total_bins: int


def setup_logging(outdir: str, verbose: bool) -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("mdb_viz")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(outdir, "viz.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _parse_selector(value: str | None) -> set[str] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "all":
        return None
    return {part.strip() for part in text.split(",") if part.strip()}


def _manifest_kind(path: str) -> str | None:
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r") as fh:
            payload = json.load(fh)
    except Exception:
        return None
    return str(payload.get("kind", "")).strip() or None


def _expand_input_tokens(inputs: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in inputs:
        hits = glob.glob(os.path.expanduser(item))
        expanded.extend(sorted(hits) if hits else [os.path.expanduser(item)])
    if len(expanded) == 1 and os.path.isfile(expanded[0]) and expanded[0].endswith(".txt"):
        with open(expanded[0], "r") as fh:
            expanded = [line.strip() for line in fh if line.strip()]
    return [os.path.abspath(path) for path in expanded]


def _discover_bundles_under(path: str) -> list[str]:
    candidates = sorted(glob.glob(os.path.join(path, "*.smdb")))
    out = [candidate for candidate in candidates if _manifest_kind(candidate) == SAMPLE_STORE_KIND]
    if out:
        return out
    for candidate in sorted(os.listdir(path)):
        child = os.path.join(path, candidate)
        if os.path.isdir(child) and _manifest_kind(child) == SAMPLE_STORE_KIND:
            out.append(child)
    return out


def resolve_inputs(inputs: list[str]) -> tuple[str, list[str], str | None]:
    bundles: list[str] = []
    cohorts: list[str] = []
    seen_bundles: set[str] = set()

    for token in _expand_input_tokens(inputs):
        if not os.path.isdir(token):
            continue
        kind = _manifest_kind(token)
        if kind == SAMPLE_STORE_KIND:
            if token not in seen_bundles:
                bundles.append(token)
                seen_bundles.add(token)
            continue
        if kind in COHORT_STORE_KINDS:
            cohorts.append(token)
            continue
        for bundle_path in _discover_bundles_under(token):
            if bundle_path not in seen_bundles:
                bundles.append(bundle_path)
                seen_bundles.add(bundle_path)

    if cohorts and bundles:
        raise ValueError("Provide either cohort stores or sample bundles for mdb viz, not both in the same command.")
    if len(cohorts) > 1:
        raise ValueError("mdb viz expects at most one cohort store input.")
    if cohorts:
        return "cohort", bundles, cohorts[0]
    if bundles:
        return "bundles", bundles, None
    raise FileNotFoundError("No valid cohort store or sample bundle inputs were found.")


def select_tracks_for_bundles(bundle_paths: list[str], assay: str | None, haplotype: str | None, strand: str | None) -> list[TrackKey]:
    assay_sel = _parse_selector(assay)
    hap_sel = _parse_selector(haplotype)
    strand_sel = _parse_selector(strand)

    seen: set[TrackKey] = set()
    for bundle_path in bundle_paths:
        for key in list_tracks(bundle_path):
            seen.add(key)

    selected = sorted(
        key
        for key in seen
        if (not assay_sel or key.assay in assay_sel)
        and (not hap_sel or key.haplotype in hap_sel)
        and (not strand_sel or key.strand in strand_sel)
    )
    if not selected:
        raise ValueError(
            "No sample bundle tracks matched the requested filters: "
            f"assay={sorted(assay_sel) if assay_sel else 'ALL'}, "
            f"haplotype={sorted(hap_sel) if hap_sel else 'ALL'}, "
            f"strand={sorted(strand_sel) if strand_sel else 'ALL'}"
        )
    return selected


def select_tracks_for_cohort(cohort_path: str, assay: str | None, haplotype: str | None, strand: str | None) -> list[TrackKey]:
    assay_sel = _parse_selector(assay)
    hap_sel = _parse_selector(haplotype)
    strand_sel = _parse_selector(strand)
    selected = sorted(
        key
        for key in available_views(cohort_path)
        if (not assay_sel or key.assay in assay_sel)
        and (not hap_sel or key.haplotype in hap_sel)
        and (not strand_sel or key.strand in strand_sel)
    )
    if not selected:
        raise ValueError(
            "No cohort tracks matched the requested filters: "
            f"assay={sorted(assay_sel) if assay_sel else 'ALL'}, "
            f"haplotype={sorted(hap_sel) if hap_sel else 'ALL'}, "
            f"strand={sorted(strand_sel) if strand_sel else 'ALL'}"
        )
    return selected


def resolve_sample_entries(mode: str, bundle_paths: list[str], cohort_path: str | None, selected_tracks: list[TrackKey]) -> list[dict[str, str]]:
    if mode == "bundles":
        out: list[dict[str, str]] = []
        for bundle_path in bundle_paths:
            manifest = sample_manifest(bundle_path)
            out.append(
                {
                    "sample_id": str(manifest["sample_id"]),
                    "bundle_path": os.path.abspath(bundle_path),
                    "platform": str(manifest["platform"]),
                    "index_path": str(manifest["index_path"]),
                }
            )
        return out

    if cohort_path is None:
        raise ValueError("Cohort input mode requires a cohort path.")

    merged: OrderedDict[str, dict[str, str]] = OrderedDict()
    for key in selected_tracks:
        columns = load_view_columns(cohort_path, key)
        for sample_id, bundle_path, platform in zip(
            columns["sample_id"],
            columns["bundle_path"],
            columns["platform"],
            strict=False,
        ):
            if sample_id not in merged:
                manifest = sample_manifest(bundle_path)
                merged[sample_id] = {
                    "sample_id": str(sample_id),
                    "bundle_path": os.path.abspath(bundle_path),
                    "platform": str(platform),
                    "index_path": str(manifest["index_path"]),
                }
    if not merged:
        raise ValueError(f"No sample entries were found in cohort store: {cohort_path}")
    return list(merged.values())


def build_bin_layout(index_path: str, bin_length: int) -> BinLayout:
    chroms, chrom_offsets, pos0 = load_reference_index(index_path)
    chrom_bin_counts: dict[str, int] = {}
    chrom_bin_offsets: dict[str, int] = {}
    total_bins = 0
    n_rows = int(pos0.shape[0])
    chrom_ends = np.asarray(
        [int(chrom_offsets[idx + 1]) if idx + 1 < len(chrom_offsets) else n_rows for idx in range(len(chroms))],
        dtype=np.int64,
    )

    for chrom, start, end in zip(chroms, chrom_offsets, chrom_ends, strict=False):
        pos_chr = pos0[int(start) : int(end)]
        n_bins = int(pos_chr[-1] // bin_length) + 1 if pos_chr.size else 0
        chrom_bin_offsets[chrom] = total_bins
        chrom_bin_counts[chrom] = n_bins
        total_bins += n_bins

    return BinLayout(
        index_path=index_path,
        chroms=list(chroms),
        chrom_offsets=np.asarray(chrom_offsets, dtype=np.int64),
        pos0=np.asarray(pos0, dtype=np.uint32),
        chrom_bin_counts=chrom_bin_counts,
        chrom_bin_offsets=chrom_bin_offsets,
        total_bins=int(total_bins),
    )


def _aggregate_bundle_track(bundle_path: str, key: TrackKey, layout: BinLayout, bin_length: int) -> np.ndarray:
    out = np.full(layout.total_bins, MATRIX_MISSING, dtype=np.uint16)

    try:
        track = read_track(bundle_path, key)
    except Exception:
        return out

    try:
        n_rows = int(layout.pos0.shape[0])
        chrom_ends = np.asarray(
            [
                int(layout.chrom_offsets[idx + 1]) if idx + 1 < len(layout.chrom_offsets) else n_rows
                for idx in range(len(layout.chroms))
            ],
            dtype=np.int64,
        )
        for chrom, start, end in zip(layout.chroms, layout.chrom_offsets, chrom_ends, strict=False):
            n_bins = int(layout.chrom_bin_counts.get(chrom, 0))
            if n_bins <= 0 or chrom not in track.chroms_present:
                continue
            values = np.asarray(track.chrom_values(chrom, allow_missing=True), dtype=np.uint16)
            valid = values != VALUE_MISSING
            if not np.any(valid):
                continue
            pos_chr = layout.pos0[int(start) : int(end)]
            bin_idx = (pos_chr[valid] // bin_length).astype(np.int64, copy=False)
            sums = np.bincount(bin_idx, weights=values[valid].astype(np.float64), minlength=n_bins)
            counts = np.bincount(bin_idx, minlength=n_bins)
            present = counts > 0
            if not np.any(present):
                continue
            means = np.rint(sums[present] / counts[present]).astype(np.uint16, copy=False)
            offset = int(layout.chrom_bin_offsets[chrom])
            out[offset + np.flatnonzero(present)] = means
        return out
    finally:
        track.close()


def _aggregate_bundle_worker(bundle_path: str, track_names: list[str], bin_length: int) -> dict[str, object]:
    manifest = sample_manifest(bundle_path)
    sample_id = str(manifest["sample_id"])
    index_path = str(manifest["index_path"])
    layout = build_bin_layout(index_path, bin_length)
    profiles = {
        track_name: _aggregate_bundle_track(bundle_path, TrackKey.from_name(track_name), layout, bin_length)
        for track_name in track_names
    }
    return {
        "sample_id": sample_id,
        "bundle_path": os.path.abspath(bundle_path),
        "index_path": index_path,
        "profiles": profiles,
    }


def aggregate_bundle_profiles(
    sample_entries: list[dict[str, str]],
    selected_tracks: list[TrackKey],
    *,
    bin_length: int,
    workers: int,
    logger: logging.Logger,
) -> tuple[BinLayout, dict[str, np.ndarray]]:
    if not sample_entries:
        raise ValueError("No sample entries were resolved for aggregation.")

    index_paths = {entry["index_path"] for entry in sample_entries}
    if len(index_paths) != 1:
        raise ValueError(f"All samples must share one reference index path, saw {sorted(index_paths)}")

    index_path = next(iter(index_paths))
    layout = build_bin_layout(index_path, bin_length)
    track_names = [key.name() for key in selected_tracks]
    results_by_bundle: dict[str, dict[str, object]] = {}

    logger.info(
        "Aggregating %d samples across %d track(s) with %d total bins",
        len(sample_entries),
        len(track_names),
        layout.total_bins,
    )

    if int(workers) <= 1:
        for entry in tqdm(sample_entries, desc="Aggregating sample bins", unit="sample"):
            results_by_bundle[entry["bundle_path"]] = _aggregate_bundle_worker(entry["bundle_path"], track_names, bin_length)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {
                executor.submit(_aggregate_bundle_worker, entry["bundle_path"], track_names, bin_length): entry["bundle_path"]
                for entry in sample_entries
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Aggregating sample bins", unit="sample"):
                bundle_path = futures[fut]
                results_by_bundle[bundle_path] = fut.result()

    matrices = {
        track_name: np.full((len(sample_entries), layout.total_bins), MATRIX_MISSING, dtype=np.uint16)
        for track_name in track_names
    }
    for sample_idx, entry in enumerate(sample_entries):
        result = results_by_bundle[entry["bundle_path"]]
        result_index = str(result["index_path"])
        if result_index != index_path:
            raise ValueError(f"Sample {entry['sample_id']} used {result_index}, expected {index_path}")
        profiles = result["profiles"]
        for track_name in track_names:
            matrices[track_name][sample_idx, :] = np.asarray(profiles[track_name], dtype=np.uint16)

    return layout, matrices


def _canonical_sample_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    base = os.path.basename(text)
    for suffix in (".smdb", ".mmdb", ".mdb", ".bam", ".bai"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    match = SAMPLE_ROOT_RE.match(base)
    if match:
        return match.group(1)
    parts = base.split("-")
    if len(parts) >= 6 and parts[0].startswith("SMHT"):
        return "-".join(parts[:6])
    return base


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def align_metadata(sample_ids: list[str], metadata_path: str | None, logger: logging.Logger) -> pd.DataFrame:
    base = pd.DataFrame({"sample_name": pd.Series(sample_ids, dtype="string")})
    base["sample_root"] = base["sample_name"].map(_canonical_sample_id)
    base["display_id"] = base["sample_root"].where(base["sample_root"].str.len() > 0, base["sample_name"])

    if not metadata_path or not os.path.isfile(metadata_path):
        logger.info("No metadata file provided; HTML grouping will be limited to sample-level fields.")
        return base

    meta = pd.read_csv(metadata_path, sep=None, engine="python")
    if meta.empty:
        logger.info("Metadata table was empty; continuing without external metadata.")
        return base

    candidate_cols = [col for col in ["sample_name", "sample_id", "id", "sample", "sample_label"] if col in meta.columns]
    if not candidate_cols:
        logger.info("Metadata table had no alignable sample columns; continuing without external metadata.")
        return base

    sample_id_set = set(base["sample_name"].astype(str))
    sample_root_set = set(base["sample_root"].astype(str))
    best_score = -1
    best_mode = ""
    best_col = ""

    for col in candidate_cols:
        values = pd.Series(meta[col], dtype="string").fillna("").astype(str).str.strip()
        exact_score = int(sum(value in sample_id_set for value in values))
        if exact_score > best_score:
            best_score = exact_score
            best_mode = "exact"
            best_col = col

        roots = values.map(_canonical_sample_id)
        root_score = int(sum(value in sample_root_set for value in roots if value))
        if root_score > best_score:
            best_score = root_score
            best_mode = "root"
            best_col = col

    if best_score <= 0:
        logger.info("Metadata could not be aligned to sample ids; continuing without external metadata.")
        return base

    logger.info("Aligning metadata using column %s (%s match)", best_col, best_mode)
    meta2 = meta.copy()
    if best_mode == "exact":
        meta2["_merge_key"] = pd.Series(meta2[best_col], dtype="string").fillna("").astype(str).str.strip()
        base2 = base.copy()
        base2["_merge_key"] = base2["sample_name"].astype(str)
    else:
        meta2["_merge_key"] = pd.Series(meta2[best_col], dtype="string").map(_canonical_sample_id)
        base2 = base.copy()
        base2["_merge_key"] = base2["sample_root"].astype(str)

    meta2 = meta2[meta2["_merge_key"] != ""].drop_duplicates("_merge_key", keep="first")
    out = base2.merge(meta2, on="_merge_key", how="left").drop(columns=["_merge_key"])
    if "sample_name" not in out.columns:
        for candidate in ("sample_name_x", "sample_name_y"):
            if candidate in out.columns:
                out = out.rename(columns={candidate: "sample_name"})
                break
    if "sample_name_x" in out.columns and "sample_name" in out.columns:
        out = out.drop(columns=["sample_name_x"])
    if "sample_name_y" in out.columns and "sample_name" in out.columns:
        out = out.drop(columns=["sample_name_y"])
    if "sample_name" not in out.columns:
        out["sample_name"] = pd.Series(sample_ids, dtype="string")

    label_candidates = ["sample_label", "sample_id", "id", "sample_name"]
    for col in label_candidates:
        if col in out.columns:
            vals = pd.Series(out[col], dtype="string").fillna("").astype(str).str.strip()
            if (vals != "").all() and vals.nunique(dropna=False) == len(out):
                mean_len = float(vals.str.len().mean())
                current = pd.Series(out["display_id"], dtype="string").fillna("").astype(str)
                if mean_len < float(current.str.len().mean()):
                    out["display_id"] = vals
                break
    return out


def _is_reasonable_category(series: pd.Series, max_unique: int = 40) -> bool:
    vals = pd.Series(series, dtype="string").dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return False
    unique_n = int(vals.nunique(dropna=True))
    return 2 <= unique_n <= max_unique


def _prepare_group_profile_tables(
    matrices: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    layout: BinLayout,
    selected_tracks: list[TrackKey],
    bin_length: int,
    out_path: str,
) -> None:
    if "tissue_name" not in metadata.columns:
        return

    sample_groups = pd.Series(metadata["tissue_name"], dtype="string").fillna("").astype(str).str.strip()
    valid_groups = [group for group in sorted(sample_groups.unique().tolist()) if group]
    if not valid_groups:
        return

    frames: list[pd.DataFrame] = []
    for key in selected_tracks:
        matrix = np.asarray(matrices[key.name()], dtype=np.uint16)
        decoded = matrix.astype(np.float32)
        missing = matrix == MATRIX_MISSING
        decoded[missing] = np.nan
        decoded[~missing] = decoded[~missing] / 10000.0

        for group in valid_groups:
            sample_idx = np.flatnonzero(sample_groups.to_numpy(dtype=object) == group)
            if sample_idx.size == 0:
                continue
            subset = decoded[sample_idx, :]
            valid_counts = np.count_nonzero(~np.isnan(subset), axis=0).astype(np.int32, copy=False)
            group_mean = np.full(subset.shape[1], np.nan, dtype=np.float32)
            present = valid_counts > 0
            if np.any(present):
                sums = np.nansum(subset[:, present], axis=0, dtype=np.float64)
                group_mean[present] = (sums / valid_counts[present]).astype(np.float32, copy=False)
            for chrom in layout.chroms:
                bin_offset = int(layout.chrom_bin_offsets[chrom])
                bin_count = int(layout.chrom_bin_counts[chrom])
                if bin_count <= 0:
                    continue
                values = group_mean[bin_offset : bin_offset + bin_count]
                present = ~np.isnan(values)
                if not np.any(present):
                    continue
                local_bins = np.flatnonzero(present)
                frames.append(
                    pd.DataFrame(
                        {
                            "track": key.name(),
                            "assay": key.assay,
                            "haplotype": key.haplotype,
                            "strand": key.strand,
                            "group_field": "tissue_name",
                            "group_value": group,
                            "chrom": chrom,
                            "bin_index": local_bins.astype(np.int64, copy=False),
                            "bin_start": (local_bins.astype(np.int64, copy=False) * int(bin_length)),
                            "mean_methylation_fraction": values[present].astype(np.float32, copy=False),
                            "n_samples": int(sample_idx.size),
                        }
                    )
                )

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_path, sep="\t", index=False, compression="gzip")


def _serialize_matrix(matrix: np.ndarray) -> str:
    return base64.b64encode(np.asarray(matrix, dtype=np.uint16).tobytes(order="C")).decode("ascii")


def _json_script(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


def write_npz_bundle(
    outdir: str,
    matrices: dict[str, np.ndarray],
    layout: BinLayout,
    metadata: pd.DataFrame,
) -> str:
    payload = {
        "chroms": np.asarray(layout.chroms, dtype=object),
        "chrom_bin_counts": np.asarray([layout.chrom_bin_counts[chrom] for chrom in layout.chroms], dtype=np.int32),
        "chrom_bin_offsets": np.asarray([layout.chrom_bin_offsets[chrom] for chrom in layout.chroms], dtype=np.int32),
        "sample_name": metadata["sample_name"].astype(str).to_numpy(dtype=object),
        "display_id": metadata["display_id"].astype(str).to_numpy(dtype=object),
    }
    for track_name, matrix in matrices.items():
        payload[f"track__{track_name}"] = np.asarray(matrix, dtype=np.uint16)
    out_path = os.path.join(outdir, "binned_profiles.npz")
    np.savez_compressed(out_path, **payload)
    return out_path


def _groupable_metadata_columns(metadata: pd.DataFrame) -> list[str]:
    preferred = [
        "tissue_name",
        "tissue_broad",
        "analysis_group",
        "preservation",
        "center",
        "technology",
        "sex",
        "donor",
        "core",
    ]
    out = [col for col in preferred if col in metadata.columns and _is_reasonable_category(metadata[col])]
    for col in metadata.columns:
        if col in out or col in {"sample_name", "display_id", "sample_root"}:
            continue
        if _is_reasonable_category(metadata[col]):
            out.append(col)
    return out


def _metadata_records_for_html(metadata: pd.DataFrame, groupable_cols: list[str]) -> list[dict[str, object]]:
    keep_cols = _dedupe_keep_order(
        [
            "sample_name",
            "display_id",
            "sample_root",
            "sample_id",
            "sample_label",
            "tissue_name",
            "tissue_broad",
            "analysis_group",
            "preservation",
            "center",
            "technology",
            "sex",
            "donor",
            "core",
        ]
        + groupable_cols
    )
    keep_cols = [col for col in keep_cols if col in metadata.columns]
    return metadata[keep_cols].fillna("").to_dict(orient="records")


def write_html(
    out_html: str,
    *,
    matrices: dict[str, np.ndarray],
    selected_tracks: list[TrackKey],
    metadata: pd.DataFrame,
    layout: BinLayout,
    bin_length: int,
    title: str,
) -> None:
    groupable_cols = _groupable_metadata_columns(metadata)
    default_group_by = "tissue_name" if "tissue_name" in groupable_cols else (groupable_cols[0] if groupable_cols else None)
    track_records = [
        {
            "name": key.name(),
            "assay": key.assay,
            "haplotype": key.haplotype,
            "strand": key.strand,
            "label": f"{key.assay} / {key.haplotype} / {key.strand}",
            "matrix_b64": _serialize_matrix(matrices[key.name()]),
        }
        for key in selected_tracks
    ]
    payload = {
        "title": title,
        "bin_length": int(bin_length),
        "missing_value": int(MATRIX_MISSING),
        "sample_count": int(len(metadata)),
        "chroms": layout.chroms,
        "chrom_bin_counts": [int(layout.chrom_bin_counts[chrom]) for chrom in layout.chroms],
        "chrom_bin_offsets": [int(layout.chrom_bin_offsets[chrom]) for chrom in layout.chroms],
        "groupable_columns": groupable_cols,
        "default_group_by": default_group_by,
        "metadata_records": _metadata_records_for_html(metadata, groupable_cols),
        "tracks": track_records,
    }

    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__MDB_VIZ_TITLE__</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --ink: #18314f;
      --muted: #60748a;
      --line: #d4ddd8;
      --paper: #f4f1e8;
      --panel: #fffdf8;
      --accent: #1d8a7a;
      --accent-2: #d86f34;
      --shadow: rgba(24, 49, 79, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Source Sans Pro", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(29, 138, 122, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(216, 111, 52, 0.12), transparent 24%),
        linear-gradient(180deg, #faf7f0 0%, var(--paper) 100%);
    }}
    .shell {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 24px 18px 28px;
    }}
    .hero {{
      display: grid;
      gap: 10px;
      margin-bottom: 18px;
    }}
    .kicker {{
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-size: 12px;
      color: var(--accent);
      font-weight: 700;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(26px, 3vw, 42px);
      line-height: 1.05;
    }}
    .lede {{
      margin: 0;
      max-width: 1000px;
      font-size: 15px;
      color: var(--muted);
    }}
    .layout {{
      display: grid;
      gap: 18px;
      grid-template-columns: 340px minmax(0, 1fr);
      align-items: start;
    }}
    .panel {{
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 18px 46px var(--shadow);
      backdrop-filter: blur(8px);
    }}
    .controls {{
      padding: 18px;
      position: sticky;
      top: 14px;
    }}
    .control-block + .control-block {{
      margin-top: 14px;
    }}
    .control-label {{
      display: block;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    select, button {{
      width: 100%;
      border: 1px solid #bcc9bf;
      border-radius: 12px;
      background: #fffef9;
      color: var(--ink);
      padding: 10px 12px;
      font-size: 14px;
    }}
    select[multiple] {{
      min-height: 184px;
      padding: 8px;
    }}
    button {{
      cursor: pointer;
      font-weight: 700;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }}
    button:hover {{
      transform: translateY(-1px);
      border-color: var(--accent);
    }}
    .button-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }}
    .summary {{
      margin-top: 16px;
      padding: 14px 14px 12px;
      border-radius: 14px;
      background: linear-gradient(135deg, rgba(29, 138, 122, 0.08), rgba(216, 111, 52, 0.07));
      border: 1px solid rgba(29, 138, 122, 0.12);
      font-size: 13px;
      color: var(--muted);
    }}
    .summary strong {{
      color: var(--ink);
    }}
    .viz {{
      padding: 10px 10px 14px;
    }}
    #plot {{
      min-height: 840px;
    }}
    .note {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
      line-height: 1.5;
    }}
    @media (max-width: 1080px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        position: static;
      }}
      #plot {{
        min-height: 680px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="kicker">mdb viz</div>
      <h1>__MDB_VIZ_TITLE__</h1>
      <p class="lede">Binned methylation profiles rendered as a heatmap. Choose a track, chromosome, and grouping field, then view merged tissue-level rows, selected sample rows, or both in one matrix.</p>
    </section>
    <div class="layout">
      <aside class="panel controls">
        <div class="control-block">
          <label class="control-label" for="trackSelect">Track</label>
          <select id="trackSelect"></select>
        </div>
        <div class="control-block">
          <label class="control-label" for="chromSelect">Chromosome</label>
          <select id="chromSelect"></select>
        </div>
        <div class="control-block">
          <label class="control-label" for="modeSelect">View</label>
          <select id="modeSelect">
            <option value="merged" selected>Merged by metadata</option>
            <option value="samples">Selected samples</option>
            <option value="both">Merged + selected samples</option>
          </select>
        </div>
        <div class="control-block">
          <label class="control-label" for="groupBySelect">Group By</label>
          <select id="groupBySelect"></select>
        </div>
        <div class="control-block">
          <label class="control-label" for="groupValuesSelect">Group Values</label>
          <select id="groupValuesSelect" multiple></select>
          <div class="button-row">
            <button id="selectAllGroups" type="button">All Groups</button>
            <button id="clearGroups" type="button">Clear Groups</button>
          </div>
        </div>
        <div class="control-block">
          <label class="control-label" for="sampleSelect">Samples</label>
          <select id="sampleSelect" multiple></select>
          <div class="button-row">
            <button id="selectVisibleSamples" type="button">Visible Samples</button>
            <button id="clearSamples" type="button">Clear Samples</button>
          </div>
        </div>
        <div class="summary" id="summaryBox"></div>
        <div class="note">Rows represent merged metadata groups or selected samples. Use the metadata and sample filters to keep the matrix readable. The full binned matrices remain available in the output directory as compressed `.npz` data.</div>
      </aside>
      <section class="panel viz">
        <div id="plot"></div>
      </section>
    </div>
  </div>
  <script>
    const DATA = __MDB_VIZ_PAYLOAD__;
    const trackSelect = document.getElementById("trackSelect");
    const chromSelect = document.getElementById("chromSelect");
    const modeSelect = document.getElementById("modeSelect");
    const groupBySelect = document.getElementById("groupBySelect");
    const groupValuesSelect = document.getElementById("groupValuesSelect");
    const sampleSelect = document.getElementById("sampleSelect");
    const summaryBox = document.getElementById("summaryBox");
    const decodedMatrices = new Map();

    function decodeBase64ToUint16(b64) {{
      const raw = atob(b64);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i += 1) {{
        bytes[i] = raw.charCodeAt(i);
      }}
      return new Uint16Array(bytes.buffer);
    }}

    function trackInfoMap() {{
      const out = new Map();
      for (const track of DATA.tracks) {{
        out.set(track.name, track);
      }}
      return out;
    }}

    const TRACKS = trackInfoMap();

    function getMatrix(trackName) {{
      if (!decodedMatrices.has(trackName)) {{
        const track = TRACKS.get(trackName);
        decodedMatrices.set(trackName, decodeBase64ToUint16(track.matrix_b64));
      }}
      return decodedMatrices.get(trackName);
    }}

    function sampleMeta(index) {{
      return DATA.metadata_records[index];
    }}

    function selectedValues(selectEl) {{
      return Array.from(selectEl.selectedOptions).map((option) => option.value);
    }}

    function setAllSelected(selectEl, values) {{
      const wanted = new Set(values);
      for (const option of selectEl.options) {{
        option.selected = wanted.has(option.value);
      }}
    }}

    function fillSelect(selectEl, items, selected) {{
      selectEl.innerHTML = "";
      const selectedSet = new Set(selected || []);
      for (const item of items) {{
        const option = document.createElement("option");
        if (typeof item === "string") {{
          option.value = item;
          option.textContent = item;
          option.selected = selectedSet.size === 0 ? true : selectedSet.has(item);
        }} else {{
          option.value = item.value;
          option.textContent = item.label;
          option.selected = selectedSet.size === 0 ? !!item.selected : selectedSet.has(item.value);
        }}
        selectEl.appendChild(option);
      }}
    }}

    function chromSlice(chrom) {{
      const idx = DATA.chroms.indexOf(chrom);
      const start = DATA.chrom_bin_offsets[idx];
      const count = DATA.chrom_bin_counts[idx];
      return {{ index: idx, start, end: start + count, count }};
    }}

    function binPositions(count) {{
      const out = new Array(count);
      for (let i = 0; i < count; i += 1) {{
        out[i] = i * DATA.bin_length;
      }}
      return out;
    }}

    function valueAt(matrix, sampleIdx, binIdx) {{
      const raw = matrix[sampleIdx * totalBins() + binIdx];
      return raw === DATA.missing_value ? null : raw / 10000.0;
    }}

    function totalBins() {{
      const counts = DATA.chrom_bin_counts;
      return counts.reduce((sum, value) => sum + value, 0);
    }}

    function groupValues(groupBy) {{
      const values = [];
      for (const record of DATA.metadata_records) {{
        const value = String(record[groupBy] || "").trim();
        if (value) {{
          values.push(value);
        }}
      }}
      return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
    }}

    function sampleMatchesGroup(record, groupBy, groups) {{
      if (!groupBy || groups.length === 0) {{
        return true;
      }}
      const value = String(record[groupBy] || "").trim();
      return groups.includes(value);
    }}

    function refreshGroupOptions() {{
      const groupBy = groupBySelect.value;
      const current = selectedValues(groupValuesSelect);
      const values = groupBy ? groupValues(groupBy) : [];
      fillSelect(groupValuesSelect, values, current.filter((value) => values.includes(value)));
      if (groupValuesSelect.selectedOptions.length === 0) {{
        setAllSelected(groupValuesSelect, values);
      }}
    }}

    function refreshSampleOptions() {{
      const groupBy = groupBySelect.value;
      const groups = selectedValues(groupValuesSelect);
      const current = new Set(selectedValues(sampleSelect));
      sampleSelect.innerHTML = "";
      for (let idx = 0; idx < DATA.metadata_records.length; idx += 1) {{
        const record = sampleMeta(idx);
        if (!sampleMatchesGroup(record, groupBy, groups)) {{
          continue;
        }}
        const label = record.display_id || record.sample_root || record.sample_name;
        const extra = groupBy ? " | " + String(record[groupBy] || "NA") : "";
        const option = document.createElement("option");
        option.value = String(idx);
        option.textContent = label + extra;
        option.selected = current.has(String(idx));
        sampleSelect.appendChild(option);
      }}
    }}

    function initControls() {{
      fillSelect(trackSelect, DATA.tracks.map((track, idx) => ({
        value: track.name,
        label: track.label,
        selected: idx === 0
      })), [DATA.tracks[0].name]);
      fillSelect(chromSelect, DATA.chroms, [DATA.chroms[0]]);
      if (DATA.groupable_columns.length > 0) {{
        fillSelect(groupBySelect, DATA.groupable_columns, [DATA.default_group_by || DATA.groupable_columns[0]]);
      }} else {{
        fillSelect(groupBySelect, [], []);
        groupBySelect.disabled = true;
        groupValuesSelect.disabled = true;
      }}
      refreshGroupOptions();
      refreshSampleOptions();
    }}

    function sampleIndicesForGroup(groupBy, value) {{
      const out = [];
      for (let idx = 0; idx < DATA.metadata_records.length; idx += 1) {{
        const record = sampleMeta(idx);
        if (String(record[groupBy] || "").trim() === value) {{
          out.push(idx);
        }}
      }}
      return out;
    }}

    function computeMeanRow(matrix, sampleIndices, start, end) {{
      const values = new Array(end - start);
      for (let bin = start; bin < end; bin += 1) {{
        let sum = 0.0;
        let count = 0;
        for (const sampleIdx of sampleIndices) {{
          const raw = matrix[sampleIdx * totalBins() + bin];
          if (raw !== DATA.missing_value) {{
            sum += raw / 10000.0;
            count += 1;
          }}
        }}
        values[bin - start] = count > 0 ? sum / count : null;
      }}
      return values;
    }}

    function buildMergedRows(matrix, chrom, groupBy, groups) {{
      if (!groupBy) {{
        return [];
      }}
      const slice = chromSlice(chrom);
      const rows = [];
      groups.forEach((groupValue) => {{
        const sampleIndices = sampleIndicesForGroup(groupBy, groupValue);
        if (sampleIndices.length === 0) {{
          return;
        }}
        rows.push({{
          label: "[group] " + groupValue + " (n=" + sampleIndices.length + ")",
          values: computeMeanRow(matrix, sampleIndices, slice.start, slice.end),
          kind: "group",
          count: sampleIndices.length,
        }});
      }});
      return rows;
    }}

    function buildSampleRows(matrix, chrom, sampleIndices) {{
      const slice = chromSlice(chrom);
      const rows = [];
      sampleIndices.forEach((sampleIdx) => {{
        const record = sampleMeta(sampleIdx);
        const values = new Array(slice.count);
        for (let bin = slice.start; bin < slice.end; bin += 1) {{
          values[bin - slice.start] = valueAt(matrix, sampleIdx, bin);
        }}
        const label = record.display_id || record.sample_root || record.sample_name;
        rows.push({{
          label: "[sample] " + label,
          values,
          kind: "sample",
          count: 1,
        }});
      }});
      return rows;
    }}

    function heatmapHeight(rowCount) {{
      return Math.max(420, Math.min(1600, 220 + rowCount * 22));
    }}

    function refreshSummary(groupBy, groups, sampleIndices, rowCount) {{
      const lines = [];
      lines.push("<strong>" + DATA.sample_count + "</strong> samples in matrix");
      lines.push("<strong>" + DATA.bin_length.toLocaleString() + "</strong> bp bins");
      if (groupBy) {{
        lines.push("Grouping field: <strong>" + groupBy + "</strong>");
      }}
      if (groups.length > 0) {{
        lines.push("Merged groups shown: <strong>" + groups.length + "</strong>");
      }}
      lines.push("Selected samples: <strong>" + sampleIndices.length + "</strong>");
      lines.push("Heatmap rows: <strong>" + rowCount + "</strong>");
      summaryBox.innerHTML = lines.join("<br>");
    }}

    function render() {{
      const trackName = trackSelect.value;
      const chrom = chromSelect.value;
      const mode = modeSelect.value;
      const groupBy = groupBySelect.value;
      const groups = selectedValues(groupValuesSelect);
      const sampleIndices = selectedValues(sampleSelect).map((value) => Number.parseInt(value, 10)).filter(Number.isFinite);
      const matrix = getMatrix(trackName);
      const rows = [];

      if (mode === "merged" || mode === "both") {{
        rows.push(...buildMergedRows(matrix, chrom, groupBy, groups));
      }}
      if (mode === "samples" || mode === "both") {{
        rows.push(...buildSampleRows(matrix, chrom, sampleIndices));
      }}

      const track = TRACKS.get(trackName);
      const slice = chromSlice(chrom);
      const x = binPositions(slice.count);
      const y = rows.map((row) => row.label);
      const z = rows.map((row) => row.values);
      const rowKinds = rows.map((row) => row.kind);
      const rowCounts = rows.map((row) => row.count);
      const traces = rows.length > 0 ? [{{
        type: "heatmap",
        x,
        y,
        z,
        zmin: 0,
        zmax: 1,
        zsmooth: false,
        hoverongaps: false,
        colorscale: [
          [0.0, "#fffaf0"],
          [0.15, "#f4d58d"],
          [0.35, "#d68c45"],
          [0.55, "#7fb285"],
          [0.75, "#2d6a73"],
          [1.0, "#18314f"]
        ],
        colorbar: {{
          title: "Methylation Fraction",
          thickness: 16,
          len: 0.86,
        }},
        customdata: rows.map((row, idx) => row.values.map(() => [rowKinds[idx], rowCounts[idx]])),
        hovertemplate:
          "Row: %{y}<br>" +
          "Bin Start: %{x:,d}<br>" +
          "Methylation: %{z:.4f}<br>" +
          "Row Type: %{customdata[0]}<br>" +
          "Members: %{customdata[1]}<extra></extra>"
      }}] : [];

      const layout = {{
        template: "none",
        paper_bgcolor: "#fffdf8",
        plot_bgcolor: "#fffefb",
        height: heatmapHeight(rows.length),
        title: {{
          text: track.label + " | " + chrom + " | " + DATA.bin_length.toLocaleString() + " bp bins",
          x: 0.01,
          y: 0.98,
          xanchor: "left"
        }},
        margin: {{ l: 180, r: 80, t: 86, b: 74 }},
        font: {{ family: "Source Sans Pro, Arial, sans-serif", size: 14, color: "#18314f" }},
        xaxis: {{
          title: "Bin Start (bp)",
          gridcolor: "#dde5df",
          zeroline: false,
          tickformat: ",d"
        }},
        yaxis: {{
          title: "Groups / Samples",
          autorange: "reversed",
          gridcolor: "#f0f3ef",
          zeroline: false,
          automargin: true
        }},
        hoverlabel: {{
          bgcolor: "rgba(255,255,255,0.97)",
          bordercolor: "#c7d4cb",
          font: {{ color: "#18314f" }}
        }},
        annotations: rows.length > 0 ? [] : [{{
          xref: "paper",
          yref: "paper",
          x: 0.5,
          y: 0.5,
          text: "No rows selected for this view",
          showarrow: false,
          font: {{ size: 18, color: "#60748a" }}
        }}]
      }};

      Plotly.react("plot", traces, layout, {{
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["select2d", "lasso2d", "hoverClosestCartesian", "hoverCompareCartesian"]
      }});
      refreshSummary(groupBy, groups, sampleIndices, rows.length);
    }}

    document.getElementById("selectAllGroups").addEventListener("click", () => {{
      setAllSelected(groupValuesSelect, Array.from(groupValuesSelect.options).map((option) => option.value));
      refreshSampleOptions();
      render();
    }});
    document.getElementById("clearGroups").addEventListener("click", () => {{
      setAllSelected(groupValuesSelect, []);
      refreshSampleOptions();
      render();
    }});
    document.getElementById("selectVisibleSamples").addEventListener("click", () => {{
      setAllSelected(sampleSelect, Array.from(sampleSelect.options).map((option) => option.value));
      render();
    }});
    document.getElementById("clearSamples").addEventListener("click", () => {{
      setAllSelected(sampleSelect, []);
      render();
    }});

    trackSelect.addEventListener("change", render);
    chromSelect.addEventListener("change", render);
    modeSelect.addEventListener("change", render);
    groupBySelect.addEventListener("change", () => {{
      refreshGroupOptions();
      refreshSampleOptions();
      render();
    }});
    groupValuesSelect.addEventListener("change", () => {{
      refreshSampleOptions();
      render();
    }});
    sampleSelect.addEventListener("change", render);

    initControls();
    render();
  </script>
</body>
</html>
"""

    html = template.replace("{{", "{").replace("}}", "}")
    html = html.replace("__MDB_VIZ_TITLE__", title)
    html = html.replace("__MDB_VIZ_PAYLOAD__", _json_script(payload))

    with open(out_html, "w") as fh:
        fh.write(html)


def viz(
    inputs: list[str],
    outdir: str,
    *,
    metadata: str | None = None,
    assay: str = "5mC,5hmC",
    haplotype: str = "combined",
    strand: str = "combined",
    bin_length: int = 100_000,
    workers: int = 1,
    verbose: bool = False,
) -> dict[str, str]:
    if int(bin_length) <= 0:
        raise ValueError("bin_length must be > 0")

    os.makedirs(outdir, exist_ok=True)
    logger = setup_logging(outdir, verbose=verbose)
    started = time.time()

    mode, bundle_paths, cohort_path = resolve_inputs(list(inputs))
    logger.info("Resolved input mode=%s cohort=%s bundles=%d", mode, cohort_path, len(bundle_paths))

    if mode == "cohort":
        selected_tracks = select_tracks_for_cohort(cohort_path, assay=assay, haplotype=haplotype, strand=strand)
    else:
        selected_tracks = select_tracks_for_bundles(bundle_paths, assay=assay, haplotype=haplotype, strand=strand)
    logger.info("Selected tracks: %s", ", ".join(key.name() for key in selected_tracks))

    sample_entries = resolve_sample_entries(mode, bundle_paths, cohort_path, selected_tracks)
    metadata_df = align_metadata([entry["sample_id"] for entry in sample_entries], metadata, logger)
    metadata_out = os.path.join(outdir, "sample_metadata_aligned.tsv")
    metadata_df.to_csv(metadata_out, sep="\t", index=False)

    layout, matrices = aggregate_bundle_profiles(
        sample_entries,
        selected_tracks,
        bin_length=int(bin_length),
        workers=max(int(workers), 1),
        logger=logger,
    )

    npz_out = write_npz_bundle(outdir, matrices, layout, metadata_df)
    group_out = os.path.join(outdir, "tissue_name_group_profiles.tsv.gz")
    _prepare_group_profile_tables(
        matrices,
        metadata_df,
        layout,
        selected_tracks,
        bin_length=int(bin_length),
        out_path=group_out,
    )

    title = f"Methylation Profiles ({int(bin_length):,} bp bins)"
    html_out = os.path.join(outdir, "methylation_viz.html")
    write_html(
        html_out,
        matrices=matrices,
        selected_tracks=selected_tracks,
        metadata=metadata_df,
        layout=layout,
        bin_length=int(bin_length),
        title=title,
    )

    manifest = {
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": [entry["bundle_path"] for entry in sample_entries],
        "cohort_path": cohort_path,
        "metadata": metadata,
        "bin_length": int(bin_length),
        "workers": int(workers),
        "n_samples": int(len(sample_entries)),
        "tracks": [key.name() for key in selected_tracks],
        "chroms": layout.chroms,
        "chrom_bin_counts": {chrom: int(layout.chrom_bin_counts[chrom]) for chrom in layout.chroms},
        "total_bins": int(layout.total_bins),
        "outputs": {
            "html": html_out,
            "npz": npz_out,
            "metadata_tsv": metadata_out,
            "tissue_group_profiles_tsv_gz": group_out if os.path.isfile(group_out) else None,
        },
        "runtime_seconds": round(time.time() - started, 3),
    }
    manifest_out = os.path.join(outdir, "viz_manifest.json")
    with open(manifest_out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote %s", html_out)
    logger.info("Completed mdb viz in %.2fs", time.time() - started)
    return manifest["outputs"]


def viz_main(args) -> None:
    viz(
        inputs=list(args.inputs),
        outdir=args.outdir,
        metadata=getattr(args, "metadata", None),
        assay=getattr(args, "assay", "5mC,5hmC"),
        haplotype=getattr(args, "haplotype", "combined"),
        strand=getattr(args, "strand", "combined"),
        bin_length=int(args.bin_length),
        workers=max(int(getattr(args, "workers", 1)), 1),
        verbose=bool(getattr(args, "verbose", False)),
    )


__all__ = ["viz", "viz_main"]
