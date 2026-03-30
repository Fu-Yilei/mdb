#!/usr/bin/env python3
"""
PCA on ONT ASM segment BEDs using DMR regions as features.

Input segment BEDs are expected to follow the modkit DMR segment schema, for example:
  #chrom  chrom_start  chrom_end  name  ...  effect_size  cohen_h  ...

By default, DMR regions are defined as the merged union of rows with ``name == different``
across the input segment BEDs. The default feature mode is binary DMR-location presence at
those cohort DMR regions. An alternate mode projects a requested segment metric onto the
same regions before PCA.
"""

from __future__ import annotations

import glob
import json
import os
import time
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mdb.pca import (
    build_color_styles,
    fit_ipca_sample_coords,
    maybe_merge_metadata,
    maybe_use_concise_sample_ids,
    pick_pairplot_hue,
    plotly_color_options,
    plotly_png_ok,
    setup_logging,
    write_pairplot_png,
    write_pairplot_with_styles,
    write_scatter_with_styles,
)


SEGMENT_METRIC_COLUMNS = ("effect_size", "cohen_h", "score")
FEATURE_MODES = ("dmr_location", "segment_metric")
SEX_CHROMS = {"chrx", "chry", "x", "y"}


@dataclass(frozen=True)
class AsmInput:
    sample_id: str
    path: str


class ArrayMatrixSource:
    def __init__(self, matrix: np.ndarray):
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got shape={matrix.shape}")
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.shape = self.matrix.shape

    def iter_blocks(self, batch_rows: int):
        n_rows = int(self.shape[0])
        for start in range(0, n_rows, batch_rows):
            yield self.matrix[start : min(start + batch_rows, n_rows), :]

    def read_rows(self, row_idx: np.ndarray) -> np.ndarray:
        return self.matrix[row_idx, :]

    def close(self) -> None:
        return None


def _tissue_broad_from_name(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    tissue = str(value).strip()
    if not tissue:
        return pd.NA

    norm = tissue.lower()
    if norm.startswith("brain"):
        return "Brain / CNS"
    if norm.startswith("skin") or "dermal fibroblast" in norm:
        return "Skin / Connective"
    if norm.startswith("muscle"):
        return "Musculoskeletal"
    if norm.startswith("heart") or norm.startswith("aorta"):
        return "Cardiovascular"
    if norm.startswith("blood"):
        return "Blood / Immune"
    if norm.startswith("lung"):
        return "Respiratory"
    if norm.startswith("colon") or norm.startswith("esophagus") or norm.startswith("liver"):
        return "Digestive"
    if norm.startswith("adrenal"):
        return "Endocrine"
    if norm.startswith("ovary") or norm.startswith("testis"):
        return "Reproductive"
    return "Other"


def add_broad_tissue_category(metadata_df: pd.DataFrame) -> pd.DataFrame:
    if "tissue_name" not in metadata_df.columns:
        return metadata_df

    out = metadata_df.copy()
    derived = out["tissue_name"].map(_tissue_broad_from_name)
    if "tissue_broad" in out.columns:
        out["tissue_broad"] = out["tissue_broad"].where(
            out["tissue_broad"].notna() & (out["tissue_broad"].astype(str).str.strip() != ""),
            derived,
        )
        return out

    insert_at = int(out.columns.get_loc("tissue_name")) + 1
    out.insert(insert_at, "tissue_broad", derived)
    return out


def _read_metadata_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    return add_broad_tissue_category(df)


def _read_manifest(path: str) -> list[str]:
    out: list[str] = []
    with open(path) as handle:
        for line in handle:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _sample_id_from_path(path: str) -> str:
    name = os.path.basename(path)
    suffix = ".segments.bed"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return os.path.basename(os.path.dirname(path.rstrip("/"))) or name


def discover_segment_beds(inputs: str | Iterable[str]) -> list[AsmInput]:
    if isinstance(inputs, (str, os.PathLike)):
        queue = [str(inputs)]
    else:
        queue = [str(x) for x in inputs]

    seen_paths: set[str] = set()
    ordered_paths: list[str] = []

    while queue:
        token = queue.pop(0)
        if not token:
            continue

        token_path = os.path.expanduser(token)
        if os.path.isfile(token_path):
            if token_path.endswith(".txt"):
                queue.extend(_read_manifest(token_path))
                continue
            if token_path.endswith(".segments.bed"):
                real = os.path.realpath(token_path)
                if real not in seen_paths:
                    seen_paths.add(real)
                    ordered_paths.append(real)
                continue

        if os.path.isdir(token_path):
            matches = sorted(
                {
                    *glob.glob(os.path.join(token_path, "*.segments.bed")),
                    *glob.glob(os.path.join(token_path, "*", "*.segments.bed")),
                }
            )
            for match in matches:
                real = os.path.realpath(match)
                if real not in seen_paths:
                    seen_paths.add(real)
                    ordered_paths.append(real)
            continue

        glob_matches = sorted(glob.glob(token_path))
        if glob_matches:
            queue.extend(glob_matches)
            continue

        raise FileNotFoundError(f"Could not resolve ASM segment input: {token}")

    if not ordered_paths:
        raise FileNotFoundError("No ONT ASM segment BEDs were discovered.")

    items = [AsmInput(sample_id=_sample_id_from_path(path), path=path) for path in ordered_paths]
    sample_ids = [item.sample_id for item in items]
    if len(sample_ids) != len(set(sample_ids)):
        dupes = sorted({sid for sid in sample_ids if sample_ids.count(sid) > 1})
        raise ValueError(f"Duplicate sample ids derived from segment BED filenames: {dupes}")
    return items


def _select_metadata_rows(metadata_df: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame | None:
    if "id" not in metadata_df.columns:
        if len(metadata_df) != len(sample_ids):
            return None
        out = metadata_df.copy()
        out.insert(0, "id", sample_ids)
        return out.reset_index(drop=True)

    out = metadata_df.copy()
    out["id"] = out["id"].astype(str)
    id_to_order = {sid: i for i, sid in enumerate(sample_ids)}
    selected = out[out["id"].isin(id_to_order)].copy()
    if len(selected) != len(sample_ids):
        return None
    selected["_order"] = selected["id"].map(id_to_order)
    selected = selected.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)
    return selected


def _metadata_candidate_paths(outdir: str, input_paths: list[str]) -> list[str]:
    p = Path(outdir).resolve()
    candidates: list[Path] = []
    for base in [p, *p.parents[:5]]:
        candidates.extend(
            [
                base / "metadata_selected_ont.tsv",
                base / "ont_5hmC" / "metadata_selected_ont.tsv",
                base / "ont" / "metadata_selected_ont.tsv",
            ]
        )
    for input_path in input_paths:
        pp = Path(input_path).resolve()
        for parent in [pp.parent, *pp.parents]:
            if parent.name == "methylation":
                candidates.extend(
                    [
                        parent / "smdb_all" / "pca" / "ont_5hmC" / "metadata_selected_ont.tsv",
                        parent / "smdb_all" / "pca" / "ont" / "metadata_selected_ont.tsv",
                    ]
                )
    seen: set[str] = set()
    out: list[str] = []
    for cand in candidates:
        s = str(cand)
        if s not in seen and cand.is_file():
            seen.add(s)
            out.append(s)
    return out


def resolve_metadata_manifest(
    metadata_path: str | None,
    *,
    sample_ids: list[str],
    outdir: str,
    input_paths: list[str],
    logger,
) -> tuple[str | None, pd.DataFrame | None]:
    if metadata_path:
        df = _read_metadata_table(metadata_path)
        selected = _select_metadata_rows(df, sample_ids)
        if selected is None:
            out_path = os.path.join(outdir, os.path.basename(metadata_path))
            df.to_csv(out_path, sep="\t", index=False)
            logger.info(
                f"Provided metadata could not be aligned to ASM samples for manifest export; "
                f"using augmented copy for merge: {out_path}"
            )
            return out_path, None
        out_path = os.path.join(outdir, os.path.basename(metadata_path))
        selected.to_csv(out_path, sep="\t", index=False)
        logger.info(f"Wrote selected metadata manifest: {out_path}")
        return out_path, selected

    for candidate in _metadata_candidate_paths(outdir, input_paths):
        df = _read_metadata_table(candidate)
        selected = _select_metadata_rows(df, sample_ids)
        if selected is None:
            continue
        out_path = os.path.join(outdir, os.path.basename(candidate))
        selected.to_csv(out_path, sep="\t", index=False)
        logger.info(f"Auto-selected metadata manifest: {candidate}")
        logger.info(f"Wrote selected metadata manifest: {out_path}")
        return out_path, selected

    logger.info("No alignable metadata manifest was discovered; HTML dropdown will be limited.")
    return None, None


def _read_segment_subset(path: str, columns: list[str]) -> pd.DataFrame:
    usecols = list(dict.fromkeys(columns))
    df = pd.read_csv(path, sep="\t", usecols=usecols)
    rename_map = {"#chrom": "chrom", "chrom_start": "start", "chrom_end": "end"}
    df = df.rename(columns=rename_map)
    if {"chrom", "start", "end"} - set(df.columns):
        raise ValueError(f"Segment BED missing required columns in {path}: {columns}")
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(np.int64)
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype(np.int64)
    df = df[df["end"] > df["start"]].copy()
    return df.sort_values(["chrom", "start", "end"], kind="mergesort", ignore_index=True)


def read_external_regions(path: str) -> pd.DataFrame:
    with open(path) as handle:
        first = handle.readline()
    header = 0 if first.startswith("#") or first.lower().startswith("chrom") else None
    df = pd.read_csv(path, sep="\t", header=header)
    if df.shape[1] < 3:
        raise ValueError(f"DMR region BED must have at least 3 columns: {path}")
    if header is None:
        df = df.iloc[:, :3].copy()
        df.columns = ["chrom", "start", "end"]
    else:
        cols = list(df.columns[:3])
        rename = {cols[0]: "chrom", cols[1]: "start", cols[2]: "end"}
        df = df.iloc[:, :3].rename(columns=rename)
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(np.int64)
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype(np.int64)
    return df[df["end"] > df["start"]].sort_values(["chrom", "start", "end"], kind="mergesort", ignore_index=True)


def merge_regions(regions: pd.DataFrame, merge_gap: int = 0) -> pd.DataFrame:
    if regions.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "region_id"])

    merged_rows: list[tuple[str, int, int]] = []
    for chrom, sub in regions.groupby("chrom", sort=True):
        starts = sub["start"].to_numpy(dtype=np.int64, copy=False)
        ends = sub["end"].to_numpy(dtype=np.int64, copy=False)
        cur_start = int(starts[0])
        cur_end = int(ends[0])
        for start, end in zip(starts[1:], ends[1:]):
            start_i = int(start)
            end_i = int(end)
            if start_i <= cur_end + int(merge_gap):
                if end_i > cur_end:
                    cur_end = end_i
            else:
                merged_rows.append((chrom, cur_start, cur_end))
                cur_start = start_i
                cur_end = end_i
        merged_rows.append((chrom, cur_start, cur_end))

    out = pd.DataFrame(merged_rows, columns=["chrom", "start", "end"])
    out["region_id"] = [f"DMR{i+1:06d}" for i in range(len(out))]
    return out


def filter_regions_exclude_sex_chromosomes(regions: pd.DataFrame) -> pd.DataFrame:
    if regions.empty:
        return regions.copy()
    chrom_norm = regions["chrom"].astype(str).str.lower()
    keep = ~chrom_norm.isin(SEX_CHROMS)
    out = regions.loc[keep].reset_index(drop=True).copy()
    if "region_id" in out.columns:
        out["region_id"] = [f"DMR{i+1:06d}" for i in range(len(out))]
    return out


def load_union_dmr_regions(segment_inputs: list[AsmInput], merge_gap: int = 0) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for item in tqdm(segment_inputs, desc="Collecting DMR regions", unit="sample"):
        df = _read_segment_subset(item.path, ["#chrom", "chrom_start", "chrom_end", "name"])
        diff = df[df["name"].astype(str) == "different"][["chrom", "start", "end"]]
        if not diff.empty:
            parts.append(diff)
    if not parts:
        raise ValueError("No rows with name=different were found across the input ASM segment BEDs.")
    all_regions = pd.concat(parts, ignore_index=True)
    return merge_regions(all_regions, merge_gap=merge_gap)


def build_region_lookup(regions: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for chrom, sub in regions.groupby("chrom", sort=False):
        lookup[chrom] = (
            sub["start"].to_numpy(dtype=np.int64, copy=False),
            sub["end"].to_numpy(dtype=np.int64, copy=False),
            sub.index.to_numpy(dtype=np.int64, copy=False),
        )
    return lookup


def project_segments_to_regions(
    segment_path: str,
    *,
    region_lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_regions: int,
    metric: str,
) -> np.ndarray:
    df = _read_segment_subset(segment_path, ["#chrom", "chrom_start", "chrom_end", metric])
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[np.isfinite(df[metric].to_numpy(dtype=np.float64, copy=False))].copy()

    values = np.full(n_regions, np.nan, dtype=np.float32)
    if df.empty:
        return values

    for chrom, sub in df.groupby("chrom", sort=False):
        if chrom not in region_lookup:
            continue
        region_starts, region_ends, region_rows = region_lookup[chrom]
        seg_starts = sub["start"].to_numpy(dtype=np.int64, copy=False)
        seg_ends = sub["end"].to_numpy(dtype=np.int64, copy=False)
        seg_vals = sub[metric].to_numpy(dtype=np.float64, copy=False)

        accum = np.zeros(region_rows.shape[0], dtype=np.float64)
        covered = np.zeros(region_rows.shape[0], dtype=np.float64)
        r = 0
        s = 0
        while r < region_rows.shape[0] and s < seg_starts.shape[0]:
            r_start = int(region_starts[r])
            r_end = int(region_ends[r])
            s_start = int(seg_starts[s])
            s_end = int(seg_ends[s])

            if s_end <= r_start:
                s += 1
                continue
            if r_end <= s_start:
                r += 1
                continue

            overlap = min(r_end, s_end) - max(r_start, s_start)
            if overlap > 0:
                accum[r] += float(seg_vals[s]) * float(overlap)
                covered[r] += float(overlap)

            if s_end <= r_end:
                s += 1
            else:
                r += 1

        ok = covered > 0
        if np.any(ok):
            out_vals = np.full(region_rows.shape[0], np.nan, dtype=np.float32)
            out_vals[ok] = (accum[ok] / covered[ok]).astype(np.float32)
            values[region_rows] = out_vals

    return values


def project_dmr_locations_to_regions(
    segment_path: str,
    *,
    region_lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_regions: int,
) -> np.ndarray:
    df = _read_segment_subset(segment_path, ["#chrom", "chrom_start", "chrom_end", "name"])
    df = df[df["name"].astype(str) == "different"].copy()

    values = np.zeros(n_regions, dtype=np.float32)
    if df.empty:
        return values

    for chrom, sub in df.groupby("chrom", sort=False):
        if chrom not in region_lookup:
            continue
        region_starts, region_ends, region_rows = region_lookup[chrom]
        seg_starts = sub["start"].to_numpy(dtype=np.int64, copy=False)
        seg_ends = sub["end"].to_numpy(dtype=np.int64, copy=False)

        r = 0
        s = 0
        while r < region_rows.shape[0] and s < seg_starts.shape[0]:
            r_start = int(region_starts[r])
            r_end = int(region_ends[r])
            s_start = int(seg_starts[s])
            s_end = int(seg_ends[s])

            if s_end <= r_start:
                s += 1
                continue
            if r_end <= s_start:
                r += 1
                continue

            if min(r_end, s_end) > max(r_start, s_start):
                values[region_rows[r]] = 1.0

            if s_end <= r_end:
                s += 1
            else:
                r += 1

    return values


def build_asm_region_matrix(
    segment_inputs: list[AsmInput],
    *,
    dmr_regions: pd.DataFrame,
    metric: str,
) -> np.ndarray:
    region_lookup = build_region_lookup(dmr_regions)
    matrix = np.full((len(dmr_regions), len(segment_inputs)), np.nan, dtype=np.float32)
    for col_idx, item in enumerate(
        tqdm(segment_inputs, desc="Projecting samples to DMRs", unit="sample")
    ):
        matrix[:, col_idx] = project_segments_to_regions(
            item.path,
            region_lookup=region_lookup,
            n_regions=len(dmr_regions),
            metric=metric,
        )
    return matrix


def build_dmr_location_matrix(
    segment_inputs: list[AsmInput],
    *,
    dmr_regions: pd.DataFrame,
) -> np.ndarray:
    region_lookup = build_region_lookup(dmr_regions)
    matrix = np.zeros((len(dmr_regions), len(segment_inputs)), dtype=np.float32)
    for col_idx, item in enumerate(
        tqdm(segment_inputs, desc="Projecting DMR locations", unit="sample")
    ):
        matrix[:, col_idx] = project_dmr_locations_to_regions(
            item.path,
            region_lookup=region_lookup,
            n_regions=len(dmr_regions),
        )
    return matrix


def filter_regions_by_support(
    matrix: np.ndarray,
    regions: pd.DataFrame,
    *,
    feature_mode: str,
    min_region_samples: int,
    min_frac_present: float,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    n_samples = int(matrix.shape[1])
    if feature_mode == "dmr_location":
        support_counts = np.sum(matrix > 0, axis=1, dtype=np.int64)
    else:
        support_counts = np.sum(~np.isnan(matrix), axis=1, dtype=np.int64)
    support_frac = support_counts.astype(np.float64) / float(max(n_samples, 1))
    keep = (support_counts >= int(min_region_samples)) & (support_frac >= float(min_frac_present))
    if not np.any(keep):
        raise ValueError(
            "No DMR regions remain after support filtering. "
            f"feature_mode={feature_mode}, min_region_samples={min_region_samples}, "
            f"min_frac_present={min_frac_present}"
        )
    return matrix[keep, :], regions.loc[keep].reset_index(drop=True), support_counts[keep]


def feature_mode_title(feature_mode: str, metric: str) -> str:
    if feature_mode == "dmr_location":
        return "dmr_location"
    return f"segment_metric:{metric}"


def asmpca(
    inputs: str | Iterable[str],
    outdir: str,
    *,
    dmr_regions: str | None = None,
    exclude_sex_chromosomes: bool = False,
    feature_mode: str = "dmr_location",
    metric: str = "effect_size",
    min_region_samples: int = 2,
    min_frac_present: float = 0.1,
    n_pcs: int = 10,
    pairplot_pcs_n: int = 5,
    pairplot_mode: str | None = None,
    pairplot_hue: str | None = None,
    pairplot_diag_kind: str = "kde",
    pairplot_corner: bool = False,
    merge_gap: int = 0,
    metadata: str | None = None,
    batch_rows: int = 50_000,
    seed: int = 1,
    plot_style: str = "studio",
    plot_style_variants: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}. Choose from {FEATURE_MODES}")
    if metric not in SEGMENT_METRIC_COLUMNS:
        raise ValueError(f"Unsupported ASM segment metric: {metric}. Choose from {SEGMENT_METRIC_COLUMNS}")
    if min_region_samples < 1:
        raise ValueError("min_region_samples must be >= 1")

    os.makedirs(outdir, exist_ok=True)
    logger = setup_logging(outdir, verbose)
    t0 = time.time()
    logger.info("==== ASM PCA run started ====")
    logger.info(f"Start time: {datetime.now().isoformat(timespec='seconds')}")

    segment_inputs = discover_segment_beds(inputs)
    logger.info(f"Resolved {len(segment_inputs)} ASM segment BEDs")
    sample_ids = [item.sample_id for item in segment_inputs]
    metadata_path_resolved, _selected_metadata = resolve_metadata_manifest(
        metadata,
        sample_ids=sample_ids,
        outdir=outdir,
        input_paths=[item.path for item in segment_inputs],
        logger=logger,
    )

    if dmr_regions:
        regions = merge_regions(read_external_regions(dmr_regions), merge_gap=merge_gap)
        region_source = dmr_regions
    else:
        regions = load_union_dmr_regions(segment_inputs, merge_gap=merge_gap)
        region_source = "union(name=different)"

    if exclude_sex_chromosomes:
        n_before = int(len(regions))
        regions = filter_regions_exclude_sex_chromosomes(regions)
        logger.info(
            f"Excluded sex chromosomes from DMR regions: kept {len(regions)} of {n_before}"
        )

    if regions.empty:
        raise ValueError("No DMR regions were available for ASM PCA.")

    if feature_mode == "dmr_location":
        matrix = build_dmr_location_matrix(segment_inputs, dmr_regions=regions)
    else:
        matrix = build_asm_region_matrix(segment_inputs, dmr_regions=regions, metric=metric)

    matrix, regions_fit, region_support = filter_regions_by_support(
        matrix,
        regions,
        feature_mode=feature_mode,
        min_region_samples=min_region_samples,
        min_frac_present=min_frac_present,
    )
    logger.info(f"ASM region matrix shape: {matrix.shape} (n_regions x n_samples)")

    source = ArrayMatrixSource(matrix)
    try:
        n_components = max(1, min(int(n_pcs), int(source.shape[0]), int(source.shape[1])))
        if n_components != int(n_pcs):
            logger.info(f"Adjusted n_pcs from {n_pcs} to {n_components} based on matrix shape")

        sample_coords, ipca, row_idx, obs_count = fit_ipca_sample_coords(
            source,
            n_components=n_components,
            frac_cpgs=1.0,
            min_frac_present=0.0,
            batch_rows=batch_rows,
            seed=seed,
            logger=logger,
        )
    finally:
        source.close()

    out = pd.DataFrame(
        {
            "id": sample_ids,
            "path": [item.path for item in segment_inputs],
            "segment_path": [item.path for item in segment_inputs],
            "sample_id": sample_ids,
            "feature_mode": feature_mode,
            "metric": metric,
            "dmr_region_source": region_source,
            "n_obs_regions_for_fit": obs_count,
        }
    )
    for i in range(sample_coords.shape[1]):
        out[f"PC{i+1}"] = sample_coords[:, i]

    out, meta = maybe_merge_metadata(out, metadata_path_resolved, logger=logger)
    out = maybe_use_concise_sample_ids(out)
    out.to_csv(os.path.join(outdir, "embedding.tsv"), sep="\t", index=False)

    dmr_regions_used = regions_fit.loc[row_idx, ["chrom", "start", "end", "region_id"]].copy()
    dmr_regions_used["support_samples"] = region_support[row_idx].astype(np.int64)
    dmr_regions_used.to_csv(
        os.path.join(outdir, "dmr_regions_used.bed"),
        sep="\t",
        index=False,
        header=False,
    )

    params = {
        "input_count": len(segment_inputs),
        "inputs": [item.path for item in segment_inputs],
        "feature_mode": feature_mode,
        "metric": metric,
        "metadata_manifest": metadata_path_resolved,
        "dmr_region_source": region_source,
        "exclude_sex_chromosomes": bool(exclude_sex_chromosomes),
        "merge_gap": int(merge_gap),
        "n_regions_total": int(len(regions)),
        "n_regions_after_support_filter": int(len(regions_fit)),
        "n_regions_used_for_fit": int(len(row_idx)),
        "min_region_samples": int(min_region_samples),
        "effective_min_frac_present": float(min_frac_present),
        "used_shape": tuple(int(x) for x in matrix.shape),
        "n_pcs": int(sample_coords.shape[1]),
        "seed": int(seed),
        "explained_variance_ratio": getattr(ipca, "explained_variance_ratio_", None).tolist()
        if getattr(ipca, "explained_variance_ratio_", None) is not None
        else None,
    }
    with open(os.path.join(outdir, "params.json"), "w") as handle:
        json.dump(params, handle, indent=2)

    png_ok = plotly_png_ok()
    plot_args = Namespace(
        plot_style=plot_style,
        plot_style_variants=plot_style_variants,
        pairplot_mode=pairplot_mode,
        pairplot_hue=pairplot_hue,
        pairplot_diag_kind=pairplot_diag_kind,
        pairplot_corner=pairplot_corner,
    )
    color_cols, hover_cols = plotly_color_options(meta, out, sample_coords.shape[1], False)
    for extra_hover in ("tissue_broad", "feature_mode", "metric", "dmr_region_source", "segment_path"):
        if extra_hover in out.columns and extra_hover not in hover_cols:
            hover_cols.append(extra_hover)
    color_styles = build_color_styles(out, color_cols)
    feature_title = feature_mode_title(feature_mode, metric)
    evr = getattr(ipca, "explained_variance_ratio_", None)
    if evr is not None and len(evr) >= 2:
        x_axis_label = f"PC1 ({float(evr[0]) * 100.0:.2f}% variance explained)"
        y_axis_label = f"PC2 ({float(evr[1]) * 100.0:.2f}% variance explained)"
    else:
        x_axis_label = "PC1"
        y_axis_label = "PC2"
    pairplot_pc_label_map: dict[str, str] = {}
    for i in range(int(pairplot_pcs_n)):
        pc = f"PC{i+1}"
        if evr is not None and i < len(evr):
            pairplot_pc_label_map[pc] = f"{pc} ({float(evr[i]) * 100.0:.2f}%)"
        else:
            pairplot_pc_label_map[pc] = pc

    if sample_coords.shape[1] >= 2:
        write_scatter_with_styles(
            df=out,
            x="PC1",
            y="PC2",
            color_cols=color_cols,
            hover_cols=hover_cols,
            title=f"PCA sample coords (PC1 vs PC2) [ASM:{feature_title}]",
            color_styles=color_styles,
            out_html=os.path.join(outdir, "pca.html"),
            args=plot_args,
            logger=logger,
            png_ok=png_ok,
            out_png=os.path.join(outdir, "pca.png"),
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
        )

    n_pair = max(1, min(int(pairplot_pcs_n), int(sample_coords.shape[1])))
    pcs = tuple(f"PC{i}" for i in range(1, n_pair + 1))
    hue = pick_pairplot_hue(out, meta, plot_args)
    manifest_for_pairplot = out.drop(columns=[f"PC{i+1}" for i in range(sample_coords.shape[1])], errors="ignore")
    write_pairplot_with_styles(
        scores=sample_coords,
        manifest=manifest_for_pairplot,
        out_html=os.path.join(outdir, "pca_pairplot.html"),
        pcs=pcs,
        color_cols=color_cols,
        hover_cols=hover_cols,
        title=f"PCA Pairplot [ASM:{feature_title}]",
        color_styles=color_styles,
        default_color_col=hue,
        corner=bool(pairplot_corner),
        pc_label_map=pairplot_pc_label_map,
        args=plot_args,
    )
    write_pairplot_png(
        scores=sample_coords,
        manifest=manifest_for_pairplot,
        out_png=os.path.join(outdir, "pca_pairplot.png"),
        pcs=pcs,
        hue=hue,
        color_styles=color_styles,
        diag_kind=pairplot_diag_kind,
        corner=bool(pairplot_corner),
        pc_label_map=pairplot_pc_label_map,
        title=f"PCA Pairplot [ASM:{feature_title}]",
    )

    logger.info(f"==== ASM PCA done in {time.time() - t0:.2f}s ====")
    return out


def asmpca_main(args) -> None:
    asmpca(
        inputs=args.inputs,
        outdir=args.outdir,
        dmr_regions=args.dmr_regions,
        exclude_sex_chromosomes=bool(args.exclude_sex_chromosomes),
        feature_mode=args.feature_mode,
        metric=args.metric,
        min_region_samples=args.min_region_samples,
        min_frac_present=args.min_frac_present,
        n_pcs=args.n_pcs,
        pairplot_pcs_n=args.pairplot_pcs_n,
        pairplot_mode=args.pairplot_mode,
        pairplot_hue=args.pairplot_hue,
        pairplot_diag_kind=args.pairplot_diag_kind,
        pairplot_corner=args.pairplot_corner,
        merge_gap=args.merge_gap,
        metadata=args.metadata,
        batch_rows=args.batch_rows,
        seed=args.seed,
        plot_style=args.plot_style,
        plot_style_variants=args.plot_style_variants,
        verbose=args.verbose,
    )
