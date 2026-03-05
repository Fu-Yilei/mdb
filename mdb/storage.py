from __future__ import annotations

import csv
import json
import os
import shutil
from collections import OrderedDict
from functools import lru_cache
from math import ceil
from typing import Any, Iterator

import numpy as np

from mdb.schema import (
    COHORT_STORE_KIND,
    COHORT_STORE_KINDS,
    COHORT_STORE_NPY_KIND,
    COHORT_STORE_ZARR_KIND,
    COVERAGE_MISSING,
    FORMAT_VERSION,
    SAMPLE_STORE_KIND,
    SampleBundle,
    SampleTrack,
    TrackKey,
    VALUE_DENOMINATOR,
    VALUE_MISSING,
)

SAMPLE_MANIFEST = "manifest.json"
TRACKS_DIR = "tracks"
TRACK_MANIFEST = "track_manifest.json"
TRACK_CHROMS_DIR = "chroms"
TRACK_VALUE_FILE = "value.npy"
TRACK_COVERAGE_FILE = "coverage.npy"

COHORT_MANIFEST = "manifest.json"
COHORT_INDEX = "index.npz"
COHORT_VIEWS_DIR = "views"
VIEW_MANIFEST = "view_manifest.json"
VIEW_COLUMNS = "columns.tsv"
VIEW_CHROMS_DIR = "chroms"
VIEW_ZARR_MATRIX = "matrix.zarr"

COHORT_VALUE_MISSING = np.uint16(0)
COHORT_VALUE_OFFSET = np.uint16(1)


def _import_zarr_runtime():
    try:
        import zarr  # type: ignore[import-not-found]
        from numcodecs import Blosc, blosc  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "zarr backend requested but zarr/numcodecs is not installed. "
            "Install dependencies: zarr>=3.1.1 and numcodecs>=0.12."
        ) from exc
    return zarr, Blosc, blosc


def _normalize_cohort_backend(backend: str | None) -> str:
    if backend is None:
        return "npy"
    normalized = str(backend).strip().lower()
    if normalized not in {"npy", "zarr"}:
        raise ValueError(f"Unsupported cohort backend: {backend}")
    return normalized


def _cohort_kind_for_backend(backend: str) -> str:
    return COHORT_STORE_ZARR_KIND if backend == "zarr" else COHORT_STORE_NPY_KIND


def _backend_for_cohort_kind(kind: str) -> str:
    if kind == COHORT_STORE_ZARR_KIND:
        return "zarr"
    if kind == COHORT_STORE_NPY_KIND:
        return "npy"
    raise ValueError(f"Unsupported cohort store kind: {kind}")


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _prepare_output_dir(path: str) -> None:
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError(f"Output path must be a directory: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


class _ArrayCache:
    def __init__(self, max_open: int = 64):
        self.max_open = max_open
        self._cache: OrderedDict[str, np.memmap] = OrderedDict()

    def get(self, path: str, *, shape: tuple[int, ...] | None = None, mode: str = "r+", dtype=np.uint16) -> np.memmap:
        arr = self._cache.pop(path, None)
        if arr is None:
            if shape is None:
                arr = np.lib.format.open_memmap(path, mode=mode)
            else:
                arr = np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)
        self._cache[path] = arr
        while len(self._cache) > self.max_open:
            _, old = self._cache.popitem(last=False)
            old.flush()
            mmap_obj = getattr(old, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        return arr

    def flush(self) -> None:
        for arr in self._cache.values():
            arr.flush()

    def close(self) -> None:
        for arr in self._cache.values():
            arr.flush()
            mmap_obj = getattr(arr, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        self._cache.clear()


@lru_cache(maxsize=16)
def load_reference_index(index_path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    idx = np.load(index_path, allow_pickle=True)
    chroms = [str(x) for x in idx["chroms"]]
    chrom_offsets = np.asarray(idx["chrom_offsets"], dtype=np.int64)
    pos0 = np.asarray(idx["pos0"], dtype=np.uint32)
    return chroms, chrom_offsets, pos0


def _index_chrom_bounds(chroms: list[str], chrom_offsets: np.ndarray, pos_all: np.ndarray, chrom: str) -> tuple[int, int] | None:
    if chrom not in chroms:
        return None
    idx = chroms.index(chrom)
    start = int(chrom_offsets[idx])
    end = int(chrom_offsets[idx + 1]) if idx + 1 < len(chrom_offsets) else int(pos_all.shape[0])
    return start, end


def _sample_track_dir(path: str, key: TrackKey) -> str:
    return os.path.join(path, TRACKS_DIR, key.name())


def _sample_chrom_dir(path: str, key: TrackKey, chrom: str) -> str:
    return os.path.join(_sample_track_dir(path, key), TRACK_CHROMS_DIR, chrom)


def _sample_value_path(path: str, key: TrackKey, chrom: str) -> str:
    return os.path.join(_sample_chrom_dir(path, key, chrom), TRACK_VALUE_FILE)


def _sample_coverage_path(path: str, key: TrackKey, chrom: str) -> str:
    return os.path.join(_sample_chrom_dir(path, key, chrom), TRACK_COVERAGE_FILE)


def write_sample_bundle(path: str, bundle: SampleBundle) -> None:
    _prepare_output_dir(path)
    os.makedirs(os.path.join(path, TRACKS_DIR), exist_ok=True)
    chroms, chrom_offsets, pos0 = load_reference_index(bundle.index_path)
    manifest = {
        "kind": SAMPLE_STORE_KIND,
        "format_version": FORMAT_VERSION,
        "sample_id": bundle.sample_id,
        "platform": bundle.platform,
        "index_path": bundle.index_path,
        "tracks": [key.name() for key in sorted(bundle.tracks)],
    }
    _write_json(os.path.join(path, SAMPLE_MANIFEST), manifest)

    n_rows = int(pos0.shape[0])
    chrom_ends = np.asarray(
        [int(chrom_offsets[i + 1]) if i + 1 < len(chrom_offsets) else n_rows for i in range(len(chroms))],
        dtype=np.int64,
    )

    for key in sorted(bundle.tracks):
        track = bundle.tracks[key]
        track.validate()
        track_dir = _sample_track_dir(path, key)
        os.makedirs(os.path.join(track_dir, TRACK_CHROMS_DIR), exist_ok=True)
        row_ids = np.asarray(track.row_ids, dtype=np.int64)
        values = np.asarray(track.values, dtype=np.uint16)
        coverage = np.asarray(track.coverage, dtype=np.uint16)
        chroms_present: list[str] = []

        for chrom, chrom_start, chrom_end in zip(chroms, chrom_offsets, chrom_ends, strict=False):
            lo = int(np.searchsorted(row_ids, np.int64(chrom_start), side="left"))
            hi = int(np.searchsorted(row_ids, np.int64(chrom_end), side="left"))
            if lo >= hi:
                continue
            chroms_present.append(chrom)
            chrom_dir = _sample_chrom_dir(path, key, chrom)
            os.makedirs(chrom_dir, exist_ok=True)
            chrom_len = int(chrom_end - chrom_start)
            value_arr = np.full(chrom_len, VALUE_MISSING, dtype=np.uint16)
            coverage_arr = np.full(chrom_len, COVERAGE_MISSING, dtype=np.uint16)
            local_rows = (row_ids[lo:hi] - int(chrom_start)).astype(np.int64, copy=False)
            value_arr[local_rows] = values[lo:hi]
            coverage_arr[local_rows] = coverage[lo:hi]
            np.save(_sample_value_path(path, key, chrom), value_arr)
            np.save(_sample_coverage_path(path, key, chrom), coverage_arr)

        _write_json(
            os.path.join(track_dir, TRACK_MANIFEST),
            {
                "assay": key.assay,
                "haplotype": key.haplotype,
                "strand": key.strand,
                "platform": track.platform,
                "input_tag": track.input_tag,
                "source_path": track.source_path,
                "min_coverage": int(track.min_coverage),
                "n_obs_rows": int(track.row_ids.shape[0]),
                "layout": "dense_reference_chromosomes",
                "chroms_present": chroms_present,
            },
        )


def sample_manifest(path: str) -> dict:
    manifest = _load_json(os.path.join(path, SAMPLE_MANIFEST))
    if manifest.get("kind") != SAMPLE_STORE_KIND:
        raise ValueError(f"Not a sample store: {path}")
    return manifest


def list_tracks(path: str) -> list[TrackKey]:
    return [TrackKey.from_name(name) for name in sample_manifest(path)["tracks"]]


def load_track_manifest(path: str, key: TrackKey) -> dict:
    return _load_json(os.path.join(_sample_track_dir(path, key), TRACK_MANIFEST))


class DenseSampleTrackReader:
    def __init__(self, path: str, key: TrackKey):
        self.path = path
        self.key = key
        self.manifest = sample_manifest(path)
        self.track_meta = load_track_manifest(path, key)
        self.platform = str(self.track_meta["platform"])
        self.input_tag = str(self.track_meta["input_tag"])
        self.source_path = str(self.track_meta["source_path"])
        self.min_coverage = int(self.track_meta["min_coverage"])
        # Backward compatibility: older sample bundles may not have n_obs_rows.
        self.n_obs_rows = int(self.track_meta.get("n_obs_rows", -1))
        self.chroms_present = list(self.track_meta.get("chroms_present", []))
        self.index_path = str(self.manifest["index_path"])
        self.chroms, self.chrom_offsets, self.pos0 = load_reference_index(self.index_path)
        self.n_rows = int(self.pos0.shape[0])
        self.chrom_ends = np.asarray(
            [int(self.chrom_offsets[i + 1]) if i + 1 < len(self.chrom_offsets) else self.n_rows for i in range(len(self.chroms))],
            dtype=np.int64,
        )
        self._value_cache = _ArrayCache(max_open=16)
        self._coverage_cache = _ArrayCache(max_open=16)

    def _chrom_len(self, chrom: str) -> int:
        bounds = _index_chrom_bounds(self.chroms, self.chrom_offsets, self.pos0, chrom)
        if bounds is None:
            raise KeyError(chrom)
        start, end = bounds
        return end - start

    def has_chrom(self, chrom: str) -> bool:
        return chrom in self.chroms_present and os.path.exists(_sample_value_path(self.path, self.key, chrom))

    def chrom_values(self, chrom: str, *, allow_missing: bool = False) -> np.ndarray:
        path = _sample_value_path(self.path, self.key, chrom)
        if os.path.exists(path):
            return self._value_cache.get(path, mode="r")
        if allow_missing:
            return np.full(self._chrom_len(chrom), VALUE_MISSING, dtype=np.uint16)
        raise FileNotFoundError(path)

    def chrom_coverage(self, chrom: str, *, allow_missing: bool = False) -> np.ndarray:
        path = _sample_coverage_path(self.path, self.key, chrom)
        if os.path.exists(path):
            return self._coverage_cache.get(path, mode="r")
        if allow_missing:
            return np.full(self._chrom_len(chrom), COVERAGE_MISSING, dtype=np.uint16)
        raise FileNotFoundError(path)

    def close(self) -> None:
        self._value_cache.close()
        self._coverage_cache.close()


def read_track(path: str, key: TrackKey) -> DenseSampleTrackReader:
    return DenseSampleTrackReader(path, key)


def _point_result_sample(sample_id: str, track: DenseSampleTrackReader, chrom: str, pos0: int, idx: int) -> dict:
    values = track.chrom_values(chrom, allow_missing=True)
    coverage = track.chrom_coverage(chrom, allow_missing=True)
    return {
        "sample_id": sample_id,
        "track": track.key.name(),
        "chrom": chrom,
        "pos0": pos0,
        "value_percent": float(values[idx]) / 100.0,
        "value_fraction": float(values[idx]) / VALUE_DENOMINATOR,
        "coverage": int(coverage[idx]),
        "source_path": track.source_path,
        "input_tag": track.input_tag,
    }


def query_sample_track(path: str, key: TrackKey, chrom: str, pos0: int) -> dict | None:
    manifest = sample_manifest(path)
    chroms, chrom_offsets, index_pos0 = load_reference_index(str(manifest["index_path"]))
    bounds = _index_chrom_bounds(chroms, chrom_offsets, index_pos0, chrom)
    if bounds is None:
        return None
    start, end = bounds
    chrom_pos = index_pos0[start:end]
    j = np.searchsorted(chrom_pos, np.uint32(pos0), side="left")
    if j >= chrom_pos.shape[0] or int(chrom_pos[j]) != pos0:
        return None
    track = read_track(path, key)
    try:
        values = track.chrom_values(chrom, allow_missing=True)
        if int(values[j]) == int(VALUE_MISSING):
            return None
        return _point_result_sample(str(manifest["sample_id"]), track, chrom, pos0, int(j))
    finally:
        track.close()


def query_sample_range(path: str, key: TrackKey, chrom: str, start_pos0: int, end_pos0: int) -> dict:
    if end_pos0 < start_pos0:
        raise ValueError("end_pos0 must be greater than or equal to start_pos0")
    manifest = sample_manifest(path)
    track = read_track(path, key)
    try:
        chroms, chrom_offsets, index_pos0 = load_reference_index(str(manifest["index_path"]))
        bounds = _index_chrom_bounds(chroms, chrom_offsets, index_pos0, chrom)
        if bounds is None:
            return {
                "sample_id": str(manifest["sample_id"]),
                "track": key.name(),
                "chrom": chrom,
                "start_pos0": start_pos0,
                "end_pos0": end_pos0,
                "source_path": track.source_path,
                "input_tag": track.input_tag,
                "count": 0,
                "records": [],
            }
        start, end = bounds
        chrom_pos = index_pos0[start:end]
        lo = int(np.searchsorted(chrom_pos, np.uint32(start_pos0), side="left"))
        hi = int(np.searchsorted(chrom_pos, np.uint32(end_pos0), side="right"))
        values = np.asarray(track.chrom_values(chrom, allow_missing=True)[lo:hi], dtype=np.uint16)
        coverage = np.asarray(track.chrom_coverage(chrom, allow_missing=True)[lo:hi], dtype=np.uint16)
        mask = values != VALUE_MISSING
        return {
            "sample_id": str(manifest["sample_id"]),
            "track": key.name(),
            "chrom": chrom,
            "start_pos0": start_pos0,
            "end_pos0": end_pos0,
            "source_path": track.source_path,
            "input_tag": track.input_tag,
            "count": int(np.count_nonzero(mask)),
            "records": [
                {
                    "pos0": int(pos),
                    "value_percent": float(val) / 100.0,
                    "value_fraction": float(val) / VALUE_DENOMINATOR,
                    "coverage": int(cov),
                }
                for pos, val, cov in zip(chrom_pos[lo:hi][mask], values[mask], coverage[mask], strict=False)
            ],
        }
    finally:
        track.close()


def detect_store_kind(path: str) -> str:
    if not os.path.isdir(path):
        raise ValueError(f"Store path must be a directory: {path}")
    manifest = _load_json(os.path.join(path, SAMPLE_MANIFEST))
    return str(manifest.get("kind", ""))


def create_cohort_store(
    path: str,
    chroms: list[str],
    chrom_offsets: np.ndarray,
    pos0: np.ndarray,
    index_path: str | None = None,
    block_size: int = 64,
    *,
    backend: str = "npy",
    zarr_row_chunk: int = 65536,
    zarr_codec: str = "zstd",
    zarr_clevel: int = 5,
    zarr_shuffle: str = "bitshuffle",
) -> None:
    backend = _normalize_cohort_backend(backend)
    _prepare_output_dir(path)
    os.makedirs(os.path.join(path, COHORT_VIEWS_DIR), exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": _cohort_kind_for_backend(backend),
        "backend": backend,
        "format_version": FORMAT_VERSION,
        "value_denominator": VALUE_DENOMINATOR,
        "missing_value": int(COHORT_VALUE_MISSING),
        "value_offset": int(COHORT_VALUE_OFFSET),
        "index_path": index_path,
        "block_size": int(block_size),
    }
    if backend == "zarr":
        if zarr_codec != "zstd":
            raise ValueError(f"Unsupported zarr codec: {zarr_codec}")
        if zarr_shuffle not in {"none", "shuffle", "bitshuffle"}:
            raise ValueError(f"Unsupported zarr shuffle mode: {zarr_shuffle}")
        if int(zarr_row_chunk) < 1:
            raise ValueError("zarr_row_chunk must be >= 1")
        manifest.update(
            {
                "zarr_row_chunk": int(zarr_row_chunk),
                "zarr_codec": str(zarr_codec),
                "zarr_clevel": int(zarr_clevel),
                "zarr_shuffle": str(zarr_shuffle),
            }
        )
    _write_json(os.path.join(path, COHORT_MANIFEST), manifest)
    np.savez(
        os.path.join(path, COHORT_INDEX),
        chroms=np.asarray(list(chroms), dtype=object),
        chrom_offsets=np.asarray(chrom_offsets, dtype=np.int64),
        pos0=np.asarray(pos0, dtype=np.uint32),
    )


def load_cohort_manifest(path: str) -> dict:
    manifest = _load_json(os.path.join(path, COHORT_MANIFEST))
    if manifest.get("kind") not in COHORT_STORE_KINDS:
        raise ValueError(f"Not a cohort store: {path}")
    if "backend" not in manifest:
        kind = str(manifest.get("kind", ""))
        manifest["backend"] = _backend_for_cohort_kind(kind)
    return manifest


def load_cohort_index(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    idx = np.load(os.path.join(path, COHORT_INDEX), allow_pickle=True)
    return [str(x) for x in idx["chroms"]], np.asarray(idx["chrom_offsets"], dtype=np.int64), np.asarray(idx["pos0"], dtype=np.uint32)


def cohort_backend(path: str) -> str:
    return _normalize_cohort_backend(str(load_cohort_manifest(path).get("backend", "npy")))


def cohort_view_dir(path: str, key: TrackKey) -> str:
    return os.path.join(path, COHORT_VIEWS_DIR, key.name())


def _chrom_dir(path: str, key: TrackKey, chrom: str) -> str:
    return os.path.join(cohort_view_dir(path, key), VIEW_CHROMS_DIR, chrom)


def _block_path(path: str, key: TrackKey, chrom: str, block_idx: int) -> str:
    return os.path.join(_chrom_dir(path, key, chrom), f"block_{block_idx:04d}.npy")


def _zarr_matrix_path(path: str, key: TrackKey, chrom: str) -> str:
    return os.path.join(_chrom_dir(path, key, chrom), VIEW_ZARR_MATRIX)


def _write_columns_tsv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "bundle_path", "platform", "source_path", "input_tag", "block_idx", "col_idx"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _view_column_rows(
    sample_ids: list[str],
    bundle_paths: list[str],
    platforms: list[str],
    source_paths: list[str],
    input_tags: list[str],
    block_size: int,
) -> list[dict]:
    rows: list[dict] = []
    for global_idx, (sample_id, bundle_path, platform, source_path, input_tag) in enumerate(
        zip(sample_ids, bundle_paths, platforms, source_paths, input_tags, strict=False)
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "bundle_path": bundle_path,
                "platform": platform,
                "source_path": source_path,
                "input_tag": input_tag,
                "block_idx": global_idx // block_size,
                "col_idx": global_idx % block_size,
            }
        )
    return rows


def load_view_columns(path: str, key: TrackKey) -> dict[str, list]:
    cols = {
        "sample_id": [],
        "bundle_path": [],
        "platform": [],
        "source_path": [],
        "input_tag": [],
        "block_idx": [],
        "col_idx": [],
    }
    with open(os.path.join(cohort_view_dir(path, key), VIEW_COLUMNS), "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cols["sample_id"].append(row["sample_id"])
            cols["bundle_path"].append(row["bundle_path"])
            cols["platform"].append(row["platform"])
            cols["source_path"].append(row["source_path"])
            cols["input_tag"].append(row["input_tag"])
            cols["block_idx"].append(int(row["block_idx"]))
            cols["col_idx"].append(int(row["col_idx"]))
    return cols


def _write_view_manifest(
    view_dir: str,
    key: TrackKey,
    *,
    n_rows: int,
    n_samples: int,
    block_size: int,
    backend: str = "npy",
    zarr_config: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "assay": key.assay,
        "haplotype": key.haplotype,
        "strand": key.strand,
        "backend": backend,
        "n_rows": int(n_rows),
        "n_samples": int(n_samples),
        "block_size": int(block_size),
        "n_blocks": int(max(1, ceil(max(n_samples, 1) / block_size))),
        "dtype": "uint16",
        "missing_value": int(COHORT_VALUE_MISSING),
        "value_offset": int(COHORT_VALUE_OFFSET),
        "layout": "chromosome_sample_blocks",
    }
    if backend == "zarr":
        payload["layout"] = "chromosome_matrix_zarr"
        if zarr_config:
            payload.update(
                {
                    "zarr_row_chunk": int(zarr_config["row_chunk"]),
                    "zarr_codec": str(zarr_config["codec"]),
                    "zarr_clevel": int(zarr_config["clevel"]),
                    "zarr_shuffle": str(zarr_config["shuffle"]),
                }
            )
    _write_json(os.path.join(view_dir, VIEW_MANIFEST), payload)


def load_view_manifest(path: str, key: TrackKey) -> dict:
    return _load_json(os.path.join(cohort_view_dir(path, key), VIEW_MANIFEST))


def _cohort_zarr_config_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_chunk": int(manifest.get("zarr_row_chunk", 65536)),
        "codec": str(manifest.get("zarr_codec", "zstd")),
        "clevel": int(manifest.get("zarr_clevel", 5)),
        "shuffle": str(manifest.get("zarr_shuffle", "bitshuffle")),
    }


def _normalize_zarr_config(zarr_config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(zarr_config or {})
    row_chunk = int(cfg.get("row_chunk", 65536))
    codec = str(cfg.get("codec", "zstd"))
    clevel = int(cfg.get("clevel", 5))
    shuffle = str(cfg.get("shuffle", "bitshuffle")).lower()
    if row_chunk < 1:
        raise ValueError("zarr row chunk must be >= 1")
    if codec != "zstd":
        raise ValueError(f"Unsupported zarr codec: {codec}")
    if shuffle not in {"none", "shuffle", "bitshuffle"}:
        raise ValueError(f"Unsupported zarr shuffle mode: {shuffle}")
    return {
        "row_chunk": row_chunk,
        "codec": codec,
        "clevel": clevel,
        "shuffle": shuffle,
    }


class NpyViewBlockWriter:
    def __init__(
        self,
        path: str,
        key: TrackKey,
        chroms: list[str],
        chrom_offsets: np.ndarray,
        n_rows: int,
        n_samples: int,
        block_size: int,
    ):
        self.path = path
        self.key = key
        self.chroms = list(chroms)
        self.chrom_offsets = np.asarray(chrom_offsets, dtype=np.int64)
        self.chrom_ends = np.asarray(
            [int(self.chrom_offsets[i + 1]) if i + 1 < len(self.chrom_offsets) else int(n_rows) for i in range(len(self.chroms))],
            dtype=np.int64,
        )
        self.n_rows = int(n_rows)
        self.n_samples = int(n_samples)
        self.block_size = int(block_size)
        self.n_blocks = int(max(1, ceil(max(n_samples, 1) / block_size)))
        self.view_dir = cohort_view_dir(path, key)
        self.cache = _ArrayCache()
        self.chrom_row_counts = {
            chrom: int(chrom_end - chrom_start)
            for chrom, chrom_start, chrom_end in zip(self.chroms, self.chrom_offsets, self.chrom_ends, strict=False)
        }

    def _ensure_block(self, chrom: str, block_idx: int) -> None:
        os.makedirs(os.path.join(self.view_dir, VIEW_CHROMS_DIR), exist_ok=True)
        chrom_dir = _chrom_dir(self.path, self.key, chrom)
        os.makedirs(chrom_dir, exist_ok=True)
        block_path = _block_path(self.path, self.key, chrom, block_idx)
        if os.path.exists(block_path):
            return
        arr = np.lib.format.open_memmap(
            block_path,
            mode="w+",
            dtype=np.uint16,
            shape=(self.chrom_row_counts[chrom], self.block_size),
        )
        arr.flush()
        mmap_obj = getattr(arr, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def _open_block(self, chrom: str, block_idx: int) -> np.memmap:
        self._ensure_block(chrom, block_idx)
        return self.cache.get(_block_path(self.path, self.key, chrom, block_idx), mode="r+")

    def write_dense_chrom(self, chrom: str, global_col_idx: int, values: np.ndarray) -> None:
        block_idx = int(global_col_idx // self.block_size)
        col_idx = int(global_col_idx % self.block_size)
        arr = self._open_block(chrom, block_idx)
        dense_values = np.asarray(values, dtype=np.uint16)
        if dense_values.shape[0] != arr.shape[0]:
            raise ValueError(
                f"Chromosome length mismatch for {chrom}: expected {arr.shape[0]} rows, got {dense_values.shape[0]}"
            )
        arr[:, col_idx] = dense_values

    def write_dense_chrom_block(
        self,
        chrom: str,
        row_lo: int,
        row_hi: int,
        col_lo: int,
        col_hi: int,
        values: np.ndarray,
    ) -> None:
        if col_hi <= col_lo or row_hi <= row_lo:
            return
        values = np.asarray(values, dtype=np.uint16)
        if values.shape != (row_hi - row_lo, col_hi - col_lo):
            raise ValueError(
                f"Block shape mismatch for {chrom}: expected {(row_hi - row_lo, col_hi - col_lo)}, got {values.shape}"
            )
        for global_col_idx in range(col_lo, col_hi):
            block_idx = int(global_col_idx // self.block_size)
            col_idx = int(global_col_idx % self.block_size)
            arr = self._open_block(chrom, block_idx)
            local_col = global_col_idx - col_lo
            arr[row_lo:row_hi, col_idx] = values[:, local_col]

    def flush(self) -> None:
        self.cache.flush()

    def close(self) -> None:
        self.cache.close()


class ZarrViewBlockWriter:
    def __init__(
        self,
        path: str,
        key: TrackKey,
        chroms: list[str],
        chrom_offsets: np.ndarray,
        n_rows: int,
        n_samples: int,
        block_size: int,
        zarr_config: dict[str, Any] | None = None,
    ):
        self.path = path
        self.key = key
        self.chroms = list(chroms)
        self.chrom_offsets = np.asarray(chrom_offsets, dtype=np.int64)
        self.chrom_ends = np.asarray(
            [int(self.chrom_offsets[i + 1]) if i + 1 < len(self.chrom_offsets) else int(n_rows) for i in range(len(self.chroms))],
            dtype=np.int64,
        )
        self.n_rows = int(n_rows)
        self.n_samples = int(n_samples)
        self.block_size = int(block_size)
        self.n_blocks = int(max(1, ceil(max(n_samples, 1) / block_size)))
        self.view_dir = cohort_view_dir(path, key)
        self.chrom_row_counts = {
            chrom: int(chrom_end - chrom_start)
            for chrom, chrom_start, chrom_end in zip(self.chroms, self.chrom_offsets, self.chrom_ends, strict=False)
        }
        self.zarr_config = _normalize_zarr_config(zarr_config)
        self._arrays: dict[str, Any] = {}
        self._zarr, self._Blosc, _ = _import_zarr_runtime()

    def _compressor(self):
        shuffle_name = self.zarr_config["shuffle"]
        if shuffle_name == "none":
            shuffle_code = self._Blosc.NOSHUFFLE
        elif shuffle_name == "shuffle":
            shuffle_code = self._Blosc.SHUFFLE
        else:
            shuffle_code = self._Blosc.BITSHUFFLE
        return self._Blosc(
            cname=self.zarr_config["codec"],
            clevel=int(self.zarr_config["clevel"]),
            shuffle=shuffle_code,
        )

    def _open_chrom(self, chrom: str):
        arr = self._arrays.get(chrom)
        if arr is not None:
            return arr
        chrom_dir = _chrom_dir(self.path, self.key, chrom)
        os.makedirs(chrom_dir, exist_ok=True)
        store_path = _zarr_matrix_path(self.path, self.key, chrom)
        rows = self.chrom_row_counts[chrom]
        if os.path.isdir(store_path):
            arr = self._zarr.open(store_path, mode="r+")
            if int(arr.shape[0]) != rows:
                raise ValueError(f"Existing zarr row count mismatch for {chrom}: expected {rows}, found {arr.shape[0]}")
            if int(arr.shape[1]) < self.n_samples:
                arr.resize((rows, self.n_samples))
        else:
            arr = self._zarr.create(
                shape=(rows, self.n_samples),
                chunks=(int(self.zarr_config["row_chunk"]), self.block_size),
                dtype=np.uint16,
                store=store_path,
                overwrite=True,
                zarr_format=2,
                compressor=self._compressor(),
                fill_value=int(COHORT_VALUE_MISSING),
            )
        self._arrays[chrom] = arr
        return arr

    def write_dense_chrom(self, chrom: str, global_col_idx: int, values: np.ndarray) -> None:
        arr = self._open_chrom(chrom)
        dense_values = np.asarray(values, dtype=np.uint16)
        if dense_values.shape[0] != arr.shape[0]:
            raise ValueError(f"Chromosome length mismatch for {chrom}: expected {arr.shape[0]}, got {dense_values.shape[0]}")
        arr[:, int(global_col_idx)] = dense_values

    def write_dense_chrom_block(
        self,
        chrom: str,
        row_lo: int,
        row_hi: int,
        col_lo: int,
        col_hi: int,
        values: np.ndarray,
    ) -> None:
        if col_hi <= col_lo or row_hi <= row_lo:
            return
        arr = self._open_chrom(chrom)
        payload = np.asarray(values, dtype=np.uint16)
        expected = (row_hi - row_lo, col_hi - col_lo)
        if payload.shape != expected:
            raise ValueError(f"Block shape mismatch for {chrom}: expected {expected}, got {payload.shape}")
        arr[row_lo:row_hi, col_lo:col_hi] = payload

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self._arrays.clear()


# Backward-compatible alias used in existing code/tests.
ViewBlockWriter = NpyViewBlockWriter


def create_view_store(
    path: str,
    key: TrackKey,
    chroms: list[str],
    chrom_offsets: np.ndarray,
    n_rows: int,
    sample_ids: list[str],
    bundle_paths: list[str],
    platforms: list[str],
    source_paths: list[str],
    input_tags: list[str],
    block_size: int,
) -> NpyViewBlockWriter | ZarrViewBlockWriter:
    cohort_manifest = load_cohort_manifest(path)
    backend = _normalize_cohort_backend(str(cohort_manifest.get("backend", "npy")))
    zarr_config = _cohort_zarr_config_from_manifest(cohort_manifest)
    view_dir = cohort_view_dir(path, key)
    os.makedirs(view_dir, exist_ok=True)
    rows = _view_column_rows(sample_ids, bundle_paths, platforms, source_paths, input_tags, block_size)
    _write_columns_tsv(os.path.join(view_dir, VIEW_COLUMNS), rows)
    _write_view_manifest(
        view_dir,
        key,
        n_rows=n_rows,
        n_samples=len(sample_ids),
        block_size=block_size,
        backend=backend,
        zarr_config=zarr_config,
    )
    if backend == "zarr":
        return ZarrViewBlockWriter(path, key, chroms, chrom_offsets, n_rows, len(sample_ids), block_size, zarr_config=zarr_config)
    return NpyViewBlockWriter(path, key, chroms, chrom_offsets, n_rows, len(sample_ids), block_size)


def append_view_columns(
    path: str,
    key: TrackKey,
    *,
    sample_ids: list[str],
    bundle_paths: list[str],
    platforms: list[str],
    source_paths: list[str],
    input_tags: list[str],
) -> tuple[NpyViewBlockWriter | ZarrViewBlockWriter, int]:
    cohort_manifest = load_cohort_manifest(path)
    backend = _normalize_cohort_backend(str(cohort_manifest.get("backend", "npy")))
    manifest = load_view_manifest(path, key)
    columns = load_view_columns(path, key)
    old_n_samples = int(manifest["n_samples"])
    block_size = int(manifest["block_size"])
    chroms, chrom_offsets, pos0 = load_cohort_index(path)
    zarr_config = _cohort_zarr_config_from_manifest(cohort_manifest)
    if backend == "zarr":
        zarr_config = {
            "row_chunk": int(manifest.get("zarr_row_chunk", zarr_config["row_chunk"])),
            "codec": str(manifest.get("zarr_codec", zarr_config["codec"])),
            "clevel": int(manifest.get("zarr_clevel", zarr_config["clevel"])),
            "shuffle": str(manifest.get("zarr_shuffle", zarr_config["shuffle"])),
        }
    rows = _view_column_rows(
        columns["sample_id"] + list(sample_ids),
        columns["bundle_path"] + list(bundle_paths),
        columns["platform"] + list(platforms),
        columns["source_path"] + list(source_paths),
        columns["input_tag"] + list(input_tags),
        block_size,
    )
    _write_columns_tsv(os.path.join(cohort_view_dir(path, key), VIEW_COLUMNS), rows)
    if backend == "zarr":
        writer: NpyViewBlockWriter | ZarrViewBlockWriter = ZarrViewBlockWriter(
            path,
            key,
            chroms,
            chrom_offsets,
            int(pos0.shape[0]),
            old_n_samples + len(sample_ids),
            block_size,
            zarr_config=zarr_config,
        )
    else:
        writer = NpyViewBlockWriter(path, key, chroms, chrom_offsets, int(pos0.shape[0]), old_n_samples + len(sample_ids), block_size)
    _write_view_manifest(
        cohort_view_dir(path, key),
        key,
        n_rows=int(pos0.shape[0]),
        n_samples=old_n_samples + len(sample_ids),
        block_size=block_size,
        backend=backend,
        zarr_config=zarr_config,
    )
    return writer, old_n_samples


def available_views(path: str) -> list[TrackKey]:
    manifest = load_cohort_manifest(path)
    if manifest["kind"] not in COHORT_STORE_KINDS:
        raise ValueError(f"Unsupported cohort store: {path}")
    views_dir = os.path.join(path, COHORT_VIEWS_DIR)
    if not os.path.isdir(views_dir):
        return []
    return [TrackKey.from_name(name) for name in sorted(name for name in os.listdir(views_dir) if os.path.isdir(os.path.join(views_dir, name)))]


def encode_cohort_dense(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint16)
    out = np.zeros(arr.shape, dtype=np.uint16)
    mask = arr != VALUE_MISSING
    out[mask] = (arr[mask].astype(np.uint32) + np.uint32(COHORT_VALUE_OFFSET)).astype(np.uint16)
    return out


def decode_cohort_scalar(value: np.uint16) -> tuple[float, float]:
    if value == COHORT_VALUE_MISSING:
        return np.nan, np.nan
    adjusted = float(int(value) - int(COHORT_VALUE_OFFSET))
    return adjusted / VALUE_DENOMINATOR, adjusted / 100.0


class NpyBlockMatrixReader:
    def __init__(self, path: str, key: TrackKey):
        self.path = path
        self.key = key
        self.view_dir = cohort_view_dir(path, key)
        self.manifest = load_view_manifest(path, key)
        self.columns = load_view_columns(path, key)
        self.chroms, self.chrom_offsets, _ = load_cohort_index(path)
        self.n_rows = int(self.manifest["n_rows"])
        self.n_samples = int(self.manifest["n_samples"])
        self.block_size = int(self.manifest["block_size"])
        self.n_blocks = int(max(1, ceil(max(self.n_samples, 1) / self.block_size)))
        self.chrom_ends = np.asarray(
            [int(self.chrom_offsets[i + 1]) if i + 1 < len(self.chrom_offsets) else self.n_rows for i in range(len(self.chroms))],
            dtype=np.int64,
        )
        self.shape = (self.n_rows, self.n_samples)
        self.cache = _ArrayCache()

    def _block_cols(self, block_idx: int) -> int:
        start = block_idx * self.block_size
        return max(0, min(self.block_size, self.n_samples - start))

    def _open_block(self, chrom: str, block_idx: int) -> np.memmap:
        return self.cache.get(_block_path(self.path, self.key, chrom, block_idx), mode="r")

    def _read_chrom_raw(self, chrom: str, lo: int, hi: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        for block_idx in range(self.n_blocks):
            cols = self._block_cols(block_idx)
            if cols <= 0:
                continue
            block_path = _block_path(self.path, self.key, chrom, block_idx)
            if os.path.exists(block_path):
                arr = self._open_block(chrom, block_idx)
                parts.append(np.asarray(arr[lo:hi, :cols], dtype=np.uint16))
            else:
                parts.append(np.zeros((hi - lo, cols), dtype=np.uint16))
        if not parts:
            return np.empty((hi - lo, 0), dtype=np.uint16)
        return np.concatenate(parts, axis=1)

    def _read_raw(self, rows) -> np.ndarray:
        if isinstance(rows, slice):
            start, stop, step = rows.indices(self.n_rows)
            if step != 1:
                return self._read_raw(np.arange(start, stop, step, dtype=np.int64))
            parts: list[np.ndarray] = []
            for chrom, chrom_start, chrom_end in zip(self.chroms, self.chrom_offsets, self.chrom_ends, strict=False):
                lo = max(start, int(chrom_start))
                hi = min(stop, int(chrom_end))
                if lo >= hi:
                    continue
                parts.append(self._read_chrom_raw(chrom, lo - int(chrom_start), hi - int(chrom_start)))
            if not parts:
                return np.empty((0, self.n_samples), dtype=np.uint16)
            return np.concatenate(parts, axis=0)

        row_idx = np.asarray(rows, dtype=np.int64)
        out = np.empty((row_idx.shape[0], self.n_samples), dtype=np.uint16)
        for chrom, chrom_start, chrom_end in zip(self.chroms, self.chrom_offsets, self.chrom_ends, strict=False):
            mask = (row_idx >= int(chrom_start)) & (row_idx < int(chrom_end))
            if not np.any(mask):
                continue
            local = (row_idx[mask] - int(chrom_start)).astype(np.int64, copy=False)
            parts: list[np.ndarray] = []
            for block_idx in range(self.n_blocks):
                cols = self._block_cols(block_idx)
                if cols <= 0:
                    continue
                block_path = _block_path(self.path, self.key, chrom, block_idx)
                if os.path.exists(block_path):
                    arr = self._open_block(chrom, block_idx)
                    parts.append(np.asarray(arr[local, :cols], dtype=np.uint16))
                else:
                    parts.append(np.zeros((local.shape[0], cols), dtype=np.uint16))
            out[mask, :] = np.concatenate(parts, axis=1)
        return out

    def get_block(self, rows) -> np.ndarray:
        raw = self._read_raw(rows)
        block = raw.astype(np.float32)
        missing = raw == COHORT_VALUE_MISSING
        block[missing] = np.nan
        if np.any(~missing):
            block[~missing] = (block[~missing] - float(COHORT_VALUE_OFFSET)) / VALUE_DENOMINATOR
        return block

    def iter_blocks(self, batch_rows: int) -> Iterator[np.ndarray]:
        for start in range(0, self.n_rows, batch_rows):
            yield self.get_block(slice(start, min(start + batch_rows, self.n_rows)))

    def read_rows(self, row_idx: np.ndarray) -> np.ndarray:
        return self.get_block(row_idx)

    def close(self) -> None:
        self.cache.close()


class ZarrBlockMatrixReader:
    def __init__(self, path: str, key: TrackKey):
        self.path = path
        self.key = key
        self.view_dir = cohort_view_dir(path, key)
        self.manifest = load_view_manifest(path, key)
        self.columns = load_view_columns(path, key)
        self.chroms, self.chrom_offsets, _ = load_cohort_index(path)
        self.n_rows = int(self.manifest["n_rows"])
        self.n_samples = int(self.manifest["n_samples"])
        self.block_size = int(self.manifest["block_size"])
        self.n_blocks = int(max(1, ceil(max(self.n_samples, 1) / self.block_size)))
        self.chrom_ends = np.asarray(
            [int(self.chrom_offsets[i + 1]) if i + 1 < len(self.chrom_offsets) else self.n_rows for i in range(len(self.chroms))],
            dtype=np.int64,
        )
        self.shape = (self.n_rows, self.n_samples)
        self._zarr, _, _ = _import_zarr_runtime()
        self._arrays: dict[str, Any | None] = {}

    def _block_cols(self, block_idx: int) -> int:
        start = block_idx * self.block_size
        return max(0, min(self.block_size, self.n_samples - start))

    def _open_chrom(self, chrom: str):
        if chrom in self._arrays:
            return self._arrays[chrom]
        store_path = _zarr_matrix_path(self.path, self.key, chrom)
        if not os.path.isdir(store_path):
            self._arrays[chrom] = None
            return None
        arr = self._zarr.open(store_path, mode="r")
        self._arrays[chrom] = arr
        return arr

    def _read_chrom_raw(self, chrom: str, lo: int, hi: int) -> np.ndarray:
        if hi <= lo:
            return np.empty((0, self.n_samples), dtype=np.uint16)
        arr = self._open_chrom(chrom)
        if arr is None:
            return np.zeros((hi - lo, self.n_samples), dtype=np.uint16)
        return np.asarray(arr[lo:hi, : self.n_samples], dtype=np.uint16)

    def _read_raw(self, rows) -> np.ndarray:
        if isinstance(rows, slice):
            start, stop, step = rows.indices(self.n_rows)
            if step != 1:
                return self._read_raw(np.arange(start, stop, step, dtype=np.int64))
            parts: list[np.ndarray] = []
            for chrom, chrom_start, chrom_end in zip(self.chroms, self.chrom_offsets, self.chrom_ends, strict=False):
                lo = max(start, int(chrom_start))
                hi = min(stop, int(chrom_end))
                if lo >= hi:
                    continue
                parts.append(self._read_chrom_raw(chrom, lo - int(chrom_start), hi - int(chrom_start)))
            if not parts:
                return np.empty((0, self.n_samples), dtype=np.uint16)
            return np.concatenate(parts, axis=0)

        row_idx = np.asarray(rows, dtype=np.int64)
        out = np.empty((row_idx.shape[0], self.n_samples), dtype=np.uint16)
        for chrom, chrom_start, chrom_end in zip(self.chroms, self.chrom_offsets, self.chrom_ends, strict=False):
            mask = (row_idx >= int(chrom_start)) & (row_idx < int(chrom_end))
            if not np.any(mask):
                continue
            local = (row_idx[mask] - int(chrom_start)).astype(np.int64, copy=False)
            arr = self._open_chrom(chrom)
            if arr is None:
                out[mask, :] = np.zeros((local.shape[0], self.n_samples), dtype=np.uint16)
            else:
                out[mask, :] = np.asarray(arr[local, : self.n_samples], dtype=np.uint16)
        return out

    def get_block(self, rows) -> np.ndarray:
        raw = self._read_raw(rows)
        block = raw.astype(np.float32)
        missing = raw == COHORT_VALUE_MISSING
        block[missing] = np.nan
        if np.any(~missing):
            block[~missing] = (block[~missing] - float(COHORT_VALUE_OFFSET)) / VALUE_DENOMINATOR
        return block

    def iter_blocks(self, batch_rows: int) -> Iterator[np.ndarray]:
        for start in range(0, self.n_rows, batch_rows):
            yield self.get_block(slice(start, min(start + batch_rows, self.n_rows)))

    def read_rows(self, row_idx: np.ndarray) -> np.ndarray:
        return self.get_block(row_idx)

    def close(self) -> None:
        self._arrays.clear()


def load_view_reader(path: str, key: TrackKey):
    backend = cohort_backend(path)
    if backend == "zarr":
        reader = ZarrBlockMatrixReader(path, key)
        return reader, reader.columns, f"{path}::views/{key.name()}/chroms/*/{VIEW_ZARR_MATRIX}"
    reader = NpyBlockMatrixReader(path, key)
    return reader, reader.columns, f"{path}::views/{key.name()}/chroms/*/block_*.npy"


def load_view_uint16_matrix(path: str, key: TrackKey) -> np.ndarray:
    backend = cohort_backend(path)
    if backend == "zarr":
        reader = ZarrBlockMatrixReader(path, key)
    else:
        reader = NpyBlockMatrixReader(path, key)
    try:
        return reader._read_raw(slice(0, reader.n_rows))
    finally:
        reader.close()


def query_cohort_point(path: str, key: TrackKey, sample_id: str, chrom: str, pos0: int) -> dict | None:
    columns = load_view_columns(path, key)
    if sample_id not in columns["sample_id"]:
        return None
    sample_idx = columns["sample_id"].index(sample_id)
    chroms, chrom_offsets, pos_all = load_cohort_index(path)
    bounds = _index_chrom_bounds(chroms, chrom_offsets, pos_all, chrom)
    if bounds is None:
        return None
    start, end = bounds
    chrom_pos = pos_all[start:end]
    j = np.searchsorted(chrom_pos, np.uint32(pos0), side="left")
    if j >= chrom_pos.shape[0] or int(chrom_pos[j]) != pos0:
        return None
    backend = cohort_backend(path)
    if backend == "zarr":
        store_path = _zarr_matrix_path(path, key, chrom)
        if os.path.isdir(store_path):
            zarr, _, _ = _import_zarr_runtime()
            arr = zarr.open(store_path, mode="r")
            raw = np.uint16(arr[int(j), int(sample_idx)])
        else:
            raw = COHORT_VALUE_MISSING
    else:
        block_idx = int(columns["block_idx"][sample_idx])
        col_idx = int(columns["col_idx"][sample_idx])
        block_path = _block_path(path, key, chrom, block_idx)
        if os.path.exists(block_path):
            arr = np.load(block_path, mmap_mode="r")
            raw = np.uint16(arr[int(j), col_idx])
        else:
            raw = COHORT_VALUE_MISSING
    value_fraction, value_percent = decode_cohort_scalar(raw)
    return {
        "sample_id": sample_id,
        "track": key.name(),
        "chrom": chrom,
        "pos0": pos0,
        "value_fraction": value_fraction,
        "value_percent": value_percent,
        "source_path": columns["source_path"][sample_idx],
        "input_tag": columns["input_tag"][sample_idx],
    }


def query_cohort_range(path: str, key: TrackKey, sample_id: str, chrom: str, start_pos0: int, end_pos0: int) -> dict:
    if end_pos0 < start_pos0:
        raise ValueError("end_pos0 must be greater than or equal to start_pos0")
    columns = load_view_columns(path, key)
    if sample_id not in columns["sample_id"]:
        return {
            "sample_id": sample_id,
            "track": key.name(),
            "chrom": chrom,
            "start_pos0": start_pos0,
            "end_pos0": end_pos0,
            "count": 0,
            "records": [],
        }
    sample_idx = columns["sample_id"].index(sample_id)
    chroms, chrom_offsets, pos_all = load_cohort_index(path)
    bounds = _index_chrom_bounds(chroms, chrom_offsets, pos_all, chrom)
    if bounds is None:
        return {
            "sample_id": sample_id,
            "track": key.name(),
            "chrom": chrom,
            "start_pos0": start_pos0,
            "end_pos0": end_pos0,
            "source_path": columns["source_path"][sample_idx],
            "input_tag": columns["input_tag"][sample_idx],
            "count": 0,
            "records": [],
        }
    start, end = bounds
    chrom_pos = pos_all[start:end]
    lo = int(np.searchsorted(chrom_pos, np.uint32(start_pos0), side="left"))
    hi = int(np.searchsorted(chrom_pos, np.uint32(end_pos0), side="right"))
    backend = cohort_backend(path)
    if backend == "zarr":
        store_path = _zarr_matrix_path(path, key, chrom)
        if os.path.isdir(store_path):
            zarr, _, _ = _import_zarr_runtime()
            arr = zarr.open(store_path, mode="r")
            raw_values = np.asarray(arr[lo:hi, int(sample_idx)], dtype=np.uint16)
        else:
            raw_values = np.full(max(hi - lo, 0), COHORT_VALUE_MISSING, dtype=np.uint16)
    else:
        block_idx = int(columns["block_idx"][sample_idx])
        col_idx = int(columns["col_idx"][sample_idx])
        block_path = _block_path(path, key, chrom, block_idx)
        if os.path.exists(block_path):
            arr = np.load(block_path, mmap_mode="r")
            raw_values = np.asarray(arr[lo:hi, col_idx], dtype=np.uint16)
        else:
            raw_values = np.full(max(hi - lo, 0), COHORT_VALUE_MISSING, dtype=np.uint16)
    records = []
    for pos, raw in zip(chrom_pos[lo:hi], raw_values, strict=False):
        value_fraction, value_percent = decode_cohort_scalar(np.uint16(raw))
        records.append({"pos0": int(pos), "value_fraction": value_fraction, "value_percent": value_percent})
    return {
        "sample_id": sample_id,
        "track": key.name(),
        "chrom": chrom,
        "start_pos0": start_pos0,
        "end_pos0": end_pos0,
        "source_path": columns["source_path"][sample_idx],
        "input_tag": columns["input_tag"][sample_idx],
        "count": len(records),
        "records": records,
    }
