from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from mdb.create import load_index
from mdb.schema import TrackKey, VALUE_MISSING
from mdb.storage import (
    append_view_columns,
    cohort_backend,
    create_cohort_store,
    create_view_store,
    encode_cohort_dense,
    list_tracks,
    load_cohort_index,
    load_cohort_manifest,
    load_track_manifest,
    load_view_columns,
    read_track,
    sample_manifest,
)


@dataclass
class ColumnEntry:
    sample_id: str
    bundle_path: str
    platform: str
    source_path: str
    input_tag: str


def _normalize_backend(value: str | None) -> str:
    backend = (value or "npy").strip().lower()
    if backend not in {"npy", "zarr"}:
        raise ValueError(f"Unsupported cohort backend: {value}")
    return backend


def _zarr_config_from_args(args) -> dict[str, Any]:
    return {
        "row_chunk": max(int(getattr(args, "zarr_row_chunk", 65536)), 1),
        "codec": str(getattr(args, "zarr_codec", "zstd")),
        "clevel": int(getattr(args, "zarr_clevel", 5)),
        "shuffle": str(getattr(args, "zarr_shuffle", "bitshuffle")),
        "codec_threads": max(int(getattr(args, "zarr_codec_threads", 4)), 1),
    }


def _configure_zarr_threads(zarr_config: dict[str, Any], backend: str) -> None:
    if backend != "zarr":
        return
    try:
        from numcodecs import blosc  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "zarr backend requested but numcodecs is not installed. "
            "Install dependencies: zarr>=3.1.1 and numcodecs>=0.12."
        ) from exc
    blosc.set_nthreads(int(zarr_config["codec_threads"]))


def _ordered_payload(row_ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if row_ids.shape[0] < 2 or np.all(row_ids[1:] >= row_ids[:-1]):
        return row_ids, values
    order = np.argsort(row_ids, kind="stable")
    return row_ids[order], values[order]


def _expand_inputs(inputs: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in inputs:
        hits = glob.glob(item)
        expanded.extend(sorted(hits) if hits else [item])
    if len(expanded) == 1 and os.path.isfile(expanded[0]) and expanded[0].endswith(".txt"):
        with open(expanded[0], "r") as f:
            expanded = [line.strip() for line in f if line.strip()]
    out = [os.path.abspath(p) for p in expanded if os.path.isdir(p)]
    if not out:
        raise FileNotFoundError("No valid input sample bundles found.")
    return out


def _modifiedc_track_names(bundle_path: str) -> list[TrackKey]:
    raw = list_tracks(bundle_path)
    pairs = sorted({(key.haplotype, key.strand) for key in raw if key.assay in {"5mC", "5hmC"}})
    return [TrackKey(assay="modifiedC", haplotype=hap, strand=strand) for hap, strand in pairs]


def _collect_track_entries(bundle_paths: list[str], modifiedc: bool) -> tuple[str, dict[TrackKey, list[ColumnEntry]]]:
    index_path = None
    track_entries: dict[TrackKey, list[ColumnEntry]] = {}
    for bundle_path in bundle_paths:
        manifest = sample_manifest(bundle_path)
        sample_id = str(manifest["sample_id"])
        platform = str(manifest["platform"])
        bundle_index_path = str(manifest["index_path"])
        if index_path is None:
            index_path = bundle_index_path
        elif bundle_index_path != index_path:
            raise ValueError(f"All sample bundles must use the same index path. Saw {bundle_index_path} and {index_path}.")

        track_names = _modifiedc_track_names(bundle_path) if modifiedc else list_tracks(bundle_path)
        for key in track_names:
            if modifiedc:
                input_tag = key.haplotype
                source_path = bundle_path
            else:
                meta = load_track_manifest(bundle_path, key)
                input_tag = str(meta["input_tag"])
                source_path = str(meta["source_path"])
            track_entries.setdefault(key, []).append(
                ColumnEntry(
                    sample_id=sample_id,
                    bundle_path=bundle_path,
                    platform=platform,
                    source_path=source_path,
                    input_tag=input_tag,
                )
            )
    if index_path is None:
        raise RuntimeError("No input bundles were supplied.")
    return index_path, track_entries


def _validate_unique_sample_ids(track_entries: dict[TrackKey, list[ColumnEntry]]) -> None:
    for key, entries in track_entries.items():
        sample_ids = [entry.sample_id for entry in entries]
        dup = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
        if dup:
            raise ValueError(f"View {key.name()} has duplicate sample ids: {', '.join(dup)}")


def _combine_modifiedc_dense(values_a: np.ndarray | None, values_b: np.ndarray | None) -> np.ndarray | None:
    if values_a is None and values_b is None:
        return None
    if values_a is None:
        return np.asarray(values_b, dtype=np.uint16).copy()
    if values_b is None:
        return np.asarray(values_a, dtype=np.uint16).copy()

    out = np.asarray(values_a, dtype=np.uint16).copy()
    values_b = np.asarray(values_b, dtype=np.uint16)
    mask_a = out != VALUE_MISSING
    mask_b = values_b != VALUE_MISSING
    only_b = ~mask_a & mask_b
    both = mask_a & mask_b
    out[only_b] = values_b[only_b]
    if np.any(both):
        summed = out[both].astype(np.uint32) + values_b[both].astype(np.uint32)
        out[both] = np.minimum(summed, np.uint32(100 * 100)).astype(np.uint16, copy=False)
    return out


def _dense_chrom_values_for_bundle(bundle_path: str, key: TrackKey, modifiedc: bool) -> dict[str, np.ndarray]:
    if modifiedc:
        readers = []
        try:
            for assay in ("5mC", "5hmC"):
                source_key = TrackKey(assay=assay, haplotype=key.haplotype, strand=key.strand)
                try:
                    readers.append(read_track(bundle_path, source_key))
                except FileNotFoundError:
                    continue
            if not readers:
                raise KeyError(f"No source tracks found to build {key.name()} from {bundle_path}")
            out: dict[str, np.ndarray] = {}
            chroms_present = sorted({chrom for reader in readers for chrom in reader.chroms_present})
            for chrom in chroms_present:
                combined = None
                for reader in readers:
                    if chrom not in reader.chroms_present:
                        continue
                    combined = _combine_modifiedc_dense(combined, reader.chrom_values(chrom))
                if combined is not None and np.any(combined != VALUE_MISSING):
                    out[chrom] = encode_cohort_dense(combined)
            return out
        finally:
            for reader in readers:
                reader.close()

    track = read_track(bundle_path, key)
    try:
        out = {}
        for chrom in track.chroms_present:
            values = track.chrom_values(chrom)
            if np.any(values != VALUE_MISSING):
                out[chrom] = encode_cohort_dense(values)
        return out
    finally:
        track.close()


def _write_view_columns(writer, entries: list[ColumnEntry], key: TrackKey, modifiedc: bool) -> None:
    for col_idx, entry in tqdm(
        enumerate(entries),
        total=len(entries),
        desc=f"Writing {key.name()}",
        leave=False,
    ):
        for chrom, values in _dense_chrom_values_for_bundle(entry.bundle_path, key, modifiedc).items():
            writer.write_dense_chrom(chrom, col_idx, values)
    writer.flush()


def _encode_dense_slice(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint16)
    out = np.zeros(arr.shape, dtype=np.uint16)
    mask = arr != VALUE_MISSING
    out[mask] = (arr[mask].astype(np.uint32) + np.uint32(1)).astype(np.uint16, copy=False)
    return out


def _load_bundle_readers(bundle_path: str, key: TrackKey, modifiedc: bool):
    if not modifiedc:
        return [read_track(bundle_path, key)]
    readers = []
    for assay in ("5mC", "5hmC"):
        source_key = TrackKey(assay=assay, haplotype=key.haplotype, strand=key.strand)
        try:
            readers.append(read_track(bundle_path, source_key))
        except FileNotFoundError:
            continue
    return readers


def _write_view_blocks(writer, entries: list[ColumnEntry], key: TrackKey, modifiedc: bool, chrom_rows: dict[str, int], row_chunk: int) -> None:
    block_size = int(writer.block_size)
    for block_start in tqdm(range(0, len(entries), block_size), desc=f"Writing {key.name()} blocks", leave=False):
        block_entries = entries[block_start : block_start + block_size]
        reader_groups = []
        try:
            for entry in block_entries:
                readers = _load_bundle_readers(entry.bundle_path, key, modifiedc)
                reader_groups.append(readers)
            for chrom, chrom_len in chrom_rows.items():
                for row_lo in range(0, chrom_len, row_chunk):
                    row_hi = min(row_lo + row_chunk, chrom_len)
                    slab = np.zeros((row_hi - row_lo, len(block_entries)), dtype=np.uint16)
                    for local_col, readers in enumerate(reader_groups):
                        if not readers:
                            continue
                        combined = None
                        for reader in readers:
                            if chrom not in reader.chroms_present:
                                continue
                            chrom_vals = np.asarray(reader.chrom_values(chrom)[row_lo:row_hi], dtype=np.uint16)
                            if combined is None:
                                combined = chrom_vals.copy()
                            else:
                                mask_a = combined != VALUE_MISSING
                                mask_b = chrom_vals != VALUE_MISSING
                                only_b = ~mask_a & mask_b
                                both = mask_a & mask_b
                                combined[only_b] = chrom_vals[only_b]
                                if np.any(both):
                                    summed = combined[both].astype(np.uint32) + chrom_vals[both].astype(np.uint32)
                                    combined[both] = np.minimum(summed, np.uint32(100 * 100)).astype(np.uint16, copy=False)
                        if combined is not None:
                            slab[:, local_col] = _encode_dense_slice(combined)
                    writer.write_dense_chrom_block(
                        chrom,
                        row_lo,
                        row_hi,
                        block_start,
                        block_start + len(block_entries),
                        slab,
                    )
        finally:
            for readers in reader_groups:
                for reader in readers:
                    reader.close()
    writer.flush()


def _build_view_store(task: tuple[str, str, list[ColumnEntry], bool, list[str], np.ndarray, int, int, str, dict[str, Any]]) -> str:
    output_path, key_name, entries, modifiedc, chroms, chrom_offsets, n_rows, block_size, backend, zarr_config = task
    key = TrackKey.from_name(key_name)
    writer = create_view_store(
        output_path,
        key=key,
        chroms=chroms,
        chrom_offsets=chrom_offsets,
        n_rows=n_rows,
        sample_ids=[e.sample_id for e in entries],
        bundle_paths=[e.bundle_path for e in entries],
        platforms=[e.platform for e in entries],
        source_paths=[e.source_path for e in entries],
        input_tags=[e.input_tag for e in entries],
        block_size=block_size,
    )
    try:
        if backend == "zarr":
            chrom_ends = np.asarray(
                [int(chrom_offsets[i + 1]) if i + 1 < len(chrom_offsets) else int(n_rows) for i in range(len(chroms))],
                dtype=np.int64,
            )
            chrom_rows = {
                chrom: int(chrom_end - chrom_start)
                for chrom, chrom_start, chrom_end in zip(chroms, chrom_offsets, chrom_ends, strict=False)
            }
            _write_view_blocks(
                writer,
                entries,
                key,
                modifiedc,
                chrom_rows=chrom_rows,
                row_chunk=int(zarr_config["row_chunk"]),
            )
        else:
            _write_view_columns(writer, entries, key, modifiedc)
    finally:
        writer.close()
    return key_name


def _write_parallel_views(
    output_path: str,
    track_entries: dict[TrackKey, list[ColumnEntry]],
    modifiedc: bool,
    chroms: list[str],
    chrom_offsets: np.ndarray,
    n_rows: int,
    block_size: int,
    workers: int,
    backend: str,
    zarr_config: dict[str, Any],
) -> None:
    future_map = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(track_entries))) as pool:
        for key in sorted(track_entries):
            task = (
                output_path,
                key.name(),
                track_entries[key],
                modifiedc,
                chroms,
                chrom_offsets,
                n_rows,
                block_size,
                backend,
                zarr_config,
            )
            future_map[pool.submit(_build_view_store, task)] = key
        for future in as_completed(future_map):
            future.result()


def _write_sequential_views(
    output_path: str,
    track_entries: dict[TrackKey, list[ColumnEntry]],
    modifiedc: bool,
    chroms: list[str],
    chrom_offsets: np.ndarray,
    n_rows: int,
    block_size: int,
    bundle_paths: list[str],
    backend: str,
    zarr_config: dict[str, Any],
) -> None:
    if backend == "zarr":
        for key in sorted(track_entries):
            _build_view_store(
                (
                    output_path,
                    key.name(),
                    track_entries[key],
                    modifiedc,
                    chroms,
                    chrom_offsets,
                    n_rows,
                    block_size,
                    backend,
                    zarr_config,
                )
            )
        return

    writers: dict[TrackKey, object] = {}
    column_lookup: dict[TrackKey, dict[str, int]] = {}
    bundle_keys: dict[str, list[TrackKey]] = {}

    for key in sorted(track_entries):
        entries = track_entries[key]
        writers[key] = create_view_store(
            output_path,
            key=key,
            chroms=chroms,
            chrom_offsets=chrom_offsets,
            n_rows=n_rows,
            sample_ids=[e.sample_id for e in entries],
            bundle_paths=[e.bundle_path for e in entries],
            platforms=[e.platform for e in entries],
            source_paths=[e.source_path for e in entries],
            input_tags=[e.input_tag for e in entries],
            block_size=block_size,
        )
        column_lookup[key] = {entry.bundle_path: idx for idx, entry in enumerate(entries)}
        for entry in entries:
            bundle_keys.setdefault(entry.bundle_path, []).append(key)

    try:
        for bundle_path in tqdm(bundle_paths, desc="Writing sample bundles"):
            for key in sorted(bundle_keys.get(bundle_path, [])):
                for chrom, values in _dense_chrom_values_for_bundle(bundle_path, key, modifiedc).items():
                    writers[key].write_dense_chrom(chrom, column_lookup[key][bundle_path], values)
        for writer in writers.values():
            writer.flush()
    finally:
        for writer in writers.values():
            writer.close()


def merge_main(args):
    bundle_paths = _expand_inputs(list(args.inputs))
    modifiedc = bool(args.modifiedc)
    workers = max(int(getattr(args, "workers", 1)), 1)
    block_size = max(int(getattr(args, "block_size", 64)), 1)
    backend = _normalize_backend(getattr(args, "cohort_backend", "npy"))
    zarr_config = _zarr_config_from_args(args)
    _configure_zarr_threads(zarr_config, backend)

    index_path, track_entries = _collect_track_entries(bundle_paths, modifiedc)
    _validate_unique_sample_ids(track_entries)
    index_payload = load_index(index_path)
    if len(index_payload) >= 3:
        chroms = index_payload[0]
        chrom_offsets = index_payload[1]
        pos0 = index_payload[2]
    else:
        raise RuntimeError(f"Unexpected index payload from load_index({index_path})")
    n_rows = int(pos0.shape[0])

    create_cohort_store(
        args.output,
        chroms,
        chrom_offsets,
        pos0,
        index_path=index_path,
        block_size=block_size,
        backend=backend,
        zarr_row_chunk=int(zarr_config["row_chunk"]),
        zarr_codec=str(zarr_config["codec"]),
        zarr_clevel=int(zarr_config["clevel"]),
        zarr_shuffle=str(zarr_config["shuffle"]),
    )
    print(f"Merging {len(bundle_paths)} sample bundles into {args.output}")

    if workers > 1 and len(track_entries) > 1:
        _write_parallel_views(
            args.output,
            track_entries,
            modifiedc,
            chroms,
            chrom_offsets,
            n_rows,
            block_size,
            workers,
            backend,
            zarr_config,
        )
    else:
        _write_sequential_views(
            args.output,
            track_entries,
            modifiedc,
            chroms,
            chrom_offsets,
            n_rows,
            block_size,
            bundle_paths,
            backend,
            zarr_config,
        )

    print(f"Wrote cohort store: {args.output}")
    print("Views:")
    for key in sorted(track_entries):
        print(f"  {key.name()} columns={len(track_entries[key])}")


def append_main(args):
    bundle_paths = _expand_inputs(list(args.inputs))
    modifiedc = bool(args.modifiedc)
    cohort_path = args.cohort

    manifest = load_cohort_manifest(cohort_path)
    existing_backend = cohort_backend(cohort_path)
    requested_backend = _normalize_backend(getattr(args, "cohort_backend", existing_backend))
    if requested_backend != existing_backend:
        print(
            f"Warning: ignoring --cohort-backend={requested_backend}; "
            f"existing cohort backend is {existing_backend}"
        )
    zarr_config = _zarr_config_from_args(args)
    _configure_zarr_threads(zarr_config, existing_backend)

    index_path, track_entries = _collect_track_entries(bundle_paths, modifiedc)
    _validate_unique_sample_ids(track_entries)
    existing_index_path = str(manifest.get("index_path") or index_path)
    if existing_index_path != index_path:
        raise ValueError(f"Cohort store index path {existing_index_path} does not match sample bundle index path {index_path}")

    chroms, chrom_offsets, pos0 = load_cohort_index(cohort_path)
    block_size = int(manifest["block_size"])
    n_rows = int(pos0.shape[0])

    for key in sorted(track_entries):
        entries = track_entries[key]
        view_dir = os.path.join(cohort_path, "views", key.name())
        if os.path.isdir(view_dir):
            existing_cols = load_view_columns(cohort_path, key)
            existing_ids = set(existing_cols["sample_id"])
            dup = sorted(existing_ids.intersection(e.sample_id for e in entries))
            if dup:
                raise ValueError(f"View {key.name()} already contains sample ids: {', '.join(dup)}")
            writer, start_col = append_view_columns(
                cohort_path,
                key,
                sample_ids=[e.sample_id for e in entries],
                bundle_paths=[e.bundle_path for e in entries],
                platforms=[e.platform for e in entries],
                source_paths=[e.source_path for e in entries],
                input_tags=[e.input_tag for e in entries],
            )
            try:
                for offset, entry in enumerate(entries, start=start_col):
                    for chrom, values in _dense_chrom_values_for_bundle(entry.bundle_path, key, modifiedc).items():
                        writer.write_dense_chrom(chrom, offset, values)
                writer.flush()
            finally:
                writer.close()
        else:
            writer = create_view_store(
                cohort_path,
                key=key,
                chroms=chroms,
                chrom_offsets=chrom_offsets,
                n_rows=n_rows,
                sample_ids=[e.sample_id for e in entries],
                bundle_paths=[e.bundle_path for e in entries],
                platforms=[e.platform for e in entries],
                source_paths=[e.source_path for e in entries],
                input_tags=[e.input_tag for e in entries],
                block_size=block_size,
            )
            try:
                _write_view_columns(writer, entries, key, modifiedc)
            finally:
                writer.close()

    print(f"Appended {len(bundle_paths)} sample bundles into {cohort_path}")
    print("Views updated:")
    for key in sorted(track_entries):
        print(f"  {key.name()} +{len(track_entries[key])} columns")
