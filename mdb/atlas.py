from __future__ import annotations

import gzip
import json
import os

import numpy as np

from mdb.schema import TrackKey, VALUE_DENOMINATOR
from mdb.storage import (
    available_views,
    create_cohort_store,
    create_view_store,
    load_view_reader,
)


def _open_text(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _read_sample_ids(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        sample_ids = [line.strip() for line in handle if line.strip()]
    if not sample_ids:
        raise ValueError(f"No sample ids found in {path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample ids found in {path}")
    return sample_ids


def _read_sniffcell_index(path: str, n_rows: int) -> dict[str, np.ndarray | list[str]]:
    starts = np.empty(n_rows, dtype=np.uint32)
    ends = np.empty(n_rows, dtype=np.uint32)
    source_starts = np.empty(n_rows, dtype=np.int64)
    source_ends = np.empty(n_rows, dtype=np.int64)
    chroms: list[str] = []
    chrom_offsets: list[int] = []
    last_chrom = None

    with _open_text(path) as handle:
        for row_idx, line in enumerate(handle):
            if row_idx >= n_rows:
                raise ValueError(f"Index has more than {n_rows:,} rows: {path}")
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                raise ValueError(f"Index row {row_idx + 1:,} has fewer than five columns")
            chrom = fields[0]
            if chrom != last_chrom:
                if chrom in chroms:
                    raise ValueError(f"Chromosome {chrom} appears in multiple non-contiguous blocks")
                chroms.append(chrom)
                chrom_offsets.append(row_idx)
                last_chrom = chrom
            starts[row_idx] = int(fields[1])
            ends[row_idx] = int(fields[2])
            source_starts[row_idx] = int(fields[3])
            source_ends[row_idx] = int(fields[4])
        observed_rows = row_idx + 1 if n_rows else 0

    if observed_rows != n_rows:
        raise ValueError(f"Index row count {observed_rows:,} does not match matrix rows {n_rows:,}")
    if np.any(ends <= starts):
        raise ValueError("Every SniffCell index interval must have end > start")
    return {
        "chroms": chroms,
        "chrom_offsets": np.asarray(chrom_offsets, dtype=np.int64),
        "start": starts,
        "end": ends,
        "source_row_start": source_starts,
        "source_row_end": source_ends,
    }


def _chrom_bounds(chroms: list[str], offsets: np.ndarray, n_rows: int) -> dict[str, tuple[int, int]]:
    return {
        chrom: (int(offsets[i]), int(offsets[i + 1]) if i + 1 < len(offsets) else int(n_rows))
        for i, chrom in enumerate(chroms)
    }


def _encode_fraction(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if np.any(arr[finite] < -1e-6) or np.any(arr[finite] > 1.0 + 1e-6):
        raise ValueError("Atlas methylation values must be fractions within [0, 1]")
    out = np.zeros(arr.shape, dtype=np.uint16)
    if np.any(finite):
        scaled = np.rint(np.clip(arr[finite], 0.0, 1.0) * VALUE_DENOMINATOR).astype(np.uint32)
        out[finite] = (scaled + np.uint32(1)).astype(np.uint16)
    return out


def _write_groups_npz(output: str, index: dict[str, np.ndarray | list[str]]) -> None:
    np.savez(
        os.path.join(output, "groups.npz"),
        method=np.asarray("sniffcell_loyfer", dtype=object),
        chroms=np.asarray(index["chroms"], dtype=object),
        chrom_offsets=np.asarray(index["chrom_offsets"], dtype=np.int64),
        start=np.asarray(index["start"], dtype=np.uint32),
        end=np.asarray(index["end"], dtype=np.uint32),
        source_row_start=np.asarray(index["source_row_start"], dtype=np.int64),
        source_row_end=np.asarray(index["source_row_end"], dtype=np.int64),
        reference_start=np.asarray(index["start"], dtype=np.uint32),
        reference_end=np.asarray(index["end"], dtype=np.uint32),
    )


def _write_legacy_view(
    *,
    output: str,
    matrix: np.ndarray,
    sample_ids: list[str],
    key: TrackKey,
    chroms: list[str],
    chrom_offsets: np.ndarray,
    batch_rows: int,
    block_size: int,
    source_path: str,
) -> None:
    writer = create_view_store(
        output,
        key=key,
        chroms=chroms,
        chrom_offsets=chrom_offsets,
        n_rows=int(matrix.shape[0]),
        sample_ids=sample_ids,
        bundle_paths=[source_path] * len(sample_ids),
        platforms=["legacy_atlas"] * len(sample_ids),
        source_paths=[source_path] * len(sample_ids),
        input_tags=[key.assay] * len(sample_ids),
        block_size=block_size,
    )
    bounds = _chrom_bounds(chroms, chrom_offsets, int(matrix.shape[0]))
    try:
        for chrom in chroms:
            global_start, global_end = bounds[chrom]
            chrom_rows = global_end - global_start
            for local_start in range(0, chrom_rows, batch_rows):
                local_end = min(local_start + batch_rows, chrom_rows)
                payload = _encode_fraction(matrix[global_start + local_start : global_start + local_end, :])
                writer.write_dense_chrom_block(
                    chrom,
                    local_start,
                    local_end,
                    0,
                    len(sample_ids),
                    payload,
                )
        writer.flush()
    finally:
        writer.close()


def _source_group_coordinates(source: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    path = os.path.join(source, "groups.npz")
    if not os.path.isfile(path):
        raise ValueError(f"Source cohort must be grouped and contain groups.npz: {source}")
    groups = np.load(path, allow_pickle=True)
    required = {"chroms", "chrom_offsets", "reference_start", "reference_end"}
    missing = required - set(groups.files)
    if missing:
        raise ValueError(f"Source groups.npz is missing: {', '.join(sorted(missing))}")
    chroms = [str(x) for x in groups["chroms"]]
    offsets = np.asarray(groups["chrom_offsets"], dtype=np.int64)
    starts = np.asarray(groups["reference_start"], dtype=np.uint32)
    ends = np.asarray(groups["reference_end"], dtype=np.uint32)
    bounds = _chrom_bounds(chroms, offsets, len(starts))
    return {
        chrom: (
            np.arange(lo, hi, dtype=np.int64),
            starts[lo:hi],
            ends[lo:hi],
        )
        for chrom, (lo, hi) in bounds.items()
    }


def _align_source_rows(
    *,
    source: str,
    target_chroms: list[str],
    target_offsets: np.ndarray,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    source_coords = _source_group_coordinates(source)
    target_bounds = _chrom_bounds(target_chroms, target_offsets, len(target_starts))
    alignment: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    matched = 0
    total = 0
    for chrom, (source_rows, source_starts, source_ends) in source_coords.items():
        total += len(source_rows)
        if chrom not in target_bounds:
            continue
        target_lo, target_hi = target_bounds[chrom]
        chrom_target_starts = target_starts[target_lo:target_hi]
        chrom_target_ends = target_ends[target_lo:target_hi]
        local_target = np.searchsorted(chrom_target_starts, source_starts, side="left")
        valid = local_target < len(chrom_target_starts)
        valid[valid] &= chrom_target_starts[local_target[valid]] == source_starts[valid]
        valid[valid] &= chrom_target_ends[local_target[valid]] == source_ends[valid]
        if not np.all(valid):
            first = int(np.flatnonzero(~valid)[0])
            raise ValueError(
                f"Source row does not exactly match the legacy Loyfer index: "
                f"{chrom}:{int(source_starts[first])}-{int(source_ends[first])}"
            )
        alignment[chrom] = (source_rows, local_target.astype(np.int64, copy=False))
        matched += len(source_rows)
    return alignment, {"source_rows": int(total), "matched_rows": int(matched)}


def _write_source_view(
    *,
    source: str,
    output: str,
    key: TrackKey,
    alignment: dict[str, tuple[np.ndarray, np.ndarray]],
    target_chroms: list[str],
    target_offsets: np.ndarray,
    n_target_rows: int,
    batch_rows: int,
    block_size: int,
) -> None:
    reader, columns, _ = load_view_reader(source, key)
    sample_ids = list(columns["sample_id"])
    writer = create_view_store(
        output,
        key=key,
        chroms=target_chroms,
        chrom_offsets=target_offsets,
        n_rows=n_target_rows,
        sample_ids=sample_ids,
        bundle_paths=list(columns["bundle_path"]),
        platforms=list(columns["platform"]),
        source_paths=list(columns["source_path"]),
        input_tags=list(columns["input_tag"]),
        block_size=block_size,
    )
    target_bounds = _chrom_bounds(target_chroms, target_offsets, n_target_rows)
    try:
        for chrom, (source_rows, target_local_rows) in alignment.items():
            target_lo, target_hi = target_bounds[chrom]
            chrom_rows = target_hi - target_lo
            for local_start in range(0, chrom_rows, batch_rows):
                local_end = min(local_start + batch_rows, chrom_rows)
                map_lo = int(np.searchsorted(target_local_rows, local_start, side="left"))
                map_hi = int(np.searchsorted(target_local_rows, local_end, side="left"))
                if map_hi <= map_lo:
                    continue
                selected_source_rows = source_rows[map_lo:map_hi]
                selected_target_rows = target_local_rows[map_lo:map_hi] - local_start
                selected_values = reader.read_rows(selected_source_rows)
                payload = np.zeros((local_end - local_start, len(sample_ids)), dtype=np.uint16)
                payload[selected_target_rows, :] = _encode_fraction(selected_values)
                writer.write_dense_chrom_block(
                    chrom,
                    local_start,
                    local_end,
                    0,
                    len(sample_ids),
                    payload,
                )
        writer.flush()
    finally:
        reader.close()
        writer.close()


def atlas_main(args) -> None:
    legacy_npy = os.path.abspath(args.legacy_npy)
    legacy_index = os.path.abspath(args.legacy_index)
    legacy_samples = os.path.abspath(args.legacy_samples)
    source_cohorts = [os.path.abspath(path) for path in args.cohorts]
    output = os.path.abspath(args.output)
    batch_rows = max(int(args.batch_rows), 1)
    block_size = max(int(args.block_size), 1)

    matrix = np.load(legacy_npy, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Legacy NPY matrix must be two-dimensional; found {matrix.shape}")
    sample_ids = _read_sample_ids(legacy_samples)
    if matrix.shape[1] != len(sample_ids):
        raise ValueError(
            f"Legacy sample count {len(sample_ids):,} does not match matrix columns {matrix.shape[1]:,}"
        )
    index = _read_sniffcell_index(legacy_index, int(matrix.shape[0]))
    chroms = list(index["chroms"])
    chrom_offsets = np.asarray(index["chrom_offsets"], dtype=np.int64)
    starts = np.asarray(index["start"], dtype=np.uint32)
    ends = np.asarray(index["end"], dtype=np.uint32)

    create_cohort_store(
        output,
        chroms=chroms,
        chrom_offsets=chrom_offsets,
        pos0=starts,
        index_path=legacy_index,
        block_size=block_size,
        backend=args.cohort_backend,
        zarr_row_chunk=int(args.zarr_row_chunk),
        zarr_codec=args.zarr_codec,
        zarr_clevel=int(args.zarr_clevel),
        zarr_shuffle=args.zarr_shuffle,
    )
    _write_groups_npz(output, index)

    legacy_key = TrackKey(args.legacy_assay, "combined", "combined")
    _write_legacy_view(
        output=output,
        matrix=matrix,
        sample_ids=sample_ids,
        key=legacy_key,
        chroms=chroms,
        chrom_offsets=chrom_offsets,
        batch_rows=batch_rows,
        block_size=block_size,
        source_path=legacy_npy,
    )

    source_summaries = []
    written_keys = {legacy_key}
    for source in source_cohorts:
        alignment, summary = _align_source_rows(
            source=source,
            target_chroms=chroms,
            target_offsets=chrom_offsets,
            target_starts=starts,
            target_ends=ends,
        )
        views = available_views(source)
        for key in views:
            if key in written_keys:
                raise ValueError(
                    f"Track {key.name()} is supplied by more than one atlas source; "
                    "use distinct assays or build separate source cohorts"
                )
            _write_source_view(
                source=source,
                output=output,
                key=key,
                alignment=alignment,
                target_chroms=chroms,
                target_offsets=chrom_offsets,
                n_target_rows=int(matrix.shape[0]),
                batch_rows=batch_rows,
                block_size=block_size,
            )
            written_keys.add(key)
        source_summaries.append(
            {
                "path": source,
                "views": [key.name() for key in views],
                **summary,
            }
        )

    manifest = {
        "kind": "combined_sniffcell_loyfer_atlas",
        "legacy_npy": legacy_npy,
        "legacy_index": legacy_index,
        "legacy_samples": legacy_samples,
        "legacy_view": legacy_key.name(),
        "legacy_rows": int(matrix.shape[0]),
        "legacy_samples_count": int(matrix.shape[1]),
        "source_cohorts": source_summaries,
        "views": sorted(key.name() for key in written_keys),
    }
    with open(os.path.join(output, "atlas_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
