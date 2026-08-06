from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.resources as importlib_resources
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mdb.schema import TrackKey, VALUE_DENOMINATOR
from mdb.storage import (
    available_views,
    cohort_backend,
    create_cohort_store,
    create_view_store,
    load_cohort_index,
    load_cohort_manifest,
    load_view_columns,
    load_view_reader,
)

GROUP_INDEX_FILE = "groups.npz"
GROUP_TABLE_FILE = "groups.tsv.gz"
GROUP_SUMMARY_FILE = "grouping_summary.json"
PREDEFINED_RESOURCE_FILES = {
    "loyfer": "loyfer_grch38_v1.npz",
    "decode": "decode_grch38_10bp_v1.npz",
}

DECODE_PAPER_URL = "https://www.nature.com/articles/s41588-024-01851-2"
LOYFER_PAPER_URL = "https://www.nature.com/articles/s41586-022-05580-6"


@dataclass(frozen=True)
class GroupIndex:
    """Compact mapping from reduced features to contiguous source-CpG rows."""

    method: str
    chroms: list[str]
    chrom_offsets: np.ndarray
    start: np.ndarray
    end: np.ndarray
    source_row_start: np.ndarray
    source_row_end: np.ndarray
    reference_start: np.ndarray
    reference_end: np.ndarray

    @property
    def n_groups(self) -> int:
        return int(self.start.shape[0])

    @property
    def n_source_cpgs(self) -> int:
        return int(np.sum(self.source_row_end - self.source_row_start, dtype=np.int64))

    def validate(self, source_n_rows: int) -> None:
        arrays = (
            self.end,
            self.source_row_start,
            self.source_row_end,
            self.reference_start,
            self.reference_end,
        )
        if any(arr.shape != self.start.shape for arr in arrays):
            raise ValueError("All group-index arrays must have the same shape")
        if self.chrom_offsets.shape[0] != len(self.chroms):
            raise ValueError("chrom_offsets length must match chroms")
        if self.n_groups == 0:
            raise ValueError("The grouping contains no CpGs from the input cohort")
        if np.any(self.end <= self.start):
            raise ValueError("Group genomic intervals must have end > start")
        if np.any(self.source_row_end <= self.source_row_start):
            raise ValueError("Every group must contain at least one source CpG")
        if int(self.source_row_start.min()) < 0 or int(self.source_row_end.max()) > source_n_rows:
            raise ValueError("Group source rows fall outside the input cohort index")
        if self.n_groups > 1 and np.any(self.source_row_start[1:] < self.source_row_end[:-1]):
            raise ValueError("Groups must be sorted and may not overlap source CpG rows")


@dataclass(frozen=True)
class PredefinedGroupingIndex:
    """Versioned genomic intervals for a named, reference-based grouping."""

    method: str
    version: str
    assembly: str
    chroms: list[str]
    chrom_offsets: np.ndarray
    start: np.ndarray
    end: np.ndarray
    left_shift_bp: int
    parameters: dict
    citation: str
    source: str

    @property
    def n_groups(self) -> int:
        return int(self.start.shape[0])

    def validate(self) -> None:
        if self.method not in PREDEFINED_RESOURCE_FILES:
            raise ValueError(f"Unsupported predefined grouping method: {self.method}")
        if self.chrom_offsets.shape[0] != len(self.chroms):
            raise ValueError("Predefined chrom_offsets length must match chroms")
        if self.start.shape != self.end.shape or self.n_groups == 0:
            raise ValueError("Predefined index requires non-empty, same-length start/end arrays")
        if np.any(self.end <= self.start):
            raise ValueError("Predefined intervals must have end > start")
        if self.left_shift_bp < 0:
            raise ValueError("left_shift_bp must be >= 0")


def _chrom_ends(chrom_offsets: np.ndarray, n_rows: int) -> np.ndarray:
    return np.asarray(
        [
            int(chrom_offsets[i + 1]) if i + 1 < len(chrom_offsets) else int(n_rows)
            for i in range(len(chrom_offsets))
        ],
        dtype=np.int64,
    )


def _concat(parts: list[np.ndarray], dtype) -> np.ndarray:
    if not parts:
        return np.asarray([], dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def save_predefined_index(path: str | Path, index: PredefinedGroupingIndex) -> None:
    index.validate()
    metadata = {
        "method": index.method,
        "version": index.version,
        "assembly": index.assembly,
        "left_shift_bp": int(index.left_shift_bp),
        "parameters": index.parameters,
        "citation": index.citation,
        "source": index.source,
        "coordinate_encoding": "per_chrom_start_delta_and_length_v1",
    }
    chrom_ends = _chrom_ends(index.chrom_offsets, index.n_groups)
    start_delta = np.empty(index.start.shape, dtype=np.uint32)
    for group_lo, group_hi in zip(index.chrom_offsets, chrom_ends, strict=False):
        group_lo = int(group_lo)
        group_hi = int(group_hi)
        if group_hi <= group_lo:
            continue
        starts = index.start[group_lo:group_hi].astype(np.uint64)
        start_delta[group_lo] = np.uint32(starts[0])
        if starts.shape[0] > 1:
            start_delta[group_lo + 1 : group_hi] = np.diff(starts).astype(np.uint32)
    length = (index.end.astype(np.uint64) - index.start.astype(np.uint64)).astype(np.uint32)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        chroms=np.asarray(index.chroms, dtype=str),
        chrom_offsets=np.asarray(index.chrom_offsets, dtype=np.int64),
        start_delta=start_delta,
        length=length,
    )


def _read_predefined_index(path_or_file) -> PredefinedGroupingIndex:
    with np.load(path_or_file, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        chroms = [str(x) for x in payload["chroms"]]
        chrom_offsets = np.asarray(payload["chrom_offsets"], dtype=np.int64)
        if "start_delta" in payload.files:
            start_delta = np.asarray(payload["start_delta"], dtype=np.uint32)
            start = np.empty(start_delta.shape, dtype=np.uint32)
            chrom_ends = _chrom_ends(chrom_offsets, int(start_delta.shape[0]))
            for group_lo, group_hi in zip(chrom_offsets, chrom_ends, strict=False):
                group_lo = int(group_lo)
                group_hi = int(group_hi)
                if group_hi > group_lo:
                    start[group_lo:group_hi] = np.cumsum(
                        start_delta[group_lo:group_hi], dtype=np.uint64
                    ).astype(np.uint32)
            end = (start.astype(np.uint64) + np.asarray(payload["length"], dtype=np.uint32)).astype(np.uint32)
        else:
            start = np.asarray(payload["start"], dtype=np.uint32)
            end = np.asarray(payload["end"], dtype=np.uint32)
        index = PredefinedGroupingIndex(
            method=str(metadata["method"]),
            version=str(metadata["version"]),
            assembly=str(metadata["assembly"]),
            chroms=chroms,
            chrom_offsets=chrom_offsets,
            start=start,
            end=end,
            left_shift_bp=int(metadata.get("left_shift_bp", 0)),
            parameters=dict(metadata.get("parameters", {})),
            citation=str(metadata.get("citation", "")),
            source=str(metadata.get("source", "")),
        )
    index.validate()
    return index


def resolve_predefined_index(method: str, index_dir: str | None = None) -> tuple[PredefinedGroupingIndex, str, str]:
    method = str(method).strip().lower()
    filename = PREDEFINED_RESOURCE_FILES.get(method)
    if filename is None:
        raise ValueError(f"No predefined grouping index for method: {method}")

    candidate_dirs: list[Path] = []
    if index_dir:
        candidate_dirs.append(Path(index_dir))
    env_dir = os.environ.get("MDB_GROUP_INDEX_DIR")
    if env_dir:
        candidate_dirs.append(Path(env_dir))
    for directory in candidate_dirs:
        candidate = directory / filename
        if candidate.is_file():
            resolved = candidate.resolve()
            return _read_predefined_index(resolved), str(resolved), _sha256(resolved)

    resource = importlib_resources.files("mdb").joinpath("resources", filename)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Stored {method} grouping index is missing ({filename}). Reinstall methdb with package data "
            "or set MDB_GROUP_INDEX_DIR to a directory containing the versioned indexes."
        )
    with importlib_resources.as_file(resource) as resource_path:
        resolved = Path(resource_path)
        index = _read_predefined_index(resolved)
        digest = _sha256(resolved)
        display_path = f"package:mdb/resources/{filename}"
    return index, display_path, digest


def map_predefined_index(input_path: str, predefined: PredefinedGroupingIndex) -> GroupIndex:
    """Map fixed GRCh38 intervals to the CpG rows present in an input cohort."""
    chroms, chrom_offsets, pos0 = load_cohort_index(input_path)
    source_ends = _chrom_ends(chrom_offsets, int(pos0.shape[0]))
    ref_ends = _chrom_ends(predefined.chrom_offsets, predefined.n_groups)
    ref_lookup = {chrom: i for i, chrom in enumerate(predefined.chroms)}
    output_offsets: list[int] = []
    fields: dict[str, list[np.ndarray]] = {
        name: [] for name in ("start", "end", "row_start", "row_end", "ref_start", "ref_end")
    }
    running = 0

    for chrom_i, chrom in enumerate(chroms):
        output_offsets.append(running)
        ref_i = ref_lookup.get(chrom)
        if ref_i is None:
            continue
        source_lo = int(chrom_offsets[chrom_i])
        source_hi = int(source_ends[chrom_i])
        local_pos = np.asarray(pos0[source_lo:source_hi], dtype=np.int64)
        group_lo = int(predefined.chrom_offsets[ref_i])
        group_hi = int(ref_ends[ref_i])
        ref_start = predefined.start[group_lo:group_hi].astype(np.int64)
        ref_end = predefined.end[group_lo:group_hi].astype(np.int64)
        query_start = np.maximum(ref_start - int(predefined.left_shift_bp), 0)
        local_start = np.searchsorted(local_pos, query_start, side="left")
        local_end = np.searchsorted(local_pos, ref_end, side="left")
        keep = local_end > local_start
        if not np.any(keep):
            continue
        local_start = local_start[keep]
        local_end = local_end[keep]
        fields["start"].append(local_pos[local_start].astype(np.uint32))
        fields["end"].append((local_pos[local_end - 1].astype(np.uint64) + 2).astype(np.uint32))
        fields["row_start"].append((local_start + source_lo).astype(np.int64))
        fields["row_end"].append((local_end + source_lo).astype(np.int64))
        fields["ref_start"].append(ref_start[keep].astype(np.uint32))
        fields["ref_end"].append(ref_end[keep].astype(np.uint32))
        running += int(np.sum(keep))

    groups = GroupIndex(
        method=predefined.method,
        chroms=list(chroms),
        chrom_offsets=np.asarray(output_offsets, dtype=np.int64),
        start=_concat(fields["start"], np.uint32),
        end=_concat(fields["end"], np.uint32),
        source_row_start=_concat(fields["row_start"], np.int64),
        source_row_end=_concat(fields["row_end"], np.int64),
        reference_start=_concat(fields["ref_start"], np.uint32),
        reference_end=_concat(fields["ref_end"], np.uint32),
    )
    groups.validate(int(pos0.shape[0]))
    return groups


def _build_group_index_from_breaks(
    *,
    method: str,
    chroms: list[str],
    chrom_offsets: np.ndarray,
    pos0: np.ndarray,
    breaks_by_chrom: list[np.ndarray],
) -> GroupIndex:
    source_ends = _chrom_ends(chrom_offsets, int(pos0.shape[0]))
    group_offsets: list[int] = []
    starts_out: list[np.ndarray] = []
    ends_out: list[np.ndarray] = []
    row_starts_out: list[np.ndarray] = []
    row_ends_out: list[np.ndarray] = []
    running = 0

    for chrom_i, _chrom in enumerate(chroms):
        group_offsets.append(running)
        row_lo = int(chrom_offsets[chrom_i])
        row_hi = int(source_ends[chrom_i])
        local_pos = np.asarray(pos0[row_lo:row_hi], dtype=np.uint32)
        if local_pos.size == 0:
            continue
        break_before = np.asarray(breaks_by_chrom[chrom_i], dtype=bool)
        if break_before.shape[0] != local_pos.shape[0]:
            raise ValueError(f"Break-vector length mismatch for {_chrom}")
        break_before[0] = True
        local_starts = np.flatnonzero(break_before).astype(np.int64)
        local_ends = np.r_[local_starts[1:], local_pos.shape[0]].astype(np.int64)
        starts_out.append(local_pos[local_starts])
        ends_out.append((local_pos[local_ends - 1].astype(np.uint64) + 2).astype(np.uint32))
        row_starts_out.append(local_starts + row_lo)
        row_ends_out.append(local_ends + row_lo)
        running += int(local_starts.shape[0])

    start = _concat(starts_out, np.uint32)
    end = _concat(ends_out, np.uint32)
    return GroupIndex(
        method=method,
        chroms=list(chroms),
        chrom_offsets=np.asarray(group_offsets, dtype=np.int64),
        start=start,
        end=end,
        source_row_start=_concat(row_starts_out, np.int64),
        source_row_end=_concat(row_ends_out, np.int64),
        reference_start=start.copy(),
        reference_end=end.copy(),
    )


def _adjacent_correlations(block: np.ndarray, min_shared_samples: int) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(block[:-1], dtype=np.float32)
    right = np.asarray(block[1:], dtype=np.float32)
    valid = np.isfinite(left) & np.isfinite(right)
    n = valid.sum(axis=1, dtype=np.int32)
    x = np.where(valid, left, 0.0)
    y = np.where(valid, right, 0.0)
    sx = x.sum(axis=1, dtype=np.float64)
    sy = y.sum(axis=1, dtype=np.float64)
    sxx = np.square(x, dtype=np.float64).sum(axis=1)
    syy = np.square(y, dtype=np.float64).sum(axis=1)
    sxy = (x.astype(np.float64) * y.astype(np.float64)).sum(axis=1)
    numerator = n.astype(np.float64) * sxy - sx * sy
    denom = np.sqrt(
        np.maximum(n.astype(np.float64) * sxx - sx * sx, 0.0)
        * np.maximum(n.astype(np.float64) * syy - sy * sy, 0.0)
    )
    corr = np.full(n.shape[0], np.nan, dtype=np.float64)
    usable = (n >= int(min_shared_samples)) & (denom > 0)
    corr[usable] = numerator[usable] / denom[usable]
    return corr, n


def build_denovo_groups(
    input_path: str,
    key: TrackKey,
    *,
    min_correlation: float = 0.8,
    max_gap_bp: int = 200,
    min_shared_samples: int = 3,
    batch_rows: int = 10_000,
    logger: logging.Logger | None = None,
) -> GroupIndex:
    """Learn groups as runs of adjacent, cross-sample-correlated CpGs."""
    if not -1.0 <= float(min_correlation) <= 1.0:
        raise ValueError("denovo min correlation must be within [-1, 1]")
    if max_gap_bp < 0 or min_shared_samples < 2 or batch_rows < 2:
        raise ValueError("denovo requires max_gap>=0, min_shared_samples>=2, and batch_rows>=2")
    if key not in available_views(input_path):
        raise ValueError(f"De novo source view is not present: {key.name()}")

    chroms, chrom_offsets, pos0 = load_cohort_index(input_path)
    ends = _chrom_ends(chrom_offsets, int(pos0.shape[0]))
    reader, _, _ = load_view_reader(input_path, key)
    breaks: list[np.ndarray] = []
    try:
        for chrom, global_lo, global_hi in zip(chroms, chrom_offsets, ends, strict=False):
            n_rows = int(global_hi - global_lo)
            before = np.ones(n_rows, dtype=bool)
            local_pos = np.asarray(pos0[int(global_lo) : int(global_hi)], dtype=np.int64)
            for lo in range(1, n_rows, int(batch_rows)):
                hi = min(n_rows, lo + int(batch_rows))
                rows = slice(int(global_lo) + lo - 1, int(global_lo) + hi)
                block = reader.read_rows(rows)
                corr, shared = _adjacent_correlations(block, min_shared_samples)
                close = (local_pos[lo:hi] - local_pos[lo - 1 : hi - 1]) <= int(max_gap_bp)
                before[lo:hi] = ~(close & (shared >= int(min_shared_samples)) & (corr >= float(min_correlation)))
            breaks.append(before)
            if logger:
                logger.info("De novo grouping %s: %s CpGs -> %s groups", chrom, f"{n_rows:,}", f"{int(before.sum()):,}")
    finally:
        reader.close()

    return _build_group_index_from_breaks(
        method="denovo", chroms=chroms, chrom_offsets=chrom_offsets, pos0=pos0, breaks_by_chrom=breaks
    )


def _group_chrom_ends(groups: GroupIndex) -> np.ndarray:
    return _chrom_ends(groups.chrom_offsets, groups.n_groups)


def save_group_index(path: str, groups: GroupIndex) -> None:
    outdir = Path(path)
    np.savez(
        outdir / GROUP_INDEX_FILE,
        method=np.asarray(groups.method),
        chroms=np.asarray(groups.chroms, dtype=object),
        chrom_offsets=groups.chrom_offsets,
        start=groups.start,
        end=groups.end,
        source_row_start=groups.source_row_start,
        source_row_end=groups.source_row_end,
        reference_start=groups.reference_start,
        reference_end=groups.reference_end,
    )
    chrom_ends = _group_chrom_ends(groups)
    with gzip.open(outdir / GROUP_TABLE_FILE, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "group_id",
                "chrom",
                "start",
                "end",
                "n_cpgs",
                "source_row_start",
                "source_row_end",
                "reference_start",
                "reference_end",
                "grouping",
            ]
        )
        for chrom, group_lo, group_hi in zip(groups.chroms, groups.chrom_offsets, chrom_ends, strict=False):
            for idx in range(int(group_lo), int(group_hi)):
                writer.writerow(
                    [
                        f"{groups.method}_{idx + 1}",
                        chrom,
                        int(groups.start[idx]),
                        int(groups.end[idx]),
                        int(groups.source_row_end[idx] - groups.source_row_start[idx]),
                        int(groups.source_row_start[idx]),
                        int(groups.source_row_end[idx]),
                        int(groups.reference_start[idx]),
                        int(groups.reference_end[idx]),
                        groups.method,
                    ]
                )


def load_group_index(path: str) -> GroupIndex:
    payload = np.load(Path(path) / GROUP_INDEX_FILE, allow_pickle=True)
    return GroupIndex(
        method=str(payload["method"].item()),
        chroms=[str(x) for x in payload["chroms"]],
        chrom_offsets=np.asarray(payload["chrom_offsets"], dtype=np.int64),
        start=np.asarray(payload["start"], dtype=np.uint32),
        end=np.asarray(payload["end"], dtype=np.uint32),
        source_row_start=np.asarray(payload["source_row_start"], dtype=np.int64),
        source_row_end=np.asarray(payload["source_row_end"], dtype=np.int64),
        reference_start=np.asarray(payload["reference_start"], dtype=np.uint32),
        reference_end=np.asarray(payload["reference_end"], dtype=np.uint32),
    )


def _encode_group_means(means: np.ndarray, counts: np.ndarray, min_observed_cpgs: int) -> np.ndarray:
    out = np.zeros(means.shape, dtype=np.uint16)
    keep = (counts >= int(min_observed_cpgs)) & np.isfinite(means)
    if np.any(keep):
        encoded = np.rint(np.clip(means[keep], 0.0, 1.0) * VALUE_DENOMINATOR).astype(np.uint32) + 1
        out[keep] = encoded.astype(np.uint16)
    return out


def _aggregate_group_batch(reader, row_starts: np.ndarray, row_ends: np.ndarray, min_observed_cpgs: int) -> np.ndarray:
    source_lo = int(row_starts[0])
    source_hi = int(row_ends[-1])
    block = np.asarray(reader.read_rows(slice(source_lo, source_hi)), dtype=np.float32)
    rel_start = row_starts.astype(np.int64) - source_lo
    rel_end = row_ends.astype(np.int64) - source_lo
    valid = np.isfinite(block)
    safe = np.where(valid, block, 0.0)

    contiguous = bool(rel_start[0] == 0 and rel_end[-1] == block.shape[0])
    if rel_start.shape[0] > 1:
        contiguous = contiguous and bool(np.all(rel_end[:-1] == rel_start[1:]))
    if contiguous:
        sums = np.add.reduceat(safe, rel_start, axis=0)
        counts = np.add.reduceat(valid.astype(np.int32), rel_start, axis=0)
    else:
        sum_prefix = np.vstack(
            [np.zeros((1, block.shape[1]), dtype=np.float64), np.cumsum(safe, axis=0, dtype=np.float64)]
        )
        count_prefix = np.vstack(
            [np.zeros((1, block.shape[1]), dtype=np.int32), np.cumsum(valid, axis=0, dtype=np.int32)]
        )
        sums = sum_prefix[rel_end] - sum_prefix[rel_start]
        counts = count_prefix[rel_end] - count_prefix[rel_start]
    means = np.divide(sums, counts, out=np.full(sums.shape, np.nan, dtype=np.float64), where=counts > 0)
    return _encode_group_means(means, counts, min_observed_cpgs)


def _copy_and_group_view(
    input_path: str,
    output_path: str,
    key: TrackKey,
    groups: GroupIndex,
    *,
    block_size: int,
    batch_groups: int,
    min_observed_cpgs: int,
    logger: logging.Logger,
) -> None:
    reader, _, _ = load_view_reader(input_path, key)
    columns = load_view_columns(input_path, key)
    writer = create_view_store(
        output_path,
        key,
        groups.chroms,
        groups.chrom_offsets,
        groups.n_groups,
        columns["sample_id"],
        columns["bundle_path"],
        columns["platform"],
        columns["source_path"],
        columns["input_tag"],
        block_size,
    )
    chrom_ends = _group_chrom_ends(groups)
    try:
        for chrom, group_lo, group_hi in zip(groups.chroms, groups.chrom_offsets, chrom_ends, strict=False):
            local_out = 0
            for batch_lo in range(int(group_lo), int(group_hi), int(batch_groups)):
                batch_hi = min(int(group_hi), batch_lo + int(batch_groups))
                encoded = _aggregate_group_batch(
                    reader,
                    groups.source_row_start[batch_lo:batch_hi],
                    groups.source_row_end[batch_lo:batch_hi],
                    min_observed_cpgs,
                )
                writer.write_dense_chrom_block(
                    chrom,
                    local_out,
                    local_out + encoded.shape[0],
                    0,
                    encoded.shape[1],
                    encoded,
                )
                local_out += encoded.shape[0]
            logger.info("Grouped view %s: %s complete", key.name(), chrom)
        writer.flush()
    finally:
        reader.close()
        writer.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_cohort(
    input_path: str,
    output_path: str,
    grouping: str,
    *,
    predefined_index_dir: str | None = None,
    denovo_key: TrackKey = TrackKey("5mC", "combined", "combined"),
    denovo_min_correlation: float = 0.8,
    denovo_max_gap_bp: int = 200,
    denovo_min_shared_samples: int = 3,
    min_observed_cpgs: int = 1,
    batch_rows: int = 10_000,
    batch_groups: int = 8_192,
    output_backend: str = "same",
    logger: logging.Logger | None = None,
) -> dict:
    logger = logger or logging.getLogger(__name__)
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    if input_path == output_path:
        raise ValueError("Grouped output must differ from the input cohort")
    if min_observed_cpgs < 1 or batch_groups < 1:
        raise ValueError("min_observed_cpgs and batch_groups must be >= 1")
    input_manifest = load_cohort_manifest(input_path)
    chroms, _, source_pos0 = load_cohort_index(input_path)
    method = str(grouping).strip().lower()
    predefined_details = None
    if method in PREDEFINED_RESOURCE_FILES:
        predefined, predefined_path, predefined_sha256 = resolve_predefined_index(method, predefined_index_dir)
        groups = map_predefined_index(input_path, predefined)
        parameters = dict(predefined.parameters)
        citation = predefined.citation
        predefined_details = {
            "method": predefined.method,
            "version": predefined.version,
            "assembly": predefined.assembly,
            "stored_index": predefined_path,
            "stored_index_sha256": predefined_sha256,
            "source": predefined.source,
        }
    elif method == "denovo":
        groups = build_denovo_groups(
            input_path,
            denovo_key,
            min_correlation=denovo_min_correlation,
            max_gap_bp=denovo_max_gap_bp,
            min_shared_samples=denovo_min_shared_samples,
            batch_rows=batch_rows,
            logger=logger,
        )
        parameters = {
            "algorithm": "adjacent_pair_pearson_connected_runs",
            "source_view": denovo_key.name(),
            "min_adjacent_profile_correlation": float(denovo_min_correlation),
            "max_adjacent_cpg_distance_bp": int(denovo_max_gap_bp),
            "min_shared_samples": int(denovo_min_shared_samples),
            "failed_pair_policy": "start_new_group",
        }
        citation = None
    else:
        raise ValueError("grouping must be one of: loyfer, decode, denovo")
    groups.validate(int(source_pos0.shape[0]))

    backend = cohort_backend(input_path) if output_backend == "same" else output_backend
    block_size = int(input_manifest.get("block_size", 64))
    create_kwargs = {
        "backend": backend,
        "block_size": block_size,
        "zarr_row_chunk": int(input_manifest.get("zarr_row_chunk", 65_536)),
        "zarr_codec": str(input_manifest.get("zarr_codec", "zstd")),
        "zarr_clevel": int(input_manifest.get("zarr_clevel", 5)),
        "zarr_shuffle": str(input_manifest.get("zarr_shuffle", "bitshuffle")),
    }
    create_cohort_store(
        output_path,
        groups.chroms,
        groups.chrom_offsets,
        groups.start,
        index_path=None,
        **create_kwargs,
    )
    save_group_index(output_path, groups)

    for key in available_views(input_path):
        _copy_and_group_view(
            input_path,
            output_path,
            key,
            groups,
            block_size=block_size,
            batch_groups=batch_groups,
            min_observed_cpgs=min_observed_cpgs,
            logger=logger,
        )

    reduction = float(source_pos0.shape[0] / groups.n_groups)
    summary = {
        "representation": "cpg_groups",
        "grouping": method,
        "source_cohort": input_path,
        "source_cpg_rows": int(source_pos0.shape[0]),
        "group_rows": groups.n_groups,
        "reduction_factor": reduction,
        "covered_source_cpg_rows": groups.n_source_cpgs,
        "aggregation": "arithmetic_mean_of_observed_cpg_beta_values",
        "min_observed_cpgs": int(min_observed_cpgs),
        "parameters": parameters,
        "predefined_index": predefined_details,
        "citation": citation,
        "coordinate_system": "0-based half-open; Loyfer reference_start/reference_end retain atlas G-anchored coordinates",
        "group_index": GROUP_INDEX_FILE,
        "group_table": GROUP_TABLE_FILE,
        "source_chromosomes": chroms,
    }
    summary_path = Path(output_path) / GROUP_SUMMARY_FILE
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest_path = Path(output_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "representation": "cpg_groups",
            "grouping": method,
            "source_cohort": input_path,
            "group_index": GROUP_INDEX_FILE,
            "group_table": GROUP_TABLE_FILE,
            "grouping_summary": GROUP_SUMMARY_FILE,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def grouping_main(args) -> dict:
    logging.basicConfig(
        level=logging.DEBUG if bool(getattr(args, "verbose", False)) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    summary = group_cohort(
        args.input,
        args.output,
        args.grouping,
        predefined_index_dir=getattr(args, "predefined_index_dir", None),
        denovo_key=TrackKey(
            getattr(args, "denovo_assay", "5mC"),
            getattr(args, "denovo_haplotype", "combined"),
            getattr(args, "denovo_strand", "combined"),
        ),
        denovo_min_correlation=getattr(args, "denovo_min_correlation", 0.8),
        denovo_max_gap_bp=getattr(args, "denovo_max_gap_bp", 200),
        denovo_min_shared_samples=getattr(args, "denovo_min_shared_samples", 3),
        min_observed_cpgs=getattr(args, "min_observed_cpgs", 1),
        batch_rows=getattr(args, "batch_rows", 10_000),
        batch_groups=getattr(args, "batch_groups", 8_192),
        output_backend=getattr(args, "cohort_backend", "same"),
        logger=logger,
    )
    logger.info(
        "Wrote %s grouped rows from %s CpGs (%.2fx reduction): %s",
        f"{summary['group_rows']:,}",
        f"{summary['source_cpg_rows']:,}",
        summary["reduction_factor"],
        args.output,
    )
    return summary
