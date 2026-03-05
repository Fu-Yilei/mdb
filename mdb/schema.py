from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

FORMAT_VERSION = "1.0"
VALUE_SCALE = 100
VALUE_DENOMINATOR = 100.0 * VALUE_SCALE
VALUE_MISSING = np.uint16(65535)
COVERAGE_MISSING = np.uint16(65535)

HAPLOTYPE_COMBINED = "combined"
STRAND_COMBINED = "combined"
SAMPLE_STORE_KIND = "sample_store_npy"
COHORT_STORE_NPY_KIND = "cohort_store_npy"
COHORT_STORE_ZARR_KIND = "cohort_store_zarr"
COHORT_STORE_KIND = COHORT_STORE_NPY_KIND
COHORT_STORE_KINDS = {COHORT_STORE_NPY_KIND, COHORT_STORE_ZARR_KIND}


@dataclass(frozen=True, order=True)
class TrackKey:
    assay: str
    haplotype: str
    strand: str

    def name(self) -> str:
        return f"{self.assay}__{self.haplotype}__{self.strand}"

    @classmethod
    def from_name(cls, name: str) -> "TrackKey":
        assay, haplotype, strand = name.split("__", 2)
        return cls(assay=assay, haplotype=haplotype, strand=strand)


@dataclass
class SampleTrack:
    key: TrackKey
    row_ids: np.ndarray
    values: np.ndarray
    coverage: np.ndarray
    platform: str
    input_tag: str
    source_path: str
    min_coverage: int

    def validate(self) -> None:
        if self.row_ids.dtype != np.uint32:
            raise ValueError("row_ids must be uint32")
        if self.values.dtype != np.uint16:
            raise ValueError("values must be uint16")
        if self.coverage.dtype != np.uint16:
            raise ValueError("coverage must be uint16")
        n = self.row_ids.shape[0]
        if self.values.shape[0] != n or self.coverage.shape[0] != n:
            raise ValueError("row_ids, values, and coverage must have the same length")
        if self.row_ids.shape[0] > 1 and np.any(self.row_ids[1:] < self.row_ids[:-1]):
            raise ValueError("row_ids must be sorted in non-decreasing order")


@dataclass
class SampleBundle:
    sample_id: str
    platform: str
    index_path: str
    tracks: Dict[TrackKey, SampleTrack]


def encode_percent(values_percent: np.ndarray) -> np.ndarray:
    arr = np.asarray(values_percent, dtype=np.float32)
    if np.any(arr < 0) or np.any(arr > 100):
        raise ValueError("methylation percent values must be within [0, 100]")
    return np.rint(arr * VALUE_SCALE).astype(np.uint16)


def decode_percent(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    out = arr.astype(np.float32)
    out[arr == VALUE_MISSING] = np.nan
    out /= VALUE_SCALE
    return out


def decode_fraction(values: np.ndarray) -> np.ndarray:
    return decode_percent(values) / 100.0


def ensure_uint16(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.uint16:
        return arr
    return arr.astype(np.uint16, copy=False)


def infer_sample_id(output_path: str) -> str:
    name = Path(output_path).name
    for suffix in (".smdb", ".mmdb", ".mdb"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(output_path).stem
