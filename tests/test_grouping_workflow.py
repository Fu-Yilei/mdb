from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mdb.grouping import (
    PREDEFINED_RESOURCE_FILES,
    PredefinedGroupingIndex,
    group_cohort,
    load_group_index,
    save_predefined_index,
)
from mdb.merge import merge_main
from mdb.parse_args import parse_args
from mdb.schema import SampleBundle, SampleTrack, TrackKey
from mdb.storage import load_view_reader, write_sample_bundle


def _write_index(path: Path) -> None:
    np.savez(
        path,
        chroms=np.asarray(["chr1"], dtype=object),
        chrom_offsets=np.asarray([0], dtype=np.int64),
        pos0=np.asarray([1, 3, 20, 23, 25], dtype=np.uint32),
    )


def _bundle(path: Path, index_path: Path, sample_id: str, fractions: list[float]) -> None:
    key = TrackKey("5mC", "combined", "combined")
    values = np.rint(np.asarray(fractions) * 10_000).astype(np.uint16)
    write_sample_bundle(
        str(path),
        SampleBundle(
            sample_id=sample_id,
            platform="pacbio",
            index_path=str(index_path),
            tracks={
                key: SampleTrack(
                    key=key,
                    row_ids=np.arange(5, dtype=np.uint32),
                    values=values,
                    coverage=np.full(5, 10, dtype=np.uint16),
                    platform="pacbio",
                    input_tag="combined",
                    source_path=f"/synthetic/{sample_id}.bed.gz",
                    min_coverage=5,
                )
            },
        ),
    )


def _cohort(tmp_path: Path) -> Path:
    index_path = tmp_path / "index.npz"
    _write_index(index_path)
    profiles = {
        "s1": [0.10, 0.20, 0.90, 0.80, 0.70],
        "s2": [0.20, 0.30, 0.80, 0.70, 0.60],
        "s3": [0.30, 0.40, 0.70, 0.60, 0.50],
    }
    bundles = []
    for sample_id, values in profiles.items():
        path = tmp_path / f"{sample_id}.smdb"
        _bundle(path, index_path, sample_id, values)
        bundles.append(path)
    cohort = tmp_path / "cohort.mmdb"
    merge_main(
        type(
            "Args",
            (),
            {
                "inputs": [str(x) for x in bundles],
                "output": str(cohort),
                "modifiedc": False,
                "workers": 1,
                "block_size": 2,
                "cohort_backend": "npy",
                "zarr_row_chunk": 2,
                "zarr_codec": "zstd",
                "zarr_clevel": 5,
                "zarr_shuffle": "bitshuffle",
                "zarr_codec_threads": 1,
            },
        )()
    )
    return cohort


def _matrix(path: Path) -> np.ndarray:
    key = TrackKey("5mC", "combined", "combined")
    reader, _, _ = load_view_reader(str(path), key)
    try:
        return reader.read_rows(slice(0, reader.shape[0]))
    finally:
        reader.close()


def _predefined_dir(tmp_path: Path) -> Path:
    resource_dir = tmp_path / "resources"
    shared = {
        "version": "test_v1",
        "assembly": "synthetic",
        "chroms": ["chr1"],
        "chrom_offsets": np.asarray([0], dtype=np.int64),
        "source": "synthetic test index",
    }
    save_predefined_index(
        resource_dir / PREDEFINED_RESOURCE_FILES["decode"],
        PredefinedGroupingIndex(
            method="decode",
            start=np.asarray([1, 20, 25], dtype=np.uint32),
            end=np.asarray([5, 25, 27], dtype=np.uint32),
            left_shift_bp=0,
            parameters={"max_adjacent_cpg_distance_bp": 10},
            citation="https://example.test/decode",
            **shared,
        ),
    )
    save_predefined_index(
        resource_dir / PREDEFINED_RESOURCE_FILES["loyfer"],
        PredefinedGroupingIndex(
            method="loyfer",
            start=np.asarray([2, 21, 26], dtype=np.uint32),
            end=np.asarray([5, 25, 27], dtype=np.uint32),
            left_shift_bp=1,
            parameters={"coordinate_anchor": "G"},
            citation="https://example.test/loyfer",
            **shared,
        ),
    )
    return resource_dir


def test_decode_grouping_builds_reduced_compatible_cohort(tmp_path: Path) -> None:
    cohort = _cohort(tmp_path)
    output = tmp_path / "decode.mmdb"
    summary = group_cohort(
        str(cohort),
        str(output),
        "decode",
        predefined_index_dir=str(_predefined_dir(tmp_path)),
        write_group_table=True,
        output_backend="npy",
    )

    assert summary["source_cpg_rows"] == 5
    assert summary["group_rows"] == 2
    assert summary["definition_group_rows_before_min_cpg_filter"] == 3
    assert summary["excluded_small_groups"] == 1
    assert summary["reduction_factor"] == 2.5
    assert summary["parameters"] == {"max_adjacent_cpg_distance_bp": 10}
    assert summary["predefined_index"]["version"] == "test_v1"
    assert np.allclose(
        _matrix(output),
        np.asarray([[0.15, 0.25, 0.35], [0.85, 0.75, 0.65]], dtype=np.float32),
    )
    groups = load_group_index(str(output))
    assert groups.start.tolist() == [1, 20]
    assert groups.end.tolist() == [5, 25]
    manifest = json.loads((output / "manifest.json").read_text())
    source_manifest = json.loads((cohort / "manifest.json").read_text())
    assert manifest["representation"] == "cpg_groups"
    assert manifest["kind"] == source_manifest["kind"]
    assert manifest["group_coordinate_index"] == "groups.npz"
    assert (output / "groups.tsv.gz").is_file()


def test_loyfer_grouping_uses_g_anchored_sniffcell_index(tmp_path: Path) -> None:
    cohort = _cohort(tmp_path)
    output = tmp_path / "loyfer.mmdb"
    summary = group_cohort(
        str(cohort),
        str(output),
        "loyfer",
        predefined_index_dir=str(_predefined_dir(tmp_path)),
        output_backend="npy",
    )
    assert summary["group_rows"] == 2
    assert np.allclose(
        _matrix(output),
        np.asarray([[0.15, 0.25, 0.35], [0.85, 0.75, 0.65]], dtype=np.float32),
    )
    groups = load_group_index(str(output))
    assert groups.start.tolist() == [1, 20]
    assert groups.reference_start.tolist() == [2, 21]


def test_denovo_grouping_uses_adjacent_profile_correlation(tmp_path: Path) -> None:
    cohort = _cohort(tmp_path)
    output = tmp_path / "denovo.mmdb"
    summary = group_cohort(
        str(cohort),
        str(output),
        "denovo",
        denovo_min_correlation=0.95,
        denovo_max_gap_bp=100,
        denovo_min_shared_samples=3,
        batch_rows=2,
        threads=2,
        output_backend="npy",
    )
    assert summary["group_rows"] == 2
    groups = load_group_index(str(output))
    assert (groups.source_row_end - groups.source_row_start).tolist() == [2, 3]
    assert np.allclose(
        _matrix(output),
        np.asarray([[0.15, 0.25, 0.35], [0.80, 0.70, 0.60]], dtype=np.float32),
    )


def test_grouping_name_is_case_insensitive() -> None:
    args = parse_args(["group", "-i", "in.mmdb", "-o", "out.mmdb", "--grouping", "DECODE"])
    assert args.grouping == "decode"
