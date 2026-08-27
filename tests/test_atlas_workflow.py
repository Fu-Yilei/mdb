from __future__ import annotations

import gzip
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mdb.atlas import atlas_main
from mdb.schema import TrackKey
from mdb.storage import (
    available_views,
    create_cohort_store,
    create_view_store,
    load_view_reader,
)


def _write_source_cohort(path: Path) -> None:
    chroms = ["chr1"]
    offsets = np.asarray([0], dtype=np.int64)
    create_cohort_store(
        str(path),
        chroms=chroms,
        chrom_offsets=offsets,
        pos0=np.asarray([20, 40], dtype=np.uint32),
        backend="zarr",
        block_size=2,
        zarr_row_chunk=2,
    )
    np.savez(
        path / "groups.npz",
        method=np.asarray("loyfer", dtype=object),
        chroms=np.asarray(chroms, dtype=object),
        chrom_offsets=offsets,
        reference_start=np.asarray([20, 40], dtype=np.uint32),
        reference_end=np.asarray([30, 50], dtype=np.uint32),
    )
    writer = create_view_store(
        str(path),
        key=TrackKey("5hmC", "combined", "combined"),
        chroms=chroms,
        chrom_offsets=offsets,
        n_rows=2,
        sample_ids=["ont_n", "ont_o"],
        bundle_paths=["n.smdb", "o.smdb"],
        platforms=["ont", "ont"],
        source_paths=["n.bed.gz", "o.bed.gz"],
        input_tags=["combined", "combined"],
        block_size=2,
    )
    try:
        writer.write_dense_chrom_block(
            "chr1",
            0,
            2,
            0,
            2,
            np.asarray([[8001, 1001], [7001, 2001]], dtype=np.uint16),
        )
    finally:
        writer.close()


def test_atlas_combines_legacy_modifiedc_and_aligned_ont_views(tmp_path: Path):
    legacy_npy = tmp_path / "legacy.npy"
    np.save(
        legacy_npy,
        np.asarray(
            [
                [0.1, 0.9],
                [0.2, 0.8],
                [np.nan, 0.7],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        ),
    )
    legacy_index = tmp_path / "legacy.index.gz"
    with gzip.open(legacy_index, "wt") as handle:
        handle.write("chr1\t10\t11\t1\t2\n")
        handle.write("chr1\t20\t30\t2\t5\n")
        handle.write("chr1\t31\t32\t5\t6\n")
        handle.write("chr1\t40\t50\t6\t9\n")
    legacy_samples = tmp_path / "legacy.samples.txt"
    legacy_samples.write_text("legacy_n\nlegacy_o\n")
    source = tmp_path / "source.mmdb"
    _write_source_cohort(source)

    output = tmp_path / "combined.mmdb"
    atlas_main(
        SimpleNamespace(
            legacy_npy=str(legacy_npy),
            legacy_index=str(legacy_index),
            legacy_samples=str(legacy_samples),
            legacy_assay="modifiedC",
            cohorts=[str(source)],
            output=str(output),
            batch_rows=2,
            block_size=2,
            cohort_backend="zarr",
            zarr_row_chunk=2,
            zarr_codec="zstd",
            zarr_clevel=5,
            zarr_shuffle="bitshuffle",
        )
    )

    assert [key.name() for key in available_views(str(output))] == [
        "5hmC__combined__combined",
        "modifiedC__combined__combined",
    ]

    legacy_reader, legacy_columns, _ = load_view_reader(
        str(output), TrackKey("modifiedC", "combined", "combined")
    )
    ont_reader, ont_columns, _ = load_view_reader(
        str(output), TrackKey("5hmC", "combined", "combined")
    )
    try:
        np.testing.assert_allclose(
            legacy_reader.get_block(slice(0, 4)),
            np.asarray([[0.1, 0.9], [0.2, 0.8], [np.nan, 0.7], [0.4, 0.6]], dtype=np.float32),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            ont_reader.get_block(slice(0, 4)),
            np.asarray([[np.nan, np.nan], [0.8, 0.1], [np.nan, np.nan], [0.7, 0.2]], dtype=np.float32),
            equal_nan=True,
        )
        assert legacy_columns["sample_id"] == ["legacy_n", "legacy_o"]
        assert ont_columns["sample_id"] == ["ont_n", "ont_o"]
    finally:
        legacy_reader.close()
        ont_reader.close()

    manifest = (output / "atlas_manifest.json").read_text()
    assert '"matched_rows": 2' in manifest
    assert (output / "groups.npz").is_file()
