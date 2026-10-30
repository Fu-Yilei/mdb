from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mdb.export as export_module
from mdb.export import export_tensorqtl
from test_tensorqtl_export import build_cohort, build_grouped_cohort


def assert_row_accounting(receipt: dict, *, total: int, considered: int) -> None:
    assert receipt["rows_input"] == total
    assert receipt["rows_considered"] == considered
    assert receipt["rows_outside_region"] == total - considered
    assert receipt["rows_written"] + sum(receipt["rows_filtered"].values()) == considered


@pytest.mark.parametrize("backend", ["npy", "zarr"])
def test_individual_region_boundaries_and_chromosome_selection(tmp_path: Path, backend: str) -> None:
    values = np.asarray([
        [0.0, 0.2, 0.8], [0.1, 0.3, 0.9], [0.2, 0.5, 0.8],
        [0.3, 0.6, 0.9], [0.4, 0.7, 1.0], [0.5, 0.7, 0.9], [0.6, 0.8, 1.0],
    ])
    source = build_cohort(
        tmp_path / "source.mmdb", values, backend=backend,
        positions=[0, 99, 100, 199, 200, 100, 199],
        chroms=["chr1", "chr2"], chrom_offsets=[0, 5],
    )
    output = tmp_path / "region.bed.gz"
    receipt = export_tensorqtl(str(source), str(output), region="chr1:100-200", batch_rows=1)
    frame = pd.read_csv(output, sep="\t")
    assert frame.iloc[:, :4].values.tolist() == [
        ["chr1", 100, 101, "chr1:101"], ["chr1", 199, 200, "chr1:200"],
    ]
    np.testing.assert_allclose(frame.iloc[:, 4:].to_numpy(), values[[2, 3]], atol=1e-7)
    assert receipt["region"] == "chr1:100-200"
    assert_row_accounting(receipt, total=7, considered=2)
    assert json.loads(Path(str(output) + ".export.json").read_text()) == receipt


@pytest.mark.parametrize("backend", ["npy", "zarr"])
def test_region_filter_counts_exclude_unselected_rows(tmp_path: Path, backend: str) -> None:
    source = build_cohort(
        tmp_path / "source.mmdb",
        np.asarray([
            [np.nan, np.nan, np.nan, np.nan],  # Outside: do not count as filtered.
            [0.0, 0.2, 0.4, 0.6],
            [0.4, 0.4, 0.4, 0.4],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 0.2],
            [0.0, np.nan, 0.4, 0.6],
            [0.4, 0.4, 0.4, 0.4],  # Outside: do not count as constant.
        ]),
        backend=backend, positions=[90, 100, 110, 120, 130, 140, 200],
    )
    output = tmp_path / "filtered_region.bed"
    receipt = export_tensorqtl(
        str(source), str(output), region="chr1:100-200", min_frac_present=0.5, batch_rows=2,
    )
    frame = pd.read_csv(output, sep="\t")
    assert frame["phenotype_id"].tolist() == ["chr1:101"]
    assert receipt["rows_filtered"] == {
        "all_missing": 1, "below_min_frac_present": 1, "remaining_missing": 1, "constant": 1,
    }
    assert_row_accounting(receipt, total=7, considered=5)


@pytest.mark.parametrize("backend", ["npy", "zarr"])
def test_default_region_receipt_accounts_for_full_database(tmp_path: Path, backend: str) -> None:
    source = build_cohort(
        tmp_path / "source.mmdb", np.asarray([[0.1, 0.3, 0.5], [0.2, 0.2, 0.2]]), backend=backend,
    )
    receipt = export_tensorqtl(str(source), str(tmp_path / "all.bed"))
    assert receipt["region"] is None
    assert_row_accounting(receipt, total=2, considered=2)


@pytest.mark.parametrize("backend", ["npy", "zarr"])
@pytest.mark.parametrize(("region", "index"), [
    ("chr1:0-101", 0),
    ("chr1:104-105", 0),
    ("chr1:105-301", 1),
    ("chr1:305-306", 1),
])
def test_group_overlap_retains_full_bounds_global_ids_and_stored_means(
    tmp_path: Path, backend: str, region: str, index: int,
) -> None:
    source = build_cohort(
        tmp_path / "source.mmdb",
        np.asarray([[0.1, 0.3, 0.5], [0.3, 0.5, 0.7], [0.2, 0.4, 0.6], [0.4, 0.6, 0.8]]),
        backend=backend, positions=[100, 103, 300, 310],
    )
    grouped = build_grouped_cohort(source, tmp_path / "grouped.mmdb", backend=backend)
    output = tmp_path / "group_region.bed.gz"
    receipt = export_tensorqtl(str(grouped), str(output), region=region, batch_rows=1)
    frame = pd.read_csv(output, sep="\t")
    expected_coordinates = [["chr1", 100, 105, "decode_1"], ["chr1", 300, 312, "decode_2"]]
    expected_means = [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]]
    assert frame.iloc[:, :4].values.tolist() == [expected_coordinates[index]]
    np.testing.assert_allclose(frame.iloc[:, 4:].to_numpy(), [expected_means[index]], atol=1e-7)
    assert receipt["region"] == region
    assert_row_accounting(receipt, total=2, considered=1)


@pytest.mark.parametrize("backend", ["npy", "zarr"])
@pytest.mark.parametrize(("grouped", "region"), [
    (False, "chr1:104-300"), (True, "chr1:105-300"), (True, "chr1:312-313"),
])
def test_region_with_no_hits_does_not_publish_files(
    tmp_path: Path, backend: str, grouped: bool, region: str,
) -> None:
    source = build_cohort(
        tmp_path / "source.mmdb", np.tile([0.1, 0.4, 0.8], (4, 1)),
        backend=backend, positions=[100, 103, 300, 310],
    )
    if grouped:
        source = build_grouped_cohort(source, tmp_path / "grouped.mmdb", backend=backend)
    output = tmp_path / "empty_region.bed.gz"
    with pytest.raises(ValueError, match="region"):
        export_tensorqtl(str(source), str(output), region=region)
    assert not output.exists()
    assert not Path(str(output) + ".export.json").exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize("backend", ["npy", "zarr"])
@pytest.mark.parametrize("region", [
    "", "chr1", "chr1:100", "chr1:100-", "chr1:-1-100", "chr1:0--100",
    "chr1:100-100", "chr1:200-100", "chr3:0-1", "chr1:1.0-2", "chr1: 0-100", "chr1:0-1,000",
])
def test_invalid_region_fails_without_outputs(tmp_path: Path, backend: str, region: str) -> None:
    source = build_cohort(
        tmp_path / "source.mmdb", np.asarray([[0.1, 0.4, 0.8]]), backend=backend, positions=[100],
    )
    output = tmp_path / "invalid_region.bed"
    with pytest.raises(ValueError):
        export_tensorqtl(str(source), str(output), region=region)
    assert not output.exists()
    assert not Path(str(output) + ".export.json").exists()


@pytest.mark.parametrize("backend", ["npy", "zarr"])
def test_only_selected_matrix_rows_are_read(tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    source = build_cohort(
        tmp_path / "source.mmdb", np.tile([0.1, 0.4, 0.8], (7, 1)),
        backend=backend, positions=[0, 99, 100, 199, 200, 100, 199],
        chroms=["chr1", "chr2"], chrom_offsets=[0, 5],
    )
    original_loader = export_module.load_view_reader
    read_rows: list[int] = []

    def tracked_loader(*args, **kwargs):
        reader, columns, metadata = original_loader(*args, **kwargs)
        original_get_block = reader.get_block

        def tracked_get_block(rows):
            read_rows.extend(range(*rows.indices(reader.n_rows)))
            return original_get_block(rows)

        monkeypatch.setattr(reader, "get_block", tracked_get_block)
        return reader, columns, metadata

    monkeypatch.setattr(export_module, "load_view_reader", tracked_loader)
    export_tensorqtl(str(source), str(tmp_path / "selected.bed"), region="chr1:100-200", batch_rows=1)
    assert read_rows == [2, 3]


@pytest.mark.parametrize("backend", ["npy", "zarr"])
def test_nested_group_intervals_with_nonmonotonic_ends(tmp_path: Path, backend: str) -> None:
    source = build_cohort(
        tmp_path / "source.mmdb",
        np.asarray([[0.1, 0.3, 0.5], [0.3, 0.5, 0.7], [0.2, 0.4, 0.6], [0.4, 0.6, 0.8]]),
        backend=backend, positions=[100, 103, 300, 310],
    )
    grouped = build_grouped_cohort(source, tmp_path / "grouped.mmdb", backend=backend)
    # Make the first synthetic group enclose the second without changing row
    # alignment or methylation values. Sorted starts do not imply sorted ends.
    index_file = grouped / "groups.npz"
    with np.load(index_file, allow_pickle=True) as stored:
        index = {name: stored[name] for name in stored.files}
    index["end"] = np.asarray([400, 312], dtype=np.uint32)
    np.savez(index_file, **index)

    output = tmp_path / "nested_region.bed"
    receipt = export_tensorqtl(str(grouped), str(output), region="chr1:350-351")
    frame = pd.read_csv(output, sep="\t")
    assert frame.iloc[:, :4].values.tolist() == [["chr1", 100, 400, "decode_1"]]
    np.testing.assert_allclose(frame.iloc[:, 4:].to_numpy(), [[0.2, 0.4, 0.6]], atol=1e-7)
    assert_row_accounting(receipt, total=2, considered=1)
