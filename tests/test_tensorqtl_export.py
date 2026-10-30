from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mdb.export import export_tensorqtl
from mdb.grouping import (
    PREDEFINED_RESOURCE_FILES,
    PredefinedGroupingIndex,
    group_cohort,
    save_predefined_index,
)
from mdb.schema import TrackKey
from mdb.storage import create_cohort_store, create_view_store


def build_cohort(
    path: Path,
    matrix: np.ndarray,
    *,
    backend: str = "npy",
    positions: list[int] | None = None,
    chroms: list[str] | None = None,
    chrom_offsets: list[int] | None = None,
    sample_ids: list[str] | None = None,
    key: TrackKey = TrackKey("5mC", "combined", "combined"),
) -> Path:
    """Write the actual cohort layout, including its offset-encoded missing values."""
    matrix = np.asarray(matrix, dtype=np.float64)
    n_rows, n_samples = matrix.shape
    chroms = chroms or ["chr1"]
    chrom_offsets_array = np.asarray(chrom_offsets or [0], dtype=np.int64)
    positions_array = np.asarray(positions if positions is not None else np.arange(n_rows) * 10, dtype=np.uint32)
    sample_ids = sample_ids or [f"s{i + 1}" for i in range(n_samples)]
    create_cohort_store(
        str(path), chroms, chrom_offsets_array, positions_array,
        block_size=2, backend=backend, zarr_row_chunk=2,
    )
    writer = create_view_store(
        str(path), key, chroms, chrom_offsets_array, n_rows, sample_ids,
        ["/synthetic/sample.smdb"] * n_samples, ["pacbio"] * n_samples,
        ["/synthetic/input.bed.gz"] * n_samples, ["combined"] * n_samples, 2,
    )
    encoded = np.zeros(matrix.shape, dtype=np.uint16)
    present = np.isfinite(matrix)
    encoded[present] = np.rint(matrix[present] * 10_000).astype(np.uint16) + 1
    ends = np.r_[chrom_offsets_array[1:], n_rows]
    try:
        for chrom, lo, hi in zip(chroms, chrom_offsets_array, ends, strict=True):
            writer.write_dense_chrom_block(chrom, 0, int(hi - lo), 0, n_samples, encoded[lo:hi])
    finally:
        writer.close()
    return path


def build_grouped_cohort(source: Path, output: Path, *, backend: str = "npy") -> Path:
    """Run the public grouping function against small explicit synthetic definitions."""
    resource_dir = output.parent / f"{output.stem}_resources"
    save_predefined_index(
        resource_dir / PREDEFINED_RESOURCE_FILES["decode"],
        PredefinedGroupingIndex(
            method="decode", version="synthetic_v1", assembly="synthetic",
            chroms=["chr1"], chrom_offsets=np.asarray([0], dtype=np.int64),
            start=np.asarray([100, 300], dtype=np.uint32),
            end=np.asarray([110, 320], dtype=np.uint32),
            left_shift_bp=0, parameters={}, citation="https://example.test/synthetic",
            source="Synthetic test data only",
        ),
    )
    group_cohort(
        str(source), str(output), "decode", predefined_index_dir=str(resource_dir),
        output_backend=backend, write_group_table=True, batch_groups=1,
    )
    return output


@pytest.mark.parametrize("backend", ["npy", "zarr"])
@pytest.mark.parametrize("suffix", [".bed", ".bed.gz"])
def test_beta_values_and_zero_based_point_coordinates(tmp_path: Path, backend: str, suffix: str) -> None:
    values = np.asarray([[0, 0.125, 1], [0.5, 0.3333, 0.2], [0.2, 0.8, 0.6]])
    cohort = build_cohort(
        tmp_path / "source.mmdb", values, backend=backend,
        positions=[0, 42, 7], chroms=["chr1", "chr2"], chrom_offsets=[0, 2],
    )
    output = tmp_path / f"phenotypes{suffix}"
    receipt = export_tensorqtl(str(cohort), str(output), batch_rows=1)
    frame = pd.read_csv(output, sep="\t")
    assert frame.columns.tolist() == ["#chr", "start", "end", "phenotype_id", "s1", "s2", "s3"]
    assert frame.iloc[:, :3].values.tolist() == [["chr1", 0, 1], ["chr1", 42, 43], ["chr2", 7, 8]]
    assert frame["phenotype_id"].is_unique
    np.testing.assert_allclose(frame.iloc[:, 4:].to_numpy(), values, atol=1e-7)
    assert receipt["rows_input"] == receipt["rows_written"] == 3
    assert receipt["imputed_cells"] == 0
    assert receipt["backend"] == backend
    assert json.loads(Path(str(output) + ".export.json").read_text()) == receipt
    if suffix.endswith(".gz"):
        with gzip.open(output, "rt") as handle:
            assert handle.readline().startswith("#chr\t")


@pytest.mark.parametrize("backend", ["npy", "zarr"])
def test_group_intervals_and_means_are_retained(tmp_path: Path, backend: str) -> None:
    source_values = np.asarray([[0.1, 0.3, 0.5], [0.3, 0.5, 0.7], [0.2, 0.4, 0.6], [0.4, 0.6, 0.8]])
    source = build_cohort(tmp_path / "source.mmdb", source_values, backend=backend, positions=[100, 103, 300, 310])
    grouped = build_grouped_cohort(source, tmp_path / "grouped.mmdb", backend=backend)
    output = tmp_path / "groups.bed.gz"
    receipt = export_tensorqtl(str(grouped), str(output), batch_rows=1)
    frame = pd.read_csv(output, sep="\t")
    # Grouping stores the covered CpG span, including the final G base; the
    # original definition bounds remain separate reference_start/reference_end.
    assert frame.iloc[:, :4].values.tolist() == [["chr1", 100, 105, "decode_1"], ["chr1", 300, 312, "decode_2"]]
    np.testing.assert_allclose(frame.iloc[:, 4:].to_numpy(), [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]], atol=1e-7)
    assert receipt["representation"] == "cpg_groups"
    assert receipt["rows_written"] == 2


@pytest.mark.parametrize("backend", ["npy", "zarr"])
def test_filtering_distinguishes_zero_missing_and_constant(tmp_path: Path, backend: str) -> None:
    values = np.asarray([
        [0, 0.2, 0.4, 0.6],
        [0.4, 0.4, 0.4, 0.4],
        [0, np.nan, 0.4, 0.6],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, 0.2],
    ])
    source = build_cohort(tmp_path / "source.mmdb", values, backend=backend)
    output = tmp_path / "filtered.bed"
    receipt = export_tensorqtl(str(source), str(output), min_frac_present=0.5, batch_rows=2)
    frame = pd.read_csv(output, sep="\t")
    assert frame.shape[0] == 1
    assert frame["s1"].iloc[0] == 0
    assert receipt["rows_filtered"] == {
        "all_missing": 1, "below_min_frac_present": 1,
        "remaining_missing": 1, "constant": 1,
    }
    assert sum(receipt["rows_filtered"].values()) + receipt["rows_written"] == receipt["rows_input"]


def test_mean_imputation_applies_after_fraction_filter(tmp_path: Path) -> None:
    source = build_cohort(tmp_path / "source.mmdb", np.asarray([
        [0, np.nan, 0.4, 0.8],
        [0.2, np.nan, np.nan, 0.8],
        [np.nan, np.nan, np.nan, np.nan],
        [0.3, 0.3, np.nan, 0.3],
    ]))
    output = tmp_path / "imputed.bed"
    receipt = export_tensorqtl(str(source), str(output), missing="mean", min_frac_present=0.75)
    frame = pd.read_csv(output, sep="\t")
    np.testing.assert_allclose(frame.iloc[:, 4:].to_numpy(), [[0, 0.4, 0.4, 0.8]], atol=1e-7)
    assert receipt["rows_written"] == 1
    assert receipt["rows_filtered"]["below_min_frac_present"] == 1
    assert receipt["rows_filtered"]["all_missing"] == 1
    assert receipt["rows_filtered"]["constant"] == 1
    assert receipt["imputed_cells"] >= 1
    assert not frame.iloc[:, 4:].isna().any().any()


def test_ordered_samples_and_mapping_apply_before_filters(tmp_path: Path) -> None:
    source = build_cohort(tmp_path / "source.mmdb", np.asarray([[0.1, np.nan, 0.8], [0.2, 0.9, 0.2]]))
    output = tmp_path / "selected.bed"
    receipt = export_tensorqtl(
        str(source), str(output), samples=["s3", "s1"], sample_map={"s1": "gA", "s3": "gC"},
    )
    frame = pd.read_csv(output, sep="\t")
    assert frame.columns[4:].tolist() == ["gC", "gA"]
    np.testing.assert_allclose(frame.iloc[:, 4:].to_numpy(), [[0.8, 0.1]])
    assert receipt["sample_ids"] == ["s3", "s1"]
    assert receipt["output_sample_ids"] == ["gC", "gA"]
    assert receipt["rows_filtered"]["constant"] == 1


@pytest.mark.parametrize("kwargs", [
    {"samples": ["s1"]}, {"samples": ["s1", "s1"]},
    {"samples": ["s1", "absent"]}, {"sample_map": {"s1": "same", "s2": "same"}},
    {"sample_map": {"s1": "bad\tid"}}, {"sample_map": {"s1": ""}},
    {"min_frac_present": -0.1}, {"min_frac_present": 1.1},
    {"missing": "zero"}, {"batch_rows": 0}, {"assay": "absent"},
])
def test_invalid_selection_or_policy_fails_without_outputs(tmp_path: Path, kwargs: dict) -> None:
    source = build_cohort(tmp_path / "source.mmdb", np.asarray([[0.1, 0.5, 0.9]]))
    output = tmp_path / "invalid.bed.gz"
    with pytest.raises((ValueError, KeyError)):
        export_tensorqtl(str(source), str(output), **kwargs)
    assert not output.exists()
    assert not Path(str(output) + ".export.json").exists()


@pytest.mark.parametrize("existing_receipt", [False, True])
def test_existing_output_or_receipt_is_preserved_by_default(tmp_path: Path, existing_receipt: bool) -> None:
    source = build_cohort(tmp_path / "source.mmdb", np.asarray([[0.1, 0.5, 0.9]]))
    output = tmp_path / "existing.bed"
    collision = Path(str(output) + ".export.json") if existing_receipt else output
    collision.write_text("preserve me\n")
    with pytest.raises((ValueError, FileExistsError)):
        export_tensorqtl(str(source), str(output))
    assert collision.read_text() == "preserve me\n"


def test_explicit_overwrite_replaces_bed_and_receipt(tmp_path: Path) -> None:
    source = build_cohort(tmp_path / "source.mmdb", np.asarray([[0.1, 0.5, 0.9]]))
    output = tmp_path / "existing.bed"
    output.write_text("old bed\n")
    Path(str(output) + ".export.json").write_text("old receipt\n")
    receipt = export_tensorqtl(str(source), str(output), overwrite=True)
    assert pd.read_csv(output, sep="\t").shape == (1, 7)
    assert json.loads(Path(str(output) + ".export.json").read_text()) == receipt


def test_export_cannot_write_inside_input_store(tmp_path: Path) -> None:
    source = build_cohort(tmp_path / "source.mmdb", np.asarray([[0.1, 0.5, 0.9]]))
    manifest = (source / "manifest.json").read_bytes()
    with pytest.raises(ValueError):
        export_tensorqtl(str(source), str(source / "phenotypes.bed"), overwrite=True)
    assert (source / "manifest.json").read_bytes() == manifest
    assert not (source / "phenotypes.bed").exists()


def test_empty_export_fails_without_publishing_header_only_file(tmp_path: Path) -> None:
    source = build_cohort(tmp_path / "source.mmdb", np.asarray([[0.5, 0.5], [np.nan, np.nan]]))
    output = tmp_path / "empty.bed.gz"
    with pytest.raises(ValueError):
        export_tensorqtl(str(source), str(output))
    assert not output.exists()
    assert not Path(str(output) + ".export.json").exists()


def test_cli_sample_file_mapping_and_separate_assay(tmp_path: Path) -> None:
    from mdb.main import main

    values = np.asarray([[0.05, 0.1, 0.2], [0.2, 0.25, 0.3]])
    source = build_cohort(
        tmp_path / "source.mmdb", values, key=TrackKey("5hmC", "combined", "combined"),
    )
    sample_file = tmp_path / "ordered_samples.txt"
    sample_file.write_text("s3\ns1\n")
    sample_map = tmp_path / "sample_map.tsv"
    sample_map.write_text("sample_id\tgenotype_id\ns1\tgenotype_A\ns3\tgenotype_C\n")
    output = tmp_path / "5hmc.bed.gz"
    main([
        "export-tensorqtl", "-i", str(source), "-o", str(output),
        "--assay", "5hmC", "--sample-file", str(sample_file),
        "--sample-map", str(sample_map), "--batch-rows", "1",
    ])
    frame = pd.read_csv(output, sep="\t")
    assert frame.columns[4:].tolist() == ["genotype_C", "genotype_A"]
    np.testing.assert_allclose(frame.iloc[:, 4:].to_numpy(), values[:, [2, 0]])
    receipt = json.loads(Path(str(output) + ".export.json").read_text())
    assert receipt["view"]["assay"] == "5hmC"
