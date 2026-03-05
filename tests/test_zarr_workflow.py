from __future__ import annotations

import gzip
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mdb.create import create_main
from mdb.index import index_main
from mdb.merge import append_main, merge_main
from mdb.schema import TrackKey
from mdb.storage import (
    available_views,
    load_view_columns,
    load_view_uint16_matrix,
    query_cohort_point,
    query_cohort_range,
)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(text)


def ont_line(chrom: str, start: int, assay: str, cov: int, pct: float, strand: str = ".") -> str:
    end = start + 1
    return f"{chrom}\t{start}\t{end}\t{assay}\t{cov}\t{strand}\t{start}\t{end}\t255,0,0\t{cov}\t{pct:.2f}\n"


def pacbio_text(rows: list[tuple[str, int, int, int]]) -> str:
    header = "##pb-cpg-tools-version=3.0.0\n#chrom\tbegin\tcov\tdiscretized_mod_score\n"
    body = "".join(f"{chrom}\t{start}\t{cov}\t{score}\n" for chrom, start, cov, score in rows)
    return header + body


def _workspace(tmp_path: Path) -> dict[str, Path]:
    ref = tmp_path / "ref.fa"
    write_text(ref, ">chr1\nACGCGTACG\n>chr2\nTACGCGCGT\n")

    index_npz = tmp_path / "index.npz"
    index_main(SimpleNamespace(reference=str(ref), output=str(index_npz), sex=False))

    ont_dir = tmp_path / "ont_sample"
    write_gzip_text(
        ont_dir / "combined.bed.gz",
        "".join(
            [
                ont_line("chr1", 1, "m", 10, 80.0),
                ont_line("chr1", 1, "h", 10, 10.0),
                ont_line("chr1", 3, "m", 10, 20.0),
                ont_line("chr1", 3, "h", 10, 5.0),
                ont_line("chr2", 2, "m", 10, 75.0),
                ont_line("chr2", 2, "h", 10, 7.0),
            ]
        ),
    )

    pacbio_prefix = tmp_path / "pb" / "sample_pb"
    write_gzip_text(
        Path(str(pacbio_prefix) + ".combined.bed.gz"),
        pacbio_text([("chr1", 1, 11, 76), ("chr1", 3, 11, 18), ("chr2", 2, 11, 66)]),
    )
    write_gzip_text(
        Path(str(pacbio_prefix) + ".hap1.bed.gz"),
        pacbio_text([("chr1", 1, 11, 88), ("chr1", 7, 11, 44), ("chr2", 4, 11, 33)]),
    )
    write_gzip_text(
        Path(str(pacbio_prefix) + ".hap2.bed.gz"),
        pacbio_text([("chr1", 1, 11, 22), ("chr1", 7, 11, 67), ("chr2", 6, 11, 45)]),
    )

    return {
        "tmp": tmp_path,
        "index": index_npz,
        "ont_dir": ont_dir,
        "pacbio_prefix": pacbio_prefix,
    }


def create_bundle(index_path: Path, platform: str, bed: Path, output: Path, sample_id: str):
    create_main(
        SimpleNamespace(
            platform=platform,
            npz=str(index_path),
            bed=str(bed),
            output=str(output),
            min_coverage=5,
            sample_id=sample_id,
            reader="auto",
            workers=1,
        )
    )


def merge_bundles(inputs: list[Path], output: Path, backend: str):
    merge_main(
        SimpleNamespace(
            inputs=[str(p) for p in inputs],
            output=str(output),
            modifiedc=False,
            workers=1,
            block_size=2,
            cohort_backend=backend,
            zarr_row_chunk=2,
            zarr_codec="zstd",
            zarr_clevel=5,
            zarr_shuffle="bitshuffle",
            zarr_codec_threads=1,
        )
    )


def append_bundles(cohort: Path, inputs: list[Path], backend: str):
    append_main(
        SimpleNamespace(
            cohort=str(cohort),
            inputs=[str(p) for p in inputs],
            modifiedc=False,
            cohort_backend=backend,
            zarr_row_chunk=2,
            zarr_codec="zstd",
            zarr_clevel=5,
            zarr_shuffle="bitshuffle",
            zarr_codec_threads=1,
        )
    )


def test_merge_zarr_matches_npy(tmp_path: Path):
    ws = _workspace(tmp_path)
    ont_bundle = ws["tmp"] / "sample_ont.smdb"
    pb_bundle = ws["tmp"] / "sample_pb.smdb"
    create_bundle(ws["index"], "ont", ws["ont_dir"], ont_bundle, "ont_a")
    create_bundle(ws["index"], "pacbio", ws["pacbio_prefix"], pb_bundle, "pb_a")

    cohort_npy = ws["tmp"] / "cohort_npy.mmdb"
    cohort_zarr = ws["tmp"] / "cohort_zarr.mmdb"
    merge_bundles([ont_bundle, pb_bundle], cohort_npy, backend="npy")
    merge_bundles([ont_bundle, pb_bundle], cohort_zarr, backend="zarr")

    npy_views = sorted(v.name() for v in available_views(str(cohort_npy)))
    zarr_views = sorted(v.name() for v in available_views(str(cohort_zarr)))
    assert npy_views == zarr_views

    key = TrackKey("5mC", "combined", "combined")
    mat_npy = load_view_uint16_matrix(str(cohort_npy), key)
    mat_zarr = load_view_uint16_matrix(str(cohort_zarr), key)
    assert np.array_equal(mat_npy, mat_zarr)

    point_npy = query_cohort_point(str(cohort_npy), key, "pb_a", "chr1", 1)
    point_zarr = query_cohort_point(str(cohort_zarr), key, "pb_a", "chr1", 1)
    assert point_npy is not None
    assert point_zarr is not None
    assert point_npy["value_percent"] == point_zarr["value_percent"]

    range_npy = query_cohort_range(str(cohort_npy), key, "pb_a", "chr1", 1, 7)
    range_zarr = query_cohort_range(str(cohort_zarr), key, "pb_a", "chr1", 1, 7)
    assert range_npy["count"] == range_zarr["count"]
    assert [r["value_percent"] for r in range_npy["records"]] == [r["value_percent"] for r in range_zarr["records"]]


def test_append_zarr_adds_columns(tmp_path: Path):
    ws = _workspace(tmp_path)
    ont_bundle = ws["tmp"] / "sample_ont.smdb"
    pb_bundle = ws["tmp"] / "sample_pb.smdb"
    create_bundle(ws["index"], "ont", ws["ont_dir"], ont_bundle, "ont_a")
    create_bundle(ws["index"], "pacbio", ws["pacbio_prefix"], pb_bundle, "pb_a")

    cohort_zarr = ws["tmp"] / "cohort_append_zarr.mmdb"
    merge_bundles([ont_bundle], cohort_zarr, backend="zarr")
    append_bundles(cohort_zarr, [pb_bundle], backend="zarr")

    key = TrackKey("5mC", "combined", "combined")
    cols = load_view_columns(str(cohort_zarr), key)
    assert cols["sample_id"] == ["ont_a", "pb_a"]

    point = query_cohort_point(str(cohort_zarr), key, "pb_a", "chr1", 1)
    assert point is not None
    assert point["value_percent"] == 76.0


def test_zarr_missing_chromosome_returns_missing_values(tmp_path: Path):
    ws = _workspace(tmp_path)
    chr1_only = ws["tmp"] / "chr1_only"
    write_gzip_text(
        chr1_only / "combined.bed.gz",
        "".join(
            [
                ont_line("chr1", 1, "m", 10, 80.0),
                ont_line("chr1", 1, "h", 10, 10.0),
                ont_line("chr1", 3, "m", 10, 20.0),
            ]
        ),
    )
    bundle = ws["tmp"] / "chr1_only.smdb"
    create_bundle(ws["index"], "ont", chr1_only, bundle, "ont_chr1_only")

    cohort = ws["tmp"] / "cohort_chr1_only_zarr.mmdb"
    merge_bundles([bundle], cohort, backend="zarr")

    key = TrackKey("5mC", "combined", "combined")
    point = query_cohort_point(str(cohort), key, "ont_chr1_only", "chr2", 2)
    assert point is not None
    assert np.isnan(point["value_percent"])

    region = query_cohort_range(str(cohort), key, "ont_chr1_only", "chr2", 2, 6)
    assert region["count"] == 3
    assert all(np.isnan(record["value_percent"]) for record in region["records"])
