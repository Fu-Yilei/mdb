from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mdb.merge import merge_main
from mdb.pca import pca_main
from mdb.schema import SampleBundle, SampleTrack, TrackKey
from mdb.storage import write_sample_bundle


def write_index(path: Path) -> None:
    np.savez(
        path,
        chroms=np.asarray(["chr1"], dtype=object),
        chrom_offsets=np.asarray([0], dtype=np.int64),
        pos0=np.asarray([1, 3, 5, 7], dtype=np.uint32),
    )


def make_bundle(path: Path, index_path: Path, sample_id: str, values: list[int]) -> None:
    key = TrackKey("5mC", "combined", "combined")
    track = SampleTrack(
        key=key,
        row_ids=np.asarray([0, 1, 2, 3], dtype=np.uint32),
        values=np.asarray(values, dtype=np.uint16),
        coverage=np.asarray([5, 6, 7, 8], dtype=np.uint16),
        platform="pacbio",
        input_tag="combined",
        source_path=f"/synthetic/{sample_id}/combined.bed.gz",
        min_coverage=5,
    )
    write_sample_bundle(
        str(path),
        SampleBundle(
            sample_id=sample_id,
            platform="pacbio",
            index_path=str(index_path),
            tracks={key: track},
        ),
    )


def run_pca(input_path: Path, outdir: Path) -> None:
    pca_main(
        SimpleNamespace(
            input=str(input_path),
            outdir=str(outdir),
            metadata=None,
            assay="5mC",
            haplotype="combined",
            strand="combined",
            pairplot_pcs_n=2,
            frac_cpgs=1.0,
            n_pcs=2,
            min_frac_present=0.0,
            batch_rows=2,
            seed=1,
            umap=False,
            umap_neighbors=15,
            umap_min_dist=0.1,
            umap_metric="euclidean",
            pairplot_mode="none",
            pairplot_hue=None,
            pairplot_diag_kind="hist",
            pairplot_corner=True,
            verbose=False,
        )
    )


def assert_outputs(outdir: Path, expected_rows: int, expected_mode: str) -> None:
    embedding = outdir / "embedding.tsv"
    params = outdir / "params.json"
    pca_html = outdir / "pca.html"
    pairplot = outdir / "pca_pairplot.png"

    assert embedding.exists()
    assert params.exists()
    assert pca_html.exists()
    assert pairplot.exists()

    df = pd.read_csv(embedding, sep="\t")
    assert len(df) == expected_rows
    assert "PC1" in df.columns
    assert "PC2" in df.columns

    meta = json.loads(params.read_text())
    assert meta["input_mode"] == expected_mode
    assert meta["n_samples"] == expected_rows


def test_pca_on_cohort_store(tmp_path: Path):
    index_path = tmp_path / "index.npz"
    write_index(index_path)

    b1 = tmp_path / "s1.smdb"
    b2 = tmp_path / "s2.smdb"
    make_bundle(b1, index_path, "s1", [1000, 2000, 3000, 4000])
    make_bundle(b2, index_path, "s2", [1500, 2500, 3500, 4500])

    cohort = tmp_path / "cohort.mmdb"
    merge_main(
        SimpleNamespace(
            inputs=[str(b1), str(b2)],
            output=str(cohort),
            modifiedc=False,
            workers=1,
            block_size=2,
            cohort_backend="zarr",
            zarr_row_chunk=2,
            zarr_codec="zstd",
            zarr_clevel=5,
            zarr_shuffle="bitshuffle",
            zarr_codec_threads=1,
        )
    )

    outdir = tmp_path / "pca_cohort"
    run_pca(cohort, outdir)
    assert_outputs(outdir, expected_rows=2, expected_mode="cohort_view")


def test_pca_on_legacy_merged_folder(tmp_path: Path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "columns.txt").write_text("/synthetic/s1.smdb\n/synthetic/s2.smdb\n")
    arr = np.asarray(
        [
            [0.10, 0.12],
            [0.20, 0.19],
            [np.nan, 0.31],
            [0.42, np.nan],
        ],
        dtype=np.float32,
    )
    np.save(legacy / "5mC.npy", arr)

    outdir = tmp_path / "pca_legacy"
    run_pca(legacy, outdir)
    assert_outputs(outdir, expected_rows=2, expected_mode="legacy_npy")
