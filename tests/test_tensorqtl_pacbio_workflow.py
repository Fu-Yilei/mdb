"""Exercise both PacBio BED schemas through create, merge, group, and export."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mdb.grouping import group_cohort
from mdb.index import index_main
from mdb.main import main
from mdb.schema import TrackKey
from mdb.storage import load_view_columns
from test_zarr_workflow import (
    create_bundle,
    merge_bundles,
    pacbio_count_text,
    pacbio_text,
    write_gzip_text,
)


@pytest.mark.parametrize("backend", ["npy", "zarr"])
@pytest.mark.parametrize("schema", ["discretized", "count"])
def test_pacbio_bed_to_tensorqtl_phenotypes(tmp_path: Path, backend: str, schema: str) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr20\nACGCGTACG\n")
    index = tmp_path / "index.npz"
    index_main(SimpleNamespace(reference=str(reference), output=str(index), sex=False))

    bundles = []
    for sample_number in range(3):
        sample_id = f"pb{sample_number + 1}"
        bed = tmp_path / f"{sample_id}.combined.bed.gz"
        scores = [20 * sample_number, 20 * (sample_number + 1), 50 + 10 * sample_number]
        coverage = [10, 10, 2 if sample_number == 0 else 10]
        if schema == "count":
            rows = [
                ("chr20", position, float(score), cov, round(score * cov / 100), cov - round(score * cov / 100))
                for position, score, cov in zip([1, 3, 7], scores, coverage, strict=True)
            ]
            text = pacbio_count_text(rows)
        else:
            text = pacbio_text([
                ("chr20", position, cov, score)
                for position, cov, score in zip([1, 3, 7], coverage, scores, strict=True)
            ])
        write_gzip_text(bed, text)
        bundle = tmp_path / f"{sample_id}.smdb"
        create_bundle(index, "pacbio", bed, bundle, sample_id)
        bundles.append(bundle)

    cohort = tmp_path / "pacbio.mmdb"
    merge_bundles(bundles, cohort, backend)
    key = TrackKey("5mC", "combined", "combined")
    assert load_view_columns(str(cohort), key)["platform"] == ["pacbio"] * 3
    bed = tmp_path / "individual.bed.gz"
    main(["export-tensorqtl", "-i", str(cohort), "-o", str(bed), "--region", "chr20:0-9"])
    frame = pd.read_csv(bed, sep="\t")
    assert frame.columns[4:].tolist() == ["pb1", "pb2", "pb3"]
    assert frame.iloc[:, :4].values.tolist() == [
        ["chr20", 1, 2, "chr20:2"], ["chr20", 3, 4, "chr20:4"],
    ]
    # Covered zero methylation is retained; the third CpG is missing in pb1
    # because coverage=2 fails create's minimum coverage of 5.
    np.testing.assert_allclose(frame.iloc[:, 4:].to_numpy(), [[0, 0.2, 0.4], [0.2, 0.4, 0.6]], atol=1e-7)

    grouped = tmp_path / "pacbio.denovo.mmdb"
    group_cohort(
        str(cohort), str(grouped), "denovo", denovo_max_gap_bp=3,
        denovo_min_correlation=0.99, output_backend=backend,
    )
    group_bed = tmp_path / "groups.bed.gz"
    main(["export-tensorqtl", "-i", str(grouped), "-o", str(group_bed)])
    group_frame = pd.read_csv(group_bed, sep="\t")
    assert group_frame.iloc[:, :4].values.tolist() == [["chr20", 1, 5, "denovo_1"]]
    np.testing.assert_allclose(group_frame.iloc[:, 4:].to_numpy(), [[0.1, 0.3, 0.5]], atol=1e-7)
