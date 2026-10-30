"""Prevent opt-in validation outputs from replacing their own input stores."""

from pathlib import Path

import numpy as np
import pytest

import run_pacbio_tensorqtl_spotcheck as pacbio_spotcheck
from test_tensorqtl_export import build_cohort


@pytest.mark.parametrize("symlink_input", [False, True], ids=["direct", "symlink"])
def test_pacbio_runner_refuses_snapshot_output_collision_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symlink_input: bool,
) -> None:
    # This is the runner's actual derived snapshot filename. A narrower rerun
    # with its previous snapshot as input must not recreate that input in place.
    output = tmp_path / "previous_run"
    count = build_cohort(
        output / "pacbio_count.regional_snapshot.mmdb",
        np.asarray([[0.1, 0.4], [0.3, 0.7]]), positions=[100, 110], chroms=["chr20"],
    )
    marker = count / "preserve_input.txt"
    marker.write_text("original source must remain intact\n")
    before = {path.relative_to(count): path.read_bytes() for path in count.rglob("*") if path.is_file()}
    source_argument = count
    if symlink_input:
        source_argument = tmp_path / "count_alias.mmdb"
        source_argument.symlink_to(count, target_is_directory=True)
    score = build_cohort(tmp_path / "score.mmdb", np.asarray([[0.2, 0.5]]), positions=[100], chroms=["chr20"])
    genotype_dir = tmp_path / "genotypes"
    genotype_dir.mkdir()
    monkeypatch.setattr("sys.argv", [
        "run_pacbio_tensorqtl_spotcheck.py",
        "--count-db", str(source_argument), "--score-db", str(score),
        "--genotype-dir", str(genotype_dir), "--outdir", str(output),
        "--region", "chr20:100-105", "--plink2", "unused-before-path-validation",
    ])

    with pytest.raises(ValueError, match="must not contain one another"):
        pacbio_spotcheck.main()

    after = {path.relative_to(count): path.read_bytes() for path in count.rglob("*") if path.is_file()}
    assert after == before
    assert marker.read_text() == "original source must remain intact\n"
    assert list(output.iterdir()) == [count]
