"""MethBat format checks and public CLI workflows, with independent expected values."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mdb.main import main
from mdb.schema import TrackKey
from mdb.storage import (
    list_tracks, load_view_columns, load_view_reader,
    query_sample_track, query_cohort_range,
)


HEADER = (
    "#chrom\tstart\tend\tname\tscore\tstrand\tmod_score\ttype\tcov\tmod_count\t"
    "unmod_count\tinferred_unmod_count\tdiff_base_count\tavg_mod_score\tavg_unmod_score\n"
)


def row(start=1, pct=33.3, track="Total", cov=12, strand=".", code="m", chrom="chr20"):
    # Deliberately unrelated score/probability fields: the importer must use mod_score.
    return [chrom, start, start + 1, code, 999, strand, pct, track, cov, 4, 8, 0, 0, 0.91, 0.08]


def write_bed(path, rows, assay="5mC", header=HEADER):
    text = f"##methbat_version=1.1.0\n##base_modification={assay}\n" + header
    text += "".join("\t".join(map(str, fields)) + "\n" for fields in rows)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)


def make_index(tmp_path):
    reference = tmp_path / "ref.fa"
    reference.write_text(">chr20\nACGCGTACG\n>chr21\nAAAA\n")
    index = tmp_path / "index.npz"
    main(["index", "-r", str(reference), "-o", str(index)])
    return index


def create(index, source, output, sample_id="sample", cov=5, platform="pacbio"):
    main([
        "create", "-p", platform, "--input-format", "methbat", "-n", str(index),
        "-b", str(source), "-o", str(output), "--sample-id", sample_id, "-c", str(cov),
    ])


@pytest.mark.parametrize("suffix", [".bed", ".bed.gz"])
def test_tracks_values_coverage_and_missingness(tmp_path, suffix):
    index = make_index(tmp_path)
    bed = tmp_path / ("sample.5mC" + suffix)
    write_bed(bed, [
        row(pct=33.3, cov=12), row(pct=0, track="hap1", cov=7),
        row(pct=100, track="hap2", cov=5), row(start=3, cov=4),
        row(start=999), row(chrom="chrUnknown"), row(chrom="chr21"),
    ])
    output = tmp_path / "sample.smdb"
    create(index, bed, output)
    assert set(list_tracks(str(output))) == {TrackKey("5mC", h, "combined") for h in ("combined", "hp1", "hp2")}
    for hap, expected, coverage in [("combined", 33.3, 12), ("hp1", 0, 7), ("hp2", 100, 5)]:
        key = TrackKey("5mC", hap, "combined")
        result = query_sample_track(str(output), key, "chr20", 1)
        assert result["value_percent"] == expected
        assert result["coverage"] == coverage
        assert query_sample_track(str(output), key, "chr20", 3) is None
    receipt = json.loads((output / "import_summary.json").read_text())
    assert receipt["files"][0]["tracks"]["5mC__combined__combined"] == {
        "rows_input": 5, "rows_below_min_coverage": 1,
        "rows_not_in_index": 3, "rows_imported": 1,
    }


def test_prefix_cpg_5hmc_strands_and_empty_files(tmp_path):
    index = make_index(tmp_path)
    prefix = tmp_path / "sample"
    write_bed(tmp_path / "sample.5mC.bed.gz", [row()])
    write_bed(tmp_path / "sample.5hmC.bed.gz", [
        row(pct=12.5, strand="+", code="h", cov=8),
        row(start=2, pct=75, strand="-", code="h", cov=9),
        row(start=4, pct=25, track="hap2", strand="-", code="h", cov=11),
        row(start=0, strand="-", code="h"),  # must not clamp -1 to reference 0
        row(start=3, strand="-", code="h"),  # canonical position 2 is not a CpG
    ], assay="5hmC")
    write_bed(tmp_path / "sample.6mA.bed.gz", [], assay="6mA")
    output = tmp_path / "sample.smdb"
    create(index, prefix, output)
    for hap, strand, pos, pct, cov in [
        ("combined", "plus", 1, 12.5, 8), ("combined", "minus", 1, 75, 9),
        ("hp2", "minus", 3, 25, 11),
    ]:
        result = query_sample_track(str(output), TrackKey("5hmC", hap, strand), "chr20", pos)
        assert (result["value_percent"], result["coverage"]) == (pct, cov)
    receipt = json.loads((output / "import_summary.json").read_text())
    assert len(receipt["ignored_6mA_files"]) == 1
    assert receipt["files"][1]["tracks"]["5hmC__combined__minus"]["rows_not_in_index"] == 2

    # Empty per-modification files are emitted by real MethBat when tags are absent.
    write_bed(tmp_path / "sample.5hmC.bed.gz", [], assay="5hmC")
    create(index, prefix, tmp_path / "empty_h.smdb")
    assert list_tracks(str(tmp_path / "empty_h.smdb")) == [TrackKey("5mC", "combined", "combined")]


@pytest.mark.parametrize("field,value,match", [
    ("start", -1, "coordinates"), ("end", 9, "coordinates"),
    ("mod_score", -0.1, "mod_score"), ("mod_score", 100.1, "mod_score"),
    ("mod_score", "NaN", "mod_score"), ("mod_score", "inf", "mod_score"),
    ("mod_score", "", "missing values"), ("cov", -1, "coverage"),
    ("cov", 0, "coverage"), ("cov", 65535, "coverage"),
    ("type", "hap3", "unknown MethBat type"), ("name", "h", "modification codes"),
    ("strand", "+", "invalid strand"),
])
def test_invalid_rows_fail_before_replacing_output(tmp_path, field, value, match):
    index = make_index(tmp_path)
    bed = tmp_path / "sample.5mC.bed"
    fields = row()
    fields[HEADER.strip().split("\t").index(field)] = value
    write_bed(bed, [fields])
    output = tmp_path / "existing.smdb"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing database")
    with pytest.raises(ValueError, match=match):
        create(index, bed, output)
    assert marker.read_text() == "existing database"


@pytest.mark.parametrize("case,match", [
    ("duplicate", "duplicate CpG"), ("6mA", "6mA is not supported"),
    ("empty", "No MethBat CpG rows"), ("off_index", "No MethBat CpG rows"),
    ("missing_header", "missing columns"), ("negative_coverage", "nonnegative"),
    ("wrong_platform", "incompatible"), ("ambiguous", "Ambiguous"),
])
def test_rejected_inputs(tmp_path, case, match):
    index = make_index(tmp_path)
    bed = tmp_path / "sample.5mC.bed"
    write_bed(bed, [row()])
    source, kwargs = bed, {}
    if case == "duplicate":
        write_bed(bed, [row(), row()])
    elif case == "6mA":
        write_bed(bed, [], assay="6mA")
    elif case == "empty":
        write_bed(bed, [])
    elif case == "off_index":
        write_bed(bed, [row(start=999)])
    elif case == "missing_header":
        write_bed(bed, [row()], header=HEADER.replace("mod_score\t", "other_score\t", 1))
    elif case == "negative_coverage":
        kwargs["cov"] = -1
    elif case == "wrong_platform":
        kwargs["platform"] = "ont"
    elif case == "ambiguous":
        write_bed(tmp_path / "sample.5mC.bed.gz", [row()])
        source = tmp_path / "sample"
    output = tmp_path / "bad.smdb"
    with pytest.raises(ValueError, match=match):
        create(index, source, output, **kwargs)
    assert not output.exists()


def test_inputs_cannot_be_inside_output(tmp_path):
    index = make_index(tmp_path)
    output = tmp_path / "protected"
    output.mkdir()
    bed = output / "sample.5mC.bed"
    write_bed(bed, [row()])
    original = bed.read_bytes()
    with pytest.raises(ValueError, match="must not contain"):
        create(index, bed, output)
    assert bed.read_bytes() == original


@pytest.mark.parametrize("backend", ["npy", "zarr"])
def test_5hmc_merges_with_modkit_strand_views(tmp_path, backend):
    from test_zarr_workflow import ont_line

    index = make_index(tmp_path)
    methbat_bed = tmp_path / "sample.5hmC.bed"
    write_bed(methbat_bed, [
        row(pct=25, strand="+", code="h"),
        row(start=2, pct=75, strand="-", code="h"),
    ], assay="5hmC")
    methbat_bundle = tmp_path / "methbat.smdb"
    create(index, methbat_bed, methbat_bundle, sample_id="pacbio")
    ont_bed = tmp_path / "ont.bed"
    ont_bed.write_text(ont_line("chr20", 1, "h", 11, 60, "+") + ont_line("chr20", 2, "h", 11, 20, "-"))
    ont_bundle = tmp_path / "ont.smdb"
    main([
        "create", "-p", "ont", "--input-format", "modkit", "-n", str(index),
        "-b", str(ont_bed), "-o", str(ont_bundle), "--sample-id", "ont",
    ])
    cohort = tmp_path / "cohort.mmdb"
    main(["merge", "-i", str(methbat_bundle), str(ont_bundle), "-o", str(cohort), "--cohort-backend", backend])
    for strand, expected in [("plus", [0.25, 0.60]), ("minus", [0.75, 0.20])]:
        key = TrackKey("5hmC", "combined", strand)
        reader, _, _ = load_view_reader(str(cohort), key)
        try:
            np.testing.assert_allclose(reader.read_rows(slice(0, 1)), [expected], atol=1e-7)
            assert np.isnan(reader.read_rows(slice(1, 3))).all()
        finally:
            reader.close()
        assert load_view_columns(str(cohort), key)["platform"] == ["pacbio", "ont"]


@pytest.mark.parametrize("backend", ["npy", "zarr"])
def test_create_merge_append_query_group_export(tmp_path, backend, capsys):
    index = make_index(tmp_path)
    bundles = []
    for number in range(3):
        bed = tmp_path / f"s{number}.5mC.bed.gz"
        rows = []
        for track, extra in [("Total", 0), ("hap1", 5), ("hap2", 10)]:
            rows.extend([
                row(start=1, pct=20 * number + extra, track=track, cov=10 + number),
                row(start=3, pct=20 * (number + 1) + extra, track=track, cov=10 + number),
                row(start=7, pct=50 + number + extra, track=track, cov=2 if number == 0 else 12),
            ])
        write_bed(bed, rows)
        bundle = tmp_path / f"s{number}.smdb"
        create(index, bed, bundle, sample_id=f"s{number}")
        bundles.append(bundle)
    cohort = tmp_path / f"{backend}.mmdb"
    main(["merge", "-i", *map(str, bundles[:2]), "-o", str(cohort), "--cohort-backend", backend])
    main(["append", "-c", str(cohort), "-i", str(bundles[2])])
    for hap, extra in [("combined", 0), ("hp1", 5), ("hp2", 10)]:
        key = TrackKey("5mC", hap, "combined")
        expected = np.array([[0, 20, 40], [20, 40, 60], [np.nan, 51, 52]]) / 100 + extra / 100
        reader, _, _ = load_view_reader(str(cohort), key)
        try:
            np.testing.assert_allclose(reader.read_rows(slice(0, 3)), expected, atol=1e-7)
        finally:
            reader.close()
        assert load_view_columns(str(cohort), key)["platform"] == ["pacbio"] * 3
        result = query_cohort_range(str(cohort), key, "s0", "chr20", 1, 7)
        assert result["count"] == 3
        assert np.isnan(result["records"][2]["value_fraction"])
        capsys.readouterr()
        main(["query", "-i", str(cohort), "--sample-id", "s1", "--haplotype", hap, "--locus", "chr20:1"])
        assert json.loads(capsys.readouterr().out)["value_fraction"] == (20 + extra) / 100
        bed = tmp_path / f"{hap}.bed.gz"
        main(["export-tensorqtl", "-i", str(cohort), "-o", str(bed), "--haplotype", hap])
        frame = pd.read_csv(bed, sep="\t")
        np.testing.assert_allclose(frame.iloc[:, 4:], expected[:2], atol=1e-7)
        assert frame["start"].tolist() == [1, 3]
    grouped = tmp_path / "groups.mmdb"
    main([
        "group", "-i", str(cohort), "-o", str(grouped), "--grouping", "denovo",
        "--denovo-max-gap-bp", "3", "--denovo-min-correlation", "0.99", "--cohort-backend", backend,
    ])
    bed = tmp_path / "groups.bed.gz"
    main(["export-tensorqtl", "-i", str(grouped), "-o", str(bed)])
    frame = pd.read_csv(bed, sep="\t")
    assert frame.iloc[:, :4].values.tolist() == [["chr20", 1, 5, "denovo_1"]]
    np.testing.assert_allclose(frame.iloc[:, 4:], [[0.1, 0.3, 0.5]], atol=1e-7)
