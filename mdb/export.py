"""Export cohort methylation views as TensorQTL phenotype BED matrices."""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from mdb.grouping import load_group_index
from mdb.schema import TrackKey
from mdb.storage import (
    available_views,
    load_cohort_index,
    load_cohort_manifest,
    load_view_reader,
)


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value or any(char.isspace() for char in value):
        raise ValueError(f"{label} must be a nonempty, whitespace-free string: {value!r}")
    return value


def _validate_index(chroms: list[str], offsets: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> None:
    n_rows = len(starts)
    if starts.ndim != 1 or ends.ndim != 1 or ends.shape != starts.shape:
        raise ValueError("The coordinate index must contain matching one-dimensional start/end arrays")
    if offsets.ndim != 1 or len(offsets) != len(chroms):
        raise ValueError("The coordinate index chromosome offsets do not match the chromosome names")
    if len(set(chroms)) != len(chroms):
        raise ValueError("The coordinate index contains duplicate chromosome names")
    for chrom in chroms:
        _identifier(chrom, "Chromosome name")
    if n_rows == 0:
        raise ValueError("The input cohort contains no phenotype rows")
    if not len(chroms) or int(offsets[0]) != 0 or np.any(offsets < 0) or np.any(offsets > n_rows):
        raise ValueError("The coordinate index has invalid chromosome offsets")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("The coordinate index chromosome offsets must be sorted")
    chrom_ends = [*offsets[1:], n_rows]
    for chrom, lo, hi in zip(chroms, offsets, chrom_ends, strict=True):
        lo, hi = int(lo), int(hi)
        chrom_starts, chrom_stops = starts[lo:hi], ends[lo:hi]
        if np.any(chrom_starts < 0) or np.any(chrom_stops <= chrom_starts):
            raise ValueError(f"Invalid zero-based half-open coordinates on {chrom}")
        unsorted = (chrom_starts[1:] < chrom_starts[:-1]) | (
            (chrom_starts[1:] == chrom_starts[:-1]) & (chrom_stops[1:] < chrom_stops[:-1])
        )
        if np.any(unsorted):
            raise ValueError(f"Phenotype coordinates must be sorted by start/end on {chrom}")


def _select_samples(columns: dict, n_samples: int, samples: list[str] | None, sample_map: Mapping[str, str] | None):
    stored_ids = list(columns["sample_id"])
    if len(stored_ids) != n_samples or len(set(stored_ids)) != len(stored_ids):
        raise ValueError("The view sample columns must match its width and have unique sample IDs")
    selected = stored_ids if samples is None else list(samples)
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise ValueError("TensorQTL export requires at least two unique selected samples")
    lookup = {sample: idx for idx, sample in enumerate(stored_ids)}
    unknown = [sample for sample in selected if sample not in lookup]
    if unknown:
        raise ValueError(f"Selected samples are absent from the view: {unknown!r}")
    mapping = dict(sample_map) if sample_map is not None else {}
    unknown_keys = [sample for sample in mapping if sample not in lookup]
    if unknown_keys:
        raise ValueError(f"Sample map contains unknown source sample IDs: {unknown_keys!r}")
    for target in mapping.values():
        _identifier(target, "Genotype sample ID")
    output_ids = [_identifier(mapping.get(sample, sample), "Output sample ID") for sample in selected]
    if len(set(output_ids)) != len(output_ids):
        raise ValueError("The selected output sample IDs must be unique after sample mapping")
    if set(output_ids) & {"#chr", "chr", "start", "end", "phenotype_id"}:
        raise ValueError("Output sample IDs must not duplicate the fixed BED column names")
    return selected, output_ids, np.asarray([lookup[sample] for sample in selected], dtype=np.int64), mapping


def _check_output_paths(input_dir: Path, output: Path, receipt: Path, overwrite: bool) -> None:
    for destination in (output, receipt):
        # Check both the directory to be modified and any existing symlink target.
        if destination.parent.resolve().is_relative_to(input_dir) or destination.resolve().is_relative_to(input_dir):
            raise ValueError("Export files must be outside the input database directory")
        if destination.is_dir():
            raise ValueError(f"Export destination is a directory: {destination}")
        if os.path.lexists(destination) and not overwrite:
            raise FileExistsError(f"Export destination already exists; use overwrite=True: {destination}")


def _region_row_ranges(
    region: str | None,
    chroms: list[str],
    offsets: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    grouped: bool,
) -> tuple[str | None, list[tuple[str, int, int]]]:
    """Select contiguous row runs without reading methylation outside a region."""
    chrom_ends = [*offsets[1:], len(starts)]
    if region is None:
        return None, [(chrom, int(lo), int(hi)) for chrom, lo, hi in zip(chroms, offsets, chrom_ends, strict=True)]
    if not isinstance(region, str) or ":" not in region:
        raise ValueError("region must use chromosome:start-end with zero-based half-open coordinates")
    chrom, interval = region.rsplit(":", 1)
    match = re.fullmatch(r"([0-9]+)-([0-9]+)", interval)
    if match is None:
        raise ValueError("region must use chromosome:start-end with nonnegative integer coordinates")
    start, end = (int(coordinate) for coordinate in match.groups())
    if start >= end:
        raise ValueError("region end must be greater than its start")
    if chrom not in chroms:
        raise ValueError(f"Region chromosome is absent from the cohort index: {chrom!r}")
    normalized = f"{chrom}:{start}-{end}"
    chrom_idx = chroms.index(chrom)
    chrom_lo, chrom_hi = int(offsets[chrom_idx]), int(chrom_ends[chrom_idx])
    chrom_starts = starts[chrom_lo:chrom_hi]
    hi = int(np.searchsorted(chrom_starts, end, side="left"))
    if not grouped:
        lo = int(np.searchsorted(chrom_starts, start, side="left"))
        return normalized, [(chrom, chrom_lo + lo, chrom_lo + hi)] if lo < hi else []
    chrom_stops = ends[chrom_lo:chrom_hi]
    if np.all(chrom_stops[1:] >= chrom_stops[:-1]):
        lo = int(np.searchsorted(chrom_stops, start, side="right"))
        return normalized, [(chrom, chrom_lo + lo, chrom_lo + hi)] if lo < hi else []
    # Nested intervals can have unsorted ends even when starts are sorted.
    # Build runs from the overlap mask, so intervening nonoverlapping rows are
    # never read from the methylation matrix.
    overlap = chrom_stops[:hi] > start
    boundaries = np.flatnonzero(np.diff(np.r_[False, overlap, False]))
    return normalized, [
        (chrom, chrom_lo + int(lo), chrom_lo + int(hi))
        for lo, hi in zip(boundaries[::2], boundaries[1::2], strict=True)
    ]


def export_tensorqtl(
    input_path: str,
    output_path: str,
    *,
    assay: str = "5mC",
    haplotype: str = "combined",
    strand: str = "combined",
    samples: list[str] | None = None,
    sample_map: Mapping[str, str] | None = None,
    region: str | None = None,
    min_frac_present: float = 1.0,
    missing: str = "drop",
    batch_rows: int = 8192,
    overwrite: bool = False,
) -> dict:
    """Write beta fractions as phenotypes, retaining existing grouped means.

    ``samples`` selects and orders source IDs; ``sample_map`` renames columns to
    genotype IDs without implicitly selecting them. CpG IDs are ``chr:pos1``;
    group IDs match ``groups.tsv.gz`` (``method_1``, ``method_2``, ...).
    ``region`` uses zero-based half-open coordinates, selecting CpG positions
    or overlapping group intervals while retaining each group's full bounds.
    Filter counts are disjoint: all-missing, insufficient presence, remaining
    missing under the drop policy, then constant after any mean imputation.
    No methylation transformation or genotype/covariate export is performed.
    """
    if missing not in {"drop", "mean"}:
        raise ValueError("missing must be 'drop' or 'mean'")
    min_frac_present = float(min_frac_present)
    if not math.isfinite(min_frac_present) or not 0.0 <= min_frac_present <= 1.0:
        raise ValueError("min_frac_present must be a finite number in [0, 1]")
    if isinstance(batch_rows, bool) or not isinstance(batch_rows, (int, np.integer)) or batch_rows <= 0:
        raise ValueError("batch_rows must be a positive integer")
    batch_rows = int(batch_rows)
    source = Path(input_path).expanduser().resolve()
    output = Path(os.path.abspath(Path(output_path).expanduser()))
    if not (str(output).endswith(".bed") or str(output).endswith(".bed.gz")):
        raise ValueError("TensorQTL output must end in .bed or .bed.gz")
    receipt = Path(str(output) + ".export.json")
    _check_output_paths(source, output, receipt, overwrite)

    manifest = load_cohort_manifest(str(source))
    representation = manifest.get("representation", "cpg")
    if representation not in {"cpg", "cpg_groups"}:
        raise ValueError(f"Unsupported cohort representation: {representation!r}")
    chroms, offsets, pos0 = load_cohort_index(str(source))
    grouping = None
    if representation == "cpg_groups":
        groups = load_group_index(str(source))
        if (
            groups.chroms != chroms
            or not np.array_equal(groups.chrom_offsets, offsets)
            or not np.array_equal(groups.start, pos0)
        ):
            raise ValueError("The group index does not align with the cohort coordinate rows")
        grouping = _identifier(groups.method, "Grouping method")
        if manifest.get("grouping", grouping) != grouping:
            raise ValueError("The group-index method does not match the cohort manifest")
        starts, ends = groups.start, groups.end
        phenotype_ids = "grouping_method + '_' + one-based global group row; matches groups.tsv.gz"
    else:
        starts = pos0
        ends = pos0.astype(np.int64) + 1
        phenotype_ids = "chromosome + ':' + one-based CpG position"
    _validate_index(chroms, offsets, starts, ends)
    if representation == "cpg":
        for lo, hi in zip(offsets, [*offsets[1:], len(starts)], strict=True):
            if int(hi) - int(lo) > 1 and np.any(starts[int(lo) + 1 : int(hi)] == starts[int(lo) : int(hi) - 1]):
                raise ValueError("Duplicate CpG positions would produce duplicate phenotype IDs")
    region, row_ranges = _region_row_ranges(region, chroms, offsets, starts, ends, grouped=grouping is not None)
    rows_considered = sum(hi - lo for _, lo, hi in row_ranges)
    if rows_considered == 0:
        raise ValueError("The requested region contains no phenotype rows; export files were not replaced")

    key = TrackKey(assay, haplotype, strand)
    if key not in available_views(str(source)):
        raise ValueError(f"Requested methylation view is unavailable: {key.name()}")
    reader, columns, _ = load_view_reader(str(source), key)
    temp_paths: list[Path] = []
    try:
        if reader.n_rows != len(starts):
            raise ValueError("The methylation view row count does not match the coordinate index")
        selected, output_ids, sample_indices, mapping = _select_samples(columns, reader.n_samples, samples, sample_map)
        filtered = {"all_missing": 0, "below_min_frac_present": 0, "remaining_missing": 0, "constant": 0}
        rows_written = 0
        imputed_cells = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        os.close(descriptor)
        bed_tmp = Path(temporary)
        temp_paths.append(bed_tmp)
        opener = gzip.open if str(output).endswith(".gz") else open
        with opener(bed_tmp, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["#chr", "start", "end", "phenotype_id", *output_ids])
            for chrom, chrom_lo, chrom_hi in row_ranges:
                for lo in range(int(chrom_lo), int(chrom_hi), batch_rows):
                    hi = min(lo + batch_rows, int(chrom_hi))
                    values = reader.get_block(slice(lo, hi))[:, sample_indices]
                    if values.shape != (hi - lo, len(selected)):
                        raise ValueError("The methylation reader returned an unexpected block shape")
                    if np.any(np.isinf(values)) or np.any(values < 0.0) or np.any(values > 1.0):
                        raise ValueError("Methylation values must be beta fractions in [0, 1] or NaN")
                    present = np.sum(~np.isnan(values), axis=1)
                    keep = present > 0
                    filtered["all_missing"] += int(np.count_nonzero(~keep))
                    insufficient = keep & (present / len(selected) < min_frac_present)
                    filtered["below_min_frac_present"] += int(np.count_nonzero(insufficient))
                    keep &= ~insufficient
                    if missing == "drop":
                        remaining = keep & (present < len(selected))
                        filtered["remaining_missing"] += int(np.count_nonzero(remaining))
                        keep &= ~remaining
                    local_rows = np.flatnonzero(keep)
                    kept_values = values[local_rows]
                    nan_mask = np.isnan(kept_values)
                    if missing == "mean" and np.any(nan_mask):
                        means = np.nansum(kept_values, axis=1, dtype=np.float64) / present[local_rows]
                        kept_values[nan_mask] = np.broadcast_to(means[:, None], kept_values.shape)[nan_mask]
                    variable = np.ptp(kept_values, axis=1) > 0.0
                    filtered["constant"] += int(np.count_nonzero(~variable))
                    if missing == "mean":
                        imputed_cells += int(np.count_nonzero(nan_mask[variable]))
                    for local_row, row_values in zip(local_rows[variable], kept_values[variable], strict=True):
                        row = lo + int(local_row)
                        feature_id = f"{grouping}_{row + 1}" if grouping is not None else f"{chrom}:{int(starts[row]) + 1}"
                        writer.writerow(
                            [chrom, int(starts[row]), int(ends[row]), feature_id, *[format(float(value), ".9g") for value in row_values]]
                        )
                        rows_written += 1
        if rows_written == 0:
            raise ValueError("No eligible nonconstant phenotype rows remain after filtering; export files were not replaced")
        summary = {
            "input_path": str(source),
            "output_path": str(output),
            "receipt_path": str(receipt),
            "backend": manifest["backend"],
            "representation": representation,
            "grouping": grouping,
            "view": {"assay": assay, "haplotype": haplotype, "strand": strand},
            "sample_ids": selected,
            "output_sample_ids": output_ids,
            "sample_map": mapping,
            "n_samples": len(selected),
            "rows_input": int(reader.n_rows),
            "rows_considered": rows_considered,
            "rows_outside_region": int(reader.n_rows) - rows_considered,
            "region": region,
            "region_selection": "overlapping group intervals; full bounds retained" if grouping is not None else "CpG positions in zero-based half-open interval",
            "rows_written": rows_written,
            "rows_filtered": filtered,
            "filter_order": list(filtered),
            "imputed_cells": imputed_cells,
            "missing_policy": missing,
            "min_frac_present": min_frac_present,
            "units": "beta fraction [0,1]",
            "transformation": "none",
            "phenotype_ids": phenotype_ids,
            "coordinate_system": "0-based half-open; individual CpGs [pos0,pos0+1), groups retain groups.npz start/end",
            "group_values": "existing arithmetic means; no reaggregation" if grouping is not None else None,
        }
        descriptor, temporary = tempfile.mkstemp(prefix=f".{receipt.name}.", suffix=".tmp", dir=output.parent)
        receipt_tmp = Path(temporary)
        temp_paths.append(receipt_tmp)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
        # Recheck immediately before publication, including after a long export.
        _check_output_paths(source, output, receipt, overwrite)
        os.replace(bed_tmp, output)
        os.replace(receipt_tmp, receipt)
        return summary
    finally:
        try:
            reader.close()
        finally:
            for temporary in temp_paths:
                temporary.unlink(missing_ok=True)


def export_tensorqtl_main(args) -> dict:
    """Parse CLI sample lists/maps and export a single methylation view."""
    samples = None
    if args.sample_file is not None:
        with open(args.sample_file, encoding="utf-8") as handle:
            samples = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    sample_map = None
    if args.sample_map is not None:
        sample_map = {}
        with open(args.sample_map, encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            if rows.fieldnames != ["sample_id", "genotype_id"]:
                raise ValueError("Sample-map TSV must have exactly the headers sample_id and genotype_id")
            for row in rows:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Sample-map TSV rows must contain exactly two values")
                source = row["sample_id"]
                if not source or "\0" in source:
                    raise ValueError("Source sample IDs in the sample-map TSV must be nonempty and contain no null bytes")
                target = _identifier(row["genotype_id"], "Genotype sample ID")
                if source in sample_map:
                    raise ValueError(f"Sample-map TSV contains duplicate source ID: {source}")
                sample_map[source] = target
    summary = export_tensorqtl(
        args.input,
        args.output,
        assay=args.assay,
        haplotype=args.haplotype,
        strand=args.strand,
        samples=samples,
        sample_map=sample_map,
        region=getattr(args, "region", None),
        min_frac_present=args.min_frac_present,
        missing=args.missing,
        batch_rows=args.batch_rows,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return summary
