from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from mdb.grouping import (
    DECODE_PAPER_URL,
    LOYFER_PAPER_URL,
    PREDEFINED_RESOURCE_FILES,
    PredefinedGroupingIndex,
    save_predefined_index,
)
from mdb.index import fasta_iter, find_cpg_pos0


def build_loyfer_index(source_path: str, output_path: str) -> PredefinedGroupingIndex:
    """Convert the SniffCell five-column block index to a compact stored resource."""
    names = ["chrom", "start", "end", "start_cpg", "end_cpg"]
    collected: dict[str, dict[str, list[np.ndarray]]] = {}
    chrom_order: list[str] = []
    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        header=None,
        names=names,
        compression="infer",
        chunksize=500_000,
        dtype={"chrom": "string", "start": "int64", "end": "int64", "start_cpg": "int64", "end_cpg": "int64"},
    ):
        for chrom, sub in chunk.groupby("chrom", sort=False):
            chrom = str(chrom)
            if chrom not in collected:
                collected[chrom] = {"start": [], "end": []}
                chrom_order.append(chrom)
            collected[chrom]["start"].append(sub["start"].to_numpy(dtype=np.uint32))
            collected[chrom]["end"].append(sub["end"].to_numpy(dtype=np.uint32))

    offsets: list[int] = []
    start_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    running = 0
    for chrom in chrom_order:
        offsets.append(running)
        starts = np.concatenate(collected[chrom]["start"])
        ends = np.concatenate(collected[chrom]["end"])
        start_parts.append(starts)
        end_parts.append(ends)
        running += int(starts.shape[0])

    index = PredefinedGroupingIndex(
        method="loyfer",
        version="grch38_v1",
        assembly="GRCh38",
        chroms=chrom_order,
        chrom_offsets=np.asarray(offsets, dtype=np.int64),
        start=np.concatenate(start_parts),
        end=np.concatenate(end_parts),
        left_shift_bp=1,
        parameters={
            "definition": "stored non-overlapping methylation blocks used by SniffCell find",
            "coordinate_anchor": "G",
        },
        citation=LOYFER_PAPER_URL,
        source="SniffMeth/atlas/all_celltypes_blocks.index.gz",
    )
    save_predefined_index(output_path, index)
    return index


def build_decode_index(
    reference_path: str,
    output_path: str,
    *,
    chromosomes: list[str] | None = None,
) -> PredefinedGroupingIndex:
    """Materialize the fixed DECODE/Nanopolish 10-bp CpG-unit definition."""
    requested = chromosomes or [f"chr{i}" for i in range(1, 23)]
    requested_set = set(requested)
    positions_by_chrom: dict[str, np.ndarray] = {}
    for chrom, sequence in fasta_iter(reference_path):
        if chrom in requested_set:
            positions_by_chrom[chrom] = find_cpg_pos0(sequence).astype(np.uint32, copy=False)
    missing = [chrom for chrom in requested if chrom not in positions_by_chrom]
    if missing:
        raise ValueError(f"Reference is missing requested chromosomes: {missing}")

    offsets: list[int] = []
    start_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    running = 0
    n_reference_cpgs = 0
    fingerprint = hashlib.sha256()
    for chrom in requested:
        pos0 = positions_by_chrom[chrom]
        fingerprint.update(chrom.encode("utf-8") + b"\0")
        fingerprint.update(pos0.tobytes())
        n_reference_cpgs += int(pos0.shape[0])
        break_before = np.ones(pos0.shape[0], dtype=bool)
        if pos0.shape[0] > 1:
            break_before[1:] = np.diff(pos0.astype(np.int64)) > 10
        local_start = np.flatnonzero(break_before).astype(np.int64)
        local_end = np.r_[local_start[1:], pos0.shape[0]].astype(np.int64)
        starts = pos0[local_start]
        ends = (pos0[local_end - 1].astype(np.uint64) + 2).astype(np.uint32)
        offsets.append(running)
        start_parts.append(starts)
        end_parts.append(ends)
        running += int(starts.shape[0])

    index = PredefinedGroupingIndex(
        method="decode",
        version="grch38_10bp_v1",
        assembly="GRCh38",
        chroms=list(requested),
        chrom_offsets=np.asarray(offsets, dtype=np.int64),
        start=np.concatenate(start_parts),
        end=np.concatenate(end_parts),
        left_shift_bp=0,
        parameters={
            "max_adjacent_cpg_distance_bp": 10,
            "definition": "Nanopolish CpG sites within 10-bp distance form one CpG unit",
            "definition_scope": "fixed GRCh38 autosomal CpG index; study-specific QC exclusions are not recreated",
            "n_reference_cpgs": n_reference_cpgs,
            "reference_cpg_sha256": fingerprint.hexdigest(),
        },
        citation=DECODE_PAPER_URL,
        source="GRCh38 autosomal reference sequence",
    )
    save_predefined_index(output_path, index)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build versioned mdb predefined grouping indexes")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loyfer-source")
    parser.add_argument("--decode-reference")
    parser.add_argument("--chromosomes", help="Comma-separated DECODE chromosome list; default=chr1-chr22")
    args = parser.parse_args(argv)
    if not args.loyfer_source and not args.decode_reference:
        parser.error("provide --loyfer-source and/or --decode-reference")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.loyfer_source:
        index = build_loyfer_index(
            args.loyfer_source,
            str(output_dir / PREDEFINED_RESOURCE_FILES["loyfer"]),
        )
        print(f"loyfer\t{index.n_groups}\t{output_dir / PREDEFINED_RESOURCE_FILES['loyfer']}")
    if args.decode_reference:
        chroms = [x.strip() for x in args.chromosomes.split(",") if x.strip()] if args.chromosomes else None
        index = build_decode_index(
            args.decode_reference,
            str(output_dir / PREDEFINED_RESOURCE_FILES["decode"]),
            chromosomes=chroms,
        )
        print(f"decode\t{index.n_groups}\t{output_dir / PREDEFINED_RESOURCE_FILES['decode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
