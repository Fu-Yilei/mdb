from __future__ import annotations

import json

from mdb.schema import TrackKey
from mdb.storage import detect_store_kind, query_cohort_point, query_cohort_range, query_sample_range, query_sample_track


def parse_locus(locus: str) -> tuple[str, int]:
    chrom, pos = locus.split(":", 1)
    return chrom, int(pos)


def parse_region(region: str) -> tuple[str, int, int]:
    chrom, coords = region.split(":", 1)
    start, end = coords.split("-", 1)
    return chrom, int(start), int(end)


def query_main(args):
    key = TrackKey(assay=args.assay, haplotype=args.haplotype, strand=args.strand)
    is_point = args.locus is not None
    if is_point:
        chrom, pos0 = parse_locus(args.locus)
    else:
        chrom, start_pos0, end_pos0 = parse_region(args.region)

    kind = detect_store_kind(args.input)

    if kind == "sample_store_npy":
        if is_point:
            result = query_sample_track(args.input, key, chrom, pos0)
        else:
            result = query_sample_range(args.input, key, chrom, start_pos0, end_pos0)
    elif kind in {"cohort_store_npy", "cohort_store_zarr"}:
        if not args.sample_id:
            raise ValueError("--sample-id is required when querying a cohort store.")
        if is_point:
            result = query_cohort_point(args.input, key, args.sample_id, chrom, pos0)
        else:
            result = query_cohort_range(args.input, key, args.sample_id, chrom, start_pos0, end_pos0)
    else:
        raise ValueError(f"Unsupported input kind: {kind}")

    if result is None:
        print("null")
    else:
        print(json.dumps(result, indent=2))
