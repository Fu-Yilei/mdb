import os
from typing import Dict, Tuple

import numpy as np

from mdb.ont_bed_parsing import ont_bed_parsing
from mdb.pacbio_bed_parsing import pacbio_bed_parsing
from mdb.schema import (
    STRAND_COMBINED,
    SampleBundle,
    SampleTrack,
    TrackKey,
    encode_percent,
    infer_sample_id,
)
from mdb.storage import write_sample_bundle


def load_index(index_npz: str):
    idx = np.load(index_npz, allow_pickle=True)
    chroms = idx["chroms"]  # object array
    chrom_offsets = idx["chrom_offsets"]  # int64
    pos0 = idx["pos0"]  # int32

    n_cpg = int(pos0.shape[0])
    chrom_slices: Dict[str, Tuple[int, int]] = {}
    for i, c in enumerate(chroms):
        c = str(c)
        s = int(chrom_offsets[i])
        e = int(chrom_offsets[i + 1]) if i + 1 < len(chrom_offsets) else n_cpg
        chrom_slices[c] = (s, e)

    chrom_to_idx = {str(c): i for i, c in enumerate(chroms)}
    allowed_chroms = list(chrom_to_idx.keys())
    return chroms, chrom_offsets, pos0, chrom_slices, allowed_chroms


def _to_sample_track(
    matrix: np.ndarray,
    col_idx: int,
    *,
    key: TrackKey,
    platform: str,
    input_tag: str,
    source_path: str,
    min_coverage: int,
) -> SampleTrack:
    dense = np.asarray(matrix[:, col_idx], dtype=np.float32)
    present = np.isfinite(dense)
    row_ids = np.flatnonzero(present).astype(np.uint32, copy=False)

    fractions = np.clip(dense[present], 0.0, 1.0).astype(np.float32, copy=False)
    values = encode_percent(fractions * 100.0)
    coverage = np.full(row_ids.shape[0], np.uint16(max(int(min_coverage), 0)), dtype=np.uint16)

    return SampleTrack(
        key=key,
        row_ids=row_ids,
        values=values,
        coverage=coverage,
        platform=platform,
        input_tag=input_tag,
        source_path=source_path,
        min_coverage=int(min_coverage),
    )


def _write_bed_map(path: str, bed_map: Dict[str, str]) -> None:
    with open(os.path.join(path, "bed_map.txt"), "w") as f:
        for file_name, bed_file in bed_map.items():
            f.write(f"{file_name}\t{bed_file}\n")


def create_main(args, logger=None):
    platform = str(args.platform).lower()
    index_npz = str(args.npz)
    bed_path = str(args.bed)
    output_mdb = str(args.output)
    min_cov = int(args.min_coverage)
    sample_id = str(args.sample_id) if getattr(args, "sample_id", None) else infer_sample_id(output_mdb)

    if platform not in {"ont", "pacbio"}:
        raise ValueError(f"Unsupported platform: {platform}, supported platforms are 'ont' and 'pacbio'")

    chroms, _chrom_offsets, pos0, chrom_slices, allowed_chroms = load_index(index_npz)
    print(f"Loaded index: {index_npz} with {len(chroms)} chromosomes and {pos0.shape[0]} CpGs")

    tracks: dict[TrackKey, SampleTrack] = {}
    bed_map: Dict[str, str] = {}

    if platform == "ont":
        _stats, matrices, bed_map = ont_bed_parsing(
            input_path=bed_path,
            chrom_slices=chrom_slices,
            cpg_pos0=pos0,
            allowed_chroms=allowed_chroms,
            min_cov=min_cov,
        )
        if len(matrices) == 2:
            m_5mc, m_5hmc = matrices
            for col_idx, (input_tag, source_path) in enumerate(bed_map.items()):
                key_5mc = TrackKey("5mC", input_tag, STRAND_COMBINED)
                key_5hmc = TrackKey("5hmC", input_tag, STRAND_COMBINED)
                tracks[key_5mc] = _to_sample_track(
                    m_5mc,
                    col_idx,
                    key=key_5mc,
                    platform=platform,
                    input_tag=input_tag,
                    source_path=source_path,
                    min_coverage=min_cov,
                )
                tracks[key_5hmc] = _to_sample_track(
                    m_5hmc,
                    col_idx,
                    key=key_5hmc,
                    platform=platform,
                    input_tag=input_tag,
                    source_path=source_path,
                    min_coverage=min_cov,
                )
        elif len(matrices) == 4:
            m_5mc_plus, m_5mc_minus, m_5hmc_plus, m_5hmc_minus = matrices
            for col_idx, (input_tag, source_path) in enumerate(bed_map.items()):
                for assay, strand, matrix in (
                    ("5mC", "plus", m_5mc_plus),
                    ("5mC", "minus", m_5mc_minus),
                    ("5hmC", "plus", m_5hmc_plus),
                    ("5hmC", "minus", m_5hmc_minus),
                ):
                    key = TrackKey(assay, input_tag, strand)
                    tracks[key] = _to_sample_track(
                        matrix,
                        col_idx,
                        key=key,
                        platform=platform,
                        input_tag=input_tag,
                        source_path=source_path,
                        min_coverage=min_cov,
                    )
        else:
            raise RuntimeError(f"Unexpected ONT matrix payload length: {len(matrices)}")

    else:
        _stats, matrix, bed_map = pacbio_bed_parsing(
            input_path=bed_path,
            chrom_slices=chrom_slices,
            cpg_pos0=pos0,
            allowed_chroms=allowed_chroms,
            min_cov=min_cov,
        )
        for col_idx, (input_tag, source_path) in enumerate(bed_map.items()):
            key = TrackKey("5mC", input_tag, STRAND_COMBINED)
            tracks[key] = _to_sample_track(
                matrix,
                col_idx,
                key=key,
                platform=platform,
                input_tag=input_tag,
                source_path=source_path,
                min_coverage=min_cov,
            )

    if not tracks:
        raise RuntimeError("No tracks were created from the provided input.")

    bundle = SampleBundle(
        sample_id=sample_id,
        platform=platform,
        index_path=os.path.abspath(index_npz),
        tracks=tracks,
    )
    write_sample_bundle(output_mdb, bundle)
    _write_bed_map(output_mdb, bed_map)

    print(f"Wrote sample bundle: {output_mdb}")
    print(f"Sample ID: {sample_id}")
    print(f"Tracks: {len(tracks)}")
