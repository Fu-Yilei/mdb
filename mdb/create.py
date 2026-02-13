import os
import sys
import time
import logging
import numpy as np
import polars as pl
from typing import Dict, Tuple, List, Optional
from mdb.ont_bed_parsing import ont_bed_parsing
from mdb.pacbio_bed_parsing import pacbio_bed_parsing


def load_index(index_npz: str):
    idx = np.load(index_npz, allow_pickle=True)
    chroms = idx["chroms"]                 # object array
    chrom_offsets = idx["chrom_offsets"]   # int64
    pos0 = idx["pos0"]                     # int32

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



def create_main(args, logger=None):
    # Inputs
    platform = args.platform
    index_npz = args.npz
    bed_path = args.bed
    output_mdb = args.output
    min_cov = args.min_coverage

    if platform not in ("ont", "pacbio"):
        raise ValueError(f"Unsupported platform: {platform}, supported platforms are 'ont' and 'pacbio'")

    chroms, chrom_offsets, pos0, chrom_slices, allowed_chroms = load_index(index_npz)
    print(f"Loaded index: {index_npz} with {len(chroms)} chromosomes and {pos0.shape[0]} CpGs")

    
    os.makedirs(output_mdb, exist_ok=True)
    if platform == "ont":
        stats, matrices, bed_map = ont_bed_parsing(    
            input_path=bed_path,
            chrom_slices=chrom_slices,
            cpg_pos0=pos0,
            allowed_chroms=allowed_chroms,
            min_cov=min_cov,
        )
        with open(os.path.join(output_mdb, "bed_map.txt"), "w") as f:
            for file_name, bed_file in bed_map.items():
                f.write(f"{file_name}\t{bed_file}\n")
        if len(matrices) != 4:
            np.save(os.path.join(output_mdb, "5mC.npy"), matrices[0])
            np.save(os.path.join(output_mdb, "5hmC.npy"), matrices[1])
            print(f"Saved 5mC and 5hmC matrices to {os.path.join(output_mdb, '5mC.npy')} and {os.path.join(output_mdb, '5hmC.npy')}")
        elif len(matrices) == 4:
            m_5mC_plus, m_5mC_minus, m_5hmC_plus, m_5hmC_minus = matrices
            np.save(os.path.join(output_mdb, "5mC_plus.npy"), m_5mC_plus)
            np.save(os.path.join(output_mdb, "5mC_minus.npy"), m_5mC_minus)
            np.save(os.path.join(output_mdb, "5hmC_plus.npy"), m_5hmC_plus)
            np.save(os.path.join(output_mdb, "5hmC_minus.npy"), m_5hmC_minus)
            print(f"Saved 5mC and 5hmC strand-specific matrices to {os.path.join(output_mdb, '5mC_plus.npy')}, {os.path.join(output_mdb, '5mC_minus.npy')}, {os.path.join(output_mdb, '5hmC_plus.npy')}, and {os.path.join(output_mdb, '5hmC_minus.npy')}")

    elif platform == "pacbio":
        stats, matrices, bed_map = pacbio_bed_parsing(
            input_path=bed_path,
            chrom_slices=chrom_slices,
            cpg_pos0=pos0,
            allowed_chroms=allowed_chroms,
            min_cov=min_cov,
        )
        np.save(os.path.join(output_mdb, "5mC.npy"), matrices)
        with open(os.path.join(output_mdb, "bed_map.txt"), "w") as f:
            for file_name, bed_file in bed_map.items():
                f.write(f"{file_name}\t{bed_file}\n")
        print(f"Saved 5mC matrix to {os.path.join(output_mdb, '5mC.npy')}")

