import numpy as np
from tqdm.auto import tqdm
import logging
import argparse


def fasta_iter(path):
    name = None
    seq_chunks = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq_chunks).upper()
                name = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if name is not None:
            yield name, "".join(seq_chunks).upper()

def find_cpg_pos0(seq: str) -> np.ndarray:
    # 0-based positions of C in "CG"
    pos = []
    i = seq.find("CG")
    while i != -1:
        pos.append(i)
        i = seq.find("CG", i + 1)
    return np.asarray(pos, dtype=np.int32)



def map_positions_to_rows(chrom: str, starts_np: np.ndarray) -> np.ndarray:
    """
    Map 0-based CpG start positions on a given chrom to global row indices.
    Returns int64 array of row indices, with -1 for positions not found.
    """
    starts_np = np.asarray(starts_np)
    if starts_np.ndim != 1:
        raise ValueError("starts_np must be a 1D array of 0-based CpG C positions.")

    if chrom not in chrom_slices:
        return np.full(starts_np.shape[0], -1, dtype=np.int64)

    s, e = chrom_slices[chrom]
    chrom_pos = cpg_pos0[s:e]  # sorted within chrom

    # Ensure searchsorted dtype compatibility (int32 is fine; int64 also fine)
    j = np.searchsorted(chrom_pos, starts_np.astype(chrom_pos.dtype, copy=False), side="left")
    ok = (j < chrom_pos.shape[0]) & (chrom_pos[j] == starts_np.astype(chrom_pos.dtype, copy=False))

    out = np.full(starts_np.shape[0], -1, dtype=np.int64)
    out[ok] = (s + j[ok]).astype(np.int64, copy=False)
    return out

def index_main(args):

    reference = args.reference
    output = args.output
    sex = args.sex

    chroms = []
    chrom_offsets = []
    cpg_pos0_all = []
    total = 0

    autosomes = [f"chr{str(i)}" for i in range(1, 23)]
    matrix_chroms = autosomes
    if sex == True:
        matrix_chroms += ["chrX", "chrY"]
    for chrom, seq in tqdm(fasta_iter(reference), desc="Scanning chromosomes", unit="chrom", total=len(matrix_chroms)):
        if chrom not in matrix_chroms:
            continue

        pos0_chr = find_cpg_pos0(seq)  # int32, sorted increasing
        chroms.append(chrom)
        chrom_offsets.append(total)    # start index into global arrays
        cpg_pos0_all.append(pos0_chr)
        total += pos0_chr.size

    # finalize offsets as int64 numpy array
    chrom_offsets = np.asarray(chrom_offsets, dtype=np.int64)

    # Build global pos0 array
    if total == 0:
        raise RuntimeError("No CpGs found for the requested chromosomes. Check FASTA contig names.")

    cpg_pos0 = np.concatenate(cpg_pos0_all).astype(np.int32, copy=False)
    n_cpg = int(cpg_pos0.shape[0])

    # Build global chrom_id array (per-row chromosome index)
    chrom_id = np.empty(n_cpg, dtype=np.int32)
    for i, pos0_chr in enumerate(cpg_pos0_all):
        start = int(chrom_offsets[i])
        chrom_id[start:start + pos0_chr.size] = i

    print("Total CpGs:", n_cpg)
    print("Chromosomes used:", chroms)

    np.savez_compressed(
        output,
        chroms=np.array(chroms, dtype=object),
        chrom_offsets=chrom_offsets,
        chrom_id=chrom_id,
        pos0=cpg_pos0,
    )
    print("Saved: ", output)
