#!/usr/bin/env python3
import glob
import os
import numpy as np
from typing import Tuple, Optional, Dict, List


def process_ont_pb_no_strand_npys(
    input_mdb_folder: str,
    modifiedc: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    No-strand layout.

    Files:
      - 5mC.npy  (required)
      - 5hmC.npy (optional)

    Logic (always first column):
      - If only 5mC.npy exists: return (5mC[:,0], None)
      - If modifiedc=False:     return (5mC[:,0], 5hmC[:,0])
      - If modifiedc=True:      return (5mC[:,0] + 5hmC[:,0], None)
    """
    p_5mc = os.path.join(input_mdb_folder, "5mC.npy")
    p_5hmc = os.path.join(input_mdb_folder, "5hmC.npy")

    if not os.path.exists(p_5mc):
        raise FileNotFoundError(f"Missing required file: {p_5mc}")

    m_5mc = np.load(p_5mc, mmap_mode="r")
    if m_5mc.ndim != 2:
        raise ValueError(f"5mC.npy must be 2D, got shape {m_5mc.shape}")

    # ---- only 5mC present ----
    if not os.path.exists(p_5hmc):
        return m_5mc[:, 0], None

    m_5hmc = np.load(p_5hmc, mmap_mode="r")
    if m_5hmc.ndim != 2:
        raise ValueError(f"5hmC.npy must be 2D, got shape {m_5hmc.shape}")

    if m_5mc.shape != m_5hmc.shape:
        raise ValueError(f"Shape mismatch: 5mC={m_5mc.shape}, 5hmC={m_5hmc.shape}")

    if modifiedc:
        return m_5mc[:, 0] + m_5hmc[:, 0], None

    return m_5mc[:, 0], m_5hmc[:, 0]


def process_ont_strand_split_npys(
    input_mdb_folder: str,
    modifiedc: bool,
) -> Dict[str, np.ndarray]:
    """
    Strand-split layout.

    Expected (subset allowed):
      - 5mC_plus.npy   (required)
      - 5mC_minus.npy  (required)
      - 5hmC_plus.npy  (optional)
      - 5hmC_minus.npy (optional)

    Always uses FIRST column only.

    Returns dict of 1D vectors (shape (n_cpg,)):

      If only 5mC +/- exist:
        {"+5mC", "-5mC", "5mC"} where "5mC" = nanmean([+5mC, -5mC])

      If 5mC and 5hmC exist:
        - modifiedc=True:
            {"+modifiedC", "-modifiedC", "modifiedC"} where +/- are strand-specific sums,
            and "modifiedC" = nanmean([+modifiedC, -modifiedC])
        - modifiedc=False:
            {"+5mC","-5mC","+5hmC","-5hmC","5mC","5hmC"} with combined as nanmean across strands.
    """
    p_5mc_p = os.path.join(input_mdb_folder, "5mC_plus.npy")
    p_5mc_m = os.path.join(input_mdb_folder, "5mC_minus.npy")
    p_5hmc_p = os.path.join(input_mdb_folder, "5hmC_plus.npy")
    p_5hmc_m = os.path.join(input_mdb_folder, "5hmC_minus.npy")

    if not os.path.exists(p_5mc_p):
        raise FileNotFoundError(f"Missing required file: {p_5mc_p}")
    if not os.path.exists(p_5mc_m):
        raise FileNotFoundError(f"Missing required file: {p_5mc_m}")

    m5_p = np.load(p_5mc_p, mmap_mode="r")
    m5_m = np.load(p_5mc_m, mmap_mode="r")

    if m5_p.ndim != 2 or m5_m.ndim != 2:
        raise ValueError(f"Expected 2D arrays. Got 5mC_plus={m5_p.shape}, 5mC_minus={m5_m.shape}")
    if m5_p.shape != m5_m.shape:
        raise ValueError(f"Shape mismatch: 5mC_plus={m5_p.shape}, 5mC_minus={m5_m.shape}")

    v5_p = m5_p[:, 0]
    v5_m = m5_m[:, 0]

    has_h = os.path.exists(p_5hmc_p) and os.path.exists(p_5hmc_m)

    if not has_h:
        return {
            "+5mC": v5_p,
            "-5mC": v5_m,
            "5mC": np.nanmean(np.stack([v5_p, v5_m], axis=0), axis=0),
        }

    h5_p = np.load(p_5hmc_p, mmap_mode="r")
    h5_m = np.load(p_5hmc_m, mmap_mode="r")

    if h5_p.ndim != 2 or h5_m.ndim != 2:
        raise ValueError(f"Expected 2D arrays. Got 5hmC_plus={h5_p.shape}, 5hmC_minus={h5_m.shape}")
    if h5_p.shape != h5_m.shape:
        raise ValueError(f"Shape mismatch: 5hmC_plus={h5_p.shape}, 5hmC_minus={h5_m.shape}")
    if h5_p.shape != m5_p.shape:
        raise ValueError(f"Shape mismatch between 5mC and 5hmC: 5mC={m5_p.shape}, 5hmC_plus={h5_p.shape}")

    vh_p = h5_p[:, 0]
    vh_m = h5_m[:, 0]

    if modifiedc:
        vmod_p = v5_p + vh_p
        vmod_m = v5_m + vh_m
        return {
            "+modifiedC": vmod_p,
            "-modifiedC": vmod_m,
            "modifiedC": np.nanmean(np.stack([vmod_p, vmod_m], axis=0), axis=0),
        }

    return {
        "+5mC": v5_p,
        "-5mC": v5_m,
        "+5hmC": vh_p,
        "-5hmC": vh_m,
        "5mC": np.nanmean(np.stack([v5_p, v5_m], axis=0), axis=0),
        "5hmC": np.nanmean(np.stack([vh_p, vh_m], axis=0), axis=0),
    }


def _is_strand_dir(mdb_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(mdb_dir, "5mC_plus.npy"))
        and os.path.exists(os.path.join(mdb_dir, "5mC_minus.npy"))
    )


def _expand_inputs(inputs: List[str]) -> List[str]:
    out: List[str] = []
    for p in inputs:
        hits = glob.glob(p)
        if hits:
            out.extend(hits)
        else:
            out.append(p)
    # keep only directories
    out = [d for d in out if os.path.isdir(d)]
    return out


def merge_main(args):
    input_dirs = list(args.inputs)
    output_dir = args.output
    modifiedc = args.modifiedc
    print(modifiedc)
    # If single file -> list of dirs
    if len(input_dirs) == 1 and os.path.isfile(input_dirs[0]):
        with open(input_dirs[0], "r") as f:
            input_dirs = [line.strip() for line in f if line.strip()]

    input_dirs = _expand_inputs(input_dirs)
    if not input_dirs:
        raise FileNotFoundError("No valid input mdb directories found.")

    print(f"Merging {len(input_dirs)} mdb databases into {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Parse all samples, track which outputs exist
    samples = []
    all_keys = set()
    n_cpg = None

    for d in input_dirs:
        stranded = _is_strand_dir(d)
        if stranded:
            data = process_ont_strand_split_npys(d, modifiedc=modifiedc)
        else:
            v1, v2 = process_ont_pb_no_strand_npys(d, modifiedc=modifiedc)
            if modifiedc:
                data = {"modifiedC": v1}
            else:
                data = {"5mC": v1}
                if v2 is not None:
                    data["5hmC"] = v2

        # determine n_cpg
        if n_cpg is None:
            # take first vector length
            first_vec = next(iter(data.values()))
            n_cpg = int(first_vec.shape[0])
        else:
            for k, v in data.items():
                if int(v.shape[0]) != n_cpg:
                    raise ValueError(f"n_cpg mismatch in {d} key={k}: {v.shape[0]} vs expected {n_cpg}")

        samples.append({"dir": d, "stranded": stranded, "data": data})
        all_keys.update(data.keys())

    out_keys = sorted(all_keys)
    n_samples = len(samples)

    # Create output memmaps and fill (float32)
    out_mats = {}
    out_paths = {}

    for k in out_keys:
        out_path = os.path.join(output_dir, f"{k}.npy")
        mat = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_cpg, n_samples))
        mat[:] = np.nan
        out_mats[k] = mat
        out_paths[k] = out_path

    # Fill each column
    for col_idx, s in enumerate(samples):
        data = s["data"]
        for k in out_keys:
            if k in data:
                out_mats[k][:, col_idx] = data[k].astype(np.float32, copy=False)

    # Flush to disk
    for k in out_keys:
        out_mats[k].flush()

    # Write column index mapping
    col_index_path = os.path.join(output_dir, "columns.tsv")
    with open(col_index_path, "w") as f:
        f.write("col_idx\tmdb_dir\tlayout\tkeys_present\n")
        for col_idx, s in enumerate(samples):
            layout = "stranded" if s["stranded"] else "unstranded"
            keys_present = ",".join(sorted(s["data"].keys()))
            f.write(f"{col_idx}\t{s['dir']}\t{layout}\t{keys_present}\n")

    # Write outputs manifest
    outputs_path = os.path.join(output_dir, "outputs.tsv")
    with open(outputs_path, "w") as f:
        f.write("name\tpath\tshape\tdtype\n")
        for k in out_keys:
            f.write(f"{k}\t{out_paths[k]}\t{n_cpg}x{n_samples}\tfloat32\n")

    print(f"Wrote column index: {col_index_path}")
    print(f"Wrote outputs manifest: {outputs_path}")
    for k in out_keys:
        print(f"Wrote merged matrix: {out_paths[k]}")
