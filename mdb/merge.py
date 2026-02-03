#!/usr/bin/env python3
import os

def process_ont_strand_npys(input_npy, output_npy):
    


def merge_main(args):
    input_dirs = args.inputs
    output_dir = args.output
    if len(input_dirs) == 1 and os.path.isfile(input_dirs[0]):
        # Read directories from text file
        with open(input_dirs[0], "r") as f:
            input_dirs = [line.strip() for line in f if line.strip()]
    for mdb_dir in input_dirs:
        if not os.path.isdir(mdb_dir):
            raise ValueError(f"Input {mdb_dir} is not a valid directory.")
    print(f"Merging {len(input_dirs)} mdb databases into {output_dir}")
    os.makedirs(output_dir, exist_ok=True)