#!/usr/bin/env python
import argparse
import sys, os
from mdb.__init__ import __version__ as version

def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="mdb",
        description="DNA methylation database builder for quick population-level analysis.",
        epilog=f"Version {version}",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"mdb {version}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Subcommand: index
    index_parser = subparsers.add_parser("index", help="Index all CpG locations on the reference genome")
    index_parser.add_argument("-r", "--reference", required=True, help="Reference FASTA file")
    index_parser.add_argument("-o", "--output", required=True, help="Output NPZ file prefix for indexed CpG locations")
    index_parser.add_argument("-s", "--sex", default=False, action="store_true", help="Include sex chromosomes in the index, default=False")

    # Subcommand: create
    create_parser = subparsers.add_parser("create", help="Create single sample-level methylation database")
    create_parser.add_argument("-p", "--platform", required=True, help="Input platform: ont|pacbio")
    create_parser.add_argument("-n", "--npz", required=True, help="Reference NPZ file from mdb index")
    create_parser.add_argument("-b", "--bed", required=True, help="Input BED file with DMR indications from Modkit>=0.6.0 or Pb-CpG-tools>=3.0.0")
    create_parser.add_argument("-o", "--output", required=True, help="Output .mdb file for the single sample-level methylation database")
    create_parser.add_argument( "-c", "--min_coverage", type=int, default=5, help="Minimum coverage threshold, default=5" )
    
    # Subcommand: merge
    merge_parser = subparsers.add_parser("merge", help="mdb databases from multiple samples into a single database: COMBINE STRAND and HAPLOTYPE")
    merge_parser.add_argument( "-i", "--inputs", nargs='+', required=True, help="Input directories to merge or a text file with list of .mdb files to merge")
    merge_parser.add_argument("-m", "--modifiedc", help="Aggregrate ONT 5mC and 5hmC as modifiedC, for merging with PacBio", action="store_true")
    merge_parser.add_argument( "-o", "--output", required=True, help="Output directory for merged database")
    
    # Subcommand: stats
    # Subcommand: PCA
    # Subcommand: strand
    

    # Subcommand: query
    query_parser = subparsers.add_parser("query", help="Query the population-level methylation database")

    if len(argv) == 0:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(argv)
    return args