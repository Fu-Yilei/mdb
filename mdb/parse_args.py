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
    merge_parser.add_argument( "-o", "--output", required=True, help="Output directory (suggest: output.mmdb) for merged database")
    
    # Subcommand: PCA
    pca_parser = subparsers.add_parser("pca", help="Perform PCA on the merged mdb database")
    pca_parser.add_argument("-i", "--input", required=True, help="Input merged mdb database for PCA")
    pca_parser.add_argument("-o", "--outdir", required=True, help="Output directory for PCA results")
    pca_parser.add_argument("-m", "--metadata",  help="Metadata TSV file with sample information, must contain 'sample' column matching merged database sample names")
    pca_parser.add_argument("-pn", "--pairplot_pcs_n", type=int, default=5, help="Top n PCs to include in pairplot, default=5")
    pca_parser.add_argument("--frac_cpgs", type=float, default=0.1, help="Fraction of eligible CpGs to sample for PCA fit, default=0.1")
    pca_parser.add_argument("--n_pcs", type=int, default=10, help="Number of PCs, default=10")
    pca_parser.add_argument("--min_frac_present", type=float, default=0.8, help="Min fraction of samples with non-NaN at CpG, default=0.8")
    pca_parser.add_argument("--batch_rows", type=int, default=400_000, help="Rows per streaming block, increase for larger RAM usage, default=400,000")
    pca_parser.add_argument("--seed", type=int, default=1, help="Random seed, default=1")
    pca_parser.add_argument("-u", "--umap", action="store_true", help="Perform UMAP embedding")
    pca_parser.add_argument("--umap_neighbors", type=int, default=15, help="UMAP number of neighbors, default=15")
    pca_parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP minimum distance, default=0.1")
    pca_parser.add_argument("--umap_metric", type=str, default="euclidean" ,help="UMAP metric, default=euclidean")
    pca_parser.add_argument("--verbose", action="store_true", help="More stderr logging (DEBUG)")
    
    # Subcommand: strand
    # strand_parser = subparsers.add_parser("strand", help="Perform strand-specific analysis of merged mdb database")


    # Subcommand: query
    # query_parser = subparsers.add_parser("query", help="Query the population-level methylation database")

    if len(argv) == 0:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(argv)
    return args