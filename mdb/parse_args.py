#!/usr/bin/env python

import argparse
import sys

from mdb.__init__ import __version__ as version


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="mdb",
        description="Population-scale DNA methylation storage and analysis toolkit.",
        epilog=f"Version {version}",
    )
    parser.add_argument("-v", "--version", action="version", version=f"mdb {version}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index all CpG locations on the reference genome")
    index_parser.add_argument("-r", "--reference", required=True, help="Reference FASTA file")
    index_parser.add_argument("-o", "--output", required=True, help="Output NPZ file for indexed CpG locations")
    index_parser.add_argument("-s", "--sex", default=False, action="store_true", help="Include chrX and chrY in the index")

    create_parser = subparsers.add_parser("create", help="Create a sample bundle directory (.smdb)")
    create_parser.add_argument("-p", "--platform", required=True, help="Input platform: ont|pacbio")
    create_parser.add_argument("-n", "--npz", required=True, help="Reference NPZ file from mdb index")
    create_parser.add_argument("-b", "--bed", required=True, help="Input BED file, BED prefix, or BED directory")
    create_parser.add_argument("-o", "--output", required=True, help="Output sample bundle directory path")
    create_parser.add_argument("-c", "--min_coverage", type=int, default=5, help="Minimum coverage threshold, default=5")
    create_parser.add_argument("--sample-id", help="Explicit sample id to store in the output bundle")
    create_parser.add_argument(
        "--reader",
        choices=("auto", "scan", "tabix"),
        default="scan",
        help="BED reader implementation, default=scan (current create path uses scan)",
    )
    create_parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Per-file workers for tabix-backed chromosome fetch, default=1",
    )

    merge_parser = subparsers.add_parser("merge", help="Merge sample bundles into a cohort store directory (.mmdb)")
    merge_parser.add_argument("-i", "--inputs", nargs="+", required=True, help="Input sample bundle directories, globs, or a .txt manifest")
    merge_parser.add_argument("-m", "--modifiedc", action="store_true", help="Build modifiedC cohort views by combining 5mC and 5hmC")
    merge_parser.add_argument("-o", "--output", required=True, help="Output cohort store directory path")
    merge_parser.add_argument("-w", "--workers", type=int, default=1, help="Parallel workers for per-view merge assembly, default=1")
    merge_parser.add_argument("--block-size", type=int, default=64, help="Samples per cohort block shard, default=64")
    merge_parser.add_argument("--cohort-backend", choices=("npy", "zarr"), default="zarr", help="Cohort storage backend, default=zarr")
    merge_parser.add_argument("--zarr-row-chunk", type=int, default=65536, help="Row chunk size for zarr cohort backend, default=65536")
    merge_parser.add_argument("--zarr-codec", choices=("zstd",), default="zstd", help="Compression codec for zarr backend, default=zstd")
    merge_parser.add_argument("--zarr-clevel", type=int, default=5, help="Compression level for zarr backend, default=5")
    merge_parser.add_argument(
        "--zarr-shuffle",
        choices=("none", "shuffle", "bitshuffle"),
        default="bitshuffle",
        help="Shuffle mode for zarr backend compression, default=bitshuffle",
    )
    merge_parser.add_argument("--zarr-codec-threads", type=int, default=4, help="Codec threads for zarr backend, default=4")

    append_parser = subparsers.add_parser("append", help="Append sample bundles into an existing cohort store")
    append_parser.add_argument("-c", "--cohort", required=True, help="Existing cohort store path to update in place")
    append_parser.add_argument("-i", "--inputs", nargs="+", required=True, help="Input sample bundle directories, globs, or a .txt manifest")
    append_parser.add_argument("-m", "--modifiedc", action="store_true", help="Append into modifiedC views by combining 5mC and 5hmC")
    append_parser.add_argument(
        "--cohort-backend",
        choices=("npy", "zarr"),
        default=None,
        help="Cohort storage backend override; default=existing cohort backend",
    )
    append_parser.add_argument("--zarr-row-chunk", type=int, default=65536, help="Row chunk size for zarr cohort backend, default=65536")
    append_parser.add_argument("--zarr-codec", choices=("zstd",), default="zstd", help="Compression codec for zarr backend, default=zstd")
    append_parser.add_argument("--zarr-clevel", type=int, default=5, help="Compression level for zarr backend, default=5")
    append_parser.add_argument(
        "--zarr-shuffle",
        choices=("none", "shuffle", "bitshuffle"),
        default="bitshuffle",
        help="Shuffle mode for zarr backend compression, default=bitshuffle",
    )
    append_parser.add_argument("--zarr-codec-threads", type=int, default=4, help="Codec threads for zarr backend, default=4")

    pca_parser = subparsers.add_parser("pca", help="PCA on cohort store view or legacy flat merged .npy folder")
    pca_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input cohort store (.mmdb) or legacy merged folder with columns.tsv/txt and *.npy",
    )
    pca_parser.add_argument("-o", "--outdir", required=True, help="Output directory for PCA results")
    pca_parser.add_argument("-m", "--metadata", help="Metadata TSV/CSV with sample_id or id column")
    pca_parser.add_argument("--assay", default="5mC", help="Track assay to analyze, default=5mC")
    pca_parser.add_argument("--haplotype", default="combined", help="Track haplotype to analyze, default=combined")
    pca_parser.add_argument("--strand", default="combined", help="Track strand to analyze, default=combined")
    pca_parser.add_argument("-pn", "--pairplot_pcs_n", type=int, default=5, help="Top n PCs to include in pairplot, default=5")
    pca_parser.add_argument("--frac_cpgs", type=float, default=0.1, help="Fraction of eligible CpGs to sample for PCA fit, default=0.1")
    pca_parser.add_argument("--n_pcs", type=int, default=10, help="Number of PCs to compute, default=10")
    pca_parser.add_argument("--min_frac_present", type=float, default=0.8, help="Min fraction of samples with non-missing data at a CpG")
    pca_parser.add_argument("--batch_rows", type=int, default=400_000, help="Rows per streaming block, default=400,000")
    pca_parser.add_argument("--seed", type=int, default=1, help="Random seed, default=1")
    pca_parser.add_argument("-u", "--umap", action="store_true", help="Perform UMAP embedding")
    pca_parser.add_argument("--umap_neighbors", type=int, default=15, help="UMAP number of neighbors, default=15")
    pca_parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP minimum distance, default=0.1")
    pca_parser.add_argument("--umap_metric", type=str, default="euclidean", help="UMAP metric, default=euclidean")
    pca_parser.add_argument(
        "--pairplot-mode",
        choices=("metadata", "sample", "none"),
        help="Pairplot coloring mode: metadata, sample, or none; default auto",
    )
    pca_parser.add_argument("--pairplot-hue", help="Preferred metadata column for pairplot hue when --pairplot-mode metadata")
    pca_parser.add_argument("--pairplot_diag_kind", choices=("kde", "hist"), default="kde", help="Pairplot diagonal kind, default=kde")
    pca_parser.add_argument("--pairplot_corner", action="store_true", help="Use lower triangle only for pairplot")
    pca_parser.add_argument(
        "--outlier_detect",
        action="store_true",
        help="Enable outlier detection and emit outlier/no-outlier reports and plots",
    )
    pca_parser.add_argument(
        "--outlier_alpha",
        type=float,
        default=0.999,
        help="Chi-square alpha cutoff for Mahalanobis outlier detection (0<alpha<1), default=0.999",
    )
    pca_parser.add_argument(
        "--outlier_n_pcs",
        type=int,
        default=10,
        help="Number of leading PCs used for outlier detection, default=10",
    )
    pca_parser.add_argument("--verbose", action="store_true", help="More stderr logging (DEBUG)")

    query_parser = subparsers.add_parser("query", help="Query a sample bundle or cohort store at one CpG locus")
    query_parser.add_argument("-i", "--input", required=True, help="Input sample bundle directory or cohort store")
    query_parser.add_argument("--sample-id", help="Required for cohort queries; ignored for sample bundle queries")
    query_parser.add_argument("--assay", default="5mC", help="Track assay to query, default=5mC")
    query_parser.add_argument("--haplotype", default="combined", help="Track haplotype to query, default=combined")
    query_parser.add_argument("--strand", default="combined", help="Track strand to query, default=combined")
    query_target = query_parser.add_mutually_exclusive_group(required=True)
    query_target.add_argument("-l", "--locus", help="Genomic locus as chrom:pos0")
    query_target.add_argument("-r", "--region", help="Genomic interval as chrom:start-end using 0-based inclusive coordinates")

    if len(argv) == 0:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args(argv)
