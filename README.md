# mdb

[![PyPI version](https://img.shields.io/pypi/v/methdb?logo=pypi&logoColor=white)](https://pypi.org/project/methdb/)
[![Python versions](https://img.shields.io/pypi/pyversions/methdb?logo=python&logoColor=white)](https://pypi.org/project/methdb/)
[![License](https://img.shields.io/github/license/Fu-Yilei/mdb)](LICENSE)

`mdb` builds and queries CpG-by-sample methylation matrices from ONT and PacBio BED inputs.

- PyPI package: `methdb`
- CLI command: `mdb`

## Install

```bash
pip install .
```

Verify:

```bash
mdb --help
mdb --version
```

## Core Concepts

- **Sample bundle (`.smdb`)**: one sample, multiple track views (assay/haplotype/strand).
- **Cohort store (`.mmdb`)**: merged sample bundles for population-scale queries.
- **Backends**:
  - `zarr` (default, compressed, block-aligned merge writes)
  - `npy` (optional compatibility backend)

## Quick Start

### 1) Build CpG index

```bash
mdb index -r GRCh38_no_alt.fa -o GRCh38.cpg_index.npz
```

Include `chrX/chrY`:

```bash
mdb index -r GRCh38_no_alt.fa -o GRCh38.cpg_index.npz --sex
```

### 2) Create sample bundle

ONT (modkit output file or directory):

```bash
mdb create \
  -p ont \
  -n GRCh38.cpg_index.npz \
  -b /path/to/ont_input \
  -o sample_ont.smdb \
  -c 5 \
  --sample-id SAMPLE_ONT
```

PacBio (prefix or directory):

```bash
mdb create \
  -p pacbio \
  -n GRCh38.cpg_index.npz \
  -b /path/to/pacbio_prefix \
  -o sample_pb.smdb \
  -c 5 \
  --sample-id SAMPLE_PB
```

### 3) Merge sample bundles into a cohort

Default backend (`zarr`):

```bash
mdb merge \
  -i sample_ont.smdb sample_pb.smdb \
  -o cohort.mmdb \
  --workers 2 \
  --block-size 64 \
  --zarr-row-chunk 65536 \
  --zarr-codec zstd \
  --zarr-clevel 5 \
  --zarr-shuffle bitshuffle \
  --zarr-codec-threads 4
```

NPY backend (explicit):

```bash
mdb merge \
  -i sample_ont.smdb sample_pb.smdb \
  -o cohort_npy.mmdb \
  --cohort-backend npy \
  --workers 2 \
  --block-size 64
```

Build modifiedC view (`5mC + 5hmC` where available):

```bash
mdb merge -i sample_ont.smdb sample_pb.smdb -o cohort_modifiedc.mmdb -m
```

### 4) Append new samples to existing cohort

```bash
mdb append \
  -c cohort.mmdb \
  -i new_sample1.smdb new_sample2.smdb
```

### 5) Query values

Point query:

```bash
mdb query \
  -i cohort.mmdb \
  --sample-id SAMPLE_PB \
  --assay 5mC \
  --haplotype combined \
  --strand combined \
  --locus chr1:10469
```

Range query:

```bash
mdb query \
  -i cohort.mmdb \
  --sample-id SAMPLE_PB \
  --assay 5mC \
  --haplotype combined \
  --strand combined \
  --region chr1:10469-12000
```

### 6) Run PCA on cohort view

```bash
mdb pca \
  -i cohort.mmdb \
  -o cohort_pca \
  --assay 5mC \
  --haplotype combined \
  --strand combined \
  --n_pcs 10 \
  --frac_cpgs 0.1 \
  --outlier_detect \
  --outlier_alpha 0.999 \
  --outlier_n_pcs 10
```

When `--outlier_detect` is enabled, `pca` also writes:

- `outlier_report.tsv` (all samples with outlier columns),
- `outliers_only.tsv` (flagged samples),
- `pca_with_outliers_marked.html`,
- `pca_no_outliers.html`,
- `pca_pairplot_no_outliers.png`.

## Important Notes

- `create --reader` currently defaults to `scan` and the active create path uses scan-based reading.
- `merge` and `append` require sample bundles created by current `mdb create` (manifest-based `.smdb` layout).
- `pca` now supports both current cohort stores (`.mmdb`) and legacy flat merged `.npy` folders.
- PCA color categories are sorted before plotting so colors stay stable across full vs no-outlier comparisons.

## License

MIT (`LICENSE`).
