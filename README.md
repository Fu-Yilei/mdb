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
  --plot_style studio \
  --plot_style_variants \
  --outlier_detect \
  --outlier_alpha 0.999 \
  --outlier_n_pcs 10
```

When `--outlier_detect` is enabled, `pca` also writes:

- `outlier_report.tsv` (all samples with outlier columns),
- `outliers_only.tsv` (flagged samples),
- `pca_with_outliers_marked.html`,
- `pca_no_outliers.html`,
- `pca_pairplot.html`,
- `pca_pairplot_no_outliers.html`,
- `pca_pairplot_no_outliers.png`.

When `--plot_style_variants` is enabled, extra style comparison HTML files are written (for example `pca_studio.html`, `pca_sunrise.html`, `pca_paper.html` depending on selected primary style).

### 7) Summarize cohort stats

```bash
mdb stats \
  -i cohort.mmdb \
  -o cohort_stats \
  -m metadata.tsv \
  --assay all \
  --haplotype all \
  --strand all \
  --plot_style studio
```

`stats` writes:

- `sample_stats.tsv`: one row per sample per cohort track with observed CpG counts and fractions.
- `track_stats.tsv`: per-track summary across assay / haplotype / strand subclasses.
- `metadata_group_stats.tsv`: per-track grouped summaries for plotted metadata columns.
- `cpg_count_scatter.html`: interactive sample-rank scatter with PCA-style color dropdown.
- `cpg_count_by_track.html`: interactive box/point plot of observed CpGs stratified by metadata across tracks.

### 8) Build binned methylation profile HTML

```bash
mdb viz \
  -i /path/to/cohort.mmdb \
  -o cohort_viz \
  -m metadata.tsv \
  --assay 5mC,5hmC \
  --haplotype combined \
  --strand combined \
  --bin-length 100000
```

`viz` also accepts a directory of `.smdb` bundles, bundle globs, or a `.txt` manifest of bundle paths. It writes:

- `methylation_viz.html`: interactive heatmap HTML for switching tracks, chromosomes, metadata-group rows, and selected sample rows.
- `binned_profiles.npz`: compressed sample-by-bin matrices for each selected track.
- `sample_metadata_aligned.tsv`: aligned metadata used by the HTML.
- `tissue_name_group_profiles.tsv.gz`: precomputed tissue-level mean profiles when `tissue_name` metadata is available.

### 9) Plot a locus-focused methylation profile from a cohort

```bash
mdb plot \
  -i cohort.mmdb \
  -o locus_plot \
  -r chr15:80150000-80200000 \
  -m metadata.tsv \
  --sample-id sampleA,sampleB \
  --assay 5mC,5hmC \
  --combine-tracks sum \
  --window-size 20 \
  --color-by tissue_name
```

`plot` is separate from `viz`: it focuses on one genomic interval from a cohort store and writes an interactive line-profile HTML with controls for:

- selected track view(s) within the requested assay / haplotype / strand filters
- sample-line mode or grouped-mean mode using the chosen metadata color field
- browser-side sliding-window smoothing
- metadata-driven recoloring when aligned metadata is provided

It writes:

- `methylation_plot.html`: interactive locus-focused profile viewer with track, color, grouping, and window controls.
- `region_profiles.npz`: compressed CpG-level matrices for the plotted region and selected tracks.
- `smoothed_profiles.tsv.gz`: smoothed profile coordinates and values using the initial command-line window size.
- `sample_metadata_aligned.tsv`: aligned metadata used by the HTML controls.
- `plot_manifest.json`: run parameters and output paths.

## Important Notes

- `create --reader` currently defaults to `scan` and the active create path uses scan-based reading.
- `merge` and `append` require sample bundles created by current `mdb create` (manifest-based `.smdb` layout).
- `pca` now supports both current cohort stores (`.mmdb`) and legacy flat merged `.npy` folders.
- PCA color categories are sorted before plotting so colors stay stable across full vs no-outlier comparisons.

## License

MIT (`LICENSE`).
