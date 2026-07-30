# mdb

[![PyPI version](https://img.shields.io/pypi/v/methdb?logo=pypi&logoColor=white)](https://pypi.org/project/methdb/)
[![Python versions](https://img.shields.io/pypi/pyversions/methdb?logo=python&logoColor=white)](https://pypi.org/project/methdb/)
[![License](https://img.shields.io/github/license/Fu-Yilei/mdb)](LICENSE)

`mdb` builds, merges, queries, and analyzes CpG-by-sample methylation matrices
from ONT and PacBio BED inputs. It also provides cohort PCA/UMAP, summary
statistics, interactive methylation visualization, ONT ASM PCA, and
strand-bias hotspot detection.

- PyPI package: `methdb`
- CLI command: `mdb`

## Install

```bash
pip install methdb
```

To install from a local checkout instead:

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

ONT (modkit output file, directory, or prefix):

```bash
mdb create \
  -p ont \
  -n GRCh38.cpg_index.npz \
  -b /path/to/ont_input \
  -o sample_ont.smdb \
  -c 5 \
  --sample-id SAMPLE_ONT
```

Accepted ONT layouts are:

- a BED/BED.GZ file, treated as the `combined` track;
- a directory containing any of `combined.bed(.gz)`, `hp1.bed(.gz)`, or
  `hp2.bed(.gz)`; or
- a prefix resolving to `<prefix>.combined.bed.gz`,
  `<prefix>.hap1.bed.gz`, and/or `<prefix>.hap2.bed.gz`.

When a modkit BED contains both `+` and `-` records, `mdb create`
automatically builds separate strand tracks.

PacBio (prefix or a specific BED file):

```bash
mdb create \
  -p pacbio \
  -n GRCh38.cpg_index.npz \
  -b /path/to/pacbio_prefix \
  -o sample_pb.smdb \
  -c 5 \
  --sample-id SAMPLE_PB
```

A PacBio prefix resolves to `<prefix>.combined.bed(.gz)`,
`<prefix>.hap1.bed(.gz)`, and `<prefix>.hap2.bed(.gz)`. Pointing directly to
a BED/BED.GZ file creates a `combined` track. Prefix or direct-file input is
recommended. Do not rely on PacBio directory discovery in the current
implementation; pass the prefix or a specific BED path instead.

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
- `pca_pairplot_no_outliers.html`,
- `pca_pairplot_no_outliers.png`.

The no-outlier plots require at least one inlier, and the no-outlier pairplot
requires at least two. Plotly PNG exports are written only when a compatible
image engine is available.

When `--plot_style_variants` is enabled, extra style comparison HTML files are written (for example `pca_studio.html`, `pca_sunrise.html`, `pca_paper.html` depending on selected primary style).

Standard PCA outputs include:

- `embedding.tsv`,
- `params.json`,
- `pca_umap.log`,
- `pca.html`,
- `pca_pairplot.html`,
- `pca_pairplot.png`.

Add `-m metadata.tsv` to make aligned metadata columns available for plot
coloring. Add `--umap` to calculate a UMAP embedding and write `umap.html`.

Restrict PCA to CpGs overlapping BED intervals:

```bash
mdb pca \
  -i cohort.mmdb \
  -o cohort_pca_regions \
  --cpg-bed regions.bed
```

Alternatively, average all CpGs within each BED interval and run PCA on the
resulting region-by-sample matrix:

```bash
mdb pca \
  -i cohort.mmdb \
  -o cohort_pca_region_means \
  --cpg-bed regions.bed \
  --cpg-bed-agg
```

The aggregation mode also writes `region_avg.tsv`. Both BED modes require a
current cohort store (`.mmdb`) because legacy flat NPY folders do not contain
genomic positions.

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
- `frac_cpg_scatter.html`: corresponding sample-rank scatter for the observed CpG fraction.
- `cpg_count_by_track.html`: interactive box/point plot of observed CpGs stratified by metadata across tracks.
- `frac_cpg_by_track.html`: corresponding box/point plot for the observed CpG fraction.
- `params.json` and `stats.log`: run parameters and detailed logging.

`metadata_group_stats.tsv` is written only when the aligned metadata provides
usable grouping columns. PNG versions of the plots are written when Plotly PNG
export is available.

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
- `viz_manifest.json`: run parameters, resolved inputs, selected tracks, and output paths.
- `viz.log`: detailed run log.

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
- `plot.log`: detailed run log.

### 10) Run PCA on ONT ASM segments

`asmpca` expects modkit DMR segment BEDs containing fields such as `name`,
`effect_size`, and `cohen_h`:

```bash
mdb asmpca \
  -i sample1.segments.bed.gz sample2.segments.bed.gz \
  -o asm_pca \
  -m metadata.tsv \
  --feature-mode dmr_location \
  --min-region-samples 2 \
  --n-pcs 10
```

Inputs may be BED paths, directories, globs, or a text manifest of paths. By
default, cohort DMR regions are the merged union of input rows whose
`name` is `different`. Supply `--dmr-regions regions.bed` to use an external
region set.

The default `dmr_location` mode uses binary DMR presence as the PCA features.
To project a segment statistic onto the regions instead:

```bash
mdb asmpca \
  -i asm_segments/*.bed.gz \
  -o asm_metric_pca \
  --feature-mode segment_metric \
  --metric effect_size
```

Core outputs are `embedding.tsv`, `dmr_regions_used.bed`, `params.json`,
`pca.html`, `pca_pairplot.html`, `pca_pairplot.png`, and `pca_umap.log`.

### 11) Detect strand-biased methylation hotspots

`strand` requires a cohort store containing matching `plus` and `minus` views
for the requested assay and haplotype:

```bash
mdb strand \
  -i cohort.mmdb \
  -o strand_hotspots \
  -m metadata.tsv \
  --group-by tissue_broad \
  --assay 5hmC \
  --haplotype combined \
  --min-paired-frac 0.8 \
  --min-mean-total 0.005 \
  --cluster-gap-bp 1000 \
  --top-n-hotspots 500 \
  --workers 4
```

The command scores strand imbalance genome-wide, clusters adjacent
same-direction CpGs, and optionally repeats hotspot calling within metadata
groups. Core outputs include:

- `per_sample_metrics.tsv.gz`,
- `hotspots_global.tsv` and `hotspots_global.bed`,
- `hotspot_sample_profiles.tsv.gz` when hotspot profiles are available,
- per-group hotspot TSV/BED and summary files when `--group-by` is used,
- interactive strand-bias and hotspot HTML reports,
- `params.json` and `strand.log`.

## Important Notes

- `create --reader` currently defaults to `scan` and the active create path uses scan-based reading.
- `create --workers` is currently accepted by the CLI but is not used by the active scan-based create path.
- `merge` and `append` require sample bundles created by current `mdb create` (manifest-based `.smdb` layout).
- `pca` now supports both current cohort stores (`.mmdb`) and legacy flat merged `.npy` folders.
- PCA color categories are sorted before plotting so colors stay stable across full vs no-outlier comparisons.

## License

MIT (`LICENSE`).
