# mdb

[![PyPI version](https://img.shields.io/pypi/v/methdb?logo=pypi&logoColor=white)](https://pypi.org/project/methdb/)
[![Python versions](https://img.shields.io/pypi/pyversions/methdb?logo=python&logoColor=white)](https://pypi.org/project/methdb/)
[![License](https://img.shields.io/github/license/Fu-Yilei/mdb)](LICENSE)

`mdb` builds, merges, queries, and analyzes CpG-by-sample methylation matrices
from ONT and PacBio BED inputs. It also provides cohort PCA/UMAP, summary
statistics, interactive methylation visualization, ONT ASM PCA, and
strand-bias hotspot detection. Cohorts can also be reduced to predefined or
data-driven CpG groups for downstream analysis.
Individual-CpG and grouped cohorts can be exported as methylation phenotype
matrices for TensorQTL with `mdb export-tensorqtl`.

- PyPI package: `methdb`
- CLI command: `mdb`

## Documentation

Browse the [GitHub Wiki](https://github.com/Fu-Yilei/mdb/wiki) for the complete workflows.

- [Build methylation phenotype matrices for TensorQTL](https://github.com/Fu-Yilei/mdb/wiki/Building-TensorQTL-inputs-with-mdb):
  ONT and PacBio inputs, individual CpGs and groups, sample matching, missing
  values, and subsequent TensorQTL commands.
- [Build an input matrix for SniffCell find](https://github.com/Fu-Yilei/mdb/wiki/Building-SniffCell-find-inputs-with-mdb):
  Reference-matrix preparation and the SniffCell input contract.
- [Run PCA across tissues](https://github.com/Fu-Yilei/mdb/wiki/PCA-of-long-read-methylation-across-tissues):
  Long-read methylation cohorts, tissue metadata, feature selection, and PCA outputs.
- [Detect strand-specific methylation](https://github.com/Fu-Yilei/mdb/wiki/Detecting-strand-specific-methylation-with-mdb):
  Paired strand views, category metadata, hotspot thresholds, and interpretation.

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
- **Grouped cohort store (`.mmdb`)**: the same cohort format, with rows reduced
  to predefined or de novo CpG groups and an additional group-coordinate index.
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

### 3b) Combine a legacy SniffCell Loyfer atlas with grouped ONT tracks

For a workflow starting with your own methylation cohort, see
[Build an input matrix for SniffCell find](https://github.com/Fu-Yilei/mdb/wiki/Building-SniffCell-find-inputs-with-mdb).

The legacy SniffCell matrix is unresolved bisulfite modified-C signal, so it is
stored as `modifiedC`. ONT `5mC` and `5hmC` remain separate MDB views; they are
never summed by this command.

```bash
mdb atlas \
  --legacy-npy atlas/all_celltypes_blocks.npy \
  --legacy-index atlas/all_celltypes_blocks.index.gz \
  --legacy-samples atlas/all_celltypes.txt \
  --legacy-assay modifiedC \
  --cohorts fans_5mc_5hmc.loyfer.mmdb \
  --output combined_loyfer_ont.mmdb
```

Every grouped MDB source row must exactly match a row in the legacy Loyfer
index. The output uses the full legacy row system and stores absent ONT rows as
missing. `atlas_manifest.json` records the sources, views, and alignment counts.

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

### 6) Build a CpG-grouped cohort database

After `mdb merge`, `mdb group` writes a separate database containing
group-level mean methylation values while retaining every assay, haplotype,
strand, and sample column. It never modifies the source `.mmdb`. Only true
multi-CpG groups are materialized by default (`--min-group-cpgs 2`); singleton
CpGs remain available exclusively through their original source-cohort views.
The grouped output is marked with `"representation": "cpg_groups"` in
`manifest.json`.

Use the packaged Loyfer atlas-block index:

```bash
mdb group -i cohort.mmdb -o cohort.loyfer.mmdb --grouping loyfer
```

This loads the packaged, versioned `loyfer_grch38_v1.npz` index containing
7,990,609 atlas blocks. Loyfer intervals are G-anchored, so the stored metadata
fixes a one-base left shift when mapping them onto the C-anchored `.mmdb` CpG
index. This is the block representation from the
[Loyfer methylation atlas](https://www.nature.com/articles/s41586-022-05580-6).

Use the DECODE/Nanopolish definition, which joins adjacent CpGs separated by at
most 10 bp:

```bash
mdb group -i cohort.mmdb -o cohort.decode.mmdb --grouping decode
```

This loads the packaged, versioned `decode_grch38_10bp_v1.npz` index containing
22,179,326 autosomal CpG units. Its parameters are fixed: adjacent CpGs at most
10 bp apart belong to the same unit. This applies the published
[DECODE/Nanopolish CpG-unit rule](https://www.nature.com/articles/s41588-024-01851-2)
to the complete GRCh38 autosomal CpG index; it does not attempt to recreate the
paper's cohort-specific Nanopolish and quality-control exclusions. There is no
runtime DECODE distance option.

The packaged definitions are immutable inputs to named modes. Set
`MDB_GROUP_INDEX_DIR` or use `--predefined-index-dir` only to relocate the same
versioned files. The file method, version, assembly, parameters, source, and
SHA-256 are written into `grouping_summary.json`, and the mapped group index is
copied into the output database.

Or learn groups from the cohort. Adjacent CpGs are joined when their
cross-sample methylation profiles meet the correlation threshold and genomic
distance limit:

```bash
mdb group \
  -i cohort.mmdb \
  -o cohort.denovo.mmdb \
  --grouping denovo \
  --denovo-min-correlation 0.8 \
  --denovo-max-gap-bp 200 \
  --denovo-min-shared-samples 3 \
  --threads 4
```

The de novo method is adjacent-correlation segmentation. It uses the requested
source view (`5mC__combined__combined` by default), computes Pearson correlation
for every neighboring CpG pair using samples observed at both positions, and
links a pair when correlation is at least 0.8, distance is at most 200 bp, and
at least three samples are shared. Connected runs of passing adjacent links are
the groups. A missing/constant/under-supported pair creates a boundary. The
learned boundaries are then reused to aggregate every view in the cohort. This
is deterministic for a given input and parameter set, but it is a greedy local
segmentation rather than a statistical change-point model. `--threads` scans
chromosomes concurrently during de novo discovery and aggregates independent
assay/haplotype/strand views concurrently when writing the grouped store.

The ordinary `index.npz` indexes reduced group rows at their GRCh38 start
coordinates, so existing cohort readers continue to work. Each grouped store
also includes `groups.npz` and `grouping_summary.json`. The additional index maps
every group row to its GRCh38 start/end, source CpG row interval, CpG count, and
predefined-reference interval. Add `--write-group-table` only when the redundant,
human-readable `groups.tsv.gz` is useful; omitting it avoids serializing millions
of text rows during routine builds.

The summary records the method, parameters, citation/provenance, aggregation
rule, achieved reduction, omitted singleton count, and unchanged source cohort.
Group values are arithmetic means of observed CpG beta values and require at
least `--min-observed-cpgs` observations per sample.

Grouped outputs retain the normal cohort-store kind and backend, so they can be
passed directly to `mdb pca`, `mdb stats`, `mdb viz`, and other cohort readers.
Point and range queries use each group's GRCh38 start in `index.npz`; use
`groups.npz` for the complete group interval and source-CpG mapping, and query
the unchanged source cohort when individual-CpG resolution is required.

As a reference benchmark, grouping a 29,152,456-CpG by 294-sample Zarr cohort
containing six 5mC/5hmC views produced the following results. These runs also
wrote `groups.tsv.gz`; elapsed time depends on storage and CPU performance.

| Mode | Threads | Retained groups | Reduction | Total time |
| --- | ---: | ---: | ---: | ---: |
| Loyfer | 4 | 4,252,542 | 6.86x | 25m 02s |
| de novo | 20 | 2,582,857 | 11.29x | 14m 35s |

For the de novo run, group discovery took 108 seconds; the remaining time was
spent aggregating and compressing the six output views.

### 6b) Export methylation phenotypes for TensorQTL

See the [complete TensorQTL workflow](https://github.com/Fu-Yilei/mdb/wiki/Building-TensorQTL-inputs-with-mdb)
for a step-by-step guide from input BED files to nominal cis-QTL mapping.

The command is available in `methdb` 0.0.7 and later. Install and verify:

```bash
python -m pip install --upgrade 'methdb>=0.0.7'
mdb export-tensorqtl --help
```

Export one assay/haplotype/strand view from an individual-CpG or grouped-CpG
cohort as a [TensorQTL phenotype BED](https://github.com/broadinstitute/tensorqtl#input-formats):

```bash
mdb export-tensorqtl \
  -i cohort.mmdb \
  -o methylation.bed.gz \
  --assay 5mC \
  --haplotype combined \
  --strand combined
```

Use a grouped database such as `cohort.decode.mmdb` as the input to export
group-level phenotypes instead. The representation is detected automatically.
Both NPY and Zarr backends are streamed in batches; no TensorQTL installation
or GPU is required for export.

For a regional export, add `--region chr20:30000000-30100000` (zero-based,
half-open). This selects individual CpGs within the interval or groups that
overlap it; group coordinates and original group IDs are retained in full.
The receipt records rows considered and rows outside the selected region.

The tab-delimited output has this header, followed by one phenotype per row:

```text
#chr\tstart\tend\tphenotype_id\tSAMPLE_1\tSAMPLE_2
```

Methylation is the phenotype, expressed as beta values in `[0, 1]`, with no
normalization or transformation applied. Individual CpGs use a one-base BED
interval at the C position and a `chrom:position` ID with a one-based position.
Grouped rows retain their existing mean methylation, full zero-based half-open
CpG-group bounds from `groups.npz`, and IDs matching `groups.tsv.gz`
(`grouping_1`, `grouping_2`, etc.). TensorQTL uses those intervals when defining
cis windows; grouped phenotypes are not anchored to a midpoint. Export each of
`5mC`, `5hmC`, and `modifiedC` separately as needed.

By default, rows with any missing selected-sample values and constant rows are
removed. To retain rows observed in at least 80% of selected samples and
explicitly fill their remaining missing cells with the observed row mean:

```bash
mdb export-tensorqtl -i cohort.mmdb -o methylation.imputed.bed.gz \
  --min-frac-present 0.8 --missing mean
```

All-missing and constant rows are always excluded. Valid zero methylation is
preserved. Read-depth thresholds were applied when creating the sample bundles;
the merged cohort does not retain per-cell coverage for additional depth filtering.

Use `--sample-file samples.txt` to select and order source samples, one ID per
line. Use `--sample-map sample_map.tsv` to rename columns for genotype matching;
the TSV requires `sample_id` and `genotype_id` headers. Samples omitted from the
mapping keep their source IDs. Final column names must be unique, and at least
two samples must be selected. Genotypes must contain matching sample IDs;
covariate columns must use the phenotype sample IDs and ordering. Genomic
coordinates must use the same reference build and
chromosome naming as the genotype input. The exporter does not read or validate
those external files.

The output can end in `.bed` or `.bed.gz`. A companion
`<output>.export.json` records the view, sample ordering, coordinates, missing
policy, imputed-cell count, and numbers of exported and filtered rows. Existing
outputs require `--overwrite`. The source database is read-only.

Python API:

```python
import mdb

summary = mdb.export_tensorqtl(
    "cohort.mmdb", "methylation.bed.gz", assay="5mC",
    samples=["SAMPLE_2", "SAMPLE_1"],
)
```

Load the exported phenotypes in TensorQTL, then supply the matching genotypes
and optional covariates for methylation QTL mapping:

```python
import tensorqtl

phenotype_df, phenotype_pos_df = tensorqtl.read_phenotype_bed("methylation.bed.gz")
```

For example, export the same selected samples from individual-CpG and grouped
cohorts, then run nominal cis-QTL mapping in an environment containing TensorQTL:

```bash
mdb export-tensorqtl -i cohort.mmdb -o cpgs.bed.gz \
  --region chr20:30000000-30100000 \
  --sample-file samples.txt --sample-map sample_map.tsv
mdb export-tensorqtl -i cohort.denovo.mmdb -o groups.bed.gz \
  --region chr20:30000000-30100000 \
  --sample-file samples.txt --sample-map sample_map.tsv

mkdir -p results/cpg results/group
python -m tensorqtl /path/to/genotypes cpgs.bed.gz cpg_qtl \
  --mode cis_nominal --window 100000 --output_dir results/cpg
python -m tensorqtl /path/to/genotypes groups.bed.gz group_qtl \
  --mode cis_nominal --window 100000 --output_dir results/group
```

Here `/path/to/genotypes` is a PLINK prefix, without the file extension, with
matching sample IDs and chromosome names. Add `--covariates covariates.tsv`
when supplying analysis covariates. Nominal outputs are written to
`<prefix>.cis_qtl_pairs.<chromosome>.parquet`; these commands do not perform
permutation testing or FDR adjustment.

`--region` selects methylation phenotypes to export. TensorQTL's separate
`--window` controls the maximum cis distance in base pairs, defaulting to
1,000,000. For an individual CpG and SNP on the same chromosome, a pair is
eligible when their distance is at most that value. Describing this as CpGs
around each SNP or SNPs around each CpG gives the same pairs; TensorQTL
internally searches variants around each phenotype. For a group, its full
interval is expanded by the chosen
distance on both sides, with boundary positions included. TensorQTL converts
the BED start to one-based coordinates before matching genotype positions.

The optional integration tests use TensorQTL's actual BED reader, nominal
cis-QTL mapping, and permutations on synthetic donors, comparing regression
results with SciPy for both individual/grouped CpGs and NPY/Zarr backends:

```bash
python -m pytest -q tests
```

The integration test module documents an isolated TensorQTL 1.0.10 environment
and saves validation receipts with its outputs. Those tests are skipped when
TensorQTL is unavailable; the exporter itself has no TensorQTL dependency.

An opt-in SMaHT real-data spot check is available separately. It reads bounded
regions from the two source cohorts, selects one methylation sample per donor,
and uses exact called SNPs shared across matching donor VCFs. In the same
TensorQTL environment, with PLINK2 available, run:

```bash
python tests/run_real_tensorqtl_spotcheck.py \
  --cpg-db /path/to/ont_cohort_zarr.mmdb \
  --group-db /path/to/ont_cohort_zarr.denovo.mmdb \
  --genotype-dir /path/to/DAC_phased_SNPs \
  --outdir .cache/real_tensorqtl_spotcheck \
  --region chr20:30000000-30100000 --window 100000
```

The runner executes both export commands and both subsequent TensorQTL commands
with real data. Separate, clearly labeled simulated genotypes check exact
cis-window boundaries. It saves the shell commands, raw outputs, source-value
checks, and validation receipt. This small, selected sample is a software
compatibility check; nominal association outputs have no biological interpretation.
TensorQTL 1.0.10 can emit `slope_se=NaN` for exactly zero correlation while
reporting `slope=0` and `p=1`. The runner records such pairs and their independent
SciPy standard errors without changing the raw TensorQTL output.

PacBio validation covers both `mod_score` count-mode BEDs and the original
`discretized_mod_score` schema through create, merge, group, and export, with
NPY and Zarr backends. An additional opt-in real-data check uses both PacBio
cohorts and builds de novo groups from a small local copy of the count-based
region across all source sample columns:

```bash
python tests/run_pacbio_tensorqtl_spotcheck.py \
  --count-db /path/to/pacbio_by_count_cohort_zarr.mmdb \
  --score-db /path/to/pacbio_cohort_zarr.mmdb \
  --genotype-dir /path/to/DAC_phased_SNPs \
  --outdir .cache/pacbio_tensorqtl_spotcheck \
  --region chr20:30000000-30100000 --window 100000
```

It checks source values, raw missing/zero encodings, locally computed group
means, sample mapping, and loaded alternate-allele dosages before running actual
TensorQTL commands for individual count-mode CpGs, grouped count-mode CpGs,
and individual original-score CpGs. Each run is checked against the specified
cis window and an independent regression on TensorQTL's actual input precision.
Nearly perfect correlations in these small donor subsets can expose
TensorQTL's float32 precision limits in p-values and standard errors. The
runner reports those comparisons separately, preserving the raw outputs;
they are not claimed to match double-precision regression results.

### 7) Run PCA on cohort view

See [PCA across tissues](https://github.com/Fu-Yilei/mdb/wiki/PCA-of-long-read-methylation-across-tissues)
for cohort preparation, tissue metadata, and the meaning of the PCA settings.

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
coloring. With metadata coloring enabled, `--pairplot-hue` selects the initial
color field for both `pca.html` and the pairplot; all usable metadata columns
remain available through the interactive dropdown.

For example, a grouped cohort is analyzed exactly like the original cohort:

```bash
mdb pca \
  -i cohort.denovo.mmdb \
  -o cohort_denovo_pca \
  -m metadata.tsv \
  --assay 5mC \
  --haplotype combined \
  --strand combined \
  --frac_cpgs 0.001 \
  --n_pcs 10 \
  --pairplot-mode metadata \
  --pairplot-hue tissue_name
```

Add `--umap` to calculate a UMAP embedding and write `umap.html`.

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

### 8) Summarize cohort stats

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

### 9) Build binned methylation profile HTML

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

### 10) Plot a locus-focused methylation profile from a cohort

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

### 11) Run PCA on ONT ASM segments

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

### 12) Detect strand-biased methylation hotspots

See [Detect strand-specific methylation](https://github.com/Fu-Yilei/mdb/wiki/Detecting-strand-specific-methylation-with-mdb)
for stranded input preparation, thresholds, and interpretation.

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
