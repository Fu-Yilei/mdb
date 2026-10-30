# PCA of long-read methylation across tissues

```bash
mdb pca -i cohorts/ont.mmdb -o results/pca_ont -m metadata/ont.tsv \
  --assay 5mC --haplotype combined --strand combined \
  --n_pcs 3 --pairplot_pcs_n 3 --frac_cpgs 0.1 \
  --min_frac_present 0.8 --batch_rows 65536 --seed 1 \
  --pairplot-mode metadata --pairplot-hue tissue_name \
  --pairplot_diag_kind hist
```

Each point represents a **sample**, colored by tissue. The input is a merged
methylation cohort with CpGs in rows and samples in columns. PCA compares
methylation profiles across genomic features; it does not treat CpGs as the
points displayed in the sample plot. Use separate ONT and PacBio analyses to
avoid interpreting a technology difference as a tissue difference.

## 1. Prepare matched reference and sample bundles

Install the checkout or release you intend to use, then inspect its CLI:

```bash
python -m pip install -e .
mdb pca --help
mkdir -p bundles cohorts metadata results
mdb index -r reference/GRCh38.fa -o reference/GRCh38.cpgs.npz --sex
```

Use the same reference build and CpG index for every sample in a cohort.
`--sex` includes chrX/chrY; omit it for an autosomal index.
Chromosome names must match the methylation BEDs. The following six-sample
examples illustrate two donors measured in three tissues; replace their paths
and sample lists with your study's inputs.

For strand-combined ONT modkit BEDs:

```bash
for donor in D01 D02; do
  for tissue in Brain Blood Liver; do
    mdb create -p ont -n reference/GRCh38.cpgs.npz \
      -b input/ont/${donor}_${tissue}.combined.bed.gz \
      -o bundles/ONT_${donor}_${tissue}.smdb \
      --sample-id ONT_${donor}_${tissue} -c 5
  done
done
mdb merge -i bundles/ONT_*.smdb -o cohorts/ont.mmdb --cohort-backend zarr
```

For PacBio pb-cpg-tools BED prefixes:

```bash
for donor in D01 D02; do
  for tissue in Brain Blood Liver; do
    mdb create -p pacbio -n reference/GRCh38.cpgs.npz \
      -b input/pacbio/${donor}_${tissue}.cpg_scores \
      -o bundles/PB_${donor}_${tissue}.smdb \
      --sample-id PB_${donor}_${tissue} -c 5
  done
done
mdb merge -i bundles/PB_*.smdb -o cohorts/pacbio.mmdb --cohort-backend zarr
```

Each PacBio prefix looks for available `.combined.bed.gz`, `.hap1.bed.gz`, and
`.hap2.bed.gz` inputs; all three are not required. A specific BED can also be supplied. Both count-mode
`mod_score` and model-mode `discretized_mod_score` are supported. Native
PacBio methylation supplies `5mC`; ONT may also supply separate `5hmC` views.
For ONT BEDs with separate strand records, creation produces `plus`/`minus`
views: select `--strand plus` or `minus` instead of `combined`.

`-c 5` (`--min_coverage 5`) filters low-coverage calls during creation.
PCA cannot apply a new depth threshold from the cohort. `.smdb` is an
individual sample bundle; `mdb pca` needs the merged `.mmdb`. Both Zarr and
NPY cohorts are supported.

## 2. Supply explicit sample-to-tissue metadata

Save `metadata/ont.tsv` with actual tab separators:

```text
id	tissue_name	donor_id	technology	batch
ONT_D01_Brain	Brain	D01	ONT	B1
ONT_D01_Blood	Blood	D01	ONT	B1
ONT_D01_Liver	Liver	D01	ONT	B1
ONT_D02_Brain	Brain	D02	ONT	B2
ONT_D02_Blood	Blood	D02	ONT	B2
ONT_D02_Liver	Liver	D02	ONT	B2
```

For `metadata/pacbio.tsv`, use `PB_` sample IDs and `PacBio` technology.
`id` must exactly match the stored `--sample-id`; `sample_id` is also accepted
as the key. `tissue_name` is the column selected by the example's hue flag.
Other annotation columns are optional. Avoid reusing output fields such as
`platform`, `path`, `source_path`, or `PC1` as metadata column names.

With unique keys, metadata joins by ID and preserves cohort order, so metadata
row order does not matter. It **does not select samples or reorder the PCA**.
Check IDs are unique and cover every sample in the chosen cohort view. Duplicate
metadata IDs can multiply output rows; unmatched IDs retain samples with missing
annotations. A missing metadata path is silently ignored. Always supply an ID
column rather than relying on positional matching.

## 3. Run the selected view and understand preprocessing

Run the opening command for ONT. Run the same settings for PacBio:

```bash
mdb pca -i cohorts/pacbio.mmdb -o results/pca_pacbio -m metadata/pacbio.tsv \
  --assay 5mC --haplotype combined --strand combined \
  --n_pcs 3 --pairplot_pcs_n 3 --frac_cpgs 0.1 \
  --min_frac_present 0.8 --batch_rows 65536 --seed 1 \
  --pairplot-mode metadata --pairplot-hue tissue_name \
  --pairplot_diag_kind hist
```

The explicit hue initializes both scatter and pairplot to tissue colors.
Interactive dropdowns retain other usable metadata columns. Coloring changes
annotations, not the fitted coordinates or sample weights.

Default settings are `5mC`, combined haplotype/strand, 10 PCs, five pairplot
PCs, 10% of eligible features, 80% presence, seed 1, and 400,000 rows per batch.
UMAP and outlier detection are off. The example requests three PCs because it
has six samples. Choose at least two PCs, no more than the sample/selected-feature
counts, and keep pairplot dimensions within the computed PCs. Every incremental
fit batch must contain at least that many features.

Current preprocessing is specific:

1. Retain genomic rows observed in at least `--min_frac_present` of samples
   in the selected view, then randomly select `--frac_cpgs` using `--seed`.
   This is not selection of the most variable CpGs. `--frac_cpgs 1.0` uses
   all eligible rows. Constant genomic rows are not explicitly removed.
2. For each sample, calculate its observed mean and standard deviation across
   selected rows, then center and scale that sample's column. Remaining missing
   entries become standardized zero, equivalent to that sample's mean imputation.
   This is **per-sample scaling**, not per-CpG scaling or CpG-row-mean imputation.
   Samples with at most one observed feature use fallback mean 0 and scale 1.
3. Fit IncrementalPCA on the features-by-samples matrix. Plotted sample
   coordinates are `components_.T × singular_values_`. They are not ordinary
   `transform()` scores from a samples-by-features fit.

Stored values decode to beta fractions in `[0,1]`; observed zero methylation
remains valid. There is no beta-to-M transformation, covariate regression, tissue
balancing, or automatic sample exclusion. Inspect observation counts: severely
sparse samples can remain. Compare runs using the same feature selection,
settings, and software versions.

## 4. Optionally use grouped or region-level features

```bash
mdb group -i cohorts/ont.mmdb -o cohorts/ont.decode.mmdb --grouping decode
mdb pca -i cohorts/ont.decode.mmdb -o results/pca_ont_decode \
  -m metadata/ont.tsv --assay 5mC --haplotype combined --strand combined \
  --n_pcs 3 --pairplot_pcs_n 3 --frac_cpgs 0.1 \
  --pairplot-mode metadata --pairplot-hue tissue_name --pairplot_diag_kind hist
```

`mdb group` reduces genomic features, not samples: each stored group mean
becomes one feature. `loyfer` and `denovo` are alternative grouping methods.
Singleton CpGs remain in the original cohort. `mdb pca` has no `--grouping`
option; tissue coloring does not average samples within tissues.

For an original individual-CpG cohort, `--cpg-bed regions.bed` restricts
features to positions inside zero-based, half-open BED intervals. Adding
`--cpg-bed-agg` instead calculates observed per-sample CpG means for each region,
writes `region_avg.tsv`, and uses all region rows before presence filtering.
On grouped cohorts, BED selection uses group starts, not full interval overlap;
region averaging would average stored group means. Use the original CpG cohort
for these BED-based examples.

## 5. Inspect outputs before interpreting separation

Standard outputs are:

- `embedding.tsv`: one row per sample, PC coordinates, joined annotations,
  and `n_obs_cpgs_for_fit` before imputation.
- `params.json`: selected view, feature count, settings, explained variance,
  and optional-analysis status.
- `pca.html` and `pca_pairplot.html`: interactive figures with color dropdowns.
- `pca_pairplot.png`: static pairplot; `pca_umap.log`: execution details.

Scatter PNGs require a working Plotly image engine. HTML figures use
CDN-hosted Plotly JavaScript. Parameters do not save feature IDs, scaling
vectors, or a reusable projection model.

Add `--umap` to embed the PCA coordinates and write `umap.html`. Optional
`--outlier_detect` writes outlier reports and marked/subset plots; removing
outliers from those figures does not refit PCA. A biologically distinct tissue
can be distant without being a failed sample.

Check tissue labels and missingness in `embedding.tsv`, then inspect donor,
batch, and technology coloring. Repeated tissues or technical replicates from
one donor are not independent donors. Tissue mixtures and cell composition,
assay identity, coverage, and platform/model effects can drive separation.
PCA is descriptive: a cluster does not establish a tissue-specific methylation
mechanism or a differential-methylation result.

Implementation details are in
[pca.py](https://github.com/Fu-Yilei/mdb/blob/main/mdb/pca.py) and
[CLI definitions](https://github.com/Fu-Yilei/mdb/blob/main/mdb/parse_args.py).
