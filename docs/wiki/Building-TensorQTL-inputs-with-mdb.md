# How to use mdb to build an input matrix for TensorQTL

```bash
mdb export-tensorqtl -i cohorts/ont.mmdb -o phenotypes/cpgs.bed.gz
mdb export-tensorqtl -i cohorts/ont.denovo.mmdb -o phenotypes/groups.bed.gz
```

These commands export **methylation as the phenotype**: one row per individual
CpG or CpG group, with samples in columns and beta values between 0 and 1.
The grouped export uses existing group means. It does not aggregate again.
Genotypes and covariates are separate inputs to TensorQTL.

## 1. Install the checkout containing the exporter

From a local mdb checkout containing this command:

```bash
python -m pip install -e .
mdb export-tensorqtl --help
mkdir -p bundles cohorts phenotypes results/cpg results/group
```

The package is named `methdb`; its command is `mdb`. An older PyPI installation
may lack the exporter: verify its help after installing this checkout.
Export requires neither TensorQTL nor a GPU. Mapping uses a separate TensorQTL
installation; compatibility was tested with TensorQTL 1.0.10 on CPU. Follow
the [official installation instructions](https://github.com/broadinstitute/tensorqtl#install)
and check `python -m tensorqtl --help` in that environment. The
[integration test](https://github.com/Fu-Yilei/mdb/blob/main/tests/test_tensorqtl_integration.py)
documents the compatible validation environment and its prerequisites.
Separate environments avoid a NumPy dependency conflict between methdb
(`numpy>=2.2`) and pandas-plink 2.3.2 (`numpy<2`). Validation used pandas-plink 2.2.9.

## 2. Build separate ONT and PacBio cohorts

Skip this step if your individual-CpG `.mmdb` databases already exist. Start
from a GRCh38 FASTA matching the methylation calls and genotypes:

```bash
mdb index -r reference/GRCh38.fa -o reference/GRCh38.cpgs.npz --sex

for n in 01 02 03; do
  mdb create -p ont -n reference/GRCh38.cpgs.npz \
    -b input/ont/DONOR_${n}.combined.bed.gz \
    -o bundles/ONT_${n}.smdb --sample-id ONT_${n} -c 5
  mdb create -p pacbio -n reference/GRCh38.cpgs.npz \
    -b input/pacbio/DONOR_${n}.cpg_scores \
    -o bundles/PB_${n}.smdb --sample-id PB_${n} -c 5
done

mdb merge -i bundles/ONT_01.smdb bundles/ONT_02.smdb bundles/ONT_03.smdb \
  -o cohorts/ont.mmdb --cohort-backend zarr
mdb merge -i bundles/PB_01.smdb bundles/PB_02.smdb bundles/PB_03.smdb \
  -o cohorts/pacbio.mmdb --cohort-backend zarr
```

The three samples illustrate file layout, not an adequate association cohort.
Use appropriate biological independence, sample size, and study covariates;
technical replicates or multiple tissues from one donor require an appropriate
study design. Keep ONT and PacBio cohorts separate for these examples.

ONT inputs are modkit methylation BEDs. Use strand-combined inputs for the
`--strand combined` examples below. A BED containing separate strands creates
`plus`/`minus` views: use `--strand plus` or `--strand minus` when exporting.

The PacBio prefix above resolves to `.combined.bed.gz`, `.hap1.bed.gz`, and
`.hap2.bed.gz` files. A specific BED path can also be supplied. Both
pb-cpg-tools schemas are supported: count-mode `mod_score` and model-mode
`discretized_mod_score`. Their percentages are converted to beta fractions.
PacBio provides `5mC`; it does not supply a separate `5hmC` view.

The creation threshold is `-c 5` or `--min_coverage 5`. The exporter cannot
apply a new read-depth threshold because cohort values do not retain per-cell
coverage measurements.

## 3. Optionally build a grouped database

Choose a grouping method; each writes a separate cohort:

```bash
mdb group -i cohorts/ont.mmdb -o cohorts/ont.loyfer.mmdb --grouping loyfer
mdb group -i cohorts/ont.mmdb -o cohorts/ont.decode.mmdb --grouping decode
mdb group -i cohorts/ont.mmdb -o cohorts/ont.denovo.mmdb --grouping denovo
```

The same commands accept `cohorts/pacbio.mmdb`. Loyfer and DECODE use packaged
GRCh38 definitions. De novo grouping learns adjacent-CpG correlations from
`5mC`, combined haplotype/strand by default; for stranded ONT, add
`--denovo-strand plus` or `minus`. All existing views retain their identities.
Group values are arithmetic means of observed CpG beta values. Singleton groups
are omitted; retain the original cohort for individual-CpG analysis.

## 4. Match phenotype columns to genotype samples

Create `samples.txt` with source IDs in the desired order:

```text
ONT_01
ONT_02
ONT_03
```

Create `sample_map.tsv` with actual tab separators:

```text
sample_id	genotype_id
ONT_01	DONOR_01
ONT_02	DONOR_02
ONT_03	DONOR_03
```

`--sample-file` selects and orders samples. `--sample-map` only renames them;
it does not select them. Unmapped IDs remain unchanged. For PacBio, use
`PB_01`, `PB_02`, and `PB_03` as source IDs in separate selection/mapping files.
Final IDs must be unique and match genotype IDs. The exporter requires at
least two selected samples for variable phenotypes; that is a format safeguard,
not a recommendation for QTL study size.

## 5. Export individual and grouped phenotypes

```bash
mdb export-tensorqtl -i cohorts/ont.mmdb -o phenotypes/cpgs.bed.gz \
  --assay 5mC --haplotype combined --strand combined \
  --sample-file samples.txt --sample-map sample_map.tsv \
  --region chr20:30000000-30100000

mdb export-tensorqtl -i cohorts/ont.denovo.mmdb -o phenotypes/groups.bed.gz \
  --assay 5mC --haplotype combined --strand combined \
  --sample-file samples.txt --sample-map sample_map.tsv \
  --region chr20:30000000-30100000
```

Export one assay/haplotype/strand view per file. Use available `hp1`/`hp2`
views for phased methylation. ONT `5mC` and `5hmC` remain separate;
`modifiedC` is available in a separate cohort explicitly built with
`mdb merge --modifiedc`.
The exporter never sums assays or combines strands.

Output starts with these tab-separated columns:

```text
#chr	start	end	phenotype_id	DONOR_01	DONOR_02	DONOR_03
```

CpGs use `[pos0,pos0+1)` and IDs such as `chr20:30000001`, whose position is
one-based. Groups retain full stored intervals and stable IDs such as
`denovo_42`. `--region` uses **zero-based, half-open** coordinates: CpGs must
lie inside it; groups need only overlap it. Groups are not clipped or moved
to midpoints. Omit `--region` to export the whole cohort.

Default filtering keeps complete, nonconstant rows. All-missing and constant
rows are always removed; true zero methylation remains valid. Filtering uses
selected samples. To require 80% observed values and explicitly impute remaining
cells with each row's observed mean, use:

```bash
mdb export-tensorqtl -i cohorts/ont.mmdb -o phenotypes/cpgs.imputed.bed.gz \
  --sample-file samples.txt --sample-map sample_map.tsv \
  --region chr20:30000000-30100000 \
  --min-frac-present 0.8 --missing mean
```

Imputation follows the presence filter; `--missing mean` alone leaves the
default 100% requirement in place. No normalization or transformation is applied.
Likewise, lowering `--min-frac-present` while keeping `--missing drop` still
removes rows with any missing cells.

| Export option | Default or requirement |
| --- | --- |
| `-i`, `--input` | Required individual/grouped cohort |
| `-o`, `--output` | Required `.bed` or `.bed.gz` |
| `--assay` | `5mC` |
| `--haplotype`, `--strand` | `combined` |
| `--sample-file` | All stored samples, stored order |
| `--sample-map` | Source IDs unchanged |
| `--region` | All rows |
| `--min-frac-present` | `1.0` |
| `--missing` | `drop`; alternative `mean` |
| `--batch-rows` | `8192` |
| `--overwrite` | Off |

Both NPY and Zarr cohorts stream in batches. The source remains read-only.
`<output>.export.json` records sample order, view, coordinates, region, missing
policy, imputed cells, and disjoint filtering counts. Existing outputs require
`--overwrite`. The Python API accepts the same export policies, with source IDs
and their mapping supplied directly:

```python
import mdb

summary = mdb.export_tensorqtl(
    "cohorts/ont.mmdb", "phenotypes/api.bed.gz",
    samples=["ONT_01", "ONT_02", "ONT_03"],
    sample_map={"ONT_01": "DONOR_01", "ONT_02": "DONOR_02", "ONT_03": "DONOR_03"},
    region="chr20:30000000-30100000",
)
```

## 6. Check inputs and run TensorQTL

In the TensorQTL environment, read back the exported matrix and check covariates
if supplying them:

```python
import numpy as np
import pandas as pd
import tensorqtl

phenotypes, positions = tensorqtl.read_phenotype_bed("phenotypes/cpgs.bed.gz")
assert not phenotypes.empty
assert phenotypes.index.is_unique and phenotypes.columns.is_unique
assert np.isfinite(phenotypes.to_numpy()).all()
assert ((phenotypes >= 0) & (phenotypes <= 1)).all().all()
```

If supplying `covariates.tsv`, check its sample order in the same Python session:

```python
# Optional covariates.tsv: covariates in rows, samples in columns.
covariates = pd.read_csv("covariates.tsv", sep="\t", index_col=0)
assert covariates.index.is_unique and covariates.columns.is_unique
assert covariates.columns.tolist() == phenotypes.columns.tolist()
assert np.isfinite(covariates.to_numpy(dtype=float)).all()
# TensorQTL's Python API instead takes covariates.T (samples x covariates).
```

MDB does not generate covariates. Use numerical covariates selected for your
analysis. The [TensorQTL input specification](https://github.com/broadinstitute/tensorqtl#input-formats)
describes its phenotype, genotype, and covariate formats.

Supply a PLINK prefix without extensions: `genotypes/cohort` represents
`.bed/.bim/.fam` or `.pgen/.pvar/.psam` files with the required reader dependencies.
PLINK's genotype `.bed` is binary; the exported phenotype `.bed` is text.
Reference build and chromosome names must agree, including any `chr` prefix.

```bash
mkdir -p results/cpg results/group
python -m tensorqtl genotypes/cohort phenotypes/cpgs.bed.gz cpg_qtl \
  --mode cis_nominal --window 100000 --output_dir results/cpg
python -m tensorqtl genotypes/cohort phenotypes/groups.bed.gz group_qtl \
  --mode cis_nominal --window 100000 --output_dir results/group
```

Add `--covariates covariates.tsv` to both commands when appropriate. Nominal
pair outputs are `<prefix>.cis_qtl_pairs.<chromosome>.parquet`; these commands
do not perform permutation testing or FDR adjustment.

`--region` selects exported phenotypes. TensorQTL's `--window` defines cis flanks,
defaulting to 1,000,000 bp. Genotype inputs must include those flanks. For
point CpGs and SNPs, searching CpGs around each SNP or SNPs around each CpG
identifies the same eligible pairs. For a group, eligibility spans inclusive
one-based SNP positions `[BED start+1-window, BED end+window]`, as implemented
in the [TensorQTL genotype reader](https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/genotypeio.py).

If no phenotypes remain, the exporter raises an error and publishes no new BED
or receipt. Check the selected region, missing values, and whether methylation
is constant across selected samples. Successful exports record filtering counts
in `<output>.export.json`.
If no cis pairs are found, check genotype chromosome names, coverage of the
chosen region and flanks, and the presence of variable genotypes.

## Validation and numerical limits

Run synthetic exporter and optional TensorQTL integration checks from the repository:

```bash
python -m pytest -q tests/test_tensorqtl_export.py \
  tests/test_tensorqtl_export_regions.py tests/test_tensorqtl_integration.py \
  tests/test_tensorqtl_pacbio_workflow.py
```

Opt-in [ONT](https://github.com/Fu-Yilei/mdb/blob/main/tests/run_real_tensorqtl_spotcheck.py)
and [PacBio](https://github.com/Fu-Yilei/mdb/blob/main/tests/run_pacbio_tensorqtl_spotcheck.py)
runners document bounded real-data checks and require explicitly supplied local
inputs. Such small compatibility subsets are not association studies.

TensorQTL 1.0.10 can return `slope_se=NaN` at exactly zero correlation while
reporting `slope=0` and `p=1`. This is an upstream numerical caveat, not a failed
export, and does not mean the methylation values are all zero.
Near-perfect fits can also produce unstable p-values and standard errors through
float32 correlation rounding. Validation flags these cases and retains raw
outputs; it does not replace standard errors with zero or claim agreement where
precision fails. Assess study design, normalization, covariates, and multiple
testing separately before interpreting methylation associations.
