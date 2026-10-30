# Detecting strand-specific methylation with mdb

```bash
mdb strand -i cohorts/stranded.mmdb -o results/strand_5mC \
  --assay 5mC --haplotype combined \
  --min-paired-frac 0.8 --min-mean-total 0.005 \
  --cluster-gap-bp 1000 --min-cluster-cpgs 3 \
  --top-n-hotspots 500 --workers 4
```

This ranks CpGs where the mean methylation of the reference plus strand differs
from the minus strand, then clusters nearby candidates with the same direction.
The command is `mdb strand`. It requires an individual-CpG cohort with both
`plus` and `minus` views for the chosen assay and haplotype. Use the original
CpG cohort rather than a CpG-grouped database: the current command treats each
indexed row as a CpG and uses its start coordinate.

## 1. Preserve strand information when building the database

Check that the installed command is available:

```bash
mdb strand --help
```

If installing from a checkout, use `python -m pip install -e .` first.
Prepare ONT modkit CpG BEDs with separate `+` and `-` records, without
`--combine-strands`. Keep native BED coordinates and all original columns.
For a reference CpG with C at zero-based position `p`, MDB expects the plus
record at `p` and the minus record at `p+1`, the reference G. During creation,
MDB subtracts one from the minus position so both measurements occupy the same
CpG row. Do not shift minus records yourself before importing them.

Each discovered BED must retain both strand symbols. MDB detects stranded
input from the first discovered BED; a plus-only file is not sufficient.
The directory layout below refers to haplotypes, with both strands inside each
file:

```text
input/SAMPLE_01/combined.bed.gz
input/SAMPLE_01/hp1.bed.gz
input/SAMPLE_01/hp2.bed.gz
```

Only available haplotypes need to be present. `combined` means combined
haplotypes here; it does not mean combined strands. A direct BED path is also
accepted and becomes the combined-haplotype input.

Build a matching reference index, create bundles, and merge them:

```bash
mkdir -p bundles cohorts results
mdb index -r reference/GRCh38.fa -o reference/GRCh38.cpgs.npz --sex

for n in 01 02 03 04; do
  mdb create -p ont -n reference/GRCh38.cpgs.npz \
    -b input/SAMPLE_${n} -o bundles/SAMPLE_${n}.smdb \
    --sample-id SAMPLE_${n} -c 5
done

mdb merge -i bundles/SAMPLE_01.smdb bundles/SAMPLE_02.smdb \
  bundles/SAMPLE_03.smdb bundles/SAMPLE_04.smdb \
  -o cohorts/stranded.mmdb --cohort-backend zarr
```

The same analysis accepts an NPY cohort. Reference build and chromosome names
must match the input calls. `-c 5` (`--min_coverage 5`) filters input BED records
at creation. Strand analysis cannot impose a new read-depth cutoff on the
merged cohort, which does not retain actual per-cell coverage measurements.

## 2. Check the views and choose the comparison

Inspect available views before selecting an assay:

```python
from mdb.storage import available_views

print([view.name() for view in available_views("cohorts/stranded.mmdb")])
```

For the first command, expect `5mC__combined__plus` and
`5mC__combined__minus`, with matching sample columns. ONT can provide separate
`5mC` and `5hmC` views; choose one explicitly. The default assay is **5hmC**.
Use `--haplotype hp1` or `hp2` only when the corresponding paired views exist.
An explicitly built `modifiedC` view can also be selected if present.

Standard PacBio creation produces combined-strand `5mC`, so those cohorts do
not provide the paired views required here. A strand-combined ONT database
also cannot recover the lost strand information: rebuild it from stranded
inputs. A missing track pair produces an error listing available views.

## 3. Understand scores and thresholds

At each CpG, a sample is paired only when both strand beta values are finite.
Missing values are not zero; observed zero methylation is valid. For the
paired samples, MDB calculates:

```text
mean_diff = mean(plus - minus)
mean_total = mean((plus + minus) / 2)
hotspot_score = abs(mean_diff) * sqrt(number_of_paired_samples)
```

Positive `mean_diff` means `plus>minus`; negative means `minus>plus`.
These are beta-fraction differences: `0.10` is ten percentage points.
`mean_abs_diff` is the absolute cohort mean difference, not the mean of each
sample's absolute difference. Opposing sample effects can therefore cancel.

| Option | Default and meaning |
| --- | --- |
| `--assay` | `5hmC` |
| `--haplotype` | `combined` |
| `--min-paired-frac` | `0.8`; at least `max(1, ceil(fraction × sample count))` paired samples |
| `--min-mean-total` | `0.005`; minimum mean strand methylation, equivalent to 0.5% |
| `--cluster-gap-bp` | `1000`; maximum separation between neighboring retained candidates |
| `--min-cluster-cpgs` | `1`; minimum retained CpGs per reported cluster |
| `--top-n-hotspots` | `500`; maximum reported clusters per comparison |
| `--batch-rows` | `65536`; streamed rows per block |
| `--workers` | `1`; parallel scan workers |
| `--plot-style` | `studio`; alternatives `sunrise`, `paper` |

`--plot-style-variants` additionally writes other styles; `--verbose` enables
detailed logging. The command scans the whole cohort index and has no
`--region` or sample-selection option.

The score is a ranking statistic, without a p-value, FDR, minimum effect-size
test, or requirement that every sample agree in direction. Zero-difference
rows can remain with score zero; their tie direction is labeled `plus>minus`.
Apply and report a suitable effect-size criterion when reviewing results.
Candidate pruning occurs before clustering, so output is a ranked shortlist,
not an exhaustive segmentation of all eligible CpGs.

## 4. Compare metadata categories

Supply a TSV/CSV with an `id` column matching stored sample IDs and a category
column. For example, save `metadata.tsv` with actual tab separators:

```text
id	tissue
SAMPLE_01	Brain
SAMPLE_02	Brain
SAMPLE_03	Blood
SAMPLE_04	Blood
```

Then run:

```bash
mdb strand -i cohorts/stranded.mmdb -o results/strand_by_tissue \
  --assay 5mC --metadata metadata.tsv --group-by tissue
```

This performs global calling and separate calling within each category.
The paired fraction is evaluated against that category's sample count.
Check `strand.log` and `params.json`: an unavailable grouping column is warned
about and category calling is skipped. Category results are descriptive;
they do not implement a between-category significance test.

## 5. Read outputs and inspect individual loci

Main files include:

- `hotspots_global.tsv` and `.bed`: ranked clusters, direction, representative
  CpG effect/support, and cluster size. BED intervals are zero-based, half-open.
- `per_sample_metrics.tsv.gz`: genome-wide paired-CpG counts, strand means,
  mean difference, and balance index. These use all paired CpGs before hotspot
  candidate filtering.
- `hotspot_sample_profiles.tsv.gz`: paired values at each cluster's
  representative CpG, when clusters exist; these are not whole-cluster averages.
- Category hotspot TSV/BED files, category summaries, and representative
  category profiles when metadata grouping is available.
- Interactive `strand_bias.html`, hotspot ranking reports, `params.json`, and
  `strand.log`. Additional images depend on the available rendering backend.

The interval file contains `chrom`, `start`, `end`, label, maximum score, and
direction. Its direction strings are `plus>minus` or `minus>plus`, rather than
the standard BED6 strand codes `+` and `-`; use the TSV for complete statistics.

Query both strands for one sample:

```bash
mdb query -i cohorts/stranded.mmdb --sample-id SAMPLE_01 \
  --assay 5mC --strand plus --locus chr20:30000000
mdb query -i cohorts/stranded.mmdb --sample-id SAMPLE_01 \
  --assay 5mC --strand minus --locus chr20:30000000

mdb plot -i cohorts/stranded.mmdb -o results/locus \
  --region chr20:30000000-30001000 --assay 5mC \
  --haplotype combined --strand plus,minus --combine-tracks none
```

Query positions are zero-based. Query and plot region endpoints are inclusive,
unlike BED endpoints. Displayed representative coordinates are one-based;
convert them before querying. The plot can overlay both strands and display
sample or metadata-group profiles.

## Interpretation

Strand imbalance compares aggregated methylation measurements from opposite
reference strands. It does not pair complementary strands from the same DNA
molecule and does not establish hemimethylation. Check read-level or duplex
evidence, alignment/calling artifacts, coverage, and sample composition before
making that claim. Strand bias is also distinct from allele-specific methylation:
haplotype contrasts compare `hp1` with `hp2`; this command compares plus with
minus within the selected haplotype.

See the [strand implementation](https://github.com/Fu-Yilei/mdb/blob/main/mdb/strand.py),
[ONT input parser](https://github.com/Fu-Yilei/mdb/blob/main/mdb/ont_bed_parsing.py),
and [CLI definitions](https://github.com/Fu-Yilei/mdb/blob/main/mdb/parse_args.py)
for the current scoring and coordinate conventions.
