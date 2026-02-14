# mdb
`mdb` is a command line toolkit for building CpG-by-sample methylation matrices from ONT (modkit) and PacBio (pb-CpG-tools) outputs.

PyPI distribution name: `methdb`  
CLI command: `mdb`

## What It Does
`mdb` has four core subcommands:

1. `mdb index`
Create a CpG index from a reference FASTA.

2. `mdb create`
Map methylation BED records onto indexed CpG rows and write per-sample `.npy` matrices.

3. `mdb merge`
Merge per-sample `.npy` matrices into a population-level CpG-by-sample matrix.

4. `mdb pca`
Run streaming PCA (and optional UMAP) on merged matrices.

## Installation
Install from source:

```bash
pip install .
```

After install:

```bash
mdb --help
mdb --version
```

Install from PyPI (once published):

```bash
pip install methdb
```

## Tool And Data Requirements
Input methylation formats are expected from:

- ONT: `modkit pileup` (v0.6+)
- PacBio: `aligned_bam_to_cpg_scores` from pb-CpG-tools (v3.0+)

Reference requirements:

- FASTA must be indexed (`.fai`) for upstream callers.
- `mdb index` currently expects chromosome names like `chr1..chr22` (and optionally `chrX`, `chrY` with `--sex`).

## Command Overview
General usage:

```text
mdb [-h] [-v] {index,create,merge,pca} ...
```

### 1) Index CpGs
```bash
mdb index \
  -r GRCh38_no_alt.fa \
  -o GRCh38_no_alt.cpg_index.npz
```

Options:

- `-r, --reference`: reference FASTA
- `-o, --output`: output `.npz`
- `-s, --sex`: include `chrX` and `chrY` (default is autosomes only)

Index output `.npz` contains:

- `chroms`
- `chrom_offsets`
- `chrom_id`
- `pos0` (0-based C positions of CpGs)

### 2) Create Per-Sample Matrices
#### ONT input (`-p ont`)
Unstranded input example:

```bash
mdb create \
  -p ont \
  -n GRCh38_no_alt.cpg_index.npz \
  -b ont/combined.bed.gz \
  -o sample_ont.mdb \
  -c 5
```

Stranded input example:

```bash
mdb create \
  -p ont \
  -n GRCh38_no_alt.cpg_index.npz \
  -b ont/combined_stranded.bed.gz \
  -o sample_ont_stranded.mdb \
  -c 5
```

ONT input discovery behavior:

- If `-b` is a file: treated as one sample (`combined`).
- If `-b` is a directory: looks for `combined.bed(.gz)`, `hp1.bed(.gz)`, `hp2.bed(.gz)`.

ONT output behavior:

- If no strand split detected: writes `5mC.npy`, `5hmC.npy`
- If strand split detected (`+` and `-` present): writes
  - `5mC_plus.npy`
  - `5mC_minus.npy`
  - `5hmC_plus.npy`
  - `5hmC_minus.npy`
- Always writes `bed_map.txt`

Notes:

- Values are stored as fractions `[0,1]` in `float32`.
- `column_10` is used as coverage; `column_11` as percent modified.
- For minus strand in split mode, coordinate is shifted by `-1` to map to canonical CpG C.

Recommended modkit command:

```bash
modkit pileup in.bam out.bed.gz \
  --bgzf \
  --modified-bases C:m C:h \
  --cpg \
  --reference GRCh38_no_alt.fa
```

Add `--combine-strands` for unstranded output.

#### PacBio input (`-p pacbio`)
Prefix-based example (recommended):

```bash
mdb create \
  -p pacbio \
  -n GRCh38_no_alt.cpg_index.npz \
  -b pb/HG002 \
  -o sample_pb.mdb \
  -c 5
```

This discovers available files among:

- `pb/HG002.combined.bed.gz`
- `pb/HG002.hap1.bed.gz`
- `pb/HG002.hap2.bed.gz`

PacBio output behavior:

- Writes `5mC.npy` (no 5hmC for PacBio)
- Writes `bed_map.txt`

PacBio parser expects:

- header line beginning with `#chrom`
- columns including `begin`, `cov`, `discretized_mod_score`

### 3) Merge Samples
Merge unstranded ONT + PacBio:

```bash
mdb merge \
  -i sample1_ont.mdb sample2_pb.mdb \
  -o merged.mmdb
```

Merge with ONT `5mC+5hmC -> modifiedC` aggregation:

```bash
mdb merge \
  -i sample1_ont.mdb sample2_pb.mdb \
  -m \
  -o merged_modifiedC.mmdb
```

Merge outputs:

- one or more merged `.npy` matrices
- `columns.tsv` (sample-column mapping)
- `outputs.tsv` (matrix manifest)

Accepted `-i/--inputs`:

- list of mdb directories
- a single text file listing mdb directories (one per line)
- glob patterns are expanded

### 4) PCA/UMAP
Run PCA:

```bash
mdb pca \
  -i merged.mmdb \
  -o pca_out
```

Run PCA + UMAP:

```bash
mdb pca \
  -i merged.mmdb \
  -o pca_out \
  -u
```

Matrix selection priority in PCA:

1. `modifiedC.npy`
2. `5mC.npy`
3. `5hmC.npy`

PCA outputs:

- `embedding.tsv`
- `params.json`
- `pca_umap.log`
- `pca.html`
- `pca_pairplot.png`
- `umap.html` (if `-u`)
- optional PNG exports when Plotly image backend is available

## End-To-End Minimal Example
```bash
# 1) Index
mdb index -r GRCh38_no_alt.fa -o GRCh38.cpg.npz

# 2) Create ONT
mdb create -p ont -n GRCh38.cpg.npz -b ont/combined.bed.gz -o ont_sample.mdb -c 5

# 3) Create PacBio
mdb create -p pacbio -n GRCh38.cpg.npz -b pb/HG002 -o pb_sample.mdb -c 5

# 4) Merge
mdb merge -i ont_sample.mdb pb_sample.mdb -m -o cohort.mmdb

# 5) PCA
mdb pca -i cohort.mmdb -o cohort.pca -u
```

## Current Behavior Notes And Limitations
- `merge` currently treats each input mdb directory as one sample and uses only the first column (`[:,0]`) from each per-sample matrix file.
- If your `create` output contains multiple columns (for example combined/hap1/hap2 in one directory), only the first column is used by `merge`.
- `pca` imports additional libraries (`pandas`, `scikit-learn`, `umap-learn`, `seaborn`, `matplotlib`) that are not listed in `install_requires`; install them separately if needed.
- For metadata in `mdb pca`, best compatibility is to provide an `id` column matching sample IDs (directory basenames in `columns.tsv`) or ensure row order exactly matches sample order.

## License
MIT License. See `LICENSE`.
