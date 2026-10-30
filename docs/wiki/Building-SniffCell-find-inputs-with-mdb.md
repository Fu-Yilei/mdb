# Build a SniffCell find input matrix with MDB

Use MDB to build a reference methylation matrix from annotated long-read samples,
then run [`sniffcell find`](https://github.com/Fu-Yilei/SniffCell) to identify
regions that distinguish the reference groups. The workflow below starts from
your own ONT or PacBio methylation BEDs; an existing legacy atlas matrix is not
required.

Use purified, sorted, or otherwise validated cell populations when the intended
output is a **cell-type** reference. Whole-tissue samples instead define
tissue-associated methylation contrasts, which can reflect cellular composition.
Naming a bulk-brain group `Neuron` does not make its methylation neuron-specific.
Include independent donors and keep their identities in accompanying metadata.

## 1. Check the installed interfaces

```bash
mdb --version
mdb group --help
sniffcell --version
sniffcell find --help
```

This page uses the public SniffCell NumPy interface, checked against
[commit d0486c9](https://github.com/Fu-Yilei/SniffCell/tree/d0486c98403e342c6d842295d28ddf243f1f7b70).
That revision accepts `-n`, `-i`, `-m`, `-cf`, and `-ck`; it does **not** accept
`--mdb`. Some newer development checkouts support direct MDB input, described at
the end of this page. Install the full `sniffcell` package using its
[installation instructions](https://github.com/Fu-Yilei/SniffCell#install-sniffcell).
The pinned [argument parser](https://github.com/Fu-Yilei/SniffCell/blob/d0486c98403e342c6d842295d28ddf243f1f7b70/src/sniffcell/parse_args.py)
and [find implementation](https://github.com/Fu-Yilei/SniffCell/blob/d0486c98403e342c6d842295d28ddf243f1f7b70/src/sniffcell/find/find.py)
define the input contract used below.

Run the Python matrix-conversion step in an environment containing MDB and NumPy.
Run SniffCell in its own environment if the packages have different dependency
requirements. The files connect the two environments.

## 2. Build sample databases and merge the reference cohort

Use one genome assembly and matching chromosome names throughout. The packaged
Loyfer grouping used here is for GRCh38. Build the CpG index once:

```bash
mdb index -r GRCh38_no_alt.fa -o GRCh38.cpg_index.npz
mkdir -p samples
```

For each ONT reference sample, import its modkit output. The sample ID becomes a
matrix column and must match the group JSON later:

```bash
mdb create -p ont -n GRCh38.cpg_index.npz \
  -b methylation/Neuron_D01.combined.bed.gz \
  -o samples/Neuron_D01.smdb -c 5 --sample-id Neuron_D01
```

Repeat for `Neuron_D02`, `Oligodendrocyte_D01`, and
`Oligodendrocyte_D02`, substituting the input, output, and sample ID. For PacBio
5mC BEDs, use `-p pacbio` with the corresponding direct BED path or file prefix.
Keep platform-specific references separate initially so that platform and
cell-type effects can be assessed. Coverage filtering happens at `mdb create`;
the example requires at least five usable calls per CpG.

The main example uses a strand-combined view. The filename suffix `.combined`
identifies a haplotype input, not necessarily a strand-combined pileup. ONT BEDs
containing both `+` and `-` records create `plus`/`minus` views instead. Inspect
the views printed by `mdb merge`; the exporter below accepts an optional strand
argument for analyzing either strand separately.

```bash
mdb merge \
  -i samples/Neuron_D01.smdb samples/Neuron_D02.smdb \
     samples/Oligodendrocyte_D01.smdb samples/Oligodendrocyte_D02.smdb \
  -o reference.mmdb

mdb group -i reference.mmdb -o reference.loyfer.mmdb \
  --grouping loyfer --min-group-cpgs 3 --min-observed-cpgs 3
```

The grouped store retains separate assay, haplotype, and strand views. Values are
arithmetic means of observed CpG beta values, rather than depth-weighted means.
Here each materialized group contains at least three reference CpGs and each
sample needs at least three observed CpGs for a group value. Retain
`reference.mmdb` for individual-CpG analysis.

Use the Loyfer representation for this SniffCell workflow. Its stored
`reference_start` and `reference_end` preserve the G-anchored interval convention
used by the SniffCell atlas. DECODE and de novo grouping are useful MDB features,
but should not be substituted into this conversion without checking downstream
coordinate interpretation. Do not manually shift the exported intervals.

## 3. Convert one MDB view to the public SniffCell input files

SniffCell expects four synchronized inputs:

| File | Contents |
| --- | --- |
| `matrix.npy` | Floating-point beta values in `[0, 1]`; rows are groups, columns are samples; missing values are `NaN` |
| `matrix.index.gz` | Headerless TSV: `chr`, `start`, `end`, `startCpG`, `endCpG`, in matrix row order |
| `samples.txt` | One sample ID per line, in matrix column order |
| `celltypes.json` | Named analyses containing group-to-sample lists |

Save this as `export_sniffcell_matrix.py`. It uses the MDB reader to decode
either the NPY or Zarr backend, preserves zero methylation and missing values,
and streams the output matrix to disk. Its fourth and fifth index fields are
the stored source-CpG row bounds, not invented genomic coordinates.

```python
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from mdb.schema import TrackKey
from mdb.storage import load_view_reader

source, destination, assay = sys.argv[1:4]
strand = sys.argv[4] if len(sys.argv) > 4 else "combined"
out = Path(destination)
key = TrackKey(assay, "combined", strand)
with np.load(Path(source) / "groups.npz", allow_pickle=True) as data:
    groups = {name: data[name].copy() for name in data.files}
if str(groups["method"].item()) not in {"loyfer", "sniffcell_loyfer"}:
    raise ValueError("This workflow requires Loyfer group coordinates")
required = {"chroms", "chrom_offsets", "reference_start", "reference_end",
            "source_row_start", "source_row_end"}
if not required.issubset(groups):
    raise ValueError(f"Missing group metadata: {required - set(groups)}")

reader, columns, _ = load_view_reader(source, key)
try:
    ids = list(columns["sample_id"])
    n_rows = len(groups["reference_start"])
    if reader.n_rows != n_rows or len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("Check row counts and unique sample IDs")
    for name in required - {"chroms", "chrom_offsets"}:
        if len(groups[name]) != n_rows:
            raise ValueError(f"Incorrect row count: {name}")
    out.mkdir(parents=True, exist_ok=False)
    matrix = np.lib.format.open_memmap(
        out / "matrix.npy", mode="w+", dtype=np.float32,
        shape=(n_rows, len(ids)))
    for start in range(0, n_rows, 8192):
        stop = min(start + 8192, n_rows)
        matrix[start:stop] = reader.get_block(slice(start, stop))
    matrix.flush()
    del matrix
    (out / "samples.txt").write_text("\n".join(ids) + "\n")
    offsets = list(map(int, groups["chrom_offsets"])) + [n_rows]
    with gzip.open(out / "matrix.index.gz", "wt") as handle:
        for chrom, lo, hi in zip(groups["chroms"], offsets[:-1], offsets[1:]):
            for row in range(lo, hi):
                fields = [str(chrom)] + [str(int(groups[name][row])) for name in
                    ("reference_start", "reference_end",
                     "source_row_start", "source_row_end")]
                handle.write("\t".join(fields) + "\n")
    (out / "export.json").write_text(json.dumps({
        "source": str(Path(source).resolve()), "view": key.name(),
        "shape": [n_rows, len(ids)], "sample_ids": ids,
        "units": "beta_fraction", "missing": "NaN",
    }, indent=2) + "\n")
finally:
    reader.close()
```

Run it once per assay, using a new output directory:

```bash
python export_sniffcell_matrix.py reference.loyfer.mmdb sniffcell_5mc 5mC
```

For a `plus`-only reference, append `plus` as the fourth argument and use a
separate directory, such as `sniffcell_5mc_plus`. That defines a strand-specific
reference; it does not combine the strands.

Do not export MDB's internal integer block files directly: their missing-value
sentinel and scaling differ from SniffCell's floating-point input. The conversion
streams, but this public SniffCell revision loads the complete NPY matrix into
memory; budget at least `4 × rows × samples` bytes for the float32 matrix plus
substantial processing overhead.

## 4. Define the biological comparison

Create `celltypes.json` with exact IDs from `sniffcell_5mc/samples.txt`:

```json
{
  "brain_reference": {
    "Neuron": ["Neuron_D01", "Neuron_D02"],
    "Oligodendrocyte": ["Oligodendrocyte_D01", "Oligodendrocyte_D02"]
  }
}
```

Use disjoint sample lists and at least two represented groups. SniffCell ignores
declared IDs absent from the matrix, so check the match before running. It
averages available samples within each group; missing samples do not become
unmethylated samples. Rows without a usable contrast may produce no call. Do not
impute values from a different cell type to force coverage.

```bash
python - <<'PY'
import json
from pathlib import Path
ids = Path("sniffcell_5mc/samples.txt").read_text().splitlines()
groups = json.loads(Path("celltypes.json").read_text())["brain_reference"]
assert len(groups) >= 2 and all(groups.values()), "Need two non-empty groups"
declared = [sample for samples in groups.values() for sample in samples]
assert len(declared) == len(set(declared)), "Samples must belong to one group"
assert not (set(declared) - set(ids)), "Some declared samples are absent"
print({group: len(samples) for group, samples in groups.items()})
print("Unused matrix columns:", sorted(set(ids) - set(declared)))
PY
```

## 5. Run sniffcell find and preserve the assay label

```bash
mkdir -p results
sniffcell find \
  -n sniffcell_5mc/matrix.npy -i sniffcell_5mc/matrix.index.gz \
  -m sniffcell_5mc/samples.txt -cf celltypes.json -ck brain_reference \
  --diff_threshold 0.40 --min_rows 1 --min_cpgs 3 --max_gap_bp 2000 \
  -o results/brain.5mC.ctdmr.tsv
```

The explicit thresholds avoid changing behavior with different default versions.
`0.40` is a beta difference of 40 percentage points. `min_rows` counts grouped
matrix rows; `min_cpgs` uses the span of source-CpG row bounds. The latter is not
a per-sample read-depth requirement. Results are effect-threshold reference
contrasts; this command does not perform donor-aware differential methylation
inference or report an FDR.

The public NumPy interface does not record the input assay. Add it explicitly
to preserve the catalog's assay identity:

```bash
python - <<'PY'
import pandas as pd
path = "results/brain.5mC.ctdmr.tsv"
df = pd.read_csv(path, sep="\t")
df["modification"] = "5mC"
df.to_csv(path, sep="\t", index=False)
print(f"{len(df)} ctDMRs; assays: {df['modification'].unique().tolist()}")
PY
```

Inspect `best_group`, `other_group`, `best_dir`, `mean_margin`,
`mean_best_value`, and `mean_rest_value`. An empty table with a header is a valid
result when coverage or contrasts do not meet thresholds. The companion
`results/brain.5mC.ctdmr.tsv.igv.bed` is for display; use the main TSV as the
annotation reference. Keep the export receipt, grouping summary, group JSON,
commands, and versions together.

Repeat into a separate directory for ONT `5hmC` if that view exists, and label
its result `5hmC`. PacBio 5mC and unresolved bisulfite `modifiedC` represent
different assay definitions. Keep those labels distinct throughout.
The pinned public [BAM methylation reader](https://github.com/Fu-Yilei/SniffCell/blob/d0486c98403e342c6d842295d28ddf243f1f7b70/src/sniffcell/anno/methyl_matrix.py)
combines 5mC/5hmC probabilities by default;
adding a catalog column alone does not change that behavior. For assay-specific
BAM annotation, use a SniffCell version with the appropriate assay selection
and check its help before applying a 5mC-only or 5hmC-only catalog.

## Optional: combine a legacy reference or use direct MDB support

`mdb atlas` can align a Loyfer-grouped long-read cohort to the complete legacy
SniffCell row system:

```bash
mdb atlas --legacy-npy atlas/all_celltypes_blocks.npy \
  --legacy-index atlas/all_celltypes_blocks.index.gz \
  --legacy-samples atlas/all_celltypes.txt --legacy-assay modifiedC \
  --cohorts reference.loyfer.mmdb -o combined_reference.mmdb
```

Obtain the three matching legacy files from the data release linked in the
[SniffCell README](https://github.com/Fu-Yilei/SniffCell#methylation-atlases).
`atlas` requires exact matching reference intervals, preserves absent rows as
missing, and writes `atlas_manifest.json`. It keeps legacy `modifiedC` and
long-read `5mC`/`5hmC` in separate views with their own sample columns. It does
not append duplicate assay/haplotype/strand views from multiple cohorts; merge
compatible long-read samples before this step. Select the relevant view and its
own sample IDs when exporting.

If your installed `sniffcell find --help` explicitly lists `--mdb` and `--assay`,
the NPY conversion can be skipped:

```bash
sniffcell find --mdb reference.loyfer.mmdb --assay 5mC \
  --haplotype combined --strand combined \
  -cf celltypes.json -ck brain_reference \
  --diff_threshold 0.40 --min_rows 1 --min_cpgs 3 --max_gap_bp 2000 \
  -o results/brain.5mC.direct.ctdmr.tsv
```

That direct interface was present in the tested development checkout but absent
from the public commit cited above. Check the installed help before using it;
the NumPy workflow is the route supported by the referenced public source.
