"""Actual TensorQTL integration on synthetic methylation phenotypes.

TensorQTL is deliberately optional for ordinary methdb installs. To reproduce
the tested environment without changing the base interpreter or adding a
TensorQTL runtime dependency to methdb::

    python -m venv --system-site-packages /tmp/mdb-tensorqtl-validation
    uv pip install --python /tmp/mdb-tensorqtl-validation/bin/python --no-deps \
        tensorqtl==1.0.10 pandas-plink==2.2.9 pgenlib==0.95.1 qtl==0.1.10 \
        Deprecated==1.3.1 bx-python==0.16.0 pyBigWig==0.3.29
    mkdir -p .cache
    /tmp/mdb-tensorqtl-validation/bin/python -m pytest \
        tests/test_tensorqtl_export.py tests/test_tensorqtl_integration.py -q \
        --basetemp=.cache/tensorqtl_validation

The base interpreter must supply methdb's dependencies, torch, scipy, dask,
xarray, cffi, pyarrow, and pytest. pandas-plink 2.2.9 accepts NumPy 2; 2.3.2
declares NumPy<2, conflicting with methdb's required NumPy>=2.2.
Each integration case saves exported BED/receipt, genotype input, nominal
association parquet, permutation results, and a validation JSON receipt under
its pytest temporary directory. These are synthetic software checks, with no
claim about any production cohort or biological association.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress

tensorqtl = pytest.importorskip("tensorqtl", reason="Optional actual TensorQTL integration dependency is not installed")
import torch  # noqa: E402
from tensorqtl import cis  # noqa: E402

from mdb.export import export_tensorqtl  # noqa: E402
from test_tensorqtl_export import build_cohort, build_grouped_cohort  # noqa: E402


@pytest.mark.parametrize("backend", ["npy", "zarr"])
@pytest.mark.parametrize("grouped", [False, True], ids=["single_cpg", "grouped_cpg"])
def test_actual_tensorqtl_nominal_and_permutation(tmp_path: Path, backend: str, grouped: bool) -> None:
    seed = 20261030
    rng = np.random.default_rng(seed)
    n_samples = 48
    source_sample_ids = [f"sample_{i:02d}" for i in range(n_samples)]
    genotype_sample_ids = [f"genotype_{i:02d}" for i in range(n_samples)]
    genotype_ids = ["synthetic_causal_A", "synthetic_causal_B", "synthetic_null"]
    genotype_values = rng.binomial(2, 0.4, size=(3, n_samples))
    methylation = np.vstack([
        0.25 + 0.11 * genotype_values[0] + rng.normal(0, 0.035, n_samples),
        0.30 + 0.11 * genotype_values[0] + rng.normal(0, 0.035, n_samples),
        0.35 - 0.10 * genotype_values[1] + rng.normal(0, 0.035, n_samples),
        0.40 - 0.10 * genotype_values[1] + rng.normal(0, 0.035, n_samples),
    ])
    assert ((methylation >= 0) & (methylation <= 1)).all()
    source = build_cohort(
        tmp_path / "source.mmdb", methylation, backend=backend,
        positions=[100, 103, 300, 310], sample_ids=source_sample_ids,
    )
    if grouped:
        source = build_grouped_cohort(source, tmp_path / "groups.mmdb", backend=backend)
    bed = tmp_path / "methylation.bed.gz"
    # Reverse phenotype samples and rename them, then let real TensorQTL align
    # its differently ordered genotype columns by ID.
    export_receipt = export_tensorqtl(
        str(source), str(bed), samples=source_sample_ids[::-1],
        sample_map=dict(zip(source_sample_ids, genotype_sample_ids, strict=True)),
        batch_rows=1,
    )
    phenotype_df, phenotype_pos_df = tensorqtl.read_phenotype_bed(str(bed))
    expected_rows = 2 if grouped else 4
    assert phenotype_df.shape == (expected_rows, n_samples)
    assert phenotype_df.columns.tolist() == genotype_sample_ids[::-1]
    assert np.isfinite(phenotype_df.to_numpy()).all()
    if grouped:
        # TensorQTL converts BED start to 1-based and retains the full interval.
        assert phenotype_pos_df[["start", "end"]].values.tolist() == [[101, 105], [301, 312]]
    else:
        assert phenotype_pos_df["pos"].tolist() == [101, 104, 301, 311]

    genotype_df = pd.DataFrame(genotype_values, index=genotype_ids, columns=genotype_sample_ids)
    variant_df = pd.DataFrame({"chrom": ["chr1"] * 3, "pos": [101, 301, 701]}, index=genotype_ids)
    genotype_df.to_csv(tmp_path / "synthetic_genotypes.tsv", sep="\t", index_label="variant_id")
    variant_df.to_csv(tmp_path / "synthetic_variant_positions.tsv", sep="\t", index_label="variant_id")
    cis.map_nominal(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df,
        prefix="synthetic", window=1000, output_dir=str(tmp_path), verbose=False,
    )
    nominal = pd.read_parquet(tmp_path / "synthetic.cis_qtl_pairs.chr1.parquet")
    assert len(nominal) == expected_rows * len(genotype_ids)
    assert set(nominal["phenotype_id"]) == set(phenotype_df.index)
    max_slope_error = 0.0
    max_log10_p_error = 0.0
    comparisons = []
    for row in nominal.itertuples(index=False):
        genotype = genotype_df.loc[row.variant_id, phenotype_df.columns].to_numpy(dtype=float)
        methylation_beta = phenotype_df.loc[row.phenotype_id].to_numpy(dtype=float)
        reference = linregress(genotype, methylation_beta)
        np.testing.assert_allclose(row.slope, reference.slope, atol=2e-7, rtol=2e-5)
        np.testing.assert_allclose(row.slope_se, reference.stderr, atol=2e-7, rtol=2e-5)
        np.testing.assert_allclose(row.pval_nominal, reference.pvalue, atol=0, rtol=5e-4)
        max_slope_error = max(max_slope_error, abs(row.slope - reference.slope))
        log10_error = abs(np.log10(row.pval_nominal) - np.log10(reference.pvalue))
        max_log10_p_error = max(max_log10_p_error, log10_error)
        comparisons.append({
            "phenotype_id": row.phenotype_id, "variant_id": row.variant_id,
            "tensorqtl_slope": float(row.slope), "scipy_slope": float(reference.slope),
            "tensorqtl_pvalue": float(row.pval_nominal), "scipy_pvalue": float(reference.pvalue),
        })
    top = nominal.loc[nominal.groupby("phenotype_id")["pval_nominal"].idxmin()].set_index("phenotype_id")
    expected_variants = ["synthetic_causal_A", "synthetic_causal_B"] if grouped else [
        "synthetic_causal_A", "synthetic_causal_A", "synthetic_causal_B", "synthetic_causal_B",
    ]
    assert top.loc[phenotype_df.index, "variant_id"].tolist() == expected_variants
    assert (top["pval_nominal"] < 1e-10).all()

    permutations = cis.map_cis(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df,
        window=1000, nperm=100, seed=seed, beta_approx=True, verbose=False,
    )
    assert len(permutations) == expected_rows
    assert permutations.loc[phenotype_df.index, "variant_id"].tolist() == expected_variants
    assert np.isfinite(permutations["pval_perm"]).all()
    assert permutations["pval_perm"].between(0, 1).all()
    assert permutations["pval_beta"].between(0, 1).all()
    permutations.to_csv(tmp_path / "synthetic.cis_permutations.tsv", sep="\t")

    receipt = {
        "status": "passed", "data": "synthetic only", "seed": seed,
        "backend": backend, "representation": "cpg_groups" if grouped else "individual_cpg",
        "phenotype": "methylation beta fraction [0,1]",
        "samples": n_samples, "phenotypes": expected_rows, "variants": len(genotype_ids),
        "nominal_pairs_checked_against_scipy": len(nominal), "permutations_per_phenotype": 100,
        "permutation_phenotypes": len(permutations),
        "known_causal_variant_recovered_for_all_phenotypes": True,
        "max_absolute_slope_error": max_slope_error,
        "max_absolute_log10_pvalue_error": max_log10_p_error,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in [
            "tensorqtl", "torch", "numpy", "pandas", "scipy", "pandas-plink", "pgenlib", "pyarrow", "zarr",
        ]},
        "bed": str(bed), "export_receipt": export_receipt,
        "nominal_parquet": str(tmp_path / "synthetic.cis_qtl_pairs.chr1.parquet"),
        "comparisons": comparisons,
    }
    (tmp_path / "tensorqtl_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
