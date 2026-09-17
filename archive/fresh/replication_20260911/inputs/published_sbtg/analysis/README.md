# Analysis from released artifacts

These scripts operate on the public, pickle-free matrices and reference data.
Run commands from the repository root.  No timestamped run directory or local
machine path is required.

## Reference evaluation

The primary entry point is `evaluation/prepare_merged_results.py`.  Defaults:

| Input or output | Release path |
| --- | --- |
| SBTG lag matrices | `results/paper/sbtg_lag_matrices.npz` |
| Optional baselines | `results/paper/baselines/*.npz` |
| Structural reference | `reference_data/connectome/` |
| Functional reference | `reference_data/functional_atlas/aligned_atlas_wild_type.npz` |
| Modulatory reference | `reference_data/modulatory_atlas/edge_lists/edgelist_MA_classes.csv` |
| Evaluation output | `results/derived/evaluation/` |

```bash
python analysis/evaluation/prepare_merged_results.py
```

The command writes:

- `structural_reference_metrics.csv`: structural-reference metrics;
- `chemical_gap_reference_metrics.csv`: Cook chemical and gap references evaluated separately;
- `functional_reference_metrics.csv`: functional-reference metrics, using `q < 0.05` as
  positives, `q_eq < 0.05` as confirmed negatives, and excluding ambiguous
  pairs;
- `modulatory_reference_metrics.csv`: transmitter-specific modulatory-reference
  metrics;
- `evaluation_metadata.json`: metric definitions, label policy, and prevalences.

AUROC and AUPRC use absolute continuous edge scores.  `f1` is the maximum over
thresholds on the evaluated vector and is therefore descriptive, not a held-out
operating point.  Weight correlations use signed weights and explicit Boolean
significance arrays.  When an archive instead provides an explicit `q_value`
array, the evaluator applies the declared `--q-alpha` threshold (0.2 by
default); it never treats arbitrary nonzero floats as Boolean significance.

All input paths and the sampling rate can be overridden:

```bash
python analysis/evaluation/prepare_merged_results.py \
  --sbtg path/to/sbtg_lag_matrices.npz \
  --baseline "VAR=path/to/var_lag_matrices.npz" \
  --output-dir path/to/evaluation
```

`--baseline` may be repeated.  A baseline archive can expose
`mu_hat_lagN` arrays or named arrays such as `VAR_lagN`, and should include a
Unicode `neuron_names` array when its order differs from SBTG.

## Discrete peak summaries

```bash
python analysis/evaluation/analyze_peaks.py
python analysis/figures/generate_peak_lag_figure.py
```

The summary reports the maximum among evaluated lags only.  It intentionally
does not estimate sub-frame peak locations or confidence intervals from a
three-point interpolation.

## Phase-specific summaries

The distributed phase snapshot lives in `reference_snapshot/phase_analysis`.
Generate the cell-type comparison with:

```bash
python analysis/figures/generate_phase_paper_figures.py
```

Both input and output directories are configurable:

```bash
python analysis/figures/generate_phase_paper_figures.py \
  --input-dir reference_snapshot/phase_analysis/comparison \
  --output-dir results/derived/figures/phase_analysis
```

Biological networks used in evaluation are independent references; agreement
with them does not by itself establish complete causal identification.
