# SBTG pipeline source

This directory contains the estimator, synthetic benchmarks, data-preparation
code, and research workflows used to produce the released SBTG results.  The
repository-level documentation describes installation and the exact public
artifacts; this page is a source-code map.

All connectivity arrays use one direction convention:

```text
matrix[target, source] = weight of source -> target
```

## Reusable modules

| Path | Purpose |
| --- | --- |
| `models/sbtg.py` | Single-lag SBTG estimator and result container |
| `models/multilag_sbtg.py` | Multi-block estimator used for lag-specific fits |
| `models/multiblock_sbtg.py` | Shared multi-block model components |
| `utils/align.py` | Node alignment and direction checks |
| `utils/labels.py` | Functional-atlas positive/confirmed-negative labeling |
| `utils/metrics.py` | Classification, density-matched, and weight metrics |
| `utils/stimulus_periods.py` | Four-period stimulus segmentation |
| `SyntheticTestingUtils.py` | Synthetic generators, baselines, and metrics |
| `SyntheticTesting.py` | Synthetic benchmark driver |

The linear VAR(2) generator stabilizes the joint companion system.  Its public
helpers are `companion_spectral_radius(A1, A2)` and
`stabilize_var(A1, A2, target=0.9)`.

## Workflow scripts

The numbered scripts are command-line research workflows.  They expect the
prepared observation data described in the repository-level data documentation
and write beneath `results/`; they are not required to inspect the included
release matrices.

Script 01 stores variable-length recordings in pickle-free `traces.npz`
archives. Pipeline readers use the same flat-values, missing-mask, and offsets
schema with `allow_pickle=False`.

| Script | Function |
| --- | --- |
| `01_prepare_data.py` | Prepare and standardize observational traces |
| `02_train_sbtg.py` | Tune and fit single-lag SBTG models |
| `03_train_baselines.py` | Fit classical baselines |
| `04_evaluate.py` | Compare estimates with structural and functional references |
| `05_temporal_analysis.py` | Fit stimulus-period-specific models |
| `06_leifer_analysis.py` | Functional-atlas analysis |
| `07_regime_analysis.py` | Regime-gated model analysis |
| `08_generate_figures.py` | Pipeline summary figures |
| `09_neuron_tables.py` | Neuron-level tables |
| `10_fdr_sensitivity.py` | FDR sensitivity analysis |
| `12_hp_objective_validation.py` | Hyperparameter-objective validation |
| `15_multilag_analysis.py` | Lag-specific SBTG estimation |
| `16_celltype_analysis.py` | Cell-type connectivity aggregation |
| `17_neuron_ei_classification.py` | Molecular E/I annotation comparison |
| `18_multilayer_analysis.py` | Modulatory-network comparison |
| `19_state_dependent_analysis.py` | State-dependent connectivity analysis |

Scripts with command-line options expose `--help`; scripts 06, 09, and 10 run
without arguments. Inputs and outputs are path-relative to the repository root
unless a CLI option overrides them. For analysis directly from the distributed artifacts, use
`analysis/evaluation/prepare_merged_results.py`; it defaults to safe release
paths and does not depend on historical run directories.

## Synthetic benchmark smoke test

From the repository root:

```bash
python pipeline/SyntheticTesting.py --mini --skip-baselines
```

Optional causal-discovery baselines require their respective third-party
packages.  The benchmark reports unavailable optional methods instead of
silently substituting another estimator.

## Portability and provenance

Paths are constructed from the repository location.  Provenance records retain
random seeds, package versions, and git state, but intentionally omit hostnames,
working directories, and absolute script paths.  Public NPZ inputs are loaded
with `allow_pickle=False` by the release analysis code.
