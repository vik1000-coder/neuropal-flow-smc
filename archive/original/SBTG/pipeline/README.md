# Pipeline Reference

Complete reference for the SBTG functional connectivity pipeline (19 scripts + synthetic benchmarks).

---

## Quick Start

```bash
./run_pipeline.sh all --trials 50    # Full pipeline with HP tuning
./run_pipeline.sh quick              # Quick test (fewer epochs/trials)
./run_pipeline.sh figures            # Figures only (requires previous runs)
```

**Defaults:** Optuna TPE sampler with `null_contrast` objective.
`run_pipeline.sh all` defaults to 5 HP trials unless `--trials N` is provided.

---

## Script Overview

| Script | Purpose | Runtime |
|--------|---------|---------|
| `01_prepare_data.py` | Data loading, imputation, connectome alignment | ~2 min |
| `02_train_sbtg.py` | HP search + SBTG training (Optuna + null_contrast) | 15 min – 4 hr |
| `03_train_baselines.py` | Baseline methods (Pearson, Granger, GLasso) | ~10 min |
| `04_evaluate.py` | Evaluation vs structural/functional benchmarks | ~5 min |
| `05_temporal_analysis.py` | Phase-specific SBTG + Optuna HP tuning | ~30 min |
| `06_leifer_analysis.py` | Extended functional-atlas analysis | ~10 min |
| `07_regime_analysis.py` | Regime-gated model interpretability | ~5 min |
| `08_generate_figures.py` | Summary visualizations (main + supplemental panels) | ~5 min |
| `09_neuron_tables.py` | Neuron significance tables | ~1 min |
| `10_fdr_sensitivity.py` | FDR threshold sensitivity analysis | ~2 min |
| `12_hp_objective_validation.py` | HP objective validation (run once) | ~1–4 hr |
| `14_organize_results.py` | Consolidate results + generate index | ~1 min |
| **`15_multilag_analysis.py`** | Unified multi-lag SBTG (3 approaches) | ~30–60 min |
| **`16_celltype_analysis.py`** | Cell-type connectivity analysis | ~1 min |
| **`17_neuron_ei_classification.py`** | Neuron E/I classification (5 methods) | ~1 min |
| **`18_multilayer_analysis.py`** | Monoamine connectome evaluation | ~1 min |
| **`19_state_dependent_analysis.py`** | State-dependent connectivity (AVA/AVB) | ~1 min |
| **`SyntheticTesting.py`** | Synthetic benchmark suite (4 DGPs x methods, n=10 default) | 2–12 hr |

Scripts 17–19 are run manually (not wired into `run_pipeline.sh`).

---

## Key Scripts

### 01_prepare_data.py

```bash
python pipeline/01_prepare_data.py --impute-missing   # Recommended (6 → 20 worms)
python pipeline/01_prepare_data.py --full-traces       # Full 240s recordings
```

Key flags: `--impute-missing`, `--full-traces`, `--min-worms N`, `--no-tail`, `--no-collapse-dv`

### 02_train_sbtg.py

```bash
python pipeline/02_train_sbtg.py --mode hp_search --n_trials 50
python pipeline/02_train_sbtg.py --mode train --use_imputed
```

Key flags: `--mode {hp_search, train}`, `--objective {null_contrast, dsm_loss}`, `--model_type {linear, feature_bilinear, regime_gated}`

### 15_multilag_analysis.py

Three approaches for lag-separated Jacobian recovery:

| Approach | Method | Per-Lag HP | Best Structural AUROC | Best Functional AUROC |
|----------|--------|-----------|-----------------|-------------------|
| A | Per-lag 2-block models | Yes | 0.533 | 0.573 |
| B | Single (p+1)-block model | No | 0.525 | 0.519 |
| **C** | **Per-lag (r+1)-block models** | **Yes** | **0.552** | **0.612** |

```bash
python pipeline/15_multilag_analysis.py --approach C                          # Default
python pipeline/15_multilag_analysis.py --approach C --tune-hp --n-hp-trials 30  # With HP tuning
python pipeline/15_multilag_analysis.py --approach all                        # All approaches
```

Key flags: `--approach {A, B, C, all}`, `--lags L1 L2 ...`, `--tune-hp`, `--n-hp-trials N`, `--p-max`, `--epochs`, `--n-folds`, `--device`

### SyntheticTesting.py

```bash
python pipeline/SyntheticTesting.py --hp-trials 20          # Full benchmarks
python pipeline/SyntheticTesting.py --mini --skip-baselines # Quick smoke test
```

Data generators and baseline wrappers live in `SyntheticTestingUtils.py`.

## Subdirectories

| Directory | Contents |
|-----------|----------|
| `models/` | SBTG model implementations (`sbtg.py`, `multilag_sbtg.py`, `multiblock_sbtg.py`) |
| `utils/` | Shared utilities (alignment, neuron types, reproducibility, stimulus periods, etc.) |
| `configs/` | Phase-specific hyperparameter configurations |
| `tests/` | Unit tests |
| `archive/` | Archived scripts (11, 13) and old documentation |

---

## Output Directory Structure

```
results/
├── intermediate/
│   ├── connectome/           # Aligned Cook matrices (A_struct.npy, A_chem.npy, etc.)
│   └── datasets/             # Prepared datasets per stimulus
├── sbtg_training/
│   ├── hyperparams/          # HP search results
│   └── models/               # Trained models
├── baselines/                # Baseline method results
├── evaluation/               # Structural/functional benchmark evaluation
├── stimulus_specific/        # Phase-specific SBTG
├── multilag_separation/      # Script 15 outputs
│   └── {timestamp}/
│       ├── result_C.npz      # mu-hat, p-values, significance per lag
│       ├── eval_cook_C.csv   # Cook evaluation
│       └── figures/           # Generated figures
├── hp_objective_validation/  # Script 12 outputs
├── figures/summary/          # All generated figures
├── tables/                   # Summary tables
└── summary/                  # Consolidated results index
```

---

## Cluster Deployment

```bash
sbatch scripts/sbtg_slurm.sh                        # Full pipeline
sbatch scripts/mini_sbtg_slurm.sh                   # Quick test
sbatch scripts/multilag_slurm.sh                    # Multi-lag analysis
sbatch scripts/synthetic_slurm.sh                   # Synthetic benchmarks
sbatch scripts/syntheticexperiment2_slurm.sh        # Synthetic scaling (n=80 default)
sbatch scripts/synthetic_baseline_slurm.sh          # Multi-lag + baselines + monoamine
sbatch scripts/external_baselines_slurm.sh nacl 50  # External DL baselines
```

SLURM scripts use `PROJECT_DIR="/path/to/your/project"` placeholder — edit before submission.
