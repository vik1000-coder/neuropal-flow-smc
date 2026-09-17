# Functional Connectome Inference via Score-Based Temporal Graphical Models

Inferring *C. elegans* functional connectivity from calcium imaging using Score-Based Temporal Graphical (SBTG) models. Networks are validated against `Cook_Synapses_2019` (structural benchmark) and `Randi_Optogenetics_2023` (functional benchmark)

## Quick Start

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
./run_pipeline.sh all --trials 50

# 4. View results
cat results/summary/RESULTS_INDEX.md
ls results/figures/summary/
```

> **Note:** The dataset `data/Head_Activity_OH16230.mat` is >100MB and may be stored in split parts. Script `01_prepare_data.py` automatically reassembles it on first run.

### Pipeline Modes

```bash
./run_pipeline.sh all             # Full pipeline (HP search + all analyses)
./run_pipeline.sh all --trials 50 # With 50 HP trials
./run_pipeline.sh quick           # Fast test run
./run_pipeline.sh figures         # Regenerate figures only
./run_pipeline.sh clean           # Remove all results
```

> `./run_pipeline.sh all` defaults to 5 HP trials unless `--trials N` is provided.

---

## Pipeline Scripts

| Script                               | Purpose                                        |
| ------------------------------------ | ---------------------------------------------- |
| `01_prepare_data.py`                 | Data loading, imputation, connectome alignment |
| `02_train_sbtg.py`                   | HP search + SBTG training (null_contrast)      |
| `03_train_baselines.py`              | Baseline methods (Pearson, Granger, GLasso)    |
| `04_evaluate.py`                     | Evaluation vs structural/functional benchmarks |
| `05_temporal_analysis.py`            | Phase-specific SBTG (null_contrast HP)         |
| `06_leifer_analysis.py`              | Extended functional-atlas analysis             |
| `07_regime_analysis.py`              | Regime-gated interpretation                    |
| `08_generate_figures.py`             | Summary visualizations                         |
| `09_neuron_tables.py`                | Neuron significance tables                     |
| `10_fdr_sensitivity.py`              | FDR sensitivity analysis                       |
| `12_hp_objective_validation.py`      | HP objective validation (run once, skipped)    |
| `14_organize_results.py`             | Consolidate results                            |
| **`15_multilag_analysis.py`**        | **Multi-lag SBTG + baselines**                 |
| **`16_celltype_analysis.py`**        | **Cell-type stats + dynamic figures**          |
| **`17_neuron_ei_classification.py`** | **Neuron E/I classification (5 methods)**      |
| **`18_multilayer_analysis.py`**      | **Monoamine connectome evaluation**            |
| **`19_state_dependent_analysis.py`** | **State-dependent connectivity (AVA/AVB)**     |
| **`SyntheticTesting.py`**            | **Synthetic benchmarks (4 DGPs x methods)**    |
| **`syntheticexperiment2.py`**        | **Reviewer-scale synthetic benchmark (n=80)**  |

Scripts 17-19 are run manually (not wired into `run_pipeline.sh`). Scripts 11 and 13 have been archived; their functionality is unified in Script 15.

---

## Project Structure

```
diffusionCircuit/
├── pipeline/                        # Core pipeline scripts (01-19)
│   ├── models/                      #   SBTG model implementations
│   ├── utils/                       #   Shared utilities
│   ├── configs/                     #   Phase-specific parameters
│   ├── tests/                       #   Unit tests
│   ├── SyntheticTesting.py          #   Synthetic benchmark driver
│   ├── SyntheticTestingUtils.py     #   Data generators & baselines
│   └── README.md                    #   Pipeline reference
├── analysis/                        # Post-hoc analysis & figures
│   ├── evaluation/                  #   Metric computation scripts
│   ├── figures/                     #   Figure generation scripts
│   ├── validation/                  #   Sanity checks
│   └── README.md                    #   Analysis guide & workflow
├── data/                            # Input data
│   ├── Head_Activity_*.mat          #   NeuroPAL recordings
│   ├── SI *.xlsx                    #   Cook connectome
│   └── S1_Dataset/                  #   Monoamine/neuropeptide networks
├── merged_results/                  # Publication result artifacts
│   ├── *.npz                        #   SBTG & baseline adjacency matrices
│   ├── figures/                     #   Evaluation CSVs & publication figures
│   ├── external_baselines/          #   NRI / NetFormer / LINT workspace
│   ├── synthetic_sbtg_results/      #   Synthetic benchmark outputs
│   └── README.md                    #   Results summary & data dictionary
├── scripts/                         # SLURM cluster job scripts
├── run_pipeline.sh                  # Main pipeline script
└── requirements.txt
```

---

## Cluster Deployment

```bash
# Empirical pipeline
sbatch scripts/sbtg_slurm.sh

# Multi-lag analysis
sbatch scripts/multilag_slurm.sh

# Synthetic benchmarks
sbatch scripts/synthetic_slurm.sh

# Reviewer-scale synthetic benchmark (includes optional external DL baselines)
sbatch scripts/syntheticexperiment2_slurm.sh --medium --n-neurons 80 --run-external-baselines

# External DL baselines (NRI / NetFormer / LINT)
sbatch scripts/external_baselines_slurm.sh nacl 50
```

SLURM scripts use placeholder paths (`PROJECT_DIR="/path/to/your/project"`) that must be edited before submission.

---

## Dependencies

See `requirements.txt` for pinned versions. Core dependencies:

- numpy, scipy, pandas, matplotlib, seaborn
- torch, scikit-learn, statsmodels, optuna
- networkx, openpyxl, h5py, wormneuroatlas, tqdm

Optional (for synthetic baselines): lingam, tigramite, notears, causalnex

Install with: `pip install -r requirements.txt`

---

## References

- Cook et al. (2019). Nature 571:63-71. (Structural connectome)
- Randi et al. (2023). Nature 623:406-414. (Functional atlas)
- Yemini et al. (2021). Cell 184:272-288. (NeuroPAL)
- Bentley et al. (2016). PLoS Comput Biol 12:e1005283. (Multilayer connectome)
