# Post-hoc Analysis Scripts

Standalone scripts for evaluation, figure generation, and validation that operate on pipeline outputs. All scripts are run from the **project root** directory.

**Prerequisites:** Complete the main pipeline (`run_pipeline.sh`) so that `results/` is populated, then ensure required NPZ artifacts are available in `merged_results/` (see `merged_results/README.md`).

---

## Directory Structure

```
analysis/
├── evaluation/       # Metric computation and subnetwork scoring
└── figures/          # Publication figure generation
```

---

## Workflow

Most scripts are independent and can be run in any order. The two exceptions:

1. **Run `evaluation/prepare_merged_results.py` first** — it assembles evaluation CSVs from raw pipeline outputs.
2. **Then** `figures/generate_merged_figures.py` and `evaluation/analyze_peaks.py`, which read those CSVs.

```bash
# Step 1: Prepare evaluation CSVs
python analysis/evaluation/prepare_merged_results.py

# Step 2: Generate main figures
python analysis/figures/generate_merged_figures.py

# Step 3: Run any additional analyses (independent of each other)
python analysis/evaluation/analyze_chem_gap.py
python analysis/evaluation/analyze_gaba_receptors.py
python analysis/evaluation/analyze_nt_comparison.py
python analysis/evaluation/analyze_synaptic_vs_nonsynaptic.py
python analysis/evaluation/analyze_peaks.py
python analysis/figures/generate_peak_lag_figure.py
```

---

## Evaluation Scripts (`evaluation/`)

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `prepare_merged_results.py` | Assemble pipeline outputs into evaluation CSVs (run first) | `merged_results/*.npz`, connectome, functional atlas, monoamine edge lists | `merged_results/figures/eval_*.csv` |
| `analyze_chem_gap.py` | Chemical synapse vs gap junction AUROC/AUPRC/F1 per lag | `result_C_merged.npz`, `A_chem.npy`, `A_gap.npy` | `figures/chem_gap_analysis.csv`, `fig_chem_gap_*.png` |
| `analyze_gaba_receptors.py` | GABA-A (ionotropic) vs GABA-B (metabotropic) recovery | `result_C_merged.npz`, `connectome_syn_gj.csv` | `figures/gaba_receptor_*.csv`, `fig_gaba_*.png` |
| `analyze_nt_comparison.py` | Neurotransmitter-specific subnetwork comparison | `result_C_merged.npz`, `connectome_syn_gj.csv`, monoamine edge lists | `figures/nt_comparison.csv`, `fig_nt_*.png` |
| `analyze_peaks.py` | Identify peak F1 lag per evaluation target | `eval_cook_C.csv`, `eval_leifer_C.csv`, `eval_monoamine_*.csv` | Console output |
| `analyze_synaptic_vs_nonsynaptic.py` | GABA chemical-synapse vs electrical (gap-junction) edge recovery | `result_C_merged.npz`, `connectome_syn_gj.csv` | `figures/synaptic_vs_nonsynaptic_*.csv`, `fig_synaptic_*.png` |
| `extract_strongest_edges.py` | Top-K directed edges per lag with FDR masking | Any `result_*.npz` (CLI arg) | `results/summary/strongest_edges.csv` |

---

## Figure Scripts (`figures/`)

| Script | Purpose | Outputs |
|--------|---------|---------|
| `generate_merged_figures.py` | Main publication figures: AUROC/AUPRC comparisons, cell-type heatmaps, monoamine plots | `merged_results/figures/fig_*.png` |
| `generate_peak_lag_figure.py` | Bar chart of peak time lags per network category | `merged_results/figures/fig_peak_lags.png` |
| `generate_phase_specific_figures.py` | Phase-specific connectivity heatmaps and E:I ratios | `merged_results/figures/phase_analysis/` |
| `generate_phase_paper_figures.py` | Clean paper-ready phase figures (Baseline/On/Steady/Off) | `results/multilag_separation/.../paper_figures/` |
| `plot_edge_dynamics.py` | Edge weight evolution line plots and heatmaps across lags | Specified via CLI |
| `plot_sankey.py` | Sankey flow diagrams of directed connectivity (Plotly) | Specified via CLI |
