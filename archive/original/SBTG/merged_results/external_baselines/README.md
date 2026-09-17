# External Deep Learning Baselines

## Overview

This directory benchmarks three external deep-learning architectures against the Score-Based Temporal Graphical (SBTG) pipeline for functional connectome inference.  Each method is trained on the same sliding-window calcium-imaging data and evaluated against the Cook et al. structural connectome and (optionally) the Leifer functional atlas.

## Prerequisites

Clone the external repos into this directory:

```bash
cd external_baselines/

# NRI (Neural Relational Inference) — Kipf et al., 2018
git clone https://github.com/ethanfetaya/NRI.git nri

# NetFormer (Transformer for Neural Population Dynamics) — Chen et al., 2024
git clone https://github.com/johnlyzhou/NetFormer.git NetFormer

# LINT (Low-Rank Inference from Neural Trajectories) — Dubreuil / Valente et al., 2022
git clone https://github.com/adrian-valente/lowrank_inference.git lowrank_inference
```

Additional Python dependencies (`einops`, `pytorch_lightning`) may be needed.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `external_analysis.py` | Train NRI / NetFormer / LINT and save predicted adjacency matrices |
| `evaluate_external.py` | Evaluate predictions against Cook (structural) and Leifer (functional) ground truths |
| `synthetic_analysis.py` | Run all three methods on synthetic VAR / Hawkes data with known connectivity |

`evaluate_external.py` computes AUROC/AUPRC on continuous scores and computes
F1 using an exact top-K edge rule (`density=0.15`) via index selection
(tie-safe), so F1 reflects exactly the intended edge budget.

---

## Reproduction

### Empirical evaluation (requires the `nacl` dataset from `01_prepare_data.py`):

```bash
python external_baselines/external_analysis.py --dataset nacl --epochs 50
python external_baselines/evaluate_external.py --dataset nacl
```

### Synthetic evaluation:

```bash
python external_baselines/synthetic_analysis.py
```

### SLURM (runs all three steps):

```bash
sbatch scripts/external_baselines_slurm.sh nacl 50
```

---

## Results on *C. elegans* Data (`nacl`)

Connectivity matrices evaluated against the Cook et al. (2019) anatomical connectome.

| Method          | Cook AUROC | Cook AUPRC | Cook F1 | Correlation |
|:----------------|----------:|----------:|--------:|------------:|
| **SBTG (Lag 1)**| **0.581** | **0.289** | **0.284**| **0.155**  |
| NRI             |   0.507   |   0.208   |  0.173  |   0.033     |
| NetFormer       |   0.505   |   0.205   |  0.176  |   0.008     |
| LINT            |   0.503   |   0.204   |  0.163  |   0.006     |

*Note*: The near-chance performance of the external baselines reflects genuine limitations of these methods on this dataset (186 windows, 80 neurons, T=10 at 4Hz). 