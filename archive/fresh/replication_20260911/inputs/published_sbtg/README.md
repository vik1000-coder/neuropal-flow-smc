# Score-Based Temporal Graphical Models for functional connectome inference

This repository contains the code, compact reference inputs, manuscript result matrices, and validated sensitivity analyses for inferring lag-specific effective connectivity from *C. elegans* calcium-imaging traces with Score-Based Temporal Graphical (SBTG) models.

Lag-specific effective connectivity means model-based directed influence among recorded variables at a specified lag. It need not coincide with static anatomy, and an observational estimate is not a model-free interventional effect.

## Contents

| Path | Contents |
|---|---|
| `pipeline/` | Preprocessing, model fitting, baselines, evaluation, and simulation code |
| `analysis/` | Reusable evaluation and figure-generation scripts |
| `experiments/` | Residual, covariance, lag-profile, phase-duration, method-agreement, and partial-observation analyses |
| `results/paper/` | Pickle-free manuscript lag matrices and compact evaluation tables |
| `results/robustness/` | Curated outputs from validated sensitivity analyses |
| `reference_data/` | Compact aligned reference networks and modulatory edge lists |
| `reference_snapshot/` | Portable production-lineage source, prepared traces, and phase anchors used for exact sensitivity fits |
| `tools/` | Safe artifact conversion and release validation |
| `tests/` | Numerical, schema, and privacy regression tests |

Paths are repository-relative; hostnames and site-specific compute layouts are not recorded.

## Environment

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The listed environment covers the core pipeline and analysis scripts. Long model fits benefit from a CUDA device but support CPU execution.

## Start with released artifacts

```python
import numpy as np

with np.load("results/paper/sbtg_lag_matrices.npz", allow_pickle=False) as result:
    lags = result["lags"]
    lag_5_effective_connectivity = result["mu_hat_lag5"]
```

The manuscript matrix is hybrid by construction: lag 1 comes from a later regime-gated fit, and lags 2 and above come from the preserved production multi-lag fit. The historical field called `sig_lag1` was continuous, so the public artifact stores it as `q_value_lag1` and does not invent a Boolean lag-1 mask. See [results/README.md](results/README.md).

## Reproduce analyses

The main empirical pipeline expects source calcium-imaging files documented in [docs/DATA.md](docs/DATA.md):

```bash
python pipeline/01_prepare_data.py --impute-missing --full-traces
python pipeline/15_multilag_analysis.py --approach C --p-max 20 \
  --lags 1 2 3 5 8 10 15 20 --tune-hp
```

Production-lineage resampling and phase-duration fits use the portable inputs in `reference_snapshot/`; experiment-specific commands are documented in [experiments/README.md](experiments/README.md). These are computationally intensive and retraining is not expected to be bit-identical across hardware. Preserved aggregate matrices and source hashes provide the lineage anchor.

## Key sensitivity conclusions

- Residual dependence after a recording-held-out VAR(20) fit is small on average (mean absolute off-diagonal correlation 0.02471) but strongly dependent on temporal model order.
- Paired structured-noise simulations show little change at residual-proxy correlation levels and clear false-positive degradation at the 0.20 stress level.
- Monoamine discrete lag peaks are unstable under recording-row resampling. Weighted profile centers are descriptive fixed-tuning sensitivities, not biological time constants or confidence intervals.
- Duration-matched phase profiles show learnable phase-local temporal differentiation. SBTG is sensitive to these differences, and their presence in observational traces motivates causal hypotheses for future perturbational study. The separation is threshold sensitive and does not itself identify causal effects.
- Method rankings are non-equivalent, but disagreement does not establish biologically unique recovery.

## Validate the release

```bash
python tools/validate_release.py .
python -m unittest discover -s tests -v
```

`release_manifest.csv` and `checksums.sha256` cover every regular file except the two index files themselves.

## Data and licensing

Third-party datasets retain their original publication terms and are not relicensed here. Raw recordings and source workbooks are omitted; compact aligned inputs are included for analysis transparency. See [docs/DATA.md](docs/DATA.md).

No open-source license has been selected for the code. Until the copyright holders add one, the code is provided for inspection with all rights reserved; see [LICENSE](LICENSE).
