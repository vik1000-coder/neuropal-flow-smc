# Robustness experiments

This package contains code for robustness analyses that complement the paper
pipeline. It uses repository-relative defaults, pickle-free NumPy
archives, deterministic seed schedules, and explicit output directories.

Run commands from the repository root with `python -m` so package imports are
stable. The reference snapshot can be checked without fitting a model:

```bash
python -c "from experiments.snapshot import validate_snapshot; validate_snapshot(True)"
```

## Experiment map

| Analysis | Fit or diagnostic entry point | Summary entry point |
|---|---|---|
| structured innovation covariance | `experiments.structured_noise` | written by the same command |
| residual dependence and VAR-order sensitivity | `experiments.residual_alignment`, `experiments.residual_order_sensitivity` | written by each command |
| recording-row bootstrap of lag profiles | `experiments.lag_profile_fit`, `experiments.run_lag_profile_campaign` | `experiments.lag_profile_summary`, `experiments.lag_profile_stability` |
| duration-matched phase profiles | `experiments.phase_duration_fit` | `experiments.phase_duration_summary` |
| phase-window/scaling sensitivities | `experiments.phase_window_fit`, `experiments.run_phase_window_campaign` | `experiments.phase_window_summary`, `experiments.phase_endpoint_summary` |
| method agreement | `experiments.method_agreement` | written by the same command |
| partial observation | `experiments.partial_observation` | written by the same command |

Examples:

```bash
python -m experiments.structured_noise --out-dir results/derived/structured_noise \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

python -m experiments.run_lag_profile_campaign \
  --output-root results/derived/lag_profile_bootstrap --bootstrap-replicates 0 1 2 3

python -m experiments.lag_profile_stability \
  --input-root results/robustness/lag_profiles/fits \
  --out-dir results/derived/lag_profile_stability --require-complete

python -m experiments.lag_profile_stability \
  --input-root results/robustness/lag_profiles/fits \
  --analysis-scope secondary_all8_diagnostic \
  --out-dir results/derived/lag_profile_stability_all8 --require-complete

python -m experiments.phase_duration_fit --out-root results/derived/phase_duration \
  --phase baseline --mode matched --replicate 0

python -m experiments.method_agreement --out-dir results/derived/method_agreement
```

The recording-row bootstrap has four independent data resamples and two
algorithmic seeds per resample. It is a descriptive fixed-tuning sensitivity
analysis, not a confidence interval. Its primary cross-lag comparison uses lags
2, 3, 5, 8, 10, 15, and 20; the lag-1 retraining uses a different estimator
lineage than the paper's regime-gated lag-1 artifact and is secondary.
The bundled `results/robustness/lag_profiles/fits/` directory contains all 88
portable fit artifacts needed to replay both stability summaries without
retraining.

The duration-matched phase analysis supports a narrower conclusion than the
unmatched profiles: activation and post-activation intervals are not separated
from matched baseline by the derived long/short index. The endpoint ratio is
exploratory and correction-sensitive, so it is kept as a separate summary.

Saved fit manifests record repository-relative logical inputs and SHA-256
digests. Logs and machine-specific execution metadata are not scientific
artifacts and should not be distributed.
