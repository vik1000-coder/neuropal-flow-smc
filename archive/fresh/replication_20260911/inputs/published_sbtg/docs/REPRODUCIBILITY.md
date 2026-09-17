# Reproducibility and lineage

## Determinism

The public pipeline explicitly seeds NumPy and PyTorch where stochastic fitting is compared across conditions. Exact bitwise agreement is not guaranteed across PyTorch versions, hardware backends, BLAS implementations, or GPU kernels. Statistical outputs should be checked against the released compact tables rather than assumed to reproduce byte for byte after retraining.

## Manuscript result lineage

The released manuscript matrix has two sources:

- lag 1: later regime-gated fit;
- lags 2, 3, 5, 8, 10, 15, and 20: preserved production multi-lag fit.

The preserved production run stored aggregate matrices and hyperparameters but not every trained fold model or the complete PyTorch RNG state. Exact historical training replay is therefore not possible. The `reference_snapshot/` source and artifact hashes establish the recoverable lineage, while new resampling fits add explicit fold-local seeds.

## Safe artifacts

Released NPZ files contain only numeric, Boolean, and Unicode arrays and load with `allow_pickle=False`. Hyperparameter dictionaries live in JSON sidecars. The legacy manuscript lag-1 `sig` field is retained as `q_value_lag1`, not Boolean-cast.

## Simulation correction

The VAR(2) generator jointly rescales lag matrices through the companion-system roots. Independently normalizing each lag matrix does not guarantee stationarity. The regression test verifies preserved support, a companion radius at or below the requested target, and finite generated traces.

## Statistical scope

Recording-row bootstrap results average the two model seeds within each resample. Four resamples imply four independent resampling units. Reported spreads are descriptive and are not calibrated confidence intervals. Phase endpoint ratios were selected after inspecting alternative summaries and are explicitly labeled exploratory.
