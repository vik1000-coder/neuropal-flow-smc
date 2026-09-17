# G8 Gaussian-shadow negative-control validation

All eight planned cells completed (4 independent G8 geometries × 2 data seeds). The shadow exactly matches G8's conditional mean and covariance but removes the control-dependent higher-order mixture.

- Train/validation/test sizes per side: 4,000 / 1,000 / 10,000.
- The raw path/history MLP never receives the control label as an input.
- Median held-out AUC: 0.4987; range: 0.4957–0.5051.
- Median absolute classifier-projected effect: 0.0094% of the matched positive G8 effect.
- Maximum absolute classifier-projected effect: 0.1597% of the matched positive G8 effect.

The negative control therefore behaves as a population null and provides no evidence of label leakage or low-order confounding.
