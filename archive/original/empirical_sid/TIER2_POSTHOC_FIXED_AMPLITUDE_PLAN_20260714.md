# Tier 2 post-hoc diagnostic: fixed-amplitude learning curves

**Frozen:** 2026-07-14 after inspecting the preregistered fixed-\(\Lambda\)
sample-size slice and before any result from this diagnostic was generated.

**Status:** exploratory diagnostic. It cannot upgrade a confirmatory recovery or
advantage claim.

## Motivation

The preregistered Tier 2 sample-size slice holds
\(\Lambda=N\delta^2 I_v\) near 3 by recalibrating mechanism amplitude at each
sample size. That is a contiguous-local-alternative efficiency test, not an
ordinary fixed-effect learning curve. This diagnostic separates the two.

## Frozen design

- New paired DGP/data seeds: 4001--4030.
- Mechanisms: m3_bounded, m4_quartic, m5_occupancy.
- Training sizes: 2,000, 8,000, and 32,000.
- For each mechanism, freeze amplitude to the value calibrated at
  \(N=8{,}000\), target \(\Lambda=3\), with the same positivity bounds as the
  preregistered suite.
- Run the identical frozen Tier 1 v4 methods, bases, hyperparameters, evaluation
  histories, metrics, and primary \(\delta=0.12\).
- No method selection, hyperparameter change, or per-cell rescue is permitted.
- Store one immutable CSV and NPZ per mechanism/seed/sample-size cell.
- Inference remains a 2,000-replicate DGP-seed bootstrap.
- Execute sequentially with one BLAS/OpenMP thread, 8 GiB RSS stop, 120 MiB
  output stop, and 2 GiB free-disk stop.

## Interpretation

- Improvement with \(N\) at fixed amplitude is evidence of sample-limited
  estimation.
- A nonzero error plateau or deterioration, especially when response-score error
  improves, is evidence of objective/representation/readout bias.
- Success of the supplied-basis normalized law remains a structural upper bound.
- These rows must not be pooled with the preregistered fixed-\(\Lambda\) rows.

