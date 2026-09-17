# E12 preregistration: direct versus composed horizons

Status: frozen before the E12 confirmation outcomes were inspected.

This experiment implements Experiment 6 in
`SID_Predictive_Distributional_Dynamics_identification_aware (1).tex`.  It
targets the resolved closure defect

\[
\Delta_{k,\phi}=E\{D[m^{\rm dir}_{k,\phi}(H)-
m^{\rm comp}_{k,\phi}(H)]\},
\]

for the current-state direction, unit history weight, endpoint mean, and the
bounded witnesses `sin(1.0 x)` and `sin(2.0 x)`.

## Frozen cells

| Cell | Data-generating process | Declared history | One-step fit | Expected status |
|---|---|---:|---|---|
| `ar1_closed` | Gaussian AR(1) | 1 | linear Gaussian | closure null |
| `ar2_closed` | Gaussian AR(2) | 2 | linear Gaussian | closure null |
| `ar2_history_short` | same Gaussian AR(2) | 1 | linear Gaussian | longer-memory alternative |
| `nonlinear_closed` | Gaussian-noise sine Markov system | 1 | linear + registered sine feature | closure null |
| `nonlinear_misspecified` | same sine Markov system | 1 | linear Gaussian | one-step misspecification alternative |

The horizons are 2, 4, and 8.  Confirmation uses DGP seeds 6001--6030,
40 independent trajectories per seed, 16 registered anchors per trajectory,
three trajectory folds, 32 common-random-number rollouts, and 99 full
trajectory-bootstrap refits.  The DGP seed is the replication/inferential unit.

## Estimation and calibration

- The direct component is estimated by a cross-fitted outer Riesz-orthogonal
  score using the observed horizon outcome.
- The composed component is the cross-fitted pathwise derivative of recursive
  rollouts from the fitted one-step conditional Gaussian law.
- Basic 95% bootstrap intervals refit both nuisances and the one-step law after
  resampling whole trajectories.  No row bootstrap is used.
- The observed horizon outcome is **not** used to correct the composed side.
  A separately labelled `naive_outer` diagnostic is retained to verify the
  theorem that this invalid construction collapses back to the direct target
  away from closure.
- Oracle targets are analytic for the AR cells and use a fixed, independent,
  high-budget Monte Carlo integration for the nonlinear cells.

## Confirmation gates

The run is complete only if all rows are finite and all 30 seeds are present.
Scientific gates are evaluated on the two primary witnesses (`mean` and
`sin_1.0`):

1. The aggregate false-positive rate across the three closure-null cells and
   all registered horizons is at most 0.10.
2. At horizon 8, rejection power is at least 0.80 separately for
   `ar2_history_short` and `nonlinear_misspecified`.
3. In `ar2_history_short`, the median absolute mean-witness defect at horizon 8
   exceeds that at horizon 2.
4. Across alternative cells, the median absolute difference between the
   deliberately invalid `naive_outer` estimate and the valid direct estimate
   is at most 0.10 times the median absolute direct target plus 0.02.  This is
   an estimator-identity audit, not an estimator for the composed side.
5. Direct-target interval coverage across the analytic AR cells is between
   0.85 and 1.00.

Development seeds and reduced bootstrap/rollout budgets may be used only for
software and numerical audits.  Any change to the cells, primary witnesses,
confirmation seeds, or gates requires a dated amendment before rerunning the
confirmation.
