# Preregistration: compact decisive Tier 1 SID validation

**Frozen:** 2026-07-14, before inspecting any new confirmatory seed results.  
**Runbook:** `/Users/vik/Downloads/SID_EMPIRICAL_VALIDATION_RUNBOOK.md`, SHA-256 `2035e75d90dd1dd6911693d09ecaa0880cce5baaf7467e7d7f25076798bd9ea7`.  
**Primary audience:** technical/methods.  
**Status:** compact confirmatory Tier 1 plus a separately labeled developmental legacy lane.

## Decision

Determine which route recovers a declared local derivative (D_vT(P_h)) and which route recovers a declared finite contrast 
\(\Delta_{T,\delta}\), without mixing those estimands. The primary scientific opportunity is a mean-blind distributional direction: variance, covariance, third cumulant, tail, occupancy, or quartic/path structure changes while the mean derivative is zero.

## Frozen hypotheses

The primary hypotheses are H1-H7 in the attached runbook. In brief: downstream derivative error may decouple from response-score error; a ratio critic may be more reliable for the clean history tangent than outcome-score differentiation; stabilization may help score-derived SID; Hodge recovery should depend on support topology; finite methods should be judged at the registered finite resolution; difficulty should depend on information and geometry rather than polynomial order alone; and D2 should expose a mean-blind distributional lag profile.

Every hypothesis may be rejected. Native score or denoising loss is never evidence of typed SID recovery.

## Frozen primary cells

- Static mechanisms: M1 mean, pure M2 covariance, four independently constructed and nuisance-matched M3 skew tilts, M4 quartic, M4 tail, and M5 occupancy.
- Response dimension: 8. M3-M5 act on a declared scalar projection and include seven independent nuisance coordinates.
- Training size: 8,000; validation and calibration: 1,600 each.
- Primary finite resolution: \(\delta=0.12\). The full delta ladder is exploratory/developmental except where explicitly marked.
- Primary response smoothing: \(\sigma=0.12\). The sigma ladder is a developmental objective diagnostic.
- Dynamic mechanisms: D1 Gaussian mean-lag anchor and D2 observed stochastic gain with six registered lags.
- Topology: connected through exactly disconnected bridge-mass ladder.

## Frozen methods and access labels

Local leaderboard: zero effect, direct raw-moment regression, normalized conditional density differentiation, joint-versus-product ratio critic, clean Hyvarinen score plus Hodge reconstruction, multi-noise DSM plus Hodge reconstruction, constrained-Gaussian DSM plus centering, and anchored residual-energy DSM plus centering. Original SBTG/mixed-field magnitude is reported only as an operator/localization comparator.

Finite leaderboard: zero effect, direct conditional endpoint moments, finite differences of observational moment regression, fitted normalized-law endpoints, calibrated balanced endpoint classifier, signed Riesz projection, and score-integrated endpoints. Endpoint-query and endpoint-label methods are explicitly marked as stronger-access methods.

The compact score implementations are smooth polynomial/energy estimators with convex or deterministic fits. The earlier neural EDM, anchored EDM, flow, MDN, and ratio-critic results are audited as developmental evidence and are not pooled into confirmatory intervals.

## Frozen metrics

- Primary typed metric: DGP-seed-level NRMSE, with NRMSE 1 equal to the zero-effect predictor.
- Secondary typed metrics: RMSE, signed correlation, slope through origin, sign accuracy, integrated absolute error, and response on nuisance channels.
- Operator metrics: response-score, mixed-field, and history-tangent RMSE/NRMSE; centering error; Hodge residual; derivative amplification.
- Lag metrics: complete-vector NRMSE, signed correlation, peak-lag error, center-of-mass error, slow-mass fraction, and mean-null leakage.
- Efficiency: wall time, failure rate, peak RSS, output bytes, and access regime.

## Seeds and inference

Development seeds are 1001-1008 and may be used only for amplitude checks, hyperparameter screening, and numerical stabilization. Confirmatory seeds are 2001-2030. Methods share DGP/data seeds. Deterministic convex fits use one fit per DGP seed; this is a declared compute deviation from the runbook's three neural initializations, not hidden pseudo-replication.

For confirmatory comparisons, nested rows are averaged within DGP seed. Confidence intervals use a paired DGP-seed bootstrap with 2,000 resamples. No histories, endpoint draws, or adapter rows are treated as independent inferential replicates.

## Recovery and advantage rules

A channel is recovered only if the upper 95% CI for NRMSE is below 1, the primary-regime slope is in [0.8, 1.2], sign accuracy exceeds its registered null, nuisance false response is below 10%, and no support/calibration/leakage gate fails. Superiority requires an upper 95% CI below 0.90 for the paired error ratio to the strongest same-access baseline and replication across at least two constructions.

## Gate and stopping rules

1. Stage 0 identities and leakage tests must pass before any learned estimator is interpreted.
2. A channel that fails with oracle inputs does not advance.
3. A score route that fails score/tangent operator recovery is retained as a negative result but does not trigger extra capacity.
4. D3-D5 and all Tier 2/3 grids remain blocked unless D1/D2 pass their primary gates.
5. NaN/Inf, non-common seeds, or test exposure invalidate the affected comparison.
6. The run stops safely if free disk falls below 2.0 GiB, output exceeds 0.15 GiB, RSS exceeds 8 GiB, or more than 10% of a method's confirmatory fits fail.

## Declared limitations of this local execution

The machine has 16 GiB RAM and only about 2.4 GiB free disk at freeze time. Therefore the program preserves 30 top-level seeds and narrows method/cell breadth rather than reducing inferential replication. It cannot, in one local pass, constitute the runbook's eventual high-dimensional neural architecture, dimension, long-memory, hidden-state, observation-filter, or foundation-model validation. If any completion item is unmet, the final artifact will be titled a diagnostic empirical report rather than a validated general claim.

