# Metric validation and falsification record

This document records the conditions under which a metric is allowed into a frozen
scorecard. It describes implemented code and tests; it is not a table of confirmatory
method outcomes. Normative estimands and claim ceilings remain in
[METRIC_CONTRACT.md](METRIC_CONTRACT.md), with theory in [THEORY.md](THEORY.md).

## Admission rule

A metric is reportable only if all four conditions hold:

1. its identifier resolves through the strict machine registry in
   `metric_contract.py` to an estimand, claim level, direction, attainable optimum,
   unit, and role;
2. the method declares the capability needed to produce the estimand;
3. the metric passes its positive, null, and adversarial falsification gates; and
4. uncertainty is computed at the independent seed, episode, or worm level rather
   than at the frame, edge, forecast origin, or Monte Carlo-draw level.

The registry accepts registered terminal statistics and explicitly patterned
lag/operation identifiers. An unknown family, misspelled statistic, doubled separator,
or semantically incompatible suffix raises an error before tidy export. A missing
capability is recorded as not applicable; it is never converted to a numerical loss.

## State-field gates

For a nonzero oracle field (J(s)), the primary error is occupied-state nISE,

\[
\operatorname{nISE}(\widehat J,J)=
\frac{E_\nu\|\widehat J(s)-J(s)\|_F^2}
     {E_\nu\|J(s)\|_F^2}.
\]

Truth and estimate must be evaluated on exactly the same registered held-out states.
The required falsifications are:

- an exact oracle has numerical-zero nISE, RMSE, and RMS-map error;
- a wrong pointwise field is penalized even when equal-and-opposite values make its
  average response correct;
- a structurally null truth does not receive an epsilon-stabilized relative score:
  nISE and cosine are undefined, while estimate RMS, maximum absolute response,
  RMS-map error, and false support remain defined;
- fixed-tail and standardized-shape thresholds in method metadata must match the DGP
  registration; a mismatch is an error, not a sensitivity result;
- finite-difference results must be stable across the registered step check and agree
  with analytic oracle derivatives within tolerance.

These gates are exercised primarily in `tests/test_state_fields.py`,
`tests/test_capabilities.py`, `tests/test_mechanistic.py`,
`tests/test_gated_response.py`, and `tests/test_operator_probes.py`.

## Distribution and channel-orthogonality gates

The following constructions distinguish law recovery from moment recovery:

- the matched-tail mixture changes standardized two-sided tail probability and fourth
  moment while matching conditional mean and variance exactly;
- pure correlation routing changes off-diagonal covariance/correlation while leaving
  every marginal mean, variance, fixed-tail, and standardized-shape functional
  invariant;
- additive, synaptic, and intrinsic mechanisms enter conditional mean but not the
  direct dispersion or standardized-shape channels;
- stochastic dispersion enters log variance while leaving the one-step conditional
  mean unchanged;
- a null mechanism has zero physical response in every registered channel.

An estimator that succeeds through leakage into the wrong channel fails mechanism
recovery even if a broad law score improves. Tests live in
`tests/test_metric_channel_orthogonality.py`, `tests/test_mechanistic.py`,
`tests/test_correlation_routing.py`, and `tests/test_intervention_evaluation.py`.

Normalized-law evaluation additionally requires the method's own log density and CDF,
or an explicit Gaussian-family declaration. Non-Gaussian forecasts cannot silently use
a Gaussian moment CDF. The fair energy and CRPS implementations remove self-pairs; their
true-law expected optima depend on the outcome distribution and are not generally zero.

## Response-kernel gates

Every registered response pair has bitwise-identical pre-operation state and uses the
same named innovations in both arms. Pair index (b) also reuses the same baseline and
exogenous fingerprint across operations, so cross-operation comparisons stay paired.
The grid includes positive operations and expected-null shams.

Before scoring an operation:

- an active operation must exceed the dimensionless beta-min
  `truth RMS / baseline innovation RMS`;
- an expected-null operation must remain below the registered null tolerance;
- an active truth's Monte Carlo-SE RMS divided by truth RMS must fall below the
  preregistered precision ceiling;
- a transient active truth must meet the preregistered minimum fraction of outputs with
  an observable post-peak (1/e) crossing inside the response window;
- receptor and release-source selectors must choose an edge that reaches the response
  functional being scored, not merely an expressed receptor;
- an explicit intervention is C1 only for one modulator or when the model declares an
  anchored modulator-label mapping;
- the Monte Carlo truth reports its kernel standard-error RMS and its ratio to truth
  RMS;
- history-conditional pairwise nRMSE reports both the fraction and the count for which
  its nonzero-truth denominator is defined.

Population-mean and history-conditional arm-history metrics are kept separate. Full
post-branch arm-history trajectories are P1 for passive predictors and secondary,
represented-operation C1 for an explicit model. The primary controlled C1 comparison is
only lag one, where the two predictions condition on the same realized branch-point
history. A full recursively controlled model kernel is not yet available.

Response summaries are deliberately complementary. The signed finite-window sum times
the sample interval is a windowed cumulative response, not an infinite-horizon effect. A
true peak at the final
lag is right-censored and excluded from peak-time error. The first post-peak (1/e)
crossing is scored only when the true transient crosses inside the window; missed
estimated crossings are a censoring mismatch. Persistent knockout or source-silencing
operations do not receive a return-to-zero score and instead receive an explicitly
window-limited end-window gain. Tests are in `tests/test_intervention_response.py`,
`tests/test_response_config.py`, and `tests/test_response_evaluation.py`.

The beta-min, truth MC-SE ceiling, and minimum transient-crossing fraction are
ground-truth precision/applicability preflights. They determine whether the registered
operation can test the intended property; they are not model-performance metrics.

## Rollout and bridge gates

Rollout inputs must have shapes `[case, time, target]` and
`[case, draw, time, target]`. Each draw is a joint trajectory; independent one-step
samples pasted across time violate the contract. The principal adversarial test uses
independent oracle AR-path draws—not leaked copies of the observations—and verifies that
they beat forecasts with the right marginals but wrong temporal dependence. Because
independent finite samples have sampling error, even an oracle forecast need not attain
zero empirical energy, autocovariance error, or spectral error. Temporal permutation,
unstable escape, and wrong extreme frequency are separate positive controls.

Overlapping blocks or multiple origins from one episode do not create independent
replicates. Any interval or paired comparison clusters at the episode/worm level. These
gates are exercised in `tests/test_rollout_metrics.py`.

The bridge adapter is currently the only method that supplies the generic runner with
joint recursive paths. Its fair time-marginal and finite-block energy scores measure
forecast accuracy against observed paths and have distribution-dependent optima. The
Brownian `reference_path_kl_mean`, by contrast, measures control effort relative to a
chosen reference; it is directionless and diagnostic, so smaller is not automatically a
better forecast. `tests/test_bridge.py` and `tests/test_reporting.py` enforce the split.

## Leakage, split, and identifiability gates

- Features, tuning, early stopping, and latent alignment use grouped train/validation
  worms; test targets cannot affect predictions.
- Response arms, observation variants, and repeated origins retain their pairing in
  uncertainty calculations.
- Latent component assignment and affine calibration are fitted on validation worms and
  frozen on test worms.
- Concentration/(K_d) scaling and expression/effect scaling are explicit invariances;
  raw parameter errors cannot be promoted to physical identification.
- Changepoint nuisance fitting, null calibration, and final FPR evaluation use disjoint
  streams and calibrate the maximized scan statistic.

Relevant checks are in `tests/test_features_metrics.py`, `tests/test_hierarchy.py`,
`tests/test_identifiability.py`, `tests/test_mechanistic_latent.py`, and
`tests/test_changepoint.py`.

## Remaining boundaries

The validated components do not establish performance on a frozen confirmatory suite.
Generic recursive path generation remains unavailable outside the bridge adapter, and a
multi-lag recursively controlled intervention kernel remains unimplemented. The current
tests validate the metrics for the bounded discrete-time simulator subclass; they do not
prove adequacy for arbitrary jump processes, an uncalibrated biological transmitter
system, or latent anatomical rewiring.
