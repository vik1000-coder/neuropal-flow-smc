# Pre-execution audit and frozen developmental scope

The supplied experiment plan asks an important and well-posed scientific
question: when should a normalized conditional density, a direct interaction
ratio, or a diffusion model be used to estimate
`grad_h log p(y | h)`?  Its central estimand, the joint-versus-product ratio
identity, the G1--G4 oracle tangents, centered mixed-derivative reconstruction,
and centered energy identity are sound under their regularity assumptions.

The plan is preserved in `/Users/vik/Downloads/` and was reviewed before any
benchmark output was inspected.  The following repairs are required before a
confirmatory H1--H7 run.

## Mathematical repairs

1. **History-dependent readout statistics.**  The identity
   `D_v E[phi(Y)|h] = Cov(phi(Y), t_v(Y;h)|h)` assumes `phi` is fixed with
   respect to `h`.  The proposed variance, skew, moving-quantile tail, and
   posterior-responsibility statistics depend on `h`.  Their total derivative
   also contains `E[D_v phi_h(Y)]`.  Confirmatory work must freeze each statistic
   at a reference history, add the direct term, or use a history-independent
   target such as a latent component indicator.
2. **History-direction scaling.**  A one-standard-deviation displacement uses
   `v.T @ Sigma_H^{-1} @ v = 1` (equivalently `v = Sigma_H^{1/2} u`), not
   `v.T @ Sigma_H @ v = 1`.  The original rule can create enormous nominally
   orthogonal G7 perturbations when `tau` is small.
3. **Finite contrasts.**  Estimation error for a finite log ratio and Taylor
   truncation error of `delta * t_v` are different objects.  H5 must compare
   estimators on the same finite target and report Taylor error separately;
   relative errors need an absolute-error companion near zero.
4. **H3 negative control.**  The uncentered path reconstruction equals the true
   tangent minus its value at the response reference.  That offset can happen
   to be small.  The structural identity and reference invariance after
   centering are the reliable tests, not a universal numeric lower bound.
5. **Fisher terminology.**  `E_p[hat ell hat ell.T]` is plug-in geometry under
   the oracle data law.  A fitted-model Fisher matrix uses expectation under
   `q`; both should be reported with different metric identifiers.

## Missing preregistration details

- Gaussian corruption and a log-uniform sigma distribution do not specify a
  diffusion sampler or probability-flow likelihood.  A confirmatory protocol
  must freeze the SDE, `sigma(t)`, terminal law/correction, solver, tolerance,
  and whether every comparator targets the clean or the same corrupted law.
- G4 interaction magnitude and bias, G5 mixture/latent scales, G7's intrinsic
  generator, G8 entropy/architecture scales, and G9's stable process and split
  gaps are not fully specified.
- H4 requests low-sample near-manifold evidence that is absent from the stated
  run matrix; H5 depends on G7 even though G7 is absent from the minimum
  completion list.
- “Best method” and metric-OR rules allow oracle-test selection.  Each
  hypothesis needs one frozen primary comparator, metric, pass algorithm, and
  inferential test before Holm adjustment.
- Generator, data, model, and evaluation seeds must be independent axes.
  Overlapping G9 windows require trajectory-held-out or gap-separated splits.

## Logged implementation repairs

- Report both the plan's V-statistic energy score and the fair off-diagonal
  U-statistic under distinct names; use the fair score for future primary tests.
- Remove the unrestricted history-only term from the developmental ratio
  critic because its gradient is not identified by joint-versus-product
  classification.  The critic is explicitly interaction-only.
- Freeze G4's omitted interaction magnitude by scaling the component-logit
  rows to a declared RMS standard deviation under `H ~ N(0,I)`; the
  developmental configuration uses `interaction_scale: 0.75`.
- Compare M5b with the exact tangent of the *same Gaussian-corrupted law*.
  Report oracle-sample centering and approximate model-sample centering
  separately.  The approximate annealed-Langevin sampler is not relabeled as a
  probability-flow likelihood route.
- Fit scalers on training data only and transform history tangents with the
  exact chain rule back to oracle coordinates.
- Store source/config/environment digests because the workspace root is not a
  Git repository.

## Resource audit

The core tier has 74 generator configurations.  Five generator seeds, three
model seeds, and roughly seven families imply about 7,500 base fits and over
22,000 fits with the three learning-rate candidates, before the stress tier.
The requested evaluation can require 10.24 million conditional draws per
checkpoint.  This is not credible on the available 16 GiB, MPS-only machine
with roughly 5.4 GiB free disk.

## Frozen developmental run

The first executable lane is therefore a correctness and cost gate, not a
hypothesis test:

- G1 (`d_y=4`), G2 (`d_y=4`, mean effect off), G3 (`d_y=1`), and G4
  (`K=4`, `d_y=4`, separation 4), with fixed parameters saved in each case
  record;
- one generator seed, one independent data seed, and one model seed;
- 2,000 training pairs, an independent validation split, and a small shared
  test/centering panel;
- M1 heteroscedastic Gaussian, M2 eight-component output-autoregressive MDN,
  M4 interaction-only ratio critic, and M5 conditional response score with
  centered mixed reconstruction at a declared matched noise level;
- exact float64 oracle, tangent, centering, finite-ratio, native-objective,
  energy-score, classifier, timing, parameter-count, and provenance checks.

No H1--H7 decision may be emitted from this developmental lane.  Passing it
only licenses freezing a repaired reduced confirmatory matrix with at least ten
independent generator seeds.
