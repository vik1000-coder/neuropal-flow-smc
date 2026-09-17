# Frozen local execution specification

Frozen before any result from the new estimator was inspected.

## Scientific order

1. E0 algebra, software, split, and artifact gates.
2. Development E1 analytic Gaussian reference and E2 controlled nuisance
   perturbations.
3. Stop if E2 does not display the exact second-order remainder.
4. If the gate passes, freeze the selected sieve capacity and regularization on
   development seeds only.
5. Run the explicitly enumerated confirmatory cells; all additions prompted by
   results use unused post-hoc seeds and an amendment.

## Seed registries

- development: 1--8;
- confirmation: 1001--1030;
- null confirmation: 2001--2100;
- post-hoc: 3001 onward.

## First executable milestone

- E0: all runbook tests that apply to the local reference implementation;
- E1: analytic nonlinear Gaussian conditional path, one future per history,
  `n=8000`, `d_H=8`, `d_Y=4`, one horizon, `Q=64`, `R=1`, five folds;
- E2: all three declared nuisance perturbation shapes and the complete
  `a,b` ladder;
- methods: ZERO, PLUG, RIESZ, ORTH, OR-A, OR-M, OR-R, oracle truth, and
  target-specific DIRECT mean/covariance heads;
- two development DGP seeds and three nuisance initializations;
- O0 for learned methods; oracle nuisance variants are controls.

## Local implementation adaptation

The first nuisance learner is a differentiable random-feature sieve containing
an intercept, linear terms, centered quadratic terms, and frozen tanh random
features.  Regression and Riesz coefficients are solved by ridge-stabilized
linear algebra.  This choice is made for auditability and CPU safety; it does
not count as the runbook's medium-budget neural, MDN, or flow comparison.

Development selection grid:

- random tanh width: 32, 64, 128;
- feature ridge: 1e-4, 1e-3;
- Riesz ridge: 1e-4, 1e-3, 1e-2.

Selection uses feature prediction error, derivative error, and held-out Riesz
probe residuals jointly.  Confirmation uses one frozen setting.

## Resource envelope

- sequential cells only;
- one BLAS/OpenMP/PyTorch thread;
- hard runner refusal below 1.0 GiB free disk;
- hard runner refusal above 10 GiB process RSS;
- compact vector-valued Parquet/NPZ score storage;
- no checkpoint accumulation after a validated closed-form fit;
- temporary outputs are atomically renamed.

## Claim boundary

This milestone can validate implementation correctness and the small-budget
projected estimator.  It cannot establish equal-compute superiority over the
runbook's flexible MDN/flow baselines or Tier B temporal/latent claims.

