# Distributional SID decisive experiments

This package implements the staged local execution of
`distributional_sid_decisive_experiments_runbook.md`.

The primary estimator targets

\[
A_p = \mathbb E[\partial_{h_p}m(H)],\qquad
m(h)=\mathbb E[\Phi(Y)\mid H=h],
\]

with the cross-fitted score

\[
\psi = \partial_{h_p}\widehat m(H)
      + \widehat\alpha_p(H)\{\Phi(Y)-\widehat m(H)\}.
\]

The first local reference learner is a differentiable random-feature sieve.
Conditional feature regression is multivariate ridge regression in that sieve;
the Riesz representer is fit by the empirical loss specified in the runbook.
This is a small-budget, auditable reference implementation rather than the
future medium-budget neural implementation.

All runs are sequential by default.  The runner forces single-threaded BLAS and
PyTorch execution, checks free disk and resident memory between cells, uses
atomic writes, retains failed rows, and never overwrites a completed seed.

The completion suite is implemented in
`src/distributional_sid/remaining_experiments.py` and adds:

- dependent AR(1) histories with blocked folds and HAC uncertainty;
- heteroskedasticity, missingness, and known/estimated/misspecified
  measurement correction;
- staged dense-geometry ablations;
- normalized MDN/NSF reference grids with internal sampling budgets; and
- a full-covariance MDN with exact analytic characteristic/autodiff readout.

The manuscript's direct-versus-composed horizon experiment is implemented in
`src/distributional_sid/closure_experiment.py`.  Its frozen 30-seed confirmation
is `runs/e12_closure_confirmation_20260716/`; the result is summarized in
`E12_CLOSURE_RESULT_20260716.md`.  Six of seven gates passed.  Closure-null
calibration, nonlinear misspecification power, defect growth, the invalid-score
audit, and AR coverage passed; the short-history AR(2) cell reached 0.70 rather
than the preregistered 0.80 horizon-8 power threshold.

The selected confirmation directories are
`e3_dependent_confirmation_20260714`,
`e8_measurement_confirmation_20260714`,
`e9_staged_confirmation_20260714`,
`e10_full_confirmation_20260714`, and
`e10_analytic_mdn_confirmation_20260714` under `runs/`.
