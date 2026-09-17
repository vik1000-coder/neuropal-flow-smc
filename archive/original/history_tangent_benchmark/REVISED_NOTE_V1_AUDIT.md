# Revised-note V1 pre-execution audit

This audit freezes the first executable test of
`SBTG_Revised_Technical_Note_July_2026.pdf`.  It is a developmental
falsification/correctness run, not a confirmatory biological or H1--H7 result.

## Decision

The note's central finite-contrast construction is mathematically coherent and
worth testing.  For an anchor history `h`, physical direction `v`, and scale
`delta`, the target is the central signed contrast under

```text
h_plus  = h + delta * v
h_minus = h - delta * v
M_delta = 0.5 * (P_plus + P_minus)
```

with bounded witness

```text
W_delta(y) = tanh((log p_plus(y) - log p_minus(y)) / 2) / delta.
```

This target is not the one-sided `h + delta` versus `h` log ratio in the
original developmental runner.  It is also not identical to the infinitesimal
history tangent at nonzero `delta`.

The first run implements only the load-bearing V1 components:

- the exact oracle central witness and direct signed channel contrasts;
- the existing fitted history tangent evaluated on the same finite target;
- the central log-ratio/tanh route for normalized models and the existing
  joint-versus-product critic;
- an amortized balanced signed-mixture classifier;
- a finite response-dictionary signed-Riesz projection;
- clean and matched positive-response-noise lanes kept as separate strata;
- a history-scale ladder.

Hodge projection, hybrid score teachers, masked scores, orthogonal time-series
inference, foundation-model transfer, and biological data are out of scope for
this gate.  A small V2 parity experiment may follow only after V1 code and
metrics pass.

## Corrections and guardrails

1. Equation (3.11)'s standard-deviation readout bound requires the fitted
   witness error to be centered.  Classifier calibration therefore uses a
   separate balanced mixture split and fits an affine logit calibration.  Raw
   and held-out centering residuals remain visible.  No centered-error bound is
   claimed for a misspecified model under the oracle mixture.
2. The logistic witness bound uses excess risk, not raw cross-entropy.  Bayes
   cross-entropy is computed only on synthetic oracle panels where `p_plus` and
   `p_minus` are known.
3. Every comparison row is keyed by `delta`, response noise, target law,
   reference law, channel dictionary, predictive source, and post-processor.
   Clean and noisy rows are never pooled.
4. Existing infinitesimal tangents are evaluated against the finite witness
   only as an explicitly scale-mismatched baseline.  The oracle infinitesimal
   tangent supplies the irreducible Taylor/reference-law gap at each scale.
5. The regularized Riesz witness norm is `a.T @ G @ a`.  The larger quantity
   `d.T @ a`, which includes the penalty contribution, is not reported as
   restricted information.
6. Response features are fixed globally from the training split.  They do not
   recenter on the evaluation history, so no omitted history derivative of the
   channel is introduced.
7. The classifier receives the anchor history and response.  It never receives
   the perturbed history, stencil sign, or any label-revealing feature.
8. Test panels are independent of adapter training and calibration panels.
9. Gaussian DGP histories have full support, but perturbed-history Mahalanobis
   radii are still recorded.  No causal or anatomical interpretation is made.
10. The current V1 matrix has one generator/data/model seed and is therefore a
    plumbing and comparative-development run.  At least ten independent
    generator seeds and a newly frozen primary endpoint are required before a
    confirmatory claim.

## Frozen developmental matrix

- DGPs: G1 nonlinear mean, G2 variance-only, G3 fixed-mean/fixed-variance skew
  mixture, and G4 low-rank multimodal mixture.
- Existing fitted sources: mean MLP, heteroscedastic Gaussian,
  autoregressive MDN, joint-versus-product ratio critic, and conditional
  diffusion.
- History scales: `0.05`, `0.10`, `0.20`, and `0.40` covariance-standardized
  direction units.
- Response-noise scales: clean (`0`) and standardized Gaussian noise (`0.10`).
  The diffusion comparison is made only in the positive-noise lane.
- Fixed channel dictionary: linear, centered quadratic, cross-product, cubic
  Hermite-style, and smooth-tail response channels.
- Primary developmental metrics: finite-witness NRMSE and channel-readout
  NRMSE on the independent oracle mixture.
- Diagnostics: absolute witness RMSE, cosine, centering, information error,
  bound use, classifier cross-entropy/Brier/ECE/oracle excess risk, Riesz Gram
  conditioning, direct-versus-witness agreement, runtime, and support radius.

The run may rank methods only within the same DGP, law, scale, response
dictionary, and evaluation panel.  Aggregate summaries are descriptive.
