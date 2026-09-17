# Frozen focused comparison: normalized autoregression versus diffusion

## Scientific question

For continuous conditional laws, when is a normalized autoregressive model the
better instrument than conditional diffusion, and when does diffusion's
generative geometry justify its indirect and more expensive history-tangent
access?

This comparison separates two endpoints:

1. **History-tangent endpoint:** accuracy and cost of estimating
   `grad_h log p(y | h)`.
2. **Generation endpoint:** proper conditional sample quality.

A method may win one endpoint and lose the other. Native NLL, denoising loss,
and classifier loss are never placed on one common scale. No synthetic result
licenses a causal, anatomical, or latent-rewiring claim.

## Repair relative to the earlier smoke test

The earlier M2 model was a compact coordinate-wise autoregressive MDN, not a
Transformer. This comparison adds a causal continuous autoregressive
Transformer with teacher forcing, an exact normalized mixture likelihood,
ancestral sampling, direct likelihood-autograd history tangents, and three
fixed coordinate orders. The MDN remains an architecture control, and a
conditional affine flow remains an ordering-free normalized control.

The primary diffusion candidates are the original VE response-score model and
an EDM-preconditioned response-score model with deterministic Heun sampling.
The development set may select one diffusion candidate for each endpoint. The
selection is frozen before confirmatory generator seeds are run.

## Development/calibration lane

The disjoint development seed is generator `41`, data `301`, and model `1301`.
It uses 5,000 training examples and tests:

- G1 nonlinear location, `d_y=8`;
- G2 covariance only, `d_y=8`;
- G3 fixed-mean/fixed-variance skew mixture, `d_y=1`;
- G4 eight-component low-rank multimodal mixture, `d_y=8`;
- G6 rough location with `omega=8`, `d_y=8`.

Learning rates `1e-4`, `3e-4`, and `1e-3` are selected by mean within-case
native-validation rank. Oracle tangent metrics do not choose the learning rate.
Oracle metrics may choose between the two declared diffusion reconstructions
on this development seed only; that choice is then frozen.

Calibration is adequate to proceed when:

- every case and metric required by its declared capability completes;
- the autoregressive Transformer reaches tangent NRMSE at most `0.75` on G1
  and at most `1.0` on at least two of G2, G3, and G4;
- the selected diffusion route reaches response-score NRMSE at most `0.75` and
  model-centered tangent NRMSE at most `0.90` on G1;
- no selected checkpoint is at the final epoch in more than 25% of cases;
- all selected samplers are finite, and diffusion's G4 fair energy score is no
  more than 50% worse than the median autoregressive-Transformer score.

Demonstrated coding or numerical failures may be repaired and the development
lane rerun. Thresholds and confirmatory decision rules may not be relaxed.

### Logged calibration repair after the first development pass

The first pass showed near-oracle held-out NLL on G2 and G4 while clean tangent
NRMSE remained above one. This demonstrates predictive--derivative
decoupling, not a failed likelihood implementation. Before any confirmatory
seed is run, add a data-only Hyvarinen history-score penalty to the normalized
Transformer. It combines `grad_h log q(y | h)` with the known standard-normal
history marginal score and uses a Hutchinson divergence estimate; it never
uses oracle tangent labels.

Evaluate frozen weights `0.3`, `1.0`, and `3.0` on the same development seed.
Choose the smallest weight that clears the unchanged calibration thresholds;
if none clears, the repair is rejected and confirmation remains blocked.

### Repair outcome (completed 2026-07-13)

All 15 declared cases completed under source digest
`9c6eae998b0e78c157afe509c0eb177075fec26301408bd5529af563ac44dba6`.
No weight cleared the unchanged gate.  Tangent NRMSE for weights
`0.3 / 1.0 / 3.0` was:

- G1: `0.505 / 0.577 / 0.655`;
- G2: `1.359 / 1.388 / 1.694`;
- G3: `1.304 / 1.375 / 1.416`;
- G4: `1.492 / 1.286 / 1.376`;
- G6: `0.691 / 0.715 / 0.704`.

The repair therefore succeeds only on the two location mechanisms, G1 and G6,
and fails every covariance/shape/occupancy mechanism, G2--G4.  It is rejected;
it is not selected for confirmation.  The focused low-dimensional comparison
remains blocked by estimator invalidity and will not be relabeled as a
confirmatory winner comparison.

The complementary G5 generation calibration was frozen before its results
were inspected.  It uses `d_y=64`, latent dimension 4, eight components,
observation noise `0.1`, three Transformer orders, and both declared diffusion
samplers.  Fair energy score is its primary endpoint; high-dimensional tangent
reconstruction is diagnostic only.

The first G5 calibration completed all five cases.  Mean Transformer fair
energy was `1.1830`; EDM was `1.3143` (11.1% worse), and the legacy sampler was
`1.7179` (45.2% worse).  Because every best checkpoint fell between epochs 56
and 59 of 60, this first pass is not treated as a converged architecture
comparison.  A convergence repair was frozen before rerunning: increase only
`max_epochs` from 60 to 150 and patience from 15 to 25, preserving all data,
models, learning rates, evaluation settings, and the primary metric.

Because the low-dimensional gate failed, the full winner-confirmation matrix
will not be run or relabeled.  A narrower post-gate invalidity replication was
instead frozen before inspecting any new seed.  It uses generator seeds
`101, 103, 107, 109, 113`, one paired data seed `401`, one paired model seed
`1401`, 10,000 training examples, all five low-dimensional mechanisms, all
three fixed Transformer orders for vector responses, and the selected legacy
diffusion tangent reconstruction.  This run can establish whether the failed
validity pattern replicates across independently generated systems; it cannot
declare an architecture winner under the frozen decision rule.

### Post-gate replication outcome (completed 2026-07-13)

Both shards completed under the identical source digest
`8846c39a3f9aa3bb529e27fe2261afc77eafe233dac1e94c2dda37a5ad673f1e`:
90 expected and observed cases, 90 unique case IDs, zero failed cases, and zero
failed metric rows.  The primary Transformer result averages coordinate orders
within generator seed before taking the median across the five generator
seeds.  Diffusion uses the implementable model-sample-centered reconstruction
at standardized `sigma=0.05`.

| Mechanism | Transformer median NRMSE | Diffusion median NRMSE | Both valid (<1)? |
| --- | ---: | ---: | --- |
| G1 location | 0.497 | 0.751 | yes |
| G2 covariance | 1.272 | 1.480 | no |
| G3 skew mixture | 1.014 | 3.852 | no |
| G4 multimodal occupancy | 1.065 | 1.030 | no |
| G6 rough location | 0.636 | 0.745 | yes |

Both routes have median NRMSE at least one on three mechanisms.  The frozen
tangent decision is therefore **inconclusive due to estimator invalidity**.
This is not a tie: it means a winner label would be scientifically invalid.
The independently evaluated clean-to-noisy oracle discrepancy is below `0.018`
on every mechanism, so corruption at `sigma=0.05` does not explain these
failures.

The extended G5 calibration also completed all five cases.  Mean Transformer
fair energy is `1.1768`; EDM is `1.3226` (12.4% worse) and legacy diffusion is
`1.7072` (45.1% worse).  Extending from 60 to 150 epochs changes the individual
Transformer scores by less than 0.8%, and the three orders span only
`1.1725--1.1829`.  This development system therefore gives contrary evidence,
not support, for a diffusion generation advantage.  It remains a one-generator
development result and is not relabeled as the frozen multi-seed G5 decision.

## Focused confirmatory matrix

If calibration passes, use ten new generator seeds and two model seeds. The
generator seed is the independent inferential unit; model-seed and coordinate-
order results are averaged within generator seed.

### Low/moderate-dimensional tangent domain

- G1, G2, G3, G4, and G6 as above;
- 10,000 training examples;
- autoregressive Transformer orders `0`, `17`, and `29` for vector outputs;
- the frozen diffusion route;
- the MDN and conditional affine flow as secondary controls;
- 100 maximum epochs and patience 20.

The Transformer primary result is the mean across the three fixed orders, not
the best test order. G3 has one response coordinate and therefore only one
order.

Diffusion mixed derivatives target a Gaussian-corrupted law. The primary
evaluation level is standardized `sigma=0.05`. Clean autoregressive and noisy
diffusion tangent errors may be compared only if the independently reported
oracle clean-to-noisy tangent discrepancy is at most `0.10`; otherwise they
remain separate estimands and no direct winner is declared.

### Difficult-geometry generation domain

- G5 near-manifold mixture, `d_y=64`, latent dimension 4, eight components,
  `sigma_obs=0.10`, and mode separation 4;
- 5,000 training examples;
- the same three Transformer orders and the frozen diffusion generator;
- fair energy score as the primary endpoint.

History-tangent reconstruction at `d_y=64` is diagnostic only. This lane tests
the proposed diffusion generation advantage.

## Frozen decision rules

All comparisons are paired by generator seed.

### A. Normalized-autoregression tangent claim

Call normalized autoregression **supported as the preferred tangent route in
this domain** only when:

- both compared estimators have median tangent NRMSE below `1.0` on at least
  three of the five mechanisms;
- the Transformer has at least 20% lower NRMSE than diffusion on at least four
  mechanisms, or is within 5% while using less than half the tangent-evaluation
  time;
- the paired 95% generator-bootstrap interval excludes a 5% Transformer
  disadvantage; and
- the result is not reversed by the MDN/flow normalized controls.

Call the claim **not supported** when those conditions fail. Call it
**inconclusive due to estimator invalidity** when both routes have median NRMSE
at least `1.0` on three or more mechanisms.

### B. Diffusion generation claim

Call a difficult-geometry diffusion advantage **supported** only when the
selected diffusion generator improves G5 fair energy score by at least 10%
relative to the mean Transformer result, the paired 95% generator-bootstrap
interval excludes no improvement, and the advantage holds against every fixed
Transformer order individually.

Otherwise call the advantage **not supported in the evaluated domain**. This
does not assert that diffusion can never help at larger dimensions or budgets.

### C. Final decision language

The final argument must use one of these scoped forms:

- normalized autoregression is preferred for tangent access, while diffusion
  is preferred for difficult-geometry generation;
- normalized autoregression is preferred for both endpoints in the evaluated
  domain;
- diffusion is preferred for both endpoints in the evaluated domain; or
- the comparison is inconclusive because one or both estimators failed their
  validity conditions.

H1--H7 from the original large plan are not automatically decided by this
focused experiment. Any overlap is reported as scoped supporting or contrary
evidence, not as proof.
