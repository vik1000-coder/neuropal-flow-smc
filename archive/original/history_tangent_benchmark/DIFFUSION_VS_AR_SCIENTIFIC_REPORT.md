# Diffusion versus autoregression for conditional history tangents

## Technical summary

The scientifically defensible conclusion is **not that one architecture wins
universally**.  In the tested low-dimensional domain, the frozen tangent
comparison is **inconclusive due to estimator invalidity**: the normalized
autoregressive Transformer and the implementable diffusion reconstruction both
have median tangent NRMSE at least 1 on G2 covariance, G3 skew mixture, and G4
multimodal occupancy.  Both are valid only on the two location mechanisms, G1
and G6.

That negative decision still yields a clear mechanism-level result.  When
history changes conditional location, direct differentiation of a normalized
Transformer likelihood is accurate, robust to three fixed coordinate orders,
and 7--20 times cheaper at tangent evaluation than the diffusion
reconstruction.  When history changes covariance, higher shape, or mixture
occupancy, excellent predictive density or response-score fit is not enough:
both routes can return an unusable history derivative.

The reason is structural.  A normalized autoregressive model directly exposes
`grad_h log q(y | h)`, but maximum likelihood does not constrain that derivative
strongly enough.  A standard diffusion model learns `grad_y log p_sigma(y | h)`,
not the history tangent; recovering the latter requires mixed integration and
an otherwise unidentified history-dependent centering constant.  In these
experiments, model-sample centering inherited sampler error and materially
worsened the implementable diffusion tangent.

The difficult-geometry G5 development result also does not support a diffusion
quality advantage.  After extending training from 60 to 150 epochs,
Transformer mean fair energy was 1.1768, EDM diffusion was 1.3226 (12.4% worse),
and legacy diffusion was 1.7072 (45.1% worse).  Diffusion sampled roughly 250
times faster, however.  This G5 result is one development system, not the frozen
multi-seed generation confirmation, so it is scoped contrary evidence rather
than a general rejection of diffusion generation.

## The replicated result is a location-versus-shape split

NRMSE is normalized by the RMS size of the oracle tangent.  NRMSE below 1 means
the estimator beats the zero-tangent baseline; this was the frozen validity
threshold.  Transformer values average the three coordinate orders within each
generator seed before taking the median across five independently generated
systems.  G3 is scalar and has one order.  Diffusion values use the implementable
model-sample-centered reconstruction at standardized noise 0.05.

| Mechanism | Transformer median (95% bootstrap CI) | Diffusion median (95% bootstrap CI) | Frozen validity result |
| --- | ---: | ---: | --- |
| G1 location | 0.497 (0.448, 0.515) | 0.751 (0.703, 0.815) | both valid |
| G2 covariance | 1.272 (1.130, 1.480) | 1.480 (1.443, 1.535) | both invalid |
| G3 skew mixture | 1.014 (0.809, 1.125) | 3.852 (2.642, 5.006) | both invalid |
| G4 multimodal occupancy | 1.065 (0.985, 1.308) | 1.030 (0.959, 1.057) | both invalid |
| G6 rough location | 0.636 (0.581, 0.729) | 0.745 (0.707, 0.783) | both valid |

The paired generator-bootstrap intervals for the mean relative Transformer
error exclude parity on every mechanism: the Transformer is 32.8--38.5% lower
on G1, 4.3--18.0% lower on G2, 69.4--75.8% lower on G3, and 8.8--17.7% lower on
G6; it is 2.1--19.7% higher on G4.  Those relative rankings do not override the
validity gate.  A method being less wrong than another method is not evidence
that its tangent is usable.

Coordinate ordering is not driving the main result.  Across vector mechanisms,
the median within-seed standard deviation over the three Transformer orders is
0.016 on G1, 0.047 on G2, 0.068 on G4, and 0.021 on G6.

## Predictive fit does not validate derivative fit

The Transformer density fit is close to the exact oracle likelihood on the
same held-out samples even where its tangent fails.  Median NLL per response
vector was 9.040 versus oracle 8.868 on G1, 8.836 versus 8.872 on G2, 1.313
versus 1.380 on G3, 13.376 versus 12.952 on G4, and 11.385 versus 11.369 on G6.
The small negative differences are finite-test-sample fluctuation; the largest
positive excess is 0.424 nats per vector on G4.  These are not grossly failed
density models, yet G2--G4 derivative recovery does not clear the baseline.

The diffusion diagnostic is even more direct.  Median response-score NRMSE is
0.357 on G1, 0.260 on G2, 0.495 on G3, 0.327 on G4, and 0.233 on G6.  Thus the
network can estimate the noisy response score reasonably well while the
model-centered history-tangent NRMSE is 1.480, 3.852, and 1.030 on G2--G4.
A standard diffusion response score is therefore not an interchangeable
history tangent.

The known corruption gap is not the explanation: median oracle clean-to-noisy
tangent discrepancy is at most 0.0176 on every mechanism, far below the frozen
0.10 comparability threshold.

## Centering is part of the diffusion estimator, not a cosmetic correction

The mixed response derivative identifies the history tangent only up to a
history-dependent constant in response space.  With oracle samples used only
for diagnostic centering, diffusion tangent NRMSE is 0.547, 1.002, 3.207,
0.865, and 0.617 on G1, G2, G3, G4, and G6.  Replacing those oracle samples with
the model's own samples changes the values to 0.751, 1.480, 3.852, 1.030, and
0.745.  In particular, the apparent G4 validity disappears in the implementable
route.

This shows why the generative and tangent endpoints cannot be collapsed.  A
diffusion response-score model may be locally accurate, while an imperfect
sampler corrupts the normalization step needed for the history derivative.

## Cost and generation expose different architectural tradeoffs

For history-tangent evaluation, the normalized Transformer is cheaper because
one reverse-mode derivative of an exact log likelihood is sufficient.  Median
diffusion-to-Transformer time ratios are 10.2 on G1, 8.6 on G2, 20.1 on scalar
G3, 7.0 on G4, and 7.4 on G6.

For high-dimensional sampling the direction reverses.  On 64-dimensional G5,
the three Transformer orders produce 83.9--86.1 samples per second because
coordinates are generated sequentially.  EDM produces about 21,616 samples per
second and legacy diffusion about 11,689 because each denoising step updates the
whole vector in parallel.  The faster samplers have worse fair energy in this
development system:

| G5 model | Fair energy (lower is better) | Relative to mean Transformer | Best / stopped epoch |
| --- | ---: | ---: | ---: |
| Transformer, order 0 | 1.1829 | +0.5% | 134 / 149 |
| Transformer, order 17 | 1.1750 | -0.2% | 117 / 142 |
| Transformer, order 29 | 1.1725 | -0.4% | 133 / 149 |
| EDM diffusion | 1.3226 | +12.4% | 146 / 149 |
| Legacy diffusion | 1.7072 | +45.1% | 149 / 149 |

Extending the budget changed individual Transformer energy scores by less than
0.8% and left their order ranking tightly clustered.  The two diffusion scores
also remained close to their 60-epoch values.  This makes the directional G5
development result credible for this system, but its single generator seed
precludes the frozen multi-seed generation claim.

## Experimental design and validation

The development lane used generator seed 41, data seed 301, model seed 1301,
5,000 training cases, and learning rates chosen only by native validation rank.
The data-only Hyvarinen history-score repair tested frozen weights 0.3, 1.0,
and 3.0 and failed the unchanged gate on every shape-changing mechanism.  It was
rejected before replication.

The post-gate replication used five new generator seeds (101, 103, 107, 109,
113), paired data seed 401, paired model seed 1401, 10,000 training examples,
100 maximum epochs, and three fixed output orders for vector Transformers.  The
generator seed is the inferential unit.  Confidence intervals enumerate all
5^5 ordinary bootstrap resamples, so they have no Monte Carlo jitter.

The replication data-quality audit found 90 expected and observed cases, 90
unique case IDs, zero failed cases, zero failed metric rows, exact seed
coverage, and one shared source and environment digest across both parallel
shards.  All 76 repository tests pass.  The analysis is reproducible with
`scripts/summarize_diffusion_vs_ar.py` and its saved seed-level, mechanism-level,
generation, and validation outputs.

## Limitations and claim boundary

- The replication has five independent generator systems but one paired data
  seed and one model initialization per system.  Coordinate-order variation is
  measured; model-seed variation is not.
- Diffusion tangents target the Gaussian-corrupted law at standardized noise
  0.05.  The oracle corruption ledger shows that this mismatch is negligible in
  the evaluated domain, but the estimand labels remain separate.
- The implementable diffusion tangent uses the selected legacy score model.
  EDM failed tangent calibration and is used only as a generation candidate.
- G5 generation has one development generator system.  It cannot support or
  reject the preregistered multi-seed H2 generation claim.
- These are exact synthetic conditional laws.  Nothing here supports a causal,
  anatomical, latent-rewiring, or biological-mechanism claim.

Overall validation status is **share with caveats**: the estimator-invalidity
conclusion and the location-versus-shape split are well supported in this
benchmark; the high-dimensional generation result is directional development
evidence only.

## Recommended next step

Do not spend the next budget merely adding seeds to the same invalid tangent
estimators; that would narrow uncertainty around a failed validity condition.
The next scientifically useful experiment is to add a tangent-specific route
that can pass G2--G4 calibration without oracle tangent labels—for example a
properly normalized conditional energy/ratio estimator or a diffusion model
with an explicit normalized history-ratio head—and only then run the frozen
10-generator, two-model-seed confirmation.

For current method selection:

1. Use normalized autoregression first for low/moderate-dimensional
   location-like history effects when exact likelihood and direct tangent access
   matter.
2. Treat held-out NLL and diffusion response-score error as necessary but not
   sufficient diagnostics; validate the history derivative directly on
   synthetic or otherwise known controls.
3. Use diffusion as a fast parallel conditional sampler only when its proper
   sample score clears the application threshold.  The tested G5 system shows
   a large throughput advantage but no quality advantage.
4. For covariance, skewness, or regime-occupancy questions, do not interpret
   either current tangent estimator scientifically until a derivative-specific
   calibration gate passes.

## Further questions

- Can a normalized ratio or energy head recover G2--G4 tangents while preserving
  Transformer density quality?
- Can diffusion centering be learned or normalized without relying on samples
  from a biased sampler?
- Does diffusion's sampling-throughput advantage become a proper-score advantage
  at larger response dimension, larger capacity, or a broader sampler budget?
- How much model-initialization variance remains after generator-seed and
  coordinate-order averaging?
