# Mechanistic neuromodulator changepoint lane

## Scope and claim boundary

This lane asks whether an observed stochastic trajectory contains one change in a
declared aspect of its law. It does **not** treat changepoint detection as connectome
recovery, anatomical rewiring, or recovery of an unobserved neuromodulator state.

The simulator begins with a stable null period and switches once at the named boundary
`neuromodulator_onset`. Every scenario uses the same realized structural parameters,
continuous latent state, and null transition law before the boundary. Independent
no-change trajectories—not rows shuffled within a trajectory—calibrate each detector.

The four channel-level claims are deliberately separate:

| Claim axis | Detector | Meaning |
| --- | --- | --- |
| Marginal activity mean | `mean_cusum` | Observed mean/activity changed |
| Conditional mean rule | `conditional_rule_chow` | A ridge history-to-next-state rule changed |
| Residual reliability/dispersion | `variance_reliability` | Frozen observed-history forecast residual scale changed |
| Conditional residual tail shape | `residual_tail_shape` | Location/scale-normalized residual shape or tails changed |
| General distribution law | `energy_distance`, `mmd_rbf` | A broad distributional change occurred; channel is not identified |

The residual claims are reduced-form observed-history claims. A lagged ridge model does
not observe concentration or receptor occupancy and is not the complete-state oracle.

## Mechanism construction

Write the complete-state one-step neural law schematically as

\[
X_{t+1}=m_{\theta_t}(X_t,C_t)+\sqrt{v_{\theta_t}(X_t,C_t)}\,\varepsilon_{t+1},
\]

where concentration/receptor dynamics are part of the living state and
`one_step_moments` supplies exact conditional moments. The scenarios compare the
active law with the null law at the same realized state:

- Mean: \(\Delta m\ne0\), \(\Delta v=0\).
- Dispersion: \(\Delta m=0\), \(\Delta\log v\ne0\).
- Matched tail: \(\Delta m=0\), \(\Delta v=0\), but the centered fourth moment changes.

The tail innovation is an exactly variance-normalized Gaussian scale mixture. If
\(B\sim\operatorname{Bernoulli}(p)\), \(Z\sim N(0,1)\), component scales are
\(\ell<1<h\), and

\[
a^2=(1-p)\ell^2+ph^2,\qquad \varepsilon=s_BZ/a,
\]

then \(E\varepsilon=0\), \(E\varepsilon^2=1\), while

\[
E\varepsilon^4=3\frac{(1-p)\ell^4+ph^4}{a^4}>3
\]

for a non-degenerate mixture. Thus this is not an approximately matched stress test;
mean and variance are analytically identical to the null conditional law.

`PostChangeStrengths` changes only active post-boundary tensors or tail-mixture
settings. It does not enter the null `match_key`. Tests verify exact equality of all
pre-boundary samples under common random numbers and verify the one-step entry channel
at the same realized state.

## Residual diagnostics

The default residual model is a low-cost frozen-prefix ridge model:

1. Fit scaling and the one-step conditional mean model on the first 25% of each
   trajectory, without reading the boundary or any oracle array.
2. Freeze that model.
3. Exclude every prefix-training residual from both sides of every scan statistic.
4. Scan only the remaining out-of-sample residuals.

This requires a declared stable initial reference period. The benchmark rejects a
designed boundary that overlaps it. `blocked_crossfit` is available as a sensitivity
when that assumption is inappropriate, but its complement-trained models can mix
regimes and should not be silently treated as a full-state oracle.

The dispersion scan compares side-centered residual scale. Its default effect size is
the maximum absolute per-channel log variance ratio, appropriate for a sparse
neuromodulator effect. A block-median second-moment estimate reduces domination by a
single sample, but it does not make the statistic invariant to a changed fourth
moment.

The tail scan separately centers and RMS-normalizes each candidate side, then compares
bounded fourth-moment features and absolute-residual exceedances. Its reported effect
is the RMS change in the winsorized standardized fourth moment. Segmentwise scale
normalization is what permits a tail-shape claim rather than a second-moment claim.

Broad MMD or energy calls remain `general_law`. They cannot be relabeled as tail
evidence. Event attribution uses only explicit mean, dispersion, and residual-tail
probes for channel-specific labels.

## Calibration and uncertainty

For method \(j\), reduce every independent matched null series to the same maximized
scan statistic

\[
M_b^{(j)}=\max_{\tau\in\mathcal T}S_j(X_b,\tau).
\]

Given \(B\) calibration series, the test p-value is

\[
p_j=\frac{1+\#\{b:M_b^{(j)}\ge M_*^{(j)}\}}{B+1}.
\]

Whole trajectories are the exchangeability unit. Serial dependence, ridge fitting,
candidate maximization, and the detector's fixed preprocessing are therefore inside
the calibrated operation. Calibration, final-null evaluation, and changed evaluation
have disjoint IDs and RNG seeds; evaluation rejects calibration-ID reuse.

Two uncertainties are reported separately:

- A Wilson interval for the observed false-positive rate on the independent final-null
  pool.
- The Beta order-statistic interval induced by using a finite calibration pool. With
  \(B=99\) and \(\alpha=.05\), the attainable rank-test rate is .05, but the fixed
  threshold's 95% null-tail interval is approximately [.0166, .1002].

`hit` is localization within tolerance without requiring significance.
`detected_within` requires both a calibrated call and a hit. `localization_delay` is
reported for every changed series, while `detected_delay` is reported only for calls.
The effect at the true boundary is computed only after detection for evaluation; it is
never an input to a detector.

Panel attribution is exploratory rather than familywise calibrated. In particular,
its any-method null call rate should not be read as a 5% test.

## Medium validation

Artifact: `results/changepoint_medium_v1.json`

Configuration:

- Four observed neurons, two modulators, 480 observed steps, 240 burn-in steps.
- One boundary at step 240; tolerance 20 steps.
- 99 independent calibration nulls, 100 independent final nulls, and 30 changed
  trajectories per mechanism.
- Six detectors, stride four, one CPU process; wall time 25.6 seconds and maximum
  resident set size about 145 MB on the validation machine.
- Base mechanistic effect strength 1.2.
- Mean multiplier 1.0.
- Dispersion multiplier 4.0. Only one of four channels is affected in this fixed
  parameter draw; its post/null conditional variance is approximately 0.49.
- Matched-tail probability .10, scales .5 and 3.0, tail-logit multiplier 1.5.

Primary claim-matched results:

| Scenario / method | Call or FPR | 95% Wilson CI | Hit | Detected within | MAE | Median delay | Median effect at truth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Null / mean CUSUM | .06 | [.028, .125] | — | — | — | — | — |
| Null / variance residual | .10 | [.055, .174] | — | — | — | — | — |
| Null / residual tail | .03 | [.010, .085] | — | — | — | — | — |
| Mean / mean CUSUM | 1.00 | [.886, 1.00] | 1.00 | 1.00 | 4.4 | 0 | .546 RMS Cohen d |
| Dispersion / variance residual | .70 | [.521, .833] | .60 | .50 | 35.5 | 4 | .740 max absolute log ratio |
| Matched tail / residual tail | 1.00 | [.886, 1.00] | .967 | .967 | 3.3 | 0 | 12.93 winsorized fourth-moment change |

These results support three narrow conclusions:

1. The mean lane is easy in this design.
2. A dedicated conditional residual-shape statistic recovers this exactly
   mean/variance-matched tail change; the earlier mean/variance/general-law probes did
   not.
3. Sparse dispersion becomes detectable under a stronger twofold scale contrast, but
   calibrated and localized recovery is still far from complete.

## Explicit negative results and limitations

- The variance residual scan calls 90% of matched-tail series even though the oracle
  conditional variance is exactly unchanged. Only 13% localize within tolerance, its
  median delay is about 106 steps, and the tail-specific detector localizes correctly.
  This is finite-sample fourth-moment sensitivity of a residual-scale estimator—not
  evidence of a variance mechanism. The channel-attribution rule prefers direct tail
  evidence when calibrated p-values tie.
- The variance scan's final-null FPR is .10 [.055, .174], at the upper edge of the
  finite-B calibration-threshold interval. This run does not establish nominal 5%
  conditional size for that detector. Increasing calibration replicates and repeating
  across independent calibration pools are required.
- The benchmark fixes one structural parameter realization. Its power is conditional
  on that biological system; parameter-seed replication is needed before claiming
  population-level recoverability.
- The default frozen-prefix model assumes the initial reference fraction is stable.
  Early or recurrent changes require another nuisance strategy.
- Fluorescence adds calcium filtering and measurement noise. The reported medium run
  uses latent neural observations and therefore does not establish fluorescence-level
  power.
- Fourth-moment evidence is intrinsically high variance. The implemented clipping and
  empirical calibration bound influence but do not remove the need for adequate
  series length.

## Related diffusionCircuit evidence: June broad versus July narrow

The separate `diffusionCircuit` changepoint benchmark supplies observed NeuroPAL and
larger synthetic stress evidence. It uses quiet pseudo-boundaries for neural calibration
and independent matched no-change time series for synthetic calibration. These are not
iid row-permutation nulls. Real detections support stimulus-locked observed activity or
observed conditional/reliability changes; they do not identify latent anatomical
rewiring. Its default SBTG entry is `SBTG-FeatureBilinear`.

The broad June artifact is
`20260609_095452_extended_both`. It covered butanone, pentanedione, and NaCl; onset and
offset; imputed and complete cohorts; and synthetic stress. Across 740 event windows,
10,360 method-event tests, 8,960 synthetic calibration rows, and 14 methods, every row
completed. For pooled neural onsets, Hotelling mean CUSUM was strongest: on the imputed
cohort `detected_within_5s=0.667`, `hit_5s=0.933`, localization MAE `1.933s`, and median
q about `0.071`; complete-cohort sensitivity was `0.778` and `1.000`. Fixed and tuned
SBTG score-product probes both had zero calibrated within-5s imputed onset detections.

The canonical July 7 artifacts are deliberately narrower:

- `20260707_194322_score_dynamics_probe_synthetic`;
- `20260707_194459_score_dynamics_probe_neural` (20 imputed butanone onsets and 80
  quiet pseudo-boundaries per method);
- `20260707_194601_score_dynamics_sensitivity_neural` (6 complete-cohort butanone
  onsets and 24 quiet pseudo-boundaries per method).

They include butanone **onset only** and compare eight selected methods. Consequently the
July table refines one conditional/reliability question; it does not supersede the June
multi-stimulus, onset/offset benchmark.

For neural rows, the serialized detection rule is exactly

\[
\texttt{detected\_within\_5s}
=\texttt{hit\_5s}\;\land\;
\{p\le .05\;\mathbf{or}\;q_{\mathrm{BH}}\le .10\}.
\]

Thus `detected_within_5s` is not a q-only metric: a localized raw-p detection also
counts. BH q-values are computed across real stimulus tests within method. Quiet null rows
have no q-value, so their `alpha05_or_q10` call reduces to raw (p\le .05).

Primary imputed butanone-onset results were:

| Method | Detected within 5s | Hit within 5s | Localization MAE | Median raw p |
| --- | ---: | ---: | ---: | ---: |
| pooled history-DSM v2 studentized precision | 0.35 | 0.70 | 4.6125s | 0.0802469 |
| Hotelling mean CUSUM | 0.30 | 0.80 | 2.9250s | 0.228395 |
| score-dynamics variance readout | 0.15 | 0.85 | 2.3625s | 0.543210 |
| residualized SBTG score-product CUSUM | 0.00 | 0.05 | 11.1250s | 0.222222 |

Every listed method had quiet pseudo-boundary FPR 0.05 under the same raw-p-or-q field.
The complete-cohort sensitivity was:

| Method | Detected within 5s | Hit within 5s | Localization MAE |
| --- | ---: | ---: | ---: |
| Hotelling mean CUSUM | 0.333333 | 0.833333 | 2.1250s |
| pooled history-DSM v2 studentized precision | 0.166667 | 0.666667 | 4.7500s |
| score-dynamics variance readout | 0.000000 | 0.833333 | 2.8750s |
| residualized SBTG score-product CUSUM | 0.000000 | 0.166667 | 10.4583s |

Complete-cohort quiet FPR was 0.041667 for each method. On the matched-null synthetic
July probe, the score-dynamics variance readout's detected-within-5s rates were 0.667 for
`matched_marginal`, 0.667 for `precision_routing`, 1.000 for `rule_change`, and 0.000 for
`nonlinear_precision_routing`; all method null-FPR values were 0.00 with only 40 matched
null simulations, so that zero is imprecise rather than proof of exact size.

The defensible July conclusion is narrow. Pooled history-DSM v2 remains the strongest
calibrated conditional/reliability entry on imputed butanone, with weaker complete-cohort
robustness. The score-dynamics variance readout is a promising localizer but has weak
calibrated neural power. Hotelling remains a strong marginal/activity comparator. SBTG
remains a negative result: it produced no localized calibrated neural detections in either
July cohort, consistent with the broad June result.

## Running the lane

Medium:

```bash
python neuromod_benchmark/scripts/run_changepoint.py \
  --config neuromod_benchmark/configs/changepoint_medium.json \
  --output neuromod_benchmark/results/changepoint_medium_v1.json
```

Full:

```bash
python neuromod_benchmark/scripts/run_changepoint.py \
  --config neuromod_benchmark/configs/changepoint_full.json \
  --output neuromod_benchmark/results/changepoint_full_v1.json
```

The runner forces single-thread BLAS defaults, writes atomically, preserves raw
calibration maxima, and records all scan settings and disjoint series IDs.

## Literature anchors

- Page, E. S. (1955), “A Test for a Change in a Parameter Occurring at an Unknown
  Point,” *Biometrika* 42:523–527,
  [doi:10.1093/biomet/42.3-4.523](https://doi.org/10.1093/biomet/42.3-4.523).
- Chow, G. C. (1960), “Tests of Equality Between Sets of Coefficients in Two Linear
  Regressions,” *Econometrica* 28:591–605,
  [doi:10.2307/1910133](https://doi.org/10.2307/1910133). The classical F reference is
  not used here because ridge fitting and boundary maximization change its null law.
- Andrews, D. W. K. (1993), “Tests for Parameter Instability and Structural Change
  with Unknown Change Point,” *Econometrica* 61:821–856,
  [doi:10.2307/2951764](https://doi.org/10.2307/2951764).
- Inclán, C. and Tiao, G. C. (1994), “Use of Cumulative Sums of Squares for
  Retrospective Detection of Changes of Variance,” *JASA* 89:913–923,
  [doi:10.1080/01621459.1994.10476824](https://doi.org/10.1080/01621459.1994.10476824).
- Gretton, A. et al. (2012), “A Kernel Two-Sample Test,” *JMLR* 13:723–773,
  [primary article](https://www.jmlr.org/papers/v13/gretton12a.html).
- Baringhaus, L. and Franz, C. (2004), “On a New Multivariate Two-Sample Test,”
  *Journal of Multivariate Analysis* 88:190–206,
  [doi:10.1016/S0047-259X(03)00079-4](https://doi.org/10.1016/S0047-259X(03)00079-4).
- Matteson, D. S. and James, N. A. (2014), “A Nonparametric Approach for Multiple
  Change Point Analysis of Multivariate Data,” *JASA* 109:334–345,
  [doi:10.1080/01621459.2013.849605](https://doi.org/10.1080/01621459.2013.849605).
- Phipson, B. and Smyth, G. K. (2010), “Permutation P-values Should Never Be Zero,”
  *Statistical Applications in Genetics and Molecular Biology* 9,
  [doi:10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585).
- Wilson, E. B. (1927), “Probable Inference, the Law of Succession, and Statistical
  Inference,” *JASA* 22:209–212,
  [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953).
- Brys, G., Hubert, M., and Struyf, A. (2006), “Robust Measures of Tail Weight,”
  *Computational Statistics & Data Analysis* 50:733–759,
  [doi:10.1016/j.csda.2004.09.012](https://doi.org/10.1016/j.csda.2004.09.012).

These references motivate the statistic families. Inferential validity in this lane
comes from the matched whole-series maximum-statistic calibration, not from importing
iid or classical parametric reference distributions.
