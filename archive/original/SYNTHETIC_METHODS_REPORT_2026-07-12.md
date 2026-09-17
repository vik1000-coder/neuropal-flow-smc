# Synthetic methods adjudication

## Technical summary

There is no universal winner because the methods learn different mathematical objects. Once the comparisons are separated by estimand, the result is fairly clean:

1. **For the conditional predictive law, normalized likelihood is best.** A Gaussian likelihood is the best simple choice for approximately Gaussian dynamics; a Student-t likelihood is the best cheap robust alternative; and an MDN is decisively best for genuinely multimodal or matched-tail laws.
2. **For physical mean and stochastic-dispersion fields in the tested mechanistic simulator, the simple penalized Gaussian NLL model is best overall.** Hyvärinen and denoising score matching can recover these fields, but do not outperform likelihood under matched information and seeds.
3. **Denoising is useful mainly as regularization and as an estimator of a corrupted response score.** It stabilizes the misspecified quadratic SID fit, but it does not automatically recover the clean law, a history score, or a physical mechanism.
4. **For finite supported history changes, direct classification or ratio estimation can work for location and low-rank interaction changes.** The present adapters do not recover covariance-only or skew-only changes, even when trained from oracle samples.
5. **SBTG is a reduced-form localization probe, not a conditional-density or physical-field estimator.** Its joint-score cross moments show moderate mean-like localization but weak and non-specific dispersion localization.
6. **Classical methods remain the strongest baselines for conditional means and graph support.** Ridge/SINDy are extremely hard to beat on mean dynamics. VAR-LiNGAM is best in the strong-hidden-confounder graph test, although no observational method escapes the degradation.
7. **There is no successful current method for synaptic gating, physical shape-tail derivatives, or robust latent causal response.** Larger sweeps of the same objectives are unlikely to fix these because the represented quantity or identification design is missing.

The recommended core stack is therefore: Gaussian/Student-t/MDN likelihood for predictive laws; Gaussian NLL or sparse dynamics for mean/dispersion fields; finite ratio/classifier routes for supported contrasts; classical graph baselines for graph recovery; and separate, explicitly operational latent/intervention models for C1 claims. Score matching and SBTG should remain specialized comparators rather than the default.

## New experiments run for this adjudication

### Matched objective panel

A new complete-state panel held the DGP, splits, seeds, forecast horizon, and evaluation metrics fixed across ten methods:

- five mechanisms: null, additive mean, synaptic gating, stochastic dispersion, and matched tail;
- five fresh DGP seeds, 8–12;
- 250/250 completed cases, no failures;
- normalized methods: ridge Gaussian, Student-t ridge, penalized Gaussian NLL, Gaussian neural NLL, Gaussian neural DSM at two noise levels, MDN likelihood, quadratic Hyvärinen SID, and quadratic DSM SID at two noise levels.

This closes the main objective-comparison gap in the earlier frozen runs.

### Ten-seed finite-contrast panel

The bounded signed classifier, finite density/ratio routes, and Riesz dictionary were rerun with ten adapter/evaluation seeds, 5004–5013, against one cleanly rebuilt base under a single source digest. All 76,760 metric rows completed. This tests adapter randomness; it does not add independent generator or predictive-model seeds.

## Method map: what each family is actually trying to learn

| Method family | Learned quantity | Optimization objective | Main claim ceiling |
|---|---|---|---|
| Ridge/VAR/Granger/SINDy | Conditional mean or sparse transition | Least squares, ridge, thresholding, sparsity | P1/D1/M1 |
| Student-t ridge | Conditional mean, scale, heavy-tailed forecast | Normalized Student-t likelihood | P1/D1/M1 |
| Gaussian NLL | Conditional Gaussian mean and log variance | Penalized normalized likelihood | P1/D1 plus fitted M1/M2 fields |
| Quadratic SID, Hyvärinen | Conditional response score with Gaussian reconstruction | Closed-form Hyvärinen score matching | Response score; reconstructed Gaussian law and fields |
| Quadratic SID, DSM | Gaussian-corrupted conditional response score | Closed-form denoising score matching | Corrupted response score; reconstructed Gaussian law and fields |
| Gaussian MLP NLL | Nonlinear conditional Gaussian law | Neural normalized likelihood | P1/D1 and differentiable field readouts |
| Gaussian MLP DSM | Corrupted nonlinear Gaussian response score | Neural DSM, selected by held-out NLL | Same parameterized Gaussian law, DSM training comparison |
| MDN | Factorized multimodal conditional density | Mixture likelihood | P1/D1 and differentiable mixture readouts |
| Conditional score MLP | Noisy response-space score | Gaussian or Student-t DSM | Score diagnostics; no normalized law |
| SBTG | Joint score of consecutive standardized states | Joint-state DSM | Reduced-form localization only |
| History Gaussian / AR-MDN | Normalized `p(y|h)` and its history derivative | NLL | Finite law and history-tangent comparisons |
| Ratio critic | Joint-versus-product interaction ratio | Binary classification | Interaction contrast; no history-only term |
| Conditional diffusion | Noisy response score and approximate sampler | VE denoising score matching | Noisy-law score unless inversion is validated |
| Signed classifier | Bounded central finite witness | Balanced classification plus calibration | Direct supported finite contrast |
| Signed Riesz | Projection of finite contrast onto fixed response features | Regularized moment solve | Restricted channel dictionary |
| Latent neuromodulated SSM | Release, concentration, occupancy, and emission law | Penalized sequential likelihood | P1/L1 and represented C1 operations |
| Conditional bridge | Endpoint law and reference-relative paths | Heteroskedastic endpoint fit plus Brownian reference | P1/D2, not mechanism recovery |
| Changepoint probes | Declared mean/rule/dispersion/tail/general change | Calibrated scan statistics | Detection and localization |

The detailed inventory is in `analysis/synthetic_methods_adjudication/method_taxonomy.csv`.

## What wins for each quantity

![Objective comparison](/Users/vik/Developer/new_sbtg_neuro/analysis/synthetic_methods_adjudication/objective_parity.png)

### Conditional-law prediction

On null, additive-mean, dispersion, and synaptic-gain scenarios, differences in NLL are small and ridge Gaussian is usually best or tied. Extra neural capacity and DSM do not help because the conditional law is close to the simple parametric family.

On the matched-tail law the ranking changes decisively:

| Method | Held-out NLL | Tail-probability RMSE |
|---|---:|---:|
| MDN K=3 | **-0.614** | **0.0095** |
| Student-t ridge | -0.530 | 0.0204 |
| Gaussian NLL | -0.288 | 0.0250 |
| Neural Gaussian DSM, σ=.25/.50 | -0.280 / -0.283 | 0.0263 / 0.0264 |
| Neural Gaussian NLL | -0.286 | 0.0268 |
| Quadratic DSM, σ=.75/1.0 | -0.283 | 0.0338 / 0.0325 |
| Quadratic Hyvärinen | 9.18, unstable | 0.0385 |

The implication is theoretical as well as empirical: if a normalized clean density is available and tractable, strict likelihood is the most direct proper objective. DSM is attractive when the normalizer is unavailable or a generative score/sampler is required, not because it is intrinsically more informative about clean conditional dynamics.

### Physical additive-mean field

In the new matched panel:

| Method | Mean-field nISE |
|---|---:|
| Penalized Gaussian NLL | **0.369** |
| Ridge Gaussian | 0.395 |
| Student-t ridge | 0.395 |
| Quadratic Hyvärinen SID | 0.473 |
| Quadratic DSM SID | 0.475–0.535 |
| Neural Gaussian NLL | 0.716 |
| MDN | 0.794 |
| Neural Gaussian DSM | 0.824–0.902 |

The earlier frozen mechanism panel is even more favorable to classical dynamics: SINDy achieved 0.194, sparse transition 0.201, Gaussian NLL 0.246, and Hyvärinen SID 0.268. The exact values differ across seeds and panel sizes, but the ordering is consistent: use the simplest correctly specified conditional-mean estimator.

### Physical stochastic-dispersion field

| Method | Log-variance-field nISE |
|---|---:|
| Penalized Gaussian NLL | **0.586** |
| Neural Gaussian DSM, σ=.25 | 0.789 |
| Quadratic Hyvärinen SID | 0.791 |
| Neural Gaussian DSM, σ=.50 | 0.794 |
| Neural Gaussian NLL | 0.809 |
| MDN | 0.872 |
| Constant-variance ridge | 1.000 |
| Quadratic DSM SID | 1.037–1.048 |
| Student-t ridge | 2.091 |

Neural DSM slightly beats neural NLL, which is genuine evidence that denoising can regularize a derivative readout. But the effect is small and both lose to the strictly parameterized Gaussian likelihood. In the earlier five-seed frozen panel, Gaussian NLL again led Hyvärinen SID, 0.436 versus 0.504.

### Matched tail and shape

MDN is the clear predictive winner because it can represent the two-scale mixture. Student-t is the useful cheap approximation. Every Gaussian family emits a constant standardized shape-tail probability and therefore scores exactly at the zero-field nISE baseline of 1. The MDN's fitted physical shape-tail derivative is worse than zero in the new panel, nISE 2.02, despite excellent predictive tail probability.

This is an important distinction: **learning the tail probability does not imply learning its derivative with respect to a physical modulator**.

### Synaptic gating

No current method recovers the total modulator-gated response field:

| Method | Synaptic-gating nISE |
|---|---:|
| Gaussian NLL / ridge / Student-t, all emit zero | **1.000** |
| Quadratic DSM SID | 2.10–2.95 |
| Quadratic Hyvärinen SID | 4.42 |
| Neural NLL / DSM / MDN | 20.97–23.40 |

The zero estimate is best. This is not evidence that gating is absent from the simulated trajectories; it means the fitted law/readout does not expose the required mixed derivative reliably. More hyperparameter search over the same readout is not the right next move.

## History tangents and finite contrasts

The point-history-gradient convergence sweep already showed that optimization was not the main problem. All 96 fits completed, all fixed-batch overfit gates passed, no selected checkpoint stopped at the epoch limit, and exact replay reproduced the results. Yet six derivative-readiness gates failed:

- G1 Gaussian tangent NRMSE 0.589;
- G1 AR-MDN 0.719;
- G2 best clean tangent 1.068;
- G3 best clean tangent 1.167;
- conditional-centering gates also failed.

M4 ratio and M5 diffusion passed some G1/G4 gates, but not enough to support a general point-gradient comparison.

![Finite contrasts](/Users/vik/Developer/new_sbtg_neuro/analysis/synthetic_methods_adjudication/finite_contrast_ten_seed.png)

At clean δ=.1 across ten adapter seeds:

| Generator | Best practically relevant route | Mean witness NRMSE | Verdict |
|---|---|---:|---|
| G1 location | Direct signed classifier / Gaussian finite ratio | about 0.61–0.65 | Useful |
| G2 covariance-only | Ratio critic | 1.057 | Does not beat zero |
| G3 skew-only | Ratio critic | 1.008 | Does not beat zero |
| G4 low-rank mixture | Ratio critic | **0.656** | Useful |

The oracle infinitesimal tangent has negligible Taylor error at this δ, but the oracle-sample classifier is still about 1.14 NRMSE on covariance and skew. Therefore:

- the current covariance/skew bottleneck is the finite classifier/dictionary geometry or weak relative signal;
- retraining a larger diffusion model cannot resolve it;
- the current signed Riesz dictionary is not viable, with NRMSE above 1 and sometimes far above it;
- finite contrasts should remain focused on G1/G4 until an oracle adapter demonstrably beats zero on G2/G3.

## SBTG and response-score DSM

SBTG learns the joint response score of standardized consecutive states, not `p(y|h)`, not the conditional response score, and not a physical neuromodulator field. Its valid comparison is localization against reduced-form edge support.

In the frozen mechanism panel:

- joint-score cross-moment AP excess for mean-like support was about 0.15–0.39;
- the same statistic was similarly positive under additive, dispersion, and synaptic scenarios, so it was not mechanism-specific;
- log-variance localization was only about 0.06–0.10;
- squared-score-covariance localization was roughly 0.00–0.05.

The linear SBTG was often as good as or better than feature-bilinear variants. This does not make SBTG useless: it can be a reduced-form screening statistic. It does mean it should not be described as recovering the physical dispersion or gating channel.

Conditional score MLPs learn a corrupted outcome score. Without a normalized reverse process, a history-score identity, or an explicitly validated channel adapter, their result is a score-fit diagnostic rather than a conditional-law or mechanism-recovery result.

## Classical graph and dynamics baselines

![Hidden confounding](/Users/vik/Developer/new_sbtg_neuro/analysis/synthetic_methods_adjudication/graph_hidden_confounding.png)

Under causal sufficiency, ridge/Granger/sparse-transition methods recover graph support at approximately 0.80 AP excess over prevalence. Under a strong hidden common driver:

- VAR-LiNGAM: 0.407;
- ridge VAR: 0.362;
- conditional Granger ridge: 0.361;
- sparse transition: 0.253;
- PCMCI partial correlation: 0.338;
- marginal lagged correlation: 0.106.

Every method degrades. VAR-LiNGAM is the strongest of the tested graph baselines under strong confounding, but the result remains observational support recovery, not causal identification with an omitted driver.

For finite-time conditional means, SINDy, sparse transition, ridge, and Granger consistently beat or tie the more elaborate distributional approaches. These must remain mandatory baselines.

## Latent models, interventions, bridges, and changepoints

### Latent neuromodulated SSM

The latent SSM is the only current family that represents release, persistent concentration, receptor occupancy, and explicit knockout operations. That gives it a higher possible claim ceiling, but not better current performance:

- calcium NLL is worse than ridge in the mixed panel, roughly -1.42 to -1.43 versus -1.58;
- aligned latent Spearman is only about 0.52–0.56;
- latent nRMSE is roughly 0.77–0.82 in the best mixed cases;
- clearance relative MAE is roughly 0.49–0.57;
- controlled common-history knockout nRMSE remains 0.63–0.92.

High-signal teacher-forced response curves are more successful. MDN/NLL and the latent SSM often achieve population response-kernel nRMSE of 0.25–0.48. But this is P1 transfer on realized arm histories, not controlled C1 recovery. The explicit common-history C1 endpoint remains weak.

### Conditional bridge

The bridge is a useful D2 path baseline, not a neuromodulator estimator. Its smallest reference diffusion (.15) gives the best path-energy score in the tested family, and error grows with horizon and reference diffusion. The comparison mostly identifies reference sensitivity; it does not show that a bridge recovers a latent mechanism.

### Changepoint probes

With 199 matched calibration trajectories and 100 independent evaluation trajectories per condition:

- mean CUSUM detects and localizes 100/100 high-signal mean changes;
- residual tail shape detects and localizes 100/100 exactly mean/variance-matched tail changes;
- the residual variance detector calls 100/100 dispersion changes but localizes only 41%;
- it also calls 95% of mean changes and 99% of matched-tail changes, while localizing those nuisance changes poorly;
- independent-null FPR is 2% for mean, 4% for conditional rule and variance, 3% for tail, but 9% for MMD and 11% for energy distance.

The mean and tail detectors work for their matched claims. The current variance detector is a broad residual-instability detector, not a channel-specific dispersion detector.

## Decision table

| Scientific task | Best current option | Best baseline | Score/diffusion verdict |
|---|---|---|---|
| Approximately Gaussian one-step law | Gaussian NLL or ridge Gaussian | Ridge VAR | DSM offers no consistent advantage |
| Heavy-tailed law | Student-t ridge | Gaussian ridge | Prefer likelihood unless only a score is available |
| Multimodal/matched-tail law | MDN likelihood | Student-t ridge | Gaussian DSM cannot represent the shape |
| Conditional mean dynamics | SINDy / sparse transition / ridge | Ridge VAR/Granger | Complex score models unnecessary |
| Physical mean susceptibility | Penalized Gaussian NLL | Ridge/Student-t | SID viable but weaker |
| Physical dispersion susceptibility | Penalized Gaussian NLL | zero/constant variance and heteroskedastic ridge | Neural DSM gives a modest within-architecture gain; closed-form DSM does not |
| Physical shape-tail derivative | No winner | zero field | MDN predicts tails but derivative recovery fails |
| Synaptic gating mixed derivative | No winner | zero field | All learned readouts worse than zero |
| Finite location contrast | Signed classifier or Gaussian finite ratio | mean-model classifier | Useful |
| Finite low-rank interaction contrast | Ratio critic | signed classifier | Useful |
| Finite covariance/skew contrast | No winner | zero witness | Oracle adapter fails; redesign first |
| Reduced-form edge screening | Linear SBTG or classical transition coefficients | ridge/SINDy | SBTG is not mechanism-specific |
| Graph support under confounding | VAR-LiNGAM | ridge/Granger/PCMCI | All degrade with hidden drive |
| Latent state / explicit operation | Latent mechanistic SSM | predictive ridge/MDN for P1 | Current L1/C1 recovery weak |
| Mean changepoint | Mean CUSUM | general MMD/energy | Strong matched result |
| Shape-tail changepoint | Residual tail-shape probe | general MMD/energy | Strong matched result |
| Dispersion changepoint | No channel-specific winner | variance residual alarm | Needs orthogonalization |

## What should be done next

1. **Stop seeking one universal score-based winner.** Maintain task-specific leaders and capability-correct comparisons.
2. **Use normalized likelihood as the default when available.** Require score/diffusion models to justify themselves through unavailable normalization, sampling needs, or demonstrated derivative gains.
3. **Make oracle-adapter tests mandatory.** No model training for G2/G3-like claims until the statistic itself beats the zero baseline using oracle samples.
4. **Develop a targeted gating estimator.** The current mixed derivative is not identified by generic law fit. A model must explicitly represent modulator-gated transition coefficients or use controlled paired interventions.
5. **Separate predictive tail fit from physical tail derivatives.** MDNs solve the former; the latter needs derivative-aware training or direct finite response contrasts.
6. **Repair dispersion changepoints with mechanism-confusion gates.** A detector earns the dispersion label only if it stays null or unlocalized under matched mean and tail changes.
7. **For latent/C1 work, simplify identification first.** Use one modulator or validation-independent labels, equalize information sets, and require common-history operation recovery before scaling model capacity.
8. **Preserve classical baselines in every suite.** Ridge, Student-t, SINDy, Granger, PCMCI, and VAR-LiNGAM are not ceremonial; several are current winners.

## Scope and validation

This report concerns synthetic and theoretical performance only. It intentionally excludes real neural-data success or failure.

The new objective panel validated 250/250 cases. The finite-contrast extension used ten same-source adapter seeds and 76,760 valid metric rows. The companion notebook executed top-to-bottom with no cell errors and regenerates the taxonomy, summary tables, and figures.

The main remaining uncertainty is end-to-end replication across independently retrained generator, data, and model seeds for the finite-contrast panel. That limitation does not affect the oracle-adapter failure: the present classifier and Riesz geometry do not recover G2/G3 even when the predictive source is oracle-generated.
