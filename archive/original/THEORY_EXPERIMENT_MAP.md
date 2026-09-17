# From SID theory to the experiments we actually ran

**Theory source:** `SID_Predictive_Distributional_Dynamics_identification_aware (1).tex` (consolidated manuscript, 2026-07-15)  
**Evidence source:** `EXPERIMENT_INDEX.md` and the canonical artifacts named below  
**Snapshot:** 2026-07-16

## The organizing idea

The project is easiest to understand as one sequence of scientific questions:

1. **What is the target?** A supported history perturbation changes a conditional future law.
2. **At what response resolution?** Mean, variance/cumulant, bounded characteristic features, or the future path law.
3. **At what history resolution?** A global average, a declared state-resolved profile, or an ambitious pointwise derivative.
4. **How is it estimated?** Likelihood score, direct feature regression, Riesz orthogonalization, two-sided contrast, or pathwise differentiation.
5. **What makes it identifiable?** Support, smooth weights, a stable Riesz representer, adequate feature resolution, and dependence-aware inference.
6. **Can it be composed through time?** Only under a declared predictive-closure assumption or with a separately calibrated composed-side estimator.
7. **What scientific claim is allowed?** Observed predictive sensitivity first; latent or causal interpretation only with additional identification.

The run directories are implementations and stress tests of these questions. They are not thirteen unrelated research projects and they should not be read as one giant method leaderboard.

## One picture of the program

```text
Observed history H  --supported direction v-->  conditional future law P(k,h)
       |                                             |
       | choose state resolution G,B                 | choose response features Phi
       v                                             v
state-resolved derivative profile          mean / variance / bounded law / path
       |                                             |
       +---------- choose estimation interface ------+
                         |
          likelihood score / direct regression /
             Riesz-ORTH / finite contrast / pathwise
                         |
                         v
             identification and inference gates
       support • Riesz norm • feature resolution • HAC
       derivative audit • closure defect • observation model
                         |
                         v
                  permitted scientific claim
 predictive sensitivity → law-level sensitivity → latent-visible → causal
```

## The theoretical ladder and its empirical status

| Theory step | Object from the manuscript | Main empirical question | Experiments that answer it | Current answer |
|---|---|---|---|---|
| 1. Gaussian seed identity | Mean derivative recovered by a conditional history-score covariance | Does the elementary identity work numerically? | Distributional SID E0–E2; foundational SID synthetic kernel; empirical Tier 1 mean cells | **Yes in regular/matched systems.** The identity and orthogonal remainder work. There is no general point-error advantage over direct regression. |
| 2. Named distributional channels | Mean, variance/log variance, covariance, cumulants, tail/occupancy functionals | Can the method recover a declared channel rather than only the mean? | Foundational gain-kernel synthetic; frozen-v2 mean/dispersion/tail suites; empirical SID M1–M5 and D2/D3/D5 | **Sometimes, when the channel and representation match.** Mean/covariance/variance-lag are recoverable; generic high-order and gating readouts often fail. |
| 3. Weak law tangent | Derivatives of bounded future tests, especially characteristic/RKHS features | Can a law change be detected when low moments are unchanged or a density score does not exist? | Distributional E4 moment-blind; E5 support motion | **Yes at declared feature resolution.** E4 needs a resolving frequency bank; E5 remains accurate at deterministic support motion where likelihood score ceases to exist. |
| 4. State-resolved projection | `E[J(H)v | G(H)=g]` represented in a registered basis | Can a global zero hide large local effects? Is exact pointwise recovery necessary? | Distributional E7 cancellation; point-history-tangent convergence suite | **The projection works; unrestricted pointwise tangents do not.** E7 recovers sign reversal at basis resolution R=2, while converged point-tangent models still fail readiness gates. |
| 5. Riesz identification and orthogonal inference | Weighted derivative coefficients with product-of-nuisance remainder | Does orthogonality remove first-order nuisance bias, and is the target supported? | E2 corruption experiment; E1/E3 coverage and HAC; E6 overlap/tail adjudication | **Algebra and regular-cell inference pass.** Dependence requires blocked splits/HAC. The original fixed Riesz sieve fails in tails/gaps; a low-dimensional density-score representer conditionally rescues it. |
| 6. Identification-aware score geometry | Score error judged through the predictive operator it induces | Does low score/NLL error imply the derivative or scientific readout is correct? | Empirical SID operator experiment; frozen-v2 objective panel; history-tangent diffusion-versus-AR report | **No.** Low response-score or predictive error can coexist with large tangent error, and method rankings change with the target functional. |
| 7. Temporal and path layer | Direct versus composed horizons; endpoint versus path features | Does one fitted transition compose correctly, and do path features detect ordering changes? | Distributional E11 path ordering; E12 closure confirmation; G8 typed/path studies | **Both programs are now run, with a mixed closure decision.** E11 detects equal-endpoint/different-order paths. E12 has calibrated closure nulls and decisive one-step-misspecification power, but short-history AR(2) power is 0.70 versus the frozen 0.80 gate. |
| 8. Partial observation and claim boundary | Observed-law tangent, visible latent subspace, then causal interpretation only with extra assumptions | What survives calcium filtering, missing neurons, hidden modulators, and observational confounding? | Distributional E8; empirical D3/D5; frozen-v2 latent/intervention suites; real neural analyses | **Only conditional or predictive claims survive.** Long-horizon correctly specified synthetic filtering can work; short/misspecified filtering, latent recovery, and controlled C1 remain weak. Real causal or anatomical-mechanism recovery is unsupported. |

## The manuscript’s eight decisive experiments: what was actually run

### Experiment 1 — Original Gaussian score block

**Theory question.** Do the analytic score, learned likelihood score, direct derivative, two-sided contrast, and Riesz-orthogonal estimate recover the same mean Jacobian in a regular Gaussian system?

**Closest completed evidence.** Distributional SID E0, E1, E2, and E3; empirical SID mean cells; the G1 location systems in the history-tangent benchmark.

**Result.** The algebra and regular recovery pass. In E1, ORTH NRMSE is 0.104 with coverage 0.940, but its point-error ratio versus PLUG is 1.002: the benefit is valid inference, not universal point-accuracy dominance. Under strong dependence, blocked splitting and HAC restore coverage to roughly 0.95.

**Permitted claim.** The Gaussian identity and orthogonal estimator work in regular cells; do not claim that score-derived estimation universally beats direct mean regression.

### Experiment 2 — Smooth leverage and cancellation

**Theory question.** Can a state-resolved profile recover a sign-changing effect whose global average is zero?

**Closest completed evidence.** Distributional SID E7 heterogeneous cancellation.

**Result.** At registered history-basis resolution R=2, the sign reversal is recovered in every run and the estimated heterogeneity energy is 3.982 versus truth 4.000.

**Permitted claim.** State resolution can reveal declared heterogeneity that a global average erases. It does not recover arbitrary pointwise structure outside the registered basis.

### Experiment 3 — Moment-blind law changes

**Theory question.** Can bounded characteristic features detect a law change that preserves the first moments?

**Closest completed evidence.** Distributional SID E4; empirical SID M3–M5/M4 target studies.

**Result.** E4 passes only after the frequency-bank amendment: power is 1.00 and 0.867 in the two moment-blind families, while the default low-frequency bank failed. Empirical SID shows the complementary lesson: generic direct or score-derived high-order readouts can be worse than zero, while a mechanism-matched normalized law can recover the supplied target.

**Permitted claim.** Bounded law features add information beyond a finite moment list at their declared frequency resolution. Feature-bank resolution is part of the estimand, not a tuning footnote.

### Experiment 4 — Support-motion limit

**Theory question.** What remains estimable when a conditional density score becomes ill-conditioned or disappears?

**Closest completed evidence.** Distributional SID E5 support motion.

**Result.** ORTH reaches NRMSE 0.016 at zero observation noise, where the ordinary likelihood score is undefined.

**Permitted claim.** The weak/bounded-feature target is more general than a density-score representation. This is one of the program’s cleanest distinctive synthetic results.

### Experiment 5 — Nuisance corruption and orthogonality

**Theory question.** Does the orthogonal score convert first-order nuisance bias into a product of nuisance errors, and when is the Riesz functional stable?

**Closest completed evidence.** E2 controlled nuisance perturbations; E3 dependence/HAC; E6 tail and overlap studies plus their adjudication.

**Result.** The product-bias relationship has R² above 0.99999997 in three controlled perturbations. Regular dependent cells pass with HAC. The original fixed Riesz sieve has severe coverage failure in heavy tails/low-density gaps; a flexible density-score representer reaches absolute error 0.0036 and coverage 1.00 in the low-dimensional gap-4 cell at n=128k.

**Permitted claim.** Orthogonality works when at least one nuisance is adequate and the target is regular. The E6 rescue is a low-dimensional post-hoc diagnosis, not evidence of general high-dimensional Riesz stability.

### Experiment 6 — Direct versus composed horizons

**Theory question.** Does a one-step conditional law reproduce direct multi-horizon derivatives under predictive closure, and does the defect grow when the declared history is insufficient?

**Completed evidence.** Distributional SID E12 implements the specified direct outer-orthogonal side, a separately cross-fitted rollout/pathwise composed side, and full trajectory-bootstrap refitting across 30 independent seeds. It includes closed AR(1), correctly declared AR(2), AR(2) with a short history, correctly specified nonlinear Markov, and nonlinear one-step-misspecification cells.

**Result.** **Completed, mixed confirmation (six of seven gates).** Closure-null FPR is 0.0148; nonlinear misspecification power is 1.00; the short-history AR(2) mean defect grows from 0.120 at horizon 2 to 0.191 at horizon 8; and analytic-AR direct coverage is 0.994. The sole failed gate is horizon-8 short-history AR(2) power, 0.70 versus 0.80. Its horizon-8 point estimates nevertheless match the oracle closely: 0.210 versus 0.209 for the mean and 0.146 versus 0.146 for `sin(x)`.

**Permitted claim.** The resolved closure diagnostic is calibrated in the registered closed systems and detects strong one-step misspecification. Insufficient history produces the predicted growing average defect, but the current trajectory-bootstrap test is underpowered for the frozen AR(2) alternative. The invalid observed-outcome outer correction empirically collapses back to the direct target and must not be used for the composed side.

### Experiment 7 — Endpoint versus path-law changes

**Theory question.** Can path features detect temporal-order changes that identical endpoint marginals cannot reveal?

**Closest completed evidence.** Distributional SID E11 and the G8 typed/path program.

**Result.** E11 full-path power is 1.00 with NRMSE 0.209, while endpoint and unordered summaries have FWER 0.056. In the developmental G8 study, contrast-aligned motif/path models achieve roughly 1.1%–3.1% median relative error, while a generic linear motif is essentially the zero baseline.

**Permitted claim.** Path-aware features provide a distinctive capability at declared path resolution. G8 remains developmental and representation-aligned rather than a general full-path confirmation.

### Experiment 8 — Partial observation

**Theory question.** Which observed-law or latent directions remain recoverable after filtering, missingness, and aliasing?

**Closest completed evidence.** Distributional E8; empirical SID D3/D5; frozen-v2 latent recovery and controlled-intervention suites; ACMMA in real worms.

**Result.** In the E8 adjudication, correctly specified filtering with rho=.98, horizon 64, and n=32k gives ORTH NRMSE 0.141, while horizon 4 gives 2.88. Misspecified correction can collapse coverage. Empirical D5 emits a nonzero variance profile under an observation-only null. Frozen-v2 latent models predict worse than ridge and controlled common-history knockout NRMSE remains roughly 0.63–0.92. ACMMA solves missing-neuron moment assembly; it does not solve latent biological identification.

**Permitted claim.** Partial-observation recovery is conditional on observability, horizon, and a credible measurement model. The real data support observed predictive statements, not arbitrary latent coordinates or receptor-specific causal effects.

## How the estimator families fit the theory

| Estimator family in the repository | Manuscript interface | What it can estimate well | What it does not automatically establish |
|---|---|---|---|
| Ridge, SINDy, VAR, Granger | Direct conditional mean/moment regression | Low-order predictive dynamics and classical baselines | Law-complete sensitivity, latent mechanism, or causal edges |
| Gaussian/Student-t NLL, MDN, flows | Analytic normalized conditional likelihood | Predictive law, calibrated samples, AD history derivatives when audited | Derivative accuracy merely from good NLL |
| Hyvärinen/DSM conditional score models | Score interface | Selected response or density scores without normalization | The required history score, Riesz weight, or physical channel unless explicitly constructed |
| Direct moment/cumulant heads | Direct feature regression | Registered mean, covariance, variance, or cumulant targets | Other law changes outside those witnesses |
| Characteristic/RKHS ORTH | Bounded feature regression plus Riesz orthogonalization | Declared law-level projections with inference in regular cells | Unlimited full-law recovery or stability outside feature/support resolution |
| Finite ratio/classifier methods | Two-sided/local ratio interface | Supported contrasts with aligned representation | Infinitesimal point derivatives or unsupported observational perturbations |
| Pathwise/stochastic-interpolant models | Reparameterization/path interface | Differentiable generated-path sensitivities and path diagnostics | Valid observational inference without calibration and first-stage uncertainty |
| SBTG joint-score probes | Precursor reduced-form score covariance | Screening/localization of observed reduced-form changes | Conditional SID law tangents, physical dispersion, or anatomical rewiring |
| Latent neuromodulated SSM | Explicit latent/operation model | Represented release, occupancy, and knockout operations | Identification of latent labels or causal operations from passive calcium alone |

## The claim ladder after all experiments

| Manuscript claim level | Current status | Evidence | What we may say now |
|---|---|---|---|
| Identity | **Supported** | E0/E1/E2; matched synthetics | Conditional functional derivatives equal score covariances under the stated regularity conditions, and the implemented algebra passes synthetic checks. |
| Projected derivative | **Supported in registered regular cells** | E7; E1/E3; selected empirical direct targets | Declared weighted or state-resolved derivatives are recoverable when support and basis resolution are adequate. |
| Orthogonal inference | **Supported conditionally** | E2 product remainder; E3 HAC; E6 adjudication | First-order nuisance bias is removed in regular cells; Riesz stability remains a gating condition. |
| Law-level sensitivity | **Supported at declared resolution** | E4, E5, E11 | Bounded characteristic/path features detect moment-blind, support-moving, and order-changing alternatives at registered resolution. |
| Temporal composition | **Completed, mixed/conditional** | Distributional E12 closure confirmation | Direct-versus-composed calibration passes under declared closure and strong misspecification is detected; the short-history AR(2) power gate misses at 0.70, so temporal composition remains a resolution- and power-qualified claim. |
| Latent predictive content | **Limited/conditional** | E8; D3/D5; frozen-v2 latent suites | Some visible latent-filter directions are recoverable in correctly specified synthetic settings; arbitrary latent recovery is not. |
| Causal effect | **Not supported by current real data** | Weak frozen-v2 controlled C1; negative real mechanism results | Real neural findings are predictive or reduced-form. They are not receptor-specific interventions or anatomical rewiring. |

## The real-neural results in this theory

The real-data work should be read only after the synthetic identification ladder.

### What the data support

- **Predictive distribution:** modeling conditional variance improves leave-one-worm-out NLL by 0.364 on 28 worms, with all targets improving. This validates a predictive-law contribution.
- **Incomplete observation infrastructure:** ACMMA reproduces the complete-case estimator at Pearson/Spearman 1.000 and improves gain split-half stability to roughly 0.46–0.56 without donor imputation.
- **Secondary localization:** some anatomical/receptor proxy AUROCs are competitive, but the targets are noisy and configuration-sensitive.

### What the data do not support

- The lag-resolved aminergic/peptidergic neuromodulator signature does not replicate under all-data ACMMA, all tested transmitters, both score-matching modes, neural DSM, and global-state controls.
- The self-gain lead lacks a robust metabotropic-specific fingerprint and is limited by per-neuron reliability.
- Passive calcium histories do not identify receptor-specific operations, latent anatomical rewiring, or causal neuromodulation.

The clean empirical conclusion is therefore:

> Distributional structure improves prediction, and the project developed useful estimators and identification diagnostics; the present data do not support the stronger anatomical or causal neuromodulator interpretation.

## A theory-first reading order

1. Read the manuscript preface and Chapters 1–3 to understand the target and the score identity.
2. Read the weak-law/RKHS and identification-aware chapters to understand why bounded functionals, not generic score MSE, define the scientific objective.
3. Read the state-resolution, Riesz, and orthogonality chapters to understand why pointwise derivative output is not enough.
4. Read the temporal layer to separate endpoint prediction, path sensitivity, and predictive closure.
5. Use this file to connect each theoretical claim to the completed experiment.
6. Use `EXPERIMENT_INDEX.md` as the evidence ledger and supersession map.
7. Open individual run directories only after identifying the theory question and canonical artifact.

## What should be built next

1. **Improve E12 power without changing its completed decision.** Derive a sequentially orthogonal composed-side estimator or prospectively increase trajectory information; preserve the 0.70 frozen result.
2. **Prospectively confirm the E6/E8/E9 rescues.** Preserve their narrower estimands and do not relabel post-hoc diagnoses as original confirmations.
3. **Implement the theory’s estimator tournament on the same DGPs.** Compare analytic score, direct feature AD, Riesz-ORTH, two-sided contrasts, and pathwise derivatives using target error, coverage, and cost.
4. **Move real-data claims onto the claim ladder.** Lead with observed predictive distribution; present anatomical correspondence as localization; keep causal language behind an explicit intervention design.
5. **Use one experiment record per theoretical question.** Each future run should declare target, history resolution, response resolution, horizon/path object, estimator interface, identification assumptions, primary metric, and claim ceiling.

## Canonical evidence

- `reports/distributional_sid_adjudication_2026-07-14/results_snapshot.json`
- `distributional_sid/runs/e12_closure_confirmation_20260716/validation.json`
- `distributional_sid/E12_CLOSURE_RESULT_20260716.md`
- `reports/sid_empirical_validation_2026-07-14/tables/final_decision.tex`
- `empirical_sid/runs/sid_tier1_primary_20260714_v4/results/claims_registry.csv`
- `history_tangent_benchmark/DIFFUSION_VS_AR_SCIENTIFIC_REPORT.md`
- `history_tangent_benchmark/results/revised_note_v1_finite_contrast_20260712/`
- `neuromod_benchmark/outputs/frozen_v2/`
- `sid_elegans/output/biolag/ALLDATA_RESULT.md`
- `sid_elegans/output/newlevers/VALIDATION.md`
- `SYNTHETIC_METHODS_REPORT_2026-07-12.md`
