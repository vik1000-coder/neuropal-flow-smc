# Experiment portfolio and result hierarchy

**Snapshot:** 2026-08-31  
**Purpose:** identify which experiment families are load-bearing, which are conditional or developmental, which produced honest negative results, and which artifacts are superseded.

**If you know the theory manuscript, start with `THEORY_EXPERIMENT_MAP.md`.** It organizes the
experiments around the manuscript's target hierarchy, eight decisive experiments, estimator
interfaces, identification gates, and claim ladder. This file is the companion evidence ledger.

For the current NeuroPAL interpretation, start with [E31: lag/reference synthesis, neuron-class effects, and cohort provenance](results/neuron_class_effects_20260831/README.md). It reads E27/E29 as the matrix/inference source, keeps E30 predictive diagnostics separate, and adds only post-hoc descriptive grouping.

## 31 August figure and dashboard documentation delivery

[FIGURES.md](FIGURES.md) indexes the [static scientific figure collection](results/figure_atlas_20260831/README.md), its plotted-data exports, validation, and [experimental questions](results/figure_atlas_20260831/QUESTIONS_AND_EVIDENCE.md). [DASHBOARD.md](DASHBOARD.md) and the [detailed explorer guide](results/neural_prediction_atlas_20260829/DASHBOARD_GUIDE.md) document the exact URL, application path, serving directory, launch/rebuild commands, all five input layers, timing/normalization, evidence checks, exports, and limitations.

This is an evidence-presentation and documentation delivery, **not a new experiment number**. No model, repaired-path sampler, matrix, reference mask, GUI data, p-value, or q-value was changed. Plots explicitly distinguish pointwise intervals, corrected discoveries, descriptive magnitudes, and selected N128 reruns. The observed AWC panels are difference-in-changes in recorded activity, not inferred source→target effects. Source hashes, export files, numerical checks, and visual QA are recorded in the figure bundle.

## Critical stimulus-provenance correction (2026-08-28)

`AUDIT_STIMULUS_PROVENANCE_20260828.md` is authoritative over older descriptions below. The three local NeuroPAL epochs are **butanone, pentanedione, and NaCl in animal-specific order**, not three butanone repeats. Epoch position matches chemical code in only 24/60 retained animal-events. Consequently, E22's chemical-specific AWC/butanone and repetition/adaptation conclusions are superseded; the presentation-encoding runs are position-conditioned only; and E23's generic sampler mechanics remain historical estimator evidence but its pooled numeric biological interpretation must be recomputed because OH15500 was treated as 4.0 rather than 4.1 Hz and its 20-second final event was truncated. The released SBTG 80-neuron lineage is historical/contextual only because it used index-wise head/tail pseudo-pairing and donor-trace imputation. Corrected work uses 54 head neurons, frozen per-worm schedules, OH16230-head as primary, and an explicitly resampled OH15500 pooled sensitivity. This correction blocks chemical-conditioned inference from a failed chemical-encoding gate; it does not block the later binary-any-stimulus E27 atlas, which preserves true chemicals only as post-sampling event strata.

The operational artifact/result index for the rebuild is `CORRECTED_CHEMICAL_EXPERIMENT_20260828.md`.

## August 26–28 compatibility-aware response update

The newest neural-response program is indexed in
`results/compatibility_path_response/EXPERIMENT_INDEX.md`. Its sealed Stage-A
result selects progressive bridge SMC as the best finite-particle estimator of
one frozen generator's repaired-response law (MSE 0.01101 versus 0.01565 for
terminal SMC and 0.03069 for direct importance). A separate Stage-B external
evaluation then loads Randi, Cook, Bentley, SBTG-current, and SBTG-published
without retraining. Progressive's lag-1 Randi AUROC is 0.540 in the matched
one-checkpoint panel, while SBTG-published remains the strongest contextual
artifact at 0.622. This is estimator progress, not a new biological or causal
claim. A later no-retraining onset-aware analysis uses the three known stimulus
onsets per worm and quiet pseudo-onsets. Early-response persistence dominates
forecasting (worm-residual rho 0.600); SBTG-published has off-diagonal rho 0.202
but is not onset-specific, and no matrix adds reliable value over persistence.
Modest onset-minus-baseline Bentley peaks do not survive across-panel
multiplicity, so no physical-delay claim is supported.

A subsequent E19 experiment broadens the learned dynamics before returning to
the temporal-cut estimator. Across 1–20 second histories, the best off-diagonal
ridge has weak positive innovation rank correlation but is worse than the
self-history conditional model on untouched energy and RMSE. A literal lagged
ESS-SMC grid then freezes a zero-offset, 0.5-second cell using folds 0–2. Its
128-particle, three-seed confirmation gain on folds 3–4 is -0.00005 Spearman
with 95% worm interval [-0.00482, 0.00472], and only 35.6% of source clamps are
compatibility-valid. Post-freeze Randi/Cook AUROCs are 0.590/0.543; no Bentley
source-lag × horizon search survives its lag-max null. The complete result is
`results/multilag_temporal_cut_20260827/REPORT.md`.

E20 then replaces the single-cut matrix with an explicitly regularized smooth
distributed-lag tensor. Target self-history, the full stimulus history, and
four population PCs are fit as nuisance structure before source histories are
residualized target by target. A group-shrunk 16-frame / 4-second model selected
on folds 0–2 improves untouched-fold Student-t NLL by +0.009436 (worm-bootstrap
95% CI [+0.008471, +0.010382]) and beats a within-worm circular-shift null
(p=0.005), but worsens RMSE by 0.001854. Quiet-window improvement exceeds onset
improvement. Globally freezing the regularization improves edge-rank stability
from 0.474 to 0.764, yet post-freeze Randi/Cook AUROCs are only 0.503/0.515 and
the best Bentley BH q is 0.894. This establishes weak reproducible predictive
lag structure, not useful mean propagation, anatomy, or physical delay. The
canonical report is `results/distributed_lag_dynamics_20260828/FINAL_REPORT.md`.

E21 broadens the estimand and cohort rather than tuning E20 further. It tests
smooth conditional mean, log-variance, and covariance lag laws plus eight
neural density architectures on the current 20-worm/54-neuron cohort, the
exact original 20-worm/80-neuron SBTG cache, and an SBTG-to-current 54-neuron
bridge. All three structured mean lanes gain only a small Student-t log score
while worsening RMSE; all Gaussian scale gates fail; and covariance energy and
variogram scores disagree. Neural MDN wins current54 energy narrowly, while
wide flow wins both SBTG cohorts. Circular source-history ablation shows that
aligned histories improve neural-density NLL, but the extracted edgewise
variance/covariance lag matrices remain weak. Semi-synthetic variance edges
require large injected effects (current54 calcium-smoothed AUROC first exceeds
0.70 at coefficient 1.5), no receptor-enrichment cell survives the global
structured-plus-neural BH family (minimum q 0.17), and the best new current54
Randi/Cook AUROCs are 0.545/0.529. The canonical report is
`results/higher_order_dynamics_20260828/FINAL_REPORT.md`.

E22 is superseded for chemical-specific biology. It treated the three epoch
positions as three butanone repeats, although each animal actually received
butanone, pentanedione, and NaCl once in animal-specific order. Its numerical
matrices remain preserved as historical position-pooled artifacts, but its
AWC/butanone labeling, repetition attenuation, adaptation, and promoted-edge
claims are invalid. A corrected observed-data analysis now selects the actual
chemical event per worm and analyzes all three chemicals separately. In the
primary 17-worm OH16230-head cohort, AWC is suppressed for actual butanone at
4 seconds (mean matched contrast -0.873 background SD, BH q=0.00183) and 10
seconds (-1.672, q=0.000079); NaCl does not yield a corresponding AWC FDR
result. This establishes stimulus-associated activity only, not an edge. The
corrected report is
`results/biological_lag_analysis_20260828/chemical_corrected_observed_20260828/REPORT.md`;
the old `BIOLOGICAL_REPORT.md` is provenance only.

E23 remains useful only as historical estimator engineering evidence. Its
direct-importance and progressive ESS-SMC mechanics ran, and the old
progressive method improved finite-particle compatibility, but the biological
numbers were computed with a global position schedule and incorrect pooled
OH15500 clock/duration assumptions. They are not chemical-specific lag
matrices, and no old onset-minus-quiet, RIB→AVB, or physical-delay conclusion
is retained. E24 therefore prohibits a **chemical-conditioned** lag launch from
that tournament. It does not prohibit E27's separate binary-any-stimulus model
or post-sampling event stratification, neither of which treats event position
as chemical identity.

E25 tests the more conservative paired-sampling route on the corrected
17-worm/54-neuron primary cohort. The frozen flow's paired common-noise
matrices are Monte Carlo stable but fail every proper-score cell gate. An
atlas-blind calcium-aware innovation model then passes held-out NLL, energy,
RMSE, scale-NLL, and fold-kernel gates. Its strict direct audit leaves one
onset/8-second boundary cell and nine tiny edges; a generated-state rollout
leaves only an AIY→AIZ 4-second scale effect. Targeted N=256 ESS-SMC has 100%
proposal validity and agrees in signed magnitude, but its proper-energy
penalty is -0.000226 with interval [-0.000775, +0.000332]. The chain therefore
stops without a promoted matrix or edge. A separately scoped post-freeze
comparison nevertheless answers the original SBTG-style matrix question: the
paired Wasserstein matrix has best eligible-source Bentley AUROC 0.525 for
monoamine, 0.491 for neuropeptide, and 0.493 for their union, versus
SBTG-published 0.581/0.544/0.537. Transmitter-specific panels find descriptive
tyramine log-SD (0.640 at 2 s) and octopamine mean/tail (0.647/0.642 at 2 s)
peaks, but only one eligible source supports each of those panels. No new lag
profile survives lag-max calibration (minimum global q 0.933), and lag-1 edge
ranks correlate only 0.043 with SBTG-published in absolute Spearman. The canonical report is
`DISTRIBUTIONAL_LAG_EXPERIMENT_20260828.md`.

E26 returns to the original repaired-flow construction and makes the source
placement genuinely lagged. Direct importance, terminal-deferred SMC,
progressive bridge SMC, and temporal-cut SMC are run on the corrected
17-worm/54-neuron native-4-Hz cohort at 0.25, 1, 2, and 4 seconds. Progressive
bridge is the strongest 32-particle estimator: compatibility validity is 0.802
and generator-seed signed matrix stability is 0.478–0.688, versus roughly 0.16
validity for the other matched methods. Its fixed lag-1 Randi/Cook
structural/chemical/gap AUROCs are 0.637/0.611/0.603/0.650, numerically above
SBTG-published's 0.622/0.565/0.558/0.621. Pooled Bentley lag correspondence does
not improve: no endpoint-mean lag maximum survives correction (minimum global
q=0.597). The result supports progressive estimator engineering and
SBTG-like descriptive matrix comparison, not a chemical-specific mechanism,
receptor-mediated causal edge, or physical delay. The canonical technical
specification is `FLOW_REPAIRED_LAG_METHODS_20260828.md`; the frozen result is
`results/four_sampler_lag_connectome_20260828/analysis/REPORT.md`.

E27 freezes the full 17-worm, 54-neuron neural prediction atlas built from the
binary-any-stimulus TCN flow at native 4 Hz. Direct N=256 and progressive bridge
N=32 cover source lags 1/4/8/16 frames, horizons 1/2/4/8/16/32 frames, seven
effect channels, and recorded event phases. All 80 raw archives validate,
yielding 4,368 effect cells, 436,800 source-target predictions, and a 1,000-row
read-only hypothesis queue. Six preselected mean/pathwise candidates were then
re-estimated with progressive N=128: all six model-relative high-minus-low
effects retained direction, the median absolute screen-to-N=128 difference was
0.01229, and magnitude agreement was 0.966. This is selection-conditioned
same-data/model/seed confirmation, not an independent replication; seed-worm
rank correlations range from -0.130 to 0.610. A separately labeled
distributional/stimulus screen did not generalize comparably: magnitude
agreement was 0.141, four chemical phase contrasts shrank, crossed zero, or
reversed, one signed ASH→OLQ endpoint-SD effect remained, and an unsigned
AWC→AVA W1 distance attenuated.

In the sealed post-freeze equal-denominator lag-1 endpoint-mean panel,
progressive scores 0.647 against Randi and 0.611/0.600/0.672 against Cook
structural/chemical/gap, versus 0.617/0.619/0.610/0.660 for direct,
0.622/0.565/0.558/0.621 for SBTG-published, and
0.524/0.524/0.517/0.585 for SBTG-current. Against Bentley, progressive's
neuropeptide AUROC 0.581 and union AUROC 0.535 at lag 1 have global-BH q=0.018;
no monoamine or transmitter-specific panel survives. The early 1–4 frame band
is descriptive correspondence, not a physical transmission delay. E27's
canonical entry point is `results/neural_prediction_atlas_20260829/README.md`;
the read-only explorer is `gui/atlas_explorer.html` and the technical report is
`technical_report/report.html` beneath that directory. Every matrix is an
observational-law, model-relative response—not a causal, anatomical,
receptor-mediated, chemical-conditioned, or independently confirmed edge.

E28 tests whether E27's external-reference advantage persists on the released
historical 20-worm/80-neuron SBTG axis. The already selected atlas-blind
80-neuron wide-flow winner is sampled with progressive bridge N=32 over all
five held-out folds at source lags 1/4/8 and horizon 1; Randi, Cook, Bentley,
and SBTG-published are opened only after 15 raw archives freeze. It does not
replicate the Randi/Cook advantage: progressive scores 0.523 against Randi and
0.535/0.534/0.543 against Cook structural/chemical/gap, versus
0.630/0.581/0.570/0.613 for SBTG-published. Paired source-bootstrap AUROC
intervals are below zero for all four comparisons. Progressive W1 has higher
Bentley neuropeptide/union point AUROCs than SBTG-published on the common
1/8-frame grid (0.546/0.538 versus 0.523/0.525), but no test survives the
70-test BH family (minimum q=0.105). The cache is not a clean superset: it
retains the released donor-imputation and index-wise pseudo-pairing lineage.
The canonical report is `results/sbtg80_progressive_sensitivity_20260830/REPORT.md`.

E29 replaces E27's post-screen candidate q-values with exact complete-family
worm inference. Four prespecified families—baseline endpoint mean, baseline
endpoint log-SD, active-minus-baseline mean, and active-minus-baseline
log-SD—reuse all 65,536 two-sided worm sign orbits and are combined with a
joint single-step max-T null. Under the strong all-lag support and genealogy
gate, 102 baseline-mean edges and 492 lag-by-horizon cells survive, but no edge
has adjusted lag structure and neither stimulus-modulated family has a
discovery. These results establish internally worm-consistent frozen-model conditional
dependencies, not causal edges or delays. A separate 0.50-support sensitivity
finds SMD→RID at lag 16/horizon 1 (joint four-family lag max-T p=0.000458), but
its minimum valid/genealogy fractions are 0.765/0.706, so it remains
sampling-limited. The targeted progressive-bridge N=128 sampler-null gate is
now sealed: all 70 raw archives validate, the raw checksum ledger contains 73
entries, and the analysis evaluates 24 candidate–metric rows in one exact
64-test max-T family over the same 65,536 sign patterns (critical value
3.8246963393338396; analysis fingerprint
`c193fc999d504f9096540fa4e5e73aeaebe88bc8e8a7d85a805c9920bd3252ca`).
Fifteen rows are `sampling_limited`, four exceed the same-state sampler controls
only, five are indistinguishable from the sampler controls, and zero exceed
both the sampler and matched quiet-time controls. No row is experiment-ready.
The final schema-v3 explorer exposes this calibration as a five-rung evidence
ladder and keeps the canonical atlas estimate separate from the matched N=128
rerun estimate; the completed code suite has 270 passing tests. Canonical
artifacts are
`results/neural_prediction_atlas_20260829/COMPLETE_FAMILY_SIGNIFICANCE_20260830.md`,
`results/neural_prediction_atlas_20260829/TARGETED_SAMPLING_NULL_CALIBRATION_20260830.md`,
and `results/neural_prediction_atlas_20260829/gui/atlas_explorer.html`.

## August 31 marginal uncertainty and distribution-shape audit

E30 completes the two targeted checks on the corrected 17-worm, 54-neuron
cohort. Independent sample-index permutations within each conditioning history
preserve every predictive marginal and leave the flow’s energy advantage
intact: shuffled-minus-original energy is −0.032%, with no primary dependence
benefit. A small 0.62% innovation-variogram benefit is secondary and persists
at N=256. Holding the Gaussian mean fixed, history-dependent variance improves
marginal CRPS by 0.248% (Holm p=0.032); the matched Student-t family improves
CRPS over correlated Gaussian by 0.500% (p=0.020). Flow still beats Student-t
energy by 4.197% on all 17 worms (p=0.000076).

This is not uniform calibration superiority: the flow predicts the declared
large-innovation event at 10.38% versus 5.91% observed, and its secondary tail
Brier is worse than Student-t on every worm. The matched models have the same
encoder architecture and residual/validation protocol, but independently
fitted means and family-specific training objectives. The remaining advantage
cannot be uniquely attributed to shape, and weak conditional residual
dependence does not imply independent neural activity. All 30 new fits,
100 evaluation archives, 40 checkpoint replays, 30 independent density checks,
and 289 regression tests pass. This follow-up does not replace the older E04
variance result on a different cohort and model specification.

Canonical interpretation: `results/distribution_structure_20260831/INSIGHTS.md`.
Full report: `results/distribution_structure_20260831/analysis/REPORT.md`.

## August 31 neuron-class and cohort-provenance audit

E31 verifies the actual Cook SI6 workbook against the frozen 54-class atlas:
22 sensory, 23 interneuron, and 9 motor pooled head classes. It records 12
mixed/disputed classifications and repeats the summaries without them. Both
existing samplers are grouped across all seven channels, 13 contexts, four
source lags, and six horizons under all-estimated and fixed all-lag support
policies, giving 157,248 descriptive rows. No retraining, new path sampling,
reference-based selection, new p-values, or changes to E27/E29 evidence occur.

State-average lag-1/horizon-1 mean-effect magnitudes are somewhat larger for
motor → interneuron (0.0548) and motor → motor (0.0513) pairs. But baseline
motor → interneuron apparent late-lag growth (0.0612 to 0.0928) disappears
under a fixed strong-support source mask (0.0603 to 0.0610), which retains only
SIA and SMB as motor sources. The existing 102 strong baseline edges remain
lag-unresolved; regrouping does not upgrade them or establish class enrichment.
The whole-worm bootstrap is descriptive and conditional on the fitted models.

The provenance trace shows 54 is inherited from pooled-head coverage and
finite-data filtering before restriction to 17 OH16230 worms. Historical 80
contains 63 head and 17 tail classes and retains donor imputation and index-wise
pseudo-pairing, so it is not a clean simultaneous larger cohort. Historical
annotation helpers differ from the actual Cook workbook; those discrepancies
are documented without changing original results. Classification, orientation,
denominators, source-only support, signed/absolute aggregation, and bootstrap
checks pass; the expanded full suite contains 307 passing tests.

Canonical summary: [results/neuron_class_effects_20260831/README.md](results/neuron_class_effects_20260831/README.md).
Portable report: [results/neuron_class_effects_20260831/report/report.html](results/neuron_class_effects_20260831/report/report.html).

## Technical summary

The project supports a narrower and more useful conclusion than “SID works” or “SID fails.” The strongest results are: (1) matched synthetic distributional effects can be recovered; (2) ACMMA is a valid way to use incomplete multi-worm observations without imputation; (3) modeling conditional variance improves held-out real-data prediction; and (4) bounded, typed, or path-aware distributional features can work when the estimand and representation are declared in advance.

The strongest biological/mechanistic claim does **not** survive. Across all 28 worms, all tested monoamine and peptide references, Hyvärinen and denoising score matching, neural DSM, global-mode controls, and data/ridge variations, there is no robust lag-resolved neuromodulator signal. Likewise, unrestricted pointwise history tangents, generic synaptic-gating recovery, and current latent/common-history intervention recovery are not ready as primary claims.

The practical rule is therefore:

- Cite the **supported method and predictive results**.
- Label the **bounded/typed synthetic results** with their exact scope.
- Treat the **real neuromodulator lag result as a negative**.
- Use development and smoke runs only for provenance, never as headline evidence.

## Theory-first result in one paragraph

The Gaussian score identity and orthogonal product remainder are validated in regular synthetic
systems. Bounded characteristic and path features then add three distinctive successes: moment-blind
law changes, deterministic support motion, and temporal-order changes. State-resolved projection
recovers cancellation that a global average misses, while unrestricted pointwise history tangents do
not pass readiness gates. Riesz orthogonality works in regular cells but remains conditional on
support and nuisance quality. Partial-observation recovery is possible only with adequate horizon and
a credible measurement model. The direct-versus-composed closure experiment is now complete with a
mixed decision: closure calibration and strong misspecification detection pass, while short-history
AR(2) power is 0.70 versus a frozen 0.80 gate. On real
neural data, predictive distribution and ACMMA results survive; the stronger neuromodulator mechanism
claim does not.

## The five results that matter most

1. **Real predictive value is the strongest empirical result.** Conditional variance improves leave-one-worm-out NLL by 0.364 on 28 worms, with all targets improving and the interval excluding zero. This is predictive evidence, not mechanism identification. Canonical evidence: `sid_elegans/output/newlevers/RESULTS.md` and `SYNTHETIC_METHODS_REPORT_2026-07-12.md`.
2. **ACMMA is a durable estimator/infrastructure contribution.** It reproduces the complete-case estimator at Pearson/Spearman 1.000 and raises gain split-half stability from roughly 0.29 to 0.46–0.56 without donor imputation. Canonical evidence: `sid_elegans/output/biolag/ALLDATA_RESULT.md`.
3. **The real lag-resolved neuromodulator claim is a comprehensive negative.** The original peptidergic dBC +3.35 depends on an 80% row split; it falls to +1.16 on all rows from the same six worms and to -0.41 with all 28 worms under ACMMA. All tested transmitter and score-matching variants have intervals including zero. Canonical evidence: `sid_elegans/output/biolag/ALLDATA_RESULT.md`.
4. **Distributional SID works for declared bounded features, not unrestricted full-law or latent deployment.** The latest E0–E11 adjudication supports regular bounded features, support motion, path ordering, and conditional E6/E8/E9 rescues, but only under declared resolution, sufficient observation horizon, correct measurement models, or typed low-rank geometry. Canonical evidence: `reports/distributional_sid_adjudication_2026-07-14/main.pdf` and `results_snapshot.json` in that directory.
5. **Estimator/representation alignment is the main bottleneck.** Direct moments or matched normalized laws win many mean, covariance, variance-lag, tail, occupancy, and quartic targets. Current generic score-derived readouts have no calibrated equal-access advantage and can fail despite low response-score error. Canonical evidence: `reports/sid_empirical_validation_2026-07-14/main.pdf` and its `tables/final_decision.tex`.

## Portfolio map

| ID | Experiment family | Current verdict | What it establishes | Canonical artifact | Use in claims |
|---|---|---|---|---|---|
| E01 | Legacy SBTG connectome and synthetic benchmarks | Foundational, limited | Corrected pipeline, reduced-form screening behavior, historical comparisons | `SBTG/merged_results/README.md` | Background only; do not treat joint-score statistics as conditional SID fields |
| E02 | Foundational SID matched synthetics | Supported | Injected conditional-variance/gain kernel recovery (corr 0.998) | `README.md`; `sid_neuromod/output/synthetic/` | Strong matched-estimand method validation |
| E03 | ACMMA all-data estimation | Supported | Exact complete-case reproduction and improved stability with incomplete worms | `sid_elegans/output/biolag/ALLDATA_RESULT.md` | Methods/infrastructure claim |
| E04 | Real-data predictive distribution | Supported | Conditional variance adds held-out predictive value; MDN/likelihood routes can improve law fit | `sid_elegans/output/newlevers/RESULTS.md`; `SYNTHETIC_METHODS_REPORT_2026-07-12.md` | Primary empirical claim; predictive only |
| E05 | Real-data anatomical/receptor correspondence | Descriptive/conditional | Some holistic AUROC competitiveness, but targets are noisy proxies and configuration-sensitive | `sid_elegans/output/HOLISTIC_ASSESSMENT.md`; `paper/compare_approaches.json` | Secondary localization evidence |
| E06 | Real lag-resolved neuromodulator signature | Negative | No robust aminergic or peptidergic lag signal after full controls and all-data estimation | `sid_elegans/output/biolag/ALLDATA_RESULT.md`; `sid_elegans/output/newlevers/VALIDATION.md` | Cite as an honest null/non-replication |
| E07 | Frozen-v2 neuromodulatory mechanism benchmark | Mixed/conditional | Prediction and matched mean/dispersion can work; gating, latent, and controlled C1 recovery remain weak | `neuromod_benchmark/outputs/frozen_v2/`; `EXPERIMENT_REVIEW_2026-07-12.md` | Capability-specific synthetic evidence |
| E08 | Pointwise history-tangent benchmark | Negative for current route | Optimization succeeds but derivative-readiness gates fail; predictive fit does not validate tangents | `history_tangent_benchmark/DIFFUSION_VS_AR_SCIENTIFIC_REPORT.md`; convergence-sweep v2 | Do not use for a general point-gradient claim |
| E09 | Finite history contrasts | Developmental/conditional | Useful for G1 location and G4 low-rank interaction; G2 covariance and G3 skew fail oracle-adapter gates | `history_tangent_benchmark/results/revised_note_v1_finite_contrast_20260712/`; `SYNTHETIC_METHODS_REPORT_2026-07-12.md` | Continue only in supported contrast lanes |
| E10 | G8 typed/path rescue experiments | Developmental positive | Contrast-aligned typed classifiers recover the quartic-path feature; generic representations fail | `analysis/g8_typed_classifier_benchmark_2026-07-13/`; empirical claims registry C12 | Development evidence, not confirmation |
| E11 | Distributional SID E0–E11 | Conditional go | Bounded-feature inference works in regular cells; E6/E8/E9 require revised nuisance, measurement, or typed geometry | `reports/distributional_sid_adjudication_2026-07-14/` | Current synthetic methods decision |
| E12 | Empirical SID Tier 1/Tier 2 | Diagnostic mixed | Direct/matched routes recover several declared targets; generic score advantage is not established | `reports/sid_empirical_validation_2026-07-14/` | Target-by-target diagnostic validation |
| E13 | Claim-aware changepoint probes | Mixed/conditional | Matched mean and tail probes succeed; current variance probe is not channel-specific | `neuromod_benchmark/docs/CHANGEPOINT.md`; `SYNTHETIC_METHODS_REPORT_2026-07-12.md` | Detection/localization only; keep claim axes separate |
| E14 | Direct versus composed horizons | Completed, mixed confirmation | Closure-null FPR 0.0148; nonlinear misspecification power 1.00; short-history AR(2) defect grows but power is 0.70 versus 0.80 gate | `distributional_sid/E12_CLOSURE_RESULT_20260716.md`; E12 confirmation run | Temporal-composition evidence at registered resolution; preserve failed power gate |
| E15 | Compatibility-aware repaired neural responses | Completed, conditional | Frozen generators yield model-relative repaired-response matrices with external signal, but strict support and lineage caveats limit superiority claims | `results/compatibility_path_response/final_analysis_20260826/REPORT.md` | Observational effective-response evidence only |
| E16 | Progressive bridge estimator benchmark | Supported estimator result | Progressive SMC lowers one-checkpoint learned-law response MSE to 0.01101 and passes all sealed advancement gates | `results/compatibility_path_response/frozen_estimator_benchmark_20260826/analysis/REPORT.md` | Monte Carlo estimator claim; advance to synthetic oracle validation |
| E17 | Progressive post-freeze external evaluation | Mixed/conditional | Progressive is slightly better on matched Randi AUROC but slightly worse on Cook; SBTG-published remains strongest contextually | `results/compatibility_path_response/postfreeze_progressive_external_20260827/REPORT.md` | Convergent validity only; do not claim anatomical or causal identification |
| E18 | Onset-aware lagged propagation | Exploratory negative for mechanism; positive neuron-identity signal | Known onsets reveal strong persistence and some source-specific temporal structure, but no onset-specific or incremental matrix forecast and no multiplicity-robust Bentley lag | `results/compatibility_path_response/onset_aware_propagation_20260827/REPORT.md` | Use to separate onset prediction from molecular/anatomical and physical-delay claims |
| E19 | Multi-lag conditional dynamics and temporal-cut SMC | Completed negative | Self history wins the conditional-density gate; higher-particle temporal-cut confirmation adds -0.00005 Spearman and has 35.6% compatibility validity; post-freeze Randi/Cook are descriptive and no Bentley temporal cell survives correction | `results/multilag_temporal_cut_20260827/REPORT.md` | Use as the canonical multi-lag/temporal-cut null and freeze its statistic for a prospective cohort |
| E20 | Residualized distributed-lag dynamics | Completed mixed/negative mechanism | A globally frozen 4-second group-shrunk lag tensor improves confirmation Student-t NLL by +0.009436 and edge-rank stability to 0.764, but worsens RMSE; quiet exceeds onset and Randi/Cook/Bentley correspondence is near chance | `results/distributed_lag_dynamics_20260828/FINAL_REPORT.md` | Use as predictive distributional lag evidence only; do not promote SMC, anatomy, or physical-delay claims |
| E21 | Higher-order conditional lag dynamics across 54/80/bridge cohorts | Completed calibrated ceiling | Neural densities use aligned source history, but structured mean/scale/covariance lag gates fail; weak variance effects are undetectable, extracted external matrices are weak, and no matched receptor result survives global BH correction | `results/higher_order_dynamics_20260828/FINAL_REPORT.md` | Use as the cross-cohort higher-order lag ceiling; retain density prediction but do not promote edgewise gain, SMC, receptor, anatomy, or delay claims |
| E22 | Biological analysis of frozen stimulus-conditioned lag dynamics | Superseded by provenance audit | Events were not selected by their raw animal-specific chemical order; the claimed three-butanone repetition/adaptation result is invalid | `AUDIT_STIMULUS_PROVENANCE_20260828.md`; `results/biological_lag_analysis_20260828/CORRECTION.md` | Historical provenance only; rerun separately for butanone, pentanedione, and NaCl |
| E23 | Corrected-onset regularized lag sampling | Historical estimator evidence / biological interpretation superseded | Direct and ESS-SMC mechanics ran, but pooled numeric results used an incorrect OH15500 clock/event duration and are not corrected chemical-specific matrices | `AUDIT_STIMULUS_PROVENANCE_20260828.md`; `results/aligned_lag_response_20260828/analysis/TECHNICAL_REPORT.md` | Preserve sampler implementation evidence only; any chemical-conditioned rerun requires frozen recording-specific schedules and a passing chemical gate |
| E24 | Stimulus provenance correction and chemical-identity rebuild | Both tournaments complete; no chemical-conditioned lag launch | Raw schedules and corrected infrastructure pass; binary wins primary energy 1.077671. In pooled sensitivity chemical one-hot wins energy 1.310290 but fails position-only (5/10 paired wins). No chemical-aware candidate passes every frozen control | `AUDIT_STIMULUS_PROVENANCE_20260828.md`; `CORRECTED_CHEMICAL_EXPERIMENT_20260828.md`; `results/conditional_distribution_benchmark/chemical_encoding_gate_20260828/REPORT.md` | Preserve the corrected predictive null and observed chemistry; E27 may use binary-any-stimulus conditioning plus event stratification, while chemical-conditioned lag inference moves to a larger prospective dataset |
| E25 | Paired common-noise distributional lag audit | Predictive model positive; edge confirmation negative; Bentley comparison weak | Calcium-aware prediction passes but AIY→AIZ fails final SMC proper scoring. Full frozen matrices show modest pooled monoamine correspondence and descriptive transmitter heterogeneity (tyramine log-SD and octopamine mean/tail), but chance neuropeptide/union W1, weaker pooled results than SBTG-published, and no calibrated lag maximum | `DISTRIBUTIONAL_LAG_EXPERIMENT_20260828.md`; `results/distributional_lag_audit_20260828/paired_bentley_correspondence_all_transmitters/REPORT.md` | Retain as descriptive full-matrix correspondence; do not promote a transmitter, edge, or delay; test on the larger prospective cohort |
| E26 | Corrected four-sampler explicit-lag matrices | Estimator positive; biological lag correspondence negative | Progressive bridge has 0.802 compatibility validity, the best generator-seed stability, and lag-1 Randi/Cook AUROCs at or above SBTG-published; no pooled or transmitter-specific endpoint-mean lag maximum survives global correction | `FLOW_REPAIRED_LAG_METHODS_20260828.md`; `results/four_sampler_lag_connectome_20260828/analysis/REPORT.md` | Preferred repaired-flow estimator evidence; matrices are descriptive observational-law responses, not chemical-specific causal edges or physical delays |
| E27 | Full neural prediction atlas and targeted N=128 audit | Atlas complete; primary model-relative effects stable; supplemental stimulus/distributional screen mostly unstable | All 80 raw archives validate; six preselected mean/pathwise effects retain direction at N=128 with 0.966 magnitude agreement, while the separate supplemental screen has 0.141 agreement. Progressive lag-1 equals or exceeds SBTG-published on Randi and all Cook panels; only Bentley neuropeptide and union lag-1 panels survive global BH | `results/neural_prediction_atlas_20260829/README.md`; `results/neural_prediction_atlas_20260829/technical_report/report.html`; `results/neural_prediction_atlas_20260829/gui/atlas_explorer.html` | Use the atlas as a read-only hypothesis generator with explicit support and selection labels; do not claim causal, anatomical, receptor-mediated, chemical-conditioned, independent-confirmation, or physical-delay identification |
| E28 | Historical SBTG80 progressive-bridge sensitivity | Completed negative replication on historical lineage | Progressive is below SBTG-published on lag-1 Randi and all Cook panels with paired source-bootstrap intervals below zero; Bentley W1 point AUROCs are modestly higher for neuropeptide/union but minimum global q is 0.105 | `results/sbtg80_progressive_sensitivity_20260830/REPORT.md` | Historical donor-imputed/pseudo-paired sensitivity only; not a clean larger-cohort replication and not a basis for causal, anatomical, receptor, or delay claims |
| E29 | E27 complete-family significance and sampler-null gate | Complete inference and calibration; no experiment-ready row | Joint four-family max-T leaves 102 baseline-mean edges and 492 cells, but zero strong-support lag-structure edges and zero strict stimulus-modulated discoveries. The sealed N=128 calibration validates 70/70 raw archives and classifies its 24 candidate–metric rows as 15 sampling-limited, 4 sampler-only, 5 indistinguishable, and 0 passing both sampler and quiet-time controls under one exact 64-test max-T family (critical value 3.8246963393338396) | `results/neural_prediction_atlas_20260829/COMPLETE_FAMILY_SIGNIFICANCE_20260830.md`; `results/neural_prediction_atlas_20260829/TARGETED_SAMPLING_NULL_CALIBRATION_20260830.md`; `results/neural_prediction_atlas_20260829/gui/atlas_explorer.html` | Use the schema-v3 explorer's five-rung ladder to separate support, numerical nonzero effect, sampler excess, quiet-time specificity, and biological confirmation; no current row justifies causal, anatomical, delay, or experimental-priority language |
| E30 | Matched uncertainty, dependence, and shape audit | Complete; predictive gains with limited dependence contribution | Shuffling preserves the flow energy lead; changing variance and Student-t improve marginal CRPS by 0.248% and 0.500%; flow beats Student-t energy by 4.197% but overpredicts large innovations (10.38% versus 5.91% observed). All 100 archives and 289 tests validate | `results/distribution_structure_20260831/INSIGHTS.md`; `results/distribution_structure_20260831/analysis/REPORT.md` | Separate marginal/mean quality from joint uncertainty; do not infer neuronal independence or unique shape attribution; validate tail-based downstream claims |

The machine-readable version of this table is `EXPERIMENT_REGISTRY.csv`.

## Result hierarchy

### Tier A — load-bearing now

- Conditional-variance predictive NLL improvement on real worms.
- ACMMA exactness and stability improvement.
- Matched synthetic conditional-variance recovery.
- Bounded-feature/path-order results under explicitly declared synthetic conditions.
- The comprehensive negative for lag-resolved anatomical/receptor neuromodulator recovery.
- The E23 direct/ESS-SMC implementation behavior as historical estimator
  evidence only; its pooled numeric and biological conclusions require a
  corrected-schedule, chemical-gated recomputation and are not E27 results.

### Tier B — useful with explicit scope

- Holistic anatomical/receptor AUROC comparisons.
- Frozen-v2 matched mean, dispersion, tail, and high-signal teacher-forced response results.
- Finite G1/G4 contrasts and G8 typed/path development results.
- Distributional SID E6/E8/E9 post-hoc rescues, which remain conditional on model, horizon, dimensionality, and representation.
- Empirical SID target-specific direct/matched-law successes.
- E12 direct-versus-composed closure diagnostics: calibrated nulls and strong misspecification detection, with underpowered short-memory rejection.
- E27's six selection-conditioned N=128 primary effects and post-freeze
  connectome comparisons, with the supplemental distributional/stimulus nulls
  displayed separately and every observational-law limitation retained.

### Tier C — do not promote without redesign

- Unrestricted pointwise history gradients.
- Generic covariance/skew finite adapters before an oracle adapter beats the zero baseline.
- Generic synaptic-gating mixed derivatives.
- Latent or receptor-specific causal claims from passive calcium prediction.
- Edgewise stimulus-recruited lag claims from the E23 onset-minus-quiet matrices; no cell passes the frozen cross-sampler/seed/timing rule.
- The E25 onset/8-second boundary cell and AIY→AIZ scale sensitivity; direct
  sampling is stable, but the proposal-valid N=256 ESS-SMC proper-score gate
  fails.
- The E29 targeted sampler-null cells. Four candidate–metric rows exceed the
  same-state SMC controls, but none also exceeds the matched quiet-time
  response; fifteen of 24 rows are support- or sensitivity-limited. Preserve
  them as calibrated frozen-model hypotheses, not experiment-ready effects.
- The residual-variance changepoint statistic as a channel-specific dispersion detector.

### Tier D — provenance only

- `sid_elegans/output/archive/logslope_metric/` and the original log-slope lag conclusion.
- `paper/main_v1_logslope_snapshot.tex`.
- `history_tangent_benchmark/results/revised_note_v1_base_20260712_adjudication_superseded/`.
- Developmental smokes, tuning runs, packaging attempts, and repeated `_v1`/`_v2` folders unless a canonical report explicitly selects them.
- `reports/distributional_sid_decisive_2026-07-14/` for E6/E8/E9 decisions; it is superseded by the later `distributional_sid_adjudication_2026-07-14/` report.
- `paper/main.pdf` for the final neuromodulator conclusion; it predates ACMMA/all-data/amines completion.
- The E22 biological report, position-encoding experiment, position-conditioned lag archives, and SBTG-published 80-neuron lineage for chemical/coupling claims; see E24.

## Canonical supersession rules

Use these rules whenever two artifacts disagree:

1. **Later adjudication beats earlier confirmation** when it explicitly audits or repairs the earlier design.
2. **Confirmed/frozen runs beat smoke, tuning, development, and post-hoc search runs.** Post-hoc results may diagnose a failure but cannot silently upgrade a preregistered claim.
3. **Machine-readable summaries beat hand-copied headline numbers.** Prefer `results_snapshot.json`, claims registries, validated seed-level summaries, and run manifests.
4. **All-data and stronger-control results beat earlier favorable subsets.** This is why the ACMMA neuromodulator null supersedes the six-worm 80%-row lead.
5. **Claim-aware metrics beat broad pooled leaderboards.** Keep predictive law, marginal activity, conditional rule/effective coupling, reliability/variance, latent recovery, and controlled intervention claims separate.

## Folder-level reading guide

| Folder | Role | How to use it |
|---|---|---|
| `sid_elegans/` | Real neural analysis | Start with `output/biolag/ALLDATA_RESULT.md` and `output/newlevers/VALIDATION.md` |
| `distributional_sid/` | Bounded-feature/orthogonal-score synthetic program | Start with the later adjudication report, not individual run folders |
| `empirical_sid/` | Target-by-target local/dynamic validation | Start with the July 14 validation report and claims registry |
| `history_tangent_benchmark/` | Point-gradient and finite-contrast benchmarking | Separate the failed point-tangent lane from the narrower finite-contrast lane |
| `neuromod_benchmark/` | Broad synthetic capability benchmark | Use `frozen_v2` reports by capability; never create one universal leaderboard |
| `analysis/` | Focused G8/stable-SID diagnostics | Development/adjudication evidence; check each validation note and manifest |
| `SBTG/` | Legacy SBTG pipeline and merged results | Historical foundation and reduced-form comparator; claim ceilings differ from SID |
| `reports/` | Human-readable syntheses | Prefer the newest report that explicitly supersedes or adjudicates an older one |
| `paper/` | Publication snapshot | Figures and narrative snapshot; final biological conclusion is currently stale |

## Recommended next experiments

1. Make an oracle-adapter/zero-baseline gate mandatory before training any new learned mechanistic readout.
2. Center the real-data paper on predictive distribution quality, ACMMA, calibration, and honest null results.
3. Continue finite or typed contrasts only where the representation beats zero on oracle data.
4. Stress E8 with deliberately misspecified observation models before any latent-neural deployment claim.
5. Repair changepoint channel specificity using a mechanism-confusion matrix rather than per-scenario power alone.
6. Prefer improved per-neuron signal quality, repeated controlled stimuli, measured global state, or interventions over merely adding more observational worms.
7. Improve closure-defect power prospectively or derive a sequentially orthogonal composed-side estimator; do not tune the completed E12 confirmation.

## Scope and evidence notes

- “Supported” means supported within the tested estimand, DGP, cohort, and information set. It does not imply biological validity or causal identification.
- Low localization error is not calibrated detection. Changepoint work should report both localization and significance.
- Hidden-driver or conditional-rule results are reduced-form observed effects, not anatomical rewiring.
- **External changepoint freshness check:** the separate `diffusionCircuit` repository has newer score-dynamics probe runs through 2026-07-07. On 20 imputed neural onsets, detected-within-5s rates were 0.35 for the pooled-history DSM reliability scan, 0.30 for Hotelling mean CUSUM, 0.15 for the score-dynamics variance readout, and 0.00 for the SBTG conditional-rule and score-dynamics joint-law/mean readouts. The six-event complete-cohort sensitivity was weaker. These small probe runs keep E13 in the mixed/conditional tier; they do not support a latent anatomical-rewiring claim.
- The original portfolio synthesis was assembled through 2026-07-16, including the frozen E12 direct-versus-composed confirmation. Earlier family-level decisions still defer to the July 14 adjudication artifacts where named.
- The compatibility-aware response section was updated through 2026-08-27. Its Stage-A estimator selection was sealed from Randi, Cook, Bentley, and SBTG artifacts; Stage B is a separately labeled post-freeze external evaluation.
- **Author context recorded 2026-08-27:** the SBTG paper's connectome/neuromodulator matching was minimal and did not use known stimulus onset. The new E18 analysis addresses onset directly without retraining, while retaining SBTG-published as a historical matrix comparator.
- **E20 update recorded 2026-08-28:** smooth residualized lag kernels are reproducible under global regularization and pass the proper-score and temporal-alignment checks, but fail the mean-dynamics gate and all post-freeze anatomical/molecular checks. This supersedes any interpretation of a descriptive peak lag as physical transmission delay.
- **E21 update recorded 2026-08-28:** the broader mean/variance/covariance and neural-density search replicates the lag-identification ceiling on both 54 and original 80 neurons. Source history helps density prediction globally, but edgewise higher-order lag effects do not pass proper-score, detectability, or multiplicity gates. This is the canonical cross-cohort answer for higher-order lag dynamics.
- **E22 correction recorded 2026-08-28:** the original chemical labels, three-repeat framing, adaptation analysis, and promoted lag edge are invalid because event position was substituted for animal-specific chemical identity. The corrected observed-data analysis uses actual events and supports chemical-specific population activity descriptions only.
- **E23 correction recorded 2026-08-28:** the sampler mechanics are preserved as historical engineering evidence, but its position-conditioned and clock-misspecified pooled outputs cannot confirm a chemical-specific lag matrix or salvage E22's edge/repetition claims.
- **E24 correction recorded 2026-08-28:** the last sentence of the E23 note and all E22 chemical/repetition claims are superseded. Raw `stims[worm,event]` encodes one of three different chemicals, not repeat number. The corrected primary tournament selects binary-any-stimulus (1.077671 balanced energy); chemical+position misses binary by +0.000495 and chemical one-hot by +0.001538. In the pooled-resampled sensitivity, chemical one-hot ranks first (1.310290) but fails position-only with 5/10 paired wins. Neither cohort has a chemical-aware candidate that passes every frozen control, so a **chemical-conditioned** lag launch remains prohibited. E27 is separately allowed because it conditions on binary stimulus presence and uses true chemicals only as post-sampling event strata. See `AUDIT_STIMULUS_PROVENANCE_20260828.md` for the exact affected/unaffected boundary and quarantined artifacts.
- **E25 recorded 2026-08-28:** common-random-number direct sampling solves much
  of the Monte Carlo instability, and an inner-tuned calcium-aware innovation
  model has small positive held-out predictive value. Nevertheless, its only
  direct cell is at the maximum fitted lag; the sole rollout survivor fails the
  final N=256 ESS-SMC proper-score gate. This is a calibrated negative for a
  promoted lag edge, not evidence that lagged predictive structure is absent.
  The later full-matrix Bentley comparison is valid under a separate descriptive
  claim: it finds a weak short-lag pooled monoamine pattern and heterogeneous
  transmitter-specific peaks (most notably tyramine log-SD and octopamine
  mean/tail), but no improvement over SBTG-published at the pooled level, no
  edge-rank agreement with SBTG, and no multiplicity-robust lag maximum.
- **E27 recorded 2026-08-30:** the full binary-any-stimulus atlas and both N=128
  audits are complete. The six primary mean/pathwise candidates show strong
  selection-conditioned magnitude agreement; the separate stimulus and
  distributional candidates mostly do not. Post-freeze Randi/Cook comparisons
  are competitive and two pooled Bentley panels survive global correction, but
  they remain descriptive receptor-connectome correspondence. Neither those
  AUROCs nor the early lag band identifies activity transmission, causal edges,
  chemical conditioning, or physical delay.
- **E28 recorded 2026-08-30:** the bounded historical 80-neuron progressive
  sensitivity does not preserve E27's Randi/Cook advantage over
  SBTG-published. Bentley W1 retains modest neuropeptide/union point-score
  advantages, but none survives the sensitivity-wide BH family. This is a
  lineage stress test, not a clean-cohort replication, because the released
  80-neuron cache uses donor imputation and index-wise pseudo-pairing.
- **E29 finalized 2026-08-30:** the complete-family analysis and targeted
  sampler-null gate are both sealed. The targeted run contains 70 validated
  raw archives with 73 raw-ledger entries; its analysis contains 24 rows in
  one exact 64-test max-T family, has critical value 3.8246963393338396, and
  has fingerprint
  `c193fc999d504f9096540fa4e5e73aeaebe88bc8e8a7d85a805c9920bd3252ca`.
  Labels are 15 `sampling_limited`, four
  `exceeds_sampling_controls_only`, five
  `indistinguishable_from_sampling_controls`, and zero
  `exceeds_sampling_and_quiet_controls`. The schema-v3 explorer presents these
  gates without promoting any row to experiment-ready status; the final code
  suite reports 270 passing tests.
- **Explorer presentation update, 2026-08-30:** plain navigation, shorter copy,
  and expandable evidence details; no numerical or inference changes. See
  [`EXPLORER_CLARITY_UPDATE_20260830.md`](results/neural_prediction_atlas_20260829/EXPLORER_CLARITY_UPDATE_20260830.md)
  for the payload-invariance check and desktop/mobile QA.
- **E32 recorded 2026-09-02:** longer training resolves the under-convergence
  in the original historical-80 conditional flow. A two-generator ensemble with
  N64 progressive bridge SMC exceeds SBTG-published lag-1 AUROC on Randi and all
  three Cook panels (0.674, 0.629, 0.617, and 0.666). Source-column bootstrap
  intervals exclude zero for all three Cook deltas but narrowly include zero for
  Randi. This supersedes E28's claim that the historical flow cannot recover the
  correspondence point-score advantage; it does not repair E28's invalid
  pseudo-pairing/donor-imputation lineage or support causal or anatomical claims.
  See [`results/sbtg80_flow_optimization_20260901/final/REPORT.md`](results/sbtg80_flow_optimization_20260901/final/REPORT.md).
