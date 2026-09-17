# What the experiments say now

## Technical summary

The project has not been a general failure. It has produced three credible methodological results: SID can recover a synthetic conditional-variance kernel, ACMMA validly uses incomplete multi-worm observations without imputation, and conditional variance improves held-out real-data likelihood. What has repeatedly failed is the stronger biological/mechanistic claim: a robust, lag-resolved, directed neuromodulator signature recoverable from these calcium traces and the current anatomical/receptor targets.

The newest experiments make the reason clearer. Ordinary predictive fit does not calibrate the derivatives, physical fields, or latent mechanisms being read out from the fitted law. In the history-tangent benchmark, 96/96 convergence-sweep fits completed and passed optimization checks, yet six preregistered derivative-readiness gates failed. In frozen-v2, flexible densities predict well and some mechanism-matched channels are partly recoverable, but synaptic gating, latent recovery, and controlled intervention recovery remain weak. The failure is therefore primarily an estimand/identification and observation problem, compounded by limited effective sample size—not just an optimizer or architecture problem.

The program should narrow rather than expand: make held-out predictive distribution quality the primary real-data claim; require oracle-adapter and zero-field gates before training learned mechanistic readouts; use finite, supported contrasts or targeted channel functionals instead of full pointwise score derivatives; and reserve physical-mechanism claims for intervention-identified or functionally anchored targets.

## What actually worked

| Result | Evidence | Assessment |
|---|---|---|
| Synthetic distributional channel | SID gain recovered an injected conditional-variance kernel at correlation 0.998 | Strong method validation, but only for the matched synthetic estimand |
| All-data estimation | ACMMA reproduced the complete-case estimator at Pearson/Spearman 1.000 and increased gain split-half stability from roughly 0.29 to 0.46–0.56 | Strong estimator/infrastructure result |
| Real predictive value | Modeling conditional variance improved leave-one-worm-out NLL by 0.364 on 28 worms; all targets improved and the interval excluded zero | Strongest real-data result; predictive, not mechanistic |
| Correct granularity | Ganglion aggregation increased gain stability from about 0.25 to 0.54 and beat random-pooling controls | Useful measurement result, but mean channels improved similarly |
| Mechanism-matched synthetic channels | Frozen-v2 partly recovered additive mean and stochastic-dispersion fields; the best reported nISE values were about 0.19 and 0.44 respectively, versus a zero-field baseline of 1 | Real partial success when estimator and truth channel match |
| Tail prediction | MDNs strongly improved held-out NLL and tail-operator error in the tail suite | Predictive-distribution success; it did not translate into physical tail-field recovery |
| Dedicated changepoint probes | In matched-null frozen-v2, the matched mean and residual-tail probes detected and localized 100/100 designed changes | Strong synthetic detection in high-signal, channel-matched settings |

## What did not hold up

### The biological lag signature did not replicate

The original peptidergic gain-slow result depended on an 80% row split. It fell from dBC +3.35 to +1.16 when all rows from the same six worms were used, and to -0.41 with all 28 worms under validated ACMMA. It also collapsed after global-mode removal and did not reproduce under denoising score matching, neural DSM, or variance-VAR. All monoamine and peptide confidence intervals included zero. This is a comprehensive negative for the current lag-resolved anatomical/receptor-target claim.

The newer functional-atlas work does not rescue the claim. Static gain correspondence was weakly above a circular-shift null (AUROC 0.555, p=0.025) but lost to pairwise Pearson (0.621); the novel kinetics test was null (Spearman 0.015, p=0.30). The root README and PROGRESS file still call this test untried, so those indexes are stale relative to the new-levers artifacts.

### The pointwise history-tangent route is not calibrated

The convergence sweep completed 96 fits with no failed selected cases, all fixed-batch overfit checks passed, and no best checkpoint landed at the final epoch. Nevertheless, G1 tangent NRMSE was 0.589 for the Gaussian model and 0.719 for the MDN, G2's best clean tangent NRMSE was 1.068, G3's was 1.167, and centering gates failed. Exact replay reproduced the decision. More epochs or more seeds would make this failure more precise, not fix it.

The finite-contrast V1 experiment is promising only in restricted regimes. At delta 0.1 across three seeds, the best non-oracle witness NRMSE was about 0.60 for the G1 location change and 0.67 for the G4 low-rank mixture, but about 1.07 for G2 covariance and 1.02 for G3 skew. Crucially, even oracle-sample signed classifiers and oracle-dictionary Riesz adapters were near or above NRMSE 1 for G2/G3. That implicates the adapter, response dictionary, and signal geometry before it implicates the fitted predictive model.

### Frozen-v2 separates prediction from mechanism recovery

All 15 suites completed: 8,937 verified result records, 11,552 planned fit invocations, and no execution failures. That is excellent infrastructure, but the substantive picture is mixed:

- MDNs and Student-t regression often beat SID/DSM on held-out NLL and tail prediction.
- Mechanism-matched mean and dispersion fields were partly recoverable.
- Synaptic-gating field recovery failed: the zero-field baseline had nISE 1, while SID variants ranged from roughly 3.15 to 7.23.
- Latent neuromodulator models had worse calcium NLL than ridge baselines and only moderate latent/kinetic recovery.
- High-signal teacher-forced intervention curves were partly recovered, but controlled common-history knockout nRMSE remained roughly 0.63–0.92.
- The new changepoint suite showed excellent matched mean/tail recovery, but the variance detector was not channel-specific: it called 95% of mean changes and 99% of matched-tail changes while localizing neither correctly. Independent-null FPR was also 11% for energy distance and 9% for MMD at a nominal 5% target.

## Why the same pattern keeps recurring

1. **The evaluation target is often downstream of the fitted objective.** NLL or denoising loss constrains the conditional law in an average sense. A history derivative, mixed derivative, physical field, or changepoint statistic can be badly wrong while predictive loss is good.

2. **The biological target is not the observed estimand.** Calcium traces can support observed activity changes, reduced-form history-to-future rules, reliability changes, and predictive likelihood. They do not by themselves identify latent anatomical rewiring or a receptor-specific neuromodulator field.

3. **Calcium and preprocessing erase the relevant geometry.** Slow filtering, deconvolution error, missing neurons, shared brain state, and standardization compress or entangle the timescales and conditional higher moments the project wants to attribute. The self-gain work shows that per-neuron reliability, not animal count alone, is the limiting factor.

4. **The problem is high-dimensional relative to independent information.** Many time points do not equal many independent interventions or animals. Flexible models can improve prediction yet have unstable Jacobians, Hessians, or channel decompositions. Simple ridge, Student-t, Pearson, and MDN baselines repeatedly win because they estimate lower-complexity objects.

5. **Several readouts are not orthogonal to nuisance channels.** The variance changepoint detector reacts strongly to mean and tail changes; global-mode removal destroys the original lag result; aggregation helps every channel. These are signatures of non-identification or leakage between claim axes.

6. **Early exploration created fragile winners.** The 80% split, ridge choice, lag choice, target choice, and global-mode handling were all consequential. Later controls correctly converted apparent positives into nulls. The solution is frozen gates and independent confirmation, not another broad search over method variants.

## Highest-value next work

### 1. Make the oracle-adapter gate mandatory

Before fitting any learned conditional law, apply the proposed statistic to oracle samples or oracle log densities. Require it to beat a zero-function baseline with margin on every intended mechanism. G2/G3 currently fail this gate. Until the oracle classifier/Riesz adapter works there, model sweeps are not informative.

For covariance and skew, optimize the mechanism-matched channel functional directly instead of whole-witness NRMSE. Use a fixed response dictionary with cross-fitting, explicit conditional centering, and channel-specific calibration. If the intended scientific claim is a variance or tail readout, the primary loss should score that readout.

### 2. Keep finite contrasts; deprioritize unrestricted point gradients

Finite `h +/- delta*v` contrasts are supported by actual neighborhoods and avoid some derivative amplification. Continue only the G1/G4 lanes initially, add derivative-aware regularization and centering penalties as comparators, and preserve the current readiness thresholds. Do not launch the full confirmatory H1–H7 matrix until a reduced G1–G4+G6 matrix clears them.

### 3. Repair channel orthogonality in changepoint detection

Construct and report a mechanism-confusion matrix, not just per-scenario power. A detector earns a channel label only if it detects its target and stays null or unlocalized under matched nuisance changes. Replace the residual-scale statistic with a genuinely tail-robust scale estimator, residualize mean changes more aggressively, and increase independent calibration pools. The current variance detector is useful as a broad instability alarm, not as evidence of dispersion.

### 4. Reframe the real-data paper around predictive distribution quality

The defensible real-data claim is that conditional variance adds held-out predictive value. Expand that result with strict leave-one-worm/strain-out evaluation, calibration/PIT checks, comparisons against Student-t and MDN baselines, and decomposition by neuron/ganglion and behavioral state. Treat anatomical, receptor, and functional correspondence as secondary localization analyses with multiplicity correction.

### 5. Use interventions or anchors for physical-mechanism claims

For latent neuromodulator models, equalize information sets before ranking methods. Use single-modulator or label-anchored simulations first; then require prospective stimulus, ligand, knockout, or receptor-perturbation generalization. Passive prediction from calcium history should remain P1/reduced-form evidence, not C1 physical identification.

### 6. Improve signal quality before adding more animals

The self-gain validation suggests that more worms alone will not fix neuron-level reliability. Higher temporal resolution, repeated controlled stimuli per worm, better simultaneous coverage, explicit behavior/global-state measurement, or targeted reporters/interventions are more valuable than a modest increase in observational animals.

## Practical stop/go rules

- **Go:** predictive NLL/calibration, ACMMA, matched synthetic mean/dispersion channels, finite G1/G4 contrasts, dedicated matched tail detection.
- **Revise first:** covariance/skew finite adapters, latent recovery, controlled knockout response, dispersion changepoints.
- **Stop as primary claims:** lag-resolved anatomical neuromodulator recovery, unrestricted point-gradient benchmarking, and broad architecture sweeps selected by downstream oracle metrics.

## Confidence and caveats

This synthesis is ready to share as a technical project review, with two caveats. First, the history-tangent and revised-note experiments are developmental and have only three generator seeds in the finite-contrast aggregation; they are readiness evidence, not confirmatory hypothesis tests. Second, frozen-v2 is synthetic and establishes recoverability under its DGPs, not biological validity. These caveats strengthen rather than weaken the central conclusion: the next bottleneck is aligning the claim, estimand, observation process, and validation gate.

Key inspected sources: root `README.md` and `PROGRESS.md`; `sid_elegans/output/biolag/ALLDATA_RESULT.md`; `sid_elegans/output/newlevers/{RESULTS,VALIDATION}.md`; `history_tangent_benchmark/{PREEXECUTION_AUDIT,REVISED_NOTE_V1_AUDIT}.md`; the convergence-sweep and finite-contrast result artifacts; and all completed `neuromod_benchmark/outputs/frozen_v2` suite reports, summaries, and matched-null changepoint results.
