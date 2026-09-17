# Paired distributional lag experiment — 2026-08-28

Status: complete. The predictive model gate passed, but the full direct →
rollout → ESS-SMC chain did not. **No lag matrix or edge is promoted.**

## Question and claim boundary

The experiment asks whether changing one source neuron's conditionally plausible
history value at one lag changes the learned conditional distribution of a
target neuron. It uses the corrected 17-worm OH16230 head cohort, 54 neurons,
4 Hz sampling, whole-worm folds, the `binary_any_stimulus` history, and the
frozen recording-specific stimulus schedules.

The output is an observational, model-relative finite contrast. It is not an
intervention, synapse, receptor action, direction of physical transmission, or
transmission-delay estimate. Cook, Randi, SBTG, receptor, transmitter, and
neuromodulator references were excluded from fitting and every selection gate.

## Estimator order

1. **Frozen conditional-flow direct audit.** Select real held-out history
   windows. Fit the conditional source resampler on training worms only. Replace
   just one source coordinate at one lag with conditional residual quartiles.
   Sample factual, low, and high histories with identical base noise.
2. **Distributional audit.** Measure target-wise Wasserstein-1 magnitude and
   signed changes in mean, log standard deviation, and tail probability. Require
   a positive proper-energy penalty on held-out observations, particle/seed
   convergence, worm-split reliability, and lag specificity.
3. **Calcium-aware model.** Because the frozen flow failed its proper-score
   gate, fit an atlas-blind structured model to innovations after a per-neuron
   quiet-period AR(1) calcium approximation. Use smooth distributed-lag bases
   for conditional mean and log variance, reconstruct the observed next-frame
   law, and choose cross-history shrinkage by inner whole-worm validation.
4. **Paired latent direct audit.** Repeat the one-coordinate conditional
   replacement with N=64 nested in N=256 common-normal draws. Lag 40 (10 s),
   outside the fitted 32-frame history, is the structural negative control.
5. **Short paired rollout.** For direct candidates only, generate low, high,
   and factual trajectories using common process noise. After horizon 2 the
   original lag-32 coordinate has left memory, so later differences must be
   mediated by generated model states.
6. **Targeted ESS-SMC.** Test only a rollout survivor. Draw a conditional-source
   proposal, softly weight toward the low/high targets, diagnose ESS and maximum
   weight, use paired systematic resampling, then free-roll with common noise.
   The effect must agree with direct sampling, have a positive proper-score
   interval, retain its signed worm-bootstrap interval, be repeat-consistent,
   and achieve at least 80% valid clamps.

## Metrics

| Metric | Meaning | Role |
| --- | --- | --- |
| Wasserstein-1 | Average marginal distance between low/high target samples | Effect magnitude; not sufficient for selection |
| Mean shift | High-minus-low conditional mean | Signed location effect |
| Log-SD shift | High-minus-low log conditional standard deviation | Signed gain/scale effect |
| Tail-probability shift | Change beyond a training-derived target threshold | Signed tail effect |
| Proper-energy penalty | Energy loss of the low/high mixture minus the factual conditional forecast on the held-out observation | Required evidence that the contrast matters predictively; positive is favorable |
| N64/N256 convergence | Rank agreement and relative error with nested particles | Monte Carlo reliability |
| Worm split/fold reliability | Rank agreement across biological partitions | Biological reproducibility |
| ESS / maximum weight | Effective proposal support and weight concentration | SMC validity, not biological evidence |

Particles are never treated as independent biological replicates. Confidence
intervals and sign consistency use worms as the inferential units.

## Results

### 1. Frozen flow: stable Monte Carlo, no useful contrast

No state×lag cell passed. The largest Wasserstein magnitudes occurred at 0.25 s
(quiet 0.002035, onset 0.002368, active 0.001816) and decayed rapidly with lag.
Seed and worm-split rank correlations were high, but every proper-score gate
failed. This is a useful distinction: the matrices were computationally
repeatable, yet did not improve the held-out conditional forecast. SMC was
therefore prohibited for this model.

### 2. Calcium-aware latent innovation model: predictive gate passed

An unshrunk cross-history model failed all predictive metrics. Inner-validation
selection favored cross-history ridge near 100,000 in every fold. The complete
archived model passed all four held-out gates:

| Full minus base improvement | Mean | 95% worm-bootstrap CI |
| --- | ---: | ---: |
| Gaussian NLL | +0.002916 | [+0.001828, +0.004211] |
| Energy score | +0.001145 | [+0.000906, +0.001407] |
| RMSE | +0.000185 | [+0.000151, +0.000223] |
| Isolated cross-history scale NLL | +0.001913 | [+0.000856, +0.003455] |

Median pairwise fold Spearman was 0.484 for mean kernels and 0.526 for scale
kernels. These are modest predictive gains, not latent-state recovery.

### 3. Paired direct audit: one boundary cell

The strict audit passed one of 21 state×lag cells: onset at lag 32 (8 s). Its
mean Wasserstein-1 magnitude was 0.000065 and proper-energy penalty was
+0.000008 with CI [+0.000001, +0.000016]. N64/N256 Spearman was 1.000,
relative error 0.009, and worm-split Spearman 0.957.

Nine source→target contrasts inside that cell cleared edge-level proper-score,
signed-effect BH q<0.10, at least 70% worm sign agreement, and particle
convergence gates: RME→AIM, ASK→FLP, URY→AIY, URA→AIY, ALA→RID, AIY→AIZ,
ALA→IL1, IL2→AWA, and ALA→RMF. Eight were primarily scale effects and one
(ASK→FLP) was a mean effect. Their Wasserstein magnitudes were only about
0.00010–0.00021.

This cell occurs exactly at the maximum fitted lag. That boundary position is a
specific artifact risk and prevents a delay interpretation even before the
downstream failure.

### 4. Paired generated-state rollout: one provisional survivor

Of 45 edge×horizon tests, only AIY→AIZ at the 4-second forecast horizon passed:
log-SD shift −0.000015, proper-energy penalty +0.001128, Wasserstein-1
0.000048, and signed BH q=0.0005. The effect is a minute reduction in the
model's conditional scale, not a mean-response edge. It authorized only a
targeted SMC sensitivity.

### 5. ESS-SMC confirmation: failed

The initial N=64 run agreed in sign and magnitude but had only 68.0% valid
event/repeat clamps and a proper-score interval crossing zero. Increasing to
N=256 was a mechanical ESS sensitivity; it changed no target, bandwidth,
contrast, confidence gate, or biological data. It achieved 100% validity and
median ESS about 92–94.

The N=256 signed estimate remained close to direct sampling: log-SD shift
−0.000019, CI [−0.000032, −0.000009], versus direct −0.000015; all three Monte
Carlo repeats had the same sign and the magnitude ratio was 1.215. However, the
proper-energy penalty was −0.000226 with CI [−0.000775, +0.000332]. The full
gate therefore failed.

## Decision

The calcium-aware model learns a small amount of reproducible multi-lag
predictive structure. Common-random-number sampling also succeeds at reducing
Monte Carlo noise. But there is no validated lag matrix: the only candidate is
at the maximum modeled lag, is extremely small, and fails the independent
proper-score SMC confirmation after proposal validity is repaired.

Accordingly:

- do not compare this candidate post hoc with Cook, Randi, SBTG, or molecular
  atlases as if it were a promoted matrix;
- do not label 8 s + 4 s as a physical delay;
- retain AIY→AIZ only as a documented failed/provisional sensitivity;
- do not tune more SMC parameters against this observed outcome.

## Post-freeze SBTG/Bentley correspondence

The proper-score decision above concerns promotion of individual biological
edges. It does **not** prohibit the original SBTG-style descriptive question:
whether the complete frozen lag matrices rank Bentley receptor edges above
non-edges. That comparison was therefore run separately, without edge
thresholding or retraining, on the exact shared 54-neuron order.

Bentley class-level monoamine and neuropeptide data are binary directed
receptor-edge existence after duplicate receptor rows are collapsed. The
primary analysis uses absolute matrix magnitude and AUROC among eligible
neuromodulator source columns; the original paper-style all-off-diagonal panel
is retained as a sensitivity. The paired sampler's primary object is the
state-averaged Wasserstein-1 matrix; mean, log-SD, and tail matrices remain
separate rather than selecting a channel on Bentley performance.

| Matrix | Best monoamine AUROC | Best neuropeptide AUROC | Best union AUROC |
| --- | ---: | ---: | ---: |
| Paired Wasserstein | 0.525 at 0.25 s | 0.491 at 0.50 s | 0.493 at 0.50 s |
| Paired mean | 0.544 at 0.25 s | 0.499 at 0.50 s | 0.506 at 1.00 s |
| Paired log-SD | 0.539 at 0.50 s | 0.522 at 1.00 s | 0.518 at 1.00 s |
| Paired tail | 0.523 at 0.50 s | 0.507 at 0.25 s | 0.506 at 1.00 s |
| SBTG-current | 0.567 at 4.00 s | 0.570 at 4.00 s | 0.605 at 4.00 s |
| SBTG-published | 0.581 at 0.25 s | 0.544 at 0.25 s | 0.537 at 0.25 s |

Thus the new sampler recovers a modest short-lag monoamine pattern, especially
in its mean matrix, but it is not better overall. Its full-distribution
Wasserstein matrix is near chance for neuropeptide and the union. After adding
the transmitter-specific panels below, none of the new
channel×state×network profiles survives within-source lag-max permutation and
global BH correction (minimum raw p 0.233; minimum q 0.933). The seemingly
larger SBTG-current union peak also does not survive the same complete family
correction (raw p 0.031; q 0.933).

### Transmitter-specific correspondence

The monoamine reference was also split into dopamine, serotonin, tyramine,
and octopamine, as in the original SBTG analysis. Values below are the best
eligible-source AUROC across lags; parentheses give the raw within-source
lag-max permutation p-value.

| Matrix | Dopamine | Serotonin | Tyramine | Octopamine |
| --- | ---: | ---: | ---: | ---: |
| Paired Wasserstein | 0.549 at 0.25 s (0.482) | 0.456 at 0.25 s (0.931) | 0.575 at 4 s (0.489) | 0.633 at 4 s (0.318) |
| Paired mean | 0.577 at 1 s (0.443) | 0.579 at 0.25 s (0.624) | 0.560 at 4 s (0.697) | 0.647 at 2 s (0.373) |
| Paired log-SD | 0.589 at 0.5 s (0.315) | 0.535 at 1 s (0.930) | 0.640 at 2 s (0.233) | 0.553 at 1 s (0.892) |
| Paired tail | 0.597 at 8 s (0.234) | 0.580 at 0.5 s (0.580) | 0.541 at 2 s (0.830) | 0.642 at 2 s (0.411) |
| SBTG-current | 0.604 at 4 s (0.192) | 0.649 at 0.5 s (0.274) | 0.591 at 0.25 s (0.573) | 0.567 at 1 s (0.808) |
| SBTG-published | 0.577 at 0.25 s (0.557) | 0.677 at 5 s (0.154) | 0.593 at 0.5 s (0.673) | 0.669 at 0.75 s (0.433) |

The descriptive pattern is heterogeneous: the paired scale channel is most
interesting for tyramine (0.640 at 2 s), while octopamine is strongest in mean
and tail changes (0.647 and 0.642 at 2 s). Dopamine has weaker location,
scale, and tail correspondence; serotonin is weak for the Wasserstein matrix
despite stronger SBTG values. This is useful for forming hypotheses about
distributional or gain-like effects, but it is not calibrated evidence of
transmitter action: dopamine has only two eligible source neurons and each
other panel has one. Source-bootstrap intervals are therefore deliberately
suppressed, and no transmitter-specific maximum survives the global family
correction (all q = 0.933).

The edge rankings are different, not merely rescaled: at lag 1, Wasserstein
absolute Spearman is 0.072 versus SBTG-current and 0.043 versus
SBTG-published, with top-10% Jaccard 0.069 and 0.106. Relative to
SBTG-published, lag-1 Wasserstein AUROC is lower for neuropeptide by -0.067
(source-bootstrap CI [-0.121, -0.013]) and union by -0.053
([-0.102, -0.004]).

This answers the intended correspondence question: the new sampling matrices
contain a weak, short-lag monoamine resemblance, but do not reproduce or
improve the broader SBTG–Bentley pattern. It remains descriptive molecular
correspondence, not evidence of receptor activity or physical delay.

The clean next test is prospective boundary sensitivity: refit a frozen
12–16-second calcium-aware history so 8 s is an interior lag, freeze the same
paired contrast and proper-score gates, and test it on additional worms (the
DANDI 000981 cluster cohort is the intended scale-up). Chemical identity should
be balanced or separately controlled rather than inferred from epoch position.

## Corrections and supersession trail

- `latent_paired_crn_N64_N256/` incorrectly labeled lag 32 as a structural
  control even though lag 32 belongs to the latent model. Its `CORRECTION.md` is
  authoritative; the corrected audit uses lag 40.
- `latent_paired_crn_N64_N256_corrected/` used a provisional loose edge gate and
  is superseded by `latent_paired_crn_N64_N256_strict/`.
- `latent_targeted_ess_smc/` completed particles but aborted during reporting on
  a column-name mismatch. It has no scientific status.
- `latent_targeted_ess_smc_v2/` is the valid N=64 failed sensitivity.
- `latent_targeted_ess_smc_N256/` is the canonical, proposal-valid SMC result.
- `latent_calcium_structured_tuned_primary_v2/` is the canonical predictive
  model because it includes the complete fold parameter archives.

## Canonical artifacts

| Artifact | Purpose |
| --- | --- |
| `results/distributional_lag_audit_20260828/analysis_N32/REPORT.md` | Frozen-flow direct audit and stop decision |
| `results/distributional_lag_audit_20260828/latent_calcium_structured_tuned_primary_v2/` | Predictive model, fold parameters, tuning traces, metrics, and gate |
| `results/distributional_lag_audit_20260828/latent_paired_crn_N64_N256_strict/` | Strict paired direct matrices, cell gates, and nine candidates |
| `results/distributional_lag_audit_20260828/latent_short_paired_rollout/` | Generated-state rollout and one provisional survivor |
| `results/distributional_lag_audit_20260828/latent_targeted_ess_smc_N256/` | Canonical failed targeted SMC confirmation |
| `results/distributional_lag_audit_20260828/paired_bentley_correspondence_all_transmitters/` | Canonical pooled and transmitter-specific Bentley lag profiles and shared-neuron SBTG comparison |
| `results/distributional_lag_audit_20260828/checksums.sha256` | Integrity inventory |
| `compatibility_neural_benchmark/distributional_lag_audit.py` | Frozen-flow paired sampler |
| `compatibility_neural_benchmark/latent_distributional_audit.py` | Latent paired direct audit |
| `compatibility_neural_benchmark/latent_paired_rollout.py` | Common-noise generated-state rollout |
| `compatibility_neural_benchmark/latent_targeted_smc.py` | Targeted ESS-SMC sensitivity |
| `compatibility_neural_benchmark/paired_lag_correspondence.py` | Post-freeze SBTG/Bentley matrix comparison |
| `conditional_neural_benchmark/latent_calcium_lag.py` | Calcium-aware structured conditional law |

Verification: the complete conditional and compatibility suite passes: **77
tests passed**, with one non-failing PyTorch Transformer warning.
