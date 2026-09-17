> Portable reading copy of [FLOW_REPAIRED_LAG_METHODS_20260828.md](../../archive/original/FLOW_REPAIRED_LAG_METHODS_20260828.md). Scientific text is preserved; local links are relocated. The frozen original remains authoritative.

# Flow-repaired response sampling and explicit lag matrices

This is the dated **E26 mathematical four-sampler specification**. Its consolidated results are in the E26 technical report *(historical link unavailable; see original)*. For the later, larger atlas and calibrated inference, use the E27/E29 atlas *(historical link unavailable; see original)*; for the current synthesis, sourced class summaries, and 54-versus-80 provenance, use the 31 August E31 report *(historical link unavailable; see original)*. Those later reports adjudicate current claims without rewriting the historical numbers below.

## Technical summary

This document defines the complete conditional-generative and repaired-response pipeline used to turn a learned neural transition law into source-by-target lag matrices. It distinguishes the four named workflows—direct importance weighting, terminal SMC, progressive bridge SMC, and temporal-cut SMC—while making an important taxonomy correction: the first three are Monte Carlo estimator designs, whereas a temporal cut is primarily the placement of a source constraint earlier than the prediction boundary. In the implemented comparison, “temporal-cut SMC” means the lagged, ESS-resampling version of bootstrap SMC, so it remains operationally distinct from terminal-deferred SMC.

The current experiment uses the corrected 17-worm OH16230 head cohort at its native 4 Hz, 54 complete-case neurons, worm-specific stimulus schedules, and the cross-validated binary-any-stimulus conditional flow. The stimulus is binary because that encoding won the atlas-blind predictive comparison; event identity is nevertheless preserved in the schedule and in the three repeated-event cells. The current experiment does not reuse the older 20-worm wide-flow checkpoints because those combined 17 OH16230 recordings with 3 resampled OH15500 recordings and predated the stimulus-provenance correction.

The primary matrix is the change in the one-frame-ahead conditional mean of each target after replacing one source's lagged one-second summary from a training-fold low state to a training-fold high state. Secondary matrices measure cumulative mean, peak, event probability, and conditional standard deviation. Every matrix is stored with **target on rows and source on columns**.

These are observational, model-relative repaired-path responses. They are not randomized physical interventions, direct synapses, receptor activation measurements, or estimates of molecular transmission delay.

## What is learned

Let \(Y_t\in\mathbb{R}^{54}\) be standardized neural activity at frame \(t\), let \(U_t\) be the stimulus encoding, and let

\[
H_t=(Y_{t-L+1:t},U_{t-L+1:t}),\qquad L=80\text{ frames}=20\text{ s}.
\]

The frozen flow learns a one-step conditional distribution

\[
p_\theta(Y_{t+1}\mid H_t).
\]

Training and model selection are grouped by worm. The current checkpoint family is:

`stim_binary_any_stimulus_tcn_flow128_dropout15_jitter_p01`

There are five held-out-worm folds. Each fold's nuisance quantities—standardization, anchor projection, source quantiles, and target event thresholds—are estimated only from its training worms.

### Corrected stimulus provenance

- Cohort: 17 OH16230 head recordings.
- Sampling: native 4.0 Hz; no temporal resampling.
- Stimulus schedule: recovered for each recording rather than assigned from a global template.
- Repeated events: three per worm, with chemical identity and event order retained in metadata.
- Model input: binary “any stimulus active” at every history frame.
- Selection firewall: Randi, Cook, Bentley, and SBTG artifacts did not enter flow training or selection.

The binary encoding means that two active chemicals share the same input value. It does **not** mean stimulus windows were reconstructed as a generic `000...111` template independently of the recording. The exact per-worm onset and offset times determine every input vector and analysis cut.

## The repaired-path estimand

Choose a prediction cut \(c\), a source neuron \(j\), a one-second source window of width \(w=4\) frames, and a lag \(\ell\). Here \(\ell\) is defined as the number of frames from the **end of the source window** to the cut. Thus

\[
W_{c,\ell}=\{c-\ell-w+1,\ldots,c-\ell\}.
\]

The current lag grid is \(\ell\in\{1,4,8,16\}\), corresponding to 0.25, 1, 2, and 4 seconds. This is a genuine source-history lag. It is not the earlier dynamic-response horizon, which instead asked how far into the future a response was summarized.

For a simulated repair prefix \(Z\), define the source statistic

\[
A_j(Z)=\frac{1}{w}\sum_{r\in W_{c,\ell}}Z_{rj}.
\]

Training-fold phase-specific quartiles provide \(q_{j,0.25}\) and \(q_{j,0.75}\). The low and high source potentials are

\[
G_j^a(Z)=\exp\left[-\frac{1}{2}
\left(\frac{A_j(Z)-q_{j,a}}{b_j}\right)^2\right],
\quad a\in\{0.25,0.75\},
\]

with \(b_j=\max(0.25\,\mathrm{IQR}_j,0.10)\).

An anchor potential keeps the rest of the simulated population near the observed episode:

\[
G_{\mathrm{anchor},j}(Z)=
\exp[-\lambda C_j(Z,Y)],\qquad \lambda=0.25.
\]

The cost \(C_j\) is computed in a rank-12 training-fold principal subspace. The designated source coordinate is omitted from this anchor **only inside its declared source window**; before and after that window it is anchored like the rest of the population. This avoids directly penalizing the requested source change while limiting unsupported whole-population drift.

The model-relative repaired law is

\[
Q_j^a(dZ)\propto P_\theta(dZ\mid H_{c-w-\ell},U)
G_{\mathrm{anchor},j}(Z)G_j^a(Z).
\]

For future feature \(\phi_k\) of target \(k\), the unnormalized contrast is

\[
\Delta_{kj}^{(\ell)}=
\mathbb{E}_{Q_j^{0.75}}[\phi_k]-
\mathbb{E}_{Q_j^{0.25}}[\phi_k].
\]

The saved effect divides event-wise by the achieved rather than requested source displacement:

\[
M_{kj}^{(\ell)}=
\frac{\Delta_{kj}^{(\ell)}}
{\max(|\widehat A_{j,\mathrm{high}}-\widehat A_{j,\mathrm{low}}|,0.10)}.
\]

This produces a response per achieved standardized source unit. Repeated events are averaged within worm and phase; worms are then averaged with equal weight. The source-to-target sampler tensor is transposed exactly once at the analysis boundary to produce the external-comparison convention `[target, source]`.

## The four sampling workflows

### 1. Direct importance weighting

Direct importance first generates a natural bank of complete repair-and-future paths from the flow. No source condition is imposed during generation. For every source \(j\), the same bank is scored under low and high source potentials and the source-specific population anchor:

\[
\widetilde w_{n,j}^{a}=G_{\mathrm{anchor},j}(Z_n)G_j^a(Z_n),
\qquad
w_{n,j}^{a}=\frac{\widetilde w_{n,j}^{a}}
{\sum_m\widetilde w_{m,j}^{a}}.
\]

The response is a pair of self-normalized weighted averages over the future portions of those paths.

Algorithm:

1. Start from the real boundary history and its exact stimulus history.
2. Generate \(N\) complete paths from the frozen flow.
3. Compute the source statistic, source-excluded anchor cost, and low/high weights for every source.
4. Normalize weights separately for low and high.
5. Compute weighted future mean, cumulative mean, peak, event probability, and standard deviation.
6. Subtract low from high and divide by the achieved source gap.

Example: if a requested high state is common among natural paths, many particles contribute and direct importance is accurate and fast. If only one of 32 paths resembles that state, its normalized weight can approach one, effective sample size collapses, and the result becomes sensitive to that single future trajectory.

Tradeoffs:

- Fastest method and the simplest estimand audit.
- Reuses one natural path bank for all source queries.
- Has no resampling-induced path duplication.
- Fails gracefully through visible low ESS and high maximum weight, but cannot manufacture support that was not sampled.
- Historically performed very well when paired with the atlas-blind wide-flow predictive winner, showing that a better generator can matter more than a more elaborate particle scheme.

### 2. Terminal SMC

Terminal SMC sequentially proposes the repair prefix from the flow and applies incremental population-anchor weights. In the explicit-lag implementation, the source potential is computed when the lagged source window closes, but source-driven resampling is **deferred until the prediction cut**. Its likelihood weight is carried through the intervening gap. A mandatory systematic resample at the cut creates an equally weighted repaired population, which is then rolled forward freely.

Algorithm:

1. Initialize a particle system at the observed boundary history.
2. Propose one repair frame at a time from the flow.
3. Apply incremental population-anchor potentials; before the source clamp, anchor-induced ESS resampling is permitted.
4. At the source-window end, apply the full low/high Gaussian source potential.
5. Carry its normalized importance weights through the lag gap without source-driven resampling.
6. Systematically resample at the cut.
7. Roll the repaired particles forward without weights and contrast low versus high.

Example: with \(\ell=8\), the full source likelihood is applied two seconds before the cut and retained through eight learned transitions. This avoids making an early discrete ancestry decision, but a highly selective source potential can leave almost all terminal weight on a few trajectories.

Tradeoffs:

- Converts a weighted endpoint into equally weighted future particles.
- Deferring the source resample avoids premature ancestry collapse.
- Can still suffer severe terminal ESS collapse when the requested source state is rare.
- More expensive than direct importance because every source and low/high query has its own particle system.

### 3. Progressive bridge SMC

Progressive bridge SMC targets the same terminal source potential but introduces it gradually across the source window. Each survivor proposes multiple children. A persistence look-ahead predicts the unfinished source-window average, and the inverse-temperature parameter \(\beta\) increases from zero to one across the four frames. If an increment would lower candidate ESS beneath the declared floor, it is shortened and the candidates are systematically resampled before tempering continues.

Algorithm:

1. Branch each of \(N\) repair survivors into two flow proposals.
2. Update the population anchor.
3. Inside the source window, compute the partial source average plus a persistence estimate of the unseen portion.
4. Increase \(\beta\) toward the frame-specific cap \(1/4,2/4,3/4,1\).
5. Choose the largest tempering increment retaining at least 65% candidate ESS; resample and continue if necessary.
6. Prune the two-way candidate population back to \(N\) after each repair frame.
7. Carry the completed source constraint through any lag gap.
8. Launch two free future descendants per repaired survivor and average them.

Example: rather than discovering at the fourth source frame that almost every path has the wrong average, the bridge rewards promising partial paths while alternative continuations still exist. Its terminal \(\beta=1\) potential is the same Gaussian clamp used by the other methods.

Tradeoffs:

- Best historical finite-particle recovery of the high-particle repaired law.
- Higher ESS and more distinct useful paths when the source constraint is selective.
- Reduces future Monte Carlo noise through two descendants per repaired root.
- Most computationally expensive; branching and future replication are bundled, so historical gains do not isolate their contributions.
- The persistence look-ahead may be imperfect for oscillatory or rapidly reversing source dynamics.

### 4. Temporal-cut SMC

Temporal-cut SMC places the source window \(\ell\) frames before the cut, applies the full source potential when that window closes, and permits ESS-triggered resampling immediately and at subsequent repair steps. It then rolls the cut-time repaired population forward freely.

Algorithm:

1. Generate the repair prefix sequentially from the observed boundary.
2. Omit the designated source from the anchor only during its lagged source window.
3. Apply the full source clamp at the source-window end.
4. If ESS is below 50% of \(N\), resample immediately.
5. Continue learned transitions and anchor updates to the cut, with ESS checks after each step.
6. Resample at the cut and freely roll forward.

Example: for a source window ending four seconds before the cut, an early resample commits ancestry to paths compatible with that historical source state, and the flow propagates those histories for 16 frames. This is closer to the intuitive “sample a changed past, then see what survives to the present” construction, but early resampling can discard path diversity.

Tradeoffs:

- Most directly implements an explicit historical source placement.
- Can repair degeneracy soon after the lagged constraint instead of carrying tiny weights.
- May introduce ancestry collapse earlier than terminal-deferred SMC.
- At zero source lag it reduces to the ordinary terminal-clamp bootstrap SMC design; it is not a fourth mathematical estimand.

## Method taxonomy and fair comparison

| Workflow | Natural bank | Sequential proposal | Source bridge | Source-driven resampling | Explicit source lag |
| --- | --- | --- | --- | --- | --- |
| Direct importance | Yes | No | No | Never | Yes |
| Terminal SMC | No | Yes | No | Deferred to cut | Yes |
| Progressive bridge SMC | No | Yes, branched | Across source window | Adaptive during bridge | Yes |
| Temporal-cut SMC | No | Yes | No | ESS-triggered from source-window end | Yes |

All workflows use the same frozen fold checkpoint, observed episode boundary, stimulus history, source-window width, phase-specific training-fold quartiles, anchor projection, clamp bandwidth, response definition, and keyed episode seed. They do not always traverse identical random paths because their batch shapes and resampling decisions differ.

## Historical performance before the corrected explicit-lag run

These results answer two different questions and should not be merged.

### Finite-particle accuracy for one frozen generator

The sealed Stage-A experiment compared three estimators to the mean of two independent 4,096-particle direct-importance references for the same learned repaired law.

| Estimator | Response MSE | Bias² | MC variance | Signed matrix rho | Valid | Runtime/four-worm fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct importance | 0.0306890 | 0.0130587 | 0.0176303 | 0.758 | 0.440 | 0.5 min |
| Terminal-clamp SMC | 0.0156454 | 0.0080981 | 0.0075473 | 0.815 | 0.435 | 21.2 min |
| Progressive bridge SMC | **0.0110076** | **0.0065799** | **0.0044277** | **0.854** | **0.911** | 34.3 min |

Progressive reduced response MSE by 29.6% versus terminal SMC and 64.1% versus direct importance. This was estimator recovery of a noisy high-particle model reference—not biological truth.

### Historical external correspondence on the shared 54-neuron subset

The previous full-ensemble comparison used old generator lineages and mostly varied **future response horizon**, not explicit source placement. Values are AUROC/AUPRC at the one-frame horizon.

| Method | Randi WT | Cook structural | Cook chemical | Cook gap |
| --- | ---: | ---: | ---: | ---: |
| Previous flow, direct importance | 0.594/0.255 | 0.544/0.317 | 0.538/0.291 | 0.557/0.082 |
| Wide-flow predictive winner, direct | **0.619/0.298** | **0.591/0.349** | **0.579/0.314** | **0.653/0.118** |
| Terminal SMC | 0.579/0.246 | 0.557/0.321 | 0.553/0.294 | 0.586/0.090 |
| Progressive SMC | 0.596/0.251 | 0.564/0.325 | 0.557/0.297 | 0.590/0.086 |
| SBTG-current | 0.524/0.198 | 0.524/0.291 | 0.517/0.268 | 0.585/0.080 |
| SBTG-published | 0.622/0.322 | 0.565/0.348 | 0.558/0.312 | 0.621/0.133 |

The wide-flow direct method was genuinely comparable to SBTG-published on these external rankings. However, that historical wide-flow model was trained on a 20-worm pooled cohort with older stimulus assumptions. It remains an important historical result but is not silently relabeled as a corrected-cohort result.

Cook is continuous/count-valued. Historical lag-1 all-pair structural-count Spearman correlations were 0.143 for wide-flow direct, 0.098 for progressive, and 0.102 for SBTG-published; correlations conditional on positive edges were only 0.066, 0.030, and 0.051. Edge-presence recovery was stronger than grading the number of anatomical contacts.

## Current corrected explicit-lag experiment

### Frozen design

- Model: corrected binary-any-stimulus flow, 5 held-out-worm folds, generator seeds 1701 and 2903; seed-specific responses are averaged within held-out worm before population aggregation.
- Cohort: 17 OH16230 worms, 54 neurons, native 4 Hz.
- Source window: 4 frames / 1 second.
- Source lags: 1, 4, 8, 16 frames / 0.25, 1, 2, 4 seconds.
- Forecast: one frame / 0.25 seconds after the cut for the primary matrix.
- Phases: baseline, onset, and active; three recorded stimulus events per worm.
- Particle budget: 32 survivors per low/high source query; progressive uses branch factor 2 and future factor 2.
- Primary channel: endpoint conditional-mean change per achieved source unit.
- Secondary channels: cumulative mean, peak, event probability, and endpoint standard-deviation change.
- Primary cross-state panel: equal average of baseline, onset, and active matrices.
- Sensitivities: phase-specific and onset-minus-baseline panels; support-qualified scopes.

### External references and their meanings

| Reference | Stored quantity | Primary metric | What it can support |
| --- | --- | --- | --- |
| Randi functional atlas | Significant positive versus tested negative/zero responses, plus signed response | AUROC/AUPRC; signed and absolute Spearman sensitivity | Functional correspondence under a separate perturbational assay |
| Cook chemical/gap/structural connectome | Nonnegative integer-like contact counts | Presence AUROC/AUPRC after `count > 0`; Spearman with counts | Anatomical edge or count correspondence |
| Bentley monoamine/neuropeptide maps | Binary ligand/receptor compatibility edges | AUROC/AUPRC on eligible source classes | Correspondence to receptor-expression compatibility, not activity or gain itself |

The aligned Bentley source counts are: monoamine-all 5, dopamine 2, serotonin 1, tyramine 1, octopamine 1, neuropeptide 30, and union 31. Transmitter-specific maxima therefore have very limited source-level replication.

### Inference and multiplicity

- Every external metric excludes the diagonal.
- Neuromodulator “eligible source” metrics include only source columns with at least one reference edge for that network.
- Lag-max p-values permute target labels independently within each eligible source, preserve each source's edge prevalence, and compare the observed maximum AUROC to the null maximum over the complete lag grid.
- Benjamini–Hochberg correction is applied across the reported lag-max family.
- Worm bootstraps resample the 17 held-out worms and repeat lag selection; they quantify animal-level matrix uncertainty without refitting the generator.
- Method-support-qualified sensitivities require at least 50% episode-level validity for a source. Unfiltered and qualified results are both retained so support filtering cannot silently improve a result.
- Specific-transmitter source counts are too small for a meaningful source-column bootstrap; the worm bootstrap and target-label permutation answer different, narrower questions.

AUROC asks whether larger absolute learned effects rank reference edges above nonedges. AUPRC exposes performance under severe class imbalance. Correlation is added for Cook because contact counts are not intrinsically binary. None of these metrics establishes that a learned effect is anatomically direct.

## Preprocessing and construction audit

The implementation fails closed on each of the following:

1. Every response archive must declare the same 54-neuron order as its checkpoint and all other archives.
2. Held-out worm IDs must be unique within a method-lag cell and must cover all 17 cohort worms exactly once across the five folds.
3. Fold assignments are joined by worm ID, not row position; extra OH15500 rows in the historical fold file are ignored.
4. Each archive stores the checkpoint SHA-256, exact cut frames, exact source-window bounds, and lag definition.
5. For every episode, `source_window_end - 1 == cut - lag` and the window width is exactly four frames.
6. Checkpoint stimulus metadata must match the cohort's stimulus-schema version and fingerprint before inference.
7. Source low/high/IQR values, anchor PCA, and event thresholds use training worms only and are recomputed for each fold and source lag.
8. The source is removed from the population anchor only within its declared lagged window.
9. The sampler's `[source, horizon, target]` tensor is transposed once to `[target, source]`; a directional spot test is saved in validation metadata.
10. Reference matrices are aligned by neuron name to the learned 54-neuron order; unmatched nodes are not shifted by position.
11. Cook/Randi/Bentley are loaded only after response archives are frozen.
12. The external metrics use off-diagonal entries, and reference-specific masks are retained exactly.

## Current results

The corrected run is complete: all 160 primary response archives and all 40 direct-importance N=256 sensitivity archives pass validation. The full frozen result is in `results/four_sampler_lag_connectome_20260828/analysis/REPORT.md`.

### Estimator support and stability

| Workflow | N | Total compute | Compatibility validity | Generator-seed signed rho range |
| --- | ---: | ---: | ---: | ---: |
| Direct importance | 32 | 4.3 min | 0.160 | 0.235–0.290 |
| Terminal SMC | 32 | 64.4 min | 0.161 | 0.236–0.277 |
| Progressive bridge SMC | 32 | 139.8 min | **0.802** | **0.478–0.688** |
| Temporal-cut SMC | 32 | 63.5 min | 0.161 | 0.290–0.540 |
| Direct sensitivity | 256 | 7.9 min | 0.420 configured / 0.252 at the common relative ESS gate | 0.366–0.423 |

Progressive bridge is the only 32-particle workflow for which most source/episode cells satisfy the declared clamp compatibility criteria. Raising the direct bank to 256 improves support, but N=32 versus N=256 signed matrix correlations remain only 0.569–0.607. Terminal and temporal-cut SMC resample low-ESS populations but do not make the pre-resampling source constraint well supported; their diagnostic validity remains near direct importance.

### Fixed lag-1 external comparison

| Method | Randi WT AUROC | Cook structural | Cook chemical | Cook gap |
| --- | ---: | ---: | ---: | ---: |
| Direct N=32 | 0.567 | 0.552 | 0.538 | 0.616 |
| Terminal SMC | 0.586 | 0.561 | 0.555 | 0.616 |
| **Progressive bridge SMC** | **0.637** | **0.611** | **0.603** | **0.650** |
| Temporal-cut SMC | 0.590 | 0.563 | 0.558 | 0.621 |
| Direct N=256 | 0.606 | 0.596 | 0.585 | 0.645 |
| SBTG-current | 0.524 | 0.524 | 0.517 | 0.585 |
| SBTG-published | 0.622 | 0.565 | 0.558 | 0.621 |

Progressive bridge therefore produces the best supported matrices and the strongest fixed-lag Randi/Cook ranking in this run. Cook count correlations are weaker than presence AUROCs: progressive's lag-1 structural Spearman is 0.174 across all pairs, 0.082 among positive Cook edges, and 0.130 after within-source averaging.

### Neuromodulator result

The four-lag endpoint-mean maxima for progressive bridge are 0.520 for pooled monoamine, 0.532 for neuropeptide, and 0.526 for their union. No method or specific-transmitter lag maximum survives the source-preserving maximum-over-lags null and global BH correction; the minimum q-value is 0.597. Descriptive peaks include progressive serotonin mean AUROC 0.659 at four seconds and direct-N=256 octopamine mean AUROC 0.725 at one second, but both panels have only one eligible source and neither is calibrated evidence.

The fair two-lag grid shared by all flow and SBTG matrices contains lag frames 1 and 8. SBTG-published reaches 0.581/0.544/0.537 for monoamine/neuropeptide/union, while the best new flow values are 0.513/0.534/0.528. Thus the new sampler can improve Randi/Cook correspondence without improving pooled receptor-connectome correspondence.

Distributional endpoint-SD and event-probability matrices were also saved. Their isolated transmitter peaks are descriptive because selection occurred across method, channel, lag, and transmitter; they were not given a separate promoted all-channel inference family.

### Bottom line

Progressive bridge SMC is the preferred finite-particle realization of this frozen repaired-flow law. The lag matrices are real, reproducible artifacts and can show SBTG-like external patterns, but their neuromodulator lag maxima remain uncalibrated and do not identify physical delays. This separates estimator progress from the still-negative biological correspondence claim.

## Interpretation boundaries and expected failure modes

The approach can fail for at least four distinct reasons:

1. **Generator error:** the conditional flow may fit predictive density while misrepresenting the cross-neuron counterfactual geometry needed by repaired sampling.
2. **Support failure:** observed data may contain too few paths compatible with a requested source state and population anchor.
3. **Estimator failure:** a finite particle method can approximate the learned repaired law poorly even when that law is well supported.
4. **Validation mismatch:** receptor existence, anatomical contact, and perturbational activity are different biological objects. A scientifically useful activity effect need not rank any one reference perfectly.

Near-chance Bentley correspondence therefore does not by itself show that the flow or sampler failed. Conversely, a high Bentley AUROC would be convergent validity, not proof of causal neuromodulator action or a physical delay.

## Reproducible implementation

- `compatibility_neural_benchmark/four_sampler_lag_runner.py`: corrected four-workflow response generation.
- `compatibility_neural_benchmark/four_sampler_lag_analysis.py`: frozen matrix construction, Randi/Cook/Bentley alignment, lag-max inference, uncertainty, and figures.
- `compatibility_neural_benchmark/core.py`: direct importance and bootstrap-SMC implementation.
- `compatibility_neural_benchmark/progressive_smc.py`: progressive lag-aware bridge.
- `compatibility_neural_benchmark/tests/test_repaired_path.py`: source-window, lag, resampling-policy, and bridge-placement tests.
- `results/four_sampler_lag_connectome_20260828/`: corrected response archives and run provenance.
- `results/four_sampler_lag_connectome_20260828/analysis/`: aligned matrices, metrics, inference, figures, checksums, and validation.

## Further questions

- Does the estimator ranking change at larger particle counts under the corrected generator?
- Are distributional channels such as endpoint SD more reproducible across worms than conditional-mean effects?
- Do stimulus-specific rather than binary-flow checkpoints improve onset matrices without weakening held-out density?
- Can a synthetic system with known lagged gain modulation separate mean, variance, and tail-effect recovery before more biological interpretation?
- Which lag features transfer from the 17-worm dataset to the larger DANDI 000981 cluster experiment?
