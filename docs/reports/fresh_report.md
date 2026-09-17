> Portable reading copy of [REPORT_FINAL.md](../../archive/fresh/replication_20260911/REPORT_FINAL.md). Scientific text is preserved; local links are relocated. The frozen original remains authoritative.

**Flow / progressive bridge SMC replication — final stopped-study interpretation**

Prepared 2026-09-16T17:03:20.268166+00:00. This report supersedes the scientific interpretation in REPORT_STOPPED.md; that original report, its figures and every frozen worker remain preserved. Supplementary descriptive analyses are documented in [amendment 003](../../archive/fresh/replication_20260911/amendments/003_final_interpretation_20260916/AMENDMENT.md). No model fitting or path sampling was performed after the user-directed stop.

**Verdict.** The fresh computational results support improved atlas ranking in a limited sense: Cook structural and chemical correspondence improves in both cohorts and survives the tested strong-support and endpoint-timing sensitivities. The broader claim that the complete derivative-based lag matrix is reliably better is not established. Smaller contrasts become substantially noisier; the clean cohort has no multiplicity-corrected lag-versus-lag-1 effects; the strict complete-case clean matrix has no surviving entries; all clean primary models reached their training cap. Historical results cannot establish independent-animal replication because their prepared inputs retain head/tail pseudo-pairing and donor copying.

**What was rerun and what was reused.** There are 30 new primary flow fits (two cohorts × five recording folds × three generator seeds), 120 new primary progressive N64 archives, 40 independent-Monte-Carlo repeats, 40 direct N4096 comparisons, 60 endpoint-timing comparisons, 1,440 diagnostic archives and 30 factual predictive evaluations. The source implementation and recordings were copied and hash-pinned. Published SBTG manuscript matrices were reused as the frozen comparator; SBTG was not refitted. The released lag-1 comparator is the hybrid manuscript matrix; other published lags follow that release's production lineage. The longer-training extension stopped at 9/15 fits and 34/60 sampling archives and contributes nothing to the primary ensemble. This is computational reproduction on existing data, not prospective biological replication or a fully independent rewrite of the flow/SMC method.

**Primary atlas correspondence.** The fixed primary comparison is nominal source lag 1, forecast horizon 1, state-average endpoint mean, averaged over three fitted seeds. Matrices are target-by-source; scoring uses absolute magnitudes, common eligible off-diagonal pairs and the same masks for flow and SBTG. AP denotes average precision, the implementation's AUPRC summary. Cook positives have adjacency greater than zero. Randi positives have q < .05; equivalently negative pairs have q_eq < .05 unless already positive; ambiguous pairs are excluded. These scores do not test effect signs or prove causal edges.

| cohort       | reference        |   edges |   flow AUROC |   SBTG AUROC |   flow AP |   SBTG AP |   AUROC difference | 95% interval      |   simultaneous lower |
|:-------------|:-----------------|--------:|-------------:|-------------:|----------:|----------:|-------------------:|:------------------|---------------------:|
| historical80 | Randi functional |    1646 |       0.6800 |       0.6302 |    0.3301 |    0.3003 |             0.0497 | [0.0072, 0.0919]  |               0.0017 |
| historical80 | Cook structural  |    6320 |       0.6329 |       0.5807 |    0.3252 |    0.2888 |             0.0522 | [0.0260, 0.0781]  |               0.0223 |
| historical80 | Cook chemical    |    6320 |       0.6222 |       0.5698 |    0.2845 |    0.2519 |             0.0524 | [0.0232, 0.0815]  |               0.0186 |
| historical80 | Cook gap         |    6320 |       0.6689 |       0.6133 |    0.1246 |    0.1069 |             0.0556 | [0.0207, 0.0910]  |               0.0152 |
| clean54      | Randi functional |    1349 |       0.6715 |       0.6217 |    0.3546 |    0.3219 |             0.0498 | [-0.0009, 0.1009] |              -0.0090 |
| clean54      | Cook structural  |    2862 |       0.6142 |       0.5651 |    0.3845 |    0.3482 |             0.0491 | [0.0230, 0.0747]  |               0.0190 |
| clean54      | Cook chemical    |    2862 |       0.6015 |       0.5576 |    0.3497 |    0.3124 |             0.0439 | [0.0180, 0.0698]  |               0.0144 |
| clean54      | Cook gap         |    2862 |       0.6780 |       0.6207 |    0.1322 |    0.1334 |             0.0573 | [0.0024, 0.1159]  |              -0.0045 |

Intervals are 10,000 paired source-column bootstrap intervals with matrices held fixed. The simultaneous lower bound uses the .05/4 lower quantile for the four references within a panel. The historical all-common comparison is the prespecified primary family; the clean cohort and other masks are sensitivities. These are conditional bootstrap summaries, not full refit/animal uncertainty or family protection across every secondary analysis. All 32 released reference-by-lag metric reproductions match the saved release to numerical precision; the final audit independently recomputed both cohorts' primary AUROC/AP/counts from their saved matrices.

Historical point differences are positive for all four panels, with positive simultaneous lower bounds. Clean structural and chemical differences have positive simultaneous lower bounds; clean Randi and gap differences remain unresolved under that rule. Clean gap AP is slightly lower for flow despite higher AUROC, so superiority is not uniform across metrics.

**Support, timing and origin sensitivities.** Strong fixed support requires validity and genealogy qualification on at least 80% of episodes at every lag, retaining 57/80 historical sources and 33/54 clean sources. Structural and chemical gains persist under this mask in both cohorts. Randi and gap gains are less robust. The timing-1 comparison uses flow source lag 0 plus horizon 1 versus SBTG lag 1; timing-8 uses flow lag 7 plus horizon 1 versus SBTG lag 8. Even matching endpoints does not equate a four-frame soft repaired-history contrast with the SBTG estimand.

| cohort       | scope                           | reference       |   delta_auroc |   simultaneous_one_sided_lower |
|:-------------|:--------------------------------|:----------------|--------------:|-------------------------------:|
| historical80 | strong_fixed_support            | randi_wild_type |        0.0186 |                        -0.0456 |
| historical80 | strong_fixed_support            | cook_struct     |        0.0506 |                         0.0151 |
| historical80 | strong_fixed_support            | cook_chem       |        0.0522 |                         0.0113 |
| historical80 | strong_fixed_support            | cook_gap        |        0.0443 |                        -0.0009 |
| historical80 | endpoint_timing_matched_1frames | randi_wild_type |        0.0470 |                        -0.0061 |
| historical80 | endpoint_timing_matched_1frames | cook_struct     |        0.0504 |                         0.0218 |
| historical80 | endpoint_timing_matched_1frames | cook_chem       |        0.0503 |                         0.0193 |
| historical80 | endpoint_timing_matched_1frames | cook_gap        |        0.0500 |                         0.0114 |
| historical80 | endpoint_timing_matched_8frames | randi_wild_type |        0.1260 |                         0.0629 |
| historical80 | endpoint_timing_matched_8frames | cook_struct     |        0.1091 |                         0.0742 |
| historical80 | endpoint_timing_matched_8frames | cook_chem       |        0.1006 |                         0.0644 |
| historical80 | endpoint_timing_matched_8frames | cook_gap        |        0.1533 |                         0.0969 |
| clean54      | strong_fixed_support            | randi_wild_type |        0.0395 |                        -0.0445 |
| clean54      | strong_fixed_support            | cook_struct     |        0.0598 |                         0.0280 |
| clean54      | strong_fixed_support            | cook_chem       |        0.0573 |                         0.0204 |
| clean54      | strong_fixed_support            | cook_gap        |        0.0227 |                        -0.0461 |
| clean54      | endpoint_timing_matched_1frames | randi_wild_type |        0.0427 |                        -0.0127 |
| clean54      | endpoint_timing_matched_1frames | cook_struct     |        0.0456 |                         0.0146 |
| clean54      | endpoint_timing_matched_1frames | cook_chem       |        0.0428 |                         0.0132 |
| clean54      | endpoint_timing_matched_1frames | cook_gap        |        0.0622 |                        -0.0017 |
| clean54      | endpoint_timing_matched_8frames | randi_wild_type |        0.1048 |                         0.0369 |
| clean54      | endpoint_timing_matched_8frames | cook_struct     |        0.0852 |                         0.0491 |
| clean54      | endpoint_timing_matched_8frames | cook_chem       |        0.0773 |                         0.0395 |
| clean54      | endpoint_timing_matched_8frames | cook_gap        |        0.1456 |                         0.0843 |

The secondary timing-8 panels show positive lower bounds for all references in both cohorts, but do not replace the fixed primary comparison or establish lag identification. Historical origin-stratified results are also heterogeneous: the head/head comparisons carry the gains, whereas tail/tail Cook structural AUROC differs by −0.0822 (ordinary 95% interval −0.1576 to −0.0050); cross-origin intervals span zero. These inherited origin labels and dependencies prevent a clean biological interpretation of the historical strata.

**Strict gap sensitivity, newly scored in this supplement.** The primary normalization divides by max(abs(achieved source gap), .10), retaining invalid episodes. The strict arrays instead require validity and a positive achieved gap of at least .10. Ordinary event, phase, recording and seed means propagate any invalid value. This deliberately demanding complete-case rule retains 12 historical source columns / 948 off-diagonal entries and zero clean entries. Zero clean coverage is not proof that every individual clean episode fails; it means no source passes every contributing episode required by this aggregation.

| cohort       | reference       | method                   | status                 |   n_edges |   n_positive |    auroc |    auprc |
|:-------------|:----------------|:-------------------------|:-----------------------|----------:|-------------:|---------:|---------:|
| historical80 | randi_wild_type | strict                   | evaluated              |       196 |           10 |   0.5484 |   0.2491 |
| historical80 | randi_wild_type | published_on_strict_mask | evaluated              |       196 |           10 |   0.5973 |   0.1397 |
| historical80 | cook_struct     | strict                   | evaluated              |       948 |          142 |   0.6485 |   0.3053 |
| historical80 | cook_struct     | published_on_strict_mask | evaluated              |       948 |          142 |   0.5877 |   0.2427 |
| historical80 | cook_chem       | strict                   | evaluated              |       948 |          113 |   0.6215 |   0.2447 |
| historical80 | cook_chem       | published_on_strict_mask | evaluated              |       948 |          113 |   0.5581 |   0.1977 |
| historical80 | cook_gap        | strict                   | evaluated              |       948 |           56 |   0.7060 |   0.2296 |
| historical80 | cook_gap        | published_on_strict_mask | evaluated              |       948 |           56 |   0.6409 |   0.1469 |
| clean54      | randi_wild_type | strict                   | no_complete_case_edges |         0 |            0 | nan      | nan      |
| clean54      | randi_wild_type | published_on_strict_mask | no_complete_case_edges |         0 |            0 | nan      | nan      |
| clean54      | cook_struct     | strict                   | no_complete_case_edges |         0 |            0 | nan      | nan      |
| clean54      | cook_struct     | published_on_strict_mask | no_complete_case_edges |         0 |            0 | nan      | nan      |
| clean54      | cook_chem       | strict                   | no_complete_case_edges |         0 |            0 | nan      | nan      |
| clean54      | cook_chem       | published_on_strict_mask | no_complete_case_edges |         0 |            0 | nan      | nan      |
| clean54      | cook_gap        | strict                   | no_complete_case_edges |         0 |            0 | nan      | nan      |
| clean54      | cook_gap        | published_on_strict_mask | no_complete_case_edges |         0 |            0 | nan      | nan      |

On the historical strict mask the primary and strict scores coincide, because every surviving denominator already exceeds the floor. Historical Randi has only 196 labeled pairs and ten positives; its strict AUROC is lower than SBTG. Cook point estimates remain higher on that restricted subset. These post-hoc point scores have no additional inferential claim. The clean strict sensitivity cannot be scored, so clean primary results cannot be presented as robust to this strict exclusion rule. No available-episode averaging was substituted to manufacture coverage.

**Reproducibility of the averaged matrices.** At lag 1 / horizon 1:

| cohort       | comparison       |   spearman |   sign_agreement |
|:-------------|:-----------------|-----------:|-----------------:|
| historical80 | generator_0_vs_1 |     0.6468 |           0.7384 |
| historical80 | generator_0_vs_2 |     0.6487 |           0.7380 |
| historical80 | generator_1_vs_2 |     0.6729 |           0.7508 |
| historical80 | independent_MC   |     0.9015 |           0.8660 |
| historical80 | direct_N4096     |     0.8754 |           0.8388 |
| clean54      | generator_0_vs_1 |     0.5615 |           0.7233 |
| clean54      | generator_0_vs_2 |     0.5514 |           0.7198 |
| clean54      | generator_1_vs_2 |     0.5579 |           0.7219 |
| clean54      | independent_MC   |     0.9037 |           0.8850 |
| clean54      | direct_N4096     |     0.8690 |           0.8613 |

Independent sampling repeats correlate around .90, while separate training seeds correlate only .55–.67. Thus training variability is material even when the averaged matrix is relatively stable to Monte Carlo repetition. Agreement with direct N4096 is around .87; correlation alone does not establish small per-cell error. The full saved table retains all seed pairs, lags, horizons and support scopes.

**Particle accuracy and computation.** A direct N16384 reference is considered resolved only where all three independent reference runs pass validity, minimum ESS is at least 12 and normalized-response SD is at most .05. Only 48.5% of historical and 41.8% of clean diagnostic cells meet that rule. The table below pools absolute error using qualified cell counts. Original figures average the eight lag/phase strata instead; both versions are retained and lead to the same qualitative interpretation. Direct N16384's own error uses a leave-one-run-out reference rather than comparing each run to a mean containing itself.

| cohort       | method      |   particles |   resolved_reference_fraction |   pooled_mae |   mean_wall_seconds |   mean_velocity_sample_evaluations |
|:-------------|:------------|------------:|------------------------------:|-------------:|--------------------:|-----------------------------------:|
| historical80 | direct      |        4096 |                        0.4849 |       0.0539 |              4.6658 |                       7086080.0000 |
| historical80 | direct      |       16384 |                        0.4849 |       0.0255 |             22.3992 |                      28344320.0000 |
| historical80 | progressive |          64 |                        0.4849 |       0.0843 |              2.6073 |                       4464640.0000 |
| historical80 | progressive |         128 |                        0.4849 |       0.0614 |              5.1231 |                       8929280.0000 |
| historical80 | progressive |         512 |                        0.4849 |       0.0342 |             19.9438 |                      35717120.0000 |
| clean54      | direct      |        4096 |                        0.4180 |       0.0636 |              8.5573 |                       8503296.0000 |
| clean54      | direct      |       16384 |                        0.4180 |       0.0263 |             35.1317 |                      34013184.0000 |
| clean54      | progressive |          64 |                        0.4180 |       0.0919 |              5.7133 |                       5357568.0000 |
| clean54      | progressive |         128 |                        0.4180 |       0.0670 |             11.2399 |                      10715136.0000 |
| clean54      | progressive |         512 |                        0.4180 |       0.0378 |             44.8148 |                      42860544.0000 |

Progressive error falls as N rises from 64 to 512 in both cohorts. This supports particle convergence over the tested range, but N64 is not already numerically settled: its qualified-reference error is larger than direct N4096. N64 is faster in these diagnostic runs, so this is a speed/accuracy tradeoff. N128 uses more velocity evaluations than direct N4096 yet has larger qualified-reference error; no broad efficiency advantage is established. A direct bank is shared across sources, whereas these progressive diagnostics sample eight selected sources, so these timings must not be extrapolated into a universal full-matrix speed claim. No exact equal-budget performance frontier was run. Unresolved direct-reference cells cannot settle which sampler is accurate. High terminal ESS after tempering/resampling is not a substitute for ancestry or accuracy checks.

**Derivative approximation: numerical adjudication.** The diagnostic uses 512 progressive particles, three MC seeds, folds 0/1, baseline/onset for the first event, lags 1/4, eight sources and all targets/horizons. Source targets move symmetrically around the quartile midpoint, while clamp bandwidth remains fixed. The following table uses horizon 1 and requires all six runs in each paired comparison to pass validity and achieved gap ≥ .10. Coverage is relative to that diagnostic panel, not the entire production matrix. `first`/`second` are fractions of the full requested source contrast. MC SD is across the three runs, not biological uncertainty.

| cohort       |   first |   second |   coverage |   mean_abs_difference |   median_mc_sd_first |   median_mc_sd_second |   spearman |   sign_agreement |
|:-------------|--------:|---------:|-----------:|----------------------:|---------------------:|----------------------:|-----------:|-----------------:|
| historical80 |  1.0000 |   0.5000 |     0.8594 |                0.0448 |               0.0408 |                0.0691 |     0.6914 |           0.7703 |
| historical80 |  0.5000 |   0.2500 |     0.7500 |                0.0777 |               0.0662 |                0.1216 |     0.4363 |           0.6711 |
| historical80 |  1.0000 |   0.2500 |     0.7500 |                0.0722 |               0.0387 |                0.1216 |     0.4993 |           0.6817 |
| clean54      |  1.0000 |   0.5000 |     1.0000 |                0.0565 |               0.0528 |                0.0816 |     0.7062 |           0.7736 |
| clean54      |  0.5000 |   0.2500 |     0.7500 |                0.0886 |               0.0810 |                0.1414 |     0.4401 |           0.6706 |
| clean54      |  1.0000 |   0.2500 |     0.7500 |                0.0830 |               0.0530 |                0.1414 |     0.5011 |           0.6906 |

Full-to-quarter rank agreement is only .499 historical / .501 clean, with sign agreement .682 / .691. On those same eligible cells, median MC SD grows from .0387 to .1216 historical and .0530 to .1414 clean. Mean absolute changes (.0722 / .0830) exceed mean absolute full-contrast effects on those cells (.0552 / .0663). Therefore the available calculations do not resolve a stable per-cell derivative as contrast shrinks.

This is chiefly a failure to demonstrate precision/convergence, not proof that the true softened response is nonlinear: the median paired change is only about .81–.83 estimated standard errors in the full-to-quarter comparison. Three repetitions provide a weak noise estimate. Requested-gap normalization gives the same rank-instability pattern, so it is not solely an achieved-gap denominator artifact. Across all six horizons, full-to-quarter rank agreement is .573 / .586, still insufficient to claim an invariant derivative matrix. Broader positive-gap and conservative .10-gap results, both normalizations and paired cell-level evidence are preserved in the supplement. The primary estimator remains a regularized finite contrast; fixed bandwidth and factual anchoring also distinguish it from a causal intervention derivative.

**Flow integration: numerical adjudication.** Original versus doubled Heun steps were compared with matched MC seeds, N128, folds 0/1, baseline first event, lags 1/16 and eight sources. This assesses the combined solver-plus-SMC estimator; small integration changes can alter discrete resampling decisions and later particle paths.

| cohort       |   coverage |   mean_abs_difference |   mean_abs_first |   median_paired_se |   median_change_over_se |   spearman |
|:-------------|-----------:|----------------------:|-----------------:|-------------------:|------------------------:|-----------:|
| historical80 |     0.8750 |                0.0377 |           0.0728 |             0.0332 |                  0.8459 |     0.8318 |
| clean54      |     1.0000 |                0.0358 |           0.0867 |             0.0353 |                  0.7916 |     0.8770 |

The horizon-1 estimates correlate .832 historical / .877 clean, with mean absolute differences .0377 / .0358. Typical changes are within the estimated paired MC uncertainty, so the saved data do not isolate a dominant deterministic ODE error. Conversely, this is not an equivalence demonstration: differences remain substantial relative to mean absolute effects, and only three MC runs were available. Larger horizons and requested-gap results are in the tables. The reported fraction exceeding two estimated standard errors is descriptive only; with three runs, two SE is not a 95% threshold, and no new cellwise significance claims are made.

All identical-clamp controls have exactly zero raw MSE in both cohorts because low/high arms use coupled randomness. This is a symmetry check, not an independent-arm noise floor. It corrects the contradictory sentence in the preserved original report that described a nonzero zero-query MSE.

**Lag evidence after joint multiplicity correction.** The exact maximum-T calculation jointly includes endpoint mean and log-SD, baseline and onset-minus-baseline, effects and lag-minus-1 contrasts, all selected off-diagonal pairs, lags and horizons. The family contains 756,504 historical and 293,832 clean cells. Enumerating all two-sided recording sign orbits gives 2^19 and 2^16 null realizations respectively. The table distinguishes significant cells (pair × lag × horizon) from unique directed pairs:

| cohort       | channel       | context   | test        |   tested_cells |   significant_cells |   distinct_directed_pairs | minimum_adjusted_p   |
|:-------------|:--------------|:----------|:------------|---------------:|--------------------:|--------------------------:|:---------------------|
| historical80 | endpoint_mean | baseline  | effect      |         108072 |                7554 |                       746 | p=1.90735e-06        |
| historical80 | endpoint_mean | baseline  | lag_minus_1 |          81054 |                  92 |                        29 | p=7.62939e-06        |
| clean54      | endpoint_mean | baseline  | effect      |          41976 |                2242 |                       272 | p=1.52588e-05        |
| clean54      | endpoint_mean | baseline  | lag_minus_1 |          31482 |                   0 |                         0 | p=0.147598           |

The clean data support baseline mean effects (2,242 cells across 272 pairs) but no lag-minus-1 changes after correction; the smallest corrected clean baseline-mean lag p is .1476. Historical data have 92 lag-change cells across 29 pairs, but the pseudo-pairing and donor-copying lineage makes their recording-row inference descriptive rather than independent-animal evidence. Neither cohort has corrected onset-minus-baseline effects/lag changes or log-SD effects/lag changes. Absence of significance is not proof of identical lags. These conditional sign-flip results assume joint symmetry and do not account for shared fitted models, refitting or preprocessing selection uncertainty.

The separate lag-maximum atlas permutation analysis detects reference correspondence across the lag scan in several panels, including state-average mean for all four references in both cohorts. That is not a test that one lag differs from another. Changing lag also changes the repair boundary and anchor interval. No physical transmission-delay matrix is established.

**Factual predictive quality.** Each method uses the same deterministic held-out history bank, 32 free draws and horizons 1/2/4/8/16/32. Energy is lower-is-better. Selected horizons are shown; the complete summaries remain linked below.

| cohort       | method               |   horizon |   energy |   coverage90 |
|:-------------|:---------------------|----------:|---------:|-------------:|
| historical80 | flow                 |         1 |   1.4923 |       0.8756 |
| historical80 | flow                 |         4 |   2.6678 |       0.8517 |
| historical80 | flow                 |        32 |   5.9739 |       0.7846 |
| historical80 | persistence_gaussian |         1 |   1.5579 |       0.8698 |
| historical80 | persistence_gaussian |         4 |   2.7640 |       0.8898 |
| historical80 | persistence_gaussian |        32 |   6.2640 |       0.8882 |
| historical80 | ridge_gaussian       |         1 |   1.5021 |       0.8336 |
| historical80 | ridge_gaussian       |         4 |   2.5948 |       0.7885 |
| historical80 | ridge_gaussian       |        32 |   5.0445 |       0.7470 |
| clean54      | flow                 |         1 |   1.0508 |       0.8777 |
| clean54      | flow                 |         4 |   1.7097 |       0.8673 |
| clean54      | flow                 |        32 |   5.0319 |       0.7541 |
| clean54      | persistence_gaussian |         1 |   1.0980 |       0.8675 |
| clean54      | persistence_gaussian |         4 |   1.6937 |       0.9071 |
| clean54      | persistence_gaussian |        32 |   4.5022 |       0.8797 |
| clean54      | ridge_gaussian       |         1 |   1.2369 |       0.7546 |
| clean54      | ridge_gaussian       |         4 |   2.1957 |       0.6375 |
| clean54      | ridge_gaussian       |        32 |   5.8608 |       0.4949 |

Historical flow beats persistence in energy at all six horizons, but ridge is better after horizon 1. Clean flow beats ridge throughout, but persistence is better from horizon 4 onward. Flow 90% interval coverage falls to .785 historical / .754 clean at horizon 32. The flow is therefore not uniformly the best or well-calibrated long-horizon predictor. These aggregate predictive comparisons are descriptive; no new significance test is implied.

**Implementation and provenance qualifications.** The code review establishes the following boundaries. MAT `norm_traces` are already normalized calcium signals, not raw camera fluorescence; classes average left/right and designated dorsal/ventral traces. Clean class selection uses pooled availability across both head strains before retaining OH16230, and is not fold-nested. Pooled affine normalization cancels under later training-fold scaling to a maximum checked error of 3.815e-6, but availability selection and historical imputation do not disappear. Training histories forward-fill internal missing values and exclude missing targets; the sampler also backfills leading missing values, but no registered episode overlaps those leading frames. The legacy L80 TCN has a 31-frame local convolutional receptive field with older-history influence through temporal GroupNorm. Model optimization/scaling is training-only; sampler nuisance quantiles/projection/thresholds use all non-test recordings including validation. Source constraints and the rank-12 anchor are soft; the source is exempt from anchoring only during its four-frame window. There is no observed neural target beyond the forecast cut in the sampled future.

Every clean primary fit reached the 50-epoch cap and triggered the planned extension. The incomplete extension cannot settle whether further training would change these findings. No result here validates physical causal effects, receptor mechanisms or prospective biological generalization.

**Reviewed figures and evidence.** All fourteen original plots were inspected. Separate copies correct overlapping subplot labels and add scope annotations; source values and original plots remain preserved. Particle plots use the original equal-stratum error averages; the table above additionally reports pooled error. Matrix plots show diagonal responses for context, but all atlas/lag scores exclude them. Heatmap indices follow the neuron order stored in each primary archive.

| figure                 | historical80                                                                                                                                   | clean54                                                                                                                              |
|:-----------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|
| reference_comparison   | [historical80](../../archive/fresh/replication_20260911/final_review_20260916/historical80/figures/reference_comparison.png)   | [clean54](../../archive/fresh/replication_20260911/final_review_20260916/clean54/figures/reference_comparison.png)   |
| reference_lag_profiles | [historical80](../../archive/fresh/replication_20260911/final_review_20260916/historical80/figures/reference_lag_profiles.png) | [clean54](../../archive/fresh/replication_20260911/final_review_20260916/clean54/figures/reference_lag_profiles.png) |
| source_support         | [historical80](../../archive/fresh/replication_20260911/final_review_20260916/historical80/figures/source_support.png)         | [clean54](../../archive/fresh/replication_20260911/final_review_20260916/clean54/figures/source_support.png)         |
| lag_effect_matrices    | [historical80](../../archive/fresh/replication_20260911/final_review_20260916/historical80/figures/lag_effect_matrices.png)    | [clean54](../../archive/fresh/replication_20260911/final_review_20260916/clean54/figures/lag_effect_matrices.png)    |
| replication_stability  | [historical80](../../archive/fresh/replication_20260911/final_review_20260916/historical80/figures/replication_stability.png)  | [clean54](../../archive/fresh/replication_20260911/final_review_20260916/clean54/figures/replication_stability.png)  |
| predictive_rollout     | [historical80](../../archive/fresh/replication_20260911/final_review_20260916/historical80/figures/predictive_rollout.png)     | [clean54](../../archive/fresh/replication_20260911/final_review_20260916/clean54/figures/predictive_rollout.png)     |
| particle_convergence   | [historical80](../../archive/fresh/replication_20260911/final_review_20260916/historical80/figures/particle_convergence.png)   | [clean54](../../archive/fresh/replication_20260911/final_review_20260916/clean54/figures/particle_convergence.png)   |

[Code-path review](../../archive/fresh/replication_20260911/code_review_20260916/REVIEW.md) · [Original protocol](../../archive/fresh/replication_20260911/PROTOCOL.md) · [Analysis specification](../../archive/fresh/replication_20260911/ANALYSIS_IMPLEMENTATION.md) · [Derivative/solver summaries](../../archive/fresh/replication_20260911/final_review_20260916/numerical_adjudication.csv) · [Strict atlas scores](../../archive/fresh/replication_20260911/final_review_20260916/strict_reference_sensitivity.csv) · [Complete lag counts](../../archive/fresh/replication_20260911/final_review_20260916/lag_counts.csv) · [Recomputed primary metrics](../../archive/fresh/replication_20260911/final_review_20260916/primary_metric_audit.csv)

The detailed original analysis tables are in [historical80](../../archive/fresh/replication_20260911/analysis_stopped_20260916/historical80) and [clean54](../../archive/fresh/replication_20260911/analysis_stopped_20260916/clean54); these include all masks, lag scans, prediction horizons and generator comparisons. Supplementary paired cells and reviewed figures are in [final_review_20260916](../../archive/fresh/replication_20260911/final_review_20260916). The preserved original report is [REPORT_STOPPED.md](../../archive/fresh/replication_20260911/REPORT_STOPPED.md); use this final interpretation for scientific conclusions. Snapshot, worker, checkpoint and artifact hashes preserve provenance. The original copied suite recorded 309 passing tests/two skips, the original replication suite passed eleven tests including its direct-sampling oracle, and the final deterministic rerun passed ten tests with the sampling oracle deliberately excluded. Analytic-oracle success validates a controlled calculation, not the fitted biological model.

**Supported claim for this stopped study:** newly trained flow-based finite contrasts improve structural/chemical atlas ranking under the reported comparisons, while derivative convergence, clean lag specificity, universal sampler efficiency and complete training-convergence robustness remain unestablished.
