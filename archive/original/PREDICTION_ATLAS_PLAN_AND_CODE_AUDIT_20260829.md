# Neural perturbation prediction atlas: completed plan and code audit

**Protocol frozen:** 2026-08-29  
**Execution and documentation audit:** 2026-08-30  
**Primary cohort:** 17 OH16230 head recordings, 54 complete-case head neurons,
native 4 Hz  
**Generator:** `stim_binary_any_stimulus_tcn_flow128_dropout15_jitter_p01`,
80-frame neural/stimulus history  
**Primary estimator:** progressive bridge SMC; direct importance is the
independent estimator sensitivity  
**Current state:** local production is complete through both earlier N=128
panels, complete-family inference, the 70-archive targeted N=128 sampler-null
calibration, external/historical audits, and the schema-v3 offline explorer.
The bounded dashboard and portable report remain validated prior E27 surfaces;
the full explorer carries the later evidence layers and passed desktop/mobile
browser QA.

## Decision and outcome

The local atlas is now a frozen, auditable hypothesis-generation product. The
complete-family analysis finds many internally worm-consistent baseline-mean
dependencies, but no strong-support lag-resolved edge. The targeted
sampler-null calibration finds no row that clears both algorithmic controls and
the matched quiet-time gate. No current row is therefore experiment-ready.
The defensible near-term use is prospective design and independent replication,
while treating exact lag localization, neuromodulator matching, and all
chemical-specific views as exploratory.

The atlas was ranked internally before Cook, Randi, Bentley, receptor maps,
SBTG-current, or SBTG-published were opened. Those references did not reorder
the 1,000-row queue, choose either N=128 panel, or alter an evidence label. This
firewall is verified by input hashes and audit ledgers.

The data support a binary **any-stimulus** generator. The raw schedules contain
butanone, pentanedione, and NaCl in animal-specific order, but chemical identity
is used only for post-sampling event stratification. A chemical panel is not a
chemical-conditioned counterfactual.

## Scientific object

For held-out history ending at cut `c`, source neuron `j`, four-frame source
window, source-window-end-to-cut lag `ell`, fixed binary stimulus history `S`,
and future horizon `h`:

1. define fold-, phase-, and lag-local low and high source targets (q25/q75);
2. repair or reweight histories toward each target while excluding the source
   from the population anchor during its declared window;
3. draw matched low/high futures with common random numbers;
4. summarize each target neuron's future marginal distribution; and
5. report high-minus-low divided by
   `max(abs(achieved_source_gap), 0.10)` for normalized coefficient-scale
   effects.

The progressive bridge introduces the terminal source constraint sequentially
with branching, adaptive tempering, and resampling. Direct importance reuses a
larger natural-path bank and applies exact self-normalized terminal weights.
They target the same fitted repaired path law but have different finite-particle
failure modes. Agreement is an estimator sensitivity, not biological
replication.

Three time quantities remain separate:

- source-to-cut lag: `ell/4` seconds;
- forecast horizon: `h/4` seconds;
- source-window-end-to-readout separation: `(ell+h)/4` seconds.

The one-second source average is not a point impulse, and none of these time
labels is automatically a synaptic, receptor, or physical transmission delay.

## Frozen grid and completed execution

The screen spans all 54 sources × 54 targets, lags 1/4/8/16 frames, horizons
1/2/4/8/16/32 frames, five event phases, two generator seeds, and five
held-out-worm folds. Final matrices are target rows by source columns.

| Layer | Final verified state |
| --- | --- |
| Direct importance N=256 | 40/40 raw archives; validation `pass`; zero failures |
| Progressive bridge N=32, seed 1701 | 20/20 raw archives; validation `pass`; zero failures |
| Progressive bridge N=32, seed 2903 | 20/20 raw archives; validation `pass`; zero failures |
| Canonical analysis | 80 raw archives reconciled; 17 worms; validation and checksum ledger pass |
| Primary N=128 | Six internally selected cells; 30/30 raw archives; raw and analysis validation/ledgers pass |
| Supplemental distributional N=128 | Six independently selected internal cells; 30/30 raw archives; raw and analysis validation/ledgers pass |
| Historical E26 | 160 exact-overlap descriptive comparisons; checksum ledger pass |
| Post-freeze references | Randi/Cook/Bentley/SBTG analysis, 999 source-preserving permutations, seed 20260829; ledgers pass |
| Complete-family inference | Four exact 65,536-pattern worm families plus lower-support sensitivity; 102 joint baseline-mean edges, 492 cells, and zero strong-support lag-structure edges |
| Targeted sampler-null N=128 | 70/70 raw archives; 24 metric rows; one exact 64-test shared-worm max-T family; raw/analysis validations and ledgers pass |
| GUI | Schema-v3 explorer validates five inputs; ledger and production browser QA pass |
| Prior E27 dashboard/report | Original builder and browser receipts remain valid for their bounded E27 scope; neither silently displays the later complete-family or sampler-null layers |

## Canonical atlas

The canonical bundle contains:

- 436,800 prediction rows × 42 columns;
- 5,616 support rows × 60 columns;
- 4,000 candidate lag profiles;
- 6,120 stimulus-composition rows;
- a frozen 1,000-row internal hypothesis queue;
- 1,155 arrays in `atlas_matrices.npz`; and
- 103 arrays in `worm_matrices.npz`.

It covers two estimators × seven channels × 13 contexts × four lags × six
horizons, retaining 100 off-diagonal edges per slice for inference/ranking.
Diagonal cells remain available for diagnostics and are excluded from network
claims.

### Metric semantics

| Saved channel | Exact meaning |
| --- | --- |
| `endpoint_mean` | Expected target at `cut+h` |
| `cumulative_mean` | Average of frames `cut+1` through `cut+h`; not an integral |
| `peak_mean` | Expected pathwise maximum through the horizon; not the peak of the mean path |
| `event_probability` | Probability of crossing the fold-training pooled target q0.90 by the horizon |
| `endpoint_sd` | `SD_high-SD_low`, using population `ddof=0` |
| `endpoint_log_sd` | `log(SD_high+1e-6)-log(SD_low+1e-6)` |
| `endpoint_wasserstein1` | Unsigned one-dimensional marginal endpoint-law distance within a state |

A phase contrast of W1, such as onset-minus-baseline, is a signed difference of
two unsigned distances. Its sign is not a direction of transport. Within-state
W1 receives no signed zero-centered null test.

### Evidence and multiplicity semantics

The canonical labels are:

- `supported_exploratory`: minimum support and declared animal, sign,
  stability, and raw-support gates pass; progressive rows also pass the f=0.10
  genealogy gate;
- `model_only`: finite and interpretable but one or more promotion gates are
  incomplete;
- `unsupported`: the minimum support/inferential definition fails.

`confirmed` is reserved for independent experimental confirmation and is never
assigned. Canonical counts are 28,485 `supported_exploratory`, 242,247
`model_only`, and 166,068 `unsupported` rows.

`sign_flip_q_value` is BH-adjusted only within the **retained, post-screen
method × channel × context × lag × horizon family**. It is not a global
full-atlas q-value and is not confirmatory after selection. There is no
`global_q_value` field in the canonical table. The external analysis uses a
different global BH family over 110 exact evaluable lag-max tests. Targeted
N=128 intervals are post-selected pointwise worm-bootstrap intervals with no
multiplicity correction.

### Support, repeatability, and lag localization

Mean valid support was 0.76770 for progressive and 0.44956 for direct. Higher
support does not remove Monte Carlo genealogy or model misspecification risk.
Of the 1,000 queue rows, 874 passed the progressive f=0.10 genealogy rule but
failed the stricter f=0.20 sensitivity; the queue is consequently concentrated
and should not be read as 1,000 independent discoveries.

The E26 exact-overlap audit found:

| Method | Median Spearman | Median sign agreement | Median RMSE |
| --- | ---: | ---: | ---: |
| Direct importance | 0.42163 | 0.66597 | 0.07066 |
| Progressive bridge | 0.35103 | 0.62806 | 0.04178 |

These 160 comparisons are historical sensitivity only: they cover horizon one,
four contexts, and five channels; the E26 and canonical progressive runs differ
in future branch factor, and direct support thresholds differ. Canonical median
lag selectivity was only 0.05959. The atlas is more informative about candidate
effects than precise delays.

## Primary N=128 panel

The six frozen state-average candidates completed 5 folds × 2 seeds × 3 lags
used by the selection, totaling 30/30 archives. Every selected effect preserved
screen direction. Median absolute N128-minus-screen change was 0.012286844 and
median magnitude agreement was 0.966324252.

| Source→target | Channel, lag/horizon | Screen | N=128 high−low (95% pointwise CI) |
| --- | --- | ---: | ---: |
| URB→URA | pathwise peak, L1/H8 | 0.201087 | 0.187281 [0.179515, 0.195868] |
| SMD→AVE | endpoint mean, L8/H32 | -0.503634 | -0.472587 [-0.500587, -0.441973] |
| RIV→OLQ | time-average mean, L4/H32 | -0.271262 | -0.293515 [-0.316199, -0.269713] |
| RIC→FLP | crossing probability, L1/H16 | 0.060719 | 0.049952 [0.041398, 0.059885] |
| RIP→URB | pathwise peak, L4/H8 | 0.198023 | 0.189461 [0.177506, 0.200514] |
| ADE→FLP | pathwise peak, L4/H4 | 0.172051 | 0.172502 [0.152219, 0.190782] |

Across all low/factual, high/factual, and high/low analysis rows, labels were 59
`consistent`, 49 `uncertain`, 18 unsigned `descriptive`, and zero
`unsupported`. A signed row is `consistent` only if its interval excludes zero,
valid fraction is at least 0.80, the f=0.10 genealogy rule passes, worm sign
consistency is at least 0.80, and both seed means share the sign. Seed-worm
Spearman is displayed but is not a gate; three consistent rows had a
nonpositive value. All six cells passed f=0.10, but only ADE→FLP passed f=0.20.

The factual arm is a learned free rollout from observed cut history, not an
observed response. Median W1 distances from each repaired arm to factual were
0.518 and 0.527 versus 0.195 between high and low, indicating a substantial
symmetric repair-versus-factual shift. Stable high-minus-low differences do not
by themselves establish factual calibration.

This is selection-conditioned same-family escalation. N=32 used one future
descendant; N=128 uses two and adds a factual arm. It is not independent
confirmation or a pure particle-count experiment.

## Supplemental distributional/stimulus N=128 panel

This separate six-cell selection used no external reference and cannot rerank
the primary queue. It also completed 30/30 raw archives and 126 analysis plus 90
quantile rows.

| Source→target | Context/channel | Screen | N=128 high−low (95% pointwise CI) | Result |
| --- | --- | ---: | ---: | --- |
| AWC→AVA | baseline W1 | 0.606745 | 0.290554 [0.2662, 0.3138] | unsigned descriptive |
| ASH→OLQ | baseline endpoint SD | -0.071418 | -0.059867 [-0.07673, -0.04243] | full signed consistency |
| ASE→ASK | butanone onset−baseline W1 | -0.135020 | -0.012300 [-0.04286, 0.01474] | uncertain |
| AFD→URY | pentanedione onset−baseline log-SD | -0.271780 | -0.000141 [-0.07022, 0.07044] | uncertain |
| AWA→RIB | onset−baseline SD | 0.087549 | 0.005383 [-0.02710, 0.04181] | uncertain |
| AIM→AIB | onset−baseline log-SD | -0.163152 | 0.004373 [-0.05331, 0.05898] | uncertain; sign reversed |

Among the five signed selected effects, direction agreement was 0.80 but median
magnitude agreement only 0.1414; ASH→OLQ was the sole selected signed survivor.
Overall row labels were 18 `consistent`, 102 `uncertain`, six `descriptive`, and
zero `unsupported`. This is direct evidence that broad gain and chemical-event
claims from the N=32 screen need N=128 or prospective follow-up.

## Complete-family and targeted sampler-null calibration

Exact complete-family inference separates evidence that an edge is nonzero
under the frozen learned law from evidence that it varies by source lag. Joint
single-step max-T correction across baseline endpoint mean, baseline endpoint
log-SD, active-minus-baseline endpoint mean, and active-minus-baseline endpoint
log-SD retains 102 baseline-mean edges and 492 individual cells. It retains
zero strong-support edges with resolved non-flat lag structure and zero strict
stimulus-modulated effects.
The lower-support SMD→RID lag result remains a sensitivity finding rather than
a promoted lag claim.

The subsequent sampler-control run did not retrain the model. It completed
70/70 N=128 raw archives for eight fixed cells and evaluated endpoint mean,
endpoint log-SD, and endpoint Wasserstein-1, producing 24 rows. One exact
shared-worm max-T family contains 64 tests: 16 signed observed-versus-zero
tests, 24 excess-over-sampler-control tests, and 24 excess-over-matched-quiet
tests. All 65,536 two-sided sign patterns for 17 worms were reused across the
family; the simultaneous critical value is 3.8246963393338396.

The enumeration is exact conditional on joint worm-vector sign symmetry;
overlapping cross-validation training sets can couple worm estimates. These are
frozen-model calibration p-values, not experimental randomization significance.

Final labels are 15 `sampling_limited`, four
`exceeds_sampling_controls_only`, five
`indistinguishable_from_sampling_controls`, and zero
`exceeds_sampling_and_quiet_controls`. Thus some selected frozen-model effects
rise above finite-particle controls, but none also establishes quiet-time
specificity. The quiet comparison is a matched baseline timing control, not a
stimulus-modulation test or biological no-effect null. No row has a
prespecified biological SESOI or independent confirmation, so no row is
experiment-ready.

## Post-freeze reference analysis

### Equal-denominator lag-1 Randi/Cook comparison

| Method | Randi AUROC/AUPRC | Cook structural | Cook chemical | Cook gap |
| --- | --- | --- | --- | --- |
| Progressive bridge | 0.647/0.329 | 0.611/0.380 | 0.600/0.347 | 0.672/0.132 |
| Direct importance | 0.617/0.320 | 0.619/0.384 | 0.610/0.347 | 0.660/0.128 |
| SBTG-published | 0.622/0.322 | 0.565/0.348 | 0.558/0.312 | 0.621/0.133 |
| SBTG-current | 0.524/0.198 | 0.524/0.291 | 0.517/0.268 | 0.585/0.080 |

These values use the exact common denominator. The analysis also reports
support-qualified Randi/Cook scores, but support filtering changes the edge
set; those scores answer a different question and cannot be compared directly
with equal-denominator SBTG. The near-Cook result is interesting, but anatomical
correspondence is not validation of neural-activity causality.

### Bentley neuromodulator lag-max comparison

Progressive state-average W1 had maximum neuropeptide AUROC 0.581 at lag 1,
permutation p=0.001, and global BH q=0.018. The monoamine/neuropeptide union was
0.535 at lag 1 with the same p and q. Direct W1 reached neuropeptide AUROC 0.590
at native lag 16 (p=0.003, q=0.047), but on the shared lag set {1,8} its lag-1
neuropeptide AUROC was 0.561 (p=0.001, q=0.018) and union AUROC was 0.528
(q=0.018).

No pooled monoamine or named dopamine/serotonin/tyramine/octopamine family
survived. Significant results concentrate in a broad early 1–4-frame band and
are structure-generic. Receptor/connectome presence is not activity, and this
analysis does not support a unique neuromodulator latency or physical-delay
claim.

## Source-cohort validator correction

The shared fold-assignment CSV contains 20 rows: 17 OH16230 source-cohort worms
and three OH15500 rows. The initial N=128 analysis validator compared raw worms
against all 20 and stopped before scientific output. The corrected
`_source_cohort_worms` implementation:

1. recovers the 17 IDs from hashed source-run stimulus schedules;
2. verifies schema, count, and uniqueness;
3. filters the shared fold table to those IDs; and
4. fails closed if any source-cohort worm is missing.

Regressions prove that out-of-cohort rows are ignored and a missing source worm
is rejected. Primary and supplemental analyses now validate exactly 17 OH16230
worms. The frozen raw archives were unchanged. A later presentation correction
made primary report sorting literally absolute-magnitude ordered and exposed
population-sign/seed-rank plus post-selection/no-multiplicity caveats; its 10
focused tests pass and numerical CSV substance is unchanged.

## Provenance, serialization, and code audit

The active raw contract is `prediction_atlas_manifest_v2` /
`prediction_atlas_response_v2`. Nine pre-audit v1 partial archives remain under
`quarantine_pre_v2_provenance/` and are excluded. Each v2 archive fingerprints
the base seed, keyed episode-seed formula, requested/resolved device, common-
noise definition, checkpoint path/hash, fold, worm, neuron order, stimulus
schema/schedule, cut, source-window, and timing metadata. Resume validation is
fail-closed.

Full raw screen arrays are
`[worm,phase,event,source,horizon,target]`. Canonical matrices transpose once to
target rows/source columns. Targeted three-arm arrays are written directly as
`[worm,phase,event,horizon,target,source]`. Synthetic directional tests place a
declared source→target effect at the correct row/column.

The combined current test run is **270 passed**, with one harmless PyTorch
warning. Algebraic W1 checks, exact zero-effect common-noise checks, lag-bound
checks, provenance/resume guards, cohort regressions, orientation tests, and
dashboard/explorer bindings pass.

## Actual storage and GUI state

The scientific bundles are under
`results/neural_prediction_atlas_20260829/`:

- `canonical/`: prediction/support tables, dense/worm NPZs, queue, stimulus
  composition, model/protocol/validation records, and checksum ledger;
- `targeted_selection_n128/` and `targeted_confirmation/{raw,analysis}/`:
  primary selection and N=128 result;
- `targeted_selection_distributional_n128/` and
  `targeted_confirmation_distributional/{raw,analysis}/`: separate
  supplemental result;
- `repeat_stability_e26/`: descriptive historical layer;
- `postfreeze_external/`: comparison-only external layer;
- `complete_family_inference_20260830/` and
  `complete_family_evidence_20260830/`: exact full-family and joint evidence;
- `sampling_null_controls_combined8_n128_20260830/{raw,analysis}/`: 70-archive
  targeted sampler-control run and 24-row calibrated result;
- `gui/`: prior E27 dashboard plus schema-v3 full explorer; and
- `technical_report/`: source-backed Markdown/artifact plus portable HTML.

There is no canonical `global_q_value` field and no
`distribution_sketches.npz`. Targeted distributional detail is stored in
`targeted_cells.csv` and `targeted_quantile_shifts.csv`. The frozen queue is an
analysis output, not a GUI-authored export.

`gui/atlas_explorer.html` is a self-contained, offline, read-only 54×54
application with matrix, perturb-one-source, lag×horizon, queue detail, primary
N=128, stimulus-composition, external-reference, complete-family, calibrated
evidence, and provenance views. Its schema-v3 contract validates canonical,
**primary** targeted, external, complete-family, and targeted sampling-null
inputs. The supplemental N=128 analysis is not embedded. The queue cannot be
saved, annotated, or exported from the current GUI.

The prior E27 dashboard builder receipt reports validation, package, browser
verification, source-dialog interaction, and 1440/390 responsive checks as
passed. Production in-app browser QA of the schema-v3 explorer passed both
viewports, selector semantics, target-row/source-column orientation,
candidate/support isolation, W1 and chemical-stratification labels, and mobile
overflow checks. No page-level exceptions occurred during the exercised
interactions; the server audit recorded only localhost HTML requests and no
external assets. The final in-app audit did not capture raw console APIs.

All eight upstream ledgers pass, covering 54 files, and the generator-owned
technical-report ledger passes. Their narrow scopes remain unchanged. The
atlas-root `checksums.sha256` is the comprehensive delivery layer over raw
archives/validation, root documentation, every nested scientific product and
ledger, GUI/dashboard artifacts and receipts, and the portable report and
receipt.

## Completed delivery and scientific plan

The portable report delivery contains:

- `technical_report/TECHNICAL_REPORT.md`;
- `technical_report/artifact.json`;
- `technical_report/report.html`;
- `technical_report/checksums.sha256`; and
- root `technical_report_builder_receipt.json`.

The report builder passed validation, packaging, source-dialog interaction, and
1440/390 verification. Independent Playwright QA confirmed the table counts,
claim-boundary language, metric presentation, zero console errors/warnings, one
local request, and no mobile overflow. The current report CLI accepts one
targeted analysis directory—the primary panel. The supplemental panel remains a
clearly linked separate analysis unless the generator is extended and
re-audited.

For prospective Dandiset 000981 work, freeze the present 54-neuron atlas as a
hypothesis set, retrain and calibrate on the larger cohort, and require true
held-out replication. Chemical-specific or physical-delay-adjacent claims need
chemical-conditioned inputs, sufficient repeated events, higher temporal
resolution, and ideally controlled perturbations. Because no current row
passes both sampler and quiet-time controls, none should be presented as an
experiment-ready priority. A prospective study may predeclare a small set of
sampler-only or N=128-stable candidates as exploratory hypotheses, with a
biological SESOI and independent worms fixed before outcome inspection.

## Durable audit index

- [`RUN_PROTOCOL.md`](results/neural_prediction_atlas_20260829/RUN_PROTOCOL.md)
- [`RUN_COMMANDS.md`](results/neural_prediction_atlas_20260829/RUN_COMMANDS.md)
- [`INPUT_DATA_AUDIT.md`](results/neural_prediction_atlas_20260829/INPUT_DATA_AUDIT.md)
- [`CODE_AND_ESTIMAND_AUDIT.md`](results/neural_prediction_atlas_20260829/CODE_AND_ESTIMAND_AUDIT.md)
- [`CANONICAL_DATA_AUDIT.md`](results/neural_prediction_atlas_20260829/CANONICAL_DATA_AUDIT.md)
- [`TARGETED_CONFIRMATION_AUDIT.md`](results/neural_prediction_atlas_20260829/TARGETED_CONFIRMATION_AUDIT.md)
- [`TARGETED_DISTRIBUTIONAL_AUDIT.md`](results/neural_prediction_atlas_20260829/TARGETED_DISTRIBUTIONAL_AUDIT.md)
- [`repeat_stability_e26/REPORT.md`](results/neural_prediction_atlas_20260829/repeat_stability_e26/REPORT.md)
- [`POSTFREEZE_EXTERNAL_AUDIT.md`](results/neural_prediction_atlas_20260829/POSTFREEZE_EXTERNAL_AUDIT.md)
- [`COMPLETE_FAMILY_SIGNIFICANCE_20260830.md`](results/neural_prediction_atlas_20260829/COMPLETE_FAMILY_SIGNIFICANCE_20260830.md)
- [`TARGETED_SAMPLING_NULL_CALIBRATION_20260830.md`](results/neural_prediction_atlas_20260829/TARGETED_SAMPLING_NULL_CALIBRATION_20260830.md)
- [`GUI_BROWSER_QA.md`](results/neural_prediction_atlas_20260829/GUI_BROWSER_QA.md)
- [`TECHNICAL_REPORT_BROWSER_QA.md`](results/neural_prediction_atlas_20260829/TECHNICAL_REPORT_BROWSER_QA.md)
- [`technical_report/TECHNICAL_REPORT.md`](results/neural_prediction_atlas_20260829/technical_report/TECHNICAL_REPORT.md)

## Claim boundary

The atlas organizes experimental hypotheses about changes under a learned
conditional activity law and soft history-repair distribution. It does not
show that a matrix entry is a synapse, direct anatomical connection, receptor
action, single-neuron causal effect, or physical delay. `supported_exploratory`
and targeted `consistent` mean that declared internal checks passed; neither
means experimentally confirmed.

The targeted calibration displayed by the schema-v3 explorer further shows
that none of its 24 selected metric rows exceeds both the declared sampling
controls and the matched quiet-time comparison. That is a failure of the stated
promotion gate, not proof that every learned model-relative contrast is
identically zero.
