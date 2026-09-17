# Flow training and progressive bridge SMC code audit

Audit date: 2026-08-31. Scope: the code and saved artifacts behind the 17-worm,
54-neuron NeuroPAL prediction atlas. Production code, checkpoints, and frozen
results were not changed. All new files are confined to this audit directory.

**Assessment: the core sampler passed the checks performed; the upstream training
analysis and reproducibility controls need revision.** I found no demonstrated
wrong-target SMC weighting error or raw-to-atlas orientation error. That is a
bounded result, not a certification of every execution path or of biological
validity. The corrections below do not currently overturn selection of the
binary-any-stimulus flow or justify promoting any atlas row to experiment-ready.

Evidence: [executable probes](probes.py), [probe results](probe_results.json),
[test log](pytest.log), and [probe log](probes.log). Current reviewed source hashes
and runtime are recorded separately in `reviewed_sources.sha256` and
`audit_environment.json`; they do not retroactively authenticate training-time
source code.

## Exact lineage inspected

| Stage | Executed lineage / implementation |
| --- | --- |
| Training run | `results/conditional_distribution_benchmark/chemical_encoding_primary_20260828_oh16230`, phase `chemical_full_cv` |
| Selected model | `stim_binary_any_stimulus_tcn_flow128_dropout15_jitter_p01`; 80-frame history; width 128; dropout 0.15; weight decay 0.00075; neural-history jitter 0.01; natural-window training |
| Training entry and split | `conditional_neural_benchmark/chemical_encoding_runner.py` → `runner._split_windows` → `runner._neural_trial` |
| Actual neural implementation | `conditional_neural_benchmark/models.py` wraps `history_tangent_benchmark/src/history_tangent_benchmark/models.py`; the latter implements the flow head **and optimizer loop** |
| Sampling | `compatibility_neural_benchmark/prediction_atlas_runner.py` → `progressive_smc.py`; common transition, anchor, and direct-importance routines in `core.py` |
| Frozen screen | `results/neural_prediction_atlas_20260829/{direct_n256,progressive_n32_s1701,progressive_n32_s2903}`; 80 raw archives |
| Canonical aggregation | `prediction_atlas_analysis.py` → `canonical/` |
| Later evidence | `full_family_inference.py`, `full_family_evidence.py`, `targeted_sampling_nulls.py`, `targeted_sampling_null_analysis.py` |
| Presentation | `prediction_atlas_dashboard.py` produces the older bounded E27 summary; `prediction_atlas_explorer.py` produces the schema-v3 explorer containing the later E29 evidence |

The flow is independent-coupling conditional flow matching. Its training target
is the next-frame residual, the interpolation is `(1-t)*noise+t*residual`, and
the velocity target is `residual-noise`. Sampling uses 24 Heun steps and adds
back the last observed/generated state. I inspected these equations directly in
`history_tangent_benchmark/models.py:1109`, `:1121`, and
`conditional_neural_benchmark/runner.py:510`. The validation noise/time draws
are fixed through a local generator, and checkpointing uses validation loss.

## Confirmed findings, ordered by practical importance

### 1. [Medium] Training comparison intervals and p-values reuse worms as independent observations

Location: [chemical_encoding_analysis.py](/Users/vik/Developer/new_sbtg_neuro/conditional_neural_benchmark/chemical_encoding_analysis.py:83).

The analysis creates ten differences indexed by five folds × two checkpoint
seeds, bootstraps those ten entries independently, and applies ordinary
one-sample t and binomial sign tests to them. Both seeds of a fold evaluate the
same animals. Different folds also use overlapping fitting sets. These are not
ten independent biological replications, so the reported intervals and p-values
do not have the advertised interpretation as ordinary independent-sample
inference.

This is observable in the saved data: for chemical-onehot minus subject-shuffled
chemical-onehot, the two seed-specific fold-difference series have correlation
**0.924**. Using audit seed 33, the ten-row bootstrap gives an interval of
**[-0.006630, -0.002021]**. Averaging seeds within fold before a five-fold
bootstrap gives **[-0.007356, -0.001404]**. This example illustrates the
dependency, not a claim that fold bootstrapping fully solves it. The direction
of interval distortion need not be the same for every model comparison.

The frozen launch gate uses mean improvement and 6/10 wins, not these p-values
or intervals. The no-chemical-aware-promotion decision therefore has not been
shown to change. The 6/10 rule can remain a declared engineering stability rule,
but its ten entries must not be described as independent animals.

Correction: retain per-worm held-out scores; reduce model seeds within worm;
use an explicitly defined biological resampling procedure. Refit/reselect within
an outer procedure, or use an independent cohort, when claiming uncertainty
for the full learned-and-selected pipeline. Until then, label the existing
fold/seed intervals as descriptive resampling summaries.

### 2. [Medium] An incompatible training resume can silently mix configurations

Locations: [resume validation](/Users/vik/Developer/new_sbtg_neuro/conditional_neural_benchmark/chemical_encoding_runner.py:77),
[completion lookup](/Users/vik/Developer/new_sbtg_neuro/conditional_neural_benchmark/chemical_encoding_runner.py:98),
[caller](/Users/vik/Developer/new_sbtg_neuro/conditional_neural_benchmark/chemical_encoding_runner.py:220).

The resume validator compares only the stimulus and fold fingerprints.
`is_done` checks phase/model ID/fold/seed, but not lag, full model configuration,
optimizer settings, evaluation settings, or checkpoint existence/hash. A
partially completed run resumed with changed `--lag`, epochs, sample count, or
model defaults can reuse old successful rows and execute the remaining rows
under different settings. A complete run can also be silently skipped despite
changed requested settings.

Reproduction: a temporary copy of the real training manifest with lag changed
from **80 to 81** is accepted by `validate_resume_manifest` when the two checked
fingerprints are unchanged. No real manifest was edited.

Correction: compare an immutable, complete run specification at resume; include
resolved model/optimizer/evaluation settings and code/input identities; validate
the checkpoint for every skipped row. Changes should create a new result lineage.
**I found no evidence that this defect contaminated the ten selected checkpoints.**

### 3. [Medium] The 31-frame-only receptive-field explanation is false for the implemented TCN

Locations: [TCNBlock](/Users/vik/Developer/new_sbtg_neuro/conditional_neural_benchmark/models.py:64),
[architecture claim](/Users/vik/Developer/new_sbtg_neuro/conditional_neural_benchmark/models.py:103),
[generated tournament explanation](/Users/vik/Developer/new_sbtg_neuro/conditional_neural_benchmark/biological_world_model_runner.py:206).

The dilated convolutions have a 31-frame receptive field when considered alone.
However, every block applies `GroupNorm(1, width)` to `[batch, channels, time]`.
Its statistics include the time dimension, so even early input frames can affect
the final context through normalization. This is consistent with the
[PyTorch implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/group_norm.cpp),
which includes dimensions after batch and channel in its normalization extent.

On the actual frozen fold-0/seed-1701 checkpoint, changing only the oldest 49
neural-history frames changed a context coordinate by **0.06635**. A random
projection of that context had nonzero gradient through those old frames
(absolute-gradient sum **3.5428**). The probe uses synthetic input to establish
architectural dependence; it does not estimate the biological importance of
those frames.

Consequently, the tournament narrative that the legacy model could not access
stimulus-phase information older than 7.75 seconds is not established by its
convolution dilation count. Intermediate TCN activations are also not strictly
prefix-causal because normalization spans later positions within the supplied
history. **This is not target-time leakage:** the supplied history still ends
before the next-frame target.

Correction: distinguish local convolutional paths from global normalization
dependence. If a strictly local/prefix-causal comparison is wanted, use a
time-local normalization and retrain as a new experiment. Do not change
normalization underneath the frozen checkpoints. Correct historical explanations
through a dated addendum rather than rewriting sealed outputs.

### 4. [Medium] The headline “equal worm×chemical” score is equal-fold weighted

Locations: [per-worm reduction](/Users/vik/Developer/new_sbtg_neuro/conditional_neural_benchmark/runner.py:278),
[leaderboard aggregation](/Users/vik/Developer/new_sbtg_neuro/conditional_neural_benchmark/chemical_encoding_runner.py:122),
and `chemical_encoding_analysis.py:146`.

Each trial correctly averages its worm×chemical cells. The leaderboard then
takes an unweighted mean of the ten fold/seed trial scores. The primary folds
have **3, 3, 3, 4, 4 worms**, or 9/9/9/12/12 worm×chemical cells per seed.
Thus an individual worm in a three-worm fold receives 4/3 the weight of a worm
in a four-worm fold in the final headline score.

| Encoding | Saved equal-fold score | Recomputed equal-worm×chemical score |
| --- | ---: | ---: |
| Binary any-stimulus | 1.077671 | **1.076736** |
| Chemical + position | 1.078166 | 1.077613 |
| Chemical one-hot | 1.079209 | 1.078554 |

The binary model remains best. Its gap to chemical+position changes from
0.000495 to 0.000877. Weighting the saved trial scores by their complete
worm×chemical-cell counts exactly repairs this particular population weighting;
it does not repair the dependence/selection issue in finding 1.

Correction: either report equal-fold cross-validation performance explicitly or
pool per-worm scores with equal worm weight. Propagate any revised headline
numbers to new dashboard/report versions without altering historical results.

### 5. [Medium] Exact historical training implementation cannot be authenticated from this lineage

Locations: [training manifest construction](/Users/vik/Developer/new_sbtg_neuro/conditional_neural_benchmark/chemical_encoding_runner.py:198)
and `history_tangent_benchmark/src/history_tangent_benchmark/models.py`.

The selected training manifest has schedule/fold identities but no launch-time
source digest, source archive, full optimizer/run specification, or raw neural
input digest. The checkpoints store model settings and fit traces, but no
implementation fingerprint. The actual flow head and optimizer are imported
from the separate `history_tangent_benchmark` tree. The generic
`runner._source_hash` covers only `conditional_neural_benchmark`, and the chemical
runner does not use it.

There **are** useful later receipts: the atlas-root checksum ledger includes
the sampler, analysis, and presentation source files, and those checked source
hashes match. `ENVIRONMENT.json` also records the delivery runtime. These are
not proof of the exact upstream training source bytes at launch. The targeted
sampler-null report already discloses its separate raw-launch provenance gap.
This audit does not replace that disclosure or pretend no source receipts exist.

Correction: future runs should snapshot the transitive training/sampling code,
resolved configuration, raw input identities, and runtime at launch; bind this
identity to checkpoints and raw response archives and enforce it on resume.
Do not backfill a historical launch hash using today's files.

### 6. [Low] The root delivery ledger is stale for two moving index files

The atlas-root ledger verifies **464/466 entries**. Its only failures are
`../../EXPERIMENT_INDEX.md` and `../../EXPERIMENT_REGISTRY.csv`, which now include
later work. The canonical numerical ledger passes **12/12**, complete-family
evidence passes **14/14**, and sampler-null analysis passes **12/12**. All other
root-ledger entries pass, including the checked raw archives and source entries.

This is a documentation-version mismatch, not evidence that the matrices were
corrupted. Keep the historical ledger intact. Snapshot shared indices inside a
future delivery bundle or record a dated supersession receipt; do not continue
claiming that the whole historical root ledger passes unchanged today.

## Checks that passed

- **307 existing tests passed**, with the known PyTorch nested-tensor warning.
  This includes training, sampler mechanics, distribution summaries, canonical
  aggregation, complete-family inference, sampler-null analysis, and presentation
  payload validation. Test coverage is not a substitute for the independent
  probes below.
- **10/10 selected checkpoint hashes match.** Recomputed training-only scaler
  means and scales match every checkpoint exactly. Train, validation, and test
  worm sets are disjoint for each fold. This validates those saved nuisance
  statistics; it does not replay training.
- **Core bridge correction is present.** At each repair step, the increment is
  `-lambda*anchor_increment + beta_previous*(energy_current-energy_previous)`,
  followed by `(beta_current-beta_previous)*energy_current`. Combined, the
  bridge contribution is `beta_current*energy_current -
  beta_previous*energy_previous`. It telescopes to the terminal source clamp
  because the last beta is one and the last energy uses the completed source
  average. Resampling transports histories, source sums, and prior energies
  together. The anchor omits the selected source only inside its source window.
- **Analytic Gaussian oracle:** 144 SMC runs cover lags 1/4/16, particle counts
  32/128/1024, two coupled state coordinates, nonzero population anchoring,
  a four-frame source window, and horizons 1/4. The reference comes from a
  Gaussian path precision matrix plus the exact quadratic anchor and source
  potential, not from the direct estimator. Across both source and target
  coordinates/horizons, run RMSE was:

  | Source lag | N=32 | N=128 | N=1024 |
  | --- | ---: | ---: | ---: |
  | 1 | 0.06379 | 0.03670 | 0.01169 |
  | 4 | 0.04200 | 0.02252 | 0.01075 |
  | 16 | 0.00434 | 0.00394 | 0.00285 |

  Errors decreased with particle count in these cases. This supports the
  implementation on the tested linear-Gaussian targets; it does not establish
  convergence for the nonlinear flow or every selective/clamped regime.
- **Independent raw-to-atlas reconstruction:** queue rank 1, URB→URA,
  state-average peak response at lag 1/horizon 8 reconstructs as **0.201087253**
  versus saved **0.201087251**. Queue rank 92, AWA→BAG baseline endpoint mean,
  reconstructs as **0.308375193** versus saved **0.308375180**. Both use 17 worms,
  two seeds reduced within worm, and the raw achieved-gap denominators. Maximum
  worm-array discrepancy is 2.98e-8 for the first and zero for the second.
- Recomputed canonical counts: **436,800 prediction rows**, **5,616 support rows**,
  and **28,485 supported-exploratory / 242,247 model-only / 166,068 unsupported**.
  These reproduce the documented summary.
- The flow sample head uses a private seeded generator; loading calls `.eval()`;
  low/high stochastic coupling is deliberate. I found no accidental active
  dropout at inference in the reviewed loader path.
- The later atlas inference reduces seeds/events within worms and shares each
  worm's sign across the full test tensor. The training fold/seed issue in
  finding 1 should not be conflated with that later implementation.

## Scientific and presentation boundaries

The latest explorer and older dashboard have different evidence scope. The
bounded `gui/dashboard.html` is an E27 summary; `gui/atlas_explorer.html` adds
complete-family and sampler-null evidence. The existing scope documentation
explains this distinction. A fresh visual inspection was not completed because
the browser rejected local-file navigation; I did not bypass that restriction.
This audit checks the saved artifacts, source, prior QA records, and automated
presentation tests, not current rendered layout.

The canonical post-screen BH values do not provide full-atlas confirmatory FDR
control. The later complete-family layer addresses family selection more
directly, but its exact sign enumeration still depends on joint sign symmetry
and does not remove dependence from overlapping cross-validation fits. The
targeted N=128 controls remain selected on this cohort. These limitations are
already disclosed in the newer methods report and are not newly discovered
implementation bugs.

Terminal ESS after resampling is not a certificate of independent paths or
small response error. The genealogy and independent same-state reruns are
necessary companion diagnostics. The later sampler-null artifact reports zero
rows passing both sampling and quiet-time gates. Nothing in this audit provides
a basis to override that result.

Changing source lag also changes repair length, observed boundary, and the
time-averaged anchor. A lag profile compares those declared repaired-path
problems; it is not automatically a physical delay kernel. All effects remain
conditional on the learned observational law. Chemical labels are event strata
under a binary-stimulus generator, not chemical-conditioned interventions.

Global affine preprocessing in the combined loader deserves documentation, but
its centering/common positive scale is removed algebraically by the subsequent
training-fold per-neuron standardization. I did not count that alone as a
demonstrated leakage bug. Cohort feature selection is fixed globally, and model
selection uses the available cross-validation results; there is no untouched
cohort establishing generalization after the entire selection process.

Remaining unverified items: original optimizer execution, nonlinear-flow
high-particle convergence over the entire atlas, sensitivity to flow ODE step
count, full refit/model-selection uncertainty, and independent biological
replication. No full GPU training or production atlas rerun was attempted.

## Recommended correction sequence

1. Harden training resume and record complete launch provenance before another
   expensive run.
2. Add a dated correction to training-score weighting, inference-unit labels,
   and the receptive-field explanation. Preserve the historical bundles.
3. Promote the independent Gaussian oracle and actual-normalization dependency
   checks into maintained scientific regression tests.
4. Before upgrading biological claims, evaluate prespecified effects on
   independent data with appropriate model/refit uncertainty and solver/particle
   convergence checks. Keep current atlas use exploratory.

Reproduce from the workspace root:

```bash
./.venv/bin/python reports/flow_smc_code_audit_20260831/probes.py
./.venv/bin/python -m pytest conditional_neural_benchmark/tests compatibility_neural_benchmark/tests -q
```
