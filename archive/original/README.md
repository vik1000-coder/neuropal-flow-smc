# Neural conditional dynamics and repaired-path sampling

This workspace is a scientific experiment archive centered on two operations:

1. **Train a conditional model** of the next neural state from recent neural and stimulus history.
2. **Sample repaired paths** from the frozen model with progressive bridge SMC to estimate model-relative low-versus-high source responses.

The current default pairing is a regularized residual conditional flow plus progressive bridge SMC. This is a predictive and Monte Carlo pipeline. Its response matrices are observational, model-relative quantities—not physical interventions, direct synapses, anatomical edges, receptor effects, or transmission delays.

## Start here

| Need | Read or run |
| --- | --- |
| View clear figures and biological questions | [`FIGURES.md`](FIGURES.md) |
| Open, understand, or maintain the current dashboard | [`DASHBOARD.md`](DASHBOARD.md) |
| Understand the two-stage pipeline | [`pipeline/README.md`](pipeline/README.md) |
| Find current plans, audits, and methods documents | [`docs/current/README.md`](docs/current/README.md) |
| Find canonical versus developmental results | [`results/README.md`](results/README.md) |
| Review the complete experiment and supersession ledger | [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) |
| Query the experiment registry programmatically | [`EXPERIMENT_REGISTRY.csv`](EXPERIMENT_REGISTRY.csv) |
| Understand older SID workstreams | [`docs/history/README.md`](docs/history/README.md) |
| Prepare the prospective DANDI:000981 campaign | [`dandi000981_lag_cluster_bundle/README_CLUSTER.md`](dandi000981_lag_cluster_bundle/README_CLUSTER.md) |

## Current pipeline

```text
NeuroPAL activity + causal stimulus history
                  |
                  v
conditional_neural_benchmark/
  residual conditional-flow training
  whole-worm validation
  one-step and multistep predictive scoring
                  |
             frozen checkpoints
                  |
                  v
compatibility_neural_benchmark/
  progressive bridge SMC
  low/high repaired-path contrasts
  ESS, ancestry, support, and stability diagnostics
                  |
                  v
results/
  frozen matrices, reports, validation, and checksums
```

### Stage 1 — conditional model

- Code: [`conditional_neural_benchmark/`](conditional_neural_benchmark/)
- Current model family: residual TCN conditional flow.
- Current regularization: dropout, modest weight decay, and small neural-history jitter.
- Selection inputs: held-out-worm energy, stimulus-balanced energy, variogram, calibration, and multistep rollout behavior.
- External atlases must not enter training, early stopping, or model selection.

### Stage 2 — repaired-path estimator

- Code: [`compatibility_neural_benchmark/`](compatibility_neural_benchmark/)
- Current primary estimator: progressive bridge SMC.
- Independent sensitivity estimator: direct importance sampling with a larger path bank.
- Required diagnostics: effective sample size, maximum weight, achieved source gap, valid-source fraction, repaired-root ancestry, generator-seed agreement, cross-sampler agreement, animal stability, and timing stability.

## Directory map

| Path | Role | Status |
| --- | --- | --- |
| `conditional_neural_benchmark/` | Conditional-model training and evaluation | **Active core** |
| `compatibility_neural_benchmark/` | Direct, terminal, temporal-cut, and progressive bridge samplers | **Active core** |
| `pipeline/` | Stable conceptual and operational map of the two stages | **Start here** |
| `docs/current/` | Current methods, audit, experiment, and atlas-plan index | **Current documentation** |
| `results/` | Frozen and developmental experiment artifacts | **See its README before citing** |
| `dandi000981_lag_cluster_bundle/` | Prospective larger-cohort cluster campaign | **Active next-data bundle** |
| `methods/` | Preserved technical method attachments | Reference |
| `cluster_specs/` | Earlier cluster specifications and ingestion scaffolding | Reference |
| `SBTG/`, `sid_neuromod/`, `sid_elegans/` | Foundational SBTG/SID implementations | Historical/foundational |
| `distributional_sid/`, `empirical_sid/`, `history_tangent_benchmark/`, `neuromod_benchmark/` | Earlier method-development workstreams | Historical/developmental |
| `analysis/`, `reports/`, `paper/`, `output/` | Earlier analyses and generated publications | Historical/reporting |
| `tmp/`, `*_work/`, caches | Regenerable or interrupted work products | Never cite as canonical |

## Canonical current evidence

The [31 August figure collection](results/figure_atlas_20260831/README.md) is a visual reading companion to these records, with plotted CSVs, significance boundaries, and [candidate experimental questions](results/figure_atlas_20260831/QUESTIONS_AND_EVIDENCE.md). It introduces no new model, sampling run, or significance test.

Use these in this order:

1. [`results/neuron_class_effects_20260831/README.md`](results/neuron_class_effects_20260831/README.md) for the current results synthesis, sourced sensory/interneuron/motor summaries, and why the head-only 54-class axis differs from historical 80.
2. [`results/neural_prediction_atlas_20260829/README.md`](results/neural_prediction_atlas_20260829/README.md) for the frozen E27 atlas and E29 complete-family/sampler-calibration evidence: some supported effects, no strong-support resolved lags.
3. [`results/distribution_structure_20260831/INSIGHTS.md`](results/distribution_structure_20260831/INSIGHTS.md) for the E30 predictive-distribution audit; it did not replace the atlas lag matrices.
4. [`results/four_sampler_lag_connectome_20260828/analysis/PROGRESSIVE_BRIDGE_SMC_TECHNICAL_REPORT.md`](results/four_sampler_lag_connectome_20260828/analysis/PROGRESSIVE_BRIDGE_SMC_TECHNICAL_REPORT.md) for the earlier corrected E26 four-sampler comparison.
5. [`results/compatibility_path_response/EXPERIMENT_INDEX.md`](results/compatibility_path_response/EXPERIMENT_INDEX.md) for the earlier conditional-model and repaired-response chronology, including the sealed finite-particle benchmark.

E31 is a post-hoc descriptive regrouping, not new training or significance testing. Historical reports retain their original scope and numbers; follow the newer explicit adjudications before interpreting an older biological claim.

## Working rules

- Preserve frozen result paths and checksums. Do not rename result folders merely for tidiness.
- Keep model selection atlas-blind. External correspondence belongs after model and matrix freeze.
- Separate conditional-model quality from sampler quality.
- Separate raw onset responses from onset-minus-matched-control responses.
- Treat the worm as the biological replication unit; particles, descendants, horizons, and events are not additional animals.
- Prefer the latest explicitly adjudicating report when documents disagree.
- Preserve negative results and supersession history.

## Quick verification

```bash
./.venv/bin/python -m pytest conditional_neural_benchmark/tests -q
./.venv/bin/python -m pytest compatibility_neural_benchmark/tests -q
```

The root directory is intentionally not being physically flattened: it contains several older independent projects and checksum-bound experiment trees. The navigation files above define the supported organization while preserving reproducibility.
