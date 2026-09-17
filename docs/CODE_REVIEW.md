# Publication code review — 17 September 2026

## Scope and evidence

The publication pass checks every archived Python file for parseability, every
archived file against its source hash, and the core NeuroPAL/synthetic suites by
execution. Scientific manual review follows data loading → folds/scaling → residual
target → flow loss/sampling → repair potentials → finite contrasts → aggregation →
reference scoring → multiplicity and numerical diagnostics. It also checks the
September synthetic oracle, analytic conditioning and dataset-level aggregation.

This is not a claim that every line in all historical exploratory programs has
been independently proven correct. Older workflows lacking result dependencies
cannot be rerun from the available archive. The boundary is explicit rather than
calling a syntax check a scientific validation. See `audit/` for actual outcomes.

## Implementation-to-documentation map

Paths below are relative to `archive/fresh/replication_20260911/` unless stated.

| Stage | Code | Checked interpretation |
|---|---|---|
| Inputs/cohorts | `source/conditional_neural_benchmark/data.py` | MAT `norm_traces`; neuron-class averaging; corrected schedules; historical donor/pseudo-pairing retained |
| Folds/scaling | same and `source/conditional_neural_benchmark/runner.py` | Whole-recording test/validation folds; scaler fit on training recordings; pooled vocabulary selection precedes splitting |
| Targets/training | `train.py`, runner `_neural_trial` | One-step standardized residual; held-out validation checkpoint; no Cook/Randi labels in model loss |
| Flow | `source/history_tangent_benchmark/src/history_tangent_benchmark/models.py`, `ConditionalFlowMatching` | Independent Gaussian/data coupling, velocity MSE, fixed validation noise, Heun ODE generation |
| History encoder | `source/conditional_neural_benchmark/models.py` | Legacy TCN local receptive field 31; GroupNorm gives indirect earlier-history influence |
| Repair law | `source/compatibility_neural_benchmark/core.py` | Soft Gaussian source-window clamp and low-rank anchor; no factual future after forecast cut |
| Progressive bridge | `source/compatibility_neural_benchmark/progressive_smc.py` | Potential increments telescope; both changing provisional energy and increasing clamp strength are accounted for |
| Response | `statistical_analysis.py`, `analyze.py` | High-minus-low divided by achieved source gap with primary absolute-gap floor; not automatic differentiation |
| Axes | `orient_raw`, aggregation | Raw source/horizon/target becomes horizon/target/source; recordings remain inference units |
| Cook/Randi | `analyze.references`, `binary` | Absolute scores; common eligible off-diagonal pairs; Randi ambiguous pairs excluded; AP is average precision |
| Bootstrap | `paired_source_bootstrap` | Resample source columns with fitted matrices held fixed; not full refit/animal uncertainty |
| Lag inference | `exact_max_t`, `lag_inference` | Joint two-sided sign-orbit maximum across declared families; assumes joint sign symmetry; historical traces not independent animals |
| Numerical controls | `adjudicate_stopped.py` | Paired MC differences, validity/positive-gap gates, reference qualification and unequal computation made explicit |
| Synthetic truth | synthetic `benchmark.py`, `analytic_effects.py` | Known derivatives, correlated Gaussian/mixture conditioning, Student-t quadrature; independent data seeds aggregated before comparisons |

## Corrections and limitations carried forward

1. Clean54 is cleaner than historical80, not a pristine fold-nested selection design.
2. Normalized traces are not camera-level raw fluorescence; class axes are not single
   identified left/right neurons. The original pooled affine scaling cancels under
   training-fold scaling to approximately 3.8e-6, but historical imputation does not.
3. Training and sampling leading-missing-value handling differ. Registered response
   windows avoid the affected leading frames; do not claim identical policies.
4. Historical80 uses an eight-frame model history; clean54 uses 80. Do not label
   every model as a 20-second local-receptive-field model.
5. Source quantiles/projection use non-test recordings, including validation;
   describe that separately from training-only optimizer/scaler data.
6. Source window is four frames, followed by a lag gap; source placement lag and
   forecast horizon are different axes. Their sum describes endpoint separation,
   not an identified physical synaptic delay.
7. Primary gap-floor matrices retain unsupported episodes. Strong-source masks and
   strict complete-case exclusions are distinct sensitivities. Strict clean54
   retains zero complete entries; this is not zero effect or failure of every episode.
8. Identical-arm common-noise zero contrasts test coupled symmetry, not independent
   sampling variance. Training seed stability is materially lower than MC repeat stability.
9. Smaller-step/solver differences must be compared with MC noise. High-particle
   references are qualified, not assumed exact. Equal N does not mean equal cost.
10. Existing bootstrap and exact-sign results condition on fitted models. Selected
    neuron examples are descriptive, selection-conditioned, and not independent confirmation.
11. No chemical-aware model passed every historical gate. No historical E22
    chemical/repetition claim is reinstated. Binary stimulus atlases remain separate.
12. The fresh study reused released SBTG matrices, not newly trained SBTG. Matched
    SBTG-vs-flow runtime superiority cannot be asserted without a matched timing experiment.
13. Synthetic GP is independent-output, and Transformer predicts three outputs;
    results do not establish 54/80-dimensional architecture superiority.
14. Original absolute paths, stale status text and one legacy invalid-escape warning
    are preserved as history. New tools use repository-relative paths; frozen
    scripts are not rewritten, avoiding invalidated hashes and silent result drift.

## Packaging fixes

The initial joint pytest invocation exposed duplicate `test_core.py` module names.
The supported runner uses importlib collection and subprocess isolation. Environment
folders, caches and Git internals are excluded. Large artifacts are restored with
checksums, atomic writes and a path-safe extraction routine. Generated publication
figures never overwrite frozen figures or reports.
