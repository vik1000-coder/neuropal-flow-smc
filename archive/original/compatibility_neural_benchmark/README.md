# Compatibility-aware NeuroPAL path responses

This package applies the repaired-path estimand in
`methods/compatibility_aware_path_responses/main.tex` to the frozen conditional
NeuroPAL generators.

## Claim boundary

Every reported quantity is an observational, model-relative response of the
learned finite-memory transition law under a fixed stimulus schedule. It is not
an identified physical intervention, direct synapse, or anatomical edge.

## Primary frozen design

- Cohort: the existing 20-worm, 54-neuron complete-case benchmark cohort.
- Splits: the original five whole-worm outer folds.
- Generators: TCN residual flow matching and TCN residual MDN-4, three seeds.
- History: 80 frames / 20 seconds.
- Source statistic: one-second average ending at the temporal cut.
- Contrast: training-fold, phase-specific 25th versus 75th percentile.
- Primary repair: four frames / one second.
- Horizons: 1, 2, 4, 8, 16, 24, 32, and 40 frames.
- Primary clamp: Gaussian bandwidth equal to 0.25 of the training-fold IQR.
- Anchor: whitened 12-dimensional population displacement, with the source
  omitted throughout the source-statistic window.
- Sampling: shared natural path importance sampling, which targets the same
  potential-reweighted repaired law while sharing expensive rollouts across all
  54 sources.
- Required diagnostics: source/repair/total log compatibility, ESS, maximum
  weight, entropy-equivalent ancestors, achieved source law, anchor cost, and
  explicit validity/abstention.

The all-pair score is the maximum absolute cumulative-mean response over the
prespecified horizons, averaged over stimulus phases and divided by the
achieved source displacement. Pairwise tests instead use a signed statistic
averaged over all prespecified phases and horizons, so no atlas label or pair
outcome selects a lag.

The package also contains a literal bootstrap-SMC estimator. It maintains one
particle system per low/high source query, applies the factual anchor one repair
frame at a time, checks ESS after every update, systematically resamples when
ESS falls below the declared threshold, applies the source clamp at the
terminal repair frame, terminally resamples, drops the weights, and rolls the
future forward freely. The direct importance estimator and SMC estimator target
the same repaired path law but have different finite-particle error.

## Commands

Run repaired responses:

```bash
.venv/bin/python -m compatibility_neural_benchmark.runner \
  --source-run results/conditional_distribution_benchmark/final_overnight_20260826 \
  --run-dir results/compatibility_path_response/primary_20260826 \
  --models flow mdn4 --folds 0 1 2 3 4 --seeds 1701 2903 4307 \
  --repair-frames 4 --n-particles 128 --device mps --resume
```

Run the matched SBTG comparator:

```bash
PYTHONPATH=.:SBTG .venv/bin/python -m compatibility_neural_benchmark.sbtg_baseline \
  --source-run results/conditional_distribution_benchmark/final_overnight_20260826 \
  --run-dir results/compatibility_path_response/sbtg_matched_20260826 \
  --lags 1 2 4 8 16 24 32 40 --folds 0 1 2 3 4 \
  --epochs 80 --inner-folds 2 --device mps --resume
```

Run bootstrap SMC with ESS-triggered resampling on the frozen flow checkpoints:

```bash
.venv/bin/python -m compatibility_neural_benchmark.smc_runner \
  --source-run results/conditional_distribution_benchmark/final_overnight_20260826 \
  --run-dir results/compatibility_path_response/smc_flow_N128_20260826 \
  --folds 0 1 2 3 4 --seeds 1701 2903 4307 \
  --n-particles 128 --horizons 1 2 4 8 16 24 32 40 \
  --resample-ess-fraction 0.5 --sampling-chunk-size 1024 \
  --device mps --resume
```

Run the repair-window sensitivity (repeat with `--repair-frames 16` and the
matching output directory):

```bash
.venv/bin/python -m compatibility_neural_benchmark.runner \
  --source-run results/conditional_distribution_benchmark/final_overnight_20260826 \
  --run-dir results/compatibility_path_response/sensitivity_B8_20260826 \
  --models flow mdn4 --folds 0 1 2 3 4 --seeds 1701 \
  --repair-frames 8 --n-particles 128 --device mps --resume
```

Run frozen external evaluation:

```bash
.venv/bin/python -m compatibility_neural_benchmark.evaluate \
  --response-run results/compatibility_path_response/primary_20260826 \
  --sbtg-run results/compatibility_path_response/sbtg_matched_20260826 \
  --source-run results/conditional_distribution_benchmark/final_overnight_20260826 \
  --sensitivity-run-8 results/compatibility_path_response/sensitivity_B8_20260826 \
  --sensitivity-run-16 results/compatibility_path_response/sensitivity_B16_20260826 \
  --particle-sensitivity-run results/compatibility_path_response/sensitivity_N256_B4_20260826 \
  --output-dir results/compatibility_path_response/final_analysis_20260826 \
  --bootstrap 2000 --permutations 2000
```

Atlas labels are read only after the response, compatibility, and SBTG
configurations are frozen. The evaluator reports both an animal-block effect
integrated over all prespecified horizons and a lower-power lag-specific family
with one global BH correction over every ordered pair and horizon.

Run the no-retraining shared-neuron comparison with the published SBTG release,
Randi, Cook, and Bentley neuromodulator networks:

```bash
.venv/bin/python -m compatibility_neural_benchmark.fair_atlas_analysis \
  --current-analysis results/compatibility_path_response/final_analysis_20260826 \
  --smc-run results/compatibility_path_response/smc_flow_N128_20260826 \
  --published-release '/Users/vik/Downloads/SBTG-public-release copy' \
  --output-dir results/compatibility_path_response/fair_atlas_analysis_20260826 \
  --bootstrap 2000
```

Generate the reproducible method-diagnostic tables and figures used by the
technical postmortem:

```bash
.venv/bin/python -m compatibility_neural_benchmark.diagnose_methods \
  --analysis-dir results/compatibility_path_response/fair_atlas_analysis_20260826 \
  --primary-run results/compatibility_path_response/primary_20260826 \
  --smc-run results/compatibility_path_response/smc_flow_N128_20260826 \
  --published-release '/Users/vik/Downloads/SBTG-public-release copy' \
  --output-dir results/compatibility_path_response/fair_atlas_analysis_20260826
```

Run the sealed one-fold frozen-generator estimator benchmark. The production
protocol uses two independent 4,096-particle direct-importance references,
three matched Monte Carlo seeds for each 128-particle estimator, and never
loads an external biological atlas during estimator selection.

Progressive repetitions use:

```bash
.venv/bin/python -m compatibility_neural_benchmark.progressive_smc_runner \
  --source-run results/conditional_distribution_benchmark/final_overnight_20260826 \
  --run-dir results/compatibility_path_response/frozen_estimator_benchmark_20260826/progressive_mc20260826 \
  --folds 0 --seeds 1701 --n-particles 128 \
  --branch-factor 2 --future-branch-factor 2 \
  --tempering-ess-fraction 0.65 --max-tempering-resamples 8 \
  --horizons 1 2 4 8 16 24 32 40 --sampling-chunk-size 4096 \
  --device mps --base-seed 20260826
```

Repeat with base seeds `20261826` and `20262826` and matching output
directories. The full evaluation command, exact run directories, calibration
note, and seed scope are saved in
`results/compatibility_path_response/frozen_estimator_benchmark_20260826/analysis/protocol.json`.
The answer-first technical report is
`results/compatibility_path_response/frozen_estimator_benchmark_20260826/analysis/REPORT.md`.

After the Stage-A estimator result is frozen, run the one-time post-freeze
external evaluation. This does not retrain or tune any method:

```bash
PYTHONPATH=. .venv/bin/python -m compatibility_neural_benchmark.postfreeze_external_analysis \
  --stage-a-analysis results/compatibility_path_response/frozen_estimator_benchmark_20260826/analysis \
  --fair-analysis results/compatibility_path_response/fair_atlas_analysis_20260826 \
  --published-release '/Users/vik/Downloads/SBTG-public-release copy' \
  --method-tex methods/compatibility_aware_path_responses/main.tex \
  --output-dir results/compatibility_path_response/postfreeze_progressive_external_20260827 \
  --bootstrap 2000 --seed 20260827
```

The current experiment chronology, result hierarchy, metric definitions, and
supersession rules are recorded in
`results/compatibility_path_response/EXPERIMENT_INDEX.md`.
