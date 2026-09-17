# Mechanism-first neuromodulator dynamics benchmark

This directory is an isolated benchmark for the question the observational codebase
cannot answer by itself: **if a living-system-like neuromodulator mechanism is the
ground truth, which statistical methods recover its transition law, physical response
channels, latent kinetics, and interventions?**

It does not modify `SBTG/`, `sid_neuromod/`, `sid_elegans/`, or their outputs.

## Architecture

```text
mechanistic.py             stable release -> concentration -> receptor -> neural DGP
mechanistic_dataset.py     grouped factual/knockout/dose datasets and exact oracles
hierarchy.py               held-out worms and paired observation variants
changepoint.py             independently calibrated change-detection lane
features.py                episode-safe histories and worm-level splits
noise_kernels.py           exact Gaussian and multivariate Student-t DSM kernels
methods/
  classical.py             Gaussian/heteroskedastic/Student-t VAR-family baselines
  dynamics.py              Granger source deletion, sparse map, discrete SINDy
  sid.py                    normalized quadratic SID/DSM with numerical gates
  neural.py                regularized Gaussian/DSM/mixture/score networks
  sbtg.py                  isolated linear and feature-bilinear SBTG score models
  bridge.py                reference-explicit conditional Brownian bridge
  mechanistic_latent.py    learned release/concentration/receptor state-space model
external_causal.py         isolated official PCMCI, VAR-LiNGAM, and PySINDy adapters
evaluation.py              capability-gated proper, operator, mechanism, and intervention metrics
metric_contract.py         machine-checked metric semantics, optima, units, and claim levels
intervention_response.py   common-history pulse/knockout/source-silencing response kernels
response_evaluation.py     model-side response-kernel panels with causal/transfer separation
rollout_metrics.py         proper joint-path score plus dependence/stability diagnostics
runner.py                  resumable sequential execution with memory/disk/thread caps
reporting.py               tidy metric export, validation, and claim-specific summaries
docs/METRIC_CONTRACT.md     normative metric definitions and claim boundaries
docs/METRIC_VALIDATION.md   falsification gates, adversarial controls, and test map
docs/THEORY.md             theorems, counterexamples, estimands, and metric gates
docs/LITERATURE_AND_INSTANTIATIONS.md
                            primary-source map and biological status of each DGP choice
```

## Claim ladder

1. Held-out conditional-law prediction: NLL, energy score, CRPS, PIT/calibration.
2. Finite-time operator probes: linear, quadratic, and tail-exceedance functions.
3. Dynamics recovery: transition Jacobians/equations and continuous graph scores.
4. Physical neuromodulator recovery: release, clearance, receptors, concentration
   susceptibilities, and total modulator-gated response tensors.
5. Latent recovery: validation-fitted alignment evaluated on held-out worms.
6. Interventions: P1 arm-history transfer, secondary explicit arm-history C1, and
   primary controlled C1 only at the exactly common-history lag-one comparison.
7. Change detection: null FPR, power, localization, delay, and channel attribution.
8. External biological correspondence: functional/anatomical maps, always secondary.

There is intentionally no universal scalar leaderboard. A graph-only method has no
NLL cell; an unnormalized score has no fabricated density; an observational graph is
not labeled causal when the modulator is hidden.

Metric identifiers are also not free-form. The machine registry rejects unknown
families and misspelled terminal statistics and attaches an estimand, claim ceiling,
direction, attainable optimum, unit, and primary/secondary/guardrail role to every
reported value.

## Reproducible environments

The main benchmark uses the workspace environment without changing it. Published
causal baselines live in `.causal_venv` and can be recreated with:

```bash
scripts/bootstrap_causal_env.sh
```

The environment is ignored by version control and currently pins Tigramite 5.2.10.1,
LiNGAM 1.12.2, and PySINDy 2.1.0.

## Tests and runs

```bash
PYTHONPATH=src ../.venv/bin/python -m pytest -q
scripts/run_pilot.sh
scripts/run_medium.sh
scripts/run_full.sh
scripts/run_robustness_medium.sh
scripts/run_memory_medium.sh
scripts/run_tail_mechanistic_medium.sh
```

Developmental boundary tuning is summarized in
[`outputs/DEVELOPMENTAL_BOUNDARY_SUMMARY.md`](outputs/DEVELOPMENTAL_BOUNDARY_SUMMARY.md).
Those seeds set fixed corruption/reference strata and capacity sensitivities; their
test values are not confirmatory outcomes.

All long runs are sequential, low priority, thread-capped, resumable, and stop if free
disk falls below 4 GiB. Outputs are constrained to `outputs/`; per-case JSON records
are atomic and include the split, selected hyperparameters, capabilities, method
metadata, wall/CPU/RSS use, environment versions, and failures.

## Interpretation essentials

- “Response gain” means a response slope/excitability/gating quantity.
- A conditional log-variance derivative is **stochastic dispersion**, not gain.
- Joint SBTG score moments are localization probes, not SID dispersion derivatives.
- DSM learns a corrupted score unless a clean inverse/reverse process is validated.
- Conditional-outcome DSM risk and SBTG joint consecutive-state risk are different
  estimands even under the same corruption kernel and scale; metric IDs encode the domain.
- A Schrödinger bridge is relative to its reference path law.
- Fair energy scores assess observed forecast accuracy and have
  distribution-dependent optima; Brownian reference KL is a directionless
  control-effort diagnostic.
- Generic joint rollout diagnostics are integrated for the bridge adapter only;
  independent one-step draws cannot be relabeled as a path forecast.
- Connectome agreement cannot validate latent concentration, stochastic dispersion,
  or tail modulation.
- Synthetic parameter recovery is not proof of biological correctness; held-out
  perturbation and across-worm tests remain mandatory.
