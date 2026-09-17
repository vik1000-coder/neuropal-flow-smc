# Source and audit notes

## User-supplied theory document

- `Hierarchical_Theory_of_Neuromodulatory_Dynamics_LaTeX 2.pdf`, 67 pages,
  created 2026-07-11.
- Relevant requirements: explicit measurement layer; five nested inferential
  levels; separate mean, gain, variance, covariance, hazard, state, and path
  estimands; multiple independent noise axes; held-out prediction,
  calibration, perturbation fidelity, influence-tensor stability, kernel
  fidelity, and dwell-time fidelity.

## Current objective adjudication

- `neuromod_benchmark/configs/objective_adjudication_20260712.yaml`.
- Five DGP seeds, ten neurons, three modulators, one history lag, one-step
  horizon, complete-state view, and ten trajectories per scenario.
- The simulator uses deterministic release conditional on the current state,
  deterministic clearance and instantaneous Hill occupancy. Neural innovations
  are conditionally Gaussian except for the registered matched-tail mixture;
  the objective panel does not pass candidate methods through the calcium
  observation layer.

## Simulator implementation

- `neuromod_benchmark/src/neuromod_benchmark/mechanistic.py`.
- Implements bounded tanh neural transitions, positive release, exponential
  clearance, Hill occupancy, typed mean/variance/tail/correlation channels, a
  linear calcium filter, and additive Gaussian fluorescence noise.
- Does not currently implement stochastic vesicle release, receptor kinetic
  states/desensitization, spatial diffusion, nonlinear fluorescence saturation,
  shared artifacts, missing neurons, switching behavioral states, or
  semi-Markov dwell laws.

## Existing robustness and memory lanes

- `neuromod_benchmark/configs/frozen_v2/robustness_light.yaml` and
  `robustness_neural.yaml`: eight neurons, one-step calcium prediction, 6/2/3
  train/validation/test worms, coefficient CVs 0/0.15/0.35, and paired
  observation variants consisting of reference, 2.5x measurement noise, and
  2x calcium decay.
- `neuromod_benchmark/configs/frozen_v2/memory_light.yaml` and
  `memory_neural.yaml`: eight neurons and maximum lags 1/2/4/8/16, but the
  underlying latent generator remains in the same mechanistic family.
- `neuromod_benchmark/configs/frozen_v2/core.yaml`: at most twelve neurons and
  direct horizons 1 and 4.

## Prior synthesis

- `SYNTHETIC_METHODS_REPORT_2026-07-12.md` and
  `reports/sid_synthesis_2026-07-12/main.tex`.
- Existing conclusion: likelihood/classical dynamics win the currently tested
  matched low-dimensional regimes; finite contrasts work for location and
  low-rank mixture changes; covariance/skew adapters and gating readouts fail
  their zero baselines.

## Interpretation boundary

The coverage audit assesses whether the current synthetic evidence supports the
intended high-dimensional, nonlinear, partially observed use case. It does not
invalidate current unit-test results. It reclassifies their external scope.
