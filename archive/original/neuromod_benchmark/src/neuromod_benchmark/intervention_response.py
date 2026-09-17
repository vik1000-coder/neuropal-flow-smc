"""Common-history perturbation response kernels and their recovery metrics.

This module defines a deliberately narrow causal benchmark contract.  A baseline
and an intervention arm share the *same realized history* up to a registered
transition and reuse the same named exogenous innovations thereafter.  The arms
therefore differ only through the requested operation and its downstream state
evolution.

The default response is the difference between the exact one-step conditional
means along the paired arm histories.  Averaging this quantity across independent
paired histories estimates the interventional mean response while avoiding the
current-step innovation noise.  Pathwise neural, concentration, calcium, and
fluorescence responses are also available.

The metric family is intentionally not collapsed to one score: field error, peak,
latency, integral, decay, sign, and null leakage test different properties of a
response kernel.  Normalized errors are undefined for a structurally null truth;
``null_leakage_rms`` is the corresponding null estimand.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Literal, Sequence

import numpy as np

from .mechanistic import (
    ExogenousNoise,
    MechanisticConfig,
    MechanisticParameters,
    MechanisticState,
    OneStepMoments,
    draw_exogenous_noise,
    generate_mechanistic_parameters,
    one_step_moments,
)


InterventionKind = Literal[
    "null", "ligand_pulse", "receptor_knockout", "release_source_silencing"
]
KernelOutcome = Literal[
    "conditional_mean",
    "latent_neural",
    "concentration",
    "clean_calcium",
    "fluorescence",
]


def _index_tuple(values: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (int, np.integer)):
        result = (int(values),)
    else:
        result = tuple(int(value) for value in values)
    if not result:
        raise ValueError("at least one index is required")
    if len(set(result)) != len(result):
        raise ValueError("intervention indices must be unique")
    return result


@dataclass(frozen=True)
class InterventionSpec:
    """A registered operation applied from one transition onward.

    ``duration_steps=None`` means that a knockout or silencing operation persists
    through the response window.  A ligand pulse is a rectangular increment to the
    selected modulator's release bias and must have a finite duration.  Source
    silencing removes only the selected ``release_weights[modulator, source]``
    entries; it does not clamp the source neuron's activity or basal release.
    """

    kind: InterventionKind
    modulator_index: int | None = None
    target_neurons: tuple[int, ...] = ()
    source_neurons: tuple[int, ...] = ()
    amplitude: float = 0.0
    duration_steps: int | None = None

    @classmethod
    def null(cls) -> "InterventionSpec":
        return cls(kind="null")

    @classmethod
    def ligand_pulse(
        cls,
        modulator_index: int,
        *,
        amplitude: float,
        duration_steps: int = 1,
    ) -> "InterventionSpec":
        return cls(
            kind="ligand_pulse",
            modulator_index=int(modulator_index),
            amplitude=float(amplitude),
            duration_steps=int(duration_steps),
        )

    @classmethod
    def receptor_knockout(
        cls,
        modulator_index: int,
        target_neurons: int | Sequence[int],
        *,
        duration_steps: int | None = None,
    ) -> "InterventionSpec":
        return cls(
            kind="receptor_knockout",
            modulator_index=int(modulator_index),
            target_neurons=_index_tuple(target_neurons),
            duration_steps=duration_steps,
        )

    @classmethod
    def release_source_silencing(
        cls,
        modulator_index: int,
        source_neurons: int | Sequence[int],
        *,
        duration_steps: int | None = None,
    ) -> "InterventionSpec":
        return cls(
            kind="release_source_silencing",
            modulator_index=int(modulator_index),
            source_neurons=_index_tuple(source_neurons),
            duration_steps=duration_steps,
        )

    def validate(self, config: MechanisticConfig) -> None:
        allowed = {
            "null",
            "ligand_pulse",
            "receptor_knockout",
            "release_source_silencing",
        }
        if self.kind not in allowed:
            raise ValueError(f"unknown intervention kind {self.kind!r}")
        if self.kind == "null":
            if (
                self.modulator_index is not None
                or self.target_neurons
                or self.source_neurons
                or self.amplitude != 0.0
                or self.duration_steps is not None
            ):
                raise ValueError("null intervention cannot carry operation fields")
            return

        if self.modulator_index is None or not (
            0 <= self.modulator_index < config.n_modulators
        ):
            raise ValueError("modulator_index is outside the configured range")
        if self.duration_steps is not None and not isinstance(
            self.duration_steps, (int, np.integer)
        ):
            raise ValueError("duration_steps must be an integer when supplied")
        if self.duration_steps is not None and self.duration_steps < 1:
            raise ValueError("duration_steps must be positive when supplied")

        if self.kind == "ligand_pulse":
            if not np.isfinite(self.amplitude):
                raise ValueError("ligand amplitude must be finite")
            if self.duration_steps is None:
                raise ValueError("ligand pulse requires a finite duration")
            if self.target_neurons or self.source_neurons:
                raise ValueError("ligand pulse cannot carry receptor/source indices")
        elif self.kind == "receptor_knockout":
            if not self.target_neurons or self.source_neurons or self.amplitude != 0.0:
                raise ValueError("receptor knockout requires only target_neurons")
            if min(self.target_neurons) < 0 or max(self.target_neurons) >= config.n_neurons:
                raise ValueError("target_neurons contains an out-of-range index")
        else:
            if not self.source_neurons or self.target_neurons or self.amplitude != 0.0:
                raise ValueError("release-source silencing requires only source_neurons")
            if min(self.source_neurons) < 0 or max(self.source_neurons) >= config.n_neurons:
                raise ValueError("source_neurons contains an out-of-range index")

    def is_active(self, transition: int, onset_step: int) -> bool:
        elapsed = int(transition) - int(onset_step)
        if self.kind == "null" or elapsed < 0:
            return False
        return self.duration_steps is None or elapsed < self.duration_steps


@dataclass(frozen=True)
class PerturbationPath:
    """Complete paired-arm state and oracle arrays.

    State arrays have length ``T+1`` and transition arrays have length ``T``.
    Thus ``conditional_mean[t]`` is the mean of ``neural_state[t+1]`` given the
    state at index ``t``.
    """

    neural_state: np.ndarray
    concentration_state: np.ndarray
    clean_calcium_state: np.ndarray
    fluorescence_state: np.ndarray
    conditional_mean: np.ndarray
    conditional_variance: np.ndarray
    release: np.ndarray
    occupancy: np.ndarray
    applied_ligand_drive: np.ndarray


@dataclass(frozen=True)
class CommonHistoryPair:
    baseline: PerturbationPath
    intervention: PerturbationPath
    specification: InterventionSpec
    onset_step: int
    horizon: int
    dt_seconds: float
    exogenous_fingerprint: str

    @property
    def common_history_exact(self) -> bool:
        """Whether every realized state/oracle agrees through the branch point."""

        stop = self.onset_step + 1
        state_equal = all(
            np.array_equal(left[:stop], right[:stop])
            for left, right in (
                (self.baseline.neural_state, self.intervention.neural_state),
                (
                    self.baseline.concentration_state,
                    self.intervention.concentration_state,
                ),
                (
                    self.baseline.clean_calcium_state,
                    self.intervention.clean_calcium_state,
                ),
                (
                    self.baseline.fluorescence_state,
                    self.intervention.fluorescence_state,
                ),
            )
        )
        transition_equal = all(
            np.array_equal(left[: self.onset_step], right[: self.onset_step])
            for left, right in (
                (self.baseline.conditional_mean, self.intervention.conditional_mean),
                (
                    self.baseline.conditional_variance,
                    self.intervention.conditional_variance,
                ),
                (self.baseline.release, self.intervention.release),
                (self.baseline.occupancy, self.intervention.occupancy),
                (
                    self.baseline.applied_ligand_drive,
                    self.intervention.applied_ligand_drive,
                ),
            )
        )
        return state_equal and transition_equal


@dataclass(frozen=True)
class ResponseKernel:
    """A mean perturbational response indexed by positive transition lag."""

    lags: np.ndarray
    values: np.ndarray
    outcome: KernelOutcome
    dt_seconds: float
    n_pairs: int = 1
    standard_error: np.ndarray | None = None

    def validate(self) -> None:
        lags = np.asarray(self.lags)
        values = np.asarray(self.values)
        if lags.ndim != 1 or len(lags) < 1:
            raise ValueError("lags must be a non-empty vector")
        if not np.all(np.diff(lags) > 0) or np.any(lags <= 0):
            raise ValueError("lags must be strictly increasing and positive")
        if values.ndim != 2 or values.shape[0] != len(lags):
            raise ValueError("kernel values must have shape [lag, output]")
        if not np.isfinite(values).all():
            raise ValueError("kernel contains non-finite values")
        if self.dt_seconds <= 0 or self.n_pairs < 1:
            raise ValueError("invalid kernel sampling metadata")
        if self.standard_error is not None:
            standard_error = np.asarray(self.standard_error)
            if standard_error.shape != values.shape:
                raise ValueError("standard_error must match kernel values")
            if np.any(standard_error < 0) or not np.isfinite(standard_error).all():
                raise ValueError("standard_error must be finite and nonnegative")


def _ambient_ligand_drive(
    config: MechanisticConfig, exogenous: ExogenousNoise
) -> np.ndarray:
    drive = np.zeros((config.total_steps, config.n_modulators), dtype=float)
    state = np.zeros(config.n_modulators, dtype=float)
    rho = math.exp(-config.dt_seconds / config.ligand_pulse_decay_seconds)
    for transition in range(config.total_steps):
        state = (
            rho * state
            + config.ligand_pulse_amplitude * exogenous.ligand_impulse[transition]
        )
        drive[transition] = state
    return drive


def _masks(
    config: MechanisticConfig, specification: InterventionSpec
) -> tuple[np.ndarray, np.ndarray]:
    receptor = np.zeros(
        (config.n_neurons, config.n_modulators), dtype=bool
    )
    source = np.zeros(
        (config.n_modulators, config.n_neurons), dtype=bool
    )
    if specification.kind == "receptor_knockout":
        receptor[list(specification.target_neurons), specification.modulator_index] = True
    elif specification.kind == "release_source_silencing":
        source[specification.modulator_index, list(specification.source_neurons)] = True
    return receptor, source


def _step_parameters(
    params: MechanisticParameters,
    ligand_drive: np.ndarray,
    source_mask: np.ndarray | None = None,
) -> MechanisticParameters:
    release_weights = params.release_weights
    if source_mask is not None and np.any(source_mask):
        release_weights = np.where(source_mask, 0.0, release_weights)
    if np.any(ligand_drive) or release_weights is not params.release_weights:
        return replace(
            params,
            release_bias=params.release_bias + ligand_drive,
            release_weights=release_weights,
        )
    return params


def _advance(
    config: MechanisticConfig,
    oracle: OneStepMoments,
    state: MechanisticState,
    exogenous: ExogenousNoise,
    transition: int,
) -> tuple[MechanisticState, np.ndarray]:
    if config.mechanism == "correlation_routing":
        loading = oracle.correlation_loading
        standardized = (
            loading * exogenous.correlation_common_normal[transition]
            + np.sqrt(1.0 - loading**2)
            * exogenous.neural_standard_normal[transition]
        )
        innovation = np.sqrt(oracle.conditional_variance) * standardized
    elif config.mechanism in {"matched_tail", "mixed"}:
        probability = oracle.mixture_probability
        high_component = exogenous.mixture_uniform[transition] < probability
        low, high = config.tail_low_scale, config.tail_high_scale
        normalizer2 = (1.0 - probability) * low**2 + probability * high**2
        component_scale = np.where(high_component, high, low) / np.sqrt(normalizer2)
        innovation = (
            np.sqrt(oracle.conditional_variance)
            * component_scale
            * exogenous.neural_standard_normal[transition]
        )
    else:
        innovation = (
            np.sqrt(oracle.conditional_variance)
            * exogenous.neural_standard_normal[transition]
        )

    next_neural = oracle.conditional_mean + innovation
    next_calcium = (
        config.calcium_rho * state.calcium
        + (1.0 - config.calcium_rho) * next_neural
    )
    fluorescence = (
        next_calcium
        + config.measurement_noise
        * exogenous.fluorescence_standard_normal[transition]
    )
    return (
        MechanisticState(next_neural, oracle.next_concentration, next_calcium),
        fluorescence,
    )


def _empty_path(config: MechanisticConfig, n_transitions: int) -> PerturbationPath:
    total, n, k = n_transitions, config.n_neurons, config.n_modulators
    return PerturbationPath(
        neural_state=np.empty((total + 1, n)),
        concentration_state=np.empty((total + 1, k)),
        clean_calcium_state=np.empty((total + 1, n)),
        fluorescence_state=np.empty((total + 1, n)),
        conditional_mean=np.empty((total, n)),
        conditional_variance=np.empty((total, n)),
        release=np.empty((total, k)),
        occupancy=np.empty((total, n, k)),
        applied_ligand_drive=np.empty((total, k)),
    )


def _record_transition(
    path: PerturbationPath,
    transition: int,
    oracle: OneStepMoments,
    next_state: MechanisticState,
    fluorescence: np.ndarray,
    applied_ligand_drive: np.ndarray,
) -> None:
    path.neural_state[transition + 1] = next_state.neural
    path.concentration_state[transition + 1] = next_state.concentration
    path.clean_calcium_state[transition + 1] = next_state.calcium
    path.fluorescence_state[transition + 1] = fluorescence
    path.conditional_mean[transition] = oracle.conditional_mean
    path.conditional_variance[transition] = oracle.conditional_variance
    path.release[transition] = oracle.release
    path.occupancy[transition] = oracle.occupancy
    path.applied_ligand_drive[transition] = applied_ligand_drive


def simulate_common_history_pair(
    config: MechanisticConfig,
    specification: InterventionSpec,
    *,
    params: MechanisticParameters | None = None,
    simulation_seed: int | None = None,
    exogenous: ExogenousNoise | None = None,
    onset_step: int | None = None,
    horizon: int | None = None,
) -> CommonHistoryPair:
    """Branch an intervention from an exactly shared realized history.

    ``onset_step`` indexes the transition on which the intervention first acts.
    The state at index ``onset_step`` is therefore the shared lag-zero state, and
    response lag one is the next state.  By default the branch occurs after the
    configured burn-in and is followed for every configured retained step.
    """

    config.validate()
    specification.validate(config)
    params = generate_mechanistic_parameters(config) if params is None else params
    params.validate(config)
    if exogenous is None:
        exogenous = draw_exogenous_noise(config, simulation_seed)

    onset = config.burn_in if onset_step is None else int(onset_step)
    response_horizon = config.n_steps if horizon is None else int(horizon)
    if onset < 0 or response_horizon < 1:
        raise ValueError("onset_step must be nonnegative and horizon positive")
    if onset + response_horizon > config.total_steps:
        raise ValueError("onset_step + horizon exceeds available exogenous transitions")

    ambient_drive = _ambient_ligand_drive(config, exogenous)
    receptor_mask, source_mask = _masks(config, specification)
    n_transitions = onset + response_horizon
    baseline = _empty_path(config, n_transitions)
    intervention = _empty_path(config, n_transitions)

    initial_neural = np.asarray(exogenous.initial_neural, dtype=float).copy()
    initial_concentration = np.full(
        config.n_modulators, config.concentration_initial, dtype=float
    )
    initial_calcium = np.zeros(config.n_neurons, dtype=float)
    baseline_state = MechanisticState(
        initial_neural.copy(), initial_concentration.copy(), initial_calcium.copy()
    )
    intervention_state = MechanisticState(
        initial_neural.copy(), initial_concentration.copy(), initial_calcium.copy()
    )
    for path in (baseline, intervention):
        path.neural_state[0] = initial_neural
        path.concentration_state[0] = initial_concentration
        path.clean_calcium_state[0] = initial_calcium
        path.fluorescence_state[0] = initial_calcium

    for transition in range(n_transitions):
        base_params = _step_parameters(params, ambient_drive[transition])
        base_oracle = one_step_moments(config, base_params, baseline_state)
        next_baseline, base_fluorescence = _advance(
            config, base_oracle, baseline_state, exogenous, transition
        )
        _record_transition(
            baseline,
            transition,
            base_oracle,
            next_baseline,
            base_fluorescence,
            ambient_drive[transition],
        )

        active = specification.is_active(transition, onset)
        if transition < onset:
            # Before the registered branch point, copy rather than recompute.  This
            # makes common-history equality a construction invariant, not a tolerance.
            intervention_state = MechanisticState(
                next_baseline.neural.copy(),
                next_baseline.concentration.copy(),
                next_baseline.calcium.copy(),
            )
            for name in (
                "neural_state",
                "concentration_state",
                "clean_calcium_state",
                "fluorescence_state",
                "conditional_mean",
                "conditional_variance",
                "release",
                "occupancy",
                "applied_ligand_drive",
            ):
                target = getattr(intervention, name)
                source = getattr(baseline, name)
                index = transition + 1 if name.endswith("_state") else transition
                target[index] = source[index]
        else:
            controlled_drive = ambient_drive[transition].copy()
            if active and specification.kind == "ligand_pulse":
                controlled_drive[specification.modulator_index] += specification.amplitude
            active_source_mask = (
                source_mask
                if active and specification.kind == "release_source_silencing"
                else None
            )
            intervention_params = _step_parameters(
                params, controlled_drive, active_source_mask
            )
            active_receptor_mask = (
                receptor_mask
                if active and specification.kind == "receptor_knockout"
                else None
            )
            intervention_oracle = one_step_moments(
                config,
                intervention_params,
                intervention_state,
                knockout_mask=active_receptor_mask,
            )
            next_intervention, intervention_fluorescence = _advance(
                config,
                intervention_oracle,
                intervention_state,
                exogenous,
                transition,
            )
            _record_transition(
                intervention,
                transition,
                intervention_oracle,
                next_intervention,
                intervention_fluorescence,
                controlled_drive,
            )
            intervention_state = next_intervention

        baseline_state = next_baseline

    pair = CommonHistoryPair(
        baseline=baseline,
        intervention=intervention,
        specification=specification,
        onset_step=onset,
        horizon=response_horizon,
        dt_seconds=config.dt_seconds,
        exogenous_fingerprint=exogenous.fingerprint,
    )
    if not pair.common_history_exact:
        raise AssertionError("intervention arms do not share an exact pre-intervention history")
    return pair


def response_kernel(
    pair: CommonHistoryPair,
    *,
    outcome: KernelOutcome = "conditional_mean",
) -> ResponseKernel:
    """Extract an intervention-minus-baseline response over positive lags."""

    onset, stop = pair.onset_step, pair.onset_step + pair.horizon
    if outcome == "conditional_mean":
        baseline = pair.baseline.conditional_mean[onset:stop]
        intervention = pair.intervention.conditional_mean[onset:stop]
    else:
        field = {
            "latent_neural": "neural_state",
            "concentration": "concentration_state",
            "clean_calcium": "clean_calcium_state",
            "fluorescence": "fluorescence_state",
        }.get(outcome)
        if field is None:
            raise ValueError(f"unknown response outcome {outcome!r}")
        baseline = getattr(pair.baseline, field)[onset + 1 : stop + 1]
        intervention = getattr(pair.intervention, field)[onset + 1 : stop + 1]
    kernel = ResponseKernel(
        lags=np.arange(1, pair.horizon + 1, dtype=int),
        values=np.asarray(intervention - baseline, dtype=float),
        outcome=outcome,
        dt_seconds=pair.dt_seconds,
    )
    kernel.validate()
    return kernel


def aggregate_response_kernels(
    pairs: Iterable[CommonHistoryPair],
    *,
    outcome: KernelOutcome = "conditional_mean",
) -> ResponseKernel:
    """Average response kernels over independent common-random-number pairs."""

    kernels = [response_kernel(pair, outcome=outcome) for pair in pairs]
    if not kernels:
        raise ValueError("at least one response pair is required")
    reference = kernels[0]
    for kernel in kernels[1:]:
        if (
            kernel.outcome != reference.outcome
            or kernel.dt_seconds != reference.dt_seconds
            or not np.array_equal(kernel.lags, reference.lags)
            or kernel.values.shape != reference.values.shape
        ):
            raise ValueError("response kernels do not share a common contract")
    stack = np.stack([kernel.values for kernel in kernels], axis=0)
    standard_error = (
        None
        if len(kernels) == 1
        else np.std(stack, axis=0, ddof=1) / math.sqrt(len(kernels))
    )
    result = ResponseKernel(
        lags=reference.lags.copy(),
        values=np.mean(stack, axis=0),
        outcome=outcome,
        dt_seconds=reference.dt_seconds,
        n_pairs=len(kernels),
        standard_error=standard_error,
    )
    result.validate()
    return result


def _safe_normalized(error: float, scale: float, null_tolerance: float) -> float:
    return float("nan") if scale <= null_tolerance else float(error / scale)


def _peak_indices(values: np.ndarray) -> np.ndarray:
    return np.argmax(np.abs(values), axis=0)


def _decay_elapsed_lags(
    values: np.ndarray, lags: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return 1/e decay times and whether each curve crosses inside the window."""

    peak_indices = _peak_indices(values)
    elapsed = np.full(values.shape[1], np.nan, dtype=float)
    observed = np.zeros(values.shape[1], dtype=bool)
    typical_step = float(np.median(np.diff(np.concatenate([[0], lags]))))
    right_boundary = float(lags[-1]) + typical_step
    for output, peak_index in enumerate(peak_indices):
        magnitude = abs(float(values[peak_index, output]))
        if magnitude == 0.0:
            continue
        candidates = np.flatnonzero(
            np.abs(values[peak_index + 1 :, output]) <= magnitude / math.e
        )
        if len(candidates):
            decay_index = peak_index + 1 + int(candidates[0])
            elapsed[output] = float(lags[decay_index] - lags[peak_index])
            observed[output] = True
        else:
            # This value is a right-censoring lower bound.  It is used only when an
            # estimate misses a decay that is observed in the ground-truth window.
            elapsed[output] = right_boundary - float(lags[peak_index])
    return elapsed, observed


def response_kernel_metrics(
    estimate: ResponseKernel,
    truth: ResponseKernel,
    *,
    beta_min_fraction: float = 0.05,
    null_tolerance: float = 1e-12,
    persistent_operation: bool = False,
    end_window_fraction: float = 0.2,
) -> dict[str, float]:
    """Score complementary properties of an estimated response kernel.

    Peak and timing summaries operate on the absolute response but the windowed
    cumulative response remains signed. Sign accuracy is evaluated only where the
    true effect exceeds ``max(null_tolerance, beta_min_fraction * max(abs(truth)))``.
    A peak at the final registered lag is right-censored and is excluded from timing
    error. Return-to-zero 1/e decay is meaningful only for transient operations; a
    persistent operation instead receives an explicitly window-limited end gain.
    """

    estimate.validate()
    truth.validate()
    if (
        beta_min_fraction < 0
        or null_tolerance < 0
        or not 0 < end_window_fraction <= 1
    ):
        raise ValueError("metric thresholds must be nonnegative")
    if (
        estimate.outcome != truth.outcome
        or estimate.dt_seconds != truth.dt_seconds
        or not np.array_equal(estimate.lags, truth.lags)
        or estimate.values.shape != truth.values.shape
    ):
        raise ValueError("estimated and true kernels do not share a contract")

    estimated = np.asarray(estimate.values, dtype=float)
    true = np.asarray(truth.values, dtype=float)
    difference = estimated - true
    rmse = float(np.sqrt(np.mean(difference**2)))
    true_rms = float(np.sqrt(np.mean(true**2)))
    estimate_rms = float(np.sqrt(np.mean(estimated**2)))

    true_peak_index = _peak_indices(true)
    estimate_peak_index = _peak_indices(estimated)
    true_peak = np.max(np.abs(true), axis=0)
    estimate_peak = np.max(np.abs(estimated), axis=0)
    beta_min = max(null_tolerance, beta_min_fraction * float(np.max(np.abs(true))))
    active_outputs = true_peak > beta_min
    active_entries = np.abs(true) > beta_min

    peak_error = np.abs(estimate_peak - true_peak)
    peak_mae = float(np.mean(peak_error))
    peak_scale = float(np.sqrt(np.mean(true_peak**2)))
    truth_peak_at_final = true_peak_index == len(truth.lags) - 1
    estimate_peak_at_final = estimate_peak_index == len(estimate.lags) - 1
    peak_timing_scored = active_outputs & ~truth_peak_at_final
    if np.any(peak_timing_scored):
        peak_lag_error = np.abs(
            estimate.lags[estimate_peak_index[peak_timing_scored]]
            - truth.lags[true_peak_index[peak_timing_scored]]
        )
        peak_lag_mae = float(np.mean(peak_lag_error))
    else:
        peak_lag_mae = float("nan")

    cumulative_true = truth.dt_seconds * np.sum(true, axis=0)
    cumulative_estimate = estimate.dt_seconds * np.sum(estimated, axis=0)
    cumulative_rmse = float(
        np.sqrt(np.mean((cumulative_estimate - cumulative_true) ** 2))
    )
    cumulative_scale = float(np.sqrt(np.mean(cumulative_true**2)))

    true_decay, true_decay_observed = _decay_elapsed_lags(true, truth.lags)
    estimate_decay, estimate_decay_observed = _decay_elapsed_lags(
        estimated, estimate.lags
    )
    decay_scored = (
        active_outputs & ~truth_peak_at_final & true_decay_observed
        if not persistent_operation
        else np.zeros_like(active_outputs)
    )
    if np.any(decay_scored):
        decay_lag_mae = float(
            np.mean(np.abs(estimate_decay[decay_scored] - true_decay[decay_scored]))
        )
        decay_censor_mismatch = float(
            np.mean(~estimate_decay_observed[decay_scored])
        )
    else:
        decay_lag_mae = float("nan")
        decay_censor_mismatch = float("nan")

    end_count = max(1, int(math.ceil(end_window_fraction * len(truth.lags))))
    if persistent_operation:
        end_true = np.mean(true[-end_count:], axis=0)
        end_estimate = np.mean(estimated[-end_count:], axis=0)
        end_gain_rmse = float(np.sqrt(np.mean((end_estimate - end_true) ** 2)))
        end_gain_scale = float(np.sqrt(np.mean(end_true**2)))
    else:
        end_gain_rmse = float("nan")
        end_gain_scale = float("nan")

    sign_accuracy = (
        float(np.mean(np.sign(estimated[active_entries]) == np.sign(true[active_entries])))
        if np.any(active_entries)
        else float("nan")
    )
    null_entries = np.abs(true) <= null_tolerance
    null_leakage = (
        float(np.sqrt(np.mean(estimated[null_entries] ** 2)))
        if np.any(null_entries)
        else float("nan")
    )

    return {
        "response_kernel.rmse": rmse,
        "response_kernel.normalized_rmse": _safe_normalized(
            rmse, true_rms, null_tolerance
        ),
        "response_kernel.truth_rms": true_rms,
        "response_kernel.estimate_rms": estimate_rms,
        "response_kernel.peak_magnitude_mae": peak_mae,
        "response_kernel.peak_magnitude_normalized_rmse": _safe_normalized(
            float(np.sqrt(np.mean(peak_error**2))), peak_scale, null_tolerance
        ),
        "response_kernel.time_to_peak_mae_steps": peak_lag_mae,
        "response_kernel.time_to_peak_mae_seconds": peak_lag_mae * truth.dt_seconds,
        "response_kernel.time_to_peak_estimable_fraction": float(
            np.mean(peak_timing_scored)
        ),
        "response_kernel.truth_peak_at_final_lag_fraction": float(
            np.mean(truth_peak_at_final & active_outputs)
        ),
        "response_kernel.estimate_peak_at_final_lag_fraction": float(
            np.mean(estimate_peak_at_final & active_outputs)
        ),
        "response_kernel.windowed_cumulative_response_rmse": cumulative_rmse,
        "response_kernel.windowed_cumulative_response_normalized_rmse": _safe_normalized(
            cumulative_rmse, cumulative_scale, null_tolerance
        ),
        "response_kernel.decay_time_mae_steps": decay_lag_mae,
        "response_kernel.decay_time_mae_seconds": decay_lag_mae * truth.dt_seconds,
        "response_kernel.decay_estimable_fraction": float(np.mean(decay_scored)),
        "response_kernel.decay_censor_mismatch_fraction": decay_censor_mismatch,
        "response_kernel.end_window_gain_rmse": end_gain_rmse,
        "response_kernel.end_window_gain_normalized_rmse": (
            _safe_normalized(end_gain_rmse, end_gain_scale, null_tolerance)
            if persistent_operation
            else float("nan")
        ),
        "response_kernel.end_window_gain_applicable": float(persistent_operation),
        "response_kernel.end_window_fraction": float(end_count / len(truth.lags)),
        "response_kernel.sign_accuracy": sign_accuracy,
        "response_kernel.active_entry_fraction": float(np.mean(active_entries)),
        "response_kernel.null_leakage_rms": null_leakage,
        "response_kernel.null_entry_fraction": float(np.mean(null_entries)),
    }


__all__ = [
    "CommonHistoryPair",
    "InterventionSpec",
    "KernelOutcome",
    "PerturbationPath",
    "ResponseKernel",
    "aggregate_response_kernels",
    "response_kernel",
    "response_kernel_metrics",
    "simulate_common_history_pair",
]
