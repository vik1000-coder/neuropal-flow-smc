"""Matched changepoint experiments for mechanistic neuromodulator dynamics.

The lane in this module is deliberately separate from predictive and graph-recovery
benchmarks.  Every changed trajectory has one named boundary, is null before that
boundary, and reuses one set of structural parameters.  Calibration is possible only
from independent, parameter-matched no-change trajectories.

The detectors expose different claim axes instead of being collapsed into one score:

* ``mean_cusum``: a marginal activity/mean change;
* ``variance_reliability``: a change in residual dispersion/reliability;
* ``conditional_rule_chow``: a change in the one-step conditional mean rule;
* ``residual_tail_shape``: a scale-normalized conditional residual-tail change;
* ``energy_distance`` and ``mmd_rbf``: a general distribution-law change.

The first four are interpretable channel/rule probes.  The last two are deliberately
broad: they can detect a matched-tail change, but do not by themselves identify it as
a tail mechanism.  Only the residual-tail probe can make a tail-specific attribution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from statistics import NormalDist
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import beta as beta_distribution

from .mechanistic import (
    MechanisticConfig,
    MechanisticParameters,
    MechanisticState,
    draw_exogenous_noise,
    generate_mechanistic_parameters,
    one_step_moments,
    with_mechanism,
)


SCENARIO_TO_MECHANISM: Final[dict[str, str]] = {
    "null": "null",
    "mean": "additive_mean",
    "dispersion": "innovation_variance",
    "matched_tail": "matched_tail",
}

TRUTH_CHANNEL: Final[dict[str, str | None]] = {
    "null": None,
    "mean": "mean",
    "dispersion": "dispersion",
    "matched_tail": "tail_shape",
}

METHOD_CLAIM_AXIS: Final[dict[str, str]] = {
    "mean_cusum": "marginal_activity_mean",
    "variance_reliability": "conditional_reliability_dispersion",
    "conditional_rule_chow": "conditional_mean_rule",
    "residual_tail_shape": "conditional_residual_tail_shape",
    "energy_distance": "general_distribution_law",
    "mmd_rbf": "general_distribution_law",
}

METHOD_CHANNEL: Final[dict[str, str]] = {
    "mean_cusum": "mean",
    "variance_reliability": "dispersion",
    "conditional_rule_chow": "mean",
    "residual_tail_shape": "tail_shape",
    "energy_distance": "general_law",
    "mmd_rbf": "general_law",
}

DEFAULT_METHODS: Final[tuple[str, ...]] = tuple(METHOD_CLAIM_AXIS)


@dataclass(frozen=True)
class PostChangeStrengths:
    """Post-boundary channel strengths without changing the matched null system.

    Multipliers act on the already-realized mechanism tensors.  Tail mixture settings
    are post-change only and always retain the analytic mean/variance normalization in
    :func:`one_step_moments`.  None of these fields enters ``match_key``: independent
    null calibration is matched to the shared pre-change system and observation view.
    """

    mean_multiplier: float = 1.0
    logvariance_multiplier: float = 1.0
    tail_logit_multiplier: float = 1.0
    tail_base_probability: float | None = None
    tail_low_scale: float | None = None
    tail_high_scale: float | None = None

    def validate(self) -> None:
        for name in (
            "mean_multiplier",
            "logvariance_multiplier",
            "tail_logit_multiplier",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.tail_base_probability is not None and not (
            0.0 < self.tail_base_probability < 0.5
        ):
            raise ValueError("tail_base_probability must lie in (0, .5)")
        if self.tail_low_scale is not None and not 0.0 < self.tail_low_scale < 1.0:
            raise ValueError("tail_low_scale must lie in (0, 1)")
        if self.tail_high_scale is not None and self.tail_high_scale <= 1.0:
            raise ValueError("tail_high_scale must exceed one")


DEFAULT_POST_CHANGE_STRENGTHS: Final[dict[str, PostChangeStrengths]] = {
    "mean": PostChangeStrengths(mean_multiplier=1.0),
    "dispersion": PostChangeStrengths(logvariance_multiplier=1.0),
    "matched_tail": PostChangeStrengths(tail_logit_multiplier=1.0),
}


def _array_digest(values: Iterable[np.ndarray]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _parameter_digest(params: MechanisticParameters) -> str:
    return _array_digest(np.asarray(getattr(params, item.name)) for item in fields(params))


def _match_key(
    config: MechanisticConfig,
    params: MechanisticParameters,
    observation: str,
) -> str:
    baseline = asdict(with_mechanism(config, "null"))
    payload = json.dumps(baseline, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(payload.encode("utf-8"))
    digest.update(observation.encode("utf-8"))
    digest.update(_parameter_digest(params).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ChangeSeries:
    """One observed trajectory and its changepoint ground truth.

    ``entry_*_shift`` arrays compare the active and null one-step laws at the same
    realized state.  They distinguish a pure mechanism entry from downstream state
    propagation, which can alter later marginal moments even when the entry channel is
    pure.
    """

    values: np.ndarray
    scenario: str
    series_id: str
    simulation_seed: int
    match_key: str
    observation: str = "latent_neural"
    boundary: int | None = None
    boundary_name: str | None = None
    post_change_strengths: PostChangeStrengths | None = None
    conditional_mean: np.ndarray | None = None
    conditional_variance: np.ndarray | None = None
    centered_fourth_moment: np.ndarray | None = None
    mixture_probability: np.ndarray | None = None
    entry_mean_shift: np.ndarray | None = None
    entry_variance_shift: np.ndarray | None = None
    entry_fourth_shift: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 1:
            raise ValueError("values must have shape [time, channel]")
        if not np.isfinite(values).all():
            raise ValueError("values contain non-finite entries")
        if self.scenario not in SCENARIO_TO_MECHANISM:
            raise ValueError(f"unknown changepoint scenario {self.scenario!r}")
        if not self.series_id or not self.match_key:
            raise ValueError("series_id and match_key must be non-empty")
        if self.scenario == "null":
            if self.boundary is not None or self.boundary_name is not None:
                raise ValueError("a no-change series cannot have a true boundary")
        else:
            if self.boundary is None or not 1 <= self.boundary < values.shape[0]:
                raise ValueError("changed series require an interior boundary")
            if not self.boundary_name:
                raise ValueError("changed series require a named boundary")
        if self.post_change_strengths is not None:
            self.post_change_strengths.validate()
        for name in (
            "conditional_mean",
            "conditional_variance",
            "centered_fourth_moment",
            "mixture_probability",
            "entry_mean_shift",
            "entry_variance_shift",
            "entry_fourth_shift",
        ):
            optional = getattr(self, name)
            if optional is not None and np.asarray(optional).shape != values.shape:
                raise ValueError(f"{name} must have the same shape as values")


def _post_change_system(
    config: MechanisticConfig,
    params: MechanisticParameters,
    scenario: str,
    strengths: PostChangeStrengths,
) -> tuple[MechanisticConfig, MechanisticParameters]:
    """Return a channel-scaled post system while leaving baseline objects untouched."""

    strengths.validate()
    post_config = with_mechanism(config, SCENARIO_TO_MECHANISM[scenario])
    post_params = params
    if scenario == "mean":
        post_params = replace(
            params,
            additive_effect=params.additive_effect * strengths.mean_multiplier,
        )
    elif scenario == "dispersion":
        post_params = replace(
            params,
            logvariance_effect=(
                params.logvariance_effect * strengths.logvariance_multiplier
            ),
        )
    elif scenario == "matched_tail":
        post_params = replace(
            params,
            tail_logit_effect=(
                params.tail_logit_effect * strengths.tail_logit_multiplier
            ),
        )
        replacements: dict[str, float] = {}
        if strengths.tail_base_probability is not None:
            replacements["tail_base_probability"] = strengths.tail_base_probability
        if strengths.tail_low_scale is not None:
            replacements["tail_low_scale"] = strengths.tail_low_scale
        if strengths.tail_high_scale is not None:
            replacements["tail_high_scale"] = strengths.tail_high_scale
        if replacements:
            post_config = replace(post_config, **replacements)
    post_config.validate()
    post_params.validate(post_config)
    return post_config, post_params


def simulate_changepoint_series(
    config: MechanisticConfig,
    *,
    scenario: str,
    boundary: int | None = None,
    params: MechanisticParameters | None = None,
    simulation_seed: int | None = None,
    observation: str = "latent_neural",
    boundary_name: str = "neuromodulator_onset",
    series_id: str | None = None,
    post_change_strengths: PostChangeStrengths | None = None,
) -> ChangeSeries:
    """Simulate a continuous-state, one-boundary mechanistic trajectory.

    Burn-in and all samples before ``boundary`` use the null transition law.  The
    structural parameters and named exogenous streams are shared across the boundary.
    A null scenario has no boundary, but otherwise matches every simulation setting.
    """

    if scenario not in SCENARIO_TO_MECHANISM:
        raise ValueError(f"unknown changepoint scenario {scenario!r}")
    baseline = with_mechanism(config, "null")
    baseline.validate()
    params = generate_mechanistic_parameters(baseline) if params is None else params
    params.validate(baseline)
    strengths = post_change_strengths or DEFAULT_POST_CHANGE_STRENGTHS.get(
        scenario, PostChangeStrengths()
    )
    post, post_params = _post_change_system(baseline, params, scenario, strengths)

    if scenario == "null":
        true_boundary = None
    else:
        if boundary is None:
            boundary = config.n_steps // 2
        if not 1 <= int(boundary) < config.n_steps:
            raise ValueError("boundary must lie strictly inside the observed series")
        true_boundary = int(boundary)

    supported_observations = {"latent_neural", "clean_calcium", "fluorescence"}
    if observation not in supported_observations:
        raise ValueError(
            f"observation must be one of {sorted(supported_observations)}, got {observation!r}"
        )

    sim_seed = config.seed if simulation_seed is None else int(simulation_seed)
    exogenous = draw_exogenous_noise(baseline, sim_seed)
    total, n, k = baseline.total_steps, baseline.n_neurons, baseline.n_modulators
    state = MechanisticState(
        neural=np.asarray(exogenous.initial_neural, dtype=float).copy(),
        concentration=np.full(k, baseline.concentration_initial, dtype=float),
        calcium=np.zeros(n, dtype=float),
    )

    latent = np.empty((total, n), dtype=float)
    calcium = np.empty((total, n), dtype=float)
    fluorescence = np.empty((total, n), dtype=float)
    means = np.empty((total, n), dtype=float)
    variances = np.empty((total, n), dtype=float)
    fourth = np.empty((total, n), dtype=float)
    mixture = np.empty((total, n), dtype=float)
    entry_mean = np.empty((total, n), dtype=float)
    entry_variance = np.empty((total, n), dtype=float)
    entry_fourth = np.empty((total, n), dtype=float)

    for t in range(total):
        observed_index = t - baseline.burn_in
        active = (
            scenario != "null"
            and true_boundary is not None
            and observed_index >= true_boundary
        )
        active_config = post if active else baseline
        active_params = post_params if active else params
        active_oracle = one_step_moments(active_config, active_params, state)
        null_oracle = (
            one_step_moments(baseline, params, state) if active else active_oracle
        )

        if active_config.mechanism == "matched_tail":
            probability = active_oracle.mixture_probability
            high_component = exogenous.mixture_uniform[t] < probability
            low, high = active_config.tail_low_scale, active_config.tail_high_scale
            normalizer2 = (1.0 - probability) * low**2 + probability * high**2
            component_scale = np.where(high_component, high, low) / np.sqrt(normalizer2)
        else:
            component_scale = np.ones(n, dtype=float)

        innovation = (
            np.sqrt(active_oracle.conditional_variance)
            * component_scale
            * exogenous.neural_standard_normal[t]
        )
        next_neural = active_oracle.conditional_mean + innovation
        rho = active_config.calcium_rho
        next_calcium = rho * state.calcium + (1.0 - rho) * next_neural
        next_fluorescence = (
            next_calcium
            + active_config.measurement_noise
            * exogenous.fluorescence_standard_normal[t]
        )

        latent[t] = next_neural
        calcium[t] = next_calcium
        fluorescence[t] = next_fluorescence
        means[t] = active_oracle.conditional_mean
        variances[t] = active_oracle.conditional_variance
        fourth[t] = active_oracle.centered_fourth_moment
        mixture[t] = active_oracle.mixture_probability
        entry_mean[t] = active_oracle.conditional_mean - null_oracle.conditional_mean
        entry_variance[t] = (
            active_oracle.conditional_variance - null_oracle.conditional_variance
        )
        entry_fourth[t] = (
            active_oracle.centered_fourth_moment
            - null_oracle.centered_fourth_moment
        )
        state = MechanisticState(
            neural=next_neural,
            concentration=active_oracle.next_concentration,
            calcium=next_calcium,
        )

    sl = slice(baseline.burn_in, total)
    observed = {
        "latent_neural": latent,
        "clean_calcium": calcium,
        "fluorescence": fluorescence,
    }[observation][sl].copy()
    identifier = series_id or f"{scenario}:{sim_seed}"
    return ChangeSeries(
        values=observed,
        scenario=scenario,
        series_id=identifier,
        simulation_seed=sim_seed,
        match_key=_match_key(baseline, params, observation),
        observation=observation,
        boundary=true_boundary,
        boundary_name=boundary_name if true_boundary is not None else None,
        post_change_strengths=strengths if true_boundary is not None else None,
        conditional_mean=means[sl].copy(),
        conditional_variance=variances[sl].copy(),
        centered_fourth_moment=fourth[sl].copy(),
        mixture_probability=mixture[sl].copy(),
        entry_mean_shift=entry_mean[sl].copy(),
        entry_variance_shift=entry_variance[sl].copy(),
        entry_fourth_shift=entry_fourth[sl].copy(),
    )


def simulate_matched_nulls(
    config: MechanisticConfig,
    *,
    n_series: int,
    params: MechanisticParameters | None = None,
    first_seed: int = 10_000,
    observation: str = "latent_neural",
    id_prefix: str = "calibration-null",
) -> tuple[ChangeSeries, ...]:
    """Generate independent no-change replicates with one shared structural system."""

    if n_series < 1:
        raise ValueError("n_series must be positive")
    baseline = with_mechanism(config, "null")
    params = generate_mechanistic_parameters(baseline) if params is None else params
    return tuple(
        simulate_changepoint_series(
            baseline,
            scenario="null",
            params=params,
            simulation_seed=first_seed + index,
            observation=observation,
            series_id=f"{id_prefix}:{index}",
        )
        for index in range(n_series)
    )


@dataclass(frozen=True)
class ScanResult:
    method: str
    claim_axis: str
    detector_channel: str
    candidate_indices: np.ndarray
    score_curve: np.ndarray
    effect_curve: np.ndarray
    effect_name: str
    tau_hat: int
    max_score: float
    effect_size: float
    diagnostic_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _ScannerOutput:
    score_curve: np.ndarray
    effect_curve: np.ndarray
    effect_name: str
    metadata: Mapping[str, Any]


def _validate_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("detectors require a [time, channel] array")
    if array.shape[0] < 8 or array.shape[1] < 1:
        raise ValueError("trajectory is too short or has no channels")
    if not np.isfinite(array).all():
        raise ValueError("detector input contains non-finite values")
    return array


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = np.std(values, axis=0, ddof=1)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (values - np.mean(values, axis=0)) / scale


def _candidate_indices(length: int, min_segment: int, stride: int) -> np.ndarray:
    if min_segment < 3:
        raise ValueError("min_segment must be at least three")
    if stride < 1:
        raise ValueError("stride must be positive")
    if length < 2 * min_segment:
        raise ValueError(
            f"length {length} cannot support two segments of length {min_segment}"
        )
    return np.arange(min_segment, length - min_segment + 1, stride, dtype=int)


def _mean_cusum(
    values: np.ndarray,
    candidates: np.ndarray,
    **_: Any,
) -> _ScannerOutput:
    length = values.shape[0]
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    total = cumulative[-1]
    variance = np.var(values, axis=0, ddof=1) + 1e-8
    scores = np.empty(candidates.size, dtype=float)
    effects = np.empty(candidates.size, dtype=float)
    for index, split in enumerate(candidates):
        left_mean = cumulative[split] / split
        right_mean = (total - cumulative[split]) / (length - split)
        weight = split * (length - split) / length
        standardized_shift = (left_mean - right_mean) / np.sqrt(variance)
        scores[index] = weight * np.sum(standardized_shift**2)
        effects[index] = float(np.sqrt(np.mean(standardized_shift**2)))
    return _ScannerOutput(scores, effects, "rms_cohen_d", {})


def _ridge_coefficients(design: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    rhs = design.T @ targets
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def _ridge_sse(design: np.ndarray, targets: np.ndarray, ridge: float) -> float:
    coefficient = _ridge_coefficients(design, targets, ridge)
    residual = targets - design @ coefficient
    return float(np.sum(residual**2))


def _transition_design(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    standardized = _standardize(values)
    design = np.column_stack((np.ones(standardized.shape[0] - 1), standardized[:-1]))
    return design, standardized[1:]


def _conditional_residuals(
    values: np.ndarray,
    *,
    ridge: float,
    residual_mode: str,
    reference_fraction: float,
    crossfit_folds: int,
    crossfit_purge: int,
    residual_lags: int,
    residual_nonlinear: bool,
    residual_ewma_decays: tuple[float, ...],
) -> tuple[np.ndarray, int, int, int]:
    """Estimate one-step residuals without fitting a separate model at each split.

    ``frozen_prefix`` learns scaling and a conditional mean rule from a declared early
    reference fraction, independent of any candidate or true boundary.  It is the
    default for experiments designed with a stable initial baseline.  ``blocked_crossfit``
    is available when no frozen-reference assumption is defensible; held-out blocks
    are predicted by models that exclude a temporal purge around each block.
    """

    if not 0.10 <= reference_fraction <= 0.45:
        raise ValueError("reference_fraction must lie in [0.10, 0.45]")
    if residual_mode not in {"frozen_prefix", "blocked_crossfit", "pooled"}:
        raise ValueError(
            "residual_mode must be 'frozen_prefix', 'blocked_crossfit', or 'pooled'"
        )
    if crossfit_folds < 2 or crossfit_purge < 0:
        raise ValueError("invalid blocked cross-fit settings")
    if residual_lags < 1:
        raise ValueError("residual_lags must be positive")
    if any(not 0.0 < decay < 1.0 for decay in residual_ewma_decays):
        raise ValueError("residual_ewma_decays must lie strictly between zero and one")

    length, dimension = values.shape
    nonlinear_multiplier = 2 if residual_nonlinear else 1
    feature_dimension = (
        1
        + nonlinear_multiplier * residual_lags * dimension
        + len(residual_ewma_decays) * dimension
    )
    reference_end = max(
        residual_lags + feature_dimension + 4,
        int(math.ceil(length * reference_fraction)),
    )
    reference_end = min(reference_end, length - feature_dimension - 4)
    if reference_end <= residual_lags + feature_dimension:
        raise ValueError("reference prefix is too short for the residual feature map")
    reference = values[:reference_end]
    center = np.mean(reference, axis=0)
    scale = np.std(reference, axis=0, ddof=1)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (values - center) / scale
    response_indices = np.arange(residual_lags, length)
    feature_blocks: list[np.ndarray] = [np.ones((response_indices.size, 1))]
    for lag in range(1, residual_lags + 1):
        feature_blocks.append(standardized[response_indices - lag])
    if residual_nonlinear:
        transformed = np.tanh(values)
        transformed_center = np.mean(np.tanh(reference), axis=0)
        transformed_scale = np.std(np.tanh(reference), axis=0, ddof=1)
        transformed_scale = np.where(transformed_scale > 1e-8, transformed_scale, 1.0)
        transformed = (transformed - transformed_center) / transformed_scale
        for lag in range(1, residual_lags + 1):
            feature_blocks.append(transformed[response_indices - lag])
    for decay in residual_ewma_decays:
        causal_ewma = np.empty_like(values)
        causal_ewma[0] = np.tanh(values[0])
        for time_index in range(1, length):
            causal_ewma[time_index] = (
                decay * causal_ewma[time_index - 1]
                + (1.0 - decay) * np.tanh(values[time_index - 1])
            )
        ewma_reference = causal_ewma[residual_lags:reference_end]
        ewma_center = np.mean(ewma_reference, axis=0)
        ewma_scale = np.std(ewma_reference, axis=0, ddof=1)
        ewma_scale = np.where(ewma_scale > 1e-8, ewma_scale, 1.0)
        feature_blocks.append(
            (causal_ewma[response_indices] - ewma_center) / ewma_scale
        )
    design = np.column_stack(feature_blocks)
    targets = standardized[response_indices]

    if residual_mode == "frozen_prefix":
        train_end = reference_end - residual_lags
        coefficient = _ridge_coefficients(design[:train_end], targets[:train_end], ridge)
        # Prefix training residuals are deliberately excluded from every scan side;
        # otherwise the left side would mix in-sample errors with out-of-sample errors.
        return targets - design @ coefficient, reference_end, train_end, residual_lags
    if residual_mode == "pooled":
        coefficient = _ridge_coefficients(design, targets, ridge)
        return targets - design @ coefficient, reference_end, 0, residual_lags

    residual = np.empty_like(targets)
    row_indices = np.arange(design.shape[0])
    folds = np.array_split(row_indices, min(crossfit_folds, design.shape[0]))
    for held_out in folds:
        if held_out.size == 0:
            continue
        excluded_start = max(0, int(held_out[0]) - crossfit_purge)
        excluded_stop = min(design.shape[0], int(held_out[-1]) + crossfit_purge + 1)
        keep = np.ones(design.shape[0], dtype=bool)
        keep[excluded_start:excluded_stop] = False
        if np.count_nonzero(keep) <= design.shape[1] + 2:
            raise ValueError("too few rows remain after blocked cross-fit purge")
        coefficient = _ridge_coefficients(design[keep], targets[keep], ridge)
        residual[held_out] = targets[held_out] - design[held_out] @ coefficient
    return residual, reference_end, 0, residual_lags


def _variance_reliability(
    values: np.ndarray,
    candidates: np.ndarray,
    *,
    ridge: float,
    residual_mode: str,
    reference_fraction: float,
    crossfit_folds: int,
    crossfit_purge: int,
    residual_lags: int,
    residual_nonlinear: bool,
    residual_ewma_decays: tuple[float, ...],
    variance_estimator: str,
    variance_block_size: int,
    variance_channel_aggregation: str,
    **_: Any,
) -> _ScannerOutput:
    residual, reference_end, scan_start, response_start = _conditional_residuals(
        values,
        ridge=ridge,
        residual_mode=residual_mode,
        reference_fraction=reference_fraction,
        crossfit_folds=crossfit_folds,
        crossfit_purge=crossfit_purge,
        residual_lags=residual_lags,
        residual_nonlinear=residual_nonlinear,
        residual_ewma_decays=residual_ewma_decays,
    )
    scores = np.zeros(candidates.size, dtype=float)
    effects = np.zeros(candidates.size, dtype=float)
    for index, boundary in enumerate(candidates):
        # Response X[t] is row t-1 of the one-step regression.
        split = boundary - response_start
        left, right = residual[scan_start:split], residual[split:]
        if boundary < reference_end + 3 or left.shape[0] < 3 or right.shape[0] < 3:
            continue
        left_variance = _segment_residual_variance(
            left,
            estimator=variance_estimator,
            block_size=variance_block_size,
        )
        right_variance = _segment_residual_variance(
            right,
            estimator=variance_estimator,
            block_size=variance_block_size,
        )
        log_ratio = np.log(left_variance) - np.log(right_variance)
        weight = left.shape[0] * right.shape[0] / (left.shape[0] + right.shape[0])
        if variance_channel_aggregation == "max":
            scores[index] = weight * float(np.max(log_ratio**2))
            effects[index] = float(np.max(np.abs(log_ratio)))
        elif variance_channel_aggregation == "sum":
            scores[index] = weight * float(np.sum(log_ratio**2))
            effects[index] = float(np.sqrt(np.mean(log_ratio**2)))
        else:
            raise ValueError("variance_channel_aggregation must be 'max' or 'sum'")
    return _ScannerOutput(
        scores,
        effects,
        (
            "max_abs_log_residual_variance_ratio"
            if variance_channel_aggregation == "max"
            else "rms_log_residual_variance_ratio"
        ),
        {
            "residual_mode": residual_mode,
            "reference_end": reference_end,
            "scan_residual_start": scan_start,
            "response_start": response_start,
            "residual_lags": residual_lags,
            "residual_nonlinear": residual_nonlinear,
            "residual_ewma_decays": residual_ewma_decays,
            "variance_estimator": variance_estimator,
            "variance_block_size": variance_block_size,
            "variance_channel_aggregation": variance_channel_aggregation,
        },
    )


def _segment_residual_variance(
    residual: np.ndarray,
    *,
    estimator: str,
    block_size: int,
) -> np.ndarray:
    """Estimate residual variance with an optional finite-variance robustification."""

    if estimator not in {"sample", "block_median"}:
        raise ValueError("variance_estimator must be 'sample' or 'block_median'")
    if block_size < 8:
        raise ValueError("variance_block_size must be at least eight")
    centered = residual - np.mean(residual, axis=0)
    if estimator == "sample" or residual.shape[0] < 3 * block_size:
        return np.mean(centered**2, axis=0) + 1e-8
    n_blocks = residual.shape[0] // block_size
    blocks = np.array_split(centered, n_blocks)
    block_second_moments = np.stack([np.mean(block**2, axis=0) for block in blocks])
    return np.median(block_second_moments, axis=0) + 1e-8


def _tail_shape_features(
    residual: np.ndarray,
    thresholds: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    centered = residual - np.mean(residual, axis=0)
    rms = np.sqrt(np.mean(centered**2, axis=0))
    standardized = centered / np.where(rms > 1e-8, rms, 1.0)
    # Clipping only bounds the influence of a single numerical outlier; values up to
    # eight residual SDs retain substantially more tail information than Gaussian data.
    fourth = np.mean(np.clip(standardized, -8.0, 8.0) ** 4, axis=0)
    exceedance = np.stack(
        [np.mean(np.abs(standardized) > threshold, axis=0) for threshold in thresholds]
    )
    return fourth, exceedance


def _residual_tail_shape(
    values: np.ndarray,
    candidates: np.ndarray,
    *,
    ridge: float,
    residual_mode: str,
    reference_fraction: float,
    crossfit_folds: int,
    crossfit_purge: int,
    residual_lags: int,
    residual_nonlinear: bool,
    residual_ewma_decays: tuple[float, ...],
    tail_thresholds: tuple[float, ...],
    **_: Any,
) -> _ScannerOutput:
    if not tail_thresholds or any(value <= 1.0 for value in tail_thresholds):
        raise ValueError("tail_thresholds must contain values greater than one")
    if tuple(sorted(set(tail_thresholds))) != tuple(tail_thresholds):
        raise ValueError("tail_thresholds must be strictly increasing")
    residual, reference_end, scan_start, response_start = _conditional_residuals(
        values,
        ridge=ridge,
        residual_mode=residual_mode,
        reference_fraction=reference_fraction,
        crossfit_folds=crossfit_folds,
        crossfit_purge=crossfit_purge,
        residual_lags=residual_lags,
        residual_nonlinear=residual_nonlinear,
        residual_ewma_decays=residual_ewma_decays,
    )
    scores = np.zeros(candidates.size, dtype=float)
    effects = np.zeros(candidates.size, dtype=float)
    for index, boundary in enumerate(candidates):
        split = boundary - response_start
        left, right = residual[scan_start:split], residual[split:]
        if boundary < reference_end + 3 or left.shape[0] < 8 or right.shape[0] < 8:
            continue
        left_fourth, left_exceedance = _tail_shape_features(left, tail_thresholds)
        right_fourth, right_exceedance = _tail_shape_features(right, tail_thresholds)
        effective_n = left.shape[0] * right.shape[0] / (left.shape[0] + right.shape[0])

        # For a Gaussian reference, sample kurtosis has variance about 24/n.  The
        # empirical exceedance terms add stabilized two-sample Bernoulli contrasts.
        log_fourth_ratio = np.log(right_fourth + 1e-8) - np.log(left_fourth + 1e-8)
        score = effective_n * float(np.sum(log_fourth_ratio**2)) / 24.0
        for threshold_index in range(len(tail_thresholds)):
            left_probability = left_exceedance[threshold_index]
            right_probability = right_exceedance[threshold_index]
            pooled = (
                left.shape[0] * left_probability + right.shape[0] * right_probability
            ) / (left.shape[0] + right.shape[0])
            sampling_variance = (
                pooled
                * (1.0 - pooled)
                * (1.0 / left.shape[0] + 1.0 / right.shape[0])
            )
            score += float(
                np.sum((right_probability - left_probability) ** 2 / (sampling_variance + 1e-6))
            )
        scores[index] = score
        effects[index] = float(np.sqrt(np.mean((right_fourth - left_fourth) ** 2)))
    return _ScannerOutput(
        scores,
        effects,
        "rms_winsorized_residual_fourth_moment_change",
        {
            "residual_mode": residual_mode,
            "reference_end": reference_end,
            "scan_residual_start": scan_start,
            "response_start": response_start,
            "residual_lags": residual_lags,
            "residual_nonlinear": residual_nonlinear,
            "residual_ewma_decays": residual_ewma_decays,
            "tail_thresholds": tail_thresholds,
            "segmentwise_self_normalized": True,
        },
    )


def _conditional_rule_chow(
    values: np.ndarray,
    candidates: np.ndarray,
    *,
    ridge: float,
    **_: Any,
) -> _ScannerOutput:
    design, targets = _transition_design(values)
    pooled_sse = _ridge_sse(design, targets, ridge)
    scores = np.zeros(candidates.size, dtype=float)
    min_rows = design.shape[1] + 2
    for index, boundary in enumerate(candidates):
        split = boundary - 1
        if split < min_rows or design.shape[0] - split < min_rows:
            continue
        split_sse = _ridge_sse(design[:split], targets[:split], ridge)
        split_sse += _ridge_sse(design[split:], targets[split:], ridge)
        scores[index] = max(0.0, pooled_sse - split_sse) / max(pooled_sse, 1e-12)
    return _ScannerOutput(scores, scores.copy(), "relative_split_ridge_sse_gain", {})


def _energy_distance(
    values: np.ndarray,
    candidates: np.ndarray,
    *,
    scan_window: int,
    **_: Any,
) -> _ScannerOutput:
    standardized = _standardize(values)
    scores = np.empty(candidates.size, dtype=float)
    for index, boundary in enumerate(candidates):
        width = min(scan_window, boundary, standardized.shape[0] - boundary)
        left = standardized[boundary - width : boundary]
        right = standardized[boundary : boundary + width]
        between = cdist(left, right).mean()
        within_left = cdist(left, left).mean()
        within_right = cdist(right, right).mean()
        scores[index] = max(0.0, 2.0 * between - within_left - within_right)
    return _ScannerOutput(scores, scores.copy(), "standardized_energy_distance", {})


def _median_bandwidth(values: np.ndarray) -> float:
    if values.shape[0] > 128:
        indices = np.linspace(0, values.shape[0] - 1, 128, dtype=int)
        values = values[indices]
    distances = pdist(values, metric="sqeuclidean")
    positive = distances[distances > 1e-12]
    return float(np.median(positive)) if positive.size else 1.0


def _mmd_rbf(
    values: np.ndarray,
    candidates: np.ndarray,
    *,
    scan_window: int,
    **_: Any,
) -> _ScannerOutput:
    standardized = _standardize(values)
    squared_bandwidth = _median_bandwidth(standardized)
    gamma = 0.5 / max(squared_bandwidth, 1e-12)
    scores = np.empty(candidates.size, dtype=float)
    for index, boundary in enumerate(candidates):
        width = min(scan_window, boundary, standardized.shape[0] - boundary)
        left = standardized[boundary - width : boundary]
        right = standardized[boundary : boundary + width]
        k_left = np.exp(-gamma * cdist(left, left, metric="sqeuclidean"))
        k_right = np.exp(-gamma * cdist(right, right, metric="sqeuclidean"))
        k_between = np.exp(-gamma * cdist(left, right, metric="sqeuclidean"))
        scores[index] = max(
            0.0,
            float(k_left.mean() + k_right.mean() - 2.0 * k_between.mean()),
        )
    return _ScannerOutput(scores, np.sqrt(scores), "standardized_rbf_mmd", {})


_SCANNERS = {
    "mean_cusum": _mean_cusum,
    "variance_reliability": _variance_reliability,
    "conditional_rule_chow": _conditional_rule_chow,
    "residual_tail_shape": _residual_tail_shape,
    "energy_distance": _energy_distance,
    "mmd_rbf": _mmd_rbf,
}


def scan_changepoint(
    series: ChangeSeries | np.ndarray,
    *,
    method: str,
    min_segment: int = 20,
    stride: int = 1,
    scan_window: int = 32,
    ridge: float = 0.1,
    residual_mode: str = "frozen_prefix",
    reference_fraction: float = 0.25,
    crossfit_folds: int = 5,
    crossfit_purge: int = 2,
    tail_thresholds: tuple[float, ...] = (2.0, 3.0),
    variance_estimator: str = "block_median",
    variance_block_size: int = 32,
    variance_channel_aggregation: str = "max",
    residual_lags: int = 1,
    residual_nonlinear: bool = False,
    residual_ewma_decays: tuple[float, ...] = (),
) -> ScanResult:
    """Return a complete offline scan curve and its maximizing split."""

    if method not in _SCANNERS:
        raise ValueError(f"unknown detector {method!r}")
    if scan_window < 3:
        raise ValueError("scan_window must be at least three")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    values = _validate_values(series.values if isinstance(series, ChangeSeries) else series)
    candidates = _candidate_indices(values.shape[0], min_segment, stride)
    output = _SCANNERS[method](
        values,
        candidates,
        scan_window=scan_window,
        ridge=ridge,
        residual_mode=residual_mode,
        reference_fraction=reference_fraction,
        crossfit_folds=crossfit_folds,
        crossfit_purge=crossfit_purge,
        tail_thresholds=tail_thresholds,
        variance_estimator=variance_estimator,
        variance_block_size=variance_block_size,
        variance_channel_aggregation=variance_channel_aggregation,
        residual_lags=residual_lags,
        residual_nonlinear=residual_nonlinear,
        residual_ewma_decays=residual_ewma_decays,
    )
    curve = output.score_curve
    effects = output.effect_curve
    if (
        curve.shape != candidates.shape
        or effects.shape != candidates.shape
        or not np.isfinite(curve).all()
        or not np.isfinite(effects).all()
    ):
        raise RuntimeError(f"detector {method!r} produced an invalid score curve")
    best = int(np.argmax(curve))
    return ScanResult(
        method=method,
        claim_axis=METHOD_CLAIM_AXIS[method],
        detector_channel=METHOD_CHANNEL[method],
        candidate_indices=candidates,
        score_curve=curve,
        effect_curve=effects,
        effect_name=output.effect_name,
        tau_hat=int(candidates[best]),
        max_score=float(curve[best]),
        effect_size=float(effects[best]),
        diagnostic_metadata=dict(output.metadata),
    )


@dataclass(frozen=True)
class NullCalibration:
    """Finite-sample empirical calibration from independent matched null series."""

    method: str
    claim_axis: str
    alpha: float
    threshold: float
    null_max_scores: np.ndarray
    calibration_ids: tuple[str, ...]
    calibration_seeds: tuple[int, ...]
    match_key: str
    scan_options: Mapping[str, Any]

    @property
    def minimum_p_value(self) -> float:
        return 1.0 / (self.null_max_scores.size + 1.0)

    @property
    def attainable_null_call_rate(self) -> float:
        """Expected rank-test size before finite calibration-threshold variation."""

        ranks_called = math.floor(self.alpha * (self.null_max_scores.size + 1) + 1e-12)
        return ranks_called / (self.null_max_scores.size + 1.0)

    def calibration_fpr_interval(
        self,
        *,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Order-statistic uncertainty in the fixed calibration threshold's null FPR.

        This is distinct from the Wilson interval on the final independent null pool.
        For a continuous null score law, the selected upper-tail probability follows a
        Beta distribution induced by the finite calibration order statistic.
        """

        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one")
        total = self.null_max_scores.size + 1
        ranks_called = math.floor(self.alpha * total + 1e-12)
        if ranks_called <= 0:
            return 0.0, 0.0
        lower_probability = (1.0 - confidence) / 2.0
        lower = beta_distribution.ppf(
            lower_probability,
            ranks_called,
            total - ranks_called,
        )
        upper = beta_distribution.ppf(
            1.0 - lower_probability,
            ranks_called,
            total - ranks_called,
        )
        return float(lower), float(upper)

    def p_value(self, score: float) -> float:
        exceedances = int(np.count_nonzero(self.null_max_scores >= score))
        return (1.0 + exceedances) / (self.null_max_scores.size + 1.0)


def calibrate_matched_nulls(
    null_series: Sequence[ChangeSeries],
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    alpha: float = 0.05,
    min_segment: int = 20,
    stride: int = 1,
    scan_window: int = 32,
    ridge: float = 0.1,
    residual_mode: str = "frozen_prefix",
    reference_fraction: float = 0.25,
    crossfit_folds: int = 5,
    crossfit_purge: int = 2,
    tail_thresholds: tuple[float, ...] = (2.0, 3.0),
    variance_estimator: str = "block_median",
    variance_block_size: int = 32,
    variance_channel_aggregation: str = "max",
    residual_lags: int = 1,
    residual_nonlinear: bool = False,
    residual_ewma_decays: tuple[float, ...] = (),
) -> dict[str, NullCalibration]:
    """Calibrate maximum scan statistics using no changed trajectories.

    The empirical p-value is ``(1 + # null maxima >= score) / (B + 1)``.  This
    preserves the split-search multiplicity because every calibration replicate is
    reduced by the same maximum-over-boundaries operation as the test trajectory.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not null_series:
        raise ValueError("at least one independent null series is required")
    ids = tuple(item.series_id for item in null_series)
    if len(set(ids)) != len(ids):
        raise ValueError("calibration series IDs must be unique")
    seeds = tuple(item.simulation_seed for item in null_series)
    if len(set(seeds)) != len(seeds):
        raise ValueError("calibration simulation seeds must be unique")
    if any(item.scenario != "null" or item.boundary is not None for item in null_series):
        raise ValueError("calibration accepts only no-change series")
    match_keys = {item.match_key for item in null_series}
    if len(match_keys) != 1:
        raise ValueError("all calibration series must share one matched-system key")
    methods = tuple(methods)
    if not methods:
        raise ValueError("at least one detector method is required")
    unknown = set(methods) - set(_SCANNERS)
    if unknown:
        raise ValueError(f"unknown detectors: {sorted(unknown)}")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must be unique")

    options = {
        "min_segment": int(min_segment),
        "stride": int(stride),
        "scan_window": int(scan_window),
        "ridge": float(ridge),
        "residual_mode": residual_mode,
        "reference_fraction": float(reference_fraction),
        "crossfit_folds": int(crossfit_folds),
        "crossfit_purge": int(crossfit_purge),
        "tail_thresholds": tuple(float(value) for value in tail_thresholds),
        "variance_estimator": variance_estimator,
        "variance_block_size": int(variance_block_size),
        "variance_channel_aggregation": variance_channel_aggregation,
        "residual_lags": int(residual_lags),
        "residual_nonlinear": bool(residual_nonlinear),
        "residual_ewma_decays": tuple(float(value) for value in residual_ewma_decays),
    }
    calibrations: dict[str, NullCalibration] = {}
    for method in methods:
        maxima = np.asarray(
            [
                scan_changepoint(item, method=method, **options).max_score
                for item in null_series
            ],
            dtype=float,
        )
        threshold = float(np.quantile(maxima, 1.0 - alpha, method="higher"))
        calibrations[method] = NullCalibration(
            method=method,
            claim_axis=METHOD_CLAIM_AXIS[method],
            alpha=float(alpha),
            threshold=threshold,
            null_max_scores=maxima,
            calibration_ids=ids,
            calibration_seeds=seeds,
            match_key=next(iter(match_keys)),
            scan_options=dict(options),
        )
    return calibrations


@dataclass(frozen=True)
class DetectionResult:
    series_id: str
    scenario: str
    truth_channel: str | None
    method: str
    claim_axis: str
    detector_channel: str
    tau_hat: int
    max_score: float
    effect_name: str
    effect_size: float
    effect_size_at_truth: float | None
    score_at_truth: float | None
    threshold: float
    p_value: float
    called: bool
    false_positive: bool | None
    hit: bool | None
    detected_within: bool | None
    localization_delay: int | None
    detected_delay: int | None
    tolerance: int


def detect_calibrated(
    series: ChangeSeries,
    calibration: NullCalibration,
    *,
    tolerance: int = 10,
) -> DetectionResult:
    """Apply one calibrated detector and keep all evaluation axes separate."""

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if series.match_key != calibration.match_key:
        raise ValueError("test trajectory does not match the calibrated system/view")
    if series.series_id in calibration.calibration_ids:
        raise ValueError("a calibration trajectory cannot be reused for evaluation")
    scan = scan_changepoint(
        series,
        method=calibration.method,
        **dict(calibration.scan_options),
    )
    p_value = calibration.p_value(scan.max_score)
    called = bool(p_value <= calibration.alpha)
    if series.boundary is None:
        false_positive: bool | None = called
        hit = None
        detected_within = None
        localization_delay = None
        detected_delay = None
        effect_size_at_truth = None
        score_at_truth = None
    else:
        delay = int(scan.tau_hat - series.boundary)
        hit = bool(abs(delay) <= tolerance)
        detected_within = bool(called and hit)
        false_positive = None
        localization_delay = delay
        detected_delay = delay if called else None
        truth_index = int(np.argmin(np.abs(scan.candidate_indices - series.boundary)))
        effect_size_at_truth = float(scan.effect_curve[truth_index])
        score_at_truth = float(scan.score_curve[truth_index])
    return DetectionResult(
        series_id=series.series_id,
        scenario=series.scenario,
        truth_channel=TRUTH_CHANNEL[series.scenario],
        method=scan.method,
        claim_axis=scan.claim_axis,
        detector_channel=scan.detector_channel,
        tau_hat=scan.tau_hat,
        max_score=scan.max_score,
        effect_name=scan.effect_name,
        effect_size=scan.effect_size,
        effect_size_at_truth=effect_size_at_truth,
        score_at_truth=score_at_truth,
        threshold=calibration.threshold,
        p_value=float(p_value),
        called=called,
        false_positive=false_positive,
        hit=hit,
        detected_within=detected_within,
        localization_delay=localization_delay,
        detected_delay=detected_delay,
        tolerance=int(tolerance),
    )


@dataclass(frozen=True)
class AttributionResult:
    series_id: str
    scenario: str
    truth_channel: str | None
    attributed_channel: str | None
    called: bool
    correct: bool | None
    evidence_p_values: Mapping[str, float]


def infer_event_channel(results: Sequence[DetectionResult]) -> AttributionResult:
    """Infer a mechanism channel from a complete detector panel.

    Only the mean, dispersion, and residual-tail probes can make channel-specific
    attributions.  A broad energy/MMD call alone remains ``general_law``; it is no
    longer mislabeled as tail evidence.  No true boundary or scenario label enters
    this decision.
    """

    if not results:
        raise ValueError("attribution requires at least one detector result")
    series_ids = {item.series_id for item in results}
    scenarios = {item.scenario for item in results}
    if len(series_ids) != 1 or len(scenarios) != 1:
        raise ValueError("all attribution inputs must describe one event")
    by_channel: dict[str, list[DetectionResult]] = {
        "mean": [],
        "dispersion": [],
        "tail_shape": [],
        "general_law": [],
    }
    for item in results:
        by_channel[item.detector_channel].append(item)

    evidence = {
        channel: min((item.p_value for item in items), default=1.0)
        for channel, items in by_channel.items()
    }
    called_channels = {
        channel
        for channel, items in by_channel.items()
        if any(item.called for item in items)
    }
    specific = called_channels & {"mean", "dispersion", "tail_shape"}
    if specific:
        # Tail wins an exact discrete-p tie because it is the narrowest claim; this
        # tie-break does not inspect the ground-truth mechanism.
        priority = {"tail_shape": 0, "dispersion": 1, "mean": 2}
        attributed = min(
            specific,
            key=lambda channel: (evidence[channel], priority[channel]),
        )
    elif "general_law" in called_channels:
        attributed = "general_law"
    else:
        attributed = None

    scenario = next(iter(scenarios))
    truth = TRUTH_CHANNEL[scenario]
    correct = None if truth is None or attributed is None else bool(attributed == truth)
    return AttributionResult(
        series_id=next(iter(series_ids)),
        scenario=scenario,
        truth_channel=truth,
        attributed_channel=attributed,
        called=bool(called_channels),
        correct=correct,
        evidence_p_values=evidence,
    )


def _optional_rate(values: Sequence[bool | None]) -> float:
    observed = [float(value) for value in values if value is not None]
    return float(np.mean(observed)) if observed else math.nan


def _wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson binomial interval, including well-behaved zero/one-count cases."""

    if total <= 0:
        return math.nan, math.nan
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _optional_interval(
    values: Sequence[bool | None],
    *,
    confidence: float,
) -> tuple[float, float]:
    observed = [bool(value) for value in values if value is not None]
    return _wilson_interval(sum(observed), len(observed), confidence=confidence)


def summarize_detections(
    results: Sequence[DetectionResult],
    *,
    confidence: float = 0.95,
) -> tuple[dict[str, Any], ...]:
    """Summarize call/FPR, localization hit, detected-within, and delay separately."""

    groups: dict[tuple[str, str], list[DetectionResult]] = {}
    for result in results:
        groups.setdefault((result.scenario, result.method), []).append(result)
    rows: list[dict[str, Any]] = []
    for (scenario, method), items in sorted(groups.items()):
        localization = [
            item.localization_delay
            for item in items
            if item.localization_delay is not None
        ]
        detected = [item.detected_delay for item in items if item.detected_delay is not None]
        calls = [item.called for item in items]
        call_count = int(sum(calls))
        call_rate = float(np.mean(calls))
        call_low, call_high = _wilson_interval(
            call_count,
            len(items),
            confidence=confidence,
        )
        hit_low, hit_high = _optional_interval(
            [item.hit for item in items], confidence=confidence
        )
        detected_low, detected_high = _optional_interval(
            [item.detected_within for item in items], confidence=confidence
        )
        truth_effects = [
            item.effect_size_at_truth
            for item in items
            if item.effect_size_at_truth is not None
        ]
        selected_effects = [item.effect_size for item in items]
        threshold_ratios = [
            item.max_score / item.threshold
            for item in items
            if item.threshold > 0
        ]
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "claim_axis": items[0].claim_axis,
                "n_events": len(items),
                "call_count": call_count,
                "call_rate": call_rate,
                "call_rate_ci_low": call_low,
                "call_rate_ci_high": call_high,
                "false_positive_rate": call_rate if scenario == "null" else math.nan,
                "false_positive_rate_ci_low": (
                    call_low if scenario == "null" else math.nan
                ),
                "false_positive_rate_ci_high": (
                    call_high if scenario == "null" else math.nan
                ),
                "hit_rate": _optional_rate([item.hit for item in items]),
                "hit_rate_ci_low": hit_low,
                "hit_rate_ci_high": hit_high,
                "detected_within_rate": _optional_rate(
                    [item.detected_within for item in items]
                ),
                "detected_within_rate_ci_low": detected_low,
                "detected_within_rate_ci_high": detected_high,
                "mean_absolute_localization_error": (
                    float(np.mean(np.abs(localization))) if localization else math.nan
                ),
                "median_localization_delay": (
                    float(np.median(localization)) if localization else math.nan
                ),
                "median_detected_delay": float(np.median(detected)) if detected else math.nan,
                "effect_name": items[0].effect_name,
                "median_selected_effect_size": float(np.median(selected_effects)),
                "median_effect_size_at_truth": (
                    float(np.median(truth_effects)) if truth_effects else math.nan
                ),
                "median_max_score_to_null_threshold": (
                    float(np.median(threshold_ratios)) if threshold_ratios else math.nan
                ),
                "median_p_value": float(np.median([item.p_value for item in items])),
            }
        )
    return tuple(rows)


def summarize_attributions(
    attributions: Sequence[AttributionResult],
    *,
    confidence: float = 0.95,
) -> tuple[dict[str, Any], ...]:
    groups: dict[str, list[AttributionResult]] = {}
    for result in attributions:
        groups.setdefault(result.scenario, []).append(result)
    rows: list[dict[str, Any]] = []
    for scenario, items in sorted(groups.items()):
        judged = [item.correct for item in items if item.correct is not None]
        calls = [item.called for item in items]
        call_low, call_high = _wilson_interval(
            sum(calls), len(calls), confidence=confidence
        )
        accuracy_low, accuracy_high = _wilson_interval(
            sum(judged), len(judged), confidence=confidence
        )
        rows.append(
            {
                "scenario": scenario,
                "n_events": len(items),
                "attribution_call_rate": float(np.mean([item.called for item in items])),
                "attribution_call_rate_ci_low": call_low,
                "attribution_call_rate_ci_high": call_high,
                "channel_accuracy_when_attributed": (
                    float(np.mean(judged)) if judged else math.nan
                ),
                "channel_accuracy_ci_low": accuracy_low,
                "channel_accuracy_ci_high": accuracy_high,
            }
        )
    return tuple(rows)


@dataclass(frozen=True)
class ChangepointBenchmarkReport:
    detections: tuple[DetectionResult, ...]
    attributions: tuple[AttributionResult, ...]
    detection_summary: tuple[dict[str, Any], ...]
    attribution_summary: tuple[dict[str, Any], ...]
    calibrations: Mapping[str, NullCalibration]
    metadata: Mapping[str, Any]


def run_changepoint_benchmark(
    config: MechanisticConfig | None = None,
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    n_calibration: int = 39,
    n_null_test: int = 8,
    n_changed: int = 8,
    alpha: float = 0.05,
    boundary_fraction: float = 0.5,
    tolerance: int = 10,
    observation: str = "latent_neural",
    seed: int = 47_000,
    min_segment: int = 20,
    stride: int = 1,
    scan_window: int = 32,
    ridge: float = 0.1,
    residual_mode: str = "frozen_prefix",
    reference_fraction: float = 0.25,
    crossfit_folds: int = 5,
    crossfit_purge: int = 2,
    tail_thresholds: tuple[float, ...] = (2.0, 3.0),
    variance_estimator: str = "block_median",
    variance_block_size: int = 32,
    variance_channel_aggregation: str = "max",
    residual_lags: int = 1,
    residual_nonlinear: bool = False,
    residual_ewma_decays: tuple[float, ...] = (),
    post_change_strengths: Mapping[str, PostChangeStrengths] | None = None,
    summary_confidence: float = 0.95,
) -> ChangepointBenchmarkReport:
    """Run a resource-light matched changepoint panel.

    Calibration, no-change evaluation, and changed evaluation occupy disjoint seed and
    ID namespaces.  One parameter realization is shared by all trajectories, so this
    estimates conditional detection behavior for a fixed biological system rather
    than conflating system heterogeneity with changepoint power.
    """

    if n_calibration < 1 or n_null_test < 1 or n_changed < 1:
        raise ValueError("all replicate counts must be positive")
    if not 0.0 < boundary_fraction < 1.0:
        raise ValueError("boundary_fraction must lie strictly between zero and one")
    methods = tuple(methods)
    strengths_by_scenario = dict(DEFAULT_POST_CHANGE_STRENGTHS)
    if post_change_strengths is not None:
        unknown_strengths = set(post_change_strengths) - {
            "mean",
            "dispersion",
            "matched_tail",
        }
        if unknown_strengths:
            raise ValueError(
                f"post-change strengths have unknown scenarios: {sorted(unknown_strengths)}"
            )
        strengths_by_scenario.update(post_change_strengths)
    for strengths in strengths_by_scenario.values():
        strengths.validate()
    base = with_mechanism(config or MechanisticConfig(mechanism="null"), "null")
    params = generate_mechanistic_parameters(base)
    boundary = int(round(base.n_steps * boundary_fraction))
    if boundary < min_segment or base.n_steps - boundary < min_segment:
        raise ValueError("the requested boundary violates min_segment")
    if (
        {"variance_reliability", "residual_tail_shape"} & set(methods)
        and residual_mode == "frozen_prefix"
        and boundary <= math.ceil(base.n_steps * reference_fraction) + 3
    ):
        raise ValueError("the designed boundary overlaps the frozen reference prefix")

    calibration_series = simulate_matched_nulls(
        base,
        n_series=n_calibration,
        params=params,
        first_seed=seed,
        observation=observation,
        id_prefix="calibration-null",
    )
    calibrations = calibrate_matched_nulls(
        calibration_series,
        methods=methods,
        alpha=alpha,
        min_segment=min_segment,
        stride=stride,
        scan_window=scan_window,
        ridge=ridge,
        residual_mode=residual_mode,
        reference_fraction=reference_fraction,
        crossfit_folds=crossfit_folds,
        crossfit_purge=crossfit_purge,
        tail_thresholds=tail_thresholds,
        variance_estimator=variance_estimator,
        variance_block_size=variance_block_size,
        variance_channel_aggregation=variance_channel_aggregation,
        residual_lags=residual_lags,
        residual_nonlinear=residual_nonlinear,
        residual_ewma_decays=residual_ewma_decays,
    )

    evaluation: list[ChangeSeries] = list(
        simulate_matched_nulls(
            base,
            n_series=n_null_test,
            params=params,
            first_seed=seed + 100_000,
            observation=observation,
            id_prefix="evaluation-null",
        )
    )
    for scenario_index, scenario in enumerate(("mean", "dispersion", "matched_tail")):
        for replicate in range(n_changed):
            evaluation.append(
                simulate_changepoint_series(
                    base,
                    scenario=scenario,
                    boundary=boundary,
                    params=params,
                    simulation_seed=seed + 200_000 + 10_000 * scenario_index + replicate,
                    observation=observation,
                    series_id=f"evaluation-{scenario}:{replicate}",
                    post_change_strengths=strengths_by_scenario[scenario],
                )
            )

    detections: list[DetectionResult] = []
    attributions: list[AttributionResult] = []
    for event in evaluation:
        event_results = [
            detect_calibrated(event, calibrations[method], tolerance=tolerance)
            for method in methods
        ]
        detections.extend(event_results)
        attributions.append(infer_event_channel(event_results))

    metadata = {
        "boundary_name": "neuromodulator_onset",
        "boundary": boundary,
        "observation": observation,
        "parameter_digest": _parameter_digest(params),
        "calibration_ids": tuple(item.series_id for item in calibration_series),
        "evaluation_ids": tuple(item.series_id for item in evaluation),
        "calibration_seed_range": (seed, seed + n_calibration - 1),
        "minimum_attainable_p_value": 1.0 / (n_calibration + 1.0),
        "post_change_strengths": {
            scenario: asdict(strengths)
            for scenario, strengths in strengths_by_scenario.items()
        },
        "scan_options": dict(next(iter(calibrations.values())).scan_options),
        "summary_confidence": summary_confidence,
    }
    frozen_detections = tuple(detections)
    frozen_attributions = tuple(attributions)
    return ChangepointBenchmarkReport(
        detections=frozen_detections,
        attributions=frozen_attributions,
        detection_summary=summarize_detections(
            frozen_detections,
            confidence=summary_confidence,
        ),
        attribution_summary=summarize_attributions(
            frozen_attributions,
            confidence=summary_confidence,
        ),
        calibrations=calibrations,
        metadata=metadata,
    )


__all__ = [
    "AttributionResult",
    "ChangeSeries",
    "ChangepointBenchmarkReport",
    "DEFAULT_METHODS",
    "DEFAULT_POST_CHANGE_STRENGTHS",
    "DetectionResult",
    "METHOD_CLAIM_AXIS",
    "NullCalibration",
    "PostChangeStrengths",
    "SCENARIO_TO_MECHANISM",
    "ScanResult",
    "TRUTH_CHANNEL",
    "calibrate_matched_nulls",
    "detect_calibrated",
    "infer_event_channel",
    "run_changepoint_benchmark",
    "scan_changepoint",
    "simulate_changepoint_series",
    "simulate_matched_nulls",
    "summarize_attributions",
    "summarize_detections",
]
