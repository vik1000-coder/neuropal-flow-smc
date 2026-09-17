from __future__ import annotations

import numpy as np
import torch
from scipy.special import logsumexp

from compatibility_neural_benchmark.core import (
    ENDPOINT_SD_FLOOR,
    GeneratorAdapter,
    RepairedResponseConfig,
    _chunked_standardized_next,
    _endpoint_log_sd_response,
    _equal_weight_endpoint_wasserstein1,
    _next_stimulus_tensor,
    causal_fill,
    systematic_resample,
    systematic_resample_to_n,
)


def _normalized(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    normalizer = float(logsumexp(log_weights))
    return np.exp(log_weights - normalizer), normalizer


def _ess(weights: np.ndarray) -> float:
    return float(1.0 / np.square(weights).sum())


def _largest_beta_for_ess(
    log_weights: np.ndarray,
    energy: np.ndarray,
    beta_lo: float,
    beta_hi: float,
    target_ess: float,
    *,
    iterations: int = 32,
) -> float:
    """Largest tempering value whose incremental weights retain target ESS."""
    if beta_hi <= beta_lo:
        return beta_hi
    high_weights, _ = _normalized(log_weights + (beta_hi - beta_lo) * energy)
    if _ess(high_weights) >= target_ess:
        return beta_hi
    lo, hi = beta_lo, beta_hi
    for _ in range(iterations):
        middle = 0.5 * (lo + hi)
        weights, _ = _normalized(log_weights + (middle - beta_lo) * energy)
        if _ess(weights) >= target_ess:
            lo = middle
        else:
            hi = middle
    return lo


def _repeat_particle_axis(values: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat particles while preserving group and within-parent order."""
    shape = values.shape
    return (
        values[:, :, None]
        .expand(shape[0], shape[1], repeats, *shape[2:])
        .reshape(shape[0], shape[1] * repeats, *shape[2:])
        .contiguous()
    )


def _paired_future_responses(
    adapter: GeneratorAdapter,
    history: torch.Tensor,
    stim_history: torch.Tensor,
    stimulus: np.ndarray,
    *,
    cut_time: int,
    selected_source_count: int,
    target_count: int,
    thresholds: np.ndarray,
    config: RepairedResponseConfig,
    seed: int,
    future_branch_factor: int,
    particle_target_indices: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Unchanged low/high free rollout used by the full-matrix estimator.

    ``particle_target_indices`` is an opt-in audit path.  When supplied, the
    endpoint particle cloud is retained only for those target coordinates.
    Production callers leave it as ``None`` and therefore keep the historical
    response dictionary and memory footprint unchanged.
    """
    if future_branch_factor > 1:
        history = _repeat_particle_axis(history, future_branch_factor)
        stim_history = _repeat_particle_axis(stim_history, future_branch_factor)
    n_future = config.n_particles * future_branch_factor
    groups = 2 * selected_source_count
    max_horizon = max(config.horizon_frames)
    modulus = 2**31 - 1
    horizon_frames = np.asarray(config.horizon_frames, dtype=np.int64)
    horizon_to_index = {int(value): i for i, value in enumerate(horizon_frames)}
    selected_targets: np.ndarray | None = None
    endpoint_particles: np.ndarray | None = None
    if particle_target_indices is not None:
        selected_targets = np.asarray(particle_target_indices, dtype=np.int64)
        if selected_targets.ndim != 1 or not len(selected_targets):
            raise ValueError(
                "particle_target_indices must be a nonempty one-dimensional array"
            )
        if np.any(selected_targets < 0) or np.any(selected_targets >= target_count):
            raise ValueError("particle_target_indices contains an out-of-range target")
        if len(np.unique(selected_targets)) != len(selected_targets):
            raise ValueError("particle_target_indices must not contain duplicates")
        endpoint_particles = np.empty(
            (
                2,
                selected_source_count,
                len(horizon_frames),
                n_future,
                len(selected_targets),
            ),
            dtype=np.float32,
        )
    feature_store = {
        key: np.empty(
            (groups, len(horizon_frames), target_count), dtype=np.float64
        )
        for key in (
            "endpoint_mean",
            "cumulative_mean",
            "peak_mean",
            "event_probability",
            "endpoint_sd",
        )
    }
    endpoint_wasserstein1 = np.empty(
        (selected_source_count, len(horizon_frames), target_count),
        dtype=np.float64,
    )
    cumulative = np.zeros((groups, n_future, target_count), dtype=np.float64)
    peak = np.full((groups, n_future, target_count), -np.inf, dtype=np.float64)
    for step in range(1, max_horizon + 1):
        absolute_time = cut_time + step
        draw = _chunked_standardized_next(
            adapter,
            history,
            stim_history,
            seed=int((seed + 10_000_019 + 104_729 * step) % modulus),
            chunk_size=config.sampling_chunk_size,
            paired_halves=True,
        )
        draw_array = draw.detach().cpu().numpy().astype(np.float64, copy=False)
        cumulative += draw_array
        peak = np.maximum(peak, draw_array)
        if step in horizon_to_index:
            index = horizon_to_index[step]
            feature_store["endpoint_mean"][:, index] = draw_array.mean(axis=1)
            feature_store["cumulative_mean"][:, index] = (
                cumulative / step
            ).mean(axis=1)
            feature_store["peak_mean"][:, index] = peak.mean(axis=1)
            feature_store["event_probability"][:, index] = (
                peak > np.asarray(thresholds)[None, None]
            ).mean(axis=1)
            feature_store["endpoint_sd"][:, index] = draw_array.std(axis=1, ddof=0)
            endpoint_wasserstein1[:, index] = _equal_weight_endpoint_wasserstein1(
                draw_array[:selected_source_count],
                draw_array[selected_source_count:],
            )
            if endpoint_particles is not None and selected_targets is not None:
                endpoint_particles[:, :, index] = draw_array.reshape(
                    2, selected_source_count, n_future, target_count
                )[..., selected_targets]

        next_stimulus = _next_stimulus_tensor(
            stimulus[absolute_time], (groups, n_future), device=adapter.device
        )
        history = torch.cat([history[:, :, 1:], draw[:, :, None]], dim=2)
        stim_history = torch.cat(
            [stim_history[:, :, 1:], next_stimulus], dim=2
        )

    m = selected_source_count
    response = {
        f"response_{key}": (value[m:] - value[:m]).astype(np.float32)
        for key, value in feature_store.items()
    }
    response["response_endpoint_log_sd"] = _endpoint_log_sd_response(
        feature_store["endpoint_sd"][:m], feature_store["endpoint_sd"][m:]
    )
    response["response_endpoint_wasserstein1"] = endpoint_wasserstein1.astype(
        np.float32
    )
    if endpoint_particles is not None and selected_targets is not None:
        response["particle_endpoint"] = endpoint_particles
        response["particle_target_index"] = selected_targets.astype(np.int16)
        response["particle_future_parent_index"] = np.repeat(
            np.arange(config.n_particles, dtype=np.int16), future_branch_factor
        )
    return response, n_future


def _three_arm_future_responses(
    adapter: GeneratorAdapter,
    repaired_history: torch.Tensor,
    repaired_stimulus_history: torch.Tensor,
    factual_history: np.ndarray,
    factual_stimulus_history: np.ndarray,
    stimulus: np.ndarray,
    *,
    cut_time: int,
    selected_source_count: int,
    target_count: int,
    thresholds: np.ndarray,
    config: RepairedResponseConfig,
    seed: int,
    future_branch_factor: int,
    endpoint_quantiles: tuple[float, ...],
) -> tuple[dict[str, np.ndarray], int]:
    """Roll low, high, and factual histories with common base-noise draws.

    The factual arm starts from the observed history at the prediction cut. It
    does not receive a source clamp or repair weight. The low and high arms are
    the exact selected-source progressive-bridge systems produced above.
    """
    m = selected_source_count
    n = int(config.n_particles)
    arms = 3
    low_high_history = repaired_history.reshape(
        2, m, n, *repaired_history.shape[2:]
    )
    low_high_stimulus = repaired_stimulus_history.reshape(
        2, m, n, *repaired_stimulus_history.shape[2:]
    )
    factual_history_tensor = torch.as_tensor(
        np.broadcast_to(
            factual_history,
            (1, m, n, *factual_history.shape),
        ).copy(),
        dtype=torch.float32,
        device=adapter.device,
    )
    factual_stimulus_tensor = torch.as_tensor(
        np.broadcast_to(
            factual_stimulus_history,
            (1, m, n, *factual_stimulus_history.shape),
        ).copy(),
        dtype=torch.float32,
        device=adapter.device,
    )
    history = torch.cat([low_high_history, factual_history_tensor], dim=0)
    stim_history = torch.cat(
        [low_high_stimulus, factual_stimulus_tensor], dim=0
    )

    def repeat_arm_particles(values: torch.Tensor, repeats: int) -> torch.Tensor:
        shape = values.shape
        return (
            values[:, :, :, None]
            .expand(shape[0], shape[1], shape[2], repeats, *shape[3:])
            .reshape(shape[0], shape[1], shape[2] * repeats, *shape[3:])
            .contiguous()
        )

    if future_branch_factor > 1:
        history = repeat_arm_particles(history, future_branch_factor)
        stim_history = repeat_arm_particles(stim_history, future_branch_factor)
    n_future = n * future_branch_factor
    max_horizon = max(config.horizon_frames)
    modulus = 2**31 - 1
    horizons = np.asarray(config.horizon_frames, dtype=np.int64)
    horizon_to_index = {int(value): i for i, value in enumerate(horizons)}
    feature_store = {
        key: np.empty(
            (arms, m, len(horizons), target_count), dtype=np.float64
        )
        for key in (
            "endpoint_mean",
            "time_average_mean",
            "pathwise_peak_mean",
            "crossing_probability",
            "endpoint_sd",
        )
    }
    pair_indices = {
        "high_low": (1, 0),
        "high_factual": (1, 2),
        "low_factual": (0, 2),
    }
    wasserstein = {
        pair: np.empty((m, len(horizons), target_count), dtype=np.float64)
        for pair in pair_indices
    }
    quantiles = np.asarray(endpoint_quantiles, dtype=np.float64)
    quantile_store = (
        np.empty(
            (arms, m, len(horizons), len(quantiles), target_count),
            dtype=np.float64,
        )
        if len(quantiles)
        else None
    )
    cumulative = np.zeros((arms, m, n_future, target_count), dtype=np.float64)
    peak = np.full_like(cumulative, -np.inf)

    for step in range(1, max_horizon + 1):
        absolute_time = cut_time + step
        common_seed = int((seed + 10_000_019 + 104_729 * step) % modulus)
        # Calling the same sampler with the same seed and identically ordered
        # source/particle rows gives all three arms the same base random draws.
        draw = torch.stack(
            [
                _chunked_standardized_next(
                    adapter,
                    history[arm],
                    stim_history[arm],
                    seed=common_seed,
                    chunk_size=config.sampling_chunk_size,
                )
                for arm in range(arms)
            ],
            dim=0,
        )
        draw_array = draw.detach().cpu().numpy().astype(np.float64, copy=False)
        cumulative += draw_array
        peak = np.maximum(peak, draw_array)
        if step in horizon_to_index:
            index = horizon_to_index[step]
            feature_store["endpoint_mean"][:, :, index] = draw_array.mean(axis=2)
            feature_store["time_average_mean"][:, :, index] = (
                cumulative / step
            ).mean(axis=2)
            feature_store["pathwise_peak_mean"][:, :, index] = peak.mean(axis=2)
            feature_store["crossing_probability"][:, :, index] = (
                peak > np.asarray(thresholds)[None, None, None]
            ).mean(axis=2)
            feature_store["endpoint_sd"][:, :, index] = draw_array.std(
                axis=2, ddof=0
            )
            for pair, (left, right) in pair_indices.items():
                wasserstein[pair][:, index] = _equal_weight_endpoint_wasserstein1(
                    draw_array[right], draw_array[left]
                )
            if quantile_store is not None:
                # np.quantile places its quantile axis first.
                current = np.quantile(draw_array, quantiles, axis=2)
                quantile_store[:, :, index] = np.transpose(
                    current, (1, 2, 0, 3)
                )

        next_stimulus = _next_stimulus_tensor(
            stimulus[absolute_time],
            (arms, m, n_future),
            device=adapter.device,
        )
        history = torch.cat([history[:, :, :, 1:], draw[:, :, :, None]], dim=3)
        stim_history = torch.cat(
            [stim_history[:, :, :, 1:], next_stimulus], dim=3
        )

    result = {
        f"arm_{key}": value.astype(np.float32)
        for key, value in feature_store.items()
    }
    for pair, (left, right) in pair_indices.items():
        for key in (
            "endpoint_mean",
            "time_average_mean",
            "pathwise_peak_mean",
            "crossing_probability",
        ):
            result[f"response_{pair}_{key}"] = (
                feature_store[key][left] - feature_store[key][right]
            ).astype(np.float32)
        result[f"response_{pair}_endpoint_log_sd"] = _endpoint_log_sd_response(
            feature_store["endpoint_sd"][right],
            feature_store["endpoint_sd"][left],
        )
        result[f"distance_{pair}_endpoint_wasserstein1"] = wasserstein[pair].astype(
            np.float32
        )
    if quantile_store is not None:
        result["arm_endpoint_quantile"] = quantile_store.astype(np.float32)
        for pair, (left, right) in pair_indices.items():
            result[f"response_{pair}_endpoint_quantile"] = (
                quantile_store[left] - quantile_store[right]
            ).astype(np.float32)
    return result, n_future


def progressive_smc_repaired_responses(
    adapter: GeneratorAdapter,
    standardized_trace: np.ndarray,
    stimulus: np.ndarray,
    *,
    cut_time: int,
    projection: np.ndarray,
    source_low: np.ndarray,
    source_high: np.ndarray,
    source_iqr: np.ndarray,
    thresholds: np.ndarray,
    config: RepairedResponseConfig,
    seed: int,
    branch_factor: int = 2,
    future_branch_factor: int = 2,
    tempering_ess_fraction: float = 0.65,
    max_tempering_resamples: int = 8,
    source_indices: np.ndarray | None = None,
    include_factual_arm: bool = False,
    endpoint_quantiles: tuple[float, ...] = (),
    particle_target_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Progressive sampler-only SMC for the frozen repaired-path law.

    Each repair parent proposes ``branch_factor`` children from the learned
    transition.  During the source window, a persistence look-ahead fills the
    as-yet-unseen source frames and the clamp is tempered from zero to one in
    equal temporal stages.  Adaptive subincrements keep candidate ESS near the
    declared floor, with systematic resampling when a proposed increment is
    too selective.  The last bridge potential is exactly the original terminal
    Gaussian source clamp, so this changes the particle estimator rather than
    the terminal estimand.

    After the terminal repair selection, every repaired particle launches
    ``future_branch_factor`` free descendants.  Averaging those descendants is
    the sampler-only analogue of Rao--Blackwellizing future transition noise;
    no flow log density is required.
    """
    config.validate()
    if adapter.lag != config.history_frames:
        raise ValueError("checkpoint and response history lengths disagree")
    if branch_factor < 1 or future_branch_factor < 1:
        raise ValueError("branch factors must be positive")
    if not (0.0 < tempering_ess_fraction <= 1.0):
        raise ValueError("tempering ESS fraction must lie in (0, 1]")
    if max_tempering_resamples < 1:
        raise ValueError("at least one tempering resample must be allowed")
    quantile_array = np.asarray(endpoint_quantiles, dtype=np.float64)
    if quantile_array.ndim != 1 or np.any(~np.isfinite(quantile_array)):
        raise ValueError("endpoint quantiles must be a finite one-dimensional list")
    if np.any(quantile_array <= 0.0) or np.any(quantile_array >= 1.0):
        raise ValueError("endpoint quantiles must lie strictly between zero and one")
    if len(np.unique(quantile_array)) != len(quantile_array):
        raise ValueError("endpoint quantiles must not contain duplicates")
    if len(quantile_array) and not include_factual_arm:
        raise ValueError("endpoint quantiles are available only with a factual arm")
    if particle_target_indices is not None and include_factual_arm:
        raise ValueError(
            "particle endpoint retention is currently available only for paired "
            "low/high rollouts"
        )

    trace = causal_fill(standardized_trace)
    start = cut_time - config.repair_frames
    history_lo = start - config.history_frames + 1
    max_horizon = max(config.horizon_frames)
    if history_lo < 0 or cut_time + max_horizon >= len(trace):
        raise ValueError("episode lacks boundary history or forecast horizon")
    initial = trace[history_lo : start + 1]
    initial_stimulus = stimulus[history_lo : start + 1]
    if initial.shape != (config.history_frames, trace.shape[1]):
        raise RuntimeError("boundary history alignment failed")

    n = int(config.n_particles)
    d = int(trace.shape[1])
    selected_sources = (
        np.arange(d, dtype=np.int64)
        if source_indices is None
        else np.asarray(source_indices, dtype=np.int64)
    )
    if selected_sources.ndim != 1 or len(selected_sources) < 1:
        raise ValueError("source_indices must be a nonempty one-dimensional array")
    if np.any(selected_sources < 0) or np.any(selected_sources >= d):
        raise ValueError("source_indices contains an out-of-range neuron index")
    if len(np.unique(selected_sources)) != len(selected_sources):
        raise ValueError("source_indices must not contain duplicates")
    m = len(selected_sources)

    def selected_parameter(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape == (d,):
            return array[selected_sources]
        if array.shape == (m,):
            return array
        raise ValueError(f"{name} must contain either all neurons or selected sources")

    low_targets = selected_parameter(source_low, "source_low")
    high_targets = selected_parameter(source_high, "source_high")
    selected_iqr = selected_parameter(source_iqr, "source_iqr")
    groups = 2 * m
    candidates = n * int(branch_factor)
    rank = int(projection.shape[1])
    source_index = np.tile(selected_sources, 2)
    target = np.concatenate([low_targets, high_targets])
    bandwidth = np.maximum(
        selected_iqr * config.epsilon_iqr_fraction, 0.10
    )
    bandwidth = np.tile(bandwidth, 2)
    factual_prefix = trace[start + 1 : cut_time + 1]
    source_end = config.repair_frames - config.source_lag_frames
    source_start = source_end - config.source_window_frames
    rng = np.random.default_rng(int(seed + 73_856_093))
    modulus = 2**31 - 1

    history = torch.as_tensor(
        np.broadcast_to(initial, (groups, n, *initial.shape)).copy(),
        dtype=torch.float32,
        device=adapter.device,
    )
    stim_history = torch.as_tensor(
        np.broadcast_to(initial_stimulus, (groups, n, *initial_stimulus.shape)).copy(),
        dtype=torch.float32,
        device=adapter.device,
    )
    projection_tensor = torch.as_tensor(
        projection, dtype=torch.float32, device=adapter.device
    )
    source_tensor = torch.as_tensor(
        source_index, dtype=torch.long, device=adapter.device
    )

    source_sum = np.zeros((groups, n), dtype=np.float64)
    anchor_cost_total = np.zeros((groups, n), dtype=np.float64)
    previous_energy = np.zeros((groups, n), dtype=np.float64)
    ancestors = np.broadcast_to(np.arange(n, dtype=np.int64), (groups, n)).copy()
    log_normalizer = np.zeros(groups, dtype=np.float64)
    equivalent_ess_steps = np.zeros((groups, config.repair_frames), dtype=np.float64)
    candidate_ess_steps = np.zeros_like(equivalent_ess_steps)
    ess_fraction_steps = np.zeros_like(equivalent_ess_steps)
    equivalent_max_weight_steps = np.zeros_like(equivalent_ess_steps)
    candidate_max_weight_steps = np.zeros_like(equivalent_ess_steps)
    tempering_resamples = np.zeros_like(equivalent_ess_steps, dtype=np.int16)
    forced_tempering = np.zeros_like(equivalent_ess_steps, dtype=np.int8)
    beta_steps = np.zeros_like(equivalent_ess_steps)
    prune_offsets = rng.random((config.repair_frames, m))
    temper_offsets = rng.random((config.repair_frames, max_tempering_resamples, m))

    weighted_source = np.zeros(groups, dtype=np.float64)
    weighted_anchor = np.zeros(groups, dtype=np.float64)
    terminal_equivalent_weights = np.full((groups, candidates), 1.0 / candidates)
    source_count = 0
    previous_beta = 0.0

    def repeat_tensor(values: torch.Tensor, repeats: int) -> torch.Tensor:
        shape = values.shape
        return (
            values[:, :, None]
            .expand(shape[0], shape[1], repeats, *shape[2:])
            .reshape(shape[0], shape[1] * repeats, *shape[2:])
            .contiguous()
        )

    def select_group(group: int, indices: np.ndarray) -> None:
        nonlocal history, stim_history, source_sum, anchor_cost_total, ancestors, previous_energy
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=adapter.device)
        history[group] = history[group, index_tensor].clone()
        stim_history[group] = stim_history[group, index_tensor].clone()
        source_sum[group] = source_sum[group, indices]
        anchor_cost_total[group] = anchor_cost_total[group, indices]
        ancestors[group] = ancestors[group, indices]
        previous_energy[group] = previous_energy[group, indices]

    for step in range(config.repair_frames):
        absolute_time = start + step + 1
        history = repeat_tensor(history, branch_factor)
        stim_history = repeat_tensor(stim_history, branch_factor)
        source_sum = np.repeat(source_sum, branch_factor, axis=1)
        anchor_cost_total = np.repeat(anchor_cost_total, branch_factor, axis=1)
        previous_energy = np.repeat(previous_energy, branch_factor, axis=1)
        ancestors = np.repeat(ancestors, branch_factor, axis=1)

        draw = _chunked_standardized_next(
            adapter,
            history,
            stim_history,
            seed=int((seed + 104_729 * step) % modulus),
            chunk_size=config.sampling_chunk_size,
            paired_halves=True,
        )
        draw_array = draw.detach().cpu().numpy().astype(np.float64, copy=False)
        draw_source = np.take_along_axis(
            draw_array,
            np.broadcast_to(source_index[:, None, None], (groups, candidates, 1)),
            axis=2,
        )[:, :, 0]

        in_source_window = source_start <= step < source_end
        if in_source_window:
            source_count += 1
            source_sum += draw_source
            remaining = config.source_window_frames - source_count
            predicted_source = (
                source_sum + remaining * draw_source
            ) / config.source_window_frames
            current_energy = -0.5 * np.square(
                (predicted_source - target[:, None]) / bandwidth[:, None]
            )
            beta_cap = source_count / config.source_window_frames
        elif step >= source_end:
            predicted_source = source_sum / config.source_window_frames
            current_energy = -0.5 * np.square(
                (predicted_source - target[:, None]) / bandwidth[:, None]
            )
            beta_cap = 1.0
        else:
            predicted_source = np.zeros_like(source_sum)
            current_energy = np.zeros_like(source_sum)
            beta_cap = 0.0

        factual = torch.as_tensor(
            factual_prefix[step], dtype=torch.float32, device=adapter.device
        )
        difference = draw - factual[None, None]
        latent = difference @ projection_tensor
        if in_source_window:
            source_difference = difference.gather(
                2,
                source_tensor[:, None, None].expand(groups, candidates, 1),
            ).squeeze(2)
            source_projection = projection_tensor[source_tensor]
            latent = latent - source_difference[..., None] * source_projection[:, None]
        incremental_anchor = (
            0.5
            * latent.square().sum(dim=-1).detach().cpu().numpy().astype(np.float64)
            / (rank * config.repair_frames)
        )
        anchor_cost_total += incremental_anchor

        next_stimulus = _next_stimulus_tensor(
            stimulus[absolute_time], (groups, candidates), device=adapter.device
        )
        history = torch.cat([history[:, :, 1:], draw[:, :, None]], dim=2)
        stim_history = torch.cat([stim_history[:, :, 1:], next_stimulus], dim=2)

        base_increment = -config.anchor_lambda * incremental_anchor
        if previous_beta > 0.0:
            base_increment += previous_beta * (current_energy - previous_energy)
        log_weights = np.full((groups, candidates), -np.log(candidates), dtype=np.float64)
        log_weights += base_increment
        weights = np.empty_like(log_weights)
        for group in range(groups):
            weights[group], increment = _normalized(log_weights[group])
            log_normalizer[group] += increment
            log_weights[group] = np.log(np.maximum(weights[group], 1e-300))

        for group in range(groups):
            beta = previous_beta
            resample_count = 0
            target_ess = tempering_ess_fraction * candidates
            while beta < beta_cap - 1e-10:
                current_weights = np.exp(log_weights[group])
                if _ess(current_weights) < target_ess - 1e-7:
                    if resample_count >= max_tempering_resamples:
                        forced_tempering[group, step] = 1
                        chosen_beta = beta_cap
                    else:
                        indices = systematic_resample(
                            current_weights,
                            rng,
                            offset=float(
                                temper_offsets[step, resample_count, group % m]
                            ),
                        )
                        select_group(group, indices)
                        current_energy[group] = current_energy[group, indices]
                        log_weights[group] = -np.log(candidates)
                        resample_count += 1
                        continue
                else:
                    chosen_beta = _largest_beta_for_ess(
                        log_weights[group],
                        current_energy[group],
                        beta,
                        beta_cap,
                        target_ess,
                    )
                    if chosen_beta <= beta + 1e-8:
                        if resample_count >= max_tempering_resamples:
                            forced_tempering[group, step] = 1
                            chosen_beta = beta_cap
                        else:
                            indices = systematic_resample(
                                current_weights,
                                rng,
                                offset=float(
                                    temper_offsets[step, resample_count, group % m]
                                ),
                            )
                            select_group(group, indices)
                            current_energy[group] = current_energy[group, indices]
                            log_weights[group] = -np.log(candidates)
                            resample_count += 1
                            continue

                updated = log_weights[group] + (chosen_beta - beta) * current_energy[group]
                weights[group], increment = _normalized(updated)
                log_normalizer[group] += increment
                log_weights[group] = np.log(np.maximum(weights[group], 1e-300))
                beta = chosen_beta
                if beta < beta_cap - 1e-10:
                    if resample_count >= max_tempering_resamples:
                        forced_tempering[group, step] = 1
                        updated = log_weights[group] + (beta_cap - beta) * current_energy[group]
                        weights[group], increment = _normalized(updated)
                        log_normalizer[group] += increment
                        log_weights[group] = np.log(np.maximum(weights[group], 1e-300))
                        beta = beta_cap
                    else:
                        indices = systematic_resample(
                            weights[group],
                            rng,
                            offset=float(
                                temper_offsets[step, resample_count, group % m]
                            ),
                        )
                        select_group(group, indices)
                        current_energy[group] = current_energy[group, indices]
                        log_weights[group] = -np.log(candidates)
                        resample_count += 1

            weights[group] = np.exp(log_weights[group])
            tempering_resamples[group, step] = resample_count
            beta_steps[group, step] = beta_cap

        raw_ess = 1.0 / np.square(weights).sum(axis=1)
        raw_max = weights.max(axis=1)
        candidate_ess_steps[:, step] = raw_ess
        equivalent_ess_steps[:, step] = raw_ess / branch_factor
        ess_fraction_steps[:, step] = raw_ess / candidates
        candidate_max_weight_steps[:, step] = raw_max
        equivalent_max_weight_steps[:, step] = raw_max * branch_factor

        if step == config.repair_frames - 1:
            source_value = source_sum / config.source_window_frames
            weighted_source = np.sum(weights * source_value, axis=1)
            weighted_anchor = np.sum(weights * anchor_cost_total, axis=1)
            terminal_equivalent_weights = weights.copy()

        prune_indices = np.stack(
            [
                systematic_resample_to_n(
                    weights[group],
                    n,
                    rng,
                    offset=float(prune_offsets[step, group % m]),
                )
                for group in range(groups)
            ]
        )
        group_tensor = torch.arange(groups, device=adapter.device)[:, None]
        index_tensor = torch.as_tensor(
            prune_indices, dtype=torch.long, device=adapter.device
        )
        history = history[group_tensor, index_tensor]
        stim_history = stim_history[group_tensor, index_tensor]
        source_sum = np.take_along_axis(source_sum, prune_indices, axis=1)
        anchor_cost_total = np.take_along_axis(anchor_cost_total, prune_indices, axis=1)
        ancestors = np.take_along_axis(ancestors, prune_indices, axis=1)
        previous_energy = np.take_along_axis(current_energy, prune_indices, axis=1)
        previous_beta = beta_cap

    repaired_ancestors = ancestors.copy()
    if include_factual_arm:
        factual_history_lo = cut_time - config.history_frames + 1
        factual_history = trace[factual_history_lo : cut_time + 1]
        factual_stimulus_history = stimulus[factual_history_lo : cut_time + 1]
        if factual_history.shape != (config.history_frames, d):
            raise RuntimeError("factual cut history alignment failed")
        response, n_future = _three_arm_future_responses(
            adapter,
            history,
            stim_history,
            factual_history,
            factual_stimulus_history,
            stimulus,
            cut_time=cut_time,
            selected_source_count=m,
            target_count=d,
            thresholds=thresholds,
            config=config,
            seed=seed,
            future_branch_factor=future_branch_factor,
            endpoint_quantiles=tuple(float(value) for value in quantile_array),
        )
    else:
        response, n_future = _paired_future_responses(
            adapter,
            history,
            stim_history,
            stimulus,
            cut_time=cut_time,
            selected_source_count=m,
            target_count=d,
            thresholds=thresholds,
            config=config,
            seed=seed,
            future_branch_factor=future_branch_factor,
            particle_target_indices=particle_target_indices,
        )
    achieved_low, achieved_high = weighted_source[:m], weighted_source[m:]
    achieved_gap = achieved_high - achieved_low
    target_gap = high_targets - low_targets
    terminal_ess_low = equivalent_ess_steps[:m, -1]
    terminal_ess_high = equivalent_ess_steps[m:, -1]
    terminal_max_low = equivalent_max_weight_steps[:m, -1]
    terminal_max_high = equivalent_max_weight_steps[m:, -1]
    valid = (
        (terminal_ess_low >= config.min_ess)
        & (terminal_ess_high >= config.min_ess)
        & (terminal_max_low <= config.max_normalized_weight)
        & (terminal_max_high <= config.max_normalized_weight)
        & (achieved_gap >= config.min_achieved_fraction * target_gap)
    )
    distinct_ancestors = np.asarray(
        [len(np.unique(repaired_ancestors[group])) for group in range(groups)],
        dtype=np.float64,
    )
    diagnostics = {
        "target_low": low_targets,
        "target_high": high_targets,
        "target_gap": target_gap,
        "achieved_low": achieved_low,
        "achieved_high": achieved_high,
        "achieved_gap": achieved_gap,
        "ess_low": terminal_ess_low,
        "ess_high": terminal_ess_high,
        "candidate_ess_low": candidate_ess_steps[:m, -1],
        "candidate_ess_high": candidate_ess_steps[m:, -1],
        "ess_fraction_low": ess_fraction_steps[:m, -1],
        "ess_fraction_high": ess_fraction_steps[m:, -1],
        "max_weight_low": terminal_max_low,
        "max_weight_high": terminal_max_high,
        "candidate_max_weight_low": candidate_max_weight_steps[:m, -1],
        "candidate_max_weight_high": candidate_max_weight_steps[m:, -1],
        "min_step_ess_low": equivalent_ess_steps[:m].min(axis=1),
        "min_step_ess_high": equivalent_ess_steps[m:].min(axis=1),
        "distinct_ancestors_low": distinct_ancestors[:m],
        "distinct_ancestors_high": distinct_ancestors[m:],
        "anchor_cost_mean_low": weighted_anchor[:m],
        "anchor_cost_mean_high": weighted_anchor[m:],
        "log10_total_low": log_normalizer[:m] / np.log(10.0),
        "log10_total_high": log_normalizer[m:] / np.log(10.0),
        "valid": valid.astype(np.float32),
        "endpoint_sd_floor": np.full(m, ENDPOINT_SD_FLOOR, dtype=np.float64),
        "step_ess_low": equivalent_ess_steps[:m],
        "step_ess_high": equivalent_ess_steps[m:],
        "step_candidate_ess_low": candidate_ess_steps[:m],
        "step_candidate_ess_high": candidate_ess_steps[m:],
        "step_ess_fraction_low": ess_fraction_steps[:m],
        "step_ess_fraction_high": ess_fraction_steps[m:],
        "step_max_weight_low": equivalent_max_weight_steps[:m],
        "step_max_weight_high": equivalent_max_weight_steps[m:],
        "step_candidate_max_weight_low": candidate_max_weight_steps[:m],
        "step_candidate_max_weight_high": candidate_max_weight_steps[m:],
        "step_tempering_resamples_low": tempering_resamples[:m],
        "step_tempering_resamples_high": tempering_resamples[m:],
        "step_forced_tempering_low": forced_tempering[:m],
        "step_forced_tempering_high": forced_tempering[m:],
        "step_beta_low": beta_steps[:m],
        "step_beta_high": beta_steps[m:],
        "branch_factor": np.full(m, branch_factor, dtype=np.float64),
        "future_branch_factor": np.full(m, future_branch_factor, dtype=np.float64),
        "candidate_particles": np.full(m, candidates, dtype=np.float64),
        "future_particles": np.full(m, n_future, dtype=np.float64),
        "source_window_start_step": np.full(m, source_start, dtype=np.float64),
        "source_window_end_step_exclusive": np.full(
            m, source_end, dtype=np.float64
        ),
    }
    if source_indices is not None:
        diagnostics["selected_source_index"] = selected_sources.astype(np.float64)
    if include_factual_arm:
        factual_source_hi = cut_time - config.source_lag_frames + 1
        factual_source_lo = factual_source_hi - config.source_window_frames
        diagnostics["factual_source"] = trace[
            factual_source_lo:factual_source_hi
        ].mean(axis=0)[selected_sources]
    if particle_target_indices is not None:
        response["particle_repair_ancestor_id"] = repaired_ancestors.reshape(
            2, m, n
        ).astype(np.int16)
    return {
        **response,
        **{
            f"diagnostic_{key}": np.asarray(value, dtype=np.float32)
            for key, value in diagnostics.items()
        },
    }


def progressive_smc_targeted_confirmation(
    adapter: GeneratorAdapter,
    standardized_trace: np.ndarray,
    stimulus: np.ndarray,
    *,
    cut_time: int,
    projection: np.ndarray,
    source_low: np.ndarray,
    source_high: np.ndarray,
    source_iqr: np.ndarray,
    thresholds: np.ndarray,
    source_indices: np.ndarray,
    config: RepairedResponseConfig,
    seed: int,
    branch_factor: int = 2,
    future_branch_factor: int = 2,
    tempering_ess_fraction: float = 0.65,
    max_tempering_resamples: int = 8,
    endpoint_quantiles: tuple[float, ...] = (),
) -> dict[str, np.ndarray]:
    """Confirm a strict source subset with low/high/factual future arms.

    Low and high use the same progressive bridge, lagged source window, and
    factual-anchor potential as the full-matrix estimator. The factual arm is
    the learned generator rolled from the observed history at the cut. Future
    base random numbers are matched across all three arms. Returned response
    tensors remain source-major ``[selected_source, horizon, target]``; the
    confirmation archive transposes once to target-row/source-column order.
    """
    selected = np.asarray(source_indices, dtype=np.int64)
    target_count = int(np.asarray(standardized_trace).shape[1])
    if selected.ndim != 1 or len(selected) < 1:
        raise ValueError("targeted confirmation requires at least one source")
    if len(selected) >= target_count:
        raise ValueError(
            "targeted confirmation requires a strict source subset; use the "
            "full-matrix estimator for all neurons"
        )
    return progressive_smc_repaired_responses(
        adapter,
        standardized_trace,
        stimulus,
        cut_time=cut_time,
        projection=projection,
        source_low=source_low,
        source_high=source_high,
        source_iqr=source_iqr,
        thresholds=thresholds,
        config=config,
        seed=seed,
        branch_factor=branch_factor,
        future_branch_factor=future_branch_factor,
        tempering_ess_fraction=tempering_ess_fraction,
        max_tempering_resamples=max_tempering_resamples,
        source_indices=selected,
        include_factual_arm=True,
        endpoint_quantiles=endpoint_quantiles,
    )
