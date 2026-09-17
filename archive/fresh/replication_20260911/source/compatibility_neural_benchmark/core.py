from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.special import logsumexp

from conditional_neural_benchmark.data import (
    Cohort,
    FoldScaler,
    StimulusSchedule,
    _stimulus_features,
    _stimulus_mask,
)
from conditional_neural_benchmark.inference import load_checkpoint


PHASES = ("baseline", "onset", "active", "offset", "recovery")
# Standard deviations are measured in fold-standardized neural units.  This
# fixed floor makes the log-scale contrast finite for a collapsed empirical
# particle cloud while remaining negligible relative to the 0.1 clamp floor.
ENDPOINT_SD_FLOOR = 1e-6
# Compatibility-only constant for quarantined historical analysis modules.
# Corrected training/sampling code uses ``Cohort.stimulus_schedules`` and never
# consults this value.
STIMULUS_PERIODS_SECONDS = (
    (60.5, 70.5),
    (120.5, 130.5),
    (180.5, 190.5),
)


@dataclass(frozen=True)
class RepairedResponseConfig:
    history_frames: int = 80
    repair_frames: int = 4
    source_window_frames: int = 4
    source_lag_frames: int = 0
    horizon_frames: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 40)
    n_particles: int = 128
    anchor_lambda: float = 0.25
    epsilon_iqr_fraction: float = 0.25
    anchor_rank: int = 12
    min_ess: float = 20.0
    max_normalized_weight: float = 0.20
    min_achieved_fraction: float = 0.25
    resample_ess_fraction: float = 0.50
    sampling_chunk_size: int = 1024

    def validate(self) -> None:
        if self.history_frames < 1 or self.repair_frames < 1:
            raise ValueError("history and repair lengths must be positive")
        if not (1 <= self.source_window_frames <= self.repair_frames):
            raise ValueError("source window must fit inside the repair prefix")
        if self.source_lag_frames < 0:
            raise ValueError("source lag must be nonnegative")
        if self.source_window_frames + self.source_lag_frames > self.repair_frames:
            raise ValueError("lagged source window must fit inside the repair prefix")
        if not self.horizon_frames or min(self.horizon_frames) < 1:
            raise ValueError("forecast horizons must be positive")
        if self.n_particles < 2:
            raise ValueError("at least two particles are required")
        if self.epsilon_iqr_fraction <= 0:
            raise ValueError("source clamp bandwidth must be positive")
        if not (0.0 < self.resample_ess_fraction <= 1.0):
            raise ValueError("resampling ESS fraction must lie in (0, 1]")
        if self.sampling_chunk_size < 1:
            raise ValueError("sampling chunk size must be positive")


@dataclass(frozen=True)
class EpisodeCut:
    phase: str
    event: int
    time: int
    chemical_code: int
    chemical_name: str


@dataclass
class GeneratorAdapter:
    model: torch.nn.Module
    checkpoint: dict
    device: torch.device

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "auto") -> "GeneratorAdapter":
        model, checkpoint, resolved = load_checkpoint(checkpoint_path, device=device)
        return cls(model=model, checkpoint=checkpoint, device=resolved)

    @property
    def lag(self) -> int:
        return int(self.checkpoint["lag"])

    @property
    def neurons(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.checkpoint["neurons"])

    @property
    def residual_target(self) -> bool:
        return bool(self.checkpoint["model_config"].get("residual_target", False))

    def sample_standardized_next(
        self,
        neural_history: torch.Tensor,
        stimulus_history: torch.Tensor,
        *,
        seed: int,
    ) -> torch.Tensor:
        if neural_history.ndim != 3 or neural_history.shape[1] != self.lag:
            raise ValueError("neural history has the wrong shape")
        if stimulus_history.ndim == 2:
            stimulus_history = stimulus_history[..., None]
        if stimulus_history.ndim != 3 or stimulus_history.shape[:2] != neural_history.shape[:2]:
            raise ValueError("stimulus history has the wrong shape")
        expected = int(self.checkpoint.get("stimulus_channels", 1))
        if stimulus_history.shape[2] != expected:
            raise ValueError(
                f"checkpoint expects {expected} stimulus channels, got "
                f"{stimulus_history.shape[2]}"
            )
        context = torch.cat([neural_history, stimulus_history], dim=2).reshape(
            len(neural_history), -1
        )
        with torch.no_grad():
            draw = self.model.sample(context, 1, seed=seed)[:, 0]
        if self.residual_target:
            draw = draw + neural_history[:, -1]
        return draw


def causal_fill(trace: np.ndarray) -> np.ndarray:
    result = np.asarray(trace, dtype=np.float32).copy()
    for column in range(result.shape[1]):
        finite = np.isfinite(result[:, column])
        last = np.maximum.accumulate(np.where(finite, np.arange(len(result)), -1))
        usable = last >= 0
        result[usable, column] = result[last[usable], column]
        if not usable.all():
            first = np.flatnonzero(finite)
            if len(first):
                result[~usable, column] = result[first[0], column]
    if not np.isfinite(result).all():
        raise ValueError("a trace coordinate has no finite observation")
    return result


def episode_cuts(
    n_frames: int,
    schedule: StimulusSchedule,
    source_window_frames: int,
    *,
    boundary_shift_frames: int = 0,
) -> list[EpisodeCut]:
    cuts: list[EpisodeCut] = []
    fps = schedule.analysis_fps
    for event, (start_s, end_s) in enumerate(schedule.event_intervals_seconds):
        onset = int(round(start_s * fps)) + int(boundary_shift_frames)
        offset = int(round(end_s * fps)) + int(boundary_shift_frames)
        candidates = {
            "baseline": onset - int(round(15.0 * fps)),
            "onset": onset + source_window_frames - 1,
            "active": onset + int(round(5.0 * fps)),
            "offset": offset + source_window_frames - 1,
            "recovery": offset + int(round(5.0 * fps)),
        }
        for phase in PHASES:
            time = candidates[phase]
            if 0 <= time < n_frames:
                cuts.append(EpisodeCut(
                    phase=phase,
                    event=event,
                    time=time,
                    chemical_code=schedule.chemical_code_by_event[event],
                    chemical_name=schedule.chemical_name_by_event[event],
                ))
    return cuts


def fit_anchor_projection(
    standardized_training_traces: Iterable[np.ndarray], rank: int
) -> np.ndarray:
    pooled = np.concatenate(tuple(standardized_training_traces), axis=0).astype(np.float64)
    pooled = pooled[np.isfinite(pooled).all(axis=1)]
    pooled = pooled - pooled.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(pooled, full_matrices=False)
    r = min(int(rank), vt.shape[0], vt.shape[1])
    variance = np.square(singular[:r]) / max(1, len(pooled) - 1)
    variance = np.maximum(variance, 1e-4)
    # A maps a standardized observed-state displacement to a whitened latent
    # population displacement. Dividing the downstream cost by r keeps the
    # anchor-strength scale stable across rank choices.
    return (vt[:r].T / np.sqrt(variance)[None]).astype(np.float32)


def training_source_quantiles(
    standardized_training_traces: Iterable[np.ndarray],
    schedules: Iterable[StimulusSchedule],
    source_window_frames: int,
) -> dict[str, dict[str, np.ndarray]]:
    buckets: dict[str, list[np.ndarray]] = {phase: [] for phase in PHASES}
    traces = tuple(standardized_training_traces)
    schedule_values = tuple(schedules)
    if len(traces) != len(schedule_values):
        raise ValueError("training traces and stimulus schedules disagree")
    for trace, schedule in zip(traces, schedule_values):
        filled = causal_fill(trace)
        for cut in episode_cuts(len(trace), schedule, source_window_frames):
            lo = cut.time - source_window_frames + 1
            if lo >= 0:
                buckets[cut.phase].append(filled[lo : cut.time + 1].mean(axis=0))
    result: dict[str, dict[str, np.ndarray]] = {}
    for phase, values in buckets.items():
        if not values:
            raise ValueError(f"no training source statistics for phase {phase}")
        array = np.asarray(values, dtype=np.float32)
        q25, q75 = np.quantile(array, [0.25, 0.75], axis=0)
        iqr = np.maximum(q75 - q25, 0.20)
        result[phase] = {
            "low": q25.astype(np.float32),
            "high": q75.astype(np.float32),
            "iqr": iqr.astype(np.float32),
        }
    return result


def normalized_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if log_weights.ndim != 1 or not np.isfinite(log_weights).any():
        raise ValueError("log weights must contain at least one finite value")
    normalizer = float(logsumexp(log_weights))
    weights = np.exp(log_weights - normalizer)
    return weights, normalizer - np.log(len(log_weights))


def systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
    *,
    offset: float | None = None,
) -> np.ndarray:
    """Systematically resample one normalized particle-weight vector."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) < 2:
        raise ValueError("systematic resampling requires a one-dimensional particle vector")
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("particle weights must be finite and nonnegative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("particle weights must have positive mass")
    weights = weights / total
    unit_offset = float(rng.random()) if offset is None else float(offset)
    if not (0.0 <= unit_offset < 1.0):
        raise ValueError("systematic-resampling offset must lie in [0, 1)")
    positions = (unit_offset + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def systematic_resample_to_n(
    weights: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    offset: float | None = None,
) -> np.ndarray:
    """Systematically draw ``n_samples`` indices from normalized weights.

    ``systematic_resample`` intentionally preserves the input population size.
    Progressive branching SMC instead expands every parent into several
    children and then prunes that candidate population back to its declared
    particle count.  Keeping that size-changing operation explicit avoids
    silently changing the behavior of the terminal-only estimator.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) < 2:
        raise ValueError("systematic resampling requires a one-dimensional particle vector")
    if n_samples < 1:
        raise ValueError("the requested resample size must be positive")
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("particle weights must be finite and nonnegative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("particle weights must have positive mass")
    weights = weights / total
    unit_offset = float(rng.random()) if offset is None else float(offset)
    if not (0.0 <= unit_offset < 1.0):
        raise ValueError("systematic-resampling offset must lie in [0, 1)")
    positions = (unit_offset + np.arange(n_samples)) / n_samples
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def _chunked_standardized_next(
    adapter: GeneratorAdapter,
    neural_history: torch.Tensor,
    stimulus_history: torch.Tensor,
    *,
    seed: int,
    chunk_size: int,
    paired_halves: bool = False,
) -> torch.Tensor:
    """Sample one step for a large particle batch without changing its order.

    When ``paired_halves`` is true, the first and second halves receive the
    same base random numbers.  Their marginal transition draws are unchanged;
    this common-random-number coupling reduces Monte Carlo variance in the
    low-versus-high repaired response.
    """
    original_shape = neural_history.shape[:-2]
    flat_history = neural_history.reshape(-1, neural_history.shape[-2], neural_history.shape[-1])
    if stimulus_history.ndim == neural_history.ndim - 1:
        flat_stimulus = stimulus_history.reshape(-1, stimulus_history.shape[-1])
    elif stimulus_history.ndim == neural_history.ndim:
        flat_stimulus = stimulus_history.reshape(
            -1, stimulus_history.shape[-2], stimulus_history.shape[-1]
        )
    else:
        raise ValueError("stimulus history rank is incompatible with neural history")
    modulus = 2**31 - 1
    def sample_flat(history_part: torch.Tensor, stimulus_part: torch.Tensor) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for chunk_index, lo in enumerate(range(0, len(history_part), chunk_size)):
            hi = min(lo + chunk_size, len(history_part))
            chunk_seed = int((seed + 15_485_863 * chunk_index) % modulus)
            chunks.append(
                adapter.sample_standardized_next(
                    history_part[lo:hi], stimulus_part[lo:hi], seed=chunk_seed
                )
            )
        return torch.cat(chunks, dim=0)

    if paired_halves:
        if not original_shape or original_shape[0] % 2:
            raise ValueError("paired-half sampling requires an even leading group dimension")
        half = original_shape[0] // 2
        rows_per_half = half * int(np.prod(original_shape[1:], dtype=np.int64))
        first = sample_flat(flat_history[:rows_per_half], flat_stimulus[:rows_per_half])
        second = sample_flat(flat_history[rows_per_half:], flat_stimulus[rows_per_half:])
        sampled = torch.cat([first, second], dim=0)
    else:
        sampled = sample_flat(flat_history, flat_stimulus)
    return sampled.reshape(*original_shape, neural_history.shape[-1])


def _next_stimulus_tensor(
    value: np.ndarray | float,
    leading_shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast one scalar/vector stimulus value and retain its time axis."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return torch.full(
            (*leading_shape, 1), float(array), dtype=torch.float32, device=device
        )
    if array.ndim != 1:
        raise ValueError("a stimulus time point must be scalar or one-dimensional")
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    view = tensor.reshape(*([1] * len(leading_shape)), 1, len(array))
    return view.expand(*leading_shape, 1, len(array))


def generate_path_bank(
    adapter: GeneratorAdapter,
    standardized_trace: np.ndarray,
    stimulus: np.ndarray,
    *,
    cut_time: int,
    config: RepairedResponseConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config.validate()
    if adapter.lag != config.history_frames:
        raise ValueError("checkpoint and response history lengths disagree")
    trace = causal_fill(standardized_trace)
    start = cut_time - config.repair_frames
    history_lo = start - config.history_frames + 1
    if history_lo < 0 or cut_time + max(config.horizon_frames) >= len(trace):
        raise ValueError("episode lacks boundary history or forecast horizon")
    initial = trace[history_lo : start + 1]
    initial_stimulus = stimulus[history_lo : start + 1]
    if initial.shape != (config.history_frames, trace.shape[1]):
        raise RuntimeError("boundary history alignment failed")
    n = config.n_particles
    history = torch.as_tensor(
        np.repeat(initial[None], n, axis=0), dtype=torch.float32, device=adapter.device
    )
    stim_history = torch.as_tensor(
        np.repeat(initial_stimulus[None], n, axis=0),
        dtype=torch.float32,
        device=adapter.device,
    )
    # Keep the rollout on-device and synchronize once at the end.  Copying every
    # single step to NumPy forces an MPS synchronization and dominates MDN
    # runtime without changing the sampled path.
    generated: list[torch.Tensor] = []
    total_steps = config.repair_frames + max(config.horizon_frames)
    for step in range(total_steps):
        absolute_time = start + step + 1
        draw = adapter.sample_standardized_next(
            history, stim_history, seed=int(seed + 104729 * step)
        )
        generated.append(draw)
        next_stim = _next_stimulus_tensor(
            stimulus[absolute_time], (n,), device=adapter.device
        )
        history = torch.cat([history[:, 1:], draw[:, None]], dim=1)
        stim_history = torch.cat([stim_history[:, 1:], next_stim], dim=1)
    array = (
        torch.stack(generated, dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    prefix = array[:, : config.repair_frames]
    future = array[:, config.repair_frames :]
    factual_prefix = trace[start + 1 : cut_time + 1]
    return prefix, future, factual_prefix


def smc_repaired_responses(
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
    resampling_policy: str = "temporal_cut",
) -> dict[str, np.ndarray]:
    """Estimate repaired responses with literal bootstrap SMC and ESS resampling.

    There is one particle system for every (low/high target, source neuron)
    query.  The learned conditional generator is the bootstrap proposal.  The
    factual-anchor potential is applied incrementally during the repair prefix;
    the source-statistic clamp is applied when the declared lagged source window
    ends.  Its weight is carried to the temporal cut.  Under ``temporal_cut``
    ESS-triggered resampling can occur as soon as that clamp is available;
    under ``terminal_deferred`` source-driven resampling is suppressed until
    the mandatory cut-time systematic resample.  Both policies then use free,
    equally weighted future rollout.
    """
    config.validate()
    if resampling_policy not in {"temporal_cut", "terminal_deferred"}:
        raise ValueError(
            "resampling_policy must be temporal_cut or terminal_deferred"
        )
    if adapter.lag != config.history_frames:
        raise ValueError("checkpoint and response history lengths disagree")
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
    groups = 2 * d
    rank = int(projection.shape[1])
    source_index = np.tile(np.arange(d, dtype=np.int64), 2)
    target = np.concatenate(
        [np.asarray(source_low, dtype=np.float64), np.asarray(source_high, dtype=np.float64)]
    )
    bandwidth = np.maximum(
        np.asarray(source_iqr, dtype=np.float64) * config.epsilon_iqr_fraction, 0.10
    )[source_index]
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
    group_tensor = torch.arange(groups, device=adapter.device)[:, None]
    source_tensor = torch.as_tensor(
        source_index, dtype=torch.long, device=adapter.device
    )

    log_weights = np.full((groups, n), -np.log(n), dtype=np.float64)
    log_normalizer = np.zeros(groups, dtype=np.float64)
    source_sum = np.zeros((groups, n), dtype=np.float64)
    source_hi_step = config.repair_frames - config.source_lag_frames
    source_lo_step = source_hi_step - config.source_window_frames
    clamp_step = source_hi_step - 1
    anchor_cost_total = np.zeros((groups, n), dtype=np.float64)
    ancestors = np.broadcast_to(np.arange(n, dtype=np.int64), (groups, n)).copy()
    ess_steps = np.zeros((groups, config.repair_frames), dtype=np.float64)
    max_weight_steps = np.zeros_like(ess_steps)
    resampled_steps = np.zeros_like(ess_steps, dtype=np.int8)
    resample_threshold = config.resample_ess_fraction * n
    factual_prefix = trace[start + 1 : cut_time + 1]
    log_anchor_at_clamp = np.full(groups, np.nan, dtype=np.float64)
    clamp_log_increment = np.full(groups, np.nan, dtype=np.float64)

    def normalize_after_increment(increment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal log_weights, log_normalizer
        updated = log_weights + increment
        log_increment = logsumexp(updated, axis=1)
        log_normalizer += log_increment
        log_weights = updated - log_increment[:, None]
        weights = np.exp(log_weights)
        return weights, log_increment

    def apply_resampling(weights: np.ndarray, selected_groups: np.ndarray) -> None:
        nonlocal history, stim_history, source_sum, anchor_cost_total, ancestors, log_weights
        if not np.any(selected_groups):
            return
        indices = np.broadcast_to(np.arange(n, dtype=np.int64), (groups, n)).copy()
        offsets = rng.random(d)
        for group in np.flatnonzero(selected_groups):
            indices[group] = systematic_resample(
                weights[group], rng, offset=float(offsets[group % d])
            )
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=adapter.device)
        history = history[group_tensor, index_tensor]
        stim_history = stim_history[group_tensor, index_tensor]
        source_sum = np.take_along_axis(source_sum, indices, axis=1)
        anchor_cost_total = np.take_along_axis(anchor_cost_total, indices, axis=1)
        ancestors = np.take_along_axis(ancestors, indices, axis=1)
        log_weights[selected_groups] = -np.log(n)

    for step in range(config.repair_frames):
        absolute_time = start + step + 1
        draw = _chunked_standardized_next(
            adapter,
            history,
            stim_history,
            seed=int((seed + 104_729 * step) % modulus),
            chunk_size=config.sampling_chunk_size,
            paired_halves=True,
        )
        draw_array = draw.detach().cpu().numpy().astype(np.float64, copy=False)
        in_source_window = source_lo_step <= step < source_hi_step
        if in_source_window:
            source_sum += np.take_along_axis(
                draw_array,
                np.broadcast_to(source_index[:, None, None], (groups, n, 1)),
                axis=2,
            )[:, :, 0]

        factual = torch.as_tensor(
            factual_prefix[step], dtype=torch.float32, device=adapter.device
        )
        difference = draw - factual[None, None]
        latent = difference @ projection_tensor
        if in_source_window:
            source_difference = difference.gather(
                2, source_tensor[:, None, None].expand(groups, n, 1)
            ).squeeze(2)
            source_projection = projection_tensor[source_tensor]
            latent = latent - source_difference[..., None] * source_projection[:, None]
        incremental_anchor = (
            0.5
            * latent.square().sum(dim=-1).detach().cpu().numpy().astype(np.float64)
            / (rank * config.repair_frames)
        )
        anchor_cost_total += incremental_anchor
        weights, _ = normalize_after_increment(
            -config.anchor_lambda * incremental_anchor
        )
        if step == source_hi_step - 1:
            log_anchor_at_clamp = log_normalizer.copy()
            source_value = source_sum / config.source_window_frames
            terminal_log_clamp = -0.5 * np.square(
                (source_value - target[:, None]) / bandwidth[:, None]
            )
            weights, clamp_log_increment = normalize_after_increment(terminal_log_clamp)
        ess_steps[:, step] = 1.0 / np.square(weights).sum(axis=1)
        max_weight_steps[:, step] = weights.max(axis=1)

        next_stimulus = _next_stimulus_tensor(
            stimulus[absolute_time], (groups, n), device=adapter.device
        )
        history = torch.cat([history[:, :, 1:], draw[:, :, None]], dim=2)
        stim_history = torch.cat([stim_history[:, :, 1:], next_stimulus], dim=2)

        if step < config.repair_frames - 1:
            selected = ess_steps[:, step] < resample_threshold
            # In the temporal-cut estimator the lagged source potential is
            # allowed to trigger resampling as soon as its window ends.  The
            # terminal-deferred variant carries that importance weight to the
            # cut and performs the mandatory resampling only there.  Before
            # the clamp, both variants may resample an anchor-degenerate cloud.
            if resampling_policy == "terminal_deferred" and step >= clamp_step:
                selected = np.zeros_like(selected)
            resampled_steps[selected, step] = 1
            apply_resampling(weights, selected)
        else:
            terminal_weights = weights.copy()

    log_total = log_normalizer.copy()
    # Resampling after an early source clamp changes particle order.  Recover
    # the source statistic from the resampled accumulator before final moments.
    source_value = source_sum / config.source_window_frames
    weighted_anchor = np.sum(terminal_weights * anchor_cost_total, axis=1)
    weighted_source = np.sum(terminal_weights * source_value, axis=1)
    terminal_offsets = rng.random(d)
    terminal_indices = np.stack(
        [
            systematic_resample(
                terminal_weights[group], rng, offset=float(terminal_offsets[group % d])
            )
            for group in range(groups)
        ]
    )
    terminal_index_tensor = torch.as_tensor(
        terminal_indices, dtype=torch.long, device=adapter.device
    )
    history = history[group_tensor, terminal_index_tensor]
    stim_history = stim_history[group_tensor, terminal_index_tensor]
    source_sum = np.take_along_axis(source_sum, terminal_indices, axis=1)
    anchor_cost_total = np.take_along_axis(anchor_cost_total, terminal_indices, axis=1)
    ancestors = np.take_along_axis(ancestors, terminal_indices, axis=1)
    resampled_steps[:, -1] = 1

    horizon_frames = np.asarray(config.horizon_frames, dtype=np.int64)
    horizon_to_index = {int(value): i for i, value in enumerate(horizon_frames)}
    feature_store = {
        key: np.empty((groups, len(horizon_frames), d), dtype=np.float64)
        for key in (
            "endpoint_mean",
            "cumulative_mean",
            "peak_mean",
            "event_probability",
            "endpoint_sd",
        )
    }
    endpoint_wasserstein1 = np.empty(
        (d, len(horizon_frames), d), dtype=np.float64
    )
    cumulative = np.zeros((groups, n, d), dtype=np.float64)
    peak = np.full((groups, n, d), -np.inf, dtype=np.float64)
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
                draw_array[:d], draw_array[d:]
            )

        next_stimulus = _next_stimulus_tensor(
            stimulus[absolute_time], (groups, n), device=adapter.device
        )
        history = torch.cat([history[:, :, 1:], draw[:, :, None]], dim=2)
        stim_history = torch.cat([stim_history[:, :, 1:], next_stimulus], dim=2)

    response = {
        f"response_{key}": (value[d:] - value[:d]).astype(np.float32)
        for key, value in feature_store.items()
    }
    response["response_endpoint_log_sd"] = _endpoint_log_sd_response(
        feature_store["endpoint_sd"][:d], feature_store["endpoint_sd"][d:]
    )
    response["response_endpoint_wasserstein1"] = endpoint_wasserstein1.astype(
        np.float32
    )
    achieved_low, achieved_high = weighted_source[:d], weighted_source[d:]
    achieved_gap = achieved_high - achieved_low
    target_gap = np.asarray(source_high) - np.asarray(source_low)
    terminal_ess_low, terminal_ess_high = (
        ess_steps[:d, clamp_step],
        ess_steps[d:, clamp_step],
    )
    terminal_max_low = max_weight_steps[:d, clamp_step]
    terminal_max_high = max_weight_steps[d:, clamp_step]
    valid = (
        (terminal_ess_low >= config.min_ess)
        & (terminal_ess_high >= config.min_ess)
        & (terminal_max_low <= config.max_normalized_weight)
        & (terminal_max_high <= config.max_normalized_weight)
        & (achieved_gap >= config.min_achieved_fraction * target_gap)
    )
    distinct_ancestors = np.asarray(
        [len(np.unique(ancestors[group])) for group in range(groups)], dtype=np.float64
    )
    diagnostics = {
        "target_low": np.asarray(source_low),
        "target_high": np.asarray(source_high),
        "target_gap": target_gap,
        "achieved_low": achieved_low,
        "achieved_high": achieved_high,
        "achieved_gap": achieved_gap,
        "ess_low": terminal_ess_low,
        "ess_high": terminal_ess_high,
        "max_weight_low": terminal_max_low,
        "max_weight_high": terminal_max_high,
        "min_step_ess_low": ess_steps[:d].min(axis=1),
        "min_step_ess_high": ess_steps[d:].min(axis=1),
        "distinct_ancestors_low": distinct_ancestors[:d],
        "distinct_ancestors_high": distinct_ancestors[d:],
        "anchor_cost_mean_low": weighted_anchor[:d],
        "anchor_cost_mean_high": weighted_anchor[d:],
        "log10_total_low": log_total[:d] / np.log(10.0),
        "log10_total_high": log_total[d:] / np.log(10.0),
        "log10_anchor_at_clamp_low": log_anchor_at_clamp[:d] / np.log(10.0),
        "log10_anchor_at_clamp_high": log_anchor_at_clamp[d:] / np.log(10.0),
        # Backward-compatible aliases.  At lag zero the clamp ends at the cut,
        # so these retain the exact historical meaning.
        "log10_anchor_low": log_anchor_at_clamp[:d] / np.log(10.0),
        "log10_anchor_high": log_anchor_at_clamp[d:] / np.log(10.0),
        "log10_clamp_increment_low": clamp_log_increment[:d] / np.log(10.0),
        "log10_clamp_increment_high": clamp_log_increment[d:] / np.log(10.0),
        "valid": valid.astype(np.float32),
        "endpoint_sd_floor": np.full(d, ENDPOINT_SD_FLOOR, dtype=np.float64),
        "step_ess_low": ess_steps[:d],
        "step_ess_high": ess_steps[d:],
        "step_max_weight_low": max_weight_steps[:d],
        "step_max_weight_high": max_weight_steps[d:],
        "step_resampled_low": resampled_steps[:d],
        "step_resampled_high": resampled_steps[d:],
        "resampling_policy_code": np.full(
            d, 0.0 if resampling_policy == "temporal_cut" else 1.0
        ),
    }
    return {
        **response,
        **{f"diagnostic_{key}": np.asarray(value, dtype=np.float32) for key, value in diagnostics.items()},
    }


def source_specific_anchor_costs(
    prefix: np.ndarray,
    factual_prefix: np.ndarray,
    projection: np.ndarray,
    source_window_frames: int,
    source_lag_frames: int = 0,
) -> np.ndarray:
    """Return [particles, sources] latent anchor costs.

    The designated source is excluded only during the source-statistic window;
    it remains anchored earlier in the repaired prefix.
    """
    difference = np.asarray(prefix, dtype=np.float32) - np.asarray(
        factual_prefix, dtype=np.float32
    )[None]
    base = np.einsum("nbd,dr->nbr", difference, projection, optimize=True)
    correction = difference[..., None] * projection[None, None, :, :]
    source_hi = prefix.shape[1] - int(source_lag_frames)
    source_lo = source_hi - int(source_window_frames)
    if source_lo < 0 or source_hi > prefix.shape[1]:
        raise ValueError("lagged source window does not fit inside the prefix")
    omit = np.zeros(prefix.shape[1], dtype=np.float32)
    omit[source_lo:source_hi] = 1.0
    repaired = base[:, :, None, :] - correction * omit[None, :, None, None]
    rank = projection.shape[1]
    return (0.5 * np.square(repaired).sum(axis=-1).mean(axis=1) / rank).astype(np.float32)


def _weighted_features(
    normalized_weights: np.ndarray,
    future: np.ndarray,
    horizons: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    # weights [sources, particles], future [particles, time, targets]
    selected = future[:, horizons - 1]
    cumulative = np.cumsum(future, axis=1) / np.arange(1, future.shape[1] + 1)[None, :, None]
    cumulative = cumulative[:, horizons - 1]
    peak = np.maximum.accumulate(future, axis=1)[:, horizons - 1]
    event = (peak > thresholds[None, None]).astype(np.float32)
    endpoint_mean = np.einsum("sn,nht->sht", normalized_weights, selected, optimize=True)
    endpoint_second = np.einsum(
        "sn,nht->sht", normalized_weights, np.square(selected), optimize=True
    )
    return {
        "endpoint_mean": endpoint_mean,
        "cumulative_mean": np.einsum(
            "sn,nht->sht", normalized_weights, cumulative, optimize=True
        ),
        "peak_mean": np.einsum("sn,nht->sht", normalized_weights, peak, optimize=True),
        "event_probability": np.einsum(
            "sn,nht->sht", normalized_weights, event, optimize=True
        ),
        "endpoint_sd": np.sqrt(np.maximum(endpoint_second - np.square(endpoint_mean), 0.0)),
    }


def _endpoint_log_sd_response(
    low_sd: np.ndarray,
    high_sd: np.ndarray,
    *,
    floor: float = ENDPOINT_SD_FLOOR,
) -> np.ndarray:
    """Return ``log(sd_high + floor) - log(sd_low + floor)``.

    The explicit floor is part of the response estimand and is also emitted as
    a diagnostic by every repaired-response estimator.
    """
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("the endpoint-SD floor must be finite and positive")
    low = np.asarray(low_sd, dtype=np.float64)
    high = np.asarray(high_sd, dtype=np.float64)
    if low.shape != high.shape:
        raise ValueError("low and high endpoint SD arrays must have matching shapes")
    if np.any(low < 0) or np.any(high < 0):
        raise ValueError("endpoint standard deviations must be nonnegative")
    return (np.log(high + floor) - np.log(low + floor)).astype(np.float32)


def _weighted_endpoint_wasserstein1(
    low_weights: np.ndarray,
    high_weights: np.ndarray,
    future: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    """Exact weighted one-dimensional endpoint W1 for a shared path bank.

    Both low/high source-regime laws reweight the same generated paths. For
    each requested horizon and target neuron, sorting endpoint values once fixes the support;
    the 1D Wasserstein distance is then the integral of the absolute difference
    between the two weighted empirical CDFs.  Sources are evaluated together.
    """
    low = np.asarray(low_weights, dtype=np.float64)
    high = np.asarray(high_weights, dtype=np.float64)
    values = np.asarray(future, dtype=np.float64)
    requested = np.asarray(horizons, dtype=np.int64)
    if low.shape != high.shape or low.ndim != 2:
        raise ValueError("low and high weights must have shape [sources, particles]")
    if values.ndim != 3 or values.shape[0] != low.shape[1]:
        raise ValueError("future must have shape [particles, time, targets]")
    if requested.ndim != 1 or np.any(requested < 1) or np.any(requested > values.shape[1]):
        raise ValueError("requested endpoint horizons fall outside the future path bank")
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("endpoint weights must be finite")
    if np.any(low < 0) or np.any(high < 0):
        raise ValueError("endpoint weights must be nonnegative")
    low_total = low.sum(axis=1, keepdims=True)
    high_total = high.sum(axis=1, keepdims=True)
    if np.any(low_total <= 0) or np.any(high_total <= 0):
        raise ValueError("endpoint weights must have positive mass")
    low = low / low_total
    high = high / high_total

    sources = low.shape[0]
    targets = values.shape[2]
    result = np.empty((sources, len(requested), targets), dtype=np.float64)
    for horizon_index, horizon in enumerate(requested):
        endpoint = values[:, int(horizon) - 1]
        for target in range(targets):
            order = np.argsort(endpoint[:, target], kind="stable")
            sorted_values = endpoint[order, target]
            widths = np.diff(sorted_values)
            if not np.any(widths):
                result[:, horizon_index, target] = 0.0
                continue
            low_cdf = np.cumsum(low[:, order], axis=1)[:, :-1]
            high_cdf = np.cumsum(high[:, order], axis=1)[:, :-1]
            result[:, horizon_index, target] = np.sum(
                np.abs(high_cdf - low_cdf) * widths[None], axis=1
            )
    return result


def _equal_weight_endpoint_wasserstein1(
    low_samples: np.ndarray,
    high_samples: np.ndarray,
) -> np.ndarray:
    """Exact empirical endpoint W1 for equally weighted particle systems."""
    low = np.asarray(low_samples, dtype=np.float64)
    high = np.asarray(high_samples, dtype=np.float64)
    if low.shape != high.shape or low.ndim != 3:
        raise ValueError(
            "low and high endpoint samples must share [sources, particles, targets]"
        )
    return np.mean(
        np.abs(np.sort(high, axis=1) - np.sort(low, axis=1)), axis=1
    )


def estimate_repaired_responses(
    prefix: np.ndarray,
    future: np.ndarray,
    factual_prefix: np.ndarray,
    projection: np.ndarray,
    source_low: np.ndarray,
    source_high: np.ndarray,
    source_iqr: np.ndarray,
    thresholds: np.ndarray,
    config: RepairedResponseConfig,
    *,
    anchor_lambda: float | None = None,
    epsilon_iqr_fraction: float | None = None,
) -> dict[str, np.ndarray]:
    config.validate()
    lam = config.anchor_lambda if anchor_lambda is None else float(anchor_lambda)
    eps_fraction = (
        config.epsilon_iqr_fraction
        if epsilon_iqr_fraction is None
        else float(epsilon_iqr_fraction)
    )
    if lam < 0 or eps_fraction <= 0:
        raise ValueError("invalid repair semantics")
    n, _, d = prefix.shape
    source_hi = prefix.shape[1] - config.source_lag_frames
    source_lo = source_hi - config.source_window_frames
    source_value = prefix[:, source_lo:source_hi].mean(axis=1)
    anchor_cost = source_specific_anchor_costs(
        prefix,
        factual_prefix,
        projection,
        config.source_window_frames,
        config.source_lag_frames,
    )
    bandwidth = np.maximum(source_iqr * eps_fraction, 0.10)
    normalized: dict[str, np.ndarray] = {}
    diagnostics: dict[str, np.ndarray] = {}
    for label, target in (("low", source_low), ("high", source_high)):
        log_clamp = -0.5 * np.square((source_value - target[None]) / bandwidth[None])
        log_total = log_clamp - lam * anchor_cost
        weights = np.empty((d, n), dtype=np.float64)
        log_alpha = np.empty(d, dtype=np.float64)
        log_source = np.empty(d, dtype=np.float64)
        for source in range(d):
            weights[source], log_alpha[source] = normalized_log_weights(log_total[:, source])
            _, log_source[source] = normalized_log_weights(log_clamp[:, source])
        normalized[label] = weights
        achieved = np.einsum("sn,ns->s", weights, source_value, optimize=True)
        ess = 1.0 / np.square(weights).sum(axis=1)
        entropy_ancestors = np.exp(
            -np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=1)
        )
        diagnostics[f"log10_total_{label}"] = log_alpha / np.log(10.0)
        diagnostics[f"log10_source_{label}"] = log_source / np.log(10.0)
        diagnostics[f"log10_repair_{label}"] = (log_alpha - log_source) / np.log(10.0)
        diagnostics[f"achieved_{label}"] = achieved
        diagnostics[f"ess_{label}"] = ess
        diagnostics[f"max_weight_{label}"] = weights.max(axis=1)
        diagnostics[f"entropy_ancestors_{label}"] = entropy_ancestors

    low_features = _weighted_features(
        normalized["low"], future, np.asarray(config.horizon_frames), thresholds
    )
    high_features = _weighted_features(
        normalized["high"], future, np.asarray(config.horizon_frames), thresholds
    )
    response = {
        f"response_{key}": (high_features[key] - low_features[key]).astype(np.float32)
        for key in low_features
    }
    response["response_endpoint_log_sd"] = _endpoint_log_sd_response(
        low_features["endpoint_sd"], high_features["endpoint_sd"]
    )
    response["response_endpoint_wasserstein1"] = _weighted_endpoint_wasserstein1(
        normalized["low"],
        normalized["high"],
        future,
        np.asarray(config.horizon_frames),
    ).astype(np.float32)
    target_gap = source_high - source_low
    achieved_gap = diagnostics["achieved_high"] - diagnostics["achieved_low"]
    valid = (
        (diagnostics["ess_low"] >= config.min_ess)
        & (diagnostics["ess_high"] >= config.min_ess)
        & (diagnostics["max_weight_low"] <= config.max_normalized_weight)
        & (diagnostics["max_weight_high"] <= config.max_normalized_weight)
        & (achieved_gap >= config.min_achieved_fraction * target_gap)
    )
    diagnostics["target_low"] = source_low
    diagnostics["target_high"] = source_high
    diagnostics["target_gap"] = target_gap
    diagnostics["achieved_gap"] = achieved_gap
    diagnostics["valid"] = valid.astype(np.float32)
    diagnostics["anchor_cost_median"] = np.median(anchor_cost, axis=0)
    diagnostics["endpoint_sd_floor"] = np.full(
        d, ENDPOINT_SD_FLOOR, dtype=np.float64
    )
    return {**response, **{f"diagnostic_{key}": value for key, value in diagnostics.items()}}


def standardize_for_checkpoint(trace: np.ndarray, checkpoint: dict) -> np.ndarray:
    mean = np.asarray(checkpoint["scaler"]["mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler"]["scale"], dtype=np.float32)
    return ((np.asarray(trace, dtype=np.float32) - mean) / scale).astype(np.float32)


def checkpoint_training_context(
    cohort: Cohort,
    fold_assignments: np.ndarray,
    fold: int,
    checkpoint: dict,
    config: RepairedResponseConfig,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], np.ndarray]:
    training = np.flatnonzero(fold_assignments != fold)
    standardized = [
        standardize_for_checkpoint(cohort.traces[int(i)], checkpoint) for i in training
    ]
    projection = fit_anchor_projection(
        (causal_fill(trace) for trace in standardized), config.anchor_rank
    )
    quantiles = training_source_quantiles(
        standardized,
        (cohort.stimulus_schedules[int(i)] for i in training),
        config.source_window_frames,
    )
    pooled = np.concatenate([causal_fill(trace) for trace in standardized], axis=0)
    thresholds = np.quantile(pooled, 0.90, axis=0).astype(np.float32)
    return projection, quantiles, thresholds


def stimulus_for_trace(
    trace: np.ndarray,
    cohort: Cohort,
    worm_index: int,
    checkpoint: dict | None = None,
) -> np.ndarray:
    worm_index = int(worm_index)
    if not (0 <= worm_index < cohort.n_worms):
        raise IndexError("worm_index is outside the cohort")
    schedule = cohort.stimulus_schedules[worm_index]
    if checkpoint is None:
        return _stimulus_features(
            len(trace), schedule, "binary_any_stimulus"
        )
    version = checkpoint.get("stimulus_schema_version")
    fingerprint = checkpoint.get("stimulus_schema_fingerprint")
    if version != cohort.stimulus_schema_version:
        raise RuntimeError("checkpoint stimulus-schema version does not match the cohort")
    if fingerprint != cohort.stimulus_schema_fingerprint:
        raise RuntimeError("checkpoint stimulus-schema fingerprint does not match the cohort")
    metadata = checkpoint.get("trial_metadata", {})
    encoding = str(metadata.get("stimulus_encoding", "binary_any_stimulus"))
    override = None
    if encoding == "chemical_onehot_subject_shuffle":
        orders = metadata.get("subject_shuffle_orders")
        if not isinstance(orders, dict) or schedule.worm_id not in orders:
            raise RuntimeError("checkpoint lacks the frozen shuffled chemical order")
        override = tuple(int(value) for value in orders[schedule.worm_id])
    return _stimulus_features(len(trace), schedule, encoding, override)
