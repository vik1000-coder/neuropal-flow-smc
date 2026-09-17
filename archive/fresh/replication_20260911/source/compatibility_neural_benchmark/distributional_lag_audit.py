from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from compatibility_neural_benchmark.core import (
    GeneratorAdapter,
    causal_fill,
    fit_anchor_projection,
    standardize_for_checkpoint,
    stimulus_for_trace,
)
from conditional_neural_benchmark.data import Cohort, StimulusSchedule, load_cohort


STATES = ("quiet", "onset", "active")
METRICS = (
    "mean_shift",
    "log_sd_shift",
    "tail_probability_shift",
    "wasserstein1",
    "paired_transport_rms",
    "proper_energy_penalty",
)
DEFAULT_MODEL_ID = "stim_binary_any_stimulus_tcn_flow128_dropout15_jitter_p01"


@dataclass(frozen=True)
class AuditConfig:
    history_frames: int = 80
    lag_frames: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    states: tuple[str, ...] = STATES
    n_particles: int = 32
    projection_rank: int = 8
    ridge_alpha: float = 2.0
    residual_quantiles: tuple[float, float] = (0.25, 0.75)
    clip_quantiles: tuple[float, float] = (0.01, 0.99)

    def validate(self) -> None:
        if self.history_frames < 33:
            raise ValueError("history must include the lag-32 structural control")
        if not self.lag_frames or min(self.lag_frames) < 1:
            raise ValueError("lag frames must be positive")
        if max(self.lag_frames) > self.history_frames:
            raise ValueError("lag exceeds the checkpoint history")
        if not set(self.states).issubset(STATES):
            raise ValueError("unknown audit state")
        if self.n_particles < 4:
            raise ValueError("at least four particles are required")
        if self.projection_rank < 1:
            raise ValueError("projection rank must be positive")
        qlo, qhi = self.residual_quantiles
        if not (0 < qlo < qhi < 1):
            raise ValueError("residual quantiles must be ordered inside (0,1)")


@dataclass(frozen=True)
class ConditionalResidualModel:
    coefficients: np.ndarray  # [source, feature + intercept]
    residual_low: np.ndarray  # [source]
    residual_high: np.ndarray  # [source]
    clip_low: np.ndarray  # [source]
    clip_high: np.ndarray  # [source]
    n_fit_rows: int

    def predict_low_high(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(self.coefficients):
            raise ValueError("conditional feature matrix has the wrong source axis")
        augmented = np.concatenate(
            [features, np.ones((len(features), 1), dtype=np.float64)], axis=1
        )
        location = np.einsum("sq,sq->s", augmented, self.coefficients)
        low = np.clip(location + self.residual_low, self.clip_low, self.clip_high)
        high = np.clip(location + self.residual_high, self.clip_low, self.clip_high)
        middle = 0.5 * (low + high)
        half_gap = np.maximum(0.5 * (high - low), 1e-4)
        return (
            (middle - half_gap).astype(np.float32),
            (middle + half_gap).astype(np.float32),
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_folds(path: Path, cohort: Cohort) -> np.ndarray:
    frame = pd.read_csv(path)
    mapping = dict(zip(frame.worm_id.astype(str), frame.outer_fold.astype(int)))
    missing = [worm for worm in cohort.worm_ids if worm not in mapping]
    if missing:
        raise RuntimeError(f"fold evidence is missing worms: {missing}")
    return np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)


def _onset_frame(schedule: StimulusSchedule, event: int) -> int:
    return int(round(schedule.event_intervals_seconds[event][0] * schedule.analysis_fps))


def evaluation_targets(schedule: StimulusSchedule, n_frames: int) -> list[tuple[str, int, int]]:
    """Return one quiet/onset/active target per event.

    The model predicts frame ``target`` from history ending at ``target-1``.
    Quiet is matched 15 seconds before onset; onset is 1 second after onset;
    active is 6 seconds after onset.
    """
    fps = schedule.analysis_fps
    offsets = {
        "quiet": -int(round(15.0 * fps)) + int(round(1.0 * fps)),
        "onset": int(round(1.0 * fps)),
        "active": int(round(6.0 * fps)),
    }
    result: list[tuple[str, int, int]] = []
    for event in range(3):
        onset = _onset_frame(schedule, event)
        for state in STATES:
            target = onset + offsets[state]
            if 80 <= target < n_frames:
                result.append((state, event, target))
    return result


def fitting_targets(schedule: StimulusSchedule, n_frames: int) -> list[tuple[str, int, int]]:
    """Return denser, state-matched targets used only to fit the resampler."""
    fps = schedule.analysis_fps
    offsets_seconds = {
        "quiet": (-15.0, -14.0, -13.0, -12.0),
        "onset": (0.25, 0.5, 1.0, 1.5, 2.0),
        "active": (3.0, 4.5, 6.0, 7.5, 9.0),
    }
    result: list[tuple[str, int, int]] = []
    for event in range(3):
        onset = _onset_frame(schedule, event)
        for state, offsets in offsets_seconds.items():
            for offset in offsets:
                target = onset + int(round(offset * fps))
                if 80 <= target < n_frames:
                    result.append((state, event, target))
    return result


def conditional_features(
    trace: np.ndarray,
    target_time: int,
    lag: int,
    projection: np.ndarray,
) -> np.ndarray:
    """Features for every source with its perturbed coordinate removed.

    Global low-rank state is evaluated both at the candidate lag and at the
    latest observed frame. The source's contribution to each projection is
    subtracted. Three source-local neighboring values are included, but never
    the value being resampled and never the unobserved target frame.
    """
    trace = np.asarray(trace, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    frame = int(target_time) - int(lag)
    latest = int(target_time) - 1
    if frame < 3 or latest >= len(trace):
        raise ValueError("target lacks the history required for conditional resampling")
    global_frame = trace[frame] @ projection
    global_latest = trace[latest] @ projection
    excluded_frame = global_frame[None] - trace[frame, :, None] * projection
    excluded_latest = global_latest[None] - trace[latest, :, None] * projection
    forward = frame + 1 if frame + 1 <= latest else frame - 3
    neighbors = trace[[frame - 1, frame - 2, forward]].T
    return np.concatenate([excluded_frame, excluded_latest, neighbors], axis=1).astype(
        np.float32
    )


def fit_conditional_resampler(
    traces: Iterable[np.ndarray],
    schedules: Iterable[StimulusSchedule],
    *,
    state: str,
    lag: int,
    projection: np.ndarray,
    ridge_alpha: float,
    residual_quantiles: tuple[float, float],
    clip_quantiles: tuple[float, float],
) -> ConditionalResidualModel:
    feature_rows: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    for trace, schedule in zip(traces, schedules):
        for candidate_state, _, target in fitting_targets(schedule, len(trace)):
            if candidate_state != state:
                continue
            feature_rows.append(conditional_features(trace, target, lag, projection))
            outcomes.append(np.asarray(trace[target - lag], dtype=np.float32))
    if len(feature_rows) < 20:
        raise RuntimeError(f"too few conditional resampling rows for {state}, lag {lag}")
    features = np.stack(feature_rows).astype(np.float64)  # [row, source, feature]
    target = np.stack(outcomes).astype(np.float64)  # [row, source]
    n_rows, n_sources, n_features = features.shape
    coefficients = np.empty((n_sources, n_features + 1), dtype=np.float64)
    residual_low = np.empty(n_sources, dtype=np.float64)
    residual_high = np.empty(n_sources, dtype=np.float64)
    penalty = np.eye(n_features + 1, dtype=np.float64) * float(ridge_alpha)
    penalty[-1, -1] = 0.0
    for source in range(n_sources):
        design = np.concatenate(
            [features[:, source], np.ones((n_rows, 1), dtype=np.float64)], axis=1
        )
        coefficients[source] = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target[:, source],
        )
        residual = target[:, source] - design @ coefficients[source]
        residual_low[source], residual_high[source] = np.quantile(
            residual, residual_quantiles
        )
    clip_low, clip_high = np.quantile(target, clip_quantiles, axis=0)
    return ConditionalResidualModel(
        coefficients=coefficients.astype(np.float32),
        residual_low=residual_low.astype(np.float32),
        residual_high=residual_high.astype(np.float32),
        clip_low=clip_low.astype(np.float32),
        clip_high=clip_high.astype(np.float32),
        n_fit_rows=n_rows,
    )


def build_low_high_histories(
    history: np.ndarray,
    *,
    lag: int,
    low: np.ndarray,
    high: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(history, dtype=np.float32)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if history.ndim != 2:
        raise ValueError("history must have shape [lag, neuron]")
    if np.any(source_indices < 0) or np.any(source_indices >= history.shape[1]):
        raise ValueError("source index outside the neural history")
    low_history = np.repeat(history[None], len(source_indices), axis=0)
    high_history = low_history.copy()
    low_history[np.arange(len(source_indices)), -lag, source_indices] = low[source_indices]
    high_history[np.arange(len(source_indices)), -lag, source_indices] = high[source_indices]
    return low_history, high_history


def sample_many(
    adapter: GeneratorAdapter,
    neural_history: np.ndarray,
    stimulus_history: np.ndarray,
    *,
    n_particles: int,
    seed: int,
) -> np.ndarray:
    neural = torch.as_tensor(neural_history, dtype=torch.float32, device=adapter.device)
    stimulus = torch.as_tensor(stimulus_history, dtype=torch.float32, device=adapter.device)
    context = torch.cat([neural, stimulus], dim=2).reshape(len(neural), -1)
    with torch.no_grad():
        draw = adapter.model.sample(context, int(n_particles), seed=int(seed))
        if adapter.residual_target:
            draw = draw + neural[:, -1, None, :]
    return draw.detach().cpu().numpy().astype(np.float32, copy=False)


def univariate_energy(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Univariate energy score for [source, sample, target] arrays."""
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    first = np.abs(samples - target[None, None]).mean(axis=1)
    usable = samples.shape[1] - samples.shape[1] % 2
    paired = np.abs(samples[:, :usable:2] - samples[:, 1:usable:2]).mean(axis=1)
    return (first - 0.5 * paired).astype(np.float32)


def two_component_energy(
    low_samples: np.ndarray, high_samples: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Energy score of an equal low/high mixture with stratified pair terms."""
    low = np.asarray(low_samples, dtype=np.float64)
    high = np.asarray(high_samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    usable = low.shape[1] - low.shape[1] % 2
    if usable < 2 or high.shape != low.shape:
        raise ValueError("low/high samples must have the same even-usable shape")
    first = 0.5 * (
        np.abs(low - target[None, None]).mean(axis=1)
        + np.abs(high - target[None, None]).mean(axis=1)
    )
    within_low = np.abs(low[:, :usable:2] - low[:, 1:usable:2]).mean(axis=1)
    within_high = np.abs(high[:, :usable:2] - high[:, 1:usable:2]).mean(axis=1)
    # Shift particle identity so the cross-component term does not reuse the
    # common base draw that intentionally couples the treatment contrast.
    cross = np.abs(low - np.roll(high, shift=1, axis=1)).mean(axis=1)
    second = 0.25 * within_low + 0.25 * within_high + 0.5 * cross
    return (first - 0.5 * second).astype(np.float32)


def distributional_effects(
    low_samples: np.ndarray,
    high_samples: np.ndarray,
    factual_samples: np.ndarray,
    target: np.ndarray,
    tail_threshold: np.ndarray,
) -> dict[str, np.ndarray]:
    low = np.asarray(low_samples, dtype=np.float64)
    high = np.asarray(high_samples, dtype=np.float64)
    factual = np.asarray(factual_samples, dtype=np.float64)
    eps = 1e-6
    low_sd = low.std(axis=1, ddof=0)
    high_sd = high.std(axis=1, ddof=0)
    return {
        "mean_shift": (high.mean(axis=1) - low.mean(axis=1)).astype(np.float32),
        "log_sd_shift": np.log((high_sd + eps) / (low_sd + eps)).astype(np.float32),
        "tail_probability_shift": (
            (np.abs(high) > tail_threshold[None, None]).mean(axis=1)
            - (np.abs(low) > tail_threshold[None, None]).mean(axis=1)
        ).astype(np.float32),
        "wasserstein1": np.mean(
            np.abs(np.sort(high, axis=1) - np.sort(low, axis=1)), axis=1
        ).astype(np.float32),
        "paired_transport_rms": np.sqrt(np.mean(np.square(high - low), axis=1)).astype(
            np.float32
        ),
        "proper_energy_penalty": (
            two_component_energy(low, high, target) - univariate_energy(factual, target)
        ).astype(np.float32),
    }


def deterministic_seed(*values: object, base: int = 20260828) -> int:
    payload = ":".join(str(value) for value in values)
    code = int(hashlib.sha256(payload.encode()).hexdigest()[:12], 16)
    return int((base + code) % (2**31 - 1))


def checkpoint_path(
    source_run: Path, model_id: str, history_frames: int, fold: int, seed: int
) -> Path:
    return (
        source_run
        / "checkpoints"
        / "chemical_full_cv"
        / f"{model_id}__L{history_frames}__f{fold}__s{seed}.pt"
    )


def run_checkpoint(
    *,
    cohort: Cohort,
    folds: np.ndarray,
    source_run: Path,
    output_dir: Path,
    model_id: str,
    fold: int,
    model_seed: int,
    config: AuditConfig,
    device: str,
    source_indices: np.ndarray,
    resume: bool,
) -> dict[str, object]:
    output = output_dir / "responses" / f"direct_distributional__f{fold}__s{model_seed}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if resume and output.exists():
        with np.load(output, allow_pickle=False) as existing:
            if str(existing["status"].item()) == "complete":
                if str(existing["stimulus_schema_fingerprint"].item()) != cohort.stimulus_schema_fingerprint:
                    raise RuntimeError("resume rejected: stimulus schema differs")
                if int(existing["n_particles"].item()) != config.n_particles:
                    raise RuntimeError("resume rejected: particle count differs")
                return {"fold": fold, "model_seed": model_seed, "status": "skipped", "output": str(output), "wall_seconds": 0.0}

    path = checkpoint_path(source_run, model_id, config.history_frames, fold, model_seed)
    if not path.exists():
        raise FileNotFoundError(path)
    adapter = GeneratorAdapter.load(str(path), device=device)
    if adapter.neurons != tuple(cohort.neurons):
        raise RuntimeError("checkpoint neuron order mismatch")
    if adapter.checkpoint.get("stimulus_schema_fingerprint") != cohort.stimulus_schema_fingerprint:
        raise RuntimeError("checkpoint stimulus schema mismatch")
    if adapter.checkpoint.get("trial_metadata", {}).get("stimulus_encoding") != "binary_any_stimulus":
        raise RuntimeError("distributional audit requires the frozen binary checkpoint")

    started = time.perf_counter()
    training = np.flatnonzero(folds != fold)
    heldout = np.flatnonzero(folds == fold)
    training_traces = [
        causal_fill(standardize_for_checkpoint(cohort.traces[int(worm)], adapter.checkpoint))
        for worm in training
    ]
    projection = fit_anchor_projection(training_traces, config.projection_rank)
    pooled = np.concatenate(training_traces, axis=0)
    tail_threshold = np.quantile(np.abs(pooled), 0.90, axis=0).astype(np.float32)
    resamplers: dict[tuple[str, int], ConditionalResidualModel] = {}
    for state in config.states:
        for lag in config.lag_frames:
            resamplers[(state, lag)] = fit_conditional_resampler(
                training_traces,
                [cohort.stimulus_schedules[int(worm)] for worm in training],
                state=state,
                lag=lag,
                projection=projection,
                ridge_alpha=config.ridge_alpha,
                residual_quantiles=config.residual_quantiles,
                clip_quantiles=config.clip_quantiles,
            )

    shape = (
        len(heldout),
        len(config.states),
        len(config.lag_frames),
        len(source_indices),
        cohort.n_neurons,
    )
    effects = {name: np.full(shape, np.nan, dtype=np.float32) for name in METRICS}
    conditional_iqr = np.full(shape[:-1], np.nan, dtype=np.float32)
    event_counts = np.zeros(shape[:3], dtype=np.int16)

    for worm_position, worm in enumerate(heldout):
        trace = causal_fill(
            standardize_for_checkpoint(cohort.traces[int(worm)], adapter.checkpoint)
        )
        stimulus = stimulus_for_trace(
            cohort.traces[int(worm)], cohort, int(worm), adapter.checkpoint
        )
        schedule = cohort.stimulus_schedules[int(worm)]
        targets = evaluation_targets(schedule, len(trace))
        for state_position, state in enumerate(config.states):
            state_targets = [(event, target) for candidate, event, target in targets if candidate == state]
            if len(state_targets) != 3:
                raise RuntimeError(f"expected three {state} anchors for {schedule.worm_id}")
            for lag_position, lag in enumerate(config.lag_frames):
                accumulator = {name: [] for name in METRICS}
                gaps: list[np.ndarray] = []
                resampler = resamplers[(state, lag)]
                for event, target_time in state_targets:
                    history = trace[target_time - config.history_frames : target_time]
                    stimulus_history = stimulus[
                        target_time - config.history_frames : target_time
                    ]
                    features = conditional_features(trace, target_time, lag, projection)
                    low, high = resampler.predict_low_high(features)
                    low_history, high_history = build_low_high_histories(
                        history,
                        lag=lag,
                        low=low,
                        high=high,
                        source_indices=source_indices,
                    )
                    repeated_stimulus = np.repeat(
                        stimulus_history[None], len(source_indices), axis=0
                    )
                    factual_history = np.repeat(
                        history[None], len(source_indices), axis=0
                    )
                    draw_seed = deterministic_seed(
                        "crn", fold, model_seed, int(worm), state, event, lag
                    )
                    low_samples = sample_many(
                        adapter,
                        low_history,
                        repeated_stimulus,
                        n_particles=config.n_particles,
                        seed=draw_seed,
                    )
                    high_samples = sample_many(
                        adapter,
                        high_history,
                        repeated_stimulus,
                        n_particles=config.n_particles,
                        seed=draw_seed,
                    )
                    factual_samples = sample_many(
                        adapter,
                        factual_history,
                        repeated_stimulus,
                        n_particles=config.n_particles,
                        seed=draw_seed,
                    )
                    result = distributional_effects(
                        low_samples,
                        high_samples,
                        factual_samples,
                        trace[target_time],
                        tail_threshold,
                    )
                    for name in METRICS:
                        accumulator[name].append(result[name])
                    gaps.append((high - low)[source_indices])
                for name in METRICS:
                    effects[name][worm_position, state_position, lag_position] = np.mean(
                        accumulator[name], axis=0
                    )
                conditional_iqr[worm_position, state_position, lag_position] = np.mean(
                    gaps, axis=0
                )
                event_counts[worm_position, state_position, lag_position] = len(state_targets)
            print(
                f"DISTRIBUTIONAL_WORM fold={fold} seed={model_seed} "
                f"worm={schedule.worm_id} state={state}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    np.savez_compressed(
        output,
        status=np.asarray("complete"),
        created_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        checkpoint=np.asarray(str(path.resolve())),
        checkpoint_sha256=np.asarray(sha256(path)),
        model_id=np.asarray(model_id),
        fold=np.asarray(fold),
        model_seed=np.asarray(model_seed),
        n_particles=np.asarray(config.n_particles),
        history_frames=np.asarray(config.history_frames),
        lag_frames=np.asarray(config.lag_frames, dtype=np.int16),
        lag_seconds=np.asarray(config.lag_frames, dtype=np.float32) / cohort.fps,
        state_names=np.asarray(config.states),
        source_indices=source_indices.astype(np.int16),
        source_neurons=np.asarray([cohort.neurons[int(i)] for i in source_indices]),
        target_neurons=np.asarray(cohort.neurons),
        heldout_worm_indices=heldout.astype(np.int16),
        heldout_worm_ids=np.asarray([cohort.worm_ids[int(i)] for i in heldout]),
        event_counts=event_counts,
        conditional_iqr=conditional_iqr,
        tail_threshold=tail_threshold,
        projection=projection,
        stimulus_schema_version=np.asarray(cohort.stimulus_schema_version),
        stimulus_schema_fingerprint=np.asarray(cohort.stimulus_schema_fingerprint),
        wall_seconds=np.asarray(elapsed),
        **effects,
    )
    del adapter
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {
        "fold": fold,
        "model_seed": model_seed,
        "status": "ok",
        "output": str(output),
        "wall_seconds": elapsed,
    }


def write_state(output_dir: Path, manifest: dict, records: list[dict[str, object]]) -> None:
    manifest = dict(manifest)
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["records"] = records
    atomic_json(output_dir / "manifest.json", manifest)
    if records:
        fields = sorted(set().union(*(record.keys() for record in records)))
        temporary = output_dir / "checkpoint_status.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, output_dir / "checkpoint_status.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--fold-evidence", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cohort-mode", choices=("oh16230_head", "pooled_resampled"), default="oh16230_head")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903])
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--states", nargs="+", choices=STATES, default=list(STATES))
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--projection-rank", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=2.0)
    parser.add_argument("--source-neurons", nargs="+")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(cohort_mode=args.cohort_mode)
    folds = load_folds(args.fold_evidence.resolve(), cohort)
    source_indices = (
        np.arange(cohort.n_neurons, dtype=np.int64)
        if not args.source_neurons
        else np.asarray([cohort.neurons.index(name) for name in args.source_neurons], dtype=np.int64)
    )
    config = AuditConfig(
        lag_frames=tuple(sorted(set(args.lags))),
        states=tuple(args.states),
        n_particles=args.particles,
        projection_rank=args.projection_rank,
        ridge_alpha=args.ridge_alpha,
    )
    config.validate()
    source_run = args.source_run.resolve()
    fold_evidence = args.fold_evidence.resolve()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "protocol": "frozen paired-CRN conditional-resampling distributional lag audit v1",
        "source_run": str(source_run),
        "fold_evidence": str(fold_evidence),
        "fold_evidence_sha256": sha256(fold_evidence),
        "model_id": args.model_id,
        "cohort_mode": cohort.cohort_mode,
        "folds": args.folds,
        "seeds": args.seeds,
        "config": {
            "history_frames": config.history_frames,
            "lag_frames": list(config.lag_frames),
            "lag_seconds": [lag / cohort.fps for lag in config.lag_frames],
            "states": list(config.states),
            "n_particles": config.n_particles,
            "projection_rank": config.projection_rank,
            "ridge_alpha": config.ridge_alpha,
            "residual_quantiles": list(config.residual_quantiles),
            "clip_quantiles": list(config.clip_quantiles),
            "evaluation_anchor_seconds": {"quiet": -14.0, "onset": 1.0, "active": 6.0},
        },
        "source_neurons": [cohort.neurons[int(i)] for i in source_indices],
        "target_neurons": list(cohort.neurons),
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_schema": cohort.stimulus_schema_dict(),
        "common_random_numbers": "identical flow base seed for factual, conditional-low, and conditional-high histories within each source/event/lag",
        "resampling_estimand": "q25-to-q75 conditional residual replacement of one observed source coordinate at one physical lag",
        "negative_controls": {
            "lag32": "structurally outside the frozen legacy TCN's 31-frame receptive field",
            "zero_contrast": "unit-tested identical low/high histories with identical base noise",
            "source_label_permutation": "analysis-stage matrix-reliability null",
            "lag_label_permutation": "analysis-stage lag-profile null",
            "reversed_history": "rejected because it is off-support and not a valid empirical null",
        },
        "frozen_gate": {
            "cell_scope": "state×lag, off-diagonal source-target entries, worm is the inferential unit",
            "proper_score": "mean proper-energy penalty > 0 with 95% worm bootstrap lower bound > 0",
            "seed_reliability": "Spearman rho >= 0.30 and source-permutation p < 0.05",
            "worm_split_reliability": "Spearman rho >= 0.20 and source-permutation p < 0.05",
            "lag_specificity": "mean W1 >= 3x the matched lag-32 structural-control mean plus 1e-5",
            "all_required": True,
        },
        "selection_firewall": "no Cook, Randi, SBTG, receptor, transmitter, or neuromodulator atlas access",
        "claim_boundary": "model-relative observational distributional sensitivity; not a causal edge, synapse, receptor action, or physical transmission delay",
    }
    existing = output_dir / "manifest.json"
    records: list[dict[str, object]] = []
    if existing.exists():
        previous = json.loads(existing.read_text())
        for field, expected in (
            ("stimulus_schema_fingerprint", cohort.stimulus_schema_fingerprint),
            ("fold_evidence_sha256", manifest["fold_evidence_sha256"]),
            ("model_id", args.model_id),
        ):
            if previous.get(field) != expected:
                raise RuntimeError(f"resume rejected: {field} differs")
        if previous.get("config") != manifest["config"]:
            raise RuntimeError("resume rejected: audit config differs")
        if previous.get("source_neurons") != manifest["source_neurons"]:
            raise RuntimeError("resume rejected: source set differs")
        records = list(previous.get("records", []))
    write_state(output_dir, manifest, records)

    completed = {
        (int(row["fold"]), int(row["model_seed"]))
        for row in records
        if row.get("status") in ("ok", "skipped")
    }
    for fold in args.folds:
        for model_seed in args.seeds:
            if (fold, model_seed) in completed:
                continue
            record = run_checkpoint(
                cohort=cohort,
                folds=folds,
                source_run=source_run,
                output_dir=output_dir,
                model_id=args.model_id,
                fold=fold,
                model_seed=model_seed,
                config=config,
                device=args.device,
                source_indices=source_indices,
                resume=args.resume,
            )
            records.append(record)
            write_state(output_dir, manifest, records)

    manifest["status"] = "complete"
    write_state(output_dir, manifest, records)


if __name__ == "__main__":
    main()
