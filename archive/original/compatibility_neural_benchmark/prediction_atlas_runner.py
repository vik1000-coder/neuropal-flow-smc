"""Production runner for the corrected NeuroPAL prediction atlas.

This module deliberately has a narrow scope: it evaluates the binary-any-
stimulus conditional flow with either direct importance weighting or
progressive bridge SMC.  Chemical labels are preserved as event metadata for
post-hoc stratification; they are not inputs to this generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from compatibility_neural_benchmark.core import (
    ENDPOINT_SD_FLOOR,
    GeneratorAdapter,
    RepairedResponseConfig,
    causal_fill,
    episode_cuts,
    estimate_repaired_responses,
    fit_anchor_projection,
    generate_path_bank,
    standardize_for_checkpoint,
    stimulus_for_trace,
)
from compatibility_neural_benchmark.progressive_smc import (
    progressive_smc_repaired_responses,
)
from conditional_neural_benchmark.data import load_cohort


METHODS = ("direct_importance", "progressive_bridge_smc")
PHASES = ("baseline", "onset", "active", "offset", "recovery")
RESPONSE_KEYS = (
    "response_endpoint_mean",
    "response_cumulative_mean",
    "response_peak_mean",
    "response_event_probability",
    "response_endpoint_sd",
    "response_endpoint_log_sd",
    "response_endpoint_wasserstein1",
)
ARCHIVE_SCHEMA_VERSION = "prediction_atlas_response_v2"
MANIFEST_SCHEMA_VERSION = "prediction_atlas_manifest_v2"
RESPONSE_AXES = ("heldout_worm", "phase", "event", "source", "horizon", "target")
GENERATOR_ENCODING = "binary_any_stimulus"
EPISODE_SEED_DEFINITION = (
    "sha256(model_id) prefix + base_seed + 1000003*fold + 1009*generator_seed "
    "+ 9176*worm_index + 131*phase_index + 17*event_index + 53*source_lag; "
    "sampler method deliberately omitted; modulo 2**31-1"
)
COMMON_NOISE_DEFINITION = (
    "within each estimator, low/high future arms use identical flow base-noise "
    "seeds and aligned particle-row order"
)
DISTRIBUTION_SCALE_FLOOR = ENDPOINT_SD_FLOOR
FROZEN_DEFAULT_METHOD = "progressive_bridge_smc"
FROZEN_PROGRESSIVE_PARTICLES = 32
FROZEN_PROGRESSIVE_FUTURE_BRANCH_FACTOR = 1
COMMON_REQUIRED_DIAGNOSTICS = (
    "achieved_low",
    "achieved_high",
    "achieved_gap",
    "target_gap",
    "ess_low",
    "ess_high",
    "max_weight_low",
    "max_weight_high",
    "valid",
    "endpoint_sd_floor",
)
PROGRESSIVE_REQUIRED_DIAGNOSTICS = (
    "distinct_ancestors_low",
    "distinct_ancestors_high",
    "step_forced_tempering_low",
    "step_forced_tempering_high",
)


METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "response_endpoint_mean": {
        "signed": True,
        "definition": "E_high[Y_target(cut+h)] - E_low[Y_target(cut+h)]",
    },
    "response_cumulative_mean": {
        "signed": True,
        "definition": "difference in expected path average from cut+1 through cut+h",
    },
    "response_peak_mean": {
        "signed": True,
        "definition": "difference in expected pathwise maximum from cut+1 through cut+h",
    },
    "response_event_probability": {
        "signed": True,
        "definition": "difference in probability of crossing the fold-training 0.90 target threshold by h",
        "threshold_quantile": 0.90,
    },
    "response_endpoint_sd": {
        "signed": True,
        "definition": "SD_high[Y_target(cut+h)] - SD_low[Y_target(cut+h)]",
        "ddof": 0,
    },
    "response_endpoint_log_sd": {
        "signed": True,
        "definition": "log(SD_high+floor) - log(SD_low+floor) at cut+h",
        "floor": DISTRIBUTION_SCALE_FLOOR,
        "ddof": 0,
    },
    "response_endpoint_wasserstein1": {
        "signed": False,
        "definition": "one-dimensional W1 distance between high and low endpoint marginals at cut+h",
        "units": "fold-standardized neural activity",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_manifest_stimulus_fingerprint(manifest: dict[str, Any]) -> str:
    """Read either supported manifest layout and reject internal disagreement."""
    top_level = manifest.get("stimulus_schema_fingerprint")
    schema = manifest.get("stimulus_schema")
    nested = schema.get("fingerprint") if isinstance(schema, dict) else None
    if top_level is not None and nested is not None and top_level != nested:
        raise RuntimeError(
            "source-run manifest has conflicting stimulus schema fingerprints"
        )
    value = top_level if top_level is not None else nested
    if not isinstance(value, str) or not value:
        raise RuntimeError("source-run manifest lacks a stimulus schema fingerprint")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_path(
    source_run: Path,
    checkpoint_phase: str,
    model_id: str,
    history_lag: int,
    fold: int,
    seed: int,
) -> Path:
    return (
        source_run
        / "checkpoints"
        / checkpoint_phase
        / f"{model_id}__L{history_lag}__f{fold}__s{seed}.pt"
    )


def output_path(
    output: Path,
    method: str,
    model_id: str,
    source_lag: int,
    fold: int,
    seed: int,
    particles: int,
) -> Path:
    return (
        output
        / "responses"
        / method
        / (
            f"{model_id}__{method}__ell{source_lag}__N{particles}"
            f"__f{fold}__s{seed}.npz"
        )
    )


def timing_metadata(
    source_lag: int, horizons: tuple[int, ...], fps: float
) -> dict[str, np.ndarray | float | str]:
    horizon_array = np.asarray(horizons, dtype=np.int16)
    return {
        "source_lag_seconds": float(source_lag / fps),
        "horizon_frames": horizon_array,
        "horizon_seconds": horizon_array.astype(np.float32) / fps,
        "source_to_readout_seconds": (
            source_lag + horizon_array.astype(np.float32)
        ) / fps,
        "lag_definition": "source-window end to prediction cut",
    }


def expected_episode_metadata(
    cohort,
    worm_indices: np.ndarray,
    source_lag: int,
    source_window_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact cut times and half-open source-window bounds."""
    worm_indices = np.asarray(worm_indices, dtype=np.int64)
    cut_times = np.full((len(worm_indices), len(PHASES), 3), -1, dtype=np.int32)
    bounds = np.full((len(worm_indices), len(PHASES), 3, 2), -1, dtype=np.int32)
    for worm_position, worm in enumerate(worm_indices):
        cuts = episode_cuts(
            len(cohort.traces[int(worm)]),
            cohort.stimulus_schedules[int(worm)],
            source_window_frames,
        )
        for cut in cuts:
            if cut.phase not in PHASES:
                continue
            phase_index = PHASES.index(cut.phase)
            if cut_times[worm_position, phase_index, cut.event] >= 0:
                raise RuntimeError("duplicate phase/event cut")
            source_hi = cut.time - source_lag + 1
            source_lo = source_hi - source_window_frames
            cut_times[worm_position, phase_index, cut.event] = cut.time
            bounds[worm_position, phase_index, cut.event] = (source_lo, source_hi)
    if np.any(cut_times < 0):
        raise RuntimeError("one or more prespecified episode cuts are unavailable")
    if np.any(bounds[..., 0] < 0):
        raise RuntimeError("one or more lagged source windows cross the trace boundary")
    if np.any(bounds[..., 1] - bounds[..., 0] != source_window_frames):
        raise RuntimeError("source-window bounds disagree with declared width")
    return cut_times, bounds


def validate_forecast_bounds(
    cohort,
    worm_indices: np.ndarray,
    cut_times: np.ndarray,
    horizons: tuple[int, ...],
) -> None:
    """Require every declared readout to lie strictly inside its trace."""
    if not horizons or min(horizons) < 1:
        raise RuntimeError("forecast horizons must be nonempty and positive")
    for worm_position, worm in enumerate(np.asarray(worm_indices, dtype=np.int64)):
        if np.any(
            cut_times[worm_position] + max(horizons)
            >= len(cohort.traces[int(worm)])
        ):
            raise RuntimeError(
                "one or more forecast horizons cross the trace boundary"
            )


def validate_resume_archive(
    archive,
    *,
    cohort,
    method: str,
    model_id: str,
    fold: int,
    seed: int,
    source_lag: int,
    particles: int,
    horizons: tuple[int, ...],
    source_window_frames: int,
    history_lag: int,
    base_seed: int,
    requested_device: str,
    checkpoint: Path,
    expected_worm_indices: np.ndarray,
) -> None:
    """Fail closed before reusing any archive at a production output path."""
    required = {
        "status",
        "archive_schema_version",
        "method",
        "model_id",
        "checkpoint",
        "checkpoint_sha256",
        "fold",
        "seed",
        "base_seed",
        "episode_seed_definition",
        "common_noise_definition",
        "requested_device",
        "resolved_device",
        "history_frames",
        "repair_frames",
        "source_lag_frames",
        "source_lag_seconds",
        "source_window_frames",
        "source_window_bounds_semantics",
        "n_particles",
        "horizon_frames",
        "horizon_seconds",
        "source_to_readout_seconds",
        "lag_definition",
        "fps",
        "phase_names",
        "response_keys",
        "response_axes",
        "worm_indices",
        "worm_ids",
        "neurons",
        "cut_times",
        "source_window_bounds",
        "stimulus_schema_version",
        "stimulus_schema_fingerprint",
        "stimulus_generator_encoding",
        "chemical_identity_conditioned",
        "chemical_code_by_worm_event",
        "chemical_name_by_worm_event",
        "distribution_scale_floor",
    }.union(RESPONSE_KEYS)
    diagnostic_names = list(COMMON_REQUIRED_DIAGNOSTICS)
    if method == "progressive_bridge_smc":
        diagnostic_names.extend(PROGRESSIVE_REQUIRED_DIAGNOSTICS)
    required.update(f"diagnostic_{name}" for name in diagnostic_names)
    missing = sorted(required.difference(archive.files))
    if missing:
        raise RuntimeError(
            "resume rejected: archive lacks required fields: " + ", ".join(missing)
        )
    if not checkpoint.exists():
        raise RuntimeError("resume rejected: checkpoint no longer exists")
    scalar_expected = {
        "status": "complete",
        "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        "method": method,
        "model_id": model_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "fold": int(fold),
        "seed": int(seed),
        "base_seed": int(base_seed),
        "episode_seed_definition": EPISODE_SEED_DEFINITION,
        "common_noise_definition": COMMON_NOISE_DEFINITION,
        "requested_device": str(requested_device),
        "history_frames": int(history_lag),
        "repair_frames": int(source_lag + source_window_frames),
        "source_lag_frames": int(source_lag),
        "source_window_frames": int(source_window_frames),
        "source_window_bounds_semantics": "[inclusive_start,exclusive_stop)",
        "n_particles": int(particles),
        "lag_definition": "source-window end to prediction cut",
        "fps": float(cohort.fps),
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_generator_encoding": GENERATOR_ENCODING,
        "chemical_identity_conditioned": False,
        "distribution_scale_floor": DISTRIBUTION_SCALE_FLOOR,
    }
    for key, expected in scalar_expected.items():
        actual = archive[key].item()
        if isinstance(expected, float):
            equal = bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-8))
        else:
            equal = actual == expected
        if not equal:
            raise RuntimeError(
                f"resume rejected: archive {key} differs "
                f"(found {actual!r}, expected {expected!r})"
            )
    resolved_device = str(archive["resolved_device"].item())
    if not resolved_device or (
        str(requested_device) != "auto" and resolved_device != str(requested_device)
    ):
        raise RuntimeError("resume rejected: archive resolved_device differs")

    expected_worm_indices = np.asarray(expected_worm_indices, dtype=np.int64)
    expected_cut_times, expected_bounds = expected_episode_metadata(
        cohort, expected_worm_indices, source_lag, source_window_frames
    )
    try:
        validate_forecast_bounds(
            cohort, expected_worm_indices, expected_cut_times, horizons
        )
    except RuntimeError as error:
        raise RuntimeError(f"resume rejected: {error}") from error
    timing = timing_metadata(source_lag, horizons, cohort.fps)
    expected_codes = np.asarray(
        [
            cohort.stimulus_schedules[int(index)].chemical_code_by_event
            for index in expected_worm_indices
        ],
        dtype=np.int8,
    )
    expected_names = np.asarray(
        [
            cohort.stimulus_schedules[int(index)].chemical_name_by_event
            for index in expected_worm_indices
        ]
    )
    array_expected = {
        "horizon_frames": timing["horizon_frames"],
        "horizon_seconds": timing["horizon_seconds"],
        "source_to_readout_seconds": timing["source_to_readout_seconds"],
        "phase_names": np.asarray(PHASES),
        "response_keys": np.asarray(RESPONSE_KEYS),
        "response_axes": np.asarray(RESPONSE_AXES),
        "worm_indices": expected_worm_indices.astype(np.int16),
        "worm_ids": np.asarray(
            [cohort.worm_ids[int(index)] for index in expected_worm_indices]
        ),
        "neurons": np.asarray(cohort.neurons),
        "cut_times": expected_cut_times,
        "source_window_bounds": expected_bounds,
        "chemical_code_by_worm_event": expected_codes,
        "chemical_name_by_worm_event": expected_names,
    }
    for key, expected in array_expected.items():
        actual = archive[key]
        if np.issubdtype(np.asarray(expected).dtype, np.floating):
            equal = np.allclose(actual, expected, rtol=0.0, atol=1e-7)
        else:
            equal = np.array_equal(actual, expected)
        if not equal:
            raise RuntimeError(f"resume rejected: archive {key} differs")
    if not np.isclose(
        float(archive["source_lag_seconds"].item()),
        float(timing["source_lag_seconds"]),
        rtol=0.0,
        atol=1e-8,
    ):
        raise RuntimeError("resume rejected: archive source_lag_seconds differs")

    shape = (
        len(expected_worm_indices),
        len(PHASES),
        3,
        cohort.n_neurons,
        len(horizons),
        cohort.n_neurons,
    )
    for key in RESPONSE_KEYS:
        value = archive[key]
        if value.shape != shape:
            raise RuntimeError(
                f"resume rejected: archive {key} shape is {value.shape}, expected {shape}"
            )
        if not np.isfinite(value).all():
            raise RuntimeError(f"resume rejected: archive {key} contains nonfinite values")
    event_probability = archive["response_event_probability"]
    if np.any(event_probability < -1.0 - 1e-6) or np.any(
        event_probability > 1.0 + 1e-6
    ):
        raise RuntimeError(
            "resume rejected: response_event_probability falls outside [-1,1]"
        )
    wasserstein = archive["response_endpoint_wasserstein1"]
    if np.any(wasserstein < -1e-7):
        raise RuntimeError(
            "resume rejected: response_endpoint_wasserstein1 is negative"
        )

    diagnostic_shape = (
        len(expected_worm_indices), len(PHASES), 3, cohort.n_neurons
    )
    for name in COMMON_REQUIRED_DIAGNOSTICS:
        key = f"diagnostic_{name}"
        value = archive[key]
        if value.shape != diagnostic_shape:
            raise RuntimeError(
                f"resume rejected: archive {key} shape is {value.shape}, "
                f"expected {diagnostic_shape}"
            )
        if not np.isfinite(value).all():
            raise RuntimeError(f"resume rejected: archive {key} contains nonfinite values")
    if np.any(archive["diagnostic_target_gap"] <= 0):
        raise RuntimeError("resume rejected: diagnostic_target_gap is not positive")
    if (
        np.any(archive["diagnostic_ess_low"] <= 0)
        or np.any(archive["diagnostic_ess_high"] <= 0)
        or np.any(archive["diagnostic_ess_low"] > particles + 1e-4)
        or np.any(archive["diagnostic_ess_high"] > particles + 1e-4)
    ):
        raise RuntimeError("resume rejected: diagnostic ESS is out of range")
    if np.any(archive["diagnostic_max_weight_low"] < 0) or np.any(
        archive["diagnostic_max_weight_high"] < 0
    ):
        raise RuntimeError("resume rejected: diagnostic max weight is negative")
    if method == "direct_importance" and (
        np.any(archive["diagnostic_max_weight_low"] > 1.0 + 1e-6)
        or np.any(archive["diagnostic_max_weight_high"] > 1.0 + 1e-6)
    ):
        raise RuntimeError("resume rejected: direct-importance support diagnostics are out of range")
    valid = archive["diagnostic_valid"]
    if not np.all(np.isin(valid, (0.0, 1.0))):
        raise RuntimeError("resume rejected: diagnostic_valid is not binary")
    if not np.allclose(
        archive["diagnostic_endpoint_sd_floor"],
        DISTRIBUTION_SCALE_FLOOR,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("resume rejected: diagnostic_endpoint_sd_floor differs")
    if method == "progressive_bridge_smc":
        for name in ("distinct_ancestors_low", "distinct_ancestors_high"):
            key = f"diagnostic_{name}"
            value = archive[key]
            if value.shape != diagnostic_shape or not np.isfinite(value).all():
                raise RuntimeError(f"resume rejected: archive {key} is invalid")
            if np.any(value < 1) or np.any(value > particles):
                raise RuntimeError(f"resume rejected: archive {key} is out of range")
        step_shape = (*diagnostic_shape, source_lag + source_window_frames)
        for name in ("step_forced_tempering_low", "step_forced_tempering_high"):
            key = f"diagnostic_{name}"
            value = archive[key]
            if value.shape != step_shape or not np.isfinite(value).all():
                raise RuntimeError(
                    f"resume rejected: archive {key} shape/range is invalid"
                )
            if not np.all(np.isin(value, (0.0, 1.0))):
                raise RuntimeError(f"resume rejected: archive {key} is not binary")

    # Every diagnostic emitted by the estimator is part of the support audit,
    # including fields beyond the small required compatibility subset above.
    # A corrupt optional diagnostic must not be allowed through a resume skip.
    allowed_diagnostic_shapes = {
        diagnostic_shape,
        (*diagnostic_shape, source_lag + source_window_frames),
    }
    for key in archive.files:
        if not key.startswith("diagnostic_"):
            continue
        value = np.asarray(archive[key])
        if value.shape not in allowed_diagnostic_shapes:
            raise RuntimeError(
                f"resume rejected: archive {key} has undeclared diagnostic shape "
                f"{value.shape}"
            )
        if not np.isfinite(value).all():
            raise RuntimeError(f"resume rejected: archive {key} contains nonfinite values")


def load_folds(path: Path, cohort) -> np.ndarray:
    frame = pd.read_csv(path)
    required = {"worm_id", "outer_fold"}
    if not required.issubset(frame.columns):
        raise RuntimeError("fold file lacks worm_id or outer_fold")
    if frame.worm_id.astype(str).duplicated().any():
        raise RuntimeError("fold file contains duplicated worm IDs")
    mapping = dict(zip(frame.worm_id.astype(str), frame.outer_fold.astype(int)))
    missing = sorted(set(cohort.worm_ids) - set(mapping))
    if missing:
        raise RuntimeError(f"fold file is missing cohort worms: {missing}")
    folds = np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)
    if set(folds.tolist()) != set(range(5)):
        raise RuntimeError("expected immutable five-fold assignment")
    return folds


def training_context(
    cohort,
    folds: np.ndarray,
    fold: int,
    checkpoint: dict,
    config: RepairedResponseConfig,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Fit nuisance summaries on training worms at the exact source lag."""
    training = np.flatnonzero(folds != fold)
    traces = [
        causal_fill(standardize_for_checkpoint(cohort.traces[int(i)], checkpoint))
        for i in training
    ]
    projection = fit_anchor_projection(traces, config.anchor_rank)
    buckets: dict[str, list[np.ndarray]] = {phase: [] for phase in PHASES}
    for trace, worm in zip(traces, training):
        cuts = episode_cuts(
            len(trace),
            cohort.stimulus_schedules[int(worm)],
            config.source_window_frames,
        )
        for cut in cuts:
            source_hi = cut.time - config.source_lag_frames + 1
            source_lo = source_hi - config.source_window_frames
            if source_lo >= 0:
                buckets[cut.phase].append(trace[source_lo:source_hi].mean(axis=0))
    quantiles: dict[str, dict[str, np.ndarray]] = {}
    for phase, values in buckets.items():
        if not values:
            raise RuntimeError(f"no lag-aligned training source statistics for {phase}")
        array = np.asarray(values, dtype=np.float32)
        q25, q75 = np.quantile(array, [0.25, 0.75], axis=0)
        quantiles[phase] = {
            "low": q25.astype(np.float32),
            "high": q75.astype(np.float32),
            "iqr": np.maximum(q75 - q25, 0.20).astype(np.float32),
        }
    pooled = np.concatenate(traces, axis=0)
    thresholds = np.quantile(pooled, 0.90, axis=0).astype(np.float32)
    return projection, quantiles, thresholds


def draw_seed(
    base: int,
    model_id: str,
    fold: int,
    seed: int,
    worm: int,
    phase: int,
    event: int,
    source_lag: int,
) -> int:
    # The method is deliberately omitted, giving matched episodes the same
    # keyed seed under both sampling strategies.
    code = int(hashlib.sha256(model_id.encode()).hexdigest()[:8], 16)
    return int(
        (
            base
            + code
            + 1_000_003 * fold
            + 1009 * seed
            + 9176 * worm
            + 131 * phase
            + 17 * event
            + 53 * source_lag
        )
        % (2**31 - 1)
    )


def estimate(
    method: str,
    adapter: GeneratorAdapter,
    standardized: np.ndarray,
    stimulus: np.ndarray,
    *,
    cut_time: int,
    projection: np.ndarray,
    quantiles: dict[str, np.ndarray],
    thresholds: np.ndarray,
    config: RepairedResponseConfig,
    seed: int,
    progressive_branch_factor: int,
    progressive_future_branch_factor: int,
) -> dict[str, np.ndarray]:
    if method == "direct_importance":
        prefix, future, factual_prefix = generate_path_bank(
            adapter,
            standardized,
            stimulus,
            cut_time=cut_time,
            config=config,
            seed=seed,
        )
        return estimate_repaired_responses(
            prefix,
            future,
            factual_prefix,
            projection,
            quantiles["low"],
            quantiles["high"],
            quantiles["iqr"],
            thresholds,
            config,
        )
    if method == "progressive_bridge_smc":
        return progressive_smc_repaired_responses(
            adapter,
            standardized,
            stimulus,
            cut_time=cut_time,
            projection=projection,
            source_low=quantiles["low"],
            source_high=quantiles["high"],
            source_iqr=quantiles["iqr"],
            thresholds=thresholds,
            config=config,
            seed=seed,
            branch_factor=progressive_branch_factor,
            future_branch_factor=progressive_future_branch_factor,
        )
    raise ValueError(f"unsupported atlas method {method}")


def _validate_checkpoint(
    adapter: GeneratorAdapter,
    *,
    cohort,
    model_id: str,
    fold: int,
    seed: int,
    history_lag: int,
    validation_profile: str = "canonical",
) -> None:
    checkpoint = adapter.checkpoint
    checks = {
        "model_config.model_id": (
            checkpoint.get("model_config", {}).get("model_id"),
            model_id,
        ),
        "fold": (checkpoint.get("fold"), fold),
        "seed": (checkpoint.get("seed"), seed),
        "lag": (checkpoint.get("lag"), history_lag),
    }
    if validation_profile == "canonical":
        checks.update(
            {
                "stimulus_channels": (checkpoint.get("stimulus_channels"), 1),
                "trial_metadata.stimulus_encoding": (
                    checkpoint.get("trial_metadata", {}).get("stimulus_encoding"),
                    GENERATOR_ENCODING,
                ),
                "trial_metadata.cohort_mode": (
                    checkpoint.get("trial_metadata", {}).get("cohort_mode"),
                    cohort.cohort_mode,
                ),
                "stimulus_schema_version": (
                    checkpoint.get("stimulus_schema_version"),
                    cohort.stimulus_schema_version,
                ),
                "stimulus_schema_fingerprint": (
                    checkpoint.get("stimulus_schema_fingerprint"),
                    cohort.stimulus_schema_fingerprint,
                ),
            }
        )
    elif validation_profile == "historical_sbtg80":
        if not str(cohort.cohort_mode).startswith("historical_sbtg_"):
            raise RuntimeError(
                "historical_sbtg80 checkpoint validation requires a historical SBTG cohort"
            )
        # The frozen higher-order tournament predates the richer stimulus
        # provenance fields.  Validate every field it does serialize and fail
        # if an optional field is present but contradictory; the immutable
        # cohort schedule remains the source of stimulus timing downstream.
        optional_checks = {
            "stimulus_channels": (checkpoint.get("stimulus_channels"), 1),
            "trial_metadata.stimulus_encoding": (
                checkpoint.get("trial_metadata", {}).get("stimulus_encoding"),
                GENERATOR_ENCODING,
            ),
            "trial_metadata.cohort_mode": (
                checkpoint.get("trial_metadata", {}).get("cohort_mode"),
                cohort.cohort_mode,
            ),
            "stimulus_schema_version": (
                checkpoint.get("stimulus_schema_version"),
                cohort.stimulus_schema_version,
            ),
            "stimulus_schema_fingerprint": (
                checkpoint.get("stimulus_schema_fingerprint"),
                cohort.stimulus_schema_fingerprint,
            ),
        }
        checks.update(
            {
                label: (actual, expected)
                for label, (actual, expected) in optional_checks.items()
                if actual is not None
            }
        )
    else:
        raise ValueError(f"unknown checkpoint validation profile {validation_profile!r}")
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(
                f"checkpoint {label} mismatch (found {actual!r}, expected {expected!r})"
            )
    if tuple(cohort.neurons) != adapter.neurons:
        raise RuntimeError("checkpoint neuron order mismatch")


def run_one(
    *,
    cohort,
    folds: np.ndarray,
    source_run: Path,
    checkpoint_phase: str,
    output: Path,
    method: str,
    model_id: str,
    history_lag: int,
    fold: int,
    seed: int,
    source_lag: int,
    particles: int,
    horizons: tuple[int, ...],
    source_window_frames: int,
    device: str,
    base_seed: int,
    min_ess: float,
    progressive_branch_factor: int,
    progressive_future_branch_factor: int,
    checkpoint_override: Path | None = None,
    checkpoint_validation_profile: str = "canonical",
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unsupported atlas method {method}")
    path = output_path(output, method, model_id, source_lag, fold, seed, particles)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = (
        Path(checkpoint_override).resolve()
        if checkpoint_override is not None
        else checkpoint_path(
            source_run, checkpoint_phase, model_id, history_lag, fold, seed
        )
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    heldout = np.flatnonzero(folds == fold)
    if not len(heldout):
        raise RuntimeError(f"fold {fold} has no held-out worms")
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            validate_resume_archive(
                existing,
                cohort=cohort,
                method=method,
                model_id=model_id,
                fold=fold,
                seed=seed,
                source_lag=source_lag,
                particles=particles,
                horizons=horizons,
                source_window_frames=source_window_frames,
                history_lag=history_lag,
                base_seed=base_seed,
                requested_device=device,
                checkpoint=ckpt_path,
                expected_worm_indices=heldout,
            )
        return {"status": "skipped", "output": str(path), "wall_seconds": 0.0}

    expected_cut_times, expected_bounds = expected_episode_metadata(
        cohort, heldout, source_lag, source_window_frames
    )
    validate_forecast_bounds(cohort, heldout, expected_cut_times, horizons)
    started = time.perf_counter()
    adapter = GeneratorAdapter.load(str(ckpt_path), device=device)
    _validate_checkpoint(
        adapter,
        cohort=cohort,
        model_id=model_id,
        fold=fold,
        seed=seed,
        history_lag=history_lag,
        validation_profile=checkpoint_validation_profile,
    )
    config = RepairedResponseConfig(
        history_frames=history_lag,
        repair_frames=source_lag + source_window_frames,
        source_window_frames=source_window_frames,
        source_lag_frames=source_lag,
        horizon_frames=horizons,
        n_particles=particles,
        min_ess=min_ess,
        resample_ess_fraction=0.50,
        sampling_chunk_size=1024,
    )
    config.validate()
    projection, quantiles, thresholds = training_context(
        cohort, folds, fold, adapter.checkpoint, config
    )
    d, h = cohort.n_neurons, len(horizons)
    archive_shape = (len(heldout), len(PHASES), 3, d, h, d)
    response = {
        key: np.full(archive_shape, np.nan, dtype=np.float32) for key in RESPONSE_KEYS
    }
    diagnostic_shapes: dict[str, tuple[int, ...]] | None = None
    diagnostic: dict[str, np.ndarray] = {}

    for worm_position, worm in enumerate(heldout):
        standardized = causal_fill(
            standardize_for_checkpoint(cohort.traces[int(worm)], adapter.checkpoint)
        )
        stimulus = stimulus_for_trace(
            cohort.traces[int(worm)],
            cohort,
            int(worm),
            (
                adapter.checkpoint
                if checkpoint_validation_profile == "canonical"
                else None
            ),
        )
        cuts = episode_cuts(
            len(cohort.traces[int(worm)]),
            cohort.stimulus_schedules[int(worm)],
            source_window_frames,
        )
        for cut in cuts:
            phase_index = PHASES.index(cut.phase)
            keyed_seed = draw_seed(
                base_seed,
                model_id,
                fold,
                seed,
                int(worm),
                phase_index,
                cut.event,
                source_lag,
            )
            result = estimate(
                method,
                adapter,
                standardized,
                stimulus,
                cut_time=cut.time,
                projection=projection,
                quantiles=quantiles[cut.phase],
                thresholds=thresholds,
                config=config,
                seed=keyed_seed,
                progressive_branch_factor=progressive_branch_factor,
                progressive_future_branch_factor=progressive_future_branch_factor,
            )
            for key in RESPONSE_KEYS:
                if key not in result:
                    raise RuntimeError(f"estimator did not return required metric {key}")
                value = np.asarray(result[key])
                if value.shape != (d, h, d) or not np.isfinite(value).all():
                    raise RuntimeError(
                        f"estimator metric {key} has invalid shape or nonfinite values"
                    )
                response[key][worm_position, phase_index, cut.event] = value
            current_shapes = {
                key.removeprefix("diagnostic_"): tuple(np.asarray(value).shape)
                for key, value in result.items()
                if key.startswith("diagnostic_") and np.asarray(value).ndim in {1, 2}
            }
            if any(not shape or shape[0] != d for shape in current_shapes.values()):
                raise RuntimeError("diagnostic arrays must begin with the source axis")
            if diagnostic_shapes is None:
                diagnostic_shapes = current_shapes
                required_diagnostics = set(COMMON_REQUIRED_DIAGNOSTICS)
                if method == "progressive_bridge_smc":
                    required_diagnostics.update(PROGRESSIVE_REQUIRED_DIAGNOSTICS)
                missing_diagnostics = sorted(
                    required_diagnostics.difference(diagnostic_shapes)
                )
                if missing_diagnostics:
                    raise RuntimeError(
                        "estimator lacks required diagnostics: "
                        + ", ".join(missing_diagnostics)
                    )
                diagnostic = {
                    name: np.full(
                        (len(heldout), len(PHASES), 3, *shape),
                        np.nan,
                        dtype=np.float32,
                    )
                    for name, shape in diagnostic_shapes.items()
                }
            if current_shapes != diagnostic_shapes:
                raise RuntimeError("diagnostic schema changed within a run")
            for name, expected_shape in diagnostic_shapes.items():
                value = np.asarray(result[f"diagnostic_{name}"], dtype=np.float32)
                if value.shape != expected_shape or not np.isfinite(value).all():
                    raise RuntimeError(f"invalid diagnostic array {name}")
                diagnostic[name][worm_position, phase_index, cut.event] = value
        print(
            f"ATLAS_WORM_DONE method={method} ell={source_lag} "
            f"fold={fold} heldout_worm_index={int(worm)}",
            flush=True,
        )

    for key, value in response.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"one or more atlas cells were not evaluated for {key}")
    for name, value in diagnostic.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"one or more diagnostic cells were not evaluated for {name}")
    elapsed = time.perf_counter() - started
    timing = timing_metadata(source_lag, horizons, cohort.fps)
    _atomic_npz(
        path,
        status=np.asarray("complete"),
        archive_schema_version=np.asarray(ARCHIVE_SCHEMA_VERSION),
        created_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        method=np.asarray(method),
        model_id=np.asarray(model_id),
        checkpoint=np.asarray(str(ckpt_path.resolve())),
        checkpoint_sha256=np.asarray(sha256(ckpt_path)),
        fold=np.asarray(fold),
        seed=np.asarray(seed),
        base_seed=np.asarray(base_seed),
        episode_seed_definition=np.asarray(EPISODE_SEED_DEFINITION),
        common_noise_definition=np.asarray(COMMON_NOISE_DEFINITION),
        requested_device=np.asarray(str(device)),
        resolved_device=np.asarray(str(adapter.device)),
        history_frames=np.asarray(history_lag),
        repair_frames=np.asarray(config.repair_frames),
        source_lag_frames=np.asarray(source_lag),
        source_lag_seconds=np.asarray(timing["source_lag_seconds"]),
        lag_definition=np.asarray(timing["lag_definition"]),
        source_window_frames=np.asarray(source_window_frames),
        source_window_bounds_semantics=np.asarray("[inclusive_start,exclusive_stop)"),
        n_particles=np.asarray(particles),
        progressive_branch_factor=np.asarray(progressive_branch_factor),
        progressive_future_branch_factor=np.asarray(
            progressive_future_branch_factor
        ),
        horizon_frames=timing["horizon_frames"],
        horizon_seconds=timing["horizon_seconds"],
        source_to_readout_seconds=timing["source_to_readout_seconds"],
        fps=np.asarray(cohort.fps),
        phase_names=np.asarray(PHASES),
        response_keys=np.asarray(RESPONSE_KEYS),
        response_axes=np.asarray(RESPONSE_AXES),
        worm_indices=heldout.astype(np.int16),
        worm_ids=np.asarray([cohort.worm_ids[int(index)] for index in heldout]),
        neurons=np.asarray(cohort.neurons),
        cut_times=expected_cut_times,
        source_window_bounds=expected_bounds,
        stimulus_schema_version=np.asarray(cohort.stimulus_schema_version),
        stimulus_schema_fingerprint=np.asarray(cohort.stimulus_schema_fingerprint),
        stimulus_generator_encoding=np.asarray(GENERATOR_ENCODING),
        chemical_identity_conditioned=np.asarray(False),
        chemical_code_by_worm_event=np.asarray(
            [
                cohort.stimulus_schedules[int(index)].chemical_code_by_event
                for index in heldout
            ],
            dtype=np.int8,
        ),
        chemical_name_by_worm_event=np.asarray(
            [
                cohort.stimulus_schedules[int(index)].chemical_name_by_event
                for index in heldout
            ]
        ),
        distribution_scale_floor=np.asarray(DISTRIBUTION_SCALE_FLOOR),
        wall_seconds=np.asarray(elapsed),
        **response,
        **{f"diagnostic_{name}": value for name, value in diagnostic.items()},
    )
    with np.load(path, allow_pickle=False) as completed:
        validate_resume_archive(
            completed,
            cohort=cohort,
            method=method,
            model_id=model_id,
            fold=fold,
            seed=seed,
            source_lag=source_lag,
            particles=particles,
            horizons=horizons,
            source_window_frames=source_window_frames,
            history_lag=history_lag,
            base_seed=base_seed,
            requested_device=device,
            checkpoint=ckpt_path,
            expected_worm_indices=heldout,
        )
    del adapter
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {"status": "ok", "output": str(path), "wall_seconds": elapsed}


def _manifest(output: Path, source_run: Path, fold_file: Path, cohort, args) -> dict[str, Any]:
    spec = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": "corrected binary-stimulus multi-horizon prediction atlas v1",
        "methods": list(args.methods),
        "folds": list(args.folds),
        "seeds": list(args.seeds),
        "base_seed": int(args.base_seed),
        "episode_seed_definition": EPISODE_SEED_DEFINITION,
        "common_noise_definition": COMMON_NOISE_DEFINITION,
        "requested_device": str(args.device),
        "source_lag_frames": list(args.source_lags),
        "horizon_frames": list(args.horizons),
        "phases": list(PHASES),
        "response_keys": list(RESPONSE_KEYS),
        "particles": int(args.particles),
        "n_worms": int(cohort.n_worms),
        "n_neurons": int(cohort.n_neurons),
        "fps": float(cohort.fps),
        "history_frames": int(args.history_lag),
        "source_window_frames": int(args.source_window_frames),
        "lag_definition": "source-window end to prediction cut",
        "source_run": str(source_run),
        "source_run_manifest_sha256": sha256(source_run / "manifest.json"),
        "checkpoint_phase": args.checkpoint_phase,
        "model_id": args.model_id,
        "cohort_mode": args.cohort_mode,
        "fold_assignments": str(fold_file),
        "fold_assignments_sha256": sha256(fold_file),
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_schema": cohort.stimulus_schema_dict(),
        "stimulus_generator_encoding": GENERATOR_ENCODING,
        "chemical_identity_conditioned": False,
        "chemical_analysis_label": (
            "event-stratified under binary-any-stimulus generator; chemical identity "
            "was not conditioned on during training or sampling"
        ),
        "source_lag_seconds": [lag / cohort.fps for lag in args.source_lags],
        "horizon_seconds": [horizon / cohort.fps for horizon in args.horizons],
        "source_to_readout_seconds": {
            str(lag): [
                (lag + horizon) / cohort.fps for horizon in args.horizons
            ]
            for lag in args.source_lags
        },
        "responses_root": str((output / "responses").resolve()),
        "response_path_template": (
            "responses/{method}/{model_id}__{method}__ell{source_lag}__N{particles}"
            "__f{fold}__s{seed}.npz"
        ),
        "response_axes": list(RESPONSE_AXES),
        "matrix_internal_orientation": "source,horizon,target",
        "metric_definitions": METRIC_DEFINITIONS,
        "distribution_scale_floor": DISTRIBUTION_SCALE_FLOOR,
        "minimum_effective_sample_size": float(args.min_ess),
        "maximum_normalized_weight": RepairedResponseConfig.max_normalized_weight,
        "minimum_achieved_source_fraction": RepairedResponseConfig.min_achieved_fraction,
        "source_clamp_iqr_fraction": RepairedResponseConfig.epsilon_iqr_fraction,
        "anchor_lambda": RepairedResponseConfig.anchor_lambda,
        "anchor_rank": RepairedResponseConfig.anchor_rank,
        "resampling_ess_fraction": 0.50,
        "progressive_branch_factor": int(args.progressive_branch_factor),
        "progressive_future_branch_factor": int(args.progressive_future_branch_factor),
        "atlas_firewall": "no connectome or receptor atlas used in estimation or tuning",
        "claim_boundary": (
            "model-relative observed-law response; neither a causal intervention nor "
            "evidence of a physical transmission delay"
        ),
    }
    fingerprint = canonical_fingerprint(spec)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_spec_fingerprint": fingerprint,
        **spec,
    }


def _write_or_validate_manifest(path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text())
        existing_spec = {
            key: value
            for key, value in existing.items()
            if key not in {"created_utc", "run_spec_fingerprint"}
        }
        recomputed = canonical_fingerprint(existing_spec)
        if existing.get("run_spec_fingerprint") != recomputed:
            raise RuntimeError("existing output manifest fails its own fingerprint")
        if recomputed != candidate["run_spec_fingerprint"]:
            raise RuntimeError("existing output manifest differs from requested run")
        return existing
    _atomic_json(path, candidate)
    return candidate


def argument_parser() -> argparse.ArgumentParser:
    """Build CLI defaults for the frozen primary progressive screen.

    Direct importance uses N=256 in a separate invocation because one global
    ``--particles`` value cannot represent both frozen estimator settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--fold-file", type=Path)
    parser.add_argument("--checkpoint-phase", default="chemical_full_cv")
    parser.add_argument(
        "--model-id",
        default="stim_binary_any_stimulus_tcn_flow128_dropout15_jitter_p01",
    )
    parser.add_argument("--cohort-mode", default="oh16230_head")
    parser.add_argument("--history-lag", type=int, default=80)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903])
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=[FROZEN_DEFAULT_METHOD]
    )
    parser.add_argument("--source-lags", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--source-window-frames", type=int, default=4)
    parser.add_argument("--particles", type=int, default=FROZEN_PROGRESSIVE_PARTICLES)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base-seed", type=int, default=20260829)
    parser.add_argument("--min-ess", type=float, default=6.0)
    parser.add_argument("--progressive-branch-factor", type=int, default=2)
    parser.add_argument(
        "--progressive-future-branch-factor",
        type=int,
        default=FROZEN_PROGRESSIVE_FUTURE_BRANCH_FACTOR,
    )
    return parser


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()

    args.methods = tuple(dict.fromkeys(args.methods))
    args.folds = tuple(sorted(set(args.folds)))
    args.seeds = tuple(sorted(set(args.seeds)))
    args.source_lags = tuple(sorted(set(args.source_lags)))
    args.horizons = tuple(sorted(set(args.horizons)))
    if not args.methods or not args.folds or not args.seeds:
        parser.error("methods, folds, and seeds must be nonempty")
    if not set(args.folds).issubset(range(5)):
        parser.error("folds must be selected from 0..4")
    if min(args.source_lags) < 0 or min(args.horizons) < 1:
        parser.error("source lags must be nonnegative and horizons positive")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_run = args.source_run.resolve()
    source_manifest_path = source_run / "manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    cohort = load_cohort(cohort_mode=args.cohort_mode)
    if source_manifest.get("cohort_mode") != args.cohort_mode:
        raise RuntimeError("source-run cohort does not match requested cohort")
    if (
        source_manifest_stimulus_fingerprint(source_manifest)
        != cohort.stimulus_schema_fingerprint
    ):
        raise RuntimeError("source-run stimulus schema does not match loaded cohort")
    fold_file = (
        args.fold_file.resolve()
        if args.fold_file is not None
        else Path(source_manifest["fold_assignments"]).resolve()
    )
    if source_manifest.get("fold_assignments_sha256") != sha256(fold_file):
        raise RuntimeError("fold assignment hash differs from source-run manifest")
    folds = load_folds(fold_file, cohort)
    manifest = _manifest(output, source_run, fold_file, cohort, args)
    _write_or_validate_manifest(output / "manifest.json", manifest)

    records: list[dict[str, object]] = []
    for method in args.methods:
        for source_lag in args.source_lags:
            for fold in args.folds:
                for seed in args.seeds:
                    print(
                        f"ATLAS_START method={method} ell={source_lag} "
                        f"fold={fold} seed={seed}",
                        flush=True,
                    )
                    try:
                        result = run_one(
                            cohort=cohort,
                            folds=folds,
                            source_run=source_run,
                            checkpoint_phase=args.checkpoint_phase,
                            output=output,
                            method=method,
                            model_id=args.model_id,
                            history_lag=args.history_lag,
                            fold=fold,
                            seed=seed,
                            source_lag=source_lag,
                            particles=args.particles,
                            horizons=args.horizons,
                            source_window_frames=args.source_window_frames,
                            device=args.device,
                            base_seed=args.base_seed,
                            min_ess=args.min_ess,
                            progressive_branch_factor=args.progressive_branch_factor,
                            progressive_future_branch_factor=args.progressive_future_branch_factor,
                        )
                    except Exception as error:
                        result = {
                            "status": "failed",
                            "error": repr(error),
                            "wall_seconds": 0.0,
                        }
                    record = {
                        "method": method,
                        "source_lag_frames": source_lag,
                        "source_lag_seconds": source_lag / cohort.fps,
                        "fold": fold,
                        "seed": seed,
                        **result,
                    }
                    records.append(record)
                    _atomic_csv(output / "run_status.csv", pd.DataFrame(records))
                    print(
                        f"ATLAS_DONE method={method} ell={source_lag} "
                        f"fold={fold} seed={seed} status={result['status']} "
                        f"seconds={result.get('wall_seconds', 0):.1f}",
                        flush=True,
                    )
    failures = [row for row in records if row["status"] == "failed"]
    expected_runs = len(args.methods) * len(args.source_lags) * len(args.folds) * len(args.seeds)
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "failed",
        "run_spec_fingerprint": manifest["run_spec_fingerprint"],
        "expected_runs": expected_runs,
        "completed_or_skipped": sum(
            row["status"] in {"ok", "skipped"} for row in records
        ),
        "failed": len(failures),
        "failures": failures,
    }
    _atomic_json(output / "validation.json", validation)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
