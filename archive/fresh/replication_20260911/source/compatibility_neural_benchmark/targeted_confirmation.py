"""Targeted progressive-bridge confirmation for atlas hypotheses.

This runner deliberately evaluates only source neurons promoted in a frozen
``hypothesis_queue.csv``.  Low and high repaired histories retain the exact
progressive-bridge source-window and factual-anchor semantics.  A third arm is
rolled from the observed history at the cut, with matched flow base noise
across low, high, and factual futures.

Chemical labels are saved as event metadata only.  The frozen production
generator is binary-any-stimulus and therefore does not condition on chemical
identity.  Outputs describe contrasts under the learned observed-data law;
they are not causal effects, synapses, receptor actions, or physical delays.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
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
    standardize_for_checkpoint,
    stimulus_for_trace,
)
from compatibility_neural_benchmark.prediction_atlas_runner import (
    GENERATOR_ENCODING,
    EPISODE_SEED_DEFINITION,
    PHASES,
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _validate_checkpoint,
    canonical_fingerprint,
    checkpoint_path,
    draw_seed,
    expected_episode_metadata,
    load_folds,
    sha256,
    source_manifest_stimulus_fingerprint,
    timing_metadata,
    training_context,
    validate_forecast_bounds,
)
from compatibility_neural_benchmark.progressive_smc import (
    progressive_smc_targeted_confirmation,
)
from conditional_neural_benchmark.data import load_cohort


ARCHIVE_SCHEMA_VERSION = "prediction_atlas_targeted_confirmation_v2"
MANIFEST_SCHEMA_VERSION = "prediction_atlas_targeted_confirmation_manifest_v2"
METHOD = "progressive_bridge_smc_targeted_three_arm"
ARM_NAMES = ("low", "high", "factual")
CONTRAST_NAMES = ("high_low", "high_factual", "low_factual")
SIGNED_METRICS = (
    "endpoint_mean",
    "time_average_mean",
    "pathwise_peak_mean",
    "crossing_probability",
    "endpoint_log_sd",
)
ARM_METRICS = (
    "endpoint_mean",
    "time_average_mean",
    "pathwise_peak_mean",
    "crossing_probability",
    "endpoint_sd",
)
SIGNED_RESPONSE_KEYS = tuple(
    f"response_{contrast}_{metric}"
    for contrast in CONTRAST_NAMES
    for metric in SIGNED_METRICS
)
DISTANCE_RESPONSE_KEYS = tuple(
    f"distance_{contrast}_endpoint_wasserstein1"
    for contrast in CONTRAST_NAMES
)
ARM_RESPONSE_KEYS = tuple(f"arm_{metric}" for metric in ARM_METRICS)
CLAIM_BOUNDARY = (
    "model-relative contrasts under the learned observed-data law; neither a "
    "causal intervention, anatomical connection, receptor action, nor physical delay"
)
FACTUAL_ARM_DEFINITION = (
    "free conditional-flow rollout from the observed history at the cut"
)
COMMON_NOISE_DEFINITION = (
    "identical flow base-noise seeds and row order across low/high/factual future arms"
)


@dataclass(frozen=True)
class ConfirmationGroup:
    source_lag_frames: int
    source_indices: tuple[int, ...]
    horizon_frames: tuple[int, ...]
    phases: tuple[str, ...]
    selection_fingerprint: str

    def to_dict(self, neurons: tuple[str, ...]) -> dict[str, Any]:
        return {
            "source_lag_frames": self.source_lag_frames,
            "source_indices": list(self.source_indices),
            "source_neurons": [neurons[index] for index in self.source_indices],
            "horizon_frames": list(self.horizon_frames),
            "phases": list(self.phases),
            "selection_fingerprint": self.selection_fingerprint,
        }


def _truthy_selection(values: pd.Series) -> np.ndarray:
    text = values.astype(str).str.strip().str.lower()
    return text.isin(
        {
            "1",
            "true",
            "yes",
            "selected",
            "approved",
            "promoted",
            "confirm",
            "confirmation",
            "queued",
        }
    ).to_numpy()


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _phases_for_queue_context(
    value: object,
    default_phases: tuple[str, ...],
) -> tuple[str, ...]:
    """Map atlas analysis contexts back to the sampled episode phases."""
    if pd.isna(value):
        return default_phases
    context = str(value).strip().lower()
    if context in PHASES:
        return (context,)
    if context == "state_average":
        return default_phases
    if context == "onset_minus_baseline" or context.endswith(
        "_onset_minus_baseline"
    ):
        return ("baseline", "onset")
    if context.endswith("_onset"):
        return ("onset",)
    raise RuntimeError(f"hypothesis queue has an unsupported context {context!r}")


def load_candidate_groups(
    path: Path,
    neurons: tuple[str, ...],
    *,
    default_lags: tuple[int, ...] = (1, 4, 8, 16),
    default_horizons: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    default_phases: tuple[str, ...] = PHASES,
) -> tuple[ConfirmationGroup, ...]:
    """Resolve selected queue rows into bounded per-lag confirmation groups.

    Candidate-specific target columns are retained in the queue provenance but
    do not restrict the rollout: once a source is selected, all 54 targets are
    summarized so matrix orientation and unexpected downstream effects remain
    auditable.  Within a lag, requested horizons and phases are unioned across
    selected rows; this is a modest compute overhead that avoids fragmented
    archives while still scaling with selected sources rather than all sources.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError("hypothesis queue is empty")
    selection_column = _first_column(
        frame, ("run_confirmation", "selected", "queue_status", "status")
    )
    if selection_column is not None:
        frame = frame.loc[_truthy_selection(frame[selection_column])].copy()
    if frame.empty:
        raise RuntimeError("hypothesis queue contains no selected confirmation rows")

    source_column = _first_column(
        frame, ("source_neuron", "source", "perturbed_neuron")
    )
    if source_column is None:
        raise RuntimeError("hypothesis queue lacks a source-neuron column")
    neuron_to_index = {name: index for index, name in enumerate(neurons)}
    source_names = frame[source_column].astype(str).str.strip()
    unknown = sorted(set(source_names) - set(neuron_to_index))
    if unknown:
        raise RuntimeError(f"hypothesis queue contains unknown source neurons: {unknown}")
    frame["__source_index"] = source_names.map(neuron_to_index)

    lag_column = _first_column(frame, ("source_lag_frames", "lag_frames"))
    horizon_column = _first_column(frame, ("horizon_frames", "forecast_horizon_frames"))
    phase_column = _first_column(
        frame, ("phase", "stimulus_state", "state", "context")
    )
    default_lags = tuple(sorted(set(int(value) for value in default_lags)))
    default_horizons = tuple(sorted(set(int(value) for value in default_horizons)))
    default_phases = tuple(phase for phase in PHASES if phase in set(default_phases))
    if not default_lags or min(default_lags) < 0:
        raise ValueError("default source lags must be nonempty and nonnegative")
    if not default_horizons or min(default_horizons) < 1:
        raise ValueError("default horizons must be nonempty and positive")
    if not default_phases:
        raise ValueError("default phases must contain a supported stimulus state")

    rows: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        lags = default_lags if lag_column is None or pd.isna(row[lag_column]) else (
            int(row[lag_column]),
        )
        horizons = (
            default_horizons
            if horizon_column is None or pd.isna(row[horizon_column])
            else (int(row[horizon_column]),)
        )
        phases = (
            default_phases
            if phase_column is None
            else _phases_for_queue_context(row[phase_column], default_phases)
        )
        if min(lags) < 0 or min(horizons) < 1:
            raise RuntimeError(f"queue row {row_index} has invalid lag or horizon")
        if not set(phases).issubset(PHASES):
            raise RuntimeError(f"queue row {row_index} has an unsupported phase")
        for lag in lags:
            rows.append(
                {
                    "lag": int(lag),
                    "source": int(row["__source_index"]),
                    "horizons": horizons,
                    "phases": phases,
                }
            )

    groups: list[ConfirmationGroup] = []
    for lag in sorted({row["lag"] for row in rows}):
        current = [row for row in rows if row["lag"] == lag]
        sources = tuple(sorted({row["source"] for row in current}))
        if len(sources) >= len(neurons):
            raise RuntimeError(
                "targeted confirmation resolved to every source neuron; use the "
                "full-matrix screen instead"
            )
        horizons = tuple(sorted({value for row in current for value in row["horizons"]}))
        phases = tuple(
            phase
            for phase in PHASES
            if phase in {value for row in current for value in row["phases"]}
        )
        spec = {
            "queue_sha256": sha256(path),
            "source_lag_frames": lag,
            "source_indices": list(sources),
            "source_neurons": [neurons[index] for index in sources],
            "horizon_frames": list(horizons),
            "phases": list(phases),
        }
        groups.append(
            ConfirmationGroup(
                source_lag_frames=lag,
                source_indices=sources,
                horizon_frames=horizons,
                phases=phases,
                selection_fingerprint=canonical_fingerprint(spec),
            )
        )
    return tuple(groups)


def confirmation_output_path(
    output: Path,
    model_id: str,
    group: ConfirmationGroup,
    fold: int,
    seed: int,
    particles: int,
) -> Path:
    return (
        output
        / "responses"
        / (
            f"{model_id}__targeted_three_arm__ell{group.source_lag_frames}"
            f"__N{particles}__f{fold}__s{seed}"
            f"__sel{group.selection_fingerprint[:12]}.npz"
        )
    )


def orient_source_horizon_target(
    value: np.ndarray,
    *,
    source_count: int,
    horizon_count: int,
    target_count: int,
) -> np.ndarray:
    """Transpose once from sampler order to horizon/target-row/source-column."""
    array = np.asarray(value)
    expected = (source_count, horizon_count, target_count)
    if array.shape != expected:
        raise ValueError(
            f"source-major response shape is {array.shape}, expected {expected}"
        )
    return np.transpose(array, (1, 2, 0))


def _strict_provenance_match(
    archive,
    *,
    scalar_expected: dict[str, Any],
    array_expected: dict[str, np.ndarray],
) -> None:
    """Fail closed on a missing or mismatched provenance field."""
    required = set(scalar_expected).union(array_expected)
    missing = sorted(required.difference(archive.files))
    if missing:
        raise RuntimeError(
            "resume rejected: archive lacks provenance fields: " + ", ".join(missing)
        )
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
    for key, expected in array_expected.items():
        actual = np.asarray(archive[key])
        expected = np.asarray(expected)
        if np.issubdtype(expected.dtype, np.floating):
            equal = np.allclose(actual, expected, rtol=0.0, atol=1e-7)
        else:
            equal = np.array_equal(actual, expected)
        if not equal:
            raise RuntimeError(f"resume rejected: archive {key} differs")


def validate_confirmation_resume_archive(
    archive,
    *,
    cohort,
    model_id: str,
    fold: int,
    seed: int,
    group: ConfirmationGroup,
    particles: int,
    source_window_frames: int,
    history_lag: int,
    checkpoint: Path,
    expected_worm_indices: np.ndarray,
    hypothesis_queue: Path,
    endpoint_quantiles: tuple[float, ...],
    branch_factor: int,
    future_branch_factor: int,
    base_seed: int,
    requested_device: str,
    min_ess: float = 24.0,
) -> None:
    """Validate a complete archive before allowing a resume skip."""
    if not checkpoint.exists():
        raise RuntimeError("resume rejected: checkpoint no longer exists")
    heldout = np.asarray(expected_worm_indices, dtype=np.int64)
    all_cut_times, all_bounds = expected_episode_metadata(
        cohort,
        heldout,
        group.source_lag_frames,
        source_window_frames,
    )
    phase_indices = [PHASES.index(phase) for phase in group.phases]
    cut_times = all_cut_times[:, phase_indices]
    bounds = all_bounds[:, phase_indices]
    validate_forecast_bounds(cohort, heldout, cut_times, group.horizon_frames)
    timing = timing_metadata(
        group.source_lag_frames, group.horizon_frames, cohort.fps
    )
    scalar_expected = {
        "status": "complete",
        "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        "method": METHOD,
        "model_id": model_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "fold": int(fold),
        "seed": int(seed),
        "base_seed": int(base_seed),
        "episode_seed_definition": EPISODE_SEED_DEFINITION,
        "requested_device": str(requested_device),
        "history_frames": int(history_lag),
        "repair_frames": int(group.source_lag_frames + source_window_frames),
        "source_lag_frames": int(group.source_lag_frames),
        "source_lag_seconds": float(timing["source_lag_seconds"]),
        "lag_definition": "source-window end to prediction cut",
        "source_window_frames": int(source_window_frames),
        "source_window_bounds_semantics": "[inclusive_start,exclusive_stop)",
        "n_particles": int(particles),
        "progressive_branch_factor": int(branch_factor),
        "progressive_future_branch_factor": int(future_branch_factor),
        "minimum_effective_sample_size": float(min_ess),
        "maximum_normalized_weight": float(
            RepairedResponseConfig.max_normalized_weight
        ),
        "minimum_achieved_source_fraction": float(
            RepairedResponseConfig.min_achieved_fraction
        ),
        "fps": float(cohort.fps),
        "selection_fingerprint": group.selection_fingerprint,
        "hypothesis_queue": str(hypothesis_queue.resolve()),
        "hypothesis_queue_sha256": sha256(hypothesis_queue),
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_generator_encoding": GENERATOR_ENCODING,
        "chemical_identity_conditioned": False,
        "distribution_scale_floor": ENDPOINT_SD_FLOOR,
        "endpoint_sd_ddof": 0,
        "endpoint_quantile_method": "numpy_linear_empirical",
        "matrix_orientation": "target_row_source_column",
        "factual_arm_definition": FACTUAL_ARM_DEFINITION,
        "common_noise_definition": COMMON_NOISE_DEFINITION,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    array_expected = {
        "horizon_frames": timing["horizon_frames"],
        "horizon_seconds": timing["horizon_seconds"],
        "source_to_readout_seconds": timing["source_to_readout_seconds"],
        "phase_names": np.asarray(group.phases),
        "arm_names": np.asarray(ARM_NAMES),
        "signed_response_keys": np.asarray(SIGNED_RESPONSE_KEYS),
        "distance_response_keys": np.asarray(DISTANCE_RESPONSE_KEYS),
        "arm_response_keys": np.asarray(ARM_RESPONSE_KEYS),
        "signed_response_axes": np.asarray(
            ("heldout_worm", "phase", "event", "horizon", "target", "source")
        ),
        "arm_response_axes": np.asarray(
            (
                "heldout_worm",
                "phase",
                "event",
                "arm",
                "horizon",
                "target",
                "source",
            )
        ),
        "selected_source_indices": np.asarray(group.source_indices, dtype=np.int16),
        "selected_source_neurons": np.asarray(
            [cohort.neurons[index] for index in group.source_indices]
        ),
        "target_neurons": np.asarray(cohort.neurons),
        "endpoint_quantiles": np.asarray(endpoint_quantiles, dtype=np.float32),
        "worm_indices": heldout.astype(np.int16),
        "worm_ids": np.asarray([cohort.worm_ids[index] for index in heldout]),
        "cut_times": cut_times,
        "source_window_bounds": bounds,
        "chemical_code_by_worm_event": np.asarray(
            [
                cohort.stimulus_schedules[index].chemical_code_by_event
                for index in heldout
            ],
            dtype=np.int8,
        ),
        "chemical_name_by_worm_event": np.asarray(
            [
                cohort.stimulus_schedules[index].chemical_name_by_event
                for index in heldout
            ]
        ),
        "event_intervals_seconds_by_worm": np.asarray(
            [
                cohort.stimulus_schedules[index].event_intervals_seconds
                for index in heldout
            ],
            dtype=np.float32,
        ),
        "schedule_source_recording_by_worm": np.asarray(
            [cohort.stimulus_schedules[index].source_recording for index in heldout]
        ),
        "schedule_native_fps_by_worm": np.asarray(
            [cohort.stimulus_schedules[index].native_fps for index in heldout],
            dtype=np.float32,
        ),
        "schedule_analysis_fps_by_worm": np.asarray(
            [cohort.stimulus_schedules[index].analysis_fps for index in heldout],
            dtype=np.float32,
        ),
        "schedule_resampling_provenance_by_worm": np.asarray(
            [
                cohort.stimulus_schedules[index].resampling_provenance
                for index in heldout
            ]
        ),
    }
    _strict_provenance_match(
        archive, scalar_expected=scalar_expected, array_expected=array_expected
    )
    if "resolved_device" not in archive.files:
        raise RuntimeError("resume rejected: archive lacks resolved_device")
    resolved_device = str(archive["resolved_device"].item())
    if not resolved_device or (
        str(requested_device) != "auto" and resolved_device != str(requested_device)
    ):
        raise RuntimeError("resume rejected: archive resolved_device differs")

    p, h, d, m = len(group.phases), len(group.horizon_frames), cohort.n_neurons, len(
        group.source_indices
    )
    signed_shape = (len(heldout), p, 3, h, d, m)
    arm_shape = (len(heldout), p, 3, len(ARM_NAMES), h, d, m)
    required_responses = set(SIGNED_RESPONSE_KEYS).union(
        DISTANCE_RESPONSE_KEYS, ARM_RESPONSE_KEYS
    )
    if endpoint_quantiles:
        required_responses.add("arm_endpoint_quantile")
        required_responses.update(
            f"response_{contrast}_endpoint_quantile"
            for contrast in CONTRAST_NAMES
        )
    missing = sorted(required_responses.difference(archive.files))
    if missing:
        raise RuntimeError(
            "resume rejected: archive lacks response fields: " + ", ".join(missing)
        )
    for key in SIGNED_RESPONSE_KEYS + DISTANCE_RESPONSE_KEYS:
        value = np.asarray(archive[key])
        if value.shape != signed_shape or not np.isfinite(value).all():
            raise RuntimeError(f"resume rejected: archive {key} shape/finiteness differs")
    for key in ARM_RESPONSE_KEYS:
        value = np.asarray(archive[key])
        if value.shape != arm_shape or not np.isfinite(value).all():
            raise RuntimeError(f"resume rejected: archive {key} shape/finiteness differs")
    for key in DISTANCE_RESPONSE_KEYS:
        if np.any(np.asarray(archive[key]) < -1e-7):
            raise RuntimeError(f"resume rejected: archive {key} is negative")
    for contrast in CONTRAST_NAMES:
        crossing = np.asarray(
            archive[f"response_{contrast}_crossing_probability"]
        )
        if np.any(crossing < -1.0 - 1e-6) or np.any(crossing > 1.0 + 1e-6):
            raise RuntimeError("resume rejected: a crossing contrast is out of range")
    arm_crossing = np.asarray(archive["arm_crossing_probability"])
    if np.any(arm_crossing < -1e-7) or np.any(arm_crossing > 1.0 + 1e-7):
        raise RuntimeError("resume rejected: an arm crossing probability is out of range")
    if endpoint_quantiles:
        arm_quantile_shape = (
            len(heldout), p, 3, len(ARM_NAMES), h, len(endpoint_quantiles), d, m
        )
        contrast_quantile_shape = (
            len(heldout), p, 3, h, len(endpoint_quantiles), d, m
        )
        arm_quantile = np.asarray(archive["arm_endpoint_quantile"])
        if arm_quantile.shape != arm_quantile_shape or not np.isfinite(
            arm_quantile
        ).all():
            raise RuntimeError("resume rejected: arm endpoint quantile shape differs")
        for contrast in CONTRAST_NAMES:
            key = f"response_{contrast}_endpoint_quantile"
            value = np.asarray(archive[key])
            if value.shape != contrast_quantile_shape or not np.isfinite(value).all():
                raise RuntimeError(f"resume rejected: archive {key} shape differs")

    diagnostic_shape = (len(heldout), p, 3, m)
    required_diagnostics = (
        "target_low",
        "target_high",
        "target_gap",
        "achieved_low",
        "achieved_high",
        "achieved_gap",
        "ess_low",
        "ess_high",
        "max_weight_low",
        "max_weight_high",
        "valid",
        "factual_source",
        "selected_source_index",
        "endpoint_sd_floor",
    )
    for name in required_diagnostics:
        key = f"diagnostic_{name}"
        if key not in archive.files:
            raise RuntimeError(f"resume rejected: archive lacks {key}")
        value = np.asarray(archive[key])
        if value.shape != diagnostic_shape or not np.isfinite(value).all():
            raise RuntimeError(f"resume rejected: archive {key} shape/finiteness differs")
    if not np.array_equal(
        archive["diagnostic_selected_source_index"],
        np.broadcast_to(
            np.asarray(group.source_indices, dtype=np.float32), diagnostic_shape
        ),
    ):
        raise RuntimeError("resume rejected: selected-source diagnostic differs")
    if not np.allclose(
        archive["diagnostic_endpoint_sd_floor"], ENDPOINT_SD_FLOOR, rtol=0, atol=0
    ):
        raise RuntimeError("resume rejected: endpoint-SD floor diagnostic differs")
    if np.any(archive["diagnostic_target_gap"] <= 0):
        raise RuntimeError("resume rejected: target source gap is not positive")
    target_gap = np.asarray(archive["diagnostic_target_gap"], dtype=np.float64)
    achieved_gap = np.asarray(
        archive["diagnostic_achieved_gap"], dtype=np.float64
    )
    if not np.allclose(
        target_gap,
        np.asarray(archive["diagnostic_target_high"], dtype=np.float64)
        - np.asarray(archive["diagnostic_target_low"], dtype=np.float64),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("resume rejected: target-gap identity failed")
    if not np.allclose(
        achieved_gap,
        np.asarray(archive["diagnostic_achieved_high"], dtype=np.float64)
        - np.asarray(archive["diagnostic_achieved_low"], dtype=np.float64),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("resume rejected: achieved-gap identity failed")
    valid = np.asarray(archive["diagnostic_valid"])
    if not np.all(np.isin(valid, (0.0, 1.0))):
        raise RuntimeError("resume rejected: support-valid diagnostic is not binary")
    ess_low = np.asarray(archive["diagnostic_ess_low"], dtype=np.float64)
    ess_high = np.asarray(archive["diagnostic_ess_high"], dtype=np.float64)
    max_low = np.asarray(
        archive["diagnostic_max_weight_low"], dtype=np.float64
    )
    max_high = np.asarray(
        archive["diagnostic_max_weight_high"], dtype=np.float64
    )
    for name, value in (("ess_low", ess_low), ("ess_high", ess_high)):
        if np.any(value < 1.0 / branch_factor - 1e-5) or np.any(
            value > particles + 1e-4
        ):
            raise RuntimeError(f"resume rejected: {name} is out of range")
    for name, value in (("max_weight_low", max_low), ("max_weight_high", max_high)):
        if np.any(value < 1.0 / particles - 1e-6) or np.any(
            value > branch_factor + 1e-6
        ):
            raise RuntimeError(f"resume rejected: {name} is out of range")
    expected_valid = (
        (ess_low >= float(min_ess))
        & (ess_high >= float(min_ess))
        & (max_low <= float(RepairedResponseConfig.max_normalized_weight))
        & (max_high <= float(RepairedResponseConfig.max_normalized_weight))
        & (
            achieved_gap
            >= float(RepairedResponseConfig.min_achieved_fraction) * target_gap
        )
    )
    if not np.array_equal(valid.astype(bool), expected_valid):
        raise RuntimeError(
            "resume rejected: diagnostic_valid does not match declared gates"
        )

    for name in ("distinct_ancestors_low", "distinct_ancestors_high"):
        key = f"diagnostic_{name}"
        if key not in archive.files:
            raise RuntimeError(f"resume rejected: archive lacks {key}")
        value = np.asarray(archive[key])
        if value.shape != diagnostic_shape or np.any(value < 1) or np.any(
            value > particles
        ) or not np.allclose(value, np.round(value), rtol=0, atol=0):
            raise RuntimeError(f"resume rejected: archive {key} is out of range")
    step_shape = (*diagnostic_shape, group.source_lag_frames + source_window_frames)
    for name in (
        "step_ess_low",
        "step_ess_high",
        "step_max_weight_low",
        "step_max_weight_high",
        "step_forced_tempering_low",
        "step_forced_tempering_high",
    ):
        key = f"diagnostic_{name}"
        if key not in archive.files:
            raise RuntimeError(f"resume rejected: archive lacks {key}")
        value = np.asarray(archive[key])
        if value.shape != step_shape or not np.isfinite(value).all():
            raise RuntimeError(f"resume rejected: archive {key} shape/finiteness differs")
        if name.startswith("step_ess") and (
            np.any(value < 1.0 / branch_factor - 1e-5)
            or np.any(value > particles + 1e-4)
        ):
            raise RuntimeError(f"resume rejected: archive {key} is out of range")
        if name.startswith("step_max_weight") and (
            np.any(value < 1.0 / particles - 1e-6)
            or np.any(value > branch_factor + 1e-6)
        ):
            raise RuntimeError(f"resume rejected: archive {key} is out of range")
        if name.startswith("step_forced_tempering") and not np.all(
            np.isin(value, (0.0, 1.0))
        ):
            raise RuntimeError(f"resume rejected: archive {key} is not binary")

    # Fail closed on every saved support diagnostic, not just the minimum
    # compatibility subset enumerated above.
    allowed_diagnostic_shapes = {diagnostic_shape, step_shape}
    for key in archive.files:
        if not key.startswith("diagnostic_"):
            continue
        value = np.asarray(archive[key])
        if value.shape not in allowed_diagnostic_shapes or not np.isfinite(value).all():
            raise RuntimeError(
                f"resume rejected: archive {key} has invalid shape/finiteness"
            )


def run_one(
    *,
    cohort,
    folds: np.ndarray,
    source_run: Path,
    checkpoint_phase: str,
    output: Path,
    model_id: str,
    history_lag: int,
    fold: int,
    seed: int,
    group: ConfirmationGroup,
    particles: int = 128,
    source_window_frames: int = 4,
    device: str = "mps",
    base_seed: int = 20260829,
    min_ess: float = 24.0,
    branch_factor: int = 2,
    future_branch_factor: int = 2,
    endpoint_quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
    hypothesis_queue: Path,
) -> dict[str, object]:
    """Run one fold/checkpoint/lag archive for a selected source subset."""
    if len(group.source_indices) >= cohort.n_neurons:
        raise RuntimeError("targeted confirmation may not run every source neuron")
    path = confirmation_output_path(output, model_id, group, fold, seed, particles)
    checkpoint = checkpoint_path(
        source_run, checkpoint_phase, model_id, history_lag, fold, seed
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    heldout = np.flatnonzero(np.asarray(folds) == fold)
    if not len(heldout):
        raise RuntimeError(f"fold {fold} has no held-out worms")
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            validate_confirmation_resume_archive(
                existing,
                cohort=cohort,
                model_id=model_id,
                fold=fold,
                seed=seed,
                group=group,
                particles=particles,
                source_window_frames=source_window_frames,
                history_lag=history_lag,
                checkpoint=checkpoint,
                expected_worm_indices=heldout,
                hypothesis_queue=hypothesis_queue,
                endpoint_quantiles=endpoint_quantiles,
                branch_factor=branch_factor,
                future_branch_factor=future_branch_factor,
                base_seed=base_seed,
                requested_device=device,
                min_ess=min_ess,
            )
        return {"status": "skipped", "output": str(path), "wall_seconds": 0.0}

    all_cut_times, all_bounds = expected_episode_metadata(
        cohort, heldout, group.source_lag_frames, source_window_frames
    )
    phase_indices = [PHASES.index(phase) for phase in group.phases]
    cut_times = all_cut_times[:, phase_indices]
    bounds = all_bounds[:, phase_indices]
    validate_forecast_bounds(cohort, heldout, cut_times, group.horizon_frames)
    started = time.perf_counter()
    adapter = GeneratorAdapter.load(str(checkpoint), device=device)
    _validate_checkpoint(
        adapter,
        cohort=cohort,
        model_id=model_id,
        fold=fold,
        seed=seed,
        history_lag=history_lag,
    )
    config = RepairedResponseConfig(
        history_frames=history_lag,
        repair_frames=group.source_lag_frames + source_window_frames,
        source_window_frames=source_window_frames,
        source_lag_frames=group.source_lag_frames,
        horizon_frames=group.horizon_frames,
        n_particles=particles,
        min_ess=min_ess,
        resample_ess_fraction=0.50,
        sampling_chunk_size=1024,
    )
    config.validate()
    projection, source_quantiles, thresholds = training_context(
        cohort, folds, fold, adapter.checkpoint, config
    )
    w, p, h, d, m = (
        len(heldout),
        len(group.phases),
        len(group.horizon_frames),
        cohort.n_neurons,
        len(group.source_indices),
    )
    signed_shape = (w, p, 3, h, d, m)
    arm_shape = (w, p, 3, len(ARM_NAMES), h, d, m)
    response: dict[str, np.ndarray] = {
        key: np.full(signed_shape, np.nan, dtype=np.float32)
        for key in SIGNED_RESPONSE_KEYS + DISTANCE_RESPONSE_KEYS
    }
    response.update(
        {
            key: np.full(arm_shape, np.nan, dtype=np.float32)
            for key in ARM_RESPONSE_KEYS
        }
    )
    if endpoint_quantiles:
        response["arm_endpoint_quantile"] = np.full(
            (w, p, 3, len(ARM_NAMES), h, len(endpoint_quantiles), d, m),
            np.nan,
            dtype=np.float32,
        )
        for contrast in CONTRAST_NAMES:
            response[f"response_{contrast}_endpoint_quantile"] = np.full(
                (w, p, 3, h, len(endpoint_quantiles), d, m),
                np.nan,
                dtype=np.float32,
            )
    diagnostic_shapes: dict[str, tuple[int, ...]] | None = None
    diagnostics: dict[str, np.ndarray] = {}

    for worm_position, worm in enumerate(heldout):
        standardized = causal_fill(
            standardize_for_checkpoint(cohort.traces[int(worm)], adapter.checkpoint)
        )
        stimulus = stimulus_for_trace(
            cohort.traces[int(worm)], cohort, int(worm), adapter.checkpoint
        )
        cuts = episode_cuts(
            len(cohort.traces[int(worm)]),
            cohort.stimulus_schedules[int(worm)],
            source_window_frames,
        )
        for cut in cuts:
            if cut.phase not in group.phases:
                continue
            phase_position = group.phases.index(cut.phase)
            keyed_seed = draw_seed(
                base_seed,
                model_id,
                fold,
                seed,
                int(worm),
                PHASES.index(cut.phase),
                cut.event,
                group.source_lag_frames,
            )
            result = progressive_smc_targeted_confirmation(
                adapter,
                standardized,
                stimulus,
                cut_time=cut.time,
                projection=projection,
                source_low=source_quantiles[cut.phase]["low"],
                source_high=source_quantiles[cut.phase]["high"],
                source_iqr=source_quantiles[cut.phase]["iqr"],
                thresholds=thresholds,
                source_indices=np.asarray(group.source_indices, dtype=np.int64),
                config=config,
                seed=keyed_seed,
                branch_factor=branch_factor,
                future_branch_factor=future_branch_factor,
                endpoint_quantiles=endpoint_quantiles,
            )
            for key in SIGNED_RESPONSE_KEYS + DISTANCE_RESPONSE_KEYS:
                value = np.asarray(result[key], dtype=np.float32)
                if value.shape != (m, h, d) or not np.isfinite(value).all():
                    raise RuntimeError(f"invalid targeted metric {key}")
                response[key][worm_position, phase_position, cut.event] = (
                    orient_source_horizon_target(
                        value,
                        source_count=m,
                        horizon_count=h,
                        target_count=d,
                    )
                )
            for key in ARM_RESPONSE_KEYS:
                value = np.asarray(result[key], dtype=np.float32)
                if value.shape != (len(ARM_NAMES), m, h, d):
                    raise RuntimeError(f"invalid targeted arm metric {key}")
                response[key][worm_position, phase_position, cut.event] = np.transpose(
                    value, (0, 2, 3, 1)
                )
            if endpoint_quantiles:
                arm_quantile = np.asarray(result["arm_endpoint_quantile"])
                response["arm_endpoint_quantile"][
                    worm_position, phase_position, cut.event
                ] = np.transpose(arm_quantile, (0, 2, 3, 4, 1))
                for contrast in CONTRAST_NAMES:
                    key = f"response_{contrast}_endpoint_quantile"
                    response[key][worm_position, phase_position, cut.event] = np.transpose(
                        np.asarray(result[key]), (1, 2, 3, 0)
                    )

            current_shapes = {
                key.removeprefix("diagnostic_"): tuple(np.asarray(value).shape)
                for key, value in result.items()
                if key.startswith("diagnostic_") and np.asarray(value).ndim in {1, 2}
            }
            if diagnostic_shapes is None:
                diagnostic_shapes = current_shapes
                diagnostics = {
                    name: np.full((w, p, 3, *shape), np.nan, dtype=np.float32)
                    for name, shape in current_shapes.items()
                }
            if current_shapes != diagnostic_shapes:
                raise RuntimeError("targeted diagnostic schema changed within a run")
            for name, shape in diagnostic_shapes.items():
                value = np.asarray(result[f"diagnostic_{name}"], dtype=np.float32)
                if value.shape != shape or not np.isfinite(value).all():
                    raise RuntimeError(f"invalid targeted diagnostic {name}")
                diagnostics[name][worm_position, phase_position, cut.event] = value
        print(
            f"TARGETED_CONFIRMATION_WORM_DONE ell={group.source_lag_frames} "
            f"fold={fold} heldout_worm_index={int(worm)} sources={m}",
            flush=True,
        )

    for key, value in response.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"one or more targeted cells were not evaluated for {key}")
    for name, value in diagnostics.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"one or more targeted diagnostics are incomplete: {name}")
    elapsed = time.perf_counter() - started
    timing = timing_metadata(
        group.source_lag_frames, group.horizon_frames, cohort.fps
    )
    _atomic_npz(
        path,
        status=np.asarray("complete"),
        archive_schema_version=np.asarray(ARCHIVE_SCHEMA_VERSION),
        created_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        method=np.asarray(METHOD),
        model_id=np.asarray(model_id),
        checkpoint=np.asarray(str(checkpoint.resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        fold=np.asarray(fold),
        seed=np.asarray(seed),
        base_seed=np.asarray(base_seed),
        episode_seed_definition=np.asarray(EPISODE_SEED_DEFINITION),
        requested_device=np.asarray(str(device)),
        resolved_device=np.asarray(str(adapter.device)),
        history_frames=np.asarray(history_lag),
        repair_frames=np.asarray(config.repair_frames),
        source_lag_frames=np.asarray(group.source_lag_frames),
        source_lag_seconds=np.asarray(timing["source_lag_seconds"]),
        lag_definition=np.asarray(timing["lag_definition"]),
        source_window_frames=np.asarray(source_window_frames),
        source_window_bounds_semantics=np.asarray("[inclusive_start,exclusive_stop)"),
        n_particles=np.asarray(particles),
        progressive_branch_factor=np.asarray(branch_factor),
        progressive_future_branch_factor=np.asarray(future_branch_factor),
        minimum_effective_sample_size=np.asarray(float(min_ess)),
        maximum_normalized_weight=np.asarray(
            float(RepairedResponseConfig.max_normalized_weight)
        ),
        minimum_achieved_source_fraction=np.asarray(
            float(RepairedResponseConfig.min_achieved_fraction)
        ),
        horizon_frames=timing["horizon_frames"],
        horizon_seconds=timing["horizon_seconds"],
        source_to_readout_seconds=timing["source_to_readout_seconds"],
        fps=np.asarray(cohort.fps),
        phase_names=np.asarray(group.phases),
        arm_names=np.asarray(ARM_NAMES),
        signed_response_keys=np.asarray(SIGNED_RESPONSE_KEYS),
        distance_response_keys=np.asarray(DISTANCE_RESPONSE_KEYS),
        arm_response_keys=np.asarray(ARM_RESPONSE_KEYS),
        signed_response_axes=np.asarray(
            ("heldout_worm", "phase", "event", "horizon", "target", "source")
        ),
        arm_response_axes=np.asarray(
            (
                "heldout_worm",
                "phase",
                "event",
                "arm",
                "horizon",
                "target",
                "source",
            )
        ),
        matrix_orientation=np.asarray("target_row_source_column"),
        selected_source_indices=np.asarray(group.source_indices, dtype=np.int16),
        selected_source_neurons=np.asarray(
            [cohort.neurons[index] for index in group.source_indices]
        ),
        target_neurons=np.asarray(cohort.neurons),
        endpoint_quantiles=np.asarray(endpoint_quantiles, dtype=np.float32),
        selection_fingerprint=np.asarray(group.selection_fingerprint),
        hypothesis_queue=np.asarray(str(hypothesis_queue.resolve())),
        hypothesis_queue_sha256=np.asarray(sha256(hypothesis_queue)),
        worm_indices=heldout.astype(np.int16),
        worm_ids=np.asarray([cohort.worm_ids[int(index)] for index in heldout]),
        cut_times=cut_times,
        source_window_bounds=bounds,
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
        event_intervals_seconds_by_worm=np.asarray(
            [
                cohort.stimulus_schedules[int(index)].event_intervals_seconds
                for index in heldout
            ],
            dtype=np.float32,
        ),
        schedule_source_recording_by_worm=np.asarray(
            [
                cohort.stimulus_schedules[int(index)].source_recording
                for index in heldout
            ]
        ),
        schedule_native_fps_by_worm=np.asarray(
            [
                cohort.stimulus_schedules[int(index)].native_fps
                for index in heldout
            ],
            dtype=np.float32,
        ),
        schedule_analysis_fps_by_worm=np.asarray(
            [
                cohort.stimulus_schedules[int(index)].analysis_fps
                for index in heldout
            ],
            dtype=np.float32,
        ),
        schedule_resampling_provenance_by_worm=np.asarray(
            [
                cohort.stimulus_schedules[int(index)].resampling_provenance
                for index in heldout
            ]
        ),
        distribution_scale_floor=np.asarray(ENDPOINT_SD_FLOOR),
        endpoint_sd_ddof=np.asarray(0),
        endpoint_quantile_method=np.asarray("numpy_linear_empirical"),
        factual_arm_definition=np.asarray(FACTUAL_ARM_DEFINITION),
        common_noise_definition=np.asarray(COMMON_NOISE_DEFINITION),
        claim_boundary=np.asarray(CLAIM_BOUNDARY),
        wall_seconds=np.asarray(elapsed),
        **response,
        **{f"diagnostic_{name}": value for name, value in diagnostics.items()},
    )
    with np.load(path, allow_pickle=False) as completed:
        validate_confirmation_resume_archive(
            completed,
            cohort=cohort,
            model_id=model_id,
            fold=fold,
            seed=seed,
            group=group,
            particles=particles,
            source_window_frames=source_window_frames,
            history_lag=history_lag,
            checkpoint=checkpoint,
            expected_worm_indices=heldout,
            hypothesis_queue=hypothesis_queue,
            endpoint_quantiles=endpoint_quantiles,
            branch_factor=branch_factor,
            future_branch_factor=future_branch_factor,
            base_seed=base_seed,
            requested_device=device,
            min_ess=min_ess,
        )
    del adapter
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {"status": "ok", "output": str(path), "wall_seconds": elapsed}


def _manifest(
    *,
    source_run: Path,
    fold_file: Path,
    hypothesis_queue: Path,
    cohort,
    groups: tuple[ConfirmationGroup, ...],
    args,
) -> dict[str, Any]:
    spec = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": "targeted progressive-bridge low/high/factual confirmation v1",
        "method": METHOD,
        "model_id": args.model_id,
        "checkpoint_phase": args.checkpoint_phase,
        "source_run": str(source_run),
        "source_run_manifest_sha256": sha256(source_run / "manifest.json"),
        "fold_assignments": str(fold_file),
        "fold_assignments_sha256": sha256(fold_file),
        "hypothesis_queue": str(hypothesis_queue),
        "hypothesis_queue_sha256": sha256(hypothesis_queue),
        "resolved_groups": [group.to_dict(cohort.neurons) for group in groups],
        "folds": list(args.folds),
        "seeds": list(args.seeds),
        "base_seed": int(args.base_seed),
        "episode_seed_definition": EPISODE_SEED_DEFINITION,
        "requested_device": str(args.device),
        "particles": args.particles,
        "history_frames": args.history_lag,
        "source_window_frames": args.source_window_frames,
        "endpoint_quantiles": list(args.endpoint_quantiles),
        "phases_available": list(PHASES),
        "arm_names": list(ARM_NAMES),
        "signed_response_keys": list(SIGNED_RESPONSE_KEYS),
        "distance_response_keys": list(DISTANCE_RESPONSE_KEYS),
        "arm_response_keys": list(ARM_RESPONSE_KEYS),
        "matrix_orientation": "target_row_source_column",
        "progressive_branch_factor": args.progressive_branch_factor,
        "progressive_future_branch_factor": args.progressive_future_branch_factor,
        "minimum_effective_sample_size": args.min_ess,
        "maximum_normalized_weight": float(
            RepairedResponseConfig.max_normalized_weight
        ),
        "minimum_achieved_source_fraction": float(
            RepairedResponseConfig.min_achieved_fraction
        ),
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_generator_encoding": GENERATOR_ENCODING,
        "chemical_identity_conditioned": False,
        "chemical_analysis_label": (
            "event-stratified under binary-any-stimulus generator; chemical identity "
            "was not conditioned on during training or sampling"
        ),
        "distribution_scale_floor": ENDPOINT_SD_FLOOR,
        "endpoint_sd_ddof": 0,
        "endpoint_quantile_method": "numpy_linear_empirical",
        "factual_arm_definition": FACTUAL_ARM_DEFINITION,
        "common_noise_definition": COMMON_NOISE_DEFINITION,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_spec_fingerprint": canonical_fingerprint(spec),
        **spec,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--hypothesis-queue", type=Path, required=True)
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
    parser.add_argument("--default-source-lags", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--default-horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--default-phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--source-window-frames", type=int, default=4)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument(
        "--endpoint-quantiles",
        nargs="*",
        type=float,
        default=[0.1, 0.25, 0.5, 0.75, 0.9],
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base-seed", type=int, default=20260829)
    parser.add_argument("--min-ess", type=float, default=24.0)
    parser.add_argument("--progressive-branch-factor", type=int, default=2)
    parser.add_argument("--progressive-future-branch-factor", type=int, default=2)
    args = parser.parse_args()

    args.folds = tuple(sorted(set(args.folds)))
    args.seeds = tuple(sorted(set(args.seeds)))
    args.default_source_lags = tuple(sorted(set(args.default_source_lags)))
    args.default_horizons = tuple(sorted(set(args.default_horizons)))
    args.default_phases = tuple(
        phase for phase in PHASES if phase in set(args.default_phases)
    )
    args.endpoint_quantiles = tuple(sorted(set(args.endpoint_quantiles)))
    if not set(args.folds).issubset(range(5)):
        parser.error("folds must be selected from 0..4")
    if args.particles < 2 or args.source_window_frames < 1:
        parser.error("particles and source-window frames must be positive")
    if any(not 0.0 < value < 1.0 for value in args.endpoint_quantiles):
        parser.error("endpoint quantiles must lie strictly between zero and one")

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
    if source_manifest_stimulus_fingerprint(source_manifest) != (
        cohort.stimulus_schema_fingerprint
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
    hypothesis_queue = args.hypothesis_queue.resolve()
    groups = load_candidate_groups(
        hypothesis_queue,
        cohort.neurons,
        default_lags=args.default_source_lags,
        default_horizons=args.default_horizons,
        default_phases=args.default_phases,
    )
    manifest = _manifest(
        source_run=source_run,
        fold_file=fold_file,
        hypothesis_queue=hypothesis_queue,
        cohort=cohort,
        groups=groups,
        args=args,
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("run_spec_fingerprint") != manifest["run_spec_fingerprint"]:
            raise RuntimeError("existing confirmation manifest differs from requested run")
    else:
        _atomic_json(manifest_path, manifest)

    records: list[dict[str, Any]] = []
    for group in groups:
        for fold in args.folds:
            for seed in args.seeds:
                try:
                    result = run_one(
                        cohort=cohort,
                        folds=folds,
                        source_run=source_run,
                        checkpoint_phase=args.checkpoint_phase,
                        output=output,
                        model_id=args.model_id,
                        history_lag=args.history_lag,
                        fold=fold,
                        seed=seed,
                        group=group,
                        particles=args.particles,
                        source_window_frames=args.source_window_frames,
                        device=args.device,
                        base_seed=args.base_seed,
                        min_ess=args.min_ess,
                        branch_factor=args.progressive_branch_factor,
                        future_branch_factor=args.progressive_future_branch_factor,
                        endpoint_quantiles=args.endpoint_quantiles,
                        hypothesis_queue=hypothesis_queue,
                    )
                except Exception as error:
                    result = {
                        "status": "failed",
                        "error": repr(error),
                        "wall_seconds": 0.0,
                    }
                records.append(
                    {
                        "source_lag_frames": group.source_lag_frames,
                        "source_count": len(group.source_indices),
                        "selection_fingerprint": group.selection_fingerprint,
                        "fold": fold,
                        "seed": seed,
                        **result,
                    }
                )
                _atomic_csv(output / "run_status.csv", pd.DataFrame(records))
    failures = [row for row in records if row["status"] == "failed"]
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "failed",
        "run_spec_fingerprint": manifest["run_spec_fingerprint"],
        "expected_runs": len(groups) * len(args.folds) * len(args.seeds),
        "completed_or_skipped": sum(
            row["status"] in {"ok", "skipped"} for row in records
        ),
        "failed": len(failures),
        "failures": failures,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _atomic_json(output / "validation.json", validation)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
