"""Canonical worm-level analysis for targeted three-arm flow confirmations.

The immutable sampler archives already use target rows and source columns.  This
module validates that contract, normalizes every event by the magnitude of its
achieved high-minus-low source gap, averages generator seeds *within* worms,
and bootstraps worms as the independent units.  It never uses anatomical or
receptor references and never assigns experimental-confirmation language.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compatibility_neural_benchmark.prediction_atlas_runner import (
    EPISODE_SEED_DEFINITION,
    PHASES,
    canonical_fingerprint,
    sha256,
)
from compatibility_neural_benchmark.targeted_confirmation import (
    ARCHIVE_SCHEMA_VERSION,
    ARM_METRICS,
    ARM_NAMES,
    CLAIM_BOUNDARY,
    COMMON_NOISE_DEFINITION,
    CONTRAST_NAMES,
    DISTANCE_RESPONSE_KEYS,
    FACTUAL_ARM_DEFINITION,
    MANIFEST_SCHEMA_VERSION,
    METHOD,
    SIGNED_METRICS,
    SIGNED_RESPONSE_KEYS,
)


ANALYSIS_SCHEMA_VERSION = "targeted_confirmation_analysis_v1"
ORIENTATION = "target_row_source_column"
CHEMICALS = ("butanone", "pentanedione", "nacl")
CHEMICAL_CODE_TO_NAME = {1: "butanone", 2: "pentanedione", 3: "nacl"}
PAIR_INDICES = {
    "high_low": (1, 0),
    "high_factual": (1, 2),
    "low_factual": (0, 2),
}
ANALYSIS_SIGNED_METRICS = SIGNED_METRICS + ("endpoint_sd",)
SCREEN_TO_TARGETED_METRIC = {
    "endpoint_mean": ("signed", "endpoint_mean"),
    "cumulative_mean": ("signed", "time_average_mean"),
    "peak_mean": ("signed", "pathwise_peak_mean"),
    "event_probability": ("signed", "crossing_probability"),
    "endpoint_sd": ("signed", "endpoint_sd"),
    "endpoint_log_sd": ("signed", "endpoint_log_sd"),
    "endpoint_wasserstein1": ("distance", "endpoint_wasserstein1"),
}
INTERPRETATION_LIMIT = (
    "model-relative repaired-law contrasts; not causal interventions, anatomical "
    "connections, receptor actions, physical delays, or experimental confirmation"
)


@dataclass(frozen=True)
class AnalysisConfig:
    bootstrap_replicates: int = 2_000
    random_seed: int = 20_260_829
    min_gap: float = 0.10
    minimum_valid_fraction: float = 0.50
    strong_valid_fraction: float = 0.80
    strong_sign_consistency: float = 0.80
    genealogy_strong_min_ancestor_fraction: float = 0.10
    genealogy_sensitivity_min_ancestor_fraction: float = 0.20
    expected_particles: int = 128

    def validate(self) -> None:
        if self.bootstrap_replicates < 32:
            raise ValueError("at least 32 worm-bootstrap replicates are required")
        if self.min_gap <= 0:
            raise ValueError("minimum achieved-gap denominator must be positive")
        if not (
            0.0
            <= self.minimum_valid_fraction
            <= self.strong_valid_fraction
            <= 1.0
        ):
            raise ValueError("support fractions are inconsistent")
        if not 0.0 <= self.strong_sign_consistency <= 1.0:
            raise ValueError("sign-consistency threshold must lie in [0,1]")
        if not (
            np.isclose(self.genealogy_strong_min_ancestor_fraction, 0.10)
            and np.isclose(self.genealogy_sensitivity_min_ancestor_fraction, 0.20)
        ):
            raise ValueError("canonical genealogy thresholds are frozen at 0.10 and 0.20")
        if self.expected_particles < 2:
            raise ValueError("expected particle count must be at least two")


@dataclass(frozen=True)
class GroupSpec:
    lag: int
    sources: tuple[int, ...]
    source_names: tuple[str, ...]
    horizons: tuple[int, ...]
    phases: tuple[str, ...]
    selection_fingerprint: str


@dataclass(frozen=True)
class CandidateCell:
    candidate_id: str
    queue_row: int
    queue_rank: int | None
    source: int
    target: int
    source_neuron: str
    target_neuron: str
    lag: int
    horizon: int
    context: str
    screen_channel: str | None


@dataclass(frozen=True)
class ArchiveRecord:
    path: Path
    lag: int
    fold: int
    seed: int
    group: GroupSpec
    worm_ids: tuple[str, ...]
    worm_indices: tuple[int, ...]
    checkpoint: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class TargetedInputs:
    run_dir: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    validation_path: Path
    queue_path: Path
    groups: tuple[GroupSpec, ...]
    records: tuple[ArchiveRecord, ...]
    candidates: tuple[CandidateCell, ...]
    neurons: tuple[str, ...]
    worm_ids: tuple[str, ...]
    worm_indices: tuple[int, ...]
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    fps: float
    endpoint_quantiles: tuple[float, ...]
    resolved_device: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    value = np.asarray(data[key])
    if value.size != 1:
        raise RuntimeError(f"archive field {key!r} must be scalar")
    return value.item()


def _finite(name: str, value: np.ndarray, path: Path) -> None:
    if not np.isfinite(value).all():
        raise RuntimeError(f"{path}: {name} contains nonfinite values")


def _truthy(values: pd.Series) -> np.ndarray:
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin(
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
        )
        .to_numpy()
    )


def _first_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _selected_queue(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError("hypothesis queue is empty")
    selection = _first_column(
        frame, ("run_confirmation", "selected", "queue_status", "status")
    )
    if selection is not None:
        frame = frame.loc[_truthy(frame[selection])].copy()
    if frame.empty:
        raise RuntimeError("hypothesis queue contains no selected rows")
    frame["__queue_row"] = frame.index.astype(int)
    return frame.reset_index(drop=True)


def _group_specs(manifest: Mapping[str, Any]) -> tuple[GroupSpec, ...]:
    raw = manifest.get("resolved_groups")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("targeted manifest has no resolved groups")
    result: list[GroupSpec] = []
    seen_lags: set[int] = set()
    for value in raw:
        if not isinstance(value, Mapping):
            raise RuntimeError("targeted manifest contains a malformed group")
        group = GroupSpec(
            lag=int(value["source_lag_frames"]),
            sources=tuple(int(x) for x in value["source_indices"]),
            source_names=tuple(str(x) for x in value["source_neurons"]),
            horizons=tuple(int(x) for x in value["horizon_frames"]),
            phases=tuple(str(x) for x in value["phases"]),
            selection_fingerprint=str(value["selection_fingerprint"]),
        )
        if group.lag in seen_lags:
            raise RuntimeError("targeted manifest contains duplicate lag groups")
        if group.lag < 0:
            raise RuntimeError("targeted group has a negative source lag")
        if not group.sources or len(set(group.sources)) != len(group.sources):
            raise RuntimeError("targeted group source indices are empty or duplicated")
        if group.sources != tuple(sorted(group.sources)):
            raise RuntimeError("targeted group source indices are not canonical")
        if len(group.sources) != len(group.source_names):
            raise RuntimeError("targeted group source names and indices disagree")
        if (
            not group.horizons
            or min(group.horizons) < 1
            or group.horizons != tuple(sorted(set(group.horizons)))
        ):
            raise RuntimeError("targeted group has an invalid horizon grid")
        if (
            not group.phases
            or not set(group.phases).issubset(PHASES)
            or group.phases
            != tuple(phase for phase in PHASES if phase in set(group.phases))
        ):
            raise RuntimeError("targeted group has unsupported phases")
        if not group.selection_fingerprint:
            raise RuntimeError("targeted group has an empty selection fingerprint")
        seen_lags.add(group.lag)
        result.append(group)
    return tuple(sorted(result, key=lambda group: group.lag))


def _context_required_phases(context: str) -> tuple[str, ...]:
    context = str(context).strip().lower()
    if context in PHASES:
        return (context,)
    if context == "state_average":
        return PHASES
    if context == "onset_minus_baseline" or context.endswith(
        "_onset_minus_baseline"
    ):
        return ("baseline", "onset")
    if context.endswith("_onset"):
        return ("onset",)
    raise RuntimeError(f"unsupported targeted queue context {context!r}")


def _load_candidates(
    queue: Path,
    *,
    groups: Sequence[GroupSpec],
    neurons: tuple[str, ...],
) -> tuple[CandidateCell, ...]:
    frame = _selected_queue(queue)
    source_column = _first_column(
        frame, ("source_neuron", "source", "perturbed_neuron")
    )
    target_column = _first_column(frame, ("target_neuron", "target"))
    lag_column = _first_column(frame, ("source_lag_frames", "lag_frames"))
    horizon_column = _first_column(
        frame, ("horizon_frames", "forecast_horizon_frames")
    )
    context_column = _first_column(
        frame, ("context", "phase", "stimulus_state", "state")
    )
    if None in (
        source_column,
        target_column,
        lag_column,
        horizon_column,
        context_column,
    ):
        raise RuntimeError(
            "canonical targeted analysis requires source, target, lag, horizon, and context"
        )
    neuron_to_index = {name: index for index, name in enumerate(neurons)}
    group_by_lag = {group.lag: group for group in groups}
    cells: list[CandidateCell] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for row_position, row in frame.iterrows():
        source_name = str(row[source_column]).strip()
        target_name = str(row[target_column]).strip()
        if source_name not in neuron_to_index or target_name not in neuron_to_index:
            raise RuntimeError("hypothesis queue contains a neuron outside the archive order")
        source, target = neuron_to_index[source_name], neuron_to_index[target_name]
        lag, horizon = int(row[lag_column]), int(row[horizon_column])
        context = str(row[context_column]).strip().lower()
        required_phases = _context_required_phases(context)
        group = group_by_lag.get(lag)
        if group is None:
            raise RuntimeError(f"queue lag {lag} is absent from targeted groups")
        if source not in group.sources or horizon not in group.horizons:
            raise RuntimeError("queue source/horizon is absent from its targeted group")
        if not set(required_phases).issubset(group.phases):
            raise RuntimeError("targeted group lacks phases required by a queue context")
        key = (source, target, lag, horizon, context)
        if key in seen:
            continue
        seen.add(key)
        queue_rank = None
        if "queue_rank" in frame and pd.notna(row["queue_rank"]):
            queue_rank = int(row["queue_rank"])
        cells.append(
            CandidateCell(
                candidate_id=f"targeted_{len(cells) + 1:04d}",
                queue_row=int(row["__queue_row"]),
                queue_rank=queue_rank,
                source=source,
                target=target,
                source_neuron=source_name,
                target_neuron=target_name,
                lag=lag,
                horizon=horizon,
                context=context,
                screen_channel=(
                    str(row["channel"]).strip()
                    if "channel" in frame and pd.notna(row["channel"])
                    else None
                ),
            )
        )
    if not cells:
        raise RuntimeError("no unique targeted candidate cells were resolved")
    return tuple(cells)


def _manifest_and_static_inputs(
    run_dir: Path, config: AnalysisConfig
) -> tuple[Path, dict[str, Any], Path, tuple[GroupSpec, ...]]:
    manifest_path = run_dir / "manifest.json"
    validation_path = run_dir / "validation.json"
    if not manifest_path.exists() or not validation_path.exists():
        raise RuntimeError("targeted run lacks manifest.json or validation.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("targeted manifest schema version mismatch")
    fingerprint = manifest.get("run_spec_fingerprint")
    spec = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_utc", "run_spec_fingerprint"}
    }
    if fingerprint != canonical_fingerprint(spec):
        raise RuntimeError("targeted manifest fails its own fingerprint")
    required = {
        "method": METHOD,
        "episode_seed_definition": EPISODE_SEED_DEFINITION,
        "common_noise_definition": COMMON_NOISE_DEFINITION,
        "matrix_orientation": ORIENTATION,
        "stimulus_generator_encoding": "binary_any_stimulus",
        "chemical_identity_conditioned": False,
        "factual_arm_definition": FACTUAL_ARM_DEFINITION,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"targeted manifest {key} mismatch")
    manifest_arrays = {
        "phases_available": list(PHASES),
        "arm_names": list(ARM_NAMES),
        "signed_response_keys": list(SIGNED_RESPONSE_KEYS),
        "distance_response_keys": list(DISTANCE_RESPONSE_KEYS),
        "arm_response_keys": [f"arm_{metric}" for metric in ARM_METRICS],
    }
    for key, expected in manifest_arrays.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"targeted manifest {key} mismatch")
    for name in ("folds", "seeds"):
        values = manifest.get(name)
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(int(value) for value in values))
        ):
            raise RuntimeError(f"targeted manifest {name} grid is invalid")
    quantiles = tuple(float(value) for value in manifest.get("endpoint_quantiles", ()))
    if quantiles != tuple(sorted(set(quantiles))) or any(
        value <= 0 or value >= 1 for value in quantiles
    ):
        raise RuntimeError("targeted manifest endpoint quantiles are invalid")
    if int(manifest.get("particles", -1)) != config.expected_particles:
        raise RuntimeError(
            f"canonical targeted analysis requires N={config.expected_particles}"
        )
    for key in (
        "minimum_effective_sample_size",
        "maximum_normalized_weight",
        "minimum_achieved_source_fraction",
    ):
        if key not in manifest or not np.isfinite(float(manifest[key])) or float(
            manifest[key]
        ) <= 0:
            raise RuntimeError(f"targeted manifest {key} is invalid")
    for path_key, hash_key in (
        ("source_run", "source_run_manifest_sha256"),
        ("fold_assignments", "fold_assignments_sha256"),
        ("hypothesis_queue", "hypothesis_queue_sha256"),
    ):
        source = Path(str(manifest[path_key])).resolve()
        if path_key == "source_run":
            source = source / "manifest.json"
        if not source.exists() or sha256(source) != str(manifest[hash_key]):
            raise RuntimeError(f"targeted manifest {path_key} path/hash audit failed")
    validation = json.loads(validation_path.read_text())
    expected_runs = (
        len(manifest["resolved_groups"])
        * len(manifest["folds"])
        * len(manifest["seeds"])
    )
    if (
        validation.get("status") != "pass"
        or validation.get("run_spec_fingerprint") != fingerprint
        or int(validation.get("expected_runs", -1)) != expected_runs
        or int(validation.get("completed_or_skipped", -1)) != expected_runs
        or int(validation.get("failed", -1)) != 0
    ):
        raise RuntimeError("targeted run validation record is incomplete or failed")
    groups = _group_specs(manifest)
    queue_hash = str(manifest["hypothesis_queue_sha256"])
    for group in groups:
        expected_fingerprint = canonical_fingerprint(
            {
                "queue_sha256": queue_hash,
                "source_lag_frames": group.lag,
                "source_indices": list(group.sources),
                "source_neurons": list(group.source_names),
                "horizon_frames": list(group.horizons),
                "phases": list(group.phases),
            }
        )
        if group.selection_fingerprint != expected_fingerprint:
            raise RuntimeError("targeted group selection fingerprint is not reproducible")
    return manifest_path, manifest, validation_path, groups


def _source_cohort_worms(
    manifest: Mapping[str, Any],
) -> tuple[str, ...] | None:
    """Recover the cohort when frozen source-run schedules expose its IDs."""
    source_manifest_path = (
        Path(str(manifest["source_run"])).resolve() / "manifest.json"
    )
    source = json.loads(source_manifest_path.read_text())
    schema = source.get("stimulus_schema")
    schedules = schema.get("schedules") if isinstance(schema, Mapping) else None
    if not isinstance(schedules, list) or not schedules:
        return None
    for key in (
        "n_worms",
        "stimulus_schema_version",
        "stimulus_schema_fingerprint",
    ):
        if key not in source:
            raise RuntimeError(f"source-run manifest lacks {key}")
    if (
        str(source["stimulus_schema_version"])
        != str(manifest["stimulus_schema_version"])
        or str(source["stimulus_schema_fingerprint"])
        != str(manifest["stimulus_schema_fingerprint"])
    ):
        raise RuntimeError("targeted and source-run stimulus schemas differ")
    assert isinstance(schema, Mapping)
    if (
        str(schema.get("version")) != str(source["stimulus_schema_version"])
        or str(schema.get("fingerprint"))
        != str(source["stimulus_schema_fingerprint"])
    ):
        raise RuntimeError("source-run stimulus schema metadata is inconsistent")
    try:
        worms = tuple(str(schedule["worm_id"]) for schedule in schedules)
        n_worms = int(source["n_worms"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("source-run cohort metadata is invalid") from error
    if (
        not worms
        or len(worms) != n_worms
        or len(set(worms)) != len(worms)
        or any(not worm for worm in worms)
    ):
        raise RuntimeError("source-run cohort worm IDs are incomplete or duplicated")
    return worms


def _archive_required_fields(endpoint_quantiles: tuple[float, ...]) -> set[str]:
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
        "requested_device",
        "resolved_device",
        "history_frames",
        "repair_frames",
        "source_lag_frames",
        "source_lag_seconds",
        "lag_definition",
        "source_window_frames",
        "source_window_bounds_semantics",
        "n_particles",
        "progressive_branch_factor",
        "progressive_future_branch_factor",
        "minimum_effective_sample_size",
        "maximum_normalized_weight",
        "minimum_achieved_source_fraction",
        "horizon_frames",
        "horizon_seconds",
        "source_to_readout_seconds",
        "fps",
        "phase_names",
        "arm_names",
        "signed_response_keys",
        "distance_response_keys",
        "arm_response_keys",
        "signed_response_axes",
        "arm_response_axes",
        "matrix_orientation",
        "selected_source_indices",
        "selected_source_neurons",
        "target_neurons",
        "endpoint_quantiles",
        "selection_fingerprint",
        "hypothesis_queue",
        "hypothesis_queue_sha256",
        "worm_indices",
        "worm_ids",
        "cut_times",
        "source_window_bounds",
        "stimulus_schema_version",
        "stimulus_schema_fingerprint",
        "stimulus_generator_encoding",
        "chemical_identity_conditioned",
        "chemical_code_by_worm_event",
        "chemical_name_by_worm_event",
        "event_intervals_seconds_by_worm",
        "schedule_source_recording_by_worm",
        "schedule_native_fps_by_worm",
        "schedule_analysis_fps_by_worm",
        "schedule_resampling_provenance_by_worm",
        "distribution_scale_floor",
        "endpoint_sd_ddof",
        "endpoint_quantile_method",
        "factual_arm_definition",
        "common_noise_definition",
        "claim_boundary",
    }
    required.update(SIGNED_RESPONSE_KEYS)
    required.update(DISTANCE_RESPONSE_KEYS)
    required.update(f"arm_{metric}" for metric in ARM_METRICS)
    required.update(
        f"diagnostic_{name}"
        for name in (
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
            "distinct_ancestors_low",
            "distinct_ancestors_high",
            "step_ess_low",
            "step_ess_high",
            "step_max_weight_low",
            "step_max_weight_high",
            "step_forced_tempering_low",
            "step_forced_tempering_high",
        )
    )
    if endpoint_quantiles:
        required.add("arm_endpoint_quantile")
        required.update(
            f"response_{contrast}_endpoint_quantile"
            for contrast in CONTRAST_NAMES
        )
    return required


def _expected_schedule_cut(
    intervals: np.ndarray,
    *,
    fps: float,
    event_index: int,
    phase: str,
    source_window_frames: int,
) -> int:
    """Reconstruct one episode cut from archived frozen-schedule provenance."""
    start, stop = np.asarray(intervals, dtype=np.float64)[event_index]
    onset = int(round(float(start) * float(fps)))
    offset = int(round(float(stop) * float(fps)))
    candidates = {
        "baseline": onset - int(round(15.0 * float(fps))),
        "onset": onset + int(source_window_frames) - 1,
        "active": onset + int(round(5.0 * float(fps))),
        "offset": offset + int(source_window_frames) - 1,
        "recovery": offset + int(round(5.0 * float(fps))),
    }
    try:
        return candidates[phase]
    except KeyError as error:
        raise RuntimeError(f"unsupported episode phase {phase!r}") from error


def _validate_archive(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    group_by_lag: Mapping[int, GroupSpec],
    endpoint_quantiles: tuple[float, ...],
) -> tuple[ArchiveRecord, tuple[str, ...], float, str]:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(_archive_required_fields(endpoint_quantiles).difference(data.files))
        if missing:
            raise RuntimeError(f"{path}: archive lacks required fields {missing}")
        scalar_expected = {
            "status": "complete",
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
            "method": METHOD,
            "model_id": str(manifest["model_id"]),
            "base_seed": int(manifest["base_seed"]),
            "episode_seed_definition": EPISODE_SEED_DEFINITION,
            "requested_device": str(manifest["requested_device"]),
            "history_frames": int(manifest["history_frames"]),
            "source_window_frames": int(manifest["source_window_frames"]),
            "n_particles": int(manifest["particles"]),
            "progressive_branch_factor": int(manifest["progressive_branch_factor"]),
            "progressive_future_branch_factor": int(
                manifest["progressive_future_branch_factor"]
            ),
            "minimum_effective_sample_size": float(
                manifest["minimum_effective_sample_size"]
            ),
            "maximum_normalized_weight": float(
                manifest["maximum_normalized_weight"]
            ),
            "minimum_achieved_source_fraction": float(
                manifest["minimum_achieved_source_fraction"]
            ),
            "source_window_bounds_semantics": "[inclusive_start,exclusive_stop)",
            "lag_definition": "source-window end to prediction cut",
            "matrix_orientation": ORIENTATION,
            "stimulus_schema_version": str(manifest["stimulus_schema_version"]),
            "stimulus_schema_fingerprint": str(
                manifest["stimulus_schema_fingerprint"]
            ),
            "stimulus_generator_encoding": "binary_any_stimulus",
            "chemical_identity_conditioned": False,
            "distribution_scale_floor": float(manifest["distribution_scale_floor"]),
            "endpoint_sd_ddof": 0,
            "endpoint_quantile_method": "numpy_linear_empirical",
            "factual_arm_definition": FACTUAL_ARM_DEFINITION,
            "common_noise_definition": COMMON_NOISE_DEFINITION,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for key, expected in scalar_expected.items():
            actual = _scalar(data, key)
            if isinstance(expected, float):
                equal = np.isclose(float(actual), expected, rtol=0, atol=1e-8)
            else:
                equal = actual == expected
            if not equal:
                raise RuntimeError(f"{path}: archive {key} mismatch")
        lag, fold, seed = (
            int(_scalar(data, "source_lag_frames")),
            int(_scalar(data, "fold")),
            int(_scalar(data, "seed")),
        )
        if fold not in manifest["folds"] or seed not in manifest["seeds"]:
            raise RuntimeError(f"{path}: archive fold/seed is outside the requested grid")
        group = group_by_lag.get(lag)
        if group is None:
            raise RuntimeError(f"{path}: archive lag is outside the requested grid")
        if int(_scalar(data, "repair_frames")) != lag + int(
            manifest["source_window_frames"]
        ):
            raise RuntimeError(f"{path}: repair length does not equal lag plus window")
        arrays_expected = {
            "horizon_frames": np.asarray(group.horizons),
            "phase_names": np.asarray(group.phases),
            "arm_names": np.asarray(ARM_NAMES),
            "signed_response_keys": np.asarray(SIGNED_RESPONSE_KEYS),
            "distance_response_keys": np.asarray(DISTANCE_RESPONSE_KEYS),
            "arm_response_keys": np.asarray(
                tuple(f"arm_{metric}" for metric in ARM_METRICS)
            ),
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
            "selected_source_indices": np.asarray(group.sources),
            "selected_source_neurons": np.asarray(group.source_names),
            "endpoint_quantiles": np.asarray(endpoint_quantiles),
        }
        for key, expected in arrays_expected.items():
            actual = np.asarray(data[key])
            if np.issubdtype(expected.dtype, np.floating):
                equal = np.allclose(actual, expected, rtol=0, atol=1e-7)
            else:
                equal = np.array_equal(actual, expected)
            if not equal:
                raise RuntimeError(f"{path}: archive {key} mismatch")
        if str(_scalar(data, "selection_fingerprint")) != group.selection_fingerprint:
            raise RuntimeError(f"{path}: archive selection fingerprint mismatch")
        targets = tuple(data["target_neurons"].astype(str))
        if not targets or len(set(targets)) != len(targets):
            raise RuntimeError(f"{path}: target-neuron order is empty or duplicated")
        d, m, h, p = (
            len(targets),
            len(group.sources),
            len(group.horizons),
            len(group.phases),
        )
        worm_ids = tuple(data["worm_ids"].astype(str))
        worm_indices = tuple(int(x) for x in data["worm_indices"])
        w = len(worm_ids)
        if (
            w < 1
            or len(worm_indices) != w
            or len(set(worm_ids)) != w
            or len(set(worm_indices)) != w
            or min(worm_indices) < 0
        ):
            raise RuntimeError(f"{path}: held-out worm metadata is invalid")
        signed_shape = (w, p, 3, h, d, m)
        arm_shape = (w, p, 3, len(ARM_NAMES), h, d, m)
        for key in SIGNED_RESPONSE_KEYS + DISTANCE_RESPONSE_KEYS:
            value = np.asarray(data[key])
            if value.shape != signed_shape:
                raise RuntimeError(f"{path}: {key} shape is not target-row/source-column")
            _finite(key, value, path)
        for key in tuple(f"arm_{metric}" for metric in ARM_METRICS):
            value = np.asarray(data[key])
            if value.shape != arm_shape:
                raise RuntimeError(f"{path}: {key} shape is invalid")
            _finite(key, value, path)
        for key in DISTANCE_RESPONSE_KEYS:
            if np.any(np.asarray(data[key]) < -1e-7):
                raise RuntimeError(f"{path}: {key} is negative")
        arm_crossing = np.asarray(data["arm_crossing_probability"])
        if np.any(arm_crossing < -1e-7) or np.any(arm_crossing > 1 + 1e-7):
            raise RuntimeError(f"{path}: arm crossing probability is out of range")
        if np.any(np.asarray(data["arm_endpoint_sd"]) < -1e-7):
            raise RuntimeError(f"{path}: arm endpoint SD is negative")
        floor = float(manifest["distribution_scale_floor"])
        arm_key_by_metric = {
            "endpoint_mean": "arm_endpoint_mean",
            "time_average_mean": "arm_time_average_mean",
            "pathwise_peak_mean": "arm_pathwise_peak_mean",
            "crossing_probability": "arm_crossing_probability",
        }
        for contrast, (left, right) in PAIR_INDICES.items():
            for metric, arm_key in arm_key_by_metric.items():
                expected = np.asarray(data[arm_key])[:, :, :, left] - np.asarray(
                    data[arm_key]
                )[:, :, :, right]
                actual = np.asarray(data[f"response_{contrast}_{metric}"])
                if not np.allclose(actual, expected, rtol=2e-5, atol=2e-6):
                    raise RuntimeError(f"{path}: {contrast}/{metric} arm identity failed")
            sd = np.asarray(data["arm_endpoint_sd"])
            expected_log_sd = np.log(sd[:, :, :, left] + floor) - np.log(
                sd[:, :, :, right] + floor
            )
            if not np.allclose(
                data[f"response_{contrast}_endpoint_log_sd"],
                expected_log_sd,
                rtol=2e-5,
                atol=2e-6,
            ):
                raise RuntimeError(f"{path}: {contrast}/endpoint_log_sd identity failed")
        if endpoint_quantiles:
            q = len(endpoint_quantiles)
            arm_quantile = np.asarray(data["arm_endpoint_quantile"])
            if arm_quantile.shape != (w, p, 3, len(ARM_NAMES), h, q, d, m):
                raise RuntimeError(f"{path}: arm endpoint quantile shape is invalid")
            _finite("arm_endpoint_quantile", arm_quantile, path)
            if np.any(np.diff(arm_quantile, axis=5) < -1e-6):
                raise RuntimeError(f"{path}: arm endpoint quantiles are not monotone")
            for contrast, (left, right) in PAIR_INDICES.items():
                actual = np.asarray(data[f"response_{contrast}_endpoint_quantile"])
                expected = arm_quantile[:, :, :, left] - arm_quantile[:, :, :, right]
                if actual.shape != (w, p, 3, h, q, d, m) or not np.allclose(
                    actual, expected, rtol=2e-5, atol=2e-6
                ):
                    raise RuntimeError(f"{path}: {contrast} quantile identity failed")
        diagnostic_shape = (w, p, 3, m)
        target_low = np.asarray(data["diagnostic_target_low"])
        target_high = np.asarray(data["diagnostic_target_high"])
        target_gap = np.asarray(data["diagnostic_target_gap"])
        achieved_low = np.asarray(data["diagnostic_achieved_low"])
        achieved_high = np.asarray(data["diagnostic_achieved_high"])
        achieved_gap = np.asarray(data["diagnostic_achieved_gap"])
        for name, value in (
            ("target_low", target_low),
            ("target_high", target_high),
            ("target_gap", target_gap),
            ("achieved_low", achieved_low),
            ("achieved_high", achieved_high),
            ("achieved_gap", achieved_gap),
        ):
            if value.shape != diagnostic_shape:
                raise RuntimeError(f"{path}: diagnostic_{name} shape is invalid")
            _finite(f"diagnostic_{name}", value, path)
        if not np.allclose(target_gap, target_high - target_low, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"{path}: target-gap identity failed")
        if np.any(target_gap <= 0):
            raise RuntimeError(f"{path}: target source gap is not positive")
        if not np.allclose(
            achieved_gap, achieved_high - achieved_low, rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError(f"{path}: achieved-gap identity failed")
        valid = np.asarray(data["diagnostic_valid"])
        if valid.shape != diagnostic_shape or not np.all(np.isin(valid, (0, 1))):
            raise RuntimeError(f"{path}: support-valid diagnostic is invalid")
        ess: dict[str, np.ndarray] = {}
        max_weight: dict[str, np.ndarray] = {}
        branch = int(manifest["progressive_branch_factor"])
        particles = int(manifest["particles"])
        for side in ("low", "high"):
            ess[side] = np.asarray(data[f"diagnostic_ess_{side}"])
            max_weight[side] = np.asarray(data[f"diagnostic_max_weight_{side}"])
            if (
                ess[side].shape != diagnostic_shape
                or max_weight[side].shape != diagnostic_shape
                or np.any(ess[side] < 1.0 / branch - 1e-5)
                or np.any(ess[side] > particles + 1e-4)
                or np.any(max_weight[side] < 1.0 / particles - 1e-6)
                or np.any(max_weight[side] > branch + 1e-6)
            ):
                raise RuntimeError(f"{path}: {side} support diagnostics are invalid")
        expected_valid = (
            (ess["low"] >= float(manifest["minimum_effective_sample_size"]))
            & (ess["high"] >= float(manifest["minimum_effective_sample_size"]))
            & (
                max_weight["low"]
                <= float(manifest["maximum_normalized_weight"])
            )
            & (
                max_weight["high"]
                <= float(manifest["maximum_normalized_weight"])
            )
            & (
                achieved_gap
                >= float(manifest["minimum_achieved_source_fraction"])
                * target_gap
            )
        )
        if not np.array_equal(valid.astype(bool), expected_valid):
            raise RuntimeError(
                f"{path}: diagnostic_valid does not match declared gates"
            )
        selected_index = np.asarray(data["diagnostic_selected_source_index"])
        if not np.array_equal(
            selected_index,
            np.broadcast_to(np.asarray(group.sources, dtype=np.float32), diagnostic_shape),
        ):
            raise RuntimeError(f"{path}: selected-source orientation diagnostic failed")
        endpoint_sd_floor = np.asarray(data["diagnostic_endpoint_sd_floor"])
        if endpoint_sd_floor.shape != diagnostic_shape or not np.allclose(
            endpoint_sd_floor, floor, rtol=0, atol=0
        ):
            raise RuntimeError(f"{path}: endpoint-SD floor diagnostic failed")
        for side in ("low", "high"):
            ancestors = np.asarray(data[f"diagnostic_distinct_ancestors_{side}"])
            if (
                ancestors.shape != diagnostic_shape
                or np.any(ancestors < 1)
                or np.any(ancestors > int(manifest["particles"]))
            ):
                raise RuntimeError(f"{path}: {side} ancestor diagnostic is invalid")
        step_shape = (
            *diagnostic_shape,
            lag + int(manifest["source_window_frames"]),
        )
        for key in data.files:
            if not key.startswith("diagnostic_"):
                continue
            value = np.asarray(data[key])
            if value.shape not in {diagnostic_shape, step_shape}:
                raise RuntimeError(f"{path}: {key} shape is invalid")
            _finite(key, value, path)
        for side in ("low", "high"):
            step_ess = np.asarray(data[f"diagnostic_step_ess_{side}"])
            step_max_weight = np.asarray(
                data[f"diagnostic_step_max_weight_{side}"]
            )
            forced = np.asarray(data[f"diagnostic_step_forced_tempering_{side}"])
            if (
                np.any(step_ess < 1.0 / branch - 1e-5)
                or np.any(step_ess > particles + 1e-4)
                or np.any(step_max_weight < 1.0 / particles - 1e-6)
                or np.any(step_max_weight > branch + 1e-6)
                or not np.all(np.isin(forced, (0, 1)))
            ):
                raise RuntimeError(f"{path}: {side} step diagnostics are invalid")
        cut_times = np.asarray(data["cut_times"])
        bounds = np.asarray(data["source_window_bounds"])
        if cut_times.shape != (w, p, 3) or bounds.shape != (w, p, 3, 2):
            raise RuntimeError(f"{path}: cut/window geometry shape is invalid")
        if not np.all(
            bounds[..., 1] - bounds[..., 0]
            == int(manifest["source_window_frames"])
        ):
            raise RuntimeError(f"{path}: source-window width geometry failed")
        if not np.array_equal(bounds[..., 1] - 1, cut_times - lag):
            raise RuntimeError(f"{path}: source-window/cut lag geometry failed")
        fps = float(_scalar(data, "fps"))
        if not np.isclose(float(_scalar(data, "source_lag_seconds")), lag / fps):
            raise RuntimeError(f"{path}: source lag seconds mismatch")
        if not np.allclose(data["horizon_seconds"], np.asarray(group.horizons) / fps):
            raise RuntimeError(f"{path}: horizon seconds mismatch")
        if not np.allclose(
            data["source_to_readout_seconds"],
            (lag + np.asarray(group.horizons)) / fps,
        ):
            raise RuntimeError(f"{path}: source-to-readout seconds mismatch")
        codes = np.asarray(data["chemical_code_by_worm_event"], dtype=np.int8)
        names = np.asarray(data["chemical_name_by_worm_event"]).astype(str)
        if codes.shape != (w, 3) or names.shape != (w, 3):
            raise RuntimeError(f"{path}: chemical schedule shape is invalid")
        for row_codes, row_names in zip(codes, names):
            if tuple(sorted(int(x) for x in row_codes)) != (1, 2, 3):
                raise RuntimeError(f"{path}: chemical codes are not a permutation")
            expected_names = tuple(CHEMICAL_CODE_TO_NAME[int(x)] for x in row_codes)
            if tuple(row_names) != expected_names:
                raise RuntimeError(f"{path}: chemical code/name mapping failed")
        intervals = np.asarray(data["event_intervals_seconds_by_worm"], dtype=float)
        source_recordings = np.asarray(
            data["schedule_source_recording_by_worm"]
        ).astype(str)
        native_fps = np.asarray(data["schedule_native_fps_by_worm"], dtype=float)
        analysis_fps = np.asarray(
            data["schedule_analysis_fps_by_worm"], dtype=float
        )
        resampling = np.asarray(
            data["schedule_resampling_provenance_by_worm"]
        ).astype(str)
        if (
            intervals.shape != (w, 3, 2)
            or source_recordings.shape != (w,)
            or native_fps.shape != (w,)
            or analysis_fps.shape != (w,)
            or resampling.shape != (w,)
            or not np.isfinite(intervals).all()
            or np.any(intervals[..., 1] <= intervals[..., 0])
            or not np.isfinite(native_fps).all()
            or np.any(native_fps <= 0)
            or not np.allclose(analysis_fps, fps, rtol=0, atol=1e-8)
            or np.any(np.char.str_len(source_recordings) == 0)
            or np.any(np.char.str_len(resampling) == 0)
        ):
            raise RuntimeError(f"{path}: schedule provenance is invalid")
        for local in range(w):
            for phase_index, phase in enumerate(group.phases):
                for event_index in range(3):
                    expected_cut = _expected_schedule_cut(
                        intervals[local],
                        fps=float(analysis_fps[local]),
                        event_index=event_index,
                        phase=phase,
                        source_window_frames=int(manifest["source_window_frames"]),
                    )
                    observed_cut = int(cut_times[local, phase_index, event_index])
                    if observed_cut != expected_cut:
                        raise RuntimeError(
                            f"{path}: cut time disagrees with the frozen event schedule "
                            f"for worm={local}, phase={phase}, event={event_index}: "
                            f"observed={observed_cut}, expected={expected_cut}"
                        )
        queue_path = Path(str(_scalar(data, "hypothesis_queue"))).resolve()
        if (
            queue_path != Path(str(manifest["hypothesis_queue"])).resolve()
            or str(_scalar(data, "hypothesis_queue_sha256"))
            != str(manifest["hypothesis_queue_sha256"])
        ):
            raise RuntimeError(f"{path}: archive queue provenance mismatch")
        checkpoint = Path(str(_scalar(data, "checkpoint"))).resolve()
        checkpoint_hash = str(_scalar(data, "checkpoint_sha256"))
        if not checkpoint.exists() or sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError(f"{path}: checkpoint path/hash audit failed")
        resolved_device = str(_scalar(data, "resolved_device"))
        if not resolved_device:
            raise RuntimeError(f"{path}: resolved device is empty")
        return (
            ArchiveRecord(
                path=path,
                lag=lag,
                fold=fold,
                seed=seed,
                group=group,
                worm_ids=worm_ids,
                worm_indices=worm_indices,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_hash,
            ),
            targets,
            fps,
            resolved_device,
        )


def discover_and_validate_targeted_inputs(
    run_dir: Path,
    *,
    config: AnalysisConfig = AnalysisConfig(),
) -> TargetedInputs:
    """Validate the complete targeted N128 grid before deriving any result."""
    config.validate()
    run_dir = Path(run_dir).resolve()
    manifest_path, manifest, validation_path, groups = _manifest_and_static_inputs(
        run_dir, config
    )
    queue_path = Path(str(manifest["hypothesis_queue"])).resolve()
    endpoint_quantiles = tuple(float(x) for x in manifest["endpoint_quantiles"])
    paths = tuple(sorted((run_dir / "responses").glob("*.npz")))
    if not paths:
        raise RuntimeError("targeted run has no response archives")
    group_by_lag = {group.lag: group for group in groups}
    records: list[ArchiveRecord] = []
    neurons: tuple[str, ...] | None = None
    fps: float | None = None
    resolved_devices: set[str] = set()
    seen_cells: set[tuple[int, int, int]] = set()
    worm_map: dict[int, str] = {}
    schedule_by_worm: dict[str, tuple[Any, ...]] = {}
    checkpoint_by_fold_seed: dict[tuple[int, int], tuple[Path, str]] = {}
    for path in paths:
        record, current_neurons, current_fps, device = _validate_archive(
            path,
            manifest=manifest,
            group_by_lag=group_by_lag,
            endpoint_quantiles=endpoint_quantiles,
        )
        cell = (record.lag, record.fold, record.seed)
        if cell in seen_cells:
            raise RuntimeError(f"duplicate targeted archive cell {cell}")
        seen_cells.add(cell)
        if neurons is None:
            neurons = current_neurons
            fps = current_fps
        elif current_neurons != neurons or not np.isclose(current_fps, fps):
            raise RuntimeError("target/neuron order or FPS changed across archives")
        for index, worm_id in zip(record.worm_indices, record.worm_ids):
            previous = worm_map.setdefault(index, worm_id)
            if previous != worm_id:
                raise RuntimeError("worm index/name mapping changed across archives")
        with np.load(path, allow_pickle=False) as data:
            for local, worm_id in enumerate(record.worm_ids):
                signature = (
                    tuple(int(x) for x in data["chemical_code_by_worm_event"][local]),
                    tuple(data["chemical_name_by_worm_event"][local].astype(str)),
                    tuple(map(tuple, data["event_intervals_seconds_by_worm"][local])),
                    str(data["schedule_source_recording_by_worm"][local]),
                    float(data["schedule_native_fps_by_worm"][local]),
                    float(data["schedule_analysis_fps_by_worm"][local]),
                    str(data["schedule_resampling_provenance_by_worm"][local]),
                )
                previous = schedule_by_worm.setdefault(worm_id, signature)
                if previous != signature:
                    raise RuntimeError("worm stimulus schedule changed across archives")
        checkpoint_cell = (record.fold, record.seed)
        checkpoint = (record.checkpoint, record.checkpoint_sha256)
        previous_checkpoint = checkpoint_by_fold_seed.setdefault(
            checkpoint_cell, checkpoint
        )
        if previous_checkpoint != checkpoint:
            raise RuntimeError("checkpoint changed across targeted lag groups")
        resolved_devices.add(device)
        records.append(record)
    expected = {
        (group.lag, int(fold), int(seed))
        for group in groups
        for fold in manifest["folds"]
        for seed in manifest["seeds"]
    }
    if seen_cells != expected:
        missing = sorted(expected - seen_cells)
        extra = sorted(seen_cells - expected)
        raise RuntimeError(f"targeted archive grid mismatch; missing={missing}, extra={extra}")
    seeds = tuple(int(x) for x in manifest["seeds"])
    folds = tuple(int(x) for x in manifest["folds"])
    source_cohort_worms = _source_cohort_worms(manifest)
    fold_frame = pd.read_csv(Path(str(manifest["fold_assignments"])).resolve())
    required_fold_columns = {"worm_index", "worm_id", "outer_fold"}
    if not required_fold_columns.issubset(fold_frame.columns) or fold_frame.empty:
        raise RuntimeError("fold assignments lack the canonical worm/fold columns")
    fold_frame = fold_frame.loc[:, ["worm_index", "worm_id", "outer_fold"]].copy()
    fold_frame["worm_index"] = pd.to_numeric(
        fold_frame["worm_index"], errors="raise"
    ).astype(int)
    fold_frame["outer_fold"] = pd.to_numeric(
        fold_frame["outer_fold"], errors="raise"
    ).astype(int)
    fold_frame["worm_id"] = fold_frame["worm_id"].astype(str)
    if (
        fold_frame.worm_index.duplicated().any()
        or fold_frame.worm_id.duplicated().any()
        or (fold_frame.worm_index < 0).any()
    ):
        raise RuntimeError("fold assignments have duplicate or invalid worm identifiers")
    fold_by_index = {
        int(row.worm_index): (str(row.worm_id), int(row.outer_fold))
        for row in fold_frame.itertuples(index=False)
    }
    fold_by_worm = {
        str(row.worm_id): int(row.outer_fold)
        for row in fold_frame.itertuples(index=False)
    }
    if source_cohort_worms is not None:
        missing_source_worms = sorted(
            set(source_cohort_worms) - set(fold_by_worm)
        )
        if missing_source_worms:
            raise RuntimeError(
                "fold assignments omit source-run cohort worms: "
                f"{missing_source_worms}"
            )
    for record in records:
        for worm_index, worm_id in zip(record.worm_indices, record.worm_ids):
            expected_assignment = fold_by_index.get(worm_index)
            if expected_assignment != (worm_id, record.fold):
                raise RuntimeError(
                    "archive held-out worms disagree with frozen fold assignments"
                )
    reference_worms: set[str] | None = None
    for group in groups:
        for seed in seeds:
            group_records = [
                record
                for record in records
                if record.lag == group.lag and record.seed == seed
            ]
            flattened = [worm for record in group_records for worm in record.worm_ids]
            if len(flattened) != len(set(flattened)):
                raise RuntimeError("a worm appears in multiple folds for one targeted grid")
            current = set(flattened)
            if reference_worms is None:
                reference_worms = current
            elif current != reference_worms:
                raise RuntimeError("held-out worm coverage changes across lags or seeds")
    assert neurons is not None and fps is not None and reference_worms is not None
    requested_fold_worms = (
        {
            worm
            for worm in source_cohort_worms
            if fold_by_worm[worm] in folds
        }
        if source_cohort_worms is not None
        else set(
            fold_frame.loc[
                fold_frame.outer_fold.isin(folds), "worm_id"
            ].astype(str)
        )
    )
    if reference_worms != requested_fold_worms:
        raise RuntimeError("targeted held-out coverage differs from requested folds")
    ordered_indices = tuple(sorted(worm_map))
    ordered_worms = tuple(worm_map[index] for index in ordered_indices)
    if set(ordered_worms) != reference_worms:
        raise RuntimeError("global worm index does not match fold coverage")
    for group in groups:
        if any(index >= len(neurons) for index in group.sources):
            raise RuntimeError("targeted source index exceeds target-neuron order")
        if tuple(neurons[index] for index in group.sources) != group.source_names:
            raise RuntimeError("targeted source names do not match neuron order")
    candidates = _load_candidates(queue_path, groups=groups, neurons=neurons)
    if len(resolved_devices) != 1:
        raise RuntimeError("resolved device changed across targeted archives")
    return TargetedInputs(
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        validation_path=validation_path,
        queue_path=queue_path,
        groups=groups,
        records=tuple(records),
        candidates=candidates,
        neurons=neurons,
        worm_ids=ordered_worms,
        worm_indices=ordered_indices,
        folds=folds,
        seeds=seeds,
        fps=fps,
        endpoint_quantiles=endpoint_quantiles,
        resolved_device=next(iter(resolved_devices)),
    )


def _contextualize(
    values: np.ndarray,
    context: str,
    codes: np.ndarray,
    *,
    phase_names: Sequence[str] = PHASES,
    contrast_reducer: str = "difference",
) -> np.ndarray:
    """Reduce [seed,worm,phase,event,...] without treating events as animals."""
    x = np.asarray(values)
    if x.ndim < 4 or codes.shape != (x.shape[1], x.shape[3]):
        raise ValueError("context arrays or chemical codes have incompatible axes")
    positions = {phase: index for index, phase in enumerate(phase_names)}

    def phase(name: str) -> np.ndarray:
        if name not in positions:
            raise RuntimeError(f"targeted archive lacks phase {name!r}")
        return x[:, :, positions[name]]

    def combine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if contrast_reducer == "difference":
            return left - right
        if contrast_reducer == "min":
            return np.minimum(left, right)
        if contrast_reducer == "max":
            return np.maximum(left, right)
        raise ValueError("unknown context contrast reducer")

    context = str(context)
    if context in positions:
        result = phase(context).mean(axis=2)
    elif context == "state_average":
        result = x.mean(axis=(2, 3))
    elif context == "onset_minus_baseline":
        result = combine(phase("onset"), phase("baseline")).mean(axis=2)
    else:
        chemical = next(
            (name for name in CHEMICALS if context.startswith(name + "_")), None
        )
        if chemical is None:
            raise RuntimeError(f"unsupported context {context!r}")
        code = {name: value for value, name in CHEMICAL_CODE_TO_NAME.items()}[chemical]
        event_position = np.argmax(codes == code, axis=1)
        if not np.all(codes[np.arange(len(codes)), event_position] == code):
            raise RuntimeError(f"one or more worms lack chemical {chemical}")
        worm = np.arange(len(codes))
        onset = phase("onset")[:, worm, event_position]
        if context == f"{chemical}_onset":
            result = onset
        elif context == f"{chemical}_onset_minus_baseline":
            baseline = phase("baseline")[:, worm, event_position]
            result = combine(onset, baseline)
        else:
            raise RuntimeError(f"unsupported chemical context {context!r}")
    if not np.isfinite(result).all():
        raise RuntimeError(f"context {context!r} contains incomplete targeted values")
    return result


def _bootstrap_weights(n_worms: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_worms, size=(replicates, n_worms))
    result = np.zeros((replicates, n_worms), dtype=np.float32)
    for index, row in enumerate(draws):
        result[index] = np.bincount(row, minlength=n_worms)
    return result / n_worms


def _pairwise_seed_spearman(seed_values: np.ndarray) -> float:
    values = np.asarray(seed_values, dtype=np.float64)
    correlations: list[float] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if np.std(values[left]) > 0 and np.std(values[right]) > 0:
                correlation = float(spearmanr(values[left], values[right]).statistic)
                if np.isfinite(correlation):
                    correlations.append(correlation)
    return float(np.mean(correlations)) if correlations else float("nan")


def _summary(
    normalized: np.ndarray,
    raw: np.ndarray,
    *,
    signed: bool,
    bootstrap_weights: np.ndarray,
) -> dict[str, float]:
    """Summarize [seed,worm] values after averaging seeds within each worm."""
    normalized = np.asarray(normalized, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    if normalized.ndim != 2 or raw.shape != normalized.shape:
        raise ValueError("summary inputs must be matching [seed,worm] arrays")
    worm = normalized.mean(axis=0)
    worm_raw = raw.mean(axis=0)
    bootstrap = bootstrap_weights @ worm
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    seed_means = normalized.mean(axis=1)
    if signed:
        sign_consistency = float(
            max(np.mean(worm > 0), np.mean(worm < 0))
        )
        seed_sign_agreement = float(
            max(np.mean(seed_means > 0), np.mean(seed_means < 0))
        )
    else:
        sign_consistency = float("nan")
        seed_sign_agreement = float("nan")
    return {
        "mean_normalized": float(worm.mean()),
        "median_normalized": float(np.median(worm)),
        "mean_raw": float(worm_raw.mean()),
        "ci_2_5": float(low),
        "ci_97_5": float(high),
        "worm_sd": float(worm.std(ddof=1)) if len(worm) > 1 else 0.0,
        "sign_consistency": sign_consistency,
        "seed_sign_agreement": seed_sign_agreement,
        "seed_worm_spearman": _pairwise_seed_spearman(normalized),
        "maximum_seed_mean_difference": float(
            seed_means.max() - seed_means.min()
        ),
    }


def _evidence_label(
    summary: Mapping[str, float],
    *,
    signed: bool,
    valid_fraction: float,
    genealogy_strong_gate_pass: bool,
    config: AnalysisConfig,
) -> str:
    if valid_fraction < config.minimum_valid_fraction:
        return "unsupported_model_output"
    if not signed:
        return "model_relative_descriptive"
    excludes_zero = summary["ci_2_5"] > 0 or summary["ci_97_5"] < 0
    if (
        excludes_zero
        and valid_fraction >= config.strong_valid_fraction
        and genealogy_strong_gate_pass
        and summary["sign_consistency"] >= config.strong_sign_consistency
        and summary["seed_sign_agreement"] >= 1.0
    ):
        return "model_relative_consistent"
    return "model_relative_uncertain"


def _empty_cell_array(inputs: TargetedInputs, trailing: tuple[int, ...] = ()) -> np.ndarray:
    return np.full(
        (len(inputs.seeds), len(inputs.worm_ids), len(PHASES), 3, *trailing),
        np.nan,
        dtype=np.float32,
    )


def _extract_candidate_arrays(
    inputs: TargetedInputs,
    *,
    min_gap: float,
) -> tuple[
    dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    np.ndarray,
]:
    """Load only selected cells while retaining seed, worm, phase, and event."""
    seed_pos = {seed: index for index, seed in enumerate(inputs.seeds)}
    worm_pos = {worm: index for index, worm in enumerate(inputs.worm_ids)}
    candidate_by_lag: dict[int, list[CandidateCell]] = {}
    for candidate in inputs.candidates:
        candidate_by_lag.setdefault(candidate.lag, []).append(candidate)
    signed: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    distances: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    quantiles: dict[str, dict[str, np.ndarray]] = {}
    support_keys = (
        "valid",
        "achieved_gap_abs",
        "ess_low",
        "ess_high",
        "max_weight_low",
        "max_weight_high",
        "distinct_ancestors_low",
        "distinct_ancestors_high",
        "forced_tempering_rate_low",
        "forced_tempering_rate_high",
    )
    unique_source = {
        (candidate.lag, candidate.source) for candidate in inputs.candidates
    }
    support = {
        f"{lag}:{source}": {
            key: _empty_cell_array(inputs) for key in support_keys
        }
        for lag, source in unique_source
    }
    q_count = len(inputs.endpoint_quantiles)
    for candidate in inputs.candidates:
        for contrast in CONTRAST_NAMES:
            for metric in ANALYSIS_SIGNED_METRICS:
                signed[(candidate.candidate_id, contrast, metric)] = (
                    _empty_cell_array(inputs),
                    _empty_cell_array(inputs),
                )
            distances[(candidate.candidate_id, contrast)] = (
                _empty_cell_array(inputs),
                _empty_cell_array(inputs),
            )
            if q_count:
                quantiles[f"{candidate.candidate_id}:{contrast}"] = {
                    "raw": _empty_cell_array(inputs, (q_count,)),
                    "normalized": _empty_cell_array(inputs, (q_count,)),
                }
    codes = np.full((len(inputs.worm_ids), 3), -1, dtype=np.int8)
    for record in inputs.records:
        si = seed_pos[record.seed]
        group = record.group
        phase_global = [PHASES.index(phase) for phase in group.phases]
        source_local = {source: index for index, source in enumerate(group.sources)}
        horizon_local = {
            horizon: index for index, horizon in enumerate(group.horizons)
        }
        with np.load(record.path, allow_pickle=False) as data:
            local_codes = np.asarray(data["chemical_code_by_worm_event"], dtype=np.int8)
            local_worm_positions = [worm_pos[worm] for worm in record.worm_ids]
            for local, wi in enumerate(local_worm_positions):
                if np.any(codes[wi] >= 0) and not np.array_equal(
                    codes[wi], local_codes[local]
                ):
                    raise RuntimeError("chemical order changes while extracting cells")
                codes[wi] = local_codes[local]

            # Support is source-specific and is extracted once per lag/source.
            for source in {candidate.source for candidate in candidate_by_lag[record.lag]}:
                sj = source_local[source]
                key = f"{record.lag}:{source}"
                source_values = {
                    "valid": np.asarray(data["diagnostic_valid"])[..., sj],
                    "achieved_gap_abs": np.abs(
                        np.asarray(data["diagnostic_achieved_gap"])[..., sj]
                    ),
                    "ess_low": np.asarray(data["diagnostic_ess_low"])[..., sj],
                    "ess_high": np.asarray(data["diagnostic_ess_high"])[..., sj],
                    "max_weight_low": np.asarray(
                        data["diagnostic_max_weight_low"]
                    )[..., sj],
                    "max_weight_high": np.asarray(
                        data["diagnostic_max_weight_high"]
                    )[..., sj],
                    "distinct_ancestors_low": np.asarray(
                        data["diagnostic_distinct_ancestors_low"]
                    )[..., sj],
                    "distinct_ancestors_high": np.asarray(
                        data["diagnostic_distinct_ancestors_high"]
                    )[..., sj],
                    "forced_tempering_rate_low": np.asarray(
                        data["diagnostic_step_forced_tempering_low"]
                    )[..., sj, :].mean(axis=-1),
                    "forced_tempering_rate_high": np.asarray(
                        data["diagnostic_step_forced_tempering_high"]
                    )[..., sj, :].mean(axis=-1),
                }
                for metric, local_value in source_values.items():
                    destination = support[key][metric]
                    for phase_local, phase_position in enumerate(phase_global):
                        destination[
                            si,
                            local_worm_positions,
                            phase_position,
                        ] = local_value[:, phase_local]

            arm_sd = np.asarray(data["arm_endpoint_sd"])
            for candidate in candidate_by_lag[record.lag]:
                sj = source_local[candidate.source]
                hi = horizon_local[candidate.horizon]
                denominator = np.maximum(
                    np.abs(np.asarray(data["diagnostic_achieved_gap"])[..., sj]),
                    min_gap,
                )
                for contrast, (left, right) in PAIR_INDICES.items():
                    for metric in ANALYSIS_SIGNED_METRICS:
                        if metric == "endpoint_sd":
                            local_raw = (
                                arm_sd[:, :, :, left, hi, candidate.target, sj]
                                - arm_sd[:, :, :, right, hi, candidate.target, sj]
                            )
                        else:
                            local_raw = np.asarray(
                                data[f"response_{contrast}_{metric}"]
                            )[:, :, :, hi, candidate.target, sj]
                        local_norm = local_raw / denominator
                        raw_destination, norm_destination = signed[
                            (candidate.candidate_id, contrast, metric)
                        ]
                        for phase_local, phase_position in enumerate(phase_global):
                            raw_destination[
                                si, local_worm_positions, phase_position
                            ] = local_raw[:, phase_local]
                            norm_destination[
                                si, local_worm_positions, phase_position
                            ] = local_norm[:, phase_local]
                    local_w1 = np.asarray(
                        data[f"distance_{contrast}_endpoint_wasserstein1"]
                    )[:, :, :, hi, candidate.target, sj]
                    local_w1_norm = local_w1 / denominator
                    raw_destination, norm_destination = distances[
                        (candidate.candidate_id, contrast)
                    ]
                    for phase_local, phase_position in enumerate(phase_global):
                        raw_destination[
                            si, local_worm_positions, phase_position
                        ] = local_w1[:, phase_local]
                        norm_destination[
                            si, local_worm_positions, phase_position
                        ] = local_w1_norm[:, phase_local]
                    if q_count:
                        local_quantile = np.asarray(
                            data[f"response_{contrast}_endpoint_quantile"]
                        )[:, :, :, hi, :, candidate.target, sj]
                        local_quantile_norm = local_quantile / denominator[..., None]
                        destination = quantiles[
                            f"{candidate.candidate_id}:{contrast}"
                        ]
                        for phase_local, phase_position in enumerate(phase_global):
                            destination["raw"][
                                si, local_worm_positions, phase_position
                            ] = local_quantile[:, phase_local]
                            destination["normalized"][
                                si, local_worm_positions, phase_position
                            ] = local_quantile_norm[:, phase_local]
    if np.any(codes < 0):
        raise RuntimeError("one or more worms lack extracted chemical provenance")
    return signed, distances, quantiles, support, codes


def _support_context_summary(
    inputs: TargetedInputs,
    support: Mapping[str, Mapping[str, np.ndarray]],
    codes: np.ndarray,
    *,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[tuple[int, int, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, int, str], dict[str, Any]] = {}
    unique = sorted(
        {(candidate.lag, candidate.source, candidate.context) for candidate in inputs.candidates}
    )
    for lag, source, context in unique:
        raw_values = support[f"{lag}:{source}"]
        particles = float(inputs.manifest["particles"])
        minimum_ancestor_fraction = np.minimum(
            raw_values["distinct_ancestors_low"],
            raw_values["distinct_ancestors_high"],
        ) / particles
        genealogy_ok_0_10 = (
            minimum_ancestor_fraction
            >= config.genealogy_strong_min_ancestor_fraction
        ).astype(np.float32)
        genealogy_ok_0_20 = (
            minimum_ancestor_fraction
            >= config.genealogy_sensitivity_min_ancestor_fraction
        ).astype(np.float32)
        values = {
            **raw_values,
            "min_distinct_ancestor_fraction": minimum_ancestor_fraction,
            "genealogy_ok_0_10": genealogy_ok_0_10,
            "genealogy_ok_0_20": genealogy_ok_0_20,
            "genealogy_valid_0_10": raw_values["valid"] * genealogy_ok_0_10,
            "genealogy_valid_0_20": raw_values["valid"] * genealogy_ok_0_20,
        }
        contextual: dict[str, np.ndarray] = {}
        for name, value in values.items():
            reducer = "difference"
            if context.endswith("minus_baseline"):
                if name in {
                    "valid",
                    "achieved_gap_abs",
                    "ess_low",
                    "ess_high",
                    "distinct_ancestors_low",
                    "distinct_ancestors_high",
                    "min_distinct_ancestor_fraction",
                    "genealogy_ok_0_10",
                    "genealogy_ok_0_20",
                    "genealogy_valid_0_10",
                    "genealogy_valid_0_20",
                }:
                    reducer = "min"
                else:
                    reducer = "max"
            contextual[name] = _contextualize(
                value,
                context,
                codes,
                contrast_reducer=reducer,
            )
        worm = {name: value.mean(axis=0) for name, value in contextual.items()}
        genealogy_valid_fraction_0_10 = float(worm["genealogy_valid_0_10"].mean())
        genealogy_valid_fraction_0_20 = float(worm["genealogy_valid_0_20"].mean())
        summary: dict[str, Any] = {
            "valid_fraction": float(worm["valid"].mean()),
            "n_worms_valid_ge_half": int(np.sum(worm["valid"] >= 0.5)),
            "mean_achieved_gap_magnitude": float(worm["achieved_gap_abs"].mean()),
            "minimum_worm_achieved_gap_magnitude": float(
                worm["achieved_gap_abs"].min()
            ),
            "mean_ess_low": float(worm["ess_low"].mean()),
            "mean_ess_high": float(worm["ess_high"].mean()),
            "minimum_worm_ess_low": float(worm["ess_low"].min()),
            "minimum_worm_ess_high": float(worm["ess_high"].min()),
            "mean_max_weight_low": float(worm["max_weight_low"].mean()),
            "mean_max_weight_high": float(worm["max_weight_high"].mean()),
            "maximum_worm_max_weight_low": float(worm["max_weight_low"].max()),
            "maximum_worm_max_weight_high": float(worm["max_weight_high"].max()),
            "mean_distinct_ancestors_low": float(
                worm["distinct_ancestors_low"].mean()
            ),
            "mean_distinct_ancestors_high": float(
                worm["distinct_ancestors_high"].mean()
            ),
            "mean_min_distinct_ancestor_fraction": float(
                worm["min_distinct_ancestor_fraction"].mean()
            ),
            "minimum_worm_min_distinct_ancestor_fraction": float(
                worm["min_distinct_ancestor_fraction"].min()
            ),
            "genealogy_min_distinct_ancestor_fraction_strong": (
                config.genealogy_strong_min_ancestor_fraction
            ),
            "genealogy_min_distinct_ancestor_fraction_sensitivity": (
                config.genealogy_sensitivity_min_ancestor_fraction
            ),
            "genealogy_ok_fraction_0_10": float(
                worm["genealogy_ok_0_10"].mean()
            ),
            "genealogy_ok_fraction_0_20": float(
                worm["genealogy_ok_0_20"].mean()
            ),
            "genealogy_valid_fraction_0_10": genealogy_valid_fraction_0_10,
            "genealogy_valid_fraction_0_20": genealogy_valid_fraction_0_20,
            "genealogy_strong_gate_pass": bool(
                genealogy_valid_fraction_0_10 >= config.strong_valid_fraction
            ),
            "genealogy_sensitivity_gate_pass": bool(
                genealogy_valid_fraction_0_20 >= config.strong_valid_fraction
            ),
            "forced_tempering_rate_low": float(
                worm["forced_tempering_rate_low"].mean()
            ),
            "forced_tempering_rate_high": float(
                worm["forced_tempering_rate_high"].mean()
            ),
        }
        lookup[(lag, source, context)] = summary
        rows.append(
            {
                "source_neuron": inputs.neurons[source],
                "source_index": source,
                "source_lag_frames": lag,
                "source_to_cut_seconds": lag / inputs.fps,
                "context": context,
                "n_worms": len(inputs.worm_ids),
                "n_model_seeds": len(inputs.seeds),
                "n_particles": int(inputs.manifest["particles"]),
                "genealogy_gate_applicable": True,
                **summary,
                "matrix_orientation": ORIENTATION,
                "interpretation_limit": INTERPRETATION_LIMIT,
            }
        )
    return pd.DataFrame(rows), lookup


def _base_row(
    inputs: TargetedInputs,
    candidate: CandidateCell,
    contrast: str,
    support: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "queue_row": candidate.queue_row,
        "queue_rank": candidate.queue_rank,
        "source_neuron": candidate.source_neuron,
        "target_neuron": candidate.target_neuron,
        "source_index": candidate.source,
        "target_index": candidate.target,
        "source_lag_frames": candidate.lag,
        "source_to_cut_seconds": candidate.lag / inputs.fps,
        "horizon_frames": candidate.horizon,
        "forecast_horizon_seconds": candidate.horizon / inputs.fps,
        "source_to_readout_seconds": (candidate.lag + candidate.horizon)
        / inputs.fps,
        "context": candidate.context,
        "contrast": contrast,
        "n_worms": len(inputs.worm_ids),
        "n_model_seeds": len(inputs.seeds),
        "n_particles": int(inputs.manifest["particles"]),
        "valid_fraction": support["valid_fraction"],
        "genealogy_gate_applicable": True,
        "genealogy_min_distinct_ancestor_fraction_strong": support[
            "genealogy_min_distinct_ancestor_fraction_strong"
        ],
        "genealogy_min_distinct_ancestor_fraction_sensitivity": support[
            "genealogy_min_distinct_ancestor_fraction_sensitivity"
        ],
        "genealogy_valid_fraction_0_10": support[
            "genealogy_valid_fraction_0_10"
        ],
        "genealogy_valid_fraction_0_20": support[
            "genealogy_valid_fraction_0_20"
        ],
        "genealogy_strong_gate_pass": support["genealogy_strong_gate_pass"],
        "genealogy_sensitivity_gate_pass": support[
            "genealogy_sensitivity_gate_pass"
        ],
        "mean_achieved_gap_magnitude": support["mean_achieved_gap_magnitude"],
        "matrix_orientation": ORIENTATION,
        "row_axis": "target_neuron",
        "column_axis": "source_neuron",
        "chemical_identity_conditioned": False,
        "factual_arm_definition": FACTUAL_ARM_DEFINITION,
        "interpretation_limit": INTERPRETATION_LIMIT,
    }


def summarize_targeted_outputs(
    inputs: TargetedInputs,
    *,
    config: AnalysisConfig = AnalysisConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return signed/W1 cells, quantile shifts, and source support tables."""
    config.validate()
    signed, distances, quantiles, support_arrays, codes = _extract_candidate_arrays(
        inputs, min_gap=config.min_gap
    )
    support_frame, support_lookup = _support_context_summary(
        inputs, support_arrays, codes, config=config
    )
    bootstrap = _bootstrap_weights(
        len(inputs.worm_ids), config.bootstrap_replicates, config.random_seed
    )
    rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    for candidate in inputs.candidates:
        support = support_lookup[(candidate.lag, candidate.source, candidate.context)]
        for contrast in CONTRAST_NAMES:
            for metric in ANALYSIS_SIGNED_METRICS:
                raw, normalized = signed[(candidate.candidate_id, contrast, metric)]
                raw_context = _contextualize(raw, candidate.context, codes)
                normalized_context = _contextualize(
                    normalized, candidate.context, codes
                )
                summary = _summary(
                    normalized_context,
                    raw_context,
                    signed=True,
                    bootstrap_weights=bootstrap,
                )
                row = {
                    **_base_row(inputs, candidate, contrast, support),
                    "metric_family": "signed_contrast",
                    "metric": metric,
                    "signed": True,
                    "metric_source": (
                        "derived_from_arm_endpoint_sd"
                        if metric == "endpoint_sd"
                        else "archive_response"
                    ),
                    **summary,
                }
                row["evidence_label"] = _evidence_label(
                    summary,
                    signed=True,
                    valid_fraction=float(support["valid_fraction"]),
                    genealogy_strong_gate_pass=bool(
                        support["genealogy_strong_gate_pass"]
                    ),
                    config=config,
                )
                rows.append(row)
            raw_w1, normalized_w1 = distances[
                (candidate.candidate_id, contrast)
            ]
            signed_w1 = candidate.context.endswith("minus_baseline")
            raw_context = _contextualize(raw_w1, candidate.context, codes)
            normalized_context = _contextualize(
                normalized_w1, candidate.context, codes
            )
            summary = _summary(
                normalized_context,
                raw_context,
                signed=signed_w1,
                bootstrap_weights=bootstrap,
            )
            row = {
                **_base_row(inputs, candidate, contrast, support),
                "metric_family": "pairwise_wasserstein1",
                "metric": "endpoint_wasserstein1",
                "signed": signed_w1,
                "metric_source": "archive_distance",
                **summary,
            }
            row["evidence_label"] = _evidence_label(
                summary,
                signed=signed_w1,
                valid_fraction=float(support["valid_fraction"]),
                genealogy_strong_gate_pass=bool(
                    support["genealogy_strong_gate_pass"]
                ),
                config=config,
            )
            rows.append(row)
            if inputs.endpoint_quantiles:
                value = quantiles[f"{candidate.candidate_id}:{contrast}"]
                raw_q = _contextualize(value["raw"], candidate.context, codes)
                normalized_q = _contextualize(
                    value["normalized"], candidate.context, codes
                )
                for quantile_index, quantile in enumerate(inputs.endpoint_quantiles):
                    summary = _summary(
                        normalized_q[..., quantile_index],
                        raw_q[..., quantile_index],
                        signed=True,
                        bootstrap_weights=bootstrap,
                    )
                    qrow = {
                        **_base_row(inputs, candidate, contrast, support),
                        "metric_family": "endpoint_quantile_shift",
                        "metric": "endpoint_quantile",
                        "endpoint_quantile": quantile,
                        "signed": True,
                        **summary,
                    }
                    qrow["evidence_label"] = _evidence_label(
                        summary,
                        signed=True,
                        valid_fraction=float(support["valid_fraction"]),
                        genealogy_strong_gate_pass=bool(
                            support["genealogy_strong_gate_pass"]
                        ),
                        config=config,
                    )
                    quantile_rows.append(qrow)
    return pd.DataFrame(rows), pd.DataFrame(quantile_rows), support_frame


def _screen_consistency(
    inputs: TargetedInputs,
    cells: pd.DataFrame,
    *,
    screen_dir: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "candidate_id",
        "source_neuron",
        "target_neuron",
        "source_lag_frames",
        "horizon_frames",
        "context",
        "screen_method",
        "targeted_method",
        "same_sampler_family",
        "comparison_type",
        "screen_channel",
        "targeted_metric",
        "screen_mean_normalized",
        "targeted_n128_mean_normalized",
        "targeted_minus_screen",
        "direction_agreement",
        "magnitude_agreement",
        "screen_valid_fraction",
        "targeted_valid_fraction",
        "consistency_label",
        "interpretation_limit",
    ]
    if screen_dir is None:
        return pd.DataFrame(columns=columns), {
            "status": "not_requested",
            "rows": 0,
            "warning": "screen-vs-N128 comparison is descriptive and selection-conditioned",
        }
    screen_dir = Path(screen_dir).resolve()
    queue = screen_dir / "hypothesis_queue.csv"
    dense_path = screen_dir / "atlas_matrices.npz"
    if not queue.exists() or not dense_path.exists():
        raise RuntimeError("screen directory lacks hypothesis_queue.csv or atlas_matrices.npz")
    if sha256(queue) != str(inputs.manifest["hypothesis_queue_sha256"]):
        raise RuntimeError("screen queue does not match the targeted frozen queue hash")
    queue_frame = _selected_queue(queue)
    rows: list[dict[str, Any]] = []
    with np.load(dense_path, allow_pickle=False) as dense:
        required_metadata = {
            "neurons",
            "methods",
            "channels",
            "contexts",
            "source_lag_frames",
            "horizon_frames",
            "orientation",
            "primary_method",
        }
        missing = sorted(required_metadata.difference(dense.files))
        if missing:
            raise RuntimeError(f"dense screen archive lacks metadata {missing}")
        if tuple(dense["neurons"].astype(str)) != inputs.neurons:
            raise RuntimeError("dense screen neuron order differs from targeted archives")
        if str(_scalar(dense, "orientation")) != ORIENTATION:
            raise RuntimeError("dense screen orientation is not target-row/source-column")
        lags = tuple(int(x) for x in dense["source_lag_frames"])
        horizons = tuple(int(x) for x in dense["horizon_frames"])
        contexts = set(dense["contexts"].astype(str))
        methods = set(dense["methods"].astype(str))
        channels = set(dense["channels"].astype(str))
        lag_pos = {value: index for index, value in enumerate(lags)}
        horizon_pos = {value: index for index, value in enumerate(horizons)}
        queue_lookup = {
            int(row["__queue_row"]): row for _, row in queue_frame.iterrows()
        }
        for candidate in inputs.candidates:
            if candidate.screen_channel is None:
                continue
            mapping = SCREEN_TO_TARGETED_METRIC.get(candidate.screen_channel)
            if mapping is None:
                raise RuntimeError(
                    f"unsupported screen channel {candidate.screen_channel!r}"
                )
            family, targeted_metric = mapping
            if (
                candidate.lag not in lag_pos
                or candidate.horizon not in horizon_pos
                or candidate.context not in contexts
                or candidate.screen_channel not in channels
            ):
                raise RuntimeError("screen candidate lies outside the dense matrix grid")
            queue_row = queue_lookup.get(candidate.queue_row)
            if queue_row is None:
                raise RuntimeError("targeted candidate cannot be traced to its frozen queue row")
            screen_method = (
                str(queue_row["method"])
                if "method" in queue_row and pd.notna(queue_row["method"])
                else str(_scalar(dense, "primary_method"))
            )
            if screen_method not in methods:
                raise RuntimeError("queue screen method is absent from dense matrices")
            matrix_key = (
                f"mean_normalized__{screen_method}__{candidate.screen_channel}"
                f"__{candidate.context}"
            )
            valid_key = f"valid_fraction__{screen_method}__{candidate.context}"
            if matrix_key not in dense.files or valid_key not in dense.files:
                raise RuntimeError("dense screen archive lacks a requested candidate slice")
            matrix = np.asarray(dense[matrix_key])
            expected_shape = (
                len(lags),
                len(horizons),
                len(inputs.neurons),
                len(inputs.neurons),
            )
            if matrix.shape != expected_shape:
                raise RuntimeError("dense screen matrix has an invalid orientation/shape")
            screen_value = float(
                matrix[
                    lag_pos[candidate.lag],
                    horizon_pos[candidate.horizon],
                    candidate.target,
                    candidate.source,
                ]
            )
            if "mean_normalized" in queue_row and pd.notna(
                queue_row["mean_normalized"]
            ):
                if not np.isclose(
                    screen_value,
                    float(queue_row["mean_normalized"]),
                    rtol=1e-5,
                    atol=2e-6,
                ):
                    raise RuntimeError("frozen queue value disagrees with dense screen matrix")
            selected = cells[
                (cells.candidate_id == candidate.candidate_id)
                & (cells.contrast == "high_low")
                & (cells.metric == targeted_metric)
                & (
                    cells.metric_family
                    == (
                        "pairwise_wasserstein1"
                        if family == "distance"
                        else "signed_contrast"
                    )
                )
            ]
            if len(selected) != 1:
                raise RuntimeError("N128 summary lacks a unique screen-comparison row")
            target_row = selected.iloc[0]
            targeted_value = float(target_row.mean_normalized)
            signed = bool(target_row.signed)
            direction = (
                float(
                    abs(screen_value) > 1e-12
                    and abs(targeted_value) > 1e-12
                    and np.sign(screen_value) == np.sign(targeted_value)
                )
                if signed
                else float("nan")
            )
            magnitude = float(
                np.clip(
                    1
                    - abs(targeted_value - screen_value)
                    / (abs(targeted_value) + abs(screen_value) + 1e-12),
                    0,
                    1,
                )
            )
            screen_valid = float(
                np.asarray(dense[valid_key])[lag_pos[candidate.lag], candidate.source]
            )
            targeted_valid = float(target_row.valid_fraction)
            same_sampler_family = screen_method == "progressive_bridge_smc"
            comparison_type = (
                "particle_escalation_within_progressive_bridge"
                if same_sampler_family
                else "cross_sampler_screen_to_progressive_bridge"
            )
            if targeted_valid < 0.5:
                label = "n128_support_limited"
            elif signed and direction < 1:
                label = "direction_discordant"
            elif signed:
                label = "direction_consistent"
            else:
                label = "unsigned_magnitude_descriptive"
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "source_neuron": candidate.source_neuron,
                    "target_neuron": candidate.target_neuron,
                    "source_lag_frames": candidate.lag,
                    "horizon_frames": candidate.horizon,
                    "context": candidate.context,
                    "screen_method": screen_method,
                    "targeted_method": METHOD,
                    "same_sampler_family": same_sampler_family,
                    "comparison_type": comparison_type,
                    "screen_channel": candidate.screen_channel,
                    "targeted_metric": targeted_metric,
                    "screen_mean_normalized": screen_value,
                    "targeted_n128_mean_normalized": targeted_value,
                    "targeted_minus_screen": targeted_value - screen_value,
                    "direction_agreement": direction,
                    "magnitude_agreement": magnitude,
                    "screen_valid_fraction": screen_valid,
                    "targeted_valid_fraction": targeted_valid,
                    "consistency_label": label,
                    "interpretation_limit": (
                        "post-screen N128 consistency; descriptive and selection-conditioned"
                    ),
                }
            )
    frame = pd.DataFrame(rows, columns=columns)
    comparable = frame
    signed_rows = comparable[np.isfinite(comparable.direction_agreement)]
    summary = {
        "status": "computed",
        "rows": len(frame),
        "direction_agreement_fraction": (
            float(signed_rows.direction_agreement.mean())
            if len(signed_rows)
            else None
        ),
        "median_magnitude_agreement": (
            float(frame.magnitude_agreement.median()) if len(frame) else None
        ),
        "median_absolute_targeted_minus_screen": (
            float(frame.targeted_minus_screen.abs().median()) if len(frame) else None
        ),
        "same_sampler_family_rows": (
            int(frame.same_sampler_family.sum()) if len(frame) else 0
        ),
        "cross_sampler_rows": (
            int((~frame.same_sampler_family.astype(bool)).sum()) if len(frame) else 0
        ),
        "warning": (
            "The N128 runs target selected screen cells; agreement is descriptive, "
            "selection-conditioned, and not independent confirmation. Direct-screen "
            "rows are additionally cross-sampler comparisons."
        ),
    }
    return frame, summary


def _report(
    inputs: TargetedInputs,
    cells: pd.DataFrame,
    quantiles: pd.DataFrame,
    support: pd.DataFrame,
    screen_summary: Mapping[str, Any],
) -> str:
    consistent = cells[cells.evidence_label == "model_relative_consistent"]
    unsupported = cells[cells.evidence_label == "unsupported_model_output"]
    top = cells.assign(abs_effect=cells.mean_normalized.abs()).sort_values(
        ["abs_effect", "valid_fraction"], ascending=[False, False]
    ).head(10)
    lines = [
        "# Targeted three-arm N128 model-relative analysis",
        "",
        f"- {len(inputs.worm_ids)} worms; seeds {list(inputs.seeds)}; "
        f"{len(inputs.candidates)} frozen candidate cells.",
        f"- {len(cells)} signed/Wasserstein summaries and {len(quantiles)} quantile summaries.",
        f"- {len(consistent)} rows met the internal model-relative consistency rule; "
        f"{len(unsupported)} were support-limited.",
        f"- Screen comparison status: {screen_summary['status']}.",
        "",
        "The low/high arms are soft repaired-history regimes and the factual arm is a "
        "learned-flow rollout from observed history. None is an experimental or causal "
        "intervention. Chemical labels are event-stratified under a binary-any-stimulus "
        "generator, and lag labels are not physical-delay estimates.",
        "",
        "Progressive-SMC genealogy is audited separately from the original ESS, "
        "maximum-weight, and achieved-gap validity flag. A row can therefore remain a "
        "valid model output while failing the stronger genealogy rule. Strong labels "
        "require at least 80% genealogy-adjusted support when each episode must retain "
        "at least 10% distinct ancestors in both arms; the 20% threshold is reported "
        "only as sensitivity.",
        "",
        "## Source support and genealogy",
        "",
        "| Source | Context | Lag | Raw valid | Genealogy valid (10%) | Genealogy valid (20%) | Strong gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in support.itertuples(index=False):
        lines.append(
            f"| {row.source_neuron} | {row.context} | {int(row.source_lag_frames)} | "
            f"{row.valid_fraction:.3f} | {row.genealogy_valid_fraction_0_10:.3f} | "
            f"{row.genealogy_valid_fraction_0_20:.3f} | "
            f"{'pass' if row.genealogy_strong_gate_pass else 'fail'} |"
        )
    lines.extend(
        [
        "",
        "## Highest-magnitude reviewed rows",
        "",
        "| Source → target | Context | Contrast | Metric | Mean | 95% worm bootstrap | Support | Label |",
        "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.source_neuron} → {row.target_neuron} | {row.context} | "
            f"{row.contrast} | {row.metric} | {row.mean_normalized:+.3f} | "
            f"[{row.ci_2_5:+.3f}, {row.ci_97_5:+.3f}] | "
            f"{row.valid_fraction:.2f} | {row.evidence_label} |"
        )
    lines.extend(
        [
            "",
            "Intervals resample worms only. Events are reduced within each worm, and "
            "generator seeds are averaged within each worm before bootstrapping.",
            "The consistency label captures a pointwise population-sign/interval rule; "
            "it does not require stable worm ranking across generator seeds. Intervals "
            "are post-selection and are not multiplicity-adjusted across the targeted "
            "cell or quantile summaries.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_checksums(output: Path) -> None:
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )


def build_targeted_confirmation_analysis(
    run_dir: Path,
    output_dir: Path,
    *,
    screen_dir: Path | None = None,
    config: AnalysisConfig = AnalysisConfig(),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the canonical internal N128 targeted-analysis bundle."""
    config.validate()
    output = Path(output_dir).resolve()
    canonical = {
        "targeted_cells.csv",
        "targeted_quantile_shifts.csv",
        "support_diagnostics.csv",
        "screen_consistency.csv",
        "summary.json",
        "validation.json",
        "protocol.json",
        "input_checksums.csv",
        "REPORT.md",
        "manifest.json",
        "checksums.sha256",
    }
    if output.exists() and any((output / name).exists() for name in canonical):
        if not overwrite:
            raise FileExistsError(f"canonical targeted outputs already exist in {output}")
        for name in canonical:
            path = output / name
            if path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    inputs = discover_and_validate_targeted_inputs(run_dir, config=config)
    cells, quantiles, support = summarize_targeted_outputs(inputs, config=config)
    screen, screen_summary = _screen_consistency(
        inputs, cells, screen_dir=screen_dir
    )
    cells.to_csv(output / "targeted_cells.csv", index=False)
    quantiles.to_csv(output / "targeted_quantile_shifts.csv", index=False)
    support.to_csv(output / "support_diagnostics.csv", index=False)
    screen.to_csv(output / "screen_consistency.csv", index=False)

    top = (
        cells.assign(abs_effect=cells.mean_normalized.abs())
        .sort_values(["valid_fraction", "abs_effect"], ascending=[False, False])
        .head(25)
        .drop(columns="abs_effect")
        .to_dict(orient="records")
    )
    evidence_counts = {
        str(key): int(value) for key, value in cells.evidence_label.value_counts().items()
    }
    summary = {
        "status": "reviewed_model_relative",
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_worms": len(inputs.worm_ids),
        "n_model_seeds": len(inputs.seeds),
        "n_candidates": len(inputs.candidates),
        "n_cell_rows": len(cells),
        "n_quantile_rows": len(quantiles),
        "evidence_label_counts": evidence_counts,
        "genealogy_support": {
            "hard_strong_min_ancestor_fraction": (
                config.genealogy_strong_min_ancestor_fraction
            ),
            "sensitivity_min_ancestor_fraction": (
                config.genealogy_sensitivity_min_ancestor_fraction
            ),
            "source_context_rows": len(support),
            "strong_gate_pass_rows": int(
                support.genealogy_strong_gate_pass.astype(bool).sum()
            ),
            "sensitivity_gate_pass_rows": int(
                support.genealogy_sensitivity_gate_pass.astype(bool).sum()
            ),
        },
        "screen_vs_n128": screen_summary,
        "top_reviewed_rows": top,
        "claim_boundary": INTERPRETATION_LIMIT,
        "experimental_confirmation_label_assigned": False,
        "external_reference_data_used": False,
    }
    _write_json(output / "summary.json", summary)

    protocol = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "input_archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        "input_manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "method": METHOD,
        "particle_count": config.expected_particles,
        "normalization": (
            f"each event contrast / max(abs(achieved high-low source gap), {config.min_gap})"
        ),
        "independent_unit": "worm",
        "event_handling": "events reduced within worm according to the frozen queue context",
        "model_seed_handling": (
            "retain seed-specific worm values, report seed agreement, then average seeds "
            "within worm before group summaries and bootstrap"
        ),
        "interval": {
            "method": "deterministic percentile bootstrap",
            "unit": "worm",
            "replicates": config.bootstrap_replicates,
            "seed": config.random_seed,
        },
        "matrix_orientation": ORIENTATION,
        "row_axis": "target_neuron",
        "column_axis": "selected source_neuron",
        "contrasts": list(CONTRAST_NAMES),
        "signed_metrics": list(ANALYSIS_SIGNED_METRICS),
        "pairwise_distance": "endpoint one-dimensional Wasserstein-1",
        "endpoint_quantiles": list(inputs.endpoint_quantiles),
        "support_rule": (
            "joint low/high progressive-bridge support is conservatively attached to all "
            "three-arm contrasts; support failures are retained"
        ),
        "genealogy_support": {
            "episode_fraction_definition": (
                "min(distinct_ancestors_low, distinct_ancestors_high) / n_particles"
            ),
            "hard_strong_min_ancestor_fraction": (
                config.genealogy_strong_min_ancestor_fraction
            ),
            "hard_strong_gate_definition": (
                "mean(diagnostic_valid * I[episode ancestor fraction >= 0.10]) "
                f">= {config.strong_valid_fraction} within source/context"
            ),
            "sensitivity_min_ancestor_fraction": (
                config.genealogy_sensitivity_min_ancestor_fraction
            ),
            "sensitivity_use": "reporting only; never changes label or ranking",
            "raw_diagnostic_valid_preserved": True,
            "model_outputs_preserved_when_genealogy_gate_fails": True,
        },
        "factual_arm_definition": FACTUAL_ARM_DEFINITION,
        "screen_consistency": (
            "optional frozen N32/dense-atlas comparison for high-low only; descriptive, "
            "selection-conditioned, and not independent confirmation"
        ),
        "chemical_warning": (
            "chemical contexts are event-stratified under a binary-any-stimulus generator, "
            "not chemical-conditioned effects"
        ),
        "claim_boundary": INTERPRETATION_LIMIT,
        "external_reference_firewall": (
            "no connectome, receptor atlas, Randi, Cook, Bentley, or SBTG reference is loaded"
        ),
        "evidence_labels": {
            "model_relative_consistent": (
                "internal support/stability/interval rule plus f=0.10 genealogy gate"
            ),
            "model_relative_uncertain": "adequate support but incomplete internal stability",
            "model_relative_descriptive": "unsigned distance with no zero-centered signed null",
            "unsupported_model_output": "joint bridge support below the declared gate",
        },
    }
    _write_json(output / "protocol.json", protocol)

    input_paths = {
        inputs.manifest_path,
        inputs.validation_path,
        inputs.queue_path,
        *(record.path for record in inputs.records),
        *(record.checkpoint for record in inputs.records),
    }
    if screen_dir is not None:
        input_paths.update(
            {
                Path(screen_dir).resolve() / "hypothesis_queue.csv",
                Path(screen_dir).resolve() / "atlas_matrices.npz",
            }
        )
    input_checksums = pd.DataFrame(
        [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path.resolve()),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(input_paths, key=lambda value: str(value))
        ]
    )
    input_checksums.to_csv(output / "input_checksums.csv", index=False)

    validation = {
        "status": "passed",
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "targeted_v2_manifest_fingerprint": True,
            "targeted_run_validation_complete": True,
            "full_group_fold_seed_grid": True,
            "heldout_worm_coverage_once_per_seed": True,
            "checkpoint_and_queue_hashes": True,
            "base_seed_requested_and_resolved_device": True,
            "actual_schedule_and_chemical_provenance": True,
            "source_window_cut_lag_geometry": True,
            "cut_times_reconstructed_from_frozen_schedule": True,
            "target_row_source_column_orientation": True,
            "all_response_and_support_values_finite": True,
            "arm_contrast_and_log_sd_identities": True,
            "wasserstein_nonnegative": True,
            "quantile_monotonicity_and_shift_identity": True,
            "eventwise_achieved_gap_normalization": True,
            "worm_level_bootstrap_and_within_worm_seed_averaging": True,
            "raw_validity_and_separate_genealogy_support_reported": True,
            "external_reference_firewall": True,
        },
        "archive_count": len(inputs.records),
        "expected_archive_count": len(inputs.groups)
        * len(inputs.folds)
        * len(inputs.seeds),
        "n_worms": len(inputs.worm_ids),
        "folds": list(inputs.folds),
        "seeds": list(inputs.seeds),
        "source_lags": [group.lag for group in inputs.groups],
        "resolved_device": inputs.resolved_device,
        "screen_vs_n128": screen_summary,
        "claim_boundary": INTERPRETATION_LIMIT,
    }
    _write_json(output / "validation.json", validation)
    (output / "REPORT.md").write_text(
        _report(inputs, cells, quantiles, support, screen_summary)
    )
    manifest = {
        "status": "complete",
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_run_dir": str(inputs.run_dir),
        "input_run_manifest_sha256": sha256(inputs.manifest_path),
        "input_queue_sha256": sha256(inputs.queue_path),
        "screen_dir": str(Path(screen_dir).resolve()) if screen_dir else None,
        "matrix_orientation": ORIENTATION,
        "artifacts": {
            "targeted_cells": "targeted_cells.csv",
            "targeted_quantile_shifts": "targeted_quantile_shifts.csv",
            "support_diagnostics": "support_diagnostics.csv",
            "screen_consistency": "screen_consistency.csv",
            "summary": "summary.json",
            "validation": "validation.json",
            "protocol": "protocol.json",
            "input_checksums": "input_checksums.csv",
            "report": "REPORT.md",
        },
        "claim_boundary": INTERPRETATION_LIMIT,
    }
    _write_json(output / "manifest.json", manifest)
    _write_checksums(output)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--screen-dir",
        type=Path,
        required=True,
        help=(
            "frozen targeted-selection bundle containing the exact staged "
            "hypothesis_queue.csv and copied atlas_matrices.npz used for the screen"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--random-seed", type=int, default=20_260_829)
    parser.add_argument("--min-gap", type=float, default=0.10)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.50)
    parser.add_argument("--strong-valid-fraction", type=float, default=0.80)
    parser.add_argument("--strong-sign-consistency", type=float, default=0.80)
    parser.add_argument("--expected-particles", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_targeted_confirmation_analysis(
        args.run_dir,
        args.output_dir,
        screen_dir=args.screen_dir,
        config=AnalysisConfig(
            bootstrap_replicates=args.bootstrap_replicates,
            random_seed=args.random_seed,
            min_gap=args.min_gap,
            minimum_valid_fraction=args.minimum_valid_fraction,
            strong_valid_fraction=args.strong_valid_fraction,
            strong_sign_consistency=args.strong_sign_consistency,
            expected_particles=args.expected_particles,
        ),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
