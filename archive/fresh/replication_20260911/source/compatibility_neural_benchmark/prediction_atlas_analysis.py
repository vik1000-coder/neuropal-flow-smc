"""Build the reviewed, worm-level neural prediction atlas.

The sampler archives are deliberately treated as an immutable source layer.  This
module validates their provenance and geometry, converts the sampler convention
``[source, horizon, target]`` to the atlas convention ``[target, source]`` exactly
once, and performs inference with worms (not events, particles, or model seeds) as
the independent units.

Chemical panels in this atlas are event-stratified views of a generator trained on
a binary any-stimulus covariate.  They are *not* chemically conditioned estimates.
The emitted metadata repeats that limitation so downstream interfaces cannot lose
it accidentally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t


METHODS = ("direct_importance", "progressive_bridge_smc")
PHASES = ("baseline", "onset", "active", "offset", "recovery")
CHANNELS = (
    "endpoint_mean",
    "cumulative_mean",
    "peak_mean",
    "event_probability",
    "endpoint_sd",
    "endpoint_log_sd",
    "endpoint_wasserstein1",
)
CHEMICALS = ("butanone", "pentanedione", "nacl")
CHEMICAL_CODE_TO_NAME = {1: "butanone", 2: "pentanedione", 3: "nacl"}
LAG_DEFINITION = "source-window end to prediction cut"
ORIENTATION = "target_row_source_column"
GENERATOR_STIMULUS_ENCODING = "binary_any_stimulus"
ARCHIVE_SCHEMA_VERSION = "prediction_atlas_response_v2"
MANIFEST_SCHEMA_VERSION = "prediction_atlas_manifest_v2"
RESPONSE_AXES = ("heldout_worm", "phase", "event", "source", "horizon", "target")
EPISODE_SEED_DEFINITION = (
    "sha256(model_id) prefix + base_seed + 1000003*fold + 1009*generator_seed "
    "+ 9176*worm_index + 131*phase_index + 17*event_index + 53*source_lag; "
    "sampler method deliberately omitted; modulo 2**31-1"
)
COMMON_NOISE_DEFINITION = (
    "within each estimator, low/high future arms use identical flow base-noise "
    "seeds and aligned particle-row order"
)
COMMON_DIAGNOSTICS = (
    "target_low",
    "target_high",
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
PROGRESSIVE_DIAGNOSTICS_1D = (
    "candidate_ess_low",
    "candidate_ess_high",
    "ess_fraction_low",
    "ess_fraction_high",
    "candidate_max_weight_low",
    "candidate_max_weight_high",
    "min_step_ess_low",
    "min_step_ess_high",
    "distinct_ancestors_low",
    "distinct_ancestors_high",
    "branch_factor",
    "future_branch_factor",
    "candidate_particles",
    "future_particles",
    "source_window_start_step",
    "source_window_end_step_exclusive",
)
PROGRESSIVE_DIAGNOSTICS_STEP = (
    "step_ess_low",
    "step_ess_high",
    "step_candidate_ess_low",
    "step_candidate_ess_high",
    "step_ess_fraction_low",
    "step_ess_fraction_high",
    "step_max_weight_low",
    "step_max_weight_high",
    "step_candidate_max_weight_low",
    "step_candidate_max_weight_high",
    "step_tempering_resamples_low",
    "step_tempering_resamples_high",
    "step_forced_tempering_low",
    "step_forced_tempering_high",
    "step_beta_low",
    "step_beta_high",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
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


def _json_dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not result:
        raise ValueError(f"cannot construct archive key from {value!r}")
    return result


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    value = data[key]
    if value.size != 1:
        raise RuntimeError(f"archive field {key!r} must be scalar")
    return value.item()


def _all_finite(name: str, value: np.ndarray, path: Path) -> None:
    if not np.isfinite(value).all():
        count = int(np.size(value) - np.isfinite(value).sum())
        raise RuntimeError(f"{path}: {name} contains {count} non-finite values")


def _normalize_chemical_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "")


@dataclass(frozen=True)
class ManifestInfo:
    path: Path
    raw: dict[str, object]
    methods: tuple[str, ...]
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    source_lags: tuple[int, ...]
    horizons: tuple[int, ...]
    phases: tuple[str, ...]
    model_id: str
    cohort_mode: str
    n_worms: int
    n_neurons: int
    fps: float
    history_frames: int
    source_window_frames: int
    schema_version: str
    schema_fingerprint: str
    schedule_by_worm: Mapping[str, Mapping[str, object]]
    source_run: Path
    source_manifest_path: Path
    source_manifest_sha256: str
    source_manifest: Mapping[str, object]
    fold_assignments: Path
    fold_assignments_sha256: str
    fold_by_worm: Mapping[str, int]
    checkpoint_phase: str
    leaderboard_path: Path
    leaderboard_sha256: str
    selected_predictive_scores: Mapping[str, object]
    chemical_encoding_gate: Mapping[str, object]


@dataclass(frozen=True)
class ArchiveRecord:
    path: Path
    manifest_path: Path
    method: str
    fold: int
    seed: int
    source_lag: int
    particles: int
    model_id: str
    checkpoint: Path
    checkpoint_sha256: str
    worm_ids: tuple[str, ...]
    worm_indices: tuple[int, ...]
    wall_seconds: float


@dataclass(frozen=True)
class AtlasInputs:
    manifests: tuple[ManifestInfo, ...]
    records: tuple[ArchiveRecord, ...]
    methods: tuple[str, ...]
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    source_lags: tuple[int, ...]
    horizons: tuple[int, ...]
    phases: tuple[str, ...]
    neurons: tuple[str, ...]
    worm_ids: tuple[str, ...]
    fps: float
    model_id: str
    cohort_mode: str
    history_frames: int
    source_window_frames: int
    schema_version: str
    schema_fingerprint: str
    resolved_device: str


@dataclass(frozen=True)
class AnalysisConfig:
    bootstrap_replicates: int = 256
    random_seed: int = 20_260_829
    min_gap: float = 0.10
    min_valid_fraction: float = 0.50
    strong_valid_fraction: float = 0.80
    strong_sign_consistency: float = 0.80
    genealogy_strong_min_ancestor_fraction: float = 0.10
    genealogy_sensitivity_min_ancestor_fraction: float = 0.20
    bh_alpha: float = 0.05
    prediction_cells_per_slice: int = 100
    hypothesis_queue_size: int = 1000
    dashboard_candidates: int = 150
    sign_flip_replicates: int = 2_048
    primary_method: str = "progressive_bridge_smc"

    def validate(self) -> None:
        if self.bootstrap_replicates < 32:
            raise ValueError("at least 32 worm-bootstrap replicates are required")
        if self.min_gap <= 0:
            raise ValueError("min_gap must be positive")
        if not 0 <= self.min_valid_fraction <= self.strong_valid_fraction <= 1:
            raise ValueError("validity thresholds are inconsistent")
        if not (
            0
            < self.genealogy_strong_min_ancestor_fraction
            < self.genealogy_sensitivity_min_ancestor_fraction
            <= 1
        ):
            raise ValueError("genealogy thresholds must satisfy 0 < strong < sensitivity <= 1")
        if not (
            math.isclose(self.genealogy_strong_min_ancestor_fraction, 0.10)
            and math.isclose(self.genealogy_sensitivity_min_ancestor_fraction, 0.20)
        ):
            raise ValueError("canonical genealogy thresholds are frozen at 0.10 and 0.20")
        if not 0 < self.bh_alpha < 1:
            raise ValueError("bh_alpha must lie in (0, 1)")
        if self.prediction_cells_per_slice < 1 or self.hypothesis_queue_size < 1:
            raise ValueError("table retention limits must be positive")
        if self.sign_flip_replicates < 256:
            raise ValueError("at least 256 candidate sign-flip replicates are required")


CONTEXTS = (
    "baseline",
    "onset",
    "active",
    "offset",
    "recovery",
    "state_average",
    "onset_minus_baseline",
    "butanone_onset",
    "pentanedione_onset",
    "nacl_onset",
    "butanone_onset_minus_baseline",
    "pentanedione_onset_minus_baseline",
    "nacl_onset_minus_baseline",
)


def context_metadata(context: str) -> dict[str, object]:
    chemical = next((name for name in CHEMICALS if context.startswith(name + "_")), None)
    is_contrast = context.endswith("onset_minus_baseline")
    return {
        "context": context,
        "chemical": chemical,
        "is_phase_contrast": is_contrast,
        "event_stratified": chemical is not None,
        "conditioning_status": (
            "exploratory_event_stratified_under_binary_any_stimulus_generator"
            if chemical is not None
            else "binary_any_stimulus_conditioned"
        ),
    }


def _require_source_provenance(
    raw: Mapping[str, object], *, manifest_path: Path
) -> dict[str, object]:
    """Independently audit the frozen generator run and fold/score inputs."""
    source_run = Path(str(raw["source_run"])).resolve()
    source_manifest_path = source_run / "manifest.json"
    if not source_run.is_dir() or not source_manifest_path.is_file():
        raise RuntimeError(f"{manifest_path}: source run or source manifest is missing")
    source_manifest_hash = sha256(source_manifest_path)
    if source_manifest_hash != str(raw["source_run_manifest_sha256"]):
        raise RuntimeError(f"{manifest_path}: source-run manifest hash audit failed")
    source = json.loads(source_manifest_path.read_text())
    source_required = {
        "status",
        "cohort_mode",
        "folds",
        "seeds",
        "lag_frames",
        "n_worms",
        "n_neurons",
        "fold_assignments",
        "fold_assignments_sha256",
        "stimulus_schema_version",
        "stimulus_schema_fingerprint",
        "stimulus_schema",
        "encodings",
    }
    missing = sorted(source_required.difference(source))
    if missing:
        raise RuntimeError(f"{source_manifest_path}: source manifest is missing {missing}")
    if source["status"] != "complete":
        raise RuntimeError(f"{source_manifest_path}: source run status is not complete")
    scalar_matches = {
        "cohort_mode": str(raw["cohort_mode"]),
        "n_worms": int(raw["n_worms"]),
        "n_neurons": int(raw["n_neurons"]),
        "lag_frames": int(raw["history_frames"]),
        "stimulus_schema_version": str(raw["stimulus_schema"]["version"]),
        "stimulus_schema_fingerprint": str(raw["stimulus_schema"]["fingerprint"]),
    }
    for key, expected in scalar_matches.items():
        value = source[key]
        if isinstance(expected, int):
            value = int(value)
        else:
            value = str(value)
        if value != expected:
            raise RuntimeError(
                f"{source_manifest_path}: source manifest {key} disagrees with sampler manifest"
            )
    if source["stimulus_schema"] != raw["stimulus_schema"]:
        raise RuntimeError(
            f"{source_manifest_path}: source stimulus schema differs from sampler manifest"
        )
    if not set(int(value) for value in raw["folds"]).issubset(
        int(value) for value in source["folds"]
    ) or not set(int(value) for value in raw["seeds"]).issubset(
        int(value) for value in source["seeds"]
    ):
        raise RuntimeError(
            f"{source_manifest_path}: sampler fold/seed grid is outside source model grid"
        )
    encodings = source["encodings"]
    if not isinstance(encodings, list) or not any(
        isinstance(item, Mapping)
        and item.get("encoding") == GENERATOR_STIMULUS_ENCODING
        and not bool(item.get("sensitivity_only", False))
        and not bool(item.get("shuffled", False))
        for item in encodings
    ):
        raise RuntimeError(
            f"{source_manifest_path}: source run lacks a primary binary-any-stimulus arm"
        )

    fold_assignments = Path(str(raw["fold_assignments"])).resolve()
    if not fold_assignments.is_file():
        raise RuntimeError(f"{manifest_path}: fold-assignment file is missing")
    fold_hash = sha256(fold_assignments)
    if fold_hash != str(raw["fold_assignments_sha256"]):
        raise RuntimeError(f"{manifest_path}: fold-assignment hash audit failed")
    source_fold_path = Path(str(source["fold_assignments"])).resolve()
    if (
        source_fold_path != fold_assignments
        or str(source["fold_assignments_sha256"]) != fold_hash
    ):
        raise RuntimeError(
            f"{source_manifest_path}: source and sampler fold-assignment provenance differ"
        )
    fold_frame = pd.read_csv(fold_assignments)
    if not {"worm_id", "outer_fold"}.issubset(fold_frame.columns):
        raise RuntimeError(f"{fold_assignments}: fold file lacks worm_id/outer_fold")
    if fold_frame["worm_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{fold_assignments}: duplicated worm IDs")
    try:
        fold_by_worm = dict(
            zip(
                fold_frame["worm_id"].astype(str),
                fold_frame["outer_fold"].astype(int),
            )
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{fold_assignments}: invalid outer-fold values") from error
    schedule_worms = [str(item["worm_id"]) for item in raw["stimulus_schema"]["schedules"]]
    if any(worm not in fold_by_worm for worm in schedule_worms):
        raise RuntimeError(f"{fold_assignments}: cohort worm is missing from fold file")
    cohort_folds = {fold_by_worm[worm] for worm in schedule_worms}
    if cohort_folds != set(int(value) for value in source["folds"]):
        raise RuntimeError(
            f"{fold_assignments}: cohort fold coverage differs from source manifest"
        )

    leaderboard_path = source_run / "chemical_leaderboard.csv"
    if not leaderboard_path.is_file():
        raise RuntimeError(f"{source_run}: chemical_leaderboard.csv is missing")
    leaderboard_hash = sha256(leaderboard_path)
    leaderboard = pd.read_csv(leaderboard_path)
    leaderboard_required = {
        "rank",
        "model_id",
        "stimulus_encoding",
        "energy__mean",
        "energy__std",
        "energy__count",
        "energy__stim_balanced__mean",
        "energy__stim_balanced__std",
        "energy__stim_balanced__count",
        "energy__worm_chemical_balanced__mean",
        "energy__worm_chemical_balanced__std",
        "energy__worm_chemical_balanced__count",
        "energy__chemical_butanone__mean",
        "energy__chemical_butanone__std",
        "energy__chemical_butanone__count",
        "energy__chemical_pentanedione__mean",
        "energy__chemical_pentanedione__std",
        "energy__chemical_pentanedione__count",
        "energy__chemical_nacl__mean",
        "energy__chemical_nacl__std",
        "energy__chemical_nacl__count",
        "variogram__mean",
        "variogram__std",
        "variogram__count",
    }
    missing = sorted(leaderboard_required.difference(leaderboard.columns))
    if missing:
        raise RuntimeError(f"{leaderboard_path}: leaderboard is missing {missing}")
    numeric_columns = sorted(
        leaderboard_required.difference({"model_id", "stimulus_encoding"})
    )
    numeric = leaderboard[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if leaderboard.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{leaderboard_path}: leaderboard scores are empty or non-finite")
    selected = leaderboard.loc[leaderboard["model_id"].astype(str) == str(raw["model_id"])]
    if len(selected) != 1:
        raise RuntimeError(
            f"{leaderboard_path}: selected generator must appear exactly once"
        )
    selected_row = selected.iloc[0]
    if (
        int(selected_row["rank"]) != 1
        or str(selected_row["stimulus_encoding"]) != GENERATOR_STIMULUS_ENCODING
        or int(leaderboard["rank"].min()) != 1
    ):
        raise RuntimeError(
            f"{leaderboard_path}: selected generator is not the rank-1 binary arm"
        )
    score_column = "energy__worm_chemical_balanced__mean"
    selected_score = float(selected_row[score_column])
    if selected_score > float(leaderboard[score_column].min()) + 1e-12:
        raise RuntimeError(
            f"{leaderboard_path}: selected generator did not win balanced energy"
        )
    chemical_mask = leaderboard["stimulus_encoding"].astype(str).isin(
        ("chemical_scalar", "chemical_onehot", "chemical_plus_position_onehot")
    )
    chemical_scores = leaderboard.loc[chemical_mask, score_column].astype(float)
    if chemical_scores.empty:
        raise RuntimeError(f"{leaderboard_path}: no chemical-encoding comparison arms")
    best_chemical = float(chemical_scores.min())
    if best_chemical < selected_score - 1e-12:
        raise RuntimeError(
            f"{leaderboard_path}: chemical encoding beat the recorded selected generator"
        )
    selected_predictive_scores: dict[str, object] = {
        "rank": int(selected_row["rank"]),
        "model_id": str(selected_row["model_id"]),
        "stimulus_encoding": str(selected_row["stimulus_encoding"]),
    }
    for column in leaderboard.columns:
        if column in selected_predictive_scores:
            continue
        value = selected_row[column]
        if column.endswith("__count"):
            selected_predictive_scores[column] = int(value)
        elif column.endswith("__mean") or column.endswith("__std"):
            selected_predictive_scores[column] = float(value)
    chemical_gate = {
        "passed": False,
        "decision": "chemical encodings did not improve the lower-is-better balanced energy over binary",
        "metric": score_column,
        "binary_score": selected_score,
        "best_chemical_score": best_chemical,
        "difference_chemical_minus_binary": best_chemical - selected_score,
    }
    return {
        "source_run": source_run,
        "source_manifest_path": source_manifest_path,
        "source_manifest_sha256": source_manifest_hash,
        "source_manifest": source,
        "fold_assignments": fold_assignments,
        "fold_assignments_sha256": fold_hash,
        "fold_by_worm": fold_by_worm,
        "leaderboard_path": leaderboard_path,
        "leaderboard_sha256": leaderboard_hash,
        "selected_predictive_scores": selected_predictive_scores,
        "chemical_encoding_gate": chemical_gate,
    }


def _require_manifest(path: Path) -> ManifestInfo:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text())
    required = {
        "created_utc",
        "run_spec_fingerprint",
        "manifest_schema_version",
        "methods",
        "folds",
        "seeds",
        "source_lag_frames",
        "horizon_frames",
        "phases",
        "model_id",
        "cohort_mode",
        "n_worms",
        "n_neurons",
        "fps",
        "history_frames",
        "source_window_frames",
        "lag_definition",
        "stimulus_schema",
        "response_keys",
        "particles",
        "response_axes",
        "matrix_internal_orientation",
        "distribution_scale_floor",
        "minimum_effective_sample_size",
        "maximum_normalized_weight",
        "minimum_achieved_source_fraction",
        "progressive_branch_factor",
        "progressive_future_branch_factor",
        "stimulus_generator_encoding",
        "chemical_identity_conditioned",
        "base_seed",
        "episode_seed_definition",
        "common_noise_definition",
        "requested_device",
        "source_run",
        "source_run_manifest_sha256",
        "checkpoint_phase",
        "fold_assignments",
        "fold_assignments_sha256",
        "stimulus_schema_version",
        "stimulus_schema_fingerprint",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise RuntimeError(f"{path}: manifest is missing {missing}")
    if raw["lag_definition"] != LAG_DEFINITION:
        raise RuntimeError(f"{path}: incompatible lag definition")
    if raw["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(f"{path}: incompatible manifest schema version")
    fingerprint_spec = {
        key: value
        for key, value in raw.items()
        if key not in {"created_utc", "run_spec_fingerprint"}
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_spec,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if str(raw["run_spec_fingerprint"]) != fingerprint:
        raise RuntimeError(f"{path}: sampler manifest fingerprint audit failed")
    if tuple(raw["response_keys"]) != tuple(f"response_{x}" for x in CHANNELS):
        raise RuntimeError(f"{path}: response-key schema mismatch")
    if tuple(raw["response_axes"]) != RESPONSE_AXES:
        raise RuntimeError(f"{path}: response-axis schema mismatch")
    if raw["matrix_internal_orientation"] != "source,horizon,target":
        raise RuntimeError(f"{path}: internal matrix orientation mismatch")
    if raw["stimulus_generator_encoding"] != GENERATOR_STIMULUS_ENCODING:
        raise RuntimeError(f"{path}: generator stimulus encoding mismatch")
    if bool(raw["chemical_identity_conditioned"]):
        raise RuntimeError(f"{path}: analysis requires the binary, non-chemical generator")
    if int(raw["base_seed"]) != 20_260_829:
        raise RuntimeError(f"{path}: unexpected base seed")
    if raw["episode_seed_definition"] != EPISODE_SEED_DEFINITION:
        raise RuntimeError(f"{path}: episode seed definition mismatch")
    if raw["common_noise_definition"] != COMMON_NOISE_DEFINITION:
        raise RuntimeError(f"{path}: common-noise definition mismatch")
    if not str(raw["requested_device"]):
        raise RuntimeError(f"{path}: requested device is empty")
    for key in (
        "distribution_scale_floor",
        "minimum_effective_sample_size",
        "maximum_normalized_weight",
        "minimum_achieved_source_fraction",
    ):
        if not np.isfinite(float(raw[key])) or float(raw[key]) <= 0:
            raise RuntimeError(f"{path}: invalid manifest parameter {key}")
    if int(raw["progressive_branch_factor"]) < 1 or int(
        raw["progressive_future_branch_factor"]
    ) < 1:
        raise RuntimeError(f"{path}: invalid progressive branch factor")
    methods = tuple(str(x) for x in raw["methods"])
    if not methods or not set(methods).issubset(METHODS):
        raise RuntimeError(f"{path}: unsupported or empty method list {methods}")
    phases = tuple(str(x) for x in raw["phases"])
    if phases != PHASES:
        raise RuntimeError(f"{path}: expected phase order {PHASES}, found {phases}")
    schema = raw["stimulus_schema"]
    if not isinstance(schema, dict):
        raise RuntimeError(f"{path}: stimulus_schema must be an object")
    for key in ("version", "fingerprint", "schedules"):
        if key not in schema:
            raise RuntimeError(f"{path}: stimulus_schema lacks {key}")
    if (
        str(raw["stimulus_schema_version"]) != str(schema["version"])
        or str(raw["stimulus_schema_fingerprint"]) != str(schema["fingerprint"])
    ):
        raise RuntimeError(f"{path}: top-level and nested stimulus schema disagree")
    if int(raw["particles"]) < 1:
        raise RuntimeError(f"{path}: particles must be positive")
    schedules = schema["schedules"]
    if not isinstance(schedules, list) or len(schedules) != int(raw["n_worms"]):
        raise RuntimeError(f"{path}: schedule count does not equal n_worms")
    schedule_by_worm: dict[str, Mapping[str, object]] = {}
    for schedule in schedules:
        worm_id = str(schedule.get("worm_id", ""))
        if not worm_id or worm_id in schedule_by_worm:
            raise RuntimeError(f"{path}: schedule worm IDs are missing or duplicated")
        schedule_required = {
            "analysis_fps",
            "native_fps",
            "event_intervals_seconds",
            "source_recording",
            "resampling_provenance",
        }
        schedule_missing = sorted(schedule_required.difference(schedule))
        if schedule_missing:
            raise RuntimeError(
                f"{path}: schedule {worm_id} lacks timing provenance {schedule_missing}"
            )
        analysis_fps = float(schedule["analysis_fps"])
        native_fps = float(schedule["native_fps"])
        if (
            not np.isfinite(analysis_fps)
            or analysis_fps <= 0
            or not np.isfinite(native_fps)
            or native_fps <= 0
            or not np.isclose(analysis_fps, float(raw["fps"]), rtol=0, atol=1e-12)
        ):
            raise RuntimeError(f"{path}: schedule {worm_id} has invalid frame-rate provenance")
        intervals = np.asarray(schedule["event_intervals_seconds"], dtype=np.float64)
        if (
            intervals.shape != (3, 2)
            or not np.isfinite(intervals).all()
            or np.any(intervals[:, 0] < 0)
            or np.any(intervals[:, 1] <= intervals[:, 0])
            or np.any(intervals[1:, 0] < intervals[:-1, 1])
        ):
            raise RuntimeError(f"{path}: schedule {worm_id} has invalid event intervals")
        if not str(schedule["source_recording"]) or not str(
            schedule["resampling_provenance"]
        ):
            raise RuntimeError(f"{path}: schedule {worm_id} has empty timing provenance")
        codes = tuple(int(x) for x in schedule.get("chemical_code_by_event", ()))
        names = tuple(str(x) for x in schedule.get("chemical_name_by_event", ()))
        _validate_chemical_row(codes, names, where=f"{path}: schedule {worm_id}")
        schedule_by_worm[worm_id] = schedule
    source_lags = tuple(int(x) for x in raw["source_lag_frames"])
    horizons = tuple(int(x) for x in raw["horizon_frames"])
    folds = tuple(int(x) for x in raw["folds"])
    seeds = tuple(int(x) for x in raw["seeds"])
    if (
        not source_lags
        or not horizons
        or not folds
        or not seeds
        or len(set(source_lags)) != len(source_lags)
        or len(set(horizons)) != len(horizons)
        or len(set(folds)) != len(folds)
        or len(set(seeds)) != len(seeds)
        or min(source_lags) < 0
        or min(horizons) < 1
    ):
        raise RuntimeError(f"{path}: invalid lag/horizon/fold/seed grid")
    source = _require_source_provenance(raw, manifest_path=path)
    return ManifestInfo(
        path=path.resolve(),
        raw=raw,
        methods=methods,
        folds=folds,
        seeds=seeds,
        source_lags=source_lags,
        horizons=horizons,
        phases=phases,
        model_id=str(raw["model_id"]),
        cohort_mode=str(raw["cohort_mode"]),
        n_worms=int(raw["n_worms"]),
        n_neurons=int(raw["n_neurons"]),
        fps=float(raw["fps"]),
        history_frames=int(raw["history_frames"]),
        source_window_frames=int(raw["source_window_frames"]),
        schema_version=str(schema["version"]),
        schema_fingerprint=str(schema["fingerprint"]),
        schedule_by_worm=schedule_by_worm,
        source_run=source["source_run"],
        source_manifest_path=source["source_manifest_path"],
        source_manifest_sha256=str(source["source_manifest_sha256"]),
        source_manifest=source["source_manifest"],
        fold_assignments=source["fold_assignments"],
        fold_assignments_sha256=str(source["fold_assignments_sha256"]),
        fold_by_worm=source["fold_by_worm"],
        checkpoint_phase=str(raw["checkpoint_phase"]),
        leaderboard_path=source["leaderboard_path"],
        leaderboard_sha256=str(source["leaderboard_sha256"]),
        selected_predictive_scores=source["selected_predictive_scores"],
        chemical_encoding_gate=source["chemical_encoding_gate"],
    )


def _validate_chemical_row(
    codes: Sequence[int], names: Sequence[str], *, where: str
) -> None:
    if tuple(sorted(int(x) for x in codes)) != (1, 2, 3) or len(names) != 3:
        raise RuntimeError(f"{where}: chemical codes must be a permutation of 1,2,3")
    for code, name in zip(codes, names):
        expected = _normalize_chemical_name(CHEMICAL_CODE_TO_NAME[int(code)])
        if _normalize_chemical_name(name) != expected:
            raise RuntimeError(
                f"{where}: chemical code/name mismatch ({code!r}, {name!r})"
            )


def _compatible_manifest(left: ManifestInfo, right: ManifestInfo) -> None:
    fields = (
        "source_lags",
        "horizons",
        "phases",
        "model_id",
        "cohort_mode",
        "n_worms",
        "n_neurons",
        "fps",
        "history_frames",
        "source_window_frames",
        "schema_version",
        "schema_fingerprint",
    )
    for field in fields:
        if getattr(left, field) != getattr(right, field):
            raise RuntimeError(
                f"manifest mismatch for {field}: {left.path} versus {right.path}"
            )
    if tuple(left.schedule_by_worm) != tuple(right.schedule_by_worm):
        raise RuntimeError("manifest schedule worm order changed across runs")
    for worm_id in left.schedule_by_worm:
        a = left.schedule_by_worm[worm_id]
        b = right.schedule_by_worm[worm_id]
        if a != b:
            raise RuntimeError(f"schedule metadata changed for {worm_id}")
    for key in (
        "manifest_schema_version",
        "response_keys",
        "response_axes",
        "matrix_internal_orientation",
        "distribution_scale_floor",
        "minimum_effective_sample_size",
        "maximum_normalized_weight",
        "minimum_achieved_source_fraction",
        "progressive_branch_factor",
        "progressive_future_branch_factor",
        "stimulus_generator_encoding",
        "chemical_identity_conditioned",
        "base_seed",
        "episode_seed_definition",
        "common_noise_definition",
        "requested_device",
        "source_run",
        "source_run_manifest_sha256",
        "checkpoint_phase",
        "fold_assignments",
        "fold_assignments_sha256",
    ):
        if left.raw[key] != right.raw[key]:
            raise RuntimeError(f"manifest mismatch for {key}")
    if (
        left.source_manifest_sha256 != right.source_manifest_sha256
        or left.fold_assignments_sha256 != right.fold_assignments_sha256
        or left.leaderboard_sha256 != right.leaderboard_sha256
        or left.selected_predictive_scores != right.selected_predictive_scores
    ):
        raise RuntimeError("source-run, fold, or predictive-score provenance changed across runs")


def _archive_paths(run_dir: Path) -> tuple[Path, ...]:
    paths = tuple(sorted((run_dir / "responses").glob("**/*.npz")))
    if not paths:
        raise RuntimeError(f"no response archives found under {run_dir / 'responses'}")
    return paths


def discover_and_validate_inputs(
    run_dirs: Sequence[Path],
    *,
    required_methods: Sequence[str] = METHODS,
) -> AtlasInputs:
    """Validate all archives and return their immutable index.

    Validation is intentionally eager: every response and diagnostic array is read
    and checked for finiteness before any derived artifact is written.
    """
    if not run_dirs:
        raise ValueError("at least one sampler run directory is required")
    manifests = tuple(_require_manifest(Path(root).resolve() / "manifest.json") for root in run_dirs)
    first = manifests[0]
    for manifest in manifests[1:]:
        _compatible_manifest(first, manifest)
    declared_methods = tuple(dict.fromkeys(m for manifest in manifests for m in manifest.methods))
    all_folds = tuple(sorted({fold for manifest in manifests for fold in manifest.folds}))
    all_seeds = tuple(sorted({seed for manifest in manifests for seed in manifest.seeds}))
    source_folds = tuple(sorted(int(value) for value in first.source_manifest["folds"]))
    source_seeds = tuple(sorted(int(value) for value in first.source_manifest["seeds"]))
    if all_folds != source_folds or all_seeds != source_seeds:
        raise RuntimeError(
            "combined sampler manifests do not cover the complete frozen source model grid "
            f"(folds {all_folds} != {source_folds} or seeds {all_seeds} != {source_seeds})"
        )
    required = tuple(str(x) for x in required_methods)
    if set(declared_methods) != set(required):
        raise RuntimeError(
            f"method coverage mismatch: declared {sorted(declared_methods)}, "
            f"required {sorted(required)}"
        )

    records: list[ArchiveRecord] = []
    cell_paths: dict[tuple[str, int, int, int], Path] = {}
    neuron_order: tuple[str, ...] | None = None
    worm_index_to_id: dict[int, str] = {}
    checkpoint_by_cell: dict[tuple[int, int], tuple[Path, str]] = {}
    particles_by_method: dict[str, int] = {}
    resolved_devices: set[str] = set()

    for run_dir, manifest in zip((Path(x).resolve() for x in run_dirs), manifests):
        for path in _archive_paths(run_dir):
            with np.load(path, allow_pickle=False) as data:
                required_fields = {
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
                    "source_lag_frames",
                    "source_lag_seconds",
                    "lag_definition",
                    "source_to_readout_seconds",
                    "repair_frames",
                    "source_window_frames",
                    "source_window_bounds_semantics",
                    "n_particles",
                    "horizon_frames",
                    "horizon_seconds",
                    "phase_names",
                    "fps",
                    "response_keys",
                    "response_axes",
                    "worm_indices",
                    "worm_ids",
                    "chemical_code_by_worm_event",
                    "chemical_name_by_worm_event",
                    "stimulus_schema_version",
                    "stimulus_schema_fingerprint",
                    "stimulus_generator_encoding",
                    "chemical_identity_conditioned",
                    "distribution_scale_floor",
                    "wall_seconds",
                    "neurons",
                    "cut_times",
                    "source_window_bounds",
                } | {f"response_{channel}" for channel in CHANNELS} | {
                    f"diagnostic_{name}" for name in COMMON_DIAGNOSTICS
                }
                method_peek = str(_scalar(data, "method")) if "method" in data.files else ""
                if method_peek == "progressive_bridge_smc":
                    required_fields |= {
                        f"diagnostic_{name}"
                        for name in PROGRESSIVE_DIAGNOSTICS_1D
                        + PROGRESSIVE_DIAGNOSTICS_STEP
                    }
                missing = sorted(required_fields.difference(data.files))
                if missing:
                    raise RuntimeError(f"{path}: archive is missing required fields {missing}")
                if _scalar(data, "status") != "complete":
                    raise RuntimeError(f"{path}: archive status is not complete")
                if str(_scalar(data, "archive_schema_version")) != ARCHIVE_SCHEMA_VERSION:
                    raise RuntimeError(f"{path}: archive schema version mismatch")
                method = str(_scalar(data, "method"))
                if method not in manifest.methods:
                    raise RuntimeError(f"{path}: method is not declared by its manifest")
                model_id = str(_scalar(data, "model_id"))
                if model_id != first.model_id:
                    raise RuntimeError(f"{path}: model ID mismatch")
                fold = int(_scalar(data, "fold"))
                seed = int(_scalar(data, "seed"))
                lag = int(_scalar(data, "source_lag_frames"))
                particles = int(_scalar(data, "n_particles"))
                if particles != int(manifest.raw["particles"]):
                    raise RuntimeError(f"{path}: particle count differs from manifest")
                wall_seconds = float(_scalar(data, "wall_seconds"))
                if not np.isfinite(wall_seconds) or wall_seconds <= 0:
                    raise RuntimeError(f"{path}: wall_seconds must be finite and positive")
                if int(_scalar(data, "base_seed")) != int(manifest.raw["base_seed"]):
                    raise RuntimeError(f"{path}: archive base seed mismatch")
                if str(_scalar(data, "episode_seed_definition")) != EPISODE_SEED_DEFINITION:
                    raise RuntimeError(f"{path}: archive episode seed definition mismatch")
                if str(_scalar(data, "common_noise_definition")) != COMMON_NOISE_DEFINITION:
                    raise RuntimeError(f"{path}: archive common-noise definition mismatch")
                if str(_scalar(data, "requested_device")) != str(
                    manifest.raw["requested_device"]
                ):
                    raise RuntimeError(f"{path}: archive requested-device mismatch")
                resolved_device = str(_scalar(data, "resolved_device"))
                if not resolved_device:
                    raise RuntimeError(f"{path}: archive resolved device is empty")
                resolved_devices.add(resolved_device)
                if (
                    fold not in manifest.folds
                    or seed not in manifest.seeds
                    or lag not in manifest.source_lags
                ):
                    raise RuntimeError(f"{path}: fold/seed/lag lies outside manifest grid")
                cell = (method, lag, fold, seed)
                if cell in cell_paths:
                    raise RuntimeError(f"duplicate archive cell {cell}: {cell_paths[cell]} and {path}")
                cell_paths[cell] = path
                if int(_scalar(data, "history_frames")) != first.history_frames:
                    raise RuntimeError(f"{path}: history length mismatch")
                source_window = int(_scalar(data, "source_window_frames"))
                if source_window != first.source_window_frames:
                    raise RuntimeError(f"{path}: source-window width mismatch")
                if str(_scalar(data, "source_window_bounds_semantics")) != "[inclusive_start,exclusive_stop)":
                    raise RuntimeError(f"{path}: source-window bound semantics mismatch")
                if int(_scalar(data, "repair_frames")) != lag + source_window:
                    raise RuntimeError(f"{path}: repair prefix does not equal lag plus source window")
                if str(_scalar(data, "lag_definition")) != LAG_DEFINITION:
                    raise RuntimeError(f"{path}: lag-definition mismatch")
                if not np.isclose(float(_scalar(data, "source_lag_seconds")), lag / first.fps):
                    raise RuntimeError(f"{path}: source-lag seconds mismatch")
                if tuple(int(x) for x in data["horizon_frames"]) != first.horizons:
                    raise RuntimeError(f"{path}: horizon grid mismatch")
                expected_horizon_seconds = np.asarray(first.horizons) / first.fps
                if not np.allclose(data["horizon_seconds"], expected_horizon_seconds):
                    raise RuntimeError(f"{path}: horizon seconds mismatch")
                expected_source_to_readout = (lag + np.asarray(first.horizons)) / first.fps
                if not np.allclose(data["source_to_readout_seconds"], expected_source_to_readout):
                    raise RuntimeError(f"{path}: source-to-readout timing mismatch")
                if tuple(data["phase_names"].astype(str)) != PHASES:
                    raise RuntimeError(f"{path}: phase order mismatch")
                if not np.isclose(float(_scalar(data, "fps")), first.fps, rtol=0, atol=1e-8):
                    raise RuntimeError(f"{path}: archive fps mismatch")
                if tuple(data["response_keys"].astype(str)) != tuple(
                    f"response_{channel}" for channel in CHANNELS
                ):
                    raise RuntimeError(f"{path}: archive response-key schema mismatch")
                if tuple(data["response_axes"].astype(str)) != RESPONSE_AXES:
                    raise RuntimeError(f"{path}: archive response-axis schema mismatch")
                if str(_scalar(data, "stimulus_schema_version")) != first.schema_version:
                    raise RuntimeError(f"{path}: stimulus schema version mismatch")
                if str(_scalar(data, "stimulus_schema_fingerprint")) != first.schema_fingerprint:
                    raise RuntimeError(f"{path}: stimulus schema fingerprint mismatch")
                if str(_scalar(data, "stimulus_generator_encoding")) != GENERATOR_STIMULUS_ENCODING:
                    raise RuntimeError(f"{path}: archive generator encoding mismatch")
                if bool(_scalar(data, "chemical_identity_conditioned")):
                    raise RuntimeError(f"{path}: archive incorrectly declares chemical conditioning")
                if not np.isclose(
                    float(_scalar(data, "distribution_scale_floor")),
                    float(manifest.raw["distribution_scale_floor"]),
                    rtol=0,
                    atol=1e-12,
                ):
                    raise RuntimeError(f"{path}: distribution scale floor mismatch")

                checkpoint = Path(str(_scalar(data, "checkpoint"))).resolve()
                checkpoint_hash = str(_scalar(data, "checkpoint_sha256"))
                if not checkpoint.exists() or sha256(checkpoint) != checkpoint_hash:
                    raise RuntimeError(f"{path}: checkpoint path/hash audit failed")
                expected_checkpoint = (
                    manifest.source_run
                    / "checkpoints"
                    / manifest.checkpoint_phase
                    / f"{model_id}__L{first.history_frames}__f{fold}__s{seed}.pt"
                ).resolve()
                if checkpoint != expected_checkpoint:
                    raise RuntimeError(
                        f"{path}: checkpoint is outside the frozen source model grid"
                    )
                checkpoint_cell = (fold, seed)
                current_checkpoint = (checkpoint, checkpoint_hash)
                previous_checkpoint = checkpoint_by_cell.get(checkpoint_cell)
                if previous_checkpoint is None:
                    checkpoint_by_cell[checkpoint_cell] = current_checkpoint
                elif previous_checkpoint != current_checkpoint:
                    raise RuntimeError(
                        f"{path}: checkpoint changed across methods/lags for {checkpoint_cell}"
                    )
                previous_particles = particles_by_method.setdefault(method, particles)
                if previous_particles != particles:
                    raise RuntimeError(f"{path}: particle count changed within method {method}")

                neurons = tuple(data["neurons"].astype(str))
                if len(neurons) != first.n_neurons or len(set(neurons)) != len(neurons):
                    raise RuntimeError(f"{path}: invalid neuron axis")
                if neuron_order is None:
                    neuron_order = neurons
                elif neurons != neuron_order:
                    raise RuntimeError(f"{path}: neuron order changed")
                worm_ids = tuple(data["worm_ids"].astype(str))
                worm_indices = tuple(int(x) for x in data["worm_indices"])
                if len(worm_ids) != len(worm_indices) or len(set(worm_ids)) != len(worm_ids):
                    raise RuntimeError(f"{path}: invalid held-out worm identifiers")
                for worm_id, worm_index in zip(worm_ids, worm_indices):
                    if not 0 <= worm_index < first.n_worms:
                        raise RuntimeError(f"{path}: worm index out of range")
                    expected_id = tuple(first.schedule_by_worm)[worm_index]
                    if worm_id != expected_id:
                        raise RuntimeError(
                            f"{path}: worm index/ID mismatch ({worm_index}, {worm_id!r})"
                        )
                    if worm_index in worm_index_to_id and worm_index_to_id[worm_index] != worm_id:
                        raise RuntimeError(f"{path}: worm index maps to multiple IDs")
                    if manifest.fold_by_worm[worm_id] != fold:
                        raise RuntimeError(
                            f"{path}: held-out worm {worm_id!r} is assigned to a different fold"
                        )
                    worm_index_to_id[worm_index] = worm_id

                codes = data["chemical_code_by_worm_event"].astype(int)
                names = data["chemical_name_by_worm_event"].astype(str)
                if codes.shape != (len(worm_ids), 3) or names.shape != codes.shape:
                    raise RuntimeError(f"{path}: invalid chemical metadata shape")
                for position, worm_id in enumerate(worm_ids):
                    _validate_chemical_row(codes[position], names[position], where=f"{path}: {worm_id}")
                    schedule = first.schedule_by_worm[worm_id]
                    if not np.array_equal(codes[position], np.asarray(schedule["chemical_code_by_event"], dtype=int)):
                        raise RuntimeError(f"{path}: chemical event permutation differs from manifest for {worm_id}")
                    if tuple(names[position]) != tuple(str(x) for x in schedule["chemical_name_by_event"]):
                        raise RuntimeError(f"{path}: chemical event names differ from manifest for {worm_id}")

                cut_times = data["cut_times"].astype(np.int64)
                bounds = data["source_window_bounds"].astype(np.int64)
                expected_cut_shape = (len(worm_ids), len(PHASES), 3)
                if cut_times.shape != expected_cut_shape or bounds.shape != expected_cut_shape + (2,):
                    raise RuntimeError(f"{path}: cut/source-window shape mismatch")
                if np.any(cut_times < 0) or np.any(bounds < 0):
                    raise RuntimeError(f"{path}: negative cut or source-window bound")
                if not np.all(bounds[..., 1] - bounds[..., 0] == source_window):
                    raise RuntimeError(f"{path}: source-window width invariant failed")
                if not np.all(bounds[..., 1] - 1 == cut_times - lag):
                    raise RuntimeError(f"{path}: source-window lag invariant failed")

                d = len(neurons)
                h = len(first.horizons)
                expected_response_shape = (len(worm_ids), len(PHASES), 3, d, h, d)
                expected_diag_shape = (len(worm_ids), len(PHASES), 3, d)
                for channel in CHANNELS:
                    value = data[f"response_{channel}"]
                    if value.shape != expected_response_shape:
                        raise RuntimeError(
                            f"{path}: response_{channel} shape {value.shape} != {expected_response_shape}"
                        )
                    _all_finite(f"response_{channel}", value, path)
                event_probability = data["response_event_probability"]
                if np.any(event_probability < -1.0 - 1e-6) or np.any(
                    event_probability > 1.0 + 1e-6
                ):
                    raise RuntimeError(f"{path}: event-probability effect lies outside [-1,1]")
                if np.any(data["response_endpoint_wasserstein1"] < -1e-7):
                    raise RuntimeError(f"{path}: endpoint W1 contains negative distances")

                common: dict[str, np.ndarray] = {}
                for name in COMMON_DIAGNOSTICS:
                    value = data[f"diagnostic_{name}"]
                    if value.shape != expected_diag_shape:
                        raise RuntimeError(f"{path}: diagnostic_{name} shape mismatch")
                    _all_finite(f"diagnostic_{name}", value, path)
                    common[name] = np.asarray(value, dtype=np.float64)
                gaps = common["achieved_gap"]
                valid = common["valid"]
                if not np.all(np.isin(valid, (0.0, 1.0))):
                    raise RuntimeError(f"{path}: diagnostic_valid must be binary")
                if np.any(common["target_gap"] <= 0):
                    raise RuntimeError(f"{path}: diagnostic_target_gap must be positive")
                if not np.allclose(
                    common["target_gap"],
                    common["target_high"] - common["target_low"],
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise RuntimeError(f"{path}: target-gap identity failed")
                if not np.allclose(
                    gaps,
                    common["achieved_high"] - common["achieved_low"],
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise RuntimeError(f"{path}: achieved-gap identity failed")
                if not np.allclose(
                    common["endpoint_sd_floor"],
                    float(manifest.raw["distribution_scale_floor"]),
                    rtol=0,
                    atol=1e-12,
                ):
                    raise RuntimeError(f"{path}: endpoint SD floor diagnostic mismatch")

                ess_low = common["ess_low"]
                ess_high = common["ess_high"]
                max_low = common["max_weight_low"]
                max_high = common["max_weight_high"]
                branch = int(manifest.raw["progressive_branch_factor"])
                if method == "direct_importance":
                    if (
                        np.any(ess_low < 1 - 1e-5)
                        or np.any(ess_high < 1 - 1e-5)
                        or np.any(ess_low > particles + 1e-4)
                        or np.any(ess_high > particles + 1e-4)
                    ):
                        raise RuntimeError(f"{path}: direct ESS lies outside [1,N]")
                    direct_lower = 1.0 / particles
                    if (
                        np.any(max_low < direct_lower - 1e-6)
                        or np.any(max_high < direct_lower - 1e-6)
                        or np.any(max_low > 1 + 1e-6)
                        or np.any(max_high > 1 + 1e-6)
                    ):
                        raise RuntimeError(
                            f"{path}: direct max weight lies outside [1/N,1]"
                        )
                else:
                    # Progressive diagnostics report branch-equivalent ESS and
                    # max weight.  Their raw candidate counterparts are audited
                    # below, including the exact branch-factor identities.
                    if (
                        np.any(ess_low < 1.0 / branch - 1e-5)
                        or np.any(ess_high < 1.0 / branch - 1e-5)
                        or np.any(ess_low > particles + 1e-4)
                        or np.any(ess_high > particles + 1e-4)
                        or np.any(max_low < 1.0 / particles - 1e-6)
                        or np.any(max_high < 1.0 / particles - 1e-6)
                        or np.any(max_low > branch + 1e-6)
                        or np.any(max_high > branch + 1e-6)
                    ):
                        raise RuntimeError(
                            f"{path}: progressive equivalent ESS/max-weight range failed"
                        )

                    one_d: dict[str, np.ndarray] = {}
                    for name in PROGRESSIVE_DIAGNOSTICS_1D:
                        value = data[f"diagnostic_{name}"]
                        if value.shape != expected_diag_shape:
                            raise RuntimeError(f"{path}: diagnostic_{name} shape mismatch")
                        _all_finite(f"diagnostic_{name}", value, path)
                        one_d[name] = np.asarray(value, dtype=np.float64)
                    steps = lag + source_window
                    step: dict[str, np.ndarray] = {}
                    for name in PROGRESSIVE_DIAGNOSTICS_STEP:
                        value = data[f"diagnostic_{name}"]
                        if value.shape != expected_diag_shape + (steps,):
                            raise RuntimeError(f"{path}: diagnostic_{name} step shape mismatch")
                        _all_finite(f"diagnostic_{name}", value, path)
                        step[name] = np.asarray(value, dtype=np.float64)
                    future_branch = int(manifest.raw["progressive_future_branch_factor"])
                    candidates = particles * branch
                    for name, expected in (
                        ("branch_factor", branch),
                        ("future_branch_factor", future_branch),
                        ("candidate_particles", candidates),
                        ("future_particles", particles * future_branch),
                        ("source_window_start_step", 0),
                        ("source_window_end_step_exclusive", source_window),
                    ):
                        if not np.allclose(one_d[name], expected, rtol=0, atol=0):
                            raise RuntimeError(f"{path}: diagnostic_{name} constant mismatch")
                    for side in ("low", "high"):
                        candidate_ess = one_d[f"candidate_ess_{side}"]
                        ess_fraction = one_d[f"ess_fraction_{side}"]
                        candidate_max = one_d[f"candidate_max_weight_{side}"]
                        if (
                            np.any(candidate_ess < 1 - 1e-5)
                            or np.any(candidate_ess > candidates + 1e-4)
                            or np.any(candidate_max < 1.0 / candidates - 1e-6)
                            or np.any(candidate_max > 1 + 1e-6)
                            or np.any(ess_fraction < 1.0 / candidates - 1e-6)
                            or np.any(ess_fraction > 1 + 1e-6)
                        ):
                            raise RuntimeError(
                                f"{path}: progressive candidate ESS/max-weight range failed"
                            )
                        if not np.allclose(
                            candidate_ess / branch,
                            common[f"ess_{side}"],
                            rtol=2e-5,
                            atol=1e-5,
                        ) or not np.allclose(
                            candidate_ess / candidates,
                            ess_fraction,
                            rtol=2e-5,
                            atol=1e-6,
                        ):
                            raise RuntimeError(f"{path}: progressive ESS identities failed")
                        if not np.allclose(
                            candidate_max * branch,
                            common[f"max_weight_{side}"],
                            rtol=2e-5,
                            atol=1e-6,
                        ):
                            raise RuntimeError(
                                f"{path}: progressive max-weight identity failed"
                            )
                        step_ess = step[f"step_ess_{side}"]
                        step_candidate_ess = step[f"step_candidate_ess_{side}"]
                        step_fraction = step[f"step_ess_fraction_{side}"]
                        step_max = step[f"step_max_weight_{side}"]
                        step_candidate_max = step[f"step_candidate_max_weight_{side}"]
                        if (
                            np.any(step_candidate_ess < 1 - 1e-5)
                            or np.any(step_candidate_ess > candidates + 1e-4)
                            or np.any(step_ess < 1.0 / branch - 1e-5)
                            or np.any(step_ess > particles + 1e-4)
                            or np.any(step_fraction < 1.0 / candidates - 1e-6)
                            or np.any(step_fraction > 1 + 1e-6)
                            or np.any(step_candidate_max < 1.0 / candidates - 1e-6)
                            or np.any(step_candidate_max > 1 + 1e-6)
                            or np.any(step_max < 1.0 / particles - 1e-6)
                            or np.any(step_max > branch + 1e-6)
                        ):
                            raise RuntimeError(
                                f"{path}: progressive step ESS/max-weight range failed"
                            )
                        if (
                            not np.allclose(step_candidate_ess / branch, step_ess, rtol=2e-5, atol=1e-5)
                            or not np.allclose(step_candidate_ess / candidates, step_fraction, rtol=2e-5, atol=1e-6)
                            or not np.allclose(step_candidate_max * branch, step_max, rtol=2e-5, atol=1e-6)
                            or not np.allclose(step_ess[..., -1], common[f"ess_{side}"], rtol=2e-5, atol=1e-5)
                            or not np.allclose(step_candidate_ess[..., -1], candidate_ess, rtol=2e-5, atol=1e-5)
                            or not np.allclose(step_max[..., -1], common[f"max_weight_{side}"], rtol=2e-5, atol=1e-6)
                            or not np.allclose(step_candidate_max[..., -1], candidate_max, rtol=2e-5, atol=1e-6)
                            or not np.allclose(step_ess.min(axis=-1), one_d[f"min_step_ess_{side}"], rtol=2e-5, atol=1e-5)
                        ):
                            raise RuntimeError(
                                f"{path}: progressive terminal/step identities failed"
                            )
                        resamples = step[f"step_tempering_resamples_{side}"]
                        forced = step[f"step_forced_tempering_{side}"]
                        beta = step[f"step_beta_{side}"]
                        if np.any(resamples < 0) or not np.allclose(resamples, np.round(resamples)):
                            raise RuntimeError(f"{path}: tempering resamples are invalid")
                        if not np.all(np.isin(forced, (0.0, 1.0))):
                            raise RuntimeError(f"{path}: forced-tempering flags are not binary")
                        expected_beta = np.concatenate(
                            (
                                np.arange(1, source_window + 1, dtype=np.float64)
                                / source_window,
                                np.ones(lag, dtype=np.float64),
                            )
                        )
                        if not np.allclose(beta, expected_beta, rtol=0, atol=1e-7):
                            raise RuntimeError(f"{path}: progressive beta schedule mismatch")
                        ancestors = one_d[f"distinct_ancestors_{side}"]
                        if (
                            np.any(ancestors < 1)
                            or np.any(ancestors > particles)
                            or not np.allclose(ancestors, np.round(ancestors))
                        ):
                            raise RuntimeError(f"{path}: distinct-ancestor count is invalid")

                expected_valid = (
                    (ess_low >= float(manifest.raw["minimum_effective_sample_size"]))
                    & (ess_high >= float(manifest.raw["minimum_effective_sample_size"]))
                    & (max_low <= float(manifest.raw["maximum_normalized_weight"]))
                    & (max_high <= float(manifest.raw["maximum_normalized_weight"]))
                    & (
                        gaps
                        >= float(manifest.raw["minimum_achieved_source_fraction"])
                        * common["target_gap"]
                    )
                )
                if not np.array_equal(valid.astype(bool), expected_valid):
                    raise RuntimeError(f"{path}: diagnostic_valid does not match declared gates")

                records.append(
                    ArchiveRecord(
                        path=path.resolve(),
                        manifest_path=manifest.path,
                        method=method,
                        fold=fold,
                        seed=seed,
                        source_lag=lag,
                        particles=particles,
                        model_id=model_id,
                        checkpoint=checkpoint,
                        checkpoint_sha256=checkpoint_hash,
                        worm_ids=worm_ids,
                        worm_indices=worm_indices,
                        wall_seconds=wall_seconds,
                    )
                )

    expected_cells = {
        (method, lag, fold, seed)
        for method in required
        for lag in first.source_lags
        for fold in all_folds
        for seed in all_seeds
    }
    if set(cell_paths) != expected_cells:
        missing = sorted(expected_cells.difference(cell_paths))
        extra = sorted(set(cell_paths).difference(expected_cells))
        raise RuntimeError(f"archive grid is incomplete (missing={missing}, extra={extra})")
    if set(worm_index_to_id) != set(range(first.n_worms)):
        raise RuntimeError("archive union does not cover every manifest worm index")

    # Every method/lag/seed must hold each worm out exactly once across folds.
    expected_worms = tuple(first.schedule_by_worm)
    for method in required:
        for lag in first.source_lags:
            for seed in all_seeds:
                seen: list[str] = []
                for record in records:
                    if (record.method, record.source_lag, record.seed) == (method, lag, seed):
                        seen.extend(record.worm_ids)
                if len(seen) != first.n_worms or set(seen) != set(expected_worms):
                    raise RuntimeError(
                        f"held-out worm coverage failed for {(method, lag, seed)}"
                    )

    if neuron_order is None:
        raise RuntimeError("no archives were loaded")
    if len(resolved_devices) != 1:
        raise RuntimeError(
            f"resolved device changed across canonical archive grid: {sorted(resolved_devices)}"
        )
    return AtlasInputs(
        manifests=manifests,
        records=tuple(sorted(records, key=lambda x: (x.method, x.source_lag, x.fold, x.seed))),
        methods=required,
        folds=all_folds,
        seeds=all_seeds,
        source_lags=first.source_lags,
        horizons=first.horizons,
        phases=first.phases,
        neurons=neuron_order,
        worm_ids=expected_worms,
        fps=first.fps,
        model_id=first.model_id,
        cohort_mode=first.cohort_mode,
        history_frames=first.history_frames,
        source_window_frames=first.source_window_frames,
        schema_version=first.schema_version,
        schema_fingerprint=first.schema_fingerprint,
        resolved_device=next(iter(resolved_devices)),
    )


def orient_response_once(response: np.ndarray) -> np.ndarray:
    """Map ``[..., source, horizon, target]`` to ``[..., horizon,target,source]``."""
    response = np.asarray(response)
    if response.ndim != 6:
        raise ValueError("sampler response must have six axes")
    return response.transpose(0, 1, 2, 4, 5, 3)


def effect_normalization_denominator(
    achieved_gap: np.ndarray, min_gap: float
) -> np.ndarray:
    """Use source-gap magnitude for every signed and unsigned response channel."""
    if min_gap <= 0:
        raise ValueError("minimum normalization gap must be positive")
    return np.maximum(np.abs(np.asarray(achieved_gap)), float(min_gap))


def _contextualize_values(
    values: np.ndarray, codes: np.ndarray
) -> dict[str, np.ndarray]:
    """Return context arrays from ``[worm,phase,event,horizon,target,source]``."""
    if values.ndim != 6 or codes.shape != (values.shape[0], values.shape[2]):
        raise ValueError("value/chemical axes are inconsistent")
    result = {
        phase: values[:, phase_index].mean(axis=1)
        for phase_index, phase in enumerate(PHASES)
    }
    result["state_average"] = values.mean(axis=(1, 2))
    result["onset_minus_baseline"] = (values[:, 1] - values[:, 0]).mean(axis=1)
    chemical_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, chemical in CHEMICAL_CODE_TO_NAME.items():
        event_position = np.argmax(codes == code, axis=1)
        if not np.all(codes[np.arange(len(codes)), event_position] == code):
            raise RuntimeError(f"one or more worms lack chemical code {code}")
        worm = np.arange(len(codes))
        onset = values[worm, 1, event_position]
        baseline = values[worm, 0, event_position]
        chemical_values[chemical] = (onset, baseline)
    for chemical in CHEMICALS:
        onset, _ = chemical_values[chemical]
        result[f"{chemical}_onset"] = onset
    for chemical in CHEMICALS:
        onset, baseline = chemical_values[chemical]
        result[f"{chemical}_onset_minus_baseline"] = onset - baseline
    if tuple(result) != CONTEXTS:
        raise AssertionError("context construction order changed")
    return result


def _contextualize_support(
    valid: np.ndarray, gaps: np.ndarray, codes: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Contextualize source support without pretending events are replicates."""
    if valid.shape != gaps.shape or valid.ndim != 4:
        raise ValueError("support arrays must be [worm,phase,event,source]")
    v: dict[str, np.ndarray] = {
        phase: valid[:, phase_index].mean(axis=1)
        for phase_index, phase in enumerate(PHASES)
    }
    g: dict[str, np.ndarray] = {
        phase: np.abs(gaps[:, phase_index]).mean(axis=1)
        for phase_index, phase in enumerate(PHASES)
    }
    v["state_average"] = valid.mean(axis=(1, 2))
    g["state_average"] = np.abs(gaps).mean(axis=(1, 2))
    v["onset_minus_baseline"] = np.minimum(valid[:, 1], valid[:, 0]).mean(axis=1)
    g["onset_minus_baseline"] = np.minimum(np.abs(gaps[:, 1]), np.abs(gaps[:, 0])).mean(axis=1)
    chemical_support: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for code, chemical in CHEMICAL_CODE_TO_NAME.items():
        event_position = np.argmax(codes == code, axis=1)
        worm = np.arange(len(codes))
        chemical_support[chemical] = (
            valid[worm, 1, event_position],
            valid[worm, 0, event_position],
            np.abs(gaps[worm, 1, event_position]),
            np.abs(gaps[worm, 0, event_position]),
        )
    for chemical in CHEMICALS:
        onset_valid, _, onset_gap, _ = chemical_support[chemical]
        v[f"{chemical}_onset"] = onset_valid
        g[f"{chemical}_onset"] = onset_gap
    for chemical in CHEMICALS:
        onset_valid, baseline_valid, onset_gap, baseline_gap = chemical_support[chemical]
        v[f"{chemical}_onset_minus_baseline"] = np.minimum(
            onset_valid, baseline_valid
        )
        g[f"{chemical}_onset_minus_baseline"] = np.minimum(
            onset_gap, baseline_gap
        )
    return v, g


def _contextualize_source_metric(
    values: np.ndarray,
    codes: np.ndarray,
    *,
    contrast_reducer: str,
) -> dict[str, np.ndarray]:
    """Contextualize ``[worm,phase,event,source]`` particle diagnostics."""
    if values.ndim != 4 or codes.shape != (values.shape[0], values.shape[2]):
        raise ValueError("source-diagnostic axes are inconsistent")
    if contrast_reducer not in {"min", "max"}:
        raise ValueError("contrast reducer must be min or max")
    reducer = np.minimum if contrast_reducer == "min" else np.maximum
    result = {
        phase: values[:, phase_index].mean(axis=1)
        for phase_index, phase in enumerate(PHASES)
    }
    result["state_average"] = values.mean(axis=(1, 2))
    result["onset_minus_baseline"] = reducer(values[:, 1], values[:, 0]).mean(
        axis=1
    )
    selected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, chemical in CHEMICAL_CODE_TO_NAME.items():
        event_position = np.argmax(codes == code, axis=1)
        worm = np.arange(len(codes))
        selected[chemical] = (
            values[worm, 1, event_position],
            values[worm, 0, event_position],
        )
    for chemical in CHEMICALS:
        result[f"{chemical}_onset"] = selected[chemical][0]
    for chemical in CHEMICALS:
        result[f"{chemical}_onset_minus_baseline"] = reducer(*selected[chemical])
    return result


def _is_signed(channel: str, context: str) -> bool:
    # W1 is unsigned within a state; a difference between state-specific W1
    # distances is signed.
    return channel != "endpoint_wasserstein1" or context.endswith("minus_baseline")


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    keep = np.isfinite(p)
    if not keep.any():
        return out
    values = p[keep]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    out[keep] = restored
    return out


def _bootstrap_counts(n_worms: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_worms, size=(replicates, n_worms), endpoint=False)
    counts = np.zeros((replicates, n_worms), dtype=np.float32)
    for index, row in enumerate(draws):
        counts[index] = np.bincount(row, minlength=n_worms)
    return counts / float(n_worms)


def _worm_summary(
    values: np.ndarray,
    *,
    signed: bool,
    bootstrap_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute worm-level statistics for ``[worm,horizon,target,source]``."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 4 or x.shape[0] < 2:
        raise ValueError("worm summary expects at least two worms")
    _all_finite("worm_context_values", x, Path("<derived>"))
    n = x.shape[0]
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    median = np.median(x, axis=0).astype(np.float32)
    bootstrap = np.tensordot(bootstrap_weights, x, axes=(1, 0)).astype(np.float32)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    sd = x.std(axis=0, ddof=1, dtype=np.float64)
    se = sd / math.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = mean / se
    if signed:
        screen_t_p_value = 2.0 * student_t.sf(np.abs(statistic), df=n - 1)
        positive = (x > 0).mean(axis=0)
        negative = (x < 0).mean(axis=0)
        sign_consistency = np.maximum(positive, negative)
    else:
        # A state-specific W1 distance is nonnegative by construction and has no
        # zero-centered signed null here.  Do not manufacture significance or a
        # sign-consistency statistic for it.
        screen_t_p_value = np.full(mean.shape, np.nan, dtype=np.float64)
        sign_consistency = np.full(mean.shape, np.nan, dtype=np.float64)
    zero_variance = se == 0
    screen_t_p_value = np.asarray(screen_t_p_value, dtype=np.float64)
    if signed:
        screen_t_p_value[zero_variance & (mean == 0)] = 1.0
        screen_t_p_value[zero_variance & (mean != 0)] = 0.0
    return {
        "mean": mean,
        "median": median,
        "ci_low": ci_low.astype(np.float32),
        "ci_high": ci_high.astype(np.float32),
        "sign_consistency": sign_consistency.astype(np.float32),
        "screen_t_p_value": screen_t_p_value.astype(np.float32),
    }


def _safe_seed_spearman(seed_matrices: np.ndarray) -> float:
    if seed_matrices.shape[0] < 2:
        return float("nan")
    d = seed_matrices.shape[-1]
    off = ~np.eye(d, dtype=bool)
    values: list[float] = []
    for left in range(seed_matrices.shape[0]):
        for right in range(left + 1, seed_matrices.shape[0]):
            a = seed_matrices[left][off]
            b = seed_matrices[right][off]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            value = float(spearmanr(a, b).statistic)
            if np.isfinite(value):
                values.append(value)
    return float(np.mean(values)) if values else float("nan")


class _NpzStream:
    """Write a NumPy-compatible compressed archive one array at a time."""

    def __init__(self, path: Path):
        self.path = path
        self._zip: zipfile.ZipFile | None = None
        self._keys: set[str] = set()

    def __enter__(self) -> "_NpzStream":
        self._zip = zipfile.ZipFile(
            self.path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        )
        return self

    def write(self, key: str, value: np.ndarray | Sequence | str | int | float) -> None:
        if self._zip is None:
            raise RuntimeError("archive writer is not open")
        key = _slug(key)
        if key in self._keys:
            raise RuntimeError(f"duplicate output archive key {key}")
        self._keys.add(key)
        with self._zip.open(f"{key}.npy", "w", force_zip64=True) as handle:
            np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._zip is not None
        self._zip.close()


class _PredictionTableSink:
    def __init__(self, output: Path):
        self.output = output
        self.format: str
        self.path: Path
        self._writer = None
        self._wrote_header = False
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq  # noqa: F401

            self.format = "parquet"
            self.path = output / "prediction_cells.parquet"
        except ImportError:
            self.format = "csv"
            self.path = output / "prediction_cells.csv"

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        if self.format == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self._writer is None:
                self._writer = pq.ParquetWriter(
                    self.path, table.schema, compression="zstd", use_dictionary=True
                )
            self._writer.write_table(table)
        else:
            frame.to_csv(
                self.path,
                mode="a",
                index=False,
                header=not self._wrote_header,
                quoting=csv.QUOTE_MINIMAL,
            )
            self._wrote_header = True

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def _records_for(
    inputs: AtlasInputs, method: str, lag: int
) -> tuple[ArchiveRecord, ...]:
    return tuple(
        record
        for record in inputs.records
        if record.method == method and record.source_lag == lag
    )


def _binary_stimulus_at_frames(
    schedule: Mapping[str, object], frames: np.ndarray
) -> np.ndarray:
    """Reconstruct the generator's exact half-open binary stimulus covariate."""
    frame_array = np.asarray(frames, dtype=np.int64)
    fps = float(schedule["analysis_fps"])
    time = frame_array.astype(np.float64) / fps
    active = np.zeros(frame_array.shape, dtype=bool)
    for start, stop in np.asarray(
        schedule["event_intervals_seconds"], dtype=np.float64
    ):
        active |= (time >= start) & (time < stop)
    return active


def _expected_schedule_cut(
    schedule: Mapping[str, object],
    *,
    event_index: int,
    phase: str,
    source_window_frames: int,
) -> int:
    """Reconstruct one episode cut from the frozen generator schedule."""
    fps = float(schedule["analysis_fps"])
    start, stop = np.asarray(
        schedule["event_intervals_seconds"], dtype=np.float64
    )[event_index]
    onset = int(round(float(start) * fps))
    offset = int(round(float(stop) * fps))
    candidates = {
        "baseline": onset - int(round(15.0 * fps)),
        "onset": onset + int(source_window_frames) - 1,
        "active": onset + int(round(5.0 * fps)),
        "offset": offset + int(source_window_frames) - 1,
        "recovery": offset + int(round(5.0 * fps)),
    }
    try:
        return candidates[phase]
    except KeyError as error:
        raise RuntimeError(f"unsupported episode phase {phase!r}") from error


def _stimulus_composition_frame(inputs: AtlasInputs) -> pd.DataFrame:
    """Return exact episode-level stimulus composition for every lag/readout.

    Cuts and source bounds come from the immutable response archives.  Stimulus
    values are reconstructed from each worm's frozen half-open schedule using
    the same frame-time rule as the generator input pipeline.  Repeated copies
    across method, seed, and fold archives must agree exactly.
    """
    observed: dict[tuple[str, int, int, int], tuple[int, int, int]] = {}
    for record in inputs.records:
        with np.load(record.path, allow_pickle=False) as data:
            cuts = np.asarray(data["cut_times"], dtype=np.int64)
            bounds = np.asarray(data["source_window_bounds"], dtype=np.int64)
            for local, worm_id in enumerate(record.worm_ids):
                for phase_index in range(len(PHASES)):
                    for event_index in range(3):
                        key = (worm_id, record.source_lag, phase_index, event_index)
                        value = (
                            int(cuts[local, phase_index, event_index]),
                            int(bounds[local, phase_index, event_index, 0]),
                            int(bounds[local, phase_index, event_index, 1]),
                        )
                        previous = observed.get(key)
                        if previous is not None and previous != value:
                            raise RuntimeError(
                                "cut/source-window geometry changed across immutable archives "
                                f"for {key}: {previous} versus {value}"
                            )
                        observed[key] = value
    expected = {
        (worm_id, lag, phase_index, event_index)
        for worm_id in inputs.worm_ids
        for lag in inputs.source_lags
        for phase_index in range(len(PHASES))
        for event_index in range(3)
    }
    if set(observed) != expected:
        missing = sorted(expected.difference(observed))[:5]
        extra = sorted(set(observed).difference(expected))[:5]
        raise RuntimeError(
            f"stimulus-composition episode grid mismatch; missing={missing}, extra={extra}"
        )

    schedule_by_worm = inputs.manifests[0].schedule_by_worm
    rows: list[dict[str, object]] = []
    for worm_id in inputs.worm_ids:
        schedule = schedule_by_worm[worm_id]
        intervals = np.asarray(schedule["event_intervals_seconds"], dtype=np.float64)
        chemical_codes = tuple(int(value) for value in schedule["chemical_code_by_event"])
        chemical_names = tuple(str(value) for value in schedule["chemical_name_by_event"])
        fps = float(schedule["analysis_fps"])
        for lag in inputs.source_lags:
            for phase_index, phase in enumerate(PHASES):
                for event_index in range(3):
                    cut, source_start, source_stop = observed[
                        (worm_id, lag, phase_index, event_index)
                    ]
                    expected_cut = _expected_schedule_cut(
                        schedule,
                        event_index=event_index,
                        phase=phase,
                        source_window_frames=inputs.source_window_frames,
                    )
                    if cut != expected_cut:
                        raise RuntimeError(
                            "archive cut time disagrees with the frozen event schedule: "
                            f"worm={worm_id}, phase={phase}, event={event_index}, "
                            f"observed={cut}, expected={expected_cut}"
                        )
                    source_frames = np.arange(source_start, source_stop, dtype=np.int64)
                    source_active = _binary_stimulus_at_frames(schedule, source_frames)
                    cut_active = bool(
                        _binary_stimulus_at_frames(
                            schedule, np.asarray([cut], dtype=np.int64)
                        )[0]
                    )
                    for horizon in inputs.horizons:
                        forecast_frames = np.arange(
                            cut + 1, cut + horizon + 1, dtype=np.int64
                        )
                        forecast_active = _binary_stimulus_at_frames(
                            schedule, forecast_frames
                        )
                        endpoint_active = bool(forecast_active[-1])
                        rows.append(
                            {
                                "worm_id": worm_id,
                                "phase": phase,
                                "event_index": event_index,
                                "event_position": event_index + 1,
                                "chemical_code": chemical_codes[event_index],
                                "chemical_name": chemical_names[event_index],
                                "source_lag_frames": lag,
                                "source_to_cut_seconds": lag / inputs.fps,
                                "horizon_frames": horizon,
                                "forecast_horizon_seconds": horizon / inputs.fps,
                                "source_to_readout_seconds": (lag + horizon)
                                / inputs.fps,
                                "schedule_analysis_fps": fps,
                                "scheduled_event_start_seconds": float(
                                    intervals[event_index, 0]
                                ),
                                "scheduled_event_stop_seconds_exclusive": float(
                                    intervals[event_index, 1]
                                ),
                                "source_window_start_frame": source_start,
                                "source_window_stop_frame_exclusive": source_stop,
                                "source_window_end_frame_inclusive": source_stop - 1,
                                "source_window_frame_count": int(source_active.size),
                                "source_window_stimulus_frame_count": int(
                                    source_active.sum()
                                ),
                                "source_window_stimulus_fraction": float(
                                    source_active.mean()
                                ),
                                "source_window_contains_stimulus": bool(
                                    source_active.any()
                                ),
                                "source_window_all_stimulus": bool(
                                    source_active.all()
                                ),
                                "source_window_crosses_stimulus_boundary": bool(
                                    source_active.any() and not source_active.all()
                                ),
                                "cut_frame": cut,
                                "cut_time_seconds": cut / fps,
                                "cut_stimulus_indicator": cut_active,
                                "forecast_window_start_frame": cut + 1,
                                "forecast_window_stop_frame_exclusive": cut + horizon + 1,
                                "forecast_window_frame_count": int(
                                    forecast_active.size
                                ),
                                "forecast_window_stimulus_frame_count": int(
                                    forecast_active.sum()
                                ),
                                "forecast_window_stimulus_fraction": float(
                                    forecast_active.mean()
                                ),
                                "forecast_window_contains_stimulus": bool(
                                    forecast_active.any()
                                ),
                                "forecast_window_all_stimulus": bool(
                                    forecast_active.all()
                                ),
                                "forecast_window_crosses_stimulus_boundary": bool(
                                    forecast_active.any() and not forecast_active.all()
                                ),
                                "forecast_endpoint_frame": cut + horizon,
                                "forecast_endpoint_time_seconds": (cut + horizon) / fps,
                                "forecast_endpoint_stimulus_indicator": endpoint_active,
                                "cut_to_endpoint_stimulus_transition": bool(
                                    cut_active != endpoint_active
                                ),
                                "stimulus_scope": "binary_any_stimulus_union_of_three_half_open_intervals",
                                "source_recording": str(schedule["source_recording"]),
                                "resampling_provenance": str(
                                    schedule["resampling_provenance"]
                                ),
                            }
                        )
    frame = pd.DataFrame(rows)
    expected_rows = (
        len(inputs.worm_ids)
        * len(PHASES)
        * 3
        * len(inputs.source_lags)
        * len(inputs.horizons)
    )
    if len(frame) != expected_rows or frame.duplicated(
        ["worm_id", "phase", "event_index", "source_lag_frames", "horizon_frames"]
    ).any():
        raise RuntimeError("stimulus-composition table has invalid row geometry")
    return frame


def _stimulus_composition_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Compact phase/lag/horizon summary for protocol and dashboard metadata."""
    rows: list[dict[str, object]] = []
    group_columns = ["phase", "source_lag_frames", "horizon_frames"]
    for (phase, lag, horizon), group in frame.groupby(group_columns, sort=True):
        row: dict[str, object] = {
            "phase": str(phase),
            "source_lag_frames": int(lag),
            "horizon_frames": int(horizon),
            "n_worm_events": int(len(group)),
        }
        for prefix, column in (
            ("source_window", "source_window_stimulus_fraction"),
            ("forecast_window", "forecast_window_stimulus_fraction"),
            ("cut", "cut_stimulus_indicator"),
            ("forecast_endpoint", "forecast_endpoint_stimulus_indicator"),
        ):
            values = group[column].astype(float)
            row[f"{prefix}_stimulus_fraction_min"] = float(values.min())
            row[f"{prefix}_stimulus_fraction_mean"] = float(values.mean())
            row[f"{prefix}_stimulus_fraction_max"] = float(values.max())
        for output_name, column in (
            (
                "source_window_crosses_stimulus_boundary_any",
                "source_window_crosses_stimulus_boundary",
            ),
            (
                "forecast_window_crosses_stimulus_boundary_any",
                "forecast_window_crosses_stimulus_boundary",
            ),
            (
                "cut_to_endpoint_stimulus_transition_any",
                "cut_to_endpoint_stimulus_transition",
            ),
        ):
            row[output_name] = bool(group[column].astype(bool).any())
        rows.append(row)
    return rows


def _load_seed_contexts(
    inputs: AtlasInputs,
    *,
    method: str,
    lag: int,
    channel: str,
    min_gap: float,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Load one method/lag/channel without retaining the full raw experiment."""
    s = len(inputs.seeds)
    w = len(inputs.worm_ids)
    p = len(PHASES)
    e = 3
    h = len(inputs.horizons)
    d = len(inputs.neurons)
    seed_pos = {seed: index for index, seed in enumerate(inputs.seeds)}
    worm_pos = {worm: index for index, worm in enumerate(inputs.worm_ids)}
    raw = np.full((s, w, p, e, h, d, d), np.nan, dtype=np.float32)
    normalized = np.full_like(raw, np.nan)
    valid = np.full((s, w, p, e, d), np.nan, dtype=np.float32)
    gaps = np.full_like(valid, np.nan)
    codes = np.full((w, e), -1, dtype=np.int8)
    for record in _records_for(inputs, method, lag):
        with np.load(record.path, allow_pickle=False) as data:
            response_source_first = data[f"response_{channel}"].astype(np.float32)
            achieved = data["diagnostic_achieved_gap"].astype(np.float32)
            local_valid = data["diagnostic_valid"].astype(np.float32)
            denominator = effect_normalization_denominator(achieved, min_gap)
            local_normalized = response_source_first / denominator[..., None, None]
            local_raw_oriented = orient_response_once(response_source_first)
            local_norm_oriented = orient_response_once(local_normalized)
            local_codes = data["chemical_code_by_worm_event"].astype(np.int8)
            si = seed_pos[record.seed]
            for local, worm_id in enumerate(record.worm_ids):
                wi = worm_pos[worm_id]
                if np.isfinite(raw[si, wi]).any():
                    raise RuntimeError(f"worm {worm_id} duplicated while loading {method}/{lag}")
                raw[si, wi] = local_raw_oriented[local]
                normalized[si, wi] = local_norm_oriented[local]
                valid[si, wi] = local_valid[local]
                gaps[si, wi] = achieved[local]
                if np.any(codes[wi] >= 0) and not np.array_equal(codes[wi], local_codes[local]):
                    raise RuntimeError(f"chemical order changed across seeds for {worm_id}")
                codes[wi] = local_codes[local]
    _all_finite("loaded raw response", raw, Path(f"<{method}/{lag}/{channel}>"))
    _all_finite("loaded normalized response", normalized, Path(f"<{method}/{lag}/{channel}>"))
    _all_finite("loaded validity", valid, Path(f"<{method}/{lag}/{channel}>"))
    if np.any(codes < 0):
        raise RuntimeError(f"chemical codes not populated for {method}/{lag}")

    raw_context: dict[str, list[np.ndarray]] = {context: [] for context in CONTEXTS}
    norm_context: dict[str, list[np.ndarray]] = {context: [] for context in CONTEXTS}
    valid_context: dict[str, list[np.ndarray]] = {context: [] for context in CONTEXTS}
    gap_context: dict[str, list[np.ndarray]] = {context: [] for context in CONTEXTS}
    for si in range(s):
        local_raw = _contextualize_values(raw[si], codes)
        local_norm = _contextualize_values(normalized[si], codes)
        local_valid, local_gap = _contextualize_support(valid[si], gaps[si], codes)
        for context in CONTEXTS:
            raw_context[context].append(local_raw[context])
            norm_context[context].append(local_norm[context])
            valid_context[context].append(local_valid[context])
            gap_context[context].append(local_gap[context])
    return (
        {key: np.stack(value) for key, value in raw_context.items()},
        {key: np.stack(value) for key, value in norm_context.items()},
        {key: np.stack(value) for key, value in valid_context.items()},
        {key: np.stack(value) for key, value in gap_context.items()},
    )


def _load_context_diagnostics(
    inputs: AtlasInputs,
    *,
    method: str,
    lag: int,
    genealogy_strong_min_ancestor_fraction: float = 0.10,
    genealogy_sensitivity_min_ancestor_fraction: float = 0.20,
) -> dict[str, dict[str, np.ndarray]]:
    """Load source-level particle diagnostics as [seed,worm,source] contexts."""
    metrics: dict[str, tuple[str, str]] = {
        "ess_low": ("diagnostic_ess_low", "min"),
        "ess_high": ("diagnostic_ess_high", "min"),
        "max_weight_low": ("diagnostic_max_weight_low", "max"),
        "max_weight_high": ("diagnostic_max_weight_high", "max"),
        "achieved_gap": ("diagnostic_achieved_gap", "min"),
        "valid": ("diagnostic_valid", "min"),
    }
    if method == "progressive_bridge_smc":
        metrics.update(
            {
                "candidate_ess_low": ("diagnostic_candidate_ess_low", "min"),
                "candidate_ess_high": ("diagnostic_candidate_ess_high", "min"),
                "candidate_max_weight_low": (
                    "diagnostic_candidate_max_weight_low",
                    "max",
                ),
                "candidate_max_weight_high": (
                    "diagnostic_candidate_max_weight_high",
                    "max",
                ),
                "min_step_ess_low": ("diagnostic_min_step_ess_low", "min"),
                "min_step_ess_high": ("diagnostic_min_step_ess_high", "min"),
                "distinct_ancestors_low": (
                    "diagnostic_distinct_ancestors_low",
                    "min",
                ),
                "distinct_ancestors_high": (
                    "diagnostic_distinct_ancestors_high",
                    "min",
                ),
                "forced_tempering_rate_low": (
                    "diagnostic_step_forced_tempering_low",
                    "max",
                ),
                "forced_tempering_rate_high": (
                    "diagnostic_step_forced_tempering_high",
                    "max",
                ),
                "tempering_resamples_mean_low": (
                    "diagnostic_step_tempering_resamples_low",
                    "max",
                ),
                "tempering_resamples_mean_high": (
                    "diagnostic_step_tempering_resamples_high",
                    "max",
                ),
            }
        )
    s, w, p, e, d = (
        len(inputs.seeds),
        len(inputs.worm_ids),
        len(PHASES),
        3,
        len(inputs.neurons),
    )
    seed_pos = {seed: index for index, seed in enumerate(inputs.seeds)}
    worm_pos = {worm: index for index, worm in enumerate(inputs.worm_ids)}
    arrays = {
        metric: np.full((s, w, p, e, d), np.nan, dtype=np.float32)
        for metric in metrics
    }
    codes = np.full((w, e), -1, dtype=np.int8)
    for record in _records_for(inputs, method, lag):
        with np.load(record.path, allow_pickle=False) as data:
            si = seed_pos[record.seed]
            local_codes = data["chemical_code_by_worm_event"].astype(np.int8)
            local_values: dict[str, np.ndarray] = {}
            for metric, (key, _) in metrics.items():
                value = data[key].astype(np.float32)
                if metric == "achieved_gap":
                    value = np.abs(value)
                if value.ndim == 5:
                    if metric.startswith("forced_tempering_rate"):
                        value = value.mean(axis=-1)
                    elif metric.startswith("tempering_resamples_mean"):
                        value = value.mean(axis=-1)
                    else:
                        raise RuntimeError(f"unexpected step-valued support metric {metric}")
                local_values[metric] = value
            for local, worm_id in enumerate(record.worm_ids):
                wi = worm_pos[worm_id]
                for metric in metrics:
                    if np.isfinite(arrays[metric][si, wi]).any():
                        raise RuntimeError(
                            f"diagnostic worm duplication for {method}/{lag}/{worm_id}"
                        )
                    arrays[metric][si, wi] = local_values[metric][local]
                if np.any(codes[wi] >= 0) and not np.array_equal(
                    codes[wi], local_codes[local]
                ):
                    raise RuntimeError(f"chemical order changed for {worm_id}")
                codes[wi] = local_codes[local]
    reducers = {metric: specification[1] for metric, specification in metrics.items()}
    if method == "progressive_bridge_smc":
        particles = {
            record.particles for record in _records_for(inputs, method, lag)
        }
        if len(particles) != 1:
            raise RuntimeError(f"particle count changed within {method}/{lag}")
        particle_count = float(next(iter(particles)))
        minimum_fraction = np.minimum(
            arrays["distinct_ancestors_low"], arrays["distinct_ancestors_high"]
        ) / particle_count
        genealogy_ok_strong = (
            minimum_fraction >= genealogy_strong_min_ancestor_fraction
        ).astype(np.float32)
        genealogy_ok_sensitivity = (
            minimum_fraction >= genealogy_sensitivity_min_ancestor_fraction
        ).astype(np.float32)
        arrays.update(
            {
                "min_distinct_ancestor_fraction": minimum_fraction.astype(
                    np.float32
                ),
                "genealogy_ok_0_10": genealogy_ok_strong,
                "genealogy_ok_0_20": genealogy_ok_sensitivity,
                "genealogy_valid_0_10": (
                    arrays["valid"] * genealogy_ok_strong
                ).astype(np.float32),
                "genealogy_valid_0_20": (
                    arrays["valid"] * genealogy_ok_sensitivity
                ).astype(np.float32),
            }
        )
        reducers.update(
            {
                "min_distinct_ancestor_fraction": "min",
                "genealogy_ok_0_10": "min",
                "genealogy_ok_0_20": "min",
                "genealogy_valid_0_10": "min",
                "genealogy_valid_0_20": "min",
            }
        )
    output: dict[str, dict[str, np.ndarray]] = {context: {} for context in CONTEXTS}
    for metric, value in arrays.items():
        _all_finite(f"loaded diagnostic {metric}", value, Path(f"<{method}/{lag}>"))
        reducer = reducers[metric]
        seed_contexts = [
            _contextualize_source_metric(value[si], codes, contrast_reducer=reducer)
            for si in range(s)
        ]
        for context in CONTEXTS:
            output[context][metric] = np.stack(
                [seed_context[context] for seed_context in seed_contexts]
            )
    return output


def _support_tier(
    *,
    q_value: float,
    ci_low: float,
    ci_high: float,
    valid_fraction: float,
    genealogy_strong_gate_pass: bool,
    sign_consistency: float,
    seed_sign_agreement: float,
    config: AnalysisConfig,
) -> str:
    if valid_fraction < config.min_valid_fraction:
        return "unsupported"
    if not np.isfinite(q_value):
        # No inferential null (notably state-specific unsigned W1) can support
        # only a model-level label, never statistical confirmation.
        return "model_only"
    excludes_zero = ci_low > 0 or ci_high < 0
    if (
        q_value <= config.bh_alpha
        and excludes_zero
        and valid_fraction >= config.strong_valid_fraction
        and genealogy_strong_gate_pass
        and sign_consistency >= config.strong_sign_consistency
        and seed_sign_agreement >= 1.0
    ):
        return "supported_exploratory"
    return "model_only"


def _candidate_sign_flip(
    worm_values: np.ndarray,
    *,
    family: str,
    replicates: int,
    base_seed: int,
) -> np.ndarray:
    """Monte Carlo two-sided sign-flip p-values for prescreened columns.

    ``worm_values`` is [worm, candidate].  A stable family-derived seed gives
    identical signs to every edge in a family and makes reruns bit-reproducible.
    """
    x = np.asarray(worm_values, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("candidate sign flips require [worm,candidate]")
    if x.shape[1] == 0:
        return np.empty(0, dtype=np.float32)
    family_seed = int.from_bytes(
        hashlib.sha256(family.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng((int(base_seed) ^ family_seed) % (2**63 - 1))
    signs = rng.integers(0, 2, size=(replicates, x.shape[0]), dtype=np.int8)
    signs = signs.astype(np.float32) * 2.0 - 1.0
    null = np.abs((signs @ x) / float(x.shape[0]))
    observed = np.abs(x.mean(axis=0))
    return ((1 + np.sum(null >= observed[None], axis=0)) / (replicates + 1)).astype(
        np.float32
    )


def _rows_for_slice(
    *,
    inputs: AtlasInputs,
    method: str,
    channel: str,
    context: str,
    lag_index: int,
    horizon_index: int,
    raw_mean: np.ndarray,
    summary: Mapping[str, np.ndarray],
    worm_values: np.ndarray,
    valid_fraction: np.ndarray,
    genealogy_valid_fraction_0_10: np.ndarray,
    genealogy_valid_fraction_0_20: np.ndarray,
    seed_sign_agreement: np.ndarray,
    seed_spearman: float,
    config: AnalysisConfig,
) -> list[dict[str, object]]:
    d = len(inputs.neurons)
    off_flat = np.flatnonzero(~np.eye(d, dtype=bool).reshape(-1))
    mean = summary["mean"][horizon_index].reshape(-1)
    # Retain a bounded, deterministic set per statistical family.  The complete
    # dense matrices remain in atlas_matrices.npz.
    score = np.abs(mean[off_flat])
    count = min(config.prediction_cells_per_slice, len(off_flat))
    chosen_local = np.argpartition(score, -count)[-count:]
    chosen = off_flat[chosen_local]
    chosen = chosen[np.argsort(-score[chosen_local], kind="stable")]
    rows: list[dict[str, object]] = []
    lag = inputs.source_lags[lag_index]
    horizon = inputs.horizons[horizon_index]
    family = f"{method}|{channel}|{context}|ell={lag}|h={horizon}"
    meta = context_metadata(context)
    for flat in chosen:
        target, source = np.unravel_index(int(flat), (d, d))
        q = float("nan")
        low = float(summary["ci_low"][horizon_index, target, source])
        high = float(summary["ci_high"][horizon_index, target, source])
        consistency = float(summary["sign_consistency"][horizon_index, target, source])
        validity = float(valid_fraction[source])
        genealogy_applicable = method == "progressive_bridge_smc"
        genealogy_valid_0_10 = float(genealogy_valid_fraction_0_10[source])
        genealogy_valid_0_20 = float(genealogy_valid_fraction_0_20[source])
        genealogy_strong_gate_pass = bool(
            not genealogy_applicable
            or genealogy_valid_0_10 >= config.strong_valid_fraction
        )
        seed_agreement = float(seed_sign_agreement[horizon_index, target, source])
        normalized_mean = float(summary["mean"][horizon_index, target, source])
        half_width = max((high - low) / 2.0, 1e-8)
        stability_factor = consistency if np.isfinite(consistency) else 1.0
        seed_factor = seed_agreement if np.isfinite(seed_agreement) else max(
            0.25, (seed_spearman + 1.0) / 2.0 if np.isfinite(seed_spearman) else 0.25
        )
        evidence_score = (
            abs(normalized_mean) / half_width
            * stability_factor
            * validity
            * max(seed_factor, 0.5)
        )
        rows.append(
            {
                "method": method,
                "model_id": inputs.model_id,
                "channel": channel,
                "context": context,
                "chemical": meta["chemical"] or "",
                "conditioning_status": meta["conditioning_status"],
                "source_neuron": inputs.neurons[source],
                "target_neuron": inputs.neurons[target],
                "source_index": int(source),
                "target_index": int(target),
                "source_lag_frames": lag,
                "source_to_cut_seconds": lag / inputs.fps,
                "horizon_frames": horizon,
                "forecast_horizon_seconds": horizon / inputs.fps,
                "source_to_readout_seconds": (lag + horizon) / inputs.fps,
                "mean_raw": float(raw_mean[horizon_index, target, source]),
                "mean_normalized": normalized_mean,
                "median_normalized": float(summary["median"][horizon_index, target, source]),
                "ci_2_5": low,
                "ci_97_5": high,
                "screen_t_p_value": float(
                    summary["screen_t_p_value"][horizon_index, target, source]
                ),
                "sign_flip_p_value": float("nan"),
                "sign_flip_q_value": q,
                "bh_family": family,
                "inference_scope": "post_screen_exploratory_no_full_family_fdr_control",
                "test_sidedness": (
                    "two_sided_candidate_sign_flip"
                    if _is_signed(channel, context)
                    else "not_tested_unsigned_distance"
                ),
                "sign_consistency": consistency,
                "valid_fraction": validity,
                "genealogy_gate_applicable": genealogy_applicable,
                "genealogy_min_distinct_ancestor_fraction_strong": (
                    config.genealogy_strong_min_ancestor_fraction
                ),
                "genealogy_min_distinct_ancestor_fraction_sensitivity": (
                    config.genealogy_sensitivity_min_ancestor_fraction
                ),
                "genealogy_valid_fraction_0_10": genealogy_valid_0_10,
                "genealogy_valid_fraction_0_20": genealogy_valid_0_20,
                "genealogy_strong_gate_pass": genealogy_strong_gate_pass,
                "genealogy_sensitivity_gate_pass": bool(
                    not genealogy_applicable
                    or genealogy_valid_0_20 >= config.strong_valid_fraction
                ),
                "seed_spearman": seed_spearman,
                "seed_sign_agreement": seed_agreement,
                "n_worms": len(inputs.worm_ids),
                "effect_direction": (
                    "positive" if normalized_mean > 0 else "negative" if normalized_mean < 0 else "zero"
                ),
                "support_tier": _support_tier(
                    q_value=q,
                    ci_low=low,
                    ci_high=high,
                    valid_fraction=validity,
                    genealogy_strong_gate_pass=genealogy_strong_gate_pass,
                    sign_consistency=consistency,
                    seed_sign_agreement=seed_agreement,
                    config=config,
                ),
                "evidence_score": float(evidence_score),
                "interpretation_limit": "model_based_lag_association_not_causal_or_physical_delay",
                "__target": target,
                "__source": source,
            }
        )
    # Sign-flip inference is deliberately restricted to prespecified
    # candidate-eligible signed cells: adequate sampler support and a bootstrap
    # interval excluding zero.  BH is applied within this explicit retained
    # family.  Unsigned state-specific W1 cells keep NaN inferential fields.
    eligible_positions = [
        index
        for index, row in enumerate(rows)
        if _is_signed(channel, context)
        and float(row["valid_fraction"]) >= config.min_valid_fraction
        and (float(row["ci_2_5"]) > 0 or float(row["ci_97_5"]) < 0)
    ]
    if eligible_positions:
        candidate_values = np.stack(
            [
                worm_values[
                    :, horizon_index, rows[index]["__target"], rows[index]["__source"]
                ]
                for index in eligible_positions
            ],
            axis=1,
        )
        p_values = _candidate_sign_flip(
            candidate_values,
            family=family,
            replicates=config.sign_flip_replicates,
            base_seed=config.random_seed,
        )
        q_values = _bh_adjust(p_values)
        for position, p_value, q_value in zip(
            eligible_positions, p_values, q_values
        ):
            row = rows[position]
            row["sign_flip_p_value"] = float(p_value)
            row["sign_flip_q_value"] = float(q_value)
            row["support_tier"] = _support_tier(
                q_value=float(q_value),
                ci_low=float(row["ci_2_5"]),
                ci_high=float(row["ci_97_5"]),
                valid_fraction=float(row["valid_fraction"]),
                genealogy_strong_gate_pass=bool(
                    row["genealogy_strong_gate_pass"]
                ),
                sign_consistency=float(row["sign_consistency"]),
                seed_sign_agreement=float(row["seed_sign_agreement"]),
                config=config,
            )
    for row in rows:
        row.pop("__target")
        row.pop("__source")
    return rows


def _orientation_self_test() -> bool:
    probe = np.zeros((1, 1, 1, 3, 1, 3), dtype=np.float32)
    probe[0, 0, 0, 1, 0, 2] = 7.0
    oriented = orient_response_once(probe)
    return bool(oriented[0, 0, 0, 0, 2, 1] == 7 and oriented[0, 0, 0, 0, 1, 2] == 0)


def _annotate_and_rank_confirmation_queue(
    rows: Sequence[dict[str, object]],
    *,
    atlas_path: Path,
    inputs: AtlasInputs,
    primary_method: str,
    cross_sampler_lookup: Mapping[tuple[str, str, int, int], float],
    config: AnalysisConfig,
) -> list[dict[str, object]]:
    """Attach per-cell counterpart agreement and rank the primary shortlist."""
    if len(inputs.methods) < 2:
        return [dict(row) for row in rows[: config.hypothesis_queue_size]]
    counterparts = [method for method in inputs.methods if method != primary_method]
    if len(counterparts) != 1:
        raise RuntimeError("confirmation queue requires exactly one counterpart method")
    counterpart_method = counterparts[0]
    lag_pos = {lag: index for index, lag in enumerate(inputs.source_lags)}
    horizon_pos = {horizon: index for index, horizon in enumerate(inputs.horizons)}
    cache: dict[tuple[str, str, str], np.ndarray] = {}
    valid_cache: dict[tuple[str, str], np.ndarray] = {}
    annotated: list[dict[str, object]] = []
    with np.load(atlas_path, allow_pickle=False) as atlas:
        for original in rows:
            row = dict(original)
            if str(row["method"]) != primary_method:
                continue
            channel = str(row["channel"])
            context = str(row["context"])
            lag = int(row["source_lag_frames"])
            horizon = int(row["horizon_frames"])
            source = int(row["source_index"])
            target = int(row["target_index"])
            counterpart_key = (counterpart_method, channel, context)
            if counterpart_key not in cache:
                cache[counterpart_key] = atlas[
                    f"mean_normalized__{counterpart_method}__{channel}__{context}"
                ]
            counterpart_mean = float(
                cache[counterpart_key][
                    lag_pos[lag], horizon_pos[horizon], target, source
                ]
            )
            valid_key = (counterpart_method, context)
            if valid_key not in valid_cache:
                valid_cache[valid_key] = atlas[
                    f"valid_fraction__{counterpart_method}__{context}"
                ]
            counterpart_validity = float(valid_cache[valid_key][lag_pos[lag], source])
            primary_mean = float(row["mean_normalized"])
            signed = _is_signed(channel, context)
            if signed:
                sign_agreement = float(
                    abs(primary_mean) > 1e-12
                    and abs(counterpart_mean) > 1e-12
                    and np.sign(primary_mean) == np.sign(counterpart_mean)
                )
            else:
                sign_agreement = float("nan")
            magnitude_agreement = float(
                1.0
                - abs(primary_mean - counterpart_mean)
                / (abs(primary_mean) + abs(counterpart_mean) + 1e-12)
            )
            magnitude_agreement = float(np.clip(magnitude_agreement, 0.0, 1.0))
            slice_rho = float(
                cross_sampler_lookup.get((channel, context, lag, horizon), np.nan)
            )
            slice_agreement = (
                float(np.clip((slice_rho + 1.0) / 2.0, 0.0, 1.0))
                if np.isfinite(slice_rho)
                else 0.5
            )
            support_agreement = float(
                min(float(row["valid_fraction"]), counterpart_validity)
            )
            sign_gate = sign_agreement if signed else 1.0
            cross_factor = float(
                sign_gate
                * magnitude_agreement
                * slice_agreement
                * support_agreement
            )
            if row["support_tier"] == "supported_exploratory" and (
                counterpart_validity < config.min_valid_fraction
                or (signed and sign_agreement < 1.0)
                or not bool(row["genealogy_strong_gate_pass"])
            ):
                row["support_tier"] = "model_only"
            promotion_eligible = bool(
                row["support_tier"] == "supported_exploratory"
                and float(row["valid_fraction"]) >= config.min_valid_fraction
                and counterpart_validity >= config.min_valid_fraction
                and bool(row["genealogy_strong_gate_pass"])
                and (not signed or sign_agreement == 1.0)
            )
            base_score = float(row["evidence_score"])
            row.update(
                {
                    "counterpart_method": counterpart_method,
                    "counterpart_mean_normalized": counterpart_mean,
                    "counterpart_valid_fraction": counterpart_validity,
                    "cross_sampler_sign_agreement": sign_agreement,
                    "cross_sampler_magnitude_agreement": magnitude_agreement,
                    "cross_sampler_spearman": slice_rho,
                    "cross_sampler_slice_agreement": slice_agreement,
                    "cross_sampler_support_agreement": support_agreement,
                    "cross_sampler_factor": cross_factor,
                    "base_evidence_score": base_score,
                    "evidence_score": base_score * cross_factor,
                    "promotion_eligible": promotion_eligible,
                }
            )
            annotated.append(row)
    annotated.sort(
        key=lambda row: (
            bool(row["promotion_eligible"]),
            float(row["evidence_score"]),
            float(row["base_evidence_score"]),
            str(row["channel"]),
            str(row["context"]),
            str(row["source_neuron"]),
            str(row["target_neuron"]),
        ),
        reverse=True,
    )
    return annotated[: config.hypothesis_queue_size]


def _dashboard_primary_matrix_rows(
    atlas_path: Path,
    *,
    inputs: AtlasInputs,
    primary_method: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Export one exact dense slice for the bounded network-matrix GUI view."""
    method = (
        "progressive_bridge_smc"
        if "progressive_bridge_smc" in inputs.methods
        else primary_method
    )
    channel = "endpoint_mean"
    context = "onset_minus_baseline"
    lag = 1 if 1 in inputs.source_lags else inputs.source_lags[0]
    horizon = 1 if 1 in inputs.horizons else inputs.horizons[0]
    lag_index = inputs.source_lags.index(lag)
    horizon_index = inputs.horizons.index(horizon)
    with np.load(atlas_path, allow_pickle=False) as atlas:
        matrix = atlas[
            f"mean_normalized__{method}__{channel}__{context}"
        ][lag_index, horizon_index]
    if matrix.shape != (len(inputs.neurons), len(inputs.neurons)):
        raise RuntimeError("dashboard matrix slice has the wrong geometry")
    rows: list[dict[str, object]] = []
    for target_index, target in enumerate(inputs.neurons):
        for source_index, source in enumerate(inputs.neurons):
            rows.append(
                {
                    "source_neuron": source,
                    "target_neuron": target,
                    "source_index": source_index,
                    "target_index": target_index,
                    "mean_normalized": float(matrix[target_index, source_index]),
                    "source_lag_frames": lag,
                    "horizon_frames": horizon,
                    "method": method,
                    "channel": channel,
                    "context": context,
                    "matrix_orientation": ORIENTATION,
                    "row_axis": "target_neuron",
                    "column_axis": "source_neuron",
                    "is_diagonal": source_index == target_index,
                }
            )
    metadata = {
        "method": method,
        "channel": channel,
        "context": context,
        "source_lag_frames": lag,
        "horizon_frames": horizon,
        "matrix_orientation": ORIENTATION,
        "row_axis": "target_neuron",
        "column_axis": "source_neuron",
        "n_rows": len(rows),
        "selection_rule": (
            "frozen default progressive/endpoint_mean/onset_minus_baseline/ell1/h1; "
            "first available exact lag/horizon fallback only when ell1 or h1 is absent"
        ),
    }
    return rows, metadata


def _append_candidate_lag_profiles(
    queue: Sequence[dict[str, object]],
    *,
    atlas_path: Path,
    worm_path: Path,
    inputs: AtlasInputs,
    primary_method: str,
    bootstrap_weights: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Attach descriptive lag localization without changing promotion or rank."""
    lag_frames = [int(value) for value in inputs.source_lags]
    lag_seconds = [float(value / inputs.fps) for value in inputs.source_lags]
    horizon_pos = {horizon: index for index, horizon in enumerate(inputs.horizons)}
    counterpart_methods = [method for method in inputs.methods if method != primary_method]
    counterpart_method = counterpart_methods[0] if len(counterpart_methods) == 1 else None
    atlas_cache: dict[tuple[str, str, str], np.ndarray] = {}
    worm_cache: dict[tuple[str, str], np.ndarray] = {}
    enriched: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    with np.load(atlas_path, allow_pickle=False) as atlas, np.load(
        worm_path, allow_pickle=False
    ) as worms:
        for original in queue:
            row = dict(original)
            channel = str(row["channel"])
            context = str(row["context"])
            horizon = int(row["horizon_frames"])
            h = horizon_pos[horizon]
            target = int(row["target_index"])
            source = int(row["source_index"])
            primary_key = (primary_method, channel, context)
            if primary_key not in atlas_cache:
                atlas_cache[primary_key] = atlas[
                    f"mean_normalized__{primary_method}__{channel}__{context}"
                ]
            primary_profile = atlas_cache[primary_key][:, h, target, source].astype(
                np.float64
            )
            if counterpart_method is not None:
                counterpart_key = (counterpart_method, channel, context)
                if counterpart_key not in atlas_cache:
                    atlas_cache[counterpart_key] = atlas[
                        f"mean_normalized__{counterpart_method}__{channel}__{context}"
                    ]
                counterpart_profile = atlas_cache[counterpart_key][
                    :, h, target, source
                ].astype(np.float64)
            else:
                counterpart_profile = np.full_like(primary_profile, np.nan)
            absolute = np.abs(primary_profile)
            peak_index = int(np.argmax(absolute))
            order = np.sort(absolute)[::-1]
            top = float(order[0])
            second = float(order[1]) if len(order) > 1 else 0.0
            selectivity = float((top - second) / (top + 1e-12))
            if (
                counterpart_method is not None
                and len(primary_profile) >= 2
                and np.std(primary_profile) > 0
                and np.std(counterpart_profile) > 0
            ):
                profile_spearman = float(
                    spearmanr(primary_profile, counterpart_profile).statistic
                )
            else:
                profile_spearman = float("nan")
            signed_profile_spearman = (
                profile_spearman if _is_signed(channel, context) else float("nan")
            )
            worm_key = (channel, context)
            if worm_key not in worm_cache:
                worm_cache[worm_key] = worms[f"normalized__{channel}__{context}"]
            # [lag,worm,horizon,target,source] -> [worm,lag].
            worm_profile = worm_cache[worm_key][:, :, h, target, source].T.astype(
                np.float32
            )
            bootstrap_profiles = bootstrap_weights @ worm_profile
            bootstrap_peak = np.argmax(np.abs(bootstrap_profiles), axis=1)
            bootstrap_rates = np.asarray(
                [(bootstrap_peak == index).mean() for index in range(len(lag_frames))],
                dtype=np.float64,
            )
            row.update(
                {
                    "lag_profile_frames": lag_frames,
                    "lag_profile_source_to_cut_seconds": lag_seconds,
                    "primary_lag_profile": [float(value) for value in primary_profile],
                    "counterpart_lag_profile": [
                        float(value) for value in counterpart_profile
                    ],
                    "peak_abs_effect_lag_frames": lag_frames[peak_index],
                    "peak_abs_effect_lag_seconds": lag_seconds[peak_index],
                    "peak_abs_effect": top,
                    "second_abs_effect": second,
                    "top_vs_second_lag_selectivity": selectivity,
                    "lag_profile_spearman": profile_spearman,
                    "signed_lag_profile_spearman": signed_profile_spearman,
                    "worm_bootstrap_peak_lag_selection_rate": float(
                        bootstrap_rates[peak_index]
                    ),
                    "worm_bootstrap_peak_lag_rates": [
                        float(value) for value in bootstrap_rates
                    ],
                    "lag_interpretation": (
                        "descriptive model lag localization; not a causal or physical delay"
                    ),
                }
            )
            enriched.append(row)
            for index, lag in enumerate(lag_frames):
                long_rows.append(
                    {
                        "queue_rank": int(row.get("queue_rank", len(enriched))),
                        "source_neuron": str(row["source_neuron"]),
                        "target_neuron": str(row["target_neuron"]),
                        "channel": channel,
                        "context": context,
                        "horizon_frames": horizon,
                        "source_lag_frames": lag,
                        "source_to_cut_seconds": lag_seconds[index],
                        "primary_method": primary_method,
                        "counterpart_method": counterpart_method or "",
                        "primary_mean_normalized": float(primary_profile[index]),
                        "counterpart_mean_normalized": float(
                            counterpart_profile[index]
                        ),
                        "is_primary_peak_abs_lag": index == peak_index,
                        "worm_bootstrap_peak_lag_rate": float(
                            bootstrap_rates[index]
                        ),
                        "top_vs_second_lag_selectivity": selectivity,
                        "lag_profile_spearman": profile_spearman,
                        "signed_lag_profile_spearman": signed_profile_spearman,
                        "interpretation": (
                            "descriptive model lag localization; not a physical delay"
                        ),
                    }
                )
    return enriched, long_rows


def build_prediction_atlas(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    config: AnalysisConfig = AnalysisConfig(),
    required_methods: Sequence[str] = METHODS,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate sampler outputs and build the frozen atlas artifact bundle."""
    config.validate()
    inputs = discover_and_validate_inputs(run_dirs, required_methods=required_methods)
    primary_method = config.primary_method
    if primary_method not in inputs.methods:
        primary_method = inputs.methods[0]
    output = Path(output_dir).resolve()
    canonical_names = {
        "worm_matrices.npz",
        "atlas_matrices.npz",
        "prediction_cells.parquet",
        "prediction_cells.csv",
        "support_cells.parquet",
        "support_cells.csv",
        "hypothesis_queue.csv",
        "candidate_lag_profiles.csv",
        "stimulus_composition.csv",
        "models.json",
        "protocol.json",
        "validation.json",
        "dashboard_snapshot.json",
        "manifest.json",
        "checksums.sha256",
    }
    if output.exists() and any((output / name).exists() for name in canonical_names) and not overwrite:
        raise FileExistsError(f"canonical atlas outputs already exist in {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in canonical_names:
            path = output / name
            if path.exists():
                path.unlink()

    stimulus_composition = _stimulus_composition_frame(inputs)
    stimulus_composition.to_csv(output / "stimulus_composition.csv", index=False)
    stimulus_composition_summary = _stimulus_composition_summary(
        stimulus_composition
    )

    bootstrap_weights = _bootstrap_counts(
        len(inputs.worm_ids), config.bootstrap_replicates, config.random_seed
    )
    table_sink = _PredictionTableSink(output)
    support_rows_by_key: dict[tuple[str, str, int, int], dict[str, object]] = {}
    seed_diagnostics: list[dict[str, object]] = []
    candidate_heap: list[tuple[float, int, dict[str, object]]] = []
    heap_counter = 0
    candidate_pool_capacity = max(
        config.hypothesis_queue_size, 5 * config.hypothesis_queue_size
    )
    atlas_path = output / "atlas_matrices.npz"
    worm_path = output / "worm_matrices.npz"
    runtime_by_method_lag: dict[tuple[str, int], dict[str, object]] = {}
    for method in inputs.methods:
        method_manifests = tuple(
            manifest for manifest in inputs.manifests if method in manifest.methods
        )
        if not method_manifests:
            raise RuntimeError(f"no run manifest declares method {method!r}")
        branch_factors = {
            int(manifest.raw["progressive_branch_factor"])
            for manifest in method_manifests
        }
        future_branch_factors = {
            int(manifest.raw["progressive_future_branch_factor"])
            for manifest in method_manifests
        }
        if len(branch_factors) != 1 or len(future_branch_factors) != 1:
            raise RuntimeError(
                f"branch-factor provenance changed across manifests for method {method!r}"
            )
        for lag in inputs.source_lags:
            matching = [
                record
                for record in inputs.records
                if record.method == method and record.source_lag == lag
            ]
            if not matching:
                raise RuntimeError(f"missing runtime records for {method}/lag{lag}")
            progressive = method == "progressive_bridge_smc"
            branch_factor = (
                next(iter(branch_factors))
                if progressive
                else 1
            )
            future_branch_factor = (
                next(iter(future_branch_factors))
                if progressive
                else 1
            )
            runtime_by_method_lag[(method, lag)] = {
                "method": method,
                "source_lag_frames": lag,
                "n_archives": len(matching),
                "n_particles": sorted({record.particles for record in matching}),
                "branch_factor": branch_factor,
                "future_branch_factor": future_branch_factor,
                "branch_factor_semantics": (
                    "progressive proposal branching and post-cut future branching"
                    if progressive
                    else "direct importance has no proposal branching; identity value 1"
                ),
                "mean_archive_wall_seconds": float(
                    np.mean([record.wall_seconds for record in matching])
                ),
                "minimum_archive_wall_seconds": float(
                    np.min([record.wall_seconds for record in matching])
                ),
                "maximum_archive_wall_seconds": float(
                    np.max([record.wall_seconds for record in matching])
                ),
            }

    metadata_arrays = {
        "neurons": np.asarray(inputs.neurons),
        "worm_ids": np.asarray(inputs.worm_ids),
        "methods": np.asarray(inputs.methods),
        "channels": np.asarray(CHANNELS),
        "contexts": np.asarray(CONTEXTS),
        "source_lag_frames": np.asarray(inputs.source_lags, dtype=np.int16),
        "source_lag_seconds": np.asarray(inputs.source_lags, dtype=np.float32) / inputs.fps,
        "horizon_frames": np.asarray(inputs.horizons, dtype=np.int16),
        "horizon_seconds": np.asarray(inputs.horizons, dtype=np.float32) / inputs.fps,
        "orientation": np.asarray(ORIENTATION),
        "primary_method": np.asarray(primary_method),
    }
    with _NpzStream(atlas_path) as atlas_writer, _NpzStream(worm_path) as worm_writer:
        for key, value in metadata_arrays.items():
            atlas_writer.write(key, value)
            worm_writer.write(key, value)
        worm_writer.write(
            "scope",
            np.asarray(
                "normalized worm matrices for primary method only; raw sampler arrays remain immutable inputs"
            ),
        )

        for method in inputs.methods:
            diagnostic_context_by_lag = {
                lag: _load_context_diagnostics(
                    inputs,
                    method=method,
                    lag=lag,
                    genealogy_strong_min_ancestor_fraction=(
                        config.genealogy_strong_min_ancestor_fraction
                    ),
                    genealogy_sensitivity_min_ancestor_fraction=(
                        config.genealogy_sensitivity_min_ancestor_fraction
                    ),
                )
                for lag in inputs.source_lags
            }
            for channel in CHANNELS:
                worm_context_lag: dict[str, list[np.ndarray]] = {
                    context: [] for context in CONTEXTS
                }
                # Atlas matrices are streamed context-by-context after all lags
                # for that channel have been collected.
                statistics_by_context: dict[str, dict[str, list[np.ndarray]]] = {
                    context: {
                        name: []
                        for name in (
                            "mean",
                            "median",
                            "ci_low",
                            "ci_high",
                            "sign_consistency",
                        )
                    }
                    for context in CONTEXTS
                }
                support_by_context: dict[str, list[np.ndarray]] = {
                    context: [] for context in CONTEXTS
                }
                genealogy_10_by_context: dict[str, list[np.ndarray]] = {
                    context: [] for context in CONTEXTS
                }
                genealogy_20_by_context: dict[str, list[np.ndarray]] = {
                    context: [] for context in CONTEXTS
                }
                seed_agreement_by_context: dict[str, list[np.ndarray]] = {
                    context: [] for context in CONTEXTS
                }

                for lag_index, lag in enumerate(inputs.source_lags):
                    diagnostic_contexts = diagnostic_context_by_lag[lag]
                    raw_context, norm_context, valid_context, gap_context = _load_seed_contexts(
                        inputs,
                        method=method,
                        lag=lag,
                        channel=channel,
                        min_gap=config.min_gap,
                    )
                    for context in CONTEXTS:
                        seed_raw = raw_context[context]  # seed,worm,horizon,target,source
                        seed_norm = norm_context[context]
                        worm_raw = seed_raw.mean(axis=0)
                        worm_norm = seed_norm.mean(axis=0)
                        worm_valid = valid_context[context].mean(axis=0)
                        worm_gap = gap_context[context].mean(axis=0)
                        valid_fraction = worm_valid.mean(axis=0)
                        diagnostic_context = diagnostic_contexts[context]
                        diagnostic_worm = {
                            name: values.mean(axis=0)
                            for name, values in diagnostic_context.items()
                        }
                        if not np.allclose(
                            diagnostic_worm["valid"], worm_valid, rtol=0, atol=1e-7
                        ) or not np.allclose(
                            diagnostic_worm["achieved_gap"], worm_gap, rtol=1e-5, atol=1e-6
                        ):
                            raise RuntimeError(
                                f"support diagnostics disagree with response loader for {method}/{lag}/{context}"
                            )
                        if method == "progressive_bridge_smc":
                            genealogy_valid_fraction_0_10 = diagnostic_worm[
                                "genealogy_valid_0_10"
                            ].mean(axis=0)
                            genealogy_valid_fraction_0_20 = diagnostic_worm[
                                "genealogy_valid_0_20"
                            ].mean(axis=0)
                        else:
                            genealogy_valid_fraction_0_10 = np.full(
                                valid_fraction.shape, np.nan, dtype=np.float32
                            )
                            genealogy_valid_fraction_0_20 = np.full(
                                valid_fraction.shape, np.nan, dtype=np.float32
                            )
                        support_by_context[context].append(valid_fraction.astype(np.float32))
                        genealogy_10_by_context[context].append(
                            genealogy_valid_fraction_0_10.astype(np.float32)
                        )
                        genealogy_20_by_context[context].append(
                            genealogy_valid_fraction_0_20.astype(np.float32)
                        )
                        if method == primary_method:
                            worm_context_lag[context].append(worm_norm.astype(np.float32))

                        seed_group = seed_norm.mean(axis=1)  # seed,horizon,target,source
                        signed = _is_signed(channel, context)
                        if signed:
                            seed_sign_agreement = np.maximum(
                                (seed_group > 0).mean(axis=0),
                                (seed_group < 0).mean(axis=0),
                            ).astype(np.float32)
                        else:
                            seed_sign_agreement = np.full(
                                seed_group.shape[1:], np.nan, dtype=np.float32
                            )
                        seed_agreement_by_context[context].append(seed_sign_agreement)
                        summary = _worm_summary(
                            worm_norm,
                            signed=signed,
                            bootstrap_weights=bootstrap_weights,
                        )
                        raw_mean = worm_raw.mean(axis=0)
                        for name in statistics_by_context[context]:
                            statistics_by_context[context][name].append(summary[name])

                        for horizon_index, horizon in enumerate(inputs.horizons):
                            seed_spearman = _safe_seed_spearman(seed_group[:, horizon_index])
                            seed_diagnostics.append(
                                {
                                    "method": method,
                                    "channel": channel,
                                    "context": context,
                                    "source_lag_frames": lag,
                                    "horizon_frames": horizon,
                                    "seed_spearman": seed_spearman,
                                    "n_model_seeds": len(inputs.seeds),
                                }
                            )
                            rows = _rows_for_slice(
                                inputs=inputs,
                                method=method,
                                channel=channel,
                                context=context,
                                lag_index=lag_index,
                                horizon_index=horizon_index,
                                raw_mean=raw_mean,
                                summary=summary,
                                worm_values=worm_norm,
                                valid_fraction=valid_fraction,
                                genealogy_valid_fraction_0_10=(
                                    genealogy_valid_fraction_0_10
                                ),
                                genealogy_valid_fraction_0_20=(
                                    genealogy_valid_fraction_0_20
                                ),
                                seed_sign_agreement=seed_sign_agreement,
                                seed_spearman=seed_spearman,
                                config=config,
                            )
                            table_sink.append(pd.DataFrame(rows))
                            for row in rows:
                                if float(row["valid_fraction"]) < config.min_valid_fraction:
                                    continue
                                if method != primary_method:
                                    continue
                                score = float(row["evidence_score"])
                                item = (score, heap_counter, row)
                                heap_counter += 1
                                if len(candidate_heap) < candidate_pool_capacity:
                                    heapq.heappush(candidate_heap, item)
                                elif score > candidate_heap[0][0]:
                                    heapq.heapreplace(candidate_heap, item)

                        if channel == CHANNELS[0]:
                            particles = next(
                                record.particles
                                for record in inputs.records
                                if record.method == method
                            )
                            min_ess_gate = float(
                                inputs.manifests[0].raw["minimum_effective_sample_size"]
                            )
                            max_weight_gate = float(
                                inputs.manifests[0].raw["maximum_normalized_weight"]
                            )
                            runtime = runtime_by_method_lag[(method, lag)]
                            for source, neuron in enumerate(inputs.neurons):
                                row = {
                                    "method": method,
                                    "context": context,
                                    "chemical": context_metadata(context)["chemical"] or "",
                                    "conditioning_status": context_metadata(context)["conditioning_status"],
                                    "source_neuron": neuron,
                                    "source_index": source,
                                    "source_lag_frames": lag,
                                    "source_to_cut_seconds": lag / inputs.fps,
                                    "valid_fraction": float(valid_fraction[source]),
                                    "n_worms_valid_ge_half": int(np.sum(worm_valid[:, source] >= 0.5)),
                                    "mean_achieved_gap_magnitude": float(
                                        np.abs(worm_gap[:, source]).mean()
                                    ),
                                    "n_worms": len(inputs.worm_ids),
                                    "support_qualified": bool(
                                        valid_fraction[source] >= config.min_valid_fraction
                                    ),
                                    "genealogy_gate_applicable": bool(
                                        method == "progressive_bridge_smc"
                                    ),
                                    "genealogy_min_distinct_ancestor_fraction_strong": (
                                        config.genealogy_strong_min_ancestor_fraction
                                    ),
                                    "genealogy_min_distinct_ancestor_fraction_sensitivity": (
                                        config.genealogy_sensitivity_min_ancestor_fraction
                                    ),
                                    "genealogy_valid_fraction_0_10": float(
                                        genealogy_valid_fraction_0_10[source]
                                    ),
                                    "genealogy_valid_fraction_0_20": float(
                                        genealogy_valid_fraction_0_20[source]
                                    ),
                                    "genealogy_strong_gate_pass": bool(
                                        method != "progressive_bridge_smc"
                                        or genealogy_valid_fraction_0_10[source]
                                        >= config.strong_valid_fraction
                                    ),
                                    "genealogy_sensitivity_gate_pass": bool(
                                        method != "progressive_bridge_smc"
                                        or genealogy_valid_fraction_0_20[source]
                                        >= config.strong_valid_fraction
                                    ),
                                    "n_particles": particles,
                                    "declared_branch_factor": int(
                                        runtime["branch_factor"]
                                    ),
                                    "declared_future_branch_factor": int(
                                        runtime["future_branch_factor"]
                                    ),
                                    "branch_factor_semantics": str(
                                        runtime["branch_factor_semantics"]
                                    ),
                                    "mean_archive_wall_seconds_method_lag": float(
                                        runtime["mean_archive_wall_seconds"]
                                    ),
                                    "n_archives_method_lag": int(runtime["n_archives"]),
                                    "ess_gate": min_ess_gate,
                                    "max_weight_gate": max_weight_gate,
                                    "mean_ess_low": float(
                                        diagnostic_worm["ess_low"][:, source].mean()
                                    ),
                                    "mean_ess_high": float(
                                        diagnostic_worm["ess_high"][:, source].mean()
                                    ),
                                    "minimum_worm_ess_low": float(
                                        diagnostic_worm["ess_low"][:, source].min()
                                    ),
                                    "minimum_worm_ess_high": float(
                                        diagnostic_worm["ess_high"][:, source].min()
                                    ),
                                    "mean_ess_fraction_low": float(
                                        diagnostic_worm["ess_low"][:, source].mean()
                                        / particles
                                    ),
                                    "mean_ess_fraction_high": float(
                                        diagnostic_worm["ess_high"][:, source].mean()
                                        / particles
                                    ),
                                    "mean_max_weight_low": float(
                                        diagnostic_worm["max_weight_low"][:, source].mean()
                                    ),
                                    "mean_max_weight_high": float(
                                        diagnostic_worm["max_weight_high"][:, source].mean()
                                    ),
                                    "maximum_worm_max_weight_low": float(
                                        diagnostic_worm["max_weight_low"][:, source].max()
                                    ),
                                    "maximum_worm_max_weight_high": float(
                                        diagnostic_worm["max_weight_high"][:, source].max()
                                    ),
                                    "ess_low_gate_pass_fraction": float(
                                        (
                                            diagnostic_worm["ess_low"][:, source]
                                            >= min_ess_gate
                                        ).mean()
                                    ),
                                    "ess_high_gate_pass_fraction": float(
                                        (
                                            diagnostic_worm["ess_high"][:, source]
                                            >= min_ess_gate
                                        ).mean()
                                    ),
                                    "max_weight_low_gate_pass_fraction": float(
                                        (
                                            diagnostic_worm["max_weight_low"][:, source]
                                            <= max_weight_gate
                                        ).mean()
                                    ),
                                    "max_weight_high_gate_pass_fraction": float(
                                        (
                                            diagnostic_worm["max_weight_high"][:, source]
                                            <= max_weight_gate
                                        ).mean()
                                    ),
                                }
                                progressive_fields = {
                                    "mean_candidate_ess_low": np.nan,
                                    "mean_candidate_ess_high": np.nan,
                                    "mean_candidate_max_weight_low": np.nan,
                                    "mean_candidate_max_weight_high": np.nan,
                                    "mean_min_step_ess_low": np.nan,
                                    "mean_min_step_ess_high": np.nan,
                                    "mean_distinct_ancestors_low": np.nan,
                                    "mean_distinct_ancestors_high": np.nan,
                                    "mean_distinct_ancestor_fraction_low": np.nan,
                                    "mean_distinct_ancestor_fraction_high": np.nan,
                                    "mean_min_distinct_ancestor_fraction": np.nan,
                                    "minimum_worm_min_distinct_ancestor_fraction": np.nan,
                                    "genealogy_ok_fraction_0_10": np.nan,
                                    "genealogy_ok_fraction_0_20": np.nan,
                                    "forced_tempering_rate_low": np.nan,
                                    "forced_tempering_rate_high": np.nan,
                                    "mean_tempering_resamples_low": np.nan,
                                    "mean_tempering_resamples_high": np.nan,
                                }
                                if method == "progressive_bridge_smc":
                                    progressive_fields = {
                                        "mean_candidate_ess_low": float(
                                            diagnostic_worm["candidate_ess_low"][:, source].mean()
                                        ),
                                        "mean_candidate_ess_high": float(
                                            diagnostic_worm["candidate_ess_high"][:, source].mean()
                                        ),
                                        "mean_candidate_max_weight_low": float(
                                            diagnostic_worm["candidate_max_weight_low"][:, source].mean()
                                        ),
                                        "mean_candidate_max_weight_high": float(
                                            diagnostic_worm["candidate_max_weight_high"][:, source].mean()
                                        ),
                                        "mean_min_step_ess_low": float(
                                            diagnostic_worm["min_step_ess_low"][:, source].mean()
                                        ),
                                        "mean_min_step_ess_high": float(
                                            diagnostic_worm["min_step_ess_high"][:, source].mean()
                                        ),
                                        "mean_distinct_ancestors_low": float(
                                            diagnostic_worm["distinct_ancestors_low"][:, source].mean()
                                        ),
                                        "mean_distinct_ancestors_high": float(
                                            diagnostic_worm["distinct_ancestors_high"][:, source].mean()
                                        ),
                                        "mean_distinct_ancestor_fraction_low": float(
                                            diagnostic_worm["distinct_ancestors_low"][:, source].mean()
                                            / particles
                                        ),
                                        "mean_distinct_ancestor_fraction_high": float(
                                            diagnostic_worm["distinct_ancestors_high"][:, source].mean()
                                            / particles
                                        ),
                                        "mean_min_distinct_ancestor_fraction": float(
                                            diagnostic_worm[
                                                "min_distinct_ancestor_fraction"
                                            ][:, source].mean()
                                        ),
                                        "minimum_worm_min_distinct_ancestor_fraction": float(
                                            diagnostic_worm[
                                                "min_distinct_ancestor_fraction"
                                            ][:, source].min()
                                        ),
                                        "genealogy_ok_fraction_0_10": float(
                                            diagnostic_worm["genealogy_ok_0_10"][
                                                :, source
                                            ].mean()
                                        ),
                                        "genealogy_ok_fraction_0_20": float(
                                            diagnostic_worm["genealogy_ok_0_20"][
                                                :, source
                                            ].mean()
                                        ),
                                        "forced_tempering_rate_low": float(
                                            diagnostic_worm["forced_tempering_rate_low"][:, source].mean()
                                        ),
                                        "forced_tempering_rate_high": float(
                                            diagnostic_worm["forced_tempering_rate_high"][:, source].mean()
                                        ),
                                        "mean_tempering_resamples_low": float(
                                            diagnostic_worm["tempering_resamples_mean_low"][:, source].mean()
                                        ),
                                        "mean_tempering_resamples_high": float(
                                            diagnostic_worm["tempering_resamples_mean_high"][:, source].mean()
                                        ),
                                    }
                                row.update(progressive_fields)
                                support_rows_by_key[(method, context, lag, source)] = row

                for context in CONTEXTS:
                    prefix = f"{method}__{channel}__{context}"
                    for name, arrays in statistics_by_context[context].items():
                        atlas_writer.write(f"{name}_normalized__{prefix}", np.stack(arrays))
                    atlas_writer.write(
                        f"seed_sign_agreement__{prefix}",
                        np.stack(seed_agreement_by_context[context]).astype(np.float16),
                    )
                    # Validity is channel- and horizon-independent; store it only
                    # once per method/context to keep the archive compact.
                    if channel == CHANNELS[0]:
                        atlas_writer.write(
                            f"valid_fraction__{method}__{context}",
                            np.stack(support_by_context[context]),
                        )
                        if method == "progressive_bridge_smc":
                            atlas_writer.write(
                                f"genealogy_valid_fraction_0_10__{method}__{context}",
                                np.stack(genealogy_10_by_context[context]),
                            )
                            atlas_writer.write(
                                f"genealogy_valid_fraction_0_20__{method}__{context}",
                                np.stack(genealogy_20_by_context[context]),
                            )
                    if method == primary_method:
                        worm_writer.write(
                            f"normalized__{channel}__{context}",
                            np.stack(worm_context_lag[context]),
                        )
    table_sink.close()

    cross_sampler_diagnostics: list[dict[str, object]] = []
    cross_sampler_lookup: dict[tuple[str, str, int, int], float] = {}
    if set(METHODS).issubset(inputs.methods):
        with np.load(atlas_path, allow_pickle=False) as atlas:
            off = ~np.eye(len(inputs.neurons), dtype=bool)
            for channel in CHANNELS:
                for context in CONTEXTS:
                    left = atlas[
                        f"mean_normalized__{METHODS[0]}__{channel}__{context}"
                    ]
                    right = atlas[
                        f"mean_normalized__{METHODS[1]}__{channel}__{context}"
                    ]
                    for lag_index, lag in enumerate(inputs.source_lags):
                        for horizon_index, horizon in enumerate(inputs.horizons):
                            a = left[lag_index, horizon_index][off]
                            b = right[lag_index, horizon_index][off]
                            value = (
                                float(spearmanr(a, b).statistic)
                                if np.std(a) > 0 and np.std(b) > 0
                                else float("nan")
                            )
                            cross_sampler_lookup[(channel, context, lag, horizon)] = value
                            cross_sampler_diagnostics.append(
                                {
                                    "channel": channel,
                                    "context": context,
                                    "source_lag_frames": lag,
                                    "horizon_frames": horizon,
                                    "direct_vs_progressive_spearman": value,
                                }
                            )

    support_frame = pd.DataFrame(list(support_rows_by_key.values()))
    diagnostic_support_summary: list[dict[str, object]] = []
    for method, group in support_frame.groupby("method", sort=True):
        diagnostic_support_summary.append(
            {
                "method": method,
                "support_rows": int(len(group)),
                "mean_valid_fraction": float(group["valid_fraction"].mean()),
                "minimum_source_context_valid_fraction": float(
                    group["valid_fraction"].min()
                ),
                "genealogy_gate_applicable": bool(
                    group["genealogy_gate_applicable"].astype(bool).any()
                ),
                "mean_genealogy_valid_fraction_0_10": float(
                    group["genealogy_valid_fraction_0_10"].mean()
                ),
                "mean_genealogy_valid_fraction_0_20": float(
                    group["genealogy_valid_fraction_0_20"].mean()
                ),
                "genealogy_strong_gate_pass_fraction": float(
                    group["genealogy_strong_gate_pass"].astype(float).mean()
                ),
                "genealogy_sensitivity_gate_pass_fraction": float(
                    group["genealogy_sensitivity_gate_pass"].astype(float).mean()
                ),
                "mean_ess_low": float(group["mean_ess_low"].mean()),
                "mean_ess_high": float(group["mean_ess_high"].mean()),
                "minimum_worm_ess_low": float(group["minimum_worm_ess_low"].min()),
                "minimum_worm_ess_high": float(group["minimum_worm_ess_high"].min()),
                "maximum_worm_max_weight_low": float(
                    group["maximum_worm_max_weight_low"].max()
                ),
                "maximum_worm_max_weight_high": float(
                    group["maximum_worm_max_weight_high"].max()
                ),
                "mean_distinct_ancestor_fraction_low": float(
                    group["mean_distinct_ancestor_fraction_low"].mean()
                ),
                "mean_distinct_ancestor_fraction_high": float(
                    group["mean_distinct_ancestor_fraction_high"].mean()
                ),
                "forced_tempering_rate_low": float(
                    group["forced_tempering_rate_low"].mean()
                ),
                "forced_tempering_rate_high": float(
                    group["forced_tempering_rate_high"].mean()
                ),
            }
        )
    diagnostic_context_summary: list[dict[str, object]] = []
    for (method, context), group in support_frame.groupby(
        ["method", "context"], sort=True
    ):
        diagnostic_context_summary.append(
            {
                "method": method,
                "context": context,
                "mean_valid_fraction": float(group["valid_fraction"].mean()),
                "minimum_source_lag_valid_fraction": float(
                    group["valid_fraction"].min()
                ),
                "genealogy_gate_applicable": bool(
                    group["genealogy_gate_applicable"].astype(bool).any()
                ),
                "mean_genealogy_valid_fraction_0_10": float(
                    group["genealogy_valid_fraction_0_10"].mean()
                ),
                "mean_genealogy_valid_fraction_0_20": float(
                    group["genealogy_valid_fraction_0_20"].mean()
                ),
                "genealogy_strong_gate_pass_fraction": float(
                    group["genealogy_strong_gate_pass"].astype(float).mean()
                ),
                "genealogy_sensitivity_gate_pass_fraction": float(
                    group["genealogy_sensitivity_gate_pass"].astype(float).mean()
                ),
                "mean_ess_low": float(group["mean_ess_low"].mean()),
                "mean_ess_high": float(group["mean_ess_high"].mean()),
                "maximum_max_weight_low": float(
                    group["maximum_worm_max_weight_low"].max()
                ),
                "maximum_max_weight_high": float(
                    group["maximum_worm_max_weight_high"].max()
                ),
                "forced_tempering_rate_low": float(
                    group["forced_tempering_rate_low"].mean()
                ),
                "forced_tempering_rate_high": float(
                    group["forced_tempering_rate_high"].mean()
                ),
            }
        )
    weakest_support_columns = [
        "method",
        "context",
        "source_neuron",
        "source_lag_frames",
        "valid_fraction",
        "genealogy_valid_fraction_0_10",
        "genealogy_valid_fraction_0_20",
        "genealogy_strong_gate_pass",
        "genealogy_sensitivity_gate_pass",
        "mean_min_distinct_ancestor_fraction",
        "mean_ess_low",
        "mean_ess_high",
        "maximum_worm_max_weight_low",
        "maximum_worm_max_weight_high",
        "forced_tempering_rate_low",
        "forced_tempering_rate_high",
    ]
    weakest_source_contexts = (
        support_frame.sort_values(
            ["valid_fraction", "method", "context", "source_lag_frames", "source_neuron"],
            kind="stable",
        )
        .head(200)[weakest_support_columns]
        .to_dict(orient="records")
    )
    try:
        support_frame.to_parquet(output / "support_cells.parquet", index=False)
        support_format = "parquet"
        support_path = output / "support_cells.parquet"
    except (ImportError, ModuleNotFoundError):
        support_frame.to_csv(output / "support_cells.csv", index=False)
        support_format = "csv"
        support_path = output / "support_cells.csv"

    candidate_pool = [item[2] for item in sorted(candidate_heap, reverse=True)]
    queue = _annotate_and_rank_confirmation_queue(
        candidate_pool,
        atlas_path=atlas_path,
        inputs=inputs,
        primary_method=primary_method,
        cross_sampler_lookup=cross_sampler_lookup,
        config=config,
    )
    queue, candidate_lag_profiles = _append_candidate_lag_profiles(
        queue,
        atlas_path=atlas_path,
        worm_path=worm_path,
        inputs=inputs,
        primary_method=primary_method,
        bootstrap_weights=bootstrap_weights,
    )
    for rank, row in enumerate(queue, start=1):
        row["queue_rank"] = rank
    queue_frame = pd.DataFrame(queue)
    if not queue_frame.empty:
        queue_frame = queue_frame[
            ["queue_rank"] + [column for column in queue_frame.columns if column != "queue_rank"]
        ]
    queue_frame.to_csv(output / "hypothesis_queue.csv", index=False)
    pd.DataFrame(candidate_lag_profiles).to_csv(
        output / "candidate_lag_profiles.csv", index=False
    )
    matrix_rows, matrix_slice_metadata = _dashboard_primary_matrix_rows(
        atlas_path,
        inputs=inputs,
        primary_method=primary_method,
    )

    checkpoint_rows = sorted(
        {
            (record.fold, record.seed, str(record.checkpoint), record.checkpoint_sha256)
            for record in inputs.records
        }
    )
    models = {
        "generator_model_id": inputs.model_id,
        "generator_stimulus_encoding": GENERATOR_STIMULUS_ENCODING,
        "chemical_conditioning": False,
        "source_run_provenance": {
            "path": str(inputs.manifests[0].source_run),
            "manifest": str(inputs.manifests[0].source_manifest_path),
            "manifest_sha256": inputs.manifests[0].source_manifest_sha256,
            "status": str(inputs.manifests[0].source_manifest["status"]),
            "fold_assignments": str(inputs.manifests[0].fold_assignments),
            "fold_assignments_sha256": inputs.manifests[0].fold_assignments_sha256,
            "chemical_leaderboard": str(inputs.manifests[0].leaderboard_path),
            "chemical_leaderboard_sha256": inputs.manifests[0].leaderboard_sha256,
        },
        "frozen_predictive_scores": dict(
            inputs.manifests[0].selected_predictive_scores
        ),
        "stimulus_encoding_selection": dict(
            inputs.manifests[0].chemical_encoding_gate
        ),
        "base_seed": int(inputs.manifests[0].raw["base_seed"]),
        "episode_seed_definition": EPISODE_SEED_DEFINITION,
        "common_noise_definition": COMMON_NOISE_DEFINITION,
        "requested_device": str(inputs.manifests[0].raw["requested_device"]),
        "resolved_device": inputs.resolved_device,
        "methods": {
            method: {
                "particles": sorted({r.particles for r in inputs.records if r.method == method}),
                "model_seeds": list(inputs.seeds),
                "folds": list(inputs.folds),
                "branch_factor": int(
                    runtime_by_method_lag[(method, inputs.source_lags[0])][
                        "branch_factor"
                    ]
                ),
                "future_branch_factor": int(
                    runtime_by_method_lag[(method, inputs.source_lags[0])][
                        "future_branch_factor"
                    ]
                ),
                "branch_factor_semantics": str(
                    runtime_by_method_lag[(method, inputs.source_lags[0])][
                        "branch_factor_semantics"
                    ]
                ),
                "runtime_by_source_lag": [
                    runtime_by_method_lag[(method, lag)]
                    for lag in inputs.source_lags
                ],
            }
            for method in inputs.methods
        },
        "checkpoints": [
            {"fold": fold, "seed": seed, "path": path, "sha256": digest}
            for fold, seed, path, digest in checkpoint_rows
        ],
    }
    _json_dump(output / "models.json", models)

    protocol = {
        "protocol": "reviewed neural prediction atlas v1",
        "cohort_mode": inputs.cohort_mode,
        "n_worms": len(inputs.worm_ids),
        "n_neurons": len(inputs.neurons),
        "fps": inputs.fps,
        "history_frames": inputs.history_frames,
        "source_window_frames": inputs.source_window_frames,
        "source_lag_frames": list(inputs.source_lags),
        "horizon_frames": list(inputs.horizons),
        "timing_definitions": {
            "source_to_cut_seconds": "source_lag_frames / fps",
            "forecast_horizon_seconds": "horizon_frames / fps",
            "source_to_readout_seconds": "(source_lag_frames + horizon_frames) / fps",
        },
        "stimulus_composition": {
            "artifact": "stimulus_composition.csv",
            "grain": (
                "worm x phase x event-position x source-lag x forecast-horizon"
            ),
            "row_count": len(stimulus_composition),
            "stimulus_scope": (
                "binary any-stimulus union of each worm's three frozen half-open "
                "event intervals"
            ),
            "source_window_frames": (
                "the exact [inclusive_start, exclusive_stop) bounds stored in each "
                "immutable response archive"
            ),
            "cut_frame": "the final observed/repaired history frame",
            "forecast_window_frames": "cut+1 through cut+horizon, inclusive",
            "forecast_endpoint_frame": "cut+horizon",
            "reconstruction_rule": (
                "frame_time=frame/analysis_fps; active when start_seconds <= "
                "frame_time < stop_seconds"
            ),
            "cut_schedule_audit": (
                "every archived phase cut is reconstructed from the frozen event "
                "intervals, analysis_fps, and source_window_frames using the exact "
                "episode_cuts phase definitions"
            ),
            "crossing_warning": (
                "source and forecast windows can cross stimulus boundaries; lag/horizon "
                "panels therefore need not be fixed-stimulus physical-delay comparisons"
            ),
        },
        "orientation": ORIENTATION,
        "orientation_operation": "sampler source and target axes transposed exactly once",
        "effect_definition": "high-source repaired response minus low-source repaired response",
        "sampler_reproducibility": {
            "base_seed": int(inputs.manifests[0].raw["base_seed"]),
            "episode_seed_definition": EPISODE_SEED_DEFINITION,
            "common_noise_definition": COMMON_NOISE_DEFINITION,
            "requested_device": str(inputs.manifests[0].raw["requested_device"]),
            "resolved_device": inputs.resolved_device,
        },
        "compute_diagnostics": {
            "wall_seconds": (
                "elapsed wall time stored in each immutable complete archive; summaries "
                "are arithmetic means across fold/seed archives within method and source lag"
            ),
            "branch_factors": (
                "progressive rows use declared proposal/future branch factors; direct "
                "importance has no branching and is explicitly represented by identity 1"
            ),
        },
        "normalization": {
            "signed_channels": f"event-wise effect / max(abs(achieved_source_gap), {config.min_gap})",
            "endpoint_wasserstein1": f"event-wise W1 / max(abs(achieved_source_gap), {config.min_gap})",
            "negative_gap_handling": (
                "negative achieved gaps can occur for invalid repairs; their magnitude is "
                "used only to keep normalized values finite, while diagnostic_valid keeps "
                "those source/context cells unsupported"
            ),
            "raw_preservation": "mean_raw retained for reviewed prediction cells; immutable sampler archives remain the complete raw source",
        },
        "channels": list(CHANNELS),
        "contexts": [context_metadata(context) for context in CONTEXTS],
        "chemical_context_warning": (
            "Chemical panels are event-stratified observations under a binary-any-stimulus "
            "generator. They are exploratory and are not chemically conditioned counterfactuals."
        ),
        "inference": {
            "independent_unit": "worm",
            "model_seed_handling": "preserve seed diagnostics, then average seeds within each worm",
            "interval": "deterministic percentile worm bootstrap",
            "bootstrap_replicates": config.bootstrap_replicates,
            "bootstrap_seed": config.random_seed,
            "full_screen_test": "worm-level t p-values are descriptive screens only and cannot assign an evidence tier",
            "signed_candidate_tests": (
                "deterministic Monte Carlo two-sided worm sign flips for retained, "
                "support-qualified cells whose worm-bootstrap interval excludes zero"
            ),
            "sign_flip_replicates": config.sign_flip_replicates,
            "unsigned_W1_state_tests": (
                "none: state-specific W1 is nonnegative and no explicit signed zero null was "
                "defined; it is classified only with source support, seed stability, and "
                "direct-versus-progressive cross-sampler concordance"
            ),
            "bh_family": (
                "separate method x channel x context x source-lag x horizon family "
                "within the explicitly retained candidate-eligible set"
            ),
            "post_screen_warning": (
                "Top-|effect| retention and interval/support screening happen before the "
                "candidate sign-flip tests. BH-adjusted values therefore describe only "
                "the retained exploratory candidate set and do not provide full-family "
                "FDR control over all directed atlas edges. They must not be presented as "
                "confirmatory p/q-values."
            ),
            "bh_alpha": config.bh_alpha,
        },
        "support_thresholds": {
            "minimum": config.min_valid_fraction,
            "strong": config.strong_valid_fraction,
            "strong_sign_consistency": config.strong_sign_consistency,
            "genealogy": {
                "applicable_method": "progressive_bridge_smc",
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
                "sensitivity_use": "reporting only; never changes tier, score, or promotion",
                "raw_diagnostic_valid_preserved": True,
                "ranking_score_adjusted": False,
            },
        },
        "evidence_tiers": {
            "confirmed": "reserved for independent experimental confirmation; never assigned here",
            "supported_exploratory": (
                "worm-level/BH/stability/raw-support gates pass and, for progressive "
                "SMC, the separate f=0.10 genealogy strong gate passes"
            ),
            "model_only": "model estimate with adequate minimum support but incomplete evidence gates",
            "unsupported": "fails minimum support or inferential definition",
        },
        "ranking": "no connectome or external reference data used",
        "confirmation_shortlist": {
            "method": primary_method,
            "counterpart_required_for_promotion": True,
            "dual_support_gate": (
                f"primary and counterpart valid_fraction must both be >= "
                f"{config.min_valid_fraction} for supported_exploratory promotion"
            ),
            "progressive_genealogy_gate": (
                "the primary progressive row must pass genealogy_strong_gate_pass; "
                "f=0.20 remains a sensitivity field only"
            ),
            "signed_effect_gate": (
                "direct and progressive per-cell group means must have the same nonzero "
                "sign; disagreement downgrades supported_exploratory to model_only"
            ),
            "candidate_pool": (
                f"top {candidate_pool_capacity} support-qualified primary-method rows by "
                "base evidence score before cross-sampler reranking"
            ),
            "cross_sampler_factor_formula": (
                "sign_gate * magnitude_agreement * slice_agreement * support_agreement; "
                "sign_gate=1 for unsigned W1 and otherwise per-cell sign agreement; "
                "magnitude_agreement=1-|primary-counterpart|/(|primary|+|counterpart|+1e-12); "
                "slice_agreement=clip((direct-vs-progressive Spearman+1)/2,0,1), or 0.5 "
                "when undefined; support_agreement=min(primary_valid_fraction, "
                "counterpart_valid_fraction)"
            ),
            "reranked_score": "base_evidence_score * cross_sampler_factor",
            "method_preference": (
                f"hypothesis_queue contains primary-method ({primary_method}) rows only; both methods "
                "remain in prediction_cells and dense atlas matrices"
            ),
        },
        "lag_localization": {
            "scope": (
                "descriptive summaries for bounded hypothesis-queue rows at each row's "
                "selected forecast horizon"
            ),
            "profile": "exact primary and counterpart normalized effects across every source lag",
            "peak_lag": "argmax absolute primary-method group-mean effect",
            "top_vs_second_selectivity": (
                "(largest absolute lag effect - second largest) / "
                "(largest absolute lag effect + 1e-12)"
            ),
            "profile_agreement": (
                "Spearman correlation of primary and counterpart lag profiles for every "
                "channel; signed_lag_profile_spearman is additionally populated only when "
                "the channel/context estimand is signed"
            ),
            "bootstrap_peak_rate": (
                "fraction of the frozen worm-bootstrap replicates selecting the group-mean "
                "peak-|effect| lag"
            ),
            "promotion_use": (
                "none: lag localization is appended after ranking and does not change "
                "support tier, promotion eligibility, or queue order"
            ),
            "claim_boundary": "descriptive model lag localization, never a physical delay",
        },
        "interpretation_limit": "lag association from a fitted conditional generator; not a causal or physical-delay estimate",
        "storage": {
            "worm_matrices": f"normalized primary-method ({primary_method}) matrices for all channels and contexts",
            "atlas_matrices": "dense reviewed group summaries for all methods/channels/contexts",
            "prediction_cells": f"top {config.prediction_cells_per_slice} off-diagonal cells per BH family; dense arrays remain in atlas_matrices.npz",
        },
        "hypothesis_queue_eligibility": (
            f"only cells with source valid_fraction >= {config.min_valid_fraction}; "
            "unsupported cells remain visible in the reviewed prediction-cell table but "
            "are excluded from the hypothesis queue. Genealogy does not remove raw/model-only "
            "rows from the candidate pool, but f=0.10 blocks strong promotion."
        ),
    }
    _json_dump(output / "protocol.json", protocol)

    validation = {
        "status": "passed",
        "problems": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_run_dirs": [str(Path(x).resolve()) for x in run_dirs],
        "input_manifests": [
            {"path": str(info.path), "sha256": sha256(info.path)} for info in inputs.manifests
        ],
        "archive_count": len(inputs.records),
        "archive_checks_completed": {
            "status_and_required_fields": True,
            "v2_seed_and_common_noise_provenance": True,
            "requested_and_resolved_device_consistency": True,
            "source_run_manifest_path_hash_and_complete_status": True,
            "source_run_cohort_schema_and_model_grid": True,
            "fold_assignment_path_hash_and_heldout_mapping": True,
            "chemical_leaderboard_path_hash_and_selected_scores": True,
            "archive_wall_seconds_finite_positive": True,
            "all_response_shapes": True,
            "all_response_and_diagnostic_values_finite": True,
            "event_probability_effect_within_minus_one_plus_one": True,
            "endpoint_wasserstein1_nonnegative": True,
            "target_gap_equals_high_minus_low": True,
            "achieved_gap_equals_high_minus_low": True,
            "direct_ess_and_max_weight_ranges": True,
            "progressive_candidate_and_equivalent_ess_identities": True,
            "progressive_step_ess_and_max_weight_ranges": True,
            "progressive_tempering_flags_resamples_and_beta_schedule": True,
            "validity_flags_recomputed_from_manifest_gates": True,
            "complete_method_lag_fold_seed_grid": True,
            "heldout_worm_coverage_once_per_seed": True,
            "checkpoint_paths_and_hashes": True,
            "stimulus_schema_version_and_fingerprint": True,
            "chemical_permutation_and_code_name_mapping": True,
            "chemical_schedule_matches_manifest": True,
            "source_window_bounds_and_lag": True,
            "cut_times_reconstructed_from_frozen_schedule": True,
            "schedule_derived_stimulus_composition_reconstructed": True,
            "horizon_and_source_to_readout_timing": True,
            "neuron_order_constant": True,
            "orientation_self_test": _orientation_self_test(),
            "dashboard_matrix_round_trip_from_dense_atlas": True,
        },
        "stimulus_schema_version": inputs.schema_version,
        "stimulus_schema_fingerprint": inputs.schema_fingerprint,
        "n_worms": len(inputs.worm_ids),
        "n_neurons": len(inputs.neurons),
        "methods": list(inputs.methods),
        "folds": list(inputs.folds),
        "seeds": list(inputs.seeds),
        "source_lags": list(inputs.source_lags),
        "horizons": list(inputs.horizons),
        "base_seed": int(inputs.manifests[0].raw["base_seed"]),
        "episode_seed_definition": EPISODE_SEED_DEFINITION,
        "common_noise_definition": COMMON_NOISE_DEFINITION,
        "requested_device": str(inputs.manifests[0].raw["requested_device"]),
        "resolved_device": inputs.resolved_device,
        "source_run_provenance": {
            "path": str(inputs.manifests[0].source_run),
            "manifest": str(inputs.manifests[0].source_manifest_path),
            "manifest_sha256": inputs.manifests[0].source_manifest_sha256,
            "status": str(inputs.manifests[0].source_manifest["status"]),
            "fold_assignments": str(inputs.manifests[0].fold_assignments),
            "fold_assignments_sha256": inputs.manifests[0].fold_assignments_sha256,
            "chemical_leaderboard": str(inputs.manifests[0].leaderboard_path),
            "chemical_leaderboard_sha256": inputs.manifests[0].leaderboard_sha256,
            "selected_predictive_scores": dict(
                inputs.manifests[0].selected_predictive_scores
            ),
            "chemical_encoding_gate": dict(
                inputs.manifests[0].chemical_encoding_gate
            ),
        },
        "runtime_compute_diagnostics": [
            runtime_by_method_lag[(method, lag)]
            for method in inputs.methods
            for lag in inputs.source_lags
        ],
        "seed_diagnostics": seed_diagnostics,
        "cross_sampler_diagnostics": cross_sampler_diagnostics,
        "particle_support_diagnostics": diagnostic_support_summary,
        "particle_support_context_diagnostics": diagnostic_context_summary,
        "weakest_source_context_diagnostics": weakest_source_contexts,
        "hypothesis_queue_audit": {
            "rows": len(queue),
            "candidate_pool_rows": len(candidate_pool),
            "primary_method": primary_method,
            "non_primary_rows": int(
                sum(str(row["method"]) != primary_method for row in queue)
            ),
            "minimum_valid_fraction": config.min_valid_fraction,
            "unsupported_rows": int(
                sum(
                    float(row["valid_fraction"]) < config.min_valid_fraction
                    for row in queue
                )
            ),
            "promoted_with_counterpart_below_support_gate": int(
                sum(
                    bool(row["promotion_eligible"])
                    and float(row["counterpart_valid_fraction"])
                    < config.min_valid_fraction
                    for row in queue
                )
            ),
            "promoted_signed_rows_with_sign_disagreement": int(
                sum(
                    bool(row["promotion_eligible"])
                    and _is_signed(str(row["channel"]), str(row["context"]))
                    and float(row["cross_sampler_sign_agreement"]) < 1.0
                    for row in queue
                )
            ),
            "promoted_with_genealogy_gate_failure": int(
                sum(
                    bool(row["promotion_eligible"])
                    and not bool(row["genealogy_strong_gate_pass"])
                    for row in queue
                )
            ),
        },
        "dashboard_matrix_slice": matrix_slice_metadata,
        "candidate_lag_profile_audit": {
            "queue_rows": len(queue),
            "profile_rows": len(candidate_lag_profiles),
            "lags_per_candidate": len(inputs.source_lags),
            "expected_profile_rows": len(queue) * len(inputs.source_lags),
            "used_for_promotion": False,
        },
    }
    if not validation["archive_checks_completed"]["orientation_self_test"]:
        raise AssertionError("orientation self-test failed")
    _json_dump(output / "validation.json", validation)

    dashboard = {
        "title": "Neural lag-effect prediction atlas",
        "status": "reviewed_model_predictions",
        "primary_method": primary_method,
        "cohort": {
            "mode": inputs.cohort_mode,
            "worms": len(inputs.worm_ids),
            "neurons": len(inputs.neurons),
            "fps": inputs.fps,
        },
        "methods": list(inputs.methods),
        "channels": list(CHANNELS),
        "contexts": [context_metadata(context) for context in CONTEXTS],
        "source_lags": [
            {"frames": lag, "seconds_to_cut": lag / inputs.fps} for lag in inputs.source_lags
        ],
        "horizons": [
            {"frames": horizon, "seconds": horizon / inputs.fps} for horizon in inputs.horizons
        ],
        "top_candidates": queue[: config.dashboard_candidates],
        "matrix_rows": matrix_rows,
        "matrix_slice": matrix_slice_metadata,
        "candidate_lag_profiles": candidate_lag_profiles,
        "stimulus_composition_summary": stimulus_composition_summary,
        "support_summary": {
            "mean_valid_fraction": float(support_frame["valid_fraction"].mean()),
            "sources_ge_minimum": int(support_frame["support_qualified"].sum()),
            "support_rows": len(support_frame),
            "progressive_genealogy_rows_passing_f_0_10_strong_gate": int(
                support_frame.loc[
                    support_frame["genealogy_gate_applicable"].astype(bool),
                    "genealogy_strong_gate_pass",
                ].sum()
            ),
            "progressive_genealogy_rows_passing_f_0_20_sensitivity_gate": int(
                support_frame.loc[
                    support_frame["genealogy_gate_applicable"].astype(bool),
                    "genealogy_sensitivity_gate_pass",
                ].sum()
            ),
        },
        "warnings": [
            protocol["chemical_context_warning"],
            protocol["interpretation_limit"],
            protocol["stimulus_composition"]["crossing_warning"],
            "No connectome data were used to rank predictions.",
        ],
    }
    _json_dump(output / "dashboard_snapshot.json", dashboard)

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "reviewed neural prediction atlas v1",
        "input_run_dirs": [str(Path(x).resolve()) for x in run_dirs],
        "primary_method": primary_method,
        "source_run_manifest_sha256": inputs.manifests[0].source_manifest_sha256,
        "fold_assignments_sha256": inputs.manifests[0].fold_assignments_sha256,
        "chemical_leaderboard_sha256": inputs.manifests[0].leaderboard_sha256,
        "artifacts": {
            "worm_matrices": "worm_matrices.npz",
            "atlas_matrices": "atlas_matrices.npz",
            "prediction_cells": table_sink.path.name,
            "prediction_cells_format": table_sink.format,
            "support_cells": support_path.name,
            "support_cells_format": support_format,
            "hypothesis_queue": "hypothesis_queue.csv",
            "candidate_lag_profiles": "candidate_lag_profiles.csv",
            "stimulus_composition": "stimulus_composition.csv",
            "models": "models.json",
            "protocol": "protocol.json",
            "validation": "validation.json",
            "dashboard_snapshot": "dashboard_snapshot.json",
        },
        "dense_matrix_orientation": ORIENTATION,
        "complete_raw_data_location": "immutable input sampler archives",
    }
    _json_dump(output / "manifest.json", manifest)

    checksum_paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths)
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=20_260_829)
    parser.add_argument("--prediction-cells-per-slice", type=int, default=100)
    parser.add_argument("--hypothesis-queue-size", type=int, default=1000)
    parser.add_argument("--sign-flip-replicates", type=int, default=2_048)
    parser.add_argument("--primary-method", choices=METHODS, default="progressive_bridge_smc")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = AnalysisConfig(
        bootstrap_replicates=args.bootstrap_replicates,
        random_seed=args.random_seed,
        prediction_cells_per_slice=args.prediction_cells_per_slice,
        hypothesis_queue_size=args.hypothesis_queue_size,
        sign_flip_replicates=args.sign_flip_replicates,
        primary_method=args.primary_method,
    )
    manifest = build_prediction_atlas(
        args.run_dir,
        args.output_dir,
        config=config,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
