"""Frozen-model sampling controls for selected NeuroPAL atlas cells.

The runner keeps the checkpoint, held-out worm, observed stimulus schedule,
source clamp, and candidate cell fixed while measuring finite-particle and
quiet-time reference behavior.  It is deliberately separate from the
production atlas and targeted-confirmation archive schemas.

Particles quantify Monte Carlo behavior; worms remain the biological
replication unit.  Quiet pseudo-boundaries are temporal-specificity controls,
not experimental interventions or anatomical nulls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from compatibility_neural_benchmark.core import (
    GeneratorAdapter,
    RepairedResponseConfig,
    causal_fill,
    episode_cuts,
    standardize_for_checkpoint,
    stimulus_for_trace,
)
from compatibility_neural_benchmark.prediction_atlas_runner import (
    GENERATOR_ENCODING,
    PHASES,
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _validate_checkpoint,
    canonical_fingerprint,
    checkpoint_path,
    draw_seed,
    load_folds,
    sha256,
    source_manifest_stimulus_fingerprint,
    training_context,
)
from compatibility_neural_benchmark.progressive_smc import (
    progressive_smc_repaired_responses,
)
from conditional_neural_benchmark.data import StimulusSchedule, load_cohort


ARCHIVE_SCHEMA_VERSION = "prediction_atlas_sampling_null_v2"
MANIFEST_SCHEMA_VERSION = "prediction_atlas_sampling_null_manifest_v2"
METHOD = "progressive_bridge_smc_sampling_nulls"
ARM_NAMES = ("low", "high")
DIAGNOSTIC_FIELDS = (
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
    "min_step_ess_low",
    "min_step_ess_high",
    "distinct_ancestors_low",
    "distinct_ancestors_high",
    "valid",
)
SUPPORTED_CONTEXT_SUFFIXES = ("_onset", "_onset_minus_baseline")
SAMPLING_SEED_DEFINITION = (
    "observed replica 0 uses the canonical episode seed; additional observed "
    "and all midpoint replicas use a SHA-256 namespaced sub-seed of that seed; "
    "every quiet pseudo-boundary reuses observed replica-0 seed for the matched "
    "phase/event, giving common random numbers across real and quiet histories"
)
MIDPOINT_DEFINITION = "arithmetic midpoint of fold-training phase q25 and q75"
PSEUDO_BOUNDARY_DEFINITION = (
    "equally spaced feasible integer boundaries inside the all-zero-stimulus "
    "interval immediately preceding the matched event; every model-history, "
    "repair, and generated-future frame is required to remain in that interval"
)
CLAIM_BOUNDARY = (
    "sampling distinguishability and quiet-time temporal specificity under a "
    "frozen learned observed-data law; not a causal effect, anatomical edge, "
    "receptor action, biological null, or physical delay"
)


def recompute_diagnostic_valid(
    data: Mapping[str, np.ndarray], prefix: str, *, min_ess: float
) -> np.ndarray:
    """Reproduce a saved SMC validity flag from its diagnostic components."""
    ess_low = np.asarray(data[f"{prefix}_diagnostic_ess_low"], dtype=np.float64)
    ess_high = np.asarray(data[f"{prefix}_diagnostic_ess_high"], dtype=np.float64)
    max_low = np.asarray(
        data[f"{prefix}_diagnostic_max_weight_low"], dtype=np.float64
    )
    max_high = np.asarray(
        data[f"{prefix}_diagnostic_max_weight_high"], dtype=np.float64
    )
    achieved_gap = np.asarray(
        data[f"{prefix}_diagnostic_achieved_gap"], dtype=np.float64
    )
    target_gap = np.asarray(data[f"{prefix}_diagnostic_target_gap"], dtype=np.float64)
    achieved_low = np.asarray(
        data[f"{prefix}_diagnostic_achieved_low"], dtype=np.float64
    )
    achieved_high = np.asarray(
        data[f"{prefix}_diagnostic_achieved_high"], dtype=np.float64
    )
    target_low = np.asarray(data[f"{prefix}_diagnostic_target_low"], dtype=np.float64)
    target_high = np.asarray(
        data[f"{prefix}_diagnostic_target_high"], dtype=np.float64
    )
    shapes = {
        tuple(array.shape)
        for array in (
            ess_low,
            ess_high,
            max_low,
            max_high,
            achieved_low,
            achieved_high,
            achieved_gap,
            target_low,
            target_high,
            target_gap,
        )
    }
    if len(shapes) != 1:
        raise RuntimeError(f"{prefix} diagnostic component shapes disagree")
    if not all(
        np.isfinite(array).all()
        for array in (
            ess_low,
            ess_high,
            max_low,
            max_high,
            achieved_low,
            achieved_high,
            achieved_gap,
            target_low,
            target_high,
            target_gap,
        )
    ):
        raise RuntimeError(f"{prefix} diagnostic components are nonfinite")
    if (
        np.any(ess_low <= 0)
        or np.any(ess_high <= 0)
        or np.any(max_low <= 0)
        or np.any(max_high <= 0)
    ):
        raise RuntimeError(f"{prefix} ESS or weight diagnostics are out of range")
    if not np.allclose(
        target_gap, target_high - target_low, rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError(f"{prefix} target-gap diagnostic is inconsistent")
    if not np.allclose(
        achieved_gap, achieved_high - achieved_low, rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError(f"{prefix} achieved-gap diagnostic is inconsistent")
    return (
        (ess_low >= min_ess)
        & (ess_high >= min_ess)
        & (max_low <= float(RepairedResponseConfig.max_normalized_weight))
        & (max_high <= float(RepairedResponseConfig.max_normalized_weight))
        & (
            achieved_gap
            >= float(RepairedResponseConfig.min_achieved_fraction) * target_gap
        )
    )


@dataclass(frozen=True)
class SelectedCell:
    candidate_id: str
    queue_rank: int
    source: int
    target: int
    source_name: str
    target_name: str
    lag: int
    horizon: int
    context: str
    phases: tuple[str, ...]
    channel: str
    selection_origin: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "queue_rank": self.queue_rank,
            "source_index": self.source,
            "target_index": self.target,
            "source_neuron": self.source_name,
            "target_neuron": self.target_name,
            "source_lag_frames": self.lag,
            "horizon_frames": self.horizon,
            "context": self.context,
            "required_phases": list(self.phases),
            "channel": self.channel,
            "selection_origin": self.selection_origin,
        }


@dataclass(frozen=True)
class SamplingGroup:
    lag: int
    horizon: int
    phases: tuple[str, ...]
    sources: tuple[int, ...]
    targets: tuple[int, ...]
    cells: tuple[SelectedCell, ...]
    selection_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_lag_frames": self.lag,
            "horizon_frames": self.horizon,
            "phases": list(self.phases),
            "source_indices": list(self.sources),
            "target_indices": list(self.targets),
            "cells": [cell.to_dict() for cell in self.cells],
            "selection_fingerprint": self.selection_fingerprint,
        }


@dataclass(frozen=True)
class QuietPseudoPlan:
    quiet_intervals: np.ndarray
    boundary_times: np.ndarray
    cut_times: np.ndarray
    source_window_bounds: np.ndarray
    quiet_verified: np.ndarray


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _truthy(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "selected", "approved", "promoted", "queued"}
    ).to_numpy()


def context_phases(context: str) -> tuple[str, ...]:
    context = str(context).strip().lower()
    if context in {"baseline", "onset"}:
        return (context,)
    if context == "onset_minus_baseline" or context.endswith(
        "_onset_minus_baseline"
    ):
        return ("baseline", "onset")
    if context.endswith("_onset"):
        return ("onset",)
    raise RuntimeError(
        f"sampling-null pseudo-boundaries support baseline/onset contexts, got {context!r}"
    )


def load_selected_cells(
    path: Path, neurons: Sequence[str]
) -> tuple[tuple[SelectedCell, ...], tuple[SamplingGroup, ...]]:
    """Load exact selected cells without unioning unrelated horizons or phases."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError("sampling-null hypothesis queue is empty")
    selection_column = next(
        (
            name
            for name in ("run_sampling_nulls", "run_confirmation", "selected")
            if name in frame.columns
        ),
        None,
    )
    if selection_column is not None:
        frame = frame.loc[_truthy(frame[selection_column])].copy()
    required = {
        "source_neuron",
        "target_neuron",
        "source_lag_frames",
        "horizon_frames",
        "context",
        "selection_origin",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"sampling-null queue lacks columns {missing}")
    if frame.empty:
        raise RuntimeError("sampling-null queue contains no selected rows")
    neuron_to_index = {str(name): index for index, name in enumerate(neurons)}
    queue_hash = sha256(path)
    cells: list[SelectedCell] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        source_name, target_name = (
            str(row["source_neuron"]).strip(),
            str(row["target_neuron"]).strip(),
        )
        if source_name not in neuron_to_index or target_name not in neuron_to_index:
            raise RuntimeError("sampling-null queue contains an unknown neuron")
        source, target = neuron_to_index[source_name], neuron_to_index[target_name]
        if source == target:
            raise RuntimeError("sampling-null queue must contain directed off-diagonal cells")
        for column, resolved in (("source_index", source), ("target_index", target)):
            if column in frame.columns and pd.notna(row[column]) and int(row[column]) != resolved:
                raise RuntimeError(f"queue {column} disagrees with the neuron name")
        lag, horizon = int(row["source_lag_frames"]), int(row["horizon_frames"])
        if lag < 0 or horizon < 1:
            raise RuntimeError("queue contains an invalid lag or horizon")
        context = str(row["context"]).strip().lower()
        phases = context_phases(context)
        key = (source, target, lag, horizon, context)
        if key in seen:
            raise RuntimeError("sampling-null queue contains a duplicate exact cell")
        seen.add(key)
        queue_rank = (
            int(row["queue_rank"])
            if "queue_rank" in frame.columns and pd.notna(row["queue_rank"])
            else position
        )
        if pd.isna(row["selection_origin"]):
            raise RuntimeError("sampling-null queue contains an empty selection_origin")
        selection_origin = str(row["selection_origin"]).strip()
        if not selection_origin:
            raise RuntimeError("sampling-null queue contains an empty selection_origin")
        cell_spec = {
            "queue_sha256": queue_hash,
            "queue_rank": queue_rank,
            "source": source,
            "target": target,
            "lag": lag,
            "horizon": horizon,
            "context": context,
            "channel": str(row.get("channel", "endpoint_wasserstein1")),
        }
        cells.append(
            SelectedCell(
                candidate_id=f"cell_{canonical_fingerprint(cell_spec)[:16]}",
                queue_rank=queue_rank,
                source=source,
                target=target,
                source_name=source_name,
                target_name=target_name,
                lag=lag,
                horizon=horizon,
                context=context,
                phases=phases,
                channel=str(row.get("channel", "endpoint_wasserstein1")),
                selection_origin=selection_origin,
            )
        )
    cells.sort(key=lambda cell: (cell.queue_rank, cell.candidate_id))

    groups: list[SamplingGroup] = []
    group_keys = sorted({(cell.lag, cell.horizon, cell.phases) for cell in cells})
    for lag, horizon, phases in group_keys:
        current = tuple(
            cell
            for cell in cells
            if (cell.lag, cell.horizon, cell.phases) == (lag, horizon, phases)
        )
        sources = tuple(sorted({cell.source for cell in current}))
        targets = tuple(sorted({cell.target for cell in current}))
        group_spec = {
            "queue_sha256": queue_hash,
            "lag": lag,
            "horizon": horizon,
            "phases": list(phases),
            "sources": list(sources),
            "targets": list(targets),
            "candidate_ids": [cell.candidate_id for cell in current],
        }
        groups.append(
            SamplingGroup(
                lag=lag,
                horizon=horizon,
                phases=phases,
                sources=sources,
                targets=targets,
                cells=current,
                selection_fingerprint=canonical_fingerprint(group_spec),
            )
        )
    return tuple(cells), tuple(groups)


def sampling_subseed(canonical_seed: int, namespace: str, replicate: int) -> int:
    if namespace == "observed" and replicate == 0:
        return int(canonical_seed)
    payload = f"sampling-null-v1|{canonical_seed}|{namespace}|{replicate}".encode()
    return int(int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31 - 1))


def _phase_offset(phase: str, fps: float, source_window_frames: int) -> int:
    if phase == "baseline":
        return -int(round(15.0 * fps))
    if phase == "onset":
        return int(source_window_frames) - 1
    raise ValueError(f"quiet pseudo-boundaries do not support phase {phase!r}")


def quiet_pseudo_plan(
    schedule: StimulusSchedule,
    *,
    phases: tuple[str, ...],
    history_frames: int,
    source_lag_frames: int,
    source_window_frames: int,
    horizon_frames: int,
    replicates: int,
    stimulus: np.ndarray,
) -> QuietPseudoPlan:
    """Construct deterministic all-zero-stimulus matched quiet boundaries."""
    if replicates < 1:
        raise ValueError("at least one quiet pseudo-boundary is required")
    fps = float(schedule.analysis_fps)
    intervals = np.asarray(schedule.event_intervals_seconds, dtype=np.float64)
    starts = np.rint(intervals[:, 0] * fps).astype(np.int32)
    stops = np.rint(intervals[:, 1] * fps).astype(np.int32)
    quiet = np.stack([np.r_[0, stops[:-1]], starts], axis=1).astype(np.int32)
    p, e = len(phases), len(intervals)
    boundaries = np.full((e, replicates), -1, dtype=np.int32)
    cuts = np.full((p, e, replicates), -1, dtype=np.int32)
    source_bounds = np.full((p, e, replicates, 2), -1, dtype=np.int32)
    verified = np.zeros((p, e, replicates), dtype=bool)
    repair_frames = int(source_lag_frames + source_window_frames)
    for event, (quiet_start, quiet_stop) in enumerate(quiet):
        offsets = [
            _phase_offset(phase, fps, source_window_frames) for phase in phases
        ]
        lower = max(
            int(quiet_start) + repair_frames + int(history_frames) - 1 - offset
            for offset in offsets
        )
        upper = min(
            int(quiet_stop) - 1 - int(horizon_frames) - offset
            for offset in offsets
        )
        if upper - lower + 1 < replicates:
            raise RuntimeError(
                f"event {event} has only {max(0, upper - lower + 1)} feasible quiet "
                f"boundaries for {replicates} requested replicas"
            )
        chosen = np.rint(np.linspace(lower, upper, replicates)).astype(np.int32)
        if len(np.unique(chosen)) != replicates:
            raise RuntimeError("quiet pseudo-boundary construction produced duplicates")
        boundaries[event] = chosen
        for phase_position, (phase, offset) in enumerate(zip(phases, offsets)):
            for replicate, boundary in enumerate(chosen):
                cut = int(boundary + offset)
                history_lo = cut - repair_frames - int(history_frames) + 1
                future_hi = cut + int(horizon_frames)
                if not (
                    int(quiet_start) <= history_lo <= cut < future_hi < int(quiet_stop)
                ):
                    raise RuntimeError("quiet pseudo-boundary escapes its feasible interval")
                source_hi = cut - int(source_lag_frames) + 1
                source_lo = source_hi - int(source_window_frames)
                cuts[phase_position, event, replicate] = cut
                source_bounds[phase_position, event, replicate] = (
                    source_lo,
                    source_hi,
                )
                current = np.asarray(stimulus[history_lo : future_hi + 1])
                verified[phase_position, event, replicate] = bool(
                    current.size and np.allclose(current, 0.0, rtol=0.0, atol=0.0)
                )
    if not verified.all():
        raise RuntimeError("one or more quiet pseudo-windows contain observed stimulus")
    return QuietPseudoPlan(
        quiet_intervals=quiet,
        boundary_times=boundaries,
        cut_times=cuts,
        source_window_bounds=source_bounds,
        quiet_verified=verified,
    )


def _group_output_path(
    output: Path,
    model_id: str,
    group: SamplingGroup,
    fold: int,
    seed: int,
    particles: int,
) -> Path:
    return (
        output
        / "responses"
        / (
            f"{model_id}__sampling_null__ell{group.lag}__h{group.horizon}"
            f"__N{particles}__f{fold}__s{seed}"
            f"__sel{group.selection_fingerprint[:12]}.npz"
        )
    )


def _result_endpoint(result: Mapping[str, np.ndarray]) -> np.ndarray:
    value = np.asarray(result["particle_endpoint"], dtype=np.float32)
    if value.ndim != 5 or value.shape[0] != 2:
        raise RuntimeError("sampler particle endpoint schema changed")
    # [arm, source, horizon, particle, target] ->
    # [arm, horizon, particle, target, source]
    return np.transpose(value, (0, 2, 3, 4, 1))


def _copy_diagnostics(
    destination: dict[str, np.ndarray],
    prefix: tuple[int, ...],
    result: Mapping[str, np.ndarray],
) -> None:
    for field in DIAGNOSTIC_FIELDS:
        value = np.asarray(result[f"diagnostic_{field}"], dtype=np.float32)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise RuntimeError(f"invalid sampling-null diagnostic {field}")
        destination[field][prefix] = value


def _allocate_diagnostics(
    prefix_shape: tuple[int, ...], source_count: int
) -> dict[str, np.ndarray]:
    return {
        field: np.full((*prefix_shape, source_count), np.nan, dtype=np.float32)
        for field in DIAGNOSTIC_FIELDS
    }


def _sample_pair(
    adapter: GeneratorAdapter,
    standardized: np.ndarray,
    stimulus: np.ndarray,
    *,
    cut_time: int,
    projection: np.ndarray,
    source_low: np.ndarray,
    source_high: np.ndarray,
    source_iqr: np.ndarray,
    thresholds: np.ndarray,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
    config: RepairedResponseConfig,
    seed: int,
    branch_factor: int,
    future_branch_factor: int,
) -> dict[str, np.ndarray]:
    return progressive_smc_repaired_responses(
        adapter,
        standardized,
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
        source_indices=np.asarray(sources, dtype=np.int64),
        include_factual_arm=False,
        particle_target_indices=np.asarray(targets, dtype=np.int64),
    )


def _candidate_archive_arrays(group: SamplingGroup) -> dict[str, np.ndarray]:
    source_local = {value: index for index, value in enumerate(group.sources)}
    target_local = {value: index for index, value in enumerate(group.targets)}
    return {
        "candidate_ids": np.asarray([cell.candidate_id for cell in group.cells]),
        "candidate_queue_rank": np.asarray(
            [cell.queue_rank for cell in group.cells], dtype=np.int32
        ),
        "candidate_source_local_index": np.asarray(
            [source_local[cell.source] for cell in group.cells], dtype=np.int16
        ),
        "candidate_target_local_index": np.asarray(
            [target_local[cell.target] for cell in group.cells], dtype=np.int16
        ),
        "candidate_source_index": np.asarray(
            [cell.source for cell in group.cells], dtype=np.int16
        ),
        "candidate_target_index": np.asarray(
            [cell.target for cell in group.cells], dtype=np.int16
        ),
        "candidate_context": np.asarray([cell.context for cell in group.cells]),
        "candidate_channel": np.asarray([cell.channel for cell in group.cells]),
        "candidate_selection_origin": np.asarray(
            [cell.selection_origin for cell in group.cells]
        ),
    }


def validate_archive(
    path: Path,
    *,
    cohort,
    group: SamplingGroup,
    checkpoint: Path,
    fold: int,
    seed: int,
    heldout: np.ndarray,
    model_id: str,
    history_frames: int,
    source_window_frames: int,
    particles: int,
    future_branch_factor: int,
    observed_replicates: int,
    midpoint_replicates: int,
    pseudo_replicates: int,
    queue_path: Path,
    run_spec_fingerprint: str,
    base_seed: int,
    min_ess: float,
    branch_factor: int,
    requested_device: str,
    expected_archive_schema_version: str = ARCHIVE_SCHEMA_VERSION,
    require_selection_origin: bool = True,
) -> None:
    """Fail closed before a completed archive may be resumed."""
    with np.load(path, allow_pickle=False) as data:
        scalar_expected = {
            "status": "complete",
            "archive_schema_version": expected_archive_schema_version,
            "method": METHOD,
            "model_id": model_id,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "fold": fold,
            "seed": seed,
            "base_seed": base_seed,
            "history_frames": history_frames,
            "repair_frames": group.lag + source_window_frames,
            "source_window_frames": source_window_frames,
            "source_lag_frames": group.lag,
            "source_lag_seconds": group.lag / float(cohort.fps),
            "horizon_frames": group.horizon,
            "horizon_seconds": group.horizon / float(cohort.fps),
            "source_to_readout_seconds": (group.lag + group.horizon)
            / float(cohort.fps),
            "n_particles": particles,
            "branch_factor": branch_factor,
            "future_branch_factor": future_branch_factor,
            "minimum_effective_sample_size": min_ess,
            "observed_sampler_replicates": observed_replicates,
            "midpoint_sampler_replicates": midpoint_replicates,
            "quiet_pseudo_boundary_replicates": pseudo_replicates,
            "selection_fingerprint": group.selection_fingerprint,
            "run_spec_fingerprint": run_spec_fingerprint,
            "archive_spec_fingerprint": canonical_fingerprint(
                {
                    "run_spec_fingerprint": run_spec_fingerprint,
                    "selection_fingerprint": group.selection_fingerprint,
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_sha256": sha256(checkpoint),
                }
            ),
            "hypothesis_queue_sha256": sha256(queue_path),
            "sampling_seed_definition": SAMPLING_SEED_DEFINITION,
            "midpoint_definition": MIDPOINT_DEFINITION,
            "quiet_pseudo_boundary_definition": PSEUDO_BOUNDARY_DEFINITION,
            "claim_boundary": CLAIM_BOUNDARY,
            "fps": float(cohort.fps),
            "source_window_bounds_semantics": "[inclusive_start,exclusive_stop)",
            "stimulus_schema_version": cohort.stimulus_schema_version,
            "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
            "stimulus_generator_encoding": GENERATOR_ENCODING,
            "chemical_identity_conditioned": False,
            "requested_device": requested_device,
        }
        missing = sorted(set(scalar_expected).difference(data.files))
        if missing:
            raise RuntimeError(f"resume rejected: archive lacks fields {missing}")
        for key, expected in scalar_expected.items():
            actual = data[key].item()
            equal = (
                bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-8))
                if isinstance(expected, float)
                else actual == expected
            )
            if not equal:
                raise RuntimeError(f"resume rejected: archive {key} differs")
        if "resolved_device" not in data.files or not str(
            data["resolved_device"].item()
        ):
            raise RuntimeError("resume rejected: archive lacks a resolved device")
        if tuple(data["worm_ids"].astype(str)) != tuple(
            cohort.worm_ids[int(index)] for index in heldout
        ):
            raise RuntimeError("resume rejected: held-out worm order differs")
        if not np.array_equal(data["worm_indices"], heldout.astype(np.int16)):
            raise RuntimeError("resume rejected: held-out worm indices differ")
        if not np.array_equal(data["phase_names"], np.asarray(group.phases)):
            raise RuntimeError("resume rejected: phase names differ")
        if not np.array_equal(data["selected_source_indices"], group.sources):
            raise RuntimeError("resume rejected: selected sources differ")
        if not np.array_equal(data["selected_target_indices"], group.targets):
            raise RuntimeError("resume rejected: selected targets differ")
        semantic_arrays = {
            "arm_names": np.asarray(ARM_NAMES),
            "endpoint_sample_axes": np.asarray(
                (
                    "heldout_worm",
                    "phase",
                    "event",
                    "sampler_replicate",
                    "arm",
                    "horizon",
                    "future_particle",
                    "selected_target",
                    "source",
                )
            ),
            "selected_source_neurons": np.asarray(
                [cohort.neurons[index] for index in group.sources]
            ),
            "selected_target_neurons": np.asarray(
                [cohort.neurons[index] for index in group.targets]
            ),
        }
        for key, expected in semantic_arrays.items():
            if key not in data.files or not np.array_equal(data[key], expected):
                raise RuntimeError(f"resume rejected: semantic axis {key} differs")
        if (
            "matrix_orientation" not in data.files
            or str(data["matrix_orientation"].item())
            != "target_row_source_column"
        ):
            raise RuntimeError("resume rejected: matrix orientation differs")
        for key, expected in _candidate_archive_arrays(group).items():
            if key == "candidate_selection_origin" and not require_selection_origin:
                continue
            if key not in data.files or not np.array_equal(data[key], expected):
                raise RuntimeError(f"resume rejected: candidate metadata {key} differs")
        schedule_expected = {
            "chemical_code_by_worm_event": np.asarray(
                [
                    cohort.stimulus_schedules[int(index)].chemical_code_by_event
                    for index in heldout
                ],
                dtype=np.int8,
            ),
            "chemical_name_by_worm_event": np.asarray(
                [
                    cohort.stimulus_schedules[int(index)].chemical_name_by_event
                    for index in heldout
                ]
            ),
            "event_intervals_seconds_by_worm": np.asarray(
                [
                    cohort.stimulus_schedules[int(index)].event_intervals_seconds
                    for index in heldout
                ],
                dtype=np.float32,
            ),
            "schedule_source_recording_by_worm": np.asarray(
                [
                    cohort.stimulus_schedules[int(index)].source_recording
                    for index in heldout
                ]
            ),
            "schedule_native_fps_by_worm": np.asarray(
                [
                    cohort.stimulus_schedules[int(index)].native_fps
                    for index in heldout
                ],
                dtype=np.float32,
            ),
            "schedule_analysis_fps_by_worm": np.asarray(
                [
                    cohort.stimulus_schedules[int(index)].analysis_fps
                    for index in heldout
                ],
                dtype=np.float32,
            ),
            "schedule_resampling_provenance_by_worm": np.asarray(
                [
                    cohort.stimulus_schedules[int(index)].resampling_provenance
                    for index in heldout
                ]
            ),
        }
        for key, expected in schedule_expected.items():
            if key not in data.files:
                raise RuntimeError(f"resume rejected: archive lacks {key}")
            actual = np.asarray(data[key])
            equal = (
                np.allclose(actual, expected, rtol=0.0, atol=1e-7)
                if np.issubdtype(expected.dtype, np.floating)
                else np.array_equal(actual, expected)
            )
            if not equal:
                raise RuntimeError(f"resume rejected: archive {key} differs")
        w, p, e, m, t = (
            len(heldout),
            len(group.phases),
            3,
            len(group.sources),
            len(group.targets),
        )
        nf = particles * future_branch_factor
        shapes = {
            "observed_endpoint_samples": (
                w, p, e, observed_replicates, 2, 1, nf, t, m
            ),
            "midpoint_endpoint_samples": (
                w, p, e, midpoint_replicates, 1, nf, t, m
            ),
            "pseudo_endpoint_samples": (
                w, p, e, pseudo_replicates, 2, 1, nf, t, m
            ),
            "observed_repair_ancestor_id": (
                w, p, e, observed_replicates, 2, m, particles
            ),
            "midpoint_repair_ancestor_id": (
                w, p, e, midpoint_replicates, m, particles
            ),
            "pseudo_repair_ancestor_id": (
                w, p, e, pseudo_replicates, 2, m, particles
            ),
        }
        for key, shape in shapes.items():
            if key not in data.files or data[key].shape != shape:
                raise RuntimeError(f"resume rejected: archive {key} shape differs")
            if not np.isfinite(data[key]).all():
                raise RuntimeError(f"resume rejected: archive {key} is nonfinite")
        parent = np.repeat(np.arange(particles, dtype=np.int16), future_branch_factor)
        if not np.array_equal(data["future_parent_index"], parent):
            raise RuntimeError("resume rejected: future-parent mapping differs")
        for key in (
            "observed_repair_ancestor_id",
            "midpoint_repair_ancestor_id",
            "pseudo_repair_ancestor_id",
        ):
            value = np.asarray(data[key])
            if np.any(value < 0) or np.any(value >= particles):
                raise RuntimeError(f"resume rejected: archive {key} is out of range")
        diagnostic_shapes = {
            "observed": (w, p, e, observed_replicates, m),
            "midpoint": (w, p, e, midpoint_replicates, m),
            "pseudo": (w, p, e, pseudo_replicates, m),
        }
        for prefix, shape in diagnostic_shapes.items():
            for field in DIAGNOSTIC_FIELDS:
                key = f"{prefix}_diagnostic_{field}"
                if key not in data.files or data[key].shape != shape:
                    raise RuntimeError(f"resume rejected: archive {key} shape differs")
                if not np.isfinite(data[key]).all():
                    raise RuntimeError(f"resume rejected: archive {key} is nonfinite")
            recomputed_valid = recompute_diagnostic_valid(
                data, prefix, min_ess=min_ess
            )
            stored_raw = np.asarray(data[f"{prefix}_diagnostic_valid"])
            if not np.isin(stored_raw, (0, 1)).all():
                raise RuntimeError(
                    f"resume rejected: archive {prefix} diagnostic validity is nonbinary"
                )
            stored_valid = stored_raw.astype(bool)
            if not np.array_equal(recomputed_valid, stored_valid):
                raise RuntimeError(
                    f"resume rejected: archive {prefix} diagnostic validity "
                    "differs from saved ESS, weight, and achieved-gap components"
                )
        if np.any(data["observed_diagnostic_target_gap"] <= 0) or np.any(
            data["pseudo_diagnostic_target_gap"] <= 0
        ):
            raise RuntimeError("resume rejected: real/pseudo source target gap is invalid")
        if not np.allclose(data["midpoint_diagnostic_target_gap"], 0.0, rtol=0, atol=1e-7):
            raise RuntimeError("resume rejected: midpoint target gap is nonzero")
        if not np.asarray(data["pseudo_quiet_verified"], dtype=bool).all():
            raise RuntimeError("resume rejected: pseudo windows are not all quiet")
        for local, worm in enumerate(heldout):
            stimulus = stimulus_for_trace(
                cohort.traces[int(worm)], cohort, int(worm), None
            )
            plan = quiet_pseudo_plan(
                cohort.stimulus_schedules[int(worm)],
                phases=group.phases,
                history_frames=history_frames,
                source_lag_frames=group.lag,
                source_window_frames=source_window_frames,
                horizon_frames=group.horizon,
                replicates=pseudo_replicates,
                stimulus=stimulus,
            )
            for key, expected in (
                ("pseudo_quiet_intervals", plan.quiet_intervals),
                ("pseudo_boundary_times", plan.boundary_times),
                ("pseudo_cut_times", plan.cut_times),
                ("pseudo_source_window_bounds", plan.source_window_bounds),
                ("pseudo_quiet_verified", plan.quiet_verified),
            ):
                if not np.array_equal(data[key][local], expected):
                    raise RuntimeError(f"resume rejected: recomputed {key} differs")


def run_one(
    *,
    cohort,
    folds: np.ndarray,
    source_run: Path,
    checkpoint_phase: str,
    output: Path,
    queue_path: Path,
    model_id: str,
    history_frames: int,
    group: SamplingGroup,
    fold: int,
    seed: int,
    particles: int,
    source_window_frames: int,
    observed_replicates: int,
    midpoint_replicates: int,
    pseudo_replicates: int,
    device: str,
    base_seed: int,
    min_ess: float,
    branch_factor: int,
    future_branch_factor: int,
    run_spec_fingerprint: str,
) -> dict[str, Any]:
    path = _group_output_path(output, model_id, group, fold, seed, particles)
    checkpoint = checkpoint_path(
        source_run, checkpoint_phase, model_id, history_frames, fold, seed
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    heldout = np.flatnonzero(np.asarray(folds) == fold)
    if not len(heldout):
        raise RuntimeError(f"fold {fold} has no held-out worms")
    if path.exists():
        validate_archive(
            path,
            cohort=cohort,
            group=group,
            checkpoint=checkpoint,
            fold=fold,
            seed=seed,
            heldout=heldout,
            model_id=model_id,
            history_frames=history_frames,
            source_window_frames=source_window_frames,
            particles=particles,
            future_branch_factor=future_branch_factor,
            observed_replicates=observed_replicates,
            midpoint_replicates=midpoint_replicates,
            pseudo_replicates=pseudo_replicates,
            queue_path=queue_path,
            run_spec_fingerprint=run_spec_fingerprint,
            base_seed=base_seed,
            min_ess=min_ess,
            branch_factor=branch_factor,
            requested_device=device,
        )
        return {"status": "skipped", "output": str(path), "wall_seconds": 0.0}

    started = time.perf_counter()
    adapter = GeneratorAdapter.load(str(checkpoint), device=device)
    _validate_checkpoint(
        adapter,
        cohort=cohort,
        model_id=model_id,
        fold=fold,
        seed=seed,
        history_lag=history_frames,
    )
    config = RepairedResponseConfig(
        history_frames=history_frames,
        repair_frames=group.lag + source_window_frames,
        source_window_frames=source_window_frames,
        source_lag_frames=group.lag,
        horizon_frames=(group.horizon,),
        n_particles=particles,
        min_ess=min_ess,
        resample_ess_fraction=0.50,
        sampling_chunk_size=1024,
    )
    config.validate()
    projection, quantiles, thresholds = training_context(
        cohort, folds, fold, adapter.checkpoint, config
    )
    w, p, e, m, t, nf = (
        len(heldout),
        len(group.phases),
        3,
        len(group.sources),
        len(group.targets),
        particles * future_branch_factor,
    )
    observed_endpoint = np.full(
        (w, p, e, observed_replicates, 2, 1, nf, t, m),
        np.nan,
        dtype=np.float32,
    )
    midpoint_endpoint = np.full(
        (w, p, e, midpoint_replicates, 1, nf, t, m),
        np.nan,
        dtype=np.float32,
    )
    pseudo_endpoint = np.full(
        (w, p, e, pseudo_replicates, 2, 1, nf, t, m),
        np.nan,
        dtype=np.float32,
    )
    observed_ancestor = np.full(
        (w, p, e, observed_replicates, 2, m, particles), -1, dtype=np.int16
    )
    midpoint_ancestor = np.full(
        (w, p, e, midpoint_replicates, m, particles), -1, dtype=np.int16
    )
    pseudo_ancestor = np.full(
        (w, p, e, pseudo_replicates, 2, m, particles), -1, dtype=np.int16
    )
    observed_diagnostic = _allocate_diagnostics((w, p, e, observed_replicates), m)
    midpoint_diagnostic = _allocate_diagnostics((w, p, e, midpoint_replicates), m)
    pseudo_diagnostic = _allocate_diagnostics((w, p, e, pseudo_replicates), m)
    observed_cut_times = np.full((w, p, e), -1, dtype=np.int32)
    observed_source_bounds = np.full((w, p, e, 2), -1, dtype=np.int32)
    quiet_intervals = np.full((w, e, 2), -1, dtype=np.int32)
    pseudo_boundaries = np.full((w, e, pseudo_replicates), -1, dtype=np.int32)
    pseudo_cuts = np.full((w, p, e, pseudo_replicates), -1, dtype=np.int32)
    pseudo_source_bounds = np.full(
        (w, p, e, pseudo_replicates, 2), -1, dtype=np.int32
    )
    pseudo_verified = np.zeros((w, p, e, pseudo_replicates), dtype=bool)

    for worm_position, worm in enumerate(heldout):
        standardized = causal_fill(
            standardize_for_checkpoint(cohort.traces[int(worm)], adapter.checkpoint)
        )
        stimulus = stimulus_for_trace(
            cohort.traces[int(worm)], cohort, int(worm), adapter.checkpoint
        )
        cut_lookup = {
            (cut.phase, cut.event): cut.time
            for cut in episode_cuts(
                len(cohort.traces[int(worm)]),
                cohort.stimulus_schedules[int(worm)],
                source_window_frames,
            )
        }
        plan = quiet_pseudo_plan(
            cohort.stimulus_schedules[int(worm)],
            phases=group.phases,
            history_frames=history_frames,
            source_lag_frames=group.lag,
            source_window_frames=source_window_frames,
            horizon_frames=group.horizon,
            replicates=pseudo_replicates,
            stimulus=stimulus,
        )
        quiet_intervals[worm_position] = plan.quiet_intervals
        pseudo_boundaries[worm_position] = plan.boundary_times
        pseudo_cuts[worm_position] = plan.cut_times
        pseudo_source_bounds[worm_position] = plan.source_window_bounds
        pseudo_verified[worm_position] = plan.quiet_verified
        for phase_position, phase in enumerate(group.phases):
            phase_values = quantiles[phase]
            midpoint = 0.5 * (phase_values["low"] + phase_values["high"])
            for event in range(3):
                cut_time = int(cut_lookup[(phase, event)])
                observed_cut_times[worm_position, phase_position, event] = cut_time
                source_hi = cut_time - group.lag + 1
                observed_source_bounds[worm_position, phase_position, event] = (
                    source_hi - source_window_frames,
                    source_hi,
                )
                canonical_seed = draw_seed(
                    base_seed,
                    model_id,
                    fold,
                    seed,
                    int(worm),
                    PHASES.index(phase),
                    event,
                    group.lag,
                )
                for replicate in range(observed_replicates):
                    result = _sample_pair(
                        adapter,
                        standardized,
                        stimulus,
                        cut_time=cut_time,
                        projection=projection,
                        source_low=phase_values["low"],
                        source_high=phase_values["high"],
                        source_iqr=phase_values["iqr"],
                        thresholds=thresholds,
                        sources=group.sources,
                        targets=group.targets,
                        config=config,
                        seed=sampling_subseed(canonical_seed, "observed", replicate),
                        branch_factor=branch_factor,
                        future_branch_factor=future_branch_factor,
                    )
                    observed_endpoint[
                        worm_position, phase_position, event, replicate
                    ] = _result_endpoint(result)
                    observed_ancestor[
                        worm_position, phase_position, event, replicate
                    ] = np.asarray(result["particle_repair_ancestor_id"], dtype=np.int16)
                    _copy_diagnostics(
                        observed_diagnostic,
                        (worm_position, phase_position, event, replicate),
                        result,
                    )
                for replicate in range(midpoint_replicates):
                    result = _sample_pair(
                        adapter,
                        standardized,
                        stimulus,
                        cut_time=cut_time,
                        projection=projection,
                        source_low=midpoint,
                        source_high=midpoint,
                        source_iqr=phase_values["iqr"],
                        thresholds=thresholds,
                        sources=group.sources,
                        targets=group.targets,
                        config=config,
                        seed=sampling_subseed(canonical_seed, "midpoint", replicate),
                        branch_factor=branch_factor,
                        future_branch_factor=future_branch_factor,
                    )
                    midpoint_endpoint[
                        worm_position, phase_position, event, replicate
                    ] = _result_endpoint(result)[0]
                    midpoint_ancestor[
                        worm_position, phase_position, event, replicate
                    ] = np.asarray(
                        result["particle_repair_ancestor_id"], dtype=np.int16
                    )[0]
                    _copy_diagnostics(
                        midpoint_diagnostic,
                        (worm_position, phase_position, event, replicate),
                        result,
                    )
                for replicate in range(pseudo_replicates):
                    result = _sample_pair(
                        adapter,
                        standardized,
                        stimulus,
                        cut_time=int(plan.cut_times[phase_position, event, replicate]),
                        projection=projection,
                        source_low=phase_values["low"],
                        source_high=phase_values["high"],
                        source_iqr=phase_values["iqr"],
                        thresholds=thresholds,
                        sources=group.sources,
                        targets=group.targets,
                        config=config,
                        seed=sampling_subseed(canonical_seed, "observed", 0),
                        branch_factor=branch_factor,
                        future_branch_factor=future_branch_factor,
                    )
                    pseudo_endpoint[
                        worm_position, phase_position, event, replicate
                    ] = _result_endpoint(result)
                    pseudo_ancestor[
                        worm_position, phase_position, event, replicate
                    ] = np.asarray(result["particle_repair_ancestor_id"], dtype=np.int16)
                    _copy_diagnostics(
                        pseudo_diagnostic,
                        (worm_position, phase_position, event, replicate),
                        result,
                    )
        print(
            f"SAMPLING_NULL_WORM_DONE ell={group.lag} h={group.horizon} "
            f"fold={fold} worm={int(worm)} sources={m} targets={t}",
            flush=True,
        )

    arrays_to_check = (
        observed_endpoint,
        midpoint_endpoint,
        pseudo_endpoint,
        observed_ancestor,
        midpoint_ancestor,
        pseudo_ancestor,
        observed_cut_times,
        observed_source_bounds,
        quiet_intervals,
        pseudo_boundaries,
        pseudo_cuts,
        pseudo_source_bounds,
    )
    if any(not np.isfinite(value).all() for value in arrays_to_check):
        raise RuntimeError("sampling-null archive contains incomplete arrays")
    for diagnostic in (observed_diagnostic, midpoint_diagnostic, pseudo_diagnostic):
        if any(not np.isfinite(value).all() for value in diagnostic.values()):
            raise RuntimeError("sampling-null archive contains incomplete diagnostics")
    elapsed = time.perf_counter() - started
    _atomic_npz(
        path,
        status=np.asarray("complete"),
        archive_schema_version=np.asarray(ARCHIVE_SCHEMA_VERSION),
        created_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        method=np.asarray(METHOD),
        model_id=np.asarray(model_id),
        checkpoint=np.asarray(str(checkpoint.resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        run_spec_fingerprint=np.asarray(run_spec_fingerprint),
        archive_spec_fingerprint=np.asarray(
            canonical_fingerprint(
                {
                    "run_spec_fingerprint": run_spec_fingerprint,
                    "selection_fingerprint": group.selection_fingerprint,
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
        ),
        fold=np.asarray(fold),
        seed=np.asarray(seed),
        base_seed=np.asarray(base_seed),
        history_frames=np.asarray(history_frames),
        source_window_frames=np.asarray(source_window_frames),
        repair_frames=np.asarray(group.lag + source_window_frames),
        source_lag_frames=np.asarray(group.lag),
        source_lag_seconds=np.asarray(group.lag / float(cohort.fps)),
        horizon_frames=np.asarray(group.horizon),
        horizon_seconds=np.asarray(group.horizon / float(cohort.fps)),
        source_to_readout_seconds=np.asarray(
            (group.lag + group.horizon) / float(cohort.fps)
        ),
        n_particles=np.asarray(particles),
        branch_factor=np.asarray(branch_factor),
        future_branch_factor=np.asarray(future_branch_factor),
        minimum_effective_sample_size=np.asarray(min_ess),
        observed_sampler_replicates=np.asarray(observed_replicates),
        midpoint_sampler_replicates=np.asarray(midpoint_replicates),
        quiet_pseudo_boundary_replicates=np.asarray(pseudo_replicates),
        selection_fingerprint=np.asarray(group.selection_fingerprint),
        hypothesis_queue=np.asarray(str(queue_path.resolve())),
        hypothesis_queue_sha256=np.asarray(sha256(queue_path)),
        sampling_seed_definition=np.asarray(SAMPLING_SEED_DEFINITION),
        midpoint_definition=np.asarray(MIDPOINT_DEFINITION),
        quiet_pseudo_boundary_definition=np.asarray(PSEUDO_BOUNDARY_DEFINITION),
        claim_boundary=np.asarray(CLAIM_BOUNDARY),
        fps=np.asarray(float(cohort.fps)),
        source_window_bounds_semantics=np.asarray(
            "[inclusive_start,exclusive_stop)"
        ),
        requested_device=np.asarray(device),
        resolved_device=np.asarray(str(adapter.device)),
        stimulus_schema_version=np.asarray(cohort.stimulus_schema_version),
        stimulus_schema_fingerprint=np.asarray(
            cohort.stimulus_schema_fingerprint
        ),
        stimulus_generator_encoding=np.asarray(GENERATOR_ENCODING),
        chemical_identity_conditioned=np.asarray(False),
        phase_names=np.asarray(group.phases),
        arm_names=np.asarray(ARM_NAMES),
        selected_source_indices=np.asarray(group.sources, dtype=np.int16),
        selected_source_neurons=np.asarray(
            [cohort.neurons[index] for index in group.sources]
        ),
        selected_target_indices=np.asarray(group.targets, dtype=np.int16),
        selected_target_neurons=np.asarray(
            [cohort.neurons[index] for index in group.targets]
        ),
        endpoint_sample_axes=np.asarray(
            (
                "heldout_worm",
                "phase",
                "event",
                "sampler_replicate",
                "arm",
                "horizon",
                "future_particle",
                "selected_target",
                "source",
            )
        ),
        matrix_orientation=np.asarray("target_row_source_column"),
        future_parent_index=np.repeat(
            np.arange(particles, dtype=np.int16), future_branch_factor
        ),
        worm_indices=heldout.astype(np.int16),
        worm_ids=np.asarray([cohort.worm_ids[int(index)] for index in heldout]),
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
        observed_cut_times=observed_cut_times,
        observed_source_window_bounds=observed_source_bounds,
        pseudo_quiet_intervals=quiet_intervals,
        pseudo_boundary_times=pseudo_boundaries,
        pseudo_cut_times=pseudo_cuts,
        pseudo_source_window_bounds=pseudo_source_bounds,
        pseudo_quiet_verified=pseudo_verified,
        observed_endpoint_samples=observed_endpoint,
        midpoint_endpoint_samples=midpoint_endpoint,
        pseudo_endpoint_samples=pseudo_endpoint,
        observed_repair_ancestor_id=observed_ancestor,
        midpoint_repair_ancestor_id=midpoint_ancestor,
        pseudo_repair_ancestor_id=pseudo_ancestor,
        wall_seconds=np.asarray(elapsed),
        **_candidate_archive_arrays(group),
        **{
            f"observed_diagnostic_{field}": value
            for field, value in observed_diagnostic.items()
        },
        **{
            f"midpoint_diagnostic_{field}": value
            for field, value in midpoint_diagnostic.items()
        },
        **{
            f"pseudo_diagnostic_{field}": value
            for field, value in pseudo_diagnostic.items()
        },
    )
    validate_archive(
        path,
        cohort=cohort,
        group=group,
        checkpoint=checkpoint,
        fold=fold,
        seed=seed,
        heldout=heldout,
        model_id=model_id,
        history_frames=history_frames,
        source_window_frames=source_window_frames,
        particles=particles,
        future_branch_factor=future_branch_factor,
        observed_replicates=observed_replicates,
        midpoint_replicates=midpoint_replicates,
        pseudo_replicates=pseudo_replicates,
        queue_path=queue_path,
        run_spec_fingerprint=run_spec_fingerprint,
        base_seed=base_seed,
        min_ess=min_ess,
        branch_factor=branch_factor,
        requested_device=device,
    )
    del adapter
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {"status": "ok", "output": str(path), "wall_seconds": elapsed}


def _manifest(
    *,
    args,
    source_run: Path,
    fold_file: Path,
    queue_path: Path,
    cohort,
    cells: tuple[SelectedCell, ...],
    groups: tuple[SamplingGroup, ...],
) -> dict[str, Any]:
    implementation_paths = {
        "targeted_sampling_nulls.py": Path(__file__).resolve(),
        "progressive_smc.py": Path(__file__).resolve().with_name(
            "progressive_smc.py"
        ),
        "prediction_atlas_runner.py": Path(__file__).resolve().with_name(
            "prediction_atlas_runner.py"
        ),
    }
    spec = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": "targeted progressive-bridge sampling-null controls v2",
        "method": METHOD,
        "source_run": str(source_run),
        "source_run_manifest_sha256": sha256(source_run / "manifest.json"),
        "fold_assignments": str(fold_file),
        "fold_assignments_sha256": sha256(fold_file),
        "hypothesis_queue": str(queue_path),
        "hypothesis_queue_sha256": sha256(queue_path),
        "model_id": args.model_id,
        "checkpoint_phase": args.checkpoint_phase,
        "cohort_mode": args.cohort_mode,
        "cohort_worms": list(cohort.worm_ids),
        "neurons": list(cohort.neurons),
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_generator_encoding": GENERATOR_ENCODING,
        "chemical_identity_conditioned": False,
        "selected_cells": [cell.to_dict() for cell in cells],
        "selection_origins": sorted({cell.selection_origin for cell in cells}),
        "resolved_groups": [group.to_dict() for group in groups],
        "folds": list(args.folds),
        "seeds": list(args.seeds),
        "base_seed": args.base_seed,
        "device": args.device,
        "history_frames": args.history_lag,
        "source_window_frames": args.source_window_frames,
        "particles": args.particles,
        "branch_factor": args.progressive_branch_factor,
        "future_branch_factor": args.progressive_future_branch_factor,
        "minimum_effective_sample_size": args.min_ess,
        "observed_sampler_replicates": args.sampler_replicates,
        "midpoint_sampler_replicates": args.midpoint_replicates,
        "quiet_pseudo_boundary_replicates": args.pseudo_boundary_replicates,
        "sampling_seed_definition": SAMPLING_SEED_DEFINITION,
        "midpoint_definition": MIDPOINT_DEFINITION,
        "quiet_pseudo_boundary_definition": PSEUDO_BOUNDARY_DEFINITION,
        "same_arm_control_definition": (
            "low_A versus low_B and high_A versus high_B from independently "
            "seeded full repair-plus-future SMC replicas of one checkpoint"
        ),
        "within_arm_split_definition": (
            "analysis-only diagnostic that keeps all future siblings sharing one "
            "terminal repair parent in the same half; distinct terminal parents may "
            "share an earlier repair ancestor; not a biological replicate"
        ),
        "inference_unit": "worm; checkpoint seeds are averaged within worm",
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation_provenance": {
            "files": {
                name: sha256(path) for name, path in implementation_paths.items()
            },
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
            },
        },
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_spec_fingerprint": canonical_fingerprint(spec),
        **spec,
    }


def write_checksums(output: Path) -> None:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    rows = [f"{sha256(path)}  {path.relative_to(output)}" for path in paths]
    _atomic_text(output / "checksums.sha256", "\n".join(rows) + "\n")


def verify_checksums(output: Path) -> dict[str, str]:
    """Verify every raw-run file is covered exactly once by the checksum ledger."""
    output = output.resolve()
    ledger_path = output / "checksums.sha256"
    if not ledger_path.exists():
        raise RuntimeError("sampling-null run lacks checksums.sha256")
    ledger: dict[str, str] = {}
    for line_number, line in enumerate(
        ledger_path.read_text().splitlines(), start=1
    ):
        if line.count("  ") != 1:
            raise RuntimeError(f"checksum line {line_number} is malformed")
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"checksum line {line_number} has an invalid digest")
        rel = Path(relative)
        if not relative or rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"checksum line {line_number} has an unsafe path")
        normalized = rel.as_posix()
        if normalized in ledger:
            raise RuntimeError("checksum ledger contains a duplicate path")
        candidate = (output / rel).resolve()
        try:
            candidate.relative_to(output)
        except ValueError as error:
            raise RuntimeError("checksum path escapes the sampling-null run") from error
        if not candidate.is_file() or sha256(candidate) != digest:
            raise RuntimeError(f"checksum verification failed for {normalized}")
        ledger[normalized] = digest
    expected = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    if set(ledger) != expected:
        raise RuntimeError("checksum ledger coverage differs from the raw file tree")
    return ledger


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
    parser.add_argument("--source-window-frames", type=int, default=4)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--sampler-replicates", type=int, default=2)
    parser.add_argument("--midpoint-replicates", type=int, default=2)
    parser.add_argument("--pseudo-boundary-replicates", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base-seed", type=int, default=20260829)
    parser.add_argument("--min-ess", type=float, default=24.0)
    parser.add_argument("--progressive-branch-factor", type=int, default=2)
    parser.add_argument("--progressive-future-branch-factor", type=int, default=2)
    args = parser.parse_args()
    args.folds = tuple(sorted(set(args.folds)))
    args.seeds = tuple(sorted(set(args.seeds)))
    if not set(args.folds).issubset(range(5)):
        parser.error("folds must be selected from 0..4")
    if args.sampler_replicates < 2 or args.midpoint_replicates < 2:
        parser.error("same-arm and midpoint controls require at least two replicas")
    if args.pseudo_boundary_replicates < 1:
        parser.error("at least one quiet pseudo-boundary is required")
    if min(
        args.history_lag,
        args.source_window_frames,
        args.particles,
        args.progressive_branch_factor,
        args.progressive_future_branch_factor,
    ) < 1:
        parser.error("history, particles, windows, and branch factors must be positive")

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
    queue_path = args.hypothesis_queue.resolve()
    cells, groups = load_selected_cells(queue_path, cohort.neurons)
    manifest = _manifest(
        args=args,
        source_run=source_run,
        fold_file=fold_file,
        queue_path=queue_path,
        cohort=cohort,
        cells=cells,
        groups=groups,
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("run_spec_fingerprint") != manifest["run_spec_fingerprint"]:
            raise RuntimeError("existing sampling-null manifest differs from requested run")
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
                        queue_path=queue_path,
                        model_id=args.model_id,
                        history_frames=args.history_lag,
                        group=group,
                        fold=fold,
                        seed=seed,
                        particles=args.particles,
                        source_window_frames=args.source_window_frames,
                        observed_replicates=args.sampler_replicates,
                        midpoint_replicates=args.midpoint_replicates,
                        pseudo_replicates=args.pseudo_boundary_replicates,
                        device=args.device,
                        base_seed=args.base_seed,
                        min_ess=args.min_ess,
                        branch_factor=args.progressive_branch_factor,
                        future_branch_factor=args.progressive_future_branch_factor,
                        run_spec_fingerprint=manifest["run_spec_fingerprint"],
                    )
                except Exception as error:
                    result = {
                        "status": "failed",
                        "error": repr(error),
                        "wall_seconds": 0.0,
                    }
                records.append(
                    {
                        "selection_fingerprint": group.selection_fingerprint,
                        "source_lag_frames": group.lag,
                        "horizon_frames": group.horizon,
                        "phases": ",".join(group.phases),
                        "source_count": len(group.sources),
                        "target_count": len(group.targets),
                        "cell_count": len(group.cells),
                        "selection_origins": ",".join(
                            sorted({cell.selection_origin for cell in group.cells})
                        ),
                        "fold": fold,
                        "seed": seed,
                        **result,
                    }
                )
                _atomic_csv(output / "run_status.csv", pd.DataFrame(records))
    failures = [record for record in records if record["status"] == "failed"]
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
        "selection_origin_counts": {
            origin: sum(cell.selection_origin == origin for cell in cells)
            for origin in sorted({cell.selection_origin for cell in cells})
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _atomic_json(output / "validation.json", validation)
    write_checksums(output)
    verify_checksums(output)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
