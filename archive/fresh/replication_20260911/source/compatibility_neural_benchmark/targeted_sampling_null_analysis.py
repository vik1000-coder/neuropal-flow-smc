"""Worm-level analysis for targeted progressive-SMC sampling controls."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.full_family_inference import (
    generate_sign_patterns,
)
from compatibility_neural_benchmark.prediction_atlas_runner import (
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    canonical_fingerprint,
    load_folds,
    sha256,
)
from compatibility_neural_benchmark.targeted_sampling_nulls import (
    ARCHIVE_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    METHOD,
    SamplingGroup,
    SelectedCell,
    _atomic_text,
    _group_output_path,
    load_selected_cells,
    recompute_diagnostic_valid,
    validate_archive,
    verify_checksums,
    write_checksums,
)
from conditional_neural_benchmark.data import STIMULUS_NAMES, load_cohort


ANALYSIS_SCHEMA_VERSION = "prediction_atlas_sampling_null_analysis_v2"
METRICS = (
    "endpoint_mean",
    "endpoint_log_sd",
    "endpoint_wasserstein1",
)
ENDPOINT_SD_FLOOR = 1e-6
CONTROL_FAMILIES = (
    "observed",
    "low_low",
    "high_high",
    "midpoint_midpoint",
    "within_low_split",
    "within_high_split",
    "quiet_pseudo",
)


@dataclass(frozen=True)
class AnalysisConfig:
    bootstrap_replicates: int = 2000
    min_gap: float = 0.10
    random_seed: int = 20260830
    strong_valid_fraction: float = 0.80
    genealogy_fraction_threshold: float = 0.10
    alpha: float = 0.05

    def validate(self) -> None:
        if self.bootstrap_replicates < 100:
            raise ValueError("at least 100 worm bootstrap replicates are required")
        if self.min_gap <= 0:
            raise ValueError("the source-gap normalization floor must be positive")
        if not 0 < self.strong_valid_fraction <= 1:
            raise ValueError("strong valid fraction must lie in (0, 1]")
        if not 0 < self.genealogy_fraction_threshold <= 1:
            raise ValueError("genealogy fraction threshold must lie in (0, 1]")
        if not math.isclose(self.genealogy_fraction_threshold, 0.10):
            raise ValueError(
                "analysis schema v2 fixes the genealogy ancestor-fraction "
                "threshold at 0.10"
            )
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")


def _parse_checksum_ledger(root: Path) -> dict[str, str]:
    return verify_checksums(root)


def _load_and_validate_run(run_dir: Path):
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    validation_path = run_dir / "validation.json"
    status_path = run_dir / "run_status.csv"
    for path in (manifest_path, validation_path, status_path):
        if not path.exists():
            raise FileNotFoundError(path)
    ledger = _parse_checksum_ledger(run_dir)
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads(validation_path.read_text())
    fingerprint = str(manifest.get("run_spec_fingerprint", ""))
    if (
        not fingerprint
        or validation.get("status") != "pass"
        or validation.get("run_spec_fingerprint") != fingerprint
        or int(validation.get("failed", -1)) != 0
    ):
        raise RuntimeError("raw sampling-null validation is incomplete or failed")
    source_run = Path(str(manifest["source_run"])).resolve()
    queue_path = Path(str(manifest["hypothesis_queue"])).resolve()
    fold_file = Path(str(manifest["fold_assignments"])).resolve()
    for path, expected in (
        (source_run / "manifest.json", manifest["source_run_manifest_sha256"]),
        (queue_path, manifest["hypothesis_queue_sha256"]),
        (fold_file, manifest["fold_assignments_sha256"]),
    ):
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"raw upstream provenance changed: {path}")
    cohort = load_cohort(cohort_mode=str(manifest["cohort_mode"]))
    if (
        tuple(manifest["cohort_worms"]) != cohort.worm_ids
        or tuple(manifest["neurons"]) != cohort.neurons
        or manifest["stimulus_schema_fingerprint"]
        != cohort.stimulus_schema_fingerprint
    ):
        raise RuntimeError("raw manifest cohort provenance differs from the loaded cohort")
    folds = load_folds(fold_file, cohort)
    cells, groups = load_selected_cells(queue_path, cohort.neurons)
    resolved_groups = [group.to_dict() for group in groups]
    manifest_groups = manifest["resolved_groups"]
    if manifest.get("manifest_schema_version") == "prediction_atlas_sampling_null_manifest_v1":
        # The live v1 run predates preservation of the queue's
        # ``selection_origin`` annotation.  Its queue is hash-pinned above, so
        # recover that annotation from the verified queue while comparing the
        # scientific group specification on the legacy projection.
        resolved_groups = json.loads(json.dumps(resolved_groups))
        for group in resolved_groups:
            for cell in group["cells"]:
                cell.pop("selection_origin", None)
    if resolved_groups != manifest_groups:
        raise RuntimeError("raw group resolution does not reproduce from the queue")
    status = pd.read_csv(status_path)
    expected_runs = len(groups) * len(manifest["folds"]) * len(manifest["seeds"])
    if (
        len(status) != expected_runs
        or status.status.isin(["ok", "skipped"]).sum() != expected_runs
        or int(validation.get("expected_runs", -1)) != expected_runs
        or int(validation.get("completed_or_skipped", -1)) != expected_runs
    ):
        raise RuntimeError("raw run grid is incomplete")

    archives: list[Path] = []
    seen_worms: dict[tuple[str, int], set[str]] = {}
    legacy_v1 = (
        manifest.get("manifest_schema_version")
        == "prediction_atlas_sampling_null_manifest_v1"
    )
    for group in groups:
        for fold in manifest["folds"]:
            heldout = np.flatnonzero(folds == int(fold))
            for seed in manifest["seeds"]:
                archive = _group_output_path(
                    run_dir,
                    str(manifest["model_id"]),
                    group,
                    int(fold),
                    int(seed),
                    int(manifest["particles"]),
                )
                relative = archive.relative_to(run_dir).as_posix()
                if relative not in ledger:
                    raise RuntimeError(f"raw ledger omits archive {relative}")
                checkpoint = (
                    source_run
                    / "checkpoints"
                    / str(manifest["checkpoint_phase"])
                    / (
                        f"{manifest['model_id']}__L{manifest['history_frames']}"
                        f"__f{fold}__s{seed}.pt"
                    )
                )
                validate_archive(
                    archive,
                    cohort=cohort,
                    group=group,
                    checkpoint=checkpoint,
                    fold=int(fold),
                    seed=int(seed),
                    heldout=heldout,
                    model_id=str(manifest["model_id"]),
                    history_frames=int(manifest["history_frames"]),
                    source_window_frames=int(manifest["source_window_frames"]),
                    particles=int(manifest["particles"]),
                    future_branch_factor=int(manifest["future_branch_factor"]),
                    observed_replicates=int(manifest["observed_sampler_replicates"]),
                    midpoint_replicates=int(manifest["midpoint_sampler_replicates"]),
                    pseudo_replicates=int(
                        manifest["quiet_pseudo_boundary_replicates"]
                    ),
                    queue_path=queue_path,
                    run_spec_fingerprint=str(manifest["run_spec_fingerprint"]),
                    base_seed=int(manifest["base_seed"]),
                    min_ess=float(manifest["minimum_effective_sample_size"]),
                    branch_factor=int(manifest["branch_factor"]),
                    requested_device=str(manifest["device"]),
                    expected_archive_schema_version=(
                        "prediction_atlas_sampling_null_v1"
                        if legacy_v1
                        else ARCHIVE_SCHEMA_VERSION
                    ),
                    require_selection_origin=not legacy_v1,
                )
                archives.append(archive)
                key = (group.selection_fingerprint, int(seed))
                seen_worms.setdefault(key, set()).update(
                    cohort.worm_ids[int(index)] for index in heldout
                )
    requested_folds = {int(value) for value in manifest["folds"]}
    expected_worms = {
        cohort.worm_ids[index]
        for index, assigned in enumerate(folds)
        if int(assigned) in requested_folds
    }
    if any(value != expected_worms for value in seen_worms.values()):
        raise RuntimeError("a group/checkpoint seed does not cover every worm exactly once")
    return (
        run_dir,
        manifest,
        validation,
        ledger,
        cohort,
        folds,
        cells,
        groups,
        tuple(archives),
    )


def endpoint_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("endpoint samples must be one-dimensional arrays")
    if (
        not len(left)
        or not len(right)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("endpoint samples must be nonempty and finite")
    pooled = np.sort(np.concatenate([left, right]))
    if len(pooled) < 2:
        wasserstein1 = 0.0
    else:
        intervals = np.diff(pooled)
        left_cdf = np.searchsorted(np.sort(left), pooled[:-1], side="right") / len(
            left
        )
        right_cdf = np.searchsorted(
            np.sort(right), pooled[:-1], side="right"
        ) / len(right)
        wasserstein1 = float(np.sum(np.abs(left_cdf - right_cdf) * intervals))
    return {
        "endpoint_mean": float(left.mean() - right.mean()),
        "endpoint_log_sd": float(
            np.log(left.std(ddof=0) + ENDPOINT_SD_FLOOR)
            - np.log(right.std(ddof=0) + ENDPOINT_SD_FLOOR)
        ),
        "endpoint_wasserstein1": wasserstein1,
    }


def parent_safe_split_masks(
    future_parent_index: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split repair parents, never sibling future descendants."""
    parent = np.asarray(future_parent_index, dtype=np.int64)
    if parent.ndim != 1 or len(parent) < 2:
        raise ValueError("future-parent index must be a nontrivial vector")
    unique = np.unique(parent)
    if len(unique) < 2:
        raise ValueError("at least two repair parents are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    left_parent = set(order[: len(order) // 2].tolist())
    left = np.asarray([value in left_parent for value in parent], dtype=bool)
    right = ~left
    if not left.any() or not right.any():
        raise RuntimeError("parent-safe split produced an empty half")
    if set(parent[left]).intersection(set(parent[right])):
        raise RuntimeError("a future sibling family crosses the split")
    return left, right


def _append_metric_rows(
    rows: list[dict[str, Any]],
    base: Mapping[str, Any],
    *,
    control: str,
    replicate: int,
    left: np.ndarray,
    right: np.ndarray,
    denominator: float,
    valid: bool,
    ancestor_fraction: float,
) -> None:
    for metric, value in endpoint_metrics(left, right).items():
        rows.append(
            {
                **base,
                "control": control,
                "control_replicate": int(replicate),
                "metric": metric,
                "raw_value": value,
                "normalized_value": value / denominator,
                "normalization_denominator": denominator,
                "valid": bool(valid),
                "minimum_ancestor_fraction": float(ancestor_fraction),
            }
        )


def _extract_event_rows(
    manifest: Mapping[str, Any],
    cells: tuple[SelectedCell, ...],
    groups: tuple[SamplingGroup, ...],
    archives: tuple[Path, ...],
    *,
    min_gap: float,
) -> pd.DataFrame:
    group_by_fingerprint = {group.selection_fingerprint: group for group in groups}
    cell_by_id = {cell.candidate_id: cell for cell in cells}
    rows: list[dict[str, Any]] = []
    for archive in archives:
        with np.load(archive, allow_pickle=False) as data:
            group = group_by_fingerprint[str(data["selection_fingerprint"].item())]
            source_local = {value: index for index, value in enumerate(group.sources)}
            target_local = {value: index for index, value in enumerate(group.targets)}
            parent_index = np.asarray(data["future_parent_index"], dtype=np.int64)
            observed = np.asarray(data["observed_endpoint_samples"], dtype=np.float64)
            midpoint = np.asarray(data["midpoint_endpoint_samples"], dtype=np.float64)
            pseudo = np.asarray(data["pseudo_endpoint_samples"], dtype=np.float64)
            obs_ancestor = np.asarray(data["observed_repair_ancestor_id"])
            mid_ancestor = np.asarray(data["midpoint_repair_ancestor_id"])
            pseudo_ancestor = np.asarray(data["pseudo_repair_ancestor_id"])
            diagnostic_valid: dict[str, np.ndarray] = {}
            for prefix in ("observed", "midpoint", "pseudo"):
                recomputed = recompute_diagnostic_valid(
                    data,
                    prefix,
                    min_ess=float(manifest["minimum_effective_sample_size"]),
                )
                stored_raw = np.asarray(data[f"{prefix}_diagnostic_valid"])
                if not np.isin(stored_raw, (0, 1)).all():
                    raise RuntimeError(f"stored {prefix} diagnostic validity is nonbinary")
                stored = stored_raw.astype(bool)
                if not np.array_equal(recomputed, stored):
                    raise RuntimeError(
                        f"stored {prefix} diagnostic validity differs from its "
                        "saved ESS, weight, and achieved-gap components"
                    )
                diagnostic_valid[prefix] = recomputed
            codes = np.asarray(data["chemical_code_by_worm_event"], dtype=np.int8)
            names = np.asarray(data["chemical_name_by_worm_event"]).astype(str)
            observed_replicates = observed.shape[3]
            midpoint_replicates = midpoint.shape[3]
            pseudo_replicates = pseudo.shape[3]
            for cell in group.cells:
                assert cell_by_id[cell.candidate_id] == cell
                si, ti = source_local[cell.source], target_local[cell.target]
                for local_worm, worm_id in enumerate(data["worm_ids"].astype(str)):
                    for phase_position, phase in enumerate(group.phases):
                        for event in range(3):
                            real_gaps = np.asarray(
                                data["observed_diagnostic_achieved_gap"][
                                    local_worm, phase_position, event, :, si
                                ],
                                dtype=np.float64,
                            )
                            if not np.isfinite(real_gaps).all():
                                raise RuntimeError("observed achieved gaps are nonfinite")
                            # Same-state and midpoint comparisons have no
                            # high-low gap of their own.  Scale them by the
                            # smallest matched real-repair gap so the null
                            # magnitude is conservatively large.
                            control_denominator = max(
                                float(np.min(np.abs(real_gaps))), min_gap
                            )
                            base = {
                                "candidate_id": cell.candidate_id,
                                "queue_rank": cell.queue_rank,
                                "source_neuron": cell.source_name,
                                "target_neuron": cell.target_name,
                                "source_index": cell.source,
                                "target_index": cell.target,
                                "source_lag_frames": cell.lag,
                                "horizon_frames": cell.horizon,
                                "context": cell.context,
                                "selected_channel": cell.channel,
                                "checkpoint_seed": int(data["seed"].item()),
                                "fold": int(data["fold"].item()),
                                "worm_id": worm_id,
                                "phase": phase,
                                "event": event,
                                "chemical_code": int(codes[local_worm, event]),
                                "chemical": names[local_worm, event],
                            }
                            for replicate in range(observed_replicates):
                                # Every high-low repair replica has its own
                                # achieved gap.  Reusing replica zero here can
                                # mis-scale an otherwise independent draw.
                                denominator = max(
                                    abs(float(real_gaps[replicate])), min_gap
                                )
                                arm_samples = observed[
                                    local_worm,
                                    phase_position,
                                    event,
                                    replicate,
                                    :,
                                    0,
                                    :,
                                    ti,
                                    si,
                                ]
                                valid = bool(
                                    diagnostic_valid["observed"][
                                        local_worm, phase_position, event, replicate, si
                                    ]
                                )
                                ancestor_fraction = min(
                                    len(
                                        np.unique(
                                            obs_ancestor[
                                                local_worm,
                                                phase_position,
                                                event,
                                                replicate,
                                                arm,
                                                si,
                                            ]
                                        )
                                    )
                                    / int(manifest["particles"])
                                    for arm in range(2)
                                )
                                _append_metric_rows(
                                    rows,
                                    base,
                                    control="observed",
                                    replicate=replicate,
                                    left=arm_samples[1],
                                    right=arm_samples[0],
                                    denominator=denominator,
                                    valid=valid,
                                    ancestor_fraction=ancestor_fraction,
                                )
                                for arm, label in ((0, "low"), (1, "high")):
                                    split_seed = int(
                                        hashlib_sha256_int(
                                            cell.candidate_id,
                                            worm_id,
                                            str(data["seed"].item()),
                                            phase,
                                            str(event),
                                            str(replicate),
                                            label,
                                        )
                                    )
                                    left_mask, right_mask = parent_safe_split_masks(
                                        parent_index, seed=split_seed
                                    )
                                    _append_metric_rows(
                                        rows,
                                        base,
                                        control=f"within_{label}_split",
                                        replicate=replicate,
                                        left=arm_samples[arm, right_mask],
                                        right=arm_samples[arm, left_mask],
                                        denominator=denominator,
                                        valid=valid,
                                        ancestor_fraction=ancestor_fraction,
                                    )
                            # Same-state full SMC replicas.  Adjacent replica pairs
                            # remain explicit if a future run requests >2 replicas.
                            for replicate in range(1, observed_replicates):
                                paired_control_denominator = max(
                                    min(
                                        abs(float(real_gaps[replicate - 1])),
                                        abs(float(real_gaps[replicate])),
                                    ),
                                    min_gap,
                                )
                                for arm, label in ((0, "low_low"), (1, "high_high")):
                                    left = observed[
                                        local_worm,
                                        phase_position,
                                        event,
                                        replicate,
                                        arm,
                                        0,
                                        :,
                                        ti,
                                        si,
                                    ]
                                    right = observed[
                                        local_worm,
                                        phase_position,
                                        event,
                                        replicate - 1,
                                        arm,
                                        0,
                                        :,
                                        ti,
                                        si,
                                    ]
                                    valid = bool(
                                        diagnostic_valid["observed"][
                                            local_worm,
                                            phase_position,
                                            event,
                                            replicate,
                                            si,
                                        ]
                                        and diagnostic_valid["observed"][
                                            local_worm,
                                            phase_position,
                                            event,
                                            replicate - 1,
                                            si,
                                        ]
                                    )
                                    ancestor_fraction = min(
                                        len(
                                            np.unique(
                                                obs_ancestor[
                                                    local_worm,
                                                    phase_position,
                                                    event,
                                                    current,
                                                    arm,
                                                    si,
                                                ]
                                            )
                                        )
                                        / int(manifest["particles"])
                                        for current in (replicate - 1, replicate)
                                    )
                                    _append_metric_rows(
                                        rows,
                                        base,
                                        control=label,
                                        replicate=replicate - 1,
                                        left=left,
                                        right=right,
                                        denominator=paired_control_denominator,
                                        valid=valid,
                                        ancestor_fraction=ancestor_fraction,
                                    )
                            for replicate in range(1, midpoint_replicates):
                                left = midpoint[
                                    local_worm,
                                    phase_position,
                                    event,
                                    replicate,
                                    0,
                                    :,
                                    ti,
                                    si,
                                ]
                                right = midpoint[
                                    local_worm,
                                    phase_position,
                                    event,
                                    replicate - 1,
                                    0,
                                    :,
                                    ti,
                                    si,
                                ]
                                valid = bool(
                                    diagnostic_valid["midpoint"][
                                        local_worm, phase_position, event, replicate, si
                                    ]
                                    and diagnostic_valid["midpoint"][
                                        local_worm,
                                        phase_position,
                                        event,
                                        replicate - 1,
                                        si,
                                    ]
                                )
                                ancestor_fraction = min(
                                    len(
                                        np.unique(
                                            mid_ancestor[
                                                local_worm,
                                                phase_position,
                                                event,
                                                current,
                                                si,
                                            ]
                                        )
                                    )
                                    / int(manifest["particles"])
                                    for current in (replicate - 1, replicate)
                                )
                                _append_metric_rows(
                                    rows,
                                    base,
                                    control="midpoint_midpoint",
                                    replicate=replicate - 1,
                                    left=left,
                                    right=right,
                                    denominator=control_denominator,
                                    valid=valid,
                                    ancestor_fraction=ancestor_fraction,
                                )
                            for replicate in range(pseudo_replicates):
                                arm_samples = pseudo[
                                    local_worm,
                                    phase_position,
                                    event,
                                    replicate,
                                    :,
                                    0,
                                    :,
                                    ti,
                                    si,
                                ]
                                pseudo_gap = float(
                                    data["pseudo_diagnostic_achieved_gap"][
                                        local_worm,
                                        phase_position,
                                        event,
                                        replicate,
                                        si,
                                    ]
                                )
                                pseudo_denominator = max(abs(pseudo_gap), min_gap)
                                valid = bool(
                                    diagnostic_valid["pseudo"][
                                        local_worm,
                                        phase_position,
                                        event,
                                        replicate,
                                        si,
                                    ]
                                )
                                ancestor_fraction = min(
                                    len(
                                        np.unique(
                                            pseudo_ancestor[
                                                local_worm,
                                                phase_position,
                                                event,
                                                replicate,
                                                arm,
                                                si,
                                            ]
                                        )
                                    )
                                    / int(manifest["particles"])
                                    for arm in range(2)
                                )
                                _append_metric_rows(
                                    rows,
                                    base,
                                    control="quiet_pseudo",
                                    replicate=replicate,
                                    left=arm_samples[1],
                                    right=arm_samples[0],
                                    denominator=pseudo_denominator,
                                    valid=valid,
                                    ancestor_fraction=ancestor_fraction,
                                )
    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame.raw_value).all() or not np.isfinite(
        frame.normalized_value
    ).all():
        raise RuntimeError("sampling-null event extraction is empty or nonfinite")
    return frame


def hashlib_sha256_int(*parts: str) -> int:
    import hashlib

    payload = "|".join(parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31 - 1)


def _contextualize_candidate(frame: pd.DataFrame, cell: SelectedCell) -> pd.DataFrame:
    x = frame.loc[frame.candidate_id == cell.candidate_id].copy()
    keys = [
        "candidate_id",
        "queue_rank",
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
        "context",
        "selected_channel",
        "checkpoint_seed",
        "worm_id",
        "control",
        "control_replicate",
        "metric",
    ]
    context = cell.context
    chemical = next(
        (name for name in STIMULUS_NAMES if context.startswith(name + "_")), None
    )
    if chemical is not None:
        x = x.loc[x.chemical == chemical].copy()
    if context == "baseline":
        x = x.loc[x.phase == "baseline"]
        result = x.groupby(keys, as_index=False).agg(
            normalized_value=("normalized_value", "mean"),
            raw_value=("raw_value", "mean"),
            valid=("valid", "min"),
            minimum_ancestor_fraction=("minimum_ancestor_fraction", "min"),
        )
    elif context.endswith("_onset") or context == "onset":
        x = x.loc[x.phase == "onset"]
        result = x.groupby(keys, as_index=False).agg(
            normalized_value=("normalized_value", "mean"),
            raw_value=("raw_value", "mean"),
            valid=("valid", "min"),
            minimum_ancestor_fraction=("minimum_ancestor_fraction", "min"),
        )
    elif context == "onset_minus_baseline" or context.endswith(
        "_onset_minus_baseline"
    ):
        pair_keys = [*keys, "event"]
        pivot = x.pivot_table(
            index=pair_keys,
            columns="phase",
            values=[
                "normalized_value",
                "raw_value",
                "valid",
                "minimum_ancestor_fraction",
            ],
            aggfunc="first",
        )
        if not {"baseline", "onset"}.issubset(pivot.columns.get_level_values(1)):
            raise RuntimeError("context contrast lacks baseline or onset values")
        contrast = pd.DataFrame(index=pivot.index)
        contrast["normalized_value"] = (
            pivot["normalized_value", "onset"]
            - pivot["normalized_value", "baseline"]
        )
        contrast["raw_value"] = (
            pivot["raw_value", "onset"] - pivot["raw_value", "baseline"]
        )
        contrast["valid"] = np.minimum(
            pivot["valid", "onset"], pivot["valid", "baseline"]
        )
        contrast["minimum_ancestor_fraction"] = np.minimum(
            pivot["minimum_ancestor_fraction", "onset"],
            pivot["minimum_ancestor_fraction", "baseline"],
        )
        contrast = contrast.reset_index()
        result = contrast.groupby(keys, as_index=False).agg(
            normalized_value=("normalized_value", "mean"),
            raw_value=("raw_value", "mean"),
            valid=("valid", "min"),
            minimum_ancestor_fraction=("minimum_ancestor_fraction", "min"),
        )
    else:
        raise RuntimeError(f"unsupported context {context!r}")
    if result.empty or not np.isfinite(result.normalized_value).all():
        raise RuntimeError(f"context reduction failed for {cell.candidate_id}")
    return result


def _bootstrap_interval(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("sign-flip values must be a finite vector")
    observed = abs(float(values.mean()))
    # Fix the first sign positive because absolute statistics identify global
    # sign complements.  The 17-worm cohort therefore has 65,536 exact patterns.
    if len(values) <= 18:
        exceed = 0
        total = 0
        for bits in itertools.product((-1.0, 1.0), repeat=len(values) - 1):
            signs = np.asarray((1.0, *bits), dtype=np.float64)
            exceed += abs(float(np.mean(values * signs))) >= observed - 1e-15
            total += 1
        return float(exceed / total)
    rng = np.random.default_rng(20260830)
    signs = rng.choice((-1.0, 1.0), size=(65536, len(values)))
    return float(np.mean(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))


def _joint_studentized_max_t(
    values: np.ndarray,
    *,
    alpha: float = 0.05,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Exact shared-worm max-T p-values for a targeted diagnostic panel.

    ``values`` is ``[test, worm]`` and must be complete.  One sign is reused
    for a worm across every selected candidate, metric, and diagnostic
    contrast.  Fixing the first sign positive enumerates the unique two-sided
    sign orbits.  The returned adjusted p-values therefore control one
    single-step family over the entire supplied panel.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2:
        raise ValueError("max-T values must have shape [test,worm]")
    if not np.isfinite(x).all():
        raise ValueError("max-T values must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    n_tests, n_worms = x.shape
    signs, mode = generate_sign_patterns(n_worms, mode="exact")
    if mode != "exact":
        raise RuntimeError("targeted max-T unexpectedly resolved non-exact mode")
    count = float(n_worms)
    sumsq = np.square(x).sum(axis=1)
    rms = np.sqrt(sumsq / count)
    se_floor = np.maximum(rms * 1e-12, np.finfo(np.float64).tiny)

    def t_from_sum(total: np.ndarray) -> np.ndarray:
        mean = total / count
        centered_ss = np.maximum(sumsq[None, :] - count * np.square(mean), 0.0)
        variance = centered_ss / (count - 1.0)
        se = np.sqrt(variance / count)
        return mean / np.maximum(se, se_floor[None, :])

    observed_t = t_from_sum(x.sum(axis=1)[None, :])[0]
    observed_abs = np.abs(observed_t)
    exceed = np.zeros(n_tests, dtype=np.int64)
    null_max = np.empty(len(signs), dtype=np.float64)
    for start in range(0, len(signs), batch_size):
        stop = min(start + batch_size, len(signs))
        signed_sum = signs[start:stop].astype(np.float64, copy=False) @ x.T
        null_t = np.abs(t_from_sum(signed_sum))
        local_max = null_t.max(axis=1)
        null_max[start:stop] = local_max
        exceed += np.sum(local_max[:, None] >= observed_abs[None, :], axis=0)
    # The all-positive observed orbit is present.  Protect against a few ulps
    # of separately evaluated BLAS arithmetic so no exact p-value becomes 0.
    adjusted = np.maximum(exceed, 1) / float(len(signs))
    try:
        critical = float(np.quantile(null_max, 1.0 - alpha, method="higher"))
    except TypeError:
        critical = float(
            np.quantile(null_max, 1.0 - alpha, interpolation="higher")
        )
    return adjusted, observed_t, null_max, critical, signs


def _bh(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(tuple(values), dtype=np.float64)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result


def _summarize(
    contextual: pd.DataFrame,
    cells: tuple[SelectedCell, ...],
    worm_order: tuple[str, ...],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    summaries: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    array_ids: list[str] = []
    observed_arrays: list[np.ndarray] = []
    mc_arrays: list[np.ndarray] = []
    pseudo_arrays: list[np.ndarray] = []
    named_control_arrays: list[np.ndarray] = []
    for cell in cells:
        current = contextual.loc[contextual.candidate_id == cell.candidate_id]
        obs_support = current.loc[current.control == "observed"]
        sampling_support = current.loc[
            current.control.isin(["low_low", "high_high", "midpoint_midpoint"])
        ]
        pseudo_support = current.loc[current.control == "quiet_pseudo"]
        observed_valid_fraction = float(obs_support.valid.mean())
        sampling_valid_fraction = float(
            sampling_support.groupby("control").valid.mean().min()
        )
        pseudo_valid_fraction = float(pseudo_support.valid.mean())
        observed_genealogy_valid_fraction = float(
            (
                obs_support.valid.astype(bool)
                & (
                    obs_support.minimum_ancestor_fraction
                    >= config.genealogy_fraction_threshold
                )
            ).mean()
        )
        sampling_genealogy_valid_fraction = float(
            sampling_support.assign(
                genealogy_valid=(
                    sampling_support.valid.astype(bool)
                    & (
                        sampling_support.minimum_ancestor_fraction
                        >= config.genealogy_fraction_threshold
                    )
                )
            )
            .groupby("control")
            .genealogy_valid.mean()
            .min()
        )
        pseudo_genealogy_valid_fraction = float(
            (
                pseudo_support.valid.astype(bool)
                & (
                    pseudo_support.minimum_ancestor_fraction
                    >= config.genealogy_fraction_threshold
                )
            ).mean()
        )
        support_rows.append(
            {
                **cell.to_dict(),
                "observed_valid_fraction": observed_valid_fraction,
                "observed_minimum_ancestor_fraction": float(
                    obs_support.minimum_ancestor_fraction.min()
                ),
                "observed_genealogy_valid_fraction_0_10": (
                    observed_genealogy_valid_fraction
                ),
                "sampling_valid_fraction": sampling_valid_fraction,
                "sampling_minimum_ancestor_fraction": float(
                    sampling_support.minimum_ancestor_fraction.min()
                ),
                "sampling_genealogy_valid_fraction_0_10": (
                    sampling_genealogy_valid_fraction
                ),
                "pseudo_valid_fraction": pseudo_valid_fraction,
                "pseudo_minimum_ancestor_fraction": float(
                    pseudo_support.minimum_ancestor_fraction.min()
                ),
                "pseudo_genealogy_valid_fraction_0_10": (
                    pseudo_genealogy_valid_fraction
                ),
            }
        )
        for metric in METRICS:
            metric_frame = current.loc[current.metric == metric]
            observed = (
                metric_frame.loc[metric_frame.control == "observed"]
                .groupby(["checkpoint_seed", "worm_id"], as_index=False)
                .normalized_value.mean()
                .groupby("worm_id")
                .normalized_value.mean()
                .reindex(worm_order)
            )
            null_controls = metric_frame.loc[
                metric_frame.control.isin(
                    ["low_low", "high_high", "midpoint_midpoint"]
                )
            ].copy()
            null_controls["magnitude"] = null_controls.normalized_value.abs()
            # Reduce nuisance draws within a named control and checkpoint seed,
            # then average checkpoint seeds within worm before comparing named
            # controls.  The envelope is the worst named control for each worm;
            # averaging the three controls could hide one unstable arm.
            named_controls = (
                null_controls.groupby(
                    ["checkpoint_seed", "worm_id", "control"], as_index=False
                )
                .magnitude.mean()
                .groupby(["worm_id", "control"])
                .magnitude.mean()
                .unstack("control")
                .reindex(worm_order)
                .reindex(
                    columns=["low_low", "high_high", "midpoint_midpoint"]
                )
            )
            if named_controls.isna().any().any():
                raise RuntimeError("a named sampling control lacks worm coverage")
            mc = named_controls.max(axis=1)
            pseudo_values = metric_frame.loc[
                metric_frame.control == "quiet_pseudo"
            ].copy()
            pseudo_values["magnitude"] = pseudo_values.normalized_value.abs()
            pseudo = (
                pseudo_values.groupby(["checkpoint_seed", "worm_id"], as_index=False)
                .magnitude.mean()
                .groupby("worm_id")
                .magnitude.mean()
                .reindex(worm_order)
            )
            if observed.isna().any() or mc.isna().any() or pseudo.isna().any():
                raise RuntimeError("a candidate/metric lacks complete worm coverage")
            obs = observed.to_numpy(dtype=np.float64)
            mc_value = mc.to_numpy(dtype=np.float64)
            pseudo_value = pseudo.to_numpy(dtype=np.float64)
            context_signed = metric != "endpoint_wasserstein1" or cell.context.endswith(
                "minus_baseline"
            )
            obs_magnitude = np.abs(obs) if context_signed else obs
            mc_excess = obs_magnitude - mc_value
            pseudo_excess = obs_magnitude - pseudo_value
            key_seed = hashlib_sha256_int(cell.candidate_id, metric, str(config.random_seed))
            effect_ci = _bootstrap_interval(obs, config.bootstrap_replicates, key_seed)
            mc_ci = _bootstrap_interval(
                mc_excess, config.bootstrap_replicates, key_seed + 1
            )
            pseudo_ci = _bootstrap_interval(
                pseudo_excess, config.bootstrap_replicates, key_seed + 2
            )
            summaries.append(
                {
                    **cell.to_dict(),
                    "metric": metric,
                    "context_signed": context_signed,
                    "n_worms": len(obs),
                    "observed_mean": float(obs.mean()),
                    "observed_ci_2_5": effect_ci[0],
                    "observed_ci_97_5": effect_ci[1],
                    "observed_sign_flip_p": _sign_flip_p(obs)
                    if context_signed
                    else np.nan,
                    "sampling_null_mean_magnitude": float(mc_value.mean()),
                    "sampling_null_p95_worm": float(np.quantile(mc_value, 0.95)),
                    "low_low_mean_magnitude": float(named_controls.low_low.mean()),
                    "high_high_mean_magnitude": float(
                        named_controls.high_high.mean()
                    ),
                    "midpoint_midpoint_mean_magnitude": float(
                        named_controls.midpoint_midpoint.mean()
                    ),
                    "sampling_excess_mean": float(mc_excess.mean()),
                    "sampling_excess_ci_2_5": mc_ci[0],
                    "sampling_excess_ci_97_5": mc_ci[1],
                    "sampling_excess_sign_flip_p": _sign_flip_p(mc_excess),
                    "quiet_pseudo_mean_magnitude": float(pseudo_value.mean()),
                    "quiet_pseudo_p95_worm": float(np.quantile(pseudo_value, 0.95)),
                    "temporal_specificity_excess_mean": float(pseudo_excess.mean()),
                    "temporal_specificity_ci_2_5": pseudo_ci[0],
                    "temporal_specificity_ci_97_5": pseudo_ci[1],
                    "temporal_specificity_sign_flip_p": _sign_flip_p(pseudo_excess),
                    "observed_to_sampling_null_ratio": float(
                        obs_magnitude.mean() / max(mc_value.mean(), 1e-12)
                    ),
                    "observed_to_quiet_pseudo_ratio": float(
                        obs_magnitude.mean() / max(pseudo_value.mean(), 1e-12)
                    ),
                    "observed_valid_fraction": observed_valid_fraction,
                    "observed_genealogy_valid_fraction_0_10": (
                        observed_genealogy_valid_fraction
                    ),
                    "sampling_valid_fraction": sampling_valid_fraction,
                    "sampling_genealogy_valid_fraction_0_10": (
                        sampling_genealogy_valid_fraction
                    ),
                    "pseudo_valid_fraction": pseudo_valid_fraction,
                    "pseudo_genealogy_valid_fraction_0_10": (
                        pseudo_genealogy_valid_fraction
                    ),
                    "evidence_label": "pending_joint_max_t_calibration",
                    "inference_unit": "worm; checkpoint seeds averaged within worm",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
            array_ids.append(f"{cell.candidate_id}:{metric}")
            observed_arrays.append(obs)
            mc_arrays.append(mc_value)
            pseudo_arrays.append(pseudo_value)
            named_control_arrays.append(
                named_controls.to_numpy(dtype=np.float64).T
            )
    summary = pd.DataFrame(summaries)
    for p_column, q_column in (
        ("observed_sign_flip_p", "observed_bh_q"),
        ("sampling_excess_sign_flip_p", "sampling_excess_bh_q"),
        (
            "temporal_specificity_sign_flip_p",
            "temporal_specificity_bh_q",
        ),
    ):
        finite = summary[p_column].notna()
        summary[q_column] = np.nan
        summary.loc[finite, q_column] = _bh(summary.loc[finite, p_column])

    observed_matrix = np.asarray(observed_arrays, dtype=np.float64)
    sampling_matrix = np.asarray(mc_arrays, dtype=np.float64)
    pseudo_matrix = np.asarray(pseudo_arrays, dtype=np.float64)
    signed_mask = summary.context_signed.to_numpy(dtype=bool)
    observed_magnitude_matrix = np.where(
        signed_mask[:, None], np.abs(observed_matrix), observed_matrix
    )
    sampling_excess_matrix = observed_magnitude_matrix - sampling_matrix
    pseudo_excess_matrix = observed_magnitude_matrix - pseudo_matrix
    joint_values: list[np.ndarray] = []
    joint_kinds: list[str] = []
    joint_rows: list[int] = []
    for row_index in range(len(summary)):
        if signed_mask[row_index]:
            joint_values.append(observed_matrix[row_index])
            joint_kinds.append("observed_signed")
            joint_rows.append(row_index)
        joint_values.append(sampling_excess_matrix[row_index])
        joint_kinds.append("sampling_excess")
        joint_rows.append(row_index)
        joint_values.append(pseudo_excess_matrix[row_index])
        joint_kinds.append("temporal_specificity_excess")
        joint_rows.append(row_index)
    (
        joint_p,
        joint_t,
        joint_null_max,
        joint_critical,
        joint_signs,
    ) = _joint_studentized_max_t(
        np.asarray(joint_values, dtype=np.float64), alpha=config.alpha
    )
    for prefix in (
        "observed",
        "sampling_excess",
        "temporal_specificity",
    ):
        summary[f"{prefix}_joint_max_t_p"] = np.nan
        summary[f"{prefix}_joint_student_t"] = np.nan
    for test_index, (kind, row_index) in enumerate(zip(joint_kinds, joint_rows)):
        prefix = {
            "observed_signed": "observed",
            "sampling_excess": "sampling_excess",
            "temporal_specificity_excess": "temporal_specificity",
        }[kind]
        summary.loc[row_index, f"{prefix}_joint_max_t_p"] = joint_p[test_index]
        summary.loc[row_index, f"{prefix}_joint_student_t"] = joint_t[test_index]

    labels: list[str] = []
    support_pass_values: list[bool] = []
    selection_eligible_values: list[bool] = []
    gate_reasons: list[str] = []
    for row in summary.itertuples(index=False):
        support_pass = all(
            value >= config.strong_valid_fraction
            for value in (
                row.observed_valid_fraction,
                row.observed_genealogy_valid_fraction_0_10,
                row.sampling_valid_fraction,
                row.sampling_genealogy_valid_fraction_0_10,
                row.pseudo_valid_fraction,
                row.pseudo_genealogy_valid_fraction_0_10,
            )
        )
        selection_eligible = row.selection_origin != "lag_sensitivity"
        support_pass_values.append(bool(support_pass))
        selection_eligible_values.append(bool(selection_eligible))
        if not selection_eligible and not support_pass:
            gate_reasons.append("sensitivity_origin_and_support_failure")
        elif not selection_eligible:
            gate_reasons.append("sensitivity_origin")
        elif not support_pass:
            gate_reasons.append("support_failure")
        else:
            gate_reasons.append("none")
        if not selection_eligible or not support_pass:
            labels.append("sampling_limited")
            continue
        observed_pass = (not row.context_signed) or (
            row.observed_joint_max_t_p <= config.alpha
            and (row.observed_ci_2_5 > 0 or row.observed_ci_97_5 < 0)
        )
        sampling_pass = (
            row.sampling_excess_mean > 0
            and row.sampling_excess_ci_2_5 > 0
            and row.sampling_excess_joint_max_t_p <= config.alpha
        )
        pseudo_pass = (
            row.temporal_specificity_excess_mean > 0
            and row.temporal_specificity_ci_2_5 > 0
            and row.temporal_specificity_joint_max_t_p <= config.alpha
        )
        if observed_pass and sampling_pass and pseudo_pass:
            labels.append("exceeds_sampling_and_quiet_controls")
        elif observed_pass and sampling_pass:
            labels.append("exceeds_sampling_controls_only")
        else:
            labels.append("indistinguishable_from_sampling_controls")
    summary["support_pass"] = support_pass_values
    summary["selection_eligible"] = selection_eligible_values
    summary["gate_reason"] = gate_reasons
    summary["evidence_label"] = labels
    arrays = {
        "candidate_metric_id": np.asarray(array_ids),
        "worm_ids": np.asarray(worm_order),
        "observed_worm_value": observed_matrix.astype(np.float32),
        "sampling_null_worm_magnitude": sampling_matrix.astype(np.float32),
        "sampling_control_names": np.asarray(
            ["low_low", "high_high", "midpoint_midpoint"]
        ),
        "sampling_control_worm_magnitude": np.asarray(
            named_control_arrays, dtype=np.float32
        ),
        "quiet_pseudo_worm_magnitude": pseudo_matrix.astype(np.float32),
        "joint_test_id": np.asarray(
            [f"{array_ids[row_index]}:{kind}" for kind, row_index in zip(joint_kinds, joint_rows)]
        ),
        "joint_test_kind": np.asarray(joint_kinds),
        "joint_test_row_index": np.asarray(joint_rows, dtype=np.int16),
        "joint_test_observed_t": joint_t.astype(np.float32),
        "joint_test_max_t_p_value": joint_p.astype(np.float32),
        "joint_null_max_abs_t": joint_null_max.astype(np.float32),
        "joint_sign_patterns": joint_signs.astype(np.int8),
        "joint_simultaneous_critical_value": np.asarray(joint_critical),
    }
    return summary, pd.DataFrame(support_rows), arrays


def _pseudo_diagnostics(archives: tuple[Path, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for archive in archives:
        with np.load(archive, allow_pickle=False) as data:
            phases = data["phase_names"].astype(str)
            for local_worm, worm in enumerate(data["worm_ids"].astype(str)):
                for event in range(3):
                    for replicate in range(
                        int(data["quiet_pseudo_boundary_replicates"].item())
                    ):
                        for phase_position, phase in enumerate(phases):
                            rows.append(
                                {
                                    "selection_fingerprint": str(
                                        data["selection_fingerprint"].item()
                                    ),
                                    "fold": int(data["fold"].item()),
                                    "checkpoint_seed": int(data["seed"].item()),
                                    "worm_id": worm,
                                    "event": event,
                                    "phase": phase,
                                    "pseudo_replicate": replicate,
                                    "quiet_start_frame": int(
                                        data["pseudo_quiet_intervals"][
                                            local_worm, event, 0
                                        ]
                                    ),
                                    "quiet_stop_frame": int(
                                        data["pseudo_quiet_intervals"][
                                            local_worm, event, 1
                                        ]
                                    ),
                                    "pseudo_boundary_frame": int(
                                        data["pseudo_boundary_times"][
                                            local_worm, event, replicate
                                        ]
                                    ),
                                    "pseudo_cut_frame": int(
                                        data["pseudo_cut_times"][
                                            local_worm,
                                            phase_position,
                                            event,
                                            replicate,
                                        ]
                                    ),
                                    "quiet_verified": bool(
                                        data["pseudo_quiet_verified"][
                                            local_worm,
                                            phase_position,
                                            event,
                                            replicate,
                                        ]
                                    ),
                                }
                            )
    return pd.DataFrame(rows)


def _input_checksums(
    run_dir: Path,
    manifest: Mapping[str, Any],
    archives: tuple[Path, ...],
) -> pd.DataFrame:
    paths = {
        run_dir / "manifest.json",
        run_dir / "validation.json",
        run_dir / "run_status.csv",
        run_dir / "checksums.sha256",
        Path(str(manifest["hypothesis_queue"])).resolve(),
        Path(str(manifest["fold_assignments"])).resolve(),
        Path(str(manifest["source_run"])).resolve() / "manifest.json",
        *archives,
    }
    for archive in archives:
        with np.load(archive, allow_pickle=False) as data:
            paths.add(Path(str(data["checkpoint"].item())).resolve())
    return pd.DataFrame(
        [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(paths)
        ]
    )


def analyze(
    run_dir: Path,
    output: Path,
    *,
    config: AnalysisConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    config.validate()
    (
        run_dir,
        raw_manifest,
        raw_validation,
        raw_ledger,
        cohort,
        _folds,
        cells,
        groups,
        archives,
    ) = _load_and_validate_run(run_dir)
    output = output.resolve()
    owned = (
        "event_control_cells.csv",
        "targeted_null_cells.csv",
        "sham_calibration.csv",
        "support_diagnostics.csv",
        "pseudo_boundary_diagnostics.csv",
        "null_inference_arrays.npz",
        "summary.json",
        "protocol.json",
        "input_checksums.csv",
        "validation.json",
        "REPORT.md",
        "manifest.json",
        "checksums.sha256",
    )
    present = [output / name for name in owned if (output / name).exists()]
    if present and not overwrite:
        raise FileExistsError(
            "sampling-null analysis outputs already exist; pass --overwrite to replace"
        )
    output.mkdir(parents=True, exist_ok=True)
    event_rows = _extract_event_rows(
        raw_manifest, cells, groups, archives, min_gap=config.min_gap
    )
    contextual = pd.concat(
        [_contextualize_candidate(event_rows, cell) for cell in cells],
        ignore_index=True,
    )
    requested_folds = {int(value) for value in raw_manifest["folds"]}
    worm_order = tuple(
        worm
        for worm, assigned in zip(cohort.worm_ids, _folds)
        if int(assigned) in requested_folds
    )
    summary, support, inference_arrays = _summarize(
        contextual, cells, worm_order, config
    )
    pseudo = _pseudo_diagnostics(archives)
    inputs = _input_checksums(run_dir, raw_manifest, archives)
    module_root = Path(__file__).resolve().parent
    implementation_paths = {
        "targeted_sampling_null_analysis.py": Path(__file__).resolve(),
        "full_family_inference.py": module_root / "full_family_inference.py",
        "prediction_atlas_runner.py": module_root / "prediction_atlas_runner.py",
        "targeted_sampling_nulls.py": module_root / "targeted_sampling_nulls.py",
        "core.py": module_root / "core.py",
        "conditional_neural_benchmark/data.py": (
            module_root.parent / "conditional_neural_benchmark" / "data.py"
        ),
    }
    analysis_implementation = {
        "files": {
            name: sha256(path) for name, path in implementation_paths.items()
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    raw_implementation = raw_manifest.get(
        "implementation_provenance",
        {
            "status": "not_recorded_by_v1_raw_manifest",
            "boundary": (
                "the raw v1 bundle pins its queue, checkpoints, source-run "
                "manifest, fold assignments, run specification, and every archive, "
                "but did not record source-file or runtime-version hashes"
            ),
        },
    )
    _atomic_csv(output / "event_control_cells.csv", event_rows)
    _atomic_csv(output / "targeted_null_cells.csv", contextual)
    _atomic_csv(output / "sham_calibration.csv", summary)
    _atomic_csv(output / "support_diagnostics.csv", support)
    _atomic_csv(output / "pseudo_boundary_diagnostics.csv", pseudo)
    _atomic_npz(output / "null_inference_arrays.npz", **inference_arrays)
    _atomic_csv(output / "input_checksums.csv", inputs)
    protocol = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "raw_run": str(run_dir),
        "raw_run_spec_fingerprint": raw_manifest["run_spec_fingerprint"],
        "raw_manifest_sha256": sha256(run_dir / "manifest.json"),
        "raw_validation_sha256": sha256(run_dir / "validation.json"),
        "raw_checksums_sha256": sha256(run_dir / "checksums.sha256"),
        "raw_implementation_provenance": raw_implementation,
        "analysis_implementation_provenance": analysis_implementation,
        "selected_cells": [cell.to_dict() for cell in cells],
        "metrics": list(METRICS),
        "control_families": list(CONTROL_FAMILIES),
        "normalization": (
            f"event value / max(abs(achieved source gap), {config.min_gap}); "
            "each observed high-low replica uses its own achieved gap; within-arm "
            "splits use that replica's gap; paired same-arm controls use the smaller "
            "of their two real-repair gaps and midpoint controls use the smallest "
            "matched real-repair gap, conservatively enlarging null magnitudes"
        ),
        "within_arm_split": (
            "future siblings with one terminal repair parent remain in one half; "
            "distinct terminal parents may still share an earlier repair ancestor, "
            "so this split is descriptive Monte Carlo diagnostics only and never an "
            "inferential replication unit"
        ),
        "context_reduction": (
            "events reduced within worm; chemical labels select each worm's observed "
            "event; onset-minus-baseline is formed within event and worm"
        ),
        "checkpoint_seed_reduction": "checkpoint seeds averaged within worm",
        "inference_unit": "worm",
        "bootstrap_replicates": config.bootstrap_replicates,
        "sign_flip": (
            "exact two-sided worm sign flips for <=18 worms, first sign fixed because "
            "absolute statistics identify global complements"
        ),
        "sampling_control_envelope": (
            "within each worm and named control, average nuisance replicates within "
            "checkpoint seed, then average checkpoint seeds; compare the observed "
            "magnitude with the maximum of low-low, high-high, and midpoint-midpoint"
        ),
        "multiplicity": (
            "primary labels require one exact shared-worm single-step max-T family "
            "over every eligible signed-observed, sampling-excess, and quiet-pseudo-"
            "excess candidate x metric test; per-contrast BH q-values are retained "
            "as exploratory diagnostics only"
        ),
        "alpha": config.alpha,
        "support_gate": {
            "valid_fraction_threshold": config.strong_valid_fraction,
            "genealogy_fraction_definition": (
                "joint fraction of diagnostically valid rows whose minimum "
                "distinct-repair-ancestor fraction across arms is at least "
                f"{config.genealogy_fraction_threshold:g}; this matches the "
                "canonical genealogy_valid_fraction_0_10 construction"
            ),
            "genealogy_valid_fraction_threshold": config.strong_valid_fraction,
            "contexts_required": ["observed", "sampling_controls", "quiet_pseudo"],
            "sampling_controls_required": [
                "low_low",
                "high_high",
                "midpoint_midpoint",
            ],
            "sampling_control_fraction_reduction": (
                "calculate validity and joint validity-plus-genealogy fractions "
                "within each named control, then gate on the minimum across "
                "low-low, high-high, and midpoint-midpoint"
            ),
            "lag_sensitivity_rows_forced_sampling_limited": True,
        },
        "selection_status": (
            "selection-conditioned robustness diagnostic; all eight cells were "
            "selected upstream and this is not independent confirmation"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _atomic_json(output / "protocol.json", protocol)
    counts = summary.evidence_label.value_counts().to_dict()
    summary_json = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_cells": len(cells),
        "candidate_metric_rows": len(summary),
        "worms": len(worm_order),
        "checkpoint_seeds": len(raw_manifest["seeds"]),
        "joint_max_t_tests": int(len(inference_arrays["joint_test_id"])),
        "joint_max_t_sign_patterns": int(
            len(inference_arrays["joint_sign_patterns"])
        ),
        "joint_max_t_critical_value": float(
            inference_arrays["joint_simultaneous_critical_value"]
        ),
        "alpha": config.alpha,
        "labels": {str(key): int(value) for key, value in counts.items()},
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _atomic_json(output / "summary.json", summary_json)
    report_lines = [
        "# Targeted progressive-SMC sampling-null analysis",
        "",
        f"- {len(cells)} exact selected cells; {len(worm_order)} worms; "
        f"checkpoint seeds {raw_manifest['seeds']}.",
        f"- {raw_manifest['observed_sampler_replicates']} independent observed "
        f"repair/future replicas, {raw_manifest['midpoint_sampler_replicates']} "
        f"midpoint replicas, and {raw_manifest['quiet_pseudo_boundary_replicates']} "
        "quiet pseudo-boundaries per matched event.",
        "- Particles and within-arm splits diagnose Monte Carlo behavior. All "
        "intervals and sign-flip tests use worms as the replication unit.",
        "- Primary labels use one exact shared-worm single-step max-T correction "
        "across every selected candidate, metric, and declared contrast.",
        "- The sampling null is the per-worm maximum of the low-low, high-high, "
        "and midpoint-midpoint controls after checkpoint-seed averaging.",
        "- A non-limited label requires observed, low-low, high-high, midpoint-"
        "midpoint, and quiet-pseudo validity plus genealogy-valid fractions of "
        f"at least {config.strong_valid_fraction:g}; "
        "the two lower-support lag-sensitivity rows remain sampling-limited by "
        "construction.",
        "",
        "## Evidence labels",
        "",
    ]
    report_lines.extend(f"- `{key}`: {value}" for key, value in sorted(counts.items()))
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            CLAIM_BOUNDARY + ".",
            "",
            "Quiet pseudo-boundaries keep the observed stimulus input at zero over "
            "the complete model-history, repair, and generated-future window. They "
            "measure event-relative temporal specificity among quiet histories; they "
            "do not test stimulus modulation and are not same-onset experimental nulls.",
            "",
        ]
    )
    _atomic_text(output / "REPORT.md", "\n".join(report_lines))
    analysis_spec = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "raw_run_spec_fingerprint": raw_manifest["run_spec_fingerprint"],
        "raw_implementation_provenance": raw_implementation,
        "analysis_implementation_provenance": analysis_implementation,
        "config": {
            "bootstrap_replicates": config.bootstrap_replicates,
            "min_gap": config.min_gap,
            "random_seed": config.random_seed,
            "strong_valid_fraction": config.strong_valid_fraction,
            "genealogy_fraction_threshold": config.genealogy_fraction_threshold,
            "alpha": config.alpha,
        },
        "outputs": list(owned),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    analysis_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_fingerprint": canonical_fingerprint(analysis_spec),
        **analysis_spec,
    }
    _atomic_json(output / "manifest.json", analysis_manifest)
    signed_mask = summary.context_signed.to_numpy(dtype=bool)
    p_null_pattern_valid = bool(
        np.array_equal(
            summary.observed_joint_max_t_p.notna().to_numpy(dtype=bool),
            signed_mask,
        )
        and summary.sampling_excess_joint_max_t_p.notna().all()
        and summary.temporal_specificity_joint_max_t_p.notna().all()
    )
    expected_joint_kinds: list[str] = []
    expected_joint_rows: list[int] = []
    expected_joint_p: list[float] = []
    for row_index, row in enumerate(summary.itertuples(index=False)):
        if bool(row.context_signed):
            expected_joint_kinds.append("observed_signed")
            expected_joint_rows.append(row_index)
            expected_joint_p.append(float(row.observed_joint_max_t_p))
        expected_joint_kinds.append("sampling_excess")
        expected_joint_rows.append(row_index)
        expected_joint_p.append(float(row.sampling_excess_joint_max_t_p))
        expected_joint_kinds.append("temporal_specificity_excess")
        expected_joint_rows.append(row_index)
        expected_joint_p.append(float(row.temporal_specificity_joint_max_t_p))
    expected_joint_ids = np.asarray(
        [
            f"{inference_arrays['candidate_metric_id'][row_index]}:{kind}"
            for kind, row_index in zip(expected_joint_kinds, expected_joint_rows)
        ]
    )
    joint_grid_complete = bool(
        len(summary) == len(cells) * len(METRICS)
        and int(signed_mask.sum()) == 16
        and len(expected_joint_kinds) == 64
        and np.array_equal(
            inference_arrays["joint_test_kind"].astype(str),
            np.asarray(expected_joint_kinds),
        )
        and np.array_equal(
            inference_arrays["joint_test_row_index"],
            np.asarray(expected_joint_rows, dtype=np.int16),
        )
        and np.array_equal(
            inference_arrays["joint_test_id"].astype(str), expected_joint_ids.astype(str)
        )
        and np.allclose(
            inference_arrays["joint_test_max_t_p_value"],
            np.asarray(expected_joint_p),
            rtol=1e-6,
            atol=1e-8,
        )
    )
    gate_reason_complete = bool(
        summary.support_pass.isin([True, False]).all()
        and summary.selection_eligible.isin([True, False]).all()
        and summary.gate_reason.isin(
            [
                "none",
                "sensitivity_origin",
                "support_failure",
                "sensitivity_origin_and_support_failure",
            ]
        ).all()
        and (
            (summary.gate_reason == "none")
            == (summary.support_pass & summary.selection_eligible)
        ).all()
        and (
            (summary.evidence_label == "sampling_limited")
            == (summary.gate_reason != "none")
        ).all()
    )
    bounded_joint_p_values = np.concatenate(
        [
            summary.loc[signed_mask, "observed_joint_max_t_p"].to_numpy(
                dtype=np.float64
            ),
            summary.sampling_excess_joint_max_t_p.to_numpy(dtype=np.float64),
            summary.temporal_specificity_joint_max_t_p.to_numpy(dtype=np.float64),
        ]
    )
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "analysis_fingerprint": analysis_manifest["analysis_fingerprint"],
        "raw_grid_complete": True,
        "raw_checksums_verified": len(raw_ledger),
        "archive_count": len(archives),
        "worm_inference_only": True,
        "particles_not_treated_as_worms": True,
        "quiet_pseudo_windows_all_verified": bool(pseudo.quiet_verified.all()),
        "finite_event_rows": bool(
            np.isfinite(event_rows.raw_value).all()
            and np.isfinite(event_rows.normalized_value).all()
        ),
        "wasserstein_nonnegative_before_context_contrast": bool(
            (
                event_rows.loc[
                    event_rows.metric == "endpoint_wasserstein1", "raw_value"
                ]
                >= -1e-7
            ).all()
        ),
        "external_reference_inputs": 0,
        "joint_max_t_p_values_bounded": bool(
            np.isfinite(bounded_joint_p_values).all()
            and (
                bounded_joint_p_values
                >= 1.0 / len(inference_arrays["joint_sign_patterns"])
            ).all()
            and (bounded_joint_p_values <= 1.0).all()
        ),
        "joint_max_t_p_value_null_pattern_valid": p_null_pattern_valid,
        "joint_max_t_test_grid_complete": joint_grid_complete,
        "gate_reason_complete": gate_reason_complete,
        "no_pending_evidence_labels": bool(
            ~summary.evidence_label.str.startswith("pending").any()
        ),
        "sampling_limited_gate_enforced": bool(
            (
                summary.loc[
                    summary.evidence_label != "sampling_limited",
                    [
                        "observed_valid_fraction",
                        "observed_genealogy_valid_fraction_0_10",
                        "sampling_valid_fraction",
                        "sampling_genealogy_valid_fraction_0_10",
                        "pseudo_valid_fraction",
                        "pseudo_genealogy_valid_fraction_0_10",
                    ],
                ]
                >= config.strong_valid_fraction
            )
            .all()
            .all()
        ),
        "sampling_controls_support_gated": bool(
            (
                summary.loc[
                    summary.evidence_label != "sampling_limited",
                    [
                        "sampling_valid_fraction",
                        "sampling_genealogy_valid_fraction_0_10",
                    ],
                ]
                >= config.strong_valid_fraction
            )
            .all()
            .all()
        ),
        "lag_sensitivity_rows_sampling_limited": bool(
            (
                summary.loc[
                    summary.selection_origin == "lag_sensitivity",
                    "evidence_label",
                ]
                == "sampling_limited"
            ).all()
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    required_validation = (
        "worm_inference_only",
        "particles_not_treated_as_worms",
        "quiet_pseudo_windows_all_verified",
        "finite_event_rows",
        "wasserstein_nonnegative_before_context_contrast",
        "joint_max_t_p_values_bounded",
        "joint_max_t_p_value_null_pattern_valid",
        "joint_max_t_test_grid_complete",
        "gate_reason_complete",
        "no_pending_evidence_labels",
        "sampling_limited_gate_enforced",
        "sampling_controls_support_gated",
        "lag_sensitivity_rows_sampling_limited",
    )
    if not all(validation[name] is True for name in required_validation):
        failed = [name for name in required_validation if validation[name] is not True]
        raise RuntimeError(f"sampling-null analysis validation failed: {failed}")
    _atomic_json(output / "validation.json", validation)
    write_checksums(output)
    verify_checksums(output)
    return summary_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--min-gap", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=20260830)
    parser.add_argument("--strong-valid-fraction", type=float, default=0.80)
    parser.add_argument("--genealogy-fraction-threshold", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    analyze(
        args.run_dir,
        args.output_dir,
        config=AnalysisConfig(
            bootstrap_replicates=args.bootstrap_replicates,
            min_gap=args.min_gap,
            random_seed=args.random_seed,
            strong_valid_fraction=args.strong_valid_fraction,
            genealogy_fraction_threshold=args.genealogy_fraction_threshold,
            alpha=args.alpha,
        ),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
