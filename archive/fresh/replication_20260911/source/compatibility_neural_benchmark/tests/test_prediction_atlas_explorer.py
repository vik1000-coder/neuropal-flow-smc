from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.prediction_atlas_explorer import (
    COMPLETE_FAMILY_REQUIRED,
    ENCODING,
    MANIFEST_SCHEMA_VERSION,
    SAMPLING_NULL_CLAIM_BOUNDARY,
    SAMPLING_NULL_CONTROLS,
    SAMPLING_NULL_METRICS,
    SAMPLING_NULL_REQUIRED,
    SAMPLING_NULL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ReportInputError,
    _quantize_slice,
    build_prediction_atlas_explorer,
)
from compatibility_neural_benchmark.prediction_atlas_runner import canonical_fingerprint
from compatibility_neural_benchmark.prediction_atlas_report import (
    ATLAS_REQUIRED,
    TARGETED_REQUIRED,
    sha256,
)
from compatibility_neural_benchmark.targeted_confirmation_analysis import ORIENTATION
from compatibility_neural_benchmark.targeted_sampling_null_analysis import (
    _bootstrap_interval as _sampling_bootstrap_interval,
    _joint_studentized_max_t as _sampling_joint_studentized_max_t,
    hashlib_sha256_int as _sampling_bootstrap_seed,
)
from compatibility_neural_benchmark.tests.test_prediction_atlas_report import (
    _write_atlas,
    _write_external,
    _write_ledger,
    _write_targeted,
)


LAGS = (1, 4, 8, 16)
HORIZONS = (1, 2, 4, 8, 16, 32)


def _dense_value(
    method_index: int,
    context_index: int,
    lag_index: int,
    horizon_index: int,
    target_index: int,
    source_index: int,
) -> float:
    return (
        method_index * 0.05
        + context_index * 0.01
        + lag_index * 0.002
        + horizon_index * 0.0003
        + target_index * 0.01
        - source_index * 0.009
    )


def _write_dense_atlas(
    root: Path,
    *,
    omit_one_dense_array: bool = False,
    declare_composition: bool = True,
) -> tuple[Path, tuple[str, ...]]:
    atlas, neurons = _write_atlas(root)
    dashboard_path = atlas / "dashboard_snapshot.json"
    dashboard = json.loads(dashboard_path.read_text())
    queue = pd.read_csv(atlas / "hypothesis_queue.csv")
    queue["genealogy_gate_applicable"] = True
    queue["genealogy_valid_fraction_0_10"] = 0.82
    queue["genealogy_valid_fraction_0_20"] = 0.91
    queue["genealogy_strong_gate_pass"] = True
    queue["genealogy_sensitivity_gate_pass"] = True
    queue["minimum_distinct_ancestor_fraction_low"] = 0.16
    queue["minimum_distinct_ancestor_fraction_high"] = 0.18
    queue.to_csv(atlas / "hypothesis_queue.csv", index=False)
    queue_contexts = set(queue["context"].astype(str))
    dashboard["channels"] = ["endpoint_mean"]
    dashboard["contexts"] = [
        item for item in dashboard["contexts"] if item["context"] in queue_contexts
    ]
    contexts = tuple(str(item["context"]) for item in dashboard["contexts"])
    dashboard_path.write_text(json.dumps(dashboard, indent=2, sort_keys=True) + "\n")

    methods = tuple(str(value) for value in dashboard["methods"])
    channels = tuple(str(value) for value in dashboard["channels"])
    dense: dict[str, np.ndarray] = {
        "orientation": np.asarray(ORIENTATION),
        "neurons": np.asarray(neurons),
        "methods": np.asarray(methods),
        "channels": np.asarray(channels),
        "contexts": np.asarray(contexts),
        "source_lag_frames": np.asarray(LAGS, dtype=np.int16),
        "horizon_frames": np.asarray(HORIZONS, dtype=np.int16),
        "primary_method": np.asarray("progressive_bridge_smc"),
    }
    target = np.arange(54, dtype=np.float32)[:, None]
    source = np.arange(54, dtype=np.float32)[None, :]
    for method_index, method in enumerate(methods):
        for channel in channels:
            for context_index, context in enumerate(contexts):
                key = f"mean_normalized__{method}__{channel}__{context}"
                if omit_one_dense_array and method_index == 0 and context_index == 0:
                    continue
                array = np.empty((len(LAGS), len(HORIZONS), 54, 54), dtype=np.float32)
                for lag_index in range(len(LAGS)):
                    for horizon_index in range(len(HORIZONS)):
                        array[lag_index, horizon_index] = (
                            method_index * 0.05
                            + context_index * 0.01
                            + lag_index * 0.002
                            + horizon_index * 0.0003
                            + target * 0.01
                            - source * 0.009
                        )
                dense[key] = array
    np.savez_compressed(atlas / "atlas_matrices.npz", **dense)
    np.savez_compressed(
        atlas / "worm_matrices.npz",
        worm_ids=np.asarray([f"worm-{index:02d}" for index in range(17)]),
    )

    manifest_path = atlas / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ledger_names = [name for name in ATLAS_REQUIRED if name != "checksums.sha256"]
    dashboard["stimulus_composition_summary"] = [
        {
            "phase": phase,
            "source_lag_frames": lag,
            "horizon_frames": horizon,
            "n_worm_events": 51,
            "source_window_stimulus_fraction_min": (
                0.0 if phase == "baseline" else 0.5
            ),
            "source_window_stimulus_fraction_mean": (
                0.0 if phase == "baseline" else 0.75
            ),
            "source_window_stimulus_fraction_max": (
                0.0 if phase == "baseline" else 1.0
            ),
            "forecast_window_stimulus_fraction_min": (
                0.0 if phase == "baseline" else 0.6
            ),
            "forecast_window_stimulus_fraction_mean": (
                0.0 if phase == "baseline" else 0.85
            ),
            "forecast_window_stimulus_fraction_max": (
                0.0 if phase == "baseline" else 1.0
            ),
            "forecast_endpoint_stimulus_fraction_mean": (
                0.0 if phase == "baseline" else 0.9
            ),
            "source_window_crosses_stimulus_boundary_any": phase == "active",
            "forecast_window_crosses_stimulus_boundary_any": phase == "active",
            "cut_to_endpoint_stimulus_transition_any": False,
        }
            for phase in ("baseline", "onset", "active", "offset", "recovery")
        for lag in LAGS
        for horizon in HORIZONS
    ]
    dashboard_path.write_text(json.dumps(dashboard, indent=2, sort_keys=True) + "\n")
    if declare_composition:
        manifest["artifacts"] = {
            "atlas_matrices": "atlas_matrices.npz",
            "stimulus_composition": "stimulus_composition.csv",
        }
    else:
        manifest.pop("artifacts", None)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_ledger(atlas, tuple([*ledger_names, "worm_matrices.npz"]))
    return atlas, neurons


def _write_complete_family_evidence(
    root: Path,
    atlas: Path,
    neurons: tuple[str, ...],
) -> Path:
    evidence = root / "complete_family_evidence"
    evidence.mkdir()
    queue = pd.read_csv(atlas / "hypothesis_queue.csv")
    candidate = queue.loc[
        (queue["method"] == "progressive_bridge_smc")
        & (queue["channel"] == "endpoint_mean")
        & (queue["context"] == "baseline")
    ].iloc[0]
    families = [
        {
            "family_id": "baseline_endpoint_mean",
            "channel": "endpoint_mean",
            "context": "baseline",
            "complete_cells": 68_688,
            "complete_edges": 2_862,
            "support_eligible_cells": 35_616,
            "support_eligible_edges": 1_484,
            "joint_primary_edge_max_t_discoveries": 102,
            "joint_primary_cell_max_t_discoveries": 492,
            "joint_primary_flat_lag_edge_discoveries": 0,
            "joint_primary_lag_contrast_cell_discoveries": 0,
            "strong_edge_lag_unresolved": 102,
            "experiment_ready": 0,
            "n_worms": 17,
            "sign_patterns": 65_536,
            "practical_status": "pending_sham_and_experimental_threshold",
        },
        {
            "family_id": "baseline_endpoint_log_sd",
            "channel": "endpoint_log_sd",
            "context": "baseline",
            "complete_cells": 68_688,
            "complete_edges": 2_862,
            "support_eligible_cells": 35_616,
            "support_eligible_edges": 1_484,
            "joint_primary_edge_max_t_discoveries": 0,
            "joint_primary_cell_max_t_discoveries": 0,
            "joint_primary_flat_lag_edge_discoveries": 0,
            "joint_primary_lag_contrast_cell_discoveries": 0,
            "strong_edge_lag_unresolved": 0,
            "experiment_ready": 0,
            "n_worms": 17,
            "sign_patterns": 65_536,
            "practical_status": "pending_sham_and_experimental_threshold",
        },
        {
            "family_id": "active_minus_baseline_endpoint_mean",
            "channel": "endpoint_mean",
            "context": "active_minus_baseline",
            "complete_cells": 57_240,
            "complete_edges": 2_862,
            "support_eligible_cells": 11_660,
            "support_eligible_edges": 583,
            "joint_primary_edge_max_t_discoveries": 0,
            "joint_primary_cell_max_t_discoveries": 0,
            "joint_primary_flat_lag_edge_discoveries": 0,
            "joint_primary_lag_contrast_cell_discoveries": 0,
            "strong_edge_lag_unresolved": 0,
            "experiment_ready": 0,
            "n_worms": 17,
            "sign_patterns": 65_536,
            "practical_status": "pending_sham_and_experimental_threshold",
        },
        {
            "family_id": "active_minus_baseline_endpoint_log_sd",
            "channel": "endpoint_log_sd",
            "context": "active_minus_baseline",
            "complete_cells": 57_240,
            "complete_edges": 2_862,
            "support_eligible_cells": 11_660,
            "support_eligible_edges": 583,
            "joint_primary_edge_max_t_discoveries": 0,
            "joint_primary_cell_max_t_discoveries": 0,
            "joint_primary_flat_lag_edge_discoveries": 0,
            "joint_primary_lag_contrast_cell_discoveries": 0,
            "strong_edge_lag_unresolved": 0,
            "experiment_ready": 0,
            "n_worms": 17,
            "sign_patterns": 65_536,
            "practical_status": "pending_sham_and_experimental_threshold",
        },
    ]
    exact_cell = {
        "family_id": "baseline_endpoint_mean",
        "channel": "endpoint_mean",
        "context": "baseline",
        "source_index": int(candidate.source_index),
        "target_index": int(candidate.target_index),
        "source_neuron": neurons[int(candidate.source_index)],
        "target_neuron": neurons[int(candidate.target_index)],
        "source_lag_frames": int(candidate.source_lag_frames),
        "horizon_frames": int(candidate.horizon_frames),
        "mean_normalized": float(candidate.mean_normalized),
        "joint_primary_cell_max_t_p_value": 1 / 65_536,
        "joint_primary_edge_max_t_p_value": 1 / 65_536,
        "joint_primary_flat_lag_max_t_p_value": 1.0,
        "evidence_label": "strong_edge_lag_unresolved",
        "experiment_ready": False,
        "planned_sham_controls": (
            "independent A/B same-arm comparisons; independent midpoint zero-gap; "
            "matched quiet pseudo-cuts; parent-safe within-arm split"
        ),
        "sesoi_status": "not_defined_pending_sham_calibration",
        "independent_confirmation_status": "not_run",
        "practical_status": "pending_sham_and_experimental_threshold",
    }
    shortlist = [
        {
            **exact_cell,
            "shortlist_rank": rank,
            "source_index": (int(candidate.source_index) + rank - 1) % 54,
            "source_neuron": neurons[(int(candidate.source_index) + rank - 1) % 54],
        }
        for rank in range(1, 7)
    ]
    sensitivity_rows = [
        {
            "family_id": "baseline_endpoint_mean",
            "source_index": source_index,
            "target_index": target_index,
            "source_neuron": neurons[source_index],
            "target_neuron": neurons[target_index],
            "source_lag_frames": 16,
            "horizon_frames": 1,
            "joint_sensitivity_lag_contrast_max_t_p_value": p_value,
            "evidence_label": "sampling_limited",
            "experiment_ready": False,
            "sensitivity_only": True,
            "planned_sham_controls": exact_cell["planned_sham_controls"],
            "practical_status": "pending_sham_and_experimental_threshold",
        }
        for source_index, target_index, p_value in (
            (49, 36, 0.000457763671875),
            (49, 34, 0.094970703125),
        )
    ]
    legacy_match = {
        "family_id": exact_cell["family_id"],
        "source_neuron": exact_cell["source_neuron"],
        "target_neuron": exact_cell["target_neuron"],
        "source_lag_frames": exact_cell["source_lag_frames"],
        "horizon_frames": exact_cell["horizon_frames"],
        "joint_primary_cell_max_t_p_value": exact_cell[
            "joint_primary_cell_max_t_p_value"
        ],
        "joint_primary_edge_max_t_p_value": exact_cell[
            "joint_primary_edge_max_t_p_value"
        ],
        "joint_primary_flat_lag_max_t_p_value": exact_cell[
            "joint_primary_flat_lag_max_t_p_value"
        ],
        "legacy_postscreen_q_value": 0.004,
        "legacy_q_label": "post_screen_legacy_not_full_family_fdr",
        "legacy_queue_rank": float(candidate.queue_rank),
        "evidence_label": "strong_edge_lag_unresolved",
    }
    explorer = {
        "schema_version": "complete-family-explorer-evidence-v1",
        "status": "complete",
        "headline": {
            "joint_primary_families": 4,
            "complete_edge_tests": 11_448,
            "support_eligible_edge_tests": 4_134,
            "joint_primary_edge_discoveries": 102,
            "joint_primary_flat_lag_edge_discoveries": 0,
            "experiment_ready": 0,
            "practical_status": "pending_sham_and_experimental_threshold",
        },
        "evidence_labels": {
            "experiment_ready": "pending gates",
            "strong_edge_lag_unresolved": "joint edge evidence, no lag evidence",
            "strong_edge_with_lag_structure": "joint edge and lag evidence",
            "exploratory": "does not pass joint max-T",
            "sampling_limited": "fails primary support gate",
            "no_complete_family_evidence": "no threshold passes",
        },
        "family_summary": families,
        "top_edges": [
            {
                **exact_cell,
                "joint_edge_rank": 1,
            }
        ],
        "sampling_null_shortlist": shortlist,
        "sensitivity_support05": {
            "available": True,
            "support_gate": "sensitivity_0.5",
            "claim_status": "sensitivity_only_never_promotes_primary_claim",
            "named_lag_rows": sensitivity_rows,
            "experiment_ready": 0,
            "warning": (
                "Every sensitivity-only row remains sampling_limited and cannot "
                "strengthen the primary or experiment-ready claim."
            ),
        },
        "legacy_queue": {
            "matched_cells": 1,
            "warning": (
                "legacy q-values were computed after top-effect retention and are not "
                "complete-family FDR values"
            ),
            "q_value_label": "post_screen_legacy_not_full_family_fdr",
            "matches": [legacy_match],
        },
    }
    protocol = {
        "schema_version": "complete-family-joint-evidence-protocol-v1",
        "joint_primary": {
            "families": [row["family_id"] for row in families],
            "independent_unit": "worm",
            "sign_patterns": 65_536,
            "alpha": 0.05,
        },
    }
    validation = {
        "schema_version": "complete-family-joint-evidence-validation-v1",
        "status": "passed",
        "input_sha256": sha256(atlas / "worm_matrices.npz"),
        **{
            field: True
            for field in (
                "input_checksums_verified",
                "identical_input_sha256",
                "identical_sign_pattern_order",
                "all_families_exact",
                "complete_family_rows",
                "edge_keys_unique_within_family",
                "cell_keys_unique_within_family",
                "diagonal_edges_absent",
                "experiment_ready_count_zero",
                "shortlist_count_six",
                "strong_and_sensitivity_input_sha256_identical",
                "strong_and_sensitivity_sign_order_identical",
                "sensitivity_only_labels_sampling_limited",
                "lag_sensitivity_shortlist_count_two",
                "sampling_null_combined_count_eight",
                "sampling_null_combined_keys_unique",
                "sampling_null_combined_composition_valid",
                "legacy_queue_sha256_pinned",
            )
        },
    }
    manifest = {
        "schema_version": "complete-family-joint-evidence-manifest-v1",
        "status": "complete",
        "artifacts": {
            "family_summary": "family_summary.csv",
            "edge_evidence": "edge_evidence.csv",
            "cell_evidence": "cell_evidence.parquet",
            "sampling_null_shortlist": "sampling_null_shortlist.csv",
            "sensitivity_family_summary": "sensitivity_family_summary.csv",
            "sensitivity_edge_evidence": "sensitivity_edge_evidence.csv",
            "sensitivity_cell_evidence": "sensitivity_cell_evidence.parquet",
            "sampling_null_lag_sensitivity_shortlist": (
                "sampling_null_lag_sensitivity_shortlist.csv"
            ),
            "sampling_null_combined": "sampling_null_combined_8.csv",
            "explorer_evidence": "explorer_evidence.json",
            "protocol": "protocol.json",
            "summary": "summary.json",
            "validation": "validation.json",
            "checksums": "checksums.sha256",
        },
    }
    (evidence / "explorer_evidence.json").write_text(
        json.dumps(explorer, indent=2, sort_keys=True) + "\n"
    )
    (evidence / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    (evidence / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "complete-family-joint-evidence-summary-v1",
                "status": "complete",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (evidence / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    (evidence / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(families).to_csv(evidence / "family_summary.csv", index=False)
    pd.DataFrame(explorer["top_edges"]).to_csv(
        evidence / "edge_evidence.csv", index=False
    )
    pd.DataFrame(shortlist).to_parquet(evidence / "cell_evidence.parquet", index=False)
    pd.DataFrame(shortlist).to_csv(
        evidence / "sampling_null_shortlist.csv", index=False
    )
    pd.DataFrame(families).to_csv(
        evidence / "sensitivity_family_summary.csv", index=False
    )
    pd.DataFrame(sensitivity_rows).to_csv(
        evidence / "sensitivity_edge_evidence.csv", index=False
    )
    pd.DataFrame(sensitivity_rows).to_parquet(
        evidence / "sensitivity_cell_evidence.parquet", index=False
    )
    pd.DataFrame(sensitivity_rows).to_csv(
        evidence / "sampling_null_lag_sensitivity_shortlist.csv", index=False
    )
    combined_rows = []
    for rank, row in enumerate([*shortlist, *sensitivity_rows], 1):
        combined_rows.append(
            {
                "queue_rank": rank,
                "channel": "endpoint_mean",
                "context": "baseline",
                "source_neuron": row["source_neuron"],
                "target_neuron": row["target_neuron"],
                "source_index": row["source_index"],
                "target_index": row["target_index"],
                "source_lag_frames": row["source_lag_frames"],
                "horizon_frames": row["horizon_frames"],
                "selection_origin": (
                    "strong_primary" if rank <= 6 else "lag_sensitivity"
                ),
                "run_sampling_nulls": "true",
            }
        )
    pd.DataFrame(combined_rows).to_csv(
        evidence / "sampling_null_combined_8.csv", index=False
    )
    _write_ledger(
        evidence,
        (
            "family_summary.csv",
            "edge_evidence.csv",
            "cell_evidence.parquet",
            "sampling_null_shortlist.csv",
            "sensitivity_family_summary.csv",
            "sensitivity_edge_evidence.csv",
            "sensitivity_cell_evidence.parquet",
            "sampling_null_lag_sensitivity_shortlist.csv",
            "sampling_null_combined_8.csv",
            "explorer_evidence.json",
            "protocol.json",
            "summary.json",
            "validation.json",
            "manifest.json",
        ),
    )
    return evidence


FIXTURE_BOOTSTRAP_REPLICATES = 200
FIXTURE_BOOTSTRAP_SEED = 20260830


@lru_cache(maxsize=1)
def _coherent_sampling_inference_fixture() -> dict[str, object]:
    """Return one internally coherent 24-row, 64-test exact calibration."""

    worm_ids = np.asarray([f"worm-{index:02d}" for index in range(17)])
    variation = np.linspace(-0.03, 0.03, 17, dtype=np.float64)
    observed_rows: list[np.ndarray] = []
    named_rows: list[np.ndarray] = []
    sampler_rows: list[np.ndarray] = []
    quiet_rows: list[np.ndarray] = []
    row_stats: list[dict[str, float | bool | str | int]] = []
    metric_ids: list[str] = []

    for rank in range(1, 9):
        sampler_pass = rank in {1, 2, 4, 5, 6, 7}
        quiet_pass = rank in {1, 3, 4, 5, 8}
        for metric in SAMPLING_NULL_METRICS:
            candidate_id = f"cell_fixture_{rank:02d}"
            metric_ids.append(f"{candidate_id}:{metric}")
            observed = 0.8 + variation + (rank - 1) * 0.001
            if sampler_pass:
                named = np.stack(
                    (
                        0.10 + variation * 0.05,
                        0.12 + variation * 0.05,
                        0.11 + variation * 0.05,
                    )
                )
            else:
                named = np.stack((observed - 0.02, observed, observed - 0.01))
            sampler = named.max(axis=0)
            quiet = (
                0.15 + variation * 0.05 if quiet_pass else observed.copy()
            )
            context_signed = metric != "endpoint_wasserstein1"
            observed_magnitude = np.abs(observed) if context_signed else observed
            sampler_excess = observed_magnitude - sampler
            quiet_excess = observed_magnitude - quiet
            key_seed = _sampling_bootstrap_seed(
                candidate_id, metric, str(FIXTURE_BOOTSTRAP_SEED)
            )
            observed_ci = _sampling_bootstrap_interval(
                observed, FIXTURE_BOOTSTRAP_REPLICATES, key_seed
            )
            sampler_ci = _sampling_bootstrap_interval(
                sampler_excess, FIXTURE_BOOTSTRAP_REPLICATES, key_seed + 1
            )
            quiet_ci = _sampling_bootstrap_interval(
                quiet_excess, FIXTURE_BOOTSTRAP_REPLICATES, key_seed + 2
            )
            observed_rows.append(observed)
            named_rows.append(named)
            sampler_rows.append(sampler)
            quiet_rows.append(quiet)
            row_stats.append(
                {
                    "rank": rank,
                    "metric": metric,
                    "context_signed": context_signed,
                    "observed_mean": float(observed.mean()),
                    "observed_ci_2_5": observed_ci[0],
                    "observed_ci_97_5": observed_ci[1],
                    "observed_joint_max_t_p": math.nan,
                    "observed_joint_student_t": math.nan,
                    "sampling_null_mean_magnitude": float(sampler.mean()),
                    "sampling_null_p95_worm": float(np.quantile(sampler, 0.95)),
                    "low_low_mean_magnitude": float(named[0].mean()),
                    "high_high_mean_magnitude": float(named[1].mean()),
                    "midpoint_midpoint_mean_magnitude": float(named[2].mean()),
                    "sampling_excess_mean": float(sampler_excess.mean()),
                    "sampling_excess_ci_2_5": sampler_ci[0],
                    "sampling_excess_ci_97_5": sampler_ci[1],
                    "sampling_excess_joint_max_t_p": math.nan,
                    "sampling_excess_joint_student_t": math.nan,
                    "quiet_pseudo_mean_magnitude": float(quiet.mean()),
                    "quiet_pseudo_p95_worm": float(np.quantile(quiet, 0.95)),
                    "temporal_specificity_excess_mean": float(quiet_excess.mean()),
                    "temporal_specificity_ci_2_5": quiet_ci[0],
                    "temporal_specificity_ci_97_5": quiet_ci[1],
                    "temporal_specificity_joint_max_t_p": math.nan,
                    "temporal_specificity_joint_student_t": math.nan,
                    "observed_to_sampling_null_ratio": float(
                        observed_magnitude.mean() / max(sampler.mean(), 1e-12)
                    ),
                    "observed_to_quiet_pseudo_ratio": float(
                        observed_magnitude.mean() / max(quiet.mean(), 1e-12)
                    ),
                }
            )

    observed_matrix = np.asarray(observed_rows, dtype=np.float64)
    named_matrix = np.asarray(named_rows, dtype=np.float64)
    sampler_matrix = np.asarray(sampler_rows, dtype=np.float64)
    quiet_matrix = np.asarray(quiet_rows, dtype=np.float64)
    signed_mask = np.asarray(
        [bool(row["context_signed"]) for row in row_stats], dtype=bool
    )
    observed_magnitude_matrix = np.where(
        signed_mask[:, None], np.abs(observed_matrix), observed_matrix
    )
    sampler_excess_matrix = observed_magnitude_matrix - sampler_matrix
    quiet_excess_matrix = observed_magnitude_matrix - quiet_matrix
    joint_values: list[np.ndarray] = []
    joint_kinds: list[str] = []
    joint_rows: list[int] = []
    joint_ids: list[str] = []
    for row_index, metric_id in enumerate(metric_ids):
        if signed_mask[row_index]:
            joint_values.append(observed_matrix[row_index])
            joint_kinds.append("observed_signed")
            joint_rows.append(row_index)
            joint_ids.append(f"{metric_id}:observed_signed")
        joint_values.extend(
            (sampler_excess_matrix[row_index], quiet_excess_matrix[row_index])
        )
        joint_kinds.extend(("sampling_excess", "temporal_specificity_excess"))
        joint_rows.extend((row_index, row_index))
        joint_ids.extend(
            (
                f"{metric_id}:sampling_excess",
                f"{metric_id}:temporal_specificity_excess",
            )
        )
    joint_p, joint_t, joint_null, joint_critical, signs = (
        _sampling_joint_studentized_max_t(np.asarray(joint_values), alpha=0.05)
    )
    for test_index, (kind, row_index) in enumerate(zip(joint_kinds, joint_rows)):
        p_column = {
            "observed_signed": "observed_joint_max_t_p",
            "sampling_excess": "sampling_excess_joint_max_t_p",
            "temporal_specificity_excess": (
                "temporal_specificity_joint_max_t_p"
            ),
        }[kind]
        t_column = {
            "observed_signed": "observed_joint_student_t",
            "sampling_excess": "sampling_excess_joint_student_t",
            "temporal_specificity_excess": (
                "temporal_specificity_joint_student_t"
            ),
        }[kind]
        row_stats[row_index][p_column] = float(joint_p[test_index])
        row_stats[row_index][t_column] = float(joint_t[test_index])

    return {
        "worm_ids": worm_ids,
        "metric_ids": np.asarray(metric_ids),
        "observed": observed_matrix,
        "named": named_matrix,
        "sampler": sampler_matrix,
        "quiet": quiet_matrix,
        "row_stats": tuple(row_stats),
        "joint_ids": np.asarray(joint_ids),
        "joint_kinds": np.asarray(joint_kinds),
        "joint_rows": np.asarray(joint_rows, dtype=np.int16),
        "joint_p": joint_p,
        "joint_t": joint_t,
        "joint_null": joint_null,
        "joint_critical": joint_critical,
        "signs": signs,
    }


def _write_sampling_null_analysis(
    root: Path,
    atlas: Path,
    neurons: tuple[str, ...],
    complete_family: Path,
) -> Path:
    combined = pd.read_csv(complete_family / "sampling_null_combined_8.csv")
    selected = []
    for row in combined.itertuples(index=False):
        selected.append(
            {
                "candidate_id": f"cell_fixture_{int(row.queue_rank):02d}",
                "queue_rank": int(row.queue_rank),
                "source_index": int(row.source_index),
                "target_index": int(row.target_index),
                "source_neuron": str(row.source_neuron),
                "target_neuron": str(row.target_neuron),
                "source_lag_frames": int(row.source_lag_frames),
                "horizon_frames": int(row.horizon_frames),
                "context": str(row.context),
                "required_phases": ["baseline"],
                "channel": str(row.channel),
                "selection_origin": str(row.selection_origin),
            }
        )

    raw = root / "sampling_null_raw"
    raw.mkdir()
    raw_fingerprint = "a" * 64
    inference = _coherent_sampling_inference_fixture()
    fixture_worm_ids = [str(value) for value in inference["worm_ids"]]
    raw_manifest = {
        "manifest_schema_version": "prediction_atlas_sampling_null_manifest_v1",
        "run_spec_fingerprint": raw_fingerprint,
        "hypothesis_queue": str(
            (complete_family / "sampling_null_combined_8.csv").resolve()
        ),
        "hypothesis_queue_sha256": sha256(
            complete_family / "sampling_null_combined_8.csv"
        ),
        "cohort_worms": fixture_worm_ids,
        "neurons": list(neurons),
        # Preserve the legacy-v1 absence of selection_origin in the raw cells.
        "selected_cells": [
            {key: value for key, value in row.items() if key != "selection_origin"}
            for row in selected
        ],
    }
    raw_validation = {
        "status": "pass",
        "run_spec_fingerprint": raw_fingerprint,
    }
    (raw / "manifest.json").write_text(
        json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n"
    )
    (raw / "validation.json").write_text(
        json.dumps(raw_validation, indent=2, sort_keys=True) + "\n"
    )
    _write_ledger(raw, ("manifest.json", "validation.json"))

    analysis = root / "sampling_null_analysis"
    analysis.mkdir()
    support_rows = []
    calibration_rows = []
    rank_label = {
        1: "exceeds_sampling_and_quiet_controls",
        2: "exceeds_sampling_controls_only",
        3: "indistinguishable_from_sampling_controls",
        4: "sampling_limited",
        5: "exceeds_sampling_and_quiet_controls",
        6: "exceeds_sampling_controls_only",
        7: "sampling_limited",
        8: "sampling_limited",
    }
    stats_by_key = {
        (int(row["rank"]), str(row["metric"])): row
        for row in inference["row_stats"]
    }
    for cell in selected:
        observed_support_fraction = 0.90
        sampling_support_fraction = (
            0.70 if cell["queue_rank"] in {4, 8} else 0.90
        )
        pseudo_support_fraction = 0.90
        support_pass = sampling_support_fraction >= 0.80
        selection_eligible = cell["selection_origin"] != "lag_sensitivity"
        if not selection_eligible and not support_pass:
            gate_reason = "sensitivity_origin_and_support_failure"
        elif not selection_eligible:
            gate_reason = "sensitivity_origin"
        elif not support_pass:
            gate_reason = "support_failure"
        else:
            gate_reason = "none"
        support_rows.append(
            {
                **cell,
                "observed_valid_fraction": observed_support_fraction,
                "observed_minimum_ancestor_fraction": 0.20,
                "observed_genealogy_valid_fraction_0_10": (
                    observed_support_fraction
                ),
                "sampling_valid_fraction": sampling_support_fraction,
                "sampling_minimum_ancestor_fraction": 0.20,
                "sampling_genealogy_valid_fraction_0_10": (
                    sampling_support_fraction
                ),
                "pseudo_valid_fraction": pseudo_support_fraction,
                "pseudo_minimum_ancestor_fraction": 0.20,
                "pseudo_genealogy_valid_fraction_0_10": pseudo_support_fraction,
            }
        )
        label = rank_label[cell["queue_rank"]]
        for metric in SAMPLING_NULL_METRICS:
            stats = stats_by_key[(cell["queue_rank"], metric)]
            calibration_rows.append(
                {
                    **cell,
                    "metric": metric,
                    **{
                        key: value
                        for key, value in stats.items()
                        if key not in {"rank", "metric"}
                    },
                    "n_worms": 17,
                    "observed_valid_fraction": observed_support_fraction,
                    "observed_genealogy_valid_fraction_0_10": (
                        observed_support_fraction
                    ),
                    "sampling_valid_fraction": sampling_support_fraction,
                    "sampling_genealogy_valid_fraction_0_10": (
                        sampling_support_fraction
                    ),
                    "pseudo_valid_fraction": pseudo_support_fraction,
                    "pseudo_genealogy_valid_fraction_0_10": (
                        pseudo_support_fraction
                    ),
                    "support_pass": support_pass,
                    "selection_eligible": selection_eligible,
                    "gate_reason": gate_reason,
                    "evidence_label": label,
                    "claim_boundary": SAMPLING_NULL_CLAIM_BOUNDARY,
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    support = pd.DataFrame(support_rows)
    calibration.to_csv(analysis / "sham_calibration.csv", index=False)
    support.to_csv(analysis / "support_diagnostics.csv", index=False)
    pd.DataFrame(
        [{"quiet_verified": True, "worm_id": "worm-00", "event": 0}]
    ).to_csv(analysis / "pseudo_boundary_diagnostics.csv", index=False)
    pd.DataFrame([{"normalized_value": 0.1}]).to_csv(
        analysis / "event_control_cells.csv", index=False
    )
    pd.DataFrame([{"normalized_value": 0.1}]).to_csv(
        analysis / "targeted_null_cells.csv", index=False
    )

    np.savez_compressed(
        analysis / "null_inference_arrays.npz",
        candidate_metric_id=inference["metric_ids"],
        worm_ids=inference["worm_ids"],
        observed_worm_value=np.asarray(inference["observed"], dtype=np.float32),
        sampling_null_worm_magnitude=np.asarray(
            inference["sampler"], dtype=np.float32
        ),
        sampling_control_names=np.asarray(
            ["low_low", "high_high", "midpoint_midpoint"]
        ),
        sampling_control_worm_magnitude=np.asarray(
            inference["named"], dtype=np.float32
        ),
        quiet_pseudo_worm_magnitude=np.asarray(
            inference["quiet"], dtype=np.float32
        ),
        joint_test_id=inference["joint_ids"],
        joint_test_kind=inference["joint_kinds"],
        joint_test_row_index=inference["joint_rows"],
        joint_test_observed_t=np.asarray(inference["joint_t"], dtype=np.float32),
        joint_test_max_t_p_value=np.asarray(inference["joint_p"], dtype=np.float32),
        joint_null_max_abs_t=np.asarray(inference["joint_null"], dtype=np.float32),
        joint_sign_patterns=inference["signs"],
        joint_simultaneous_critical_value=np.asarray(
            inference["joint_critical"], dtype=np.float32
        ),
    )
    input_rows = [
        {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (
            raw / "manifest.json",
            raw / "validation.json",
            raw / "checksums.sha256",
        )
    ]
    pd.DataFrame(input_rows).to_csv(analysis / "input_checksums.csv", index=False)
    protocol = {
        "analysis_schema_version": SAMPLING_NULL_SCHEMA_VERSION,
        "raw_run": str(raw.resolve()),
        "raw_run_spec_fingerprint": raw_fingerprint,
        "raw_manifest_sha256": sha256(raw / "manifest.json"),
        "raw_validation_sha256": sha256(raw / "validation.json"),
        "raw_checksums_sha256": sha256(raw / "checksums.sha256"),
        "selected_cells": selected,
        "metrics": list(SAMPLING_NULL_METRICS),
        "control_families": list(SAMPLING_NULL_CONTROLS),
        "inference_unit": "worm",
        "checkpoint_seed_reduction": "checkpoint seeds averaged within worm",
        "sampling_control_envelope": (
            "compare with the maximum of low-low, high-high, and "
            "midpoint-midpoint"
        ),
        "multiplicity": "one exact shared-worm single-step max-T family",
        "bootstrap_replicates": FIXTURE_BOOTSTRAP_REPLICATES,
        "alpha": 0.05,
        "support_gate": {
            "valid_fraction_threshold": 0.80,
            "genealogy_valid_fraction_threshold": 0.80,
            "contexts_required": [
                "observed",
                "sampling_controls",
                "quiet_pseudo",
            ],
            "sampling_controls_required": [
                "low_low",
                "high_high",
                "midpoint_midpoint",
            ],
            "sampling_control_fraction_reduction": (
                "calculate validity and genealogy fractions within each named "
                "control, then gate on the minimum across low-low, high-high, "
                "and midpoint-midpoint"
            ),
            "lag_sensitivity_rows_forced_sampling_limited": True,
        },
        "selection_status": "selection-conditioned robustness diagnostic",
        "claim_boundary": SAMPLING_NULL_CLAIM_BOUNDARY,
    }
    (analysis / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    label_counts = {
        str(key): int(value)
        for key, value in calibration.evidence_label.value_counts().items()
    }
    summary = {
        "selected_cells": 8,
        "candidate_metric_rows": 24,
        "worms": 17,
        "checkpoint_seeds": 2,
        "joint_max_t_tests": 64,
        "joint_max_t_sign_patterns": 65_536,
        "joint_max_t_critical_value": float(inference["joint_critical"]),
        "alpha": 0.05,
        "labels": label_counts,
        "claim_boundary": SAMPLING_NULL_CLAIM_BOUNDARY,
    }
    (analysis / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    validation = {
        "status": "pass",
        "analysis_fingerprint": "pending",
        "raw_grid_complete": True,
        "raw_checksums_verified": 3,
        "worm_inference_only": True,
        "particles_not_treated_as_worms": True,
        "quiet_pseudo_windows_all_verified": True,
        "finite_event_rows": True,
        "wasserstein_nonnegative_before_context_contrast": True,
        "external_reference_inputs": 0,
        "joint_max_t_p_values_bounded": True,
        "joint_max_t_p_value_null_pattern_valid": True,
        "joint_max_t_test_grid_complete": True,
        "no_pending_evidence_labels": True,
        "sampling_limited_gate_enforced": True,
        "sampling_controls_support_gated": True,
        "gate_reason_complete": True,
        "lag_sensitivity_rows_sampling_limited": True,
        "claim_boundary": SAMPLING_NULL_CLAIM_BOUNDARY,
    }
    (analysis / "REPORT.md").write_text(
        "# Fixture sampling-null analysis\n\nSelection-conditioned and model-relative.\n"
    )
    analysis_spec = {
        "analysis_schema_version": SAMPLING_NULL_SCHEMA_VERSION,
        "raw_run_spec_fingerprint": raw_fingerprint,
        "config": {
            "alpha": 0.05,
            "bootstrap_replicates": FIXTURE_BOOTSTRAP_REPLICATES,
            "random_seed": FIXTURE_BOOTSTRAP_SEED,
            "strong_valid_fraction": 0.80,
            "genealogy_fraction_threshold": 0.10,
        },
        "outputs": list(SAMPLING_NULL_REQUIRED),
        "claim_boundary": SAMPLING_NULL_CLAIM_BOUNDARY,
    }
    fingerprint = canonical_fingerprint(analysis_spec)
    validation["analysis_fingerprint"] = fingerprint
    (analysis / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "created_utc": "2026-08-30T00:00:00+00:00",
        "analysis_fingerprint": fingerprint,
        **analysis_spec,
    }
    (analysis / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    _write_ledger(
        analysis,
        tuple(name for name in SAMPLING_NULL_REQUIRED if name != "checksums.sha256"),
    )
    return analysis


def _bundle(
    root: Path,
    *,
    omit_one_dense_array: bool = False,
    declare_composition: bool = True,
) -> tuple[Path, Path, Path, Path, Path]:
    atlas, neurons = _write_dense_atlas(
        root,
        omit_one_dense_array=omit_one_dense_array,
        declare_composition=declare_composition,
    )
    targeted = _write_targeted(root, atlas, neurons)
    support_path = targeted / "support_diagnostics.csv"
    support = pd.read_csv(support_path)
    support["genealogy_gate_applicable"] = True
    support["genealogy_valid_fraction_0_10"] = 0.84
    support["genealogy_valid_fraction_0_20"] = 0.93
    support["genealogy_strong_gate_pass"] = True
    support["genealogy_sensitivity_gate_pass"] = True
    support["minimum_distinct_ancestor_fraction_low"] = 0.17
    support["minimum_distinct_ancestor_fraction_high"] = 0.19
    support.to_csv(support_path, index=False)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )
    external = _write_external(root, atlas)
    complete_family = _write_complete_family_evidence(root, atlas, neurons)
    sampling_null = _write_sampling_null_analysis(
        root, atlas, neurons, complete_family
    )
    return atlas, targeted, external, complete_family, sampling_null


def _decode(record: dict[str, object]) -> np.ndarray:
    raw = base64.b64decode(str(record["data"]), validate=True)
    shape = tuple(int(value) for value in record["shape"])
    return np.frombuffer(raw, dtype="<i2").reshape(shape) * float(record["scale"])


def _refresh_sampling_null_ledger(root: Path) -> None:
    _write_ledger(
        root,
        tuple(name for name in SAMPLING_NULL_REQUIRED if name != "checksums.sha256"),
    )


def _rewrite_sampling_null_archive(
    root: Path, mutate: Callable[[dict[str, np.ndarray]], None]
) -> dict[str, np.ndarray]:
    path = root / "null_inference_arrays.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)
    return arrays


def _refingerprint_sampling_null_manifest(root: Path) -> None:
    manifest_path = root / "manifest.json"
    validation_path = root / "validation.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_spec = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_utc", "analysis_fingerprint"}
    }
    fingerprint = canonical_fingerprint(manifest_spec)
    manifest["analysis_fingerprint"] = fingerprint
    validation = json.loads(validation_path.read_text())
    validation["analysis_fingerprint"] = fingerprint
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )


def test_signed_int16_quantization_preserves_target_row_source_column() -> None:
    target = np.arange(54, dtype=float)[:, None]
    source = np.arange(54, dtype=float)[None, :]
    matrix = target * 0.75 - source * 0.125
    record = _quantize_slice(matrix)
    decoded = _decode(record)

    assert record["encoding"] == ENCODING
    assert record["order"] == "C_target_row_source_column"
    assert record["shape"] == [54, 54]
    assert len(base64.b64decode(record["data"])) == 54 * 54 * 2
    assert decoded[7, 13] == pytest.approx(matrix[7, 13], abs=float(record["scale"]) / 2)
    assert decoded.ravel(order="C")[7 * 54 + 13] == decoded[7, 13]
    assert decoded[13, 7] != pytest.approx(decoded[7, 13])
    assert decoded.min() < 0 < decoded.max()


def test_explorer_schema_dense_round_trip_and_fully_offline_html(tmp_path: Path) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    output = tmp_path / "explorer"
    manifest = build_prediction_atlas_explorer(
        atlas,
        targeted,
        external,
        complete_family,
        sampling_null,
        output,
        provenance_root=tmp_path,
    )

    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert manifest["offline"] is True
    assert manifest["network_dependencies"] == []
    assert manifest["optional_stimulus_composition"] == "included"
    assert set(path.name for path in output.iterdir()) == {
        "atlas_explorer.html",
        "explorer_data.json",
        "manifest.json",
        "checksums.sha256",
    }

    payload = json.loads((output / "explorer_data.json").read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["offline"] is True
    assert payload["atlas"]["orientation"] == ORIENTATION
    assert payload["atlas"]["row_axis"] == "target_neuron"
    assert payload["atlas"]["column_axis"] == "source_neuron"
    assert payload["atlas"]["context_options"]
    assert {
        "context",
        "event_stratum",
        "phase_or_contrast",
        "event_stratified",
    }.issubset(payload["atlas"]["context_options"][0])
    assert payload["atlas"]["internal_array_axes"] == [
        "source_lag_frames",
        "horizon_frames",
        "target_neuron",
        "source_neuron",
    ]
    assert "per_slice_scale / 2" in payload["atlas"]["encoding"][
        "maximum_absolute_error"
    ]
    assert payload["atlas"]["slice_count"] == (
        len(payload["atlas"]["methods"])
        * len(payload["atlas"]["channels"])
        * len(payload["atlas"]["contexts"])
        * len(LAGS)
        * len(HORIZONS)
    )
    method = "progressive_bridge_smc"
    method_index = payload["atlas"]["methods"].index(method)
    context = payload["atlas"]["contexts"][1]
    record = payload["atlas"]["slices"][method]["endpoint_mean"][context]["4"]["8"]
    decoded = _decode(record)
    expected = _dense_value(method_index, 1, 1, 3, 7, 13)
    assert decoded[7, 13] == pytest.approx(expected, abs=float(record["scale"]) / 2 + 1e-7)
    assert payload["stimulus_composition"]["status"] == "included"
    assert (
        payload["stimulus_composition"]["representation"]
        == "compact_phase_by_lag_by_horizon_summary"
    )
    assert len(payload["stimulus_composition"]["rows"]) == 5 * len(LAGS) * len(HORIZONS)
    assert payload["stimulus_composition"]["rows"][0]["phase"] == "baseline"
    assert payload["targeted_n128"]["cells"]
    assert "genealogy_valid_fraction_0_10" in payload["candidate_queue"][0]
    assert (
        "genealogy_strong_gate_pass"
        in payload["targeted_n128"]["support_diagnostics"][0]
    )
    assert payload["external_references"]["comparisons"]
    assert payload["statistical_evidence"]["schema_version"] == (
        "complete-family-explorer-evidence-v1"
    )
    assert payload["statistical_evidence"]["headline"] == {
        "joint_primary_families": 4,
        "complete_edge_tests": 11_448,
        "support_eligible_edge_tests": 4_134,
        "joint_primary_edge_discoveries": 102,
        "joint_primary_flat_lag_edge_discoveries": 0,
        "experiment_ready": 0,
        "practical_status": "pending_sham_and_experimental_threshold",
    }
    assert payload["statistical_evidence"]["sensitivity_support05"][
        "claim_status"
    ] == "sensitivity_only_never_promotes_primary_claim"
    assert all(
        row["evidence_label"] == "sampling_limited"
        for row in payload["statistical_evidence"]["sensitivity_support05"][
            "named_lag_rows"
        ]
    )
    assert payload["sampling_null_calibration"]["summary"]["selected_cells"] == 8
    assert payload["sampling_null_calibration"]["summary"][
        "candidate_metric_rows"
    ] == 24
    assert len(payload["sampling_null_calibration"]["cells"]) == 24
    assert {
        row["evidence_label"]
        for row in payload["sampling_null_calibration"]["cells"]
    } == {
        "exceeds_sampling_and_quiet_controls",
        "exceeds_sampling_controls_only",
        "indistinguishable_from_sampling_controls",
        "sampling_limited",
    }
    assert len(payload["sampling_null_calibration"]["support_diagnostics"]) == 8
    assert {
        row["gate_reason"] for row in payload["sampling_null_calibration"]["cells"]
    } == {
        "none",
        "support_failure",
        "sensitivity_origin",
        "sensitivity_origin_and_support_failure",
    }
    assert [row["bundle"] for row in payload["provenance"]] == [
        "canonical_atlas",
        "targeted_n128",
        "postfreeze_external",
        "complete_family_evidence",
        "sampling_null_analysis",
    ]
    boundary_text = " ".join(
        item["label"] + " " + item["detail"] for item in payload["claim_boundaries"]
    ).lower()
    for phrase in (
        "model-relative and noncausal",
        "event-stratified under binary-any-stimulus",
        "lag is not a physical delay",
        "factual arm model rollout",
            "sampling-null calibration is selection-conditioned",
            "raw v1 implementation provenance is incomplete",
            "no prediction is experiment-ready",
        "baseline quiet control is not stimulus modulation",
        "exact calibration p-values require joint sign symmetry",
        "overlapping cross-fit training sets can couple held-out estimates",
        "not experimental randomization significance",
    ):
        assert phrase in boundary_text

    html = (output / "atlas_explorer.html").read_text()
    for control in (
        "method-control",
        "channel-control",
        "context-control",
        "phase-control",
        "lag-control",
        "horizon-control",
        "source-control",
        "target-control",
    ):
        assert f'id="{control}"' in html
    for view in (
        "matrix-view",
        "cell-view",
        "downstream-view",
        "queue-view",
        "targeted-view",
        "composition-view",
        "external-view",
        "provenance-view",
        "statistical-evidence-view",
    ):
        assert f'id="{view}"' in html
    lowered = html.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "<script src=" not in lowered
    assert "<link " not in lowered
    assert "fetch(" not in lowered
    assert "connect-src 'none'" in lowered
    assert 'type="application/json" id="atlas-payload"' in html
    assert "per-slice scale" in html
    assert "colors share one" in lowered
    assert "normalized channel effect" in html
    assert "wasserstein-1 is unsigned within a state" in lowered
    assert "a between-phase difference can be signed" in lowered
    assert '!Boolean(x.signed)' in html
    assert "chemical event-stratum selector does not filter this table" in lowered
    assert "state average shows all five phases" in lowered
    assert "all five required inputs" in lowered
    assert "across 17 worms" in lowered
    assert "65,536 two-sided sign patterns" in lowered
    assert "four composition-matched families" in lowered
    assert "colors show effect magnitude only" in lowered
    assert "sampling reliability" in lowered
    assert "sampling support and particle ancestry" in lowered
    assert "canonical atlas estimate" in lowered
    assert "separate matched n128 rerun" in lowered
    assert "sampler and quiet-time checks use a separate matched n128 rerun" in lowered
    assert "not the canonical atlas estimate above" in lowered
    assert "n128 evidence, where available, is reported separately below" in lowered
    assert "named sampler-control worst-case validity" in lowered
    assert "minimum across low–low, high–high, and midpoint–midpoint" in lowered
    assert "response above sampler noise" in lowered
    assert "quiet-time specificity" in lowered
    assert "biological validation" in lowered
    assert "compare the three calibrated outcomes" in lowered
    assert "this cell is baseline, so this is not evidence of stimulus modulation" in lowered
    assert "selection-conditioned and model-relative" in lowered
    assert "not a biological null" in lowered
    assert "a biological null, causal effect, or physical delay" in lowered
    assert "this exact cell was not among the eight" in lowered
    assert "separate, selection-conditioned 64-test n128 family" in lowered
    assert "intervals are descriptive worm-bootstrap intervals" in lowered
    assert "no signed observed-null p-value is defined" in lowered
    assert "before an experiment" in lowered
    assert "not passing does not mean the learned effect is zero" in lowered
    for label in (
        "Exceeds both controls",
        "Exceeds sampler noise only",
        "Not separated from controls",
        "Support or selection restriction",
        "Support check failed",
        "Sensitivity analysis only",
        "Sensitivity only · support failed",
    ):
        assert label in html
    assert "strong edge; lag unresolved" in lowered
    assert "sampling-limited" in html
    assert "experiment-ready" in html
    assert "legacy screen only" in lowered
    assert "post-screen exploratory result" in lowered
    assert "legacy post-screen q value" in lowered
    assert "sensitivity-only evidence cannot strengthen" in lowered
    assert "generic significant" not in lowered
    # The clearer layout must still allow internally scrollable tables and long
    # status labels to shrink at the 390 px QA viewport.
    assert ".card{min-width:0" in html
    assert ".card-head{display:flex;flex-wrap:wrap" in html
    assert (
        ".datum{padding:13px;background:#22282c;border-radius:4px;"
        "overflow-wrap:anywhere}"
    ) in html
    assert ".pill{display:inline-flex;max-width:100%" in html
    assert "white-space:normal;overflow-wrap:anywhere" in html
    assert ".controls{position:sticky" not in html

    # Concise labels replace the tutorial-like header; detailed definitions and
    # assumptions remain available in collapsed native disclosures.
    for phrase in (
        "Neural prediction atlas",
        "Connection and timing",
        "Candidate predictions",
        "Reference comparisons",
        "Methods &amp; data",
        "Predicted difference",
        "Sampling method &amp; control definitions",
        "Show technical screening fields",
        "Show all external comparison rows",
        "A history coordinate is not a physical transmission delay",
        "it is not an extra conditioning input",
    ):
        assert phrase in html
    markup = html.split('<script type="application/json" id="atlas-payload">', 1)[0]
    assert markup.count('id="calibration-picks"') == 1
    for label, control in (
        ("Source neuron", "source-control"),
        ("Target neuron", "target-control"),
        ("Outcome", "channel-control"),
        ("Stimulus period", "phase-control"),
        ("Event subset", "context-control"),
        ("History lag", "lag-control"),
        ("Forecast horizon", "horizon-control"),
    ):
        assert f'<label>{label}<select id="{control}"' in markup
    assert "radial-gradient" not in markup
    assert "tab-number" not in markup
    assert "rung-number" not in markup
    assert "What this can show" not in markup
    assert "Start with one biological question" not in markup
    # Gradients remain meaningful in the quantitative color legend only.
    css = markup.split("<style>", 1)[1].split("</style>", 1)[0]
    gradient_rules = [rule for rule in css.split("}") if "gradient(" in rule]
    assert len(gradient_rules) == 2
    assert all(rule.lstrip().startswith(".legend-bar") for rule in gradient_rules)
    control_definitions = markup.split('<details class="advanced">', 1)[1].split(
        "</details>", 1
    )[0]
    for phrase in (
        "Source → target",
        "History lag / forecast horizon",
        "Stimulus period / event subset",
        "before the prediction cut",
        "it is not an extra conditioning input",
    ):
        assert phrase in control_definitions
    inference_definitions = markup.split(
        '<details class="secondary-details" id="inference-definitions">', 1
    )[1].split("</details>", 1)[0]
    for phrase in (
        "65,536 two-sided sign patterns across 17 worms",
        "four composition-matched families",
        "joint worm-vector sign symmetry",
        "Overlapping cross-fit training sets can couple held-out estimates",
        "not experimental randomization significance",
        "does not correct upstream selection",
        "not proof of zero effect",
        "quiet-time specificity is not stimulus modulation",
    ):
        assert phrase in inference_definitions
    rung_code = html.split("function evidenceRung(", 1)[1].split(
        "function intervalText", 1
    )[0]
    assert '<details data-evidence-check="${e(number)}">' in rung_code
    assert "<summary>" in rung_code
    assert '<p class="rung-detail">${e(detail)}</p></details>' in rung_code
    assert " open" not in rung_code
    evidence_code = html.split("function renderStatisticalEvidence()", 1)[1].split(
        "function renderMatrix()", 1
    )[0]
    # All alternatives render through one of exactly five expandable check
    # types; no test or caveat was deleted to make the page shorter.
    check_calls = set(re.findall(r'evidenceRung\((\d),"([^"]+)"', evidence_code))
    assert check_calls == {
        ("1", "Sampling reliability"),
        ("2", "Effect and lag evidence"),
        ("3", "Response above sampler noise"),
        ("4", "Quiet-time specificity"),
        ("5", "Biological validation"),
    }
    assert "All six support and ancestry checks pass" in evidence_code
    assert "Atlas cell p=${pText(cellP)} · lag-difference p=${pText(lagP)}" in evidence_code
    assert "N128 effect ${f(calibration.observed_mean,4)}, 95% interval" in evidence_code
    assert "worms analysed" in evidence_code
    assert "worms, the independent units" not in evidence_code
    for human_label in (
        "Endpoint activity",
        "Peak activity",
        "Endpoint variability",
        "Whole-distribution difference (Wasserstein-1)",
        "Progressive bridge SMC (primary)",
        "Direct importance sampling (comparison)",
    ):
        assert human_label in html
    for panel in ("explore-panel", "candidates-panel", "references-panel", "methods-panel"):
        assert f'id="{panel}"' in html
    assert 'id="result-sentence" aria-live="polite"' in html
    assert 'aria-describedby="matrix-readout matrix-scale-note"' in html
    assert 'aria-pressed="${selected}"' in html

    # The first strict-support sampler calibration initializes the complete atlas
    # state. Manual edits still detach candidate-specific evidence instead of
    # silently showing stale evidence.
    assert "const firstCandidate=payload.candidate_queue.length" in html
    assert "firstCalibration=[...(payload.sampling_null_calibration.cells||[])]" in html
    assert "source:initialSelection?Number(initialSelection.source_index):0" in html
    assert "target:initialSelection?Number(initialSelection.target_index):0" in html
    assert "queueRank:firstCalibration?null:firstCandidate" in html
    assert 'id="calibration-picks"' in html
    assert "function renderCalibrationPicks()" in html
    assert "8 fixed selections; not a ranking" in html
    assert "function markCustom(){state.queueRank=null;}" in html
    assert "||payload.candidate_queue[0]" not in html
    assert "Select a candidate to see its frozen record and N128 follow-up" in html
    current_evidence_code = html.split("function currentEvidence()", 1)[1].split(
        "function evidenceForCandidate", 1
    )[0]
    assert "queueRank" not in current_evidence_code
    matrix_code = html.split("function renderMatrix()", 1)[1].split(
        "function renderSurface", 1
    )[0]
    assert "statistical_evidence" not in matrix_code

    # The default candidate list is bounded to the N128-reviewed subset; the
    # full 1,000-row screen remains available only on explicit request.
    assert "reviewedRanks.has(Number(item.x.queue_rank))" in html
    assert "queueExpanded?indexed" in html
    assert "Show all ${indexed.length} screening candidates" in html

    # Signed and distributional metrics receive different plain-language
    # interpretations, including the special within-state W1 direction rule.
    assert "This distance is unsigned" in html
    assert "This sign compares distances; it is not a direction of transport" in html
    assert "is ${highlight} normalized units <strong>${direction}</strong>" in html
    assert (
        'channel==="endpoint_wasserstein1"&&!String(context).endsWith("minus_baseline")'
        in html
    )
    assert "auroc 0.5 is chance" in lowered
    assert "source-preserving max-lag null" in lowered
    assert "this queue row was not one of the frozen n128 candidates" in lowered
    assert "data.cells.slice" not in html
    assert "data.quantile_shifts.slice" not in html
    assert "if(!support.length)support=data.support_diagnostics" not in html
    assert "if(!screen.length)screen=data.screen_consistency" not in html
    assert 'String(x.context)===String(selectedCandidate.context)' in html
    assert 'Number(selectedCandidate.source_index)' in html
    assert 'Number(selectedCandidate.source_lag_frames)' in html
    assert 'state.context==="state_average"' not in html
    for targeted_table in (
        "targeted-quantile-table",
        "targeted-support-table",
        "targeted-screen-table",
    ):
        assert f'id="{targeted_table}"' in html

    for name, expected_hash in (
        (
            manifest["outputs"]["html"]["file"],
            manifest["outputs"]["html"]["sha256"],
        ),
        (
            manifest["outputs"]["payload"]["file"],
            manifest["outputs"]["payload"]["sha256"],
        ),
    ):
        assert sha256(output / name) == expected_hash


def test_explorer_fails_closed_on_missing_dense_inventory(tmp_path: Path) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(
        tmp_path, omit_one_dense_array=True
    )
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="dense normalized-mean inventory"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


def test_explorer_fails_closed_before_output_on_checksum_tamper(tmp_path: Path) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    with (atlas / "atlas_matrices.npz").open("ab") as handle:
        handle.write(b"tampered")
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="checksum verification failed"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


def test_explorer_fails_closed_on_complete_family_evidence_tamper(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    with (complete_family / "explorer_evidence.json").open("a") as handle:
        handle.write("tampered")
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="checksum verification failed"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("shortlist_timing", "shortlist timing mapping"),
        ("legacy_queue", "does not rejoin the frozen queue"),
        ("shortlist_cell_p_floor", "shortlist cell p-value lies outside"),
        ("top_edge_p_floor", "top_edges flat-lag p-value lies outside"),
        ("top_edge_label", "top_edges label/p-value contract"),
        ("legacy_lag_label", "legacy label/p-value contract"),
        ("sensitivity_p_floor", "sensitivity evidence key failed"),
        ("false_experiment_ready", "readiness gates failed"),
    ),
)
def test_complete_family_evidence_semantic_mapping_is_fail_closed(
    tmp_path: Path, kind: str, message: str
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    explorer_path = complete_family / "explorer_evidence.json"
    evidence = json.loads(explorer_path.read_text())
    if kind == "shortlist_timing":
        evidence["sampling_null_shortlist"][0]["source_lag_frames"] = 3
    elif kind == "legacy_queue":
        evidence["legacy_queue"]["matches"][0]["source_lag_frames"] = 16
    elif kind == "shortlist_cell_p_floor":
        evidence["sampling_null_shortlist"][0][
            "joint_primary_cell_max_t_p_value"
        ] = 0.0
    elif kind == "top_edge_p_floor":
        evidence["top_edges"][0][
            "joint_primary_flat_lag_max_t_p_value"
        ] = 0.0
    elif kind == "top_edge_label":
        evidence["top_edges"][0]["joint_primary_edge_max_t_p_value"] = 1.0
    elif kind == "legacy_lag_label":
        evidence["legacy_queue"]["matches"][0][
            "joint_primary_flat_lag_max_t_p_value"
        ] = 1.0 / 65_536
    elif kind == "sensitivity_p_floor":
        evidence["sensitivity_support05"]["named_lag_rows"][0][
            "joint_sensitivity_lag_contrast_max_t_p_value"
        ] = 0.0
    else:
        evidence["sampling_null_shortlist"][0]["experiment_ready"] = True
    explorer_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    _write_ledger(
        complete_family,
        tuple(name for name in COMPLETE_FAMILY_REQUIRED if name != "checksums.sha256"),
    )

    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match=message):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


def test_explorer_refuses_untracked_optional_composition(tmp_path: Path) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(
        tmp_path, declare_composition=False
    )
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="manifest.*checksum"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


def test_explorer_refuses_own_existing_outputs_but_preserves_gui_siblings(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    output = tmp_path / "explorer"
    output.mkdir()
    (output / "artifact.json").write_text("portable dashboard sibling\n")
    (output / "dashboard.html").write_text("portable dashboard sibling\n")
    build_prediction_atlas_explorer(
        atlas,
        targeted,
        external,
        complete_family,
        sampling_null,
        output,
        provenance_root=tmp_path,
    )
    payload_hash = sha256(output / "explorer_data.json")

    with pytest.raises(FileExistsError, match="already exist"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    (output / "user-notes.txt").write_text("keep me\n")
    build_prediction_atlas_explorer(
        atlas,
        targeted,
        external,
        complete_family,
        sampling_null,
        output,
        provenance_root=tmp_path,
        overwrite=True,
    )
    assert (output / "user-notes.txt").read_text() == "keep me\n"
    assert (output / "artifact.json").read_text() == "portable dashboard sibling\n"
    assert (output / "dashboard.html").read_text() == "portable dashboard sibling\n"
    assert sha256(output / "explorer_data.json") != payload_hash


def test_explorer_refuses_output_inside_validated_input_bundle(tmp_path: Path) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    nested_output = atlas / "gui"
    with pytest.raises(ReportInputError, match="must not equal or be nested"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            nested_output,
            provenance_root=tmp_path,
        )
    assert not nested_output.exists()


def test_explorer_validates_all_owned_paths_before_overwrite(tmp_path: Path) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    output = tmp_path / "explorer"
    build_prediction_atlas_explorer(
        atlas,
        targeted,
        external,
        complete_family,
        sampling_null,
        output,
        provenance_root=tmp_path,
    )
    payload_hash = sha256(output / "explorer_data.json")
    html_path = output / "atlas_explorer.html"
    html_path.unlink()
    html_path.mkdir()

    with pytest.raises(ReportInputError, match="not a regular file"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
            overwrite=True,
        )

    assert sha256(output / "explorer_data.json") == payload_hash


def test_sampling_calibration_labels_reproduce_after_checksum_verification(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    calibration_path = sampling_null / "sham_calibration.csv"
    calibration = pd.read_csv(calibration_path)
    selected = (
        (calibration.queue_rank == 1)
        & (calibration.metric == "endpoint_mean")
    )
    assert calibration.loc[selected, "evidence_label"].item() == (
        "exceeds_sampling_and_quiet_controls"
    )
    # Use another allowed label and refresh the ledger.  The explorer must
    # reject the scientific mismatch, not merely accept a valid checksum.
    calibration.loc[selected, "evidence_label"] = "exceeds_sampling_controls_only"
    calibration.to_csv(calibration_path, index=False)
    _write_ledger(
        sampling_null,
        tuple(name for name in SAMPLING_NULL_REQUIRED if name != "checksums.sha256"),
    )
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="label does not reproduce"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


def test_sampling_calibration_raw_lineage_is_live_and_fail_closed(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    protocol = json.loads((sampling_null / "protocol.json").read_text())
    raw_manifest = Path(protocol["raw_run"]) / "manifest.json"
    with raw_manifest.open("a") as handle:
        handle.write("tampered")
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="raw-analysis lineage"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


def test_sampling_calibration_joint_array_mapping_is_fail_closed(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    arrays_path = sampling_null / "null_inference_arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    joint_ids = arrays["joint_test_id"].copy()
    joint_ids[[0, 1]] = joint_ids[[1, 0]]
    arrays["joint_test_id"] = joint_ids
    np.savez_compressed(arrays_path, **arrays)
    _write_ledger(
        sampling_null,
        tuple(name for name in SAMPLING_NULL_REQUIRED if name != "checksums.sha256"),
    )

    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="inference-array contract"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("duplicate_worm", "inference-array contract"),
        ("named_control_envelope", "inference-array contract"),
        ("shared_sign", "exact max-T mathematical contract"),
        ("null_maximum", "exact max-T mathematical contract"),
        ("studentized_t", "inference-array contract"),
    ),
)
def test_sampling_calibration_saved_array_math_is_fail_closed(
    tmp_path: Path, kind: str, message: str
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        if kind == "duplicate_worm":
            arrays["worm_ids"][1] = arrays["worm_ids"][0]
        elif kind == "named_control_envelope":
            arrays["sampling_control_worm_magnitude"][0, 0, 0] = (
                arrays["sampling_null_worm_magnitude"][0, 0] + 0.25
            )
        elif kind == "shared_sign":
            arrays["joint_sign_patterns"][1, 1] *= -1
        elif kind == "null_maximum":
            arrays["joint_null_max_abs_t"][1] += 0.25
        else:
            arrays["joint_test_observed_t"][0] += 0.25

    _rewrite_sampling_null_archive(sampling_null, mutate)
    _refresh_sampling_null_ledger(sampling_null)
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match=message):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "column",
    (
        "observed_mean",
        "observed_ci_2_5",
        "sampling_null_p95_worm",
        "sampling_excess_ci_97_5",
        "quiet_pseudo_mean_magnitude",
        "observed_to_quiet_pseudo_ratio",
    ),
)
def test_sampling_calibration_csv_reductions_are_fail_closed(
    tmp_path: Path, column: str
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    calibration_path = sampling_null / "sham_calibration.csv"
    calibration = pd.read_csv(calibration_path)
    calibration.loc[0, column] = float(calibration.loc[0, column]) + 0.05
    calibration.to_csv(calibration_path, index=False)
    _refresh_sampling_null_ledger(sampling_null)
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="CSV/worm-array reduction"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


def test_sampling_calibration_studentized_t_is_recomputed(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    arrays = _rewrite_sampling_null_archive(sampling_null, lambda value: None)
    changed_t = float(arrays["joint_test_observed_t"][0]) + 0.25
    arrays["joint_test_observed_t"][0] = changed_t
    np.savez_compressed(sampling_null / "null_inference_arrays.npz", **arrays)
    calibration_path = sampling_null / "sham_calibration.csv"
    calibration = pd.read_csv(calibration_path)
    calibration.loc[0, "observed_joint_student_t"] = changed_t
    calibration.to_csv(calibration_path, index=False)
    _refresh_sampling_null_ledger(sampling_null)
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="exact max-T mathematical contract"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


def test_sampling_calibration_adjusted_p_and_critical_value_are_recomputed(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    calibration_path = sampling_null / "sham_calibration.csv"
    calibration = pd.read_csv(calibration_path)
    arrays = _rewrite_sampling_null_archive(sampling_null, lambda value: None)
    changed_p = float(arrays["joint_test_max_t_p_value"][0]) + 1.0 / 65_536
    arrays["joint_test_max_t_p_value"][0] = changed_p
    calibration.loc[0, "observed_joint_max_t_p"] = changed_p
    calibration.to_csv(calibration_path, index=False)
    arrays_path = sampling_null / "null_inference_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    _refresh_sampling_null_ledger(sampling_null)
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="exact max-T mathematical contract"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()

    # A fresh fixture proves that matching NPZ/summary critical-value tampering
    # also cannot pass simply because the two saved products agree with each other.
    second = tmp_path / "second"
    second.mkdir()
    atlas, targeted, external, complete_family, sampling_null = _bundle(second)
    arrays = _rewrite_sampling_null_archive(sampling_null, lambda value: None)
    arrays["joint_simultaneous_critical_value"] = np.asarray(
        float(arrays["joint_simultaneous_critical_value"]) + 0.25,
        dtype=np.float32,
    )
    np.savez_compressed(sampling_null / "null_inference_arrays.npz", **arrays)
    summary_path = sampling_null / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["joint_max_t_critical_value"] = float(
        arrays["joint_simultaneous_critical_value"]
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _refresh_sampling_null_ledger(sampling_null)
    output = second / "explorer"
    with pytest.raises(ReportInputError, match="exact max-T mathematical contract"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=second,
        )
    assert not output.exists()


@pytest.mark.parametrize("manifest_field", ("random_seed", "bootstrap_replicates"))
def test_sampling_calibration_bootstrap_intervals_are_deterministically_recomputed(
    tmp_path: Path, manifest_field: str
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    manifest_path = sampling_null / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"][manifest_field] = int(manifest["config"][manifest_field]) + 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if manifest_field == "bootstrap_replicates":
        protocol_path = sampling_null / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["bootstrap_replicates"] = manifest["config"][manifest_field]
        protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    _refingerprint_sampling_null_manifest(sampling_null)
    _refresh_sampling_null_ledger(sampling_null)
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="CSV/worm-array reduction"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "fraction_column",
    (
        "observed_valid_fraction",
        "observed_genealogy_valid_fraction_0_10",
        "sampling_valid_fraction",
        "sampling_genealogy_valid_fraction_0_10",
        "pseudo_valid_fraction",
        "pseudo_genealogy_valid_fraction_0_10",
    ),
)
def test_sampling_calibration_all_support_components_gate_labels(
    tmp_path: Path, fraction_column: str
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    support_path = sampling_null / "support_diagnostics.csv"
    support = pd.read_csv(support_path)
    candidate_id = str(support.loc[support.queue_rank == 1, "candidate_id"].item())
    support.loc[support.candidate_id == candidate_id, fraction_column] = 0.70
    support.to_csv(support_path, index=False)
    calibration_path = sampling_null / "sham_calibration.csv"
    calibration = pd.read_csv(calibration_path)
    calibration.loc[calibration.candidate_id == candidate_id, fraction_column] = 0.70
    calibration.loc[calibration.candidate_id == candidate_id, "support_pass"] = False
    calibration.loc[calibration.candidate_id == candidate_id, "gate_reason"] = (
        "support_failure"
    )
    calibration.to_csv(calibration_path, index=False)
    _refresh_sampling_null_ledger(sampling_null)
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="label does not reproduce"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


def test_sampling_calibration_lag_sensitivity_cannot_be_promoted(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    calibration_path = sampling_null / "sham_calibration.csv"
    calibration = pd.read_csv(calibration_path)
    mask = calibration.selection_origin == "lag_sensitivity"
    calibration.loc[mask, "evidence_label"] = (
        "exceeds_sampling_and_quiet_controls"
    )
    calibration.to_csv(calibration_path, index=False)
    _refresh_sampling_null_ledger(sampling_null)
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="label does not reproduce"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("queue_rank", "column", "value"),
    (
        (4, "support_pass", True),
        (7, "selection_eligible", True),
        (8, "gate_reason", "sensitivity_origin"),
    ),
)
def test_sampling_calibration_durable_gate_metadata_is_recomputed(
    tmp_path: Path, queue_rank: int, column: str, value: object
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    calibration_path = sampling_null / "sham_calibration.csv"
    calibration = pd.read_csv(calibration_path)
    calibration.loc[calibration.queue_rank == queue_rank, column] = value
    calibration.to_csv(calibration_path, index=False)
    _refresh_sampling_null_ledger(sampling_null)
    output = tmp_path / "explorer"
    with pytest.raises(ReportInputError, match="gate metadata does not reproduce"):
        build_prediction_atlas_explorer(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            output,
            provenance_root=tmp_path,
        )
    assert not output.exists()


def test_sampling_frontend_matches_exact_metric_and_is_custom_safe(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, complete_family, sampling_null = _bundle(tmp_path)
    output = tmp_path / "explorer"
    build_prediction_atlas_explorer(
        atlas,
        targeted,
        external,
        complete_family,
        sampling_null,
        output,
        provenance_root=tmp_path,
    )
    html = (output / "atlas_explorer.html").read_text()
    sampling_match = html.split("function sameSamplingCell", 1)[1].split(
        "function calibrationLabel", 1
    )[0]
    for key in (
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
        "context",
    ):
        assert key in sampling_match
    assert 'String(row.metric)===String(state.channel)' in sampling_match
    assert 'String(state.method)!=="progressive_bridge_smc"' in sampling_match
    assert "queueRank" not in sampling_match
    assert "This exact cell was not among the eight N128 calibrations" in html
    assert "Support was audited here, but this outcome was not calibrated" in html
    assert "No metric-level evidence is borrowed" in html
    assert 'metricDetails.hidden=!metricRows.length' in html
    assert "[hidden]{display:none!important}" in html
    assert "A result from another cell or metric is not substituted" in html
    assert "Number(row.source_index)===Number(sourceIndex)" in sampling_match
    assert "Number(row.target_index)===Number(targetIndex)" in sampling_match
    assert "Number(state.source)" not in sampling_match
    assert "Number(state.target)" not in sampling_match
    assert "function currentSamplingSupport()" in html
    assert "function positiveExcessPass(row,prefix)" in html
    assert "values.samplingValid>=validThreshold" in html
    assert "values.samplingGenealogy>=genealogyThreshold" in html
    assert "finiteNumber(cellP)&&Number(cellP)<=0.05" in html
    assert "Test passes · restricted" in html
    assert "separate 64-test max-T" in html
    assert "const quietStatus=passesQuiet?" in html
    assert "const samplerStatus=fullSamplerPass?" in html
