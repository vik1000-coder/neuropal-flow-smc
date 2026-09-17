"""Build a self-contained, offline explorer for the reviewed neural atlas.

The explorer consumes only the checksum-verified canonical atlas, the linked
targeted N128 analysis, the post-freeze external-comparison bundle, the
complete-family statistical-evidence bundle, and the selection-conditioned
N128 sampling-null calibration.  Dense normalized-mean matrices are stored in
the HTML payload as symmetric signed int16 slices (one scale per lag/horizon
matrix) and decoded locally in the browser.  The generated HTML contains no
network request or remote dependency.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.prediction_atlas_report import (
    AtlasEvidence,
    ExternalEvidence,
    ReportInputError,
    TargetedEvidence,
    _json_safe,
    _read_json,
    _safe_relative,
    _validate_atlas,
    _validate_external,
    _validate_targeted,
    _verify_bundle_checksums,
    sha256,
)
from compatibility_neural_benchmark.prediction_atlas_runner import (
    canonical_fingerprint,
)
from compatibility_neural_benchmark.targeted_sampling_null_analysis import (
    _bootstrap_interval as _sampling_bootstrap_interval,
    _joint_studentized_max_t as _sampling_joint_studentized_max_t,
    hashlib_sha256_int as _sampling_bootstrap_seed,
)
from compatibility_neural_benchmark.targeted_confirmation_analysis import ORIENTATION


SCHEMA_VERSION = "prediction_atlas_explorer_payload_v3"
MANIFEST_SCHEMA_VERSION = "prediction_atlas_explorer_manifest_v3"
TITLE = "Neural prediction atlas explorer"
OUTPUT_FILES = (
    "atlas_explorer.html",
    "explorer_data.json",
    "manifest.json",
    "checksums.sha256",
)
OPTIONAL_COMPOSITION_FILE = "stimulus_composition.csv"
ENCODING = "signed_int16_le_base64"
COMPLETE_FAMILY_REQUIRED = (
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
    "checksums.sha256",
)
COMPLETE_FAMILY_IDS = (
    "baseline_endpoint_mean",
    "baseline_endpoint_log_sd",
    "active_minus_baseline_endpoint_mean",
    "active_minus_baseline_endpoint_log_sd",
)
COMPLETE_FAMILY_DIMENSIONS = {
    "baseline_endpoint_mean": ("endpoint_mean", "baseline"),
    "baseline_endpoint_log_sd": ("endpoint_log_sd", "baseline"),
    "active_minus_baseline_endpoint_mean": (
        "endpoint_mean",
        "active_minus_baseline",
    ),
    "active_minus_baseline_endpoint_log_sd": (
        "endpoint_log_sd",
        "active_minus_baseline",
    ),
}
COMPLETE_FAMILY_LABELS = {
    "strong_edge_lag_unresolved",
    "strong_edge_with_lag_structure",
    "exploratory",
    "sampling_limited",
    "no_complete_family_evidence",
}
SAMPLING_NULL_SCHEMA_VERSION = "prediction_atlas_sampling_null_analysis_v2"
SAMPLING_NULL_REQUIRED = (
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
SAMPLING_NULL_METRICS = (
    "endpoint_mean",
    "endpoint_log_sd",
    "endpoint_wasserstein1",
)
SAMPLING_NULL_CONTROLS = (
    "observed",
    "low_low",
    "high_high",
    "midpoint_midpoint",
    "within_low_split",
    "within_high_split",
    "quiet_pseudo",
)
SAMPLING_NULL_LABELS = {
    "exceeds_sampling_and_quiet_controls",
    "exceeds_sampling_controls_only",
    "indistinguishable_from_sampling_controls",
    "sampling_limited",
}
SAMPLING_NULL_SELECTION_ORIGINS = {
    "strong_primary",
    "lag_sensitivity",
}
EXACT_WORM_SIGN_PATTERNS = 65_536
EXACT_WORM_P_FLOOR = 1.0 / EXACT_WORM_SIGN_PATTERNS
SAMPLING_NULL_CLAIM_BOUNDARY = (
    "sampling distinguishability and quiet-time temporal specificity under a "
    "frozen learned observed-data law; not a causal effect, anatomical edge, "
    "receptor action, biological null, or physical delay"
)


@dataclass(frozen=True)
class CompleteFamilyEvidence:
    root: Path
    manifest: Mapping[str, Any]
    explorer: Mapping[str, Any]
    protocol: Mapping[str, Any]
    summary: Mapping[str, Any]
    validation: Mapping[str, Any]
    checksums: Mapping[str, str]


@dataclass(frozen=True)
class SamplingNullEvidence:
    root: Path
    raw_root: Path
    manifest: Mapping[str, Any]
    protocol: Mapping[str, Any]
    summary: Mapping[str, Any]
    validation: Mapping[str, Any]
    calibration: pd.DataFrame
    support: pd.DataFrame
    checksums: Mapping[str, str]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in frame.to_dict(orient="records")]


def _validate_complete_family_evidence(
    evidence_dir: Path, *, atlas: AtlasEvidence
) -> CompleteFamilyEvidence:
    """Validate the compact statistical layer and its canonical-atlas link."""

    root = Path(evidence_dir).resolve()
    checksums = _verify_bundle_checksums(root, COMPLETE_FAMILY_REQUIRED)
    manifest = _read_json(root / "manifest.json")
    explorer = _read_json(root / "explorer_evidence.json")
    protocol = _read_json(root / "protocol.json")
    summary = _read_json(root / "summary.json")
    validation = _read_json(root / "validation.json")
    expected_artifacts = {
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
    }
    if (
        manifest.get("schema_version")
        != "complete-family-joint-evidence-manifest-v1"
        or manifest.get("status") != "complete"
        or manifest.get("artifacts") != expected_artifacts
    ):
        raise ReportInputError("complete-family evidence manifest contract failed")
    if (
        explorer.get("schema_version") != "complete-family-explorer-evidence-v1"
        or explorer.get("status") != "complete"
        or protocol.get("schema_version")
        != "complete-family-joint-evidence-protocol-v1"
        or summary.get("schema_version")
        != "complete-family-joint-evidence-summary-v1"
        or summary.get("status") != "complete"
        or validation.get("schema_version")
        != "complete-family-joint-evidence-validation-v1"
        or validation.get("status") != "passed"
    ):
        raise ReportInputError("complete-family evidence schema/status contract failed")
    required_validation_flags = (
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
    if not all(validation.get(field) is True for field in required_validation_flags):
        raise ReportInputError("complete-family evidence validation receipt failed")
    worm_matrix_sha = atlas.checksums.get("worm_matrices.npz")
    if (
        not worm_matrix_sha
        or validation.get("input_sha256") != worm_matrix_sha
    ):
        raise ReportInputError(
            "complete-family evidence is not linked to the reviewed worm matrices"
        )

    joint = protocol.get("joint_primary")
    if not isinstance(joint, Mapping) or (
        tuple(joint.get("families", ())) != COMPLETE_FAMILY_IDS
        or joint.get("independent_unit") != "worm"
        or int(joint.get("sign_patterns", -1)) != 65_536
        or float(joint.get("alpha", -1)) != 0.05
    ):
        raise ReportInputError("complete-family exact-test protocol contract failed")
    family_rows = explorer.get("family_summary")
    if (
        not isinstance(family_rows, list)
        or len(family_rows) != len(COMPLETE_FAMILY_IDS)
        or not all(isinstance(row, Mapping) for row in family_rows)
        or {str(row.get("family_id")) for row in family_rows}
        != set(COMPLETE_FAMILY_IDS)
    ):
        raise ReportInputError("complete-family explorer family inventory failed")
    for row in family_rows:
        try:
            expected_channel, expected_context = COMPLETE_FAMILY_DIMENSIONS[
                str(row.get("family_id"))
            ]
            valid = (
                row.get("channel") == expected_channel
                and row.get("context") == expected_context
                and int(row.get("n_worms", -1)) == atlas.n_worms == 17
                and int(row.get("sign_patterns", -1)) == 65_536
                and int(row.get("complete_edges", -1)) == 54 * 53
                and int(row.get("complete_cells", -1)) > 0
                and 0
                <= int(row.get("support_eligible_edges", -1))
                <= int(row.get("complete_edges", -1))
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ReportInputError("complete-family explorer family counts are invalid")
    headline = explorer.get("headline")
    if not isinstance(headline, Mapping):
        raise ReportInputError("complete-family explorer headline is missing")
    try:
        headline_valid = (
            int(headline.get("joint_primary_families", -1))
            == len(COMPLETE_FAMILY_IDS)
            and int(headline.get("complete_edge_tests", -1))
            == len(COMPLETE_FAMILY_IDS) * 54 * 53
            and int(headline.get("support_eligible_edge_tests", -1))
            == sum(int(row["support_eligible_edges"]) for row in family_rows)
            and int(headline.get("joint_primary_edge_discoveries", -1))
            == sum(
                int(row["joint_primary_edge_max_t_discoveries"])
                for row in family_rows
            )
            and int(headline.get("joint_primary_flat_lag_edge_discoveries", -1))
            == sum(
                int(row["joint_primary_flat_lag_edge_discoveries"])
                for row in family_rows
            )
            and int(headline.get("experiment_ready", -1)) == 0
            and sum(int(row["experiment_ready"]) for row in family_rows) == 0
            and headline.get("practical_status")
            == "pending_sham_and_experimental_threshold"
        )
    except (TypeError, ValueError, KeyError):
        headline_valid = False
    if not headline_valid:
        raise ReportInputError("complete-family explorer headline counts are invalid")
    labels = explorer.get("evidence_labels")
    if not isinstance(labels, Mapping) or not COMPLETE_FAMILY_LABELS.issubset(labels):
        raise ReportInputError("complete-family explorer evidence labels are incomplete")
    atlas_lags = {int(value) for value in atlas.protocol.get("source_lag_frames", ())}
    atlas_horizons = {
        int(value) for value in atlas.protocol.get("horizon_frames", ())
    }
    if not atlas_lags or not atlas_horizons:
        raise ReportInputError("canonical atlas timing axes are missing")
    for collection in ("top_edges", "sampling_null_shortlist"):
        rows = explorer.get(collection)
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ReportInputError(f"complete-family explorer {collection} is invalid")
        invalid_labels = {
            str(row.get("evidence_label"))
            for row in rows
            if str(row.get("evidence_label")) not in COMPLETE_FAMILY_LABELS
        }
        if invalid_labels:
            raise ReportInputError(
                f"complete-family explorer {collection} has unknown evidence labels"
            )
        for row in rows:
            family_id = str(row.get("family_id"))
            source = str(row.get("source_neuron"))
            target = str(row.get("target_neuron"))
            expected_channel, expected_context = COMPLETE_FAMILY_DIMENSIONS.get(
                family_id, (None, None)
            )
            if (
                family_id not in COMPLETE_FAMILY_DIMENSIONS
                or row.get("channel") != expected_channel
                or row.get("context") != expected_context
                or source not in atlas.neurons
                or target not in atlas.neurons
                or source == target
            ):
                raise ReportInputError(
                    f"complete-family explorer {collection} has an invalid edge key"
                )
            if "source_index" in row and (
                int(row["source_index"]) != atlas.neurons.index(source)
                or int(row["target_index"]) != atlas.neurons.index(target)
            ):
                raise ReportInputError(
                    f"complete-family explorer {collection} neuron mapping failed"
                )
            if collection == "sampling_null_shortlist" and (
                int(row.get("source_lag_frames", -1)) not in atlas_lags
                or int(row.get("horizon_frames", -1)) not in atlas_horizons
            ):
                raise ReportInputError(
                    "complete-family explorer shortlist timing mapping failed"
                )
            label = str(row.get("evidence_label"))
            if (
                row.get("experiment_ready") is not False
                or row.get("practical_status")
                != "pending_sham_and_experimental_threshold"
                or (
                    collection == "sampling_null_shortlist"
                    and (
                        row.get("sesoi_status")
                        != "not_defined_pending_sham_calibration"
                        or row.get("independent_confirmation_status") != "not_run"
                    )
                )
            ):
                raise ReportInputError(
                    f"complete-family explorer {collection} readiness gates failed"
                )
            edge_p = _exact_probability(
                row.get("joint_primary_edge_max_t_p_value"),
                label=f"complete-family {collection} edge p-value",
            )
            flat_lag_p = _exact_probability(
                row.get("joint_primary_flat_lag_max_t_p_value"),
                label=f"complete-family {collection} flat-lag p-value",
            )
            if collection == "sampling_null_shortlist":
                _exact_probability(
                    row.get("joint_primary_cell_max_t_p_value"),
                    label="complete-family shortlist cell p-value",
                )
            label_valid = (
                (
                    label == "strong_edge_lag_unresolved"
                    and edge_p <= 0.05
                    and flat_lag_p > 0.05
                )
                or (
                    label == "strong_edge_with_lag_structure"
                    and edge_p <= 0.05
                    and flat_lag_p <= 0.05
                )
                or (
                    label in {"exploratory", "no_complete_family_evidence"}
                    and edge_p > 0.05
                )
                or label == "sampling_limited"
            )
            if not label_valid:
                raise ReportInputError(
                    f"complete-family explorer {collection} label/p-value contract failed"
                )
    legacy = explorer.get("legacy_queue")
    if (
        not isinstance(legacy, Mapping)
        or not isinstance(legacy.get("warning"), str)
        or not isinstance(legacy.get("matches"), list)
        or int(legacy.get("matched_cells", -1)) != len(legacy.get("matches", ()))
    ):
        raise ReportInputError("complete-family explorer legacy-queue linkage failed")
    if any(
        str(row.get("family_id")) not in COMPLETE_FAMILY_DIMENSIONS
        or str(row.get("source_neuron")) not in atlas.neurons
        or str(row.get("target_neuron")) not in atlas.neurons
        or int(row.get("source_lag_frames", -1)) <= 0
        or int(row.get("horizon_frames", -1)) <= 0
        or str(row.get("evidence_label")) not in COMPLETE_FAMILY_LABELS
        for row in legacy.get("matches", ())
    ):
        raise ReportInputError("complete-family explorer legacy evidence keys failed")
    queue_by_rank = {
        int(row.queue_rank): row for row in atlas.queue.itertuples(index=False)
    }
    for row in legacy.get("matches", ()):
        try:
            rank = int(row["legacy_queue_rank"])
            queue_row = queue_by_rank[rank]
            expected_channel, expected_context = COMPLETE_FAMILY_DIMENSIONS[
                str(row["family_id"])
            ]
            valid = (
                str(queue_row.method) == "progressive_bridge_smc"
                and str(queue_row.channel) == expected_channel
                and str(queue_row.context) == expected_context
                and str(queue_row.source_neuron) == str(row["source_neuron"])
                and str(queue_row.target_neuron) == str(row["target_neuron"])
                and int(queue_row.source_lag_frames)
                == int(row["source_lag_frames"])
                and int(queue_row.horizon_frames) == int(row["horizon_frames"])
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ReportInputError(
                "complete-family legacy evidence does not rejoin the frozen queue"
            )
        legacy_label = str(row.get("evidence_label"))
        if legacy_label in {"sampling_limited", "no_complete_family_evidence"}:
            if any(
                row.get(field) is not None
                for field in (
                    "joint_primary_cell_max_t_p_value",
                    "joint_primary_edge_max_t_p_value",
                    "joint_primary_flat_lag_max_t_p_value",
                )
            ):
                raise ReportInputError(
                    "complete-family limited legacy evidence unexpectedly has p-values"
                )
            edge_p = None
        else:
            _exact_probability(
                row.get("joint_primary_cell_max_t_p_value"),
                label="complete-family legacy cell p-value",
            )
            edge_p = _exact_probability(
                row.get("joint_primary_edge_max_t_p_value"),
                label="complete-family legacy edge p-value",
            )
        flat_lag_p = (
            _exact_probability(
                row.get("joint_primary_flat_lag_max_t_p_value"),
                label="complete-family legacy flat-lag p-value",
            )
            if row.get("joint_primary_flat_lag_max_t_p_value") is not None
            else None
        )
        if (
            legacy_label == "strong_edge_lag_unresolved"
            and (
                edge_p is None
                or edge_p > 0.05
                or (flat_lag_p is not None and flat_lag_p <= 0.05)
            )
        ) or (
            legacy_label == "strong_edge_with_lag_structure"
            and (
                edge_p is None
                or edge_p > 0.05
                or flat_lag_p is None
                or flat_lag_p > 0.05
            )
        ) or (
            legacy_label in {"exploratory", "no_complete_family_evidence"}
            and edge_p is not None
            and edge_p <= 0.05
        ):
            raise ReportInputError(
                "complete-family legacy label/p-value contract failed"
            )
        _unit_interval_probability(
            row.get("legacy_postscreen_q_value"),
            label="complete-family legacy post-screen q-value",
        )
    sensitivity = explorer.get("sensitivity_support05")
    if (
        not isinstance(sensitivity, Mapping)
        or sensitivity.get("available") is not True
        or sensitivity.get("support_gate") != "sensitivity_0.5"
        or sensitivity.get("claim_status")
        != "sensitivity_only_never_promotes_primary_claim"
        or int(sensitivity.get("experiment_ready", -1)) != 0
        or not isinstance(sensitivity.get("warning"), str)
        or not isinstance(sensitivity.get("named_lag_rows"), list)
        or len(sensitivity.get("named_lag_rows", ())) != 2
        or any(
            row.get("evidence_label") != "sampling_limited"
            or row.get("experiment_ready") is not False
            or row.get("sensitivity_only") is not True
            for row in sensitivity.get("named_lag_rows", ())
        )
    ):
        raise ReportInputError("complete-family sensitivity evidence contract failed")
    for row in sensitivity["named_lag_rows"]:
        source = str(row.get("source_neuron"))
        target = str(row.get("target_neuron"))
        try:
            valid = (
                str(row.get("family_id")) in COMPLETE_FAMILY_DIMENSIONS
                and source in atlas.neurons
                and target in atlas.neurons
                and source != target
                and int(row.get("source_index", -1)) == atlas.neurons.index(source)
                and int(row.get("target_index", -1)) == atlas.neurons.index(target)
                and int(row.get("source_lag_frames", -1)) in atlas_lags
                and int(row.get("horizon_frames", -1)) in atlas_horizons
                and EXACT_WORM_P_FLOOR
                <= float(row.get("joint_sensitivity_lag_contrast_max_t_p_value"))
                <= 1.0
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ReportInputError("complete-family sensitivity evidence key failed")

    combined = pd.read_csv(root / "sampling_null_combined_8.csv")
    combined_columns = [
        "queue_rank",
        "channel",
        "context",
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
        "selection_origin",
        "run_sampling_nulls",
    ]
    if list(combined.columns) != combined_columns or len(combined) != 8:
        raise ReportInputError("combined sampling-null queue schema/count failed")
    key_columns = [
        "channel",
        "context",
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
    ]
    run_flags = combined["run_sampling_nulls"].astype(str).str.lower()
    if (
        list(pd.to_numeric(combined["queue_rank"], errors="coerce"))
        != list(range(1, 9))
        or combined.duplicated(key_columns).any()
        or list(combined["selection_origin"].astype(str))
        != ["strong_primary"] * 6 + ["lag_sensitivity"] * 2
        or not (run_flags == "true").all()
        or not (combined["channel"].astype(str) == "endpoint_mean").all()
        or not (combined["context"].astype(str) == "baseline").all()
    ):
        raise ReportInputError("combined sampling-null queue contract failed")
    for row in combined.itertuples(index=False):
        source = str(row.source_neuron)
        target = str(row.target_neuron)
        if (
            source not in atlas.neurons
            or target not in atlas.neurons
            or source == target
            or int(row.source_index) != atlas.neurons.index(source)
            or int(row.target_index) != atlas.neurons.index(target)
            or int(row.source_lag_frames) <= 0
            or int(row.horizon_frames) <= 0
        ):
            raise ReportInputError("combined sampling-null neuron/key mapping failed")
    compact_key = lambda row: (
        int(row.get("source_index", -1)),
        int(row.get("target_index", -1)),
        int(row.get("source_lag_frames", -1)),
        int(row.get("horizon_frames", -1)),
    )
    expected_primary = {
        compact_key(row) for row in explorer["sampling_null_shortlist"]
    }
    expected_lag = {compact_key(row) for row in sensitivity["named_lag_rows"]}
    observed_primary = {
        (
            int(row.source_index),
            int(row.target_index),
            int(row.source_lag_frames),
            int(row.horizon_frames),
        )
        for row in combined.loc[
            combined.selection_origin == "strong_primary"
        ].itertuples(index=False)
    }
    observed_lag = {
        (
            int(row.source_index),
            int(row.target_index),
            int(row.source_lag_frames),
            int(row.horizon_frames),
        )
        for row in combined.loc[
            combined.selection_origin == "lag_sensitivity"
        ].itertuples(index=False)
    }
    if observed_primary != expected_primary or observed_lag != expected_lag:
        raise ReportInputError("combined sampling-null queue lineage failed")
    return CompleteFamilyEvidence(
        root=root,
        manifest=manifest,
        explorer=explorer,
        protocol=protocol,
        summary=summary,
        validation=validation,
        checksums=checksums,
    )


def _sampling_cell_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return (
            str(row["source_neuron"]),
            str(row["target_neuron"]),
            int(row["source_index"]),
            int(row["target_index"]),
            int(row["source_lag_frames"]),
            int(row["horizon_frames"]),
            str(row["context"]),
            str(row.get("channel", row.get("selected_channel"))),
            str(row["selection_origin"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReportInputError("sampling-null calibration has an invalid cell key") from error


def _strict_bool_series(series: pd.Series, *, label: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false")).all():
        raise ReportInputError(f"{label} contains a non-boolean value")
    return normalized == "true"


def _exact_probability(value: Any, *, label: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ReportInputError(f"{label} is not numeric") from error
    if not math.isfinite(probability) or not EXACT_WORM_P_FLOOR <= probability <= 1.0:
        raise ReportInputError(
            f"{label} lies outside the exact [{EXACT_WORM_P_FLOOR:g}, 1] range"
        )
    return probability


def _unit_interval_probability(value: Any, *, label: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ReportInputError(f"{label} is not numeric") from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ReportInputError(f"{label} lies outside [0, 1]")
    return probability


def _validate_sampling_null_evidence(
    analysis_dir: Path,
    *,
    atlas: AtlasEvidence,
    complete_family: CompleteFamilyEvidence,
    provenance_root: Path,
) -> SamplingNullEvidence:
    """Validate the selection-conditioned N128 sampler-control layer.

    The analysis is deliberately a compact, fixed eight-cell calibration.  It
    is not a generic lookup table and it is not allowed to float away from the
    exact complete-family queue or raw-run manifest that selected those cells.
    """

    root = Path(analysis_dir).resolve()
    checksums = _verify_bundle_checksums(root, SAMPLING_NULL_REQUIRED)
    expected_ledger = set(SAMPLING_NULL_REQUIRED).difference({"checksums.sha256"})
    observed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if set(checksums) != expected_ledger or observed_files != set(
        SAMPLING_NULL_REQUIRED
    ):
        raise ReportInputError(
            "sampling-null checksum ledger must cover exactly the analysis outputs"
        )
    manifest = _read_json(root / "manifest.json")
    protocol = _read_json(root / "protocol.json")
    summary = _read_json(root / "summary.json")
    validation = _read_json(root / "validation.json")
    if (
        manifest.get("analysis_schema_version") != SAMPLING_NULL_SCHEMA_VERSION
        or protocol.get("analysis_schema_version") != SAMPLING_NULL_SCHEMA_VERSION
    ):
        raise ReportInputError("sampling-null analysis v2 schema contract failed")
    analysis_fingerprint = str(manifest.get("analysis_fingerprint", ""))
    manifest_spec = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_utc", "analysis_fingerprint"}
    }
    if (
        len(analysis_fingerprint) != 64
        or canonical_fingerprint(manifest_spec) != analysis_fingerprint
        or validation.get("status") != "pass"
        or validation.get("analysis_fingerprint") != analysis_fingerprint
    ):
        raise ReportInputError("sampling-null analysis fingerprint/status contract failed")
    if tuple(manifest.get("outputs", ())) != SAMPLING_NULL_REQUIRED:
        raise ReportInputError("sampling-null analysis output inventory failed")

    required_validation_flags = (
        "raw_grid_complete",
        "worm_inference_only",
        "particles_not_treated_as_worms",
        "quiet_pseudo_windows_all_verified",
        "finite_event_rows",
        "wasserstein_nonnegative_before_context_contrast",
        "joint_max_t_p_values_bounded",
        "joint_max_t_p_value_null_pattern_valid",
        "joint_max_t_test_grid_complete",
        "no_pending_evidence_labels",
        "sampling_limited_gate_enforced",
        "sampling_controls_support_gated",
        "gate_reason_complete",
        "lag_sensitivity_rows_sampling_limited",
    )
    if not all(validation.get(field) is True for field in required_validation_flags):
        raise ReportInputError("sampling-null analysis validation receipt failed")
    if (
        int(validation.get("external_reference_inputs", -1)) != 0
        or int(validation.get("raw_checksums_verified", 0)) <= 0
        or validation.get("claim_boundary") != SAMPLING_NULL_CLAIM_BOUNDARY
        or protocol.get("claim_boundary") != SAMPLING_NULL_CLAIM_BOUNDARY
    ):
        raise ReportInputError("sampling-null analysis claim boundary failed")

    try:
        manifest_config = manifest.get("config", {})
        bootstrap_replicates = int(protocol.get("bootstrap_replicates", -1))
        bootstrap_seed = int(manifest_config.get("random_seed", -1))
        protocol_valid = (
            tuple(protocol.get("metrics", ())) == SAMPLING_NULL_METRICS
            and tuple(protocol.get("control_families", ()))
            == SAMPLING_NULL_CONTROLS
            and protocol.get("inference_unit") == "worm"
            and "checkpoint seeds averaged within worm"
            in str(protocol.get("checkpoint_seed_reduction", ""))
            and "maximum of low-low, high-high, and midpoint-midpoint"
            in str(protocol.get("sampling_control_envelope", ""))
            and "single-step max-T"
            in str(protocol.get("multiplicity", ""))
            and "selection-conditioned"
            in str(protocol.get("selection_status", ""))
            and float(protocol.get("alpha", -1)) == 0.05
            and bootstrap_replicates >= 100
            and int(manifest_config.get("bootstrap_replicates", -1))
            == bootstrap_replicates
            and bootstrap_seed >= 0
            and float(manifest_config.get("alpha", -1)) == 0.05
            and float(
                manifest_config.get("genealogy_fraction_threshold", -1)
            )
            == 0.10
        )
    except (TypeError, ValueError):
        protocol_valid = False
    if not protocol_valid:
        raise ReportInputError("sampling-null statistical protocol contract failed")

    selected = protocol.get("selected_cells")
    if (
        not isinstance(selected, list)
        or len(selected) != 8
        or not all(isinstance(row, Mapping) for row in selected)
    ):
        raise ReportInputError("sampling-null selected-cell inventory failed")
    selected_keys = [_sampling_cell_key(row) for row in selected]
    if (
        len(set(selected_keys)) != 8
        or [key[-1] for key in selected_keys].count("strong_primary") != 6
        or [key[-1] for key in selected_keys].count("lag_sensitivity") != 2
        or any(key[6] != "baseline" or key[7] != "endpoint_mean" for key in selected_keys)
        or any(tuple(row.get("required_phases", ())) != ("baseline",) for row in selected)
    ):
        raise ReportInputError("sampling-null selected-cell composition failed")
    candidate_ids = [str(row.get("candidate_id", "")) for row in selected]
    if any(not value for value in candidate_ids) or len(set(candidate_ids)) != 8:
        raise ReportInputError("sampling-null candidate identifiers are invalid")
    selected_by_id = dict(zip(candidate_ids, selected))
    for row in selected:
        source = str(row["source_neuron"])
        target = str(row["target_neuron"])
        if (
            source not in atlas.neurons
            or target not in atlas.neurons
            or source == target
            or int(row["source_index"]) != atlas.neurons.index(source)
            or int(row["target_index"]) != atlas.neurons.index(target)
        ):
            raise ReportInputError("sampling-null selected-cell neuron mapping failed")

    combined = pd.read_csv(complete_family.root / "sampling_null_combined_8.csv")
    combined_keys = {
        (
            str(row.source_neuron),
            str(row.target_neuron),
            int(row.source_index),
            int(row.target_index),
            int(row.source_lag_frames),
            int(row.horizon_frames),
            str(row.context),
            str(row.channel),
            str(row.selection_origin),
        )
        for row in combined.itertuples(index=False)
    }
    if set(selected_keys) != combined_keys:
        raise ReportInputError("sampling-null cells do not reproduce the frozen queue")

    raw_root = Path(str(protocol.get("raw_run", ""))).resolve()
    _safe_relative(raw_root, provenance_root)
    raw_manifest_path = raw_root / "manifest.json"
    raw_validation_path = raw_root / "validation.json"
    raw_checksums_path = raw_root / "checksums.sha256"
    for path, field in (
        (raw_manifest_path, "raw_manifest_sha256"),
        (raw_validation_path, "raw_validation_sha256"),
        (raw_checksums_path, "raw_checksums_sha256"),
    ):
        if not path.is_file() or sha256(path) != protocol.get(field):
            raise ReportInputError(f"sampling-null raw-analysis lineage failed at {path.name}")
    raw_manifest = _read_json(raw_manifest_path)
    raw_validation = _read_json(raw_validation_path)
    raw_fingerprint = str(protocol.get("raw_run_spec_fingerprint", ""))
    raw_cohort_worms = tuple(str(value) for value in raw_manifest.get("cohort_worms", ()))
    if (
        not raw_fingerprint
        or manifest.get("raw_run_spec_fingerprint") != raw_fingerprint
        or raw_manifest.get("run_spec_fingerprint") != raw_fingerprint
        or raw_validation.get("run_spec_fingerprint") != raw_fingerprint
        or raw_validation.get("status") != "pass"
        or tuple(raw_manifest.get("neurons", ())) != atlas.neurons
        or len(raw_cohort_worms) != atlas.n_worms
        or len(set(raw_cohort_worms)) != atlas.n_worms
        or any(not value for value in raw_cohort_worms)
        or raw_manifest.get("hypothesis_queue_sha256")
        != complete_family.checksums.get("sampling_null_combined_8.csv")
    ):
        raise ReportInputError("sampling-null raw-run provenance contract failed")
    raw_queue = Path(str(raw_manifest.get("hypothesis_queue", ""))).resolve()
    if (
        raw_queue != (complete_family.root / "sampling_null_combined_8.csv").resolve()
        or not raw_queue.is_file()
        or sha256(raw_queue) != raw_manifest.get("hypothesis_queue_sha256")
    ):
        raise ReportInputError("sampling-null raw queue lineage failed")
    raw_selected = raw_manifest.get("selected_cells")
    if not isinstance(raw_selected, list) or len(raw_selected) != 8:
        raise ReportInputError("sampling-null raw selected-cell inventory failed")
    raw_projection = {
        (
            str(row.get("candidate_id")),
            str(row.get("source_neuron")),
            str(row.get("target_neuron")),
            int(row.get("source_index", -1)),
            int(row.get("target_index", -1)),
            int(row.get("source_lag_frames", -1)),
            int(row.get("horizon_frames", -1)),
            str(row.get("context")),
            str(row.get("channel")),
        )
        for row in raw_selected
    }
    analysis_projection = {
        (
            str(row.get("candidate_id")),
            str(row.get("source_neuron")),
            str(row.get("target_neuron")),
            int(row.get("source_index", -1)),
            int(row.get("target_index", -1)),
            int(row.get("source_lag_frames", -1)),
            int(row.get("horizon_frames", -1)),
            str(row.get("context")),
            str(row.get("channel")),
        )
        for row in selected
    }
    if raw_projection != analysis_projection:
        raise ReportInputError("sampling-null raw/analysis cell lineage failed")

    input_checksums = pd.read_csv(root / "input_checksums.csv")
    if set(("path", "sha256", "bytes")).difference(input_checksums.columns):
        raise ReportInputError("sampling-null input checksum table is incomplete")
    if input_checksums["path"].astype(str).duplicated().any():
        raise ReportInputError("sampling-null input checksum table has duplicate paths")
    input_by_path = input_checksums.set_index(input_checksums["path"].astype(str))
    for path in (raw_manifest_path, raw_validation_path, raw_checksums_path):
        key = str(path)
        if key not in input_by_path.index or str(input_by_path.loc[key, "sha256"]) != sha256(path):
            raise ReportInputError("sampling-null input table breaks raw-analysis lineage")

    calibration = pd.read_csv(root / "sham_calibration.csv")
    required_calibration_columns = {
        "candidate_id",
        "queue_rank",
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
        "context",
        "channel",
        "selection_origin",
        "metric",
        "context_signed",
        "n_worms",
        "observed_mean",
        "observed_ci_2_5",
        "observed_ci_97_5",
        "observed_joint_max_t_p",
        "observed_joint_student_t",
        "sampling_null_mean_magnitude",
        "sampling_null_p95_worm",
        "low_low_mean_magnitude",
        "high_high_mean_magnitude",
        "midpoint_midpoint_mean_magnitude",
        "sampling_excess_mean",
        "sampling_excess_ci_2_5",
        "sampling_excess_ci_97_5",
        "sampling_excess_joint_max_t_p",
        "sampling_excess_joint_student_t",
        "quiet_pseudo_mean_magnitude",
        "quiet_pseudo_p95_worm",
        "temporal_specificity_excess_mean",
        "temporal_specificity_ci_2_5",
        "temporal_specificity_ci_97_5",
        "temporal_specificity_joint_max_t_p",
        "temporal_specificity_joint_student_t",
        "observed_valid_fraction",
        "observed_genealogy_valid_fraction_0_10",
        "sampling_valid_fraction",
        "sampling_genealogy_valid_fraction_0_10",
        "pseudo_valid_fraction",
        "pseudo_genealogy_valid_fraction_0_10",
        "observed_to_sampling_null_ratio",
        "observed_to_quiet_pseudo_ratio",
        "support_pass",
        "selection_eligible",
        "gate_reason",
        "evidence_label",
        "claim_boundary",
    }
    missing = sorted(required_calibration_columns.difference(calibration.columns))
    if missing or len(calibration) != 24:
        raise ReportInputError(
            f"sampling-null calibration schema/count failed (missing={missing})"
        )
    calibration["context_signed"] = _strict_bool_series(
        calibration["context_signed"], label="sampling-null context_signed"
    )
    calibration["support_pass"] = _strict_bool_series(
        calibration["support_pass"], label="sampling-null support_pass"
    )
    calibration["selection_eligible"] = _strict_bool_series(
        calibration["selection_eligible"],
        label="sampling-null selection_eligible",
    )
    numeric_columns = sorted(
        required_calibration_columns.difference(
            {
                "candidate_id",
                "source_neuron",
                "target_neuron",
                "context",
                "channel",
                "selection_origin",
                "metric",
                "context_signed",
                "support_pass",
                "selection_eligible",
                "gate_reason",
                "evidence_label",
                "claim_boundary",
                "observed_joint_max_t_p",
                "observed_joint_student_t",
            }
        )
    )
    for column in numeric_columns:
        calibration[column] = pd.to_numeric(calibration[column], errors="coerce")
        if not np.isfinite(calibration[column]).all():
            raise ReportInputError(f"sampling-null calibration {column} is nonfinite")
    calibration["observed_joint_max_t_p"] = pd.to_numeric(
        calibration["observed_joint_max_t_p"], errors="coerce"
    )
    calibration["observed_joint_student_t"] = pd.to_numeric(
        calibration["observed_joint_student_t"], errors="coerce"
    )
    if (
        calibration[["candidate_id", "metric"]].duplicated().any()
        or set(calibration.metric.astype(str)) != set(SAMPLING_NULL_METRICS)
        or set(calibration.evidence_label.astype(str)).difference(SAMPLING_NULL_LABELS)
        or set(calibration.selection_origin.astype(str)).difference(
            SAMPLING_NULL_SELECTION_ORIGINS
        )
        or set(calibration.gate_reason.astype(str)).difference(
            {
                "none",
                "sensitivity_origin",
                "support_failure",
                "sensitivity_origin_and_support_failure",
            }
        )
        or set(calibration.claim_boundary.astype(str))
        != {SAMPLING_NULL_CLAIM_BOUNDARY}
        or atlas.n_worms != 17
        or not (calibration.n_worms == atlas.n_worms).all()
    ):
        raise ReportInputError("sampling-null calibration row inventory failed")
    metric_sets = calibration.groupby("candidate_id").metric.agg(
        lambda values: set(values.astype(str))
    )
    if (
        set(metric_sets.index.astype(str)) != set(candidate_ids)
        or not metric_sets.map(lambda value: value == set(SAMPLING_NULL_METRICS)).all()
    ):
        raise ReportInputError("sampling-null calibration metric grid is incomplete")
    for row in calibration.to_dict(orient="records"):
        candidate_id = str(row["candidate_id"])
        if candidate_id not in selected_by_id or _sampling_cell_key(row) != _sampling_cell_key(
            selected_by_id[candidate_id]
        ):
            raise ReportInputError("sampling-null calibration cell mapping failed")
        metric = str(row["metric"])
        expected_signed = metric != "endpoint_wasserstein1"
        observed_p = row["observed_joint_max_t_p"]
        observed_t = row["observed_joint_student_t"]
        if (
            bool(row["context_signed"]) is not expected_signed
            or (expected_signed and not np.isfinite(observed_p))
            or (expected_signed and not np.isfinite(observed_t))
            or (not expected_signed and not pd.isna(observed_p))
            or (not expected_signed and not pd.isna(observed_t))
        ):
            raise ReportInputError("sampling-null signed-metric contract failed")

    sign_patterns = int(summary.get("joint_max_t_sign_patterns", -1))
    p_floor = 1.0 / sign_patterns if sign_patterns > 0 else math.inf
    bounded_columns = (
        "sampling_excess_joint_max_t_p",
        "temporal_specificity_joint_max_t_p",
    )
    if sign_patterns != 65_536 or any(
        not calibration[column].between(p_floor, 1.0).all()
        for column in bounded_columns
    ):
        raise ReportInputError("sampling-null joint max-T p-values are invalid")
    signed_p = calibration.loc[
        calibration.context_signed, "observed_joint_max_t_p"
    ]
    if not signed_p.between(p_floor, 1.0).all():
        raise ReportInputError("sampling-null observed joint max-T p-values are invalid")

    support = pd.read_csv(root / "support_diagnostics.csv")
    required_support_columns = {
        "candidate_id",
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
        "context",
        "channel",
        "selection_origin",
        "observed_valid_fraction",
        "observed_minimum_ancestor_fraction",
        "observed_genealogy_valid_fraction_0_10",
        "sampling_valid_fraction",
        "sampling_minimum_ancestor_fraction",
        "sampling_genealogy_valid_fraction_0_10",
        "pseudo_valid_fraction",
        "pseudo_minimum_ancestor_fraction",
        "pseudo_genealogy_valid_fraction_0_10",
    }
    if set(required_support_columns).difference(support.columns) or len(support) != 8:
        raise ReportInputError("sampling-null support diagnostic schema/count failed")
    if support.candidate_id.astype(str).duplicated().any():
        raise ReportInputError("sampling-null support diagnostic keys are duplicated")
    support_by_id = {str(row["candidate_id"]): row for row in support.to_dict("records")}
    fraction_columns = (
        "observed_valid_fraction",
        "observed_minimum_ancestor_fraction",
        "observed_genealogy_valid_fraction_0_10",
        "sampling_valid_fraction",
        "sampling_minimum_ancestor_fraction",
        "sampling_genealogy_valid_fraction_0_10",
        "pseudo_valid_fraction",
        "pseudo_minimum_ancestor_fraction",
        "pseudo_genealogy_valid_fraction_0_10",
    )
    for column in fraction_columns:
        support[column] = pd.to_numeric(support[column], errors="coerce")
        if not support[column].between(0.0, 1.0).all():
            raise ReportInputError(f"sampling-null support {column} is invalid")
    for candidate_id, selected_row in selected_by_id.items():
        if candidate_id not in support_by_id or _sampling_cell_key(
            support_by_id[candidate_id]
        ) != _sampling_cell_key(selected_row):
            raise ReportInputError("sampling-null support cell mapping failed")

    gate = protocol.get("support_gate")
    try:
        support_threshold = float(gate["valid_fraction_threshold"])
        genealogy_threshold = float(gate["genealogy_valid_fraction_threshold"])
        gate_valid = (
            gate.get("lag_sensitivity_rows_forced_sampling_limited") is True
            and tuple(gate.get("contexts_required", ()))
            == ("observed", "sampling_controls", "quiet_pseudo")
            and tuple(gate.get("sampling_controls_required", ()))
            == ("low_low", "high_high", "midpoint_midpoint")
            and "minimum across" in str(
                gate.get("sampling_control_fraction_reduction", "")
            )
            and 0 < support_threshold <= 1
            and 0 < genealogy_threshold <= 1
            and float(manifest_config.get("strong_valid_fraction", -1))
            == support_threshold
        )
    except (KeyError, TypeError, ValueError):
        gate_valid = False
    if not gate_valid:
        raise ReportInputError("sampling-null support-gate protocol failed")
    alpha = float(protocol["alpha"])
    for row in calibration.itertuples(index=False):
        support_row = support.loc[
            support.candidate_id.astype(str) == str(row.candidate_id)
        ].iloc[0]
        for fraction_column in (
            "observed_valid_fraction",
            "observed_genealogy_valid_fraction_0_10",
            "sampling_valid_fraction",
            "sampling_genealogy_valid_fraction_0_10",
            "pseudo_valid_fraction",
            "pseudo_genealogy_valid_fraction_0_10",
        ):
            if not math.isclose(
                float(getattr(row, fraction_column)),
                float(support_row[fraction_column]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ReportInputError(
                    "sampling-null calibration/support fractions disagree"
                )
        support_pass = (
            float(support_row.observed_valid_fraction) >= support_threshold
            and float(support_row.observed_genealogy_valid_fraction_0_10)
            >= genealogy_threshold
            and float(support_row.sampling_valid_fraction) >= support_threshold
            and float(support_row.sampling_genealogy_valid_fraction_0_10)
            >= genealogy_threshold
            and float(support_row.pseudo_valid_fraction) >= support_threshold
            and float(support_row.pseudo_genealogy_valid_fraction_0_10)
            >= genealogy_threshold
        )
        selection_eligible = str(row.selection_origin) != "lag_sensitivity"
        if not selection_eligible and not support_pass:
            expected_gate_reason = "sensitivity_origin_and_support_failure"
        elif not selection_eligible:
            expected_gate_reason = "sensitivity_origin"
        elif not support_pass:
            expected_gate_reason = "support_failure"
        else:
            expected_gate_reason = "none"
        if (
            bool(row.support_pass) is not support_pass
            or bool(row.selection_eligible) is not selection_eligible
            or str(row.gate_reason) != expected_gate_reason
        ):
            raise ReportInputError(
                "sampling-null durable gate metadata does not reproduce"
            )
        if not selection_eligible or not support_pass:
            expected_label = "sampling_limited"
        else:
            observed_pass = (not bool(row.context_signed)) or (
                float(row.observed_joint_max_t_p) <= alpha
                and (
                    float(row.observed_ci_2_5) > 0
                    or float(row.observed_ci_97_5) < 0
                )
            )
            sampling_pass = (
                float(row.sampling_excess_mean) > 0
                and float(row.sampling_excess_ci_2_5) > 0
                and float(row.sampling_excess_joint_max_t_p) <= alpha
            )
            quiet_pass = (
                float(row.temporal_specificity_excess_mean) > 0
                and float(row.temporal_specificity_ci_2_5) > 0
                and float(row.temporal_specificity_joint_max_t_p) <= alpha
            )
            if observed_pass and sampling_pass and quiet_pass:
                expected_label = "exceeds_sampling_and_quiet_controls"
            elif observed_pass and sampling_pass:
                expected_label = "exceeds_sampling_controls_only"
            else:
                expected_label = "indistinguishable_from_sampling_controls"
        if str(row.evidence_label) != expected_label:
            raise ReportInputError("sampling-null evidence label does not reproduce")

    expected_counts = {
        str(key): int(value)
        for key, value in calibration.evidence_label.value_counts().items()
    }
    if (
        int(summary.get("selected_cells", -1)) != 8
        or int(summary.get("candidate_metric_rows", -1)) != 24
        or int(summary.get("worms", -1)) != atlas.n_worms
        or int(summary.get("joint_max_t_tests", -1)) != 64
        or float(summary.get("alpha", -1)) != alpha
        or summary.get("labels") != expected_counts
        or summary.get("claim_boundary") != SAMPLING_NULL_CLAIM_BOUNDARY
    ):
        raise ReportInputError("sampling-null summary contract failed")

    pseudo = pd.read_csv(root / "pseudo_boundary_diagnostics.csv")
    if "quiet_verified" not in pseudo.columns or pseudo.empty or not _strict_bool_series(
        pseudo["quiet_verified"], label="sampling-null quiet_verified"
    ).all():
        raise ReportInputError("sampling-null quiet pseudo-boundary audit failed")

    with np.load(root / "null_inference_arrays.npz", allow_pickle=False) as arrays:
        required_arrays = {
            "candidate_metric_id",
            "worm_ids",
            "observed_worm_value",
            "sampling_null_worm_magnitude",
            "sampling_control_names",
            "sampling_control_worm_magnitude",
            "quiet_pseudo_worm_magnitude",
            "joint_test_id",
            "joint_test_kind",
            "joint_test_row_index",
            "joint_test_observed_t",
            "joint_test_max_t_p_value",
            "joint_null_max_abs_t",
            "joint_sign_patterns",
            "joint_simultaneous_critical_value",
        }
        if required_arrays.difference(arrays.files):
            raise ReportInputError("sampling-null inference arrays are incomplete")
        metric_ids = tuple(arrays["candidate_metric_id"].astype(str))
        worm_ids = tuple(arrays["worm_ids"].astype(str))
        expected_metric_ids = tuple(
            f"{row.candidate_id}:{row.metric}"
            for row in calibration.itertuples(index=False)
        )
        joint_p = np.asarray(arrays["joint_test_max_t_p_value"], dtype=float)
        joint_t = np.asarray(arrays["joint_test_observed_t"], dtype=float)
        joint_null = np.asarray(arrays["joint_null_max_abs_t"], dtype=float)
        joint_critical = float(
            np.asarray(arrays["joint_simultaneous_critical_value"]).item()
        )
        joint_kinds = np.asarray(arrays["joint_test_kind"]).astype(str)
        joint_rows = np.asarray(arrays["joint_test_row_index"], dtype=int)
        joint_ids = np.asarray(arrays["joint_test_id"]).astype(str)
        sign_matrix = np.asarray(arrays["joint_sign_patterns"])
        observed_worm = np.asarray(arrays["observed_worm_value"], dtype=float)
        sampling_worm = np.asarray(
            arrays["sampling_null_worm_magnitude"], dtype=float
        )
        named_sampling_worm = np.asarray(
            arrays["sampling_control_worm_magnitude"], dtype=float
        )
        quiet_worm = np.asarray(arrays["quiet_pseudo_worm_magnitude"], dtype=float)
        if (
            observed_worm.shape != (24, 17)
            or sampling_worm.shape != (24, 17)
            or named_sampling_worm.shape != (24, 3, 17)
            or quiet_worm.shape != (24, 17)
        ):
            raise ReportInputError("sampling-null inference-array shape contract failed")
        expected_joint_ids: list[str] = []
        expected_joint_kinds: list[str] = []
        expected_joint_rows: list[int] = []
        expected_joint_p: list[float] = []
        expected_joint_t: list[float] = []
        expected_joint_values: list[np.ndarray] = []
        signed_mask = calibration.context_signed.to_numpy(dtype=bool)
        observed_magnitude = np.where(
            signed_mask[:, None], np.abs(observed_worm), observed_worm
        )
        sampling_excess_worm = observed_magnitude - sampling_worm
        quiet_excess_worm = observed_magnitude - quiet_worm
        for row_index, row in enumerate(calibration.itertuples(index=False)):
            metric_id = f"{row.candidate_id}:{row.metric}"
            if bool(row.context_signed):
                expected_joint_ids.append(f"{metric_id}:observed_signed")
                expected_joint_kinds.append("observed_signed")
                expected_joint_rows.append(row_index)
                expected_joint_p.append(float(row.observed_joint_max_t_p))
                expected_joint_t.append(float(row.observed_joint_student_t))
                expected_joint_values.append(observed_worm[row_index])
            for kind, p_column, t_column in (
                (
                    "sampling_excess",
                    "sampling_excess_joint_max_t_p",
                    "sampling_excess_joint_student_t",
                ),
                (
                    "temporal_specificity_excess",
                    "temporal_specificity_joint_max_t_p",
                    "temporal_specificity_joint_student_t",
                ),
            ):
                expected_joint_ids.append(f"{metric_id}:{kind}")
                expected_joint_kinds.append(kind)
                expected_joint_rows.append(row_index)
                expected_joint_p.append(float(getattr(row, p_column)))
                expected_joint_t.append(float(getattr(row, t_column)))
                expected_joint_values.append(
                    sampling_excess_worm[row_index]
                    if kind == "sampling_excess"
                    else quiet_excess_worm[row_index]
                )
        array_valid = (
            len(metric_ids) == 24
            and metric_ids == expected_metric_ids
            and len(set(metric_ids)) == 24
            and worm_ids == raw_cohort_worms
            and len(set(worm_ids)) == 17
            and tuple(arrays["sampling_control_names"].astype(str))
            == ("low_low", "high_high", "midpoint_midpoint")
            and observed_worm.shape == (24, 17)
            and sampling_worm.shape == (24, 17)
            and named_sampling_worm.shape == (24, 3, 17)
            and quiet_worm.shape == (24, 17)
            and all(
                np.isfinite(value).all()
                for value in (
                    observed_worm,
                    sampling_worm,
                    named_sampling_worm,
                    quiet_worm,
                )
            )
            and bool((sampling_worm >= 0).all())
            and bool((named_sampling_worm >= 0).all())
            and bool((quiet_worm >= 0).all())
            and np.allclose(
                sampling_worm,
                named_sampling_worm.max(axis=1),
                rtol=0.0,
                atol=1e-7,
            )
            and joint_ids.shape == (64,)
            and len(set(joint_ids)) == 64
            and tuple(joint_ids) == tuple(expected_joint_ids)
            and joint_kinds.shape == (64,)
            and tuple(joint_kinds) == tuple(expected_joint_kinds)
            and int(np.sum(joint_kinds == "observed_signed")) == 16
            and int(np.sum(joint_kinds == "sampling_excess")) == 24
            and int(np.sum(joint_kinds == "temporal_specificity_excess")) == 24
            and joint_rows.shape == (64,)
            and tuple(joint_rows) == tuple(expected_joint_rows)
            and bool(np.all((joint_rows >= 0) & (joint_rows < 24)))
            and sign_matrix.shape == (65_536, 17)
            and bool(np.isin(sign_matrix, (-1, 1)).all())
            and bool((sign_matrix[:, 0] == 1).all())
            and joint_null.shape == (65_536,)
            and np.isfinite(joint_null).all()
            and joint_t.shape == (64,)
            and np.isfinite(joint_t).all()
            and math.isfinite(joint_critical)
            and joint_p.shape == (64,)
            and np.isfinite(joint_p).all()
            and bool(np.all((joint_p >= p_floor) & (joint_p <= 1.0)))
            and np.allclose(
                joint_p,
                np.asarray(expected_joint_p, dtype=float),
                rtol=0.0,
                atol=1e-7,
            )
            and np.allclose(
                joint_t,
                np.asarray(expected_joint_t, dtype=float),
                rtol=1e-6,
                atol=1e-6,
            )
        )
        if not array_valid:
            raise ReportInputError("sampling-null inference-array contract failed")

        # Reconstruct every displayed scalar from the saved worm-level arrays.
        # The archive uses float32 for portability, whereas the analysis computes
        # in float64 before writing; tolerances cover only that storage rounding.
        scalar_atol = 2e-5
        for row_index, row in enumerate(calibration.itertuples(index=False)):
            observed = observed_worm[row_index]
            sampler = sampling_worm[row_index]
            named = named_sampling_worm[row_index]
            quiet = quiet_worm[row_index]
            sampler_excess = sampling_excess_worm[row_index]
            quiet_excess = quiet_excess_worm[row_index]
            key_seed = _sampling_bootstrap_seed(
                str(row.candidate_id), str(row.metric), str(bootstrap_seed)
            )
            observed_ci = _sampling_bootstrap_interval(
                observed, bootstrap_replicates, key_seed
            )
            sampler_ci = _sampling_bootstrap_interval(
                sampler_excess, bootstrap_replicates, key_seed + 1
            )
            quiet_ci = _sampling_bootstrap_interval(
                quiet_excess, bootstrap_replicates, key_seed + 2
            )
            expected_scalars = {
                "observed_mean": observed.mean(),
                "observed_ci_2_5": observed_ci[0],
                "observed_ci_97_5": observed_ci[1],
                "sampling_null_mean_magnitude": sampler.mean(),
                "sampling_null_p95_worm": np.quantile(sampler, 0.95),
                "low_low_mean_magnitude": named[0].mean(),
                "high_high_mean_magnitude": named[1].mean(),
                "midpoint_midpoint_mean_magnitude": named[2].mean(),
                "sampling_excess_mean": sampler_excess.mean(),
                "sampling_excess_ci_2_5": sampler_ci[0],
                "sampling_excess_ci_97_5": sampler_ci[1],
                "quiet_pseudo_mean_magnitude": quiet.mean(),
                "quiet_pseudo_p95_worm": np.quantile(quiet, 0.95),
                "temporal_specificity_excess_mean": quiet_excess.mean(),
                "temporal_specificity_ci_2_5": quiet_ci[0],
                "temporal_specificity_ci_97_5": quiet_ci[1],
                "observed_to_sampling_null_ratio": (
                    observed_magnitude[row_index].mean()
                    / max(sampler.mean(), 1e-12)
                ),
                "observed_to_quiet_pseudo_ratio": (
                    observed_magnitude[row_index].mean()
                    / max(quiet.mean(), 1e-12)
                ),
            }
            for column, expected in expected_scalars.items():
                if not np.isclose(
                    float(getattr(row, column)),
                    float(expected),
                    rtol=2e-5,
                    atol=scalar_atol,
                ):
                    raise ReportInputError(
                        f"sampling-null CSV/worm-array reduction failed for {column}"
                    )

        (
            recomputed_p,
            recomputed_t,
            recomputed_null,
            recomputed_critical,
            recomputed_signs,
        ) = _sampling_joint_studentized_max_t(
            np.asarray(expected_joint_values, dtype=np.float64), alpha=alpha
        )
        p_rounding_atol = 1e-7
        mathematical_valid = (
            np.array_equal(sign_matrix, recomputed_signs)
            and np.allclose(joint_t, recomputed_t, rtol=5e-5, atol=5e-5)
            and np.allclose(
                joint_null, recomputed_null, rtol=5e-5, atol=5e-5
            )
            and np.allclose(
                joint_p,
                recomputed_p,
                rtol=0.0,
                atol=p_rounding_atol,
            )
            and math.isclose(
                joint_critical,
                recomputed_critical,
                rel_tol=5e-5,
                abs_tol=5e-5,
            )
            and math.isclose(
                float(summary.get("joint_max_t_critical_value", math.nan)),
                recomputed_critical,
                rel_tol=5e-5,
                abs_tol=5e-5,
            )
        )
        if not mathematical_valid:
            raise ReportInputError(
                "sampling-null exact max-T mathematical contract failed"
            )

    return SamplingNullEvidence(
        root=root,
        raw_root=raw_root,
        manifest=manifest,
        protocol=protocol,
        summary=summary,
        validation=validation,
        calibration=calibration,
        support=support,
        checksums=checksums,
    )


def _string_axis(value: np.ndarray, *, label: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or not len(array):
        raise ReportInputError(f"dense atlas {label} axis must be nonempty and one-dimensional")
    result = tuple(str(item) for item in array)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ReportInputError(f"dense atlas {label} axis is empty or duplicated")
    return result


def _integer_axis(value: np.ndarray, *, label: str) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or not len(array):
        raise ReportInputError(f"dense atlas {label} axis must be nonempty and one-dimensional")
    numeric = pd.to_numeric(pd.Series(array), errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
        or (numeric <= 0).any()
    ):
        raise ReportInputError(f"dense atlas {label} axis must contain positive integers")
    result = tuple(int(item) for item in numeric)
    if len(set(result)) != len(result):
        raise ReportInputError(f"dense atlas {label} axis is duplicated")
    return result


def _quantize_slice(matrix: np.ndarray) -> dict[str, Any]:
    """Quantize one target-row/source-column slice symmetrically to int16.

    C-order flattening is deliberate: byte position ``target * n + source`` is
    the value at target row and source column.  ``-32768`` is unused so the
    positive and negative ranges are symmetric around zero.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not values.size:
        raise ReportInputError("a dense normalized-mean slice is not a square matrix")
    if not np.isfinite(values).all():
        raise ReportInputError("a dense normalized-mean slice contains non-finite values")
    maximum = float(np.max(np.abs(values)))
    scale = maximum / 32767.0 if maximum > 0 else 1.0
    quantized = np.rint(values / scale).clip(-32767, 32767).astype("<i2")
    return {
        "encoding": ENCODING,
        "scale": scale,
        "zero_point": 0,
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "order": "C_target_row_source_column",
        "data": base64.b64encode(quantized.tobytes(order="C")).decode("ascii"),
    }


def _dense_atlas_payload(atlas: AtlasEvidence) -> dict[str, Any]:
    dense_path = atlas.root / "atlas_matrices.npz"
    with np.load(dense_path, allow_pickle=False) as dense:
        required_metadata = {
            "orientation",
            "neurons",
            "methods",
            "channels",
            "contexts",
            "source_lag_frames",
            "horizon_frames",
        }
        missing_metadata = sorted(required_metadata.difference(dense.files))
        if missing_metadata:
            raise ReportInputError(
                f"dense atlas lacks explorer metadata arrays {missing_metadata}"
            )
        if str(np.asarray(dense["orientation"]).item()) != ORIENTATION:
            raise ReportInputError("dense atlas explorer orientation contract failed")
        neurons = _string_axis(dense["neurons"], label="neurons")
        methods = _string_axis(dense["methods"], label="methods")
        channels = _string_axis(dense["channels"], label="channels")
        contexts = _string_axis(dense["contexts"], label="contexts")
        lags = _integer_axis(dense["source_lag_frames"], label="source lag")
        horizons = _integer_axis(dense["horizon_frames"], label="horizon")
        dashboard_contexts = tuple(
            str(item.get("context"))
            for item in atlas.dashboard.get("contexts", ())
            if isinstance(item, Mapping)
        )
        protocol_lags = tuple(int(value) for value in atlas.protocol.get("source_lag_frames", ()))
        protocol_horizons = tuple(int(value) for value in atlas.protocol.get("horizon_frames", ()))
        if (
            neurons != atlas.neurons
            or methods != tuple(str(value) for value in atlas.dashboard.get("methods", ()))
            or channels != tuple(str(value) for value in atlas.dashboard.get("channels", ()))
            or contexts != dashboard_contexts
            or lags != protocol_lags
            or horizons != protocol_horizons
        ):
            raise ReportInputError("dense atlas axes disagree with reviewed atlas metadata")
        expected_keys = {
            f"mean_normalized__{method}__{channel}__{context}"
            for method in methods
            for channel in channels
            for context in contexts
        }
        observed_keys = {
            key for key in dense.files if key.startswith("mean_normalized__")
        }
        missing = sorted(expected_keys.difference(observed_keys))
        unexpected = sorted(observed_keys.difference(expected_keys))
        if missing or unexpected:
            raise ReportInputError(
                "dense normalized-mean inventory differs from its metadata "
                f"(missing={missing[:3]}, unexpected={unexpected[:3]})"
            )
        shape = (len(lags), len(horizons), len(neurons), len(neurons))
        slices: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
        slice_count = 0
        for method in methods:
            method_slices: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
            for channel in channels:
                channel_slices: dict[str, dict[str, dict[str, Any]]] = {}
                for context in contexts:
                    key = f"mean_normalized__{method}__{channel}__{context}"
                    array = np.asarray(dense[key])
                    if array.shape != shape:
                        raise ReportInputError(
                            f"{key} has shape {array.shape}; expected {shape}"
                        )
                    if not np.isfinite(array).all():
                        raise ReportInputError(f"{key} contains non-finite values")
                    lag_slices: dict[str, dict[str, Any]] = {}
                    for lag_index, lag in enumerate(lags):
                        horizon_slices: dict[str, Any] = {}
                        for horizon_index, horizon in enumerate(horizons):
                            horizon_slices[str(horizon)] = _quantize_slice(
                                array[lag_index, horizon_index]
                            )
                            slice_count += 1
                        lag_slices[str(lag)] = horizon_slices
                    channel_slices[context] = lag_slices
                method_slices[channel] = channel_slices
            slices[method] = method_slices
    context_options = []
    dashboard_by_context = {
        str(item.get("context")): item
        for item in atlas.dashboard.get("contexts", ())
        if isinstance(item, Mapping)
    }
    for context in contexts:
        metadata = dashboard_by_context.get(context, {})
        chemical = metadata.get("chemical")
        stratum = str(chemical) if chemical else "all_scheduled_events"
        prefix = f"{stratum}_"
        phase_or_contrast = (
            context[len(prefix) :]
            if stratum != "all_scheduled_events" and context.startswith(prefix)
            else context
        )
        context_options.append(
            {
                "context": context,
                "event_stratum": stratum,
                "phase_or_contrast": phase_or_contrast,
                "event_stratified": bool(metadata.get("event_stratified", False)),
            }
        )
    return {
        "orientation": ORIENTATION,
        "row_axis": "target_neuron",
        "column_axis": "source_neuron",
        "internal_array_axes": [
            "source_lag_frames",
            "horizon_frames",
            "target_neuron",
            "source_neuron",
        ],
        "neurons": list(neurons),
        "methods": list(methods),
        "channels": list(channels),
        "contexts": list(contexts),
        "context_options": context_options,
        "source_lag_frames": list(lags),
        "horizon_frames": list(horizons),
        "n_neurons": len(neurons),
        "slice_count": slice_count,
        "encoding": {
            "name": ENCODING,
            "quantization": "round(value / per_slice_scale), symmetric [-32767,32767]",
            "decode": "value = int16_little_endian * per_slice_scale",
            "maximum_absolute_error": (
                "per_slice_scale / 2, apart from final float32 browser decode rounding"
            ),
            "flattening": "C order: target row first, source column second",
        },
        "slices": slices,
    }


def _optional_stimulus_composition(atlas: AtlasEvidence) -> dict[str, Any]:
    """Expose the compact timing summary only with raw-file manifest/ledger lineage."""

    artifacts = atlas.manifest.get("artifacts")
    declared = isinstance(artifacts, Mapping) and (
        artifacts.get("stimulus_composition") == OPTIONAL_COMPOSITION_FILE
        or OPTIONAL_COMPOSITION_FILE in {str(value) for value in artifacts.values()}
    )
    ledgered = OPTIONAL_COMPOSITION_FILE in atlas.checksums
    path = atlas.root / OPTIONAL_COMPOSITION_FILE
    exists = path.is_file()
    if not any((declared, ledgered, exists)):
        return {"status": "not_declared", "columns": [], "rows": []}
    if not all((declared, ledgered, exists)):
        raise ReportInputError(
            "optional stimulus composition must be declared by the manifest, "
            "covered by checksums.sha256, and present on disk"
        )
    if sha256(path) != atlas.checksums[OPTIONAL_COMPOSITION_FILE]:
        raise ReportInputError("optional stimulus composition checksum changed")
    raw_summary = atlas.dashboard.get("stimulus_composition_summary")
    if (
        not isinstance(raw_summary, list)
        or not raw_summary
        or not all(isinstance(row, Mapping) for row in raw_summary)
    ):
        raise ReportInputError(
            "declared stimulus composition lacks its compact dashboard summary"
        )
    rows = [_json_safe(dict(row)) for row in raw_summary]
    required = {"phase", "source_lag_frames", "horizon_frames"}
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if required.difference(columns):
        raise ReportInputError("stimulus composition summary lacks timing keys")
    keys: set[tuple[str, int, int]] = set()
    for row in rows:
        try:
            key = (
                str(row["phase"]),
                int(row["source_lag_frames"]),
                int(row["horizon_frames"]),
            )
        except (TypeError, ValueError) as error:
            raise ReportInputError(
                "stimulus composition summary timing keys are invalid"
            ) from error
        if key in keys:
            raise ReportInputError("stimulus composition summary duplicates a timing cell")
        keys.add(key)
    phases = {key[0] for key in keys}
    canonical_phases = {"baseline", "onset", "active", "offset", "recovery"}
    if phases != canonical_phases:
        raise ReportInputError(
            "stimulus composition summary does not cover the canonical phase axis"
        )
    expected = {
        (phase, lag, horizon)
        for phase in phases
        for lag in atlas.protocol.get("source_lag_frames", ())
        for horizon in atlas.protocol.get("horizon_frames", ())
    }
    if keys != expected:
        raise ReportInputError("stimulus composition summary timing grid is incomplete")
    protocol_composition = atlas.protocol.get("stimulus_composition")
    return {
        "status": "included",
        "file": OPTIONAL_COMPOSITION_FILE,
        "sha256": atlas.checksums[OPTIONAL_COMPOSITION_FILE],
        "representation": "compact_phase_by_lag_by_horizon_summary",
        "raw_grain": (
            protocol_composition.get("grain")
            if isinstance(protocol_composition, Mapping)
            else "worm x phase x event-position x source-lag x forecast-horizon"
        ),
        "raw_row_count": (
            protocol_composition.get("row_count")
            if isinstance(protocol_composition, Mapping)
            else None
        ),
        "columns": columns,
        "rows": rows,
    }


def _claim_boundaries(
    atlas: AtlasEvidence,
    targeted: TargetedEvidence,
    complete_family: CompleteFamilyEvidence,
    sampling_null: SamplingNullEvidence,
) -> list[dict[str, str]]:
    factual = str(
        targeted.protocol.get(
            "factual_arm",
            targeted.protocol.get(
                "factual_definition",
                "free conditional-generator rollout from observed history at the cut",
            ),
        )
    )
    raw_implementation = sampling_null.protocol.get(
        "raw_implementation_provenance", {}
    )
    if isinstance(raw_implementation, Mapping):
        raw_implementation_boundary = str(
            raw_implementation.get(
                "boundary",
                "the v1 raw manifest did not record source-file or runtime hashes",
            )
        )
    else:
        raw_implementation_boundary = (
            "the v1 raw manifest did not record source-file or runtime hashes"
        )
    return [
        {
            "label": "Model-relative and noncausal",
            "detail": (
                "Values are fitted-generator lag associations under repaired model laws; "
                "they are not causal effects, anatomical connections, receptor actions, "
                "or experimental confirmation."
            ),
        },
        {
            "label": "Chemical contexts are event-stratified under binary-any-stimulus",
            "detail": (
                "Chemical labels stratify observed scheduled events. The generator uses "
                "binary_any_stimulus and is not chemical-identity conditioned."
            ),
        },
        {
            "label": "Lag is not a physical delay",
            "detail": (
                "Source lag is a source-window location relative to the cut; forecast horizon "
                "is a separate readout interval. Neither estimates propagation time."
            ),
        },
        {
            "label": "Factual arm model rollout",
            "detail": f"The factual arm is a model rollout, specifically: {factual}.",
        },
        {
            "label": "Sampling-null calibration is selection-conditioned",
            "detail": (
                "The N128 checks compare eight upstream-selected cells with the worst "
                "of three sampler controls and a quiet-time pseudo-boundary. They are "
                "model-relative robustness diagnostics, not a biological null."
            ),
        },
        {
            "label": "Exact calibration p-values require joint sign symmetry",
            "detail": (
                "Exact enumeration is conditional on joint worm-vector sign symmetry. "
                "Overlapping cross-fit training sets can couple held-out estimates, so "
                "these are model-relative calibration p-values, not experimental "
                "randomization significance."
            ),
        },
        {
            "label": "Raw v1 implementation provenance is incomplete",
            "detail": (
                f"Raw-bundle provenance note: {raw_implementation_boundary}. Current source hashes "
                "cannot retroactively prove the bytes used to launch that run."
            ),
        },
        {
            "label": "No prediction is experiment-ready",
            "detail": (
                "Complete-family worm max-T and the targeted sampler calibration do "
                "not define a smallest effect size of interest or provide independent "
                "biological confirmation; those gates remain missing."
            ),
        },
        {
            "label": "Baseline quiet control is not stimulus modulation",
            "detail": (
                "All eight current calibration cells use baseline windows. Their quiet-"
                "time comparison asks about temporal specificity within the frozen "
                "model law; it does not test a stimulus-induced change."
            ),
        },
        {
            "label": "External references are post-freeze context only",
            "detail": (
                "External comparisons were computed after the internal queue was frozen and "
                "did not rank, select, or relabel internal candidates."
            ),
        },
    ]


def _input_provenance(
    atlas: AtlasEvidence,
    targeted: TargetedEvidence,
    external: ExternalEvidence,
    complete_family: CompleteFamilyEvidence,
    sampling_null: SamplingNullEvidence,
    *,
    provenance_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for label, evidence in (
        ("canonical_atlas", atlas),
        ("targeted_n128", targeted),
        ("postfreeze_external", external),
        ("complete_family_evidence", complete_family),
        ("sampling_null_analysis", sampling_null),
    ):
        rows.append(
            {
                "bundle": label,
                "path": _safe_relative(evidence.root, provenance_root),
                "manifest_sha256": sha256(evidence.root / "manifest.json"),
                "checksums_sha256": sha256(evidence.root / "checksums.sha256"),
                "verified_files": len(evidence.checksums),
            }
        )
    return rows


def _build_payload(
    atlas: AtlasEvidence,
    targeted: TargetedEvidence,
    external: ExternalEvidence,
    complete_family: CompleteFamilyEvidence,
    sampling_null: SamplingNullEvidence,
    *,
    provenance_root: Path,
    generated_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "title": TITLE,
        "offline": True,
        "atlas": _dense_atlas_payload(atlas),
        "candidate_queue": _records(atlas.queue),
        "targeted_n128": {
            "summary": _json_safe(targeted.summary),
            "protocol": _json_safe(targeted.protocol),
            "cells": _records(targeted.cells),
            "quantile_shifts": _records(targeted.quantiles),
            "support_diagnostics": _records(targeted.support),
            "screen_consistency": _records(targeted.screen),
        },
        "external_references": {
            "analysis_role": external.manifest.get("analysis_role"),
            "ranking_effect": external.manifest.get("ranking_effect"),
            "comparisons": _records(external.comparisons),
            "lagmax_inference": _records(external.lagmax),
        },
        "statistical_evidence": _json_safe(complete_family.explorer),
        "sampling_null_calibration": {
            "summary": _json_safe(sampling_null.summary),
            "protocol": _json_safe(sampling_null.protocol),
            "validation": _json_safe(sampling_null.validation),
            "cells": _records(sampling_null.calibration),
            "support_diagnostics": _records(sampling_null.support),
            "interpretation": (
                "Selection-conditioned, model-relative N128 robustness calibration; "
                "not a biological null, causal effect, anatomical connection, receptor "
                "action, or physical delay. All current cells are baseline, so the "
                "quiet-time control is not a test of stimulus modulation."
            ),
        },
        "stimulus_composition": _optional_stimulus_composition(atlas),
        "claim_boundaries": _claim_boundaries(
            atlas, targeted, complete_family, sampling_null
        ),
        "provenance": _input_provenance(
            atlas,
            targeted,
            external,
            complete_family,
            sampling_null,
            provenance_root=provenance_root,
        ),
    }


def _prepare_output(output_dir: Path, *, overwrite: bool) -> Path:
    output = output_dir.resolve()
    if output.exists():
        if not output.is_dir():
            raise ReportInputError("explorer output path exists and is not a directory")
        existing = [path for path in output.iterdir() if path.name in OUTPUT_FILES]
        if existing and not overwrite:
            raise FileExistsError(f"explorer outputs already exist in {output}")
        if overwrite:
            for path in existing:
                if not path.is_file():
                    raise ReportInputError("an explorer output path is not a regular file")
            for path in existing:
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    return output


def _validate_output_location(output_dir: Path, input_roots: Sequence[Path]) -> None:
    output = output_dir.resolve()
    for root_value in input_roots:
        root = root_value.resolve()
        try:
            output.relative_to(root)
        except ValueError:
            continue
        raise ReportInputError(
            "explorer output must not equal or be nested inside a validated input bundle"
        )


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; connect-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Neural prediction atlas explorer</title>
<style>
:root{color-scheme:dark;--bg:#141719;--panel:#1b1f22;--panel2:#22282c;--line:#394247;--text:#ecefed;--muted:#adb7ba;--cyan:#a5cdc5;--amber:#d8b980;--blue:#4f90d9;--red:#d86155;--focus:#e4ca93;--max:1380px;font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--text);font-size:15px;line-height:1.55}
button,select{font:inherit}button{cursor:pointer}button:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid var(--focus);outline-offset:3px}
header{max-width:var(--max);margin:auto;padding:32px clamp(20px,4vw,52px) 22px}h1{font-size:30px;line-height:1.2;font-weight:650;margin:0 0 9px;letter-spacing:-.025em}h2{font-size:21px;line-height:1.3;font-weight:620;margin:0 0 10px}h3{font-size:16px;line-height:1.4;font-weight:620;margin:0 0 8px}p{margin-top:0}.eyebrow{color:var(--muted);font-size:12px;margin-bottom:7px}.sub{color:var(--muted);max-width:76ch;margin-bottom:7px}.header-boundary{color:var(--muted);font-size:12px;margin:0}
.tab-wrap{border-bottom:1px solid var(--line)}.tabs{display:flex;gap:26px;max-width:var(--max);margin:auto;padding:0 clamp(20px,4vw,52px)}.tab{flex:0 1 auto;padding:13px 0;border:0;border-bottom:2px solid transparent;background:none;color:var(--muted);font-size:14px;min-height:46px}.tab[aria-selected="true"]{border-color:var(--cyan);color:var(--text);font-weight:650}.tab:hover{color:var(--text)}
main{max-width:var(--max);margin:auto;padding:26px clamp(20px,4vw,52px) 60px}.view-panel{display:grid;gap:22px}.view-panel[hidden]{display:none}.explore-top{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,.9fr);gap:22px;align-items:start}.grid{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(280px,.8fr);gap:22px}
.card{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:24px}.wide{grid-column:1/-1}.card-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:16px}.meta{color:var(--muted);font-size:13px;max-width:86ch}.pill{display:inline-flex;max-width:100%;color:var(--muted);font-size:12px;white-space:normal;overflow-wrap:anywhere}
.control-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:18px}.control-grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}label{display:grid;align-content:start;gap:7px;color:var(--text);font-weight:500;font-size:13px}select{width:100%;min-width:0;background:#14191c;color:var(--text);border:1px solid #4a555b;border-radius:4px;padding:10px;min-height:44px}.help{color:var(--muted);font-size:12px;font-weight:400;line-height:1.45}.control-help{margin:8px 0 0;font-size:12px;color:var(--muted)}
details{min-width:0;border:1px solid var(--line);border-radius:4px;background:transparent}summary{padding:13px 15px;font-weight:500;color:var(--text);cursor:pointer;min-height:44px}details[open]>summary{border-bottom:1px solid var(--line)}.details-body{padding:16px}.details-body>:last-child{margin-bottom:0}.advanced{margin-top:20px}.control-definitions{margin:14px 0 0;color:var(--muted);font-size:13px}.control-definitions dt{color:var(--text);margin-top:10px}.control-definitions dd{margin:3px 0 0}
.inline-note{color:var(--muted);font-size:13px}.keyboard-note{color:var(--muted);font-size:13px;margin:12px 0 0}.result-card{border-top:2px solid #819e97}.result-kicker{color:var(--muted);font-size:12px;margin-bottom:8px}.result-question{font-size:14px;color:var(--muted);margin:8px 0 22px}.result-sentence{font-size:22px;line-height:1.5;margin:0 0 20px;max-width:42ch}.result-sentence strong{font-weight:600}.result-value{color:var(--text);font-weight:650;font-variant-numeric:tabular-nums}.result-context{display:flex;flex-wrap:wrap;gap:6px 14px;color:var(--muted);font-size:12px}.context-chip{display:inline}.boundary-line{margin:20px 0 0;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
.evidence-card details,.evidence-card .details-body,.evidence-card .scroll{min-width:0;max-width:100%}.evidence-card .details-body{overflow:hidden}.evidence-lead{color:var(--muted);font-size:13px}.calibration-picker{margin:14px 0 6px}.calibration-picks{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.calibration-pick{min-height:52px;text-align:left;padding:9px 11px;border:1px solid #4a555b;border-radius:4px;background:#1d2327;color:var(--text);font-size:13px}.calibration-pick small{display:block;color:var(--muted);font-size:11px;margin-top:3px}.calibration-pick[aria-pressed="true"]{border-color:var(--cyan);background:#25332f}.evidence-current{margin:0}.evidence-status{display:inline-flex;align-items:center;max-width:100%;padding:4px 8px;border:1px solid var(--line);border-radius:4px;color:var(--text);font-size:13px;font-weight:550}.evidence-status.strong{border-color:#6c9385;color:#b8d6c8}.evidence-status.exploratory{border-color:#8b7955;color:#e2c78f}.evidence-status.limited{border-color:#806761;color:#dcb9ad}.evidence-copy{margin:11px 0 18px;max-width:86ch;color:var(--muted);font-size:13px}
.evidence-ladder{list-style:none;padding:0;margin:0 0 18px;border-top:1px solid var(--line)}.evidence-rung{border-bottom:1px solid var(--line)}.evidence-rung details{border:0;border-radius:0}.evidence-rung summary{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 18px;padding:16px 0;list-style:none}.evidence-rung summary::-webkit-details-marker{display:none}.rung-title{font-size:14px;font-weight:550}.rung-title::before{content:"+";display:inline-block;width:20px;color:var(--muted);font-weight:400}.evidence-rung details[open] .rung-title::before{content:"−"}.rung-brief{grid-column:1/-1;padding-left:20px;color:var(--muted);font-size:13px;font-weight:400}.rung-detail{color:var(--muted);font-size:13px;margin:0;padding:14px 20px 18px;max-width:95ch}.rung-state{align-self:start;padding:2px 6px;border-radius:3px;font-size:12px;font-weight:500;white-space:normal;overflow-wrap:anywhere;text-align:right}.rung-state.pass{color:#b7d9c6;background:#263b31}.rung-state.caution{color:#e2c78f;background:#3a3223}.rung-state.fail{color:#dfb8ab;background:#3a2b27}.rung-state.missing{color:#b7bfc4;background:#2b3136}
.evidence-headline{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;margin:6px 0 20px}.evidence-stat strong{display:block;font-size:24px;line-height:1.2;font-weight:550;font-variant-numeric:tabular-nums}.evidence-stat span{display:block;color:var(--muted);font-size:12px;margin-top:5px}.evidence-note{color:var(--muted);font-size:13px;margin:12px 0 18px}.evidence-table th,.evidence-table td{white-space:normal}.evidence-table .number{text-align:right}.secondary-details{margin-top:12px}.screen-disclosure{margin-top:14px}.screen-warning{color:#d8bd89;margin:0 0 13px}.queue-evidence{font-weight:500}
.canvas-wrap{position:relative;max-width:700px;margin:auto}canvas{display:block;width:100%;height:auto;background:#141719;border:1px solid var(--line);image-rendering:pixelated;cursor:crosshair}.axis{text-align:center;color:var(--muted);font-size:12px;margin:10px 0}.readout{font-variant-numeric:tabular-nums;color:var(--text);padding:12px 0;border-top:1px solid var(--line);margin-top:14px;min-height:48px;font-size:13px}.legend{display:grid;grid-template-columns:auto minmax(150px,380px) auto;gap:10px;align-items:center;justify-content:center;margin:14px auto 3px;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}.legend-center{display:grid;gap:5px}.legend-bar{height:9px;border:1px solid #626a70;border-radius:2px;background:linear-gradient(90deg,#407fc2,#eff4f7 50%,#d85244)}.legend-bar.unsigned{background:linear-gradient(90deg,#eff4f7,#c28934)}.legend-labels{display:flex;justify-content:space-between}.scale-note{text-align:center;color:var(--muted);font-size:12px;max-width:70ch;margin:7px auto 0}
.surface{overflow:auto}.surface table{min-width:680px}.surface button{border:1px solid rgba(255,255,255,.2);min-width:82px;min-height:44px;padding:12px 9px;color:#fff;border-radius:3px;font-variant-numeric:tabular-nums;text-shadow:0 1px 2px #000}.surface button.selected{box-shadow:0 0 0 3px var(--focus) inset}.timing-help{display:flex;flex-wrap:wrap;gap:6px 24px;margin-bottom:15px;color:var(--muted);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:var(--muted);background:#252c30;position:sticky;top:0;font-weight:500}th,td{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}tbody tr:hover{background:#262e32}.scroll{overflow:auto;max-height:470px;border:1px solid var(--line);border-radius:4px}button.row{all:unset;display:block;color:var(--cyan);cursor:pointer;font-weight:550}button.row:hover{text-decoration:underline}button.row:focus-visible{outline:2px solid var(--focus);outline-offset:3px}.number{font-variant-numeric:tabular-nums;text-align:right}.positive{color:#e8aea0}.negative{color:#a4c7e7}.unsigned{color:#dec496}.muted{color:var(--muted)}
.primary-button,.secondary-button{border-radius:4px;padding:10px 14px;min-height:44px;font-weight:550}.primary-button{background:var(--cyan);color:#172923;border:1px solid var(--cyan)}.secondary-button{background:#252e33;color:var(--text);border:1px solid #4a555b}.queue-actions{display:flex;justify-content:flex-end;margin-top:14px}.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.datum{padding:13px;background:#22282c;border-radius:4px;overflow-wrap:anywhere}.datum span{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}.candidate-summary{padding:16px 0;border-bottom:1px solid var(--line);margin-bottom:16px}.empty{color:var(--muted);padding:22px;text-align:center}.claims,.boundary{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px}.claim,.boundary div{font-size:13px;color:var(--muted)}.claim strong,.boundary strong{color:var(--text);display:block;margin-bottom:6px;font-weight:550}
.explain-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}.explain-grid div{color:var(--muted);font-size:14px}.explain-grid strong{display:block;color:var(--text);margin-bottom:5px;font-weight:550}.technical-stack{display:grid;gap:20px}.lede{color:var(--muted);max-width:78ch;font-size:14px}.orientation-diagram{color:var(--muted);font-size:13px;text-align:center;margin:12px 0}code{overflow-wrap:anywhere}footer{padding:22px;color:var(--muted);border-top:1px solid var(--line);font-size:12px;text-align:center}
@media(max-width:1100px){.explore-top{grid-template-columns:1fr}.control-grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:950px){.grid,.explain-grid{grid-template-columns:1fr}}
@media(max-width:650px){header{padding-top:24px}h1{font-size:26px}.tabs{gap:20px;overflow:auto}.tab{flex-shrink:0}.card{padding:20px}.control-grid.three{grid-template-columns:1fr}.control-grid{gap:12px}.legend{grid-template-columns:auto minmax(110px,1fr) auto}.evidence-rung summary{grid-template-columns:1fr}.rung-state{justify-self:start;margin-left:20px;grid-row:2;text-align:left}.rung-brief{grid-row:3}.calibration-picks{grid-template-columns:1fr}.evidence-headline{grid-template-columns:repeat(2,minmax(0,1fr))}.claims,.boundary{grid-template-columns:1fr}.result-sentence{font-size:21px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style>
</head>
<body>
<header>
<div class="eyebrow">NeuroPAL · 54 neurons · 17 worms</div>
<h1>Neural prediction atlas</h1>
<p class="sub">Compare predicted target responses under higher and lower repaired source histories.</p>
<p class="header-boundary">Model estimates, not causal effects. History lag is not a transmission delay.</p>
</header>
<nav class="tab-wrap" aria-label="Explorer sections"><div class="tabs" role="tablist">
<button class="tab" id="explore-tab" role="tab" aria-selected="true" aria-controls="explore-panel" data-view="explore">Explore</button>
<button class="tab" id="candidates-tab" role="tab" aria-selected="false" aria-controls="candidates-panel" data-view="candidates">Candidates</button>
<button class="tab" id="references-tab" role="tab" aria-selected="false" aria-controls="references-panel" data-view="references">References</button>
<button class="tab" id="methods-tab" role="tab" aria-selected="false" aria-controls="methods-panel" data-view="methods">Methods &amp; data</button>
</div></nav>
<main>
<section class="view-panel" id="explore-panel" role="tabpanel" aria-labelledby="explore-tab">
<div class="explore-top">
<article class="card start-card">
<h2>Connection and timing</h2>
<details class="calibration-picker"><summary>Calibrated selections · 8 cells</summary><div class="details-body"><div class="calibration-picks" id="calibration-picks"></div><p class="meta" style="margin:10px 0 0">8 fixed selections; not a ranking. The first six passed the upstream support gate; the last two are sensitivity-only.</p></div></details>
<div class="control-grid">
<label>Source neuron<select id="source-control"></select></label>
<label>Target neuron<select id="target-control"></select></label>
</div>
<div class="control-grid three">
<label>Outcome<select id="channel-control" aria-describedby="channel-help"></select></label>
<label>Stimulus period<select id="phase-control"></select></label>
<label>Event subset<select id="context-control" aria-describedby="subset-help"></select></label>
</div>
<p class="control-help" id="channel-help">Mean target activity at the forecast endpoint.</p>
<p class="control-help" id="subset-help">Event subsets are analysis groups; the model uses a binary stimulus input.</p>
<div class="control-grid">
<label>History lag<select id="lag-control"></select></label>
<label>Forecast horizon<select id="horizon-control"></select></label>
</div>
<details class="advanced"><summary>Sampling method &amp; control definitions</summary><div class="details-body"><label>Sampling approach<select id="method-control"></select></label><p class="control-help">Progressive bridge SMC is primary; direct importance is a comparison.</p><dl class="control-definitions"><dt>Source → target</dt><dd>The source history is repaired toward supported low and high states. We compare the resulting target response distributions.</dd><dt>History lag / forecast horizon</dt><dd>Lag locates the repaired source window before the prediction cut; horizon locates the target readout after it. A history coordinate is not a physical transmission delay.</dd><dt>Stimulus period / event subset</dt><dd>Periods define when a window occurs relative to stimulation. Chemical identity selects an analysis subset; it is not an extra conditioning input.</dd></dl></div></details>
</article>
<article class="card result-card" aria-labelledby="current-result-title">
<div class="result-kicker">Canonical atlas estimate</div><h2 id="current-result-title">Predicted difference</h2>
<div class="result-question" id="current-question"></div><p class="result-sentence" id="result-sentence" aria-live="polite"></p><div class="result-context" id="result-context"></div>
<p class="boundary-line">Higher vs lower repaired source history. Normalized units; N128 evidence, where available, is reported separately below.</p>
</article>
</div>
<article class="card evidence-card" id="statistical-evidence-view" aria-labelledby="statistical-evidence-title">
<div class="card-head"><div><h2 id="statistical-evidence-title">Evidence for this selection</h2><div class="meta">Model-relative tests, not biological confirmation. Colors show effect magnitude only.</div></div></div>
<section class="evidence-current" aria-label="Current selection evidence"><span id="evidence-current-label" class="evidence-status" hidden></span><p id="evidence-current" class="evidence-copy" aria-live="polite"></p><ol id="evidence-ladder" class="evidence-ladder"></ol><details class="secondary-details" id="sampling-metric-details"><summary>Compare the three calibrated outcomes</summary><div class="details-body"><p class="meta">Rows match source, target, lag, horizon, baseline context, and metric exactly. “Worst sampler” is the per-worm maximum of low–low, high–high, and midpoint–midpoint controls. P-values use the separate, selection-conditioned 64-test N128 family; intervals are descriptive worm-bootstrap intervals.</p><div class="scroll evidence-table"><table><thead><tr><th>Metric</th><th class="number">Observed</th><th>Observed evidence</th><th class="number">Worst sampler</th><th>Sampler excess</th><th>Quiet-time excess</th><th>Overall calibration</th></tr></thead><tbody id="sampling-metric-body"></tbody></table></div></div></details><details class="secondary-details"><summary>Before an experiment</summary><div class="details-body"><div id="evidence-next-step" class="inline-note"></div></div></details></section>
<details class="secondary-details" id="inference-definitions"><summary>Test definitions &amp; limits</summary><div class="details-body"><p class="meta">The complete-family analysis uses all 65,536 two-sided sign patterns across 17 worms and jointly adjusts four composition-matched families. The separate N128 calibration tests eight preselected cells against sampler controls and quiet-time pseudo-boundaries.</p><p class="meta">Exact enumeration assumes joint worm-vector sign symmetry. Overlapping cross-fit training sets can couple held-out estimates. These are model-relative calibration p-values—not experimental randomization significance. The 64-test adjustment does not correct upstream selection.</p><p class="meta">A result that does not pass a control is not proof of zero effect. In baseline windows, quiet-time specificity is not stimulus modulation.</p></div></details>
<details class="secondary-details" id="atlas-overview"><summary>Across the atlas</summary><div class="details-body">
<div class="evidence-headline" id="evidence-headline"></div>
<p class="evidence-note" id="evidence-practical-status"></p>
<details><summary>How the joint test works and results by family</summary><div class="details-body"><p>The same worm sign is reused across every edge, lag, horizon, and family in each exact pattern. A single joint max-T null then accounts for searching all four primary families. The edge test asks whether any lag/horizon cell on a directed edge is unusually large. The separate flat-lag test asks whether the effect changes across source-lag coordinates.</p><p class="meta">The two active-minus-baseline families are family-summary results only in this explorer: that derived context was tested from the canonical worm tensors but is not a selectable dense matrix slice. Baseline exact-cell evidence remains selectable.</p><div class="scroll evidence-table"><table><thead><tr><th>Primary family</th><th class="number">Complete edges</th><th class="number">Support-eligible</th><th class="number">Strong edges</th><th class="number">Cells differing from zero</th><th class="number">Non-flat lag edges</th></tr></thead><tbody id="evidence-family-body"></tbody></table></div><p class="meta" style="margin-top:14px">The 0.5-support rerun is sensitivity-only. Its named lag rows remain sampling-limited and cannot strengthen a primary or experiment-ready claim.</p><div id="evidence-sensitivity"></div></div></details>
</div></details>
</article>
<article class="card" id="cell-view"><div class="card-head"><div><h2>Lag × forecast</h2><div class="meta" id="cell-label"></div></div></div><div class="timing-help"><div><strong>Rows:</strong> source history lag.</div><div><strong>Columns:</strong> target forecast horizon.</div></div><div class="surface" id="cell-surface"></div></article>
<section class="grid">
<article class="card" id="matrix-view"><div class="card-head"><div><h2>All connections</h2><div class="meta">54 × 54 source–target pairs. Select a square, or use the neuron controls above.</div></div><span class="pill" id="slice-label"></span></div><div class="orientation-diagram">Rows: target neuron · Columns: source neuron</div><div class="canvas-wrap"><canvas id="matrix-canvas" width="648" height="648" role="img" aria-label="Target-row source-column neural matrix" aria-describedby="matrix-readout matrix-scale-note"></canvas></div><div class="legend"><span id="legend-left"></span><div class="legend-center"><div class="legend-bar" id="matrix-legend-bar"></div><div class="legend-labels"><span id="legend-negative"></span><span>0</span><span id="legend-positive"></span></div></div><span id="legend-right"></span></div><div class="scale-note" id="matrix-scale-note">per-slice scale</div><div class="readout" id="matrix-readout" aria-live="polite"></div></article>
<article class="card" id="downstream-view"><div class="card-head"><div><h2>Largest target responses</h2><div class="meta">Top 18 by absolute effect for this source; not a significance ranking.</div></div></div><div class="scroll"><table><thead><tr><th>Rank</th><th>Target neuron</th><th class="number">Effect</th></tr></thead><tbody id="downstream-body"></tbody></table></div></article>
</section>
</section>
<section class="view-panel" id="candidates-panel" role="tabpanel" aria-labelledby="candidates-tab" hidden>
<article class="card wide" id="queue-view"><div class="card-head"><div><h2>Candidate predictions</h2><div class="meta">Frozen model-based selections. External references did not affect ranking.</div></div><span class="pill" id="queue-count"></span></div><p class="lede">Showing the six original N128 follow-up candidates. Expand the queue to browse all 1,000.</p><details class="secondary-details" style="margin-bottom:18px"><summary>Queue and evidence labels</summary><div class="details-body"><p class="meta">Randi, Cook, Bentley, and SBTG were excluded from selection. An exact cell-and-metric match uses the newer selection-conditioned sampler calibration; otherwise the label uses complete-family or legacy screening evidence. These evidence levels are not interchangeable.</p></div></details><div class="scroll"><table><thead><tr><th>Rank</th><th>Connection</th><th>Biological context</th><th>Timing</th><th class="number">Effect</th><th>Statistical evidence</th><th>Atlas-screen support</th></tr></thead><tbody id="queue-body"></tbody></table></div><div class="queue-actions"><button class="secondary-button" id="queue-toggle" type="button"></button></div><h3 style="margin-top:24px">Selected candidate</h3><div id="queue-detail"></div></article>
<article class="card wide" id="targeted-view"><div class="card-head"><div><h2>Original N128 follow-up · 6 candidates</h2><div class="meta">Earlier three-arm follow-up; separate from the eight-cell sampler/quiet calibration.</div></div><span class="pill">128 particles</span></div><div class="detail-grid" id="targeted-summary"></div><div id="targeted-interpretation" class="inline-note" style="margin-top:16px"></div><details style="margin-top:18px"><summary>Outcome estimates</summary><div class="details-body"><p class="meta">Wasserstein-1 is unsigned within a state; a between-phase difference can be signed, but it is not a direction of transport.</p><div class="scroll"><table><thead><tr><th>Candidate</th><th>Contrast</th><th>Metric</th><th class="number">Channel effect</th><th>95% interval</th><th>Evidence label</th></tr></thead><tbody id="targeted-body"></tbody></table></div></div></details><details style="margin-top:12px"><summary>Technical diagnostics: quantiles, sampler support, and screen consistency</summary><div class="details-body technical-stack"><div><h3>Endpoint quantile shifts</h3><div id="targeted-quantile-table"></div></div><div><h3>Sampler support and genealogy diagnostics</h3><div id="targeted-support-table"></div></div><div><h3>Screen versus N128 consistency</h3><div id="targeted-screen-table"></div></div></div></details></article>
</section>
<section class="view-panel" id="references-panel" role="tabpanel" aria-labelledby="references-tab" hidden>
<article class="card wide"><h2>Reference comparisons</h2><p class="lede">Agreement with external references, evaluated after selection. These comparisons did not train or rank the atlas and do not validate causal effects.</p><div class="explain-grid"><div><strong>AUROC</strong>How often a known reference edge scores above a non-edge. AUROC 0.5 is chance.</div><div><strong>AUPRC</strong>Precision–recall performance, which depends strongly on how rare the reference edges are.</div><div><strong>Spearman correlation</strong>Whether the rank order of continuous learned scores tracks continuous reference weights.</div></div></article>
<article class="card wide" id="external-view"><div class="card-head"><div><h2>Randi, Cook, Bentley &amp; SBTG</h2><div class="meta">The source-preserving max-lag null accounts for lag selection; BH q adjusts across tested rows.</div></div><span class="pill">post-freeze</span></div><details><summary>Show all external comparison rows</summary><div class="details-body"><div class="scroll"><table><thead><tr><th>Analysis panel</th><th>Reference or network</th><th>Timing</th><th class="number">AUROC</th><th class="number">AUPRC</th><th class="number">Spearman</th><th class="number">BH q</th></tr></thead><tbody id="external-body"></tbody></table></div></div></details><details style="margin-top:12px"><summary>Show neuromodulator lag-max inference</summary><div class="details-body"><p class="meta">A best lag here is a statistical history coordinate. It does not establish a biological propagation delay.</p><div class="scroll"><table id="lagmax-table"></table></div></div></details></article>
</section>
<section class="view-panel" id="methods-panel" role="tabpanel" aria-labelledby="methods-tab" hidden>
<article class="card wide"><h2>How effects are estimated</h2><p class="lede">A conditional flow predicts future neural activity from neural and stimulus history. Sampling compares supported high and low source histories under matched settings.</p><div class="explain-grid"><div><strong>1 · Condition</strong>Start from real observed neural-history and stimulus windows.</div><div><strong>2 · Repair and sample</strong>Shift one source history toward supported low or high states and generate future paths.</div><div><strong>3 · Compare</strong>Summarize changes in target mean, peak, variability, event probability, or the whole distribution.</div></div><p class="inline-note" style="margin-top:18px">The normalized channel effect is model-relative and noncausal. Lag is not a physical delay. Colors share one per-slice scale only within the displayed matrix or lag × horizon grid.</p></article>
<article class="card wide"><div class="card-head"><div><h2>Scope and limitations</h2></div></div><section class="boundary" id="boundary-strip"></section></article>
<article class="card wide" id="composition-view"><div class="card-head"><div><h2>Stimulus-window composition</h2><div class="meta">Schedule-derived occupancy pooled across all three chemicals. The chemical event-stratum selector does not filter this table; state average shows all five phases.</div></div><span class="pill" id="composition-status"></span></div><details><summary>Show the composition rows for the selected timing</summary><div class="details-body"><div id="composition-table"></div></div></details></article>
<article class="card wide" id="provenance-view"><div class="card-head"><div><h2>Data and provenance</h2><div class="meta">All five required inputs passed their canonical validators before this file was written.</div></div><span class="pill">Validated inputs</span></div><details><summary>Show every claim boundary</summary><div class="details-body"><div class="claims" id="claim-list"></div></div></details><details style="margin-top:12px"><summary>Show verified input bundles and checksums</summary><div class="details-body"><div class="scroll"><table><thead><tr><th>Bundle</th><th>Portable path</th><th>Manifest SHA-256</th><th>Ledger SHA-256</th><th>Files</th></tr></thead><tbody id="provenance-body"></tbody></table></div></div></details></article>
</section>
</main>
<footer>Offline · checksum-verified inputs · displayed matrices use per-slice quantization; canonical values are preserved.</footer>
<script type="application/json" id="atlas-payload">__PAYLOAD__</script>
<script>
"use strict";
const payload=JSON.parse(document.getElementById("atlas-payload").textContent);
const atlas=payload.atlas, n=atlas.n_neurons, cache=new Map();
const controls={method:document.getElementById("method-control"),channel:document.getElementById("channel-control"),stratum:document.getElementById("context-control"),phase:document.getElementById("phase-control"),lag:document.getElementById("lag-control"),horizon:document.getElementById("horizon-control"),source:document.getElementById("source-control"),target:document.getElementById("target-control")};
const preferred=(items,value)=>items.includes(value)?value:items[0];
const frameSeconds=0.25;
const methodLabels={progressive_bridge_smc:"Progressive bridge SMC (primary)",direct_importance:"Direct importance sampling (comparison)"};
const channelLabels={endpoint_mean:"Endpoint activity",cumulative_mean:"Cumulative activity",peak_mean:"Peak activity",event_probability:"Event probability",endpoint_sd:"Endpoint variability",endpoint_log_sd:"Log endpoint variability",endpoint_wasserstein1:"Whole-distribution difference (Wasserstein-1)"};
const channelHelp={endpoint_mean:"Mean target activity at the forecast endpoint.",cumulative_mean:"Accumulated target activity across the forecast window.",peak_mean:"Peak target activity reached in the forecast window.",event_probability:"Probability that the defined target event occurs.",endpoint_sd:"Spread of target activity at the forecast endpoint.",endpoint_log_sd:"Relative change in endpoint variability on a log scale.",endpoint_wasserstein1:"Overall distance between predicted endpoint distributions; within a state it has no positive or negative direction."};
const phaseLabels={baseline:"Before stimulus (baseline)",onset:"Stimulus onset",active:"During stimulus",offset:"Stimulus offset",recovery:"After stimulus (recovery)",state_average:"Average across all stimulus periods",onset_minus_baseline:"Onset compared with baseline"};
const stratumLabels={all_scheduled_events:"All three stimulus types",butanone:"Butanone events",pentanedione:"Pentanedione events",nacl:"NaCl events"};
const supportLabels={supported_exploratory:"Supported exploratory",sensitivity_only:"Sensitivity only",unsupported:"Unsupported"};
const familyLabels={baseline_endpoint_mean:"Baseline · endpoint mean",baseline_endpoint_log_sd:"Baseline · endpoint log variability",active_minus_baseline_endpoint_mean:"Active minus baseline · endpoint mean",active_minus_baseline_endpoint_log_sd:"Active minus baseline · endpoint log variability"};
const evidenceLabelText={experiment_ready:"Experiment-ready",strong_edge_lag_unresolved:"Strong edge; lag unresolved",strong_edge_with_lag_structure:"Strong edge with lag structure",exploratory:"Exploratory",sampling_limited:"Sampling-limited"};
const calibrationLabelText={exceeds_sampling_and_quiet_controls:"Exceeds both controls",exceeds_sampling_controls_only:"Exceeds sampler noise only",indistinguishable_from_sampling_controls:"Not separated from controls",sampling_limited:"Support or selection restriction"};
const gateReasonLabel={none:"No restriction",support_failure:"Support check failed",sensitivity_origin:"Sensitivity analysis only",sensitivity_origin_and_support_failure:"Sensitivity only · support failed"};
const calibrationLabelTone={exceeds_sampling_and_quiet_controls:"strong",exceeds_sampling_controls_only:"exploratory",indistinguishable_from_sampling_controls:"limited",sampling_limited:"limited"};
const pretty=(value,map={})=>map[value]||String(value??"—").replaceAll("_"," ");
const firstCandidate=payload.candidate_queue.length?payload.candidate_queue[0]:null,firstCalibration=[...(payload.sampling_null_calibration.cells||[])].filter(row=>String(row.metric)==="endpoint_mean"&&String(row.selection_origin)==="strong_primary").sort((a,b)=>Number(a.queue_rank)-Number(b.queue_rank))[0]||null,initialSelection=firstCalibration||firstCandidate;
const initialContext=initialSelection&&atlas.contexts.includes(String(initialSelection.context))?String(initialSelection.context):preferred(atlas.contexts,"onset_minus_baseline"),initialContextOption=atlas.context_options.find(x=>x.context===initialContext);
const state={method:firstCalibration?"progressive_bridge_smc":firstCandidate&&atlas.methods.includes(String(firstCandidate.method))?String(firstCandidate.method):preferred(atlas.methods,"progressive_bridge_smc"),channel:firstCalibration?String(firstCalibration.metric):firstCandidate&&atlas.channels.includes(String(firstCandidate.channel))?String(firstCandidate.channel):preferred(atlas.channels,"endpoint_mean"),context:initialContext,stratum:initialContextOption.event_stratum,phase:initialContextOption.phase_or_contrast,lag:initialSelection&&atlas.source_lag_frames.includes(Number(initialSelection.source_lag_frames))?Number(initialSelection.source_lag_frames):atlas.source_lag_frames[0],horizon:initialSelection&&atlas.horizon_frames.includes(Number(initialSelection.horizon_frames))?Number(initialSelection.horizon_frames):atlas.horizon_frames[0],source:initialSelection?Number(initialSelection.source_index):0,target:initialSelection?Number(initialSelection.target_index):0,queueRank:firstCalibration?null:firstCandidate?Number(firstCandidate.queue_rank):null};
let queueExpanded=false,externalRendered=false,provenanceRendered=false;
const e=value=>String(value??"—").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const f=(value,digits=4)=>value===null||value===undefined||!Number.isFinite(Number(value))?"—":Number(value).toFixed(digits).replace(/\.?0+$/,"");
const seconds=frames=>`${f(Number(frames)*frameSeconds,2)} s`;
const option=(value,label=value)=>`<option value="${e(value)}">${e(label)}</option>`;
controls.method.innerHTML=atlas.methods.map(x=>option(x,pretty(x,methodLabels))).join("");
controls.channel.innerHTML=atlas.channels.map(x=>option(x,pretty(x,channelLabels))).join("");
const strata=[...new Set(atlas.context_options.map(x=>x.event_stratum))];
controls.stratum.innerHTML=strata.map(x=>option(x,pretty(x,stratumLabels))).join("");
controls.lag.innerHTML=atlas.source_lag_frames.map(x=>option(x,`${seconds(x)} back (${x} frame${x===1?"":"s"})`)).join("");
controls.horizon.innerHTML=atlas.horizon_frames.map(x=>option(x,`${seconds(x)} ahead (${x} frame${x===1?"":"s"})`)).join("");
controls.source.innerHTML=atlas.neurons.map((x,i)=>option(i,x)).join("");
controls.target.innerHTML=atlas.neurons.map((x,i)=>option(i,x)).join("");
function setCompositeContext(context){const item=atlas.context_options.find(x=>x.context===context);if(!item)return;state.context=item.context;state.stratum=item.event_stratum;state.phase=item.phase_or_contrast;}
function sync(){for(const key of ["method","channel","lag","horizon","source","target"])controls[key].value=String(state[key]);controls.stratum.value=state.stratum;const available=atlas.context_options.filter(x=>x.event_stratum===state.stratum);controls.phase.innerHTML=available.map(x=>option(x.phase_or_contrast,pretty(x.phase_or_contrast,phaseLabels))).join("");if(!available.some(x=>x.phase_or_contrast===state.phase)){state.phase=available[0].phase_or_contrast;state.context=available[0].context;}controls.phase.value=state.phase;document.getElementById("channel-help").textContent=channelHelp[state.channel]||"Choose a feature of the predicted response distribution.";}
function markCustom(){state.queueRank=null;}
function familyFor(method,channel,context){if(String(method)!=="progressive_bridge_smc")return null;const row=payload.statistical_evidence.family_summary.find(x=>String(x.channel)===String(channel)&&String(x.context)===String(context));return row?String(row.family_id):null;}
function sameEvidenceCell(row,familyId,sourceNeuron,targetNeuron,lag,horizon){return String(row.family_id)===String(familyId)&&String(row.source_neuron)===String(sourceNeuron)&&String(row.target_neuron)===String(targetNeuron)&&Number(row.source_lag_frames)===Number(lag)&&Number(row.horizon_frames)===Number(horizon);}
function evidenceLabel(row){if(!row)return null;if(row.experiment_ready===true)return evidenceLabelText.experiment_ready;if(row.evidence_label==="strong_edge_lag_unresolved")return evidenceLabelText.strong_edge_lag_unresolved;if(row.evidence_label==="strong_edge_with_lag_structure")return evidenceLabelText.strong_edge_with_lag_structure;if(row.evidence_label==="sampling_limited")return evidenceLabelText.sampling_limited;if(row.evidence_label==="exploratory")return evidenceLabelText.exploratory;return null;}
function evidenceClass(label){return label===evidenceLabelText.strong_edge_lag_unresolved||label===evidenceLabelText.strong_edge_with_lag_structure?"strong":label===evidenceLabelText.sampling_limited?"limited":"exploratory";}
function currentEvidence(){const familyId=familyFor(state.method,state.channel,state.context),sourceNeuron=atlas.neurons[state.source],targetNeuron=atlas.neurons[state.target];if(!familyId)return {familyId:null,cell:null,edge:null};const primary=payload.statistical_evidence.sampling_null_shortlist||[],sensitivity=payload.statistical_evidence.sensitivity_support05&&payload.statistical_evidence.sensitivity_support05.named_lag_rows||[],legacy=payload.statistical_evidence.legacy_queue&&payload.statistical_evidence.legacy_queue.matches||[],cell=[...primary,...sensitivity,...legacy].find(row=>sameEvidenceCell(row,familyId,sourceNeuron,targetNeuron,state.lag,state.horizon))||null,edge=(payload.statistical_evidence.top_edges||[]).find(row=>String(row.family_id)===familyId&&String(row.source_neuron)===sourceNeuron&&String(row.target_neuron)===targetNeuron)||null;return {familyId,cell,edge};}
function evidenceForCandidate(candidate){const familyId=familyFor(candidate.method,candidate.channel,candidate.context);if(!familyId)return null;const rows=payload.statistical_evidence.legacy_queue&&payload.statistical_evidence.legacy_queue.matches||[];return rows.find(row=>sameEvidenceCell(row,familyId,String(candidate.source_neuron),String(candidate.target_neuron),Number(candidate.source_lag_frames),Number(candidate.horizon_frames))&&Number(row.legacy_queue_rank)===Number(candidate.queue_rank))||null;}
function sameSamplingCell(row,sourceNeuron,targetNeuron,sourceIndex,targetIndex,lag,horizon,context){return String(row.source_neuron)===String(sourceNeuron)&&String(row.target_neuron)===String(targetNeuron)&&Number(row.source_index)===Number(sourceIndex)&&Number(row.target_index)===Number(targetIndex)&&Number(row.source_lag_frames)===Number(lag)&&Number(row.horizon_frames)===Number(horizon)&&String(row.context)===String(context);}
function samplingMetricsForCurrent(){if(String(state.method)!=="progressive_bridge_smc")return [];const sourceNeuron=atlas.neurons[state.source],targetNeuron=atlas.neurons[state.target];return (payload.sampling_null_calibration.cells||[]).filter(row=>sameSamplingCell(row,sourceNeuron,targetNeuron,state.source,state.target,state.lag,state.horizon,state.context));}
function currentSamplingCalibration(){return samplingMetricsForCurrent().find(row=>String(row.metric)===String(state.channel))||null;}
function currentSamplingSupport(){const rows=samplingMetricsForCurrent();if(!rows.length)return null;return (payload.sampling_null_calibration.support_diagnostics||[]).find(row=>String(row.candidate_id)===String(rows[0].candidate_id))||null;}
function samplingForCandidate(candidate){if(String(candidate.method)!=="progressive_bridge_smc")return null;return (payload.sampling_null_calibration.cells||[]).find(row=>sameSamplingCell(row,candidate.source_neuron,candidate.target_neuron,candidate.source_index,candidate.target_index,candidate.source_lag_frames,candidate.horizon_frames,candidate.context)&&String(row.metric)===String(candidate.channel))||null;}
function calibrationLabel(row){if(!row)return null;if(String(row.evidence_label)==="sampling_limited")return gateReasonLabel[String(row.gate_reason)]||calibrationLabelText.sampling_limited;return calibrationLabelText[String(row.evidence_label)]||null;}
function finiteNumber(value){return value!==null&&value!==undefined&&value!==""&&Number.isFinite(Number(value));}
function pText(value){return !finiteNumber(value)?"—":Number(value).toPrecision(4).replace(/\.0+e/,"e");}
function supportAssessment(support){const gate=payload.sampling_null_calibration.protocol.support_gate||{},validThreshold=Number(gate.valid_fraction_threshold??0.80),genealogyThreshold=Number(gate.genealogy_valid_fraction_threshold??0.80),values=support?{observedValid:Number(support.observed_valid_fraction),observedGenealogy:Number(support.observed_genealogy_valid_fraction_0_10),samplingValid:Number(support.sampling_valid_fraction),samplingGenealogy:Number(support.sampling_genealogy_valid_fraction_0_10),quietValid:Number(support.pseudo_valid_fraction),quietGenealogy:Number(support.pseudo_genealogy_valid_fraction_0_10)}:null,pass=!!values&&Object.values(values).every(Number.isFinite)&&values.observedValid>=validThreshold&&values.samplingValid>=validThreshold&&values.quietValid>=validThreshold&&values.observedGenealogy>=genealogyThreshold&&values.samplingGenealogy>=genealogyThreshold&&values.quietGenealogy>=genealogyThreshold;return {available:!!values,pass,validThreshold,genealogyThreshold,values};}
function signedObservedPass(row){if(!row||row.context_signed!==true)return null;return finiteNumber(row.observed_joint_max_t_p)&&Number(row.observed_joint_max_t_p)<=Number(payload.sampling_null_calibration.protocol.alpha)&&finiteNumber(row.observed_ci_2_5)&&finiteNumber(row.observed_ci_97_5)&&(Number(row.observed_ci_2_5)>0||Number(row.observed_ci_97_5)<0);}
function positiveExcessPass(row,prefix){const alpha=Number(payload.sampling_null_calibration.protocol.alpha),mean=row[`${prefix}_mean`],low=row[`${prefix}_ci_2_5`],p=row[`${prefix}_joint_max_t_p`];return finiteNumber(mean)&&finiteNumber(low)&&finiteNumber(p)&&Number(mean)>0&&Number(low)>0&&Number(p)<=alpha;}
function record(method=state.method,channel=state.channel,context=state.context,lag=state.lag,horizon=state.horizon){return atlas.slices[method][channel][context][String(lag)][String(horizon)];}
function decode(method=state.method,channel=state.channel,context=state.context,lag=state.lag,horizon=state.horizon){const key=[method,channel,context,lag,horizon].join("\u001f");if(cache.has(key))return cache.get(key);const item=record(method,channel,context,lag,horizon),raw=atob(item.data),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);const view=new DataView(bytes.buffer),out=new Float32Array(n*n);for(let i=0;i<out.length;i++)out[i]=view.getInt16(i*2,true)*item.scale;cache.set(key,out);return out;}
const unsignedMetric=(channel,context)=>channel==="endpoint_wasserstein1"&&!String(context).endsWith("minus_baseline");
const unsignedChannel=()=>unsignedMetric(state.channel,state.context);
function color(value,limit,unsigned=false){const x=Math.max(-1,Math.min(1,value/(limit||1))),a=unsigned?Math.max(0,x):Math.abs(x),white=[239,244,247],edge=unsigned?[194,137,52]:x>=0?[216,82,68]:[64,127,194],rgb=white.map((v,i)=>Math.round(v+(edge[i]-v)*a));return `rgb(${rgb.join(",")})`;}
function renderBoundaries(){document.getElementById("boundary-strip").innerHTML=payload.claim_boundaries.slice(0,4).map(x=>`<div><strong>${e(x.label)}</strong>${e(x.detail)}</div>`).join("");}
function effectPhrase(value,channel=state.channel,context=state.context,target=atlas.neurons[state.target],source=atlas.neurons[state.source]){
  const magnitude=f(Math.abs(Number(value)),4),highlight=`<span class="result-value">${e(magnitude)}</span>`;
  if(channel==="endpoint_wasserstein1"&&unsignedMetric(channel,context))return `<strong>${e(target)}</strong> predictions are ${highlight} normalized units apart between the high- and low-source histories. This distance is unsigned.`;
  if(channel==="endpoint_wasserstein1"){const direction=Number(value)>=0?"greater":"smaller";return `<strong>${e(target)}</strong> distributional separation is ${highlight} normalized units <strong>${direction}</strong> at onset than baseline. This sign compares distances; it is not a direction of transport.`;}
  const direction=Number(value)>=0?"higher":"lower";
  if(String(context).endsWith("_minus_baseline"))return `The high–low effect on <strong>${e(target)}</strong> ${e(pretty(channel,channelLabels).toLowerCase())} is ${highlight} normalized units <strong>${direction}</strong> at onset than baseline. This compares two model-relative effects.`;
  return `Predicted <strong>${e(target)}</strong> ${e(pretty(channel,channelLabels).toLowerCase())} is ${highlight} normalized units <strong>${direction}</strong> with higher <strong>${e(source)}</strong> history.`;
}
function renderSummary(){const value=decode()[state.target*n+state.source];document.getElementById("current-question").textContent=`${atlas.neurons[state.source]} → ${atlas.neurons[state.target]} · ${seconds(state.lag)} back / ${seconds(state.horizon)} ahead`;document.getElementById("result-sentence").innerHTML=effectPhrase(value);const candidate=state.queueRank===null?null:payload.candidate_queue.find(x=>Number(x.queue_rank)===state.queueRank);const chips=[pretty(state.phase,phaseLabels),pretty(state.stratum,stratumLabels),pretty(state.method,methodLabels),...(candidate?[`Frozen candidate #${candidate.queue_rank}`]:[])];document.getElementById("result-context").innerHTML=chips.map(x=>`<span class="context-chip">${e(x)}</span>`).join("");}
function calibratedCellRows(){return (payload.sampling_null_calibration.cells||[]).filter(row=>String(row.metric)==="endpoint_mean").sort((a,b)=>Number(a.queue_rank)-Number(b.queue_rank));}
function renderCalibrationPicks(){const rows=calibratedCellRows(),container=document.getElementById("calibration-picks");container.innerHTML=rows.map(row=>{const selected=String(state.method)==="progressive_bridge_smc"&&Number(state.source)===Number(row.source_index)&&Number(state.target)===Number(row.target_index)&&Number(state.lag)===Number(row.source_lag_frames)&&Number(state.horizon)===Number(row.horizon_frames)&&String(state.context)===String(row.context);return `<button class="calibration-pick" type="button" data-candidate-id="${e(row.candidate_id)}" aria-pressed="${selected}"><span>${e(row.source_neuron)} → ${e(row.target_neuron)}</span><small>${e(seconds(row.source_lag_frames))} back / ${e(seconds(row.horizon_frames))} ahead${String(row.selection_origin)==="lag_sensitivity"?" · sensitivity":""}</small></button>`;}).join("");container.querySelectorAll("button").forEach(button=>button.addEventListener("click",()=>{const row=rows.find(item=>String(item.candidate_id)===String(button.dataset.candidateId));if(!row)return;state.method="progressive_bridge_smc";state.channel="endpoint_mean";state.source=Number(row.source_index);state.target=Number(row.target_index);state.lag=Number(row.source_lag_frames);state.horizon=Number(row.horizon_frames);setCompositeContext(String(row.context));state.queueRank=null;sync();renderAll();}));}
function evidenceRung(number,title,status,tone,detail,brief){const descriptions={1:"Sampling support and particle ancestry.",2:"Tests of a nonzero effect and differences across lags.",3:"High–low response compared with same-state sampler reruns.",4:"Response compared with other quiet history windows.",5:"Independent validation and a meaningful-effect threshold are still needed."};return `<li class="evidence-rung"><details data-evidence-check="${e(number)}"><summary><span class="rung-title">${e(title)}</span><span class="rung-state ${e(tone)}">${e(status)}</span><span class="rung-brief">${e(brief||descriptions[number])}</span></summary><p class="rung-detail">${e(detail)}</p></details></li>`;}
function intervalText(low,high){return `[${f(low,4)}, ${f(high,4)}]`;}
function renderStatisticalEvidence(){
  const data=payload.statistical_evidence,headline=data.headline,families=data.family_summary,current=currentEvidence(),calibration=currentSamplingCalibration(),metricRows=samplingMetricsForCurrent(),support=currentSamplingSupport(),supportResult=supportAssessment(support),completeLabel=evidenceLabel(current.cell)||evidenceLabel(current.edge),n128Label=calibrationLabel(calibration),labelNode=document.getElementById("evidence-current-label"),copy=document.getElementById("evidence-current"),ladder=[];
  const label=n128Label||completeLabel;labelNode.hidden=!label;if(label){labelNode.textContent=calibration?`N128: ${n128Label}`:`Complete-family: ${completeLabel}`;labelNode.className=`evidence-status ${calibration?calibrationLabelTone[String(calibration.evidence_label)]||"exploratory":evidenceClass(label)}`;}
  copy.textContent=calibration?"Sampler and quiet-time checks use a separate matched N128 rerun (128 particles), not the canonical atlas estimate above.":metricRows.length?"Support was audited here, but this outcome was not calibrated. No metric-level evidence is borrowed.":"This exact cell was not among the eight N128 calibrations. No evidence is borrowed from another selection.";
  if(supportResult.available){
    const v=supportResult.values,lagSensitivity=metricRows.some(row=>String(row.selection_origin)==="lag_sensitivity"),supportDetail=`Observed validity ${f(v.observedValid,3)} and genealogy-valid ${f(v.observedGenealogy,3)}; named sampler-control worst-case validity ${f(v.samplingValid,3)} and genealogy-valid ${f(v.samplingGenealogy,3)}; quiet-time validity ${f(v.quietValid,3)} and genealogy-valid ${f(v.quietGenealogy,3)}. Sampler-control fractions are the minimum across low–low, high–high, and midpoint–midpoint. Frozen thresholds are ${f(supportResult.validThreshold,2)} for validity and ${f(supportResult.genealogyThreshold,2)} for genealogy.${lagSensitivity?" This cell came from the lower-support lag-sensitivity audit, so its overall claim remains sensitivity-only even if this sampler-support rung passes.":""}`;
    ladder.push(evidenceRung(1,"Sampling reliability",supportResult.pass?"Pass":"Support failed",supportResult.pass?"pass":"caution",supportDetail,supportResult.pass?"All six support and ancestry checks pass.":"At least one support or ancestry check falls below its threshold."));
  }else ladder.push(evidenceRung(1,"Sampling reliability","Not run","missing","No support audit exists for this exact selection. The N128 calibration covers eight baseline progressive-bridge cells."));
  if(!current.familyId)ladder.push(evidenceRung(2,"Effect and lag evidence","Outside family","missing","The current method, metric, or context is outside the four declared complete-family tests."));
  else if(current.cell&&current.cell.sensitivity_only===true)ladder.push(evidenceRung(2,"Effect and lag evidence","Sensitivity only","caution",`This exact cell is outside the primary support family. Its sensitivity lag max-T p=${pText(current.cell.joint_sensitivity_lag_contrast_max_t_p_value)}. Sensitivity-only evidence cannot strengthen or promote a primary claim.`));
  else if(current.cell){const cellP=current.cell.joint_primary_cell_max_t_p_value,edgeP=current.cell.joint_primary_edge_max_t_p_value,lagP=current.cell.joint_primary_flat_lag_max_t_p_value??(current.edge&&current.edge.joint_primary_flat_lag_max_t_p_value),cellPass=finiteNumber(cellP)&&Number(cellP)<=0.05,edgePass=finiteNumber(edgeP)&&Number(edgeP)<=0.05,lagPass=finiteNumber(lagP)&&Number(lagP)<=0.05,gated=String(current.cell.evidence_label)==="sampling_limited";const status=gated&&(cellPass||edgePass)?"Test passes · support failed":cellPass&&lagPass?"Effect and lag differences detected":cellPass?"Effect detected · lag unresolved":edgePass?"Edge detected · cell unresolved":"Not passed",tone=gated&&(cellPass||edgePass)?"caution":cellPass?"pass":edgePass?"caution":"fail";ladder.push(evidenceRung(2,"Effect and lag evidence",status,tone,`Exact joint cell p=${pText(cellP)}; joint edge p=${pText(edgeP)}; flat-lag p=${pText(lagP)}. A cell differing from zero is distinct from evidence that source-lag coordinates differ.${gated?" This row fails the primary support gate and cannot be promoted.":""}`,`Atlas cell p=${pText(cellP)} · lag-difference p=${pText(lagP)}.`));}
  else if(current.edge){const edgeP=current.edge.joint_primary_edge_max_t_p_value,lagP=current.edge.joint_primary_flat_lag_max_t_p_value,edgePass=finiteNumber(edgeP)&&Number(edgeP)<=0.05,lagPass=finiteNumber(lagP)&&Number(lagP)<=0.05,gated=String(current.edge.evidence_label)==="sampling_limited",status=gated&&edgePass?"Test passes · support failed":edgePass?(lagPass?"Edge and lag structure detected":"Edge detected; timing unresolved"):"Not passed",tone=gated&&edgePass?"caution":edgePass?"pass":"fail";ladder.push(evidenceRung(2,"Effect and lag evidence",status,tone,`Joint edge p=${pText(edgeP)}; flat-lag p=${pText(lagP)}. This exact lag/horizon cell is not localized.${gated?" The edge fails the primary support gate and cannot be promoted.":""}`,`Edge p=${pText(edgeP)} · lag-difference p=${pText(lagP)}; this cell is not localized.`));}
  else ladder.push(evidenceRung(2,"Effect and lag evidence","Not passed","fail","No exact cell or directed-edge discovery matches the current complete-family key."));
  if(calibration){
    const observedPass=signedObservedPass(calibration),passesSampler=positiveExcessPass(calibration,"sampling_excess"),passesQuiet=positiveExcessPass(calibration,"temporal_specificity"),overallGated=String(calibration.evidence_label)==="sampling_limited",gateDetail=overallGated?` ${gateReasonLabel[String(calibration.gate_reason)]||calibrationLabelText.sampling_limited}.`:"",fullSamplerPass=(observedPass!==false)&&passesSampler,observedDetail=observedPass===null?`The matched N128 rerun estimate is ${f(calibration.observed_mean,4)} with interval ${intervalText(calibration.observed_ci_2_5,calibration.observed_ci_97_5)}. Wasserstein-1 is unsigned, so no signed observed-null p-value is defined; only its excess over controls is tested.`:`The matched N128 rerun estimate is ${f(calibration.observed_mean,4)}, with interval ${intervalText(calibration.observed_ci_2_5,calibration.observed_ci_97_5)} and 64-test max-T p=${pText(calibration.observed_joint_max_t_p)} (${observedPass?"passes":"does not pass"}).`;
    const samplerStatus=fullSamplerPass?(overallGated?"Test passes · restricted":"Pass"):(overallGated?"Not passed · restricted":"Not passed");
    ladder.push(evidenceRung(3,"Response above sampler noise",samplerStatus,fullSamplerPass?(overallGated?"caution":"pass"):"fail",`${observedDetail} Observed magnitude is compared with the per-worm worst of low–low, high–high, and midpoint–midpoint controls. Sampler excess ${f(calibration.sampling_excess_mean,4)}, interval ${intervalText(calibration.sampling_excess_ci_2_5,calibration.sampling_excess_ci_97_5)}, separate 64-test max-T p=${pText(calibration.sampling_excess_joint_max_t_p)} (${passesSampler?"passes":"does not pass"}).${gateDetail}`,`N128 effect ${f(calibration.observed_mean,4)}, 95% interval ${intervalText(calibration.observed_ci_2_5,calibration.observed_ci_97_5)}; sampler-excess p=${pText(calibration.sampling_excess_joint_max_t_p)}.`));
    const quietStatus=passesQuiet?(overallGated?"Test passes · restricted":"Pass"):(overallGated?"Not passed · restricted":"Not passed");
    ladder.push(evidenceRung(4,"Quiet-time specificity",quietStatus,passesQuiet?(overallGated?"caution":"pass"):"fail",`Excess over the quiet pseudo-boundary is ${f(calibration.temporal_specificity_excess_mean,4)}, interval ${intervalText(calibration.temporal_specificity_ci_2_5,calibration.temporal_specificity_ci_97_5)}, separate 64-test max-T p=${pText(calibration.temporal_specificity_joint_max_t_p)} (${passesQuiet?"passes":"does not pass"}).${gateDetail} This cell is baseline, so this is not evidence of stimulus modulation. Not passing does not mean the learned effect is zero.`,`Quiet-time excess ${f(calibration.temporal_specificity_excess_mean,4)}; p=${pText(calibration.temporal_specificity_joint_max_t_p)}. Not a stimulus-modulation test.`));
  }else if(metricRows.length){
    ladder.push(evidenceRung(3,"Response above sampler noise","Outcome not calibrated","missing",`This cell was calibrated only for ${metricRows.map(row=>pretty(row.metric,channelLabels)).join(", ")}. The selected ${pretty(state.channel,channelLabels).toLowerCase()} outcome was not tested.`));
    ladder.push(evidenceRung(4,"Quiet-time specificity","Outcome not calibrated","missing","No quiet-time comparison exists for this exact metric. A comparison from another metric is not substituted."));
  }else{
    ladder.push(evidenceRung(3,"Response above sampler noise","Not run","missing","No matched N128 signed-response or low–low, high–high, and midpoint–midpoint calibration exists for this exact cell and metric."));
    ladder.push(evidenceRung(4,"Quiet-time specificity","Not run","missing","No matched quiet pseudo-boundary comparison exists. A result from another cell or metric is not substituted."));
  }
  ladder.push(evidenceRung(5,"Biological validation","Not tested","missing","No smallest effect size of interest (SESOI) has been declared, and no independent biological experiment confirms the prediction. This calibration is selection-conditioned and model-relative—not a biological null, causal effect, or physical delay."));
  document.getElementById("evidence-ladder").innerHTML=ladder.join("");
  const nextOutcome=state.channel==="endpoint_wasserstein1"?"predeclare a distributional endpoint such as Wasserstein distance":"predeclare the signed target outcome and a biologically meaningful minimum effect",calibrationState=calibration&&String(calibration.evidence_label),calibrationGate=calibration&&String(calibration.gate_reason),nextScope=calibrationState==="exceeds_sampling_and_quiet_controls"?"This clears both fitted-model controls; among the current outputs it is a candidate for preregistered independent replication, not an immediate causal intervention claim.":calibrationState==="exceeds_sampling_controls_only"?"It separates from sampler noise but not the quiet-time control; resolve temporal specificity in independent data before prioritizing a biological perturbation.":calibrationState==="indistinguishable_from_sampling_controls"?"It does not clear the full fitted-model calibration; prioritize sampler and model diagnosis over a biological perturbation.":calibrationState==="sampling_limited"&&calibrationGate==="support_failure"?"Its numerical result is claim-gated because the support/genealogy gate failed; seek stable support in more independent data.":calibrationState==="sampling_limited"&&calibrationGate==="sensitivity_origin"?"Its numerical result is claim-gated because this was a sensitivity-origin selection; independently predeclare and replicate it before promotion.":calibrationState==="sampling_limited"?"Its numerical result is claim-gated by both sensitivity-origin selection and a support/genealogy failure; resolve both before promotion.":current.edge?"The current complete-family result supports an edge-level replication question, but not a claim about which lag is the physical delay.":"Treat this as a model-method follow-up until an edge-level result is independently reproduced.";document.getElementById("evidence-next-step").innerHTML=`${e(nextScope)} Before an experiment, ${e(nextOutcome)}. Feasibility and independent confirmation remain required.${state.context==="baseline"?" This baseline result does not define a stimulus-modulation test.":""}`;
  const metricDetails=document.getElementById("sampling-metric-details");metricDetails.hidden=!metricRows.length;
  document.getElementById("sampling-metric-body").innerHTML=metricRows.map(row=>{const observedPass=signedObservedPass(row),samplerPass=positiveExcessPass(row,"sampling_excess"),quietPass=positiveExcessPass(row,"temporal_specificity"),observedEvidence=observedPass===null?"Unsigned W1<br><span class=\"muted\">no signed zero-null test</span>":`p=${e(pText(row.observed_joint_max_t_p))}<br><span class="muted">${observedPass?"passes":"does not pass"}</span>`;return `<tr><td>${e(pretty(row.metric,channelLabels))}${String(row.metric)===String(state.channel)?'<br><span class="muted">current metric</span>':""}</td><td class="number">${f(row.observed_mean,4)}<br><span class="muted">${e(intervalText(row.observed_ci_2_5,row.observed_ci_97_5))}</span></td><td>${observedEvidence}</td><td class="number">${f(row.sampling_null_mean_magnitude,4)}</td><td>${f(row.sampling_excess_mean,4)}<br><span class="muted">${e(intervalText(row.sampling_excess_ci_2_5,row.sampling_excess_ci_97_5))} · p=${e(pText(row.sampling_excess_joint_max_t_p))} · ${samplerPass?"passes":"does not pass"}</span></td><td>${f(row.temporal_specificity_excess_mean,4)}<br><span class="muted">${e(intervalText(row.temporal_specificity_ci_2_5,row.temporal_specificity_ci_97_5))} · p=${e(pText(row.temporal_specificity_joint_max_t_p))} · ${quietPass?"passes":"does not pass"}</span></td><td>${e(calibrationLabel(row)||pretty(row.evidence_label))}</td></tr>`;}).join("");
  const localized=families.reduce((total,row)=>total+Number(row.joint_primary_cell_max_t_discoveries||0),0),lagCells=families.reduce((total,row)=>total+Number(row.joint_primary_lag_contrast_cell_discoveries||0),0),nWorms=families.length?families[0].n_worms:"—",patterns=families.length?families[0].sign_patterns:"—",stats=[[nWorms,"worms analysed"],[Number(patterns).toLocaleString(),"exact two-sided sign patterns"],[headline.joint_primary_edge_discoveries,"strong directed edges"],[localized,"lag×horizon cells differing from zero—not lag differences"],[headline.joint_primary_flat_lag_edge_discoveries,"non-flat lag edges"],[headline.experiment_ready,"experiment-ready predictions"]];
  document.getElementById("evidence-headline").innerHTML=stats.map(([value,text])=>`<div class="evidence-stat"><strong>${e(value)}</strong><span>${e(text)}</span></div>`).join("");
  const calibrationSummary=payload.sampling_null_calibration.summary||{},labelCounts=calibrationSummary.labels||{},both=Number(labelCounts.exceeds_sampling_and_quiet_controls||0),samplerOnly=Number(labelCounts.exceeds_sampling_controls_only||0),notSeparated=Number(labelCounts.indistinguishable_from_sampling_controls||0),gated=Number(labelCounts.sampling_limited||0);document.getElementById("evidence-practical-status").textContent=`Primary result: ${headline.joint_primary_edge_discoveries} strong edges, but no non-flat lag edges and no lag-contrast cells (${lagCells}). Across ${calibrationSummary.candidate_metric_rows||24} N128 metric rows, ${both} clear both fitted-model controls, ${samplerOnly} clear only the sampler control, ${notSeparated} do not clear the full calibration, and ${gated} are support- or sensitivity-gated. This separate 64-test analysis is selection-conditioned and does not correct the upstream selection. SESOI and independent confirmation are still missing, so none are experiment-ready.`;
  document.getElementById("evidence-family-body").innerHTML=families.map(row=>`<tr><td>${e(pretty(row.family_id,familyLabels))}</td><td class="number">${e(row.complete_edges)}</td><td class="number">${e(row.support_eligible_edges)}</td><td class="number">${e(row.joint_primary_edge_max_t_discoveries)}</td><td class="number">${e(row.joint_primary_cell_max_t_discoveries)}</td><td class="number">${e(row.joint_primary_flat_lag_edge_discoveries)}</td></tr>`).join("");
  const sensitivity=data.sensitivity_support05||{},sensitivityRows=sensitivity.named_lag_rows||[];document.getElementById("evidence-sensitivity").innerHTML=sensitivityRows.length?`<div class="scroll evidence-table" style="margin-top:14px"><table><thead><tr><th>Sensitivity-only connection</th><th>Timing</th><th class="number">Joint lag max-T p</th><th>Evidence</th></tr></thead><tbody>${sensitivityRows.map(row=>`<tr><td>${e(row.source_neuron)} → ${e(row.target_neuron)}</td><td>${e(seconds(row.source_lag_frames))} back · ${e(seconds(row.horizon_frames))} ahead</td><td class="number">${e(pText(row.joint_sensitivity_lag_contrast_max_t_p_value))}</td><td>${e(evidenceLabel(row)||evidenceLabelText.sampling_limited)}</td></tr>`).join("")}</tbody></table></div><p class="meta" style="margin-top:10px">${e(sensitivity.warning)}</p>`:"";
}
function renderMatrix(){
  const values=decode(),canvas=document.getElementById("matrix-canvas"),ctx=canvas.getContext("2d"),cell=canvas.width/n,limit=Math.max(...values.map(Math.abs),1e-12),unsigned=unsignedChannel();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for(let target=0;target<n;target++)for(let source=0;source<n;source++){ctx.fillStyle=color(values[target*n+source],limit,unsigned);ctx.fillRect(source*cell,target*cell,Math.ceil(cell),Math.ceil(cell));}
  ctx.strokeStyle="#f7d774";ctx.lineWidth=3;ctx.strokeRect(state.source*cell+1,state.target*cell+1,cell-2,cell-2);
  const value=values[state.target*n+state.source],direction=unsigned?"unsigned distance":Number(value)>=0?"higher target response":"lower target response";
  document.getElementById("matrix-readout").textContent=`${atlas.neurons[state.source]} → ${atlas.neurons[state.target]}: ${direction}, ${f(value,6)} normalized channel effect. This is target row ${state.target}, source column ${state.source}.`;
  document.getElementById("slice-label").textContent=`${pretty(state.method,methodLabels)} · ${seconds(state.lag)} back / ${seconds(state.horizon)} ahead`;
  document.getElementById("matrix-legend-bar").classList.toggle("unsigned",unsigned);
  document.getElementById("legend-left").textContent=unsigned?"zero":"lower";
  document.getElementById("legend-right").textContent=unsigned?"farther apart":"higher";
  document.getElementById("legend-negative").textContent=unsigned?"0":`−${f(limit,4)}`;
  document.getElementById("legend-positive").textContent=`+${f(limit,4)}`;
  document.getElementById("matrix-scale-note").textContent=unsigned?"Per-slice scale · unsigned distance. Compare color intensity only within this matrix.":"Per-slice scale · blue means lower and red means higher. Compare color intensity only within this matrix.";
}
function renderSurface(){
  const all=[];for(const lag of atlas.source_lag_frames)for(const horizon of atlas.horizon_frames)all.push(decode(state.method,state.channel,state.context,lag,horizon)[state.target*n+state.source]);
  const limit=Math.max(...all.map(Math.abs),1e-12);
  let html="<table><thead><tr><th>source history ↓ / target forecast →</th>"+atlas.horizon_frames.map(x=>`<th>${seconds(x)} ahead</th>`).join("")+"</tr></thead><tbody>";
  for(const lag of atlas.source_lag_frames){html+=`<tr><th>${seconds(lag)} back</th>`;for(const horizon of atlas.horizon_frames){const value=decode(state.method,state.channel,state.context,lag,horizon)[state.target*n+state.source],selected=lag===state.lag&&horizon===state.horizon;html+=`<td><button class="timing-cell${selected?" selected":""}" data-lag="${lag}" data-horizon="${horizon}" aria-pressed="${selected}" aria-label="${e(seconds(lag))} back and ${e(seconds(horizon))} ahead: effect ${e(f(value,4))}" style="background:${color(value,limit,unsignedChannel())}">${Number(value)>=0?"+":""}${f(value,4)}</button></td>`;}html+="</tr>";}
  document.getElementById("cell-surface").innerHTML=html+"</tbody></table>";
  document.getElementById("cell-label").textContent=`${atlas.neurons[state.source]} → ${atlas.neurons[state.target]} · ${pretty(state.channel,channelLabels)} · ${pretty(state.stratum,stratumLabels)} · ${pretty(state.phase,phaseLabels)} · colors share one ±${f(limit,4)} limit only within this grid`;
  document.querySelectorAll(".timing-cell").forEach(button=>button.addEventListener("click",()=>{markCustom();state.lag=Number(button.dataset.lag);state.horizon=Number(button.dataset.horizon);sync();renderAll();}));
}
function renderDownstream(){const values=decode(),rows=atlas.neurons.map((name,target)=>({name,target,value:values[target*n+state.source]})).filter(x=>x.target!==state.source).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)).slice(0,18);document.getElementById("downstream-body").innerHTML=rows.map((x,i)=>`<tr><td>${i+1}</td><td><button class="row target-pick" data-target="${x.target}">${e(x.name)}</button></td><td class="number ${rowClass(x.value,unsignedChannel())}">${Number(x.value)>=0?"+":""}${f(x.value,6)}</td></tr>`).join("");document.querySelectorAll(".target-pick").forEach(button=>button.addEventListener("click",()=>{markCustom();state.target=Number(button.dataset.target);sync();renderAll();}));}
function rowClass(value,unsigned=false){return unsigned?"unsigned":Number(value)>=0?"positive":"negative";}
function renderQueue(){
  const reviewedRanks=new Set(payload.targeted_n128.cells.map(x=>Number(x.queue_rank))),indexed=payload.candidate_queue.map((x,index)=>({x,index}));
  let visible=queueExpanded?indexed:indexed.filter(item=>reviewedRanks.has(Number(item.x.queue_rank)));if(!visible.length&&!queueExpanded)visible=indexed.slice(0,25);
  document.getElementById("queue-count").textContent=`Showing ${visible.length} of ${indexed.length}`;
  document.getElementById("queue-toggle").textContent=queueExpanded?`Show ${reviewedRanks.size} reviewed candidates`:`Show all ${indexed.length} screening candidates`;
  document.getElementById("queue-body").innerHTML=visible.map(({x,index})=>{const matched=evidenceForCandidate(x),calibrated=samplingForCandidate(x),label=calibrationLabel(calibrated)||evidenceLabel(matched)||"Legacy screen only";return `<tr><td><button class="row candidate-pick" data-index="${index}">#${e(x.queue_rank)}</button></td><td>${e(x.source_neuron)} → ${e(x.target_neuron)}</td><td>${e(pretty(x.context,phaseLabels))}</td><td>${e(seconds(x.source_lag_frames))} back<br><span class="muted">${e(seconds(x.horizon_frames))} ahead</span></td><td class="number ${rowClass(x.mean_normalized,unsignedMetric(String(x.channel),String(x.context)))}">${Number(x.mean_normalized)>=0?"+":""}${f(x.mean_normalized,5)}</td><td class="queue-evidence">${e(label)}</td><td>${e(pretty(x.support_tier,supportLabels))}</td></tr>`;}).join("");
  document.querySelectorAll(".candidate-pick").forEach(button=>button.addEventListener("click",()=>selectCandidate(Number(button.dataset.index))));
  renderQueueDetail();
}
function selectCandidate(index){const x=payload.candidate_queue[index];if(!x)return;state.queueRank=Number(x.queue_rank);for(const [key,value] of [["method",x.method],["channel",x.channel]])if((key==="method"?atlas.methods:atlas.channels).includes(String(value)))state[key]=String(value);if(atlas.contexts.includes(String(x.context)))setCompositeContext(String(x.context));if(atlas.source_lag_frames.includes(Number(x.source_lag_frames)))state.lag=Number(x.source_lag_frames);if(atlas.horizon_frames.includes(Number(x.horizon_frames)))state.horizon=Number(x.horizon_frames);state.source=Number(x.source_index);state.target=Number(x.target_index);sync();renderAll();}
function renderQueueDetail(){
  const x=state.queueRank===null?null:payload.candidate_queue.find(r=>Number(r.queue_rank)===state.queueRank);
  if(!x){document.getElementById("queue-detail").innerHTML='<div class="empty">Select a candidate to see its frozen record and N128 follow-up.</div>';return;}
  const value=Number(x.mean_normalized),direction=unsignedMetric(String(x.channel),String(x.context))?"unsigned distributional distance":value>=0?"higher target response":"lower target response",matched=evidenceForCandidate(x),calibrated=samplingForCandidate(x),statisticalLabel=calibrationLabel(calibrated)||evidenceLabel(matched)||"Legacy screen only";
  const fields=[["Connection",`${x.source_neuron} → ${x.target_neuron}`],["Outcome",pretty(x.channel,channelLabels)],["Context",pretty(x.context,phaseLabels)],["Timing",`${seconds(x.source_lag_frames)} back · ${seconds(x.horizon_frames)} ahead`],["Effect",`${f(value,6)} · ${direction}`],["Statistical evidence",statisticalLabel],["Atlas-screen support",pretty(x.support_tier,supportLabels)]];
  let html=`<div class="candidate-summary"><strong>Frozen candidate #${e(x.queue_rank)}</strong><p style="margin:6px 0 12px">${effectPhrase(value,String(x.channel),String(x.context),String(x.target_neuron),String(x.source_neuron))}</p><button class="primary-button" id="open-candidate" type="button">Explore this connection</button></div><div class="detail-grid">`+fields.map(([label,value])=>`<div class="datum"><span>${e(label)}</span>${e(value)}</div>`).join("")+"</div>";
  const legacyQ=matched&&matched.legacy_postscreen_q_value!==undefined?matched.legacy_postscreen_q_value:x.sign_flip_q_value,screenFields=[["Legacy post-screen q value",f(legacyQ,7)],["Screen 95% interval",`[${f(x.ci_2_5,5)}, ${f(x.ci_97_5,5)}]`],["Inference scope",x.inference_scope||"post-screen exploratory"]];
  html+=`<details class="screen-disclosure"><summary>Post-screen exploratory result</summary><div class="details-body"><p class="screen-warning">${e(payload.statistical_evidence.legacy_queue.warning)}</p><div class="detail-grid">${screenFields.map(([label,item])=>`<div class="datum"><span>${e(label)}</span>${e(item)}</div>`).join("")}</div></div></details>`;
  const technical=["valid_fraction","sign_consistency","seed_sign_agreement","interpretation_limit",...Object.keys(x).filter(k=>k.includes("genealogy")||k.includes("ancestry"))];
  html+='<details style="margin-top:14px"><summary>Show technical screening fields</summary><div class="details-body"><div class="detail-grid">'+technical.filter(k=>k in x).map(k=>`<div class="datum"><span>${e(k.replaceAll("_"," "))}</span>${e(f(x[k],6)==="—"?x[k]:f(x[k],6))}</div>`).join("")+"</div></div></details>";
  document.getElementById("queue-detail").innerHTML=html;
  document.getElementById("open-candidate").addEventListener("click",()=>{switchView("explore");document.getElementById("explore-panel").scrollIntoView({block:"start"});});
}
function presentColumns(rows,preferred){const seen=new Set(rows.flatMap(x=>Object.keys(x)));return preferred.filter(x=>seen.has(x));}
function renderTargeted(){
  const data=payload.targeted_n128,summary=data.summary||{},summaryFields=[["n_candidates","Frozen candidates"],["n_cell_rows","Outcome rows"],["n_quantile_rows","Quantile rows"],["n_worms","Worms"]];
  document.getElementById("targeted-summary").innerHTML=summaryFields.filter(([key])=>key in summary).map(([key,label])=>`<div class="datum"><span>${e(label)}</span>${e(summary[key])}</div>`).join("");
  const selectedCells=state.queueRank===null?[]:data.cells.filter(x=>Number(x.queue_rank)===state.queueRank),hasCandidate=selectedCells.length>0,selectedCandidate=hasCandidate?selectedCells[0]:null,candidateIds=new Set(selectedCells.map(x=>String(x.candidate_id))),rows=selectedCells;
  document.getElementById("targeted-interpretation").textContent=state.queueRank===null?"Select a reviewed candidate for its N128 follow-up.":hasCandidate?`${rows.length} N128 outcome rows are available for frozen candidate #${state.queueRank}. Open the tables below only when you need metric-by-metric or sampler diagnostics.`:`Frozen candidate #${state.queueRank} was not one of the candidates selected for the N128 follow-up.`;
  document.getElementById("targeted-body").innerHTML=rows.map(x=>`<tr><td>${e(x.candidate_id)}<br><span class="muted">${e(x.source_neuron)} → ${e(x.target_neuron)}</span></td><td>${e(pretty(x.contrast,phaseLabels))}</td><td>${e(pretty(x.metric,channelLabels))}</td><td class="number ${rowClass(x.mean_normalized,String(x.metric)==="endpoint_wasserstein1"&&!Boolean(x.signed))}">${Number(x.mean_normalized)>=0?"+":""}${f(x.mean_normalized,5)}</td><td>[${f(x.ci_2_5,4)}, ${f(x.ci_97_5,4)}]</td><td>${e(x.evidence_label)}</td></tr>`).join("")||'<tr><td colspan="6" class="empty">This queue row was not one of the frozen N128 candidates.</td></tr>';
  const quantiles=hasCandidate?data.quantile_shifts.filter(x=>candidateIds.has(String(x.candidate_id))):[],quantileColumns=presentColumns(quantiles,["candidate_id","contrast","endpoint_quantile","mean_normalized","ci_2_5","ci_97_5","valid_fraction"]);
  document.getElementById("targeted-quantile-table").innerHTML=genericTable(quantileColumns,quantiles);
  const support=hasCandidate?data.support_diagnostics.filter(x=>Number(x.source_index)===Number(selectedCandidate.source_index)&&Number(x.source_lag_frames)===Number(selectedCandidate.source_lag_frames)&&String(x.context)===String(selectedCandidate.context)):[],supportKeys=[...new Set(support.flatMap(x=>Object.keys(x)))],genealogy=supportKeys.filter(x=>x.includes("genealogy")||x.includes("ancestry")),supportColumns=presentColumns(support,["source_neuron","source_lag_frames","context","valid_fraction","mean_achieved_gap_magnitude","mean_ess_low","mean_ess_high",...genealogy]);
  document.getElementById("targeted-support-table").innerHTML=genericTable(supportColumns,support);
  const screen=hasCandidate?data.screen_consistency.filter(x=>candidateIds.has(String(x.candidate_id))):[],screenColumns=presentColumns(screen,["candidate_id","screen_method","targeted_method","comparison_type","screen_mean_normalized","targeted_n128_mean_normalized","targeted_minus_screen","direction_agreement","magnitude_agreement","screen_valid_fraction","targeted_valid_fraction","consistency_label"]);
  document.getElementById("targeted-screen-table").innerHTML=genericTable(screenColumns,screen);
}
function genericTable(columns,rows){if(!rows.length)return '<div class="empty">No rows are available for this view.</div>';return '<div class="scroll"><table><thead><tr>'+columns.map(x=>`<th>${e(x.replaceAll("_"," "))}</th>`).join("")+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+columns.map(x=>`<td>${e(row[x])}</td>`).join("")+'</tr>').join("")+'</tbody></table></div>';}
function renderComposition(){const data=payload.stimulus_composition;if(data.status!=="included"){document.getElementById("composition-status").textContent=data.status.replaceAll("_"," ");document.getElementById("composition-table").innerHTML=genericTable(data.columns,data.rows);return;}let rows=data.rows.filter(x=>Number(x.source_lag_frames)===state.lag&&Number(x.horizon_frames)===state.horizon),phaseRows=rows.filter(x=>String(x.phase)===state.phase);if(phaseRows.length)rows=phaseRows;const phaseScope=state.phase==="state_average"?"all five phases":phaseRows.length?pretty(state.phase,phaseLabels):"all phases";document.getElementById("composition-status").textContent=`Pooled chemicals · ${phaseScope} · ${rows.length} row${rows.length===1?"":"s"} · ${seconds(state.lag)} history / ${seconds(state.horizon)} forecast`;document.getElementById("composition-table").innerHTML=genericTable(data.columns,rows);}
function renderExternal(){const data=payload.external_references;document.getElementById("external-body").innerHTML=data.comparisons.map(x=>`<tr><td>${e(x.panel)}</td><td>${e(x.reference_or_network)}</td><td>${e(x.timing_label)}</td><td class="number">${f(x.auroc,4)}</td><td class="number">${f(x.auprc,4)}</td><td class="number">${f(x.continuous_spearman,4)}</td><td class="number">${f(x.bh_q,4)}</td></tr>`).join("");const rows=data.lagmax_inference,columns=rows.length?Object.keys(rows[0]):[];document.getElementById("lagmax-table").innerHTML=rows.length?'<thead><tr>'+columns.map(x=>`<th>${e(x.replaceAll("_"," "))}</th>`).join("")+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+columns.map(x=>`<td>${e(row[x])}</td>`).join("")+'</tr>').join("")+'</tbody>':'<tbody><tr><td class="empty">No lag-max rows.</td></tr></tbody>';}
function renderProvenance(){document.getElementById("claim-list").innerHTML=payload.claim_boundaries.map(x=>`<div class="claim"><strong>${e(x.label)}</strong>${e(x.detail)}</div>`).join("");document.getElementById("provenance-body").innerHTML=payload.provenance.map(x=>`<tr><td>${e(x.bundle)}</td><td>${e(x.path)}</td><td><code>${e(x.manifest_sha256)}</code></td><td><code>${e(x.checksums_sha256)}</code></td><td>${e(x.verified_files)}</td></tr>`).join("");}
function renderAll(){renderSummary();renderCalibrationPicks();renderStatisticalEvidence();renderMatrix();renderSurface();renderDownstream();renderQueueDetail();renderTargeted();renderComposition();}
function switchView(view){
  document.querySelectorAll(".view-panel").forEach(panel=>{panel.hidden=panel.id!==`${view}-panel`;});
  document.querySelectorAll(".tab").forEach(tab=>{tab.setAttribute("aria-selected",String(tab.dataset.view===view));});
  if(view==="references"&&!externalRendered){renderExternal();externalRendered=true;}
  if(view==="methods"&&!provenanceRendered){renderBoundaries();renderProvenance();provenanceRendered=true;}
}
for(const key of ["method","channel"]){controls[key].addEventListener("change",()=>{markCustom();state[key]=controls[key].value;sync();renderAll();});}
controls.stratum.addEventListener("change",()=>{markCustom();const available=atlas.context_options.filter(x=>x.event_stratum===controls.stratum.value),chosen=available.find(x=>x.phase_or_contrast===state.phase)||available[0];setCompositeContext(chosen.context);sync();renderAll();});
controls.phase.addEventListener("change",()=>{markCustom();const chosen=atlas.context_options.find(x=>x.event_stratum===state.stratum&&x.phase_or_contrast===controls.phase.value);setCompositeContext(chosen.context);sync();renderAll();});
for(const key of ["lag","horizon","source","target"]){controls[key].addEventListener("change",()=>{markCustom();state[key]=Number(controls[key].value);renderAll();});}
document.querySelectorAll(".tab").forEach(tab=>tab.addEventListener("click",()=>switchView(tab.dataset.view)));
document.getElementById("queue-toggle").addEventListener("click",()=>{queueExpanded=!queueExpanded;renderQueue();});
const canvas=document.getElementById("matrix-canvas");
canvas.addEventListener("click",event=>{const box=canvas.getBoundingClientRect();markCustom();state.source=Math.min(n-1,Math.max(0,Math.floor((event.clientX-box.left)/box.width*n)));state.target=Math.min(n-1,Math.max(0,Math.floor((event.clientY-box.top)/box.height*n)));sync();renderAll();});
canvas.addEventListener("mousemove",event=>{const box=canvas.getBoundingClientRect(),source=Math.min(n-1,Math.max(0,Math.floor((event.clientX-box.left)/box.width*n))),target=Math.min(n-1,Math.max(0,Math.floor((event.clientY-box.top)/box.height*n))),value=decode()[target*n+source],direction=unsignedChannel()?"unsigned distance":Number(value)>=0?"higher":"lower";document.getElementById("matrix-readout").textContent=`${atlas.neurons[source]} → ${atlas.neurons[target]}: ${direction}, ${f(value,6)} normalized channel effect. Target row ${target}, source column ${source}.`;});
canvas.addEventListener("mouseleave",renderMatrix);
sync();renderQueue();renderAll();
</script>
</body>
</html>
'''


def _render_html(payload_json: str) -> str:
    embedded = (
        payload_json.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return _HTML_TEMPLATE.replace("__PAYLOAD__", embedded)


def build_prediction_atlas_explorer(
    atlas_dir: Path,
    targeted_analysis_dir: Path,
    postfreeze_external_dir: Path,
    complete_family_evidence_dir: Path,
    sampling_null_analysis_dir: Path,
    output_dir: Path,
    *,
    provenance_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate all canonical inputs and write a local-only explorer bundle."""

    provenance = (
        Path.cwd().resolve() if provenance_root is None else Path(provenance_root).resolve()
    )
    atlas = _validate_atlas(Path(atlas_dir))
    complete_family = _validate_complete_family_evidence(
        Path(complete_family_evidence_dir), atlas=atlas
    )
    sampling_null = _validate_sampling_null_evidence(
        Path(sampling_null_analysis_dir),
        atlas=atlas,
        complete_family=complete_family,
        provenance_root=provenance,
    )
    targeted = _validate_targeted(
        Path(targeted_analysis_dir),
        n_worms=atlas.n_worms,
        neurons=atlas.neurons,
        atlas_dir=atlas.root,
        atlas_checksums=atlas.checksums,
        provenance_root=provenance,
    )
    external = _validate_external(Path(postfreeze_external_dir), atlas)
    for path in (
        atlas.root,
        targeted.root,
        external.root,
        complete_family.root,
        sampling_null.root,
        sampling_null.raw_root,
    ):
        _safe_relative(path, provenance)
    _validate_output_location(
        Path(output_dir),
        (
            atlas.root,
            targeted.root,
            external.root,
            complete_family.root,
            sampling_null.root,
            sampling_null.raw_root,
        ),
    )

    generated = datetime.now(timezone.utc).isoformat()
    payload = _build_payload(
        atlas,
        targeted,
        external,
        complete_family,
        sampling_null,
        provenance_root=provenance,
        generated_utc=generated,
    )
    payload_json = json.dumps(
        _json_safe(payload), separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ) + "\n"
    html = _render_html(payload_json)
    output = _prepare_output(Path(output_dir), overwrite=overwrite)
    payload_name = "explorer_data.json"
    html_name = "atlas_explorer.html"
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": generated,
        "offline": True,
        "network_dependencies": [],
        "payload_schema_version": SCHEMA_VERSION,
        "inputs": payload["provenance"],
        "optional_stimulus_composition": payload["stimulus_composition"]["status"],
        "dense_encoding": payload["atlas"]["encoding"],
        "matrix_orientation": ORIENTATION,
        "outputs": {
            "html": {
                "file": html_name,
                "sha256": _sha256_text(html),
                "bytes": len(html.encode("utf-8")),
            },
            "payload": {
                "file": payload_name,
                "sha256": _sha256_text(payload_json),
                "bytes": len(payload_json.encode("utf-8")),
            },
        },
    }
    manifest_json = json.dumps(
        _json_safe(manifest), indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    _atomic_text(output / payload_name, payload_json)
    _atomic_text(output / html_name, html)
    _atomic_text(output / "manifest.json", manifest_json)
    ledger = "".join(
        f"{sha256(output / name)}  {name}\n"
        for name in (payload_name, html_name, "manifest.json")
    )
    _atomic_text(output / "checksums.sha256", ledger)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--targeted-analysis-dir", type=Path, required=True)
    parser.add_argument("--postfreeze-external-dir", type=Path, required=True)
    parser.add_argument("--complete-family-evidence-dir", type=Path, required=True)
    parser.add_argument("--sampling-null-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance-root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_prediction_atlas_explorer(
        args.atlas_dir,
        args.targeted_analysis_dir,
        args.postfreeze_external_dir,
        args.complete_family_evidence_dir,
        args.sampling_null_analysis_dir,
        args.output_dir,
        provenance_root=args.provenance_root,
        overwrite=args.overwrite,
    )
    print(str((args.output_dir / "atlas_explorer.html").resolve()))


if __name__ == "__main__":
    main()
