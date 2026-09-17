"""Freeze a bounded internal-only shortlist for N128 targeted confirmation.

The selector reads only the canonical prediction-atlas bundle.  It does not
load, join, or score against Randi, Cook, Bentley, SBTG, neuromodulator, or any
other external reference.  Selection is deterministic: promotion eligibility
is preferred first, followed by the frozen atlas evidence score, with channel
and source diversity applied as an explicit bounded design rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


SCHEMA_VERSION = "prediction_atlas_targeted_selection_v1"
SELECTION_FILENAME = "hypothesis_queue.csv"
RATIONALE_FILENAME = "selection_rationale.md"
MANIFEST_FILENAME = "manifest.json"
CHECKSUM_FILENAME = "checksums.sha256"
PARTICLES = 128
MAX_SELECTION_ROWS = 6
PHASES = ("baseline", "onset", "active", "offset", "recovery")
ORIENTATION = "target_row_source_column"
CONFIRMATION_METHOD = "progressive_bridge_smc_targeted_three_arm"
SUPPORTED_CHANNELS = (
    "endpoint_mean",
    "cumulative_mean",
    "peak_mean",
    "event_probability",
    "endpoint_sd",
    "endpoint_log_sd",
    "endpoint_wasserstein1",
)
SUPPORTED_CONTEXTS = (
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
# Exact schema emitted by prediction_atlas_analysis.  An allowlist is safer than
# trying to enumerate every possible name an external-reference join could add.
CANONICAL_QUEUE_COLUMNS = (
    "queue_rank",
    "method",
    "model_id",
    "channel",
    "context",
    "chemical",
    "conditioning_status",
    "source_neuron",
    "target_neuron",
    "source_index",
    "target_index",
    "source_lag_frames",
    "source_to_cut_seconds",
    "horizon_frames",
    "forecast_horizon_seconds",
    "source_to_readout_seconds",
    "mean_raw",
    "mean_normalized",
    "median_normalized",
    "ci_2_5",
    "ci_97_5",
    "screen_t_p_value",
    "sign_flip_p_value",
    "sign_flip_q_value",
    "bh_family",
    "inference_scope",
    "test_sidedness",
    "sign_consistency",
    "valid_fraction",
    "genealogy_gate_applicable",
    "genealogy_min_distinct_ancestor_fraction_strong",
    "genealogy_min_distinct_ancestor_fraction_sensitivity",
    "genealogy_valid_fraction_0_10",
    "genealogy_valid_fraction_0_20",
    "genealogy_strong_gate_pass",
    "genealogy_sensitivity_gate_pass",
    "seed_spearman",
    "seed_sign_agreement",
    "n_worms",
    "effect_direction",
    "support_tier",
    "evidence_score",
    "interpretation_limit",
    "counterpart_method",
    "counterpart_mean_normalized",
    "counterpart_valid_fraction",
    "cross_sampler_sign_agreement",
    "cross_sampler_magnitude_agreement",
    "cross_sampler_spearman",
    "cross_sampler_slice_agreement",
    "cross_sampler_support_agreement",
    "cross_sampler_factor",
    "base_evidence_score",
    "promotion_eligible",
    "lag_profile_frames",
    "lag_profile_source_to_cut_seconds",
    "primary_lag_profile",
    "counterpart_lag_profile",
    "peak_abs_effect_lag_frames",
    "peak_abs_effect_lag_seconds",
    "peak_abs_effect",
    "second_abs_effect",
    "top_vs_second_lag_selectivity",
    "lag_profile_spearman",
    "signed_lag_profile_spearman",
    "worm_bootstrap_peak_lag_selection_rate",
    "worm_bootstrap_peak_lag_rates",
    "lag_interpretation",
)
REQUIRED_QUEUE_COLUMNS = (
    "queue_rank",
    "method",
    "channel",
    "context",
    "source_neuron",
    "target_neuron",
    "source_index",
    "target_index",
    "source_lag_frames",
    "horizon_frames",
    "promotion_eligible",
    "evidence_score",
    "genealogy_gate_applicable",
    "genealogy_strong_gate_pass",
)
ADDED_COLUMNS = ("run_confirmation",)
SELECTION_METADATA_COLUMNS = (
    "selection_rank",
    "selection_rule",
    "frozen_hypothesis_queue_sha256",
)
REQUIRED_ATLAS_FILES = (
    "manifest.json",
    "protocol.json",
    "validation.json",
    "models.json",
    "atlas_matrices.npz",
    "hypothesis_queue.csv",
)
EXTERNAL_FIELD_EXACT = {
    "reference",
    "reference_or_network",
    "network",
    "panel",
    "auroc",
    "auprc",
    "permutation_p",
    "bh_q",
    "scope",
    "scope_or_grid",
    "training_lineage",
    "shared_neuron_comparability",
    "comparison_note",
    "continuous_spearman",
    "absolute_spearman",
    "best_auroc",
    "best_lag_frames",
    "max_lag_permutation_p",
    "max_lag_bh_q",
    "null_max_mean",
    "null_max_95pct",
}
EXTERNAL_FIELD_TOKEN = re.compile(
    r"(?:^|_)(?:randi|cook|bentley|sbtg|connectome|neuromodulator|"
    r"external_reference|external_metric|lagmax|max_lag)(?:_|$)",
    re.IGNORECASE,
)
CLAIM_BOUNDARY = (
    "internal model-relative shortlist for higher-particle confirmation; not an "
    "external-reference ranking, causal effect, anatomical connection, or physical delay"
)


class TargetedSelectionError(RuntimeError):
    """Raised when an atlas cannot be safely frozen for confirmation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise TargetedSelectionError(f"could not read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise TargetedSelectionError(f"{path.name} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _reject_external_fields(names: Iterable[object], *, label: str) -> None:
    forbidden: list[str] = []
    for raw in names:
        name = str(raw).strip()
        normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if normalized in EXTERNAL_FIELD_EXACT or EXTERNAL_FIELD_TOKEN.search(normalized):
            forbidden.append(name)
    if forbidden:
        raise TargetedSelectionError(
            f"{label} contains prohibited external-reference fields: "
            + ", ".join(sorted(forbidden))
        )


def _parse_checksum_inventory(atlas_dir: Path) -> dict[str, str]:
    path = atlas_dir / CHECKSUM_FILENAME
    if not path.is_file():
        raise TargetedSelectionError("canonical atlas lacks checksums.sha256")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise TargetedSelectionError(
                f"checksums.sha256 line {line_number} is malformed"
            )
        relative = Path(parts[1])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 1
            or not relative.name
        ):
            raise TargetedSelectionError("checksum inventory contains an unsafe path")
        name = relative.name
        if name in entries:
            raise TargetedSelectionError(f"checksum inventory duplicates {name}")
        entries[name] = parts[0]
    missing = sorted(set(REQUIRED_ATLAS_FILES).difference(entries))
    if missing:
        raise TargetedSelectionError(
            f"checksum inventory does not cover canonical inputs: {missing}"
        )
    # Reject any post-freeze external inventory before opening its listed files.
    _reject_external_fields(entries, label="checksum inventory")
    for name, expected in entries.items():
        source = atlas_dir / name
        if not source.is_file():
            raise TargetedSelectionError(f"checksum inventory file is missing: {name}")
        if sha256(source) != expected:
            raise TargetedSelectionError(f"canonical checksum failed for {name}")
    return entries


def _strict_boolean(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise TargetedSelectionError(f"{label} must contain only canonical booleans")


def _finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TargetedSelectionError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise TargetedSelectionError(f"{label} must be finite")
    return result


def _integer(value: object, *, label: str, minimum: int) -> int:
    number = _finite_float(value, label=label)
    if not number.is_integer() or number < minimum:
        raise TargetedSelectionError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return int(number)


def targeted_phases_for_context(context: object) -> tuple[str, ...]:
    """Resolve exactly the contexts accepted by targeted_confirmation."""

    value = str(context).strip().lower()
    if value not in SUPPORTED_CONTEXTS:
        raise TargetedSelectionError(
            f"queue context {value!r} is not one of the canonical targeted contexts"
        )
    if value in PHASES:
        return (value,)
    if value == "state_average":
        return PHASES
    if value == "onset_minus_baseline" or value.endswith("_onset_minus_baseline"):
        return ("baseline", "onset")
    if value.endswith("_onset"):
        return ("onset",)
    raise TargetedSelectionError(
        f"queue context {value!r} is incompatible with targeted confirmation"
    )


def _validate_atlas_metadata(
    atlas_dir: Path,
    *,
    queue_rows: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read_json(atlas_dir / "manifest.json")
    protocol = _read_json(atlas_dir / "protocol.json")
    validation = _read_json(atlas_dir / "validation.json")
    models = _read_json(atlas_dir / "models.json")
    if manifest.get("status") != "complete":
        raise TargetedSelectionError("canonical atlas manifest is not complete")
    if manifest.get("protocol") != "reviewed neural prediction atlas v1":
        raise TargetedSelectionError("atlas manifest protocol version is not canonical")
    if manifest.get("primary_method") != "progressive_bridge_smc":
        raise TargetedSelectionError("atlas manifest primary method is not progressive bridge SMC")
    if manifest.get("dense_matrix_orientation") != ORIENTATION:
        raise TargetedSelectionError("atlas manifest orientation is not target-row/source-column")
    artifacts = manifest.get("artifacts")
    expected_artifacts = {
        "hypothesis_queue": "hypothesis_queue.csv",
        "atlas_matrices": "atlas_matrices.npz",
        "models": "models.json",
        "protocol": "protocol.json",
        "validation": "validation.json",
    }
    if not isinstance(artifacts, dict) or any(
        artifacts.get(key) != value for key, value in expected_artifacts.items()
    ):
        raise TargetedSelectionError("atlas manifest artifact mapping is not canonical")
    if protocol.get("ranking") != "no connectome or external reference data used":
        raise TargetedSelectionError("canonical internal-ranking firewall is absent")
    if protocol.get("orientation") != ORIENTATION:
        raise TargetedSelectionError("atlas protocol orientation is not target-row/source-column")
    if protocol.get("n_worms") != 17 or protocol.get("n_neurons") != 54:
        raise TargetedSelectionError("selector requires the reviewed 17-worm, 54-neuron atlas")
    channels = protocol.get("channels")
    if channels != list(SUPPORTED_CHANNELS):
        raise TargetedSelectionError("atlas protocol channel grid is not canonical")
    contexts = protocol.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise TargetedSelectionError("atlas protocol lacks context metadata")
    protocol_contexts: set[str] = set()
    for item in contexts:
        if not isinstance(item, dict) or not isinstance(item.get("context"), str):
            raise TargetedSelectionError("atlas protocol contains malformed context metadata")
        targeted_phases_for_context(item["context"])
        protocol_contexts.add(item["context"])
    if protocol_contexts != set(SUPPORTED_CONTEXTS) or len(contexts) != len(
        SUPPORTED_CONTEXTS
    ):
        raise TargetedSelectionError("atlas protocol context grid is not canonical")
    warning = str(protocol.get("chemical_context_warning", "")).lower()
    interpretation = str(protocol.get("interpretation_limit", "")).lower()
    if "not chemically conditioned" not in warning:
        raise TargetedSelectionError("atlas protocol lost the chemical-conditioning boundary")
    if "not a causal" not in interpretation or "physical-delay" not in interpretation:
        raise TargetedSelectionError("atlas protocol lost the causal/timing claim boundary")
    evidence_tiers = protocol.get("evidence_tiers")
    if not isinstance(evidence_tiers, dict) or set(evidence_tiers) != {
        "confirmed",
        "supported_exploratory",
        "model_only",
        "unsupported",
    }:
        raise TargetedSelectionError("atlas protocol evidence tiers are not canonical")
    timing = protocol.get("timing_definitions")
    if not isinstance(timing, dict) or set(timing) != {
        "source_to_cut_seconds",
        "forecast_horizon_seconds",
        "source_to_readout_seconds",
    }:
        raise TargetedSelectionError("atlas protocol timing definitions are incomplete")
    if models.get("generator_stimulus_encoding") != "binary_any_stimulus":
        raise TargetedSelectionError("atlas generator stimulus encoding is not binary-any-stimulus")
    if models.get("chemical_conditioning") is not False:
        raise TargetedSelectionError("atlas incorrectly declares chemical conditioning")
    if validation.get("status") != "passed" or validation.get("problems") not in (
        [],
        None,
    ):
        raise TargetedSelectionError("canonical atlas validation did not pass cleanly")
    checks = validation.get("archive_checks_completed")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise TargetedSelectionError("canonical archive checks are absent or incomplete")
    if validation.get("n_worms") != 17 or validation.get("n_neurons") != 54:
        raise TargetedSelectionError("validation cohort geometry differs from the protocol")
    audit = validation.get("hypothesis_queue_audit")
    if not isinstance(audit, dict):
        raise TargetedSelectionError("validation lacks the hypothesis-queue audit")
    if int(audit.get("rows", -1)) != queue_rows:
        raise TargetedSelectionError("validated hypothesis-queue row count has changed")
    for field in (
        "non_primary_rows",
        "unsupported_rows",
        "promoted_with_counterpart_below_support_gate",
        "promoted_signed_rows_with_sign_disagreement",
        "promoted_with_genealogy_gate_failure",
    ):
        if int(audit.get(field, -1)) != 0:
            raise TargetedSelectionError(f"hypothesis-queue audit failed: {field}")
    lag_audit = validation.get("candidate_lag_profile_audit")
    if not isinstance(lag_audit, dict):
        raise TargetedSelectionError("validation lacks the candidate-lag-profile audit")
    if (
        int(lag_audit.get("queue_rows", -1)) != queue_rows
        or int(lag_audit.get("expected_profile_rows", -1))
        != int(lag_audit.get("profile_rows", -2))
        or lag_audit.get("used_for_promotion") is not False
    ):
        raise TargetedSelectionError("candidate-lag-profile audit is inconsistent")
    return manifest, protocol, validation, models


def _load_and_validate_queue(
    path: Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[bool], list[float], int]:
    try:
        # Keep every original field as text so list-valued lag profiles and
        # intentionally blank inferential cells are not normalized on rewrite.
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as error:
        raise TargetedSelectionError(f"could not read hypothesis_queue.csv: {error}") from error
    if frame.empty:
        raise TargetedSelectionError("canonical hypothesis queue is empty")
    if frame.columns.duplicated().any():
        raise TargetedSelectionError("hypothesis queue contains duplicate columns")
    _reject_external_fields(frame.columns, label="hypothesis queue")
    if tuple(frame.columns) != CANONICAL_QUEUE_COLUMNS:
        missing_canonical = sorted(set(CANONICAL_QUEUE_COLUMNS).difference(frame.columns))
        unexpected = sorted(set(frame.columns).difference(CANONICAL_QUEUE_COLUMNS))
        raise TargetedSelectionError(
            "hypothesis queue schema is not canonical "
            f"(missing={missing_canonical}, unexpected={unexpected})"
        )
    missing = sorted(set(REQUIRED_QUEUE_COLUMNS).difference(frame.columns))
    if missing:
        raise TargetedSelectionError(f"hypothesis queue is missing columns: {missing}")
    reserved = sorted(
        set(ADDED_COLUMNS + SELECTION_METADATA_COLUMNS).intersection(frame.columns)
    )
    if reserved:
        raise TargetedSelectionError(
            f"hypothesis queue already contains stale selection fields: {reserved}"
        )
    queue_ranks = [
        _integer(value, label="queue_rank", minimum=1) for value in frame["queue_rank"]
    ]
    if len(set(queue_ranks)) != len(queue_ranks) or sorted(queue_ranks) != list(
        range(1, len(frame) + 1)
    ):
        raise TargetedSelectionError("queue_rank must be unique and contiguous from one")
    promotions = [
        _strict_boolean(value, label="promotion_eligible")
        for value in frame["promotion_eligible"]
    ]
    scores = [
        _finite_float(value, label="evidence_score") for value in frame["evidence_score"]
    ]
    if any(value < 0 for value in scores):
        raise TargetedSelectionError("evidence_score must be nonnegative")
    protocol_contexts = None
    fps = None
    protocol_lags: set[int] | None = None
    protocol_horizons: set[int] | None = None
    if protocol is not None:
        protocol_contexts = {
            str(item["context"])
            for item in protocol.get("contexts", [])
            if isinstance(item, dict) and "context" in item
        }
        fps = _finite_float(protocol.get("fps"), label="protocol fps")
        if fps <= 0:
            raise TargetedSelectionError("protocol fps must be positive")
        protocol_lags = {
            _integer(value, label="protocol source lag", minimum=1)
            for value in protocol.get("source_lag_frames", [])
        }
        protocol_horizons = {
            _integer(value, label="protocol horizon", minimum=1)
            for value in protocol.get("horizon_frames", [])
        }
        if not protocol_lags or not protocol_horizons:
            raise TargetedSelectionError("protocol lag/horizon grids are empty")
    seen_keys: set[tuple[object, ...]] = set()
    diagonal = 0
    for position, row in frame.iterrows():
        source = str(row["source_neuron"]).strip()
        target = str(row["target_neuron"]).strip()
        if not source or not target:
            raise TargetedSelectionError("source and target neuron names must be nonempty")
        source_index = _integer(row["source_index"], label="source_index", minimum=0)
        target_index = _integer(row["target_index"], label="target_index", minimum=0)
        if source_index >= 54 or target_index >= 54:
            raise TargetedSelectionError("queue neuron index lies outside the 54-neuron axis")
        if (source == target) != (source_index == target_index):
            raise TargetedSelectionError("neuron names and indices disagree about a diagonal")
        if source == target:
            diagonal += 1
        lag = _integer(
            row["source_lag_frames"], label="source_lag_frames", minimum=1
        )
        horizon = _integer(row["horizon_frames"], label="horizon_frames", minimum=1)
        channel = str(row["channel"]).strip()
        if channel not in SUPPORTED_CHANNELS:
            raise TargetedSelectionError(
                f"queue channel {channel!r} is incompatible with targeted confirmation"
            )
        context = str(row["context"]).strip().lower()
        targeted_phases_for_context(context)
        if protocol_contexts is not None and context not in protocol_contexts:
            raise TargetedSelectionError(f"queue context {context!r} is absent from protocol")
        if protocol_lags is not None and lag not in protocol_lags:
            raise TargetedSelectionError("queue source lag is absent from protocol")
        if protocol_horizons is not None and horizon not in protocol_horizons:
            raise TargetedSelectionError("queue horizon is absent from protocol")
        if fps is not None:
            timings = (
                ("source_to_cut_seconds", lag / fps),
                ("forecast_horizon_seconds", horizon / fps),
                ("source_to_readout_seconds", (lag + horizon) / fps),
            )
            for field, expected in timings:
                actual = _finite_float(row[field], label=field)
                if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                    raise TargetedSelectionError(f"queue timing identity failed: {field}")
        if str(row["method"]).strip() != "progressive_bridge_smc":
            raise TargetedSelectionError(
                "canonical targeted queue must contain progressive_bridge_smc rows only"
            )
        if str(row["counterpart_method"]).strip() != "direct_importance":
            raise TargetedSelectionError("queue counterpart method is not direct importance")
        if _integer(row["n_worms"], label="queue n_worms", minimum=1) != 17:
            raise TargetedSelectionError("queue row does not represent all 17 worms")
        primary_valid = _finite_float(row["valid_fraction"], label="valid_fraction")
        if not 0.5 <= primary_valid <= 1:
            raise TargetedSelectionError("queue contains a source below the support gate")
        genealogy_applicable = _strict_boolean(
            row["genealogy_gate_applicable"], label="genealogy_gate_applicable"
        )
        if not genealogy_applicable:
            raise TargetedSelectionError(
                "progressive targeted queue row does not declare genealogy support"
            )
        genealogy_strong_threshold = _finite_float(
            row["genealogy_min_distinct_ancestor_fraction_strong"],
            label="genealogy strong ancestor threshold",
        )
        genealogy_sensitivity_threshold = _finite_float(
            row["genealogy_min_distinct_ancestor_fraction_sensitivity"],
            label="genealogy sensitivity ancestor threshold",
        )
        if not math.isclose(
            genealogy_strong_threshold, 0.10, rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(
            genealogy_sensitivity_threshold, 0.20, rel_tol=0.0, abs_tol=1e-12
        ):
            raise TargetedSelectionError("queue genealogy thresholds are not canonical")
        genealogy_valid_0_10 = _finite_float(
            row["genealogy_valid_fraction_0_10"],
            label="genealogy_valid_fraction_0_10",
        )
        genealogy_valid_0_20 = _finite_float(
            row["genealogy_valid_fraction_0_20"],
            label="genealogy_valid_fraction_0_20",
        )
        if not 0 <= genealogy_valid_0_20 <= genealogy_valid_0_10 <= primary_valid <= 1:
            raise TargetedSelectionError(
                "genealogy-adjusted support must be nested within raw validity"
            )
        genealogy_strong_pass = _strict_boolean(
            row["genealogy_strong_gate_pass"], label="genealogy_strong_gate_pass"
        )
        genealogy_sensitivity_pass = _strict_boolean(
            row["genealogy_sensitivity_gate_pass"],
            label="genealogy_sensitivity_gate_pass",
        )
        if genealogy_strong_pass != (genealogy_valid_0_10 >= 0.80):
            raise TargetedSelectionError("queue f=0.10 genealogy gate identity failed")
        if genealogy_sensitivity_pass != (genealogy_valid_0_20 >= 0.80):
            raise TargetedSelectionError("queue f=0.20 genealogy sensitivity identity failed")
        if promotions[position] and not genealogy_strong_pass:
            raise TargetedSelectionError(
                "promotion-eligible queue row fails the f=0.10 genealogy gate"
            )
        counterpart_valid = _finite_float(
            row["counterpart_valid_fraction"], label="counterpart_valid_fraction"
        )
        if not 0 <= counterpart_valid <= 1:
            raise TargetedSelectionError("counterpart_valid_fraction lies outside [0,1]")
        key = (channel, context, source_index, target_index, lag, horizon)
        if key in seen_keys:
            raise TargetedSelectionError("hypothesis queue contains a duplicate candidate cell")
        seen_keys.add(key)
    return frame, promotions, scores, diagonal


def select_rows(
    frame: pd.DataFrame,
    promotions: Sequence[bool],
    scores: Sequence[float],
    *,
    max_rows: int = MAX_SELECTION_ROWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the deterministic channel/source-diverse selection policy."""

    if not 1 <= max_rows <= MAX_SELECTION_ROWS:
        raise ValueError(f"max_rows must lie in 1..{MAX_SELECTION_ROWS}")
    if len(frame) != len(promotions) or len(frame) != len(scores):
        raise ValueError("selection priority arrays differ from the queue length")
    eligible = [
        position
        for position, row in frame.iterrows()
        if str(row["source_neuron"]).strip() != str(row["target_neuron"]).strip()
    ]
    if not eligible:
        raise TargetedSelectionError("no off-diagonal candidates remain")
    ordered = sorted(
        eligible,
        key=lambda position: (
            -int(promotions[position]),
            -float(scores[position]),
            int(frame.at[position, "queue_rank"]),
            str(frame.at[position, "channel"]),
            str(frame.at[position, "context"]),
            str(frame.at[position, "source_neuron"]),
            str(frame.at[position, "target_neuron"]),
            int(frame.at[position, "source_lag_frames"]),
            int(frame.at[position, "horizon_frames"]),
        ),
    )
    chosen: list[int] = []
    rules: list[str] = []
    chosen_set: set[int] = set()
    used_sources: set[str] = set()
    covered_channels: set[str] = set()

    def take(position: int, rule: str) -> None:
        chosen.append(position)
        rules.append(rule)
        chosen_set.add(position)
        used_sources.add(str(frame.at[position, "source_neuron"]).strip())
        covered_channels.add(str(frame.at[position, "channel"]).strip())

    # Pass one: channel coverage, requiring a new source for every selected row.
    for position in ordered:
        if len(chosen) >= max_rows:
            break
        channel = str(frame.at[position, "channel"]).strip()
        source = str(frame.at[position, "source_neuron"]).strip()
        if channel not in covered_channels and source not in used_sources:
            take(position, "channel_coverage_unique_source")

    # Pass two: fill the compute budget without reusing source neurons where possible.
    for position in ordered:
        if len(chosen) >= max_rows:
            break
        source = str(frame.at[position, "source_neuron"]).strip()
        if position not in chosen_set and source not in used_sources:
            take(position, "ranked_fill_unique_source")

    # Pass three: only reuse sources if uniqueness cannot fill the bounded budget.
    for position in ordered:
        if len(chosen) >= max_rows:
            break
        if position not in chosen_set:
            take(position, "ranked_fill_reused_source")

    result = frame.iloc[chosen].copy().reset_index(drop=True)
    return result, rules


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _rationale(
    selected: pd.DataFrame,
    *,
    source_rows: int,
    diagonal_rows: int,
    frozen_queue_sha256: str,
) -> str:
    lines = [
        "# Frozen N128 targeted-confirmation selection",
        "",
        (
            f"Selected **{len(selected)}** off-diagonal hypotheses from {source_rows} "
            "canonical internal queue rows for progressive-bridge confirmation with "
            f"**N={PARTICLES} particles**."
        ),
        "",
        "## Selection rule",
        "",
        (
            "Rows are ordered by `promotion_eligible` and then descending "
            "`evidence_score`. The first pass covers distinct effect channels with "
            "distinct source neurons; remaining slots keep sources unique where "
            "possible, then reuse a source only if necessary."
        ),
        "",
        f"Diagonal rows excluded: {diagonal_rows}.",
        "",
        "## Frozen hypotheses",
        "",
        "| Rank | Channel | Source → target | Context | Lag | Horizon | Promoted | Evidence | Rule |",
        "|---:|---|---|---|---:|---:|---|---:|---|",
    ]
    for _, row in selected.iterrows():
        lines.append(
            "| {rank} | {channel} | {source} → {target} | {context} | {lag} | "
            "{horizon} | {promoted} | {score:.6g} | {rule} |".format(
                rank=int(row["selection_rank"]),
                channel=_markdown_cell(row["channel"]),
                source=_markdown_cell(row["source_neuron"]),
                target=_markdown_cell(row["target_neuron"]),
                context=_markdown_cell(row["context"]),
                lag=int(row["source_lag_frames"]),
                horizon=int(row["horizon_frames"]),
                promoted=_strict_boolean(
                    row["promotion_eligible"], label="promotion_eligible"
                ),
                score=float(row["evidence_score"]),
                rule=_markdown_cell(row["selection_rule"]),
            )
        )
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            (
                "This freeze used only the canonical model-derived hypothesis queue. "
                "No anatomical, functional-connectome, neuromodulator, or historical "
                "SBTG result was loaded or used for selection. Confirmation remains "
                "model-relative and is not a causal or physical-delay estimate."
            ),
            "",
            f"Frozen canonical queue SHA-256: `{frozen_queue_sha256}`",
            "",
        ]
    )
    return "\n".join(lines)


def freeze_targeted_selection(
    atlas_dir: Path,
    output_dir: Path,
    *,
    max_rows: int = MAX_SELECTION_ROWS,
) -> dict[str, Any]:
    """Validate the canonical atlas and atomically freeze an N128 shortlist."""

    if not 1 <= int(max_rows) <= MAX_SELECTION_ROWS:
        raise ValueError(f"max_rows must lie in 1..{MAX_SELECTION_ROWS}")
    atlas_dir = Path(atlas_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not atlas_dir.is_dir():
        raise FileNotFoundError(atlas_dir)
    if _paths_overlap(atlas_dir, output_dir):
        raise ValueError("selection output must be disjoint from the canonical atlas")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing selection bundle: {output_dir}"
        )
    for name in REQUIRED_ATLAS_FILES + (CHECKSUM_FILENAME,):
        if not (atlas_dir / name).is_file():
            raise TargetedSelectionError(f"canonical atlas is missing {name}")

    inventory = _parse_checksum_inventory(atlas_dir)
    frozen_queue_hash = inventory["hypothesis_queue.csv"]
    # Parse once to establish the exact validated row count, then validate the
    # metadata sidecars against that frozen queue.
    frame, promotions, scores, diagonal_rows = _load_and_validate_queue(
        atlas_dir / "hypothesis_queue.csv"
    )
    manifest, protocol, validation, _models = _validate_atlas_metadata(
        atlas_dir, queue_rows=len(frame)
    )
    # Repeat queue validation against protocol context declarations.
    frame, promotions, scores, diagonal_rows = _load_and_validate_queue(
        atlas_dir / "hypothesis_queue.csv", protocol=protocol
    )
    selected, rules = select_rows(
        frame, promotions, scores, max_rows=int(max_rows)
    )
    original_columns = list(frame.columns)
    selected_metadata = selected.copy()
    selected_metadata["selection_rank"] = list(range(1, len(selected) + 1))
    selected_metadata["selection_rule"] = rules
    selected_metadata["frozen_hypothesis_queue_sha256"] = frozen_queue_hash
    staged_queue = selected.copy()
    staged_queue["run_confirmation"] = True
    if list(staged_queue.columns) != original_columns + list(ADDED_COLUMNS):
        raise AssertionError("selection output did not preserve the source queue columns")
    selection_fingerprint = _canonical_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "particles": PARTICLES,
            "max_rows": int(max_rows),
            "frozen_hypothesis_queue_sha256": frozen_queue_hash,
            "selected_queue_ranks": [int(value) for value in selected["queue_rank"]],
            "selection_rules": list(rules),
        }
    )
    bundle_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_atlas": str(atlas_dir),
        "source_manifest_sha256": inventory["manifest.json"],
        "source_protocol_sha256": inventory["protocol.json"],
        "source_validation_sha256": inventory["validation.json"],
        "source_atlas_matrices_sha256": inventory["atlas_matrices.npz"],
        "frozen_hypothesis_queue_sha256": frozen_queue_hash,
        "selection_fingerprint": selection_fingerprint,
        "particles": PARTICLES,
        "confirmation_method": CONFIRMATION_METHOD,
        "maximum_rows": int(max_rows),
        "selected_rows": len(selected),
        "source_queue_rows": len(frame),
        "excluded_diagonal_rows": diagonal_rows,
        "selected_queue_ranks": [int(value) for value in selected["queue_rank"]],
        "selected_rows_metadata": [
            {
                "selection_rank": int(row["selection_rank"]),
                "selection_rule": str(row["selection_rule"]),
                "queue_rank": int(row["queue_rank"]),
                "channel": str(row["channel"]),
                "context": str(row["context"]),
                "source_neuron": str(row["source_neuron"]),
                "target_neuron": str(row["target_neuron"]),
                "frozen_hypothesis_queue_sha256": str(
                    row["frozen_hypothesis_queue_sha256"]
                ),
            }
            for _, row in selected_metadata.iterrows()
        ],
        "selected_channels": list(dict.fromkeys(selected["channel"].astype(str))),
        "selected_source_neurons": list(
            dict.fromkeys(selected["source_neuron"].astype(str))
        ),
        "unique_selected_sources": int(selected["source_neuron"].nunique()),
        "selection_priority": [
            "promotion_eligible descending",
            "evidence_score descending",
            "queue_rank ascending deterministic tie break",
        ],
        "selection_passes": [
            "one row per effect channel with a unique source neuron",
            "ranked fill with unique source neurons",
            "ranked fill allowing source reuse only if necessary",
        ],
        "targeted_context_mapping": {
            "phase": "the matching single phase",
            "state_average": list(PHASES),
            "*_onset": ["onset"],
            "*_onset_minus_baseline": ["baseline", "onset"],
        },
        "source_orientation": protocol["orientation"],
        "ranking_inputs_used": ["canonical internal hypothesis_queue.csv only"],
        "external_reference_inputs_used": [],
        "external_fields_detected": [],
        "claim_boundary": CLAIM_BOUNDARY,
        "artifacts": {
            "selection": SELECTION_FILENAME,
            "screen_dense_matrices": "atlas_matrices.npz",
            "rationale": RATIONALE_FILENAME,
            "manifest": MANIFEST_FILENAME,
            "checksums": CHECKSUM_FILENAME,
        },
        "source_validation_status": validation["status"],
        "source_manifest_status": manifest["status"],
        "screen_dir_compatible": True,
        "screen_dir_usage": (
            "pass this bundle as targeted_confirmation_analysis --screen-dir; "
            "its hypothesis_queue.csv is the exact queue used by the targeted run"
        ),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        staged_queue.to_csv(temporary / SELECTION_FILENAME, index=False)
        # A real copy is deliberate: neither the immutable canonical matrix nor
        # the staged bundle can be changed through a shared inode or symlink.
        shutil.copy2(
            atlas_dir / "atlas_matrices.npz", temporary / "atlas_matrices.npz"
        )
        if sha256(temporary / "atlas_matrices.npz") != inventory["atlas_matrices.npz"]:
            raise TargetedSelectionError("staged atlas_matrices.npz copy failed checksum")
        (temporary / RATIONALE_FILENAME).write_text(
            _rationale(
                selected_metadata,
                source_rows=len(frame),
                diagonal_rows=diagonal_rows,
                frozen_queue_sha256=frozen_queue_hash,
            ),
            encoding="utf-8",
        )
        bundle_manifest["selection_sha256"] = sha256(
            temporary / SELECTION_FILENAME
        )
        _write_json(temporary / MANIFEST_FILENAME, bundle_manifest)
        output_files = (
            SELECTION_FILENAME,
            "atlas_matrices.npz",
            RATIONALE_FILENAME,
            MANIFEST_FILENAME,
        )
        (temporary / CHECKSUM_FILENAME).write_text(
            "".join(
                f"{sha256(temporary / name)}  {name}\n" for name in output_files
            ),
            encoding="utf-8",
        )
        # Close the check-use gap: a source mutation during selection invalidates
        # the freeze, even if the earlier validation succeeded.
        for name, expected in inventory.items():
            if sha256(atlas_dir / name) != expected:
                raise TargetedSelectionError(
                    f"canonical source changed during selection: {name}"
                )
        if output_dir.exists():
            raise FileExistsError(output_dir)
        os.rename(temporary, output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return bundle_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=MAX_SELECTION_ROWS,
        help=f"bounded confirmation shortlist size (1..{MAX_SELECTION_ROWS})",
    )
    args = parser.parse_args()
    if not 1 <= args.max_rows <= MAX_SELECTION_ROWS:
        parser.error(f"--max-rows must lie in 1..{MAX_SELECTION_ROWS}")
    return args


def main() -> None:
    args = _parse_args()
    manifest = freeze_targeted_selection(
        args.atlas_dir,
        args.output_dir,
        max_rows=args.max_rows,
    )
    print(
        f"FROZEN_TARGETED_SELECTION rows={manifest['selected_rows']} "
        f"particles={manifest['particles']} output={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
