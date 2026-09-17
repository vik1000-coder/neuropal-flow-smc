"""Freeze an internal-only distributional/stimulus confirmation shortlist.

This selector is deliberately separate from the primary hypothesis-queue
selector.  It reads the reviewed canonical prediction table, dense matrices,
and source-support table directly so that variance and distributional channels
cannot be crowded out by a promotion-first mean-effect queue.

The six frozen design strata balance three effect channels and three context
classes.  Selection never reads connectomes, neuromodulator references, SBTG,
or any other external benchmark.  It maximizes distinct source neurons first,
then summed internal model evidence, and finally uses stable lexical ties.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compatibility_neural_benchmark.prediction_atlas_select_targeted import (
    ADDED_COLUMNS,
    CANONICAL_QUEUE_COLUMNS,
    CHECKSUM_FILENAME,
    CONFIRMATION_METHOD,
    ORIENTATION,
    PARTICLES,
    PHASES,
    SELECTION_FILENAME,
    TargetedSelectionError,
    _canonical_fingerprint,
    _parse_checksum_inventory,
    _paths_overlap,
    _read_json,
    _reject_external_fields,
    _write_json,
    sha256,
)


SCHEMA_VERSION = "prediction_atlas_distributional_stimulus_selection_v1"
RATIONALE_FILENAME = "selection_rationale.md"
MANIFEST_FILENAME = "manifest.json"
MAX_SELECTION_ROWS = 6
PRIMARY_METHOD = "progressive_bridge_smc"
COUNTERPART_METHOD = "direct_importance"
CHANNELS = ("endpoint_sd", "endpoint_log_sd", "endpoint_wasserstein1")
PHASE_CONTEXTS = frozenset(PHASES)
CHEMICAL_CONTRAST_CONTEXTS = frozenset(
    {
        "butanone_onset_minus_baseline",
        "pentanedione_onset_minus_baseline",
        "nacl_onset_minus_baseline",
    }
)
GLOBAL_CONTRAST_CONTEXT = "onset_minus_baseline"
CANONICAL_PREDICTION_COLUMNS = tuple(
    column
    for column in CANONICAL_QUEUE_COLUMNS[
        : CANONICAL_QUEUE_COLUMNS.index("counterpart_method")
    ]
    if column != "queue_rank"
)
REQUIRED_SOURCE_FILES = (
    "manifest.json",
    "protocol.json",
    "validation.json",
    "models.json",
    "prediction_cells.parquet",
    "support_cells.parquet",
    "atlas_matrices.npz",
    "hypothesis_queue.csv",
    CHECKSUM_FILENAME,
)
REQUIRED_SUPPORT_COLUMNS = (
    "method",
    "context",
    "source_neuron",
    "source_index",
    "source_lag_frames",
    "valid_fraction",
    "genealogy_gate_applicable",
    "genealogy_valid_fraction_0_10",
    "genealogy_valid_fraction_0_20",
    "genealogy_strong_gate_pass",
    "genealogy_sensitivity_gate_pass",
)
ALLOWED_SUPPORT_COLUMNS = frozenset(
    {
        "method",
        "context",
        "chemical",
        "conditioning_status",
        "source_neuron",
        "source_index",
        "source_lag_frames",
        "source_to_cut_seconds",
        "valid_fraction",
        "n_worms_valid_ge_half",
        "mean_achieved_gap_magnitude",
        "n_worms",
        "support_qualified",
        "genealogy_gate_applicable",
        "genealogy_min_distinct_ancestor_fraction_strong",
        "genealogy_min_distinct_ancestor_fraction_sensitivity",
        "genealogy_valid_fraction_0_10",
        "genealogy_valid_fraction_0_20",
        "genealogy_strong_gate_pass",
        "genealogy_sensitivity_gate_pass",
        "n_particles",
        "declared_branch_factor",
        "declared_future_branch_factor",
        "branch_factor_semantics",
        "mean_archive_wall_seconds_method_lag",
        "n_archives_method_lag",
        "ess_gate",
        "max_weight_gate",
        "mean_ess_low",
        "mean_ess_high",
        "minimum_worm_ess_low",
        "minimum_worm_ess_high",
        "mean_ess_fraction_low",
        "mean_ess_fraction_high",
        "mean_max_weight_low",
        "mean_max_weight_high",
        "maximum_worm_max_weight_low",
        "maximum_worm_max_weight_high",
        "ess_low_gate_pass_fraction",
        "ess_high_gate_pass_fraction",
        "max_weight_low_gate_pass_fraction",
        "max_weight_high_gate_pass_fraction",
        "mean_candidate_ess_low",
        "mean_candidate_ess_high",
        "mean_candidate_max_weight_low",
        "mean_candidate_max_weight_high",
        "mean_min_step_ess_low",
        "mean_min_step_ess_high",
        "mean_distinct_ancestors_low",
        "mean_distinct_ancestors_high",
        "mean_distinct_ancestor_fraction_low",
        "mean_distinct_ancestor_fraction_high",
        "mean_min_distinct_ancestor_fraction",
        "minimum_worm_min_distinct_ancestor_fraction",
        "genealogy_ok_fraction_0_10",
        "genealogy_ok_fraction_0_20",
        "forced_tempering_rate_low",
        "forced_tempering_rate_high",
        "mean_tempering_resamples_low",
        "mean_tempering_resamples_high",
    }
)
JSON_TOP_LEVEL_ALLOWLIST = {
    "manifest.json": frozenset(
        {
            "artifacts",
            "chemical_leaderboard_sha256",
            "complete_raw_data_location",
            "created_utc",
            "dense_matrix_orientation",
            "fold_assignments_sha256",
            "input_run_dirs",
            "primary_method",
            "protocol",
            "source_run_manifest_sha256",
            "status",
        }
    ),
    "protocol.json": frozenset(
        {
            "channels",
            "chemical_context_warning",
            "cohort_mode",
            "compute_diagnostics",
            "confirmation_shortlist",
            "contexts",
            "effect_definition",
            "evidence_tiers",
            "fps",
            "history_frames",
            "horizon_frames",
            "hypothesis_queue_eligibility",
            "inference",
            "interpretation_limit",
            "lag_localization",
            "n_neurons",
            "n_worms",
            "normalization",
            "orientation",
            "orientation_operation",
            "protocol",
            "ranking",
            "sampler_reproducibility",
            "source_lag_frames",
            "source_window_frames",
            "stimulus_composition",
            "storage",
            "support_thresholds",
            "timing_definitions",
        }
    ),
    "validation.json": frozenset(
        {
            "archive_checks_completed",
            "archive_count",
            "base_seed",
            "candidate_lag_profile_audit",
            "common_noise_definition",
            "created_utc",
            "cross_sampler_diagnostics",
            "dashboard_matrix_slice",
            "episode_seed_definition",
            "folds",
            "horizons",
            "hypothesis_queue_audit",
            "input_manifests",
            "input_run_dirs",
            "methods",
            "n_neurons",
            "n_worms",
            "particle_support_context_diagnostics",
            "particle_support_diagnostics",
            "problems",
            "requested_device",
            "resolved_device",
            "runtime_compute_diagnostics",
            "seed_diagnostics",
            "seeds",
            "source_lags",
            "source_run_provenance",
            "status",
            "stimulus_schema_fingerprint",
            "stimulus_schema_version",
            "weakest_source_context_diagnostics",
        }
    ),
    "models.json": frozenset(
        {
            "base_seed",
            "checkpoints",
            "chemical_conditioning",
            "common_noise_definition",
            "episode_seed_definition",
            "frozen_predictive_scores",
            "generator_model_id",
            "generator_stimulus_encoding",
            "methods",
            "requested_device",
            "resolved_device",
            "source_run_provenance",
            "stimulus_encoding_selection",
        }
    ),
}
ADDITIONAL_EXTERNAL_FIELD_TOKEN = re.compile(
    r"(?:^|_)(?:anatomy|anatomical|synapse|receptor|ground_truth)(?:_|$)",
    re.IGNORECASE,
)
STRICT_BOOLEAN_COLUMNS = (
    "support_qualified",
    "genealogy_gate_applicable",
    "genealogy_strong_gate_pass",
    "genealogy_sensitivity_gate_pass",
)
CLAIM_BOUNDARY = (
    "internal model-relative distributional/stimulus shortlist for higher-particle "
    "confirmation; not an external-reference ranking, causal effect, anatomical "
    "connection, receptor action, or physical delay"
)


@dataclass(frozen=True)
class DesignStratum:
    name: str
    channel: str
    context_class: str


DESIGN_STRATA = (
    DesignStratum(
        "endpoint_sd__onset_minus_baseline",
        "endpoint_sd",
        "global_contrast",
    ),
    DesignStratum("endpoint_sd__phase_specific", "endpoint_sd", "phase"),
    DesignStratum(
        "endpoint_log_sd__onset_minus_baseline",
        "endpoint_log_sd",
        "global_contrast",
    ),
    DesignStratum(
        "endpoint_log_sd__chemical_onset_minus_baseline",
        "endpoint_log_sd",
        "chemical_contrast",
    ),
    DesignStratum(
        "endpoint_wasserstein1__chemical_onset_minus_baseline",
        "endpoint_wasserstein1",
        "chemical_contrast",
    ),
    DesignStratum(
        "endpoint_wasserstein1__phase_specific",
        "endpoint_wasserstein1",
        "phase",
    ),
)
STRATUM_BY_NAME = {value.name: value for value in DESIGN_STRATA}


class DistributionalSelectionError(TargetedSelectionError):
    """Raised when the exploratory shortlist cannot be frozen safely."""


def _context_class(context: str) -> str | None:
    if context == GLOBAL_CONTRAST_CONTEXT:
        return "global_contrast"
    if context in CHEMICAL_CONTRAST_CONTEXTS:
        return "chemical_contrast"
    if context in PHASE_CONTEXTS:
        return "phase"
    return None


def _design_stratum(channel: str, context: str) -> str | None:
    context_class = _context_class(context)
    if context_class is None:
        return None
    for value in DESIGN_STRATA:
        if value.channel == channel and value.context_class == context_class:
            return value.name
    return None


def _is_signed(channel: str, context: str) -> bool:
    return channel != "endpoint_wasserstein1" or context.endswith(
        "minus_baseline"
    )


def _finite(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise DistributionalSelectionError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise DistributionalSelectionError(f"{label} must be finite")
    return number


def _json_field_paths(value: object, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            paths.append(path)
            paths.extend(_json_field_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_json_field_paths(child, path))
    return paths


def _validate_json_provenance(filename: str, value: Mapping[str, Any]) -> None:
    allowed = JSON_TOP_LEVEL_ALLOWLIST[filename]
    unexpected = sorted(set(value).difference(allowed))
    paths = _json_field_paths(value)
    try:
        _reject_external_fields(paths, label=filename)
    except TargetedSelectionError as error:
        raise DistributionalSelectionError(str(error)) from error
    additional = sorted(
        path
        for path in paths
        if ADDITIONAL_EXTERNAL_FIELD_TOKEN.search(
            re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")
        )
    )
    if additional:
        raise DistributionalSelectionError(
            f"{filename} contains prohibited external-reference fields: "
            + ", ".join(additional)
        )
    if unexpected:
        raise DistributionalSelectionError(
            f"{filename} contains unreviewed top-level fields: {unexpected}"
        )


def _require_boolean_columns(frame: pd.DataFrame, *, label: str) -> None:
    for column in STRICT_BOOLEAN_COLUMNS:
        if column not in frame.columns:
            continue
        if not pd.api.types.is_bool_dtype(frame[column].dtype) or frame[column].isna().any():
            raise DistributionalSelectionError(
                f"{label}.{column} must use a non-null boolean dtype; "
                "string and numeric boolean encodings are rejected"
            )


def _source_bundle(
    atlas_dir: Path,
) -> tuple[
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    for name in REQUIRED_SOURCE_FILES:
        if not (atlas_dir / name).is_file():
            raise DistributionalSelectionError(f"canonical atlas is missing {name}")
    try:
        inventory = _parse_checksum_inventory(atlas_dir)
    except TargetedSelectionError as error:
        raise DistributionalSelectionError(str(error)) from error
    for name in ("prediction_cells.parquet", "support_cells.parquet"):
        if name not in inventory:
            raise DistributionalSelectionError(
                f"canonical checksum inventory does not cover {name}"
            )
    manifest = _read_json(atlas_dir / "manifest.json")
    protocol = _read_json(atlas_dir / "protocol.json")
    validation = _read_json(atlas_dir / "validation.json")
    models = _read_json(atlas_dir / "models.json")
    for filename, value in (
        ("manifest.json", manifest),
        ("protocol.json", protocol),
        ("validation.json", validation),
        ("models.json", models),
    ):
        _validate_json_provenance(filename, value)
    if manifest.get("status") != "complete":
        raise DistributionalSelectionError("canonical atlas manifest is not complete")
    if manifest.get("primary_method") != PRIMARY_METHOD:
        raise DistributionalSelectionError("canonical primary method changed")
    if manifest.get("dense_matrix_orientation") != ORIENTATION:
        raise DistributionalSelectionError("canonical matrix orientation changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or any(
        artifacts.get(name) != filename
        for name, filename in {
            "prediction_cells": "prediction_cells.parquet",
            "support_cells": "support_cells.parquet",
            "atlas_matrices": "atlas_matrices.npz",
        }.items()
    ):
        raise DistributionalSelectionError("canonical artifact mapping is incomplete")
    if protocol.get("ranking") != "no connectome or external reference data used":
        raise DistributionalSelectionError("canonical internal-ranking firewall is absent")
    if (
        protocol.get("orientation") != ORIENTATION
        or protocol.get("n_worms") != 17
        or protocol.get("n_neurons") != 54
        or protocol.get("source_lag_frames") != [1, 4, 8, 16]
        or protocol.get("horizon_frames") != [1, 2, 4, 8, 16, 32]
    ):
        raise DistributionalSelectionError("canonical cohort or timing grid changed")
    if models.get("generator_stimulus_encoding") != "binary_any_stimulus":
        raise DistributionalSelectionError("generator encoding is not binary-any-stimulus")
    if models.get("chemical_conditioning") is not False:
        raise DistributionalSelectionError("generator incorrectly declares chemical conditioning")
    checks = validation.get("archive_checks_completed")
    if (
        validation.get("status") != "passed"
        or validation.get("problems") not in ([], None)
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise DistributionalSelectionError("canonical validation did not pass cleanly")
    try:
        prediction = pd.read_parquet(atlas_dir / "prediction_cells.parquet")
        support = pd.read_parquet(atlas_dir / "support_cells.parquet")
    except Exception as error:
        raise DistributionalSelectionError(
            f"could not read canonical parquet inputs: {error}"
        ) from error
    if tuple(prediction.columns) != CANONICAL_PREDICTION_COLUMNS:
        missing = sorted(set(CANONICAL_PREDICTION_COLUMNS).difference(prediction.columns))
        extra = sorted(set(prediction.columns).difference(CANONICAL_PREDICTION_COLUMNS))
        raise DistributionalSelectionError(
            f"prediction-cell schema changed (missing={missing}, extra={extra})"
        )
    missing_support = sorted(set(REQUIRED_SUPPORT_COLUMNS).difference(support.columns))
    if missing_support:
        raise DistributionalSelectionError(
            f"support table is missing columns: {missing_support}"
        )
    try:
        _reject_external_fields(prediction.columns, label="prediction cells")
        _reject_external_fields(support.columns, label="support cells")
    except TargetedSelectionError as error:
        raise DistributionalSelectionError(str(error)) from error
    unexpected_support = sorted(set(support.columns).difference(ALLOWED_SUPPORT_COLUMNS))
    if unexpected_support:
        raise DistributionalSelectionError(
            f"support table contains unreviewed columns: {unexpected_support}"
        )
    _require_boolean_columns(prediction, label="prediction cells")
    _require_boolean_columns(support, label="support cells")
    if prediction.empty or support.empty:
        raise DistributionalSelectionError("canonical source tables are empty")
    prediction_key = (
        "method",
        "channel",
        "context",
        "source_lag_frames",
        "horizon_frames",
        "source_index",
        "target_index",
    )
    support_key = ("method", "context", "source_lag_frames", "source_index")
    if prediction.duplicated(list(prediction_key)).any():
        raise DistributionalSelectionError("prediction cells contain duplicate keys")
    if support.duplicated(list(support_key)).any():
        raise DistributionalSelectionError("support cells contain duplicate keys")
    return inventory, manifest, protocol, prediction, support


def _atlas_metadata(atlas_dir: Path) -> dict[str, Any]:
    path = atlas_dir / "atlas_matrices.npz"
    with np.load(path, allow_pickle=False) as atlas:
        required = {
            "neurons",
            "methods",
            "channels",
            "contexts",
            "source_lag_frames",
            "horizon_frames",
            "orientation",
            "primary_method",
        }
        missing = sorted(required.difference(atlas.files))
        if missing:
            raise DistributionalSelectionError(
                f"dense atlas lacks metadata: {missing}"
            )
        metadata = {
            "neurons": tuple(atlas["neurons"].astype(str)),
            "methods": tuple(atlas["methods"].astype(str)),
            "channels": tuple(atlas["channels"].astype(str)),
            "contexts": tuple(atlas["contexts"].astype(str)),
            "lags": tuple(int(value) for value in atlas["source_lag_frames"]),
            "horizons": tuple(int(value) for value in atlas["horizon_frames"]),
            "orientation": str(np.asarray(atlas["orientation"]).item()),
            "primary_method": str(np.asarray(atlas["primary_method"]).item()),
        }
    if (
        len(metadata["neurons"]) != 54
        or len(set(metadata["neurons"])) != 54
        or (
            metadata["methods"] != (PRIMARY_METHOD, COUNTERPART_METHOD)
            and metadata["methods"] != (COUNTERPART_METHOD, PRIMARY_METHOD)
        )
        or metadata["orientation"] != ORIENTATION
        or metadata["primary_method"] != PRIMARY_METHOD
        or metadata["lags"] != (1, 4, 8, 16)
        or metadata["horizons"] != (1, 2, 4, 8, 16, 32)
        or not set(CHANNELS).issubset(metadata["channels"])
    ):
        raise DistributionalSelectionError("dense atlas metadata is not canonical")
    return metadata


def _row_tie_key(row: pd.Series) -> tuple[object, ...]:
    return (
        str(row["design_stratum"]),
        str(row["context"]),
        str(row["source_neuron"]),
        str(row["target_neuron"]),
        int(row["source_lag_frames"]),
        int(row["horizon_frames"]),
        int(row["source_index"]),
        int(row["target_index"]),
    )


def build_eligible_candidates(
    atlas_dir: Path,
    prediction: pd.DataFrame,
    support: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return fully annotated candidates that pass the frozen design gates."""

    allowed_contexts = (
        {GLOBAL_CONTRAST_CONTEXT}
        | set(CHEMICAL_CONTRAST_CONTEXTS)
        | set(PHASE_CONTEXTS)
    )
    frame = prediction.loc[
        prediction["method"].eq(PRIMARY_METHOD)
        & prediction["channel"].isin(CHANNELS)
        & prediction["context"].isin(allowed_contexts)
        & prediction["source_index"].ne(prediction["target_index"])
    ].copy()
    frame["design_stratum"] = [
        _design_stratum(str(channel), str(context))
        for channel, context in zip(frame["channel"], frame["context"])
    ]
    frame = frame.loc[frame["design_stratum"].notna()].copy()
    support_columns = list(REQUIRED_SUPPORT_COLUMNS)
    support_primary = support.loc[
        support["method"].eq(PRIMARY_METHOD), support_columns
    ].rename(
        columns={
            "valid_fraction": "support_valid_fraction",
            "genealogy_gate_applicable": "support_genealogy_gate_applicable",
            "genealogy_valid_fraction_0_10": "support_genealogy_valid_fraction_0_10",
            "genealogy_valid_fraction_0_20": "support_genealogy_valid_fraction_0_20",
            "genealogy_strong_gate_pass": "support_genealogy_strong_gate_pass",
            "genealogy_sensitivity_gate_pass": "support_genealogy_sensitivity_gate_pass",
        }
    )
    join_key = ["method", "context", "source_neuron", "source_index", "source_lag_frames"]
    frame = frame.merge(
        support_primary,
        on=join_key,
        how="left",
        validate="many_to_one",
    )
    if frame["support_valid_fraction"].isna().any():
        raise DistributionalSelectionError("candidate support join dropped one or more rows")
    if not np.allclose(
        frame["valid_fraction"], frame["support_valid_fraction"], rtol=0, atol=1e-7
    ):
        raise DistributionalSelectionError("prediction and support validity disagree")
    if not (
        frame["genealogy_gate_applicable"].to_numpy(dtype=bool)
        == frame["support_genealogy_gate_applicable"].to_numpy(dtype=bool)
    ).all():
        raise DistributionalSelectionError(
            "prediction and support genealogy applicability disagree"
        )
    for prediction_field, support_field in (
        (
            "genealogy_valid_fraction_0_10",
            "support_genealogy_valid_fraction_0_10",
        ),
        (
            "genealogy_valid_fraction_0_20",
            "support_genealogy_valid_fraction_0_20",
        ),
    ):
        if not np.allclose(
            frame[prediction_field], frame[support_field], rtol=0, atol=1e-7
        ):
            raise DistributionalSelectionError(
                "prediction and support genealogy fractions disagree"
            )
    if not (
        frame["genealogy_strong_gate_pass"].to_numpy(dtype=bool)
        == frame["support_genealogy_strong_gate_pass"].to_numpy(dtype=bool)
    ).all():
        raise DistributionalSelectionError("prediction and support genealogy disagree")
    if not (
        frame["genealogy_sensitivity_gate_pass"].to_numpy(dtype=bool)
        == frame["support_genealogy_sensitivity_gate_pass"].to_numpy(dtype=bool)
    ).all():
        raise DistributionalSelectionError(
            "prediction and support genealogy sensitivity gates disagree"
        )
    frame = frame.loc[
        frame["valid_fraction"].ge(0.5)
        & frame["genealogy_gate_applicable"]
        & frame["genealogy_strong_gate_pass"]
    ].copy()
    lags = tuple(metadata["lags"])
    horizons = tuple(metadata["horizons"])
    lag_pos = {value: index for index, value in enumerate(lags)}
    horizon_pos = {value: index for index, value in enumerate(horizons)}
    off_diagonal = ~np.eye(len(metadata["neurons"]), dtype=bool)
    annotated: list[pd.DataFrame] = []
    with np.load(atlas_dir / "atlas_matrices.npz", allow_pickle=False) as atlas:
        for (channel, context), group in frame.groupby(
            ["channel", "context"], sort=True
        ):
            primary_key = f"mean_normalized__{PRIMARY_METHOD}__{channel}__{context}"
            counterpart_key = (
                f"mean_normalized__{COUNTERPART_METHOD}__{channel}__{context}"
            )
            counterpart_valid_key = (
                f"valid_fraction__{COUNTERPART_METHOD}__{context}"
            )
            if any(
                key not in atlas.files
                for key in (primary_key, counterpart_key, counterpart_valid_key)
            ):
                raise DistributionalSelectionError(
                    f"dense atlas lacks an internal comparison for {channel}/{context}"
                )
            primary = np.asarray(atlas[primary_key])
            counterpart = np.asarray(atlas[counterpart_key])
            counterpart_valid = np.asarray(atlas[counterpart_valid_key])
            expected_shape = (
                len(lags),
                len(horizons),
                len(metadata["neurons"]),
                len(metadata["neurons"]),
            )
            if primary.shape != expected_shape or counterpart.shape != expected_shape:
                raise DistributionalSelectionError("dense candidate matrix has wrong shape")
            current = group.copy()
            counterpart_values: list[float] = []
            counterpart_support: list[float] = []
            slice_correlations: list[float] = []
            for row in current.itertuples(index=False):
                li = lag_pos[int(row.source_lag_frames)]
                hi = horizon_pos[int(row.horizon_frames)]
                source = int(row.source_index)
                target = int(row.target_index)
                saved = float(primary[li, hi, target, source])
                if not math.isclose(
                    saved,
                    float(row.mean_normalized),
                    rel_tol=1e-6,
                    abs_tol=2e-7,
                ):
                    raise DistributionalSelectionError(
                        "prediction row disagrees with target-row/source-column dense matrix"
                    )
                counterpart_values.append(float(counterpart[li, hi, target, source]))
                counterpart_support.append(float(counterpart_valid[li, source]))
                left = primary[li, hi][off_diagonal]
                right = counterpart[li, hi][off_diagonal]
                rho = (
                    float(spearmanr(left, right).statistic)
                    if np.std(left) > 0 and np.std(right) > 0
                    else float("nan")
                )
                slice_correlations.append(rho)
            current["counterpart_method"] = COUNTERPART_METHOD
            current["counterpart_mean_normalized"] = counterpart_values
            current["counterpart_valid_fraction"] = counterpart_support
            current["cross_sampler_spearman"] = slice_correlations
            annotated.append(current)
    if not annotated:
        raise DistributionalSelectionError("no candidates survived raw support gates")
    frame = pd.concat(annotated, ignore_index=True)
    frame["is_signed"] = [
        _is_signed(str(channel), str(context))
        for channel, context in zip(frame["channel"], frame["context"])
    ]
    primary_value = frame["mean_normalized"].astype(float)
    counterpart_value = frame["counterpart_mean_normalized"].astype(float)
    same_sign = (
        primary_value.abs().gt(1e-12)
        & counterpart_value.abs().gt(1e-12)
        & np.sign(primary_value).eq(np.sign(counterpart_value))
    )
    frame["cross_sampler_sign_agreement"] = np.where(
        frame["is_signed"], same_sign.astype(float), np.nan
    )
    magnitude = 1.0 - (primary_value - counterpart_value).abs() / (
        primary_value.abs() + counterpart_value.abs() + 1e-12
    )
    frame["cross_sampler_magnitude_agreement"] = magnitude.clip(0.0, 1.0)
    frame["cross_sampler_slice_agreement"] = np.where(
        np.isfinite(frame["cross_sampler_spearman"]),
        ((frame["cross_sampler_spearman"] + 1.0) / 2.0).clip(0.0, 1.0),
        0.5,
    )
    frame["cross_sampler_support_agreement"] = np.minimum(
        frame["valid_fraction"].astype(float),
        frame["counterpart_valid_fraction"].astype(float),
    )
    sign_gate = np.where(
        frame["is_signed"], frame["cross_sampler_sign_agreement"], 1.0
    )
    frame["cross_sampler_factor"] = (
        sign_gate
        * frame["cross_sampler_magnitude_agreement"]
        * frame["cross_sampler_slice_agreement"]
        * frame["cross_sampler_support_agreement"]
    )
    frame["base_evidence_score"] = frame["evidence_score"].astype(float)
    frame["evidence_score"] = (
        frame["base_evidence_score"] * frame["cross_sampler_factor"]
    )
    signed_eligible = (
        frame["support_tier"].eq("supported_exploratory")
        & frame["counterpart_valid_fraction"].ge(0.5)
        & same_sign
    )
    unsigned_eligible = (
        frame["channel"].eq("endpoint_wasserstein1")
        & frame["context"].isin(PHASE_CONTEXTS)
        & frame["support_tier"].isin(["model_only", "supported_exploratory"])
    )
    frame["promotion_eligible"] = frame["is_signed"] & signed_eligible
    frame = frame.loc[
        np.where(frame["is_signed"], signed_eligible, unsigned_eligible)
    ].copy()
    if frame.empty:
        raise DistributionalSelectionError("no candidates passed signed/unsigned gates")
    if not np.isfinite(frame["evidence_score"]).all() or (
        frame["evidence_score"] < 0
    ).any():
        raise DistributionalSelectionError("internal evidence scores are invalid")
    counts = {
        name: int((frame["design_stratum"] == name).sum())
        for name in STRATUM_BY_NAME
    }
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise DistributionalSelectionError(
            f"frozen design has no eligible candidate for strata: {missing}"
        )
    return frame.reset_index(drop=True), counts


def _choice_key(
    choices: Sequence[int | None], candidates: pd.DataFrame
) -> tuple[tuple[object, ...], ...]:
    placeholder = ("~", "~", "~", "~", 10**9, 10**9, 10**9, 10**9)
    return tuple(
        placeholder if position is None else _row_tie_key(candidates.loc[position])
        for position in choices
    )


def _better_state(
    proposed: tuple[float, tuple[int | None, ...]],
    current: tuple[float, tuple[int | None, ...]] | None,
    candidates: pd.DataFrame,
) -> bool:
    if current is None:
        return True
    if proposed[0] > current[0]:
        return True
    if proposed[0] < current[0]:
        return False
    return _choice_key(proposed[1], candidates) < _choice_key(
        current[1], candidates
    )


def select_balanced_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select one row per stratum with maximum feasible source diversity."""

    candidates = candidates.copy().reset_index(drop=True)
    stratum_position = {
        value.name: index for index, value in enumerate(DESIGN_STRATA)
    }
    best_by_source_stratum: dict[tuple[str, int], int] = {}
    ordered = sorted(
        range(len(candidates)),
        key=lambda position: (
            -float(candidates.at[position, "evidence_score"]),
            _row_tie_key(candidates.loc[position]),
        ),
    )
    for position in ordered:
        source = str(candidates.at[position, "source_neuron"])
        stratum = stratum_position[str(candidates.at[position, "design_stratum"])]
        best_by_source_stratum.setdefault((source, stratum), position)

    empty_choices: tuple[int | None, ...] = (None,) * len(DESIGN_STRATA)
    states: dict[int, tuple[float, tuple[int | None, ...]]] = {
        0: (0.0, empty_choices)
    }
    for source in sorted(candidates["source_neuron"].astype(str).unique()):
        previous = dict(states)
        updated = dict(states)
        available = [
            (stratum, best_by_source_stratum[(source, stratum)])
            for stratum in range(len(DESIGN_STRATA))
            if (source, stratum) in best_by_source_stratum
        ]
        for mask, state in previous.items():
            for stratum, position in available:
                bit = 1 << stratum
                if mask & bit:
                    continue
                choices = list(state[1])
                choices[stratum] = position
                proposed = (
                    state[0] + float(candidates.at[position, "evidence_score"]),
                    tuple(choices),
                )
                new_mask = mask | bit
                if _better_state(proposed, updated.get(new_mask), candidates):
                    updated[new_mask] = proposed
        states = updated

    maximum_coverage = max(mask.bit_count() for mask in states)
    finalists = [
        (mask, state)
        for mask, state in states.items()
        if mask.bit_count() == maximum_coverage
    ]
    best_fill = {
        stratum: next(
            position
            for position in ordered
            if str(candidates.at[position, "design_stratum"])
            == DESIGN_STRATA[stratum].name
        )
        for stratum in range(len(DESIGN_STRATA))
    }
    best_completed: tuple[float, tuple[int | None, ...]] | None = None
    for _, state in finalists:
        choices = list(state[1])
        total = float(state[0])
        for stratum, position in enumerate(choices):
            if position is not None:
                continue
            fill = best_fill[stratum]
            choices[stratum] = fill
            total += float(candidates.at[fill, "evidence_score"])
        completed = (total, tuple(choices))
        if _better_state(completed, best_completed, candidates):
            best_completed = completed
    if best_completed is None:
        raise AssertionError("balanced selection did not produce a completed state")
    choices = list(best_completed[1])
    selected = candidates.loc[[int(value) for value in choices]].copy()
    if len(selected) != MAX_SELECTION_ROWS or selected["design_stratum"].nunique() != len(
        DESIGN_STRATA
    ):
        raise AssertionError("balanced selection did not cover every frozen stratum")
    selected["selection_source_reused"] = selected["source_neuron"].duplicated(
        keep=False
    )
    if selected["source_neuron"].nunique() != maximum_coverage:
        raise AssertionError("balanced selection lost maximum source diversity")
    selected = selected.sort_values(
        [
            "evidence_score",
            "design_stratum",
            "context",
            "source_neuron",
            "target_neuron",
            "source_lag_frames",
            "horizon_frames",
        ],
        ascending=[False, True, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return selected


def _queue_rows(
    atlas_dir: Path,
    selected: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    lags = tuple(metadata["lags"])
    horizons = tuple(metadata["horizons"])
    horizon_pos = {value: index for index, value in enumerate(horizons)}
    rows: list[dict[str, Any]] = []
    with np.load(atlas_dir / "atlas_matrices.npz", allow_pickle=False) as atlas:
        for rank, (_, source_row) in enumerate(selected.iterrows(), start=1):
            row = source_row.to_dict()
            channel = str(row["channel"])
            context = str(row["context"])
            horizon_index = horizon_pos[int(row["horizon_frames"])]
            target = int(row["target_index"])
            source = int(row["source_index"])
            primary = np.asarray(
                atlas[f"mean_normalized__{PRIMARY_METHOD}__{channel}__{context}"]
            )[:, horizon_index, target, source].astype(float)
            counterpart = np.asarray(
                atlas[
                    f"mean_normalized__{COUNTERPART_METHOD}__{channel}__{context}"
                ]
            )[:, horizon_index, target, source].astype(float)
            absolute = np.abs(primary)
            peak_index = int(np.argmax(absolute))
            ordered = np.sort(absolute)[::-1]
            top = float(ordered[0])
            second = float(ordered[1])
            selectivity = float((top - second) / (top + 1e-12))
            profile_rho = (
                float(spearmanr(primary, counterpart).statistic)
                if np.std(primary) > 0 and np.std(counterpart) > 0
                else float("nan")
            )
            row.update(
                {
                    "queue_rank": rank,
                    "lag_profile_frames": json.dumps(list(lags)),
                    "lag_profile_source_to_cut_seconds": json.dumps(
                        [value / 4.0 for value in lags]
                    ),
                    "primary_lag_profile": json.dumps(primary.tolist()),
                    "counterpart_lag_profile": json.dumps(counterpart.tolist()),
                    "peak_abs_effect_lag_frames": lags[peak_index],
                    "peak_abs_effect_lag_seconds": lags[peak_index] / 4.0,
                    "peak_abs_effect": top,
                    "second_abs_effect": second,
                    "top_vs_second_lag_selectivity": selectivity,
                    "lag_profile_spearman": profile_rho,
                    "signed_lag_profile_spearman": (
                        profile_rho if bool(row["is_signed"]) else np.nan
                    ),
                    # Lag localization is descriptive and intentionally not a
                    # selection input.  The prediction table does not retain
                    # worm-bootstrap lag rates for rows outside the primary queue.
                    "worm_bootstrap_peak_lag_selection_rate": np.nan,
                    "worm_bootstrap_peak_lag_rates": "",
                    "lag_interpretation": (
                        "descriptive model lag localization; not a causal or physical delay"
                    ),
                    "run_confirmation": True,
                }
            )
            output: dict[str, Any] = {}
            for column in CANONICAL_QUEUE_COLUMNS:
                value = row.get(column, "")
                if pd.isna(value):
                    value = ""
                output[column] = value
            output["run_confirmation"] = True
            rows.append(output)
    frame = pd.DataFrame(
        rows, columns=list(CANONICAL_QUEUE_COLUMNS) + list(ADDED_COLUMNS)
    )
    if frame["queue_rank"].tolist() != list(range(1, len(frame) + 1)):
        raise AssertionError("distributional queue ranks are not contiguous")
    return frame


def _rationale(
    selected: pd.DataFrame,
    *,
    source_prediction_rows: int,
    eligible_counts: Mapping[str, int],
) -> str:
    lines = [
        "# Frozen distributional/stimulus targeted-confirmation selection",
        "",
        (
            f"Selected **{len(selected)}** off-diagonal hypotheses directly from "
            f"{source_prediction_rows:,} reviewed canonical prediction rows."
        ),
        "",
        "## Frozen design",
        "",
        (
            "The design gives endpoint SD, endpoint log-SD, and endpoint "
            "Wasserstein-1 two slots each. It also gives the global "
            "onset-minus-baseline contrast, a chemical onset-minus-baseline contrast, "
            "and a single observed phase two slots each."
        ),
        "",
        (
            "All rows require progressive raw support >= 0.50 and the f=0.10 "
            "genealogy gate. Signed rows additionally require a canonical "
            "supported-exploratory label, direct support >= 0.50, and the same "
            "nonzero direct/progressive sign. State-specific W1 is unsigned and may "
            "therefore remain model-only."
        ),
        "",
        (
            "The assignment maximizes the number of distinct source neurons, then "
            "the sum of canonical internal evidence after direct/progressive "
            "agreement weighting, then stable lexical ties. No lag-localization "
            "field is used for selection."
        ),
        "",
        "## Selected hypotheses",
        "",
        "| Rank | Stratum | Channel | Source -> target | Context | Lag | Horizon | Tier | Internal evidence |",
        "|---:|---|---|---|---|---:|---:|---|---:|",
    ]
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        lines.append(
            "| {rank} | {stratum} | {channel} | {source} -> {target} | {context} | "
            "{lag} | {horizon} | {tier} | {score:.6g} |".format(
                rank=rank,
                stratum=str(row["design_stratum"]),
                channel=str(row["channel"]),
                source=str(row["source_neuron"]),
                target=str(row["target_neuron"]),
                context=str(row["context"]),
                lag=int(row["source_lag_frames"]),
                horizon=int(row["horizon_frames"]),
                tier=str(row["support_tier"]),
                score=float(row["evidence_score"]),
            )
        )
    lines.extend(
        [
            "",
            "## Eligible rows by stratum",
            "",
        ]
    )
    for value in DESIGN_STRATA:
        lines.append(f"- `{value.name}`: {int(eligible_counts[value.name]):,}")
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            (
                "Chemical rows are event-stratified under the binary-any-stimulus "
                "generator; they are not chemically conditioned counterfactuals. "
                "The higher-particle run is a selection-conditioned model-consistency "
                "check, not independent experimental confirmation."
            ),
            "",
            (
                "No anatomical, functional-connectome, neuromodulator, historical "
                "SBTG, or other external reference was loaded or used for selection."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def freeze_distributional_selection(
    atlas_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate and atomically freeze the six-row exploratory shortlist."""

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
    inventory, source_manifest, protocol, prediction, support = _source_bundle(
        atlas_dir
    )
    metadata = _atlas_metadata(atlas_dir)
    neuron_to_index = {
        neuron: index for index, neuron in enumerate(metadata["neurons"])
    }
    if not (
        prediction["source_neuron"].map(neuron_to_index).eq(prediction["source_index"]).all()
        and prediction["target_neuron"].map(neuron_to_index).eq(prediction["target_index"]).all()
    ):
        raise DistributionalSelectionError("prediction neuron labels and indices disagree")
    candidates, eligible_counts = build_eligible_candidates(
        atlas_dir, prediction, support, metadata
    )
    selected = select_balanced_candidates(candidates)
    staged_queue = _queue_rows(atlas_dir, selected, metadata)
    selected_metadata = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        selected_metadata.append(
            {
                "selection_rank": rank,
                "design_stratum": str(row["design_stratum"]),
                "channel": str(row["channel"]),
                "context": str(row["context"]),
                "chemical": str(row["chemical"]),
                "source_neuron": str(row["source_neuron"]),
                "target_neuron": str(row["target_neuron"]),
                "source_index": int(row["source_index"]),
                "target_index": int(row["target_index"]),
                "source_lag_frames": int(row["source_lag_frames"]),
                "horizon_frames": int(row["horizon_frames"]),
                "signed_estimand": bool(row["is_signed"]),
                "support_tier": str(row["support_tier"]),
                "primary_valid_fraction": float(row["valid_fraction"]),
                "genealogy_valid_fraction_0_10": float(
                    row["genealogy_valid_fraction_0_10"]
                ),
                "counterpart_valid_fraction": float(
                    row["counterpart_valid_fraction"]
                ),
                "base_evidence_score": float(row["base_evidence_score"]),
                "internal_evidence_score": float(row["evidence_score"]),
                "source_reused": bool(row["selection_source_reused"]),
            }
        )
    selection_fingerprint = _canonical_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "source_prediction_cells_sha256": inventory["prediction_cells.parquet"],
            "source_support_cells_sha256": inventory["support_cells.parquet"],
            "source_atlas_matrices_sha256": inventory["atlas_matrices.npz"],
            "selected_prediction_keys": [
                {
                    key: row[key]
                    for key in (
                        "channel",
                        "context",
                        "source_index",
                        "target_index",
                        "source_lag_frames",
                        "horizon_frames",
                    )
                }
                for row in selected_metadata
            ],
        }
    )
    bundle_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source_atlas_created_utc": source_manifest.get("created_utc"),
        "source_atlas": str(atlas_dir),
        "source_manifest_sha256": inventory["manifest.json"],
        "source_protocol_sha256": inventory["protocol.json"],
        "source_validation_sha256": inventory["validation.json"],
        "source_prediction_cells_sha256": inventory["prediction_cells.parquet"],
        "source_support_cells_sha256": inventory["support_cells.parquet"],
        "source_atlas_matrices_sha256": inventory["atlas_matrices.npz"],
        "source_hypothesis_queue_used": False,
        "selection_fingerprint": selection_fingerprint,
        "particles": PARTICLES,
        "confirmation_method": CONFIRMATION_METHOD,
        "maximum_rows": MAX_SELECTION_ROWS,
        "selected_rows": len(selected),
        "source_prediction_rows": len(prediction),
        "eligible_candidate_rows": len(candidates),
        "eligible_rows_by_stratum": eligible_counts,
        "selected_rows_metadata": selected_metadata,
        "selected_channels": list(dict.fromkeys(selected["channel"].astype(str))),
        "selected_contexts": list(dict.fromkeys(selected["context"].astype(str))),
        "selected_source_neurons": list(
            dict.fromkeys(selected["source_neuron"].astype(str))
        ),
        "unique_selected_sources": int(selected["source_neuron"].nunique()),
        "design_strata": [
            {
                "name": value.name,
                "channel": value.channel,
                "context_class": value.context_class,
            }
            for value in DESIGN_STRATA
        ],
        "eligibility": {
            "off_diagonal": True,
            "primary_valid_fraction_minimum": 0.5,
            "genealogy_f_0_10_gate_required": True,
            "signed_rows": (
                "supported_exploratory; direct support >=0.5; same nonzero "
                "direct/progressive sign"
            ),
            "unsigned_state_wasserstein1": "model_only permitted",
        },
        "selection_priority": [
            "maximize frozen design strata assigned distinct source neurons",
            "maximize summed internal evidence",
            "stable lexical candidate keys",
            "reuse a source only when full unique assignment is impossible",
        ],
        "internal_evidence_formula": (
            "canonical prediction evidence * sign gate * direct/progressive magnitude "
            "agreement * slice agreement * minimum source support"
        ),
        "lag_localization_used_for_selection": False,
        "ranking_inputs_used": [
            "canonical prediction_cells.parquet",
            "canonical support_cells.parquet",
            "canonical atlas_matrices.npz",
        ],
        "external_reference_inputs_used": [],
        "external_fields_detected": [],
        "source_orientation": protocol["orientation"],
        "claim_boundary": CLAIM_BOUNDARY,
        "chemical_conditioning": False,
        "screen_dir_compatible": True,
        "downstream_schema_extension_required": False,
        "downstream_compatibility": (
            "existing targeted_confirmation and targeted_confirmation_analysis accept "
            "the canonical 68 queue columns plus run_confirmation and the copied dense matrix"
        ),
        "artifacts": {
            "selection": SELECTION_FILENAME,
            "screen_dense_matrices": "atlas_matrices.npz",
            "rationale": RATIONALE_FILENAME,
            "manifest": MANIFEST_FILENAME,
            "checksums": CHECKSUM_FILENAME,
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        staged_queue.to_csv(
            temporary / SELECTION_FILENAME, index=False, lineterminator="\n"
        )
        shutil.copyfile(
            atlas_dir / "atlas_matrices.npz", temporary / "atlas_matrices.npz"
        )
        if sha256(temporary / "atlas_matrices.npz") != inventory["atlas_matrices.npz"]:
            raise DistributionalSelectionError("dense matrix copy failed checksum")
        (temporary / RATIONALE_FILENAME).write_text(
            _rationale(
                selected,
                source_prediction_rows=len(prediction),
                eligible_counts=eligible_counts,
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
        for name, expected in inventory.items():
            if sha256(atlas_dir / name) != expected:
                raise DistributionalSelectionError(
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = freeze_distributional_selection(args.atlas_dir, args.output_dir)
    print(
        f"FROZEN_DISTRIBUTIONAL_SELECTION rows={manifest['selected_rows']} "
        f"particles={manifest['particles']} output={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
