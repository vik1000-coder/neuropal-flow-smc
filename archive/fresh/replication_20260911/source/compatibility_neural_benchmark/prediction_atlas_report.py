"""Build the canonical portable technical-report artifact for the neural atlas.

The generator reads only reviewed, checksum-backed products.  It writes a
``surface: report`` ``artifact.json`` for the installed Data Analytics builder
and a reproducible Markdown companion.  It deliberately does not render HTML,
load raw reference releases, or alter any atlas, targeted, or post-freeze
result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.prediction_atlas_dashboard import (
    DashboardInputError,
    _candidate_rows,
    _crosscheck_snapshot_candidates,
    _primary_agreement_profile,
    _read_queue,
    _stimulus_context_rows,
    _validate_bundle_metadata,
    TARGETED_METHOD,
    TARGETED_SELECTION_SCHEMA,
)
from compatibility_neural_benchmark.prediction_atlas_external_analysis import (
    EXPECTED_NEUROMODULATOR_SOURCES,
    REFERENCE_RELEASE_FILES,
)
from compatibility_neural_benchmark.targeted_confirmation_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    INTERPRETATION_LIMIT,
    ORIENTATION,
)


TITLE = "Neural lag-effect atlas: technical results and claim boundaries"
ATLAS_REQUIRED = (
    "manifest.json",
    "protocol.json",
    "validation.json",
    "models.json",
    "dashboard_snapshot.json",
    "atlas_matrices.npz",
    "hypothesis_queue.csv",
    "stimulus_composition.csv",
    "checksums.sha256",
)
TARGETED_REQUIRED = (
    "manifest.json",
    "protocol.json",
    "validation.json",
    "summary.json",
    "targeted_cells.csv",
    "targeted_quantile_shifts.csv",
    "support_diagnostics.csv",
    "screen_consistency.csv",
    "input_checksums.csv",
    "checksums.sha256",
)
EXTERNAL_REQUIRED = (
    "manifest.json",
    "randi_cook_metrics.csv",
    "neuromodulator_metrics.csv",
    "neuromodulator_lagmax_inference.csv",
    "primary_randi_cook_profile.csv",
    "primary_neuromodulator_profile.csv",
    "dashboard_external_summary.csv",
    "input_checksums.sha256",
    "checksums.sha256",
)
ROOT_DOCUMENTS = (
    "README.md",
    "RUN_PROTOCOL.md",
    "CODE_AND_ESTIMAND_AUDIT.md",
    "INPUT_DATA_AUDIT.md",
)
OUTPUT_FILES = (
    "artifact.json",
    "TECHNICAL_REPORT.md",
    "checksums.sha256",
    "report.html",
)
PRIMARY_METHOD = "progressive_bridge_smc"
MAX_QUEUE_ROWS = 25
MAX_TARGETED_CANDIDATES = 6
MAX_TARGETED_ROWS = 108
MAX_QUANTILE_ROWS = 120
MAX_SUPPORT_ROWS = 40
MAX_SCREEN_ROWS = 25
MAX_EXTERNAL_ROWS = 100
MAX_LAGMAX_ROWS = 100
MAX_STIMULUS_ROWS = 40
MAX_ARTIFACT_BYTES = 3_000_000
CHECKSUM_PATTERN = re.compile(r"[0-9a-f]{64}")
ALLOWED_TARGETED_LABELS = {
    "model_relative_consistent",
    "model_relative_uncertain",
    "model_relative_descriptive",
    "unsupported_model_output",
}


class ReportInputError(RuntimeError):
    """Raised when a canonical report input fails closed validation."""


@dataclass(frozen=True)
class AtlasEvidence:
    root: Path
    manifest: Mapping[str, Any]
    protocol: Mapping[str, Any]
    validation: Mapping[str, Any]
    models: Mapping[str, Any]
    dashboard: Mapping[str, Any]
    queue: pd.DataFrame
    candidate_rows: list[dict[str, Any]]
    stability_rows: list[dict[str, Any]]
    context_rows: list[dict[str, Any]]
    stimulus_rows: list[dict[str, Any]]
    n_worms: int
    n_neurons: int
    neurons: tuple[str, ...]
    fps: float
    checksums: Mapping[str, str]


@dataclass(frozen=True)
class TargetedEvidence:
    root: Path
    manifest: Mapping[str, Any]
    protocol: Mapping[str, Any]
    validation: Mapping[str, Any]
    summary: Mapping[str, Any]
    cells: pd.DataFrame
    quantiles: pd.DataFrame
    support: pd.DataFrame
    screen: pd.DataFrame
    primary_rows: list[dict[str, Any]]
    detail_rows: list[dict[str, Any]]
    quantile_rows: list[dict[str, Any]]
    support_rows: list[dict[str, Any]]
    checksums: Mapping[str, str]


@dataclass(frozen=True)
class ExternalEvidence:
    root: Path
    manifest: Mapping[str, Any]
    comparisons: pd.DataFrame
    lagmax: pd.DataFrame
    comparison_rows: list[dict[str, Any]]
    lagmax_rows: list[dict[str, Any]]
    checksums: Mapping[str, str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportInputError(f"required report input is missing: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReportInputError(f"could not read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ReportInputError(f"{path.name} must contain a JSON object")
    return value


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise ReportInputError(f"required report input is missing: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ReportInputError(f"could not read {label}: {error}") from error


def _strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ReportInputError(f"{label} must be a canonical boolean")


def _safe_relative(path: Path, provenance_root: Path) -> str:
    root = provenance_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ReportInputError(
            f"{resolved.name} lies outside the declared provenance root"
        ) from error
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ReportInputError("portable provenance paths must be local and safe")
    return relative.as_posix()


def _linked_path(
    value: Any, *, base: Path, provenance_root: Path, label: str
) -> Path:
    if value is None or not str(value).strip():
        raise ReportInputError(f"targeted manifest lacks {label}")
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    _safe_relative(resolved, provenance_root)
    return resolved


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is pd.NA or value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value) if not isinstance(value, str) else value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in frame.to_dict(orient="records")]


def _parse_checksum_ledger(path: Path, *, nested: bool = False) -> dict[str, str]:
    if not path.is_file():
        raise ReportInputError(f"missing checksum ledger: {path}")
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not CHECKSUM_PATTERN.fullmatch(parts[0]):
            raise ReportInputError(f"{path.name} line {line_number} is malformed")
        name = parts[1]
        candidate = Path(name)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or not candidate.parts
            or (not nested and len(candidate.parts) != 1)
        ):
            raise ReportInputError(f"{path.name} contains an unsafe path")
        if name in result:
            raise ReportInputError(f"{path.name} duplicates {name}")
        result[name] = parts[0]
    if not result:
        raise ReportInputError(f"{path.name} is empty")
    return result


def _verify_bundle_checksums(root: Path, required: Sequence[str]) -> dict[str, str]:
    for name in required:
        if not (root / name).is_file():
            raise ReportInputError(f"canonical bundle is incomplete: missing {name}")
    ledger = _parse_checksum_ledger(root / "checksums.sha256")
    for name in required:
        if name == "checksums.sha256":
            continue
        expected = ledger.get(name)
        if expected is None:
            raise ReportInputError(f"checksum ledger omits required file {name}")
    for name, expected in ledger.items():
        candidate = root / name
        if not candidate.is_file() or sha256(candidate) != expected:
            raise ReportInputError(f"checksum verification failed for {root.name}/{name}")
    return ledger


def _numeric(
    frame: pd.DataFrame,
    fields: Sequence[str],
    *,
    label: str,
    allow_missing: bool = False,
) -> None:
    missing = sorted(set(fields).difference(frame.columns))
    if missing:
        raise ReportInputError(f"{label} lacks required columns {missing}")
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
        if allow_missing:
            invalid = ~np.isfinite(values) & ~pd.isna(frame[field]).to_numpy()
        else:
            invalid = ~np.isfinite(values)
        if invalid.any():
            raise ReportInputError(f"{label}.{field} contains invalid numeric values")
        frame[field] = values


def _validate_atlas(atlas_dir: Path) -> AtlasEvidence:
    root = atlas_dir.resolve()
    checksums = _verify_bundle_checksums(root, ATLAS_REQUIRED)
    manifest = _read_json(root / "manifest.json")
    protocol = _read_json(root / "protocol.json")
    validation = _read_json(root / "validation.json")
    models = _read_json(root / "models.json")
    dashboard = _read_json(root / "dashboard_snapshot.json")
    if (
        manifest.get("status") != "complete"
        or manifest.get("protocol") != "reviewed neural prediction atlas v1"
        or manifest.get("primary_method") != PRIMARY_METHOD
        or manifest.get("dense_matrix_orientation") != ORIENTATION
    ):
        raise ReportInputError("canonical atlas manifest contract failed")
    try:
        n_worms, n_neurons, fps = _validate_bundle_metadata(
            dashboard, validation, protocol, models
        )
        queue = _read_queue(root / "hypothesis_queue.csv", n_worms=n_worms, fps=fps)
        _crosscheck_snapshot_candidates(dashboard, queue)
        stability = _primary_agreement_profile(dashboard, validation, fps)
        context_rows = _stimulus_context_rows(dashboard, validation)
    except DashboardInputError as error:
        raise ReportInputError(str(error)) from error
    if not (queue["method"].astype(str) == PRIMARY_METHOD).all():
        raise ReportInputError("canonical queue contains a non-primary sampler row")
    if (queue["valid_fraction"].astype(float) < 0.50).any():
        raise ReportInputError("canonical queue contains a support-ineligible row")
    with np.load(root / "atlas_matrices.npz", allow_pickle=False) as dense:
        required = {"orientation", "neurons", "methods", "channels", "contexts"}
        if required.difference(dense.files):
            raise ReportInputError("dense atlas metadata is incomplete")
        if str(np.asarray(dense["orientation"]).item()) != ORIENTATION:
            raise ReportInputError("dense atlas orientation contract failed")
        neurons = tuple(dense["neurons"].astype(str))
        if len(neurons) != n_neurons or len(set(neurons)) != n_neurons:
            raise ReportInputError("dense atlas neuron axis is invalid")
    if {"source_index", "target_index"}.difference(queue.columns):
        raise ReportInputError("canonical queue omits neuron-axis indices")
    source_indices = pd.to_numeric(queue["source_index"], errors="coerce").to_numpy()
    target_indices = pd.to_numeric(queue["target_index"], errors="coerce").to_numpy()
    if (
        not np.isfinite(source_indices).all()
        or not np.isfinite(target_indices).all()
        or not np.equal(source_indices, np.floor(source_indices)).all()
        or not np.equal(target_indices, np.floor(target_indices)).all()
        or (source_indices < 0).any()
        or (target_indices < 0).any()
        or (source_indices >= n_neurons).any()
        or (target_indices >= n_neurons).any()
    ):
        raise ReportInputError("canonical queue neuron-axis indices are invalid")
    neuron_array = np.asarray(neurons)
    if (
        not np.array_equal(
            queue["source_neuron"].astype(str).to_numpy(),
            neuron_array[source_indices.astype(int)],
        )
        or not np.array_equal(
            queue["target_neuron"].astype(str).to_numpy(),
            neuron_array[target_indices.astype(int)],
        )
    ):
        raise ReportInputError("canonical queue neuron names do not match dense axes")
    summary = dashboard.get("support_summary")
    if not isinstance(summary, Mapping):
        raise ReportInputError("dashboard support summary is absent")
    mean_valid = float(summary.get("mean_valid_fraction", np.nan))
    if not math.isfinite(mean_valid) or not 0 <= mean_valid <= 1:
        raise ReportInputError("dashboard mean validity is invalid")
    stimulus_summary = dashboard.get("stimulus_composition_summary", [])
    if not isinstance(stimulus_summary, list):
        raise ReportInputError("dashboard stimulus-composition summary is invalid")
    stimulus_rows: list[dict[str, Any]] = []
    if stimulus_summary:
        composition_path = root / "stimulus_composition.csv"
        composition_protocol = protocol.get("stimulus_composition")
        if (
            not composition_path.is_file()
            or "stimulus_composition.csv" not in checksums
            or not isinstance(composition_protocol, Mapping)
            or composition_protocol.get("artifact") != "stimulus_composition.csv"
        ):
            raise ReportInputError(
                "declared stimulus-composition summary lacks its checksummed exact table"
            )
        exact_composition = _read_csv(
            composition_path, label="stimulus_composition.csv"
        )
        exact_required = {
            "worm_id",
            "phase",
            "event_index",
            "source_lag_frames",
            "horizon_frames",
            "source_window_stimulus_fraction",
            "cut_stimulus_indicator",
            "forecast_window_stimulus_fraction",
            "forecast_endpoint_stimulus_indicator",
        }
        if (
            exact_composition.empty
            or exact_required.difference(exact_composition.columns)
            or len(exact_composition)
            != int(composition_protocol.get("row_count", -1))
        ):
            raise ReportInputError(
                "exact stimulus-composition table disagrees with its protocol"
            )
        stimulus_frame = pd.DataFrame(stimulus_summary)
        required_stimulus = {
            "phase",
            "source_lag_frames",
            "horizon_frames",
            "n_worm_events",
            "source_window_stimulus_fraction_min",
            "source_window_stimulus_fraction_mean",
            "source_window_stimulus_fraction_max",
            "forecast_window_stimulus_fraction_min",
            "forecast_window_stimulus_fraction_mean",
            "forecast_window_stimulus_fraction_max",
            "forecast_endpoint_stimulus_fraction_mean",
            "source_window_crosses_stimulus_boundary_any",
            "forecast_window_crosses_stimulus_boundary_any",
            "cut_to_endpoint_stimulus_transition_any",
        }
        missing_stimulus = sorted(required_stimulus.difference(stimulus_frame.columns))
        if missing_stimulus:
            raise ReportInputError(
                f"stimulus-composition summary lacks fields {missing_stimulus}"
            )
        numeric_stimulus = tuple(
            sorted(
                required_stimulus.difference(
                    {
                        "phase",
                        "source_window_crosses_stimulus_boundary_any",
                        "forecast_window_crosses_stimulus_boundary_any",
                        "cut_to_endpoint_stimulus_transition_any",
                    }
                )
            )
        )
        _numeric(stimulus_frame, numeric_stimulus, label="stimulus_composition_summary")
        fraction_columns = [
            column for column in numeric_stimulus if "fraction" in column
        ]
        if (
            stimulus_frame.duplicated(
                ["phase", "source_lag_frames", "horizon_frames"]
            ).any()
            or (stimulus_frame["n_worm_events"] <= 0).any()
            or not (stimulus_frame["n_worm_events"] == n_worms * 3).all()
            or any(
                ((stimulus_frame[column] < 0) | (stimulus_frame[column] > 1)).any()
                for column in fraction_columns
            )
        ):
            raise ReportInputError("stimulus-composition summary fails grain/range checks")
        crossing = stimulus_frame[
            stimulus_frame[
                [
                    "source_window_crosses_stimulus_boundary_any",
                    "forecast_window_crosses_stimulus_boundary_any",
                    "cut_to_endpoint_stimulus_transition_any",
                ]
            ]
            .astype(bool)
            .any(axis=1)
        ]
        selected_stimulus = crossing if not crossing.empty else stimulus_frame
        stimulus_rows = _records(selected_stimulus.head(MAX_STIMULUS_ROWS))
    candidates = _candidate_rows(queue, min(MAX_QUEUE_ROWS, len(queue)))
    return AtlasEvidence(
        root=root,
        manifest=manifest,
        protocol=protocol,
        validation=validation,
        models=models,
        dashboard=dashboard,
        queue=queue,
        candidate_rows=candidates,
        stability_rows=stability,
        context_rows=context_rows,
        stimulus_rows=stimulus_rows,
        n_worms=n_worms,
        n_neurons=n_neurons,
        neurons=neurons,
        fps=fps,
        checksums=checksums,
    )


def _validate_targeted(
    targeted_dir: Path,
    *,
    n_worms: int,
    neurons: Sequence[str],
    atlas_dir: Path,
    atlas_checksums: Mapping[str, str],
    provenance_root: Path,
) -> TargetedEvidence:
    root = targeted_dir.resolve()
    checksums = _verify_bundle_checksums(root, TARGETED_REQUIRED)
    manifest = _read_json(root / "manifest.json")
    protocol = _read_json(root / "protocol.json")
    validation = _read_json(root / "validation.json")
    summary = _read_json(root / "summary.json")
    n_neurons = len(neurons)
    if (
        manifest.get("status") != "complete"
        or manifest.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION
        or manifest.get("matrix_orientation") != ORIENTATION
    ):
        raise ReportInputError("targeted-analysis manifest contract failed")
    checks = validation.get("checks")
    if (
        validation.get("status") != "passed"
        or validation.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or int(validation.get("archive_count", -1))
        != int(validation.get("expected_archive_count", -2))
        or int(validation.get("n_worms", -1)) != n_worms
    ):
        raise ReportInputError("targeted-analysis validation did not pass completely")
    if (
        summary.get("status") != "reviewed_model_relative"
        or summary.get("experimental_confirmation_label_assigned") is not False
        or summary.get("external_reference_data_used") is not False
        or int(summary.get("n_worms", -1)) != n_worms
    ):
        raise ReportInputError("targeted summary violates its model-relative boundary")
    if (
        protocol.get("independent_unit") != "worm"
        or protocol.get("matrix_orientation") != ORIENTATION
        or int(protocol.get("particle_count", -1)) != 128
        or protocol.get("external_reference_firewall") is None
        or "not chemical-conditioned" not in str(protocol.get("chemical_warning", ""))
        or "not causal" not in str(protocol.get("claim_boundary", ""))
    ):
        raise ReportInputError("targeted protocol boundary or analysis grain failed")
    input_checksums = _read_csv(
        root / "input_checksums.csv", label="targeted input_checksums.csv"
    )
    if set(("path", "sha256")).difference(input_checksums.columns) or input_checksums.empty:
        raise ReportInputError("targeted input checksum inventory is malformed")
    linked_input_hashes: dict[Path, str] = {}
    for row in input_checksums.itertuples(index=False):
        path = Path(str(row.path)).resolve()
        digest = str(row.sha256)
        if (
            path in linked_input_hashes
            or CHECKSUM_PATTERN.fullmatch(digest) is None
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ReportInputError("a targeted raw input changed after analysis")
        linked_input_hashes[path] = digest

    selection_dir = _linked_path(
        manifest.get("screen_dir"),
        base=root,
        provenance_root=provenance_root,
        label="screen_dir",
    )
    _verify_bundle_checksums(
        selection_dir,
        (
            "manifest.json",
            "hypothesis_queue.csv",
            "atlas_matrices.npz",
            "selection_rationale.md",
            "checksums.sha256",
        ),
    )
    selection_manifest = _read_json(selection_dir / "manifest.json")
    selection_queue_path = selection_dir / "hypothesis_queue.csv"
    selection_dense_path = selection_dir / "atlas_matrices.npz"
    selection_queue_sha = sha256(selection_queue_path)
    try:
        selection_maximum_rows = int(selection_manifest.get("maximum_rows", -1))
    except (TypeError, ValueError) as error:
        raise ReportInputError("targeted selection maximum_rows is invalid") from error
    source_atlas = _linked_path(
        selection_manifest.get("source_atlas"),
        base=selection_dir,
        provenance_root=provenance_root,
        label="selection source_atlas",
    )
    canonical_queue_sha = atlas_checksums.get("hypothesis_queue.csv")
    if (
        canonical_queue_sha is None
        or selection_manifest.get("schema_version") != TARGETED_SELECTION_SCHEMA
        or selection_manifest.get("status") != "complete"
        or source_atlas != atlas_dir.resolve()
        or selection_manifest.get("source_manifest_sha256")
        != atlas_checksums.get("manifest.json")
        or selection_manifest.get("source_protocol_sha256")
        != atlas_checksums.get("protocol.json")
        or selection_manifest.get("source_validation_sha256")
        != atlas_checksums.get("validation.json")
        or selection_manifest.get("source_atlas_matrices_sha256")
        != atlas_checksums.get("atlas_matrices.npz")
        or sha256(selection_dense_path)
        != atlas_checksums.get("atlas_matrices.npz")
        or selection_manifest.get("frozen_hypothesis_queue_sha256")
        != canonical_queue_sha
        or selection_manifest.get("selection_sha256") != selection_queue_sha
        or not 1 <= selection_maximum_rows <= MAX_TARGETED_CANDIDATES
        or int(selection_manifest.get("particles", -1)) != 128
        or selection_manifest.get("confirmation_method") != TARGETED_METHOD
        or selection_manifest.get("screen_dir_compatible") is not True
        or selection_manifest.get("external_reference_inputs_used") != []
        or selection_manifest.get("external_fields_detected") != []
        or manifest.get("input_queue_sha256") != selection_queue_sha
        or linked_input_hashes.get(selection_queue_path.resolve())
        != selection_queue_sha
        or linked_input_hashes.get(selection_dense_path.resolve())
        != sha256(selection_dense_path)
    ):
        raise ReportInputError(
            "targeted analysis is not linked through the frozen atlas selection"
        )

    run_dir = _linked_path(
        manifest.get("input_run_dir"),
        base=root,
        provenance_root=provenance_root,
        label="input_run_dir",
    )
    run_manifest_path = run_dir / "manifest.json"
    expected_run_manifest_sha = str(
        manifest.get("input_run_manifest_sha256", "")
    )
    if (
        CHECKSUM_PATTERN.fullmatch(expected_run_manifest_sha) is None
        or not run_manifest_path.is_file()
        or sha256(run_manifest_path) != expected_run_manifest_sha
        or linked_input_hashes.get(run_manifest_path.resolve())
        != expected_run_manifest_sha
    ):
        raise ReportInputError("targeted raw-run manifest linkage failed")
    run_manifest = _read_json(run_manifest_path)
    run_queue = _linked_path(
        run_manifest.get("hypothesis_queue"),
        base=run_dir,
        provenance_root=provenance_root,
        label="targeted run hypothesis_queue",
    )
    if (
        run_queue != selection_queue_path.resolve()
        or run_manifest.get("hypothesis_queue_sha256") != selection_queue_sha
        or manifest.get("input_queue_sha256") != selection_queue_sha
        or run_manifest.get("method") != TARGETED_METHOD
        or int(run_manifest.get("particles", -1)) != 128
        or run_manifest.get("matrix_orientation") != ORIENTATION
        or run_manifest.get("stimulus_generator_encoding")
        != "binary_any_stimulus"
        or run_manifest.get("chemical_identity_conditioned") is not False
    ):
        raise ReportInputError("targeted raw run is not linked to the staged N128 queue")
    canonical_text = pd.read_csv(
        atlas_dir / "hypothesis_queue.csv", dtype=str, keep_default_na=False
    )
    selected_text = pd.read_csv(
        selection_queue_path, dtype=str, keep_default_na=False
    )
    if (
        list(selected_text.columns)
        != [*canonical_text.columns, "run_confirmation"]
        or selected_text.empty
        or len(selected_text) > MAX_TARGETED_CANDIDATES
        or len(selected_text) > selection_maximum_rows
        or not selected_text["run_confirmation"]
        .astype(str)
        .str.lower()
        .isin(("true", "1"))
        .all()
        or canonical_text["queue_rank"].duplicated().any()
        or selected_text["queue_rank"].duplicated().any()
        or int(selection_manifest.get("selected_rows", -1)) != len(selected_text)
        or int(summary.get("n_candidates", -1)) != len(selected_text)
    ):
        raise ReportInputError("targeted staged selection queue is malformed")
    canonical_by_rank = canonical_text.set_index("queue_rank", drop=False)
    for row in selected_text.itertuples(index=False):
        rank = str(row.queue_rank)
        if rank not in canonical_by_rank.index:
            raise ReportInputError("targeted selection rank is absent from canonical queue")
        canonical_row = canonical_by_rank.loc[rank]
        if isinstance(canonical_row, pd.DataFrame) or any(
            str(getattr(row, column)) != str(canonical_row[column])
            for column in canonical_text.columns
        ):
            raise ReportInputError("targeted selection row differs from canonical queue")

    cells = _read_csv(root / "targeted_cells.csv", label="targeted_cells.csv")
    quantiles = _read_csv(
        root / "targeted_quantile_shifts.csv", label="targeted_quantile_shifts.csv"
    )
    support = _read_csv(
        root / "support_diagnostics.csv", label="support_diagnostics.csv"
    )
    screen = _read_csv(root / "screen_consistency.csv", label="screen_consistency.csv")
    required_cells = {
        "candidate_id",
        "queue_rank",
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
        "context",
        "contrast",
        "metric_family",
        "metric",
        "signed",
        "mean_normalized",
        "mean_raw",
        "ci_2_5",
        "ci_97_5",
        "valid_fraction",
        "n_worms",
        "n_model_seeds",
        "n_particles",
        "matrix_orientation",
        "row_axis",
        "column_axis",
        "evidence_label",
        "interpretation_limit",
    }
    missing = sorted(required_cells.difference(cells.columns))
    if missing or cells.empty:
        raise ReportInputError(f"targeted cells are incomplete: {missing}")
    _numeric(
        cells,
        (
            "queue_rank",
            "source_index",
            "target_index",
            "source_lag_frames",
            "horizon_frames",
            "mean_normalized",
            "mean_raw",
            "ci_2_5",
            "ci_97_5",
            "valid_fraction",
            "n_worms",
            "n_model_seeds",
            "n_particles",
        ),
        label="targeted_cells",
    )
    if (
        len(cells) != int(summary.get("n_cell_rows", -1))
        or cells["candidate_id"].nunique() != int(summary.get("n_candidates", -1))
        or not (cells["n_worms"] == n_worms).all()
        or not (cells["n_particles"] == 128).all()
        or not (cells["matrix_orientation"] == ORIENTATION).all()
        or not (cells["row_axis"] == "target_neuron").all()
        or not (cells["column_axis"] == "source_neuron").all()
        or ((cells["valid_fraction"] < 0) | (cells["valid_fraction"] > 1)).any()
        or (cells["ci_2_5"] > cells["ci_97_5"]).any()
        or (cells["source_lag_frames"] <= 0).any()
        or (cells["horizon_frames"] <= 0).any()
        or not np.equal(
            cells["source_lag_frames"], np.floor(cells["source_lag_frames"])
        ).all()
        or not np.equal(
            cells["horizon_frames"], np.floor(cells["horizon_frames"])
        ).all()
        or (cells["n_model_seeds"] <= 0).any()
        or not np.equal(cells["n_model_seeds"], np.floor(cells["n_model_seeds"])).all()
        or (cells["target_index"] < 0).any()
        or (cells["source_index"] < 0).any()
        or (cells["target_index"] >= n_neurons).any()
        or (cells["source_index"] >= n_neurons).any()
        or not np.equal(cells["target_index"], np.floor(cells["target_index"])).all()
        or not np.equal(cells["source_index"], np.floor(cells["source_index"])).all()
        or not set(cells["contrast"]).issubset(
            {"high_low", "high_factual", "low_factual"}
        )
        or not set(cells["evidence_label"]).issubset(ALLOWED_TARGETED_LABELS)
        or cells["evidence_label"].astype(str).str.contains("confirmed", case=False).any()
        or not (cells["interpretation_limit"].astype(str) == INTERPRETATION_LIMIT).all()
    ):
        raise ReportInputError("targeted cells fail orientation, grain, or label checks")
    expected_candidate_ids = [
        f"targeted_{index + 1:04d}" for index in range(len(selected_text))
    ]
    if set(cells["candidate_id"].astype(str)) != set(expected_candidate_ids):
        raise ReportInputError("targeted cells do not cover the frozen selected queue")
    for index, selected_row in selected_text.reset_index(drop=True).iterrows():
        candidate = expected_candidate_ids[index]
        current = cells[cells["candidate_id"].astype(str) == candidate]
        expected_numeric = {
            "queue_rank": int(selected_row["queue_rank"]),
            "source_index": int(selected_row["source_index"]),
            "target_index": int(selected_row["target_index"]),
            "source_lag_frames": int(selected_row["source_lag_frames"]),
            "horizon_frames": int(selected_row["horizon_frames"]),
        }
        expected_text = {
            "source_neuron": str(selected_row["source_neuron"]),
            "target_neuron": str(selected_row["target_neuron"]),
            "context": str(selected_row["context"]),
        }
        if (
            any(
                current[field].nunique() != 1
                or int(current[field].iloc[0]) != value
                for field, value in expected_numeric.items()
            )
            or any(
                current[field].astype(str).nunique() != 1
                or str(current[field].iloc[0]) != value
                for field, value in expected_text.items()
            )
        ):
            raise ReportInputError(
                "targeted cell metadata differs from the frozen selected queue"
            )
    source_indices = cells["source_index"].astype(int).to_numpy()
    target_indices = cells["target_index"].astype(int).to_numpy()
    neuron_array = np.asarray(tuple(str(value) for value in neurons))
    if (
        not np.array_equal(
            cells["source_neuron"].astype(str).to_numpy(), neuron_array[source_indices]
        )
        or not np.array_equal(
            cells["target_neuron"].astype(str).to_numpy(), neuron_array[target_indices]
        )
    ):
        raise ReportInputError("targeted source/target names do not match atlas indices")
    contrasts = ("high_low", "high_factual", "low_factual")
    signed_metrics = tuple(str(value) for value in protocol.get("signed_metrics", ()))
    if not signed_metrics or "endpoint_wasserstein1" in signed_metrics:
        raise ReportInputError("targeted signed-metric protocol is invalid")
    expected_metrics = set(signed_metrics) | {"endpoint_wasserstein1"}
    expected_grid = {
        (candidate, contrast, metric)
        for candidate in cells["candidate_id"].astype(str).unique()
        for contrast in contrasts
        for metric in expected_metrics
    }
    actual_grid = set(
        zip(
            cells["candidate_id"].astype(str),
            cells["contrast"].astype(str),
            cells["metric"].astype(str),
        )
    )
    if len(cells) != len(actual_grid) or actual_grid != expected_grid:
        raise ReportInputError("targeted cells do not contain the full requested metric grid")
    expected_family = np.where(
        cells["metric"].astype(str) == "endpoint_wasserstein1",
        "pairwise_wasserstein1",
        "signed_contrast",
    )
    if not np.array_equal(cells["metric_family"].astype(str).to_numpy(), expected_family):
        raise ReportInputError("targeted metric-family labels do not match their formulas")
    parsed_signed = np.asarray(
        [_strict_bool(value, label="targeted signed") for value in cells["signed"]],
        dtype=bool,
    )
    expected_signed = (
        cells["metric"].astype(str).to_numpy() != "endpoint_wasserstein1"
    ) | cells["context"].astype(str).str.endswith("minus_baseline").to_numpy()
    if not np.array_equal(parsed_signed, expected_signed):
        raise ReportInputError(
            "targeted signed flags disagree with metric/context semantics"
        )
    cells["signed"] = parsed_signed
    unsigned_w1 = ~parsed_signed
    if unsigned_w1.any() and (
        (cells.loc[unsigned_w1, "mean_normalized"] < -1e-12).any()
        or (cells.loc[unsigned_w1, "mean_raw"] < -1e-12).any()
        or (cells.loc[unsigned_w1, "ci_2_5"] < -1e-12).any()
    ):
        raise ReportInputError("unsigned targeted Wasserstein rows must be nonnegative")
    if len(quantiles) != int(summary.get("n_quantile_rows", -1)):
        raise ReportInputError("targeted quantile row count differs from summary")
    evidence_counts = summary.get("evidence_label_counts")
    observed_counts = {
        str(key): int(value)
        for key, value in cells["evidence_label"].value_counts().items()
    }
    if not isinstance(evidence_counts, Mapping) or {
        str(key): int(value) for key, value in evidence_counts.items()
    } != observed_counts:
        raise ReportInputError(
            "targeted evidence-label counts disagree with targeted cells"
        )
    if not quantiles.empty:
        required_quantile = {
            "candidate_id",
            "source_neuron",
            "target_neuron",
            "source_index",
            "target_index",
            "source_lag_frames",
            "horizon_frames",
            "context",
            "contrast",
            "signed",
            "endpoint_quantile",
            "mean_normalized",
            "ci_2_5",
            "ci_97_5",
            "valid_fraction",
            "matrix_orientation",
            "row_axis",
            "column_axis",
        }
        if required_quantile.difference(quantiles.columns):
            raise ReportInputError("targeted quantile shifts have an incompatible schema")
        parsed_quantile_signed = np.asarray(
            [
                _strict_bool(value, label="targeted quantile signed")
                for value in quantiles["signed"]
            ],
            dtype=bool,
        )
        if not parsed_quantile_signed.all():
            raise ReportInputError("targeted quantile shifts must remain signed")
        quantiles["signed"] = parsed_quantile_signed
        _numeric(
            quantiles,
            (
                "endpoint_quantile",
                "source_index",
                "target_index",
                "source_lag_frames",
                "horizon_frames",
                "mean_normalized",
                "ci_2_5",
                "ci_97_5",
                "valid_fraction",
            ),
            label="targeted_quantiles",
        )
        if (
            ((quantiles["endpoint_quantile"] <= 0) | (quantiles["endpoint_quantile"] >= 1)).any()
            or ((quantiles["valid_fraction"] < 0) | (quantiles["valid_fraction"] > 1)).any()
            or (quantiles["ci_2_5"] > quantiles["ci_97_5"]).any()
            or not (quantiles["matrix_orientation"] == ORIENTATION).all()
            or not (quantiles["row_axis"] == "target_neuron").all()
            or not (quantiles["column_axis"] == "source_neuron").all()
        ):
            raise ReportInputError("targeted quantile shifts fail range/orientation checks")
        q_source = quantiles["source_index"].astype(int).to_numpy()
        q_target = quantiles["target_index"].astype(int).to_numpy()
        if (
            (q_source < 0).any()
            or (q_source >= n_neurons).any()
            or (q_target < 0).any()
            or (q_target >= n_neurons).any()
            or not np.equal(quantiles["source_index"], q_source).all()
            or not np.equal(quantiles["target_index"], q_target).all()
            or not np.array_equal(
                quantiles["source_neuron"].astype(str).to_numpy(), neuron_array[q_source]
            )
            or not np.array_equal(
                quantiles["target_neuron"].astype(str).to_numpy(), neuron_array[q_target]
            )
        ):
            raise ReportInputError("targeted quantile neuron axes do not match the atlas")
        candidate_metadata = (
            cells.drop_duplicates("candidate_id")
            .set_index("candidate_id")[
                [
                    "source_neuron",
                    "target_neuron",
                    "source_index",
                    "target_index",
                    "source_lag_frames",
                    "horizon_frames",
                    "context",
                ]
            ]
        )
        for row in quantiles.itertuples(index=False):
            candidate = str(row.candidate_id)
            if candidate not in candidate_metadata.index:
                raise ReportInputError(
                    "targeted quantile candidate is absent from N128 cells"
                )
            expected = candidate_metadata.loc[candidate]
            if (
                str(row.source_neuron) != str(expected.source_neuron)
                or str(row.target_neuron) != str(expected.target_neuron)
                or int(row.source_index) != int(expected.source_index)
                or int(row.target_index) != int(expected.target_index)
                or int(row.source_lag_frames) != int(expected.source_lag_frames)
                or int(row.horizon_frames) != int(expected.horizon_frames)
                or str(row.context) != str(expected.context)
            ):
                raise ReportInputError(
                    "targeted quantile metadata disagrees with its selected candidate"
                )
        quantiles["pair_label"] = (
            quantiles["source_neuron"].astype(str)
            + " → "
            + quantiles["target_neuron"].astype(str)
        )
        endpoint_quantiles = tuple(
            float(value) for value in protocol.get("endpoint_quantiles", ())
        )
        expected_quantile_grid = {
            (candidate, contrast, quantile)
            for candidate in cells["candidate_id"].astype(str).unique()
            for contrast in contrasts
            for quantile in endpoint_quantiles
        }
        actual_quantile_grid = set(
            zip(
                quantiles["candidate_id"].astype(str),
                quantiles["contrast"].astype(str),
                quantiles["endpoint_quantile"].astype(float),
            )
        )
        if (
            not endpoint_quantiles
            or len(quantiles) != len(actual_quantile_grid)
            or actual_quantile_grid != expected_quantile_grid
        ):
            raise ReportInputError("targeted quantiles do not contain the full requested grid")
    required_support = {
        "source_neuron",
        "source_index",
        "source_lag_frames",
        "context",
        "valid_fraction",
        "mean_achieved_gap_magnitude",
        "mean_ess_low",
        "mean_ess_high",
        "n_worms",
        "n_particles",
        "matrix_orientation",
    }
    if support.empty or required_support.difference(support.columns):
        raise ReportInputError("targeted support diagnostics have an incompatible schema")
    _numeric(
        support,
        (
            "source_index",
            "source_lag_frames",
            "valid_fraction",
            "mean_achieved_gap_magnitude",
            "mean_ess_low",
            "mean_ess_high",
            "n_worms",
            "n_particles",
        ),
        label="targeted_support",
    )
    if (
        not (support["n_worms"] == n_worms).all()
        or not (support["n_particles"] == 128).all()
        or not (support["matrix_orientation"] == ORIENTATION).all()
        or ((support["valid_fraction"] < 0) | (support["valid_fraction"] > 1)).any()
        or (support["mean_achieved_gap_magnitude"] < 0).any()
        or (support["mean_ess_low"] < 0).any()
        or (support["mean_ess_low"] > support["n_particles"]).any()
        or (support["mean_ess_high"] < 0).any()
        or (support["mean_ess_high"] > support["n_particles"]).any()
        or (support["source_index"] < 0).any()
        or (support["source_index"] >= n_neurons).any()
        or not np.equal(support["source_index"], np.floor(support["source_index"])).all()
    ):
        raise ReportInputError("targeted support diagnostics fail grain/range checks")
    support_source = support["source_index"].astype(int).to_numpy()
    if not np.array_equal(
        support["source_neuron"].astype(str).to_numpy(), neuron_array[support_source]
    ):
        raise ReportInputError("targeted support source names do not match atlas indices")
    required_support_keys = set(
        zip(
            cells["source_index"].astype(int),
            cells["source_lag_frames"].astype(int),
            cells["context"].astype(str),
        )
    )
    actual_support_keys = set(
        zip(
            support["source_index"].astype(int),
            support["source_lag_frames"].astype(int),
            support["context"].astype(str),
        )
    )
    if not required_support_keys.issubset(actual_support_keys):
        raise ReportInputError("targeted support diagnostics omit a selected source cell")

    screen_summary = summary.get("screen_vs_n128")
    if not isinstance(screen_summary, Mapping):
        raise ReportInputError("targeted summary omits screen-versus-N128 provenance")
    if screen.empty:
        if screen_summary.get("status") != "not_requested" or int(
            screen_summary.get("rows", -1)
        ) != 0:
            raise ReportInputError("empty targeted screen table disagrees with summary")
    else:
        required_screen = {
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
        }
        missing_screen = sorted(required_screen.difference(screen.columns))
        if missing_screen:
            raise ReportInputError(
                f"targeted screen consistency lacks columns {missing_screen}"
            )
        _numeric(
            screen,
            (
                "source_lag_frames",
                "horizon_frames",
                "screen_mean_normalized",
                "targeted_n128_mean_normalized",
                "targeted_minus_screen",
                "direction_agreement",
                "magnitude_agreement",
                "screen_valid_fraction",
                "targeted_valid_fraction",
            ),
            label="screen_consistency",
            allow_missing=True,
        )
        candidate_ids = set(cells["candidate_id"].astype(str))
        if (
            screen_summary.get("status") != "computed"
            or int(screen_summary.get("rows", -1)) != len(screen)
            or len(screen) != screen["candidate_id"].nunique()
            or set(screen["candidate_id"].astype(str)) != candidate_ids
        ):
            raise ReportInputError("targeted screen consistency grid disagrees with summary")
        finite_direction = screen["direction_agreement"].dropna().astype(float)
        for field in (
            "magnitude_agreement",
            "screen_valid_fraction",
            "targeted_valid_fraction",
        ):
            finite = screen[field].dropna().astype(float)
            if ((finite < 0) | (finite > 1)).any():
                raise ReportInputError(f"targeted screen {field} lies outside [0,1]")
        if ((finite_direction < 0) | (finite_direction > 1)).any():
            raise ReportInputError(
                "targeted screen direction agreement lies outside [0,1]"
            )
        observed_direction = (
            float(finite_direction.mean()) if len(finite_direction) else None
        )
        observed_magnitude = float(screen["magnitude_agreement"].median())
        reported_direction = screen_summary.get("direction_agreement_fraction")
        reported_magnitude = screen_summary.get("median_magnitude_agreement")
        if (
            (observed_direction is None and reported_direction is not None)
            or (
                observed_direction is not None
                and (
                    reported_direction is None
                    or not np.isclose(float(reported_direction), observed_direction)
                )
            )
            or reported_magnitude is None
            or not np.isclose(float(reported_magnitude), observed_magnitude)
        ):
            raise ReportInputError(
                "targeted screen summary statistics disagree with screen rows"
            )
        cell_metadata = (
            cells.drop_duplicates("candidate_id")
            .set_index("candidate_id")[["source_neuron", "target_neuron"]]
            .astype(str)
        )
        aligned = screen.set_index("candidate_id")
        if any(
            aligned.loc[candidate, field] != cell_metadata.loc[candidate, field]
            for candidate in candidate_ids
            for field in ("source_neuron", "target_neuron")
        ):
            raise ReportInputError("targeted screen neuron labels disagree with N128 cells")
        screen["pair_label"] = (
            screen["source_neuron"].astype(str)
            + " → "
            + screen["target_neuron"].astype(str)
        )

    selected_candidates = list(cells["candidate_id"].drop_duplicates())[
        :MAX_TARGETED_CANDIDATES
    ]
    selected = cells[cells["candidate_id"].isin(selected_candidates)].copy()
    screen_metric = (
        screen.set_index("candidate_id")["targeted_metric"].to_dict()
        if not screen.empty
        and {"candidate_id", "targeted_metric"}.issubset(screen.columns)
        else {}
    )
    primary_parts: list[pd.DataFrame] = []
    for candidate in selected_candidates:
        current = selected[selected["candidate_id"] == candidate]
        metric = str(screen_metric.get(candidate, "endpoint_mean"))
        matched = current[current["metric"] == metric]
        if matched.empty:
            matched = current[current["metric"] == "endpoint_mean"]
        primary_parts.append(matched)
    primary = pd.concat(primary_parts, ignore_index=True) if primary_parts else selected.head(0)
    primary["candidate_label"] = (
        "#"
        + primary["candidate_id"].astype(str).str.replace("targeted_", "", regex=False)
        + " "
        + primary["source_neuron"].astype(str)
        + "→"
        + primary["target_neuron"].astype(str)
        + " ℓ"
        + primary["source_lag_frames"].astype(int).astype(str)
        + "/h"
        + primary["horizon_frames"].astype(int).astype(str)
    )
    primary["contrast_label"] = primary["contrast"].str.replace("_", " − ")
    desired_metrics = {"endpoint_mean", "endpoint_log_sd", "endpoint_wasserstein1"}
    details = selected[selected["metric"].isin(desired_metrics)].copy().head(
        MAX_TARGETED_ROWS
    )
    details["pair_label"] = (
        details["source_neuron"].astype(str)
        + " → "
        + details["target_neuron"].astype(str)
    )
    qrows = quantiles[quantiles["candidate_id"].isin(selected_candidates)].copy()
    qrows = qrows.sort_values(
        ["candidate_id", "contrast", "endpoint_quantile"]
    ).head(MAX_QUANTILE_ROWS)
    srows = support.sort_values(
        ["valid_fraction", "source_neuron"], ascending=[True, True]
    ).head(MAX_SUPPORT_ROWS)
    return TargetedEvidence(
        root=root,
        manifest=manifest,
        protocol=protocol,
        validation=validation,
        summary=summary,
        cells=cells,
        quantiles=quantiles,
        support=support,
        screen=screen,
        primary_rows=_records(primary.head(MAX_TARGETED_ROWS)),
        detail_rows=_records(details),
        quantile_rows=_records(qrows),
        support_rows=_records(srows),
        checksums=checksums,
    )


def _validate_external(external_dir: Path, atlas: AtlasEvidence) -> ExternalEvidence:
    root = external_dir.resolve()
    checksums = _verify_bundle_checksums(root, EXTERNAL_REQUIRED)
    manifest = _read_json(root / "manifest.json")
    if (
        manifest.get("status") != "passed"
        or manifest.get("analysis_role") != "post_freeze_external_convergence_only"
        or not str(manifest.get("ranking_effect", "")).startswith("none")
        or manifest.get("internal_firewall_reverified_after_analysis") is not True
        or manifest.get("internal_prediction_ranking_inputs") != []
    ):
        raise ReportInputError("post-freeze external firewall contract failed")
    firewall = manifest.get("internal_firewall")
    expected_firewall = {
        "internal_manifest_sha256": sha256(atlas.root / "manifest.json"),
        "internal_protocol_sha256": sha256(atlas.root / "protocol.json"),
        "internal_validation_sha256": sha256(atlas.root / "validation.json"),
        "internal_atlas_matrices_sha256": sha256(
            atlas.root / "atlas_matrices.npz"
        ),
        "internal_checksums_sha256": sha256(atlas.root / "checksums.sha256"),
        "frozen_hypothesis_queue_sha256": sha256(
            atlas.root / "hypothesis_queue.csv"
        ),
    }
    if not isinstance(firewall, Mapping) or any(
        str(firewall.get(key)) != value for key, value in expected_firewall.items()
    ):
        raise ReportInputError("post-freeze external input is not this frozen atlas")
    input_ledger = _parse_checksum_ledger(
        root / "input_checksums.sha256", nested=True
    )
    expected_input: dict[str, str] = {
        "internal_atlas/atlas_matrices.npz": expected_firewall[
            "internal_atlas_matrices_sha256"
        ],
        "internal_atlas/checksums.sha256": expected_firewall[
            "internal_checksums_sha256"
        ],
        "internal_atlas/hypothesis_queue.csv": expected_firewall[
            "frozen_hypothesis_queue_sha256"
        ]
    }
    reference_hashes = manifest.get("reference_release_file_sha256")
    sbtg_hashes = manifest.get("sbtg_archive_provenance_sha256")
    if (
        not isinstance(reference_hashes, Mapping)
        or not reference_hashes
        or not isinstance(sbtg_hashes, Mapping)
        or not sbtg_hashes
    ):
        raise ReportInputError("post-freeze external provenance maps are missing")
    if set(str(value) for value in reference_hashes) != set(REFERENCE_RELEASE_FILES):
        raise ReportInputError("post-freeze reference-release inventory is not canonical")
    for relative, digest in reference_hashes.items():
        expected_input[f"reference_release/{relative}"] = str(digest)
    for name, digest in sbtg_hashes.items():
        expected_input[f"sbtg_archive/{name}"] = str(digest)
    if input_ledger != expected_input:
        raise ReportInputError("post-freeze input checksum ledger differs from manifest")
    reference_release = Path(str(manifest.get("reference_release", ""))).resolve()
    for relative, digest in reference_hashes.items():
        relative_path = Path(str(relative))
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
        ):
            raise ReportInputError("post-freeze reference provenance path is unsafe")
        path = reference_release / relative_path
        if not path.is_file() or sha256(path) != str(digest):
            raise ReportInputError("a post-freeze reference-release input changed")
    sbtg_archive = Path(str(manifest.get("sbtg_archive", ""))).resolve()
    if (
        not sbtg_archive.is_file()
        or sha256(sbtg_archive) != str(manifest.get("sbtg_archive_sha256", ""))
    ):
        raise ReportInputError("the post-freeze SBTG comparator changed")
    if set(str(value) for value in sbtg_hashes) != {
        sbtg_archive.name,
        "protocol.json",
        "validation.json",
        "checksums.sha256",
    }:
        raise ReportInputError("post-freeze SBTG provenance inventory is incomplete")
    for name, digest in sbtg_hashes.items():
        relative_path = Path(str(name))
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
        ):
            raise ReportInputError("post-freeze SBTG provenance path is unsafe")
        path = sbtg_archive.parent / relative_path
        if not path.is_file() or sha256(path) != str(digest):
            raise ReportInputError("a post-freeze SBTG provenance input changed")

    comparisons = _read_csv(
        root / "dashboard_external_summary.csv",
        label="dashboard_external_summary.csv",
    )
    lagmax = _read_csv(
        root / "neuromodulator_lagmax_inference.csv",
        label="neuromodulator_lagmax_inference.csv",
    )
    expected_rows = manifest.get("rows")
    if (
        not isinstance(expected_rows, Mapping)
        or len(comparisons) != int(expected_rows.get("dashboard_external_summary", -1))
        or len(lagmax) != int(expected_rows.get("neuromodulator_lagmax_inference", -1))
    ):
        raise ReportInputError("post-freeze CSV row counts differ from manifest")
    comparison_required = {
        "panel",
        "method",
        "channel",
        "context",
        "reference_or_network",
        "timing_label",
        "scope_or_grid",
        "auroc",
        "auprc",
        "continuous_spearman",
        "permutation_p",
        "bh_q",
        "comparison_note",
    }
    if comparisons.empty or comparison_required.difference(comparisons.columns):
        raise ReportInputError("post-freeze dashboard summary schema failed")
    _numeric(
        comparisons,
        (
            "auroc",
            "auprc",
            "continuous_spearman",
            "permutation_p",
            "bh_q",
        ),
        label="dashboard_external_summary",
        allow_missing=True,
    )
    for field in ("auroc", "auprc", "permutation_p", "bh_q"):
        finite = comparisons[field].dropna().astype(float)
        if ((finite < 0) | (finite > 1)).any():
            raise ReportInputError(f"post-freeze {field} lies outside [0,1]")
    finite_spearman = comparisons["continuous_spearman"].dropna().astype(float)
    if ((finite_spearman < -1) | (finite_spearman > 1)).any():
        raise ReportInputError("post-freeze continuous Spearman lies outside [-1,1]")
    panels = set(comparisons["panel"].astype(str))
    if not {
        "randi_cook_prespecified_flow_h1",
        "randi_cook_contextual_sbtg_shared54_lineage_mismatch",
        "neuromodulator_native_grid_lagmax",
    }.issubset(panels):
        raise ReportInputError("post-freeze report panels are incomplete")
    lag_required = {
        "method",
        "channel",
        "context",
        "network",
        "lag_grid",
        "best_lag_frames",
        "horizon_frames",
        "best_auroc",
        "max_lag_permutation_p",
        "max_lag_bh_q",
        "n_eligible_sources",
        "n_positive",
        "lag_semantics",
        "timing_comparability",
    }
    if lagmax.empty or lag_required.difference(lagmax.columns):
        raise ReportInputError("neuromodulator lag-max schema failed")
    _numeric(
        lagmax,
        (
            "best_lag_frames",
            "horizon_frames",
            "best_auroc",
            "max_lag_permutation_p",
            "max_lag_bh_q",
            "n_eligible_sources",
            "n_positive",
        ),
        label="neuromodulator_lagmax",
        allow_missing=True,
    )
    finite_q = lagmax["max_lag_bh_q"].dropna().astype(float)
    finite_auc = lagmax["best_auroc"].dropna().astype(float)
    finite_p = lagmax["max_lag_permutation_p"].dropna().astype(float)
    if (
        ((finite_q < 0) | (finite_q > 1)).any()
        or ((finite_auc < 0) | (finite_auc > 1)).any()
        or ((finite_p < 0) | (finite_p > 1)).any()
    ):
        raise ReportInputError("post-freeze lag-max AUROC/p/q metrics lie outside [0,1]")
    for field in ("best_lag_frames", "horizon_frames"):
        finite = lagmax[field].dropna().astype(float)
        if (finite <= 0).any() or not np.equal(finite, np.floor(finite)).all():
            raise ReportInputError(f"post-freeze lag-max {field} is not a positive integer")
    for field in ("n_eligible_sources", "n_positive"):
        values = lagmax[field].to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ReportInputError(f"post-freeze lag-max {field} is not a nonnegative integer")
    observed_networks = set(lagmax["network"].astype(str))
    if not set(EXPECTED_NEUROMODULATOR_SOURCES).issubset(observed_networks):
        raise ReportInputError("neuromodulator lag-max networks are incomplete")

    compare = comparisons[
        comparisons["panel"].isin(
            {
                "randi_cook_prespecified_flow_h1",
                "randi_cook_contextual_sbtg_shared54_lineage_mismatch",
            }
        )
    ].copy()
    compare = compare.head(MAX_EXTERNAL_ROWS)
    method_labels = {
        "progressive_bridge_smc": "Progressive SMC",
        "direct_importance": "Direct importance",
        "sbtg_current": "SBTG current",
        "sbtg_published": "SBTG published",
    }
    reference_labels = {
        "randi_wild_type": "Randi WT",
        "cook_struct_54": "Cook structure",
        "cook_chem_54": "Cook chemical",
        "cook_gap_54": "Cook gap",
    }
    compare["comparison_label"] = (
        compare["reference_or_network"].astype(str).map(
            lambda value: reference_labels.get(value, value)
        )
        + " · "
        + compare["method"].astype(str).map(
            lambda value: method_labels.get(value, value)
        )
    )
    compare["comparison_family"] = np.where(
        compare["method"].astype(str).isin(("direct_importance", "progressive_bridge_smc")),
        "Flow",
        "SBTG",
    )
    lag_chart = lagmax[
        (lagmax["lag_grid"] == "native_method_grid")
        & (lagmax["channel"] == "endpoint_mean")
        & (lagmax["context"] == "state_average")
        & lagmax["method"].isin(("progressive_bridge_smc", "direct_importance"))
    ].copy()
    if lag_chart.empty:
        lag_chart = lagmax[lagmax["lag_grid"] == "native_method_grid"].copy()
    lag_chart = lag_chart.sort_values(
        ["network", "method", "channel", "context"]
    ).head(MAX_LAGMAX_ROWS)
    network_labels = {
        "monoamine_all": "All monoamines",
        "monoamine_dopamine": "Dopamine",
        "monoamine_serotonin": "Serotonin",
        "monoamine_tyramine": "Tyramine",
        "monoamine_octopamine": "Octopamine",
        "neuropeptide_all": "Neuropeptides",
        "neuromodulator_union": "Union",
    }
    lag_chart["network_method_label"] = (
        lag_chart["network"].astype(str).map(
            lambda value: network_labels.get(value, value)
        )
        + " · "
        + lag_chart["method"].astype(str).map(
            lambda value: "SMC"
            if value == "progressive_bridge_smc"
            else ("Direct" if value == "direct_importance" else value)
        )
    )
    return ExternalEvidence(
        root=root,
        manifest=manifest,
        comparisons=comparisons,
        lagmax=lagmax,
        comparison_rows=_records(compare),
        lagmax_rows=_records(lag_chart),
        checksums=checksums,
    )


def _validate_root_documents(atlas_root: Path) -> list[dict[str, Any]]:
    root = atlas_root.resolve()
    rows: list[dict[str, Any]] = []
    contents: dict[str, str] = {}
    for name in ROOT_DOCUMENTS:
        path = root / name
        if not path.is_file() or path.stat().st_size < 128:
            raise ReportInputError(f"atlas root document is missing/incomplete: {name}")
        text = path.read_text()
        contents[name] = text.lower()
        rows.append(
            {
                "document": name,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "status": "hashed at report build (not upstream-frozen)",
            }
        )
    combined = "\n".join(contents.values())
    boundaries = {
        "noncausal": "not causal" in combined,
        "physical_delay": "physical" in combined and "delay" in combined,
        "binary_stimulus": "binary" in combined and "stimulus" in combined,
        "chemical_boundary": any(
            phrase in combined
            for phrase in (
                "not chemically conditioned",
                "not chemical-conditioned",
                "never chemical-conditioned",
            )
        ),
        "orientation": "target rows" in combined and "source columns" in combined,
    }
    if not all(boundaries.values()):
        missing = sorted(key for key, value in boundaries.items() if not value)
        raise ReportInputError(f"root protocol/audits omit claim boundaries: {missing}")
    return rows


def _source(
    *,
    source_id: str,
    label: str,
    path: str,
    command: str,
    executed_at: str,
    description: str,
    tables_used: Sequence[str],
    filters: Sequence[str],
    metric_definitions: Sequence[str],
    artifact_path: str | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    if (artifact_path is None) != (dataset is None):
        raise ValueError("artifact_path and dataset must be supplied together")
    if dataset is not None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", dataset):
            raise ValueError("snapshot dataset names must be safe identifiers")
        escaped_artifact = str(artifact_path).replace("'", "''")
        sql = (
            "SELECT UNNEST(snapshot.datasets."
            f"{dataset}, recursive := true) "
            f"FROM read_json_auto('{escaped_artifact}')"
        )
        description = (
            f"{description} The SQL reads the exact bounded snapshot rows materialized "
            f"by: {command}"
        )
        used = [str(artifact_path), *tables_used]
    else:
        escaped = path.replace("'", "''")
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            sql = f"SELECT * FROM read_csv_auto('{escaped}', header = true)"
        elif suffix == ".json":
            sql = f"SELECT * FROM read_json_auto('{escaped}')"
        else:
            sql = f"SELECT content FROM read_text('{escaped}')"
        used = list(tables_used)
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb_local_file",
            "language": "sql",
            "query": sql,
            "sql": sql,
            "description": description,
            "executed_at": executed_at,
            "tables_used": list(dict.fromkeys(used)),
            "filters": list(filters),
            "metric_definitions": list(metric_definitions),
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "not estimable"
    return f"{float(value):.{digits}f}"


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows were available under the declared filter._"
    header = "| " + " | ".join(column.replace("_", " ") for column in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = _json_safe(row.get(column))
            if isinstance(value, float):
                text = _fmt(value)
            elif value is None:
                text = "—"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join((header, separator, *body))


def _prepare_output(output_dir: Path, overwrite: bool) -> Path:
    output = output_dir.resolve()
    if output.exists():
        if not output.is_dir():
            raise ReportInputError("report output path exists and is not a directory")
        unexpected = [path for path in output.iterdir() if path.name not in OUTPUT_FILES]
        existing = [path for path in output.iterdir() if path.name in OUTPUT_FILES]
        if existing and not overwrite:
            raise FileExistsError(f"report outputs already exist in {output}")
        if overwrite and unexpected:
            raise ReportInputError(
                "overwrite refused because report output contains unrelated paths"
            )
        if overwrite:
            for path in existing:
                if not path.is_file():
                    raise ReportInputError("a report output path is not a regular file")
            for path in existing:
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    return output


def _validate_output_location(
    output_dir: Path,
    *,
    immutable_input_roots: Sequence[Path],
    document_root: Path,
) -> None:
    """Reject any output location that could overwrite a validated input.

    The intended ``technical_report/`` directory may live beneath the atlas
    document root, but it must remain disjoint from the canonical, targeted,
    and external bundles.  Validation deliberately runs before overwrite
    cleanup.
    """

    output = output_dir.resolve()
    for candidate in immutable_input_roots:
        root = candidate.resolve()
        if output == root or output in root.parents or root in output.parents:
            raise ReportInputError(
                "report output must be disjoint from every validated data bundle"
            )
    docs = document_root.resolve()
    if output == docs or output in docs.parents:
        raise ReportInputError(
            "report output must not equal or contain the validated document root"
        )


def build_prediction_atlas_report(
    atlas_dir: Path,
    targeted_analysis_dir: Path,
    postfreeze_external_dir: Path,
    atlas_root: Path,
    output_dir: Path,
    *,
    provenance_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate canonical inputs and write the report artifact plus Markdown."""

    provenance = (
        Path.cwd().resolve() if provenance_root is None else provenance_root.resolve()
    )
    atlas = _validate_atlas(Path(atlas_dir))
    targeted = _validate_targeted(
        Path(targeted_analysis_dir),
        n_worms=atlas.n_worms,
        neurons=atlas.neurons,
        atlas_dir=atlas.root,
        atlas_checksums=atlas.checksums,
        provenance_root=provenance,
    )
    external = _validate_external(Path(postfreeze_external_dir), atlas)
    root_rows = _validate_root_documents(Path(atlas_root))
    requested_output = Path(output_dir).resolve()
    _safe_relative(requested_output, provenance)
    _validate_output_location(
        requested_output,
        immutable_input_roots=(atlas.root, targeted.root, external.root),
        document_root=Path(atlas_root),
    )
    output = _prepare_output(requested_output, overwrite)
    for path in (atlas.root, targeted.root, external.root, Path(atlas_root).resolve(), output):
        _safe_relative(path, provenance)

    generated = datetime.now(timezone.utc).isoformat()
    rel_atlas = _safe_relative(atlas.root, provenance)
    rel_targeted = _safe_relative(targeted.root, provenance)
    rel_external = _safe_relative(external.root, provenance)
    rel_root = _safe_relative(Path(atlas_root), provenance)
    rel_output = _safe_relative(output, provenance)
    arguments = [
        "python",
        "-m",
        "compatibility_neural_benchmark.prediction_atlas_report",
        "--atlas-dir",
        rel_atlas,
        "--targeted-analysis-dir",
        rel_targeted,
        "--postfreeze-external-dir",
        rel_external,
        "--atlas-root",
        rel_root,
        "--output-dir",
        rel_output,
        "--provenance-root",
        ".",
    ]
    command = " ".join(shlex.quote(value) for value in arguments)
    artifact_source_path = f"{rel_output}/artifact.json"

    source_paths = {
        "atlas_dashboard": f"{rel_atlas}/dashboard_snapshot.json",
        "atlas_validation": f"{rel_atlas}/validation.json",
        "atlas_protocol": f"{rel_atlas}/protocol.json",
        "atlas_queue": f"{rel_atlas}/hypothesis_queue.csv",
        "atlas_stimulus_composition": f"{rel_atlas}/stimulus_composition.csv",
        "targeted_summary": f"{rel_targeted}/summary.json",
        "targeted_cells": f"{rel_targeted}/targeted_cells.csv",
        "targeted_quantiles": f"{rel_targeted}/targeted_quantile_shifts.csv",
        "targeted_support": f"{rel_targeted}/support_diagnostics.csv",
        "targeted_screen": f"{rel_targeted}/screen_consistency.csv",
        "external_comparison": f"{rel_external}/dashboard_external_summary.csv",
        "external_lagmax": f"{rel_external}/neuromodulator_lagmax_inference.csv",
        "atlas_ledger": f"{rel_atlas}/checksums.sha256",
        "targeted_ledger": f"{rel_targeted}/checksums.sha256",
        "external_ledger": f"{rel_external}/checksums.sha256",
        "root_protocol": f"{rel_root}/RUN_PROTOCOL.md",
        "root_code_audit": f"{rel_root}/CODE_AND_ESTIMAND_AUDIT.md",
    }
    sources = [
        _source(
            source_id="report_summary_source",
            label="Report headline summary",
            path=source_paths["atlas_dashboard"],
            command=command,
            executed_at=generated,
            description="Bounded headline cohort, support, targeted-sensitivity, and lag-max counts.",
            tables_used=[
                source_paths["atlas_dashboard"],
                source_paths["atlas_validation"],
                source_paths["atlas_queue"],
                source_paths["targeted_summary"],
                source_paths["external_lagmax"],
            ],
            filters=["reviewed canonical inputs only"],
            metric_definitions=[
                "Counts and mean support are copied or recomputed from checksum-verified canonical summaries."
            ],
            artifact_path=artifact_source_path,
            dataset="report_summary",
        ),
        _source(
            source_id="atlas_dashboard_source",
            label="Reviewed atlas summary",
            path=source_paths["atlas_dashboard"],
            command=command,
            executed_at=str(atlas.validation.get("created_utc", generated)),
            description="Reviewed cohort, support, sampler, context, and candidate snapshot.",
            tables_used=[source_paths["atlas_dashboard"]],
            filters=["17 held-out worms", "54-neuron target-row/source-column axis"],
            metric_definitions=[
                "Mean validity is the fraction of sampled events satisfying the declared repaired-law support gate.",
            ],
        ),
        _source(
            source_id="internal_narrative_source",
            label="Internal support and stability summary",
            path=source_paths["atlas_validation"],
            command=command,
            executed_at=str(atlas.validation.get("created_utc", generated)),
            description="Exact scalar summary used by the internal-support narrative.",
            tables_used=[
                source_paths["atlas_validation"],
                source_paths["atlas_dashboard"],
                source_paths["atlas_queue"],
            ],
            filters=["complete primary endpoint-mean onset-minus-baseline timing grid"],
            metric_definitions=["Agreement range and median use all finite 4×6 timing-grid Spearman values."],
            artifact_path=artifact_source_path,
            dataset="internal_summary",
        ),
        _source(
            source_id="atlas_validation_source",
            label="Internal support and cross-sampler audit",
            path=source_paths["atlas_validation"],
            command=command,
            executed_at=str(atlas.validation.get("created_utc", generated)),
            description="Validated direct-versus-progressive matrix agreement over the full lag/horizon grid.",
            tables_used=[
                source_paths["atlas_validation"],
                source_paths["atlas_dashboard"],
            ],
            filters=["endpoint mean", "onset minus baseline", "all off-diagonal directed cells"],
            metric_definitions=[
                "Cross-sampler Spearman correlates direct-importance and progressive-bridge matrices at the same lag, horizon, channel, and context.",
            ],
            artifact_path=artifact_source_path,
            dataset="internal_stability",
        ),
        _source(
            source_id="atlas_queue_source",
            label="Frozen internal hypothesis queue",
            path=source_paths["atlas_queue"],
            command=command,
            executed_at=str(atlas.validation.get("created_utc", generated)),
            description="Atlas-only ranking frozen before any external-reference analysis.",
            tables_used=[source_paths["atlas_queue"]],
            filters=[f"first {len(atlas.candidate_rows)} bounded rows", "valid fraction at least 0.50"],
            metric_definitions=[
                "Evidence score combines model effect, worm interval, support, seed stability, and cross-sampler agreement.",
            ],
            artifact_path=artifact_source_path,
            dataset="candidate_rows",
        ),
        _source(
            source_id="atlas_protocol_source",
            label="Atlas metric and estimand protocol",
            path=source_paths["atlas_protocol"],
            command=command,
            executed_at=str(atlas.validation.get("created_utc", generated)),
            description="Canonical timing, effect, normalization, chemistry, inference, and claim definitions.",
            tables_used=[
                source_paths["atlas_protocol"],
                source_paths["atlas_dashboard"],
                f"{rel_targeted}/protocol.json",
                source_paths["root_protocol"],
                *(
                    [source_paths["atlas_stimulus_composition"]]
                    if atlas.stimulus_rows
                    else []
                ),
            ],
            filters=["reviewed neural prediction atlas v1"],
            metric_definitions=[
                "Source-to-cut is lag/fps; forecast is horizon/fps; source-to-readout is their sum.",
                "Matrices are target rows by source columns after exactly one transpose.",
                "Stimulus composition is reconstructed on the exact frozen half-open schedule grid.",
            ],
            artifact_path=artifact_source_path,
            dataset="definitions",
        ),
        _source(
            source_id="targeted_summary_source",
            label="Selected N128 three-arm summary",
            path=source_paths["targeted_summary"],
            command=command,
            executed_at=str(targeted.validation.get("created_utc", generated)),
            description="Worm-level review of selected progressive-bridge low/high/factual model arms.",
            tables_used=[
                source_paths["targeted_summary"],
                source_paths["targeted_support"],
                source_paths["targeted_screen"],
            ],
            filters=["N=128 particles", "selected frozen internal cells", "seeds averaged within worm"],
            metric_definitions=[
                "Each event contrast is divided by max(abs achieved high-low source gap, 0.10).",
            ],
        ),
        _source(
            source_id="targeted_narrative_source",
            label="Selected N128 narrative summary",
            path=source_paths["targeted_summary"],
            command=command,
            executed_at=str(targeted.validation.get("created_utc", generated)),
            description="Exact scalar summary used by the N128 three-arm narrative.",
            tables_used=[
                source_paths["targeted_summary"],
                f"{rel_targeted}/protocol.json",
                source_paths["targeted_screen"],
            ],
            filters=["selected frozen atlas cells", "N=128 particles"],
            metric_definitions=[
                "Screen direction and magnitude agreement are descriptive and selection-conditioned."
            ],
            artifact_path=artifact_source_path,
            dataset="targeted_narrative",
        ),
        _source(
            source_id="targeted_primary_source",
            label="Selected N128 primary three-arm contrasts",
            path=source_paths["targeted_cells"],
            command=command,
            executed_at=str(targeted.validation.get("created_utc", generated)),
            description="High-low, high-factual, and low-factual signed and Wasserstein summaries.",
            tables_used=[source_paths["targeted_cells"], source_paths["targeted_quantiles"]],
            filters=["one frozen screen-selected metric per selected candidate", "three arm contrasts"],
            metric_definitions=[
                "Factual is a free conditional-flow rollout from observed history, not an observed counterfactual future.",
                "Endpoint Wasserstein-1 is unsigned within a state; a phase difference can be signed.",
            ],
            artifact_path=artifact_source_path,
            dataset="targeted_primary",
        ),
        _source(
            source_id="targeted_details_source",
            label="Selected N128 distributional detail rows",
            path=source_paths["targeted_cells"],
            command=command,
            executed_at=str(targeted.validation.get("created_utc", generated)),
            description="Bounded endpoint mean, log-SD, and Wasserstein-1 rows for all three arm contrasts.",
            tables_used=[source_paths["targeted_cells"]],
            filters=[f"all {len(targeted.detail_rows)} bounded distributional metric rows"],
            metric_definitions=[
                "Signed metrics report left-arm minus right-arm; Wasserstein-1 is unsigned within a state, while a between-phase difference can be signed."
            ],
            artifact_path=artifact_source_path,
            dataset="targeted_details",
        ),
        _source(
            source_id="targeted_quantiles_source",
            label="Selected N128 endpoint quantile shifts",
            path=source_paths["targeted_quantiles"],
            command=command,
            executed_at=str(targeted.validation.get("created_utc", generated)),
            description="Bounded low/high/factual endpoint quantile contrasts.",
            tables_used=[source_paths["targeted_quantiles"]],
            filters=[f"all {len(targeted.quantile_rows)} bounded quantile rows"],
            metric_definitions=["Each quantile shift is normalized event-wise by achieved source-gap magnitude."],
            artifact_path=artifact_source_path,
            dataset="targeted_quantiles",
        ),
        _source(
            source_id="targeted_support_source",
            label="Selected N128 support diagnostics",
            path=source_paths["targeted_support"],
            command=command,
            executed_at=str(targeted.validation.get("created_utc", generated)),
            description="Bounded low/high ESS, achieved-gap, and validity diagnostics.",
            tables_used=[source_paths["targeted_support"]],
            filters=[f"lowest-support {len(targeted.support_rows)} rows"],
            metric_definitions=["Validity conservatively uses joint low/high progressive-bridge support."],
            artifact_path=artifact_source_path,
            dataset="targeted_support",
        ),
        _source(
            source_id="targeted_screen_source",
            label="Frozen screen versus selected N128",
            path=source_paths["targeted_screen"],
            command=command,
            executed_at=str(targeted.validation.get("created_utc", generated)),
            description="Selection-conditioned screen-versus-N128 consistency rows.",
            tables_used=[source_paths["targeted_screen"]],
            filters=[f"first {min(len(targeted.screen), MAX_SCREEN_ROWS)} bounded rows"],
            metric_definitions=[
                "Direction and magnitude agreement compare the frozen screen with N128 high-low; direct-screen rows also change sampler family."
            ],
            artifact_path=artifact_source_path,
            dataset="targeted_screen",
        ),
        _source(
            source_id="external_comparison_source",
            label="Post-freeze Randi, Cook, and SBTG comparison",
            path=source_paths["external_comparison"],
            command=command,
            executed_at=str(external.manifest.get("created_utc", generated)),
            description="Bounded post-freeze convergence metrics that never altered internal ranking.",
            tables_used=[source_paths["external_comparison"]],
            filters=[
                "prespecified flow H1 support-qualified rows",
                "contextual SBTG shared-54 all-estimated rows",
            ],
            metric_definitions=[
                "AUROC compares absolute model-effect magnitude with binary response or edge existence.",
                "Continuous Spearman is reference-specific and is not interchangeable with AUROC.",
            ],
            artifact_path=artifact_source_path,
            dataset="external_comparisons",
        ),
        _source(
            source_id="external_lagmax_source",
            label="Post-freeze Bentley neuromodulator lag-max analysis",
            path=source_paths["external_lagmax"],
            command=command,
            executed_at=str(external.manifest.get("created_utc", generated)),
            description="Source-preserving max-lag permutation analysis with global BH correction.",
            tables_used=[source_paths["external_lagmax"]],
            filters=["native method lag grid", "eligible neuromodulator source columns"],
            metric_definitions=[
                "Best AUROC is selected over the declared lag grid inside each permutation-calibrated test.",
                "Bentley edges encode receptor/pathway availability, not measured neural activity or transmission delay.",
            ],
            artifact_path=artifact_source_path,
            dataset="neuromod_lagmax",
        ),
        _source(
            source_id="root_audit_source",
            label="Run protocol and code/estimand audits",
            path=source_paths["root_code_audit"],
            command=command,
            executed_at=generated,
            description="Human-readable protocol, input-data audit, and code/estimand audit; hashes are captured at report build time.",
            tables_used=[f"{rel_root}/{name}" for name in ROOT_DOCUMENTS],
            filters=["documents read and SHA-256 hashed before report build"],
            metric_definitions=[],
        ),
        _source(
            source_id="provenance_source",
            label="Verified report-input inventories",
            path=source_paths["atlas_ledger"],
            command=command,
            executed_at=generated,
            description="Verified bundle checksum inventories plus contemporaneous SHA-256 hashes for root technical documents.",
            tables_used=[
                source_paths["atlas_ledger"],
                source_paths["targeted_ledger"],
                source_paths["external_ledger"],
                *[f"{rel_root}/{name}" for name in ROOT_DOCUMENTS],
            ],
            filters=["all ledger entries rehashed before any report row was constructed"],
            metric_definitions=["SHA-256 equality is exact; any mismatch aborts the build."],
            artifact_path=artifact_source_path,
            dataset="provenance",
        ),
    ]

    support_summary = atlas.dashboard["support_summary"]
    targeted_counts = targeted.summary.get("evidence_label_counts", {})
    finite_q = external.lagmax["max_lag_bh_q"].dropna().astype(float)
    lagmax_significant = int((finite_q <= 0.05).sum())
    agreement_values = np.asarray(
        [
            row["direct_vs_progressive_spearman"]
            for row in atlas.stability_rows
            if row.get("direct_vs_progressive_spearman") is not None
        ],
        dtype=float,
    )
    median_agreement = float(np.median(agreement_values))
    minimum_agreement = float(np.min(agreement_values))
    maximum_agreement = float(np.max(agreement_values))
    progressive_genealogy_pass = support_summary.get(
        "progressive_genealogy_rows_passing_f_0_10_strong_gate"
    )
    stimulus_crossing_rows = len(atlas.stimulus_rows)
    report_summary = [
        {
            "n_worms": atlas.n_worms,
            "n_neurons": atlas.n_neurons,
            "mean_valid_fraction": float(support_summary["mean_valid_fraction"]),
            "retained_candidates": len(atlas.queue),
            "targeted_candidates": int(targeted.summary["n_candidates"]),
            "targeted_consistent_rows": int(
                targeted_counts.get("model_relative_consistent", 0)
            ),
            "targeted_support_limited_rows": int(
                targeted_counts.get("unsupported_model_output", 0)
            ),
            "lagmax_tests": int(len(finite_q)),
            "lagmax_q_le_0_05": lagmax_significant,
            "median_direct_vs_progressive_spearman": median_agreement,
            "minimum_direct_vs_progressive_spearman": minimum_agreement,
            "maximum_direct_vs_progressive_spearman": maximum_agreement,
        }
    ]
    internal_summary = [
        {
            "retained_candidates": len(atlas.queue),
            "n_worms": atlas.n_worms,
            "mean_valid_fraction": float(support_summary["mean_valid_fraction"]),
            "median_direct_vs_progressive_spearman": median_agreement,
            "minimum_direct_vs_progressive_spearman": minimum_agreement,
            "maximum_direct_vs_progressive_spearman": maximum_agreement,
            "claim_scope": "model-relative, noncausal, binary-stimulus conditioned, not a physical delay",
            "progressive_genealogy_strong_gate_pass_rows": progressive_genealogy_pass,
            "displayed_stimulus_boundary_crossing_rows": stimulus_crossing_rows,
        }
    ]
    screen_summary_values = targeted.summary["screen_vs_n128"]
    targeted_narrative = [
        {
            "n_candidates": int(targeted.summary["n_candidates"]),
            "n_worms": atlas.n_worms,
            "n_particles": 128,
            "screen_status": str(screen_summary_values.get("status")),
            "screen_direction_agreement_fraction": screen_summary_values.get(
                "direction_agreement_fraction"
            ),
            "screen_median_magnitude_agreement": screen_summary_values.get(
                "median_magnitude_agreement"
            ),
            "normalization": "event contrast / max(abs achieved high-low gap, 0.10)",
            "claim_scope": "model-relative sensitivity, not experiment or physical delay",
        }
    ]
    definitions = [
        {
            "term": "Cohort",
            "definition": f"{atlas.n_worms} held-out worms and {atlas.n_neurons} aligned head-neuron classes at {atlas.fps:g} Hz.",
        },
        {
            "term": "Matrix orientation",
            "definition": "Target neurons are rows; repaired source neurons are columns.",
        },
        {
            "term": "High-low effect",
            "definition": "Future response under high-source repaired history minus low-source repaired history, divided event-wise by max(|achieved high-low source gap|, 0.10).",
        },
        {
            "term": "Direct importance",
            "definition": "Reweights a shared natural conditional-flow path bank; efficient but vulnerable to low effective sample size.",
        },
        {
            "term": "Progressive bridge SMC",
            "definition": "Introduces the repaired source constraint across the prefix with tempering, branching, and resampling before free rollout.",
        },
        {
            "term": "Factual arm",
            "definition": "Free rollout from observed history under the fitted flow; a model baseline, not ground-truth counterfactual activity.",
        },
        {
            "term": "Distribution metrics",
            "definition": "Endpoint log-SD tracks relative scale change; endpoint Wasserstein-1 tracks full one-dimensional marginal displacement.",
        },
        {
            "term": "Timing",
            "definition": "Source-to-cut lag, forecast horizon, and their sum to readout are distinct; none is a physical transmission delay.",
        },
        {
            "term": "Stimulus identity",
            "definition": "The generator conditions on binary any-stimulus history; chemical panels only stratify events observed after fitting.",
        },
        {
            "term": "Stimulus-window composition",
            "definition": (
                "The exact per-worm table reconstructs binary stimulus occupancy in the "
                "source window, at the cut, across cut+1…cut+h, and at cut+h from frozen "
                "half-open schedules; boundary-crossing panels are not fixed-stimulus delays."
            ),
        },
        {
            "term": "Progressive-SMC genealogy",
            "definition": (
                "Raw diagnostic validity is preserved. Strong promotion separately requires "
                "at least 80% validity after requiring both arms to retain at least 10% "
                "distinct ancestors per episode; 20% is reporting-only sensitivity."
            ),
        },
        {
            "term": "Inference unit",
            "definition": "Worm; events are reduced within worm and model seeds are averaged within worm before bootstrap intervals.",
        },
    ]
    provenance_rows = [
        {
            "artifact": f"atlas/{name}",
            "status": "checksum verified",
            "sha256": atlas.checksums[name],
        }
        for name in ATLAS_REQUIRED
        if name != "checksums.sha256"
    ]
    provenance_rows.extend(
        {
            "artifact": f"targeted/{name}",
            "status": "checksum verified",
            "sha256": targeted.checksums[name],
        }
        for name in TARGETED_REQUIRED
        if name != "checksums.sha256"
    )
    provenance_rows.extend(
        {
            "artifact": f"external/{name}",
            "status": "checksum verified",
            "sha256": external.checksums[name],
        }
        for name in EXTERNAL_REQUIRED
        if name != "checksums.sha256"
    )
    provenance_rows.extend(root_rows)

    datasets = {
        "report_summary": report_summary,
        "internal_summary": internal_summary,
        "targeted_narrative": targeted_narrative,
        "internal_stability": atlas.stability_rows,
        "context_stability": atlas.context_rows,
        "stimulus_composition": atlas.stimulus_rows,
        "candidate_rows": atlas.candidate_rows,
        "targeted_primary": targeted.primary_rows,
        "targeted_details": targeted.detail_rows,
        "targeted_quantiles": targeted.quantile_rows,
        "targeted_support": targeted.support_rows,
        "targeted_screen": _records(targeted.screen.head(MAX_SCREEN_ROWS)),
        "external_comparisons": external.comparison_rows,
        "neuromod_lagmax": external.lagmax_rows,
        "definitions": definitions,
        "provenance": provenance_rows,
    }
    cards = [
        {
            "id": "cohort_card",
            "dataset": "report_summary",
            "sourceId": "report_summary_source",
            "description": "Held-out worms are the independent statistical units.",
            "metrics": [
                {"label": "Held-out worms", "field": "n_worms", "format": "number"},
            ],
        },
        {
            "id": "neurons_card",
            "dataset": "report_summary",
            "sourceId": "report_summary_source",
            "description": "Aligned neuron classes on both axes of each dense directed matrix.",
            "metrics": [
                {"label": "Neuron classes", "field": "n_neurons", "format": "number"},
            ],
        },
        {
            "id": "support_card",
            "dataset": "report_summary",
            "sourceId": "report_summary_source",
            "description": "Mean source/context/lag support validity across the reviewed atlas.",
            "metrics": [
                {"label": "Mean valid fraction", "field": "mean_valid_fraction", "format": "percent"},
            ],
        },
        {
            "id": "targeted_card",
            "dataset": "report_summary",
            "sourceId": "report_summary_source",
            "description": "Selected cells evaluated with N128 low, high, and factual model arms.",
            "metrics": [
                {"label": "N128 selected cells", "field": "targeted_candidates", "format": "number"},
            ],
        },
        {
            "id": "targeted_consistency_card",
            "dataset": "report_summary",
            "sourceId": "report_summary_source",
            "description": "N128 candidate × contrast × metric rows meeting the internal model-relative consistency label; not experimental confirmations.",
            "metrics": [
                {"label": "Internally consistent metric rows", "field": "targeted_consistent_rows", "format": "number"},
            ],
        },
        {
            "id": "lagmax_card",
            "dataset": "report_summary",
            "sourceId": "report_summary_source",
            "description": "Post-freeze lag-max tests passing the global BH q≤0.05 threshold; this is not a physical-delay result.",
            "metrics": [
                {"label": "Lag-max q≤0.05", "field": "lagmax_q_le_0_05", "format": "number"},
                {"label": "Evaluable lag-max tests", "field": "lagmax_tests", "format": "number"},
            ],
        },
    ]

    charts: list[dict[str, Any]] = [
        {
            "id": "internal_stability_chart",
            "title": "Direct-versus-progressive matrix agreement",
            "subtitle": "Endpoint-mean onset-minus-baseline effects across the complete source-lag × forecast-horizon grid.",
            "intent": "relationship",
            "question": "At which lag/horizon cells do the two finite-particle estimators preserve similar directed matrix rankings?",
            "rationale": "A heatmap preserves the full 4×6 timing grid and keeps source-to-cut separate from forecast horizon.",
            "type": "heatmap",
            "dataset": "internal_stability",
            "sourceId": "atlas_validation_source",
            "encodings": {
                "x": {"field": "source_lag_label", "type": "ordinal", "label": "Source-to-cut lag"},
                "y": {"field": "direct_vs_progressive_spearman", "type": "quantitative", "label": "Matrix Spearman"},
                "color": {"field": "forecast_horizon_label", "type": "ordinal", "label": "Forecast horizon"},
                "tooltip": [
                    {"field": "source_to_readout_seconds", "type": "quantitative", "label": "Source-to-readout seconds"},
                    {"field": "n_worms", "type": "quantitative", "label": "Worms"},
                ],
            },
            "palette": {"kind": "diverging", "midpoint": 0},
            "layout": "full",
        }
    ]
    primary_metrics = {
        str(row.get("metric")) for row in targeted.primary_rows if row.get("metric")
    }
    if len(targeted.primary_rows) >= 4 and len(primary_metrics) == 1:
        primary_metric = next(iter(primary_metrics))
        charts.append(
            {
                "id": "targeted_three_arm_chart",
                "title": "Selected N128 three-arm model contrasts",
                "subtitle": f"One common response estimand ({primary_metric}) across high−low, high−factual, and low−factual arms; worm-level means with support retained.",
                "intent": "comparison",
                "question": "Do selected effects persist, and where does the factual rollout sit relative to the repaired low/high regimes?",
                "rationale": "A grouped horizontal comparison accommodates neuron-pair labels and preserves contrast identity.",
                "type": "horizontalBar",
                "dataset": "targeted_primary",
                "sourceId": "targeted_primary_source",
                "encodings": {
                    "x": {"field": "candidate_label", "type": "nominal", "label": "Source → target"},
                    "y": {"field": "mean_normalized", "type": "quantitative", "label": "Normalized effect"},
                    "color": {"field": "contrast_label", "type": "nominal", "label": "Arm contrast"},
                    "tooltip": [
                        {"field": "candidate_id", "type": "nominal", "label": "Candidate"},
                        {"field": "context", "type": "nominal", "label": "Context"},
                        {"field": "metric", "type": "nominal", "label": "Metric"},
                        {"field": "ci_2_5", "type": "quantitative", "label": "CI 2.5%"},
                        {"field": "ci_97_5", "type": "quantitative", "label": "CI 97.5%"},
                        {"field": "valid_fraction", "type": "quantitative", "label": "Valid fraction", "format": "percent"},
                    ],
                },
                "palette": {"kind": "categorical"},
                "layout": "full",
            }
        )
    if len(external.comparison_rows) >= 4:
        charts.append(
            {
                "id": "external_comparison_chart",
                "title": "Post-freeze Randi and Cook binary correspondence",
                "subtitle": "AUROC and prevalence-dependent AUPRC; compare like references/scopes only. Flow and SBTG timing/training lineages differ.",
                "intent": "relationship",
                "question": "How do frozen flow effects and contextual SBTG matrices align with Randi/Cook binary labels?",
                "rationale": "A compact AUROC-versus-AUPRC map avoids implying a single total ordering and keeps method/reference identity in audit tooltips.",
                "type": "scatter",
                "dataset": "external_comparisons",
                "sourceId": "external_comparison_source",
                "encodings": {
                    "x": {"field": "auroc", "type": "quantitative", "label": "AUROC"},
                    "y": {"field": "auprc", "type": "quantitative", "label": "AUPRC"},
                    "color": {"field": "comparison_family", "type": "nominal", "label": "Estimator family"},
                    "tooltip": [
                        {"field": "reference_or_network", "type": "text", "label": "Reference"},
                        {"field": "method", "type": "text", "label": "Method"},
                        {"field": "scope_or_grid", "type": "text", "label": "Eligible scope"},
                        {"field": "continuous_spearman", "type": "quantitative", "label": "Continuous Spearman"},
                        {"field": "timing_label", "type": "text", "label": "Timing"},
                    ],
                },
                "layout": "full",
            }
        )
    if len(external.lagmax_rows) >= 4:
        charts.append(
            {
                "id": "neuromod_lagmax_chart",
                "title": "Neuromodulator lag-grid maximum correspondence",
                "subtitle": "Observed maximum AUROC; p-values use a source-preserving max-lag null. Receptor/pathway availability is not activity or delay.",
                "intent": "comparison",
                "question": "Which neuromodulator networks show the largest post-freeze correspondence after accounting for lag selection?",
                "rationale": "A category comparison shows network/method magnitude while tooltips retain q-values and selected lag.",
                "type": "horizontalBar",
                "dataset": "neuromod_lagmax",
                "sourceId": "external_lagmax_source",
                "encodings": {
                    "x": {"field": "network_method_label", "type": "nominal", "label": "Network · method"},
                    "y": {"field": "best_auroc", "type": "quantitative", "label": "Max-lag AUROC"},
                    "color": {"field": "method", "type": "nominal", "label": "Method"},
                    "tooltip": [
                        {"field": "best_lag_frames", "type": "quantitative", "label": "Selected lag frames"},
                        {"field": "max_lag_bh_q", "type": "quantitative", "label": "Global BH q"},
                        {"field": "n_eligible_sources", "type": "quantitative", "label": "Eligible sources"},
                        {"field": "n_positive", "type": "quantitative", "label": "Positive reference edges"},
                    ],
                },
                "referenceLines": [{"axis": "x", "value": 0.5, "label": "Chance"}],
                "palette": {"kind": "categorical"},
                "layout": "full",
            }
        )

    tables = [
        {
            "id": "internal_candidate_table",
            "title": "Frozen internal atlas candidates",
            "subtitle": "Atlas-only ranked hypotheses; target is the response row and source is the repaired-history column.",
            "dataset": "candidate_rows",
            "sourceId": "atlas_queue_source",
            "defaultSort": {"field": "queue_rank", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "queue_rank", "label": "Rank", "format": "number"},
                {"field": "pair_label", "label": "Source → target", "type": "text"},
                {"field": "channel", "label": "Metric", "type": "text"},
                {"field": "context", "label": "Context", "type": "text"},
                {"field": "source_lag_frames", "label": "Lag frames", "format": "number"},
                {"field": "horizon_frames", "label": "Horizon frames", "format": "number"},
                {"field": "mean_normalized", "label": "Normalized mean", "format": "number"},
                {"field": "valid_fraction", "label": "Valid fraction", "format": "percent"},
                {"field": "cross_sampler_spearman", "label": "Sampler Spearman", "format": "number"},
                {"field": "evidence_score", "label": "Evidence score", "format": "number"},
            ],
        },
        {
            "id": "targeted_primary_table",
            "title": "Frozen N128 candidate-specific response estimands",
            "subtitle": "Each candidate retains its own selected response metric. Magnitudes from unlike metrics are not pooled on one quantitative axis.",
            "dataset": "targeted_primary",
            "sourceId": "targeted_primary_source",
            "defaultSort": {"field": "candidate_id", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "candidate_id", "label": "Candidate", "type": "text"},
                {"field": "candidate_label", "label": "Source → target / timing", "type": "text"},
                {"field": "context", "label": "Context", "type": "text"},
                {"field": "contrast", "label": "Contrast", "type": "text"},
                {"field": "metric", "label": "Selected metric", "type": "text"},
                {"field": "mean_normalized", "label": "Mean normalized", "format": "number"},
                {"field": "ci_2_5", "label": "CI 2.5%", "format": "number"},
                {"field": "ci_97_5", "label": "CI 97.5%", "format": "number"},
                {"field": "valid_fraction", "label": "Valid fraction", "format": "percent"},
                {"field": "evidence_label", "label": "Internal label", "type": "text"},
            ],
        },
        {
            "id": "targeted_detail_table",
            "title": "Selected N128 distributional contrasts",
            "subtitle": "Exact model-relative high−low/high−factual/low−factual rows; target is the row axis and source is the column axis.",
            "dataset": "targeted_details",
            "sourceId": "targeted_details_source",
            "defaultSort": {"field": "candidate_id", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "candidate_id", "label": "Candidate", "type": "text"},
                {"field": "pair_label", "label": "Source → target", "type": "text"},
                {"field": "context", "label": "Context", "type": "text"},
                {"field": "source_lag_frames", "label": "Lag frames", "format": "number"},
                {"field": "horizon_frames", "label": "Horizon frames", "format": "number"},
                {"field": "contrast", "label": "Contrast", "type": "text"},
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "mean_normalized", "label": "Mean normalized", "format": "number"},
                {"field": "ci_2_5", "label": "CI 2.5%", "format": "number"},
                {"field": "ci_97_5", "label": "CI 97.5%", "format": "number"},
                {"field": "valid_fraction", "label": "Valid fraction", "format": "percent"},
                {"field": "evidence_label", "label": "Internal label", "type": "text"},
            ],
        },
        {
            "id": "targeted_support_table",
            "title": "N128 source support diagnostics",
            "subtitle": "Lowest-support selected source/lag/context rows; low/high ESS and achieved gaps govern interpretability.",
            "dataset": "targeted_support",
            "sourceId": "targeted_support_source",
            "defaultSort": {"field": "valid_fraction", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "source_neuron", "label": "Source", "type": "text"},
                {"field": "source_lag_frames", "label": "Lag frames", "format": "number"},
                {"field": "context", "label": "Context", "type": "text"},
                {"field": "valid_fraction", "label": "Valid fraction", "format": "percent"},
                {"field": "mean_achieved_gap_magnitude", "label": "Mean |gap|", "format": "number"},
                {"field": "mean_ess_low", "label": "Mean ESS low", "format": "number"},
                {"field": "mean_ess_high", "label": "Mean ESS high", "format": "number"},
            ],
        },
        {
            "id": "external_comparison_table",
            "title": "Randi, Cook, and SBTG contextual comparison",
            "subtitle": "Post-freeze only; AUPRC depends on edge prevalence, and SBTG/repaired-flow timing and scope are not the same estimand.",
            "dataset": "external_comparisons",
            "sourceId": "external_comparison_source",
            "defaultSort": {"field": "reference_or_network", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "reference_or_network", "label": "Reference", "type": "text"},
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "channel", "label": "Channel", "type": "text"},
                {"field": "context", "label": "Context", "type": "text"},
                {"field": "timing_label", "label": "Timing", "type": "text"},
                {"field": "scope_or_grid", "label": "Eligible scope", "type": "text"},
                {"field": "auroc", "label": "AUROC", "format": "number"},
                {"field": "auprc", "label": "AUPRC", "format": "number"},
                {"field": "continuous_spearman", "label": "Continuous Spearman", "format": "number"},
            ],
        },
        {
            "id": "neuromod_lagmax_table",
            "title": "Bentley neuromodulator lag-max inference",
            "subtitle": "Source-preserving max-lag null and global BH correction; selected lag is not a physical delay.",
            "dataset": "neuromod_lagmax",
            "sourceId": "external_lagmax_source",
            "defaultSort": {"field": "max_lag_bh_q", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "network", "label": "Network", "type": "text"},
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "channel", "label": "Channel", "type": "text"},
                {"field": "context", "label": "Context", "type": "text"},
                {"field": "best_lag_frames", "label": "Best lag frames", "format": "number"},
                {"field": "best_auroc", "label": "Best AUROC", "format": "number"},
                {"field": "max_lag_permutation_p", "label": "Max-lag p", "format": "number"},
                {"field": "max_lag_bh_q", "label": "Global BH q", "format": "number"},
                {"field": "n_eligible_sources", "label": "Sources", "format": "number"},
                {"field": "n_positive", "label": "Positive edges", "format": "number"},
            ],
        },
        {
            "id": "definitions_table",
            "title": "Metric, cohort, method, and timing definitions",
            "subtitle": "Definitions required before interpreting any matrix, lag profile, or external correspondence.",
            "dataset": "definitions",
            "sourceId": "atlas_protocol_source",
            "defaultSort": {"field": "term", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "term", "label": "Term", "type": "text"},
                {"field": "definition", "label": "Definition", "type": "text"},
            ],
        },
        {
            "id": "provenance_table",
            "title": "Canonical input integrity inventory",
            "subtitle": "Bundle ledgers were verified; root technical documents were hashed contemporaneously because no upstream document ledger exists.",
            "dataset": "provenance",
            "sourceId": "provenance_source",
            "defaultSort": {"field": "status", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "artifact", "label": "Bundle artifact", "type": "text"},
                {"field": "document", "label": "Root document", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
            ],
        },
    ]
    if atlas.stimulus_rows:
        tables.insert(
            1,
            {
                "id": "stimulus_composition_table",
                "title": "Exact-schedule stimulus boundary audit",
                "subtitle": (
                    "Phase/lag/horizon summaries with a source-window, forecast-window, "
                    "or cut-to-endpoint binary-stimulus transition; full per-worm rows "
                    "remain in stimulus_composition.csv."
                ),
                "dataset": "stimulus_composition",
                "sourceId": "atlas_protocol_source",
                "defaultSort": {"field": "phase", "direction": "asc"},
                "density": "compact",
                "layout": "full",
                "columns": [
                    {"field": "phase", "label": "Phase", "type": "text"},
                    {"field": "source_lag_frames", "label": "Lag", "format": "number"},
                    {"field": "horizon_frames", "label": "Horizon", "format": "number"},
                    {
                        "field": "source_window_stimulus_fraction_mean",
                        "label": "Source stimulus",
                        "format": "percent",
                    },
                    {
                        "field": "forecast_window_stimulus_fraction_mean",
                        "label": "Forecast stimulus",
                        "format": "percent",
                    },
                    {
                        "field": "forecast_endpoint_stimulus_fraction_mean",
                        "label": "Endpoint stimulus",
                        "format": "percent",
                    },
                    {
                        "field": "source_window_crosses_stimulus_boundary_any",
                        "label": "Source crosses",
                        "type": "boolean",
                    },
                    {
                        "field": "forecast_window_crosses_stimulus_boundary_any",
                        "label": "Forecast crosses",
                        "type": "boolean",
                    },
                    {
                        "field": "cut_to_endpoint_stimulus_transition_any",
                        "label": "Cut→endpoint changes",
                        "type": "boolean",
                    },
                ],
            },
        )
    if "genealogy_valid_fraction_0_10" in targeted.support.columns:
        targeted_support_table = next(
            table for table in tables if table["id"] == "targeted_support_table"
        )
        targeted_support_table["subtitle"] = (
            "Raw support remains separate from progressive-SMC genealogy; f=0.10 "
            "guards strong labels and f=0.20 is sensitivity-only."
        )
        targeted_support_table["columns"].extend(
            [
                {
                    "field": "genealogy_valid_fraction_0_10",
                    "label": "Genealogy valid (10%)",
                    "format": "percent",
                },
                {
                    "field": "genealogy_valid_fraction_0_20",
                    "label": "Genealogy valid (20%)",
                    "format": "percent",
                },
                {
                    "field": "genealogy_strong_gate_pass",
                    "label": "Strong genealogy gate",
                    "type": "boolean",
                },
            ]
        )
    if not targeted.screen.empty:
        tables.insert(
            3,
            {
                "id": "targeted_screen_table",
                "title": "Frozen screen versus selected N128 sensitivity",
                "subtitle": "Selection-conditioned consistency only; direct-screen rows also change sampler family.",
                "dataset": "targeted_screen",
                "sourceId": "targeted_screen_source",
                "defaultSort": {"field": "candidate_id", "direction": "asc"},
                "density": "compact",
                "layout": "full",
                "columns": [
                    {"field": "candidate_id", "label": "Candidate", "type": "text"},
                    {"field": "pair_label", "label": "Source → target", "type": "text"},
                    {"field": "screen_method", "label": "Screen method", "type": "text"},
                    {"field": "screen_channel", "label": "Screen metric", "type": "text"},
                    {"field": "screen_mean_normalized", "label": "Screen", "format": "number"},
                    {"field": "targeted_n128_mean_normalized", "label": "N128 high−low", "format": "number"},
                    {"field": "targeted_minus_screen", "label": "N128 − screen", "format": "number"},
                    {"field": "direction_agreement", "label": "Direction agrees", "format": "percent"},
                    {"field": "magnitude_agreement", "label": "Magnitude agreement", "format": "percent"},
                    {"field": "consistency_label", "label": "Internal label", "type": "text"},
                ],
            },
        )
    if targeted.quantile_rows:
        tables.insert(
            1,
            {
                "id": "targeted_quantile_table",
                "title": "Selected N128 endpoint quantile shifts",
                "subtitle": "Low/high/factual marginal quantile contrasts; quantiles supplement mean and scale summaries.",
                "dataset": "targeted_quantiles",
                "sourceId": "targeted_quantiles_source",
                "defaultSort": {"field": "candidate_id", "direction": "asc"},
                "density": "compact",
                "layout": "full",
                "columns": [
                    {"field": "candidate_id", "label": "Candidate", "type": "text"},
                    {"field": "pair_label", "label": "Source → target", "type": "text"},
                    {"field": "context", "label": "Event stratum", "type": "text"},
                    {"field": "contrast", "label": "Contrast", "type": "text"},
                    {"field": "endpoint_quantile", "label": "Quantile", "format": "number"},
                    {"field": "mean_normalized", "label": "Mean normalized shift", "format": "number", "movement": True},
                    {"field": "ci_2_5", "label": "CI 2.5%", "format": "number", "movement": True},
                    {"field": "ci_97_5", "label": "CI 97.5%", "format": "number", "movement": True},
                    {"field": "valid_fraction", "label": "Valid fraction", "format": "percent"},
                ],
            },
        )

    consistent_rows = int(targeted_counts.get("model_relative_consistent", 0))
    support_limited_rows = int(targeted_counts.get("unsupported_model_output", 0))
    reference_comparisons: list[str] = []
    comparison_frame = pd.DataFrame(external.comparison_rows)
    flow_methods = {"progressive_bridge_smc", "direct_importance"}
    sbtg_methods = {"sbtg_current", "sbtg_published"}
    for reference in sorted(comparison_frame["reference_or_network"].astype(str).unique()):
        current = comparison_frame[comparison_frame["reference_or_network"] == reference]
        flow = current[current["method"].isin(flow_methods)].dropna(subset=["auroc"])
        sbtg = current[current["method"].isin(sbtg_methods)].dropna(subset=["auroc"])
        if flow.empty or sbtg.empty:
            continue
        flow_row = flow.loc[flow["auroc"].astype(float).idxmax()]
        sbtg_row = sbtg.loc[sbtg["auroc"].astype(float).idxmax()]
        reference_comparisons.append(
            f"{reference.replace('_', ' ')}: largest descriptive flow AUROC among "
            "the included lag rows "
            f"{float(flow_row['auroc']):.3f} ({flow_row['method']}; "
            f"scope {flow_row['scope_or_grid']}); largest contextual SBTG AUROC "
            f"{float(sbtg_row['auroc']):.3f} ({sbtg_row['method']}; "
            f"scope {sbtg_row['scope_or_grid']})"
        )
    comparison_sentence = (
        "; ".join(reference_comparisons) + ". "
        if reference_comparisons
        else "No reference had both a finite prespecified-flow and contextual-SBTG AUROC row. "
    )
    genealogy_sentence = (
        f" The separate f=0.10 progressive-SMC genealogy gate passes for "
        f"{int(progressive_genealogy_pass)} source/context/lag rows; f=0.20 is "
        "reported only as sensitivity."
        if progressive_genealogy_pass is not None
        else ""
    )
    stimulus_sentence = (
        f" The exact schedule audit identifies {stimulus_crossing_rows} displayed "
        "phase/lag/horizon summaries in which a source or forecast window crosses a "
        "binary-stimulus boundary (the full per-worm table remains checksummed)."
        if stimulus_crossing_rows
        else ""
    )
    technical_summary = (
        "## Technical summary\n\n"
        "The reviewed atlas estimates **model-relative conditional distribution changes** "
        f"from repaired lagged source histories. Across the complete primary timing grid, "
        f"median direct-versus-progressive matrix Spearman is {median_agreement:.3f} "
        f"(range {minimum_agreement:.3f} to {maximum_agreement:.3f}). The selected N128 "
        f"analysis labels {consistent_rows} of {len(targeted.cells)} metric rows internally "
        f"model-relative consistent and {support_limited_rows} support-limited; it is a "
        "higher-particle sensitivity check, not an experiment. Randi, Cook, SBTG, and Bentley "
        f"were loaded only after internal freeze; {lagmax_significant} of {len(finite_q)} "
        "evaluable Bentley max-lag tests have global BH q≤0.05. These remain contextual checks. "
        "Chemical panels are event strata under a binary-any-stimulus generator, and no lag "
        "label is evidence of a physical transmission delay."
        + genealogy_sentence
        + stimulus_sentence
    )
    internal_text = (
        "## Internal support and sampler stability constrain every lag matrix\n\n"
        f"The atlas contains {len(atlas.queue)} support-qualified internal candidates from "
        f"{atlas.n_worms} worms. Mean validity across the reviewed support table is "
        f"{float(support_summary['mean_valid_fraction']):.1%}. The chart compares matrix "
        "rankings from direct importance and progressive bridge SMC at identical lag and "
        f"forecast cells; the median is {median_agreement:.3f}, with a "
        f"{minimum_agreement:.3f}–{maximum_agreement:.3f} range. Low or heterogeneous "
        "agreement is a robustness warning, not a "
        "reason to reinterpret one sampler as ground truth. Every cell remains model-relative "
        "and noncausal; chemical-named contexts are event strata under binary conditioning, "
        "and source-to-cut plus forecast timing is not a physical delay."
        + genealogy_sentence
        + stimulus_sentence
    )
    targeted_text = (
        "## Selected N128 three-arm results test model sensitivity, not causality\n\n"
        f"The targeted analysis evaluates {int(targeted.summary['n_candidates'])} selected "
        "source/target/context/horizon cells with matched future base noise across low, high, "
        "and factual flow arms. Contrasts are normalized event-wise by "
        "max(|achieved high-low source gap|, 0.10), seeds are averaged within worms, "
        "and intervals resample worms. "
        "The factual arm is another model rollout; high−factual and low−factual therefore "
        "locate the observed-history rollout relative to the repaired regimes but do not estimate an "
        "experimental intervention. Chemical-named rows retain binary-stimulus conditioning, "
        "and their selected source-to-cut/forecast bins are not physical delays."
    )
    targeted_primary_metrics = {
        str(row.get("metric")) for row in targeted.primary_rows if row.get("metric")
    }
    if len(targeted_primary_metrics) > 1:
        targeted_text += (
            " The frozen primary table preserves each candidate's selected response "
            "metric; magnitudes from unlike metrics are not pooled on one quantitative axis."
        )
    if "genealogy_strong_gate_pass" in targeted.support.columns:
        targeted_text += (
            " Targeted strong labels use the same separate f=0.10 genealogy gate; "
            "raw valid model rows remain visible when that gate fails, while f=0.20 "
            "is sensitivity-only."
        )
    if not targeted.screen.empty:
        direction = targeted.summary["screen_vs_n128"].get(
            "direction_agreement_fraction"
        )
        magnitude = targeted.summary["screen_vs_n128"].get(
            "median_magnitude_agreement"
        )
        targeted_text += (
            f" Across the frozen selected-cell screen, direction agreement is "
            f"{_fmt(direction)} and median magnitude agreement is {_fmt(magnitude)}; "
            "this comparison is selection-conditioned, and direct-screen rows also change "
            "sampler family."
        )
    external_text = (
        "## Randi, Cook, and SBTG provide contextual correspondence only\n\n"
        "AUROC evaluates binary response/edge existence, while continuous Spearman evaluates "
        "reference magnitudes where available; they answer different questions. AUPRC is "
        "also prevalence-dependent, so it is interpretable only against a like reference and "
        "eligible edge scope. All rows were "
        "computed after the atlas queue was frozen. "
        + comparison_sentence
        + "The reported maxima in that sentence are descriptive selections across included "
        "lag rows, not max-lag-corrected head-to-head tests. "
        + "These side-by-side values are not a head-to-head method-performance delta: "
        "flow rows use support-qualified source/edge scope, while contextual SBTG rows use "
        "their declared all-estimated shared-neuron scope. "
        + "SBTG-current and SBTG-published are aligned "
        "to the same 54-neuron target-row/source-column subset, but their training lineage and "
        "historical lag-score timing differ from repaired-flow source-to-cut plus forecast timing. "
        "This is a noncausal contextual comparison; chemical labels remain binary-generator event "
        "strata, and no timing-bin correspondence identifies a physical delay."
    )
    bentley_text = (
        "## Bentley neuromodulator correspondence does not establish physical delays\n\n"
        f"Among {len(finite_q)} evaluable planned max-lag tests, {lagmax_significant} have "
        "global BH q≤0.05. The source-preserving null calibrates the p-value for selecting "
        "the maximum across lag; the displayed best AUROC itself is the observed maximum, "
        "not an adjusted effect size. "
        "Even a low q-value would indicate correspondence with receptor/pathway-edge "
        "availability under this model screen—not measured neuromodulator activity, synaptic "
        "transmission, causality, or a physical propagation delay. Chemical-named flow contexts "
        "remain event strata under binary-any-stimulus conditioning."
    )
    limitations_text = (
        "## Limitations and robustness boundaries\n\n"
        "- **Observed-law model, not `do` intervention.** Repairs are soft conditional-history regimes under a fitted generator.\n"
        "- **Stimulus identity is not conditioned.** Butanone, pentanedione, and NaCl panels stratify real event positions while the model input remains binary any-stimulus.\n"
        "- **Support is estimand-specific.** Low ESS, concentrated weights, small achieved gaps, or weak ancestor diversity limit interpretation even when outputs are finite.\n"
        "- **Genealogy is a separate strong-evidence guard.** The original validity flag and model-only rows remain unchanged; f=0.10 can block promotion, while f=0.20 is reporting-only.\n"
        "- **Stimulus boundaries can cross timing windows.** Exact source/cut/forecast/endpoint composition must be checked before interpreting a lag/horizon panel as fixed-stimulus dynamics.\n"
        "- **Worm count limits precision.** Particles, events, seeds, targets, and time points are not independent biological replicates.\n"
        "- **External references are imperfect.** Randi perturbation response, Cook anatomy, Bentley receptor/pathway edges, and SBTG score products are not neural-activity ground truth.\n"
        "- **Lag localization is descriptive.** Source-to-cut, forecast, and source-to-readout timing must remain separate; a selected frame bin is not a physical delay."
    )
    methodology_text = (
        "## How repaired histories become directed lag matrices\n\n"
        "For each held-out event and source neuron, the four-frame source window ends "
        "`ℓ` frames before the prediction cut. Fold-, phase-, and lag-specific q25/q75 "
        "source regimes and a source-omitted population anchor define compatible low and "
        "high repaired histories. Progressive bridge SMC introduces the source constraint "
        "sequentially across the repair prefix: each parent produces two repair proposals, "
        "the bridge potential is adaptively tempered when candidate ESS would be too low, "
        "particles are resampled, and each repaired particle then receives one free future "
        "rollout. Direct importance instead self-normalizes low/high weights over one shared "
        "N=256 natural-path bank. The two procedures target the same fitted repaired-path law; "
        "branching, tempering, and resampling change finite-particle behavior, not the target law.\n\n"
        "Low and high rollouts use matched random-number streams within each estimator. For "
        "every target neuron, the high-minus-low distributional response is divided event-wise "
        "by `max(abs(achieved_source_gap), 0.10)`. Events are reduced within worm, generator "
        "seeds are retained for stability and then averaged within worm, and uncertainty "
        "resamples worms. Canonical arrays are written once as target rows by source columns. "
        "This is a soft, support-adaptive repaired-history association under the learned "
        "conditional law—not a single-neuron `do` intervention."
    )
    next_steps_text = (
        "## Recommended next steps\n\n"
        "1. Pre-register a small set of source/target/stimulus/lag hypotheses that jointly pass support, sampler-stability, and N128 sensitivity review.\n"
        "2. Collect or process more held-out worms and replicate the full worm-level analysis on Dandiset 000981 using the frozen cluster specification.\n"
        "3. Fit a chemically conditioned generator before claiming chemical-specific response effects; keep the present event-stratified results as a baseline.\n"
        "4. Prioritize experimental perturbations that measure both location and distributional changes—mean, variance/gain, crossing probability, and endpoint quantiles.\n"
        "5. Treat external connectome/receptor correspondence as secondary validation and keep it outside future internal model selection."
    )
    questions_text = (
        "## Further questions\n\n"
        "- Which selected effects reproduce across new worms, acquisition dates, and model checkpoints?\n"
        "- Do variance, gain, tail, or threshold-crossing effects stabilize at different lags than conditional means?\n"
        "- Which predictions remain after a chemically conditioned stimulus representation replaces the binary indicator?\n"
        "- Can targeted perturbation experiments distinguish source-history association from common drive and latent behavioral state?\n"
        "- Does a larger cohort sharpen lag profiles, or reveal that the apparent localization is primarily estimator variance?"
    )

    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {"id": "technical_summary", "type": "markdown", "sourceId": "report_summary_source", "body": technical_summary},
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {
            "id": "definitions_section",
            "type": "markdown",
            "sourceId": "atlas_protocol_source",
            "body": "## Metric, cohort, and method definitions\n\nThe table fixes the population, matrix orientation, timing axes, repaired-history estimand, sampler roles, factual-arm meaning, and distribution metrics before the detailed evidence.",
        },
        {"id": "definitions_table_block", "type": "table", "tableId": "definitions_table"},
        {
            "id": "methodology_section",
            "type": "markdown",
            "sourceId": "root_audit_source",
            "body": methodology_text,
        },
        {"id": "internal_section", "type": "markdown", "sourceId": "internal_narrative_source", "body": internal_text},
        {"id": "internal_chart", "type": "chart", "chartId": "internal_stability_chart"},
        {"id": "internal_candidates", "type": "table", "tableId": "internal_candidate_table"},
        {"id": "targeted_section", "type": "markdown", "sourceId": "targeted_narrative_source", "body": targeted_text},
    ]
    if atlas.stimulus_rows:
        blocks.insert(
            next(
                index
                for index, block in enumerate(blocks)
                if block["id"] == "internal_candidates"
            ),
            {
                "id": "stimulus_composition",
                "type": "table",
                "tableId": "stimulus_composition_table",
            },
        )
    if any(chart["id"] == "targeted_three_arm_chart" for chart in charts):
        blocks.append({"id": "targeted_chart", "type": "chart", "chartId": "targeted_three_arm_chart"})
    blocks.extend(
        [
            {"id": "targeted_primary", "type": "table", "tableId": "targeted_primary_table"},
            {"id": "targeted_detail", "type": "table", "tableId": "targeted_detail_table"},
            {"id": "targeted_support", "type": "table", "tableId": "targeted_support_table"},
        ]
    )
    if any(table["id"] == "targeted_screen_table" for table in tables):
        blocks.append(
            {"id": "targeted_screen", "type": "table", "tableId": "targeted_screen_table"}
        )
    if any(table["id"] == "targeted_quantile_table" for table in tables):
        blocks.append({"id": "targeted_quantiles", "type": "table", "tableId": "targeted_quantile_table"})
    blocks.append({"id": "external_section", "type": "markdown", "sourceId": "external_comparison_source", "body": external_text})
    if any(chart["id"] == "external_comparison_chart" for chart in charts):
        blocks.append({"id": "external_chart", "type": "chart", "chartId": "external_comparison_chart"})
    blocks.append({"id": "external_table", "type": "table", "tableId": "external_comparison_table"})
    blocks.append({"id": "bentley_section", "type": "markdown", "sourceId": "external_lagmax_source", "body": bentley_text})
    if any(chart["id"] == "neuromod_lagmax_chart" for chart in charts):
        blocks.append({"id": "bentley_chart", "type": "chart", "chartId": "neuromod_lagmax_chart"})
    blocks.extend(
        [
            {"id": "bentley_table", "type": "table", "tableId": "neuromod_lagmax_table"},
            {
                "id": "reproducibility_section",
                "type": "markdown",
                "sourceId": "provenance_source",
                "body": "## Reproducibility and integrity\n\nEvery canonical bundle ledger was reconciled and every listed file was rehashed. Targeted raw-input hashes, the frozen-atlas firewall, external reference-release hashes, and SBTG archive sidecars were also verified. Root protocol/audit documents have contemporaneous report-build hashes, not upstream-frozen expected digests; the distinction is retained in the inventory.",
            },
            {"id": "provenance_table_block", "type": "table", "tableId": "provenance_table"},
            {"id": "limitations", "type": "markdown", "sourceId": "root_audit_source", "body": limitations_text},
            {"id": "next_steps", "type": "markdown", "body": next_steps_text},
            {"id": "further_questions", "type": "markdown", "body": questions_text},
        ]
    )

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Technical synthesis of the reviewed internal atlas, selected N128 three-arm sensitivity, and post-freeze contextual reference analyses.",
            "generatedAt": generated,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [
                {"id": source["id"], "label": source["label"], "path": source["path"]}
                for source in sources
            ],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }
    serialized = json.dumps(
        _json_safe(artifact), indent=2, sort_keys=False, allow_nan=False
    ) + "\n"
    if len(serialized.encode("utf-8")) >= MAX_ARTIFACT_BYTES:
        raise ReportInputError("portable report artifact exceeds three megabytes")

    markdown = "\n\n".join(
        [
            f"# {TITLE}",
            technical_summary.replace("## Technical summary\n\n", "## Technical summary\n\n"),
            "## Metric, cohort, and method definitions\n\n"
            + _markdown_table(definitions, ("term", "definition")),
            methodology_text,
            internal_text,
            (
                "### Exact-schedule stimulus boundary audit\n\n"
                + _markdown_table(
                    atlas.stimulus_rows,
                    (
                        "phase",
                        "source_lag_frames",
                        "horizon_frames",
                        "source_window_stimulus_fraction_mean",
                        "forecast_window_stimulus_fraction_mean",
                        "forecast_endpoint_stimulus_fraction_mean",
                        "source_window_crosses_stimulus_boundary_any",
                        "forecast_window_crosses_stimulus_boundary_any",
                        "cut_to_endpoint_stimulus_transition_any",
                    ),
                )
                if atlas.stimulus_rows
                else ""
            ),
            _markdown_table(
                atlas.stability_rows,
                (
                    "source_lag_frames",
                    "horizon_frames",
                    "direct_vs_progressive_spearman",
                    "source_to_readout_seconds",
                ),
            ),
            "### Frozen internal atlas candidates",
            _markdown_table(
                atlas.candidate_rows,
                (
                    "queue_rank",
                    "pair_label",
                    "channel",
                    "context",
                    "source_lag_frames",
                    "horizon_frames",
                    "mean_normalized",
                    "valid_fraction",
                    "cross_sampler_spearman",
                    "evidence_score",
                ),
            ),
            targeted_text,
            _markdown_table(
                targeted.detail_rows,
                (
                    "candidate_id",
                    "pair_label",
                    "context",
                    "contrast",
                    "metric",
                    "mean_normalized",
                    "ci_2_5",
                    "ci_97_5",
                    "valid_fraction",
                ),
            ),
            "### Selected N128 source support and genealogy",
            _markdown_table(
                targeted.support_rows,
                tuple(
                    column
                    for column in (
                        "source_neuron",
                        "source_lag_frames",
                        "context",
                        "valid_fraction",
                        "genealogy_valid_fraction_0_10",
                        "genealogy_valid_fraction_0_20",
                        "genealogy_strong_gate_pass",
                        "mean_ess_low",
                        "mean_ess_high",
                    )
                    if column in targeted.support.columns
                ),
            ),
            "### Selected N128 endpoint quantile shifts",
            _markdown_table(
                targeted.quantile_rows,
                (
                    "candidate_id",
                    "pair_label",
                    "context",
                    "contrast",
                    "endpoint_quantile",
                    "mean_normalized",
                    "ci_2_5",
                    "ci_97_5",
                    "valid_fraction",
                ),
            ),
            (
                "### Frozen screen versus selected N128\n\n"
                + _markdown_table(
                    _records(targeted.screen.head(MAX_SCREEN_ROWS)),
                    (
                        "candidate_id",
                        "pair_label",
                        "screen_method",
                        "screen_channel",
                        "screen_mean_normalized",
                        "targeted_n128_mean_normalized",
                        "targeted_minus_screen",
                        "direction_agreement",
                        "magnitude_agreement",
                        "consistency_label",
                    ),
                )
                if not targeted.screen.empty
                else "### Frozen screen versus selected N128\n\n_Not requested in the canonical targeted analysis._"
            ),
            external_text,
            _markdown_table(
                external.comparison_rows[:30],
                (
                    "reference_or_network",
                    "method",
                    "channel",
                    "context",
                    "scope_or_grid",
                    "auroc",
                    "auprc",
                    "continuous_spearman",
                ),
            ),
            bentley_text,
            _markdown_table(
                external.lagmax_rows[:30],
                (
                    "network",
                    "method",
                    "channel",
                    "context",
                    "best_lag_frames",
                    "best_auroc",
                    "max_lag_bh_q",
                ),
            ),
            limitations_text,
            next_steps_text,
            questions_text,
            "## Reproducibility inventory\n\n"
            "`artifact.json` is the canonical report surface. This Markdown file is a "
            "readable source companion generated from the same validated bounded rows. "
            "Bundle-ledger rows are verified against frozen expected hashes; root-document "
            "rows are contemporaneous report-build hashes because no upstream document "
            "ledger exists.\n\n"
            + _markdown_table(
                provenance_rows,
                tuple(
                    column
                    for column in ("artifact", "document", "status", "sha256")
                    if any(column in row for row in provenance_rows)
                ),
            ),
        ]
    ).strip() + "\n"
    _atomic_text(output / "artifact.json", serialized)
    _atomic_text(output / "TECHNICAL_REPORT.md", markdown)
    _atomic_text(
        output / "checksums.sha256",
        "".join(
            f"{sha256(output / name)}  {name}\n"
            for name in ("artifact.json", "TECHNICAL_REPORT.md")
        ),
    )
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--targeted-analysis-dir", type=Path, required=True)
    parser.add_argument("--postfreeze-external-dir", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance-root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_prediction_atlas_report(
        args.atlas_dir,
        args.targeted_analysis_dir,
        args.postfreeze_external_dir,
        args.atlas_root,
        args.output_dir,
        provenance_root=args.provenance_root,
        overwrite=args.overwrite,
    )
    print(str((args.output_dir / "artifact.json").resolve()))


if __name__ == "__main__":
    main()
