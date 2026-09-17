"""Generate the canonical portable-dashboard artifact for the prediction atlas.

The module authors ``artifact.json`` only.  HTML is intentionally left to the
installed Data Analytics portable-artifact builder so there is one reader,
chart runtime, and validation contract.  Embedded datasets are compact views of
the reviewed atlas bundle; at most one complete 54x54 matrix slice is pivoted to
54 rows, while dense archives and full Parquet tables remain outside the payload.
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
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


REQUIRED_FILES = (
    "dashboard_snapshot.json",
    "hypothesis_queue.csv",
    "validation.json",
    "protocol.json",
    "models.json",
    "checksums.sha256",
)
PRIMARY_METHOD = "progressive_bridge_smc"
PRIMARY_CHANNEL = "endpoint_mean"
PRIMARY_CONTEXT = "onset_minus_baseline"
ORIENTATION = "target_row_source_column"
ALLOWED_EVIDENCE_TIERS = {
    "confirmed",
    "supported_exploratory",
    "model_only",
    "unsupported",
}
CHEMICALS = ("butanone", "pentanedione", "nacl")
MAX_QUEUE_ROWS = 5_000
MAX_MATRIX_ROWS = 2_916
MATRIX_DIMENSION = 54
MAX_PROFILE_CANDIDATES = 12
MAX_EXTERNAL_COLUMNS = 12
MAX_TARGETED_CANDIDATES = 6
MAX_TARGETED_SENSITIVITY_ROWS = 18
TARGETED_ANALYSIS_SCHEMA = "targeted_confirmation_analysis_v1"
TARGETED_SELECTION_SCHEMA = "prediction_atlas_targeted_selection_v1"
TARGETED_METHOD = "progressive_bridge_smc_targeted_three_arm"
TARGETED_CONTRASTS = ("high_low", "high_factual", "low_factual")
TARGETED_EVIDENCE_LABELS = {
    "model_relative_consistent",
    "model_relative_uncertain",
    "model_relative_descriptive",
    "unsupported_model_output",
}
CHANNEL_LABELS = {
    "endpoint_mean": "Endpoint mean shift",
    "cumulative_mean": "Time-averaged mean shift",
    "peak_mean": "Expected pathwise peak shift",
    "event_probability": "Threshold-crossing-by-horizon probability shift",
    "endpoint_sd": "Endpoint SD shift",
    "endpoint_log_sd": "Endpoint log-SD shift",
    "endpoint_wasserstein1": "Endpoint Wasserstein-1 distance",
}
SCREEN_TO_TARGETED_METRIC = {
    "endpoint_mean": "endpoint_mean",
    "cumulative_mean": "time_average_mean",
    "peak_mean": "pathwise_peak_mean",
    "event_probability": "crossing_probability",
    "endpoint_sd": "endpoint_sd",
    "endpoint_log_sd": "endpoint_log_sd",
    "endpoint_wasserstein1": "endpoint_wasserstein1",
}
TARGETED_REQUIRED_FILES = (
    "manifest.json",
    "validation.json",
    "protocol.json",
    "summary.json",
    "targeted_cells.csv",
    "screen_consistency.csv",
    "input_checksums.csv",
)
UNSAFE_FIELD = re.compile(
    r"^(?:password|passwd|pwd|secret|api[-_]?key|access[-_]?token|"
    r"refresh[-_]?token|token|authorization|cookie|private[-_]?key|credential)$",
    re.IGNORECASE,
)


class DashboardInputError(RuntimeError):
    """Raised when the reviewed atlas bundle cannot safely back a dashboard."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DashboardInputError(f"required dashboard input is missing: {path.name}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardInputError(f"could not read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise DashboardInputError(f"{path.name} must contain a JSON object")
    return value


def _safe_relative(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise DashboardInputError(
            f"{resolved.name} lies outside the declared provenance root"
        ) from error
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise DashboardInputError("portable provenance paths must be local and traversal-free")
    return relative.as_posix()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DashboardInputError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise DashboardInputError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    result = _finite(value, label)
    if not result.is_integer():
        raise DashboardInputError(f"{label} must be an integer")
    return int(result)


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if pd.isna(value):
        return None
    return str(value)


def _verify_checksums(atlas_dir: Path) -> dict[str, str]:
    path = atlas_dir / "checksums.sha256"
    if not path.exists():
        raise DashboardInputError("required dashboard input is missing: checksums.sha256")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise DashboardInputError(
                f"checksums.sha256 line {line_number} is malformed"
            )
        filename = parts[1]
        candidate = Path(filename)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or len(candidate.parts) != 1
            or not candidate.name
        ):
            raise DashboardInputError("checksum inventory contains an unsafe path")
        if filename in entries:
            raise DashboardInputError(
                f"checksum inventory duplicates {filename}"
            )
        entries[filename] = parts[0]
    for filename in REQUIRED_FILES[:-1]:
        source = atlas_dir / filename
        if filename not in entries:
            raise DashboardInputError(f"checksums.sha256 does not cover {filename}")
        if not source.exists() or sha256(source) != entries[filename]:
            raise DashboardInputError(f"checksum validation failed for {filename}")
    return entries


def _require_keys(value: dict[str, Any], keys: Iterable[str], label: str) -> None:
    missing = sorted(set(keys).difference(value))
    if missing:
        raise DashboardInputError(f"{label} is missing required fields: {missing}")


def _validate_bundle_metadata(
    dashboard: dict[str, Any],
    validation: dict[str, Any],
    protocol: dict[str, Any],
    models: dict[str, Any],
) -> tuple[int, int, float]:
    _require_keys(
        dashboard,
        (
            "title",
            "status",
            "primary_method",
            "cohort",
            "methods",
            "channels",
            "contexts",
            "source_lags",
            "horizons",
            "top_candidates",
            "support_summary",
            "warnings",
        ),
        "dashboard_snapshot.json",
    )
    cohort = dashboard["cohort"]
    if not isinstance(cohort, dict):
        raise DashboardInputError("dashboard cohort must be an object")
    _require_keys(cohort, ("worms", "neurons", "fps"), "dashboard cohort")
    n_worms = _integer(cohort["worms"], "cohort.worms")
    n_neurons = _integer(cohort["neurons"], "cohort.neurons")
    fps = _finite(cohort["fps"], "cohort.fps")
    if n_worms != 17 or n_neurons != 54 or fps <= 0:
        raise DashboardInputError(
            "portable atlas dashboard requires the reviewed 17-worm, 54-neuron cohort"
        )
    if dashboard["status"] != "reviewed_model_predictions":
        raise DashboardInputError("dashboard snapshot is not marked reviewed_model_predictions")
    if dashboard["primary_method"] != PRIMARY_METHOD:
        raise DashboardInputError("progressive bridge SMC must be the primary dashboard method")
    if PRIMARY_METHOD not in dashboard["methods"]:
        raise DashboardInputError("primary method is absent from dashboard methods")
    if PRIMARY_CHANNEL not in dashboard["channels"]:
        raise DashboardInputError("endpoint_mean is absent from dashboard channels")

    if validation.get("status") != "passed" or validation.get("problems") not in ([], None):
        raise DashboardInputError("validation.json does not report a clean passed audit")
    checks = validation.get("archive_checks_completed")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise DashboardInputError("one or more canonical archive checks did not pass")
    if _integer(validation.get("n_worms"), "validation.n_worms") != n_worms:
        raise DashboardInputError("validation worm count differs from dashboard snapshot")
    if _integer(validation.get("n_neurons"), "validation.n_neurons") != n_neurons:
        raise DashboardInputError("validation neuron count differs from dashboard snapshot")

    _require_keys(
        protocol,
        (
            "n_worms",
            "n_neurons",
            "fps",
            "source_lag_frames",
            "horizon_frames",
            "timing_definitions",
            "orientation",
            "chemical_context_warning",
            "evidence_tiers",
            "interpretation_limit",
            "ranking",
        ),
        "protocol.json",
    )
    if (
        _integer(protocol["n_worms"], "protocol.n_worms") != n_worms
        or _integer(protocol["n_neurons"], "protocol.n_neurons") != n_neurons
        or not np.isclose(_finite(protocol["fps"], "protocol.fps"), fps)
    ):
        raise DashboardInputError("protocol cohort metadata does not reconcile")
    if protocol["orientation"] != ORIENTATION:
        raise DashboardInputError("protocol orientation is not target-row/source-column")
    timing = protocol["timing_definitions"]
    if not isinstance(timing, dict) or set(
        (
            "source_to_cut_seconds",
            "forecast_horizon_seconds",
            "source_to_readout_seconds",
        )
    ).difference(timing):
        raise DashboardInputError("protocol lacks the three distinct timing definitions")
    if "not chemically conditioned" not in str(protocol["chemical_context_warning"]):
        raise DashboardInputError("protocol lacks the chemical-conditioning claim boundary")
    if "not a causal" not in str(protocol["interpretation_limit"]):
        raise DashboardInputError("protocol lacks the noncausal claim boundary")
    if "no connectome or external reference data used" not in str(protocol["ranking"]):
        raise DashboardInputError("protocol does not preserve the external-reference firewall")
    if set(protocol["evidence_tiers"]) != ALLOWED_EVIDENCE_TIERS:
        raise DashboardInputError("protocol evidence tiers are incomplete or changed")

    _require_keys(
        models,
        ("generator_model_id", "generator_stimulus_encoding", "chemical_conditioning", "methods"),
        "models.json",
    )
    if models["generator_stimulus_encoding"] != "binary_any_stimulus":
        raise DashboardInputError("dashboard requires the binary-any-stimulus generator")
    if models["chemical_conditioning"] is not False:
        raise DashboardInputError("chemical identity must not be represented as conditioned")
    if PRIMARY_METHOD not in models["methods"]:
        raise DashboardInputError("models.json lacks the primary method")
    return n_worms, n_neurons, fps


QUEUE_REQUIRED_COLUMNS = {
    "queue_rank",
    "method",
    "channel",
    "context",
    "conditioning_status",
    "source_neuron",
    "target_neuron",
    "source_lag_frames",
    "source_to_cut_seconds",
    "horizon_frames",
    "forecast_horizon_seconds",
    "source_to_readout_seconds",
    "mean_normalized",
    "ci_2_5",
    "ci_97_5",
    "sign_consistency",
    "valid_fraction",
    "seed_sign_agreement",
    "n_worms",
    "support_tier",
    "evidence_score",
    "cross_sampler_spearman",
}


def _read_queue(path: Path, *, n_worms: int, fps: float) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise DashboardInputError(f"could not read hypothesis_queue.csv: {error}") from error
    missing = sorted(QUEUE_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise DashboardInputError(f"hypothesis_queue.csv is missing columns: {missing}")
    if frame.empty or len(frame) > MAX_QUEUE_ROWS:
        raise DashboardInputError(
            f"hypothesis queue must contain between 1 and {MAX_QUEUE_ROWS} reviewed rows"
        )
    ranks = pd.to_numeric(frame["queue_rank"], errors="coerce").to_numpy()
    if not np.isfinite(ranks).all() or not np.array_equal(
        ranks.astype(int), np.arange(1, len(frame) + 1)
    ):
        raise DashboardInputError("hypothesis queue ranks must be unique and contiguous")
    numeric_required = (
        "source_lag_frames",
        "source_to_cut_seconds",
        "horizon_frames",
        "forecast_horizon_seconds",
        "source_to_readout_seconds",
        "mean_normalized",
        "ci_2_5",
        "ci_97_5",
        "sign_consistency",
        "valid_fraction",
        "evidence_score",
        "n_worms",
    )
    for column in numeric_required:
        value = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(value).all():
            raise DashboardInputError(f"hypothesis queue column {column} is nonnumeric/nonfinite")
        frame[column] = value
    for column in ("queue_rank", "source_lag_frames", "horizon_frames", "n_worms"):
        integer_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(integer_values == np.floor(integer_values)):
            raise DashboardInputError(f"hypothesis queue column {column} must be integral")
        frame[column] = integer_values.astype(int)
    if not np.all(frame["n_worms"].to_numpy() == n_worms):
        raise DashboardInputError("hypothesis queue uses a different worm count")
    lag = frame["source_lag_frames"].to_numpy(dtype=float)
    horizon = frame["horizon_frames"].to_numpy(dtype=float)
    if not (
        np.allclose(frame["source_to_cut_seconds"], lag / fps)
        and np.allclose(frame["forecast_horizon_seconds"], horizon / fps)
        and np.allclose(frame["source_to_readout_seconds"], (lag + horizon) / fps)
    ):
        raise DashboardInputError("hypothesis queue timing columns do not reconcile")
    if np.any((frame["valid_fraction"] < 0) | (frame["valid_fraction"] > 1)):
        raise DashboardInputError("hypothesis queue valid_fraction lies outside [0,1]")
    if np.any(frame["evidence_score"] < 0):
        raise DashboardInputError("hypothesis queue evidence_score is negative")
    if not set(frame["support_tier"].astype(str)).issubset(ALLOWED_EVIDENCE_TIERS):
        raise DashboardInputError("hypothesis queue contains an unknown evidence tier")
    chemical_context = frame["context"].astype(str).str.startswith(CHEMICALS)
    chemistry_label = frame["conditioning_status"].astype(str)
    if chemical_context.any() and not chemistry_label[chemical_context].str.contains(
        "exploratory_event_stratified", regex=False
    ).all():
        raise DashboardInputError("chemical queue rows are not labeled event-stratified")
    return frame


def _crosscheck_snapshot_candidates(
    dashboard: dict[str, Any], frame: pd.DataFrame
) -> None:
    candidates = dashboard["top_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise DashboardInputError("dashboard snapshot has no bounded top_candidates")
    if len(candidates) > len(frame):
        raise DashboardInputError("snapshot candidate count exceeds the queue")
    keys = (
        "queue_rank",
        "method",
        "channel",
        "context",
        "source_neuron",
        "target_neuron",
        "source_lag_frames",
        "horizon_frames",
    )
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise DashboardInputError("snapshot candidates must be objects")
        queue_row = frame.iloc[index]
        for key in keys:
            if str(candidate.get(key)) != str(_json_scalar(queue_row[key])):
                raise DashboardInputError(
                    f"dashboard candidate {index + 1} does not match hypothesis_queue.csv ({key})"
                )


def _candidate_rows(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    selected = frame.iloc[:limit]
    rows: list[dict[str, Any]] = []
    for _, source in selected.iterrows():
        lag = int(source["source_lag_frames"])
        horizon = int(source["horizon_frames"])
        lag_s = float(source["source_to_cut_seconds"])
        horizon_s = float(source["forecast_horizon_seconds"])
        readout_s = float(source["source_to_readout_seconds"])
        rank = int(source["queue_rank"])
        row = {column: _json_scalar(source[column]) for column in selected.columns}
        row.update(
            {
                "queue_rank": rank,
                "pair_label": f"{source['source_neuron']} → {source['target_neuron']}",
                "channel_label": CHANNEL_LABELS.get(
                    str(source["channel"]), str(source["channel"])
                ),
                "context_label": str(source["context"]).replace("_", " "),
                "candidate_label": (
                    f"#{rank} {source['source_neuron']} → {source['target_neuron']} "
                    f"· ℓ{lag}/h{horizon}"
                ),
                "source_lag_filter": f"{lag} frames ({lag_s:g} s to cut)",
                "horizon_filter": f"{horizon} frames ({horizon_s:g} s forecast)",
                "timing_label": (
                    f"to cut {lag_s:g} s · forecast {horizon_s:g} s · "
                    f"source-to-readout {readout_s:g} s"
                ),
                "absolute_mean_normalized": abs(float(source["mean_normalized"])),
                "ci_width": float(source["ci_97_5"] - source["ci_2_5"]),
                "chemistry_scope": (
                    "exploratory event-stratified; not chemically conditioned"
                    if str(source["context"]).startswith(CHEMICALS)
                    else "binary-any-stimulus context"
                ),
            }
        )
        rows.append(row)
    return rows


def _primary_agreement_profile(
    dashboard: dict[str, Any], validation: dict[str, Any], fps: float
) -> list[dict[str, Any]]:
    diagnostics = validation.get("cross_sampler_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise DashboardInputError("validation lacks cross-sampler diagnostics")
    lags = [int(item["frames"]) for item in dashboard["source_lags"]]
    horizons = [int(item["frames"]) for item in dashboard["horizons"]]
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for item in diagnostics:
        if not isinstance(item, dict):
            raise DashboardInputError("cross-sampler diagnostics must be objects")
        if item.get("channel") != PRIMARY_CHANNEL or item.get("context") != PRIMARY_CONTEXT:
            continue
        lag = _integer(item.get("source_lag_frames"), "cross-sampler source lag")
        horizon = _integer(item.get("horizon_frames"), "cross-sampler horizon")
        if (lag, horizon) in lookup:
            raise DashboardInputError("primary cross-sampler timing cell is duplicated")
        value = item.get("direct_vs_progressive_spearman")
        agreement = None if value is None else _finite(value, "cross-sampler Spearman")
        if agreement is not None and not -1 <= agreement <= 1:
            raise DashboardInputError("cross-sampler Spearman lies outside [-1,1]")
        lookup[(lag, horizon)] = {
            "source_lag_frames": lag,
            "horizon_frames": horizon,
            "source_lag_label": f"ℓ={lag} ({lag / fps:g} s to cut)",
            "forecast_horizon_label": f"h={horizon} ({horizon / fps:g} s forecast)",
            "source_to_cut_seconds": lag / fps,
            "forecast_horizon_seconds": horizon / fps,
            "source_to_readout_seconds": (lag + horizon) / fps,
            "direct_vs_progressive_spearman": agreement,
            "method_comparison": "direct importance vs progressive bridge SMC",
            "channel": PRIMARY_CHANNEL,
            "context": PRIMARY_CONTEXT,
            "n_worms": 17,
        }
    expected = {(lag, horizon) for lag in lags for horizon in horizons}
    if set(lookup) != expected:
        raise DashboardInputError("primary cross-sampler lag/horizon grid is incomplete")
    if not any(row["direct_vs_progressive_spearman"] is not None for row in lookup.values()):
        raise DashboardInputError("primary cross-sampler profile contains no finite values")
    return [lookup[(lag, horizon)] for lag in lags for horizon in horizons]


def _stimulus_context_rows(
    dashboard: dict[str, Any], validation: dict[str, Any]
) -> list[dict[str, Any]]:
    contexts = dashboard["contexts"]
    if not isinstance(contexts, list) or not contexts:
        raise DashboardInputError("dashboard contexts are missing")
    metadata: dict[str, dict[str, Any]] = {}
    for item in contexts:
        if not isinstance(item, dict) or not item.get("context"):
            raise DashboardInputError("dashboard context metadata is malformed")
        metadata[str(item["context"])] = item
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in validation.get("cross_sampler_diagnostics", []):
        if item.get("channel") != PRIMARY_CHANNEL:
            continue
        value = item.get("direct_vs_progressive_spearman")
        if value is not None:
            buckets[str(item.get("context"))].append(
                _finite(value, "context cross-sampler Spearman")
            )
    rows: list[dict[str, Any]] = []
    for context, item in metadata.items():
        values = buckets.get(context, [])
        rows.append(
            {
                "context": context,
                "context_label": context.replace("_", " "),
                "mean_direct_vs_progressive_spearman": mean(values) if values else None,
                "median_direct_vs_progressive_spearman": median(values) if values else None,
                "timing_cells_with_finite_agreement": len(values),
                "event_stratified": bool(item.get("event_stratified", False)),
                "conditioning_scope": (
                    "exploratory event-stratified; not chemically conditioned"
                    if item.get("event_stratified")
                    else "binary-any-stimulus context"
                ),
                "n_worms": 17,
            }
        )
    if not rows:
        raise DashboardInputError("no stimulus-context rows could be constructed")
    return rows


def _optional_matrix_rows(dashboard: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one complete source-major long matrix into 54 target rows.

    The portable heatmap consumes one categorical x field plus a numeric
    ``y.fields`` collection.  Storing one row per target and one numeric field
    per source therefore preserves the scientific target-row/source-column
    orientation while keeping the embedded dataset below the 2,000-row cap.
    """

    raw = dashboard.get("matrix_rows")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw or len(raw) > MAX_MATRIX_ROWS:
        raise DashboardInputError(
            f"optional matrix_rows must contain 1..{MAX_MATRIX_ROWS} rows"
        )
    required = {
        "source_neuron",
        "target_neuron",
        "mean_normalized",
        "source_lag_frames",
        "horizon_frames",
        "method",
        "channel",
        "context",
    }
    long_rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or required.difference(item):
            raise DashboardInputError("optional matrix_rows has an incompatible schema")
        if (
            item["method"] != PRIMARY_METHOD
            or item["channel"] != PRIMARY_CHANNEL
            or item["context"] != PRIMARY_CONTEXT
        ):
            continue
        source = str(item["source_neuron"])
        target = str(item["target_neuron"])
        if not source or not target:
            raise DashboardInputError("matrix neuron names must be nonempty")
        long_rows.append(
            {
                "source_neuron": source,
                "target_neuron": target,
                "mean_normalized": _finite(item["mean_normalized"], "matrix value"),
                "source_lag_frames": _integer(
                    item["source_lag_frames"], "matrix source lag"
                ),
                "horizon_frames": _integer(
                    item["horizon_frames"], "matrix forecast horizon"
                ),
                "method": str(item["method"]),
                "channel": str(item["channel"]),
                "context": str(item["context"]),
            }
        )
    if not long_rows:
        return None
    slices = {
        (row["source_lag_frames"], row["horizon_frames"]) for row in long_rows
    }
    if len(slices) != 1:
        return None

    source_neurons = list(dict.fromkeys(row["source_neuron"] for row in long_rows))
    target_neurons = list(dict.fromkeys(row["target_neuron"] for row in long_rows))
    if len(source_neurons) != MATRIX_DIMENSION or len(target_neurons) != MATRIX_DIMENSION:
        raise DashboardInputError(
            "primary matrix slice must contain exactly 54 source and 54 target neurons"
        )
    if len(long_rows) != MATRIX_DIMENSION * MATRIX_DIMENSION:
        raise DashboardInputError("primary matrix slice must contain all 54x54 cells")
    reserved = {
        "target_neuron",
        "source_lag_frames",
        "horizon_frames",
        "method",
        "channel",
        "context",
    }
    collisions = sorted(reserved.intersection(source_neurons))
    if collisions:
        raise DashboardInputError(
            f"source neuron names collide with matrix metadata fields: {collisions}"
        )
    cells: dict[tuple[str, str], float] = {}
    for row in long_rows:
        key = (row["target_neuron"], row["source_neuron"])
        if key in cells:
            raise DashboardInputError("primary matrix slice duplicates a target/source cell")
        cells[key] = float(row["mean_normalized"])
    expected = {
        (target, source) for target in target_neurons for source in source_neurons
    }
    if set(cells) != expected:
        raise DashboardInputError("primary matrix slice is not a complete Cartesian grid")

    first = long_rows[0]
    wide_rows: list[dict[str, Any]] = []
    for target in target_neurons:
        wide_rows.append(
            {
                "target_neuron": target,
                "source_lag_frames": first["source_lag_frames"],
                "horizon_frames": first["horizon_frames"],
                "method": PRIMARY_METHOD,
                "channel": PRIMARY_CHANNEL,
                "context": PRIMARY_CONTEXT,
                **{source: cells[(target, source)] for source in source_neurons},
            }
        )
    return {
        "rows": wide_rows,
        "source_neurons": source_neurons,
        "target_neurons": target_neurons,
        "source_lag_frames": first["source_lag_frames"],
        "horizon_frames": first["horizon_frames"],
    }


def _candidate_lag_profile_rows(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    raw = dashboard.get("candidate_lag_profiles")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise DashboardInputError("candidate_lag_profiles must be a list")
    expected_lags = [int(item["frames"]) for item in dashboard["source_lags"]]
    if not raw or len(raw) > MAX_QUEUE_ROWS * len(expected_lags):
        raise DashboardInputError("candidate_lag_profiles has an invalid row count")
    top_candidates = dashboard["top_candidates"][:MAX_PROFILE_CANDIDATES]
    retained_ranks = {
        _integer(item.get("queue_rank"), "candidate lag-profile queue rank")
        for item in top_candidates
    }
    required = {
        "queue_rank",
        "source_neuron",
        "target_neuron",
        "channel",
        "context",
        "horizon_frames",
        "source_lag_frames",
        "source_to_cut_seconds",
        "primary_method",
        "counterpart_method",
        "primary_mean_normalized",
        "counterpart_mean_normalized",
        "is_primary_peak_abs_lag",
        "worm_bootstrap_peak_lag_rate",
        "top_vs_second_lag_selectivity",
        "signed_lag_profile_spearman",
        "interpretation",
    }
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or required.difference(item):
            raise DashboardInputError("candidate_lag_profiles has an incompatible schema")
        rank = _integer(item["queue_rank"], "candidate lag-profile queue rank")
        if rank not in retained_ranks:
            continue
        lag = _integer(item["source_lag_frames"], "candidate lag-profile lag")
        horizon = _integer(item["horizon_frames"], "candidate lag-profile horizon")
        lag_seconds = _finite(
            item["source_to_cut_seconds"], "candidate lag-profile source-to-cut time"
        )
        if lag not in expected_lags or not np.isclose(
            lag_seconds, lag / float(dashboard["cohort"]["fps"])
        ):
            raise DashboardInputError("candidate lag-profile timing does not reconcile")
        primary = _finite(
            item["primary_mean_normalized"], "candidate lag-profile primary effect"
        )
        counterpart_raw = item["counterpart_mean_normalized"]
        counterpart = (
            None
            if counterpart_raw is None
            else _finite(counterpart_raw, "candidate lag-profile counterpart effect")
        )
        bootstrap_rate = _finite(
            item["worm_bootstrap_peak_lag_rate"],
            "candidate lag-profile bootstrap peak rate",
        )
        selectivity = _finite(
            item["top_vs_second_lag_selectivity"],
            "candidate lag-profile selectivity",
        )
        if not 0 <= bootstrap_rate <= 1 or not 0 <= selectivity <= 1 + 1e-8:
            raise DashboardInputError("candidate lag-profile stability values are out of range")
        agreement_raw = item["signed_lag_profile_spearman"]
        agreement = (
            None
            if agreement_raw is None
            else _finite(agreement_raw, "candidate lag-profile agreement")
        )
        if agreement is not None and not -1 <= agreement <= 1:
            raise DashboardInputError("candidate lag-profile agreement lies outside [-1,1]")
        interpretation = str(item["interpretation"])
        if "not a physical delay" not in interpretation:
            raise DashboardInputError("candidate lag-profile lacks the physical-delay boundary")
        source = str(item["source_neuron"])
        target = str(item["target_neuron"])
        rows.append(
            {
                "queue_rank": rank,
                "candidate_label": f"#{rank} {source} → {target}",
                "source_neuron": source,
                "target_neuron": target,
                "channel": str(item["channel"]),
                "context": str(item["context"]),
                "horizon_frames": horizon,
                "source_lag_frames": lag,
                "source_to_cut_seconds": lag_seconds,
                "lag_label": f"ℓ={lag} ({lag_seconds:g} s to cut)",
                "primary_method": str(item["primary_method"]),
                "counterpart_method": str(item["counterpart_method"]),
                "primary_mean_normalized": primary,
                "counterpart_mean_normalized": counterpart,
                "is_primary_peak_abs_lag": bool(item["is_primary_peak_abs_lag"]),
                "worm_bootstrap_peak_lag_rate": bootstrap_rate,
                "top_vs_second_lag_selectivity": selectivity,
                "signed_lag_profile_spearman": agreement,
                "interpretation": interpretation,
            }
        )
    if not rows:
        raise DashboardInputError("candidate_lag_profiles has no rows for top candidates")
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rank[int(row["queue_rank"])].append(row)
    if set(by_rank) != retained_ranks:
        raise DashboardInputError("candidate lag profiles do not cover every retained top candidate")
    for rank, group in by_rank.items():
        lags = [int(row["source_lag_frames"]) for row in group]
        if sorted(lags) != sorted(expected_lags):
            raise DashboardInputError(f"candidate lag profile {rank} has an incomplete lag grid")
        if sum(bool(row["is_primary_peak_abs_lag"]) for row in group) != 1:
            raise DashboardInputError(f"candidate lag profile {rank} lacks one unique peak")
        flagged = next(row for row in group if row["is_primary_peak_abs_lag"])
        maximum = max(abs(float(row["primary_mean_normalized"])) for row in group)
        if not np.isclose(abs(float(flagged["primary_mean_normalized"])), maximum):
            raise DashboardInputError(f"candidate lag profile {rank} peak flag is inconsistent")
    return sorted(rows, key=lambda row: (int(row["queue_rank"]), int(row["source_lag_frames"])))


def _external_rows(
    path: Path | None,
    max_rows: int,
    *,
    atlas_dir: Path,
    atlas_checksums: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if path is None:
        return [], []
    path = Path(path).resolve()
    if path.name != "dashboard_external_summary.csv":
        raise DashboardInputError(
            "external-reference must be the canonical "
            "dashboard_external_summary.csv from a post-freeze analysis bundle"
        )
    external_dir = path.parent
    _verify_checksum_bundle(
        external_dir,
        required=(
            "manifest.json",
            "dashboard_external_summary.csv",
            "input_checksums.sha256",
        ),
        label="post-freeze external analysis",
    )
    manifest = _read_json(external_dir / "manifest.json")
    firewall = manifest.get("internal_firewall")
    expected_firewall = {
        "internal_manifest_sha256": atlas_checksums.get("manifest.json"),
        "internal_protocol_sha256": atlas_checksums.get("protocol.json"),
        "internal_validation_sha256": atlas_checksums.get("validation.json"),
        "internal_atlas_matrices_sha256": atlas_checksums.get("atlas_matrices.npz"),
        "internal_checksums_sha256": sha256(atlas_dir / "checksums.sha256"),
        "frozen_hypothesis_queue_sha256": atlas_checksums.get(
            "hypothesis_queue.csv"
        ),
    }
    if (
        any(value is None for value in expected_firewall.values())
        or manifest.get("status") != "passed"
        or manifest.get("analysis_role")
        != "post_freeze_external_convergence_only"
        or not str(manifest.get("ranking_effect", "")).startswith("none")
        or manifest.get("internal_firewall_reverified_after_analysis") is not True
        or manifest.get("internal_prediction_ranking_inputs") != []
        or not isinstance(firewall, Mapping)
        or any(str(firewall.get(key)) != str(value) for key, value in expected_firewall.items())
    ):
        raise DashboardInputError(
            "external-reference bundle is not linked to this frozen atlas"
        )
    input_ledger: dict[str, str] = {}
    for line_number, line in enumerate(
        (external_dir / "input_checksums.sha256").read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise DashboardInputError(
                f"external input checksum line {line_number} is malformed"
            )
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise DashboardInputError(
                "external input checksum inventory has an unsafe path"
            )
        name = relative.as_posix()
        if name in input_ledger:
            raise DashboardInputError(
                f"external input checksum inventory duplicates {name}"
            )
        input_ledger[name] = parts[0]
    expected_internal_inputs = {
        "internal_atlas/atlas_matrices.npz": str(
            expected_firewall["internal_atlas_matrices_sha256"]
        ),
        "internal_atlas/checksums.sha256": str(
            expected_firewall["internal_checksums_sha256"]
        ),
        "internal_atlas/hypothesis_queue.csv": str(
            expected_firewall["frozen_hypothesis_queue_sha256"]
        ),
    }
    if any(input_ledger.get(key) != value for key, value in expected_internal_inputs.items()):
        raise DashboardInputError(
            "external input checksum inventory does not bind this frozen atlas"
        )
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise DashboardInputError(f"could not read external-reference CSV: {error}") from error
    if frame.empty:
        raise DashboardInputError("external-reference CSV is empty")
    columns = list(frame.columns[:MAX_EXTERNAL_COLUMNS])
    if any(UNSAFE_FIELD.fullmatch(str(column)) for column in columns):
        raise DashboardInputError("external-reference CSV contains an unsafe field name")
    frame = frame.loc[:, columns].iloc[:max_rows]
    rows = [
        {str(column): _json_scalar(value) for column, value in source.items()}
        for source in frame.to_dict(orient="records")
    ]
    specs: list[dict[str, str]] = []
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        is_numeric = bool(frame[column].notna().any() and numeric[frame[column].notna()].notna().all())
        specs.append(
            {
                "field": str(column),
                "label": str(column).replace("_", " ").title(),
                "type": "number" if is_numeric else "text",
            }
        )
    return rows, specs


def _verify_checksum_bundle(
    directory: Path,
    *,
    required: Iterable[str],
    label: str,
) -> dict[str, str]:
    """Verify every safely named file declared by one canonical bundle."""

    path = directory / "checksums.sha256"
    if not path.is_file():
        raise DashboardInputError(f"{label} lacks checksums.sha256")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise DashboardInputError(
                f"{label} checksum line {line_number} is malformed"
            )
        relative = Path(parts[1])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 1
            or not relative.name
        ):
            raise DashboardInputError(f"{label} checksum inventory has an unsafe path")
        if relative.name in entries:
            raise DashboardInputError(
                f"{label} checksum inventory duplicates {relative.name}"
            )
        entries[relative.name] = parts[0]
    missing = sorted(set(required).difference(entries))
    if missing:
        raise DashboardInputError(f"{label} checksums omit required files: {missing}")
    for name, expected in entries.items():
        source = directory / name
        if not source.is_file() or sha256(source) != expected:
            raise DashboardInputError(f"{label} checksum failed for {name}")
    return entries


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise DashboardInputError(f"{label} must be a canonical boolean")


def _linked_path(value: Any, *, base: Path, root: Path, label: str) -> Path:
    if value is None or not str(value).strip():
        raise DashboardInputError(f"targeted manifest lacks {label}")
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    _safe_relative(resolved, root)
    return resolved


def _targeted_reference_rows(
    targeted_dir: Path | None,
    *,
    atlas_dir: Path,
    atlas_checksums: Mapping[str, str],
    canonical_queue: pd.DataFrame,
    provenance_root: Path,
    fps: float,
) -> dict[str, Any] | None:
    """Load a bounded, queue-linked N128 sensitivity layer.

    This function deliberately returns comparison datasets only.  Nothing it
    reads is passed into candidate ranking, defaults, or evidence-tier logic.
    """

    if targeted_dir is None:
        return None
    targeted = Path(targeted_dir).resolve()
    if not targeted.is_dir():
        raise DashboardInputError("targeted-reference must be an analysis directory")
    _safe_relative(targeted, provenance_root)
    checksums = _verify_checksum_bundle(
        targeted,
        required=TARGETED_REQUIRED_FILES,
        label="targeted analysis",
    )
    manifest = _read_json(targeted / "manifest.json")
    validation = _read_json(targeted / "validation.json")
    protocol = _read_json(targeted / "protocol.json")
    summary = _read_json(targeted / "summary.json")
    expected_artifacts = {
        "targeted_cells": "targeted_cells.csv",
        "screen_consistency": "screen_consistency.csv",
        "summary": "summary.json",
        "validation": "validation.json",
        "protocol": "protocol.json",
        "input_checksums": "input_checksums.csv",
    }
    if (
        manifest.get("status") != "complete"
        or manifest.get("analysis_schema_version") != TARGETED_ANALYSIS_SCHEMA
        or manifest.get("matrix_orientation") != ORIENTATION
        or not isinstance(manifest.get("artifacts"), dict)
        or any(
            manifest["artifacts"].get(key) != value
            for key, value in expected_artifacts.items()
        )
    ):
        raise DashboardInputError("targeted analysis manifest is incomplete or incompatible")
    if (
        validation.get("status") != "passed"
        or validation.get("analysis_schema_version") != TARGETED_ANALYSIS_SCHEMA
        or _integer(validation.get("n_worms"), "targeted validation worms") != 17
    ):
        raise DashboardInputError("targeted analysis validation did not pass")
    targeted_checks = validation.get("checks")
    if not isinstance(targeted_checks, dict) or not targeted_checks or not all(
        value is True for value in targeted_checks.values()
    ):
        raise DashboardInputError("targeted analysis checks are absent or incomplete")
    if targeted_checks.get("external_reference_firewall") is not True:
        raise DashboardInputError("targeted analysis external-reference firewall is absent")
    if (
        protocol.get("analysis_schema_version") != TARGETED_ANALYSIS_SCHEMA
        or protocol.get("method") != TARGETED_METHOD
        or _integer(protocol.get("particle_count"), "targeted particle count") != 128
        or protocol.get("matrix_orientation") != ORIENTATION
        or tuple(protocol.get("contrasts", ())) != TARGETED_CONTRASTS
    ):
        raise DashboardInputError("targeted N128 protocol is incompatible")
    factual_definition = str(protocol.get("factual_arm_definition", ""))
    if "rollout" not in factual_definition.lower() or "observed history" not in (
        factual_definition.lower()
    ):
        raise DashboardInputError("targeted factual arm is not defined as a model rollout")
    targeted_boundary = str(protocol.get("claim_boundary", "")).lower()
    if "not causal" not in targeted_boundary or "physical delay" not in (
        targeted_boundary.replace("-", " ")
    ):
        raise DashboardInputError("targeted protocol lost its causal/timing boundary")
    firewall = str(protocol.get("external_reference_firewall", "")).lower()
    if "no connectome" not in firewall or "sbtg" not in firewall:
        raise DashboardInputError("targeted protocol external-reference firewall is incomplete")
    if (
        summary.get("status") != "reviewed_model_relative"
        or summary.get("analysis_schema_version") != TARGETED_ANALYSIS_SCHEMA
        or _integer(summary.get("n_worms"), "targeted summary worms") != 17
        or summary.get("experimental_confirmation_label_assigned") is not False
        or summary.get("external_reference_data_used") is not False
    ):
        raise DashboardInputError("targeted summary is not a reviewed internal sensitivity")

    # Resolve the immutable selection bundle that linked the canonical queue to
    # the targeted run.  All linked paths must remain inside provenance_root.
    selection_dir = _linked_path(
        manifest.get("screen_dir"),
        base=targeted,
        root=provenance_root,
        label="screen_dir",
    )
    selection_checksums = _verify_checksum_bundle(
        selection_dir,
        required=(
            "manifest.json",
            "hypothesis_queue.csv",
            "atlas_matrices.npz",
            "selection_rationale.md",
        ),
        label="targeted selection",
    )
    selection_manifest = _read_json(selection_dir / "manifest.json")
    canonical_queue_hash = atlas_checksums.get("hypothesis_queue.csv")
    if not canonical_queue_hash:
        raise DashboardInputError("canonical atlas checksum inventory lacks its queue")
    canonical_link_hashes: dict[str, str] = {}
    for name in (
        "manifest.json",
        "protocol.json",
        "validation.json",
        "atlas_matrices.npz",
    ):
        expected = atlas_checksums.get(name)
        candidate = atlas_dir / name
        if expected is None or not candidate.is_file() or sha256(candidate) != expected:
            raise DashboardInputError(
                f"canonical atlas checksum inventory does not bind {name}"
            )
        canonical_link_hashes[name] = expected
    selection_queue = selection_dir / "hypothesis_queue.csv"
    selection_dense = selection_dir / "atlas_matrices.npz"
    selection_queue_hash = sha256(selection_queue)
    selection_maximum = _integer(
        selection_manifest.get("maximum_rows"), "selection maximum rows"
    )
    linked_source_atlas = _linked_path(
        selection_manifest.get("source_atlas"),
        base=selection_dir,
        root=provenance_root,
        label="selection source_atlas",
    )
    if (
        selection_manifest.get("schema_version") != TARGETED_SELECTION_SCHEMA
        or selection_manifest.get("status") != "complete"
        or not 1 <= selection_maximum <= MAX_TARGETED_CANDIDATES
        or _integer(selection_manifest.get("particles"), "selection particles") != 128
        or selection_manifest.get("confirmation_method") != TARGETED_METHOD
        or selection_manifest.get("screen_dir_compatible") is not True
        or linked_source_atlas != atlas_dir.resolve()
        or selection_manifest.get("source_manifest_sha256")
        != canonical_link_hashes["manifest.json"]
        or selection_manifest.get("source_protocol_sha256")
        != canonical_link_hashes["protocol.json"]
        or selection_manifest.get("source_validation_sha256")
        != canonical_link_hashes["validation.json"]
        or selection_manifest.get("source_atlas_matrices_sha256")
        != canonical_link_hashes["atlas_matrices.npz"]
        or sha256(selection_dense) != canonical_link_hashes["atlas_matrices.npz"]
        or selection_manifest.get("frozen_hypothesis_queue_sha256")
        != canonical_queue_hash
        or selection_manifest.get("selection_sha256") != selection_queue_hash
        or selection_checksums.get("hypothesis_queue.csv") != selection_queue_hash
        or selection_manifest.get("external_reference_inputs_used") != []
        or selection_manifest.get("external_fields_detected") != []
    ):
        raise DashboardInputError("targeted selection is not linked to the canonical queue")

    canonical_text = pd.read_csv(
        atlas_dir / "hypothesis_queue.csv", dtype=str, keep_default_na=False
    )
    selected_text = pd.read_csv(selection_queue, dtype=str, keep_default_na=False)
    expected_columns = list(canonical_text.columns) + ["run_confirmation"]
    if list(selected_text.columns) != expected_columns:
        raise DashboardInputError(
            "targeted selection queue must preserve canonical columns plus run_confirmation"
        )
    if not 1 <= len(selected_text) <= MAX_TARGETED_CANDIDATES:
        raise DashboardInputError("targeted selection must contain 1..6 rows")
    if not all(
        _strict_bool(value, "run_confirmation")
        for value in selected_text["run_confirmation"]
    ):
        raise DashboardInputError("targeted selection contains an unselected row")
    if canonical_text["queue_rank"].duplicated().any() or selected_text[
        "queue_rank"
    ].duplicated().any():
        raise DashboardInputError("targeted/canonical queue ranks are duplicated")
    canonical_by_rank = canonical_text.set_index("queue_rank", drop=False)
    for _, selected_row in selected_text.iterrows():
        rank = str(selected_row["queue_rank"])
        if rank not in canonical_by_rank.index:
            raise DashboardInputError("targeted queue rank is absent from the canonical queue")
        canonical_row = canonical_by_rank.loc[rank]
        if isinstance(canonical_row, pd.DataFrame):
            raise DashboardInputError("canonical queue rank is not unique")
        for column in canonical_text.columns:
            if str(selected_row[column]) != str(canonical_row[column]):
                raise DashboardInputError(
                    f"targeted queue row differs from canonical queue ({column})"
                )
    if (
        _integer(selection_manifest.get("selected_rows"), "selection row count")
        != len(selected_text)
        or len(selected_text) > selection_maximum
        or _integer(summary.get("n_candidates"), "targeted candidate count")
        != len(selected_text)
    ):
        raise DashboardInputError("targeted candidate counts do not reconcile")

    # Confirm the analysis input manifest used this exact staged queue.
    run_dir = _linked_path(
        manifest.get("input_run_dir"),
        base=targeted,
        root=provenance_root,
        label="input_run_dir",
    )
    run_manifest_path = run_dir / "manifest.json"
    if not run_manifest_path.is_file() or sha256(run_manifest_path) != manifest.get(
        "input_run_manifest_sha256"
    ):
        raise DashboardInputError("targeted input-run manifest hash does not reconcile")
    run_manifest = _read_json(run_manifest_path)
    run_queue = _linked_path(
        run_manifest.get("hypothesis_queue"),
        base=run_dir,
        root=provenance_root,
        label="targeted run hypothesis_queue",
    )
    if (
        run_queue != selection_queue.resolve()
        or run_manifest.get("hypothesis_queue_sha256") != selection_queue_hash
        or manifest.get("input_queue_sha256") != selection_queue_hash
        or run_manifest.get("method") != TARGETED_METHOD
        or _integer(run_manifest.get("particles"), "targeted run particles") != 128
        or run_manifest.get("matrix_orientation") != ORIENTATION
        or run_manifest.get("stimulus_generator_encoding") != "binary_any_stimulus"
        or run_manifest.get("chemical_identity_conditioned") is not False
    ):
        raise DashboardInputError("targeted run is not linked to the staged N128 queue")
    try:
        input_checksums = pd.read_csv(targeted / "input_checksums.csv")
    except (OSError, pd.errors.ParserError) as error:
        raise DashboardInputError(f"could not read targeted input checksums: {error}") from error
    if not {"path", "sha256", "size_bytes"}.issubset(input_checksums.columns):
        raise DashboardInputError("targeted input checksum table is malformed")
    linked_hashes: dict[Path, str] = {}
    for _, item in input_checksums.iterrows():
        linked = _linked_path(
            item["path"], base=targeted, root=provenance_root, label="input checksum path"
        )
        if linked in linked_hashes:
            raise DashboardInputError("targeted input checksum table duplicates a path")
        linked_hashes[linked] = str(item["sha256"])
    if (
        linked_hashes.get(selection_queue.resolve()) != selection_queue_hash
        or linked_hashes.get(run_manifest_path.resolve()) != sha256(run_manifest_path)
        or linked_hashes.get(selection_dense.resolve()) != sha256(selection_dense)
    ):
        raise DashboardInputError("targeted input checksum linkage is incomplete")

    try:
        cells = pd.read_csv(targeted / "targeted_cells.csv")
        screen = pd.read_csv(targeted / "screen_consistency.csv")
    except (OSError, pd.errors.ParserError) as error:
        raise DashboardInputError(f"could not read targeted comparison tables: {error}") from error
    cell_required = {
        "candidate_id",
        "queue_rank",
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "source_to_cut_seconds",
        "horizon_frames",
        "forecast_horizon_seconds",
        "source_to_readout_seconds",
        "context",
        "contrast",
        "n_worms",
        "n_particles",
        "valid_fraction",
        "matrix_orientation",
        "row_axis",
        "column_axis",
        "chemical_identity_conditioned",
        "factual_arm_definition",
        "interpretation_limit",
        "metric",
        "signed",
        "mean_normalized",
        "ci_2_5",
        "ci_97_5",
        "evidence_label",
    }
    if cells.empty or cell_required.difference(cells.columns):
        raise DashboardInputError("targeted_cells.csv has an incompatible schema")
    if len(cells) != _integer(summary.get("n_cell_rows"), "targeted cell rows"):
        raise DashboardInputError("targeted cell row count differs from summary")
    expected_candidate_ids = [
        f"targeted_{index + 1:04d}" for index in range(len(selected_text))
    ]
    if set(cells["candidate_id"].astype(str)) != set(expected_candidate_ids):
        raise DashboardInputError("targeted cells do not cover the selected candidates")
    for _, row in cells.iterrows():
        signed = _strict_bool(row["signed"], "targeted signed")
        expected_signed = str(row["metric"]) != "endpoint_wasserstein1" or str(
            row["context"]
        ).endswith("minus_baseline")
        for field in (
            "source_lag_frames",
            "source_to_cut_seconds",
            "horizon_frames",
            "forecast_horizon_seconds",
            "source_to_readout_seconds",
            "n_worms",
            "n_particles",
            "valid_fraction",
            "mean_normalized",
            "ci_2_5",
            "ci_97_5",
        ):
            _finite(row[field], f"targeted {field}")
        if (
            int(row["n_worms"]) != 17
            or int(row["n_particles"]) != 128
            or not 0 <= float(row["valid_fraction"]) <= 1
            or str(row["matrix_orientation"]) != ORIENTATION
            or str(row["row_axis"]) != "target_neuron"
            or str(row["column_axis"]) != "source_neuron"
            or _strict_bool(
                row["chemical_identity_conditioned"],
                "targeted chemical_identity_conditioned",
            )
            or str(row["factual_arm_definition"]) != factual_definition
            or str(row["evidence_label"]) not in TARGETED_EVIDENCE_LABELS
            or signed != expected_signed
        ):
            raise DashboardInputError("targeted cell metadata is inconsistent")
        interpretation = str(row["interpretation_limit"]).lower().replace("-", " ")
        if "not causal" not in interpretation or "physical delay" not in interpretation:
            raise DashboardInputError("targeted cell lost the scientific claim boundary")

    sensitivity_rows: list[dict[str, Any]] = []
    high_low_lookup: dict[str, float] = {}
    selected_by_candidate: dict[str, pd.Series] = {}
    contrast_labels = {
        "high_low": "high repaired − low repaired",
        "high_factual": "high repaired − factual model rollout",
        "low_factual": "low repaired − factual model rollout",
    }
    for index, (_, selected_row) in enumerate(selected_text.iterrows()):
        candidate_id = expected_candidate_ids[index]
        selected_by_candidate[candidate_id] = selected_row
        channel = str(selected_row["channel"])
        targeted_metric = SCREEN_TO_TARGETED_METRIC.get(channel)
        if targeted_metric is None:
            raise DashboardInputError("selected screen channel lacks a targeted metric")
        candidate_rows = cells[
            (cells["candidate_id"].astype(str) == candidate_id)
            & (cells["metric"].astype(str) == targeted_metric)
        ]
        for contrast in TARGETED_CONTRASTS:
            matching = candidate_rows[
                candidate_rows["contrast"].astype(str) == contrast
            ]
            if len(matching) != 1:
                raise DashboardInputError(
                    "targeted sensitivity lacks one row per selected contrast/metric"
                )
            row = matching.iloc[0]
            for field in (
                "queue_rank",
                "source_neuron",
                "target_neuron",
                "source_index",
                "target_index",
                "source_lag_frames",
                "horizon_frames",
                "context",
            ):
                if str(row[field]) != str(selected_row[field]):
                    raise DashboardInputError(
                        f"targeted candidate differs from staged queue ({field})"
                    )
            lag = int(row["source_lag_frames"])
            horizon = int(row["horizon_frames"])
            if not (
                np.isclose(float(row["source_to_cut_seconds"]), lag / fps)
                and np.isclose(float(row["forecast_horizon_seconds"]), horizon / fps)
                and np.isclose(
                    float(row["source_to_readout_seconds"]), (lag + horizon) / fps
                )
            ):
                raise DashboardInputError("targeted candidate timing does not reconcile")
            mean_value = float(row["mean_normalized"])
            if contrast == "high_low":
                high_low_lookup[candidate_id] = mean_value
            sensitivity_rows.append(
                {
                    "candidate_id": candidate_id,
                    "queue_rank": int(row["queue_rank"]),
                    "candidate_label": (
                        f"#{int(row['queue_rank'])} {row['source_neuron']} → "
                        f"{row['target_neuron']}"
                    ),
                    "source_neuron": str(row["source_neuron"]),
                    "target_neuron": str(row["target_neuron"]),
                    "screen_channel": channel,
                    "targeted_metric": targeted_metric,
                    "signed": _strict_bool(row["signed"], "targeted signed"),
                    "context": str(row["context"]),
                    "source_lag_frames": lag,
                    "horizon_frames": horizon,
                    "timing_label": (
                        f"to cut {lag / fps:g} s · forecast {horizon / fps:g} s · "
                        f"source-to-readout {(lag + horizon) / fps:g} s"
                    ),
                    "contrast": contrast,
                    "contrast_label": contrast_labels[contrast],
                    "contrast_involves_factual_model_rollout": contrast != "high_low",
                    "mean_normalized": mean_value,
                    "ci_2_5": float(row["ci_2_5"]),
                    "ci_97_5": float(row["ci_97_5"]),
                    "valid_fraction": float(row["valid_fraction"]),
                    "n_worms": 17,
                    "n_particles": 128,
                    "n128_internal_label": str(row["evidence_label"]),
                    "factual_arm_definition": factual_definition,
                    "factual_arm_scope": (
                        "learned-flow rollout from observed history; model-generated, "
                        "not an observed response or causal intervention"
                    ),
                    "interpretation_limit": str(row["interpretation_limit"]),
                }
            )
    if len(sensitivity_rows) > MAX_TARGETED_SENSITIVITY_ROWS:
        raise DashboardInputError("targeted sensitivity table exceeds its bound")

    screen_required = {
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
    screen_summary = summary.get("screen_vs_n128")
    validation_screen = validation.get("screen_vs_n128")
    if (
        screen.empty
        or screen_required.difference(screen.columns)
        or not isinstance(screen_summary, dict)
        or screen_summary.get("status") != "computed"
        or not isinstance(validation_screen, dict)
        or validation_screen.get("status") != "computed"
        or _integer(screen_summary.get("rows"), "targeted screen rows") != len(screen)
        or _integer(validation_screen.get("rows"), "validation screen rows")
        != len(screen)
        or len(screen) != len(selected_text)
    ):
        raise DashboardInputError("targeted screen consistency is absent or incomplete")
    if set(screen["candidate_id"].astype(str)) != set(expected_candidate_ids):
        raise DashboardInputError("screen consistency does not cover every selected cell")
    screen_rows: list[dict[str, Any]] = []
    canonical_numeric = canonical_queue.set_index("queue_rank", drop=False)
    for _, row in screen.iterrows():
        candidate_id = str(row["candidate_id"])
        selected_row = selected_by_candidate[candidate_id]
        for field in (
            "source_neuron",
            "target_neuron",
            "source_lag_frames",
            "horizon_frames",
            "context",
        ):
            if str(row[field]) != str(selected_row[field]):
                raise DashboardInputError(
                    f"screen consistency differs from selected queue ({field})"
                )
        screen_value = _finite(row["screen_mean_normalized"], "screen effect")
        targeted_value = _finite(
            row["targeted_n128_mean_normalized"], "targeted N128 effect"
        )
        delta = _finite(row["targeted_minus_screen"], "targeted minus screen")
        magnitude = _finite(row["magnitude_agreement"], "magnitude agreement")
        screen_valid = _finite(row["screen_valid_fraction"], "screen validity")
        targeted_valid = _finite(
            row["targeted_valid_fraction"], "targeted validity"
        )
        if not np.isclose(targeted_value - screen_value, delta):
            raise DashboardInputError("targeted-minus-screen identity failed")
        if not np.isclose(targeted_value, high_low_lookup[candidate_id]):
            raise DashboardInputError("screen table N128 value differs from targeted cells")
        queue_rank = int(selected_row["queue_rank"])
        if queue_rank not in canonical_numeric.index or not np.isclose(
            screen_value,
            float(canonical_numeric.loc[queue_rank, "mean_normalized"]),
        ):
            raise DashboardInputError("screen effect differs from the canonical queue")
        expected_metric = SCREEN_TO_TARGETED_METRIC[str(selected_row["channel"])]
        screen_method = str(row["screen_method"])
        same_sampler_family = _strict_bool(
            row["same_sampler_family"], "same_sampler_family"
        )
        expected_same_sampler = screen_method == PRIMARY_METHOD
        expected_comparison_type = (
            "particle_escalation_within_progressive_bridge"
            if expected_same_sampler
            else "cross_sampler_screen_to_progressive_bridge"
        )
        if (
            str(row["screen_channel"]) != str(selected_row["channel"])
            or str(row["targeted_metric"]) != expected_metric
            or screen_method != str(selected_row["method"])
            or str(row["targeted_method"]) != TARGETED_METHOD
            or same_sampler_family != expected_same_sampler
            or str(row["comparison_type"]) != expected_comparison_type
            or str(row["consistency_label"])
            not in {
                "n128_support_limited",
                "direction_discordant",
                "direction_consistent",
                "unsigned_magnitude_descriptive",
            }
            or not 0 <= magnitude <= 1
            or not 0 <= screen_valid <= 1
            or not 0 <= targeted_valid <= 1
        ):
            raise DashboardInputError("screen consistency metadata is inconsistent")
        direction_raw = row["direction_agreement"]
        direction = (
            None
            if pd.isna(direction_raw)
            else _finite(direction_raw, "direction agreement")
        )
        if direction is not None and direction not in (0.0, 1.0):
            raise DashboardInputError("direction agreement must be zero, one, or undefined")
        candidate_label = (
            f"#{queue_rank} {row['source_neuron']} → {row['target_neuron']}"
        )
        screen_rows.append(
            {
                "candidate_id": candidate_id,
                "queue_rank": queue_rank,
                "candidate_label": candidate_label,
                "source_neuron": str(row["source_neuron"]),
                "target_neuron": str(row["target_neuron"]),
                "context": str(row["context"]),
                "source_lag_frames": int(row["source_lag_frames"]),
                "horizon_frames": int(row["horizon_frames"]),
                "screen_method": screen_method,
                "targeted_method": str(row["targeted_method"]),
                "comparison_type": str(row["comparison_type"]),
                "same_sampler_family": same_sampler_family,
                "screen_channel": str(row["screen_channel"]),
                "targeted_metric": str(row["targeted_metric"]),
                "screen_mean_normalized": screen_value,
                "targeted_n128_mean_normalized": targeted_value,
                "targeted_minus_screen": delta,
                "direction_agreement": direction,
                "magnitude_agreement": magnitude,
                "screen_valid_fraction": screen_valid,
                "targeted_valid_fraction": targeted_valid,
                "consistency_label": str(row["consistency_label"]),
                "interpretation_limit": (
                    "post-screen N128 comparison is descriptive, selection-conditioned, "
                    "and never changes atlas rank or evidence tier"
                ),
            }
        )

    return {
        "sensitivity_rows": sensitivity_rows,
        "screen_rows": sorted(screen_rows, key=lambda item: int(item["queue_rank"])),
        "checksums": checksums,
        "selection_checksums": selection_checksums,
        "selection_dir": selection_dir,
        "generated_at": str(validation.get("created_utc", "")),
        "candidate_count": len(selected_text),
        "screen_summary": screen_summary,
        "factual_arm_definition": factual_definition,
        "claim_boundary": str(protocol.get("claim_boundary")),
    }


def _source(
    *,
    source_id: str,
    label: str,
    path: str,
    command: str,
    executed_at: str,
    description: str,
    tables_used: list[str],
    filters: list[str],
    metric_definitions: list[str],
) -> dict[str, Any]:
    # The portable artifact verifier requires a literal SQL source excerpt for
    # every quantitative widget.  These bounded DuckDB statements reproduce
    # the raw local-file read; ``query`` retains the exact Python command that
    # performs the reviewed joins, validation, and snapshot construction.
    sql_path = path.replace("'", "''")
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        sql = f"SELECT * FROM read_csv_auto('{sql_path}', header = true)"
    elif suffix == ".json":
        sql = f"SELECT * FROM read_json_auto('{sql_path}')"
    else:
        sql = f"SELECT '{sql_path}' AS canonical_local_path"
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb_local_file",
            "language": "shell",
            "query": command,
            "sql": sql,
            "description": description,
            "executed_at": executed_at,
            "tables_used": tables_used,
            "filters": filters,
            "metric_definitions": metric_definitions,
        },
    }


def _preferred_filter_defaults(rows: list[dict[str, Any]]) -> dict[str, str]:
    preferences = (
        (PRIMARY_METHOD, PRIMARY_CHANNEL, PRIMARY_CONTEXT),
        (PRIMARY_METHOD, PRIMARY_CHANNEL, None),
        (PRIMARY_METHOD, None, None),
    )
    for method, channel, context in preferences:
        matching = [
            row
            for row in rows
            if row["method"] == method
            and (channel is None or row["channel"] == channel)
            and (context is None or row["context"] == context)
        ]
        if matching:
            first = matching[0]
            return {
                "method": str(first["method"]),
                "channel": str(first["channel"]),
                "context": str(first["context"]),
            }
    first = rows[0]
    return {
        "method": str(first["method"]),
        "channel": str(first["channel"]),
        "context": str(first["context"]),
    }


def build_dashboard_artifact(
    atlas_dir: Path,
    output_path: Path,
    *,
    provenance_root: Path | None = None,
    external_reference: Path | None = None,
    targeted_reference: Path | None = None,
    max_candidates: int = 150,
    max_external_rows: int = 100,
) -> dict[str, Any]:
    """Validate a canonical atlas and atomically write its portable artifact."""
    if not 1 <= max_candidates <= 500:
        raise ValueError("max_candidates must lie in [1,500]")
    if not 1 <= max_external_rows <= 250:
        raise ValueError("max_external_rows must lie in [1,250]")
    root = Path.cwd().resolve() if provenance_root is None else Path(provenance_root).resolve()
    atlas = Path(atlas_dir).resolve()
    output = Path(output_path).resolve()
    external = None if external_reference is None else Path(external_reference).resolve()
    targeted = None if targeted_reference is None else Path(targeted_reference).resolve()
    protected_directories = [atlas]
    if targeted is not None:
        protected_directories.append(targeted)
    if external is not None:
        protected_directories.append(external.parent)
    if any(directory == output or directory in output.parents for directory in protected_directories):
        raise DashboardInputError(
            "dashboard output must be outside canonical atlas, targeted, and external input bundles"
        )
    atlas_relative = _safe_relative(atlas, root)
    output_relative = _safe_relative(output, root)
    external_relative = None if external is None else _safe_relative(external, root)
    targeted_relative = None if targeted is None else _safe_relative(targeted, root)
    for filename in REQUIRED_FILES:
        if not (atlas / filename).exists():
            raise DashboardInputError(f"required dashboard input is missing: {filename}")
    checksums = _verify_checksums(atlas)
    dashboard = _read_json(atlas / "dashboard_snapshot.json")
    validation = _read_json(atlas / "validation.json")
    protocol = _read_json(atlas / "protocol.json")
    models = _read_json(atlas / "models.json")
    n_worms, n_neurons, fps = _validate_bundle_metadata(
        dashboard, validation, protocol, models
    )
    queue = _read_queue(atlas / "hypothesis_queue.csv", n_worms=n_worms, fps=fps)
    _crosscheck_snapshot_candidates(dashboard, queue)
    candidates = _candidate_rows(queue, min(max_candidates, len(queue)))
    defaults = _preferred_filter_defaults(candidates)
    primary_candidates = [
        row
        for row in candidates
        if row["method"] == defaults["method"]
        and row["channel"] == defaults["channel"]
        and row["context"] == defaults["context"]
    ]
    if not primary_candidates:
        raise DashboardInputError("preferred candidate slice is unexpectedly empty")
    agreement = _primary_agreement_profile(dashboard, validation, fps)
    context_rows = _stimulus_context_rows(dashboard, validation)
    matrix_data = _optional_matrix_rows(dashboard)
    lag_profile_rows = _candidate_lag_profile_rows(dashboard)
    lag_profile_chart_rows = (
        [
            row
            for row in lag_profile_rows
            if row["queue_rank"] == lag_profile_rows[0]["queue_rank"]
        ]
        if lag_profile_rows
        else []
    )
    external_rows, external_columns = _external_rows(
        external,
        max_external_rows,
        atlas_dir=atlas,
        atlas_checksums=checksums,
    )
    targeted_data = _targeted_reference_rows(
        targeted,
        atlas_dir=atlas,
        atlas_checksums=checksums,
        canonical_queue=queue,
        provenance_root=root,
        fps=fps,
    )

    summary = dashboard["support_summary"]
    _require_keys(
        summary,
        ("mean_valid_fraction", "sources_ge_minimum", "support_rows"),
        "support_summary",
    )
    mean_valid = _finite(summary["mean_valid_fraction"], "mean_valid_fraction")
    qualified = _integer(summary["sources_ge_minimum"], "sources_ge_minimum")
    support_rows = _integer(summary["support_rows"], "support_rows")
    if not 0 <= mean_valid <= 1 or not 0 <= qualified <= support_rows:
        raise DashboardInputError("support summary values are out of range")
    checks = validation["archive_checks_completed"]
    generated_at = str(validation.get("created_utc", ""))
    if not generated_at:
        raise DashboardInputError("validation.json lacks created_utc")

    arguments = [
        "python",
        "-m",
        "compatibility_neural_benchmark.prediction_atlas_dashboard",
        "--atlas-dir",
        atlas_relative,
        "--output",
        output_relative,
        "--provenance-root",
        ".",
        "--max-candidates",
        str(max_candidates),
        "--max-external-rows",
        str(max_external_rows),
    ]
    if external_relative is not None:
        arguments.extend(("--external-reference", external_relative))
    if targeted_relative is not None:
        arguments.extend(("--targeted-reference", targeted_relative))
    command = " ".join(shlex.quote(value) for value in arguments)
    rel = {name: f"{atlas_relative}/{name}" for name in REQUIRED_FILES}
    source_specs = [
        _source(
            source_id="dashboard_snapshot_source",
            label="Reviewed atlas dashboard snapshot",
            path=rel["dashboard_snapshot.json"],
            command=command,
            executed_at=generated_at,
            description="Loads the compact reviewed atlas snapshot and cohort/support summaries.",
            tables_used=[rel["dashboard_snapshot.json"]],
            filters=["bounded reviewed snapshot; no dense matrix or Parquet embedding"],
            metric_definitions=[
                "Mean valid fraction is the mean sampler-support validity in the reviewed support table.",
                "Cohort size is 17 worms and 54 head-neuron classes.",
            ],
        ),
        _source(
            source_id="hypothesis_queue_source",
            label="Atlas hypothesis queue",
            path=rel["hypothesis_queue.csv"],
            command=command,
            executed_at=generated_at,
            description="Loads rank-ordered model candidates from the atlas-only hypothesis queue.",
            tables_used=[rel["hypothesis_queue.csv"]],
            filters=[
                f"first {len(candidates)} of {len(queue)} reviewed queue rows embedded",
                "ranking excludes connectome and other external-reference data",
            ],
            metric_definitions=[
                "Evidence score combines normalized effect magnitude, worm-bootstrap interval width, sign consistency, sampler validity, and seed agreement.",
                "Evidence tiers are confirmed, supported exploratory, model only, and unsupported; confirmed is reserved for independent experiments.",
            ],
        ),
        _source(
            source_id="validation_source",
            label="Atlas validation audit",
            path=rel["validation.json"],
            command=command,
            executed_at=generated_at,
            description="Loads completed archive checks and direct-versus-progressive diagnostics.",
            tables_used=[rel["validation.json"]],
            filters=[f"{len(checks)} of {len(checks)} required archive checks passed"],
            metric_definitions=[
                "Sampler agreement is the Spearman correlation of off-diagonal target-row/source-column effect matrices.",
            ],
        ),
        _source(
            source_id="protocol_source",
            label="Atlas protocol and claim boundaries",
            path=rel["protocol.json"],
            command=command,
            executed_at=generated_at,
            description="Loads effect, timing, orientation, inference, chemistry, and evidence-tier definitions.",
            tables_used=[rel["protocol.json"]],
            filters=["reviewed neural prediction atlas v1"],
            metric_definitions=[
                "Source-to-cut time is source lag divided by 4 Hz.",
                "Forecast horizon is horizon frames divided by 4 Hz.",
                "Source-to-readout time is source lag plus forecast horizon, divided by 4 Hz.",
            ],
        ),
        _source(
            source_id="models_source",
            label="Generator and sampler model inventory",
            path=rel["models.json"],
            command=command,
            executed_at=generated_at,
            description="Loads generator encoding, sampler methods, folds, seeds, and particle counts.",
            tables_used=[rel["models.json"]],
            filters=["binary-any-stimulus generator; chemical identity not conditioned"],
            metric_definitions=[],
        ),
    ]
    required_source_paths = [rel[name] for name in REQUIRED_FILES[:-1]]
    source_specs.append(
        _source(
            source_id="atlas_bundle_source",
            label="Canonical reviewed atlas bundle",
            path=atlas_relative,
            command=command,
            executed_at=generated_at,
            description="Reconciles reviewed snapshot, queue, validation, protocol, and models files after SHA-256 verification.",
            tables_used=required_source_paths,
            filters=["all required canonical files passed checksums.sha256"],
            metric_definitions=[],
        )
    )
    if external is not None:
        source_specs.append(
            _source(
                source_id="external_reference_source",
                label="Post-freeze external-reference comparison",
                path=external_relative or "",
                command=command,
                executed_at=generated_at,
                description="Loads bounded post-freeze reference rows for comparison only.",
                tables_used=[external_relative or ""],
                filters=[
                    f"first {len(external_rows)} rows and {len(external_columns)} columns embedded",
                    "not used to rank hypothesis-queue candidates or assign evidence tiers",
                ],
                metric_definitions=[],
            )
        )
    if targeted_data is not None:
        targeted_cells_relative = f"{targeted_relative}/targeted_cells.csv"
        targeted_screen_relative = f"{targeted_relative}/screen_consistency.csv"
        targeted_protocol_relative = f"{targeted_relative}/protocol.json"
        targeted_generated_at = str(targeted_data["generated_at"])
        if not targeted_generated_at:
            raise DashboardInputError("targeted validation lacks created_utc")
        source_specs.extend(
            [
                _source(
                    source_id="targeted_cells_source",
                    label="Targeted three-arm N128 sensitivity",
                    path=targeted_cells_relative,
                    command=command,
                    executed_at=targeted_generated_at,
                    description=(
                        "Loads the selected screen metric across low/high/factual "
                        "three-arm contrasts for each frozen candidate."
                    ),
                    tables_used=[targeted_cells_relative],
                    filters=[
                        f"{len(targeted_data['sensitivity_rows'])} bounded rows; at most six candidates × three contrasts",
                        "N=128 progressive bridge; comparison only and never used for atlas ranking or tiers",
                    ],
                    metric_definitions=[
                        "Normalized effect is the selected targeted contrast divided by max(abs(achieved high-low source-gap), 0.10).",
                        "The factual arm is a learned-flow rollout from observed history, not an observed response or causal intervention.",
                    ],
                ),
                _source(
                    source_id="targeted_screen_source",
                    label="Screen versus targeted N128 consistency",
                    path=targeted_screen_relative,
                    command=command,
                    executed_at=targeted_generated_at,
                    description=(
                        "Loads the frozen selected-cell comparison between the original "
                        "screen and N128 targeted high-low response."
                    ),
                    tables_used=[targeted_screen_relative],
                    filters=[
                        f"{len(targeted_data['screen_rows'])} selected cells",
                        "post-screen, selection-conditioned sensitivity only",
                    ],
                    metric_definitions=[
                        "Targeted minus screen equals the N128 high-low normalized effect minus the frozen screen effect.",
                        "Magnitude agreement is 1-|N128-screen|/(|N128|+|screen|+1e-12).",
                    ],
                ),
                _source(
                    source_id="targeted_protocol_source",
                    label="Targeted N128 protocol and claim boundary",
                    path=targeted_protocol_relative,
                    command=command,
                    executed_at=targeted_generated_at,
                    description=(
                        "Loads the N128 three-arm definitions, worm-level inference, "
                        "factual-rollout definition, and scientific boundaries."
                    ),
                    tables_used=[targeted_protocol_relative],
                    filters=["validated internal N128 analysis; external-reference firewall passed"],
                    metric_definitions=[],
                ),
            ]
        )

    datasets: dict[str, list[dict[str, Any]]] = {
        "atlas_summary": [
            {
                "n_worms": n_worms,
                "n_neurons": n_neurons,
                "mean_valid_fraction": mean_valid,
                "qualified_support_rows": qualified,
                "support_rows": support_rows,
                "retained_candidates": len(queue),
                "embedded_candidates": len(candidates),
                "validation_check_fraction": 1.0,
                "archive_count": _integer(validation.get("archive_count"), "archive_count"),
            }
        ],
        "candidate_rows": candidates,
        "primary_candidate_rows": primary_candidates,
        "lag_horizon_profile": agreement,
        "stimulus_context_rows": context_rows,
        "provenance_rows": [
            {
                "artifact": name,
                "role": {
                    "dashboard_snapshot.json": "bounded atlas snapshot",
                    "hypothesis_queue.csv": "read-only queue companion",
                    "validation.json": "archive and statistical audit",
                    "protocol.json": "estimand and claim definitions",
                    "models.json": "generator/sampler inventory",
                }[name],
                "status": "checksum verified",
                "path": rel[name],
                "sha256": checksums[name],
            }
            for name in REQUIRED_FILES[:-1]
        ]
        + (
            [
                {
                    "artifact": "targeted N128 analysis",
                    "role": "post-screen three-arm sensitivity; comparison only",
                    "status": "checksum and queue-link verified",
                    "path": targeted_relative or "",
                    "sha256": targeted_data["checksums"]["manifest.json"],
                },
                {
                    "artifact": "targeted staged queue",
                    "role": "bounded canonical-row selection linked to the frozen atlas queue",
                    "status": "checksum and source-queue hash verified",
                    "path": _safe_relative(
                        targeted_data["selection_dir"] / "hypothesis_queue.csv", root
                    ),
                    "sha256": targeted_data["selection_checksums"][
                        "hypothesis_queue.csv"
                    ],
                },
            ]
            if targeted_data is not None
            else [
                {
                    "artifact": "targeted N128 analysis",
                    "role": "optional post-screen three-arm sensitivity",
                    "status": "optional, not supplied",
                    "path": "",
                    "sha256": "",
                }
            ]
        )
        + [
            {
                "artifact": "post-freeze external reference",
                "role": "comparison only; never ranking or evidence-tier input",
                "status": "loaded" if external is not None else "optional, not supplied",
                "path": external_relative or "",
                "sha256": sha256(external) if external is not None else "",
            }
        ],
    }
    if matrix_data is not None:
        datasets["primary_matrix_slice"] = matrix_data["rows"]
    if lag_profile_rows:
        datasets["candidate_lag_profile_rows"] = lag_profile_rows
        datasets["candidate_lag_profile_chart_rows"] = lag_profile_chart_rows
    if external_rows:
        datasets["external_reference_rows"] = external_rows
    if targeted_data is not None:
        datasets["targeted_sensitivity_rows"] = targeted_data["sensitivity_rows"]
        datasets["targeted_screen_consistency_rows"] = targeted_data["screen_rows"]

    cards = [
        {
            "id": "worms_card",
            "description": "Independent statistical units in the reviewed primary cohort.",
            "dataset": "atlas_summary",
            "sourceId": "dashboard_snapshot_source",
            "metrics": [{"label": "Held-out worms", "field": "n_worms", "format": "number"}],
        },
        {
            "id": "neurons_card",
            "description": "Target-row and source-column neuron classes in each dense matrix.",
            "dataset": "atlas_summary",
            "sourceId": "dashboard_snapshot_source",
            "metrics": [{"label": "Neuron classes", "field": "n_neurons", "format": "number"}],
        },
        {
            "id": "validity_card",
            "description": "Mean sampler-support validity across reviewed source/context/lag rows.",
            "dataset": "atlas_summary",
            "sourceId": "dashboard_snapshot_source",
            "metrics": [{"label": "Mean valid fraction", "field": "mean_valid_fraction", "format": "percent"}],
        },
        {
            "id": "qualified_card",
            "description": "Source/context/lag support rows meeting the protocol minimum.",
            "dataset": "atlas_summary",
            "sourceId": "dashboard_snapshot_source",
            "metrics": [
                {"label": "Support-qualified rows", "field": "qualified_support_rows", "format": "compact"},
                {"label": "All support rows", "field": "support_rows", "format": "compact"},
            ],
        },
        {
            "id": "queue_card",
            "description": "Atlas-only candidates retained in the read-only queue companion.",
            "dataset": "atlas_summary",
            "sourceId": "hypothesis_queue_source",
            "metrics": [{"label": "Retained candidates", "field": "retained_candidates", "format": "compact"}],
        },
        {
            "id": "validation_card",
            "description": "Required canonical archive checks completed successfully.",
            "dataset": "atlas_summary",
            "sourceId": "validation_source",
            "metrics": [{"label": "Audit checks passed", "field": "validation_check_fraction", "format": "percent"}],
        },
    ]

    if matrix_data is not None:
        main_chart = {
            "id": "primary_matrix_chart",
            "title": "Primary normalized endpoint-mean matrix slice",
            "subtitle": (
                f"Target neurons are rows and source neurons are columns; "
                f"ℓ={matrix_data['source_lag_frames']} frames and "
                f"h={matrix_data['horizon_frames']} frames. Cell color follows the "
                "renderer’s min-to-max scale and does not encode a zero point or sign; "
                "hover for the signed value."
            ),
            "intent": "relationship",
            "question": "Which source-column and target-row pairs carry the largest normalized model effects in this reviewed slice?",
            "rationale": "A heatmap preserves the directed 54×54 matrix geometry without embedding other slices.",
            "type": "heatmap",
            "dataset": "primary_matrix_slice",
            "sourceId": "dashboard_snapshot_source",
            "encodings": {
                "x": {
                    "field": "target_neuron",
                    "type": "nominal",
                    "label": "Target neuron (row)",
                },
                "y": {
                    "fields": matrix_data["source_neurons"],
                    "type": "quantitative",
                    "label": "Source neuron (column)",
                },
            },
            "layout": "full",
        }
    else:
        main_chart = {
            "id": "lag_horizon_chart",
            "title": "Sampler agreement by source lag and forecast horizon",
            "subtitle": (
                "Endpoint-mean onset-minus-baseline matrices; source-to-cut and "
                "forecast timing are shown on separate axes."
            ),
            "intent": "relationship",
            "question": "Where do direct importance and progressive bridge SMC give similar directed effect matrices?",
            "rationale": "The bounded heatmap preserves the complete 4×6 lag/horizon grid without embedding dense 54×54 matrices.",
            "comparisonContext": {
                "grain": "source-lag × forecast-horizon cell",
                "normalization": "Spearman correlation across off-diagonal target-row/source-column effects",
                "unit": "correlation",
                "semanticFamily": "sampler agreement",
            },
            "type": "heatmap",
            "dataset": "lag_horizon_profile",
            "sourceId": "validation_source",
            "encodings": {
                "x": {"field": "source_lag_label", "type": "ordinal", "label": "Source-to-cut lag"},
                # The portable chart contract uses y for the numeric heatmap cell
                # value and color for the row/category grouping.
                "y": {"field": "direct_vs_progressive_spearman", "type": "quantitative", "label": "Direct vs progressive Spearman"},
                "color": {"field": "forecast_horizon_label", "type": "ordinal", "label": "Forecast horizon"},
                "tooltip": [
                    {"field": "source_to_readout_seconds", "type": "quantitative", "label": "Source-to-readout seconds"},
                    {"field": "n_worms", "type": "quantitative", "label": "Worms"},
                ],
            },
            "palette": {"kind": "diverging", "midpoint": 0},
            "layout": "full",
        }

    charts = [
        main_chart,
        {
            "id": "candidate_ranking_chart",
            "title": "Ranked candidates in the primary review slice",
            "subtitle": (
                f"{defaults['method']} · {defaults['channel']} · "
                f"{defaults['context']}; external references are excluded."
            ),
            "intent": "comparison",
            "question": "Which retained source-to-target candidates rank highest within the frozen primary review slice?",
            "rationale": "A sorted horizontal bar makes the bounded top candidates directly comparable while retaining long neuron-pair labels.",
            "type": "horizontalBar",
            "dataset": "primary_candidate_rows",
            "sourceId": "hypothesis_queue_source",
            "encodings": {
                "x": {"field": "candidate_label", "type": "nominal", "label": "Candidate"},
                "y": {"field": "evidence_score", "type": "quantitative", "label": "Evidence score"},
                "tooltip": [
                    {"field": "mean_normalized", "type": "quantitative", "label": "Mean normalized effect"},
                    {"field": "valid_fraction", "type": "quantitative", "label": "Valid fraction", "format": "percent"},
                    {"field": "timing_label", "type": "text", "label": "Timing"},
                    {"field": "support_tier", "type": "nominal", "label": "Evidence tier"},
                ],
            },
            "settings": {"sort": "descending", "limit": 12, "showValues": True},
            "layout": "full",
        },
        {
            "id": "stimulus_context_chart",
            "title": "Sampler agreement by stimulus context",
            "subtitle": "Endpoint-mean agreement across lag/horizon cells; chemical contexts are exploratory event strata.",
            "intent": "comparison",
            "question": "How consistently do direct and progressive estimators align across stimulus-defined contexts?",
            "rationale": "A horizontal comparison accommodates the long context labels and keeps chemistry caveats in the source rows.",
            "type": "horizontalBar",
            "dataset": "stimulus_context_rows",
            "sourceId": "validation_source",
            "encodings": {
                "x": {"field": "context_label", "type": "nominal", "label": "Stimulus context"},
                "y": {"field": "mean_direct_vs_progressive_spearman", "type": "quantitative", "label": "Mean sampler Spearman"},
                "tooltip": [
                    {"field": "conditioning_scope", "type": "text", "label": "Conditioning scope"},
                    {"field": "timing_cells_with_finite_agreement", "type": "quantitative", "label": "Finite timing cells"},
                ],
            },
            "settings": {"sort": "descending", "showValues": True},
            "layout": "full",
        },
    ]
    if lag_profile_rows:
        charts.append(
            {
                "id": "candidate_lag_profile_chart",
                "title": "Top-ranked candidate effect across source lag",
                "subtitle": (
                    f"{lag_profile_chart_rows[0]['candidate_label']}; primary-method normalized "
                    "effect. Any peak is descriptive model localization, not a physical delay."
                ),
                "intent": "trend",
                "question": (
                    "How does the top-ranked candidate's model effect vary across the "
                    "prespecified source-to-cut lag grid?"
                ),
                "rationale": (
                    "Bars keep the four prespecified lag bins discrete and avoid implying "
                    "a continuous trajectory or causal delay."
                ),
                "comparisonContext": {
                    "grain": "candidate × source-to-cut lag",
                    "normalization": "worm-aggregated normalized model effect",
                    "unit": "normalized effect",
                    "semanticFamily": "descriptive lag profile",
                },
                "type": "bar",
                "dataset": "candidate_lag_profile_chart_rows",
                "sourceId": "dashboard_snapshot_source",
                "encodings": {
                    "x": {
                        "field": "source_lag_frames",
                        "type": "quantitative",
                        "label": "Source-to-cut lag (frames)",
                    },
                    "y": {
                        "field": "primary_mean_normalized",
                        "type": "quantitative",
                        "label": "Primary normalized effect",
                    },
                    "tooltip": [
                        {
                            "field": "counterpart_mean_normalized",
                            "type": "quantitative",
                            "label": "Counterpart normalized effect",
                        },
                        {
                            "field": "worm_bootstrap_peak_lag_rate",
                            "type": "quantitative",
                            "label": "Worm-bootstrap peak-lag rate",
                            "format": "percent",
                        },
                        {
                            "field": "top_vs_second_lag_selectivity",
                            "type": "quantitative",
                            "label": "Top-vs-second selectivity",
                        },
                        {
                            "field": "interpretation",
                            "type": "text",
                            "label": "Interpretation",
                        },
                    ],
                },
                "settings": {"sort": "ascending", "showValues": True},
                "layout": "full",
            }
        )
    if targeted_data is not None:
        charts.append(
            {
                "id": "targeted_screen_comparison_chart",
                "title": "Screen effect versus targeted N128 effect",
                "subtitle": (
                    "Frozen selected cells; N128 high-low response compared with the "
                    "original screen. Descriptive and selection-conditioned."
                ),
                "intent": "comparison",
                "question": (
                    "How do the selected cells' normalized screen effects compare with "
                    "their targeted N128 high-low effects?"
                ),
                "rationale": (
                    "Paired bars keep the at-most-six selected cells individually "
                    "identifiable and avoid an underpowered scatter."
                ),
                "comparisonContext": {
                    "grain": "one frozen selected source-target cell",
                    "normalization": "effect divided by max(abs(achieved high-low source gap), 0.10)",
                    "unit": "normalized model effect",
                    "semanticFamily": "post-screen N128 sensitivity",
                },
                "type": "horizontalBar",
                "dataset": "targeted_screen_consistency_rows",
                "sourceId": "targeted_screen_source",
                "encodings": {
                    "x": {
                        "field": "candidate_label",
                        "type": "nominal",
                        "label": "Selected source → target cell",
                    },
                    "y": {
                        "fields": [
                            "screen_mean_normalized",
                            "targeted_n128_mean_normalized",
                        ],
                        "type": "quantitative",
                        "label": "Normalized model effect",
                    },
                    "tooltip": [
                        {
                            "field": "candidate_label",
                            "type": "text",
                            "label": "Selected cell",
                        },
                        {
                            "field": "targeted_minus_screen",
                            "type": "quantitative",
                            "label": "N128 minus screen",
                        },
                        {
                            "field": "magnitude_agreement",
                            "type": "quantitative",
                            "label": "Magnitude agreement",
                        },
                        {
                            "field": "interpretation_limit",
                            "type": "text",
                            "label": "Interpretation",
                        },
                    ],
                },
                "palette": {"kind": "categorical"},
                "settings": {"showValues": True},
                "layout": "full",
            }
        )
    candidate_columns = [
        {"field": "queue_rank", "label": "Rank", "format": "number"},
        {"field": "pair_label", "label": "Source → target", "type": "text"},
        {"field": "method", "label": "Method", "type": "text"},
        {"field": "channel_label", "label": "Effect channel", "type": "text"},
        {"field": "context_label", "label": "Stimulus context", "type": "text"},
        {"field": "timing_label", "label": "Timing", "type": "text"},
        {"field": "mean_normalized", "label": "Mean normalized", "format": "number"},
        {"field": "ci_2_5", "label": "CI 2.5%", "format": "number"},
        {"field": "ci_97_5", "label": "CI 97.5%", "format": "number"},
        {"field": "valid_fraction", "label": "Valid fraction", "format": "percent"},
        {"field": "support_tier", "label": "Evidence tier", "type": "text"},
        {"field": "chemistry_scope", "label": "Stimulus scope", "type": "text"},
    ]
    tables = [
        {
            "id": "candidate_table",
            "title": "Candidate evidence and support",
            "subtitle": (
                "Full bounded queue companion across retained metrics and contexts; "
                "worm is the independent unit (n=17)."
            ),
            "dataset": "candidate_rows",
            "sourceId": "hypothesis_queue_source",
            "defaultSort": {"field": "queue_rank", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": candidate_columns,
        },
        {
            "id": "provenance_table",
            "title": "Audit and provenance inventory",
            "subtitle": "Canonical local files, checksum status, and post-freeze reference boundary.",
            "dataset": "provenance_rows",
            "sourceId": "atlas_bundle_source",
            "defaultSort": {"field": "artifact", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "artifact", "label": "Artifact", "type": "text"},
                {"field": "role", "label": "Role", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "path", "label": "Canonical local path", "type": "text"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
            ],
        },
    ]
    if lag_profile_rows:
        tables.append(
            {
                "id": "candidate_lag_profile_table",
                "title": "Candidate lag-profile audit",
                "subtitle": (
                    "Worm-bootstrap peak rates and sampler agreement are descriptive; peak "
                    "lag is not a causal or physical-delay estimate."
                ),
                "dataset": "candidate_lag_profile_rows",
                "sourceId": "dashboard_snapshot_source",
                "defaultSort": {"field": "queue_rank", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "queue_rank", "label": "Rank", "format": "number"},
                    {"field": "candidate_label", "label": "Source → target", "type": "text"},
                    {"field": "lag_label", "label": "Source-to-cut lag", "type": "text"},
                    {
                        "field": "primary_mean_normalized",
                        "label": "Primary effect",
                        "format": "number",
                    },
                    {
                        "field": "counterpart_mean_normalized",
                        "label": "Counterpart effect",
                        "format": "number",
                    },
                    {
                        "field": "worm_bootstrap_peak_lag_rate",
                        "label": "Bootstrap peak rate",
                        "format": "percent",
                    },
                    {"field": "interpretation", "label": "Boundary", "type": "text"},
                ],
            }
        )
    if targeted_data is not None:
        tables.extend(
            [
                {
                    "id": "targeted_screen_consistency_table",
                    "title": "Selected-cell screen and N128 sensitivity",
                    "subtitle": (
                        "Comparison only; selection-conditioned N128 results do not "
                        "change atlas queue rank or evidence tier."
                    ),
                    "dataset": "targeted_screen_consistency_rows",
                    "sourceId": "targeted_screen_source",
                    "defaultSort": {"field": "queue_rank", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "queue_rank", "label": "Atlas rank", "format": "number"},
                        {"field": "candidate_label", "label": "Source → target", "type": "text"},
                        {"field": "screen_channel", "label": "Screen channel", "type": "text"},
                        {"field": "context", "label": "Context", "type": "text"},
                        {"field": "screen_mean_normalized", "label": "Screen", "format": "number"},
                        {"field": "targeted_n128_mean_normalized", "label": "N128 high-low", "format": "number"},
                        {"field": "targeted_minus_screen", "label": "N128 − screen", "format": "number"},
                        {"field": "targeted_valid_fraction", "label": "N128 valid fraction", "format": "percent"},
                        {"field": "consistency_label", "label": "Consistency", "type": "text"},
                    ],
                },
                {
                    "id": "targeted_three_arm_table",
                    "title": "Targeted N128 three-arm sensitivity",
                    "subtitle": (
                        "The factual arm is a learned-flow model rollout from observed "
                        "history—not an observed response or causal intervention."
                    ),
                    "dataset": "targeted_sensitivity_rows",
                    "sourceId": "targeted_cells_source",
                    "defaultSort": {"field": "queue_rank", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "queue_rank", "label": "Atlas rank", "format": "number"},
                        {"field": "candidate_label", "label": "Source → target", "type": "text"},
                        {"field": "screen_channel", "label": "Selected channel", "type": "text"},
                        {"field": "contrast_label", "label": "N128 contrast", "type": "text"},
                        {"field": "mean_normalized", "label": "Normalized effect", "format": "number"},
                        {"field": "ci_2_5", "label": "CI 2.5%", "format": "number"},
                        {"field": "ci_97_5", "label": "CI 97.5%", "format": "number"},
                        {"field": "valid_fraction", "label": "Valid fraction", "format": "percent"},
                        {"field": "n128_internal_label", "label": "N128 internal label", "type": "text"},
                        {"field": "factual_arm_scope", "label": "Factual-arm boundary", "type": "text"},
                    ],
                },
            ]
        )
    if external_rows:
        tables.append(
            {
                "id": "external_reference_table",
                "title": "Post-freeze external-reference rows",
                "subtitle": "Comparison only; never used for queue ranking or evidence-tier assignment.",
                "dataset": "external_reference_rows",
                "sourceId": "external_reference_source",
                "defaultSort": {"field": external_columns[0]["field"], "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": external_columns,
            }
        )

    # The portable dashboard reader exposes dashboard filters only when every
    # visible card, chart, and table dataset can be targeted by the same
    # control.  Candidate-only controls are therefore intentionally omitted:
    # declaring them would produce an empty toolbar and would also leave the
    # supposedly filtered ranking chart mixed across estimands.  The companion
    # standalone atlas explorer owns cross-dataset selectors; this bounded
    # portable summary freezes the ranking chart to one explicit primary slice.
    filters: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = [
        {
            "id": "metrics_block",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
        },
        {
            "id": "reading_guide",
            "type": "markdown",
            "sourceId": "protocol_source",
            "body": (
                "## How to read the atlas\n\n"
                "- **Matrix orientation:** target neurons are rows; soft-repaired source-history neurons are columns.\n"
                "- **Timing:** source-to-cut lag, forecast horizon, and source-to-readout time are distinct quantities.\n"
                "- **Normalization:** each event response is divided by `max(abs(achieved high-low source gap), 0.10)` before worm aggregation; Wasserstein-1 is unsigned within a state, while a between-phase difference can be signed.\n"
                "- **Stimulus model:** the generator uses a binary any-stimulus history. Chemical panels are exploratory event-stratified views, not chemically conditioned effects.\n"
                "- **Claim boundary:** effects are model-relative lag associations, not causal interventions or physical transmission delays.\n"
                "- **Candidate peak lag:** localization and worm-bootstrap peak rates are descriptive stability summaries, not physical delays.\n"
                "- **Inference:** worms are independent units (n=17); events, particles, horizons, and seeds are not treated as independent worms."
            ),
        },
        {
            "id": "portable_scope",
            "type": "markdown",
            "body": (
                "## Portable summary scope\n\n"
                "This self-contained dashboard is a bounded reviewed summary, not the "
                "full interactive atlas explorer. Its ranking chart is frozen to "
                f"`{defaults['method']} / {defaults['channel']} / {defaults['context']}` "
                "so methods, effect definitions, and stimulus strata are never mixed "
                "behind a nonfunctional control. The candidate table retains the full "
                "bounded queue for exact lookup."
            ),
        },
        {"id": "main_chart_block", "type": "chart", "chartId": main_chart["id"]},
        {
            "id": "candidate_ranking_block",
            "type": "chart",
            "chartId": "candidate_ranking_chart",
        },
        {"id": "candidate_table_block", "type": "table", "tableId": "candidate_table"},
        {
            "id": "stimulus_context_block",
            "type": "chart",
            "chartId": "stimulus_context_chart",
        },
        {
            "id": "evidence_tiers",
            "type": "markdown",
            "sourceId": "protocol_source",
            "body": (
                "## Evidence tiers\n\n"
                "**Confirmed** is reserved for independent experimental confirmation. "
                "**Supported exploratory** passes the worm-level multiplicity, stability, and support gates. "
                "**Model only** retains a model estimate without all evidence gates. "
                "**Unsupported** fails minimum support or lacks a defined inferential comparison."
            ),
        },
        {"id": "provenance_block", "type": "table", "tableId": "provenance_table"},
    ]
    if lag_profile_rows:
        blocks[5:5] = [
            {
                "id": "candidate_lag_profile_block",
                "type": "chart",
                "chartId": "candidate_lag_profile_chart",
            },
            {
                "id": "candidate_lag_profile_table_block",
                "type": "table",
                "tableId": "candidate_lag_profile_table",
            },
        ]
    if targeted_data is not None:
        provenance_position = next(
            index
            for index, block in enumerate(blocks)
            if block["id"] == "provenance_block"
        )
        blocks[provenance_position:provenance_position] = [
            {
                "id": "targeted_n128_guide",
                "type": "markdown",
                "sourceId": "targeted_protocol_source",
                "body": (
                    "## Targeted N128 sensitivity\n\n"
                    "These are post-screen comparisons for the frozen selected cells. "
                    "They do **not** re-rank the atlas queue, change an evidence tier, "
                    "or constitute independent confirmation. The factual arm is a "
                    "learned-flow rollout from observed history—a model-generated "
                    "reference arm, not an observed response or causal intervention."
                ),
            },
            {
                "id": "targeted_screen_comparison_block",
                "type": "chart",
                "chartId": "targeted_screen_comparison_chart",
            },
            {
                "id": "targeted_screen_table_block",
                "type": "table",
                "tableId": "targeted_screen_consistency_table",
            },
            {
                "id": "targeted_three_arm_table_block",
                "type": "table",
                "tableId": "targeted_three_arm_table",
            },
        ]
    if external_rows:
        blocks.append(
            {
                "id": "external_reference_block",
                "type": "table",
                "tableId": "external_reference_table",
            }
        )

    artifact = {
        "surface": "dashboard",
        "manifest": {
            "version": 1,
            "surface": "dashboard",
            "title": "Neural lag-effect prediction atlas",
            "description": (
                "Bounded read-only summary of reviewed model predictions; the separate "
                "standalone explorer provides cross-dataset atlas controls."
            ),
            "generatedAt": generated_at,
            "filters": filters,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [
                {"id": source["id"], "label": source["label"], "path": source["path"]}
                for source in source_specs
            ],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": source_specs,
    }
    serialized = json.dumps(artifact, allow_nan=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) >= 3_000_000:
        raise DashboardInputError("portable artifact would exceed the three-megabyte limit")
    _atomic_json(output, artifact)
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--external-reference",
        type=Path,
        help=(
            "checksum-linked EXTERNAL_ANALYSIS_DIR/dashboard_external_summary.csv "
            "from prediction_atlas_external_analysis"
        ),
    )
    parser.add_argument(
        "--targeted-reference",
        type=Path,
        help="validated targeted-confirmation analysis directory (comparison only)",
    )
    parser.add_argument("--max-candidates", type=int, default=150)
    parser.add_argument("--max-external-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_dashboard_artifact(
        args.atlas_dir,
        args.output,
        provenance_root=args.provenance_root,
        external_reference=args.external_reference,
        targeted_reference=args.targeted_reference,
        max_candidates=args.max_candidates,
        max_external_rows=args.max_external_rows,
    )
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
