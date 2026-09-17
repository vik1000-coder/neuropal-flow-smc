"""Checksum-bound repeat-stability audit against historical E26 matrices.

This module compares a *completed* reviewed prediction atlas with the frozen E26
four-sampler analysis.  It deliberately implements only the scientific-estimand
intersection that is exact across the two studies.  It never reads connectome
references and never uses or modifies the atlas hypothesis queue.

The historical and current runs are independent Monte Carlo realizations.  The
comparison is therefore descriptive repeat stability, not a duplicate-run check,
an inferential test, a causal estimate, or evidence for a physical delay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ORIENTATION = "target_row_source_column"
LAG_DEFINITION = "source-window end to prediction cut"
NORMALIZATION = "event-wise response / max(abs(achieved_source_gap), 0.10)"
HORIZON = 1
MIN_GAP = 0.10

# These names are a reviewed semantic allowlist, not a name-based intersection.
OVERLAP_CHANNELS = (
    "endpoint_mean",
    "cumulative_mean",
    "peak_mean",
    "event_probability",
    "endpoint_sd",
)
OVERLAP_CONTEXTS = (
    "baseline",
    "onset",
    "active",
    "onset_minus_baseline",
)
HISTORICAL_METHOD = {
    "direct_importance": "direct_importance_n256",
    "progressive_bridge_smc": "progressive_bridge_smc",
}
RAW_METHOD = {
    "direct_importance": "direct_importance",
    "progressive_bridge_smc": "progressive_bridge_smc",
}
EXPECTED_PHASES = ("baseline", "onset", "active")
OUTPUT_NAMES = {
    "repeat_stability.csv",
    "manifest.json",
    "REPORT.md",
    "checksums.sha256",
    "AUDIT_NOTE.md",
}


class NoExactOverlapError(RuntimeError):
    """Raised only when valid artifacts have no reviewed estimand intersection."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if isinstance(item, Mapping):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    value = data[key]
    if value.size != 1:
        raise RuntimeError(f"archive field {key!r} must be scalar")
    return value.item()


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _parse_checksum_ledger(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"missing checksum ledger: {path}")
    listed: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        digest = parts[0] if len(parts) == 2 else ""
        label = parts[1] if len(parts) == 2 else ""
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not label
        ):
            raise RuntimeError(f"malformed checksum line {line_number} in {path}")
        basename = Path(label).name
        if not basename or basename in listed:
            raise RuntimeError(f"duplicate or unsafe checksum entry in {path}: {label!r}")
        listed[basename] = digest
    return listed


def _verify_checksum_inventory(root: Path, required: Sequence[str]) -> dict[str, str]:
    listed = _parse_checksum_ledger(root / "checksums.sha256")
    verified: dict[str, str] = {}
    for name in required:
        candidate = root / name
        if not candidate.is_file():
            raise RuntimeError(f"required artifact is missing: {candidate}")
        actual = sha256(candidate)
        if listed.get(name) != actual:
            raise RuntimeError(f"checksum failed for {candidate}")
        verified[name] = actual
    return verified


def _require_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required JSON artifact is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


@dataclass(frozen=True)
class AtlasAudit:
    root: Path
    manifest: Mapping[str, Any]
    protocol: Mapping[str, Any]
    validation: Mapping[str, Any]
    models: Mapping[str, Any]
    neurons: tuple[str, ...]
    methods: tuple[str, ...]
    channels: tuple[str, ...]
    contexts: tuple[str, ...]
    lags: tuple[int, ...]
    horizons: tuple[int, ...]
    fps: float
    history_frames: int
    source_window_frames: int
    source_run: Path
    source_manifest_sha256: str
    fold_assignments: Path
    fold_assignments_sha256: str
    checkpoint_rows: frozenset[tuple[int, int, str, str]]
    method_configs: Mapping[str, Mapping[str, Any]]
    runner_manifest_hashes: Mapping[str, str]
    hashes: Mapping[str, str]
    queue_sha256: str


@dataclass(frozen=True)
class HistoricalAudit:
    analysis_root: Path
    archive_path: Path
    primary_run: Path
    direct_run: Path
    neurons: tuple[str, ...]
    lags: tuple[int, ...]
    fps: float
    history_frames: int
    source_window_frames: int
    source_run: Path
    source_manifest_sha256: str
    fold_assignments: Path
    fold_assignments_sha256: str
    checkpoint_rows: frozenset[tuple[int, int, str, str]]
    method_configs: Mapping[str, Mapping[str, Any]]
    run_sidecar_hashes: Mapping[str, str]
    matrices: Mapping[tuple[str, str, str, int], np.ndarray]
    hashes: Mapping[str, str]
    selected_response_hashes: Mapping[str, str]
    reconstruction_checks: int


def _verify_atlas(atlas_dir: Path) -> AtlasAudit:
    root = atlas_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"completed canonical atlas directory does not exist: {root}"
        )
    required = (
        "manifest.json",
        "protocol.json",
        "validation.json",
        "models.json",
        "atlas_matrices.npz",
        "hypothesis_queue.csv",
    )
    hashes = _verify_checksum_inventory(root, required)
    manifest = _require_json(root / "manifest.json")
    protocol = _require_json(root / "protocol.json")
    validation = _require_json(root / "validation.json")
    models = _require_json(root / "models.json")
    if manifest.get("status") != "complete":
        raise RuntimeError("canonical atlas manifest is not complete")
    if validation.get("status") != "passed" or validation.get("problems"):
        raise RuntimeError("canonical atlas validation did not pass cleanly")
    checks = validation.get("archive_checks_completed")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError("canonical atlas archive checks are incomplete")
    if protocol.get("orientation") != ORIENTATION:
        raise RuntimeError("canonical atlas orientation is not target-row/source-column")
    if protocol.get("orientation_operation") != (
        "sampler source and target axes transposed exactly once"
    ):
        raise RuntimeError("canonical atlas does not certify exactly one transpose")
    if protocol.get("ranking") != "no connectome or external reference data used":
        raise RuntimeError("canonical atlas ranking firewall is absent")
    if protocol.get("effect_definition") != (
        "high-source repaired response minus low-source repaired response"
    ):
        raise RuntimeError("canonical atlas effect definition changed")
    normalization = protocol.get("normalization")
    if not isinstance(normalization, dict) or "max(abs(achieved_source_gap), 0.1)" not in str(
        normalization.get("signed_channels", "")
    ):
        raise RuntimeError("canonical atlas signed-channel normalization changed")

    with np.load(root / "atlas_matrices.npz", allow_pickle=False) as data:
        metadata = {
            "neurons",
            "methods",
            "channels",
            "contexts",
            "source_lag_frames",
            "horizon_frames",
            "orientation",
        }
        missing = sorted(metadata.difference(data.files))
        if missing:
            raise RuntimeError(f"canonical dense archive is missing metadata: {missing}")
        neurons = tuple(data["neurons"].astype(str))
        methods = tuple(data["methods"].astype(str))
        channels = tuple(data["channels"].astype(str))
        contexts = tuple(data["contexts"].astype(str))
        lags = tuple(int(value) for value in data["source_lag_frames"])
        horizons = tuple(int(value) for value in data["horizon_frames"])
        if str(_scalar(data, "orientation")) != ORIENTATION:
            raise RuntimeError("canonical dense archive orientation label changed")
        if len(neurons) < 2 or len(set(neurons)) != len(neurons):
            raise RuntimeError("canonical neuron axis is empty or duplicated")
        if len(set(lags)) != len(lags) or min(lags, default=0) <= 0:
            raise RuntimeError("canonical lag axis is invalid")
        if len(set(horizons)) != len(horizons) or min(horizons, default=0) <= 0:
            raise RuntimeError("canonical horizon axis is invalid")

    if int(protocol.get("n_neurons", -1)) != len(neurons):
        raise RuntimeError("canonical protocol neuron count disagrees with archive")
    if tuple(int(value) for value in protocol.get("source_lag_frames", ())) != lags:
        raise RuntimeError("canonical protocol lag grid disagrees with dense archive")
    if tuple(int(value) for value in protocol.get("horizon_frames", ())) != horizons:
        raise RuntimeError("canonical protocol horizon grid disagrees with dense archive")

    source = models.get("source_run_provenance")
    if not isinstance(source, dict):
        raise RuntimeError("canonical models file lacks source-run provenance")
    source_run = Path(str(source.get("path", ""))).resolve()
    source_manifest = Path(str(source.get("manifest", ""))).resolve()
    if source_manifest != source_run / "manifest.json" or not source_manifest.is_file():
        raise RuntimeError("canonical source-run manifest path is invalid")
    source_hash = sha256(source_manifest)
    if source_hash != str(source.get("manifest_sha256", "")):
        raise RuntimeError("canonical source-run manifest hash failed")
    if source_hash != str(manifest.get("source_run_manifest_sha256", "")):
        raise RuntimeError("canonical manifest/source-run hash mismatch")
    source_raw = _require_json(source_manifest)
    if source_raw.get("status") != "complete" or source.get("status") != "complete":
        raise RuntimeError("canonical generator source run is not complete")
    fold_path = Path(str(source.get("fold_assignments", ""))).resolve()
    if not fold_path.is_file():
        raise RuntimeError("canonical fold-assignment file is missing")
    fold_hash = sha256(fold_path)
    if fold_hash != str(source.get("fold_assignments_sha256", "")):
        raise RuntimeError("canonical fold-assignment hash failed")
    if fold_hash != str(manifest.get("fold_assignments_sha256", "")):
        raise RuntimeError("canonical manifest/fold hash mismatch")

    checkpoint_rows: set[tuple[int, int, str, str]] = set()
    checkpoint_values = models.get("checkpoints")
    if not isinstance(checkpoint_values, list) or not checkpoint_values:
        raise RuntimeError("canonical models file lacks checkpoint lineage")
    for row in checkpoint_values:
        if not isinstance(row, dict):
            raise RuntimeError("canonical checkpoint row is malformed")
        checkpoint = Path(str(row.get("path", ""))).resolve()
        digest = str(row.get("sha256", ""))
        if not checkpoint.is_file() or sha256(checkpoint) != digest:
            raise RuntimeError(f"canonical checkpoint hash failed: {checkpoint}")
        item = (int(row["fold"]), int(row["seed"]), str(checkpoint), digest)
        if item in checkpoint_rows:
            raise RuntimeError("canonical checkpoint lineage contains a duplicate")
        checkpoint_rows.add(item)

    declared_method_configs = models.get("methods")
    if not isinstance(declared_method_configs, dict):
        raise RuntimeError("canonical models file lacks method configurations")
    for method in HISTORICAL_METHOD:
        config = declared_method_configs.get(method)
        if not isinstance(config, dict):
            raise NoExactOverlapError(f"canonical atlas lacks method {method}")
        particles = config.get("particles")
        if not isinstance(particles, list) or len(particles) != 1:
            raise RuntimeError(f"canonical particle declaration is ambiguous for {method}")

    # The canonical validation binds the v2 runner manifests.  Rehash them and
    # independently check the timing/axis fields used by this comparison.
    input_manifests = validation.get("input_manifests")
    if not isinstance(input_manifests, list) or not input_manifests:
        raise RuntimeError("canonical validation lacks input-manifest lineage")
    runner_method_configs: dict[str, dict[str, Any]] = {}
    runner_manifest_hashes: dict[str, str] = {}
    for row in input_manifests:
        path = Path(str(row.get("path", ""))).resolve()
        current_hash = sha256(path) if path.is_file() else ""
        if current_hash != str(row.get("sha256", "")):
            raise RuntimeError(f"canonical input-manifest hash failed: {path}")
        runner_manifest_hashes[str(path)] = current_hash
        raw = _require_json(path)
        if raw.get("manifest_schema_version") != "prediction_atlas_manifest_v2":
            raise RuntimeError("canonical runner manifest is not provenance schema v2")
        if tuple(raw.get("response_axes", ())) != (
            "heldout_worm",
            "phase",
            "event",
            "source",
            "horizon",
            "target",
        ):
            raise RuntimeError("canonical runner response axes changed")
        if raw.get("matrix_internal_orientation") != "source,horizon,target":
            raise RuntimeError("canonical runner internal orientation changed")
        if raw.get("lag_definition") != LAG_DEFINITION:
            raise RuntimeError("canonical runner lag definition changed")
        if tuple(raw.get("phases", ()))[:3] != EXPECTED_PHASES:
            raise RuntimeError("canonical first-three phase definitions changed")
        if str(Path(str(raw.get("source_run", ""))).resolve()) != str(source_run):
            raise RuntimeError("canonical runner source run changed")
        if str(raw.get("source_run_manifest_sha256", "")) != source_hash:
            raise RuntimeError("canonical runner source-manifest hash changed")
        if str(raw.get("fold_assignments_sha256", "")) != fold_hash:
            raise RuntimeError("canonical runner fold hash changed")
        raw_methods = tuple(str(value) for value in raw.get("methods", ()))
        if not raw_methods:
            raise RuntimeError("canonical runner manifest has no methods")
        for method in raw_methods:
            current = {
                "particles": int(raw["particles"]),
                "minimum_effective_sample_size": float(
                    raw["minimum_effective_sample_size"]
                ),
                "branch_factor": (
                    int(raw["progressive_branch_factor"])
                    if method == "progressive_bridge_smc"
                    else 1
                ),
                "future_branch_factor": (
                    int(raw["progressive_future_branch_factor"])
                    if method == "progressive_bridge_smc"
                    else 1
                ),
            }
            previous = runner_method_configs.get(method)
            if previous is not None and previous != current:
                raise RuntimeError(
                    f"canonical runner configuration changed across manifests for {method}"
                )
            runner_method_configs[method] = current
    method_configs: dict[str, dict[str, Any]] = {}
    for method in HISTORICAL_METHOD:
        runner = runner_method_configs.get(method)
        declared = declared_method_configs[method]
        if runner is None:
            raise RuntimeError(f"canonical runner provenance lacks method {method}")
        if list(declared["particles"]) != [runner["particles"]]:
            raise RuntimeError(f"canonical models/runner particle mismatch for {method}")
        if int(declared.get("branch_factor", -1)) != runner["branch_factor"] or int(
            declared.get("future_branch_factor", -1)
        ) != runner["future_branch_factor"]:
            raise RuntimeError(f"canonical models/runner branch mismatch for {method}")
        method_configs[method] = {**declared, **runner}

    queue_hash = hashes["hypothesis_queue.csv"]
    return AtlasAudit(
        root=root,
        manifest=manifest,
        protocol=protocol,
        validation=validation,
        models=models,
        neurons=neurons,
        methods=methods,
        channels=channels,
        contexts=contexts,
        lags=lags,
        horizons=horizons,
        fps=float(protocol["fps"]),
        history_frames=int(protocol["history_frames"]),
        source_window_frames=int(protocol["source_window_frames"]),
        source_run=source_run,
        source_manifest_sha256=source_hash,
        fold_assignments=fold_path,
        fold_assignments_sha256=fold_hash,
        checkpoint_rows=frozenset(checkpoint_rows),
        method_configs=method_configs,
        runner_manifest_hashes=runner_manifest_hashes,
        hashes=hashes,
        queue_sha256=queue_hash,
    )


def _validate_historical_run_manifest(
    root: Path, *, expected_method: str, expected_particles: int
) -> dict[str, Any]:
    manifest = _require_json(root / "manifest.json")
    validation = _require_json(root / "validation.json")
    if validation.get("status") != "pass" or validation.get("failed") != 0:
        raise RuntimeError(f"historical run did not pass: {root}")
    if expected_method not in manifest.get("methods", ()):
        raise RuntimeError(f"historical run lacks method {expected_method}: {root}")
    if int(manifest.get("particles", -1)) != expected_particles:
        raise RuntimeError(f"historical particle count changed: {root}")
    if manifest.get("lag_definition") != LAG_DEFINITION:
        raise RuntimeError(f"historical lag definition changed: {root}")
    if tuple(manifest.get("phases", ())) != EXPECTED_PHASES:
        raise RuntimeError(f"historical phase grid changed: {root}")
    if tuple(int(value) for value in manifest.get("horizon_frames", ())) != (HORIZON,):
        raise RuntimeError(f"historical horizon changed: {root}")
    fold_path = Path(str(manifest.get("fold_assignments", ""))).resolve()
    if not fold_path.is_file() or sha256(fold_path) != str(
        manifest.get("fold_assignments_sha256", "")
    ):
        raise RuntimeError(f"historical fold-assignment hash failed: {root}")
    return manifest


def _historical_response_paths(
    checksums: Mapping[str, str], *, raw_method: str, particles: int
) -> tuple[Path, ...]:
    token = f"__{raw_method}__"
    particle_token = f"__N{particles}__"
    paths = tuple(
        sorted(
            Path(name).resolve()
            for name in checksums
            if token in Path(name).name and particle_token in Path(name).name
        )
    )
    if not paths:
        raise NoExactOverlapError(
            f"historical input inventory lacks {raw_method} N={particles} archives"
        )
    return paths


def _reconstruct_historical_method(
    *,
    canonical_method: str,
    historical_method: str,
    paths: Sequence[Path],
    expected_hashes: Mapping[str, str],
    run_manifest: Mapping[str, Any],
    aligned: np.lib.npyio.NpzFile,
    neurons: tuple[str, ...],
) -> tuple[
    dict[tuple[str, str, str, int], np.ndarray],
    set[tuple[int, int, str, str]],
    dict[str, str],
    int,
]:
    raw_method = RAW_METHOD[canonical_method]
    lags = tuple(int(value) for value in run_manifest["source_lag_frames"])
    folds = tuple(int(value) for value in run_manifest["folds"])
    seeds = tuple(int(value) for value in run_manifest["seeds"])
    particles = int(run_manifest["particles"])
    expected_grid = {(lag, fold, seed) for lag in lags for fold in folds for seed in seeds}
    seen_grid: set[tuple[int, int, int]] = set()
    by_lag: dict[int, dict[str, dict[int, dict[str, np.ndarray]]]] = {
        lag: {channel: {seed: {} for seed in seeds} for channel in OVERLAP_CHANNELS}
        for lag in lags
    }
    checkpoint_rows: set[tuple[int, int, str, str]] = set()
    selected_hashes: dict[str, str] = {}
    worm_fold: dict[str, int] = {}

    required_fields = {
        "status",
        "method",
        "model_id",
        "checkpoint",
        "checkpoint_sha256",
        "fold",
        "seed",
        "history_frames",
        "repair_frames",
        "source_lag_frames",
        "lag_definition",
        "source_window_frames",
        "n_particles",
        "horizon_frames",
        "phase_names",
        "worm_ids",
        "neurons",
        "cut_times",
        "source_window_bounds",
        "diagnostic_achieved_gap",
        "diagnostic_achieved_low",
        "diagnostic_achieved_high",
        "diagnostic_target_gap",
        "diagnostic_target_low",
        "diagnostic_target_high",
    } | {f"response_{channel}" for channel in OVERLAP_CHANNELS}

    for path in paths:
        expected = expected_hashes.get(str(path))
        actual = sha256(path) if path.is_file() else ""
        if expected != actual:
            raise RuntimeError(f"historical response checksum failed: {path}")
        selected_hashes[str(path)] = actual
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(required_fields.difference(data.files))
            if missing:
                raise RuntimeError(f"historical response archive lacks {missing}: {path}")
            if str(_scalar(data, "status")) != "complete":
                raise RuntimeError(f"historical response archive is incomplete: {path}")
            if str(_scalar(data, "method")) != raw_method:
                raise RuntimeError(f"historical response method mismatch: {path}")
            fold = int(_scalar(data, "fold"))
            seed = int(_scalar(data, "seed"))
            lag = int(_scalar(data, "source_lag_frames"))
            grid_cell = (lag, fold, seed)
            if grid_cell in seen_grid:
                raise RuntimeError(f"duplicate historical response grid cell: {grid_cell}")
            seen_grid.add(grid_cell)
            scalar_expected = {
                "model_id": str(run_manifest["model_id"]),
                "history_frames": int(run_manifest["history_frames"]),
                "source_window_frames": int(run_manifest["source_window_frames"]),
                "n_particles": particles,
                "lag_definition": LAG_DEFINITION,
            }
            for key, value in scalar_expected.items():
                observed = _scalar(data, key)
                if isinstance(value, int):
                    observed = int(observed)
                else:
                    observed = str(observed)
                if observed != value:
                    raise RuntimeError(f"historical response {key} mismatch: {path}")
            if int(_scalar(data, "repair_frames")) != lag + int(
                run_manifest["source_window_frames"]
            ):
                raise RuntimeError(f"historical repair-prefix geometry failed: {path}")
            if tuple(int(value) for value in data["horizon_frames"]) != (HORIZON,):
                raise RuntimeError(f"historical response horizon mismatch: {path}")
            if tuple(data["phase_names"].astype(str)) != EXPECTED_PHASES:
                raise RuntimeError(f"historical response phase order mismatch: {path}")
            if tuple(data["neurons"].astype(str)) != neurons:
                raise RuntimeError(f"historical response neuron order mismatch: {path}")
            bounds = np.asarray(data["source_window_bounds"])
            cuts = np.asarray(data["cut_times"])
            width = int(run_manifest["source_window_frames"])
            if (
                bounds.shape != (*cuts.shape, 2)
                or not np.all(bounds[..., 1] - bounds[..., 0] == width)
                or not np.all(bounds[..., 1] - 1 == cuts - lag)
            ):
                raise RuntimeError(f"historical source-window timing failed: {path}")
            achieved = np.asarray(data["diagnostic_achieved_gap"], dtype=np.float32)
            achieved_low = np.asarray(data["diagnostic_achieved_low"], dtype=np.float32)
            achieved_high = np.asarray(data["diagnostic_achieved_high"], dtype=np.float32)
            target_gap = np.asarray(data["diagnostic_target_gap"], dtype=np.float32)
            target_low = np.asarray(data["diagnostic_target_low"], dtype=np.float32)
            target_high = np.asarray(data["diagnostic_target_high"], dtype=np.float32)
            if not all(
                np.isfinite(value).all()
                for value in (
                    achieved,
                    achieved_low,
                    achieved_high,
                    target_gap,
                    target_low,
                    target_high,
                )
            ):
                raise RuntimeError(f"historical source diagnostics are nonfinite: {path}")
            if not np.allclose(
                achieved, achieved_high - achieved_low, rtol=1e-5, atol=1e-6
            ) or not np.allclose(
                target_gap, target_high - target_low, rtol=1e-5, atol=1e-6
            ):
                raise RuntimeError(f"historical source-gap arithmetic failed: {path}")
            worm_ids = tuple(data["worm_ids"].astype(str))
            if len(worm_ids) != len(set(worm_ids)):
                raise RuntimeError(f"historical held-out worms are duplicated: {path}")
            for worm in worm_ids:
                previous = worm_fold.get(worm)
                if previous is None:
                    worm_fold[worm] = fold
                elif previous != fold:
                    raise RuntimeError(f"historical worm crosses held-out folds: {worm}")
            checkpoint = Path(str(_scalar(data, "checkpoint"))).resolve()
            digest = str(_scalar(data, "checkpoint_sha256"))
            if not checkpoint.is_file() or sha256(checkpoint) != digest:
                raise RuntimeError(f"historical checkpoint hash failed: {checkpoint}")
            checkpoint_rows.add((fold, seed, str(checkpoint), digest))

            denominator = np.maximum(np.abs(achieved), MIN_GAP)[..., None, None]
            for channel in OVERLAP_CHANNELS:
                response = np.asarray(data[f"response_{channel}"], dtype=np.float32)
                expected_shape = (*achieved.shape, 1, len(neurons))
                if response.shape != expected_shape or not np.isfinite(response).all():
                    raise RuntimeError(
                        f"historical response shape/finite audit failed for {channel}: {path}"
                    )
                if channel == "event_probability" and (
                    np.any(response < -1.0 - 1e-6) or np.any(response > 1.0 + 1e-6)
                ):
                    raise RuntimeError(f"historical event-probability effect is invalid: {path}")
                # [worm,phase,event,source,horizon,target] ->
                # [worm,phase,horizon,target,source], exactly one transpose.
                oriented = (response / denominator).mean(axis=2).transpose(0, 1, 3, 4, 2)
                for local, worm in enumerate(worm_ids):
                    if worm in by_lag[lag][channel][seed]:
                        raise RuntimeError(f"historical worm/seed duplicate: {worm}/{seed}")
                    by_lag[lag][channel][seed][worm] = oriented[local]

    if seen_grid != expected_grid:
        missing = sorted(expected_grid - seen_grid)
        extra = sorted(seen_grid - expected_grid)
        raise RuntimeError(f"historical response grid mismatch; missing={missing}, extra={extra}")

    matrices: dict[tuple[str, str, str, int], np.ndarray] = {}
    reconstruction_checks = 0
    for channel in OVERLAP_CHANNELS:
        for context in OVERLAP_CONTEXTS:
            prefix = f"{historical_method}__{channel}__{context}"
            for suffix in ("lags", "worm_ids", "worm_matrices", "matrices"):
                if f"{prefix}__{suffix}" not in aligned.files:
                    raise NoExactOverlapError(
                        f"historical aligned archive lacks reviewed key {prefix}__{suffix}"
                    )
            archive_lags = tuple(int(value) for value in aligned[f"{prefix}__lags"])
            if archive_lags != lags:
                raise RuntimeError(f"historical aligned lag order changed for {prefix}")
            archive_worms = tuple(aligned[f"{prefix}__worm_ids"].astype(str))
            if len(archive_worms) != int(run_manifest["n_worms"]):
                raise RuntimeError(f"historical aligned worm count changed for {prefix}")
            archive_worm_matrix = np.asarray(
                aligned[f"{prefix}__worm_matrices"], dtype=np.float32
            )
            archive_matrix = np.asarray(aligned[f"{prefix}__matrices"], dtype=np.float32)
            expected_shape = (len(lags), len(archive_worms), len(neurons), len(neurons))
            if archive_worm_matrix.shape != expected_shape or archive_matrix.shape != (
                len(lags),
                len(neurons),
                len(neurons),
            ):
                raise RuntimeError(f"historical aligned matrix geometry failed for {prefix}")
            for lag_index, lag in enumerate(lags):
                per_worm: list[np.ndarray] = []
                for worm in archive_worms:
                    repeats = []
                    for seed in seeds:
                        value = by_lag[lag][channel][seed].get(worm)
                        if value is None:
                            raise RuntimeError(
                                f"historical worm lacks seed repeat: {worm}/{lag}/{seed}"
                            )
                        repeats.append(value)
                    phase_value = np.mean(repeats, axis=0).astype(np.float32)
                    if context == "baseline":
                        selected = phase_value[0, 0]
                    elif context == "onset":
                        selected = phase_value[1, 0]
                    elif context == "active":
                        selected = phase_value[2, 0]
                    else:
                        selected = phase_value[1, 0] - phase_value[0, 0]
                    per_worm.append(selected)
                rebuilt_worm = np.stack(per_worm).astype(np.float32)
                rebuilt_group = rebuilt_worm.mean(axis=0).astype(np.float32)
                if not np.array_equal(rebuilt_worm, archive_worm_matrix[lag_index]):
                    raise RuntimeError(
                        f"historical raw-to-worm reconstruction is not bitwise exact for "
                        f"{prefix}/lag{lag}"
                    )
                if not np.array_equal(rebuilt_group, archive_matrix[lag_index]):
                    raise RuntimeError(
                        f"historical worm-to-group reconstruction is not bitwise exact for "
                        f"{prefix}/lag{lag}"
                    )
                matrices[(canonical_method, channel, context, lag)] = archive_matrix[
                    lag_index
                ].copy()
                reconstruction_checks += 1
    return matrices, checkpoint_rows, selected_hashes, reconstruction_checks


def _verify_historical(historical_analysis_dir: Path) -> HistoricalAudit:
    analysis = historical_analysis_dir.resolve()
    if not analysis.is_dir():
        raise FileNotFoundError(f"historical E26 analysis directory does not exist: {analysis}")
    required = (
        "aligned_four_sampler_lag_matrices.npz",
        "validation.json",
        "input_checksums.json",
        "REPORT.md",
    )
    hashes = _verify_checksum_inventory(analysis, required)
    validation = _require_json(analysis / "validation.json")
    if (
        validation.get("status") != "pass"
        or validation.get("matrix_orientation") != "target,source"
        or validation.get("internal_sampler_orientation") != "source,horizon,target"
        or validation.get("transpose_count_at_analysis_boundary") != 1
        or validation.get("orientation_spot_check") is not True
        or validation.get("checkpoint_paths_and_sha256_reverified") is not True
        or validation.get("worm_to_fold_mapping_reverified") is not True
        or validation.get("source_window_and_lag_bounds_reverified") is not True
    ):
        raise RuntimeError("historical E26 analysis validation is incomplete")
    if validation.get("normalization") != (
        "response divided by max(abs(achieved high-low source gap), 0.10), "
        "event-wise before averaging"
    ):
        raise RuntimeError("historical E26 normalization changed")
    if validation.get("lag_definition") != LAG_DEFINITION:
        raise RuntimeError("historical E26 lag definition changed")

    input_checksums_raw = _require_json(analysis / "input_checksums.json")
    input_checksums: dict[str, str] = {}
    for name, digest in input_checksums_raw.items():
        resolved = str(Path(name).resolve())
        if (
            len(str(digest)) != 64
            or any(character not in "0123456789abcdef" for character in str(digest))
            or resolved in input_checksums
        ):
            raise RuntimeError("historical input-checksum inventory is malformed")
        input_checksums[resolved] = str(digest)

    progressive_paths = _historical_response_paths(
        input_checksums, raw_method="progressive_bridge_smc", particles=32
    )
    direct_paths = _historical_response_paths(
        input_checksums, raw_method="direct_importance", particles=256
    )
    primary_run = analysis.parent.resolve()
    progressive_roots = {path.parents[2] for path in progressive_paths}
    direct_roots = {path.parents[2] for path in direct_paths}
    if progressive_roots != {primary_run} or len(direct_roots) != 1:
        raise RuntimeError("historical response inventory resolves to ambiguous run roots")
    direct_run = next(iter(direct_roots)).resolve()
    progressive_manifest = _validate_historical_run_manifest(
        primary_run, expected_method="progressive_bridge_smc", expected_particles=32
    )
    direct_manifest = _validate_historical_run_manifest(
        direct_run, expected_method="direct_importance", expected_particles=256
    )
    common_keys = (
        "source_run",
        "checkpoint_phase",
        "model_id",
        "cohort_mode",
        "n_worms",
        "n_neurons",
        "fps",
        "history_frames",
        "source_lag_frames",
        "source_window_frames",
        "horizon_frames",
        "phases",
        "folds",
        "seeds",
        "fold_assignments",
        "fold_assignments_sha256",
        "stimulus_schema",
    )
    for key in common_keys:
        if progressive_manifest.get(key) != direct_manifest.get(key):
            raise RuntimeError(f"historical direct/progressive manifests disagree on {key}")
    source_run = Path(str(progressive_manifest["source_run"])).resolve()
    source_manifest = source_run / "manifest.json"
    if not source_manifest.is_file():
        raise RuntimeError("historical source-run manifest is missing")
    source_hash = sha256(source_manifest)
    source_raw = _require_json(source_manifest)
    if source_raw.get("status") != "complete":
        raise RuntimeError("historical source generator run is not complete")
    fold_path = Path(str(progressive_manifest["fold_assignments"])).resolve()
    fold_hash = sha256(fold_path)
    if fold_hash != str(progressive_manifest["fold_assignments_sha256"]):
        raise RuntimeError("historical fold-assignment hash changed")

    archive_path = analysis / "aligned_four_sampler_lag_matrices.npz"
    all_matrices: dict[tuple[str, str, str, int], np.ndarray] = {}
    checkpoints: set[tuple[int, int, str, str]] = set()
    selected_hashes: dict[str, str] = {}
    reconstruction_checks = 0
    with np.load(archive_path, allow_pickle=False) as aligned:
        if "neurons" not in aligned.files:
            raise RuntimeError("historical aligned archive lacks neuron axis")
        neurons = tuple(aligned["neurons"].astype(str))
        if len(neurons) != int(progressive_manifest["n_neurons"]):
            raise RuntimeError("historical neuron count disagrees with manifest")
        if len(set(neurons)) != len(neurons):
            raise RuntimeError("historical neuron axis is duplicated")
        for canonical_method, historical_method, paths, run_manifest in (
            (
                "direct_importance",
                "direct_importance_n256",
                direct_paths,
                direct_manifest,
            ),
            (
                "progressive_bridge_smc",
                "progressive_bridge_smc",
                progressive_paths,
                progressive_manifest,
            ),
        ):
            matrices, rows, path_hashes, checks = _reconstruct_historical_method(
                canonical_method=canonical_method,
                historical_method=historical_method,
                paths=paths,
                expected_hashes=input_checksums,
                run_manifest=run_manifest,
                aligned=aligned,
                neurons=neurons,
            )
            all_matrices.update(matrices)
            checkpoints.update(rows)
            selected_hashes.update(path_hashes)
            reconstruction_checks += checks

    method_configs = {
        "direct_importance": {
            "particles": int(direct_manifest["particles"]),
            "minimum_effective_sample_size": float(
                direct_manifest["minimum_effective_sample_size"]
            ),
            "branch_factor": 1,
            "future_branch_factor": 1,
        },
        "progressive_bridge_smc": {
            "particles": int(progressive_manifest["particles"]),
            "minimum_effective_sample_size": float(
                progressive_manifest["minimum_effective_sample_size"]
            ),
            "branch_factor": int(progressive_manifest["progressive_branch_factor"]),
            "future_branch_factor": int(
                progressive_manifest["progressive_future_branch_factor"]
            ),
        },
    }
    run_sidecar_hashes = {
        str(primary_run / "manifest.json"): sha256(primary_run / "manifest.json"),
        str(primary_run / "validation.json"): sha256(primary_run / "validation.json"),
        str(direct_run / "manifest.json"): sha256(direct_run / "manifest.json"),
        str(direct_run / "validation.json"): sha256(direct_run / "validation.json"),
        str(source_manifest): source_hash,
        str(fold_path): fold_hash,
    }
    return HistoricalAudit(
        analysis_root=analysis,
        archive_path=archive_path,
        primary_run=primary_run,
        direct_run=direct_run,
        neurons=neurons,
        lags=tuple(int(value) for value in progressive_manifest["source_lag_frames"]),
        fps=float(progressive_manifest["fps"]),
        history_frames=int(progressive_manifest["history_frames"]),
        source_window_frames=int(progressive_manifest["source_window_frames"]),
        source_run=source_run,
        source_manifest_sha256=source_hash,
        fold_assignments=fold_path,
        fold_assignments_sha256=fold_hash,
        checkpoint_rows=frozenset(checkpoints),
        method_configs=method_configs,
        run_sidecar_hashes=run_sidecar_hashes,
        matrices=all_matrices,
        hashes=hashes,
        selected_response_hashes=selected_hashes,
        reconstruction_checks=reconstruction_checks,
    )


def _semantic_overlap(atlas: AtlasAudit, historical: HistoricalAudit) -> dict[str, Any]:
    if atlas.neurons != historical.neurons:
        raise RuntimeError("canonical and historical neuron order differs")
    scalar_pairs = {
        "fps": (atlas.fps, historical.fps),
        "history_frames": (atlas.history_frames, historical.history_frames),
        "source_window_frames": (
            atlas.source_window_frames,
            historical.source_window_frames,
        ),
        "source_run": (str(atlas.source_run), str(historical.source_run)),
        "source_manifest_sha256": (
            atlas.source_manifest_sha256,
            historical.source_manifest_sha256,
        ),
        "fold_assignments": (
            str(atlas.fold_assignments),
            str(historical.fold_assignments),
        ),
        "fold_assignments_sha256": (
            atlas.fold_assignments_sha256,
            historical.fold_assignments_sha256,
        ),
    }
    for label, (current, previous) in scalar_pairs.items():
        if current != previous:
            raise RuntimeError(f"canonical/historical lineage mismatch: {label}")
    if atlas.checkpoint_rows != historical.checkpoint_rows:
        missing = sorted(historical.checkpoint_rows - atlas.checkpoint_rows)
        extra = sorted(atlas.checkpoint_rows - historical.checkpoint_rows)
        raise RuntimeError(
            f"canonical/historical checkpoint lineage mismatch; missing={missing}, extra={extra}"
        )
    methods = tuple(method for method in HISTORICAL_METHOD if method in atlas.methods)
    channels = tuple(channel for channel in OVERLAP_CHANNELS if channel in atlas.channels)
    contexts = tuple(context for context in OVERLAP_CONTEXTS if context in atlas.contexts)
    lags = tuple(lag for lag in historical.lags if lag in atlas.lags)
    if HORIZON not in atlas.horizons or not methods or not channels or not contexts or not lags:
        raise NoExactOverlapError("no reviewed E26/canonical scientific-estimand overlap")
    return {
        "methods": methods,
        "channels": channels,
        "contexts": contexts,
        "lags": lags,
        "horizon": HORIZON,
        "excluded": {
            "state_average": (
                "not comparable: E26 averages baseline/onset/active; canonical atlas "
                "averages baseline/onset/active/offset/recovery"
            ),
            "offset_and_recovery": "absent from E26",
            "chemical_event_strata": "absent from E26 and exploratory in canonical atlas",
            "endpoint_log_sd_and_wasserstein1": "absent from E26",
            "horizons_greater_than_one": "absent from E26",
            "historical_direct_importance_n32": (
                "not the canonical direct configuration; canonical direct maps to "
                "historical direct_importance_n256"
            ),
        },
    }


def _matrix_metrics(current: np.ndarray, historical: np.ndarray) -> dict[str, Any]:
    current = np.asarray(current, dtype=np.float64)
    historical = np.asarray(historical, dtype=np.float64)
    if current.shape != historical.shape or current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise RuntimeError("repeat-stability matrices have incompatible geometry")
    if not np.isfinite(current).all() or not np.isfinite(historical).all():
        raise RuntimeError("repeat-stability matrices contain nonfinite values")
    off = ~np.eye(current.shape[0], dtype=bool)
    x = current[off]
    y = historical[off]
    if np.std(x) == 0 or np.std(y) == 0:
        rho = float("nan")
    else:
        rho = float(spearmanr(x, y).statistic)
    signs_x = np.sign(x)
    signs_y = np.sign(y)
    both_nonzero = (signs_x != 0) & (signs_y != 0)
    return {
        "matrix_spearman": rho,
        "sign_agreement": float(np.mean(signs_x == signs_y)),
        "sign_agreement_definition": (
            "fraction of all off-diagonal entries with identical {-1,0,+1} sign"
        ),
        "both_nonzero_sign_agreement": (
            float(np.mean(signs_x[both_nonzero] == signs_y[both_nonzero]))
            if both_nonzero.any()
            else float("nan")
        ),
        "n_both_nonzero_sign_edges": int(both_nonzero.sum()),
        "rmse": float(np.sqrt(np.mean(np.square(x - y)))),
        "mean_difference": float(np.mean(x - y)),
        "mean_absolute_current": float(np.mean(np.abs(x))),
        "mean_absolute_historical": float(np.mean(np.abs(y))),
        "n_off_diagonal_edges": int(off.sum()),
    }


def _comparison_rows(
    atlas: AtlasAudit,
    historical: HistoricalAudit,
    overlap: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    horizon_index = atlas.horizons.index(HORIZON)
    with np.load(atlas.root / "atlas_matrices.npz", allow_pickle=False) as dense:
        for method in overlap["methods"]:
            new_config = atlas.method_configs[method]
            old_config = historical.method_configs[method]
            new_particles = int(new_config["particles"])
            for channel in overlap["channels"]:
                for context in overlap["contexts"]:
                    key = f"mean_normalized__{method}__{channel}__{context}"
                    if key not in dense.files:
                        raise RuntimeError(f"canonical dense archive lacks exact-overlap key {key}")
                    values = np.asarray(dense[key], dtype=np.float32)
                    expected = (
                        len(atlas.lags),
                        len(atlas.horizons),
                        len(atlas.neurons),
                        len(atlas.neurons),
                    )
                    if values.shape != expected or not np.isfinite(values).all():
                        raise RuntimeError(f"canonical dense matrix geometry failed for {key}")
                    for lag in overlap["lags"]:
                        lag_index = atlas.lags.index(lag)
                        previous = historical.matrices.get((method, channel, context, lag))
                        if previous is None:
                            raise RuntimeError(
                                f"historical reconstruction lacks {method}/{channel}/{context}/lag{lag}"
                            )
                        metric = _matrix_metrics(values[lag_index, horizon_index], previous)
                        old_particles = int(old_config["particles"])
                        old_branch = int(old_config["branch_factor"])
                        old_future = int(old_config["future_branch_factor"])
                        new_branch = int(new_config.get("branch_factor", 1))
                        new_future = int(new_config.get("future_branch_factor", 1))
                        rows.append(
                            {
                                "method": method,
                                "historical_method": HISTORICAL_METHOD[method],
                                "channel": channel,
                                "context": context,
                                "source_lag_frames": lag,
                                "source_to_cut_seconds": lag / atlas.fps,
                                "horizon_frames": HORIZON,
                                "forecast_horizon_seconds": HORIZON / atlas.fps,
                                "source_window_end_to_readout_seconds": (
                                    lag + HORIZON
                                )
                                / atlas.fps,
                                "n_neurons": len(atlas.neurons),
                                "matrix_orientation": ORIENTATION,
                                "edge_scope": "off_diagonal",
                                "effect_scale": NORMALIZATION,
                                "scientific_estimand_exact": True,
                                "response_estimator_configuration_exact": bool(
                                    new_particles == old_particles
                                    and new_branch == old_branch
                                    and new_future == old_future
                                ),
                                "support_gate_configuration_exact": bool(
                                    float(new_config["minimum_effective_sample_size"])
                                    == float(
                                        old_config["minimum_effective_sample_size"]
                                    )
                                ),
                                "finite_mc_realization_exact": False,
                                "current_particles": new_particles,
                                "historical_particles": old_particles,
                                "current_branch_factor": new_branch,
                                "historical_branch_factor": old_branch,
                                "current_future_branch_factor": new_future,
                                "historical_future_branch_factor": old_future,
                                "support_masks_compared": False,
                                **metric,
                            }
                        )
    expected_rows = (
        len(overlap["methods"])
        * len(overlap["channels"])
        * len(overlap["contexts"])
        * len(overlap["lags"])
    )
    if len(rows) != expected_rows:
        raise AssertionError("repeat-stability row count changed")
    return rows


def _report_text(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> str:
    method_rows: list[str] = []
    for method, group in frame.groupby("method", sort=False):
        method_rows.append(
            "| {method} | {rows} | {rho:.3f} | {sign:.3f} | {rmse:.4f} |".format(
                method=method,
                rows=len(group),
                rho=float(group["matrix_spearman"].median()),
                sign=float(group["sign_agreement"].median()),
                rmse=float(group["rmse"].median()),
            )
        )
    excluded = manifest["semantic_overlap"]["excluded"]
    exclusion_lines = "\n".join(f"- **{key}:** {value}" for key, value in excluded.items())
    return f"""# Historical E26 repeat-stability audit

## Result

This audit compares {len(frame)} exact scientific-estimand matrix pairs between the completed canonical atlas and historical E26. It is a descriptive independent-Monte-Carlo repeat check. It does not change the hypothesis queue, evidence tiers, candidate order, or any source artifact.

| Canonical method | Matrix slices | Median Spearman | Median sign agreement | Median RMSE |
|---|---:|---:|---:|---:|
{chr(10).join(method_rows)}

The metrics use all {int(frame['n_off_diagonal_edges'].iloc[0])} directed off-diagonal entries per 54-neuron matrix (or the declared neuron count in a synthetic audit). Spearman compares ranks, sign agreement treats negative/zero/positive as three explicit categories, and RMSE is on the achieved-source-gap-normalized effect scale.

## Exact comparison scope

- Methods: canonical `direct_importance` maps to E26 `direct_importance_n256`; progressive bridge maps by the same name.
- Lags: {', '.join(str(value) for value in manifest['semantic_overlap']['lags'])} frames.
- Horizon: one frame ({1 / float(manifest['fps']):.2f} s).
- Contexts: {', '.join(manifest['semantic_overlap']['contexts'])}.
- Channels: {', '.join(manifest['semantic_overlap']['channels'])}.
- Timing: source-window end is `cut - lag`; readout is `cut + 1`, so source-window-end to readout is `(lag + 1) / fps`.
- Aggregation: normalize each event by `max(abs(achieved gap), 0.10)`, average three events, average two model seeds within worm, then give worms equal weight; orient once to target rows/source columns.

At h=1, cumulative mean and peak mean are algebraically the same functional as endpoint mean, although all three saved channels are checked independently.

## Deliberate exclusions

{exclusion_lines}

## Provenance and interpretation

The source generator, source manifest hash, fold file/hash, checkpoint path/hash set, neuron order, lag geometry, and historical raw-to-aligned reconstruction all match. Every compared historical matrix was rebuilt from checksum-verified raw E26 response archives and reproduced bitwise before comparison.

The finite Monte Carlo configurations are not identical in every case. E26 progressive bridge used future branch factor 2 versus 1 in the canonical atlas; this changes finite-particle precision, not the repaired target law. E26 direct N=256 used an ESS validity threshold of 20 versus 6; that changes support flags, not response values. In addition, E26 onset-minus-baseline support took the minimum of phasewise event means, whereas the canonical atlas averages eventwise phase minima. Effect values are algebraically identical, but support masks are not, so support is not compared. E26 also did not serialize its base seed or v2 response schema, so this artifact remains historical sensitivity evidence rather than canonical-v2 provenance.

No connectome, receptor atlas, Randi, Cook, Bentley, or SBTG data enter this audit. Repeat stability does not establish causality or a physical transmission delay.
"""


def _prepare_output(output: Path, *, overwrite: bool) -> Path:
    output = output.resolve()
    if output.exists():
        existing = [output / name for name in OUTPUT_NAMES if (output / name).exists()]
        if existing and not overwrite:
            raise FileExistsError(f"repeat-stability outputs already exist in {output}")
        if overwrite:
            for path in existing:
                path.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _write_incompatibility_note(
    output_dir: Path,
    *,
    message: str,
    atlas_dir: Path,
    historical_analysis_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    output = _prepare_output(output_dir, overwrite=overwrite)
    output.mkdir(parents=True, exist_ok=True)
    note = (
        "# E26 repeat-stability audit not run\n\n"
        "No exact reviewed scientific-estimand overlap could be established, so no "
        "matrix comparison was forced.\n\n"
        f"Reason: {message}\n\n"
        "The canonical hypothesis queue and evidence tiers were not read for content or changed.\n"
    )
    (output / "AUDIT_NOTE.md").write_text(note)
    manifest = {
        "status": "no_exact_overlap",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": message,
        "atlas_dir": str(atlas_dir.resolve()),
        "historical_analysis_dir": str(historical_analysis_dir.resolve()),
        "comparison_rows": 0,
        "queue_or_evidence_tier_effect": "none",
        "artifacts": {"audit_note": "AUDIT_NOTE.md"},
    }
    _json_dump(output / "manifest.json", manifest)
    paths = (output / "AUDIT_NOTE.md", output / "manifest.json")
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in paths)
    )
    return manifest


def run_repeat_stability_audit(
    atlas_dir: Path,
    historical_analysis_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    note_if_no_overlap: bool = True,
) -> dict[str, Any]:
    """Run the bounded, fail-closed historical repeat-stability comparison."""
    atlas_path = Path(atlas_dir).resolve()
    historical_path = Path(historical_analysis_dir).resolve()
    output = Path(output_dir).resolve()
    for source in (atlas_path, historical_path):
        if _paths_overlap(output, source):
            raise RuntimeError("repeat-stability output must be disjoint from every input")
    # A missing canonical atlas is not semantic non-overlap: it means the task is
    # premature.  Fail before creating any audit artifact.
    atlas = _verify_atlas(atlas_path)
    historical = _verify_historical(historical_path)
    for source in (
        atlas.root,
        atlas.source_run,
        historical.analysis_root,
        historical.primary_run,
        historical.direct_run,
        historical.source_run,
    ):
        if _paths_overlap(output, source):
            raise RuntimeError(
                "repeat-stability output must be disjoint from canonical, historical, "
                "and generator source trees"
            )
    try:
        overlap = _semantic_overlap(atlas, historical)
    except NoExactOverlapError as error:
        if not note_if_no_overlap:
            raise
        return _write_incompatibility_note(
            output,
            message=str(error),
            atlas_dir=atlas_path,
            historical_analysis_dir=historical_path,
            overwrite=overwrite,
        )

    output = _prepare_output(output, overwrite=overwrite)
    output.mkdir(parents=True, exist_ok=True)
    queue_before = sha256(atlas.root / "hypothesis_queue.csv")
    atlas_ledger_before = sha256(atlas.root / "checksums.sha256")
    rows = _comparison_rows(atlas, historical, overlap)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise NoExactOverlapError("reviewed overlap unexpectedly produced no rows")
    frame.to_csv(output / "repeat_stability.csv", index=False)

    queue_after = sha256(atlas.root / "hypothesis_queue.csv")
    atlas_ledger_after = sha256(atlas.root / "checksums.sha256")
    if queue_before != atlas.queue_sha256 or queue_after != queue_before:
        raise RuntimeError("canonical hypothesis queue changed during repeat audit")
    if atlas_ledger_after != atlas_ledger_before:
        raise RuntimeError("canonical checksum ledger changed during repeat audit")

    manifest: dict[str, Any] = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "historical E26 independent-Monte-Carlo repeat stability v1",
        "comparison_rows": len(frame),
        "fps": atlas.fps,
        "n_neurons": len(atlas.neurons),
        "matrix_orientation": ORIENTATION,
        "edge_scope": "off_diagonal",
        "normalization": NORMALIZATION,
        "metric_definitions": {
            "matrix_spearman": "Spearman rank correlation over directed off-diagonal entries",
            "sign_agreement": (
                "fraction of directed off-diagonal entries with identical {-1,0,+1} sign"
            ),
            "rmse": "root mean squared difference over directed off-diagonal normalized effects",
        },
        "semantic_overlap": overlap,
        "lineage": {
            "source_run": str(atlas.source_run),
            "source_manifest_sha256": atlas.source_manifest_sha256,
            "fold_assignments": str(atlas.fold_assignments),
            "fold_assignments_sha256": atlas.fold_assignments_sha256,
            "checkpoint_rows": len(atlas.checkpoint_rows),
            "checkpoint_sets_exact": True,
            "neuron_order_exact": True,
            "neuron_order_sha256": hashlib.sha256(
                "\0".join(atlas.neurons).encode("utf-8")
            ).hexdigest(),
            "timing_geometry_exact": True,
            "historical_raw_to_aligned_bitwise_checks": historical.reconstruction_checks,
        },
        "configuration_differences": {
            "direct_importance": {
                "historical_method": "direct_importance_n256",
                "particles": "256 in both",
                "minimum_ess": "E26 20 versus canonical 6; support-only difference",
            },
            "progressive_bridge_smc": {
                "particles": "32 in both",
                "branch_factor": "2 in both",
                "future_branch_factor": "E26 2 versus canonical 1",
            },
            "random_realization": (
                "independent Monte Carlo runs; E26 base seed was not serialized, while "
                "canonical v2 binds its seed formula and base seed"
            ),
            "support_aggregation": (
                "not compared: E26 onset-minus-baseline takes the minimum of phasewise "
                "event means; canonical takes the mean of eventwise phase minima; direct "
                "also uses a different minimum-ESS diagnostic gate"
            ),
        },
        "input_hashes": {
            "canonical": {
                **atlas.hashes,
                "checksums.sha256": atlas_ledger_before,
            },
            "canonical_runner_manifests": dict(atlas.runner_manifest_hashes),
            "historical_analysis": {
                **historical.hashes,
                "checksums.sha256": sha256(historical.analysis_root / "checksums.sha256"),
            },
            "historical_run_sidecars": dict(historical.run_sidecar_hashes),
            "selected_historical_response_archives": {
                "count": len(historical.selected_response_hashes),
                "aggregate_sha256": hashlib.sha256(
                    "".join(
                        f"{path}\0{digest}\n"
                        for path, digest in sorted(
                            historical.selected_response_hashes.items()
                        )
                    ).encode("utf-8")
                ).hexdigest(),
            },
        },
        "queue_firewall": {
            "frozen_hypothesis_queue_sha256_before": queue_before,
            "frozen_hypothesis_queue_sha256_after": queue_after,
            "unchanged": True,
            "queue_content_used": False,
            "evidence_tiers_used": False,
            "ranking_or_promotion_effect": "none",
        },
        "external_reference_inputs": [],
        "claim_boundary": (
            "descriptive model repeat stability; not causal, confirmatory, or evidence "
            "of a physical transmission delay"
        ),
        "historical_provenance_grade": (
            "pre-v2 historical sensitivity evidence: immutable response and analysis "
            "hashes/reconstruction verified, but E26 did not serialize base seed, device, "
            "response schema version, response-axis label, or source-manifest hash"
        ),
        "validation": {
            "canonical_checksum_inventory": True,
            "historical_checksum_inventory": True,
            "selected_historical_raw_archive_hashes": True,
            "historical_raw_to_aligned_reconstruction_bitwise_exact": True,
            "checkpoint_path_and_hash_sets_exact": True,
            "neuron_order_exact": True,
            "source_window_and_horizon_geometry_exact": True,
            "orientation_target_row_source_column": True,
            "state_average_excluded": True,
            "queue_and_evidence_tiers_unchanged": True,
        },
        "artifacts": {
            "metrics": "repeat_stability.csv",
            "report": "REPORT.md",
            "manifest": "manifest.json",
        },
    }
    _json_dump(output / "manifest.json", manifest)
    (output / "REPORT.md").write_text(_report_text(frame, manifest))
    checksum_paths = (
        output / "repeat_stability.csv",
        output / "manifest.json",
        output / "REPORT.md",
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths)
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument(
        "--historical-analysis-dir",
        type=Path,
        default=Path("results/four_sampler_lag_connectome_20260828/analysis"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-incompatibility-note",
        action="store_true",
        help="raise rather than write AUDIT_NOTE.md when valid artifacts have no exact overlap",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_repeat_stability_audit(
        args.atlas_dir,
        args.historical_analysis_dir,
        args.output_dir,
        overwrite=args.overwrite,
        note_if_no_overlap=not args.no_incompatibility_note,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
