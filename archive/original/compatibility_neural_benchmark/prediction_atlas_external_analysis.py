"""Post-freeze external validation for the reviewed neural prediction atlas.

This module is intentionally downstream of ``prediction_atlas_analysis``.  It
requires a completed internal atlas, records the frozen hypothesis-queue hash,
and never modifies internal ranks or evidence tiers.  Randi, Cook, Bentley, and
SBTG are convergence references, not ground truth for activity or causal edges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compatibility_neural_benchmark.four_sampler_lag_analysis import (
    correlation_metrics,
)
from compatibility_neural_benchmark.latent_distributional_audit import (
    benjamini_hochberg,
)
from compatibility_neural_benchmark.paired_lag_correspondence import (
    permute_within_source,
)
from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    load_references,
    source_macro_metrics,
)


PRIMARY_REFERENCES = (
    "randi_wild_type",
    "cook_struct_54",
    "cook_chem_54",
    "cook_gap_54",
)
EXPECTED_NEUROMODULATOR_SOURCES = {
    "monoamine_all": 5,
    "monoamine_dopamine": 2,
    "monoamine_serotonin": 1,
    "monoamine_tyramine": 1,
    "monoamine_octopamine": 1,
    "neuropeptide_all": 30,
    "neuromodulator_union": 31,
}
PRIMARY_LAGMAX_SLICES = (
    ("progressive_bridge_smc", "endpoint_mean", "state_average", 1),
    ("progressive_bridge_smc", "endpoint_log_sd", "state_average", 1),
    ("progressive_bridge_smc", "endpoint_wasserstein1", "state_average", 1),
    ("progressive_bridge_smc", "endpoint_mean", "onset_minus_baseline", 1),
    ("direct_importance", "endpoint_mean", "state_average", 1),
    ("direct_importance", "endpoint_log_sd", "state_average", 1),
    ("direct_importance", "endpoint_wasserstein1", "state_average", 1),
    ("direct_importance", "endpoint_mean", "onset_minus_baseline", 1),
)
ORIENTATION = "target_row_source_column"
SUPPORT_THRESHOLD = 0.50
GLOBAL_BH_FAMILY = (
    "all_planned_method_channel_context_network_lag_grid_max_lag_tests"
)
REFERENCE_RELEASE_FILES = (
    "reference_data/functional_atlas/aligned_atlas_wild_type.npz",
    "reference_data/functional_atlas/aligned_atlas_unc31.npz",
    "reference_data/connectome/nodes.json",
    "reference_data/connectome/A_struct.npy",
    "reference_data/connectome/A_chem.npy",
    "reference_data/connectome/A_gap.npy",
    "reference_data/modulatory_atlas/edge_lists/edgelist_MA_classes.csv",
    "reference_data/modulatory_atlas/edge_lists/edgelist_NP_classes.csv",
)
OUTPUT_FILES = {
    "randi_cook_metrics.csv",
    "neuromodulator_metrics.csv",
    "neuromodulator_lagmax_inference.csv",
    "primary_randi_cook_profile.csv",
    "primary_neuromodulator_profile.csv",
    "dashboard_external_summary.csv",
    "manifest.json",
    "input_checksums.sha256",
    "checksums.sha256",
}
SBTG_LINEAGE = {
    "sbtg_current": (
        "corrected 54-neuron raw-head cohort with no tail pseudo-pairing or donor copying; "
        "SBTG estimator training remains distinct from the flow generator"
    ),
    "sbtg_published": (
        "released 80-neuron pseudo-paired/donor-imputed training lineage, with the frozen "
        "matrix subset to the exact shared 54-neuron order"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    def clean(item):
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        return item

    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")


def _reference_release_hashes(release: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in REFERENCE_RELEASE_FILES:
        path = release / relative
        if not path.is_file():
            raise RuntimeError(f"external reference release is incomplete: missing {relative}")
        hashes[relative] = sha256(path)
    return hashes


def _verify_sbtg_archive_provenance(archive: Path) -> dict[str, str]:
    root = archive.parent
    protocol_path = root / "protocol.json"
    validation_path = root / "validation.json"
    checksums_path = root / "checksums.sha256"
    for path in (protocol_path, validation_path, checksums_path):
        if not path.is_file():
            raise RuntimeError(
                f"SBTG aligned archive lacks required provenance sidecar: {path.name}"
            )
    protocol = json.loads(protocol_path.read_text())
    validation = json.loads(validation_path.read_text())
    if protocol.get("orientation") != "all saved comparison matrices are [target, source]":
        raise RuntimeError("SBTG sidecar does not declare target-row/source-column orientation")
    if (
        validation.get("status") != "pass"
        or validation.get("neuron_order_exact_match") is not True
        or validation.get("paired_orientation_transposed_once") is not True
    ):
        raise RuntimeError("SBTG sidecar orientation/neuron-order validation did not pass")
    listed: dict[str, str] = {}
    for line in checksums_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise RuntimeError("SBTG checksum sidecar is malformed")
        listed[parts[1]] = parts[0]
    if listed.get(archive.name) != sha256(archive):
        raise RuntimeError("SBTG aligned archive fails its source checksum inventory")
    return {
        archive.name: sha256(archive),
        "protocol.json": sha256(protocol_path),
        "validation.json": sha256(validation_path),
        "checksums.sha256": sha256(checksums_path),
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path contains the other."""

    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _timing_fields(item: dict[str, object], fps: float) -> dict[str, object]:
    lag = int(item["lag_frames"])
    if str(item.get("family", "")).startswith("sbtg"):
        return {
            "lag_index_seconds": lag / fps,
            "source_to_cut_seconds": None,
            "forecast_horizon_seconds": None,
            "source_to_readout_seconds": None,
            "lag_semantics": "historical_sbtg_score_product_time_lag",
            "timing_comparability": (
                "frame-index/time-bin correspondence only; not the repaired-flow estimand"
            ),
        }
    horizon = int(item["horizon_frames"])
    return {
        "lag_index_seconds": lag / fps,
        "source_to_cut_seconds": lag / fps,
        "forecast_horizon_seconds": horizon / fps,
        "source_to_readout_seconds": (lag + horizon) / fps,
        "lag_semantics": "repaired_source_window_end_to_prediction_cut",
        "timing_comparability": "flow source-to-cut plus forecast timing",
    }


def _reference_semantics(name: str) -> dict[str, str]:
    if name.startswith("randi_"):
        return {
            "reference_target_type": (
                "binary significant-response label plus continuous signed dff"
            ),
            "binary_metric_definition": "absolute model effect versus response existence",
            "continuous_metric_definition": (
                "absolute/signed Spearman versus aligned dff on the tested mask"
            ),
        }
    return {
        "reference_target_type": "binary edge existence plus continuous connection weight",
        "binary_metric_definition": "absolute model effect versus edge existence",
        "continuous_metric_definition": (
            "absolute Spearman versus raw/log1p weight, including positive-edge sensitivity"
        ),
    }


def _verify_internal_atlas(atlas_dir: Path) -> dict[str, object]:
    required = (
        "manifest.json",
        "protocol.json",
        "validation.json",
        "atlas_matrices.npz",
        "hypothesis_queue.csv",
        "checksums.sha256",
    )
    for name in required:
        if not (atlas_dir / name).is_file():
            raise RuntimeError(f"internal atlas is incomplete: missing {name}")
    validation = json.loads((atlas_dir / "validation.json").read_text())
    if validation.get("status") != "passed" or validation.get("problems"):
        raise RuntimeError("internal atlas validation did not pass cleanly")
    checks = validation.get("archive_checks_completed")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError("internal atlas archive checks are absent or incomplete")
    listed: dict[str, str] = {}
    for line_number, line in enumerate(
        (atlas_dir / "checksums.sha256").read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or any(
            character not in "0123456789abcdef" for character in parts[0]
        ):
            raise RuntimeError(f"internal checksum line {line_number} is malformed")
        name = parts[1]
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
            raise RuntimeError("internal checksum inventory contains an unsafe path")
        if name in listed:
            raise RuntimeError(f"internal checksum inventory duplicates {name}")
        listed[name] = parts[0]
    for name in required[:-1]:
        expected = listed.get(name)
        actual = sha256(atlas_dir / name)
        if expected != actual:
            raise RuntimeError(f"internal atlas checksum failed for {name}")
    protocol = json.loads((atlas_dir / "protocol.json").read_text())
    if protocol.get("ranking") != "no connectome or external reference data used":
        raise RuntimeError("internal ranking firewall is absent")
    if protocol.get("orientation") != ORIENTATION:
        raise RuntimeError("internal atlas orientation is not target-row/source-column")
    manifest = json.loads((atlas_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("internal atlas manifest is not complete")
    return {
        "internal_manifest_sha256": sha256(atlas_dir / "manifest.json"),
        "internal_protocol_sha256": sha256(atlas_dir / "protocol.json"),
        "internal_validation_sha256": sha256(atlas_dir / "validation.json"),
        "internal_atlas_matrices_sha256": sha256(
            atlas_dir / "atlas_matrices.npz"
        ),
        "internal_checksums_sha256": sha256(atlas_dir / "checksums.sha256"),
        "frozen_hypothesis_queue_sha256": sha256(atlas_dir / "hypothesis_queue.csv"),
        "internal_created_utc": manifest.get("created_utc"),
        "ranking_inputs_used": [],
        "orientation": ORIENTATION,
    }


def _iter_atlas_slices(atlas: np.lib.npyio.NpzFile) -> Iterable[dict[str, object]]:
    methods = atlas["methods"].astype(str).tolist()
    channels = atlas["channels"].astype(str).tolist()
    contexts = atlas["contexts"].astype(str).tolist()
    lags = atlas["source_lag_frames"].astype(int)
    horizons = atlas["horizon_frames"].astype(int)
    if len(set(lags.tolist())) != len(lags) or np.any(lags <= 0):
        raise RuntimeError("atlas source-lag grid must be unique and positive")
    if len(set(horizons.tolist())) != len(horizons) or np.any(horizons <= 0):
        raise RuntimeError("atlas forecast-horizon grid must be unique and positive")
    for method in methods:
        for channel in channels:
            for context in contexts:
                matrices = atlas[f"mean_normalized__{method}__{channel}__{context}"]
                support = atlas[f"valid_fraction__{method}__{context}"]
                expected = (len(lags), len(horizons), len(atlas["neurons"]), len(atlas["neurons"]))
                if matrices.shape != expected or support.shape != (len(lags), len(atlas["neurons"])):
                    raise RuntimeError(f"atlas slice geometry failed for {method}/{channel}/{context}")
                if not np.isfinite(matrices).all() or not np.isfinite(support).all():
                    raise RuntimeError(f"atlas slice is non-finite for {method}/{channel}/{context}")
                for li, lag in enumerate(lags):
                    for hi, horizon in enumerate(horizons):
                        yield {
                            "method": method,
                            "channel": channel,
                            "context": context,
                            "lag_frames": int(lag),
                            "horizon_frames": int(horizon),
                            "matrix": matrices[li, hi],
                            "support": support[li],
                            "support_available": True,
                            "family": "flow_repaired_conditional_generator",
                        }


def _reference_rows(
    slices: Iterable[dict[str, object]],
    references: dict[str, dict[str, np.ndarray]],
    *,
    fps: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in slices:
        matrix = np.asarray(item["matrix"])
        support = np.asarray(item["support"]) >= SUPPORT_THRESHOLD
        support_available = bool(item.get("support_available", True))
        for reference_name in PRIMARY_REFERENCES:
            reference = references[reference_name]
            scopes = [("all_estimated", reference["mask"])]
            if support_available:
                scopes.append(
                    (
                        "support_qualified_sources",
                        reference["mask"] & support[None, :],
                    )
                )
            for scope, mask in scopes:
                rows.append(
                    {
                        **{k: item[k] for k in ("method", "channel", "context", "lag_frames", "horizon_frames")},
                        **_timing_fields(item, fps),
                        "reference": reference_name,
                        "scope": scope,
                        "support_gate_applicable": support_available,
                        "support_threshold": SUPPORT_THRESHOLD if support_available else None,
                        "n_support_sources": int(support.sum()) if support_available else None,
                        "training_lineage": item.get(
                            "training_lineage", "reviewed 54-neuron conditional-flow atlas"
                        ),
                        "shared_neuron_comparability": item.get(
                            "shared_neuron_comparability", "native reviewed 54-neuron axis"
                        ),
                        **_reference_semantics(reference_name),
                        **binary_metrics(matrix, reference["labels"], mask),
                        **source_macro_metrics(matrix, reference["labels"], mask),
                        **correlation_metrics(matrix, reference, mask),
                    }
                )
    return rows


def _neuromodulator_rows(
    slices: Iterable[dict[str, object]],
    networks: dict[str, np.ndarray],
    *,
    fps: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    d = next(iter(networks.values())).shape[0]
    off = ~np.eye(d, dtype=bool)
    for item in slices:
        matrix = np.asarray(item["matrix"])
        support = np.asarray(item["support"]) >= SUPPORT_THRESHOLD
        support_available = bool(item.get("support_available", True))
        for network_name, labels in networks.items():
            eligible = labels.any(axis=0)
            scopes = [
                ("all_pairs_legacy", off),
                ("eligible_sources", off & eligible[None, :]),
            ]
            if support_available:
                scopes.append(
                    (
                        "eligible_support_qualified",
                        off & eligible[None, :] & support[None, :],
                    )
                )
            for scope, mask in scopes:
                rows.append(
                    {
                        **{k: item[k] for k in ("method", "channel", "context", "lag_frames", "horizon_frames")},
                        **_timing_fields(item, fps),
                        "network": network_name,
                        "scope": scope,
                        "reference_target_type": "binary receptor/pathway-edge existence",
                        "binary_metric_definition": (
                            "absolute model effect versus receptor/pathway-edge existence"
                        ),
                        "support_gate_applicable": support_available,
                        "support_threshold": SUPPORT_THRESHOLD if support_available else None,
                        "n_eligible_sources": int(eligible.sum()),
                        "n_supported_eligible_sources": (
                            int(np.sum(eligible & support)) if support_available else None
                        ),
                        "training_lineage": item.get(
                            "training_lineage", "reviewed 54-neuron conditional-flow atlas"
                        ),
                        "shared_neuron_comparability": item.get(
                            "shared_neuron_comparability", "native reviewed 54-neuron axis"
                        ),
                        **binary_metrics(matrix, labels, mask),
                        **source_macro_metrics(matrix, labels, mask),
                    }
                )
    return rows


def _load_sbtg(
    sbtg_archive: Path, neurons: tuple[str, ...]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with np.load(sbtg_archive, allow_pickle=False) as data:
        if tuple(data["neurons"].astype(str)) != neurons:
            raise RuntimeError("SBTG comparator neuron order does not match the reviewed atlas")
        for method in ("sbtg_current", "sbtg_published"):
            matrix_key = (
                f"{method}__matrices"
                if f"{method}__matrices" in data.files
                else f"{method}__signed"
            )
            if matrix_key not in data.files or f"{method}__lags" not in data.files:
                raise RuntimeError(f"SBTG comparator archive lacks {method} matrices/lags")
            matrices = data[matrix_key].astype(np.float32)
            lags = data[f"{method}__lags"].astype(int)
            expected = (len(lags), len(neurons), len(neurons))
            if matrices.shape != expected:
                raise RuntimeError(
                    f"SBTG comparator geometry mismatch for {method}: {matrices.shape} != {expected}"
                )
            if (
                not len(lags)
                or len(set(lags.tolist())) != len(lags)
                or np.any(lags <= 0)
                or not np.isfinite(matrices).all()
            ):
                raise RuntimeError(f"SBTG comparator values/lags are invalid for {method}")
            # The aligned historical archives do not carry the repaired-flow
            # ESS/weight compatibility diagnostic.  Never synthesize that gate
            # or label SBTG rows as support-qualified.
            support = np.ones(len(neurons), dtype=np.float32)
            for index, lag in enumerate(lags):
                rows.append(
                    {
                        "method": method,
                        "channel": "score_product",
                        "context": "all_windows",
                        "lag_frames": int(lag),
                        "horizon_frames": None,
                        "matrix": matrices[index],
                        "support": support,
                        "support_available": False,
                        "family": "sbtg_historical",
                        "training_lineage": SBTG_LINEAGE[method],
                        "shared_neuron_comparability": (
                            "exact target-row/source-column 54-node subset; training lineage not matched"
                        ),
                    }
                )
    return rows


def _lagmax_rows(
    selected: dict[str, list[dict[str, object]]],
    networks: dict[str, np.ndarray],
    *,
    permutations: int,
    seed: int,
    lag_grid: str,
    fps: float = 4.0,
    forced_lags: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for family, items in selected.items():
        if not items:
            raise RuntimeError(f"planned lag-max family is empty: {family}")
        ordered = sorted(items, key=lambda x: int(x["lag_frames"]))
        if len({int(item["lag_frames"]) for item in ordered}) != len(ordered):
            raise RuntimeError(f"planned lag-max family has duplicate lags: {family}")
        if forced_lags is not None:
            available = {int(item["lag_frames"]): item for item in ordered}
            if not set(forced_lags).issubset(available):
                missing = sorted(set(forced_lags).difference(available))
                raise RuntimeError(
                    f"planned common lag grid is unavailable for {family}: missing {missing}"
                )
            ordered = [available[lag] for lag in forced_lags]
        matrices = [np.asarray(item["matrix"]) for item in ordered]
        lags = [int(item["lag_frames"]) for item in ordered]
        support_states = {bool(item.get("support_available", True)) for item in ordered}
        if len(support_states) != 1:
            raise RuntimeError(f"support semantics change across lags in {family}")
        support_available = support_states.pop()
        common_support = np.ones(matrices[0].shape[1], dtype=bool)
        if support_available:
            common_support = np.logical_and.reduce(
                [
                    np.asarray(item["support"], dtype=float) >= SUPPORT_THRESHOLD
                    for item in ordered
                ]
            )
        for network_name, labels in networks.items():
            reference_eligible = labels.any(axis=0)
            eligible = reference_eligible & common_support
            mask = (~np.eye(len(labels), dtype=bool)) & eligible[None, :]
            y = labels[mask]
            template = ordered[0]
            timing = _timing_fields(template, fps)
            base_row = {
                "lag_grid": lag_grid,
                "family": family,
                "method": template["method"],
                "channel": template["channel"],
                "context": template["context"],
                "horizon_frames": (
                    int(template["horizon_frames"])
                    if template["horizon_frames"] is not None
                    else None
                ),
                "network": network_name,
                "n_reference_eligible_sources": int(reference_eligible.sum()),
                "n_eligible_sources": int(eligible.sum()),
                "n_common_supported_eligible_sources": int(eligible.sum()),
                "support_gate_applicable": support_available,
                "support_scope": (
                    "intersection_across_all_candidate_lags"
                    if support_available
                    else "all_shared_sources_sbtg_validity_one"
                ),
                "support_threshold": SUPPORT_THRESHOLD if support_available else None,
                "n_positive": int(y.sum()),
                "candidate_lags_frames": ",".join(str(x) for x in lags),
                "lag_semantics": timing["lag_semantics"],
                "timing_comparability": timing["timing_comparability"],
                "inference_limit": (
                    "reference_correspondence_not_physical_delay_or_causality"
                ),
            }
            if not len(y) or y.min() == y.max():
                reason = (
                    "no_common_supported_eligible_edges"
                    if not len(y)
                    else "reference_labels_have_one_class_on_common_support"
                )
                rows.append(
                    {
                        **base_row,
                        "evaluable": False,
                        "non_evaluable_reason": reason,
                        "best_lag_frames": None,
                        "best_lag_index_seconds": None,
                        "best_source_to_cut_seconds": None,
                        "forecast_horizon_seconds": timing[
                            "forecast_horizon_seconds"
                        ],
                        "best_source_to_readout_seconds": None,
                        "best_auroc": np.nan,
                        "max_lag_permutation_p": np.nan,
                        "null_max_mean": np.nan,
                        "null_max_95pct": np.nan,
                    }
                )
                continue
            observed = np.asarray(
                [roc_auc_score(y, np.abs(matrix[mask])) for matrix in matrices]
            )
            best = int(np.argmax(observed))
            null_max = np.empty(permutations, dtype=np.float64)
            for repeat in range(permutations):
                permuted = permute_within_source(labels, eligible, rng)
                null_max[repeat] = max(
                    roc_auc_score(permuted[mask], np.abs(matrix[mask]))
                    for matrix in matrices
                )
            template = ordered[best]
            timing = _timing_fields(
                {**template, "lag_frames": lags[best]}, fps
            )
            rows.append(
                {
                    **base_row,
                    "evaluable": True,
                    "non_evaluable_reason": "",
                    "best_lag_frames": lags[best],
                    "best_lag_index_seconds": lags[best] / fps,
                    "best_source_to_cut_seconds": timing["source_to_cut_seconds"],
                    "forecast_horizon_seconds": timing["forecast_horizon_seconds"],
                    "best_source_to_readout_seconds": timing[
                        "source_to_readout_seconds"
                    ],
                    "lag_semantics": timing["lag_semantics"],
                    "timing_comparability": timing["timing_comparability"],
                    "best_auroc": float(observed[best]),
                    "max_lag_permutation_p": float(
                        (1 + np.sum(null_max >= observed[best])) / (permutations + 1)
                    ),
                    "null_max_mean": float(null_max.mean()),
                    "null_max_95pct": float(np.quantile(null_max, 0.95)),
                }
            )
    return rows


def _dashboard_external_summary(
    reference_frame: pd.DataFrame,
    neuromod_frame: pd.DataFrame,
    lagmax: pd.DataFrame,
) -> pd.DataFrame:
    """Build the single bounded post-freeze CSV consumed by the dashboard."""

    rows: list[dict[str, object]] = []

    def timing_label(row: pd.Series, *, selected: bool = False) -> str:
        lag_field = "best_lag_frames" if selected else "lag_frames"
        lag = row.get(lag_field)
        if pd.isna(lag):
            return "not evaluable"
        if str(row.get("method", "")).startswith("sbtg"):
            return f"historical SBTG lag={int(lag)} frames; flow forecast N/A"
        horizon = row.get("horizon_frames")
        return (
            f"source-to-cut={int(lag)} frames; forecast="
            f"{int(horizon)} frames; source-to-readout={int(lag) + int(horizon)} frames"
        )

    def append_descriptive(row: pd.Series, panel: str, key: str) -> None:
        rows.append(
            {
                "panel": panel,
                "method": row["method"],
                "channel": row["channel"],
                "context": row["context"],
                "reference_or_network": row[key],
                "timing_label": timing_label(row),
                "scope_or_grid": row["scope"],
                "auroc": row.get("auroc", np.nan),
                "auprc": row.get("auprc", np.nan),
                "continuous_spearman": row.get("absolute_spearman", np.nan),
                "permutation_p": np.nan,
                "bh_q": np.nan,
                "comparison_note": (
                    "post-freeze convergence only; never used for internal ranking or evidence tiers"
                ),
            }
        )

    flow_reference = reference_frame[
        reference_frame["method"].isin(
            ["progressive_bridge_smc", "direct_importance"]
        )
        & (reference_frame["channel"] == "endpoint_mean")
        & (reference_frame["context"] == "state_average")
        & (reference_frame["horizon_frames"] == 1)
        & (reference_frame["scope"] == "support_qualified_sources")
    ]
    for _, row in flow_reference.iterrows():
        append_descriptive(row, "randi_cook_prespecified_flow_h1", "reference")

    sbtg_reference = reference_frame[
        reference_frame["method"].isin(["sbtg_current", "sbtg_published"])
        & reference_frame["lag_frames"].isin([1, 4, 8, 16])
        & (reference_frame["scope"] == "all_estimated")
    ]
    for _, row in sbtg_reference.iterrows():
        append_descriptive(
            row,
            "randi_cook_contextual_sbtg_shared54_lineage_mismatch",
            "reference",
        )

    primary_neuromod = neuromod_frame[
        neuromod_frame["method"].isin(
            ["progressive_bridge_smc", "direct_importance"]
        )
        & neuromod_frame["channel"].isin(
            ["endpoint_mean", "endpoint_log_sd", "endpoint_wasserstein1"]
        )
        & neuromod_frame["context"].isin(
            ["state_average", "onset_minus_baseline"]
        )
        & (neuromod_frame["lag_frames"] == 1)
        & (neuromod_frame["horizon_frames"] == 1)
        & (neuromod_frame["scope"] == "eligible_support_qualified")
    ]
    for _, row in primary_neuromod.iterrows():
        append_descriptive(row, "neuromodulator_primary_lag1_h1", "network")

    for _, row in lagmax[lagmax["lag_grid"] == "native_method_grid"].iterrows():
        rows.append(
            {
                "panel": "neuromodulator_native_grid_lagmax",
                "method": row["method"],
                "channel": row["channel"],
                "context": row["context"],
                "reference_or_network": row["network"],
                "timing_label": timing_label(row, selected=True),
                "scope_or_grid": row["lag_grid"],
                "auroc": row.get("best_auroc", np.nan),
                "auprc": np.nan,
                "continuous_spearman": np.nan,
                "permutation_p": row.get("max_lag_permutation_p", np.nan),
                "bh_q": row.get("max_lag_bh_q", np.nan),
                "comparison_note": (
                    "source-preserving max-lag null; global BH; not a physical-delay test"
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty or len(result) > 250:
        raise RuntimeError(
            f"dashboard external summary must contain 1..250 rows, observed {len(result)}"
        )
    return result


def run_external_analysis(
    atlas_dir: Path,
    output_dir: Path,
    *,
    reference_release: Path,
    sbtg_archive: Path,
    permutations: int = 999,
    seed: int = 20_260_829,
    overwrite: bool = False,
) -> dict[str, object]:
    atlas_dir = atlas_dir.resolve()
    output_dir = output_dir.resolve()
    reference_release = reference_release.resolve()
    sbtg_archive = sbtg_archive.resolve()
    if permutations < 99:
        raise ValueError("at least 99 max-lag permutations are required")
    for protected in (atlas_dir, reference_release, sbtg_archive):
        if _paths_overlap(output_dir, protected):
            raise ValueError(
                f"external output must be disjoint from protected input: {protected}"
            )
    if not sbtg_archive.is_file():
        raise FileNotFoundError(sbtg_archive)
    firewall = _verify_internal_atlas(atlas_dir)
    reference_hashes = _reference_release_hashes(reference_release)
    sbtg_hash = sha256(sbtg_archive)
    sbtg_provenance_hashes = _verify_sbtg_archive_provenance(sbtg_archive)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        unexpected = [path for path in output_dir.iterdir() if path.name not in OUTPUT_FILES]
        if unexpected:
            raise RuntimeError(
                "overwrite refused because output contains unrelated paths: "
                + ", ".join(sorted(path.name for path in unexpected))
            )
        for name in OUTPUT_FILES:
            path = output_dir / name
            if path.is_file():
                path.unlink()

    protocol = json.loads((atlas_dir / "protocol.json").read_text())
    with np.load(atlas_dir / "atlas_matrices.npz", allow_pickle=False) as atlas:
        if "orientation" not in atlas.files or str(atlas["orientation"].item()) != ORIENTATION:
            raise RuntimeError("dense atlas orientation is not target-row/source-column")
        neurons = tuple(atlas["neurons"].astype(str))
        if len(neurons) != 54 or len(set(neurons)) != 54:
            raise RuntimeError("external analysis requires the reviewed 54-neuron axis")
        methods = atlas["methods"].astype(str).tolist()
        channels = atlas["channels"].astype(str).tolist()
        contexts = atlas["contexts"].astype(str).tolist()
        fps = float(protocol["fps"])
        if (
            atlas["source_lag_frames"].astype(int).tolist()
            != [int(value) for value in protocol["source_lag_frames"]]
            or atlas["horizon_frames"].astype(int).tolist()
            != [int(value) for value in protocol["horizon_frames"]]
            or channels != [str(value) for value in protocol["channels"]]
            or contexts
            != [str(value["context"]) for value in protocol["contexts"]]
        ):
            raise RuntimeError("dense atlas timing/channel/context metadata differs from protocol")
        atlas_slices = list(_iter_atlas_slices(atlas))

    references, networks = load_references(reference_release, list(neurons))
    expected_shape = (len(neurons), len(neurons))
    for name, reference in references.items():
        for field in ("labels", "mask"):
            if np.asarray(reference[field]).shape != expected_shape:
                raise RuntimeError(f"reference geometry mismatch: {name}/{field}")
    for name, network in networks.items():
        if np.asarray(network).shape != expected_shape:
            raise RuntimeError(f"neuromodulator geometry mismatch: {name}")
    actual_sources = {name: int(matrix.any(axis=0).sum()) for name, matrix in networks.items()}
    if actual_sources != EXPECTED_NEUROMODULATOR_SOURCES:
        raise RuntimeError(f"neuromodulator source-count audit failed: {actual_sources}")

    sbtg_slices = _load_sbtg(sbtg_archive, neurons)
    all_slices = atlas_slices + sbtg_slices
    reference_frame = pd.DataFrame(_reference_rows(all_slices, references, fps=fps))
    neuromod_frame = pd.DataFrame(_neuromodulator_rows(all_slices, networks, fps=fps))
    reference_frame.to_csv(output_dir / "randi_cook_metrics.csv", index=False)
    neuromod_frame.to_csv(output_dir / "neuromodulator_metrics.csv", index=False)

    selected: dict[str, list[dict[str, object]]] = {}
    for method, channel, context, horizon in PRIMARY_LAGMAX_SLICES:
        key = f"{method}__{channel}__{context}__h{horizon}"
        selected[key] = [
            item
            for item in atlas_slices
            if item["method"] == method
            and item["channel"] == channel
            and item["context"] == context
            and int(item["horizon_frames"]) == horizon
        ]
    for method in ("sbtg_current", "sbtg_published"):
        selected[method] = [item for item in sbtg_slices if item["method"] == method]
    native = _lagmax_rows(
        selected,
        networks,
        permutations=permutations,
        seed=seed,
        lag_grid="native_method_grid",
        fps=fps,
    )
    common = _lagmax_rows(
        selected,
        networks,
        permutations=permutations,
        seed=seed + 1,
        lag_grid="common_1_8_frames",
        fps=fps,
        forced_lags=(1, 8),
    )
    lagmax = pd.DataFrame(native + common)
    if lagmax.empty:
        raise RuntimeError("planned neuromodulator lag-max analysis produced no tests")
    p_values = lagmax["max_lag_permutation_p"].to_numpy(dtype=float)
    finite_tests = np.isfinite(p_values)
    adjusted = np.full(len(lagmax), np.nan, dtype=float)
    if finite_tests.any():
        adjusted[finite_tests] = benjamini_hochberg(p_values[finite_tests])
    lagmax["max_lag_bh_q"] = adjusted
    lagmax["bh_family"] = GLOBAL_BH_FAMILY
    lagmax["n_bh_tests"] = int(finite_tests.sum())
    lagmax.to_csv(output_dir / "neuromodulator_lagmax_inference.csv", index=False)

    dashboard_summary = _dashboard_external_summary(
        reference_frame, neuromod_frame, lagmax
    )
    dashboard_summary.to_csv(
        output_dir / "dashboard_external_summary.csv", index=False
    )

    primary_mask = (
        (reference_frame["method"] == "progressive_bridge_smc")
        & (reference_frame["channel"] == "endpoint_mean")
        & (reference_frame["context"] == "state_average")
        & (reference_frame["horizon_frames"] == 1)
        & (reference_frame["scope"] == "support_qualified_sources")
    )
    primary_reference = reference_frame[primary_mask].copy()
    primary_reference.to_csv(output_dir / "primary_randi_cook_profile.csv", index=False)
    primary_neuromod = neuromod_frame[
        (neuromod_frame["method"] == "progressive_bridge_smc")
        & (neuromod_frame["channel"].isin(["endpoint_mean", "endpoint_log_sd", "endpoint_wasserstein1"]))
        & (neuromod_frame["context"].isin(["state_average", "onset_minus_baseline"]))
        & (neuromod_frame["horizon_frames"] == 1)
        & (neuromod_frame["scope"] == "eligible_support_qualified")
    ].copy()
    primary_neuromod.to_csv(output_dir / "primary_neuromodulator_profile.csv", index=False)

    if _verify_internal_atlas(atlas_dir) != firewall:
        raise RuntimeError("internal atlas changed during post-freeze external analysis")
    if _reference_release_hashes(reference_release) != reference_hashes:
        raise RuntimeError("external reference release changed during analysis")
    if sha256(sbtg_archive) != sbtg_hash:
        raise RuntimeError("SBTG comparator archive changed during analysis")
    if _verify_sbtg_archive_provenance(sbtg_archive) != sbtg_provenance_hashes:
        raise RuntimeError("SBTG comparator provenance changed during analysis")
    input_checksum_lines = [
        f"{digest}  reference_release/{relative}\n"
        for relative, digest in sorted(reference_hashes.items())
    ]
    input_checksum_lines.extend(
        [
            (
                f"{firewall['internal_atlas_matrices_sha256']}  "
                "internal_atlas/atlas_matrices.npz\n"
            ),
            (
                f"{firewall['internal_checksums_sha256']}  "
                "internal_atlas/checksums.sha256\n"
            ),
            (
                f"{firewall['frozen_hypothesis_queue_sha256']}  "
                "internal_atlas/hypothesis_queue.csv\n"
            ),
        ]
    )
    input_checksum_lines.extend(
        f"{digest}  sbtg_archive/{name}\n"
        for name, digest in sorted(sbtg_provenance_hashes.items())
    )
    (output_dir / "input_checksums.sha256").write_text(
        "".join(input_checksum_lines)
    )

    manifest = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "post_freeze_external_convergence_only",
        "ranking_effect": "none; internal hypothesis queue and evidence tiers are immutable",
        "claim_boundary": (
            "Randi/Cook/Bentley are imperfect structural, perturbational, or receptor/pathway "
            "references, not neural-activity ground truth; lag maxima are not physical delays."
        ),
        "internal_firewall": firewall,
        "internal_firewall_reverified_after_analysis": True,
        "internal_prediction_ranking_inputs": [],
        "reference_release": str(reference_release),
        "reference_release_files_are_read_only": True,
        "reference_release_file_sha256": reference_hashes,
        "input_checksum_inventory": "input_checksums.sha256",
        "sbtg_archive": str(sbtg_archive),
        "sbtg_archive_sha256": sbtg_hash,
        "sbtg_archive_provenance_sha256": sbtg_provenance_hashes,
        "sbtg_shared_neuron_comparability": (
            "both matrices are in the exact reviewed target-row/source-column 54-neuron "
            "order; SBTG-published retains its historical 80-neuron pseudo-paired/"
            "donor-imputed training lineage and is contextual only"
        ),
        "sbtg_training_lineages": SBTG_LINEAGE,
        "timing_semantics": {
            "flow": (
                "source lag is source-window-end to prediction cut; horizon is cut to "
                "readout; source-to-readout is their sum"
            ),
            "sbtg": (
                "historical score-product time_lag; frame-number comparisons are time-bin "
                "correspondence only, with forecast and source-to-readout undefined"
            ),
        },
        "metric_semantics": {
            "binary": (
                "AUROC/AUPRC use absolute model-effect magnitude against binary response, "
                "anatomical-edge, or receptor/pathway-edge existence"
            ),
            "continuous": (
                "Spearman metrics are reported separately for Randi dff and Cook connection weights"
            ),
        },
        "support_semantics": (
            f"flow descriptive support uses source valid_fraction >= {SUPPORT_THRESHOLD}; "
            "flow lag-max tests use the intersection across every candidate lag; SBTG has "
            "no repaired-flow compatibility gate and uses all shared sources (validity one)"
        ),
        "lagmax_null": (
            "reference labels permuted within each eligible source column, shared across "
            "candidate lags within a permutation; maximum AUROC calibrates lag selection"
        ),
        "bh_family": GLOBAL_BH_FAMILY,
        "n_bh_tests": int(finite_tests.sum()),
        "methods": methods + ["sbtg_current", "sbtg_published"],
        "channels": channels,
        "contexts": contexts,
        "neuromodulator_source_counts": actual_sources,
        "permutations": permutations,
        "inference_seed": seed,
        "rows": {
            "randi_cook_metrics": len(reference_frame),
            "neuromodulator_metrics": len(neuromod_frame),
            "neuromodulator_lagmax_inference": len(lagmax),
            "dashboard_external_summary": len(dashboard_summary),
        },
        "orientation_audit": "input atlas is target-row/source-column; reference matrices aligned to identical neuron order",
        "dashboard_external_summary": (
            "dashboard_external_summary.csv is bounded to <=250 post-freeze comparison rows "
            "and cannot modify the internal hypothesis queue"
        ),
    }
    _json_dump(output_dir / "manifest.json", manifest)
    paths = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in paths)
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-release", type=Path, required=True)
    parser.add_argument("--sbtg-archive", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20_260_829)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_external_analysis(
        args.atlas_dir,
        args.output_dir,
        reference_release=args.reference_release,
        sbtg_archive=args.sbtg_archive,
        permutations=args.permutations,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
