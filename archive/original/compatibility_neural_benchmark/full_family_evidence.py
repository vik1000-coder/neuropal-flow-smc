"""Combine exact complete-family NeuroPAL runs into one evidence bundle.

The four input families share the same exact worm-level sign-pattern order.
Their saved null maxima can therefore be combined elementwise to form a joint
single-step max-T reference distribution without rerunning the sampler or the
conditional generator.  The resulting p-values control the joint primary
search over all support-eligible cells in all four families.

This bundle is deliberately an internal statistical-evidence layer.  No row is
called experiment-ready: sham/null-distance calibration, a prespecified
smallest effect size of interest, and independent biological confirmation are
all still pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.full_family_inference import (
    _pinned_file_provenance,
    benjamini_hochberg,
)


EXPECTED_FAMILIES: Mapping[str, tuple[str, str]] = {
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
EDGE_KEY = ["family_id", "source_index", "target_index"]
CELL_KEY = EDGE_KEY + ["source_lag_frames", "horizon_frames"]
QUEUE_KEY = [
    "channel",
    "context",
    "source_index",
    "target_index",
    "source_lag_frames",
    "horizon_frames",
]
SAMPLING_NULL_COMBINED_COLUMNS = [
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
PRACTICAL_STATUS = "pending_sham_and_experimental_threshold"
LEGACY_Q_LABEL = "post_screen_legacy_not_full_family_fdr"
ALPHA = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(directory: Path) -> dict[str, object]:
    """Verify every entry in one run's checksums file."""
    root = Path(directory).resolve()
    checksum_path = root / "checksums.sha256"
    if not checksum_path.exists():
        raise FileNotFoundError(checksum_path)
    checked: list[str] = []
    for line_number, raw in enumerate(checksum_path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            expected, name = line.split(None, 1)
        except ValueError as error:
            raise ValueError(
                f"malformed checksum line {line_number} in {checksum_path}"
            ) from error
        name = name.strip()
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise ValueError(
                f"checksum mismatch for {path}: expected {expected}, got {observed}"
            )
        checked.append(name)
    if not checked:
        raise ValueError(f"no checksum entries found in {checksum_path}")
    return {
        "directory": str(root),
        "checksums_file": str(checksum_path),
        "verified_files": checked,
        "verified_count": len(checked),
    }


def exact_upper_tail_p(null_maximum: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Exact upper-tail p-values from an observed-containing sign orbit."""
    null = np.asarray(null_maximum, dtype=np.float64)
    values = np.asarray(observed, dtype=np.float64)
    if null.ndim != 1 or not len(null) or not np.isfinite(null).all():
        raise ValueError("null maximum must be a nonempty finite vector")
    ordered = np.sort(null)
    flat = values.reshape(-1)
    out = np.full(flat.shape, np.nan, dtype=np.float64)
    keep = np.isfinite(flat)
    if keep.any():
        first_ge = np.searchsorted(ordered, flat[keep], side="left")
        out[keep] = (len(ordered) - first_ge) / float(len(ordered))
    return out.reshape(values.shape)


def _json_dump(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.copy().astype(object)
    clean[pd.isna(clean)] = None
    return clean.to_dict(orient="records")


@dataclass(frozen=True)
class FamilyInput:
    family_id: str
    channel: str
    context: str
    directory: Path
    edge: pd.DataFrame
    cell: pd.DataFrame
    null_max: np.ndarray
    lag_null_max: np.ndarray
    signs: np.ndarray
    protocol: dict[str, object]
    summary: dict[str, object]
    input_sha256: str
    checksum_validation: dict[str, object]


def _load_family(directory: Path, family_id: str) -> FamilyInput:
    root = Path(directory).resolve()
    if family_id not in EXPECTED_FAMILIES:
        raise ValueError(f"unexpected family id {family_id!r}")
    channel, context = EXPECTED_FAMILIES[family_id]
    checksum_validation = verify_checksums(root)
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete" or summary.get("status") != "complete":
        raise ValueError(f"family {family_id} is not complete")
    if summary.get("channel") != channel or summary.get("context") != context:
        raise ValueError(
            f"family {family_id} has channel/context "
            f"{summary.get('channel')}/{summary.get('context')}"
        )
    randomization = protocol.get("randomization", {})
    if randomization.get("resolved_mode") != "exact":
        raise ValueError(f"family {family_id} is not an exact sign-flip run")
    family_protocol = protocol.get("family", {})
    if int(family_protocol.get("directed_edges", -1)) != 54 * 53:
        raise ValueError(f"family {family_id} is not the complete 54-neuron edge family")

    edge = pd.read_csv(root / "edge_inference.csv")
    cell = pd.read_csv(root / "cell_inference.csv")
    required_edge = {
        "source_index",
        "target_index",
        "source_neuron",
        "target_neuron",
        "support_eligible",
        "edge_max_abs_t",
        "edge_omnibus_p_value",
        "edge_bh_q_value",
        "flat_lag_max_abs_t",
        "flat_lag_omnibus_p_value",
        "flat_lag_bh_q_value",
    }
    required_cell = {
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
        "support_eligible",
        "mean_normalized",
        "standard_error",
        "student_t",
        "pointwise_sign_flip_p_value",
        "global_max_t_p_value",
        "lag_contrast_student_t",
        "lag_contrast_global_max_t_p_value",
    }
    if missing := sorted(required_edge.difference(edge.columns)):
        raise KeyError(f"{family_id} edge table is missing {missing}")
    if missing := sorted(required_cell.difference(cell.columns)):
        raise KeyError(f"{family_id} cell table is missing {missing}")
    if len(edge) != 54 * 53:
        raise ValueError(f"{family_id} edge table has {len(edge)} rows")
    if edge.duplicated(["source_index", "target_index"]).any():
        raise ValueError(f"{family_id} edge keys are not unique")
    if cell.duplicated(
        ["source_index", "target_index", "source_lag_frames", "horizon_frames"]
    ).any():
        raise ValueError(f"{family_id} cell keys are not unique")
    if (edge.source_index == edge.target_index).any():
        raise ValueError(f"{family_id} contains diagonal edges")

    with np.load(root / "inference_arrays.npz", allow_pickle=False) as arrays:
        signs = np.asarray(arrays["sign_patterns"], dtype=np.int8)
        null_max = np.asarray(arrays["global_max_abs_t"], dtype=np.float64)
        lag_null_max = np.asarray(
            arrays["global_max_abs_lag_contrast_t"], dtype=np.float64
        )
        stored_channel = str(np.asarray(arrays["channel"]).item())
        stored_context = str(np.asarray(arrays["context"]).item())
    if stored_channel != channel or stored_context != context:
        raise ValueError(f"{family_id} NPZ channel/context metadata disagree")
    if signs.ndim != 2 or len(signs) != len(null_max):
        raise ValueError(f"{family_id} sign/null dimensions disagree")
    if len(lag_null_max) != len(signs) or not np.isfinite(lag_null_max).all():
        raise ValueError(f"{family_id} lag-null dimensions are invalid")
    sign_hash = hashlib.sha256(signs.tobytes()).hexdigest()
    if sign_hash != randomization.get("sign_patterns_sha256"):
        raise ValueError(f"{family_id} sign-pattern hash disagrees with protocol")
    input_sha256 = str(protocol.get("input", {}).get("sha256", ""))
    if not input_sha256:
        raise ValueError(f"{family_id} lacks its canonical input hash")

    edge.insert(0, "family_id", family_id)
    cell.insert(0, "family_id", family_id)
    return FamilyInput(
        family_id=family_id,
        channel=channel,
        context=context,
        directory=root,
        edge=edge,
        cell=cell,
        null_max=null_max,
        lag_null_max=lag_null_max,
        signs=signs,
        protocol=protocol,
        summary=summary,
        input_sha256=input_sha256,
        checksum_validation=checksum_validation,
    )


def load_and_validate_families(root: Path) -> list[FamilyInput]:
    base = Path(root).resolve()
    families = [
        _load_family(base / family_id, family_id)
        for family_id in EXPECTED_FAMILIES
    ]
    reference = families[0]
    for family in families[1:]:
        if family.input_sha256 != reference.input_sha256:
            raise ValueError("canonical worm-matrix input hashes differ across families")
        if not np.array_equal(family.signs, reference.signs):
            raise ValueError("exact sign-pattern ordering differs across families")
    return families


def _classify_edge(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    eligible = frame["support_eligible"].astype(bool).to_numpy()
    joint_edge = frame["joint_primary_edge_max_t_p_value"].to_numpy() <= alpha
    joint_lag = frame["joint_primary_flat_lag_max_t_p_value"].to_numpy() <= alpha
    exploratory = (
        (frame["edge_bh_q_value"].to_numpy() <= alpha)
        | (frame["global_four_family_edge_bh_q_value"].to_numpy() <= alpha)
        | (frame["edge_omnibus_p_value"].to_numpy() <= alpha)
        | (frame["flat_lag_bh_q_value"].to_numpy() <= alpha)
    )
    label = np.full(len(frame), "no_complete_family_evidence", dtype=object)
    label[eligible & exploratory] = "exploratory"
    label[eligible & joint_edge & ~joint_lag] = "strong_edge_lag_unresolved"
    label[eligible & joint_edge & joint_lag] = "strong_edge_with_lag_structure"
    label[~eligible] = "sampling_limited"
    return label


def _compute_joint_tables(
    families: Sequence[FamilyInput], alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute joint four-family max-T and pooled exploratory BH tables."""
    null_stack = np.stack([family.null_max for family in families])
    lag_null_stack = np.stack([family.lag_null_max for family in families])
    joint_null = np.max(null_stack, axis=0)
    joint_lag_null = np.max(lag_null_stack, axis=0)
    edge = pd.concat([family.edge for family in families], ignore_index=True)
    cell = pd.concat([family.cell for family in families], ignore_index=True)
    edge["joint_primary_edge_max_t_p_value"] = exact_upper_tail_p(
        joint_null, edge["edge_max_abs_t"].to_numpy()
    )
    edge["joint_primary_flat_lag_max_t_p_value"] = exact_upper_tail_p(
        joint_lag_null, edge["flat_lag_max_abs_t"].to_numpy()
    )
    edge_p_for_bh = np.where(
        np.isfinite(edge["edge_omnibus_p_value"]),
        edge["edge_omnibus_p_value"],
        1.0,
    )
    flat_p_for_bh = np.where(
        np.isfinite(edge["flat_lag_omnibus_p_value"]),
        edge["flat_lag_omnibus_p_value"],
        1.0,
    )
    edge["global_four_family_edge_bh_q_value"] = benjamini_hochberg(edge_p_for_bh)
    edge["global_four_family_flat_lag_bh_q_value"] = benjamini_hochberg(
        flat_p_for_bh
    )
    edge_join = edge[
        EDGE_KEY
        + [
            "joint_primary_edge_max_t_p_value",
            "joint_primary_flat_lag_max_t_p_value",
            "global_four_family_edge_bh_q_value",
            "global_four_family_flat_lag_bh_q_value",
        ]
    ]
    cell = cell.merge(edge_join, on=EDGE_KEY, how="left", validate="many_to_one")
    cell["joint_primary_cell_max_t_p_value"] = exact_upper_tail_p(
        joint_null, np.abs(cell["student_t"].to_numpy())
    )
    cell["joint_primary_lag_contrast_max_t_p_value"] = exact_upper_tail_p(
        joint_lag_null, np.abs(cell["lag_contrast_student_t"].to_numpy())
    )
    return edge, cell, joint_null, joint_lag_null


def _join_legacy_queue(cell: pd.DataFrame, queue_path: Path | None) -> pd.DataFrame:
    result = cell.copy()
    result["legacy_queue_match"] = False
    result["legacy_queue_rank"] = np.nan
    result["legacy_postscreen_q_value"] = np.nan
    result["legacy_inference_scope"] = "not_present"
    result["legacy_q_label"] = "not_present"
    if queue_path is None:
        return result
    path = Path(queue_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    queue = pd.read_csv(path)
    missing = sorted(set(QUEUE_KEY + ["queue_rank", "sign_flip_q_value", "inference_scope"]).difference(queue.columns))
    if missing:
        raise KeyError(f"legacy queue is missing columns {missing}")
    if queue.duplicated(QUEUE_KEY).any():
        raise ValueError("legacy queue exact join keys are not unique")
    legacy = queue[
        QUEUE_KEY + ["queue_rank", "sign_flip_q_value", "inference_scope"]
    ].rename(
        columns={
            "queue_rank": "legacy_queue_rank_joined",
            "sign_flip_q_value": "legacy_postscreen_q_value_joined",
            "inference_scope": "legacy_inference_scope_joined",
        }
    )
    result = result.merge(legacy, on=QUEUE_KEY, how="left", validate="many_to_one")
    matched = result["legacy_queue_rank_joined"].notna()
    result.loc[matched, "legacy_queue_match"] = True
    result.loc[matched, "legacy_queue_rank"] = result.loc[
        matched, "legacy_queue_rank_joined"
    ]
    result.loc[matched, "legacy_postscreen_q_value"] = result.loc[
        matched, "legacy_postscreen_q_value_joined"
    ]
    result.loc[matched, "legacy_inference_scope"] = result.loc[
        matched, "legacy_inference_scope_joined"
    ]
    result.loc[matched, "legacy_q_label"] = LEGACY_Q_LABEL
    return result.drop(
        columns=[
            "legacy_queue_rank_joined",
            "legacy_postscreen_q_value_joined",
            "legacy_inference_scope_joined",
        ]
    )


def _select_sham_shortlist(cell: pd.DataFrame, count: int = 6) -> pd.DataFrame:
    pool = cell.loc[
        (cell["family_id"] == "baseline_endpoint_mean")
        & cell["support_eligible"].astype(bool)
        & (cell["joint_primary_edge_max_t_p_value"] <= ALPHA)
    ].copy()
    if len(pool) < count:
        raise RuntimeError("fewer than six support-eligible joint-maxT edge cells")
    pool["joint_cell_pass"] = pool["joint_primary_cell_max_t_p_value"] <= ALPHA
    pool["absolute_student_t"] = pool["student_t"].abs()
    pool = pool.sort_values(
        [
            "joint_cell_pass",
            "absolute_student_t",
            "source_index",
            "target_index",
            "source_lag_frames",
            "horizon_frames",
        ],
        ascending=[False, False, True, True, True, True],
        kind="stable",
    )
    chosen: list[int] = []
    seen_sources: set[int] = set()
    for index, row in pool.iterrows():
        source = int(row.source_index)
        if source in seen_sources:
            continue
        chosen.append(index)
        seen_sources.add(source)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        for index in pool.index:
            if index not in chosen:
                chosen.append(index)
            if len(chosen) == count:
                break
    shortlist = pool.loc[chosen].copy().reset_index(drop=True)
    shortlist.insert(0, "shortlist_rank", np.arange(1, len(shortlist) + 1))
    shortlist["selection_rule"] = (
        "baseline endpoint_mean; support-eligible joint-maxT edge; prioritize "
        "joint-maxT localized cells then descending absolute studentized effect; "
        "take distinct source neurons first; deterministic index/lag/horizon tie-break"
    )
    shortlist["planned_sham_controls"] = (
        "independent A/B same-arm comparisons; independent midpoint zero-gap; "
        "matched quiet pseudo-cuts; parent-safe within-arm split"
    )
    shortlist["sesoi_status"] = "not_defined_pending_sham_calibration"
    shortlist["independent_confirmation_status"] = "not_run"
    shortlist["experiment_ready"] = False
    shortlist["practical_status"] = PRACTICAL_STATUS
    keep = [
        "shortlist_rank",
        "family_id",
        "channel",
        "context",
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "source_lag_seconds",
        "horizon_frames",
        "horizon_seconds",
        "mean_normalized",
        "standard_error",
        "student_t",
        "joint_primary_cell_max_t_p_value",
        "joint_primary_edge_max_t_p_value",
        "joint_primary_flat_lag_max_t_p_value",
        "evidence_label",
        "cell_localization_label",
        "legacy_queue_match",
        "legacy_queue_rank",
        "legacy_postscreen_q_value",
        "legacy_q_label",
        "selection_rule",
        "planned_sham_controls",
        "sesoi_status",
        "independent_confirmation_status",
        "experiment_ready",
        "practical_status",
    ]
    return shortlist[keep]


def _build_sampling_null_combined_queue(
    primary_shortlist: pd.DataFrame,
    lag_sensitivity_shortlist: pd.DataFrame,
) -> pd.DataFrame:
    """Build the owned six-primary plus two-lag sampling-null queue.

    The output schema and row order are frozen because the raw sampler pins
    this CSV by SHA256.  In particular, ``run_sampling_nulls`` is the literal
    lower-case string ``true`` so pandas emits byte-identical CSV content
    across the canonical build and its downstream runner.
    """
    primary_required = {
        "shortlist_rank",
        "channel",
        "context",
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
    }
    lag_required = {
        "sensitivity_shortlist_rank",
        "family_id",
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
    }
    if missing := sorted(primary_required.difference(primary_shortlist.columns)):
        raise KeyError(f"primary sampling-null shortlist is missing {missing}")
    if missing := sorted(lag_required.difference(lag_sensitivity_shortlist.columns)):
        raise KeyError(f"lag sampling-null shortlist is missing {missing}")
    if len(primary_shortlist) != 6 or len(lag_sensitivity_shortlist) != 2:
        raise ValueError(
            "combined sampling-null queue requires six primary and two lag rows"
        )

    primary = primary_shortlist.sort_values(
        "shortlist_rank", kind="stable"
    ).copy()
    primary["selection_origin"] = "strong_primary"
    lag = lag_sensitivity_shortlist.sort_values(
        "sensitivity_shortlist_rank", kind="stable"
    ).copy()
    if not (lag["family_id"].astype(str) == "baseline_endpoint_mean").all():
        raise ValueError(
            "lag sampling-null rows must belong to baseline_endpoint_mean"
        )
    lag["channel"] = "endpoint_mean"
    lag["context"] = "baseline"
    lag["selection_origin"] = "lag_sensitivity"
    combined = pd.concat([primary, lag], ignore_index=True, sort=False)
    combined.insert(0, "queue_rank", np.arange(1, 9, dtype=np.int64))
    combined["run_sampling_nulls"] = "true"
    for column in (
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
    ):
        numeric = pd.to_numeric(combined[column], errors="raise").to_numpy()
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"combined sampling-null {column} must be integral")
        combined[column] = numeric.astype(np.int64)
    combined = combined[SAMPLING_NULL_COMBINED_COLUMNS]
    if combined.duplicated(QUEUE_KEY).any():
        raise ValueError("combined sampling-null queue keys are not unique")
    if set(combined["selection_origin"]) != {
        "strong_primary",
        "lag_sensitivity",
    }:
        raise RuntimeError("combined sampling-null selection origins are invalid")
    return combined


def _build_sensitivity_supplement(
    families: Sequence[FamilyInput],
    strong_edge: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build the explicitly non-primary 0.5-support supplement."""
    edge, cell, _, _ = _compute_joint_tables(families, alpha)
    edge = edge.rename(
        columns={
            "joint_primary_edge_max_t_p_value": (
                "joint_sensitivity_edge_max_t_p_value"
            ),
            "joint_primary_flat_lag_max_t_p_value": (
                "joint_sensitivity_flat_lag_max_t_p_value"
            ),
            "global_four_family_edge_bh_q_value": (
                "sensitivity_global_four_family_edge_bh_q_value"
            ),
            "global_four_family_flat_lag_bh_q_value": (
                "sensitivity_global_four_family_flat_lag_bh_q_value"
            ),
        }
    )
    cell = cell.rename(
        columns={
            "joint_primary_edge_max_t_p_value": (
                "joint_sensitivity_edge_max_t_p_value"
            ),
            "joint_primary_flat_lag_max_t_p_value": (
                "joint_sensitivity_flat_lag_max_t_p_value"
            ),
            "global_four_family_edge_bh_q_value": (
                "sensitivity_global_four_family_edge_bh_q_value"
            ),
            "global_four_family_flat_lag_bh_q_value": (
                "sensitivity_global_four_family_flat_lag_bh_q_value"
            ),
            "joint_primary_cell_max_t_p_value": (
                "joint_sensitivity_cell_max_t_p_value"
            ),
            "joint_primary_lag_contrast_max_t_p_value": (
                "joint_sensitivity_lag_contrast_max_t_p_value"
            ),
        }
    )
    primary_support = strong_edge[EDGE_KEY + ["support_eligible"]].rename(
        columns={"support_eligible": "strong_primary_support_eligible"}
    )
    edge = edge.merge(primary_support, on=EDGE_KEY, how="left", validate="one_to_one")
    edge["strong_primary_support_eligible"] = edge[
        "strong_primary_support_eligible"
    ].fillna(False).astype(bool)
    edge["sensitivity_support_eligible"] = edge["support_eligible"].astype(bool)
    edge["sensitivity_only"] = (
        edge["sensitivity_support_eligible"]
        & ~edge["strong_primary_support_eligible"]
    )
    # The sensitivity supplement can never create a strong label.  Rows newly
    # admitted by the 0.5 gate remain sampling-limited by construction.
    edge["evidence_label"] = "sampling_limited"
    edge.loc[
        edge["strong_primary_support_eligible"], "evidence_label"
    ] = "strong_gate_overlap_reference"
    edge["experiment_ready"] = False
    edge["practical_status"] = PRACTICAL_STATUS
    sensitivity_edge_join = edge[
        EDGE_KEY
        + [
            "sensitivity_support_eligible",
            "strong_primary_support_eligible",
            "sensitivity_only",
            "joint_sensitivity_edge_max_t_p_value",
            "joint_sensitivity_flat_lag_max_t_p_value",
            "flat_lag_bh_q_value",
            "sensitivity_global_four_family_edge_bh_q_value",
            "sensitivity_global_four_family_flat_lag_bh_q_value",
            "evidence_label",
        ]
    ]
    # Remove duplicated edge-level fields created by _compute_joint_tables so
    # the sensitivity labels are joined from one frozen edge table.
    for column in sensitivity_edge_join.columns:
        if column in EDGE_KEY:
            continue
        if column in cell.columns:
            cell = cell.drop(columns=column)
    cell = cell.merge(
        sensitivity_edge_join, on=EDGE_KEY, how="left", validate="many_to_one"
    )
    cell["experiment_ready"] = False
    cell["practical_status"] = PRACTICAL_STATUS

    family_rows: list[dict[str, object]] = []
    for family in families:
        e = edge.loc[edge.family_id == family.family_id]
        c = cell.loc[cell.family_id == family.family_id]
        family_rows.append(
            {
                "family_id": family.family_id,
                "channel": family.channel,
                "context": family.context,
                "support_gate": "sensitivity_0.5",
                "complete_edges": len(e),
                "sensitivity_support_eligible_edges": int(
                    e.sensitivity_support_eligible.sum()
                ),
                "sensitivity_only_edges": int(e.sensitivity_only.sum()),
                "complete_cells": len(c),
                "sensitivity_support_eligible_cells": int(
                    c.sensitivity_support_eligible.sum()
                ),
                "within_family_edge_bh_discoveries": int(
                    (e.edge_bh_q_value <= alpha).sum()
                ),
                "within_family_flat_lag_bh_discoveries": int(
                    (e.flat_lag_bh_q_value <= alpha).sum()
                ),
                "joint_sensitivity_edge_max_t_discoveries": int(
                    (e.joint_sensitivity_edge_max_t_p_value <= alpha).sum()
                ),
                "joint_sensitivity_flat_lag_edge_discoveries": int(
                    (e.joint_sensitivity_flat_lag_max_t_p_value <= alpha).sum()
                ),
                "joint_sensitivity_lag_contrast_cell_discoveries": int(
                    (c.joint_sensitivity_lag_contrast_max_t_p_value <= alpha).sum()
                ),
                "experiment_ready": 0,
                "practical_status": PRACTICAL_STATUS,
            }
        )
    family_summary = pd.DataFrame(family_rows)

    lag_shortlist = cell.loc[
        (cell.family_id == "baseline_endpoint_mean")
        & (cell.source_neuron == "SMD")
        & cell.target_neuron.isin(["RID", "RIB"])
        & (cell.source_lag_frames == 16)
        & (cell.horizon_frames == 1)
    ].copy()
    if len(lag_shortlist) != 2 or set(lag_shortlist.target_neuron) != {"RID", "RIB"}:
        raise RuntimeError(
            "expected exact SMD→RID and SMD→RIB lag16/h1 sensitivity cells"
        )
    lag_shortlist = lag_shortlist.sort_values(
        "target_neuron", ascending=False, kind="stable"
    ).reset_index(drop=True)
    # Explicit order matches decreasing joint sensitivity evidence: RID first.
    lag_shortlist = lag_shortlist.sort_values(
        "joint_sensitivity_flat_lag_max_t_p_value", kind="stable"
    ).reset_index(drop=True)
    lag_shortlist.insert(0, "sensitivity_shortlist_rank", [1, 2])
    lag_shortlist["selection_rule"] = (
        "the two baseline endpoint_mean edges passing within-family 0.5-support "
        "flat-lag BH; inspect their prespecified peak cell lag16/h1 under the "
        "joint four-family lag max-T null"
    )
    lag_shortlist["planned_sham_controls"] = (
        "independent A/B same-arm comparisons; independent midpoint zero-gap; "
        "matched quiet pseudo-cuts; parent-safe within-arm split"
    )
    lag_shortlist["evidence_label"] = "sampling_limited"
    lag_shortlist["experiment_ready"] = False
    lag_shortlist["practical_status"] = PRACTICAL_STATUS
    lag_keep = [
        "sensitivity_shortlist_rank",
        "family_id",
        "source_neuron",
        "target_neuron",
        "source_index",
        "target_index",
        "source_lag_frames",
        "horizon_frames",
        "lag_contrast_mean",
        "lag_contrast_student_t",
        "flat_lag_bh_q_value",
        "joint_sensitivity_flat_lag_max_t_p_value",
        "joint_sensitivity_lag_contrast_max_t_p_value",
        "sensitivity_only",
        "evidence_label",
        "selection_rule",
        "planned_sham_controls",
        "experiment_ready",
        "practical_status",
    ]
    lag_shortlist = lag_shortlist[lag_keep]
    summary = {
        "support_gate": "sensitivity_0.5",
        "claim_status": "sensitivity_only_never_promotes_primary_claim",
        "complete_edges": len(edge),
        "sensitivity_support_eligible_edges": int(
            edge.sensitivity_support_eligible.sum()
        ),
        "sensitivity_only_edges": int(edge.sensitivity_only.sum()),
        "joint_sensitivity_edge_discoveries": int(
            (edge.joint_sensitivity_edge_max_t_p_value <= alpha).sum()
        ),
        "joint_sensitivity_flat_lag_edge_discoveries": int(
            (edge.joint_sensitivity_flat_lag_max_t_p_value <= alpha).sum()
        ),
        "joint_sensitivity_lag_contrast_cell_discoveries": int(
            (cell.joint_sensitivity_lag_contrast_max_t_p_value <= alpha).sum()
        ),
        "baseline_endpoint_mean_within_family_flat_lag_bh_edges": 2,
        "named_lag_rows": _json_records(lag_shortlist),
        "experiment_ready": 0,
        "practical_status": PRACTICAL_STATUS,
    }
    return edge, cell, family_summary, lag_shortlist, summary


def build_joint_evidence(
    family_root: Path,
    output_dir: Path,
    *,
    sensitivity_family_root: Path | None = None,
    queue_path: Path | None = None,
    alpha: float = ALPHA,
    overwrite: bool = False,
) -> dict[str, object]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    families = load_and_validate_families(family_root)
    legacy_queue_provenance = _pinned_file_provenance(
        Path(queue_path).resolve() if queue_path is not None else None
    )
    output = Path(output_dir).resolve()
    artifact_names = {
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
    }
    if output.exists() and any((output / name).exists() for name in artifact_names):
        if not overwrite:
            raise FileExistsError(f"joint evidence bundle already exists in {output}")
        for name in artifact_names:
            path = output / name
            if path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)

    edge, cell, joint_null, joint_lag_null = _compute_joint_tables(families, alpha)
    edge["evidence_label"] = _classify_edge(edge, alpha)
    edge["experiment_ready"] = False
    edge["sham_gate_status"] = "not_run"
    edge["sesoi_status"] = "not_defined_pending_sham_calibration"
    edge["independent_confirmation_status"] = "not_run"
    edge["practical_status"] = PRACTICAL_STATUS

    edge_label_join = edge[
        EDGE_KEY
        + [
            "evidence_label",
        ]
    ]
    cell = cell.merge(
        edge_label_join, on=EDGE_KEY, how="left", validate="many_to_one"
    )
    cell["cell_localization_label"] = "not_jointly_localized"
    eligible = cell["support_eligible"].astype(bool)
    cell.loc[~eligible, "cell_localization_label"] = "sampling_limited"
    cell.loc[
        eligible & (cell["joint_primary_cell_max_t_p_value"] <= alpha),
        "cell_localization_label",
    ] = "joint_max_t_localized_cell"
    cell.loc[
        eligible
        & (cell["joint_primary_lag_contrast_max_t_p_value"] <= alpha),
        "cell_localization_label",
    ] = "joint_max_t_localized_lag_contrast"
    both = (
        eligible
        & (cell["joint_primary_cell_max_t_p_value"] <= alpha)
        & (cell["joint_primary_lag_contrast_max_t_p_value"] <= alpha)
    )
    cell.loc[both, "cell_localization_label"] = (
        "joint_max_t_localized_effect_and_lag_contrast"
    )
    cell["experiment_ready"] = False
    cell["practical_status"] = PRACTICAL_STATUS
    cell = _join_legacy_queue(cell, queue_path)

    family_rows: list[dict[str, object]] = []
    for family in families:
        e = edge.loc[edge.family_id == family.family_id]
        c = cell.loc[cell.family_id == family.family_id]
        family_rows.append(
            {
                "family_id": family.family_id,
                "channel": family.channel,
                "context": family.context,
                "n_worms": int(family.summary["n_worms"]),
                "sign_patterns": len(family.signs),
                "complete_edges": len(e),
                "support_eligible_edges": int(e.support_eligible.sum()),
                "complete_cells": len(c),
                "support_eligible_cells": int(c.support_eligible.sum()),
                "family_edge_bh_discoveries": int(
                    (e.edge_bh_q_value <= alpha).sum()
                ),
                "global_four_family_edge_bh_discoveries": int(
                    (e.global_four_family_edge_bh_q_value <= alpha).sum()
                ),
                "joint_primary_edge_max_t_discoveries": int(
                    (e.joint_primary_edge_max_t_p_value <= alpha).sum()
                ),
                "joint_primary_cell_max_t_discoveries": int(
                    (c.joint_primary_cell_max_t_p_value <= alpha).sum()
                ),
                "joint_primary_flat_lag_edge_discoveries": int(
                    (e.joint_primary_flat_lag_max_t_p_value <= alpha).sum()
                ),
                "joint_primary_lag_contrast_cell_discoveries": int(
                    (c.joint_primary_lag_contrast_max_t_p_value <= alpha).sum()
                ),
                "strong_edge_lag_unresolved": int(
                    (e.evidence_label == "strong_edge_lag_unresolved").sum()
                ),
                "experiment_ready": 0,
                "practical_status": PRACTICAL_STATUS,
            }
        )
    family_summary = pd.DataFrame(family_rows)
    shortlist = _select_sham_shortlist(cell, count=6)
    sensitivity_families: list[FamilyInput] | None = None
    sensitivity_edge: pd.DataFrame | None = None
    sensitivity_cell: pd.DataFrame | None = None
    sensitivity_family_summary: pd.DataFrame | None = None
    lag_sensitivity_shortlist: pd.DataFrame | None = None
    sampling_null_combined: pd.DataFrame | None = None
    sensitivity_summary: dict[str, object] | None = None
    if sensitivity_family_root is not None:
        sensitivity_families = load_and_validate_families(sensitivity_family_root)
        if sensitivity_families[0].input_sha256 != families[0].input_sha256:
            raise ValueError("strong and sensitivity canonical input hashes differ")
        if not np.array_equal(sensitivity_families[0].signs, families[0].signs):
            raise ValueError("strong and sensitivity sign-pattern ordering differs")
        (
            sensitivity_edge,
            sensitivity_cell,
            sensitivity_family_summary,
            lag_sensitivity_shortlist,
            sensitivity_summary,
        ) = _build_sensitivity_supplement(
            sensitivity_families,
            edge,
            alpha=alpha,
        )
        sampling_null_combined = _build_sampling_null_combined_queue(
            shortlist, lag_sensitivity_shortlist
        )

    # Rank only after all labels and joint p-values are frozen.
    edge = edge.sort_values(
        [
            "joint_primary_edge_max_t_p_value",
            "global_four_family_edge_bh_q_value",
            "edge_max_abs_t",
            "family_id",
            "source_index",
            "target_index",
        ],
        ascending=[True, True, False, True, True, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    edge.insert(0, "joint_edge_rank", np.arange(1, len(edge) + 1))

    _atomic_csv(family_summary, output / "family_summary.csv")
    _atomic_csv(edge, output / "edge_evidence.csv")
    _atomic_parquet(cell, output / "cell_evidence.parquet")
    _atomic_csv(shortlist, output / "sampling_null_shortlist.csv")
    if sensitivity_edge is not None:
        assert sensitivity_cell is not None
        assert sensitivity_family_summary is not None
        assert lag_sensitivity_shortlist is not None
        _atomic_csv(
            sensitivity_family_summary, output / "sensitivity_family_summary.csv"
        )
        _atomic_csv(sensitivity_edge, output / "sensitivity_edge_evidence.csv")
        _atomic_parquet(
            sensitivity_cell, output / "sensitivity_cell_evidence.parquet"
        )
        _atomic_csv(
            lag_sensitivity_shortlist,
            output / "sampling_null_lag_sensitivity_shortlist.csv",
        )
        assert sampling_null_combined is not None
        _atomic_csv(
            sampling_null_combined,
            output / "sampling_null_combined_8.csv",
        )

    top_edge_columns = [
        "joint_edge_rank",
        "family_id",
        "channel",
        "context",
        "source_neuron",
        "target_neuron",
        "edge_max_abs_t",
        "joint_primary_edge_max_t_p_value",
        "joint_primary_flat_lag_max_t_p_value",
        "edge_bh_q_value",
        "global_four_family_edge_bh_q_value",
        "evidence_label",
        "experiment_ready",
        "practical_status",
    ]
    legacy_matches = cell.loc[cell.legacy_queue_match].sort_values(
        "legacy_queue_rank", kind="stable"
    )
    legacy_columns = [
        "family_id",
        "source_neuron",
        "target_neuron",
        "source_lag_frames",
        "horizon_frames",
        "legacy_queue_rank",
        "legacy_postscreen_q_value",
        "legacy_q_label",
        "joint_primary_cell_max_t_p_value",
        "joint_primary_edge_max_t_p_value",
        "evidence_label",
    ]
    # The browser needs every joint-primary discovery for exact edge lookup.
    # Retaining only a decorative top-N list would mislabel a real discovery
    # outside that prefix as "effect estimate only".  Keep every strict edge,
    # plus the top 30 rows so the compact payload still exposes a bounded
    # exploratory context when fewer than 30 strict edges exist.
    explorer_edges = pd.concat(
        [
            edge.loc[edge.joint_primary_edge_max_t_p_value <= alpha],
            edge.head(30),
        ],
        ignore_index=True,
    ).drop_duplicates(
        ["family_id", "source_index", "target_index"], keep="first"
    )
    explorer_edges = explorer_edges.sort_values(
        "joint_edge_rank", kind="stable"
    ).reset_index(drop=True)
    explorer = {
        "schema_version": "complete-family-explorer-evidence-v1",
        "status": "complete",
        "headline": {
            "joint_primary_families": len(families),
            "complete_edge_tests": len(edge),
            "support_eligible_edge_tests": int(edge.support_eligible.sum()),
            "joint_primary_edge_discoveries": int(
                (edge.joint_primary_edge_max_t_p_value <= alpha).sum()
            ),
            "joint_primary_flat_lag_edge_discoveries": int(
                (edge.joint_primary_flat_lag_max_t_p_value <= alpha).sum()
            ),
            "experiment_ready": 0,
            "practical_status": PRACTICAL_STATUS,
        },
        "evidence_labels": {
            "strong_edge_lag_unresolved": (
                "joint four-family single-step max-T edge p<=0.05, but joint "
                "flat-lag max-T p>0.05"
            ),
            "strong_edge_with_lag_structure": (
                "both joint edge and joint flat-lag max-T p<=0.05; still a model "
                "coordinate, not a physical delay"
            ),
            "exploratory": (
                "an internal/family/global-BH signal that does not pass joint max-T"
            ),
            "sampling_limited": "source fails the frozen all-lag support gate",
            "no_complete_family_evidence": "no declared internal threshold passes",
            "experiment_ready": (
                "intentionally impossible until sham, SESOI, and independent "
                "confirmation gates are implemented"
            ),
        },
        "family_summary": _json_records(family_summary),
        "top_edges": _json_records(explorer_edges[top_edge_columns]),
        "sampling_null_shortlist": _json_records(shortlist),
        "sensitivity_support05": (
            {
                "available": True,
                **(sensitivity_summary or {}),
                "family_summary": _json_records(sensitivity_family_summary),
                "warning": (
                    "Every sensitivity-only row remains sampling_limited and "
                    "cannot strengthen the primary or experiment-ready claim."
                ),
            }
            if sensitivity_family_summary is not None
            else {"available": False}
        ),
        "legacy_queue": {
            "matched_cells": int(len(legacy_matches)),
            "q_value_label": LEGACY_Q_LABEL,
            "warning": (
                "legacy q-values were computed after top-effect retention and are "
                "not complete-family FDR values"
            ),
            "matches": _json_records(legacy_matches[legacy_columns].head(50)),
        },
    }
    _json_dump(output / "explorer_evidence.json", explorer)

    protocol = {
        "schema_version": "complete-family-joint-evidence-protocol-v1",
        "joint_primary": {
            "families": list(EXPECTED_FAMILIES),
            "independent_unit": "worm",
            "sign_patterns": len(families[0].signs),
            "sign_pattern_order": (
                "identical exact two-sided sign orbits in every family; validated "
                "byte-for-byte before elementwise null-max combination"
            ),
            "cell_and_edge_adjustment": (
                "elementwise maximum of the four saved family null maxima; exact "
                "upper-tail single-step max-T p-value"
            ),
            "lag_adjustment": (
                "elementwise maximum of four saved centered-lag null maxima"
            ),
            "alpha": alpha,
        },
        "exploratory_bh": {
            "family_size": len(edge),
            "rule": (
                "BH across all four complete 2,862-edge families; support-ineligible "
                "raw p-values enter as p=1"
            ),
            "interpretation": "exploratory, not the experiment-ready gate",
        },
        "support": {
            "rule": "inherited frozen strong all-lag support/genealogy gate",
            "ineligible_label": "sampling_limited",
        },
        "legacy_queue": {
            "join": QUEUE_KEY,
            "join_type": "exact keys only",
            "q_label": LEGACY_Q_LABEL,
            "input": legacy_queue_provenance,
        },
        "shortlist": {
            "count": 6,
            "family": "baseline_endpoint_mean",
            "selection_rule": shortlist.selection_rule.iloc[0],
            "purpose": "sham/null and provisional effect-threshold calibration",
            "planned_sham_controls": shortlist.planned_sham_controls.iloc[0],
            "combined_queue": (
                {
                    "artifact": "sampling_null_combined_8.csv",
                    "rows": len(sampling_null_combined),
                    "composition": "six strong-primary plus two lag-sensitivity rows",
                    "ordering": (
                        "primary shortlist_rank followed by sensitivity_shortlist_rank"
                    ),
                }
                if sampling_null_combined is not None
                else None
            ),
        },
        "support05_sensitivity": {
            "available": sensitivity_families is not None,
            "input_root": (
                str(Path(sensitivity_family_root).resolve())
                if sensitivity_family_root is not None
                else None
            ),
            "joint_null": (
                "the same four-family exact max-T calculation, performed only "
                "within the 0.5-support supplement"
            ),
            "label_rule": (
                "every row admitted only by the 0.5 gate is sampling_limited; "
                "no sensitivity row can become strong or experiment-ready"
            ),
            "named_flat_lag_followup": (
                "baseline endpoint_mean SMD→RID and SMD→RIB at lag16/h1"
                if sensitivity_families is not None
                else None
            ),
        },
        "experiment_ready_gate": {
            "available": False,
            "missing": [
                "sham-calibrated null excess",
                "prespecified experimental smallest effect size of interest",
                "independent biological confirmation",
            ],
            "practical_status": PRACTICAL_STATUS,
        },
        "claim_boundary": (
            "frozen-model evidence across available worms; not causal, not a "
            "physical delay, and not independent experimental validation"
        ),
    }
    _json_dump(output / "protocol.json", protocol)

    summary = {
        "schema_version": "complete-family-joint-evidence-summary-v1",
        "status": "complete",
        "families": len(families),
        "exact_sign_patterns": len(families[0].signs),
        "complete_edges": len(edge),
        "support_eligible_edges": int(edge.support_eligible.sum()),
        "complete_cells": len(cell),
        "support_eligible_cells": int(cell.support_eligible.sum()),
        "joint_primary_edge_discoveries": int(
            (edge.joint_primary_edge_max_t_p_value <= alpha).sum()
        ),
        "joint_primary_cell_discoveries": int(
            (cell.joint_primary_cell_max_t_p_value <= alpha).sum()
        ),
        "joint_primary_flat_lag_edge_discoveries": int(
            (edge.joint_primary_flat_lag_max_t_p_value <= alpha).sum()
        ),
        "joint_primary_lag_contrast_cell_discoveries": int(
            (cell.joint_primary_lag_contrast_max_t_p_value <= alpha).sum()
        ),
        "global_four_family_bh_edge_discoveries": int(
            (edge.global_four_family_edge_bh_q_value <= alpha).sum()
        ),
        "strong_edge_lag_unresolved": int(
            (edge.evidence_label == "strong_edge_lag_unresolved").sum()
        ),
        "strong_edge_with_lag_structure": int(
            (edge.evidence_label == "strong_edge_with_lag_structure").sum()
        ),
        "sampling_limited_edges": int(
            (edge.evidence_label == "sampling_limited").sum()
        ),
        "legacy_queue_exact_matches": int(cell.legacy_queue_match.sum()),
        "sampling_null_shortlist": len(shortlist),
        "sampling_null_combined_rows": (
            len(sampling_null_combined)
            if sampling_null_combined is not None
            else 0
        ),
        "support05_sensitivity_available": sensitivity_summary is not None,
        "support05_sensitivity": sensitivity_summary,
        "experiment_ready": 0,
        "practical_status": PRACTICAL_STATUS,
    }
    _json_dump(output / "summary.json", summary)

    validation = {
        "schema_version": "complete-family-joint-evidence-validation-v1",
        "status": "passed",
        "input_checksums_verified": True,
        "family_checksum_validations": [
            family.checksum_validation for family in families
        ],
        "sensitivity_family_checksum_validations": (
            [family.checksum_validation for family in sensitivity_families]
            if sensitivity_families is not None
            else []
        ),
        "identical_input_sha256": True,
        "input_sha256": families[0].input_sha256,
        "identical_sign_pattern_order": True,
        "sign_patterns_sha256": hashlib.sha256(
            families[0].signs.tobytes()
        ).hexdigest(),
        "all_families_exact": True,
        "complete_family_rows": len(edge) == 4 * 54 * 53,
        "edge_keys_unique_within_family": not edge.duplicated(EDGE_KEY).any(),
        "cell_keys_unique_within_family": not cell.duplicated(CELL_KEY).any(),
        "diagonal_edges_absent": bool(
            (edge.source_index != edge.target_index).all()
        ),
        "experiment_ready_count_zero": not edge.experiment_ready.any(),
        "shortlist_count_six": len(shortlist) == 6,
        "sampling_null_combined_count_eight": bool(
            sampling_null_combined is not None
            and len(sampling_null_combined) == 8
        ),
        "sampling_null_combined_keys_unique": bool(
            sampling_null_combined is not None
            and not sampling_null_combined.duplicated(QUEUE_KEY).any()
        ),
        "sampling_null_combined_composition_valid": bool(
            sampling_null_combined is not None
            and list(sampling_null_combined.selection_origin)[:6]
            == ["strong_primary"] * 6
            and list(sampling_null_combined.selection_origin)[6:]
            == ["lag_sensitivity"] * 2
        ),
        "legacy_queue_provenance": legacy_queue_provenance,
        "legacy_queue_sha256_pinned": bool(
            legacy_queue_provenance is None
            or legacy_queue_provenance.get("sha256")
        ),
        "strong_and_sensitivity_input_sha256_identical": (
            sensitivity_families is None
            or sensitivity_families[0].input_sha256 == families[0].input_sha256
        ),
        "strong_and_sensitivity_sign_order_identical": (
            sensitivity_families is None
            or np.array_equal(sensitivity_families[0].signs, families[0].signs)
        ),
        "sensitivity_only_labels_sampling_limited": (
            bool(
                sensitivity_edge is None
                or (
                    sensitivity_edge.loc[
                        sensitivity_edge.sensitivity_only, "evidence_label"
                    ]
                    == "sampling_limited"
                ).all()
            )
        ),
        "lag_sensitivity_shortlist_count_two": (
            lag_sensitivity_shortlist is None or len(lag_sensitivity_shortlist) == 2
        ),
    }
    if not all(
        value
        for key, value in validation.items()
        if key
        in {
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
            "legacy_queue_sha256_pinned",
        }
    ):
        raise RuntimeError("joint evidence validation failed")
    if sensitivity_families is not None and not all(
        validation[key]
        for key in (
            "sampling_null_combined_count_eight",
            "sampling_null_combined_keys_unique",
            "sampling_null_combined_composition_valid",
        )
    ):
        raise RuntimeError("combined sampling-null queue validation failed")
    _json_dump(output / "validation.json", validation)

    manifest = {
        "schema_version": "complete-family-joint-evidence-manifest-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(Path(family_root).resolve()),
        "sensitivity_input_root": (
            str(Path(sensitivity_family_root).resolve())
            if sensitivity_family_root is not None
            else None
        ),
        "output_dir": str(output),
        "inputs": {
            "legacy_queue": legacy_queue_provenance,
        },
        "artifacts": {
            "family_summary": "family_summary.csv",
            "edge_evidence": "edge_evidence.csv",
            "cell_evidence": "cell_evidence.parquet",
            "sampling_null_shortlist": "sampling_null_shortlist.csv",
            **(
                {
                    "sensitivity_family_summary": "sensitivity_family_summary.csv",
                    "sensitivity_edge_evidence": "sensitivity_edge_evidence.csv",
                    "sensitivity_cell_evidence": "sensitivity_cell_evidence.parquet",
                    "sampling_null_lag_sensitivity_shortlist": (
                        "sampling_null_lag_sensitivity_shortlist.csv"
                    ),
                    "sampling_null_combined": "sampling_null_combined_8.csv",
                }
                if sensitivity_families is not None
                else {}
            ),
            "explorer_evidence": "explorer_evidence.json",
            "protocol": "protocol.json",
            "summary": "summary.json",
            "validation": "validation.json",
            "checksums": "checksums.sha256",
        },
    }
    _json_dump(output / "manifest.json", manifest)
    paths = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in paths)
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--sensitivity-family-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queue-path", type=Path)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_joint_evidence(
        args.family_root,
        args.output_dir,
        sensitivity_family_root=args.sensitivity_family_root,
        queue_path=args.queue_path,
        alpha=args.alpha,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
