from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.prediction_atlas_external_analysis import (
    _dashboard_external_summary,
    _lagmax_rows,
    _load_sbtg,
    _neuromodulator_rows,
    _paths_overlap,
    _reference_rows,
    _verify_sbtg_archive_provenance,
    _verify_internal_atlas,
    sha256,
)


def _slice(matrix: np.ndarray, lag: int) -> dict[str, object]:
    return {
        "method": "progressive_bridge_smc",
        "channel": "endpoint_mean",
        "context": "state_average",
        "lag_frames": lag,
        "horizon_frames": 1,
        "matrix": matrix,
        "support": np.ones(matrix.shape[0], dtype=np.float32),
    }


def test_external_rows_preserve_target_row_source_column_and_continuous_metrics() -> None:
    # Positive reference edge: source column 1 -> target row 0.
    matrix = np.zeros((3, 3), dtype=np.float32)
    matrix[0, 1] = 9.0
    labels = np.zeros((3, 3), dtype=np.int8)
    labels[0, 1] = 1
    off = ~np.eye(3, dtype=bool)
    references = {
        name: {"labels": labels, "mask": off, "weight": labels.astype(float)}
        for name in ("randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54")
    }
    rows = _reference_rows([_slice(matrix, 1)], references, fps=4.0)
    assert len(rows) == 8
    assert all(row["auroc"] == 1.0 for row in rows)
    assert all(row["source_to_readout_seconds"] == 0.5 for row in rows)


def test_neuromodulator_rows_and_lagmax_use_source_preserving_null() -> None:
    strong = np.zeros((4, 4), dtype=np.float32)
    weak = np.zeros((4, 4), dtype=np.float32)
    labels = np.zeros((4, 4), dtype=np.int8)
    labels[0, 1] = 1
    strong[0, 1] = 10.0
    weak[0, 1] = 1.0
    weak[2, 1] = 2.0
    slices = [_slice(weak, 1), _slice(strong, 8)]
    networks = {"monoamine_dopamine": labels}
    descriptive = _neuromodulator_rows(slices, networks, fps=4.0)
    assert len(descriptive) == 6
    inference = _lagmax_rows(
        {"family": slices},
        networks,
        permutations=99,
        seed=7,
        lag_grid="test",
    )
    assert len(inference) == 1
    assert inference[0]["best_lag_frames"] == 8
    assert inference[0]["inference_limit"].endswith("not_physical_delay_or_causality")


def test_lagmax_uses_common_support_and_marks_non_evaluable() -> None:
    labels = np.zeros((4, 4), dtype=np.int8)
    labels[0, 1] = 1
    first = _slice(np.eye(4, dtype=np.float32), 1)
    second = _slice(np.eye(4, dtype=np.float32), 8)
    first["support"] = np.asarray([1, 1, 1, 1], dtype=np.float32)
    second["support"] = np.asarray([1, 0, 1, 1], dtype=np.float32)
    rows = _lagmax_rows(
        {"flow": [first, second]},
        {"monoamine_dopamine": labels},
        permutations=99,
        seed=3,
        lag_grid="native",
    )
    assert len(rows) == 1
    assert rows[0]["evaluable"] is False
    assert rows[0]["n_reference_eligible_sources"] == 1
    assert rows[0]["n_common_supported_eligible_sources"] == 0
    assert rows[0]["support_scope"] == "intersection_across_all_candidate_lags"


def test_sbtg_loader_accepts_real_archive_schema_without_flow_validity(tmp_path) -> None:
    neurons = ("A", "B", "C")
    archive = tmp_path / "aligned.npz"
    matrices = np.zeros((2, 3, 3), dtype=np.float32)
    matrices[0, 0, 1] = 2.0
    np.savez_compressed(
        archive,
        neurons=np.asarray(neurons),
        sbtg_current__lags=np.asarray([1, 8]),
        sbtg_current__matrices=matrices,
        sbtg_published__lags=np.asarray([1, 8]),
        sbtg_published__matrices=matrices,
    )
    (tmp_path / "protocol.json").write_text(
        json.dumps(
            {"orientation": "all saved comparison matrices are [target, source]"}
        )
    )
    (tmp_path / "validation.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "neuron_order_exact_match": True,
                "paired_orientation_transposed_once": True,
            }
        )
    )
    (tmp_path / "checksums.sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n"
    )
    provenance = _verify_sbtg_archive_provenance(archive)
    assert provenance[archive.name] == sha256(archive)
    slices = _load_sbtg(archive, neurons)
    assert len(slices) == 4
    assert all(item["horizon_frames"] is None for item in slices)
    assert all(item["support_available"] is False for item in slices)
    assert "80-neuron" in next(
        item["training_lineage"]
        for item in slices
        if item["method"] == "sbtg_published"
    )

    labels = np.zeros((3, 3), dtype=np.int8)
    labels[0, 1] = 1
    off = ~np.eye(3, dtype=bool)
    references = {
        name: {"labels": labels, "mask": off, "weight": labels.astype(float)}
        for name in ("randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54")
    }
    rows = _reference_rows(slices[:1], references, fps=4.0)
    assert len(rows) == 4  # no duplicate, inapplicable flow-support scope
    assert all(row["source_to_cut_seconds"] is None for row in rows)
    assert all(row["forecast_horizon_seconds"] is None for row in rows)
    assert all(row["source_to_readout_seconds"] is None for row in rows)
    assert all(row["lag_index_seconds"] == 0.25 for row in rows)


def test_internal_checksum_firewall_and_output_overlap_are_fail_closed(tmp_path) -> None:
    atlas = tmp_path / "atlas"
    atlas.mkdir()
    (atlas / "manifest.json").write_text(
        json.dumps({"status": "complete", "created_utc": "2026-08-29T00:00:00Z"})
    )
    (atlas / "protocol.json").write_text(
        json.dumps(
            {
                "ranking": "no connectome or external reference data used",
                "orientation": "target_row_source_column",
            }
        )
    )
    (atlas / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "problems": [],
                "archive_checks_completed": {"all_required": True},
            }
        )
    )
    np.savez_compressed(atlas / "atlas_matrices.npz", placeholder=np.asarray(1))
    (atlas / "hypothesis_queue.csv").write_text("queue_rank\n1\n")
    names = (
        "manifest.json",
        "protocol.json",
        "validation.json",
        "atlas_matrices.npz",
        "hypothesis_queue.csv",
    )
    (atlas / "checksums.sha256").write_text(
        "".join(f"{sha256(atlas / name)}  {name}\n" for name in names)
    )
    firewall = _verify_internal_atlas(atlas)
    assert firewall["ranking_inputs_used"] == []
    assert firewall["internal_atlas_matrices_sha256"] == sha256(
        atlas / "atlas_matrices.npz"
    )
    assert firewall["internal_checksums_sha256"] == sha256(
        atlas / "checksums.sha256"
    )
    assert _paths_overlap(atlas, atlas / "external")
    assert not _paths_overlap(atlas, tmp_path / "separate")
    (atlas / "hypothesis_queue.csv").write_text("queue_rank\n2\n")
    with pytest.raises(RuntimeError, match="checksum failed"):
        _verify_internal_atlas(atlas)


def test_dashboard_external_summary_is_bounded_and_has_single_union_schema() -> None:
    reference = pd.DataFrame(
        [
            {
                "method": "progressive_bridge_smc",
                "channel": "endpoint_mean",
                "context": "state_average",
                "reference": "cook_struct_54",
                "lag_frames": 1,
                "horizon_frames": 1,
                "scope": "support_qualified_sources",
                "auroc": 0.6,
                "auprc": 0.2,
                "absolute_spearman": 0.1,
            }
        ]
    )
    neuromod = pd.DataFrame(
        [
            {
                "method": "direct_importance",
                "channel": "endpoint_log_sd",
                "context": "onset_minus_baseline",
                "network": "monoamine_dopamine",
                "lag_frames": 1,
                "horizon_frames": 1,
                "scope": "eligible_support_qualified",
                "auroc": 0.7,
                "auprc": 0.1,
            }
        ]
    )
    lagmax = pd.DataFrame(
        [
            {
                "lag_grid": "native_method_grid",
                "method": "direct_importance",
                "channel": "endpoint_log_sd",
                "context": "state_average",
                "network": "monoamine_dopamine",
                "best_lag_frames": 4,
                "horizon_frames": 1,
                "best_auroc": 0.72,
                "max_lag_permutation_p": 0.04,
                "max_lag_bh_q": 0.2,
            }
        ]
    )
    summary = _dashboard_external_summary(reference, neuromod, lagmax)
    assert len(summary) == 3
    assert len(summary) <= 250
    assert {"panel", "reference_or_network", "timing_label"}.issubset(summary.columns)
    assert summary.iloc[-1]["bh_q"] == 0.2
