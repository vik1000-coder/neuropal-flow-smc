from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from compatibility_neural_benchmark.prediction_atlas_analysis import CONTEXTS
from compatibility_neural_benchmark.prediction_atlas_dashboard import (
    DashboardInputError,
    build_dashboard_artifact,
    sha256,
)


CREATED = "2026-08-29T20:00:00+00:00"
LAGS = (1, 4, 8, 16)
HORIZONS = (1, 2, 4, 8, 16, 32)


def _context_metadata(context: str) -> dict[str, object]:
    chemical = next(
        (name for name in ("butanone", "pentanedione", "nacl") if context.startswith(name + "_")),
        None,
    )
    return {
        "context": context,
        "chemical": chemical,
        "is_phase_contrast": context.endswith("onset_minus_baseline"),
        "event_stratified": chemical is not None,
        "conditioning_status": (
            "exploratory_event_stratified_under_binary_any_stimulus_generator"
            if chemical is not None
            else "binary_any_stimulus_conditioned"
        ),
    }


def _queue_rows() -> list[dict[str, object]]:
    contexts = (
        "onset_minus_baseline",
        "butanone_onset",
        "baseline",
        "active",
        "pentanedione_onset_minus_baseline",
        "recovery",
    )
    rows = []
    for index in range(30):
        lag = LAGS[index % len(LAGS)]
        horizon = HORIZONS[index % len(HORIZONS)]
        context = contexts[index % len(contexts)]
        chemical = context.startswith(("butanone", "pentanedione", "nacl"))
        effect = 0.25 + 0.01 * index
        rows.append(
            {
                "queue_rank": index + 1,
                "method": "progressive_bridge_smc" if index < 24 else "direct_importance",
                "model_id": "binary-flow-test",
                "channel": "endpoint_mean" if index < 26 else "endpoint_log_sd",
                "context": context,
                "chemical": "butanone" if "butanone" in context else "",
                "conditioning_status": (
                    "exploratory_event_stratified_under_binary_any_stimulus_generator"
                    if chemical
                    else "binary_any_stimulus_conditioned"
                ),
                "source_neuron": f"S{index % 54:02d}",
                "target_neuron": f"T{(index + 7) % 54:02d}",
                "source_index": index % 54,
                "target_index": (index + 7) % 54,
                "source_lag_frames": lag,
                "source_to_cut_seconds": lag / 4.0,
                "horizon_frames": horizon,
                "forecast_horizon_seconds": horizon / 4.0,
                "source_to_readout_seconds": (lag + horizon) / 4.0,
                "mean_raw": effect * 2,
                "mean_normalized": effect,
                "median_normalized": effect - 0.01,
                "ci_2_5": effect - 0.08,
                "ci_97_5": effect + 0.08,
                "screen_t_p_value": 0.02,
                "sign_flip_p_value": 0.03,
                "sign_flip_q_value": 0.08,
                "bh_family": "fixture",
                "test_sidedness": "two_sided_candidate_sign_flip",
                "sign_consistency": 0.85,
                "valid_fraction": 0.82,
                "seed_spearman": 0.65,
                "seed_sign_agreement": 1.0,
                "n_worms": 17,
                "effect_direction": "positive",
                "support_tier": "model_only",
                "evidence_score": 10.0 - 0.1 * index,
                "interpretation_limit": "model_based_lag_association_not_causal_or_physical_delay",
                "cross_sampler_spearman": 0.6,
            }
        )
    return rows


def _refresh_checksums(atlas: Path) -> None:
    names = (
        "dashboard_snapshot.json",
        "hypothesis_queue.csv",
        "manifest.json",
        "validation.json",
        "protocol.json",
        "models.json",
        "atlas_matrices.npz",
    )
    (atlas / "checksums.sha256").write_text(
        "".join(f"{sha256(atlas / name)}  {name}\n" for name in names)
    )


def _write_bundle_checksums(directory: Path, names: tuple[str, ...]) -> None:
    (directory / "checksums.sha256").write_text(
        "".join(f"{sha256(directory / name)}  {name}\n" for name in names)
    )


def _write_external_fixture(
    root: Path, atlas: Path, rows: list[dict[str, object]]
) -> Path:
    """Write the bounded CSV inside a checksum-linked post-freeze bundle."""

    external = root / "external_analysis"
    external.mkdir(parents=True)
    summary = external / "dashboard_external_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    firewall = {
        "internal_manifest_sha256": sha256(atlas / "manifest.json"),
        "internal_protocol_sha256": sha256(atlas / "protocol.json"),
        "internal_validation_sha256": sha256(atlas / "validation.json"),
        "internal_atlas_matrices_sha256": sha256(atlas / "atlas_matrices.npz"),
        "internal_checksums_sha256": sha256(atlas / "checksums.sha256"),
        "frozen_hypothesis_queue_sha256": sha256(
            atlas / "hypothesis_queue.csv"
        ),
    }
    (external / "input_checksums.sha256").write_text(
        "".join(
            (
                f"{firewall['internal_atlas_matrices_sha256']}  "
                "internal_atlas/atlas_matrices.npz\n",
                f"{firewall['internal_checksums_sha256']}  "
                "internal_atlas/checksums.sha256\n",
                f"{firewall['frozen_hypothesis_queue_sha256']}  "
                "internal_atlas/hypothesis_queue.csv\n",
            )
        )
    )
    (external / "manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "analysis_role": "post_freeze_external_convergence_only",
                "ranking_effect": "none; internal ranking is immutable",
                "internal_firewall_reverified_after_analysis": True,
                "internal_prediction_ranking_inputs": [],
                "internal_firewall": firewall,
            }
        )
    )
    _write_bundle_checksums(
        external,
        (
            "manifest.json",
            "dashboard_external_summary.csv",
            "input_checksums.sha256",
        ),
    )
    return summary


def _write_targeted_fixture(root: Path, atlas: Path) -> Path:
    """Write a minimal checksum-linked N128 analysis/selection/run chain."""

    factual = "free conditional-flow rollout from the observed history at the cut"
    boundary = (
        "model-relative repaired-law contrasts; not causal interventions, anatomical "
        "connections, receptor actions, physical delays, or experimental confirmation"
    )
    selection = root / "targeted_selection"
    selection.mkdir(parents=True)
    canonical_text = pd.read_csv(
        atlas / "hypothesis_queue.csv", dtype=str, keep_default_na=False
    )
    selected = canonical_text.iloc[:2].copy()
    selected["run_confirmation"] = True
    selected.to_csv(selection / "hypothesis_queue.csv", index=False)
    (selection / "atlas_matrices.npz").write_bytes(
        (atlas / "atlas_matrices.npz").read_bytes()
    )
    (selection / "selection_rationale.md").write_text(
        "# Frozen targeted selection\n\nInternal atlas rows only.\n"
    )
    selection_manifest = {
        "schema_version": "prediction_atlas_targeted_selection_v1",
        "status": "complete",
        "created_utc": CREATED,
        "source_atlas": str(atlas.resolve()),
        "source_manifest_sha256": sha256(atlas / "manifest.json"),
        "source_protocol_sha256": sha256(atlas / "protocol.json"),
        "source_validation_sha256": sha256(atlas / "validation.json"),
        "source_atlas_matrices_sha256": sha256(atlas / "atlas_matrices.npz"),
        "frozen_hypothesis_queue_sha256": sha256(
            atlas / "hypothesis_queue.csv"
        ),
        "selection_sha256": sha256(selection / "hypothesis_queue.csv"),
        "particles": 128,
        "maximum_rows": 6,
        "confirmation_method": "progressive_bridge_smc_targeted_three_arm",
        "screen_dir_compatible": True,
        "selected_rows": len(selected),
        "external_reference_inputs_used": [],
        "external_fields_detected": [],
    }
    (selection / "manifest.json").write_text(json.dumps(selection_manifest))
    _write_bundle_checksums(
        selection,
        (
            "manifest.json",
            "hypothesis_queue.csv",
            "atlas_matrices.npz",
            "selection_rationale.md",
        ),
    )

    run = root / "targeted_run"
    run.mkdir()
    run_manifest = {
        "method": "progressive_bridge_smc_targeted_three_arm",
        "particles": 128,
        "hypothesis_queue": str((selection / "hypothesis_queue.csv").resolve()),
        "hypothesis_queue_sha256": sha256(selection / "hypothesis_queue.csv"),
        "matrix_orientation": "target_row_source_column",
        "stimulus_generator_encoding": "binary_any_stimulus",
        "chemical_identity_conditioned": False,
    }
    (run / "manifest.json").write_text(json.dumps(run_manifest))

    targeted = root / "targeted_analysis"
    targeted.mkdir()
    cell_rows: list[dict[str, object]] = []
    screen_rows: list[dict[str, object]] = []
    for candidate_index, (_, row) in enumerate(selected.iterrows(), start=1):
        candidate_id = f"targeted_{candidate_index:04d}"
        screen_value = float(row["mean_normalized"])
        targeted_value = screen_value + 0.02 * candidate_index
        contrast_values = {
            "high_low": targeted_value,
            "high_factual": targeted_value / 2,
            "low_factual": -targeted_value / 2,
        }
        for contrast, value in contrast_values.items():
            cell_rows.append(
                {
                    "candidate_id": candidate_id,
                    "queue_rank": int(row["queue_rank"]),
                    "source_neuron": row["source_neuron"],
                    "target_neuron": row["target_neuron"],
                    "source_index": int(row["source_index"]),
                    "target_index": int(row["target_index"]),
                    "source_lag_frames": int(row["source_lag_frames"]),
                    "source_to_cut_seconds": float(row["source_to_cut_seconds"]),
                    "horizon_frames": int(row["horizon_frames"]),
                    "forecast_horizon_seconds": float(
                        row["forecast_horizon_seconds"]
                    ),
                    "source_to_readout_seconds": float(
                        row["source_to_readout_seconds"]
                    ),
                    "context": row["context"],
                    "contrast": contrast,
                    "n_worms": 17,
                    "n_particles": 128,
                    "valid_fraction": 0.9,
                    "matrix_orientation": "target_row_source_column",
                    "row_axis": "target_neuron",
                    "column_axis": "source_neuron",
                    "chemical_identity_conditioned": False,
                    "factual_arm_definition": factual,
                    "interpretation_limit": boundary,
                    "metric": "endpoint_mean",
                    "signed": True,
                    "mean_normalized": value,
                    "ci_2_5": value - 0.1,
                    "ci_97_5": value + 0.1,
                    "evidence_label": "model_relative_consistent",
                }
            )
        magnitude = 1 - abs(targeted_value - screen_value) / (
            abs(targeted_value) + abs(screen_value) + 1e-12
        )
        screen_rows.append(
            {
                "candidate_id": candidate_id,
                "source_neuron": row["source_neuron"],
                "target_neuron": row["target_neuron"],
                "source_lag_frames": int(row["source_lag_frames"]),
                "horizon_frames": int(row["horizon_frames"]),
                "context": row["context"],
                "screen_method": row["method"],
                "targeted_method": "progressive_bridge_smc_targeted_three_arm",
                "same_sampler_family": row["method"] == "progressive_bridge_smc",
                "comparison_type": "particle_escalation_within_progressive_bridge",
                "screen_channel": row["channel"],
                "targeted_metric": "endpoint_mean",
                "screen_mean_normalized": screen_value,
                "targeted_n128_mean_normalized": targeted_value,
                "targeted_minus_screen": targeted_value - screen_value,
                "direction_agreement": 1.0,
                "magnitude_agreement": magnitude,
                "screen_valid_fraction": float(row["valid_fraction"]),
                "targeted_valid_fraction": 0.9,
                "consistency_label": "direction_consistent",
                "interpretation_limit": "descriptive and selection-conditioned",
            }
        )
    pd.DataFrame(cell_rows).to_csv(targeted / "targeted_cells.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(
        targeted / "screen_consistency.csv", index=False
    )
    screen_summary = {"status": "computed", "rows": len(screen_rows)}
    (targeted / "protocol.json").write_text(
        json.dumps(
            {
                "analysis_schema_version": "targeted_confirmation_analysis_v1",
                "method": "progressive_bridge_smc_targeted_three_arm",
                "particle_count": 128,
                "matrix_orientation": "target_row_source_column",
                "contrasts": ["high_low", "high_factual", "low_factual"],
                "factual_arm_definition": factual,
                "claim_boundary": boundary,
                "external_reference_firewall": (
                    "no connectome, receptor atlas, Randi, Cook, Bentley, or SBTG "
                    "reference is loaded"
                ),
            }
        )
    )
    (targeted / "summary.json").write_text(
        json.dumps(
            {
                "status": "reviewed_model_relative",
                "analysis_schema_version": "targeted_confirmation_analysis_v1",
                "n_worms": 17,
                "n_candidates": len(selected),
                "n_cell_rows": len(cell_rows),
                "screen_vs_n128": screen_summary,
                "experimental_confirmation_label_assigned": False,
                "external_reference_data_used": False,
            }
        )
    )
    (targeted / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "analysis_schema_version": "targeted_confirmation_analysis_v1",
                "created_utc": CREATED,
                "n_worms": 17,
                "checks": {
                    "target_row_source_column_orientation": True,
                    "worm_level_inference": True,
                    "external_reference_firewall": True,
                },
                "screen_vs_n128": screen_summary,
            }
        )
    )
    input_paths = (
        run / "manifest.json",
        selection / "hypothesis_queue.csv",
        selection / "atlas_matrices.npz",
    )
    pd.DataFrame(
        [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in input_paths
        ]
    ).to_csv(targeted / "input_checksums.csv", index=False)
    (targeted / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "analysis_schema_version": "targeted_confirmation_analysis_v1",
                "created_utc": CREATED,
                "input_run_dir": str(run.resolve()),
                "input_run_manifest_sha256": sha256(run / "manifest.json"),
                "input_queue_sha256": sha256(selection / "hypothesis_queue.csv"),
                "screen_dir": str(selection.resolve()),
                "matrix_orientation": "target_row_source_column",
                "artifacts": {
                    "targeted_cells": "targeted_cells.csv",
                    "screen_consistency": "screen_consistency.csv",
                    "summary": "summary.json",
                    "validation": "validation.json",
                    "protocol": "protocol.json",
                    "input_checksums": "input_checksums.csv",
                },
            }
        )
    )
    _write_bundle_checksums(
        targeted,
        (
            "manifest.json",
            "validation.json",
            "protocol.json",
            "summary.json",
            "targeted_cells.csv",
            "screen_consistency.csv",
            "input_checksums.csv",
        ),
    )
    return targeted


def _write_fixture(root: Path) -> Path:
    atlas = root / "atlas"
    atlas.mkdir(parents=True)
    queue_rows = _queue_rows()
    pd.DataFrame(queue_rows).to_csv(atlas / "hypothesis_queue.csv", index=False)
    lag_profiles = []
    for candidate in queue_rows[:10]:
        profile = {1: 0.10, 4: 0.50, 8: 0.20, 16: 0.15}
        counterpart = {1: 0.05, 4: 0.40, 8: 0.10, 16: 0.08}
        bootstrap = {1: 0.10, 4: 0.70, 8: 0.15, 16: 0.05}
        for lag in LAGS:
            lag_profiles.append(
                {
                    "queue_rank": candidate["queue_rank"],
                    "source_neuron": candidate["source_neuron"],
                    "target_neuron": candidate["target_neuron"],
                    "channel": candidate["channel"],
                    "context": candidate["context"],
                    "horizon_frames": candidate["horizon_frames"],
                    "source_lag_frames": lag,
                    "source_to_cut_seconds": lag / 4.0,
                    "primary_method": "progressive_bridge_smc",
                    "counterpart_method": "direct_importance",
                    "primary_mean_normalized": profile[lag],
                    "counterpart_mean_normalized": counterpart[lag],
                    "is_primary_peak_abs_lag": lag == 4,
                    "worm_bootstrap_peak_lag_rate": bootstrap[lag],
                    "top_vs_second_lag_selectivity": 0.60,
                    "signed_lag_profile_spearman": 0.90,
                    "interpretation": (
                        "descriptive model lag localization; not a physical delay"
                    ),
                }
            )
    dashboard = {
        "title": "Neural lag-effect prediction atlas",
        "status": "reviewed_model_predictions",
        "primary_method": "progressive_bridge_smc",
        "cohort": {"mode": "oh16230_head", "worms": 17, "neurons": 54, "fps": 4.0},
        "methods": ["direct_importance", "progressive_bridge_smc"],
        "channels": [
            "endpoint_mean",
            "cumulative_mean",
            "peak_mean",
            "event_probability",
            "endpoint_sd",
            "endpoint_log_sd",
            "endpoint_wasserstein1",
        ],
        "contexts": [_context_metadata(context) for context in CONTEXTS],
        "source_lags": [
            {"frames": lag, "seconds_to_cut": lag / 4.0} for lag in LAGS
        ],
        "horizons": [
            {"frames": horizon, "seconds": horizon / 4.0} for horizon in HORIZONS
        ],
        "top_candidates": queue_rows[:10],
        "candidate_lag_profiles": lag_profiles,
        "support_summary": {
            "mean_valid_fraction": 0.81,
            "sources_ge_minimum": 1_900,
            "support_rows": 2_808,
        },
        "warnings": [
            "Chemical panels are event-stratified and not chemically conditioned.",
            "Model-relative and noncausal.",
        ],
    }
    (atlas / "dashboard_snapshot.json").write_text(json.dumps(dashboard))
    diagnostics = []
    for channel in ("endpoint_mean", "endpoint_log_sd"):
        for context_index, context in enumerate(CONTEXTS):
            for lag in LAGS:
                for horizon in HORIZONS:
                    diagnostics.append(
                        {
                            "channel": channel,
                            "context": context,
                            "source_lag_frames": lag,
                            "horizon_frames": horizon,
                            "direct_vs_progressive_spearman": (
                                0.2 + 0.01 * context_index + 0.002 * lag - 0.001 * horizon
                            ),
                        }
                    )
    validation = {
        "status": "passed",
        "problems": [],
        "created_utc": CREATED,
        "archive_count": 80,
        "archive_checks_completed": {
            "status_and_required_fields": True,
            "all_response_shapes": True,
            "all_response_and_diagnostic_values_finite": True,
            "complete_method_lag_fold_seed_grid": True,
            "heldout_worm_coverage_once_per_seed": True,
            "checkpoint_paths_and_hashes": True,
            "stimulus_schema_version_and_fingerprint": True,
            "chemical_permutation_and_code_name_mapping": True,
            "chemical_schedule_matches_manifest": True,
            "source_window_bounds_and_lag": True,
            "horizon_and_source_to_readout_timing": True,
            "neuron_order_constant": True,
            "orientation_self_test": True,
        },
        "n_worms": 17,
        "n_neurons": 54,
        "cross_sampler_diagnostics": diagnostics,
    }
    (atlas / "validation.json").write_text(json.dumps(validation))
    protocol = {
        "protocol": "reviewed neural prediction atlas v1",
        "n_worms": 17,
        "n_neurons": 54,
        "fps": 4.0,
        "source_lag_frames": list(LAGS),
        "horizon_frames": list(HORIZONS),
        "timing_definitions": {
            "source_to_cut_seconds": "source_lag_frames / fps",
            "forecast_horizon_seconds": "horizon_frames / fps",
            "source_to_readout_seconds": "(source_lag_frames + horizon_frames) / fps",
        },
        "orientation": "target_row_source_column",
        "chemical_context_warning": (
            "Chemical panels are event-stratified observations under a binary-any-stimulus "
            "generator. They are exploratory and are not chemically conditioned counterfactuals."
        ),
        "evidence_tiers": {
            "confirmed": "independent experiment",
            "supported_exploratory": "all gates",
            "model_only": "incomplete gates",
            "unsupported": "support failure",
        },
        "interpretation_limit": (
            "lag association from a fitted conditional generator; not a causal or physical-delay estimate"
        ),
        "ranking": "no connectome or external reference data used",
    }
    (atlas / "protocol.json").write_text(json.dumps(protocol))
    models = {
        "generator_model_id": "binary-flow-test",
        "generator_stimulus_encoding": "binary_any_stimulus",
        "chemical_conditioning": False,
        "methods": {
            "direct_importance": {"particles": [256], "model_seeds": [1701, 2903], "folds": list(range(5))},
            "progressive_bridge_smc": {"particles": [32], "model_seeds": [1701, 2903], "folds": list(range(5))},
        },
        "checkpoints": [],
    }
    (atlas / "models.json").write_text(json.dumps(models))
    (atlas / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol": "reviewed neural prediction atlas v1",
                "primary_method": "progressive_bridge_smc",
                "dense_matrix_orientation": "target_row_source_column",
            }
        )
    )
    (atlas / "atlas_matrices.npz").write_bytes(
        b"bounded synthetic canonical matrix fixture"
    )
    _refresh_checksums(atlas)
    return atlas


def _add_full_matrix_slice(atlas: Path) -> tuple[list[str], list[str]]:
    dashboard_path = atlas / "dashboard_snapshot.json"
    dashboard = json.loads(dashboard_path.read_text())
    source_neurons = [f"S{index:02d}" for index in range(54)]
    target_neurons = [f"T{index:02d}" for index in range(54)]
    dashboard["matrix_rows"] = [
        {
            "source_neuron": source,
            "target_neuron": target,
            "mean_normalized": (
                7.25 if (target, source) == ("T07", "S00") else -0.1
            ),
            "source_lag_frames": 4,
            "horizon_frames": 8,
            "method": "progressive_bridge_smc",
            "channel": "endpoint_mean",
            "context": "onset_minus_baseline",
        }
        for target in target_neurons
        for source in source_neurons
    ]
    dashboard_path.write_text(json.dumps(dashboard))
    _refresh_checksums(atlas)
    return source_neurons, target_neurons


def test_dashboard_artifact_is_bounded_source_backed_and_explicit(tmp_path):
    atlas = _write_fixture(tmp_path)
    output = tmp_path / "artifact.json"
    artifact = build_dashboard_artifact(
        atlas,
        output,
        provenance_root=tmp_path,
        max_candidates=20,
    )
    assert output.exists()
    assert artifact["surface"] == "dashboard"
    assert artifact["snapshot"]["status"] == "ready"
    assert len(artifact["snapshot"]["datasets"]["candidate_rows"]) == 20
    assert artifact["snapshot"]["datasets"]["candidate_rows"][0][
        "channel_label"
    ] == "Endpoint mean shift"
    assert len(artifact["snapshot"]["datasets"]["lag_horizon_profile"]) == 24
    assert len(artifact["snapshot"]["datasets"]["candidate_lag_profile_rows"]) == 40
    assert "prediction_cells" not in artifact["snapshot"]["datasets"]
    assert "primary_matrix_slice" not in artifact["snapshot"]["datasets"]
    # Dashboard filters are toolbar/global in the installed portable reader.
    # Candidate-only filters would be silently suppressed and their defaults
    # would not filter the ranking chart, so this bounded summary declares no
    # misleading controls and freezes the chart to one explicit primary slice.
    assert artifact["manifest"]["filters"] == []
    primary_rows = artifact["snapshot"]["datasets"]["primary_candidate_rows"]
    assert primary_rows
    assert {
        (row["method"], row["channel"], row["context"])
        for row in primary_rows
    } == {
        ("progressive_bridge_smc", "endpoint_mean", "onset_minus_baseline")
    }
    chart_types = {chart["id"]: chart["type"] for chart in artifact["manifest"]["charts"]}
    assert chart_types["lag_horizon_chart"] == "heatmap"
    assert chart_types["candidate_ranking_chart"] == "horizontalBar"
    assert chart_types["candidate_lag_profile_chart"] == "bar"
    ranking_chart = next(
        chart
        for chart in artifact["manifest"]["charts"]
        if chart["id"] == "candidate_ranking_chart"
    )
    assert ranking_chart["dataset"] == "primary_candidate_rows"
    assert "primary review slice" in ranking_chart["title"]
    portable_scope = next(
        block
        for block in artifact["manifest"]["blocks"]
        if block["id"] == "portable_scope"
    )
    assert "not the full interactive atlas explorer" in portable_scope["body"]
    assert "target neurons are rows" in artifact["manifest"]["blocks"][1]["body"]
    assert "not chemically conditioned" in artifact["manifest"]["blocks"][1]["body"]
    assert "not causal" in artifact["manifest"]["blocks"][1]["body"]
    lag_table = next(
        table
        for table in artifact["manifest"]["tables"]
        if table["id"] == "candidate_lag_profile_table"
    )
    assert "not a causal or physical-delay" in lag_table["subtitle"]
    assert all(not Path(source["path"]).is_absolute() for source in artifact["sources"])
    assert all(".." not in Path(source["path"]).parts for source in artifact["sources"])
    assert all("python -m compatibility_neural_benchmark.prediction_atlas_dashboard" in source["query"]["query"] for source in artifact["sources"])
    assert all(source["query"]["sql"].startswith("SELECT ") for source in artifact["sources"])
    provenance = artifact["snapshot"]["datasets"]["provenance_rows"]
    assert provenance[-1]["status"] == "optional, not supplied"
    assert json.loads(output.read_text()) == artifact


def test_dashboard_prefers_bounded_primary_matrix_slice_when_supplied(tmp_path):
    atlas = _write_fixture(tmp_path)
    source_neurons, target_neurons = _add_full_matrix_slice(atlas)
    artifact = build_dashboard_artifact(
        atlas,
        tmp_path / "artifact-matrix.json",
        provenance_root=tmp_path,
        max_candidates=20,
    )
    matrix = artifact["snapshot"]["datasets"]["primary_matrix_slice"]
    assert len(matrix) == 54
    assert len(matrix) < 2_000
    assert [row["target_neuron"] for row in matrix] == target_neurons
    assert matrix[7]["S00"] == pytest.approx(7.25)
    assert matrix[0]["S07"] == pytest.approx(-0.1)
    assert set(source_neurons).issubset(matrix[0])
    chart = artifact["manifest"]["charts"][0]
    assert chart["id"] == "primary_matrix_chart"
    assert chart["encodings"]["x"]["field"] == "target_neuron"
    assert chart["encodings"]["x"]["label"] == "Target neuron (row)"
    assert chart["encodings"]["y"]["fields"] == source_neurons
    assert chart["encodings"]["y"]["label"] == "Source neuron (column)"
    assert "color" not in chart["encodings"]
    assert "Target neurons are rows" in chart["subtitle"]
    assert "hover for the signed value" in chart["subtitle"]
    assert "does not encode a zero point or sign" in chart["subtitle"]
    assert "palette" not in chart


def test_dashboard_optional_external_reference_is_bounded_and_postfreeze(tmp_path):
    atlas = _write_fixture(tmp_path)
    external = _write_external_fixture(
        tmp_path,
        atlas,
        [
            {"reference": f"edge-{index}", "correlation": index / 10, "network": "Cook"}
            for index in range(8)
        ],
    )
    artifact = build_dashboard_artifact(
        atlas,
        tmp_path / "artifact-external.json",
        provenance_root=tmp_path,
        external_reference=external,
        max_candidates=10,
        max_external_rows=3,
    )
    assert len(artifact["snapshot"]["datasets"]["external_reference_rows"]) == 3
    table = next(
        table
        for table in artifact["manifest"]["tables"]
        if table["id"] == "external_reference_table"
    )
    assert "never used" in table["subtitle"]
    source = next(
        source for source in artifact["sources"] if source["id"] == "external_reference_source"
    )
    assert source["path"] == "external_analysis/dashboard_external_summary.csv"
    assert "not used to rank" in " ".join(source["query"]["filters"])


def test_dashboard_embeds_validated_targeted_n128_as_comparison_only(tmp_path):
    atlas = _write_fixture(tmp_path)
    _add_full_matrix_slice(atlas)
    targeted = _write_targeted_fixture(tmp_path, atlas)
    external = _write_external_fixture(
        tmp_path,
        atlas,
        [{"reference": "Cook", "correlation": 0.2, "network": "chemical"}],
    )
    baseline = build_dashboard_artifact(
        atlas,
        tmp_path / "artifact-baseline.json",
        provenance_root=tmp_path,
        max_candidates=20,
    )
    artifact = build_dashboard_artifact(
        atlas,
        tmp_path / "artifact-targeted.json",
        provenance_root=tmp_path,
        targeted_reference=targeted,
        external_reference=external,
        max_candidates=20,
    )
    datasets = artifact["snapshot"]["datasets"]
    assert len(datasets["targeted_sensitivity_rows"]) == 6
    assert len(datasets["targeted_screen_consistency_rows"]) == 2
    assert len(datasets["external_reference_rows"]) == 1
    assert datasets["candidate_rows"] == baseline["snapshot"]["datasets"][
        "candidate_rows"
    ]
    assert all(
        row["n_particles"] == 128 and row["n_worms"] == 17
        for row in datasets["targeted_sensitivity_rows"]
    )
    factual_rows = [
        row
        for row in datasets["targeted_sensitivity_rows"]
        if row["contrast_involves_factual_model_rollout"]
    ]
    assert factual_rows
    assert all("not an observed response" in row["factual_arm_scope"] for row in factual_rows)
    charts = {chart["id"]: chart for chart in artifact["manifest"]["charts"]}
    assert charts["targeted_screen_comparison_chart"]["type"] == "horizontalBar"
    assert charts["targeted_screen_comparison_chart"]["encodings"]["x"]["field"] == (
        "candidate_label"
    )
    assert charts["targeted_screen_comparison_chart"]["encodings"]["y"]["fields"] == [
        "screen_mean_normalized",
        "targeted_n128_mean_normalized",
    ]
    assert charts["targeted_screen_comparison_chart"]["comparisonContext"][
        "normalization"
    ] == "effect divided by max(abs(achieved high-low source gap), 0.10)"
    tables = {table["id"]: table for table in artifact["manifest"]["tables"]}
    assert "targeted_three_arm_table" in tables
    assert "targeted_screen_consistency_table" in tables
    targeted_guide = next(
        block
        for block in artifact["manifest"]["blocks"]
        if block["id"] == "targeted_n128_guide"
    )
    assert "do **not** re-rank" in targeted_guide["body"]
    assert "not an observed response or causal intervention" in targeted_guide["body"]
    source_ids = {source["id"] for source in artifact["sources"]}
    assert {
        "targeted_cells_source",
        "targeted_screen_source",
        "targeted_protocol_source",
        "external_reference_source",
    }.issubset(source_ids)
    targeted_source = next(
        source for source in artifact["sources"] if source["id"] == "targeted_cells_source"
    )
    assert any(
        "max(abs(achieved high-low source-gap), 0.10)" in definition
        for definition in targeted_source["query"]["metric_definitions"]
    )


def test_dashboard_targeted_reference_fails_closed_on_tamper_and_queue_link(tmp_path):
    atlas = _write_fixture(tmp_path)
    targeted = _write_targeted_fixture(tmp_path, atlas)
    cells_path = targeted / "targeted_cells.csv"
    cells_path.write_text(cells_path.read_text() + "\n")
    with pytest.raises(DashboardInputError, match="checksum failed"):
        build_dashboard_artifact(
            atlas,
            tmp_path / "bad-targeted-checksum.json",
            provenance_root=tmp_path,
            targeted_reference=targeted,
        )

    targeted = _write_targeted_fixture(tmp_path / "linked", atlas)
    selection = tmp_path / "linked" / "targeted_selection"
    manifest_path = selection / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frozen_hypothesis_queue_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    _write_bundle_checksums(
        selection,
        (
            "manifest.json",
            "hypothesis_queue.csv",
            "atlas_matrices.npz",
            "selection_rationale.md",
        ),
    )
    with pytest.raises(DashboardInputError, match="linked to the canonical queue"):
        build_dashboard_artifact(
            atlas,
            tmp_path / "bad-targeted-link.json",
            provenance_root=tmp_path,
            targeted_reference=targeted,
        )

    targeted = _write_targeted_fixture(tmp_path / "dense", atlas)
    selection = tmp_path / "dense" / "targeted_selection"
    (selection / "atlas_matrices.npz").write_bytes(b"different dense matrix")
    _write_bundle_checksums(
        selection,
        (
            "manifest.json",
            "hypothesis_queue.csv",
            "atlas_matrices.npz",
            "selection_rationale.md",
        ),
    )
    with pytest.raises(DashboardInputError, match="linked to the canonical queue"):
        build_dashboard_artifact(
            atlas,
            tmp_path / "bad-targeted-dense-link.json",
            provenance_root=tmp_path,
            targeted_reference=targeted,
        )

    targeted = _write_targeted_fixture(tmp_path / "signed", atlas)
    cells_path = targeted / "targeted_cells.csv"
    cells = pd.read_csv(cells_path)
    cells.loc[0, "signed"] = False
    cells.to_csv(cells_path, index=False)
    _write_bundle_checksums(
        targeted,
        (
            "manifest.json",
            "validation.json",
            "protocol.json",
            "summary.json",
            "targeted_cells.csv",
            "screen_consistency.csv",
            "input_checksums.csv",
        ),
    )
    with pytest.raises(DashboardInputError, match="metadata is inconsistent"):
        build_dashboard_artifact(
            atlas,
            tmp_path / "bad-targeted-signed.json",
            provenance_root=tmp_path,
            targeted_reference=targeted,
        )


def test_dashboard_fails_closed_on_validation_and_timing_mismatch(tmp_path):
    atlas = _write_fixture(tmp_path)
    validation_path = atlas / "validation.json"
    validation = json.loads(validation_path.read_text())
    validation["status"] = "failed"
    validation_path.write_text(json.dumps(validation))
    _refresh_checksums(atlas)
    with pytest.raises(DashboardInputError, match="clean passed audit"):
        build_dashboard_artifact(
            atlas, tmp_path / "bad-validation.json", provenance_root=tmp_path
        )

    validation["status"] = "passed"
    validation_path.write_text(json.dumps(validation))
    queue_path = atlas / "hypothesis_queue.csv"
    queue = pd.read_csv(queue_path)
    queue.loc[0, "source_to_readout_seconds"] += 1
    queue.to_csv(queue_path, index=False)
    _refresh_checksums(atlas)
    with pytest.raises(DashboardInputError, match="timing columns do not reconcile"):
        build_dashboard_artifact(
            atlas, tmp_path / "bad-timing.json", provenance_root=tmp_path
        )


def test_dashboard_rejects_input_overwrite_duplicate_ledger_and_rogue_external(tmp_path):
    atlas = _write_fixture(tmp_path)
    with pytest.raises(DashboardInputError, match="output must be outside"):
        build_dashboard_artifact(
            atlas,
            atlas / "artifact.json",
            provenance_root=tmp_path,
        )

    ledger = atlas / "checksums.sha256"
    ledger.write_text(ledger.read_text() + ledger.read_text().splitlines()[0] + "\n")
    with pytest.raises(DashboardInputError, match="duplicates"):
        build_dashboard_artifact(
            atlas,
            tmp_path / "duplicate-ledger.json",
            provenance_root=tmp_path,
        )

    atlas = _write_fixture(tmp_path / "rogue")
    rogue = tmp_path / "rogue_external" / "rogue.csv"
    rogue.parent.mkdir()
    pd.DataFrame([{"reference": "Cook", "correlation": 0.2}]).to_csv(
        rogue, index=False
    )
    with pytest.raises(DashboardInputError, match="canonical dashboard_external_summary"):
        build_dashboard_artifact(
            atlas,
            tmp_path / "rogue-output.json",
            provenance_root=tmp_path,
            external_reference=rogue,
        )


def test_synthetic_artifact_passes_installed_portable_delivery(tmp_path):
    node = shutil.which("node")
    builder = Path(
        "/Users/vik/.codex/plugins/cache/openai-curated-remote/data-analytics/"
        "0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs"
    )
    if node is None or not builder.exists():
        pytest.skip("installed Data Analytics portable builder is unavailable")
    atlas = _write_fixture(tmp_path)
    _add_full_matrix_slice(atlas)
    targeted = _write_targeted_fixture(tmp_path, atlas)
    artifact_path = tmp_path / "artifact.json"
    html_path = tmp_path / "dashboard.html"
    build_dashboard_artifact(
        atlas,
        artifact_path,
        provenance_root=tmp_path,
        targeted_reference=targeted,
        max_candidates=20,
    )
    completed = subprocess.run(
        [
            node,
            str(builder),
            "--input",
            str(artifact_path),
            "--output",
            str(html_path),
            "--timeout-ms",
            "20000",
        ],
        text=True,
        capture_output=True,
        timeout=40,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["stages"]["validation"] == "passed"
    assert receipt["stages"]["package"] == "passed"
    assert receipt["stages"]["verification"] in {"passed", "structural_only"}
    assert html_path.exists()
