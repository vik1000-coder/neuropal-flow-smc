from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.prediction_atlas_select_targeted import (
    ADDED_COLUMNS,
    CANONICAL_QUEUE_COLUMNS,
    SUPPORTED_CHANNELS,
    SUPPORTED_CONTEXTS,
    TargetedSelectionError,
    freeze_targeted_selection,
    sha256,
    targeted_phases_for_context,
)
from compatibility_neural_benchmark.targeted_confirmation import load_candidate_groups


def _row(
    rank: int,
    *,
    channel: str,
    source_index: int,
    target_index: int,
    promoted: bool,
    score: float,
    context: str = "onset_minus_baseline",
    lag: int = 1,
    horizon: int = 1,
) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in CANONICAL_QUEUE_COLUMNS}
    row.update(
        {
            "queue_rank": rank,
            "method": "progressive_bridge_smc",
            "model_id": "synthetic_binary_flow",
            "channel": channel,
            "context": context,
            "chemical": "",
            "conditioning_status": "binary_any_stimulus_not_chemical",
            "source_neuron": f"S{source_index}",
            "target_neuron": (
                f"S{source_index}" if source_index == target_index else f"T{target_index}"
            ),
            "source_index": source_index,
            "target_index": target_index,
            "source_lag_frames": lag,
            "source_to_cut_seconds": lag / 4.0,
            "horizon_frames": horizon,
            "forecast_horizon_seconds": horizon / 4.0,
            "source_to_readout_seconds": (lag + horizon) / 4.0,
            "mean_raw": 0.2,
            "mean_normalized": 0.3,
            "median_normalized": 0.25,
            "ci_2_5": 0.05,
            "ci_97_5": 0.5,
            "screen_t_p_value": 0.01,
            "sign_flip_p_value": 0.02,
            "sign_flip_q_value": 0.04,
            "bh_family": "internal",
            "inference_scope": "post_screen_exploratory_no_full_family_fdr_control",
            "test_sidedness": "two_sided_candidate_sign_flip",
            "sign_consistency": 0.9,
            "valid_fraction": 0.8,
            "genealogy_gate_applicable": True,
            "genealogy_min_distinct_ancestor_fraction_strong": 0.10,
            "genealogy_min_distinct_ancestor_fraction_sensitivity": 0.20,
            "genealogy_valid_fraction_0_10": 0.8,
            "genealogy_valid_fraction_0_20": 0.7,
            "genealogy_strong_gate_pass": True,
            "genealogy_sensitivity_gate_pass": False,
            "seed_spearman": 0.8,
            "seed_sign_agreement": 1.0,
            "n_worms": 17,
            "effect_direction": "positive",
            "support_tier": "supported_exploratory" if promoted else "model_only",
            "evidence_score": score,
            "interpretation_limit": "model_based_lag_association_not_causal_or_physical_delay",
            "counterpart_method": "direct_importance",
            "counterpart_mean_normalized": 0.2,
            "counterpart_valid_fraction": 0.75,
            "cross_sampler_sign_agreement": 1.0,
            "cross_sampler_magnitude_agreement": 0.8,
            "cross_sampler_spearman": 0.7,
            "cross_sampler_slice_agreement": 0.85,
            "cross_sampler_support_agreement": 0.75,
            "cross_sampler_factor": 0.5,
            "base_evidence_score": score * 2,
            "promotion_eligible": promoted,
            "lag_profile_frames": "[1, 4, 8, 16]",
            "lag_profile_source_to_cut_seconds": "[0.25, 1.0, 2.0, 4.0]",
            "primary_lag_profile": "[0.3, 0.2, 0.1, 0.05]",
            "counterpart_lag_profile": "[0.2, 0.15, 0.1, 0.05]",
            "peak_abs_effect_lag_frames": 1,
            "peak_abs_effect_lag_seconds": 0.25,
            "peak_abs_effect": 0.3,
            "second_abs_effect": 0.2,
            "top_vs_second_lag_selectivity": 1 / 3,
            "lag_profile_spearman": 1.0,
            "signed_lag_profile_spearman": 1.0,
            "worm_bootstrap_peak_lag_selection_rate": 0.75,
            "worm_bootstrap_peak_lag_rates": "[0.75, 0.15, 0.05, 0.05]",
            "lag_interpretation": "descriptive_model_lag_not_physical_delay",
        }
    )
    return row


def _checksum_bundle(atlas: Path) -> None:
    names = sorted(
        path.name
        for path in atlas.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (atlas / "checksums.sha256").write_text(
        "".join(f"{sha256(atlas / name)}  {name}\n" for name in names)
    )


def _atlas(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    atlas = tmp_path / "atlas"
    atlas.mkdir()
    pd.DataFrame(rows, columns=CANONICAL_QUEUE_COLUMNS).to_csv(
        atlas / "hypothesis_queue.csv", index=False
    )
    np.savez_compressed(atlas / "atlas_matrices.npz", orientation=np.asarray("target_row_source_column"))
    (atlas / "models.json").write_text(
        json.dumps(
            {
                "generator_stimulus_encoding": "binary_any_stimulus",
                "chemical_conditioning": False,
            }
        )
    )
    (atlas / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol": "reviewed neural prediction atlas v1",
                "primary_method": "progressive_bridge_smc",
                "dense_matrix_orientation": "target_row_source_column",
                "artifacts": {
                    "hypothesis_queue": "hypothesis_queue.csv",
                    "atlas_matrices": "atlas_matrices.npz",
                    "models": "models.json",
                    "protocol": "protocol.json",
                    "validation": "validation.json",
                },
            }
        )
    )
    (atlas / "protocol.json").write_text(
        json.dumps(
            {
                "protocol": "reviewed neural prediction atlas v1",
                "n_worms": 17,
                "n_neurons": 54,
                "fps": 4.0,
                "source_lag_frames": [1, 4, 8, 16],
                "horizon_frames": [1, 2, 4, 8, 16, 32],
                "orientation": "target_row_source_column",
                "channels": list(SUPPORTED_CHANNELS),
                "contexts": [{"context": value} for value in SUPPORTED_CONTEXTS],
                "timing_definitions": {
                    "source_to_cut_seconds": "source_lag_frames / fps",
                    "forecast_horizon_seconds": "horizon_frames / fps",
                    "source_to_readout_seconds": "(source_lag_frames + horizon_frames) / fps",
                },
                "chemical_context_warning": "event-stratified and not chemically conditioned",
                "interpretation_limit": "not a causal or physical-delay estimate",
                "evidence_tiers": {
                    "confirmed": "reserved",
                    "supported_exploratory": "supported",
                    "model_only": "model",
                    "unsupported": "unsupported",
                },
                "ranking": "no connectome or external reference data used",
                "support_thresholds": {
                    "minimum": 0.5,
                    "strong": 0.8,
                    "strong_sign_consistency": 0.8,
                },
            }
        )
    )
    (atlas / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "problems": [],
                "n_worms": 17,
                "n_neurons": 54,
                "archive_checks_completed": {"orientation_self_test": True},
                "hypothesis_queue_audit": {
                    "rows": len(rows),
                    "non_primary_rows": 0,
                    "unsupported_rows": 0,
                    "promoted_with_counterpart_below_support_gate": 0,
                    "promoted_signed_rows_with_sign_disagreement": 0,
                    "promoted_with_genealogy_gate_failure": 0,
                },
                "candidate_lag_profile_audit": {
                    "queue_rows": len(rows),
                    "profile_rows": len(rows) * 4,
                    "expected_profile_rows": len(rows) * 4,
                    "used_for_promotion": False,
                },
            }
        )
    )
    _checksum_bundle(atlas)
    return atlas


def _diverse_rows() -> list[dict[str, object]]:
    return [
        _row(1, channel="endpoint_mean", source_index=1, target_index=20, promoted=True, score=100),
        _row(2, channel="endpoint_log_sd", source_index=1, target_index=21, promoted=True, score=90),
        _row(3, channel="endpoint_log_sd", source_index=2, target_index=22, promoted=True, score=80),
        _row(4, channel="endpoint_wasserstein1", source_index=3, target_index=23, promoted=True, score=70),
        _row(5, channel="cumulative_mean", source_index=4, target_index=24, promoted=False, score=200),
        _row(6, channel="peak_mean", source_index=5, target_index=25, promoted=False, score=190),
        _row(7, channel="event_probability", source_index=6, target_index=26, promoted=False, score=180),
        _row(8, channel="endpoint_sd", source_index=7, target_index=27, promoted=False, score=170),
        _row(9, channel="endpoint_mean", source_index=8, target_index=8, promoted=True, score=1000),
    ]


def test_freeze_is_bounded_diverse_reproducible_and_targeted_compatible(tmp_path: Path) -> None:
    atlas = _atlas(tmp_path, _diverse_rows())
    first = tmp_path / "selection_one"
    second = tmp_path / "selection_two"
    manifest = freeze_targeted_selection(atlas, first)
    freeze_targeted_selection(atlas, second)

    assert (first / "hypothesis_queue.csv").read_bytes() == (
        second / "hypothesis_queue.csv"
    ).read_bytes()
    selected = pd.read_csv(first / "hypothesis_queue.csv")
    assert list(selected.columns) == list(CANONICAL_QUEUE_COLUMNS) + list(ADDED_COLUMNS)
    assert list(selected.columns) == list(CANONICAL_QUEUE_COLUMNS) + [
        "run_confirmation"
    ]
    assert "selection_rank" not in selected
    assert "selection_rule" not in selected
    assert "frozen_hypothesis_queue_sha256" not in selected
    assert len(selected) == 6
    assert selected["run_confirmation"].all()
    assert selected["source_neuron"].nunique() == 6
    assert selected["channel"].nunique() == 6
    assert not (selected["source_index"] == selected["target_index"]).any()
    assert selected["queue_rank"].tolist() == [1, 3, 4, 5, 6, 7]
    assert manifest["particles"] == 128
    assert manifest["excluded_diagonal_rows"] == 1
    assert manifest["external_reference_inputs_used"] == []
    assert [row["selection_rank"] for row in manifest["selected_rows_metadata"]] == list(
        range(1, 7)
    )
    assert {
        row["frozen_hypothesis_queue_sha256"]
        for row in manifest["selected_rows_metadata"]
    } == {sha256(atlas / "hypothesis_queue.csv")}
    source = pd.read_csv(
        atlas / "hypothesis_queue.csv", dtype=str, keep_default_na=False
    ).set_index("queue_rank", drop=False)
    staged = pd.read_csv(
        first / "hypothesis_queue.csv", dtype=str, keep_default_na=False
    )
    expected = source.loc[staged["queue_rank"], list(CANONICAL_QUEUE_COLUMNS)]
    pd.testing.assert_frame_equal(
        staged.loc[:, list(CANONICAL_QUEUE_COLUMNS)].reset_index(drop=True),
        expected.reset_index(drop=True),
    )
    assert "No anatomical" in (first / "selection_rationale.md").read_text()

    checksum_lines = (first / "checksums.sha256").read_text().splitlines()
    assert len(checksum_lines) == 4
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        assert sha256(first / name) == digest
    assert sha256(first / "atlas_matrices.npz") == sha256(
        atlas / "atlas_matrices.npz"
    )

    neurons = tuple([f"S{index}" for index in range(10)] + [f"T{index}" for index in range(10, 54)])
    # Selected targets are irrelevant to sampler grouping; selected sources and
    # contexts must nevertheless resolve under the real runner mapping.
    groups = load_candidate_groups(first / "hypothesis_queue.csv", neurons)
    assert groups
    assert all(group.phases == ("baseline", "onset") for group in groups)


def test_promotion_precedes_evidence_within_diversity_pass(tmp_path: Path) -> None:
    rows = [
        _row(1, channel="endpoint_mean", source_index=1, target_index=20, promoted=False, score=1000),
        _row(2, channel="endpoint_mean", source_index=2, target_index=21, promoted=True, score=1),
        _row(3, channel="endpoint_mean", source_index=3, target_index=22, promoted=False, score=900),
    ]
    atlas = _atlas(tmp_path, rows)
    output = tmp_path / "selection"
    freeze_targeted_selection(atlas, output, max_rows=2)
    selected = pd.read_csv(output / "hypothesis_queue.csv")
    assert selected["queue_rank"].tolist() == [2, 1]
    manifest = json.loads((output / "manifest.json").read_text())
    assert [row["selection_rule"] for row in manifest["selected_rows_metadata"]] == [
        "channel_coverage_unique_source",
        "ranked_fill_unique_source",
    ]


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ("baseline", ("baseline",)),
        ("state_average", ("baseline", "onset", "active", "offset", "recovery")),
        ("butanone_onset", ("onset",)),
        ("nacl_onset_minus_baseline", ("baseline", "onset")),
    ],
)
def test_targeted_context_mapping(context: str, expected: tuple[str, ...]) -> None:
    assert targeted_phases_for_context(context) == expected


def test_unknown_suffix_context_is_rejected() -> None:
    with pytest.raises(TargetedSelectionError, match="not one of the canonical"):
        targeted_phases_for_context("cook_onset")


def test_checksum_tamper_fails_before_selection(tmp_path: Path) -> None:
    atlas = _atlas(tmp_path, _diverse_rows())
    with (atlas / "hypothesis_queue.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(TargetedSelectionError, match="checksum failed"):
        freeze_targeted_selection(atlas, tmp_path / "selection")
    assert not (tmp_path / "selection").exists()


def test_external_field_and_failed_validation_are_rejected(tmp_path: Path) -> None:
    atlas = _atlas(tmp_path, _diverse_rows())
    frame = pd.read_csv(atlas / "hypothesis_queue.csv")
    frame["cook_auroc"] = 0.9
    frame.to_csv(atlas / "hypothesis_queue.csv", index=False)
    _checksum_bundle(atlas)
    with pytest.raises(TargetedSelectionError, match="external-reference fields"):
        freeze_targeted_selection(atlas, tmp_path / "external_selection")

    frame = frame.drop(columns="cook_auroc")
    frame.to_csv(atlas / "hypothesis_queue.csv", index=False)
    validation = json.loads((atlas / "validation.json").read_text())
    validation["status"] = "failed"
    (atlas / "validation.json").write_text(json.dumps(validation))
    _checksum_bundle(atlas)
    with pytest.raises(TargetedSelectionError, match="did not pass"):
        freeze_targeted_selection(atlas, tmp_path / "failed_selection")


def test_existing_or_overlapping_output_is_never_overwritten(tmp_path: Path) -> None:
    atlas = _atlas(tmp_path, _diverse_rows())
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        freeze_targeted_selection(atlas, existing)
    assert (existing / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="disjoint"):
        freeze_targeted_selection(atlas, atlas / "selection")
