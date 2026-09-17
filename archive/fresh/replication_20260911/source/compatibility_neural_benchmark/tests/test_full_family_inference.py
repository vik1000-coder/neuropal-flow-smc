from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.full_family_inference import (
    _randomization_p,
    benjamini_hochberg,
    generate_sign_patterns,
    infer_complete_family,
    load_support_eligibility,
    run_full_family_inference,
    signed_student_t,
)


def test_benjamini_hochberg_handles_complete_family_and_nans():
    values = np.asarray([0.001, 0.02, np.nan, 0.04])
    adjusted = benjamini_hochberg(values)
    np.testing.assert_allclose(adjusted[[0, 1, 3]], [0.003, 0.03, 0.04])
    assert np.isnan(adjusted[2])
    assert np.all(np.diff(adjusted[[0, 1, 3]]) >= 0)


def test_exact_sign_patterns_are_unique_two_sided_orbits():
    signs, mode = generate_sign_patterns(5, mode="exact", max_exact_patterns=16)
    assert mode == "exact"
    assert signs.shape == (16, 5)
    np.testing.assert_array_equal(signs[:, 0], np.ones(16, dtype=np.int8))
    assert len(np.unique(signs, axis=0)) == 16
    assert set(np.unique(signs)) == {-1, 1}


def test_exact_randomization_p_never_drops_observed_orbit():
    # Batched and separately computed statistics can differ by a few ulps.
    # The all-positive observed assignment is nevertheless part of the exact
    # sign orbit, so an exact p-value is mathematically bounded below by 1/B.
    p_value = _randomization_p(np.asarray([0, 2]), 65_536, "exact")
    np.testing.assert_allclose(p_value, [1 / 65_536, 2 / 65_536])


def test_shared_signs_are_applied_to_every_cell():
    values = np.asarray(
        [
            [[1.0, 10.0]],
            [[2.0, 20.0]],
            [[4.0, 40.0]],
            [[8.0, 80.0]],
        ]
    )  # [worm, edge, cell]
    signs = np.asarray([[1, -1, 1, -1], [1, 1, -1, -1]], dtype=np.int8)
    statistic = signed_student_t(values, signs)
    # The second cell is an exact positive rescaling of the first.  Reusing the
    # same worm signs must therefore give identical studentized null values.
    np.testing.assert_allclose(statistic[:, 0, 0], statistic[:, 0, 1])


def test_synthetic_injected_edge_survives_edge_family_and_lag_test():
    rng = np.random.default_rng(81)
    n_worms, n_edges, n_lags, n_horizons = 12, 20, 3, 2
    values = rng.normal(0.0, 0.35, (n_worms, n_edges, n_lags, n_horizons))
    # A stable edge with one much stronger lag: the edge omnibus and flat-lag
    # omnibus should both dominate their respective complete edge families.
    values[:, 4, :, 0] += np.asarray([0.0, 0.0, 4.0])[None, :]
    result = infer_complete_family(
        values,
        mode="exact",
        max_exact_patterns=2 ** (n_worms - 1),
        batch_size=64,
    )
    assert result.mean.shape == (n_edges, n_lags, n_horizons)
    assert result.edge_p_value.shape == (n_edges,)
    assert result.edge_bh_q_value[4] < 0.05
    assert result.flat_lag_bh_q_value[4] < 0.05
    assert result.edge_bh_q_value[4] == np.nanmin(result.edge_bh_q_value)
    assert result.flat_lag_bh_q_value[4] == np.nanmin(
        result.flat_lag_bh_q_value
    )


def test_null_tensor_has_no_false_signal_when_worm_means_cancel_exactly():
    rng = np.random.default_rng(7)
    half = rng.normal(size=(5, 8, 2, 2))
    values = np.concatenate([half, -half], axis=0)
    result = infer_complete_family(
        values,
        mode="exact",
        max_exact_patterns=2 ** 9,
        batch_size=32,
    )
    np.testing.assert_allclose(result.t_statistic, 0.0, atol=1e-12)
    assert np.all(result.edge_p_value > 0.95)
    assert not np.any(result.edge_bh_q_value < 0.05)


def test_monte_carlo_results_are_reproducible_with_fixed_seed():
    values = np.random.default_rng(3).normal(size=(9, 5, 2, 2))
    first = infer_complete_family(
        values,
        mode="monte-carlo",
        replicates=257,
        seed=123,
        batch_size=31,
    )
    second = infer_complete_family(
        values,
        mode="monte-carlo",
        replicates=257,
        seed=123,
        batch_size=64,
    )
    np.testing.assert_array_equal(first.signs, second.signs)
    np.testing.assert_array_equal(first.edge_p_value, second.edge_p_value)
    np.testing.assert_array_equal(
        first.global_max_t_p_value, second.global_max_t_p_value
    )


def _write_worm_archive(path: Path) -> None:
    rng = np.random.default_rng(44)
    neurons = np.asarray(["A", "B", "C", "D"])
    worms = np.asarray([f"worm-{index}" for index in range(8)])
    lags = np.asarray([1, 4], dtype=np.int16)
    horizons = np.asarray([1, 2, 4], dtype=np.int16)
    values = rng.normal(0.0, 0.4, (2, 8, 3, 4, 4)).astype(np.float32)
    values[1, :, 2, 1, 0] += 3.0
    np.savez_compressed(
        path,
        neurons=neurons,
        worm_ids=worms,
        methods=np.asarray(["direct_importance", "progressive_bridge_smc"]),
        channels=np.asarray(["endpoint_mean", "endpoint_wasserstein1"]),
        contexts=np.asarray(["baseline", "onset_minus_baseline"]),
        source_lag_frames=lags,
        source_lag_seconds=lags.astype(np.float32) / 4.0,
        horizon_frames=horizons,
        horizon_seconds=horizons.astype(np.float32) / 4.0,
        orientation=np.asarray("target_row_source_column"),
        primary_method=np.asarray("progressive_bridge_smc"),
        scope=np.asarray("synthetic"),
        normalized__endpoint_mean__baseline=values,
        normalized__endpoint_mean__active=values + 1.0,
        normalized__endpoint_wasserstein1__baseline=np.abs(values),
    )


def test_runner_excludes_diagonal_and_writes_auditable_bundle(tmp_path: Path):
    archive = tmp_path / "worms.npz"
    output = tmp_path / "out"
    _write_worm_archive(archive)
    manifest = run_full_family_inference(
        archive,
        output,
        channel="endpoint_mean",
        context="baseline",
        mode="exact",
        max_exact_patterns=2 ** 7,
        batch_size=17,
    )
    assert manifest["status"] == "complete"
    edge = pd.read_csv(output / "edge_inference.csv")
    cells = pd.read_csv(output / "cell_inference.csv")
    assert len(edge) == 4 * 3
    assert len(cells) == 4 * 3 * 2 * 3
    assert np.all(edge.target_index != edge.source_index)
    assert {
        "edge_omnibus_p_value",
        "edge_bh_q_value",
        "flat_lag_bh_q_value",
    }.issubset(edge.columns)
    assert {
        "global_max_t_p_value",
        "zero_null_reference_band_low",
        "lag_contrast_global_max_t_p_value",
    }.issubset(cells.columns)
    with np.load(output / "inference_arrays.npz") as dense:
        assert dense["mean_normalized"].shape == (4, 4, 2, 3)
        assert dense["sign_patterns"].shape == (128, 8)
        assert np.all(np.isnan(np.diagonal(dense["edge_bh_q_value"])))
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["replication"]["independent_unit"] == "worm"
    assert protocol["family"]["directed_edges"] == 12
    assert "shared_sign_rule" in protocol["randomization"]
    assert (output / "checksums.sha256").exists()


def test_unsigned_within_state_distance_requires_sham_null(tmp_path: Path):
    archive = tmp_path / "worms.npz"
    _write_worm_archive(archive)
    with pytest.raises(ValueError, match="sham-distance"):
        run_full_family_inference(
            archive,
            tmp_path / "bad",
            channel="endpoint_wasserstein1",
            context="baseline",
            mode="monte-carlo",
            replicates=16,
        )


def test_active_minus_baseline_is_derived_at_worm_level(tmp_path: Path):
    archive = tmp_path / "worms.npz"
    output = tmp_path / "contrast"
    _write_worm_archive(archive)
    run_full_family_inference(
        archive,
        output,
        channel="endpoint_mean",
        context="baseline",
        contrast="active-minus-baseline",
        mode="monte-carlo",
        replicates=31,
    )
    cells = pd.read_csv(output / "cell_inference.csv")
    np.testing.assert_allclose(cells.mean_normalized, 1.0)
    assert set(cells.context) == {"active_minus_baseline"}
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["derived_contrast"]["name"] == "active-minus-baseline"


def test_support_gate_is_joint_across_contexts_and_every_lag(tmp_path: Path):
    rows = []
    for context in ("baseline", "active"):
        for source in range(3):
            for lag in (1, 4):
                valid = 0.9
                genealogy = 0.9
                if source == 1 and lag == 4:
                    valid = 0.79
                if source == 2 and context == "active":
                    genealogy = 0.79
                rows.append(
                    {
                        "method": "progressive_bridge_smc",
                        "context": context,
                        "source_index": source,
                        "source_lag_frames": lag,
                        "valid_fraction": valid,
                        "support_qualified": True,
                        "genealogy_gate_applicable": True,
                        "genealogy_valid_fraction_0_10": genealogy,
                        "genealogy_strong_gate_pass": True,
                    }
                )
    path = tmp_path / "support.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    support = load_support_eligibility(
        path,
        method="progressive_bridge_smc",
        support_contexts=("baseline", "active"),
        n_neurons=3,
        selected_lags=(1, 4),
        strong_threshold=0.8,
        genealogy_threshold=0.8,
        sensitivity_threshold=0.5,
    )
    np.testing.assert_array_equal(
        support.strong_source_eligible, [True, False, False]
    )
    np.testing.assert_array_equal(
        support.sensitivity_source_eligible, [True, True, True]
    )


def test_runner_marks_support_ineligible_edges_untested(tmp_path: Path):
    archive = tmp_path / "worms.npz"
    output = tmp_path / "gated"
    _write_worm_archive(archive)
    rows = []
    for source in range(4):
        for lag in (1, 4):
            passed = source == 0
            rows.append(
                {
                    "method": "progressive_bridge_smc",
                    "context": "baseline",
                    "source_index": source,
                    "source_lag_frames": lag,
                    "valid_fraction": 0.9 if passed else 0.2,
                    "support_qualified": passed,
                    "genealogy_gate_applicable": True,
                    "genealogy_valid_fraction_0_10": 0.9 if passed else 0.2,
                    "genealogy_strong_gate_pass": passed,
                }
            )
    support_path = tmp_path / "support.csv"
    pd.DataFrame(rows).to_csv(support_path, index=False)
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()
    parent_ledger = tmp_path / "checksums.sha256"
    parent_ledger.write_text(f"{support_sha256}  support.csv\n")
    parent_ledger_sha256 = hashlib.sha256(parent_ledger.read_bytes()).hexdigest()
    run_full_family_inference(
        archive,
        output,
        channel="endpoint_mean",
        context="baseline",
        support_cells=support_path,
        mode="exact",
        max_exact_patterns=128,
    )
    edge = pd.read_csv(output / "edge_inference.csv")
    assert edge.support_eligible.sum() == 3
    unsupported = edge.loc[~edge.support_eligible]
    assert unsupported.edge_omnibus_p_value.isna().all()
    np.testing.assert_allclose(unsupported.edge_bh_q_value, 1.0)
    assert not unsupported.edge_bh_reject.any()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["family"]["support_eligible_directed_edges"] == 3
    support_protocol = protocol["support_gate"]
    assert support_protocol["sha256"] == support_sha256
    assert support_protocol["parent_checksum_ledger"] == {
        "path": str(parent_ledger.resolve()),
        "sha256": parent_ledger_sha256,
        "entry_name": "support.csv",
        "entry_sha256": support_sha256,
    }
    validation = json.loads((output / "validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["support_table"]["sha256"] == support_sha256
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["inputs"]["support_table"] == validation["support_table"]
    assert manifest["artifacts"]["validation"] == "validation.json"
    assert "validation.json" in (output / "checksums.sha256").read_text()


def test_runner_rejects_stale_covering_support_ledger(tmp_path: Path):
    archive = tmp_path / "worms.npz"
    _write_worm_archive(archive)
    rows = []
    for source in range(4):
        for lag in (1, 4):
            rows.append(
                {
                    "method": "progressive_bridge_smc",
                    "context": "baseline",
                    "source_index": source,
                    "source_lag_frames": lag,
                    "valid_fraction": 0.9,
                    "support_qualified": True,
                    "genealogy_gate_applicable": True,
                    "genealogy_valid_fraction_0_10": 0.9,
                    "genealogy_strong_gate_pass": True,
                }
            )
    support_path = tmp_path / "support.csv"
    pd.DataFrame(rows).to_csv(support_path, index=False)
    (tmp_path / "checksums.sha256").write_text(f"{'0' * 64}  support.csv\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_full_family_inference(
            archive,
            tmp_path / "stale",
            channel="endpoint_mean",
            context="baseline",
            support_cells=support_path,
            mode="exact",
            max_exact_patterns=128,
        )


def test_runner_can_use_reported_minimum_support_as_sensitivity_only(tmp_path: Path):
    archive = tmp_path / "worms.npz"
    output = tmp_path / "sensitivity"
    _write_worm_archive(archive)
    rows = []
    for source in range(4):
        for lag in (1, 4):
            valid = 0.9 if source == 0 else 0.6 if source == 1 else 0.2
            rows.append(
                {
                    "method": "progressive_bridge_smc",
                    "context": "baseline",
                    "source_index": source,
                    "source_lag_frames": lag,
                    "valid_fraction": valid,
                    "support_qualified": valid >= 0.5,
                    "genealogy_gate_applicable": True,
                    "genealogy_valid_fraction_0_10": valid,
                    "genealogy_strong_gate_pass": valid >= 0.8,
                }
            )
    support_path = tmp_path / "support.csv"
    pd.DataFrame(rows).to_csv(support_path, index=False)
    run_full_family_inference(
        archive,
        output,
        channel="endpoint_mean",
        context="baseline",
        support_cells=support_path,
        support_gate="sensitivity",
        mode="exact",
        max_exact_patterns=128,
    )
    edge = pd.read_csv(output / "edge_inference.csv")
    assert edge.support_eligible.sum() == 6
    assert set(edge.support_gate) == {"sensitivity"}
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["support_gate"]["applied_gate"] == "sensitivity"
    assert protocol["support_gate"]["inference_eligible_sources"] == 2
