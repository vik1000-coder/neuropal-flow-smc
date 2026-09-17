from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.prediction_atlas_runner import sha256
from compatibility_neural_benchmark.targeted_sampling_null_analysis import (
    AnalysisConfig,
    _extract_event_rows,
    _bh,
    _joint_studentized_max_t,
    _parse_checksum_ledger,
    _sign_flip_p,
    _summarize,
    endpoint_metrics,
    parent_safe_split_masks,
)
from compatibility_neural_benchmark.targeted_sampling_nulls import (
    SamplingGroup,
    SelectedCell,
    recompute_diagnostic_valid,
)


def test_endpoint_metrics_have_declared_sign_and_wasserstein() -> None:
    low = np.asarray([-2.0, -1.0, 0.0, 1.0])
    high = low + 1.5
    result = endpoint_metrics(high, low)
    assert result["endpoint_mean"] == pytest.approx(1.5)
    assert result["endpoint_log_sd"] == pytest.approx(0.0)
    assert result["endpoint_wasserstein1"] == pytest.approx(1.5)


def test_endpoint_wasserstein_supports_unequal_parent_safe_halves() -> None:
    result = endpoint_metrics(np.asarray([0.0, 2.0]), np.asarray([1.0]))
    assert result["endpoint_mean"] == pytest.approx(0.0)
    assert result["endpoint_wasserstein1"] == pytest.approx(1.0)


def test_parent_safe_split_never_separates_future_siblings() -> None:
    parent = np.repeat(np.arange(16), 2)
    left, right = parent_safe_split_masks(parent, seed=17)
    assert left.sum() == right.sum() == 16
    assert not set(parent[left]).intersection(set(parent[right]))
    for value in np.unique(parent):
        positions = np.flatnonzero(parent == value)
        assert left[positions].all() or right[positions].all()


def test_sign_flip_and_bh_are_bounded_and_deterministic() -> None:
    values = np.asarray([1.0, 1.5, 2.0, 0.5, 1.25])
    p = _sign_flip_p(values)
    assert 0 < p <= 1
    assert p == _sign_flip_p(values)
    q = _bh([0.01, 0.03, 0.20])
    assert np.all((q >= 0) & (q <= 1))
    assert q[0] <= q[1] <= q[2]


def test_schema_v2_locks_named_genealogy_threshold() -> None:
    AnalysisConfig(genealogy_fraction_threshold=0.10).validate()
    with pytest.raises(ValueError, match="fixes the genealogy"):
        AnalysisConfig(genealogy_fraction_threshold=0.20).validate()


def test_equivalent_max_weight_above_one_is_legal_but_invalid() -> None:
    prefix = "observed"
    diagnostics = {
        f"{prefix}_diagnostic_ess_low": np.asarray([10.0]),
        f"{prefix}_diagnostic_ess_high": np.asarray([10.0]),
        f"{prefix}_diagnostic_max_weight_low": np.asarray([1.5]),
        f"{prefix}_diagnostic_max_weight_high": np.asarray([0.1]),
        f"{prefix}_diagnostic_achieved_low": np.asarray([0.0]),
        f"{prefix}_diagnostic_achieved_high": np.asarray([1.0]),
        f"{prefix}_diagnostic_achieved_gap": np.asarray([1.0]),
        f"{prefix}_diagnostic_target_low": np.asarray([0.0]),
        f"{prefix}_diagnostic_target_high": np.asarray([1.0]),
        f"{prefix}_diagnostic_target_gap": np.asarray([1.0]),
    }
    valid = recompute_diagnostic_valid(diagnostics, prefix, min_ess=5.0)
    np.testing.assert_array_equal(valid, [False])


def test_checksum_parser_rejects_unsafe_and_tampered_entries(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("safe\n")
    ledger = tmp_path / "checksums.sha256"
    ledger.write_text(f"{sha256(payload)}  payload.txt\n")
    assert _parse_checksum_ledger(tmp_path) == {"payload.txt": sha256(payload)}

    ledger.write_text(f"{sha256(payload)}  ../payload.txt\n")
    with pytest.raises(RuntimeError, match="unsafe path"):
        _parse_checksum_ledger(tmp_path)

    ledger.write_text(f"{'0' * 64}  payload.txt\n")
    with pytest.raises(RuntimeError, match="verification failed"):
        _parse_checksum_ledger(tmp_path)


def test_checksum_parser_requires_exact_tree_coverage(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one")
    second.write_text("two")
    (tmp_path / "checksums.sha256").write_text(
        f"{sha256(first)}  first.txt\n"
    )
    with pytest.raises(RuntimeError, match="coverage differs"):
        _parse_checksum_ledger(tmp_path)


def _selected_cell(
    candidate_id: str,
    *,
    selection_origin: str = "strong_primary",
) -> SelectedCell:
    return SelectedCell(
        candidate_id=candidate_id,
        queue_rank=1,
        source=0,
        target=1,
        source_name="SRC",
        target_name="DST",
        lag=1,
        horizon=1,
        context="baseline",
        phases=("baseline",),
        channel="endpoint_mean",
        selection_origin=selection_origin,
    )


def _contextual_fixture(
    cells: tuple[SelectedCell, ...], *, n_worms: int = 5
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Small already-contextualized panel with independently tunable gates."""
    worm_ids = tuple(f"worm_{index}" for index in range(n_worms))
    rows: list[dict[str, object]] = []
    values = {
        "observed": 2.0,
        "low_low": 0.05,
        "high_high": 0.10,
        "midpoint_midpoint": 0.90,
        "quiet_pseudo": 0.15,
    }
    for cell in cells:
        for metric in ("endpoint_mean", "endpoint_log_sd", "endpoint_wasserstein1"):
            for worm_index, worm_id in enumerate(worm_ids):
                for control, value in values.items():
                    valid = True
                    ancestor_fraction = 0.50
                    if (
                        cell.candidate_id == "observed_invalid"
                        and control == "observed"
                        and worm_index >= 3
                    ):
                        valid = False
                    if (
                        cell.candidate_id == "pseudo_invalid"
                        and control == "quiet_pseudo"
                        and worm_index >= 3
                    ):
                        valid = False
                    if (
                        cell.candidate_id == "sampling_invalid"
                        and control == "midpoint_midpoint"
                        and worm_index >= 3
                    ):
                        valid = False
                    if (
                        cell.candidate_id == "observed_genealogy"
                        and control == "observed"
                        and worm_index >= 3
                    ):
                        ancestor_fraction = 0.05
                    if (
                        cell.candidate_id == "pseudo_genealogy"
                        and control == "quiet_pseudo"
                        and worm_index >= 3
                    ):
                        ancestor_fraction = 0.05
                    if (
                        cell.candidate_id == "sampling_genealogy"
                        and control == "midpoint_midpoint"
                        and worm_index >= 3
                    ):
                        ancestor_fraction = 0.05
                    if (
                        cell.candidate_id == "disjoint_observed_failures"
                        and control == "observed"
                    ):
                        if worm_index == 0:
                            valid = False
                        if worm_index == 1:
                            ancestor_fraction = 0.05
                    rows.append(
                        {
                            "candidate_id": cell.candidate_id,
                            "checkpoint_seed": 1701,
                            "worm_id": worm_id,
                            "control": control,
                            "metric": metric,
                            "normalized_value": value + 0.01 * worm_index,
                            "valid": valid,
                            "minimum_ancestor_fraction": ancestor_fraction,
                        }
                    )
    return pd.DataFrame(rows), worm_ids


def test_joint_max_t_reuses_exact_worm_signs_across_entire_panel() -> None:
    values = np.asarray(
        [
            [0.4, 1.1, -0.3, 2.0, 0.7],
            [-1.5, 0.2, 0.8, -0.4, 1.7],
        ],
        dtype=np.float64,
    )
    p_value, observed_t, null_max, critical, signs = _joint_studentized_max_t(
        values, alpha=0.05, batch_size=3
    )

    assert signs.shape == (2 ** (values.shape[1] - 1), values.shape[1])
    np.testing.assert_array_equal(signs[:, 0], 1)
    assert len(np.unique(signs, axis=0)) == len(signs)
    assert np.all((p_value >= 1.0 / len(signs)) & (p_value <= 1.0))
    assert observed_t.shape == (len(values),)
    assert np.isfinite(critical)

    # Independently recompute every null maximum.  Each row must use the same
    # worm-sign vector within a pattern, not its own per-test randomization.
    manual_max: list[float] = []
    for sign in signs:
        statistics: list[float] = []
        for row in values:
            signed = row * sign
            floor = np.sqrt(np.mean(np.square(row))) * 1e-12
            se = signed.std(ddof=1) / np.sqrt(len(signed))
            statistics.append(abs(float(signed.mean() / max(se, floor))))
        manual_max.append(max(statistics))
    np.testing.assert_allclose(null_max, manual_max, rtol=1e-12, atol=1e-12)


def test_sampling_comparator_is_worst_named_control_after_worm_reduction() -> None:
    cell = _selected_cell("strong")
    contextual, worms = _contextual_fixture((cell,))
    summary, _support, arrays = _summarize(
        contextual,
        (cell,),
        worms,
        AnalysisConfig(bootstrap_replicates=100),
    )
    row = summary.loc[summary.metric == "endpoint_mean"].iloc[0]

    assert row.low_low_mean_magnitude == pytest.approx(0.07)
    assert row.high_high_mean_magnitude == pytest.approx(0.12)
    assert row.midpoint_midpoint_mean_magnitude == pytest.approx(0.92)
    assert row.sampling_null_mean_magnitude == pytest.approx(0.92)
    assert row.sampling_null_mean_magnitude != pytest.approx(
        np.mean(
            [
                row.low_low_mean_magnitude,
                row.high_high_mean_magnitude,
                row.midpoint_midpoint_mean_magnitude,
            ]
        )
    )
    np.testing.assert_allclose(arrays["sampling_null_worm_magnitude"][0], 0.9 + 0.01 * np.arange(5))


def test_labels_require_both_validity_and_genealogy_and_preserve_sensitivity() -> None:
    cells = (
        _selected_cell("strong"),
        _selected_cell("observed_invalid"),
        _selected_cell("pseudo_invalid"),
        _selected_cell("observed_genealogy"),
        _selected_cell("pseudo_genealogy"),
        _selected_cell("sampling_invalid"),
        _selected_cell("sampling_genealogy"),
        _selected_cell("disjoint_observed_failures"),
        _selected_cell("lag_sensitivity", selection_origin="lag_sensitivity"),
    )
    contextual, worms = _contextual_fixture(cells)
    summary, support, _arrays = _summarize(
        contextual,
        cells,
        worms,
        AnalysisConfig(bootstrap_replicates=100),
    )

    labels = summary.groupby("candidate_id").evidence_label.unique()
    assert set(labels["strong"]) != {"sampling_limited"}
    for candidate_id in (
        "observed_invalid",
        "pseudo_invalid",
        "observed_genealogy",
        "pseudo_genealogy",
        "sampling_invalid",
        "sampling_genealogy",
        "disjoint_observed_failures",
        "lag_sensitivity",
    ):
        assert set(labels[candidate_id]) == {"sampling_limited"}

    gate_rows = summary.loc[summary.metric == "endpoint_mean"].set_index("candidate_id")
    assert bool(gate_rows.loc["strong", "support_pass"])
    assert bool(gate_rows.loc["strong", "selection_eligible"])
    assert gate_rows.loc["strong", "gate_reason"] == "none"
    assert not bool(gate_rows.loc["sampling_invalid", "support_pass"])
    assert gate_rows.loc["sampling_invalid", "gate_reason"] == "support_failure"
    assert bool(gate_rows.loc["lag_sensitivity", "support_pass"])
    assert not bool(gate_rows.loc["lag_sensitivity", "selection_eligible"])
    assert gate_rows.loc["lag_sensitivity", "gate_reason"] == "sensitivity_origin"

    indexed = support.set_index("candidate_id")
    assert indexed.loc["observed_invalid", "observed_valid_fraction"] == pytest.approx(0.6)
    assert indexed.loc["pseudo_invalid", "pseudo_valid_fraction"] == pytest.approx(0.6)
    assert indexed.loc[
        "observed_genealogy", "observed_genealogy_valid_fraction_0_10"
    ] == pytest.approx(0.6)
    assert indexed.loc[
        "pseudo_genealogy", "pseudo_genealogy_valid_fraction_0_10"
    ] == pytest.approx(0.6)
    # The sampler gate uses the worst named control rather than a pooled
    # fraction, so a weak midpoint control cannot be diluted by healthy tails.
    assert indexed.loc[
        "sampling_invalid", "sampling_valid_fraction"
    ] == pytest.approx(0.6)
    assert indexed.loc[
        "sampling_genealogy", "sampling_genealogy_valid_fraction_0_10"
    ] == pytest.approx(0.6)
    # Validity and ancestry failures occur in different worms: each marginal
    # fraction is 0.8, but the canonical joint validity fraction is only 0.6.
    assert indexed.loc[
        "disjoint_observed_failures", "observed_valid_fraction"
    ] == pytest.approx(0.8)
    assert indexed.loc[
        "disjoint_observed_failures", "observed_genealogy_valid_fraction_0_10"
    ] == pytest.approx(0.6)


def test_event_extraction_normalizes_each_high_low_replica_by_its_own_gap(
    tmp_path: Path,
) -> None:
    cell = _selected_cell("gap_fixture")
    group = SamplingGroup(
        lag=1,
        horizon=1,
        phases=("baseline",),
        sources=(0,),
        targets=(1,),
        cells=(cell,),
        selection_fingerprint="gap-fixture-fingerprint",
    )
    archive = tmp_path / "fixture.npz"
    observed = np.zeros((1, 1, 3, 2, 2, 1, 2, 1, 1), dtype=np.float32)
    observed[:, :, :, 0, 1, 0, :, 0, 0] = 2.0
    observed[:, :, :, 1, 1, 0, :, 0, 0] = 3.0
    midpoint = np.zeros((1, 1, 3, 2, 1, 2, 1, 1), dtype=np.float32)
    midpoint[:, :, :, 1, 0, :, 0, 0] = 1.0
    pseudo = np.zeros((1, 1, 3, 1, 2, 1, 2, 1, 1), dtype=np.float32)
    pseudo[:, :, :, 0, 1, 0, :, 0, 0] = 1.0
    observed_ancestor = np.broadcast_to(
        np.arange(2, dtype=np.int16), (1, 1, 3, 2, 2, 1, 2)
    ).copy()
    midpoint_ancestor = np.broadcast_to(
        np.arange(2, dtype=np.int16), (1, 1, 3, 2, 1, 2)
    ).copy()
    pseudo_ancestor = np.broadcast_to(
        np.arange(2, dtype=np.int16), (1, 1, 3, 1, 2, 1, 2)
    ).copy()
    observed_gap = np.broadcast_to(
        np.asarray([0.5, 2.0], dtype=np.float32)[None, None, None, :, None],
        (1, 1, 3, 2, 1),
    ).copy()
    diagnostic_components: dict[str, np.ndarray] = {}
    for prefix, shape, achieved_gap, target_gap in (
        ("observed", (1, 1, 3, 2, 1), observed_gap, 1.0),
        ("midpoint", (1, 1, 3, 2, 1), np.zeros((1, 1, 3, 2, 1)), 0.0),
        ("pseudo", (1, 1, 3, 1, 1), np.full((1, 1, 3, 1, 1), 4.0), 1.0),
    ):
        diagnostic_components.update(
            {
                f"{prefix}_diagnostic_ess_low": np.full(shape, 2.0),
                f"{prefix}_diagnostic_ess_high": np.full(shape, 2.0),
                f"{prefix}_diagnostic_max_weight_low": np.full(shape, 0.10),
                f"{prefix}_diagnostic_max_weight_high": np.full(shape, 0.10),
                f"{prefix}_diagnostic_achieved_low": np.zeros(shape),
                f"{prefix}_diagnostic_achieved_high": np.asarray(achieved_gap),
                f"{prefix}_diagnostic_achieved_gap": np.asarray(achieved_gap),
                f"{prefix}_diagnostic_target_low": np.zeros(shape),
                f"{prefix}_diagnostic_target_high": np.full(shape, target_gap),
                f"{prefix}_diagnostic_target_gap": np.full(shape, target_gap),
                f"{prefix}_diagnostic_valid": np.ones(shape, dtype=np.float32),
            }
        )
    np.savez(
        archive,
        selection_fingerprint=np.asarray(group.selection_fingerprint),
        future_parent_index=np.arange(2, dtype=np.int16),
        observed_endpoint_samples=observed,
        midpoint_endpoint_samples=midpoint,
        pseudo_endpoint_samples=pseudo,
        observed_repair_ancestor_id=observed_ancestor,
        midpoint_repair_ancestor_id=midpoint_ancestor,
        pseudo_repair_ancestor_id=pseudo_ancestor,
        chemical_code_by_worm_event=np.asarray([[1, 2, 3]], dtype=np.int8),
        chemical_name_by_worm_event=np.asarray(
            [["butanone", "pentanedione", "nacl"]]
        ),
        seed=np.asarray(1701),
        fold=np.asarray(0),
        worm_ids=np.asarray(["worm_0"]),
        **diagnostic_components,
    )

    frame = _extract_event_rows(
        {"particles": 2, "minimum_effective_sample_size": 1.0},
        (cell,),
        (group,),
        (archive,),
        min_gap=0.10,
    )
    observed_mean = frame.loc[
        (frame.control == "observed")
        & (frame.metric == "endpoint_mean")
        & (frame.event == 0)
    ].sort_values("control_replicate")
    np.testing.assert_allclose(observed_mean.raw_value, [2.0, 3.0])
    np.testing.assert_allclose(observed_mean.normalized_value, [4.0, 1.5])

    high_high = frame.loc[
        (frame.control == "high_high")
        & (frame.metric == "endpoint_mean")
        & (frame.event == 0),
        "normalized_value",
    ]
    midpoint_midpoint = frame.loc[
        (frame.control == "midpoint_midpoint")
        & (frame.metric == "endpoint_mean")
        & (frame.event == 0),
        "normalized_value",
    ]
    quiet_pseudo = frame.loc[
        (frame.control == "quiet_pseudo")
        & (frame.metric == "endpoint_mean")
        & (frame.event == 0),
        "normalized_value",
    ]
    np.testing.assert_allclose(high_high, [2.0])
    np.testing.assert_allclose(midpoint_midpoint, [2.0])
    np.testing.assert_allclose(quiet_pseudo, [0.25])

    with np.load(archive, allow_pickle=False) as saved:
        tampered = {key: saved[key] for key in saved.files}
    tampered["observed_diagnostic_valid"] = np.zeros(
        (1, 1, 3, 2, 1), dtype=np.float32
    )
    np.savez(archive, **tampered)
    with pytest.raises(RuntimeError, match="stored observed diagnostic validity"):
        _extract_event_rows(
            {"particles": 2, "minimum_effective_sample_size": 1.0},
            (cell,),
            (group,),
            (archive,),
            min_gap=0.10,
        )

    tampered["observed_diagnostic_valid"] = np.ones(
        (1, 1, 3, 2, 1), dtype=np.float32
    )
    tampered["observed_diagnostic_achieved_gap"] = (
        tampered["observed_diagnostic_achieved_gap"] + 0.25
    )
    np.savez(archive, **tampered)
    with pytest.raises(RuntimeError, match="achieved-gap diagnostic is inconsistent"):
        _extract_event_rows(
            {"particles": 2, "minimum_effective_sample_size": 1.0},
            (cell,),
            (group,),
            (archive,),
            min_gap=0.10,
        )
