from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.neuron_class_effects import (
    CLASSES,
    PRIMARY,
    align_classes,
    analyze,
    block_values,
    bootstrap_weights,
    edge_mask,
    intervals,
    strong_sources,
)


DIRECT = "direct_importance"
NEURONS = np.asarray(["S1", "I1", "M1", "S2"])
LABELS = np.asarray(["sensory", "interneuron", "motor", "sensory"])


def _classification() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neuron": NEURONS,
            "primary_class": LABELS,
            "mixed_or_disputed": [False, False, True, False],
        }
    )


def _support_arrays() -> tuple[np.ndarray, np.ndarray]:
    valid = np.ones((4, 4), dtype=float)
    genealogy = np.ones_like(valid)
    # Exactly 0.8 passes. Failure at even one lag excludes a source for all lags.
    valid[:, 0] = genealogy[:, 0] = 0.8
    valid[-1, 1] = np.nextafter(0.8, 0.0)
    genealogy[1, 2] = np.nextafter(0.8, 0.0)
    return valid, genealogy


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "canonical"
    root.mkdir()
    classification_path = tmp_path / "classes.csv"
    _classification().iloc[::-1].to_csv(classification_path, index=False)
    lags = np.asarray([1, 4, 8, 16])
    horizons = np.asarray([1, 8])
    worm_ids = np.asarray(["worm-a", "worm-b", "worm-c"])
    shared = dict(
        neurons=NEURONS,
        worm_ids=worm_ids,
        source_lag_frames=lags,
        horizon_frames=horizons,
    )
    atlas = dict(
        **shared,
        orientation=np.asarray("target_row_source_column"),
        methods=np.asarray([PRIMARY, DIRECT]),
        channels=np.asarray(["endpoint_mean", "endpoint_wasserstein1"]),
        contexts=np.asarray(["baseline", "onset_minus_baseline"]),
        source_lag_seconds=lags * 0.25,
        horizon_seconds=horizons * 0.25,
    )
    worms = dict(**shared, primary_method=np.asarray(PRIMARY))
    valid, genealogy = _support_arrays()
    # [lag, worm, horizon, target, source], deliberately asymmetric.
    effect = np.broadcast_to(
        10 * np.arange(4)[:, None] + np.arange(4)[None, :], (4, 3, 2, 4, 4)
    ).astype(float).copy()
    effect[..., 1, 0] = np.asarray([4.0, -4.0, 0.0])[None, :, None]
    effect[..., 1, 3] = 2.0
    for context in atlas["contexts"]:
        for method in atlas["methods"]:
            atlas[f"valid_fraction__{method}__{context}"] = valid
            if method == PRIMARY:
                atlas[f"genealogy_valid_fraction_0_10__{method}__{context}"] = genealogy
        for channel in atlas["channels"]:
            array = effect.copy()
            if channel == "endpoint_wasserstein1":
                array = np.abs(array) + 0.25
            if context.endswith("_minus_baseline"):
                array *= -1
            worms[f"normalized__{channel}__{context}"] = array
            atlas[f"mean_normalized__{PRIMARY}__{channel}__{context}"] = array.mean(axis=1)
            atlas[f"mean_normalized__{DIRECT}__{channel}__{context}"] = array.mean(axis=1) + 100
    np.savez_compressed(root / "atlas_matrices.npz", **atlas)
    np.savez_compressed(root / "worm_matrices.npz", **worms)
    (root / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ("atlas_matrices.npz", "worm_matrices.npz")
        )
    )
    return root, classification_path


def test_edge_masks_use_target_rows_source_columns_and_are_asymmetric() -> None:
    all_nodes = np.ones(4, dtype=bool)
    mask = edge_mask(LABELS, "sensory", "interneuron", all_nodes, all_nodes)
    expected = np.zeros((4, 4), dtype=bool)
    expected[1, [0, 3]] = True
    np.testing.assert_array_equal(mask, expected)
    reverse = edge_mask(LABELS, "interneuron", "sensory", all_nodes, all_nodes)
    np.testing.assert_array_equal(reverse, expected.T)
    matrix = 10 * np.arange(4)[:, None] + np.arange(4)[None, :]
    assert block_values(matrix, mask)[0] == pytest.approx(11.5)
    assert block_values(matrix, reverse)[0] == pytest.approx(16.0)


@pytest.mark.parametrize("included", [np.ones(4, bool), np.asarray([True, True, False, True])])
def test_nine_class_masks_partition_all_offdiagonal_pairs(included: np.ndarray) -> None:
    masks = [
        edge_mask(LABELS, source, target, included, np.ones(4, bool))
        for source in CLASSES
        for target in CLASSES
    ]
    combined = np.sum(masks, axis=0)
    expected = included[:, None] & included[None, :]
    np.fill_diagonal(expected, False)
    np.testing.assert_array_equal(combined, expected.astype(int))
    n = int(included.sum())
    assert combined.sum() == n * (n - 1)
    assert not np.diag(combined).any()


def test_support_gates_sources_only_not_target_coordinates() -> None:
    source_support = np.asarray([True, False, False, True])
    mask = edge_mask(LABELS, "sensory", "interneuron", np.ones(4, bool), source_support)
    assert mask[1, 0] and mask[1, 3]  # I1 remains a target despite failed source support.
    reverse = edge_mask(LABELS, "interneuron", "sensory", np.ones(4, bool), source_support)
    assert not reverse.any()


def test_strong_support_is_fixed_across_all_lags_and_uses_ancestry_only_for_smc() -> None:
    valid, genealogy = _support_arrays()
    atlas = {
        f"valid_fraction__{PRIMARY}__baseline": valid,
        f"genealogy_valid_fraction_0_10__{PRIMARY}__baseline": genealogy,
        f"valid_fraction__{DIRECT}__baseline": valid,
    }
    np.testing.assert_array_equal(strong_sources(atlas, PRIMARY, "baseline"), [True, False, False, True])
    # No genealogy array is provided for direct importance; it must not borrow one.
    np.testing.assert_array_equal(strong_sources(atlas, DIRECT, "baseline"), [True, False, True, True])
    valid[0, 3] = np.nan
    assert not strong_sources(atlas, PRIMARY, "baseline")[3]


def test_classification_is_exactly_reordered_and_mixed_flags_are_boolean() -> None:
    frame = _classification().iloc[::-1].copy()
    frame["mixed_or_disputed"] = ["FALSE", "True", "false", "FALSE"]
    result = align_classes(frame, NEURONS)
    assert result.neuron.tolist() == NEURONS.tolist()
    assert result.primary_class.tolist() == LABELS.tolist()
    assert result.mixed_or_disputed.tolist() == [False, False, True, False]
    assert result.mixed_or_disputed.dtype == bool


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "exact atlas vocabulary"),
        ("extra", "exact atlas vocabulary"),
        ("duplicate", "Duplicate neuron classification"),
        ("unknown_class", "Unexpected primary class"),
        ("unknown_flag", "Mixed-role flags"),
    ],
)
def test_classification_rejects_missing_extra_duplicate_and_unknown_rows(failure: str, message: str) -> None:
    frame = _classification()
    if failure == "missing":
        frame = frame.iloc[:-1]
    elif failure in {"extra", "duplicate"}:
        extra = frame.iloc[[0]].copy()
        if failure == "extra":
            extra.loc[:, "neuron"] = "EXTRA"
        frame = pd.concat([frame, extra], ignore_index=True)
    elif failure == "unknown_class":
        frame.loc[0, "primary_class"] = "unknown"
    else:
        frame["mixed_or_disputed"] = frame.mixed_or_disputed.astype(object)
        frame.loc[0, "mixed_or_disputed"] = "maybe"
    with pytest.raises(ValueError, match=message):
        align_classes(frame, NEURONS)


def test_signed_pair_mean_and_mean_absolute_pair_effect_do_not_conflate_cancellation() -> None:
    mask = np.asarray([[False, True], [True, False]])
    matrix = np.asarray([[0.0, -3.0], [3.0, 0.0]])
    signed, magnitude = block_values(matrix, mask)
    assert signed == 0
    assert magnitude == 3
    assert magnitude != abs(signed)


def test_pooled_edge_magnitude_differs_from_absolute_effect_within_worms() -> None:
    mask = np.asarray([[False, True], [True, False]])
    per_worm = np.asarray([[[0.0, -3.0], [3.0, 0.0]], [[0.0, 3.0], [-3.0, 0.0]]])
    _, pooled_magnitude = block_values(per_worm.mean(axis=0), mask)
    _, within_worm_magnitude = block_values(per_worm, mask)
    assert pooled_magnitude == 0
    np.testing.assert_array_equal(within_worm_magnitude, [3, 3])


def test_empty_blocks_keep_leading_axes_and_selected_nonfinite_values_fail() -> None:
    matrix = np.zeros((4, 3, 2, 4, 4))
    mask = np.zeros((4, 4), dtype=bool)
    signed, magnitude = block_values(matrix, mask)
    assert signed.shape == magnitude.shape == (4, 3, 2)
    assert np.isnan(signed).all() and np.isnan(magnitude).all()
    matrix[..., 0, 1] = np.nan
    mask[1, 0] = True
    assert np.isfinite(block_values(matrix, mask)[0]).all()  # An unselected NaN is harmless.
    mask[0, 1] = True
    with pytest.raises(ValueError, match="Nonfinite effect in eligible block"):
        block_values(matrix, mask)


def test_bootstrap_weights_resample_whole_worms_and_are_reproducible() -> None:
    weights = bootstrap_weights(3, 25, 101)
    assert weights.shape == (25, 3)
    np.testing.assert_array_equal(weights, bootstrap_weights(3, 25, 101))
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    np.testing.assert_allclose(weights * 3, np.rint(weights * 3))
    assert np.all(weights >= 0)
    assert not np.array_equal(weights, bootstrap_weights(3, 25, 102))


def test_intervals_apply_the_same_worm_draw_to_every_lag_and_horizon() -> None:
    values = np.asarray([[[0, 10], [2, 12], [8, 18]], [[100, 110], [102, 112], [108, 118]]], dtype=float)
    weights = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]], dtype=float)
    got = intervals(values, weights)
    expected = np.empty((2, 2, 2))
    for lag in range(2):
        for horizon in range(2):
            whole_worm_means = [np.dot(weight, values[lag, :, horizon]) for weight in weights]
            expected[:, lag, horizon] = np.quantile(whole_worm_means, [0.025, 0.975])
    np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(got[:, 1, :] - got[:, 0, :], 100)
    np.testing.assert_allclose(got[:, :, 1] - got[:, :, 0], 10)


def test_constant_worm_values_have_degenerate_correct_confidence_intervals() -> None:
    values = np.broadcast_to(np.asarray([[2.0, -5.0], [7.0, 13.0]])[:, None, :], (2, 3, 2))
    got = intervals(values, bootstrap_weights(3, 100, 21))
    np.testing.assert_allclose(got[0], values[:, 0, :])
    np.testing.assert_allclose(got[1], values[:, 0, :])


def test_analysis_preserves_denominators_support_cancellation_and_uncertainty_scope(tmp_path: Path) -> None:
    root, classification = _write_fixture(tmp_path)
    output = tmp_path / "class_results"
    metadata = analyze(root, classification, output, repetitions=100, seed=91)
    frame = pd.read_csv(output / "class_effects.csv")
    denominator = pd.read_csv(output / "class_denominators.csv")
    assert metadata["row_count"] == len(frame) == 2304
    assert metadata["max_worm_mean_reconstruction_error"] == 0
    assert metadata["max_pair_weighted_group_reconstruction_error"] < 1e-12
    assert denominator.groupby("roster").possible_directed_pairs.sum().to_dict() == {
        "cook_primary": 12,
        "exclude_mixed_disputed": 6,
    }
    rows = frame[(frame.method == PRIMARY) & (frame.roster == "cook_primary") &
                 (frame.support_policy == "strong_common_lags") & (frame.channel == "endpoint_mean") &
                 (frame.context == "baseline") & (frame.source_class == "sensory") &
                 (frame.target_class == "interneuron")]
    assert len(rows) == 8
    assert rows.n_pairs.eq(2).all() and rows.n_sources.eq(2).all()
    assert rows.n_targets.eq(1).all()  # I1 failed source support, not target inclusion.
    np.testing.assert_allclose(rows.signed_mean, 1)
    np.testing.assert_allclose(rows.mean_absolute_pooled_edge, 1)
    np.testing.assert_allclose(rows.mean_absolute_within_worm, 7 / 3)
    reverse = frame[(frame.method == PRIMARY) & (frame.support_policy == "strong_common_lags") &
                    (frame.source_class == "interneuron")]
    assert reverse.n_pairs.eq(0).all()
    assert reverse.status.eq("no_eligible_pairs").all()
    assert reverse.signed_mean.isna().all()
    direct = frame[frame.method == DIRECT]
    assert direct.uncertainty_status.eq("not_available").all()
    assert direct.signed_ci_low.isna().all() and direct.signed_ci_high.isna().all()
    assert direct.within_worm_abs_ci_low.isna().all()
    assert not frame[(frame.channel == "endpoint_wasserstein1") & (frame.context == "baseline")].effect_is_signed.any()
    assert frame[(frame.channel == "endpoint_wasserstein1") & (frame.context == "onset_minus_baseline")].effect_is_signed.all()
    with np.load(output / "class_worm_effects.npz", allow_pickle=False) as arrays:
        key = "cook_primary__strong_common_lags__endpoint_mean__baseline__sensory_to_interneuron"
        signed = arrays[f"signed__{key}"]
        assert signed.shape == (4, 3, 2)
        np.testing.assert_allclose(signed[0, :, 0], [3, -1, 1])
        np.testing.assert_allclose(arrays[f"mean_absolute__{key}"][0, :, 0], [3, 3, 1])
        expected_ci = intervals(signed, bootstrap_weights(3, 100, 91))
        np.testing.assert_allclose(rows.signed_ci_low.to_numpy().reshape(4, 2), expected_ci[0])
        np.testing.assert_allclose(rows.signed_ci_high.to_numpy().reshape(4, 2), expected_ci[1])
    with pytest.raises(FileExistsError, match="existing results are immutable"):
        analyze(root, classification, output, repetitions=100, seed=91)
