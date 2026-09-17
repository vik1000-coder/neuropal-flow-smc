from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.models import build_head

from query_ood_robustness.core import (
    HistorySupportModel,
    construct_matched_query_pair,
    direct_importance_sampling,
    effective_sample_size,
    normalize_log_weights,
    progressive_bridge_smc,
    systematic_resample,
)
from query_ood_robustness.pseudo_ood_runner import _temporal_exclusion
from query_ood_robustness.repaired_path_runner import _sensitivity_queries


def test_weights_and_ess() -> None:
    weights, _ = normalize_log_weights(np.zeros(8))
    assert np.isclose(weights.sum(), 1)
    assert effective_sample_size(weights) == 8
    assert effective_sample_size(np.array([1.0, 0.0])) == 1


def test_systematic_resampling_is_seeded_and_in_range() -> None:
    weights = np.array([0.05, 0.10, 0.15, 0.70])
    one = systematic_resample(weights, np.random.default_rng(41))
    two = systematic_resample(weights, np.random.default_rng(41))
    np.testing.assert_array_equal(one, two)
    assert one.min() >= 0 and one.max() < len(weights)


def test_matched_query_pair_invariants_and_immutability() -> None:
    rng = np.random.default_rng(7)
    history = rng.normal(size=(20, 5))
    stimulus = rng.integers(0, 2, size=(20, 1))
    before = history.copy()
    pair = construct_matched_query_pair(
        history, stimulus, source=2, target_value=history[-1, 2] + 1.5
    )
    np.testing.assert_array_equal(history, before)
    np.testing.assert_array_equal(pair.stimulus, stimulus)
    assert pair.coherent[-1, 2] == pair.incoherent[-1, 2]
    assert np.isclose(pair.coherent_norm, pair.incoherent_norm)
    repeated = construct_matched_query_pair(
        history, stimulus, source=2, target_value=history[-1, 2] + 1.5
    )
    np.testing.assert_array_equal(pair.coherent, repeated.coherent)
    np.testing.assert_array_equal(pair.incoherent, repeated.incoherent)


def test_progressive_bridge_gaussian_conditional_mean() -> None:
    correlation = 0.8
    covariance = np.array([[1.0, correlation], [correlation, 1.0]])
    chol = np.linalg.cholesky(covariance)
    target = 2.5
    bandwidth = 0.25

    def sample(n: int, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(n, 2)) @ chol.T

    def potential(value: np.ndarray) -> np.ndarray:
        return -0.5 * np.square((value[:, 0] - target) / bandwidth)

    analytic = correlation * target / (1.0 + bandwidth**2)
    errors = []
    for seed in range(4):
        result = progressive_bridge_smc(
            sample,
            potential,
            lambda value: value[:, 1],
            n_particles=1024,
            seed=seed + 100,
            rejuvenation_steps=3,
        )
        errors.append(abs(result.estimate - analytic))
        assert result.finite
        assert result.stages.beta_end.iloc[-1] == 1.0
        assert np.all(result.stages.ess > 0)
    assert np.mean(errors) < 0.12


def test_direct_importance_mild_query() -> None:
    def sample(n: int, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(n, 2))

    result = direct_importance_sampling(
        sample,
        lambda value: -0.5 * np.square(value[:, 0] / 2.0),
        lambda value: value[:, 1],
        n_particles=2048,
        seed=19,
    )
    assert abs(result.estimate) < 0.08
    assert 0 < result.final_ess_fraction <= 1


def test_support_model_penalizes_matched_terminal_spike() -> None:
    rng = np.random.default_rng(18)
    rows, length, neurons = 240, 20, 5
    histories = np.empty((rows, length, neurons))
    histories[:, 0] = rng.normal(scale=0.3, size=(rows, neurons))
    for time in range(1, length):
        histories[:, time] = 0.92 * histories[:, time - 1] + rng.normal(
            scale=0.08, size=(rows, neurons)
        )
    stimulus = np.zeros((rows, length, 1))
    model = HistorySupportModel(k=5, pca_cap=8, seed=31).fit(
        histories[:160], stimulus[:160], histories[160:200], stimulus[160:200]
    )
    pair = construct_matched_query_pair(
        histories[205], stimulus[205], source=1,
        target_value=histories[205, -1, 1] + 1.0,
    )
    scores = model.score(
        np.stack([pair.coherent, pair.incoherent]),
        np.stack([pair.stimulus, pair.stimulus]),
        source=1,
    )
    assert scores.iloc[1].curvature_percentile >= scores.iloc[0].curvature_percentile
    assert scores.iloc[1].history_support_value <= scores.iloc[0].history_support_value


def test_support_value_calibration_is_bounded() -> None:
    rng = np.random.default_rng(3)
    histories = rng.normal(size=(80, 12, 3))
    stimulus = np.zeros((80, 12, 1))
    model = HistorySupportModel(k=4, pca_cap=5).fit(
        histories[:50], stimulus[:50], histories[50:65], stimulus[50:65]
    )
    score = model.score(histories[65:], stimulus[65:])
    assert score.history_support_value.between(0, 1).all()
    assert score.nearest_history_distance.gt(0).all()


def test_temporal_exclusion_removes_overlapping_windows() -> None:
    class MinimalWindows:
        target = np.zeros((10, 1))
        worm = np.asarray([0] * 5 + [1] * 5)
        time = np.asarray([0, 20, 40, 100, 200, 0, 20, 40, 100, 200])

    retained = _temporal_exclusion(MinimalWindows(), np.asarray([2, 7]), radius=30)
    np.testing.assert_array_equal(retained, np.asarray([0, 3, 4, 5, 8, 9]))


def test_repaired_path_sensitivity_selection_spans_support() -> None:
    frame = pd.DataFrame({
        "query_class": ["supported_history_common_event"] * 17,
        "history_support_value": np.linspace(0.01, 0.99, 17),
        "fold": np.arange(17) % 5,
        "worm_id": [f"w{index}" for index in range(17)],
    })
    selected = _sensitivity_queries(frame, 7)
    assert len(selected) == 7
    assert selected.history_support_value.iloc[0] == frame.history_support_value.min()
    assert selected.history_support_value.iloc[-1] == frame.history_support_value.max()


def test_mdn_fixed_order_sampling_is_reproducible() -> None:
    model = build_head(
        "autoregressive_mdn", q=5, dy=4,
        params={"hidden": 8, "layers": 1, "components": 4, "permutation_seed": 101},
    ).eval()
    history = torch.zeros((3, 5))
    one = model.sample(history, 7, seed=41)
    two = model.sample(history, 7, seed=41)
    torch.testing.assert_close(one, two, rtol=0, atol=0)
    assert one.shape == (3, 7, 4)
    assert sorted(model.coordinate_permutation.tolist()) == [0, 1, 2, 3]
