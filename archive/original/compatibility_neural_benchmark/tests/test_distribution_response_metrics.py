from __future__ import annotations

import numpy as np

from compatibility_neural_benchmark.core import (
    ENDPOINT_SD_FLOOR,
    RepairedResponseConfig,
    _endpoint_log_sd_response,
    _weighted_endpoint_wasserstein1,
    estimate_repaired_responses,
)


def test_weighted_endpoint_wasserstein1_matches_hand_integral() -> None:
    # On support [0, 1, 3], the CDF gap is 0.5 over widths 1 and 2.
    future = np.asarray([[[0.0]], [[1.0]], [[3.0]]], dtype=np.float64)
    low = np.asarray([[0.5, 0.5, 0.0]], dtype=np.float64)
    high = np.asarray([[0.0, 0.5, 0.5]], dtype=np.float64)
    result = _weighted_endpoint_wasserstein1(low, high, future, np.asarray([1]))
    np.testing.assert_allclose(result, np.asarray([[[1.5]]]), atol=1e-12)


def test_endpoint_log_sd_response_uses_declared_floor() -> None:
    low = np.asarray([[[0.0, 1.0]]])
    high = np.asarray([[[0.0, 2.0]]])
    result = _endpoint_log_sd_response(low, high)
    expected = np.log(high + ENDPOINT_SD_FLOOR) - np.log(
        low + ENDPOINT_SD_FLOOR
    )
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    assert ENDPOINT_SD_FLOOR == 1e-6


def test_direct_response_records_distributional_metrics_and_floor() -> None:
    rng = np.random.default_rng(91)
    prefix = rng.normal(size=(64, 1, 2)).astype(np.float32)
    future = rng.normal(size=(64, 1, 2)).astype(np.float32)
    target = np.zeros(2, dtype=np.float32)
    result = estimate_repaired_responses(
        prefix,
        future,
        np.zeros((1, 2), dtype=np.float32),
        np.eye(2, dtype=np.float32),
        target,
        target,
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        RepairedResponseConfig(
            history_frames=1,
            repair_frames=1,
            source_window_frames=1,
            horizon_frames=(1,),
            n_particles=64,
        ),
    )
    assert result["response_endpoint_log_sd"].shape == (2, 1, 2)
    assert result["response_endpoint_wasserstein1"].shape == (2, 1, 2)
    np.testing.assert_array_equal(result["response_endpoint_log_sd"], 0.0)
    np.testing.assert_array_equal(result["response_endpoint_wasserstein1"], 0.0)
    np.testing.assert_allclose(
        result["diagnostic_endpoint_sd_floor"], ENDPOINT_SD_FLOOR
    )
