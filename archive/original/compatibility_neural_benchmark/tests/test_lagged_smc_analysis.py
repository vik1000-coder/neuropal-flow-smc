from __future__ import annotations

import numpy as np

from compatibility_neural_benchmark.lagged_smc_analysis import (
    _fit_scalar_forecast,
    build_vectors,
)


def test_build_vectors_uses_lagged_source_and_postcut_target() -> None:
    trace = np.arange(80, dtype=np.float64)[:, None]
    cuts = np.asarray([[[20]]])
    source, current, target = build_vectors(
        (trace,), cuts, source_lag=4, horizons=np.asarray([1, 4]), source_window=4
    )
    # Source is frames 13:17 against frames 9:13; current is 17:21 against 13:17.
    assert np.isclose(source[0, 0, 0, 0], 4.0)
    assert np.isclose(current[0, 0, 0, 0], 4.0)
    assert np.isclose(target[0, 0, 0, 0, 0], 6.5)
    assert np.isclose(target[0, 0, 0, 1, 0], 8.0)


def test_scalar_forecast_can_use_propagated_feature() -> None:
    rng = np.random.default_rng(7)
    worms, events, neurons = 7, 3, 12
    source = rng.normal(size=(worms, events, neurons))
    current = rng.normal(size=(worms, events, neurons))
    propagated = rng.normal(size=(worms, events, neurons))
    target = 0.2 * current - 0.1 * source + 1.5 * propagated
    training = np.arange(1, worms)
    mask = np.ones(neurons, dtype=bool)
    base, full, outcome, beta = _fit_scalar_forecast(
        source, current, propagated, target, training, 0, mask
    )
    base_mse = np.mean((base - outcome) ** 2)
    full_mse = np.mean((full - outcome) ** 2)
    assert full_mse < 1e-20
    assert full_mse < base_mse
    assert np.isclose(beta[3], 1.5)
