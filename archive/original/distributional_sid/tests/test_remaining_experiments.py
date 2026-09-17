from __future__ import annotations

import numpy as np

from distributional_sid.core import CharacteristicBank
from distributional_sid.remaining_experiments import (
    GeometryCell,
    MeasurementCell,
    _ar1_histories,
    _fit_geometry_seed,
    _measurement_oracle,
    _sample_measurement_paths,
)


def test_ar1_histories_have_stationary_scale_and_requested_dependence() -> None:
    rng = np.random.default_rng(912)
    histories = _ar1_histories(rng, 30_000, 3, 0.8)
    assert np.allclose(np.std(histories, axis=0), 1.0, atol=0.04)
    correlation = np.corrcoef(histories[:-1, 0], histories[1:, 0])[0, 1]
    assert abs(correlation - 0.8) < 0.025


def test_measurement_oracle_matches_central_difference() -> None:
    rng = np.random.default_rng(913)
    cell = MeasurementCell(
        "oracle_check",
        rho=0.7,
        latent_amplitude=0.8,
        crosstalk=0.2,
        heteroskedastic_strength=0.6,
        missing_rate=0.25,
        missing_history_coefficient=0.7,
        include_mask=True,
    )
    bank_h = rng.normal(size=(3000, 4))
    bank_y, _, _ = _sample_measurement_paths(bank_h, rng, cell)
    bank = CharacteristicBank.fit(bank_y, 12, seed=9912)
    histories = rng.normal(size=(12, 4))
    _, derivative = _measurement_oracle(histories, bank, cell)
    delta = 1e-5
    plus = histories.copy()
    minus = histories.copy()
    plus[:, 0] += delta
    minus[:, 0] -= delta
    mean_plus, _ = _measurement_oracle(plus, bank, cell)
    mean_minus, _ = _measurement_oracle(minus, bank, cell)
    numerical = (mean_plus - mean_minus) / (2.0 * delta)
    assert np.max(np.abs(derivative - numerical)) < 2e-7


def test_streaming_geometry_fit_is_finite_on_small_cell() -> None:
    cell = GeometryCell(
        "test", history_dim=4, response_dim=4, rank=1,
        n_train=400, rotation=True, source_sparsity="dense", query_count=8,
        seeds=1,
    )
    rows = _fit_geometry_seed(cell, seed=1001, method_init=0)
    assert len(rows) == 3
    for row in rows:
        for key in (
            "feature_tensor_nrmse",
            "covariance_tensor_nrmse",
            "largest_subspace_angle_degrees",
            "source_subspace_angle_degrees",
        ):
            assert np.isfinite(row[key])
