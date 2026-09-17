from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from distributional_sid.core import (
    AnalyticGaussianPathDGP,
    CharacteristicBank,
    HistorySieve,
    embargoed_block_splits,
    ridge_coefficients,
    riesz_coefficients,
)


def test_characteristic_bank_reproducible() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(size=(200, 3))
    a = CharacteristicBank.fit(y, q=16, seed=7)
    b = CharacteristicBank.fit(y, q=16, seed=7)
    assert a.digest() == b.digest()
    np.testing.assert_array_equal(a.transform(y), b.transform(y))


def test_history_sieve_derivative_matches_finite_difference() -> None:
    rng = np.random.default_rng(3)
    h = rng.normal(size=(300, 5))
    sieve = HistorySieve(5, 24, 11).fit(h)
    point = h[:12].copy()
    analytic = sieve.derivative(point, 2)
    delta = 1e-5
    plus = point.copy()
    minus = point.copy()
    plus[:, 2] += delta
    minus[:, 2] -= delta
    finite = (sieve.transform(plus) - sieve.transform(minus)) / (2.0 * delta)
    np.testing.assert_allclose(analytic, finite, rtol=2e-8, atol=2e-9)


def test_gaussian_characteristic_derivative_matches_finite_difference() -> None:
    dgp = AnalyticGaussianPathDGP(seed=4, history_dim=5, response_dim=3)
    rng = np.random.default_rng(8)
    h_bank = dgp.sample_histories(rng, 500)
    y_bank = dgp.sample_responses(h_bank, rng)
    bank = CharacteristicBank.fit(y_bank, 20, 19)
    h = dgp.sample_histories(rng, 32)
    analytic = dgp.conditional_feature_derivative(h, 1, bank)
    delta = 1e-5
    plus = h.copy()
    minus = h.copy()
    plus[:, 1] += delta
    minus[:, 1] -= delta
    finite = (
        dgp.conditional_feature_mean(plus, bank) - dgp.conditional_feature_mean(minus, bank)
    ) / (2.0 * delta)
    np.testing.assert_allclose(analytic, finite, rtol=2e-7, atol=2e-8)


def test_gaussian_riesz_sieve_recovers_linear_identity() -> None:
    rng = np.random.default_rng(5)
    h = rng.normal(size=(30_000, 4))
    sieve = HistorySieve(4, 32, 17).fit(h)
    z = sieve.transform(h)
    dz = sieve.derivative(h, 2)
    coefficient = riesz_coefficients(z, dz, ridge=1e-6)
    alpha = z @ coefficient
    assert abs(np.mean(alpha)) < 0.02
    moments = np.mean(alpha[:, None] * h, axis=0)
    np.testing.assert_allclose(moments, np.eye(4)[2], atol=0.025)


def test_orthogonal_identity_with_oracle_nuisances() -> None:
    rng = np.random.default_rng(6)
    dgp = AnalyticGaussianPathDGP(seed=6, history_dim=4, response_dim=2)
    h_bank = dgp.sample_histories(rng, 1000)
    y_bank = dgp.sample_responses(h_bank, rng)
    bank = CharacteristicBank.fit(y_bank, 12, 31)
    h = dgp.sample_histories(rng, 100_000)
    y = dgp.sample_responses(h, rng)
    phi = bank.transform(y)
    m = dgp.conditional_feature_mean(h, bank)
    dm = dgp.conditional_feature_derivative(h, 0, bank)
    alpha = dgp.oracle_riesz(h, 0)
    psi = dm + alpha[:, None] * (phi - m)
    error = np.mean(psi, axis=0) - np.mean(dm, axis=0)
    standard_error = np.std(psi - dm, axis=0, ddof=1) / np.sqrt(h.shape[0])
    assert np.all(np.abs(error) <= 4.0 * standard_error + 1e-4)


def test_embargo_splits_exclude_window_footprint() -> None:
    footprint = 12
    for train, test in embargoed_block_splits(1000, 5, footprint):
        expanded = set()
        for index in test:
            expanded.update(range(max(0, index - footprint), min(1000, index + footprint + 1)))
        assert not expanded.intersection(set(train.tolist()))


def test_uniform_boundary_identity_requires_vanishing_weight() -> None:
    grid = np.linspace(-1.0, 1.0, 1_000_001)
    g = np.exp(grid)
    dg = np.exp(grid)
    invalid_lhs = np.trapezoid(np.zeros_like(grid) * g, grid) / 2.0
    invalid_rhs = np.trapezoid(dg, grid) / 2.0
    assert abs(invalid_lhs - invalid_rhs) > 0.5
    b = 1.0 - grid**2
    alpha = 2.0 * grid
    valid_lhs = np.trapezoid(alpha * g, grid) / 2.0
    valid_rhs = np.trapezoid(b * dg, grid) / 2.0
    assert abs(valid_lhs - valid_rhs) < 1e-10
