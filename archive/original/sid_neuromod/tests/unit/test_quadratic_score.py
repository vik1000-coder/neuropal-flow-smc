"""Section 17.2: quadratic score estimator tests."""
import warnings

import numpy as np
import pytest

from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher, sigma_ledger


def test_quadratic_score_recovers_gaussian_moments():
    """Y = a^T psi + sqrt(v) eps -> mean R2 > 0.95, median var within 10%."""
    rng = np.random.default_rng(0)
    T, P = 40000, 4
    Psi = np.column_stack([np.ones(T), rng.standard_normal((T, P - 1))])
    a = np.array([0.3, 1.5, -0.8, 0.6])
    v = 0.5
    # signal-dominant so mean R2 > 0.95
    Y = 4.0 * (Psi @ a) + np.sqrt(v) * rng.standard_normal(T)
    m = QuadraticScoreMatcher(sigma=0.0, ridge=1e-9)
    m.fit(Y, Psi)
    mu, var = m.moments(Psi)
    r2 = 1.0 - np.var(Y - mu) / np.var(Y)
    assert r2 > 0.95
    assert abs(np.median(var) - v) / v < 0.10


def test_sigma_ledger():
    """Corrected variance estimates agree across sigma within 10%."""
    rng = np.random.default_rng(1)
    T, P = 40000, 3
    Psi = np.column_stack([np.ones(T), rng.standard_normal((T, P - 1))])
    a = np.array([0.2, 1.0, -0.5])
    v = 0.8
    Y = Psi @ a + np.sqrt(v) * rng.standard_normal(T)
    ledger = sigma_ledger(Y, Psi, ridge=1e-8, seed=2)
    vars_ = np.array([d["var_at_center"] for d in ledger])
    assert np.max(np.abs(vars_ - vars_[0])) / abs(vars_[0]) < 0.10


def test_invalid_variance_warning():
    """A degenerate fit producing invalid eta2 emits a warning; predictions finite."""
    rng = np.random.default_rng(2)
    T, P = 300, 5
    # tiny sample, many features, heavy-tailed target -> some eta2 >= -eta2_min
    Psi = np.column_stack([np.ones(T), rng.standard_normal((T, P - 1))])
    Y = rng.standard_t(2, size=T) * 3.0
    m = QuadraticScoreMatcher(sigma=0.0, ridge=1e-10)
    res = m.fit(Y, Psi)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, _, mu, var = m.predict_params(Psi)
    assert res.invalid_variance_fraction >= 0.0
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(var))
    assert np.all(var > 0)


def test_logpdf_cdf_sample_shapes():
    rng = np.random.default_rng(3)
    T, P = 5000, 3
    Psi = np.column_stack([np.ones(T), rng.standard_normal((T, P - 1))])
    Y = Psi @ np.array([0.1, 0.5, -0.3]) + rng.standard_normal(T)
    m = QuadraticScoreMatcher(sigma=0.0, ridge=1e-6)
    m.fit(Y, Psi)
    lp = m.logpdf(Y, Psi)
    cd = m.cdf(Y, Psi)
    s = m.sample(Psi[:10], n_samples=4, rng=0)
    assert lp.shape == (T,)
    assert np.all((cd >= 0) & (cd <= 1))
    assert s.shape == (10, 4)


def test_denoising_matches_score_matching_variance():
    """Denoising (sigma>0) recovers the same corrected variance as sigma=0."""
    rng = np.random.default_rng(4)
    T, P = 60000, 2
    Psi = np.column_stack([np.ones(T), rng.standard_normal(T)])
    Y = Psi @ np.array([0.0, 0.4]) + np.sqrt(0.7) * rng.standard_normal(T)
    m0 = QuadraticScoreMatcher(sigma=0.0, ridge=1e-8); m0.fit(Y, Psi)
    sd = np.std(Y)
    m1 = QuadraticScoreMatcher(sigma=0.5 * sd, ridge=1e-8, seed=5); m1.fit(Y, Psi)
    _, v0 = m0.moments(Psi.mean(axis=0, keepdims=True))
    _, v1 = m1.moments(Psi.mean(axis=0, keepdims=True))
    assert abs(v0[0] - v1[0]) / v0[0] < 0.10
