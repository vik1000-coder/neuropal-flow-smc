"""Section 17.3: readout tests."""
import numpy as np

from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher
from sid_neuromod.readouts.covariance import covariance_readout
from sid_neuromod.readouts.mean_gain_tail import (center_history, d_tail_high,
                                                  readout_at_center)


def test_mean_readout_matches_known_linear_coefficient():
    """Homoscedastic Gaussian Y = beta f + eps -> D_mean ~ beta, D_gain ~ 0."""
    rng = np.random.default_rng(0)
    T = 60000
    f = rng.standard_normal(T)
    beta = 0.7
    Y = beta * f + np.sqrt(0.5) * rng.standard_normal(T)
    Psi = np.column_stack([np.ones(T), f - f.mean()])
    m = QuadraticScoreMatcher(sigma=0.0, ridge=1e-9)
    res = m.fit(Y, Psi)
    d_mean = readout_at_center(res, Psi, 1, "mean")
    d_gain = readout_at_center(res, Psi, 1, "gain_log_variance")
    assert abs(d_mean - beta) < 0.02
    assert abs(d_gain) < 0.02


def test_gain_readout_matches_known_log_variance_coefficient():
    """Y ~ N(0, exp(c f)) -> D_gain ~ c, D_mean ~ 0."""
    rng = np.random.default_rng(1)
    T = 200000
    f = rng.standard_normal(T)
    c = 0.5
    Y = np.sqrt(np.exp(c * f)) * rng.standard_normal(T)
    Psi = np.column_stack([np.ones(T), f - f.mean()])
    m = QuadraticScoreMatcher(sigma=0.0, ridge=1e-8)
    res = m.fit(Y, Psi)
    d_gain = readout_at_center(res, Psi, 1, "gain_log_variance")
    d_mean = readout_at_center(res, Psi, 1, "mean")
    # projection/local-linearization bias makes the estimate low but same sign/order
    assert 0.3 < d_gain < 0.7
    assert abs(d_mean) < 0.03


def test_tail_derivative_finite_difference():
    """Analytic high-tail derivative matches finite difference (rel err < 1e-4)."""
    rng = np.random.default_rng(2)
    T = 40000
    f = rng.standard_normal(T)
    Y = 0.4 * f + np.sqrt(np.exp(0.3 * f)) * rng.standard_normal(T)
    Psi = np.column_stack([np.ones(T), f - f.mean()])
    m = QuadraticScoreMatcher(sigma=0.0, ridge=1e-8)
    res = m.fit(Y, Psi)
    q = float(np.quantile(Y, 0.9))
    center = center_history(Psi)
    ana = d_tail_high(res, center, 1, q)[0]

    # finite difference of P(Y>q) w.r.t. feature 1 at the center history
    from scipy.stats import norm
    from sid_neuromod.readouts.mean_gain_tail import _eff_params
    eps = 1e-5
    cp = center.copy(); cp[0, 1] += eps
    cm = center.copy(); cm[0, 1] -= eps
    _, _, _, vp, mup = _eff_params(res, cp)
    _, _, _, vm, mum = _eff_params(res, cm)
    pp = 1 - norm.cdf((q - mup[0]) / np.sqrt(vp[0]))
    pm = 1 - norm.cdf((q - mum[0]) / np.sqrt(vm[0]))
    fd = (pp - pm) / (2 * eps)
    assert abs(ana - fd) / (abs(fd) + 1e-12) < 1e-4


def test_covariance_readout_matches_analytic_mean():
    """Derivative-free covariance route ~ analytic mean readout (Stein shift)."""
    rng = np.random.default_rng(3)
    T = 40000
    f = rng.standard_normal(T)
    Y = 0.6 * f + np.sqrt(0.4) * rng.standard_normal(T)
    Psi = np.column_stack([np.ones(T), f - f.mean()])
    m = QuadraticScoreMatcher(sigma=0.0, ridge=1e-9)
    res = m.fit(Y, Psi)
    center = center_history(Psi)
    ana = readout_at_center(res, Psi, 1, "mean")
    cov = covariance_readout(res, center, 1, phi="mean", M=20000, rng=0)
    assert abs(ana - cov) < 0.02
