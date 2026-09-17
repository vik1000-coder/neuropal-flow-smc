"""Additional inference coverage: sandwich, cross-target HAC, bootstrap, BH-FDR."""
import numpy as np

from sid_neuromod.inference.bootstrap import bootstrap_ci, moving_block_bootstrap
from sid_neuromod.inference.hac import cross_newey_west, newey_west
from sid_neuromod.inference.multiple_testing import (benjamini_hochberg,
                                                     two_sided_pvalues)
from sid_neuromod.inference.sandwich import (cross_theta_covariance,
                                             readout_covariance,
                                             theta_covariance)
from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher


def _fit(seed=0):
    rng = np.random.default_rng(seed)
    T = 8000
    f = rng.standard_normal(T)
    Y = 0.5 * f + np.sqrt(np.exp(0.3 * f)) * rng.standard_normal(T)
    Psi = np.column_stack([np.ones(T), f - f.mean()])
    return QuadraticScoreMatcher(sigma=0.0, ridge=1e-6).fit(Y, Psi)


def test_theta_covariance_psd():
    res = _fit()
    C = theta_covariance(res, n_lags=10)
    assert np.all(np.linalg.eigvalsh(0.5 * (C + C.T)) > -1e-8)


def test_cross_target_covariance_shape():
    r1 = _fit(0); r2 = _fit(1)
    C = cross_theta_covariance(r1, r2, n_lags=10)
    assert C.shape == (r1.theta.shape[0], r2.theta.shape[0])


def test_readout_covariance_delta_method():
    res = _fit()
    from sid_neuromod.readouts.mean_gain_tail import center_history, d_mean

    center = center_history  # noqa
    Psi_center = np.zeros((1, res.P)); Psi_center[0, 0] = 1.0

    def readout(theta):
        r = res.__class__(theta=theta, P=res.P, sigma=res.sigma, ridge=res.ridge,
                          A=res.A, xi=res.xi, n_samples=res.n_samples)
        return np.array([d_mean(r, Psi_center, 1)[0]])

    r, cov, J = readout_covariance(res, readout, n_lags=10)
    assert r.shape == (1,) and cov.shape == (1, 1)
    assert cov[0, 0] > 0


def test_cross_newey_west_symmetry_with_self():
    rng = np.random.default_rng(4)
    xi = rng.standard_normal((3000, 2))
    S_self = cross_newey_west(xi, xi, n_lags=5)
    S_nw = newey_west(xi, n_lags=5)
    # cross with itself at lag structure equals the standard HAC up to symmetrization
    assert np.allclose(0.5 * (S_self + S_self.T), S_nw, atol=1e-10)


def test_benjamini_hochberg_controls():
    z = np.array([6.0, 5.0, 0.2, -0.1, 0.05])
    p = two_sided_pvalues(z)
    q, rej = benjamini_hochberg(p, alpha=0.05)
    assert rej[0] and rej[1]
    assert not rej[2:].any()
    assert np.all((q >= 0) & (q <= 1))


def test_moving_block_bootstrap():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(2000)
    reps = moving_block_bootstrap(x, np.mean, block=50, n_boot=200, rng=0)
    lo, hi = bootstrap_ci(reps)
    assert lo < np.mean(x) < hi
    assert len(reps) == 200
