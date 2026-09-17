"""Section 17.4: HAC / inference tests."""
import numpy as np

from sid_neuromod.inference.hac import newey_west
from sid_neuromod.inference.simultaneous_bands import simultaneous_bands
from sid_neuromod.utils.linalg import min_eig


def test_hac_white_noise_matches_sample_covariance():
    """HAC with lag 0 equals the sample covariance."""
    rng = np.random.default_rng(0)
    xi = rng.standard_normal((5000, 3))
    S0 = newey_west(xi, n_lags=0)
    xc = xi - xi.mean(axis=0)
    samp = (xc.T @ xc) / xi.shape[0]
    assert np.allclose(S0, samp, atol=1e-12)


def test_hac_psd_or_near_psd():
    """Bartlett-kernel HAC is PSD (min eigenvalue > -1e-8)."""
    rng = np.random.default_rng(1)
    # autocorrelated estimating functions
    e = rng.standard_normal((4000, 2))
    xi = np.copy(e)
    for t in range(1, xi.shape[0]):
        xi[t] = 0.6 * xi[t - 1] + e[t]
    S = newey_west(xi, n_lags=20)
    assert min_eig(S) > -1e-8


def test_simultaneous_band_coverage_independent_gaussian():
    """MC coverage within +-3% of nominal for iid Gaussian readouts."""
    rng = np.random.default_rng(2)
    m = 8
    true = np.zeros(m)
    C = np.eye(m) * 0.04  # se = 0.2
    n_trials = 1500
    covered = 0
    for _ in range(n_trials):
        r = true + rng.standard_normal(m) * 0.2
        band = simultaneous_bands(r, C, alpha=0.05, n_mc=4000, rng=rng)
        if np.all((band.ci_low <= true) & (true <= band.ci_high)):
            covered += 1
    cov = covered / n_trials
    assert abs(cov - 0.95) < 0.03


def test_default_hac_lags_grows():
    from sid_neuromod.inference.hac import default_hac_lags
    assert default_hac_lags(100) >= 20
    assert default_hac_lags(10_000_000) > default_hac_lags(1000)
