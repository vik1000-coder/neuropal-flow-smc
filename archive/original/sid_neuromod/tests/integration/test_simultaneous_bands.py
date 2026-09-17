"""Section 18 / E6: simultaneous sup-t bands coverage and correlation handling."""
import numpy as np

from sid_neuromod.inference.simultaneous_bands import simultaneous_bands, sup_t_critical


def test_sup_t_critical_grows_with_dimension():
    """The sup-t critical value increases with the number of simultaneous readouts."""
    q_small = sup_t_critical(np.eye(3) * 0.1, alpha=0.05, n_mc=8000, rng=0)
    q_large = sup_t_critical(np.eye(30) * 0.1, alpha=0.05, n_mc=8000, rng=0)
    assert q_large > q_small
    # pointwise 95% z is ~1.96; simultaneous is larger
    assert q_small > 1.96


def test_simultaneous_coverage_correlated_gaussian():
    """Empirical simultaneous coverage in [0.90, 0.98] for a correlated readout vector."""
    rng = np.random.default_rng(0)
    m = 12
    # correlated covariance via a random low-rank + diagonal structure
    Bmat = rng.standard_normal((m, 3))
    C = 0.02 * (Bmat @ Bmat.T) + 0.02 * np.eye(m)
    L = np.linalg.cholesky(C)
    true = np.zeros(m)
    n_trials = 800
    covered = 0
    for _ in range(n_trials):
        r = true + L @ rng.standard_normal(m)
        band = simultaneous_bands(r, C, alpha=0.05, n_mc=3000, rng=rng)
        if np.all((band.ci_low <= true) & (true <= band.ci_high)):
            covered += 1
    cov = covered / n_trials
    assert 0.90 <= cov <= 0.98, f"coverage {cov}"


def test_bands_flag_significant_nonzero_effect():
    """A clearly nonzero effect is flagged significant; a zero effect is not."""
    m = 6
    C = np.eye(m) * 0.01  # se = 0.1
    r = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # first entry ~10 se from 0
    band = simultaneous_bands(r, C, alpha=0.05, n_mc=5000, rng=0)
    assert band.significant[0]
    assert not band.significant[1]
