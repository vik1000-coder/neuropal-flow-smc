"""Level-2 GaussianNLL model and baseline sanity."""
import numpy as np

from sid_neuromod.baselines.ridge_ar import fit_ridge_ar
from sid_neuromod.baselines.var import fit_var
from sid_neuromod.models.gaussian_nll import GaussianNLL


def test_gaussian_nll_recovers_mean_and_variance():
    """Level-2 model recovers a linear mean and a log-variance coefficient."""
    rng = np.random.default_rng(0)
    T = 30000
    f = rng.standard_normal(T)
    Psi = np.column_stack([np.ones(T), f])
    mu_true = 0.5 * f
    var_true = np.exp(0.4 * f)
    Y = mu_true + np.sqrt(var_true) * rng.standard_normal(T)
    m = GaussianNLL(ridge=1e-5)
    r = m.fit(Y, Psi)
    # mean weight ~ 0.5, log-variance slope ~ 0.4
    assert abs(r.a[1] - 0.5) < 0.05
    assert abs(r.b[1] - 0.4) < 0.08
    mu, var = m.moments(Psi[:5])
    assert np.all(var > 0)
    assert np.isfinite(m.nll(Y, Psi))


def test_gaussian_nll_never_invalid_variance():
    """Unlike the closed-form model, the Level-2 model cannot emit invalid variance."""
    rng = np.random.default_rng(1)
    T = 500
    Psi = np.column_stack([np.ones(T), rng.standard_normal((T, 4))])
    Y = rng.standard_t(2, size=T) * 2
    m = GaussianNLL(ridge=1e-3)
    m.fit(Y, Psi)
    _, var = m.moments(Psi)
    assert np.all(var > 0) and np.all(np.isfinite(var))


def test_ridge_ar_baseline_near_zero_on_white_series():
    """Ridge-AR R2 ~ 0 on a white series (no mean-predictable structure)."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(20000)
    res = fit_ridge_ar(x, n_lags=8, alpha=1.0)
    assert res.r2 < 0.01


def test_var_baseline_runs():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((3000, 3))
    X[1:, 1] += 0.6 * X[:-1, 0]  # a real mean edge
    res = fit_var(X, order=2)
    assert res.r2 >= 0.0
    assert np.isfinite(res.heldout_nll)
