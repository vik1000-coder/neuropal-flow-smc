"""Section 17.5: PIT tests."""
import numpy as np

from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher
from sid_neuromod.monitoring.channels import dispersion_channel
from sid_neuromod.monitoring.pit import gaussian_pit, pit_diagnostics


def test_pit_uniform_correct_model():
    """PITs from a correctly-specified Gaussian are uniform; mean near 0.5."""
    rng = np.random.default_rng(0)
    n_seeds = 8
    good = 0
    means = []
    for s in range(n_seeds):
        r = np.random.default_rng(s)
        T = 20000
        f = r.standard_normal(T)
        mu = 0.5 * f
        var = np.exp(0.3 * f)
        y = mu + np.sqrt(var) * r.standard_normal(T)
        u = gaussian_pit(y, mu, var)
        d = pit_diagnostics(u)
        means.append(d["pit_mean"])
        if d["pit_ks_pvalue"] > 0.01:
            good += 1
    assert good >= n_seeds - 1  # KS p usually > 0.01
    assert abs(np.mean(means) - 0.5) < 0.02


def test_pit_detects_variance_misspecification():
    """Mean-only constant-variance fit to heteroscedastic data -> dispersion drifts."""
    rng = np.random.default_rng(1)
    T = 40000
    f = rng.standard_normal(T)
    var = np.exp(1.2 * f)  # strong heteroscedasticity
    y = np.sqrt(var) * rng.standard_normal(T)
    # constant-variance model: predict mu=0, var=Var(y)
    mu = np.zeros(T)
    v = np.full(T, float(np.var(y)))
    u = gaussian_pit(y, mu, v)
    disp = dispersion_channel(u)
    # dispersion channel mean differs from null (0) — over-dispersion => positive
    assert abs(np.mean(disp)) > 0.02
