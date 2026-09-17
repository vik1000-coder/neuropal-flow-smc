"""Section 17.1: filter-bank tests."""
import numpy as np
import pytest

from sid_neuromod.features.filter_bank import exp_filter_bank, exp_filter_direct


def test_filter_bank_constant_signal():
    """Constant signal z=1 -> filter approaches 1, monotone from initialized 0."""
    tau = 2.0
    duration = 30.0
    t = np.linspace(0, duration, 3000)
    z = np.ones_like(t)
    f = exp_filter_bank(z, t, [tau])[:, 0, 0]
    assert f[0] == 0.0
    assert np.all(np.diff(f) >= -1e-12)  # monotone nondecreasing
    final_err = abs(f[-1] - 1.0)
    assert final_err < np.exp(-duration / tau) + 1e-6


def test_filter_bank_no_future_leakage():
    """Features before t* are unchanged by values after t*."""
    t = np.arange(200, dtype=float)
    z = np.zeros(200)
    tstar = 100
    f_before = exp_filter_bank(z, t, [1.0, 5.0])
    z2 = z.copy()
    z2[tstar:] = 100.0  # huge jump after t*
    f_after = exp_filter_bank(z2, t, [1.0, 5.0])
    # feature at index k depends only on z[:k]; index tstar uses z[tstar-1] (=0), so
    # features at indices 0..tstar are identical
    assert np.allclose(f_before[: tstar + 1], f_after[: tstar + 1])
    # and they diverge afterwards
    assert not np.allclose(f_before[tstar + 2:], f_after[tstar + 2:])


def test_filter_bank_irregular_grid():
    """Recursive implementation matches direct ZOH integration on irregular grid."""
    rng = np.random.default_rng(7)
    dt = rng.exponential(0.4, size=800)
    t = np.cumsum(dt)
    t -= t[0]
    z = rng.standard_normal(800)
    for tau in (0.5, 3.0, 17.0):
        f_rec = exp_filter_bank(z, t, [tau])[:, 0, 0]
        f_dir = exp_filter_direct(z, t, tau)
        assert np.max(np.abs(f_rec - f_dir)) < 1e-10


def test_filter_bank_rejects_nonincreasing_time():
    t = np.array([0.0, 1.0, 1.0, 2.0])
    z = np.ones(4)
    with pytest.raises(ValueError):
        exp_filter_bank(z, t, [1.0])
