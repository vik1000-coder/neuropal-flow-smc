"""Section 17.6: e-process tests."""
import numpy as np
import pytest

from sid_neuromod.monitoring.eprocess import (EProcess, first_alarm,
                                              robbins_log_evalue)


@pytest.mark.slow
def test_eprocess_supermartingale_smoke():
    """Under iid bounded zero-mean channels, false alarm rate <= 2*alpha."""
    alpha = 0.05
    n, reps = 2000, 300
    rng = np.random.default_rng(0)
    ep = EProcess(eps=0.0)
    fa = 0
    for _ in range(reps):
        d = rng.uniform(-1, 1, size=n)  # bounded [-1,1], mean 0
        if first_alarm(ep.run(d), alpha=alpha) is not None:
            fa += 1
    assert fa / reps <= 2 * alpha


def test_eprocess_null_validity_fast():
    """Fast null check: running e-value rarely alarms under the null."""
    alpha = 0.05
    n, reps = 1500, 60
    rng = np.random.default_rng(1)
    ep = EProcess(eps=0.0)
    fa = sum(first_alarm(ep.run(rng.uniform(-1, 1, n)), alpha) is not None
             for _ in range(reps))
    assert fa / reps <= 0.15


def test_eprocess_detects_positive_drift():
    """Under drift, alarm in a majority of runs; median delay finite."""
    alpha = 0.05
    n, reps = 3000, 60
    rng = np.random.default_rng(2)
    ep = EProcess(eps=0.0)
    alarms, delays = 0, []
    for _ in range(reps):
        d = np.clip(0.15 + rng.uniform(-1, 1, size=n), -1, 1)  # +0.15 drift
        a = first_alarm(ep.run(d), alpha=alpha)
        if a is not None:
            alarms += 1
            delays.append(a)
    assert alarms > reps / 2
    assert np.isfinite(np.median(delays))


def test_robbins_evalue_null_expectation():
    """Robbins mixture e-value has expectation ~<= 1 at a fixed time under the null."""
    rng = np.random.default_rng(3)
    n_mc = 20000
    length = np.full(n_mc, 500.0)
    # null: sum of 500 iid bounded zero-mean increments (variance proxy c=1)
    S = np.array([np.sum(rng.uniform(-1, 1, 500)) for _ in range(2000)])
    length = np.full(len(S), 500.0)
    ev = np.exp(robbins_log_evalue(S, length, c=1.0, eps=0.0))
    assert np.mean(ev) <= 1.2  # e-value: E[M] <= 1 (MC tolerance)


def test_dyadic_detector_detects_late_change():
    """A change partway through is caught by the dyadic-restart detector."""
    rng = np.random.default_rng(4)
    n = 8000
    d = rng.uniform(-1, 1, n)
    d[4000:] += 0.2  # drift after the midpoint
    ep = EProcess(eps=0.0)
    trace = ep.run_detector(d, c=1.0)
    assert first_alarm(trace, alpha=0.01) is not None
