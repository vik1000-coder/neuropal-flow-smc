r"""Exponential filter bank with exact zero-order-hold (ZOH) updates.

For a source signal :math:`z_j(t)` and timescale :math:`\tau`, the causal feature is

.. math::

    f_{j,\tau}(t) = \int_0^\infty \frac{1}{\tau} e^{-s/\tau} z_j(t-s)\, ds .

For discretely (possibly irregularly) sampled data we use the exact zero-order-hold
recursion (Section 7.3 of the plan):

.. math::

    f_{j,\tau}(t_{k+1}) = e^{-\Delta t_k / \tau} f_{j,\tau}(t_k)
        + (1 - e^{-\Delta t_k / \tau}) z_j(t_k).

This holds the input constant on each inter-sample interval (ZOH), which is the exact
integral of the exponential kernel against a piecewise-constant input, and gives
sampling-rate-stable coefficients. The recursion is strictly causal: the feature at
time ``t_{k+1}`` uses the *held* value ``z_j(t_k)`` and never future samples.
"""
from __future__ import annotations

import numpy as np

from ..utils.arrays import as_2d


def exp_filter_bank(
    z: np.ndarray,
    timestamps_s: np.ndarray,
    timescales_s,
    *,
    init: str = "zero",
) -> np.ndarray:
    r"""Compute the exponential filter bank for one or many source signals.

    Parameters
    ----------
    z : array [T] or [T, J]
        Source signal(s), sampled at ``timestamps_s``.
    timestamps_s : array [T]
        Strictly increasing sample times.
    timescales_s : sequence of floats, length K
        The exponential timescales (in seconds).
    init : {"zero", "first"}
        Filter initialization. ``"zero"`` starts the state at 0 (default, matches the
        unit tests' "monotone from 0" expectation). ``"first"`` warms the state to the
        first sample value.

    Returns
    -------
    F : array [T, J, K]
        ``F[k, j, r]`` is filter output for signal ``j`` at timescale ``r``, time ``k``.
        The output at time ``k`` depends only on ``z[:k+1]`` (strictly causal: the
        value at ``k`` incorporates ``z[k]`` only through no-future ZOH — see note).

    Notes
    -----
    The value stored at index ``k`` is the filter state *before* absorbing an
    innovation from a future interval; concretely ``F[k]`` uses ``z[0..k]`` and the
    intervals ``t[1]-t[0], ..., t[k]-t[k-1]``. This is the strictly-causal convention:
    ``F[k]`` is a function of history up to and including ``t_k`` only.
    """
    z = as_2d(z)  # [T, J]
    t = np.asarray(timestamps_s, dtype=float)
    T, J = z.shape
    if t.shape[0] != T:
        raise ValueError(f"timestamps length {t.shape[0]} != signal length {T}")
    taus = np.atleast_1d(np.asarray(timescales_s, dtype=float))
    if np.any(taus <= 0):
        raise ValueError("timescales must be positive")
    K = taus.shape[0]

    dt = np.diff(t)  # [T-1]
    if np.any(dt <= 0):
        raise ValueError("timestamps must be strictly increasing")

    F = np.empty((T, J, K), dtype=float)
    if init == "zero":
        state = np.zeros((J, K), dtype=float)
    elif init == "first":
        state = np.repeat(z[0][:, None], K, axis=1)
    else:  # pragma: no cover - guarded
        raise ValueError("init must be 'zero' or 'first'")

    F[0] = state
    for k in range(1, T):
        decay = np.exp(-dt[k - 1] / taus)  # [K]
        # ZOH: absorb the value held over (t_{k-1}, t_k], i.e. z[k-1].
        state = decay[None, :] * state + (1.0 - decay[None, :]) * z[k - 1][:, None]
        F[k] = state
    return F


def exp_filter_direct(
    z: np.ndarray,
    timestamps_s: np.ndarray,
    tau: float,
) -> np.ndarray:
    r"""Reference (slow) ZOH filter for a single signal & timescale, by direct summation.

    Computes, for each ``k``, the exact integral of the exponential kernel against the
    piecewise-constant (ZOH) reconstruction of ``z`` over the past, starting from a
    zero initial state at ``t[0]``. Used by unit tests to check the recursion.
    """
    z = np.asarray(z, dtype=float).ravel()
    t = np.asarray(timestamps_s, dtype=float)
    T = z.shape[0]
    out = np.zeros(T, dtype=float)
    state = 0.0
    for k in range(1, T):
        dt = t[k] - t[k - 1]
        decay = np.exp(-dt / tau)
        state = decay * state + (1.0 - decay) * z[k - 1]
        out[k] = state
    return out
