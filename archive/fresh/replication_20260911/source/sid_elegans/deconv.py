r"""Calcium deconvolution: recover a continuous ACTIVITY signal from GCaMP dF/F.

The observed calcium is a slow, low-pass, nonlinear transform of neural activity, so its
conditional-variance structure is dominated by GCaMP kinetics + measurement noise rather
than by neural computation. We deconvolve an AR(1) calcium model

    c_t = gamma * c_{t-1} + s_t,   s_t >= 0,   y_t = c_t + noise,

recovering the non-negative activity ``s`` (a firing-rate proxy) via the OASIS active-set
algorithm (Friedrich, Zhou & Paninski 2017), implemented natively. ``gamma`` is the
calcium decay per frame (``exp(-1/(tau*fps))``); we estimate it per neuron from the
autocorrelation. Deconvolving before the filter-bank readout removes the calcium confound
and yields a continuous-time-consistent activity representation.
"""
from __future__ import annotations

import numpy as np


def estimate_gamma(y: np.ndarray, fps: float = 4.0, lo: float = 0.5, hi: float = 0.98) -> float:
    """Estimate the AR(1) decay ``gamma`` from lag-1/lag-0 autocovariance, clipped."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return 0.85
    y = y - y.mean()
    g0 = float(np.dot(y, y)) / len(y)
    g1 = float(np.dot(y[1:], y[:-1])) / (len(y) - 1)
    if g0 <= 0:
        return 0.85
    return float(np.clip(g1 / g0, lo, hi))


def oasis_ar1(y: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """OASIS AR(1) non-negative deconvolution. Returns ``(c denoised calcium, s activity)``.

    Solves min ||c - y||^2 s.t. s_t = c_t - gamma c_{t-1} >= 0 by the pool active-set method.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    if T == 0:
        return y.copy(), y.copy()
    # pools: numerator val (sum y*g^k), denom wt (sum g^{2k}), onset t, length l
    val = np.zeros(T); wt = np.zeros(T); tt = np.zeros(T, int); ll = np.zeros(T, int)
    i = 0
    val[0], wt[0], tt[0], ll[0] = y[0], 1.0, 0, 1
    for j in range(1, T):
        i += 1
        val[i], wt[i], tt[i], ll[i] = y[j], 1.0, j, 1
        # merge while the AR(1) non-negativity (height_i >= g^{l_{i-1}} height_{i-1}) is violated
        while i > 0 and (val[i] / wt[i]) < (gamma ** ll[i - 1]) * (val[i - 1] / wt[i - 1]):
            gpow = gamma ** ll[i - 1]
            val[i - 1] = val[i - 1] + gpow * val[i]
            wt[i - 1] = wt[i - 1] + (gpow ** 2) * wt[i]
            ll[i - 1] = ll[i - 1] + ll[i]
            i -= 1
    c = np.zeros(T)
    for k in range(i + 1):
        h = max(val[k] / wt[k], 0.0)
        idx = tt[k] + np.arange(ll[k])
        c[idx] = h * gamma ** np.arange(ll[k])
    s = np.empty(T)
    s[0] = max(c[0], 0.0)
    s[1:] = np.maximum(c[1:] - gamma * c[:-1], 0.0)
    return c, s


def deconvolve_trace(y: np.ndarray, fps: float = 4.0, baseline_pct: float = 10.0):
    """Deconvolve one dF/F trace to activity; NaN-safe. Returns activity ``s`` (len T)."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    finite = np.isfinite(y)
    if finite.sum() < 10:
        return out
    yv = y[finite]
    base = np.percentile(yv, baseline_pct)
    yv = yv - base                       # baseline to ~0 (dF/F can dip negative from noise)
    gamma = estimate_gamma(yv, fps)
    _, s = oasis_ar1(yv, gamma)
    out[finite] = s
    return out


def deconvolve_matrix(X: np.ndarray, fps: float = 4.0) -> np.ndarray:
    """Deconvolve every column (neuron) of a ``[T, N]`` trace matrix."""
    X = np.asarray(X, dtype=float)
    S = np.empty_like(X)
    for j in range(X.shape[1]):
        S[:, j] = deconvolve_trace(X[:, j], fps)
    return S
