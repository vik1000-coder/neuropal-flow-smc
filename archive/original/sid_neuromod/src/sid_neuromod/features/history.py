"""Source-signal construction and causal history/target builders.

Source signal kinds (Section 7.2):
  raw_zscore, delta, positive_part, negative_part, variance_channel (z=log x^2).

Targets (Section 7.1):
  next  : Y_i(t) = X_i(t + horizon)
  delta : Y_i(t) = X_i(t + horizon) - X_i(t)
"""
from __future__ import annotations

import numpy as np

from ..utils.arrays import as_2d, safe_log_square


def source_signal(x: np.ndarray, kind: str) -> np.ndarray:
    """Construct a source signal from a single (already z-scored) neuron trace ``x``.

    Returns an array the same length as ``x``; ``delta`` prepends 0 so length is kept.
    """
    x = np.asarray(x, dtype=float).ravel()
    if kind == "raw_zscore":
        return x
    if kind == "delta":
        d = np.empty_like(x)
        d[0] = 0.0
        d[1:] = np.diff(x)
        return d
    if kind == "positive_part":
        return np.maximum(x, 0.0)
    if kind == "negative_part":
        return np.minimum(x, 0.0)
    if kind == "variance_channel":
        return safe_log_square(x)
    raise ValueError(f"unknown source signal kind: {kind!r}")


def make_target(X: np.ndarray, horizon_steps: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Build targets for horizon in *samples*.

    Returns ``(Y, valid_index)`` where ``Y`` has shape ``[T, N]`` with the last
    ``horizon_steps`` rows invalid, and ``valid_index`` is the boolean mask of usable
    rows (those for which ``t + horizon`` exists).
    """
    X = as_2d(X)
    T, N = X.shape
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1")
    Y = np.full((T, N), np.nan)
    valid = np.zeros(T, dtype=bool)
    hi = T - horizon_steps
    if mode == "next":
        Y[:hi] = X[horizon_steps:]
    elif mode == "delta":
        Y[:hi] = X[horizon_steps:] - X[:hi]
    else:
        raise ValueError("mode must be 'next' or 'delta'")
    valid[:hi] = True
    return Y, valid


def default_timescale_grid(median_frame_s: float, duration_s: float,
                           n: int = 10) -> np.ndarray:
    """Log-spaced timescale grid adapted to sampling rate & recording length (7.3)."""
    tmin = 2.0 * median_frame_s
    tmax = min(0.25 * duration_s, 600.0)
    if tmax <= tmin:
        tmax = tmin * 10.0
    return np.geomspace(tmin, tmax, num=n)
