r"""Preprocessing: train-only z-scoring, optional detrend (Sections 5.2, 13.2).

Scaling parameters are fit on the train interval only and applied to the rest.
"""
from __future__ import annotations

import numpy as np

from ..features.scaling import Scaler


def zscore_train(X: np.ndarray, train_slice: slice, eps: float = 1e-8):
    """Z-score each neuron using train-interval statistics only.

    Returns ``(X_z, scaler)`` where ``scaler`` stores the train mean/std.
    """
    X = np.asarray(X, dtype=float)
    scaler = Scaler()
    scaler.fit(X[train_slice])
    return scaler.transform(X), scaler


def detrend_linear(X: np.ndarray, train_slice: slice) -> np.ndarray:
    """Remove a per-neuron linear trend fit on the train interval (photobleaching)."""
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    t = np.arange(T)
    ttr = t[train_slice]
    out = X.copy()
    for j in range(N):
        A = np.column_stack([ttr, np.ones_like(ttr)])
        coef, *_ = np.linalg.lstsq(A, X[train_slice, j], rcond=None)
        out[:, j] = X[:, j] - (coef[0] * t + coef[1])
    return out
