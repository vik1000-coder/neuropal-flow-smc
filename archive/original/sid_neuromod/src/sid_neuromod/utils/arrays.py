"""Array utilities."""
from __future__ import annotations

import numpy as np


def as_2d(x: np.ndarray) -> np.ndarray:
    """Ensure a 2D array of shape ``[T, N]`` from ``[T]`` or ``[T, N]`` input."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {x.shape}")
    return x


def zscore(x: np.ndarray, mean: np.ndarray | None = None,
           std: np.ndarray | None = None, eps: float = 1e-8):
    """Z-score ``x`` columnwise. If ``mean``/``std`` are given, apply them (test path)."""
    x = as_2d(x)
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std, mean, std


def center(x: np.ndarray, mean: np.ndarray | None = None):
    """Subtract columnwise mean; return centered array and the mean used."""
    x = as_2d(x)
    if mean is None:
        mean = x.mean(axis=0)
    return x - mean, mean


def safe_log_square(x: np.ndarray, floor: float = -8.0) -> np.ndarray:
    """Variance-channel observable ``z = max(log x^2, floor)`` (floors the log-chi2 tail)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        z = np.log(x * x)
    return np.maximum(z, floor)
