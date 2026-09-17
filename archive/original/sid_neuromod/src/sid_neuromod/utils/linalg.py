"""Small numerical-linear-algebra helpers used across the package."""
from __future__ import annotations

import numpy as np


def ridge_solve(A: np.ndarray, b: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Solve ``(A + ridge*I) x = b`` robustly.

    Falls back to a least-squares solve if the ridge-regularized system is still
    singular. ``A`` is assumed square.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    M = A + ridge * np.eye(n)
    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:  # pragma: no cover - rare
        return np.linalg.lstsq(M, b, rcond=None)[0]


def nearest_psd(C: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD matrix by eigenvalue clipping."""
    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def corr_from_cov(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Correlation matrix from a covariance matrix (safe against zero variances)."""
    C = np.asarray(C, dtype=float)
    d = np.sqrt(np.clip(np.diag(C), eps, None))
    corr = C / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def min_eig(C: np.ndarray) -> float:
    """Minimum eigenvalue of a symmetric matrix."""
    C = 0.5 * (np.asarray(C, dtype=float) + np.asarray(C, dtype=float).T)
    return float(np.linalg.eigvalsh(C)[0])
