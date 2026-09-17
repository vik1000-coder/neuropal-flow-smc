"""Reference baselines in the correct ``[post, pre]`` orientation.

- Pearson lag cross-correlation: ``P[j, i] = corr(x_i(t), x_j(t+lag))`` (edge i -> j).
- Ridge-VAR (multivariate linear conditional mean): regress ``x_j(t+lag)`` on all
  ``x_i(t)``; coefficient on source ``i`` is the directed linear drive ``i -> j``. This is
  the conditional-mean analog of the sid_neuromod ``mean`` channel, and the natural
  linear comparator to the distributional channels.
"""
from __future__ import annotations

import numpy as np


def _pool(X_list, lag):
    now, fut = [], []
    for X in X_list:
        if X.shape[0] <= lag:
            continue
        now.append(np.nan_to_num(X[:X.shape[0] - lag], nan=0.0))
        fut.append(np.nan_to_num(X[lag:], nan=0.0))
    return np.concatenate(now, axis=0), np.concatenate(fut, axis=0)


def pearson_lag(X_list, lag: int) -> np.ndarray:
    """Directed lagged Pearson correlation ``P[j, i] = corr(x_i(t), x_j(t+lag))``."""
    Xn, Xf = _pool(X_list, lag)
    N = Xn.shape[1]
    Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-8)
    Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-8)
    # C[i, j] = corr(x_i(t), x_j(t+lag)); we want P[post=j, pre=i] = C[i, j]
    C = (Xn.T @ Xf) / Xn.shape[0]
    P = C.T.copy()
    np.fill_diagonal(P, 0.0)
    return P


def ridge_var(X_list, lag: int, ridge: float = 1.0) -> np.ndarray:
    """Multivariate ridge regression of ``x_j(t+lag)`` on all ``x_i(t)``.

    Returns ``B[j, i]`` = coefficient of source ``i`` predicting target ``j`` (``[post, pre]``).
    """
    Xn, Xf = _pool(X_list, lag)
    N = Xn.shape[1]
    Xc = Xn - Xn.mean(0)
    G = Xc.T @ Xc + ridge * np.eye(N)
    Ginv = np.linalg.inv(G)
    B = np.zeros((N, N))
    for j in range(N):
        y = Xf[:, j] - Xf[:, j].mean()
        coef = Ginv @ (Xc.T @ y)     # length N, coef[i] = source i -> target j
        B[j, :] = coef
    np.fill_diagonal(B, 0.0)
    return B
