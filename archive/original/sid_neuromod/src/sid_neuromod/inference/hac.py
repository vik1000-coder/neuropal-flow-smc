r"""HAC / Newey-West long-run covariance (Section 10.1).

For per-sample estimating functions :math:`\xi_t \in \mathbb{R}^p`,

.. math::

   S = \Gamma_0 + \sum_{\ell=1}^{L}\, w_\ell\,(\Gamma_\ell + \Gamma_\ell^\top),
   \qquad w_\ell = 1 - \frac{\ell}{L+1},

the Bartlett-kernel HAC estimator (Newey-West, 1987), which is positive semi-definite
by construction.
"""
from __future__ import annotations

import numpy as np


def default_hac_lags(T: int) -> int:
    r"""Default HAC lag ``max(20, ceil(4 (T/100)^{2/9}))`` (Section 10.1)."""
    return int(max(20, np.ceil(4.0 * (T / 100.0) ** (2.0 / 9.0))))


def newey_west(xi: np.ndarray, n_lags: int | None = None) -> np.ndarray:
    r"""Bartlett-kernel HAC covariance of the mean of ``xi``.

    Parameters
    ----------
    xi : array [T, p]
        Per-sample estimating functions (need not be mean-zero; they are centered).
    n_lags : int or None
        Truncation lag ``L``. ``None`` uses :func:`default_hac_lags`. ``L=0`` gives the
        plain sample covariance :math:`\Gamma_0`.

    Returns
    -------
    S : array [p, p]
        Long-run covariance estimate (symmetric, PSD).
    """
    xi = np.asarray(xi, dtype=float)
    if xi.ndim == 1:
        xi = xi[:, None]
    T, p = xi.shape
    if n_lags is None:
        n_lags = default_hac_lags(T)
    n_lags = int(min(n_lags, T - 1))
    xc = xi - xi.mean(axis=0, keepdims=True)

    S = (xc.T @ xc) / T  # Gamma_0
    for lag in range(1, n_lags + 1):
        w = 1.0 - lag / (n_lags + 1.0)
        G = (xc[lag:].T @ xc[:-lag]) / T  # Gamma_lag
        S += w * (G + G.T)
    return 0.5 * (S + S.T)


def cross_newey_west(xi_i: np.ndarray, xi_k: np.ndarray,
                     n_lags: int | None = None) -> np.ndarray:
    r"""Cross-HAC covariance :math:`S_{ik}` between two targets' estimating functions.

    .. math::

        S_{ik} = \sum_{\ell=-L}^{L} w_{|\ell|}
                 \mathrm{Cov}(\xi^{(i)}_t, \xi^{(k)}_{t-\ell}).
    """
    xi_i = np.asarray(xi_i, dtype=float)
    xi_k = np.asarray(xi_k, dtype=float)
    T = xi_i.shape[0]
    if n_lags is None:
        n_lags = default_hac_lags(T)
    n_lags = int(min(n_lags, T - 1))
    ai = xi_i - xi_i.mean(axis=0, keepdims=True)
    ak = xi_k - xi_k.mean(axis=0, keepdims=True)

    S = (ai.T @ ak) / T  # lag 0
    for lag in range(1, n_lags + 1):
        w = 1.0 - lag / (n_lags + 1.0)
        # positive lag: xi_i[t], xi_k[t-lag]
        Gp = (ai[lag:].T @ ak[:-lag]) / T
        # negative lag: xi_i[t], xi_k[t+lag]
        Gn = (ai[:-lag].T @ ak[lag:]) / T
        S += w * (Gp + Gn)
    return S
