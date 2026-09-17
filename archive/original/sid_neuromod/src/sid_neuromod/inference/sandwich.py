r"""Sandwich covariance of the estimator and delta-method readout covariance (10.1-10.2).

The estimator solves :math:`A_T\hat\theta + c_T = 0`, so

.. math::

   \widehat{\mathrm{Cov}}(\hat\theta) = A_T^{-1} S A_T^{-\top} / T,

with ``S`` the HAC long-run covariance of the per-sample estimating functions. For a
readout vector :math:`r(\theta)` with Jacobian :math:`J=\partial r/\partial\theta`,

.. math::

   \widehat{\mathrm{Cov}}(r) = J\,\widehat{\mathrm{Cov}}(\hat\theta)\,J^\top .
"""
from __future__ import annotations

import numpy as np

from ..models.quadratic_score import FitResult, QuadraticScoreMatcher
from ..utils.linalg import corr_from_cov
from .hac import cross_newey_west, newey_west


def theta_covariance(res: FitResult, n_lags: int | None = None) -> np.ndarray:
    r"""Sandwich covariance :math:`A^{-1} S A^{-\top}/T` for one target's ``theta``."""
    S = newey_west(res.xi, n_lags=n_lags)
    Ainv = np.linalg.inv(res.A)
    T = res.n_samples
    return (Ainv @ S @ Ainv.T) / T


def cross_theta_covariance(res_i: FitResult, res_k: FitResult,
                           n_lags: int | None = None) -> np.ndarray:
    r"""Cross-target covariance :math:`A_i^{-1}S_{ik}A_k^{-\top}/T`."""
    Sik = cross_newey_west(res_i.xi, res_k.xi, n_lags=n_lags)
    Ai_inv = np.linalg.inv(res_i.A)
    Ak_inv = np.linalg.inv(res_k.A)
    T = res_i.n_samples
    return (Ai_inv @ Sik @ Ak_inv.T) / T


def readout_jacobian_fd(res: FitResult, readout_fn, eps: float = 1e-6) -> np.ndarray:
    r"""Central finite-difference Jacobian :math:`J = \partial r/\partial\theta`.

    ``readout_fn(theta) -> r`` maps the flat parameter vector to the readout vector.
    Priority per Section 10.2: finite-difference first (this), analytic later.
    """
    theta0 = res.theta.copy()
    r0 = np.atleast_1d(np.asarray(readout_fn(theta0), dtype=float))
    m = r0.shape[0]
    n = theta0.shape[0]
    J = np.zeros((m, n))
    for j in range(n):
        step = eps * max(1.0, abs(theta0[j]))
        tp = theta0.copy(); tp[j] += step
        tm = theta0.copy(); tm[j] -= step
        rp = np.atleast_1d(np.asarray(readout_fn(tp), dtype=float))
        rm = np.atleast_1d(np.asarray(readout_fn(tm), dtype=float))
        J[:, j] = (rp - rm) / (2.0 * step)
    return J


def readout_covariance(res: FitResult, readout_fn, n_lags: int | None = None,
                       eps: float = 1e-6):
    """Delta-method covariance of a readout vector; returns ``(r, cov, J)``."""
    r = np.atleast_1d(np.asarray(readout_fn(res.theta), dtype=float))
    J = readout_jacobian_fd(res, readout_fn, eps=eps)
    cov_theta = theta_covariance(res, n_lags=n_lags)
    cov_r = J @ cov_theta @ J.T
    return r, 0.5 * (cov_r + cov_r.T), J
