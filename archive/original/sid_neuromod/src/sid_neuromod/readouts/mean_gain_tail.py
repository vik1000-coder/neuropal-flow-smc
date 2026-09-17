r"""Analytic mean / variance / gain / tail readouts (Section 9.2).

For the natural-parameter Gaussian with :math:`\eta_1 = \theta_1^\top\psi`,
:math:`\eta_2 = \theta_2^\top\psi`, :math:`v_\mathrm{eff} = -1/(2\eta_2)`,
:math:`v = v_\mathrm{eff} - \sigma^2`, :math:`\mu = \eta_1 v_\mathrm{eff}`, the
derivatives of conditional functionals with respect to feature coordinate ``k`` are:

.. math::

   \frac{\partial v}{\partial \psi_k} = 2 v_\mathrm{eff}^2 \theta_{2,k}, \qquad
   \frac{\partial \mu}{\partial \psi_k} = v_\mathrm{eff}\theta_{1,k}
        + 2\eta_1 v_\mathrm{eff}^2 \theta_{2,k}, \qquad
   \frac{\partial \log v}{\partial \psi_k} = \frac{1}{v}\frac{\partial v}{\partial \psi_k}.

High tail (:math:`P(Y>q) = 1-\Phi(a)`, :math:`a=(q-\mu)/\sqrt v`):

.. math::

   \frac{\partial P(Y>q)}{\partial \psi_k} = \phi(a)\left[
        \frac{1}{\sqrt v}\frac{\partial\mu}{\partial\psi_k}
        + \frac{q-\mu}{2 v^{3/2}}\frac{\partial v}{\partial\psi_k}\right].

Low tail (:math:`P(Y<q)=\Phi(a)`) has the negated bracket.

These are evaluated per-history and (by default) averaged over the test histories to
give the global readout :math:`D^\phi_{i\leftarrow j}(\tau)`.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..models.quadratic_score import FitResult


def _eff_params(res: FitResult, Psi: np.ndarray):
    """Return per-row (eta1, eta2, v_eff, v, mu) with prediction-time clamps."""
    Psi = np.asarray(Psi, dtype=float)
    eta1 = Psi @ res.theta1
    eta2 = Psi @ res.theta2
    eta2c = np.minimum(eta2, -res.eta2_min)
    v_eff = -1.0 / (2.0 * eta2c)
    v = np.maximum(v_eff - res.sigma ** 2, res.var_min)
    mu = eta1 * v_eff
    return eta1, eta2c, v_eff, v, mu


def d_mean(res: FitResult, Psi: np.ndarray, k: int) -> np.ndarray:
    r""":math:`\partial\mu/\partial\psi_k` per row of ``Psi``."""
    eta1, _, v_eff, _, _ = _eff_params(res, Psi)
    t1k = res.theta1[k]
    t2k = res.theta2[k]
    return v_eff * t1k + 2.0 * eta1 * v_eff ** 2 * t2k


def d_variance(res: FitResult, Psi: np.ndarray, k: int) -> np.ndarray:
    r""":math:`\partial v/\partial\psi_k` per row."""
    _, _, v_eff, _, _ = _eff_params(res, Psi)
    return 2.0 * v_eff ** 2 * res.theta2[k]


def d_log_variance(res: FitResult, Psi: np.ndarray, k: int) -> np.ndarray:
    r""":math:`\partial\log v/\partial\psi_k` per row (the gain channel)."""
    _, _, _, v, _ = _eff_params(res, Psi)
    return d_variance(res, Psi, k) / v


def d_tail_high(res: FitResult, Psi: np.ndarray, k: int, q: float) -> np.ndarray:
    r""":math:`\partial P(Y>q)/\partial\psi_k` per row."""
    _, _, _, v, mu = _eff_params(res, Psi)
    sv = np.sqrt(v)
    a = (q - mu) / sv
    dmu = d_mean(res, Psi, k)
    dv = d_variance(res, Psi, k)
    return norm.pdf(a) * (dmu / sv + (q - mu) / (2.0 * v ** 1.5) * dv)


def d_tail_low(res: FitResult, Psi: np.ndarray, k: int, q: float) -> np.ndarray:
    r""":math:`\partial P(Y<q)/\partial\psi_k` per row."""
    _, _, _, v, mu = _eff_params(res, Psi)
    sv = np.sqrt(v)
    a = (q - mu) / sv
    dmu = d_mean(res, Psi, k)
    dv = d_variance(res, Psi, k)
    return norm.pdf(a) * (-dmu / sv - (q - mu) / (2.0 * v ** 1.5) * dv)


CHANNELS = ("mean", "variance", "gain_log_variance", "tail_high", "tail_low")


def readout_at(res: FitResult, Psi: np.ndarray, k: int, channel: str,
               q_high: float | None = None, q_low: float | None = None) -> np.ndarray:
    """Per-row derivative of ``channel`` w.r.t. feature ``k``."""
    if channel == "mean":
        return d_mean(res, Psi, k)
    if channel == "variance":
        return d_variance(res, Psi, k)
    if channel == "gain_log_variance":
        return d_log_variance(res, Psi, k)
    if channel == "tail_high":
        if q_high is None:
            raise ValueError("q_high required for tail_high")
        return d_tail_high(res, Psi, k, q_high)
    if channel == "tail_low":
        if q_low is None:
            raise ValueError("q_low required for tail_low")
        return d_tail_low(res, Psi, k, q_low)
    raise ValueError(f"unknown channel {channel!r}")


def averaged_readout(res: FitResult, Psi: np.ndarray, k: int, channel: str,
                     q_high: float | None = None, q_low: float | None = None,
                     mask: np.ndarray | None = None) -> float:
    r"""Global-average readout :math:`D^\phi_{i\leftarrow j}(\tau)` over histories.

    ``mask`` selects a subset of rows (state-conditioned readouts).
    """
    vals = readout_at(res, Psi, k, channel, q_high, q_low)
    if mask is not None:
        vals = vals[mask]
    return float(np.mean(vals))


def readout_vector(res: FitResult, Psi: np.ndarray, feature_indices, channel: str,
                   q_high: float | None = None, q_low: float | None = None,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """Averaged readout for a set of feature columns (a directed multilag kernel)."""
    return np.array([
        averaged_readout(res, Psi, k, channel, q_high, q_low, mask)
        for k in feature_indices
    ])


def center_history(Psi: np.ndarray) -> np.ndarray:
    """The single 'typical' history: columnwise mean of the design matrix, as ``[1, P]``."""
    return np.asarray(Psi, dtype=float).mean(axis=0, keepdims=True)


def readout_at_center(res: FitResult, Psi: np.ndarray, k: int, channel: str,
                      q_high: float | None = None, q_low: float | None = None) -> float:
    r"""Readout evaluated at the centered history (stable single-history kernel value).

    This matches the empirics' ``variance_lag_readout``: evaluating the kernel at the
    mean history avoids the ``1/v`` blow-up that a global average over stochastic-
    volatility histories suffers (small-``v`` histories otherwise dominate).
    """
    center = center_history(Psi)
    return float(readout_at(res, center, k, channel, q_high, q_low)[0])


def center_readout_vector(res: FitResult, Psi: np.ndarray, feature_indices,
                          channel: str, q_high: float | None = None,
                          q_low: float | None = None) -> np.ndarray:
    """Center-history readout for a set of feature columns (directed multilag kernel)."""
    center = center_history(Psi)
    return np.array([
        float(readout_at(res, center, k, channel, q_high, q_low)[0])
        for k in feature_indices
    ])
