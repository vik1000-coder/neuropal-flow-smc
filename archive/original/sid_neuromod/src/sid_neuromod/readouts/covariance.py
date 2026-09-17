r"""Derivative-free covariance readout (Stein shift, Proposition 4.4).

.. math::

   \frac{\partial}{\partial\psi_k}\mathbb{E}[\phi(Y)\mid H_t]
      = \mathrm{Cov}_{\rho(\cdot\mid H_t)}\big(\phi(Y), \ell_k(Y;H_t)\big).

We draw ``M`` Monte Carlo samples from the fitted conditional and take the empirical
covariance against the lag-influence score. This equals the analytic derivative up to
:math:`O(M^{-1/2})` sampling noise (the same object, computed without differentiating
the model). Standard statistics ``phi``:

  * ``mean``     : ``phi(y) = y``
  * ``variance`` : ``phi(y) = (y - mu)^2``
"""
from __future__ import annotations

import numpy as np

from ..models.quadratic_score import FitResult
from ..utils.rng import get_rng
from .mean_gain_tail import _eff_params
from .score import lag_score


def covariance_readout(res: FitResult, Psi_row: np.ndarray, k: int,
                       phi: str = "mean", M: int = 4000, rng=None) -> float:
    """Monte-Carlo covariance readout at a single history for feature ``k``."""
    rng = get_rng(rng)
    Psi_row = np.asarray(Psi_row, dtype=float).reshape(1, -1)
    _, _, _, v, mu = _eff_params(res, Psi_row)
    mu = float(mu[0]); v = float(v[0])
    y = mu + np.sqrt(v) * rng.standard_normal(M)
    if phi == "mean":
        phiv = y
    elif phi == "variance":
        phiv = (y - mu) ** 2
    else:
        raise ValueError(f"unknown phi {phi!r}")
    ell = lag_score(res, Psi_row, y, k)
    return float(np.cov(phiv, ell, bias=True)[0, 1])
