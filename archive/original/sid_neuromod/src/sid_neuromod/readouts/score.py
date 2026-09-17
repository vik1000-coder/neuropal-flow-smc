r"""Lag-influence score for the quadratic-T exponential family (Section 2.2, 6.1).

For :math:`T(y)=(y,y^2)` and natural parameters linear in features, the lag-influence
score with respect to feature coordinate ``k`` is

.. math::

   \ell_k(y;h) = \theta_{1,k}\,(y - \mathbb{E}[y\mid h])
               + \theta_{2,k}\,(y^2 - \mathbb{E}[y^2\mid h]),

an exact, closed-form centering (Proposition 6.1(a)). Its projection on ``y`` gives
mean effects, on ``y^2`` gives variance effects.
"""
from __future__ import annotations

import numpy as np

from ..models.quadratic_score import FitResult
from .mean_gain_tail import _eff_params


def lag_score(res: FitResult, Psi_row: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    r"""Evaluate :math:`\ell_k(y;h)` at a single history ``Psi_row`` for samples ``y``.

    ``E[y|h] = mu`` and ``E[y^2|h] = mu^2 + v`` under the fitted Gaussian conditional.
    """
    Psi_row = np.asarray(Psi_row, dtype=float).reshape(1, -1)
    _, _, _, v, mu = _eff_params(res, Psi_row)
    mu = float(mu[0]); v = float(v[0])
    ey2 = mu ** 2 + v
    y = np.asarray(y, dtype=float)
    return res.theta1[k] * (y - mu) + res.theta2[k] * (y ** 2 - ey2)
