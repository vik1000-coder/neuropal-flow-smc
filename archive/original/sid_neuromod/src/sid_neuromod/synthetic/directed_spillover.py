r"""E6: directed spillover with two latent volatility factors (Section 12.6).

.. math::

   w'_1 = 0.9 w_1 + 0.4 \xi_1,\quad
   w'_2 = 0.7 w_2 + 0.3 w_1 + 0.3 \xi_2,\quad
   x_i = e^{w_i/2}\varepsilon_i.

One directional structural coupling (``w1 -> w2``). Predictive directed kernels can be
nonzero even in the structural-zero direction under partial observation — the warning
of Proposition 16.4 (predictive != structural directedness).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.rng import get_rng


@dataclass
class SpilloverData:
    x: np.ndarray   # [T, 2]
    w: np.ndarray   # [T, 2]


def simulate_directed_spillover(T: int = 300_000, a1: float = 0.9, a2: float = 0.7,
                                c21: float = 0.3, b1: float = 0.4, b2: float = 0.3,
                                seed: int = 0, burn_in: int = 1000) -> SpilloverData:
    rng = get_rng(seed)
    n = T + burn_in
    xi1 = rng.standard_normal(n)
    xi2 = rng.standard_normal(n)
    eps = rng.standard_normal((n, 2))
    w = np.zeros((n, 2))
    for t in range(1, n):
        w[t, 0] = a1 * w[t - 1, 0] + b1 * xi1[t]
        w[t, 1] = a2 * w[t - 1, 1] + c21 * w[t - 1, 0] + b2 * xi2[t]
    x = np.exp(w / 2.0) * eps
    return SpilloverData(x=x[burn_in:], w=w[burn_in:])
