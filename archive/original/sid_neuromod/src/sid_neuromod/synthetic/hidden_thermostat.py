r"""E1 / Example 2.1: the hidden thermostat (stochastic volatility).

A latent AR(1) "thermostat" sets the temperature of an observable that is otherwise
pure noise:

.. math::

   w_{t+1} = a\,w_t + b\,\xi_t, \qquad x_t = e^{w_t/2}\,\varepsilon_t,
   \qquad \xi,\varepsilon \sim \mathcal{N}(0,1)\ \text{i.i.d.},

with default ``(a, b) = (0.9, 0.4)``. Facts: ``E[x_{t+1}|past] = 0`` exactly and
``Cov(x_t, x_{t+k}) = 0`` for ``k != 0`` — the process is white with zero conditional
mean, invisible to every conditional-mean method. Yet ``Var(x_{t+1}|past)`` is a
nondegenerate function of the past: all dynamics live in the conditional variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.rng import get_rng


@dataclass
class ThermostatData:
    x: np.ndarray          # observed series [T]
    w: np.ndarray          # latent log-volatility [T]
    a: float
    b: float


def simulate_hidden_thermostat(T: int = 200_000, a: float = 0.9, b: float = 0.4,
                               seed: int = 0, burn_in: int = 1000) -> ThermostatData:
    """Simulate the hidden thermostat. Deterministic given ``seed``."""
    rng = get_rng(seed)
    n = T + burn_in
    xi = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    w = np.empty(n)
    # start at stationary variance b^2/(1-a^2)
    w[0] = np.sqrt(b ** 2 / (1 - a ** 2)) * rng.standard_normal()
    for t in range(1, n):
        w[t] = a * w[t - 1] + b * xi[t]
    x = np.exp(w / 2.0) * eps
    return ThermostatData(x=x[burn_in:], w=w[burn_in:], a=a, b=b)
