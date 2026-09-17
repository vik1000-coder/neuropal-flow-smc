r"""E3: conditional-law change scenarios (Section 12.3).

Scenarios (change at ``t_change``):
  * ``null``   : no change.
  * ``scale``  : emission scale multiplied (variance changes, mean unchanged).
  * ``memory`` : latent AR coefficient ``a`` changes (marginal variance ~ stable).
  * ``mean``   : additive drift added post-change (simple mean change).

All are built on the hidden-thermostat SV process so that mean-only detectors fail on
the scale/memory scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.rng import get_rng


@dataclass
class ChangeStream:
    x: np.ndarray
    w: np.ndarray
    t_change: int
    scenario: str


def simulate_change_stream(T: int = 22_000, t_change: int = 6_000,
                           scenario: str = "scale", a: float = 0.9, b: float = 0.4,
                           scale_factor: float = 1.4, a_post: float = 0.5,
                           mean_shift: float = 0.5, seed: int = 0,
                           burn_in: int = 1000) -> ChangeStream:
    rng = get_rng(seed)
    n = T + burn_in
    tc = t_change + burn_in
    eps = rng.standard_normal(n)
    xi = rng.standard_normal(n)
    w = np.empty(n)
    w[0] = np.sqrt(b ** 2 / (1 - a ** 2)) * rng.standard_normal()
    a_series = np.full(n, a)
    if scenario == "memory":
        a_series[tc:] = a_post
    for t in range(1, n):
        w[t] = a_series[t] * w[t - 1] + b * xi[t]
    scale = np.ones(n)
    if scenario == "scale":
        scale[tc:] = scale_factor
    x = scale * np.exp(w / 2.0) * eps
    if scenario == "mean":
        x[tc:] = x[tc:] + mean_shift
    elif scenario == "null":
        pass
    return ChangeStream(x=x[burn_in:], w=w[burn_in:], t_change=t_change,
                        scenario=scenario)
