r"""E2: OU / AR process with additive measurement noise (Lemma 7.4, Section 11)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.rng import get_rng


@dataclass
class OUNoisyData:
    x_clean: np.ndarray
    x_obs: np.ndarray
    dt: float
    gamma: float
    D: float
    R: float


def simulate_ou_noisy(T: int = 500_000, gamma: float = 1.0, D: float = 1.0,
                      R: float = 0.5, dt: float = 0.02, seed: int = 0) -> OUNoisyData:
    r"""Exact-discretization OU with measurement noise.

    ``x_{k+1} = e^{-gamma dt} x_k + sqrt(sigma^2 (1-e^{-2 gamma dt})) eta_k``,
    ``sigma^2 = D/gamma``; observed as ``x_obs = x + N(0, R)``.
    """
    rng = get_rng(seed)
    sig2 = D / gamma
    phi = np.exp(-gamma * dt)
    innov_sd = np.sqrt(sig2 * (1 - phi ** 2))
    x = np.empty(T)
    x[0] = np.sqrt(sig2) * rng.standard_normal()
    eta = rng.standard_normal(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + innov_sd * eta[t]
    x_obs = x + np.sqrt(R) * rng.standard_normal(T)
    return OUNoisyData(x_clean=x, x_obs=x_obs, dt=dt, gamma=gamma, D=D, R=R)


def simulate_ar1_eiv(T: int = 300_000, a: float = 0.8, R: float = 1.0,
                     seed: int = 0):
    r"""AR(1) with errors-in-variables (measurement noise). Returns ``(x_clean, x_obs)``.

    Attenuation oracle: the naive lag-1 coefficient is ``a * Var(x)/(Var(x)+R)``.
    """
    rng = get_rng(seed)
    x = np.empty(T)
    x[0] = rng.standard_normal()
    e = rng.standard_normal(T)
    for t in range(1, T):
        x[t] = a * x[t - 1] + e[t]
    x_obs = x + np.sqrt(R) * rng.standard_normal(T)
    return x, x_obs
