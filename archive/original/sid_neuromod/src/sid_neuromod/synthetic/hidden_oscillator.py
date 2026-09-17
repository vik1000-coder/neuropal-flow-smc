r"""E4 / Example 2.2: the hidden oscillator (partially observed linear system).

A scalar observable coupled to a hidden 2D rotation:

.. math::

   s_{t+1} = A s_t + Q^{1/2}\,\text{noise}, \qquad x = s^{(1)},

with ``A = [[Axx, Axw],[Awx, Aww]]``, ``Aww = r R(theta)``. The memory kernel is
:math:`K_u \propto r^{u-1}\cos((u-1)\theta)` (damped oscillation); ESPRIT on a fitted
kernel recovers the hidden modulus ``r`` and angle ``theta``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.rng import get_rng


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


@dataclass
class OscillatorData:
    x: np.ndarray
    s: np.ndarray
    A: np.ndarray
    r: float
    theta: float


def simulate_hidden_oscillator(T: int = 400_000, Axx: float = 0.2,
                               Axw=(0.5, 0.0), Awx=(0.4, 0.0), r: float = 0.9,
                               theta: float = 0.5, q_hidden: float = 0.0,
                               seed: int = 0, burn_in: int = 1000) -> OscillatorData:
    rng = get_rng(seed)
    Axw = np.asarray(Axw, dtype=float)
    Awx = np.asarray(Awx, dtype=float)
    Aww = r * rotation(theta)
    A = np.zeros((3, 3))
    A[0, 0] = Axx
    A[0, 1:] = Axw
    A[1:, 0] = Awx
    A[1:, 1:] = Aww
    Q = np.zeros((3, 3))
    Q[0, 0] = 1.0
    if q_hidden > 0:
        Q[1, 1] = Q[2, 2] = q_hidden
    Qsqrt = np.sqrt(np.diag(Q))
    n = T + burn_in
    s = np.zeros((n, 3))
    for t in range(1, n):
        s[t] = A @ s[t - 1] + Qsqrt * rng.standard_normal(3)
    return OscillatorData(x=s[burn_in:, 0], s=s[burn_in:], A=A, r=r, theta=theta)


def esprit_modes(kernel: np.ndarray, model_order: int = 2):
    """ESPRIT: recover complex modes from a (real) damped kernel tail via Hankel SVD."""
    k = np.asarray(kernel, dtype=float)
    L = len(k)
    n = L // 2
    H = np.array([[k[i + j] for j in range(n)] for i in range(L - n + 1)])
    U, sv, _ = np.linalg.svd(H, full_matrices=False)
    Us = U[:, :model_order]
    U1, U2 = Us[:-1], Us[1:]
    Phi = np.linalg.pinv(U1) @ U2
    eig = np.linalg.eigvals(Phi)
    return eig, sv
