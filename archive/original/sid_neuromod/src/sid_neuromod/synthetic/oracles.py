r"""Exact / filtering-theoretic oracles for the synthetic systems.

E1 (hidden thermostat). The variance-lag kernel has a steady-state Kalman oracle. On
:math:`z_t = \log x_t^2 = w_t + \log\varepsilon_t^2`, a noisy observation of the latent
AR(1) ``w`` with observation-noise variance :math:`\pi^2/2` (the variance of
:math:`\log\chi^2_1`), the steady-state one-step predictor of ``w`` from the ``z``
history has impulse response

.. math::

   g_u = a\,K_g\,r^{u-1}, \qquad r = a(1 - K_g),\quad u = 1,2,\dots,

with :math:`K_g` the steady-state Kalman gain. Since
:math:`\log \mathrm{Var}(x_{t+1}\mid h) \approx \mathbb{E}[w_{t+1}\mid h] + \text{const}`,
this is exactly the oracle for the gain (log-variance) kernel.
"""
from __future__ import annotations

import numpy as np

LOG_CHI2_1_VAR = np.pi ** 2 / 2.0  # variance of log(chi^2_1)


def steady_state_kalman_gain(a: float, q: float, r_obs: float) -> float:
    r"""Steady-state Kalman gain for ``w_{t+1}=a w_t + noise (var q)``, obs noise ``r_obs``.

    Solves the scalar Riccati :math:`P^2 + P(R(1-a^2)-Q) - QR = 0` for the positive
    prior variance ``P``, then ``K = P/(P+R)``.
    """
    Q, R = float(q), float(r_obs)
    b_coef = R * (1 - a ** 2) - Q
    disc = b_coef ** 2 + 4.0 * Q * R
    P = (-b_coef + np.sqrt(disc)) / 2.0
    return float(P / (P + R))


def thermostat_variance_kernel(a: float = 0.9, b: float = 0.4, L: int = 12,
                               r_obs: float = LOG_CHI2_1_VAR) -> np.ndarray:
    r"""Oracle gain (log-variance) kernel ``g_u = a K_g r^{u-1}`` for ``u=1..L``."""
    Kg = steady_state_kalman_gain(a, b ** 2, r_obs)
    r = a * (1.0 - Kg)
    u = np.arange(L)
    return a * Kg * (r ** u)


def thermostat_contraction(a: float = 0.9, b: float = 0.4,
                           r_obs: float = LOG_CHI2_1_VAR) -> float:
    """Kalman contraction ``r = a(1 - K_g)`` (geometric decay ratio of the kernel)."""
    Kg = steady_state_kalman_gain(a, b ** 2, r_obs)
    return float(a * (1.0 - Kg))


# ---------------------------------------------------------------- E2 OU oracle
def ou_increment_stats(gamma: float, D: float, R: float, dt: float):
    r"""Exact increment statistics of a noisy OU process (Lemma 7.4).

    ``dx = -gamma x dt + sqrt(2D) dW`` observed as ``x_hat = x + v``, ``v ~ (0, R)``.
    Returns ``(var_increment, cov_adjacent_increments)`` for sampling interval ``dt``:

    .. math::

       \mathrm{Var}(\hat x_{t+\Delta}-\hat x_t)
           = 2\sigma_\infty^2(1-e^{-\gamma\Delta}) + 2R,\\
       \mathrm{Cov}(\hat x_{t+2\Delta}-\hat x_{t+\Delta},\hat x_{t+\Delta}-\hat x_t)
           = -\sigma_\infty^2(1-e^{-\gamma\Delta})^2 - R.
    """
    sig2 = D / gamma
    e = np.exp(-gamma * dt)
    var_inc = 2 * sig2 * (1 - e) + 2 * R
    cov_adj = -sig2 * (1 - e) ** 2 - R
    return float(var_inc), float(cov_adj)


def ou_recover_gamma_D_R(dts, var_incs, cov_adjs):
    r"""Recover ``(gamma, D, R)`` from increment statistics across sampling intervals.

    Uses the curvature-corrected basis: fit ``var_inc = 2 sigma^2 (1-e^{-gamma dt}) + 2R``
    and ``cov_adj = -sigma^2 (1-e^{-gamma dt})^2 - R`` jointly over the ``dt`` menu by a
    small nonlinear least squares. Returns ``dict(gamma, D, R)``.
    """
    from scipy.optimize import least_squares

    dts = np.asarray(dts, dtype=float)
    var_incs = np.asarray(var_incs, dtype=float)
    cov_adjs = np.asarray(cov_adjs, dtype=float)

    def resid(p):
        gamma, sig2, R = p
        e = np.exp(-gamma * dts)
        vi = 2 * sig2 * (1 - e) + 2 * R
        ca = -sig2 * (1 - e) ** 2 - R
        return np.concatenate([vi - var_incs, ca - cov_adjs])

    p0 = np.array([1.0, max(var_incs.mean(), 1e-3), max(-cov_adjs.min(), 1e-3)])
    sol = least_squares(resid, p0, bounds=([1e-4, 1e-6, 0.0], [100, 1e4, 1e4]))
    gamma, sig2, R = sol.x
    return {"gamma": float(gamma), "D": float(gamma * sig2), "R": float(R)}
