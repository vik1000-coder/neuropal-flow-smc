r"""Level-2 stable Gaussian model (Section 6.2).

Numerically stable fallback: :math:`\mu_i(t) = a_i^\top \psi(t)`,
:math:`\log v_i(t) = b_i^\top \psi(t)`, fit by Gaussian negative log-likelihood with
ridge/elastic-net regularization. Unlike the closed-form estimator this cannot emit an
invalid variance, so it is used for deployment where the natural-parameter model
produces positive ``eta2`` on held-out histories.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class GaussianNLLResult:
    a: np.ndarray  # mean weights [P]
    b: np.ndarray  # log-variance weights [P]
    P: int


class GaussianNLL:
    """Mean + log-variance Gaussian model fit by penalized NLL."""

    def __init__(self, ridge: float = 1e-3, maxiter: int = 500):
        self.ridge = float(ridge)
        self.maxiter = int(maxiter)
        self.result_: GaussianNLLResult | None = None

    def _unpack(self, w, P):
        return w[:P], w[P:]

    def _objective(self, w, Y, Psi):
        P = Psi.shape[1]
        a, b = self._unpack(w, P)
        mu = Psi @ a
        logv = np.clip(Psi @ b, -20.0, 20.0)
        v = np.exp(logv)
        resid = Y - mu
        nll = 0.5 * np.mean(logv + resid ** 2 / v)
        nll += self.ridge * (np.sum(a ** 2) + np.sum(b ** 2))
        # gradient
        dmu = -(resid / v)                      # d nll / d mu (per sample)
        dlogv = 0.5 * (1.0 - resid ** 2 / v)    # d nll / d logv
        ga = Psi.T @ dmu / len(Y) + 2 * self.ridge * a
        gb = Psi.T @ dlogv / len(Y) + 2 * self.ridge * b
        return nll, np.concatenate([ga, gb])

    def fit(self, Y: np.ndarray, Psi: np.ndarray) -> GaussianNLLResult:
        Y = np.asarray(Y, dtype=float).ravel()
        Psi = np.asarray(Psi, dtype=float)
        T, P = Psi.shape
        # warm start: OLS mean, log of residual variance in intercept (col 0)
        a0 = np.linalg.lstsq(Psi, Y, rcond=None)[0]
        b0 = np.zeros(P)
        resid_var = float(np.var(Y - Psi @ a0)) or 1.0
        b0[0] = np.log(resid_var)
        w0 = np.concatenate([a0, b0])
        res = minimize(self._objective, w0, args=(Y, Psi), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.maxiter})
        a, b = self._unpack(res.x, P)
        self.result_ = GaussianNLLResult(a=a, b=b, P=P)
        return self.result_

    def moments(self, Psi: np.ndarray):
        r = self.result_
        if r is None:
            raise RuntimeError("model must be fit before use")
        Psi = np.asarray(Psi, dtype=float)
        mu = Psi @ r.a
        var = np.exp(np.clip(Psi @ r.b, -20.0, 20.0))
        return mu, var

    def nll(self, Y: np.ndarray, Psi: np.ndarray) -> float:
        Y = np.asarray(Y, dtype=float).ravel()
        mu, var = self.moments(Psi)
        return float(0.5 * np.mean(np.log(2 * np.pi * var) + (Y - mu) ** 2 / var))
