r"""Closed-form quadratic-T conditional score-matching estimator (Algorithm 1).

Model (target-wise, ``d = 1``). For target neuron ``i`` with feature vector
:math:`\psi(t) \in \mathbb{R}^P`,

.. math::

    Y_i(t) \mid H_t \sim \mathcal{N}(\mu_i(t), v_i(t)),

parameterized through natural parameters that are *linear* in the features:

.. math::

    \eta_1(t) = \theta_1^\top \psi(t), \qquad \eta_2(t) = \theta_2^\top \psi(t),

with sufficient statistic :math:`T(y) = (y, y^2)`, conditional score
:math:`s_\theta(y,h) = \eta_1(h) + 2\eta_2(h) y`, and

.. math::

    v_\mathrm{eff}(t) = -\frac{1}{2\eta_2(t)}, \quad
    v(t) = v_\mathrm{eff}(t) - \sigma^2, \quad
    \mu(t) = -\frac{\eta_1(t)}{2\eta_2(t)} = \eta_1(t)\, v_\mathrm{eff}(t).

Fitting is a single regularized linear solve (Proposition 6.2): both Hyvärinen and
denoising score matching are quadratic in :math:`\theta`.

Algorithm 1 (denoising / Hyvärinen closed form):

  * ``sigma > 0``:  draw ``zeta ~ N(0, I)``, ``Ytil = Y + sigma*zeta``,
    ``g = (Ytil - Y)/sigma^2``, ``U = [Psi, 2*Ytil*Psi]``, ``A = U'U/T``,
    ``theta = -(A + ridge I)^{-1} U' g / T``.
  * ``sigma = 0``:  ``U = [Psi, 2*Y*Psi]``, ``A = U'U/T``,
    ``c = [0_P, 2*mean(Psi)]``, ``theta = -(A + ridge I)^{-1} c``.

The estimating function per sample (for the HAC sandwich) is the gradient of the
per-sample score-matching loss w.r.t. ``theta``: ``xi_t = U_t (U_t' theta + g_t)``
for ``sigma > 0`` and ``xi_t = U_t (U_t' theta) + (0_P, 2 psi_t)`` for ``sigma = 0``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from ..utils.linalg import ridge_solve
from ..utils.rng import get_rng


@dataclass
class FitResult:
    """Fitted parameters and bookkeeping for one target."""

    theta: np.ndarray            # [2P] = [theta1 (P), theta2 (P)]
    P: int
    sigma: float
    ridge: float
    A: np.ndarray                # [2P, 2P] design gram U'U/T
    xi: np.ndarray               # [T, 2P] per-sample estimating functions
    eta2_min: float = 1e-6
    var_min: float = 1e-6
    invalid_variance_fraction: float = 0.0
    n_samples: int = 0
    sd_y: float = 1.0

    @property
    def theta1(self) -> np.ndarray:
        return self.theta[: self.P]

    @property
    def theta2(self) -> np.ndarray:
        return self.theta[self.P:]


class QuadraticScoreMatcher:
    """Convex closed-form quadratic-T conditional density estimator.

    Parameters
    ----------
    sigma : float
        Denoising corruption level (0 => Hyvärinen score matching).
    ridge : float
        Nonnegative ridge regularization on ``theta``.
    eta2_min, var_min : float
        Numerical clamps applied at *prediction* time only.
    seed : int
        RNG seed for the denoising corruption draw.
    """

    def __init__(self, sigma: float = 0.0, ridge: float = 1e-6,
                 eta2_min: float = 1e-6, var_min: float = 1e-6, seed: int = 0):
        self.sigma = float(sigma)
        self.ridge = float(ridge)
        self.eta2_min = float(eta2_min)
        self.var_min = float(var_min)
        self.seed = seed
        self.result_: FitResult | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, Y: np.ndarray, Psi: np.ndarray,
            sigma: float | None = None, ridge: float | None = None) -> FitResult:
        Y = np.asarray(Y, dtype=float).ravel()
        Psi = np.asarray(Psi, dtype=float)
        if Psi.ndim != 2:
            raise ValueError("Psi must be 2D [T, P]")
        T, P = Psi.shape
        if Y.shape[0] != T:
            raise ValueError(f"Y length {Y.shape[0]} != Psi rows {T}")
        sigma = self.sigma if sigma is None else float(sigma)
        ridge = self.ridge if ridge is None else float(ridge)
        sd_y = float(np.std(Y)) or 1.0

        if sigma > 0:
            rng = get_rng(self.seed)
            zeta = rng.standard_normal(T)
            Ytil = Y + sigma * zeta
            g = (Ytil - Y) / (sigma ** 2)            # = zeta / sigma
            U = np.concatenate([Psi, 2.0 * Ytil[:, None] * Psi], axis=1)  # [T, 2P]
            A = (U.T @ U) / T
            rhs = (U.T @ g) / T
            theta = -ridge_solve(A, rhs, ridge)
            # estimating function xi_t = U_t (U_t' theta + g_t)
            resid = U @ theta + g                     # [T]
            xi = U * resid[:, None]
        else:
            U = np.concatenate([Psi, 2.0 * Y[:, None] * Psi], axis=1)
            A = (U.T @ U) / T
            c = np.concatenate([np.zeros(P), 2.0 * Psi.mean(axis=0)])
            theta = -ridge_solve(A, c, ridge)
            # xi_t = U_t (U_t' theta) + (0_P, 2 psi_t)
            base = U * (U @ theta)[:, None]
            extra = np.concatenate([np.zeros((T, P)), 2.0 * Psi], axis=1)
            xi = base + extra

        res = FitResult(theta=theta, P=P, sigma=sigma, ridge=ridge, A=A, xi=xi,
                        eta2_min=self.eta2_min, var_min=self.var_min,
                        n_samples=T, sd_y=sd_y)
        # record invalid-variance fraction on the training features
        eta1, eta2, mu, var = self._raw_params(res, Psi)
        res.invalid_variance_fraction = float(np.mean(eta2 >= -self.eta2_min))
        self.result_ = res
        return res

    # ------------------------------------------------------------- prediction
    def _raw_params(self, res: FitResult, Psi: np.ndarray):
        """Return (eta1, eta2, mu, var) with prediction-time clamps applied."""
        Psi = np.asarray(Psi, dtype=float)
        eta1 = Psi @ res.theta1
        eta2 = Psi @ res.theta2
        eta2_clamped = np.minimum(eta2, -res.eta2_min)  # keep eta2 <= -eta2_min
        v_eff = -1.0 / (2.0 * eta2_clamped)
        var = v_eff - res.sigma ** 2
        var = np.maximum(var, res.var_min)
        mu = eta1 * v_eff
        return eta1, eta2, mu, var

    def predict_params(self, Psi: np.ndarray):
        """Return ``(eta1, eta2, mu, var)`` for each row of ``Psi``."""
        res = self._require()
        n_invalid = 0
        Psi = np.asarray(Psi, dtype=float)
        eta1 = Psi @ res.theta1
        eta2 = Psi @ res.theta2
        n_invalid = int(np.sum(eta2 >= -res.eta2_min))
        if n_invalid > 0.05 * len(eta2):
            warnings.warn(
                f"invalid eta2 (>= -eta2_min) in {n_invalid}/{len(eta2)} rows "
                f"({100*n_invalid/len(eta2):.1f}%); clamping for prediction",
                RuntimeWarning, stacklevel=2,
            )
        elif n_invalid > 0.01 * len(eta2):
            warnings.warn(
                f"invalid eta2 in {n_invalid}/{len(eta2)} rows; clamping",
                RuntimeWarning, stacklevel=2,
            )
        return self._raw_params(res, Psi)

    def moments(self, Psi: np.ndarray):
        """Return ``(mu, var)``."""
        _, _, mu, var = self.predict_params(Psi)
        return mu, var

    def logpdf(self, Y: np.ndarray, Psi: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=float).ravel()
        _, _, mu, var = self.predict_params(Psi)
        return norm.logpdf(Y, loc=mu, scale=np.sqrt(var))

    def nll(self, Y: np.ndarray, Psi: np.ndarray) -> float:
        """Mean negative log likelihood on held-out data."""
        return float(-np.mean(self.logpdf(Y, Psi)))

    def cdf(self, Y: np.ndarray, Psi: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=float).ravel()
        _, _, mu, var = self.predict_params(Psi)
        return norm.cdf(Y, loc=mu, scale=np.sqrt(var))

    def sample(self, Psi: np.ndarray, n_samples: int = 1, rng=None) -> np.ndarray:
        rng = get_rng(rng)
        _, _, mu, var = self.predict_params(Psi)
        sd = np.sqrt(var)
        draws = rng.standard_normal((len(mu), n_samples))
        return mu[:, None] + sd[:, None] * draws

    def estimating_functions(self, Y=None, Psi=None) -> np.ndarray:
        """Per-sample estimating functions (from the fit), shape ``[T, 2P]``."""
        return self._require().xi

    def _require(self) -> FitResult:
        if self.result_ is None:
            raise RuntimeError("model must be fit before use")
        return self.result_


def sigma_ledger(Y: np.ndarray, Psi: np.ndarray, ridge: float = 1e-6,
                 sigma_fracs=(0.0, 0.25, 0.5, 1.0), seed: int = 0):
    """Run the estimator across the denoising sigma grid (Section 8.3).

    Returns a list of dicts with the corrected-variance readout at the mean history,
    used as a cross-sigma consistency diagnostic. After subtracting ``sigma^2`` the
    variance/gain readouts should agree across ``sigma``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    sd = float(np.std(Y)) or 1.0
    psi_mean = np.asarray(Psi, dtype=float).mean(axis=0, keepdims=True)
    out = []
    for frac in sigma_fracs:
        sigma = frac * sd
        m = QuadraticScoreMatcher(sigma=sigma, ridge=ridge, seed=seed)
        m.fit(Y, Psi)
        _, _, mu, var = m.predict_params(psi_mean)
        out.append({"sigma_frac": frac, "sigma": sigma,
                    "mean_at_center": float(mu[0]), "var_at_center": float(var[0])})
    return out
