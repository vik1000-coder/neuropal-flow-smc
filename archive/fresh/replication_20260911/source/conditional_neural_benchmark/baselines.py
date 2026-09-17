from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge


@dataclass
class GaussianBaseline:
    name: str
    coef: np.ndarray | None
    intercept: np.ndarray | None
    chol: np.ndarray
    logdet: float
    residuals: np.ndarray | None = None

    @classmethod
    def persistence(cls, history: np.ndarray, target: np.ndarray) -> "GaussianBaseline":
        mean = history[:, -1, : target.shape[1]]
        residual = target - mean
        cov = LedoitWolf().fit(residual).covariance_.astype(np.float64)
        cov.flat[:: len(cov) + 1] += 1e-6
        chol = np.linalg.cholesky(cov)
        return cls("persistence_full_gaussian", None, None, chol, 2 * np.log(np.diag(chol)).sum())

    @classmethod
    def ridge(
        cls, features: np.ndarray, target: np.ndarray, alpha: float
    ) -> "GaussianBaseline":
        reg = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
        reg.fit(features, target)
        residual = target - reg.predict(features)
        cov = LedoitWolf().fit(residual).covariance_.astype(np.float64)
        cov.flat[:: len(cov) + 1] += 1e-6
        chol = np.linalg.cholesky(cov)
        return cls(
            f"ridge_full_gaussian_a{alpha:g}",
            reg.coef_.astype(np.float32),
            reg.intercept_.astype(np.float32),
            chol,
            2 * np.log(np.diag(chol)).sum(),
        )

    def mean(self, *, history: np.ndarray | None = None, features: np.ndarray | None = None) -> np.ndarray:
        if self.coef is None:
            if history is None:
                raise ValueError("persistence needs history")
            return history[:, -1, : self.chol.shape[0]]
        if features is None:
            raise ValueError("ridge needs features")
        return features @ self.coef.T + self.intercept

    def sample(
        self,
        n_samples: int,
        seed: int,
        *,
        history: np.ndarray | None = None,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        mean = self.mean(history=history, features=features)
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((len(mean), n_samples, mean.shape[1]))
        return (mean[:, None] + noise @ self.chol.T).astype(np.float32)

    def log_prob(
        self,
        target: np.ndarray,
        *,
        history: np.ndarray | None = None,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        mean = self.mean(history=history, features=features)
        z = solve_triangular(self.chol, (target - mean).T, lower=True).T
        return -0.5 * (
            np.square(z).sum(axis=1) + self.logdet + target.shape[1] * math.log(2 * math.pi)
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        values = {"chol": self.chol, "logdet": np.asarray(self.logdet)}
        if self.coef is not None:
            values["coef"] = self.coef
            values["intercept"] = self.intercept
        return values
