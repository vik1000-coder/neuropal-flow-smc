"""Feature/target scaling fit on train only (Section 7.5)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scaler:
    """Columnwise standardization, fit on a training slice only.

    ``center_only=True`` subtracts the mean but keeps unit scale (used for the
    quadratic-score design matrix, where features enter linearly and the intercept
    absorbs the mean; centering makes the readout "at the mean history").
    """

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    center_only: bool = False
    eps: float = 1e-8

    def fit(self, X: np.ndarray) -> "Scaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        if self.center_only:
            self.std_ = np.ones(X.shape[1])
        else:
            std = X.std(axis=0)
            self.std_ = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler must be fit before transform")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
            "center_only": self.center_only,
        }
