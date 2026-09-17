"""Common estimator interface and coefficient bookkeeping."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..capabilities import Capabilities, ResourceProfile
from ..schema import Prediction, SupervisedData


@dataclass
class Standardizer:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray

    @classmethod
    def fit(cls, data: SupervisedData) -> "Standardizer":
        x_mean = np.mean(data.features, axis=0)
        x_scale = np.std(data.features, axis=0)
        y_mean = np.mean(data.targets, axis=0)
        y_scale = np.std(data.targets, axis=0)
        return cls(
            x_mean=x_mean,
            x_scale=np.where(x_scale > 1e-8, x_scale, 1.0),
            y_mean=y_mean,
            y_scale=np.where(y_scale > 1e-8, y_scale, 1.0),
        )

    def x(self, values: np.ndarray) -> np.ndarray:
        return (values - self.x_mean) / self.x_scale

    def y(self, values: np.ndarray) -> np.ndarray:
        return (values - self.y_mean) / self.y_scale

    def inverse_mean(self, values: np.ndarray) -> np.ndarray:
        return values * self.y_scale + self.y_mean

    def inverse_variance(self, values: np.ndarray) -> np.ndarray:
        return values * self.y_scale**2

    def coefficient_to_original(self, coefficient: np.ndarray) -> np.ndarray:
        """Convert [target, feature] standardized coefficients to original units."""

        return coefficient * self.y_scale[:, None] / self.x_scale[None, :]

    def logvariance_coefficient_to_original(self, coefficient: np.ndarray) -> np.ndarray:
        return coefficient / self.x_scale[None, :]


def aggregate_features(
    coefficient: np.ndarray,
    source_index: np.ndarray,
    n_targets: int,
    *,
    signed: bool,
) -> np.ndarray:
    """Aggregate history features into a [target, source] readout."""

    coefficient = np.asarray(coefficient, dtype=float)
    n_sources = int(np.max(source_index)) + 1
    result = np.zeros((n_targets, n_sources))
    for source in range(n_sources):
        columns = np.flatnonzero(source_index == source)
        if signed:
            # Weight recent lags most; this is a declared summary, while the full
            # feature tensor is retained in Prediction.channels.
            weights = 1.0 / (1.0 + np.arange(len(columns)))
            result[:, source] = coefficient[:, columns] @ weights
        else:
            result[:, source] = np.sqrt(np.sum(coefficient[:, columns] ** 2, axis=1))
    return result


class Estimator(ABC):
    name: str
    capabilities = Capabilities(predictive_mean=True, predictive_distribution="normalized")
    resource_profile = ResourceProfile()

    @abstractmethod
    def fit(self, train: SupervisedData, validation: SupervisedData | None = None) -> "Estimator":
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {}
