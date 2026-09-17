"""Normalized multivariate Gaussian baselines with explicit covariance laws."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from sklearn.linear_model import Ridge

from ..capabilities import Capabilities, ResourceProfile
from ..schema import Prediction, SupervisedData
from .base import Estimator, Standardizer, aggregate_features


def _project_correlation(matrix: np.ndarray, eigen_floor: float = 1e-4) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    positive = (vectors * np.maximum(values, eigen_floor)[None, :]) @ vectors.T
    scale = np.sqrt(np.maximum(np.diag(positive), eigen_floor))
    return positive / scale[:, None] / scale[None, :]


def _multivariate_prediction(
    mean: np.ndarray,
    covariance: np.ndarray,
    data: SupervisedData,
    channels: dict[str, np.ndarray],
    *,
    seed: int,
    n_samples: int,
) -> Prediction:
    n_rows, n_targets = mean.shape
    if covariance.shape != (n_rows, n_targets, n_targets):
        raise ValueError("covariance must have shape [row,target,target]")
    variance = np.diagonal(covariance, axis1=1, axis2=2).copy()
    centered = data.targets - mean
    joint_log_prob = np.empty(n_rows, dtype=float)
    factors = np.empty_like(covariance)
    for row in range(n_rows):
        factor = np.linalg.cholesky(covariance[row])
        factors[row] = factor
        solved = np.linalg.solve(factor, centered[row])
        logdet = 2.0 * np.sum(np.log(np.diag(factor)))
        joint_log_prob[row] = -0.5 * (
            n_targets * np.log(2.0 * np.pi) + logdet + solved @ solved
        )
    # The benchmark's NLL convention is per scalar target. Repeating joint/N keeps
    # multivariate and factorized methods on that common unit.
    log_prob = np.broadcast_to(
        (joint_log_prob / n_targets)[:, None], mean.shape
    ).copy()
    samples = None
    if n_samples:
        standard = np.random.default_rng(seed).standard_normal(
            (n_rows, n_samples, n_targets)
        )
        samples = mean[:, None, :] + np.einsum("rij,rsj->rsi", factors, standard)
    return Prediction(
        mean=mean,
        variance=variance,
        log_prob=log_prob,
        cdf=ndtr(centered / np.sqrt(variance)),
        samples=samples,
        channels=channels,
        metadata={
            "joint_density": True,
            "joint_nll_unit": "per target dimension",
            "conditional_covariance": covariance,
        },
    )


class FullCovarianceRidge(Estimator):
    """Linear conditional mean with a shrunk constant full residual covariance."""

    name = "ridge_full_covariance"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=(
            "linear_transition_coefficient",
            "conditional_correlation_derivative",
        ),
        graph_scores=True,
    )
    resource_profile = ResourceProfile("medium_cpu", threads=1, estimated_peak_gb=1.0)

    def __init__(self, ridge: float = 1.0, covariance_shrinkage: float = .1, seed: int = 0):
        if not 0 <= covariance_shrinkage <= 1:
            raise ValueError("covariance_shrinkage must lie in [0,1]")
        self.ridge = float(ridge)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.seed = int(seed)

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.scaler_ = Standardizer.fit(train)
        x, y = self.scaler_.x(train.features), self.scaler_.y(train.targets)
        self.mean_model_ = Ridge(alpha=self.ridge).fit(x, y)
        residual = y - self.mean_model_.predict(x)
        empirical = np.cov(residual, rowvar=False, ddof=1)
        diagonal = np.diag(np.diag(empirical))
        covariance = (
            (1.0 - self.covariance_shrinkage) * empirical
            + self.covariance_shrinkage * diagonal
            + 1e-5 * np.eye(y.shape[1])
        )
        self.covariance_z_ = _project_correlation(
            covariance
            / np.sqrt(np.diag(covariance))[:, None]
            / np.sqrt(np.diag(covariance))[None, :]
        ) * np.sqrt(np.diag(covariance))[:, None] * np.sqrt(np.diag(covariance))[None, :]
        coefficient = self.scaler_.coefficient_to_original(self.mean_model_.coef_)
        n = y.shape[1]
        self.channels_ = {
            "linear_transition_coefficient_feature": coefficient,
            "linear_transition_coefficient": aggregate_features(
                coefficient, train.source_index, n, signed=True
            ),
            "residual_correlation": np.corrcoef(residual, rowvar=False),
            "conditional_correlation_derivative_feature": np.zeros(
                (n, n, train.features.shape[1])
            ),
        }
        return self

    def _mean(self, features: np.ndarray) -> np.ndarray:
        return self.scaler_.inverse_mean(
            self.mean_model_.predict(self.scaler_.x(features))
        )

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        mean = self._mean(data.features)
        scale = self.scaler_.y_scale
        covariance = self.covariance_z_ * scale[:, None] * scale[None, :]
        covariance = np.broadcast_to(covariance, (len(mean),) + covariance.shape).copy()
        return _multivariate_prediction(
            mean,
            covariance,
            data,
            self.channels_,
            seed=self.seed + 1,
            n_samples=n_samples,
        )

    def metadata(self):
        return {
            "ridge": self.ridge,
            "covariance_shrinkage": self.covariance_shrinkage,
            "covariance_law": "constant full residual covariance",
            "orientation": "[target, source]",
        }


class ConditionalCovarianceRidge(Estimator):
    """Linear mean/log-variance/pair-correlation model with SPD projection."""

    name = "conditional_covariance_ridge"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=(
            "linear_transition_coefficient",
            "conditional_log_variance_derivative",
            "conditional_covariance_derivative",
            "conditional_correlation_derivative",
        ),
        graph_scores=True,
    )
    resource_profile = ResourceProfile("medium_cpu", threads=1, estimated_peak_gb=1.5)

    def __init__(
        self,
        mean_ridge: float = 1.0,
        variance_ridge: float = 10.0,
        correlation_ridge: float = 10.0,
        correlation_scale: float = .45,
        variance_floor: float = .03,
        seed: int = 0,
    ):
        if not 0 < correlation_scale < 1:
            raise ValueError("correlation_scale must lie in (0,1)")
        self.mean_ridge = float(mean_ridge)
        self.variance_ridge = float(variance_ridge)
        self.correlation_ridge = float(correlation_ridge)
        self.correlation_scale = float(correlation_scale)
        self.variance_floor = float(variance_floor)
        self.seed = int(seed)

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.scaler_ = Standardizer.fit(train)
        x, y = self.scaler_.x(train.features), self.scaler_.y(train.targets)
        self.mean_model_ = Ridge(alpha=self.mean_ridge).fit(x, y)
        residual = y - self.mean_model_.predict(x)
        self.variance_model_ = Ridge(alpha=self.variance_ridge).fit(
            x, np.log(residual**2 + self.variance_floor)
        )
        fitted_variance = np.exp(self.variance_model_.predict(x))
        standardized = residual / np.sqrt(np.maximum(fitted_variance, 1e-6))
        self.pairs_ = tuple(
            (left, right)
            for left in range(y.shape[1])
            for right in range(left + 1, y.shape[1])
        )
        pair_targets = np.column_stack(
            [np.clip(standardized[:, left] * standardized[:, right], -4.0, 4.0)
             for left, right in self.pairs_]
        )
        self.correlation_model_ = Ridge(alpha=self.correlation_ridge).fit(x, pair_targets)
        coefficient = self.scaler_.coefficient_to_original(self.mean_model_.coef_)
        logvariance = self.scaler_.logvariance_coefficient_to_original(
            self.variance_model_.coef_
        )
        n = y.shape[1]
        covariance_jac, correlation_jac = self._average_covariance_jacobian(train.features)
        self.channels_ = {
            "linear_transition_coefficient_feature": coefficient,
            "linear_transition_coefficient": aggregate_features(
                coefficient, train.source_index, n, signed=True
            ),
            "conditional_log_variance_derivative_feature": logvariance,
            "conditional_log_variance_derivative": aggregate_features(
                logvariance, train.source_index, n, signed=True
            ),
            "conditional_covariance_derivative_feature": covariance_jac,
            "conditional_correlation_derivative_feature": correlation_jac,
        }
        return self

    def _covariance_z(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.scaler_.x(features)
        variance = np.maximum(np.exp(self.variance_model_.predict(x)), 1e-6)
        pair_values = self.correlation_scale * np.tanh(self.correlation_model_.predict(x))
        n_rows, n = len(x), variance.shape[1]
        correlations = np.broadcast_to(np.eye(n), (n_rows, n, n)).copy()
        for pair, (left, right) in enumerate(self.pairs_):
            correlations[:, left, right] = pair_values[:, pair]
            correlations[:, right, left] = pair_values[:, pair]
        for row in range(n_rows):
            correlations[row] = _project_correlation(correlations[row])
        sd = np.sqrt(variance)
        covariance = correlations * sd[:, :, None] * sd[:, None, :]
        return covariance, correlations

    def _average_covariance_jacobian(
        self, features: np.ndarray, max_rows: int = 192
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(features) > max_rows:
            features = features[np.linspace(0, len(features) - 1, max_rows, dtype=int)]
        n = self.scaler_.y_mean.size
        covariance_jac = np.zeros((n, n, features.shape[1]))
        correlation_jac = np.zeros_like(covariance_jac)
        step = 1e-4 * np.maximum(self.scaler_.x_scale, 1.0)
        for column in range(features.shape[1]):
            plus, minus = features.copy(), features.copy()
            plus[:, column] += step[column]
            minus[:, column] -= step[column]
            covariance_plus, correlation_plus = self._covariance_z(plus)
            covariance_minus, correlation_minus = self._covariance_z(minus)
            covariance_jac[:, :, column] = np.mean(
                (covariance_plus - covariance_minus) / (2.0 * step[column]), axis=0
            )
            correlation_jac[:, :, column] = np.mean(
                (correlation_plus - correlation_minus) / (2.0 * step[column]), axis=0
            )
        scale = self.scaler_.y_scale
        covariance_jac *= scale[:, None, None] * scale[None, :, None]
        return covariance_jac, correlation_jac

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        x = self.scaler_.x(data.features)
        mean = self.scaler_.inverse_mean(self.mean_model_.predict(x))
        covariance_z, _ = self._covariance_z(data.features)
        scale = self.scaler_.y_scale
        covariance = covariance_z * scale[None, :, None] * scale[None, None, :]
        return _multivariate_prediction(
            mean,
            covariance,
            data,
            self.channels_,
            seed=self.seed + 1,
            n_samples=n_samples,
        )

    def metadata(self):
        return {
            "mean_ridge": self.mean_ridge,
            "variance_ridge": self.variance_ridge,
            "correlation_ridge": self.correlation_ridge,
            "correlation_scale": self.correlation_scale,
            "variance_floor": self.variance_floor,
            "spd_parameterization": "pairwise tanh followed by eigenvalue-floor projection",
            "orientation": "[target, source]",
        }


__all__ = ["ConditionalCovarianceRidge", "FullCovarianceRidge"]
