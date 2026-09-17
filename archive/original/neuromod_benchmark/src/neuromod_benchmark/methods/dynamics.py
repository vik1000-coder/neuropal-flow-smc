"""Episode-safe dynamical-system and directed-prediction baselines."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures

from ..capabilities import Capabilities, ResourceProfile, UnsupportedCapability
from ..metrics import gaussian_log_prob
from ..schema import Prediction, SupervisedData
from .base import Estimator, Standardizer, aggregate_features


def _gaussian_prediction(
    mean: np.ndarray,
    variance: np.ndarray,
    data: SupervisedData,
    channels: dict[str, np.ndarray],
    seed: int,
    n_samples: int,
) -> Prediction:
    variance = np.maximum(np.asarray(variance), 1e-8)
    samples = None
    if n_samples:
        rng = np.random.default_rng(seed)
        samples = mean[:, None, :] + np.sqrt(variance[:, None, :]) * rng.standard_normal(
            (len(mean), n_samples, mean.shape[1])
        )
    return Prediction(
        mean=mean,
        variance=variance,
        log_prob=gaussian_log_prob(data.targets, mean, variance),
        cdf=ndtr((data.targets - mean) / np.sqrt(variance)),
        samples=samples,
        channels=channels,
    )


class LaggedCorrelation(Estimator):
    """Marginal lagged association. It intentionally has no density capability."""

    name = "lagged_correlation"
    capabilities = Capabilities(
        effect_channels=("marginal_lagged_association",), graph_scores=True
    )

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        n_targets = train.targets.shape[1]
        n_sources = int(np.max(train.source_index)) + 1
        score = np.zeros((n_targets, n_sources))
        # Use the most recent feature for each source. Other lags are separate
        # configurations rather than silently maximized over.
        for source in range(n_sources):
            column = np.flatnonzero(train.source_index == source)[0]
            x = train.features[:, column]
            for target in range(n_targets):
                y = train.targets[:, target]
                score[target, source] = (
                    np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0.0
                )
        self.channels_ = {"marginal_lagged_association": score}
        return self

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        del data, n_samples
        raise UnsupportedCapability(self.name, "predictive_distribution")


class ConditionalGrangerRidge(Estimator):
    """Source-deletion predictive improvement on a held-out episode split."""

    name = "conditional_granger_ridge"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=("linear_transition_coefficient",),
        graph_scores=True,
    )
    resource_profile = ResourceProfile("medium_cpu", threads=2, estimated_peak_gb=1.0)

    def __init__(self, ridge: float = 1.0):
        self.ridge = float(ridge)

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        if validation is None or not len(validation.targets):
            raise ValueError("ConditionalGrangerRidge requires held-out validation episodes")
        self.scaler_ = Standardizer.fit(train)
        x_train = self.scaler_.x(train.features)
        y_train = self.scaler_.y(train.targets)
        x_val = self.scaler_.x(validation.features)
        y_val = self.scaler_.y(validation.targets)
        self.full_ = Ridge(alpha=self.ridge).fit(x_train, y_train)
        residual_train = y_train - self.full_.predict(x_train)
        self.residual_variance_ = np.maximum(np.var(residual_train, axis=0), 1e-6)
        full_error = (y_val - self.full_.predict(x_val)) ** 2
        n_targets = y_train.shape[1]
        n_sources = int(np.max(train.source_index)) + 1
        score = np.zeros((n_targets, n_sources))
        for source in range(n_sources):
            keep = train.source_index != source
            reduced = Ridge(alpha=self.ridge).fit(x_train[:, keep], y_train)
            reduced_error = (y_val - reduced.predict(x_val[:, keep])) ** 2
            # Positive values indicate source-deletion degradation. Negative values
            # are retained as a diagnostic rather than clipped into fake evidence.
            score[:, source] = np.mean(reduced_error - full_error, axis=0)
        coefficient = self.scaler_.coefficient_to_original(self.full_.coef_)
        self.channels_ = {
            "linear_transition_coefficient_feature": coefficient,
            "linear_transition_coefficient": aggregate_features(
                coefficient, train.source_index, n_targets, signed=True
            ),
            "conditional_granger_validation_improvement": score,
        }
        return self

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        mean_z = self.full_.predict(self.scaler_.x(data.features))
        mean = self.scaler_.inverse_mean(mean_z)
        variance = np.broadcast_to(
            self.scaler_.inverse_variance(self.residual_variance_), mean.shape
        ).copy()
        return _gaussian_prediction(mean, variance, data, self.channels_, 0, n_samples)

    def metadata(self):
        return {"ridge": self.ridge, "graph_semantics": "held-out source-deletion MSE"}


class SparseTransition(Estimator):
    """Lagged sparse transition model; explicitly not mislabeled as DYNOTEARS."""

    name = "lagged_sparse_transition"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=("linear_transition_coefficient",),
        graph_scores=True,
        equations=True,
    )

    def __init__(self, alpha: float = 1e-2, maxiter: int = 5_000):
        self.alpha = float(alpha)
        self.maxiter = int(maxiter)

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.scaler_ = Standardizer.fit(train)
        x = self.scaler_.x(train.features)
        y = self.scaler_.y(train.targets)
        self.models_ = []
        coefficient = np.zeros((y.shape[1], x.shape[1]))
        prediction = np.zeros_like(y)
        for target in range(y.shape[1]):
            model = Lasso(alpha=self.alpha, max_iter=self.maxiter).fit(x, y[:, target])
            self.models_.append(model)
            coefficient[target] = model.coef_
            prediction[:, target] = model.predict(x)
        self.residual_variance_ = np.maximum(np.var(y - prediction, axis=0), 1e-6)
        coefficient_original = self.scaler_.coefficient_to_original(coefficient)
        self.channels_ = {
            "linear_transition_coefficient_feature": coefficient_original,
            "linear_transition_coefficient": aggregate_features(
                coefficient_original, train.source_index, y.shape[1], signed=True
            ),
            "linear_transition_strength": aggregate_features(
                coefficient_original, train.source_index, y.shape[1], signed=False
            ),
        }
        return self

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        x = self.scaler_.x(data.features)
        mean_z = np.column_stack([model.predict(x) for model in self.models_])
        mean = self.scaler_.inverse_mean(mean_z)
        variance = np.broadcast_to(
            self.scaler_.inverse_variance(self.residual_variance_), mean.shape
        ).copy()
        return _gaussian_prediction(mean, variance, data, self.channels_, 0, n_samples)

    def metadata(self):
        return {"alpha": self.alpha, "maxiter": self.maxiter, "acyclicity": "not imposed"}


class DiscreteSINDy(Estimator):
    """Sparse discrete transition map with a declared feature library.

    This avoids passing future states to a continuous-time ``x_dot`` API. A separate
    continuous-time adapter would need derivative estimation and is a different task.
    """

    name = "sindy_discrete"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=("conditional_mean_derivative",),
        graph_scores=True,
        equations=True,
    )
    resource_profile = ResourceProfile("medium_cpu", threads=2, estimated_peak_gb=1.5)

    def __init__(
        self,
        degree: int = 2,
        threshold: float = 1e-2,
        ridge: float = 1e-3,
        max_iter: int = 10,
    ):
        if degree not in {1, 2, 3}:
            raise ValueError("degree must be 1, 2, or 3")
        self.degree = int(degree)
        self.threshold = float(threshold)
        self.ridge = float(ridge)
        self.max_iter = int(max_iter)

    def _stlsq(self, theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_features = theta.shape[1]
        gram = theta.T @ theta + self.ridge * np.eye(n_features)
        coefficient = np.linalg.solve(gram, theta.T @ y)
        for _ in range(self.max_iter):
            old_support = np.abs(coefficient) >= self.threshold
            for target in range(y.shape[1]):
                keep = old_support[:, target]
                coefficient[:, target] = 0.0
                if np.any(keep):
                    local = theta[:, keep]
                    coefficient[keep, target] = np.linalg.solve(
                        local.T @ local + self.ridge * np.eye(np.sum(keep)),
                        local.T @ y[:, target],
                    )
            if np.array_equal(old_support, np.abs(coefficient) >= self.threshold):
                break
        coefficient[np.abs(coefficient) < self.threshold] = 0.0
        return coefficient

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.scaler_ = Standardizer.fit(train)
        x = self.scaler_.x(train.features)
        y = self.scaler_.y(train.targets)
        self.library_ = PolynomialFeatures(degree=self.degree, include_bias=True)
        theta = self.library_.fit_transform(x)
        self.coefficient_ = self._stlsq(theta, y)
        prediction = theta @ self.coefficient_
        self.residual_variance_ = np.maximum(np.var(y - prediction, axis=0), 1e-6)
        jacobian = self._average_jacobian(train.features)
        self.channels_ = {
            "conditional_mean_derivative_feature": jacobian,
            "conditional_mean_derivative": aggregate_features(
                jacobian, train.source_index, y.shape[1], signed=True
            ),
            "conditional_mean_strength": aggregate_features(
                jacobian, train.source_index, y.shape[1], signed=False
            ),
            "equation_coefficients": self.coefficient_.T,
        }
        return self

    def _predict_mean(self, features: np.ndarray) -> np.ndarray:
        theta = self.library_.transform(self.scaler_.x(features))
        return self.scaler_.inverse_mean(theta @ self.coefficient_)

    def _average_jacobian(self, features: np.ndarray, max_rows: int = 256) -> np.ndarray:
        if len(features) > max_rows:
            indices = np.linspace(0, len(features) - 1, max_rows, dtype=int)
            features = features[indices]
        jacobian = np.zeros((self.scaler_.y_mean.size, features.shape[1]))
        step = 1e-4 * np.maximum(self.scaler_.x_scale, 1.0)
        for column in range(features.shape[1]):
            plus = features.copy()
            minus = features.copy()
            plus[:, column] += step[column]
            minus[:, column] -= step[column]
            derivative = (self._predict_mean(plus) - self._predict_mean(minus)) / (
                2 * step[column]
            )
            jacobian[:, column] = np.mean(derivative, axis=0)
        return jacobian

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        mean = self._predict_mean(data.features)
        variance = np.broadcast_to(
            self.scaler_.inverse_variance(self.residual_variance_), mean.shape
        ).copy()
        return _gaussian_prediction(mean, variance, data, self.channels_, 0, n_samples)

    def metadata(self):
        return {
            "degree": self.degree,
            "threshold": self.threshold,
            "ridge": self.ridge,
            "formulation": "discrete transition map",
            "library_features": tuple(self.library_.get_feature_names_out()),
        }
