"""Capability-correct adapters for SID quadratic and Gaussian-NLL models."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import ndtr

from sid_neuromod.models.gaussian_nll import GaussianNLL
from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher
from sid_neuromod.readouts.mean_gain_tail import readout_vector

from ..capabilities import Capabilities, ResourceProfile
from ..metrics import gaussian_log_prob
from ..schema import Prediction, SupervisedData
from .base import Estimator, Standardizer, aggregate_features


def _design(scaler: Standardizer, features: np.ndarray) -> np.ndarray:
    x = scaler.x(features)
    return np.column_stack([np.ones(len(x)), x])


class SIDQuadratic(Estimator):
    """Target-wise normalized Gaussian SID with Hyvarinen or Gaussian DSM fitting."""

    name = "sid_quadratic"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=(
            "conditional_mean_derivative",
            "conditional_log_variance_derivative",
            "conditional_tail_high_derivative",
            "conditional_shape_tail_derivative",
        ),
        graph_scores=True,
    )
    resource_profile = ResourceProfile("medium_cpu", threads=2, estimated_peak_gb=1.0)

    def __init__(
        self,
        ridge: float = 1e-2,
        sigma_fraction: float = 0.0,
        n_corruptions: int = 1,
        seed: int = 0,
        readout: str = "average",
        tail_threshold: float = 0.35,
    ):
        if ridge < 0 or sigma_fraction < 0 or n_corruptions < 1:
            raise ValueError("invalid SID hyperparameters")
        if readout not in {"average", "center"}:
            raise ValueError("readout must be 'average' or 'center'")
        self.ridge = float(ridge)
        self.sigma_fraction = float(sigma_fraction)
        self.n_corruptions = int(n_corruptions)
        self.seed = int(seed)
        self.readout = readout
        self.tail_threshold = float(tail_threshold)

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.scaler_ = Standardizer.fit(train)
        y = self.scaler_.y(train.targets)
        psi = _design(self.scaler_, train.features)
        self.models_: list[QuadraticScoreMatcher] = []
        self.fit_results_ = []
        for target in range(y.shape[1]):
            clean = y[:, target]
            design = psi
            if self.n_corruptions > 1 and self.sigma_fraction > 0:
                clean = np.tile(clean, self.n_corruptions)
                design = np.tile(design, (self.n_corruptions, 1))
            model = QuadraticScoreMatcher(
                sigma=self.sigma_fraction,
                ridge=self.ridge,
                seed=self.seed + 104_729 * target,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fit = model.fit(clean, design)
            self.models_.append(model)
            self.fit_results_.append(fit)
        self.training_design_ = psi
        self.source_index_ = train.source_index.copy()
        self.tail_low_ = (
            -self.tail_threshold - self.scaler_.y_mean
        ) / self.scaler_.y_scale
        self.tail_high_ = (
            self.tail_threshold - self.scaler_.y_mean
        ) / self.scaler_.y_scale
        self.channels_ = self._readouts(psi, train.targets.shape[1])
        return self

    def _readouts(self, psi: np.ndarray, n_targets: int) -> dict[str, np.ndarray]:
        n_features = psi.shape[1] - 1
        channels_z = {
            "mean": np.zeros((n_targets, n_features)),
            "logvariance": np.zeros((n_targets, n_features)),
            "tail_high": np.zeros((n_targets, n_features)),
            "tail_low": np.zeros((n_targets, n_features)),
        }
        evaluation = psi.mean(axis=0, keepdims=True) if self.readout == "center" else psi
        indices = np.arange(1, psi.shape[1])
        for target, fit in enumerate(self.fit_results_):
            channels_z["mean"][target] = readout_vector(fit, evaluation, indices, "mean")
            channels_z["logvariance"][target] = readout_vector(
                fit, evaluation, indices, "gain_log_variance"
            )
            channels_z["tail_high"][target] = readout_vector(
                fit, evaluation, indices, "tail_high", q_high=self.tail_high_[target]
            )
            channels_z["tail_low"][target] = readout_vector(
                fit, evaluation, indices, "tail_low", q_low=self.tail_low_[target]
            )

        mean_feature = channels_z["mean"] * self.scaler_.y_scale[:, None] / self.scaler_.x_scale
        logvar_feature = channels_z["logvariance"] / self.scaler_.x_scale
        tail_high_feature = channels_z["tail_high"] / self.scaler_.x_scale
        tail_low_feature = channels_z["tail_low"] / self.scaler_.x_scale
        shape_tail_feature = np.zeros_like(tail_high_feature)
        return {
            "conditional_mean_derivative_feature": mean_feature,
            "conditional_log_variance_derivative_feature": logvar_feature,
            "conditional_tail_high_derivative_feature": tail_high_feature,
            "conditional_tail_low_derivative_feature": tail_low_feature,
            "conditional_shape_tail_derivative_feature": shape_tail_feature,
            "conditional_mean_derivative": aggregate_features(
                mean_feature, self.source_index_, n_targets, signed=True
            ),
            "conditional_log_variance_derivative": aggregate_features(
                logvar_feature, self.source_index_, n_targets, signed=True
            ),
            "conditional_log_variance_strength": aggregate_features(
                logvar_feature, self.source_index_, n_targets, signed=False
            ),
            "conditional_tail_high_derivative": aggregate_features(
                tail_high_feature, self.source_index_, n_targets, signed=True
            ),
            "conditional_tail_low_derivative": aggregate_features(
                tail_low_feature, self.source_index_, n_targets, signed=True
            ),
            "conditional_shape_tail_derivative": aggregate_features(
                shape_tail_feature, self.source_index_, n_targets, signed=True
            ),
        }

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        psi = _design(self.scaler_, data.features)
        mean_z = np.zeros_like(data.targets)
        variance_z = np.zeros_like(data.targets)
        invalid = []
        near_boundary = []
        for target, (model, fit) in enumerate(zip(self.models_, self.fit_results_)):
            eta2 = psi @ fit.theta2
            invalid.append(np.mean(eta2 >= -fit.eta2_min))
            near_boundary.append(np.mean(eta2 > -0.01))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_z[:, target], variance_z[:, target] = model.moments(psi)
        mean = self.scaler_.inverse_mean(mean_z)
        variance = self.scaler_.inverse_variance(variance_z)
        log_prob = gaussian_log_prob(data.targets, mean, variance)
        cdf = ndtr((data.targets - mean) / np.sqrt(np.maximum(variance, 1e-10)))
        samples = None
        if n_samples:
            rng = np.random.default_rng(self.seed + 1)
            samples = mean[:, None, :] + np.sqrt(variance[:, None, :]) * rng.standard_normal(
                (len(mean), n_samples, mean.shape[1])
            )
        extreme_variance = np.mean((variance_z > 100.0) | (variance_z < 1e-4))
        return Prediction(
            mean=mean,
            variance=variance,
            log_prob=log_prob,
            cdf=cdf,
            samples=samples,
            channels=self.channels_,
            metadata={
                "invalid_variance_fraction": float(np.mean(invalid)),
                "near_natural_variance_boundary_fraction": float(np.mean(near_boundary)),
                "extreme_variance_fraction": float(extreme_variance),
                "standardized_variance_p99": float(np.quantile(variance_z, 0.99)),
                "numerical_valid": bool(
                    np.mean(invalid) <= 0.05
                    and extreme_variance <= 0.01
                    and np.isfinite(mean_z).all()
                    and np.isfinite(variance_z).all()
                ),
            },
        )

    def metadata(self):
        return {
            "ridge": self.ridge,
            "sigma_fraction": self.sigma_fraction,
            "n_corruptions": self.n_corruptions,
            "readout": self.readout,
            "semantics": "conditional Gaussian score; sigma is standardized target units",
            "tail_threshold_physical": self.tail_threshold,
        }


class GaussianNLLAdapter(Estimator):
    """Stable normalized conditional Gaussian model from sid_neuromod."""

    name = "gaussian_nll"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=(
            "conditional_mean_derivative",
            "conditional_log_variance_derivative",
        ),
        graph_scores=True,
    )
    resource_profile = ResourceProfile("medium_cpu", threads=2, estimated_peak_gb=1.0)

    def __init__(self, ridge: float = 1e-3, maxiter: int = 300):
        self.ridge = float(ridge)
        self.maxiter = int(maxiter)

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.scaler_ = Standardizer.fit(train)
        psi = _design(self.scaler_, train.features)
        y = self.scaler_.y(train.targets)
        self.models_ = []
        mean_coef = np.zeros((y.shape[1], train.features.shape[1]))
        logvar_coef = np.zeros_like(mean_coef)
        for target in range(y.shape[1]):
            model = GaussianNLL(ridge=self.ridge, maxiter=self.maxiter)
            result = model.fit(y[:, target], psi)
            self.models_.append(model)
            mean_coef[target] = result.a[1:] * self.scaler_.y_scale[target] / self.scaler_.x_scale
            logvar_coef[target] = result.b[1:] / self.scaler_.x_scale
        self.channels_ = {
            "conditional_mean_derivative_feature": mean_coef,
            "conditional_log_variance_derivative_feature": logvar_coef,
            "conditional_mean_derivative": aggregate_features(
                mean_coef, train.source_index, y.shape[1], signed=True
            ),
            "conditional_log_variance_derivative": aggregate_features(
                logvar_coef, train.source_index, y.shape[1], signed=True
            ),
            "conditional_log_variance_strength": aggregate_features(
                logvar_coef, train.source_index, y.shape[1], signed=False
            ),
        }
        return self

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        psi = _design(self.scaler_, data.features)
        mean_z = np.zeros_like(data.targets)
        variance_z = np.zeros_like(data.targets)
        for target, model in enumerate(self.models_):
            mean_z[:, target], variance_z[:, target] = model.moments(psi)
        mean = self.scaler_.inverse_mean(mean_z)
        variance = self.scaler_.inverse_variance(variance_z)
        samples = None
        if n_samples:
            rng = np.random.default_rng(0)
            samples = mean[:, None, :] + np.sqrt(variance[:, None, :]) * rng.standard_normal(
                (len(mean), n_samples, mean.shape[1])
            )
        return Prediction(
            mean=mean,
            variance=variance,
            log_prob=gaussian_log_prob(data.targets, mean, variance),
            cdf=ndtr((data.targets - mean) / np.sqrt(np.maximum(variance, 1e-10))),
            samples=samples,
            channels=self.channels_,
        )

    def metadata(self):
        return {"ridge": self.ridge, "maxiter": self.maxiter}
