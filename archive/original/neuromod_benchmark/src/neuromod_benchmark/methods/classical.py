"""Transparent classical baselines and shrinkage history bridges."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge

from ..capabilities import Capabilities
from ..metrics import gaussian_log_prob
from ..schema import Prediction, SupervisedData
from .base import Estimator, Standardizer, aggregate_features


class ConstantGaussian(Estimator):
    name = "constant_gaussian"
    capabilities = Capabilities(predictive_mean=True, predictive_distribution="normalized")

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.mean_ = np.mean(train.targets, axis=0)
        self.variance_ = np.maximum(np.var(train.targets, axis=0), 1e-6)
        return self

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        mean = np.broadcast_to(self.mean_, data.targets.shape).copy()
        variance = np.broadcast_to(self.variance_, data.targets.shape).copy()
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
            cdf=ndtr((data.targets - mean) / np.sqrt(variance)),
            samples=samples,
            metadata={"distribution_family": "gaussian"},
        )


class RidgeGaussian(Estimator):
    name = "ridge_var"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=("linear_transition_coefficient",),
        graph_scores=True,
    )

    def __init__(self, ridge: float = 1.0):
        self.ridge = float(ridge)

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.scaler_ = Standardizer.fit(train)
        x = self.scaler_.x(train.features)
        y = self.scaler_.y(train.targets)
        self.model_ = Ridge(alpha=self.ridge, fit_intercept=True).fit(x, y)
        residual = y - self.model_.predict(x)
        self.residual_variance_ = np.maximum(np.var(residual, axis=0), 1e-6)
        coefficient = self.scaler_.coefficient_to_original(self.model_.coef_)
        self.channels_ = {
            "mean_feature": coefficient,
            "mean": aggregate_features(
                coefficient, train.source_index, train.targets.shape[1], signed=True
            ),
            "mean_strength": aggregate_features(
                coefficient, train.source_index, train.targets.shape[1], signed=False
            ),
        }
        self.channels_["linear_transition_coefficient_feature"] = self.channels_["mean_feature"]
        self.channels_["linear_transition_coefficient"] = self.channels_["mean"]
        return self

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        standardized_mean = self.model_.predict(self.scaler_.x(data.features))
        mean = self.scaler_.inverse_mean(standardized_mean)
        variance = np.broadcast_to(
            self.scaler_.inverse_variance(self.residual_variance_), mean.shape
        ).copy()
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
            cdf=ndtr((data.targets - mean) / np.sqrt(variance)),
            samples=samples,
            channels=self.channels_,
            metadata={"distribution_family": "gaussian"},
        )

    def metadata(self):
        return {"ridge": self.ridge}


class HeteroskedasticRidge(RidgeGaussian):
    """Two-stage mean/log-variance bridge with independent shrinkage penalties."""

    name = "heteroskedastic_ridge"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=(
            "conditional_mean_derivative",
            "conditional_log_variance_derivative",
        ),
        graph_scores=True,
    )

    def __init__(self, mean_ridge: float = 1.0, variance_ridge: float = 10.0):
        super().__init__(mean_ridge)
        self.mean_ridge = float(mean_ridge)
        self.variance_ridge = float(variance_ridge)

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        super().fit(train)
        # ``RidgeGaussian.fit`` exposes its mean coefficient under the reduced-
        # form ``linear_transition_coefficient`` name.  This subclass gives the
        # same coefficient the more specific conditional-mean-derivative
        # semantics below.  Retaining both canonical aliases would make the
        # emitted channel contract disagree with this estimator's declaration
        # and would double-count one fitted quantity under two claim labels.
        self.channels_.pop("linear_transition_coefficient_feature", None)
        self.channels_.pop("linear_transition_coefficient", None)
        x = self.scaler_.x(train.features)
        y = self.scaler_.y(train.targets)
        residual = y - self.model_.predict(x)
        log_squared = np.log(residual**2 + 1e-5)
        self.variance_model_ = Ridge(alpha=self.variance_ridge, fit_intercept=True).fit(
            x, log_squared
        )
        # E[log(Z^2)] for Z~N(0,1) is approximately -1.27036.
        self.log_chi_square_bias_ = 1.270362845
        # Variance calibration is fit on training outcomes only. Reusing the outer
        # validation targets here would make the subsequent validation NLL an
        # in-sample calibration score during hyperparameter selection.
        calibration_data = train
        raw = self._raw_variance_standardized(calibration_data.features)
        predicted_mean = self.model_.predict(self.scaler_.x(calibration_data.features))
        residual2 = self.scaler_.y(calibration_data.targets) - predicted_mean
        ratio = residual2**2 / np.maximum(raw, 1e-8)
        self.variance_calibration_ = np.clip(np.mean(ratio, axis=0), 0.1, 10.0)
        coefficient = self.scaler_.logvariance_coefficient_to_original(
            self.variance_model_.coef_
        )
        self.channels_.update(
            {
                "logvariance_feature": coefficient,
                "logvariance": aggregate_features(
                    coefficient, train.source_index, train.targets.shape[1], signed=True
                ),
                "logvariance_strength": aggregate_features(
                    coefficient, train.source_index, train.targets.shape[1], signed=False
                ),
            }
        )
        self.channels_["conditional_mean_derivative_feature"] = self.channels_["mean_feature"]
        self.channels_["conditional_mean_derivative"] = self.channels_["mean"]
        self.channels_["conditional_log_variance_derivative_feature"] = self.channels_[
            "logvariance_feature"
        ]
        self.channels_["conditional_log_variance_derivative"] = self.channels_["logvariance"]
        return self

    def _raw_variance_standardized(self, features: np.ndarray) -> np.ndarray:
        logvariance = self.variance_model_.predict(self.scaler_.x(features))
        return np.exp(np.clip(logvariance + self.log_chi_square_bias_, -10, 6))

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        mean_z = self.model_.predict(self.scaler_.x(data.features))
        mean = self.scaler_.inverse_mean(mean_z)
        variance_z = self._raw_variance_standardized(data.features) * self.variance_calibration_
        variance = self.scaler_.inverse_variance(variance_z)
        log_prob = gaussian_log_prob(data.targets, mean, variance)
        cdf = ndtr((data.targets - mean) / np.sqrt(np.maximum(variance, 1e-10)))
        samples = None
        if n_samples:
            rng = np.random.default_rng(0)
            samples = mean[:, None, :] + np.sqrt(variance[:, None, :]) * rng.standard_normal(
                (len(mean), n_samples, mean.shape[1])
            )
        return Prediction(
            mean=mean,
            variance=variance,
            log_prob=log_prob,
            cdf=cdf,
            samples=samples,
            channels=self.channels_,
        )

    def metadata(self):
        return {
            "mean_ridge": self.mean_ridge,
            "variance_ridge": self.variance_ridge,
            "variance_calibration_split": "training_only",
        }


class StudentTRidge(HeteroskedasticRidge):
    """Non-Gaussian reference model with conditional mean/scale and Student-t tails."""

    name = "student_t_ridge"

    def __init__(self, mean_ridge: float = 1.0, variance_ridge: float = 10.0, df: float = 5.0):
        if df <= 2:
            raise ValueError("df must exceed two for a finite variance")
        super().__init__(mean_ridge, variance_ridge)
        self.df = float(df)

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        gaussian = super().predict(data, n_samples=0)
        # Scale is chosen so the Student-t variance equals the modeled variance.
        scale = np.sqrt(gaussian.variance * (self.df - 2.0) / self.df)
        standardized = (data.targets - gaussian.mean) / np.maximum(scale, 1e-10)
        log_prob = student_t.logpdf(standardized, df=self.df) - np.log(
            np.maximum(scale, 1e-10)
        )
        cdf = student_t.cdf(standardized, df=self.df)
        samples = None
        if n_samples:
            rng = np.random.default_rng(0)
            samples = gaussian.mean[:, None, :] + scale[:, None, :] * rng.standard_t(
                self.df, size=(len(scale), n_samples, scale.shape[1])
            )
        return Prediction(
            mean=gaussian.mean,
            variance=gaussian.variance,
            log_prob=log_prob,
            cdf=cdf,
            samples=samples,
            channels=gaussian.channels,
        )

    def metadata(self):
        result = super().metadata()
        result["df"] = self.df
        return result
