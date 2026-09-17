from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression, Ridge

from .dgps import EPS, GaussianCovarianceDGP, GaussianMeanDGP, SmoothTiltDGP, raw_to_functional


def scalar_history_basis(h: np.ndarray, degree: int = 5) -> np.ndarray:
    h = np.asarray(h, dtype=float).reshape(-1)
    return np.stack([h**power for power in range(degree + 1)], axis=1)


def scalar_history_basis_derivative(h: np.ndarray, degree: int = 5) -> np.ndarray:
    h = np.asarray(h, dtype=float).reshape(-1)
    values = []
    for power in range(degree + 1):
        values.append(np.zeros_like(h) if power == 0 else power * h ** (power - 1))
    return np.stack(values, axis=1)


def response_features(y: np.ndarray, covariance: bool = False) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if covariance:
        y0 = y[:, 0]
        y1 = y[:, 1]
        return np.stack(
            [
                y0,
                y1,
                y0**2,
                y1**2,
                y0 * y1,
                y0**3,
                y1**3,
                y0**2 * y1,
                y0 * y1**2,
                np.tanh(y0),
                np.tanh(y1),
            ],
            axis=1,
        )
    value = y[:, 0] if y.ndim == 2 else y.reshape(-1)
    scaled = value / 3.0
    return np.stack(
        [
            scaled,
            scaled**2,
            scaled**3,
            scaled**4,
            scaled**5,
            np.tanh(value),
            (value > 1.5).astype(float),
            (value > 0.75).astype(float),
            np.exp(-((value - 1.45) / 0.45) ** 2),
        ],
        axis=1,
    )


def conditional_classifier_features(h: np.ndarray, y: np.ndarray, covariance: bool = False) -> np.ndarray:
    h = np.asarray(h, dtype=float).reshape(-1)
    response = response_features(y, covariance=covariance)
    history = scalar_history_basis(h, degree=3)[:, 1:]
    interactions = np.concatenate(
        [history[:, power : power + 1] * response for power in range(history.shape[1])], axis=1
    )
    return np.concatenate([history, response, interactions], axis=1)


def conditional_classifier_history_derivative(
    h: np.ndarray, y: np.ndarray, coefficients: np.ndarray, covariance: bool = False
) -> np.ndarray:
    h = np.asarray(h, dtype=float).reshape(-1)
    response = response_features(y, covariance=covariance)
    response_dim = response.shape[1]
    history_dim = 3
    derivative = np.zeros(h.size)
    history_coefficients = coefficients[:history_dim]
    derivative += history_coefficients[0] + 2.0 * history_coefficients[1] * h + 3.0 * history_coefficients[2] * h**2
    offset = history_dim + response_dim
    for power in range(1, history_dim + 1):
        block = coefficients[offset : offset + response_dim]
        derivative += power * h ** (power - 1) * (response @ block)
        offset += response_dim
    return derivative


@dataclass
class RawMomentRegressor:
    channel: str
    degree: int = 5
    ridge: float = 1e-3
    threshold: float | None = None

    def fit(self, h: np.ndarray, y: np.ndarray) -> "RawMomentRegressor":
        h = np.asarray(h, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float)
        design = scalar_history_basis(h, self.degree)
        if self.channel == "covariance":
            targets = np.stack([y[:, 0], y[:, 1], y[:, 0] * y[:, 1]], axis=1)
        else:
            value = y[:, 0] if y.ndim == 2 else y.reshape(-1)
            targets = np.stack([value, value**2, value**3, value**4], axis=1)
            if self.channel in {"tail_probability", "mode_occupancy"}:
                threshold = float(self.threshold)
                targets = np.concatenate([targets, (value > threshold)[:, None]], axis=1)
        self.model = Ridge(alpha=self.ridge, fit_intercept=False)
        self.model.fit(design, targets)
        return self

    def _predict_heads(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design = scalar_history_basis(h, self.degree)
        derivative_design = scalar_history_basis_derivative(h, self.degree)
        coefficients = np.asarray(self.model.coef_, dtype=float)
        predicted = design @ coefficients.T
        derivative = derivative_design @ coefficients.T
        return predicted, derivative

    def target(self, h: np.ndarray) -> np.ndarray:
        predicted, _ = self._predict_heads(h)
        if self.channel == "covariance":
            return predicted[:, 2] - predicted[:, 0] * predicted[:, 1]
        if self.channel in {"tail_probability", "mode_occupancy"}:
            return predicted[:, 4]
        return raw_to_functional(predicted[:, :4], self.channel)

    def local_effect(self, h: np.ndarray) -> np.ndarray:
        predicted, derivative = self._predict_heads(h)
        if self.channel == "covariance":
            return derivative[:, 2] - derivative[:, 0] * predicted[:, 1] - predicted[:, 0] * derivative[:, 1]
        if self.channel in {"tail_probability", "mode_occupancy"}:
            return derivative[:, 4]
        from .dgps import raw_functional_derivative

        return raw_functional_derivative(predicted[:, :4], derivative[:, :4], self.channel)

    def finite_effect(self, h: np.ndarray, delta: float) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        return (self.target(h + delta) - self.target(h - delta)) / (2.0 * delta)


class RatioCritic:
    def __init__(self, covariance: bool = False, c: float = 1.0) -> None:
        self.covariance = bool(covariance)
        self.c = float(c)

    def fit(self, h: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> "RatioCritic":
        h = np.asarray(h, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float)
        shuffled = y[rng.permutation(y.shape[0])]
        design = np.concatenate(
            [
                conditional_classifier_features(h, y, self.covariance),
                conditional_classifier_features(h, shuffled, self.covariance),
            ],
            axis=0,
        )
        labels = np.concatenate([np.ones(h.size), np.zeros(h.size)])
        self.model = LogisticRegression(
            C=self.c,
            penalty="l2",
            fit_intercept=True,
            solver="lbfgs",
            max_iter=500,
            random_state=0,
            n_jobs=1,
        )
        self.model.fit(design, labels)
        return self

    def tangent(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        return conditional_classifier_history_derivative(
            h, y, np.asarray(self.model.coef_[0]), covariance=self.covariance
        )

    def centered_tangent(
        self,
        h: np.ndarray,
        y: np.ndarray,
        center_sampler: Callable[[np.ndarray], np.ndarray],
        center_draws: int = 128,
    ) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape(-1)
        estimate = self.tangent(h, y)
        unique, inverse = np.unique(h, return_inverse=True)
        centers = np.zeros(unique.size)
        for index, anchor in enumerate(unique):
            anchors = np.full(center_draws, anchor)
            draws = center_sampler(anchors)
            centers[index] = np.mean(self.tangent(anchors, draws))
        return estimate - centers[inverse]


def _score_feature_matrices(
    h: np.ndarray, y: np.ndarray, sigma: np.ndarray | float, degree_h: int = 3, degree_y: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(h, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), h.shape)
    scaled_y = y / 3.0
    columns = []
    dy_columns = []
    dh_columns = []
    for sigma_power in range(2):
        sigma_term = sigma**sigma_power
        for h_power in range(degree_h + 1):
            for y_power in range(degree_y + 1):
                columns.append(sigma_term * h**h_power * scaled_y**y_power)
                dy_columns.append(
                    np.zeros_like(y)
                    if y_power == 0
                    else sigma_term * h**h_power * (y_power / 3.0) * scaled_y ** (y_power - 1)
                )
                dh_columns.append(
                    np.zeros_like(h)
                    if h_power == 0
                    else sigma_term * h_power * h ** (h_power - 1) * scaled_y**y_power
                )
    return np.stack(columns, axis=1), np.stack(dy_columns, axis=1), np.stack(dh_columns, axis=1)


class PolynomialScoreModel:
    def __init__(self, kind: str, ridge: float = 1e-3, primary_sigma: float = 0.12) -> None:
        self.kind = str(kind)
        self.ridge = float(ridge)
        self.primary_sigma = float(primary_sigma)
        self.coefficients: np.ndarray | None = None
        self.base_regressor: RawMomentRegressor | None = None

    def fit(
        self,
        h: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        sigma_ladder: tuple[float, ...] = (0.03, 0.06, 0.12, 0.25),
    ) -> "PolynomialScoreModel":
        h = np.asarray(h, dtype=float).reshape(-1)
        value = np.asarray(y, dtype=float).reshape(-1)
        if self.kind == "hyvarinen":
            design, derivative_y, _ = _score_feature_matrices(h, value, 0.0)
            gram = design.T @ design / h.size + self.ridge * np.eye(design.shape[1])
            right = -np.mean(derivative_y, axis=0)
            self.coefficients = np.linalg.solve(gram, right)
            return self

        self.base_regressor = RawMomentRegressor(channel="variance", degree=5, ridge=1e-3)
        self.base_regressor.fit(h, value[:, None])
        if self.kind == "gaussian_dsm":
            return self

        repeated_h = []
        noisy = []
        sigmas = []
        targets = []
        for sigma in sigma_ladder:
            epsilon = rng.normal(size=value.size)
            repeated_h.append(h)
            noisy.append(value + sigma * epsilon)
            sigmas.append(np.full(value.size, sigma))
            targets.append(-epsilon / sigma)
        hh = np.concatenate(repeated_h)
        zz = np.concatenate(noisy)
        ss = np.concatenate(sigmas)
        target = np.concatenate(targets)
        design, _, _ = _score_feature_matrices(hh, zz, ss)
        if self.kind == "anchored":
            base = self._base_score(hh, zz, ss)
            target = target - base
            penalty = self.ridge * np.linspace(1.0, 8.0, design.shape[1])
        else:
            penalty = self.ridge * np.ones(design.shape[1])
        gram = design.T @ design / hh.size + np.diag(penalty)
        right = design.T @ target / hh.size
        self.coefficients = np.linalg.solve(gram, right)
        return self

    def _base_moments(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.base_regressor is None:
            raise RuntimeError("base model has not been fitted")
        predicted, _ = self.base_regressor._predict_heads(h)
        mean = predicted[:, 0]
        variance = np.maximum(predicted[:, 1] - mean**2, 0.05)
        return mean, variance

    def _base_score(self, h: np.ndarray, y: np.ndarray, sigma: np.ndarray | float) -> np.ndarray:
        mean, variance = self._base_moments(np.asarray(h))
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), mean.shape)
        return -(np.asarray(y) - mean) / (variance + sigma**2)

    def score(self, h: np.ndarray, y: np.ndarray, sigma: float | np.ndarray = 0.0) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        sigma_array = np.broadcast_to(np.asarray(sigma, dtype=float), h.shape)
        if self.kind == "gaussian_dsm":
            return self._base_score(h, y, sigma_array)
        design, _, _ = _score_feature_matrices(h, y, sigma_array)
        residual = design @ np.asarray(self.coefficients)
        if self.kind == "anchored":
            return self._base_score(h, y, sigma_array) + residual
        return residual

    def mixed_field(
        self, h: np.ndarray, y: np.ndarray, sigma: float = 0.0, delta: float = 1e-3
    ) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        return (self.score(h + delta, y, sigma) - self.score(h - delta, y, sigma)) / (2.0 * delta)

    def reconstructed_tangent_grid(
        self,
        h: float,
        grid: np.ndarray,
        center_density: np.ndarray,
        weights: np.ndarray,
        sigma: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        anchors = np.full(grid.size, h)
        field = self.mixed_field(anchors, grid, sigma=sigma)
        potential = cumulative_trapezoid(field, grid, initial=0.0)
        center = np.sum(weights * center_density * potential)
        potential -= center
        gradient = np.gradient(potential, grid, edge_order=2)
        residual = float(np.sqrt(np.sum(weights * center_density * (gradient - field) ** 2)))
        return potential, residual

    def density_grid(self, h: float, grid: np.ndarray, weights: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        score = self.score(np.full(grid.size, h), grid, sigma=sigma)
        log_density = cumulative_trapezoid(score, grid, initial=0.0)
        log_density -= np.max(log_density)
        density = np.exp(log_density)
        density /= np.sum(weights * density)
        return density


class NormalizedStaticModel:
    def __init__(self, dgp) -> None:
        self.dgp = dgp

    def fit(self, h: np.ndarray, y: np.ndarray) -> "NormalizedStaticModel":
        h = np.asarray(h, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float)
        if isinstance(self.dgp, SmoothTiltDGP):
            psi = self.dgp.psi_at(y[:, 0])

            def objective(amplitude: float) -> float:
                factor = 1.0 + amplitude * np.tanh(h) * psi
                if np.min(factor) <= 0:
                    return np.inf
                return float(-np.sum(np.log(factor)))

            result = minimize_scalar(objective, bounds=(-0.85, 0.85), method="bounded")
            self.amplitude = float(result.x)
        elif isinstance(self.dgp, GaussianMeanDGP):
            self.regressor = RawMomentRegressor("mean", degree=5).fit(h, y)
        elif isinstance(self.dgp, GaussianCovarianceDGP):
            self.regressor = RawMomentRegressor("covariance", degree=5).fit(h, y)
        else:
            raise TypeError(type(self.dgp))
        return self

    def tangent(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float)
        if isinstance(self.dgp, SmoothTiltDGP):
            t = self.amplitude * np.tanh(h)
            dt = self.amplitude / np.cosh(h) ** 2
            psi = self.dgp.psi_at(y[:, 0])
            return dt * psi / np.maximum(1.0 + t * psi, EPS)
        if isinstance(self.dgp, GaussianMeanDGP):
            predicted, derivative = self.regressor._predict_heads(h)
            mean = predicted[:, 0]
            variance = np.maximum(predicted[:, 1] - mean**2, 0.05)
            dmean = derivative[:, 0]
            dvariance = derivative[:, 1] - 2.0 * mean * dmean
            residual = y[:, 0] - mean
            return residual * dmean / variance + 0.5 * dvariance * (residual**2 / variance**2 - 1.0 / variance)
        rho = np.clip(self.regressor.target(h), -0.85, 0.85)
        drho = self.regressor.local_effect(h)
        y0, y1 = y[:, 0], y[:, 1]
        denominator = np.maximum(1.0 - rho**2, EPS)
        derivative_rho = (
            rho / denominator
            + y0 * y1 / denominator
            - rho * (y0**2 - 2.0 * rho * y0 * y1 + y1**2) / denominator**2
        )
        return drho * derivative_rho

    def target(self, h: np.ndarray, channel: str) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        if isinstance(self.dgp, SmoothTiltDGP):
            original = self.dgp.amplitude
            self.dgp.amplitude = self.amplitude
            try:
                return self.dgp.target(h, channel)
            finally:
                self.dgp.amplitude = original
        return self.regressor.target(h)

    def local_effect(self, h: np.ndarray, channel: str) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        if isinstance(self.dgp, SmoothTiltDGP):
            original = self.dgp.amplitude
            self.dgp.amplitude = self.amplitude
            try:
                return self.dgp.local_effect(h, channel)
            finally:
                self.dgp.amplitude = original
        return self.regressor.local_effect(h)

    def finite_effect(self, h: np.ndarray, delta: float, channel: str) -> np.ndarray:
        return (self.target(h + delta, channel) - self.target(h - delta, channel)) / (2.0 * delta)


class BalancedEndpointAdapter:
    def __init__(self, kind: str, delta: float, covariance: bool = False, ridge: float = 1e-2) -> None:
        self.kind = str(kind)
        self.delta = float(delta)
        self.covariance = bool(covariance)
        self.ridge = float(ridge)

    def fit(
        self,
        train_h: np.ndarray,
        train_y: np.ndarray,
        train_label: np.ndarray,
        calibration_h: np.ndarray,
        calibration_y: np.ndarray,
        calibration_label: np.ndarray,
    ) -> "BalancedEndpointAdapter":
        design = conditional_classifier_features(train_h, train_y, self.covariance)
        labels = np.asarray(train_label, dtype=int)
        if self.kind == "classifier":
            self.model = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                fit_intercept=True,
                max_iter=500,
                random_state=0,
                n_jobs=1,
            ).fit(design, labels)
            calibration_design = conditional_classifier_features(
                calibration_h, calibration_y, self.covariance
            )
            logits = self.model.decision_function(calibration_design)

            def nll(parameters: np.ndarray) -> float:
                adjusted = parameters[0] * logits + parameters[1]
                return float(np.mean(np.logaddexp(0.0, adjusted) - calibration_label * adjusted))

            result = minimize(nll, np.array([1.0, 0.0]), method="BFGS")
            self.calibration = np.asarray(result.x, dtype=float)
        else:
            self.model = Ridge(alpha=self.ridge).fit(design, 2.0 * labels - 1.0)
            self.calibration = np.array([1.0, 0.0])
        return self

    def signed_posterior(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        design = conditional_classifier_features(h, y, self.covariance)
        if self.kind == "classifier":
            logits = self.model.decision_function(design)
            logits = self.calibration[0] * logits + self.calibration[1]
            return np.tanh(0.5 * logits)
        return np.clip(self.model.predict(design), -1.0, 1.0)

    def witness(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.signed_posterior(h, y) / self.delta


def finite_from_witness(
    y: np.ndarray,
    witness: np.ndarray,
    delta: float,
    channel: str,
    threshold: float | None = None,
) -> float:
    y = np.asarray(y, dtype=float)
    witness = np.asarray(witness, dtype=float).reshape(-1)
    if channel == "covariance":
        features = np.stack([y[:, 0], y[:, 1], y[:, 0] * y[:, 1]], axis=1)
        pooled = np.mean(features, axis=0)
        differences = np.mean(features * witness[:, None], axis=0)
        plus = pooled + delta * differences
        minus = pooled - delta * differences
        covariance_plus = plus[2] - plus[0] * plus[1]
        covariance_minus = minus[2] - minus[0] * minus[1]
        return float((covariance_plus - covariance_minus) / (2.0 * delta))
    value = y[:, 0] if y.ndim == 2 else y.reshape(-1)
    if channel in {"tail_probability", "mode_occupancy"}:
        feature = (value > float(threshold)).astype(float)
        return float(np.mean(feature * witness))
    features = np.stack([value, value**2, value**3, value**4], axis=1)
    pooled = np.mean(features, axis=0)
    differences = np.mean(features * witness[:, None], axis=0)
    plus = pooled + delta * differences
    minus = pooled - delta * differences
    return float((raw_to_functional(plus, channel) - raw_to_functional(minus, channel)) / (2.0 * delta))
