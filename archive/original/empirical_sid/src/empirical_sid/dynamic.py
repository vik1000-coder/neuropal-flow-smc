from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression, Ridge

from .dgps import (
    DynamicGaussianMeanDGP,
    DynamicStochasticGainDGP,
    influence_from_raw,
    raw_functional_derivative,
    raw_to_functional,
    trap_weights,
)
from .estimators import finite_from_witness
from .metrics import lag_metrics


def history_design(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return np.concatenate([np.ones((h.shape[0], 1)), h, h**2, h**3], axis=1)


def history_design_derivative(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    n, q = h.shape
    result = np.zeros((n, q, 1 + 3 * q))
    for lag in range(q):
        result[:, lag, 1 + lag] = 1.0
        result[:, lag, 1 + q + lag] = 2.0 * h[:, lag]
        result[:, lag, 1 + 2 * q + lag] = 3.0 * h[:, lag] ** 2
    return result


@dataclass
class DynamicMomentRegressor:
    ridge: float = 1e-2

    def fit(self, h: np.ndarray, y: np.ndarray) -> "DynamicMomentRegressor":
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = np.stack([y, np.zeros_like(y)], axis=1)
        targets = np.stack(
            [y[:, 0], y[:, 1], y[:, 0] ** 2, y[:, 0] * y[:, 1], y[:, 0] ** 3],
            axis=1,
        )
        self.model = Ridge(alpha=self.ridge, fit_intercept=False).fit(history_design(h), targets)
        return self

    def heads(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design = history_design(h)
        derivative_design = history_design_derivative(h)
        coefficients = np.asarray(self.model.coef_, dtype=float)
        predicted = design @ coefficients.T
        derivative = np.einsum("nqd,kd->nqk", derivative_design, coefficients)
        return predicted, derivative

    def target(self, h: np.ndarray, channel: str) -> np.ndarray:
        heads, _ = self.heads(h)
        mean0, mean1, raw2, cross, raw3 = [heads[:, index] for index in range(5)]
        if channel == "mean":
            return mean0
        if channel == "variance":
            return raw2 - mean0**2
        if channel == "covariance":
            return cross - mean0 * mean1
        if channel == "third_cumulant":
            return raw3 - 3.0 * mean0 * raw2 + 2.0 * mean0**3
        raise KeyError(channel)

    def effect(self, h: np.ndarray, channel: str) -> np.ndarray:
        heads, derivative = self.heads(h)
        mean0, mean1, raw2, _cross, _raw3 = [heads[:, index] for index in range(5)]
        dmean0 = derivative[:, :, 0]
        dmean1 = derivative[:, :, 1]
        draw2 = derivative[:, :, 2]
        dcross = derivative[:, :, 3]
        draw3 = derivative[:, :, 4]
        if channel == "mean":
            return dmean0
        if channel == "variance":
            return draw2 - 2.0 * mean0[:, None] * dmean0
        if channel == "covariance":
            return dcross - dmean0 * mean1[:, None] - mean0[:, None] * dmean1
        if channel == "third_cumulant":
            return (
                draw3
                - 3.0 * (dmean0 * raw2[:, None] + mean0[:, None] * draw2)
                + 6.0 * mean0[:, None] ** 2 * dmean0
            )
        raise KeyError(channel)

    def finite_effect(self, h: np.ndarray, delta: float, channel: str) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        result = np.zeros((h.shape[0], h.shape[1]))
        for lag in range(h.shape[1]):
            plus = h.copy()
            minus = h.copy()
            plus[:, lag] += delta
            minus[:, lag] -= delta
            result[:, lag] = (self.target(plus, channel) - self.target(minus, channel)) / (
                2.0 * delta
            )
        return result


class CorrectGainMixture:
    def __init__(self, dgp: DynamicStochasticGainDGP) -> None:
        self.dgp = dgp

    def fit(self, h: np.ndarray, y: np.ndarray) -> "CorrectGainMixture":
        h = np.asarray(h, dtype=float)
        y = np.asarray(y, dtype=float)
        q = h.shape[1]

        def objective(parameters: np.ndarray) -> float:
            probability = expit(parameters[0] + h @ parameters[1:])
            mean1 = (1.0 - probability)[:, None] * self.dgp.gain_vector[None, :]
            mean0 = -probability[:, None] * self.dgp.gain_vector[None, :]
            log1 = np.log(np.maximum(probability, 1e-12)) - 0.5 * np.sum(
                ((y - mean1) / self.dgp.noise_sd) ** 2, axis=1
            )
            log0 = np.log(np.maximum(1.0 - probability, 1e-12)) - 0.5 * np.sum(
                ((y - mean0) / self.dgp.noise_sd) ** 2, axis=1
            )
            return float(-np.mean(logsumexp(np.stack([log0, log1], axis=1), axis=1)))

        initial = np.zeros(q + 1)
        initial[0] = self.dgp.alpha
        result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": 300})
        self.parameters = np.asarray(result.x)
        self.converged = bool(result.success)
        return self

    def probability(self, h: np.ndarray) -> np.ndarray:
        return expit(self.parameters[0] + np.asarray(h) @ self.parameters[1:])

    def effect(self, h: np.ndarray, channel: str) -> np.ndarray:
        probability = self.probability(h)
        derivative = probability[:, None] * (1.0 - probability[:, None]) * self.parameters[None, 1:]
        if channel == "mean":
            return np.zeros_like(derivative)
        if channel == "variance":
            return self.dgp.gain**2 * (1.0 - 2.0 * probability[:, None]) * derivative
        if channel == "covariance":
            product = self.dgp.gain_vector[0] * self.dgp.gain_vector[1]
            return product * (1.0 - 2.0 * probability[:, None]) * derivative
        if channel == "third_cumulant":
            factor = 1.0 - 6.0 * probability + 6.0 * probability**2
            return self.dgp.gain**3 * factor[:, None] * derivative
        raise KeyError(channel)

    def target(self, h: np.ndarray, channel: str) -> np.ndarray:
        probability = self.probability(h)
        if channel == "mean":
            return np.zeros_like(probability)
        if channel == "variance":
            return self.dgp.noise_sd**2 + self.dgp.gain**2 * probability * (1.0 - probability)
        if channel == "covariance":
            return self.dgp.gain_vector[0] * self.dgp.gain_vector[1] * probability * (1.0 - probability)
        if channel == "third_cumulant":
            return self.dgp.gain**3 * probability * (1.0 - probability) * (1.0 - 2.0 * probability)
        raise KeyError(channel)

    def finite_effect(self, h: np.ndarray, delta: float, channel: str) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        result = np.zeros((h.shape[0], h.shape[1]))
        for lag in range(h.shape[1]):
            plus = h.copy()
            minus = h.copy()
            plus[:, lag] += delta
            minus[:, lag] -= delta
            result[:, lag] = (self.target(plus, channel) - self.target(minus, channel)) / (
                2.0 * delta
            )
        return result


def dynamic_response_features(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = np.stack([y, np.zeros_like(y)], axis=1)
    y0 = y[:, 0] / 3.0
    y1 = y[:, 1] / 3.0
    return np.stack(
        [y0, y1, y0**2, y1**2, y0 * y1, y0**3, np.tanh(y[:, 0]), np.tanh(y[:, 1])],
        axis=1,
    )


class DynamicRatioCritic:
    def fit(self, h: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> "DynamicRatioCritic":
        h = np.asarray(h, dtype=float)
        y = np.asarray(y, dtype=float)
        shuffled = y[rng.permutation(y.shape[0])]
        joint = self._design(h, y)
        product = self._design(h, shuffled)
        design = np.concatenate([joint, product], axis=0)
        label = np.concatenate([np.ones(h.shape[0]), np.zeros(h.shape[0])])
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=0).fit(
            design, label
        )
        self.q = h.shape[1]
        self.r = dynamic_response_features(y).shape[1]
        return self

    def _design(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        response = dynamic_response_features(y)
        interaction = np.concatenate([h[:, lag : lag + 1] * response for lag in range(h.shape[1])], axis=1)
        return np.concatenate([h, response, interaction], axis=1)

    def tangent(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        response = dynamic_response_features(y)
        coefficient = np.asarray(self.model.coef_[0])
        result = np.broadcast_to(coefficient[: self.q], (h.shape[0], self.q)).copy()
        offset = self.q + self.r
        for lag in range(self.q):
            block = coefficient[offset + lag * self.r : offset + (lag + 1) * self.r]
            result[:, lag] += response @ block
        return result


def influence_dynamic(y: np.ndarray, channel: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y0, y1 = y[:, 0], y[:, 1]
    mean0, mean1 = np.mean(y0), np.mean(y1)
    centered0 = y0 - mean0
    if channel == "mean":
        return centered0
    if channel == "variance":
        variance = np.mean(centered0**2)
        return centered0**2 - variance
    if channel == "covariance":
        covariance = np.mean(centered0 * (y1 - mean1))
        return centered0 * (y1 - mean1) - covariance
    if channel == "third_cumulant":
        variance = np.mean(centered0**2)
        third = np.mean(centered0**3)
        return centered0**3 - 3.0 * variance * centered0 - third
    raise KeyError(channel)


def ratio_effect(
    critic: DynamicRatioCritic,
    dgp,
    histories: np.ndarray,
    channel: str,
    rng: np.random.Generator,
    draws: int = 256,
) -> np.ndarray:
    result = np.zeros((histories.shape[0], histories.shape[1]))
    for index, history in enumerate(histories):
        repeated = np.repeat(history[None, :], draws, axis=0)
        y = dgp.sample_vector(repeated, rng) if hasattr(dgp, "sample_vector") else np.stack(
            [dgp.sample(repeated, rng), rng.normal(size=draws)], axis=1
        )
        tangent = critic.tangent(repeated, y)
        tangent -= np.mean(tangent, axis=0, keepdims=True)
        influence = influence_dynamic(y, channel)
        result[index] = np.mean(influence[:, None] * tangent, axis=0)
    return result


def _dynamic_score_design(h: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1) / 3.0
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    blocks = []
    for sigma_power in range(2):
        scale = sigma**sigma_power
        response = np.stack([scale * y**power for power in range(6)], axis=1)
        blocks.append(response)
        blocks.extend([h[:, lag : lag + 1] * response for lag in range(h.shape[1])])
    return np.concatenate(blocks, axis=1)


class DynamicAnchoredScore:
    def __init__(self, ridge: float = 1.0) -> None:
        self.ridge = float(ridge)

    def fit(self, h: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> "DynamicAnchoredScore":
        self.base = DynamicMomentRegressor(ridge=1e-2).fit(h, np.stack([y, np.zeros_like(y)], axis=1))
        designs = []
        targets = []
        for sigma in [0.03, 0.06, 0.12, 0.25]:
            epsilon = rng.normal(size=y.size)
            z = y + sigma * epsilon
            design = _dynamic_score_design(h, z, np.full(y.size, sigma))
            base_score = self._base_score(h, z, sigma)
            designs.append(design)
            targets.append(-epsilon / sigma - base_score)
        design = np.concatenate(designs, axis=0)
        target = np.concatenate(targets)
        gram = design.T @ design / design.shape[0] + self.ridge * np.eye(design.shape[1])
        right = design.T @ target / design.shape[0]
        self.coefficients = np.linalg.solve(gram, right)
        self.q = h.shape[1]
        return self

    def _base_score(self, h: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        heads, _ = self.base.heads(h)
        mean = heads[:, 0]
        variance = np.maximum(heads[:, 2] - mean**2, 0.05)
        return -(np.asarray(y) - mean) / (variance + sigma**2)

    def score(self, h: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        design = _dynamic_score_design(h, y, np.full(np.asarray(y).size, sigma))
        return self._base_score(h, y, sigma) + design @ self.coefficients

    def mixed(self, h: np.ndarray, y: np.ndarray, sigma: float, delta: float = 1e-3) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        result = np.zeros((h.shape[0], h.shape[1]))
        for lag in range(h.shape[1]):
            plus = h.copy()
            minus = h.copy()
            plus[:, lag] += delta
            minus[:, lag] -= delta
            result[:, lag] = (self.score(plus, y, sigma) - self.score(minus, y, sigma)) / (
                2.0 * delta
            )
        return result


def true_noised_density(dgp: DynamicStochasticGainDGP, h: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    probability = float(dgp.pi(h[None, :])[0])
    sd = np.sqrt(dgp.noise_sd**2 + sigma**2)
    mean1 = dgp.gain * (1.0 - probability)
    mean0 = -dgp.gain * probability
    density = probability * np.exp(-0.5 * ((grid - mean1) / sd) ** 2) + (
        1.0 - probability
    ) * np.exp(-0.5 * ((grid - mean0) / sd) ** 2)
    return density


def score_effect(
    model: DynamicAnchoredScore,
    dgp: DynamicStochasticGainDGP,
    histories: np.ndarray,
    channel: str,
    sigma: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-5.0, 5.0, 501)
    weights = trap_weights(grid)
    result = np.zeros((histories.shape[0], histories.shape[1]))
    sbtg = np.zeros_like(result)
    for index, history in enumerate(histories):
        repeated = np.repeat(history[None, :], grid.size, axis=0)
        density = true_noised_density(dgp, history, grid, sigma)
        density /= np.sum(weights * density)
        field = model.mixed(repeated, grid, sigma)
        raw = np.asarray([np.sum(weights * density * grid**power) for power in range(1, 5)])
        influence = influence_from_raw(grid, raw, channel)
        for lag in range(histories.shape[1]):
            tangent = cumulative_trapezoid(field[:, lag], grid, initial=0.0)
            tangent -= np.sum(weights * density * tangent)
            result[index, lag] = np.sum(weights * density * influence * tangent)
            sbtg[index, lag] = np.sqrt(np.sum(weights * density * field[:, lag] ** 2))
    return result, sbtg


def finite_classifier_profile(
    dgp: DynamicStochasticGainDGP,
    histories: np.ndarray,
    channel: str,
    delta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    q = histories.shape[1]
    profile = np.zeros(q)
    for lag in range(q):
        anchors = dgp.sample_histories(600, rng)
        repeated = np.repeat(anchors, 6, axis=0)
        plus_h = repeated.copy()
        minus_h = repeated.copy()
        plus_h[:, lag] += delta
        minus_h[:, lag] -= delta
        plus = dgp.sample_vector(plus_h, rng)
        minus = dgp.sample_vector(minus_h, rng)
        y = np.concatenate([plus, minus], axis=0)
        labels = np.concatenate([np.ones(plus.shape[0]), np.zeros(minus.shape[0])])
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=0).fit(
            dynamic_response_features(y), labels
        )
        estimates = []
        for history in histories:
            repeated_eval = np.repeat(history[None, :], 256, axis=0)
            plus_eval = repeated_eval.copy()
            minus_eval = repeated_eval.copy()
            plus_eval[:, lag] += delta
            minus_eval[:, lag] -= delta
            y_eval = np.concatenate(
                [dgp.sample_vector(plus_eval, rng), dgp.sample_vector(minus_eval, rng)], axis=0
            )
            signed = np.tanh(0.5 * model.decision_function(dynamic_response_features(y_eval)))
            estimates.append(
                finite_from_witness(y_eval, signed / delta, delta, channel, threshold=None)
            )
        profile[lag] = float(np.mean(estimates))
    return profile


def _metric_rows(
    config: dict[str, Any],
    seed: int,
    mechanism: str,
    channel: str,
    method: str,
    estimand: str,
    metrics: dict[str, float],
    access: str,
    wall_time: float,
) -> list[dict[str, Any]]:
    rows = []
    for name, value in metrics.items():
        rows.append(
            {
                "study_id": config["study_id"],
                "phase": "confirmatory",
                "dgp_family": mechanism,
                "dgp_seed": seed,
                "data_seed": seed + 500_000,
                "model_seed": seed + 600_000,
                "adapter_seed": seed + 700_000,
                "eval_seed": seed + 800_000,
                "method_id": method,
                "estimand_type": estimand,
                "channel": channel,
                "delta": config["dynamic"]["endpoint_delta"] if "finite" in estimand else 0.0,
                "sigma": 0.12 if "S7" in method or "SBTG" in method else 0.0,
                "information_access": access,
                "metric_name": name,
                "metric_value": float(value),
                "status": "ok",
                "failure_code": "",
                "wall_time": wall_time,
            }
        )
    return rows


def run_dynamic_case(
    config: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    lags = int(config["dynamic"]["lags"])
    rho = float(config["dynamic"]["rho"])
    n_train = int(config["dynamic"]["d1_n_train"])

    d1 = DynamicGaussianMeanDGP(lags=lags, rho=rho)
    h1 = d1.sample_histories(n_train, rng)
    y1_scalar = d1.sample(h1, rng)
    y1 = np.stack([y1_scalar, rng.normal(size=n_train)], axis=1)
    eval1 = d1.sample_histories(96, rng)
    truth1 = np.mean(d1.lag_effect(eval1), axis=0)
    arrays["d1__evaluation_histories"] = eval1
    arrays["d1__mean__truth_profile"] = truth1
    direct1 = DynamicMomentRegressor().fit(h1, y1)
    ratio1 = DynamicRatioCritic().fit(h1, y1, rng)
    estimates1 = {
        "B0_ZERO": (np.zeros(lags), "observational"),
        "B1_MEAN_POLY": (np.mean(direct1.effect(eval1, "mean"), axis=0), "observational"),
        "B4_GAUSSIAN_NLL": (np.mean(direct1.effect(eval1, "mean"), axis=0), "normalized_density"),
        "A2_RATIO_CRITIC": (
            np.mean(ratio_effect(ratio1, d1, eval1, "mean", rng, draws=192), axis=0),
            "observational",
        ),
    }
    for method, (estimate, access) in estimates1.items():
        arrays[f"d1__mean__{method.lower()}__profile"] = estimate
        rows.extend(
            _metric_rows(
                config,
                seed,
                d1.mechanism,
                "mean",
                method,
                "dynamic_local",
                lag_metrics(estimate, truth1),
                access,
                time.perf_counter() - started,
            )
        )

    d2 = DynamicStochasticGainDGP(
        lags=lags,
        rho=rho,
        primary_pi=float(config["dynamic"]["primary_pi"]),
    )
    h2 = d2.sample_histories(int(config["dynamic"]["d2_n_train"]), rng)
    y2 = d2.sample_vector(h2, rng)
    eval2 = d2.sample_histories(64, rng)
    direct2 = DynamicMomentRegressor().fit(h2, y2)
    mixture = CorrectGainMixture(d2).fit(h2, y2)
    critic = DynamicRatioCritic().fit(h2, y2, rng)
    score = DynamicAnchoredScore(
        ridge=float(config.get("score_selection", {}).get("anchored_ridge", 1.0))
    ).fit(h2, y2[:, 0], rng)
    delta = float(config["dynamic"]["endpoint_delta"])
    for channel in ["mean", "variance", "covariance", "third_cumulant"]:
        truth = np.mean(d2.lag_effect(eval2, channel), axis=0)
        truth_matrix = d2.lag_effect(eval2, channel)
        direct_matrix = direct2.effect(eval2, channel)
        mixture_matrix = mixture.effect(eval2, channel)
        ratio_matrix = ratio_effect(critic, d2, eval2, channel, rng, draws=192)
        direct_estimate = np.mean(direct_matrix, axis=0)
        mixture_estimate = np.mean(mixture_matrix, axis=0)
        ratio_estimate = np.mean(ratio_matrix, axis=0)
        arrays["d2__evaluation_histories"] = eval2
        arrays[f"d2__{channel}__truth_matrix"] = truth_matrix
        arrays[f"d2__{channel}__b2_raw_moments_matrix"] = direct_matrix
        arrays[f"d2__{channel}__b7_mdn_correct_matrix"] = mixture_matrix
        arrays[f"d2__{channel}__a2_ratio_critic_matrix"] = ratio_matrix
        local = {
            "B0_ZERO": (np.zeros(lags), "observational"),
            "B2_RAW_MOMENTS": (direct_estimate, "observational"),
            "B7_MDN_CORRECT": (mixture_estimate, "normalized_density"),
            "A2_RATIO_CRITIC": (ratio_estimate, "observational"),
        }
        sbtg_profile = None
        if channel != "covariance":
            score_estimate, sbtg = score_effect(score, d2, eval2, channel)
            arrays[f"d2__{channel}__s7_anchored_matrix"] = score_estimate
            arrays[f"d2__{channel}__original_sbtg_matrix"] = sbtg
            local["S7_ANCHORED_ENERGY__A7_HODGE"] = (
                np.mean(score_estimate, axis=0),
                "response_score",
            )
            sbtg_profile = np.mean(sbtg, axis=0)
        for method, (estimate, access) in local.items():
            arrays[f"d2__{channel}__{method.lower()}__profile"] = estimate
            if channel == "mean":
                metrics = {
                    "mean_null_rms": float(np.sqrt(np.mean(estimate**2))),
                    "mean_null_max_abs": float(np.max(np.abs(estimate))),
                }
            else:
                metrics = lag_metrics(estimate, truth)
            rows.extend(
                _metric_rows(
                    config,
                    seed,
                    d2.mechanism,
                    channel,
                    method,
                    "dynamic_local",
                    metrics,
                    access,
                    time.perf_counter() - started,
                )
            )
        if sbtg_profile is not None and channel != "mean":
            correlation = (
                float(np.corrcoef(sbtg_profile, np.abs(truth))[0, 1])
                if np.std(sbtg_profile) > 0 and np.std(np.abs(truth)) > 0
                else float("nan")
            )
            rows.extend(
                _metric_rows(
                    config,
                    seed,
                    d2.mechanism,
                    channel,
                    "ORIGINAL_SBTG_K_NORM",
                    "dynamic_operator",
                    {"absolute_lag_profile_correlation": correlation},
                    "response_score",
                    time.perf_counter() - started,
                )
            )

        truth_finite = np.mean(
            np.stack([d2.finite_effect(eval2, lag, delta, channel) for lag in range(lags)], axis=1),
            axis=0,
        )
        direct_finite = np.mean(direct2.finite_effect(eval2, delta, channel), axis=0)
        mixture_finite = np.mean(mixture.finite_effect(eval2, delta, channel), axis=0)
        finite_methods = {
            "B0_ZERO": (np.zeros(lags), "observational"),
            "B2_RAW_MOMENTS_FINITE": (direct_finite, "observational"),
            "B7_MDN_CORRECT_ENDPOINTS": (mixture_finite, "normalized_density"),
        }
        if channel != "mean":
            finite_methods["A8_ENDPOINT_CLASSIFIER"] = (
                finite_classifier_profile(d2, eval2[:24], channel, delta, rng),
                "endpoint_labels_and_queries",
            )
        for method, (estimate, access) in finite_methods.items():
            arrays[f"d2__{channel}__finite__{method.lower()}__profile"] = estimate
            if channel == "mean":
                metrics = {
                    "mean_null_rms": float(np.sqrt(np.mean(estimate**2))),
                    "mean_null_max_abs": float(np.max(np.abs(estimate))),
                }
            else:
                metrics = lag_metrics(estimate, truth_finite)
            rows.extend(
                _metric_rows(
                    config,
                    seed,
                    d2.mechanism,
                    channel,
                    method,
                    "dynamic_finite",
                    metrics,
                    access,
                    time.perf_counter() - started,
                )
            )
        arrays[f"d2__{channel}__finite__truth_profile"] = truth_finite
    return pd.DataFrame(rows), arrays
