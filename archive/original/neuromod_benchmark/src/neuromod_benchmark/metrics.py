"""Metrics matched to predictive, mechanistic, causal, and change estimands."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from .schema import Prediction


EPS = 1e-10


def gaussian_log_prob(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    variance = np.maximum(np.asarray(variance, dtype=float), EPS)
    return -0.5 * (np.log(2 * np.pi * variance) + (np.asarray(y) - mean) ** 2 / variance)


def gaussian_crps(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Closed-form CRPS; lower is better and the score is strictly proper."""

    sd = np.sqrt(np.maximum(np.asarray(variance, dtype=float), EPS))
    z = (np.asarray(y) - mean) / sd
    return sd * (z * (2 * ndtr(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def pit_ece(pit: np.ndarray, bins: int = 10) -> float:
    """L1 deviation of the PIT histogram from uniformity."""

    values = np.asarray(pit, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    counts, _ = np.histogram(values, bins=np.linspace(0, 1, bins + 1))
    proportions = counts / counts.sum()
    return float(np.mean(np.abs(proportions - 1.0 / bins)))


def pit_cvm(pit: np.ndarray) -> float:
    values = np.sort(np.asarray(pit, dtype=float).ravel())
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return float("nan")
    expected = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    return float(1.0 / (12 * n) + np.sum((values - expected) ** 2))


def interval_coverage_error(
    y: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    level: float = 0.9,
) -> float:
    z = norm.ppf((1 + level) / 2)
    sd = np.sqrt(np.maximum(variance, EPS))
    coverage = np.mean((y >= mean - z * sd) & (y <= mean + z * sd))
    return float(abs(coverage - level))


def pit_interval_coverage_error(pit: np.ndarray, level: float = 0.9) -> float:
    """Coverage error of the forecast's own equal-tailed central interval.

    For a continuous predictive CDF, ``alpha/2 <= F(Y) <= 1-alpha/2`` is exactly
    the event that ``Y`` lies in that distribution's central interval.  This avoids
    imposing Gaussian quantiles on Student-t or mixture forecasts.
    """

    values = np.asarray(pit, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    alpha = 1.0 - float(level)
    coverage = np.mean((values >= alpha / 2.0) & (values <= 1.0 - alpha / 2.0))
    return float(abs(coverage - level))


def energy_score(y: np.ndarray, samples: np.ndarray, max_pairs: int = 64) -> float:
    """Fair finite-ensemble multivariate energy score.

    The second term is the off-diagonal U-statistic. Including self-pairs would
    make methods with different ensemble sizes incomparable.
    """

    y = np.asarray(y, dtype=float)
    draws = np.asarray(samples, dtype=float)
    if draws.ndim != 3 or draws.shape[0] != y.shape[0] or draws.shape[2] != y.shape[1]:
        raise ValueError("samples must have shape [observation, draw, target]")
    if draws.shape[1] > max_pairs:
        draws = draws[:, :max_pairs]
    first = np.linalg.norm(draws - y[:, None, :], axis=2).mean(axis=1)
    m = draws.shape[1]
    if m < 2:
        raise ValueError("fair energy score requires at least two draws")
    pairwise = np.linalg.norm(draws[:, :, None, :] - draws[:, None, :, :], axis=3)
    off_diagonal = np.sum(pairwise, axis=(1, 2)) / (m * (m - 1))
    return float(np.mean(first - 0.5 * off_diagonal))


def ensemble_crps(y: np.ndarray, samples: np.ndarray) -> float:
    """Fair finite-ensemble CRPS averaged over observations and targets."""

    y = np.asarray(y, dtype=float)
    draws = np.asarray(samples, dtype=float)
    if draws.ndim != 3 or draws.shape[0] != y.shape[0] or draws.shape[2] != y.shape[1]:
        raise ValueError("samples must have shape [observation, draw, target]")
    m = draws.shape[1]
    if m < 2:
        raise ValueError("fair ensemble CRPS requires at least two draws")
    first = np.mean(np.abs(draws - y[:, None, :]), axis=1)
    pairwise = np.abs(draws[:, :, None, :] - draws[:, None, :, :])
    second = np.sum(pairwise, axis=(1, 2)) / (m * (m - 1))
    return float(np.mean(first - 0.5 * second))


def pit_serial_metrics(pit: np.ndarray, groups: np.ndarray, max_lag: int = 4) -> dict[str, float]:
    """Serial PIT diagnostics computed within episodes only."""

    values = np.asarray(pit, dtype=float)
    if values.ndim == 2:
        values = values.mean(axis=1)
    groups = np.asarray(groups)
    centered_products = {lag: [] for lag in range(1, max_lag + 1)}
    squared_products = {lag: [] for lag in range(1, max_lag + 1)}
    for group in np.unique(groups):
        u = values[groups == group] - 0.5
        squared = u**2 - 1.0 / 12.0
        for lag in range(1, max_lag + 1):
            if len(u) > lag:
                centered_products[lag].append(u[lag:] * u[:-lag])
                squared_products[lag].append(squared[lag:] * squared[:-lag])
    result = {}
    for lag in range(1, max_lag + 1):
        result[f"pit_centered_product_lag{lag}"] = (
            float(np.mean(np.concatenate(centered_products[lag])))
            if centered_products[lag]
            else float("nan")
        )
        result[f"pit_squared_product_lag{lag}"] = (
            float(np.mean(np.concatenate(squared_products[lag])))
            if squared_products[lag]
            else float("nan")
        )
    return result


def predictive_metrics(prediction: Prediction, y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    mean = np.asarray(prediction.mean, dtype=float)
    variance = np.maximum(np.asarray(prediction.variance, dtype=float), EPS)
    if y.shape != mean.shape or y.shape != variance.shape:
        raise ValueError("target, mean, and variance shapes must match")
    log_prob = prediction.log_prob
    pit = prediction.cdf
    declared_family = prediction.metadata.get("distribution_family")
    if log_prob is None or pit is None:
        if declared_family != "gaussian":
            raise ValueError(
                "a normalized forecast must provide log_prob and cdf, or explicitly "
                "declare distribution_family='gaussian'"
            )
        log_prob = gaussian_log_prob(y, mean, variance)
        pit = ndtr((y - mean) / np.sqrt(variance))
    result = {
        "nll": float(-np.mean(log_prob)),
        "crps_gaussian_moment": float(np.mean(gaussian_crps(y, mean, variance))),
        "rmse_mean": float(np.sqrt(np.mean((y - mean) ** 2))),
        "pit_ece": pit_ece(pit),
        "pit_cvm": pit_cvm(pit),
        "coverage_error_90": pit_interval_coverage_error(pit, 0.9),
    }
    if prediction.samples is not None:
        result["energy_score"] = energy_score(y, prediction.samples)
        result["crps_ensemble_fair"] = ensemble_crps(y, prediction.samples)
    for name in (
        "invalid_variance_fraction",
        "near_natural_variance_boundary_fraction",
        "extreme_variance_fraction",
        "standardized_variance_p99",
        "numerical_valid",
        "min_component_usage",
        "component_entropy",
    ):
        value = prediction.metadata.get(name)
        if value is not None:
            result[name] = float(value)
    return result


def _edge_vectors(
    truth: np.ndarray,
    score: np.ndarray,
    exclude_diagonal: bool,
    beta_min: float | None,
):
    truth = np.asarray(truth)
    score = np.asarray(score, dtype=float)
    if truth.shape != score.shape or truth.ndim != 2:
        raise ValueError("truth and score must be same-shaped matrices")
    mask = np.ones(truth.shape, dtype=bool)
    if exclude_diagonal and truth.shape[0] == truth.shape[1]:
        np.fill_diagonal(mask, False)
    truth_values = np.abs(truth[mask])
    if beta_min is None:
        beta_min = max(EPS, 0.05 * float(np.max(truth_values, initial=0.0)))
    return (truth_values >= beta_min).astype(int), np.abs(score[mask]), float(beta_min)


def graph_metrics(
    truth: np.ndarray,
    score: np.ndarray,
    *,
    exclude_diagonal: bool = True,
    beta_min: float | None = None,
) -> dict[str, float]:
    labels, values, beta_min = _edge_vectors(
        truth, score, exclude_diagonal, beta_min
    )
    prevalence = float(labels.mean()) if len(labels) else float("nan")
    result = {"prevalence": prevalence, "beta_min": beta_min}
    if labels.sum() == 0:
        result.update({"auprc": float("nan"), "auroc": float("nan")})
        result["null_max_score"] = float(np.max(values)) if len(values) else float("nan")
        result["null_mean_score"] = float(np.mean(values)) if len(values) else float("nan")
        return result
    result["auprc"] = float(average_precision_score(labels, values))
    result["auprc_lift_over_prevalence"] = float(result["auprc"] - prevalence)
    result["auroc"] = (
        float(roc_auc_score(labels, values)) if labels.sum() < len(labels) else float("nan")
    )
    k = int(labels.sum())
    threshold = np.sort(values)[-k]
    definite = values > threshold
    tied = values == threshold
    needed = k - int(np.sum(definite))
    expected_true_ties = (
        needed * float(np.mean(labels[tied])) if np.any(tied) else 0.0
    )
    result["precision_at_true_k"] = float(
        (np.sum(labels[definite]) + expected_true_ties) / k
    )
    return result


def matrix_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if truth.shape != estimate.shape:
        raise ValueError("matrix shapes do not match")
    error = estimate - truth
    denom = np.linalg.norm(truth)
    active = np.abs(truth) > EPS
    cosine_denom = np.linalg.norm(truth.ravel()) * np.linalg.norm(estimate.ravel())
    return {
        "relative_frobenius": float(np.linalg.norm(error) / max(denom, EPS)),
        "mae": float(np.mean(np.abs(error))),
        "cosine": float(np.dot(truth.ravel(), estimate.ravel()) / max(cosine_denom, EPS)),
        "sign_accuracy_active": (
            float(np.mean(np.sign(truth[active]) == np.sign(estimate[active])))
            if np.any(active)
            else float("nan")
        ),
    }


def counterfactual_metrics(
    factual: np.ndarray,
    counterfactual: np.ndarray,
    factual_hat: np.ndarray,
    counterfactual_hat: np.ndarray,
) -> dict[str, float]:
    """Recovery of paired conditional-mean contrasts on matched arm histories."""

    effect = np.asarray(factual) - np.asarray(counterfactual)
    effect_hat = np.asarray(factual_hat) - np.asarray(counterfactual_hat)
    return {
        "effect_rmse": float(np.sqrt(np.mean((effect - effect_hat) ** 2))),
        "effect_bias": float(np.mean(effect_hat - effect)),
        "effect_correlation": float(
            np.corrcoef(effect.ravel(), effect_hat.ravel())[0, 1]
            if np.std(effect) > EPS and np.std(effect_hat) > EPS
            else np.nan
        ),
    }


def changepoint_metrics(
    true_index: int | None,
    detected_indices: list[int],
    length: int,
    tolerance: int,
) -> dict[str, Any]:
    detected = sorted(int(i) for i in detected_indices if 0 <= i < length)
    if true_index is None:
        return {
            "false_positive": float(bool(detected)),
            "n_false_alarms": float(len(detected)),
            "detected": float("nan"),
            "delay": float("nan"),
        }
    candidates = [i for i in detected if i >= true_index - tolerance]
    nearest = min(candidates, key=lambda i: abs(i - true_index)) if candidates else None
    return {
        "false_positive": float(any(i < true_index - tolerance for i in detected)),
        "n_false_alarms": float(sum(i < true_index - tolerance for i in detected)),
        "detected": float(nearest is not None and abs(nearest - true_index) <= tolerance),
        "delay": float(nearest - true_index) if nearest is not None else float("nan"),
    }
