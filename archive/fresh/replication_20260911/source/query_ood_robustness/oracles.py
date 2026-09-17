from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class OracleControl:
    name: str
    description: str
    sample: Callable[[int, int, float], Array]
    source: int = 0
    target: int = 1


def _normal_sampler(mean: Array, covariance: Array) -> Callable[[int, int, float], Array]:
    chol = np.linalg.cholesky(covariance)

    def sample(n: int, seed: int, history: float) -> Array:
        shifted = np.asarray(mean, dtype=float).copy()
        shifted[0] += 0.3 * history
        shifted[1] += 0.2 * np.tanh(history)
        return shifted + np.random.default_rng(seed).normal(size=(n, len(mean))) @ chol.T

    return sample


def oracle_controls() -> list[OracleControl]:
    dimension = 4
    identity = np.eye(dimension)
    correlated = np.array(
        [
            [1.0, 0.75, 0.15, 0.0],
            [0.75, 1.0, 0.25, 0.1],
            [0.15, 0.25, 1.0, 0.55],
            [0.0, 0.1, 0.55, 1.0],
        ]
    )

    def history_diagonal(n: int, seed: int, history: float) -> Array:
        scale = np.array([0.55 + 0.30 * abs(history), 0.8, 1.1, 0.65])
        mean = np.array([0.35 * history, 0.25 * np.tanh(history), 0.0, 0.0])
        return mean + np.random.default_rng(seed).normal(size=(n, dimension)) * scale

    def history_lowrank(n: int, seed: int, history: float) -> Array:
        rng = np.random.default_rng(seed)
        factor = np.array([0.8, 0.65 * np.tanh(1 + history), 0.2, -0.25])
        diagonal = np.array([0.5, 0.55, 0.8, 0.75])
        return (
            np.array([0.2 * history, 0.1 * history**2, 0.0, 0.0])
            + rng.normal(size=(n, dimension)) * diagonal
            + rng.normal(size=(n, 1)) * factor
        )

    def student(n: int, seed: int, history: float) -> Array:
        rng = np.random.default_rng(seed)
        z = rng.normal(size=(n, dimension)) @ np.linalg.cholesky(correlated).T
        chi = rng.chisquare(4.5, size=(n, 1))
        return np.array([0.25 * history, 0.2 * np.tanh(history), 0, 0]) + z * np.sqrt(2.5 / chi)

    def skewed(n: int, seed: int, history: float) -> Array:
        rng = np.random.default_rng(seed)
        latent = rng.normal(size=(n, 1))
        base = rng.normal(scale=0.55, size=(n, dimension))
        base[:, 0] += np.exp(0.65 * latent[:, 0]) - np.exp(0.65**2 / 2)
        base[:, 1] += 0.7 * latent[:, 0] + 0.18 * base[:, 0] ** 2
        base[:, 0] += 0.25 * history
        return base

    def mixture(n: int, seed: int, history: float) -> Array:
        rng = np.random.default_rng(seed)
        probability = 1 / (1 + np.exp(-1.2 * history))
        component = rng.random(n) < probability
        means = np.where(component[:, None], np.array([1.1, 0.9, -0.2, 0.3]),
                         np.array([-0.8, -0.5, 0.25, -0.2]))
        noise = rng.normal(scale=np.array([0.55, 0.5, 0.75, 0.65]), size=(n, dimension))
        noise[:, 1] += 0.45 * noise[:, 0]
        return means + noise

    def nonlinear_mean(n: int, seed: int, history: float) -> Array:
        mean = np.array(
            [0.4 * history, 0.7 * np.tanh(1.3 * history) + 0.15 * history**2, 0.0, 0.0]
        )
        return mean + np.random.default_rng(seed).normal(size=(n, dimension)) @ np.linalg.cholesky(correlated).T

    return [
        OracleControl("diagonal_gaussian", "diagonal Gaussian", _normal_sampler(np.zeros(4), identity)),
        OracleControl("correlated_gaussian", "correlated Gaussian", _normal_sampler(np.zeros(4), correlated)),
        OracleControl("history_diagonal_variance", "history-dependent diagonal variance", history_diagonal),
        OracleControl("history_lowrank_covariance", "history-dependent low-rank covariance", history_lowrank),
        OracleControl("student_t", "finite-variance Student-t tails", student),
        OracleControl("skewed", "skewed residual distribution", skewed),
        OracleControl("history_mixture", "history-dependent mixture occupancy", mixture),
        OracleControl("nonlinear_mean", "nonlinear conditional mean", nonlinear_mean),
    ]


def gaussian_soft_constraint_truth(
    mean: Array,
    covariance: Array,
    *,
    source: int,
    target: int,
    requested: float,
    bandwidth: float,
) -> float:
    """Exact E[Y_target] under a Gaussian observation of Y_source."""
    gain = covariance[target, source] / (
        covariance[source, source] + bandwidth**2
    )
    return float(mean[target] + gain * (requested - mean[source]))


def oracle_soft_effect(
    control: OracleControl,
    *,
    history: float,
    low_target: float,
    high_target: float,
    bandwidth: float,
    n_reference: int,
    seed: int,
    chunk: int = 200_000,
) -> tuple[float, float, float]:
    """High-precision self-normalized oracle effect and conservative MCSE."""
    moments: dict[str, list[float]] = {
        "low_w": [], "low_wy": [], "low_wy2": [],
        "high_w": [], "high_wy": [], "high_wy2": [],
    }
    remaining = int(n_reference)
    part = 0
    while remaining:
        count = min(chunk, remaining)
        draw = control.sample(count, seed + 104729 * part, history)
        response = draw[:, control.target]
        for label, requested in (("low", low_target), ("high", high_target)):
            weight = np.exp(-0.5 * np.square((draw[:, control.source] - requested) / bandwidth))
            moments[f"{label}_w"].append(float(weight.sum()))
            moments[f"{label}_wy"].append(float(np.sum(weight * response)))
            moments[f"{label}_wy2"].append(float(np.sum(weight * response**2)))
        remaining -= count
        part += 1
    estimates = {}
    variances = {}
    probabilities = {}
    for label in ("low", "high"):
        weight = sum(moments[f"{label}_w"])
        mean = sum(moments[f"{label}_wy"]) / weight
        variance = max(0.0, sum(moments[f"{label}_wy2"]) / weight - mean**2)
        probability = weight / n_reference
        effective = max(1.0, n_reference * probability)
        estimates[label] = mean
        variances[label] = variance / effective
        probabilities[label] = probability
    return (
        float(estimates["high"] - estimates["low"]),
        float(np.sqrt(variances["high"] + variances["low"])),
        float(min(probabilities.values())),
    )
