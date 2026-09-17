from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, ndtri
from scipy.special import eval_legendre
from scipy.stats import qmc
from sklearn.model_selection import KFold


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path, excluded: Iterable[str] = ("runs", "reports", "__pycache__")) -> str:
    excluded = set(excluded)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.relative_to(root).parts):
            continue
        relative = str(path.relative_to(root)).encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def resource_guard(min_free_gib: float = 1.0, max_rss_gib: float = 10.0) -> dict[str, float]:
    import psutil

    usage = shutil.disk_usage(Path.cwd())
    free_gib = usage.free / 2**30
    rss_gib = psutil.Process(os.getpid()).memory_info().rss / 2**30
    if free_gib < min_free_gib:
        raise RuntimeError(f"disk guard: {free_gib:.3f} GiB free < {min_free_gib:.3f} GiB")
    if rss_gib > max_rss_gib:
        raise RuntimeError(f"memory guard: {rss_gib:.3f} GiB RSS > {max_rss_gib:.3f} GiB")
    return {"free_disk_gib": free_gib, "rss_gib": rss_gib}


@dataclass(frozen=True)
class CharacteristicBank:
    median: np.ndarray
    scale: np.ndarray
    frequencies: np.ndarray
    seed: int

    @classmethod
    def fit(
        cls,
        y_train: np.ndarray,
        q: int,
        seed: int,
        amplitude_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    ) -> "CharacteristicBank":
        y_train = np.asarray(y_train, dtype=float)
        if y_train.ndim == 1:
            y_train = y_train[:, None]
        median = np.median(y_train, axis=0)
        mad = np.median(np.abs(y_train - median), axis=0)
        scale = np.maximum(1.4826 * mad, 0.1)
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(q, y_train.shape[1]))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        amplitude = np.resize(np.asarray(amplitude_scales, dtype=float), q)
        amplitude = amplitude * rng.uniform(0.95, 1.05, size=q)
        frequencies = directions * amplitude[:, None]
        return cls(median=median, scale=scale, frequencies=frequencies, seed=int(seed))

    @property
    def q(self) -> int:
        return int(self.frequencies.shape[0])

    @property
    def output_dim(self) -> int:
        return 2 * self.q

    @property
    def raw_frequencies(self) -> np.ndarray:
        return self.frequencies / self.scale[None, :]

    @property
    def raw_phase_offset(self) -> np.ndarray:
        return -(self.raw_frequencies @ self.median)

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        phase = ((y - self.median) / self.scale) @ self.frequencies.T
        return np.concatenate([np.cos(phase), np.sin(phase)], axis=1)

    def digest(self) -> str:
        digest = hashlib.sha256()
        for array in (self.median, self.scale, self.frequencies):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        digest.update(str(self.seed).encode())
        return digest.hexdigest()

    def save(self, path: Path) -> None:
        atomic_npz(
            path,
            median=self.median,
            scale=self.scale,
            frequencies=self.frequencies,
            seed=np.array([self.seed], dtype=np.int64),
        )


class HistorySieve:
    """Differentiable frozen random-feature sieve with fold-specific scaling."""

    def __init__(self, input_dim: int, random_width: int, seed: int) -> None:
        self.input_dim = int(input_dim)
        self.random_width = int(random_width)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.random_weights = rng.normal(
            scale=1.0 / math.sqrt(max(1, input_dim)), size=(random_width, input_dim)
        )
        self.random_bias = rng.uniform(-1.0, 1.0, size=random_width)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.input_dim + self.random_width

    def fit(self, h: np.ndarray) -> "HistorySieve":
        h = np.asarray(h, dtype=float)
        self.mean_ = np.mean(h, axis=0)
        self.scale_ = np.maximum(np.std(h, axis=0), 1e-6)
        return self

    def _standardize(self, h: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("sieve must be fit before transform")
        return (np.asarray(h, dtype=float) - self.mean_) / self.scale_

    def transform(self, h: np.ndarray) -> np.ndarray:
        z = self._standardize(h)
        nonlinear = np.tanh(z @ self.random_weights.T + self.random_bias)
        return np.concatenate(
            [np.ones((z.shape[0], 1)), z, z**2 - 1.0, nonlinear], axis=1
        )

    def derivative(self, h: np.ndarray, direction: int) -> np.ndarray:
        z = self._standardize(h)
        direction = int(direction)
        inv_scale = 1.0 / float(self.scale_[direction])
        derivative = np.zeros((z.shape[0], self.output_dim), dtype=float)
        derivative[:, 1 + direction] = inv_scale
        derivative[:, 1 + self.input_dim + direction] = 2.0 * z[:, direction] * inv_scale
        nonlinear = np.tanh(z @ self.random_weights.T + self.random_bias)
        derivative[:, 1 + 2 * self.input_dim :] = (
            (1.0 - nonlinear**2)
            * self.random_weights[:, direction][None, :]
            * inv_scale
        )
        return derivative


def ridge_coefficients(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    gram = design.T @ design / design.shape[0]
    penalty = float(ridge) * np.eye(gram.shape[0])
    penalty[0, 0] = float(ridge) * 0.01
    right = design.T @ target / design.shape[0]
    try:
        return np.linalg.solve(gram + penalty, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + penalty, right, rcond=1e-10)[0]


def riesz_coefficients(
    design: np.ndarray, derivative_design: np.ndarray, ridge: float
) -> np.ndarray:
    design = np.asarray(design, dtype=float)
    derivative_design = np.asarray(derivative_design, dtype=float)
    gram = design.T @ design / design.shape[0]
    penalty = float(ridge) * np.eye(gram.shape[0])
    penalty[0, 0] = float(ridge) * 0.01
    right = np.mean(derivative_design, axis=0)
    try:
        return np.linalg.solve(gram + penalty, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + penalty, right, rcond=1e-10)[0]


class AnalyticGaussianPathDGP:
    def __init__(self, seed: int, history_dim: int = 8, response_dim: int = 4, rank: int = 3) -> None:
        self.seed = int(seed)
        self.history_dim = int(history_dim)
        self.response_dim = int(response_dim)
        self.rank = int(rank)
        rng = np.random.default_rng(9173 + 1009 * seed)
        self.mean_linear = rng.normal(scale=0.16, size=(response_dim, history_dim))
        self.mean_frequency = rng.normal(
            scale=0.45 / math.sqrt(history_dim), size=(response_dim, history_dim)
        )
        self.mean_nonlinear = rng.uniform(0.12, 0.28, size=response_dim)
        raw = rng.normal(size=(response_dim, response_dim))
        self.base_covariance = 0.35 * np.eye(response_dim) + 0.03 * (raw @ raw.T) / response_dim
        self.cov_offset = rng.normal(loc=-0.4, scale=0.25, size=rank)
        self.cov_history = rng.normal(scale=0.38 / math.sqrt(history_dim), size=(rank, history_dim))
        directions = rng.normal(size=(rank, response_dim))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        self.cov_directions = 0.32 * directions

    def sample_histories(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(size=(int(n), self.history_dim))

    def conditional_mean(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        phase = h @ self.mean_frequency.T
        return h @ self.mean_linear.T + np.sin(phase) * self.mean_nonlinear

    def conditional_mean_derivative(self, h: np.ndarray, direction: int) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        phase = h @ self.mean_frequency.T
        return self.mean_linear[:, direction][None, :] + (
            np.cos(phase)
            * self.mean_nonlinear[None, :]
            * self.mean_frequency[:, direction][None, :]
        )

    def conditional_covariance(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        logits = self.cov_offset[None, :] + h @ self.cov_history.T
        weights = np.logaddexp(0.0, logits)
        outer = np.einsum("ki,kj->kij", self.cov_directions, self.cov_directions)
        return self.base_covariance[None, :, :] + np.einsum("nk,kij->nij", weights, outer)

    def conditional_covariance_derivative(self, h: np.ndarray, direction: int) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        logits = self.cov_offset[None, :] + h @ self.cov_history.T
        dweights = expit(logits) * self.cov_history[:, direction][None, :]
        outer = np.einsum("ki,kj->kij", self.cov_directions, self.cov_directions)
        return np.einsum("nk,kij->nij", dweights, outer)

    def sample_responses(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean = self.conditional_mean(h)
        covariance = self.conditional_covariance(h)
        noise = rng.normal(size=mean.shape)
        chol = np.linalg.cholesky(covariance)
        return mean + np.einsum("nij,nj->ni", chol, noise)

    def conditional_feature_mean(self, h: np.ndarray, bank: CharacteristicBank) -> np.ndarray:
        mean = self.conditional_mean(h)
        covariance = self.conditional_covariance(h)
        u = bank.raw_frequencies
        phase = mean @ u.T + bank.raw_phase_offset[None, :]
        variance = np.einsum("qi,nij,qj->nq", u, covariance, u)
        complex_mean = np.exp(1j * phase - 0.5 * variance)
        return np.concatenate([complex_mean.real, complex_mean.imag], axis=1)

    def conditional_feature_derivative(
        self, h: np.ndarray, direction: int, bank: CharacteristicBank
    ) -> np.ndarray:
        mean = self.conditional_mean(h)
        covariance = self.conditional_covariance(h)
        dmean = self.conditional_mean_derivative(h, direction)
        dcovariance = self.conditional_covariance_derivative(h, direction)
        u = bank.raw_frequencies
        phase = mean @ u.T + bank.raw_phase_offset[None, :]
        variance = np.einsum("qi,nij,qj->nq", u, covariance, u)
        complex_mean = np.exp(1j * phase - 0.5 * variance)
        factor = 1j * (dmean @ u.T) - 0.5 * np.einsum("qi,nij,qj->nq", u, dcovariance, u)
        derivative = complex_mean * factor
        return np.concatenate([derivative.real, derivative.imag], axis=1)

    def oracle_riesz(self, h: np.ndarray, direction: int) -> np.ndarray:
        return np.asarray(h, dtype=float)[:, int(direction)]

    def qmc_histories(self, power: int, scramble_seed: int) -> np.ndarray:
        sampler = qmc.Sobol(d=self.history_dim, scramble=True, seed=scramble_seed)
        unit = sampler.random_base2(m=int(power))
        unit = np.clip(unit, 1e-12, 1.0 - 1e-12)
        return ndtri(unit)


class MomentBlindDiscreteDGP:
    """Smoothed discrete tilt whose raw moment derivatives through order k vanish."""

    family = "finite_difference_mixture"

    def __init__(
        self,
        seed: int,
        k: int = 4,
        smoothing_sigma: float = 0.2,
        amplitude_fraction: float = 0.5,
        spacing: float = 1.0,
    ) -> None:
        self.seed = int(seed)
        self.k = int(k)
        self.history_dim = 1
        self.response_dim = 1
        self.smoothing_sigma = float(smoothing_sigma)
        support = spacing * np.arange(k + 2, dtype=float)
        self.support = support - np.mean(support)
        coefficients = np.array(
            [(-1.0) ** j * math.comb(k + 1, j) for j in range(k + 2)], dtype=float
        )
        self.coefficients = coefficients / np.sum(np.abs(coefficients))
        self.base_weights = np.full(k + 2, 1.0 / (k + 2))
        positivity_limit = np.min(
            self.base_weights[np.abs(self.coefficients) > 0]
            / np.abs(self.coefficients[np.abs(self.coefficients) > 0])
        )
        self.amplitude = float(amplitude_fraction) * float(positivity_limit)

    def sample_histories(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(size=(int(n), 1))

    def _weights(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape(-1)
        weights = self.base_weights[None, :] + self.amplitude * np.tanh(h)[:, None] * self.coefficients[None, :]
        if np.min(weights) <= 0:
            raise RuntimeError("moment-blind mixture violated positivity")
        return weights

    def sample_responses(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        weights = self._weights(h)
        uniforms = rng.random(weights.shape[0])
        categories = np.sum(uniforms[:, None] > np.cumsum(weights, axis=1), axis=1)
        y = self.support[categories] + self.smoothing_sigma * rng.normal(size=weights.shape[0])
        return y[:, None]

    def _complex_parts(self, bank: CharacteristicBank) -> tuple[np.ndarray, np.ndarray]:
        u = bank.raw_frequencies[:, 0]
        phase = np.exp(1j * bank.raw_phase_offset)
        smoothing = np.exp(-0.5 * self.smoothing_sigma**2 * u**2)
        atoms = np.exp(1j * u[:, None] * self.support[None, :])
        base = phase * smoothing * (atoms @ self.base_weights)
        tilt = phase * smoothing * (atoms @ self.coefficients)
        return base, tilt

    def conditional_feature_mean(self, h: np.ndarray, bank: CharacteristicBank) -> np.ndarray:
        base, tilt = self._complex_parts(bank)
        complex_mean = base[None, :] + self.amplitude * np.tanh(np.asarray(h)[:, 0])[:, None] * tilt[None, :]
        return np.concatenate([complex_mean.real, complex_mean.imag], axis=1)

    def conditional_feature_derivative(self, h: np.ndarray, direction: int, bank: CharacteristicBank) -> np.ndarray:
        if int(direction) != 0:
            raise ValueError("moment-blind DGP has one history direction")
        _, tilt = self._complex_parts(bank)
        history = np.asarray(h)[:, 0]
        derivative = self.amplitude * (1.0 - np.tanh(history) ** 2)[:, None] * tilt[None, :]
        return np.concatenate([derivative.real, derivative.imag], axis=1)

    def oracle_riesz(self, h: np.ndarray, direction: int) -> np.ndarray:
        return np.asarray(h, dtype=float)[:, 0]

    def moment_derivative(self, order: int) -> float:
        if int(order) <= self.k:
            return 0.0
        return self.amplitude * float(np.sum(self.coefficients * self.support**int(order)))


class MomentBlindLegendreDGP:
    """Continuous Legendre tilt orthogonal to polynomial moments through order k."""

    family = "continuous_legendre_tilt"

    def __init__(self, seed: int, k: int = 4, amplitude: float = 0.5) -> None:
        self.seed = int(seed)
        self.k = int(k)
        self.degree = k + 1
        self.amplitude = float(amplitude)
        self.history_dim = 1
        self.response_dim = 1
        nodes, weights = np.polynomial.legendre.leggauss(256)
        self.quadrature_nodes = nodes
        self.quadrature_weights = weights
        self.legendre_values = eval_legendre(self.degree, nodes)

    def sample_histories(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(size=(int(n), 1))

    def sample_responses(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        history = np.asarray(h, dtype=float)[:, 0]
        strength = self.amplitude * np.tanh(history)
        output = np.empty(history.size)
        remaining = np.arange(history.size)
        while remaining.size:
            proposal = rng.uniform(-1.0, 1.0, size=remaining.size)
            numerator = 1.0 + strength[remaining] * eval_legendre(self.degree, proposal)
            denominator = 1.0 + np.abs(strength[remaining])
            accept = rng.random(remaining.size) < numerator / denominator
            output[remaining[accept]] = proposal[accept]
            remaining = remaining[~accept]
        return output[:, None]

    def _complex_parts(self, bank: CharacteristicBank) -> tuple[np.ndarray, np.ndarray]:
        u = bank.raw_frequencies[:, 0]
        phase = np.exp(1j * bank.raw_phase_offset)
        oscillation = np.exp(1j * u[:, None] * self.quadrature_nodes[None, :])
        base = phase * 0.5 * (oscillation @ self.quadrature_weights)
        tilt = phase * 0.5 * (
            oscillation @ (self.quadrature_weights * self.legendre_values)
        )
        return base, tilt

    def conditional_feature_mean(self, h: np.ndarray, bank: CharacteristicBank) -> np.ndarray:
        base, tilt = self._complex_parts(bank)
        complex_mean = base[None, :] + self.amplitude * np.tanh(np.asarray(h)[:, 0])[:, None] * tilt[None, :]
        return np.concatenate([complex_mean.real, complex_mean.imag], axis=1)

    def conditional_feature_derivative(self, h: np.ndarray, direction: int, bank: CharacteristicBank) -> np.ndarray:
        if int(direction) != 0:
            raise ValueError("moment-blind DGP has one history direction")
        _, tilt = self._complex_parts(bank)
        history = np.asarray(h)[:, 0]
        derivative = self.amplitude * (1.0 - np.tanh(history) ** 2)[:, None] * tilt[None, :]
        return np.concatenate([derivative.real, derivative.imag], axis=1)

    def oracle_riesz(self, h: np.ndarray, direction: int) -> np.ndarray:
        return np.asarray(h, dtype=float)[:, 0]

    def moment_derivative(self, order: int) -> float:
        if int(order) <= self.k:
            return 0.0
        integrand = self.quadrature_nodes**int(order) * self.legendre_values
        return self.amplitude * 0.5 * float(np.sum(self.quadrature_weights * integrand))


class SupportMotionDGP:
    """Conditional location law with a well-defined deterministic limit."""

    family = "support_motion"

    def __init__(self, seed: int, noise_sigma: float) -> None:
        self.seed = int(seed)
        self.noise_sigma = float(noise_sigma)
        self.history_dim = 1
        self.response_dim = 1

    def sample_histories(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(size=(int(n), 1))

    @staticmethod
    def location(h: np.ndarray) -> np.ndarray:
        value = np.asarray(h, dtype=float)[:, 0]
        return value + 0.3 * np.sin(2.0 * value)

    @staticmethod
    def location_derivative(h: np.ndarray) -> np.ndarray:
        value = np.asarray(h, dtype=float)[:, 0]
        return 1.0 + 0.6 * np.cos(2.0 * value)

    def sample_responses(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return (
            self.location(h) + self.noise_sigma * rng.normal(size=np.asarray(h).shape[0])
        )[:, None]

    def conditional_feature_mean(self, h: np.ndarray, bank: CharacteristicBank) -> np.ndarray:
        u = bank.raw_frequencies[:, 0]
        phase = self.location(h)[:, None] * u[None, :] + bank.raw_phase_offset[None, :]
        complex_mean = np.exp(1j * phase - 0.5 * self.noise_sigma**2 * u[None, :] ** 2)
        return np.concatenate([complex_mean.real, complex_mean.imag], axis=1)

    def conditional_feature_derivative(self, h: np.ndarray, direction: int, bank: CharacteristicBank) -> np.ndarray:
        if int(direction) != 0:
            raise ValueError("support-motion DGP has one history direction")
        u = bank.raw_frequencies[:, 0]
        phase = self.location(h)[:, None] * u[None, :] + bank.raw_phase_offset[None, :]
        complex_mean = np.exp(1j * phase - 0.5 * self.noise_sigma**2 * u[None, :] ** 2)
        derivative = complex_mean * (1j * self.location_derivative(h)[:, None] * u[None, :])
        return np.concatenate([derivative.real, derivative.imag], axis=1)

    def oracle_riesz(self, h: np.ndarray, direction: int) -> np.ndarray:
        return np.asarray(h, dtype=float)[:, 0]

    def likelihood_history_score(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.noise_sigma == 0:
            raise ValueError("likelihood history score is unavailable at sigma=0")
        residual = np.asarray(y, dtype=float)[:, 0] - self.location(h)
        return residual * self.location_derivative(h) / self.noise_sigma**2


def split_overlap(train_raw: Iterable[int], eval_raw: Iterable[int]) -> np.ndarray:
    return np.intersect1d(np.fromiter(train_raw, dtype=np.int64), np.fromiter(eval_raw, dtype=np.int64))


def embargoed_block_splits(n_time: int, n_folds: int, footprint: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(int(n_time))
    blocks = np.array_split(indices, int(n_folds))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        lo = max(0, int(block[0]) - int(footprint))
        hi = min(int(n_time), int(block[-1]) + 1 + int(footprint))
        mask = np.ones(n_time, dtype=bool)
        mask[lo:hi] = False
        splits.append((indices[mask], block.copy()))
    return splits


def independent_folds(n: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    dummy = np.empty(int(n))
    return [(train, test) for train, test in splitter.split(dummy)]


def vector_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    denominator = float(np.sum(truth**2))
    nrmse = math.sqrt(float(np.sum((estimate - truth) ** 2)) / max(denominator, 1e-15))
    slope = float(np.dot(truth, estimate) / max(denominator, 1e-15))
    threshold = 0.1 * math.sqrt(denominator / max(1, truth.size))
    active = np.abs(truth) > threshold
    sign_accuracy = float(np.mean(np.sign(estimate[active]) == np.sign(truth[active]))) if np.any(active) else math.nan
    return {"nrmse": nrmse, "calibration_slope": slope, "sign_accuracy": sign_accuracy}


def bootstrap_interval(values: np.ndarray, seed: int, replicates: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(replicates), values.size))
    draws = np.mean(values[indices], axis=1)
    return float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
