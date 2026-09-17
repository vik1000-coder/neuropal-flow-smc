from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import expit, logit


EPS = 1e-12


def trap_weights(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    weights = np.empty_like(grid)
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    return weights


def raw_to_functional(raw: np.ndarray, channel: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    m1, m2, m3, m4 = [raw[..., index] for index in range(4)]
    if channel == "mean":
        return m1
    if channel == "variance":
        return m2 - m1**2
    if channel == "covariance":
        return m2
    if channel == "third_cumulant":
        return m3 - 3.0 * m1 * m2 + 2.0 * m1**3
    if channel == "fourth_central":
        return m4 - 4.0 * m1 * m3 + 6.0 * m1**2 * m2 - 3.0 * m1**4
    raise KeyError(channel)


def raw_functional_derivative(raw: np.ndarray, draw: np.ndarray, channel: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    draw = np.asarray(draw, dtype=float)
    m1, m2, m3, _m4 = [raw[..., index] for index in range(4)]
    d1, d2, d3, d4 = [draw[..., index] for index in range(4)]
    if channel == "mean":
        return d1
    if channel == "variance":
        return d2 - 2.0 * m1 * d1
    if channel == "covariance":
        return d2
    if channel == "third_cumulant":
        return d3 - 3.0 * (d1 * m2 + m1 * d2) + 6.0 * m1**2 * d1
    if channel == "fourth_central":
        return (
            d4
            - 4.0 * (d1 * m3 + m1 * d3)
            + 12.0 * m1 * d1 * m2
            + 6.0 * m1**2 * d2
            - 12.0 * m1**3 * d1
        )
    raise KeyError(channel)


def influence_from_raw(y: np.ndarray, raw: np.ndarray, channel: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    raw = np.asarray(raw, dtype=float)
    m1, m2, m3, _m4 = [raw[..., index] for index in range(4)]
    variance = m2 - m1**2
    third = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
    fourth = raw_to_functional(raw, "fourth_central")
    centered = y - m1
    if channel == "mean":
        return centered
    if channel == "variance":
        return centered**2 - variance
    if channel == "third_cumulant":
        return centered**3 - 3.0 * variance * centered - third
    if channel == "fourth_central":
        return centered**4 - fourth - 4.0 * third * centered
    raise KeyError(channel)


@dataclass(frozen=True)
class TiltDefinition:
    mechanism: str
    channel: str
    nuisance_degree: int
    variant: str
    threshold: float | None = None


TILT_DEFINITIONS: dict[str, TiltDefinition] = {
    "m3_cubic": TiltDefinition("m3_cubic", "third_cumulant", 2, "cubic"),
    "m3_bounded": TiltDefinition("m3_bounded", "third_cumulant", 2, "bounded"),
    "m3_local": TiltDefinition("m3_local", "third_cumulant", 2, "local"),
    "m3_asymmetric": TiltDefinition("m3_asymmetric", "third_cumulant", 2, "asymmetric"),
    "m4_quartic": TiltDefinition("m4_quartic", "fourth_central", 3, "quartic"),
    "m4_tail": TiltDefinition("m4_tail", "tail_probability", 3, "tail", 1.5),
    "m5_occupancy": TiltDefinition("m5_occupancy", "mode_occupancy", 2, "occupancy", 0.75),
}


class SmoothTiltDGP:
    """A normalized, positive signed tilt embedded in one coordinate of R^8."""

    def __init__(
        self,
        mechanism: str,
        amplitude: float = 0.55,
        response_dim: int = 8,
        grid_size: int = 4001,
    ) -> None:
        if mechanism not in TILT_DEFINITIONS:
            raise KeyError(mechanism)
        if not 0.0 < amplitude < 0.9:
            raise ValueError("amplitude must lie in (0, 0.9)")
        self.definition = TILT_DEFINITIONS[mechanism]
        self.mechanism = mechanism
        self.channel = self.definition.channel
        self.amplitude = float(amplitude)
        self.response_dim = int(response_dim)
        self.grid = np.linspace(-5.0, 5.0, int(grid_size))
        self.weights = trap_weights(self.grid)
        base = np.exp(-0.5 * self.grid**2) / np.sqrt(2.0 * np.pi)
        base /= np.sum(base * self.weights)
        self.base_density = base
        self.psi = self._construct_psi()
        self.psi_prime = np.gradient(self.psi, self.grid, edge_order=2)
        powers = np.stack([self.grid**order for order in range(1, 5)], axis=1)
        self.raw_base = np.sum(self.weights[:, None] * base[:, None] * powers, axis=0)
        self.raw_psi = np.sum(
            self.weights[:, None] * base[:, None] * self.psi[:, None] * powers,
            axis=0,
        )
        feature = self._declared_feature(self.grid)
        self.feature_base = float(np.sum(self.weights * base * feature))
        self.feature_psi = float(np.sum(self.weights * base * self.psi * feature))
        self.orthogonality = {
            degree: float(np.sum(self.weights * base * self.psi * self.grid**degree))
            for degree in range(self.definition.nuisance_degree + 1)
        }

    def _raw_feature(self, y: np.ndarray) -> np.ndarray:
        variant = self.definition.variant
        if variant == "cubic":
            return (y**3 - 3.0 * y) * np.exp(-0.16 * y**2)
        if variant == "bounded":
            return np.sin(1.6 * y)
        if variant == "local":
            return np.exp(-((y - 1.25) / 0.45) ** 2) - np.exp(-((y + 1.25) / 0.45) ** 2)
        if variant == "asymmetric":
            return np.sin(1.6 * (y - 0.2))
        if variant == "quartic":
            return np.cos(3.0 * y)
        if variant == "tail":
            return expit((np.abs(y) - 1.7) / 0.12)
        if variant == "occupancy":
            return np.exp(-((y - 1.45) / 0.38) ** 2)
        raise KeyError(variant)

    def _declared_feature(self, y: np.ndarray) -> np.ndarray:
        if self.channel == "tail_probability":
            return (y > float(self.definition.threshold)).astype(float)
        if self.channel == "mode_occupancy":
            return (y > float(self.definition.threshold)).astype(float)
        return self._raw_feature(y)

    def _construct_psi(self) -> np.ndarray:
        raw = self._raw_feature(self.grid)
        nuisance = np.stack(
            [self.grid**degree for degree in range(self.definition.nuisance_degree + 1)], axis=1
        )
        weighted = nuisance * np.sqrt(self.base_density * self.weights)[:, None]
        target = raw * np.sqrt(self.base_density * self.weights)
        coefficients, *_ = np.linalg.lstsq(weighted, target, rcond=None)
        residual = raw - nuisance @ coefficients
        residual -= np.sum(self.weights * self.base_density * residual)
        scale = np.max(np.abs(residual))
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError("failed to construct a bounded signed tilt")
        return residual / scale

    def t(self, h: np.ndarray | float) -> np.ndarray:
        return self.amplitude * np.tanh(np.asarray(h, dtype=float))

    def dt(self, h: np.ndarray | float) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        return self.amplitude / np.cosh(h) ** 2

    def density_grid(self, h: float) -> np.ndarray:
        density = self.base_density * (1.0 + float(self.t(h)) * self.psi)
        density = np.maximum(density, 0.0)
        density /= np.sum(density * self.weights)
        return density

    def sample_projection(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape(-1)
        result = np.empty_like(h)
        remaining = np.arange(h.size)
        while remaining.size:
            candidate = rng.normal(size=remaining.size)
            inside = np.abs(candidate) <= 5.0
            psi = np.interp(np.clip(candidate, -5.0, 5.0), self.grid, self.psi)
            t = self.t(h[remaining])
            probability = np.where(inside, (1.0 + t * psi) / (1.0 + np.abs(t)), 0.0)
            accepted = rng.random(remaining.size) < probability
            result[remaining[accepted]] = candidate[accepted]
            remaining = remaining[~accepted]
        return result

    def sample(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape(-1)
        y = rng.normal(size=(h.size, self.response_dim))
        y[:, 0] = self.sample_projection(h, rng)
        return y

    def psi_at(self, y: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(y, dtype=float), self.grid, self.psi)

    def psi_prime_at(self, y: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(y, dtype=float), self.grid, self.psi_prime)

    def score_projection(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        h = np.asarray(h, dtype=float)
        t = self.t(h)
        psi = self.psi_at(y)
        return -y + t * self.psi_prime_at(y) / np.maximum(1.0 + t * psi, EPS)

    def history_tangent(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        h = np.asarray(h, dtype=float)
        t = self.t(h)
        return self.dt(h) * self.psi_at(y) / np.maximum(1.0 + t * self.psi_at(y), EPS)

    def mixed_field(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        h = np.asarray(h, dtype=float)
        denominator = np.maximum(1.0 + self.t(h) * self.psi_at(y), EPS)
        return self.dt(h) * self.psi_prime_at(y) / denominator**2

    def raw_moments(self, h: np.ndarray | float) -> np.ndarray:
        t = np.asarray(self.t(h), dtype=float)
        return self.raw_base + t[..., None] * self.raw_psi

    def raw_moment_derivative(self, h: np.ndarray | float) -> np.ndarray:
        return np.asarray(self.dt(h))[..., None] * self.raw_psi

    def target(self, h: np.ndarray | float, channel: str | None = None) -> np.ndarray:
        channel = channel or self.channel
        if channel in {"tail_probability", "mode_occupancy"}:
            return self.feature_base + self.t(h) * self.feature_psi
        return raw_to_functional(self.raw_moments(h), channel)

    def local_effect(self, h: np.ndarray | float, channel: str | None = None) -> np.ndarray:
        channel = channel or self.channel
        if channel in {"tail_probability", "mode_occupancy"}:
            return self.dt(h) * self.feature_psi
        return raw_functional_derivative(
            self.raw_moments(h), self.raw_moment_derivative(h), channel
        )

    def finite_effect(
        self, h: np.ndarray | float, delta: float, channel: str | None = None
    ) -> np.ndarray:
        channel = channel or self.channel
        h = np.asarray(h, dtype=float)
        return (self.target(h + delta, channel) - self.target(h - delta, channel)) / (2.0 * delta)

    def influence(self, y: np.ndarray, h: np.ndarray, channel: str | None = None) -> np.ndarray:
        channel = channel or self.channel
        y = np.asarray(y, dtype=float)
        h = np.asarray(h, dtype=float)
        if channel in {"tail_probability", "mode_occupancy"}:
            return self._declared_feature(y) - self.target(h, channel)
        return influence_from_raw(y, self.raw_moments(h), channel)

    def information(self, h: float = 0.0) -> float:
        ell = self.history_tangent(self.grid, np.full_like(self.grid, h))
        return float(np.sum(self.weights * self.density_grid(h) * ell**2))

    def noised_density_score_tangent(
        self, z: np.ndarray, h: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=float).reshape(-1)
        h = np.asarray(h, dtype=float).reshape(-1)
        if z.size != h.size:
            raise ValueError("z and h must have the same size")
        if sigma <= 0:
            return self.score_projection(z, h), self.history_tangent(z, h)
        score = np.empty_like(z)
        tangent = np.empty_like(z)
        normalizer = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        for start in range(0, z.size, 256):
            stop = min(z.size, start + 256)
            zz = z[start:stop, None]
            hh = h[start:stop]
            base = self.base_density[None, :] * (
                1.0 + self.t(hh)[:, None] * self.psi[None, :]
            )
            kernel = normalizer * np.exp(-0.5 * ((zz - self.grid[None, :]) / sigma) ** 2)
            mass = np.sum(kernel * base * self.weights[None, :], axis=1)
            first = np.sum(kernel * base * self.grid[None, :] * self.weights[None, :], axis=1)
            score[start:stop] = (first / np.maximum(mass, EPS) - z[start:stop]) / sigma**2
            numerator = np.sum(
                kernel
                * self.base_density[None, :]
                * self.dt(hh)[:, None]
                * self.psi[None, :]
                * self.weights[None, :],
                axis=1,
            )
            tangent[start:stop] = numerator / np.maximum(mass, EPS)
        return score, tangent


class GaussianMeanDGP:
    mechanism = "m1_mean"
    channel = "mean"

    def __init__(self, amplitude: float = 0.55, response_dim: int = 8) -> None:
        self.amplitude = float(amplitude)
        self.response_dim = int(response_dim)

    def mean(self, h: np.ndarray | float) -> np.ndarray:
        return self.amplitude * np.tanh(np.asarray(h, dtype=float))

    def dmean(self, h: np.ndarray | float) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        return self.amplitude / np.cosh(h) ** 2

    def sample(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape(-1)
        y = rng.normal(size=(h.size, self.response_dim))
        y[:, 0] += self.mean(h)
        return y

    def raw_moments(self, h: np.ndarray | float) -> np.ndarray:
        mu = self.mean(h)
        return np.stack([mu, 1.0 + mu**2, mu**3 + 3.0 * mu, mu**4 + 6.0 * mu**2 + 3.0], axis=-1)

    def target(self, h: np.ndarray | float, channel: str = "mean") -> np.ndarray:
        return raw_to_functional(self.raw_moments(h), channel)

    def local_effect(self, h: np.ndarray | float, channel: str = "mean") -> np.ndarray:
        h = np.asarray(h, dtype=float)
        if channel == "mean":
            return self.dmean(h)
        return np.zeros_like(h)

    def finite_effect(self, h: np.ndarray | float, delta: float, channel: str = "mean") -> np.ndarray:
        h = np.asarray(h, dtype=float)
        return (self.target(h + delta, channel) - self.target(h - delta, channel)) / (2.0 * delta)

    def score_projection(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        return -(np.asarray(y) - self.mean(h))

    def history_tangent(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        return (np.asarray(y) - self.mean(h)) * self.dmean(h)

    def mixed_field(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.dmean(h), np.asarray(y).shape)

    def influence(self, y: np.ndarray, h: np.ndarray, channel: str = "mean") -> np.ndarray:
        return influence_from_raw(np.asarray(y), self.raw_moments(h), channel)

    def information(self, h: float = 0.0) -> float:
        return float(self.dmean(h) ** 2)

    def noised_density_score_tangent(
        self, z: np.ndarray, h: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        variance = 1.0 + sigma**2
        score = -(np.asarray(z) - self.mean(h)) / variance
        tangent = (np.asarray(z) - self.mean(h)) * self.dmean(h) / variance
        return score, tangent


class GaussianCovarianceDGP:
    mechanism = "m2_covariance"
    channel = "covariance"

    def __init__(self, amplitude: float = 0.45, response_dim: int = 8) -> None:
        self.amplitude = float(amplitude)
        self.response_dim = int(response_dim)

    def rho(self, h: np.ndarray | float) -> np.ndarray:
        return self.amplitude * np.tanh(np.asarray(h, dtype=float))

    def drho(self, h: np.ndarray | float) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        return self.amplitude / np.cosh(h) ** 2

    def sample(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape(-1)
        y = rng.normal(size=(h.size, self.response_dim))
        rho = self.rho(h)
        y[:, 1] = rho * y[:, 0] + np.sqrt(np.maximum(1.0 - rho**2, EPS)) * y[:, 1]
        return y

    def target(self, h: np.ndarray | float, channel: str = "covariance") -> np.ndarray:
        h = np.asarray(h, dtype=float)
        if channel == "covariance":
            return self.rho(h)
        return np.zeros_like(h)

    def local_effect(self, h: np.ndarray | float, channel: str = "covariance") -> np.ndarray:
        h = np.asarray(h, dtype=float)
        if channel == "covariance":
            return self.drho(h)
        return np.zeros_like(h)

    def finite_effect(self, h: np.ndarray | float, delta: float, channel: str = "covariance") -> np.ndarray:
        h = np.asarray(h, dtype=float)
        return (self.target(h + delta, channel) - self.target(h - delta, channel)) / (2.0 * delta)

    def history_tangent(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        h = np.asarray(h, dtype=float).reshape(-1)
        rho = self.rho(h)
        drho = self.drho(h)
        y0, y1 = y[:, 0], y[:, 1]
        denominator = np.maximum(1.0 - rho**2, EPS)
        derivative_rho = (
            rho / denominator
            + y0 * y1 / denominator
            - rho * (y0**2 - 2.0 * rho * y0 * y1 + y1**2) / denominator**2
        )
        return drho * derivative_rho

    def influence(self, y: np.ndarray, h: np.ndarray, channel: str = "covariance") -> np.ndarray:
        if channel != "covariance":
            return np.zeros(np.asarray(y).shape[0])
        y = np.asarray(y, dtype=float)
        return y[:, 0] * y[:, 1] - self.rho(h)

    def information(self, h: float = 0.0) -> float:
        rho = float(self.rho(h))
        return float(self.drho(h) ** 2 * (1.0 + rho**2) / np.maximum(1.0 - rho**2, EPS) ** 2)


class DynamicGaussianMeanDGP:
    mechanism = "d1_gaussian_lag"
    channel = "mean"

    def __init__(self, lags: int = 6, rho: float = 0.8, noise_sd: float = 0.7) -> None:
        self.lags = int(lags)
        kernel = np.array([rho**index for index in range(self.lags)], dtype=float)
        self.kernel = 0.65 * kernel / np.linalg.norm(kernel)
        self.noise_sd = float(noise_sd)

    def sample_histories(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(int(n), self.lags))

    def mean(self, h: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(h) @ self.kernel)

    def lag_effect(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h)
        scalar = 1.0 / np.cosh(h @ self.kernel) ** 2
        return scalar[:, None] * self.kernel[None, :]

    def sample(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.mean(h) + self.noise_sd * rng.normal(size=np.asarray(h).shape[0])


class DynamicStochasticGainDGP:
    mechanism = "d2_observed_stochastic_gain"

    def __init__(
        self,
        lags: int = 6,
        rho: float = 0.8,
        primary_pi: float = 0.30,
        gain: float = 2.0,
        noise_sd: float = 0.55,
    ) -> None:
        self.lags = int(lags)
        kernel = np.array([rho**index for index in range(self.lags)], dtype=float)
        self.kernel = 0.75 * kernel / np.linalg.norm(kernel)
        self.alpha = float(logit(primary_pi))
        self.gain = float(gain)
        self.gain_vector = np.array([self.gain, -0.75 * self.gain], dtype=float)
        self.noise_sd = float(noise_sd)

    def sample_histories(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(scale=0.7, size=(int(n), self.lags))

    def pi(self, h: np.ndarray) -> np.ndarray:
        return expit(self.alpha + np.asarray(h) @ self.kernel)

    def dpi(self, h: np.ndarray) -> np.ndarray:
        probability = self.pi(h)
        return probability[:, None] * (1.0 - probability[:, None]) * self.kernel[None, :]

    def sample(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        probability = self.pi(h)
        state = rng.binomial(1, probability)
        return self.gain * (state - probability) + self.noise_sd * rng.normal(size=probability.size)

    def sample_vector(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        probability = self.pi(h)
        state = rng.binomial(1, probability)
        centered = state - probability
        signal = centered[:, None] * self.gain_vector[None, :]
        return signal + self.noise_sd * rng.normal(size=signal.shape)

    def raw_moments(self, h: np.ndarray) -> np.ndarray:
        probability = self.pi(h)
        variance = self.noise_sd**2 + self.gain**2 * probability * (1.0 - probability)
        third = self.gain**3 * probability * (1.0 - probability) * (1.0 - 2.0 * probability)
        fourth_signal = self.gain**4 * probability * (1.0 - probability) * (
            (1.0 - probability) ** 3 + probability**3
        )
        fourth = fourth_signal + 6.0 * self.noise_sd**2 * (
            self.gain**2 * probability * (1.0 - probability)
        ) + 3.0 * self.noise_sd**4
        zeros = np.zeros_like(probability)
        return np.stack([zeros, variance, third, fourth], axis=-1)

    def lag_effect(self, h: np.ndarray, channel: str) -> np.ndarray:
        probability = self.pi(h)
        derivative = self.dpi(h)
        if channel == "mean":
            return np.zeros_like(derivative)
        if channel in {"variance", "covariance"}:
            gain_product = (
                self.gain**2
                if channel == "variance"
                else self.gain_vector[0] * self.gain_vector[1]
            )
            return gain_product * (1.0 - 2.0 * probability[:, None]) * derivative
        if channel == "third_cumulant":
            factor = 1.0 - 6.0 * probability + 6.0 * probability**2
            return self.gain**3 * factor[:, None] * derivative
        raise KeyError(channel)

    def finite_effect(self, h: np.ndarray, lag: int, delta: float, channel: str) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        plus = h.copy()
        minus = h.copy()
        plus[:, lag] += delta
        minus[:, lag] -= delta
        if channel == "mean":
            return np.zeros(h.shape[0])
        if channel == "covariance":
            probability_plus = self.pi(plus)
            probability_minus = self.pi(minus)
            target_plus = (
                self.gain_vector[0]
                * self.gain_vector[1]
                * probability_plus
                * (1.0 - probability_plus)
            )
            target_minus = (
                self.gain_vector[0]
                * self.gain_vector[1]
                * probability_minus
                * (1.0 - probability_minus)
            )
            return (target_plus - target_minus) / (2.0 * delta)
        target_plus = raw_to_functional(self.raw_moments(plus), channel)
        target_minus = raw_to_functional(self.raw_moments(minus), channel)
        return (target_plus - target_minus) / (2.0 * delta)

    def score(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        probability = self.pi(h)
        mean1 = self.gain * (1.0 - probability)
        mean0 = -self.gain * probability
        log1 = np.log(np.maximum(probability, EPS)) - 0.5 * ((y - mean1) / self.noise_sd) ** 2
        log0 = np.log(np.maximum(1.0 - probability, EPS)) - 0.5 * ((y - mean0) / self.noise_sd) ** 2
        maximum = np.maximum(log1, log0)
        weight1 = np.exp(log1 - maximum) / (np.exp(log1 - maximum) + np.exp(log0 - maximum))
        return -(
            weight1 * (y - mean1) + (1.0 - weight1) * (y - mean0)
        ) / self.noise_sd**2


def make_static_dgp(mechanism: str, amplitude: float | None = None):
    if mechanism == "m1_mean":
        return GaussianMeanDGP(amplitude=0.55 if amplitude is None else amplitude)
    if mechanism == "m2_covariance":
        return GaussianCovarianceDGP(amplitude=0.45 if amplitude is None else amplitude)
    return SmoothTiltDGP(mechanism, amplitude=0.55 if amplitude is None else amplitude)


def all_static_mechanisms() -> Iterable[str]:
    yield "m1_mean"
    yield "m2_covariance"
    yield from TILT_DEFINITIONS
