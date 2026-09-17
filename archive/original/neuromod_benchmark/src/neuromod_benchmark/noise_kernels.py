"""Exact corruption-score targets for denoising score matching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class NoiseKernel(Protocol):
    name: str

    def corrupt(
        self, clean: np.ndarray, scale: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]: ...

    def conditional_score(
        self, noisy: np.ndarray, clean: np.ndarray, scale: float, aux: dict
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class GaussianNoise:
    name: str = "gaussian"

    def corrupt(self, clean, scale, rng):
        if scale <= 0:
            raise ValueError("scale must be positive")
        noise = rng.standard_normal(np.asarray(clean).shape)
        return np.asarray(clean) + scale * noise, {"noise": noise}

    def conditional_score(self, noisy, clean, scale, aux):
        del aux
        return -(np.asarray(noisy) - np.asarray(clean)) / scale**2


@dataclass(frozen=True)
class StudentTNoise:
    """Isotropic multivariate Student-t corruption with an exact location score."""

    df: float = 5.0
    name: str = "student_t"

    def __post_init__(self):
        if self.df <= 2:
            raise ValueError("df must exceed two")

    def corrupt(self, clean, scale, rng):
        if scale <= 0:
            raise ValueError("scale must be positive")
        clean = np.asarray(clean, dtype=float)
        if clean.ndim == 1:
            clean = clean[:, None]
        gaussian = rng.standard_normal(clean.shape)
        chi_square = rng.chisquare(self.df, size=(clean.shape[0], 1))
        noise = gaussian / np.sqrt(chi_square / self.df)
        return clean + scale * noise, {"chi_square": chi_square}

    def conditional_score(self, noisy, clean, scale, aux):
        del aux
        delta = np.asarray(noisy, dtype=float) - np.asarray(clean, dtype=float)
        dimension = delta.shape[-1]
        squared_norm = np.sum(delta**2, axis=-1, keepdims=True)
        return -((self.df + dimension) * delta) / (self.df * scale**2 + squared_norm)


def finite_difference_score(
    kernel: NoiseKernel,
    noisy: np.ndarray,
    clean: np.ndarray,
    scale: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Numerical audit for the two included kernel log densities."""

    noisy = np.asarray(noisy, dtype=float)
    clean = np.asarray(clean, dtype=float)
    result = np.zeros_like(noisy)

    def log_density(value):
        delta = value - clean
        if isinstance(kernel, GaussianNoise):
            return -0.5 * np.sum(delta**2, axis=-1) / scale**2
        if isinstance(kernel, StudentTNoise):
            d = delta.shape[-1]
            return -0.5 * (kernel.df + d) * np.log1p(
                np.sum(delta**2, axis=-1) / (kernel.df * scale**2)
            )
        raise TypeError("finite-difference audit is not implemented for this kernel")

    for coordinate in range(noisy.shape[-1]):
        plus = noisy.copy()
        minus = noisy.copy()
        plus[..., coordinate] += epsilon
        minus[..., coordinate] -= epsilon
        result[..., coordinate] = (log_density(plus) - log_density(minus)) / (2 * epsilon)
    return result
