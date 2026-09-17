"""Exact continuous synthetic conditional laws and history-tangent oracles.

The oracle layer is intentionally independent of the trainable benchmark models.  All
parameters, samples, densities, and derivatives live on CPU in ``torch.float64``.  The
seven legacy generators implement G1--G7 from the preregistered benchmark
specification; G8 adds an audited whole-path, pure higher-shape stress test.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any

import torch


DTYPE = torch.float64
DEVICE = torch.device("cpu")
LOG_2PI = math.log(2.0 * math.pi)


def _tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=DTYPE, device=DEVICE)


def _generator(seed: int) -> torch.Generator:
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _unit_rows(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _unit_columns(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=0, keepdim=True).clamp_min(1e-12)


def _orthogonal(rows: int, columns: int, rng: torch.Generator) -> torch.Tensor:
    if rows < columns:
        raise ValueError("orthogonal matrix needs rows >= columns")
    raw = torch.randn((rows, columns), generator=rng, dtype=DTYPE)
    q, r = torch.linalg.qr(raw, mode="reduced")
    signs = torch.where(torch.diagonal(r) < 0, -torch.ones((), dtype=DTYPE), 1.0)
    return q * signs


def _location_parameters(
    q: int,
    dy: int,
    hidden: int,
    rng: torch.Generator,
    target_signal_sd: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    w1 = _unit_rows(torch.randn((hidden, q), generator=rng, dtype=DTYPE))
    w2 = torch.randn((dy, hidden), generator=rng, dtype=DTYPE) / math.sqrt(hidden)
    calibration_h = torch.randn((4096, q), generator=rng, dtype=DTYPE)
    raw_mean = torch.tanh(calibration_h @ w1.T) @ w2.T
    median_sd = raw_mean.std(dim=0, unbiased=False).median().clamp_min(1e-8)
    w2 = w2 * (float(target_signal_sd) / median_sd)
    return w1, w2


def _sigma_vector(sigma: float | torch.Tensor, dy: int) -> torch.Tensor:
    value = _tensor(sigma)
    if value.ndim == 0:
        value = value.expand(dy)
    elif value.ndim != 1 or value.numel() != dy:
        raise ValueError(f"sigma must be scalar or length dy={dy}")
    if not bool(torch.all(torch.isfinite(value))) or bool(torch.any(value < 0)):
        raise ValueError("sigma must be finite and non-negative")
    return value


def _broadcast_y_h(
    y: torch.Tensor | Any,
    h: torch.Tensor | Any,
    dy: int,
    q: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = _tensor(y)
    h = _tensor(h)
    if y.ndim < 1 or y.shape[-1] != dy:
        raise ValueError(f"y must end in dy={dy}, got {tuple(y.shape)}")
    if h.ndim < 1 or h.shape[-1] != q:
        raise ValueError(f"h must end in q={q}, got {tuple(h.shape)}")
    y_leading = y.shape[:-1]
    h_leading = h.shape[:-1]
    if len(h_leading) < len(y_leading):
        h = h.reshape(h_leading + (1,) * (len(y_leading) - len(h_leading)) + (q,))
    elif len(y_leading) < len(h_leading):
        y = y.reshape(y_leading + (1,) * (len(h_leading) - len(y_leading)) + (dy,))
    leading = torch.broadcast_shapes(y.shape[:-1], h.shape[:-1])
    return y.expand(leading + (dy,)), h.expand(leading + (q,))


def _mvn_log_prob(y: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    dy = y.shape[-1]
    difference = (y - mean).unsqueeze(-1)
    chol = torch.linalg.cholesky(covariance)
    leading = torch.broadcast_shapes(difference.shape[:-2], chol.shape[:-2])
    difference = difference.expand(leading + (dy, 1))
    chol = chol.expand(leading + (dy, dy))
    whitened = torch.linalg.solve_triangular(chol, difference, upper=False).squeeze(-1)
    quadratic = whitened.square().sum(dim=-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    return -0.5 * (dy * LOG_2PI + logdet + quadratic)


def _diagonal_log_prob(
    y: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor
) -> torch.Tensor:
    variance = variance.clamp_min(1e-14)
    return (-0.5 * (LOG_2PI + torch.log(variance) + (y - mean).square() / variance)).sum(
        dim=-1
    )


def _mixture_diagonal_log_prob(
    y: torch.Tensor,
    log_weights: torch.Tensor,
    means: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    component = -0.5 * (
        LOG_2PI
        + torch.log(variance)
        + (y.unsqueeze(-2) - means).square() / variance
    ).sum(dim=-1)
    return torch.logsumexp(log_weights + component, dim=-1)


def _mixture_full_log_prob(
    y: torch.Tensor,
    log_weights: torch.Tensor,
    means: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    components = torch.stack(
        [_mvn_log_prob(y, means[index], covariance) for index in range(means.shape[0])],
        dim=-1,
    )
    return torch.logsumexp(log_weights + components, dim=-1)


class ConditionalDGP(ABC):
    """Common exact-oracle contract for a continuous conditional law."""

    def __init__(self, name: str, q: int, dy: int, seed: int = 0):
        if q < 1 or dy < 1:
            raise ValueError("q and dy must be positive")
        self.name = str(name)
        self.q = int(q)
        self.dy = int(dy)
        self.seed = int(seed)

    def sample_history(self, n: int, seed: int) -> torch.Tensor:
        if n < 1:
            raise ValueError("n must be positive")
        return torch.randn((n, self.q), generator=_generator(seed), dtype=DTYPE)

    @abstractmethod
    def sample_response(self, h: torch.Tensor, n_per_h: int, seed: int) -> torch.Tensor:
        """Return conditional draws with shape ``[n_history, n_per_h, dy]``."""

    @abstractmethod
    def _log_prob_aligned(
        self, y: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate aligned arrays; ``sigma`` is a length-``dy`` noise standard deviation."""

    def log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y_aligned, h_aligned = _broadcast_y_h(y, h, self.dy, self.q)
        return self._log_prob_aligned(y_aligned, h_aligned, torch.zeros(self.dy, dtype=DTYPE))

    def noisy_log_prob(
        self, y: torch.Tensor, h: torch.Tensor, sigma: float | torch.Tensor
    ) -> torch.Tensor:
        """Log density after independent response noise with coordinate-wise ``sigma``."""

        y_aligned, h_aligned = _broadcast_y_h(y, h, self.dy, self.q)
        return self._log_prob_aligned(y_aligned, h_aligned, _sigma_vector(sigma, self.dy))

    def history_tangent(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y_aligned, h_aligned = _broadcast_y_h(y, h, self.dy, self.q)
        history = h_aligned.detach().clone().requires_grad_(True)
        value = self._log_prob_aligned(
            y_aligned.detach(), history, torch.zeros(self.dy, dtype=DTYPE)
        )
        return torch.autograd.grad(value.sum(), history)[0].detach()

    def noisy_history_tangent(
        self, y: torch.Tensor, h: torch.Tensor, sigma: float | torch.Tensor
    ) -> torch.Tensor:
        y_aligned, h_aligned = _broadcast_y_h(y, h, self.dy, self.q)
        history = h_aligned.detach().clone().requires_grad_(True)
        value = self._log_prob_aligned(
            y_aligned.detach(), history, _sigma_vector(sigma, self.dy)
        )
        return torch.autograd.grad(value.sum(), history)[0].detach()

    def response_score(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y_aligned, h_aligned = _broadcast_y_h(y, h, self.dy, self.q)
        response = y_aligned.detach().clone().requires_grad_(True)
        value = self._log_prob_aligned(
            response, h_aligned.detach(), torch.zeros(self.dy, dtype=DTYPE)
        )
        return torch.autograd.grad(value.sum(), response)[0].detach()

    def noisy_response_score(
        self, y: torch.Tensor, h: torch.Tensor, sigma: float | torch.Tensor
    ) -> torch.Tensor:
        y_aligned, h_aligned = _broadcast_y_h(y, h, self.dy, self.q)
        response = y_aligned.detach().clone().requires_grad_(True)
        value = self._log_prob_aligned(
            response, h_aligned.detach(), _sigma_vector(sigma, self.dy)
        )
        return torch.autograd.grad(value.sum(), response)[0].detach()

    def finite_log_ratio(
        self,
        y: torch.Tensor,
        h: torch.Tensor,
        v: torch.Tensor,
        delta: float,
    ) -> torch.Tensor:
        direction = _tensor(v)
        if direction.shape[-1] != self.q:
            raise ValueError(f"v must end in q={self.q}")
        return self.log_prob(y, _tensor(h) + float(delta) * direction) - self.log_prob(y, h)

    def mechanism_directions(self) -> dict[str, torch.Tensor]:
        return {}

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "q": self.q, "dy": self.dy, "generator_seed": self.seed}


class G1LocationGaussian(ConditionalDGP):
    """G1: nonlinear location-only multivariate Gaussian."""

    def __init__(self, seed: int = 0, q: int = 8, dy: int = 4, hidden: int = 16):
        super().__init__("g1_location", q, dy, seed)
        rng = _generator(seed)
        self.hidden = int(hidden)
        self.w1, self.w2 = _location_parameters(q, dy, hidden, rng)
        self.rotation = _orthogonal(dy, dy, rng)
        self.base_sd = torch.linspace(0.5, 1.0, dy, dtype=DTYPE)
        self.covariance = (
            self.rotation @ torch.diag(self.base_sd.square()) @ self.rotation.T
        )

    def mean(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(h @ self.w1.T) @ self.w2.T

    def sample_response(self, h: torch.Tensor, n_per_h: int, seed: int) -> torch.Tensor:
        h = _tensor(h)
        if h.ndim != 2 or h.shape[1] != self.q or n_per_h < 1:
            raise ValueError("h must be [n,q] and n_per_h must be positive")
        noise = torch.randn(
            (len(h), n_per_h, self.dy), generator=_generator(seed), dtype=DTYPE
        )
        transformed = (noise * self.base_sd) @ self.rotation.T
        return self.mean(h)[:, None, :] + transformed

    def _log_prob_aligned(
        self, y: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        covariance = self.covariance + torch.diag(sigma.square())
        return _mvn_log_prob(y, self.mean(h), covariance)

    def mechanism_directions(self) -> dict[str, torch.Tensor]:
        return {f"mean_hidden_{i}": self.w1[i].clone() for i in range(self.hidden)}

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "hidden_width": self.hidden,
            "baseline_sd": self.base_sd.tolist(),
        }


class G2CovarianceGaussian(ConditionalDGP):
    """G2: Gaussian with history-dependent covariance and optional mean."""

    def __init__(
        self,
        seed: int = 0,
        q: int = 8,
        dy: int = 4,
        hidden: int = 16,
        alpha_m: float = 0.0,
    ):
        super().__init__("g2_covariance", q, dy, seed)
        rng = _generator(seed)
        self.hidden = int(hidden)
        self.alpha_m = float(alpha_m)
        if self.alpha_m not in {0.0, 0.3}:
            raise ValueError("alpha_m must be 0 or 0.3")
        self.w1, self.w2 = _location_parameters(q, dy, hidden, rng)
        self.a = _unit_rows(torch.randn((dy, q), generator=rng, dtype=DTYPE))
        baseline_sd = torch.linspace(0.5, 1.0, dy, dtype=DTYPE)
        self.b = torch.log(baseline_sd.square())
        self.rotation = _orthogonal(dy, dy, rng)

    def mean(self, h: torch.Tensor) -> torch.Tensor:
        return self.alpha_m * (torch.tanh(h @ self.w1.T) @ self.w2.T)

    def eigen_variance(self, h: torch.Tensor) -> torch.Tensor:
        routed = (h @ self.a.T) / math.sqrt(self.q)
        return torch.exp(self.b + 0.6 * torch.tanh(routed))

    def covariance_at(self, h: torch.Tensor) -> torch.Tensor:
        variance = self.eigen_variance(h)
        return torch.einsum("ij,...j,kj->...ik", self.rotation, variance, self.rotation)

    def sample_response(self, h: torch.Tensor, n_per_h: int, seed: int) -> torch.Tensor:
        h = _tensor(h)
        if h.ndim != 2 or h.shape[1] != self.q or n_per_h < 1:
            raise ValueError("h must be [n,q] and n_per_h must be positive")
        noise = torch.randn(
            (len(h), n_per_h, self.dy), generator=_generator(seed), dtype=DTYPE
        )
        eigen_noise = noise * self.eigen_variance(h).sqrt()[:, None, :]
        return self.mean(h)[:, None, :] + eigen_noise @ self.rotation.T

    def _log_prob_aligned(
        self, y: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        covariance = self.covariance_at(h) + torch.diag(sigma.square())
        return _mvn_log_prob(y, self.mean(h), covariance)

    def mechanism_directions(self) -> dict[str, torch.Tensor]:
        directions = {f"covariance_{i}": self.a[i].clone() for i in range(self.dy)}
        if self.alpha_m:
            directions.update(
                {f"mean_hidden_{i}": self.w1[i].clone() for i in range(self.hidden)}
            )
        return directions

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "alpha_m": self.alpha_m, "hidden_width": self.hidden}


class G3SkewMixture(ConditionalDGP):
    """G3: mixture-weight skewness change with exactly fixed mean and variance."""

    def __init__(self, seed: int = 0, q: int = 8, dy: int = 1):
        super().__init__("g3_skew", q, dy, seed)
        rng = _generator(seed)
        self.direction = _unit_rows(torch.randn((1, q), generator=rng, dtype=DTYPE))[0]
        self.component_means = torch.tensor([-3.0, -1.0, 1.0, 2.0], dtype=DTYPE)
        self.component_sd = 0.25
        self.base_weights = torch.full((4,), 0.25, dtype=DTYPE)
        self.weight_direction = torch.tensor(
            [-0.2, 2.0 / 3.0, -1.0, 8.0 / 15.0], dtype=DTYPE
        )

    def modulation(self, h: torch.Tensor) -> torch.Tensor:
        return 0.12 * torch.tanh(h @ self.direction)

    def weights(self, h: torch.Tensor) -> torch.Tensor:
        value = self.base_weights + self.modulation(h).unsqueeze(-1) * self.weight_direction
        if bool(torch.any(value <= 0)):
            raise RuntimeError("G3 mixture weights left their positive support")
        return value

    def sample_response(self, h: torch.Tensor, n_per_h: int, seed: int) -> torch.Tensor:
        h = _tensor(h)
        if h.ndim != 2 or h.shape[1] != self.q or n_per_h < 1:
            raise ValueError("h must be [n,q] and n_per_h must be positive")
        rng = _generator(seed)
        component = torch.multinomial(
            self.weights(h), n_per_h, replacement=True, generator=rng
        )
        informative = self.component_means[component] + self.component_sd * torch.randn(
            (len(h), n_per_h), generator=rng, dtype=DTYPE
        )
        response = torch.empty((len(h), n_per_h, self.dy), dtype=DTYPE)
        response[..., 0] = informative
        if self.dy > 1:
            response[..., 1:] = torch.randn(
                (len(h), n_per_h, self.dy - 1), generator=rng, dtype=DTYPE
            )
        return response

    def _log_prob_aligned(
        self, y: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        weights = self.weights(h)
        variance0 = self.component_sd**2 + sigma[0].square()
        component = -0.5 * (
            LOG_2PI
            + torch.log(variance0)
            + (y[..., 0].unsqueeze(-1) - self.component_means).square() / variance0
        )
        value = torch.logsumexp(torch.log(weights) + component, dim=-1)
        if self.dy > 1:
            nuisance_variance = 1.0 + sigma[1:].square()
            value = value + _diagonal_log_prob(
                y[..., 1:], torch.zeros_like(y[..., 1:]), nuisance_variance
            )
        return value

    def mechanism_directions(self) -> dict[str, torch.Tensor]:
        return {"skewness": self.direction.clone()}

    def metadata(self) -> dict[str, Any]:
        fixed_mean = float(self.base_weights @ self.component_means)
        fixed_second = float(
            self.base_weights @ (self.component_means.square() + self.component_sd**2)
        )
        return {
            **super().metadata(),
            "fixed_informative_mean": fixed_mean,
            "fixed_informative_variance": fixed_second - fixed_mean**2,
            "component_sd": self.component_sd,
        }


class G4LowRankMixture(ConditionalDGP):
    """G4: low-rank history modulation of a complex isotropic Gaussian mixture."""

    def __init__(
        self,
        seed: int = 0,
        q: int = 8,
        dy: int = 8,
        n_components: int = 4,
        rank: int = 2,
        mode_separation: float = 4.0,
        component_sigma: float = 1.0,
        interaction_scale: float = 1.0,
    ):
        super().__init__("g4_low_rank_mixture", q, dy, seed)
        if n_components < 2 or rank < 1 or rank > min(n_components, q):
            raise ValueError("invalid mixture component count or interaction rank")
        if mode_separation <= 0 or component_sigma <= 0 or interaction_scale <= 0:
            raise ValueError("separation, component sigma, and interaction scale must be positive")
        rng = _generator(seed)
        self.n_components = int(n_components)
        self.rank = int(rank)
        self.component_sigma = float(component_sigma)
        self.mode_separation = float(mode_separation)
        self.u = torch.randn((n_components, rank), generator=rng, dtype=DTYPE)
        self.v = torch.randn((q, rank), generator=rng, dtype=DTYPE)
        self.b_matrix = self.u @ self.v.T
        self.b_matrix = self.b_matrix - self.b_matrix.mean(dim=0, keepdim=True)
        # The supplied plan omitted the magnitude of B.  Freeze it explicitly:
        # each component logit has RMS standard deviation `interaction_scale`
        # under H~N(0,I), before the shared softmax competition.
        row_rms = self.b_matrix.square().sum(dim=1).mean().sqrt().clamp_min(1e-12)
        self.interaction_scale = float(interaction_scale)
        self.b_matrix = self.b_matrix * (self.interaction_scale / row_rms)
        self.logit_bias = torch.zeros(n_components, dtype=DTYPE)
        directions = _unit_rows(
            torch.randn((n_components, dy), generator=rng, dtype=DTYPE)
        )
        radius = self.mode_separation * self.component_sigma
        self.component_means = radius * directions

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.b_matrix.T + self.logit_bias

    def weights(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(h), dim=-1)

    def sample_response(self, h: torch.Tensor, n_per_h: int, seed: int) -> torch.Tensor:
        h = _tensor(h)
        if h.ndim != 2 or h.shape[1] != self.q or n_per_h < 1:
            raise ValueError("h must be [n,q] and n_per_h must be positive")
        rng = _generator(seed)
        component = torch.multinomial(
            self.weights(h), n_per_h, replacement=True, generator=rng
        )
        means = self.component_means[component]
        noise = torch.randn(means.shape, generator=rng, dtype=DTYPE)
        return means + self.component_sigma * noise

    def _log_prob_aligned(
        self, y: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        variance = self.component_sigma**2 + sigma.square()
        return _mixture_diagonal_log_prob(
            y,
            torch.log_softmax(self.logits(h), dim=-1),
            self.component_means,
            variance,
        )

    def mechanism_directions(self) -> dict[str, torch.Tensor]:
        normalized = _unit_columns(self.v)
        return {f"interaction_rank_{i}": normalized[:, i].clone() for i in range(self.rank)}

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "n_components": self.n_components,
            "interaction_rank": self.rank,
            "interaction_scale": self.interaction_scale,
            "mode_separation": self.mode_separation,
            "component_sigma": self.component_sigma,
        }


class G5NearManifoldMixture(ConditionalDGP):
    """G5: high-dimensional Gaussian mixture concentrated near a linear manifold."""

    def __init__(
        self,
        seed: int = 0,
        q: int = 8,
        dy: int = 32,
        latent_dim: int = 2,
        n_components: int = 8,
        rank: int = 2,
        sigma_z: float = 0.5,
        sigma_obs: float = 0.1,
        mode_separation: float = 4.0,
    ):
        super().__init__("g5_near_manifold", q, dy, seed)
        if not 1 <= latent_dim <= dy:
            raise ValueError("latent_dim must lie in [1,dy]")
        if rank < 1 or rank > min(n_components, q) or sigma_z <= 0 or sigma_obs <= 0:
            raise ValueError("invalid rank or noise scale")
        rng = _generator(seed)
        self.latent_dim = int(latent_dim)
        self.n_components = int(n_components)
        self.rank = int(rank)
        self.sigma_z = float(sigma_z)
        self.sigma_obs = float(sigma_obs)
        self.mode_separation = float(mode_separation)
        self.u = torch.randn((n_components, rank), generator=rng, dtype=DTYPE)
        self.v = torch.randn((q, rank), generator=rng, dtype=DTYPE)
        self.b_matrix = self.u @ self.v.T
        self.b_matrix -= self.b_matrix.mean(dim=0, keepdim=True)
        self.logit_bias = torch.zeros(n_components, dtype=DTYPE)
        self.embedding = _orthogonal(dy, latent_dim, rng)
        latent_direction = _unit_rows(
            torch.randn((n_components, latent_dim), generator=rng, dtype=DTYPE)
        )
        self.latent_means = mode_separation * sigma_z * latent_direction
        self.component_means = self.latent_means @ self.embedding.T
        self.base_covariance = (
            sigma_z**2 * (self.embedding @ self.embedding.T)
            + sigma_obs**2 * torch.eye(dy, dtype=DTYPE)
        )

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.b_matrix.T + self.logit_bias

    def weights(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(h), dim=-1)

    def sample_response(self, h: torch.Tensor, n_per_h: int, seed: int) -> torch.Tensor:
        h = _tensor(h)
        if h.ndim != 2 or h.shape[1] != self.q or n_per_h < 1:
            raise ValueError("h must be [n,q] and n_per_h must be positive")
        rng = _generator(seed)
        component = torch.multinomial(
            self.weights(h), n_per_h, replacement=True, generator=rng
        )
        latent = self.latent_means[component] + self.sigma_z * torch.randn(
            (len(h), n_per_h, self.latent_dim), generator=rng, dtype=DTYPE
        )
        observed_noise = self.sigma_obs * torch.randn(
            (len(h), n_per_h, self.dy), generator=rng, dtype=DTYPE
        )
        return latent @ self.embedding.T + observed_noise

    def _log_prob_aligned(
        self, y: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        covariance = self.base_covariance + torch.diag(sigma.square())
        return _mixture_full_log_prob(
            y,
            torch.log_softmax(self.logits(h), dim=-1),
            self.component_means,
            covariance,
        )

    def mechanism_directions(self) -> dict[str, torch.Tensor]:
        normalized = _unit_columns(self.v)
        return {f"interaction_rank_{i}": normalized[:, i].clone() for i in range(self.rank)}

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "latent_dim": self.latent_dim,
            "n_components": self.n_components,
            "interaction_rank": self.rank,
            "sigma_z": self.sigma_z,
            "sigma_obs": self.sigma_obs,
            "mode_separation": self.mode_separation,
        }


class G6RoughLocationGaussian(ConditionalDGP):
    """G6: small density perturbation with an order-one high-frequency derivative."""

    def __init__(
        self,
        seed: int = 0,
        q: int = 8,
        dy: int = 8,
        hidden: int = 16,
        omega: float = 8.0,
        roughness_amplitude: float = 0.5,
    ):
        super().__init__("g6_rough", q, dy, seed)
        if omega <= 0:
            raise ValueError("omega must be positive")
        rng = _generator(seed)
        self.hidden = int(hidden)
        self.omega = float(omega)
        self.roughness_amplitude = float(roughness_amplitude)
        self.w1, self.w2 = _location_parameters(q, dy, hidden, rng)
        self.response_direction = _unit_rows(
            torch.randn((1, dy), generator=rng, dtype=DTYPE)
        )[0]

    def mean(self, h: torch.Tensor) -> torch.Tensor:
        smooth = 0.5 * (torch.tanh(h @ self.w1.T) @ self.w2.T)
        rough = (
            self.roughness_amplitude
            / self.omega
            * torch.sin(self.omega * h[..., 0]).unsqueeze(-1)
            * self.response_direction
        )
        return smooth + rough

    def sample_response(self, h: torch.Tensor, n_per_h: int, seed: int) -> torch.Tensor:
        h = _tensor(h)
        if h.ndim != 2 or h.shape[1] != self.q or n_per_h < 1:
            raise ValueError("h must be [n,q] and n_per_h must be positive")
        noise = torch.randn(
            (len(h), n_per_h, self.dy), generator=_generator(seed), dtype=DTYPE
        )
        return self.mean(h)[:, None, :] + noise

    def _log_prob_aligned(
        self, y: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        return _diagonal_log_prob(y, self.mean(h), 1.0 + sigma.square())

    def mechanism_directions(self) -> dict[str, torch.Tensor]:
        axis = torch.zeros(self.q, dtype=DTYPE)
        axis[0] = 1.0
        return {"rough_history_axis": axis}

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "omega": self.omega,
            "roughness_amplitude": self.roughness_amplitude,
            "hidden_width": self.hidden,
        }


class G7ConcentratedHistory(ConditionalDGP):
    """G7: conditional law supported only on intrinsic history coordinates."""

    def __init__(
        self,
        seed: int = 0,
        q: int = 32,
        dy: int = 8,
        intrinsic_dim: int = 4,
        tau: float = 0.1,
        intrinsic: str = "g4",
    ):
        if not 1 <= intrinsic_dim < q or tau <= 0:
            raise ValueError("intrinsic_dim must lie in [1,q) and tau must be positive")
        super().__init__("g7_support", q, dy, seed)
        self.intrinsic_dim = int(intrinsic_dim)
        self.tau = float(tau)
        full_basis = _orthogonal(q, q, _generator(seed))
        self.support_basis = full_basis[:, :intrinsic_dim]
        self.orthogonal_basis = full_basis[:, intrinsic_dim:]
        intrinsic_key = intrinsic.lower().replace("-", "_")
        if intrinsic_key in {"g2", "g2_covariance", "covariance"}:
            self.intrinsic = G2CovarianceGaussian(
                seed=seed + 17_003, q=intrinsic_dim, dy=dy, alpha_m=0.3
            )
            self.intrinsic_kind = "g2"
        elif intrinsic_key in {"g4", "g4_mixture", "mixture"}:
            self.intrinsic = G4LowRankMixture(
                seed=seed + 17_003,
                q=intrinsic_dim,
                dy=dy,
                n_components=4,
                rank=min(2, intrinsic_dim),
                mode_separation=4.0,
            )
            self.intrinsic_kind = "g4"
        else:
            raise ValueError("intrinsic must be 'g2' or 'g4'")

    def intrinsic_history(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.support_basis

    def sample_history(self, n: int, seed: int) -> torch.Tensor:
        if n < 1:
            raise ValueError("n must be positive")
        rng = _generator(seed)
        z = torch.randn((n, self.intrinsic_dim), generator=rng, dtype=DTYPE)
        ambient = torch.randn((n, self.q), generator=rng, dtype=DTYPE)
        return z @ self.support_basis.T + self.tau * ambient

    def sample_response(self, h: torch.Tensor, n_per_h: int, seed: int) -> torch.Tensor:
        h = _tensor(h)
        if h.ndim != 2 or h.shape[1] != self.q:
            raise ValueError("h must be [n,q]")
        return self.intrinsic.sample_response(self.intrinsic_history(h), n_per_h, seed)

    def _log_prob_aligned(
        self, y: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        return self.intrinsic._log_prob_aligned(y, self.intrinsic_history(h), sigma)

    def mechanism_directions(self) -> dict[str, torch.Tensor]:
        directions = {
            f"supported_{i}": self.support_basis[:, i].clone()
            for i in range(self.intrinsic_dim)
        }
        directions.update(
            {
                f"orthogonal_{i}": self.orthogonal_basis[:, i].clone()
                for i in range(min(4, self.q - self.intrinsic_dim))
            }
        )
        return directions

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "intrinsic_dim": self.intrinsic_dim,
            "tau": self.tau,
            "intrinsic_law": self.intrinsic_kind,
        }


class G8FunctionalShapeMixture(ConditionalDGP):
    """G8: correlated future paths with a pure higher-shape regime change.

    The response is a flattened ``future_steps x n_channels`` path.  Two
    equally weighted signed motif frames have exactly the same zero mean and
    identity covariance in motif coordinates.  The final observed history
    coordinate changes only the mixing weight between those frames.  Hence the
    declared direction changes path topology and higher moments while the
    conditional mean and covariance remain exactly fixed.
    """

    def __init__(
        self,
        seed: int = 0,
        q: int = 25,
        dy: int = 48,
        n_channels: int = 4,
        motif_rank: int = 4,
        motif_scale: float = 0.45,
        noise_sd: float = 0.08,
        correlated_noise_scale: float = 0.06,
        shape_sensitivity: float = 1.2,
    ):
        if q < n_channels + 1 or (q - 1) % n_channels != 0:
            raise ValueError("q-1 must be a positive multiple of n_channels")
        if dy < n_channels or dy % n_channels != 0:
            raise ValueError("dy must be a positive multiple of n_channels")
        if not 2 <= motif_rank <= min(8, dy):
            raise ValueError("motif_rank must lie in [2,min(8,dy)]")
        if motif_scale <= 0 or noise_sd <= 0 or correlated_noise_scale < 0:
            raise ValueError("path scales must be positive")
        if shape_sensitivity <= 0:
            raise ValueError("shape_sensitivity must be positive")
        super().__init__("g8_functional_shape", q, dy, seed)
        self.n_channels = int(n_channels)
        self.history_steps = (q - 1) // n_channels
        self.future_steps = dy // n_channels
        self.motif_rank = int(motif_rank)
        self.motif_scale = float(motif_scale)
        self.noise_sd = float(noise_sd)
        self.correlated_noise_scale = float(correlated_noise_scale)
        self.shape_sensitivity = float(shape_sensitivity)
        rng = _generator(seed)

        raw_dynamics = torch.randn(
            (n_channels, n_channels), generator=rng, dtype=DTYPE
        ) / math.sqrt(n_channels)
        spectral = torch.linalg.eigvals(raw_dynamics).abs().max().real.clamp_min(1e-8)
        self.dynamics = 0.7 * raw_dynamics / spectral

        channel_grid = torch.arange(n_channels, dtype=DTYPE) / max(n_channels, 1)
        time_grid = torch.arange(self.future_steps, dtype=DTYPE) / max(self.future_steps, 1)
        motif_columns = []
        for index in range(motif_rank):
            frequency = 1 + index // 2
            phase = 2.0 * math.pi * (
                channel_grid[None, :] + ((-1.0) ** index) * time_grid[:, None]
            ) * frequency
            motif = torch.cos(phase) if index % 2 == 0 else torch.sin(phase)
            motif_columns.append(motif.reshape(-1))
        raw_motifs = torch.stack(motif_columns, dim=1)
        raw_motif_rank = int(torch.linalg.matrix_rank(raw_motifs).item())
        if raw_motif_rank != motif_rank:
            raise ValueError(
                "the requested G8 time/channel grid has a rank-deficient motif basis "
                f"(rank={raw_motif_rank}, requested={motif_rank})"
            )
        self.motifs, _ = torch.linalg.qr(raw_motifs, mode="reduced")

        rotation_raw = torch.randn(
            (motif_rank, motif_rank), generator=rng, dtype=DTYPE
        )
        self.motif_rotation, _ = torch.linalg.qr(rotation_raw)
        rotation_rank = int(torch.linalg.matrix_rank(self.motif_rotation).item())
        rotation_error = torch.max(
            torch.abs(
                self.motif_rotation.T @ self.motif_rotation
                - torch.eye(motif_rank, dtype=DTYPE)
            )
        )
        if rotation_rank != motif_rank or float(rotation_error) > 1e-10:
            raise RuntimeError("G8 motif rotation is not full-rank orthogonal")
        quartic_separation = motif_rank**2 - motif_rank * float(
            self.motif_rotation.pow(4).sum()
        )
        if quartic_separation <= 1e-8:
            raise RuntimeError(
                "G8 motif rotation is quartically degenerate; choose another generator seed"
            )
        axes = math.sqrt(motif_rank) * torch.eye(motif_rank, dtype=DTYPE)
        frame_a = torch.cat([axes, -axes], dim=0)
        rotated = axes @ self.motif_rotation.T
        frame_b = torch.cat([rotated, -rotated], dim=0)
        self.latent_codes = torch.cat([frame_a, frame_b], dim=0)
        self.components_per_frame = 2 * motif_rank
        self.component_offsets = (
            self.motif_scale * self.latent_codes @ self.motifs.T
        )

        smooth_columns = []
        for frequency in (1, 2, 3):
            phase = 2.0 * math.pi * (
                channel_grid[None, :] + 0.5 * time_grid[:, None]
            ) * frequency
            smooth_columns.append(torch.cos(phase).reshape(-1))
            smooth_columns.append(torch.sin(phase).reshape(-1))
        smooth = torch.stack(smooth_columns, dim=1)
        smooth, _ = torch.linalg.qr(smooth, mode="reduced")
        self.noise_modes = smooth
        self.base_covariance = (
            self.noise_sd**2 * torch.eye(dy, dtype=DTYPE)
            + self.correlated_noise_scale**2 * (smooth @ smooth.T)
        )
        self.base_cholesky = torch.linalg.cholesky(self.base_covariance)
        # Both signed frames have identity covariance in motif coordinates, so
        # this is the exact conditional covariance for every history/control.
        self.conditional_covariance = self.base_covariance + self.motif_scale**2 * (
            self.motifs @ self.motifs.T
        )

    def sample_history(self, n: int, seed: int) -> torch.Tensor:
        if n < 1:
            raise ValueError("n must be positive")
        rng = _generator(seed)
        state = 0.5 * torch.randn(
            (n, self.n_channels), generator=rng, dtype=DTYPE
        )
        trajectory = []
        for _ in range(self.history_steps):
            innovation = 0.18 * torch.randn(
                state.shape, generator=rng, dtype=DTYPE
            )
            state = torch.tanh(0.65 * state + 0.22 * torch.tanh(state) @ self.dynamics.T + innovation)
            trajectory.append(state)
        control = torch.randn((n, 1), generator=rng, dtype=DTYPE)
        return torch.cat([torch.stack(trajectory, dim=1).reshape(n, -1), control], dim=-1)

    def path_mean(self, h: torch.Tensor) -> torch.Tensor:
        h = _tensor(h)
        state = h[..., -1 - self.n_channels : -1]
        future = []
        for _ in range(self.future_steps):
            state = 0.78 * state + 0.16 * torch.tanh(state) @ self.dynamics.T
            future.append(state)
        return torch.stack(future, dim=-2).reshape(*h.shape[:-1], self.dy)

    def shape_probability(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.shape_sensitivity * h[..., -1])

    def log_weights(self, h: torch.Tensor) -> torch.Tensor:
        probability = self.shape_probability(h).clamp(1e-12, 1.0 - 1e-12)
        log_count = math.log(self.components_per_frame)
        first = (torch.log1p(-probability) - log_count)[..., None].expand(
            *probability.shape, self.components_per_frame
        )
        second = (torch.log(probability) - log_count)[..., None].expand(
            *probability.shape, self.components_per_frame
        )
        return torch.cat([first, second], dim=-1)

    def component_means(self, h: torch.Tensor) -> torch.Tensor:
        return self.path_mean(h)[..., None, :] + self.component_offsets

    def covariance_at(self, h: torch.Tensor | None = None) -> torch.Tensor:
        """Return the exact, control-invariant conditional path covariance."""

        if h is None:
            return self.conditional_covariance.clone()
        history = _tensor(h)
        if history.ndim < 1 or history.shape[-1] != self.q:
            raise ValueError(f"h must end in q={self.q}")
        return self.conditional_covariance.expand(
            history.shape[:-1] + (self.dy, self.dy)
        )

    def sample_response(self, h: torch.Tensor, n_per_h: int, seed: int) -> torch.Tensor:
        h = _tensor(h)
        if h.ndim != 2 or h.shape[1] != self.q or n_per_h < 1:
            raise ValueError("h must be [n,q] and n_per_h must be positive")
        rng = _generator(seed)
        weights = torch.exp(self.log_weights(h))
        component = torch.multinomial(weights, n_per_h, replacement=True, generator=rng)
        means = self.component_means(h)
        selected = means.gather(
            1, component[..., None].expand(-1, -1, self.dy)
        )
        noise = torch.randn(
            (len(h), n_per_h, self.dy), generator=rng, dtype=DTYPE
        )
        return selected + noise @ self.base_cholesky.T

    def _log_prob_aligned(
        self, y: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        covariance = self.base_covariance + torch.diag(sigma.square())
        chol = torch.linalg.cholesky(covariance)
        means = self.component_means(h)
        difference = y[..., None, :] - means
        whitened = torch.linalg.solve_triangular(
            chol, difference.unsqueeze(-1), upper=False
        ).squeeze(-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        component = -0.5 * (
            self.dy * LOG_2PI + logdet + whitened.square().sum(dim=-1)
        )
        return torch.logsumexp(self.log_weights(h) + component, dim=-1)

    def mechanism_directions(self) -> dict[str, torch.Tensor]:
        direction = torch.zeros(self.q, dtype=DTYPE)
        direction[-1] = 1.0
        return {"shape_only_control": direction}

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "n_channels": self.n_channels,
            "history_steps": self.history_steps,
            "future_steps": self.future_steps,
            "motif_rank": self.motif_rank,
            "n_components": 4 * self.motif_rank,
            "motif_scale": self.motif_scale,
            "noise_sd": self.noise_sd,
            "correlated_noise_scale": self.correlated_noise_scale,
            "shape_sensitivity": self.shape_sensitivity,
            "shape_only_mean_covariance_invariant": True,
        }


class G8GaussianShadow(G8FunctionalShapeMixture):
    """Exact Gaussian negative control for :class:`G8FunctionalShapeMixture`.

    For the same seed and constructor parameters, the shadow shares G8's
    nonlinear path mean and its full conditional covariance.  It replaces the
    two signed motif frames by one Gaussian law, so the final history control
    has exactly zero effect on every aspect of the conditional distribution.
    This separates genuine higher-shape sensitivity from accidental responses
    to mean, covariance, or path correlation.
    """

    def __init__(
        self,
        seed: int = 0,
        q: int = 25,
        dy: int = 48,
        n_channels: int = 4,
        motif_rank: int = 4,
        motif_scale: float = 0.45,
        noise_sd: float = 0.08,
        correlated_noise_scale: float = 0.06,
        shape_sensitivity: float = 1.2,
    ):
        super().__init__(
            seed=seed,
            q=q,
            dy=dy,
            n_channels=n_channels,
            motif_rank=motif_rank,
            motif_scale=motif_scale,
            noise_sd=noise_sd,
            correlated_noise_scale=correlated_noise_scale,
            shape_sensitivity=shape_sensitivity,
        )
        self.name = "g8_gaussian_shadow"
        self.shadow_cholesky = torch.linalg.cholesky(self.conditional_covariance)

    @classmethod
    def from_g8(cls, dgp: G8FunctionalShapeMixture) -> "G8GaussianShadow":
        """Reconstruct the exactly parameter-matched shadow for ``dgp``."""

        if not isinstance(dgp, G8FunctionalShapeMixture):
            raise TypeError("dgp must be a G8FunctionalShapeMixture")
        return cls(
            seed=dgp.seed,
            q=dgp.q,
            dy=dgp.dy,
            n_channels=dgp.n_channels,
            motif_rank=dgp.motif_rank,
            motif_scale=dgp.motif_scale,
            noise_sd=dgp.noise_sd,
            correlated_noise_scale=dgp.correlated_noise_scale,
            shape_sensitivity=dgp.shape_sensitivity,
        )

    def sample_response(self, h: torch.Tensor, n_per_h: int, seed: int) -> torch.Tensor:
        h = _tensor(h)
        if h.ndim != 2 or h.shape[1] != self.q or n_per_h < 1:
            raise ValueError("h must be [n,q] and n_per_h must be positive")
        noise = torch.randn(
            (len(h), n_per_h, self.dy), generator=_generator(seed), dtype=DTYPE
        )
        return self.path_mean(h)[:, None, :] + noise @ self.shadow_cholesky.T

    def _log_prob_aligned(
        self, y: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        covariance = self.conditional_covariance + torch.diag(sigma.square())
        return _mvn_log_prob(y, self.path_mean(h), covariance)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "shadow_of": "g8_functional_shape",
            "shape_control_null": True,
            "matches_g8_conditional_mean": True,
            "matches_g8_conditional_covariance": True,
        }


_FACTORY = {
    "g1": G1LocationGaussian,
    "g1_location": G1LocationGaussian,
    "g1_location_gaussian": G1LocationGaussian,
    "g1_location_only": G1LocationGaussian,
    "location": G1LocationGaussian,
    "g2": G2CovarianceGaussian,
    "g2_covariance": G2CovarianceGaussian,
    "g2_covariance_gaussian": G2CovarianceGaussian,
    "covariance": G2CovarianceGaussian,
    "g3": G3SkewMixture,
    "g3_skew": G3SkewMixture,
    "g3_skewness": G3SkewMixture,
    "skew": G3SkewMixture,
    "g4": G4LowRankMixture,
    "g4_low_rank_mixture": G4LowRankMixture,
    "g4_mixture": G4LowRankMixture,
    "low_rank_mixture": G4LowRankMixture,
    "g5": G5NearManifoldMixture,
    "g5_near_manifold": G5NearManifoldMixture,
    "g5_manifold": G5NearManifoldMixture,
    "near_manifold": G5NearManifoldMixture,
    "g6": G6RoughLocationGaussian,
    "g6_rough": G6RoughLocationGaussian,
    "g6_roughness": G6RoughLocationGaussian,
    "rough": G6RoughLocationGaussian,
    "g7": G7ConcentratedHistory,
    "g7_support": G7ConcentratedHistory,
    "g7_concentrated_history": G7ConcentratedHistory,
    "support": G7ConcentratedHistory,
    "g8": G8FunctionalShapeMixture,
    "g8_functional": G8FunctionalShapeMixture,
    "g8_functional_shape": G8FunctionalShapeMixture,
    "functional_shape": G8FunctionalShapeMixture,
    "g8_shadow": G8GaussianShadow,
    "g8_gaussian_shadow": G8GaussianShadow,
    "functional_shape_shadow": G8GaussianShadow,
}


def make_dgp(name: str, seed: int = 0, **kwargs: Any) -> ConditionalDGP:
    """Construct a deterministic fixed-parameter continuous generator."""

    key = str(name).strip().lower().replace("-", "_")
    if key not in _FACTORY:
        raise ValueError(
            f"unknown DGP {name!r}; choose one of g1, g2, g3, g4, g5, g6, g7, "
            "g8, g8_shadow"
        )
    aliases = {
        "history_dim": "q",
        "response_dim": "dy",
        "d_y": "dy",
        "hidden_width": "hidden",
        "k": "n_components",
        "K": "n_components",
        "interaction_rank": "rank",
        "r": "rank",
        "r_over_sigma": "mode_separation",
        "R_over_sigma": "mode_separation",
        "r_z": "latent_dim",
        "r_h": "intrinsic_dim",
        "intrinsic_generator": "intrinsic",
    }
    normalized: dict[str, Any] = {}
    for parameter, value in kwargs.items():
        target = aliases.get(parameter, parameter)
        if target in normalized:
            raise ValueError(f"duplicate DGP parameter after alias normalization: {target!r}")
        normalized[target] = value
    return _FACTORY[key](seed=seed, **normalized)


__all__ = [
    "ConditionalDGP",
    "G1LocationGaussian",
    "G2CovarianceGaussian",
    "G3SkewMixture",
    "G4LowRankMixture",
    "G5NearManifoldMixture",
    "G6RoughLocationGaussian",
    "G7ConcentratedHistory",
    "G8FunctionalShapeMixture",
    "G8GaussianShadow",
    "make_dgp",
]
