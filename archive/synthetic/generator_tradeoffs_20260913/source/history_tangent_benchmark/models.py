"""Neural model families used by the developmental history-tangent benchmark.

The module deliberately keeps every capability explicit.  In particular, a
conditional score model is a response-score estimator and sampler; it only
becomes a history-tangent estimator through the centered mixed-derivative
construction implemented below.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


LOG_2PI = math.log(2.0 * math.pi)


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a requested training device without moving oracle work off CPU."""
    if name == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


@dataclass(frozen=True)
class Capabilities:
    normalized_density: bool
    sampler: bool
    history_tangent: bool
    response_score: bool = False
    tangent_requires_centering: bool = False


@dataclass
class FitTrace:
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int
    stopped_epoch: int
    wall_seconds: float
    parameter_count: int
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidualMLP(nn.Module):
    """Small residual MLP with the plan's SiLU/LayerNorm convention."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int, layers: int = 3):
        super().__init__()
        if hidden < 4 or layers < 1:
            raise ValueError("hidden must be >= 4 and layers must be positive")
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))
            for _ in range(layers)
        )
        self.output = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.silu(self.input(x))
        for index, block in enumerate(self.blocks):
            update = block(z)
            z = z + update if index % 2 else update
        return self.output(z)


class ConditionalModel(nn.Module):
    name = "base"
    capabilities = Capabilities(False, False, False)

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def validation_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.native_loss(h, y)

    def history_tangent(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if not self.capabilities.normalized_density:
            raise NotImplementedError(f"{self.name} has no direct normalized tangent")
        h_req = h.detach().clone().requires_grad_(True)
        log_density = self.log_prob(y.detach(), h_req)
        return torch.autograd.grad(log_density.sum(), h_req)[0]

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class MeanMLP(ConditionalModel):
    """M0: deterministic conditional mean with a labeled fixed residual model."""

    name = "mean_mlp"
    capabilities = Capabilities(False, True, False)

    def __init__(self, q: int, dy: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.q, self.dy = int(q), int(dy)
        self.network = ResidualMLP(q, dy, hidden, layers)
        self.register_buffer("residual_std", torch.ones(dy))

    def mean(self, h: torch.Tensor) -> torch.Tensor:
        return self.network(h)

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.mean(h), y)

    def finalize_residual(self, h: torch.Tensor, y: torch.Tensor) -> None:
        with torch.no_grad():
            scale = torch.sqrt(torch.mean((y - self.mean(h)) ** 2, dim=0).clamp_min(1e-6))
            self.residual_std.copy_(scale)

    def sample(self, h: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        mean = self.mean(h)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            (len(h), n_samples, self.dy), generator=generator, dtype=mean.dtype
        ).to(mean.device)
        return mean[:, None, :] + self.residual_std[None, None, :] * noise

    def mean_jacobian(self, h: torch.Tensor) -> torch.Tensor:
        rows = []
        for target in range(self.dy):
            h_req = h.detach().clone().requires_grad_(True)
            value = self.mean(h_req)[:, target]
            rows.append(torch.autograd.grad(value.sum(), h_req)[0])
        return torch.stack(rows, dim=1)


class HeteroscedasticGaussian(ConditionalModel):
    """M1 smoke implementation: normalized diagonal heteroscedastic Gaussian."""

    name = "heteroscedastic_gaussian"
    capabilities = Capabilities(True, True, True, response_score=True)

    def __init__(self, q: int, dy: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.q, self.dy = int(q), int(dy)
        self.network = ResidualMLP(q, 2 * dy, hidden, layers)

    def parameters_at(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = torch.chunk(self.network(h), 2, dim=-1)
        return mean, log_std.clamp(-5.0, 3.0)

    def log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.parameters_at(h)
        z = (y - mean) * torch.exp(-log_std)
        return (-0.5 * (z.square() + LOG_2PI) - log_std).sum(dim=-1)

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(y, h).mean()

    def sample(self, h: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        mean, log_std = self.parameters_at(h)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            (len(h), n_samples, self.dy), generator=generator, dtype=mean.dtype
        ).to(mean.device)
        return mean[:, None, :] + torch.exp(log_std)[:, None, :] * noise

    def response_score(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.parameters_at(h)
        return -(y - mean) * torch.exp(-2.0 * log_std)


class ConstrainedGaussianDSM(HeteroscedasticGaussian):
    """Positive Gaussian law fit by a multi-noise denoising-score objective.

    This is the objective-controlled comparator for Gaussian NLL.  The clean
    variance is parameterized as ``exp(2 log_std)`` and clamped away from zero,
    so DSM cannot cross the quadratic natural-variance boundary.
    """

    name = "constrained_gaussian_dsm"

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 64,
        layers: int = 3,
        sigma_min: float = 0.03,
        sigma_max: float = 1.0,
    ):
        super().__init__(q=q, dy=dy, hidden=hidden, layers=layers)
        if not 0 < sigma_min < sigma_max:
            raise ValueError("need 0 < sigma_min < sigma_max")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def _dsm_loss(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        *,
        uniform: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sigma = torch.exp(
            math.log(self.sigma_min)
            + uniform * math.log(self.sigma_max / self.sigma_min)
        )
        noisy = y + sigma * noise
        mean, log_std = self.parameters_at(h)
        variance = torch.exp(2.0 * log_std) + sigma.square()
        predicted_score = -(noisy - mean) / variance
        target_score = -noise / sigma
        return torch.mean(sigma.square() * (predicted_score - target_score).square())

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand((len(y), 1), dtype=y.dtype, device=y.device)
        return self._dsm_loss(h, y, uniform=uniform, noise=torch.randn_like(y))

    def validation_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(76123)
        uniform = torch.rand((len(y), 1), generator=generator, dtype=y.dtype).to(y.device)
        noise = torch.randn(y.shape, generator=generator, dtype=y.dtype).to(y.device)
        return self._dsm_loss(h, y, uniform=uniform, noise=noise)


class ConditionalAffineFlow(ConditionalModel):
    """Compact conditional RealNVP baseline with an exact normalized density."""

    name = "conditional_affine_flow"
    capabilities = Capabilities(True, True, True, response_score=True)

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 64,
        layers: int = 2,
        coupling_layers: int = 6,
        max_log_scale: float = 1.5,
    ):
        super().__init__()
        if coupling_layers < 2 or max_log_scale <= 0:
            raise ValueError("flow needs at least two coupling layers and positive scale bound")
        self.q, self.dy = int(q), int(dy)
        self.max_log_scale = float(max_log_scale)
        self.conditioners = nn.ModuleList(
            ResidualMLP(q + dy, 2 * dy, hidden, layers) for _ in range(coupling_layers)
        )
        for index in range(coupling_layers):
            if dy == 1:
                mask = torch.zeros(1)
            else:
                mask = ((torch.arange(dy) + index) % 2).to(torch.float32)
            self.register_buffer(f"mask_{index}", mask)

    def _mask(self, index: int, value: torch.Tensor) -> torch.Tensor:
        return getattr(self, f"mask_{index}").to(dtype=value.dtype, device=value.device)

    def _coupling_parameters(
        self, index: int, value: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self._mask(index, value)
        raw_shift, raw_scale = torch.chunk(
            self.conditioners[index](torch.cat([h, value * mask], dim=-1)), 2, dim=-1
        )
        transformed = 1.0 - mask
        shift = raw_shift * transformed
        log_scale = self.max_log_scale * torch.tanh(raw_scale) * transformed
        return mask, shift, log_scale

    def _forward(self, y: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = y
        logdet = torch.zeros(y.shape[:-1], dtype=y.dtype, device=y.device)
        for index in range(len(self.conditioners)):
            mask, shift, log_scale = self._coupling_parameters(index, value, h)
            value = value * mask + (1.0 - mask) * (value - shift) * torch.exp(-log_scale)
            logdet = logdet - log_scale.sum(dim=-1)
        return value, logdet

    def _inverse(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        value = z
        for index in reversed(range(len(self.conditioners))):
            mask, shift, log_scale = self._coupling_parameters(index, value, h)
            value = value * mask + (1.0 - mask) * (value * torch.exp(log_scale) + shift)
        return value

    def log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z, logdet = self._forward(y, h)
        return -0.5 * (z.square() + LOG_2PI).sum(dim=-1) + logdet

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(y, h).mean()

    def sample(self, h: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        repeated = h[:, None, :].expand(len(h), n_samples, self.q).reshape(-1, self.q)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn((len(repeated), self.dy), generator=generator, dtype=h.dtype).to(h.device)
        return self._inverse(z, repeated).reshape(len(h), n_samples, self.dy)

    def response_score(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y_req = y.detach().clone().requires_grad_(True)
        return torch.autograd.grad(self.log_prob(y_req, h.detach()).sum(), y_req)[0]


class AutoregressiveMixtureDensity(ConditionalModel):
    """M2: normalized output-autoregressive Gaussian-mixture density.

    A separate conditional network is used for each output coordinate.  This is
    a compact masked-autoregressive implementation for the low-dimensional
    smoke tier; every conditional uses the prescribed eight Gaussian mixture
    components by default.
    """

    name = "autoregressive_mdn"
    capabilities = Capabilities(True, True, True, response_score=True)

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 64,
        layers: int = 2,
        components: int = 8,
        permutation_seed: int = 0,
    ):
        super().__init__()
        self.q, self.dy, self.components = int(q), int(dy), int(components)
        self.permutation_seed = int(permutation_seed)
        permutation = (
            np.arange(self.dy, dtype=np.int64)
            if self.permutation_seed == 0
            else np.random.default_rng(self.permutation_seed).permutation(self.dy)
        )
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(self.dy, dtype=np.int64)
        self.register_buffer(
            "coordinate_permutation", torch.as_tensor(permutation), persistent=False
        )
        self.register_buffer(
            "inverse_coordinate_permutation", torch.as_tensor(inverse), persistent=False
        )
        self.conditionals = nn.ModuleList(
            ResidualMLP(q + output_index, 3 * components, hidden, layers)
            for output_index in range(dy)
        )

    def _conditional_parameters(
        self, output_index: int, h: torch.Tensor, previous: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = h if output_index == 0 else torch.cat([h, previous], dim=-1)
        logits, mean, log_std = torch.chunk(self.conditionals[output_index](context), 3, dim=-1)
        return logits, mean, log_std.clamp(-5.0, 2.5)

    def log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ordered = y.index_select(-1, self.coordinate_permutation)
        total = torch.zeros(y.shape[:-1], dtype=y.dtype, device=y.device)
        for output_index in range(self.dy):
            logits, mean, log_std = self._conditional_parameters(
                output_index, h, ordered[..., :output_index]
            )
            target = ordered[..., output_index, None]
            z = (target - mean) * torch.exp(-log_std)
            component_log_prob = -0.5 * (z.square() + LOG_2PI) - log_std
            total = total + torch.logsumexp(
                F.log_softmax(logits, dim=-1) + component_log_prob, dim=-1
            )
        return total

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(y, h).mean()

    def sample(self, h: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        batch = len(h)
        repeated_h = h[:, None, :].expand(batch, n_samples, self.q).reshape(-1, self.q)
        generated = torch.empty(
            (batch * n_samples, 0), dtype=h.dtype, device=h.device
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # Preserve the original alternating CPU RNG stream, but transfer all
        # uniforms and Gaussian draws to the accelerator in two batches.  The
        # previous per-coordinate `.to(h.device)` calls forced 2 * dy device
        # synchronizations for every autoregressive sample.
        uniform_draws = []
        gaussian_draws = []
        for _ in range(self.dy):
            uniform_draws.append(
                torch.rand((len(repeated_h), 1), generator=generator, dtype=h.dtype)
            )
            gaussian_draws.append(
                torch.randn((len(repeated_h),), generator=generator, dtype=h.dtype)
            )
        uniform_draws_device = torch.stack(uniform_draws, dim=1).to(h.device)
        gaussian_draws_device = torch.stack(gaussian_draws, dim=1).to(h.device)
        for output_index in range(self.dy):
            logits, mean, log_std = self._conditional_parameters(
                output_index, repeated_h, generated
            )
            uniforms = uniform_draws_device[:, output_index]
            cumulative = F.softmax(logits, dim=-1).cumsum(dim=-1)
            component = (uniforms > cumulative).sum(dim=-1).clamp_max(self.components - 1)
            chosen_mean = mean.gather(1, component[:, None]).squeeze(1)
            chosen_std = torch.exp(log_std.gather(1, component[:, None]).squeeze(1))
            noise = gaussian_draws_device[:, output_index]
            draw = chosen_mean + chosen_std * noise
            generated = torch.cat([generated, draw[:, None]], dim=-1)
        ordered = generated.reshape(batch, n_samples, self.dy)
        response = torch.empty_like(ordered)
        response[..., self.coordinate_permutation] = ordered
        return response

    def response_score(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y_req = y.detach().clone().requires_grad_(True)
        score = torch.autograd.grad(self.log_prob(y_req, h.detach()).sum(), y_req)[0]
        return score


class AutoregressiveTransformerDensity(ConditionalModel):
    """Normalized continuous autoregressive Transformer mixture density.

    Response coordinates are treated as a short causal sequence.  Token ``j``
    receives the full conditioning history, a coordinate embedding, and the
    teacher-forced value of coordinate ``j - 1``.  A causal Transformer then
    predicts an eight-component Gaussian mixture for coordinate ``j``.  The
    optional fixed coordinate permutation makes ordering sensitivity an
    explicit experimental axis instead of an implicit implementation detail.
    """

    name = "autoregressive_transformer"
    capabilities = Capabilities(True, True, True, response_score=True)

    def __init__(
        self,
        q: int,
        dy: int,
        d_model: int = 48,
        nhead: int = 4,
        transformer_layers: int = 2,
        feedforward: int = 128,
        components: int = 8,
        dropout: float = 0.0,
        permutation_seed: int = 0,
    ):
        super().__init__()
        if d_model < 8 or nhead < 1 or d_model % nhead:
            raise ValueError("d_model must be >= 8 and divisible by nhead")
        if transformer_layers < 1 or feedforward < d_model or components < 1:
            raise ValueError("invalid Transformer depth, feedforward width, or components")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.q = int(q)
        self.dy = int(dy)
        self.d_model = int(d_model)
        self.components = int(components)
        self.permutation_seed = int(permutation_seed)

        if self.permutation_seed == 0:
            permutation = np.arange(self.dy, dtype=np.int64)
        else:
            permutation = np.random.default_rng(self.permutation_seed).permutation(self.dy)
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(self.dy, dtype=np.int64)
        self.register_buffer("coordinate_permutation", torch.as_tensor(permutation))
        self.register_buffer("inverse_coordinate_permutation", torch.as_tensor(inverse))

        self.history_projection = nn.Sequential(
            nn.Linear(self.q, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
        )
        self.previous_value_projection = nn.Linear(1, self.d_model)
        self.coordinate_embedding = nn.Embedding(self.dy, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=int(transformer_layers), norm=nn.LayerNorm(self.d_model)
        )
        self.mixture_head = nn.Linear(self.d_model, 3 * self.components)

    def _ordered(self, y: torch.Tensor) -> torch.Tensor:
        return y.index_select(-1, self.coordinate_permutation)

    def _conditional_parameters(
        self, h: torch.Tensor, ordered_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h.ndim != 2 or ordered_y.ndim != 2 or len(h) != len(ordered_y):
            raise ValueError("h and y must be matching rank-two batches")
        if h.shape[1] != self.q or ordered_y.shape[1] != self.dy:
            raise ValueError("history or response dimension mismatch")
        previous = torch.zeros_like(ordered_y)
        if self.dy > 1:
            previous[:, 1:] = ordered_y[:, :-1]
        positions = torch.arange(self.dy, device=h.device)
        tokens = (
            self.history_projection(h)[:, None, :]
            + self.previous_value_projection(previous[..., None])
            + self.coordinate_embedding(positions)[None, :, :]
        )
        causal_mask = torch.triu(
            torch.ones((self.dy, self.dy), dtype=torch.bool, device=h.device),
            diagonal=1,
        )
        encoded = self.transformer(tokens, mask=causal_mask)
        logits, means, log_stds = torch.chunk(self.mixture_head(encoded), 3, dim=-1)
        return logits, means, log_stds.clamp(-5.0, 2.5)

    def log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ordered = self._ordered(y)
        logits, means, log_stds = self._conditional_parameters(h, ordered)
        target = ordered[..., None]
        z = (target - means) * torch.exp(-log_stds)
        component = -0.5 * (z.square() + LOG_2PI) - log_stds
        return torch.logsumexp(F.log_softmax(logits, dim=-1) + component, dim=-1).sum(
            dim=-1
        )

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(y, h).mean()

    def sample(self, h: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        batch = len(h)
        repeated_h = h[:, None, :].expand(batch, n_samples, self.q).reshape(-1, self.q)
        ordered = torch.zeros(
            (len(repeated_h), self.dy), dtype=h.dtype, device=h.device
        )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        for output_index in range(self.dy):
            logits, means, log_stds = self._conditional_parameters(repeated_h, ordered)
            probability = F.softmax(logits[:, output_index], dim=-1)
            uniforms = torch.rand(
                (len(repeated_h), 1), generator=generator, dtype=h.dtype
            ).to(h.device)
            component = (uniforms > probability.cumsum(dim=-1)).sum(dim=-1).clamp_max(
                self.components - 1
            )
            chosen_mean = means[:, output_index].gather(1, component[:, None]).squeeze(1)
            chosen_std = torch.exp(
                log_stds[:, output_index].gather(1, component[:, None]).squeeze(1)
            )
            noise = torch.randn(
                (len(repeated_h),), generator=generator, dtype=h.dtype
            ).to(h.device)
            ordered[:, output_index] = chosen_mean + chosen_std * noise
        response = torch.empty_like(ordered)
        response[:, self.coordinate_permutation] = ordered
        return response.reshape(batch, n_samples, self.dy)

    def response_score(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y_req = y.detach().clone().requires_grad_(True)
        return torch.autograd.grad(self.log_prob(y_req, h.detach()).sum(), y_req)[0]


class RatioCritic(ConditionalModel):
    """M4 interaction-only joint-versus-product density-ratio critic.

    The plan's unrestricted history-only term is intentionally removed: its
    derivative is unidentified by the classification objective and can create
    an arbitrary nuisance tangent.  This developmental implementation uses a
    bilinear interaction plus a nonlinear function of interaction features.
    """

    name = "ratio_critic"
    capabilities = Capabilities(False, False, True)

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 64,
        layers: int = 2,
        rank: int = 32,
    ):
        super().__init__()
        self.q, self.dy, self.rank = int(q), int(dy), int(rank)
        self.history_features = ResidualMLP(q, rank, hidden, layers)
        self.response_features = ResidualMLP(dy, rank, hidden, layers)
        self.residual = nn.Sequential(
            nn.Linear(rank, max(8, rank // 2)),
            nn.SiLU(),
            nn.Linear(max(8, rank // 2), 1),
        )

    def log_ratio(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        history = self.history_features(h)
        response = self.response_features(y)
        interaction = history * response / math.sqrt(self.rank)
        return interaction.sum(dim=-1) + self.residual(interaction).squeeze(-1)

    @staticmethod
    def _negative_index(n: int, device: torch.device, random: bool) -> torch.Tensor:
        if n < 2:
            raise ValueError("ratio training requires batches of at least two")
        if random:
            offset = int(torch.randint(1, n, (1,), device=device).item())
        else:
            offset = 1
        return torch.roll(torch.arange(n, device=device), shifts=offset)

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        negative = y[self._negative_index(len(y), y.device, random=self.training)]
        positive_logits = self.log_ratio(h, y)
        negative_logits = self.log_ratio(h, negative)
        return 0.5 * (
            F.softplus(-positive_logits).mean() + F.softplus(negative_logits).mean()
        )

    def validation_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        value = self.native_loss(h, y)
        self.train(was_training)
        return value

    def history_tangent(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_req = h.detach().clone().requires_grad_(True)
        ratio = self.log_ratio(h_req, y.detach())
        return torch.autograd.grad(ratio.sum(), h_req)[0]

    def diagnostics(self, h: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        self.eval()
        with torch.no_grad():
            negative = y[self._negative_index(len(y), y.device, random=False)]
            positive = torch.sigmoid(self.log_ratio(h, y))
            negative_prob = torch.sigmoid(self.log_ratio(h, negative))
            probabilities = torch.cat([positive, negative_prob])
            labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative_prob)])
            brier = torch.mean((probabilities - labels).square())
            saturation = torch.mean(
                ((probabilities < 0.01) | (probabilities > 0.99)).to(probabilities.dtype)
            )
        return {
            "brier": float(brier.cpu()),
            "saturation_rate": float(saturation.cpu()),
            "mean_positive_probability": float(positive.mean().cpu()),
            "mean_negative_probability": float(negative_prob.mean().cpu()),
        }


class ConditionalVectorDiffusion(ConditionalModel):
    """M5 response-only VE denoising score model with a valid M5b audit route."""

    name = "conditional_vector_diffusion"
    capabilities = Capabilities(
        False, True, True, response_score=True, tangent_requires_centering=True
    )

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 96,
        layers: int = 3,
        sigma_min: float = 0.01,
        sigma_max: float = 2.0,
    ):
        super().__init__()
        if not 0 < sigma_min < sigma_max:
            raise ValueError("need 0 < sigma_min < sigma_max")
        self.q, self.dy = int(q), int(dy)
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.network = ResidualMLP(q + dy + 5, dy, hidden, layers)

    def _sigma_features(self, sigma: torch.Tensor) -> torch.Tensor:
        log_sigma = torch.log(sigma)
        scaled = (log_sigma - math.log(self.sigma_min)) / math.log(
            self.sigma_max / self.sigma_min
        )
        return torch.cat(
            [scaled, torch.sin(math.pi * scaled), torch.cos(math.pi * scaled), sigma, 1 / (1 + sigma)],
            dim=-1,
        )

    def score(self, y_noisy: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            sigma = torch.full(
                (*y_noisy.shape[:-1], 1), float(sigma), dtype=y_noisy.dtype, device=y_noisy.device
            )
        elif sigma.ndim == 0:
            sigma = sigma.expand(*y_noisy.shape[:-1], 1)
        elif sigma.shape[-1] != 1:
            sigma = sigma[..., None]
        return self.network(torch.cat([h, y_noisy, self._sigma_features(sigma)], dim=-1))

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand((len(y), 1), dtype=y.dtype, device=y.device)
        sigma = torch.exp(
            math.log(self.sigma_min)
            + uniform * math.log(self.sigma_max / self.sigma_min)
        )
        noise = torch.randn_like(y)
        noisy = y + sigma * noise
        predicted = self.score(noisy, h, sigma)
        target = -noise / sigma
        return torch.mean(sigma.square() * (predicted - target).square())

    def validation_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(99173)
        uniform = torch.rand((len(y), 1), generator=generator, dtype=y.dtype).to(y.device)
        noise = torch.randn(y.shape, generator=generator, dtype=y.dtype).to(y.device)
        sigma = torch.exp(
            math.log(self.sigma_min)
            + uniform * math.log(self.sigma_max / self.sigma_min)
        )
        predicted = self.score(y + sigma * noise, h, sigma)
        return torch.mean(sigma.square() * (predicted + noise / sigma).square())

    def response_score(
        self, y_noisy: torch.Tensor, h: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        return self.score(y_noisy, h, sigma)

    def sample(
        self,
        h: torch.Tensor,
        n_samples: int,
        seed: int = 0,
        levels: int = 16,
        steps_per_level: int = 4,
        step_size: float = 0.03,
    ) -> torch.Tensor:
        """Approximate clean samples via annealed Langevin dynamics.

        This sampler is intentionally labeled approximate.  It supports smoke
        diagnostics and M5b model-centering, not probability-flow likelihood.
        """
        batch = len(h)
        repeated_h = h[:, None, :].expand(batch, n_samples, self.q).reshape(-1, self.q)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.randn(
            (len(repeated_h), self.dy), generator=generator, dtype=h.dtype
        ).to(h.device)
        y = initial * math.sqrt(1.0 + self.sigma_max**2)
        schedule = torch.logspace(
            math.log10(self.sigma_max),
            math.log10(self.sigma_min),
            levels,
            dtype=h.dtype,
            device="cpu",
        ).to(h.device)
        with torch.no_grad():
            for sigma in schedule:
                step = step_size * float((sigma / self.sigma_max) ** 2)
                for _ in range(steps_per_level):
                    noise = torch.randn(y.shape, generator=generator, dtype=h.dtype).to(h.device)
                    y = y + step * self.score(y, repeated_h, sigma) + math.sqrt(2 * step) * noise
        return y.reshape(batch, n_samples, self.dy)

    def sample_noisy(
        self, h: torch.Tensor, n_samples: int, sigma: float, seed: int = 0
    ) -> torch.Tensor:
        clean = self.sample(h, n_samples, seed=seed)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype).to(clean.device)
        return clean + float(sigma) * noise

    def mixed_history_tangent_uncentered(
        self,
        y_noisy: torch.Tensor,
        h: torch.Tensor,
        sigma: float,
        quadrature_points: int = 8,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate J_h s_y along a straight response-space path."""
        if y_noisy.ndim != 2 or h.ndim != 2 or len(y_noisy) != len(h):
            raise ValueError("y_noisy and h must be matching rank-two batches")
        y0 = torch.zeros_like(y_noisy) if reference is None else reference
        delta = y_noisy - y0
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
        result = torch.zeros_like(h)
        for node, weight in zip(nodes, weights, strict=True):
            fraction = 0.5 * (float(node) + 1.0)
            h_req = h.detach().clone().requires_grad_(True)
            point = y0 + fraction * delta
            score = self.score(point.detach(), h_req, float(sigma))
            directional = (score * delta).sum(dim=-1)
            gradient = torch.autograd.grad(directional.sum(), h_req)[0]
            result = result + 0.5 * float(weight) * gradient.detach()
        return result

    def mixed_history_tangent(
        self,
        y_noisy: torch.Tensor,
        h: torch.Tensor,
        sigma: float,
        center_samples: torch.Tensor,
        quadrature_points: int = 8,
        center_chunk_size: int = 256,
    ) -> torch.Tensor:
        """Conditionally center the M5b reconstruction with declared samples."""
        if center_samples.ndim != 3 or center_samples.shape[0] != len(h):
            raise ValueError("center_samples must have shape [batch, samples, dy]")
        target = self.mixed_history_tangent_uncentered(
            y_noisy, h, sigma, quadrature_points=quadrature_points
        )
        batch, samples, dy = center_samples.shape
        flat_y = center_samples.reshape(batch * samples, dy)
        flat_h = h[:, None, :].expand(batch, samples, self.q).reshape(batch * samples, self.q)
        pieces = []
        for start in range(0, len(flat_y), center_chunk_size):
            pieces.append(
                self.mixed_history_tangent_uncentered(
                    flat_y[start : start + center_chunk_size],
                    flat_h[start : start + center_chunk_size],
                    sigma,
                    quadrature_points=quadrature_points,
                )
            )
        center = torch.cat(pieces, dim=0).reshape(batch, samples, self.q).mean(dim=1)
        return target - center


class ConditionalEDMDiffusion(ConditionalVectorDiffusion):
    """EDM-preconditioned conditional denoiser with deterministic Heun sampling."""

    name = "conditional_edm_diffusion"

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 96,
        layers: int = 3,
        sigma_min: float = 0.01,
        sigma_max: float = 3.0,
        sigma_data: float = 0.7,
        p_mean: float = -0.8,
        p_std: float = 1.2,
        sample_steps: int = 24,
        rho: float = 7.0,
    ):
        super().__init__(q=q, dy=dy, hidden=hidden, layers=layers,
                         sigma_min=sigma_min, sigma_max=sigma_max)
        if sigma_data <= 0 or sample_steps < 4 or rho <= 0:
            raise ValueError("sigma_data/rho must be positive and sample_steps >= 4")
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sample_steps = int(sample_steps)
        self.rho = float(rho)

    def _sigma_tensor(self, y: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            return torch.full((*y.shape[:-1], 1), float(sigma), dtype=y.dtype, device=y.device)
        if sigma.ndim == 0:
            return sigma.expand(*y.shape[:-1], 1)
        return sigma if sigma.shape[-1] == 1 else sigma[..., None]

    def denoise(
        self, y_noisy: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor | float
    ) -> torch.Tensor:
        sigma = self._sigma_tensor(y_noisy, sigma)
        sd2 = self.sigma_data**2
        denom = torch.sqrt(sigma.square() + sd2)
        c_skip = sd2 / (sigma.square() + sd2)
        c_out = sigma * self.sigma_data / denom
        c_in = 1.0 / denom
        raw = self.network(
            torch.cat([h, c_in * y_noisy, self._sigma_features(sigma)], dim=-1)
        )
        return c_skip * y_noisy + c_out * raw

    def score(self, y_noisy: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        sigma_t = self._sigma_tensor(y_noisy, sigma)
        return (self.denoise(y_noisy, h, sigma_t) - y_noisy) / sigma_t.square().clamp_min(1e-8)

    def _edm_loss(
        self, h: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        noisy = y + sigma * noise
        prediction = self.denoise(noisy, h, sigma)
        weight = (sigma.square() + self.sigma_data**2) / (
            (sigma * self.sigma_data).square().clamp_min(1e-8)
        )
        return torch.mean(weight * (prediction - y).square())

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(
            self.p_mean + self.p_std * torch.randn((len(y), 1), dtype=y.dtype, device=y.device)
        ).clamp(self.sigma_min, self.sigma_max)
        return self._edm_loss(h, y, sigma, torch.randn_like(y))

    def validation_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(31847)
        normal = torch.randn((len(y), 1), generator=generator, dtype=y.dtype).to(y.device)
        sigma = torch.exp(self.p_mean + self.p_std * normal).clamp(
            self.sigma_min, self.sigma_max
        )
        noise = torch.randn(y.shape, generator=generator, dtype=y.dtype).to(y.device)
        return self._edm_loss(h, y, sigma, noise)

    def sample(
        self,
        h: torch.Tensor,
        n_samples: int,
        seed: int = 0,
        levels: int | None = None,
        **_: Any,
    ) -> torch.Tensor:
        steps = self.sample_steps if levels is None else max(4, int(levels))
        repeated = h[:, None, :].expand(len(h), n_samples, self.q).reshape(-1, self.q)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        value = self.sigma_max * torch.randn(
            (len(repeated), self.dy), generator=generator, dtype=h.dtype
        ).to(h.device)
        ramp = torch.linspace(0.0, 1.0, steps, dtype=h.dtype, device="cpu")
        schedule = (
            self.sigma_max ** (1.0 / self.rho)
            + ramp
            * (
                self.sigma_min ** (1.0 / self.rho)
                - self.sigma_max ** (1.0 / self.rho)
            )
        ).pow(self.rho).to(h.device)
        schedule = torch.cat([schedule, torch.zeros(1, dtype=h.dtype, device=h.device)])
        with torch.no_grad():
            for index in range(len(schedule) - 1):
                sigma, sigma_next = schedule[index], schedule[index + 1]
                denoised = self.denoise(value, repeated, sigma)
                derivative = (value - denoised) / sigma.clamp_min(1e-8)
                proposal = value + (sigma_next - sigma) * derivative
                if float(sigma_next) > 0:
                    next_denoised = self.denoise(proposal, repeated, sigma_next)
                    next_derivative = (proposal - next_denoised) / sigma_next.clamp_min(1e-8)
                    value = value + (sigma_next - sigma) * 0.5 * (
                        derivative + next_derivative
                    )
                else:
                    value = proposal
        return value.reshape(len(h), n_samples, self.dy)


class GaussianAnchoredEDM(ConditionalVectorDiffusion):
    """EDM denoiser preconditioned around a learned conditional Gaussian base."""

    name = "gaussian_anchored_edm"

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 96,
        layers: int = 3,
        sigma_min: float = 0.01,
        sigma_max: float = 3.0,
        p_mean: float = -0.8,
        p_std: float = 1.2,
        sample_steps: int = 24,
        rho: float = 7.0,
        base_nll_weight: float = 0.1,
    ):
        super().__init__(
            q=q, dy=dy, hidden=hidden, layers=layers,
            sigma_min=sigma_min, sigma_max=sigma_max,
        )
        if sample_steps < 4 or rho <= 0 or base_nll_weight <= 0:
            raise ValueError("invalid anchored EDM integration or base weight")
        self.sample_steps = int(sample_steps)
        self.rho = float(rho)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.base_nll_weight = float(base_nll_weight)
        self.base_network = ResidualMLP(q, 2 * dy, hidden, layers)

    def base_parameters(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = torch.chunk(self.base_network(h), 2, dim=-1)
        return mean, log_std.clamp(-5.0, 3.0)

    def base_log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.base_parameters(h)
        z = (y - mean) * torch.exp(-log_std)
        return (-0.5 * (z.square() + LOG_2PI) - log_std).sum(dim=-1)

    def _sigma_tensor(self, y: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            return torch.full((*y.shape[:-1], 1), float(sigma), dtype=y.dtype, device=y.device)
        if sigma.ndim == 0:
            return sigma.expand(*y.shape[:-1], 1)
        return sigma if sigma.shape[-1] == 1 else sigma[..., None]

    def denoise(
        self, y_noisy: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor | float
    ) -> torch.Tensor:
        sigma = self._sigma_tensor(y_noisy, sigma)
        mean, log_std = self.base_parameters(h)
        std = torch.exp(log_std)
        variance = std.square()
        denom = torch.sqrt(sigma.square() + variance)
        c_skip = variance / (sigma.square() + variance)
        c_out = sigma * std / denom
        residual = (y_noisy - mean) / denom
        raw = self.network(
            torch.cat([h, residual, self._sigma_features(sigma)], dim=-1)
        )
        return mean + c_skip * (y_noisy - mean) + c_out * raw

    def score(
        self, y_noisy: torch.Tensor, h: torch.Tensor, sigma: torch.Tensor | float
    ) -> torch.Tensor:
        sigma_t = self._sigma_tensor(y_noisy, sigma)
        return (self.denoise(y_noisy, h, sigma_t) - y_noisy) / sigma_t.square().clamp_min(1e-8)

    def _loss(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        *,
        sigma: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        mean, log_std = self.base_parameters(h)
        variance = torch.exp(2.0 * log_std)
        prediction = self.denoise(y + sigma * noise, h, sigma)
        weight = (sigma.square() + variance) / (
            sigma.square() * variance
        ).clamp_min(1e-8)
        edm = torch.mean(weight * (prediction - y).square())
        return edm + self.base_nll_weight * (-self.base_log_prob(y, h).mean())

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(
            self.p_mean + self.p_std * torch.randn((len(y), 1), dtype=y.dtype, device=y.device)
        ).clamp(self.sigma_min, self.sigma_max)
        return self._loss(h, y, sigma=sigma, noise=torch.randn_like(y))

    def validation_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(64219)
        normal = torch.randn((len(y), 1), generator=generator, dtype=y.dtype).to(y.device)
        sigma = torch.exp(self.p_mean + self.p_std * normal).clamp(
            self.sigma_min, self.sigma_max
        )
        noise = torch.randn(y.shape, generator=generator, dtype=y.dtype).to(y.device)
        return self._loss(h, y, sigma=sigma, noise=noise)

    def sample(
        self,
        h: torch.Tensor,
        n_samples: int,
        seed: int = 0,
        levels: int | None = None,
        **_: Any,
    ) -> torch.Tensor:
        steps = self.sample_steps if levels is None else max(4, int(levels))
        repeated = h[:, None, :].expand(len(h), n_samples, self.q).reshape(-1, self.q)
        mean, _ = self.base_parameters(repeated)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        value = mean + self.sigma_max * torch.randn(
            mean.shape, generator=generator, dtype=h.dtype
        ).to(h.device)
        ramp = torch.linspace(0.0, 1.0, steps, dtype=h.dtype, device="cpu")
        schedule = (
            self.sigma_max ** (1.0 / self.rho)
            + ramp
            * (
                self.sigma_min ** (1.0 / self.rho)
                - self.sigma_max ** (1.0 / self.rho)
            )
        ).pow(self.rho).to(h.device)
        schedule = torch.cat([schedule, torch.zeros(1, dtype=h.dtype, device=h.device)])
        with torch.no_grad():
            for index in range(len(schedule) - 1):
                sigma, sigma_next = schedule[index], schedule[index + 1]
                derivative = (value - self.denoise(value, repeated, sigma)) / sigma.clamp_min(1e-8)
                proposal = value + (sigma_next - sigma) * derivative
                if float(sigma_next) > 0:
                    next_derivative = (
                        proposal - self.denoise(proposal, repeated, sigma_next)
                    ) / sigma_next.clamp_min(1e-8)
                    value = value + 0.5 * (sigma_next - sigma) * (
                        derivative + next_derivative
                    )
                else:
                    value = proposal
        return value.reshape(len(h), n_samples, self.dy)


class ConditionalFlowMatching(ConditionalModel):
    """Conditional independent-coupling flow matching with Heun integration."""

    name = "conditional_flow_matching"
    capabilities = Capabilities(False, True, False)

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 96,
        layers: int = 3,
        sample_steps: int = 24,
    ):
        super().__init__()
        if sample_steps < 4:
            raise ValueError("sample_steps must be at least four")
        self.q, self.dy = int(q), int(dy)
        self.sample_steps = int(sample_steps)
        self.network = ResidualMLP(q + dy + 3, dy, hidden, layers)

    @staticmethod
    def _time_features(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)

    def velocity(self, value: torch.Tensor, h: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.full((*value.shape[:-1], 1), float(t), dtype=value.dtype, device=value.device)
        elif t.ndim == 0:
            t = t.expand(*value.shape[:-1], 1)
        elif t.shape[-1] != 1:
            t = t[..., None]
        return self.network(torch.cat([h, value, self._time_features(t)], dim=-1))

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        base = torch.randn_like(y)
        t = torch.rand((len(y), 1), dtype=y.dtype, device=y.device)
        interpolated = (1.0 - t) * base + t * y
        return F.mse_loss(self.velocity(interpolated, h, t), y - base)

    def validation_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(44017)
        base = torch.randn(y.shape, generator=generator, dtype=y.dtype).to(y.device)
        t = torch.rand((len(y), 1), generator=generator, dtype=y.dtype).to(y.device)
        return F.mse_loss(self.velocity((1.0 - t) * base + t * y, h, t), y - base)

    def sample(self, h: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        repeated = h[:, None, :].expand(len(h), n_samples, self.q).reshape(-1, self.q)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        value = torch.randn((len(repeated), self.dy), generator=generator, dtype=h.dtype).to(h.device)
        grid = torch.linspace(0.0, 1.0, self.sample_steps + 1, dtype=h.dtype, device=h.device)
        with torch.no_grad():
            for index in range(self.sample_steps):
                t0, t1 = grid[index], grid[index + 1]
                dt = t1 - t0
                velocity = self.velocity(value, repeated, t0)
                proposal = value + dt * velocity
                value = value + 0.5 * dt * (
                    velocity + self.velocity(proposal, repeated, t1)
                )
        return value.reshape(len(h), n_samples, self.dy)


class GaussianSourceFlowMatching(ConditionalModel):
    """Conditional flow matching that transports a learned Gaussian baseline.

    The source already captures conditional location and scale, leaving the
    vector field to learn non-Gaussian residual structure.  Training remains a
    simulation-free square regression and sampling preserves a deterministic
    common-base coupling across histories when the same seed is used.
    """

    name = "gaussian_source_flow_matching"
    capabilities = Capabilities(False, True, False)

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 96,
        layers: int = 3,
        sample_steps: int = 24,
        base_nll_weight: float = 0.25,
    ):
        super().__init__()
        if sample_steps < 4 or base_nll_weight <= 0:
            raise ValueError("sample_steps must be >=4 and base_nll_weight positive")
        self.q, self.dy = int(q), int(dy)
        self.sample_steps = int(sample_steps)
        self.base_nll_weight = float(base_nll_weight)
        self.base_network = ResidualMLP(q, 2 * dy, hidden, layers)
        self.velocity_network = ResidualMLP(q + dy + 3, dy, hidden, layers)

    @staticmethod
    def _time_features(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)

    def base_parameters(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = torch.chunk(self.base_network(h), 2, dim=-1)
        return mean, log_std.clamp(-5.0, 3.0)

    def base_log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.base_parameters(h)
        z = (y - mean) * torch.exp(-log_std)
        return (-0.5 * (z.square() + LOG_2PI) - log_std).sum(dim=-1)

    def velocity(self, value: torch.Tensor, h: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.full((*value.shape[:-1], 1), float(t), dtype=value.dtype, device=value.device)
        elif t.ndim == 0:
            t = t.expand(*value.shape[:-1], 1)
        elif t.shape[-1] != 1:
            t = t[..., None]
        return self.velocity_network(
            torch.cat([h, value, self._time_features(t)], dim=-1)
        )

    def _loss(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        *,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        mean, log_std = self.base_parameters(h)
        source = (mean + torch.exp(log_std) * noise).detach()
        interpolated = (1.0 - t) * source + t * y
        flow_loss = F.mse_loss(self.velocity(interpolated, h, t), y - source)
        return flow_loss + self.base_nll_weight * (-self.base_log_prob(y, h).mean())

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = torch.rand((len(y), 1), dtype=y.dtype, device=y.device)
        return self._loss(h, y, noise=torch.randn_like(y), t=t)

    def validation_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(52091)
        noise = torch.randn(y.shape, generator=generator, dtype=y.dtype).to(y.device)
        t = torch.rand((len(y), 1), generator=generator, dtype=y.dtype).to(y.device)
        return self._loss(h, y, noise=noise, t=t)

    def sample(self, h: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        repeated = h[:, None, :].expand(len(h), n_samples, self.q).reshape(-1, self.q)
        mean, log_std = self.base_parameters(repeated)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(mean.shape, generator=generator, dtype=h.dtype).to(h.device)
        value = mean + torch.exp(log_std) * noise
        grid = torch.linspace(0.0, 1.0, self.sample_steps + 1, dtype=h.dtype, device=h.device)
        with torch.no_grad():
            for index in range(self.sample_steps):
                t0, t1 = grid[index], grid[index + 1]
                dt = t1 - t0
                velocity = self.velocity(value, repeated, t0)
                proposal = value + dt * velocity
                value = value + 0.5 * dt * (
                    velocity + self.velocity(proposal, repeated, t1)
                )
        return value.reshape(len(h), n_samples, self.dy)


class BoundedEnergyRatio(ConditionalModel):
    """Tail-safe conditional energy tilt learned by density-ratio classification.

    The reference ``q(y|h)`` is a diagonal Gaussian and the learned law is

    ``p_theta(y|h) proportional to q_theta(y|h) exp(B tanh(f_theta(y,h)))``.

    The bounded tilt guarantees a finite conditional normalizer and bounded
    importance weights.  Equal-prior data-versus-reference classification
    estimates the log density ratio without ever subtracting noisy variances.
    Exact samples use rejection sampling from the Gaussian reference: a proposal
    is accepted when ``log(U) <= log_tilt(y, h) - B``.  Its history score is the
    history derivative of the unnormalized log density centered with samples
    from the tilted law, so the unknown normalizer cancels.
    """

    name = "bounded_energy_ratio"
    capabilities = Capabilities(
        normalized_density=False,
        sampler=True,
        history_tangent=True,
        response_score=True,
        tangent_requires_centering=True,
    )

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 64,
        layers: int = 2,
        tilt_bound: float = 3.0,
        base_nll_weight: float = 0.25,
        oversample: int = 6,
        centering_samples: int = 48,
        rejection_batch_size: int = 4096,
    ):
        super().__init__()
        if tilt_bound <= 0 or base_nll_weight <= 0:
            raise ValueError("tilt_bound and base_nll_weight must be positive")
        if oversample < 2 or centering_samples < 8:
            raise ValueError("oversample must be >=2 and centering_samples >=8")
        if rejection_batch_size < oversample:
            raise ValueError("rejection_batch_size must be >= oversample")
        self.q, self.dy = int(q), int(dy)
        self.tilt_bound = float(tilt_bound)
        self.base_nll_weight = float(base_nll_weight)
        # ``oversample`` is retained for configuration compatibility.  It now
        # controls the number of independent rejection attempts made per
        # pending output slot in one round; it is not a finite SIR pool.
        self.oversample = int(oversample)
        self.centering_samples = int(centering_samples)
        self.rejection_batch_size = int(rejection_batch_size)
        self._last_sampling_diagnostics: dict[str, Any] = {}
        self.base_network = ResidualMLP(q, 2 * dy, hidden, layers)
        self.tilt_network = ResidualMLP(q + dy, 1, hidden, layers)

    def base_parameters(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = torch.chunk(self.base_network(h), 2, dim=-1)
        return mean, log_std.clamp(-5.0, 3.0)

    def base_log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.base_parameters(h)
        z = (y - mean) * torch.exp(-log_std)
        return (-0.5 * (z.square() + LOG_2PI) - log_std).sum(dim=-1)

    def log_tilt(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        raw = self.tilt_network(torch.cat([h, y], dim=-1)).squeeze(-1)
        return self.tilt_bound * torch.tanh(raw)

    def log_unnormalized(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.base_log_prob(y, h) + self.log_tilt(y, h)

    def _base_draw(self, h: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.base_parameters(h)
        return mean + torch.exp(log_std) * noise

    def _ratio_loss(
        self, h: torch.Tensor, y: torch.Tensor, *, noise: torch.Tensor
    ) -> torch.Tensor:
        base_nll = -self.base_log_prob(y, h).mean()
        # The Gaussian reference is trained by its own proper score.  Detaching
        # negatives prevents the classifier from moving the reference merely to
        # make discrimination easier.
        negative = self._base_draw(h, noise).detach()
        positive_logit = self.log_tilt(y, h)
        negative_logit = self.log_tilt(negative, h)
        classification = 0.5 * (
            F.softplus(-positive_logit).mean() + F.softplus(negative_logit).mean()
        )
        return classification + self.base_nll_weight * base_nll

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._ratio_loss(h, y, noise=torch.randn_like(y))

    def validation_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(90317)
        noise = torch.randn(y.shape, generator=generator, dtype=y.dtype).to(y.device)
        return self._ratio_loss(h, y, noise=noise)

    def sample(self, h: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        """Draw exact iid samples from the bounded conditional energy law.

        For ``u = log_tilt`` and ``B = tilt_bound``, the target density is
        proportional to ``q exp(u)`` and ``q exp(B)`` is an envelope because
        ``u <= B``.  Each requested output slot therefore runs an independent
        rejection chain with acceptance probability ``exp(u - B)``.  Several
        attempts are evaluated together for throughput, but only proposals up
        to the first acceptance in a slot count as consumed.  The size of every
        materialized proposal tensor is bounded by ``rejection_batch_size``.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if len(h) == 0:
            self._last_sampling_diagnostics = {
                "accepted_samples": 0,
                "effective_proposals": 0,
                "generated_proposals": 0,
                "acceptance_rate": float("nan"),
                "theoretical_acceptance_lower_bound": math.exp(
                    -2.0 * self.tilt_bound
                ),
                "rounds": 0,
                "seed": int(seed),
            }
            return h.new_empty((0, n_samples, self.dy))

        attempts_per_slot = self.oversample
        slots_per_batch = max(1, self.rejection_batch_size // attempts_per_slot)
        requested = len(h) * n_samples
        slot_histories = torch.arange(len(h), device=h.device).repeat_interleave(
            n_samples
        )
        pending = torch.arange(requested, device=h.device)
        samples = h.new_empty((requested, self.dy))
        proposals_by_slot = torch.zeros(
            requested, dtype=torch.int64, device=h.device
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        generated_proposals = 0
        max_generated_batch = 0
        rounds = 0

        # Sampling is an inference operation.  Disabling autograd here also
        # bounds memory when rejection takes several rounds at a low acceptance
        # rate (which is still at least exp(-2B) in exact arithmetic).
        with torch.no_grad():
            while pending.numel():
                rounds += 1
                next_pending: list[torch.Tensor] = []
                for start in range(0, pending.numel(), slots_per_batch):
                    slot_ids = pending[start : start + slots_per_batch]
                    history_ids = slot_histories[slot_ids]
                    history = h[history_ids]
                    mean, log_std = self.base_parameters(history)
                    noise = torch.randn(
                        (len(slot_ids), attempts_per_slot, self.dy),
                        generator=generator,
                        dtype=h.dtype,
                    ).to(h.device)
                    candidates = (
                        mean[:, None, :]
                        + torch.exp(log_std)[:, None, :] * noise
                    )
                    repeated_h = history[:, None, :].expand(
                        len(slot_ids), attempts_per_slot, self.q
                    )
                    log_acceptance = self.log_tilt(
                        candidates.reshape(-1, self.dy),
                        repeated_h.reshape(-1, self.q),
                    ).reshape(len(slot_ids), attempts_per_slot) - self.tilt_bound
                    uniforms = torch.rand(
                        (len(slot_ids), attempts_per_slot),
                        generator=generator,
                        dtype=h.dtype,
                    ).to(h.device)
                    accepted = torch.log(
                        uniforms.clamp_min(torch.finfo(h.dtype).tiny)
                    ) <= log_acceptance
                    has_acceptance = accepted.any(dim=1)
                    first_acceptance = accepted.to(torch.int64).argmax(dim=1)
                    consumed = torch.where(
                        has_acceptance,
                        first_acceptance + 1,
                        torch.full_like(first_acceptance, attempts_per_slot),
                    )
                    proposals_by_slot[slot_ids] += consumed
                    generated_batch = len(slot_ids) * attempts_per_slot
                    generated_proposals += generated_batch
                    max_generated_batch = max(max_generated_batch, generated_batch)

                    accepted_slots = slot_ids[has_acceptance]
                    if accepted_slots.numel():
                        accepted_rows = torch.nonzero(
                            has_acceptance, as_tuple=False
                        ).squeeze(-1)
                        samples[accepted_slots] = candidates[
                            accepted_rows, first_acceptance[has_acceptance]
                        ]
                    if (~has_acceptance).any():
                        next_pending.append(slot_ids[~has_acceptance])
                pending = (
                    torch.cat(next_pending)
                    if next_pending
                    else pending.new_empty((0,))
                )

        effective_proposals = int(proposals_by_slot.sum().item())
        self._last_sampling_diagnostics = {
            "accepted_samples": requested,
            "effective_proposals": effective_proposals,
            "generated_proposals": generated_proposals,
            "acceptance_rate": requested / effective_proposals,
            "theoretical_acceptance_lower_bound": math.exp(
                -2.0 * self.tilt_bound
            ),
            "rounds": rounds,
            "max_proposals_for_one_sample": int(proposals_by_slot.max().item()),
            "attempts_per_slot_per_round": attempts_per_slot,
            "proposal_batch_size_cap": self.rejection_batch_size,
            "max_generated_proposals_in_one_batch": max_generated_batch,
            "seed": int(seed),
        }
        return samples.reshape(len(h), n_samples, self.dy)

    def sampling_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics from the most recent exact sampling call."""
        return dict(self._last_sampling_diagnostics)

    def response_score(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y_req = y.detach().clone().requires_grad_(True)
        return torch.autograd.grad(
            self.log_unnormalized(y_req, h.detach()).sum(), y_req
        )[0]

    def _uncentered_history_score(
        self, y: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        h_req = h.detach().clone().requires_grad_(True)
        return torch.autograd.grad(
            self.log_unnormalized(y.detach(), h_req).sum(), h_req
        )[0]

    def history_tangent(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        observed = self._uncentered_history_score(y, h)
        with torch.no_grad():
            draws = self.sample(h.detach(), self.centering_samples, seed=73021)
        repeated_h = h[:, None, :].expand(
            len(h), self.centering_samples, self.q
        ).reshape(-1, self.q)
        centered = self._uncentered_history_score(
            draws.reshape(-1, self.dy), repeated_h
        ).reshape(len(h), self.centering_samples, self.q).mean(dim=1)
        return observed - centered


def build_model(name: str, q: int, dy: int, params: Mapping[str, Any] | None = None) -> ConditionalModel:
    params = dict(params or {})
    # Imported lazily because the stochastic-interpolant module reuses the
    # shared ConditionalModel/ResidualMLP primitives defined in this module.
    from .stochastic_interpolant import ConditionalPointSourceStochasticInterpolant

    factories = {
        "mean_mlp": MeanMLP,
        "heteroscedastic_gaussian": HeteroscedasticGaussian,
        "constrained_gaussian_dsm": ConstrainedGaussianDSM,
        "conditional_affine_flow": ConditionalAffineFlow,
        "autoregressive_mdn": AutoregressiveMixtureDensity,
        "autoregressive_transformer": AutoregressiveTransformerDensity,
        "ratio_critic": RatioCritic,
        "conditional_vector_diffusion": ConditionalVectorDiffusion,
        "conditional_edm_diffusion": ConditionalEDMDiffusion,
        "gaussian_anchored_edm": GaussianAnchoredEDM,
        "conditional_flow_matching": ConditionalFlowMatching,
        "gaussian_source_flow_matching": GaussianSourceFlowMatching,
        "bounded_energy_ratio": BoundedEnergyRatio,
        "conditional_point_source_stochastic_interpolant": ConditionalPointSourceStochasticInterpolant,
    }
    if name not in factories:
        raise KeyError(f"unknown model {name!r}; expected one of {sorted(factories)}")
    return factories[name](q=q, dy=dy, **params)


def history_score_matching_penalty(
    model: ConditionalModel,
    h: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Hutchinson Hyvarinen loss for the joint history score.

    The focused comparison uses standard-normal history DGPs. In standardized
    training coordinates the history marginal score is therefore approximately
    ``-h``. A normalized conditional model supplies the other joint-score term,
    ``grad_h log q(y | h)``. This penalty uses observed pairs only and never
    consumes the synthetic oracle tangent.
    """
    if not model.capabilities.normalized_density:
        raise ValueError("history score matching requires a normalized density")
    # The fused CPU attention kernel has no double-backward implementation.
    # Force the mathematically equivalent reference kernel only inside this
    # second-order regularizer; ordinary likelihood fitting keeps the fast path.
    with sdpa_kernel(SDPBackend.MATH):
        h_req = h.detach().clone().requires_grad_(True)
        conditional_log_prob = model.log_prob(y.detach(), h_req)
        conditional_score = torch.autograd.grad(
            conditional_log_prob.sum(), h_req, create_graph=True
        )[0]
        joint_score = conditional_score - h_req
        probe = torch.empty_like(h_req).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        projected = torch.sum(joint_score * probe)
        jacobian_transpose_probe = torch.autograd.grad(
            projected, h_req, create_graph=True
        )[0]
        divergence = torch.sum(jacobian_transpose_probe * probe, dim=-1)
        squared = 0.5 * torch.sum(joint_score.square(), dim=-1)
        return torch.mean(squared + divergence) / float(h.shape[-1])


def fit_model(
    model: ConditionalModel,
    train_h: np.ndarray,
    train_y: np.ndarray,
    validation_h: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
    device: str = "auto",
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 10,
    patience: int = 5,
    gradient_clip: float = 1.0,
    history_score_matching_weight: float = 0.0,
    input_noise_std: float = 0.0,
    input_noise_mask: np.ndarray | None = None,
    input_noise_draws: int = 0,
) -> FitTrace:
    """Fit by the model's native objective with validation-only checkpointing."""
    if len(train_h) != len(train_y) or len(validation_h) != len(validation_y):
        raise ValueError("history/response row counts differ")
    if len(train_h) < 2 or len(validation_h) < 2:
        raise ValueError("training and validation splits need at least two rows")
    if (
        not math.isfinite(float(history_score_matching_weight))
        or history_score_matching_weight < 0
    ):
        raise ValueError("history_score_matching_weight must be finite and nonnegative")
    if history_score_matching_weight and not model.capabilities.normalized_density:
        raise ValueError("history score matching is only defined for normalized models")
    if not math.isfinite(float(input_noise_std)) or input_noise_std < 0:
        raise ValueError("input_noise_std must be finite and nonnegative")
    if input_noise_draws < 0:
        raise ValueError("input_noise_draws must be nonnegative")
    if input_noise_std > 0 and input_noise_draws < 1:
        raise ValueError("positive input noise requires at least one noise draw")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    target_device = resolve_device(device)
    model.to(target_device)
    h_train = torch.as_tensor(train_h, dtype=torch.float32, device=target_device)
    y_train = torch.as_tensor(train_y, dtype=torch.float32, device=target_device)
    h_validation = torch.as_tensor(validation_h, dtype=torch.float32, device=target_device)
    y_validation = torch.as_tensor(validation_y, dtype=torch.float32, device=target_device)
    noise_mask = None
    if input_noise_mask is not None:
        mask = np.asarray(input_noise_mask, dtype=np.float32)
        if mask.shape != (train_h.shape[1],):
            raise ValueError("input_noise_mask must match the flattened history width")
        noise_mask = torch.as_tensor(mask, dtype=torch.float32, device=target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        permutation = rng.permutation(len(h_train))
        batch_losses = []
        for start in range(0, len(permutation), batch_size):
            index = torch.as_tensor(
                permutation[start : start + batch_size], dtype=torch.long, device=target_device
            )
            if len(index) < 2 and isinstance(model, RatioCritic):
                continue
            optimizer.zero_grad(set_to_none=True)
            batch_h = h_train[index]
            batch_y = y_train[index]
            losses = [model.native_loss(batch_h, batch_y)]
            if input_noise_std > 0:
                for _ in range(input_noise_draws):
                    noise = torch.randn_like(batch_h) * float(input_noise_std)
                    if noise_mask is not None:
                        noise = noise * noise_mask
                    losses.append(model.native_loss(batch_h + noise, batch_y))
            loss = torch.stack(losses).mean()
            if history_score_matching_weight:
                loss = loss + float(
                    history_score_matching_weight
                ) * history_score_matching_penalty(model, batch_h, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {model.name} loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        if not batch_losses:
            raise RuntimeError("no training batches were evaluated")
        model.eval()
        with torch.no_grad():
            validation_loss = float(model.validation_loss(h_validation, y_validation).cpu())
        train_losses.append(float(np.mean(batch_losses)))
        validation_losses.append(validation_loss)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    model.to(target_device)
    model.eval()
    if isinstance(model, MeanMLP):
        model.finalize_residual(h_train, y_train)
    wall = time.perf_counter() - started
    return FitTrace(
        train_loss=train_losses,
        validation_loss=validation_losses,
        best_epoch=best_epoch,
        stopped_epoch=len(train_losses) - 1,
        wall_seconds=wall,
        parameter_count=model.parameter_count,
        device=str(target_device),
    )
