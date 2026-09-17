"""Conditional point-source stochastic interpolant for probabilistic forecasting.

This module implements the point-source construction used in the focused
stochastic-interpolant experiment described by Chen et al. (ICML 2024).  It is
deliberately separate from :mod:`history_tangent_benchmark.models` so that the
experiment can be tested before it is admitted to the benchmark registry.

For a conditional-mean anchor ``a(h)``, response ``y``, and standard Gaussian
``xi``, the training path and regression target are

``I_t = (1-t) a(h) + beta(t) y + epsilon (1-t) sqrt(t) xi``

and

``R_t = -a(h) + beta_dot(t) y - epsilon sqrt(t) xi``.

The learned field is sampled with the native Euler--Maruyama SDE

``dX_t = b_theta(X_t, t, h) dt + epsilon (1-t) dW_t``, ``X_0 = a(h)``.

The default protocol fixes ``epsilon=1`` and compares only ``beta(t)=t`` with
``beta(t)=t**2``.  The latter removes the nonzero response impulse at ``t=0``.
This is a sampler backend, not a normalized density or a history-tangent
estimator.
"""
from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .models import Capabilities, ConditionalModel, LOG_2PI, ResidualMLP


BetaSchedule = Literal["linear", "squared"]


class LearnedGaussianMeanAnchor(nn.Module):
    """Diagonal-Gaussian model whose conditional mean anchors the interpolant.

    The scale is used only to train the anchor with a proper conditional
    Gaussian log score.  The stochastic interpolant itself starts from the
    resulting conditional mean, not from a draw from this Gaussian.
    """

    def __init__(self, q: int, dy: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.q, self.dy = int(q), int(dy)
        self.network = ResidualMLP(self.q, 2 * self.dy, hidden, layers)

    def parameters_at(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = torch.chunk(self.network(h), 2, dim=-1)
        return mean, log_std.clamp(-5.0, 3.0)

    def mean(self, h: torch.Tensor) -> torch.Tensor:
        return self.parameters_at(h)[0]

    def log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.parameters_at(h)
        standardized = (y - mean) * torch.exp(-log_std)
        return (-0.5 * (standardized.square() + LOG_2PI) - log_std).sum(dim=-1)

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(y, h).mean()


class ConditionalPointSourceStochasticInterpolant(ConditionalModel):
    """Conditional point-source stochastic-interpolant drift estimator.

    Parameters
    ----------
    anchor:
        Optional module supplying a conditional mean.  The adapter recognizes,
        in order, ``anchor.mean(h)``, ``anchor.parameters_at(h)[0]``, or
        ``anchor(h)``.  This permits a prefit
        :class:`~history_tangent_benchmark.models.HeteroscedasticGaussian` to be
        supplied without a wrapper.
    freeze_anchor:
        When ``None`` (the default), a user-supplied anchor is frozen and the
        internally created Gaussian anchor is learned.  Set explicitly to
        override that behavior.  The drift regression always sees a detached
        anchor; a learned anchor is optimized only through its own proper loss.
    anchor_loss_weight:
        Weight on the learned anchor objective.  An anchor exposing
        ``native_loss`` uses that objective; otherwise conditional-mean MSE is
        used.  The term is omitted for a frozen anchor.
    """

    name = "conditional_point_source_stochastic_interpolant"
    capabilities = Capabilities(False, True, False)

    def __init__(
        self,
        q: int,
        dy: int,
        hidden: int = 96,
        layers: int = 3,
        *,
        beta_schedule: BetaSchedule = "linear",
        epsilon: float = 1.0,
        sample_steps: int = 128,
        anchor: nn.Module | None = None,
        freeze_anchor: bool | None = None,
        anchor_hidden: int = 64,
        anchor_layers: int = 2,
        anchor_loss_weight: float = 0.25,
    ):
        super().__init__()
        if beta_schedule not in {"linear", "squared"}:
            raise ValueError("beta_schedule must be 'linear' or 'squared'")
        if not math.isfinite(float(epsilon)) or not math.isclose(float(epsilon), 1.0):
            raise ValueError("epsilon is fixed at 1.0 in the registered comparison")
        if sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        if not math.isfinite(float(anchor_loss_weight)) or anchor_loss_weight < 0:
            raise ValueError("anchor_loss_weight must be finite and nonnegative")

        self.q, self.dy = int(q), int(dy)
        self.beta_schedule: BetaSchedule = beta_schedule
        self.epsilon = float(epsilon)
        self.sample_steps = int(sample_steps)
        self.anchor_loss_weight = float(anchor_loss_weight)

        supplied_anchor = anchor is not None
        if anchor is None:
            anchor = LearnedGaussianMeanAnchor(
                self.q, self.dy, hidden=anchor_hidden, layers=anchor_layers
            )
        self.anchor = anchor
        should_freeze = supplied_anchor if freeze_anchor is None else bool(freeze_anchor)
        self.set_anchor_trainable(not should_freeze)

        self.drift_network = ResidualMLP(self.q + self.dy + 3, self.dy, hidden, layers)

    def set_anchor_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the anchor parameters for two-stage experiments."""
        self.anchor_frozen = not bool(trainable)
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(bool(trainable))

    @staticmethod
    def _time_tensor(
        value: torch.Tensor, t: torch.Tensor | float
    ) -> torch.Tensor:
        if not torch.is_tensor(t):
            result = torch.full(
                (*value.shape[:-1], 1),
                float(t),
                dtype=value.dtype,
                device=value.device,
            )
        else:
            result = t.to(dtype=value.dtype, device=value.device)
            if result.ndim == 0:
                result = result.expand(*value.shape[:-1], 1)
            elif result.ndim == value.ndim - 1:
                result = result[..., None]
            elif result.shape[-1] != 1:
                raise ValueError("time tensor must have a singleton final dimension")
            try:
                result = torch.broadcast_to(result, (*value.shape[:-1], 1))
            except RuntimeError as error:
                raise ValueError("time tensor is not broadcastable to the batch") from error
        if not torch.isfinite(result).all():
            raise ValueError("time must be finite")
        return result

    def beta(self, t: torch.Tensor | float) -> torch.Tensor | float:
        """Evaluate the registered interpolation schedule."""
        return t if self.beta_schedule == "linear" else t * t

    def beta_derivative(self, t: torch.Tensor | float) -> torch.Tensor | float:
        """Evaluate the analytic derivative of the registered schedule."""
        if self.beta_schedule == "linear":
            return torch.ones_like(t) if torch.is_tensor(t) else 1.0
        return 2.0 * t

    @staticmethod
    def _time_features(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)

    def anchor_mean(self, h: torch.Tensor) -> torch.Tensor:
        """Return ``a(h)`` from a learned or externally supplied anchor."""
        mean_method = getattr(self.anchor, "mean", None)
        parameters_method = getattr(self.anchor, "parameters_at", None)
        if callable(mean_method):
            mean = mean_method(h)
        elif callable(parameters_method):
            parameters = parameters_method(h)
            if not isinstance(parameters, tuple) or not parameters:
                raise TypeError("anchor.parameters_at(h) must return a nonempty tuple")
            mean = parameters[0]
        else:
            mean = self.anchor(h)
        if not torch.is_tensor(mean) or mean.shape != (*h.shape[:-1], self.dy):
            raise ValueError(
                f"anchor mean must have shape {(*h.shape[:-1], self.dy)}, "
                f"got {getattr(mean, 'shape', None)}"
            )
        return mean

    def drift(
        self,
        value: torch.Tensor,
        h: torch.Tensor,
        t: torch.Tensor | float,
    ) -> torch.Tensor:
        """Evaluate the learned conditional stochastic-interpolant field."""
        if value.shape[:-1] != h.shape[:-1] or value.shape[-1] != self.dy:
            raise ValueError("value and history must have matching batch dimensions")
        if h.shape[-1] != self.q:
            raise ValueError(f"history final dimension must be {self.q}")
        time = self._time_tensor(value, t)
        return self.drift_network(
            torch.cat([h, value, self._time_features(time)], dim=-1)
        )

    def interpolant_and_target(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor | float,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the path state and Chen et al. regression target."""
        if h.ndim != 2 or y.ndim != 2 or noise.shape != y.shape or len(h) != len(y):
            raise ValueError("need rank-two matching history, response, and noise batches")
        if h.shape[-1] != self.q or y.shape[-1] != self.dy:
            raise ValueError("history or response dimension does not match the model")
        time = self._time_tensor(y, t)
        if torch.any((time < 0) | (time > 1)):
            raise ValueError("interpolation time must lie in [0, 1]")
        anchor = self.anchor_mean(h).detach()
        root_time = torch.sqrt(time.clamp_min(0.0))
        state = (
            (1.0 - time) * anchor
            + self.beta(time) * y
            + self.epsilon * (1.0 - time) * root_time * noise
        )
        target = (
            -anchor
            + self.beta_derivative(time) * y
            - self.epsilon * root_time * noise
        )
        return state, target

    def field_loss(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor | float,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Squared drift-regression loss for a declared time/noise batch."""
        state, target = self.interpolant_and_target(h, y, t, noise)
        return F.mse_loss(self.drift(state, h, t), target)

    def _anchor_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.anchor_frozen or self.anchor_loss_weight == 0:
            return y.new_zeros(())
        native_loss = getattr(self.anchor, "native_loss", None)
        if callable(native_loss):
            return native_loss(h, y)
        return F.mse_loss(self.anchor_mean(h), y)

    def _objective(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        *,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return self.field_loss(h, y, t, noise) + self.anchor_loss_weight * self._anchor_loss(
            h, y
        )

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        time = torch.rand((len(y), 1), dtype=y.dtype, device=y.device)
        return self._objective(h, y, t=time, noise=torch.randn_like(y))

    def validation_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(73421)
        time = torch.rand((len(y), 1), generator=generator, dtype=y.dtype).to(y.device)
        noise = torch.randn(y.shape, generator=generator, dtype=y.dtype).to(y.device)
        return self._objective(h, y, t=time, noise=noise)

    def sample(
        self,
        h: torch.Tensor,
        n_samples: int,
        seed: int = 0,
        *,
        steps: int | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Draw approximate samples with the native Euler--Maruyama SDE.

        The CPU-seeded noise stream is copied to the model device so the same
        seed is reproducible on a fixed backend.  Supplying the same seed and
        sample count across histories also provides a common-random-number
        coupling for finite contrasts.
        """
        if h.ndim != 2 or h.shape[-1] != self.q:
            raise ValueError(f"history must have shape [batch, {self.q}]")
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        integration_steps = self.sample_steps if steps is None else int(steps)
        if integration_steps < 1:
            raise ValueError("steps must be positive")

        repeated = h[:, None, :].expand(len(h), n_samples, self.q).reshape(-1, self.q)
        value = self.anchor_mean(repeated).detach().clone()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        grid = torch.linspace(
            0.0,
            1.0,
            integration_steps + 1,
            dtype=h.dtype,
            device=h.device,
        )
        with torch.no_grad():
            for index in range(integration_steps):
                time = grid[index]
                dt = grid[index + 1] - time
                noise = torch.randn(
                    value.shape, generator=generator, dtype=value.dtype, device="cpu"
                ).to(value.device)
                value = (
                    value
                    + dt * self.drift(value, repeated, time)
                    + self.epsilon * (1.0 - time) * torch.sqrt(dt) * noise
                )
        return value.reshape(len(h), n_samples, self.dy)


__all__ = [
    "BetaSchedule",
    "ConditionalPointSourceStochasticInterpolant",
    "LearnedGaussianMeanAnchor",
]
