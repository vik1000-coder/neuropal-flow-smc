"""Matched conditional Gaussian and finite-variance elliptical Student-t laws.

Both families parameterize the *covariance* as diag(exp(2s)) + FF', so
Student-t degrees of freedom change shape without silently changing variance.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from history_tangent_benchmark.models import Capabilities, ConditionalModel, ResidualMLP


class ConditionalElliptical(ConditionalModel):
    capabilities = Capabilities(True, True, False)

    def __init__(self, q: int, dy: int, hidden: int = 128, layers: int = 4,
                 rank: int = 8, student: bool = False, df_min: float = 2.1,
                 df_max: float = 50.0, df_initial: float = 8.0):
        super().__init__()
        if not 0 <= rank <= dy:
            raise ValueError("rank must be between zero and response dimension")
        if not 2 < df_min < df_initial < df_max:
            raise ValueError("Student-t bounds must guarantee finite covariance")
        self.q, self.dy, self.rank = q, dy, rank
        self.student, self.df_min, self.df_max = student, df_min, df_max
        self.name = "structure_student_t" if student else "structure_gaussian"
        self.network = ResidualMLP(q, dy * (2 + rank), hidden, layers)
        if student:
            probability = (df_initial - df_min) / (df_max - df_min)
            self.raw_df = nn.Parameter(torch.tensor(math.log(probability / (1 - probability))))

    @property
    def degrees_of_freedom(self) -> torch.Tensor:
        if not self.student:
            raise AttributeError("Gaussian has no degrees-of-freedom parameter")
        return self.df_min + (self.df_max - self.df_min) * torch.sigmoid(self.raw_df)

    def parameters_at(self, h: torch.Tensor):
        raw = self.network(h)
        mean = raw[:, :self.dy]
        log_std = raw[:, self.dy:2 * self.dy].clamp(-5.0, 2.5)
        factor = raw[:, 2 * self.dy:].reshape(len(h), self.dy, self.rank)
        if self.rank:
            factor = factor / math.sqrt(self.rank)
        return mean, log_std, factor

    def covariance_terms(self, y: torch.Tensor, h: torch.Tensor):
        mean, log_std, factor = self.parameters_at(h)
        inv_diag = torch.exp(-2 * log_std)
        residual = y - mean
        quadratic = (residual.square() * inv_diag).sum(-1)
        logdet = 2 * log_std.sum(-1)
        if self.rank:
            weighted_factor = factor * inv_diag[..., None]
            eye = torch.eye(self.rank, dtype=y.dtype, device=y.device)[None]
            chol = torch.linalg.cholesky(eye + factor.transpose(1, 2) @ weighted_factor)
            projection = factor.transpose(1, 2) @ (residual * inv_diag)[..., None]
            correction = torch.cholesky_solve(projection, chol)
            quadratic = quadratic - (projection * correction).sum((1, 2))
            logdet = logdet + 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        return quadratic.clamp_min(0), logdet

    def log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        quadratic, logdet = self.covariance_terms(y, h)
        if not self.student:
            return -0.5 * (self.dy * math.log(2 * math.pi) + logdet + quadratic)
        nu = self.degrees_of_freedom
        # The scale matrix of t_nu is (nu-2)/nu times the declared covariance.
        return (torch.lgamma((nu + self.dy) / 2) - torch.lgamma(nu / 2)
                - 0.5 * (self.dy * torch.log((nu - 2) * math.pi) + logdet)
                - 0.5 * (nu + self.dy) * torch.log1p(quadratic / (nu - 2)))

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(y, h).mean()

    def sample(self, h: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        mean, log_std, factor = self.parameters_at(h)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((len(h), n_samples, self.dy), generator=generator,
                            dtype=h.dtype).to(h.device)
        residual = torch.exp(log_std)[:, None] * noise
        if self.rank:
            lowrank = torch.randn((len(h), n_samples, self.rank), generator=generator,
                                  dtype=h.dtype).to(h.device)
            residual = residual + torch.einsum("bkr,bdr->bkd", lowrank, factor)
        if self.student:
            nu = float(self.degrees_of_freedom.detach().cpu())
            rng = np.random.default_rng(np.random.SeedSequence([seed, 9173]))
            chi_square = rng.chisquare(nu, size=(len(h), n_samples, 1))
            multiplier = torch.as_tensor(np.sqrt((nu - 2) / chi_square),
                                         dtype=h.dtype, device=h.device)
            # One common multiplier per joint sample: elliptical multivariate t.
            residual = residual * multiplier
        return mean[:, None] + residual
