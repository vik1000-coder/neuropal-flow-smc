from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
HTB_SRC = ROOT / "history_tangent_benchmark" / "src"
if str(HTB_SRC) not in sys.path:
    sys.path.insert(0, str(HTB_SRC))

from history_tangent_benchmark.models import (  # noqa: E402
    Capabilities,
    ConditionalModel,
    MeanMLP,
    ResidualMLP,
    build_model as build_head,
)


class FlatEncoder(nn.Module):
    def __init__(self, lag: int, channels: int, width: int, dropout: float):
        super().__init__()
        q = lag * channels
        self.net = nn.Sequential(
            nn.Linear(q, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        return self.net(flat)


class GRUEncoder(nn.Module):
    def __init__(self, lag: int, channels: int, width: int, dropout: float):
        super().__init__()
        self.lag, self.channels = lag, channels
        self.input = nn.Linear(channels, width)
        self.gru = nn.GRU(width, width, num_layers=2, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        seq = flat.reshape(len(flat), self.lag, self.channels)
        _, hidden = self.gru(F.silu(self.input(seq)))
        return self.norm(hidden[-1])


class TCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)
        self.norm = nn.GroupNorm(1, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (2 * self.dilation, 0))
        return x + self.dropout(F.silu(self.norm(self.conv(padded))))


class TCNEncoder(nn.Module):
    def __init__(self, lag: int, channels: int, width: int, dropout: float):
        super().__init__()
        self.lag, self.channels = lag, channels
        self.input = nn.Conv1d(channels, width, kernel_size=1)
        self.blocks = nn.Sequential(
            *[TCNBlock(width, d, dropout) for d in (1, 2, 4, 8) if d < lag]
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        seq = flat.reshape(len(flat), self.lag, self.channels).transpose(1, 2)
        encoded = self.blocks(self.input(seq))
        return self.norm(encoded[:, :, -1])


def _covering_dilations(lag: int) -> tuple[int, ...]:
    """Return causal TCN dilations whose receptive field covers ``lag`` frames."""
    dilations: list[int] = []
    receptive_field = 1
    dilation = 1
    while receptive_field < lag:
        dilations.append(dilation)
        receptive_field += 2 * dilation
        dilation *= 2
    return tuple(dilations)


class FullHistoryTCNEncoder(nn.Module):
    """Causal TCN guaranteed to see the complete declared history.

    The legacy TCN intentionally remains unchanged so old checkpoints continue
    to load exactly.  At L=80 its four blocks see only 31 frames; this encoder
    adds the 16- and 32-frame dilations and therefore covers all 80 frames.
    """

    def __init__(self, lag: int, channels: int, width: int, dropout: float):
        super().__init__()
        self.lag, self.channels = lag, channels
        self.dilations = _covering_dilations(lag)
        self.input = nn.Conv1d(channels, width, kernel_size=1)
        self.blocks = nn.Sequential(
            *[TCNBlock(width, dilation, dropout) for dilation in self.dilations]
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        seq = flat.reshape(len(flat), self.lag, self.channels).transpose(1, 2)
        encoded = self.blocks(self.input(seq))
        return self.norm(encoded[:, :, -1])


class MultiscaleTCNEncoder(nn.Module):
    """Full-history causal TCN with short, medium, and global state summaries."""

    def __init__(self, lag: int, channels: int, width: int, dropout: float):
        super().__init__()
        self.lag, self.channels = lag, channels
        self.input = nn.Conv1d(channels, width, kernel_size=1)
        self.blocks = nn.Sequential(
            *[TCNBlock(width, dilation, dropout) for dilation in _covering_dilations(lag)]
        )
        self.fuse = nn.Sequential(
            nn.Linear(5 * width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Dropout(dropout),
        )

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        seq = flat.reshape(len(flat), self.lag, self.channels).transpose(1, 2)
        encoded = self.blocks(self.input(seq))
        summaries = (
            encoded[:, :, -1],
            encoded[:, :, -min(4, self.lag) :].mean(dim=2),
            encoded[:, :, -min(16, self.lag) :].mean(dim=2),
            encoded.mean(dim=2),
            encoded.std(dim=2, unbiased=False),
        )
        return self.fuse(torch.cat(summaries, dim=1))


class StimulusPhaseTCNEncoder(nn.Module):
    """Full-history TCN with deterministic binary-stimulus phase summaries.

    The summaries do not add future information.  They make pulse exposure and
    time since the most recent onset/offset directly available instead of
    requiring the neural encoder to rediscover those quantities from an 80-bit
    stimulus channel.
    """

    def __init__(
        self, lag: int, channels: int, width: int, dropout: float,
        n_neurons: int | None = None,
    ):
        super().__init__()
        self.lag, self.channels = lag, channels
        if n_neurons is None or not (0 < n_neurons < channels):
            raise ValueError("StimulusPhaseTCNEncoder requires an explicit neural boundary")
        self.n_neurons = int(n_neurons)
        self.temporal = FullHistoryTCNEncoder(lag, channels, width, dropout)
        self.phase = nn.Sequential(
            nn.Linear(8, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Dropout(dropout),
        )

    def _phase_features(self, flat: torch.Tensor) -> torch.Tensor:
        seq = flat.reshape(len(flat), self.lag, self.channels)
        stimulus = seq[:, :, self.n_neurons :].amax(dim=2)
        changes = stimulus[:, 1:] - stimulus[:, :-1]
        positions = torch.arange(
            1, self.lag, dtype=stimulus.dtype, device=stimulus.device
        )[None]
        missing = torch.full_like(positions, -1.0)
        last_onset = torch.where(changes > 0.5, positions, missing).amax(dim=1)
        last_offset = torch.where(changes < -0.5, positions, missing).amax(dim=1)
        denominator = float(max(1, self.lag - 1))
        onset_seen = (last_onset >= 0).to(stimulus.dtype)
        offset_seen = (last_offset >= 0).to(stimulus.dtype)
        onset_age = torch.where(
            onset_seen > 0,
            (self.lag - 1 - last_onset) / denominator,
            torch.ones_like(last_onset),
        )
        offset_age = torch.where(
            offset_seen > 0,
            (self.lag - 1 - last_offset) / denominator,
            torch.ones_like(last_offset),
        )
        return torch.stack(
            (
                stimulus[:, -1],
                stimulus.mean(dim=1),
                stimulus[:, -min(4, self.lag) :].mean(dim=1),
                stimulus[:, -min(16, self.lag) :].mean(dim=1),
                onset_age,
                offset_age,
                onset_seen,
                offset_seen,
            ),
            dim=1,
        )

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(flat)
        phase = self.phase(self._phase_features(flat))
        return self.fuse(torch.cat((temporal, phase), dim=1))


class TransformerHistoryEncoder(nn.Module):
    def __init__(self, lag: int, channels: int, width: int, dropout: float):
        super().__init__()
        heads = 4 if width % 4 == 0 else 2
        self.lag, self.channels = lag, channels
        self.input = nn.Linear(channels, width)
        self.position = nn.Parameter(torch.randn(lag, width) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(width)

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        seq = flat.reshape(len(flat), self.lag, self.channels)
        z = self.input(seq) + self.position[None]
        return self.norm(self.encoder(z)[:, -1])


class EncodedConditionalModel(ConditionalModel):
    """Jointly train a temporal history encoder and a distributional head."""

    def __init__(self, encoder: nn.Module, head: ConditionalModel, name: str):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.name = name
        self.capabilities = head.capabilities

    def context(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history)

    def native_loss(self, history: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head.native_loss(self.context(history), y)

    def validation_loss(self, history: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head.validation_loss(self.context(history), y)

    def log_prob(self, y: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.head.log_prob(y, self.context(history))

    def sample(self, history: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        return self.head.sample(self.context(history), n_samples, seed=seed)

    def finalize(self, train_history: torch.Tensor, train_y: torch.Tensor) -> None:
        if isinstance(self.head, MeanMLP):
            self.head.finalize_residual(self.context(train_history), train_y)


class ConditionalLowRankGaussian(ConditionalModel):
    """Normalized Gaussian with history-dependent low-rank joint covariance."""

    name = "lowrank_gaussian"
    capabilities = Capabilities(True, True, False, response_score=True)

    def __init__(self, q: int, dy: int, hidden: int = 64, layers: int = 2, rank: int = 8):
        super().__init__()
        if rank < 1 or rank > dy:
            raise ValueError("low-rank covariance rank must lie in [1, dy]")
        self.q, self.dy, self.rank = int(q), int(dy), int(rank)
        self.network = ResidualMLP(q, 2 * dy + dy * rank, hidden, layers)

    def parameters_at(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.network(h)
        mean = raw[:, : self.dy]
        log_std = raw[:, self.dy : 2 * self.dy].clamp(-5.0, 2.5)
        factor = raw[:, 2 * self.dy :].reshape(-1, self.dy, self.rank) / math.sqrt(self.rank)
        return mean, log_std, factor

    def log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mean, log_std, factor = self.parameters_at(h)
        inv_diag = torch.exp(-2.0 * log_std)
        weighted_factor = factor * inv_diag[..., None]
        eye = torch.eye(self.rank, dtype=y.dtype, device=y.device)[None]
        middle = eye + factor.transpose(1, 2) @ weighted_factor
        chol = torch.linalg.cholesky(middle)
        residual = y - mean
        weighted_residual = residual * inv_diag
        projection = factor.transpose(1, 2) @ weighted_residual[..., None]
        correction = torch.cholesky_solve(projection, chol)
        quadratic = (residual * weighted_residual).sum(dim=1) - (
            projection.transpose(1, 2) @ correction
        ).flatten()
        logdet = 2.0 * log_std.sum(dim=1) + 2.0 * torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1)
        ).sum(dim=1)
        return -0.5 * (quadratic + logdet + self.dy * math.log(2.0 * math.pi))

    def native_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(y, h).mean()

    def sample(self, h: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        mean, log_std, factor = self.parameters_at(h)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        diagonal_noise = torch.randn(
            (len(h), n_samples, self.dy), generator=generator, dtype=h.dtype
        ).to(h.device)
        rank_noise = torch.randn(
            (len(h), n_samples, self.rank), generator=generator, dtype=h.dtype
        ).to(h.device)
        correlated = torch.einsum("bkr,bdr->bkd", rank_noise, factor)
        return mean[:, None] + torch.exp(log_std)[:, None] * diagonal_noise + correlated


def build_encoded_model(
    *,
    head_name: str,
    encoder_name: str,
    lag: int,
    channels: int,
    dy: int,
    width: int = 64,
    dropout: float = 0.0,
    head_params: dict[str, Any] | None = None,
) -> EncodedConditionalModel:
    encoders = {
        "flat": FlatEncoder,
        "gru": GRUEncoder,
        "tcn": TCNEncoder,
        "full_history_tcn": FullHistoryTCNEncoder,
        "multiscale_tcn": MultiscaleTCNEncoder,
        "stimulus_phase_tcn": StimulusPhaseTCNEncoder,
        "transformer": TransformerHistoryEncoder,
    }
    if encoder_name not in encoders:
        raise KeyError(f"unknown encoder {encoder_name}")
    if encoder_name == "stimulus_phase_tcn":
        encoder = StimulusPhaseTCNEncoder(
            lag, channels, width, dropout, n_neurons=dy
        )
    else:
        encoder = encoders[encoder_name](lag, channels, width, dropout)
    if head_name in {"structure_gaussian", "structure_student_t"}:
        from conditional_neural_benchmark.distribution_structure_models import ConditionalElliptical
        head = ConditionalElliptical(
            q=width, dy=dy, student=head_name == "structure_student_t", **(head_params or {})
        )
    elif head_name == "lowrank_gaussian":
        head = ConditionalLowRankGaussian(q=width, dy=dy, **(head_params or {}))
    else:
        head = build_head(head_name, q=width, dy=dy, params=head_params or {})
    return EncodedConditionalModel(
        encoder, head, name=f"{encoder_name}__{head_name}"
    )
