"""Leakage-resistant SBTG joint-score models.

This is an isolated benchmark implementation. It preserves SBTG's energy-based
joint score while adding episode-held-out validation, early stopping, dropout,
weight decay, exact corruption kernels, and continuous score statistics. It does
not pretend that the joint squared-score covariance is SID's conditional gain.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..capabilities import Capabilities, ResourceProfile, UnsupportedCapability
from ..noise_kernels import GaussianNoise, NoiseKernel, StudentTNoise
from ..schema import Prediction, SupervisedData
from .base import Estimator


def _network(input_dim: int, output_dim: int, hidden: int, layers: int, dropout: float):
    modules: list[nn.Module] = []
    width = input_dim
    for _ in range(layers):
        modules.extend([nn.Linear(width, hidden), nn.SiLU()])
        if dropout:
            modules.append(nn.Dropout(dropout))
        width = hidden
    modules.append(nn.Linear(width, output_dim))
    return nn.Sequential(*modules)


class _SBTGEnergy(nn.Module):
    def __init__(
        self,
        n: int,
        *,
        model_type: str,
        hidden: int,
        layers: int,
        feature_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.n = n
        self.model_type = model_type
        self.g0 = _network(n, 1, hidden, layers, dropout)
        self.g1 = _network(n, 1, hidden, layers, dropout)
        if model_type == "linear":
            self.W = nn.Parameter(torch.empty(n, n))
            nn.init.uniform_(self.W, -0.05, 0.05)
            self.phi = self.psi = None
        elif model_type == "feature_bilinear":
            self.W = nn.Parameter(torch.empty(feature_dim, feature_dim))
            nn.init.uniform_(self.W, -0.05, 0.05)
            self.phi = _network(n, feature_dim, hidden, layers, dropout)
            self.psi = _network(n, feature_dim, hidden, layers, dropout)
        else:
            raise ValueError("model_type must be linear or feature_bilinear")

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        x0, x1 = torch.split(z, self.n, dim=-1)
        energy = self.g0(x0).squeeze(-1) + self.g1(x1).squeeze(-1)
        if self.model_type == "linear":
            coupling = torch.einsum("bi,ij,bj->b", x1, self.W, x0)
        else:
            coupling = torch.einsum("bi,ij,bj->b", self.psi(x1), self.W, self.phi(x0))
        return energy + coupling

    def score(self, z: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not z.requires_grad:
            z = z.detach().clone().requires_grad_(True)
        energy = self.energy(z).sum()
        gradient = torch.autograd.grad(
            energy, z, create_graph=create_graph, retain_graph=create_graph
        )[0]
        return -gradient, z


@dataclass
class SBTGTrace:
    train_dsm: list[float]
    validation_dsm: list[float]
    best_epoch: int
    stopped_epoch: int


class SBTGJointScore(Estimator):
    name = "sbtg_joint_score"
    score_domain = "joint_consecutive_state"
    capabilities = Capabilities(
        predictive_distribution="unnormalized_score",
        effect_channels=("joint_score_cross_moment", "joint_squared_score_covariance"),
        graph_scores=True,
    )
    resource_profile = ResourceProfile(
        "accelerator_neural", threads=1, uses_accelerator=True, estimated_peak_gb=2.5
    )

    def __init__(
        self,
        *,
        model_type: str = "feature_bilinear",
        kernel: NoiseKernel | None = None,
        noise_scale: float = 0.2,
        noise_scales: tuple[float, ...] | list[float] | None = None,
        hidden: int = 64,
        layers: int = 2,
        feature_dim: int = 16,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        coupling_l1: float = 0.0,
        batch_size: int = 128,
        max_epochs: int = 200,
        patience: int = 20,
        min_delta: float = 1e-4,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.kernel = GaussianNoise() if kernel is None else kernel
        self.noise_scale = float(noise_scale)
        self.noise_scales = (
            (self.noise_scale,)
            if noise_scales is None
            else tuple(float(scale) for scale in noise_scales)
        )
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.feature_dim = int(feature_dim)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.coupling_l1 = float(coupling_l1)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.seed = int(seed)
        self.device_name = device
        if not self.noise_scales or any(scale <= 0 for scale in self.noise_scales):
            raise ValueError("all noise scales must be positive")

    @staticmethod
    def _joint(data: SupervisedData) -> np.ndarray:
        n = data.targets.shape[1]
        current_columns = []
        for source in range(n):
            columns = np.flatnonzero(data.source_index == source)
            if not len(columns):
                raise ValueError(f"history has no feature for source {source}")
            current_columns.append(columns[0])
        return np.column_stack([data.features[:, current_columns], data.targets])

    def _standardize_fit(self, train: SupervisedData):
        z = self._joint(train)
        self.z_mean_ = np.mean(z, axis=0)
        self.z_scale_ = np.std(z, axis=0)
        self.z_scale_[self.z_scale_ < 1e-8] = 1.0
        return (z - self.z_mean_) / self.z_scale_

    def _standardize(self, data: SupervisedData):
        return (self._joint(data) - self.z_mean_) / self.z_scale_

    def _corrupt(self, clean: np.ndarray, rng: np.random.Generator, scale: float):
        return self._corrupt_with(self.kernel, clean, rng, scale)

    @staticmethod
    def _corrupt_with(
        kernel: NoiseKernel,
        clean: np.ndarray,
        rng: np.random.Generator,
        scale: float,
    ):
        noisy, aux = kernel.corrupt(clean, scale, rng)
        target = kernel.conditional_score(noisy, clean, scale, aux)
        return noisy, target

    def _reference_kernel(self) -> NoiseKernel:
        return StudentTNoise(df=5.0) if isinstance(self.kernel, StudentTNoise) else GaussianNoise()

    def _fixed_reference_risks(
        self,
        data: SupervisedData,
        *,
        seed: int,
        scales: tuple[float, ...] = (0.1, 0.25, 0.5),
    ) -> tuple[str, dict[float, float]]:
        clean = self._standardize(data)
        kernel = self._reference_kernel()
        risks = {}
        self.model_.eval()
        rng = np.random.default_rng(seed)
        for scale in scales:
            noisy, target = self._corrupt_with(kernel, clean, rng, scale)
            noisy_t = torch.as_tensor(noisy, dtype=torch.float32, device=self.device_)
            target_t = torch.as_tensor(target, dtype=torch.float32, device=self.device_)
            with torch.enable_grad():
                score, _ = self.model_.score(noisy_t, create_graph=False)
            risks[scale] = float(
                scale**2 * torch.mean((score.detach() - target_t) ** 2).cpu()
            )
        label = "student_t_df5" if isinstance(kernel, StudentTNoise) else "gaussian"
        return label, risks

    def selection_dsm_risk(self, data: SupervisedData, seed: int = 0) -> float:
        _, risks = self._fixed_reference_risks(data, seed=seed)
        return float(np.mean(list(risks.values())))

    def dsm_reference_metrics(self, data: SupervisedData, seed: int = 0) -> dict[str, float]:
        label, risks = self._fixed_reference_risks(data, seed=seed)
        result = {
            f"fixed_reference.{label}.scale_{scale:g}": value
            for scale, value in risks.items()
        }
        result[f"fixed_reference.{label}.ladder_mean"] = float(
            np.mean(list(risks.values()))
        )
        return result

    def _loss(self, clean: np.ndarray, rng: np.random.Generator, training: bool) -> torch.Tensor:
        losses = []
        for scale in self.noise_scales:
            noisy, target = self._corrupt(clean, rng, scale)
            noisy_t = torch.as_tensor(noisy, dtype=torch.float32, device=self.device_)
            target_t = torch.as_tensor(target, dtype=torch.float32, device=self.device_)
            with torch.enable_grad():
                score, _ = self.model_.score(noisy_t, create_graph=training)
                losses.append(scale**2 * torch.mean((score - target_t) ** 2))
        loss = torch.stack(losses).mean()
        if training and self.coupling_l1:
            loss = loss + self.coupling_l1 * torch.mean(torch.abs(self.model_.W))
        return loss

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        if validation is None or not len(validation.targets):
            raise ValueError("SBTGJointScore requires held-out validation episodes")
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.device_ = torch.device(self.device_name)
        z_train = self._standardize_fit(train)
        z_validation = self._standardize(validation)
        self.n_ = train.targets.shape[1]
        self.model_ = _SBTGEnergy(
            self.n_,
            model_type=self.model_type,
            hidden=self.hidden,
            layers=self.layers,
            feature_dim=self.feature_dim,
            dropout=self.dropout,
        ).to(self.device_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        rng = np.random.default_rng(self.seed)
        best = float("inf")
        best_state = copy.deepcopy(self.model_.state_dict())
        best_epoch = 0
        wait = 0
        train_trace: list[float] = []
        validation_trace: list[float] = []
        for epoch in range(self.max_epochs):
            self.model_.train()
            order = rng.permutation(len(z_train))
            losses = []
            for start in range(0, len(order), self.batch_size):
                batch = z_train[order[start : start + self.batch_size]]
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(batch, rng, training=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite SBTG loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.model_.eval()
            validation_loss = float(
                self._loss(
                    z_validation,
                    np.random.default_rng(self.seed + 999_983),
                    training=False,
                ).detach().cpu()
            )
            train_trace.append(float(np.mean(losses)))
            validation_trace.append(validation_loss)
            if validation_loss < best - self.min_delta:
                best = validation_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
            if wait >= self.patience:
                break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.trace_ = SBTGTrace(train_trace, validation_trace, best_epoch, epoch)
        self.channels_ = self._score_channels(z_validation)
        return self

    def _scores(self, z: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        batch_size = self.batch_size if batch_size is None else batch_size
        scores = []
        self.model_.eval()
        for start in range(0, len(z), batch_size):
            tensor = torch.as_tensor(
                z[start : start + batch_size], dtype=torch.float32, device=self.device_
            )
            with torch.enable_grad():
                score, _ = self.model_.score(tensor, create_graph=False)
            scores.append(score.detach().cpu().numpy())
        return np.concatenate(scores)

    def _interaction_hessian(self, z: np.ndarray, max_rows: int = 128) -> np.ndarray:
        if len(z) > max_rows:
            z = z[np.linspace(0, len(z) - 1, max_rows, dtype=int)]
        tensor = torch.as_tensor(z, dtype=torch.float32, device=self.device_).requires_grad_(True)
        with torch.enable_grad():
            score, differentiable_z = self.model_.score(tensor, create_graph=True)
            result = np.zeros((self.n_, self.n_))
            for target in range(self.n_):
                gradient = torch.autograd.grad(
                    score[:, self.n_ + target].sum(),
                    differentiable_z,
                    retain_graph=True,
                )[0]
                result[target] = gradient[:, : self.n_].detach().cpu().numpy().mean(axis=0)
        return result

    def _score_channels(self, z: np.ndarray):
        score = self._scores(z)
        s0, s1 = score[:, : self.n_], score[:, self.n_ :]
        cross = np.einsum("tj,ti->tji", s1, s0).mean(axis=0)
        squared_covariance = np.einsum(
            "tj,ti->ji",
            s1**2 - np.mean(s1**2, axis=0),
            s0**2 - np.mean(s0**2, axis=0),
        ) / max(1, len(score) - 1)
        interaction = self._interaction_hessian(z)
        channels = {
            "joint_score_cross_moment": cross,
            "joint_squared_score_covariance": squared_covariance,
            "joint_score_interaction_hessian": interaction,
        }
        if self.model_type == "linear":
            channels["energy_coupling_parameter"] = self.model_.W.detach().cpu().numpy()
        return channels

    def dsm_risk(self, data: SupervisedData, seed: int = 0) -> float:
        z = self._standardize(data)
        self.model_.eval()
        return float(self._loss(z, np.random.default_rng(seed), training=False).detach().cpu())

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        del data, n_samples
        raise UnsupportedCapability(self.name, "normalized_predictive_distribution")

    def metadata(self):
        result = {
            "model_type": self.model_type,
            "kernel": self.kernel.name,
            "noise_scale": self.noise_scale,
            "noise_scales": self.noise_scales,
            "ladder_objective": "mean of scale-squared DSM risks",
            "score_domain": self.score_domain,
            "hidden": self.hidden,
            "layers": self.layers,
            "feature_dim": self.feature_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "coupling_l1": self.coupling_l1,
            "best_epoch": self.trace_.best_epoch,
            "stopped_epoch": self.trace_.stopped_epoch,
            "best_validation_dsm": min(self.trace_.validation_dsm),
            "orientation": "[target, source]",
            "warning": "joint squared-score covariance is not conditional log-variance gain",
        }
        if isinstance(self.kernel, StudentTNoise):
            result["kernel_df"] = self.kernel.df
        return result
