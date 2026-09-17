"""Regularized neural conditional-density and exact-kernel DSM adapters."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logsumexp, ndtr
from torch import nn

from ..capabilities import Capabilities, ResourceProfile, UnsupportedCapability
from ..metrics import gaussian_log_prob
from ..noise_kernels import GaussianNoise, NoiseKernel, StudentTNoise
from ..schema import Prediction, SupervisedData
from .base import Estimator, Standardizer, aggregate_features


def _device(name: str) -> torch.device:
    if name == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _mlp(input_dim: int, output_dim: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    modules: list[nn.Module] = []
    width = input_dim
    for _ in range(layers):
        modules.extend([nn.Linear(width, hidden), nn.SiLU()])
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        width = hidden
    modules.append(nn.Linear(width, output_dim))
    return nn.Sequential(*modules)


@dataclass
class TrainingTrace:
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int
    stopped_epoch: int


class _NeuralBase(Estimator):
    resource_profile = ResourceProfile(
        "accelerator_neural", threads=1, uses_accelerator=True, estimated_peak_gb=2.5
    )

    def __init__(
        self,
        *,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 200,
        patience: int = 20,
        min_delta: float = 1e-4,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.seed = int(seed)
        self.device_name = device
        if self.hidden < 1 or self.layers < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid network architecture")

    def _initialize(self, train: SupervisedData) -> tuple[np.ndarray, np.ndarray]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(1)
        self.device_ = _device(self.device_name)
        self.scaler_ = Standardizer.fit(train)
        return self.scaler_.x(train.features), self.scaler_.y(train.targets)

    def _fit_loop(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> TrainingTrace:
        optimizer = torch.optim.AdamW(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        rng = np.random.default_rng(self.seed)
        best_loss = float("inf")
        best_state = copy.deepcopy(self.network_.state_dict())
        best_epoch = 0
        wait = 0
        training_loss: list[float] = []
        validation_loss: list[float] = []
        for epoch in range(self.max_epochs):
            self.network_.train()
            permutation = rng.permutation(len(x_train))
            epoch_losses = []
            for start in range(0, len(permutation), self.batch_size):
                index = permutation[start : start + self.batch_size]
                xb = torch.as_tensor(x_train[index], dtype=torch.float32, device=self.device_)
                yb = torch.as_tensor(y_train[index], dtype=torch.float32, device=self.device_)
                optimizer.zero_grad(set_to_none=True)
                loss = self._training_loss(xb, yb, rng)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite neural loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network_.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.network_.eval()
            with torch.no_grad():
                xv = torch.as_tensor(x_validation, dtype=torch.float32, device=self.device_)
                yv = torch.as_tensor(y_validation, dtype=torch.float32, device=self.device_)
                val = float(self._validation_loss(xv, yv).detach().cpu())
            training_loss.append(float(np.mean(epoch_losses)))
            validation_loss.append(val)
            if val < best_loss - self.min_delta:
                best_loss = val
                best_state = copy.deepcopy(self.network_.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
            if wait >= self.patience:
                break
        self.network_.load_state_dict(best_state)
        self.network_.eval()
        return TrainingTrace(training_loss, validation_loss, best_epoch, epoch)

    def metadata(self):
        return {
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "seed": self.seed,
            "device": str(self.device_),
            "best_epoch": self.trace_.best_epoch,
            "stopped_epoch": self.trace_.stopped_epoch,
            "best_validation_loss": min(self.trace_.validation_loss),
        }


class GaussianMLP(_NeuralBase):
    """Conditional Gaussian network trained by NLL or Gaussian DSM.

    Validation and early stopping always use normalized held-out log score, even
    when the fitting objective is DSM. This keeps model selection method-agnostic.
    """

    name = "gaussian_mlp"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=(
            "conditional_mean_derivative",
            "conditional_log_variance_derivative",
            "conditional_tail_high_derivative",
            "conditional_shape_tail_derivative",
        ),
        graph_scores=True,
    )

    def __init__(
        self,
        *,
        objective: str = "nll",
        dsm_sigma: float = 0.2,
        tail_threshold: float = 0.35,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if objective not in {"nll", "dsm"}:
            raise ValueError("objective must be nll or dsm")
        if dsm_sigma <= 0:
            raise ValueError("dsm_sigma must be positive")
        self.objective = objective
        self.dsm_sigma = float(dsm_sigma)
        self.tail_threshold = float(tail_threshold)

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        if validation is None or not len(validation.targets):
            raise ValueError("GaussianMLP requires independent validation episodes")
        x, y = self._initialize(train)
        xv = self.scaler_.x(validation.features)
        yv = self.scaler_.y(validation.targets)
        self.n_targets_ = y.shape[1]
        self.tail_high_ = (
            self.tail_threshold - self.scaler_.y_mean
        ) / self.scaler_.y_scale
        self.network_ = _mlp(x.shape[1], 2 * self.n_targets_, self.hidden, self.layers, self.dropout)
        self.network_.to(self.device_)
        self.trace_ = self._fit_loop(x, y, xv, yv)
        self.source_index_ = train.source_index.copy()
        self.channels_ = self._readouts(train.features)
        return self

    def _parameters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network_(x)
        mean, logvariance = torch.chunk(output, 2, dim=1)
        return mean, torch.clamp(logvariance, -8.0, 5.0)

    def _nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean, logvariance = self._parameters(x)
        return 0.5 * torch.mean(logvariance + (y - mean) ** 2 * torch.exp(-logvariance))

    def _training_loss(self, x, y, rng):
        del rng
        if self.objective == "nll":
            return self._nll(x, y)
        noise = torch.randn_like(y)
        noisy = y + self.dsm_sigma * noise
        mean, logvariance = self._parameters(x)
        variance = torch.exp(logvariance)
        predicted_score = (mean - noisy) / (variance + self.dsm_sigma**2)
        target_score = (y - noisy) / self.dsm_sigma**2
        return self.dsm_sigma**2 * torch.mean((predicted_score - target_score) ** 2)

    def _validation_loss(self, x, y):
        return self._nll(x, y)

    def _readouts(self, features: np.ndarray, max_rows: int = 256):
        if len(features) > max_rows:
            index = np.linspace(0, len(features) - 1, max_rows, dtype=int)
            features = features[index]
        x = torch.as_tensor(
            self.scaler_.x(features), dtype=torch.float32, device=self.device_
        ).requires_grad_(True)
        mean, logvariance = self._parameters(x)
        threshold = torch.as_tensor(
            self.tail_high_, dtype=torch.float32, device=self.device_
        )[None, :]
        z = (threshold - mean) / torch.exp(0.5 * logvariance)
        tail_probability = 0.5 * torch.erfc(z / np.sqrt(2.0))
        mean_jac = np.zeros((self.n_targets_, features.shape[1]))
        logvar_jac = np.zeros_like(mean_jac)
        tail_jac = np.zeros_like(mean_jac)
        for target in range(self.n_targets_):
            grad_mean = torch.autograd.grad(mean[:, target].sum(), x, retain_graph=True)[0]
            grad_logvar = torch.autograd.grad(logvariance[:, target].sum(), x, retain_graph=True)[0]
            grad_tail = torch.autograd.grad(
                tail_probability[:, target].sum(), x, retain_graph=True
            )[0]
            mean_jac[target] = grad_mean.detach().cpu().numpy().mean(axis=0)
            logvar_jac[target] = grad_logvar.detach().cpu().numpy().mean(axis=0)
            tail_jac[target] = grad_tail.detach().cpu().numpy().mean(axis=0)
        mean_jac *= self.scaler_.y_scale[:, None] / self.scaler_.x_scale
        logvar_jac /= self.scaler_.x_scale
        tail_jac /= self.scaler_.x_scale
        shape_tail_feature = np.zeros_like(tail_jac)
        return {
            "conditional_mean_derivative_feature": mean_jac,
            "conditional_log_variance_derivative_feature": logvar_jac,
            "conditional_mean_derivative": aggregate_features(
                mean_jac, self.source_index_, self.n_targets_, signed=True
            ),
            "conditional_log_variance_derivative": aggregate_features(
                logvar_jac, self.source_index_, self.n_targets_, signed=True
            ),
            "conditional_log_variance_strength": aggregate_features(
                logvar_jac, self.source_index_, self.n_targets_, signed=False
            ),
            "conditional_tail_high_derivative_feature": tail_jac,
            "conditional_tail_high_derivative": aggregate_features(
                tail_jac, self.source_index_, self.n_targets_, signed=True
            ),
            "conditional_shape_tail_derivative_feature": shape_tail_feature,
            "conditional_shape_tail_derivative": aggregate_features(
                shape_tail_feature,
                self.source_index_,
                self.n_targets_,
                signed=True,
            ),
        }

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        with torch.no_grad():
            x = torch.as_tensor(
                self.scaler_.x(data.features), dtype=torch.float32, device=self.device_
            )
            mean_z, logvariance_z = self._parameters(x)
        mean = self.scaler_.inverse_mean(mean_z.cpu().numpy())
        variance = self.scaler_.inverse_variance(np.exp(logvariance_z.cpu().numpy()))
        samples = None
        if n_samples:
            rng = np.random.default_rng(self.seed + 1)
            samples = mean[:, None, :] + np.sqrt(variance[:, None, :]) * rng.standard_normal(
                (len(mean), n_samples, mean.shape[1])
            )
        return Prediction(
            mean=mean,
            variance=variance,
            log_prob=gaussian_log_prob(data.targets, mean, variance),
            cdf=ndtr((data.targets - mean) / np.sqrt(np.maximum(variance, 1e-10))),
            samples=samples,
            channels=self.channels_,
        )

    def metadata(self):
        result = super().metadata()
        result.update(
            {
                "objective": self.objective,
                "dsm_sigma": self.dsm_sigma,
                "tail_threshold_physical": self.tail_threshold,
            }
        )
        return result


class MixtureDensityMLP(_NeuralBase):
    """Factorized Gaussian-mixture conditional density with collapse diagnostics."""

    name = "mdn"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=(
            "conditional_mean_derivative",
            "conditional_log_variance_derivative",
            "conditional_tail_high_derivative",
            "conditional_shape_tail_derivative",
        ),
        graph_scores=True,
    )

    def __init__(
        self,
        *,
        components: int = 3,
        variance_floor: float = 1e-3,
        tail_threshold: float = 0.35,
        shape_tail_z: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if components < 1:
            raise ValueError("components must be positive")
        self.components = int(components)
        self.variance_floor = float(variance_floor)
        self.tail_threshold = float(tail_threshold)
        self.shape_tail_z = float(shape_tail_z)
        if self.shape_tail_z <= 0:
            raise ValueError("shape_tail_z must be positive")

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        if validation is None or not len(validation.targets):
            raise ValueError("MixtureDensityMLP requires independent validation episodes")
        x, y = self._initialize(train)
        xv = self.scaler_.x(validation.features)
        yv = self.scaler_.y(validation.targets)
        self.n_targets_ = y.shape[1]
        self.tail_high_ = (
            self.tail_threshold - self.scaler_.y_mean
        ) / self.scaler_.y_scale
        output = self.n_targets_ * self.components * 3
        self.network_ = _mlp(x.shape[1], output, self.hidden, self.layers, self.dropout)
        self.network_.to(self.device_)
        self.trace_ = self._fit_loop(x, y, xv, yv)
        self.source_index_ = train.source_index.copy()
        self.channels_ = self._readouts(train.features)
        return self

    def _parameters(self, x):
        raw = self.network_(x).reshape(-1, self.n_targets_, self.components, 3)
        logits = raw[..., 0]
        means = raw[..., 1]
        variances = torch.nn.functional.softplus(raw[..., 2]) + self.variance_floor
        return logits, means, variances

    def _log_prob_z(self, x, y):
        logits, means, variances = self._parameters(x)
        log_weights = torch.log_softmax(logits, dim=-1)
        log_component = -0.5 * (
            torch.log(2 * torch.pi * variances)
            + (y[..., None] - means) ** 2 / variances
        )
        return torch.logsumexp(log_weights + log_component, dim=-1)

    def _training_loss(self, x, y, rng):
        del rng
        return -torch.mean(self._log_prob_z(x, y))

    def _validation_loss(self, x, y):
        return -torch.mean(self._log_prob_z(x, y))

    def _moments_tensor(self, x):
        logits, means, variances = self._parameters(x)
        weights = torch.softmax(logits, dim=-1)
        mean = torch.sum(weights * means, dim=-1)
        second = torch.sum(weights * (variances + means**2), dim=-1)
        variance = torch.clamp(second - mean**2, min=1e-6)
        return mean, variance

    def _readouts(self, features: np.ndarray, max_rows: int = 192):
        if len(features) > max_rows:
            features = features[np.linspace(0, len(features) - 1, max_rows, dtype=int)]
        x = torch.as_tensor(
            self.scaler_.x(features), dtype=torch.float32, device=self.device_
        ).requires_grad_(True)
        mean, variance = self._moments_tensor(x)
        logits, component_mean, component_variance = self._parameters(x)
        weights = torch.softmax(logits, dim=-1)
        threshold = torch.as_tensor(
            self.tail_high_, dtype=torch.float32, device=self.device_
        )[None, :, None]
        z = (threshold - component_mean) / torch.sqrt(component_variance)
        component_survival = 0.5 * torch.erfc(z / np.sqrt(2.0))
        tail_probability = torch.sum(weights * component_survival, dim=-1)
        shape_threshold_high = (
            mean + self.shape_tail_z * torch.sqrt(variance)
        )[:, :, None]
        shape_threshold_low = (
            mean - self.shape_tail_z * torch.sqrt(variance)
        )[:, :, None]
        shape_high = 0.5 * torch.erfc(
            (shape_threshold_high - component_mean)
            / torch.sqrt(component_variance)
            / np.sqrt(2.0)
        )
        shape_low = 0.5 * torch.erfc(
            (component_mean - shape_threshold_low)
            / torch.sqrt(component_variance)
            / np.sqrt(2.0)
        )
        shape_tail_probability = torch.sum(
            weights * (shape_high + shape_low), dim=-1
        )
        mean_jac = np.zeros((self.n_targets_, features.shape[1]))
        logvar_jac = np.zeros_like(mean_jac)
        tail_jac = np.zeros_like(mean_jac)
        shape_tail_jac = np.zeros_like(mean_jac)
        for target in range(self.n_targets_):
            gm = torch.autograd.grad(mean[:, target].sum(), x, retain_graph=True)[0]
            gv = torch.autograd.grad(torch.log(variance[:, target]).sum(), x, retain_graph=True)[0]
            gt = torch.autograd.grad(
                tail_probability[:, target].sum(), x, retain_graph=True
            )[0]
            gs = torch.autograd.grad(
                shape_tail_probability[:, target].sum(), x, retain_graph=True
            )[0]
            mean_jac[target] = gm.detach().cpu().numpy().mean(axis=0)
            logvar_jac[target] = gv.detach().cpu().numpy().mean(axis=0)
            tail_jac[target] = gt.detach().cpu().numpy().mean(axis=0)
            shape_tail_jac[target] = gs.detach().cpu().numpy().mean(axis=0)
        mean_jac *= self.scaler_.y_scale[:, None] / self.scaler_.x_scale
        logvar_jac /= self.scaler_.x_scale
        tail_jac /= self.scaler_.x_scale
        shape_tail_jac /= self.scaler_.x_scale
        return {
            "conditional_mean_derivative_feature": mean_jac,
            "conditional_log_variance_derivative_feature": logvar_jac,
            "conditional_mean_derivative": aggregate_features(
                mean_jac, self.source_index_, self.n_targets_, signed=True
            ),
            "conditional_log_variance_derivative": aggregate_features(
                logvar_jac, self.source_index_, self.n_targets_, signed=True
            ),
            "conditional_tail_high_derivative_feature": tail_jac,
            "conditional_tail_high_derivative": aggregate_features(
                tail_jac, self.source_index_, self.n_targets_, signed=True
            ),
            "conditional_shape_tail_derivative_feature": shape_tail_jac,
            "conditional_shape_tail_derivative": aggregate_features(
                shape_tail_jac,
                self.source_index_,
                self.n_targets_,
                signed=True,
            ),
        }

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        with torch.no_grad():
            x = torch.as_tensor(
                self.scaler_.x(data.features), dtype=torch.float32, device=self.device_
            )
            logits_t, means_t, variances_t = self._parameters(x)
            log_prob_z = self._log_prob_z(
                x,
                torch.as_tensor(
                    self.scaler_.y(data.targets), dtype=torch.float32, device=self.device_
                ),
            )
            weights = torch.softmax(logits_t, dim=-1).cpu().numpy()
            means_z = means_t.cpu().numpy()
            variances_z = variances_t.cpu().numpy()
        mean_z = np.sum(weights * means_z, axis=-1)
        second_z = np.sum(weights * (variances_z + means_z**2), axis=-1)
        variance_z = np.maximum(second_z - mean_z**2, 1e-8)
        mean = self.scaler_.inverse_mean(mean_z)
        variance = self.scaler_.inverse_variance(variance_z)
        target_z = self.scaler_.y(data.targets)
        standardized = (target_z[..., None] - means_z) / np.sqrt(variances_z)
        cdf = np.sum(weights * ndtr(standardized), axis=-1)
        log_prob = log_prob_z.cpu().numpy() - np.log(self.scaler_.y_scale)[None, :]
        samples = None
        if n_samples:
            rng = np.random.default_rng(self.seed + 1)
            samples_z = np.empty((len(mean), n_samples, self.n_targets_))
            for row in range(len(mean)):
                for target in range(self.n_targets_):
                    component = rng.choice(self.components, size=n_samples, p=weights[row, target])
                    samples_z[row, :, target] = rng.normal(
                        means_z[row, target, component],
                        np.sqrt(variances_z[row, target, component]),
                    )
            samples = samples_z * self.scaler_.y_scale[None, None, :] + self.scaler_.y_mean
        utilization = weights.mean(axis=0)
        metadata = {
            "min_component_usage": float(utilization.min()),
            "component_entropy": float(-np.mean(np.sum(utilization * np.log(utilization + 1e-12), axis=1))),
        }
        return Prediction(
            mean=mean,
            variance=variance,
            log_prob=log_prob,
            cdf=cdf,
            samples=samples,
            channels=self.channels_,
            metadata=metadata,
        )

    def metadata(self):
        result = super().metadata()
        result.update(
            {
                "components": self.components,
                "variance_floor": self.variance_floor,
                "tail_threshold_physical": self.tail_threshold,
                "shape_tail_z": self.shape_tail_z,
            }
        )
        return result


class ConditionalScoreMLP(_NeuralBase):
    """Unnormalized conditional score trained with an exact corruption kernel.

    This adapter supports held-out DSM risk, not NLL/CRPS. It therefore cannot be
    silently compared in normalized-density tables.
    """

    name = "conditional_score_mlp"
    score_domain = "conditional_outcome"
    capabilities = Capabilities(predictive_distribution="unnormalized_score")

    def __init__(
        self,
        *,
        kernel: NoiseKernel | None = None,
        noise_scale: float = 0.2,
        noise_scales: tuple[float, ...] | list[float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel = GaussianNoise() if kernel is None else kernel
        self.noise_scale = float(noise_scale)
        self.noise_scales = (
            (self.noise_scale,)
            if noise_scales is None
            else tuple(float(scale) for scale in noise_scales)
        )
        if not self.noise_scales or any(scale <= 0 for scale in self.noise_scales):
            raise ValueError("all noise scales must be positive")

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        if validation is None or not len(validation.targets):
            raise ValueError("ConditionalScoreMLP requires independent validation episodes")
        x, y = self._initialize(train)
        xv = self.scaler_.x(validation.features)
        yv = self.scaler_.y(validation.targets)
        self.n_targets_ = y.shape[1]
        self.network_ = _mlp(
            x.shape[1] + self.n_targets_, self.n_targets_, self.hidden, self.layers, self.dropout
        )
        self.network_.to(self.device_)
        self.trace_ = self._fit_loop(x, y, xv, yv)
        return self

    def _corruption(self, y: torch.Tensor, rng: np.random.Generator, scale: float):
        return self._corruption_with(self.kernel, y, rng, scale)

    def _corruption_with(
        self,
        kernel: NoiseKernel,
        y: torch.Tensor,
        rng: np.random.Generator,
        scale: float,
    ):
        noisy, aux = kernel.corrupt(y.detach().cpu().numpy(), scale, rng)
        target = kernel.conditional_score(
            noisy, y.detach().cpu().numpy(), scale, aux
        )
        return (
            torch.as_tensor(noisy, dtype=torch.float32, device=self.device_),
            torch.as_tensor(target, dtype=torch.float32, device=self.device_),
        )

    def _reference_kernel(self) -> NoiseKernel:
        return StudentTNoise(df=5.0) if isinstance(self.kernel, StudentTNoise) else GaussianNoise()

    def _fixed_reference_risks(
        self,
        data: SupervisedData,
        *,
        seed: int,
        scales: tuple[float, ...] = (0.1, 0.25, 0.5),
    ) -> tuple[str, dict[float, float]]:
        kernel = self._reference_kernel()
        self.network_.eval()
        x = torch.as_tensor(
            self.scaler_.x(data.features), dtype=torch.float32, device=self.device_
        )
        y = torch.as_tensor(
            self.scaler_.y(data.targets), dtype=torch.float32, device=self.device_
        )
        risks = {}
        with torch.no_grad():
            rng = np.random.default_rng(seed)
            for scale in scales:
                noisy, target = self._corruption_with(kernel, y, rng, scale)
                score = self.network_(torch.cat([x, noisy], dim=1))
                risks[scale] = float(
                    (scale**2 * torch.mean((score - target) ** 2)).cpu()
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

    def _training_loss(self, x, y, rng):
        losses = []
        for scale in self.noise_scales:
            noisy, target = self._corruption(y, rng, scale)
            score = self.network_(torch.cat([x, noisy], dim=1))
            losses.append(scale**2 * torch.mean((score - target) ** 2))
        return torch.stack(losses).mean()

    def _validation_loss(self, x, y):
        # Fixed validation stream for stable early stopping.
        rng = np.random.default_rng(self.seed + 999_983)
        losses = []
        for scale in self.noise_scales:
            noisy, target = self._corruption(y, rng, scale)
            score = self.network_(torch.cat([x, noisy], dim=1))
            losses.append(scale**2 * torch.mean((score - target) ** 2))
        return torch.stack(losses).mean()

    def dsm_risk(self, data: SupervisedData, seed: int = 0) -> float:
        self.network_.eval()
        x = torch.as_tensor(
            self.scaler_.x(data.features), dtype=torch.float32, device=self.device_
        )
        y = torch.as_tensor(self.scaler_.y(data.targets), dtype=torch.float32, device=self.device_)
        with torch.no_grad():
            rng = np.random.default_rng(seed)
            losses = []
            for scale in self.noise_scales:
                noisy, target = self._corruption(y, rng, scale)
                score = self.network_(torch.cat([x, noisy], dim=1))
                losses.append(scale**2 * torch.mean((score - target) ** 2))
            return float(torch.stack(losses).mean().cpu())

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        del data, n_samples
        raise UnsupportedCapability(self.name, "normalized_predictive_distribution")

    def metadata(self):
        result = super().metadata()
        result.update(
            {
                "kernel": self.kernel.name,
                "noise_scale": self.noise_scale,
                "noise_scales": self.noise_scales,
                "ladder_objective": "mean of scale-squared DSM risks",
                "score_domain": self.score_domain,
            }
        )
        if isinstance(self.kernel, StudentTNoise):
            result["kernel_df"] = self.kernel.df
        return result
