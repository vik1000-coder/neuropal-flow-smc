"""An explicit learned release→concentration→receptor stochastic state model.

Unlike a generic history MLP, this model contains a persistent positive latent
modulator, learnable clearance, activity-dependent release, saturating receptor
occupancy, and separate synaptic, intrinsic, additive, and stochastic-dispersion
entry tensors. It is deliberately compact enough for recovery studies on this host.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logsumexp, ndtr
from torch import nn

from ..capabilities import Capabilities, ResourceProfile
from ..metrics import gaussian_log_prob
from ..schema import Prediction, SupervisedData
from .base import Estimator, Standardizer, aggregate_features


class _MechanisticCore(nn.Module):
    def __init__(self, n: int, k: int, n_inputs: int):
        super().__init__()
        self.n, self.k = n, k
        self.release_raw = nn.Parameter(torch.randn(k, n) * 0.08)
        self.release_bias = nn.Parameter(torch.full((k,), -0.5))
        self.release_input_raw = nn.Parameter(torch.zeros(k, n_inputs))
        self.rho_logit = nn.Parameter(torch.full((k,), 2.0))
        self.expression_logit = nn.Parameter(torch.zeros(n, k))
        self.kd_raw = nn.Parameter(torch.zeros(n, k))
        self.base_connectivity_raw = nn.Parameter(torch.randn(n, n) * 0.08)
        self.state_decay_logit = nn.Parameter(torch.zeros(n))
        self.additive_effect = nn.Parameter(torch.zeros(n, k))
        self.synaptic_effect = nn.Parameter(torch.zeros(n, n, k))
        self.intrinsic_slope_effect = nn.Parameter(torch.zeros(n, k))
        self.intrinsic_threshold_effect = nn.Parameter(torch.zeros(n, k))
        self.logvariance_effect = nn.Parameter(torch.zeros(n, k))
        self.tail_logit_effect = nn.Parameter(torch.zeros(n, k))
        self.base_tail_logit = nn.Parameter(torch.full((n,), -2.442347))  # logit(.08)
        # Targets are standardized before fitting, so unit variance is the
        # calibrated neutral initialization. Starting at exp(-2) caused a large,
        # avoidable NLL transient in short recordings.
        self.base_logvariance = nn.Parameter(torch.zeros(n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.register_buffer("off_diagonal", 1.0 - torch.eye(n))

    def parameters_physical(self):
        release_weights = torch.nn.functional.softplus(self.release_raw)
        release_weights = release_weights / (release_weights.sum(dim=1, keepdim=True) + 1e-8)
        rho = 0.80 + 0.199 * torch.sigmoid(self.rho_logit)
        expression = torch.sigmoid(self.expression_logit)
        kd = torch.nn.functional.softplus(self.kd_raw) + 0.05
        raw_a = torch.tanh(self.base_connectivity_raw) * self.off_diagonal
        a = 0.45 * raw_a / torch.clamp(raw_a.abs().sum(dim=1, keepdim=True), min=1.0)
        decay = 0.75 * torch.sigmoid(self.state_decay_logit)
        return release_weights, rho, expression, kd, a, decay

    def step(
        self,
        x: torch.Tensor,
        concentration: torch.Tensor,
        stimulus: torch.Tensor | None = None,
        *,
        knockout=False,
    ):
        release_weights, rho, expression, kd, a, decay = self.parameters_physical()
        if stimulus is None or self.release_input_raw.shape[1] == 0:
            input_drive = 0.0
        else:
            input_drive = torch.einsum(
                "ku,bu->bk", torch.nn.functional.softplus(self.release_input_raw), stimulus
            )
        release = torch.nn.functional.softplus(
            torch.einsum("kn,bn->bk", release_weights, torch.tanh(x))
            + self.release_bias
            + input_drive
        )
        next_concentration = rho * concentration + (1.0 - rho) * release
        occupancy = expression[None, :, :] * next_concentration[:, None, :] / (
            kd[None, :, :] + next_concentration[:, None, :]
        )
        if knockout is True:
            occupancy = torch.zeros_like(occupancy)
        elif isinstance(knockout, torch.Tensor):
            if knockout.shape != (self.n, self.k):
                raise ValueError("receptor knockout mask has the wrong shape")
            occupancy = occupancy.masked_fill(knockout[None, :, :].bool(), 0.0)
        slope = torch.exp(torch.einsum("nk,bnk->bn", self.intrinsic_slope_effect, occupancy))
        threshold = torch.einsum("nk,bnk->bn", self.intrinsic_threshold_effect, occupancy)
        intrinsic = torch.tanh(slope * (x - threshold))
        log_gain = torch.einsum("nsk,bnk->bns", self.synaptic_effect, occupancy)
        effective_a = a[None, :, :] * torch.exp(torch.clamp(log_gain, -2.0, 2.0))
        recurrent = torch.einsum("bns,bs->bn", effective_a, torch.tanh(x))
        additive = torch.einsum("nk,bnk->bn", self.additive_effect, occupancy)
        mean = self.bias + decay * intrinsic + recurrent + additive
        logvariance = self.base_logvariance + torch.einsum(
            "nk,bnk->bn", self.logvariance_effect, occupancy
        )
        logvariance = torch.clamp(logvariance, -8.0, 4.0)
        tail_probability = torch.sigmoid(
            self.base_tail_logit
            + torch.einsum("nk,bnk->bn", self.tail_logit_effect, occupancy)
        )
        return mean, logvariance, tail_probability, next_concentration, occupancy, release

    def sequence(
        self,
        x: torch.Tensor,
        stimulus: torch.Tensor | None = None,
        initial: torch.Tensor | None = None,
        *,
        knockout=False,
    ):
        concentration = (
            torch.zeros((1, self.k), dtype=x.dtype, device=x.device)
            if initial is None
            else initial
        )
        means, logvariances, tail_probabilities, states, previous_states = [], [], [], [], []
        for t in range(len(x)):
            previous_states.append(concentration)
            stimulus_t = None if stimulus is None else stimulus[t : t + 1]
            knockout_t = (
                knockout[t]
                if isinstance(knockout, torch.Tensor) and knockout.ndim == 3
                else knockout
            )
            mean, logvariance, tail_probability, concentration, _, _ = self.step(
                x[t : t + 1], concentration, stimulus_t, knockout=knockout_t
            )
            means.append(mean)
            logvariances.append(logvariance)
            tail_probabilities.append(tail_probability)
            states.append(concentration)
        return (
            torch.cat(means),
            torch.cat(logvariances),
            torch.cat(tail_probabilities),
            concentration,
            torch.cat(states),
            torch.cat(previous_states),
        )


@dataclass
class MechanisticTrainingTrace:
    train_nll: list[float]
    validation_nll: list[float]
    best_epoch: int
    stopped_epoch: int


class LatentNeuromodulatedSSM(Estimator):
    """Learned neuromodulator state-space model with normalized Gaussian emissions."""

    name = "latent_neuromodulated_ssm"
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
        latent_state=True,
        interventions=True,
    )
    resource_profile = ResourceProfile(
        "accelerator_neural", threads=1, uses_accelerator=True, estimated_peak_gb=2.5
    )

    def __init__(
        self,
        *,
        n_modulators: int = 2,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        effect_l1: float = 1e-3,
        release_l1: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 20,
        min_delta: float = 1e-4,
        truncation: int = 64,
        warmup: int = 4,
        emission: str = "gaussian",
        tail_low_scale: float = 0.5,
        tail_high_scale: float = 3.0,
        tail_threshold: float = 0.35,
        shape_tail_z: float = 2.0,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.n_modulators = int(n_modulators)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.effect_l1 = float(effect_l1)
        self.release_l1 = float(release_l1)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.truncation = int(truncation)
        self.warmup = int(warmup)
        self.emission = str(emission)
        self.tail_low_scale = float(tail_low_scale)
        self.tail_high_scale = float(tail_high_scale)
        self.tail_threshold = float(tail_threshold)
        self.shape_tail_z = float(shape_tail_z)
        self.seed = int(seed)
        self.device_name = device
        if self.n_modulators < 1 or self.truncation < 2:
            raise ValueError("invalid latent model dimensions")
        if self.emission not in {"gaussian", "matched_tail_mixture"}:
            raise ValueError("emission must be gaussian or matched_tail_mixture")
        if not 0 < self.tail_low_scale < 1 < self.tail_high_scale:
            raise ValueError("tail scales must satisfy 0 < low < 1 < high")
        if self.shape_tail_z <= 0:
            raise ValueError("shape_tail_z must be positive")

    def _current_columns(self, data: SupervisedData) -> np.ndarray:
        return np.asarray(
            [np.flatnonzero(data.source_index == source)[0] for source in range(data.targets.shape[1])]
        )

    @staticmethod
    def _stimulus_columns(data: SupervisedData) -> np.ndarray:
        return np.asarray(
            [index for index, name in enumerate(data.feature_names) if name.startswith("stimulus")],
            dtype=int,
        )

    def _sequences(self, data: SupervisedData):
        columns = self._current_columns(data)
        stimulus_columns = self._stimulus_columns(data)
        x_all = self.scaler_.x(data.features)[:, columns]
        stimulus_all = data.features[:, stimulus_columns]
        y_all = self.scaler_.y(data.targets)
        sequences = []
        for group in np.unique(data.groups):
            index = np.flatnonzero(data.groups == group)
            index = index[np.argsort(data.times[index])]
            sequences.append((index, x_all[index], y_all[index], stimulus_all[index]))
        return sequences

    def _nll(self, mean, logvariance, y, tail_probability=None):
        if self.emission == "gaussian":
            return 0.5 * torch.mean(
                logvariance + (y - mean) ** 2 * torch.exp(-logvariance)
            )
        if tail_probability is None:
            raise ValueError("mixture emission requires tail probabilities")
        probability = torch.clamp(tail_probability, 1e-5, 1.0 - 1e-5)
        low2, high2 = self.tail_low_scale**2, self.tail_high_scale**2
        normalizer2 = (1.0 - probability) * low2 + probability * high2
        low_logvariance = logvariance + np.log(low2) - torch.log(normalizer2)
        high_logvariance = logvariance + np.log(high2) - torch.log(normalizer2)
        low_log_density = -0.5 * (
            np.log(2.0 * np.pi)
            + low_logvariance
            + (y - mean) ** 2 * torch.exp(-low_logvariance)
        )
        high_log_density = -0.5 * (
            np.log(2.0 * np.pi)
            + high_logvariance
            + (y - mean) ** 2 * torch.exp(-high_logvariance)
        )
        log_density = torch.logaddexp(
            torch.log1p(-probability) + low_log_density,
            torch.log(probability) + high_log_density,
        )
        return -torch.mean(log_density)

    def _penalty(self):
        effect = (
            self.model_.additive_effect.abs().mean()
            + self.model_.synaptic_effect.abs().mean()
            + self.model_.intrinsic_slope_effect.abs().mean()
            + self.model_.intrinsic_threshold_effect.abs().mean()
            + self.model_.logvariance_effect.abs().mean()
            + self.model_.tail_logit_effect.abs().mean()
        )
        release, *_ = self.model_.parameters_physical()
        release_entropy = -torch.sum(release * torch.log(release + 1e-8), dim=1).mean()
        return self.effect_l1 * effect + self.release_l1 * release_entropy

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        if validation is None or not len(validation.targets):
            raise ValueError("LatentNeuromodulatedSSM requires held-out validation episodes")
        if train.horizon != 1:
            raise ValueError("latent state-space fitting currently supports horizon 1")
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.device_ = torch.device(self.device_name)
        self.scaler_ = Standardizer.fit(train)
        self.n_targets_ = train.targets.shape[1]
        self.n_inputs_ = len(self._stimulus_columns(train))
        self.model_ = _MechanisticCore(
            self.n_targets_, self.n_modulators, self.n_inputs_
        ).to(self.device_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        train_sequences = self._sequences(train)
        validation_sequences = self._sequences(validation)
        rng = np.random.default_rng(self.seed)
        best = float("inf")
        best_state = copy.deepcopy(self.model_.state_dict())
        best_epoch = 0
        wait = 0
        train_trace, validation_trace = [], []
        for epoch in range(self.max_epochs):
            self.model_.train()
            losses = []
            for sequence_index in rng.permutation(len(train_sequences)):
                _, x_np, y_np, stimulus_np = train_sequences[sequence_index]
                concentration = torch.zeros((1, self.n_modulators), device=self.device_)
                for start in range(0, len(x_np), self.truncation):
                    x = torch.as_tensor(
                        x_np[start : start + self.truncation],
                        dtype=torch.float32,
                        device=self.device_,
                    )
                    y = torch.as_tensor(
                        y_np[start : start + self.truncation],
                        dtype=torch.float32,
                        device=self.device_,
                    )
                    stimulus = torch.as_tensor(
                        stimulus_np[start : start + self.truncation],
                        dtype=torch.float32,
                        device=self.device_,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    mean, logvariance, tail_probability, concentration, _, _ = self.model_.sequence(
                        x, stimulus, concentration
                    )
                    drop = min(self.warmup if start == 0 else 0, max(0, len(x) - 1))
                    loss = self._nll(
                        mean[drop:],
                        logvariance[drop:],
                        y[drop:],
                        tail_probability[drop:],
                    ) + self._penalty()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite latent-SSM loss at epoch {epoch}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                    optimizer.step()
                    concentration = concentration.detach()
                    losses.append(float(loss.detach().cpu()))
            validation_loss = self._validation_nll(validation_sequences)
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
        self.trace_ = MechanisticTrainingTrace(train_trace, validation_trace, best_epoch, epoch)
        self.source_index_ = train.source_index.copy()
        self.channels_ = self._readouts(train)
        return self

    def _validation_nll(self, sequences):
        self.model_.eval()
        losses = []
        with torch.no_grad():
            for _, x_np, y_np, stimulus_np in sequences:
                x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device_)
                y = torch.as_tensor(y_np, dtype=torch.float32, device=self.device_)
                stimulus = torch.as_tensor(
                    stimulus_np, dtype=torch.float32, device=self.device_
                )
                mean, logvariance, tail_probability, _, _, _ = self.model_.sequence(
                    x, stimulus
                )
                drop = min(self.warmup, max(0, len(x) - 1))
                losses.append(
                    float(
                        self._nll(
                            mean[drop:],
                            logvariance[drop:],
                            y[drop:],
                            tail_probability[drop:],
                        ).cpu()
                    )
                )
        return float(np.mean(losses))

    def _latent_and_previous(self, data: SupervisedData):
        states = np.zeros((len(data.targets), self.n_modulators))
        previous = np.zeros_like(states)
        means = np.zeros_like(data.targets)
        logvariances = np.zeros_like(data.targets)
        tail_probabilities = np.zeros_like(data.targets)
        self.model_.eval()
        with torch.no_grad():
            for index, x_np, _, stimulus_np in self._sequences(data):
                x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device_)
                stimulus = torch.as_tensor(
                    stimulus_np, dtype=torch.float32, device=self.device_
                )
                mean, logvariance, tail_probability, _, state, prev = self.model_.sequence(
                    x, stimulus
                )
                states[index] = state.cpu().numpy()
                previous[index] = prev.cpu().numpy()
                means[index] = mean.cpu().numpy()
                logvariances[index] = logvariance.cpu().numpy()
                tail_probabilities[index] = tail_probability.cpu().numpy()
        return means, logvariances, tail_probabilities, states, previous

    def latent_states(self, data: SupervisedData) -> np.ndarray:
        return self._latent_and_previous(data)[3]

    def _readouts(self, data: SupervisedData, max_rows: int = 192):
        _, _, _, _, previous = self._latent_and_previous(data)
        columns = self._current_columns(data)
        stimulus_columns = self._stimulus_columns(data)
        x_all = self.scaler_.x(data.features)[:, columns]
        stimulus_all = data.features[:, stimulus_columns]
        if len(x_all) > max_rows:
            index = np.linspace(0, len(x_all) - 1, max_rows, dtype=int)
            x_all, previous, stimulus_all = x_all[index], previous[index], stimulus_all[index]
        x = torch.as_tensor(x_all, dtype=torch.float32, device=self.device_).requires_grad_(True)
        concentration = torch.as_tensor(previous, dtype=torch.float32, device=self.device_)
        stimulus = torch.as_tensor(stimulus_all, dtype=torch.float32, device=self.device_)
        mean, logvariance, tail_probability, _, _, _ = self.model_.step(
            x, concentration, stimulus
        )
        threshold_z = torch.as_tensor(
            (self.tail_threshold - self.scaler_.y_mean) / self.scaler_.y_scale,
            dtype=torch.float32,
            device=self.device_,
        )
        if self.emission == "matched_tail_mixture":
            low2, high2 = self.tail_low_scale**2, self.tail_high_scale**2
            normalizer2 = (1.0 - tail_probability) * low2 + tail_probability * high2
            low_sd = torch.exp(0.5 * logvariance) * self.tail_low_scale / torch.sqrt(
                normalizer2
            )
            high_sd = torch.exp(0.5 * logvariance) * self.tail_high_scale / torch.sqrt(
                normalizer2
            )
            low_survival = 0.5 * torch.erfc(
                (threshold_z - mean) / low_sd / np.sqrt(2.0)
            )
            high_survival = 0.5 * torch.erfc(
                (threshold_z - mean) / high_sd / np.sqrt(2.0)
            )
            upper_tail = (1.0 - tail_probability) * low_survival + tail_probability * high_survival
            shape_low = 0.5 * torch.erfc(
                self.shape_tail_z
                * torch.sqrt(normalizer2)
                / self.tail_low_scale
                / np.sqrt(2.0)
            )
            shape_high = 0.5 * torch.erfc(
                self.shape_tail_z
                * torch.sqrt(normalizer2)
                / self.tail_high_scale
                / np.sqrt(2.0)
            )
            shape_tail = 2.0 * (
                (1.0 - tail_probability) * shape_low
                + tail_probability * shape_high
            )
        else:
            upper_tail = 0.5 * torch.erfc(
                (threshold_z - mean) / torch.exp(0.5 * logvariance) / np.sqrt(2.0)
            )
            gaussian_shape_tail = float(2.0 * ndtr(-self.shape_tail_z))
            shape_tail = mean * 0.0 + gaussian_shape_tail
        mean_jac = np.zeros((self.n_targets_, self.n_targets_))
        logvariance_jac = np.zeros_like(mean_jac)
        tail_jac = np.zeros_like(mean_jac)
        shape_tail_jac = np.zeros_like(mean_jac)
        for target in range(self.n_targets_):
            gm = torch.autograd.grad(mean[:, target].sum(), x, retain_graph=True)[0]
            gv = torch.autograd.grad(logvariance[:, target].sum(), x, retain_graph=True)[0]
            gt = torch.autograd.grad(upper_tail[:, target].sum(), x, retain_graph=True)[0]
            gs = torch.autograd.grad(shape_tail[:, target].sum(), x, retain_graph=True)[0]
            mean_jac[target] = gm.detach().cpu().numpy().mean(axis=0)
            logvariance_jac[target] = gv.detach().cpu().numpy().mean(axis=0)
            tail_jac[target] = gt.detach().cpu().numpy().mean(axis=0)
            shape_tail_jac[target] = gs.detach().cpu().numpy().mean(axis=0)
        x_scale = self.scaler_.x_scale[columns]
        mean_jac *= self.scaler_.y_scale[:, None] / x_scale
        logvariance_jac /= x_scale
        tail_jac /= x_scale
        shape_tail_jac /= x_scale
        return {
            "conditional_mean_derivative": mean_jac,
            "conditional_log_variance_derivative": logvariance_jac,
            "conditional_tail_high_derivative": tail_jac,
            "conditional_shape_tail_derivative": shape_tail_jac,
            "conditional_mean_derivative_feature": mean_jac,
            "conditional_log_variance_derivative_feature": logvariance_jac,
            "conditional_tail_high_derivative_feature": tail_jac,
            "conditional_shape_tail_derivative_feature": shape_tail_jac,
        }

    def _prediction(self, data: SupervisedData, *, knockout, n_samples: int):
        no_knockout = knockout is False or knockout is None
        if no_knockout:
            mean_z, logvariance_z, tail_probability, _, _ = self._latent_and_previous(data)
        else:
            mean_z = np.zeros_like(data.targets)
            logvariance_z = np.zeros_like(data.targets)
            tail_probability = np.zeros_like(data.targets)
            self.model_.eval()
            with torch.no_grad():
                for index, x_np, _, stimulus_np in self._sequences(data):
                    knockout_argument = (
                        True
                        if knockout is True
                        else torch.as_tensor(
                            (
                                knockout[index]
                                if np.asarray(knockout).ndim == 3
                                else knockout
                            ),
                            dtype=torch.bool,
                            device=self.device_,
                        )
                    )
                    x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device_)
                    stimulus = torch.as_tensor(
                        stimulus_np, dtype=torch.float32, device=self.device_
                    )
                    mean, logvariance, probability, _, _, _ = self.model_.sequence(
                        x, stimulus, knockout=knockout_argument
                    )
                    mean_z[index] = mean.cpu().numpy()
                    logvariance_z[index] = logvariance.cpu().numpy()
                    tail_probability[index] = probability.cpu().numpy()
        mean = self.scaler_.inverse_mean(mean_z)
        variance = self.scaler_.inverse_variance(np.exp(logvariance_z))
        rng = np.random.default_rng(self.seed + 1)
        samples = None
        prediction_metadata = {"emission": self.emission}
        if self.emission == "matched_tail_mixture":
            probability = np.clip(tail_probability, 1e-8, 1.0 - 1e-8)
            low2, high2 = self.tail_low_scale**2, self.tail_high_scale**2
            normalizer2 = (1.0 - probability) * low2 + probability * high2
            low_variance = variance * low2 / normalizer2
            high_variance = variance * high2 / normalizer2
            low_log_density = gaussian_log_prob(data.targets, mean, low_variance)
            high_log_density = gaussian_log_prob(data.targets, mean, high_variance)
            log_prob = logsumexp(
                np.stack(
                    [
                        np.log1p(-probability) + low_log_density,
                        np.log(probability) + high_log_density,
                    ]
                ),
                axis=0,
            )
            cdf = (
                (1.0 - probability)
                * ndtr((data.targets - mean) / np.sqrt(low_variance))
                + probability * ndtr((data.targets - mean) / np.sqrt(high_variance))
            )
            prediction_metadata.update(
                {
                    "mixture_probability": probability,
                    "component_variance_low": low_variance,
                    "component_variance_high": high_variance,
                }
            )
            if n_samples:
                high_component = rng.random((len(mean), n_samples, mean.shape[1])) < probability[:, None, :]
                component_sd = np.where(
                    high_component,
                    np.sqrt(high_variance)[:, None, :],
                    np.sqrt(low_variance)[:, None, :],
                )
                samples = mean[:, None, :] + component_sd * rng.standard_normal(
                    (len(mean), n_samples, mean.shape[1])
                )
        else:
            log_prob = gaussian_log_prob(data.targets, mean, variance)
            cdf = ndtr((data.targets - mean) / np.sqrt(np.maximum(variance, 1e-10)))
            if n_samples:
                samples = mean[:, None, :] + np.sqrt(variance[:, None, :]) * rng.standard_normal(
                    (len(mean), n_samples, mean.shape[1])
                )
        return Prediction(
            mean=mean,
            variance=variance,
            log_prob=log_prob,
            cdf=cdf,
            samples=samples,
            channels=self.channels_,
            metadata=prediction_metadata,
        )

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        return self._prediction(data, knockout=False, n_samples=n_samples)

    def predict_receptor_knockout(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        return self._prediction(data, knockout=True, n_samples=n_samples)

    def predict_intervention(
        self,
        data: SupervisedData,
        specification,
        *,
        n_samples: int = 0,
    ) -> Prediction:
        """Apply a represented receptor-specific knockout in learned coordinates."""

        if getattr(specification, "kind", None) != "receptor_knockout":
            raise ValueError("latent SSM currently represents receptor knockouts only")
        mask = np.zeros((self.n_targets_, self.n_modulators), dtype=bool)
        modulator = int(specification.modulator_index)
        targets = tuple(int(value) for value in specification.target_neurons)
        if not 0 <= modulator < self.n_modulators or not targets:
            raise ValueError("invalid receptor-specific knockout")
        if min(targets) < 0 or max(targets) >= self.n_targets_:
            raise ValueError("receptor knockout target is out of range")
        mask[list(targets), modulator] = True
        # Registered response panels encode the branch point as time zero. A
        # row-wise schedule keeps the operation inactive during the shared
        # filtering context and turns a finite knockout back off on time.
        active = np.asarray(data.times) >= 0
        duration = getattr(specification, "duration_steps", None)
        if duration is not None:
            active &= np.asarray(data.times) < int(duration)
        schedule = active[:, None, None] & mask[None, :, :]
        return self._prediction(data, knockout=schedule, n_samples=n_samples)

    def metadata(self):
        release, rho, expression, kd, a, decay = self.model_.parameters_physical()
        return {
            "n_modulators": self.n_modulators,
            "n_inputs": self.n_inputs_,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "effect_l1": self.effect_l1,
            "release_l1": self.release_l1,
            "release_penalty": "row-entropy sparsity",
            "emission": self.emission,
            "tail_low_scale": self.tail_low_scale,
            "tail_high_scale": self.tail_high_scale,
            "tail_threshold": self.tail_threshold,
            "shape_tail_z": self.shape_tail_z,
            "truncation": self.truncation,
            "warmup": self.warmup,
            "best_epoch": self.trace_.best_epoch,
            "stopped_epoch": self.trace_.stopped_epoch,
            "best_validation_nll": min(self.trace_.validation_nll),
            "learned_clearance_rho": rho.detach().cpu().numpy(),
            "learned_release_weights": release.detach().cpu().numpy(),
            "learned_release_input": torch.nn.functional.softplus(
                self.model_.release_input_raw
            ).detach().cpu().numpy(),
            "learned_receptor_expression": expression.detach().cpu().numpy(),
            "learned_receptor_kd": kd.detach().cpu().numpy(),
            "learned_baseline_connectivity": a.detach().cpu().numpy(),
            "learned_state_decay": decay.detach().cpu().numpy(),
            "learned_additive_effect": self.model_.additive_effect.detach().cpu().numpy(),
            "learned_synaptic_effect": self.model_.synaptic_effect.detach().cpu().numpy(),
            "learned_intrinsic_slope_effect": (
                self.model_.intrinsic_slope_effect.detach().cpu().numpy()
            ),
            "learned_intrinsic_threshold_effect": (
                self.model_.intrinsic_threshold_effect.detach().cpu().numpy()
            ),
            "learned_logvariance_effect": self.model_.logvariance_effect.detach().cpu().numpy(),
            "learned_tail_logit_effect": self.model_.tail_logit_effect.detach().cpu().numpy(),
            "learned_tail_base_probability": torch.sigmoid(
                self.model_.base_tail_logit
            ).detach().cpu().numpy(),
            "identification": "latent concentration evaluated modulo validation-fitted assignment/affine map",
            "latent_orientation": "positive_concentration",
            "uses_current_modulator_features": False,
            "effective_information_set": (
                "observed_neural_history_plus_stimulus; modulator reconstructed "
                "recursively as a latent state"
            ),
        }
