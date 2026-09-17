"""Stable finite signed-contrast estimators from the revised SBTG note.

The central object is the mixture-gauge witness for histories ``h +/- delta*v``.
This module deliberately keeps the post-processor separate from the predictive
generator: callers provide balanced signed-mixture samples, and the classifier
never receives the perturbed history or stencil sign as an input feature.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


EPS = 1e-12


def _finite_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {array.ndim}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def central_mixture_witness(
    log_prob_plus: Any, log_prob_minus: Any, delta: float
) -> np.ndarray:
    """Return the exact bounded central witness from two log densities.

    ``tanh`` is the stable form of
    ``(p_plus-p_minus)/(delta*(p_plus+p_minus))``.
    """

    if not math.isfinite(float(delta)) or float(delta) <= 0:
        raise ValueError("delta must be finite and positive")
    plus = _finite_array(log_prob_plus, "log_prob_plus")
    minus = _finite_array(log_prob_minus, "log_prob_minus")
    if plus.shape != minus.shape:
        raise ValueError("log-probability arrays must have the same shape")
    return np.tanh(0.5 * (plus - minus)) / float(delta)


def signed_mixture_readout(
    channels: Any, labels: Any, coefficient_norm: float
) -> np.ndarray:
    """Estimate ``sum c_r E[phi(Y)|h_r]`` from the signed mixture.

    ``coefficient_norm`` is ``C=sum(abs(c_r))``.  For the central first
    difference it is ``1/delta``; balanced labels are ``-1`` and ``+1``.
    """

    if not math.isfinite(float(coefficient_norm)) or coefficient_norm <= 0:
        raise ValueError("coefficient_norm must be finite and positive")
    values = _finite_array(channels, "channels")
    sign = _finite_array(labels, "labels", ndim=1)
    if values.shape[0] != len(sign):
        raise ValueError("channels and labels must have the same first dimension")
    if not np.all(np.isin(sign, (-1.0, 1.0))):
        raise ValueError("labels must be -1 or +1")
    return float(coefficient_norm) * np.mean(sign.reshape((-1,) + (1,) * (values.ndim - 1)) * values, axis=0)


def expected_calibration_error(
    probabilities: Any, labels: Any, *, bins: int = 10
) -> float:
    probability = _finite_array(probabilities, "probabilities", ndim=1)
    target = _finite_array(labels, "labels", ndim=1)
    if len(probability) != len(target) or not np.all((probability >= 0) & (probability <= 1)):
        raise ValueError("probabilities and labels are incompatible")
    if bins < 2:
        raise ValueError("bins must be at least two")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if np.any(selected):
            result += float(np.mean(selected)) * abs(
                float(np.mean(probability[selected])) - float(np.mean(target[selected]))
            )
    return float(result)


def oracle_logistic_excess_risk(true_probability: Any, fitted_probability: Any) -> float:
    """Population-style logistic regret on a synthetic oracle mixture panel."""

    truth = _finite_array(true_probability, "true_probability", ndim=1)
    fitted = _finite_array(fitted_probability, "fitted_probability", ndim=1)
    if truth.shape != fitted.shape or not np.all((truth >= 0) & (truth <= 1)):
        raise ValueError("probability arrays are incompatible")
    fitted = np.clip(fitted, EPS, 1.0 - EPS)
    truth_safe = np.clip(truth, EPS, 1.0 - EPS)
    cross_entropy = -np.mean(truth * np.log(fitted) + (1.0 - truth) * np.log(1.0 - fitted))
    bayes = -np.mean(
        truth * np.log(truth_safe) + (1.0 - truth) * np.log(1.0 - truth_safe)
    )
    return float(max(0.0, cross_entropy - bayes))


@dataclass(frozen=True)
class FixedResponseDictionary:
    """History-independent response features frozen from a training split."""

    response_mean: np.ndarray
    response_scale: np.ndarray
    feature_mean: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def fit(cls, response: Any, *, epsilon: float = 1e-8) -> "FixedResponseDictionary":
        y = _finite_array(response, "response", ndim=2)
        mean = np.mean(y, axis=0)
        scale = np.std(y, axis=0)
        scale = np.where(scale > epsilon, scale, 1.0)
        raw, names = cls._raw((y - mean) / scale)
        return cls(mean, scale, np.mean(raw, axis=0), names)

    @staticmethod
    def _raw(z: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        columns: list[np.ndarray] = []
        names: list[str] = []
        dimension = z.shape[1]
        for index in range(dimension):
            columns.append(z[:, index])
            names.append(f"linear_{index}")
        for index in range(dimension):
            columns.append(np.square(z[:, index]))
            names.append(f"quadratic_{index}")
        for left in range(dimension):
            for right in range(left + 1, dimension):
                columns.append(z[:, left] * z[:, right])
                names.append(f"cross_{left}_{right}")
        for index in range(dimension):
            columns.append(z[:, index] ** 3 - 3.0 * z[:, index])
            names.append(f"cubic_{index}")
        for index in range(dimension):
            columns.append(np.tanh(z[:, index] - 1.0))
            names.append(f"smooth_tail_{index}")
        return np.column_stack(columns), tuple(names)

    def transform(self, response: Any) -> np.ndarray:
        y = _finite_array(response, "response", ndim=2)
        if y.shape[1] != len(self.response_mean):
            raise ValueError("response dimension differs from the fitted dictionary")
        raw, names = self._raw((y - self.response_mean) / self.response_scale)
        if names != self.names:
            raise RuntimeError("response dictionary ordering changed")
        return raw - self.feature_mean


@dataclass(frozen=True)
class SignedRieszFit:
    coefficients: np.ndarray
    gram: np.ndarray
    moment: np.ndarray
    feature_center: np.ndarray
    ridge: float
    condition_number: float
    witness_norm_sq: float
    training_centering: float


class SignedRieszRegressor:
    """Finite-dimensional signed least-squares/Riesz projection."""

    def __init__(self, dictionary: FixedResponseDictionary, ridge: float = 1e-3):
        if ridge < 0 or not math.isfinite(float(ridge)):
            raise ValueError("ridge must be finite and nonnegative")
        self.dictionary = dictionary
        self.ridge = float(ridge)
        self.fit_: SignedRieszFit | None = None

    def fit(self, response: Any, labels: Any, coefficient_norm: float) -> SignedRieszFit:
        raw_features = self.dictionary.transform(response)
        sign = _finite_array(labels, "labels", ndim=1)
        if len(raw_features) != len(sign) or not np.all(np.isin(sign, (-1.0, 1.0))):
            raise ValueError("Riesz labels must be matching -1/+1 values")
        # The response functions are frozen globally, while their constant
        # component is removed on this independent signed-mixture fit.  This
        # enforces the Riesz centering condition without redefining channels.
        feature_center = np.mean(raw_features, axis=0)
        features = raw_features - feature_center
        gram = features.T @ features / len(features)
        moment = float(coefficient_norm) * np.mean(sign[:, None] * features, axis=0)
        regularized = gram + self.ridge * np.eye(gram.shape[0])
        coefficients = np.linalg.solve(regularized, moment)
        witness = features @ coefficients
        result = SignedRieszFit(
            coefficients=coefficients,
            gram=gram,
            moment=moment,
            feature_center=feature_center,
            ridge=self.ridge,
            condition_number=float(np.linalg.cond(regularized)),
            witness_norm_sq=float(coefficients @ gram @ coefficients),
            training_centering=float(abs(np.mean(witness))),
        )
        self.fit_ = result
        return result

    def predict(self, response: Any) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("Riesz regressor must be fit before prediction")
        features = self.dictionary.transform(response) - self.fit_.feature_center
        return features @ self.fit_.coefficients


class _ClassifierNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int):
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(max(0, layers - 1)):
            modules.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        modules.append(nn.Linear(hidden, 1))
        self.network = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


@dataclass(frozen=True)
class ClassifierFit:
    best_epoch: int
    validation_loss: float
    training_seconds: float
    temperature: float
    calibration_bias: float
    calibration_centering: float


class AmortizedSignedClassifier:
    """Balanced +/- predictive-law classifier with independent calibration."""

    def __init__(
        self,
        history_dim: int,
        response_dim: int,
        *,
        hidden: int = 48,
        layers: int = 2,
        seed: int = 0,
    ):
        if history_dim < 1 or response_dim < 1 or hidden < 4 or layers < 1:
            raise ValueError("classifier dimensions are invalid")
        self.history_dim = int(history_dim)
        self.response_dim = int(response_dim)
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        self.network = _ClassifierNetwork(history_dim + response_dim, hidden, layers)
        self.input_mean: np.ndarray | None = None
        self.input_scale: np.ndarray | None = None
        self.temperature = 1.0
        self.calibration_bias = 0.0
        self.fit_: ClassifierFit | None = None

    def _input(self, history: Any, response: Any, *, fit: bool = False) -> np.ndarray:
        h = _finite_array(history, "history", ndim=2)
        y = _finite_array(response, "response", ndim=2)
        if len(h) != len(y) or h.shape[1] != self.history_dim or y.shape[1] != self.response_dim:
            raise ValueError("classifier history/response shapes are incompatible")
        value = np.concatenate([h, y], axis=1)
        if fit:
            self.input_mean = np.mean(value, axis=0)
            scale = np.std(value, axis=0)
            self.input_scale = np.where(scale > 1e-8, scale, 1.0)
        if self.input_mean is None or self.input_scale is None:
            raise RuntimeError("classifier input standardization is not fit")
        return ((value - self.input_mean) / self.input_scale).astype(np.float32)

    @staticmethod
    def _targets(labels: Any) -> np.ndarray:
        sign = _finite_array(labels, "labels", ndim=1)
        if not np.all(np.isin(sign, (-1.0, 1.0))):
            raise ValueError("classifier labels must be -1 or +1")
        return ((sign + 1.0) / 2.0).astype(np.float32)

    def fit(
        self,
        history: Any,
        response: Any,
        labels: Any,
        calibration_history: Any,
        calibration_response: Any,
        calibration_labels: Any,
        *,
        max_epochs: int = 40,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 8,
    ) -> ClassifierFit:
        if max_epochs < 1 or batch_size < 2 or patience < 1:
            raise ValueError("classifier training controls are invalid")
        x = self._input(history, response, fit=True)
        target = self._targets(labels)
        x_cal = self._input(calibration_history, calibration_response)
        target_cal = self._targets(calibration_labels)
        if len(x) != len(target) or len(x_cal) != len(target_cal):
            raise ValueError("classifier labels do not match inputs")
        if abs(float(np.mean(target_cal)) - 0.5) > 1e-8:
            raise ValueError("calibration split must be exactly balanced")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        x_tensor = torch.as_tensor(x)
        target_tensor = torch.as_tensor(target)
        x_cal_tensor = torch.as_tensor(x_cal)
        target_cal_tensor = torch.as_tensor(target_cal)
        best_loss = float("inf")
        best_epoch = -1
        best_state = copy.deepcopy(self.network.state_dict())
        wait = 0
        started = time.perf_counter()
        for epoch in range(max_epochs):
            self.network.train()
            permutation = rng.permutation(len(x))
            for start in range(0, len(permutation), batch_size):
                index = torch.as_tensor(permutation[start : start + batch_size], dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(
                    self.network(x_tensor[index]), target_tensor[index]
                )
                loss.backward()
                optimizer.step()
            self.network.eval()
            with torch.no_grad():
                validation = float(
                    F.binary_cross_entropy_with_logits(
                        self.network(x_cal_tensor), target_cal_tensor
                    )
                )
            if validation < best_loss - 1e-6:
                best_loss = validation
                best_epoch = epoch
                best_state = copy.deepcopy(self.network.state_dict())
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break
        self.network.load_state_dict(best_state)
        self.network.eval()
        with torch.no_grad():
            calibration_logits = self.network(x_cal_tensor).detach()
        log_temperature = torch.zeros((), requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        calibrator = torch.optim.LBFGS(
            [log_temperature, bias], lr=0.25, max_iter=80, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            calibrator.zero_grad()
            adjusted = calibration_logits / torch.exp(log_temperature).clamp_min(1e-4) + bias
            value = F.binary_cross_entropy_with_logits(adjusted, target_cal_tensor)
            value.backward()
            return value

        calibrator.step(closure)
        self.temperature = float(torch.exp(log_temperature.detach()).clamp(1e-3, 1e3))
        self.calibration_bias = float(bias.detach())
        with torch.no_grad():
            calibrated = torch.sigmoid(
                calibration_logits / self.temperature + self.calibration_bias
            )
        fit = ClassifierFit(
            best_epoch=best_epoch,
            validation_loss=best_loss,
            training_seconds=float(time.perf_counter() - started),
            temperature=self.temperature,
            calibration_bias=self.calibration_bias,
            calibration_centering=float(abs(2.0 * calibrated.mean().item() - 1.0)),
        )
        self.fit_ = fit
        return fit

    def predict_logit(self, history: Any, response: Any) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("classifier must be fit before prediction")
        value = torch.as_tensor(self._input(history, response))
        self.network.eval()
        with torch.no_grad():
            logits = self.network(value) / self.temperature + self.calibration_bias
        return logits.cpu().numpy().astype(np.float64)

    def predict_probability(self, history: Any, response: Any) -> np.ndarray:
        logits = self.predict_logit(history, response)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    def predict_witness(
        self, history: Any, response: Any, coefficient_norm: float
    ) -> np.ndarray:
        if coefficient_norm <= 0 or not math.isfinite(float(coefficient_norm)):
            raise ValueError("coefficient_norm must be finite and positive")
        return float(coefficient_norm) * (
            2.0 * self.predict_probability(history, response) - 1.0
        )


__all__ = [
    "AmortizedSignedClassifier",
    "ClassifierFit",
    "FixedResponseDictionary",
    "SignedRieszFit",
    "SignedRieszRegressor",
    "central_mixture_witness",
    "expected_calibration_error",
    "oracle_logistic_excess_risk",
    "signed_mixture_readout",
]
