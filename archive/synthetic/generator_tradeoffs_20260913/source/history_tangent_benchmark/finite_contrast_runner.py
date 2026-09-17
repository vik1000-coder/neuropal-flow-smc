"""Developmental V1 comparison for the revised finite-contrast SBTG note."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from .dgps import ConditionalDGP, make_dgp
from .finite_contrast import (
    AmortizedSignedClassifier,
    FixedResponseDictionary,
    SignedRieszRegressor,
    central_mixture_witness,
    expected_calibration_error,
    oracle_logistic_excess_risk,
    signed_mixture_readout,
)
from .metrics import TrainScaler
from .models import ConditionalModel, ConditionalVectorDiffusion, RatioCritic, build_model
from .reporting import load_case_records
from .serialization import (
    atomic_json,
    enforce_disk_safety,
    environment_manifest,
    sha256_file,
    source_tree_sha256,
    stable_sha256,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-12


@dataclass(frozen=True)
class FiniteContrastConfig:
    name: str
    base_output: str
    output_dir: str
    technical_note: str
    deltas: tuple[float, ...]
    response_noise_sigmas: tuple[float, ...]
    adapter_seed: int
    adapter_train_anchors: int
    adapter_calibration_anchors: int
    evaluation_anchors: int
    evaluation_draws_per_side: int
    riesz_draws_per_side: int
    diffusion_center_draws: int
    riesz_ridge: float
    classifier_hidden: int
    classifier_layers: int
    classifier_epochs: int
    classifier_batch_size: int
    classifier_learning_rate: float
    classifier_patience: int
    min_free_disk_gb: float
    max_output_gb: float

    def validate(self) -> None:
        if not self.name or not self.base_output or not self.output_dir:
            raise ValueError("name and output paths must be nonempty")
        if not self.deltas or any(value <= 0 for value in self.deltas):
            raise ValueError("deltas must be positive")
        if not self.response_noise_sigmas or any(value < 0 for value in self.response_noise_sigmas):
            raise ValueError("response noise scales must be nonnegative")
        counts = (
            self.adapter_train_anchors,
            self.adapter_calibration_anchors,
            self.evaluation_anchors,
            self.evaluation_draws_per_side,
            self.riesz_draws_per_side,
            self.diffusion_center_draws,
            self.classifier_hidden,
            self.classifier_layers,
            self.classifier_epochs,
            self.classifier_batch_size,
            self.classifier_patience,
        )
        if any(value < 1 for value in counts):
            raise ValueError("finite-contrast counts must be positive")
        if self.riesz_ridge < 0 or self.classifier_learning_rate <= 0:
            raise ValueError("regularization/training controls are invalid")
        if self.min_free_disk_gb <= 0 or self.max_output_gb <= 0:
            raise ValueError("disk limits must be positive")


def load_finite_contrast_config(path: str | Path) -> FiniteContrastConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("finite-contrast config must be a mapping")
    allowed = set(FiniteContrastConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    missing = allowed - set(raw)
    if unknown or missing:
        raise ValueError(f"finite-contrast config unknown={sorted(unknown)} missing={sorted(missing)}")
    resolved = dict(raw)
    resolved["deltas"] = tuple(float(value) for value in raw["deltas"])
    resolved["response_noise_sigmas"] = tuple(
        float(value) for value in raw["response_noise_sigmas"]
    )
    config = FiniteContrastConfig(**resolved)
    config.validate()
    return config


def _resolve_inside(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    resolved = path.resolve()
    if resolved != PACKAGE_ROOT and PACKAGE_ROOT not in resolved.parents:
        raise ValueError(f"path escapes isolated package: {resolved}")
    return resolved


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scaler(payload: Mapping[str, Any]) -> TrainScaler:
    return TrainScaler(
        history_mean=np.asarray(payload["history_mean"], dtype=np.float64),
        history_scale=np.asarray(payload["history_scale"], dtype=np.float64),
        response_mean=np.asarray(payload["response_mean"], dtype=np.float64),
        response_scale=np.asarray(payload["response_scale"], dtype=np.float64),
        n_train=int(payload["n_train"]),
    )


def _load_model(base_output: Path, record: Mapping[str, Any], dgp: ConditionalDGP) -> tuple[ConditionalModel, TrainScaler]:
    artifact = record["artifacts"]["checkpoint"]
    checkpoint_path = base_output / artifact["path"]
    if sha256_file(checkpoint_path) != artifact["sha256"]:
        raise RuntimeError(f"checkpoint digest mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_spec = record["model"]
    model = build_model(
        model_spec["kind"], q=dgp.q, dy=dgp.dy, params=dict(model_spec["params"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device("cpu"))
    model.eval()
    return model, _scaler(checkpoint["scaler"])


def _training_data(dgp: ConditionalDGP, data_seed: int, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    history = dgp.sample_history(n_train, data_seed + 10_001)
    response = dgp.sample_response(history, 1, data_seed + 10_002)[:, 0, :]
    return _as_numpy(history), _as_numpy(response)


def _direction(dgp: ConditionalDGP, training_history: np.ndarray) -> tuple[str, np.ndarray]:
    directions = dgp.mechanism_directions()
    if directions:
        name, value = next(iter(directions.items()))
        vector = _as_numpy(value).reshape(-1).astype(np.float64)
    else:
        name = "fallback_axis_0"
        vector = np.eye(dgp.q, dtype=np.float64)[0]
    covariance = np.atleast_2d(np.cov(training_history, rowvar=False))
    inverse = np.linalg.pinv(covariance)
    norm = float(np.sqrt(vector @ inverse @ vector))
    if not math.isfinite(norm) or norm <= EPS:
        raise ValueError(f"invalid mechanism direction {name!r}")
    return str(name), vector / norm


def _sample_oracle(
    dgp: ConditionalDGP,
    scaler: TrainScaler,
    history: np.ndarray,
    n_draws: int,
    seed: int,
    response_noise_sigma: float,
) -> np.ndarray:
    clean = _as_numpy(
        dgp.sample_response(torch.as_tensor(history, dtype=torch.float64), n_draws, seed)
    )
    standardized = np.asarray(scaler.transform_response(clean), dtype=np.float64)
    if response_noise_sigma > 0:
        noise = np.random.default_rng(seed + 1_000_003).standard_normal(standardized.shape)
        standardized = standardized + float(response_noise_sigma) * noise
    return np.asarray(scaler.inverse_response(standardized), dtype=np.float64)


def _sample_model(
    model: ConditionalModel,
    scaler: TrainScaler,
    history: np.ndarray,
    n_draws: int,
    seed: int,
    response_noise_sigma: float,
) -> np.ndarray:
    history_s = torch.as_tensor(
        np.asarray(scaler.transform_history(history), dtype=np.float32)
    )
    with torch.no_grad():
        if isinstance(model, ConditionalVectorDiffusion) and response_noise_sigma > 0:
            draws_s = model.sample_noisy(
                history_s, n_draws, sigma=response_noise_sigma, seed=seed
            )
        else:
            draws_s = model.sample(history_s, n_samples=n_draws, seed=seed)
            if response_noise_sigma > 0:
                generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
                noise = torch.randn(draws_s.shape, generator=generator, dtype=draws_s.dtype)
                draws_s = draws_s + float(response_noise_sigma) * noise
    return np.asarray(scaler.inverse_response(_as_numpy(draws_s)), dtype=np.float64)


SampleFunction = Callable[[np.ndarray, int, int, float], np.ndarray]


def _balanced_samples(
    sampler: SampleFunction,
    anchors: np.ndarray,
    direction: np.ndarray,
    delta: float,
    n_draws: int,
    seed: int,
    response_noise_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plus = sampler(
        anchors + float(delta) * direction, n_draws, seed, response_noise_sigma
    )
    minus = sampler(
        anchors - float(delta) * direction, n_draws, seed, response_noise_sigma
    )
    repeated = np.repeat(anchors[:, None, :], n_draws, axis=1).reshape(-1, anchors.shape[1])
    history = np.concatenate([repeated, repeated], axis=0)
    response = np.concatenate(
        [plus.reshape(-1, plus.shape[-1]), minus.reshape(-1, minus.shape[-1])], axis=0
    )
    labels = np.concatenate(
        [np.ones(len(repeated), dtype=np.float64), -np.ones(len(repeated), dtype=np.float64)]
    )
    return history, response, labels


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _panel(
    dgp: ConditionalDGP,
    scaler: TrainScaler,
    dictionary: FixedResponseDictionary,
    anchors: np.ndarray,
    direction: np.ndarray,
    delta: float,
    n_draws: int,
    seed: int,
    response_noise_sigma: float,
) -> dict[str, Any]:
    sampler = lambda h, n, s, tau: _sample_oracle(dgp, scaler, h, n, s, tau)
    history, response, labels = _balanced_samples(
        sampler, anchors, direction, delta, n_draws, seed, response_noise_sigma
    )
    plus_history = history + float(delta) * direction
    minus_history = history - float(delta) * direction
    y = torch.as_tensor(response, dtype=torch.float64)
    h_plus = torch.as_tensor(plus_history, dtype=torch.float64)
    h_minus = torch.as_tensor(minus_history, dtype=torch.float64)
    if response_noise_sigma > 0:
        sigma_original = torch.as_tensor(
            np.asarray(scaler.response_scale) * float(response_noise_sigma),
            dtype=torch.float64,
        )
        log_plus = _as_numpy(dgp.noisy_log_prob(y, h_plus, sigma_original))
        log_minus = _as_numpy(dgp.noisy_log_prob(y, h_minus, sigma_original))
        tangent = _as_numpy(
            dgp.noisy_history_tangent(
                y, torch.as_tensor(history, dtype=torch.float64), sigma_original
            )
        )
    else:
        log_plus = _as_numpy(dgp.log_prob(y, h_plus))
        log_minus = _as_numpy(dgp.log_prob(y, h_minus))
        tangent = _as_numpy(
            dgp.history_tangent(y, torch.as_tensor(history, dtype=torch.float64))
        )
    witness = central_mixture_witness(log_plus, log_minus, delta)
    channels = dictionary.transform(response)
    direct = signed_mixture_readout(channels, labels, 1.0 / delta)
    probability = _sigmoid(log_plus - log_minus)
    return {
        "history": history,
        "response": response,
        "labels": labels,
        "channels": channels,
        "channel_names": dictionary.names,
        "direct_readout": direct,
        "witness": witness,
        "true_probability": probability,
        "oracle_tangent": tangent @ direction,
        "anchors": anchors,
        "n_draws": n_draws,
    }


def _nrmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(estimate - truth)) / max(np.mean(np.square(truth)), EPS)))


def _cosine(estimate: np.ndarray, truth: np.ndarray) -> float:
    numerator = float(np.sum(estimate * truth))
    denominator = float(np.linalg.norm(estimate) * np.linalg.norm(truth))
    return numerator / max(denominator, EPS)


def _metric_row(context: Mapping[str, Any], metric_id: str, value: float, unit: str, n: int) -> dict[str, Any]:
    if not math.isfinite(float(value)):
        raise FloatingPointError(f"non-finite metric {metric_id}")
    return {
        **context,
        "metric_id": metric_id,
        "value": float(value),
        "unit": unit,
        "n": int(n),
        "status": "ok",
    }


def _score_witness(
    context: Mapping[str, Any], estimate: np.ndarray, panel: Mapping[str, Any]
) -> list[dict[str, Any]]:
    truth = np.asarray(panel["witness"], dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(truth.shape)
    channels = np.asarray(panel["channels"], dtype=np.float64)
    target_readout = np.asarray(panel["direct_readout"], dtype=np.float64)
    estimated_readout = np.mean(channels * estimate[:, None], axis=0)
    rows = [
        _metric_row(context, "witness_nrmse", _nrmse(estimate, truth), "ratio", len(truth)),
        _metric_row(
            context,
            "witness_rmse",
            float(np.sqrt(np.mean(np.square(estimate - truth)))),
            "inverse_history_units",
            len(truth),
        ),
        _metric_row(context, "witness_cosine", _cosine(estimate, truth), "correlation", len(truth)),
        _metric_row(context, "witness_centering_abs", abs(float(np.mean(estimate))), "inverse_history_units", len(truth)),
        _metric_row(
            context,
            "finite_information_relative_error",
            abs(float(np.mean(np.square(estimate)) - np.mean(np.square(truth))))
            / max(float(np.mean(np.square(truth))), EPS),
            "ratio",
            len(truth),
        ),
        _metric_row(
            context,
            "witness_bound_use",
            float(np.max(np.abs(estimate)) * float(context["delta"])),
            "fraction_of_1_over_delta",
            len(truth),
        ),
        _metric_row(
            context,
            "channel_readout_nrmse",
            _nrmse(estimated_readout, target_readout),
            "ratio",
            len(target_readout),
        ),
        _metric_row(
            context,
            "channel_readout_rmse",
            float(np.sqrt(np.mean(np.square(estimated_readout - target_readout)))),
            "standardized_channel_per_history_unit",
            len(target_readout),
        ),
    ]
    names = tuple(panel["channel_names"])
    for family in ("linear", "quadratic", "cross", "cubic", "smooth_tail"):
        index = np.asarray([name.startswith(family) for name in names])
        if np.any(index):
            rows.append(
                _metric_row(
                    context,
                    f"channel_{family}_nrmse",
                    _nrmse(estimated_readout[index], target_readout[index]),
                    "ratio",
                    int(np.sum(index)),
                )
            )
    return rows


def _direct_channel_rows(
    context: Mapping[str, Any], response: np.ndarray, labels: np.ndarray,
    dictionary: FixedResponseDictionary, panel: Mapping[str, Any]
) -> list[dict[str, Any]]:
    estimate = signed_mixture_readout(
        dictionary.transform(response), labels, 1.0 / float(context["delta"])
    )
    truth = np.asarray(panel["direct_readout"])
    return [
        _metric_row(context, "channel_readout_nrmse", _nrmse(estimate, truth), "ratio", len(truth)),
        _metric_row(
            context,
            "channel_readout_rmse",
            float(np.sqrt(np.mean(np.square(estimate - truth)))),
            "standardized_channel_per_history_unit",
            len(truth),
        ),
    ]


def _model_tangent(
    model: ConditionalModel,
    scaler: TrainScaler,
    dgp: ConditionalDGP,
    panel: Mapping[str, Any],
    direction: np.ndarray,
    response_noise_sigma: float,
    center_draws: int,
    seed: int,
) -> np.ndarray | None:
    history = np.asarray(panel["history"])
    response = np.asarray(panel["response"])
    if isinstance(model, ConditionalVectorDiffusion):
        if response_noise_sigma <= 0:
            return None
        response_s = np.asarray(scaler.transform_response(response), dtype=np.float32)
        center = _sample_oracle(
            dgp, scaler, history, center_draws, seed, response_noise_sigma
        )
        center_s = np.asarray(scaler.transform_response(center), dtype=np.float32)
        estimate_s = model.mixed_history_tangent(
            torch.as_tensor(response_s),
            torch.as_tensor(np.asarray(scaler.transform_history(history), dtype=np.float32)),
            response_noise_sigma,
            torch.as_tensor(center_s),
            quadrature_points=4,
        )
        estimate = _as_numpy(scaler.history_tangent_to_original(estimate_s))
        return estimate @ direction
    if response_noise_sigma > 0 or not model.capabilities.history_tangent:
        return None
    response_s = torch.as_tensor(
        np.asarray(scaler.transform_response(response), dtype=np.float32)
    )
    history_s = torch.as_tensor(
        np.asarray(scaler.transform_history(history), dtype=np.float32)
    )
    estimate_s = model.history_tangent(response_s, history_s)
    estimate = _as_numpy(scaler.history_tangent_to_original(estimate_s))
    return estimate @ direction


def _central_model_witness(
    model: ConditionalModel,
    scaler: TrainScaler,
    panel: Mapping[str, Any],
    direction: np.ndarray,
    delta: float,
) -> np.ndarray | None:
    if not (model.capabilities.normalized_density or isinstance(model, RatioCritic)):
        return None
    response = torch.as_tensor(
        np.asarray(scaler.transform_response(panel["response"]), dtype=np.float32)
    )
    history = np.asarray(panel["history"])
    plus = torch.as_tensor(
        np.asarray(scaler.transform_history(history + delta * direction), dtype=np.float32)
    )
    minus = torch.as_tensor(
        np.asarray(scaler.transform_history(history - delta * direction), dtype=np.float32)
    )
    with torch.no_grad():
        if isinstance(model, RatioCritic):
            log_plus = _as_numpy(model.log_ratio(plus, response))
            log_minus = _as_numpy(model.log_ratio(minus, response))
        else:
            log_plus = _as_numpy(model.log_prob(response, plus))
            log_minus = _as_numpy(model.log_prob(response, minus))
    return central_mixture_witness(log_plus, log_minus, delta)


def _context(
    record: Mapping[str, Any] | None,
    dgp: ConditionalDGP,
    method: str,
    delta: float,
    response_noise_sigma: float,
    direction_name: str,
) -> dict[str, Any]:
    model = {} if record is None else record["model"]
    return {
        "case_id": "oracle" if record is None else record["case_id"],
        "generator_id": dgp.name,
        "generator_kind": dgp.name,
        "generator_seed": dgp.seed,
        "data_seed": None if record is None else record["data_seed"],
        "model_name": "oracle_predictive_law" if record is None else model["name"],
        "model_kind": "oracle" if record is None else model["kind"],
        "model_seed": None if record is None else model["seed"],
        "method": method,
        "target": "central_finite_mixture_witness",
        "reference_law": "oracle_absolute_coefficient_mixture",
        "law": "clean" if response_noise_sigma == 0 else "noisy",
        "response_noise_sigma_standardized": float(response_noise_sigma),
        "delta": float(delta),
        "direction": direction_name,
        "channel_dictionary": "fixed_polynomial_cross_cubic_smooth_tail_v1",
    }


def _run_source_adapters(
    *,
    config: FiniteContrastConfig,
    record: Mapping[str, Any] | None,
    dgp: ConditionalDGP,
    scaler: TrainScaler,
    dictionary: FixedResponseDictionary,
    sampler: SampleFunction,
    train_anchors: np.ndarray,
    calibration_anchors: np.ndarray,
    evaluation_anchors: np.ndarray,
    direction_name: str,
    direction: np.ndarray,
    delta: float,
    response_noise_sigma: float,
    panel: Mapping[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    train = _balanced_samples(
        sampler, train_anchors, direction, delta, 1, seed + 11, response_noise_sigma
    )
    calibration = _balanced_samples(
        sampler, calibration_anchors, direction, delta, 1, seed + 21, response_noise_sigma
    )
    classifier = AmortizedSignedClassifier(
        dgp.q,
        dgp.dy,
        hidden=config.classifier_hidden,
        layers=config.classifier_layers,
        seed=seed + 31,
    )
    fit = classifier.fit(
        *train,
        *calibration,
        max_epochs=config.classifier_epochs,
        batch_size=config.classifier_batch_size,
        learning_rate=config.classifier_learning_rate,
        patience=config.classifier_patience,
    )
    classifier_context = _context(
        record, dgp, "signed_classifier", delta, response_noise_sigma, direction_name
    )
    probability = classifier.predict_probability(panel["history"], panel["response"])
    estimate = (2.0 * probability - 1.0) / delta
    rows.extend(_score_witness(classifier_context, estimate, panel))
    labels01 = (np.asarray(panel["labels"]) + 1.0) / 2.0
    clipped = np.clip(probability, EPS, 1.0 - EPS)
    cross_entropy = -float(
        np.mean(labels01 * np.log(clipped) + (1.0 - labels01) * np.log(1.0 - clipped))
    )
    excess = oracle_logistic_excess_risk(panel["true_probability"], probability)
    witness_mse = float(np.mean(np.square(estimate - panel["witness"])))
    bound = 2.0 * (1.0 / delta) ** 2 * excess
    for metric_id, value, unit in (
        ("classifier_cross_entropy", cross_entropy, "nats"),
        ("classifier_brier", float(np.mean(np.square(probability - labels01))), "probability_squared"),
        ("classifier_ece", expected_calibration_error(probability, labels01), "probability"),
        ("classifier_oracle_excess_risk", excess, "nats"),
        ("classifier_witness_bound_ratio", witness_mse / max(bound, EPS), "ratio"),
        ("classifier_saturation", float(np.mean((probability < 0.01) | (probability > 0.99))), "rate"),
        ("classifier_training_seconds", fit.training_seconds, "seconds"),
    ):
        rows.append(_metric_row(classifier_context, metric_id, value, unit, len(probability)))
    diagnostics["classifier"] = asdict(fit)

    source_history, source_response, source_labels = _balanced_samples(
        sampler,
        evaluation_anchors,
        direction,
        delta,
        config.riesz_draws_per_side,
        seed + 41,
        response_noise_sigma,
    )
    direct_context = _context(
        record, dgp, "direct_coupled_samples", delta, response_noise_sigma, direction_name
    )
    rows.extend(
        _direct_channel_rows(
            direct_context, source_response, source_labels, dictionary, panel
        )
    )

    oracle_rows_per_anchor = 2 * int(panel["n_draws"])
    source_rows_per_anchor = 2 * config.riesz_draws_per_side
    riesz_prediction: list[np.ndarray] = []
    conditions: list[float] = []
    norms: list[float] = []
    centerings: list[float] = []
    for anchor_index in range(len(evaluation_anchors)):
        source_slice = np.r_[
            np.arange(
                anchor_index * config.riesz_draws_per_side,
                (anchor_index + 1) * config.riesz_draws_per_side,
            ),
            np.arange(
                len(evaluation_anchors) * config.riesz_draws_per_side
                + anchor_index * config.riesz_draws_per_side,
                len(evaluation_anchors) * config.riesz_draws_per_side
                + (anchor_index + 1) * config.riesz_draws_per_side,
            ),
        ]
        test_slice = np.r_[
            np.arange(
                anchor_index * panel["n_draws"],
                (anchor_index + 1) * panel["n_draws"],
            ),
            np.arange(
                len(evaluation_anchors) * panel["n_draws"]
                + anchor_index * panel["n_draws"],
                len(evaluation_anchors) * panel["n_draws"]
                + (anchor_index + 1) * panel["n_draws"],
            ),
        ]
        assert len(source_slice) == source_rows_per_anchor
        assert len(test_slice) == oracle_rows_per_anchor
        regressor = SignedRieszRegressor(dictionary, ridge=config.riesz_ridge)
        riesz_fit = regressor.fit(
            source_response[source_slice], source_labels[source_slice], 1.0 / delta
        )
        riesz_prediction.append(regressor.predict(panel["response"][test_slice]))
        conditions.append(riesz_fit.condition_number)
        norms.append(riesz_fit.witness_norm_sq)
        centerings.append(riesz_fit.training_centering)
    riesz_estimate = np.empty_like(panel["witness"], dtype=np.float64)
    for anchor_index, prediction in enumerate(riesz_prediction):
        test_slice = np.r_[
            np.arange(anchor_index * panel["n_draws"], (anchor_index + 1) * panel["n_draws"]),
            np.arange(
                len(evaluation_anchors) * panel["n_draws"] + anchor_index * panel["n_draws"],
                len(evaluation_anchors) * panel["n_draws"] + (anchor_index + 1) * panel["n_draws"],
            ),
        ]
        riesz_estimate[test_slice] = prediction
    riesz_context = _context(
        record, dgp, "signed_riesz_dictionary", delta, response_noise_sigma, direction_name
    )
    rows.extend(_score_witness(riesz_context, riesz_estimate, panel))
    for metric_id, value, unit in (
        ("riesz_condition_number_median", float(np.median(conditions)), "condition_number"),
        ("riesz_projected_information_mean", float(np.mean(norms)), "inverse_history_units_squared"),
        ("riesz_training_centering_mean", float(np.mean(centerings)), "inverse_history_units"),
    ):
        rows.append(_metric_row(riesz_context, metric_id, value, unit, len(conditions)))
    diagnostics["riesz"] = {
        "condition_number_median": float(np.median(conditions)),
        "projected_information_mean": float(np.mean(norms)),
        "training_centering_mean": float(np.mean(centerings)),
    }
    return rows, diagnostics


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_markdown_summary(metrics: pd.DataFrame, output: Path) -> Path:
    primary = metrics[metrics["metric_id"].isin(["witness_nrmse", "channel_readout_nrmse"])]
    focus = primary[
        (primary["law"] == "clean")
        & np.isclose(primary["delta"].astype(float), 0.1)
        & (primary["metric_id"] == "witness_nrmse")
        & (primary["method"] != "oracle_infinitesimal_tangent")
    ].copy()
    focus = focus.sort_values(["generator_id", "value", "method"])
    columns = ["generator_id", "model_name", "method", "value", "n"]
    table = focus[columns].to_markdown(index=False) if not focus.empty else "_No matching rows._"
    text = f"""# Revised-note finite-contrast V1 developmental report

> This is a one-generator/data/model-seed developmental comparison. It does not
> support a confirmatory H1--H7, biological, causal, or anatomical claim.

## Run status

- Metric rows: {len(metrics)}
- DGPs: {metrics['generator_id'].nunique()}
- Predictive sources: {metrics['model_name'].nunique()}
- Methods: {metrics['method'].nunique()}
- History scales: {', '.join(f'{value:g}' for value in sorted(metrics['delta'].unique()))}
- Clean/noisy rows remain separate.

## Common-target comparison at delta=0.1, clean law

Lower witness NRMSE is better. Oracle rows are calibration ceilings; they are
not learned-model competitors.

{table}

## Interpretation boundary

All finite methods target the central `h +/- delta*v` contrast under the oracle
absolute-coefficient mixture. Raw history tangents are included only as an
explicit finite-target approximation, so their error includes the scale gap.
The diffusion lane is evaluated only at positive response noise and must not be
ranked against clean-law rows. See `REVISED_NOTE_V1_AUDIT.md` for the frozen
guardrails and theory corrections.
"""
    path = output / "finite_contrast_report.md"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def run_finite_contrast(config_or_path: FiniteContrastConfig | str | Path) -> dict[str, Any]:
    config = (
        config_or_path
        if isinstance(config_or_path, FiniteContrastConfig)
        else load_finite_contrast_config(config_or_path)
    )
    config.validate()
    torch.set_num_threads(1)
    np.random.seed(config.adapter_seed)
    base_output = _resolve_inside(config.base_output)
    output = _resolve_inside(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enforce_disk_safety(
        output, min_free_gb=config.min_free_disk_gb, max_output_gb=config.max_output_gb
    )
    manifest_path = base_output / "run_manifest.json"
    base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("status") not in {"complete", "complete_with_failures"}:
        raise RuntimeError("base benchmark is not complete")
    records = [record for record in load_case_records(base_output) if record["status"] == "ok"]
    if not records:
        raise RuntimeError("base benchmark has no successful records")
    source_digest = source_tree_sha256(PACKAGE_ROOT)
    if base_manifest.get("source_tree_sha256") != source_digest:
        raise RuntimeError(
            "base run source digest differs from current source; rerun the frozen base config"
        )
    note = Path(config.technical_note).expanduser().resolve()
    if not note.is_file():
        raise FileNotFoundError(note)
    started = time.time()
    rows: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []

    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            str(record["generator"]["id"]),
            int(record["generator"]["seed"]),
            int(record["data_seed"]),
        )
        grouped.setdefault(key, []).append(record)

    for group_index, group_records in enumerate(grouped.values()):
        exemplar = group_records[0]
        dgp = make_dgp(
            exemplar["generator"]["kind"],
            seed=int(exemplar["generator"]["seed"]),
            **dict(exemplar["generator"]["params"]),
        )
        first_model, scaler = _load_model(base_output, exemplar, dgp)
        del first_model
        n_train = int(exemplar["diagnostics"]["split_sizes"]["train"])
        training_history, training_response = _training_data(
            dgp, int(exemplar["data_seed"]), n_train
        )
        dictionary = FixedResponseDictionary.fit(training_response)
        direction_name, direction = _direction(dgp, training_history)
        anchor_seed = config.adapter_seed + 100_000 * (group_index + 1)
        train_anchors = _as_numpy(
            dgp.sample_history(config.adapter_train_anchors, anchor_seed + 1)
        )
        calibration_anchors = _as_numpy(
            dgp.sample_history(config.adapter_calibration_anchors, anchor_seed + 2)
        )
        evaluation_anchors = _as_numpy(
            dgp.sample_history(config.evaluation_anchors, anchor_seed + 3)
        )
        covariance = np.atleast_2d(np.cov(training_history, rowvar=False))
        inverse = np.linalg.pinv(covariance)
        mean = np.mean(training_history, axis=0)

        for delta_index, delta in enumerate(config.deltas):
            for noise_index, tau in enumerate(config.response_noise_sigmas):
                panel_seed = anchor_seed + 10_000 * (delta_index + 1) + 1_000 * (noise_index + 1)
                oracle_panel = _panel(
                    dgp,
                    scaler,
                    dictionary,
                    evaluation_anchors,
                    direction,
                    delta,
                    config.evaluation_draws_per_side,
                    panel_seed,
                    tau,
                )
                oracle_context = _context(
                    None,
                    dgp,
                    "oracle_finite_witness",
                    delta,
                    tau,
                    direction_name,
                )
                rows.extend(_score_witness(oracle_context, oracle_panel["witness"], oracle_panel))
                identity_error = _nrmse(
                    np.mean(
                        oracle_panel["channels"] * oracle_panel["witness"][:, None], axis=0
                    ),
                    oracle_panel["direct_readout"],
                )
                rows.append(
                    _metric_row(
                        oracle_context,
                        "oracle_direct_witness_readout_nrmse",
                        identity_error,
                        "ratio",
                        len(dictionary.names),
                    )
                )
                tangent_context = _context(
                    None,
                    dgp,
                    "oracle_infinitesimal_tangent",
                    delta,
                    tau,
                    direction_name,
                )
                rows.extend(
                    _score_witness(
                        tangent_context, oracle_panel["oracle_tangent"], oracle_panel
                    )
                )
                oracle_sampler = lambda h, n, s, noise: _sample_oracle(
                    dgp, scaler, h, n, s, noise
                )
                oracle_adapter_rows, oracle_diagnostics = _run_source_adapters(
                    config=config,
                    record=None,
                    dgp=dgp,
                    scaler=scaler,
                    dictionary=dictionary,
                    sampler=oracle_sampler,
                    train_anchors=train_anchors,
                    calibration_anchors=calibration_anchors,
                    evaluation_anchors=evaluation_anchors,
                    direction_name=direction_name,
                    direction=direction,
                    delta=delta,
                    response_noise_sigma=tau,
                    panel=oracle_panel,
                    seed=panel_seed + 100,
                )
                rows.extend(oracle_adapter_rows)
                diagnostic_records.append(
                    {
                        "case_id": "oracle",
                        "generator_id": exemplar["generator"]["id"],
                        "delta": delta,
                        "response_noise_sigma_standardized": tau,
                        **oracle_diagnostics,
                    }
                )

                for case_index, record in enumerate(group_records):
                    model, model_scaler = _load_model(base_output, record, dgp)
                    if not np.allclose(model_scaler.history_mean, scaler.history_mean) or not np.allclose(
                        model_scaler.response_mean, scaler.response_mean
                    ):
                        raise RuntimeError("models in one generator/data group use different scalers")
                    if isinstance(model, RatioCritic) and tau > 0:
                        continue
                    tangent = _model_tangent(
                        model,
                        scaler,
                        dgp,
                        oracle_panel,
                        direction,
                        tau,
                        config.diffusion_center_draws,
                        panel_seed + 200 + case_index,
                    )
                    if tangent is not None:
                        tangent_context = _context(
                            record,
                            dgp,
                            "history_tangent_finite_target",
                            delta,
                            tau,
                            direction_name,
                        )
                        rows.extend(_score_witness(tangent_context, tangent, oracle_panel))
                    if tau == 0:
                        ratio = _central_model_witness(
                            model, scaler, oracle_panel, direction, delta
                        )
                        if ratio is not None:
                            ratio_context = _context(
                                record,
                                dgp,
                                "central_model_log_ratio",
                                delta,
                                tau,
                                direction_name,
                            )
                            rows.extend(_score_witness(ratio_context, ratio, oracle_panel))
                    if model.capabilities.sampler:
                        model_sampler = lambda h, n, s, noise, m=model: _sample_model(
                            m, scaler, h, n, s, noise
                        )
                        adapter_rows, adapter_diagnostics = _run_source_adapters(
                            config=config,
                            record=record,
                            dgp=dgp,
                            scaler=scaler,
                            dictionary=dictionary,
                            sampler=model_sampler,
                            train_anchors=train_anchors,
                            calibration_anchors=calibration_anchors,
                            evaluation_anchors=evaluation_anchors,
                            direction_name=direction_name,
                            direction=direction,
                            delta=delta,
                            response_noise_sigma=tau,
                            panel=oracle_panel,
                            seed=panel_seed + 1_000 + 100 * case_index,
                        )
                        rows.extend(adapter_rows)
                        diagnostic_records.append(
                            {
                                "case_id": record["case_id"],
                                "generator_id": record["generator"]["id"],
                                "model_name": record["model"]["name"],
                                "delta": delta,
                                "response_noise_sigma_standardized": tau,
                                **adapter_diagnostics,
                            }
                        )
                perturbed = np.concatenate(
                    [evaluation_anchors + delta * direction, evaluation_anchors - delta * direction]
                )
                centered = perturbed - mean
                radii = np.sqrt(np.einsum("ni,ij,nj->n", centered, inverse, centered))
                diagnostic_records.append(
                    {
                        "case_id": "support",
                        "generator_id": exemplar["generator"]["id"],
                        "delta": delta,
                        "response_noise_sigma_standardized": tau,
                        "max_perturbed_history_mahalanobis_radius": float(np.max(radii)),
                    }
                )

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(
        [
            "law",
            "generator_id",
            "delta",
            "model_name",
            "method",
            "metric_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    _write_csv_atomic(metrics, output / "finite_contrast_metrics.csv")
    atomic_json(output / "finite_contrast_diagnostics.json", {"rows": diagnostic_records})
    report_path = _write_markdown_summary(metrics, output)
    manifest = {
        "schema_version": "1",
        "name": config.name,
        "status": "complete",
        "developmental_only": True,
        "hypothesis_decisions": "none",
        "config": asdict(config),
        "config_sha256": stable_sha256(asdict(config)),
        "source_tree_sha256": source_digest,
        "base_manifest_sha256": sha256_file(manifest_path),
        "base_source_tree_sha256": base_manifest["source_tree_sha256"],
        "technical_note_sha256": sha256_file(note),
        "environment": environment_manifest(PACKAGE_ROOT),
        "metric_rows": len(metrics),
        "diagnostic_rows": len(diagnostic_records),
        "started_unix": started,
        "finished_unix": time.time(),
        "artifacts": {
            "metrics": "finite_contrast_metrics.csv",
            "diagnostics": "finite_contrast_diagnostics.json",
            "report": report_path.name,
        },
    }
    atomic_json(output / "run_manifest.json", manifest)
    enforce_disk_safety(
        output, min_free_gb=config.min_free_disk_gb, max_output_gb=config.max_output_gb
    )
    return manifest


__all__ = [
    "FiniteContrastConfig",
    "load_finite_contrast_config",
    "run_finite_contrast",
]
