"""Resumable developmental runner for the history-tangent benchmark.

The runner intentionally implements only the audited continuous smoke lane.  A
case record never conflates clean and noisy estimands, and absent capabilities
are materialized as explicit not-applicable metric rows.
"""
from __future__ import annotations

from dataclasses import asdict
import itertools
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .config import BenchmarkConfig, GeneratorSpec, ModelSpec, load_config
from .dgps import ConditionalDGP, make_dgp
from .metrics import (
    MetricStatus,
    ModelCapabilities,
    TrainScaler,
    conditional_centering_error,
    energy_score_fair,
    energy_score_vstat,
    finite_ratio_nrmse,
    tangent_cosine,
    tangent_nrmse,
    timed_call,
)
from .models import (
    ConditionalEDMDiffusion,
    ConditionalModel,
    ConditionalVectorDiffusion,
    MeanMLP,
    RatioCritic,
    build_model,
    fit_model,
)
from .serialization import (
    atomic_json,
    enforce_disk_safety,
    ensure_within,
    environment_manifest,
    sha256_file,
    source_tree_sha256,
    stable_sha256,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


def _setting(settings: Mapping[str, Any], name: str, default: Any) -> Any:
    return settings[name] if name in settings else default


def _model_training_setting(
    model: ModelSpec,
    suite_settings: Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    """Resolve a frozen single-value model override before the suite default."""
    if name in model.tune:
        values = tuple(model.tune[name])
        if len(values) != 1:
            raise ValueError(
                f"executed model override {model.name}.{name} must contain exactly one value"
            )
        return values[0]
    return _setting(suite_settings, name, default)


def _expanded_generator_specs(spec: GeneratorSpec) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    base = _plain_mapping(spec.params)
    if not spec.grid:
        yield base, {}
        return
    keys = tuple(spec.grid)
    for values in itertools.product(*(tuple(spec.grid[key]) for key in keys)):
        grid = dict(zip(keys, values, strict=True))
        params = dict(base)
        params.update(grid)
        yield params, grid


def _model_applies(generator: GeneratorSpec, model: ModelSpec) -> bool:
    if generator.models is not None and model.name not in generator.models:
        return False
    if model.generators is not None and generator.id not in model.generators:
        return False
    return True


def _declared_capabilities(model_kind: str) -> dict[str, bool]:
    declarations = {
        "mean_mlp": {
            "normalized_density": False,
            "sampler": True,
            "history_tangent": False,
            "response_score": False,
            "tangent_requires_centering": False,
        },
        "heteroscedastic_gaussian": {
            "normalized_density": True,
            "sampler": True,
            "history_tangent": True,
            "response_score": True,
            "tangent_requires_centering": False,
        },
        "constrained_gaussian_dsm": {
            "normalized_density": True,
            "sampler": True,
            "history_tangent": True,
            "response_score": True,
            "tangent_requires_centering": False,
        },
        "conditional_affine_flow": {
            "normalized_density": True,
            "sampler": True,
            "history_tangent": True,
            "response_score": True,
            "tangent_requires_centering": False,
        },
        "autoregressive_mdn": {
            "normalized_density": True,
            "sampler": True,
            "history_tangent": True,
            "response_score": True,
            "tangent_requires_centering": False,
        },
        "autoregressive_transformer": {
            "normalized_density": True,
            "sampler": True,
            "history_tangent": True,
            "response_score": True,
            "tangent_requires_centering": False,
        },
        "ratio_critic": {
            "normalized_density": False,
            "sampler": False,
            "history_tangent": True,
            "response_score": False,
            "tangent_requires_centering": False,
        },
        "conditional_vector_diffusion": {
            "normalized_density": False,
            "sampler": True,
            "history_tangent": True,
            "response_score": True,
            "tangent_requires_centering": True,
        },
        "conditional_edm_diffusion": {
            "normalized_density": False,
            "sampler": True,
            "history_tangent": True,
            "response_score": True,
            "tangent_requires_centering": True,
        },
        "gaussian_anchored_edm": {
            "normalized_density": False,
            "sampler": True,
            "history_tangent": True,
            "response_score": True,
            "tangent_requires_centering": True,
        },
        "conditional_flow_matching": {
            "normalized_density": False,
            "sampler": True,
            "history_tangent": False,
            "response_score": False,
            "tangent_requires_centering": False,
        },
        "gaussian_source_flow_matching": {
            "normalized_density": False,
            "sampler": True,
            "history_tangent": False,
            "response_score": False,
            "tangent_requires_centering": False,
        },
        "bounded_energy_ratio": {
            "normalized_density": False,
            "sampler": True,
            "history_tangent": True,
            "response_score": True,
            "tangent_requires_centering": True,
        },
        "conditional_point_source_stochastic_interpolant": {
            "normalized_density": False,
            "sampler": True,
            "history_tangent": False,
            "response_score": False,
            "tangent_requires_centering": False,
        },
    }
    if model_kind not in declarations:
        return {}
    return declarations[model_kind]


def _metric(
    metric_id: str,
    value: float | None = None,
    *,
    status: str = "ok",
    estimand: str,
    centering: str = "none",
    unit: str = "dimensionless",
    n: int | None = None,
    reason: str | None = None,
    noise_sigma_standardized: float | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "metric_id": metric_id,
        "status": status,
        "value": None,
        "estimand": estimand,
        "centering": centering,
        "unit": unit,
        "n": n,
        "reason": reason,
    }
    if status == MetricStatus.OK.value:
        if value is None or not math.isfinite(float(value)):
            record["status"] = MetricStatus.FAILED.value
            record["reason"] = "non-finite metric value"
        else:
            record["value"] = float(value)
            record["reason"] = None
    elif not reason:
        raise ValueError("a non-ok metric needs a reason")
    if noise_sigma_standardized is not None:
        record["noise_sigma_standardized"] = float(noise_sigma_standardized)
    return record


def _na(
    metric_id: str,
    capability: str,
    *,
    estimand: str,
    centering: str = "none",
    noise_sigma_standardized: float | None = None,
) -> dict[str, Any]:
    return _metric(
        metric_id,
        status=MetricStatus.NOT_APPLICABLE.value,
        estimand=estimand,
        centering=centering,
        reason=f"model does not expose {capability}",
        noise_sigma_standardized=noise_sigma_standardized,
    )


def _skipped(
    metric_id: str,
    reason: str,
    *,
    estimand: str,
    centering: str = "none",
    noise_sigma_standardized: float | None = None,
) -> dict[str, Any]:
    return _metric(
        metric_id,
        status=MetricStatus.SKIPPED.value,
        estimand=estimand,
        centering=centering,
        reason=reason,
        noise_sigma_standardized=noise_sigma_standardized,
    )


def _sample_pairs(
    dgp: ConditionalDGP,
    n: int,
    *,
    history_seed: int,
    response_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    history = dgp.sample_history(n, history_seed)
    response = dgp.sample_response(history, 1, response_seed)[:, 0, :]
    return _as_numpy(history), _as_numpy(response)


def _device_of(model: ConditionalModel) -> torch.device:
    return next(model.parameters()).device


def _tensor(value: np.ndarray, model: ConditionalModel) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=_device_of(model))


def _estimate_direct_tangent(
    model: ConditionalModel,
    scaler: TrainScaler,
    response: np.ndarray,
    history: np.ndarray,
) -> np.ndarray:
    response_s = _tensor(np.asarray(scaler.transform_response(response)), model)
    history_s = _tensor(np.asarray(scaler.transform_history(history)), model)
    estimate_s = model.history_tangent(response_s, history_s)
    return _as_numpy(scaler.history_tangent_to_original(estimate_s))


def _absolute_rms(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(truth)) ** 2)))


def _normalized_direction(
    dgp: ConditionalDGP, training_history: np.ndarray
) -> tuple[str, np.ndarray]:
    directions = dgp.mechanism_directions()
    if directions:
        name, raw = next(iter(directions.items()))
        vector = _as_numpy(raw).reshape(-1)
    else:
        name = "fallback_axis_0"
        vector = np.zeros(dgp.q, dtype=np.float64)
        vector[0] = 1.0
    if vector.size != dgp.q:
        raise ValueError(f"mechanism direction {name!r} is not length q={dgp.q}")
    covariance = np.cov(np.asarray(training_history), rowvar=False)
    inverse = np.linalg.pinv(np.atleast_2d(covariance))
    norm = float(np.sqrt(vector @ inverse @ vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"mechanism direction {name!r} has zero covariance-metric norm")
    return name, vector / norm


def _conditional_draws(
    dgp: ConditionalDGP,
    history: np.ndarray,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    return _as_numpy(
        dgp.sample_response(
            torch.as_tensor(history, dtype=torch.float64), n_draws, seed
        )
    )


def _noisy_standardized_draws(
    dgp: ConditionalDGP,
    scaler: TrainScaler,
    history: np.ndarray,
    n_draws: int,
    *,
    sigma: float,
    response_seed: int,
    noise_seed: int,
) -> np.ndarray:
    clean = _conditional_draws(dgp, history, n_draws, response_seed)
    clean_s = np.asarray(scaler.transform_response(clean), dtype=np.float32)
    noise = np.random.default_rng(noise_seed).standard_normal(clean_s.shape).astype(np.float32)
    return clean_s + float(sigma) * noise


def _save_torch_atomic(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _generator_state(dgp: ConditionalDGP) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, value in vars(dgp).items():
        if torch.is_tensor(value):
            state[name] = value.detach().cpu()
        elif isinstance(value, (str, int, float, bool, type(None))):
            state[name] = value
        elif isinstance(value, ConditionalDGP):
            state[name] = _generator_state(value)
    return state


def _save_generator_parameters(
    output: Path,
    dgp: ConditionalDGP,
    generator_payload: Mapping[str, Any],
) -> dict[str, str]:
    identifier = stable_sha256(generator_payload)[:16]
    path = output / "generator_parameters" / f"{identifier}.pt"
    if not path.exists():
        digest = _save_torch_atomic(
            path,
            {
                "generator": dict(generator_payload),
                "metadata": dgp.metadata(),
                "state": _generator_state(dgp),
            },
        )
    else:
        digest = sha256_file(path)
    return {"path": str(path.relative_to(output)), "sha256": digest}


def _save_checkpoint(
    output: Path,
    case_id: str,
    model: ConditionalModel,
    scaler: TrainScaler,
    model_payload: Mapping[str, Any],
) -> dict[str, str]:
    path = output / "checkpoints" / f"{case_id}.pt"
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    digest = _save_torch_atomic(
        path,
        {
            "model": dict(model_payload),
            "state_dict": state,
            "scaler": {
                "history_mean": scaler.history_mean,
                "history_scale": scaler.history_scale,
                "response_mean": scaler.response_mean,
                "response_scale": scaler.response_scale,
                "n_train": scaler.n_train,
            },
        },
    )
    return {"path": str(path.relative_to(output)), "sha256": digest}


def _evaluate_sampling(
    model: ConditionalModel,
    scaler: TrainScaler,
    history: np.ndarray,
    response: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    estimand: str = "clean",
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    capabilities = ModelCapabilities.from_model(model)
    if not capabilities.conditional_samples:
        return [
            _na("energy_score_v", "conditional_samples", estimand=estimand),
            _na("energy_score_fair", "conditional_samples", estimand=estimand),
            _na("samples_per_second", "conditional_samples", estimand=estimand),
        ], None
    history_s = _tensor(np.asarray(scaler.transform_history(history)), model)

    def draw() -> torch.Tensor:
        with torch.no_grad():
            return model.sample(history_s, n_samples=n_samples, seed=seed)

    samples_s, timing = timed_call(draw, repeats=1, n_examples=len(history) * n_samples)
    samples = np.asarray(scaler.inverse_response(_as_numpy(samples_s)))
    metrics = [
        _metric(
            "energy_score_v",
            energy_score_vstat(response, samples),
            estimand=estimand,
            unit="response_units",
            n=len(history),
        ),
        _metric(
            "energy_score_fair",
            energy_score_fair(response, samples),
            estimand=estimand,
            unit="response_units",
            n=len(history),
        ),
        _metric(
            "samples_per_second",
            timing.examples_per_second,
            estimand=estimand,
            unit="samples_per_second",
            n=len(history) * n_samples,
        ),
    ]
    return metrics, samples


def _evaluate_clean_tangent(
    dgp: ConditionalDGP,
    model: ConditionalModel,
    scaler: TrainScaler,
    training_history: np.ndarray,
    test_history: np.ndarray,
    test_response: np.ndarray,
    *,
    tangent_pairs: int,
    centering_histories: int,
    centering_draws: int,
    delta: float,
    evaluation_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    capabilities = ModelCapabilities.from_model(model)
    if not capabilities.history_tangent:
        for metric_id in (
            "tangent_nrmse",
            "tangent_cosine",
            "tangent_absolute_rms",
            "centering_error",
            "finite_ratio_nrmse",
            "finite_ratio_absolute_rmse",
            "tangent_eval_ms_per_example",
        ):
            metrics.append(_na(metric_id, "history_tangent", estimand="clean"))
        return metrics, {}

    n_tangent = min(tangent_pairs, len(test_history))
    history = test_history[:n_tangent]
    response = test_response[:n_tangent]
    truth = _as_numpy(
        dgp.history_tangent(
            torch.as_tensor(response, dtype=torch.float64),
            torch.as_tensor(history, dtype=torch.float64),
        )
    )
    estimate, timing = timed_call(
        _estimate_direct_tangent,
        model,
        scaler,
        response,
        history,
        repeats=1,
        n_examples=n_tangent,
    )
    metrics.extend(
        [
            _metric("tangent_nrmse", tangent_nrmse(estimate, truth), estimand="clean", n=n_tangent),
            _metric("tangent_cosine", tangent_cosine(estimate, truth), estimand="clean", n=n_tangent),
            _metric(
                "tangent_absolute_rms",
                _absolute_rms(estimate, truth),
                estimand="clean",
                unit="inverse_history_units",
                n=n_tangent,
            ),
            _metric(
                "tangent_eval_ms_per_example",
                timing.milliseconds_per_example,
                estimand="clean",
                unit="milliseconds_per_example",
                n=n_tangent,
            ),
        ]
    )

    n_histories = min(centering_histories, len(test_history))
    center_history = test_history[:n_histories]
    draws = _conditional_draws(
        dgp, center_history, centering_draws, evaluation_seed + 101
    )
    flat_history = np.repeat(center_history[:, None, :], centering_draws, axis=1).reshape(
        -1, dgp.q
    )
    flat_draws = draws.reshape(-1, dgp.dy)
    estimated_draws = _estimate_direct_tangent(model, scaler, flat_draws, flat_history).reshape(
        n_histories, centering_draws, dgp.q
    )
    true_draws = _as_numpy(
        dgp.history_tangent(
            torch.as_tensor(draws, dtype=torch.float64),
            torch.as_tensor(center_history, dtype=torch.float64),
        )
    )
    metrics.append(
        _metric(
            "centering_error",
            conditional_centering_error(estimated_draws, true_tangent_draws=true_draws),
            estimand="clean",
            n=n_histories * centering_draws,
        )
    )

    direction_name, direction = _normalized_direction(dgp, training_history)
    model_history_plus = history + float(delta) * direction
    true_ratio = _as_numpy(
        dgp.finite_log_ratio(
            torch.as_tensor(response, dtype=torch.float64),
            torch.as_tensor(history, dtype=torch.float64),
            torch.as_tensor(direction, dtype=torch.float64),
            float(delta),
        )
    )
    response_s = _tensor(np.asarray(scaler.transform_response(response)), model)
    history_s = _tensor(np.asarray(scaler.transform_history(history)), model)
    history_plus_s = _tensor(np.asarray(scaler.transform_history(model_history_plus)), model)
    if model.capabilities.normalized_density:
        with torch.no_grad():
            model_ratio = _as_numpy(
                model.log_prob(response_s, history_plus_s) - model.log_prob(response_s, history_s)
            )
    elif isinstance(model, RatioCritic):
        with torch.no_grad():
            model_ratio = _as_numpy(
                model.log_ratio(history_plus_s, response_s) - model.log_ratio(history_s, response_s)
            )
    else:
        model_ratio = None
    if model_ratio is None:
        metrics.extend(
            [
                _na("finite_ratio_nrmse", "finite_log_ratio", estimand="clean"),
                _na("finite_ratio_absolute_rmse", "finite_log_ratio", estimand="clean"),
            ]
        )
    else:
        metrics.extend(
            [
                _metric(
                    "finite_ratio_nrmse",
                    finite_ratio_nrmse(model_ratio, true_ratio),
                    estimand="clean",
                    n=n_tangent,
                ),
                _metric(
                    "finite_ratio_absolute_rmse",
                    _absolute_rms(model_ratio, true_ratio),
                    estimand="clean",
                    unit="log_density_ratio",
                    n=n_tangent,
                ),
            ]
        )
    diagnostics = {
        "finite_ratio_direction": direction_name,
        "finite_ratio_delta": float(delta),
        "finite_ratio_direction_original": direction.tolist(),
    }
    return metrics, diagnostics


def _mixed_tangent_numpy(
    model: ConditionalVectorDiffusion,
    scaler: TrainScaler,
    response_noisy_s: np.ndarray,
    history: np.ndarray,
    center_samples_s: np.ndarray | None,
    *,
    sigma: float,
    quadrature_points: int,
) -> np.ndarray:
    y = _tensor(response_noisy_s, model)
    h = _tensor(np.asarray(scaler.transform_history(history)), model)
    if center_samples_s is None:
        estimate_s = model.mixed_history_tangent_uncentered(
            y, h, sigma, quadrature_points=quadrature_points
        )
    else:
        center = _tensor(center_samples_s, model)
        estimate_s = model.mixed_history_tangent(
            y,
            h,
            sigma,
            center,
            quadrature_points=quadrature_points,
        )
    return _as_numpy(scaler.history_tangent_to_original(estimate_s))


def _evaluate_diffusion(
    dgp: ConditionalDGP,
    model: ConditionalVectorDiffusion,
    scaler: TrainScaler,
    test_history: np.ndarray,
    test_response: np.ndarray,
    *,
    sigma: float,
    tangent_pairs: int,
    centering_histories: int,
    centering_draws: int,
    centering_target_draws: int,
    quadrature_points: int,
    energy_cases: int,
    energy_samples: int,
    evaluation_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics: list[dict[str, Any]] = [
        _na("nll_original", "exact_log_prob", estimand="noisy", noise_sigma_standardized=sigma)
    ]
    n_tangent = min(tangent_pairs, len(test_history))
    history = test_history[:n_tangent]
    clean_response_s = np.asarray(
        scaler.transform_response(test_response[:n_tangent]), dtype=np.float32
    )
    noise = np.random.default_rng(evaluation_seed).standard_normal(clean_response_s.shape).astype(
        np.float32
    )
    response_noisy_s = clean_response_s + float(sigma) * noise
    response_noisy = np.asarray(scaler.inverse_response(response_noisy_s))
    sigma_original = np.asarray(scaler.response_scale) * float(sigma)
    truth = _as_numpy(
        dgp.noisy_history_tangent(
            torch.as_tensor(response_noisy, dtype=torch.float64),
            torch.as_tensor(history, dtype=torch.float64),
            torch.as_tensor(sigma_original, dtype=torch.float64),
        )
    )
    clean_truth_at_noisy_response = _as_numpy(
        dgp.history_tangent(
            torch.as_tensor(response_noisy, dtype=torch.float64),
            torch.as_tensor(history, dtype=torch.float64),
        )
    )
    metrics.append(
        _metric(
            "oracle_clean_to_noisy_tangent_nrmse",
            tangent_nrmse(clean_truth_at_noisy_response, truth),
            estimand="noisy",
            centering="none",
            n=n_tangent,
            noise_sigma_standardized=sigma,
        )
    )

    oracle_center_s = _noisy_standardized_draws(
        dgp,
        scaler,
        history,
        centering_draws,
        sigma=sigma,
        response_seed=evaluation_seed + 11,
        noise_seed=evaluation_seed + 12,
    )
    oracle_estimate, tangent_timing = timed_call(
        _mixed_tangent_numpy,
        model,
        scaler,
        response_noisy_s,
        history,
        oracle_center_s,
        sigma=sigma,
        quadrature_points=quadrature_points,
        repeats=1,
        n_examples=n_tangent,
    )
    history_s = _tensor(np.asarray(scaler.transform_history(history)), model)
    model_center_s = _as_numpy(
        model.sample_noisy(
            history_s, centering_draws, sigma=sigma, seed=evaluation_seed + 21
        )
    )
    model_estimate = _mixed_tangent_numpy(
        model,
        scaler,
        response_noisy_s,
        history,
        model_center_s,
        sigma=sigma,
        quadrature_points=quadrature_points,
    )
    uncentered = _mixed_tangent_numpy(
        model,
        scaler,
        response_noisy_s,
        history,
        None,
        sigma=sigma,
        quadrature_points=quadrature_points,
    )
    for label, estimate in (
        ("oracle_samples", oracle_estimate),
        ("model_samples", model_estimate),
        ("none", uncentered),
    ):
        metrics.extend(
            [
                _metric(
                    "tangent_nrmse",
                    tangent_nrmse(estimate, truth),
                    estimand="noisy",
                    centering=label,
                    n=n_tangent,
                    noise_sigma_standardized=sigma,
                ),
                _metric(
                    "tangent_cosine",
                    tangent_cosine(estimate, truth),
                    estimand="noisy",
                    centering=label,
                    n=n_tangent,
                    noise_sigma_standardized=sigma,
                ),
                _metric(
                    "tangent_absolute_rms",
                    _absolute_rms(estimate, truth),
                    estimand="noisy",
                    centering=label,
                    unit="inverse_history_units",
                    n=n_tangent,
                    noise_sigma_standardized=sigma,
                ),
            ]
        )
    metrics.append(
        _metric(
            "tangent_eval_ms_per_example",
            tangent_timing.milliseconds_per_example,
            estimand="noisy",
            centering="oracle_samples",
            unit="milliseconds_per_example",
            n=n_tangent,
            noise_sigma_standardized=sigma,
        )
    )

    predicted_score_s = _as_numpy(
        model.response_score(
            _tensor(response_noisy_s, model),
            _tensor(np.asarray(scaler.transform_history(history)), model),
            sigma,
        )
    )
    predicted_score = predicted_score_s / np.asarray(scaler.response_scale)
    true_score = _as_numpy(
        dgp.noisy_response_score(
            torch.as_tensor(response_noisy, dtype=torch.float64),
            torch.as_tensor(history, dtype=torch.float64),
            torch.as_tensor(sigma_original, dtype=torch.float64),
        )
    )
    metrics.append(
        _metric(
            "response_score_nrmse",
            tangent_nrmse(predicted_score, true_score),
            estimand="noisy",
            n=n_tangent,
            noise_sigma_standardized=sigma,
        )
    )

    n_histories = min(centering_histories, len(test_history))
    center_history = test_history[:n_histories]
    target_s = _noisy_standardized_draws(
        dgp,
        scaler,
        center_history,
        centering_target_draws,
        sigma=sigma,
        response_seed=evaluation_seed + 31,
        noise_seed=evaluation_seed + 32,
    )
    target_original = np.asarray(scaler.inverse_response(target_s))
    true_draws = _as_numpy(
        dgp.noisy_history_tangent(
            torch.as_tensor(target_original, dtype=torch.float64),
            torch.as_tensor(center_history, dtype=torch.float64),
            torch.as_tensor(sigma_original, dtype=torch.float64),
        )
    )
    independent_oracle_center_s = _noisy_standardized_draws(
        dgp,
        scaler,
        center_history,
        centering_draws,
        sigma=sigma,
        response_seed=evaluation_seed + 41,
        noise_seed=evaluation_seed + 42,
    )
    center_history_s = _tensor(np.asarray(scaler.transform_history(center_history)), model)
    independent_model_center_s = _as_numpy(
        model.sample_noisy(
            center_history_s,
            centering_draws,
            sigma=sigma,
            seed=evaluation_seed + 43,
        )
    )
    flat_target_s = target_s.reshape(-1, dgp.dy)
    flat_history = np.repeat(
        center_history[:, None, :], centering_target_draws, axis=1
    ).reshape(-1, dgp.q)
    for label, pool in (
        ("oracle_samples", independent_oracle_center_s),
        ("model_samples", independent_model_center_s),
    ):
        repeated_pool = np.repeat(
            pool[:, None, :, :], centering_target_draws, axis=1
        ).reshape(-1, centering_draws, dgp.dy)
        estimated_draws = _mixed_tangent_numpy(
            model,
            scaler,
            flat_target_s,
            flat_history,
            repeated_pool,
            sigma=sigma,
            quadrature_points=quadrature_points,
        ).reshape(n_histories, centering_target_draws, dgp.q)
        metrics.append(
            _metric(
                "centering_error",
                conditional_centering_error(
                    estimated_draws, true_tangent_draws=true_draws
                ),
                estimand="noisy",
                centering=label,
                n=n_histories * centering_target_draws,
                noise_sigma_standardized=sigma,
            )
        )

    metrics.extend(
        [
            _skipped(
                "finite_ratio_nrmse",
                "developmental M5b finite-path integration is not implemented",
                estimand="noisy",
                centering="oracle_samples",
                noise_sigma_standardized=sigma,
            ),
            _skipped(
                "finite_ratio_absolute_rmse",
                "developmental M5b finite-path integration is not implemented",
                estimand="noisy",
                centering="oracle_samples",
                noise_sigma_standardized=sigma,
            ),
        ]
    )

    n_energy = min(energy_cases, len(test_history))
    sampling_metrics, _ = _evaluate_sampling(
        model,
        scaler,
        test_history[:n_energy],
        test_response[:n_energy],
        n_samples=energy_samples,
        seed=evaluation_seed + 51,
        estimand="approximate_clean_sampler",
    )
    metrics.extend(sampling_metrics)
    diagnostics = {
        "diffusion_sampler": "annealed_langevin_developmental",
        "diffusion_sigma_standardized": float(sigma),
        "diffusion_sigma_original": sigma_original.tolist(),
        "quadrature_points": int(quadrature_points),
        "oracle_center_draws": int(centering_draws),
        "model_center_draws": int(centering_draws),
    }
    return metrics, diagnostics


def _evaluate_case(
    dgp: ConditionalDGP,
    model: ConditionalModel,
    scaler: TrainScaler,
    trace: Any,
    train_history: np.ndarray,
    test_history: np.ndarray,
    test_response: np.ndarray,
    config: BenchmarkConfig,
    *,
    evaluation_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation = config.evaluation
    energy_cases = int(_setting(evaluation, "energy_cases", 64))
    energy_samples = int(_setting(evaluation, "energy_samples", 32))
    tangent_pairs = int(_setting(evaluation, "tangent_pairs", 64))
    centering_histories = int(_setting(evaluation, "centering_histories", 8))
    centering_draws = int(_setting(evaluation, "centering_draws", 32))
    centering_target_draws = int(_setting(evaluation, "centering_target_draws", 8))
    finite_delta = float(_setting(evaluation, "finite_delta", 0.1))
    metrics = [
        _metric(
            "native_validation_loss",
            min(trace.validation_loss),
            estimand="native_objective",
            n=len(trace.validation_loss),
        )
    ]
    diagnostics: dict[str, Any] = {}
    n_energy = min(energy_cases, len(test_history))

    if isinstance(model, ConditionalVectorDiffusion):
        diffusion_metrics, diffusion_diagnostics = _evaluate_diffusion(
            dgp,
            model,
            scaler,
            test_history,
            test_response,
            sigma=float(_setting(evaluation, "diffusion_sigma", 0.1)),
            tangent_pairs=tangent_pairs,
            centering_histories=centering_histories,
            centering_draws=centering_draws,
            centering_target_draws=centering_target_draws,
            quadrature_points=int(_setting(evaluation, "quadrature_points", 4)),
            energy_cases=energy_cases,
            energy_samples=energy_samples,
            evaluation_seed=evaluation_seed,
        )
        metrics.extend(diffusion_metrics)
        diagnostics.update(diffusion_diagnostics)
        return metrics, diagnostics

    capabilities = ModelCapabilities.from_model(model)
    if capabilities.exact_log_prob:
        history_s = _tensor(
            np.asarray(scaler.transform_history(test_history[:n_energy])), model
        )
        response_s = _tensor(
            np.asarray(scaler.transform_response(test_response[:n_energy])), model
        )
        with torch.no_grad():
            log_prob_s = _as_numpy(model.log_prob(response_s, history_s))
        log_prob = scaler.log_prob_to_original(log_prob_s)
        metrics.append(
            _metric(
                "nll_original",
                -float(np.mean(log_prob)),
                estimand="clean",
                unit="nats_per_response_vector",
                n=n_energy,
            )
        )
    else:
        metrics.append(_na("nll_original", "exact_log_prob", estimand="clean"))

    sampling_metrics, _ = _evaluate_sampling(
        model,
        scaler,
        test_history[:n_energy],
        test_response[:n_energy],
        n_samples=energy_samples,
        seed=evaluation_seed + 1,
        estimand="clean",
    )
    metrics.extend(sampling_metrics)
    tangent_metrics, tangent_diagnostics = _evaluate_clean_tangent(
        dgp,
        model,
        scaler,
        train_history,
        test_history,
        test_response,
        tangent_pairs=tangent_pairs,
        centering_histories=centering_histories,
        centering_draws=centering_draws,
        delta=finite_delta,
        evaluation_seed=evaluation_seed,
    )
    metrics.extend(tangent_metrics)
    diagnostics.update(tangent_diagnostics)
    if isinstance(model, RatioCritic):
        diagnostics["ratio_classifier"] = model.diagnostics(
            _tensor(np.asarray(scaler.transform_history(test_history[:n_energy])), model),
            _tensor(np.asarray(scaler.transform_response(test_response[:n_energy])), model),
        )
    return metrics, diagnostics


def _case_payload(
    config: BenchmarkConfig,
    generator: GeneratorSpec,
    generator_params: Mapping[str, Any],
    generator_seed: int,
    data_seed: int,
    model: ModelSpec,
    model_seed: int,
    source_digest: str,
) -> dict[str, Any]:
    return {
        "config_sha256": config.config_sha256,
        "source_tree_sha256": source_digest,
        "generator": {
            "id": generator.id,
            "kind": generator.kind,
            "params": dict(generator_params),
            "seed": generator_seed,
        },
        "data_seed": data_seed,
        "model": {
            "name": model.name,
            "kind": model.kind,
            "params": _plain_mapping(model.params),
            "training_overrides": {
                key: list(values) for key, values in model.tune.items()
            },
            "seed": model_seed,
        },
    }


def _validate_existing_record(path: Path, case_id: str) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("case_id") == case_id and record.get("status") in {"ok", "failed"}


def _expected_cases(config: BenchmarkConfig) -> int:
    count = 0
    for generator in config.generators:
        configurations = sum(1 for _ in _expanded_generator_specs(generator))
        methods = sum(_model_applies(generator, model) for model in config.models)
        count += configurations * methods * config.n_seed_triples
    return count


def run_benchmark(
    config_or_path: BenchmarkConfig | str | Path,
    max_cases: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the audited developmental matrix and atomically persist every case."""
    config = (
        config_or_path
        if isinstance(config_or_path, BenchmarkConfig)
        else load_config(config_or_path)
    )
    config.validate()
    if config.tier not in {"development", "smoke"}:
        raise RuntimeError(
            "this local runner is intentionally limited to development/smoke tiers; "
            "freeze an external-compute runner before core or stress execution"
        )
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive")
    output = ensure_within(PACKAGE_ROOT, config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enforce_disk_safety(
        output,
        min_free_gb=config.resource.min_free_disk_gb,
        max_output_gb=config.resource.max_output_gb,
    )
    source_digest = source_tree_sha256(PACKAGE_ROOT)
    environment = environment_manifest(PACKAGE_ROOT)
    manifest_path = output / "run_manifest.json"
    results_dir = output / "results"
    existing_completed = 0
    existing_failed = 0
    if manifest_path.exists() and any((output / "results").glob("*.json")):
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config.config_sha256:
            raise RuntimeError("output contains a different frozen configuration")
        if previous.get("source_tree_sha256") != source_digest:
            raise RuntimeError("output contains results from a different source digest")
        if previous.get("environment", {}).get("environment_sha256") != environment.get(
            "environment_sha256"
        ):
            raise RuntimeError("output contains results from a different execution environment")
        if resume:
            for path in results_dir.glob("*.json"):
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if existing.get("case_id") != path.stem:
                    continue
                if existing.get("status") == "ok":
                    existing_completed += 1
                elif existing.get("status") == "failed":
                    existing_failed += 1
    manifest: dict[str, Any] = {
        "schema_version": "1",
        "name": config.name,
        "tier": config.tier,
        "status": "running",
        "config": config.to_dict(),
        "config_sha256": config.config_sha256,
        "source_tree_sha256": source_digest,
        "environment": environment,
        "expected_cases": _expected_cases(config),
        "completed": existing_completed,
        "failed": existing_failed,
        "skipped_existing": 0,
        "started_unix": time.time(),
        "hypothesis_decisions": {f"H{index}": "not_evaluated" for index in range(1, 8)},
    }
    atomic_json(manifest_path, manifest)
    results_dir.mkdir(parents=True, exist_ok=True)
    attempted = 0

    for generator_spec in config.generators:
        for generator_params, grid_values in _expanded_generator_specs(generator_spec):
            for generator_seed, data_seed, model_seed in config.iter_seed_triples():
                dgp = make_dgp(
                    generator_spec.kind, seed=generator_seed, **generator_params
                )
                generator_payload = {
                    "id": generator_spec.id,
                    "kind": generator_spec.kind,
                    "params": generator_params,
                    "grid": grid_values,
                    "seed": generator_seed,
                }
                generator_artifact = _save_generator_parameters(
                    output, dgp, generator_payload
                )
                n_train = int(_setting(config.training, "n_train", 2_000))
                n_validation = config.split.validation_size(n_train)
                train_h, train_y = _sample_pairs(
                    dgp,
                    n_train,
                    history_seed=data_seed + 10_001,
                    response_seed=data_seed + 10_002,
                )
                validation_h, validation_y = _sample_pairs(
                    dgp,
                    n_validation,
                    history_seed=data_seed + 20_001,
                    response_seed=data_seed + 20_002,
                )
                test_h, test_y = _sample_pairs(
                    dgp,
                    config.split.test_size,
                    history_seed=data_seed + 30_001,
                    response_seed=data_seed + 30_002,
                )
                scaler = TrainScaler.fit(train_h, train_y)

                for model_spec in config.models:
                    if not _model_applies(generator_spec, model_spec):
                        continue
                    if max_cases is not None and attempted >= max_cases:
                        manifest["status"] = "partial_max_cases"
                        manifest["finished_unix"] = time.time()
                        atomic_json(manifest_path, manifest)
                        _summarize_if_available(output)
                        return manifest
                    enforce_disk_safety(
                        output,
                        min_free_gb=config.resource.min_free_disk_gb,
                        max_output_gb=config.resource.max_output_gb,
                    )
                    payload = _case_payload(
                        config,
                        generator_spec,
                        generator_params,
                        generator_seed,
                        data_seed,
                        model_spec,
                        model_seed,
                        source_digest,
                    )
                    case_id = stable_sha256(payload)[:20]
                    result_path = results_dir / f"{case_id}.json"
                    if resume and result_path.exists() and _validate_existing_record(
                        result_path, case_id
                    ):
                        manifest["skipped_existing"] += 1
                        continue
                    attempted += 1
                    record: dict[str, Any] = {
                        "schema_version": "1",
                        "case_id": case_id,
                        "status": "running",
                        "generator": payload["generator"],
                        "data_seed": data_seed,
                        "model": payload["model"],
                        "fit": {},
                        "metrics": [],
                        "diagnostics": {
                            "generator_metadata": dgp.metadata(),
                            "split_sizes": {
                                "train": n_train,
                                "validation": n_validation,
                                "test": len(test_h),
                            },
                        },
                        "artifacts": {"generator_parameters": generator_artifact},
                        "provenance": {
                            "config_sha256": config.config_sha256,
                            "source_tree_sha256": source_digest,
                            "environment_sha256": environment["environment_sha256"],
                        },
                        "started_unix": time.time(),
                    }
                    record["model"]["capabilities"] = _declared_capabilities(
                        model_spec.kind
                    )
                    atomic_json(result_path, record)
                    try:
                        # Initialization is part of the model-seed axis.  Seed
                        # before constructing modules, not only before fitting.
                        torch.manual_seed(model_seed)
                        np.random.seed(model_seed)
                        model = build_model(
                            model_spec.kind,
                            q=dgp.q,
                            dy=dgp.dy,
                            params=_plain_mapping(model_spec.params),
                        )
                        trace = fit_model(
                            model,
                            np.asarray(scaler.transform_history(train_h), dtype=np.float32),
                            np.asarray(scaler.transform_response(train_y), dtype=np.float32),
                            np.asarray(
                                scaler.transform_history(validation_h), dtype=np.float32
                            ),
                            np.asarray(
                                scaler.transform_response(validation_y), dtype=np.float32
                            ),
                            seed=model_seed,
                            device=str(
                                _model_training_setting(
                                    model_spec, config.training, "device", "auto"
                                )
                            ),
                            learning_rate=float(
                                _model_training_setting(
                                    model_spec,
                                    config.training,
                                    "learning_rate",
                                    3e-4,
                                )
                            ),
                            weight_decay=float(
                                _model_training_setting(
                                    model_spec,
                                    config.training,
                                    "weight_decay",
                                    1e-4,
                                )
                            ),
                            batch_size=int(
                                _model_training_setting(
                                    model_spec, config.training, "batch_size", 256
                                )
                            ),
                            max_epochs=int(
                                _model_training_setting(
                                    model_spec, config.training, "max_epochs", 10
                                )
                            ),
                            patience=int(
                                _model_training_setting(
                                    model_spec, config.training, "patience", 5
                                )
                            ),
                            gradient_clip=float(
                                _model_training_setting(
                                    model_spec,
                                    config.training,
                                    "gradient_clip",
                                    1.0,
                                )
                            ),
                            history_score_matching_weight=float(
                                _model_training_setting(
                                    model_spec,
                                    config.training,
                                    "history_score_matching_weight",
                                    0.0,
                                )
                            ),
                        )
                        metrics, diagnostics = _evaluate_case(
                            dgp,
                            model,
                            scaler,
                            trace,
                            train_h,
                            test_h,
                            test_y,
                            config,
                            evaluation_seed=data_seed + model_seed + 40_001,
                        )
                        checkpoint = _save_checkpoint(
                            output,
                            case_id,
                            model,
                            scaler,
                            payload["model"],
                        )
                        record["status"] = "ok"
                        record["fit"] = trace.to_dict()
                        record["metrics"] = metrics
                        record["diagnostics"].update(diagnostics)
                        record["model"]["capabilities"] = asdict(model.capabilities)
                        record["artifacts"]["checkpoint"] = checkpoint
                        manifest["completed"] += 1
                    except Exception as error:
                        record["status"] = "failed"
                        record["error"] = {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(limit=24),
                        }
                        manifest["failed"] += 1
                    record["finished_unix"] = time.time()
                    atomic_json(result_path, record)
                    atomic_json(manifest_path, manifest)

    manifest["status"] = "complete" if manifest["failed"] == 0 else "complete_with_failures"
    manifest["finished_unix"] = time.time()
    atomic_json(manifest_path, manifest)
    _summarize_if_available(output)
    return manifest


def _summarize_if_available(output: Path) -> None:
    try:
        from .reporting import summarize_output

        summarize_output(output)
    except ImportError:
        return


def validate_oracles(
    config_or_path: BenchmarkConfig | str | Path | None = None,
) -> dict[str, Any]:
    """Run a lightweight independent numerical validation of configured oracles."""
    if config_or_path is None:
        specifications = [
            ("g1", {"dy": 4}),
            ("g2", {"dy": 4, "alpha_m": 0.0}),
            ("g3", {"dy": 1}),
            ("g4", {"dy": 4, "n_components": 4, "mode_separation": 4.0}),
        ]
        output = None
    else:
        config = (
            config_or_path
            if isinstance(config_or_path, BenchmarkConfig)
            else load_config(config_or_path)
        )
        specifications = [
            (spec.kind, params)
            for spec in config.generators
            for params, _ in _expanded_generator_specs(spec)
        ]
        output = ensure_within(PACKAGE_ROOT, config.output_dir)
    rows = []
    for index, (kind, params) in enumerate(specifications):
        dgp = make_dgp(kind, seed=700 + index, **params)
        history = dgp.sample_history(8, 800 + index)
        response = dgp.sample_response(history, 1, 900 + index)[:, 0, :]
        tangent = dgp.history_tangent(response, history)
        direction = torch.randn(dgp.q, generator=torch.Generator().manual_seed(1000 + index), dtype=torch.float64)
        direction = direction / direction.norm()
        epsilon = 1e-5
        finite = (
            dgp.log_prob(response, history + epsilon * direction)
            - dgp.log_prob(response, history - epsilon * direction)
        ) / (2 * epsilon)
        directional = tangent @ direction
        relative_fd_error = float(
            torch.linalg.norm(finite - directional)
            / torch.linalg.norm(directional).clamp_min(1e-12)
        )
        center_history = dgp.sample_history(4, 1100 + index)
        draws = dgp.sample_response(center_history, 2048, 1200 + index)
        tangent_draws = dgp.history_tangent(draws, center_history)
        conditional_mean = tangent_draws.mean(dim=1)
        centering_error = float(
            torch.sqrt(
                conditional_mean.square().sum(dim=-1).mean()
                / tangent_draws.square().sum(dim=-1).mean().clamp_min(1e-12)
            )
        )
        ratio = dgp.finite_log_ratio(response, history, direction, 1e-4) / 1e-4
        ratio_limit_error = float(
            torch.linalg.norm(ratio - directional)
            / torch.linalg.norm(directional).clamp_min(1e-12)
        )
        rows.append(
            {
                "generator": kind,
                "params": params,
                "finite_difference_relative_error": relative_fd_error,
                "centering_relative_error": centering_error,
                "finite_ratio_limit_relative_error": ratio_limit_error,
                "passed": bool(
                    relative_fd_error < 1e-6
                    and centering_error < 0.08
                    and ratio_limit_error < 0.01
                ),
            }
        )
    result = {
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "oracle_calculations_dtype": "torch.float64",
        "centering_draws_per_history": 2048,
        "rows": rows,
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "oracle_validation.json", result)
    return result


__all__ = ["run_benchmark", "validate_oracles"]
