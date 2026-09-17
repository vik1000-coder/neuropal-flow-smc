"""Capability-aware metrics and numerical helpers for the synthetic benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
import time
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


class MetricStatus(str, Enum):
    OK = "ok"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ModelCapabilities:
    """Native model objects; absent capabilities produce explicit N/A metrics."""

    exact_log_prob: bool = False
    conditional_samples: bool = False
    history_tangent: bool = False
    finite_log_ratio: bool = False
    response_score: bool = False
    pathwise_readouts: bool = False
    probability_flow_likelihood: bool = False

    @property
    def normalized_density(self) -> bool:
        return self.exact_log_prob

    @property
    def sampler(self) -> bool:
        return self.conditional_samples

    @classmethod
    def from_model(cls, model: Any) -> "ModelCapabilities":
        """Adapt either this package's model declarations or a plain object."""

        source = getattr(model, "capabilities", model)
        exact_log_prob = bool(
            getattr(source, "exact_log_prob", getattr(source, "normalized_density", False))
        )
        conditional_samples = bool(
            getattr(source, "conditional_samples", getattr(source, "sampler", False))
        )
        return cls(
            exact_log_prob=exact_log_prob,
            conditional_samples=conditional_samples,
            history_tangent=bool(getattr(source, "history_tangent", False)),
            finite_log_ratio=bool(
                getattr(source, "finite_log_ratio", exact_log_prob)
            ),
            response_score=bool(getattr(source, "response_score", False)),
            pathwise_readouts=bool(getattr(source, "pathwise_readouts", False)),
            probability_flow_likelihood=bool(
                getattr(source, "probability_flow_likelihood", False)
            ),
        )

    def supports(self, capability: str) -> bool:
        if not hasattr(self, capability):
            raise ValueError(f"unknown model capability {capability!r}")
        return bool(getattr(self, capability))


@dataclass(frozen=True)
class MetricValue:
    """A scalar result whose applicability and failure state cannot be ambiguous."""

    metric_id: str
    status: MetricStatus
    value: float | None = None
    reason: str | None = None
    required_capability: str | None = None
    n: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.status, MetricStatus):
            object.__setattr__(self, "status", MetricStatus(self.status))
        if not self.metric_id:
            raise ValueError("metric_id must be nonempty")
        if self.status is MetricStatus.OK:
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, Real)
                or not math.isfinite(float(self.value))
            ):
                raise ValueError("an ok metric requires a finite numeric value")
            if self.reason is not None:
                raise ValueError("an ok metric cannot carry a failure reason")
        else:
            if self.value is not None:
                raise ValueError("a non-ok metric cannot carry a numeric value")
            if not self.reason:
                raise ValueError("a non-ok metric requires a reason")
        if self.n is not None and (isinstance(self.n, bool) or int(self.n) < 0):
            raise ValueError("metric sample size must be nonnegative")

    @classmethod
    def ok(
        cls,
        metric_id: str,
        value: float,
        *,
        n: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MetricValue":
        return cls(
            metric_id=metric_id,
            status=MetricStatus.OK,
            value=float(value),
            n=n,
            metadata=MappingProxyType(dict(metadata or {})),
        )

    @classmethod
    def not_applicable(
        cls, metric_id: str, capability: str, reason: str | None = None
    ) -> "MetricValue":
        return cls(
            metric_id=metric_id,
            status=MetricStatus.NOT_APPLICABLE,
            reason=reason or f"model does not expose {capability}",
            required_capability=capability,
        )

    @classmethod
    def failed(cls, metric_id: str, reason: str) -> "MetricValue":
        return cls(metric_id=metric_id, status=MetricStatus.FAILED, reason=reason)

    @classmethod
    def skipped(cls, metric_id: str, reason: str) -> "MetricValue":
        return cls(metric_id=metric_id, status=MetricStatus.SKIPPED, reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "status": self.status.value,
            "value": self.value,
            "reason": self.reason,
            "required_capability": self.required_capability,
            "n": self.n,
            "metadata": dict(self.metadata),
        }


def metric_for_capability(
    metric_id: str,
    capabilities: ModelCapabilities,
    capability: str,
    compute: Callable[[], float],
    *,
    n: int | None = None,
) -> MetricValue:
    if not isinstance(capabilities, ModelCapabilities):
        capabilities = ModelCapabilities.from_model(capabilities)
    if not capabilities.supports(capability):
        return MetricValue.not_applicable(metric_id, capability)
    try:
        return MetricValue.ok(metric_id, compute(), n=n)
    except Exception as error:
        return MetricValue.failed(metric_id, f"{type(error).__name__}: {error}")


def _numpy_training_array(value: Any, name: str) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two training examples")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _stat_like(statistic: np.ndarray, value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            # Scaler statistics are deliberately read-only; construct a tensor
            # copy so PyTorch never exposes undefined write-through behavior.
            return torch.tensor(statistic, dtype=value.dtype, device=value.device)
    except ImportError:
        pass
    return statistic


def _require_event_shape(value: Any, expected: tuple[int, ...], name: str) -> None:
    shape = tuple(int(item) for item in np.shape(value))
    if expected and (len(shape) < len(expected) or shape[-len(expected) :] != expected):
        raise ValueError(f"{name} must end in event shape {expected}, got {shape}")


@dataclass(frozen=True)
class TrainScaler:
    """Train-only affine scaling with exact density/tangent coordinate maps."""

    history_mean: np.ndarray
    history_scale: np.ndarray
    response_mean: np.ndarray
    response_scale: np.ndarray
    n_train: int
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for mean_name, scale_name in (
            ("history_mean", "history_scale"),
            ("response_mean", "response_scale"),
        ):
            mean = np.asarray(getattr(self, mean_name), dtype=np.float64).copy()
            scale = np.asarray(getattr(self, scale_name), dtype=np.float64).copy()
            if mean.shape != scale.shape:
                raise ValueError(f"{mean_name} and {scale_name} shapes differ")
            if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("scaler statistics must be finite with positive scales")
            mean.setflags(write=False)
            scale.setflags(write=False)
            object.__setattr__(self, mean_name, mean)
            object.__setattr__(self, scale_name, scale)
        if self.n_train < 2:
            raise ValueError("a scaler requires at least two training examples")

    @classmethod
    def fit(
        cls,
        history_train: Any,
        response_train: Any,
        *,
        epsilon: float = 1e-8,
    ) -> "TrainScaler":
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        history = _numpy_training_array(history_train, "history_train")
        response = _numpy_training_array(response_train, "response_train")
        if history.shape[0] != response.shape[0]:
            raise ValueError("history and response training sizes differ")
        history_scale = np.std(history, axis=0, ddof=0)
        response_scale = np.std(response, axis=0, ddof=0)
        return cls(
            history_mean=np.mean(history, axis=0),
            history_scale=np.where(history_scale > epsilon, history_scale, 1.0),
            response_mean=np.mean(response, axis=0),
            response_scale=np.where(response_scale > epsilon, response_scale, 1.0),
            n_train=int(history.shape[0]),
            epsilon=float(epsilon),
        )

    def transform_history(self, history: Any) -> Any:
        _require_event_shape(history, self.history_mean.shape, "history")
        mean = _stat_like(self.history_mean, history)
        scale = _stat_like(self.history_scale, history)
        return (history - mean) / scale

    def inverse_history(self, history_standardized: Any) -> Any:
        _require_event_shape(
            history_standardized, self.history_mean.shape, "history_standardized"
        )
        mean = _stat_like(self.history_mean, history_standardized)
        scale = _stat_like(self.history_scale, history_standardized)
        return history_standardized * scale + mean

    def transform_response(self, response: Any) -> Any:
        _require_event_shape(response, self.response_mean.shape, "response")
        mean = _stat_like(self.response_mean, response)
        scale = _stat_like(self.response_scale, response)
        return (response - mean) / scale

    def inverse_response(self, response_standardized: Any) -> Any:
        _require_event_shape(
            response_standardized, self.response_mean.shape, "response_standardized"
        )
        mean = _stat_like(self.response_mean, response_standardized)
        scale = _stat_like(self.response_scale, response_standardized)
        return response_standardized * scale + mean

    def history_tangent_to_original(self, tangent_standardized: Any) -> Any:
        """Map ``d log q_s / d h_s`` to ``d log q / d h`` exactly."""

        _require_event_shape(
            tangent_standardized, self.history_scale.shape, "tangent_standardized"
        )
        scale = _stat_like(self.history_scale, tangent_standardized)
        return tangent_standardized / scale

    def history_tangent_to_standardized(self, tangent_original: Any) -> Any:
        _require_event_shape(tangent_original, self.history_scale.shape, "tangent_original")
        scale = _stat_like(self.history_scale, tangent_original)
        return tangent_original * scale

    def direction_to_standardized(self, direction_original: Any) -> Any:
        """Map an original-coordinate displacement direction into scaled history."""

        _require_event_shape(direction_original, self.history_scale.shape, "direction_original")
        scale = _stat_like(self.history_scale, direction_original)
        return direction_original / scale

    def direction_to_original(self, direction_standardized: Any) -> Any:
        _require_event_shape(
            direction_standardized, self.history_scale.shape, "direction_standardized"
        )
        scale = _stat_like(self.history_scale, direction_standardized)
        return direction_standardized * scale

    @property
    def response_log_abs_det(self) -> float:
        return float(np.sum(np.log(self.response_scale)))

    def log_prob_to_original(self, standardized_log_prob: Any) -> Any:
        """Apply the response change-of-variables term for a joint density."""

        return standardized_log_prob - self.response_log_abs_det

    def log_prob_to_standardized(self, original_log_prob: Any) -> Any:
        return original_log_prob + self.response_log_abs_det


# Concise compatibility alias.
Standardizer = TrainScaler


def _energy_inputs(observed: Any, samples: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(observed, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)
    if y.ndim < 1 or draws.ndim != y.ndim + 1:
        raise ValueError("samples must have shape [case, draw, *event_shape]")
    if draws.shape[0] != y.shape[0] or draws.shape[2:] != y.shape[1:]:
        raise ValueError("observed and sample event shapes do not match")
    if draws.shape[1] < 1:
        raise ValueError("at least one conditional sample is required")
    if not np.isfinite(y).all() or not np.isfinite(draws).all():
        raise ValueError("energy score inputs must be finite")
    return y.reshape(y.shape[0], -1), draws.reshape(draws.shape[0], draws.shape[1], -1)


def _energy_score(
    observed: Any,
    samples: Any,
    *,
    fair: bool,
    observation_chunk_size: int,
    draw_chunk_size: int,
    reduction: Literal["mean", "none"],
) -> float | np.ndarray:
    y, draws = _energy_inputs(observed, samples)
    n_cases, n_draws, _ = draws.shape
    if fair and n_draws < 2:
        raise ValueError("the fair energy score requires at least two draws")
    if observation_chunk_size < 1 or draw_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    result = np.empty(n_cases, dtype=np.float64)
    pair_denominator = n_draws * (n_draws - 1 if fair else n_draws)
    for case_start in range(0, n_cases, observation_chunk_size):
        case_stop = min(n_cases, case_start + observation_chunk_size)
        y_block = y[case_start:case_stop]
        draws_block = draws[case_start:case_stop]
        first_sum = np.zeros(case_stop - case_start, dtype=np.float64)
        pair_sum = np.zeros(case_stop - case_start, dtype=np.float64)
        for left_start in range(0, n_draws, draw_chunk_size):
            left_stop = min(n_draws, left_start + draw_chunk_size)
            left = draws_block[:, left_start:left_stop]
            first_sum += np.linalg.norm(left - y_block[:, None, :], axis=-1).sum(axis=1)
            for right_start in range(0, n_draws, draw_chunk_size):
                right_stop = min(n_draws, right_start + draw_chunk_size)
                right = draws_block[:, right_start:right_stop]
                distances = np.linalg.norm(
                    left[:, :, None, :] - right[:, None, :, :], axis=-1
                )
                pair_sum += distances.sum(axis=(1, 2))
        result[case_start:case_stop] = (
            first_sum / n_draws - 0.5 * pair_sum / pair_denominator
        )
    if reduction == "none":
        return result
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    return float(np.mean(result))


def energy_score_vstat(
    observed: Any,
    samples: Any,
    *,
    observation_chunk_size: int = 64,
    draw_chunk_size: int = 16,
    reduction: Literal["mean", "none"] = "mean",
) -> float | np.ndarray:
    """The plan-specified V-statistic, including zero self-pairs."""

    return _energy_score(
        observed,
        samples,
        fair=False,
        observation_chunk_size=observation_chunk_size,
        draw_chunk_size=draw_chunk_size,
        reduction=reduction,
    )


def energy_score_fair(
    observed: Any,
    samples: Any,
    *,
    observation_chunk_size: int = 64,
    draw_chunk_size: int = 16,
    reduction: Literal["mean", "none"] = "mean",
) -> float | np.ndarray:
    """Fair finite-ensemble energy score with off-diagonal U-statistic."""

    return _energy_score(
        observed,
        samples,
        fair=True,
        observation_chunk_size=observation_chunk_size,
        draw_chunk_size=draw_chunk_size,
        reduction=reduction,
    )


def tangent_nrmse(estimate: Any, truth: Any, *, epsilon: float = EPS) -> float:
    estimate_array = np.asarray(estimate, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    if estimate_array.shape != truth_array.shape:
        raise ValueError("estimated and true tangent shapes differ")
    return float(
        np.sqrt(
            np.sum((estimate_array - truth_array) ** 2)
            / (np.sum(truth_array**2) + epsilon)
        )
    )


def tangent_cosine(estimate: Any, truth: Any, *, epsilon: float = EPS) -> float:
    estimate_array = np.asarray(estimate, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    if estimate_array.shape != truth_array.shape:
        raise ValueError("estimated and true tangent shapes differ")
    if estimate_array.ndim == 1:
        estimate_array = estimate_array[None, :]
        truth_array = truth_array[None, :]
    estimate_flat = estimate_array.reshape(estimate_array.shape[0], -1)
    truth_flat = truth_array.reshape(truth_array.shape[0], -1)
    numerator = np.sum(estimate_flat * truth_flat, axis=1)
    denominator = (
        np.linalg.norm(estimate_flat, axis=1) * np.linalg.norm(truth_flat, axis=1)
        + epsilon
    )
    return float(np.mean(numerator / denominator))


def conditional_centering_error(
    estimated_tangent_draws: Any,
    *,
    true_tangent_draws: Any | None = None,
    truth_energy: float | None = None,
    epsilon: float = EPS,
) -> float:
    """Conditional tangent centering error over ``[history, draw, ...]`` arrays."""

    estimate = np.asarray(estimated_tangent_draws, dtype=np.float64)
    if estimate.ndim < 3:
        raise ValueError("estimated tangent draws need [history, draw, tangent...] axes")
    conditional_mean = np.mean(estimate, axis=1)
    numerator = float(
        np.mean(np.sum(conditional_mean.reshape(conditional_mean.shape[0], -1) ** 2, axis=1))
    )
    if (true_tangent_draws is None) == (truth_energy is None):
        raise ValueError("provide exactly one of true_tangent_draws or truth_energy")
    if true_tangent_draws is not None:
        truth = np.asarray(true_tangent_draws, dtype=np.float64)
        if truth.shape != estimate.shape:
            raise ValueError("true and estimated conditional tangent arrays differ")
        truth_flat = truth.reshape(truth.shape[0], truth.shape[1], -1)
        denominator = float(np.mean(np.sum(truth_flat**2, axis=-1)))
    else:
        denominator = float(truth_energy)
    if not math.isfinite(denominator) or denominator < 0:
        raise ValueError("truth tangent energy must be finite and nonnegative")
    return float(np.sqrt(numerator / (denominator + epsilon)))


def finite_ratio_nrmse(estimate: Any, truth: Any, *, epsilon: float = EPS) -> float:
    estimate_array = np.asarray(estimate, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    if estimate_array.shape != truth_array.shape:
        raise ValueError("estimated and true finite-ratio shapes differ")
    return float(
        np.sqrt(
            np.mean((estimate_array - truth_array) ** 2)
            / (np.mean(truth_array**2) + epsilon)
        )
    )


def count_parameters(model: Any, *, trainable_only: bool = True) -> int:
    if not hasattr(model, "parameters"):
        raise TypeError("model does not expose parameters()")
    total = 0
    for parameter in model.parameters():
        if not trainable_only or bool(getattr(parameter, "requires_grad", False)):
            total += int(parameter.numel())
    return total


def _synchronize_accelerator() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
    except ImportError:
        pass


@dataclass(frozen=True)
class TimingResult:
    seconds: tuple[float, ...]
    n_examples: int | None = None

    @property
    def repeats(self) -> int:
        return len(self.seconds)

    @property
    def mean_seconds(self) -> float:
        return float(np.mean(self.seconds))

    @property
    def median_seconds(self) -> float:
        return float(np.median(self.seconds))

    @property
    def milliseconds_per_example(self) -> float | None:
        if self.n_examples is None:
            return None
        return 1_000.0 * self.median_seconds / self.n_examples

    @property
    def examples_per_second(self) -> float | None:
        if self.n_examples is None:
            return None
        return self.n_examples / self.median_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeats": self.repeats,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "n_examples": self.n_examples,
            "milliseconds_per_example": self.milliseconds_per_example,
            "examples_per_second": self.examples_per_second,
        }


def timed_call(
    function: Callable[..., Any],
    *args: Any,
    repeats: int = 1,
    warmup: int = 0,
    n_examples: int | None = None,
    synchronize_accelerator: bool = True,
    **kwargs: Any,
) -> tuple[Any, TimingResult]:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    if n_examples is not None and n_examples < 1:
        raise ValueError("n_examples must be positive")
    result = None
    for _ in range(warmup):
        result = function(*args, **kwargs)
    timings = []
    for _ in range(repeats):
        if synchronize_accelerator:
            _synchronize_accelerator()
        started = time.perf_counter()
        result = function(*args, **kwargs)
        if synchronize_accelerator:
            _synchronize_accelerator()
        timings.append(time.perf_counter() - started)
    return result, TimingResult(tuple(timings), n_examples=n_examples)


@dataclass(frozen=True)
class HierarchicalBootstrapResult:
    log_ratio: float
    ratio: float
    ci95_log_ratio: tuple[float, float]
    ci95_ratio: tuple[float, float]
    probability_practical_improvement: float
    practical_improvement: float
    lower_is_better: bool
    statistic: str
    n_generator_seeds: int
    n_data_seed_instances: int
    n_paired_fits: int
    n_bootstrap: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_ratio": self.log_ratio,
            "ratio": self.ratio,
            "ci95_log_ratio": list(self.ci95_log_ratio),
            "ci95_ratio": list(self.ci95_ratio),
            "probability_practical_improvement": self.probability_practical_improvement,
            "practical_improvement": self.practical_improvement,
            "lower_is_better": self.lower_is_better,
            "statistic": self.statistic,
            "n_generator_seeds": self.n_generator_seeds,
            "n_data_seed_instances": self.n_data_seed_instances,
            "n_paired_fits": self.n_paired_fits,
            "n_bootstrap": self.n_bootstrap,
        }


def hierarchical_paired_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    generator_seeds: Sequence[Any],
    model_seeds: Sequence[Any],
    *,
    data_seeds: Sequence[Any] | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    statistic: Literal["median", "mean"] = "median",
    lower_is_better: bool = True,
    practical_improvement: float = 0.0,
) -> HierarchicalBootstrapResult:
    """Generator-first hierarchical bootstrap of paired positive metric ratios.

    Generator seeds are sampled first. Data seeds are resampled within a selected
    generator, and model seeds are resampled within a selected data set. The
    resulting generator effects are combined with the requested statistic.
    """

    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    generators = np.asarray(generator_seeds)
    models = np.asarray(model_seeds)
    data = (
        np.zeros(len(a), dtype=np.int64)
        if data_seeds is None
        else np.asarray(data_seeds)
    )
    if not (a.ndim == b.ndim == generators.ndim == data.ndim == models.ndim == 1):
        raise ValueError("bootstrap inputs must be one-dimensional")
    if not (len(a) == len(b) == len(generators) == len(data) == len(models)) or len(a) == 0:
        raise ValueError("bootstrap inputs must be aligned and nonempty")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("paired ratio metrics must be finite and strictly positive")
    pairs = [
        (str(generator), str(data_seed), str(model))
        for generator, data_seed, model in zip(generators, data, models)
    ]
    if len(pairs) != len(set(pairs)):
        raise ValueError("generator/data/model seed triples must be unique")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'")
    if not 0 <= practical_improvement < 1:
        raise ValueError("practical_improvement must lie in [0, 1)")
    log_ratios = np.log(a / b)
    unique_generators = list(dict.fromkeys(str(value) for value in generators))
    nested: dict[str, dict[str, np.ndarray]] = {}
    for generator in unique_generators:
        generator_mask = np.asarray(
            [str(value) == generator for value in generators], dtype=bool
        )
        generator_data = list(dict.fromkeys(str(value) for value in data[generator_mask]))
        nested[generator] = {}
        for data_seed in generator_data:
            mask = generator_mask & np.asarray(
                [str(value) == data_seed for value in data], dtype=bool
            )
            nested[generator][data_seed] = log_ratios[mask]
    reducer = np.median if statistic == "median" else np.mean
    generator_effects = np.asarray(
        [
            np.mean([np.mean(values) for values in nested[generator].values()])
            for generator in unique_generators
        ]
    )
    point_log = float(reducer(generator_effects))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        selected_generators = rng.integers(0, len(unique_generators), size=len(unique_generators))
        selected_effects = np.empty(len(unique_generators), dtype=np.float64)
        for position, generator_index in enumerate(selected_generators):
            data_groups = list(nested[unique_generators[int(generator_index)]].values())
            selected_data = rng.integers(0, len(data_groups), size=len(data_groups))
            data_effects = np.empty(len(data_groups), dtype=np.float64)
            for data_position, data_index in enumerate(selected_data):
                values = data_groups[int(data_index)]
                selected = values[rng.integers(0, len(values), size=len(values))]
                data_effects[data_position] = float(np.mean(selected))
            selected_effects[position] = float(np.mean(data_effects))
        bootstrap[index] = float(reducer(selected_effects))
    ci_log_array = np.quantile(bootstrap, [0.025, 0.975])
    bootstrap_ratios = np.exp(bootstrap)
    if lower_is_better:
        threshold = 1.0 - practical_improvement
        probability = float(np.mean(bootstrap_ratios <= threshold))
    else:
        threshold = 1.0 + practical_improvement
        probability = float(np.mean(bootstrap_ratios >= threshold))
    return HierarchicalBootstrapResult(
        log_ratio=point_log,
        ratio=float(np.exp(point_log)),
        ci95_log_ratio=(float(ci_log_array[0]), float(ci_log_array[1])),
        ci95_ratio=(float(np.exp(ci_log_array[0])), float(np.exp(ci_log_array[1]))),
        probability_practical_improvement=probability,
        practical_improvement=float(practical_improvement),
        lower_is_better=bool(lower_is_better),
        statistic=statistic,
        n_generator_seeds=len(unique_generators),
        n_data_seed_instances=sum(len(groups) for groups in nested.values()),
        n_paired_fits=len(a),
        n_bootstrap=int(n_bootstrap),
    )


def hierarchical_paired_bootstrap_frame(
    frame: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    value_col: str = "value",
    method_col: str = "method",
    generator_seed_col: str = "generator_seed",
    data_seed_col: str = "data_seed",
    model_seed_col: str = "model_seed",
    **kwargs: Any,
) -> HierarchicalBootstrapResult:
    """DataFrame wrapper using the exact paired seed intersection."""

    required = {
        value_col,
        method_col,
        generator_seed_col,
        data_seed_col,
        model_seed_col,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"bootstrap frame is missing columns {sorted(missing)}")
    selected = frame[frame[method_col].isin([method_a, method_b])]
    keys = [generator_seed_col, data_seed_col, model_seed_col, method_col]
    if selected.duplicated(keys).any():
        raise ValueError("bootstrap frame has duplicate method/seed records")
    a = selected[selected[method_col] == method_a][
        [generator_seed_col, data_seed_col, model_seed_col, value_col]
    ].rename(columns={value_col: "value_a"})
    b = selected[selected[method_col] == method_b][
        [generator_seed_col, data_seed_col, model_seed_col, value_col]
    ].rename(columns={value_col: "value_b"})
    paired = a.merge(
        b, on=[generator_seed_col, data_seed_col, model_seed_col], how="inner"
    )
    if paired.empty:
        raise ValueError("methods have no paired generator/model seed intersection")
    return hierarchical_paired_bootstrap(
        paired["value_a"].to_numpy(),
        paired["value_b"].to_numpy(),
        paired[generator_seed_col].to_numpy(),
        paired[model_seed_col].to_numpy(),
        data_seeds=paired[data_seed_col].to_numpy(),
        **kwargs,
    )


# Readable aliases used by analysis code.
energy_score = energy_score_vstat
fair_energy_score = energy_score_fair
ratio_nrmse = finite_ratio_nrmse


__all__ = [
    "EPS",
    "HierarchicalBootstrapResult",
    "MetricStatus",
    "MetricValue",
    "ModelCapabilities",
    "Standardizer",
    "TimingResult",
    "TrainScaler",
    "conditional_centering_error",
    "count_parameters",
    "energy_score",
    "energy_score_fair",
    "energy_score_vstat",
    "fair_energy_score",
    "finite_ratio_nrmse",
    "hierarchical_paired_bootstrap",
    "hierarchical_paired_bootstrap_frame",
    "metric_for_capability",
    "ratio_nrmse",
    "tangent_cosine",
    "tangent_nrmse",
    "timed_call",
]
