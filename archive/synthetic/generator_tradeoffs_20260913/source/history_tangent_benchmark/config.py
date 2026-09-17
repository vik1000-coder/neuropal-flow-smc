"""Versioned, immutable YAML configuration for the history-tangent benchmark.

Generator-parameter, sampled-data, and model-initialization seeds are deliberately
separate axes. They may use the same integer labels, but are never collapsed into
one replicate identifier.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any
import math
import re

import yaml


_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")
_TIERS = frozenset({"development", "smoke", "core", "stress"})


def _freeze(value: Any) -> Any:
    """Recursively freeze YAML containers without changing scalar values."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_thaw(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must start with a letter and contain only letters, "
            "digits, '.', '_', or '-'"
        )
    return value


def _positive_int(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be an integer")
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be {qualifier}")
    return result


def _seeds(values: Sequence[Any], field_name: str) -> tuple[int, ...]:
    result = tuple(_positive_int(value, field_name, allow_zero=True) for value in values)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique seeds")
    return result


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")


@dataclass(frozen=True)
class ResourceBudget:
    """Safety ceilings applied by benchmark workers and their orchestrator."""

    max_threads: int = 2
    max_parallel_jobs: int = 1
    max_rss_gb: float = 8.0
    max_accelerator_gb: float | None = None
    max_output_gb: float = 2.0
    min_free_disk_gb: float = 4.0
    case_timeout_minutes: float = 180.0
    checkpoint_minutes: float = 20.0
    nice: int = 19

    def validate(self) -> None:
        _positive_int(self.max_threads, "max_threads")
        _positive_int(self.max_parallel_jobs, "max_parallel_jobs")
        for name in (
            "max_rss_gb",
            "max_output_gb",
            "min_free_disk_gb",
            "case_timeout_minutes",
            "checkpoint_minutes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_accelerator_gb is not None:
            if (
                isinstance(self.max_accelerator_gb, bool)
                or not isinstance(self.max_accelerator_gb, Real)
                or not math.isfinite(float(self.max_accelerator_gb))
                or float(self.max_accelerator_gb) <= 0
            ):
                raise ValueError("max_accelerator_gb must be positive and finite")
        if isinstance(self.nice, bool) or not isinstance(self.nice, Integral):
            raise ValueError("nice must be an integer")
        if not -20 <= int(self.nice) <= 20:
            raise ValueError("nice must lie in [-20, 20]")


@dataclass(frozen=True)
class SplitConfig:
    """Frozen split and oracle-cache sizes.

    ``validation_fraction`` is applied to the requested training size, then
    clipped to the registered minimum and maximum.
    """

    validation_fraction: float = 0.2
    validation_min: int = 2_000
    validation_max: int = 20_000
    test_size: int = 20_000
    evaluation_histories: int = 512
    conditional_oracle_draws: int = 1_024

    def validate(self) -> None:
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, Real)
            or not 0 < float(self.validation_fraction) < 1
        ):
            raise ValueError("validation_fraction must lie in (0, 1)")
        for name in (
            "validation_min",
            "validation_max",
            "test_size",
            "evaluation_histories",
            "conditional_oracle_draws",
        ):
            _positive_int(getattr(self, name), name)
        if self.validation_min > self.validation_max:
            raise ValueError("validation_min must not exceed validation_max")

    def validation_size(self, training_size: int) -> int:
        n_train = _positive_int(training_size, "training_size")
        proposed = int(round(self.validation_fraction * n_train))
        return min(self.validation_max, max(self.validation_min, proposed))


@dataclass(frozen=True)
class GeneratorSpec:
    id: str
    kind: str
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    grid: Mapping[str, tuple[Any, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    models: tuple[str, ...] | None = None

    def validate(self) -> None:
        _identifier(self.id, "generator id")
        _identifier(self.kind, f"generator {self.id!r} kind")
        for name, values in self.grid.items():
            _identifier(name, f"generator {self.id!r} grid field")
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"generator {self.id!r} grid {name!r} must be nonempty")
        if self.models is not None:
            if not self.models or len(self.models) != len(set(self.models)):
                raise ValueError(f"generator {self.id!r} models must be unique and nonempty")
            for model in self.models:
                _identifier(model, f"generator {self.id!r} model")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": _thaw(self.params),
            "grid": _thaw(self.grid),
            "models": None if self.models is None else list(self.models),
        }


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    tune: Mapping[str, tuple[Any, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    generators: tuple[str, ...] | None = None
    capabilities: Mapping[str, bool] = field(
        default_factory=lambda: MappingProxyType({})
    )
    parameter_budget: str = "small"

    @property
    def id(self) -> str:
        return self.name

    def validate(self) -> None:
        _identifier(self.name, "model name")
        _identifier(self.kind, f"model {self.name!r} kind")
        _identifier(self.parameter_budget, f"model {self.name!r} parameter_budget")
        for name, values in self.tune.items():
            _identifier(name, f"model {self.name!r} tuning field")
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"model {self.name!r} tuning axis {name!r} must be nonempty")
        if self.generators is not None:
            if not self.generators or len(self.generators) != len(set(self.generators)):
                raise ValueError(
                    f"model {self.name!r} generators must be unique and nonempty"
                )
            for generator in self.generators:
                _identifier(generator, f"model {self.name!r} generator")
        for capability, value in self.capabilities.items():
            _identifier(capability, f"model {self.name!r} capability")
            if not isinstance(value, bool):
                raise ValueError(f"model {self.name!r} capability values must be booleans")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "params": _thaw(self.params),
            "tune": _thaw(self.tune),
            "generators": None if self.generators is None else list(self.generators),
            "capabilities": _thaw(self.capabilities),
            "parameter_budget": self.parameter_budget,
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: str
    name: str
    tier: str
    output_dir: str
    generator_seeds: tuple[int, ...]
    data_seeds: tuple[int, ...]
    model_seeds: tuple[int, ...]
    generators: tuple[GeneratorSpec, ...]
    models: tuple[ModelSpec, ...]
    split: SplitConfig = SplitConfig()
    resource: ResourceBudget = ResourceBudget()
    training: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    evaluation: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def n_seed_triples(self) -> int:
        return len(self.generator_seeds) * len(self.data_seeds) * len(self.model_seeds)

    def iter_seed_triples(self) -> Iterator[tuple[int, int, int]]:
        for generator_seed in self.generator_seeds:
            for data_seed in self.data_seeds:
                for model_seed in self.model_seeds:
                    yield generator_seed, data_seed, model_seed

    def validate(self) -> None:
        if self.schema_version != "1":
            raise ValueError(f"unsupported benchmark schema {self.schema_version!r}")
        _identifier(self.name, "benchmark name")
        if self.tier not in _TIERS:
            raise ValueError(f"tier must be one of {sorted(_TIERS)}")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path")
        _seeds(self.generator_seeds, "generator_seeds")
        _seeds(self.data_seeds, "data_seeds")
        _seeds(self.model_seeds, "model_seeds")
        if not self.generators or not self.models:
            raise ValueError("a benchmark needs at least one generator and one model")
        generator_ids = [spec.id for spec in self.generators]
        model_ids = [spec.name for spec in self.models]
        if len(generator_ids) != len(set(generator_ids)):
            raise ValueError("generator IDs must be unique")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model names must be unique")
        for spec in self.generators:
            spec.validate()
            if spec.models is not None and set(spec.models) - set(model_ids):
                raise ValueError(
                    f"generator {spec.id!r} references unknown models "
                    f"{sorted(set(spec.models) - set(model_ids))}"
                )
        for spec in self.models:
            spec.validate()
            if spec.generators is not None and set(spec.generators) - set(generator_ids):
                raise ValueError(
                    f"model {spec.name!r} references unknown generators "
                    f"{sorted(set(spec.generators) - set(generator_ids))}"
                )
        self.split.validate()
        self.resource.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "tier": self.tier,
            "output_dir": self.output_dir,
            "generator_seeds": list(self.generator_seeds),
            "data_seeds": list(self.data_seeds),
            "model_seeds": list(self.model_seeds),
            "generators": [spec.to_dict() for spec in self.generators],
            "models": [spec.to_dict() for spec in self.models],
            "split": {
                field.name: _thaw(getattr(self.split, field.name))
                for field in fields(self.split)
            },
            "resource": {
                field.name: _thaw(getattr(self.resource, field.name))
                for field in fields(self.resource)
            },
            "training": _thaw(self.training),
            "evaluation": _thaw(self.evaluation),
            "metadata": _thaw(self.metadata),
        }

    @property
    def config_sha256(self) -> str:
        from .serialization import stable_sha256

        return stable_sha256(self.to_dict())


def _generator(raw: Mapping[str, Any]) -> GeneratorSpec:
    _reject_unknown(raw, {"id", "name", "kind", "family", "params", "grid", "models"}, "generator")
    identifier = raw.get("id", raw.get("name"))
    if identifier is None:
        raise ValueError("generator requires id")
    kind = raw.get("kind", raw.get("family", identifier))
    grid_raw = raw.get("grid", {})
    if not isinstance(grid_raw, Mapping):
        raise ValueError(f"generator {identifier!r} grid must be a mapping")
    spec = GeneratorSpec(
        id=str(identifier),
        kind=str(kind),
        params=_freeze(raw.get("params", {})),
        grid=MappingProxyType(
            {str(key): tuple(_freeze(value)) for key, value in grid_raw.items()}
        ),
        models=None if raw.get("models") is None else tuple(str(x) for x in raw["models"]),
    )
    spec.validate()
    return spec


def _model(raw: Mapping[str, Any]) -> ModelSpec:
    _reject_unknown(
        raw,
        {
            "id",
            "name",
            "kind",
            "family",
            "params",
            "tune",
            "generators",
            "capabilities",
            "parameter_budget",
        },
        "model",
    )
    name = raw.get("name", raw.get("id"))
    if name is None:
        raise ValueError("model requires name")
    kind = raw.get("kind", raw.get("family", name))
    tune_raw = raw.get("tune", {})
    if not isinstance(tune_raw, Mapping):
        raise ValueError(f"model {name!r} tune must be a mapping")
    spec = ModelSpec(
        name=str(name),
        kind=str(kind),
        params=_freeze(raw.get("params", {})),
        tune=MappingProxyType(
            {str(key): tuple(_freeze(value)) for key, value in tune_raw.items()}
        ),
        generators=(
            None
            if raw.get("generators") is None
            else tuple(str(x) for x in raw["generators"])
        ),
        capabilities=_freeze(raw.get("capabilities", {})),
        parameter_budget=str(raw.get("parameter_budget", "small")),
    )
    spec.validate()
    return spec


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a frozen suite without silently accepting unknown fields."""

    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark YAML must contain a mapping")
    _reject_unknown(
        raw,
        {
            "schema_version",
            "name",
            "tier",
            "output_dir",
            "generator_seeds",
            "data_seeds",
            "model_seeds",
            "generators",
            "dgps",
            "models",
            "split",
            "resource",
            "training",
            "evaluation",
            "metadata",
        },
        "benchmark",
    )
    if "generators" in raw and "dgps" in raw:
        raise ValueError("use generators, not both generators and dgps")
    missing_seed_axes = [
        name
        for name in ("generator_seeds", "data_seeds", "model_seeds")
        if name not in raw
    ]
    if missing_seed_axes:
        raise ValueError(
            "frozen configs must explicitly declare independent seed axes: "
            f"missing {missing_seed_axes}"
        )
    split_raw = raw.get("split", {})
    resource_raw = raw.get("resource", {})
    if not isinstance(split_raw, Mapping) or not isinstance(resource_raw, Mapping):
        raise ValueError("split and resource must be mappings")
    _reject_unknown(split_raw, {field.name for field in fields(SplitConfig)}, "split")
    _reject_unknown(resource_raw, {field.name for field in fields(ResourceBudget)}, "resource")
    generator_seed_values = raw["generator_seeds"]
    generators_raw = raw.get("generators", raw.get("dgps", ()))
    suite = BenchmarkConfig(
        schema_version=str(raw.get("schema_version", "")),
        name=str(raw.get("name", "")),
        tier=str(raw.get("tier", "smoke")),
        output_dir=str(raw.get("output_dir", f"outputs/{raw.get('name', 'benchmark')}")),
        generator_seeds=_seeds(tuple(generator_seed_values), "generator_seeds"),
        data_seeds=_seeds(tuple(raw["data_seeds"]), "data_seeds"),
        model_seeds=_seeds(tuple(raw["model_seeds"]), "model_seeds"),
        generators=tuple(_generator(item) for item in generators_raw),
        models=tuple(_model(item) for item in raw.get("models", ())),
        split=SplitConfig(**dict(split_raw)),
        resource=ResourceBudget(**dict(resource_raw)),
        training=_freeze(raw.get("training", {})),
        evaluation=_freeze(raw.get("evaluation", {})),
        metadata=_freeze(raw.get("metadata", {})),
    )
    suite.validate()
    return suite


# Familiar aliases for callers coming from the existing benchmark package.
SuiteConfig = BenchmarkConfig
load_suite = load_config


__all__ = [
    "BenchmarkConfig",
    "GeneratorSpec",
    "ModelSpec",
    "ResourceBudget",
    "SplitConfig",
    "SuiteConfig",
    "load_config",
    "load_suite",
]
