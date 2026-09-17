"""Versioned YAML suite configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

from .schema import ResourceBudget


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    family: str
    params: Mapping[str, Any]
    n_trajectories: int = 6
    views: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MethodSpec:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    tune: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    views: tuple[str, ...] | None = None
    horizons: tuple[int, ...] | None = None
    scenarios: tuple[str, ...] | None = None
    seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ResponseOperationSpec:
    id: str
    kind: str
    modulator_index: int | None = None
    target_neurons: tuple[int, ...] = ()
    source_neurons: tuple[int, ...] = ()
    amplitude: float = 0.0
    duration_steps: int | None = None
    selection: str | None = None
    expected_effect: str | None = None

    @property
    def effect_role(self) -> str:
        """Registered active/null role, with a safe default for old YAML."""

        if self.expected_effect is not None:
            return self.expected_effect
        return "null" if self.kind == "null" else "active"

    def validate(self) -> None:
        """Validate the operation before it is bound to a simulator instance.

        This layer checks the declarative contract. Simulator-dependent upper
        bounds for neuron and modulator indices are checked after binding.
        """

        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.id or "") is None:
            raise ValueError(
                "response-kernel operation IDs must be identifier-safe "
                "([A-Za-z][A-Za-z0-9_]*)"
            )
        allowed = {
            "null",
            "ligand_pulse",
            "receptor_knockout",
            "release_source_silencing",
        }
        if not isinstance(self.kind, str) or self.kind not in allowed:
            raise ValueError(f"unknown response-kernel operation {self.kind!r}")
        if not (
            self.expected_effect is None
            or isinstance(self.expected_effect, str)
            and self.expected_effect in {"active", "null"}
        ):
            raise ValueError("expected_effect must be 'active' or 'null'")
        if self.kind == "null" and self.effect_role != "null":
            raise ValueError("the null operation must have expected_effect='null'")
        if not (
            self.selection is None
            or isinstance(self.selection, str)
            and self.selection
            in {
                "first_active",
                "first_inactive",
                "first_active_or_expressed_control",
            }
        ):
            raise ValueError("unknown response-kernel intervention selection rule")

        if isinstance(self.modulator_index, bool) or (
            self.modulator_index is not None
            and (
                not isinstance(self.modulator_index, Integral)
                or self.modulator_index < 0
            )
        ):
            raise ValueError("modulator_index must be a nonnegative integer")
        if isinstance(self.amplitude, bool) or not isinstance(self.amplitude, Real):
            raise ValueError("intervention amplitude must be numeric")
        if not math.isfinite(float(self.amplitude)):
            raise ValueError("intervention amplitude must be finite")
        if self.duration_steps is not None and (
            isinstance(self.duration_steps, bool)
            or not isinstance(self.duration_steps, Integral)
            or self.duration_steps < 1
        ):
            raise ValueError("duration_steps must be a positive integer")
        for field_name, indices in (
            ("target_neurons", self.target_neurons),
            ("source_neurons", self.source_neurons),
        ):
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in indices
            ):
                raise ValueError(f"{field_name} must contain integer indices")
            if len(set(indices)) != len(indices) or any(value < 0 for value in indices):
                raise ValueError(f"{field_name} must contain unique nonnegative indices")

        if self.kind == "null":
            if (
                self.modulator_index is not None
                or self.target_neurons
                or self.source_neurons
                or self.amplitude != 0.0
                or self.duration_steps is not None
                or self.selection is not None
            ):
                raise ValueError("null operation cannot carry intervention fields")
            return

        if self.modulator_index is None:
            raise ValueError(f"{self.kind} requires modulator_index")
        if self.kind == "ligand_pulse":
            if self.amplitude <= 0.0 or self.duration_steps is None:
                raise ValueError(
                    "ligand_pulse requires positive finite amplitude and duration_steps"
                )
            if self.target_neurons or self.source_neurons or self.selection is not None:
                raise ValueError("ligand_pulse forbids target/source indices and selection")
            return

        if self.amplitude != 0.0:
            raise ValueError(f"{self.kind} forbids amplitude")
        if self.kind == "receptor_knockout":
            if self.source_neurons:
                raise ValueError("receptor_knockout forbids source_neurons")
            explicit = bool(self.target_neurons)
        else:
            if self.target_neurons:
                raise ValueError("release_source_silencing forbids target_neurons")
            explicit = bool(self.source_neurons)
        if explicit == (self.selection is not None):
            raise ValueError(
                f"{self.kind} requires exactly one of explicit indices or selection"
            )
        if self.effect_role == "active" and self.selection == "first_inactive":
            raise ValueError("an active operation cannot select first_inactive")
        if self.effect_role == "null" and self.selection == "first_active":
            raise ValueError("a sham operation cannot select first_active")


@dataclass(frozen=True)
class ResponseKernelEvaluationSpec:
    horizon: int
    n_pairs: int
    operations: tuple[ResponseOperationSpec, ...]
    minimum_active_truth_rms_to_innovation_rms: float = 1e-4
    maximum_truth_mc_se_to_truth_rms_ratio: float = 10.0
    minimum_transient_decay_estimable_fraction: float = 0.0

    def validate(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, Integral):
            raise ValueError("response-kernel horizon must be an integer")
        if isinstance(self.n_pairs, bool) or not isinstance(self.n_pairs, Integral):
            raise ValueError("response-kernel n_pairs must be an integer")
        if self.horizon < 4:
            raise ValueError("response-kernel horizon must be at least four steps")
        if self.n_pairs < 2:
            raise ValueError("response-kernel evaluation needs at least two paired histories")
        if not self.operations:
            raise ValueError("response-kernel evaluation needs registered operations")
        if (
            isinstance(self.minimum_active_truth_rms_to_innovation_rms, bool)
            or not isinstance(
                self.minimum_active_truth_rms_to_innovation_rms, Real
            )
            or not math.isfinite(self.minimum_active_truth_rms_to_innovation_rms)
            or self.minimum_active_truth_rms_to_innovation_rms <= 0
        ):
            raise ValueError(
                "minimum_active_truth_rms_to_innovation_rms must be positive and finite"
            )
        if (
            isinstance(self.maximum_truth_mc_se_to_truth_rms_ratio, bool)
            or not isinstance(self.maximum_truth_mc_se_to_truth_rms_ratio, Real)
            or not math.isfinite(self.maximum_truth_mc_se_to_truth_rms_ratio)
            or self.maximum_truth_mc_se_to_truth_rms_ratio <= 0
        ):
            raise ValueError(
                "maximum_truth_mc_se_to_truth_rms_ratio must be positive and finite"
            )
        if (
            isinstance(self.minimum_transient_decay_estimable_fraction, bool)
            or not isinstance(
                self.minimum_transient_decay_estimable_fraction, Real
            )
            or not math.isfinite(
                self.minimum_transient_decay_estimable_fraction
            )
            or not 0
            <= self.minimum_transient_decay_estimable_fraction
            <= 1
        ):
            raise ValueError(
                "minimum_transient_decay_estimable_fraction must lie in [0,1]"
            )
        identifiers = [operation.id for operation in self.operations]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("response-kernel operation IDs must be unique")
        for operation in self.operations:
            operation.validate()


@dataclass(frozen=True)
class SuiteConfig:
    schema_version: str
    name: str
    scenarios: tuple[ScenarioSpec, ...]
    methods: tuple[MethodSpec, ...]
    history_lags: tuple[int, ...]
    horizons: tuple[int, ...]
    views: tuple[str, ...]
    seeds: tuple[int, ...]
    validation_fraction: float
    test_fraction: float
    output_dir: str
    resource: ResourceBudget
    save_channels: bool = True
    n_predictive_samples: int = 16
    response_kernel_evaluation: ResponseKernelEvaluationSpec | None = None

    def validate(self) -> None:
        if self.schema_version != "1":
            raise ValueError(f"unsupported suite schema {self.schema_version!r}")
        if not self.scenarios or not self.methods:
            raise ValueError("suite requires scenarios and methods")
        if not self.history_lags or min(self.history_lags) < 1:
            raise ValueError("history_lags must be positive")
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if set(self.views) - {"complete_state", "latent", "calcium"}:
            raise ValueError("unknown data view")
        if not 0 < self.validation_fraction < .5 or not 0 < self.test_fraction < .5:
            raise ValueError("invalid split fractions")
        if self.validation_fraction + self.test_fraction >= .8:
            raise ValueError("split leaves too little training data")
        if self.n_predictive_samples < 0 or self.n_predictive_samples == 1:
            raise ValueError("n_predictive_samples must be zero or at least two")
        self.resource.validate()
        if self.response_kernel_evaluation is not None:
            self.response_kernel_evaluation.validate()
        for scenario in self.scenarios:
            if scenario.family not in {"mechanistic", "legacy_generalized"}:
                raise ValueError(f"unknown scenario family {scenario.family!r}")
            if scenario.n_trajectories < 3:
                raise ValueError("scenario needs at least three independent trajectories")


def _response_integer(value: Any, field_name: str) -> int:
    """Parse response-operation indices without silently truncating YAML floats."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def load_suite(path: str | Path) -> SuiteConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict):
        raise ValueError("suite YAML must contain a mapping")
    scenarios = tuple(
        ScenarioSpec(
            id=item["id"],
            family=item.get("family", "mechanistic"),
            params=item.get("params", {}),
            n_trajectories=int(item.get("n_trajectories", 6)),
            views=None if item.get("views") is None else tuple(item["views"]),
        )
        for item in raw.get("scenarios", [])
    )
    methods = tuple(
        MethodSpec(
            name=item["name"],
            params=item.get("params", {}),
            tune=item.get("tune", {}),
            views=None if item.get("views") is None else tuple(item["views"]),
            horizons=None if item.get("horizons") is None else tuple(item["horizons"]),
            scenarios=None if item.get("scenarios") is None else tuple(item["scenarios"]),
            seeds=None if item.get("seeds") is None else tuple(int(seed) for seed in item["seeds"]),
        )
        for item in raw.get("methods", [])
    )
    resource = ResourceBudget(**raw.get("resource", {}))
    response_raw = raw.get("response_kernel_evaluation")
    response_kernel_evaluation = None
    if response_raw is not None:
        response_kernel_evaluation = ResponseKernelEvaluationSpec(
            horizon=_response_integer(response_raw["horizon"], "response horizon"),
            n_pairs=_response_integer(response_raw["n_pairs"], "response n_pairs"),
            operations=tuple(
                ResponseOperationSpec(
                    id=str(item["id"]),
                    kind=str(item["kind"]),
                    modulator_index=(
                        None
                        if item.get("modulator_index") is None
                        else _response_integer(
                            item["modulator_index"], "modulator_index"
                        )
                    ),
                    target_neurons=tuple(
                        _response_integer(value, "target_neurons")
                        for value in item.get("target_neurons", ())
                    ),
                    source_neurons=tuple(
                        _response_integer(value, "source_neurons")
                        for value in item.get("source_neurons", ())
                    ),
                    amplitude=float(item.get("amplitude", 0.0)),
                    duration_steps=(
                        None
                        if item.get("duration_steps") is None
                        else _response_integer(
                            item["duration_steps"], "duration_steps"
                        )
                    ),
                    selection=item.get("selection"),
                    expected_effect=item.get("expected_effect"),
                )
                for item in response_raw.get("operations", ())
            ),
            minimum_active_truth_rms_to_innovation_rms=float(
                response_raw.get(
                    "minimum_active_truth_rms_to_innovation_rms", 1e-4
                )
            ),
            maximum_truth_mc_se_to_truth_rms_ratio=float(
                response_raw.get(
                    "maximum_truth_mc_se_to_truth_rms_ratio", 10.0
                )
            ),
            minimum_transient_decay_estimable_fraction=float(
                response_raw.get(
                    "minimum_transient_decay_estimable_fraction", 0.0
                )
            ),
        )
    suite = SuiteConfig(
        schema_version=str(raw.get("schema_version", "")),
        name=raw["name"],
        scenarios=scenarios,
        methods=methods,
        history_lags=tuple(raw.get("history_lags", [1])),
        horizons=tuple(raw.get("horizons", [1])),
        views=tuple(raw.get("views", ["complete_state", "calcium"])),
        seeds=tuple(raw.get("seeds", [0])),
        validation_fraction=float(raw.get("validation_fraction", .2)),
        test_fraction=float(raw.get("test_fraction", .2)),
        output_dir=str(raw.get("output_dir", f"outputs/{raw['name']}")),
        resource=resource,
        save_channels=bool(raw.get("save_channels", True)),
        n_predictive_samples=int(raw.get("n_predictive_samples", 16)),
        response_kernel_evaluation=response_kernel_evaluation,
    )
    suite.validate()
    return suite
