"""Hierarchical worm population and paired observation robustness layer.

This module is an isolated layer over :mod:`neuromod_benchmark.mechanistic`.  It draws
reproducible worm-specific parameter deviations with declared coefficients of variation,
while preserving structural support, coefficient signs, positivity, and the recurrent
row-norm constraint certified by the reference simulator.

Observation-only variants reuse one exogenous-noise object.  Calcium decay and
fluorescence noise never feed back into latent neural dynamics, so every paired variant
must have a bitwise-identical latent trajectory and latent hash.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Final, Mapping

import numpy as np

from .mechanistic import (
    MechanisticConfig,
    MechanisticParameters,
    MechanisticTrajectory,
    NamedRNGStreams,
    draw_exogenous_noise,
    generate_mechanistic_parameters,
    simulate_mechanistic,
    stability_certificate,
)


SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test")


@dataclass(frozen=True)
class ObservationVariant:
    """A change restricted to the observation process."""

    name: str
    measurement_noise_multiplier: float = 1.0
    calcium_decay_multiplier: float = 1.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("observation variant name must be non-empty")
        if self.measurement_noise_multiplier < 0:
            raise ValueError("measurement-noise multiplier must be non-negative")
        if self.calcium_decay_multiplier <= 0:
            raise ValueError("calcium-decay multiplier must be positive")


DEFAULT_OBSERVATION_VARIANTS: Final[tuple[ObservationVariant, ...]] = (
    ObservationVariant("reference"),
    ObservationVariant("high_measurement_noise", measurement_noise_multiplier=2.5),
    ObservationVariant("slow_calcium", calcium_decay_multiplier=2.0),
)


@dataclass(frozen=True)
class HierarchyConfig:
    """Population split sizes and requested coefficient-of-variation hierarchy."""

    n_train_worms: int = 6
    n_validation_worms: int = 3
    n_test_worms: int = 3
    recurrent_weight_cv: float = 0.15
    clearance_time_cv: float = 0.15
    receptor_expression_cv: float = 0.20
    receptor_kd_cv: float = 0.20
    calcium_decay_cv: float = 0.15
    seed: int = 91_337
    observation_variants: tuple[ObservationVariant, ...] = DEFAULT_OBSERVATION_VARIANTS

    def validate(self) -> None:
        if self.n_train_worms < 1:
            raise ValueError("at least one training worm is required")
        if self.n_validation_worms < 1 or self.n_test_worms < 1:
            raise ValueError("validation and test splits must be non-empty")
        for name in (
            "recurrent_weight_cv",
            "clearance_time_cv",
            "receptor_expression_cv",
            "receptor_kd_cv",
            "calcium_decay_cv",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 2.0:
                raise ValueError(f"{name} must lie in [0,2]")
        if self.seed < 0:
            raise ValueError("hierarchy seed must be non-negative")
        if not self.observation_variants:
            raise ValueError("at least one observation variant is required")
        names = []
        for variant in self.observation_variants:
            variant.validate()
            names.append(variant.name)
        if len(names) != len(set(names)):
            raise ValueError("observation variant names must be unique")

    @property
    def split_counts(self) -> Mapping[str, int]:
        return {
            "train": self.n_train_worms,
            "validation": self.n_validation_worms,
            "test": self.n_test_worms,
        }

    @property
    def total_worms(self) -> int:
        return sum(self.split_counts.values())

    @property
    def declared_coefficient_cvs(self) -> Mapping[str, float]:
        return {
            "recurrent_weight": self.recurrent_weight_cv,
            "clearance_time": self.clearance_time_cv,
            "receptor_expression": self.receptor_expression_cv,
            "receptor_kd": self.receptor_kd_cv,
            "calcium_decay": self.calcium_decay_cv,
        }


def _lognormal_multiplier(
    rng: np.random.Generator, shape: tuple[int, ...], coefficient_of_variation: float
) -> np.ndarray:
    """Mean-one lognormal multipliers parameterized by their requested CV."""

    cv = float(coefficient_of_variation)
    if cv == 0:
        return np.ones(shape, dtype=float)
    sigma = math.sqrt(math.log1p(cv**2))
    return np.exp(sigma * rng.standard_normal(shape) - 0.5 * sigma**2)


def _derived_seed(root_seed: int, name: str) -> int:
    words = NamedRNGStreams(root_seed).seed_sequence(name).generate_state(2, dtype=np.uint32)
    return int(words[0]) | (int(words[1]) << 32)


@dataclass(frozen=True)
class WormMultipliers:
    """Effective realized population-to-worm coefficient ratios."""

    recurrent_weight: np.ndarray
    clearance_time: np.ndarray
    receptor_expression: np.ndarray
    receptor_kd: np.ndarray
    calcium_decay: float


@dataclass(frozen=True)
class WormSpec:
    """One worm's parameters, configuration, split, and reproducible seeds."""

    worm_id: str
    split: str
    split_index: int
    parameter_seed: int
    simulation_seed: int
    config: MechanisticConfig
    parameters: MechanisticParameters
    multipliers: WormMultipliers

    def validate(
        self,
        population_config: MechanisticConfig,
        population_parameters: MechanisticParameters,
    ) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}")
        self.parameters.validate(self.config)
        if self.config.n_neurons != population_config.n_neurons:
            raise ValueError("worm neuron count differs from population")
        if self.config.n_modulators != population_config.n_modulators:
            raise ValueError("worm modulator count differs from population")

        base_a = population_parameters.baseline_connectivity
        worm_a = self.parameters.baseline_connectivity
        if not np.array_equal(worm_a != 0, base_a != 0):
            raise ValueError("worm recurrent support differs from population")
        if not np.array_equal(np.sign(worm_a), np.sign(base_a)):
            raise ValueError("worm recurrent signs differ from population")
        base_expression = population_parameters.receptor_expression
        worm_expression = self.parameters.receptor_expression
        if not np.array_equal(worm_expression > 0, base_expression > 0):
            raise ValueError("worm receptor support differs from population")
        if np.any(worm_expression < 0) or np.any(worm_expression > 1):
            raise ValueError("worm receptor expression violates [0,1]")
        if np.any(self.parameters.receptor_kd <= 0):
            raise ValueError("worm receptor Kd must remain positive")
        if np.any(
            (self.parameters.clearance_rho <= 0)
            | (self.parameters.clearance_rho >= 1)
        ):
            raise ValueError("worm clearance rho must remain in (0,1)")
        certificate = stability_certificate(self.config, self.parameters)
        if certificate.baseline_max_row_l1 > self.config.recurrent_row_l1 + 1e-12:
            raise ValueError("worm recurrent dynamics exceed the stability bound")


def _effective_ratio(
    changed: np.ndarray, baseline: np.ndarray, active: np.ndarray | None = None
) -> np.ndarray:
    changed = np.asarray(changed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if active is None:
        active = baseline != 0
    ratio = np.ones_like(changed, dtype=float)
    ratio[active] = changed[active] / baseline[active]
    return ratio


def draw_worm_spec(
    population_config: MechanisticConfig,
    population_parameters: MechanisticParameters,
    hierarchy: HierarchyConfig,
    *,
    split: str,
    split_index: int,
) -> WormSpec:
    """Draw one worm independently from its stable ID-specific named stream."""

    population_config.validate()
    population_parameters.validate(population_config)
    hierarchy.validate()
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split_index < 0:
        raise ValueError("split_index must be non-negative")

    worm_id = f"{split}_worm_{split_index:04d}"
    stream_name = f"hierarchy/{worm_id}/parameters"
    parameter_seed = _derived_seed(hierarchy.seed, stream_name)
    simulation_seed = _derived_seed(hierarchy.seed, f"hierarchy/{worm_id}/simulation")
    # The recorded child seed alone is sufficient to recreate this worm draw.
    rng = np.random.default_rng(parameter_seed)

    # Recurrent coefficients receive positive multiplicative deviations, preserving
    # exact zeros and signs.  Rows that would exceed the reference stability budget are
    # contracted as a whole; the final effective multiplier is recorded.
    base_a = population_parameters.baseline_connectivity
    recurrent_requested = _lognormal_multiplier(
        rng, base_a.shape, hierarchy.recurrent_weight_cv
    )
    worm_a = base_a * recurrent_requested
    for target in range(population_config.n_neurons):
        row_l1 = float(np.sum(np.abs(worm_a[target])))
        if row_l1 > population_config.recurrent_row_l1:
            worm_a[target] *= population_config.recurrent_row_l1 / row_l1
    recurrent_effective = _effective_ratio(worm_a, base_a)

    # Draw clearance in physical seconds, then exactly discretize it.
    base_tau = -population_config.dt_seconds / np.log(
        population_parameters.clearance_rho
    )
    clearance_multiplier = _lognormal_multiplier(
        rng, base_tau.shape, hierarchy.clearance_time_cv
    )
    worm_tau = base_tau * clearance_multiplier
    worm_clearance_rho = np.exp(-population_config.dt_seconds / worm_tau)

    base_expression = population_parameters.receptor_expression
    expression_requested = _lognormal_multiplier(
        rng, base_expression.shape, hierarchy.receptor_expression_cv
    )
    expression_active = base_expression > 0
    worm_expression = np.zeros_like(base_expression)
    worm_expression[expression_active] = np.minimum(
        1.0,
        base_expression[expression_active] * expression_requested[expression_active],
    )
    expression_effective = _effective_ratio(
        worm_expression, base_expression, expression_active
    )

    kd_requested = _lognormal_multiplier(
        rng, population_parameters.receptor_kd.shape, hierarchy.receptor_kd_cv
    )
    worm_kd = population_parameters.receptor_kd * kd_requested

    calcium_multiplier = float(
        _lognormal_multiplier(rng, (1,), hierarchy.calcium_decay_cv)[0]
    )
    worm_calcium_decay = population_config.calcium_decay_seconds * calcium_multiplier
    worm_config = replace(
        population_config,
        calcium_decay_seconds=worm_calcium_decay,
        seed=simulation_seed,
    )
    worm_parameters = replace(
        population_parameters,
        baseline_connectivity=worm_a,
        clearance_rho=worm_clearance_rho,
        receptor_expression=worm_expression,
        receptor_kd=worm_kd,
    )
    multipliers = WormMultipliers(
        recurrent_weight=recurrent_effective,
        clearance_time=clearance_multiplier,
        receptor_expression=expression_effective,
        receptor_kd=kd_requested,
        calcium_decay=calcium_multiplier,
    )
    spec = WormSpec(
        worm_id=worm_id,
        split=split,
        split_index=split_index,
        parameter_seed=parameter_seed,
        simulation_seed=simulation_seed,
        config=worm_config,
        parameters=worm_parameters,
        multipliers=multipliers,
    )
    spec.validate(population_config, population_parameters)
    return spec


def generate_worm_specs(
    population_config: MechanisticConfig,
    hierarchy: HierarchyConfig,
    population_parameters: MechanisticParameters | None = None,
) -> tuple[MechanisticParameters, tuple[WormSpec, ...]]:
    """Generate split-labeled worm parameter sets without simulating trajectories."""

    population_config.validate()
    hierarchy.validate()
    if population_parameters is None:
        population_parameters = generate_mechanistic_parameters(population_config)
    population_parameters.validate(population_config)
    specs = []
    for split in SPLITS:
        for index in range(hierarchy.split_counts[split]):
            specs.append(
                draw_worm_spec(
                    population_config,
                    population_parameters,
                    hierarchy,
                    split=split,
                    split_index=index,
                )
            )
    return population_parameters, tuple(specs)


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class ObservationRecord:
    """One observation-only rendering of a fixed worm latent realization."""

    variant: ObservationVariant
    config: MechanisticConfig
    trajectory: MechanisticTrajectory
    latent_hash: str


def simulate_observation_variants(
    worm: WormSpec,
    variants: tuple[ObservationVariant, ...] = DEFAULT_OBSERVATION_VARIANTS,
) -> Mapping[str, ObservationRecord]:
    """Render paired observation variants with an exact shared latent realization."""

    if not variants:
        raise ValueError("at least one observation variant is required")
    names = []
    for variant in variants:
        variant.validate()
        names.append(variant.name)
    if len(names) != len(set(names)):
        raise ValueError("observation variant names must be unique")

    exogenous = draw_exogenous_noise(worm.config, worm.simulation_seed)
    records: dict[str, ObservationRecord] = {}
    reference_latent: np.ndarray | None = None
    reference_hash: str | None = None
    for variant in variants:
        variant_config = replace(
            worm.config,
            measurement_noise=(
                worm.config.measurement_noise
                * variant.measurement_noise_multiplier
            ),
            calcium_decay_seconds=(
                worm.config.calcium_decay_seconds
                * variant.calcium_decay_multiplier
            ),
        )
        trajectory = simulate_mechanistic(
            variant_config,
            worm.parameters,
            exogenous=exogenous,
        )
        latent_hash = _array_hash(trajectory.latent_neural)
        if reference_latent is None:
            reference_latent = trajectory.latent_neural
            reference_hash = latent_hash
        else:
            if not np.array_equal(trajectory.latent_neural, reference_latent):
                raise RuntimeError(
                    "observation-only variant changed latent neural dynamics"
                )
            if latent_hash != reference_hash:
                raise RuntimeError("identical latent arrays produced different hashes")
        records[variant.name] = ObservationRecord(
            variant=variant,
            config=variant_config,
            trajectory=trajectory,
            latent_hash=latent_hash,
        )
    return records


@dataclass(frozen=True)
class WormDataset:
    spec: WormSpec
    observations: Mapping[str, ObservationRecord]

    @property
    def latent_hash(self) -> str:
        hashes = {record.latent_hash for record in self.observations.values()}
        if len(hashes) != 1:
            raise RuntimeError("worm observation variants do not share one latent hash")
        return next(iter(hashes))


def _sample_cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    mean = float(np.mean(values))
    if mean == 0:
        return 0.0
    return float(np.std(values, ddof=0) / abs(mean))


def realized_coefficient_cvs(
    specs: tuple[WormSpec, ...],
    population_parameters: MechanisticParameters,
) -> Mapping[str, float]:
    """Summarize realized effective multipliers across worms and active coefficients."""

    if not specs:
        raise ValueError("at least one worm spec is required")
    recurrent_active = population_parameters.baseline_connectivity != 0
    expression_active = population_parameters.receptor_expression > 0
    recurrent = np.concatenate(
        [spec.multipliers.recurrent_weight[recurrent_active] for spec in specs]
    )
    clearance = np.concatenate([spec.multipliers.clearance_time for spec in specs])
    expression = np.concatenate(
        [spec.multipliers.receptor_expression[expression_active] for spec in specs]
    )
    kd = np.concatenate([spec.multipliers.receptor_kd.ravel() for spec in specs])
    calcium = np.asarray([spec.multipliers.calcium_decay for spec in specs])
    return {
        "recurrent_weight": _sample_cv(recurrent),
        "clearance_time": _sample_cv(clearance),
        "receptor_expression": _sample_cv(expression),
        "receptor_kd": _sample_cv(kd),
        "calcium_decay": _sample_cv(calcium),
    }


@dataclass(frozen=True)
class HierarchicalDataset:
    population_config: MechanisticConfig
    hierarchy_config: HierarchyConfig
    population_parameters: MechanisticParameters
    worms: tuple[WormDataset, ...]
    split_worm_ids: Mapping[str, tuple[str, ...]]
    metadata: Mapping[str, object]

    def worms_for_split(self, split: str) -> tuple[WormDataset, ...]:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        return tuple(worm for worm in self.worms if worm.spec.split == split)


def generate_hierarchical_dataset(
    population_config: MechanisticConfig,
    hierarchy: HierarchyConfig,
    population_parameters: MechanisticParameters | None = None,
) -> HierarchicalDataset:
    """Generate train/validation/test worms and paired observation renderings."""

    population_parameters, specs = generate_worm_specs(
        population_config,
        hierarchy,
        population_parameters,
    )
    worms = tuple(
        WormDataset(
            spec=spec,
            observations=simulate_observation_variants(
                spec, hierarchy.observation_variants
            ),
        )
        for spec in specs
    )
    split_worm_ids = {
        split: tuple(worm.spec.worm_id for worm in worms if worm.spec.split == split)
        for split in SPLITS
    }
    all_ids = [worm.spec.worm_id for worm in worms]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("worm IDs overlap across declared splits")
    latent_hash_invariant = all(
        len({record.latent_hash for record in worm.observations.values()}) == 1
        for worm in worms
    )
    return HierarchicalDataset(
        population_config=population_config,
        hierarchy_config=hierarchy,
        population_parameters=population_parameters,
        worms=worms,
        split_worm_ids=split_worm_ids,
        metadata={
            "declared_coefficient_cvs": dict(hierarchy.declared_coefficient_cvs),
            "realized_coefficient_cvs": dict(
                realized_coefficient_cvs(specs, population_parameters)
            ),
            "split_counts": dict(hierarchy.split_counts),
            "split_unit": "worm",
            "orientation": "[target, source]",
            "observation_variants_share_exogenous_noise": True,
            "observation_variants_latent_hash_invariant": latent_hash_invariant,
        },
    )


__all__ = [
    "DEFAULT_OBSERVATION_VARIANTS",
    "HierarchicalDataset",
    "HierarchyConfig",
    "ObservationRecord",
    "ObservationVariant",
    "SPLITS",
    "WormDataset",
    "WormMultipliers",
    "WormSpec",
    "draw_worm_spec",
    "generate_hierarchical_dataset",
    "generate_worm_specs",
    "realized_coefficient_cvs",
    "simulate_observation_variants",
]
