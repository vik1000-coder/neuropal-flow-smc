"""Typed benchmark contracts.

The contracts deliberately separate three objects that are often conflated:

* the structural data-generating mechanism;
* the observational conditional law available to a predictor; and
* a graph/readout produced by an estimator.

That separation lets the benchmark score predictive, mechanistic, and causal
claims against the corresponding oracle instead of treating a connectome as a
universal ground truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


VALID_MECHANISMS = {
    "null",
    "additive_mean",
    "synaptic_gain",
    "intrinsic_gain",
    "innovation_variance",
    "tail_burst",
    "mixed",
}


@dataclass(frozen=True)
class ResourceBudget:
    """Hard limits used by runners rather than informal recommendations."""

    max_threads: int = 2
    max_parallel_jobs: int = 1
    max_rss_gb: float = 6.0
    max_output_gb: float = 1.5
    nice: int = 19
    checkpoint_minutes: int = 20
    stop_free_disk_gb: float = 4.0

    def validate(self) -> None:
        if self.max_threads < 1 or self.max_parallel_jobs < 1:
            raise ValueError("thread and job limits must be positive")
        if self.max_rss_gb <= 0 or self.max_output_gb <= 0:
            raise ValueError("memory and output limits must be positive")
        if self.stop_free_disk_gb < 1:
            raise ValueError("stop_free_disk_gb must leave a meaningful safety margin")


@dataclass(frozen=True)
class DGPConfig:
    """Configuration for a generalized stochastic neuromodulatory state-space model."""

    n_neurons: int = 8
    n_modulators: int = 2
    n_trajectories: int = 6
    n_steps: int = 1_200
    burn_in: int = 200
    mechanism: str = "mixed"
    network_density: float = 0.22
    state_decay: float = 0.62
    coupling_radius: float = 0.28
    modulator_rho: float = 0.94
    release_strength: float = 0.32
    modulator_noise: float = 0.025
    base_noise: float = 0.16
    effect_strength: float = 0.32
    receptor_density: float = 0.35
    burst_base_probability: float = 0.025
    burst_scale: float = 3.5
    calcium_decay: float = 0.82
    measurement_noise: float = 0.035
    missing_rate: float = 0.0
    stimulus_rate: float = 0.025
    hidden_driver_strength: float = 0.0
    change_fraction: float | None = None
    change_multiplier: float = 1.75
    include_counterfactual: bool = True
    store_pointwise_jacobians: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.mechanism not in VALID_MECHANISMS:
            raise ValueError(f"unknown mechanism {self.mechanism!r}")
        if self.n_neurons < 2 or self.n_modulators < 1:
            raise ValueError("need at least two neurons and one modulator")
        if self.n_trajectories < 3:
            raise ValueError("at least three trajectories are required for grouped splits")
        if self.n_steps < 20 or self.burn_in < 0:
            raise ValueError("trajectory is too short")
        for name in ("network_density", "receptor_density", "missing_rate", "stimulus_rate"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not 0 <= self.state_decay < 1 or not 0 <= self.modulator_rho < 1:
            raise ValueError("autoregressive decay parameters must lie in [0, 1)")
        if self.base_noise <= 0 or self.burst_scale < 1:
            raise ValueError("noise scale must be positive and burst_scale >= 1")
        if self.change_fraction is not None and not 0.1 <= self.change_fraction <= 0.9:
            raise ValueError("change_fraction must leave usable pre/post segments")


@dataclass(frozen=True)
class MethodConfig:
    name: str
    enabled: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)
    tune: Mapping[str, Sequence[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str = "smoke"
    dgps: tuple[DGPConfig, ...] = field(default_factory=lambda: (DGPConfig(),))
    methods: tuple[MethodConfig, ...] = field(default_factory=tuple)
    horizons: tuple[int, ...] = (1, 3, 8)
    history_lags: tuple[int, ...] = (1, 2, 4, 8)
    views: tuple[str, ...] = ("complete_state", "latent", "calcium")
    seeds: tuple[int, ...] = (0, 1, 2)
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    resource: ResourceBudget = field(default_factory=ResourceBudget)
    output_dir: str = "outputs/smoke"
    save_raw_data: bool = False

    def validate(self) -> None:
        if not self.dgps:
            raise ValueError("at least one DGP is required")
        for dgp in self.dgps:
            dgp.validate()
        self.resource.validate()
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if not self.history_lags or min(self.history_lags) < 1:
            raise ValueError("history lags must be positive")
        if not 0 < self.validation_fraction < 0.5 or not 0 < self.test_fraction < 0.5:
            raise ValueError("validation/test fractions must lie in (0, .5)")
        if self.validation_fraction + self.test_fraction >= 0.8:
            raise ValueError("not enough trajectories remain for fitting")
        bad_views = set(self.views) - {"complete_state", "latent", "calcium"}
        if bad_views:
            raise ValueError(f"unsupported views: {sorted(bad_views)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralParameters:
    """Ground-truth parameterization, always using [target, source] orientation."""

    baseline_connectivity: np.ndarray
    release_map: np.ndarray
    additive_receptors: np.ndarray
    intrinsic_receptors: np.ndarray
    variance_receptors: np.ndarray
    tail_receptors: np.ndarray
    synaptic_gain: np.ndarray
    direct_logvariance: np.ndarray
    input_loading: np.ndarray
    modulator_input_loading: np.ndarray
    hidden_loading: np.ndarray
    base_logvariance: np.ndarray

    def validate(self, n: int, k: int) -> None:
        expected = {
            "baseline_connectivity": (n, n),
            "release_map": (k, n),
            "additive_receptors": (n, k),
            "intrinsic_receptors": (n, k),
            "variance_receptors": (n, k),
            "tail_receptors": (n, k),
            "synaptic_gain": (k, n, n),
            "direct_logvariance": (n, n),
            "input_loading": (n,),
            "modulator_input_loading": (k,),
            "hidden_loading": (n,),
            "base_logvariance": (n,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")


@dataclass
class Trajectory:
    latent: np.ndarray
    calcium: np.ndarray
    modulator: np.ndarray
    stimulus: np.ndarray
    conditional_mean: np.ndarray
    conditional_variance: np.ndarray
    burst_probability: np.ndarray
    hidden_driver: np.ndarray
    observed_mask: np.ndarray
    average_mean_jacobian: np.ndarray
    average_logvariance_jacobian: np.ndarray
    average_tail_jacobian: np.ndarray
    pointwise_mean_jacobian: np.ndarray | None = None
    pointwise_logvariance_jacobian: np.ndarray | None = None
    pointwise_tail_jacobian: np.ndarray | None = None
    upper_tail_probability: np.ndarray | None = None
    conditional_covariance: np.ndarray | None = None
    conditional_correlation: np.ndarray | None = None
    average_covariance_jacobian: np.ndarray | None = None
    average_correlation_jacobian: np.ndarray | None = None
    pointwise_covariance_jacobian: np.ndarray | None = None
    pointwise_correlation_jacobian: np.ndarray | None = None
    covariance_concentration_susceptibility: np.ndarray | None = None
    correlation_concentration_susceptibility: np.ndarray | None = None
    shape_tail_probability: np.ndarray | None = None
    average_shape_tail_jacobian: np.ndarray | None = None
    pointwise_shape_tail_jacobian: np.ndarray | None = None


@dataclass
class Dataset:
    config: Any
    parameters: Any
    trajectories: list[Trajectory]
    no_modulation: list[Trajectory] | None
    mechanism_support: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    interventions: dict[str, list[Trajectory]] = field(default_factory=dict)
    intervention_metadata: dict[str, Any] = field(default_factory=dict)

    def arrays(self, view: str) -> list[np.ndarray]:
        if view not in {"latent", "calcium"}:
            raise ValueError(f"unknown view {view!r}")
        return [getattr(tr, view) for tr in self.trajectories]


@dataclass
class SupervisedData:
    features: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    trajectory_ids: np.ndarray
    times: np.ndarray
    feature_names: tuple[str, ...]
    source_index: np.ndarray
    view: str
    horizon: int
    oracle_mean: np.ndarray | None = None
    oracle_variance: np.ndarray | None = None
    oracle_burst_probability: np.ndarray | None = None
    oracle_tail_probability: np.ndarray | None = None
    oracle_covariance: np.ndarray | None = None
    oracle_correlation: np.ndarray | None = None
    oracle_shape_tail_probability: np.ndarray | None = None


@dataclass
class Prediction:
    mean: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray | None = None
    correlation: np.ndarray | None = None
    cdf: np.ndarray | None = None
    samples: np.ndarray | None = None
    log_prob: np.ndarray | None = None
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def ensure_within(root: Path, path: Path) -> Path:
    """Resolve an output path and reject accidental writes outside the benchmark."""

    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"output path {resolved} escapes isolated root {resolved_root}")
    return resolved
