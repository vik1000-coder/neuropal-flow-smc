"""Capability declarations prevent invalid cross-method comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PredictiveDistribution = Literal["none", "normalized", "unnormalized_score"]


@dataclass(frozen=True)
class Capabilities:
    predictive_mean: bool = False
    predictive_distribution: PredictiveDistribution = "none"
    effect_channels: tuple[str, ...] = ()
    graph_scores: bool = False
    equations: bool = False
    latent_state: bool = False
    interventions: bool = False
    recursive_path_samples: bool = False
    online_change: bool = False
    offline_change: bool = False

    def supports(self, capability: str) -> bool:
        if capability == "predictive_distribution":
            return self.predictive_distribution != "none"
        if not hasattr(self, capability):
            raise ValueError(f"unknown capability {capability!r}")
        return bool(getattr(self, capability))


@dataclass(frozen=True)
class ResourceProfile:
    scheduling_class: str = "tiny_cpu"
    threads: int = 1
    uses_accelerator: bool = False
    estimated_peak_gb: float = 0.5


class UnsupportedCapability(RuntimeError):
    def __init__(self, method: str, capability: str):
        super().__init__(f"{method!r} does not support capability {capability!r}")
        self.method = method
        self.capability = capability


CANONICAL_CHANNELS = (
    "conditional_mean_derivative",
    "conditional_log_variance_derivative",
    "conditional_tail_high_derivative",
    "conditional_shape_tail_derivative",
    "conditional_covariance_derivative",
    "conditional_correlation_derivative",
    "joint_score_cross_moment",
    "joint_squared_score_covariance",
    "linear_transition_coefficient",
    "conditional_independence_strength",
    "marginal_lagged_association",
)


def validate_channel_contract(capabilities: Capabilities, channels: dict) -> None:
    """Require semantic channels to agree with the estimator declaration."""

    declared = set(capabilities.effect_channels)
    available = set(channels)
    missing = [
        name
        for name in declared
        if name not in available and f"{name}_feature" not in available
    ]
    if missing:
        raise ValueError(f"declared effect channels are missing: {sorted(missing)}")
    emitted = set()
    for name in available:
        base = name[:-8] if name.endswith("_feature") else name
        if base in CANONICAL_CHANNELS:
            emitted.add(base)
    undeclared = emitted - declared
    if undeclared:
        raise ValueError(f"semantic effect channels were not declared: {sorted(undeclared)}")
