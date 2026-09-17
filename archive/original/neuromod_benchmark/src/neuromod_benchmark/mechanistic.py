"""Reference mechanistic stochastic neuromodulator simulator.

This module is intentionally self-contained.  It does not depend on the benchmark's
existing ``schema`` or ``dgp`` modules, so it can be reviewed and integrated without
changing their public contracts.

The reference process has an explicit causal chain

    neural activity -> positive release -> concentration/clearance
    -> Hill receptor occupancy -> neural transition law -> calcium fluorescence.

Neuromodulation can enter exactly one of six one-step channels (or the legacy union):
additive mean, multiplicative synaptic efficacy, intrinsic excitability, innovation
variance, a mean/variance-matched non-Gaussian tail mixture, or correlation routing
through bounded one-factor loadings.  ``mixed`` intentionally retains the original five
channels and does not silently include correlation routing.  All one-step moments and
Jacobians returned here are analytic derivatives with respect to the current latent
neural state while holding the current concentration and calcium states fixed.

The neural drift is globally bounded (all state dependence enters through ``tanh`` and
bounded receptor occupancy), which avoids the companion-instability failure of legacy
VAR synthetic generators.  Counterfactual receptor knockouts reuse named exogenous RNG
streams exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Final, Iterable

import numpy as np
from scipy.special import ndtr


MECHANISMS: Final[frozenset[str]] = frozenset(
    {
        "null",
        "additive_mean",
        "synaptic_gain",
        "intrinsic_excitability",
        "innovation_variance",
        "matched_tail",
        "correlation_routing",
        "mixed",
    }
)
# Backward-compatibility contract: correlation routing is a separate pure scenario.
_LEGACY_MIXED_ACTIVE: Final[frozenset[str]] = frozenset(
    {
        "additive_mean",
        "synaptic_gain",
        "intrinsic_excitability",
        "innovation_variance",
        "matched_tail",
    }
)


def _active_mechanisms(mechanism: str) -> frozenset[str]:
    if mechanism == "null":
        return frozenset()
    if mechanism == "mixed":
        return _LEGACY_MIXED_ACTIVE
    return frozenset({mechanism})


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""

    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""

    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=float)))


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


def hill_occupancy(
    concentration: np.ndarray,
    receptor_kd: np.ndarray,
    hill_coefficient: np.ndarray,
    receptor_expression: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hill occupancy and derivative with respect to concentration.

    ``concentration`` has shape ``[modulator]`` and receptor arrays have shape
    ``[target, modulator]``.  Keeping this map explicit exposes the concentration /
    affinity scale gauge used by the benchmark's identifiability contract.
    """

    concentration = np.asarray(concentration, dtype=float)
    kd = np.asarray(receptor_kd, dtype=float)
    coefficient = np.asarray(hill_coefficient, dtype=float)
    expression = np.asarray(receptor_expression, dtype=float)
    if concentration.ndim != 1 or kd.shape != coefficient.shape or kd.shape != expression.shape:
        raise ValueError("Hill concentration/receptor shapes do not match")
    if kd.ndim != 2 or kd.shape[1] != len(concentration):
        raise ValueError("receptor arrays must have shape [target, modulator]")
    if np.any(concentration <= 0) or np.any(kd <= 0) or np.any(coefficient <= 0):
        raise ValueError("Hill concentration, affinity, and coefficient must be positive")
    c = concentration[None, :]
    kd_h = kd**coefficient
    c_h = c**coefficient
    denominator = kd_h + c_h
    occupancy = expression * c_h / denominator
    derivative = (
        expression
        * coefficient
        * kd_h
        * c ** (coefficient - 1.0)
        / denominator**2
    )
    return occupancy, derivative


@dataclass(frozen=True)
class MechanisticConfig:
    """Configuration independent of the rest of ``neuromod_benchmark``.

    Parameters are deliberately modest.  Effect tensors are drawn once for every
    mechanism, so configs that differ only in ``mechanism`` share the same baseline
    system under a common seed.
    """

    n_neurons: int = 6
    n_modulators: int = 2
    n_steps: int = 512
    burn_in: int = 512
    dt_seconds: float = 0.25
    mechanism: str = "mixed"
    seed: int = 0

    state_decay: float = 0.32
    recurrent_density: float = 0.25
    recurrent_row_l1: float = 0.28
    neural_bias_scale: float = 0.03
    base_noise: float = 0.18

    release_density: float = 0.35
    release_gain: float = 0.9
    release_bias: float = -0.35
    clearance_seconds_min: float = 4.0
    clearance_seconds_max: float = 40.0
    concentration_initial: float = 0.20

    receptor_density: float = 0.45
    effect_density: float = 0.55
    effect_strength: float = 0.45
    hill_coefficient_min: float = 1.0
    hill_coefficient_max: float = 2.0
    receptor_kd_min: float = 0.15
    receptor_kd_max: float = 0.90
    receptor_expression_min: float = 0.45
    receptor_expression_max: float = 1.0

    tail_base_probability: float = 0.08
    tail_low_scale: float = 0.50
    tail_high_scale: float = 3.0
    tail_threshold: float = 0.35
    shape_tail_z: float = 2.0

    # Correlation routing is globally positive definite through
    # eta_i = lambda_i f + sqrt(1-lambda_i^2) epsilon_i, |lambda_i| < max.
    correlation_max_loading: float = 0.65

    calcium_decay_seconds: float = 1.5
    measurement_noise: float = 0.03
    ligand_pulse_rate_hz: float = 0.0
    ligand_pulse_amplitude: float = 2.0
    ligand_pulse_decay_seconds: float = 2.0

    def validate(self) -> None:
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism {self.mechanism!r}")
        if self.n_neurons < 2 or self.n_modulators < 1:
            raise ValueError("need at least two neurons and one modulator")
        if self.n_steps < 2 or self.burn_in < 0:
            raise ValueError("n_steps must be >=2 and burn_in non-negative")
        if self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        if not 0.0 <= self.state_decay < 1.0:
            raise ValueError("state_decay must lie in [0, 1)")
        for name in (
            "recurrent_density",
            "release_density",
            "receptor_density",
            "effect_density",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if self.recurrent_row_l1 < 0 or self.release_gain <= 0 or self.base_noise <= 0:
            raise ValueError("dynamical scales must be positive")
        if (
            self.clearance_seconds_min <= 0
            or self.clearance_seconds_max < self.clearance_seconds_min
        ):
            raise ValueError("invalid clearance time range")
        if self.concentration_initial <= 0:
            raise ValueError("concentration_initial must be strictly positive")
        if not 0 < self.receptor_expression_min <= self.receptor_expression_max <= 1:
            raise ValueError("receptor expression must lie in (0,1]")
        if not 0 < self.receptor_kd_min <= self.receptor_kd_max:
            raise ValueError("receptor affinities must be positive")
        if not 0 < self.hill_coefficient_min <= self.hill_coefficient_max:
            raise ValueError("Hill coefficients must be positive")
        if not 0 < self.tail_base_probability < 0.5:
            raise ValueError("tail_base_probability must lie in (0,.5)")
        if not 0 < self.tail_low_scale < 1 < self.tail_high_scale:
            raise ValueError("tail scales must satisfy 0 < low < 1 < high")
        if self.shape_tail_z <= 0:
            raise ValueError("shape_tail_z must be positive")
        if not 0 < self.correlation_max_loading < 1:
            raise ValueError("correlation_max_loading must lie strictly in (0,1)")
        if self.calcium_decay_seconds <= 0 or self.measurement_noise < 0:
            raise ValueError("invalid calcium observation parameters")
        if self.ligand_pulse_rate_hz < 0 or self.ligand_pulse_amplitude < 0:
            raise ValueError("ligand pulse rate/amplitude must be non-negative")
        if self.ligand_pulse_decay_seconds <= 0:
            raise ValueError("ligand pulse decay must be positive")
        if self.ligand_pulse_rate_hz * self.dt_seconds > 0.5:
            raise ValueError("ligand pulse probability per frame is too large")

    @property
    def total_steps(self) -> int:
        return self.n_steps + self.burn_in

    @property
    def calcium_rho(self) -> float:
        return float(math.exp(-self.dt_seconds / self.calcium_decay_seconds))


class NamedRNGStreams:
    """Stateless named random streams derived reproducibly from one integer seed.

    Python's randomized ``hash`` is deliberately not used.  Asking for the same stream
    twice returns a fresh generator at the same initial state; different names are
    deterministically independent child streams.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)

    def seed_sequence(self, name: str) -> np.random.SeedSequence:
        if not name:
            raise ValueError("RNG stream name must be non-empty")
        digest = hashlib.blake2s(name.encode("utf-8"), digest_size=16).digest()
        words = np.frombuffer(digest, dtype=np.uint32).astype(np.uint64)
        entropy = [self.seed & 0xFFFFFFFF, (self.seed >> 32) & 0xFFFFFFFF]
        entropy.extend(int(word) for word in words)
        return np.random.SeedSequence(entropy)

    def generator(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self.seed_sequence(name))


def _signed_effect(
    rng: np.random.Generator,
    mask: np.ndarray,
    strength: float,
    *,
    positive: bool = False,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    magnitude = strength * rng.uniform(0.55, 1.0, size=mask.shape)
    sign = np.ones(mask.shape) if positive else rng.choice([-1.0, 1.0], size=mask.shape)
    return np.where(mask, sign * magnitude, 0.0)


def _ensure_one(mask: np.ndarray, candidates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ensure a boolean mask selects at least one candidate."""

    mask = np.asarray(mask, dtype=bool).copy()
    candidates = np.asarray(candidates, dtype=bool)
    if np.any(mask & candidates):
        return mask
    locations = np.argwhere(candidates)
    if not len(locations):
        raise ValueError("cannot activate an effect: candidate set is empty")
    loc = locations[int(rng.integers(0, len(locations)))]
    mask[tuple(loc)] = True
    return mask


@dataclass(frozen=True)
class MechanisticParameters:
    """Realized structural parameters; matrix orientation is ``[target, source]``."""

    baseline_connectivity: np.ndarray  # [N,N]
    neural_bias: np.ndarray  # [N]
    release_weights: np.ndarray  # [K,N], nonnegative
    release_bias: np.ndarray  # [K]
    clearance_rho: np.ndarray  # [K]
    receptor_expression: np.ndarray  # [N,K], in [0,1]
    receptor_kd: np.ndarray  # [N,K], positive
    hill_coefficient: np.ndarray  # [N,K], positive
    additive_effect: np.ndarray  # [N,K]
    synaptic_effect: np.ndarray  # [N,N,K]
    intrinsic_log_slope_effect: np.ndarray  # [N,K]
    intrinsic_threshold_effect: np.ndarray  # [N,K]
    logvariance_effect: np.ndarray  # [N,K]
    tail_logit_effect: np.ndarray  # [N,K]
    correlation_loading_effect: np.ndarray  # [N,K]
    base_variance: np.ndarray  # [N]

    def validate(self, config: MechanisticConfig) -> None:
        n, k = config.n_neurons, config.n_modulators
        expected = {
            "baseline_connectivity": (n, n),
            "neural_bias": (n,),
            "release_weights": (k, n),
            "release_bias": (k,),
            "clearance_rho": (k,),
            "receptor_expression": (n, k),
            "receptor_kd": (n, k),
            "hill_coefficient": (n, k),
            "additive_effect": (n, k),
            "synaptic_effect": (n, n, k),
            "intrinsic_log_slope_effect": (n, k),
            "intrinsic_threshold_effect": (n, k),
            "logvariance_effect": (n, k),
            "tail_logit_effect": (n, k),
            "correlation_loading_effect": (n, k),
            "base_variance": (n,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if np.any(self.release_weights < 0):
            raise ValueError("release weights must be nonnegative")
        if np.any((self.clearance_rho <= 0) | (self.clearance_rho >= 1)):
            raise ValueError("clearance_rho must lie strictly between zero and one")
        if np.any(self.receptor_expression < 0) or np.any(self.receptor_expression > 1):
            raise ValueError("receptor expression must lie in [0,1]")
        if np.any(self.receptor_kd <= 0) or np.any(self.hill_coefficient <= 0):
            raise ValueError("Hill parameters must be positive")
        if np.any(self.base_variance <= 0):
            raise ValueError("base variances must be positive")
        if not np.allclose(np.diag(self.baseline_connectivity), 0.0):
            raise ValueError("baseline connectivity diagonal must be zero")
        row_l1 = np.sum(np.abs(self.baseline_connectivity), axis=1)
        if np.max(row_l1, initial=0.0) > config.recurrent_row_l1 + 1e-12:
            raise ValueError("baseline recurrent row norm exceeds configured bound")


def generate_mechanistic_parameters(config: MechanisticConfig) -> MechanisticParameters:
    """Draw all potential mechanism tensors from the named ``parameters`` stream."""

    config.validate()
    rng = NamedRNGStreams(config.seed).generator("parameters")
    n, k = config.n_neurons, config.n_modulators

    # At least one incoming off-diagonal edge per target, then row-wise L1 scaling.
    support = rng.random((n, n)) < config.recurrent_density
    np.fill_diagonal(support, False)
    for target in range(n):
        if not np.any(support[target]):
            source_pool = np.delete(np.arange(n), target)
            support[target, int(rng.choice(source_pool))] = True
    a = np.where(support, rng.normal(size=(n, n)), 0.0)
    np.fill_diagonal(a, 0.0)
    for target in range(n):
        row_norm = float(np.sum(np.abs(a[target])))
        if row_norm > 0:
            desired = config.recurrent_row_l1 * float(rng.uniform(0.55, 1.0))
            a[target] *= desired / row_norm

    release_support = rng.random((k, n)) < config.release_density
    for mod in range(k):
        if not np.any(release_support[mod]):
            release_support[mod, int(rng.integers(0, n))] = True
    release_weights = np.zeros((k, n), dtype=float)
    for mod in range(k):
        values = rng.uniform(0.5, 1.0, size=int(release_support[mod].sum()))
        values *= config.release_gain / float(values.sum())
        release_weights[mod, release_support[mod]] = values

    receptor_support = rng.random((n, k)) < config.receptor_density
    for mod in range(k):
        if not np.any(receptor_support[:, mod]):
            receptor_support[int(rng.integers(0, n)), mod] = True
    expression = np.where(
        receptor_support,
        rng.uniform(
            config.receptor_expression_min,
            config.receptor_expression_max,
            size=(n, k),
        ),
        0.0,
    )

    effect_mask = (rng.random((n, k)) < config.effect_density) & receptor_support
    effect_mask = _ensure_one(effect_mask, receptor_support, rng)

    syn_candidates = support[:, :, None] & receptor_support[:, None, :]
    syn_mask = (rng.random((n, n, k)) < config.effect_density) & syn_candidates
    syn_mask = _ensure_one(syn_mask, syn_candidates, rng)

    clearance_tau = np.exp(
        rng.uniform(
            math.log(config.clearance_seconds_min),
            math.log(config.clearance_seconds_max),
            size=k,
        )
    )
    clearance_rho = np.exp(-config.dt_seconds / clearance_tau)

    # Keep this legacy draw order frozen: medium/full results generated before the
    # correlation extension must remain exactly reproducible for all old mechanisms.
    neural_bias = rng.normal(0.0, config.neural_bias_scale, size=n)
    receptor_kd = rng.uniform(
        config.receptor_kd_min, config.receptor_kd_max, size=(n, k)
    )
    hill_coefficient = rng.uniform(
        config.hill_coefficient_min,
        config.hill_coefficient_max,
        size=(n, k),
    )
    additive_effect = _signed_effect(rng, effect_mask, config.effect_strength)
    synaptic_effect = _signed_effect(rng, syn_mask, 0.7 * config.effect_strength)
    intrinsic_log_slope_effect = _signed_effect(
        rng, effect_mask, 0.65 * config.effect_strength
    )
    intrinsic_threshold_effect = _signed_effect(
        rng, effect_mask, 0.45 * config.effect_strength
    )
    logvariance_effect = _signed_effect(rng, effect_mask, config.effect_strength)
    tail_logit_effect = _signed_effect(
        rng, effect_mask, 1.5 * config.effect_strength, positive=True
    )

    # A one-factor covariance needs at least two target loadings.  Augment receptor
    # expression only in the new pure scenario, after every legacy random draw.
    if config.mechanism == "correlation_routing":
        expression = expression.copy()
        receptor_rows = np.flatnonzero(np.any(expression > 0, axis=1))
        if len(receptor_rows) < 2:
            missing = np.flatnonzero(~np.any(expression > 0, axis=1))
            expression[int(missing[0]), 0] = config.receptor_expression_min
    correlation_candidates = expression > 0
    correlation_mask = (
        rng.random((n, k)) < config.effect_density
    ) & correlation_candidates
    if config.mechanism == "correlation_routing":
        selected_rows = set(np.flatnonzero(np.any(correlation_mask, axis=1)).tolist())
        for target in np.flatnonzero(np.any(correlation_candidates, axis=1)):
            if len(selected_rows) >= 2:
                break
            if int(target) in selected_rows:
                continue
            modulator = int(np.flatnonzero(correlation_candidates[target])[0])
            correlation_mask[target, modulator] = True
            selected_rows.add(int(target))
    correlation_loading_effect = _signed_effect(
        rng, correlation_mask, config.effect_strength
    )

    params = MechanisticParameters(
        baseline_connectivity=a,
        neural_bias=neural_bias,
        release_weights=release_weights,
        release_bias=np.full(k, config.release_bias, dtype=float),
        clearance_rho=clearance_rho,
        receptor_expression=expression,
        receptor_kd=receptor_kd,
        hill_coefficient=hill_coefficient,
        additive_effect=additive_effect,
        synaptic_effect=synaptic_effect,
        intrinsic_log_slope_effect=intrinsic_log_slope_effect,
        intrinsic_threshold_effect=intrinsic_threshold_effect,
        logvariance_effect=logvariance_effect,
        tail_logit_effect=tail_logit_effect,
        correlation_loading_effect=correlation_loading_effect,
        base_variance=np.full(n, config.base_noise**2, dtype=float),
    )
    params.validate(config)
    return params


@dataclass(frozen=True)
class MechanisticState:
    neural: np.ndarray  # [N]
    concentration: np.ndarray  # [K], strictly positive
    calcium: np.ndarray  # [N]

    def validate(self, config: MechanisticConfig) -> None:
        if np.asarray(self.neural).shape != (config.n_neurons,):
            raise ValueError("neural state has wrong shape")
        if np.asarray(self.concentration).shape != (config.n_modulators,):
            raise ValueError("concentration state has wrong shape")
        if np.asarray(self.calcium).shape != (config.n_neurons,):
            raise ValueError("calcium state has wrong shape")
        if not (
            np.isfinite(self.neural).all()
            and np.isfinite(self.concentration).all()
            and np.isfinite(self.calcium).all()
        ):
            raise ValueError("state contains non-finite values")
        if np.any(self.concentration <= 0):
            raise ValueError("concentrations must be strictly positive")


@dataclass(frozen=True)
class OneStepMoments:
    """Exact complete-state one-step oracle and derivatives with respect to ``x_t``."""

    release: np.ndarray
    next_concentration: np.ndarray
    raw_occupancy: np.ndarray
    occupancy: np.ndarray
    conditional_mean: np.ndarray
    conditional_variance: np.ndarray
    conditional_covariance: np.ndarray
    conditional_correlation: np.ndarray
    correlation_loading: np.ndarray
    centered_fourth_moment: np.ndarray
    upper_tail_probability: np.ndarray
    shape_tail_probability: np.ndarray
    mixture_probability: np.ndarray
    calcium_mean: np.ndarray
    calcium_variance: np.ndarray
    fluorescence_mean: np.ndarray
    fluorescence_variance: np.ndarray
    release_jacobian: np.ndarray
    concentration_jacobian: np.ndarray
    occupancy_jacobian: np.ndarray
    occupancy_concentration_jacobian: np.ndarray
    mean_jacobian: np.ndarray
    logvariance_jacobian: np.ndarray
    variance_jacobian: np.ndarray
    covariance_jacobian: np.ndarray
    correlation_jacobian: np.ndarray
    correlation_loading_jacobian: np.ndarray
    covariance_concentration_susceptibility: np.ndarray
    correlation_concentration_susceptibility: np.ndarray
    correlation_loading_concentration_susceptibility: np.ndarray
    upper_tail_jacobian: np.ndarray
    shape_tail_jacobian: np.ndarray
    calcium_covariance: np.ndarray
    fluorescence_covariance: np.ndarray
    calcium_mean_jacobian: np.ndarray
    calcium_variance_jacobian: np.ndarray


def _validate_knockout_mask(
    knockout_mask: np.ndarray | None, config: MechanisticConfig
) -> np.ndarray:
    if knockout_mask is None:
        return np.zeros((config.n_neurons, config.n_modulators), dtype=bool)
    mask = np.asarray(knockout_mask, dtype=bool)
    if mask.shape != (config.n_neurons, config.n_modulators):
        raise ValueError(
            "knockout mask must have shape "
            f"{(config.n_neurons, config.n_modulators)}, got {mask.shape}"
        )
    return mask


def one_step_moments(
    config: MechanisticConfig,
    params: MechanisticParameters,
    state: MechanisticState,
    *,
    knockout_mask: np.ndarray | None = None,
) -> OneStepMoments:
    """Return exact one-step moments and state Jacobians.

    The receptor/concentration chain is deterministic conditional on the current state.
    Innovation randomness enters only after the returned conditional law is formed.
    ``knockout_mask[i,k]`` removes receptor ``k`` from target neuron ``i`` without
    changing release or concentration.
    """

    config.validate()
    params.validate(config)
    state.validate(config)
    knockout = _validate_knockout_mask(knockout_mask, config)
    active = _active_mechanisms(config.mechanism)
    n, k = config.n_neurons, config.n_modulators

    x = np.asarray(state.neural, dtype=float)
    concentration = np.asarray(state.concentration, dtype=float)
    tanh_x = np.tanh(x)
    sech2_x = 1.0 - tanh_x**2

    # Positive release and clearance.
    release_linear = params.release_bias + params.release_weights @ tanh_x
    release = _softplus(release_linear)
    release_jac = (
        _sigmoid(release_linear)[:, None]
        * params.release_weights
        * sech2_x[None, :]
    )
    next_concentration = (
        params.clearance_rho * concentration + (1.0 - params.clearance_rho) * release
    )
    concentration_jac = (1.0 - params.clearance_rho)[:, None] * release_jac

    # Hill receptor occupancy and its chain-rule derivative through release.
    raw_occupancy, occupancy_dc = hill_occupancy(
        next_concentration,
        params.receptor_kd,
        params.hill_coefficient,
        params.receptor_expression,
    )
    raw_occupancy_jac = occupancy_dc[:, :, None] * concentration_jac[None, :, :]
    occupancy_concentration_jac = (
        occupancy_dc[:, :, None]
        * params.clearance_rho[None, :, None]
        * np.eye(k)[None, :, :]
    )
    occupancy = raw_occupancy.copy()
    occupancy_jac = raw_occupancy_jac.copy()
    occupancy[knockout] = 0.0
    occupancy_jac[knockout] = 0.0
    occupancy_concentration_jac[knockout] = 0.0

    # Select the mechanism entry tensors without changing shared parameters.
    additive = (
        params.additive_effect
        if "additive_mean" in active
        else np.zeros_like(params.additive_effect)
    )
    synaptic = (
        params.synaptic_effect
        if "synaptic_gain" in active
        else np.zeros_like(params.synaptic_effect)
    )
    intrinsic_slope = (
        params.intrinsic_log_slope_effect
        if "intrinsic_excitability" in active
        else np.zeros_like(params.intrinsic_log_slope_effect)
    )
    intrinsic_threshold = (
        params.intrinsic_threshold_effect
        if "intrinsic_excitability" in active
        else np.zeros_like(params.intrinsic_threshold_effect)
    )
    variance_effect = (
        params.logvariance_effect
        if "innovation_variance" in active
        else np.zeros_like(params.logvariance_effect)
    )
    tail_effect = (
        params.tail_logit_effect
        if "matched_tail" in active
        else np.zeros_like(params.tail_logit_effect)
    )
    correlation_effect = (
        params.correlation_loading_effect
        if "correlation_routing" in active
        else np.zeros_like(params.correlation_loading_effect)
    )

    # Cell-intrinsic threshold and slope.
    log_kappa = np.einsum("ik,ik->i", intrinsic_slope, occupancy)
    dlog_kappa = np.einsum("ik,ikj->ij", intrinsic_slope, occupancy_jac)
    kappa = np.exp(log_kappa)
    threshold = np.einsum("ik,ik->i", intrinsic_threshold, occupancy)
    dthreshold = np.einsum("ik,ikj->ij", intrinsic_threshold, occupancy_jac)
    intrinsic_argument = kappa * (x - threshold)
    intrinsic_activation = np.tanh(intrinsic_argument)
    dintrinsic_argument = kappa[:, None] * (
        np.eye(n)
        - dthreshold
        + (x - threshold)[:, None] * dlog_kappa
    )
    dintrinsic = (1.0 - intrinsic_activation**2)[:, None] * dintrinsic_argument

    # Multiplicative postsynaptic receptor control of existing recurrent edges.
    log_synaptic_gain = np.einsum("isk,ik->is", synaptic, occupancy)
    dlog_synaptic_gain = np.einsum("isk,ikj->isj", synaptic, occupancy_jac)
    effective_connectivity = params.baseline_connectivity * np.exp(log_synaptic_gain)
    deffective_connectivity = (
        effective_connectivity[:, :, None] * dlog_synaptic_gain
    )
    recurrent_drive = effective_connectivity @ tanh_x
    recurrent_jac = (
        np.einsum("isj,s->ij", deffective_connectivity, tanh_x)
        + effective_connectivity * sech2_x[None, :]
    )

    additive_drive = np.einsum("ik,ik->i", additive, occupancy)
    additive_jac = np.einsum("ik,ikj->ij", additive, occupancy_jac)
    conditional_mean = (
        params.neural_bias
        + config.state_decay * intrinsic_activation
        + recurrent_drive
        + additive_drive
    )
    mean_jacobian = config.state_decay * dintrinsic + recurrent_jac + additive_jac

    # Innovation variance entry.  This is positive by log parameterization.
    log_variance = np.log(params.base_variance) + np.einsum(
        "ik,ik->i", variance_effect, occupancy
    )
    logvariance_jacobian = np.einsum(
        "ik,ikj->ij", variance_effect, occupancy_jac
    )
    conditional_variance = np.exp(log_variance)
    variance_jacobian = conditional_variance[:, None] * logvariance_jacobian

    # Globally positive-definite one-factor correlation routing.  For every state,
    # eta_i = lambda_i f + sqrt(1-lambda_i^2) epsilon_i has unit marginal variance,
    # R_ii=1, R_ij=lambda_i lambda_j, and lambda_min(R) >= 1-lambda_max^2 > 0.
    loading_argument = np.einsum("ik,ik->i", correlation_effect, occupancy)
    loading_argument_jac = np.einsum(
        "ik,ikj->ij", correlation_effect, occupancy_jac
    )
    loading_argument_concentration = np.einsum(
        "ik,ikl->il", correlation_effect, occupancy_concentration_jac
    )
    loading_tanh = np.tanh(loading_argument)
    correlation_loading = config.correlation_max_loading * loading_tanh
    loading_slope = config.correlation_max_loading * (1.0 - loading_tanh**2)
    correlation_loading_jacobian = loading_slope[:, None] * loading_argument_jac
    correlation_loading_concentration = (
        loading_slope[:, None] * loading_argument_concentration
    )

    conditional_correlation = np.outer(correlation_loading, correlation_loading)
    np.fill_diagonal(conditional_correlation, 1.0)
    correlation_jacobian = (
        np.einsum("is,j->ijs", correlation_loading_jacobian, correlation_loading)
        + np.einsum("i,js->ijs", correlation_loading, correlation_loading_jacobian)
    )
    correlation_concentration = (
        np.einsum(
            "ik,j->ijk",
            correlation_loading_concentration,
            correlation_loading,
        )
        + np.einsum(
            "i,jk->ijk",
            correlation_loading,
            correlation_loading_concentration,
        )
    )
    diagonal = np.arange(n)
    correlation_jacobian[diagonal, diagonal, :] = 0.0
    correlation_concentration[diagonal, diagonal, :] = 0.0

    root_variance = np.sqrt(conditional_variance)
    marginal_scale = np.outer(root_variance, root_variance)
    conditional_covariance = marginal_scale * conditional_correlation
    logvariance_concentration = np.einsum(
        "ik,ikl->il", variance_effect, occupancy_concentration_jac
    )
    logvariance_pair_jac = (
        logvariance_jacobian[:, None, :] + logvariance_jacobian[None, :, :]
    )
    logvariance_pair_concentration = (
        logvariance_concentration[:, None, :]
        + logvariance_concentration[None, :, :]
    )
    covariance_jacobian = marginal_scale[:, :, None] * (
        correlation_jacobian
        + 0.5 * conditional_correlation[:, :, None] * logvariance_pair_jac
    )
    covariance_concentration = marginal_scale[:, :, None] * (
        correlation_concentration
        + 0.5
        * conditional_correlation[:, :, None]
        * logvariance_pair_concentration
    )

    threshold_vector = np.full(n, config.tail_threshold, dtype=float)
    dlog_sd = 0.5 * logvariance_jacobian

    if "matched_tail" in active:
        base_logit = math.log(config.tail_base_probability / (1.0 - config.tail_base_probability))
        mixture_logit = base_logit + np.einsum("ik,ik->i", tail_effect, occupancy)
        mixture_probability = _sigmoid(mixture_logit)
        dmixture_logit = np.einsum("ik,ikj->ij", tail_effect, occupancy_jac)
        dp = mixture_probability[:, None] * (1.0 - mixture_probability[:, None]) * dmixture_logit

        low = config.tail_low_scale
        high = config.tail_high_scale
        low2, high2 = low**2, high**2
        delta2 = high2 - low2
        normalizer2 = low2 + mixture_probability * delta2
        dnormalizer2 = delta2 * dp
        root_normalizer = np.sqrt(normalizer2)
        sd_low = root_variance * low / root_normalizer
        sd_high = root_variance * high / root_normalizer
        z_low = (threshold_vector - conditional_mean) / sd_low
        z_high = (threshold_vector - conditional_mean) / sd_high
        survival_low = ndtr(-z_low)
        survival_high = ndtr(-z_high)
        upper_tail_probability = (
            (1.0 - mixture_probability) * survival_low
            + mixture_probability * survival_high
        )

        dlog_component_sd = dlog_sd - 0.5 * dnormalizer2 / normalizer2[:, None]
        dz_low = (
            -mean_jacobian / sd_low[:, None]
            - z_low[:, None] * dlog_component_sd
        )
        dz_high = (
            -mean_jacobian / sd_high[:, None]
            - z_high[:, None] * dlog_component_sd
        )
        dsurvival_low = -_normal_pdf(z_low)[:, None] * dz_low
        dsurvival_high = -_normal_pdf(z_high)[:, None] * dz_high
        upper_tail_jacobian = (
            dp * (survival_high - survival_low)[:, None]
            + (1.0 - mixture_probability)[:, None] * dsurvival_low
            + mixture_probability[:, None] * dsurvival_high
        )
        shape_z_low = config.shape_tail_z * root_normalizer / low
        shape_z_high = config.shape_tail_z * root_normalizer / high
        shape_survival_low = ndtr(-shape_z_low)
        shape_survival_high = ndtr(-shape_z_high)
        shape_tail_probability = 2.0 * (
            (1.0 - mixture_probability) * shape_survival_low
            + mixture_probability * shape_survival_high
        )
        dshape_z_low = (
            0.5 * shape_z_low[:, None] * dnormalizer2 / normalizer2[:, None]
        )
        dshape_z_high = (
            0.5 * shape_z_high[:, None] * dnormalizer2 / normalizer2[:, None]
        )
        dshape_survival_low = -_normal_pdf(shape_z_low)[:, None] * dshape_z_low
        dshape_survival_high = -_normal_pdf(shape_z_high)[:, None] * dshape_z_high
        shape_tail_jacobian = 2.0 * (
            dp * (shape_survival_high - shape_survival_low)[:, None]
            + (1.0 - mixture_probability)[:, None] * dshape_survival_low
            + mixture_probability[:, None] * dshape_survival_high
        )
        standardized_fourth = (
            3.0
            * (
                (1.0 - mixture_probability) * low**4
                + mixture_probability * high**4
            )
            / normalizer2**2
        )
        centered_fourth = conditional_variance**2 * standardized_fourth
    else:
        mixture_probability = np.zeros(n, dtype=float)
        z = (threshold_vector - conditional_mean) / root_variance
        upper_tail_probability = ndtr(-z)
        dz = -mean_jacobian / root_variance[:, None] - z[:, None] * dlog_sd
        upper_tail_jacobian = -_normal_pdf(z)[:, None] * dz
        shape_tail_probability = np.full(
            n, 2.0 * ndtr(-config.shape_tail_z), dtype=float
        )
        shape_tail_jacobian = np.zeros((n, n), dtype=float)
        centered_fourth = 3.0 * conditional_variance**2

    calcium_rho = config.calcium_rho
    calcium_mean = calcium_rho * state.calcium + (1.0 - calcium_rho) * conditional_mean
    calcium_variance = (1.0 - calcium_rho) ** 2 * conditional_variance
    calcium_covariance = (1.0 - calcium_rho) ** 2 * conditional_covariance
    fluorescence_mean = calcium_mean.copy()
    fluorescence_variance = calcium_variance + config.measurement_noise**2
    fluorescence_covariance = calcium_covariance.copy()
    fluorescence_covariance[diagonal, diagonal] += config.measurement_noise**2
    calcium_mean_jacobian = (1.0 - calcium_rho) * mean_jacobian
    calcium_variance_jacobian = (1.0 - calcium_rho) ** 2 * variance_jacobian

    return OneStepMoments(
        release=release,
        next_concentration=next_concentration,
        raw_occupancy=raw_occupancy,
        occupancy=occupancy,
        conditional_mean=conditional_mean,
        conditional_variance=conditional_variance,
        conditional_covariance=conditional_covariance,
        conditional_correlation=conditional_correlation,
        correlation_loading=correlation_loading,
        centered_fourth_moment=centered_fourth,
        upper_tail_probability=upper_tail_probability,
        shape_tail_probability=shape_tail_probability,
        mixture_probability=mixture_probability,
        calcium_mean=calcium_mean,
        calcium_variance=calcium_variance,
        fluorescence_mean=fluorescence_mean,
        fluorescence_variance=fluorescence_variance,
        release_jacobian=release_jac,
        concentration_jacobian=concentration_jac,
        occupancy_jacobian=occupancy_jac,
        occupancy_concentration_jacobian=occupancy_concentration_jac,
        mean_jacobian=mean_jacobian,
        logvariance_jacobian=logvariance_jacobian,
        variance_jacobian=variance_jacobian,
        covariance_jacobian=covariance_jacobian,
        correlation_jacobian=correlation_jacobian,
        correlation_loading_jacobian=correlation_loading_jacobian,
        covariance_concentration_susceptibility=covariance_concentration,
        correlation_concentration_susceptibility=correlation_concentration,
        correlation_loading_concentration_susceptibility=(
            correlation_loading_concentration
        ),
        upper_tail_jacobian=upper_tail_jacobian,
        shape_tail_jacobian=shape_tail_jacobian,
        calcium_covariance=calcium_covariance,
        fluorescence_covariance=fluorescence_covariance,
        calcium_mean_jacobian=calcium_mean_jacobian,
        calcium_variance_jacobian=calcium_variance_jacobian,
    )


@dataclass(frozen=True)
class StabilityCertificate:
    """Conservative global bounds for the bounded neural transition law."""

    baseline_max_row_l1: float
    modulated_max_row_l1: float
    conditional_mean_abs_bound: float
    variance_lower_bound: float
    variance_upper_bound: float
    correlation_eigenvalue_lower_bound: float


def stability_certificate(
    config: MechanisticConfig, params: MechanisticParameters
) -> StabilityCertificate:
    """Return bounds valid for all neural states and receptor occupancies."""

    config.validate()
    params.validate(config)
    active = _active_mechanisms(config.mechanism)
    max_o = params.receptor_expression
    if "synaptic_gain" in active:
        max_log_gain = np.einsum("isk,ik->is", np.abs(params.synaptic_effect), max_o)
    else:
        max_log_gain = np.zeros_like(params.baseline_connectivity)
    max_abs_connectivity = np.abs(params.baseline_connectivity) * np.exp(max_log_gain)
    modulated_rows = np.sum(max_abs_connectivity, axis=1)

    if "additive_mean" in active:
        additive_bound = np.einsum("ik,ik->i", np.abs(params.additive_effect), max_o)
    else:
        additive_bound = np.zeros(config.n_neurons)
    mean_bound = np.max(
        np.abs(params.neural_bias)
        + config.state_decay
        + modulated_rows
        + additive_bound
    )

    if "innovation_variance" in active:
        positive = np.maximum(params.logvariance_effect, 0.0)
        negative = np.minimum(params.logvariance_effect, 0.0)
        upper_log = np.log(params.base_variance) + np.einsum("ik,ik->i", positive, max_o)
        lower_log = np.log(params.base_variance) + np.einsum("ik,ik->i", negative, max_o)
    else:
        upper_log = lower_log = np.log(params.base_variance)
    return StabilityCertificate(
        baseline_max_row_l1=float(np.max(np.sum(np.abs(params.baseline_connectivity), axis=1))),
        modulated_max_row_l1=float(np.max(modulated_rows)),
        conditional_mean_abs_bound=float(mean_bound),
        variance_lower_bound=float(np.min(np.exp(lower_log))),
        variance_upper_bound=float(np.max(np.exp(upper_log))),
        correlation_eigenvalue_lower_bound=(
            1.0 - config.correlation_max_loading**2
            if "correlation_routing" in active
            else 1.0
        ),
    )


@dataclass(frozen=True)
class ExogenousNoise:
    """Named exogenous innovations shared by factual and knockout simulations."""

    initial_neural: np.ndarray
    neural_standard_normal: np.ndarray
    correlation_common_normal: np.ndarray
    mixture_uniform: np.ndarray
    fluorescence_standard_normal: np.ndarray
    ligand_impulse: np.ndarray
    fingerprint: str


def _fingerprint(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def draw_exogenous_noise(config: MechanisticConfig, seed: int | None = None) -> ExogenousNoise:
    """Draw simulation randomness from stable, semantically named streams."""

    config.validate()
    root = config.seed if seed is None else int(seed)
    streams = NamedRNGStreams(root)
    initial = 0.1 * streams.generator("initial_neural").standard_normal(config.n_neurons)
    neural = streams.generator("neural_innovation").standard_normal(
        (config.total_steps, config.n_neurons)
    )
    correlation_common = streams.generator("correlation_common_factor").standard_normal(
        config.total_steps
    )
    mixture = streams.generator("tail_component").random(
        (config.total_steps, config.n_neurons)
    )
    fluorescence = streams.generator("fluorescence_noise").standard_normal(
        (config.total_steps, config.n_neurons)
    )
    ligand_impulse = (
        streams.generator("ligand_pulse").random(
            (config.total_steps, config.n_modulators)
        )
        < config.ligand_pulse_rate_hz * config.dt_seconds
    ).astype(float)
    fingerprint_arrays = [initial, neural, mixture, fluorescence, ligand_impulse]
    if config.mechanism == "correlation_routing":
        fingerprint_arrays.append(correlation_common)
    return ExogenousNoise(
        initial_neural=initial,
        neural_standard_normal=neural,
        correlation_common_normal=correlation_common,
        mixture_uniform=mixture,
        fluorescence_standard_normal=fluorescence,
        ligand_impulse=ligand_impulse,
        fingerprint=_fingerprint(fingerprint_arrays),
    )


@dataclass(frozen=True)
class MechanisticTrajectory:
    latent_neural: np.ndarray
    release: np.ndarray
    concentration: np.ndarray
    raw_occupancy: np.ndarray
    occupancy: np.ndarray
    clean_calcium: np.ndarray
    fluorescence: np.ndarray
    conditional_mean: np.ndarray
    conditional_variance: np.ndarray
    conditional_covariance: np.ndarray | None
    conditional_correlation: np.ndarray | None
    correlation_loading: np.ndarray | None
    centered_fourth_moment: np.ndarray
    upper_tail_probability: np.ndarray
    shape_tail_probability: np.ndarray
    mixture_probability: np.ndarray
    mean_jacobian: np.ndarray
    logvariance_jacobian: np.ndarray
    upper_tail_jacobian: np.ndarray
    shape_tail_jacobian: np.ndarray
    covariance_jacobian: np.ndarray | None
    correlation_jacobian: np.ndarray | None
    covariance_concentration_susceptibility: np.ndarray | None
    correlation_concentration_susceptibility: np.ndarray | None
    ligand_drive: np.ndarray
    ligand_impulse: np.ndarray
    exogenous_fingerprint: str
    stability: StabilityCertificate
    mechanism: str


def _validate_exogenous(config: MechanisticConfig, exogenous: ExogenousNoise) -> None:
    n, total = config.n_neurons, config.total_steps
    if exogenous.initial_neural.shape != (n,):
        raise ValueError("initial_neural has wrong shape")
    expected = (total, n)
    for name in (
        "neural_standard_normal",
        "mixture_uniform",
        "fluorescence_standard_normal",
    ):
        if getattr(exogenous, name).shape != expected:
            raise ValueError(f"{name} has wrong shape")
    if exogenous.correlation_common_normal.shape != (total,):
        raise ValueError("correlation_common_normal has wrong shape")
    if exogenous.ligand_impulse.shape != (total, config.n_modulators):
        raise ValueError("ligand_impulse has wrong shape")


def simulate_mechanistic(
    config: MechanisticConfig,
    params: MechanisticParameters | None = None,
    *,
    simulation_seed: int | None = None,
    knockout_mask: np.ndarray | None = None,
    exogenous: ExogenousNoise | None = None,
) -> MechanisticTrajectory:
    """Simulate one trajectory with exact pointwise oracle arrays."""

    config.validate()
    params = generate_mechanistic_parameters(config) if params is None else params
    params.validate(config)
    knockout = _validate_knockout_mask(knockout_mask, config)
    if exogenous is None:
        exogenous = draw_exogenous_noise(config, simulation_seed)
    _validate_exogenous(config, exogenous)

    total, n, k = config.total_steps, config.n_neurons, config.n_modulators
    x = np.asarray(exogenous.initial_neural, dtype=float).copy()
    concentration = np.full(k, config.concentration_initial, dtype=float)
    calcium = np.zeros(n, dtype=float)
    state = MechanisticState(x, concentration, calcium)

    latent = np.empty((total, n))
    releases = np.empty((total, k))
    concentrations = np.empty((total, k))
    raw_occupancies = np.empty((total, n, k))
    occupancies = np.empty((total, n, k))
    clean_calcium = np.empty((total, n))
    fluorescence = np.empty((total, n))
    means = np.empty((total, n))
    variances = np.empty((total, n))
    correlation_active = "correlation_routing" in _active_mechanisms(config.mechanism)
    covariances = np.empty((total, n, n)) if correlation_active else None
    correlations = np.empty((total, n, n)) if correlation_active else None
    correlation_loadings = np.empty((total, n)) if correlation_active else None
    fourth = np.empty((total, n))
    tail_probability = np.empty((total, n))
    shape_tail_probability = np.empty((total, n))
    mixture_probability = np.empty((total, n))
    mean_jacobians = np.empty((total, n, n))
    logvariance_jacobians = np.empty((total, n, n))
    tail_jacobians = np.empty((total, n, n))
    shape_tail_jacobians = np.empty((total, n, n))
    covariance_jacobians = (
        np.empty((total, n, n, n)) if correlation_active else None
    )
    correlation_jacobians = (
        np.empty((total, n, n, n)) if correlation_active else None
    )
    covariance_concentration = (
        np.empty((total, n, n, k)) if correlation_active else None
    )
    correlation_concentration = (
        np.empty((total, n, n, k)) if correlation_active else None
    )
    ligand_drives = np.empty((total, k))

    tail_active = "matched_tail" in _active_mechanisms(config.mechanism)
    low, high = config.tail_low_scale, config.tail_high_scale
    calcium_rho = config.calcium_rho
    ligand_rho = math.exp(-config.dt_seconds / config.ligand_pulse_decay_seconds)
    ligand_drive = np.zeros(k, dtype=float)

    for t in range(total):
        ligand_drive = (
            ligand_rho * ligand_drive
            + config.ligand_pulse_amplitude * exogenous.ligand_impulse[t]
        )
        step_params = (
            params
            if not np.any(ligand_drive)
            else replace(params, release_bias=params.release_bias + ligand_drive)
        )
        oracle = one_step_moments(config, step_params, state, knockout_mask=knockout)
        if correlation_active:
            loading = oracle.correlation_loading
            standardized_innovation = (
                loading * exogenous.correlation_common_normal[t]
                + np.sqrt(1.0 - loading**2)
                * exogenous.neural_standard_normal[t]
            )
            innovation = np.sqrt(oracle.conditional_variance) * standardized_innovation
        elif tail_active:
            p = oracle.mixture_probability
            high_component = exogenous.mixture_uniform[t] < p
            normalizer2 = (1.0 - p) * low**2 + p * high**2
            component_scale = np.where(high_component, high, low) / np.sqrt(normalizer2)
            # Preserve the legacy multiplication order bit-for-bit.
            innovation = (
                np.sqrt(oracle.conditional_variance)
                * component_scale
                * exogenous.neural_standard_normal[t]
            )
        else:
            component_scale = np.ones(n)
            # Preserve the legacy multiplication by a unit component scale.
            innovation = (
                np.sqrt(oracle.conditional_variance)
                * component_scale
                * exogenous.neural_standard_normal[t]
            )
        next_x = oracle.conditional_mean + innovation
        next_calcium = calcium_rho * state.calcium + (1.0 - calcium_rho) * next_x
        next_fluorescence = (
            next_calcium
            + config.measurement_noise * exogenous.fluorescence_standard_normal[t]
        )

        latent[t] = next_x
        releases[t] = oracle.release
        concentrations[t] = oracle.next_concentration
        raw_occupancies[t] = oracle.raw_occupancy
        occupancies[t] = oracle.occupancy
        clean_calcium[t] = next_calcium
        fluorescence[t] = next_fluorescence
        means[t] = oracle.conditional_mean
        variances[t] = oracle.conditional_variance
        if correlation_active:
            covariances[t] = oracle.conditional_covariance
            correlations[t] = oracle.conditional_correlation
            correlation_loadings[t] = oracle.correlation_loading
            covariance_jacobians[t] = oracle.covariance_jacobian
            correlation_jacobians[t] = oracle.correlation_jacobian
            covariance_concentration[t] = (
                oracle.covariance_concentration_susceptibility
            )
            correlation_concentration[t] = (
                oracle.correlation_concentration_susceptibility
            )
        fourth[t] = oracle.centered_fourth_moment
        tail_probability[t] = oracle.upper_tail_probability
        shape_tail_probability[t] = oracle.shape_tail_probability
        mixture_probability[t] = oracle.mixture_probability
        mean_jacobians[t] = oracle.mean_jacobian
        logvariance_jacobians[t] = oracle.logvariance_jacobian
        tail_jacobians[t] = oracle.upper_tail_jacobian
        shape_tail_jacobians[t] = oracle.shape_tail_jacobian
        ligand_drives[t] = ligand_drive

        state = MechanisticState(next_x, oracle.next_concentration, next_calcium)

    sl = slice(config.burn_in, total)
    return MechanisticTrajectory(
        latent_neural=latent[sl].copy(),
        release=releases[sl].copy(),
        concentration=concentrations[sl].copy(),
        raw_occupancy=raw_occupancies[sl].copy(),
        occupancy=occupancies[sl].copy(),
        clean_calcium=clean_calcium[sl].copy(),
        fluorescence=fluorescence[sl].copy(),
        conditional_mean=means[sl].copy(),
        conditional_variance=variances[sl].copy(),
        conditional_covariance=(
            None if covariances is None else covariances[sl].copy()
        ),
        conditional_correlation=(
            None if correlations is None else correlations[sl].copy()
        ),
        correlation_loading=(
            None
            if correlation_loadings is None
            else correlation_loadings[sl].copy()
        ),
        centered_fourth_moment=fourth[sl].copy(),
        upper_tail_probability=tail_probability[sl].copy(),
        shape_tail_probability=shape_tail_probability[sl].copy(),
        mixture_probability=mixture_probability[sl].copy(),
        mean_jacobian=mean_jacobians[sl].copy(),
        logvariance_jacobian=logvariance_jacobians[sl].copy(),
        upper_tail_jacobian=tail_jacobians[sl].copy(),
        shape_tail_jacobian=shape_tail_jacobians[sl].copy(),
        covariance_jacobian=(
            None
            if covariance_jacobians is None
            else covariance_jacobians[sl].copy()
        ),
        correlation_jacobian=(
            None
            if correlation_jacobians is None
            else correlation_jacobians[sl].copy()
        ),
        covariance_concentration_susceptibility=(
            None
            if covariance_concentration is None
            else covariance_concentration[sl].copy()
        ),
        correlation_concentration_susceptibility=(
            None
            if correlation_concentration is None
            else correlation_concentration[sl].copy()
        ),
        ligand_drive=ligand_drives[sl].copy(),
        ligand_impulse=exogenous.ligand_impulse[sl].copy(),
        exogenous_fingerprint=exogenous.fingerprint,
        stability=stability_certificate(config, params),
        mechanism=config.mechanism,
    )


@dataclass(frozen=True)
class KnockoutPair:
    factual: MechanisticTrajectory
    receptor_knockout: MechanisticTrajectory
    parameters: MechanisticParameters
    knockout_mask: np.ndarray


def simulate_receptor_knockout_pair(
    config: MechanisticConfig,
    params: MechanisticParameters | None = None,
    *,
    simulation_seed: int | None = None,
    knockout_mask: np.ndarray | None = None,
) -> KnockoutPair:
    """Simulate factual and receptor-knockout trajectories with common random numbers."""

    config.validate()
    params = generate_mechanistic_parameters(config) if params is None else params
    params.validate(config)
    if knockout_mask is None:
        knockout_mask = params.receptor_expression > 0
    mask = _validate_knockout_mask(knockout_mask, config)
    exogenous = draw_exogenous_noise(config, simulation_seed)
    factual = simulate_mechanistic(config, params, exogenous=exogenous)
    knockout = simulate_mechanistic(
        config,
        params,
        knockout_mask=mask,
        exogenous=exogenous,
    )
    return KnockoutPair(
        factual=factual,
        receptor_knockout=knockout,
        parameters=params,
        knockout_mask=mask.copy(),
    )


def with_mechanism(config: MechanisticConfig, mechanism: str) -> MechanisticConfig:
    """Convenience helper retaining every non-mechanism configuration field."""

    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    return replace(config, mechanism=mechanism)


__all__ = [
    "ExogenousNoise",
    "KnockoutPair",
    "MECHANISMS",
    "MechanisticConfig",
    "MechanisticParameters",
    "MechanisticState",
    "MechanisticTrajectory",
    "NamedRNGStreams",
    "OneStepMoments",
    "StabilityCertificate",
    "draw_exogenous_noise",
    "generate_mechanistic_parameters",
    "one_step_moments",
    "simulate_mechanistic",
    "simulate_receptor_knockout_pair",
    "stability_certificate",
    "with_mechanism",
]
