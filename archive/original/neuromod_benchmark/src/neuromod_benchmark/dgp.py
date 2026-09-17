"""Mechanistically explicit stochastic neuromodulator simulator.

The simulator exposes exact conditional-law and structural oracles.  It is not a
claim that one set of equations is a complete biological model.  It is a family
of controlled instantiations whose mechanisms can be switched independently so
that recovery claims have falsifiable ground truth.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .schema import DGPConfig, Dataset, StructuralParameters, Trajectory


def _spectral_scale(matrix: np.ndarray, radius: float) -> np.ndarray:
    eig = np.linalg.eigvals(matrix)
    current = float(np.max(np.abs(eig))) if eig.size else 0.0
    return matrix if current <= radius or current == 0 else matrix * (radius / current)


def _signed_sparse(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    density: float,
    scale: float,
) -> np.ndarray:
    mask = rng.random(shape) < density
    values = rng.normal(0.0, scale, size=shape)
    out = mask * values
    # Avoid arbitrarily tiny "true" effects that make support scoring ill-posed.
    active = np.abs(out) > 0
    out[active] += np.sign(out[active]) * 0.25 * scale
    return out


def generate_parameters(config: DGPConfig, seed: int | None = None) -> StructuralParameters:
    """Draw one parameter set; all trajectories then share this ground truth."""

    config.validate()
    rng = np.random.default_rng(config.seed if seed is None else seed)
    n, k = config.n_neurons, config.n_modulators

    a = _signed_sparse(rng, (n, n), config.network_density, 0.35)
    np.fill_diagonal(a, 0.0)
    a = _spectral_scale(a, config.coupling_radius)

    release = np.zeros((k, n))
    for mod in range(k):
        count = max(1, min(n, int(np.ceil(config.network_density * n))))
        sources = rng.choice(n, size=count, replace=False)
        release[mod, sources] = rng.choice([-1.0, 1.0], size=count)
    release *= config.release_strength

    receptor_scale = config.effect_strength
    add = _signed_sparse(rng, (n, k), config.receptor_density, receptor_scale)
    intrinsic = _signed_sparse(rng, (n, k), config.receptor_density, receptor_scale)
    variance = _signed_sparse(rng, (n, k), config.receptor_density, receptor_scale)
    tail = _signed_sparse(rng, (n, k), config.receptor_density, receptor_scale)

    gain = np.zeros((k, n, n))
    for mod in range(k):
        gain[mod] = _signed_sparse(
            rng, (n, n), config.receptor_density * config.network_density, receptor_scale
        )
        np.fill_diagonal(gain[mod], 0.0)
        # Modulation of a synapse is only defined where the baseline wiring exists.
        gain[mod] *= a != 0

    direct_logvar = variance @ release
    max_abs = float(np.max(np.abs(direct_logvar))) if direct_logvar.size else 0.0
    if max_abs > receptor_scale:
        direct_logvar *= receptor_scale / max_abs

    mechanisms = _expanded_mechanisms(config.mechanism)
    if "additive_mean" not in mechanisms:
        add.fill(0)
    if "intrinsic_gain" not in mechanisms:
        intrinsic.fill(0)
    if "innovation_variance" not in mechanisms:
        variance.fill(0)
        direct_logvar.fill(0)
    if "tail_burst" not in mechanisms:
        tail.fill(0)
    if "synaptic_gain" not in mechanisms:
        gain.fill(0)

    params = StructuralParameters(
        baseline_connectivity=a,
        release_map=release,
        additive_receptors=add,
        intrinsic_receptors=intrinsic,
        variance_receptors=variance,
        tail_receptors=tail,
        synaptic_gain=gain,
        direct_logvariance=direct_logvar,
        input_loading=rng.normal(0, 0.18, size=n),
        modulator_input_loading=rng.normal(0.25, 0.08, size=k),
        hidden_loading=rng.normal(0, config.hidden_driver_strength, size=n),
        base_logvariance=np.full(n, 2.0 * np.log(config.base_noise)),
    )
    params.validate(n, k)
    return params


def _expanded_mechanisms(name: str) -> set[str]:
    if name == "mixed":
        return {
            "additive_mean",
            "synaptic_gain",
            "intrinsic_gain",
            "innovation_variance",
            "tail_burst",
        }
    if name == "null":
        return set()
    return {name}


def _stimulus(rng: np.random.Generator, length: int, rate: float) -> np.ndarray:
    impulses = (rng.random(length) < rate) * rng.choice([-1.0, 1.0], size=length)
    signal = np.empty(length)
    state = 0.0
    for t in range(length):
        state = 0.8 * state + impulses[t]
        signal[t] = state
    return signal


def simulate_trajectory(
    config: DGPConfig,
    params: StructuralParameters,
    seed: int,
    *,
    disable_modulation: bool = False,
    modulator_clamp: np.ndarray | None = None,
) -> Trajectory:
    """Simulate one trajectory, optionally under a matched-noise intervention."""

    config.validate()
    rng = np.random.default_rng(seed)
    n, k = config.n_neurons, config.n_modulators
    total = config.n_steps + config.burn_in
    stimulus = _stimulus(rng, total, config.stimulus_rate)
    state_noise = rng.standard_normal((total, n))
    mod_noise = rng.standard_normal((total, k))
    obs_noise = rng.standard_normal((total, n))
    burst_uniform = rng.random((total, n))
    hidden_noise = rng.standard_normal(total)
    missing_uniform = rng.random((total, n))

    x = np.zeros((total, n))
    calcium = np.zeros((total, n))
    mod = np.zeros((total, k))
    cond_mean = np.zeros((total, n))
    cond_var = np.zeros((total, n))
    burst_prob = np.zeros((total, n))
    hidden = np.zeros(total)
    mean_jac_sum = np.zeros((n, n))
    logvar_jac_sum = np.zeros((n, n))
    tail_jac_sum = np.zeros((n, n))
    point_mean = [] if config.store_pointwise_jacobians else None
    point_logvar = [] if config.store_pointwise_jacobians else None
    point_tail = [] if config.store_pointwise_jacobians else None

    x[0] = 0.05 * state_noise[0]
    hidden[0] = hidden_noise[0]
    mechanisms = _expanded_mechanisms(config.mechanism)
    cp = None if config.change_fraction is None else int(total * config.change_fraction)

    for t in range(total - 1):
        hidden[t + 1] = 0.9 * hidden[t] + 0.25 * hidden_noise[t + 1]
        activity = np.tanh(x[t])
        release_signal = np.tanh(params.release_map @ activity)
        next_mod = (
            config.modulator_rho * mod[t]
            + (1.0 - config.modulator_rho) * release_signal
            + params.modulator_input_loading * stimulus[t]
            + config.modulator_noise * mod_noise[t + 1]
        )
        next_mod = np.clip(next_mod, -3.0, 3.0)
        if disable_modulation or not mechanisms:
            next_mod.fill(0.0)
        if modulator_clamp is not None:
            next_mod[:] = np.asarray(modulator_clamp, dtype=float)
        mod[t + 1] = next_mod

        effect = 1.0 if cp is None or t < cp else config.change_multiplier
        m = effect * mod[t]
        a_eff = params.baseline_connectivity + np.tensordot(m, params.synaptic_gain, axes=1)
        intrinsic = params.intrinsic_receptors @ m
        mu = (
            config.state_decay * x[t]
            + a_eff @ activity
            + intrinsic * activity
            + params.additive_receptors @ m
            + params.input_loading * stimulus[t]
            + params.hidden_loading * hidden[t]
        )

        logvar = (
            params.base_logvariance
            + params.variance_receptors @ m
            + effect * (params.direct_logvariance @ activity)
        )
        logvar = np.clip(logvar, -8.0, 1.5)
        base_var = np.exp(logvar)

        if "tail_burst" in mechanisms:
            logits = -3.66 + params.tail_receptors @ m
            p_burst = 1.0 / (1.0 + np.exp(-np.clip(logits, -12, 12)))
        else:
            p_burst = np.zeros(n)
        burst = burst_uniform[t + 1] < p_burst
        scale_multiplier = np.where(burst, config.burst_scale, 1.0)
        innovation = np.sqrt(base_var) * scale_multiplier * state_noise[t + 1]
        x[t + 1] = mu + innovation
        calcium[t + 1] = (
            config.calcium_decay * calcium[t]
            + (1.0 - config.calcium_decay) * x[t + 1]
            + config.measurement_noise * obs_noise[t + 1]
        )

        cond_mean[t + 1] = mu
        cond_var[t + 1] = base_var * (
            (1.0 - p_burst) + p_burst * config.burst_scale**2
        )
        burst_prob[t + 1] = p_burst

        sech2 = 1.0 - activity**2
        mean_jac = (
            config.state_decay * np.eye(n)
            + a_eff * sech2[None, :]
            + np.diag(intrinsic * sech2)
        )
        logvar_jac = effect * params.direct_logvariance * sech2[None, :]
        # Tail is mediated by the stateful modulator, so its instantaneous fixed-m
        # Jacobian is zero.  The release/receptor maps retain the delayed oracle.
        tail_jac = np.zeros((n, n))
        if t >= config.burn_in:
            mean_jac_sum += mean_jac
            logvar_jac_sum += logvar_jac
            tail_jac_sum += tail_jac
            if point_mean is not None:
                point_mean.append(mean_jac.copy())
                point_logvar.append(logvar_jac.copy())
                point_tail.append(tail_jac.copy())

    sl = slice(config.burn_in, total)
    mask = missing_uniform[sl] >= config.missing_rate
    observed_calcium = calcium[sl].copy()
    observed_calcium[~mask] = np.nan
    count = max(1, total - config.burn_in - 1)

    def align_pointwise(values):
        if values is None:
            return None
        array = np.asarray(values)
        # Pointwise oracle index is the target time in the retained trajectory.
        # The first retained target has no retained predecessor and is never used
        # by a positive-horizon supervised row.
        return np.concatenate([np.zeros_like(array[:1]), array], axis=0)

    return Trajectory(
        latent=x[sl],
        calcium=observed_calcium,
        modulator=mod[sl],
        stimulus=stimulus[sl],
        conditional_mean=cond_mean[sl],
        conditional_variance=cond_var[sl],
        burst_probability=burst_prob[sl],
        hidden_driver=hidden[sl],
        observed_mask=mask,
        average_mean_jacobian=mean_jac_sum / count,
        average_logvariance_jacobian=logvar_jac_sum / count,
        average_tail_jacobian=tail_jac_sum / count,
        pointwise_mean_jacobian=align_pointwise(point_mean),
        pointwise_logvariance_jacobian=align_pointwise(point_logvar),
        pointwise_tail_jacobian=align_pointwise(point_tail),
    )


def mechanism_support(params: StructuralParameters) -> dict[str, np.ndarray]:
    """Return nonzero structural supports with explicit semantic channel names."""

    eps = 1e-12
    mediated_mean = params.additive_receptors @ params.release_map
    mediated_variance = params.variance_receptors @ params.release_map
    mediated_tail = params.tail_receptors @ params.release_map
    return {
        "wired_connectivity": np.abs(params.baseline_connectivity) > eps,
        "conditional_mean": np.abs(params.baseline_connectivity) > eps,
        "neuromod_additive_mean": np.abs(mediated_mean) > eps,
        "neuromod_synaptic_gain": np.any(np.abs(params.synaptic_gain) > eps, axis=0),
        "neuromod_intrinsic_gain": np.abs(params.intrinsic_receptors) > eps,
        "conditional_logvariance": np.abs(params.direct_logvariance) > eps,
        "neuromod_variance": np.abs(mediated_variance) > eps,
        "neuromod_tail": np.abs(mediated_tail) > eps,
        "release": np.abs(params.release_map) > eps,
    }


def simulate_dataset(config: DGPConfig) -> Dataset:
    """Simulate independent trajectories plus matched no-modulation counterfactuals."""

    config.validate()
    params = generate_parameters(config)
    trajectory_seeds = [config.seed * 100_003 + 10_007 + i for i in range(config.n_trajectories)]
    trajectories = [simulate_trajectory(config, params, seed) for seed in trajectory_seeds]
    no_modulation = None
    if config.include_counterfactual:
        no_modulation = [
            simulate_trajectory(config, params, seed, disable_modulation=True)
            for seed in trajectory_seeds
        ]
    return Dataset(
        config=replace(config),
        parameters=params,
        trajectories=trajectories,
        no_modulation=no_modulation,
        mechanism_support=mechanism_support(params),
        metadata={
            "trajectory_seeds": trajectory_seeds,
            "orientation": "[target, source]",
            "oracle_levels": ["structural", "full-state conditional", "observed process"],
            "counterfactual": "common-random-numbers modulation knockout",
        },
    )
