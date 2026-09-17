"""Adapter from the rigorous mechanistic simulator to benchmark data contracts."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .mechanistic import (
    MechanisticConfig,
    MechanisticParameters,
    draw_exogenous_noise,
    generate_mechanistic_parameters,
    simulate_mechanistic,
)
from .schema import Dataset, Trajectory


def _trajectory(trace) -> Trajectory:
    n_steps, n = trace.latent_neural.shape
    mask = np.ones((n_steps, n), dtype=bool)
    # ``trace.ligand_drive[t]`` generated ``latent[t]`` from the preceding state.
    # Supervised row t predicts latent[t+1], so expose the controlled input applied
    # over that transition. The final value is never used by a positive-horizon row.
    transition_stimulus = np.concatenate(
        [trace.ligand_drive[1:], trace.ligand_drive[-1:]], axis=0
    )
    return Trajectory(
        latent=trace.latent_neural,
        calcium=trace.fluorescence,
        modulator=trace.concentration,
        stimulus=transition_stimulus,
        conditional_mean=trace.conditional_mean,
        conditional_variance=trace.conditional_variance,
        burst_probability=trace.mixture_probability,
        hidden_driver=np.zeros(n_steps),
        observed_mask=mask,
        average_mean_jacobian=np.mean(trace.mean_jacobian[1:], axis=0),
        average_logvariance_jacobian=np.mean(trace.logvariance_jacobian[1:], axis=0),
        average_tail_jacobian=np.mean(trace.upper_tail_jacobian[1:], axis=0),
        average_shape_tail_jacobian=np.mean(
            trace.shape_tail_jacobian[1:], axis=0
        ),
        pointwise_mean_jacobian=trace.mean_jacobian,
        pointwise_logvariance_jacobian=trace.logvariance_jacobian,
        pointwise_tail_jacobian=trace.upper_tail_jacobian,
        pointwise_shape_tail_jacobian=trace.shape_tail_jacobian,
        upper_tail_probability=trace.upper_tail_probability,
        shape_tail_probability=trace.shape_tail_probability,
        conditional_covariance=trace.conditional_covariance,
        conditional_correlation=trace.conditional_correlation,
        average_covariance_jacobian=(
            None
            if trace.covariance_jacobian is None
            else np.mean(trace.covariance_jacobian[1:], axis=0)
        ),
        average_correlation_jacobian=(
            None
            if trace.correlation_jacobian is None
            else np.mean(trace.correlation_jacobian[1:], axis=0)
        ),
        pointwise_covariance_jacobian=trace.covariance_jacobian,
        pointwise_correlation_jacobian=trace.correlation_jacobian,
        covariance_concentration_susceptibility=(
            trace.covariance_concentration_susceptibility
        ),
        correlation_concentration_susceptibility=(
            trace.correlation_concentration_susceptibility
        ),
    )


def _support(
    params: MechanisticParameters,
    trajectories: list[Trajectory],
    mechanism: str,
):
    """Return support for the *active law*, not merely pre-drawn potential tensors.

    All potential effect tensors are generated under a common seed so scenarios can
    share the same baseline system.  Inactive tensors are structural zeros in the
    realized DGP and must therefore never appear in a ground-truth support map.
    """

    eps = 1e-10
    active = {mechanism} if mechanism not in {"null", "mixed"} else (
        set() if mechanism == "null" else {
            "additive_mean",
            "synaptic_gain",
            "intrinsic_excitability",
            "innovation_variance",
            "matched_tail",
        }
    )
    mean = np.mean([tr.average_mean_jacobian for tr in trajectories], axis=0)
    logvariance = np.mean([tr.average_logvariance_jacobian for tr in trajectories], axis=0)
    tail = np.mean([tr.average_tail_jacobian for tr in trajectories], axis=0)
    shape_tail = np.mean(
        [tr.average_shape_tail_jacobian for tr in trajectories], axis=0
    )
    result = {
        "wired_connectivity": np.abs(params.baseline_connectivity) > eps,
        "complete_state_mean_jacobian": np.abs(mean) > eps,
        "complete_state_logvariance_jacobian": np.abs(logvariance) > eps,
        "complete_state_tail_jacobian": np.abs(tail) > eps,
        "complete_state_shape_tail_jacobian": np.abs(shape_tail) > eps,
        "release": np.abs(params.release_weights) > eps,
        "receptor_expression": np.abs(params.receptor_expression) > eps,
        "neuromod_additive_mean": (
            np.abs(params.additive_effect) > eps
            if "additive_mean" in active
            else np.zeros_like(params.additive_effect, dtype=bool)
        ),
        "neuromod_synaptic_gain": (
            np.any(np.abs(params.synaptic_effect) > eps, axis=2)
            if "synaptic_gain" in active
            else np.zeros(params.synaptic_effect.shape[:2], dtype=bool)
        ),
        "neuromod_intrinsic": (
            np.abs(params.intrinsic_log_slope_effect) > eps
            if "intrinsic_excitability" in active
            else np.zeros_like(params.intrinsic_log_slope_effect, dtype=bool)
        ),
        "neuromod_variance": (
            np.abs(params.logvariance_effect) > eps
            if "innovation_variance" in active
            else np.zeros_like(params.logvariance_effect, dtype=bool)
        ),
        "neuromod_tail": (
            np.abs(params.tail_logit_effect) > eps
            if "matched_tail" in active
            else np.zeros_like(params.tail_logit_effect, dtype=bool)
        ),
        "neuromod_correlation_loading_support": (
            np.abs(params.correlation_loading_effect) > eps
            if "correlation_routing" in active
            else np.zeros_like(params.correlation_loading_effect, dtype=bool)
        ),
        "neuromod_correlation_loading_weight": (
            params.correlation_loading_effect.copy()
            if "correlation_routing" in active
            else np.zeros_like(params.correlation_loading_effect)
        ),
    }
    covariance_susceptibility = [
        tr.covariance_concentration_susceptibility
        for tr in trajectories
        if tr.covariance_concentration_susceptibility is not None
    ]
    correlation_susceptibility = [
        tr.correlation_concentration_susceptibility
        for tr in trajectories
        if tr.correlation_concentration_susceptibility is not None
    ]
    if covariance_susceptibility:
        result["continuous_covariance_susceptibility"] = np.mean(
            np.concatenate(covariance_susceptibility, axis=0), axis=0
        )
    if correlation_susceptibility:
        result["continuous_correlation_susceptibility"] = np.mean(
            np.concatenate(correlation_susceptibility, axis=0), axis=0
        )
    return result


def simulate_mechanistic_dataset(
    config: MechanisticConfig,
    *,
    n_trajectories: int = 6,
) -> Dataset:
    """Generate worm/episode replicates sharing parameters and paired receptor knockouts."""

    config.validate()
    if n_trajectories < 3:
        raise ValueError("at least three trajectories are required for grouped evaluation")
    params = generate_mechanistic_parameters(config)
    factual: list[Trajectory] = []
    knockout: list[Trajectory] = []
    zero_dose: list[Trajectory] = []
    double_dose: list[Trajectory] = []
    fingerprints: list[str] = []
    for episode in range(n_trajectories):
        simulation_seed = config.seed * 100_003 + 20_011 + episode
        exogenous = draw_exogenous_noise(config, simulation_seed)
        factual_trace = simulate_mechanistic(config, params, exogenous=exogenous)
        knockout_trace = simulate_mechanistic(
            config,
            params,
            knockout_mask=params.receptor_expression > 0,
            exogenous=exogenous,
        )
        zero_trace = simulate_mechanistic(
            replace(config, ligand_pulse_amplitude=0.0), params, exogenous=exogenous
        )
        double_trace = simulate_mechanistic(
            replace(config, ligand_pulse_amplitude=2.0 * config.ligand_pulse_amplitude),
            params,
            exogenous=exogenous,
        )
        factual.append(_trajectory(factual_trace))
        knockout.append(_trajectory(knockout_trace))
        zero_dose.append(_trajectory(zero_trace))
        double_dose.append(_trajectory(double_trace))
        fingerprints.append(factual_trace.exogenous_fingerprint)
        if len(
            {
                factual_trace.exogenous_fingerprint,
                knockout_trace.exogenous_fingerprint,
                zero_trace.exogenous_fingerprint,
                double_trace.exogenous_fingerprint,
            }
        ) != 1:
            raise AssertionError("counterfactual pair did not share exogenous noise")
    return Dataset(
        config=config,
        parameters=params,
        trajectories=factual,
        no_modulation=knockout,
        mechanism_support=_support(params, factual, config.mechanism),
        metadata={
            "family": "release_concentration_hill_reference",
            "trajectory_count": n_trajectories,
            "trajectory_seeds": [config.seed * 100_003 + 20_011 + i for i in range(n_trajectories)],
            "exogenous_fingerprints": fingerprints,
            "orientation": "[target, source]",
            "oracle_levels": [
                "structural_parameter",
                "complete_state_conditional",
                "observed_history_predictive",
                "interventional_receptor_knockout",
                "measurement",
            ],
            "tail_moments": "conditional mean and variance matched exactly",
            "correlation_routing": (
                "bounded one-factor loadings preserve every marginal mean and variance"
            ),
            "continuous_oracles": [
                "conditional_covariance",
                "conditional_correlation",
                "physical_covariance_susceptibility",
                "physical_correlation_susceptibility",
            ],
            "counterfactual": "common-random-numbers receptor knockout",
        },
        interventions={
            "ligand_zero_dose": zero_dose,
            "ligand_double_dose": double_dose,
        },
        intervention_metadata={
            "ligand_zero_dose": {
                "operation": "set randomized ligand-pulse amplitude to zero",
                "dose_multiplier": 0.0,
                "common_random_numbers": True,
            },
            "ligand_double_dose": {
                "operation": "double randomized ligand-pulse amplitude",
                "dose_multiplier": 2.0,
                "common_random_numbers": True,
            },
        },
    )


def dataset_with_trajectories(dataset: Dataset, trajectories: list[Trajectory]) -> Dataset:
    """Create an evaluation view without mutating the factual dataset."""

    return replace(
        dataset,
        trajectories=trajectories,
        no_modulation=None,
        interventions={},
        intervention_metadata={},
    )
