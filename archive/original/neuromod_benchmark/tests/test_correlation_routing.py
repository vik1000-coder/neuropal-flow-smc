"""Correlation-routing invariants, exact oracles, and continuous evaluation."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from neuromod_benchmark.evaluation import (
    oracle_functional_metrics,
    physical_modulator_susceptibility_metrics,
)
from neuromod_benchmark.features import build_supervised
from neuromod_benchmark.mechanistic import (
    MechanisticConfig,
    MechanisticState,
    draw_exogenous_noise,
    generate_mechanistic_parameters,
    one_step_moments,
    simulate_mechanistic,
    simulate_receptor_knockout_pair,
    stability_certificate,
    with_mechanism,
)
from neuromod_benchmark.mechanistic_dataset import simulate_mechanistic_dataset
from neuromod_benchmark.schema import Prediction


def correlation_config(**kwargs) -> MechanisticConfig:
    values = dict(
        n_neurons=5,
        n_modulators=2,
        n_steps=80,
        burn_in=30,
        mechanism="correlation_routing",
        effect_strength=1.4,
        correlation_max_loading=0.75,
        ligand_pulse_rate_hz=0.08,
        ligand_pulse_amplitude=2.0,
        seed=707,
    )
    values.update(kwargs)
    return MechanisticConfig(**values)


def reference_state(config: MechanisticConfig) -> MechanisticState:
    return MechanisticState(
        neural=np.linspace(-0.40, 0.50, config.n_neurons),
        concentration=np.linspace(0.20, 0.65, config.n_modulators),
        calcium=np.linspace(-0.10, 0.15, config.n_neurons),
    )


def _finite_difference_neural(config, params, state, field, epsilon=1e-6):
    baseline = np.asarray(getattr(one_step_moments(config, params, state), field))
    derivative = np.empty(baseline.shape + (config.n_neurons,))
    for source in range(config.n_neurons):
        plus = state.neural.copy()
        minus = state.neural.copy()
        plus[source] += epsilon
        minus[source] -= epsilon
        plus_value = getattr(
            one_step_moments(
                config,
                params,
                MechanisticState(plus, state.concentration, state.calcium),
            ),
            field,
        )
        minus_value = getattr(
            one_step_moments(
                config,
                params,
                MechanisticState(minus, state.concentration, state.calcium),
            ),
            field,
        )
        derivative[..., source] = (plus_value - minus_value) / (2.0 * epsilon)
    return derivative


def _finite_difference_concentration(config, params, state, field, epsilon=1e-6):
    baseline = np.asarray(getattr(one_step_moments(config, params, state), field))
    derivative = np.empty(baseline.shape + (config.n_modulators,))
    for modulator in range(config.n_modulators):
        plus = state.concentration.copy()
        minus = state.concentration.copy()
        plus[modulator] += epsilon
        minus[modulator] -= epsilon
        plus_value = getattr(
            one_step_moments(
                config,
                params,
                MechanisticState(state.neural, plus, state.calcium),
            ),
            field,
        )
        minus_value = getattr(
            one_step_moments(
                config,
                params,
                MechanisticState(state.neural, minus, state.calcium),
            ),
            field,
        )
        derivative[..., modulator] = (plus_value - minus_value) / (2.0 * epsilon)
    return derivative


def test_pure_correlation_routing_preserves_every_marginal_mean_and_variance() -> None:
    config = correlation_config()
    params = generate_mechanistic_parameters(config)
    state = reference_state(config)
    routed = one_step_moments(config, params, state)
    null = one_step_moments(with_mechanism(config, "null"), params, state)

    np.testing.assert_array_equal(routed.conditional_mean, null.conditional_mean)
    np.testing.assert_array_equal(routed.conditional_variance, null.conditional_variance)
    np.testing.assert_array_equal(
        routed.centered_fourth_moment, null.centered_fourth_moment
    )
    np.testing.assert_array_equal(
        routed.upper_tail_probability, null.upper_tail_probability
    )
    np.testing.assert_array_equal(
        np.diag(routed.conditional_covariance), routed.conditional_variance
    )
    np.testing.assert_array_equal(np.diag(routed.conditional_correlation), 1.0)
    assert np.max(
        np.abs(
            routed.conditional_correlation
            - np.eye(config.n_neurons)
        )
    ) > 1e-5
    assert np.all(routed.mixture_probability == 0.0)


def test_factor_correlation_is_globally_positive_definite_and_bounded() -> None:
    worst_eigenvalue = np.inf
    for seed in range(30):
        config = correlation_config(seed=seed)
        params = generate_mechanistic_parameters(config)
        certificate = stability_certificate(config, params)
        rng = np.random.default_rng(50_000 + seed)
        for _ in range(12):
            state = MechanisticState(
                neural=rng.normal(size=config.n_neurons),
                concentration=np.exp(rng.uniform(-2.0, 1.0, size=config.n_modulators)),
                calcium=rng.normal(size=config.n_neurons),
            )
            oracle = one_step_moments(config, params, state)
            eigenvalue = float(np.min(np.linalg.eigvalsh(oracle.conditional_correlation)))
            worst_eigenvalue = min(worst_eigenvalue, eigenvalue)
            assert eigenvalue >= certificate.correlation_eigenvalue_lower_bound - 1e-12
            assert np.max(np.abs(oracle.correlation_loading)) < 1.0
            assert np.max(np.abs(oracle.correlation_loading)) <= (
                config.correlation_max_loading + 1e-14
            )
            assert np.min(np.linalg.eigvalsh(oracle.conditional_covariance)) > 0
    assert worst_eigenvalue > 0


def test_covariance_correlation_and_loading_neural_jacobians_are_exact() -> None:
    config = correlation_config()
    params = generate_mechanistic_parameters(config)
    state = reference_state(config)
    oracle = one_step_moments(config, params, state)
    np.testing.assert_allclose(
        oracle.covariance_jacobian,
        _finite_difference_neural(
            config, params, state, "conditional_covariance"
        ),
        rtol=5e-5,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        oracle.correlation_jacobian,
        _finite_difference_neural(
            config, params, state, "conditional_correlation"
        ),
        rtol=5e-5,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        oracle.correlation_loading_jacobian,
        _finite_difference_neural(config, params, state, "correlation_loading"),
        rtol=5e-5,
        atol=5e-9,
    )


def test_physical_concentration_susceptibilities_are_exact() -> None:
    config = correlation_config()
    params = generate_mechanistic_parameters(config)
    state = reference_state(config)
    oracle = one_step_moments(config, params, state)
    np.testing.assert_allclose(
        oracle.covariance_concentration_susceptibility,
        _finite_difference_concentration(
            config, params, state, "conditional_covariance"
        ),
        rtol=5e-5,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        oracle.correlation_concentration_susceptibility,
        _finite_difference_concentration(
            config, params, state, "conditional_correlation"
        ),
        rtol=5e-5,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        oracle.correlation_loading_concentration_susceptibility,
        _finite_difference_concentration(
            config, params, state, "correlation_loading"
        ),
        rtol=5e-5,
        atol=5e-9,
    )


def test_one_factor_monte_carlo_matches_exact_covariance_and_correlation() -> None:
    config = correlation_config(n_neurons=6, n_modulators=3, seed=19)
    params = generate_mechanistic_parameters(config)
    state = reference_state(config)
    oracle = one_step_moments(config, params, state)
    rng = np.random.default_rng(1441)
    count = 300_000
    common = rng.standard_normal((count, 1))
    independent = rng.standard_normal((count, config.n_neurons))
    loading = oracle.correlation_loading
    standardized = (
        common * loading[None, :]
        + independent * np.sqrt(1.0 - loading**2)[None, :]
    )
    innovations = standardized * np.sqrt(oracle.conditional_variance)[None, :]
    empirical_covariance = np.cov(innovations, rowvar=False, bias=True)
    empirical_correlation = np.corrcoef(innovations, rowvar=False)
    np.testing.assert_allclose(
        np.mean(innovations, axis=0), 0.0, atol=1.5e-3
    )
    np.testing.assert_allclose(
        empirical_covariance, oracle.conditional_covariance, rtol=0.04, atol=2.5e-4
    )
    np.testing.assert_allclose(
        empirical_correlation, oracle.conditional_correlation, rtol=0.08, atol=5e-3
    )


def test_named_common_factor_is_reproducible_and_shared_by_knockout_pair() -> None:
    config = correlation_config(n_steps=120, burn_in=40)
    first = draw_exogenous_noise(config, seed=818)
    repeated = draw_exogenous_noise(config, seed=818)
    np.testing.assert_array_equal(
        first.correlation_common_normal, repeated.correlation_common_normal
    )
    assert first.fingerprint == repeated.fingerprint

    pair = simulate_receptor_knockout_pair(config, simulation_seed=818)
    assert (
        pair.factual.exogenous_fingerprint
        == pair.receptor_knockout.exogenous_fingerprint
    )
    assert not np.array_equal(
        pair.factual.latent_neural, pair.receptor_knockout.latent_neural
    )


def test_long_correlation_rollout_respects_marginal_and_stability_oracles() -> None:
    config = correlation_config(
        n_neurons=9,
        n_modulators=3,
        n_steps=1_200,
        burn_in=300,
        seed=52,
    )
    params = generate_mechanistic_parameters(config)
    trace = simulate_mechanistic(config, params, simulation_seed=901)
    assert trace.conditional_covariance is not None
    assert trace.conditional_correlation is not None
    assert np.isfinite(trace.latent_neural).all()
    np.testing.assert_allclose(
        np.diagonal(trace.conditional_covariance, axis1=1, axis2=2),
        trace.conditional_variance,
        atol=1e-14,
    )
    for matrix in trace.conditional_correlation[::31]:
        assert np.min(np.linalg.eigvalsh(matrix)) >= (
            trace.stability.correlation_eigenvalue_lower_bound - 1e-12
        )
    assert np.max(np.abs(trace.conditional_mean)) <= (
        trace.stability.conditional_mean_abs_bound + 1e-12
    )


def test_dataset_schema_exposes_continuous_covariance_oracles() -> None:
    config = correlation_config(n_neurons=4, n_modulators=2, n_steps=55, burn_in=30)
    dataset = simulate_mechanistic_dataset(config, n_trajectories=3)
    supervised = build_supervised(
        dataset, view="complete_state", history_lags=(1,), horizon=1
    )
    assert supervised.oracle_covariance is not None
    assert supervised.oracle_correlation is not None
    assert supervised.oracle_covariance.shape[1:] == (4, 4)
    assert supervised.oracle_correlation.shape[1:] == (4, 4)
    assert "continuous_covariance_susceptibility" in dataset.mechanism_support
    assert "continuous_correlation_susceptibility" in dataset.mechanism_support
    assert np.any(
        np.abs(dataset.mechanism_support["continuous_correlation_susceptibility"])
        > 0
    )

    prediction = Prediction(
        mean=supervised.oracle_mean.copy(),
        variance=supervised.oracle_variance.copy(),
        covariance=supervised.oracle_covariance.copy(),
        correlation=supervised.oracle_correlation.copy(),
    )
    metrics = oracle_functional_metrics(prediction, supervised)
    assert metrics["oracle.covariance_continuous.relative_frobenius"] == 0.0
    assert metrics["oracle.correlation_continuous.relative_frobenius"] == 0.0
    assert metrics["oracle.correlation_offdiagonal_rmse"] == 0.0


def test_physical_evaluator_scores_continuous_covariance_susceptibility() -> None:
    config = correlation_config(n_neurons=4, n_modulators=2, n_steps=45, burn_in=25)
    dataset = simulate_mechanistic_dataset(config, n_trajectories=3)
    supervised = build_supervised(
        dataset, view="complete_state", history_lags=(1,), horizon=1
    )
    max_states = 24
    indices = np.arange(len(supervised.targets))
    if len(indices) > max_states:
        indices = indices[np.linspace(0, len(indices) - 1, max_states, dtype=int)]
    covariance = np.zeros((4, 4, 2))
    correlation = np.zeros((4, 4, 2))
    for row in indices:
        group, time = int(supervised.groups[row]), int(supervised.times[row])
        trajectory = dataset.trajectories[group]
        row_parameters = replace(
            dataset.parameters,
            release_bias=(
                dataset.parameters.release_bias + trajectory.stimulus[time]
            ),
        )
        oracle = one_step_moments(
            config,
            row_parameters,
            MechanisticState(
                trajectory.latent[time],
                trajectory.modulator[time],
                np.zeros(4),
            ),
        )
        covariance += oracle.covariance_concentration_susceptibility
        correlation += oracle.correlation_concentration_susceptibility
    covariance /= len(indices)
    correlation /= len(indices)
    feature_count = len(supervised.feature_names)
    modulator_columns = [
        i
        for i, name in enumerate(supervised.feature_names)
        if name.startswith("modulator")
    ]
    covariance_feature = np.zeros((4, 4, feature_count))
    correlation_feature = np.zeros((4, 4, feature_count))
    covariance_feature[..., modulator_columns] = covariance
    correlation_feature[..., modulator_columns] = correlation

    class ExactChannelEstimator:
        channels_ = {
            "conditional_covariance_derivative_feature": covariance_feature,
            "conditional_correlation_derivative_feature": correlation_feature,
        }

    metrics = physical_modulator_susceptibility_metrics(
        ExactChannelEstimator(), dataset, supervised, max_states=max_states
    )
    assert metrics[
        "neuromodulator.physical_covariance_susceptibility.relative_frobenius"
    ] < 1e-12
    assert metrics[
        "neuromodulator.physical_correlation_susceptibility.relative_frobenius"
    ] < 1e-12
    assert metrics[
        "neuromodulator.physical_correlation_susceptibility.cosine"
    ] > 1 - 1e-12


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=20)
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def test_legacy_mixed_scenario_remains_bitwise_backward_compatible() -> None:
    config = MechanisticConfig(
        n_neurons=5,
        n_modulators=2,
        n_steps=40,
        burn_in=20,
        mechanism="mixed",
        seed=314,
        ligand_pulse_rate_hz=0.08,
        ligand_pulse_amplitude=2.5,
    )
    params = generate_mechanistic_parameters(config)
    trace = simulate_mechanistic(config, params, simulation_seed=2718)
    parameter_hash = _hash_arrays(
        params.baseline_connectivity,
        params.receptor_expression,
        params.additive_effect,
        params.synaptic_effect,
        params.intrinsic_log_slope_effect,
        params.logvariance_effect,
        params.tail_logit_effect,
    )
    trajectory_hash = _hash_arrays(
        trace.latent_neural,
        trace.conditional_mean,
        trace.conditional_variance,
        trace.mixture_probability,
    )
    assert parameter_hash == "997212a91613de174c00e6d7103265baf625bbe9"
    assert trajectory_hash == "e5f23fbcba1af4df1b9b869812d5eab86394dc5c"
    assert trace.exogenous_fingerprint == "47c5dac2fd934dc9bb24b1197f1100c0"
    assert trace.conditional_covariance is None


def test_correlation_parameter_has_at_least_two_receptor_target_loadings() -> None:
    for seed in range(100):
        config = correlation_config(seed=seed)
        params = generate_mechanistic_parameters(config)
        active_rows = np.flatnonzero(
            np.any(np.abs(params.correlation_loading_effect) > 0, axis=1)
        )
        assert len(active_rows) >= 2
        assert np.all(
            (np.abs(params.correlation_loading_effect) > 0)
            <= (params.receptor_expression > 0)
        )
