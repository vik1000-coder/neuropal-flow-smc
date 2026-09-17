"""Scientific invariants for the self-contained mechanistic reference simulator."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtr

from neuromod_benchmark.mechanistic import (
    MECHANISMS,
    MechanisticConfig,
    MechanisticState,
    NamedRNGStreams,
    draw_exogenous_noise,
    generate_mechanistic_parameters,
    one_step_moments,
    simulate_mechanistic,
    simulate_receptor_knockout_pair,
    stability_certificate,
    with_mechanism,
)


def reference_config(**kwargs) -> MechanisticConfig:
    values = dict(
        n_neurons=5,
        n_modulators=2,
        n_steps=80,
        burn_in=20,
        mechanism="null",
        seed=17,
    )
    values.update(kwargs)
    return MechanisticConfig(**values)


def reference_state(config: MechanisticConfig) -> MechanisticState:
    return MechanisticState(
        neural=np.linspace(-0.35, 0.45, config.n_neurons),
        concentration=np.linspace(0.18, 0.52, config.n_modulators),
        calcium=np.linspace(-0.10, 0.16, config.n_neurons),
    )


def test_named_rng_streams_are_stable_and_separated() -> None:
    streams = NamedRNGStreams(123456789)
    a1 = streams.generator("neural_innovation").standard_normal(64)
    a2 = streams.generator("neural_innovation").standard_normal(64)
    b = streams.generator("fluorescence_noise").standard_normal(64)
    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(a1, b)
    # The result is independent of request order because streams are stateless/named.
    reverse = NamedRNGStreams(123456789)
    _ = reverse.generator("fluorescence_noise").standard_normal(64)
    np.testing.assert_array_equal(
        a1, reverse.generator("neural_innovation").standard_normal(64)
    )


def test_structural_parameters_respect_support_and_stability_constraints() -> None:
    config = reference_config(mechanism="mixed")
    params = generate_mechanistic_parameters(config)
    params.validate(config)

    row_l1 = np.sum(np.abs(params.baseline_connectivity), axis=1)
    assert np.max(row_l1) <= config.recurrent_row_l1 + 1e-12
    assert np.all(params.release_weights >= 0)
    assert np.all(params.release_weights.sum(axis=1) > 0)
    assert np.all((params.receptor_expression >= 0) & (params.receptor_expression <= 1))

    syn_support = np.abs(params.synaptic_effect) > 0
    assert np.all(~syn_support | (params.baseline_connectivity[:, :, None] != 0))
    assert np.all(
        ~syn_support | (params.receptor_expression[:, None, :] > 0)
    )

    certificate = stability_certificate(config, params)
    assert certificate.baseline_max_row_l1 <= config.recurrent_row_l1 + 1e-12
    assert np.isfinite(certificate.conditional_mean_abs_bound)
    assert 0 < certificate.variance_lower_bound <= certificate.variance_upper_bound


def _finite_difference(
    config: MechanisticConfig,
    params,
    state: MechanisticState,
    field: str,
    epsilon: float = 1e-6,
) -> np.ndarray:
    baseline = np.asarray(getattr(one_step_moments(config, params, state), field))
    out = np.empty(baseline.shape + (config.n_neurons,))
    for source in range(config.n_neurons):
        plus_x = state.neural.copy()
        minus_x = state.neural.copy()
        plus_x[source] += epsilon
        minus_x[source] -= epsilon
        plus = one_step_moments(
            config,
            params,
            MechanisticState(plus_x, state.concentration, state.calcium),
        )
        minus = one_step_moments(
            config,
            params,
            MechanisticState(minus_x, state.concentration, state.calcium),
        )
        numerator = np.asarray(getattr(plus, field)) - np.asarray(getattr(minus, field))
        out[..., source] = numerator / (2.0 * epsilon)
    return out


def test_release_concentration_and_hill_occupancy_jacobians_are_exact() -> None:
    config = reference_config()
    params = generate_mechanistic_parameters(config)
    state = reference_state(config)
    oracle = one_step_moments(config, params, state)

    assert np.all(oracle.release > 0)
    assert np.all(oracle.next_concentration > 0)
    assert np.all(oracle.raw_occupancy >= 0)
    assert np.all(oracle.raw_occupancy <= params.receptor_expression + 1e-14)

    np.testing.assert_allclose(
        oracle.release_jacobian,
        _finite_difference(config, params, state, "release"),
        rtol=2e-6,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        oracle.concentration_jacobian,
        _finite_difference(config, params, state, "next_concentration"),
        rtol=2e-6,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        oracle.occupancy_jacobian,
        _finite_difference(config, params, state, "occupancy"),
        rtol=3e-6,
        atol=3e-9,
    )


@pytest.mark.parametrize("mechanism", sorted(MECHANISMS))
def test_exact_one_step_mean_variance_and_tail_jacobians(mechanism: str) -> None:
    config = reference_config(mechanism=mechanism, n_neurons=4)
    params = generate_mechanistic_parameters(config)
    state = reference_state(config)
    oracle = one_step_moments(config, params, state)

    np.testing.assert_allclose(
        oracle.mean_jacobian,
        _finite_difference(config, params, state, "conditional_mean"),
        rtol=3e-5,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        oracle.variance_jacobian,
        _finite_difference(config, params, state, "conditional_variance"),
        rtol=3e-5,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        oracle.upper_tail_jacobian,
        _finite_difference(config, params, state, "upper_tail_probability"),
        rtol=5e-5,
        atol=5e-8,
    )


@pytest.mark.parametrize(
    ("mechanism", "changes_mean", "changes_variance"),
    [
        ("additive_mean", True, False),
        ("synaptic_gain", True, False),
        ("intrinsic_excitability", True, False),
        ("innovation_variance", False, True),
    ],
)
def test_isolated_mechanisms_change_only_their_one_step_entry_channel(
    mechanism: str, changes_mean: bool, changes_variance: bool
) -> None:
    null_config = reference_config(mechanism="null")
    params = generate_mechanistic_parameters(null_config)
    state = reference_state(null_config)
    null = one_step_moments(null_config, params, state)
    active = one_step_moments(with_mechanism(null_config, mechanism), params, state)

    mean_delta = float(np.max(np.abs(active.conditional_mean - null.conditional_mean)))
    variance_delta = float(
        np.max(np.abs(active.conditional_variance - null.conditional_variance))
    )
    assert (mean_delta > 1e-7) is changes_mean
    assert (variance_delta > 1e-9) is changes_variance

    if not changes_mean:
        np.testing.assert_allclose(active.conditional_mean, null.conditional_mean, atol=1e-14)
        np.testing.assert_allclose(active.mean_jacobian, null.mean_jacobian, atol=1e-14)
    if not changes_variance:
        np.testing.assert_allclose(
            active.conditional_variance, null.conditional_variance, atol=1e-14
        )
        np.testing.assert_allclose(active.logvariance_jacobian, 0.0, atol=1e-14)
        np.testing.assert_allclose(
            active.centered_fourth_moment, null.centered_fourth_moment, atol=1e-14
        )
    assert np.all(active.mixture_probability == 0.0)


def test_matched_tail_changes_tails_but_exactly_matches_mean_and_variance() -> None:
    null_config = reference_config(mechanism="null")
    tail_config = with_mechanism(null_config, "matched_tail")
    params = generate_mechanistic_parameters(null_config)
    state = reference_state(null_config)
    null = one_step_moments(null_config, params, state)
    tail = one_step_moments(tail_config, params, state)

    np.testing.assert_array_equal(tail.conditional_mean, null.conditional_mean)
    np.testing.assert_array_equal(tail.conditional_variance, null.conditional_variance)
    np.testing.assert_array_equal(tail.logvariance_jacobian, null.logvariance_jacobian)
    assert np.any(tail.mixture_probability > 0)
    assert np.max(np.abs(tail.centered_fourth_moment - null.centered_fourth_moment)) > 1e-5
    assert np.max(np.abs(tail.upper_tail_probability - null.upper_tail_probability)) > 1e-5
    assert np.max(np.abs(tail.shape_tail_probability - null.shape_tail_probability)) > 1e-5
    np.testing.assert_allclose(
        null.shape_tail_probability,
        2.0 * ndtr(-null_config.shape_tail_z),
        atol=1e-14,
    )
    np.testing.assert_allclose(null.shape_tail_jacobian, 0.0, atol=1e-14)

    # Direct Monte Carlo check for the most tail-active target.
    target = int(np.argmax(tail.mixture_probability))
    p = float(tail.mixture_probability[target])
    low, high = tail_config.tail_low_scale, tail_config.tail_high_scale
    normalizer2 = (1.0 - p) * low**2 + p * high**2
    rng = np.random.default_rng(991)
    count = 250_000
    scale = np.where(rng.random(count) < p, high, low) / np.sqrt(normalizer2)
    samples = (
        tail.conditional_mean[target]
        + np.sqrt(tail.conditional_variance[target]) * scale * rng.standard_normal(count)
    )
    assert abs(float(np.mean(samples)) - tail.conditional_mean[target]) < 0.003
    variance_error = abs(float(np.var(samples)) - tail.conditional_variance[target])
    assert variance_error / tail.conditional_variance[target] < 0.025
    standardized = (
        samples - tail.conditional_mean[target]
    ) / np.sqrt(tail.conditional_variance[target])
    empirical_shape_tail = np.mean(np.abs(standardized) > tail_config.shape_tail_z)
    assert abs(empirical_shape_tail - tail.shape_tail_probability[target]) < .003

    source = 0
    epsilon = 1e-5
    plus_neural = state.neural.copy()
    minus_neural = state.neural.copy()
    plus_neural[source] += epsilon
    minus_neural[source] -= epsilon
    plus = one_step_moments(
        tail_config,
        params,
        replace(state, neural=plus_neural),
    )
    minus = one_step_moments(
        tail_config,
        params,
        replace(state, neural=minus_neural),
    )
    finite_difference = (
        plus.shape_tail_probability - minus.shape_tail_probability
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        tail.shape_tail_jacobian[:, source], finite_difference, rtol=2e-4, atol=2e-7
    )


def test_receptor_knockout_preserves_upstream_release_and_removes_occupancy() -> None:
    config = reference_config(mechanism="additive_mean")
    params = generate_mechanistic_parameters(config)
    state = reference_state(config)
    knockout = params.receptor_expression > 0
    factual = one_step_moments(config, params, state)
    deleted = one_step_moments(config, params, state, knockout_mask=knockout)

    np.testing.assert_array_equal(factual.release, deleted.release)
    np.testing.assert_array_equal(factual.next_concentration, deleted.next_concentration)
    np.testing.assert_array_equal(factual.raw_occupancy, deleted.raw_occupancy)
    assert np.all(deleted.occupancy[knockout] == 0)
    assert np.max(np.abs(factual.conditional_mean - deleted.conditional_mean)) > 1e-7


def test_common_random_number_knockout_is_exact_under_null_and_effective_when_active() -> None:
    null_config = reference_config(mechanism="null", n_steps=120, burn_in=30)
    null_pair = simulate_receptor_knockout_pair(null_config, simulation_seed=404)
    assert (
        null_pair.factual.exogenous_fingerprint
        == null_pair.receptor_knockout.exogenous_fingerprint
    )
    np.testing.assert_array_equal(
        null_pair.factual.latent_neural, null_pair.receptor_knockout.latent_neural
    )
    np.testing.assert_array_equal(
        null_pair.factual.fluorescence, null_pair.receptor_knockout.fluorescence
    )

    active_config = with_mechanism(null_config, "additive_mean")
    active_pair = simulate_receptor_knockout_pair(active_config, simulation_seed=404)
    assert (
        active_pair.factual.exogenous_fingerprint
        == active_pair.receptor_knockout.exogenous_fingerprint
    )
    assert np.mean(
        np.abs(
            active_pair.factual.latent_neural
            - active_pair.receptor_knockout.latent_neural
        )
    ) > 1e-4


def test_simulation_reproducibility_and_observation_stream_separation() -> None:
    config = reference_config(mechanism="mixed", n_steps=100, burn_in=25)
    params = generate_mechanistic_parameters(config)
    first = simulate_mechanistic(config, params, simulation_seed=808)
    second = simulate_mechanistic(config, params, simulation_seed=808)
    np.testing.assert_array_equal(first.latent_neural, second.latent_neural)
    np.testing.assert_array_equal(first.fluorescence, second.fluorescence)

    noisier = replace(config, measurement_noise=0.25)
    altered_observation = simulate_mechanistic(noisier, params, simulation_seed=808)
    # Named observation noise never feeds back into latent or clean calcium dynamics.
    np.testing.assert_array_equal(first.latent_neural, altered_observation.latent_neural)
    np.testing.assert_array_equal(first.clean_calcium, altered_observation.clean_calcium)
    assert not np.array_equal(first.fluorescence, altered_observation.fluorescence)


@pytest.mark.parametrize("mechanism", ["null", "matched_tail"])
def test_exact_calcium_and_fluorescence_one_step_moments(mechanism: str) -> None:
    config = reference_config(mechanism=mechanism)
    params = generate_mechanistic_parameters(config)
    state = reference_state(config)
    oracle = one_step_moments(config, params, state)
    rng = np.random.default_rng(1201)
    count = 180_000
    target = int(np.argmax(oracle.mixture_probability)) if mechanism == "matched_tail" else 0

    if mechanism == "matched_tail":
        p = float(oracle.mixture_probability[target])
        normalizer2 = (
            (1.0 - p) * config.tail_low_scale**2
            + p * config.tail_high_scale**2
        )
        component = np.where(
            rng.random(count) < p,
            config.tail_high_scale,
            config.tail_low_scale,
        ) / np.sqrt(normalizer2)
    else:
        component = np.ones(count)
    x_next = (
        oracle.conditional_mean[target]
        + np.sqrt(oracle.conditional_variance[target])
        * component
        * rng.standard_normal(count)
    )
    calcium = (
        config.calcium_rho * state.calcium[target]
        + (1.0 - config.calcium_rho) * x_next
    )
    fluorescence = calcium + config.measurement_noise * rng.standard_normal(count)
    assert abs(float(np.mean(calcium)) - oracle.calcium_mean[target]) < 0.002
    calcium_variance_error = abs(float(np.var(calcium)) - oracle.calcium_variance[target])
    assert calcium_variance_error / oracle.calcium_variance[target] < 0.03
    assert abs(float(np.mean(fluorescence)) - oracle.fluorescence_mean[target]) < 0.002
    fluorescence_variance_error = abs(
        float(np.var(fluorescence)) - oracle.fluorescence_variance[target]
    )
    assert fluorescence_variance_error / oracle.fluorescence_variance[target] < 0.03


def test_long_mixed_rollout_is_finite_positive_and_within_global_mean_bound() -> None:
    config = reference_config(
        mechanism="mixed",
        n_neurons=8,
        n_modulators=3,
        n_steps=1_500,
        burn_in=300,
        seed=29,
    )
    params = generate_mechanistic_parameters(config)
    trajectory = simulate_mechanistic(config, params, simulation_seed=919)
    assert trajectory.latent_neural.shape == (1_500, 8)
    assert np.isfinite(trajectory.latent_neural).all()
    assert np.isfinite(trajectory.fluorescence).all()
    assert np.all(trajectory.concentration > 0)
    assert np.all(trajectory.occupancy >= 0)
    assert np.all(
        trajectory.raw_occupancy
        <= params.receptor_expression[None, :, :] + 1e-14
    )
    assert np.max(np.abs(trajectory.conditional_mean)) <= (
        trajectory.stability.conditional_mean_abs_bound + 1e-12
    )
    assert np.min(trajectory.conditional_variance) >= (
        trajectory.stability.variance_lower_bound - 1e-12
    )
    assert np.max(trajectory.conditional_variance) <= (
        trajectory.stability.variance_upper_bound + 1e-12
    )


def test_exogenous_noise_fingerprint_and_shapes_are_reproducible() -> None:
    config = reference_config(n_steps=25, burn_in=5)
    one = draw_exogenous_noise(config, seed=55)
    two = draw_exogenous_noise(config, seed=55)
    assert one.fingerprint == two.fingerprint
    np.testing.assert_array_equal(one.neural_standard_normal, two.neural_standard_normal)
    assert one.neural_standard_normal.shape == (30, 5)
    assert one.mixture_uniform.shape == (30, 5)


def test_public_mechanism_helper_rejects_unknown_mechanisms() -> None:
    config = reference_config()
    with pytest.raises(ValueError, match="unknown mechanism"):
        with_mechanism(config, "not_a_mechanism")
