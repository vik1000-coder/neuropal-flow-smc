import numpy as np

from neuromod_benchmark.intervention_response import (
    InterventionSpec,
    ResponseKernel,
    aggregate_response_kernels,
    response_kernel,
    response_kernel_metrics,
    simulate_common_history_pair,
)
from neuromod_benchmark.mechanistic import (
    MechanisticConfig,
    generate_mechanistic_parameters,
)


def _config(*, mechanism="additive_mean", seed=71):
    return MechanisticConfig(
        n_neurons=4,
        n_modulators=2,
        n_steps=36,
        burn_in=44,
        mechanism=mechanism,
        seed=seed,
        ligand_pulse_rate_hz=0.1,
        ligand_pulse_amplitude=1.25,
    )


def test_exact_oracle_kernel_has_zero_error_on_every_defined_summary():
    config = _config()
    params = generate_mechanistic_parameters(config)
    modulator = int(np.argwhere(np.abs(params.additive_effect) > 0)[0, 1])
    pairs = [
        simulate_common_history_pair(
            config,
            InterventionSpec.ligand_pulse(
                modulator, amplitude=1.75, duration_steps=3
            ),
            params=params,
            simulation_seed=100 + seed,
            horizon=30,
        )
        for seed in range(4)
    ]
    truth = aggregate_response_kernels(pairs, outcome="conditional_mean")
    oracle_estimate = ResponseKernel(
        lags=truth.lags.copy(),
        values=truth.values.copy(),
        outcome=truth.outcome,
        dt_seconds=truth.dt_seconds,
        n_pairs=truth.n_pairs,
        standard_error=truth.standard_error.copy(),
    )
    metrics = response_kernel_metrics(oracle_estimate, truth)

    assert metrics["response_kernel.truth_rms"] > 0
    for key in (
        "response_kernel.rmse",
        "response_kernel.normalized_rmse",
        "response_kernel.peak_magnitude_mae",
        "response_kernel.peak_magnitude_normalized_rmse",
        "response_kernel.time_to_peak_mae_steps",
        "response_kernel.windowed_cumulative_response_rmse",
        "response_kernel.windowed_cumulative_response_normalized_rmse",
    ):
        assert metrics[key] == 0.0
    assert metrics["response_kernel.sign_accuracy"] == 1.0
    if metrics["response_kernel.decay_estimable_fraction"] > 0:
        assert metrics["response_kernel.decay_time_mae_steps"] == 0.0


def test_null_intervention_has_exactly_zero_response_and_zero_null_leakage():
    config = _config(mechanism="mixed")
    pair = simulate_common_history_pair(
        config,
        InterventionSpec.null(),
        simulation_seed=991,
        horizon=25,
    )

    assert pair.common_history_exact
    np.testing.assert_array_equal(
        pair.baseline.neural_state, pair.intervention.neural_state
    )
    np.testing.assert_array_equal(
        pair.baseline.concentration_state, pair.intervention.concentration_state
    )
    kernel = response_kernel(pair, outcome="conditional_mean")
    np.testing.assert_array_equal(kernel.values, 0.0)
    metrics = response_kernel_metrics(kernel, kernel)
    assert np.isnan(metrics["response_kernel.normalized_rmse"])
    assert metrics["response_kernel.null_entry_fraction"] == 1.0
    assert metrics["response_kernel.null_leakage_rms"] == 0.0


def test_receptor_specific_knockout_branches_only_after_common_history():
    config = _config(seed=83)
    params = generate_mechanistic_parameters(config)
    active = np.argwhere(np.abs(params.additive_effect) > 0)
    assert len(active)
    target, modulator = (int(value) for value in active[0])
    onset = 30
    pair = simulate_common_history_pair(
        config,
        InterventionSpec.receptor_knockout(modulator, target),
        params=params,
        simulation_seed=321,
        onset_step=onset,
        horizon=35,
    )

    assert pair.common_history_exact
    np.testing.assert_array_equal(
        pair.baseline.neural_state[: onset + 1],
        pair.intervention.neural_state[: onset + 1],
    )
    np.testing.assert_array_equal(
        pair.baseline.occupancy[:onset], pair.intervention.occupancy[:onset]
    )
    assert pair.intervention.occupancy[onset, target, modulator] == 0.0
    unaffected = np.ones_like(pair.intervention.occupancy[onset], dtype=bool)
    unaffected[target, modulator] = False
    np.testing.assert_allclose(
        pair.intervention.occupancy[onset][unaffected],
        pair.baseline.occupancy[onset][unaffected],
    )
    assert np.max(np.abs(response_kernel(pair).values)) > 0


def test_release_source_silencing_uses_selected_release_edge_and_shared_noise():
    config = _config(seed=89)
    params = generate_mechanistic_parameters(config)
    # Select a modulator that has both a release source and an active additive target.
    candidates = [
        modulator
        for modulator in range(config.n_modulators)
        if np.any(params.release_weights[modulator] > 0)
        and np.any(np.abs(params.additive_effect[:, modulator]) > 0)
    ]
    assert candidates
    modulator = candidates[0]
    source = int(np.flatnonzero(params.release_weights[modulator] > 0)[0])
    pair = simulate_common_history_pair(
        config,
        InterventionSpec.release_source_silencing(modulator, source),
        params=params,
        simulation_seed=654,
        onset_step=35,
        horizon=35,
    )

    assert pair.common_history_exact
    assert pair.exogenous_fingerprint
    # Silencing changes release at onset without directly clamping neural state.
    assert not np.isclose(
        pair.baseline.release[35, modulator],
        pair.intervention.release[35, modulator],
    )
    concentration_kernel = response_kernel(pair, outcome="concentration")
    assert np.max(np.abs(concentration_kernel.values[:, modulator])) > 0


def test_wrong_kernel_is_penalized_in_field_peak_integral_and_sign():
    config = _config(seed=97)
    pair = simulate_common_history_pair(
        config,
        InterventionSpec.ligand_pulse(1, amplitude=2.0, duration_steps=2),
        simulation_seed=777,
        horizon=32,
    )
    truth = response_kernel(pair)
    wrong_values = -0.4 * np.roll(truth.values, shift=2, axis=0)
    wrong_values[:2] = 0.0
    wrong = ResponseKernel(
        lags=truth.lags.copy(),
        values=wrong_values,
        outcome=truth.outcome,
        dt_seconds=truth.dt_seconds,
    )
    metrics = response_kernel_metrics(wrong, truth)

    assert metrics["response_kernel.normalized_rmse"] > 1.0
    assert metrics["response_kernel.peak_magnitude_mae"] > 0.0
    assert metrics["response_kernel.windowed_cumulative_response_rmse"] > 0.0
    assert metrics["response_kernel.sign_accuracy"] < 0.5


def test_final_lag_peak_is_censored_and_persistent_effect_uses_end_window_gain():
    lags = np.arange(1, 7)
    truth = ResponseKernel(
        lags=lags,
        values=np.arange(1.0, 7.0)[:, None],
        outcome="conditional_mean",
        dt_seconds=0.25,
    )
    metrics = response_kernel_metrics(
        truth, truth, persistent_operation=True, end_window_fraction=0.3
    )

    assert metrics["response_kernel.truth_peak_at_final_lag_fraction"] == 1.0
    assert metrics["response_kernel.time_to_peak_estimable_fraction"] == 0.0
    assert np.isnan(metrics["response_kernel.time_to_peak_mae_steps"])
    assert metrics["response_kernel.decay_estimable_fraction"] == 0.0
    assert np.isnan(metrics["response_kernel.decay_time_mae_steps"])
    assert metrics["response_kernel.end_window_gain_applicable"] == 1.0
    assert metrics["response_kernel.end_window_gain_rmse"] == 0.0
    assert "response_kernel.integrated_response_rmse" not in metrics
