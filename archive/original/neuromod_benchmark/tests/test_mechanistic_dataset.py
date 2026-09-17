from dataclasses import replace

import numpy as np

from neuromod_benchmark.features import build_supervised
from neuromod_benchmark.mechanistic import (
    MechanisticConfig,
    MechanisticState,
    one_step_moments,
)
from neuromod_benchmark.mechanistic_dataset import simulate_mechanistic_dataset


def test_adapter_exposes_complete_state_oracles_and_paired_knockout():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=4,
            n_modulators=1,
            n_steps=50,
            burn_in=20,
            mechanism="matched_tail",
            seed=8,
        ),
        n_trajectories=3,
    )
    data = build_supervised(dataset, view="complete_state", history_lags=(1,), horizon=1)
    assert data.oracle_mean is not None
    assert data.oracle_variance is not None
    assert data.features.shape[1] == 4 + 1 + 1  # neural state, one ligand input, concentration
    assert dataset.no_modulation is not None
    assert set(dataset.interventions) == {"ligand_zero_dose", "ligand_double_dose"}
    assert dataset.metadata["tail_moments"] == "conditional mean and variance matched exactly"
    assert np.isfinite(data.oracle_variance).all()


def test_ligand_dose_panel_shares_randomness_and_changes_only_dose_schedule():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=80,
            burn_in=100,
            mechanism="mixed",
            ligand_pulse_rate_hz=.2,
            ligand_pulse_amplitude=2.0,
            seed=17,
        ),
        n_trajectories=3,
    )
    factual = dataset.trajectories[0]
    zero = dataset.interventions["ligand_zero_dose"][0]
    double = dataset.interventions["ligand_double_dose"][0]
    assert np.allclose(zero.stimulus, 0.0)
    assert np.all(double.stimulus >= factual.stimulus)
    assert np.allclose(double.stimulus, 2.0 * factual.stimulus)
    assert not np.array_equal(zero.latent, factual.latent)


def test_mechanism_support_exposes_only_effects_active_in_the_realized_law():
    null = simulate_mechanistic_dataset(
        MechanisticConfig(n_neurons=4, n_modulators=1, n_steps=30, burn_in=20,
                          mechanism="null", seed=21),
        n_trajectories=3,
    )
    for name in (
        "neuromod_additive_mean",
        "neuromod_synaptic_gain",
        "neuromod_intrinsic",
        "neuromod_variance",
        "neuromod_tail",
        "neuromod_correlation_loading_support",
    ):
        assert not np.any(null.mechanism_support[name])

    tail = simulate_mechanistic_dataset(
        MechanisticConfig(n_neurons=4, n_modulators=1, n_steps=30, burn_in=20,
                          mechanism="matched_tail", seed=21),
        n_trajectories=3,
    )
    assert np.any(tail.mechanism_support["neuromod_tail"])
    assert not np.any(tail.mechanism_support["neuromod_variance"])


def test_every_one_step_oracle_is_reconstructible_from_declared_row_features():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=4,
            n_modulators=2,
            n_steps=70,
            burn_in=80,
            mechanism="mixed",
            ligand_pulse_rate_hz=.2,
            ligand_pulse_amplitude=2.0,
            seed=29,
        ),
        n_trajectories=3,
    )
    for trajectory in dataset.trajectories:
        for time in range(len(trajectory.latent) - 1):
            row_parameters = replace(
                dataset.parameters,
                release_bias=(
                    dataset.parameters.release_bias + trajectory.stimulus[time]
                ),
            )
            oracle = one_step_moments(
                dataset.config,
                row_parameters,
                MechanisticState(
                    trajectory.latent[time],
                    trajectory.modulator[time],
                    np.zeros(dataset.config.n_neurons),
                ),
            )
            np.testing.assert_allclose(
                oracle.conditional_mean,
                trajectory.conditional_mean[time + 1],
                rtol=0,
                atol=2e-15,
            )
            np.testing.assert_allclose(
                oracle.conditional_variance,
                trajectory.conditional_variance[time + 1],
                rtol=0,
                atol=2e-15,
            )
            np.testing.assert_allclose(
                oracle.next_concentration,
                trajectory.modulator[time + 1],
                rtol=0,
                atol=2e-15,
            )
            np.testing.assert_allclose(
                oracle.shape_tail_probability,
                trajectory.shape_tail_probability[time + 1],
                rtol=0,
                atol=2e-15,
            )
