import numpy as np

from neuromod_benchmark.dgp import generate_parameters, simulate_dataset, simulate_trajectory
from neuromod_benchmark.schema import DGPConfig


def small_config(**kwargs):
    defaults = dict(
        n_neurons=5,
        n_modulators=2,
        n_trajectories=3,
        n_steps=80,
        burn_in=10,
        seed=7,
    )
    defaults.update(kwargs)
    return DGPConfig(**defaults)


def test_shapes_and_finite_oracles():
    dataset = simulate_dataset(small_config())
    trajectory = dataset.trajectories[0]
    assert trajectory.latent.shape == (80, 5)
    assert trajectory.modulator.shape == (80, 2)
    assert np.isfinite(trajectory.latent).all()
    assert np.isfinite(trajectory.conditional_variance).all()
    assert np.all(trajectory.conditional_variance > 0)
    assert trajectory.average_mean_jacobian.shape == (5, 5)


def test_common_random_number_null_intervention_is_exact_for_null_dgp():
    config = small_config(mechanism="null")
    params = generate_parameters(config)
    factual = simulate_trajectory(config, params, seed=111)
    knockout = simulate_trajectory(config, params, seed=111, disable_modulation=True)
    np.testing.assert_allclose(factual.latent, knockout.latent)
    np.testing.assert_allclose(factual.calcium, knockout.calcium)


def test_non_null_knockout_changes_trajectory():
    dataset = simulate_dataset(small_config(mechanism="mixed"))
    assert dataset.no_modulation is not None
    difference = np.mean(
        np.abs(dataset.trajectories[0].latent - dataset.no_modulation[0].latent)
    )
    assert difference > 1e-4


def test_orientation_and_support_are_explicit():
    dataset = simulate_dataset(small_config(mechanism="innovation_variance"))
    assert dataset.metadata["orientation"] == "[target, source]"
    truth = dataset.mechanism_support["conditional_logvariance"]
    np.testing.assert_array_equal(truth, np.abs(dataset.parameters.direct_logvariance) > 0)


def test_legacy_pointwise_oracle_is_indexed_by_retained_target_time():
    config = small_config(store_pointwise_jacobians=True, mechanism="null")
    params = generate_parameters(config)
    trajectory = simulate_trajectory(config, params, seed=113)
    assert trajectory.pointwise_mean_jacobian.shape[0] == config.n_steps
    state = trajectory.latent[0]
    activity = np.tanh(state)
    expected = (
        config.state_decay * np.eye(config.n_neurons)
        + params.baseline_connectivity
        * (1.0 - activity**2)[None, :]
    )
    np.testing.assert_allclose(
        trajectory.pointwise_mean_jacobian[1], expected, rtol=0, atol=1e-14
    )
