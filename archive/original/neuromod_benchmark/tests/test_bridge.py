import numpy as np

from neuromod_benchmark.capabilities import validate_channel_contract
from neuromod_benchmark.dgp import simulate_dataset
from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.methods.bridge import ConditionalBrownianBridge
from neuromod_benchmark.evaluation import evaluate_estimator
from neuromod_benchmark.schema import DGPConfig


def test_conditional_bridge_has_exact_path_endpoints_and_finite_cost():
    dataset = simulate_dataset(
        DGPConfig(n_neurons=3, n_modulators=1, n_trajectories=5, n_steps=70, burn_in=5)
    )
    data = build_supervised(dataset, view="latent", history_lags=(1,), horizon=1)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=0).masks(data)
    train, validation, test = [subset(data, mask) for mask in masks]
    model = ConditionalBrownianBridge(reference_diffusion=.3).fit(train, validation)
    validate_channel_contract(model.capabilities, model.channels_)
    assert "linear_transition_coefficient" not in model.channels_
    assert model.metadata()["reference_diffusion_unit"] == (
        "target activity per sqrt(simulation step)"
    )
    paths = model.sample_paths(test, n_paths=4, n_steps=5, seed=3)
    assert paths.shape == (len(test.targets), 4, 6, 3)
    expected_start = np.broadcast_to(test.features[:, None, :3], paths[:, :, 0].shape)
    np.testing.assert_allclose(paths[:, :, 0], expected_start, atol=1e-12)
    assert np.isfinite(model.path_relative_entropy(test)).all()


def test_bridge_interior_path_metrics_are_finite_and_separate_from_endpoint():
    dataset = simulate_dataset(
        DGPConfig(n_neurons=3, n_modulators=1, n_trajectories=5, n_steps=80, burn_in=5)
    )
    data = build_supervised(dataset, view="complete_state", history_lags=(1,), horizon=4)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=1).masks(data)
    train, validation, test = [subset(data, mask) for mask in masks]
    model = ConditionalBrownianBridge(reference_diffusion=.3).fit(train, validation)
    metrics, _ = evaluate_estimator(
        model,
        dataset,
        test,
        validation,
        history_lags=(1,),
        n_samples=8,
    )
    assert metrics
    assert all(np.isfinite(value) for value in metrics.values())
    assert "bridge.interior_energy_mean" in metrics
    assert "bridge.endpoint_energy" in metrics
    assert "rollout.path_energy_score_fair" in metrics
    assert "rollout.autocovariance_nrmse" in metrics
    assert "rollout.spectral_density_nise" in metrics


def test_bridge_path_noise_uses_same_horizon_time_as_reference_endpoint():
    dataset = simulate_dataset(
        DGPConfig(n_neurons=2, n_modulators=1, n_trajectories=5, n_steps=60, burn_in=5)
    )
    data = build_supervised(dataset, view="complete_state", history_lags=(1,), horizon=4)
    masks = grouped_split(
        data.groups, validation_fraction=.2, test_fraction=.2, seed=2
    ).masks(data)
    train, validation, test = [subset(data, mask) for mask in masks]
    one = subset(test, np.arange(len(test.targets)) == 0)
    diffusion = .3
    model = ConditionalBrownianBridge(reference_diffusion=diffusion).fit(
        train, validation
    )
    paths = model.sample_paths(one, n_paths=40_000, n_steps=4, seed=8)[0]
    midpoint_linear = .5 * (paths[:, 0] + paths[:, -1])
    bridge_noise = paths[:, 2] - midpoint_linear
    empirical = float(np.var(bridge_noise[:, 0]))
    expected = diffusion**2 * one.horizon * .5 * (1.0 - .5)
    assert abs(empirical - expected) / expected < .04
