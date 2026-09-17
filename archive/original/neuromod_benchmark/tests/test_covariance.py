import numpy as np

from neuromod_benchmark.capabilities import validate_channel_contract
from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.mechanistic import MechanisticConfig
from neuromod_benchmark.mechanistic_dataset import simulate_mechanistic_dataset
from neuromod_benchmark.methods.covariance import (
    ConditionalCovarianceRidge,
    FullCovarianceRidge,
)


def _parts():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=4,
            n_modulators=1,
            n_steps=75,
            burn_in=90,
            mechanism="innovation_variance",
            ligand_pulse_rate_hz=.1,
            seed=41,
        ),
        n_trajectories=5,
    )
    data = build_supervised(dataset, view="complete_state", history_lags=(1,), horizon=1)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=1).masks(data)
    return [subset(data, mask) for mask in masks]


def test_multivariate_covariance_baselines_are_spd_normalized_and_oriented():
    train, validation, test = _parts()
    methods = (
        FullCovarianceRidge(ridge=1, covariance_shrinkage=.1),
        ConditionalCovarianceRidge(
            mean_ridge=1,
            variance_ridge=5,
            correlation_ridge=5,
            correlation_scale=.35,
        ),
    )
    for model in methods:
        model.fit(train, validation)
        validate_channel_contract(model.capabilities, model.channels_)
        prediction = model.predict(test, n_samples=4)
        covariance = prediction.metadata["conditional_covariance"]
        assert prediction.log_prob.shape == test.targets.shape
        assert prediction.samples.shape == (len(test.targets), 4, 4)
        assert np.isfinite(prediction.log_prob).all()
        assert np.all(np.linalg.eigvalsh(covariance) > 0)
        assert model.channels_["linear_transition_coefficient"].shape == (4, 4)


def test_conditional_covariance_derivative_tensor_uses_target_target_feature_orientation():
    train, validation, _ = _parts()
    model = ConditionalCovarianceRidge().fit(train, validation)
    tensor = model.channels_["conditional_correlation_derivative_feature"]
    assert tensor.shape == (4, 4, train.features.shape[1])
    np.testing.assert_allclose(tensor, np.swapaxes(tensor, 0, 1), atol=1e-10)
