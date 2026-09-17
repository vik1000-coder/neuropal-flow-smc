import numpy as np
import pytest

from neuromod_benchmark.capabilities import UnsupportedCapability
from neuromod_benchmark.dgp import simulate_dataset
from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.methods.neural import ConditionalScoreMLP, GaussianMLP, MixtureDensityMLP
from neuromod_benchmark.noise_kernels import StudentTNoise
from neuromod_benchmark.schema import DGPConfig


def tiny_parts():
    dataset = simulate_dataset(
        DGPConfig(
            n_neurons=3,
            n_modulators=1,
            n_trajectories=5,
            n_steps=65,
            burn_in=5,
            mechanism="tail_burst",
            seed=21,
        )
    )
    data = build_supervised(dataset, view="latent", history_lags=(1,), horizon=1)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=0).masks(data)
    return [subset(data, mask) for mask in masks]


@pytest.mark.parametrize("objective", ["nll", "dsm"])
def test_gaussian_mlp_early_stopped_predictions(objective):
    train, validation, test = tiny_parts()
    model = GaussianMLP(
        objective=objective,
        hidden=12,
        layers=1,
        dropout=.1,
        max_epochs=5,
        patience=2,
        batch_size=64,
        seed=2,
    ).fit(train, validation)
    prediction = model.predict(test, n_samples=2)
    assert prediction.mean.shape == test.targets.shape
    assert np.all(prediction.variance > 0)
    assert model.trace_.best_epoch < 5
    np.testing.assert_allclose(
        model.tail_high_,
        (0.35 - model.scaler_.y_mean) / model.scaler_.y_scale,
    )
    np.testing.assert_allclose(
        prediction.channels["conditional_shape_tail_derivative"], 0.0, atol=1e-14
    )


def test_mdn_normalized_outputs_and_usage_diagnostics():
    train, validation, test = tiny_parts()
    model = MixtureDensityMLP(
        components=2,
        hidden=12,
        layers=1,
        max_epochs=4,
        patience=2,
        batch_size=64,
        seed=3,
    ).fit(train, validation)
    prediction = model.predict(test, n_samples=2)
    assert np.isfinite(prediction.log_prob).all()
    assert prediction.metadata["min_component_usage"] > 0
    np.testing.assert_allclose(
        model.tail_high_,
        (0.35 - model.scaler_.y_mean) / model.scaler_.y_scale,
    )
    assert np.isfinite(
        prediction.channels["conditional_shape_tail_derivative"]
    ).all()


def test_student_t_dsm_is_score_only():
    train, validation, test = tiny_parts()
    model = ConditionalScoreMLP(
        kernel=StudentTNoise(df=5),
        noise_scale=.3,
        noise_scales=(.1, .3),
        hidden=10,
        layers=1,
        max_epochs=4,
        patience=2,
        batch_size=64,
        seed=4,
    ).fit(train, validation)
    assert np.isfinite(model.dsm_risk(test, seed=5))
    assert np.isfinite(model.selection_dsm_risk(test, seed=5))
    reference = model.dsm_reference_metrics(test, seed=5)
    assert "fixed_reference.student_t_df5.ladder_mean" in reference
    assert model.metadata()["noise_scales"] == (.1, .3)
    assert model.metadata()["score_domain"] == "conditional_outcome"
    with pytest.raises(UnsupportedCapability):
        model.predict(test)
