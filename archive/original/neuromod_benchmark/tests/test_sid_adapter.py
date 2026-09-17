import numpy as np

from neuromod_benchmark.dgp import simulate_dataset
from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.metrics import predictive_metrics
from neuromod_benchmark.methods.sid import GaussianNLLAdapter, SIDQuadratic
from neuromod_benchmark.schema import DGPConfig


def split_data():
    dataset = simulate_dataset(
        DGPConfig(
            n_neurons=3,
            n_modulators=1,
            n_trajectories=5,
            n_steps=140,
            burn_in=20,
            mechanism="innovation_variance",
            seed=9,
        )
    )
    data = build_supervised(dataset, view="latent", history_lags=(1, 2), horizon=1)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=2).masks(data)
    return [subset(data, mask) for mask in masks]


def test_sid_hyvarinen_and_dsm_end_to_end():
    train, validation, test = split_data()
    for sigma in (0.0, 0.2):
        model = SIDQuadratic(ridge=.1, sigma_fraction=sigma, n_corruptions=2, seed=1)
        model.fit(train, validation)
        prediction = model.predict(test, n_samples=3)
        metrics = predictive_metrics(prediction, test.targets)
        assert np.isfinite(metrics["nll"])
        assert prediction.channels["conditional_log_variance_derivative"].shape == (3, 3)
        np.testing.assert_allclose(
            model.tail_high_,
            (0.35 - model.scaler_.y_mean) / model.scaler_.y_scale,
        )


def test_gaussian_nll_adapter_end_to_end():
    train, validation, test = split_data()
    model = GaussianNLLAdapter(ridge=.01, maxiter=50).fit(train, validation)
    prediction = model.predict(test)
    assert np.isfinite(prediction.variance).all()
    assert np.all(prediction.variance > 0)
