from neuromod_benchmark.capabilities import validate_channel_contract
from neuromod_benchmark.dgp import simulate_dataset
from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.metrics import predictive_metrics
from neuromod_benchmark.methods.classical import HeteroskedasticRidge, RidgeGaussian, StudentTRidge
from neuromod_benchmark.schema import DGPConfig


def split_data():
    dataset = simulate_dataset(
        DGPConfig(
            n_neurons=4,
            n_modulators=1,
            n_trajectories=5,
            n_steps=100,
            burn_in=10,
            mechanism="innovation_variance",
            seed=3,
        )
    )
    data = build_supervised(dataset, view="latent", history_lags=(1, 2), horizon=1)
    split = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=0)
    masks = split.masks(data)
    return [subset(data, mask) for mask in masks]


def test_classical_models_produce_finite_predictions():
    train, validation, test = split_data()
    for model in (
        RidgeGaussian(ridge=1.0),
        HeteroskedasticRidge(mean_ridge=1.0, variance_ridge=10.0),
        StudentTRidge(mean_ridge=1.0, variance_ridge=10.0, df=5),
    ):
        model.fit(train, validation)
        validate_channel_contract(model.capabilities, model.channels_)
        prediction = model.predict(test, n_samples=4)
        metrics = predictive_metrics(prediction, test.targets)
        assert metrics["nll"] == metrics["nll"]
        assert prediction.channels["mean"].shape == (4, 4)
        if isinstance(model, (HeteroskedasticRidge, StudentTRidge)):
            assert "linear_transition_coefficient" not in model.channels_
