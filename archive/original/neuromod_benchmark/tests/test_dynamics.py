import numpy as np
import pytest

from neuromod_benchmark.capabilities import UnsupportedCapability
from neuromod_benchmark.dgp import simulate_dataset
from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.methods.dynamics import (
    ConditionalGrangerRidge,
    DiscreteSINDy,
    LaggedCorrelation,
    SparseTransition,
)
from neuromod_benchmark.schema import DGPConfig


def data_parts():
    dataset = simulate_dataset(
        DGPConfig(
            n_neurons=4,
            n_modulators=1,
            n_trajectories=5,
            n_steps=90,
            burn_in=10,
            mechanism="additive_mean",
            seed=12,
        )
    )
    data = build_supervised(dataset, view="latent", history_lags=(1, 2), horizon=1)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=1).masks(data)
    return [subset(data, mask) for mask in masks]


def test_graph_only_correlation_refuses_fake_density():
    train, validation, test = data_parts()
    model = LaggedCorrelation().fit(train, validation)
    assert model.channels_["marginal_lagged_association"].shape == (4, 4)
    with pytest.raises(UnsupportedCapability):
        model.predict(test)


def test_dynamic_baselines_have_canonical_orientation():
    train, validation, test = data_parts()
    models = (
        ConditionalGrangerRidge(ridge=1),
        SparseTransition(alpha=.01),
        DiscreteSINDy(degree=2, threshold=.01, ridge=.01),
    )
    for model in models:
        model.fit(train, validation)
        prediction = model.predict(test)
        assert prediction.mean.shape == test.targets.shape
        channel = next(value for key, value in prediction.channels.items() if key in {
            "linear_transition_coefficient", "conditional_mean_derivative"
        })
        assert channel.shape == (4, 4)  # [target, source]
        assert np.isfinite(prediction.variance).all()
