from pathlib import Path

import numpy as np
import pytest

from neuromod_benchmark.capabilities import UnsupportedCapability
from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.mechanistic import MechanisticConfig
from neuromod_benchmark.mechanistic_dataset import simulate_mechanistic_dataset
from neuromod_benchmark.external_causal import (
    DEFAULT_WORKER_PYTHON,
    PCMCIParCorr,
    PySINDyDiscreteOfficial,
    VARLiNGAMOfficial,
)


pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_WORKER_PYTHON).exists(), reason="optional causal environment not installed"
)


def _parts():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=90,
            burn_in=120,
            mechanism="additive_mean",
            ligand_pulse_rate_hz=0.1,
            seed=14,
        ),
        n_trajectories=5,
    )
    data = build_supervised(dataset, view="complete_state", history_lags=(1,), horizon=1)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=3).masks(data)
    return [subset(data, mask) for mask in masks]


def test_official_pysindy_is_predictive_and_oriented():
    train, validation, test = _parts()
    model = PySINDyDiscreteOfficial(degree=2, threshold=.05, alpha=.02).fit(
        train, validation
    )
    prediction = model.predict(test, n_samples=4)
    assert prediction.mean.shape == test.targets.shape
    assert prediction.samples.shape == (len(test.targets), 4, test.targets.shape[1])
    assert model.channels_["conditional_mean_derivative"].shape == (3, 3)
    assert np.isfinite(prediction.log_prob).all()
    assert model.metadata()["algorithm"] == "DiscreteSINDy-STLSQ"


def test_pcmci_and_var_lingam_are_graph_only_and_oriented():
    train, validation, test = _parts()
    methods = (
        (PCMCIParCorr(tau_max=1), "conditional_independence_strength"),
        (VARLiNGAMOfficial(lags=1, prune=False), "linear_transition_coefficient"),
    )
    for model, channel in methods:
        model.fit(train, validation)
        assert model.channels_[channel].shape == (3, 3)
        assert np.isfinite(model.channels_[channel]).all()
        with pytest.raises(UnsupportedCapability):
            model.predict(test)


def test_published_causal_adapters_reject_duplicate_lag_variables():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(n_neurons=3, n_modulators=1, n_steps=50, burn_in=70, seed=2),
        n_trajectories=4,
    )
    data = build_supervised(dataset, view="latent", history_lags=(1, 2), horizon=1)
    with pytest.raises(UnsupportedCapability):
        PCMCIParCorr().fit(data)
