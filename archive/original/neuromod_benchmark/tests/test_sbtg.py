import numpy as np
import pytest

from neuromod_benchmark.capabilities import UnsupportedCapability
from neuromod_benchmark.dgp import simulate_dataset
from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.methods.sbtg import SBTGJointScore
from neuromod_benchmark.noise_kernels import StudentTNoise
from neuromod_benchmark.schema import DGPConfig


def parts():
    dataset = simulate_dataset(
        DGPConfig(
            n_neurons=3,
            n_modulators=1,
            n_trajectories=5,
            n_steps=60,
            burn_in=5,
            mechanism="innovation_variance",
            seed=31,
        )
    )
    data = build_supervised(dataset, view="latent", history_lags=(1,), horizon=1)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=0).masks(data)
    return [subset(data, mask) for mask in masks]


@pytest.mark.parametrize("model_type", ["linear", "feature_bilinear"])
def test_sbtg_continuous_channels_and_heldout_dsm(model_type):
    train, validation, test = parts()
    model = SBTGJointScore(
        model_type=model_type,
        kernel=StudentTNoise(df=5),
        noise_scales=(.1, .3),
        hidden=10,
        layers=1,
        feature_dim=4,
        max_epochs=4,
        patience=2,
        batch_size=64,
        seed=1,
    ).fit(train, validation)
    assert np.isfinite(model.dsm_risk(test, seed=2))
    assert np.isfinite(model.selection_dsm_risk(test, seed=2))
    assert (
        "fixed_reference.student_t_df5.ladder_mean"
        in model.dsm_reference_metrics(test, seed=2)
    )
    assert model.channels_["joint_score_cross_moment"].shape == (3, 3)
    assert model.channels_["joint_squared_score_covariance"].shape == (3, 3)
    assert model.metadata()["orientation"] == "[target, source]"
    assert model.metadata()["noise_scales"] == (.1, .3)
    assert model.metadata()["score_domain"] == "joint_consecutive_state"
    with pytest.raises(UnsupportedCapability):
        model.predict(test)
