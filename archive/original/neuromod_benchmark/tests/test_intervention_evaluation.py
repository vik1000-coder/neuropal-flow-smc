import numpy as np
import pytest

from neuromod_benchmark.capabilities import Capabilities
from neuromod_benchmark.evaluation import evaluate_estimator
from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.mechanistic import MechanisticConfig
from neuromod_benchmark.mechanistic_dataset import simulate_mechanistic_dataset
from neuromod_benchmark.methods.classical import RidgeGaussian
from neuromod_benchmark.schema import Prediction


def test_ligand_dose_arms_have_conditional_mean_and_environment_law_metrics():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=70,
            burn_in=90,
            mechanism="additive_mean",
            ligand_pulse_rate_hz=.15,
            ligand_pulse_amplitude=2.0,
            seed=23,
        ),
        n_trajectories=5,
    )
    data = build_supervised(dataset, view="complete_state", history_lags=(1,), horizon=1)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=2).masks(data)
    train, validation, test = [subset(data, mask) for mask in masks]
    model = RidgeGaussian(ridge=1.0).fit(train, validation)
    metrics, _ = evaluate_estimator(
        model,
        dataset,
        test,
        validation=validation,
        history_lags=(1,),
        n_samples=4,
    )
    assert (
        "arm_history_environment_transfer.ligand_zero_dose.conditional_mean.effect_rmse"
        in metrics
    )
    assert (
        "arm_history_environment_transfer.ligand_double_dose.predictive.nll"
        in metrics
    )
    assert not any(key.startswith("intervention.ligand") for key in metrics)


class _OracleConditionalMean:
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
    )

    def predict(self, data, *, n_samples=0):
        del n_samples
        assert data.oracle_mean is not None
        assert data.oracle_variance is not None
        return Prediction(
            mean=data.oracle_mean,
            variance=data.oracle_variance,
            metadata={"distribution_family": "gaussian"},
        )


@pytest.mark.parametrize(
    "mechanism", ["innovation_variance", "matched_tail", "correlation_routing"]
)
def test_exact_oracle_mean_has_zero_mean_effect_error_despite_pathwise_noise(mechanism):
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=65,
            burn_in=75,
            mechanism=mechanism,
            ligand_pulse_rate_hz=.15,
            ligand_pulse_amplitude=2.0,
            seed=31,
        ),
        n_trajectories=5,
    )
    data = build_supervised(dataset, view="latent", history_lags=(1,), horizon=1)
    masks = grouped_split(
        data.groups, validation_fraction=.2, test_fraction=.2, seed=4
    ).masks(data)
    _, validation, test = [subset(data, mask) for mask in masks]
    metrics, _ = evaluate_estimator(
        _OracleConditionalMean(),
        dataset,
        test,
        validation=validation,
        history_lags=(1,),
        n_samples=0,
    )
    for intervention in ("ligand_zero_dose", "ligand_double_dose"):
        key = (
            f"arm_history_environment_transfer.{intervention}."
            "conditional_mean.effect_rmse"
        )
        assert np.isclose(metrics[key], 0.0)
