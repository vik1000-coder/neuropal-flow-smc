import numpy as np
import torch

from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.capabilities import Capabilities
from neuromod_benchmark.evaluation import latent_recovery_metrics
from neuromod_benchmark.intervention_response import InterventionSpec
from neuromod_benchmark.mechanistic import MechanisticConfig
from neuromod_benchmark.mechanistic_dataset import simulate_mechanistic_dataset
from neuromod_benchmark.methods.mechanistic_latent import (
    LatentNeuromodulatedSSM,
    _MechanisticCore,
)


class _OracleLatentWithPotentialEffects:
    capabilities = Capabilities(latent_state=True)

    def __init__(self, dataset):
        self.dataset = dataset

    def latent_states(self, data):
        return np.asarray(
            [
                self.dataset.trajectories[int(group)].modulator[int(time + data.horizon)]
                for group, time in zip(data.groups, data.times)
            ]
        )

    def metadata(self):
        parameters = self.dataset.parameters
        return {
            "latent_orientation": "positive_concentration",
            "learned_additive_effect": parameters.additive_effect,
            "learned_synaptic_effect": parameters.synaptic_effect,
            "learned_intrinsic_slope_effect": parameters.intrinsic_log_slope_effect,
            "learned_intrinsic_threshold_effect": parameters.intrinsic_threshold_effect,
            "learned_logvariance_effect": parameters.logvariance_effect,
            "learned_tail_logit_effect": parameters.tail_logit_effect,
        }


class _SignReversedPositiveLatent(_OracleLatentWithPotentialEffects):
    def latent_states(self, data):
        return -super().latent_states(data)


def test_explicit_latent_neuromodulator_model_trains_and_recovers_positive_state():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=60,
            burn_in=30,
            mechanism="innovation_variance",
            seed=14,
        ),
        n_trajectories=5,
    )
    data = build_supervised(dataset, view="latent", history_lags=(1,), horizon=1)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=0).masks(data)
    train, validation, test = [subset(data, mask) for mask in masks]
    model = LatentNeuromodulatedSSM(
        n_modulators=1,
        max_epochs=4,
        patience=2,
        truncation=32,
        warmup=2,
        seed=2,
    ).fit(train, validation)
    prediction = model.predict(test, n_samples=2)
    states = model.latent_states(test)
    assert prediction.mean.shape == test.targets.shape
    assert states.shape == (len(test.targets), 1)
    assert np.all(states > 0)
    assert np.all(prediction.variance > 0)
    knockout = model.predict_receptor_knockout(test)
    assert knockout.mean.shape == test.targets.shape
    specific = model.predict_intervention(
        test, InterventionSpec.receptor_knockout(0, 0)
    )
    assert specific.mean.shape == test.targets.shape
    metadata = model.metadata()
    assert metadata["uses_current_modulator_features"] is False
    assert "modulator reconstructed" in metadata["effective_information_set"]


def test_mechanistic_core_applies_rowwise_knockout_only_on_scheduled_steps():
    core = _MechanisticCore(n=2, k=1, n_inputs=0)
    with torch.no_grad():
        core.additive_effect.fill_(1.0)
        core.synaptic_effect.zero_()
        core.intrinsic_slope_effect.zero_()
        core.intrinsic_threshold_effect.zero_()
        core.logvariance_effect.zero_()
        core.tail_logit_effect.zero_()
        core.bias.zero_()
    x = torch.zeros((6, 2))
    stimulus = torch.empty((6, 0))
    schedule = torch.zeros((6, 2, 1), dtype=torch.bool)
    schedule[2:4, 0, 0] = True
    baseline_mean = core.sequence(x, stimulus)[0].detach().numpy()
    scheduled_mean = core.sequence(x, stimulus, knockout=schedule)[0].detach().numpy()

    np.testing.assert_allclose(scheduled_mean[:2], baseline_mean[:2])
    np.testing.assert_allclose(scheduled_mean[4:], baseline_mean[4:])
    assert np.all(scheduled_mean[2:4, 0] < baseline_mean[2:4, 0])
    np.testing.assert_allclose(scheduled_mean[2:4, 1], baseline_mean[2:4, 1])


def test_explicit_mixture_ssm_defines_a_normalized_mean_variance_matched_tail_law():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=65,
            burn_in=80,
            mechanism="matched_tail",
            effect_strength=2.0,
            ligand_pulse_rate_hz=.1,
            seed=19,
        ),
        n_trajectories=5,
    )
    data = build_supervised(dataset, view="latent", history_lags=(1,), horizon=1)
    masks = grouped_split(data.groups, validation_fraction=.2, test_fraction=.2, seed=1).masks(data)
    train, validation, test = [subset(data, mask) for mask in masks]
    model = LatentNeuromodulatedSSM(
        n_modulators=1,
        emission="matched_tail_mixture",
        max_epochs=4,
        patience=2,
        truncation=32,
        warmup=2,
        seed=3,
    ).fit(train, validation)
    prediction = model.predict(test, n_samples=2_000)
    assert np.isfinite(prediction.log_prob).all()
    assert np.all((prediction.cdf >= 0) & (prediction.cdf <= 1))
    empirical_mean = prediction.samples.mean(axis=1)
    assert np.mean(np.abs(empirical_mean - prediction.mean)) < .03
    probability = prediction.metadata["mixture_probability"]
    analytic_variance = (
        (1.0 - probability) * prediction.metadata["component_variance_low"]
        + probability * prediction.metadata["component_variance_high"]
    )
    np.testing.assert_allclose(analytic_variance, prediction.variance, rtol=2e-14)
    assert "conditional_tail_high_derivative" in prediction.channels
    assert "conditional_shape_tail_derivative" in prediction.channels
    assert model.metadata()["emission"] == "matched_tail_mixture"


def test_latent_recovery_treats_predrawn_inactive_effects_as_zero_truth():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=45,
            burn_in=40,
            mechanism="null",
            seed=24,
        ),
        n_trajectories=5,
    )
    data = build_supervised(dataset, view="latent", history_lags=(1,), horizon=1)
    masks = grouped_split(
        data.groups, validation_fraction=.2, test_fraction=.2, seed=5
    ).masks(data)
    _, validation, test = [subset(data, mask) for mask in masks]
    metrics = latent_recovery_metrics(
        _OracleLatentWithPotentialEffects(dataset), dataset, validation, test
    )
    assert metrics["latent.test_spearman"] > .999
    assert metrics["latent.mechanism.additive_mean.inactive_rms"] > 0
    assert "latent.mechanism.additive_mean.cosine" not in metrics
    assert "latent.mechanism.stochastic_dispersion.cosine" not in metrics


def test_positive_latent_orientation_reports_raw_negative_alignment_slope():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=45,
            burn_in=40,
            mechanism="null",
            seed=241,
        ),
        n_trajectories=5,
    )
    data = build_supervised(dataset, view="latent", history_lags=(1,), horizon=1)
    masks = grouped_split(
        data.groups, validation_fraction=.2, test_fraction=.2, seed=5
    ).masks(data)
    _, validation, test = [subset(data, mask) for mask in masks]
    metrics = latent_recovery_metrics(
        _SignReversedPositiveLatent(dataset), dataset, validation, test
    )
    assert metrics["latent.alignment_negative_slope_fraction"] == 1.0


def test_calcium_latent_tracking_does_not_emit_physical_parameter_recovery():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=45,
            burn_in=40,
            mechanism="mixed",
            seed=25,
        ),
        n_trajectories=5,
    )
    data = build_supervised(dataset, view="calcium", history_lags=(1,), horizon=1)
    masks = grouped_split(
        data.groups, validation_fraction=.2, test_fraction=.2, seed=6
    ).masks(data)
    _, validation, test = [subset(data, mask) for mask in masks]
    metrics = latent_recovery_metrics(
        _OracleLatentWithPotentialEffects(dataset), dataset, validation, test
    )
    assert metrics["latent.test_spearman"] > .999
    assert (
        metrics["latent.parameter_recovery_omitted_unidentified_observation_model"]
        == 1.0
    )
    assert "latent.clearance_tau_mae_seconds" not in metrics
    assert "latent.mechanism.additive_mean.cosine" not in metrics
