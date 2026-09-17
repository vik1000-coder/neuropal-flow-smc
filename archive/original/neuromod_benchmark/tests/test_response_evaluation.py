import numpy as np
import pytest

from neuromod_benchmark.capabilities import Capabilities
from neuromod_benchmark.config import (
    ResponseKernelEvaluationSpec,
    ResponseOperationSpec,
)
from neuromod_benchmark.features import build_supervised
from neuromod_benchmark.mechanistic import MechanisticConfig
from neuromod_benchmark.mechanistic_dataset import simulate_mechanistic_dataset
from neuromod_benchmark.metric_contract import resolve_metric_contract
from neuromod_benchmark.methods.mechanistic_latent import LatentNeuromodulatedSSM
from neuromod_benchmark.response_evaluation import (
    _active_receptor_targets,
    evaluate_response_panels,
    prepare_response_panels,
)
from neuromod_benchmark.schema import Prediction


class _OracleArmPredictor:
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
    )

    def predict(self, data, *, n_samples=0):
        del n_samples
        return Prediction(
            mean=data.oracle_mean.copy(),
            variance=data.oracle_variance.copy(),
            metadata={"distribution_family": "gaussian"},
        )


class _ExplicitOracleArmPredictor(_OracleArmPredictor):
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        interventions=True,
    )

    def __init__(self, *, anchored=False):
        self.anchored = anchored
        self.explicit_calls = 0

    def predict_intervention(self, data, specification, *, n_samples=0):
        assert specification.kind == "receptor_knockout"
        self.explicit_calls += 1
        return self.predict(data, n_samples=n_samples)

    def metadata(self):
        return {"modulator_label_mapping_anchored": self.anchored}


class _LeakyExplicitPredictor(_ExplicitOracleArmPredictor):
    def predict_intervention(self, data, specification, *, n_samples=0):
        prediction = super().predict_intervention(
            data, specification, n_samples=n_samples
        )
        prediction.mean = prediction.mean + 0.2
        return prediction


def _panel_spec():
    return ResponseKernelEvaluationSpec(
        horizon=12,
        n_pairs=3,
        operations=(
            ResponseOperationSpec(id="null", kind="null"),
            ResponseOperationSpec(
                id="pulse",
                kind="ligand_pulse",
                modulator_index=0,
                amplitude=1.5,
                duration_steps=2,
            ),
            ResponseOperationSpec(
                id="receptor_ko",
                kind="receptor_knockout",
                modulator_index=0,
                selection="first_active",
            ),
            ResponseOperationSpec(
                id="source_silence",
                kind="release_source_silencing",
                modulator_index=0,
                selection="first_active",
            ),
        ),
    )


def test_response_panels_match_training_feature_contract_and_select_active_edges():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=4,
            n_modulators=1,
            n_steps=40,
            burn_in=45,
            mechanism="mixed",
            seed=121,
        ),
        n_trajectories=4,
    )
    history = (1, 3)
    panels = prepare_response_panels(dataset, _panel_spec(), history_lags=history)
    training = build_supervised(
        dataset, view="complete_state", history_lags=history, horizon=1
    )
    assert len(panels) == 4
    assert panels[0].baseline.feature_names == training.feature_names
    assert all(panel.baseline.features.shape == panel.intervention.features.shape for panel in panels)
    assert np.any(panels[0].baseline.times < 0)
    assert np.sum(panels[0].post_onset_mask) == len(panels[0].pairs) * 12
    np.testing.assert_array_equal(
        panels[0].baseline.features[~panels[0].post_onset_mask],
        panels[0].intervention.features[~panels[0].post_onset_mask],
    )
    receptor = next(panel for panel in panels if panel.identifier == "receptor_ko")
    source = next(panel for panel in panels if panel.identifier == "source_silence")
    assert receptor.specification.target_neurons
    assert source.specification.source_neurons
    # Pair i is the same Monte Carlo history under every registered operation.
    for pair_index in range(3):
        reference = panels[0].pairs[pair_index]
        for panel in panels[1:]:
            assert panel.pairs[pair_index].exogenous_fingerprint == reference.exogenous_fingerprint
            np.testing.assert_array_equal(
                panel.pairs[pair_index].baseline.neural_state,
                reference.baseline.neural_state,
            )


def test_oracle_arm_predictor_is_exact_and_claim_depends_on_explicit_operation():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=32,
            burn_in=36,
            mechanism="additive_mean",
            seed=123,
        ),
        n_trajectories=4,
    )
    panels = prepare_response_panels(dataset, _panel_spec(), history_lags=(1,))
    passive = evaluate_response_panels(_OracleArmPredictor(), panels)
    explicit = evaluate_response_panels(_ExplicitOracleArmPredictor(), panels)

    pulse_key = (
        "arm_history_environment_transfer.pulse.population_mean.response_kernel.normalized_rmse"
    )
    causal_key = (
        "causal_intervention.receptor_ko.common_history_lag1.normalized_rmse"
    )
    assert passive[pulse_key] == 0.0
    assert explicit[causal_key] == 0.0
    assert (
        explicit[
            "intervention.receptor_ko.arm_history.population_mean.response_kernel.normalized_rmse"
        ]
        == 0.0
    )
    assert (
        explicit[
            "intervention.receptor_ko.arm_history.history_conditional.response_kernel.pairwise_rmse_mean"
        ]
        == 0.0
    )
    assert not any(
        key.startswith("causal_intervention.receptor_ko.response_kernel")
        for key in explicit
    )
    assert not any(key.startswith("causal_intervention") for key in passive)
    assert (
        passive[
            "arm_history_environment_transfer.null.population_mean.response_kernel.null_leakage_rms"
        ]
        == 0.0
    )
    contracts = {
        key: resolve_metric_contract(key) for key in set(passive) | set(explicit)
    }
    assert all(contract.claim_level in {"P1", "C1"} for contract in contracts.values())
    assert (
        contracts[
            "intervention.receptor_ko.arm_history.history_conditional.response_kernel.pairwise_normalized_rmse_mean"
        ].estimand
        == "history_conditional_intervention_response"
    )
    assert (
        contracts[
            "intervention.receptor_ko.arm_history.population_mean.response_kernel.normalized_rmse"
        ].role
        == "secondary"
    )
    assert contracts[
        "causal_intervention.receptor_ko.common_history_lag1.normalized_rmse"
    ].role == "primary"
    assert contracts[
        "arm_history_environment_transfer.pulse.population_mean.response_kernel.truth_rms_to_innovation_rms"
    ].unit == "dimensionless"
    assert contracts[
        "arm_history_environment_transfer.pulse.population_mean.response_kernel.end_window_gain_applicable"
    ].unit == "dimensionless"
    assert contracts[
        "arm_history_environment_transfer.pulse.history_conditional.response_kernel.pairwise_normalized_rmse_defined_count"
    ].unit == "count"


def test_finite_latent_knockout_schedule_excludes_context_and_turns_off(monkeypatch):
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=28,
            burn_in=32,
            mechanism="additive_mean",
            seed=133,
        ),
        n_trajectories=4,
    )
    panels = prepare_response_panels(dataset, _panel_spec(), history_lags=(1, 3))
    panel = next(panel for panel in panels if panel.identifier == "receptor_ko")
    model = LatentNeuromodulatedSSM(n_modulators=1)
    model.n_targets_ = dataset.config.n_neurons
    captured = {}

    def capture(data, *, knockout, n_samples):
        del data, n_samples
        captured["schedule"] = knockout
        return object()

    monkeypatch.setattr(model, "_prediction", capture)
    finite = type(panel.specification).receptor_knockout(
        0, panel.specification.target_neurons, duration_steps=2
    )
    model.predict_intervention(panel.intervention, finite)
    schedule = captured["schedule"]
    active = schedule[:, panel.specification.target_neurons[0], 0]
    expected = (panel.intervention.times >= 0) & (panel.intervention.times < 2)
    np.testing.assert_array_equal(active, expected)
    assert not np.any(schedule[panel.intervention.times < 0])


def test_expected_null_receptor_sham_exercises_explicit_api_and_scores_leakage():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=5,
            n_modulators=1,
            n_steps=30,
            burn_in=35,
            mechanism="additive_mean",
            receptor_density=0.35,
            seed=141,
        ),
        n_trajectories=4,
    )
    spec = ResponseKernelEvaluationSpec(
        horizon=8,
        n_pairs=3,
        operations=(
            ResponseOperationSpec(
                id="receptor_sham",
                kind="receptor_knockout",
                modulator_index=0,
                selection="first_inactive",
                expected_effect="null",
            ),
        ),
    )
    panels = prepare_response_panels(dataset, spec, history_lags=(1,))
    assert panels[0].truth_rms_to_innovation_rms == 0.0
    estimator = _LeakyExplicitPredictor()
    metrics = evaluate_response_panels(estimator, panels)
    assert estimator.explicit_calls == 1
    assert (
        metrics[
            "intervention.receptor_sham.arm_history.population_mean.response_kernel.null_leakage_rms"
        ]
        == pytest.approx(0.2)
    )
    assert metrics[
        "causal_intervention.receptor_sham.common_history_lag1.null_leakage_rms"
    ] == pytest.approx(0.2)


def test_multimodulator_explicit_claim_requires_anchored_label_mapping():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=4,
            n_modulators=2,
            n_steps=30,
            burn_in=36,
            mechanism="mixed",
            seed=151,
        ),
        n_trajectories=4,
    )
    spec = ResponseKernelEvaluationSpec(
        horizon=8,
        n_pairs=2,
        operations=(
            ResponseOperationSpec(
                id="receptor_ko",
                kind="receptor_knockout",
                modulator_index=0,
                selection="first_active",
            ),
        ),
        minimum_active_truth_rms_to_innovation_rms=1e-8,
    )
    panels = prepare_response_panels(dataset, spec, history_lags=(1,))
    unanchored = _ExplicitOracleArmPredictor()
    unanchored_metrics = evaluate_response_panels(unanchored, panels)
    assert unanchored.explicit_calls == 0
    assert not any(key.startswith("causal_intervention.") for key in unanchored_metrics)

    anchored = _ExplicitOracleArmPredictor(anchored=True)
    anchored_metrics = evaluate_response_panels(anchored, panels)
    assert anchored.explicit_calls == 1
    assert any(key.startswith("causal_intervention.") for key in anchored_metrics)


def test_mean_response_selector_excludes_pure_stochastic_receptor_effects():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=5,
            n_modulators=1,
            n_steps=25,
            burn_in=30,
            mechanism="innovation_variance",
            receptor_density=1.0,
            seed=161,
        ),
        n_trajectories=4,
    )
    assert np.any(np.abs(dataset.parameters.logvariance_effect[:, 0]) > 0)
    assert len(_active_receptor_targets(dataset, 0)) == 0


def test_expressed_control_fallback_is_explicit_and_only_used_for_null_role():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=4,
            n_modulators=1,
            n_steps=25,
            burn_in=30,
            mechanism="null",
            receptor_density=1.0,
            seed=171,
        ),
        n_trajectories=4,
    )
    control = ResponseKernelEvaluationSpec(
        horizon=8,
        n_pairs=2,
        operations=(
            ResponseOperationSpec(
                id="expressed_control",
                kind="receptor_knockout",
                modulator_index=0,
                selection="first_active_or_expressed_control",
                expected_effect="null",
            ),
        ),
    )
    panels = prepare_response_panels(dataset, control, history_lags=(1,))
    assert panels[0].specification.target_neurons
    assert panels[0].truth_rms_to_innovation_rms == 0.0

    wrongly_active = ResponseKernelEvaluationSpec(
        horizon=8,
        n_pairs=2,
        operations=(
            ResponseOperationSpec(
                id="wrongly_active",
                kind="receptor_knockout",
                modulator_index=0,
                selection="first_active_or_expressed_control",
                expected_effect="active",
            ),
        ),
    )
    with pytest.raises(ValueError, match="no active"):
        prepare_response_panels(dataset, wrongly_active, history_lags=(1,))


def test_active_response_preflight_gates_truth_precision_and_decay_observability():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=4,
            n_modulators=1,
            n_steps=40,
            burn_in=45,
            mechanism="additive_mean",
            seed=181,
        ),
        n_trajectories=4,
    )
    operation = ResponseOperationSpec(
        id="pulse",
        kind="ligand_pulse",
        modulator_index=0,
        amplitude=1.5,
        duration_steps=2,
    )
    with pytest.raises(ValueError, match="MC-SE/truth RMS"):
        prepare_response_panels(
            dataset,
            ResponseKernelEvaluationSpec(
                horizon=12,
                n_pairs=3,
                operations=(operation,),
                minimum_active_truth_rms_to_innovation_rms=1e-8,
                maximum_truth_mc_se_to_truth_rms_ratio=1e-12,
            ),
            history_lags=(1,),
        )
    with pytest.raises(ValueError, match="1/e crossing estimable fraction"):
        prepare_response_panels(
            dataset,
            ResponseKernelEvaluationSpec(
                horizon=8,
                n_pairs=3,
                operations=(operation,),
                minimum_active_truth_rms_to_innovation_rms=1e-8,
                maximum_truth_mc_se_to_truth_rms_ratio=1e6,
                minimum_transient_decay_estimable_fraction=1.0,
            ),
            history_lags=(1,),
        )
