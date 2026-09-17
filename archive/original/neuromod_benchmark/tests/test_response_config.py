import math
from pathlib import Path

import pytest

from neuromod_benchmark.config import (
    ResponseKernelEvaluationSpec,
    ResponseOperationSpec,
    load_suite,
)


def test_response_operation_contract_accepts_each_well_formed_kind():
    operations = (
        ResponseOperationSpec(id="null_control", kind="null"),
        ResponseOperationSpec(
            id="pulse_m0",
            kind="ligand_pulse",
            modulator_index=0,
            amplitude=1.5,
            duration_steps=3,
        ),
        ResponseOperationSpec(
            id="ko_m0",
            kind="receptor_knockout",
            modulator_index=0,
            target_neurons=(1,),
        ),
        ResponseOperationSpec(
            id="silence_m0",
            kind="release_source_silencing",
            modulator_index=0,
            selection="first_active",
        ),
        ResponseOperationSpec(
            id="expressed_control",
            kind="receptor_knockout",
            modulator_index=0,
            selection="first_active_or_expressed_control",
            expected_effect="null",
        ),
    )
    for operation in operations:
        operation.validate()


@pytest.mark.parametrize(
    "operation",
    (
        ResponseOperationSpec(id="unsafe.id", kind="null"),
        ResponseOperationSpec(id="9starts_numeric", kind="null"),
        ResponseOperationSpec(id="null_fields", kind="null", duration_steps=1),
        ResponseOperationSpec(
            id="zero_pulse",
            kind="ligand_pulse",
            modulator_index=0,
            amplitude=0.0,
            duration_steps=1,
        ),
        ResponseOperationSpec(
            id="infinite_pulse",
            kind="ligand_pulse",
            modulator_index=0,
            amplitude=math.inf,
            duration_steps=1,
        ),
        ResponseOperationSpec(
            id="durationless_pulse",
            kind="ligand_pulse",
            modulator_index=0,
            amplitude=1.0,
        ),
        ResponseOperationSpec(
            id="ko_missing_target",
            kind="receptor_knockout",
            modulator_index=0,
        ),
        ResponseOperationSpec(
            id="ko_ambiguous_target",
            kind="receptor_knockout",
            modulator_index=0,
            target_neurons=(1,),
            selection="first_active",
        ),
        ResponseOperationSpec(
            id="ko_bad_duration",
            kind="receptor_knockout",
            modulator_index=0,
            target_neurons=(1,),
            duration_steps=0,
        ),
        ResponseOperationSpec(
            id="ko_duplicate_target",
            kind="receptor_knockout",
            modulator_index=0,
            target_neurons=(1, 1),
        ),
        ResponseOperationSpec(
            id="ko_wrong_field",
            kind="receptor_knockout",
            modulator_index=0,
            target_neurons=(1,),
            source_neurons=(2,),
        ),
        ResponseOperationSpec(
            id="active_inactive_selection",
            kind="receptor_knockout",
            modulator_index=0,
            selection="first_inactive",
        ),
        ResponseOperationSpec(
            id="null_active_selection",
            kind="receptor_knockout",
            modulator_index=0,
            selection="first_active",
            expected_effect="null",
        ),
    ),
)
def test_response_operation_contract_rejects_malformed_operations(operation):
    with pytest.raises(ValueError):
        operation.validate()


def test_response_evaluation_contract_checks_identifiers_roles_and_dimensionless_threshold():
    with pytest.raises(ValueError, match="unique"):
        ResponseKernelEvaluationSpec(
            horizon=8,
            n_pairs=2,
            operations=(
                ResponseOperationSpec(id="same", kind="null"),
                ResponseOperationSpec(id="same", kind="null"),
            ),
        ).validate()
    with pytest.raises(ValueError, match="maximum_truth_mc"):
        ResponseKernelEvaluationSpec(
            horizon=8,
            n_pairs=2,
            operations=(ResponseOperationSpec(id="null", kind="null"),),
            maximum_truth_mc_se_to_truth_rms_ratio=0.0,
        ).validate()
    with pytest.raises(ValueError, match="minimum_transient_decay"):
        ResponseKernelEvaluationSpec(
            horizon=8,
            n_pairs=2,
            operations=(ResponseOperationSpec(id="null", kind="null"),),
            minimum_transient_decay_estimable_fraction=1.1,
        ).validate()
    with pytest.raises(ValueError, match="innovation_rms"):
        ResponseKernelEvaluationSpec(
            horizon=8,
            n_pairs=2,
            operations=(ResponseOperationSpec(id="null", kind="null"),),
            minimum_active_truth_rms_to_innovation_rms=0.0,
        ).validate()


def test_yaml_and_programmatic_response_precision_defaults_are_identical():
    loaded = load_suite(
        Path(__file__).resolve().parents[1] / "configs" / "response_smoke.yaml"
    ).response_kernel_evaluation
    default = ResponseKernelEvaluationSpec(
        horizon=8,
        n_pairs=2,
        operations=(ResponseOperationSpec(id="null", kind="null"),),
    )
    assert loaded is not None
    assert (
        loaded.maximum_truth_mc_se_to_truth_rms_ratio
        == default.maximum_truth_mc_se_to_truth_rms_ratio
    )
    assert (
        loaded.minimum_transient_decay_estimable_fraction
        == default.minimum_transient_decay_estimable_fraction
    )
