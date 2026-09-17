import numpy as np
import pytest
from types import SimpleNamespace

from neuromod_benchmark.capabilities import Capabilities, validate_channel_contract
from neuromod_benchmark.evaluation import metric_parameter_binding


def test_declared_effect_channel_must_be_emitted():
    capabilities = Capabilities(effect_channels=("conditional_mean_derivative",))
    with pytest.raises(ValueError, match="missing"):
        validate_channel_contract(capabilities, {})


def test_semantic_effect_channel_must_be_declared():
    channels = {"conditional_mean_derivative": np.eye(2)}
    with pytest.raises(ValueError, match="not declared"):
        validate_channel_contract(Capabilities(), channels)


def test_feature_tensor_satisfies_declared_channel():
    capabilities = Capabilities(
        effect_channels=("conditional_covariance_derivative",)
    )
    validate_channel_contract(
        capabilities,
        {"conditional_covariance_derivative_feature": np.zeros((2, 2, 3))},
    )


class _ThresholdEstimator:
    capabilities = Capabilities(
        effect_channels=(
            "conditional_tail_high_derivative",
            "conditional_shape_tail_derivative",
        )
    )

    def __init__(self, threshold=.35, shape=2.0):
        self.threshold = threshold
        self.shape = shape

    def metadata(self):
        return {
            "tail_threshold_physical": self.threshold,
            "shape_tail_z": self.shape,
        }


def test_metric_thresholds_are_bound_to_the_registered_oracle():
    dataset = SimpleNamespace(
        config=SimpleNamespace(tail_threshold=.35, shape_tail_z=2.0)
    )
    assert metric_parameter_binding(_ThresholdEstimator(), dataset)[
        "diagnostic.metric_parameter_binding_valid"
    ] == 1.0
    with pytest.raises(ValueError, match="tail threshold mismatch"):
        metric_parameter_binding(_ThresholdEstimator(threshold=.5), dataset)
    with pytest.raises(ValueError, match="shape-tail threshold mismatch"):
        metric_parameter_binding(_ThresholdEstimator(shape=2.5), dataset)
