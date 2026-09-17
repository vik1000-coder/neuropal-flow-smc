import numpy as np

from neuromod_benchmark.evaluation import _state_field_summary


def test_state_field_metric_does_not_reward_signed_cancellation():
    positive = np.asarray([[1.0, -0.5], [0.25, 0.75]])
    truth = np.stack([positive, -positive])
    estimate = np.zeros_like(truth)
    metrics = _state_field_summary(truth, estimate)

    # The signed average is exactly zero in both fields, but the zero estimator
    # misses all state-dependent dynamics and must incur unit normalized error.
    assert metrics["signed_mean_mae"] == 0.0
    assert metrics["truth_rms"] > 0
    assert metrics["rmse"] > 0
    assert metrics["nise"] == 1.0
    assert metrics["rms_map_mae"] > 0


def test_state_field_exact_oracle_and_null_contracts_are_distinct():
    truth = np.asarray([[[1.0, -0.5]], [[0.25, 0.75]]])
    exact = _state_field_summary(truth, truth.copy())
    assert exact["rmse"] == 0.0
    assert exact["nise"] == 0.0
    assert exact["field_cosine"] == 1.0

    null_truth = np.zeros_like(truth)
    leaked = np.full_like(truth, 0.2)
    null = _state_field_summary(null_truth, leaked)
    assert np.isnan(null["nise"])
    assert np.isnan(null["field_cosine"])
    assert null["estimate_rms"] == 0.2
    assert null["null_estimate_max_abs"] == 0.2
