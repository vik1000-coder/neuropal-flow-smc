import numpy as np

from neuromod_benchmark.evaluation import transition_operator_probe_metrics
from neuromod_benchmark.schema import Prediction, SupervisedData


def test_exact_moment_and_tail_operator_probes_are_zero():
    mean = np.asarray([[.1, -.2], [.2, -.1], [.3, 0.0], [.4, .1]])
    variance = np.full_like(mean, .25)
    targets = mean.copy()
    samples = np.empty((4, 2, 2))
    samples[:, 0, :] = -1.0
    samples[:, 1, :] = 1.0
    data = SupervisedData(
        features=np.zeros((4, 2)),
        targets=targets,
        groups=np.asarray([0, 0, 1, 1]),
        trajectory_ids=np.asarray([0, 0, 1, 1]),
        times=np.asarray([0, 1, 0, 1]),
        feature_names=("x0:lag1", "x1:lag1"),
        source_index=np.asarray([0, 1]),
        view="complete_state",
        horizon=1,
        oracle_mean=mean,
        oracle_variance=variance,
        oracle_tail_probability=np.full_like(mean, .5),
        oracle_shape_tail_probability=np.mean(
            np.abs((samples - mean[:, None, :]) / np.sqrt(variance[:, None, :]))
            > 2.0,
            axis=1,
        ),
    )
    prediction = Prediction(mean=mean, variance=variance, samples=samples)
    metrics = transition_operator_probe_metrics(prediction, data, tail_threshold=0.0)
    assert metrics["operator.linear_probe_nrmse"] == 0.0
    assert metrics["operator.quadratic_probe_nrmse"] == 0.0
    assert metrics["operator.tail_exceedance_rmse"] == 0.0
    assert metrics["operator.shape_tail_exceedance_rmse"] == 0.0


def test_matched_mean_variance_do_not_hide_wrong_standardized_shape():
    mean = np.zeros((6, 1))
    variance = np.ones_like(mean)
    # The forecast samples have the declared mean/variance scale but no |Z|>2
    # events, while the oracle law has a 25% standardized shape-tail probability.
    samples = np.empty((6, 2, 1))
    samples[:, 0, 0] = -1.0
    samples[:, 1, 0] = 1.0
    data = SupervisedData(
        features=np.zeros((6, 1)),
        targets=mean.copy(),
        groups=np.asarray([0, 0, 1, 1, 2, 2]),
        trajectory_ids=np.asarray([0, 0, 1, 1, 2, 2]),
        times=np.asarray([0, 1, 0, 1, 0, 1]),
        feature_names=("x0:lag1",),
        source_index=np.asarray([0]),
        view="complete_state",
        horizon=1,
        oracle_mean=mean.copy(),
        oracle_variance=variance.copy(),
        oracle_shape_tail_probability=np.full_like(mean, 0.25),
    )
    prediction = Prediction(mean=mean, variance=variance, samples=samples)
    metrics = transition_operator_probe_metrics(
        prediction, data, tail_threshold=0.0, shape_tail_z=2.0
    )
    assert metrics["operator.linear_probe_nrmse"] == 0.0
    assert metrics["operator.quadratic_probe_nrmse"] == 0.0
    assert metrics["operator.shape_tail_exceedance_rmse"] == 0.25
