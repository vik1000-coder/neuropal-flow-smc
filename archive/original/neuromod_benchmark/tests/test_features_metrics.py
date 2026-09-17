import numpy as np
import pytest

from neuromod_benchmark.dgp import simulate_dataset
from neuromod_benchmark.features import build_supervised, grouped_split
from neuromod_benchmark.metrics import graph_metrics, predictive_metrics
from neuromod_benchmark.schema import DGPConfig, Prediction


def dataset():
    return simulate_dataset(
        DGPConfig(n_neurons=4, n_modulators=1, n_trajectories=5, n_steps=60, burn_in=5)
    )


def test_supervised_never_crosses_group_boundaries():
    data = build_supervised(dataset(), view="latent", history_lags=(1, 3), horizon=2)
    assert data.features.shape[1] == 2 * 4 + 1
    assert set(np.unique(data.groups)) == set(range(5))
    for group in np.unique(data.groups):
        group_times = data.times[data.groups == group]
        assert np.all(np.diff(group_times) == 1)


def test_group_split_disjoint():
    data = build_supervised(dataset(), view="latent", history_lags=(1,), horizon=1)
    split = grouped_split(data.groups, validation_fraction=0.2, test_fraction=0.2, seed=4)
    assert set(split.train_groups).isdisjoint(split.validation_groups)
    assert set(split.train_groups).isdisjoint(split.test_groups)
    assert set(split.validation_groups).isdisjoint(split.test_groups)


def test_missing_calcium_is_never_forward_filled_into_scored_targets():
    generated = simulate_dataset(
        DGPConfig(
            n_neurons=3,
            n_modulators=1,
            n_trajectories=4,
            n_steps=80,
            burn_in=5,
            missing_rate=.5,
            seed=12,
        )
    )
    data = build_supervised(
        generated, view="calcium", history_lags=(1,), horizon=1
    )
    expected = sum(
        np.all(trajectory.observed_mask[1:], axis=1).sum()
        for trajectory in generated.trajectories
    )
    assert len(data.targets) == expected
    assert np.isfinite(data.targets).all()


def test_calibrated_prediction_has_small_errors():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(20_000, 1))
    prediction = Prediction(
        mean=np.zeros_like(y),
        variance=np.ones_like(y),
        metadata={"distribution_family": "gaussian"},
    )
    metrics = predictive_metrics(prediction, y)
    assert metrics["coverage_error_90"] < 0.02
    assert metrics["pit_ece"] < 0.01


def test_normalized_law_is_not_silently_inferred_from_moments():
    y = np.zeros((8, 1))
    prediction = Prediction(mean=np.zeros_like(y), variance=np.ones_like(y))
    with pytest.raises(ValueError, match="must provide log_prob and cdf"):
        predictive_metrics(prediction, y)


def test_coverage_uses_forecast_cdf_not_gaussian_moment_quantiles():
    # A non-Gaussian forecast may have the same first two moments as a Gaussian but
    # different quantiles. Uniform PIT values encode exact central coverage even
    # though this deliberately arbitrary mean/variance pair would not.
    n = 10_000
    pit = ((np.arange(n) + .5) / n)[:, None]
    y = np.full((n, 1), 50.0)
    prediction = Prediction(
        mean=np.zeros_like(y),
        variance=np.ones_like(y),
        cdf=pit,
        log_prob=np.zeros_like(y),
    )
    metrics = predictive_metrics(prediction, y)
    assert metrics["coverage_error_90"] < 1e-12


def test_null_graph_is_not_assigned_fake_auroc():
    truth = np.zeros((4, 4))
    score = np.ones((4, 4))
    result = graph_metrics(truth, score)
    assert np.isnan(result["auroc"])
    assert result["null_max_score"] == 1.0


def test_graph_precision_at_k_is_tie_aware_and_reports_ap_lift():
    truth = np.asarray([[0.0, 1.0], [0.0, 0.0]])
    score = np.ones((2, 2))
    result = graph_metrics(truth, score, exclude_diagonal=False)
    assert result["precision_at_true_k"] == .25
    assert np.isclose(result["auprc_lift_over_prevalence"], 0.0)
