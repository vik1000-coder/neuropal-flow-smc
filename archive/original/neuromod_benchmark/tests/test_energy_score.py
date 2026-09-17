import numpy as np

from neuromod_benchmark.metrics import energy_score, ensemble_crps, pit_serial_metrics


def test_fair_ensemble_scores_do_not_include_diagonal_pairs():
    y = np.array([[0.0]])
    samples = np.array([[[-1.0], [1.0]]])
    # First term is 1, off-diagonal pair expectation is 2, so score is zero.
    assert energy_score(y, samples) == 0.0
    assert ensemble_crps(y, samples) == 0.0


def test_pit_serial_products_never_cross_episode_boundaries():
    pit = np.array([0.1, 0.9, 0.1, 0.9])
    groups = np.array([0, 0, 1, 1])
    result = pit_serial_metrics(pit, groups, max_lag=1)
    assert result["pit_centered_product_lag1"] < 0
