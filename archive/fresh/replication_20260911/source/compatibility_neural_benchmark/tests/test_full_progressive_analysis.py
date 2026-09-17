from __future__ import annotations

import numpy as np

from compatibility_neural_benchmark.full_progressive_analysis import count_metrics


def test_count_metrics_separate_presence_from_positive_strength():
    score = np.asarray(
        [
            [0.0, 10.0, 2.0],
            [1.0, 0.0, 3.0],
            [4.0, 5.0, 0.0],
        ]
    )
    weight = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [3.0, 4.0, 0.0],
        ]
    )
    result = count_metrics(score, weight, ~np.eye(3, dtype=bool))
    assert result["n_edges"] == 6
    assert result["n_positive"] == 4
    assert np.isfinite(result["all_pair_spearman"])
    assert np.isfinite(result["positive_edge_spearman"])
    assert 0 <= result["ndcg_all_pairs"] <= 1
