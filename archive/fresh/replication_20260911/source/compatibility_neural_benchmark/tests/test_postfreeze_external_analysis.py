from __future__ import annotations

import numpy as np

from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    relationship_rows,
    source_macro_metrics,
)


def test_binary_metrics_rank_absolute_signed_scores():
    score = np.asarray([[0.0, -4.0], [0.2, 0.0]])
    labels = np.asarray([[0, 1], [0, 0]])
    mask = ~np.eye(2, dtype=bool)
    metric = binary_metrics(score, labels, mask)
    assert metric["auroc"] == 1.0
    assert metric["auprc"] == 1.0


def test_source_macro_skips_columns_without_both_classes():
    score = np.asarray(
        [
            [0.0, 0.2, 0.1],
            [0.5, 0.0, 0.2],
            [0.1, 0.7, 0.0],
        ]
    )
    labels = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    mask = ~np.eye(3, dtype=bool)
    result = source_macro_metrics(score, labels, mask)
    assert result["n_evaluable_sources"] == 2


def test_relationships_use_only_common_lags_and_offdiagonal_entries():
    a = np.asarray([[[9.0, 1.0], [2.0, 9.0]], [[8.0, 3.0], [4.0, 8.0]]])
    b = np.asarray([[[7.0, 2.0], [4.0, 7.0]]])
    methods = {
        "a": {"lags": np.asarray([1, 2]), "signed": a},
        "b": {"lags": np.asarray([2]), "signed": b},
    }
    result = relationship_rows(methods)
    assert result.lag_frames.tolist() == [2]
    assert result.n_edges.tolist() == [2]
