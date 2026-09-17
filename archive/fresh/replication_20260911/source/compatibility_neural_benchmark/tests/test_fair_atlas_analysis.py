from __future__ import annotations

import numpy as np

from compatibility_neural_benchmark.fair_atlas_analysis import (
    align_square,
    binary_metrics,
    build_reference_adjacency,
)


def test_align_square_reorders_both_target_and_source_axes():
    matrix = np.asarray([[11.0, 12.0], [21.0, 22.0]])
    aligned = align_square(matrix, ["B", "A"], ["A", "B"])
    np.testing.assert_array_equal(aligned, np.asarray([[22.0, 21.0], [12.0, 11.0]]))


def test_edge_list_adjacency_is_target_row_source_column(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text("A,B,dopamine,r1\n")
    adjacency, raw, retained = build_reference_adjacency(path, ["A", "B"])
    assert raw == retained == 1
    assert adjacency[1, 0] == 1
    assert adjacency[0, 1] == 0


def test_binary_metrics_use_absolute_continuous_scores():
    score = np.asarray([[0.0, -3.0], [0.1, 0.0]])
    labels = np.asarray([[0, 1], [0, 0]])
    mask = ~np.eye(2, dtype=bool)
    result = binary_metrics(score, labels, mask)
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0

