from __future__ import annotations

import numpy as np

from compatibility_neural_benchmark.onset_aware_analysis import (
    _column_l2,
    _pattern_metrics,
    leave_one_worm_residual,
)


def test_leave_one_worm_residual_uses_only_other_worms():
    values = np.asarray([[[1.0]], [[3.0]], [[8.0]]])
    residual = leave_one_worm_residual(values, np.arange(3))
    np.testing.assert_allclose(residual[:, 0, 0], [-4.5, -1.5, 6.0])


def test_column_l2_normalizes_nonzero_source_columns():
    matrix = np.asarray([[3.0, 0.0], [4.0, 0.0]])
    normalized = _column_l2(matrix)
    np.testing.assert_allclose(np.linalg.norm(normalized[:, 0]), 1.0)
    np.testing.assert_array_equal(normalized[:, 1], 0.0)


def test_pattern_metrics_recognize_identical_rank_pattern():
    values = np.asarray([-1.0, 0.0, 2.0, 4.0])
    result = _pattern_metrics(values, values)
    assert np.isclose(result["pearson"], 1.0)
    assert np.isclose(result["spearman"], 1.0)
    assert np.isclose(result["cosine"], 1.0)
