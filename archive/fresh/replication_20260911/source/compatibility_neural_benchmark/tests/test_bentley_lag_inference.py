from __future__ import annotations

import numpy as np

from compatibility_neural_benchmark.bentley_lag_inference import (
    eligible_source_auroc,
    permute_targets_within_source,
)
from compatibility_neural_benchmark.onset_post_analysis import benjamini_hochberg


def test_eligible_source_auroc_preserves_target_source_orientation():
    labels = np.zeros((3, 3), dtype=int)
    labels[1, 0] = 1
    matrix = np.zeros((1, 3, 3), dtype=float)
    matrix[0, 1, 0] = 5.0
    assert eligible_source_auroc(matrix, labels, np.asarray([0]))[0] == 1.0


def test_within_source_permutation_preserves_edge_counts():
    labels = np.asarray([[0, 1, 0], [1, 0, 0], [0, 1, 0]])
    result = permute_targets_within_source(labels, np.random.default_rng(3))
    np.testing.assert_array_equal(result.sum(axis=0), labels.sum(axis=0))


def test_benjamini_hochberg_is_monotone_in_sorted_p_values():
    values = np.asarray([0.04, 0.001, 0.02])
    adjusted = benjamini_hochberg(values)
    ordered = adjusted[np.argsort(values)]
    assert np.all(np.diff(ordered) >= 0)
    assert np.all(adjusted >= values)
