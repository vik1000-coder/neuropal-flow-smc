from __future__ import annotations

import numpy as np

from conditional_neural_benchmark.higher_order_postfreeze import (
    _matched_source_statistic,
    _signed_max,
    matched_enrichment,
)


def test_signed_max_keeps_sign_at_largest_absolute_lag() -> None:
    kernel = np.zeros((3, 2, 2), dtype=float)
    kernel[0, 0, 1] = 1.0
    kernel[2, 0, 1] = -3.0
    assert _signed_max(kernel)[0, 1] == -3.0


def test_matched_receptor_enrichment_detects_constructed_signal() -> None:
    rng = np.random.default_rng(7)
    d = 10
    adjacency = np.zeros((d, d), dtype=int)
    adjacency[[1, 2, 3], 0] = 1
    adjacency[[4, 5, 6], 7] = 1
    score = rng.normal(0.0, 0.01, size=(d, d))
    score[adjacency > 0] += 2.0
    covariates = rng.normal(size=(d, 3))
    observed, sources = _matched_source_statistic(
        score, adjacency, covariates, rng, False
    )
    result = matched_enrichment(
        score, adjacency, covariates, seed=9, permutations=199
    )
    assert sources == 2
    assert observed > 1.0
    assert result["matched_enrichment"] > 1.0
    assert result["permutation_p"] <= 0.05
