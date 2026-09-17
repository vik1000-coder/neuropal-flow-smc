from __future__ import annotations

import numpy as np

from compatibility_neural_benchmark.prediction_atlas_analysis import _bootstrap_counts
from compatibility_neural_benchmark.sbtg80_full_progressive_atlas import (
    CHANNELS,
    CONTEXTS,
    HORIZONS,
    SOURCE_LAGS,
    TAIL_ORIGIN_NEURONS,
    _head_tail_rows,
    _origin_masks,
)


def test_full_grid_and_historical_tail_origin_panel_are_frozen() -> None:
    assert SOURCE_LAGS == (1, 4, 8, 16)
    assert HORIZONS == (1, 2, 4, 8, 16, 32)
    assert len(CHANNELS) == 7
    assert len(CONTEXTS) == 13
    assert len(TAIL_ORIGIN_NEURONS) == 17
    assert len(set(TAIL_ORIGIN_NEURONS)) == 17


def test_origin_masks_use_target_rows_source_columns_and_exclude_self_edges() -> None:
    neurons = ("AWB", "AWA", "DVA", "PHA")
    masks = _origin_masks(neurons)
    assert masks[("head", "tail")][2, 0]
    assert not masks[("head", "tail")][0, 2]
    assert masks[("tail", "head")][0, 2]
    assert not masks[("tail", "head")][2, 0]
    assert masks[("head", "head")].sum() == 2
    assert masks[("tail", "tail")].sum() == 2
    assert not any(np.diag(mask).any() for mask in masks.values())


def test_head_tail_intervals_resample_whole_worms() -> None:
    neurons = ("AWB", "AWA", "DVA", "PHA")
    values = np.zeros((4, len(HORIZONS), 4, 4), dtype=np.float32)
    for worm in range(4):
        values[worm] = float(worm + 1)
    rows = _head_tail_rows(
        channel="endpoint_mean",
        context="state_average",
        lag=1,
        normalized=values,
        support=np.ones(4, dtype=np.float32),
        neurons=neurons,
        bootstrap_weights=_bootstrap_counts(4, 64, 17),
    )
    primary = next(
        row
        for row in rows
        if row["horizon_frames"] == 1
        and row["source_origin"] == "head"
        and row["target_origin"] == "tail"
        and row["scope"] == "all_sources"
    )
    assert primary["n_directed_pairs"] == 4
    assert primary["mean_effect"] == 2.5
    assert primary["mean_effect_ci_low"] < primary["mean_effect"]
    assert primary["mean_effect_ci_high"] > primary["mean_effect"]
    assert primary["interval_unit"] == "whole_worm_percentile_bootstrap"


def test_support_scope_removes_only_columns_for_unsupported_sources() -> None:
    neurons = ("AWB", "AWA", "DVA", "PHA")
    rows = _head_tail_rows(
        channel="endpoint_mean",
        context="state_average",
        lag=1,
        normalized=np.ones((4, len(HORIZONS), 4, 4), dtype=np.float32),
        support=np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        neurons=neurons,
        bootstrap_weights=_bootstrap_counts(4, 32, 9),
    )
    head_tail = {
        row["scope"]: row
        for row in rows
        if row["horizon_frames"] == 1
        and row["source_origin"] == "head"
        and row["target_origin"] == "tail"
    }
    assert head_tail["all_sources"]["n_directed_pairs"] == 4
    assert head_tail["support_qualified_sources"]["n_directed_pairs"] == 2

