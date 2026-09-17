"""Coverage for preprocessing, splitting, history, arrays, config."""
import numpy as np
import pytest

from sid_neuromod.data.preprocessing import detrend_linear, zscore_train
from sid_neuromod.data.splitting import (contiguous_splits, kfold_contiguous,
                                         train_test_split_contiguous)
from sid_neuromod.features.history import (default_timescale_grid, make_target,
                                           source_signal)
from sid_neuromod.utils.arrays import as_2d, safe_log_square, zscore
from sid_neuromod.utils.config import Config, config_hash


def test_zscore_train_uses_train_stats_only():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 2)) * 3 + 5
    Xz, scaler = zscore_train(X, slice(0, 500))
    # train slice standardized to ~0 mean, ~1 std
    assert abs(Xz[:500].mean()) < 0.05
    assert abs(Xz[:500].std() - 1.0) < 0.05
    assert scaler.mean_ is not None


def test_detrend_removes_linear_trend():
    T = 1000
    t = np.arange(T)
    X = (0.01 * t + np.random.default_rng(1).standard_normal(T))[:, None]
    Xd = detrend_linear(X, slice(0, 700))
    # slope of detrended series near 0
    slope = np.polyfit(t, Xd[:, 0], 1)[0]
    assert abs(slope) < 1e-3


def test_contiguous_splits_partition():
    sp = contiguous_splits(1000, 0.5, 0.2, 0.1, 0.2)
    assert sp.train.start == 0
    assert sp.stream.stop == 1000
    # non-overlapping and contiguous
    assert sp.train.stop == sp.calibration.start
    assert sp.calibration.stop == sp.debias.start
    assert sp.debias.stop == sp.stream.start
    d = sp.as_intervals()
    assert d["train"][0] == 0


def test_train_test_and_kfold():
    tr, te = train_test_split_contiguous(100, 0.7)
    assert tr == slice(0, 70) and te == slice(70, 100)
    folds = list(kfold_contiguous(90, k=3))
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_source_signals():
    x = np.array([1.0, 2.0, 0.0, -1.0, 3.0])
    assert np.allclose(source_signal(x, "raw_zscore"), x)
    assert source_signal(x, "delta")[0] == 0.0
    assert np.allclose(source_signal(x, "delta")[1:], np.diff(x))
    assert np.all(source_signal(x, "positive_part") >= 0)
    assert np.all(source_signal(x, "negative_part") <= 0)
    with pytest.raises(ValueError):
        source_signal(x, "nonsense")


def test_make_target_modes():
    X = np.arange(10, dtype=float)[:, None]
    Yn, vn = make_target(X, 1, "next")
    assert np.allclose(Yn[vn].ravel(), X[1:, 0])
    Yd, vd = make_target(X, 1, "delta")
    assert np.allclose(Yd[vd].ravel(), 1.0)
    with pytest.raises(ValueError):
        make_target(X, 0, "next")


def test_default_timescale_grid_monotone():
    grid = default_timescale_grid(median_frame_s=0.25, duration_s=1200.0, n=10)
    assert len(grid) == 10
    assert np.all(np.diff(grid) > 0)
    assert grid[0] >= 0.5


def test_arrays_helpers():
    x = np.array([1.0, 2.0, 3.0])
    assert as_2d(x).shape == (3, 1)
    z, m, s = zscore(x)
    assert abs(z.mean()) < 1e-12
    assert safe_log_square(np.array([0.0]))[0] == -8.0


def test_config_dotaccess_and_hash():
    cfg = Config({"model": {"ridge": 0.5}, "a": 1})
    assert cfg.model.ridge == 0.5
    assert cfg.get_path("model.ridge") == 0.5
    assert cfg.get_path("model.missing", 7) == 7
    h1 = config_hash({"a": 1, "b": 2})
    h2 = config_hash({"b": 2, "a": 1})
    assert h1 == h2  # order-independent
