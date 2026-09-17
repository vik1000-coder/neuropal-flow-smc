import numpy as np
import torch

from conditional_neural_benchmark.data import (
    Windows,
    balance_window_strata,
    resample_window_strata,
)
from conditional_neural_benchmark.models import _covering_dilations, build_encoded_model


def test_covering_dilations_cover_l80_without_changing_legacy_encoder():
    dilations = _covering_dilations(80)
    assert dilations == (1, 2, 4, 8, 16, 32)
    assert 1 + 2 * sum(dilations) >= 80


def test_new_encoders_return_declared_width():
    x = torch.randn(3, 80 * 6)
    for encoder in ("full_history_tcn", "multiscale_tcn", "stimulus_phase_tcn"):
        model = build_encoded_model(
            head_name="conditional_flow_matching", encoder_name=encoder,
            lag=80, channels=6, dy=5, width=16, dropout=0.1,
            head_params={"hidden": 16, "layers": 1, "sample_steps": 4},
        )
        assert model.context(x).shape == (3, 16)


def test_balanced_window_view_is_equal_and_reproducible():
    labels = np.asarray(["off"] * 10 + ["onset"] * 2 + ["mixed"] * 3)
    n = len(labels)
    windows = Windows(
        history=np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2),
        target=np.zeros((n, 1), dtype=np.float32),
        worm=np.zeros(n, dtype=np.int16), time=np.arange(n), stratum=labels,
        chemical_code=np.zeros(n, dtype=np.int8),
        event_position=np.zeros(n, dtype=np.int8),
    )
    first = balance_window_strata(windows, seed=3, total_rows=15)
    second = balance_window_strata(windows, seed=3, total_rows=15)
    assert dict(zip(*np.unique(first.stratum, return_counts=True))) == {
        "mixed": 5, "off": 5, "onset": 5,
    }
    np.testing.assert_array_equal(first.history, second.history)


def test_moderate_stratum_resampling_has_declared_counts():
    labels = np.asarray(["off"] * 10 + ["onset"] * 2 + ["mixed"] * 3)
    n = len(labels)
    windows = Windows(
        history=np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2),
        target=np.zeros((n, 1), dtype=np.float32),
        worm=np.zeros(n, dtype=np.int16), time=np.arange(n), stratum=labels,
        chemical_code=np.zeros(n, dtype=np.int8),
        event_position=np.zeros(n, dtype=np.int8),
    )
    sampled = resample_window_strata(
        windows, proportions={"off": 0.4, "onset": 0.3, "mixed": 0.3},
        seed=7, total_rows=20,
    )
    assert dict(zip(*np.unique(sampled.stratum, return_counts=True))) == {
        "mixed": 6, "off": 8, "onset": 6,
    }
