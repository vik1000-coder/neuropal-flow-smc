from __future__ import annotations

import numpy as np
import torch

from conditional_neural_benchmark.data import (
    Cohort,
    FoldScaler,
    StimulusSchedule,
    _stimulus_features,
    choose_evaluation_indices,
    make_windows,
)
from conditional_neural_benchmark.metrics import energy_score_rows, summarize_metrics
from conditional_neural_benchmark.models import build_encoded_model


def _schedule(
    worm_id="s:w", *, fps=4.0,
    intervals=((2.0, 3.0), (4.0, 5.0), (6.0, 7.0)),
    order=(2, 1, 3),
):
    names = ("butanone", "pentanedione", "nacl")
    return StimulusSchedule(
        worm_id=worm_id,
        strain="s",
        source_recording="head.mat",
        native_fps=fps,
        analysis_fps=4.0,
        stimulus_names=names,
        event_intervals_seconds=intervals,
        chemical_code_by_event=order,
        chemical_name_by_event=tuple(names[value - 1] for value in order),
        resampling_provenance="none_native_grid",
    )


def test_windows_align_target_and_binary_stimulus():
    trace = np.arange(60, dtype=np.float32).reshape(20, 3)
    cohort = Cohort(
        (trace,), ("s:w",), ("s",), ("a", "b", "c"), 4.0, 1.0,
        stimulus_schedules=(_schedule(),),
    )
    scaler = FoldScaler(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    windows = make_windows(cohort, [0], lag=4, scaler=scaler)
    assert windows.history.shape == (16, 4, 4)
    np.testing.assert_array_equal(windows.history[0, :, :-1], trace[:4])
    np.testing.assert_array_equal(windows.target[0], trace[4])
    assert set(np.unique(windows.history[..., -1])) <= {0.0, 1.0}


def test_chemical_encodings_use_raw_one_based_codes_and_are_active_only():
    schedule = _schedule(
        intervals=((60.5, 70.5), (120.5, 130.5), (180.5, 190.5))
    )
    scalar = _stimulus_features(800, schedule, "chemical_scalar")
    onehot = _stimulus_features(800, schedule, "chemical_onehot")
    assert scalar.shape == (800, 1)
    assert onehot.shape == (800, 3)
    assert scalar[60 * 4, 0] == 0.0
    assert scalar[61 * 4, 0] == 2.0
    assert scalar[121 * 4, 0] == 1.0
    assert scalar[181 * 4, 0] == 3.0
    np.testing.assert_array_equal(onehot[61 * 4], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(onehot[121 * 4], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(onehot[181 * 4], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(onehot[191 * 4], [0.0, 0.0, 0.0])


def test_onehot_windows_add_three_channels_and_preserve_strata():
    trace = np.arange(2400, dtype=np.float32).reshape(800, 3)
    schedule = _schedule(
        intervals=((60.5, 70.5), (120.5, 130.5), (180.5, 190.5))
    )
    cohort = Cohort(
        (trace,), ("s:w",), ("s",), ("a", "b", "c"), 4.0, 1.0,
        stimulus_schedules=(schedule,),
    )
    scaler = FoldScaler(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    binary = make_windows(cohort, [0], lag=4, scaler=scaler)
    onehot = make_windows(
        cohort, [0], lag=4, scaler=scaler,
        stimulus_encoding="chemical_onehot",
    )
    assert onehot.history.shape == (796, 4, 6)
    np.testing.assert_array_equal(binary.stratum, onehot.stratum)
    np.testing.assert_array_equal(onehot.history[244, -1, -3:], [0.0, 1.0, 0.0])


def test_shuffled_chemical_encoding_preserves_composition():
    schedule = _schedule()
    shuffled = _stimulus_features(
        40, schedule, "chemical_onehot_subject_shuffle", (3, 2, 1)
    )
    active_labels = np.flatnonzero(shuffled.sum(axis=0) > 0)
    np.testing.assert_array_equal(active_labels, [0, 1, 2])
    assert np.all(shuffled.sum(axis=1) <= 1)


def test_energy_score_prefers_exact_samples():
    target = np.zeros((8, 2), dtype=np.float32)
    exact = np.zeros((8, 4, 2), dtype=np.float32)
    far = np.ones((8, 4, 2), dtype=np.float32) * 5
    assert energy_score_rows(exact, target).mean() < energy_score_rows(far, target).mean()


def test_all_temporal_encoders_and_gaussian_head():
    for encoder in ("flat", "gru", "tcn", "transformer"):
        model = build_encoded_model(
            head_name="heteroscedastic_gaussian",
            encoder_name=encoder,
            lag=4,
            channels=6,
            dy=5,
            width=16,
        )
        history = torch.randn(3, 24)
        y = torch.randn(3, 5)
        loss = model.native_loss(history, y)
        assert torch.isfinite(loss)
        assert model.sample(history, 2).shape == (3, 2, 5)


def test_evaluation_keeps_all_rare_transitions_and_reweights_natural_score():
    strata = np.asarray(["off"] * 100 + ["onset"] * 4 + ["offset"] * 3)
    idx = choose_evaluation_indices(strata, max_rows=20, seed=4)
    assert set(np.flatnonzero(strata != "off")) <= set(idx.tolist())
    eval_strata = strata[idx]
    values = np.where(eval_strata == "off", 1.0, 9.0)
    result = summarize_metrics(
        {"energy": values}, eval_strata, None, n_dim=1, population_strata=strata
    )
    expected = (100 * 1.0 + 7 * 9.0) / 107
    assert np.isclose(result["energy"], expected)
    assert np.isclose(result["energy__stim_balanced"], (1.0 + 9.0 + 9.0) / 3)
