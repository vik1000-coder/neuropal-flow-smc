from __future__ import annotations

import numpy as np

from conditional_neural_benchmark.data import StimulusSchedule
from conditional_neural_benchmark.latent_calcium_lag import (
    fit_calcium_decay,
    innovation_trace,
    state_for_frames,
)


def _schedule() -> StimulusSchedule:
    return StimulusSchedule(
        worm_id="OH16230:test",
        strain="OH16230",
        source_recording="test.mat",
        native_fps=4.0,
        analysis_fps=4.0,
        stimulus_names=("butanone", "pentanedione", "nacl"),
        event_intervals_seconds=((60.5, 70.5), (120.5, 130.5), (180.5, 190.5)),
        chemical_code_by_event=(2, 1, 3),
        chemical_name_by_event=("pentanedione", "butanone", "nacl"),
        resampling_provenance="none_native_grid",
    )


def test_quiet_decay_fit_and_innovation_reconstruction() -> None:
    rng = np.random.default_rng(7)
    alpha = np.asarray([0.80, 0.93], dtype=np.float32)
    intercept = np.asarray([0.04, -0.02], dtype=np.float32)
    trace = np.zeros((900, 2), dtype=np.float32)
    for index in range(1, len(trace)):
        trace[index] = intercept + alpha * trace[index - 1] + rng.normal(0, 0.03, 2)
    fitted_alpha, fitted_intercept = fit_calcium_decay([trace], [_schedule()])
    np.testing.assert_allclose(fitted_alpha, alpha, atol=0.03)
    innovations = innovation_trace(trace, fitted_alpha, fitted_intercept)
    reconstructed = fitted_intercept + fitted_alpha * trace[:-1] + innovations[1:]
    np.testing.assert_allclose(reconstructed, trace[1:], atol=1e-6)


def test_state_labels_respect_actual_event_intervals() -> None:
    states = state_for_frames(900, _schedule())
    onset = round(60.5 * 4)
    assert states[onset] == "onset"
    assert states[onset + 9] == "active"
    assert states[onset - 1] == "quiet"
    assert states[round(70.5 * 4)] == "quiet"
