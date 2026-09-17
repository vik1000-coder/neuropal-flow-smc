from __future__ import annotations

import numpy as np

from compatibility_neural_benchmark.biological_lag_analysis import (
    PRIMARY_CLASS,
    annotation_table,
    benjamini_hochberg,
    event_aligned_response_tables,
)
from compatibility_neural_benchmark.core import STIMULUS_PERIODS_SECONDS


def test_benjamini_hochberg_preserves_ordered_adjustment():
    p_values = np.asarray([0.01, 0.04, 0.03, np.nan])
    adjusted = benjamini_hochberg(p_values)
    np.testing.assert_allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_primary_annotation_covers_frozen_shared_neuron_set():
    neurons = tuple(sorted(set().union(*PRIMARY_CLASS.values())))
    frame = annotation_table(neurons)
    assert len(frame) == 54
    assert set(frame.neuron) == set(neurons)
    assert frame.primary_class.notna().all()


def test_event_alignment_uses_local_baseline_and_quiet_control():
    fps = 4.0
    n_frames = int((STIMULUS_PERIODS_SECONDS[-1][1] + 15) * fps)
    traces = []
    amplitudes = np.linspace(1.0, 2.0, 12)
    for amplitude in amplitudes:
        trace = np.zeros((n_frames, 1), dtype=float)
        for start, _ in STIMULUS_PERIODS_SECONDS:
            onset = int(round(start * fps))
            trace[onset : onset + 5, 0] = amplitude
        traces.append(trace)
    aligned, latency = event_aligned_response_tables(
        tuple(traces), ("AWC",), fps, maximum_seconds=1.0
    )
    averaged = aligned[(aligned.scope == "event_average") & (aligned.neuron == "AWC")]
    np.testing.assert_allclose(averaged.onset_minus_quiet, np.mean(amplitudes))
    assert latency.iloc[0].latency_seconds == 0.0
    assert latency.iloc[0].latency_direction == "increase"
