from __future__ import annotations

import numpy as np

from conditional_neural_benchmark.data import load_cohort, load_sbtg_cohort


def test_frozen_sbtg_cohort_and_current_neuron_bridge() -> None:
    current = load_cohort()
    original = load_sbtg_cohort()
    bridge = load_sbtg_cohort(neuron_subset=current.neurons)

    assert original.n_worms == 20
    assert original.n_neurons == 80
    assert bridge.n_worms == original.n_worms
    assert bridge.neurons == current.neurons
    assert bridge.n_neurons == current.n_neurons == 54
    assert all(trace.shape[1] == 80 for trace in original.traces)
    assert all(trace.shape[1] == 54 for trace in bridge.traces)
    assert original.raw_nonfinite_values > 0
    assert np.isfinite(original.traces[0][8:-8]).all()
