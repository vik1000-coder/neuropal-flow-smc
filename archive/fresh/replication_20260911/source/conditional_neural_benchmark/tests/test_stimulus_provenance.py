from __future__ import annotations

import numpy as np
import pytest

from compatibility_neural_benchmark.core import stimulus_for_trace
from conditional_neural_benchmark.data import (
    STIMULUS_NAMES,
    _stimulus_features,
    load_cohort,
)
from conditional_neural_benchmark.chemical_encoding_runner import (
    validate_resume_manifest,
)
from sid_elegans import combined_data


def test_raw_names_codes_known_worm_and_position_mismatch_count():
    cohort = load_cohort(cohort_mode="pooled_resampled")
    assert cohort.n_worms == 20
    assert cohort.n_neurons == 54
    assert all(schedule.stimulus_names == STIMULUS_NAMES for schedule in cohort.stimulus_schedules)
    assert all(
        tuple(sorted(schedule.chemical_code_by_event)) == (1, 2, 3)
        for schedule in cohort.stimulus_schedules
    )
    known = cohort.stimulus_schedules[cohort.worm_ids.index("OH16230:0924_01")]
    assert known.chemical_code_by_event == (2, 1, 3)
    assert known.chemical_name_by_event[0] == "pentanedione"
    matched = sum(
        code == event + 1
        for schedule in cohort.stimulus_schedules
        for event, code in enumerate(schedule.chemical_code_by_event)
    )
    assert matched == 24


def test_known_worm_first_event_activates_pentanedione_not_butanone():
    cohort = load_cohort(cohort_mode="oh16230_head")
    worm = cohort.worm_ids.index("OH16230:0924_01")
    schedule = cohort.stimulus_schedules[worm]
    features = _stimulus_features(
        len(cohort.traces[worm]), schedule, "chemical_onehot"
    )
    onset = round(60.5 * cohort.fps)
    np.testing.assert_array_equal(features[onset], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(features[onset - 1], [0.0, 0.0, 0.0])


def test_oh15500_is_resampled_and_keeps_twenty_second_final_event():
    cohort = load_cohort(cohort_mode="pooled_resampled")
    positions = [
        index for index, schedule in enumerate(cohort.stimulus_schedules)
        if schedule.strain == "OH15500"
    ]
    assert len(positions) == 3
    for index in positions:
        schedule = cohort.stimulus_schedules[index]
        assert schedule.native_fps == 4.1
        assert schedule.analysis_fps == 4.0
        assert schedule.event_intervals_seconds[2] == (180.5, 200.5)
        assert "4.1_to_4" in schedule.resampling_provenance
        active = _stimulus_features(
            len(cohort.traces[index]), schedule, "binary_any_stimulus"
        )[:, 0]
        assert active[round(200.5 * 4.0) - 1] == 1.0
        assert active[round(200.5 * 4.0)] == 0.0


def test_head_tail_merge_fails_without_explicit_simultaneous_pairing():
    head = combined_data._prep.load_neuropal_data(
        combined_data.REPO / "data", include_tail=False
    )
    tail = combined_data._prep.load_tail_data(combined_data.REPO / "data")
    with pytest.raises(RuntimeError, match="pairing table"):
        combined_data._prep.merge_head_tail_data(head, tail)


def test_checkpoint_schema_mismatch_is_rejected():
    cohort = load_cohort(cohort_mode="oh16230_head")
    checkpoint = {
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": "wrong",
        "trial_metadata": {"stimulus_encoding": "chemical_onehot"},
    }
    with pytest.raises(RuntimeError, match="fingerprint"):
        stimulus_for_trace(cohort.traces[0], cohort, 0, checkpoint)


def test_resume_rejects_stimulus_schema_fingerprint_change(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"stimulus_schema_fingerprint":"old","fold_assignments_sha256":"fold"}\n'
    )
    with pytest.raises(RuntimeError, match="stimulus-schema"):
        validate_resume_manifest(manifest, "new", "fold")
