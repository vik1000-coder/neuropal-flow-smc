from compatibility_neural_benchmark.aligned_lag_response_runner import aligned_cuts
from compatibility_neural_benchmark.aligned_lag_analysis import reindex_events_by_chemical
from conditional_neural_benchmark.data import StimulusSchedule
import numpy as np


def test_aligned_cuts_shift_onset_and_matched_quiet_equally():
    schedule = StimulusSchedule(
        worm_id="OH16230:0924_01", strain="OH16230", source_recording="head.mat",
        native_fps=4.0, analysis_fps=4.0,
        stimulus_names=("butanone", "pentanedione", "nacl"),
        event_intervals_seconds=((60.5, 70.5), (120.5, 130.5), (180.5, 190.5)),
        chemical_code_by_event=(2, 1, 3),
        chemical_name_by_event=("pentanedione", "butanone", "nacl"),
        resampling_provenance="none_native_grid",
    )
    cuts = aligned_cuts(1000, schedule, 7)
    quiet = [cut for phase, event, cut, _, _ in cuts if phase == "quiet" and event == 0][0]
    onset = [cut for phase, event, cut, _, _ in cuts if phase == "onset_aligned" and event == 0][0]
    assert onset == round(60.5 * 4.0) + 7
    assert onset - quiet == 60


def test_event_archives_are_reindexed_by_true_chemical_code():
    # worm, phase, event, value
    array = np.asarray([[[[20], [10], [30]]]], dtype=np.float32)
    reordered = reindex_events_by_chemical(array, np.asarray([[2, 1, 3]]))
    np.testing.assert_array_equal(reordered[0, 0, :, 0], [10, 20, 30])
