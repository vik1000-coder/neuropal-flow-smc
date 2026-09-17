from __future__ import annotations

import numpy as np
import pytest

from compatibility_neural_benchmark.four_sampler_lag_analysis import (
    normalize_and_orient,
)
from compatibility_neural_benchmark.four_sampler_lag_runner import (
    validate_resume_archive,
)
from conditional_neural_benchmark.data import Cohort, StimulusSchedule


def test_normalize_and_orient_maps_source_target_exactly_once():
    response = np.zeros((1, 3, 2, 4, 1, 4), dtype=np.float32)
    gap = np.ones((1, 3, 2, 4), dtype=np.float32)
    response[0, 1, 0, 2, 0, 3] = 8.0
    response[0, 1, 1, 2, 0, 3] = 4.0
    gap[0, 1, :, 2] = 2.0
    oriented = normalize_and_orient(response, gap)
    assert oriented.shape == (3, 1, 1, 4, 4)
    # Event-wise normalized values are 4 and 2; their mean is 3.  Source 2,
    # target 3 must land at matrix[target=3, source=2].
    assert oriented[1, 0, 0, 3, 2] == 3.0
    assert oriented[1, 0, 0, 2, 3] == 0.0


def test_four_sampler_resume_archive_fails_closed_on_stimulus_schema(tmp_path):
    schedules = tuple(
        StimulusSchedule(
            worm_id=f"worm-{index}",
            strain="test",
            source_recording=f"recording-{index}",
            native_fps=4.0,
            analysis_fps=4.0,
            stimulus_names=("butanone", "pentanedione", "nacl"),
            event_intervals_seconds=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),
            chemical_code_by_event=(2, 1, 3),
            chemical_name_by_event=("pentanedione", "butanone", "nacl"),
            resampling_provenance="none_native_grid",
        )
        for index in range(2)
    )
    cohort = Cohort(
        traces=(np.zeros((30, 2)), np.zeros((30, 2))),
        worm_ids=("worm-0", "worm-1"),
        strains=("test", "test"),
        neurons=("A", "B"),
        fps=4.0,
        coverage=1.0,
        stimulus_schedules=schedules,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    import hashlib

    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    archive_path = tmp_path / "archive.npz"
    np.savez_compressed(
        archive_path,
        status=np.asarray("complete"),
        method=np.asarray("progressive_bridge_smc"),
        model_id=np.asarray("model"),
        checkpoint=np.asarray(str(checkpoint.resolve())),
        checkpoint_sha256=np.asarray(checkpoint_hash),
        fold=np.asarray(0),
        seed=np.asarray(17),
        history_frames=np.asarray(8),
        source_lag_frames=np.asarray(4),
        source_window_frames=np.asarray(4),
        n_particles=np.asarray(32),
        horizon_frames=np.asarray([1, 4], dtype=np.int16),
        worm_indices=np.asarray([1], dtype=np.int16),
        worm_ids=np.asarray(["worm-1"]),
        neurons=np.asarray(["A", "B"]),
        stimulus_schema_version=np.asarray(cohort.stimulus_schema_version),
        stimulus_schema_fingerprint=np.asarray(cohort.stimulus_schema_fingerprint),
        chemical_code_by_worm_event=np.asarray([[2, 1, 3]], dtype=np.int8),
        chemical_name_by_worm_event=np.asarray(
            [["pentanedione", "butanone", "nacl"]]
        ),
    )
    kwargs = dict(
        cohort=cohort,
        method="progressive_bridge_smc",
        model_id="model",
        fold=0,
        seed=17,
        source_lag=4,
        particles=32,
        horizons=(1, 4),
        source_window_frames=4,
        history_lag=8,
        checkpoint=checkpoint,
        expected_worm_indices=np.asarray([1]),
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        validate_resume_archive(archive, **kwargs)

    altered = Cohort(
        **{
            **cohort.__dict__,
            "stimulus_schema_version": "changed-schema",
        }
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        with pytest.raises(RuntimeError, match="stimulus_schema_version"):
            validate_resume_archive(archive, **{**kwargs, "cohort": altered})
