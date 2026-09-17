from __future__ import annotations

import json

import numpy as np
import pytest

from compatibility_neural_benchmark.prediction_atlas_runner import (
    ARCHIVE_SCHEMA_VERSION,
    DISTRIBUTION_SCALE_FLOOR,
    EPISODE_SEED_DEFINITION,
    COMMON_NOISE_DEFINITION,
    FROZEN_DEFAULT_METHOD,
    FROZEN_PROGRESSIVE_FUTURE_BRANCH_FACTOR,
    FROZEN_PROGRESSIVE_PARTICLES,
    GENERATOR_ENCODING,
    PHASES,
    RESPONSE_AXES,
    RESPONSE_KEYS,
    _validate_checkpoint,
    _write_or_validate_manifest,
    argument_parser,
    canonical_fingerprint,
    expected_episode_metadata,
    sha256,
    source_manifest_stimulus_fingerprint,
    timing_metadata,
    validate_resume_archive,
)
from conditional_neural_benchmark.data import Cohort, StimulusSchedule


def _cohort() -> Cohort:
    schedules = tuple(
        StimulusSchedule(
            worm_id=f"worm-{index}",
            strain="test",
            source_recording=f"recording-{index}",
            native_fps=4.0,
            analysis_fps=4.0,
            stimulus_names=("butanone", "pentanedione", "nacl"),
            event_intervals_seconds=((60.5, 70.5), (120.5, 130.5), (180.5, 190.5)),
            chemical_code_by_event=(2, 1, 3),
            chemical_name_by_event=("pentanedione", "butanone", "nacl"),
            resampling_provenance="none_native_grid",
        )
        for index in range(2)
    )
    return Cohort(
        traces=(np.zeros((1000, 2)), np.zeros((1000, 2))),
        worm_ids=("worm-0", "worm-1"),
        strains=("test", "test"),
        neurons=("A", "B"),
        fps=4.0,
        coverage=1.0,
        stimulus_schedules=schedules,
    )


def _archive_payload(cohort: Cohort, checkpoint, *, source_lag: int = 4):
    worm_indices = np.asarray([1], dtype=np.int16)
    horizons = (1, 4)
    cut_times, bounds = expected_episode_metadata(
        cohort, worm_indices, source_lag, source_window_frames=4
    )
    timing = timing_metadata(source_lag, horizons, cohort.fps)
    response_shape = (1, len(PHASES), 3, 2, len(horizons), 2)
    payload = {
        "status": np.asarray("complete"),
        "archive_schema_version": np.asarray(ARCHIVE_SCHEMA_VERSION),
        "method": np.asarray("progressive_bridge_smc"),
        "model_id": np.asarray("model"),
        "checkpoint": np.asarray(str(checkpoint.resolve())),
        "checkpoint_sha256": np.asarray(sha256(checkpoint)),
        "fold": np.asarray(0),
        "seed": np.asarray(17),
        "base_seed": np.asarray(20260829),
        "episode_seed_definition": np.asarray(EPISODE_SEED_DEFINITION),
        "common_noise_definition": np.asarray(COMMON_NOISE_DEFINITION),
        "requested_device": np.asarray("mps"),
        "resolved_device": np.asarray("mps"),
        "history_frames": np.asarray(8),
        "repair_frames": np.asarray(source_lag + 4),
        "source_lag_frames": np.asarray(source_lag),
        "source_lag_seconds": np.asarray(timing["source_lag_seconds"]),
        "source_window_frames": np.asarray(4),
        "source_window_bounds_semantics": np.asarray(
            "[inclusive_start,exclusive_stop)"
        ),
        "n_particles": np.asarray(32),
        "horizon_frames": timing["horizon_frames"],
        "horizon_seconds": timing["horizon_seconds"],
        "source_to_readout_seconds": timing["source_to_readout_seconds"],
        "lag_definition": np.asarray(timing["lag_definition"]),
        "fps": np.asarray(4.0),
        "phase_names": np.asarray(PHASES),
        "response_keys": np.asarray(RESPONSE_KEYS),
        "response_axes": np.asarray(RESPONSE_AXES),
        "worm_indices": worm_indices,
        "worm_ids": np.asarray(["worm-1"]),
        "neurons": np.asarray(["A", "B"]),
        "cut_times": cut_times,
        "source_window_bounds": bounds,
        "stimulus_schema_version": np.asarray(cohort.stimulus_schema_version),
        "stimulus_schema_fingerprint": np.asarray(
            cohort.stimulus_schema_fingerprint
        ),
        "stimulus_generator_encoding": np.asarray(GENERATOR_ENCODING),
        "chemical_identity_conditioned": np.asarray(False),
        "chemical_code_by_worm_event": np.asarray([[2, 1, 3]], dtype=np.int8),
        "chemical_name_by_worm_event": np.asarray(
            [["pentanedione", "butanone", "nacl"]]
        ),
        "distribution_scale_floor": np.asarray(DISTRIBUTION_SCALE_FLOOR),
    }
    payload.update(
        {key: np.zeros(response_shape, dtype=np.float32) for key in RESPONSE_KEYS}
    )
    diagnostic_shape = (1, len(PHASES), 3, 2)
    payload.update(
        {
            "diagnostic_achieved_low": np.zeros(diagnostic_shape, dtype=np.float32),
            "diagnostic_achieved_high": np.ones(diagnostic_shape, dtype=np.float32),
            "diagnostic_achieved_gap": np.ones(diagnostic_shape, dtype=np.float32),
            "diagnostic_target_gap": np.ones(diagnostic_shape, dtype=np.float32),
            "diagnostic_ess_low": np.full(diagnostic_shape, 10.0, dtype=np.float32),
            "diagnostic_ess_high": np.full(diagnostic_shape, 10.0, dtype=np.float32),
            "diagnostic_max_weight_low": np.full(
                diagnostic_shape, 0.1, dtype=np.float32
            ),
            "diagnostic_max_weight_high": np.full(
                diagnostic_shape, 0.1, dtype=np.float32
            ),
            "diagnostic_valid": np.ones(diagnostic_shape, dtype=np.float32),
            "diagnostic_endpoint_sd_floor": np.full(
                diagnostic_shape, DISTRIBUTION_SCALE_FLOOR, dtype=np.float32
            ),
            "diagnostic_distinct_ancestors_low": np.full(
                diagnostic_shape, 10.0, dtype=np.float32
            ),
            "diagnostic_distinct_ancestors_high": np.full(
                diagnostic_shape, 10.0, dtype=np.float32
            ),
            "diagnostic_step_forced_tempering_low": np.zeros(
                (*diagnostic_shape, source_lag + 4), dtype=np.float32
            ),
            "diagnostic_step_forced_tempering_high": np.zeros(
                (*diagnostic_shape, source_lag + 4), dtype=np.float32
            ),
        }
    )
    return payload


def _validation_kwargs(cohort: Cohort, checkpoint):
    return {
        "cohort": cohort,
        "method": "progressive_bridge_smc",
        "model_id": "model",
        "fold": 0,
        "seed": 17,
        "source_lag": 4,
        "particles": 32,
        "horizons": (1, 4),
        "source_window_frames": 4,
        "history_lag": 8,
        "base_seed": 20260829,
        "requested_device": "mps",
        "checkpoint": checkpoint,
        "expected_worm_indices": np.asarray([1]),
    }


def test_episode_metadata_contains_all_five_phases_and_exact_timing():
    cohort = _cohort()
    cuts, bounds = expected_episode_metadata(
        cohort, np.asarray([1]), source_lag=4, source_window_frames=4
    )
    assert PHASES == ("baseline", "onset", "active", "offset", "recovery")
    # First event: onset=round(60.5*4)=242 and offset=round(70.5*4)=282.
    np.testing.assert_array_equal(cuts[0, :, 0], [182, 245, 262, 285, 302])
    # At onset cut 245 and ell=4, the source window is frames 238..241.
    np.testing.assert_array_equal(bounds[0, PHASES.index("onset"), 0], [238, 242])
    timing = timing_metadata(4, (1, 4), cohort.fps)
    assert timing["source_lag_seconds"] == 1.0
    np.testing.assert_allclose(timing["horizon_seconds"], [0.25, 1.0])
    np.testing.assert_allclose(timing["source_to_readout_seconds"], [1.25, 2.0])


def test_cli_defaults_match_frozen_primary_progressive_protocol():
    args = argument_parser().parse_args(
        ["--output-dir", "output", "--source-run", "source"]
    )
    assert args.methods == [FROZEN_DEFAULT_METHOD] == ["progressive_bridge_smc"]
    assert args.particles == FROZEN_PROGRESSIVE_PARTICLES == 32
    assert (
        args.progressive_future_branch_factor
        == FROZEN_PROGRESSIVE_FUTURE_BRANCH_FACTOR
        == 1
    )


def test_source_manifest_fingerprint_accepts_nested_and_rejects_conflict():
    assert (
        source_manifest_stimulus_fingerprint(
            {"stimulus_schema": {"fingerprint": "nested-value"}}
        )
        == "nested-value"
    )
    assert (
        source_manifest_stimulus_fingerprint(
            {
                "stimulus_schema_fingerprint": "same-value",
                "stimulus_schema": {"fingerprint": "same-value"},
            }
        )
        == "same-value"
    )
    with pytest.raises(RuntimeError, match="conflicting"):
        source_manifest_stimulus_fingerprint(
            {
                "stimulus_schema_fingerprint": "top",
                "stimulus_schema": {"fingerprint": "nested"},
            }
        )


def test_checkpoint_validation_rejects_mislabeled_stimulus_encoding():
    cohort = _cohort()

    class Adapter:
        neurons = cohort.neurons
        checkpoint = {
            "model_config": {"model_id": "model"},
            "fold": 0,
            "seed": 17,
            "lag": 8,
            "stimulus_channels": 1,
            "trial_metadata": {
                "stimulus_encoding": "event_position_scalar",
                "cohort_mode": cohort.cohort_mode,
            },
            "stimulus_schema_version": cohort.stimulus_schema_version,
            "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        }

    with pytest.raises(RuntimeError, match="stimulus_encoding mismatch"):
        _validate_checkpoint(
            Adapter(),
            cohort=cohort,
            model_id="model",
            fold=0,
            seed=17,
            history_lag=8,
        )


def test_historical_sbtg80_checkpoint_profile_is_explicit_and_fail_closed():
    cohort = _cohort()
    cohort = Cohort(
        traces=cohort.traces,
        worm_ids=cohort.worm_ids,
        strains=cohort.strains,
        neurons=cohort.neurons,
        fps=cohort.fps,
        coverage=cohort.coverage,
        stimulus_schedules=cohort.stimulus_schedules,
        cohort_mode="historical_sbtg_full_traces_imputed",
    )

    class HistoricalAdapter:
        neurons = cohort.neurons
        checkpoint = {
            "model_config": {"model_id": "model"},
            "fold": 0,
            "seed": 17,
            "lag": 8,
        }

    _validate_checkpoint(
        HistoricalAdapter(),
        cohort=cohort,
        model_id="model",
        fold=0,
        seed=17,
        history_lag=8,
        validation_profile="historical_sbtg80",
    )

    HistoricalAdapter.checkpoint["stimulus_channels"] = 2
    with pytest.raises(RuntimeError, match="stimulus_channels mismatch"):
        _validate_checkpoint(
            HistoricalAdapter(),
            cohort=cohort,
            model_id="model",
            fold=0,
            seed=17,
            history_lag=8,
            validation_profile="historical_sbtg80",
        )

    with pytest.raises(RuntimeError, match="requires a historical SBTG cohort"):
        _validate_checkpoint(
            HistoricalAdapter(),
            cohort=_cohort(),
            model_id="model",
            fold=0,
            seed=17,
            history_lag=8,
            validation_profile="historical_sbtg80",
        )


def test_existing_manifest_is_rehashed_before_resume(tmp_path):
    path = tmp_path / "manifest.json"
    spec = {"manifest_schema_version": "test", "particles": 32}
    candidate = {
        "created_utc": "2026-08-29T00:00:00+00:00",
        "run_spec_fingerprint": canonical_fingerprint(spec),
        **spec,
    }
    _write_or_validate_manifest(path, candidate)
    assert _write_or_validate_manifest(path, candidate) == candidate
    tampered = {**candidate, "particles": 64}
    path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="fails its own fingerprint"):
        _write_or_validate_manifest(path, candidate)


def test_prediction_atlas_resume_validation_is_strict(tmp_path):
    cohort = _cohort()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = _archive_payload(cohort, checkpoint)
    archive_path = tmp_path / "archive.npz"
    np.savez_compressed(archive_path, **payload)
    kwargs = _validation_kwargs(cohort, checkpoint)
    with np.load(archive_path, allow_pickle=False) as archive:
        validate_resume_archive(archive, **kwargs)

    with np.load(archive_path, allow_pickle=False) as archive:
        with pytest.raises(RuntimeError, match="base_seed"):
            validate_resume_archive(archive, **{**kwargs, "base_seed": 20260830})

    altered = {**payload, "phase_names": np.asarray(PHASES[::-1])}
    np.savez_compressed(archive_path, **altered)
    with np.load(archive_path, allow_pickle=False) as archive:
        with pytest.raises(RuntimeError, match="phase_names"):
            validate_resume_archive(archive, **kwargs)

    bad_bounds = payload["source_window_bounds"].copy()
    bad_bounds[0, 1, 0, 0] -= 1
    altered = {**payload, "source_window_bounds": bad_bounds}
    np.savez_compressed(archive_path, **altered)
    with np.load(archive_path, allow_pickle=False) as archive:
        with pytest.raises(RuntimeError, match="source_window_bounds"):
            validate_resume_archive(archive, **kwargs)

    nonfinite = payload["response_endpoint_mean"].copy()
    nonfinite[0, 0, 0, 0, 0, 0] = np.nan
    altered = {**payload, "response_endpoint_mean": nonfinite}
    np.savez_compressed(archive_path, **altered)
    with np.load(archive_path, allow_pickle=False) as archive:
        with pytest.raises(RuntimeError, match="nonfinite"):
            validate_resume_archive(archive, **kwargs)

    missing_diagnostic = {
        key: value for key, value in payload.items() if key != "diagnostic_achieved_gap"
    }
    np.savez_compressed(archive_path, **missing_diagnostic)
    with np.load(archive_path, allow_pickle=False) as archive:
        with pytest.raises(RuntimeError, match="diagnostic_achieved_gap"):
            validate_resume_archive(archive, **kwargs)

    nonfinite_diagnostic = payload["diagnostic_ess_low"].copy()
    nonfinite_diagnostic[0, 0, 0, 0] = np.nan
    altered = {**payload, "diagnostic_ess_low": nonfinite_diagnostic}
    np.savez_compressed(archive_path, **altered)
    with np.load(archive_path, allow_pickle=False) as archive:
        with pytest.raises(RuntimeError, match="diagnostic_ess_low.*nonfinite"):
            validate_resume_archive(archive, **kwargs)


def test_resume_rejects_corrupt_optional_support_diagnostic(tmp_path):
    cohort = _cohort()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = _archive_payload(cohort, checkpoint)
    diagnostic_shape = (1, len(PHASES), 3, cohort.n_neurons)
    payload["diagnostic_anchor_cost_mean_low"] = np.full(
        diagnostic_shape, np.nan, dtype=np.float32
    )
    archive_path = tmp_path / "archive.npz"
    np.savez_compressed(archive_path, **payload)
    with np.load(archive_path, allow_pickle=False) as archive:
        with pytest.raises(RuntimeError, match="contains nonfinite values"):
            validate_resume_archive(
                archive, **_validation_kwargs(cohort, checkpoint)
            )
