from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.prediction_atlas_analysis import (
    AnalysisConfig,
    CHANNELS,
    METHODS,
    PHASES,
    _append_candidate_lag_profiles,
    _bootstrap_counts,
    _stimulus_composition_frame,
    build_prediction_atlas,
    discover_and_validate_inputs,
    effect_normalization_denominator,
    orient_response_once,
)


def _set_manifest_fingerprint(manifest: dict[str, object]) -> None:
    spec = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_utc", "run_spec_fingerprint"}
    }
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["run_spec_fingerprint"] = hashlib.sha256(payload.encode()).hexdigest()


def _write_synthetic_run(
    root: Path,
    *,
    corrupt_chemical: bool = False,
    unsupported_source: int | None = None,
    opposite_direct: bool = False,
    low_genealogy: bool = False,
) -> Path:
    root.mkdir(parents=True)
    response_root = root / "responses"
    worms = tuple(f"worm-{index}" for index in range(4))
    neurons = ("source-A", "target-B")
    codes_by_worm = (
        (1, 2, 3),
        (2, 1, 3),
        (3, 2, 1),
        (1, 3, 2),
    )
    schedules = []
    for worm, codes in zip(worms, codes_by_worm):
        schedules.append(
            {
                "worm_id": worm,
                "source_recording": f"synthetic-recording-{worm}",
                "native_fps": 4.0,
                "analysis_fps": 4.0,
                "event_intervals_seconds": [
                    [20.0, 30.0],
                    [50.0, 60.0],
                    [80.0, 90.0],
                ],
                "resampling_provenance": "none_native_grid",
                "chemical_code_by_event": list(codes),
                "chemical_name_by_event": [
                    {1: "butanone", 2: "pentanedione", 3: "nacl"}[code]
                    for code in codes
                ],
            }
        )
    fold_file = root / "fold_assignments.csv"
    pd.DataFrame(
        {
            "worm_id": worms,
            "outer_fold": [0, 0, 1, 1],
        }
    ).to_csv(fold_file, index=False)
    fold_hash = hashlib.sha256(fold_file.read_bytes()).hexdigest()
    source_run = root / "source_run"
    source_run.mkdir()
    stimulus_schema = {
        "version": "synthetic-v1",
        "fingerprint": "synthetic-fingerprint",
        "schedules": schedules,
    }
    source_manifest = {
        "status": "complete",
        "cohort_mode": "synthetic",
        "folds": [0, 1],
        "seeds": [11, 29],
        "lag_frames": 8,
        "n_worms": 4,
        "n_neurons": 2,
        "fold_assignments": str(fold_file.resolve()),
        "fold_assignments_sha256": fold_hash,
        "stimulus_schema_version": "synthetic-v1",
        "stimulus_schema_fingerprint": "synthetic-fingerprint",
        "stimulus_schema": stimulus_schema,
        "encodings": [
            {
                "encoding": "binary_any_stimulus",
                "sensitivity_only": False,
                "shuffled": False,
            },
            {
                "encoding": "chemical_onehot",
                "sensitivity_only": False,
                "shuffled": False,
            },
        ],
    }
    source_manifest_path = source_run / "manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest))
    source_manifest_hash = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
    leaderboard_rows = []
    for rank, model_id, encoding, offset in (
        (1, "binary-flow-test", "binary_any_stimulus", 0.0),
        (2, "chemical-flow-test", "chemical_onehot", 0.1),
    ):
        leaderboard_rows.append(
            {
                "rank": rank,
                "model_id": model_id,
                "stimulus_encoding": encoding,
                "energy__mean": 1.0 + offset,
                "energy__std": 0.1,
                "energy__count": 4,
                "energy__stim_balanced__mean": 1.05 + offset,
                "energy__stim_balanced__std": 0.11,
                "energy__stim_balanced__count": 4,
                "energy__worm_chemical_balanced__mean": 1.08 + offset,
                "energy__worm_chemical_balanced__std": 0.12,
                "energy__worm_chemical_balanced__count": 4,
                "energy__chemical_butanone__mean": 1.01 + offset,
                "energy__chemical_butanone__std": 0.13,
                "energy__chemical_butanone__count": 4,
                "energy__chemical_pentanedione__mean": 1.12 + offset,
                "energy__chemical_pentanedione__std": 0.14,
                "energy__chemical_pentanedione__count": 4,
                "energy__chemical_nacl__mean": 1.09 + offset,
                "energy__chemical_nacl__std": 0.15,
                "energy__chemical_nacl__count": 4,
                "variogram__mean": 0.03 + offset / 100,
                "variogram__std": 0.002,
                "variogram__count": 4,
            }
        )
    pd.DataFrame(leaderboard_rows).to_csv(
        source_run / "chemical_leaderboard.csv", index=False
    )
    manifest = {
        "created_utc": "2026-08-29T00:00:00+00:00",
        "manifest_schema_version": "prediction_atlas_manifest_v2",
        "protocol": "synthetic atlas runner",
        "methods": list(METHODS),
        "folds": [0, 1],
        "seeds": [11, 29],
        "source_lag_frames": [1],
        "horizon_frames": [1, 2],
        "phases": list(PHASES),
        "model_id": "binary-flow-test",
        "cohort_mode": "synthetic",
        "n_worms": 4,
        "n_neurons": 2,
        "fps": 4.0,
        "history_frames": 8,
        "source_window_frames": 2,
        "lag_definition": "source-window end to prediction cut",
        "response_keys": [f"response_{channel}" for channel in CHANNELS],
        "particles": 12,
        "response_axes": [
            "heldout_worm",
            "phase",
            "event",
            "source",
            "horizon",
            "target",
        ],
        "matrix_internal_orientation": "source,horizon,target",
        "distribution_scale_floor": 1e-6,
        "minimum_effective_sample_size": 2.0,
        "maximum_normalized_weight": 0.5,
        "minimum_achieved_source_fraction": 0.25,
        "progressive_branch_factor": 2,
        "progressive_future_branch_factor": 2,
        "stimulus_generator_encoding": "binary_any_stimulus",
        "chemical_identity_conditioned": False,
        "base_seed": 20_260_829,
        "episode_seed_definition": (
            "sha256(model_id) prefix + base_seed + 1000003*fold + 1009*generator_seed "
            "+ 9176*worm_index + 131*phase_index + 17*event_index + 53*source_lag; "
            "sampler method deliberately omitted; modulo 2**31-1"
        ),
        "common_noise_definition": (
            "within each estimator, low/high future arms use identical flow base-noise "
            "seeds and aligned particle-row order"
        ),
        "requested_device": "cpu",
        "source_run": str(source_run.resolve()),
        "source_run_manifest_sha256": source_manifest_hash,
        "checkpoint_phase": "synthetic_full_cv",
        "fold_assignments": str(fold_file.resolve()),
        "fold_assignments_sha256": fold_hash,
        "stimulus_schema_version": "synthetic-v1",
        "stimulus_schema_fingerprint": "synthetic-fingerprint",
        "stimulus_schema": stimulus_schema,
    }
    _set_manifest_fingerprint(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    checkpoints: dict[tuple[int, int], tuple[Path, str]] = {}
    for fold in (0, 1):
        for seed in (11, 29):
            checkpoint = (
                source_run
                / "checkpoints"
                / "synthetic_full_cv"
                / f"binary-flow-test__L8__f{fold}__s{seed}.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"checkpoint {fold} {seed}".encode())
            checkpoints[(fold, seed)] = (
                checkpoint,
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
    for method_position, method in enumerate(METHODS):
        method_root = response_root / method
        method_root.mkdir(parents=True)
        for fold in (0, 1):
            heldout = np.asarray([fold * 2, fold * 2 + 1], dtype=np.int16)
            for seed in (11, 29):
                checkpoint, checkpoint_hash = checkpoints[(fold, seed)]
                shape = (len(heldout), 5, 3, 2, 2, 2)
                response = {
                    f"response_{channel}": np.zeros(shape, dtype=np.float32)
                    for channel in CHANNELS
                }
                gap = np.full((len(heldout), 5, 3, 2), 2.0, dtype=np.float32)
                valid = np.ones_like(gap)
                particles = 12
                target_low = np.zeros_like(gap)
                target_high = np.full_like(gap, 2.0)
                achieved_low = np.zeros_like(gap)
                achieved_high = np.full_like(gap, 2.0)
                if method == "direct_importance":
                    ess = np.full_like(gap, 5.0)
                    max_weight = np.full_like(gap, 0.20)
                else:
                    # Candidate ESS=12 with branch factor 2 -> equivalent ESS=6.
                    ess = np.full_like(gap, 6.0)
                    max_weight = np.full_like(gap, 1.0 / 6.0)
                for local, worm_index in enumerate(heldout):
                    for phase in range(5):
                        for event in range(3):
                            for horizon in range(2):
                                effect = (
                                    0.2
                                    + phase * 0.35
                                    + event * 0.07
                                    + horizon * 0.11
                                    + int(worm_index) * 0.03
                                    + (seed == 29) * 0.01
                                    + method_position * 0.02
                                )
                                for channel in CHANNELS:
                                    if channel == "event_probability":
                                        value = effect / 5.0
                                    elif channel == "endpoint_wasserstein1":
                                        value = abs(effect)
                                    else:
                                        value = effect
                                    if (
                                        opposite_direct
                                        and method == "direct_importance"
                                        and channel != "endpoint_wasserstein1"
                                    ):
                                        value = -value
                                    # Sampler orientation is source=0 -> target=1.
                                    response[f"response_{channel}"][
                                        local, phase, event, 0, horizon, 1
                                    ] = gap[local, phase, event, 0] * value
                cut_times = np.empty((len(heldout), 5, 3), dtype=np.int32)
                for local, worm_index in enumerate(heldout):
                    intervals = schedules[int(worm_index)][
                        "event_intervals_seconds"
                    ]
                    for event in range(3):
                        onset = int(round(intervals[event][0] * 4.0))
                        offset = int(round(intervals[event][1] * 4.0))
                        cut_times[local, :, event] = np.asarray(
                            [
                                onset - int(round(15.0 * 4.0)),
                                onset + 1,
                                onset + int(round(5.0 * 4.0)),
                                offset + 1,
                                offset + int(round(5.0 * 4.0)),
                            ],
                            dtype=np.int32,
                        )
                bounds = np.empty(cut_times.shape + (2,), dtype=np.int32)
                bounds[..., 1] = cut_times  # hi - 1 == cut - lag for lag=1
                bounds[..., 0] = bounds[..., 1] - 2
                local_codes = np.asarray(
                    [codes_by_worm[int(index)] for index in heldout], dtype=np.int8
                )
                local_names = np.asarray(
                    [schedules[int(index)]["chemical_name_by_event"] for index in heldout]
                )
                if corrupt_chemical and method == METHODS[0] and fold == 0 and seed == 11:
                    local_codes[0] = np.asarray([1, 1, 3], dtype=np.int8)
                path = method_root / f"{method}__ell1__f{fold}__s{seed}.npz"
                diagnostics = {
                    "diagnostic_target_low": target_low,
                    "diagnostic_target_high": target_high,
                    "diagnostic_target_gap": target_high - target_low,
                    "diagnostic_achieved_low": achieved_low,
                    "diagnostic_achieved_high": achieved_high,
                    "diagnostic_achieved_gap": gap,
                    "diagnostic_ess_low": ess,
                    "diagnostic_ess_high": ess,
                    "diagnostic_max_weight_low": max_weight,
                    "diagnostic_max_weight_high": max_weight,
                    "diagnostic_valid": valid,
                    "diagnostic_endpoint_sd_floor": np.full_like(gap, 1e-6),
                }
                if method == "progressive_bridge_smc":
                    steps = 3
                    step_shape = gap.shape + (steps,)
                    step_candidate_ess = np.full(step_shape, 12.0, dtype=np.float32)
                    step_ess = step_candidate_ess / 2.0
                    step_candidate_max = np.full(
                        step_shape, 1.0 / 12.0, dtype=np.float32
                    )
                    step_max = step_candidate_max * 2.0
                    beta = np.broadcast_to(
                        np.asarray([0.5, 1.0, 1.0], dtype=np.float32), step_shape
                    ).copy()
                    diagnostics.update(
                        {
                            "diagnostic_candidate_ess_low": np.full_like(gap, 12.0),
                            "diagnostic_candidate_ess_high": np.full_like(gap, 12.0),
                            "diagnostic_ess_fraction_low": np.full_like(gap, 0.5),
                            "diagnostic_ess_fraction_high": np.full_like(gap, 0.5),
                            "diagnostic_candidate_max_weight_low": np.full_like(
                                gap, 1.0 / 12.0
                            ),
                            "diagnostic_candidate_max_weight_high": np.full_like(
                                gap, 1.0 / 12.0
                            ),
                            "diagnostic_min_step_ess_low": np.full_like(gap, 6.0),
                            "diagnostic_min_step_ess_high": np.full_like(gap, 6.0),
                            "diagnostic_distinct_ancestors_low": np.full_like(
                                gap, 1.0 if low_genealogy else 8.0
                            ),
                            "diagnostic_distinct_ancestors_high": np.full_like(
                                gap, 1.0 if low_genealogy else 8.0
                            ),
                            "diagnostic_branch_factor": np.full_like(gap, 2.0),
                            "diagnostic_future_branch_factor": np.full_like(gap, 2.0),
                            "diagnostic_candidate_particles": np.full_like(gap, 24.0),
                            "diagnostic_future_particles": np.full_like(gap, 24.0),
                            "diagnostic_source_window_start_step": np.zeros_like(gap),
                            "diagnostic_source_window_end_step_exclusive": np.full_like(
                                gap, 2.0
                            ),
                            "diagnostic_step_ess_low": step_ess,
                            "diagnostic_step_ess_high": step_ess,
                            "diagnostic_step_candidate_ess_low": step_candidate_ess,
                            "diagnostic_step_candidate_ess_high": step_candidate_ess,
                            "diagnostic_step_ess_fraction_low": np.full(
                                step_shape, 0.5, dtype=np.float32
                            ),
                            "diagnostic_step_ess_fraction_high": np.full(
                                step_shape, 0.5, dtype=np.float32
                            ),
                            "diagnostic_step_max_weight_low": step_max,
                            "diagnostic_step_max_weight_high": step_max,
                            "diagnostic_step_candidate_max_weight_low": step_candidate_max,
                            "diagnostic_step_candidate_max_weight_high": step_candidate_max,
                            "diagnostic_step_tempering_resamples_low": np.zeros(
                                step_shape, dtype=np.float32
                            ),
                            "diagnostic_step_tempering_resamples_high": np.zeros(
                                step_shape, dtype=np.float32
                            ),
                            "diagnostic_step_forced_tempering_low": np.zeros(
                                step_shape, dtype=np.float32
                            ),
                            "diagnostic_step_forced_tempering_high": np.zeros(
                                step_shape, dtype=np.float32
                            ),
                            "diagnostic_step_beta_low": beta,
                            "diagnostic_step_beta_high": beta,
                        }
                    )
                if unsupported_source is not None:
                    source = int(unsupported_source)
                    diagnostics["diagnostic_valid"][..., source] = 0.0
                    if method == "direct_importance":
                        diagnostics["diagnostic_ess_low"][..., source] = 1.0
                        diagnostics["diagnostic_ess_high"][..., source] = 1.0
                        diagnostics["diagnostic_max_weight_low"][..., source] = 1.0
                        diagnostics["diagnostic_max_weight_high"][..., source] = 1.0
                    else:
                        for side in ("low", "high"):
                            diagnostics[f"diagnostic_ess_{side}"][..., source] = 1.0
                            diagnostics[f"diagnostic_candidate_ess_{side}"][..., source] = 2.0
                            diagnostics[f"diagnostic_ess_fraction_{side}"][..., source] = 2.0 / 24.0
                            diagnostics[f"diagnostic_max_weight_{side}"][..., source] = 1.0
                            diagnostics[f"diagnostic_candidate_max_weight_{side}"][..., source] = 0.5
                            diagnostics[f"diagnostic_min_step_ess_{side}"][..., source] = 1.0
                            diagnostics[f"diagnostic_step_ess_{side}"][..., source, :] = 1.0
                            diagnostics[f"diagnostic_step_candidate_ess_{side}"][..., source, :] = 2.0
                            diagnostics[f"diagnostic_step_ess_fraction_{side}"][..., source, :] = 2.0 / 24.0
                            diagnostics[f"diagnostic_step_max_weight_{side}"][..., source, :] = 1.0
                            diagnostics[f"diagnostic_step_candidate_max_weight_{side}"][..., source, :] = 0.5
                np.savez_compressed(
                    path,
                    status=np.asarray("complete"),
                    archive_schema_version=np.asarray("prediction_atlas_response_v2"),
                    method=np.asarray(method),
                    model_id=np.asarray("binary-flow-test"),
                    checkpoint=np.asarray(str(checkpoint.resolve())),
                    checkpoint_sha256=np.asarray(checkpoint_hash),
                    fold=np.asarray(fold),
                    seed=np.asarray(seed),
                    base_seed=np.asarray(20_260_829),
                    episode_seed_definition=np.asarray(
                        manifest["episode_seed_definition"]
                    ),
                    common_noise_definition=np.asarray(
                        manifest["common_noise_definition"]
                    ),
                    requested_device=np.asarray("cpu"),
                    resolved_device=np.asarray("cpu"),
                    history_frames=np.asarray(8),
                    source_lag_frames=np.asarray(1),
                    source_lag_seconds=np.asarray(0.25),
                    lag_definition=np.asarray("source-window end to prediction cut"),
                    source_to_readout_seconds=np.asarray([0.5, 0.75]),
                    repair_frames=np.asarray(3),
                    source_window_frames=np.asarray(2),
                    source_window_bounds_semantics=np.asarray(
                        "[inclusive_start,exclusive_stop)"
                    ),
                    n_particles=np.asarray(particles),
                    horizon_frames=np.asarray([1, 2], dtype=np.int16),
                    horizon_seconds=np.asarray([0.25, 0.5]),
                    phase_names=np.asarray(PHASES),
                    fps=np.asarray(4.0),
                    response_keys=np.asarray(
                        [f"response_{channel}" for channel in CHANNELS]
                    ),
                    response_axes=np.asarray(
                        [
                            "heldout_worm",
                            "phase",
                            "event",
                            "source",
                            "horizon",
                            "target",
                        ]
                    ),
                    worm_indices=heldout,
                    worm_ids=np.asarray([worms[int(index)] for index in heldout]),
                    chemical_code_by_worm_event=local_codes,
                    chemical_name_by_worm_event=local_names,
                    stimulus_schema_version=np.asarray("synthetic-v1"),
                    stimulus_schema_fingerprint=np.asarray("synthetic-fingerprint"),
                    stimulus_generator_encoding=np.asarray("binary_any_stimulus"),
                    chemical_identity_conditioned=np.asarray(False),
                    distribution_scale_floor=np.asarray(1e-6),
                    wall_seconds=np.asarray(
                        1.25 + method_position + fold * 0.1 + (seed == 29) * 0.05
                    ),
                    neurons=np.asarray(neurons),
                    cut_times=cut_times,
                    source_window_bounds=bounds,
                    **response,
                    **diagnostics,
                )
    return root


def _rewrite_npz(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    values.update(updates)
    np.savez_compressed(path, **values)


def test_orient_response_once_places_target_rows_and_source_columns():
    response = np.zeros((1, 1, 1, 3, 1, 3), dtype=np.float32)
    response[0, 0, 0, 1, 0, 2] = 9
    result = orient_response_once(response)
    assert result.shape == (1, 1, 1, 1, 3, 3)
    assert result[0, 0, 0, 0, 2, 1] == 9
    assert result[0, 0, 0, 0, 1, 2] == 0


def test_negative_achieved_gap_uses_magnitude_for_w1_normalization():
    denominator = effect_normalization_denominator(
        np.asarray([-0.4, -0.02, 0.3]), 0.10
    )
    np.testing.assert_allclose(denominator, [0.4, 0.10, 0.3])


def test_build_prediction_atlas_writes_reviewed_bundle_and_orientation(tmp_path):
    run = _write_synthetic_run(tmp_path / "run")
    output = tmp_path / "atlas"
    build_prediction_atlas(
        [run],
        output,
        config=AnalysisConfig(
            bootstrap_replicates=32,
            sign_flip_replicates=256,
            prediction_cells_per_slice=2,
            hypothesis_queue_size=20,
            dashboard_candidates=10,
        ),
    )
    expected = {
        "worm_matrices.npz",
        "atlas_matrices.npz",
        "hypothesis_queue.csv",
        "candidate_lag_profiles.csv",
        "stimulus_composition.csv",
        "models.json",
        "protocol.json",
        "validation.json",
        "dashboard_snapshot.json",
        "manifest.json",
        "checksums.sha256",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    with np.load(output / "atlas_matrices.npz", allow_pickle=False) as atlas:
        matrix = atlas[
            "mean_normalized__progressive_bridge_smc__endpoint_mean__onset"
        ]
        assert matrix.shape == (1, 2, 2, 2)
        # The injected source=0,target=1 signal must be row 1, column 0.
        assert matrix[0, 0, 1, 0] > 0
        assert matrix[0, 0, 0, 1] == 0
    validation = json.loads((output / "validation.json").read_text())
    assert validation["status"] == "passed"
    assert all(validation["archive_checks_completed"].values())
    dashboard = json.loads((output / "dashboard_snapshot.json").read_text())
    assert len(dashboard["matrix_rows"]) == 4
    with np.load(output / "atlas_matrices.npz", allow_pickle=False) as atlas:
        dense = atlas[
            "mean_normalized__progressive_bridge_smc__endpoint_mean__onset_minus_baseline"
        ][0, 0]
    for row in dashboard["matrix_rows"]:
        assert row["matrix_orientation"] == "target_row_source_column"
        assert row["row_axis"] == "target_neuron"
        assert row["column_axis"] == "source_neuron"
        assert row["mean_normalized"] == pytest.approx(
            dense[row["target_index"], row["source_index"]]
        )
        assert row["is_diagonal"] == (
            row["target_index"] == row["source_index"]
        )
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["ranking"] == "no connectome or external reference data used"
    assert "not chemically conditioned" in protocol["chemical_context_warning"]
    assert "do not provide full-family FDR control" in protocol["inference"]["post_screen_warning"]
    support = pd.read_parquet(output / "support_cells.parquet")
    assert {
        "mean_ess_low",
        "mean_max_weight_high",
        "mean_distinct_ancestor_fraction_low",
        "forced_tempering_rate_high",
        "declared_branch_factor",
        "declared_future_branch_factor",
        "mean_archive_wall_seconds_method_lag",
        "branch_factor_semantics",
        "genealogy_valid_fraction_0_10",
        "genealogy_valid_fraction_0_20",
        "genealogy_strong_gate_pass",
        "genealogy_sensitivity_gate_pass",
        "mean_min_distinct_ancestor_fraction",
    }.issubset(support.columns)
    direct_support = support[support["method"] == "direct_importance"]
    progressive_support = support[support["method"] == "progressive_bridge_smc"]
    assert set(direct_support["declared_branch_factor"]) == {1}
    assert set(progressive_support["declared_branch_factor"]) == {2}
    assert progressive_support["genealogy_strong_gate_pass"].all()
    assert progressive_support["genealogy_sensitivity_gate_pass"].all()
    assert progressive_support["genealogy_valid_fraction_0_10"].eq(1.0).all()
    assert progressive_support["genealogy_valid_fraction_0_20"].eq(1.0).all()
    assert not direct_support["genealogy_gate_applicable"].any()
    assert (support["mean_archive_wall_seconds_method_lag"] > 0).all()
    assert len(validation["runtime_compute_diagnostics"]) == 2
    models = json.loads((output / "models.json").read_text())
    assert models["frozen_predictive_scores"]["rank"] == 1
    assert models["frozen_predictive_scores"][
        "energy__worm_chemical_balanced__mean"
    ] == pytest.approx(1.08)
    assert models["stimulus_encoding_selection"]["passed"] is False
    assert models["source_run_provenance"]["chemical_leaderboard_sha256"]
    composition = pd.read_csv(output / "stimulus_composition.csv")
    assert len(composition) == 4 * 5 * 3 * 1 * 2
    exact = composition[
        (composition.worm_id == "worm-0")
        & (composition.phase == "onset")
        & (composition.event_index == 0)
        & (composition.source_lag_frames == 1)
        & (composition.horizon_frames == 1)
    ].iloc[0]
    assert exact.source_window_stimulus_fraction == 0.5
    assert exact.cut_stimulus_indicator
    assert exact.forecast_window_stimulus_fraction == 1.0
    assert exact.forecast_endpoint_stimulus_indicator
    assert protocol["stimulus_composition"]["artifact"] == "stimulus_composition.csv"
    assert protocol["support_thresholds"]["genealogy"][
        "hard_strong_min_ancestor_fraction"
    ] == 0.10
    checksum_lines = (output / "checksums.sha256").read_text().strip().splitlines()
    assert checksum_lines
    for line in checksum_lines:
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest


def test_validation_fails_closed_on_bad_chemical_permutation(tmp_path):
    run = _write_synthetic_run(tmp_path / "run", corrupt_chemical=True)
    with pytest.raises(RuntimeError, match="chemical codes must be a permutation"):
        discover_and_validate_inputs([run])


def test_multiple_method_and_seed_subset_run_dirs_form_one_complete_grid(tmp_path):
    full = _write_synthetic_run(tmp_path / "full")
    base_manifest = json.loads((full / "manifest.json").read_text())
    run_specs = (
        ("direct", ["direct_importance"], [11, 29]),
        ("progressive_11", ["progressive_bridge_smc"], [11]),
        ("progressive_29", ["progressive_bridge_smc"], [29]),
    )
    roots = []
    for name, methods, seeds in run_specs:
        root = tmp_path / name
        root.mkdir()
        manifest = {**base_manifest, "methods": methods, "seeds": seeds}
        _set_manifest_fingerprint(manifest)
        (root / "manifest.json").write_text(json.dumps(manifest))
        for method in methods:
            destination = root / "responses" / method
            destination.mkdir(parents=True)
            for seed in seeds:
                for source in (full / "responses" / method).glob(f"*__s{seed}.npz"):
                    shutil.copy2(source, destination / source.name)
        roots.append(root)
    inputs = discover_and_validate_inputs(roots)
    assert inputs.methods == METHODS
    assert inputs.seeds == (11, 29)
    assert len(inputs.records) == 8


def test_particle_diagnostic_ranges_fail_closed(tmp_path):
    run = _write_synthetic_run(tmp_path / "run")
    path = next((run / "responses" / "direct_importance").glob("*.npz"))
    with np.load(path, allow_pickle=False) as data:
        bad = data["diagnostic_max_weight_low"].copy()
    bad.flat[0] = 0.0
    _rewrite_npz(path, diagnostic_max_weight_low=bad)
    with pytest.raises(RuntimeError, match="direct max weight lies outside"):
        discover_and_validate_inputs([run])


def test_progressive_tempering_schedule_fails_closed(tmp_path):
    run = _write_synthetic_run(tmp_path / "run")
    path = next((run / "responses" / "progressive_bridge_smc").glob("*.npz"))
    with np.load(path, allow_pickle=False) as data:
        bad = data["diagnostic_step_beta_low"].copy()
    bad.flat[0] = 0.25
    _rewrite_npz(path, diagnostic_step_beta_low=bad)
    with pytest.raises(RuntimeError, match="beta schedule mismatch"):
        discover_and_validate_inputs([run])


def test_target_gap_identity_fails_closed(tmp_path):
    run = _write_synthetic_run(tmp_path / "run")
    path = next((run / "responses" / "direct_importance").glob("*.npz"))
    with np.load(path, allow_pickle=False) as data:
        bad = data["diagnostic_target_gap"].copy()
    bad.flat[0] += 0.5
    _rewrite_npz(path, diagnostic_target_gap=bad)
    with pytest.raises(RuntimeError, match="target-gap identity failed"):
        discover_and_validate_inputs([run])


def test_stimulus_composition_rejects_cuts_shifted_from_frozen_schedule(tmp_path):
    run = _write_synthetic_run(tmp_path / "run")
    for path in (run / "responses").glob("*/*.npz"):
        with np.load(path, allow_pickle=False) as data:
            cut_times = np.asarray(data["cut_times"]) + 1
            bounds = np.asarray(data["source_window_bounds"]) + 1
        _rewrite_npz(path, cut_times=cut_times, source_window_bounds=bounds)
    inputs = discover_and_validate_inputs([run])
    with pytest.raises(RuntimeError, match="frozen event schedule"):
        _stimulus_composition_frame(inputs)


def test_v2_seed_and_common_noise_provenance_fail_closed(tmp_path):
    run = _write_synthetic_run(tmp_path / "run")
    path = next((run / "responses" / "direct_importance").glob("*.npz"))
    _rewrite_npz(path, base_seed=np.asarray(17))
    with pytest.raises(RuntimeError, match="archive base seed mismatch"):
        discover_and_validate_inputs([run])


def test_source_run_manifest_tamper_and_status_fail_closed(tmp_path):
    run = _write_synthetic_run(tmp_path / "run")
    runner_manifest_path = run / "manifest.json"
    runner_manifest = json.loads(runner_manifest_path.read_text())
    source_manifest_path = Path(runner_manifest["source_run"]) / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["status"] = "partial"
    source_manifest_path.write_text(json.dumps(source_manifest))
    with pytest.raises(RuntimeError, match="source-run manifest hash audit failed"):
        discover_and_validate_inputs([run])
    runner_manifest["source_run_manifest_sha256"] = hashlib.sha256(
        source_manifest_path.read_bytes()
    ).hexdigest()
    _set_manifest_fingerprint(runner_manifest)
    runner_manifest_path.write_text(json.dumps(runner_manifest))
    with pytest.raises(RuntimeError, match="source run status is not complete"):
        discover_and_validate_inputs([run])


def test_fold_assignment_tamper_fails_closed(tmp_path):
    run = _write_synthetic_run(tmp_path / "run")
    runner_manifest = json.loads((run / "manifest.json").read_text())
    fold_path = Path(runner_manifest["fold_assignments"])
    fold_path.write_text(fold_path.read_text() + "extra,0\n")
    with pytest.raises(RuntimeError, match="fold-assignment hash audit failed"):
        discover_and_validate_inputs([run])


def test_hypothesis_queue_excludes_sources_below_validity_gate(tmp_path):
    run = _write_synthetic_run(tmp_path / "run", unsupported_source=1)
    output = tmp_path / "atlas"
    build_prediction_atlas(
        [run],
        output,
        config=AnalysisConfig(
            bootstrap_replicates=32,
            sign_flip_replicates=256,
            prediction_cells_per_slice=2,
            hypothesis_queue_size=50,
            dashboard_candidates=10,
        ),
    )
    queue = pd.read_csv(output / "hypothesis_queue.csv")
    assert not queue.empty
    assert (queue["valid_fraction"] >= 0.50).all()
    assert "target-B" not in set(queue["source_neuron"])
    reviewed = pd.read_parquet(output / "prediction_cells.parquet")
    assert "target-B" in set(reviewed["source_neuron"])
    assert (
        reviewed.loc[reviewed["source_neuron"] == "target-B", "support_tier"]
        == "unsupported"
    ).all()
    validation = json.loads((output / "validation.json").read_text())
    assert validation["hypothesis_queue_audit"]["unsupported_rows"] == 0


def test_confirmation_queue_requires_per_cell_cross_sampler_agreement(tmp_path):
    run = _write_synthetic_run(tmp_path / "run", opposite_direct=True)
    output = tmp_path / "atlas"
    build_prediction_atlas(
        [run],
        output,
        config=AnalysisConfig(
            bootstrap_replicates=32,
            sign_flip_replicates=256,
            prediction_cells_per_slice=2,
            hypothesis_queue_size=500,
            dashboard_candidates=10,
        ),
    )
    queue = pd.read_csv(output / "hypothesis_queue.csv")
    assert set(queue["method"]) == {"progressive_bridge_smc"}
    signed = queue[
        (queue["channel"] == "endpoint_mean")
        & (queue["mean_normalized"].abs() > 1e-8)
    ]
    assert not signed.empty
    assert (signed["counterpart_mean_normalized"] < 0).all()
    assert (signed["cross_sampler_sign_agreement"] == 0).all()
    assert (signed["cross_sampler_factor"] == 0).all()
    assert (~signed["promotion_eligible"]).all()
    assert (signed["support_tier"] != "supported_exploratory").all()
    reviewed = pd.read_parquet(output / "prediction_cells.parquet")
    assert set(reviewed["method"]) == set(METHODS)
    protocol = json.loads((output / "protocol.json").read_text())
    formula = protocol["confirmation_shortlist"]["cross_sampler_factor_formula"]
    assert "sign_gate * magnitude_agreement" in formula
    validation = json.loads((output / "validation.json").read_text())
    audit = validation["hypothesis_queue_audit"]
    assert audit["non_primary_rows"] == 0
    assert audit["promoted_signed_rows_with_sign_disagreement"] == 0


def test_genealogy_gate_blocks_strong_promotion_but_preserves_raw_model_rows(
    tmp_path,
):
    run = _write_synthetic_run(tmp_path / "run", low_genealogy=True)
    output = tmp_path / "atlas"
    build_prediction_atlas(
        [run],
        output,
        config=AnalysisConfig(
            bootstrap_replicates=32,
            sign_flip_replicates=256,
            prediction_cells_per_slice=2,
            hypothesis_queue_size=500,
            dashboard_candidates=10,
        ),
    )
    reviewed = pd.read_parquet(output / "prediction_cells.parquet")
    progressive = reviewed[reviewed.method == "progressive_bridge_smc"]
    assert not progressive.empty
    assert progressive.valid_fraction.eq(1.0).all()
    assert progressive.genealogy_valid_fraction_0_10.eq(0.0).all()
    assert progressive.genealogy_valid_fraction_0_20.eq(0.0).all()
    assert (~progressive.genealogy_strong_gate_pass).all()
    assert (progressive.support_tier != "supported_exploratory").all()
    queue = pd.read_csv(output / "hypothesis_queue.csv")
    assert not queue.empty
    assert queue.valid_fraction.eq(1.0).all()
    assert (~queue.genealogy_strong_gate_pass).all()
    assert (~queue.promotion_eligible).all()
    validation = json.loads((output / "validation.json").read_text())
    assert validation["hypothesis_queue_audit"][
        "promoted_with_genealogy_gate_failure"
    ] == 0


def test_candidate_lag_profile_localization_and_bootstrap_rate(tmp_path):
    lags = (1, 4, 8, 16)
    neurons = ("S", "T")
    primary = np.zeros((4, 1, 2, 2), dtype=np.float32)
    counterpart = np.zeros_like(primary)
    primary[:, 0, 1, 0] = [0.10, 0.50, 0.20, 0.15]
    counterpart[:, 0, 1, 0] = [0.05, 0.40, 0.10, 0.08]
    atlas_path = tmp_path / "atlas.npz"
    np.savez_compressed(
        atlas_path,
        mean_normalized__progressive_bridge_smc__endpoint_mean__onset=primary,
        mean_normalized__direct_importance__endpoint_mean__onset=counterpart,
        mean_normalized__progressive_bridge_smc__endpoint_wasserstein1__onset=primary,
        mean_normalized__direct_importance__endpoint_wasserstein1__onset=counterpart,
    )
    worm = np.zeros((4, 4, 1, 2, 2), dtype=np.float32)
    for index in range(4):
        worm[:, index, 0, 1, 0] = primary[:, 0, 1, 0] * (0.8 + index * 0.1)
    worm_path = tmp_path / "worms.npz"
    np.savez_compressed(
        worm_path,
        normalized__endpoint_mean__onset=worm,
        normalized__endpoint_wasserstein1__onset=worm,
    )
    queue = [
        {
            "method": "progressive_bridge_smc",
            "channel": "endpoint_mean",
            "context": "onset",
            "horizon_frames": 1,
            "source_index": 0,
            "target_index": 1,
            "source_neuron": "S",
            "target_neuron": "T",
            "promotion_eligible": True,
            "support_tier": "supported_exploratory",
        },
        {
            "method": "progressive_bridge_smc",
            "channel": "endpoint_wasserstein1",
            "context": "onset",
            "horizon_frames": 1,
            "source_index": 0,
            "target_index": 1,
            "source_neuron": "S",
            "target_neuron": "T",
            "promotion_eligible": False,
            "support_tier": "model_only",
        },
    ]
    inputs = SimpleNamespace(
        source_lags=lags,
        horizons=(1,),
        neurons=neurons,
        methods=("direct_importance", "progressive_bridge_smc"),
        fps=4.0,
    )
    enriched, long_rows = _append_candidate_lag_profiles(
        queue,
        atlas_path=atlas_path,
        worm_path=worm_path,
        inputs=inputs,
        primary_method="progressive_bridge_smc",
        bootstrap_weights=_bootstrap_counts(4, 32, 123),
    )
    row = enriched[0]
    assert row["primary_lag_profile"] == pytest.approx([0.10, 0.50, 0.20, 0.15])
    assert row["counterpart_lag_profile"] == pytest.approx([0.05, 0.40, 0.10, 0.08])
    assert row["peak_abs_effect_lag_frames"] == 4
    assert row["peak_abs_effect_lag_seconds"] == 1.0
    assert row["top_vs_second_lag_selectivity"] == pytest.approx(0.60)
    assert row["worm_bootstrap_peak_lag_selection_rate"] == 1.0
    assert row["signed_lag_profile_spearman"] > 0.9
    assert row["promotion_eligible"] is True
    unsigned = enriched[1]
    assert unsigned["lag_profile_spearman"] > 0.9
    assert np.isnan(unsigned["signed_lag_profile_spearman"])
    assert len(long_rows) == 8
    assert sum(item["is_primary_peak_abs_lag"] for item in long_rows) == 2
