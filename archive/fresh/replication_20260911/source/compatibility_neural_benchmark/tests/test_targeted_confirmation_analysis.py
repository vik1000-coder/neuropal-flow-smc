from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.prediction_atlas_runner import (
    EPISODE_SEED_DEFINITION,
    canonical_fingerprint,
)
from compatibility_neural_benchmark.targeted_confirmation import (
    ARCHIVE_SCHEMA_VERSION,
    ARM_METRICS,
    ARM_NAMES,
    CLAIM_BOUNDARY,
    CONTRAST_NAMES,
    DISTANCE_RESPONSE_KEYS,
    MANIFEST_SCHEMA_VERSION,
    METHOD,
    SIGNED_METRICS,
    SIGNED_RESPONSE_KEYS,
)
from compatibility_neural_benchmark.targeted_confirmation_analysis import (
    AnalysisConfig,
    COMMON_NOISE_DEFINITION,
    FACTUAL_ARM_DEFINITION,
    build_targeted_confirmation_analysis,
    discover_and_validate_targeted_inputs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_targeted_run(
    root: Path,
    *,
    low_genealogy: bool = False,
    extra_fold_worm: bool = False,
    omitted_source_worm: bool = False,
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    source = root / "source"
    source.mkdir()
    source_worms = (
        ["w0", "w1", "w2"] if omitted_source_worm else ["w0", "w1"]
    )
    source_manifest = {
        "source": "synthetic",
        "cohort_mode": "synthetic",
        "n_worms": len(source_worms),
        "stimulus_schema_version": "synthetic-v1",
        "stimulus_schema_fingerprint": "synthetic-fingerprint",
        "stimulus_schema": {
            "version": "synthetic-v1",
            "fingerprint": "synthetic-fingerprint",
            "schedules": [{"worm_id": worm} for worm in source_worms],
        },
    }
    (source / "manifest.json").write_text(json.dumps(source_manifest))
    fold_file = root / "folds.csv"
    include_third_fold_row = extra_fold_worm or omitted_source_worm
    fold_worm_indices = [0, 1, 2] if include_third_fold_row else [0, 1]
    third_worm = "w2" if omitted_source_worm else "other-strain-w2"
    fold_worm_ids = (
        ["w0", "w1", third_worm]
        if include_third_fold_row
        else ["w0", "w1"]
    )
    outer_folds = [0, 1, 0] if include_third_fold_row else [0, 1]
    pd.DataFrame(
        {
            "worm_index": fold_worm_indices,
            "worm_id": fold_worm_ids,
            "outer_fold": outer_folds,
        }
    ).to_csv(fold_file, index=False)
    queue = root / "hypothesis_queue.csv"
    pd.DataFrame(
        [
            {
                "queue_rank": 1,
                "source_neuron": "B",
                "target_neuron": "A",
                "source_lag_frames": 1,
                "horizon_frames": 1,
                "context": "onset_minus_baseline",
                "channel": "endpoint_mean",
                "method": "progressive_bridge_smc",
                "mean_normalized": 2.0,
                "selected": True,
            }
        ]
    ).to_csv(queue, index=False)
    selection_fingerprint = canonical_fingerprint(
        {
            "queue_sha256": _sha(queue),
            "source_lag_frames": 1,
            "source_indices": [1],
            "source_neurons": ["B"],
            "horizon_frames": [1, 2],
            "phases": ["baseline", "onset"],
        }
    )
    group = {
        "source_lag_frames": 1,
        "source_indices": [1],
        "source_neurons": ["B"],
        "horizon_frames": [1, 2],
        "phases": ["baseline", "onset"],
        "selection_fingerprint": selection_fingerprint,
    }
    spec = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": "targeted progressive-bridge low/high/factual confirmation v1",
        "method": METHOD,
        "model_id": "synthetic-flow",
        "checkpoint_phase": "test",
        "source_run": str(source.resolve()),
        "source_run_manifest_sha256": _sha(source / "manifest.json"),
        "fold_assignments": str(fold_file.resolve()),
        "fold_assignments_sha256": _sha(fold_file),
        "hypothesis_queue": str(queue.resolve()),
        "hypothesis_queue_sha256": _sha(queue),
        "resolved_groups": [group],
        "folds": [0, 1],
        "seeds": [11, 29],
        "base_seed": 20260829,
        "episode_seed_definition": EPISODE_SEED_DEFINITION,
        "requested_device": "cpu",
        "particles": 128,
        "history_frames": 8,
        "source_window_frames": 1,
        "endpoint_quantiles": [0.25, 0.5, 0.75],
        "phases_available": ["baseline", "onset", "active", "offset", "recovery"],
        "arm_names": list(ARM_NAMES),
        "signed_response_keys": list(SIGNED_RESPONSE_KEYS),
        "distance_response_keys": list(DISTANCE_RESPONSE_KEYS),
        "arm_response_keys": [f"arm_{metric}" for metric in ARM_METRICS],
        "matrix_orientation": "target_row_source_column",
        "progressive_branch_factor": 2,
        "progressive_future_branch_factor": 2,
        "minimum_effective_sample_size": 24.0,
        "maximum_normalized_weight": 0.2,
        "minimum_achieved_source_fraction": 0.25,
        "stimulus_schema_version": "synthetic-v1",
        "stimulus_schema_fingerprint": "synthetic-fingerprint",
        "stimulus_generator_encoding": "binary_any_stimulus",
        "chemical_identity_conditioned": False,
        "chemical_analysis_label": "event-stratified",
        "distribution_scale_floor": 1e-6,
        "endpoint_sd_ddof": 0,
        "endpoint_quantile_method": "numpy_linear_empirical",
        "factual_arm_definition": FACTUAL_ARM_DEFINITION,
        "common_noise_definition": COMMON_NOISE_DEFINITION,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    manifest = {
        "created_utc": "2026-08-29T00:00:00+00:00",
        "run_spec_fingerprint": canonical_fingerprint(spec),
        **spec,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    responses = root / "responses"
    responses.mkdir()
    neurons = np.asarray(["A", "B"])
    quantile_levels = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
    for fold, worm in enumerate(("w0", "w1")):
        for seed in (11, 29):
            checkpoint = root / f"checkpoint_f{fold}_s{seed}.pt"
            checkpoint.write_bytes(f"checkpoint-{fold}-{seed}".encode())
            w, p, e, h, d, m, q = 1, 2, 3, 2, 2, 1, 3
            arm_shape = (w, p, e, 3, h, d, m)
            arms = {
                metric: np.zeros(arm_shape, dtype=np.float32)
                for metric in ARM_METRICS
            }
            seed_shift = 0.2 if seed == 29 else 0.0
            worm_shift = 2.0 * fold
            for phase in range(p):
                high = 2.0 if phase == 0 else 6.0 + worm_shift + seed_shift
                for horizon in range(h):
                    arms["endpoint_mean"][0, phase, :, 0, horizon, 0, 0] = 0.0
                    arms["endpoint_mean"][0, phase, :, 1, horizon, 0, 0] = high
                    arms["endpoint_mean"][0, phase, :, 2, horizon, 0, 0] = 1.0
                    arms["time_average_mean"][0, phase, :, :, horizon, 0, 0] = (
                        arms["endpoint_mean"][0, phase, :, :, horizon, 0, 0]
                    )
                    arms["pathwise_peak_mean"][0, phase, :, :, horizon, 0, 0] = (
                        arms["endpoint_mean"][0, phase, :, :, horizon, 0, 0]
                    )
                    arms["crossing_probability"][0, phase, :, 0, horizon, 0, 0] = 0.1
                    arms["crossing_probability"][0, phase, :, 1, horizon, 0, 0] = (
                        0.3 if phase == 0 else 0.7
                    )
                    arms["crossing_probability"][0, phase, :, 2, horizon, 0, 0] = 0.2
                    arms["endpoint_sd"][0, phase, :, 0, horizon, 0, 0] = 1.0
                    arms["endpoint_sd"][0, phase, :, 1, horizon, 0, 0] = (
                        2.0 if phase == 0 else 3.0
                    )
                    arms["endpoint_sd"][0, phase, :, 2, horizon, 0, 0] = 1.5
            response: dict[str, np.ndarray] = {}
            arm_key = {
                "endpoint_mean": "endpoint_mean",
                "time_average_mean": "time_average_mean",
                "pathwise_peak_mean": "pathwise_peak_mean",
                "crossing_probability": "crossing_probability",
            }
            pairs = {"high_low": (1, 0), "high_factual": (1, 2), "low_factual": (0, 2)}
            for contrast, (left, right) in pairs.items():
                for metric in arm_key:
                    response[f"response_{contrast}_{metric}"] = (
                        arms[metric][:, :, :, left] - arms[metric][:, :, :, right]
                    )
                response[f"response_{contrast}_endpoint_log_sd"] = (
                    np.log(arms["endpoint_sd"][:, :, :, left] + 1e-6)
                    - np.log(arms["endpoint_sd"][:, :, :, right] + 1e-6)
                )
                distance = np.zeros((w, p, e, h, d, m), dtype=np.float32)
                distance[..., 0, 0] = 1.0 + 0.5 * np.arange(p)[None, :, None, None]
                response[f"distance_{contrast}_endpoint_wasserstein1"] = distance
            arm_quantile = np.zeros((w, p, e, 3, h, q, d, m), dtype=np.float32)
            base_q = np.asarray([-1.0, 0.0, 1.0])
            for phase in range(p):
                high_shift = 2.0 if phase == 0 else 6.0 + worm_shift + seed_shift
                arm_quantile[0, phase, :, 0, :, :, 0, 0] = base_q
                arm_quantile[0, phase, :, 1, :, :, 0, 0] = base_q + high_shift
                arm_quantile[0, phase, :, 2, :, :, 0, 0] = base_q + 1.0
            response["arm_endpoint_quantile"] = arm_quantile
            for contrast, (left, right) in pairs.items():
                response[f"response_{contrast}_endpoint_quantile"] = (
                    arm_quantile[:, :, :, left] - arm_quantile[:, :, :, right]
                )
            diagnostic_shape = (w, p, e, m)
            target_low = np.full(diagnostic_shape, -1.0, dtype=np.float32)
            target_high = np.full(diagnostic_shape, 1.0, dtype=np.float32)
            achieved_low = target_low.copy()
            achieved_high = target_high.copy()
            cut_times = np.asarray(
                [[[20, 140, 260], [80, 200, 320]]], dtype=np.int32
            )
            bounds = np.stack([cut_times - 1, cut_times], axis=-1)
            codes = np.asarray([[1, 2, 3]], dtype=np.int8)
            names = np.asarray([["butanone", "pentanedione", "nacl"]])
            step_shape = (*diagnostic_shape, 2)
            path = responses / f"targeted__ell1__f{fold}__s{seed}.npz"
            np.savez_compressed(
                path,
                status=np.asarray("complete"),
                archive_schema_version=np.asarray(ARCHIVE_SCHEMA_VERSION),
                method=np.asarray(METHOD),
                model_id=np.asarray("synthetic-flow"),
                checkpoint=np.asarray(str(checkpoint.resolve())),
                checkpoint_sha256=np.asarray(_sha(checkpoint)),
                fold=np.asarray(fold),
                seed=np.asarray(seed),
                base_seed=np.asarray(20260829),
                episode_seed_definition=np.asarray(EPISODE_SEED_DEFINITION),
                requested_device=np.asarray("cpu"),
                resolved_device=np.asarray("cpu"),
                history_frames=np.asarray(8),
                repair_frames=np.asarray(2),
                source_lag_frames=np.asarray(1),
                source_lag_seconds=np.asarray(0.25),
                lag_definition=np.asarray("source-window end to prediction cut"),
                source_window_frames=np.asarray(1),
                source_window_bounds_semantics=np.asarray("[inclusive_start,exclusive_stop)"),
                n_particles=np.asarray(128),
                progressive_branch_factor=np.asarray(2),
                progressive_future_branch_factor=np.asarray(2),
                minimum_effective_sample_size=np.asarray(24.0),
                maximum_normalized_weight=np.asarray(0.2),
                minimum_achieved_source_fraction=np.asarray(0.25),
                horizon_frames=np.asarray([1, 2]),
                horizon_seconds=np.asarray([0.25, 0.5]),
                source_to_readout_seconds=np.asarray([0.5, 0.75]),
                fps=np.asarray(4.0),
                phase_names=np.asarray(["baseline", "onset"]),
                arm_names=np.asarray(ARM_NAMES),
                signed_response_keys=np.asarray(SIGNED_RESPONSE_KEYS),
                distance_response_keys=np.asarray(DISTANCE_RESPONSE_KEYS),
                arm_response_keys=np.asarray([f"arm_{metric}" for metric in ARM_METRICS]),
                signed_response_axes=np.asarray(("heldout_worm", "phase", "event", "horizon", "target", "source")),
                arm_response_axes=np.asarray(("heldout_worm", "phase", "event", "arm", "horizon", "target", "source")),
                matrix_orientation=np.asarray("target_row_source_column"),
                selected_source_indices=np.asarray([1]),
                selected_source_neurons=np.asarray(["B"]),
                target_neurons=neurons,
                endpoint_quantiles=quantile_levels,
                selection_fingerprint=np.asarray(selection_fingerprint),
                hypothesis_queue=np.asarray(str(queue.resolve())),
                hypothesis_queue_sha256=np.asarray(_sha(queue)),
                worm_indices=np.asarray([fold]),
                worm_ids=np.asarray([worm]),
                cut_times=cut_times,
                source_window_bounds=bounds,
                stimulus_schema_version=np.asarray("synthetic-v1"),
                stimulus_schema_fingerprint=np.asarray("synthetic-fingerprint"),
                stimulus_generator_encoding=np.asarray("binary_any_stimulus"),
                chemical_identity_conditioned=np.asarray(False),
                chemical_code_by_worm_event=codes,
                chemical_name_by_worm_event=names,
                event_intervals_seconds_by_worm=np.asarray(
                    [[[20, 30], [50, 60], [80, 90]]], dtype=np.float32
                ),
                schedule_source_recording_by_worm=np.asarray([f"recording-{fold}"]),
                schedule_native_fps_by_worm=np.asarray([4.0]),
                schedule_analysis_fps_by_worm=np.asarray([4.0]),
                schedule_resampling_provenance_by_worm=np.asarray(["none"]),
                distribution_scale_floor=np.asarray(1e-6),
                endpoint_sd_ddof=np.asarray(0),
                endpoint_quantile_method=np.asarray("numpy_linear_empirical"),
                factual_arm_definition=np.asarray(FACTUAL_ARM_DEFINITION),
                common_noise_definition=np.asarray(COMMON_NOISE_DEFINITION),
                claim_boundary=np.asarray(CLAIM_BOUNDARY),
                diagnostic_target_low=target_low,
                diagnostic_target_high=target_high,
                diagnostic_target_gap=target_high - target_low,
                diagnostic_achieved_low=achieved_low,
                diagnostic_achieved_high=achieved_high,
                diagnostic_achieved_gap=achieved_high - achieved_low,
                diagnostic_ess_low=np.full(diagnostic_shape, 100.0),
                diagnostic_ess_high=np.full(diagnostic_shape, 100.0),
                diagnostic_max_weight_low=np.full(diagnostic_shape, 0.02),
                diagnostic_max_weight_high=np.full(diagnostic_shape, 0.02),
                diagnostic_valid=np.ones(diagnostic_shape),
                diagnostic_factual_source=np.zeros(diagnostic_shape),
                diagnostic_selected_source_index=np.ones(diagnostic_shape),
                diagnostic_endpoint_sd_floor=np.full(diagnostic_shape, 1e-6),
                diagnostic_distinct_ancestors_low=np.full(
                    diagnostic_shape, 3.0 if low_genealogy else 80.0
                ),
                diagnostic_distinct_ancestors_high=np.full(
                    diagnostic_shape, 3.0 if low_genealogy else 80.0
                ),
                diagnostic_step_ess_low=np.full(step_shape, 100.0),
                diagnostic_step_ess_high=np.full(step_shape, 100.0),
                diagnostic_step_max_weight_low=np.full(step_shape, 0.02),
                diagnostic_step_max_weight_high=np.full(step_shape, 0.02),
                diagnostic_step_forced_tempering_low=np.zeros(step_shape),
                diagnostic_step_forced_tempering_high=np.zeros(step_shape),
                **{f"arm_{key}": value for key, value in arms.items()},
                **response,
            )
    validation = {
        "status": "pass",
        "run_spec_fingerprint": manifest["run_spec_fingerprint"],
        "expected_runs": 4,
        "completed_or_skipped": 4,
        "failed": 0,
    }
    (root / "validation.json").write_text(json.dumps(validation))
    screen = root / "screen"
    screen.mkdir()
    (screen / "hypothesis_queue.csv").write_bytes(queue.read_bytes())
    matrix = np.zeros((1, 2, 2, 2), dtype=np.float32)
    matrix[0, 0, 0, 1] = 2.0
    valid = np.ones((1, 2), dtype=np.float32)
    np.savez_compressed(
        screen / "atlas_matrices.npz",
        neurons=neurons,
        methods=np.asarray(["progressive_bridge_smc"]),
        channels=np.asarray(["endpoint_mean"]),
        contexts=np.asarray(["onset_minus_baseline"]),
        source_lag_frames=np.asarray([1]),
        horizon_frames=np.asarray([1, 2]),
        orientation=np.asarray("target_row_source_column"),
        primary_method=np.asarray("progressive_bridge_smc"),
        mean_normalized__progressive_bridge_smc__endpoint_mean__onset_minus_baseline=matrix,
        valid_fraction__progressive_bridge_smc__onset_minus_baseline=valid,
    )
    return root, screen


def test_build_targeted_analysis_preserves_worm_unit_orientation_and_screen(tmp_path):
    run, screen = _synthetic_targeted_run(tmp_path / "run")
    output = tmp_path / "analysis"
    build_targeted_confirmation_analysis(
        run,
        output,
        screen_dir=screen,
        config=AnalysisConfig(bootstrap_replicates=64),
    )
    cells = pd.read_csv(output / "targeted_cells.csv")
    row = cells[
        (cells.contrast == "high_low")
        & (cells.metric == "endpoint_mean")
    ].iloc[0]
    assert row.source_neuron == "B" and row.target_neuron == "A"
    assert row.source_index == 1 and row.target_index == 0
    assert row.mean_normalized == pytest.approx(2.55)
    assert row.n_worms == 2 and row.n_model_seeds == 2
    assert len(cells) == 21
    quantiles = pd.read_csv(output / "targeted_quantile_shifts.csv")
    assert len(quantiles) == 9
    consistency = pd.read_csv(output / "screen_consistency.csv")
    assert consistency.iloc[0].direction_agreement == 1.0
    assert consistency.iloc[0].targeted_n128_mean_normalized == pytest.approx(2.55)
    assert bool(consistency.iloc[0].same_sampler_family)
    assert (
        consistency.iloc[0].comparison_type
        == "particle_escalation_within_progressive_bridge"
    )
    assert not cells.evidence_label.str.contains("confirmed", case=False).any()
    support = pd.read_csv(output / "support_diagnostics.csv")
    assert support.valid_fraction.tolist() == [1.0]
    assert support.mean_achieved_gap_magnitude.tolist() == [2.0]
    assert support.mean_ess_low.tolist() == [100.0]
    assert support.mean_distinct_ancestors_low.tolist() == [80.0]
    assert support.genealogy_valid_fraction_0_10.tolist() == [1.0]
    assert support.genealogy_valid_fraction_0_20.tolist() == [1.0]
    assert support.genealogy_strong_gate_pass.tolist() == [True]
    assert support.genealogy_sensitivity_gate_pass.tolist() == [True]
    assert cells.genealogy_strong_gate_pass.all()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["genealogy_support"]["hard_strong_min_ancestor_fraction"] == 0.10
    report = (output / "REPORT.md").read_text()
    assert "Source support and genealogy" in report
    assert "does not require stable worm ranking" in report
    assert "not multiplicity-adjusted" in report
    first_reviewed_row = report.split("## Highest-magnitude reviewed rows", 1)[1].splitlines()[5]
    displayed_mean = float(first_reviewed_row.split("|")[5].strip())
    assert abs(displayed_mean) == pytest.approx(
        cells.mean_normalized.abs().max(), abs=5e-4
    )
    for line in (output / "checksums.sha256").read_text().strip().splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha(output / filename) == digest


def test_targeted_genealogy_gate_downgrades_strong_label_without_dropping_rows(
    tmp_path,
):
    run, screen = _synthetic_targeted_run(
        tmp_path / "run", low_genealogy=True
    )
    output = tmp_path / "analysis"
    build_targeted_confirmation_analysis(
        run,
        output,
        screen_dir=screen,
        config=AnalysisConfig(bootstrap_replicates=64),
    )
    support = pd.read_csv(output / "support_diagnostics.csv")
    assert support.valid_fraction.tolist() == [1.0]
    assert support.genealogy_valid_fraction_0_10.tolist() == [0.0]
    assert support.genealogy_valid_fraction_0_20.tolist() == [0.0]
    assert support.genealogy_strong_gate_pass.tolist() == [False]
    cells = pd.read_csv(output / "targeted_cells.csv")
    assert len(cells) == 21
    assert cells.valid_fraction.eq(1.0).all()
    assert (~cells.genealogy_strong_gate_pass).all()
    signed = cells[cells.signed.astype(bool)]
    assert "model_relative_consistent" not in set(signed.evidence_label)
    assert "model_relative_uncertain" in set(signed.evidence_label)


def test_targeted_analysis_fails_closed_on_incomplete_requested_grid(tmp_path):
    run, _ = _synthetic_targeted_run(tmp_path / "run")
    next((run / "responses").glob("*f1__s29.npz")).unlink()
    with pytest.raises(RuntimeError, match="grid mismatch"):
        discover_and_validate_targeted_inputs(
            run, config=AnalysisConfig(bootstrap_replicates=32)
        )


def test_targeted_analysis_ignores_out_of_cohort_rows_in_shared_fold_file(tmp_path):
    run, _ = _synthetic_targeted_run(
        tmp_path / "run", extra_fold_worm=True
    )
    inputs = discover_and_validate_targeted_inputs(
        run, config=AnalysisConfig(bootstrap_replicates=32)
    )
    assert inputs.worm_ids == ("w0", "w1")


def test_targeted_analysis_rejects_cohort_worm_omitted_from_every_archive(tmp_path):
    run, _ = _synthetic_targeted_run(
        tmp_path / "run", omitted_source_worm=True
    )
    with pytest.raises(RuntimeError, match="coverage differs"):
        discover_and_validate_targeted_inputs(
            run, config=AnalysisConfig(bootstrap_replicates=32)
        )


def test_targeted_analysis_fails_closed_on_queue_hash_change(tmp_path):
    run, _ = _synthetic_targeted_run(tmp_path / "run")
    with (run / "hypothesis_queue.csv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(RuntimeError, match="hypothesis_queue path/hash"):
        discover_and_validate_targeted_inputs(
            run, config=AnalysisConfig(bootstrap_replicates=32)
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "selection_fingerprint",
            np.asarray("wrong-selection"),
            "selection fingerprint",
        ),
        (
            "signed_response_axes",
            np.asarray(
                ("heldout_worm", "phase", "event", "horizon", "source", "target")
            ),
            "signed_response_axes",
        ),
    ),
)
def test_targeted_analysis_rejects_selection_or_axis_provenance(
    tmp_path, field, replacement, message
):
    run, _ = _synthetic_targeted_run(tmp_path / "run")
    archive = next((run / "responses").glob("*.npz"))
    with np.load(archive, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    payload[field] = replacement
    np.savez_compressed(archive, **payload)
    with pytest.raises(RuntimeError, match=message):
        discover_and_validate_targeted_inputs(
            run, config=AnalysisConfig(bootstrap_replicates=32)
        )


def test_targeted_analysis_rejects_valid_flag_that_disagrees_with_gates(tmp_path):
    run, _ = _synthetic_targeted_run(tmp_path / "run")
    archive = next((run / "responses").glob("*.npz"))
    with np.load(archive, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    payload["diagnostic_valid"] = np.zeros_like(payload["diagnostic_valid"])
    np.savez_compressed(archive, **payload)
    with pytest.raises(RuntimeError, match="diagnostic_valid does not match declared gates"):
        discover_and_validate_targeted_inputs(
            run, config=AnalysisConfig(bootstrap_replicates=32)
        )


def test_targeted_analysis_rejects_cuts_shifted_from_frozen_schedule(tmp_path):
    run, _ = _synthetic_targeted_run(tmp_path / "run")
    archive = next((run / "responses").glob("*.npz"))
    with np.load(archive, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    payload["cut_times"] = payload["cut_times"] + 1
    payload["source_window_bounds"] = payload["source_window_bounds"] + 1
    np.savez_compressed(archive, **payload)
    with pytest.raises(RuntimeError, match="frozen event schedule"):
        discover_and_validate_targeted_inputs(
            run, config=AnalysisConfig(bootstrap_replicates=32)
        )
