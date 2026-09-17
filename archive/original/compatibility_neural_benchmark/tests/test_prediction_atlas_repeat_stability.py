from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.prediction_atlas_repeat_stability import (
    OVERLAP_CHANNELS,
    OVERLAP_CONTEXTS,
    ORIENTATION,
    run_repeat_stability_audit,
    sha256,
)


METHODS = ("direct_importance", "progressive_bridge_smc")
LAGS = (1, 4)
SEEDS = (11, 22)
FOLDS = (0, 1)
WORMS = ("W0", "W1", "W2", "W3")
NEURONS = ("A", "B", "C")


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _ledger(root: Path, names: tuple[str, ...]) -> None:
    (root / "checksums.sha256").write_text(
        "".join(f"{sha256(root / name)}  {name}\n" for name in names)
    )


def _source_fixture(root: Path) -> tuple[Path, Path, str, str, list[dict[str, object]]]:
    source = root / "source_run"
    source.mkdir()
    fold_path = root / "fold_assignments.csv"
    pd.DataFrame(
        {"worm_id": WORMS, "outer_fold": [0, 1, 0, 1]}
    ).to_csv(fold_path, index=False)
    _dump(source / "manifest.json", {"status": "complete"})
    source_hash = sha256(source / "manifest.json")
    fold_hash = sha256(fold_path)
    checkpoints: list[dict[str, object]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            path = source / "checkpoints" / f"model__f{fold}__s{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"checkpoint-{fold}-{seed}".encode())
            checkpoints.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
            )
    return source, fold_path, source_hash, fold_hash, checkpoints


def _run_manifest(
    *,
    root: Path,
    source: Path,
    fold_path: Path,
    fold_hash: str,
    method: str,
    particles: int,
    min_ess: float,
    future_branch: int,
) -> dict[str, object]:
    manifest = {
        "protocol": "corrected four-workflow flow-repaired explicit source-lag analysis v1",
        "methods": [method],
        "source_run": str(source.resolve()),
        "checkpoint_phase": "chemical_full_cv",
        "model_id": "stim_binary_any_stimulus_tcn_flow128_dropout15_jitter_p01",
        "cohort_mode": "oh16230_head",
        "n_worms": len(WORMS),
        "n_neurons": len(NEURONS),
        "fps": 4.0,
        "history_frames": 80,
        "source_lag_frames": list(LAGS),
        "lag_definition": "source-window end to prediction cut",
        "source_window_frames": 4,
        "horizon_frames": [1],
        "phases": ["baseline", "onset", "active"],
        "particles": particles,
        "minimum_effective_sample_size": min_ess,
        "progressive_branch_factor": 2,
        "progressive_future_branch_factor": future_branch,
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "stimulus_schema": {"version": "test", "fingerprint": "abc"},
        "fold_assignments": str(fold_path.resolve()),
        "fold_assignments_sha256": fold_hash,
        "matrix_internal_orientation": "source,horizon,target",
    }
    _dump(root / "manifest.json", manifest)
    _dump(
        root / "validation.json",
        {
            "status": "pass",
            "expected_runs": len(LAGS) * len(FOLDS) * len(SEEDS),
            "completed_or_skipped": len(LAGS) * len(FOLDS) * len(SEEDS),
            "failed": 0,
            "failures": [],
        },
    )
    return manifest


def _raw_response(
    *,
    worm_indices: list[int],
    lag: int,
    seed: int,
    channel_index: int,
) -> np.ndarray:
    value = np.empty((len(worm_indices), 3, 3, len(NEURONS), 1, len(NEURONS)), dtype=np.float32)
    scale = 0.01 if OVERLAP_CHANNELS[channel_index] == "event_probability" else 0.2 + 0.05 * channel_index
    for local, worm in enumerate(worm_indices):
        for phase in range(3):
            for event in range(3):
                for source in range(len(NEURONS)):
                    for target in range(len(NEURONS)):
                        signed = (target - source) + 0.15 * (worm + 1) + 0.04 * phase
                        value[local, phase, event, source, 0, target] = scale * (
                            signed + 0.01 * event + 0.001 * seed + 0.003 * lag
                        )
    return value


def _historical_fixture(
    root: Path,
    *,
    source: Path,
    fold_path: Path,
    fold_hash: str,
    checkpoints: list[dict[str, object]],
) -> tuple[Path, dict[tuple[str, str, str, int], np.ndarray]]:
    primary = root / "historical"
    direct = root / "historical_direct_n256"
    primary.mkdir()
    direct.mkdir()
    progressive_manifest = _run_manifest(
        root=primary,
        source=source,
        fold_path=fold_path,
        fold_hash=fold_hash,
        method="progressive_bridge_smc",
        particles=32,
        min_ess=6.0,
        future_branch=2,
    )
    direct_manifest = _run_manifest(
        root=direct,
        source=source,
        fold_path=fold_path,
        fold_hash=fold_hash,
        method="direct_importance",
        particles=256,
        min_ess=20.0,
        future_branch=2,
    )
    checkpoint_lookup = {
        (int(row["fold"]), int(row["seed"])): row for row in checkpoints
    }
    input_hashes: dict[str, str] = {}
    oriented: dict[
        tuple[str, int, str, int, str], np.ndarray
    ] = {}
    for canonical, raw_method, run, manifest in (
        ("direct_importance", "direct_importance", direct, direct_manifest),
        (
            "progressive_bridge_smc",
            "progressive_bridge_smc",
            primary,
            progressive_manifest,
        ),
    ):
        particles = int(manifest["particles"])
        for lag in LAGS:
            for fold in FOLDS:
                heldout = [index for index in range(len(WORMS)) if index % 2 == fold]
                for seed in SEEDS:
                    checkpoint = checkpoint_lookup[(fold, seed)]
                    cut_times = np.empty((len(heldout), 3, 3), dtype=np.int32)
                    for phase in range(3):
                        for event in range(3):
                            cut_times[:, phase, event] = 60 + 100 * event + 15 * phase
                    stop = cut_times - lag + 1
                    bounds = np.stack((stop - 4, stop), axis=-1)
                    achieved_low = np.zeros((len(heldout), 3, 3, len(NEURONS)), dtype=np.float32)
                    achieved_high = np.ones_like(achieved_low)
                    target_low = np.zeros_like(achieved_low)
                    target_high = np.ones_like(achieved_low)
                    responses = {
                        f"response_{channel}": _raw_response(
                            worm_indices=heldout,
                            lag=lag,
                            seed=seed,
                            channel_index=channel_index,
                        )
                        for channel_index, channel in enumerate(OVERLAP_CHANNELS)
                    }
                    path = (
                        run
                        / "responses"
                        / raw_method
                        / (
                            f"model__{raw_method}__ell{lag}__N{particles}"
                            f"__f{fold}__s{seed}.npz"
                        )
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        path,
                        status=np.asarray("complete"),
                        method=np.asarray(raw_method),
                        model_id=np.asarray(manifest["model_id"]),
                        checkpoint=np.asarray(checkpoint["path"]),
                        checkpoint_sha256=np.asarray(checkpoint["sha256"]),
                        fold=np.asarray(fold),
                        seed=np.asarray(seed),
                        history_frames=np.asarray(80),
                        repair_frames=np.asarray(lag + 4),
                        source_lag_frames=np.asarray(lag),
                        lag_definition=np.asarray("source-window end to prediction cut"),
                        source_window_frames=np.asarray(4),
                        n_particles=np.asarray(particles),
                        horizon_frames=np.asarray([1], dtype=np.int16),
                        phase_names=np.asarray(["baseline", "onset", "active"]),
                        worm_ids=np.asarray([WORMS[index] for index in heldout]),
                        neurons=np.asarray(NEURONS),
                        cut_times=cut_times,
                        source_window_bounds=bounds,
                        diagnostic_achieved_gap=achieved_high - achieved_low,
                        diagnostic_achieved_low=achieved_low,
                        diagnostic_achieved_high=achieved_high,
                        diagnostic_target_gap=target_high - target_low,
                        diagnostic_target_low=target_low,
                        diagnostic_target_high=target_high,
                        **responses,
                    )
                    input_hashes[str(path.resolve())] = sha256(path)
                    for channel in OVERLAP_CHANNELS:
                        value = responses[f"response_{channel}"].mean(axis=2).transpose(
                            0, 1, 3, 4, 2
                        )
                        for local, worm_index in enumerate(heldout):
                            oriented[
                                (canonical, lag, channel, seed, WORMS[worm_index])
                            ] = value[local]

    analysis = primary / "analysis"
    analysis.mkdir()
    archive: dict[str, np.ndarray] = {"neurons": np.asarray(NEURONS)}
    group_matrices: dict[tuple[str, str, str, int], np.ndarray] = {}
    for canonical, historical in (
        ("direct_importance", "direct_importance_n256"),
        ("progressive_bridge_smc", "progressive_bridge_smc"),
    ):
        for channel in OVERLAP_CHANNELS:
            for context in OVERLAP_CONTEXTS:
                worm_lags = []
                for lag in LAGS:
                    worms = []
                    for worm in WORMS:
                        phase = np.mean(
                            [oriented[(canonical, lag, channel, seed, worm)] for seed in SEEDS],
                            axis=0,
                        ).astype(np.float32)
                        if context == "baseline":
                            selected = phase[0, 0]
                        elif context == "onset":
                            selected = phase[1, 0]
                        elif context == "active":
                            selected = phase[2, 0]
                        else:
                            selected = phase[1, 0] - phase[0, 0]
                        worms.append(selected)
                    worm_matrix = np.stack(worms).astype(np.float32)
                    worm_lags.append(worm_matrix)
                    group_matrices[(canonical, channel, context, lag)] = worm_matrix.mean(
                        axis=0
                    ).astype(np.float32)
                prefix = f"{historical}__{channel}__{context}"
                archive[f"{prefix}__lags"] = np.asarray(LAGS, dtype=np.int16)
                archive[f"{prefix}__worm_ids"] = np.asarray(WORMS)
                archive[f"{prefix}__worm_matrices"] = np.stack(worm_lags)
                archive[f"{prefix}__matrices"] = np.stack(
                    [group_matrices[(canonical, channel, context, lag)] for lag in LAGS]
                )
    np.savez_compressed(analysis / "aligned_four_sampler_lag_matrices.npz", **archive)
    _dump(analysis / "input_checksums.json", input_hashes)
    _dump(
        analysis / "validation.json",
        {
            "status": "pass",
            "matrix_orientation": "target,source",
            "internal_sampler_orientation": "source,horizon,target",
            "transpose_count_at_analysis_boundary": 1,
            "orientation_spot_check": True,
            "checkpoint_paths_and_sha256_reverified": True,
            "worm_to_fold_mapping_reverified": True,
            "source_window_and_lag_bounds_reverified": True,
            "normalization": (
                "response divided by max(abs(achieved high-low source gap), 0.10), "
                "event-wise before averaging"
            ),
            "lag_definition": "source-window end to prediction cut",
        },
    )
    (analysis / "REPORT.md").write_text("# Synthetic historical E26 fixture\n")
    _ledger(
        analysis,
        (
            "aligned_four_sampler_lag_matrices.npz",
            "validation.json",
            "input_checksums.json",
            "REPORT.md",
        ),
    )
    return analysis, group_matrices


def _atlas_fixture(
    root: Path,
    *,
    source: Path,
    fold_path: Path,
    source_hash: str,
    fold_hash: str,
    checkpoints: list[dict[str, object]],
    historical_matrices: dict[tuple[str, str, str, int], np.ndarray],
) -> Path:
    atlas = root / "atlas"
    atlas.mkdir()
    run_manifests = []
    for method, particles in (("direct_importance", 256), ("progressive_bridge_smc", 32)):
        path = root / f"atlas_run_{method}" / "manifest.json"
        value = {
            "manifest_schema_version": "prediction_atlas_manifest_v2",
            "methods": [method],
            "source_run": str(source.resolve()),
            "source_run_manifest_sha256": source_hash,
            "fold_assignments": str(fold_path.resolve()),
            "fold_assignments_sha256": fold_hash,
            "response_axes": [
                "heldout_worm",
                "phase",
                "event",
                "source",
                "horizon",
                "target",
            ],
            "matrix_internal_orientation": "source,horizon,target",
            "lag_definition": "source-window end to prediction cut",
            "phases": ["baseline", "onset", "active", "offset", "recovery"],
            "particles": particles,
            "minimum_effective_sample_size": 6.0,
            "progressive_branch_factor": 2,
            "progressive_future_branch_factor": 1,
        }
        _dump(path, value)
        run_manifests.append({"path": str(path.resolve()), "sha256": sha256(path)})

    arrays: dict[str, np.ndarray] = {
        "neurons": np.asarray(NEURONS),
        "methods": np.asarray(METHODS),
        "channels": np.asarray(OVERLAP_CHANNELS),
        "contexts": np.asarray((*OVERLAP_CONTEXTS, "state_average")),
        "source_lag_frames": np.asarray(LAGS, dtype=np.int16),
        "horizon_frames": np.asarray([1, 2], dtype=np.int16),
        "orientation": np.asarray(ORIENTATION),
    }
    for method in METHODS:
        for channel in OVERLAP_CHANNELS:
            for context in OVERLAP_CONTEXTS:
                matrices = []
                for lag in LAGS:
                    old = historical_matrices[(method, channel, context, lag)]
                    matrices.append(np.stack((2.0 * old, 3.0 * old)).astype(np.float32))
                arrays[f"mean_normalized__{method}__{channel}__{context}"] = np.stack(
                    matrices
                )
    np.savez_compressed(atlas / "atlas_matrices.npz", **arrays)
    _dump(
        atlas / "models.json",
        {
            "source_run_provenance": {
                "path": str(source.resolve()),
                "manifest": str((source / "manifest.json").resolve()),
                "manifest_sha256": source_hash,
                "status": "complete",
                "fold_assignments": str(fold_path.resolve()),
                "fold_assignments_sha256": fold_hash,
            },
            "methods": {
                "direct_importance": {
                    "particles": [256],
                    "branch_factor": 1,
                    "future_branch_factor": 1,
                },
                "progressive_bridge_smc": {
                    "particles": [32],
                    "branch_factor": 2,
                    "future_branch_factor": 1,
                },
            },
            "checkpoints": checkpoints,
        },
    )
    _dump(
        atlas / "protocol.json",
        {
            "orientation": ORIENTATION,
            "orientation_operation": "sampler source and target axes transposed exactly once",
            "ranking": "no connectome or external reference data used",
            "effect_definition": "high-source repaired response minus low-source repaired response",
            "normalization": {
                "signed_channels": "event-wise effect / max(abs(achieved_source_gap), 0.1)"
            },
            "n_neurons": len(NEURONS),
            "fps": 4.0,
            "history_frames": 80,
            "source_window_frames": 4,
            "source_lag_frames": list(LAGS),
            "horizon_frames": [1, 2],
        },
    )
    _dump(
        atlas / "validation.json",
        {
            "status": "passed",
            "problems": [],
            "archive_checks_completed": {"all": True},
            "input_manifests": run_manifests,
        },
    )
    _dump(
        atlas / "manifest.json",
        {
            "status": "complete",
            "source_run_manifest_sha256": source_hash,
            "fold_assignments_sha256": fold_hash,
        },
    )
    (atlas / "hypothesis_queue.csv").write_text("queue_rank,evidence_tier\n1,model_only\n")
    _ledger(
        atlas,
        (
            "manifest.json",
            "protocol.json",
            "validation.json",
            "models.json",
            "atlas_matrices.npz",
            "hypothesis_queue.csv",
        ),
    )
    return atlas


@pytest.fixture
def artifact_pair(tmp_path: Path) -> tuple[Path, Path]:
    source, fold_path, source_hash, fold_hash, checkpoints = _source_fixture(tmp_path)
    historical, matrices = _historical_fixture(
        tmp_path,
        source=source,
        fold_path=fold_path,
        fold_hash=fold_hash,
        checkpoints=checkpoints,
    )
    atlas = _atlas_fixture(
        tmp_path,
        source=source,
        fold_path=fold_path,
        source_hash=source_hash,
        fold_hash=fold_hash,
        checkpoints=checkpoints,
        historical_matrices=matrices,
    )
    return atlas, historical


def test_exact_overlap_audit_is_bounded_and_never_changes_queue(
    artifact_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    atlas, historical = artifact_pair
    queue_before = (atlas / "hypothesis_queue.csv").read_bytes()
    result = run_repeat_stability_audit(
        atlas, historical, tmp_path / "repeat_output"
    )
    output = tmp_path / "repeat_output"
    assert result["status"] == "complete"
    frame = pd.read_csv(output / "repeat_stability.csv")
    assert len(frame) == len(METHODS) * len(OVERLAP_CHANNELS) * len(OVERLAP_CONTEXTS) * len(LAGS)
    assert set(frame.method) == set(METHODS)
    assert set(frame.historical_method) == {
        "direct_importance_n256",
        "progressive_bridge_smc",
    }
    assert set(frame.context) == set(OVERLAP_CONTEXTS)
    assert "state_average" not in set(frame.context)
    assert np.allclose(frame.matrix_spearman, 1.0)
    assert np.allclose(frame.sign_agreement, 1.0)
    assert frame.rmse.gt(0).all()
    assert frame.scientific_estimand_exact.all()
    assert frame.loc[
        frame.method == "direct_importance", "response_estimator_configuration_exact"
    ].all()
    assert not frame.loc[
        frame.method == "progressive_bridge_smc", "response_estimator_configuration_exact"
    ].any()
    assert not frame.finite_mc_realization_exact.any()
    assert not frame.loc[
        frame.method == "direct_importance", "support_gate_configuration_exact"
    ].any()
    assert frame.loc[
        frame.method == "progressive_bridge_smc", "support_gate_configuration_exact"
    ].all()
    assert (atlas / "hypothesis_queue.csv").read_bytes() == queue_before
    assert result["queue_firewall"]["ranking_or_promotion_effect"] == "none"
    assert result["external_reference_inputs"] == []
    assert set(path.name for path in output.iterdir()) == {
        "repeat_stability.csv",
        "manifest.json",
        "REPORT.md",
        "checksums.sha256",
    }
    _verify_output_ledger(output)


def _verify_output_ledger(output: Path) -> None:
    rows = {}
    for line in (output / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        rows[name] = digest
    assert rows == {
        name: sha256(output / name)
        for name in ("repeat_stability.csv", "manifest.json", "REPORT.md")
        if (output / name).exists()
    } or rows == {
        name: sha256(output / name)
        for name in ("AUDIT_NOTE.md", "manifest.json")
        if (output / name).exists()
    }


def test_historical_matrix_transpose_is_detected_from_raw_reconstruction(
    artifact_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    atlas, historical = artifact_pair
    archive_path = historical / "aligned_four_sampler_lag_matrices.npz"
    with np.load(archive_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    key = "direct_importance_n256__endpoint_mean__baseline__worm_matrices"
    arrays[key] = arrays[key].transpose(0, 1, 3, 2)
    group_key = "direct_importance_n256__endpoint_mean__baseline__matrices"
    arrays[group_key] = arrays[group_key].transpose(0, 2, 1)
    np.savez_compressed(archive_path, **arrays)
    _ledger(
        historical,
        (
            "aligned_four_sampler_lag_matrices.npz",
            "validation.json",
            "input_checksums.json",
            "REPORT.md",
        ),
    )
    with pytest.raises(RuntimeError, match="raw-to-worm reconstruction"):
        run_repeat_stability_audit(atlas, historical, tmp_path / "bad_output")


def test_lineage_timing_mismatch_fails_closed_even_with_updated_checksum(
    artifact_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    atlas, historical = artifact_pair
    protocol = json.loads((atlas / "protocol.json").read_text())
    protocol["fps"] = 5.0
    _dump(atlas / "protocol.json", protocol)
    _ledger(
        atlas,
        (
            "manifest.json",
            "protocol.json",
            "validation.json",
            "models.json",
            "atlas_matrices.npz",
            "hypothesis_queue.csv",
        ),
    )
    with pytest.raises(RuntimeError, match="lineage mismatch: fps"):
        run_repeat_stability_audit(atlas, historical, tmp_path / "bad_output")


def test_valid_artifacts_without_reviewed_context_overlap_get_note_not_metrics(
    artifact_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    atlas, historical = artifact_pair
    archive_path = atlas / "atlas_matrices.npz"
    with np.load(archive_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["contexts"] = np.asarray(["state_average"])
    np.savez_compressed(archive_path, **arrays)
    _ledger(
        atlas,
        (
            "manifest.json",
            "protocol.json",
            "validation.json",
            "models.json",
            "atlas_matrices.npz",
            "hypothesis_queue.csv",
        ),
    )
    output = tmp_path / "no_overlap"
    result = run_repeat_stability_audit(atlas, historical, output)
    assert result["status"] == "no_exact_overlap"
    assert (output / "AUDIT_NOTE.md").is_file()
    assert not (output / "repeat_stability.csv").exists()
    _verify_output_ledger(output)


def test_missing_canonical_atlas_creates_no_audit_note(tmp_path: Path) -> None:
    output = tmp_path / "premature"
    with pytest.raises(FileNotFoundError, match="completed canonical atlas"):
        run_repeat_stability_audit(
            tmp_path / "missing_atlas",
            tmp_path / "missing_historical",
            output,
        )
    assert not output.exists()


def test_output_must_be_disjoint_from_canonical_atlas(
    artifact_pair: tuple[Path, Path]
) -> None:
    atlas, historical = artifact_pair
    with pytest.raises(RuntimeError, match="disjoint"):
        run_repeat_stability_audit(atlas, historical, atlas / "repeat")


def test_canonical_queue_checksum_tamper_fails_before_output(
    artifact_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    atlas, historical = artifact_pair
    (atlas / "hypothesis_queue.csv").write_text(
        "queue_rank,evidence_tier\n1,supported_exploratory\n"
    )
    output = tmp_path / "tampered"
    with pytest.raises(RuntimeError, match="checksum failed"):
        run_repeat_stability_audit(atlas, historical, output)
    assert not output.exists()
