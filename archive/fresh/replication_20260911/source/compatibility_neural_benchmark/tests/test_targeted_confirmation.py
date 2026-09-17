from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from compatibility_neural_benchmark.core import RepairedResponseConfig
from compatibility_neural_benchmark.progressive_smc import (
    _three_arm_future_responses,
    progressive_smc_targeted_confirmation,
)
from compatibility_neural_benchmark.targeted_confirmation import (
    COMMON_NOISE_DEFINITION,
    ConfirmationGroup,
    _manifest,
    _strict_provenance_match,
    load_candidate_groups,
    orient_source_horizon_target,
)


class NoiseAdapter:
    lag = 1
    device = torch.device("cpu")

    @staticmethod
    def sample_standardized_next(neural_history, stimulus_history, *, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(
            (len(neural_history), neural_history.shape[-1]), generator=generator
        )


class DirectedAdapter:
    lag = 1
    device = torch.device("cpu")

    @staticmethod
    def sample_standardized_next(neural_history, stimulus_history, *, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = torch.zeros(
            (len(neural_history), neural_history.shape[-1]), dtype=torch.float32
        )
        # Source neuron 1 is stochastic during repair. At the next step its
        # lagged state drives target neuron 0 and no other target.
        result[:, 1] = torch.randn(len(neural_history), generator=generator)
        result[:, 0] = neural_history[:, -1, 1]
        return result


def _config(*, particles: int, horizons: tuple[int, ...]) -> RepairedResponseConfig:
    return RepairedResponseConfig(
        history_frames=1,
        repair_frames=1,
        source_window_frames=1,
        source_lag_frames=0,
        horizon_frames=horizons,
        n_particles=particles,
        anchor_lambda=0.0,
        epsilon_iqr_fraction=0.4,
        min_ess=1,
        max_normalized_weight=1.0,
        sampling_chunk_size=128,
    )


def test_three_arm_future_common_noise_is_exact_for_identical_histories() -> None:
    config = _config(particles=16, horizons=(1, 2))
    repaired_history = torch.zeros((2, 16, 1, 3), dtype=torch.float32)
    repaired_stimulus = torch.zeros((2, 16, 1), dtype=torch.float32)
    result, n_future = _three_arm_future_responses(
        NoiseAdapter(),
        repaired_history,
        repaired_stimulus,
        np.zeros((1, 3), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        cut_time=2,
        selected_source_count=1,
        target_count=3,
        thresholds=np.zeros(3, dtype=np.float32),
        config=config,
        seed=67,
        future_branch_factor=1,
        endpoint_quantiles=(0.25, 0.5, 0.75),
    )
    assert n_future == 16
    for key, value in result.items():
        if key.startswith("response_") or key.startswith("distance_"):
            np.testing.assert_array_equal(value, 0.0)


def test_three_arm_common_noise_gives_zero_effect_when_transition_ignores_history() -> None:
    result = progressive_smc_targeted_confirmation(
        NoiseAdapter(),
        np.zeros((7, 3), dtype=np.float32),
        np.zeros((7, 2), dtype=np.float32),
        cut_time=3,
        projection=np.eye(3, dtype=np.float32),
        source_low=np.zeros(3, dtype=np.float32),
        source_high=np.zeros(3, dtype=np.float32),
        source_iqr=np.ones(3, dtype=np.float32),
        thresholds=np.zeros(3, dtype=np.float32),
        source_indices=np.asarray([1]),
        config=_config(particles=64, horizons=(1, 2)),
        seed=71,
        branch_factor=2,
        future_branch_factor=1,
        endpoint_quantiles=(0.25, 0.5, 0.75),
    )
    assert result["arm_endpoint_mean"].shape == (3, 1, 2, 3)
    assert result["arm_endpoint_quantile"].shape == (3, 1, 2, 3, 3)
    for key, value in result.items():
        if key.startswith("response_") or key.startswith("distance_"):
            np.testing.assert_allclose(value, 0.0, atol=1e-7)


def test_targeted_direction_is_placed_at_declared_target_and_source() -> None:
    result = progressive_smc_targeted_confirmation(
        DirectedAdapter(),
        np.zeros((6, 3), dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        cut_time=3,
        projection=np.eye(3, dtype=np.float32),
        source_low=np.full(3, -0.8, dtype=np.float32),
        source_high=np.full(3, 0.8, dtype=np.float32),
        source_iqr=np.ones(3, dtype=np.float32),
        thresholds=np.zeros(3, dtype=np.float32),
        source_indices=np.asarray([1]),
        config=_config(particles=512, horizons=(1,)),
        seed=73,
        branch_factor=2,
        future_branch_factor=1,
    )
    effect = result["response_high_low_endpoint_mean"]
    assert effect.shape == (1, 1, 3)
    assert effect[0, 0, 0] > 0.5
    np.testing.assert_allclose(effect[0, 0, 2], 0.0, atol=1e-7)

    oriented = orient_source_horizon_target(
        effect, source_count=1, horizon_count=1, target_count=3
    )
    assert oriented.shape == (1, 3, 1)
    assert oriented[0, 0, 0] == effect[0, 0, 0]


def test_candidate_queue_resolves_only_selected_sources(tmp_path: Path) -> None:
    queue = tmp_path / "hypothesis_queue.csv"
    pd.DataFrame(
        [
            {
                "source_neuron": "B",
                "source_lag_frames": 4,
                "horizon_frames": 2,
                "phase": "onset",
                "selected": True,
            },
            {
                "source_neuron": "D",
                "source_lag_frames": 4,
                "horizon_frames": 8,
                "phase": "active",
                "selected": True,
            },
            {
                "source_neuron": "A",
                "source_lag_frames": 1,
                "horizon_frames": 1,
                "phase": "baseline",
                "selected": False,
            },
        ]
    ).to_csv(queue, index=False)
    groups = load_candidate_groups(queue, ("A", "B", "C", "D"))
    assert len(groups) == 1
    assert groups[0].source_lag_frames == 4
    assert groups[0].source_indices == (1, 3)
    assert groups[0].horizon_frames == (2, 8)
    assert groups[0].phases == ("onset", "active")


def test_candidate_queue_maps_atlas_context_to_required_episode_phases(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "hypothesis_queue.csv"
    pd.DataFrame(
        [
            {
                "source_neuron": "B",
                "source_lag_frames": 4,
                "horizon_frames": 2,
                "context": "butanone_onset_minus_baseline",
                "selected": True,
            }
        ]
    ).to_csv(queue, index=False)
    groups = load_candidate_groups(queue, ("A", "B", "C"))
    assert groups[0].phases == ("baseline", "onset")


def test_strict_resume_provenance_rejects_changed_selection(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.npz"
    np.savez_compressed(
        archive_path,
        status=np.asarray("complete"),
        base_seed=np.asarray(20260829),
        requested_device=np.asarray("mps"),
        selection_fingerprint=np.asarray("frozen-selection"),
        selected_source_indices=np.asarray([1, 3], dtype=np.int16),
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        _strict_provenance_match(
            archive,
            scalar_expected={
                "status": "complete",
                "base_seed": 20260829,
                "requested_device": "mps",
                "selection_fingerprint": "frozen-selection",
            },
            array_expected={
                "selected_source_indices": np.asarray([1, 3], dtype=np.int16)
            },
        )
        with pytest.raises(RuntimeError, match="selection_fingerprint differs"):
            _strict_provenance_match(
                archive,
                scalar_expected={
                    "status": "complete",
                    "selection_fingerprint": "different-selection",
                },
                array_expected={
                    "selected_source_indices": np.asarray([1, 3], dtype=np.int16)
                },
            )
        with pytest.raises(RuntimeError, match="base_seed differs"):
            _strict_provenance_match(
                archive,
                scalar_expected={"base_seed": 20260830},
                array_expected={},
            )
        with pytest.raises(RuntimeError, match="requested_device differs"):
            _strict_provenance_match(
                archive,
                scalar_expected={"requested_device": "cpu"},
                array_expected={},
            )
        with pytest.raises(RuntimeError, match="lacks provenance fields"):
            _strict_provenance_match(
                archive,
                scalar_expected={"checkpoint_sha256": "missing"},
                array_expected={},
            )


def test_v2_manifest_uses_the_archive_common_noise_contract(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    source_run.mkdir()
    (source_run / "manifest.json").write_text("{}\n")
    fold_file = tmp_path / "folds.csv"
    fold_file.write_text("worm_index,worm_id,outer_fold\n0,w0,0\n")
    queue = tmp_path / "queue.csv"
    queue.write_text("source_neuron\nB\n")
    cohort = SimpleNamespace(
        neurons=("A", "B"),
        stimulus_schema_version="synthetic-v1",
        stimulus_schema_fingerprint="synthetic-fingerprint",
    )
    group = ConfirmationGroup(
        source_lag_frames=1,
        source_indices=(1,),
        horizon_frames=(1,),
        phases=("onset",),
        selection_fingerprint="selection",
    )
    args = SimpleNamespace(
        model_id="flow",
        checkpoint_phase="test",
        folds=(0,),
        seeds=(11,),
        base_seed=20260829,
        device="cpu",
        particles=128,
        history_lag=8,
        source_window_frames=1,
        endpoint_quantiles=(0.25, 0.5, 0.75),
        progressive_branch_factor=2,
        progressive_future_branch_factor=2,
        min_ess=24.0,
    )
    manifest = _manifest(
        source_run=source_run,
        fold_file=fold_file,
        hypothesis_queue=queue,
        cohort=cohort,
        groups=(group,),
        args=args,
    )
    assert manifest["common_noise_definition"] == COMMON_NOISE_DEFINITION
    assert manifest["minimum_effective_sample_size"] == 24.0
    assert manifest["maximum_normalized_weight"] == 0.20
    assert manifest["minimum_achieved_source_fraction"] == 0.25
