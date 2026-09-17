from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from compatibility_neural_benchmark.core import RepairedResponseConfig
from compatibility_neural_benchmark.progressive_smc import (
    progressive_smc_repaired_responses,
)
from compatibility_neural_benchmark.targeted_sampling_nulls import (
    ARCHIVE_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    _candidate_archive_arrays,
    context_phases,
    load_selected_cells,
    quiet_pseudo_plan,
    sampling_subseed,
)
from conditional_neural_benchmark.data import StimulusSchedule


class DirectedNoiseAdapter:
    lag = 2
    device = torch.device("cpu")

    @staticmethod
    def sample_standardized_next(neural_history, stimulus_history, *, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            (len(neural_history), neural_history.shape[-1]), generator=generator
        )
        result = 0.4 * neural_history[:, -1] + 0.2 * noise
        result[:, 0] += neural_history[:, -1, 1]
        return result


def _config() -> RepairedResponseConfig:
    return RepairedResponseConfig(
        history_frames=2,
        repair_frames=3,
        source_window_frames=2,
        source_lag_frames=1,
        horizon_frames=(1, 2),
        n_particles=16,
        anchor_lambda=0.0,
        epsilon_iqr_fraction=0.5,
        min_ess=1.0,
        max_normalized_weight=1.0,
        sampling_chunk_size=128,
    )


def test_particle_endpoint_opt_in_preserves_aggregate_response() -> None:
    result = progressive_smc_repaired_responses(
        DirectedNoiseAdapter(),
        np.zeros((14, 3), dtype=np.float32),
        np.zeros(14, dtype=np.float32),
        cut_time=7,
        projection=np.eye(3, dtype=np.float32),
        source_low=np.full(3, -0.5, dtype=np.float32),
        source_high=np.full(3, 0.5, dtype=np.float32),
        source_iqr=np.ones(3, dtype=np.float32),
        thresholds=np.zeros(3, dtype=np.float32),
        config=_config(),
        seed=101,
        branch_factor=2,
        future_branch_factor=2,
        source_indices=np.asarray([1]),
        particle_target_indices=np.asarray([0, 2]),
    )
    endpoint = result["particle_endpoint"]
    assert endpoint.shape == (2, 1, 2, 32, 2)
    np.testing.assert_array_equal(result["particle_target_index"], [0, 2])
    np.testing.assert_array_equal(
        result["particle_future_parent_index"], np.repeat(np.arange(16), 2)
    )
    assert result["particle_repair_ancestor_id"].shape == (2, 1, 16)
    derived = endpoint[1].mean(axis=2) - endpoint[0].mean(axis=2)
    np.testing.assert_allclose(
        derived,
        result["response_endpoint_mean"][:, :, [0, 2]],
        rtol=1e-6,
        atol=1e-7,
    )


def test_equal_midpoint_targets_are_exactly_paired() -> None:
    midpoint = np.full(3, 0.25, dtype=np.float32)
    result = progressive_smc_repaired_responses(
        DirectedNoiseAdapter(),
        np.zeros((14, 3), dtype=np.float32),
        np.zeros(14, dtype=np.float32),
        cut_time=7,
        projection=np.eye(3, dtype=np.float32),
        source_low=midpoint,
        source_high=midpoint,
        source_iqr=np.ones(3, dtype=np.float32),
        thresholds=np.zeros(3, dtype=np.float32),
        config=_config(),
        seed=103,
        branch_factor=2,
        future_branch_factor=2,
        source_indices=np.asarray([1]),
        particle_target_indices=np.asarray([0]),
    )
    np.testing.assert_array_equal(
        result["particle_endpoint"][0], result["particle_endpoint"][1]
    )
    np.testing.assert_array_equal(result["diagnostic_target_gap"], 0.0)
    np.testing.assert_array_equal(result["diagnostic_achieved_gap"], 0.0)


def test_selected_cells_group_only_exact_horizon_and_phase_sets(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    pd.DataFrame(
        [
            {
                "queue_rank": 1,
                "source_neuron": "A",
                "target_neuron": "C",
                "source_index": 0,
                "target_index": 2,
                "source_lag_frames": 4,
                "horizon_frames": 32,
                "context": "baseline",
                "channel": "endpoint_wasserstein1",
                "selection_origin": "strong_primary",
                "run_sampling_nulls": True,
            },
            {
                "queue_rank": 2,
                "source_neuron": "B",
                "target_neuron": "D",
                "source_index": 1,
                "target_index": 3,
                "source_lag_frames": 4,
                "horizon_frames": 1,
                "context": "onset_minus_baseline",
                "channel": "endpoint_mean",
                "selection_origin": "lag_sensitivity",
                "run_sampling_nulls": True,
            },
        ]
    ).to_csv(queue, index=False)
    cells, groups = load_selected_cells(queue, ("A", "B", "C", "D"))
    assert len(cells) == 2
    assert len(groups) == 2
    assert {(group.horizon, group.phases) for group in groups} == {
        (32, ("baseline",)),
        (1, ("baseline", "onset")),
    }
    assert tuple(cell.selection_origin for cell in cells) == (
        "strong_primary",
        "lag_sensitivity",
    )
    assert tuple(cell.to_dict()["selection_origin"] for cell in cells) == (
        "strong_primary",
        "lag_sensitivity",
    )
    assert {
        group.horizon: [
            cell["selection_origin"] for cell in group.to_dict()["cells"]
        ]
        for group in groups
    } == {1: ["lag_sensitivity"], 32: ["strong_primary"]}
    archived_origins = {
        group.horizon: _candidate_archive_arrays(group)[
            "candidate_selection_origin"
        ].tolist()
        for group in groups
    }
    assert archived_origins == {1: ["lag_sensitivity"], 32: ["strong_primary"]}


def test_selected_cells_require_nonempty_selection_origin(tmp_path: Path) -> None:
    base = {
        "source_neuron": "A",
        "target_neuron": "B",
        "source_lag_frames": 1,
        "horizon_frames": 2,
        "context": "baseline",
        "run_sampling_nulls": True,
    }
    queue = tmp_path / "missing.csv"
    pd.DataFrame([base]).to_csv(queue, index=False)
    with pytest.raises(RuntimeError, match="lacks columns.*selection_origin"):
        load_selected_cells(queue, ("A", "B"))

    queue = tmp_path / "empty.csv"
    pd.DataFrame([{**base, "selection_origin": ""}]).to_csv(queue, index=False)
    with pytest.raises(RuntimeError, match="empty selection_origin"):
        load_selected_cells(queue, ("A", "B"))


def test_selection_origin_archive_change_is_schema_versioned() -> None:
    assert ARCHIVE_SCHEMA_VERSION == "prediction_atlas_sampling_null_v2"
    assert MANIFEST_SCHEMA_VERSION == "prediction_atlas_sampling_null_manifest_v2"


def _schedule() -> StimulusSchedule:
    return StimulusSchedule(
        worm_id="w0",
        strain="synthetic",
        source_recording="synthetic.mat",
        native_fps=4.0,
        analysis_fps=4.0,
        stimulus_names=("butanone", "pentanedione", "nacl"),
        event_intervals_seconds=((60.5, 70.5), (120.5, 130.5), (180.5, 190.5)),
        chemical_code_by_event=(1, 2, 3),
        chemical_name_by_event=("butanone", "pentanedione", "nacl"),
        resampling_provenance="none",
    )


def test_quiet_pseudo_plan_preserves_complete_zero_stimulus_windows() -> None:
    stimulus = np.zeros(960, dtype=np.float32)
    for start, stop in _schedule().event_intervals_seconds:
        stimulus[int(round(start * 4)) : int(round(stop * 4))] = 1.0
    plan = quiet_pseudo_plan(
        _schedule(),
        phases=("baseline", "onset"),
        history_frames=80,
        source_lag_frames=4,
        source_window_frames=4,
        horizon_frames=32,
        replicates=4,
        stimulus=stimulus,
    )
    assert plan.boundary_times.shape == (3, 4)
    assert plan.cut_times.shape == (2, 3, 4)
    assert plan.quiet_verified.all()
    np.testing.assert_array_equal(plan.boundary_times[1], [429, 435, 440, 446])
    assert np.all(np.diff(plan.boundary_times, axis=1) > 0)


def test_quiet_pseudo_plan_rejects_contaminated_history() -> None:
    stimulus = np.zeros(960, dtype=np.float32)
    stimulus[:] = 1.0
    with pytest.raises(RuntimeError, match="contain observed stimulus"):
        quiet_pseudo_plan(
            _schedule(),
            phases=("baseline",),
            history_frames=80,
            source_lag_frames=4,
            source_window_frames=4,
            horizon_frames=32,
            replicates=2,
            stimulus=stimulus,
        )


def test_sampling_subseeds_are_reproducible_and_namespaced() -> None:
    canonical = 12345
    assert sampling_subseed(canonical, "observed", 0) == canonical
    values = {
        sampling_subseed(canonical, "observed", 1),
        sampling_subseed(canonical, "midpoint", 0),
        sampling_subseed(canonical, "midpoint", 1),
    }
    assert len(values) == 3
    assert sampling_subseed(canonical, "midpoint", 1) == sampling_subseed(
        canonical, "midpoint", 1
    )


def test_context_phases_fails_closed_outside_baseline_onset() -> None:
    assert context_phases("butanone_onset_minus_baseline") == (
        "baseline",
        "onset",
    )
    with pytest.raises(RuntimeError, match="support baseline/onset"):
        context_phases("state_average")
