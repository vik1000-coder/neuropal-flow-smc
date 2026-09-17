from __future__ import annotations

import numpy as np

from distributional_sid.closure_experiment import (
    CELLS,
    ClosureConfig,
    OneStepModel,
    _ar_truth,
    _rollout_features,
    run_cell_seed,
)


def test_ar_truth_separates_correct_and_short_history() -> None:
    config = ClosureConfig(horizons=(2, 4, 8), witnesses=("mean", "sin_1.0"))
    closed = _ar_truth(next(cell for cell in CELLS if cell.name == "ar2_closed"), config)
    short = _ar_truth(next(cell for cell in CELLS if cell.name == "ar2_history_short"), config)
    assert np.max(np.abs(closed["defect"])) < 1e-12
    assert abs(short["defect"][2, 0]) > abs(short["defect"][0, 0]) > 0.05


def test_rollout_pathwise_derivative_matches_common_noise_difference() -> None:
    model = OneStepModel("sine", np.array([0.0, 0.60, 0.60]), 0.30, 1)
    histories = np.array([[-0.3], [0.2], [0.8]])
    rng = np.random.default_rng(818)
    normals = rng.normal(size=(histories.shape[0], 2048, 4))
    _, derivative = _rollout_features(
        model, histories, (4,), ("mean", "sin_1.0"), normals
    )
    delta = 1e-5
    plus = histories.copy()
    minus = histories.copy()
    plus[:, 0] += delta
    minus[:, 0] -= delta
    value_plus, _ = _rollout_features(model, plus, (4,), ("mean", "sin_1.0"), normals)
    value_minus, _ = _rollout_features(model, minus, (4,), ("mean", "sin_1.0"), normals)
    finite = (value_plus - value_minus) / (2.0 * delta)
    assert np.max(np.abs(finite - derivative)) < 2e-7


def test_small_cell_seed_is_finite_and_has_registered_schema() -> None:
    config = ClosureConfig(
        stage="smoke",
        n_trajectories=8,
        trajectory_length=80,
        burn_in=100,
        anchors_per_trajectory=4,
        n_folds=2,
        horizons=(2,),
        witnesses=("mean", "sin_1.0"),
        rollout_draws=4,
        bootstrap_replicates=3,
    )
    frame = run_cell_seed(CELLS[0], config, 17)
    assert len(frame) == 2
    assert set(frame.witness) == {"mean", "sin_1.0"}
    assert np.all(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()))
    assert np.all(frame.fit_status == "ok")
