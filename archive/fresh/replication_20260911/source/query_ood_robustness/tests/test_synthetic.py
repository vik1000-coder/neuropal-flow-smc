from __future__ import annotations

import numpy as np

from query_ood_robustness.oracles import oracle_controls, oracle_soft_effect
from query_ood_robustness.synthetic import (
    FitzHughNagumoNetwork,
    NonlinearVAR,
    VAR_MECHANISMS,
    fhn_solver_error,
)


def test_all_analytic_oracles_are_reproducible_and_finite() -> None:
    for control in oracle_controls():
        one = control.sample(64, 17, 0.4)
        two = control.sample(64, 17, 0.4)
        np.testing.assert_array_equal(one, two)
        assert one.shape == (64, 4)
        assert np.isfinite(one).all()


def test_oracle_reference_converges_and_has_positive_mass() -> None:
    effect, mcse, mass = oracle_soft_effect(
        oracle_controls()[1], history=0.3, low_target=0.0,
        high_target=2.0, bandwidth=0.35, n_reference=100_000, seed=91,
    )
    assert effect > 0.8
    assert mcse < 0.03
    assert 0 < mass < 1


def test_var_stability_and_dataset_shapes() -> None:
    for index, mechanism in enumerate(VAR_MECHANISMS):
        system = NonlinearVAR(100 + index, mechanism)
        assert system.spectral_radius <= 0.6200001
        dataset = system.dataset(900)
        assert dataset.train_h.shape[1] == 7
        assert dataset.train_y.shape[1] == 6
        draw = dataset.oracle_transition(128, 44)
        assert draw.shape == (128, 6)
        assert np.isfinite(draw).all()
        conditioned = dataset.oracle_conditioned(dataset.anchor_h, 32, 45)
        assert conditioned.shape == (32, 6)


def test_fhn_solver_and_paired_intervention() -> None:
    assert fhn_solver_error(12, duration_frames=6) < 0.03
    dataset = FitzHughNagumoNetwork(12).dataset(700)
    one = dataset.pulse_truth(0.5, 2, 128, 771)
    two = dataset.pulse_truth(0.5, 2, 128, 771)
    np.testing.assert_array_equal(one, two)
    assert one.shape == (8,)
    assert np.max(np.abs(one)) > 0


def test_fhn_observation_transition_reproducible() -> None:
    dataset = FitzHughNagumoNetwork(19).dataset(650)
    assert dataset.train_h.shape[1] == 10
    one = dataset.oracle_transition(32, 1001)
    two = dataset.oracle_transition(32, 1001)
    np.testing.assert_array_equal(one, two)
    assert one.shape == (32, 8)
    conditioned = dataset.oracle_conditioned(dataset.anchor_h, 16, 1002)
    assert conditioned.shape == (16, 8)
