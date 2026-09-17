import numpy as np

from empirical_sid.dgps import DynamicStochasticGainDGP, SmoothTiltDGP, TILT_DEFINITIONS
from empirical_sid.oracles import run_all_oracle_checks


def test_signed_tilts_are_positive_normalized_and_orthogonal():
    for mechanism in TILT_DEFINITIONS:
        dgp = SmoothTiltDGP(mechanism)
        assert np.min(1.0 + dgp.t(1.0) * dgp.psi) > 0.0
        assert abs(np.sum(dgp.weights * dgp.density_grid(0.4)) - 1.0) < 1e-10
        assert max(abs(value) for value in dgp.orthogonality.values()) < 2e-10


def test_all_stage_zero_oracle_checks_pass():
    checks, _ = run_all_oracle_checks([1.0, 0.1, 0.01, 0.001, 0.0001, 0.0])
    failed = [check for check in checks if not check.passed]
    assert not failed, failed


def test_stochastic_gain_mean_is_exactly_blind():
    dgp = DynamicStochasticGainDGP()
    histories = dgp.sample_histories(32, np.random.default_rng(4))
    assert np.max(np.abs(dgp.lag_effect(histories, "mean"))) == 0.0
