from pathlib import Path
import runpy

import numpy as np


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "run_tier2_dynamic.py",
    run_name="tier2_dynamic_test",
)


def finite_derivative(dgp, histories, channel, step=1e-5):
    result = np.zeros_like(histories)
    for lag in range(histories.shape[1]):
        plus = histories.copy()
        minus = histories.copy()
        plus[:, lag] += step
        minus[:, lag] -= step
        result[:, lag] = (dgp.target(plus, channel) - dgp.target(minus, channel)) / (
            2.0 * step
        )
    return result


def test_hidden_modulator_oracle_derivative_and_overcomplete_null():
    dgp = MODULE["HiddenModulatorDGP"](20)
    histories = np.random.default_rng(44).normal(size=(8, 20))
    analytic = dgp.lag_effect(histories, "variance")
    numeric = finite_derivative(dgp, histories, "variance")
    assert np.max(np.abs(analytic - numeric)) < 2e-7
    assert np.max(np.abs(analytic[:, 12:])) == 0.0
    assert np.max(np.abs(dgp.lag_effect(histories, "mean"))) == 0.0


def test_observation_filter_oracle_derivative_and_null():
    dgp = MODULE["ObservationFilterDGP"](0.65, 0.25, False)
    histories = np.random.default_rng(45).normal(size=(8, 7))
    analytic = dgp.lag_effect(histories, "variance")
    numeric = finite_derivative(dgp, histories, "variance")
    assert np.max(np.abs(analytic - numeric)) < 2e-7
    null = MODULE["ObservationFilterDGP"](0.65, 0.25, True)
    assert np.max(np.abs(null.lag_effect(histories, "variance"))) == 0.0
    assert np.max(np.abs(null.lag_effect(histories, "mean"))) == 0.0
