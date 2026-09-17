"""Regression tests: pin the headline synthetic metrics within tolerances."""
import json
from pathlib import Path

import numpy as np
import pytest

from sid_neuromod.experiments.run_synthetic import run_hidden_thermostat
from sid_neuromod.utils.config import Config

EXP = Path(__file__).parent / "expected_metrics"


def test_hidden_thermostat_regression(tmp_path):
    expected = json.load(open(EXP / "hidden_thermostat_seed0.json"))
    cfg = Config({
        "experiment": {"kind": "hidden_thermostat", "T": 200000, "a": 0.9,
                       "b": 0.4, "n_lags": 12},
        "random_seed": 0, "model": {"ridge": 1e-6},
        "inference": {"alpha": 0.05, "n_mc_sup_t": 4000},
    })
    diag = run_hidden_thermostat(cfg, tmp_path)
    assert diag["variance_kernel_corr"] >= expected["variance_kernel_corr_min"]
    assert diag["mean_r2"] < expected["mean_r2_max"]
    assert diag["n_significant_gain_lags"] >= expected["n_significant_gain_lags"] - 1
    assert diag["sigma_ledger_max_deviation"] < expected["sigma_ledger_max_deviation_max"]
    # oracle values are analytic constants — pin them tightly
    assert abs(diag["contraction_oracle"] - expected["contraction_oracle"]) < 1e-3
    assert abs(diag["gain_c1_oracle"] - expected["gain_c1_oracle"]) < 1e-3


@pytest.mark.slow
def test_change_detection_regression():
    from sid_neuromod.experiments.run_synthetic import monitor_scenario
    expected = json.load(open(EXP / "change_detection_seed0.json"))
    n = 20
    null_fa = sum(monitor_scenario("null", seed=1000 * r, T=22000)[0].alarmed
                  for r in range(n)) / n
    scale = sum(monitor_scenario("scale", seed=1000 * r, T=22000)[0].alarmed
                for r in range(n)) / n
    assert null_fa <= expected["null_false_alarm_rate_max"]
    assert scale >= expected["scale_detection_power_min"]
