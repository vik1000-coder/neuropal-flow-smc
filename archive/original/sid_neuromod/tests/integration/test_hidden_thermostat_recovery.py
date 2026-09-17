"""Section 18.1: hidden thermostat recovery (E1)."""
import json
from pathlib import Path

import numpy as np

from sid_neuromod.experiments.run_synthetic import run_hidden_thermostat
from sid_neuromod.utils.config import Config


def test_hidden_thermostat_recovery(tmp_path):
    cfg = Config({
        "experiment": {"kind": "hidden_thermostat", "T": 200000, "a": 0.9,
                       "b": 0.4, "n_lags": 12},
        "random_seed": 0,
        "model": {"ridge": 1e-6},
        "inference": {"alpha": 0.05, "n_mc_sup_t": 5000},
    })
    diag = run_hidden_thermostat(cfg, tmp_path)

    # Pass criteria (Section 18.1)
    assert diag["mean_r2"] < 1e-3, f"mean_r2={diag['mean_r2']}"
    assert diag["variance_kernel_corr"] > 0.90, diag["variance_kernel_corr"]
    assert diag["n_significant_gain_lags"] >= 8, diag["n_significant_gain_lags"]
    # sigma-ledger consistency (Section 12.1: < 20%)
    assert diag["sigma_ledger_max_deviation"] < 0.20

    # artifacts
    assert (Path(tmp_path) / "diagnostics.json").exists()
    assert (Path(tmp_path) / "readouts.parquet").exists()
    assert (Path(tmp_path) / "figures" / "kernel_recovery.png").exists()
    saved = json.load(open(Path(tmp_path) / "diagnostics.json"))
    assert saved["variance_kernel_corr"] > 0.90
