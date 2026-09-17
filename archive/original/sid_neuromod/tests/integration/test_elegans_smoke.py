"""Section 18.3: elegans smoke test on a tiny mock dataset."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sid_neuromod.data.io import make_mock_elegans, save_npz
from sid_neuromod.experiments.make_report import build_report
from sid_neuromod.experiments.run_elegans_fit import run
from sid_neuromod.utils.config import Config

REQUIRED_READOUT_COLS = {
    "target_id", "source_id", "timescale_s", "channel", "estimate",
    "standard_error", "z_score", "p_value", "q_value", "ci_low", "ci_high",
    "significant", "feature_name", "model_class",
}
REQUIRED_DIAG_KEYS = {
    "pit_mean", "pit_variance", "invalid_variance_fraction", "heldout_nll",
    "baseline_var_heldout_nll", "baseline_ridge_r2",
}


def test_elegans_smoke_pipeline_completes(tmp_path):
    cfg = Config({
        "dataset": {"path": "does/not/exist.npz"},
        "random_seed": 0,
        "splits": {"train_fraction": 0.5, "calibration_fraction": 0.2,
                   "debias_fraction": 0.1, "test_fraction": 0.2},
        "target": {"mode": "delta", "horizon_s": 0.5},
        "features": {"source_signals": ["raw_zscore"],
                     "timescales_s": [1, 2, 5, 10], "include_behavior": True},
        "model": {"ridge": 1e-3},
        "inference": {"alpha": 0.05, "n_mc_sup_t": 1500},
        "mock": {"T": 2500, "N": 5},
    })
    diag = run(cfg, tmp_path)

    # schemas valid
    readouts = pd.read_parquet(Path(tmp_path) / "readouts.parquet")
    assert REQUIRED_READOUT_COLS.issubset(set(readouts.columns))
    assert len(readouts) > 0
    saved_diag = json.load(open(Path(tmp_path) / "diagnostics.json"))
    assert REQUIRED_DIAG_KEYS.issubset(set(saved_diag.keys()))
    meta = json.load(open(Path(tmp_path) / "fit_metadata.json"))
    for key in ("dataset_id", "n_timepoints", "n_neurons", "timescales_s",
                "model_class", "train_interval", "test_interval"):
        assert key in meta

    # PIT calibration in a sane range on the mock
    assert 0.3 < diag["pit_mean"] < 0.7
    # figures generated
    assert (Path(tmp_path) / "figures").exists()

    # report builds and contains the required caveat language
    text = build_report(tmp_path)
    assert "predictive distributional effects, not direct structural synapses" in text
    assert "Limitations and caveats" in text


def test_no_future_leakage_in_targets(tmp_path):
    """The delta target uses only x[t+h]-x[t]; last h rows must be invalid (dropped)."""
    from sid_neuromod.features.history import make_target
    X = np.arange(20, dtype=float).reshape(-1, 1)
    Y, valid = make_target(X, horizon_steps=2, mode="delta")
    assert not valid[-2:].any()
    assert valid[:-2].all()
    # value check: Y[t] = X[t+2]-X[t] = 2 for the ramp
    assert np.allclose(Y[valid], 2.0)


def test_saved_dataset_roundtrip(tmp_path):
    ds = make_mock_elegans(T=500, N=3, seed=1)
    p = Path(tmp_path) / "d.npz"
    save_npz(p, ds)
    from sid_neuromod.data.io import load_npz
    ds2 = load_npz(p)
    assert ds2.T == ds.T and ds2.N == ds.N
    assert np.allclose(ds2.X, ds.X)
