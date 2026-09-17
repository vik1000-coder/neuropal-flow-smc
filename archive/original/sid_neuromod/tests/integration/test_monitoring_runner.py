"""End-to-end monitoring runner: detect an injected conditional-law change."""
import json
from pathlib import Path

import numpy as np

from sid_neuromod.data.io import save_npz
from sid_neuromod.data.schema import NeuralDataset
from sid_neuromod.experiments.run_elegans_monitoring import run
from sid_neuromod.utils.config import Config


def _make_change_dataset(T=8000, fs=4.0, t_change=4000, seed=0):
    """One SV neuron whose emission scale jumps at t_change (variance-only change)."""
    rng = np.random.default_rng(seed)
    w = np.zeros(T)
    for k in range(1, T):
        w[k] = 0.9 * w[k - 1] + 0.4 * rng.standard_normal()
    eps = rng.standard_normal(T)
    scale = np.ones(T); scale[t_change:] = 1.6
    x0 = scale * np.exp(w / 2) * eps
    # a second, stationary neuron
    x1 = rng.standard_normal(T)
    X = np.column_stack([x0, x1])
    t = np.arange(T) / fs
    return NeuralDataset(X=X, timestamps_s=t, neuron_ids=np.array(["A", "B"]),
                         dataset_id="change_mock")


def test_monitoring_runner_detects_injected_change(tmp_path):
    ds = _make_change_dataset(T=8000, t_change=4000, seed=1)
    p = Path(tmp_path) / "change.npz"
    save_npz(p, ds)
    cfg = Config({
        "dataset": {"path": str(p)},
        "random_seed": 0,
        "target": {"mode": "delta", "horizon_s": 0.25},
        "features": {"timescales_s": [1, 2, 5, 10]},
        "monitoring": {"alpha": 0.01, "delta": 0.01},
    })
    # baseline = pre-change interval; monitor = spans the change
    report = run(cfg, baseline_interval=[0, 3800], monitor_interval=[3800, 8000],
                 out_dir=tmp_path, target=0)
    assert report["alarmed"], report
    assert report["dominant_channel"] in {"dispersion", "serial"}
    assert (Path(tmp_path) / "figures" / "eprocess.png").exists()
    saved = json.load(open(Path(tmp_path) / "monitoring_report.json"))
    assert saved["alarmed"]
