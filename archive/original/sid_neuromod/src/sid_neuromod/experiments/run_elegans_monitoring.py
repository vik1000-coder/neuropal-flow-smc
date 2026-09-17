r"""Sequential monitoring of a C. elegans recording (Sections 11, 15).

Fits the conditional density on a baseline interval, then monitors PIT channels over a
later interval, reporting the e-process trace, alarm time, dominant channel, and a
'what changed' pre/post readout delta table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.io import load_dataset, make_mock_elegans
from ..data.preprocessing import zscore_train
from ..features.design import build_design
from ..features.history import make_target
from ..models.quadratic_score import QuadraticScoreMatcher
from ..monitoring.changepoint import detect_change
from ..monitoring.channels import dispersion_channel, serial_channel
from ..monitoring.eprocess import deadband_epsilon
from ..monitoring.pit import gaussian_pit
from ..readouts.mean_gain_tail import center_history, readout_at
from ..utils.config import load_config
from ..utils.logging import get_logger

log = get_logger()


def run(cfg, baseline_interval, monitor_interval, out_dir, target=0):
    out = Path(out_dir); (out / "figures").mkdir(parents=True, exist_ok=True)
    ds = _load(cfg)
    b0, b1 = baseline_interval
    m0, m1 = monitor_interval
    train_slice = slice(b0, b1)
    Xz, _ = zscore_train(ds.X, train_slice)
    timescales = list(cfg.get_path("features.timescales_s", [1, 2, 5, 10, 20, 45]))
    horizon_steps = max(1, int(round(float(cfg.get_path("target.horizon_s", 1.0))
                                     / ds.median_frame_s)))
    Y_all, valid = make_target(Xz, horizon_steps, cfg.get_path("target.mode", "delta"))

    design = build_design(Xz, ds.timestamps_s, ds.neuron_ids,
                          timescales_s=timescales, train_slice=train_slice, center=True)
    Psi = design.Psi
    yi = Y_all[:, target]
    base = np.zeros(ds.T, bool); base[b0:b1] = True; base &= valid
    mon = np.zeros(ds.T, bool); mon[m0:m1] = True; mon &= valid

    model = QuadraticScoreMatcher(sigma=0.0, ridge=1e-4)
    res = model.fit(yi[base], Psi[base])
    mu, var = model.moments(Psi[mon])
    u = gaussian_pit(yi[mon], mu, var)
    # debias offset from tail of baseline
    dslice = slice(max(b0, b1 - (b1 - b0) // 5), b1)
    dmask = np.zeros(ds.T, bool); dmask[dslice] = True; dmask &= valid
    mud, vard = model.moments(Psi[dmask])
    ud = gaussian_pit(yi[dmask], mud, vard)
    channels = {
        "dispersion": dispersion_channel(u) - dispersion_channel(ud).mean(),
        "serial": serial_channel(u) - serial_channel(ud).mean(),
    }
    alpha = float(cfg.get_path("monitoring.alpha", 0.01))
    eps = deadband_epsilon(int(dmask.sum()), n_channels=len(channels),
                           delta=float(cfg.get_path("monitoring.delta", 0.01)))
    cr = detect_change(channels, alpha=alpha, eps=eps)

    report = {
        "alarmed": bool(cr.alarmed),
        "alarm_time_index": cr.alarm_time,
        "dominant_channel": cr.dominant_channel,
        "lower_ci_time": cr.lower_ci_time,
        "upper_ci_time": cr.upper_ci_time,
        "alpha": alpha, "deadband_eps": eps,
        "baseline_interval": [b0, b1], "monitor_interval": [m0, m1],
    }
    with open(out / "monitoring_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    from ..viz.plots import eprocess_trace
    eprocess_trace(cr.e_trace, 1.0 / alpha, out / "figures" / "eprocess.png",
                   alarm=cr.alarm_time, title="Monitoring e-process")
    log.info("monitoring: alarmed=%s at %s (channel=%s)",
             cr.alarmed, cr.alarm_time, cr.dominant_channel)
    return report


def _load(cfg):
    path = cfg.get_path("dataset.path")
    if path and Path(path).exists():
        return load_dataset(path)
    return make_mock_elegans(T=int(cfg.get_path("mock.T", 4000)),
                             N=int(cfg.get_path("mock.N", 6)),
                             seed=int(cfg.get("random_seed", 0)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Monitor a C. elegans recording")
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline-interval", nargs=2, type=int, required=True)
    ap.add_argument("--monitor-interval", nargs=2, type=int, required=True)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_dir = args.out or cfg.get_path("outputs.output_dir", "output/elegans/monitoring")
    return run(cfg, args.baseline_interval, args.monitor_interval, out_dir, args.target)


if __name__ == "__main__":
    main()
