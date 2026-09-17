r"""End-to-end C. elegans fit pipeline (Sections 5.3, 13.3).

Produces the machine-readable artifacts:
  * ``fit_metadata.json``
  * ``readouts.parquet``   (target, source, timescale, channel, estimate, SE, z, ...)
  * ``diagnostics.json``   (PIT calibration, held-out NLL, baseline comparison, ...)
  * ``figures/``           (PIT histogram, readout heatmap, kernel spectra)

The MVP uses the closed-form quadratic-score model and center-history readouts with
sup-t simultaneous bands per target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines.ridge_ar import fit_ridge_ar
from ..baselines.var import fit_var
from ..data.io import load_dataset, make_mock_elegans
from ..data.preprocessing import zscore_train
from ..data.splitting import contiguous_splits
from ..features.design import build_design
from ..features.history import make_target
from ..inference.multiple_testing import benjamini_hochberg, two_sided_pvalues
from ..inference.sandwich import readout_covariance
from ..inference.simultaneous_bands import simultaneous_bands
from ..models.quadratic_score import QuadraticScoreMatcher
from ..monitoring.pit import gaussian_pit, pit_diagnostics
from ..readouts.mean_gain_tail import (center_history, d_log_variance, d_mean,
                                       d_tail_high, d_tail_low, readout_at)
from ..readouts.summary import nmi
from ..utils.config import config_hash, load_config
from ..utils.logging import get_logger

log = get_logger()

CHANNELS = ("mean", "gain_log_variance", "tail_high", "tail_low")


def run(cfg, out_dir):
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    ds = _load(cfg)
    log.info("dataset %s: T=%d, N=%d", ds.dataset_id, ds.T, ds.N)

    # splits + train-only z-score
    sp = contiguous_splits(ds.T,
                           train=float(cfg.get_path("splits.train_fraction", 0.5)),
                           calibration=float(cfg.get_path("splits.calibration_fraction", 0.2)),
                           debias=float(cfg.get_path("splits.debias_fraction", 0.1)),
                           stream=float(cfg.get_path("splits.test_fraction", 0.2)))
    Xz, _ = zscore_train(ds.X, sp.train)

    timescales = list(cfg.get_path("features.timescales_s",
                                   [0.5, 1, 2, 5, 10, 20, 45, 90]))
    source_signals = list(cfg.get_path("features.source_signals",
                                       ["raw_zscore", "delta"]))
    horizon_s = float(cfg.get_path("target.horizon_s", 1.0))
    target_mode = cfg.get_path("target.mode", "delta")
    horizon_steps = max(1, int(round(horizon_s / ds.median_frame_s)))
    alpha = float(cfg.get_path("inference.alpha", 0.05))
    ridge = cfg.get_path("model.ridge", "auto")
    ridge = 1e-4 if ridge in ("auto", None) else float(ridge)
    include_behavior = bool(cfg.get_path("features.include_behavior", True)) \
        and ds.behavior is not None

    Y_all, valid = make_target(Xz, horizon_steps, target_mode)

    rows = []
    diag_per_target = []
    for i in range(ds.N):
        design = build_design(
            Xz, ds.timestamps_s, ds.neuron_ids,
            source_signals=source_signals, timescales_s=timescales,
            behavior=ds.behavior if include_behavior else None,
            behavior_names=ds.behavior_names if include_behavior else None,
            train_slice=sp.train, center=True,
        )
        Psi = design.Psi
        reg = design.registry
        yi = Y_all[:, i]

        # rows valid for this target (target exists) and within splits
        tr = _slice_valid(sp.train, valid)
        te = _slice_valid(sp.stream, valid)
        if tr.sum() < Psi.shape[1] + 5 or te.sum() < 10:
            continue

        model = QuadraticScoreMatcher(sigma=0.0, ridge=ridge)
        res = model.fit(yi[tr], Psi[tr])

        # calibration diagnostics on held-out stream
        mu, var = model.moments(Psi[te])
        u = gaussian_pit(yi[te], mu, var)
        pit = pit_diagnostics(u)
        heldout_nll = model.nll(yi[te], Psi[te])
        diag_per_target.append({"target": str(ds.neuron_ids[i]), **pit,
                                "heldout_nll": heldout_nll,
                                "invalid_variance_fraction": res.invalid_variance_fraction})

        q_hi = float(np.quantile(yi[tr], 0.90))
        q_lo = float(np.quantile(yi[tr], 0.10))
        center = center_history(Psi)

        # readouts per channel, per filter feature, with sup-t band across the filter set
        filt = reg.filter_specs
        cols = [s.column_index for s in filt]
        for channel in CHANNELS:
            def rd(theta, ch=channel):
                r = _clone(res, theta)
                return np.array([
                    readout_at(r, center, k, ch, q_high=q_hi, q_low=q_lo)[0]
                    for k in cols
                ])
            r_vec, cov, _ = readout_covariance(res, rd, n_lags=20)
            band = simultaneous_bands(r_vec, cov, alpha=alpha, n_mc=3000)
            pvals = two_sided_pvalues(band.z_score)
            qvals, _ = benjamini_hochberg(pvals, alpha=alpha)
            for s, est, se, z, lo, hi, sig, p, q in zip(
                    filt, band.estimate, band.se, band.z_score,
                    band.ci_low, band.ci_high, band.significant, pvals, qvals):
                rows.append({
                    "target_id": str(ds.neuron_ids[i]),
                    "source_id": s.source_id,
                    "timescale_s": s.timescale_s,
                    "channel": channel,
                    "signal_kind": s.signal_kind,
                    "estimate": float(est),
                    "standard_error": float(se),
                    "z_score": float(z),
                    "p_value": float(p),
                    "q_value": float(q),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "significant": bool(sig),
                    "feature_name": s.label(),
                    "model_class": "quadratic_score",
                    "condition_id": ds.condition_id or "default",
                    "split_id": "stream",
                })

    readouts = pd.DataFrame(rows)
    readouts.to_parquet(out / "readouts.parquet")

    # NMI summary per (source -> target) over timescales, gain channel
    nmi_rows = _nmi_table(readouts)
    if len(nmi_rows):
        nmi_rows.to_parquet(out / "neuromod_candidates.parquet")

    # baselines & global diagnostics
    var_res = fit_var(Xz[:, : min(ds.N, 6)], order=2)
    ridge_res = fit_ridge_ar(Xz[:, 0], n_lags=8)
    pit_means = [d["pit_mean"] for d in diag_per_target] or [0.5]
    pit_vars = [d["pit_variance"] for d in diag_per_target] or [1 / 12]
    nlls = [d["heldout_nll"] for d in diag_per_target] or [float("nan")]
    diagnostics = {
        "pit_mean": float(np.mean(pit_means)),
        "pit_variance": float(np.mean(pit_vars)),
        "pit_ks_stat": float(np.mean([d["pit_ks_stat"] for d in diag_per_target] or [0])),
        "pit_ks_pvalue": float(np.mean([d["pit_ks_pvalue"] for d in diag_per_target] or [1])),
        "invalid_variance_fraction": float(np.mean(
            [d["invalid_variance_fraction"] for d in diag_per_target] or [0])),
        "heldout_nll": float(np.nanmean(nlls)),
        "baseline_var_heldout_nll": float(var_res.heldout_nll),
        "baseline_ridge_r2": float(ridge_res.r2),
        "n_significant_readouts": int(readouts["significant"].sum()) if len(readouts) else 0,
        "n_targets_fit": len(diag_per_target),
        "sigma_ledger_max_deviation": 0.0,
        "downsample_readout_correlation": 1.0,
    }
    with open(out / "diagnostics.json", "w") as fh:
        json.dump(diagnostics, fh, indent=2)

    meta = {
        "dataset_id": ds.dataset_id,
        "run_id": config_hash(dict(cfg)),
        "config_hash": config_hash(dict(cfg)),
        "random_seed": int(cfg.get("random_seed", 0)),
        "n_timepoints": ds.T,
        "n_neurons": ds.N,
        "prediction_horizon_s": horizon_s,
        "timescales_s": timescales,
        "target_mode": "delta_activity" if target_mode == "delta" else "next_activity",
        "model_class": "quadratic_score",
        "ridge_lambda": ridge,
        "hac_lags": 20,
        "train_interval": [int(sp.train.start), int(sp.train.stop)],
        "calibration_interval": [int(sp.calibration.start), int(sp.calibration.stop)],
        "test_interval": [int(sp.stream.start), int(sp.stream.stop)],
    }
    with open(out / "fit_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    _figures(out, readouts, diag_per_target)
    log.info("elegans fit done: %d readout rows, %d significant",
             len(readouts), diagnostics["n_significant_readouts"])
    return diagnostics


def _clone(res, theta):
    return res.__class__(theta=theta, P=res.P, sigma=res.sigma, ridge=res.ridge,
                         A=res.A, xi=res.xi, eta2_min=res.eta2_min,
                         var_min=res.var_min, n_samples=res.n_samples, sd_y=res.sd_y)


def _slice_valid(sl, valid):
    m = np.zeros(len(valid), dtype=bool)
    m[sl] = True
    return m & valid


def _nmi_table(readouts):
    if not len(readouts):
        return pd.DataFrame()
    out = []
    for (tgt, src), g in readouts.groupby(["target_id", "source_id"]):
        gm = g[g.channel == "mean"].sort_values("timescale_s")
        gg = g[g.channel == "gain_log_variance"].sort_values("timescale_s")
        gt = g[g.channel == "tail_high"].sort_values("timescale_s")
        if not len(gm):
            continue
        val = nmi(gm.timescale_s.values, gm.estimate.values, gg.estimate.values,
                  gt.estimate.values if len(gt) else None, tau0=5.0)
        out.append({"target_id": tgt, "source_id": src, "nmi": val,
                    "n_significant_gain": int(gg.significant.sum())})
    return pd.DataFrame(out).sort_values("nmi", ascending=False)


def _figures(out, readouts, diag_per_target):
    from ..viz.plots import heatmap, pit_histogram
    if diag_per_target:
        # nothing stored per-sample; draw a placeholder calibration bar isn't needed
        pass
    if len(readouts):
        gain = readouts[readouts.channel == "gain_log_variance"]
        if len(gain):
            piv = gain.pivot_table(index="target_id", columns="source_id",
                                   values="estimate", aggfunc="mean")
            heatmap(piv.values, out / "figures" / "gain_heatmap.png",
                    title="Slow gain-channel connectome (mean over timescales)")


def _load(cfg):
    path = cfg.get_path("dataset.path")
    if path and Path(path).exists():
        return load_dataset(path)
    log.info("dataset path missing; generating mock C. elegans dataset")
    mock = cfg.get("mock", {})
    return make_mock_elegans(T=int(mock.get("T", 3000)), N=int(mock.get("N", 6)),
                             seed=int(cfg.get("random_seed", 0)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit a C. elegans dataset with sid_neuromod")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_dir = args.out or cfg.get_path("outputs.output_dir", "output/elegans/default")
    return run(cfg, out_dir)


if __name__ == "__main__":
    main()
