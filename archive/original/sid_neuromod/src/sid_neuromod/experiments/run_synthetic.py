r"""Run a synthetic validation experiment from a config (Sections 18.1-18.2).

Dispatches on ``experiment.kind``:
  * ``hidden_thermostat``  -> E1 variance-kernel recovery vs Kalman oracle
  * ``change_detection``   -> E3 null/scale/memory change detection
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..baselines.ridge_ar import fit_ridge_ar
from ..inference.sandwich import readout_covariance
from ..inference.simultaneous_bands import simultaneous_bands
from ..models.quadratic_score import QuadraticScoreMatcher, sigma_ledger
from ..monitoring.changepoint import detect_change
from ..monitoring.channels import dispersion_channel, serial_channel
from ..monitoring.eprocess import deadband_epsilon
from ..monitoring.pit import gaussian_pit
from ..readouts.mean_gain_tail import (center_readout_vector, d_log_variance,
                                       center_history)
from ..synthetic.change_scenarios import simulate_change_stream
from ..synthetic.hidden_thermostat import simulate_hidden_thermostat
from ..synthetic.oracles import (thermostat_contraction,
                                 thermostat_variance_kernel)
from ..utils.arrays import safe_log_square
from ..utils.config import config_hash, load_config
from ..utils.logging import get_logger
from ..utils.rng import get_rng

log = get_logger()


def _thermostat_design(x, L):
    """Centered design: intercept + z=log x^2 lags 1..L; target = next z-scored x."""
    xz = (x - x.mean()) / x.std()
    z = safe_log_square(xz)
    zc = z - z.mean()
    Tn = len(xz) - 1
    cols = [np.ones(Tn)]
    for u in range(1, L + 1):
        lag = np.zeros(Tn)
        lag[u:] = zc[: Tn - u]
        cols.append(lag)
    Psi = np.column_stack(cols)
    Y = xz[1: 1 + Tn]
    return Y, Psi


def run_hidden_thermostat(cfg, out_dir):
    exp = cfg.get("experiment", {})
    T = int(exp.get("T", 200_000))
    a = float(exp.get("a", 0.9))
    b = float(exp.get("b", 0.4))
    L = int(exp.get("n_lags", 12))
    seed = int(cfg.get("random_seed", 0))
    ridge = float(cfg.get_path("model.ridge", 1e-6) or 1e-6)

    log.info("E1 hidden thermostat: T=%d, a=%.2f, b=%.2f, L=%d", T, a, b, L)
    data = simulate_hidden_thermostat(T=T, a=a, b=b, seed=seed)
    Y, Psi = _thermostat_design(data.x, L)

    model = QuadraticScoreMatcher(sigma=0.0, ridge=ridge, seed=seed)
    res = model.fit(Y, Psi)

    lag_cols = list(range(1, L + 1))
    gain = center_readout_vector(res, Psi, lag_cols, "gain_log_variance")
    oracle = thermostat_variance_kernel(a, b, L)
    corr = float(np.corrcoef(gain, oracle)[0, 1])

    # inference: sup-t band on the gain kernel (center-history readout)
    center = center_history(Psi)

    def gain_readout(theta):
        r = res.__class__(theta=theta, P=res.P, sigma=res.sigma, ridge=res.ridge,
                          A=res.A, xi=res.xi, eta2_min=res.eta2_min,
                          var_min=res.var_min, n_samples=res.n_samples, sd_y=res.sd_y)
        return np.array([d_log_variance(r, center, k)[0] for k in lag_cols])

    r_vec, cov, _ = readout_covariance(res, gain_readout, n_lags=20)
    band = simultaneous_bands(r_vec, cov, alpha=0.05, n_mc=5000, rng=seed)
    n_sig = int(np.sum(band.significant))

    # mean baseline (ridge AR on raw x) -> R2 ~ 0
    ridge_ar = fit_ridge_ar(data.x, n_lags=L, alpha=1.0)

    # sigma ledger consistency
    ledger = sigma_ledger(Y, Psi, ridge=ridge, seed=seed)
    var_center = np.array([d["var_at_center"] for d in ledger])
    ledger_dev = float(np.max(np.abs(var_center - var_center[0])) / abs(var_center[0]))

    diagnostics = {
        "mean_r2": float(ridge_ar.r2),
        "variance_kernel_corr": corr,
        "n_significant_gain_lags": n_sig,
        "contraction_estimate": float(gain[1] / gain[0]) if gain[0] else float("nan"),
        "contraction_oracle": thermostat_contraction(a, b),
        "sigma_ledger_max_deviation": ledger_dev,
        "invalid_variance_fraction": float(res.invalid_variance_fraction),
        "gain_c1_estimate": float(gain[0]),
        "gain_c1_oracle": float(oracle[0]),
    }
    _write_outputs(out_dir, cfg, diagnostics,
                   readouts=_kernel_table(lag_cols, gain, oracle, band))
    _kernel_figure(out_dir, lag_cols, gain, oracle, band)
    log.info("E1 done: corr=%.3f, n_sig=%d, mean_r2=%.2e",
             corr, n_sig, ridge_ar.r2)
    return diagnostics


def run_change_detection(cfg, out_dir):
    exp = cfg.get("experiment", {})
    scenarios = exp.get("scenarios", ["null", "scale", "memory"])
    n_rep = int(exp.get("n_replications", 20))
    T = int(exp.get("T", 22_000))
    t_change = int(exp.get("t_change", 6_000))
    alpha = float(cfg.get_path("monitoring.alpha", 0.01) or 0.01)
    delta = float(cfg.get_path("monitoring.delta", 0.01) or 0.01)
    L = int(exp.get("n_lags", 6))
    seed0 = int(cfg.get("random_seed", 0))

    results = {}
    for scenario in scenarios:
        alarms = 0
        delays = []
        for rep in range(n_rep):
            seed = seed0 + 1000 * rep
            res, tc = monitor_scenario(scenario, seed=seed, T=T, t_change=t_change,
                                       alpha=alpha, delta=delta, L=L)
            if res.alarmed:
                if scenario == "null":
                    alarms += 1  # any alarm under null is a false alarm
                elif res.alarm_time is not None and res.alarm_time >= tc:
                    alarms += 1
                    delays.append(res.alarm_time - tc)
        rate = alarms / n_rep
        med_delay = float(np.median(delays)) if delays else None
        results[scenario] = {"detections": alarms, "rate": rate,
                             "median_delay": med_delay}
        log.info("E3 %-8s: rate=%.2f, median_delay=%s", scenario, rate, med_delay)

    diagnostics = {
        "null_false_alarm_rate": results.get("null", {}).get("rate", 0.0),
        "scale_detection_power": results.get("scale", {}).get("rate", 0.0),
        "memory_detection_power": results.get("memory", {}).get("rate", 0.0),
        "alpha": alpha,
        "delta": delta,
        "per_scenario": results,
    }
    _write_outputs(out_dir, cfg, diagnostics, readouts=None)
    return diagnostics


def _build_z_design(x, L, z_ref_mean=None, z_ref_std=None, x_mean=None, x_std=None):
    """Design of intercept + z=log x^2 lags 1..L for target = next z-scored x.

    Z-scoring / centering references (``*_mean``/``*_std``) come from the *baseline*
    so the monitored stream is transformed with baseline statistics (no leakage).
    """
    x = np.asarray(x, dtype=float)
    if x_mean is None:
        x_mean = x.mean()
    if x_std is None:
        x_std = x.std() or 1.0
    xz = (x - x_mean) / x_std
    z = safe_log_square(xz)
    if z_ref_mean is None:
        z_ref_mean = z.mean()
    zc = z - z_ref_mean
    Tn = len(xz) - 1
    cols = [np.ones(Tn)]
    for u in range(1, L + 1):
        lag = np.zeros(Tn); lag[u:] = zc[: Tn - u]; cols.append(lag)
    Psi = np.column_stack(cols); Y = xz[1: 1 + Tn]
    refs = dict(x_mean=x_mean, x_std=x_std, z_ref_mean=z_ref_mean)
    return Y, Psi, refs


def _lag_filter(z, decay=0.9):
    """Slow causal EWMA of the variance-channel observable (a source filter g_t)."""
    g = np.zeros_like(z)
    for k in range(1, len(z)):
        g[k] = decay * g[k - 1] + (1 - decay) * z[k - 1]
    return g


def monitor_scenario(scenario, seed, T=22_000, t_change=6_000, alpha=0.01,
                     delta=0.01, L=6, baseline_T=60_000, a=0.9, b=0.4):
    """Fit the conditional density on a separate *null* baseline, then monitor a fresh
    stream that contains the change at ``t_change``. Returns ``(ChangeResult, t_change)``.

    Channels: dispersion, serial, and a lag-kernel channel (dispersion gated by a slow
    source filter) which is what catches a memory-only change (Table 2).
    """
    from ..monitoring.channels import lag_channel, var_proxy
    # baseline: pure null process, split into train / debias
    base = simulate_change_stream(T=baseline_T, t_change=baseline_T + 1,
                                  scenario="null", a=a, b=b, seed=seed + 1)
    Yb, Psib, refs = _build_z_design(base.x, L)
    nb = len(Yb)
    # train on the first half; use the (large) second half for debias so the dead band
    # ~ sqrt(2 c log(2K/delta)/n_db) is small (the residual calibration error is tiny
    # when the model is fit on a genuine null baseline).
    i_tr = int(0.5 * nb); i_db = int(0.5 * nb)
    model = QuadraticScoreMatcher(sigma=0.0, ridge=1e-6, seed=seed)
    model.fit(Yb[:i_tr], Psib[:i_tr])

    zb = np.log(np.maximum((base.x - refs["x_mean"]) ** 2 / refs["x_std"] ** 2, 1e-8))
    gb = _lag_filter(zb - refs["z_ref_mean"])
    g_mean, g_std = float(gb.mean()), float(gb.std() or 1.0)

    # debias offsets & dead bands from a held-out baseline slice
    mud, vard = model.moments(Psib[i_db:])
    ud = gaussian_pit(Yb[i_db:], mud, vard)
    gd = gb[1: 1 + nb][i_db:] if len(gb) >= nb + 1 else gb[i_db:i_db + len(ud)]
    off = {
        "dispersion": dispersion_channel(ud).mean(),
        "serial": serial_channel(ud).mean(),
        "lag": lag_channel(ud, gd, g_mean, g_std).mean(),
    }
    n_db = len(ud)
    eps_map = {name: deadband_epsilon(n_db, n_channels=3, delta=delta,
                                      c=var_proxy(name)) for name in off}

    # monitored stream with the change, transformed by baseline references
    stream = simulate_change_stream(T=T, t_change=t_change, scenario=scenario,
                                    a=a, b=b, seed=seed)
    Ys, Psis, _ = _build_z_design(stream.x, L, z_ref_mean=refs["z_ref_mean"],
                                  x_mean=refs["x_mean"], x_std=refs["x_std"])
    mu, var = model.moments(Psis)
    u = gaussian_pit(Ys, mu, var)
    zs = np.log(np.maximum((stream.x - refs["x_mean"]) ** 2 / refs["x_std"] ** 2, 1e-8))
    gs = _lag_filter(zs - refs["z_ref_mean"])[1: 1 + len(u)]
    channels = {
        "dispersion": dispersion_channel(u) - off["dispersion"],
        "serial": serial_channel(u) - off["serial"],
        "lag": lag_channel(u, gs, g_mean, g_std) - off["lag"],
    }
    res = detect_change(channels, alpha=alpha, eps_map=eps_map, mode="average")
    return res, t_change


# ------------------------------------------------------------------ outputs
def _kernel_table(lags, gain, oracle, band):
    import pandas as pd
    return pd.DataFrame({
        "timescale_s": lags,
        "channel": "gain_log_variance",
        "estimate": gain,
        "oracle": oracle,
        "standard_error": band.se,
        "z_score": band.z_score,
        "ci_low": band.ci_low,
        "ci_high": band.ci_high,
        "significant": band.significant,
    })


def _kernel_figure(out_dir, lags, gain, oracle, band):
    from ..viz.plots import kernel_recovery_plot
    kernel_recovery_plot(lags, gain, oracle,
                         Path(out_dir) / "figures" / "kernel_recovery.png",
                         title="Hidden-thermostat gain kernel vs Kalman oracle",
                         ci_low=band.ci_low, ci_high=band.ci_high)


def _write_outputs(out_dir, cfg, diagnostics, readouts=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "diagnostics.json", "w") as fh:
        json.dump(diagnostics, fh, indent=2)
    meta = {"config_hash": config_hash(dict(cfg)),
            "random_seed": int(cfg.get("random_seed", 0)),
            "experiment": cfg.get("experiment", {}).get("kind", "unknown")}
    with open(out / "fit_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    if readouts is not None:
        readouts.to_parquet(out / "readouts.parquet")


KINDS = {
    "hidden_thermostat": run_hidden_thermostat,
    "change_detection": run_change_detection,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a synthetic sid_neuromod experiment")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    kind = cfg.get("experiment", {}).get("kind")
    if kind not in KINDS:
        raise SystemExit(f"unknown experiment kind {kind!r}; choose {list(KINDS)}")
    out_dir = args.out or cfg.get_path("outputs.output_dir",
                                       f"output/synthetic/{kind}")
    return KINDS[kind](cfg, out_dir)


if __name__ == "__main__":
    main()
