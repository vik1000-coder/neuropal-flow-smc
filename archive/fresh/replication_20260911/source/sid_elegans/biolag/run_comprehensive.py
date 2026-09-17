r"""Comprehensive band-concordance comparison, using ALL the data (28 worms, donor-imputed) and
a head-to-head of estimators + baselines:

  SID_hyv    -- closed-form SID, Hyvarinen score matching (sigma=0)          mean + gain
  SID_dsm    -- closed-form SID, denoising score matching (sigma=0.5*sd(Y))  mean + gain
  cross_corr -- lagged cross-correlation (mean-only baseline)                mean
  ridge_lag  -- single-lag multivariate ridge (mean-only baseline)           mean

Run on two data configs for comparison:
  28w_imputed -- all 28 worms x 84 neurons, random-DONOR imputation (realistic variance, decoupled)
  6w_clean    -- 6 complete-case worms x 80 neurons (no imputation; the earlier primary corner)

Same controlled pipeline as Phase 2: per-lag AUROC curve -> circular-shift null-referenced z ->
band concordance BC = mean z(expected band) - mean z(complementary band); confirmatory dBC =
BC_gain - BC_mean for the SID estimators; primary + global-mode-removed blocks; source-variance
control (-> 0). Surrogate fits are SHARED across estimators. Writes comprehensive_results.json.

Run: PYTHONPATH=.:SBTG ./.venv/bin/python sid_elegans/biolag/run_comprehensive.py
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from sid_elegans.biolag import config as C
from sid_elegans.biolag import metric as M
from sid_elegans.biolag.references import build_registry
from sid_elegans.combined_data import load_combined
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.impute import impute_random_donor
from sid_elegans.lagres import metric as mt
from sid_elegans.lagres.effects import cross_corr, ridge_lag

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "sid_elegans" / "output" / "biolag"
LAGS = C.LAG_FRAMES
# gentle-by-default; overridable via env so we do not overload a shared machine
N_SURR = int(os.environ.get("COMP_NSURR", 25))
N_BOOT = int(os.environ.get("COMP_NBOOT", 25))
THROTTLE = float(os.environ.get("COMP_THROTTLE", 0.6))   # sleep (s) between worm-sets, lets CPU breathe
CONFIGS = os.environ.get("COMP_CONFIGS", "28w_imputed,6w_clean").split(",")
REFS = ["neuropeptide:all", "neuropeptide:pdf-1", "neuropeptide:ntc-1",
        "monoamine:serotonin", "structural:gap", "structural:chemical"]
# (estimator, channel) columns to score; SID estimators carry gain, baselines mean-only
CHANNELS = [("SID_hyv", "mean"), ("SID_hyv", "gain"),
            ("SID_dsm", "mean"), ("SID_dsm", "gain"),
            ("cross_corr", "mean"), ("ridge_lag", "mean")]
SID_ESTIMATORS = ["SID_hyv", "SID_dsm"]


def est_matrices(X, names, l):
    """All estimators' coupling matrices at lag l (SID fit once, extract mean+gain)."""
    hyv = fit_distributional_connectome(X, names, lag=l, sigma_frac=0.0, ridge=1e-2).matrices
    dsm = fit_distributional_connectome(X, names, lag=l, sigma_frac=0.5, ridge=1e-2).matrices
    return {"SID_hyv": {"mean": hyv["mean"], "gain": hyv["gain"]},
            "SID_dsm": {"mean": dsm["mean"], "gain": dsm["gain"]},
            "cross_corr": {"mean": cross_corr(X, l)},
            "ridge_lag": {"mean": ridge_lag(X, l)}}


def curves(X, names, refs, mask):
    """{(est,ch): {ref: auroc-curve over lags}} for one worm set."""
    out = {ec: {r.name: [] for r in refs} for ec in CHANNELS}
    for l in LAGS:
        m = est_matrices(X, names, l)
        for (est, ch) in CHANNELS:
            mat = m[est][ch]
            for r in refs:
                out[(est, ch)][r.name].append(mt.corr_score(mat, r.adjacency, mask, "auroc"))
    return {ec: {rn: np.array(v) for rn, v in d.items()} for ec, d in out.items()}


def bc_from(cur, refs, mu0, sd0):
    """band concordance per (est,ch,ref) + dBC per (sid_est,ref)."""
    bc = {}
    for (est, ch) in CHANNELS:
        for r in refs:
            z = (cur[(est, ch)][r.name] - mu0[(est, ch)][r.name]) / sd0[(est, ch)][r.name]
            bc[(est, ch, r.name)] = M.band_concordance(z, r.expected_band)
    dbc = {}
    for est in SID_ESTIMATORS:
        for r in refs:
            dbc[(est, r.name)] = bc[(est, "gain", r.name)] - bc[(est, "mean", r.name)]
    return bc, dbc


def stat(x):
    x = np.array([v for v in x if np.isfinite(v)])
    if x.size == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan}
    return {"mean": float(x.mean()), "lo": float(np.percentile(x, 2.5)),
            "hi": float(np.percentile(x, 97.5))}


def run_block(X, names, refs, mask, tag, rng):
    obs = curves(X, names, refs, mask)
    surr = []
    for s in range(N_SURR):
        surr.append(curves(M.circshift(X, rng), names, refs, mask))
        if THROTTLE:
            time.sleep(THROTTLE)
        if (s + 1) % 10 == 0:
            print(f"  [{tag}] surrogate {s+1}/{N_SURR}", flush=True)
    mu0 = {ec: {r.name: np.nanmean([sc[ec][r.name] for sc in surr], 0) for r in refs} for ec in CHANNELS}
    sd0 = {ec: {r.name: np.nanstd([sc[ec][r.name] for sc in surr], 0) + 1e-9 for r in refs} for ec in CHANNELS}
    obc, odbc = bc_from(obs, refs, mu0, sd0)
    nbc = [bc_from(sc, refs, mu0, sd0) for sc in surr]
    boot = {"bc": [], "dbc": []}
    for b in range(N_BOOT):
        idx = rng.integers(0, len(X), len(X)).tolist()
        bbc, bdbc = bc_from(curves([X[i] for i in idx], names, refs, mask), refs, mu0, sd0)
        boot["bc"].append(bbc); boot["dbc"].append(bdbc)
        if THROTTLE:
            time.sleep(THROTTLE)
        if (b + 1) % 10 == 0:
            print(f"  [{tag}] bootstrap {b+1}/{N_BOOT}", flush=True)
    # assemble
    res = {"BC": {}, "dBC": {}}
    for key, val in obc.items():
        est, ch, rn = key
        nd = np.array([nb[0][key] for nb in nbc])
        res["BC"][f"{est}|{ch}|{rn}"] = {
            "obs": val, "p": float((np.sum(nd >= val) + 1) / (len(nd) + 1)),
            **{f"boot_{k}": v for k, v in stat([bb[key] for bb in boot["bc"]]).items()}}
    for key, val in odbc.items():
        est, rn = key
        nd = np.array([nb[1][key] for nb in nbc])
        res["dBC"][f"{est}|{rn}"] = {
            "obs": val, "p": float((np.sum(nd >= val) + 1) / (len(nd) + 1)),
            **{f"boot_{k}": v for k, v in stat([bb[key] for bb in boot["dbc"]]).items()}}
    prim = next(r for r in refs if r.name == "neuropeptide:all")
    res["source_variance_BC"] = float(
        M.band_concordance(M.source_variance_curve(X, names, prim, mask), prim.expected_band))
    return res


def load_config(tag):
    if tag == "28w_imputed":
        X, names, _ = load_combined(complete_case=False, min_worms_per_neuron=6,
                                    signal="deconv", verbose=True)
        X, n_imp = impute_random_donor(X, seed=0)
        print(f"[{tag}] donor-imputed {n_imp} cells", flush=True)
    else:  # 6w_clean
        X, names, _ = load_combined(coverage_frac=0.6, complete_case=True,
                                    signal="deconv", verbose=True)
    return X, names


def _save(results):
    # merge with any prior file so re-runs of single configs accumulate
    p = OUT / "comprehensive_results.json"
    prior = json.load(open(p)) if p.exists() else {}
    prior.update(results)
    json.dump(prior, open(p, "w"), indent=2, default=float)


def main():
    print(f"[config] N_SURR={N_SURR} N_BOOT={N_BOOT} THROTTLE={THROTTLE}s CONFIGS={CONFIGS} "
          f"BLAS_threads={os.environ.get('OPENBLAS_NUM_THREADS','?')}", flush=True)
    results = {}
    for tag in CONFIGS:
        X, names = load_config(tag)
        refs = build_registry(names, min_specific_edges=10)
        refs = [r for r in refs if r.name in REFS]
        mask = mt.eval_mask(len(names)); rng = np.random.default_rng(0)
        print(f"=== {tag}: {len(X)} worms, {len(names)} neurons ===", flush=True)
        primary = run_block(X, names, refs, mask, f"{tag}-prim", rng)
        results[tag] = {"n_worms": len(X), "n_neurons": len(names), "primary": primary}
        _save(results)  # write after primary block (crash-resilient)
        gmode = run_block(M.remove_global_mode(X, n_pc=1), names, refs, mask, f"{tag}-gmode", rng)
        results[tag]["global_mode_removed"] = gmode
        _save(results)
        # console headline: neuropeptide dBC for each SID estimator, primary vs global
        print(f"\n[{tag}] neuropeptide:all dBC (gain-mean):", flush=True)
        for est in SID_ESTIMATORS:
            p = primary["dBC"][f"{est}|neuropeptide:all"]; g = gmode["dBC"][f"{est}|neuropeptide:all"]
            print(f"    {est}: PRIMARY {p['obs']:+.2f} (p={p['p']:.3f}, boot[{p['boot_lo']:+.2f},{p['boot_hi']:+.2f}])"
                  f"  GLOBAL {g['obs']:+.2f} (p={g['p']:.3f})", flush=True)
        print(f"    baselines BC_mean (primary): cross_corr="
              f"{primary['BC']['cross_corr|mean|neuropeptide:all']['obs']:+.2f} "
              f"ridge_lag={primary['BC']['ridge_lag|mean|neuropeptide:all']['obs']:+.2f}"
              f"  src-var ctrl={primary['source_variance_BC']:+.2f}", flush=True)
    print(f"\nwrote {OUT/'comprehensive_results.json'}")


if __name__ == "__main__":
    main()
