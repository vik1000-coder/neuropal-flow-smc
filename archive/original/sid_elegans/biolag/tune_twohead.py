r"""Null-contrast hyperparameter tuning of the two-head DSM neural model.

The two-head model IS "SBTG's neural denoising-score-matching score analysed with SID's
approach": a neural DSM conditional density whose mean/gain channels are read out by autograd
(twohead.py). We tune its hyperparameters with the SAME objective SBTG uses -- the
NULL-CONTRAST -- selecting HPs that maximise how far the real coupling signal sits above a
coupling-destroying null:

    contrast_ch = (mean|coupling_real[offdiag]| - mean_k mean|coupling_null_k|) / std_k(...)

We use a per-neuron CIRCULAR-SHIFT null (preserves each neuron's own variance/autocorrelation,
destroys directed cross-neuron coupling) -- more principled than SBTG's row-permutation and
consistent with the biolag surrogate elsewhere. Primary objective = GAIN-channel contrast (the
distributional readout of interest); mean-channel contrast tracked too. Label-free (never touches
the connectome), so it cannot overfit the evaluation.

Random search (optuna absent). Writes output/biolag/twohead_tuning.json.
Run: PYTHONPATH=.:SBTG ./.venv/bin/python sid_elegans/biolag/tune_twohead.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from sid_elegans.biolag import metric as M
from sid_elegans.combined_data import load_combined
from sid_elegans.twohead import fit_two_head

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "sid_elegans" / "output" / "biolag"
N_TRIALS = 40
K_NULL = 4
TUNE_HORIZON = 5          # a representative mid lag (1.25 s) for HP selection
EPOCHS_REAL = 200
EPOCHS_NULL = 80          # nulls need only the coupling magnitude, not full convergence

SPACE = {
    "hidden": [64, 128, 256],
    "trunk_layers": [2, 3],
    "sigma_c": [0.2, 0.3, 0.5],
    "sigma_m": [0.3],
    "lam_marg": [0.5, 1.0, 2.0],
    "weight_decay": [1e-4, 1e-3, 1e-2],
    "lr": [1e-3, 2e-3],
    "source_tau_s": [None, 5.0, 20.0],
}


def _sig(mat):
    off = ~np.eye(mat.shape[0], dtype=bool)
    a = np.abs(mat[off]); a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def null_contrast(X, names, cfg, rng):
    """Fit real + K circular-shift nulls; return (contrast_gain, contrast_mean, real signals)."""
    real = fit_two_head(X, names, horizon=TUNE_HORIZON, epochs=EPOCHS_REAL, seed=0, **cfg).matrices
    rg, rm = _sig(real["gain"]), _sig(real["mean"])
    ng, nm = [], []
    for k in range(K_NULL):
        Xs = M.circshift(X, rng)
        nl = fit_two_head(Xs, names, horizon=TUNE_HORIZON, epochs=EPOCHS_NULL, seed=k, **cfg).matrices
        ng.append(_sig(nl["gain"])); nm.append(_sig(nl["mean"]))
    cg = (rg - np.mean(ng)) / (np.std(ng) + 1e-9)
    cm = (rm - np.mean(nm)) / (np.std(nm) + 1e-9)
    return float(cg), float(cm), rg, rm


def main():
    X, names, _ = load_combined(coverage_frac=0.6, complete_case=True, signal="deconv",
                               verbose=True)
    rng = np.random.default_rng(0)
    trials = []
    for t in range(N_TRIALS):
        cfg = {k: (rng.choice(v).item() if not isinstance(v[0], (float, type(None)))
                   else rng.choice(np.array(v, dtype=object))) for k, v in SPACE.items()}
        cfg = {k: (None if v is None else (float(v) if k in ("sigma_c", "sigma_m", "lam_marg",
                   "weight_decay", "lr", "source_tau_s") else int(v))) for k, v in cfg.items()}
        try:
            cg, cm, rg, rm = null_contrast(X, names, cfg, rng)
        except Exception as e:
            cg = cm = rg = rm = float("nan")
            print(f"  trial {t} FAILED: {e}", flush=True)
        trials.append({"trial": t, "cfg": cfg, "contrast_gain": cg, "contrast_mean": cm,
                       "real_gain": rg, "real_mean": rm})
        print(f"  trial {t+1}/{N_TRIALS} contrast_gain={cg:+.2f} contrast_mean={cm:+.2f} "
              f"cfg={cfg}", flush=True)

    valid = [tr for tr in trials if np.isfinite(tr["contrast_gain"])]
    best = max(valid, key=lambda tr: tr["contrast_gain"]) if valid else None
    json.dump({"n_trials": N_TRIALS, "k_null": K_NULL, "tune_horizon": TUNE_HORIZON,
               "objective": "gain-channel null-contrast (circular-shift null)",
               "best": best, "trials": sorted(trials, key=lambda t: -(t["contrast_gain"]
                                              if np.isfinite(t["contrast_gain"]) else -9))},
              open(OUT / "twohead_tuning.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT/'twohead_tuning.json'}")
    if best:
        print(f"BEST contrast_gain={best['contrast_gain']:+.2f} (mean-ch {best['contrast_mean']:+.2f})")
        print(f"BEST cfg = {best['cfg']}")


if __name__ == "__main__":
    main()
