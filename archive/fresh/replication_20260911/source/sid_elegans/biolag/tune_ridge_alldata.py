r"""SBTG insight #1 (regularization/stability tuning) applied to using ALL the data.

Imputation made the 28-worm gain connectome bootstrap-UNSTABLE. Our prior finding (and SBTG's
ethos): stronger regularization raises split-half reproducibility. So before committing to a full
re-run, sweep the SID ridge on the 28-worm donor-imputed data and measure, self-supervised (no
label peeking): (a) split-half STABILITY of the gain connectome, (b) SBTG-style NULL-CONTRAST of
the gain signal (real vs circular-shift). If a higher ridge sharply raises stability, the full
multiple-imputation re-run is justified.

Gentle by default (few splits/draws). Writes ridge_alldata_sweep.json.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np

from sid_elegans.biolag import config as C
from sid_elegans.biolag import metric as M
from sid_elegans.combined_data import load_combined
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.impute import impute_random_donor
from sid_elegans.stability import split_half_stability

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "sid_elegans" / "output" / "biolag"
RIDGES = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
LAG = 10                # 2.5 s, in the slow band (the effect's band)
N_SPLITS = 5
K_NULL = 3


def gain_at(Xs, names, ridge):
    return fit_distributional_connectome(Xs, names, lag=LAG, ridge=ridge, sigma_frac=0.0).matrices["gain"]


def null_contrast(Ximp, names, ridge, rng):
    real = float(np.nanmean(np.abs(gain_at(Ximp, names, ridge)[~np.eye(len(names), dtype=bool)])))
    nulls = []
    for _ in range(K_NULL):
        g = gain_at(M.circshift(Ximp, rng), names, ridge)
        nulls.append(float(np.nanmean(np.abs(g[~np.eye(len(names), dtype=bool)]))))
    return (real - np.mean(nulls)) / (np.std(nulls) + 1e-9)


def main():
    X, names, _ = load_combined(complete_case=False, min_worms_per_neuron=6,
                                signal="deconv", verbose=True)
    Ximp, n_imp = impute_random_donor(X, seed=0)
    print(f"28w imputed ({n_imp} cells), {len(names)} neurons; sweeping ridge at lag {LAG} "
          f"({LAG/C.FPS:.1f}s)\n", flush=True)
    rng = np.random.default_rng(0)
    rows = []
    for ridge in RIDGES:
        st = split_half_stability(lambda Xs: gain_at(Xs, names, ridge), Ximp,
                                  n_splits=N_SPLITS, seed=0)
        nc = null_contrast(Ximp, names, ridge, rng)
        rows.append({"ridge": ridge, "gain_split_half_spearman": st["spearman_mean"],
                     "gain_split_half_jaccard": st["jaccard_mean"], "gain_null_contrast": nc})
        print(f"  ridge={ridge:6.2f}  gain split-half spearman={st['spearman_mean']:+.3f} "
              f"jaccard={st['jaccard_mean']:.3f}  null-contrast={nc:+.2f}", flush=True)
    best = max(rows, key=lambda r: r["gain_split_half_spearman"])
    json.dump({"lag": LAG, "rows": rows, "best_by_stability": best},
              open(OUT / "ridge_alldata_sweep.json", "w"), indent=2, default=float)
    print(f"\nbest ridge by split-half stability: {best['ridge']} "
          f"(spearman {best['gain_split_half_spearman']:+.3f})", flush=True)
    print(f"wrote {OUT/'ridge_alldata_sweep.json'}")


if __name__ == "__main__":
    main()
