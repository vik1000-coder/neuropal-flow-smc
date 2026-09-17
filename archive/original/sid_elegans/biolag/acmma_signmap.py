r"""Disentangle the ACMMA sign flip: is neuropeptide dBC's sign driven by DATA (6w vs 28w) or by
RIDGE (1e-2 vs 3.0)? Map observed dBC (z-scored vs a small circular-shift null) over both axes.
Fast (point estimate + small surrogate, no bootstrap). Gentle."""
from __future__ import annotations
import os, time, warnings
import numpy as np
from sid_elegans.acmma import fit_acmma_connectome
from sid_elegans.biolag import config as C
from sid_elegans.biolag import metric as M
from sid_elegans.biolag.references import build_registry
from sid_elegans.combined_data import load_combined
from sid_elegans.lagres import metric as mt

warnings.filterwarnings("ignore")
LAGS = C.LAG_FRAMES; N_SURR = 12; THROTTLE = 0.3


def curves(X, names, R, mask, ridge):
    cm, cg = [], []
    for l in LAGS:
        m, _ = fit_acmma_connectome(X, names, lag=l, ridge=ridge, min_triple=2000, shrink=0.05)
        cm.append(mt.corr_score(m["mean"], R, mask, "auroc"))
        cg.append(mt.corr_score(m["gain"], R, mask, "auroc"))
    return np.array(cm), np.array(cg)


def dbc(X, names, R, mask, ridge, rng):
    om, og = curves(X, names, R, mask, ridge)
    sm, sg = [], []
    for _ in range(N_SURR):
        a, b = curves(M.circshift(X, rng), names, R, mask, ridge)
        sm.append(a); sg.append(b)
        if THROTTLE:
            time.sleep(THROTTLE)
    mu_m, sd_m = np.nanmean(sm, 0), np.nanstd(sm, 0) + 1e-9
    mu_g, sd_g = np.nanmean(sg, 0), np.nanstd(sg, 0) + 1e-9
    bcg = M.band_concordance((og - mu_g) / sd_g, "slow")
    bcm = M.band_concordance((om - mu_m) / sd_m, "slow")
    return bcg - bcm, bcg, bcm


def main():
    X6, n6, _ = load_combined(coverage_frac=0.6, complete_case=True, signal="deconv", verbose=False)
    X28, n28, _ = load_combined(complete_case=False, min_worms_per_neuron=6, signal="deconv", verbose=False)
    print("neuropeptide dBC (gain-mean band concordance) across DATA x RIDGE:\n", flush=True)
    print(f"{'data':10s} {'ridge':>7s} {'dBC':>8s} {'BC_gain':>9s} {'BC_mean':>9s}", flush=True)
    for tag, X, names in [("6w_clean", X6, n6), ("28w_acmma", X28, n28)]:
        R = next(r for r in build_registry(names, 10) if r.name == "neuropeptide:all").adjacency
        mask = mt.eval_mask(len(names)); rng = np.random.default_rng(0)
        for ridge in [0.01, 0.3, 1.0, 3.0]:
            d, bg, bm = dbc(X, names, R, mask, ridge, rng)
            print(f"{tag:10s} {ridge:7.2f} {d:+8.2f} {bg:+9.2f} {bm:+9.2f}", flush=True)


if __name__ == "__main__":
    main()
