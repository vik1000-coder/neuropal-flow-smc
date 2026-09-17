r"""Rigorous verification of the MDN serotonin signal: data scaling (6 vs 20 worms),
robustness (raw vs deconvolved), significance, stability, and whether it survives
partialling out BOTH source- and target-variance. Writes output/mdn_verify.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from sid_elegans.combined_data import load_combined
from sid_elegans.evaluate import _offdiag
from sid_elegans.ground_truth import load_all_monoamine_layers
from sid_elegans.mdn import fit_mdn
from sid_elegans.significance import perm_p
from sid_elegans.stability import split_half_stability

OUT = Path(__file__).resolve().parent / "output"


def partial_out(M, controls, gt):
    m = _offdiag(np.abs(M)); g = (_offdiag(gt) > 0).astype(int)
    Xc = np.column_stack([np.ones(len(m))] + [_offdiag(c) for c in controls])
    v = np.isfinite(m) & np.all(np.isfinite(Xc), 1)
    beta, *_ = np.linalg.lstsq(Xc[v], m[v], rcond=None)
    resid = m.copy(); resid[v] = m[v] - Xc[v] @ beta
    vv = np.isfinite(resid)
    return float(roc_auc_score(g[vv], resid[vv]))


def mdn_gain(Xs, names, sd=0):
    return fit_mdn(Xs, names, source_tau_s=20.0, k=3, epochs=350, hidden=128,
                   weight_decay=1e-2, seed=sd).matrices["gain"]


def main():
    rows = []
    for cov, label in [(0.6, "6w/80n"), (0.9, "20w/56n")]:
        for signal in ["deconv", "raw"]:
            X_list, names, fps = load_combined(coverage_frac=cov, complete_case=True,
                                               signal=signal, verbose=False)
            mono = load_all_monoamine_layers(names)
            gt = mono["serotonin"]; N = len(names)
            allX = np.concatenate(X_list, 0)
            srcvar = np.tile(np.nanvar(allX, 0)[None, :], (N, 1)); np.fill_diagonal(srcvar, 0)
            tgtvar = np.tile(np.nanvar(allX, 0)[:, None], (1, N)); np.fill_diagonal(tgtvar, 0)
            ens = np.mean([np.abs(mdn_gain(X_list, names, sd)) for sd in range(3)], axis=0)
            a, p = perm_p(ens, gt, n=1500)
            resid = partial_out(ens, [srcvar, tgtvar], gt)
            st = split_half_stability(lambda Xs: np.abs(mdn_gain(Xs, names, 0)),
                                      X_list, n_splits=3)["spearman_mean"]
            row = {"config": label, "signal": signal, "auroc": a, "perm_p": p,
                   "partial_src_tgt_var": resid, "stability": st,
                   "n_worms": len(X_list), "n_neurons": N,
                   "n_edges": int((gt > 0).sum())}
            rows.append(row)
            print(f"{label:8s} {signal:6s}: serotonin AUROC={a:.3f}(p={p:.4f}) "
                  f"partial(src+tgt)={resid:.3f} stability={st:.3f}", flush=True)
    OUT.mkdir(exist_ok=True)
    json.dump(rows, open(OUT / "mdn_verify.json", "w"), indent=2)
    print(f"wrote {OUT/'mdn_verify.json'}", flush=True)


if __name__ == "__main__":
    main()
