r"""#3: the expressive estimators (ensembled two-head + non-Gaussian MDN) on the
DECONVOLVED signal, with the decisive control — does ANY method carry distributional
signal *beyond* source-variance magnitude?

For each method we (a) score |coupling| vs each connectome, and (b) partial out the
source-variance baseline (regress |coupling| on source-variance over off-diagonal edges)
and score the RESIDUAL. If a method beats chance only before partialling, its "signal" was
just magnitude.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from sid_elegans.combined_data import load_combined
from sid_elegans.evaluate import _offdiag
from sid_elegans.ground_truth import load_all_monoamine_layers, load_cook
from sid_elegans.mdn import fit_mdn
from sid_elegans.multiscale import fit_multiscale_connectome
from sid_elegans.twohead import fit_two_head

OUT = Path(__file__).resolve().parent / "output"


def _auroc(score_vec, gt_vec):
    v = np.isfinite(score_vec) & np.isfinite(gt_vec)
    s, g = score_vec[v], (gt_vec[v] > 0).astype(int)
    if g.sum() in (0, len(g)):
        return np.nan
    return roc_auc_score(g, s)


def _perm_p(score_vec, gt_vec, n=2000, seed=0):
    v = np.isfinite(score_vec) & np.isfinite(gt_vec)
    s, g = score_vec[v], (gt_vec[v] > 0).astype(int)
    if g.sum() in (0, len(g)):
        return np.nan, np.nan
    obs = roc_auc_score(g, s); rng = np.random.default_rng(seed)
    ge = sum(roc_auc_score(g, rng.permutation(s)) >= obs for _ in range(n))
    return obs, (ge + 1) / (n + 1)


def ensemble_two_head(X_list, names, seeds=(0, 1, 2), **kw):
    G = None
    for sd in seeds:
        g = np.abs(fit_two_head(X_list, names, seed=sd, **kw).matrices["gain"])
        G = g if G is None else G + g
    return G / len(seeds)


def main():
    OUT.mkdir(exist_ok=True)
    X_list, names, fps = load_combined(coverage_frac=0.6, complete_case=True,
                                       signal="deconv", verbose=True)
    cook = load_cook(names); mono = load_all_monoamine_layers(names)
    N = len(names)
    allX = np.concatenate(X_list, 0)
    srcvar = np.tile(np.nanvar(allX, 0)[None, :], (N, 1)); np.fill_diagonal(srcvar, 0)

    # methods (all |coupling|)
    print("training closed-form / two-head ensemble / MDN on deconvolved data ...")
    cf = fit_multiscale_connectome(X_list, names, ridge=3.0)
    th = ensemble_two_head(X_list, names, source_tau_s=20.0, epochs=400, hidden=64,
                           weight_decay=1e-1)
    md = fit_mdn(X_list, names, source_tau_s=20.0, k=3, epochs=500, hidden=128,
                 weight_decay=1e-2, verbose=True)
    methods = {
        "source_variance": np.abs(srcvar),
        "closed_form_gain": np.abs(cf.matrices["gain_slow"]),
        "two_head_gain(ens)": th,
        "MDN_gain": np.abs(md.matrices["gain"]),
    }
    targets = {"tyramine": mono["tyramine"], "serotonin": mono["serotonin"],
               "dopamine": mono["dopamine"], "Cook": cook["struct"]}

    sv = _offdiag(srcvar)
    print("\n=== AUROC | raw vs source-variance-partialled residual (perm p) ===")
    print(f"{'method':22s}" + "".join(f"{t:>22s}" for t in targets))
    for mname, M in methods.items():
        cells = []
        for tname, gt in targets.items():
            m = _offdiag(M); g = _offdiag(gt)
            raw = _auroc(m, g)
            # partial out source-variance: residual of |coupling| ~ srcvar
            vfin = np.isfinite(m) & np.isfinite(sv)
            if vfin.sum() > 5 and np.std(sv[vfin]) > 1e-12:
                b = np.polyfit(sv[vfin], m[vfin], 1)
                resid = m.copy(); resid[vfin] = m[vfin] - (b[0] * sv[vfin] + b[1])
            else:
                resid = m
            res_auroc, res_p = _perm_p(resid, g, n=1500)
            cells.append(f"{raw:.3f}/{res_auroc:.3f}(p{res_p:.3f})")
        print(f"{mname:22s}" + "".join(f"{c:>22s}" for c in cells))
    print("\n(each cell: raw AUROC / source-variance-partialled residual AUROC (perm p))")
    print("A method carries distributional signal beyond magnitude only if the PARTIALLED "
          "residual stays above chance.")


if __name__ == "__main__":
    main()
