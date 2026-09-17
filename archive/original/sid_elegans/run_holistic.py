r"""Holistic assessment: does the score-identified distributional approach do BETTER
OVERALL across ALL the (imperfect, receptor-based) targets — not whether it perfectly
matches any single one?

Targets are noisy proxies: the Cook connectome is anatomy (not function), and the
monoamine/neuropeptide "networks" are RECEPTOR-EXPRESSION predictions (source expresses
the transmitter, target expresses a cognate receptor) — not measured signaling. So the
meaningful range is ~0.55-0.75; a near-1.0 AUROC on a noisy proxy is an over-fit red flag,
not a triumph. We compare every method across every target and summarize by MEAN AUROC and
MEAN RANK across the biological targets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sid_elegans.baselines import pearson_lag
from sid_elegans.combined_data import load_combined
from sid_elegans.evaluate import score_matrix
from sid_elegans.ground_truth import (load_all_monoamine_layers, load_cook,
                                      load_neuropeptide_layer)
from sid_elegans.mdn import fit_mdn
from sid_elegans.multiscale import fit_multiscale_connectome

OUT = Path(__file__).resolve().parent / "output"


def _rank_composite(*mats):
    """Rank-sum composite of several |coupling| matrices (equal-weight distributional score)."""
    n = mats[0].shape[0]
    off = ~np.eye(n, dtype=bool)
    comp = np.zeros((n, n))
    total = np.zeros(off.sum())
    for M in mats:
        total = total + rankdata(np.abs(M[off]))
    comp[off] = total
    return comp


def build(signal="deconv"):
    X_list, names, fps = load_combined(coverage_frac=0.6, complete_case=True,
                                       signal=signal, verbose=True)
    N = len(names)
    cook = load_cook(names); mono = load_all_monoamine_layers(names)
    npep = load_neuropeptide_layer(names)
    allX = np.concatenate(X_list, 0)
    srcvar = np.tile(np.nanvar(allX, 0)[None, :], (N, 1)); np.fill_diagonal(srcvar, 0)
    tgtvar = np.tile(np.nanvar(allX, 0)[:, None], (1, N)); np.fill_diagonal(tgtvar, 0)

    cf = fit_multiscale_connectome(X_list, names, ridge=3.0)
    md = fit_mdn(X_list, names, source_tau_s=20.0, k=3, epochs=400, hidden=128,
                 weight_decay=1e-2, seed=0)
    methods = {
        "source_variance": srcvar,
        "target_variance": tgtvar,
        "Pearson": pearson_lag(X_list, 1),
        "sid_mean(fast)": cf.matrices["mean_fast"],
        "sid_gain(slow)": cf.matrices["gain_slow"],
        "sid_mean+gain": _rank_composite(cf.matrices["mean_fast"], cf.matrices["gain_slow"]),
        "MDN_mean": md.matrices["mean"],
        "MDN_gain": md.matrices["gain"],
        "MDN_mean+gain": _rank_composite(md.matrices["mean"], md.matrices["gain"]),
    }
    targets = {
        "Cook_chem": cook["chem"], "Cook_gap": cook["gap"],
        "dopamine": mono["dopamine"], "serotonin": mono["serotonin"],
        "tyramine": mono["tyramine"], "octopamine": mono["octopamine"],
        "neuropeptide": npep,
    }
    return methods, targets


def main():
    OUT.mkdir(exist_ok=True)
    methods, targets = build("deconv")
    # AUROC matrix
    A = pd.DataFrame(index=list(methods), columns=list(targets), dtype=float)
    edges = {}
    for tname, gt in targets.items():
        edges[tname] = int((gt > 0).sum())
        for mname, M in methods.items():
            A.loc[mname, tname] = score_matrix(M, gt)["auroc"]

    # overall summaries across the biological (non-structural-artifact) targets
    bio = ["dopamine", "serotonin", "tyramine", "octopamine", "neuropeptide"]
    struct = ["Cook_chem", "Cook_gap"]
    A["mean_bio"] = A[bio].mean(axis=1)
    A["mean_struct"] = A[struct].mean(axis=1)
    A["mean_all"] = A[bio + struct].mean(axis=1)
    # mean rank across all targets (higher AUROC -> rank 1 = best)
    ranks = A[bio + struct].rank(axis=0, ascending=False)
    A["mean_rank"] = ranks.mean(axis=1)

    A = A.round(3)
    A.to_csv(OUT / "holistic_auroc.csv")
    print("\nedges per target:", edges)
    print("\n=== method x target AUROC (deconvolved) ===")
    print(A.to_string())
    print("\nLower mean_rank = better overall. Note near-1.0 on a noisy receptor proxy "
          "(e.g. source_variance on tyramine) is an over-fit red flag, not a win.")

    _figure(A, list(targets))
    return A


def _figure(A, targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = A.loc[:, targets].values.astype(float)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.35, vmax=0.75, aspect="auto")
    ax.set_xticks(range(len(targets))); ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_yticks(range(len(A.index))); ax.set_yticklabels(A.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="AUROC")
    ax.set_title("Holistic method x target AUROC (deconvolved; receptor targets are noisy proxies)")
    fig.tight_layout(); fig.savefig(OUT / "fig_holistic_heatmap.png", dpi=120); plt.close(fig)
    print(f"[holistic] wrote {OUT/'fig_holistic_heatmap.png'}")


if __name__ == "__main__":
    main()
