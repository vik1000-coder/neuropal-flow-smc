r"""#1 + #3 on the DECONVOLVED (continuous activity) signal, controls-first.

Deconvolve the calcium to an activity estimate, then run the closed-form distributional
readouts AND the two-head, scored against Cook + monoamine — with the trivial controls
(source-variance, target-variance, Pearson) computed up front so we can see whether ANY
sophisticated method beats them once the calcium confound is removed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sid_elegans.baselines import pearson_lag
from sid_elegans.combined_data import load_combined
from sid_elegans.evaluate import score_matrix
from sid_elegans.ground_truth import load_all_monoamine_layers, load_cook
from sid_elegans.multiscale import fit_multiscale_connectome
from sid_elegans.significance import perm_p
from sid_elegans.stability import split_half_stability

OUT = Path(__file__).resolve().parent / "output"


def _variance_baselines(X_list, N):
    allX = np.concatenate(X_list, axis=0)
    v = np.nanvar(allX, axis=0)
    srcvar = np.tile(v[None, :], (N, 1)); np.fill_diagonal(srcvar, 0)   # [post,pre]: col=source var
    tgtvar = np.tile(v[:, None], (1, N)); np.fill_diagonal(tgtvar, 0)   # row=target var
    return srcvar, tgtvar


def _methods(X_list, names):
    N = len(names)
    srcvar, tgtvar = _variance_baselines(X_list, N)
    res = fit_multiscale_connectome(X_list, names, ridge=3.0)
    m = {
        "source_variance": srcvar,
        "target_variance": tgtvar,
        "pearson": pearson_lag(X_list, 1),
        "sid_mean_fast": res.matrices["mean_fast"],
        "sid_gain_slow": res.matrices["gain_slow"],
    }
    return m


def run(signal="deconv", include_two_head=True):
    X_list, names, fps = load_combined(coverage_frac=0.6, complete_case=True, signal=signal)
    cook = load_cook(names); mono = load_all_monoamine_layers(names)
    N = len(names)
    targets = {"Cook_struct": cook["struct"], **{f"mono_{k}": v for k, v in mono.items()}}
    methods = _methods(X_list, names)

    if include_two_head:
        from sid_elegans.twohead import fit_two_head
        th = fit_two_head(X_list, names, source_tau_s=20.0, epochs=300, hidden=64,
                          weight_decay=1e-1, seed=0)
        methods["two_head_gain"] = th.matrices["gain"]

    rows = []
    for mname, M in methods.items():
        for tname, gt in targets.items():
            s = score_matrix(M, gt)
            rows.append({"signal": signal, "method": mname, "target": tname, **s})
    df = pd.DataFrame(rows)

    # significance vs chance for the key gain-channel results
    for nt in ["tyramine", "dopamine", "serotonin", "octopamine"]:
        a, p = perm_p(methods["sid_gain_slow"], mono[nt], n=2000)
        rows.append({"signal": signal, "method": "sid_gain_slow", "target": f"perm_{nt}",
                     "auroc": a, "auprc": np.nan, "spearman": np.nan, "f1": np.nan,
                     "n_eval": 0, "n_pos": int((mono[nt] > 0).sum()), "perm_p": p})
    return df, methods, mono, cook, X_list, names


def main():
    OUT.mkdir(exist_ok=True)
    # run on both raw and deconvolved so the effect of deconvolution is explicit
    all_df = []
    for signal in ["raw", "deconv"]:
        df, methods, mono, cook, X_list, names = run(signal=signal)
        all_df.append(df)
        print(f"\n=== signal={signal} — AUROC (off-diagonal directed) ===")
        piv = df[df.target.isin(["Cook_struct", "mono_tyramine", "mono_serotonin",
                                 "mono_dopamine"])].pivot_table(
            index="method", columns="target", values="auroc")
        print(piv.round(3).to_string())
        # gain-channel significance vs chance + vs source-variance baseline
        for nt in ["tyramine", "serotonin", "dopamine"]:
            ag, pg = perm_p(methods["sid_gain_slow"], mono[nt], n=2000)
            asv = score_matrix(methods["source_variance"], mono[nt])["auroc"]
            print(f"   {nt}: sid_gain={ag:.3f} (p={pg:.4f}) vs source_variance={asv:.3f}")
    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(OUT / "deconv_results.csv", index=False)
    print(f"\n[deconv] wrote {OUT/'deconv_results.csv'}")
    return out


if __name__ == "__main__":
    main()
