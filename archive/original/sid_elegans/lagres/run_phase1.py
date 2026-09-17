r"""Phase 1: build the data bridge + all MAIN-venv per-lag effect matrices.

Saves output/lagres/effects_main.pkl = {method: {lag: [N,N] [post,pre]}} plus meta
(neuron names, lag grid, per-neuron variance controls, reference networks), and
/tmp/worms.npz for the scratch-venv causal methods.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from sid_elegans.combined_data import load_combined
from sid_elegans.ground_truth import (load_all_monoamine_layers, load_cook,
                                      load_neuropeptide_layer)
from sid_elegans.lagres import effects as fx

OUT = Path(__file__).resolve().parents[1] / "output" / "lagres"
LAGS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X_list, names, fps = load_combined(coverage_frac=0.6, complete_case=True,
                                       signal="deconv", verbose=True)
    N = len(names)
    Lmax = max(LAGS)

    # references (all [post,pre], aligned to names)
    cook = load_cook(names); mono = load_all_monoamine_layers(names)
    refs = {"cook_chem": cook["chem"], "cook_gap": cook["gap"],
            "dopamine": mono["dopamine"], "serotonin": mono["serotonin"],
            "tyramine": mono["tyramine"], "octopamine": mono["octopamine"],
            "neuropeptide": load_neuropeptide_layer(names)}

    # per-neuron variance controls (the trivial baselines to beat at every lag)
    allX = np.concatenate([np.nan_to_num(x - np.nanmean(x, 0), nan=0.0) for x in X_list], 0)
    node_var = allX.var(0)

    # data bridge for scratch venv
    np.savez("/tmp/worms.npz", names=np.array(names), fps=fps,
             n=len(X_list), **{f"w{i}": x for i, x in enumerate(X_list)})

    eff = {}
    def timed(tag, fn):
        t = time.time(); r = fn(); print(f"  {tag}: {time.time()-t:.1f}s"); return r

    # --- fast per-lag mean-channel ---
    eff["cross_corr"] = timed("cross_corr", lambda: {l: fx.cross_corr(X_list, l) for l in LAGS})
    eff["ridge_lag"] = timed("ridge_lag", lambda: {l: fx.ridge_lag(X_list, l) for l in LAGS})
    # joint (partial) VAR + DYNOTEARS: fit once to Lmax, extract grid lags
    vp = timed("var_partial(p=40)", lambda: fx.var_partial(X_list, Lmax))
    eff["var_partial"] = {l: vp[l] for l in LAGS if l in vp}
    dy = timed("dynotears(L=40)", lambda: fx.dynotears(X_list, Lmax, lambda_a=0.03))
    eff["dynotears"] = {l: dy[l] for l in LAGS if l in dy}

    # --- SBTG baseline (precomputed lags 1..20) ---
    eff["sbtg_mu"] = {}
    for l in LAGS:
        m = fx.sbtg_mu_hat(l, names)
        if m is not None:
            eff["sbtg_mu"][l] = m
    print(f"  sbtg_mu: lags {sorted(eff['sbtg_mu'])}")

    # --- SID distributional channels (per lag) ---
    def sid_all():
        out = {"SID_mean": {}, "SID_gain": {}, "SID_tail": {}}
        for l in LAGS:
            c = fx.sid_channels(X_list, names, l)
            for k in out:
                out[k][l] = c[k]
        return out
    sid = timed("SID mean/gain/tail (10 lags)", sid_all)
    eff.update(sid)

    # --- MDN mean & gain (per lag, torch cpu) ---
    def mdn_all():
        out = {"MDN_mean": {}, "MDN_gain": {}}
        for l in LAGS:
            c = fx.mdn_channels(X_list, names, l, epochs=250)
            out["MDN_mean"][l] = c["MDN_mean"]; out["MDN_gain"][l] = c["MDN_gain"]
        return out
    mdn = timed("MDN mean/gain (10 lags)", mdn_all)
    eff.update(mdn)

    meta = {"names": names, "lags": LAGS, "node_var": node_var, "refs": refs, "fps": fps,
            "channel_kind": fx.CHANNEL_KIND}
    with open(OUT / "effects_main.pkl", "wb") as f:
        pickle.dump({"effects": eff, "meta": meta}, f)
    print(f"[phase1] wrote {OUT/'effects_main.pkl'} — methods: {list(eff)}")
    print(f"[phase1] wrote /tmp/worms.npz — {len(X_list)} worms, N={N}")


if __name__ == "__main__":
    main()
