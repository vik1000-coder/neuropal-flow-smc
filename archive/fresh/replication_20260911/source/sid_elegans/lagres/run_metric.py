r"""Phase 4: lag-resolved correspondence curves + relative analysis + controls + figures.

Loads all per-lag effect matrices (main + scratch + pcmci), scores each against every
reference network as a curve over lag, applies the within-method relative normalization and
the per-lag variance controls, computes shape statistics, and writes figures + a summary CSV.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sid_elegans.lagres import effects as fx
from sid_elegans.lagres import metric as mt

OUT = Path(__file__).resolve().parents[1] / "output" / "lagres"
SHORT_MAX, LONG_MIN = 3, 15   # frames: short <=3 (<=0.75s), long >=15 (>=3.75s)


def _load():
    """Load Phase-1 main effects + fold in scratch (VAR-LiNGAM/SINDy) and PCMCI+ effects."""
    with open(OUT / "effects_main.pkl", "rb") as f:
        blob = pickle.load(f)
    eff, meta = blob["effects"], blob["meta"]
    # fold in scratch (varlingam_L, sindy_marg_L, sindy_part_L): prefer the persistent
    # copies in output/lagres, fall back to the /tmp working files.
    def _find(name):
        for p in (OUT / name, Path("/tmp") / name):
            if p.exists():
                return p
        return None
    sp = _find("scratch_effects.npz")
    if sp is not None:
        d = np.load(sp, allow_pickle=True)
        for grp in ["varlingam", "sindy_marg", "sindy_part"]:
            eff.setdefault(grp, {})
            for k in d.files:
                if k.startswith(grp + "_"):
                    eff[grp][int(k.split("_")[-1])] = d[k]
    pc = _find("pcmci_effects.npz")
    if pc is not None:
        d = np.load(pc, allow_pickle=True)
        eff["pcmci"] = {int(k.split("_")[-1]): d[k] for k in d.files}
    return eff, meta


MEAN_METHODS = ["cross_corr", "ridge_lag", "var_partial", "dynotears", "sbtg_mu",
                "varlingam", "sindy_marg", "sindy_part", "pcmci", "SID_mean", "MDN_mean"]
DIST_METHODS = ["SID_gain", "SID_tail", "MDN_gain"]


def main():
    """Score every method's per-lag matrices against every reference, write curves,
    shape-statistics, relative/variance-control summaries, figures, and SYNTHESIS.md."""
    eff, meta = _load()
    names, LAGS, node_var, refs = meta["names"], meta["lags"], meta["node_var"], meta["refs"]
    N = len(names)
    mask = mt.eval_mask(N)
    methods = [m for m in MEAN_METHODS + DIST_METHODS if m in eff]
    print(f"methods present: {methods}")

    rows = []
    curves = {}   # (method, ref) -> (lags_used, auroc curve)
    for m in methods:
        lags_m = [l for l in LAGS if l in eff[m]]
        kind = "dist" if m in DIST_METHODS else "mean"
        for rname, R in refs.items():
            c = np.array([mt.corr_score(eff[m][l], R, mask, "auroc") for l in lags_m])
            curves[(m, rname)] = (lags_m, c)
            # excess over the trivial source/target-variance control (constant per lag here)
            csrc = mt.corr_score(mt.control_matrix(s_src=node_var), R, mask, "auroc")
            ctgt = mt.corr_score(mt.control_matrix(s_tgt=node_var), R, mask, "auroc")
            ctrl = np.nanmax([csrc, ctgt])
            ss = mt.shape_stats(c, lags_m, SHORT_MAX, LONG_MIN)
            rows.append({"method": m, "kind": kind, "reference": rname,
                         "auroc_mean": float(np.nanmean(c)), "auroc_max": float(np.nanmax(c)),
                         "peak_lag_fr": ss["peak_lag_fr"], "logslope": ss["logslope"],
                         "fast_slow_contrast": ss["fast_slow_contrast"],
                         "excess_over_var_max": float(np.nanmax(c) - ctrl),
                         "control_auroc": float(ctrl)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lagres_summary.csv", index=False)

    _figures(curves, meta, methods)
    _synthesis(df)
    return df


def _figures(curves, meta, methods):
    """Write the correspondence-vs-lag figures (structural mean-channel panel; mean-vs-
    distributional panels for serotonin & neuropeptide, raw and relative-normalized)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fps = meta["fps"]

    def plot_ref(rname, method_list, title, fname, normalize=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in method_list:
            if (m, rname) not in curves:
                continue
            lags, c = curves[(m, rname)]
            y = mt.rel_zscore(c) if normalize else c
            ls = "-o" if fx.CHANNEL_KIND.get(m, "mean") == "mean" else "-s"
            lw = 1.2 if fx.CHANNEL_KIND.get(m, "mean") == "mean" else 2.4
            ax.plot(np.array(lags) / fps, y, ls, label=m, lw=lw, markersize=4,
                    alpha=0.85)
        if not normalize:
            ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("relative AUROC (z, across lags)" if normalize else "AUROC vs " + rname)
        ax.set_title(title); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=120); plt.close(fig)

    present_mean = [m for m in MEAN_METHODS if m in methods]
    dist = [m for m in DIST_METHODS if m in methods]
    # (a) validation: mean-channel methods on structural connectome
    plot_ref("cook_chem", present_mean, "Mean-channel methods vs Cook chemical (structural)",
             "fig_lag_structural_mean.png")
    # (b) modulatory: mean vs distributional on serotonin + neuropeptide
    for ref in ["serotonin", "neuropeptide"]:
        plot_ref(ref, present_mean + dist, f"Mean vs distributional channels vs {ref}",
                 f"fig_lag_{ref}.png")
        plot_ref(ref, ["SID_mean", "MDN_mean", "SID_gain", "MDN_gain", "SID_tail"],
                 f"SID/MDN mean vs gain/tail vs {ref} (relative)", f"fig_lag_{ref}_rel.png",
                 normalize=True)
    print(f"[phase4] wrote figures to {OUT}")


def _synthesis(df):
    """Write SYNTHESIS.md: the structural validation table + modulatory-reference tables
    (distributional vs mean channel log-slopes)."""
    lines = ["# Lag-resolved correspondence — synthesis", ""]
    # H1 validation: do mean-channel methods agree that structural match peaks short (neg logslope)?
    struct = df[(df.reference == "cook_chem") & (df.kind == "mean")]
    lines.append("## Structural connectome (cook_chem), mean-channel methods")
    lines.append("logslope<0 = match stronger at SHORT lags (fast synaptic); peak_lag in frames.\n")
    lines.append(struct[["method", "auroc_mean", "peak_lag_fr", "logslope",
                         "fast_slow_contrast"]].round(3).to_string(index=False))
    # H2 payoff: distributional channels vs metabotropic/peptidergic refs
    lines.append("\n## Modulatory refs — distributional vs mean channel (logslope>0 = SLOW)")
    for ref in ["serotonin", "neuropeptide", "dopamine"]:
        sub = df[df.reference == ref]
        lines.append(f"\n### {ref}")
        lines.append(sub[["method", "kind", "auroc_mean", "auroc_max", "peak_lag_fr",
                          "logslope", "excess_over_var_max"]].round(3).to_string(index=False))
    (OUT / "SYNTHESIS.md").write_text("\n".join(lines))
    print(f"[phase4] wrote {OUT/'SYNTHESIS.md'}")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
