r"""Generate the publication figure set for the SID-elegans writeup.

All figures are rebuilt from the saved artifacts (no refitting):
  - sid_neuromod/output/synthetic/hidden_thermostat/{readouts.parquet,diagnostics.json}
  - sid_elegans/output/holistic_auroc.csv
  - sid_elegans/output/lagres/effects_main.pkl        (per-lag effect matrices + refs)
  - sid_elegans/output/lagres/contrast_logslope.json  (bootstrap within-SID contrast)

Writes vector PDF (for LaTeX) + PNG (for quick view) into paper/figures/.
Run:  PYTHONPATH=.:SBTG ./.venv/bin/python paper/make_figures.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sid_elegans.lagres import metric as mt

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
LAGRES = ROOT / "sid_elegans" / "output" / "lagres"
FPS = 4.0

# ----- consistent style -----
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 140,
})
C_MEAN = "#3b6fb0"     # blue  -> mean channel
C_GAIN = "#e07b39"     # orange-> gain channel
C_TAIL = "#4c9a5a"     # green -> tail channel
C_GREY = "#8a8a8a"
C_RED = "#c0392b"


def _save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- curves helper
def _load_effects():
    blob = pickle.load(open(LAGRES / "effects_main.pkl", "rb"))
    eff, meta = blob["effects"], blob["meta"]
    return eff, meta


def _curve(eff, refs, method, ref, lags, mask):
    return np.array([mt.corr_score(eff[method][l], refs[ref], mask, "auroc") for l in lags])


# ================================================================ FIG 1 synthetic
def fig_synthetic():
    df = pd.read_parquet(ROOT / "sid_neuromod/output/synthetic/hidden_thermostat/readouts.parquet")
    diag = json.load(open(ROOT / "sid_neuromod/output/synthetic/hidden_thermostat/diagnostics.json"))
    ts = df["timescale_s"].to_numpy()
    est, orc = df["estimate"].to_numpy(), df["oracle"].to_numpy()
    lo, hi = df["ci_low"].to_numpy(), df["ci_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ts, orc, "-", color=C_GREY, lw=2.2, label="Kalman oracle kernel", zorder=2)
    ax.fill_between(ts, lo, hi, color=C_GAIN, alpha=0.20, lw=0, zorder=1)
    ax.plot(ts, est, "o-", color=C_GAIN, lw=1.8, ms=5,
            label="SID gain channel (estimate $\\pm$95% CI)", zorder=3)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("lag / timescale (steps)")
    ax.set_ylabel(r"$\partial\,\log\mathrm{Var}[y\mid H]\,/\,\partial x_{t-u}$")
    ax.set_title("Synthetic hidden-thermostat: gain channel recovers a\n"
                 "conditional-variance memory kernel invisible to the mean")
    txt = (f"kernel corr = {diag['variance_kernel_corr']:.3f}\n"
           f"contraction: est {diag['contraction_estimate']:.3f} / "
           f"oracle {diag['contraction_oracle']:.3f}\n"
           f"mean-model $R^2$ = {diag['mean_r2']:+.3f}\n"
           f"significant gain lags: {diag['n_significant_gain_lags']}/12")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#cccccc"))
    ax.legend(loc="center right", fontsize=9)
    _save(fig, "fig1_synthetic_gain")


# ================================================================ FIG 2 holistic
def fig_holistic():
    df = pd.read_csv(ROOT / "sid_elegans/output/holistic_auroc.csv", index_col=0)
    targets = ["Cook_chem", "Cook_gap", "dopamine", "serotonin", "tyramine",
               "octopamine", "neuropeptide"]
    order = ["source_variance", "target_variance", "Pearson", "sid_mean(fast)",
             "sid_gain(slow)", "sid_mean+gain", "MDN_mean", "MDN_gain", "MDN_mean+gain"]
    df = df.loc[order]
    M = df[targets].to_numpy()
    ranks = df["mean_rank"].to_numpy()

    fig, (axh, axr) = plt.subplots(1, 2, figsize=(11.4, 4.6),
                                   gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.06})
    im = axh.imshow(M, cmap="RdYlGn", vmin=0.35, vmax=0.75, aspect="auto")
    axh.set_xticks(range(len(targets)))
    axh.set_xticklabels(targets, rotation=35, ha="right")
    axh.set_yticks(range(len(order)))
    axh.set_yticklabels(order)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axh.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                     color="black")
    axh.set_title("Directed-edge AUROC, method $\\times$ target (deconvolved)")
    axh.grid(False)
    cb = fig.colorbar(im, ax=axh, fraction=0.032, pad=0.02)
    cb.set_label("AUROC")
    # mean-rank companion (lower is better)
    ypos = np.arange(len(order))
    colors = [C_GAIN if ("sid_" in m or "MDN" in m) else (C_MEAN if m == "Pearson" else C_GREY)
              for m in order]
    axr.barh(ypos, ranks, color=colors, alpha=0.85)
    axr.set_yticks(ypos)
    axr.set_yticklabels([])
    axr.invert_yaxis()
    axr.set_xlabel("mean rank\n(lower = better)")
    axr.set_title("overall")
    best = ranks.min()
    axr.axvline(best, color="k", ls=":", lw=1)
    for y, r in zip(ypos, ranks):
        axr.text(r + 0.05, y, f"{r:.1f}", va="center", fontsize=8.5)
    axr.grid(False, axis="y")
    _save(fig, "fig2_holistic")


# ================================================================ FIG 3 control / retraction
def fig_control():
    df = pd.read_csv(ROOT / "sid_elegans/output/holistic_auroc.csv", index_col=0)
    targets = ["Cook_chem", "Cook_gap", "dopamine", "serotonin", "tyramine",
               "octopamine", "neuropeptide"]
    sv = df.loc["source_variance", targets].to_numpy()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    colors = [C_RED if t == "tyramine" else C_GREY for t in targets]
    ax.bar(range(len(targets)), sv, color=colors, alpha=0.9)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=30, ha="right")
    ax.set_ylabel("AUROC of the source-variance baseline")
    ax.set_ylim(0, 1.0)
    ax.set_title("The retracted 'tyramine' signal is a trivial artifact\n"
                 "(a no-model source-variance ranker) — yet it is the WORST method overall")
    ax.annotate("0.88 on tyramine\n(but rank 6.9/9 =\nworst method overall)",
                xy=(4, sv[4]), xytext=(2.1, 0.80), fontsize=9, color=C_RED,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
    _save(fig, "fig3_source_variance_control")


# ================================================================ FIG 4 lag curves (within-SID)
def fig_lagcurves():
    eff, meta = _load_effects()
    names, lags, refs = meta["names"], meta["lags"], meta["refs"]
    mask = mt.eval_mask(len(names))
    lag_s = np.array(lags) / FPS

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=False)
    for ax, ref, title in zip(axes, ["serotonin", "neuropeptide"],
                              ["serotonin (metabotropic monoamine)",
                               "neuropeptide (peptidergic, 757 edges)"]):
        cm = _curve(eff, refs, "SID_mean", ref, lags, mask)
        cg = _curve(eff, refs, "SID_gain", ref, lags, mask)
        ct = _curve(eff, refs, "SID_tail", ref, lags, mask)
        for c, col, lab in [(cm, C_MEAN, "SID mean"), (cg, C_GAIN, "SID gain"),
                            (ct, C_TAIL, "SID tail")]:
            ax.plot(lag_s, c, "o-", color=col, lw=1.9, ms=5, label=lab)
            pk = lag_s[int(np.nanargmax(c))]
            ax.axvline(pk, color=col, ls=":", lw=1.0, alpha=0.7)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("lag (s, log scale)")
        ax.set_title(f"vs {title}")
        ax.set_xticks([0.25, 0.5, 1, 2.5, 5, 10])
        ax.set_xticklabels(["0.25", "0.5", "1", "2.5", "5", "10"])
    axes[0].set_ylabel("AUROC vs reference")
    axes[0].legend(loc="upper right")
    fig.suptitle("Within the same SID fit, the distributional (gain/tail) channels' correspondence "
                 "peaks at LONGER lags than the mean channel's", y=1.02, fontsize=12)
    _save(fig, "fig4_lag_curves")


# ================================================================ FIG 5 headline contrast
def fig_contrast():
    d = json.load(open(LAGRES / "contrast_logslope.json"))
    refs = ["cook_chem", "serotonin", "dopamine", "neuropeptide"]
    labels = ["Cook chemical\n(structural CONTROL)", "serotonin", "dopamine",
              "neuropeptide\n(peptidergic)"]
    x = np.arange(len(refs))
    w = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for k, (chan, col, off) in enumerate([("gain_minus_mean", C_GAIN, -w / 2),
                                          ("tail_minus_mean", C_TAIL, +w / 2)]):
        mean = [d[r][chan]["mean"] for r in refs]
        lo = [d[r][chan]["mean"] - d[r][chan]["lo"] for r in refs]
        hi = [d[r][chan]["hi"] - d[r][chan]["mean"] for r in refs]
        sig = [d[r][chan]["lo"] > 0 or d[r][chan]["hi"] < 0 for r in refs]
        for i in range(len(refs)):
            ec = "black" if sig[i] else "none"
            ax.errorbar(x[i] + off, mean[i], yerr=[[lo[i]], [hi[i]]], fmt="o",
                        color=col, ms=8 if sig[i] else 6, capsize=4, lw=1.8,
                        markeredgecolor=ec, markeredgewidth=1.4 if sig[i] else 0,
                        label=("gain $-$ mean" if chan.startswith("gain") else "tail $-$ mean")
                        if i == 0 else None)
            if sig[i]:
                ax.annotate("$\\ast$", (x[i] + off, mean[i] + hi[i]), ha="center",
                            va="bottom", fontsize=15, color=col)
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-0.0005, 0.0005, color="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta$ log-slope of AUROC-vs-lag" "\n"
                  r"(distributional $-$ mean channel)")
    ax.set_title("Channel $\\times$ reference double dissociation (paired within-SID, "
                 "worm bootstrap)\n>0 = distributional channel leans to longer lags; "
                 "$\\ast$ = 95% CI excludes 0")
    ax.legend(loc="upper left")
    _save(fig, "fig5_contrast_double_dissociation")


# ================================================================ FIG 6 structural validation
def fig_structural():
    eff, meta = _load_effects()
    names, lags, refs = meta["names"], meta["lags"], meta["refs"]
    mask = mt.eval_mask(len(names))
    lag_s = np.array(lags) / FPS
    mean_methods = ["cross_corr", "ridge_lag", "var_partial", "sbtg_mu",
                    "SID_mean", "MDN_mean"]
    labelmap = {"cross_corr": "cross-corr", "ridge_lag": "ridge-lag",
                "var_partial": "VAR (partial)", "sbtg_mu": "SBTG $\\hat\\mu$",
                "SID_mean": "SID mean", "MDN_mean": "MDN mean"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m in mean_methods:
        ml = [l for l in lags if l in eff[m]]
        c = np.array([mt.corr_score(eff[m][l], refs["cook_chem"], mask, "auroc") for l in ml])
        ax.plot(np.array(ml) / FPS, c, "o-", lw=1.5, ms=4, alpha=0.85, label=labelmap[m])
    ax.axhline(0.5, color="k", ls=":", lw=1.2, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("lag (s, log scale)")
    ax.set_ylabel("AUROC vs Cook chemical connectome")
    ax.set_xticks([0.25, 0.5, 1, 2.5, 5, 10]); ax.set_xticklabels(["0.25", "0.5", "1", "2.5", "5", "10"])
    ax.set_ylim(0.42, 0.60)
    ax.set_title("Validation: every mean-channel method (classical + causal) sits\n"
                 "near chance on the wired connectome, with a shared weak short-lag preference")
    ax.legend(ncol=2, fontsize=9)
    _save(fig, "fig6_structural_validation")


# ================================================================ FIG 7 robustness forest plot
def fig_robustness():
    """Neuropeptide gain-mean log-slope across the worm-count x conditioning sweep."""
    head = json.load(open(LAGRES / "contrast_logslope.json"))
    rob = json.load(open(Path(__file__).resolve().parent / "robustness_worms.json"))
    cmp = json.load(open(Path(__file__).resolve().parent / "compare_approaches.json"))

    def g(d):  # (mean, lo, hi)
        return d["mean"], d["lo"], d["hi"]
    rows = [
        ("6w / 80n  (headline, max power)", "partial",
         g(head["neuropeptide"]["gain_minus_mean"]), g(head["cook_chem"]["gain_minus_mean"])),
        ("14w / 60n", "partial",
         g(rob["14w_60n"]["refs"]["neuropeptide"]["gain_minus_mean"]),
         g(rob["14w_60n"]["refs"]["cook_chem"]["gain_minus_mean"])),
        ("20w / 56n", "partial",
         g(rob["20w_56n"]["refs"]["neuropeptide"]["gain_minus_mean"]),
         g(rob["20w_56n"]["refs"]["cook_chem"]["gain_minus_mean"])),
        ("28w / 84n  donor-impute", "partial",
         g(cmp["approaches"]["donor_multivariate"]["neuropeptide"]["gain_minus_mean"]),
         g(cmp["approaches"]["donor_multivariate"]["cook_chem"]["gain_minus_mean"])),
        ("28w / 84n  pairwise", "marginal",
         g(cmp["approaches"]["pairwise"]["neuropeptide"]["gain_minus_mean"]),
         g(cmp["approaches"]["pairwise"]["cook_chem"]["gain_minus_mean"])),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, kind, npv, ctl) in zip(y, rows):
        col = C_GAIN if kind == "partial" else C_GREY
        sig = npv[1] > 0 or npv[2] < 0
        # neuropeptide
        ax.errorbar(npv[0], yi + 0.13, xerr=[[npv[0] - npv[1]], [npv[2] - npv[0]]], fmt="o",
                    color=col, ms=9 if sig else 6, capsize=4, lw=2,
                    markeredgecolor="black" if sig else "none", markeredgewidth=1.4 if sig else 0)
        if sig:
            ax.annotate("$\\ast$", (npv[2], yi + 0.13), fontsize=15, va="center", ha="left",
                        color=col)
        # control (cook chem), hollow small grey
        ax.errorbar(ctl[0], yi - 0.15, xerr=[[ctl[0] - ctl[1]], [ctl[2] - ctl[0]]], fmt="s",
                    color="white", ecolor="#b0b0b0", ms=6, capsize=3, lw=1.2,
                    markeredgecolor="#888", markeredgewidth=1.1)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel(r"$\Delta$ log-slope (gain $-$ mean),  neuropeptide (filled) & Cook-chem control (hollow)")
    ax.set_title("Robustness of the peptidergic gain$-$mean lag-shift across worm count "
                 "and conditioning\n(filled = neuropeptide; hollow grey = structural control; "
                 "$\\ast$ = 95% CI excludes 0)", fontsize=10.5)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", color=C_GAIN, ls="", ms=8, label="partial (neuropeptide)"),
        Line2D([], [], marker="o", color=C_GREY, ls="", ms=8, label="marginal (neuropeptide)"),
        Line2D([], [], marker="s", color="white", markeredgecolor="#888", ls="", ms=7,
               label="structural control")], loc="lower right", fontsize=8.5)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    _save(fig, "fig7_robustness_forest")


# ================================================================ FIG 8 surrogate-null adjudication
def fig_surrogate():
    """Observed within-SID contrast vs the circular-shift surrogate null, per reference."""
    adj = json.load(open(Path(__file__).resolve().parent / "adjudicate.json"))
    sur = adj["surrogate"]
    refs = ["cook_chem", "serotonin", "dopamine", "neuropeptide"]
    labels = ["Cook chem\n(control)", "serotonin", "dopamine", "neuropeptide\n(peptidergic)"]
    y = np.arange(len(refs))[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for yi, r in zip(y, refs):
        s = sur[r]
        # null band
        ax.plot([s["null_gain_lo"], s["null_gain_hi"]], [yi, yi], color="#b8b8b8", lw=7,
                solid_capstyle="round", alpha=0.7,
                label="surrogate null (95%)" if r == "cook_chem" else None)
        ax.plot(s["null_gain_mean"], yi, "|", color="#666", ms=12, mew=2)
        # real value
        real = s["real_gain_minus_mean"]
        outside = real > s["null_gain_hi"] or real < s["null_gain_lo"]
        ax.plot(real, yi, "o", color=C_GAIN, ms=11 if outside else 8,
                markeredgecolor="black" if outside else "none", markeredgewidth=1.5,
                label="observed (real data)" if r == "cook_chem" else None, zorder=5)
        ax.annotate(f"p={s['p_gain']:.3f}", (max(real, s['null_gain_hi']), yi),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9,
                    color="black" if outside else "#888",
                    fontweight="bold" if outside else "normal")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\Delta$ log-slope (gain $-$ mean)")
    ax.set_title("Surrogate null: destroy cross-neuron coupling, refit the same partial SID\n"
                 "(only the peptidergic effect lies outside its null $\\Rightarrow$ not an "
                 "estimator artifact)", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.028, 0.045)
    _save(fig, "fig8_surrogate_null")


if __name__ == "__main__":
    fig_synthetic()
    fig_holistic()
    fig_control()
    fig_lagcurves()
    fig_contrast()
    fig_structural()
    fig_robustness()
    fig_surrogate()
    print("\nall figures written to", FIG)
