r"""Figures for Phase 2 band-concordance results (reads output/biolag/phase2_results.json).

fig_phase2_forest.png   -- BC_gain (bootstrap CI) across all references, confirmatory highlighted,
                           surrogate-significant starred, source-variance control at 0.
fig_phase2_confirm.png  -- neuropeptide BC_mean / BC_gain / dBC(gain-mean), PRIMARY vs
                           global-mode-removed, with the gap/chem + source-variance anchors.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "sid_elegans" / "output" / "biolag"
FAMILY_COLOR = {"gap": "#7a7a7a", "chemical": "#9aa0a6", "monoamine": "#8e44ad",
                "neuropeptide": "#e07b39"}
FAMILY_ORDER = ["gap", "chemical", "monoamine", "neuropeptide"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})


def _refs(block):
    return [k for k in block if not k.startswith("_")]


def fig_forest(res):
    block = res["primary"]
    refs = sorted(_refs(block), key=lambda k: (FAMILY_ORDER.index(block[k]["family"]),
                                               -block[k]["n_edges"]))
    y = np.arange(len(refs))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for yi, k in zip(y, refs):
        d = block[k]; b = d["BC_gain_boot"]
        col = FAMILY_COLOR[d["family"]]
        sig = d["p_BCgain_surrogate"] < 0.05 and (b["lo"] > 0 or b["hi"] < 0)
        ax.errorbar(b["mean"], yi, xerr=[[b["mean"] - b["lo"]], [b["hi"] - b["mean"]]],
                    fmt="o", color=col, ms=8 if sig else 6, capsize=3, lw=1.8,
                    markeredgecolor="black" if sig else "none", markeredgewidth=1.3 if sig else 0)
        star = "  $\\ast$" if sig else ""
        conf = "  ← CONFIRMATORY" if k == res["config"]["primary"] else ""
        ax.text(ax.get_xlim()[1], yi, f"  {k} ({d['n_edges']}ed, p={d['p_BCgain_surrogate']:.3f}){star}{conf}",
                va="center", fontsize=7.8, color="black")
    ax.axvline(0, color="k", lw=1)
    sv = block["_control_source_variance_BC"]
    ax.axvline(sv, color="crimson", ls=":", lw=1.4)
    ax.text(sv, len(refs) - 0.4, f" source-variance control = {sv:+.2f}", color="crimson",
            fontsize=8, rotation=90, va="top")
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel(r"$BC_{\rm gain}$ = mean $z$(expected band) $-$ mean $z$(complementary band)")
    ax.set_title("Phase 2: does each reference's GAIN channel concentrate its above-null\n"
                 "correspondence in the biologically-expected band? (bootstrap CI; "
                 "$\\ast$=surrogate p<0.05 & CI excludes 0)", fontsize=10)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 2.6)
    fig.tight_layout(); fig.savefig(OUT / "fig_phase2_forest.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_confirm(res):
    prim, gm = res["primary"], res["global_mode_removed"]
    P = res["config"]["primary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, block, title in [(axes[0], prim, "PRIMARY (all 6 worms)"),
                             (axes[1], gm, "GLOBAL brain-state mode removed")]:
        d = block[P]
        items = [("BC_mean", d["BC_mean_boot"], "#3b6fb0", d.get("p_BCgain_surrogate")),
                 ("BC_gain", d["BC_gain_boot"], "#e07b39", d["p_BCgain_surrogate"]),
                 ("dBC (gain-mean)", d["dBC_gain_boot"], "#c0392b", d["p_dBC_surrogate"])]
        x = np.arange(len(items))
        for xi, (lab, b, col, p) in zip(x, items):
            sig = (b["lo"] > 0 or b["hi"] < 0)
            ax.errorbar(xi, b["mean"], yerr=[[b["mean"] - b["lo"]], [b["hi"] - b["mean"]]],
                        fmt="o", color=col, ms=9 if sig else 6, capsize=4, lw=2,
                        markeredgecolor="black" if sig else "none", markeredgewidth=1.4 if sig else 0)
            if p is not None:
                ax.annotate(f"p={p:.3f}", (xi, b["hi"]), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=8.5,
                            fontweight="bold" if p < 0.05 else "normal")
        # anchors
        ax.axhline(0, color="k", lw=1)
        ax.plot(-0.6, block["_control_source_variance_BC"], "s", color="crimson", ms=7)
        ax.text(-0.6, block["_control_source_variance_BC"], " src-var\n ctrl", color="crimson",
                fontsize=7, va="center")
        ax.plot([3.4, 3.6], [block["structural:gap"]["BC_gain_obs"],
                             block["structural:chemical"]["BC_gain_obs"]], "v", color="#7a7a7a", ms=7)
        ax.text(3.5, block["structural:gap"]["BC_gain_obs"], " gap/chem\n gain (fast ref)",
                fontsize=7, va="center", color="#555")
        ax.set_xticks(x); ax.set_xticklabels([i[0] for i in items], fontsize=9)
        ax.set_xlim(-1.1, 4.2)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("band concordance (z units)")
    fig.suptitle("Confirmatory test — neuropeptide (peptidergic, 757 edges): gain vs mean band "
                 "concordance\n(dBC>0 = gain more slow-concentrated than mean; must survive "
                 "global-mode removal + controls null)", y=1.06, fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "fig_phase2_confirm.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    res = json.load(open(OUT / "phase2_results.json"))
    fig_forest(res); fig_confirm(res)
    print("wrote fig_phase2_forest.png, fig_phase2_confirm.png")


if __name__ == "__main__":
    main()
