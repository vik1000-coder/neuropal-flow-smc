r"""Phase 1 of the biologically-principled lag analysis: the AUDITABLE CORRESPONDENCE CURVES.

For every SID channel (mean / gain / tail) and every reference connectome (structural + each
neuromodulator class + each specific transmitter/peptide), compute the correspondence-vs-lag
curve and draw it with the biologically-expected band shaded and the observed peak marked, so
one can SEE whether each channel peaks where biology predicts.

This is descriptive (no inferential claim yet); the null-referenced BATC score + controls come
next. Uses the CACHED per-lag effect matrices in output/lagres/effects_main.pkl (no refitting).
Everything is driven by sid_elegans/biolag/config.py.

Run: PYTHONPATH=.:SBTG ./.venv/bin/python sid_elegans/biolag/run_curves.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sid_elegans.biolag import config as C
from sid_elegans.biolag.references import build_registry, registry_table
from sid_elegans.lagres import metric as mt

ROOT = Path(__file__).resolve().parents[2]
LAGRES = ROOT / "sid_elegans" / "output" / "lagres"
OUT = ROOT / "sid_elegans" / "output" / "biolag"
CURVES = OUT / "curves"
OUT.mkdir(parents=True, exist_ok=True); CURVES.mkdir(exist_ok=True)

# Method registry: effect-key -> (display label, kind, color, linewidth, linestyle).
# kind: 'dist' = distributional channel (warm, bold, peak-marked);
#       'mean_ours' = our mean channel; 'mean_base' = classical/causal mean baseline (grey).
METHODS = {
    # distributional channels (the ones of interest)
    "SID_gain":   ("SID gain",    "dist",      "#e07b39", 2.2, "-"),
    "SID_tail":   ("SID tail",    "dist",      "#4c9a5a", 2.2, "-"),
    "MDN_gain":   ("MDN gain",    "dist",      "#c0392b", 2.2, "-"),
    # our mean channels
    "SID_mean":   ("SID mean",    "mean_ours", "#3b6fb0", 1.8, "-"),
    "MDN_mean":   ("MDN mean",    "mean_ours", "#6a5acd", 1.6, "-"),
    # classical / causal mean-only baselines
    "cross_corr": ("cross-corr",  "mean_base", "#9aa0a6", 1.0, "--"),
    "ridge_lag":  ("ridge-lag",   "mean_base", "#9aa0a6", 1.0, "--"),
    "var_partial":("VAR-partial", "mean_base", "#9aa0a6", 1.0, "--"),
    "dynotears":  ("DYNOTEARS",   "mean_base", "#9aa0a6", 1.0, "--"),
    "sbtg_mu":    ("SBTG mu-hat", "mean_base", "#8a6d3b", 1.2, ":"),
    "varlingam":  ("VAR-LiNGAM",  "mean_base", "#9aa0a6", 1.0, "--"),
    "sindy_marg": ("SINDy-marg",  "mean_base", "#9aa0a6", 1.0, "--"),
    "sindy_part": ("SINDy-part",  "mean_base", "#9aa0a6", 1.0, "--"),
    "pcmci":      ("PCMCI+",      "mean_base", "#9aa0a6", 1.0, "--"),
}
DIST_KEYS = [k for k, v in METHODS.items() if v[1] == "dist"]
FAMILY_ORDER = ["gap", "chemical", "monoamine", "neuropeptide"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})


def _safe(name):
    return name.replace(":", "__")


def load_cached():
    """Cached per-lag matrices: SID + MDN + classical (effects_main) + scratch + pcmci."""
    import numpy as np
    blob = pickle.load(open(LAGRES / "effects_main.pkl", "rb"))
    eff, meta = blob["effects"], blob["meta"]
    for fn in ["scratch_effects.npz", "pcmci_effects.npz"]:
        p = LAGRES / fn
        if p.exists():
            d = np.load(p, allow_pickle=True)
            for k in d.files:
                grp, lag = k.rsplit("_", 1)
                eff.setdefault(grp, {})[int(lag)] = d[k]
    return eff, meta["names"], meta["lags"]


def compute_curves(eff, names, lags, refs):
    """Return tidy DataFrame: method, label, kind, reference, family, level, lag_s, auroc."""
    mask = mt.eval_mask(len(names))
    rows = []
    for mkey, (label, kind, *_ ) in METHODS.items():
        if mkey not in eff:
            continue
        for r in refs:
            for l in lags:
                if l not in eff[mkey]:
                    continue
                a = mt.corr_score(eff[mkey][l], r.adjacency, mask, "auroc")
                rows.append({"method": mkey, "label": label, "kind": kind,
                             "reference": r.name, "family": r.family, "level": r.level,
                             "n_edges": r.n_edges, "expected_band": r.expected_band,
                             "expected_channel": r.expected_channel,
                             "lag_s": l / C.FPS, "auroc": a})
    return pd.DataFrame(rows)


def _shade_band(ax, band_label):
    lo, hi = C.bands()[band_label]
    ax.axvspan(lo, hi, color="#cbb", alpha=0.18, lw=0,
               label=f"expected: {band_label} ({lo:g}-{hi:g}s)")


def band_peak_summary(df):
    """Per (method, reference): peak lag (s) and whether it falls in the expected band."""
    rows = []
    for (m, ref), g in df.groupby(["method", "reference"]):
        g = g.dropna(subset=["auroc"])
        if g.empty:
            continue
        pk = g.loc[g["auroc"].idxmax()]
        lo, hi = C.bands()[pk["expected_band"]]
        rows.append({"method": m, "label": pk["label"], "kind": pk["kind"], "reference": ref,
                     "family": pk["family"], "level": pk["level"], "n_edges": int(pk["n_edges"]),
                     "expected_channel": pk["expected_channel"], "expected_band": pk["expected_band"],
                     "peak_lag_s": pk["lag_s"], "peak_auroc": round(pk["auroc"], 3),
                     "peak_in_expected_band": bool(lo <= pk["lag_s"] <= hi)})
    return pd.DataFrame(rows).sort_values(["family", "reference", "kind", "method"])


def _plot_methods(ax, sub, mark_peaks=True, small=False):
    """Draw every available method on ax: mean baselines (grey) under, dist channels (warm) over."""
    order = [k for k in METHODS if METHODS[k][1] == "mean_base"] + \
            [k for k in METHODS if METHODS[k][1] == "mean_ours"] + DIST_KEYS
    for mkey in order:
        c = sub[sub["method"] == mkey].sort_values("lag_s")
        if c.empty:
            continue
        label, kind, color, lw, ls = METHODS[mkey]
        c = c.dropna(subset=["auroc"])
        if c.empty:
            continue
        ax.plot(c["lag_s"], c["auroc"], ls, color=color, lw=lw if not small else lw * 0.7,
                marker="o" if kind == "dist" else None, ms=3 if not small else 2,
                alpha=0.9 if kind != "mean_base" else 0.5, label=label, zorder=3 if kind == "dist" else 1)
        if mark_peaks and kind == "dist":
            pk = c.loc[c["auroc"].idxmax()]
            ax.plot(pk["lag_s"], pk["auroc"], "*", color=color, ms=13 if not small else 8,
                    mec="k", mew=0.5, zorder=4)
    ax.axhline(0.5, color="k", ls=":", lw=0.9)
    ax.set_xscale("log")


def plot_reference(df, r):
    sub = df[df["reference"] == r.name]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _shade_band(ax, r.expected_band)
    _plot_methods(ax, sub)
    ax.set_xticks([0.25, 0.5, 1, 2.5, 5, 10]); ax.set_xticklabels(["0.25", "0.5", "1", "2.5", "5", "10"])
    ax.set_xlabel("lag (s, log)"); ax.set_ylabel("AUROC vs reference")
    mixed = "  [MIXED fast+slow]" if r.name in C.MIXED else ""
    ax.set_title(f"{r.name}  ({r.family}, {r.level}, {r.n_edges} edges){mixed}\n"
                 f"expected: {r.expected_channel} channel peaks in the {r.expected_band} band "
                 f"(dist = warm/bold + star; mean baselines = grey)", fontsize=9)
    ax.legend(fontsize=6.8, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(CURVES / f"{_safe(r.name)}.png", dpi=130); plt.close(fig)


def plot_overview(df, refs):
    """Small-multiples grid grouped by family: all estimators, dist bold + peak, mean baselines grey."""
    refs_sorted = sorted(refs, key=lambda r: (FAMILY_ORDER.index(r.family), -r.n_edges))
    n = len(refs_sorted); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.6 * nrow), squeeze=False)
    for idx, r in enumerate(refs_sorted):
        ax = axes[idx // ncol][idx % ncol]
        _shade_band(ax, r.expected_band)
        _plot_methods(ax, df[df["reference"] == r.name], small=True)
        ax.set_xticks([0.25, 1, 10]); ax.set_xticklabels(["0.25", "1", "10"])
        ax.set_title(f"{r.name}\n{r.n_edges} ed, exp {r.expected_band}", fontsize=7.5)
        ax.tick_params(labelsize=7); ax.get_legend().remove() if ax.get_legend() else None
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=METHODS[k][2], marker="o", ls=METHODS[k][4],
                      label=METHODS[k][0]) for k in DIST_KEYS + ["SID_mean", "MDN_mean"]]
    handles.append(Line2D([], [], color="#9aa0a6", ls="--", label="mean baselines (grey)"))
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Correspondence vs lag, ALL estimators x all references (shaded = expected band; "
                 "star = distributional peak)\nDESCRIPTIVE only -- null-referenced BATC + controls pending",
                 y=1.06, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "overview_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_index(reg_tbl, peak_tbl):
    lines = ["# Biologically-principled lag analysis -- Phase 1 (descriptive curves)", "",
             C.PROVISIONAL_NOTE, "",
             "**Status:** descriptive correspondence curves only. The null-referenced BATC score, "
             "the global-mode + surrogate controls, and the confirmatory test are the next phase.",
             "", "## Caveats (from the adversarial metric review)", ""]
    for k, v in C.CAVEATS.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Reference registry", "", reg_tbl.to_markdown(index=False), "",
              "## Peak lag per (channel, reference) vs expected band", "",
              peak_tbl.to_markdown(index=False), "",
              "## Figures", "- `overview_grid.png` -- all references at a glance",
              "- `curves/<reference>.png` -- per-reference channel curves with expected band + peaks"]
    (OUT / "README.md").write_text("\n".join(lines))


def main():
    import warnings; warnings.filterwarnings("ignore")
    eff, names, lags = load_cached()
    refs = build_registry(names, min_specific_edges=10)
    df = compute_curves(eff, names, lags, refs)
    df.to_csv(OUT / "correspondence_curves.csv", index=False)
    peak = band_peak_summary(df); peak.to_csv(OUT / "band_peak_summary.csv", index=False)
    for r in refs:
        plot_reference(df, r)
    plot_overview(df, refs)
    write_index(registry_table(refs), peak)
    print(f"[biolag] {len(refs)} references, {df['method'].nunique()} channels; "
          f"wrote curves + overview + summary to {OUT}")
    # quick console read: does the confirmatory reference behave as predicted?
    npall = peak[peak["reference"] == "neuropeptide:all"].sort_values("kind")
    print("\nneuropeptide:all peak lags (s), all estimators:")
    print(npall[["label", "kind", "peak_lag_s", "peak_auroc", "peak_in_expected_band"]].to_string(index=False))


if __name__ == "__main__":
    main()
