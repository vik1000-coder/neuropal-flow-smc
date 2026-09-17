r"""Significance analysis for the headline comparisons.

Because several monoamine layers have few edges (9-28), raw AUROC differences are noisy.
We add:
  - a permutation-null p-value that a channel's AUROC exceeds chance, and
  - a paired bootstrap CI on the AUROC difference between two channels/methods (resample
    the off-diagonal edge set with replacement).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from sid_elegans.data import load_traces
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.evaluate import _offdiag
from sid_elegans.ground_truth import load_all_monoamine_layers, load_cook
from sid_elegans.baselines import pearson_lag

OUT = Path(__file__).resolve().parent / "output"


def _vecs(M, gt):
    ys = np.abs(_offdiag(M)); yt = (_offdiag(gt) > 0).astype(int)
    v = np.isfinite(ys)
    return ys[v], yt[v]


def auroc(M, gt):
    ys, yt = _vecs(M, gt)
    if yt.sum() in (0, len(yt)):
        return np.nan
    return roc_auc_score(yt, ys)


def perm_p(M, gt, n=2000, seed=0):
    ys, yt = _vecs(M, gt)
    if yt.sum() in (0, len(yt)):
        return np.nan, np.nan
    obs = roc_auc_score(yt, ys)
    rng = np.random.default_rng(seed)
    ge = sum(roc_auc_score(yt, rng.permutation(ys)) >= obs for _ in range(n))
    return obs, (ge + 1) / (n + 1)


def boot_diff(Ma, Mb, gt, n=2000, seed=0):
    """Bootstrap CI on AUROC(Ma) - AUROC(Mb) over the shared edge set."""
    ysa, yt = _vecs(Ma, gt)
    ysb, _ = _vecs(Mb, gt)
    rng = np.random.default_rng(seed)
    m = len(yt)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        yti = yt[idx]
        if yti.sum() in (0, len(yti)):
            continue
        diffs.append(roc_auc_score(yti, ysa[idx]) - roc_auc_score(yti, ysb[idx]))
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main():
    X_list, names, fps = load_traces()
    cook = load_cook(names)
    mono = load_all_monoamine_layers(names)

    lines = ["", "## Significance analysis", ""]

    # --- Structural: sid_mean vs pearson (lag 1) ---
    r1 = fit_distributional_connectome(X_list, names, lag=1, ridge=1e-2)
    P = pearson_lag(X_list, 1)
    a_mean, p_mean = perm_p(r1.matrices["mean"], cook["struct"])
    a_pear, p_pear = perm_p(P, cook["struct"])
    md, lo, hi = boot_diff(P, r1.matrices["mean"], cook["struct"])
    lines.append("**Cook structural (lag 1)** — permutation p that AUROC > chance:")
    lines.append(f"- sid_mean AUROC={a_mean:.3f} (p={p_mean:.3g}); "
                 f"pearson AUROC={a_pear:.3f} (p={p_pear:.3g})")
    lines.append(f"- AUROC(pearson) - AUROC(sid_mean) = {md:+.3f} "
                 f"[95% CI {lo:+.3f}, {hi:+.3f}]"
                 f"{'  (pearson significantly higher)' if lo > 0 else '  (not significant)'}")
    lines.append("")

    # --- Monoamine: mean vs gain at a slow timescale (thesis test) ---
    lines.append("**Neuromodulator recovery** — slow-filter sources; is the *gain* channel "
                 "above chance, and does it beat the *mean* channel?\n")
    lines.append("| transmitter | tau_s | n_edges | AUROC mean | AUROC gain | gain p | "
                 "gain−mean [95% CI] |")
    lines.append("|---|--:|--:|--:|--:|--:|---|")
    tau_by_nt = {"dopamine": 8.0, "tyramine": 20.0, "serotonin": 5.0,
                 "octopamine": 2.0, "monoamine_all": 20.0}
    for nt, tau_s in tau_by_nt.items():
        res = fit_distributional_connectome(X_list, names, lag=1, ridge=1e-2,
                                            source_tau=tau_s * fps)
        gt = mono[nt]
        a_m = auroc(res.matrices["mean"], gt)
        a_g, p_g = perm_p(res.matrices["gain"], gt)
        md, lo, hi = boot_diff(res.matrices["gain"], res.matrices["mean"], gt)
        star = "**" if lo > 0 else ""
        lines.append(f"| {nt} | {tau_s:g} | {int((gt>0).sum())} | {a_m:.3f} | "
                     f"{star}{a_g:.3f}{star} | {p_g:.3g} | "
                     f"{md:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines.append("")
    lines.append("*Bold gain AUROC = its 95% bootstrap advantage over the mean channel "
                 "excludes 0.*")

    txt = (OUT / "REPORT.md").read_text() + "\n".join(lines)
    (OUT / "REPORT.md").write_text(txt)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
