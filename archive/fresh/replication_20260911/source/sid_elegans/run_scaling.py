r"""Data-scaling experiment: does adding worms (OH16230 + OH15500) strengthen the
closed-form tyramine result and make the two-head reliable?

Uses the largest complete-case pool (cov>=0.9: 20 worms, 56 neurons, scale-preserving
standardization) and subsamples {6,10,14,20} worms at FIXED neurons, measuring for each
the tyramine slow-gain AUROC and the split-half stability of both the closed-form and the
two-head models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sid_elegans.combined_data import load_combined
from sid_elegans.ground_truth import load_all_monoamine_layers
from sid_elegans.multiscale import fit_multiscale_connectome
from sid_elegans.significance import auroc, perm_p
from sid_elegans.stability import split_half_stability
from sid_elegans.twohead import fit_two_head

OUT = Path(__file__).resolve().parent / "output"


def main():
    X20, names, fps = load_combined(coverage_frac=0.9, complete_case=True, verbose=True)
    mono = load_all_monoamine_layers(names)
    gt = mono["tyramine"]
    print(f"tyramine edges at {len(names)} neurons: {int((gt>0).sum())}")

    rng = np.random.default_rng(0)
    cf = lambda Xs: fit_multiscale_connectome(Xs, names, ridge=3.0).matrices["gain_slow"]
    th = lambda Xs: fit_two_head(Xs, names, source_tau_s=20.0, epochs=250, hidden=64,
                                 weight_decay=1e-1, seed=0).matrices["gain"]
    rows = []
    for nworm in [6, 10, 14, 20]:
        if nworm > len(X20):
            continue
        idx = rng.choice(len(X20), nworm, replace=False)
        Xs = [X20[i] for i in idx]
        # closed-form
        cf_auroc, cf_p = perm_p(fit_multiscale_connectome(Xs, names, ridge=3.0).matrices["gain_slow"], gt, n=1500)
        cf_stab = split_half_stability(cf, Xs, n_splits=4)["spearman_mean"]
        # two-head
        th_full = fit_two_head(Xs, names, source_tau_s=20.0, epochs=300, hidden=64,
                               weight_decay=1e-1, seed=0)
        th_auroc = auroc(th_full.matrices["gain"], gt)
        th_stab = split_half_stability(th, Xs, n_splits=4)["spearman_mean"] if nworm >= 6 else np.nan
        rows.append({"n_worms": nworm, "cf_tyramine_auroc": cf_auroc, "cf_p": cf_p,
                     "cf_stability": cf_stab, "th_tyramine_auroc": th_auroc,
                     "th_stability": th_stab})
        print(f"  n_worms={nworm:2d} | closed-form: AUROC={cf_auroc:.3f} (p={cf_p:.4f}) "
              f"stab={cf_stab:.3f} | two-head: AUROC={th_auroc:.3f} stab={th_stab:.3f}")

    json.dump(rows, open(OUT / "scaling_results.json", "w"), indent=2)
    _figure(rows)
    return rows


def _figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nw = [r["n_worms"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(nw, [r["cf_tyramine_auroc"] for r in rows], "C0-o", label="closed-form")
    a1.plot(nw, [r["th_tyramine_auroc"] for r in rows], "C1-s", label="two-head")
    a1.axhline(0.5, color="k", ls=":", lw=1)
    a1.set_xlabel("# worms"); a1.set_ylabel("tyramine gain AUROC"); a1.set_title("Recovery vs data")
    a1.legend()
    a2.plot(nw, [r["cf_stability"] for r in rows], "C0-o", label="closed-form")
    a2.plot(nw, [r["th_stability"] for r in rows], "C1-s", label="two-head")
    a2.axhline(0.8, color="green", ls=":", lw=1, label="reliable (~0.8)")
    a2.set_xlabel("# worms"); a2.set_ylabel("split-half stability (Spearman)")
    a2.set_title("Reproducibility vs data"); a2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_data_scaling.png", dpi=120)
    plt.close(fig)
    print(f"[scaling] wrote {OUT/'fig_data_scaling.png'}")


if __name__ == "__main__":
    main()
