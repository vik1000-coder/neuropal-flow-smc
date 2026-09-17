r"""Phase 3 (SCRATCH venv): PCMCI+ per-lag |partial-corr| effect matrices, [post,pre].
Run: /tmp/baseline_feas/bin/python sid_elegans/lagres/run_pcmci.py <tau_max> [pc_alpha]
Reads /tmp/worms.npz, writes /tmp/pcmci_effects.npz (keys 'pcmci_<tau>').
E(tau)[j,i] = |val_matrix[i,j,tau]| (i->j) -> [post,pre].
"""
import sys
import time

import numpy as np


def _data(X_list, names):
    from tigramite import data_processing as pp
    data = {}
    for w, x in enumerate(X_list):
        x = np.asarray(x, float).copy()
        col = np.nanmean(x, 0); ii = np.where(np.isnan(x))
        x[ii] = np.take(col, ii[1])
        data[w] = x
    return pp.DataFrame(data, analysis_mode='multiple', var_names=list(names))


def run(tau_max, pc_alpha=0.05, max_conds_dim=1, max_conds_px=1):
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    d = np.load("/tmp/worms.npz", allow_pickle=True)
    n = int(d["n"]); X_list = [d[f"w{i}"] for i in range(n)]
    names = [str(s) for s in d["names"]]
    df = _data(X_list, names)
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    t = time.time()
    res = pcmci.run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha,
                              max_conds_dim=max_conds_dim,
                              max_conds_px=max_conds_px, max_conds_px_lagged=max_conds_px)
    el = time.time() - t
    val = res['val_matrix']
    out = {}
    for tau in range(1, tau_max + 1):
        e = np.abs(val[:, :, tau]).T.copy(); np.fill_diagonal(e, 0.0)
        out[f"pcmci_{tau}"] = e
    print(f"  PCMCI+ N={len(names)} tau_max={tau_max}: {el:.1f}s")
    return out, el


if __name__ == "__main__":
    tau_max = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    out, el = run(tau_max, pc_alpha=alpha)
    np.savez("/tmp/pcmci_effects.npz", **out)
    print(f"  wrote /tmp/pcmci_effects.npz — {len(out)} matrices ({el:.0f}s)")
