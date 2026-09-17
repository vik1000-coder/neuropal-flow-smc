r"""Phase 2 — inferential band-concordance (coarse fast/slow), with the controls the
adversarial review demanded.

Per (channel c, reference r) we form the AUROC-vs-lag curve C_c(k), null-reference it against a
circular-shift surrogate into z_c(k) = (C_c - mu0)/sd0 (regenerated at the scored worm count),
and score BAND CONCORDANCE:

    BC_c,r = mean_{k in expected band} z_c(k) - mean_{k in complementary band} z_c(k)

(expected band = 'slow' for modulatory refs, 'fast' for structural). BC>0 = the channel's
above-null correspondence is concentrated in the biologically-expected band.

Headline (pre-registered): within one SID fit, dBC_r = BC_gain - BC_mean for neuropeptide, with
(i) worm-bootstrap CI, (ii) BC_gain independently clearing the surrogate null, (iii) positive
controls (source-variance -> 0; gain on gap/chem -> not slow) returning null, and (iv) survival
after removing the shared global brain-state mode. Everything else is exploratory.

Coarse fast/slow only (calcium-kernel floor makes finer bands unresolvable); no log-Gaussian
template (avoids researcher-DOF); relative/within-fit quantities only (absolute seconds untrusted).
"""
from __future__ import annotations

import numpy as np

from sid_elegans.biolag import config as C
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.lagres import metric as mt

LAGS = C.LAG_FRAMES


def _band_masks():
    """Boolean masks over LAGS for the active fast/slow bands."""
    bs = C.bands()
    sec = np.array(LAGS) / C.FPS
    out = {}
    for name, (lo, hi) in bs.items():
        out[name] = (sec >= lo) & (sec <= hi)
    return out


BANDS = _band_masks()


def band_concordance(z, expected_band):
    """BC = mean z in expected band - mean z in the complementary band."""
    comp = "fast" if expected_band == "slow" else "slow"
    ze, zc = z[BANDS[expected_band]], z[BANDS[comp]]
    if not np.isfinite(ze).any() or not np.isfinite(zc).any():
        return np.nan
    return float(np.nanmean(ze) - np.nanmean(zc))


# ---------------- SID channels + trivial control channel ----------------
def sid_curves(X_list, names, refs, mask):
    """AUROC-vs-lag curves for SID mean/gain/tail on a worm set. refs: list of Reference."""
    curves = {r.name: {"mean": [], "gain": [], "tail": []} for r in refs}
    for l in LAGS:
        m = fit_distributional_connectome(X_list, names, lag=l, target_mode="next",
                                          ridge=1e-2, fps=C.FPS).matrices
        tail = np.abs(m["tail_hi"]) + np.abs(m["tail_lo"])
        for r in refs:
            curves[r.name]["mean"].append(mt.corr_score(m["mean"], r.adjacency, mask, "auroc"))
            curves[r.name]["gain"].append(mt.corr_score(m["gain"], r.adjacency, mask, "auroc"))
            curves[r.name]["tail"].append(mt.corr_score(tail, r.adjacency, mask, "auroc"))
    return {rn: {ch: np.array(v) for ch, v in d.items()} for rn, d in curves.items()}


def source_variance_curve(X_list, names, ref, mask):
    """Trivial control: rank edges by source-neuron activity variance (lag-flat, no model)."""
    V = np.concatenate([np.nan_to_num(x, nan=0.0) for x in X_list], 0).var(0)  # [N]
    Cmat = np.tile(V[None, :], (len(V), 1)); np.fill_diagonal(Cmat, 0.0)
    a = mt.corr_score(Cmat, ref.adjacency, mask, "auroc")
    return np.full(len(LAGS), a)     # identical at every lag by construction -> BC ~ 0


# ---------------- surrogates / global-mode ----------------
def circshift(X_list, rng):
    out = []
    for X in X_list:
        T, N = X.shape
        Xs = np.empty_like(X)
        for k in range(N):
            Xs[:, k] = np.roll(X[:, k], int(rng.integers(1, T)))
        out.append(Xs)
    return out


def remove_global_mode(X_list, n_pc=1):
    """Return residuals after projecting out the top-n_pc shared components per worm."""
    out = []
    for X in X_list:
        Xc = np.nan_to_num(X, nan=0.0)
        mu = Xc.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
        recon = (U[:, :n_pc] * S[:n_pc]) @ Vt[:n_pc]
        out.append((Xc - mu) - recon)
    return out
