r"""ACMMA -- Available-Case Multivariate Moment Assembly for the SID gain connectome.

Uses ALL worms with ZERO imputation. The closed-form quadratic-score fit consumes data only
through moments: theta = -(A+ridge I)^{-1} c, A = U'U/T, U = [psi, 2Y*psi], psi = [1, z_1..z_N]
(z = centered sources), c = [0_P, 2*mean(psi)]. Every entry of A is a real co-moment of
(1, z_i, Y); ACMMA estimates each entry AVAILABLE-CASE -- summed only over the worms/rows where
the specific neurons in that entry are simultaneously recorded, normalized by that entry's own
co-observation count. No missing (Y, x_i) cell is ever fabricated, so the decoupling artifact that
broke donor imputation cannot arise. Multivariate (keeps all N source columns) so it retains the
partial conditioning that controls the global-state confound (unlike the marginal pairwise.py).

De-scoped per adversarial review: HARD-MASK thin-triple entries (no shrink-to-class, no
imputation); PSD repair (shrinkage + eigen-clip) with a clipped-eigen-mass diagnostic; readout
byte-identical to estimator.py. Validate on the 6x80 complete-case corner (must reproduce it).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from sid_neuromod.utils.linalg import ridge_solve

CHANNELS = ("mean", "gain", "tail_hi", "tail_lo")


def _pool_target(X_list, j, lag):
    """Pooled (Y [R], Xsrc [R,N]) across worms recording j; NaN sources kept for masking."""
    Ys, Xs = [], []
    for X in X_list:
        T = X.shape[0]
        if T <= lag:
            continue
        y = X[lag:, j]
        xs = X[:T - lag, :]
        keep = np.isfinite(y)
        if keep.sum() == 0:
            continue
        Ys.append(y[keep]); Xs.append(xs[keep])
    if not Ys:
        return None, None
    return np.concatenate(Ys), np.concatenate(Xs, 0)


def _nearest_psd(A, shrink=0.05, floor=1e-6):
    """Ledoit-Wolf-style shrinkage + eigen-clip. Returns (A_psd, clipped_mass_fraction)."""
    P2 = A.shape[0]
    A = 0.5 * (A + A.T)
    A = (1 - shrink) * A + shrink * (np.trace(A) / P2) * np.eye(P2)
    w, V = np.linalg.eigh(A)
    clipped = float(np.sum(np.abs(w[w < floor]))) / (float(np.sum(np.abs(w))) + 1e-12)
    w = np.clip(w, floor, None)
    return (V * w) @ V.T, clipped


def fit_acmma_connectome(X_list, names, lag, ridge=1.0, min_triple=8, shrink=0.05,
                         eta2_min=1e-6, var_min=1e-6, q_hi=0.90, q_lo=0.10, sigma_frac=0.0):
    """Available-case multivariate SID connectome at one lag. Returns (matrices, diag).

    ``sigma_frac>0`` = denoising score matching at sigma = sigma_frac * sd(Y): in expectation the
    only change is the Y^2-block gains +4 sigma^2 * (source-covariance block) and the readout
    subtracts sigma^2 -- so we apply it deterministically (no noise draw), exactly.
    """
    N = len(names); P = N + 1
    mats = {c: np.zeros((N, N)) for c in CHANNELS}
    self_diag = {c: np.zeros(N) for c in CHANNELS}   # self-coupling (self-gain), pre-zeroing
    clipped_mass = np.full(N, np.nan)
    masked_frac = np.full(N, np.nan)

    for j in range(N):
        Y, Xsrc = _pool_target(X_list, j, lag)
        if Y is None or len(Y) < 3 * N:
            continue
        M = np.isfinite(Xsrc)                          # [R,N] present
        # available-case per-source centering
        with np.errstate(invalid="ignore"):
            src_mean = np.where(M.any(0), np.nanmean(np.where(M, Xsrc, np.nan), 0), 0.0)
        Z = np.where(M, Xsrc - src_mean, 0.0)          # centered, 0 where absent (auto-masks)
        Mf = M.astype(float)
        Y2 = Y ** 2

        # co-observation counts and available-case sums
        Nik = Mf.T @ Mf                                 # [N,N] rows both present
        cnt = Mf.sum(0)                                 # [N] rows present
        n_all = float(len(Y))
        S0 = Z.T @ Z; S1 = (Z * Y[:, None]).T @ Z; S2 = (Z * Y2[:, None]).T @ Z
        sY = float(Y.sum()); sY2 = float(Y2.sum())
        sYz = (Y[:, None] * Z).sum(0); sY2z = (Y2[:, None] * Z).sum(0)

        def norm2(S, Nc):                               # entrywise divide by own count
            out = np.zeros_like(S); ok = Nc > 0
            out[ok] = S[ok] / Nc[ok]; return out
        s0 = norm2(S0, Nik); s1 = norm2(S1, Nik); s2 = norm2(S2, Nik)
        ci = np.where(cnt > 0, cnt, 1.0)
        yz = sYz / ci; y2z = sY2z / ci

        # HARD MASK: gain-bearing (S1,S2) entries with thin triple support -> 0 (no fabrication).
        # triple support proxy: for a target j, entry (i,k) needs j,i,k co-observed; Nik already
        # requires i,k present and the pooled rows require j present, so Nik IS the triple count.
        thin = Nik < min_triple
        s1[thin] = 0.0; s2[thin] = 0.0
        masked_frac[j] = float(thin.mean())

        # assemble 2P x 2P A (index 0 = intercept; 1..N = centered sources)
        A11 = np.zeros((P, P)); A12 = np.zeros((P, P)); A22 = np.zeros((P, P))
        A11[0, 0] = 1.0
        A11[1:, 1:] = s0
        A12[0, 0] = 2 * sY / n_all
        A12[0, 1:] = 2 * yz; A12[1:, 0] = 2 * yz
        A12[1:, 1:] = 2 * s1
        A22[0, 0] = 4 * sY2 / n_all
        A22[0, 1:] = 4 * y2z; A22[1:, 0] = 4 * y2z
        A22[1:, 1:] = 4 * s2
        # denoising score matching (sigma>0): Y^2-block += 4 sigma^2 * source-cov block
        if sigma_frac > 0:
            vy = max(sY2 / n_all - (sY / n_all) ** 2, 0.0)
            sigma = float(sigma_frac) * np.sqrt(vy)
            A22 = A22 + 4.0 * sigma ** 2 * A11
        else:
            sigma = 0.0
        A = np.block([[A11, A12], [A12.T, A22]])
        A_psd, clipped_mass[j] = _nearest_psd(A, shrink=shrink)

        c = np.zeros(2 * P); c[P] = 2.0                 # c = [0_P, 2*mean(psi)] = [0.., 2, 0..]
        theta = -ridge_solve(A_psd, c, ridge)
        t1, t2 = theta[:P], theta[P:]

        eta1 = t1[0]; eta2 = min(t2[0], -eta2_min)
        v_eff = -1.0 / (2.0 * eta2); v = max(v_eff - sigma ** 2, var_min); mu = eta1 * v_eff
        d_mean = v_eff * t1 + 2.0 * eta1 * v_eff ** 2 * t2
        d_var = 2.0 * v_eff ** 2 * t2
        d_gain = d_var / v
        sv = np.sqrt(v)
        qh = float(np.quantile(Y, q_hi)); ql = float(np.quantile(Y, q_lo))
        a_hi = (qh - mu) / sv; a_lo = (ql - mu) / sv
        d_thi = norm.pdf(a_hi) * (d_mean / sv + (qh - mu) / (2 * v ** 1.5) * d_var)
        d_tlo = norm.pdf(a_lo) * (-d_mean / sv - (ql - mu) / (2 * v ** 1.5) * d_var)
        for cn, val in (("mean", d_mean), ("gain", d_gain), ("tail_hi", d_thi), ("tail_lo", d_tlo)):
            row = val[1:]; row = np.where(np.isfinite(row), row, 0.0)
            mats[cn][j, :] = row
            sv_j = val[1 + j]
            self_diag[cn][j] = sv_j if np.isfinite(sv_j) else 0.0   # capture before zeroing
        for cn in CHANNELS:
            mats[cn][j, j] = 0.0

    diag = {"clipped_eigen_mass_median": float(np.nanmedian(clipped_mass)),
            "clipped_eigen_mass_max": float(np.nanmax(clipped_mass)),
            "masked_gain_frac_median": float(np.nanmedian(masked_frac)),
            "self_diag": self_diag}
    return mats, diag
