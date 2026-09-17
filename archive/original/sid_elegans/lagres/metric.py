r"""RELATIVE lag-correspondence metric for directed effect matrices E(l) vs reference nets.

Convention (NEVER transposed here): both E and R are A[post, pre] = A[target, source];
A[j,i] != 0 means edge i -> j. The metric scores |E(l)|_offdiag against (R>0)_offdiag.
All comparisons are WITHIN (method, channel, reference) ACROSS the lag axis.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------- masks & one-lag scores ----------

def eval_mask(N, both_present=None, symmetric_ref=False):
    """Off-diagonal evaluation mask.
    both_present: bool[N] neurons that exist in the reference's native support; pairs
      touching an absent neuron are dropped (avoid fake negatives from zero-fill reindex).
    symmetric_ref: for gap/undirected refs, the mask stays full off-diagonal (score the
      symmetrized |E|+|E.T| upstream instead of masking)."""
    m = ~np.eye(N, dtype=bool)
    if both_present is not None:
        p = np.asarray(both_present, bool)
        m &= np.outer(p, p)
    return m

def _vec(E, R, mask):
    ys = np.abs(np.asarray(E))[mask]
    rw = np.asarray(R)[mask].astype(float)          # weighted ref (for rank-corr)
    yt = (rw > 0).astype(int)                        # binary ref (for AUROC/AUPRC)
    ok = np.isfinite(ys) & np.isfinite(rw)
    return ys[ok], yt[ok], rw[ok]

def corr_score(E, R, mask, metric="auroc"):
    ys, yt, rw = _vec(E, R, mask)
    if len(yt) == 0:
        return np.nan
    if metric in ("auroc", "auprc"):
        if yt.sum() == 0 or yt.sum() == len(yt):
            return np.nan
        return (roc_auc_score if metric == "auroc" else average_precision_score)(yt, ys)
    if metric == "spearman":                          # rank corr of |E| vs weighted R
        if ys.std() < 1e-12 or rw.std() < 1e-12:
            return np.nan
        return float(spearmanr(ys, rw).statistic)
    raise ValueError(metric)

def corr_curve(E_by_lag, R, mask, lags, metric="auroc"):
    """corr(l) over the ordered lag list `lags`. E_by_lag: {lag_frames: [N,N]}."""
    return np.array([corr_score(E_by_lag[l], R, mask, metric) for l in lags], float)

# ---------- RELATIVE within-method-across-lag normalizations ----------

def rel_minus_mean(c):        return c - np.nanmean(c)
def rel_zscore(c):
    s = np.nanstd(c); return (c - np.nanmean(c)) / s if s > 1e-12 else c * 0.0
def rel_ratio_max(c):
    m = np.nanmax(np.abs(c)); return c / m if m > 1e-12 else c * 0.0
def rel_excess(c, c_null):    return c - c_null      # per-lag null-anchored (RECOMMENDED stage 1)

# ---------- SHAPE statistics of a curve over lags (lags in FRAMES) ----------

def shape_stats(c, lags_frames, short_max_fr, long_min_fr):
    c = np.asarray(c, float); L = np.asarray(lags_frames, float)
    ok = np.isfinite(c); c, L = c[ok], L[ok]
    out = {}
    out["peak_lag_fr"] = float(L[int(np.nanargmax(c))]) if len(c) else np.nan
    # monotone trend
    if len(c) >= 3 and np.std(c) > 1e-12:
        out["spearman_trend"] = float(spearmanr(L, c).statistic)
        beta = np.polyfit(np.log2(L), c, 1)[0]        # slope of corr vs log2(lag)
        out["logslope"] = float(beta)                 # <0 fast/structural, >0 slow/modulatory
    else:
        out["spearman_trend"] = out["logslope"] = np.nan
    # lag specificity / peakedness on the nonneg-excess curve
    w = np.clip(c - np.nanmin(c), 0, None)
    if w.sum() > 0:
        p = w / w.sum()
        pp = p[p > 0]
        H = float(-np.sum(pp * np.log(pp)))
        out["negentropy_spec"] = float(np.log(len(p)) - H)       # high = lag-localized
        out["participation_ratio"] = float((w.sum() ** 2) / (np.sum(w ** 2)))  # low = localized
    else:
        out["negentropy_spec"] = out["participation_ratio"] = np.nan
    out["max_over_mean"] = float(np.nanmax(c) / np.nanmean(c)) if np.nanmean(c) != 0 else np.nan
    # fast-vs-slow contrast (positive = short lags match better)
    sm = c[L <= short_max_fr]; lm = c[L >= long_min_fr]
    out["fast_slow_contrast"] = float(np.nanmean(sm) - np.nanmean(lm)) if len(sm) and len(lm) else np.nan
    return out

# ---------- PER-LAG NULLS ----------

def null_label_perm(E, R, mask, metric="auroc", nperm=2000, rng=None):
    """Edge-label permutation: shuffle |E| offdiag entries. Preserves score marginal, kills
    E-R alignment. Optimistic (ignores E/R degree structure). Returns (obs, pval, null_mean)."""
    rng = rng or np.random.default_rng(0)
    ys, yt, _ = _vec(E, R, mask)
    if yt.sum() in (0, len(yt)):
        return np.nan, np.nan, np.nan
    scorer = roc_auc_score if metric == "auroc" else average_precision_score
    obs = scorer(yt, ys)
    nd = np.array([scorer(yt, rng.permutation(ys)) for _ in range(nperm)])
    return float(obs), float((np.sum(nd >= obs) + 1) / (nperm + 1)), float(nd.mean())

def null_node_perm(E, R, mask, metric="auroc", nperm=2000, rng=None):
    """Structure-preserving null: apply one neuron permutation pi to BOTH axes of E (keeps
    E's degree/hub structure, destroys alignment to R). Recommended, stronger than label perm."""
    rng = rng or np.random.default_rng(0)
    N = E.shape[0]
    obs = corr_score(E, R, mask, metric)
    if not np.isfinite(obs):
        return np.nan, np.nan, np.nan
    nd = []
    for _ in range(nperm):
        pi = rng.permutation(N)
        nd.append(corr_score(E[np.ix_(pi, pi)], R, mask, metric))
    nd = np.array([x for x in nd if np.isfinite(x)])
    return float(obs), float((np.sum(nd >= obs) + 1) / (len(nd) + 1)), float(nd.mean())

def control_matrix(s_src=None, s_tgt=None, N=None):
    """Trivial node-activity control C[j,i]. source-variance: C[j,i]=s_src[i];
    target-variance: C[j,i]=s_tgt[j]. Pass lag-l autocovariance vectors for a per-lag control."""
    if s_src is not None:
        N = len(s_src); return np.tile(np.asarray(s_src, float)[None, :], (N, 1))
    if s_tgt is not None:
        N = len(s_tgt); return np.tile(np.asarray(s_tgt, float)[:, None], (1, N))
    raise ValueError("give s_src or s_tgt")

def excess_over_trivial(E, R, mask, s_src_l, s_tgt_l, metric="auroc"):
    """corr(l) minus the best trivial (source/target activity) baseline AT THIS LAG."""
    c = corr_score(E, R, mask, metric)
    c_src = corr_score(control_matrix(s_src=s_src_l), R, mask, metric)
    c_tgt = corr_score(control_matrix(s_tgt=s_tgt_l), R, mask, metric)
    return c - np.nanmax([c_src, c_tgt]), c, c_src, c_tgt

# ---------- UNCERTAINTY: worm bootstrap / jackknife over curves & shape ----------

def worm_bootstrap_curve(fit_E_by_lag, W, R, mask, lags, metric="auroc",
                         nboot=1000, rng=None, ci=(2.5, 97.5)):
    """fit_E_by_lag(worm_idx_list)->{lag:E}. Resample worms w/ replacement, refit, rescore.
    Returns dict: mean curve, per-lag CI, and bootstrap array of shape stats via shape_stats."""
    rng = rng or np.random.default_rng(0)
    curves = []
    for _ in range(nboot):
        idx = rng.integers(0, W, W).tolist()
        E_by_lag = fit_E_by_lag(idx)
        curves.append(corr_curve(E_by_lag, R, mask, lags, metric))
    C = np.vstack(curves)
    return {"mean": np.nanmean(C, 0), "lo": np.nanpercentile(C, ci[0], 0),
            "hi": np.nanpercentile(C, ci[1], 0), "boot_curves": C}

def per_worm_curves(fit_E_by_lag, W, R, mask, lags, metric="auroc"):
    """corr_w(l) fitting each worm alone; mean +/- t-CI across worms (secondary, noisy)."""
    C = np.vstack([corr_curve(fit_E_by_lag([w]), R, mask, lags, metric) for w in range(W)])
    m = np.nanmean(C, 0); sd = np.nanstd(C, 0, ddof=1); n = np.sum(np.isfinite(C), 0)
    se = sd / np.sqrt(np.maximum(n, 1))
    return {"mean": m, "se": se, "worm_curves": C}

def split_half_shape_stability(fit_E_by_lag, W, R, mask, lags, metric="auroc",
                               n_splits=20, rng=None):
    """Split worms in half, score each curve, agreement = Pearson & Spearman of the two
    curves across lags + peak-lag match. Returns mean/sd over splits."""
    rng = rng or np.random.default_rng(0)
    pear, spear, pkmatch = [], [], []
    for _ in range(n_splits):
        perm = rng.permutation(W); h = W // 2
        cA = corr_curve(fit_E_by_lag(perm[:h].tolist()), R, mask, lags, metric)
        cB = corr_curve(fit_E_by_lag(perm[h:2 * h].tolist()), R, mask, lags, metric)
        ok = np.isfinite(cA) & np.isfinite(cB)
        if ok.sum() < 3 or cA[ok].std() < 1e-12 or cB[ok].std() < 1e-12:
            continue
        pear.append(np.corrcoef(cA[ok], cB[ok])[0, 1])
        spear.append(spearmanr(cA[ok], cB[ok]).statistic)
        pkmatch.append(float(np.nanargmax(cA) == np.nanargmax(cB)))
    f = lambda x: (float(np.mean(x)), float(np.std(x))) if x else (np.nan, np.nan)
    return {"pearson": f(pear), "spearman": f(spear), "peak_match": f(pkmatch)}

# ---------- CROSS-METHOD consistency ----------

def kendall_w(rank_matrix):
    """Kendall's W concordance of M methods ranking L lags. rank_matrix: [M, L] of ranks."""
    R = np.asarray(rank_matrix, float); M, L = R.shape
    Rj = R.sum(0); S = np.sum((Rj - Rj.mean()) ** 2)
    return float(12 * S / (M ** 2 * (L ** 3 - L)))

def cross_method_concordance(curves_by_method):
    """curves_by_method: {method: corr_curve}. Returns mean pairwise Pearson of normalized
    curves + Kendall's W over lag rankings. High W among mean-channel methods = shared
    structural-lag profile; a distributional (gain/tail) curve should be an OUTLIER here."""
    names = list(curves_by_method); Cs = [rel_zscore(curves_by_method[k]) for k in names]
    ok = np.all(np.isfinite(np.vstack(Cs)), 0)
    Cs = [c[ok] for c in Cs]
    ps = [np.corrcoef(Cs[a], Cs[b])[0, 1] for a in range(len(Cs)) for b in range(a + 1, len(Cs))]
    ranks = np.vstack([rankdata(c) for c in Cs])
    return {"methods": names, "mean_pairwise_pearson": float(np.mean(ps)) if ps else np.nan,
            "kendall_w": kendall_w(ranks)}
