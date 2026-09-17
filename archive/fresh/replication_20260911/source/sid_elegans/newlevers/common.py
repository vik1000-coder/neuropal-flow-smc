r"""Shared scaffolding for the new-levers investigation.

Everything the eight analysis scripts need in common: cached data loaders, the global
brain-state signal (as a *modulator* variable, not a nuisance to remove), the neuron->class
pooling operator, receptor-expressing node sets, variance-matched control selection, and a
tiny results-IO helper. All conventions match the repo: X_list is a list of ``[T, N]``
globally-standardized per-worm matrices, ``neuron_names`` is class-level, matrices are
``[post, pre]``, fps=4.0.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "output" / "newlevers"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# data (cached — loading pools the whole SBTG repo, so do it once per config)
# --------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def get_data(config: str = "6w_clean_raw"):
    """Return ``(X_list, neuron_names, fps)`` for a named config (cached).

    config = ``{6w_clean|28w}_{raw|deconv}``. 6w_clean = complete-case (no NaN, ~80 neurons,
    ~6 worms); 28w = available-case (NaN columns, ~84 neurons, all worms). deconv applies
    OASIS AR(1) calcium deconvolution before standardization.
    """
    from sid_elegans.combined_data import load_combined
    worms, signal = config.split("_")[0], config.split("_")[-1]
    complete = worms == "6w"
    X, names, fps = load_combined(complete_case=complete, signal=signal, verbose=False)
    return X, list(names), fps


# --------------------------------------------------------------------------------------
# the global brain-state signal — the MODULATOR variable (hook 2)
# --------------------------------------------------------------------------------------
def global_signal(X, method: str = "mean") -> np.ndarray:
    """Per-worm global brain-state timeseries ``g(t)`` of shape ``[T]``.

    ``method='mean'``: nanmean across present neurons each frame (matches the ACMMA
    NaN-aware global-signal regression). ``method='pc1'``: first principal-component score
    (matches ``metric.remove_global_mode``). Returned centered; NaN frames -> 0.
    """
    Xc = np.asarray(X, float)
    if method == "mean":
        g = np.nanmean(Xc, axis=1)
    elif method == "pc1":
        Z = np.nan_to_num(Xc, nan=0.0)
        Z = Z - Z.mean(axis=0, keepdims=True)
        # first left singular vector * singular value = PC1 score
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
        g = U[:, 0] * S[0]
        # DETERMINISTIC POLARITY: the SVD sign is arbitrary and independent per worm, so raw
        # PC1 'hi' frames would mean opposite brain states in different worms (mixing phases
        # when pooled). Align PC1 with the mean-activity pole so 'hi' is the SAME state everywhere.
        m = np.nan_to_num(Xc, nan=0.0).mean(axis=1)
        if np.std(m) > 1e-12 and np.corrcoef(g, m)[0, 1] < 0:
            g = -g
    else:
        raise ValueError("method must be 'mean' or 'pc1'")
    g = np.nan_to_num(g, nan=0.0)
    return g - g.mean()


def global_signals(X_list, method: str = "mean") -> list:
    return [global_signal(X, method) for X in X_list]


def phase_masks(X_list, method: str = "mean", q: float = 0.5):
    """Per-worm boolean masks splitting frames into HIGH vs LOW global-state phase.

    Returns ``(hi_list, lo_list)`` of per-worm ``[T]`` bool arrays, split at the per-worm
    ``q`` quantile of ``g(t)``. Used to compare gain-coupling across brain-state phases.
    """
    hi, lo = [], []
    for X in X_list:
        g = global_signal(X, method)
        thr = np.quantile(g, q)
        hi.append(g >= thr)
        lo.append(g < thr)
    return hi, lo


def subset_frames(X_list, masks):
    """Return X_list restricted to the frames selected by per-worm ``masks`` (keeps worms
    with >= 3*N surviving frames so the estimator has support)."""
    out = []
    for X, m in zip(X_list, masks):
        if m.sum() >= 3 * X.shape[1]:
            out.append(X[m])
    return out


# --------------------------------------------------------------------------------------
# neuron -> class pooling (hook 4)
# --------------------------------------------------------------------------------------
def coarse_class_labels(neuron_names) -> list:
    """Coarser anatomical class for each (already class-level) neuron: strip a trailing
    integer so numbered members merge (VA1..VA12 -> VA, DA1.. -> DA, IL1/IL2 kept). Neurons
    with no trailing digit are unchanged. This is the extra pooling headroom on top of the
    repo's L/R + D/V collapse."""
    labels = []
    for n in neuron_names:
        s = str(n).strip().upper()
        # strip trailing digits (motor/sensory numbered members) but keep 2-char stems
        j = len(s)
        while j > 2 and s[j - 1].isdigit():
            j -= 1
        labels.append(s[:j])
    return labels


@functools.lru_cache(maxsize=1)
def _ganglion_lookup():
    """neuron-class -> ganglion, from the wormneuroatlas ganglion map (network-free)."""
    import json
    from sid_elegans.newlevers.funatlas import atlas_h5_path, _to_class
    try:
        import wormneuroatlas as wa
        p = Path(wa.__file__).resolve().parent / "data" / "aconnectome_ids_ganglia.json"
        d = json.load(open(p))
    except Exception:
        return {}
    lut = {}
    for k, v in d.items():
        if not isinstance(v, list):
            continue
        # a "ganglion" key's values are NEURON ids (no 'ganglion'/'bulb'/'cord' substrings)
        if any(isinstance(x, str) and any(w in x.lower() for w in
               ("ganglion", "bulb", "cord", "ring")) for x in v):
            continue
        for nm in v:
            lut[str(nm).strip().upper()] = k
    return lut


def ganglion_labels(neuron_names) -> list:
    """Ganglion label per neuron (a genuinely coarser ~10-14-way grouping than the already
    class-level names). Neurons absent from the atlas map keep their own name (singleton)."""
    lut = _ganglion_lookup()
    support = {str(n).strip().upper() for n in neuron_names}
    from sid_elegans.newlevers.funatlas import _to_class
    # invert: atlas neuron -> class -> ganglion
    cls_gang = {}
    for atlas_nm, gang in lut.items():
        c = _to_class(atlas_nm, support)
        if c is not None:
            cls_gang.setdefault(c, gang)
    return [cls_gang.get(str(n).strip().upper(), f"_{n}") for n in neuron_names]


def pooling_operator(neuron_names, labels=None):
    """Return ``(G, classes)`` where ``G`` is a ``[C, N]`` row-normalized mean-pool operator
    (``G @ v`` averages neuron values within a class) and ``classes`` the ``C`` class names."""
    if labels is None:
        labels = coarse_class_labels(neuron_names)
    classes = sorted(set(labels))
    ci = {c: k for k, c in enumerate(classes)}
    C, N = len(classes), len(neuron_names)
    G = np.zeros((C, N))
    for i, lab in enumerate(labels):
        G[ci[lab], i] = 1.0
    G /= G.sum(axis=1, keepdims=True)
    return G, classes


def pool_matrix(M, G):
    """Pool a ``[N,N]`` [post,pre] matrix to ``[C,C]`` via ``G @ M @ G.T`` (block-mean over
    post-rows and pre-cols); zero the class diagonal (self-class)."""
    Mc = G @ np.asarray(M, float) @ G.T
    np.fill_diagonal(Mc, 0.0)
    return Mc


# --------------------------------------------------------------------------------------
# receptor-expressing node sets + variance-matched controls (hooks 2/3/5)
# --------------------------------------------------------------------------------------
def receptor_expressing(neuron_names, modulator: str = "neuropeptide"):
    """Length-N boolean: neurons that are on the TARGET (post/receptor) side of a modulator
    layer, i.e. rows ``i`` with any ``adjacency[i, :] > 0`` (adjacency is [post,pre]).

    ``modulator`` in {'neuropeptide','monoamine','dopamine','serotonin','tyramine',
    'octopamine'}.
    """
    from sid_elegans.ground_truth import (build_monoamine_layer, load_monoamine_edges,
                                           load_neuropeptide_layer)
    if modulator == "neuropeptide":
        A = load_neuropeptide_layer(neuron_names)
    elif modulator == "monoamine":
        A = build_monoamine_layer(load_monoamine_edges(), neuron_names, None)
    else:
        A = build_monoamine_layer(load_monoamine_edges(), neuron_names, modulator)
    return (np.asarray(A, float) > 0).any(axis=1)


def source_variance(X_list) -> np.ndarray:
    """Per-neuron variance over pooled (NaN->0) frames — the trivial 'is it just marginal
    variance' axis, reused for variance-matched control selection."""
    allX = np.concatenate([np.nan_to_num(np.asarray(X, float), nan=0.0) for X in X_list], axis=0)
    return allX.var(axis=0)


def variance_matched_controls(expressing, V, rng, n=None):
    """Pick a control set of NON-expressing neurons matched to the expressing set on the
    variance vector ``V`` (nearest-neighbour in log-variance, without replacement). Returns a
    length-N boolean mask over control neurons."""
    expressing = np.asarray(expressing, bool)
    idx_exp = np.where(expressing)[0]
    idx_pool = np.where(~expressing)[0]
    if n is None:
        n = len(idx_exp)
    n = min(n, len(idx_pool))
    lv = np.log(np.asarray(V, float) + 1e-12)
    chosen, used = [], set()
    order = rng.permutation(idx_exp)
    for e in order:
        if len(chosen) >= n:
            break
        cand = [p for p in idx_pool if p not in used]
        if not cand:
            break
        c = min(cand, key=lambda p: abs(lv[p] - lv[e]))
        chosen.append(c); used.add(c)
    mask = np.zeros(len(expressing), bool)
    mask[chosen] = True
    return mask


def matched_groups(expressing, feats, rng, caliper=None):
    """Equal-size, greedily nearest-matched (expressing, control) subsets on standardized
    features ``feats`` ([N] or [N,k]). This FIXES the failure mode of
    ``variance_matched_controls`` when the expressing set is LARGER than the non-expressing
    pool: there the old routine just took the whole pool (no matching). Here we always match
    the SMALLER group into the larger (subsampling the larger), so both returned masks have
    size = n_pairs and are genuinely matched on ``feats``.

    Returns ``(exp_mask, ctrl_mask, info)``. ``info`` = {n_pairs, resid_imbalance_std (per
    feature, in pool-sd units), pool_n, exp_n}. Pass a ``caliper`` (in standardized distance)
    to drop poorly-matched pairs.
    """
    expressing = np.asarray(expressing, bool)
    F = np.asarray(feats, float)
    if F.ndim == 1:
        F = F[:, None]
    finite = np.isfinite(F).all(axis=1)
    exp_idx = np.where(expressing & finite)[0]
    pool_idx = np.where((~expressing) & finite)[0]
    N = len(expressing)
    exp_mask = np.zeros(N, bool); ctrl_mask = np.zeros(N, bool)
    if len(exp_idx) == 0 or len(pool_idx) == 0:
        return exp_mask, ctrl_mask, {"n_pairs": 0, "resid_imbalance_std": None,
                                     "pool_n": int(len(pool_idx)), "exp_n": int(len(exp_idx))}
    ref = np.concatenate([exp_idx, pool_idx])
    mu = F[ref].mean(0); sd = F[ref].std(0) + 1e-9
    Fn = (F - mu) / sd
    small, large = (exp_idx, pool_idx) if len(exp_idx) <= len(pool_idx) else (pool_idx, exp_idx)
    used = set(); pairs = []
    for s in rng.permutation(small):
        best, bestd = None, np.inf
        for l in large:
            if l in used:
                continue
            d = float(((Fn[l] - Fn[s]) ** 2).sum())
            if d < bestd:
                bestd, best = d, l
        if best is None:
            break
        if caliper is not None and np.sqrt(bestd) > caliper:
            continue
        used.add(best); pairs.append((int(s), int(best)))
    for s, l in pairs:
        for idx in (s, l):
            (exp_mask if expressing[idx] else ctrl_mask)[idx] = True
    imb = ((Fn[exp_mask].mean(0) - Fn[ctrl_mask].mean(0)).tolist()
           if pairs else None)
    return exp_mask, ctrl_mask, {"n_pairs": len(pairs), "resid_imbalance_std": imb,
                                 "pool_n": int(len(pool_idx)), "exp_n": int(len(exp_idx))}


def regress_out(values, covariates):
    """Return residuals of ``values`` [N] after OLS on ``covariates`` [N] or [N,k] (+intercept),
    using only rows finite in both. Non-finite rows are returned as NaN. For the gating
    loading-confound control: regress state-gain on (log-variance, global-mode loading) and test
    the residual, which is robust when receptor status and loading are collinear."""
    v = np.asarray(values, float)
    X = np.asarray(covariates, float)
    if X.ndim == 1:
        X = X[:, None]
    good = np.isfinite(v) & np.isfinite(X).all(axis=1)
    out = np.full(len(v), np.nan)
    if good.sum() < X.shape[1] + 2:
        return out
    A = np.column_stack([np.ones(good.sum()), X[good]])
    beta, *_ = np.linalg.lstsq(A, v[good], rcond=None)
    out[good] = v[good] - A @ beta
    return out


# --------------------------------------------------------------------------------------
# results IO
# --------------------------------------------------------------------------------------
def save_json(name: str, obj: dict):
    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_san(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    p = OUT / (name if name.endswith(".json") else name + ".json")
    p.write_text(json.dumps(_san(obj), indent=2))
    return p


def env_int(key, default):
    return int(os.environ.get(key, default))


def env_float(key, default):
    return float(os.environ.get(key, default))
