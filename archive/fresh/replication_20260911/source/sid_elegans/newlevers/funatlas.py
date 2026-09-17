r"""Network-free loader of the Randi/Leifer functional 'signal-propagation' atlas.

Reads ``funatlas.h5`` (bundled in the installed ``wormneuroatlas`` package) DIRECTLY, with
h5py — it does NOT construct ``wormneuroatlas.NeuroAtlas()`` (whose ``__init__`` makes a
WormBase DB-version HTTP request that fails offline / returns bad JSON, and which the repo's
existing ``ground_truth.load_leifer`` silently swallows to ``None``). This makes the
functional target usable for an unattended overnight run.

Orientation (verified against the package source):
    ``dFF[i, j]`` = mean delta-F/F **response in neuron i** when **neuron j is stimulated**
    = effect of source j on target i = ``[post=i, pre=j]``.
This matches the repo's universal ``[post, pre] = [target, source]`` convention, so NO
transpose is applied before scoring against the repo's estimator matrices.

Beyond the static connectome the atlas ships per-edge exponential-convolution **kinetics**
(``kernels``); we evaluate each kernel and summarise it as a characteristic timescale
(amplitude-weighted centroid of ``|h(t)|``). That per-edge timescale is what lets us test
whether the SID gain-channel's preferred lag tracks a *measured causal* timescale — the
direct lag test the anatomical/receptor targets could never provide.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------
# locate the bundled atlas + optional repo name-collapse
# --------------------------------------------------------------------------------------
def atlas_h5_path() -> Path | None:
    """Path to the bundled ``funatlas.h5`` inside the installed wormneuroatlas package."""
    try:
        import wormneuroatlas as wa
        p = Path(wa.__file__).resolve().parent / "data" / "funatlas.h5"
        return p if p.exists() else None
    except Exception:
        return None


def _norm(name: str) -> str:
    return str(name).strip().upper()


@functools.lru_cache(maxsize=1)
def _repo_collapse():
    """The repo's own D/V-subtype collapse (so atlas names map to the SAME class labels the
    calcium loader uses). Falls back to identity if the SBTG package is not importable."""
    try:
        from pipeline.utils.align import collapse_dv_subtypes  # needs PYTHONPATH=.:SBTG
        return collapse_dv_subtypes
    except Exception:
        return lambda n: n


def _to_class(atlas_name: str, support: set[str]) -> str | None:
    """Map a full atlas neuron id (e.g. 'ADAL', 'AWCON') to the repo's class-level support.

    Strategy: uppercase; the atlas encodes the two AWC states as AWCON/AWCOFF -> AWC; apply
    the repo D/V collapse; then accept an exact hit, else drop a trailing L/R, else give up.
    """
    n = _norm(atlas_name)
    if n.startswith("AWC"):
        n = "AWC"
    n = _norm(_repo_collapse()(n))
    if n in support:
        return n
    if len(n) > 1 and n[-1] in ("L", "R") and n[:-1] in support:
        return n[:-1]
    return None


# --------------------------------------------------------------------------------------
# kinetics: characteristic timescale of a per-edge exponential-convolution kernel
# --------------------------------------------------------------------------------------
def _kernel_timescale(flat, t_max: float = 30.0, dt: float = 0.05) -> float:
    """Amplitude-weighted centroid (seconds) of ``|h(t)|`` for one atlas kernel.

    ``flat`` is the raw h5 kernel entry: a 1-D array that reshapes to rows
    ``[g, factor, power_t, branch]`` (kernels_keys = 'g,factor,power_t,branch'). We evaluate
    ``h(t) = sum factor * t**power_t * exp(-g t)`` on a grid and return
    ``int t|h| / int |h|``. Returns NaN if the kernel is empty/degenerate.
    """
    arr = np.asarray(flat, dtype=float).ravel()
    if arr.size == 0 or arr.size % 4 != 0:
        return np.nan
    rows = arr.reshape(-1, 4)
    g, fac, pt = rows[:, 0], rows[:, 1], rows[:, 2]
    keep = np.isfinite(g) & np.isfinite(fac) & (np.abs(fac) > 0)
    if not keep.any():
        return np.nan
    g, fac, pt = g[keep], fac[keep], pt[keep]
    t = np.arange(dt, t_max + dt, dt)
    # h(t) = sum_k fac_k * t^pt_k * exp(-g_k t); build term-by-term (delicate near-cancelling
    # amplitudes are by design — float64 keeps enough significant digits for a centroid).
    h = np.zeros_like(t)
    for gk, ck, pk in zip(g, fac, pt):
        if gk < 0:                      # a growing exponential is nonphysical here; skip
            continue
        mult = 1.0 if pk == 0 else np.power(t, pk)
        h += ck * mult * np.exp(-gk * t)
    ah = np.abs(h)
    tot = ah.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.nan
    return float((t * ah).sum() / tot)


# --------------------------------------------------------------------------------------
# main loader
# --------------------------------------------------------------------------------------
def load_funatlas(neuron_names, strain: str = "wt", alpha: float = 0.05,
                  with_kinetics: bool = True, verbose: bool = True) -> dict | None:
    """Return the functional atlas aligned to class-level ``neuron_names`` ([post, pre]).

    Keys (all ``[N, N]`` float64 unless noted, N = len(neuron_names)):
      ``dFF``       mean signal-propagation amplitude (graded, signed), NaN where unmeasured.
      ``absdFF``    ``|dFF|`` with NaN->0 (ready graded target for AUROC).
      ``q``         min q-value over merged members (NaN where unmeasured).
      ``positive``  binary (q < alpha).
      ``occ``       summed occurrence (trial count) per class pair.
      ``timescale`` characteristic response timescale in **seconds** (NaN if no kinetics).
      ``support``   ``[N]`` bool: neuron measured in the atlas (as source or target).
      ``both_present`` ``[N,N]`` bool: both endpoints measured (off-diagonal) — the eval mask.
      ``eval_mask`` ``both_present`` AND (positive OR confirmed-negative) — fake-negative-safe.
    Returns ``None`` if the atlas file cannot be found/read.
    """
    import h5py

    path = atlas_h5_path()
    if path is None:
        return None
    try:
        f = h5py.File(str(path), "r")
    except Exception:
        return None

    a_ids = [b.decode("utf-8") if isinstance(b, (bytes, np.bytes_)) else str(b)
             for b in f["neuron_ids"][:]]
    g = f[strain]
    dff_a = np.asarray(g["dFF"][:], float)          # [A,A] [post,pre]
    q_a = np.asarray(g["q"][:], float)
    qeq_a = np.asarray(g["q_eq"][:], float) if "q_eq" in g else np.full_like(q_a, np.nan)
    occ_a = np.asarray(g["occ1"][:], float)
    ker = g["kernels"] if (with_kinetics and "kernels" in g) else None

    support_set = {_norm(n) for n in neuron_names}
    N = len(neuron_names)
    name_idx = {_norm(n): i for i, n in enumerate(neuron_names)}

    # atlas index -> repo class index (or -1)
    cls = np.full(len(a_ids), -1, dtype=int)
    for ai, an in enumerate(a_ids):
        c = _to_class(an, support_set)
        if c is not None:
            cls[ai] = name_idx[c]

    # accumulate member atlas pairs into class-pair cells
    dFF = np.full((N, N), np.nan)
    q = np.full((N, N), np.nan)
    occ = np.zeros((N, N))
    ts_num = np.zeros((N, N)); ts_den = np.zeros((N, N))
    pos = np.zeros((N, N), bool)
    conf_neg = np.zeros((N, N), bool)
    measured = np.zeros((N, N), bool)

    dff_sum = np.zeros((N, N)); dff_cnt = np.zeros((N, N))
    mapped = np.where(cls >= 0)[0]
    for pi in mapped:              # post (row / target)
        cp = cls[pi]
        for pj in mapped:         # pre (col / source)
            cq = cls[pj]
            if occ_a[pi, pj] <= 0 and not np.isfinite(dff_a[pi, pj]):
                continue
            measured[cp, cq] = True
            occ[cp, cq] += max(occ_a[pi, pj], 0.0)
            v = dff_a[pi, pj]
            if np.isfinite(v):
                dff_sum[cp, cq] += v; dff_cnt[cp, cq] += 1
            qv = q_a[pi, pj]
            if np.isfinite(qv):
                q[cp, cq] = qv if not np.isfinite(q[cp, cq]) else min(q[cp, cq], qv)
                if qv < alpha:
                    pos[cp, cq] = True
            if np.isfinite(qeq_a[pi, pj]) and qeq_a[pi, pj] < alpha:
                conf_neg[cp, cq] = True
            if ker is not None and np.isfinite(v):
                tau = _kernel_timescale(ker[pi, pj])
                if np.isfinite(tau):
                    w = abs(v) * max(occ_a[pi, pj], 1.0)
                    ts_num[cp, cq] += w * tau; ts_den[cp, cq] += w

    good = dff_cnt > 0
    dFF[good] = dff_sum[good] / dff_cnt[good]
    timescale = np.full((N, N), np.nan)
    tgood = ts_den > 0
    timescale[tgood] = ts_num[tgood] / ts_den[tgood]

    conf_neg &= ~pos                          # a positive overrides a confirmed-negative
    diag = np.eye(N, dtype=bool)
    support = (measured.any(axis=1) | measured.any(axis=0))
    both_present = np.outer(support, support) & ~diag
    eval_mask = both_present & (pos | conf_neg)

    absdFF = np.nan_to_num(np.abs(dFF), nan=0.0)
    positive = pos.astype(float)
    positive[diag] = 0.0
    graded = np.nan_to_num(np.where(pos, np.abs(dFF), 0.0), nan=0.0)   # signif-gated |dFF|
    graded[diag] = 0.0

    if verbose:
        n_sup = int(support.sum())
        print(f"[funatlas] {n_sup}/{N} neurons in atlas support; "
              f"{int((positive[both_present] > 0).sum())} positive edges (q<{alpha}) on support; "
              f"{int(eval_mask.sum())} confirmed edges in eval mask; "
              f"kinetics on {int(np.isfinite(timescale).sum())} edges "
              f"(median tau={np.nanmedian(timescale):.2f}s)")
    f.close()
    return {
        "dFF": dFF, "absdFF": absdFF, "graded": graded, "q": q, "positive": positive, "occ": occ,
        "timescale": timescale, "support": support, "both_present": both_present,
        "eval_mask": eval_mask, "neuron_names": list(neuron_names), "alpha": alpha,
    }


def functional_reference(neuron_names, alpha: float = 0.05):
    """Return ``(Reference, atlas_dict)`` for the functional connectome, or ``(None, None)``.

    The Reference plugs straight into the biolag scoring machinery (build_registry-compatible
    dataclass, ``[post, pre]`` binary adjacency); the atlas dict carries the eval mask + the
    per-edge kinetics for the timescale test.
    """
    from sid_elegans.biolag import config as C
    from sid_elegans.biolag.references import Reference
    fa = load_funatlas(neuron_names, alpha=alpha, with_kinetics=True, verbose=False)
    if fa is None:
        return None, None
    A = fa["positive"]
    N = len(neuron_names)
    ne = int((A[~np.eye(N, dtype=bool)] > 0).sum())
    ref = Reference("functional:leifer", "functional", "class", np.asarray(A, float), ne,
                    C.expected_band_of("functional"), C.FAMILIES["functional"]["expected_channel"])
    return ref, fa
