r"""LEVER #6 — FROZEN NEURAL FEATURES + CLOSED-FORM READOUT  (the direct SBTG borrow).

Question
--------
Does *freezing* the two-head trunk and putting the exact closed-form
quadratic-score head (the same estimator used in ``fit_distributional_connectome``)
on top of the frozen features beat

  (a) the raw closed-form estimator in NEURON space (no trunk), and
  (b) the fully end-to-end two-head readout (trunk + heads trained jointly)?

The hypothesis is the classic SBTG (self-supervised backbone -> tiny closed-form
head) borrow: a nonlinear feature space learned once, then a convex readout that
cannot overfit the transition signal, should give *more reproducible* couplings
than end-to-end autograd through a nonlinear head, while still beating a purely
linear (raw) estimator because the features are nonlinear.

Design (faithful to the spec)
-----------------------------
Config: ``6w_clean_deconv`` ONLY (data-limited). device="cpu".

 1. ``fit_two_head(X, names, source_tau_s=TAU_S, horizon=H, hidden=HID,
    epochs=EPOCHS, device="cpu", return_model=True)`` -> res. Freeze
    ``res.model.trunk`` (``requires_grad_(False)``).
 2. Rebuild the pooled standardized history ``H_std = (H - res.h_mu)/res.h_sd``
    (reusing ``twohead._slow_filter`` / ``twohead._pool`` with
    ``res.meta['source_tau_s']`` / ``['horizon']``) and the raw futures ``Xf``.
    Embed ``Z = res.model.trunk(H_std)``  ->  [M, hidden].
 3. For each target j: closed-form ``QuadraticScoreMatcher`` on
    ``Psi = [1, Z - mean(Z)]`` with ``Y = Xf[:, j]``, then read the feature-space
    mean/gain at the mean history via the SAME ``estimator._center_readouts`` used
    by the raw estimator (feature col 0 = intercept, cols 1.. = Z features).
 4. Map the feature-space readout back to NEURON space through the trunk Jacobian
    ``J = dZ/du`` (``torch.autograd.functional.jacobian`` on ``model.trunk`` at the
    mean standardized input ``u_bar``, shape [hidden, N]):

        neuron_read[j, :] = (feat_read_j[1:] @ J) / res.h_sd

    (undoing the history standardization ``du/dh = 1/h_sd``; a first-order
    linearization of the frozen trunk at the mean history). Build [N,N] mean+gain
    matrices in ``[post, pre]`` convention, diagonal zeroed.

Comparison
----------
For each method in {frozen, raw, e2e} and channel in {mean, gain}: AUROC of
``|offdiag(M)|`` vs every ``build_registry`` reference (structural + monoamine +
neuropeptide). Plus split-half worm stability of the gain channel for each method
(``stability.split_half_stability``). The pre-registered confirmatory target is
``neuropeptide:all`` on the gain channel.

All heavy counts come from env vars; heavy work only under ``__main__``; results
via ``common.save_json("frozen", ...)``. The whole neural block is wrapped in
try/except so a partial JSON (at least the raw method) is always emitted.
"""
from __future__ import annotations

import os
import time
import traceback

import numpy as np

# ----- import-time must be cheap: only light imports here -----
from sid_elegans.newlevers import common

# --------------------------------------------------------------------------------------
# env-tunable knobs (HARD RULE 1/2/5): every heavy count is read from the environment.
# --------------------------------------------------------------------------------------
SMOKE = os.environ.get("NEWLEVERS_SMOKE", "0") == "1"
SEED = common.env_int("NL_SEED", 0)
CONFIG = os.environ.get("NL_CONFIG", "6w_clean_deconv")  # this lever is 6w_clean_deconv only
THROTTLE = common.env_float("NL_THROTTLE", 0.3)

# neural-net size / training (device is ALWAYS cpu — MPS autograd is flaky here)
TAU_S = common.env_float("NL_TAU_S", 5.0)         # slow-filter source timescale (seconds)
HORIZON = common.env_int("NL_HORIZON", 1)          # prediction horizon (frames)
RIDGE = common.env_float("NL_RIDGE", 1e-2)         # closed-form head ridge (matches estimator)

if SMOKE:
    HIDDEN = common.env_int("NL_HIDDEN", 32)
    EPOCHS = min(common.env_int("NL_EPOCHS", 20), 30)
    STAB_EPOCHS = min(common.env_int("NL_STAB_EPOCHS", 10), 30)
    NSPLITS = common.env_int("NL_NSPLITS", 2)
else:
    HIDDEN = common.env_int("NL_HIDDEN", 128)
    EPOCHS = common.env_int("NL_EPOCHS", 150)
    STAB_EPOCHS = common.env_int("NL_STAB_EPOCHS", 60)
    NSPLITS = common.env_int("NL_NSPLITS", 5)

# whether to run the (expensive) neural split-half stability (raw stability always runs)
STAB_NEURAL = os.environ.get("NL_STAB_NEURAL", "1") == "1"


# --------------------------------------------------------------------------------------
# frozen-feature readout  (Steps 2-4)
# --------------------------------------------------------------------------------------
def frozen_matrices(res, X_list, names, ridge=RIDGE, seed=SEED):
    """Closed-form mean/gain connectome on the FROZEN trunk features.

    ``res`` is a fitted ``TwoHeadResult`` (return_model=True). ``X_list`` is the worm
    set the trunk was trained on (so ``res.h_mu``/``res.h_sd`` are the right standardizer).
    Returns ``{'mean': [N,N], 'gain': [N,N]}`` in ``[post, pre]``, diagonal zeroed.
    """
    import torch
    from sid_elegans.estimator import _center_readouts
    from sid_elegans.twohead import _pool, _slow_filter
    from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher

    N = len(names)
    trunk = res.model.trunk
    for p in trunk.parameters():
        p.requires_grad_(False)
    trunk.eval()

    tau_s = res.meta["source_tau_s"]
    horizon = res.meta["horizon"]
    fps = res.meta["fps"]

    # Step 2: rebuild pooled standardized history + raw futures, embed through trunk.
    H_list = _slow_filter(X_list, tau_s, fps) if tau_s else \
        [np.nan_to_num(x, nan=0.0) for x in X_list]
    Hn, Xf = _pool(X_list, H_list, horizon)
    H_std = (Hn - res.h_mu) / res.h_sd
    with torch.no_grad():
        Z = trunk(torch.tensor(H_std, dtype=torch.float32)).detach().numpy()  # [M, hidden]
    Z_mu = Z.mean(axis=0, keepdims=True)
    Zc = Z - Z_mu   # center features (matches estimator centering of sources)

    # Step 4 prep: trunk Jacobian dZ/du at the mean standardized input -> [hidden, N].
    u_bar = torch.tensor(H_std.mean(axis=0), dtype=torch.float32)
    J = torch.autograd.functional.jacobian(trunk, u_bar).detach().numpy()  # [hidden, N]

    mean_cpl = np.zeros((N, N))
    gain_cpl = np.zeros((N, N))
    ones = np.ones((len(Xf), 1))
    for j in range(N):
        try:
            Y = Xf[:, j]
            if not np.isfinite(Y).any() or np.std(Y) < 1e-9:
                continue
            Psi = np.concatenate([ones, Zc], axis=1)              # [M, 1+hidden]
            m = QuadraticScoreMatcher(sigma=0.0, ridge=ridge, seed=seed)
            r = m.fit(Y, Psi)
            q_hi = float(np.nanquantile(Y, 0.90))
            q_lo = float(np.nanquantile(Y, 0.10))
            rd = _center_readouts(r, q_hi, q_lo)                  # feat-space readouts
            # cols 1.. are the Z features; chain through the Jacobian, undo h standardization
            mean_cpl[j, :] = (rd["mean"][1:] @ J) / res.h_sd
            gain_cpl[j, :] = (rd["gain"][1:] @ J) / res.h_sd
        except Exception:
            continue  # leave this target's row at zero on a degenerate solve
    np.fill_diagonal(mean_cpl, 0.0)
    np.fill_diagonal(gain_cpl, 0.0)
    return {"mean": mean_cpl, "gain": gain_cpl}


# --------------------------------------------------------------------------------------
# raw closed-form estimator, matched to the two-head's (tau, horizon)
# --------------------------------------------------------------------------------------
def raw_matrices(X_list, names, ridge=RIDGE, seed=SEED, heldout_frac=0.2):
    """Raw NEURON-space closed-form connectome comparable to the frozen/e2e ones:
    lag = horizon, sources = the same slow-filter (tau_s -> frames).

    ``heldout_frac`` controls the closed-form fit's train/held-out split. For the
    apples-to-apples point-estimate comparison against frozen/e2e (which both fit
    the readout on 100% of frames) pass ``heldout_frac=0.0`` so the raw baseline is
    also fit on the full data. The split-half stability path keeps the default 0.2
    (it already achieves fairness by refitting on worm halves)."""
    from sid_elegans.estimator import fit_distributional_connectome
    source_tau_frames = TAU_S * res_fps() if TAU_S else None
    r = fit_distributional_connectome(
        X_list, names, lag=HORIZON, target_mode="next", ridge=ridge,
        source_tau=source_tau_frames, fps=res_fps(), seed=seed,
        heldout_frac=heldout_frac)
    return {"mean": r.matrices["mean"], "gain": r.matrices["gain"]}


def res_fps():
    return 4.0  # fps is fixed for this data (see conventions); avoids threading it everywhere


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------
def auroc_table(mats, refs):
    """{channel: {ref_name: auroc}} for the mean+gain matrices vs every reference.

    Uses the shared harness AUROC (full off-diagonal, |coupling| vs ref>0)."""
    from sid_elegans.newlevers.harness import auroc_at_lag
    out = {}
    for ch in ("mean", "gain"):
        M = mats[ch]
        row = {}
        for r in refs:
            try:
                a = auroc_at_lag(M, r.adjacency, mask=None)
            except Exception:
                a = float("nan")
            row[r.name] = None if (a is None or not np.isfinite(a)) else float(a)
        out[ch] = row
    return out


def _mean_over(rowdict, names):
    vals = [rowdict[n] for n in names if rowdict.get(n) is not None and np.isfinite(rowdict[n])]
    return float(np.mean(vals)) if vals else None


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    import warnings

    import torch

    # The closed-form estimator clamps invalid eta2 at prediction time and warns; this is
    # documented, expected behaviour on this data, so silence it to keep the overnight log
    # readable (it does NOT change any numbers — clamping happens regardless).
    warnings.filterwarnings("ignore", message=r".*invalid eta2.*", category=RuntimeWarning)

    # Honor (do not raise) the driver's thread caps for torch too.
    omp = os.environ.get("OMP_NUM_THREADS")
    if omp:
        try:
            torch.set_num_threads(min(torch.get_num_threads(), int(omp)))
        except Exception:
            pass

    t_start = time.time()
    rng = np.random.default_rng(SEED)

    if CONFIG != "6w_clean_deconv":
        print(f"[frozen] WARNING: this lever is designed for 6w_clean_deconv; "
              f"NL_CONFIG={CONFIG} was requested — proceeding but results may be off-spec.")

    X, names, fps = common.get_data(CONFIG)
    N = len(names)
    print(f"[frozen] config={CONFIG}  worms={len(X)}  neurons={N}  fps={fps}  "
          f"smoke={SMOKE}  hidden={HIDDEN}  epochs={EPOCHS}  stab_epochs={STAB_EPOCHS} "
          f"nsplits={NSPLITS} tau_s={TAU_S} horizon={HORIZON} ridge={RIDGE}")

    from sid_elegans.biolag.references import build_registry
    from sid_elegans.stability import split_half_stability
    from sid_elegans.twohead import fit_two_head

    refs = build_registry(names, verbose=False)
    ref_names = [r.name for r in refs]
    modulatory = [r.name for r in refs if r.family in ("monoamine", "neuropeptide")]
    structural = [r.name for r in refs if r.family in ("gap", "chemical")]
    CONF = "neuropeptide:all"  # pre-registered confirmatory (gain channel)

    out = {
        "lever": "frozen (idea #6): frozen two-head trunk + closed-form quadratic-score head",
        "config": CONFIG, "n_worms": len(X), "n_neurons": N, "fps": fps,
        "params": {"source_tau_s": TAU_S, "horizon": HORIZON, "hidden": HIDDEN,
                   "epochs": EPOCHS, "ridge": RIDGE, "stab_epochs": STAB_EPOCHS,
                   "n_splits": NSPLITS, "seed": SEED, "smoke": SMOKE,
                   "device": "cpu", "stab_neural": STAB_NEURAL},
        "confirmatory_target": {"reference": CONF, "channel": "gain"},
        "registry": [{"name": r.name, "family": r.family, "level": r.level,
                      "n_edges": r.n_edges} for r in refs],
        "auroc": {}, "auroc_summary": {}, "stability": {}, "neural_ok": False,
        "error": None,
    }

    # ---- (a) RAW closed-form: always compute (cheap, never fails the neural way) ----
    # Point-estimate baseline is fit on the FULL data (heldout_frac=0.0) so it is on
    # equal footing with frozen/e2e (both read out on 100% of frames) — reviewer fix.
    try:
        raw = raw_matrices(X, names, heldout_frac=0.0)
        out["auroc"]["raw"] = auroc_table(raw, refs)
        print(f"[frozen] raw gain AUROC[{CONF}] = {out['auroc']['raw']['gain'].get(CONF)}")
    except Exception as e:
        out["auroc"]["raw"] = None
        out["error_raw"] = f"{e}\n{traceback.format_exc()}"
        print(f"[frozen] RAW failed: {e}")

    # ---- (b) + (c) neural: two-head fit -> frozen readout + e2e readout ----
    frozen = e2e = None
    try:
        t0 = time.time()
        res = fit_two_head(X, names, source_tau_s=TAU_S, horizon=HORIZON, hidden=HIDDEN,
                           epochs=EPOCHS, device="cpu", return_model=True, seed=SEED,
                           verbose=False)
        print(f"[frozen] two-head fit: {time.time()-t0:.1f}s  "
              f"(cond={res.history['cond'][-1]:.4f} marg={res.history['marg'][-1]:.4f})")

        e2e = {"mean": res.matrices["mean"], "gain": res.matrices["gain"]}
        frozen = frozen_matrices(res, X, names)

        out["auroc"]["e2e"] = auroc_table(e2e, refs)
        out["auroc"]["frozen"] = auroc_table(frozen, refs)
        out["neural_ok"] = True
        print(f"[frozen] frozen gain AUROC[{CONF}] = {out['auroc']['frozen']['gain'].get(CONF)}")
        print(f"[frozen]    e2e gain AUROC[{CONF}] = {out['auroc']['e2e']['gain'].get(CONF)}")
    except Exception as e:
        out["error"] = f"{e}\n{traceback.format_exc()}"
        print(f"[frozen] NEURAL block failed (emitting partial JSON): {e}")

    # ---- summaries (mean AUROC over reference groups, gain channel) ----
    for method in ("raw", "frozen", "e2e"):
        tab = out["auroc"].get(method)
        if not tab:
            continue
        out["auroc_summary"][method] = {
            "gain_confirmatory": tab["gain"].get(CONF),
            "gain_mean_all_refs": _mean_over(tab["gain"], ref_names),
            "gain_mean_modulatory": _mean_over(tab["gain"], modulatory),
            "mean_mean_structural": _mean_over(tab["mean"], structural),
        }

    # ---- split-half worm stability of the GAIN channel (primary distributional channel) ----
    # raw stability: always (cheap).
    try:
        def raw_gain_fit(Xs):
            return raw_matrices(Xs, names)["gain"]
        out["stability"]["raw_gain"] = split_half_stability(
            raw_gain_fit, X, n_splits=NSPLITS, seed=SEED)
        print(f"[frozen] stability raw_gain spearman="
              f"{out['stability']['raw_gain']['spearman_mean']:.3f}")
    except Exception as e:
        out["stability"]["raw_gain"] = {"error": str(e)}

    # neural stability: refit the trunk on each half (reduced STAB_EPOCHS to stay gentle).
    if STAB_NEURAL and (frozen is not None):
        try:
            def _fit_sub(Xs):
                r = fit_two_head(Xs, names, source_tau_s=TAU_S, horizon=HORIZON,
                                 hidden=HIDDEN, epochs=STAB_EPOCHS, device="cpu",
                                 return_model=True, seed=SEED, verbose=False)
                if THROTTLE:
                    time.sleep(THROTTLE)   # machine-friendly pause between heavy refits
                return r

            def frozen_gain_fit(Xs):
                r = _fit_sub(Xs)
                return frozen_matrices(r, Xs, names)["gain"]

            def e2e_gain_fit(Xs):
                r = _fit_sub(Xs)
                return r.matrices["gain"]

            out["stability"]["frozen_gain"] = split_half_stability(
                frozen_gain_fit, X, n_splits=NSPLITS, seed=SEED)
            print(f"[frozen] stability frozen_gain spearman="
                  f"{out['stability']['frozen_gain']['spearman_mean']:.3f}")
            out["stability"]["e2e_gain"] = split_half_stability(
                e2e_gain_fit, X, n_splits=NSPLITS, seed=SEED)
            print(f"[frozen] stability e2e_gain spearman="
                  f"{out['stability']['e2e_gain']['spearman_mean']:.3f}")
        except Exception as e:
            out["stability"]["neural_error"] = f"{e}\n{traceback.format_exc()}"
            print(f"[frozen] neural stability failed: {e}")

    # ---- verdict ----
    out["verdict"] = _verdict(out)
    out["runtime_s"] = round(time.time() - t_start, 1)

    path = common.save_json("frozen", out)
    print("\n" + "=" * 78)
    print("[frozen] VERDICT:", out["verdict"])
    print(f"[frozen] wrote {path}  ({out['runtime_s']}s)")
    _print_compact(out)
    return out


def _verdict(out):
    """Short human verdict answering: does frozen beat e2e and/or raw?"""
    s = out.get("auroc_summary", {})
    stab = out.get("stability", {})

    def g(method, key):
        v = s.get(method, {}).get(key)
        return v if (v is not None and np.isfinite(v)) else None

    def sp(method):
        d = stab.get(method, {})
        v = d.get("spearman_mean") if isinstance(d, dict) else None
        return v if (v is not None and np.isfinite(v)) else None

    parts = []
    conf = {m: g(m, "gain_confirmatory") for m in ("frozen", "raw", "e2e")}
    if all(conf[m] is not None for m in conf):
        best = max(conf, key=lambda m: conf[m])
        parts.append(f"confirmatory gain AUROC[neuropeptide:all]: "
                     f"frozen={conf['frozen']:.3f} raw={conf['raw']:.3f} "
                     f"e2e={conf['e2e']:.3f} -> best={best}")
    modu = {m: g(m, "gain_mean_modulatory") for m in ("frozen", "raw", "e2e")}
    if all(modu[m] is not None for m in modu):
        best = max(modu, key=lambda m: modu[m])
        parts.append(f"mean modulatory gain AUROC: frozen={modu['frozen']:.3f} "
                     f"raw={modu['raw']:.3f} e2e={modu['e2e']:.3f} -> best={best}")
    spd = {m: sp(f"{m}_gain") for m in ("frozen", "raw", "e2e")}
    if all(spd[m] is not None for m in spd):
        best = max(spd, key=lambda m: spd[m])
        parts.append(f"gain split-half stability (spearman): frozen={spd['frozen']:.3f} "
                     f"raw={spd['raw']:.3f} e2e={spd['e2e']:.3f} -> most stable={best}")

    if not parts:
        return "insufficient results to compare (see error/neural_ok)."

    # headline: freezing hypothesis = frozen >= e2e on stability AND >= raw on AUROC
    head = []
    if conf.get("frozen") is not None and conf.get("e2e") is not None:
        head.append("frozen>e2e" if conf["frozen"] > conf["e2e"] else "e2e>=frozen")
    if spd.get("frozen") is not None and spd.get("e2e") is not None:
        head.append("frozen more stable than e2e" if spd["frozen"] > spd["e2e"]
                    else "e2e >= frozen on stability")
    prefix = ("FREEZING HELPS: " if head[:1] == ["frozen>e2e"] else "") if head else ""
    return prefix + " | ".join(parts)


def _print_compact(out):
    print("-" * 78)
    print(f"{'method':>8} | {'gain[NP:all]':>12} | {'gain modul.':>11} | "
          f"{'mean struct':>11} | {'gain stab rho':>13}")
    stab = out.get("stability", {})
    for m in ("frozen", "raw", "e2e"):
        s = out.get("auroc_summary", {}).get(m, {})
        st = stab.get(f"{m}_gain", {})
        strho = st.get("spearman_mean") if isinstance(st, dict) else None

        def f(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else "  -  "
        print(f"{m:>8} | {f(s.get('gain_confirmatory')):>12} | "
              f"{f(s.get('gain_mean_modulatory')):>11} | "
              f"{f(s.get('mean_mean_structural')):>11} | {f(strho):>13}")
    print("-" * 78)


if __name__ == "__main__":
    main()
