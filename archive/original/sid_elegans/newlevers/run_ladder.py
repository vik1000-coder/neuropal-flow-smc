r"""LEVER #8 -- DENOISING NOISE-LADDER for the SID gain connectome.

Idea
----
The distributional estimator can be fit with denoising score matching at a noise level
``sigma = sigma_frac * sd(Y)``.  Crucially the readout SUBTRACTS ``sigma**2`` (estimator.py
`_center_readouts` and acmma.py), so every gain matrix on the ladder
``sigma_frac in SIGMA_FRAC_LADDER = [0, 0.25, 0.5, 1.0]`` is on a COMMON scale (they estimate
the same population ``d log Var / dx`` object, only with different amounts of Tikhonov-like
smoothing on the score).  Because they share a scale, aggregating ACROSS the ladder is a pure
variance-reduction move on the gain channel:

    LADDER_GAIN = a precision/consistency-weighted mean of the per-sigma gain matrices.

We build two aggregators and compare them to the single-scale (``sigma=0``) baseline:
  * ``ladder_mean``  : equal-weight mean over the 4 noise scales (plain variance reduction).
  * ``ladder_gated`` : the mean times a per-edge SIGN-CONSISTENCY gate in [0,1]
                       (``|sum_s sign(g_s)| / S``) -- edges whose sign FLIPS across the ladder
                       (i.e. are dominated by scale-specific noise) are shrunk toward 0.

Evaluations (per the lever spec)
  (i)  split_half_stability of the gain matrix: baseline vs ladder aggregators. Expect UP.
  (ii) AUROC vs monoamine:all, neuropeptide:all, monoamine:tyramine, monoamine:serotonin,
       computed at every lag on the ladder grid.
  (iii) CONFIRMATORY: a PRE-SPECIFIED contrast fixed a priori (ladder_mean vs baseline on
        monoamine:serotonin at a pre-registered slow-band lag of 15 frames / 3.75 s), worm-
        bootstrapped as a single test -- this is the headline result.
  (iv) EXPLORATORY: the argmax-selected 'best' contrast over the target x lag x aggregator grid,
        reported with BOTH a naive CI (winning cell held fixed) and a MAX-STATISTIC CI (the argmax
        is RE-SELECTED in every bootstrap replicate so the interval reflects the selection over the
        ~80-cell grid). This is exploratory and does NOT drive the verdict.

Configs: 6w_clean_deconv (SID closed-form) + 28w_deconv (ACMMA available-case).  All heavy
counts come from env vars; heavy work is under __main__ only.  Results -> ladder.json.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np

# these imports are cheap (no data pooling happens at import time)
from sid_elegans.newlevers import common
from sid_elegans.newlevers import harness
from sid_elegans.biolag import config as BC
from sid_elegans.biolag.references import build_registry
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.acmma import fit_acmma_connectome
from sid_elegans.stability import _agreement

# the four AUROC targets required by the lever spec
TARGETS = ["monoamine:all", "neuropeptide:all", "monoamine:tyramine", "monoamine:serotonin"]
PRIMARY_CONFIG = "6w_clean_deconv"
ESTIMATORS = ["baseline", "ladder_mean", "ladder_gated"]

# PRE-REGISTERED confirmatory contrast (fixed a priori, ONE test -> no multiplicity).
# ladder_mean vs baseline on serotonin at a slow-band lag; this drives the verdict.
PRESPEC_CONFIG = PRIMARY_CONFIG
PRESPEC_TARGET = "monoamine:serotonin"
PRESPEC_LAG = 15                 # frames == 3.75 s at 4 fps, pre-registered slow band
PRESPEC_AGG = "ladder_mean"


# --------------------------------------------------------------------------------------
# gain fitting: one config-appropriate estimator, one lag, one noise scale
# --------------------------------------------------------------------------------------
def fit_gain(X_list, names, lag, sigma_frac, config):
    """Return the [N,N] gain matrix ([post,pre], diag zeroed) at one lag / one noise scale.

    6w_clean -> SID closed form (fit_distributional_connectome); 28w -> ACMMA available-case
    (ridge=0.01, min_triple=2000 to match harness/biolag).  Robust: on any failure returns a
    zero matrix (a degenerate-but-finite gain), so downstream scoring/stability never crash.
    """
    N = len(names)
    try:
        if config.startswith("6w"):
            res = fit_distributional_connectome(X_list, names, lag=lag, sigma_frac=float(sigma_frac))
            G = res.matrices["gain"]
        else:
            mats, _ = fit_acmma_connectome(X_list, names, lag=lag, ridge=0.01,
                                           min_triple=2000, sigma_frac=float(sigma_frac))
            G = mats["gain"]
        G = np.asarray(G, float)
        return np.where(np.isfinite(G), G, 0.0)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[warn] fit_gain failed config={config} lag={lag} sigma={sigma_frac}: {e}")
        return np.zeros((N, N))


def ladder_stack(X_list, names, lag, sigmas, config):
    """Fit the gain matrix at every noise scale on the ladder -> {sigma_frac: [N,N]}."""
    return {float(s): fit_gain(X_list, names, lag, s, config) for s in sigmas}


def aggregate(stack, sigmas):
    """Collapse a per-sigma gain stack into the baseline + two ladder aggregators.

    baseline     : the single-scale sigma=0 gain matrix (sigmas[0] is 0.0 on the ladder).
    ladder_mean  : equal-weight mean over the noise scales (variance reduction on a common scale).
    ladder_gated : ladder_mean * per-edge sign-consistency gate in [0,1]; edges whose sign flips
                   across the ladder (scale-specific noise) are shrunk toward zero.
    """
    keys = [float(s) for s in sigmas]
    arr = np.stack([stack[k] for k in keys], axis=0)          # [S, N, N]
    m = arr.mean(axis=0)
    sign_consistency = np.abs(np.sign(arr).sum(axis=0)) / arr.shape[0]   # in [0,1]
    baseline = stack[keys[0]]                                  # sigma_frac == 0.0
    return {"baseline": baseline, "ladder_mean": m, "ladder_gated": m * sign_consistency}


def auroc_of(M, target):
    """AUROC of |offdiag(M)| vs (target>0) over ALL off-diagonal edges (neuromod targets have
    full support, so no confirmed-edge mask is needed -- matches the biolag neuromod scoring)."""
    return harness.auroc_at_lag(np.asarray(M, float), np.asarray(target, float), mask=None)


# --------------------------------------------------------------------------------------
# shared-fit split-half stability across all three estimators at once
# --------------------------------------------------------------------------------------
def multi_stability(X_list, names, lag, sigmas, config, n_splits, seed, throttle):
    """Split-half stability (Spearman + top-10% Jaccard of |offdiag|) for baseline /
    ladder_mean / ladder_gated, fitting the 4-sigma stack ONCE per half so all three share the
    same underlying fits (the baseline is just the sigma=0 slice of that stack)."""
    rng = np.random.default_rng(seed)
    W = len(X_list)
    h = W // 2
    acc = {est: {"rho": [], "jac": []} for est in ESTIMATORS}
    if h < 1:
        return {est: dict(spearman_mean=float("nan"), spearman_sd=float("nan"),
                          jaccard_mean=float("nan"), jaccard_sd=float("nan"), n_splits=0)
                for est in ESTIMATORS}
    for _ in range(n_splits):
        perm = rng.permutation(W)
        idxA, idxB = perm[:h], perm[h:2 * h]
        stackA = ladder_stack([X_list[i] for i in idxA], names, lag, sigmas, config)
        if throttle:
            time.sleep(throttle)
        stackB = ladder_stack([X_list[i] for i in idxB], names, lag, sigmas, config)
        if throttle:
            time.sleep(throttle)
        agA, agB = aggregate(stackA, sigmas), aggregate(stackB, sigmas)
        for est in ESTIMATORS:
            rho, jac = _agreement(agA[est], agB[est])
            acc[est]["rho"].append(rho)
            acc[est]["jac"].append(jac)
    out = {}
    for est in ESTIMATORS:
        r = np.array(acc[est]["rho"], float)
        j = np.array(acc[est]["jac"], float)
        out[est] = dict(spearman_mean=float(r.mean()), spearman_sd=float(r.std()),
                        jaccard_mean=float(j.mean()), jaccard_sd=float(j.std()),
                        n_splits=int(len(r)))
    return out


def load_targets(names):
    """Return {target_name: adjacency [N,N]} for the four required neuromodulator references."""
    reg = {r.name: r for r in build_registry(names, min_specific_edges=10, verbose=False)}
    out = {}
    for t in TARGETS:
        if t in reg:
            out[t] = np.asarray(reg[t].adjacency, float)
        else:
            print(f"[warn] target {t} not in registry on this support; skipping")
    return out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")  # estimator emits benign 'invalid eta2' clamap warnings

    SMOKE = os.environ.get("NEWLEVERS_SMOKE", "0") == "1"
    SEED = common.env_int("NL_SEED", 0)
    sigmas = list(BC.SIGMA_FRAC_LADDER)                       # [0.0, 0.25, 0.5, 1.0]

    # config selection: explicit NL_CONFIG pins a single config; else full run does BOTH configs
    # (the lever's spec), and a smoke run does only the primary (fast) config.
    nl_config = os.environ.get("NL_CONFIG", "").strip()
    if nl_config:
        configs = [nl_config]
    elif SMOKE:
        configs = [PRIMARY_CONFIG]
    else:
        configs = ["6w_clean_deconv", "28w_deconv"]

    if SMOKE:
        lags = [3, 10]
        stab_lags = [3]
        n_splits = 2
        n_boot = min(5, common.env_int("NL_NBOOT", 200))
        throttle = 0.0
    else:
        lags = list(BC.LAG_FRAMES)
        stab_lags = [3, 15]                                  # one fast, one slow band anchor
        n_splits = common.env_int("NL_NSPLITS", 5)
        n_boot = common.env_int("NL_NBOOT", 200)
        throttle = common.env_float("NL_THROTTLE", 0.3)

    meta = dict(smoke=SMOKE, configs=configs, sigmas=sigmas, lags=lags, stab_lags=stab_lags,
                n_splits=n_splits, n_boot=n_boot, throttle=throttle, seed=SEED, targets=TARGETS,
                note=("gain matrices are sigma^2-corrected -> common scale -> ladder aggregation "
                      "is variance reduction; ladder_gated additionally shrinks sign-flipping edges. "
                      "NL_CONFIG (if set) pins a single config; unset => full run does both configs."))
    meta["prespec"] = dict(config=PRESPEC_CONFIG, target=PRESPEC_TARGET,
                           lag=PRESPEC_LAG, aggregator=PRESPEC_AGG)
    results = {"meta": meta, "per_config": {}, "confirmatory": None,
               "best_contrast": None, "verdict": {}}

    # collector for the global best AUROC contrast (ladder aggregator - baseline)
    best = {"contrast": -np.inf}

    for config in configs:
        cres = {"error": None}
        try:
            X_list, names, fps = common.get_data(config)
            N = len(names)
            targets = load_targets(names)
            cres["n_worms"] = len(X_list)
            cres["n_neurons"] = N
            cres["target_edges"] = {t: int((A[~np.eye(N, dtype=bool)] > 0).sum())
                                    for t, A in targets.items()}

            # ---- full-data per-lag ladder fits (cache the aggregates) ----
            aggs = {}
            for L in lags:
                stack = ladder_stack(X_list, names, L, sigmas, config)
                aggs[L] = aggregate(stack, sigmas)
                if throttle:
                    time.sleep(throttle)

            # ---- (ii) AUROC per lag, per estimator, per target ----
            auroc = {t: {est: [] for est in ESTIMATORS} for t in targets}
            for t, A in targets.items():
                for L in lags:
                    for est in ESTIMATORS:
                        auroc[t][est].append(auroc_of(aggs[L][est], A))
            cres["auroc"] = {t: {est: [None if not np.isfinite(v) else float(v)
                                       for v in auroc[t][est]] for est in ESTIMATORS}
                             for t in targets}

            # track best local contrast (ladder aggregator vs baseline) for this config
            local_best = {"contrast": -np.inf}
            for t in targets:
                base = np.array(auroc[t]["baseline"], float)
                for est in ("ladder_mean", "ladder_gated"):
                    cur = np.array(auroc[t][est], float)
                    d = cur - base
                    for li, L in enumerate(lags):
                        if np.isfinite(d[li]) and d[li] > local_best["contrast"]:
                            local_best = {"contrast": float(d[li]), "target": t,
                                          "lag": int(L), "aggregator": est,
                                          "auroc_ladder": float(cur[li]),
                                          "auroc_baseline": float(base[li])}
            cres["best_contrast_local"] = local_best
            if local_best["contrast"] > best["contrast"]:
                best = {**local_best, "config": config}

            # ---- (i) split-half stability at the stability lags ----
            stab = {}
            for sl in stab_lags:
                stab[str(sl)] = multi_stability(X_list, names, sl, sigmas, config,
                                                n_splits=n_splits, seed=SEED, throttle=throttle)
            cres["stability"] = stab

        except Exception as e:  # keep going; emit partial JSON
            cres["error"] = repr(e)
            print(f"[error] config {config} failed: {e!r}")
        results["per_config"][config] = cres

    # ---- (iii) CONFIRMATORY: pre-specified contrast, fixed a priori, single worm-bootstrap ----
    confirm = dict(config=PRESPEC_CONFIG, target=PRESPEC_TARGET, lag=PRESPEC_LAG,
                   aggregator=PRESPEC_AGG, prespecified=True, error=None)
    try:
        Xc, namesc, _ = common.get_data(PRESPEC_CONFIG)
        targetsc = load_targets(namesc)
        if PRESPEC_TARGET not in targetsc:
            confirm["error"] = f"pre-specified target {PRESPEC_TARGET} not on this support"
        else:
            Ac = targetsc[PRESPEC_TARGET]

            def confirm_stat(Xs):
                ag = aggregate(ladder_stack(Xs, namesc, PRESPEC_LAG, sigmas, PRESPEC_CONFIG), sigmas)
                a_lad = auroc_of(ag[PRESPEC_AGG], Ac)
                a_base = auroc_of(ag["baseline"], Ac)
                if not (np.isfinite(a_lad) and np.isfinite(a_base)):
                    return np.nan
                return a_lad - a_base

            boot = harness.worm_bootstrap(confirm_stat, Xc, n_boot=n_boot,
                                          rng=np.random.default_rng(SEED), throttle=throttle)
            confirm["bootstrap"] = boot
            confirm["contrast"] = boot.get("obs")
    except Exception as e:
        confirm["error"] = repr(e)
        print(f"[error] confirmatory bootstrap failed: {e!r}")
    results["confirmatory"] = confirm

    # ---- (iv) EXPLORATORY: argmax 'best' contrast with a naive CI AND a max-statistic CI ----
    # The naive CI holds the winning cell fixed (ignores selection); the max-statistic CI
    # RE-SELECTS the argmax over the full target x lag x aggregator grid in every replicate so
    # the interval properly reflects that we cherry-picked the best of ~80 contrasts.
    best["kind"] = "exploratory"
    if np.isfinite(best.get("contrast", -np.inf)) and "config" in best:
        bc, bt, bl, bagg = best["config"], best["target"], best["lag"], best["aggregator"]
        try:
            Xb, namesb, _ = common.get_data(bc)
            targetsb = load_targets(namesb)
            A = targetsb[bt]
            grid_lags = list(lags)

            # naive: winning contrast (target x lag x aggregator) held fixed across replicates
            def stat_naive(Xs):
                ag = aggregate(ladder_stack(Xs, namesb, bl, sigmas, bc), sigmas)
                a_lad = auroc_of(ag[bagg], A)
                a_base = auroc_of(ag["baseline"], A)
                if not (np.isfinite(a_lad) and np.isfinite(a_base)):
                    return np.nan
                return a_lad - a_base

            # max-statistic: re-select the argmax contrast over the SAME grid every replicate
            def stat_maxstat(Xs):
                best_d = -np.inf
                for L in grid_lags:
                    ag = aggregate(ladder_stack(Xs, namesb, L, sigmas, bc), sigmas)
                    for t, At in targetsb.items():
                        a_base = auroc_of(ag["baseline"], At)
                        if not np.isfinite(a_base):
                            continue
                        for est in ("ladder_mean", "ladder_gated"):
                            a_lad = auroc_of(ag[est], At)
                            if np.isfinite(a_lad):
                                best_d = max(best_d, a_lad - a_base)
                return best_d if np.isfinite(best_d) else np.nan

            best["bootstrap_naive"] = harness.worm_bootstrap(
                stat_naive, Xb, n_boot=n_boot, rng=np.random.default_rng(SEED), throttle=throttle)
            best["bootstrap_maxstat"] = harness.worm_bootstrap(
                stat_maxstat, Xb, n_boot=n_boot, rng=np.random.default_rng(SEED), throttle=throttle)
        except Exception as e:
            best["bootstrap_naive"] = {"error": repr(e)}
            best["bootstrap_maxstat"] = {"error": repr(e)}
            print(f"[error] exploratory bootstrap failed: {e!r}")
    results["best_contrast"] = best

    # ---- verdicts ----
    # stability: did either ladder aggregator raise split-half Spearman above baseline (averaged
    # over the stability lags and the analysed configs)?
    def _mean_stab(est):
        vals = []
        for config in configs:
            st = results["per_config"].get(config, {}).get("stability", {})
            for sl, d in st.items():
                v = d.get(est, {}).get("spearman_mean")
                if v is not None and np.isfinite(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    stab_base = _mean_stab("baseline")
    stab_mean = _mean_stab("ladder_mean")
    stab_gated = _mean_stab("ladder_gated")
    stability_up = bool(np.isfinite(stab_base) and
                        (max(stab_mean, stab_gated) > stab_base + 1e-6))

    # CONFIRMATORY verdict (headline): keys off the PRE-SPECIFIED contrast, not the argmax.
    cboot = confirm.get("bootstrap", {}) if isinstance(confirm.get("bootstrap"), dict) else {}
    clo, cobs = cboot.get("lo"), cboot.get("obs")
    if confirm.get("error"):
        confirmatory = f"confirmatory contrast unavailable ({confirm['error']})"
        confirmed = False
    elif clo is not None and np.isfinite(clo) and clo > 0:
        confirmatory = ("CONFIRMED: pre-specified ladder_mean vs baseline on monoamine:serotonin "
                        f"at lag {PRESPEC_LAG} improves AUROC (bootstrap 95% CI excludes 0)")
        confirmed = True
    elif cobs is not None and np.isfinite(cobs) and cobs > 0:
        confirmatory = ("pre-specified contrast is positive but its bootstrap CI includes 0 "
                        "(NOT significant)")
        confirmed = False
    else:
        confirmatory = "pre-specified contrast shows no AUROC improvement"
        confirmed = False

    # EXPLORATORY note (does NOT drive the verdict): does the argmax survive its max-statistic CI?
    mboot = best.get("bootstrap_maxstat", {}) if isinstance(best.get("bootstrap_maxstat"), dict) else {}
    nboot = best.get("bootstrap_naive", {}) if isinstance(best.get("bootstrap_naive"), dict) else {}
    mlo = mboot.get("lo")
    if "config" not in best:
        exploratory = "no exploratory best contrast available"
    elif mlo is not None and np.isfinite(mlo) and mlo > 0:
        exploratory = ("exploratory argmax SURVIVES selection correction "
                       "(max-statistic 95% CI excludes 0)")
    else:
        exploratory = ("exploratory argmax does NOT survive selection correction "
                       "(max-statistic CI includes 0) -- likely winner's curse")

    results["verdict"] = dict(
        stability_baseline_mean=stab_base, stability_ladder_mean_mean=stab_mean,
        stability_ladder_gated_mean=stab_gated, stability_up=stability_up,
        confirmed=confirmed, confirmatory=confirmatory, exploratory=exploratory,
        confirmatory_contrast=confirm.get("contrast"),
        confirmatory_ci=[cboot.get("lo"), cboot.get("hi")],
        exploratory_contrast=best.get("contrast") if "config" in best else None,
        exploratory_ci_naive=[nboot.get("lo"), nboot.get("hi")],
        exploratory_ci_maxstat=[mboot.get("lo"), mboot.get("hi")],
        text=(f"Noise-ladder gain aggregation {'RAISES' if stability_up else 'does NOT raise'} "
              f"split-half stability (baseline {stab_base:.3f} vs "
              f"ladder_mean {stab_mean:.3f} / ladder_gated {stab_gated:.3f}). "
              f"HEADLINE (confirmatory): {confirmatory}. "
              f"Exploratory: {exploratory}."))

    common.save_json("ladder", results)

    # ---- concise human summary ----
    print("\n================ LEVER #8  DENOISING NOISE-LADDER ================")
    print(f"configs={configs}  sigmas={sigmas}  lags={lags}  n_splits={n_splits}  n_boot={n_boot}")
    for config in configs:
        c = results["per_config"].get(config, {})
        if c.get("error"):
            print(f"[{config}] ERROR: {c['error']}")
            continue
        print(f"\n[{config}] {c.get('n_worms')} worms x {c.get('n_neurons')} neurons")
        st = c.get("stability", {})
        for sl, d in st.items():
            print(f"  stability lag={sl}:  "
                  f"baseline rho={d['baseline']['spearman_mean']:.3f}  "
                  f"ladder_mean rho={d['ladder_mean']['spearman_mean']:.3f}  "
                  f"ladder_gated rho={d['ladder_gated']['spearman_mean']:.3f}")
        lb = c.get("best_contrast_local", {})
        if "target" in lb:
            print(f"  best local AUROC contrast: {lb['aggregator']} vs baseline on {lb['target']} "
                  f"lag={lb['lag']}: {lb['auroc_baseline']:.3f} -> {lb['auroc_ladder']:.3f} "
                  f"({lb['contrast']:+.3f})")
    cf = results["confirmatory"] or {}
    print(f"\nCONFIRMATORY (pre-specified): {cf.get('aggregator')} vs baseline | "
          f"config={cf.get('config')} target={cf.get('target')} lag={cf.get('lag')}")
    cbt = cf.get("bootstrap", {}) if isinstance(cf.get("bootstrap"), dict) else {}
    if "lo" in cbt:
        print(f"  worm-bootstrap: obs={cbt['obs']:+.3f}  mean={cbt['mean']:+.3f}  "
              f"95% CI=[{cbt['lo']:+.3f}, {cbt['hi']:+.3f}]  n_boot={cbt['n_boot']}")
    elif cf.get("error"):
        print(f"  (unavailable: {cf['error']})")

    b = results["best_contrast"]
    if "config" in b:
        print(f"\nEXPLORATORY best contrast (argmax): {b['aggregator']} vs baseline | "
              f"config={b['config']} target={b['target']} lag={b['lag']}  "
              f"obs={b['contrast']:+.3f}")
        nbt = b.get("bootstrap_naive", {}) if isinstance(b.get("bootstrap_naive"), dict) else {}
        mbt = b.get("bootstrap_maxstat", {}) if isinstance(b.get("bootstrap_maxstat"), dict) else {}
        if "lo" in nbt:
            print(f"  naive CI (cell fixed):     obs={nbt['obs']:+.3f}  "
                  f"95% CI=[{nbt['lo']:+.3f}, {nbt['hi']:+.3f}]  n_boot={nbt['n_boot']}")
        if "lo" in mbt:
            print(f"  max-statistic CI (re-sel): obs={mbt['obs']:+.3f}  "
                  f"95% CI=[{mbt['lo']:+.3f}, {mbt['hi']:+.3f}]  n_boot={mbt['n_boot']}")
    print(f"\nVERDICT: {results['verdict']['text']}")
    print("=================================================================\n")
    return results


if __name__ == "__main__":
    main()
