r"""Phase 2 (SCRATCH venv): VAR-LiNGAM + SINDy per-lag effect matrices, all [post,pre].
Run with: /tmp/baseline_feas/bin/python sid_elegans/lagres/scratch_effects.py
Reads /tmp/worms.npz, writes /tmp/scratch_effects.npz (keys like 'varlingam_5','sindy_marg_5').
Standalone (numpy + lingam + pysindy only). Functions verified during planning.
"""
import time

import numpy as np

LAGS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
P_JOINT = 20   # joint order for partial methods (lags 1..20)


def _clean(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    return np.where(np.isnan(X), mu, X)


def pool_pairs(X_list, lag):
    now, fut = [], []
    for X in X_list:
        X = _clean(X)
        if X.shape[0] <= lag:
            continue
        now.append(X[:X.shape[0] - lag]); fut.append(X[lag:])
    return np.concatenate(now, 0), np.concatenate(fut, 0)


def pooled_var_design(X_list, p):
    Zs, Ys = [], []
    for X in X_list:
        X = _clean(X); T, N = X.shape
        if T <= p:
            continue
        blocks = [X[p - l:T - l] for l in range(1, p + 1)]
        Zs.append(np.concatenate(blocks, 1)); Ys.append(X[p:])
    return np.concatenate(Zs, 0), np.concatenate(Ys, 0)


def varlingam_partial(X_list, p, prune=False):
    from lingam import VARLiNGAM
    Z, Y = pooled_var_design(X_list, p); N = Y.shape[1]
    W, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    M_taus = np.stack([W[(l - 1) * N:l * N, :].T for l in range(1, p + 1)], 0)  # [post,pre]
    Xcat = np.concatenate([_clean(x) for x in X_list], 0)
    m = VARLiNGAM(lags=p, criterion=None, prune=prune, ar_coefs=M_taus, random_state=0).fit(Xcat)
    A = m.adjacency_matrices_
    out = {}
    for l in range(1, p + 1):
        e = A[l].copy(); np.fill_diagonal(e, 0.0); out[l] = e
    return out


def sindy_marginal(X_list, lag, threshold=0.0):
    import pysindy as ps
    Xn, Xf = pool_pairs(X_list, lag)
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=1, include_bias=True),
                     optimizer=ps.STLSQ(threshold=threshold, alpha=0.0))
    model.fit(Xn, x_dot=Xf, t=1.0)
    E = model.coefficients()[:, 1:].copy()   # [post,pre]
    np.fill_diagonal(E, 0.0)
    return E


def sindy_partial_delay(X_list, p, threshold=0.0):
    import pysindy as ps
    Zs, Ys = [], []
    for X in X_list:
        X = _clean(X); T, N = X.shape
        if T <= p:
            continue
        blocks = [X[p - 1 - k:T - 1 - k] for k in range(p)]
        Zs.append(np.concatenate(blocks, 1)); Ys.append(X[p:])
    Z = np.concatenate(Zs, 0); Y = np.concatenate(Ys, 0); N = Y.shape[1]
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=1, include_bias=True),
                     optimizer=ps.STLSQ(threshold=threshold, alpha=0.0))
    model.fit(Z, x_dot=Y, t=1.0)
    C = model.coefficients()
    out = {}
    for k in range(p):
        Ek = C[:, 1 + k * N:1 + (k + 1) * N].copy(); np.fill_diagonal(Ek, 0.0)
        out[k + 1] = Ek
    return out


def main():
    d = np.load("/tmp/worms.npz", allow_pickle=True)
    n = int(d["n"]); X_list = [d[f"w{i}"] for i in range(n)]
    N = X_list[0].shape[1]
    print(f"scratch: {n} worms, N={N}")
    out = {}

    t = time.time()
    vp = varlingam_partial(X_list, P_JOINT, prune=False)
    for l in [x for x in LAGS if x <= P_JOINT]:
        out[f"varlingam_{l}"] = vp[l]
    print(f"  VAR-LiNGAM partial(p={P_JOINT}): {time.time()-t:.1f}s")

    t = time.time()
    for l in LAGS:
        out[f"sindy_marg_{l}"] = sindy_marginal(X_list, l)
    print(f"  SINDy marginal (all lags): {time.time()-t:.1f}s")

    t = time.time()
    sp = sindy_partial_delay(X_list, P_JOINT)
    for l in [x for x in LAGS if x <= P_JOINT]:
        out[f"sindy_part_{l}"] = sp[l]
    print(f"  SINDy partial-delay(p={P_JOINT}): {time.time()-t:.1f}s")

    np.savez("/tmp/scratch_effects.npz", **out)
    print(f"  wrote /tmp/scratch_effects.npz — {len(out)} matrices")


if __name__ == "__main__":
    main()
