import numpy as np
from sid_elegans.newlevers import common as cm

X, names, fps = cm.get_data("6w_clean_raw")
N = len(names); print(f"data: {len(X)} worms, N={N}, fps={fps}")

# global signal + phases
g = cm.global_signal(X[0], "mean"); g2 = cm.global_signal(X[0], "pc1")
print("global_signal mean/pc1 shapes:", g.shape, g2.shape, "corr:", round(float(np.corrcoef(g,g2)[0,1]),3))
hi, lo = cm.phase_masks(X, "mean"); Xhi = cm.subset_frames(X, hi)
print("phase split: hi worms kept =", len(Xhi), " hi frac worm0 =", round(float(hi[0].mean()),2))

# class pooling
G, classes = cm.pooling_operator(names)
print(f"coarse classes: {len(classes)} (from {N}) e.g.", classes[:6])
M = np.random.default_rng(0).standard_normal((N,N)); Mc = cm.pool_matrix(M, G)
print("pooled matrix shape:", Mc.shape, "diag zero:", bool(np.allclose(np.diag(Mc),0)))

# receptor sets
for mod in ["neuropeptide","monoamine","serotonin","tyramine"]:
    e = cm.receptor_expressing(names, mod); print(f"  receptor-expressing[{mod}]: {int(e.sum())}/{N}")

# variance-matched controls
V = cm.source_variance(X); exp = cm.receptor_expressing(names,"neuropeptide")
ctrl = cm.variance_matched_controls(exp, V, np.random.default_rng(0))
print("variance-matched controls:", int(ctrl.sum()), " overlap w/ expressing:", int((ctrl&exp).sum()))
print("  logV expressing mean %.2f vs control mean %.2f" % (np.log(V[exp]+1e-9).mean(), np.log(V[ctrl]+1e-9).mean()))

# self-gain exposure (estimator)
from sid_elegans.estimator import fit_distributional_connectome
res = fit_distributional_connectome(X, names, lag=5)
assert res.diagonal is not None, "diagonal not exposed"
sg = res.diagonal["gain"]
print("SELF-GAIN (estimator) lag5: nonzero targets =", int((sg!=0).sum()), "/", N,
      " range [%.3f, %.3f]" % (sg.min(), sg.max()))

# self-gain exposure (acmma)
from sid_elegans.acmma import fit_acmma_connectome
mats, diag = fit_acmma_connectome(X, names, lag=5, ridge=0.01, min_triple=2)
print("SELF-GAIN (acmma) present:", "self_diag" in diag, " nonzero =", int((diag["self_diag"]["gain"]!=0).sum()))

# predict_params for #5
from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher
qsm = QuadraticScoreMatcher(ridge=1e-2)
Y = X[0][6:,0]; Psi = np.concatenate([np.ones((len(Y),1)), np.nan_to_num(X[0][:-6],nan=0.0)],1)
fit = qsm.fit(Y[:400], Psi[:400])
pp = qsm.predict_params(Psi[400:450]); print("predict_params returns", len(pp), "arrays; nll:", round(float(qsm.nll(Y[400:450],Psi[400:450])),3))
print("SMOKE OK")
