# CLUSTER: eval_and_neural

## conventions
CENTRAL DATA CONVENTIONS a new script MUST respect (all paths under /Users/vik/Developer/new_sbtg_neuro/sid_elegans):

X_list: list of per-worm float ndarrays, each shape [T_w, N] (axis0=time/frames, axis1=neuron). N and column order are IDENTICAL across worms and equal to `neuron_names`. Arrays MAY contain NaN; every consumer here does its own np.nan_to_num(X, nan=0.0) (baselines._pool, twohead/_slow_filter+_pool, mdn same). Sampling default fps=4.0 Hz. Built upstream by data.load_traces / combined_data.load_combined (return (X_list, names, fps)).

neuron_names: list[str], length N; index i in a matrix column = this neuron.

MATRIX ORIENTATION = [post, pre] EVERYWHERE. M has shape [N, N]; M[j, i] is the directed coupling from source/pre i (COLUMN) to target/post j (ROW). This holds for baselines.pearson_lag/ridge_var, twohead/mdn `.matrices['mean'|'gain']`, estimator.fit_distributional_connectome (docstring: matrices[channel][i,j] = source j -> target i), AND ground-truth matrices (load_cook/load_leifer/load_all_monoamine_layers). NEW ground-truth/target matrices (e.g. functional connectome) MUST be produced in [post,pre] and aligned to `neuron_names` order or AUROC is silently transposed. MEMORY note: the Cook connectome was historically saved [pre,post] (transpose bug); ground_truth.load_cook returns the corrected [post,pre] — do not re-transpose.

DIAGONAL: self-edges are ZEROED by convention. baselines/twohead/mdn all call np.fill_diagonal(M, 0.0) as the last step; evaluate._offdiag / stability._agreement drop the diagonal before scoring. The gain/mean DIAGONAL is actually computed inside the twohead/mdn readout loop (the per-target gradient vector includes its own index) and then discarded by fill_diagonal — so self-gain is recoverable pre-zeroing.

SCORING is SIGN-AGNOSTIC: score_matrix / stability use np.abs(offdiag) as the score; ground truth is binarized gt>0. A signed functional connectome loses its sign at scoring unless you change this.

CHANNELS: 'mean' = d E[y_j|h]/d h_i (conditional-mean / linear-drive channel, comparable to pearson/ridge_var), 'gain' = d log Var[y_j|h]/d h_i (the neuromodulatory / state-dependent-gain channel). Couplings are read out by autograd averaging the SIGNED gradient over histories, then divided by h_sd to undo history standardization (units = raw source units).

UNITS GOTCHA: twohead/mdn `source_tau_s` is in SECONDS (fed to exp_filter_bank via t=arange(T)/fps). estimator.fit_distributional_connectome `source_tau` is in FRAMES. Do not mix. `horizon` (twohead/mdn) and `lag` (baselines/estimator) are both integer frame offsets: history=X[:T-h], future=X[h:].

STANDARDIZATION: trunk/history features are z-scored per-column on the pooled data (h_mu, h_sd = Hn.mean(0), Hn.std(0)+1e-6); the FUTURE/target is left raw. Gradients are rescaled by /h_sd at readout.

GLOBAL RNG: fit_two_head and fit_mdn call torch.manual_seed(seed) AND np.random.seed(seed) globally (mutating process RNG), plus a local np.random.default_rng(seed) for the eval subsample. Reproducible but a side effect on the caller.

source_variance / target_variance trivial baselines (defined inline in run_holistic.py:51-52, run_deconv.py, run_expressive.py): srcvar = tile(nanvar(allX,0)[None,:],(N,1)) -> constant DOWN columns, i.e. each source column i carries var(neuron i); tgtvar = tile(...[:,None]) -> constant across rows. Both diag-zeroed. These are the 'is the method just reading off marginal variance?' controls; new gain-channel claims should beat source_variance on the SAME target.

## key_functions
### score_matrix  (evaluate.py)
- sig: `score_matrix(coupling: np.ndarray, gt: np.ndarray, binarize_gt: bool = True) -> dict`
- returns: dict with keys: n_eval:int, n_pos:int, auroc:float, auprc:float, spearman:float, f1:float. auroc/auprc = sklearn on y_score=|offdiag(coupling)| vs y_true=(offdiag(gt)>0). spearman = Spearman(|score|, continuous offdiag gt weight). f1 at top-k where k=#positives. All-nan floats returned when degenerate.
- purpose: THE evaluation entry point: AUROC/AUPRC/Spearman/F1 of a [post,pre] coupling matrix against a [post,pre] ground-truth connectome, off-diagonal only.
- gotchas: Both args must be [post,pre] and index-aligned to the same neuron_names, else AUROC is transposed silently. Returns auroc=NaN (not error) when y_true is empty, all-0, or all-1 after NaN-drop. valid mask = isfinite(score)&isfinite(gt) drops NaN edges consistently on BOTH. binarize_gt=False keeps continuous gt for AUROC (thresholded >0). |coupling| discards sign. F1 threshold uses np.sort(...)[::-1][min(k,len-1)] (strict >), so ties near the boundary can shift the F1 slightly. No in-place mutation of inputs.

### _offdiag  (evaluate.py)
- sig: `_offdiag(A) -> np.ndarray`
- returns: 1-D ndarray length N*(N-1), row-major order over the off-diagonal (mask ~np.eye(N,bool)), dtype of A.
- purpose: Flatten off-diagonal entries; shared basis for every metric so score and gt line up element-wise.
- gotchas: Assumes square A. Order is deterministic row-major; a class-aggregated matrix must be square [C,C] to reuse it.

### split_half_stability  (stability.py)
- sig: `split_half_stability(fit_matrix, X_list, n_splits=5, seed=0, top_frac=0.1) -> dict`
- returns: dict{spearman_mean, spearman_sd, jaccard_mean, jaccard_sd} of floats. Per split: _agreement returns (Spearman of |offdiag(A)| vs |offdiag(B)|, top-k Jaccard at k=max(1,top_frac*len)).
- purpose: Label-free (never touches ground truth) reproducibility score: fit the connectome on two disjoint random halves of the worms and measure off-diagonal agreement. Self-supervised model-selection objective.
- gotchas: fit_matrix is a callable X_subset(list) -> [N,N] ndarray. Calls it 2*n_splits times -> expensive for neural fits (run_scaling.py:49 gates two-head to nworm>=6). Uses W//2 per half (perm[:h] and perm[h:2h]); with odd #worms one worm is dropped each split. Returns (0.0,0.0) for a split if either matrix has std<1e-12 (degenerate/constant). np.abs applied -> sign-agnostic.

### tune_by_stability  (stability.py)
- sig: `tune_by_stability(fit_matrix_for, X_list, grid, n_splits=5, seed=0, top_frac=0.1) -> (best_config: dict, table: list[dict])`
- returns: (best, table). table = list of {**cfg, **stability_dict}; best = argmax over spearman_mean. best is the merged cfg+stability dict.
- purpose: Grid-search hyperparameters by split-half stability without peeking at labels.
- gotchas: fit_matrix_for(cfg) must RETURN a closure X_subset -> [N,N] (two-level currying). grid is a list of dicts. Selection key is hardcoded to spearman_mean (not jaccard).

### pearson_lag  (baselines.py)
- sig: `pearson_lag(X_list, lag: int) -> np.ndarray`
- returns: P: ndarray [N,N] float64, [post,pre]: P[j,i]=corr(x_i(t), x_j(t+lag)). Diagonal zeroed.
- purpose: Trivial directed lagged-correlation baseline / comparator to the mean channel.
- gotchas: Pools all worms via _pool (NaN->0). Standardizes columns with (x-mean)/(std+1e-8). Worms with T<=lag are skipped. Returns a fresh array (C.T.copy()).

### ridge_var  (baselines.py)
- sig: `ridge_var(X_list, lag: int, ridge: float = 1.0) -> np.ndarray`
- returns: B: ndarray [N,N] float64, [post,pre]: B[j,i] = ridge-regression coefficient of source i predicting target j at the lag. Diagonal zeroed.
- purpose: Multivariate linear conditional-mean (ridge-VAR) baseline — the linear analog of the sid_neuromod mean channel.
- gotchas: Inverts one NxN Gram (G = Xc.T@Xc + ridge*I) via np.linalg.inv then loops j over targets. Centers sources and each target but does NOT scale to unit variance (unlike pearson_lag), so coefficients are in raw units. NaN->0 pooling. Not intercept-augmented (both sides centered).

### fit_two_head  (twohead.py)
- sig: `fit_two_head(X_list, neuron_names, source_tau_s=20.0, horizon=1, hidden=128, trunk_layers=2, epochs=300, lr=2e-3, weight_decay=1e-4, batch=512, sigma_c=0.3, sigma_m=0.3, lam_marg=1.0, fps=4.0, seed=0, device=None, verbose=False) -> TwoHeadResult`
- returns: TwoHeadResult(neuron_names:list, matrices:{'mean':[N,N] float64, 'gain':[N,N] float64} both [post,pre] diag-zeroed, history:{'cond':list[float],'marg':list[float]}). The trained nn.Module is NOT returned.
- purpose: Train the two-head DSM model (shared trunk + Gaussian conditional (mu,logv) head + separate history-marginal score head) and read out nonlinear mean/gain couplings via autograd through the conditional head.
- gotchas: MODEL DISCARDED after fit — no frozen trunk / no held-out scorer exposed (blocks frozen_features and predictive_likelihood without editing this fn to also return `model`, h_mu, h_sd). device auto = 'mps' if available else 'cpu'; result arrays copied to CPU numpy. Sets global torch+numpy seeds. History h = exp_filter_bank(X, taus=[source_tau_s sec]) when source_tau_s truthy, else raw X; source_tau_s=0/None -> instantaneous. History z-scored (h_mu,h_sd); readout gradients /h_sd. DSM bias correction: v_true=clamp(exp(logv)-sigma_c**2, min=1e-4) at line 160 (single scalar sigma_c). Diagonal computed then zeroed at line 170. Eval subsample n_eval=min(4000,M). logv clamped [-6,4].

### TwoHead (module)  (twohead.py)
- sig: `TwoHead(n, hidden=128, trunk_layers=2, marg_layers=2, logv_floor=-6.0, logv_ceil=4.0); .conditional(h)->(mu,logv); .marginal_score(h)->s; attrs .trunk (Sequential N->hidden), .mu_head, .logv_head, .marg`
- returns: .conditional(h): (mu:[B,N], logv:[B,N] clamped). .trunk(h): [B,hidden] frozen-feature embedding. .marginal_score(h): [B,N].
- purpose: The architecture: FROZEN feature embedding = self.trunk(h) (shared over targets); linear mu_head/logv_head are the closed-form-able probes on top. This is the object frozen_features needs.
- gotchas: trunk input is the STANDARDIZED history (N-dim), not raw X — a frozen-feature probe must apply the same (h-h_mu)/h_sd used at fit time (currently not returned by fit_two_head). conditional() returns EFFECTIVE logv (includes +sigma_c^2 DSM inflation); subtract sigma_c**2 for the true predictive variance.

### fit_mdn  (mdn.py)
- sig: `fit_mdn(X_list, neuron_names, source_tau_s=20.0, horizon=1, k=3, hidden=128, layers=2, epochs=400, lr=2e-3, weight_decay=1e-2, batch=512, fps=4.0, seed=0, device=None, verbose=False) -> MDNResult`
- returns: MDNResult(neuron_names:list, matrices:{'mean':[N,N] float64,'gain':[N,N] float64} [post,pre] diag-zeroed, history:list[float] (per-epoch mean NLL)). Model NOT returned.
- purpose: K-component Gaussian mixture-density conditional per target, fit by MLE; reads out mean = d E[y|h]/dh and gain = d log Var[y|h]/dh through the mixture moments. The genuinely non-Gaussian estimator.
- gotchas: MLE (no DSM noise / no sigma_c) — noise_ladder does NOT apply here. Model discarded (same held-out/frozen limitation as twohead). Global seeds set. Var clamped min=1e-4. Same standardization + /h_sd readout. n_eval=min(4000,M).

### MDN (module)  (mdn.py)
- sig: `MDN(n, k=3, hidden=128, layers=2, logv_floor=-6.0, logv_ceil=4.0); .params(h)->(logpi,mu,logv) each [B,N,k]; .nll(h,y)->scalar; .moments(h)->(mean,var) each [B,N]`
- returns: .nll(h,y): scalar tensor = -mean over B and N of per-target logsumexp mixture log-density. .moments: (mean[B,N], var[B,N] clamped min 1e-4). .trunk: Sequential N->hidden.
- purpose: Holds the predictive-likelihood surface (.nll) and closed-form conditional moments. .nll is the natural held-out one-step predictive-likelihood objective for idea 5.
- gotchas: .nll AVERAGES over targets (mean over N) -> a single scalar; to LOCALIZE likelihood to receptor-expressing vs control neurons you need PER-TARGET NLL = replace `-logp.mean()` (mdn.py:64) with `-logp.mean(0)` (length-N vector). y is passed raw (unstandardized future); h must be standardized identically to training.

### exp_filter_bank  (filter_bank.py)
- sig: `exp_filter_bank(z, timestamps_s, timescales_s, *, init='zero') -> np.ndarray`
- returns: F: ndarray [T, J, K] float64. F[k,j,r] = causal exponential filter of signal j at timescale r, time k. Strictly causal (uses z[0..k]).
- purpose: Slow-drive source featurizer used by twohead._slow_filter / mdn._slow_filter (both call it with a single tau and squeeze [:, :, 0]).
- gotchas: timestamps must be STRICTLY increasing (raises on dt<=0); timescales must be >0 (raises). t built as arange(T)/fps in the callers (seconds). init='zero' means the first ~tau seconds ramp up from 0. as_2d imported from sid_neuromod.utils.arrays.

## integration_hooks
- **1 functional_target**: PURE evaluate.py hook — score_matrix is already target-agnostic. ground_truth.load_leifer ALREADY EXISTS and is imported at run_eval.py:27 (Randi/Leifer functional data). Build the functional matrix in [post,pre] aligned to `neuron_names`, add it to the `targets` dict in run_holistic.py:70-73 / run_eval.py, and call score_matrix(M, leifer_gt). Verify load_leifer's orientation matches [post,pre] before trusting AUROC (transpose is silent). If the functional connectome is signed/continuous, decide binarize_gt (default True thresholds >0); |coupling| on the method side is already sign-agnostic. Requires the wormneuroatlas package/env that load_leifer depends on.
- **2 gating_global_mode**: twohead.py:155-169 and mdn.py:133-144 readout loops. Currently gmu/glv are averaged over ALL eval histories via .mean(0) (unconditional gain). To CONDITION on the global brain-state mode: compute a global-state scalar per pooled frame (e.g. leading PC of Hn), select the eval subsample `ev` (twohead.py:157 / mdn.py:135-136) stratified by global-mode phase/quantile, and compute gain_cpl PER STRATUM, then test whether receptor-expressing edges' gain varies across strata more than variance-matched controls. Alternative (more invasive): append the global-mode value as an extra trunk input column in _pool (twohead.py:88 / mdn.py:93) and read the cross-derivative d(gain)/d(mode). The stratified-readout route needs no retraining.
- **3 self_gain_diagonal**: DIRECT: the self-gain is computed then thrown away. twohead.py:169 sets gain_cpl[j,:]=glv.mean(0)/h_sd (the length-N gradient includes index j = d log v_j / d h_j), and twohead.py:170 `np.fill_diagonal(gain_cpl, 0.0)` erases it. Same at mdn.py:144-145. To expose: capture diag_gain[j] = (glv.mean(0)/h_sd)[j] inside the loop BEFORE fill_diagonal, return it on the result (add a field to TwoHeadResult/MDNResult), and test state-dependent self-gain on receptor-expressing vs control neurons. Also note evaluate._offdiag/score_matrix drop the diagonal, so a separate 1-D scorer is needed for the diagonal channel.
- **4 class_aggregation**: Aggregation is UPSTREAM of scoring; evaluate.score_matrix needs no change but requires a SQUARE [C,C] input. Add a pooling helper that maps neuron_names -> class labels and averages coupling within class blocks (preserving [post,pre]: block-mean over post-rows and pre-cols) into [C,C], build the class-level gt [C,C] in [post,pre], then score_matrix(class_M, class_gt). Feed the aggregated matrix through stability.split_half_stability too (its fit_matrix closure can wrap fit_*+pooling to return [C,C]). Keep diagonal zeroed; _offdiag assumes square.
- **5 predictive_likelihood**: Two surfaces in-cluster. (a) mdn.MDN.nll (mdn.py:58) is the ready-made per-frame predictive NLL — but it averages over targets; change `-logp.mean()` to `-logp.mean(0)` to get PER-TARGET NLL so you can localize the gain-channel likelihood gain to receptor-expressing vs matched-control neurons. (b) fit_mdn/fit_two_head DISCARD the model, so add a held-out worm/time split and return the trained model (or expose h_mu,h_sd + a scorer) to evaluate NLL on held-out (h,y). twohead has no NLL method — compute the Gaussian conditional log-lik from model.conditional(h) using v_true=exp(logv)-sigma_c**2. NOTE the existing closed-form analog: estimator.fit_distributional_connectome (estimator.py:118-165) already produces a per-target held-out `nll[j]` via QuadraticScoreMatcher.nll on a contiguous 0.2 held-out split — reuse that pattern/localize it to the receptor set for the ablation (gain channel on vs off).
- **6 frozen_features**: Train via fit_two_head, then use TwoHead.trunk (twohead.py:54, N->hidden Sequential) as the frozen embedding. Blocker: fit_two_head returns only matrices/history — edit it (twohead.py:171) to also return `model`, `h_mu`, `h_sd`. Then freeze (model.trunk.requires_grad_(False)), compute Z = model.trunk(torch.tensor((Hn-h_mu)/h_sd)) for pooled frames, and feed Z as the source design into the closed-form quadratic-score estimator (sid_neuromod.models.quadratic_score.QuadraticScoreMatcher, used by estimator.fit_distributional_connectome) — i.e. a linear probe / closed-form mean+gain readout in frozen-feature space. Apply the SAME standardization the trunk saw. Compare against the end-to-end autograd readout.
- **7 hierarchical_shrinkage**: Primarily an acmma.py (other cluster) change to the per-edge moments. In-cluster impact is only downstream: the shrunken [post,pre] matrix is scored unchanged by evaluate.score_matrix, and split_half_stability can quantify the variance-reduction benefit of shrinkage (shrunk fits should show higher spearman_mean/jaccard_mean). No signature change needed in my files; just confirm the shrunk matrix stays [post,pre], diag-zeroed, index-aligned.
- **8 noise_ladder**: twohead.py conditional-DSM block (lines 132-139) uses ONE scalar sigma_c, and the readout (twohead.py:160) subtracts a single sigma_c**2. To couple a geometric ladder: sample sigma from a geometric grid per batch (or sum the conditional DSM loss over ladder rungs) so one model sees multiple noise scales, and make the readout noise-consistent (v_true subtraction must match the readout sigma). The closed-form analog already parameterizes this: estimator.fit_distributional_connectome has `sigma_frac` (estimator.py:134, sigma = sigma_frac*std(Y), readout subtracts sigma^2) — a ladder = fit across a geometric sweep of sigma_frac and combine. Does NOT apply to mdn.py (MLE, no injected noise).

## resource_notes
Compute/memory for a 16GB machine:

POOLING (biggest allocation): baselines._pool, twohead._pool, mdn._pool each np.concatenate ALL (T_w - lag/horizon) frames of ALL worms into one array of shape [M, N], M = total pooled frames. For ~tens of worms x a few thousand frames and N~100-200, M ~ 1e5, so Hn/Xf ~ 1e5*200*8B ~ 160MB each (float64), plus float32 torch copies Ht/Ft ~ 80MB each. Fine. Risk only if worm count or N grows an order of magnitude.

TWO-HEAD / MDN READOUT LOOP is the compute hotspot and the retain-graph memory risk: `for j in range(N): torch.autograd.grad(..., retain_graph=True)` does 2*N backward passes over an eval batch of n_eval=min(4000,M) samples, holding the WHOLE graph alive across all j (retain_graph=True) — activations for [n_eval x N] persist for the duration. With N~200 and hidden=128 this is OK (well under 16GB) but scales with N*hidden*n_eval; raising n_eval, hidden, or N together is the OOM path. MDN adds a factor k (k=3) in params tensors [B,N,k].

MPS: fit_two_head/fit_mdn default to device='mps' when torch.backends.mps.is_available(), else CPU. Training does epochs (300-500) x (M/batch=512) steps with per-step randperm(M) on device; the autograd.grad readout runs on MPS too — watch for occasional MPS autograd fallbacks/instability; pass device='cpu' if MPS misbehaves. Results are copied to CPU numpy.

STABILITY is a compute MULTIPLIER, not memory: split_half_stability calls the fit function 2*n_splits times (default 10). Wrapping fit_two_head/fit_mdn in it means 10 full neural trainings — run_scaling.py:49 deliberately gates two-head stability to nworm>=6 for this reason. tune_by_stability multiplies again by len(grid).

noise_ladder (idea 8): summing the conditional DSM loss over L ladder rungs multiplies per-step activation memory by ~L (each rung a fresh corrupted batch through the trunk); keep batch modest or accumulate rungs sequentially to avoid inflating the graph L-fold.

BASELINES are cheap: ridge_var inverts one NxN Gram + N target regressions; pearson_lag is one NxN matmul. Negligible.
