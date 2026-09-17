# CLUSTER: core_estimators

## conventions
CENTRAL DATA CONVENTIONS a new script must respect:

X_list: a Python list of per-worm numpy arrays, each shape [T_w, N], dtype float64, TIME x NEURONS (neurons are COLUMNS). Values are GLOBALLY z-scored (mean 0, unit sd) across the pooled dataset (combined_data.py:106-109 subtracts one global mu / divides by one global_sd). Unrecorded neurons are NaN columns (available-case) in the non-imputed path; the imputed path (data.py load_traces) has no NaN. N is fixed across worms (canonical N=80 for the OH16230/OH15500 calcium data; each column position = same neuron across all worms). Windows are built PER WORM then concatenated/pooled — never window across a worm boundary (both files loop `for X in X_list` and drop worms with T<=lag). fps=4.0 Hz is the sampling rate returned alongside (X_list, neuron_names, fps) by the loaders.

neuron_names: length-N list[str], column-aligned to every X in X_list. Ground truth is re-indexed onto this ordering via ground_truth._reindex (ground_truth.py:29). Class-level names map through ground_truth.to_class (ground_truth.py:93).

ORIENTATION — every returned coupling matrix is [post, pre] = [target, source]: mats[channel][j, i] = coupling from SOURCE neuron i to TARGET neuron j. Row = post/target, col = pre/source. This matches the Cook/monoamine/Leifer ground truth as loaded. A new script must NOT transpose before scoring against those targets. Shape [N, N], dtype float64.

CHANNELS = ("mean","gain","tail_hi","tail_lo") in BOTH files. mean = dE[Y]/dx_i (VAR-comparable predictive drive); gain = dlogVar[Y]/dx_i (excitability/neuromodulation — the headline channel); tail_hi/tail_lo = d P(Y ≷ q)/dx_i.

TARGET/FEATURE causal convention: for target j at lag L, target Y = x_j(t+L) (target_mode="next") or x_j(t+L)-x_j(t) ("delta"); features psi(t) = [1, all-neuron activity at t] INCLUDING the target's own column (multivariate/partial conditioning). Features are centered on the pooled/available-case source mean so the closed-form readout is "at the typical history" (sources=0, intercept=1).

DIAGONAL ZEROING: the self edge mats[c][j,j] is force-zeroed for ALL channels (estimator.py:174-177; acmma.py:139-140) AFTER the readout is computed — the self-gain value exists inside the readout vector but is discarded. Self-gain for target j lives at rd["gain"][j+1] (index 0 = intercept, indices 1..N = sources) before zeroing.

NaN HANDLING differs between the two estimators and is the core methodological split: estimator.py IMPUTES source NaN to 0 (the standardized mean) — _filter_sources does np.nan_to_num BEFORE filtering (estimator.py:47) and _build_target_design does np.nan_to_num on Xsrc (estimator.py:92); rows with a NaN TARGET are dropped (estimator.py:90-92). acmma.py MASKS instead: it keeps NaN sources, builds a present-mask M, centers available-case per source, sets absent cells to 0 in Z (auto-masking, acmma.py:75-76), and normalizes every co-moment entry by its OWN co-observation count Nik (acmma.py:88-93); thin-support gain entries (Nik<min_triple) are HARD-MASKED to 0 (acmma.py:98-99). Both fits consume data ONLY through moments A=U'U/T with U=[psi, 2Y*psi], theta=-(A+ridge I)^{-1} c, c=[0_P, 2*mean(psi)] — so ACMMA reproduces estimator.py exactly on complete-case data.

Targets with fewer than 3*N pooled valid samples are SKIPPED (estimator.py:149, acmma.py:70), leaving that row of every matrix as all-zeros and heldout_nll[j]=NaN — a new script scoring AUROC must handle zero rows.

## key_functions
### fit_distributional_connectome  (estimator.py)
- sig: `fit_distributional_connectome(X_list, neuron_names, lag: int, target_mode: str='next', ridge: float=1e-2, heldout_frac: float=0.2, source_tau: float|None=None, fps: float=4.0, seed: int=0, sigma_frac: float=0.0) -> ConnectomeResult`
- returns: ConnectomeResult dataclass. .matrices: dict channel->[N,N] float64, [post,pre] (mats[c][j,i]=source i -> target j); diagonal zeroed. .invalid_fraction: [N] float64 (per-target fraction of TRAIN rows with non-normalizable curvature eta2>=-eta2_min; 0.0 for skipped targets). .heldout_nll: [N] float64 init NaN (per-target held-out mean NLL; NaN if target skipped or nll raised). .lag int, .neuron_names list.
- purpose: Main closed-form quadratic-score estimator: fits sid_neuromod QuadraticScoreMatcher once per target on IMPUTED (NaN->0) pooled features, reads out the 4-channel directed distributional connectome at a single lag. The gain channel is the headline neuromodulation readout.
- gotchas: Imputes source NaN to 0 (mean) — NOT available-case. Contiguous train/test split (first (1-heldout_frac) rows train, rest test) over the CONCATENATED pooled array, so held-out tail is the last worm(s), not per-worm balanced (matters for predictive_likelihood). q_hi/q_lo computed on TRAIN Y only. Diagonal force-zeroed for all channels at :174-177 AFTER readout. sigma_frac>0 = denoising score matching with sigma=sigma_frac*std(Y[:ntr]); denoising draws ACTUAL Gaussian noise via seeded rng inside QuadraticScoreMatcher.fit, so results are STOCHASTIC/seed-dependent (unlike acmma). source_tau (in FRAMES) switches sources to causal exp-filtered slow drive. No global state, no in-place mutation of X_list.

### _build_target_design  (estimator.py)
- sig: `_build_target_design(X_list, j, lag, target_mode, source_tau=None, fps=4.0) -> (Y, Xsrc)`
- returns: Y: [R] float64 target future value/increment (NaN-target rows dropped). Xsrc: [R, N] float64 source features at feature time, NaN->0 imputed. R = pooled valid rows across worms.
- purpose: Assembles the pooled (target, sources) design for one target j at one lag. THE integration point for feature-space changes (frozen_features, gating global-mode as modulator).
- gotchas: When source_tau given, _filter_sources np.nan_to_num's the WHOLE X to 0 before exp-filtering (mid-signal gaps become 0 drive). x_fut/x_now_raw use RAW X (target with NaN); Xsrc imputed to 0. Includes target's OWN column in Xsrc (self column later zeroed in matrix).

### _center_readouts  (estimator.py)
- sig: `_center_readouts(res: FitResult, q_hi, q_lo) -> dict`
- returns: dict channel -> length-P (P=N+1) float64 vector over feature columns; index 0 = intercept, indices 1..N = sources. mats[c][j,:] is later set to rd[c][1:].
- purpose: Vectorized center-history (sources=0, intercept=1) readout of the 4 distributional channels from fitted theta1/theta2. Where self-gain would be exposed: rd['gain'][j+1].
- gotchas: eta2 clamped to min(theta2[0], -res.eta2_min); v = max(v_eff - sigma^2, res.var_min) — gain (d_var/v) SATURATES when variance hits the var_min floor. Reads res.eta2_min/res.var_min/res.sigma off the FitResult (defaults 1e-6). d_gain = 2*v_eff^2*theta2 / v.

### _filter_sources  (estimator.py)
- sig: `_filter_sources(X, tau_frames, fps=4.0) -> Xf[T,N]`
- returns: [T, N] float64 causal exp-filtered slow drive of every neuron column.
- purpose: Slow integrated-drive feature transform used when source_tau is set.
- gotchas: np.nan_to_num(X,0.0) BEFORE filtering; tau_s = tau_frames/fps (tau passed in FRAMES). Wraps sid_neuromod exp_filter_bank with a single timescale, init='zero'.

### ConnectomeResult  (estimator.py)
- sig: `@dataclass ConnectomeResult(lag:int, neuron_names:list, matrices:dict, invalid_fraction:np.ndarray, heldout_nll:np.ndarray)`
- returns: container; matrices[channel] is [N,N] [post,pre].
- purpose: Return type of fit_distributional_connectome; the object new scoring scripts consume.
- gotchas: matrices dict is freshly allocated per call (no shared/global state).

### fit_acmma_connectome  (acmma.py)
- sig: `fit_acmma_connectome(X_list, names, lag, ridge=1.0, min_triple=8, shrink=0.05, eta2_min=1e-6, var_min=1e-6, q_hi=0.90, q_lo=0.10, sigma_frac=0.0) -> (mats, diag)`
- returns: mats: dict channel->[N,N] float64 [post,pre], diagonal zeroed, non-finite readout entries replaced by 0 (acmma.py:137). diag: dict {'clipped_eigen_mass_median','clipped_eigen_mass_max','masked_gain_frac_median'} (floats over targets). NO ConnectomeResult wrapper, NO neuron_names, NO heldout_nll — different return shape than estimator.py.
- purpose: Available-Case Multivariate Moment Assembly: same closed-form fit but assembles each co-moment entry available-case (masking, zero imputation) so all worms are usable without fabricating missing (Y,x_i) cells. Reproduces estimator.py on complete-case data.
- gotchas: Different signature/return than estimator.py (ridge default 1.0 vs 1e-2; higher ridge because moment matrix is noisier). sigma_frac>0 is DETERMINISTIC/exact: no noise draw — it adds 4*sigma^2*A11 to the Y^2 block (acmma.py:112-118), so unlike estimator.py it is seed-free/reproducible. HARD-MASK: gain-bearing S1/S2 entries with co-observation Nik<min_triple set to 0 (acmma.py:98-99) — a de-scoped shrink-to-class was intentionally removed (see module docstring). _nearest_psd shrinks+eigen-clips the assembled A (Ledoit-Wolf-style). Per-target O(P^3) eigh.

### _pool_target  (acmma.py)
- sig: `_pool_target(X_list, j, lag) -> (Y, Xsrc) or (None, None)`
- returns: Y: [R] float64; Xsrc: [R,N] float64 with NaN sources KEPT (for masking). None,None if no worm records target j.
- purpose: Available-case pooling analogue of estimator._build_target_design (does NOT impute sources).
- gotchas: Only drops rows with NaN TARGET (keep=isfinite(y)); source NaN retained. Returns raw activity (no source_tau path — ACMMA has no slow-filter option).

### _nearest_psd  (acmma.py)
- sig: `_nearest_psd(A, shrink=0.05, floor=1e-6) -> (A_psd, clipped_mass_fraction)`
- returns: A_psd: [2P,2P] float64 symmetric PSD; clipped_mass_fraction: float.
- purpose: Ledoit-Wolf shrinkage toward scaled identity then eigenvalue clip; the population-prior shrinkage lever for hierarchical_shrinkage lives at this stage (currently shrinks the MOMENT MATRIX toward identity, not per-edge toward a class prior).
- gotchas: Symmetrizes A first; shrinks toward (trace/2P)*I; clips eigenvalues below floor. Diagnostic clipped mass = |neg eigen mass|/|total|.

### QuadraticScoreMatcher.fit  (quadratic_score.py)
- sig: `QuadraticScoreMatcher(sigma=0.0, ridge=1e-6, eta2_min=1e-6, var_min=1e-6, seed=0).fit(Y[T], Psi[T,P], sigma=None, ridge=None) -> FitResult`
- returns: FitResult(theta[2P], P, sigma, ridge, A[2P,2P]=U'U/T, xi[T,2P] per-sample estimating fns, eta2_min, var_min, invalid_variance_fraction, n_samples, sd_y). Properties .theta1=theta[:P], .theta2=theta[P:].
- purpose: The Algorithm-1 closed-form solve underneath estimator.py. Call directly to run the estimator in a custom feature space (frozen_features) or with a custom Psi (gating modulator, class-aggregated features).
- gotchas: sigma>0 draws seeded Gaussian noise (get_rng(self.seed)) — STOCHASTIC. sigma=0 is exact Hyvarinen (c=[0_P, 2*mean(Psi)]). Requires Y length == Psi rows; Psi must be 2D. invalid_variance_fraction computed on TRAIN features. Stores result_ on the instance (mutates self).

### QuadraticScoreMatcher.nll / logpdf / predict_params  (quadratic_score.py)
- sig: `nll(Y[T], Psi[T,P])->float ; logpdf(Y,Psi)->[T] ; predict_params(Psi)->(eta1,eta2,mu,var)`
- returns: nll: scalar mean NLL. logpdf: [T] float64 Gaussian log density. predict_params: 4x[T] arrays with prediction-time clamps (eta2<=-eta2_min, var>=var_min).
- purpose: Held-out scoring surface for predictive_likelihood (idea 5) — estimator.py already calls model.nll(Y[ntr:],Psi[ntr:]) at estimator.py:165.
- gotchas: predict_params emits RuntimeWarnings when >1%/>5% of rows have invalid eta2. To ablate the gain channel for a likelihood-improvement test, construct a restricted FitResult with theta2 source-components zeroed (keep theta2[0]) or fit a homoscedastic Gaussian, then call nll — the estimator couples mean and variance in one design so there is no built-in gain-off switch.

### exp_filter_bank  (filter_bank.py)
- sig: `exp_filter_bank(z[T]|[T,J], timestamps_s[T], timescales_s(len K), *, init='zero') -> F[T,J,K]`
- returns: [T, J, K] float64, strictly causal (F[k] uses z[0..k]).
- purpose: Exact ZOH causal exponential filter used by _filter_sources for slow-drive features / lag-resolved timescales.
- gotchas: timescales in SECONDS here (estimator converts frames->s). Requires strictly increasing timestamps (raises otherwise). init='zero' warms from 0.

### ridge_solve  (linalg.py)
- sig: `ridge_solve(A, b, ridge=0.0) -> x`
- returns: x solving (A+ridge*I)x=b; lstsq fallback if singular.
- purpose: The regularized linear solve at the heart of both fits (acmma.py:123, quadratic_score.py:123/131).
- gotchas: Adds ridge*I to a COPY (no mutation). Both estimators negate the solution: theta=-ridge_solve(...).

### sigma_ledger  (quadratic_score.py)
- sig: `sigma_ledger(Y, Psi, ridge=1e-6, sigma_fracs=(0.0,0.25,0.5,1.0), seed=0) -> list[dict]`
- returns: list of {sigma_frac, sigma, mean_at_center, var_at_center} across the sigma grid.
- purpose: Existing per-target denoising-ladder DIAGNOSTIC (cross-sigma consistency of the center readout) — the seed pattern to imitate/extend for noise_ladder (idea 8).
- gotchas: Only returns center-history mu/var, NOT connectome matrices; refits independently per sigma; does not COMBINE sigmas into one estimator.

## integration_hooks
- **functional_target (score vs Randi/Leifer functional connectome)**: Does NOT touch estimator/acmma internals — they already emit mats[channel] as [N,N] [post,pre] float64 aligned to neuron_names. Load the functional target via the EXISTING ground_truth.load_leifer(neuron_names, alpha=0.05) (/Users/vik/Developer/new_sbtg_neuro/sid_elegans/ground_truth.py:133) — it already re-indexes onto neuron_names via _reindex (ground_truth.py:29) in [post,pre]. If pulling fresh from wormneuroatlas, re-index with ground_truth._reindex(A, src_names, neuron_names) and confirm Randi is stored [post,pre] (their raw matrix is often [pre,post] — transpose to match). Then AUROC mats['gain']/mats['mean'] vs the aligned functional matrix using the existing evaluate.py scorer. Skipped-target rows are all-zero; mask them out of scoring.
- **gating_global_mode (condition on / modulate by global brain-state mode)**: Extend feature assembly in estimator._build_target_design (estimator.py:76-92) or the Psi construction at estimator.py:154. Compute the global mode g(t) (e.g. first PC of X, or the state phase) per worm, then augment features: to CONDITION, append g(t) as an extra source column; to test MODULATION of gain coupling, append interaction columns x_i(t)*g(t). The gain channel d_gain=dlogVar/dx_i then carries a phase-interaction term. Because _center_readouts assumes col0=intercept and cols1..N=sources, adding columns changes P and the [1:] slice at estimator.py:174 — you must re-map interaction columns back to (source i) rather than treating them as extra neurons. Cleanest: call QuadraticScoreMatcher.fit directly on your custom Psi=[1, x, g, x*g] per target and read theta2 phase-interaction coefficients yourself. Variance-matched controls: reuse the same Psi-augmentation on permuted/control neuron sets.
- **self_gain_diagonal (expose zeroed self-excitability gain)**: The value already exists: in estimator._center_readouts the self-gain for target j is rd['gain'][j+1] (index0=intercept). It is discarded by the diagonal-zeroing at estimator.py:174-177 (and acmma.py:139-140). Add a self_gain output BEFORE zeroing: capture self_gain[j]=rd['gain'][1:][j] (equivalently rd[c][j+1]) inside the loop at estimator.py:173-174, then add a self_gain field to ConnectomeResult (estimator.py:53-59). Same for acmma at acmma.py:136-140 (capture row[j] before mats[cn][j,j]=0.0). GOTCHA: self-gain saturates when v hits var_min (estimator.py:108) — flag targets with invalid_fraction high. For state-dependent self-gain, combine with the gating hook (interact the self column with global phase).
- **class_aggregation (aggregate estimator to neuron classes)**: Two clean options. (A) POST-hoc: build a class-membership map with ground_truth.to_class (ground_truth.py:93), then average the OUTPUT mats[c][post,pre] within (post_class,pre_class) blocks to a [C,C] matrix — cheapest, no re-fit, but averages couplings not moments. (B) PRE-aggregate: collapse X_list columns to class means (per worm, average neuron columns sharing a class) BEFORE calling fit_distributional_connectome, giving class-level features/targets and more samples per target (the len(Y)<3*N skip guard at estimator.py:149 becomes 3*C, far easier to satisfy — the power win). For available-case data prefer aggregating inside acmma's moment sums (acmma.py:84-93) so co-observation counts are pooled across a class. Match to class-level ground truth already produced by build_monoamine_layer/to_class.
- **predictive_likelihood (held-out one-step likelihood improvement from gain channel)**: estimator.py already computes per-target held-out NLL (model.nll(Y[ntr:],Psi[ntr:]) at estimator.py:165 -> ConnectomeResult.heldout_nll). For a GAIN-localized improvement, fit the full model then a gain-ablated baseline and diff their held-out NLL: ablate by building a FitResult copy with theta2 source components zeroed (keep theta2[0], the homoscedastic intercept) and calling QuadraticScoreMatcher.logpdf/nll (quadratic_score.py:185-192) on the SAME held-out rows. Localize by computing the delta-NLL only for receptor-expressing target neurons vs variance-matched controls. GOTCHA: the held-out split is CONTIGUOUS over the pooled concatenation (estimator.py:157) so the test tail = last worm(s); switch to a per-worm or interleaved split for an honest one-step measure. Skipped targets have NaN nll.
- **frozen_features (linear probe of closed-form estimator on frozen two-head trunk)**: Train+freeze the trunk via fit_two_head (twohead.py:100, returns TwoHeadResult) — reuse its _slow_filter/_pool history builder (twohead.py:78-97) to get frozen features Phi(t)[T,D] per worm. Then bypass fit_distributional_connectome and call QuadraticScoreMatcher.fit(Y_j, Psi=[1, Phi]) per target (quadratic_score.py:102) and read theta2 with the SAME _center_readouts math (estimator.py:96-119). GOTCHA: in feature space the readout columns are FEATURES not neurons, so mats is [target, feature] — to recover a [post,pre] neuron connectome you must project feature-gradients back through the trunk Jacobian dPhi/dx_i (chain rule), otherwise you cannot align to neuron-level ground truth. Simplest first cut: keep Phi as an augmentation of the raw source columns so the first N readout entries remain neuron-indexed.
- **hierarchical_shrinkage (empirical-Bayes/James-Stein of ACMMA per-edge moments toward class prior)**: Extend fit_acmma_connectome. Currently thin entries are HARD-MASKED to 0 (acmma.py:98-99) and only the assembled matrix is shrunk toward identity (_nearest_psd, acmma.py:44-52) — this re-introduces the intentionally de-scoped shrink-to-class (see module docstring). Implement as a TWO-PASS over the target loop (acmma.py:68): pass 1 accumulate per-target normalized moments s1/s2 (acmma.py:91) and their co-observation counts Nik; compute a population/class prior (mean s1/s2 within each pre/post class via ground_truth.to_class); pass 2 shrink each entry s1_ij <- (Nik_ij*s1_ij + kappa*prior)/(Nik_ij+kappa) with kappa an EB/James-Stein weight ~ inversely proportional to per-edge sample count, then assemble A and solve (acmma.py:102-123). Replace the min_triple hard mask with this shrink for low-Nik entries. Keep readout byte-identical (acmma.py:126-138).
- **noise_ladder (couple a geometric sigma_frac ladder into one estimator)**: Both estimators take a single sigma_frac (estimator.py:132 / acmma.py:56); sigma_ledger (quadratic_score.py:216) already runs a grid but only for a diagnostic center readout. Prefer building on ACMMA's DETERMINISTIC denoising (acmma.py:112-118: A22 += 4*sigma^2*A11) rather than estimator.py's stochastic noise draw (quadratic_score.py:116-119) so ladder rungs are reproducible and directly comparable. To COUPLE rungs into one fit: for a geometric sigma grid, stack the per-rung Y^2-block corrections into a single joint moment problem, or fit each rung and pool readouts with inverse-variance weights. Concrete hook: wrap fit_acmma_connectome over sigma_fracs=geomspace(...) reusing the assembled A11/A22 (assemble the base moments ONCE per target at acmma.py:103-111, then add 4*sigma^2*A11 per rung and re-solve — cheap, avoids re-pooling). The var subtraction v=v_eff-sigma^2 (acmma.py:127) already makes rungs' gain readouts commensurable.

## resource_notes
Both estimators are light on a 16GB machine for the canonical N=80, ~few-dozen worms dataset. Per target the design is U=[Psi,2Y*Psi] of shape [R, 2P] with P=N+1; A=U'U/T is [2P,2P] (~[162,162] at N=80) — trivial. The whole fit is a length-N loop of one ridge_solve each (estimator) or one eigh + one solve each (acmma). ACMMA is the heavier of the two: per target it forms Nik/S0/S1/S2 as [N,N] and does an eigh on the [2P,2P] block (O(P^3)); total ~O(N * P^3). At N=80 this is milliseconds-to-seconds; at N~300 (full C. elegans) it is ~O(300*600^3) ≈ tens of seconds and still <1GB. No single array is large.

Real OOM risks come from what NEW scripts stack around these calls, not the calls themselves: (1) a lag x sigma_frac x seed x channel sweep that KEEPS every [N,N] ConnectomeResult in memory — at N=300, one connectome is 4 channels * 300*300*8B ≈ 2.9MB, fine individually but a 20-lag x 5-sigma x 100-permutation null grid is ~30GB if all retained; write to disk / reduce to scalars per rung instead. (2) frozen_features with a wide trunk: Psi=[1,Phi] of feature dim D makes A [2(D+1),2(D+1)] and eigh O(D^3) — keep D modest (a few hundred) or the per-target eigh dominates. (3) functional_target via wormneuroatlas can pull large atlas objects; load once and cache. (4) hierarchical_shrinkage two-pass must hold per-target S1/S2 [N,N] for all N targets simultaneously = N*[N,N] = O(N^3) floats (at N=300 ~216M floats ≈ 1.7GB) — stream/aggregate into the class-prior accumulators instead of materializing all targets. No MPS/GPU is used by estimator.py or acmma.py (pure numpy/scipy); only twohead.py (torch) would touch MPS for frozen_features."]}
