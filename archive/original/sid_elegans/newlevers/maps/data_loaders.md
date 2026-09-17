# CLUSTER: data_loaders

## conventions
CENTRAL DATA OBJECT: every loader returns the triple (X_list, neuron_names, fps). X_list is a PYTHON LIST of per-worm np.ndarray of shape [T_w, N] (rows=timepoints, cols=neurons), dtype float64. It is NOT a stacked 3D array: T_w varies per worm (combined) or is ~925-929 (canonical). fps is always 4.0 (float). Columns are aligned across worms to neuron_names.

NEURON-NAME / INDEX CONVENTION: neuron_names (called `nodes` in combined_data.load_combined) is a sorted() alphabetical list of UPPERCASE strings. Names are already reduced to the C. elegans CLASS level by two collapses done inside the loaders: (1) Left/Right averaging and (2) D/V-subtype collapse (RMDD/RMDV->RMD) via collapse_dv_subtypes. So a "neuron" here is really a bilateral+DV class. Any ground-truth matrix (connectome/receptor/functional) MUST be re-indexed to this exact sorted-uppercase order; index i in a worm matrix column == neuron_names[i].

[post,pre] vs [pre,post]: NONE of these three files load a connectome WEIGHT matrix. combined_data reads ONLY the node NAME list (connectome/nodes.json, 94 names, line 61) to filter which neurons to keep. Therefore no orientation is set in this cluster. The known Cook [pre,post] transpose bug lives entirely downstream in the target/eval code; what this cluster fixes is the INDEX SPACE (neuron_names order) that any [post,pre] or [pre,post] target must be aligned to.

STANDARDIZATION (scale-preserving, combined_data lines 106-109): per-neuron CENTERING (subtract per-column mu = np.nanmean over all worms concatenated) but division by a SINGLE global scalar global_sd = np.nanstd(allX-mu)+1e-8 (NOT per-neuron). This deliberately preserves relative variance magnitudes across neurons (the gain channel depends on them; per-neuron z-scoring was verified to collapse the tyramine result 0.73->0.35). New scripts must NOT re-z-score per neuron. data.py's canonical datasets are already standardized by the repo pipeline.

DECONVOLUTION: optional continuous-activity signal, applied per-column BEFORE standardization (combined_data line 96-98) when signal='deconv'. Produces a non-negative firing-rate proxy s via OASIS AR(1). fps=4.0 is hardcoded at the call site.

NaN / MISSINGNESS (available-case): complete_case=True (default) => zero NaNs; a neuron is kept only if covered in >= coverage_frac*total_worms worms AND in connectome, then only worms having EVERY selected neuron are kept. complete_case=False => neuron kept if in >= min_worms_per_neuron worms; ALL worms kept and missing (worm,neuron) cells are FULL-NaN columns (np.full(minlen, nan)); downstream estimator mean-imputes per feature (no cross-worm leakage). Because standardization uses nanmean/nanstd, NaNs survive as NaNs in the returned X_list. Deconvolution is NaN-safe (leaves NaN where a column has <10 finite samples).

TIME ORDERING: worm matrices are head-aligned (t=0 at recording start) and trimmed to the shortest neuron trace within that worm (_worm_matrix line 46). Rows are consecutive 4 Hz frames, so one-step temporal models (predictive likelihood, AR) can use row t -> t+1 directly.

COUNTS: combined_data pools OH16230 head+tail (21 worms) + OH15500 head (7) = up to 28 worms; N is DATA-DEPENDENT (varies with coverage_frac/complete_case, NOT fixed). data.load_traces('full_traces_imputed') is the pre-baked canonical set: 20 worms, T in 925-929, N=80 fixed, float64, no NaN.

TWO DISTINCT ENTRY POINTS: data.load_traces = fast, reads a pre-computed .npy the repo pipeline already built (canonical 80-neuron OH16230). combined_data.load_combined = builds fresh from the raw .mat files at call time, pools both strains, lets you choose coverage/imputation/deconv. They are not interchangeable (different worms, different N, different provenance)."

## key_functions
### load_combined  (combined_data.py)
- sig: `load_combined(coverage_frac=0.6, complete_case=True, min_worms_per_neuron=6, signal='raw', verbose=True)`
- returns: (X_list, nodes, fps): X_list list of [T_w,N] float64 (NaN cols only if complete_case=False), nodes sorted uppercase class names (N data-dependent), fps=4.0
- purpose: Fresh pooled loader (OH16230 head+tail + OH15500 head), connectome+coverage neuron selection, optional deconv, scale-preserving standardization.
- gotchas: Module import execs SBTG/pipeline/01_prepare_data.py (side effects, prints); needs SBTG on PYTHONPATH; reloads 181MB of .mat each call, no cache; not in-place; signal in {'raw','deconv'} else ValueError; NaNs propagate via nanmean/nanstd.

### load_traces  (data.py)
- sig: `load_traces(dataset='full_traces_imputed')`
- returns: (X_list, names, fps): X_list list of [T_w,80] float64 (20 worms, T 925-929, no NaN in imputed variant), names len-80 uppercase, fps=4.0
- purpose: Fast loader of the repo's pre-baked already-standardized canonical dataset (.npy).
- gotchas: N fixed at 80, different worm/neuron set than load_combined; already standardized (don't re-z-score); dataset dir choices: full_traces, full_traces_imputed, butanone, nacl, pentanedione.

### deconvolve_matrix  (deconv.py)
- sig: `deconvolve_matrix(X, fps=4.0)`
- returns: np.ndarray same shape [T,N] float64; per-column non-negative activity s; NaN where a column has <10 finite samples
- purpose: Column-wise OASIS AR(1) deconvolution (the signal='deconv' path).
- gotchas: New array not in-place; pure-python loops; assumes neurons on axis=1; per-column gamma; NaN-safe per column.

### deconvolve_trace  (deconv.py)
- sig: `deconvolve_trace(y, fps=4.0, baseline_pct=10.0)`
- returns: np.ndarray len T; NaN where input non-finite or <10 finite; else non-negative activity with 10th-pct baseline removed
- purpose: Single-trace NaN-safe deconvolution unit.
- gotchas: Subtracts per-trace percentile baseline (absolute offset discarded); writes back only at finite positions.

### oasis_ar1  (deconv.py)
- sig: `oasis_ar1(y, gamma)`
- returns: (c, s): c denoised calcium len T, s=max(c_t-gamma*c_{t-1},0) len T, both float64
- purpose: Native OASIS active-set AR(1) non-negative deconvolution.
- gotchas: No NaN handling (caller strips); deterministic O(T).

### estimate_gamma  (deconv.py)
- sig: `estimate_gamma(y, fps=4.0, lo=0.5, hi=0.98)`
- returns: float in [lo,hi]
- purpose: Estimate AR(1) calcium decay from lag1/lag0 autocovariance, clipped.
- gotchas: Returns 0.85 fallback if <10 finite samples or nonpositive variance; fps unused.

### _worm_matrix  (combined_data.py)
- sig: `_worm_matrix(data, worm_idx, nodes, complete_case=True)`
- returns: np.ndarray [minlen, len(nodes)] float64 or None; columns ordered as nodes; missing neuron -> None (complete) or full-NaN column
- purpose: Assemble one worm's [T,N] matrix, head-aligned, trimmed to shortest column.
- gotchas: Internal helper; per-worm T from within-worm min length; relies on data['name_to_indices'].

### collect_worm_trace  (01_prepare_data.py)
- sig: `collect_worm_trace(all_traces, neuron_indices, worm_idx, num_worms) -> np.ndarray | None`
- returns: 1-D averaged trace (L/R + DV variants, mean over variants, tail-aligned) or None if absent
- purpose: Underlying per-(neuron,worm) trace extractor combined_data calls as _prep.collect_worm_trace.
- gotchas: L at offset=worm_idx, R at offset=worm_idx+num_worms; None means missing; tail-aligns (seg[-min_len:]) internally.

### load_neuropal_data  (01_prepare_data.py)
- sig: `load_neuropal_data(data_dir, include_tail=False, collapse_dv=False) -> Dict`
- returns: dict: neuron_names, norm_traces, fps, stim_names, stim_times, stims_per_worm, worm_ids, name_to_indices, neuron_names_original, dv_collapsed
- purpose: Repo loader combined_data uses for OH16230 (include_tail=True, collapse_dv=True).
- gotchas: Prints; FileNotFoundError if .mat missing (part reassembly); exposes stim_times/stim_names that combined_data drops.

### get_neuron_type  (neuron_types.py)
- sig: `get_neuron_type(neuron_name) -> str`
- returns: 'sensory'|'interneuron'|'motor'|'unknown'
- purpose: Class->functional-category map for class_aggregation / hierarchical_shrinkage grouping (companions: get_neurons_by_type, NEURON_TYPE_MAP).
- gotchas: Not imported by cluster files; import from pipeline.utils.neuron_types (needs SBTG on path); expects collapsed uppercase class name.

### collapse_dv_subtypes  (align.py)
- sig: `collapse_dv_subtypes(name: str) -> str`
- returns: parent class name (RMDD->RMD) or unchanged
- purpose: The D/V-collapse rule that defines the class-level naming both loaders use; reuse when mapping external target IDs (e.g. wormneuroatlas) into load_combined's index space.
- gotchas: Backed by DV_COLLAPSE_PATTERNS dict; normalize_neuron_name applied first.

## integration_hooks
- **1 functional_target (Randi/Leifer functional connectome via wormneuroatlas)**: load_combined does NOT load any weight matrix — it only pulls connectome NODE NAMES at combined_data.py:61 (json.load(open(CONN/'nodes.json'))). So the data-loader touchpoint is purely the returned `nodes` (neuron_names): a new functional-target loader must build a [len(nodes) x len(nodes)] matrix and reindex it to this sorted-uppercase collapsed-class order. wormneuroatlas uses per-neuron (often L/R-split) IDs, so map its IDs through pipeline.utils.align.collapse_dv_subtypes + uppercase + L/R-average to match load_combined's class names before reindexing. Existing scaffold to mirror: SBTG/pipeline/utils/leifer.py. Keep load_combined unchanged; add the target loader alongside and align on `nodes`.
- **2 gating_global_mode (condition on global brain-state mode instead of removing it)**: The global mode is not computed in this cluster; the only global operation is per-neuron centering (mu = np.nanmean, combined_data.py:107) and a single global_sd (line 108). Hook: call load_combined(complete_case=True) to get NaN-free X_list, then compute the global-state modulator as the leading PC / row-mean of np.concatenate(X_list, axis=0) (already mean-centered per neuron, so PC1 is the shared mode). Do this per worm on x (each [T_w,N]) so the modulator is time-aligned to the gain channel. If you need it BEFORE standardization, replicate lines 106-109 or add a small helper; do not per-neuron z-score. Deconvolve first (signal='deconv') if you want the mode in activity space.
- **3 self_gain_diagonal (expose zeroed self-excitability diagonal)**: No diagonal handling exists in the data_loaders cluster — the diagonal zeroing is in the downstream ACMMA/estimator code, not these files. The only relevant contribution here: load_combined guarantees a consistent neuron index i (neuron_names[i]) so a diagonal (i,i) self-gain term is well-defined per neuron; use get_neuron_type / a receptor-expression list keyed on `nodes` to select which diagonal entries to test. No change needed in combined_data.py/deconv.py/data.py.
- **4 class_aggregation (aggregate estimator to neuron CLASSES)**: Two aggregation levels are available. (a) The names load_combined returns are ALREADY L/R + D/V collapsed classes (via load_neuropal_data(collapse_dv=True) at combined_data.py:59 and _load15 at combined_data.py:125-129), so bilateral/DV aggregation is done for you. (b) For further grouping to functional CLASS matching class-level ground truth, map each name in `nodes` through pipeline.utils.neuron_types.get_neuron_type (neuron_types.py:70) or build a custom class map. Aggregate the estimator by grouping columns of each X_list matrix (or the per-edge estimates) by the class label of neuron_names[i]. To customize which raw variants merge, edit the DV_COLLAPSE_PATTERNS behind align.collapse_dv_subtypes (align.py:116).
- **5 predictive_likelihood (held-out one-step likelihood from gain channel)**: X_list rows are consecutive 4 Hz frames, head-aligned (t=0 at start, _worm_matrix line 46), so a one-step model uses x[t] -> x[t+1] directly per worm. Use load_combined(complete_case=True) to avoid NaN gaps in the target (complete_case=False leaves full-NaN columns that break one-step prediction on those neurons). For a firing-rate proxy better suited to one-step dynamics, pass signal='deconv'. Localize to receptor vs control neurons by indexing columns via neuron_names. Hold out worms (X_list is per-worm) for the held-out split.
- **6 frozen_features (train trunk self-supervised, freeze, linear probe)**: Pure consumer of the standardized X_list; no loader change. Feed load_combined/load_traces output as the trunk's input. Respect the scale-preserving standardization (do not per-neuron z-score before the trunk). Use load_traces('full_traces_imputed') for the fixed 80-neuron canonical set if you want a stable feature dimension across runs; use load_combined for the pooled multi-strain set (variable N).
- **7 hierarchical_shrinkage (empirical-Bayes shrinkage toward class prior)**: The class grouping needed to define the population/class prior comes from this cluster's name conventions: group per-edge/per-neuron moments by get_neuron_type(neuron_names[i]) (neuron_types.py) or by the collapsed class name itself. neuron_names from load_combined is the exact index space to attach shrinkage groups to. No change to combined_data.py/deconv.py/data.py; just consume `nodes` + the type map.
- **8 noise_ladder (geometric ladder of denoising sigma scales)**: The current denoising stage is calcium deconvolution (deconv.py), applied once at combined_data.py:96-98. deconv exposes no sigma_frac knob today; the tunable denoising params are baseline_pct (deconvolve_trace, deconv.py:68) and gamma (estimate_gamma clip lo/hi, deconv.py:20). If the noise ladder belongs at the deconvolution stage, produce multiple deconvolved copies by calling deconvolve_matrix with a swept param and stacking; note load_combined only accepts signal in {'raw','deconv'} (combined_data.py:96-100), so you'd extend that branch or bypass it and call sid_elegans.deconv.deconvolve_matrix directly on raw X_list. NOTE: sigma_frac-style Gaussian denoising is more likely a filter-bank/estimator parameter than a deconv parameter — confirm where the estimator's noise scale lives before threading it here.

## resource_notes
Safe on a 16GB machine. Disk footprint of the raw inputs: Head_Activity_OH16230.mat 118MB + Head_Activity_OH15500.mat 37MB + Tail_Activity_OH16230.mat 26MB. load_combined loads ALL of these via scipy.io.loadmat(simplify_cells=True) on EVERY call with no caching; simplify_cells expands the nested cell arrays, so transient peak memory is a few hundred MB to ~1-2GB, but nowhere near 16GB. The returned X_list is tiny (<=28 worms x ~925 frames x <=94 neurons float64 ~= 20MB; the canonical load_traces set is 20x~928x80 ~= 12MB). Deconvolution (signal='deconv') is CPU-bound not memory-bound: pure-python OASIS is O(T) per column, run over ~50-94 columns x 28 worms x ~925 frames — seconds to low-tens-of-seconds, negligible RAM. No GPU/MPS/torch anywhere in this cluster (all numpy/scipy). The only real cost multiplier to watch: calling load_combined repeatedly (e.g. inside a noise_ladder or CV loop) re-reads the 181MB of .mat files each time — load once and reuse X_list, or cache, rather than re-invoking. The module-level importlib exec of SBTG/pipeline/01_prepare_data.py at import time runs that file's top-level code once (prints, path setup); harmless but a side effect to be aware of if importing combined_data in a tight subprocess spawn loop."
