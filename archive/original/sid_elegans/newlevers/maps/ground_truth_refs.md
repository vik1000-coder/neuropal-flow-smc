# CLUSTER: ground_truth_refs

## conventions
ORIENTATION (load-bearing): every adjacency in this cluster is [post, pre] == [target, source]. A[i,j] > 0 means edge j -> i (source j signals to target i). This is stated in ground_truth.py:2-4 and Reference.adjacency (references.py:27). load_leifer's q is also [post,pre] (ground_truth.py:144 comment). A new script must NEVER transpose these before scoring, and must align its estimator (which may be [pre,post] or [post,pre] depending on your trunk) to [post,pre] first.

NEURON SUPPORT / X_list: all builders take a `neuron_names` list (the "support") and align everything to THAT order via _reindex / an idx dict. neuron_names is expected to be the CLASS-LEVEL functional/calcium neuron order (e.g. the ~80-neuron functional support referenced in references.py:6-8; the connectome nodes.json is 94 class labels like 'ADA','ADE','AIY' with NO L/R suffix). Names are normalized with _norm = str(name).strip().upper() everywhere (ground_truth.py:25-26). Bilateral edge-list names are collapsed to class by dropping a trailing 'L'/'R' when the stripped name is in the support (build_monoamine_layer.to_class, ground_truth.py:93-100). Missing neurons become all-zero rows/cols silently (_reindex, ground_truth.py:37-42).

WEIGHTED vs BINARY (verified): structural refs (Cook chem/gap) carry raw SYNAPSE COUNTS (chem max 2366, gap max 777, integer-valued floats), NOT binarized. Modulator refs (monoamine, neuropeptide, leifer positive) are BINARY 0.0/1.0. A scorer that assumes 0/1 will be wrong on structural refs; binarize with (A>0) or rank appropriately. dtype is float64 for all; shape is (N,N) where N=len(neuron_names).

DIAGONAL: modulator refs have a ZERO diagonal by construction — build_monoamine_layer skips s==t (ground_truth.py:105) and _n_edges masks the diagonal (references.py:39). Cook structural refs RETAIN a nonzero diagonal (verified chem trace=3071; load_cook does not zero it). So there is NO self-modulation ground truth for the aminergic/peptidergic families.

CLASS-LEVEL: the ground truth is already class-level (support has no L/R; to_class collapses bilateral). Note an internal inconsistency to respect: neuropeptide:all comes from edgelist_NP_classes.csv (via load_neuropeptide_layer, ground_truth.py:125) while per-peptide specifics come from edgelist_NP.csv (references.py:45,81); both are then class-collapsed by to_class.

CHANNELS/BANDS: config.CHANNELS = ["mean","gain","tail"]; DISTRIBUTIONAL=["gain","tail"] (config.py:90-91). Each family maps to an expected_channel ('mean' for gap/chemical, 'distributional' for monoamine/neuropeptide) and expected band under the ACTIVE two-band scheme (fast/slow), resolved via config.expected_band_of (config.py:82-86). The single pre-registered confirmatory test is CONFIRMATORY={reference:'neuropeptide:all', channel:'gain', statistic:'dist_minus_mean', band:'slow'} (config.py:95-96). config.MIXED marks serotonin/tyramine as biophysically mixed fast+slow (config.py:101). LAG grid is LAG_FRAMES=[1,2,3,5,8,10,15,20,30,40] at FPS=4.0 (config.py:16-18).

REGISTRY is the audit surface: build_registry(neuron_names) -> ordered list[Reference]; each Reference has .name ('family:label'), .family, .level (structural|class|specific), .adjacency [post,pre] float, .n_edges, .expected_band, .expected_channel. registry_table(refs) gives a DataFrame. This is the canonical set of scoring targets a new analysis should iterate over.

## key_functions
### load_cook  (ground_truth.py:45)
- sig: `load_cook(neuron_names) -> dict[str, np.ndarray]`
- returns: dict{'chem','gap','struct'} each (N,N) float64, [post,pre], N=len(neuron_names). WEIGHTED synapse counts (not binary). 'struct'=chem+gap. NONZERO diagonal retained (Cook self-values).
- purpose: Load Cook chemical + gap connectomes from SBTG/results/intermediate/connectome/{A_chem,A_gap}.npy + nodes.json (94 classes), defensively transpose-correct to [post,pre], and reindex to neuron_names order.
- gotchas: Transpose auto-correction (lines 60-64) only fires if BOTH 'AIY' and 'RIA' are in nodes; on a custom support lacking them, orientation is NOT corrected. Gap matrix symmetrized via np.maximum(A_gap,A_gap.T) (line 65). Reads files from disk each call (no cache). Missing neurons -> zero rows/cols silently. Requires SBTG/ artifacts present.

### build_monoamine_layer  (ground_truth.py:81)
- sig: `build_monoamine_layer(edges: pd.DataFrame, neuron_names, transmitter=None) -> np.ndarray`
- returns: (N,N) float64 BINARY 0/1, [post,pre]. A[idx[target],idx[source]]=1. ZERO diagonal (self-edges skipped).
- purpose: Turn a source/target edge DataFrame into a binary [post,pre] adjacency on neuron_names; filters to one transmitter if given (None=all).
- gotchas: Collapses bilateral L/R to class via to_class (lines 93-100). Skips self-edges (s==t, line 105) so diagonal is always 0. Binary only — loses edge weight/multiplicity. Iterates rows (iterrows) — fine at N~94 but O(rows). Expects columns named 'source','target','transmitter'.

### load_all_monoamine_layers  (ground_truth.py:111)
- sig: `load_all_monoamine_layers(neuron_names) -> dict[str, np.ndarray]`
- returns: dict keyed by 'dopamine','serotonin','tyramine','octopamine','monoamine_all', each (N,N) float64 binary [post,pre].
- purpose: Convenience: build every per-transmitter monoamine layer + combined from edgelist_MA.csv.
- gotchas: octopamine has only ~9 edges on the 94-support (below the specific-edge threshold used by build_registry) — thin/unreliable. Reads edgelist_MA.csv each call.

### load_neuropeptide_layer  (ground_truth.py:119)
- sig: `load_neuropeptide_layer(neuron_names) -> np.ndarray`
- returns: (N,N) float64 binary [post,pre]. Zero diagonal. This is the core peptidergic 'neuropeptide:all' target.
- purpose: Build the class-level neuropeptide receptor network from edgelist_NP_classes.csv (source expresses peptide, target expresses cognate receptor).
- gotchas: Uses the CLASS file (edgelist_NP_classes.csv), unlike per-peptide specifics which use edgelist_NP.csv. Sets transmitter='np' then calls build_monoamine_layer(...,transmitter=None). Densest modulator target (~1016 edges).

### load_leifer  (ground_truth.py:133)
- sig: `load_leifer(neuron_names, alpha: float = 0.05)`
- returns: dict{'q':(N,N) float64 [post,pre] reindexed q-values, 'positive':(N,N) float64 binary = (q<alpha)} OR None if wormneuroatlas unavailable/errors. NOTE: dict does NOT include q_eq or a negative/eval mask despite the docstring.
- purpose: Load Randi/Leifer FUNCTIONAL connectome via wormneuroatlas.NeuroAtlas; the target for the functional_target analysis.
- gotchas: RETURNS None IN THIS ENV (verified). Two blockers, both swallowed by the bare `except Exception: return None` at line 151: (1) wa.NeuroAtlas() construction (line 143) makes a WormBase DB-version HTTP request with no offline flag -> JSONDecodeError offline; (2) line 145 calls get_signal_propagation_q(mode='eq') but the real signature is get_signal_propagation_q(self, strain='wt') — no 'mode' kwarg — so it raises TypeError; the correct method is the SEPARATE atlas.get_signal_propagation_q_eq(). The hasattr guard on line 146 checks the same method name so it's always True. q_eq is computed but never returned; the 'confirmed-negative' mask promised in the docstring (lines 135-137) is NOT built. np.nan_to_num(q,nan=1.0) treats missing q as non-significant. atlas.neuron_ids ~ number of atlas neurons (excludes CAN).

### _reindex  (ground_truth.py:29)
- sig: `_reindex(A: np.ndarray, src_names, dst_names) -> np.ndarray`
- returns: (len(dst_names),len(dst_names)) float64, orientation preserved. Missing dst neurons -> all-zero rows/cols.
- purpose: Reindex a square [post,pre] matrix from src order to dst order via _norm-keyed lookup.
- gotchas: O(N^2) pure-Python double loop (fine for N~94). Silent zero-fill for missing neurons. Names matched with _norm (upper+strip).

### Reference (dataclass)  (references.py:23)
- sig: `Reference(name:str, family:str, level:str, adjacency:np.ndarray, n_edges:int, expected_band:str, expected_channel:str); .label property -> name.split(':',1)[-1]`
- returns: Record; adjacency is (N,N) [post,pre] float (binary for modulator/leifer, weighted counts for structural).
- purpose: One scoring target with its biological family, resolution level, expected band + channel. The unit a new analysis iterates over.
- gotchas: adjacency holds ONLY an N×N edge matrix — no field for node-level (diagonal/membership) targets, relevant to self_gain_diagonal and receptor-set localization. family must be a key in config.FAMILIES or the add() call raises KeyError.

### build_registry  (references.py:51)
- sig: `build_registry(neuron_names, min_specific_edges: int = 10, verbose: bool = True) -> list[Reference]`
- returns: Ordered list[Reference]: structural:chemical, structural:gap, monoamine:all, monoamine:<tx> specifics, neuropeptide:all, neuropeptide:<pep> specifics. ~17 refs on the 94-support.
- purpose: Assemble the full reference registry (all connectome + neuromodulator targets) on a given support; this is the canonical hook for adding new targets (e.g. functional).
- gotchas: Drops specifics with < min_specific_edges edges (logs them). Calls load_cook/load_*_layer each invocation (re-reads disk). verbose prints to stdout. Uses edgelist_NP.csv for peptide specifics but load_neuropeptide_layer (NP_classes.csv) for neuropeptide:all. Inner add() (lines 60-65) is where a new family/level would be appended.

### registry_table  (references.py:99)
- sig: `registry_table(refs) -> pd.DataFrame`
- returns: DataFrame cols: name,family,level,n_edges,expected_band,expected_channel (one row per ref).
- purpose: Auditable flat view of the registry.
- gotchas: Does not include the adjacency arrays.

### expected_band_of  (config.py:82)
- sig: `expected_band_of(family: str) -> str`
- returns: str band label ('fast'/'slow' under ACTIVE two-band; else band_4).
- purpose: Map a family to its expected band under the ACTIVE scheme.
- gotchas: Depends on module-global ACTIVE_BANDS='two' (config.py:75). bands() (config.py:78) returns TWO_BAND or FOUR_BAND (seconds ranges). Changing ACTIVE_BANDS silently reshapes every band label.

### FAMILIES / channel constants  (config.py:26)
- sig: `FAMILIES: dict[str,dict] (keys gap,chemical,monoamine,neuropeptide; each has expected_speed, expected_channel, band_4, mechanism, rationale); CHANNELS=['mean','gain','tail'] (line 90); DISTRIBUTIONAL=['gain','tail'] (line 91); CONFIRMATORY dict (line 95); MIXED set (line 101)`
- returns: module-level constants.
- purpose: Single source of biological priors: which channel/band each family should peak in; the one pre-registered confirmatory test.
- gotchas: build_registry indexes FAMILIES[family]['expected_channel'] directly (references.py:65) — a new family (e.g. 'functional') MUST be added to FAMILIES first or registry construction raises KeyError.

## integration_hooks
- **1. functional_target: score against the Randi/Leifer functional connectome**: Fix and extend load_leifer (ground_truth.py:133-152). It currently ALWAYS returns None here. Two fixes: (a) construction — wa.NeuroAtlas() (line 143) needs WormBase network; either run once online and cache the arrays to a local .npy/.npz under SBTG/data, or wrap construction so it loads from a saved cache, since NeuroAtlas exposes no offline flag; (b) replace the invalid get_signal_propagation_q(mode='eq') (line 145) with the real method atlas.get_signal_propagation_q_eq() and actually return it — add keys 'q_eq'/'negative'/'mask' so the eval mask promised in the docstring (q<alpha positives OR q_eq<alpha confirmed negatives) exists. Then register it: add a 'functional' entry to config.FAMILIES (config.py:26) with expected_channel (likely 'mean' or its own label) so build_registry's FAMILIES lookup (references.py:65) succeeds, and add an add('functional:leifer','functional','functional', leifer['positive']) call inside build_registry (references.py, after line 80) guarded on load_leifer(...) is not None. q is already [post,pre] and _reindex'd, so it flows through existing scoring unchanged. Cache the NeuroAtlas object — construction is slow and network-bound.
- **2. gating_global_mode: condition on the global brain-state mode instead of removing it**: No ground-truth matrix change. The confound is documented in config.CAVEATS['global_mode'] (config.py:114-117). Add a new config block next to CONFIRMATORY (config.py:95) — e.g. a GATING dict naming the modulator variable (top global PC / state phase) and the target families (expected_channel=='distributional': monoamine, neuropeptide). Your new analysis iterates build_registry(neuron_names) refs, filters to r.expected_channel=='distributional' (Reference field, references.py:31), and tests gain-coupling modulation by global-state phase vs variance-matched controls. Reuse Reference.adjacency [post,pre] to pick candidate edges; reuse expected_band_of / bands() for the slow band. The gating logic itself lives in the estimator, not these files.
- **3. self_gain_diagonal: expose the zeroed diagonal (self-excitability) channel**: KEY CONSTRAINT: all modulator ground truths have a ZERO diagonal — build_monoamine_layer explicitly skips s==t (ground_truth.py:105) and _n_edges masks it (references.py:39). So there is no self-edge target for aminergic/peptidergic families. To score state-dependent self-gain you must build a NODE-LEVEL target (1-D boolean over neuron_names of receptor-expressing neurons), not an N×N matrix. Add a new builder in ground_truth.py alongside build_monoamine_layer that reads the 'target'/'receptor' columns of edgelist_MA.csv / edgelist_NP.csv and returns a length-N membership vector (target-side = receptor-expressing). Reference.adjacency (references.py:27) only holds N×N, so either store the membership as diag(vector) or extend the dataclass with a node_target field. Cook structural refs DO retain a nonzero diagonal (load_cook does not zero it) but that is synapse self-count, not the gain target.
- **4. class_aggregation: aggregate the estimator to neuron CLASSES**: The registry is ALREADY class-level (support = class labels with no L/R; ground truth built on it). To aggregate your per-neuron estimator to match, reuse the exact class-collapse rule in build_monoamine_layer.to_class (ground_truth.py:93-100): _norm then drop a trailing L/R when the stem is in neuron_names. Map estimator neuron ids -> the SAME neuron_names order you pass to build_registry, then average/pool estimator edges within each (post_class,pre_class) cell. Mirror _reindex (ground_truth.py:29-42) for the alignment. Pass the identical neuron_names to build_registry so estimator and targets share indexing. Watch the NP class/specific file mismatch (neuropeptide:all from NP_classes.csv, specifics from NP.csv, references.py:45,81).
- **5. predictive_likelihood: held-out one-step predictive likelihood localized to receptor-expressing vs control neurons**: Consume build_registry outputs as the localization masks, not as AUROC targets. For a modulator Reference, the set of receptor-expressing (post) neurons = rows i with any adjacency[i,:]>0 (adjacency is [post,pre], references.py:31). Build the receptor-expressing node set from that row-support (or from a new node-membership builder as in hook 3), then form variance/degree-matched control neurons. No change required to these files beyond optionally exposing a helper that returns the post-support set. The likelihood computation itself is estimator-side.
- **6. frozen_features: linear probe / closed-form quadratic score on frozen trunk features**: These files are not touched by the freezing/trunk. The only interface is the final scoring step, which must still iterate build_registry(neuron_names) references (references.py:51) and use config.CHANNELS/DISTRIBUTIONAL (config.py:90-91) and CONFIRMATORY (config.py:95) exactly as the current estimator does, so results stay comparable. Keep neuron_names identical to the registry support.
- **7. hierarchical_shrinkage: empirical-Bayes / James-Stein shrinkage toward a class/population prior**: The grouping hierarchy for the prior IS the registry structure: shrink per-edge ACMMA moments toward priors defined by Reference.family and Reference.level (references.py:26,30-31 — gap|chemical|monoamine|neuropeptide × structural|class|specific). Call build_registry once, use registry_table (references.py:99) to get the (family,level) grouping, and pool moments within family (or within a specific reference's edge set A>0). config.FAMILIES (config.py:26) gives the family taxonomy. No code change to these files — just consume Reference.family/level and adjacency support.
- **8. noise_ladder: geometric ladder of denoising sigma_frac coupled into one estimator**: Not implemented in these files (denoising is estimator/preprocessing-side). To honor the config-centralization principle (config.py:5 'nothing biological is hardcoded elsewhere'), add the sigma_frac geometric ladder as a new module-level constant next to LAG_FRAMES (config.py:17), e.g. SIGMA_FRAC_LADDER = [...], so the analysis and figures read it from one place. No ground_truth.py or references.py change.

## resource_notes
Everything in this cluster is tiny and 16GB-safe. All matrices are (N,N) float64 with N≈80–94 (~70KB each); full registry ~17 arrays. build_registry on the 94-node support runs in ~0.3s (measured). Edge-list CSVs are small (edgelist_MA.csv 2626 rows, edgelist_NP.csv 8931 rows, 202KB); build_monoamine_layer uses pandas iterrows but at these sizes it is sub-second. No large allocations, no GPU/MPS use anywhere in these three files. The ONLY heavy/fragile external dependency is load_leifer -> wormneuroatlas.NeuroAtlas(): construction makes WormBase HTTP requests (fails offline here) and, when online, downloads/loads atlas data and connectomes (load_connectomes=True by default) — this is the slow, network-bound, cache-worthy step; construct it once and reuse (do not call load_leifer per-reference or in a loop). No OOM risk. Repeated build_registry / load_cook calls re-read disk each time (no caching) — cheap but avoid calling inside tight inner loops if you scale N up.
