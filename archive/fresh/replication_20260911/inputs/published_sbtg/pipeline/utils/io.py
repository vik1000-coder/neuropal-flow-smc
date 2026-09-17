"""
Data loading utilities with provenance tracking.

This module provides canonical loaders for all data sources used in the pipeline.
All loaders:
1. Return data in a consistent format
2. Include node_order with every matrix
3. Log data provenance (shapes, NaN counts, etc.)
4. Respect worm boundaries for time series data
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .align import normalize_neuron_name, DIRECTION_CONVENTION


def save_trace_collection(path: Path, traces: List[np.ndarray]) -> None:
    """Save variable-length traces without Python objects or pickle.

    The archive stores all rows in one numeric matrix plus integer offsets. A
    Boolean mask preserves non-finite entries while the stored values remain
    finite, so callers can use ``allow_pickle=False``.
    """
    path = Path(path)
    arrays = [np.asarray(trace, dtype=np.float64) for trace in traces]
    if not arrays:
        raise ValueError("Trace collection is empty")
    if arrays[0].ndim != 2:
        raise ValueError("Every trace must be two-dimensional")
    n_features = arrays[0].shape[1]
    if any(trace.ndim != 2 or trace.shape[1] != n_features for trace in arrays):
        raise ValueError("Every trace must be two-dimensional with a shared feature count")

    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(trace) for trace in arrays], dtype=np.int64)
    values = np.concatenate(arrays, axis=0)
    missing = ~np.isfinite(values)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=values, missing=missing, offsets=offsets)


def load_trace_collection(path: Path) -> List[np.ndarray]:
    """Load a trace archive written by :func:`save_trace_collection`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"values", "missing", "offsets"}
        missing_keys = required.difference(archive.files)
        if missing_keys:
            raise ValueError(f"Trace archive is missing fields: {sorted(missing_keys)}")
        values = np.asarray(archive["values"], dtype=np.float64)
        missing = np.asarray(archive["missing"])
        offsets = np.asarray(archive["offsets"])

    if values.ndim != 2:
        raise ValueError("Trace archive values must be two-dimensional")
    if missing.dtype != np.bool_ or missing.shape != values.shape:
        raise ValueError("Trace archive missing mask must be Boolean and match values")
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) < 2:
        raise ValueError("Trace archive offsets must be a one-dimensional integer array")
    if offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) < 0):
        raise ValueError("Trace archive offsets are inconsistent with values")
    if not np.all(np.isfinite(values)):
        raise ValueError("Trace archive values must be finite")

    restored = values.copy()
    restored[missing] = np.nan
    return [
        restored[int(start):int(stop)].copy()
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]


@dataclass
class NeuroPALData:
    """
    Container for NeuroPAL calcium imaging data.
    
    IMPORTANT: Data is organized per-worm to avoid cross-worm discontinuities.
    Methods that need continuous time series should operate per-worm.
    """
    # Per-worm traces: List of (T_w, n_neurons) arrays
    # Each array is a single worm's recording
    traces_per_worm: List[np.ndarray]
    
    # Neuron names in consistent order
    neuron_names: List[str]
    
    # Worm identifiers
    worm_ids: List[str]
    
    # Stimulus information
    stim_names: List[str]
    stim_times: np.ndarray  # (n_stimuli, 2) start/end times
    stims_per_worm: List[np.ndarray]  # Stimulus order for each worm
    
    # Recording parameters
    fps: float
    
    # Data provenance
    source_file: str
    n_nan_values: int
    n_neurons_dropped: int
    
    @property
    def n_worms(self) -> int:
        return len(self.worm_ids)
    
    @property
    def n_neurons(self) -> int:
        return len(self.neuron_names)
    
    def get_concatenated_traces(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get concatenated traces WITH worm boundary markers.
        
        Returns:
            Tuple of:
            - traces: (total_T, n_neurons) concatenated traces
            - worm_boundaries: Array of indices where each worm starts
            
        WARNING: Only use this for methods that properly handle boundaries!
        """
        traces = np.concatenate(self.traces_per_worm, axis=0)
        boundaries = np.cumsum([0] + [t.shape[0] for t in self.traces_per_worm[:-1]])
        return traces, boundaries
    
    def get_stimulus_windows(
        self,
        stimulus_name: str,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Get stimulus response windows per worm.
        
        Args:
            stimulus_name: Name of stimulus ("nacl", "butanone", "pentanedione")
            
        Returns:
            List of (worm_idx, window_array) tuples
            Each window_array is (n_frames, n_neurons)
        """
        stim_idx = None
        for i, name in enumerate(self.stim_names):
            if name.lower() == stimulus_name.lower():
                stim_idx = i
                break
        
        if stim_idx is None:
            raise ValueError(f"Unknown stimulus: {stimulus_name}")
        
        start_sec, end_sec = self.stim_times[stim_idx]
        start_frame = int(start_sec * self.fps)
        end_frame = int(end_sec * self.fps)
        
        windows = []
        for worm_idx, traces in enumerate(self.traces_per_worm):
            if end_frame <= traces.shape[0]:
                window = traces[start_frame:end_frame, :]
                windows.append((worm_idx, window))
        
        return windows


def load_neuropal_data(
    mat_file: Path,
    min_worms: int = 1,
    normalize_names: bool = True,
) -> NeuroPALData:
    """
    Load NeuroPAL calcium imaging data from MAT file.
    
    Args:
        mat_file: Path to MAT file
        min_worms: Minimum number of worms a neuron must appear in
        normalize_names: If True, normalize neuron names to uppercase
        
    Returns:
        NeuroPALData object with per-worm organization
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required to load MAT files")
    
    mat_file = Path(mat_file)
    mat = loadmat(mat_file, simplify_cells=True)
    
    # Extract neuron names
    raw_names = mat['neurons']
    if normalize_names:
        neuron_names = [normalize_neuron_name(str(n)) for n in raw_names]
    else:
        neuron_names = [str(n) for n in raw_names]
    
    # Extract traces
    norm_traces = mat['norm_traces']
    
    # Determine number of worms
    worm_ids = [str(f) for f in mat['files']]
    n_worms = len(worm_ids)
    
    # Build per-worm traces
    traces_per_worm = []
    n_nan_total = 0
    
    # Get trace length from first valid trace
    trace_length = None
    for neuron_traces in norm_traces:
        if isinstance(neuron_traces, np.ndarray) and len(neuron_traces) > 0:
            if isinstance(neuron_traces[0], np.ndarray):
                trace_length = len(neuron_traces[0])
                break
    
    if trace_length is None:
        raise ValueError("Could not determine trace length from data")
    
    for worm_idx in range(n_worms):
        # Collect traces for this worm
        worm_traces = []
        
        for neuron_idx, neuron_traces in enumerate(norm_traces):
            if isinstance(neuron_traces, np.ndarray) and len(neuron_traces) > worm_idx:
                trace = neuron_traces[worm_idx]
                if isinstance(trace, np.ndarray) and len(trace) >= trace_length:
                    trace = trace[:trace_length].astype(float)
                else:
                    trace = np.full(trace_length, np.nan)
            else:
                trace = np.full(trace_length, np.nan)
            
            n_nan_total += np.sum(np.isnan(trace))
            worm_traces.append(trace)
        
        worm_traces = np.column_stack(worm_traces)
        traces_per_worm.append(worm_traces)
    
    # Stimulus information
    stim_names = [str(s) for s in mat['stim_names']]
    stim_times = np.asarray(mat['stim_times'], dtype=float)
    stims_per_worm = [np.asarray(row, dtype=int) for row in mat['stims']]
    fps = float(mat['fps'])
    
    return NeuroPALData(
        traces_per_worm=traces_per_worm,
        neuron_names=neuron_names,
        worm_ids=worm_ids,
        stim_names=stim_names,
        stim_times=stim_times,
        stims_per_worm=stims_per_worm,
        fps=fps,
        source_file=str(mat_file),
        n_nan_values=n_nan_total,
        n_neurons_dropped=0,
    )


def load_structural_connectome(
    connectome_dir: Path,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Load structural connectome from preprocessed files.
    
    Args:
        connectome_dir: Directory containing A_struct.npy and nodes.json
        
    Returns:
        Tuple of:
        - A_struct: Adjacency matrix (n, n) with convention A[post, pre]
        - node_order: List of neuron names
        - metadata: Dict with connectome statistics
    """
    connectome_dir = Path(connectome_dir)
    
    struct_file = connectome_dir / "A_struct.npy"
    nodes_file = connectome_dir / "nodes.json"
    
    if not struct_file.exists():
        raise FileNotFoundError(f"Structural connectome not found: {struct_file}")
    if not nodes_file.exists():
        raise FileNotFoundError(f"Node order not found: {nodes_file}")
    
    A_struct = np.load(struct_file, allow_pickle=False)
    with open(nodes_file, 'r') as f:
        node_order = json.load(f)
    
    n = len(node_order)
    n_edges = int(np.sum(A_struct > 0))
    
    metadata = {
        'n_neurons': n,
        'n_edges': n_edges,
        'density': n_edges / (n * (n - 1)),
        'direction_convention': DIRECTION_CONVENTION,
        # Keep provenance portable: the caller's absolute filesystem prefix is
        # not part of the scientific metadata.
        'source_dir': connectome_dir.name,
    }
    
    return A_struct, node_order, metadata


def load_leifer_atlas(
    atlas_dir: Path,
    genotype: str = "wild-type",
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load Leifer functional atlas.
    
    Args:
        atlas_dir: Directory containing aligned_atlas_*.npz files
        genotype: "wild-type" or "unc-31"
        
    Returns:
        Tuple of:
        - data: Dict with 'q', 'q_eq', 'amplitude' matrices
        - node_order: List of neuron names
    """
    atlas_dir = Path(atlas_dir)
    atlas_file = atlas_dir / f"aligned_atlas_{genotype}.npz"
    
    if not atlas_file.exists():
        raise FileNotFoundError(f"Leifer atlas not found: {atlas_file}")
    
    with np.load(atlas_file, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    
    # Extract node order
    if 'neurons' in data:
        node_order = list(data['neurons'])
    elif 'node_order' in data:
        node_order = list(data['node_order'])
    else:
        raise ValueError(f"No node_order found in {atlas_file}")
    
    return data, node_order


@dataclass
class ResultBundle:
    """
    Container for SBTG result files.
    
    Enforces the separation of continuous scores vs thresholded graphs.
    """
    # Continuous scores (for AUROC/AUPRC)
    mu_hat: np.ndarray  # Coupling estimates
    p_mean: np.ndarray  # P-values for mean test
    p_volatility: Optional[np.ndarray]  # P-values for volatility test
    
    # Thresholded graphs (for binary metrics)
    sign_adj: np.ndarray  # Signed adjacency {-1, 0, +1}
    volatility_adj: np.ndarray  # Volatility adjacency {0, 1}
    
    # Node order (REQUIRED)
    node_order: List[str]
    
    # Configuration
    config: Dict[str, Any]
    
    @property
    def combined_adj(self) -> np.ndarray:
        """Combined adjacency: sign ∪ volatility edges."""
        combined = self.sign_adj.copy().astype(float)
        volatility_only = (self.sign_adj == 0) & (self.volatility_adj != 0)
        combined[volatility_only] = 1.0
        return combined


def save_result_bundle(
    output_dir: Path,
    mu_hat: np.ndarray,
    p_mean: np.ndarray,
    sign_adj: np.ndarray,
    node_order: List[str],
    config: Dict[str, Any],
    p_volatility: Optional[np.ndarray] = None,
    volatility_adj: Optional[np.ndarray] = None,
) -> None:
    """
    Save SBTG results with proper separation of continuous vs binary outputs.
    
    Args:
        output_dir: Directory to save results
        mu_hat: Coupling estimates (continuous)
        p_mean: P-values for mean test (continuous)
        sign_adj: Signed adjacency after FDR (binary/ternary)
        node_order: Neuron names (REQUIRED)
        config: Configuration dict
        p_volatility: Optional volatility p-values
        volatility_adj: Optional volatility adjacency
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save continuous scores
    np.savez(
        output_dir / "continuous_scores.npz",
        mu_hat=mu_hat,
        p_mean=p_mean,
        p_volatility=p_volatility if p_volatility is not None else np.array([]),
        node_order=np.asarray(node_order, dtype=str),
    )
    
    # Save thresholded graphs
    np.savez(
        output_dir / "thresholded_graphs.npz",
        sign_adj=sign_adj,
        volatility_adj=volatility_adj if volatility_adj is not None else np.zeros_like(sign_adj),
        node_order=np.asarray(node_order, dtype=str),
    )
    
    # Save legacy result.npz for backward compatibility
    np.savez(
        output_dir / "result.npz",
        sign_adj=sign_adj,
        volatility_adj=volatility_adj if volatility_adj is not None else np.zeros_like(sign_adj),
        mu_hat=mu_hat,
        p_mean=p_mean,
        p_volatility=p_volatility if p_volatility is not None else np.array([]),
    )
    
    # Save node order separately for easy inspection
    with open(output_dir / "node_order.json", 'w') as f:
        json.dump(node_order, f, indent=2)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        # Convert numpy types to Python types
        config_clean = {}
        for k, v in config.items():
            if isinstance(v, np.ndarray):
                config_clean[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                config_clean[k] = float(v)
            else:
                config_clean[k] = v
        json.dump(config_clean, f, indent=2)


def load_result_bundle(
    result_dir: Path,
    expected_node_order: Optional[List[str]] = None,
) -> ResultBundle:
    """
    Load SBTG results and verify node order.
    
    Args:
        result_dir: Directory containing result files
        expected_node_order: If provided, verify node order matches
        
    Returns:
        ResultBundle object
    """
    result_dir = Path(result_dir)
    
    # Try new format first, fall back to legacy
    if (result_dir / "continuous_scores.npz").exists():
        with np.load(result_dir / "continuous_scores.npz", allow_pickle=False) as cont:
            mu_hat = np.asarray(cont['mu_hat'])
            p_mean = np.asarray(cont['p_mean'])
            p_volatility = (
                np.asarray(cont['p_volatility'])
                if cont['p_volatility'].size > 0
                else None
            )
            node_order = [str(value) for value in cont['node_order']]
        with np.load(result_dir / "thresholded_graphs.npz", allow_pickle=False) as thresh:
            sign_adj = np.asarray(thresh['sign_adj'])
            volatility_adj = np.asarray(thresh['volatility_adj'])
    else:
        # Legacy format
        with np.load(result_dir / "result.npz", allow_pickle=False) as data:
            mu_hat = np.asarray(data.get('mu_hat', data.get('W_param', np.array([]))))
            p_mean = np.asarray(data.get('p_mean', np.array([])))
            p_volatility = (
                np.asarray(data['p_volatility']) if 'p_volatility' in data else None
            )
            sign_adj = np.asarray(data['sign_adj'])
            volatility_adj = np.asarray(
                data.get('volatility_adj', np.zeros_like(sign_adj))
            )
        
        # Try to load node order
        node_order_file = result_dir / "node_order.json"
        neuron_names_file = result_dir / "neuron_names.json"
        
        if node_order_file.exists():
            with open(node_order_file) as f:
                node_order = json.load(f)
        elif neuron_names_file.exists():
            with open(neuron_names_file) as f:
                node_order = json.load(f)
        else:
            raise FileNotFoundError(
                f"No node_order.json found in {result_dir}. "
                "Results without node order are not valid."
            )
    
    # Validate node order if expected
    if expected_node_order is not None:
        from .align import validate_node_order
        validate_node_order(node_order, expected_node_order, context=str(result_dir))
    
    # Load config
    config_file = result_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}
    
    return ResultBundle(
        mu_hat=mu_hat,
        p_mean=p_mean,
        p_volatility=p_volatility,
        sign_adj=sign_adj,
        volatility_adj=volatility_adj,
        node_order=node_order,
        config=config,
    )
