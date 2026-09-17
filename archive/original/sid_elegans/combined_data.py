r"""Pool the repo's *head-only* NeuroPAL recordings into one dataset.

Combines OH16230 head (21 worms) with OH15500 head (7 worms) using the repo's own
loaders (D/V collapse, L/R averaging), selects neurons that are (a) in the Cook connectome
and (b) covered in enough worms across both strains, keeps the complete-case worms
(non-imputed), and returns per-worm z-scored ``[T, N]`` matrices pooled across strains.

Tail recordings are intentionally excluded.  The released head/tail files do not
contain a verified simultaneous-recording pairing table and cannot be merged by
array index.  OH15500 is explicitly resampled from its native 4.1 Hz grid to the
4.0 Hz analysis grid before pooling.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent / "SBTG"
CONN = REPO / "results" / "intermediate" / "connectome"

_spec = importlib.util.spec_from_file_location("prep", str(REPO / "pipeline" / "01_prepare_data.py"))
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)


def _worm_matrix(data, worm_idx, nodes, complete_case=True):
    """Build a [T, len(nodes)] matrix for one worm.

    ``complete_case``: return None if any node is missing. Otherwise fill missing nodes
    with NaN (per-feature mean-imputation is applied downstream by the estimator — this is
    NOT trace duplication, so it does not leak dynamics across worms).
    """
    num_worms = len(data["worm_ids"])
    n2i = data["name_to_indices"]
    traces = {}
    for node in nodes:
        tr = None
        if node in n2i:
            tr = _prep.collect_worm_trace(data["norm_traces"], n2i[node], worm_idx, num_worms)
        if tr is None and complete_case:
            return None
        traces[node] = None if tr is None else np.asarray(tr, dtype=float)
    lens = [len(t) for t in traces.values() if t is not None]
    if not lens:
        return None
    minlen = min(lens)
    cols = [(t[:minlen] if t is not None else np.full(minlen, np.nan)) for t in traces.values()]
    return np.stack(cols, axis=1)   # head-align (t=0 at start)


def _resample_matrix(matrix: np.ndarray, native_fps: float, analysis_fps: float) -> np.ndarray:
    """Linearly resample a trace matrix on an explicit zero-origin time grid."""
    matrix = np.asarray(matrix, dtype=float)
    if np.isclose(native_fps, analysis_fps):
        return matrix.copy()
    source_time = np.arange(len(matrix), dtype=np.float64) / float(native_fps)
    target_time = np.arange(
        int(np.floor(source_time[-1] * analysis_fps)) + 1, dtype=np.float64
    ) / float(analysis_fps)
    result = np.empty((len(target_time), matrix.shape[1]), dtype=np.float64)
    for column in range(matrix.shape[1]):
        result[:, column] = np.interp(target_time, source_time, matrix[:, column])
    return result


def load_combined(coverage_frac: float = 0.6, complete_case: bool = True,
                  min_worms_per_neuron: int = 6, signal: str = "raw", verbose: bool = True):
    """Return ``(X_list, neuron_names, fps)`` pooled across OH16230 + OH15500.

    ``complete_case=True``: keep a neuron if it appears in >= ``coverage_frac`` of all
    worms AND is in the connectome; then keep only worms that have every selected neuron.
    ``complete_case=False``: keep a neuron if it appears in >= ``min_worms_per_neuron``
    worms AND is in the connectome; keep ALL worms, filling missing (worm, neuron) entries
    with NaN (mean-imputed downstream — no cross-worm leakage). This uses all the data.
    """
    data16 = _prep.load_neuropal_data(REPO / "data", include_tail=False, collapse_dv=True)
    data15 = _load15()
    conn_nodes = {n.upper() for n in json.load(open(CONN / "nodes.json"))}

    datasets = [("OH16230", data16), ("OH15500", data15)]
    total_worms = sum(len(d["worm_ids"]) for _, d in datasets)

    # coverage of each connectome node across all worms of both strains
    cover = {}
    for node in conn_nodes:
        c = 0
        for _, d in datasets:
            nw = len(d["worm_ids"])
            if node in d["name_to_indices"]:
                for w in range(nw):
                    if _prep.collect_worm_trace(d["norm_traces"], d["name_to_indices"][node],
                                                w, nw) is not None:
                        c += 1
        cover[node] = c
    if complete_case:
        nodes = sorted([n for n, c in cover.items() if c >= coverage_frac * total_worms])
    else:
        nodes = sorted([n for n, c in cover.items() if c >= min_worms_per_neuron])

    X_list, strains = [], []
    for name, d in datasets:
        for w in range(len(d["worm_ids"])):
            M = _worm_matrix(d, w, nodes, complete_case=complete_case)
            if M is not None:
                M = _resample_matrix(M, float(d["fps"]), 4.0)
                X_list.append(M); strains.append(name)
    if verbose:
        mode = "complete-case" if complete_case else "NaN-imputed"
        print(f"[combined] {total_worms} total worms; {len(nodes)} neurons; "
              f"{len(X_list)} {mode} worms ({strains.count('OH16230')} OH16230 + "
              f"{strains.count('OH15500')} OH15500); signal={signal}")

    # Optional calcium deconvolution -> continuous activity (before standardization)
    if signal == "deconv":
        from sid_elegans.deconv import deconvolve_matrix
        X_list = [deconvolve_matrix(x, fps=4.0) for x in X_list]
    elif signal != "raw":
        raise ValueError("signal must be 'raw' or 'deconv'")

    # Standardize like the repo: CENTER each neuron but PRESERVE relative variances
    # (divide by a single global scale, not per-neuron). Per-neuron z-scoring amplifies
    # quiet-neuron noise and destroys the variance-magnitude signal the gain channel uses
    # (verified: it collapses the validated tyramine result 0.73 -> 0.35).
    allX = np.concatenate(X_list, axis=0)
    mu = np.nanmean(allX, axis=0)
    global_sd = float(np.nanstd(allX - mu)) + 1e-8
    X_list = [(x - mu) / global_sd for x in X_list]
    return X_list, nodes, 4.0


def _load15():
    """Load OH15500 head with the repo loader (no tail file for this strain)."""
    import scipy.io as sio
    from pipeline.utils.align import collapse_dv_subtypes
    mat = sio.loadmat(REPO / "data" / "Head_Activity_OH15500.mat", simplify_cells=True)
    data = {
        "neuron_names": [str(n).strip().upper() for n in mat["neurons"]],
        "norm_traces": mat["norm_traces"],
        "fps": float(mat["fps"]),
        "worm_ids": [str(f) for f in mat["files"]],
        "stim_names": [str(s) for s in mat["stim_names"]],
        "stim_times": np.asarray(mat["stim_times"], dtype=float),
        "stims_per_worm": [np.asarray(row, dtype=int) for row in mat["stims"]],
        "source_recording": str(REPO / "data" / "Head_Activity_OH15500.mat"),
    }
    original = data["neuron_names"]
    collapsed = [collapse_dv_subtypes(n) for n in original]
    n2i = {}
    for idx, c in enumerate(collapsed):
        n2i.setdefault(c, []).append(idx)
    data["neuron_names"] = sorted(n2i.keys())
    data["name_to_indices"] = n2i
    return data
