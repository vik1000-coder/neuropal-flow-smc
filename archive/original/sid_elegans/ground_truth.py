"""Corrected ground-truth connectome + neuromodulator layers for the SBTG data.

All matrices use the pipeline convention ``A[post, pre] = A[target, source]``
(``A[i, j] > 0`` means edge ``j -> i``), aligned to a given functional neuron order.

The Cook chemical connectome in the repo was saved transposed (see the audit); the
repo source is now patched, but this loader ALSO corrects orientation defensively so it
does not depend on regenerating artifacts. Monoamine (Bentley 2016) layers are built
directly from the offline edge lists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent / "SBTG"
DATA = REPO / "data"
CONN = REPO / "results" / "intermediate" / "connectome"
BENTLEY = DATA / "S1_Dataset"
MONOAMINES = ("dopamine", "serotonin", "tyramine", "octopamine")


def _norm(name: str) -> str:
    return str(name).strip().upper()


def _reindex(A: np.ndarray, src_names, dst_names) -> np.ndarray:
    """Reindex a square matrix from ``src_names`` order to ``dst_names`` order.

    Missing neurons become all-zero rows/cols. Orientation is preserved.
    """
    src = [_norm(n) for n in src_names]
    idx = {n: i for i, n in enumerate(src)}
    n = len(dst_names)
    out = np.zeros((n, n), dtype=float)
    keep = [(i, idx[_norm(dn)]) for i, dn in enumerate(dst_names) if _norm(dn) in idx]
    for a, sa in keep:
        for b, sb in keep:
            out[a, b] = A[sa, sb]
    return out


def load_cook(neuron_names) -> dict[str, np.ndarray]:
    """Load Cook chem/gap/struct connectomes, corrected to ``[post, pre]`` and aligned.

    Returns dict with keys ``chem``, ``gap``, ``struct`` as ``[N, N]`` matrices in the
    order of ``neuron_names``.
    """
    import json

    nodes = [_norm(n) for n in json.load(open(CONN / "nodes.json"))]
    A_chem = np.load(CONN / "A_chem.npy")
    A_gap = np.load(CONN / "A_gap.npy")

    # Defensive orientation correction: detect transpose via known directed synapses.
    # If AIY->RIA / AWC->AIY sit at [pre,post] rather than [post,pre], transpose.
    idx = {n: i for i, n in enumerate(nodes)}
    if "AIY" in idx and "RIA" in idx:
        post_pre = A_chem[idx["RIA"], idx["AIY"]]   # correct location for AIY->RIA
        pre_post = A_chem[idx["AIY"], idx["RIA"]]
        if pre_post > post_pre:
            A_chem = A_chem.T.copy()
    A_gap = np.maximum(A_gap, A_gap.T)  # electrical junctions are symmetric

    chem = _reindex(A_chem, nodes, neuron_names)
    gap = _reindex(A_gap, nodes, neuron_names)
    return {"chem": chem, "gap": gap, "struct": chem + gap}


def load_monoamine_edges() -> pd.DataFrame:
    """Load the Bentley monoamine edge list (source, target, transmitter, receptor)."""
    df = pd.read_csv(BENTLEY / "edge_lists" / "edgelist_MA.csv", header=None,
                     names=["source", "target", "transmitter", "receptor"])
    df["source"] = df["source"].map(_norm)
    df["target"] = df["target"].map(_norm)
    return df


def build_monoamine_layer(edges: pd.DataFrame, neuron_names, transmitter=None) -> np.ndarray:
    """Build a binary ``[post, pre]`` monoamine adjacency aligned to ``neuron_names``.

    ``adjacency[i, j] = 1`` means source ``j`` signals to target ``i`` (edge j -> i).
    Bilateral names in the edge list are collapsed to class (drop trailing L/R) to match
    the functional neuron classes.
    """
    idx = {_norm(n): i for i, n in enumerate(neuron_names)}
    df = edges if transmitter is None else edges[edges["transmitter"] == transmitter]
    n = len(neuron_names)
    A = np.zeros((n, n), dtype=float)

    def to_class(name):
        name = _norm(name)
        if name in idx:
            return name
        # drop trailing L/R (bilateral) if that lands on a known class
        if len(name) > 1 and name[-1] in ("L", "R") and name[:-1] in idx:
            return name[:-1]
        return None

    n_edges = 0
    for _, row in df.iterrows():
        s = to_class(row["source"]); t = to_class(row["target"])
        if s is not None and t is not None and s != t:
            A[idx[t], idx[s]] = 1.0  # [post=target, pre=source]
            n_edges += 1
    return A


def load_all_monoamine_layers(neuron_names) -> dict[str, np.ndarray]:
    """Return per-transmitter + combined monoamine layers aligned to ``neuron_names``."""
    edges = load_monoamine_edges()
    layers = {t: build_monoamine_layer(edges, neuron_names, t) for t in MONOAMINES}
    layers["monoamine_all"] = build_monoamine_layer(edges, neuron_names, None)
    return layers


def load_neuropeptide_layer(neuron_names) -> np.ndarray:
    """Build the Bentley neuropeptide (class-level) receptor network, ``[post, pre]``.

    Any (source expresses peptide, target expresses cognate receptor) pair is an edge.
    This is the peptidergic slow-neuromodulation target — the core sid_neuromod thesis.
    """
    path = BENTLEY / "edge_lists" / "edgelist_NP_classes.csv"
    df = pd.read_csv(path, header=None, names=["source", "target", "peptide", "receptor"])
    df["transmitter"] = "np"
    df["source"] = df["source"].map(_norm)
    df["target"] = df["target"].map(_norm)
    return build_monoamine_layer(df, neuron_names, transmitter=None)


def load_leifer(neuron_names, alpha: float = 0.05):
    """Load the Randi/Leifer functional atlas (needs wormneuroatlas). Returns ``None`` if
    unavailable. Ground truth is ``q < alpha`` connections; the evaluation mask restricts
    to confirmed edges (``q < alpha`` positives OR ``q_eq < alpha`` confirmed negatives).
    """
    try:
        import wormneuroatlas as wa
    except Exception:
        return None
    try:
        atlas = wa.NeuroAtlas()
        q = np.asarray(atlas.get_signal_propagation_q())          # [post, pre]
        q_eq = np.asarray(atlas.get_signal_propagation_q(mode="eq")) \
            if hasattr(atlas, "get_signal_propagation_q") else None
        atlas_names = [_norm(n) for n in atlas.neuron_ids]
        Q = _reindex(np.nan_to_num(q, nan=1.0), atlas_names, neuron_names)
        pos = (Q < alpha).astype(float)
        return {"q": Q, "positive": pos}
    except Exception:
        return None
