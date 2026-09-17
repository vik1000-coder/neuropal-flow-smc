r"""Reference-network registry: structural + neuromodulator connectomes, at CLASS and
SPECIFIC-transmitter/peptide resolution, each tagged with its biological family and the
expected timescale/channel from config.py.

This is the audit surface for "which connectome/neuromodulator am I matching against":
every reference is one Reference record with a family, a resolution level, an edge count on
the current neuron support, and its expected band/channel. Specific transmitters/peptides
with too few edges on the 80-neuron support to be analysable are dropped (and logged).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sid_elegans.biolag import config as C
from sid_elegans.ground_truth import (BENTLEY, _norm, build_monoamine_layer, load_cook,
                                       load_monoamine_edges, load_neuropeptide_layer)


@dataclass
class Reference:
    name: str                 # unique key, e.g. "monoamine:serotonin" or "neuropeptide:pdf-1"
    family: str               # gap | chemical | monoamine | neuropeptide
    level: str                # structural | class | specific
    adjacency: np.ndarray     # [N,N] [post,pre] binary
    n_edges: int              # off-diagonal directed edges on this neuron support
    expected_band: str        # from config (fast/slow under the active scheme)
    expected_channel: str     # mean | distributional

    @property
    def label(self):
        return self.name.split(":", 1)[-1]


def _n_edges(A):
    A = np.asarray(A); N = A.shape[0]; m = ~np.eye(N, dtype=bool)
    return int((A[m] > 0).sum())


def _peptide_edges() -> pd.DataFrame:
    """Full Bentley neuropeptide edge list (source, target, peptide, receptor)."""
    df = pd.read_csv(BENTLEY / "edge_lists" / "edgelist_NP.csv", header=None,
                     names=["source", "target", "peptide", "receptor"])
    df["source"] = df["source"].map(_norm); df["target"] = df["target"].map(_norm)
    return df


def build_registry(neuron_names, min_specific_edges: int = 10, verbose: bool = True):
    """Return an ordered list of :class:`Reference` on ``neuron_names``.

    Includes: structural gap + chemical; monoamine class + each specific monoamine;
    neuropeptide class + each specific peptide with >= ``min_specific_edges`` edges.
    """
    refs: list[Reference] = []
    dropped: list[tuple] = []

    def add(name, family, level, A):
        ne = _n_edges(A)
        if level == "specific" and ne < min_specific_edges:
            dropped.append((name, ne)); return
        refs.append(Reference(name, family, level, np.asarray(A, float), ne,
                              C.expected_band_of(family), C.FAMILIES[family]["expected_channel"]))

    # ---- structural ----
    cook = load_cook(neuron_names)
    add("structural:chemical", "chemical", "structural", cook["chem"])
    add("structural:gap", "gap", "structural", cook["gap"])

    # ---- monoamines: class + specific ----
    ma = load_monoamine_edges()
    add("monoamine:all", "monoamine", "class", build_monoamine_layer(ma, neuron_names, None))
    for tx in sorted(ma["transmitter"].unique()):
        add(f"monoamine:{tx}", "monoamine", "specific",
            build_monoamine_layer(ma, neuron_names, tx))

    # ---- neuropeptides: class + specific ----
    add("neuropeptide:all", "neuropeptide", "class", load_neuropeptide_layer(neuron_names))
    npe = _peptide_edges()
    for pep in sorted(npe["peptide"].unique()):
        sub = npe[npe["peptide"] == pep].copy()
        sub["transmitter"] = pep
        add(f"neuropeptide:{pep}", "neuropeptide", "specific",
            build_monoamine_layer(sub, neuron_names, None))

    if verbose:
        print(f"[registry] {len(refs)} references on {len(neuron_names)} neurons "
              f"({sum(r.level=='structural' for r in refs)} structural, "
              f"{sum(r.level=='class' for r in refs)} class, "
              f"{sum(r.level=='specific' for r in refs)} specific)")
        if dropped:
            print(f"[registry] dropped {len(dropped)} specifics with < {min_specific_edges} "
                  f"edges: {', '.join(f'{n}({e})' for n, e in dropped)}")
    return refs


def registry_table(refs) -> pd.DataFrame:
    """Auditable table of the registry (one row per reference)."""
    return pd.DataFrame([{
        "name": r.name, "family": r.family, "level": r.level, "n_edges": r.n_edges,
        "expected_band": r.expected_band, "expected_channel": r.expected_channel,
    } for r in refs])
