r"""Per-neuron neurotransmitter-receptor loads by MECHANISM (metabotropic vs ionotropic),
from the CeNGEN single-cell expression atlas bundled with wormneuroatlas.

The point: neuromodulation is a *metabotropic* (slow GPCR) phenomenon. If the self-gain readout
is really picking up neuromodulation, its SLOW-band signal should track a neuron's *metabotropic*
receptor load, and NOT its *ionotropic* (fast ligand-gated channel) load. Several transmitters
have BOTH kinds of receptor (serotonin: metabotropic ser-1/4/5/7 vs ionotropic MOD-1; tyramine:
metabotropic tyra-2/3 vs ionotropic LGC-55), enabling a within-transmitter double dissociation
that no trivial confound reproduces.

Receptor pharmacology is standard C. elegans (WormBook / Bargmann 1998 / Bentley 2016): all
neuropeptide receptors and the aminergic GPCRs are metabotropic (slow); the ligand-gated channels
(mod-1, lgc-55, unc-49, nicotinic acr-*, iGluRs) are ionotropic (fast).
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

# ---- receptor panels by mechanism (gene names as in CeNGEN) -------------------------------
METABOTROPIC = {
    "serotonin":  ["ser-1", "ser-4", "ser-5", "ser-7"],
    "tyramine":   ["tyra-2", "tyra-3", "ser-2"],
    "octopamine": ["octr-1", "ser-3", "ser-6"],
    "dopamine":   ["dop-1", "dop-2", "dop-3", "dop-4", "dop-5", "dop-6"],
    "acetylcholine": ["gar-1", "gar-2", "gar-3"],           # muscarinic
    "gaba":       ["gbb-1", "gbb-2"],
    "glutamate":  ["mgl-1", "mgl-2", "mgl-3"],
}
IONOTROPIC = {
    "serotonin":  ["mod-1"],
    "tyramine":   ["lgc-55"],
    "dopamine":   ["lgc-53"],
    "acetylcholine": ["acr-2", "acr-3", "acr-5", "unc-29", "unc-38", "unc-63", "deg-3", "des-2"],
    "gaba":       ["unc-49", "lgc-35", "lgc-36", "lgc-37", "exp-1"],
    "glutamate":  ["glr-1", "glr-2", "glr-3", "glr-4", "glr-5", "nmr-1", "nmr-2", "avr-15"],
}
# neuropeptide receptors are ALL metabotropic GPCRs (the peptidergic target)
NEUROPEPTIDE_GPCR = ["npr-1", "npr-2", "npr-3", "npr-4", "npr-5", "npr-6", "npr-9", "npr-10",
                     "npr-11", "npr-12", "pdfr-1", "frpr-3", "frpr-18", "dmsr-1", "dmsr-2",
                     "ntr-1", "ckr-1", "ckr-2", "egl-6", "seb-2", "trhr-1"]


@functools.lru_cache(maxsize=4)
def _cengen(threshold: int = 2) -> pd.DataFrame:
    """CeNGEN thresholded expression: genes x neuron-classes (binary). threshold 1..4 =
    liberal..stringent. Returns a DataFrame indexed by lowercase gene name."""
    import wormneuroatlas as wa
    names = {1: "liberal_threshold1", 2: "medium_threshold2",
             3: "conservative_threshold3", 4: "stringent_threshold4"}
    d = Path(wa.__file__).resolve().parent / "data" / f"cengen_021821_{names[threshold]}.csv"
    df = pd.read_csv(d)
    df = df.set_index(df["gene_name"].astype(str).str.lower())
    meta = [c for c in [df.columns[0], "gene_name", "Wormbase_ID"] if c in df.columns]
    return df.drop(columns=meta)


def _panel_load(genes, neuron_names, threshold=2) -> np.ndarray:
    """Per-neuron count of expressed genes from ``genes`` (aligned to neuron_names; NaN if the
    neuron class is absent from CeNGEN)."""
    df = _cengen(threshold)
    cols = {c.upper(): c for c in df.columns}
    present = [g.lower() for g in genes if g.lower() in df.index]
    out = np.full(len(neuron_names), np.nan)
    for i, n in enumerate(neuron_names):
        c = cols.get(str(n).strip().upper())
        if c is None:
            continue
        out[i] = float((df.loc[present, c] > 0).sum()) if present else 0.0
    return out


def mechanism_loads(neuron_names, threshold: int = 2) -> dict:
    """Per-neuron metabotropic vs ionotropic receptor loads (aligned to neuron_names).

    Returns dict with:
      ``metab_total`` / ``iono_total``  — summed over all transmitters,
      ``np_gpcr``                        — neuropeptide GPCR load (the peptidergic axis),
      per-transmitter ``metab_<tx>`` / ``iono_<tx>`` (serotonin, tyramine have both),
      ``in_cengen``                      — bool mask of neurons present in CeNGEN.
    All are graded counts (0..panel size); NaN where the neuron is absent from CeNGEN.
    """
    out = {}
    metab_all, iono_all = [], []
    for tx, genes in METABOTROPIC.items():
        out[f"metab_{tx}"] = _panel_load(genes, neuron_names, threshold)
        metab_all.append(out[f"metab_{tx}"])
    for tx, genes in IONOTROPIC.items():
        out[f"iono_{tx}"] = _panel_load(genes, neuron_names, threshold)
        iono_all.append(out[f"iono_{tx}"])
    out["np_gpcr"] = _panel_load(NEUROPEPTIDE_GPCR, neuron_names, threshold)
    out["metab_total"] = np.nansum(np.array(metab_all), axis=0)
    out["iono_total"] = np.nansum(np.array(iono_all), axis=0)
    # in-cengen mask: any panel returned a finite value
    stacked = np.array(metab_all + iono_all)
    out["in_cengen"] = np.isfinite(stacked).any(axis=0)
    # blank out the totals where the neuron is absent
    out["metab_total"][~out["in_cengen"]] = np.nan
    out["iono_total"][~out["in_cengen"]] = np.nan
    return out
