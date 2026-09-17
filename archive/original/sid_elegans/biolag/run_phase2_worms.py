r"""Robustness of the Phase-2 band-concordance to worm count (the gap the log-slope battery
covered but the new metric did not). Re-runs the SAME confirmatory + global-mode pipeline on
the higher-worm / fewer-neuron corners of the frontier -- all REAL worms, no imputation:
  coverage 0.8 -> 14 worms / 60 neurons;  coverage 0.9 -> 20 worms / 56 neurons.
(6w/80n is the already-reported primary in phase2_results.json.)

Caveat: fewer neurons cuts the neuropeptide reference power (757 edges at 80n -> ~445 at 60n
-> ~357 at 56n), so a loss of significance can reflect reference power, not the effect vanishing.
Writes phase2_{W}w.json per config.
Run: PYTHONPATH=.:SBTG ./.venv/bin/python sid_elegans/biolag/run_phase2_worms.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import sid_elegans.biolag.run_phase2 as P
from sid_elegans.biolag import metric as M
from sid_elegans.biolag.references import build_registry
from sid_elegans.combined_data import load_combined
from sid_elegans.lagres import metric as mt

warnings.filterwarnings("ignore")
P.N_SURR = 60
P.N_BOOT = 60
OUT = Path(__file__).resolve().parents[2] / "sid_elegans" / "output" / "biolag"


def main():
    for cf in [0.8, 0.9]:
        X, names, _ = load_combined(coverage_frac=cf, complete_case=True, signal="deconv",
                                   verbose=True)
        W, N = len(X), len(names)
        refs = build_registry(names, min_specific_edges=10)
        mask = mt.eval_mask(N)
        npref = next((r for r in refs if r.name == "neuropeptide:all"), None)
        ne = npref.n_edges if npref else 0
        print(f"=== {W} worms / {N} neurons (neuropeptide edges={ne}) ===", flush=True)
        primary = P.run_block(X, names, refs, mask, f"{W}w-primary")
        gmode = P.run_block(M.remove_global_mode(X, n_pc=1), names, refs, mask, f"{W}w-gmode")
        out = {"n_worms": W, "n_neurons": N, "np_edges": ne, "n_surr": P.N_SURR,
               "n_boot": P.N_BOOT, "primary": primary, "global_mode_removed": gmode}
        json.dump(out, open(OUT / f"phase2_{W}w.json", "w"), indent=2, default=float)
        p = primary["neuropeptide:all"]; g = gmode["neuropeptide:all"]
        print(f"[{W}w/{N}n] neuropeptide:all  PRIMARY dBC={p['dBC_gain_obs']:+.2f} "
              f"(p={p['p_dBC_surrogate']:.3f}) BC_gain={p['BC_gain_obs']:+.2f} "
              f"[{p['BC_gain_boot']['lo']:+.2f},{p['BC_gain_boot']['hi']:+.2f}]  ||  "
              f"GLOBAL-REMOVED dBC={g['dBC_gain_obs']:+.2f} (p={g['p_dBC_surrogate']:.3f})", flush=True)
        print(f"          controls: src-var={primary['_control_source_variance_BC']:+.2f} "
              f"gap dBC={primary['structural:gap']['dBC_gain_obs']:+.2f} "
              f"chem dBC={primary['structural:chemical']['dBC_gain_obs']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
