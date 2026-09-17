r"""Committed validation of ACMMA: on the complete-case corner (no missing data), the available-case
moment assembly must reproduce the standard closed-form SID fit EXACTLY (it is the same estimator,
just fed the data differently). Backs the "correlation 1.000" claim in ALLDATA_RESULT.md / README.

Run: PYTHONPATH=.:SBTG ./.venv/bin/python sid_elegans/biolag/validate_acmma.py
Expected: Spearman = Pearson = 1.000 for mean and gain at each lag/sigma (match requires
estimator heldout_frac=0.0 and ACMMA shrink=0.0, i.e. same rows, no PSD shrinkage).
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.stats import spearmanr
from sid_elegans.acmma import fit_acmma_connectome
from sid_elegans.combined_data import load_combined
from sid_elegans.estimator import fit_distributional_connectome

warnings.filterwarnings("ignore")


def main():
    X, names, _ = load_combined(coverage_frac=0.6, complete_case=True, signal="deconv", verbose=False)
    off = ~np.eye(len(names), dtype=bool)
    print(f"ACMMA vs complete-case SID on {len(X)} worms / {len(names)} neurons "
          f"(matched: heldout=0, shrink=0):")
    # (A) Hyvarinen (sigma=0): EXACT reproduction is the load-bearing claim -> assert 1.000.
    ok = True
    for lag in (5, 10):
        ref = fit_distributional_connectome(X, names, lag=lag, ridge=1e-2, heldout_frac=0.0).matrices
        m, _ = fit_acmma_connectome(X, names, lag=lag, ridge=1e-2, min_triple=8, shrink=0.0)
        for ch in ("mean", "gain"):
            sp = spearmanr(np.abs(m[ch][off]), np.abs(ref[ch][off])).statistic
            pe = np.corrcoef(m[ch][off], ref[ch][off])[0, 1]
            flag = "OK" if (sp > 0.999 and pe > 0.999) else "FAIL"
            ok &= flag == "OK"
            print(f"  [Hyvarinen sigma=0] lag={lag} {ch:4s}: spearman={sp:.4f} pearson={pe:.4f}  {flag}")
    # (B) DSM (sigma>0): informational only. ACMMA-DSM is the DETERMINISTIC expected form; the
    # closed-form reference draws random noise, so they differ by construction (corr ~0.75, not 1.0).
    ref = fit_distributional_connectome(X, names, lag=5, ridge=1e-2, heldout_frac=0.0, sigma_frac=0.5).matrices
    m, _ = fit_acmma_connectome(X, names, lag=5, ridge=1e-2, min_triple=8, shrink=0.0, sigma_frac=0.5)
    r = spearmanr(np.abs(m["gain"][off]), np.abs(ref["gain"][off])).statistic
    print(f"  [DSM sigma=0.5, informational] gain spearman={r:.3f} (expected ~0.75: deterministic "
          f"vs noise-draw DSM -- NOT a failure)")
    print("\nVALIDATION", "PASSED (Hyvarinen exact)" if ok else "FAILED")


if __name__ == "__main__":
    main()
