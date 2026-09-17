"""Scoring of a directed coupling matrix against a ground-truth connectome.

Reproduces the SBTG repo's metric definitions exactly (verified against
pipeline/15_multilag_analysis.py): score = ``|coupling[offdiag]|``, ground truth =
``(A[offdiag] > 0)``, sklearn AUROC / AUPRC, Spearman of ``|coupling|`` vs weights, and
F1 at ground-truth-prevalence density. Off-diagonal only; NaNs dropped consistently.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _offdiag(A):
    n = A.shape[0]
    return A[~np.eye(n, dtype=bool)]


def score_matrix(coupling: np.ndarray, gt: np.ndarray, binarize_gt: bool = True) -> dict:
    """AUROC/AUPRC/Spearman/F1 of ``|coupling|`` vs ground truth (off-diagonal)."""
    y_score = np.abs(_offdiag(coupling))
    y_gt = _offdiag(gt).astype(float)
    y_true = (y_gt > 0).astype(int) if binarize_gt else y_gt

    valid = np.isfinite(y_score) & np.isfinite(y_gt)
    y_score, y_true, y_gt = y_score[valid], y_true[valid], y_gt[valid]

    out = {"n_eval": int(len(y_true)), "n_pos": int(y_true.sum()),
           "auroc": np.nan, "auprc": np.nan, "spearman": np.nan, "f1": np.nan}
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return out
    out["auroc"] = float(roc_auc_score(y_true, y_score))
    out["auprc"] = float(average_precision_score(y_true, y_score))
    # Spearman of |score| vs continuous ground-truth weight
    if np.std(y_score) > 1e-10 and np.std(y_gt) > 1e-10:
        rho, _ = spearmanr(y_score, y_gt)
        out["spearman"] = float(rho) if np.isfinite(rho) else np.nan
    # F1 at ground-truth prevalence density (top-k by |score|)
    k = int(y_true.sum())
    thr = np.sort(y_score)[::-1][min(k, len(y_score) - 1)]
    y_pred = (y_score > thr).astype(int)
    if y_pred.sum() > 0:
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return out
