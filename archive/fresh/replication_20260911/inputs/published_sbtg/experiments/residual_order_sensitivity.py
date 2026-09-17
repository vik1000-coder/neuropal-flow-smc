#!/usr/bin/env python3
"""Recording-held-out residual-correlation sensitivity across VAR orders.

This diagnostic asks how strongly the empirical residual-correlation matrix depends
on the amount of linear temporal structure removed before correlations are measured.
It deliberately does not identify the structural innovation covariance Omega.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import (DEFAULT_DATASET, DEFAULT_SBTG, load_dataset,
                    load_monoamine_networks, load_reference_networks, load_sbtg,
                    partial_rank_correlation, reorder_square, safe_auroc,
                    safe_spearman)
from experiments.residual_alignment import make_var_rows, cross_fitted_residuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sbtg-file", type=Path, default=DEFAULT_SBTG)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def corr_statistics(corr: np.ndarray) -> dict[str, float]:
    offdiag = ~np.eye(len(corr), dtype=bool)
    values = np.abs(corr[offdiag])
    return {
        "mean_abs_residual_corr": float(values.mean()),
        "median_abs_residual_corr": float(np.median(values)),
        "p90_abs_residual_corr": float(np.quantile(values, 0.90)),
        "p95_abs_residual_corr": float(np.quantile(values, 0.95)),
        "max_abs_residual_corr": float(values.max()),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces, names, row_meta = load_dataset(args.dataset_dir)
    row_meta.to_csv(args.out_dir / "recording_rows.csv", index=False)

    sbtg_mu, _, sbtg_names = load_sbtg(args.sbtg_file)
    if sbtg_names != names:
        sbtg_mu = {
            lag: reorder_square(matrix, sbtg_names, names)[0]
            for lag, matrix in sbtg_mu.items()
        }
    references = load_reference_networks(names)
    offdiag = ~np.eye(len(names), dtype=bool)
    for network, labels in load_monoamine_networks(names).items():
        references[network] = (labels, offdiag)

    summary_rows: list[dict] = []
    alignment_rows: list[dict] = []
    exclusion_rows: list[dict] = []

    for order in args.orders:
        order_dir = args.out_dir / f"order_{order:02d}"
        order_dir.mkdir(parents=True, exist_ok=True)
        result_file = order_dir / "residual_correlation.npz"
        folds_file = order_dir / "fold_predictions.csv"

        if result_file.exists() and folds_file.exists():
            saved = np.load(result_file, allow_pickle=False)
            corr = np.asarray(saved["corr"], dtype=float)
            n_residual_rows = int(saved["n_residual_rows"])
            folds = pd.read_csv(folds_file)
            status = "reused validated artifact"
        else:
            X, y, groups = make_var_rows(traces, order)
            residual, folds, tuning = cross_fitted_residuals(
                X, y, groups, len(traces), args.outer_folds, 2,
                [args.ridge_alpha], args.seed,
            )
            corr = np.corrcoef(residual, rowvar=False)
            n_residual_rows = len(residual)
            folds.to_csv(folds_file, index=False)
            tuning.to_csv(order_dir / "fixed_alpha.csv", index=False)
            np.savez_compressed(
                result_file, corr=corr, abs_corr=np.abs(corr),
                neuron_names=np.asarray(names), order=order,
                n_residual_rows=n_residual_rows,
            )
            status = "fit locally"
            del X, y, groups, residual
            gc.collect()

        abs_corr = np.abs(corr)
        summary_rows.append({
            "var_order": order,
            "n_recordings": len(traces),
            "n_neurons": len(names),
            "n_residual_rows": n_residual_rows,
            "ridge_alpha": args.ridge_alpha,
            "heldout_mse_mean": float(folds["mse"].mean()),
            "constant_mean_mse_mean": float(folds["constant_mean_mse"].mean()),
            "artifact_status": status,
            **corr_statistics(corr),
        })

        for reference, (labels, mask) in references.items():
            residual_reference = safe_spearman(
                abs_corr[mask], labels[mask].astype(float)
            )
            alignment_rows.append({
                "var_order": order,
                "comparison": "residual_reference",
                "reference": reference,
                "lag": np.nan,
                "spearman": residual_reference,
                "partial_spearman": np.nan,
                "auroc": safe_auroc(labels[mask], abs_corr[mask]),
                "n_pairs": int(mask.sum()),
            })
            for lag, matrix in sbtg_mu.items():
                score = np.abs(matrix)
                alignment_rows.append({
                    "var_order": order,
                    "comparison": "sbtg_reference",
                    "reference": reference,
                    "lag": lag,
                    "spearman": safe_spearman(
                        score[mask], labels[mask].astype(float)
                    ),
                    "partial_spearman": partial_rank_correlation(
                        score[mask], labels[mask].astype(float), abs_corr[mask]
                    ),
                    "auroc": safe_auroc(labels[mask], score[mask]),
                    "n_pairs": int(mask.sum()),
                })
                for excluded_fraction in (0.0, 0.10, 0.25):
                    kept = mask.copy()
                    if excluded_fraction:
                        cutoff = np.quantile(
                            abs_corr[mask], 1.0 - excluded_fraction
                        )
                        kept &= abs_corr < cutoff
                    exclusion_rows.append({
                        "var_order": order,
                        "reference": reference,
                        "lag": lag,
                        "excluded_top_residual_fraction": excluded_fraction,
                        "n_pairs": int(kept.sum()),
                        "n_positive": int(labels[kept].sum()),
                        "auroc": safe_auroc(labels[kept], score[kept]),
                    })

        for lag, matrix in sbtg_mu.items():
            alignment_rows.append({
                "var_order": order,
                "comparison": "sbtg_residual",
                "reference": "residual_proxy",
                "lag": lag,
                "spearman": safe_spearman(
                    np.abs(matrix)[offdiag], abs_corr[offdiag]
                ),
                "partial_spearman": np.nan,
                "auroc": np.nan,
                "n_pairs": int(offdiag.sum()),
            })

        pd.DataFrame(summary_rows).to_csv(
            args.out_dir / "residual_order_summary.csv", index=False
        )
        pd.DataFrame(alignment_rows).to_csv(
            args.out_dir / "residual_order_alignment.csv", index=False
        )
        pd.DataFrame(exclusion_rows).to_csv(
            args.out_dir / "residual_order_exclusions.csv", index=False
        )

    manifest = {
        "status": "complete",
        "question": (
            "How sensitive is empirical residual dependence to the amount of "
            "linear temporal structure removed?"
        ),
        "orders": args.orders,
        "outer_folds": args.outer_folds,
        "ridge_alpha": args.ridge_alpha,
        "seed": args.seed,
        "data_unit": "prepared recording row",
        "limitations": [
            "Residual correlations are a model-dependent diagnostic, not structural Omega.",
            "Ridge VAR models do not remove nonlinear or hidden-state dynamics.",
            "Cook/Randi alignment cannot distinguish genuine propagation from confounding.",
        ],
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
