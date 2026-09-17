#!/usr/bin/env python3
"""Paired structured-noise synthetic stress test with structured innovation covariance.

The same stationary VAR(2) graph, initial states, and standard-normal innovation
draws are reused across covariance conditions for a seed.  The experiment tests
recovery at true lags 1--2 and false-positive calls at null lags 3 and 5.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import spearmanr, t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


N = 12
T = 500
N_SEGMENTS = 2
LAGS = (1, 2, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--target-correlations", type=float, nargs="+",
                        default=[0.025, 0.06, 0.20])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def offdiag(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix)[~np.eye(len(matrix), dtype=bool)]


def mean_abs_offdiag(matrix: np.ndarray) -> float:
    return float(np.mean(np.abs(offdiag(matrix))))


def normalize_to_correlation(matrix: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.maximum(np.diag(matrix), 1e-12))
    corr = matrix / np.outer(diagonal, diagonal)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr


def solve_monotone_parameter(builder, target: float, initial_high: float = 1.0,
                             maximum_high: float = 32.0) -> tuple[np.ndarray, float]:
    if target <= 0:
        return builder(0.0), 0.0
    low, high = 0.0, initial_high
    while mean_abs_offdiag(builder(high)) < target and high < maximum_high:
        high *= 2.0
    if mean_abs_offdiag(builder(high)) < target:
        return builder(high), high
    for _ in range(60):
        middle = (low + high) / 2.0
        if mean_abs_offdiag(builder(middle)) < target:
            low = middle
        else:
            high = middle
    value = (low + high) / 2.0
    return builder(value), value


def build_correlation(kind: str, target: float, A1: np.ndarray,
                      A2: np.ndarray) -> tuple[np.ndarray, float]:
    if kind == "isotropic":
        return np.eye(len(A1)), 0.0
    if kind == "equicorr":
        matrix = (1.0 - target) * np.eye(len(A1)) + target * np.ones_like(A1)
        return matrix, target
    if kind == "ar1":
        indices = np.arange(len(A1))
        return solve_monotone_parameter(
            lambda rho: rho ** np.abs(indices[:, None] - indices[None, :]),
            target, initial_high=0.5, maximum_high=0.999999,
        )
    if kind == "graph_diffusion":
        support = (np.abs(A1) > 1e-12) | (np.abs(A2) > 1e-12)
        skeleton = (support | support.T).astype(float)
        np.fill_diagonal(skeleton, 0.0)
        spectral_norm = max(float(np.linalg.norm(skeleton, ord=2)), 1.0)
        generator = skeleton / spectral_norm
        return solve_monotone_parameter(
            lambda beta: normalize_to_correlation(expm(beta * generator)),
            target, initial_high=1.0, maximum_high=32.0,
        )
    raise ValueError(f"Unknown covariance kind: {kind}")


def graph_and_latent_draws(seed: int):
    from pipeline.SyntheticTestingUtils import _make_sparse_matrix
    from experiments.synthetic_utils import stabilize_var

    graph_rng = np.random.default_rng(seed)
    A1 = _make_sparse_matrix(N, sparsity=0.1, scale=0.8, rng=graph_rng)
    A2 = _make_sparse_matrix(N, sparsity=0.1, scale=0.5, rng=graph_rng)
    A1, A2, radius_before, radius_after = stabilize_var(A1, A2, target=0.9)

    noise_rng = np.random.default_rng(seed + 100_000)
    initial = noise_rng.normal(size=(N_SEGMENTS, 2, N))
    latent = noise_rng.standard_normal(size=(N_SEGMENTS, T - 2, N))
    return A1, A2, radius_before, radius_after, initial, latent


def simulate(A1: np.ndarray, A2: np.ndarray, correlation: np.ndarray,
             initial: np.ndarray, latent: np.ndarray) -> list[np.ndarray]:
    sigma = 0.1
    covariance = sigma ** 2 * correlation
    cholesky = np.linalg.cholesky(covariance + 1e-10 * np.eye(N))
    segments: list[np.ndarray] = []
    for segment in range(N_SEGMENTS):
        X = np.zeros((T, N), dtype=float)
        X[:2] = initial[segment]
        for t in range(2, T):
            innovation = cholesky @ latent[segment, t - 2]
            X[t] = A1 @ X[t - 1] + A2 @ X[t - 2] + innovation
        segments.append(X)
    return segments


def weighted_metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = offdiag(truth).astype(bool)
    values = np.abs(offdiag(score))
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auroc": np.nan, "auprc": np.nan, "f1_topk": np.nan}
    k = int(labels.sum())
    selected = np.argpartition(values, -k)[-k:]
    prediction = np.zeros(len(values), dtype=bool)
    prediction[selected] = True
    true_positive = int((prediction & labels).sum())
    f1 = 2.0 * true_positive / (prediction.sum() + labels.sum())
    return {
        "auroc": float(roc_auc_score(labels.astype(int), values)),
        "auprc": float(average_precision_score(labels.astype(int), values)),
        "f1_topk": float(f1),
    }


def binary_metrics(truth: np.ndarray, significant: np.ndarray) -> dict[str, float]:
    labels = offdiag(truth).astype(bool)
    prediction = offdiag(significant).astype(bool)
    true_positive = int((prediction & labels).sum())
    false_positive = int((prediction & ~labels).sum())
    false_negative = int((~prediction & labels).sum())
    precision = true_positive / (true_positive + false_positive) if prediction.sum() else 0.0
    recall = true_positive / (true_positive + false_negative) if labels.sum() else np.nan
    f1 = (2.0 * precision * recall / (precision + recall)
          if np.isfinite(recall) and precision + recall > 0 else 0.0)
    return {
        "significant_edges": int(prediction.sum()),
        "true_positive_edges": true_positive,
        "false_positive_edges": false_positive,
        "precision_significant": float(precision),
        "recall_significant": float(recall),
        "f1_significant": float(f1),
    }


def condition_id(kind: str, target: float) -> str:
    return f"{kind}_corr_{target:.3f}".replace(".", "p")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    os.replace(temporary, path)


def run_task(task: dict) -> dict:
    import torch
    from pipeline.SyntheticTestingUtils import fit_minimal_with_timeout

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    seed = int(task["seed"])
    kind = str(task["kind"])
    target = float(task["target"])
    output_file = Path(task["output_file"])
    if output_file.exists():
        with output_file.open() as handle:
            saved = json.load(handle)
        if saved.get("status") == "complete":
            return saved

    started = time.time()
    A1, A2, radius_before, radius_after, initial, latent = graph_and_latent_draws(seed)
    correlation, covariance_parameter = build_correlation(kind, target, A1, A2)
    X_list = simulate(A1, A2, correlation, initial, latent)

    estimator_kwargs = {
        "tune_hp": False,
        "lags": list(LAGS),
        "epochs": int(task["epochs"]),
        "n_folds": 3,
        "hac_max_lag": 5,
        "fdr_alpha": 0.10,
        "fdr_method": "by",
        "device": str(task["device"]),
        "verbose": False,
        "random_state": seed,
    }
    result = fit_minimal_with_timeout(estimator_kwargs, X_list)
    truths = {
        1: np.abs(A1) > 1e-12,
        2: np.abs(A2) > 1e-12,
        3: np.zeros_like(A1, dtype=bool),
        5: np.zeros_like(A1, dtype=bool),
    }
    union_truth = truths[1] | truths[2]
    covariance_alignment = weighted_metrics(union_truth, correlation)["auroc"]

    rows = []
    for lag in LAGS:
        score = np.asarray(result.mu_hat[lag], dtype=float)
        significant = np.asarray(result.significant[lag], dtype=bool)
        truth = truths[lag]
        covariance_values = np.abs(offdiag(correlation))
        score_values = np.abs(offdiag(score))
        score_covariance_spearman = (
            float(spearmanr(score_values, covariance_values).statistic)
            if np.std(covariance_values) > 1e-12 else np.nan
        )
        row = {
            "seed": seed,
            "covariance_kind": kind,
            "target_mean_abs_corr": target,
            "actual_mean_abs_corr": mean_abs_offdiag(correlation),
            "covariance_parameter": covariance_parameter,
            "covariance_union_graph_auroc": covariance_alignment,
            "lag": lag,
            "lag_is_null": lag not in (1, 2),
            "mean_abs_score": float(np.mean(score_values)),
            "p95_abs_score": float(np.quantile(score_values, 0.95)),
            "score_covariance_spearman": score_covariance_spearman,
            **binary_metrics(truth, significant),
        }
        if lag in (1, 2):
            row.update(weighted_metrics(truth, score))
        else:
            row.update({"auroc": np.nan, "auprc": np.nan, "f1_topk": np.nan})
            row["null_lag_false_positive_rate"] = (
                row["significant_edges"] / (N * (N - 1))
            )
        rows.append(row)

    payload = {
        "status": "complete",
        "condition_id": condition_id(kind, target),
        "seed": seed,
        "covariance_kind": kind,
        "target_mean_abs_corr": target,
        "actual_mean_abs_corr": mean_abs_offdiag(correlation),
        "covariance_parameter": covariance_parameter,
        "companion_radius_before": radius_before,
        "companion_radius_after": radius_after,
        "elapsed_seconds": time.time() - started,
        "rows": rows,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output_file, payload)
    return payload


def aggregate(payloads: list[dict], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [row for payload in payloads for row in payload["rows"]]
    frame = pd.DataFrame(rows).sort_values(
        ["covariance_kind", "target_mean_abs_corr", "seed", "lag"]
    )
    frame.to_csv(out_dir / "residual_structured_noise_metrics.csv", index=False)
    numeric = [
        "actual_mean_abs_corr", "covariance_union_graph_auroc",
        "mean_abs_score", "p95_abs_score", "score_covariance_spearman",
        "significant_edges", "true_positive_edges", "false_positive_edges",
        "precision_significant", "recall_significant", "f1_significant",
        "auroc", "auprc", "f1_topk", "null_lag_false_positive_rate",
    ]
    summary = frame.groupby(
        ["covariance_kind", "target_mean_abs_corr", "lag", "lag_is_null"],
        dropna=False,
    )[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_dir / "residual_structured_noise_summary.csv", index=False)

    baseline = frame.loc[
        frame.covariance_kind == "isotropic",
        ["seed", "lag", *numeric],
    ].copy()
    baseline = baseline.rename(columns={name: f"{name}_isotropic" for name in numeric})
    paired = frame.loc[frame.covariance_kind != "isotropic"].merge(
        baseline, on=["seed", "lag"], how="left", validate="many_to_one"
    )
    delta_metrics = [
        "auroc", "auprc", "f1_topk", "significant_edges",
        "false_positive_edges", "precision_significant", "f1_significant",
        "mean_abs_score", "p95_abs_score", "null_lag_false_positive_rate",
    ]
    for metric in delta_metrics:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_isotropic"]
    paired.to_csv(out_dir / "residual_structured_noise_paired_differences.csv", index=False)

    paired_summary_rows = []
    for keys, group in paired.groupby(
        ["covariance_kind", "target_mean_abs_corr", "lag", "lag_is_null"],
        dropna=False,
    ):
        row = dict(zip(
            ["covariance_kind", "target_mean_abs_corr", "lag", "lag_is_null"],
            keys,
        ))
        for metric in delta_metrics:
            values = group[f"delta_{metric}"].dropna().to_numpy(dtype=float)
            row[f"delta_{metric}_n"] = len(values)
            row[f"delta_{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"delta_{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
            if len(values) > 1:
                half_width = float(
                    t.ppf(0.975, len(values) - 1)
                    * values.std(ddof=1) / np.sqrt(len(values))
                )
                row[f"delta_{metric}_ci95_low"] = float(values.mean() - half_width)
                row[f"delta_{metric}_ci95_high"] = float(values.mean() + half_width)
            else:
                row[f"delta_{metric}_ci95_low"] = np.nan
                row[f"delta_{metric}_ci95_high"] = np.nan
        paired_summary_rows.append(row)
    pd.DataFrame(paired_summary_rows).to_csv(
        out_dir / "residual_structured_noise_paired_summary.csv", index=False
    )
    return frame, summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_dir = args.out_dir / "tasks"
    task_dir.mkdir(exist_ok=True)

    conditions = [("isotropic", 0.0)]
    conditions.extend(
        (kind, target)
        for kind in ("equicorr", "ar1", "graph_diffusion")
        for target in args.target_correlations
    )
    tasks = []
    for seed in args.seeds:
        for kind, target in conditions:
            tasks.append({
                "seed": seed,
                "kind": kind,
                "target": target,
                "epochs": args.epochs,
                "device": args.device,
                "output_file": str(
                    task_dir / f"seed_{seed:02d}_{condition_id(kind, target)}.json"
                ),
            })

    started = time.time()
    payloads: list[dict] = []

    def record_payload(completed: int, task: dict, payload: dict) -> None:
        payloads.append(payload)
        aggregate(payloads, args.out_dir)
        print(
            f"[{completed}/{len(tasks)}] seed={task['seed']} "
            f"condition={condition_id(task['kind'], task['target'])} "
            f"elapsed={payload['elapsed_seconds']:.1f}s",
            flush=True,
        )

    if args.workers == 1:
        # This path also works in restricted environments where POSIX semaphore
        # discovery for ProcessPoolExecutor is unavailable.
        for completed, task in enumerate(tasks, start=1):
            try:
                payload = run_task(task)
            except Exception as error:
                failure = {
                    "status": "failed", "error": repr(error), **task,
                }
                atomic_json(Path(task["output_file"]).with_suffix(".failed.json"), failure)
                raise
            record_payload(completed, task, payload)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = {executor.submit(run_task, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    payload = future.result()
                except Exception as error:
                    failure = {
                        "status": "failed", "error": repr(error), **task,
                    }
                    atomic_json(Path(task["output_file"]).with_suffix(".failed.json"), failure)
                    raise
                record_payload(completed, task, payload)

    frame, summary = aggregate(payloads, args.out_dir)
    manifest = {
        "status": "complete",
        "n_tasks": len(tasks),
        "seeds": args.seeds,
        "conditions": [
            {"kind": kind, "target_mean_abs_corr": target}
            for kind, target in conditions
        ],
        "lags": list(LAGS),
        "true_direct_lags": [1, 2],
        "null_direct_lags": [3, 5],
        "epochs": args.epochs,
        "workers": args.workers,
        "elapsed_seconds": time.time() - started,
        "pairing": (
            "Within a seed, graph, initial states, and standard-normal innovation "
            "draws are identical across covariance conditions."
        ),
        "limitations": [
            "Small stationary linear VAR(2) graphs do not reproduce the full neural setting.",
            "Graph-diffusion covariance is a controlled anatomy-aligned stress test, not an empirical estimate of Omega.",
            (
                f"Fixed hyperparameters and {len(args.seeds)} paired seeds quantify "
                "sensitivity, not complete training uncertainty."
            ),
        ],
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    print(summary.to_string(index=False))
    print(f"Completed {len(frame)} lag-level rows in {manifest['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
