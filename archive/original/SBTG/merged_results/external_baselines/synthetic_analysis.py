#!/usr/bin/env python3
"""
Evaluate external deep-learning baselines on controlled synthetic data.

Generates VAR and Hawkes datasets with known ground-truth connectivity,
runs NRI / NetFormer / LINT, and reports AUROC / AUPRC / F1 / Correlation.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.SyntheticTestingUtils import generate_var_data, generate_hawkes_like_data
from merged_results.external_baselines.external_analysis import (
    run_nri,
    run_netformer,
    run_lint,
)
from merged_results.external_baselines.evaluate_external import compute_all_metrics


def _normalize_windows(X: np.ndarray) -> np.ndarray:
    """Normalize [B, N, T] windows for numerical stability in external models."""
    if X.ndim != 3:
        return X
    Xc = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    Xf = Xc.astype(np.float32, copy=True)
    # Per-neuron normalization over batch/time axes.
    mean = Xf.mean(axis=(0, 2), keepdims=True)
    std = Xf.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    Xf = (Xf - mean) / std
    # Clip outliers to keep training stable on high-dimensional VAR cases.
    Xf = np.clip(Xf, -8.0, 8.0)
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=8.0, neginf=-8.0)
    return Xf


def main():
    parser = argparse.ArgumentParser(
        description="Run external deep-learning baselines on synthetic data."
    )
    parser.add_argument("--n-neurons", type=int, default=20, help="Number of neurons.")
    parser.add_argument("--T", type=int, default=400, help="Sequence length.")
    parser.add_argument("--m-stim", type=int, default=1, help="Number of stimuli.")
    parser.add_argument(
        "--noise-level",
        type=str,
        default="low",
        choices=["low", "high"],
        help="Noise level for synthetic generation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--window-size", type=int, default=10, help="Window length.")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for NRI/NetFormer.")
    parser.add_argument("--lint-epochs", type=int, default=60, help="Epochs for LINT.")
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=["VAR", "Hawkes"],
        help="Synthetic families to evaluate: VAR and/or Hawkes.",
    )
    args = parser.parse_args()

    print("Evaluating external baselines on synthetic data ...")

    n_neurons = args.n_neurons
    T = args.T
    m_stim = args.m_stim
    noise_level = args.noise_level
    seed = args.seed
    window_size = args.window_size
    epochs = args.epochs

    results = []

    all_families = {
        "VAR": generate_var_data,
        "Hawkes": generate_hawkes_like_data,
    }
    selected_families = [f for f in args.families if f in all_families]
    for family in selected_families:
        gen_func = all_families[family]
        print(f"\nGenerating {family} dataset ...")
        X_list, truth_dict = gen_func(
            n=n_neurons, T=T, m_stim=m_stim,
            noise_level=noise_level, seed=seed,
        )

        W_true = truth_dict[1]

        windows = []
        for x in X_list:
            if x.shape[0] < window_size:
                continue
            for i in range(x.shape[0] - window_size + 1):
                windows.append(x[i : i + window_size, :])

        X = np.stack(windows).transpose((0, 2, 1))  # [B, N, T]
        X = _normalize_windows(X)

        for method_name, func in [
            ("NRI", lambda X_: run_nri(X_, epochs=epochs)),
            ("NetFormer", lambda X_: run_netformer(X_, epochs=epochs)),
            ("LINT", lambda X_: run_lint(X_, epochs=args.lint_epochs, rank=5)),
        ]:
            adj = func(X)

            if adj is not None:
                metrics = compute_all_metrics(adj, W_true)
                print(f"  [{family}] {method_name}  "
                      f"AUROC={metrics['AUROC']:.4f}  "
                      f"AUPRC={metrics['AUPRC']:.4f}  "
                      f"F1={metrics['F1']:.4f}  "
                      f"Corr={metrics['Correlation']:.4f}")
                row = {"Dataset": family, "Method": method_name}
                row.update(metrics)
                results.append(row)

    df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("SYNTHETIC BENCHMARK")
    print("=" * 50)
    print(df.to_markdown(index=False))

    out_csv = (PROJECT_ROOT / "merged_results" / "external_baselines"
               / "evaluation_synthetic.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
