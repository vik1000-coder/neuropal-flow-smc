"""Quantify partial-observation bias with a known synthetic graph.

A full VAR(2) graph over N_FULL neurons is generated and fitted once.  For each
nested observed subset, the analysis compares (a) that full fit restricted to the exact
observed edges with (b) a new fit in which the complement is hidden.  Holding
the scored edges fixed separates marginalization from graph dimension and node
composition. Edges among observed neurons belong to the planted graph, while
the partial fit's activity is confounded by the marginalized hidden neurons
(Thm B.5b bias).

Outputs default to `results/derived/partial_observation/`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.SyntheticTestingUtils import fit_minimal_with_timeout
# The experiment uses joint companion-system stabilization and preserves the
# planted edge support.
from experiments.synthetic_utils import hide_neurons, generate_var_data_stable as generate_var_data


def _offdiag(matrix):
    matrix = np.asarray(matrix)
    return matrix[~np.eye(len(matrix), dtype=bool)]


def weighted_metrics(truth_bool, score):
    """Threshold-free recovery metrics and top-k F1."""
    truth = _offdiag(truth_bool).astype(int)
    values = np.abs(_offdiag(score))
    valid = np.isfinite(values)
    truth, values = truth[valid], values[valid]
    if truth.sum() == 0 or truth.sum() == len(truth):
        return {"auroc": np.nan, "auprc": np.nan, "f1_topk": np.nan}
    k = int(truth.sum())
    selected = np.argpartition(values, -k)[-k:]
    predicted = np.zeros_like(truth)
    predicted[selected] = 1
    true_positive = int((predicted & truth).sum())
    return {
        "auroc": float(roc_auc_score(truth, values)),
        "auprc": float(average_precision_score(truth, values)),
        "f1_topk": float(true_positive / k),
    }


def binary_metrics(truth_bool, adjacency):
    """Precision, recall, and F1 for a Boolean adjacency estimate."""
    truth = _offdiag(truth_bool).astype(bool)
    predicted = _offdiag(adjacency).astype(bool)
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    precision = true_positive / (true_positive + false_positive) if predicted.sum() else 0.0
    recall = true_positive / (true_positive + false_negative) if truth.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision_signif": precision,
        "recall_signif": recall,
        "f1_signif": f1,
        "pred_edges": int(predicted.sum()),
    }

DEFAULT_OUT = ROOT / "results" / "derived" / "partial_observation"

# --- config ---
N_FULL   = 12
T        = 600
M_STIM   = 2
SEEDS    = [0, 1, 2, 3, 4, 5]
N_HIDDEN = [0, 2, 4]           # fractions 0, 0.167, 0.333 (observed 12, 10, 8)
EPOCHS   = 40


def fit_sbtg_lag1(X_list, seed):
    # The estimator's random_state controls NumPy folds but not
    # Torch initialization/DSM noise, so seed both explicitly for reproducible
    # sensitivity-analysis numbers.
    np.random.seed(seed)
    torch.manual_seed(seed)
    kwargs = dict(tune_hp=False, lags=[1], epochs=EPOCHS, n_folds=3,
                  hac_max_lag=5, fdr_alpha=0.1, fdr_method="by",
                  device="cpu", verbose=False, random_state=seed)
    return fit_minimal_with_timeout(kwargs, X_list)


def bootstrap_mean_ci(values, seed=2026, n_boot=10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        Xl_full, truth = generate_var_data(n=N_FULL, T=T, m_stim=M_STIM, noise_level="low", seed=seed)
        truth1_full = truth[1]
        full_result = fit_sbtg_lag1(Xl_full, seed)
        # One permutation per DGP makes the hidden sets nested:
        # hidden(2) is a subset of hidden(4).
        node_order = np.random.default_rng(seed + 10_000).permutation(N_FULL)

        for n_hidden in N_HIDDEN:
            frac = n_hidden / N_FULL
            hidden = node_order[:n_hidden]
            Xl_obs, truth1_obs, observed = hide_neurons(
                Xl_full,
                truth1_full,
                frac_hidden=frac,
                seed=seed,
                hidden_indices=hidden,
            )
            n_obs = len(observed)
            print(f"[seed {seed}] hidden={n_hidden} observed={n_obs} "
                  f"(true edges among observed={int(truth1_obs.sum())})", flush=True)

            full_mu = full_result.mu_hat[1][np.ix_(observed, observed)]
            full_adj = full_result.get_adjacency_for_lag(1)[np.ix_(observed, observed)]
            partial_result = (
                full_result if n_hidden == 0 else fit_sbtg_lag1(Xl_obs, seed)
            )
            partial_mu = partial_result.mu_hat[1]
            partial_adj = partial_result.get_adjacency_for_lag(1)
            estimate_gap = float(np.linalg.norm(partial_mu - full_mu, ord="fro"))

            variants = (
                ("full_fit_restricted", full_mu, full_adj),
                ("partial_fit", partial_mu, partial_adj),
            )
            for variant, mu1, adj1 in variants:
                metrics = dict(
                    method="SBTG",
                    variant=variant,
                    seed=seed,
                    training_seed=seed,
                    n_hidden=n_hidden,
                    frac_hidden=frac,
                    n_observed=n_obs,
                    hidden_indices=",".join(map(str, np.sort(hidden))),
                    observed_indices=",".join(map(str, observed)),
                    estimate_gap_fro=estimate_gap,
                )
                metrics.update(weighted_metrics(truth1_obs, mu1))
                metrics.update(binary_metrics(truth1_obs, adj1))
                rows.append(metrics)
                print(
                    f"    {variant:19s} AUROC={metrics['auroc']:.3f} "
                    f"F1sig={metrics['f1_signif']:.3f} "
                    f"prec={metrics['precision_signif']:.3f} "
                    f"edges={metrics['pred_edges']}",
                    flush=True,
                )

            pd.DataFrame(rows).to_csv(
                args.out_dir / "hidden_neuron_paired_metrics.csv", index=False
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "hidden_neuron_paired_metrics.csv", index=False)
    print(f"\nSaved {args.out_dir / 'hidden_neuron_paired_metrics.csv'}")

    summ = df.groupby(["variant", "frac_hidden"]).agg(
        auroc_m=("auroc", "mean"), auroc_s=("auroc", "std"),
        auprc_m=("auprc", "mean"), auprc_s=("auprc", "std"),
        f1sig_m=("f1_signif", "mean"), prec_m=("precision_signif", "mean"),
    ).reset_index()
    paired = (
        df.pivot(
            index=["seed", "frac_hidden"],
            columns="variant",
            values=["auroc", "auprc", "precision_signif", "f1_signif"],
        )
        .reset_index()
    )
    paired.columns = [
        "_".join(str(part) for part in col if part != "")
        for col in paired.columns.to_flat_index()
    ]
    delta_rows = []
    for frac, group in paired.groupby("frac_hidden"):
        row = {"frac_hidden": frac, "n_seeds": len(group)}
        for metric in ("auroc", "auprc", "precision_signif", "f1_signif"):
            delta = (
                group[f"{metric}_partial_fit"]
                - group[f"{metric}_full_fit_restricted"]
            )
            lo, hi = bootstrap_mean_ci(delta, seed=2026 + int(100 * frac))
            row.update(
                {
                    f"{metric}_delta_mean": float(delta.mean()),
                    f"{metric}_delta_sd": float(delta.std(ddof=1)),
                    f"{metric}_delta_ci_low": float(lo),
                    f"{metric}_delta_ci_high": float(hi),
                }
            )
        delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(args.out_dir / "hidden_neuron_paired_summary.csv", index=False)
    print("\n=== absolute means over seeds ===")
    print(summ.to_string(index=False))
    print("\n=== paired partial-minus-full changes ===")
    print(delta_df.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for variant in summ["variant"].unique():
        s = summ[summ["variant"] == variant]
        ax[0].errorbar(
            s["frac_hidden"],
            s["auroc_m"],
            yerr=s["auroc_s"],
            marker="o",
            capsize=3,
            label=variant.replace("_", " "),
        )
    ax[0].set_xlabel("fraction of neurons hidden"); ax[0].set_ylabel("lag-1 AUROC (observed submatrix)")
    ax[0].set_title("Same-edge recovery with and without hidden nodes")
    ax[0].axhline(0.5, ls="--", c="gray")
    ax[0].legend()

    ax[1].axhline(0, ls="--", c="gray")
    ax[1].errorbar(
        delta_df["frac_hidden"],
        delta_df["auroc_delta_mean"],
        yerr=[
            delta_df["auroc_delta_mean"] - delta_df["auroc_delta_ci_low"],
            delta_df["auroc_delta_ci_high"] - delta_df["auroc_delta_mean"],
        ],
        marker="o",
        capsize=3,
    )
    ax[1].set_xlabel("fraction of neurons hidden")
    ax[1].set_ylabel("paired AUROC change (partial - full)")
    ax[1].set_title("Marginalization effect on identical target edges")
    fig.tight_layout(); fig.savefig(args.out_dir / "hidden_neuron_curve.png", dpi=130)
    print(f"Saved {args.out_dir / 'hidden_neuron_curve.png'}")


if __name__ == "__main__":
    main()
