from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compatibility_neural_benchmark.onset_aware_analysis import (
    DEFAULT_OUTPUT,
    DEFAULT_RELEASE,
    METHOD_LABELS,
    _prepare_matrix,
)
from compatibility_neural_benchmark.postfreeze_external_analysis import load_references


DYNAMIC_METHODS = (
    "wide_flow_direct",
    "progressive_smc",
    "old_flow_direct",
    "terminal_smc",
)
STATIC_METHODS = ("sbtg_current", "sbtg_published")
NETWORKS = ("monoamine_all", "neuropeptide_all")


def eligible_source_auroc(
    matrices: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    """AUROC per lag, resampling source columns by their original identities."""
    values: list[float] = []
    d = labels.shape[0]
    for matrix in matrices:
        scores: list[np.ndarray] = []
        outcomes: list[np.ndarray] = []
        for source in sources:
            targets = np.arange(d) != source
            scores.append(np.abs(matrix[targets, source]))
            outcomes.append(labels[targets, source].astype(int))
        score = np.concatenate(scores)
        outcome = np.concatenate(outcomes)
        values.append(float(roc_auc_score(outcome, score)))
    return np.asarray(values)


def permute_targets_within_source(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = np.asarray(labels, int).copy()
    for source in np.flatnonzero(labels.any(axis=0)):
        targets = np.flatnonzero(np.arange(labels.shape[0]) != source)
        result[targets, source] = rng.permutation(result[targets, source])
    return result


def load_panels(archive: Path) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    with np.load(archive, allow_pickle=False) as data:
        neurons = data["neurons"].astype(str).tolist()
        lags = data["lags"].astype(int)
        phases = data["phases"].astype(str).tolist()
        panels: dict[str, dict[str, np.ndarray]] = {}
        onset = phases.index("onset")
        baseline = phases.index("baseline")
        for method in DYNAMIC_METHODS:
            matrices = data[f"{method}__matrices"].astype(float).mean(axis=0)
            for phase, array in (
                ("onset", matrices[onset]),
                ("onset_minus_baseline", matrices[onset] - matrices[baseline]),
            ):
                panels[f"{method}|{phase}"] = {
                    "method": method,
                    "phase": phase,
                    "lags": lags,
                    "matrices": np.stack([_prepare_matrix(x, "raw") for x in array]),
                }
        for method in STATIC_METHODS:
            method_lags = data[f"{method}__lags"].astype(int)
            matrices = data[f"{method}__matrices"].astype(float)
            panels[f"{method}|static"] = {
                "method": method,
                "phase": "static",
                "lags": method_lags,
                "matrices": np.stack([_prepare_matrix(x, "raw") for x in matrices]),
            }
    return neurons, panels


def run_inference(
    archive: Path,
    release: Path,
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    neurons, panels = load_panels(archive)
    _, networks = load_references(release, neurons)
    rng = np.random.default_rng(seed)
    lag_rows: list[dict] = []
    peak_rows: list[dict] = []
    for panel_name, panel in panels.items():
        for network_name in NETWORKS:
            labels = networks[network_name]
            eligible = np.flatnonzero(labels.any(axis=0))
            observed = eligible_source_auroc(panel["matrices"], labels, eligible)
            bootstrap = np.empty((n_bootstrap, len(panel["lags"])), dtype=float)
            peak_counts = np.zeros(len(panel["lags"]), dtype=int)
            for replicate in range(n_bootstrap):
                sampled = rng.choice(eligible, size=len(eligible), replace=True)
                bootstrap[replicate] = eligible_source_auroc(
                    panel["matrices"], labels, sampled
                )
                peak_counts[int(np.nanargmax(bootstrap[replicate]))] += 1
            for lag_index, lag in enumerate(panel["lags"]):
                lag_rows.append(
                    {
                        "panel": panel_name,
                        "method": panel["method"],
                        "method_label": METHOD_LABELS[panel["method"]],
                        "matrix_phase": panel["phase"],
                        "network": network_name,
                        "lag_frames": int(lag),
                        "lag_seconds": float(lag / 4.0),
                        "n_eligible_sources": int(len(eligible)),
                        "auroc": float(observed[lag_index]),
                        "source_bootstrap_ci_low": float(
                            np.quantile(bootstrap[:, lag_index], 0.025)
                        ),
                        "source_bootstrap_ci_high": float(
                            np.quantile(bootstrap[:, lag_index], 0.975)
                        ),
                        "source_bootstrap_probability_above_chance": float(
                            np.mean(bootstrap[:, lag_index] > 0.5)
                        ),
                        "bootstrap_peak_selection_rate": float(
                            peak_counts[lag_index] / n_bootstrap
                        ),
                    }
                )
            observed_peak_index = int(np.nanargmax(observed))
            null_maximum = np.empty(n_permutations, dtype=float)
            null_peak = np.empty(n_permutations, dtype=int)
            for replicate in range(n_permutations):
                permuted = permute_targets_within_source(labels, rng)
                values = eligible_source_auroc(
                    panel["matrices"], permuted, eligible
                )
                null_maximum[replicate] = float(np.max(values))
                null_peak[replicate] = int(np.argmax(values))
            peak_rows.append(
                {
                    "panel": panel_name,
                    "method": panel["method"],
                    "method_label": METHOD_LABELS[panel["method"]],
                    "matrix_phase": panel["phase"],
                    "network": network_name,
                    "n_eligible_sources": int(len(eligible)),
                    "n_lags_searched": int(len(panel["lags"])),
                    "observed_best_lag_frames": int(panel["lags"][observed_peak_index]),
                    "observed_best_lag_seconds": float(
                        panel["lags"][observed_peak_index] / 4.0
                    ),
                    "observed_peak_auroc": float(observed[observed_peak_index]),
                    "peak_lag_source_bootstrap_selection_rate": float(
                        peak_counts[observed_peak_index] / n_bootstrap
                    ),
                    "within_source_label_permutation_null_max_mean": float(
                        np.mean(null_maximum)
                    ),
                    "within_source_label_permutation_null_max_ci_low": float(
                        np.quantile(null_maximum, 0.025)
                    ),
                    "within_source_label_permutation_null_max_ci_high": float(
                        np.quantile(null_maximum, 0.975)
                    ),
                    "lag_max_adjusted_p_value": float(
                        (1 + np.sum(null_maximum >= observed[observed_peak_index]))
                        / (n_permutations + 1)
                    ),
                    "n_bootstrap": int(n_bootstrap),
                    "n_permutations": int(n_permutations),
                }
            )
    peaks = pd.DataFrame(peak_rows)
    order = np.argsort(peaks.lag_max_adjusted_p_value.to_numpy())
    ranked = peaks.lag_max_adjusted_p_value.to_numpy()[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    peaks["panel_bh_q_value"] = 1.0
    peaks.loc[peaks.index[order], "panel_bh_q_value"] = np.minimum(adjusted, 1.0)
    return pd.DataFrame(lag_rows), peaks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    archive = args.output / "aligned_phase_matrices.npz"
    lag_rows, peaks = run_inference(
        archive,
        args.release,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    lag_rows.to_csv(args.output / "bentley_lag_uncertainty.csv", index=False)
    peaks.to_csv(args.output / "bentley_best_lag_inference.csv", index=False)
    manifest = {
        "analysis": "Bentley eligible-source lag uncertainty and within-source lag-max null",
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "bootstrap_unit": "eligible source neuron class",
        "null": "shuffle target labels within each eligible source; preserve source edge counts; compare maximum AUROC over the method-native lag grid",
        "multiplicity": "lag maximum controlled within panel; BH q-value across method-phase-network panels",
    }
    (args.output / "bentley_lag_inference_protocol.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
