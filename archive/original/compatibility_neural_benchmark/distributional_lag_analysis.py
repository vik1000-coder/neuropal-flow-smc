from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compatibility_neural_benchmark.distributional_lag_audit import METRICS


PRIMARY_METRIC = "wasserstein1"
SIGNED_METRICS = ("mean_shift", "log_sd_shift", "tail_probability_shift")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 3 or np.std(first[finite]) <= 1e-12 or np.std(second[finite]) <= 1e-12:
        return 0.0
    return float(spearmanr(first[finite], second[finite]).statistic)


def offdiagonal(matrix: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    mask = np.ones(matrix.shape, dtype=bool)
    for row, source in enumerate(source_indices):
        if source < matrix.shape[1]:
            mask[row, source] = False
    return matrix[mask]


def source_permutation_p(
    first: np.ndarray,
    second: np.ndarray,
    source_indices: np.ndarray,
    *,
    seed: int,
    permutations: int = 499,
) -> tuple[float, float, float]:
    observed = safe_spearman(offdiagonal(first, source_indices), offdiagonal(second, source_indices))
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = second[rng.permutation(len(second))]
        null[index] = safe_spearman(
            offdiagonal(first, source_indices), offdiagonal(shuffled, source_indices)
        )
    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    return observed, p_value, float(np.quantile(null, 0.95))


def worm_bootstrap_interval(
    values: np.ndarray, *, seed: int, draws: int = 4000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame, dict[tuple, np.ndarray]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("distributional audit run is incomplete")
    rows: list[dict[str, object]] = []
    matrices: dict[tuple, np.ndarray] = {}
    expected_fingerprint = manifest["stimulus_schema_fingerprint"]
    for path in sorted((run_dir / "responses").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(data["status"].item()) != "complete":
                raise RuntimeError(f"incomplete response archive: {path}")
            if str(data["stimulus_schema_fingerprint"].item()) != expected_fingerprint:
                raise RuntimeError(f"stimulus schema mismatch: {path}")
            fold = int(data["fold"].item())
            model_seed = int(data["model_seed"].item())
            states = data["state_names"].astype(str)
            lags = data["lag_frames"].astype(int)
            worms = data["heldout_worm_ids"].astype(str)
            sources = data["source_indices"].astype(int)
            for worm_position, worm_id in enumerate(worms):
                for state_position, state in enumerate(states):
                    for lag_position, lag in enumerate(lags):
                        key_base = (worm_id, model_seed, state, int(lag))
                        for metric in METRICS:
                            matrix = data[metric][worm_position, state_position, lag_position].astype(float)
                            matrices[(*key_base, metric)] = matrix
                        rows.append(
                            {
                                "worm_id": worm_id,
                                "fold": fold,
                                "model_seed": model_seed,
                                "state": state,
                                "lag_frames": int(lag),
                                "lag_seconds": float(lag) / 4.0,
                                "conditional_iqr_mean": float(
                                    data["conditional_iqr"][worm_position, state_position, lag_position].mean()
                                ),
                            }
                        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no response archives found")
    source_indices = np.asarray(
        [manifest["target_neurons"].index(name) for name in manifest["source_neurons"]],
        dtype=np.int64,
    )
    manifest["source_indices"] = source_indices.tolist()
    return manifest, frame, matrices


def aggregate_matrix(
    frame: pd.DataFrame,
    matrices: dict[tuple, np.ndarray],
    *,
    state: str,
    lag: int,
    metric: str,
    seeds: set[int] | None = None,
    worms: set[str] | None = None,
) -> np.ndarray:
    use = frame[(frame.state == state) & (frame.lag_frames == lag)]
    if seeds is not None:
        use = use[use.model_seed.isin(seeds)]
    if worms is not None:
        use = use[use.worm_id.isin(worms)]
    values = [
        matrices[(row.worm_id, int(row.model_seed), state, lag, metric)]
        for row in use.itertuples()
    ]
    if not values:
        raise RuntimeError("empty aggregation cell")
    return np.mean(values, axis=0)


def worm_scalar_values(
    frame: pd.DataFrame,
    matrices: dict[tuple, np.ndarray],
    source_indices: np.ndarray,
    *,
    state: str,
    lag: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    worms: list[str] = []
    use = frame[(frame.state == state) & (frame.lag_frames == lag)]
    for worm_id, group in use.groupby("worm_id"):
        matrices_for_worm = [
            matrices[(worm_id, int(row.model_seed), state, lag, metric)]
            for row in group.itertuples()
        ]
        matrix = np.mean(matrices_for_worm, axis=0)
        values.append(float(np.mean(offdiagonal(matrix, source_indices))))
        worms.append(str(worm_id))
    return np.asarray(worms), np.asarray(values, dtype=float)


def analyze(run_dir: Path, output_dir: Path) -> dict:
    manifest, frame, matrices = load_run(run_dir)
    source_indices = np.asarray(manifest["source_indices"], dtype=np.int64)
    states = tuple(manifest["config"]["states"])
    lags = tuple(int(value) for value in manifest["config"]["lag_frames"])
    seeds = sorted(set(frame.model_seed.astype(int)))
    if len(seeds) != 2:
        raise RuntimeError("the frozen reliability analysis requires exactly two model seeds")
    worm_ids = sorted(set(frame.worm_id.astype(str)))
    split_a = set(worm_ids[::2])
    split_b = set(worm_ids[1::2])
    if not split_a or not split_b:
        raise RuntimeError("worm split is empty")

    reliability_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    lag32_by_state: dict[str, float] = {}
    for state in states:
        if 32 in lags:
            control = aggregate_matrix(
                frame, matrices, state=state, lag=32, metric=PRIMARY_METRIC
            )
            lag32_by_state[state] = float(np.mean(offdiagonal(control, source_indices)))
        else:
            lag32_by_state[state] = np.nan

        for lag in lags:
            for metric in METRICS:
                matrix = aggregate_matrix(frame, matrices, state=state, lag=lag, metric=metric)
                vector = offdiagonal(matrix, source_indices)
                summary_rows.append(
                    {
                        "state": state,
                        "lag_frames": lag,
                        "lag_seconds": lag / 4.0,
                        "metric": metric,
                        "mean": float(np.mean(vector)),
                        "mean_absolute": float(np.mean(np.abs(vector))),
                        "median": float(np.median(vector)),
                        "q90_absolute": float(np.quantile(np.abs(vector), 0.90)),
                    }
                )

            seed_matrices = [
                aggregate_matrix(
                    frame, matrices, state=state, lag=lag, metric=PRIMARY_METRIC,
                    seeds={seed},
                )
                for seed in seeds
            ]
            seed_rho, seed_p, seed_null95 = source_permutation_p(
                seed_matrices[0], seed_matrices[1], source_indices,
                seed=20260828 + 101 * lag + 17 * states.index(state),
            )
            split_matrices = [
                aggregate_matrix(
                    frame, matrices, state=state, lag=lag, metric=PRIMARY_METRIC,
                    worms=worms,
                )
                for worms in (split_a, split_b)
            ]
            split_rho, split_p, split_null95 = source_permutation_p(
                split_matrices[0], split_matrices[1], source_indices,
                seed=20260829 + 103 * lag + 19 * states.index(state),
            )
            _, proper_values = worm_scalar_values(
                frame,
                matrices,
                source_indices,
                state=state,
                lag=lag,
                metric="proper_energy_penalty",
            )
            proper_low, proper_high = worm_bootstrap_interval(
                proper_values, seed=20260830 + 107 * lag + states.index(state)
            )
            w1_matrix = aggregate_matrix(
                frame, matrices, state=state, lag=lag, metric=PRIMARY_METRIC
            )
            w1_mean = float(np.mean(offdiagonal(w1_matrix, source_indices)))
            control = lag32_by_state[state]
            proper_pass = bool(float(np.mean(proper_values)) > 0 and proper_low > 0)
            seed_pass = bool(seed_rho >= 0.30 and seed_p < 0.05)
            split_pass = bool(split_rho >= 0.20 and split_p < 0.05)
            lag_pass = bool(
                lag != 32
                and np.isfinite(control)
                and w1_mean >= 3.0 * control + 1e-5
            )
            passed = bool(proper_pass and seed_pass and split_pass and lag_pass)
            reliability_rows.extend(
                [
                    {
                        "state": state,
                        "lag_frames": lag,
                        "lag_seconds": lag / 4.0,
                        "comparison": "model_seed",
                        "spearman_rho": seed_rho,
                        "source_permutation_p": seed_p,
                        "null_q95": seed_null95,
                    },
                    {
                        "state": state,
                        "lag_frames": lag,
                        "lag_seconds": lag / 4.0,
                        "comparison": "worm_split",
                        "spearman_rho": split_rho,
                        "source_permutation_p": split_p,
                        "null_q95": split_null95,
                    },
                ]
            )
            gate_rows.append(
                {
                    "state": state,
                    "lag_frames": lag,
                    "lag_seconds": lag / 4.0,
                    "mean_wasserstein1": w1_mean,
                    "lag32_control_wasserstein1": control,
                    "mean_proper_energy_penalty": float(np.mean(proper_values)),
                    "proper_energy_ci_low": proper_low,
                    "proper_energy_ci_high": proper_high,
                    "seed_rho": seed_rho,
                    "seed_permutation_p": seed_p,
                    "worm_split_rho": split_rho,
                    "worm_split_permutation_p": split_p,
                    "proper_score_pass": proper_pass,
                    "seed_reliability_pass": seed_pass,
                    "worm_split_reliability_pass": split_pass,
                    "lag_specificity_pass": lag_pass,
                    "all_gates_pass": passed,
                }
            )

    gate_frame = pd.DataFrame(gate_rows)
    passed_cells = gate_frame[gate_frame.all_gates_pass]
    edge_rows: list[dict[str, object]] = []
    neurons = tuple(manifest["target_neurons"])
    for cell in passed_cells.itertuples():
        state, lag = str(cell.state), int(cell.lag_frames)
        aggregates = {
            metric: aggregate_matrix(frame, matrices, state=state, lag=lag, metric=metric)
            for metric in METRICS
        }
        score = aggregates[PRIMARY_METRIC] * np.maximum(
            aggregates["proper_energy_penalty"], 0.0
        )
        for row, source in enumerate(source_indices):
            score[row, source] = -np.inf
        flat_order = np.argsort(score.ravel())[::-1]
        kept = 0
        for flat_index in flat_order:
            row, target = np.unravel_index(flat_index, score.shape)
            if not np.isfinite(score[row, target]) or score[row, target] <= 0:
                continue
            source = int(source_indices[row])
            dominant = max(SIGNED_METRICS, key=lambda name: abs(aggregates[name][row, target]))
            signed_by_worm: list[float] = []
            use = frame[(frame.state == state) & (frame.lag_frames == lag)]
            for worm_id, group in use.groupby("worm_id"):
                signed_by_worm.append(
                    float(
                        np.mean(
                            [
                                matrices[(worm_id, int(item.model_seed), state, lag, dominant)][row, target]
                                for item in group.itertuples()
                            ]
                        )
                    )
                )
            signed_by_worm_array = np.asarray(signed_by_worm)
            aggregate_sign = np.sign(aggregates[dominant][row, target])
            sign_consistency = float(np.mean(np.sign(signed_by_worm_array) == aggregate_sign))
            edge_rows.append(
                {
                    "state": state,
                    "lag_frames": lag,
                    "lag_seconds": lag / 4.0,
                    "source": neurons[source],
                    "target": neurons[target],
                    "source_index": source,
                    "target_index": target,
                    "selection_score": float(score[row, target]),
                    "wasserstein1": float(aggregates[PRIMARY_METRIC][row, target]),
                    "proper_energy_penalty": float(aggregates["proper_energy_penalty"][row, target]),
                    "mean_shift": float(aggregates["mean_shift"][row, target]),
                    "log_sd_shift": float(aggregates["log_sd_shift"][row, target]),
                    "tail_probability_shift": float(aggregates["tail_probability_shift"][row, target]),
                    "dominant_signed_metric": dominant,
                    "worm_sign_consistency": sign_consistency,
                    "propagation_candidate": bool(sign_consistency >= 0.70),
                }
            )
            kept += 1
            if kept >= 20:
                break

    edge_frame = pd.DataFrame(edge_rows)
    if not edge_frame.empty:
        edge_frame = edge_frame.sort_values(
            ["propagation_candidate", "selection_score"], ascending=[False, False]
        ).reset_index(drop=True)
    candidates = (
        edge_frame[edge_frame.propagation_candidate]
        if not edge_frame.empty
        else pd.DataFrame()
    )
    selected_sources = sorted(set(candidates.source.astype(str))) if not candidates.empty else []
    smc_authorized = bool(len(passed_cells) and selected_sources)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / "metric_summary.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(output_dir / "matrix_reliability.csv", index=False)
    gate_frame.to_csv(output_dir / "direct_gate.csv", index=False)
    edge_frame.to_csv(output_dir / "candidate_edges.csv", index=False)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir.resolve()),
        "primary_metric": PRIMARY_METRIC,
        "passed_state_lag_cells": [
            {"state": str(row.state), "lag_frames": int(row.lag_frames), "lag_seconds": float(row.lag_seconds)}
            for row in passed_cells.itertuples()
        ],
        "selected_source_neurons_for_convergence": selected_sources,
        "direct_convergence_authorized": bool(selected_sources),
        "temporal_cut_smc_authorized": smc_authorized,
        "selection_firewall": manifest["selection_firewall"],
        "claim_boundary": manifest["claim_boundary"],
    }
    atomic_json(output_dir / "gate_decision.json", decision)

    lines = [
        "# Frozen paired-CRN distributional lag audit",
        "",
        "## Decision",
        "",
    ]
    if smc_authorized:
        lines.append(
            f"**{len(passed_cells)} state×lag cells passed the frozen direct-effect gate.** "
            f"Convergence reruns are authorized for {len(selected_sources)} source neurons; temporal-cut ESS-SMC remains conditional on N=64/N=256 convergence."
        )
    else:
        lines.append(
            "**No direct distributional cell produced a propagation-ready edge under the frozen gate.** "
            "Temporal-cut ESS-SMC must not be launched from this audit."
        )
    lines.extend(
        [
            "",
            "The primary matrix is target-wise one-dimensional Wasserstein distance between conditional-low and conditional-high source-history replacements. Mean, log-scale, and tail-probability shifts are retained as separate signed summaries. Positive proper-energy penalty means the factual source history predicts the held-out target better than the two-point conditional-resampling mixture.",
            "",
            "| State | Lag (s) | W1 | Proper penalty [95% worm CI] | Seed ρ | Split ρ | Pass |",
            "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for row in gate_frame.itertuples():
        lines.append(
            f"| {row.state} | {row.lag_seconds:.2f} | {row.mean_wasserstein1:.6f} | "
            f"{row.mean_proper_energy_penalty:+.6f} [{row.proper_energy_ci_low:+.6f}, {row.proper_energy_ci_high:+.6f}] | "
            f"{row.seed_rho:.3f} | {row.worm_split_rho:.3f} | {'yes' if row.all_gates_pass else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Lag 8.0 seconds (32 frames) is a structural negative control: it is outside the frozen legacy TCN's 31-frame receptive field. Source-label permutations calibrate matrix reliability. Reversed histories were intentionally not used because they are off the empirical support.",
            "",
            "This is an atlas-blind, model-relative observational sensitivity analysis. No result identifies a causal edge, synapse, receptor action, or physical transmission delay.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": manifest,
        "worm_split_a": sorted(split_a),
        "worm_split_b": sorted(split_b),
        "analysis_gate": manifest["frozen_gate"],
    }
    atomic_json(output_dir / "provenance.json", provenance)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
