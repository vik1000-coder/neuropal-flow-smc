from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from compatibility_neural_benchmark.core import causal_fill, fit_anchor_projection
from compatibility_neural_benchmark.distributional_lag_analysis import (
    source_permutation_p,
    worm_bootstrap_interval,
)
from compatibility_neural_benchmark.distributional_lag_audit import (
    METRICS,
    conditional_features,
    deterministic_seed,
    distributional_effects,
    evaluation_targets,
    fit_conditional_resampler,
)
from compatibility_neural_benchmark.latent_distributional_audit import (
    PARTICLE_COUNTS,
    benjamini_hochberg,
    fold_parameters,
    predict_parameters,
    standardized_traces,
)
from conditional_neural_benchmark.data import _stimulus_features, load_cohort
from conditional_neural_benchmark.latent_calcium_lag import innovation_trace, load_folds


HORIZONS = (1, 2, 4, 8, 16)
SOURCE_LAG = 32


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def projected_batch(
    observed_history: np.ndarray,
    innovation_history: np.ndarray,
    stimulus_history: np.ndarray,
    basis: np.ndarray,
    pca_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    chronological = basis[::-1]
    neural = np.einsum(
        "bld,lq->bdq", innovation_history, chronological, optimize=True
    ).reshape(len(innovation_history), -1)
    global_history = observed_history @ pca_components.T
    global_basis = np.einsum(
        "blg,lq->bgq", global_history, chronological, optimize=True
    ).reshape(len(observed_history), -1)
    stimulus_basis = stimulus_history @ chronological
    common = np.concatenate(
        [
            global_basis,
            stimulus_basis,
            stimulus_history[:, -1, None],
            np.abs(np.diff(stimulus_history, axis=1)).sum(axis=1)[:, None],
        ],
        axis=1,
    )
    return neural.astype(np.float32), common.astype(np.float32)


def paired_rollout(
    *,
    observed_history: np.ndarray,
    innovation_history: np.ndarray,
    stimulus_history: np.ndarray,
    future_stimulus: np.ndarray,
    source_indices: np.ndarray,
    source_low: np.ndarray,
    source_high: np.ndarray,
    parameters: dict[str, np.ndarray],
    basis: np.ndarray,
    seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Roll low/high/factual histories with source-paired common noise."""
    m = len(source_indices)
    n = max(PARTICLE_COUNTS)
    d = observed_history.shape[1]
    observed = np.broadcast_to(
        observed_history, (3, m, n, *observed_history.shape)
    ).copy()
    innovation = np.broadcast_to(
        innovation_history, (3, m, n, *innovation_history.shape)
    ).copy()
    stimulus = np.broadcast_to(
        stimulus_history, (3, m, n, len(stimulus_history))
    ).copy()
    for position, source in enumerate(source_indices):
        innovation[0, position, :, -SOURCE_LAG, source] = source_low[source]
        innovation[1, position, :, -SOURCE_LAG, source] = source_high[source]
    rng = np.random.default_rng(seed)
    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for step in range(1, max(HORIZONS) + 1):
        flat_observed = observed.reshape(-1, len(basis), d)
        flat_innovation = innovation.reshape(-1, len(basis), d)
        flat_stimulus = stimulus.reshape(-1, len(basis))
        neural, common = projected_batch(
            flat_observed,
            flat_innovation,
            flat_stimulus,
            basis,
            parameters["pca_components"],
        )
        mean_innovation, logvar = predict_parameters(neural, common, parameters)
        observation_base = (
            parameters["intercept"][None]
            + parameters["alpha"][None] * flat_observed[:, -1]
        )
        common_noise = rng.normal(size=(m, n, d)).astype(np.float32)
        noise = np.broadcast_to(common_noise, (3, m, n, d)).reshape(-1, d)
        next_innovation = mean_innovation + np.exp(0.5 * logvar) * noise
        next_observed = observation_base + next_innovation
        next_innovation = next_innovation.reshape(3, m, n, d)
        next_observed = next_observed.reshape(3, m, n, d)
        if step in HORIZONS:
            result[step] = (
                next_observed[0].copy(),
                next_observed[1].copy(),
                next_observed[2].copy(),
            )
        observed = np.concatenate(
            [observed[..., 1:, :], next_observed[..., None, :]], axis=-2
        )
        innovation = np.concatenate(
            [innovation[..., 1:, :], next_innovation[..., None, :]], axis=-2
        )
        next_stimulus = np.full(
            (3, m, n, 1), future_stimulus[step - 1], dtype=np.float32
        )
        stimulus = np.concatenate([stimulus[..., 1:], next_stimulus], axis=-1)
    return result


def run_fold(
    cohort,
    folds: np.ndarray,
    fold: int,
    model_dir: Path,
    basis: np.ndarray,
    source_indices: np.ndarray,
    output: Path,
) -> dict[str, object]:
    parameters = fold_parameters(model_dir / f"fold_{fold}.npz")
    traces = standardized_traces(cohort, parameters)
    innovations = [
        innovation_trace(trace, parameters["alpha"], parameters["intercept"])
        for trace in traces
    ]
    training = np.flatnonzero(folds != fold)
    heldout = np.flatnonzero(folds == fold)
    projection = fit_anchor_projection(
        [innovations[int(worm)] for worm in training], 8
    )
    resampler = fit_conditional_resampler(
        [innovations[int(worm)] for worm in training],
        [cohort.stimulus_schedules[int(worm)] for worm in training],
        state="onset",
        lag=SOURCE_LAG,
        projection=projection,
        ridge_alpha=2.0,
        residual_quantiles=(0.25, 0.75),
        clip_quantiles=(0.01, 0.99),
    )
    pooled = np.concatenate([traces[int(worm)] for worm in training], axis=0)
    tail_threshold = np.quantile(np.abs(pooled), 0.90, axis=0).astype(np.float32)
    d = cohort.n_neurons
    shape = (len(heldout), len(HORIZONS), len(source_indices), d)
    effects = {
        count: {metric: np.full(shape, np.nan, dtype=np.float32) for metric in METRICS}
        for count in PARTICLE_COUNTS
    }
    for worm_position, worm in enumerate(heldout):
        trace = traces[int(worm)]
        innovation = innovations[int(worm)]
        schedule = cohort.stimulus_schedules[int(worm)]
        stimulus = _stimulus_features(
            len(trace), schedule, "binary_any_stimulus"
        )[:, 0]
        onset_targets = [
            (event, target)
            for state, event, target in evaluation_targets(schedule, len(trace))
            if state == "onset"
        ]
        accumulators = {
            count: {
                horizon: {metric: [] for metric in METRICS}
                for horizon in HORIZONS
            }
            for count in PARTICLE_COUNTS
        }
        for event, target_time in onset_targets:
            observed_history = trace[target_time - len(basis) : target_time]
            innovation_history = innovation[target_time - len(basis) : target_time]
            stimulus_history = stimulus[target_time - len(basis) : target_time]
            future_stimulus = stimulus[target_time : target_time + max(HORIZONS)]
            low, high = resampler.predict_low_high(
                conditional_features(innovation, target_time, SOURCE_LAG, projection)
            )
            rollouts = paired_rollout(
                observed_history=observed_history,
                innovation_history=innovation_history,
                stimulus_history=stimulus_history,
                future_stimulus=future_stimulus,
                source_indices=source_indices,
                source_low=low,
                source_high=high,
                parameters=parameters,
                basis=basis,
                seed=deterministic_seed("latent_rollout", fold, int(worm), event),
            )
            for horizon in HORIZONS:
                low_samples, high_samples, factual_samples = rollouts[horizon]
                actual = trace[target_time + horizon - 1]
                for count in PARTICLE_COUNTS:
                    result = distributional_effects(
                        low_samples[:, :count],
                        high_samples[:, :count],
                        factual_samples[:, :count],
                        actual,
                        tail_threshold,
                    )
                    for metric in METRICS:
                        accumulators[count][horizon][metric].append(result[metric])
        for count in PARTICLE_COUNTS:
            for horizon_position, horizon in enumerate(HORIZONS):
                for metric in METRICS:
                    effects[count][metric][worm_position, horizon_position] = np.mean(
                        accumulators[count][horizon][metric], axis=0
                    )
        print(f"LATENT_ROLLOUT fold={fold} worm={schedule.worm_id}", flush=True)
    path = output / "responses" / f"latent_rollout__f{fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"N{count}__{metric}": value
        for count, by_metric in effects.items()
        for metric, value in by_metric.items()
    }
    np.savez_compressed(
        path,
        status=np.asarray("complete"),
        fold=np.asarray(fold),
        heldout_worm_ids=np.asarray([cohort.worm_ids[int(worm)] for worm in heldout]),
        source_indices=source_indices.astype(np.int16),
        source_neurons=np.asarray([cohort.neurons[int(source)] for source in source_indices]),
        target_neurons=np.asarray(cohort.neurons),
        horizon_frames=np.asarray(HORIZONS, dtype=np.int16),
        horizon_seconds=np.asarray(HORIZONS, dtype=np.float32) / cohort.fps,
        source_lag_frames=np.asarray(SOURCE_LAG),
        source_lag_seconds=np.asarray(SOURCE_LAG / cohort.fps),
        stimulus_schema_version=np.asarray(cohort.stimulus_schema_version),
        stimulus_schema_fingerprint=np.asarray(cohort.stimulus_schema_fingerprint),
        **payload,
    )
    return {"fold": fold, "status": "ok", "output": str(path)}


def load_results(output: Path) -> tuple[pd.DataFrame, dict[tuple, np.ndarray], np.ndarray, tuple[str, ...]]:
    rows: list[dict[str, object]] = []
    values: dict[tuple, np.ndarray] = {}
    source_indices: np.ndarray | None = None
    targets: tuple[str, ...] | None = None
    for path in sorted((output / "responses").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if source_indices is None:
                source_indices = data["source_indices"].astype(int)
                targets = tuple(data["target_neurons"].astype(str))
            for worm_position, worm in enumerate(data["heldout_worm_ids"].astype(str)):
                for horizon_position, horizon in enumerate(data["horizon_frames"].astype(int)):
                    rows.append({"worm_id": worm, "horizon_frames": int(horizon)})
                    for count in PARTICLE_COUNTS:
                        for metric in METRICS:
                            values[(worm, int(horizon), count, metric)] = data[
                                f"N{count}__{metric}"
                            ][worm_position, horizon_position].astype(float)
    if source_indices is None or targets is None:
        raise RuntimeError("no rollout archives")
    return pd.DataFrame(rows), values, source_indices, targets


def analyze(output: Path, edge_file: Path) -> dict:
    frame, values, source_indices, neurons = load_results(output)
    edges = pd.read_csv(edge_file)
    edges = edges[edges.propagation_candidate].copy()
    worms = sorted(set(frame.worm_id.astype(str)))
    edge_rows: list[dict[str, object]] = []
    for edge in edges.itertuples():
        source_position = int(np.flatnonzero(source_indices == int(edge.source_index))[0])
        target = int(edge.target_index)
        for horizon in HORIZONS:
            metric_arrays = {
                metric: np.asarray(
                    [values[(worm, horizon, 256, metric)][source_position, target] for worm in worms]
                )
                for metric in METRICS
            }
            dominant = str(edge.dominant_signed_metric)
            signed = metric_arrays[dominant]
            proper = metric_arrays["proper_energy_penalty"]
            signed_low, signed_high = worm_bootstrap_interval(
                signed,
                seed=deterministic_seed("roll_signed", edge.source, edge.target, horizon),
            )
            proper_low, proper_high = worm_bootstrap_interval(
                proper,
                seed=deterministic_seed("roll_proper", edge.source, edge.target, horizon),
            )
            n64 = np.asarray(
                [values[(worm, horizon, 64, "wasserstein1")][source_position, target] for worm in worms]
            )
            n256 = metric_arrays["wasserstein1"]
            relative_error = float(
                abs(n64.mean() - n256.mean()) / max(abs(n256.mean()), 1e-12)
            )
            aggregate_sign = np.sign(signed.mean())
            sign_consistency = float(np.mean(np.sign(signed) == aggregate_sign))
            edge_rows.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "source_index": int(edge.source_index),
                    "target_index": target,
                    "source_lag_frames": SOURCE_LAG,
                    "source_lag_seconds": SOURCE_LAG / 4.0,
                    "horizon_frames": horizon,
                    "horizon_seconds": horizon / 4.0,
                    "dominant_signed_metric": dominant,
                    "signed_effect": float(signed.mean()),
                    "signed_ci_low": signed_low,
                    "signed_ci_high": signed_high,
                    "signed_p_value": float(ttest_1samp(signed, 0.0).pvalue),
                    "worm_sign_consistency": sign_consistency,
                    "proper_energy_penalty": float(proper.mean()),
                    "proper_ci_low": proper_low,
                    "proper_ci_high": proper_high,
                    "wasserstein1": float(n256.mean()),
                    "N64_N256_relative_error": relative_error,
                    "provisional": bool(
                        horizon >= 2
                        and proper_low > 0
                        and (signed_low > 0 or signed_high < 0)
                        and sign_consistency >= 0.70
                        and relative_error <= 0.10
                    ),
                }
            )
    result = pd.DataFrame(edge_rows)
    result["signed_bh_q"] = benjamini_hochberg(result.signed_p_value.to_numpy())
    result["propagation_pass"] = result.provisional & (result.signed_bh_q < 0.10)
    result.to_csv(output / "edge_horizon_gate.csv", index=False)
    passed = result[result.propagation_pass]
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tested_direct_edges": int(len(edges)),
        "tested_horizons": list(HORIZONS),
        "propagated_edge_horizon_hits": int(len(passed)),
        "propagated_unique_edges": int(passed[["source", "target"]].drop_duplicates().shape[0]),
        "temporal_cut_ess_smc_authorized": bool(len(passed)),
        "selection_firewall": "no atlas/connectome/receptor/SBTG target used",
        "claim_boundary": "paired generated-state propagation in a learned observational law, not causal or physical transmission",
    }
    atomic_json(output / "gate_decision.json", decision)
    lines = [
        "# Short paired rollout of direct distributional candidates",
        "",
        "## Decision",
        "",
        (
            f"**{len(passed)} edge×horizon effects ({decision['propagated_unique_edges']} unique edges) survive after the original lag coordinate leaves model memory.** A targeted ESS-SMC sensitivity is authorized."
            if len(passed)
            else "**No direct candidate remains predictively useful after the original lag coordinate leaves model memory.** ESS-SMC is not authorized."
        ),
        "",
        "Low, high, and factual paths use common process noise. N=64 is nested inside N=256. Horizons begin at the first forecasted frame; at horizon 2 and later the perturbed 8-second coordinate has already dropped from the 32-frame model history.",
        "",
    ]
    if len(passed):
        lines.extend(
            [
                "| Source→target | Horizon (s) | Mechanism | Effect | Proper penalty | W1 | BH q |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in passed.sort_values("signed_bh_q").itertuples():
            lines.append(
                f"| {row.source}→{row.target} | {row.horizon_seconds:.2f} | {row.dominant_signed_metric} | "
                f"{row.signed_effect:+.6f} | {row.proper_energy_penalty:+.6f} | "
                f"{row.wasserstein1:.6f} | {row.signed_bh_q:.4f} |"
            )
    lines.extend(
        [
            "",
            "The rollout is a model-relative finite contrast. Persistence is not evidence of a synapse, causal influence, receptor action, or physical delay.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--direct-audit-dir", type=Path, required=True)
    parser.add_argument("--fold-evidence", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model_dir = args.model_dir.resolve()
    direct = args.direct_audit_dir.resolve()
    direct_gate = json.loads((direct / "gate_decision.json").read_text())
    if not direct_gate.get("paired_rollout_authorized"):
        raise RuntimeError("direct audit did not authorize paired rollout")
    candidates = pd.read_csv(direct / "candidate_edges.csv")
    candidates = candidates[candidates.propagation_candidate]
    source_names = sorted(set(candidates.source.astype(str)))
    cohort = load_cohort(cohort_mode="oh16230_head")
    source_indices = np.asarray([cohort.neurons.index(name) for name in source_names], dtype=np.int64)
    folds = load_folds(args.fold_evidence.resolve(), cohort)
    basis = np.load(model_dir / "lag_basis.npy")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "protocol": "paired common-noise short rollout after strict direct distributional gate v1",
        "model_dir": str(model_dir),
        "direct_audit_dir": str(direct),
        "direct_gate": direct_gate,
        "source_neurons": source_names,
        "source_lag_frames": SOURCE_LAG,
        "source_lag_seconds": SOURCE_LAG / cohort.fps,
        "horizon_frames": list(HORIZONS),
        "horizon_seconds": [horizon / cohort.fps for horizon in HORIZONS],
        "particle_counts": list(PARTICLE_COUNTS),
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "selection_firewall": "no atlas/connectome/receptor/SBTG target used",
        "claim_boundary": "paired generated-state propagation in a learned observational law, not causal or physical transmission",
    }
    atomic_json(output / "manifest.json", manifest)
    records = [
        run_fold(cohort, folds, fold, model_dir, basis, source_indices, output)
        for fold in range(5)
    ]
    pd.DataFrame(records).to_csv(output / "fold_status.csv", index=False)
    decision = analyze(output, direct / "candidate_edges.csv")
    manifest["status"] = "complete"
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["gate_decision"] = decision
    atomic_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
