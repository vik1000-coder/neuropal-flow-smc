from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

from compatibility_neural_benchmark.core import causal_fill, fit_anchor_projection
from compatibility_neural_benchmark.distributional_lag_analysis import (
    offdiagonal,
    safe_spearman,
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
from conditional_neural_benchmark.data import _stimulus_features, load_cohort
from conditional_neural_benchmark.latent_calcium_lag import (
    innovation_trace,
    load_folds,
)


STATES = ("quiet", "onset", "active")
LAGS = (1, 2, 4, 8, 16, 32, 40)
PARTICLE_COUNTS = (64, 256)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def projected_features(
    observed_history: np.ndarray,
    innovation_history: np.ndarray,
    stimulus_history: np.ndarray,
    basis: np.ndarray,
    pca_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    chronological = basis[::-1]
    neural = np.einsum(
        "ld,lq->dq", innovation_history, chronological, optimize=True
    ).reshape(-1)
    global_history = observed_history @ pca_components.T
    global_basis = np.einsum(
        "lg,lq->gq", global_history, chronological, optimize=True
    ).reshape(-1)
    stimulus_basis = stimulus_history @ chronological
    common = np.concatenate(
        [
            global_basis,
            stimulus_basis,
            np.asarray([stimulus_history[-1]], dtype=np.float32),
            np.asarray([np.abs(np.diff(stimulus_history)).sum()], dtype=np.float32),
        ]
    )
    return neural.astype(np.float32), common.astype(np.float32)


def predict_parameters(
    neural: np.ndarray, common: np.ndarray, parameters: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    neural = np.atleast_2d(np.asarray(neural, dtype=np.float32))
    common = np.atleast_2d(np.asarray(common, dtype=np.float32))
    neural_standard = (
        neural - parameters["neural_feature_mean"]
    ) / parameters["neural_feature_scale"]
    common_standard = (
        common - parameters["common_feature_mean"]
    ) / parameters["common_feature_scale"]
    neural_expected = (
        common_standard @ parameters["common_to_neural_coef"].T
        + parameters["common_to_neural_intercept"]
    )
    neural_residual = neural_standard - neural_expected
    d = len(parameters["alpha"])
    q = parameters["mean_kernel_basis"].shape[2]
    mean_base = np.empty((len(neural), d), dtype=np.float32)
    scale_base = np.empty_like(mean_base)
    for target in range(d):
        self_features = neural_standard[:, target * q : (target + 1) * q]
        design = np.concatenate([common_standard, self_features], axis=1)
        mean_base[:, target] = (
            design @ parameters["mean_base_coef"][target]
            + parameters["mean_base_intercept"][target]
        )
        scale_base[:, target] = (
            design @ parameters["scale_base_coef"][target]
            + parameters["scale_base_intercept"][target]
        )
    mean = mean_base + neural_residual @ parameters["mean_kernel_basis"].reshape(d, -1).T
    logvar = (
        scale_base
        + neural_residual @ parameters["scale_kernel_basis"].reshape(d, -1).T
        + parameters["full_offset"]
    )
    return mean.astype(np.float32), np.clip(logvar, -10.0, 5.0).astype(np.float32)


def fold_parameters(path: Path) -> dict[str, np.ndarray]:
    required = (
        "alpha",
        "intercept",
        "mean_kernel_basis",
        "scale_kernel_basis",
        "mean_base_coef",
        "mean_base_intercept",
        "scale_base_coef",
        "scale_base_intercept",
        "common_to_neural_coef",
        "common_to_neural_intercept",
        "neural_feature_mean",
        "neural_feature_scale",
        "common_feature_mean",
        "common_feature_scale",
        "pca_components",
        "observation_scaler_mean",
        "observation_scaler_scale",
        "full_offset",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise RuntimeError(f"latent fold archive lacks parameters: {missing}")
        return {name: data[name].astype(np.float32) for name in required}


def standardized_traces(cohort, parameters: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [
        causal_fill(
            (np.asarray(trace, dtype=np.float32) - parameters["observation_scaler_mean"])
            / parameters["observation_scaler_scale"]
        )
        for trace in cohort.traces
    ]


def run_fold(
    cohort,
    folds: np.ndarray,
    fold: int,
    model_dir: Path,
    basis: np.ndarray,
    output_dir: Path,
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
    resamplers = {
        (state, lag): fit_conditional_resampler(
            [innovations[int(worm)] for worm in training],
            [cohort.stimulus_schedules[int(worm)] for worm in training],
            state=state,
            lag=lag,
            projection=projection,
            ridge_alpha=2.0,
            residual_quantiles=(0.25, 0.75),
            clip_quantiles=(0.01, 0.99),
        )
        for state in STATES
        for lag in LAGS
    }
    pooled = np.concatenate([traces[int(worm)] for worm in training], axis=0)
    tail_threshold = np.quantile(np.abs(pooled), 0.90, axis=0).astype(np.float32)
    d = cohort.n_neurons
    shape = (len(heldout), len(STATES), len(LAGS), d, d)
    effects = {
        count: {metric: np.full(shape, np.nan, dtype=np.float32) for metric in METRICS}
        for count in PARTICLE_COUNTS
    }
    conditional_iqr = np.full(shape[:-1], np.nan, dtype=np.float32)

    for worm_position, worm in enumerate(heldout):
        trace = traces[int(worm)]
        innovation = innovations[int(worm)]
        schedule = cohort.stimulus_schedules[int(worm)]
        stimulus = _stimulus_features(
            len(trace), schedule, "binary_any_stimulus"
        )[:, 0]
        targets = evaluation_targets(schedule, len(trace))
        for state_position, state in enumerate(STATES):
            state_targets = [
                (event, target) for candidate, event, target in targets if candidate == state
            ]
            for lag_position, lag in enumerate(LAGS):
                accumulators = {
                    count: {metric: [] for metric in METRICS}
                    for count in PARTICLE_COUNTS
                }
                gaps: list[np.ndarray] = []
                for event, target_time in state_targets:
                    observed_history = trace[target_time - len(basis) : target_time]
                    innovation_history = innovation[target_time - len(basis) : target_time]
                    stimulus_history = stimulus[target_time - len(basis) : target_time]
                    neural, common = projected_features(
                        observed_history,
                        innovation_history,
                        stimulus_history,
                        basis,
                        parameters["pca_components"],
                    )
                    factual_mean, factual_logvar = predict_parameters(neural, common, parameters)
                    low, high = resamplers[(state, lag)].predict_low_high(
                        conditional_features(innovation, target_time, lag, projection)
                    )
                    gaps.append(high - low)
                    raw_basis = neural.reshape(d, -1)
                    low_features = np.repeat(raw_basis[None], d, axis=0)
                    high_features = low_features.copy()
                    weight = (
                        basis[lag - 1]
                        if lag <= len(basis)
                        else np.zeros(basis.shape[1], dtype=np.float32)
                    )
                    for source in range(d):
                        factual_value = innovation[target_time - lag, source]
                        low_features[source, source] += (low[source] - factual_value) * weight
                        high_features[source, source] += (high[source] - factual_value) * weight
                    common_batch = np.repeat(common[None], d, axis=0)
                    low_mean, low_logvar = predict_parameters(
                        low_features.reshape(d, -1), common_batch, parameters
                    )
                    high_mean, high_logvar = predict_parameters(
                        high_features.reshape(d, -1), common_batch, parameters
                    )
                    observation_base = (
                        parameters["intercept"]
                        + parameters["alpha"] * trace[target_time - 1]
                    )
                    low_mean += observation_base
                    high_mean += observation_base
                    factual_mean += observation_base
                    rng = np.random.default_rng(
                        deterministic_seed("latent_crn", fold, int(worm), state, event, lag)
                    )
                    noise = rng.normal(size=(d, max(PARTICLE_COUNTS), d)).astype(np.float32)
                    for count in PARTICLE_COUNTS:
                        use_noise = noise[:, :count]
                        low_samples = low_mean[:, None] + np.exp(0.5 * low_logvar)[:, None] * use_noise
                        high_samples = high_mean[:, None] + np.exp(0.5 * high_logvar)[:, None] * use_noise
                        factual_samples = (
                            np.repeat(factual_mean, d, axis=0)[:, None]
                            + np.exp(0.5 * np.repeat(factual_logvar, d, axis=0))[:, None]
                            * use_noise
                        )
                        result = distributional_effects(
                            low_samples,
                            high_samples,
                            factual_samples,
                            trace[target_time],
                            tail_threshold,
                        )
                        for metric in METRICS:
                            accumulators[count][metric].append(result[metric])
                for count in PARTICLE_COUNTS:
                    for metric in METRICS:
                        effects[count][metric][worm_position, state_position, lag_position] = np.mean(
                            accumulators[count][metric], axis=0
                        )
                conditional_iqr[worm_position, state_position, lag_position] = np.mean(
                    gaps, axis=0
                )
            print(
                f"LATENT_AUDIT fold={fold} worm={schedule.worm_id} state={state}",
                flush=True,
            )

    path = output_dir / "responses" / f"latent_distributional__f{fold}.npz"
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
        heldout_worm_indices=heldout.astype(np.int16),
        heldout_worm_ids=np.asarray([cohort.worm_ids[int(worm)] for worm in heldout]),
        state_names=np.asarray(STATES),
        lag_frames=np.asarray(LAGS, dtype=np.int16),
        lag_seconds=np.asarray(LAGS, dtype=np.float32) / cohort.fps,
        source_indices=np.arange(d, dtype=np.int16),
        neurons=np.asarray(cohort.neurons),
        conditional_iqr=conditional_iqr,
        stimulus_schema_version=np.asarray(cohort.stimulus_schema_version),
        stimulus_schema_fingerprint=np.asarray(cohort.stimulus_schema_fingerprint),
        **payload,
    )
    return {"fold": fold, "status": "ok", "output": str(path)}


def load_results(output_dir: Path) -> tuple[pd.DataFrame, dict[tuple, np.ndarray], tuple[str, ...]]:
    rows: list[dict[str, object]] = []
    matrices: dict[tuple, np.ndarray] = {}
    neurons: tuple[str, ...] | None = None
    for path in sorted((output_dir / "responses").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            fold = int(data["fold"].item())
            current_neurons = tuple(data["neurons"].astype(str))
            if neurons is None:
                neurons = current_neurons
            elif neurons != current_neurons:
                raise RuntimeError("neuron mismatch across latent audit archives")
            for worm_position, worm in enumerate(data["heldout_worm_ids"].astype(str)):
                for state_position, state in enumerate(data["state_names"].astype(str)):
                    for lag_position, lag in enumerate(data["lag_frames"].astype(int)):
                        rows.append(
                            {
                                "worm_id": worm,
                                "fold": fold,
                                "state": state,
                                "lag_frames": int(lag),
                                "lag_seconds": int(lag) / 4.0,
                            }
                        )
                        for count in PARTICLE_COUNTS:
                            for metric in METRICS:
                                matrices[(worm, state, int(lag), count, metric)] = data[
                                    f"N{count}__{metric}"
                                ][worm_position, state_position, lag_position].astype(float)
    if neurons is None:
        raise RuntimeError("no latent distributional archives")
    return pd.DataFrame(rows), matrices, neurons


def aggregate(
    frame: pd.DataFrame,
    matrices: dict[tuple, np.ndarray],
    *,
    state: str,
    lag: int,
    count: int,
    metric: str,
    worms: set[str] | None = None,
) -> np.ndarray:
    use = frame[(frame.state == state) & (frame.lag_frames == lag)]
    if worms is not None:
        use = use[use.worm_id.isin(worms)]
    return np.mean(
        [matrices[(row.worm_id, state, lag, count, metric)] for row in use.itertuples()],
        axis=0,
    )


def analyze(output_dir: Path, model_dir: Path) -> dict:
    frame, matrices, neurons = load_results(output_dir)
    source_indices = np.arange(len(neurons), dtype=np.int64)
    worm_ids = sorted(set(frame.worm_id.astype(str)))
    halves = (set(worm_ids[::2]), set(worm_ids[1::2]))
    model_gate = json.loads((model_dir / "gate_decision.json").read_text())
    gate_rows: list[dict[str, object]] = []
    for state in STATES:
        control = float(
            np.mean(
                offdiagonal(
                    aggregate(
                        frame, matrices, state=state, lag=40, count=256,
                        metric="wasserstein1",
                    ),
                    source_indices,
                )
            )
        )
        for lag in LAGS:
            n64 = aggregate(
                frame, matrices, state=state, lag=lag, count=64, metric="wasserstein1"
            )
            n256 = aggregate(
                frame, matrices, state=state, lag=lag, count=256, metric="wasserstein1"
            )
            v64, v256 = offdiagonal(n64, source_indices), offdiagonal(n256, source_indices)
            convergence_rho = safe_spearman(v64, v256)
            relative_error = float(
                np.linalg.norm(v64 - v256) / max(np.linalg.norm(v256), 1e-12)
            )
            split_matrices = [
                aggregate(
                    frame, matrices, state=state, lag=lag, count=256,
                    metric="wasserstein1", worms=half,
                )
                for half in halves
            ]
            split_rho, split_p, _ = source_permutation_p(
                split_matrices[0], split_matrices[1], source_indices,
                seed=20260828 + 103 * lag + STATES.index(state),
            )
            proper_values: list[float] = []
            for worm in worm_ids:
                matrix = matrices[(worm, state, lag, 256, "proper_energy_penalty")]
                proper_values.append(float(np.mean(offdiagonal(matrix, source_indices))))
            proper_low, proper_high = worm_bootstrap_interval(
                np.asarray(proper_values),
                seed=20260829 + 107 * lag + STATES.index(state),
            )
            mean_proper = float(np.mean(proper_values))
            mean_w1 = float(np.mean(v256))
            convergence_pass = convergence_rho >= 0.95 and relative_error <= 0.10
            split_pass = split_rho >= 0.20 and split_p < 0.05
            proper_pass = mean_proper > 0 and proper_low > 0
            lag_pass = lag != 40 and mean_w1 >= 3.0 * control + 1e-5
            passed = bool(
                model_gate["all_gates_pass"]
                and convergence_pass
                and split_pass
                and proper_pass
                and lag_pass
            )
            gate_rows.append(
                {
                    "state": state,
                    "lag_frames": lag,
                    "lag_seconds": lag / 4.0,
                    "mean_wasserstein1_N256": mean_w1,
                    "lag40_structural_control": control,
                    "mean_proper_energy_penalty": mean_proper,
                    "proper_ci_low": proper_low,
                    "proper_ci_high": proper_high,
                    "N64_N256_spearman": convergence_rho,
                    "N64_N256_relative_error": relative_error,
                    "worm_split_spearman": split_rho,
                    "worm_split_permutation_p": split_p,
                    "model_gate_pass": bool(model_gate["all_gates_pass"]),
                    "convergence_pass": convergence_pass,
                    "worm_split_pass": split_pass,
                    "proper_score_pass": proper_pass,
                    "lag_specificity_pass": lag_pass,
                    "all_gates_pass": passed,
                }
            )
    gate_frame = pd.DataFrame(gate_rows)
    gate_frame.to_csv(output_dir / "direct_gate.csv", index=False)
    passed = gate_frame[gate_frame.all_gates_pass]
    edge_rows: list[dict[str, object]] = []
    for cell in passed.itertuples():
        state, lag = str(cell.state), int(cell.lag_frames)
        w1 = aggregate(frame, matrices, state=state, lag=lag, count=256, metric="wasserstein1")
        w1_n64 = aggregate(frame, matrices, state=state, lag=lag, count=64, metric="wasserstein1")
        proper = aggregate(
            frame, matrices, state=state, lag=lag, count=256,
            metric="proper_energy_penalty",
        )
        mean = aggregate(frame, matrices, state=state, lag=lag, count=256, metric="mean_shift")
        scale = aggregate(frame, matrices, state=state, lag=lag, count=256, metric="log_sd_shift")
        tail = aggregate(
            frame, matrices, state=state, lag=lag, count=256,
            metric="tail_probability_shift",
        )
        score = w1 * np.maximum(proper, 0.0)
        np.fill_diagonal(score, -np.inf)
        for flat in np.argsort(score.ravel())[::-1][:40]:
            source, target = np.unravel_index(flat, score.shape)
            if not np.isfinite(score[source, target]) or score[source, target] <= 0:
                continue
            signed = {
                "mean_shift": mean[source, target],
                "log_sd_shift": scale[source, target],
                "tail_probability_shift": tail[source, target],
            }
            dominant = max(signed, key=lambda key: abs(signed[key]))
            worm_values = np.asarray(
                [matrices[(worm, state, lag, 256, dominant)][source, target] for worm in worm_ids]
            )
            worm_proper = np.asarray(
                [
                    matrices[(worm, state, lag, 256, "proper_energy_penalty")][source, target]
                    for worm in worm_ids
                ]
            )
            proper_low, proper_high = worm_bootstrap_interval(
                worm_proper,
                seed=deterministic_seed("edge_proper", state, lag, source, target),
            )
            signed_low, signed_high = worm_bootstrap_interval(
                worm_values,
                seed=deterministic_seed("edge_signed", state, lag, source, target),
            )
            sign_consistency = float(np.mean(np.sign(worm_values) == np.sign(signed[dominant])))
            edge_relative_error = float(
                abs(w1_n64[source, target] - w1[source, target])
                / max(abs(w1[source, target]), 1e-12)
            )
            signed_p = float(ttest_1samp(worm_values, 0.0).pvalue)
            edge_rows.append(
                {
                    "state": state,
                    "lag_frames": lag,
                    "lag_seconds": lag / 4.0,
                    "source": neurons[source],
                    "target": neurons[target],
                    "source_index": source,
                    "target_index": target,
                    "selection_score": float(score[source, target]),
                    "wasserstein1": float(w1[source, target]),
                    "proper_energy_penalty": float(proper[source, target]),
                    "proper_ci_low": proper_low,
                    "proper_ci_high": proper_high,
                    "mean_shift": float(mean[source, target]),
                    "log_sd_shift": float(scale[source, target]),
                    "tail_probability_shift": float(tail[source, target]),
                    "dominant_signed_metric": dominant,
                    "signed_ci_low": signed_low,
                    "signed_ci_high": signed_high,
                    "signed_p_value": signed_p,
                    "worm_sign_consistency": sign_consistency,
                    "edge_N64_N256_relative_error": edge_relative_error,
                    "provisional_candidate": bool(
                        sign_consistency >= 0.70
                        and proper_low > 0
                        and (signed_low > 0 or signed_high < 0)
                        and edge_relative_error <= 0.10
                    ),
                }
            )
    edge_frame = pd.DataFrame(edge_rows)
    if not edge_frame.empty:
        edge_frame["signed_bh_q"] = benjamini_hochberg(
            edge_frame.signed_p_value.to_numpy()
        )
        edge_frame["propagation_candidate"] = (
            edge_frame.provisional_candidate & (edge_frame.signed_bh_q < 0.10)
        )
        edge_frame = edge_frame.sort_values(
            ["propagation_candidate", "selection_score"], ascending=[False, False]
        )
    edge_frame.to_csv(output_dir / "candidate_edges.csv", index=False)
    propagation = (
        edge_frame[edge_frame.propagation_candidate]
        if not edge_frame.empty
        else pd.DataFrame()
    )
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_gate_pass": bool(model_gate["all_gates_pass"]),
        "passed_state_lag_cells": [
            {"state": str(row.state), "lag_frames": int(row.lag_frames), "lag_seconds": float(row.lag_seconds)}
            for row in passed.itertuples()
        ],
        "propagation_candidate_edges": int(len(propagation)),
        "paired_rollout_authorized": bool(len(propagation)),
        "temporal_cut_smc_authorized": bool(len(propagation)),
        "selection_firewall": "no atlas/connectome/receptor/SBTG target used",
        "claim_boundary": "model-relative calcium-aware distributional sensitivity, not causal edges, anatomy, or physical delay",
    }
    atomic_json(output_dir / "gate_decision.json", decision)
    lines = [
        "# Calcium-aware paired distributional lag audit",
        "",
        "## Decision",
        "",
        (
            f"**{len(passed)} state×lag cells passed and {len(propagation)} propagation-ready edges remain.** A short paired rollout is authorized."
            if len(propagation)
            else "**No propagation-ready direct distributional edge passed the complete gate.** Temporal-cut SMC remains unauthorized."
        ),
        "",
        "The N=64 draws are a strict prefix of the N=256 common-noise draws. Worms, not particles, are the inferential units.",
        "",
        "| State | Lag (s) | W1 | Proper penalty [95% CI] | N64/256 ρ | Rel. error | Split ρ | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in gate_frame.itertuples():
        lines.append(
            f"| {row.state} | {row.lag_seconds:.2f} | {row.mean_wasserstein1_N256:.6f} | "
            f"{row.mean_proper_energy_penalty:+.6f} [{row.proper_ci_low:+.6f}, {row.proper_ci_high:+.6f}] | "
            f"{row.N64_N256_spearman:.3f} | {row.N64_N256_relative_error:.3f} | "
            f"{row.worm_split_spearman:.3f} | {'yes' if row.all_gates_pass else 'no'} |"
        )
    lines.extend(
        [
            "",
            "No external atlas entered estimation or selection. These are finite contrasts of a learned observational calcium-aware law.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fold-evidence", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model_dir = args.model_dir.resolve()
    model_manifest = json.loads((model_dir / "manifest.json").read_text())
    model_gate = json.loads((model_dir / "gate_decision.json").read_text())
    if not model_gate.get("direct_distributional_sampling_authorized"):
        raise RuntimeError("latent model did not authorize direct distributional sampling")
    cohort = load_cohort(cohort_mode="oh16230_head")
    if model_manifest["stimulus_schema_fingerprint"] != cohort.stimulus_schema_fingerprint:
        raise RuntimeError("latent model stimulus schema mismatch")
    folds = load_folds(args.fold_evidence.resolve(), cohort)
    basis = np.load(model_dir / "lag_basis.npy")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "protocol": "paired common-normal calcium-aware distributional lag audit v1",
        "model_dir": str(model_dir),
        "model_gate": model_gate,
        "fold_evidence": str(args.fold_evidence.resolve()),
        "particle_counts": list(PARTICLE_COUNTS),
        "particle_nesting": "N64 is the first 64 common-normal draws of N256",
        "states": list(STATES),
        "lag_frames": list(LAGS),
        "lag_seconds": [lag / cohort.fps for lag in LAGS],
        "structural_negative_control": "lag 40 / 10 s is outside the fitted 32-frame smooth latent-history basis",
        "resampling": "conditional innovation residual q25-to-q75 replacement at one source and lag",
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "selection_firewall": "no atlas/connectome/receptor/SBTG target used",
        "claim_boundary": "model-relative calcium-aware distributional sensitivity, not causal edges, anatomy, or physical delay",
    }
    atomic_json(output / "manifest.json", manifest)
    records = []
    for fold in range(5):
        records.append(run_fold(cohort, folds, fold, model_dir, basis, output))
    pd.DataFrame(records).to_csv(output / "fold_status.csv", index=False)
    decision = analyze(output, model_dir)
    manifest["status"] = "complete"
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["gate_decision"] = decision
    atomic_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
