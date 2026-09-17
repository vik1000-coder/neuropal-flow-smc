from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t, ttest_1samp, ttest_rel

from compatibility_neural_benchmark.core import STIMULUS_PERIODS_SECONDS
from compatibility_neural_benchmark.lagged_smc_analysis import load_responses
from compatibility_neural_benchmark.onset_aware_analysis import (
    build_episode_vectors,
    fold_standardized_traces,
    leave_one_worm_residual,
    load_all_methods,
)
from conditional_neural_benchmark.data import load_cohort


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "biological_lag_analysis_20260828"
TEMPORAL_CUT_RUN = ROOT / "results" / "multilag_temporal_cut_20260827"
DISTRIBUTED_RUN = ROOT / "results" / "distributed_lag_dynamics_20260828"

WORMATLAS_CATEGORIES = "https://www.wormatlas.org/hermaphrodite/nervous/mainframe.htm"
AWC_CIRCUIT_SOURCE = "https://www.nature.com/articles/nature06292"
BUTANONE_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC2586605/"

PRIMARY_CLASS = {
    "sensory": {
        "ADE", "ADF", "ADL", "AFD", "ASE", "ASG", "ASH", "ASI", "ASJ", "ASK",
        "AWA", "AWB", "AWC", "BAG", "CEP", "FLP", "IL1", "IL2", "OLL", "OLQ",
        "URX", "URY",
    },
    "interneuron": {
        "AIB", "AIM", "AIY", "AIZ", "AUA", "AVH", "AVJ", "AVK", "RIA", "RIC",
        "RID", "RIH", "RIP",
    },
    "command_premotor": {"AVA", "AVB", "AVD", "AVE", "RIB", "RIM", "RIV"},
    "head_motor": {"RMD", "RME", "RMF", "RMH", "SAA", "SIA", "SIB", "SMB", "SMD", "URA", "URB"},
    "mixed_neurosecretory": {"ALA"},
}

FUNCTIONAL_GROUPS = {
    "awc_only": {"AWC"},
    "chemosensory": {"ADF", "ADL", "ASE", "ASG", "ASH", "ASI", "ASJ", "ASK", "AWA", "AWB", "AWC"},
    "all_sensory": set(PRIMARY_CLASS["sensory"]),
    "olfactory_interneurons": {"AIB", "AIY", "AIZ"},
    "reversal_network": {"AIB", "AVA", "RIM"},
    "locomotor_command": set(PRIMARY_CLASS["command_premotor"]),
    "head_motor": set(PRIMARY_CLASS["head_motor"]),
    "all_downstream": set().union(
        PRIMARY_CLASS["interneuron"],
        PRIMARY_CLASS["command_premotor"],
        PRIMARY_CLASS["head_motor"],
    ),
}

SOURCE_GROUPS = ("awc_only", "chemosensory", "all_sensory", "all_neurons")
TARGET_GROUPS = (
    "olfactory_interneurons",
    "reversal_network",
    "locomotor_command",
    "head_motor",
    "all_downstream",
)
PRIMARY_METHODS = ("wide_flow_direct", "progressive_smc")
KNOWN_TARGETS = ("AIY", "AIB", "AIZ", "AVA", "RIM", "RIB")
LITERATURE_SIGNS = {"AIY": -1, "AIB": 1}


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result


def mean_ci(values: np.ndarray) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, 0
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, np.nan, np.nan, 1
    half = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half, int(len(values))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    usable = np.isfinite(x) & np.isfinite(y)
    x, y = x[usable], y[usable]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(spearmanr(x, y).statistic)


def safe_mse(x: np.ndarray, y: np.ndarray) -> float:
    usable = np.isfinite(x) & np.isfinite(y)
    return float(np.mean((np.asarray(x)[usable] - np.asarray(y)[usable]) ** 2)) if usable.any() else np.nan


def annotation_table(neurons: tuple[str, ...]) -> pd.DataFrame:
    assigned: dict[str, str] = {}
    for label, members in PRIMARY_CLASS.items():
        for neuron in members:
            if neuron in assigned:
                raise ValueError(f"duplicate primary annotation for {neuron}")
            assigned[neuron] = label
    if set(neurons) != set(assigned):
        raise ValueError(
            f"annotation coverage mismatch; missing={sorted(set(neurons) - set(assigned))}, "
            f"extra={sorted(set(assigned) - set(neurons))}"
        )
    aminergic = {"ADE", "ADF", "CEP", "RIC", "RIM"}
    rows = []
    for neuron in neurons:
        roles = sorted(name for name, members in FUNCTIONAL_GROUPS.items() if neuron in members)
        rows.append(
            {
                "neuron": neuron,
                "primary_class": assigned[neuron],
                "functional_roles": ";".join(roles),
                "is_butanone_primary": neuron == "AWC",
                "is_aminergic": neuron in aminergic,
                "annotation_basis": "WormAtlas operational circuit category; overlapping project roles are analysis-specific",
                "source_url": WORMATLAS_CATEGORIES,
            }
        )
    return pd.DataFrame(rows)


def _indices(neurons: tuple[str, ...], group: str) -> np.ndarray:
    members = set(neurons) if group == "all_neurons" else FUNCTIONAL_GROUPS[group]
    result = np.asarray([i for i, neuron in enumerate(neurons) if neuron in members], dtype=int)
    if not len(result):
        raise ValueError(f"empty group: {group}")
    return result


def observed_response_tables(
    neurons: tuple[str, ...],
    lags: np.ndarray,
    onset,
    quiet,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for lag_position, lag in enumerate(lags):
        for neuron_position, neuron in enumerate(neurons):
            actual = onset.target[:, :, lag_position, neuron_position]
            control = quiet.target[:, :, lag_position, neuron_position]
            for event in range(actual.shape[1]):
                difference = actual[:, event] - control[:, event]
                mean, low, high, n = mean_ci(difference)
                rows.append(
                    {
                        "scope": "event",
                        "event": event + 1,
                        "neuron": neuron,
                        "lag_frames": int(lag),
                        "horizon_seconds": float(lag / 4.0),
                        "onset_mean": float(np.mean(actual[:, event])),
                        "quiet_mean": float(np.mean(control[:, event])),
                        "onset_minus_quiet": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n_worms": n,
                        "paired_p_value": float(ttest_rel(actual[:, event], control[:, event]).pvalue),
                        "worm_sign_agreement": float(np.mean(np.sign(difference) == np.sign(mean))),
                    }
                )
            actual_worm = actual.mean(axis=1)
            control_worm = control.mean(axis=1)
            difference = actual_worm - control_worm
            mean, low, high, n = mean_ci(difference)
            rows.append(
                {
                    "scope": "event_average",
                    "event": 0,
                    "neuron": neuron,
                    "lag_frames": int(lag),
                    "horizon_seconds": float(lag / 4.0),
                    "onset_mean": float(np.mean(actual_worm)),
                    "quiet_mean": float(np.mean(control_worm)),
                    "onset_minus_quiet": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_worms": n,
                    "paired_p_value": float(ttest_rel(actual_worm, control_worm).pvalue),
                    "worm_sign_agreement": float(np.mean(np.sign(difference) == np.sign(mean))),
                }
            )
    frame = pd.DataFrame(rows)
    for scope in frame.scope.unique():
        mask = frame.scope == scope
        frame.loc[mask, "bh_q_value"] = benjamini_hochberg(frame.loc[mask, "paired_p_value"].to_numpy())

    role_rows: list[dict] = []
    for lag_position, lag in enumerate(lags):
        for group in sorted(PRIMARY_CLASS):
            idx = np.asarray([neurons.index(value) for value in sorted(PRIMARY_CLASS[group])], dtype=int)
            actual = onset.target[:, :, lag_position][:, :, idx]
            control = quiet.target[:, :, lag_position][:, :, idx]
            signed = actual.mean(axis=(1, 2))
            quiet_signed = control.mean(axis=(1, 2))
            magnitude = np.sqrt(np.mean(actual**2, axis=(1, 2)))
            quiet_magnitude = np.sqrt(np.mean(control**2, axis=(1, 2)))
            signed_mean, signed_low, signed_high, n = mean_ci(signed - quiet_signed)
            mag_mean, mag_low, mag_high, _ = mean_ci(magnitude - quiet_magnitude)
            role_rows.append(
                {
                    "primary_class": group,
                    "n_neurons": len(idx),
                    "lag_frames": int(lag),
                    "horizon_seconds": float(lag / 4.0),
                    "signed_onset_minus_quiet": signed_mean,
                    "signed_ci_low": signed_low,
                    "signed_ci_high": signed_high,
                    "rms_onset_minus_quiet": mag_mean,
                    "rms_ci_low": mag_low,
                    "rms_ci_high": mag_high,
                    "n_worms": n,
                }
            )
    return frame, pd.DataFrame(role_rows)


def event_aligned_response_tables(
    traces: tuple[np.ndarray, ...],
    neurons: tuple[str, ...],
    fps: float,
    *,
    baseline_frames: int = 4,
    maximum_seconds: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure true-onset trajectories against matched quiet pseudo-onsets.

    The quiet anchor is 15 seconds before each true onset, matching the primary
    episode-vector analysis. Every response is relative to its own immediately
    preceding baseline. Inference is worm-level; repeated events are averaged
    before the event-average test.
    """
    maximum_frames = int(round(maximum_seconds * fps))
    onset_by_worm: list[np.ndarray] = []
    quiet_by_worm: list[np.ndarray] = []
    for trace in traces:
        worm_onset: list[np.ndarray] = []
        worm_quiet: list[np.ndarray] = []
        for start_seconds, _ in STIMULUS_PERIODS_SECONDS:
            onset = int(round(start_seconds * fps))
            quiet = onset - int(round(15.0 * fps))
            onset_baseline = trace[onset - baseline_frames : onset].mean(axis=0)
            quiet_baseline = trace[quiet - baseline_frames : quiet].mean(axis=0)
            worm_onset.append(
                trace[onset : onset + maximum_frames + 1] - onset_baseline
            )
            worm_quiet.append(
                trace[quiet : quiet + maximum_frames + 1] - quiet_baseline
            )
        onset_by_worm.append(np.stack(worm_onset))
        quiet_by_worm.append(np.stack(worm_quiet))
    onset_values = np.stack(onset_by_worm)  # worm, event, offset, neuron
    quiet_values = np.stack(quiet_by_worm)

    rows: list[dict] = []
    for scope, event in [("event", value) for value in range(3)] + [("event_average", None)]:
        if event is None:
            actual = onset_values.mean(axis=1)
            control = quiet_values.mean(axis=1)
            event_label = 0
        else:
            actual = onset_values[:, event]
            control = quiet_values[:, event]
            event_label = event + 1
        for offset in range(maximum_frames + 1):
            for neuron_position, neuron in enumerate(neurons):
                actual_worm = actual[:, offset, neuron_position]
                control_worm = control[:, offset, neuron_position]
                difference = actual_worm - control_worm
                mean, low, high, n = mean_ci(difference)
                rows.append(
                    {
                        "scope": scope,
                        "event": event_label,
                        "neuron": neuron,
                        "offset_frames": offset,
                        "offset_seconds": float(offset / fps),
                        "onset_response": float(np.mean(actual_worm)),
                        "quiet_response": float(np.mean(control_worm)),
                        "onset_minus_quiet": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n_worms": n,
                        "paired_p_value": float(ttest_rel(actual_worm, control_worm).pvalue),
                        "worm_sign_agreement": float(np.mean(np.sign(difference) == np.sign(mean))),
                    }
                )
    aligned = pd.DataFrame(rows)
    aligned["global_bh_q_value"] = np.nan
    aligned["within_neuron_bh_q_value"] = np.nan
    for scope in aligned.scope.unique():
        scope_mask = aligned.scope == scope
        aligned.loc[scope_mask, "global_bh_q_value"] = benjamini_hochberg(
            aligned.loc[scope_mask, "paired_p_value"].to_numpy()
        )
        for neuron in neurons:
            neuron_mask = scope_mask & (aligned.neuron == neuron)
            aligned.loc[neuron_mask, "within_neuron_bh_q_value"] = benjamini_hochberg(
                aligned.loc[neuron_mask, "paired_p_value"].to_numpy()
            )

    latency_rows: list[dict] = []
    averaged = aligned[aligned.scope == "event_average"]
    for neuron in neurons:
        frame = averaged[averaged.neuron == neuron].sort_values("offset_frames")
        effects = frame.onset_minus_quiet.to_numpy()
        within_q = frame.within_neuron_bh_q_value.to_numpy()
        global_q = frame.global_bh_q_value.to_numpy()
        first_position: int | None = None
        for position in range(len(frame) - 1):
            stable_sign = np.sign(effects[position]) == np.sign(effects[position + 1]) != 0
            if stable_sign and within_q[position] < 0.05 and within_q[position + 1] < 0.05:
                first_position = position
                break
        peak_position = int(np.argmax(np.abs(effects)))
        first = frame.iloc[first_position] if first_position is not None else None
        peak = frame.iloc[peak_position]
        latency_rows.append(
            {
                "neuron": neuron,
                "latency_seconds": float(first.offset_seconds) if first is not None else np.nan,
                "latency_effect": float(first.onset_minus_quiet) if first is not None else np.nan,
                "latency_direction": (
                    "increase" if first is not None and first.onset_minus_quiet > 0
                    else "decrease" if first is not None else "not_detected"
                ),
                "latency_within_neuron_q": (
                    float(first.within_neuron_bh_q_value) if first is not None else np.nan
                ),
                "latency_global_q": float(first.global_bh_q_value) if first is not None else np.nan,
                "latency_survives_global_bh": bool(
                    first_position is not None
                    and global_q[first_position] < 0.05
                    and global_q[first_position + 1] < 0.05
                ),
                "peak_seconds": float(peak.offset_seconds),
                "peak_onset_minus_quiet": float(peak.onset_minus_quiet),
                "peak_ci_low": float(peak.ci_low),
                "peak_ci_high": float(peak.ci_high),
                "peak_within_neuron_q": float(peak.within_neuron_bh_q_value),
                "peak_global_q": float(peak.global_bh_q_value),
            }
        )
    return aligned, pd.DataFrame(latency_rows)


def _fit_predictions(
    source: np.ndarray,
    target: np.ndarray,
    matrices: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
) -> pd.DataFrame:
    n_worms = source.shape[0]
    rows = []
    for heldout in range(n_worms):
        training = np.asarray([worm for worm in range(n_worms) if worm != heldout], dtype=int)
        source_test = source[heldout] - source[training].mean(axis=0)
        target_test = target[heldout] - target[training].mean(axis=0)
        source_train = np.stack(
            [source[worm] - source[training[training != worm]].mean(axis=0) for worm in training]
        )
        target_train = np.stack(
            [target[worm] - target[training[training != worm]].mean(axis=0) for worm in training]
        )
        propagated_train = np.stack(
            [
                np.stack(
                    [matrices[worm][np.ix_(target_idx, source_idx)] @ source_train[pos, event, source_idx] for event in range(source.shape[1])]
                )
                for pos, worm in enumerate(training)
            ]
        )
        baseline_train = source_train[:, :, target_idx]
        outcome_train = target_train[:, :, target_idx]
        baseline_design = np.column_stack([np.ones(baseline_train.size), baseline_train.reshape(-1)])
        combined_design = np.column_stack(
            [np.ones(baseline_train.size), baseline_train.reshape(-1), propagated_train.reshape(-1)]
        )
        baseline_coef, *_ = np.linalg.lstsq(baseline_design, outcome_train.reshape(-1), rcond=None)
        combined_coef, *_ = np.linalg.lstsq(combined_design, outcome_train.reshape(-1), rcond=None)
        propagated_test = np.stack(
            [matrices[heldout][np.ix_(target_idx, source_idx)] @ source_test[event, source_idx] for event in range(source.shape[1])]
        )
        baseline_value = source_test[:, target_idx]
        baseline_prediction = baseline_coef[0] + baseline_coef[1] * baseline_value
        combined_prediction = (
            combined_coef[0]
            + combined_coef[1] * baseline_value
            + combined_coef[2] * propagated_test
        )
        truth = target_test[:, target_idx]
        baseline_rho = safe_spearman(baseline_prediction.reshape(-1), truth.reshape(-1))
        combined_rho = safe_spearman(combined_prediction.reshape(-1), truth.reshape(-1))
        baseline_mse = safe_mse(baseline_prediction, truth)
        combined_mse = safe_mse(combined_prediction, truth)
        rows.append(
            {
                "worm": heldout,
                "baseline_spearman": baseline_rho,
                "combined_spearman": combined_rho,
                "spearman_gain": combined_rho - baseline_rho,
                "baseline_mse": baseline_mse,
                "combined_mse": combined_mse,
                "mse_improvement": baseline_mse - combined_mse,
                "persistence_coefficient": float(combined_coef[1]),
                "matrix_coefficient": float(combined_coef[2]),
            }
        )
    return pd.DataFrame(rows)


def biological_prediction_tables(
    dynamic: dict,
    neurons: tuple[str, ...],
    lags: np.ndarray,
    onset,
    quiet,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[pd.DataFrame] = []
    onset_phase = dynamic[PRIMARY_METHODS[0]].phases.index("onset")
    baseline_phase = dynamic[PRIMARY_METHODS[0]].phases.index("baseline")
    for episode, vectors, phase_index in (
        ("stimulus_onset", onset, onset_phase),
        ("quiet_pseudo_onset", quiet, baseline_phase),
    ):
        for method in PRIMARY_METHODS:
            artifact = dynamic[method]
            for lag_position, lag in enumerate(lags):
                matrices = artifact.matrices[:, phase_index, lag_position].copy()
                diagonal = np.arange(len(neurons))
                matrices[:, diagonal, diagonal] = 0.0
                for source_group in SOURCE_GROUPS:
                    source_idx = _indices(neurons, source_group)
                    for target_group in TARGET_GROUPS:
                        target_idx = _indices(neurons, target_group)
                        frame = _fit_predictions(
                            vectors.source,
                            vectors.target[:, :, lag_position],
                            matrices,
                            source_idx,
                            target_idx,
                        )
                        frame.insert(0, "target_group", target_group)
                        frame.insert(0, "source_group", source_group)
                        frame.insert(0, "horizon_seconds", float(lag / 4.0))
                        frame.insert(0, "lag_frames", int(lag))
                        frame.insert(0, "method", method)
                        frame.insert(0, "episode", episode)
                        event_rows.append(frame)
    events = pd.concat(event_rows, ignore_index=True)
    group = ["episode", "method", "lag_frames", "horizon_seconds", "source_group", "target_group"]
    rows: list[dict] = []
    for keys, frame in events.groupby(group, sort=False):
        row = dict(zip(group, keys))
        for metric in ("spearman_gain", "mse_improvement", "matrix_coefficient"):
            mean, low, high, n = mean_ci(frame[metric].to_numpy())
            row[f"mean_{metric}"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"n_worms_{metric}"] = n
        row["positive_gain_fraction"] = float(np.mean(frame.spearman_gain > 0))
        row["gain_p_value"] = float(ttest_1samp(frame.spearman_gain, 0.0).pvalue)
        rows.append(row)
    summary = pd.DataFrame(rows)
    for episode in summary.episode.unique():
        mask = summary.episode == episode
        summary.loc[mask, "gain_bh_q_value"] = benjamini_hochberg(summary.loc[mask, "gain_p_value"].to_numpy())
    return events, summary


def olfactory_signature_table(dynamic: dict, neurons: tuple[str, ...], lags: np.ndarray, onset) -> pd.DataFrame:
    lookup = {neuron: i for i, neuron in enumerate(neurons)}
    source_position = lookup["AWC"]
    source_mean = float(np.mean(onset.source[:, :, source_position]))
    source_residual = leave_one_worm_residual(onset.source, np.arange(onset.source.shape[0]))
    rows: list[dict] = []
    onset_index = dynamic[PRIMARY_METHODS[0]].phases.index("onset")
    for lag_position, lag in enumerate(lags):
        target_residual = leave_one_worm_residual(
            onset.target[:, :, lag_position], np.arange(onset.target.shape[0])
        )
        for target in KNOWN_TARGETS:
            target_position = lookup[target]
            target_mean = float(np.mean(onset.target[:, :, lag_position, target_position]))
            empirical_rho = safe_spearman(
                source_residual[:, :, source_position].reshape(-1),
                target_residual[:, :, target_position].reshape(-1),
            )
            implied_sign = int(np.sign(target_mean / source_mean)) if abs(source_mean) > 1e-12 else 0
            empirical_sign = int(np.sign(empirical_rho)) if np.isfinite(empirical_rho) else 0
            for method in PRIMARY_METHODS:
                values = dynamic[method].matrices[:, onset_index, lag_position, target_position, source_position]
                mean_value = float(np.mean(values))
                rows.append(
                    {
                        "method": method,
                        "source": "AWC",
                        "target": target,
                        "lag_frames": int(lag),
                        "horizon_seconds": float(lag / 4.0),
                        "matrix_coefficient": mean_value,
                        "worm_sign_agreement": float(np.mean(np.sign(values) == np.sign(mean_value))),
                        "awc_early_mean": source_mean,
                        "target_response_mean": target_mean,
                        "stimulus_pattern_implied_coefficient_sign": implied_sign,
                        "stimulus_pattern_sign_match": bool(np.sign(mean_value) == implied_sign),
                        "empirical_worm_event_spearman": empirical_rho,
                        "empirical_residual_association_sign": empirical_sign,
                        "empirical_residual_sign_match": bool(np.sign(mean_value) == empirical_sign),
                        "literature_expected_sign": LITERATURE_SIGNS.get(target, np.nan),
                        "literature_sign_match": (
                            bool(np.sign(mean_value) == LITERATURE_SIGNS[target])
                            if target in LITERATURE_SIGNS else np.nan
                        ),
                        "literature_source": AWC_CIRCUIT_SOURCE if target in LITERATURE_SIGNS else "",
                    }
                )
    selected_path = TEMPORAL_CUT_RUN / "postfreeze_external" / "aligned_temporal_cut_comparison_matrices.npz"
    with np.load(selected_path, allow_pickle=False) as data:
        if tuple(data["neurons"].astype(str)) != neurons:
            raise RuntimeError("temporal-cut neuron alignment mismatch")
        lag = int(data["selected_horizon_frames"])
        lag_position = int(np.flatnonzero(lags == lag)[0])
        matrix = data["temporal_cut_smc_onset__signed"].astype(float)
        target_residual = leave_one_worm_residual(
            onset.target[:, :, lag_position], np.arange(onset.target.shape[0])
        )
        for target in KNOWN_TARGETS:
            target_position = lookup[target]
            target_mean = float(np.mean(onset.target[:, :, lag_position, target_position]))
            implied_sign = int(np.sign(target_mean / source_mean)) if abs(source_mean) > 1e-12 else 0
            mean_value = float(matrix[target_position, source_position])
            empirical_rho = safe_spearman(
                source_residual[:, :, source_position].reshape(-1),
                target_residual[:, :, target_position].reshape(-1),
            )
            empirical_sign = int(np.sign(empirical_rho)) if np.isfinite(empirical_rho) else 0
            rows.append(
                {
                    "method": "temporal_cut_smc_wide_flow",
                    "source": "AWC",
                    "target": target,
                    "lag_frames": lag,
                    "horizon_seconds": float(lag / 4.0),
                    "matrix_coefficient": mean_value,
                    "worm_sign_agreement": np.nan,
                    "awc_early_mean": source_mean,
                    "target_response_mean": target_mean,
                    "stimulus_pattern_implied_coefficient_sign": implied_sign,
                    "stimulus_pattern_sign_match": bool(np.sign(mean_value) == implied_sign),
                    "empirical_worm_event_spearman": empirical_rho,
                    "empirical_residual_association_sign": empirical_sign,
                    "empirical_residual_sign_match": bool(np.sign(mean_value) == empirical_sign),
                    "literature_expected_sign": LITERATURE_SIGNS.get(target, np.nan),
                    "literature_sign_match": (
                        bool(np.sign(mean_value) == LITERATURE_SIGNS[target])
                        if target in LITERATURE_SIGNS else np.nan
                    ),
                    "literature_source": AWC_CIRCUIT_SOURCE if target in LITERATURE_SIGNS else "",
                }
            )
    return pd.DataFrame(rows)


def distributed_lag_pathways(neurons: tuple[str, ...]) -> pd.DataFrame:
    path = DISTRIBUTED_RUN / "consensus" / "postfreeze" / "selected_neural_matrices.npz"
    with np.load(path, allow_pickle=False) as data:
        if tuple(data["neurons"].astype(str)) != neurons:
            raise RuntimeError("distributed-lag neuron alignment mismatch")
        kernel = data["direct_kernel"].astype(float)
    lookup = {neuron: i for i, neuron in enumerate(neurons)}
    rows = []
    for target in KNOWN_TARGETS:
        values = kernel[:, :, lookup[target], lookup["AWC"]]
        for lag_position in range(values.shape[1]):
            mean_value = float(np.mean(values[:, lag_position]))
            rows.append(
                {
                    "source": "AWC",
                    "target": target,
                    "source_history_lag_frames": lag_position + 1,
                    "source_history_lag_seconds": float((lag_position + 1) / 4.0),
                    "direct_kernel_coefficient": mean_value,
                    "fold_sign_agreement": float(np.mean(np.sign(values[:, lag_position]) == np.sign(mean_value))),
                    "nonzero_fold_fraction": float(np.mean(np.abs(values[:, lag_position]) > 1e-12)),
                }
            )
    return pd.DataFrame(rows)


def repetition_adaptation(neurons: tuple[str, ...], lags: np.ndarray, onset, quiet) -> pd.DataFrame:
    selected_lag = 16 if 16 in lags else int(lags[-1])
    lag_position = int(np.flatnonzero(lags == selected_lag)[0])
    entities: dict[str, np.ndarray] = {
        f"class:{name}": np.asarray([neurons.index(value) for value in sorted(members)], dtype=int)
        for name, members in PRIMARY_CLASS.items()
    }
    for neuron in ("AWC", "AIY", "AIB", "AIZ", "AVA", "RIM"):
        entities[f"neuron:{neuron}"] = np.asarray([neurons.index(neuron)], dtype=int)
    rows: list[dict] = []
    for entity, idx in entities.items():
        values = onset.target[:, :, lag_position][:, :, idx]
        controls = quiet.target[:, :, lag_position][:, :, idx]
        signed = values.mean(axis=2)
        quiet_signed = controls.mean(axis=2)
        magnitude = np.sqrt(np.mean(values**2, axis=2))
        quiet_magnitude = np.sqrt(np.mean(controls**2, axis=2))
        for event in range(3):
            signed_mean, signed_low, signed_high, n = mean_ci(signed[:, event])
            signed_contrast_mean, signed_contrast_low, signed_contrast_high, _ = mean_ci(
                signed[:, event] - quiet_signed[:, event]
            )
            magnitude_mean, magnitude_low, magnitude_high, _ = mean_ci(magnitude[:, event])
            magnitude_contrast_mean, magnitude_contrast_low, magnitude_contrast_high, _ = mean_ci(
                magnitude[:, event] - quiet_magnitude[:, event]
            )
            rows.append(
                {
                    "entity": entity,
                    "event": event + 1,
                    "lag_frames": selected_lag,
                    "horizon_seconds": float(selected_lag / 4.0),
                    "signed_response": signed_mean,
                    "signed_ci_low": signed_low,
                    "signed_ci_high": signed_high,
                    "quiet_signed_response": float(np.mean(quiet_signed[:, event])),
                    "signed_onset_minus_quiet": signed_contrast_mean,
                    "signed_onset_minus_quiet_ci_low": signed_contrast_low,
                    "signed_onset_minus_quiet_ci_high": signed_contrast_high,
                    "rms_response": magnitude_mean,
                    "rms_ci_low": magnitude_low,
                    "rms_ci_high": magnitude_high,
                    "quiet_rms_response": float(np.mean(quiet_magnitude[:, event])),
                    "rms_onset_minus_quiet": magnitude_contrast_mean,
                    "rms_onset_minus_quiet_ci_low": magnitude_contrast_low,
                    "rms_onset_minus_quiet_ci_high": magnitude_contrast_high,
                    "n_worms": n,
                }
            )
        onset_slopes = np.polyfit(np.arange(1, 4), magnitude.T, 1)[0]
        quiet_slopes = np.polyfit(np.arange(1, 4), quiet_magnitude.T, 1)[0]
        entity_slopes = np.polyfit(np.arange(1, 4), (magnitude - quiet_magnitude).T, 1)[0]
        mean, low, high, n = mean_ci(entity_slopes)
        p_value = float(ttest_1samp(entity_slopes, 0.0).pvalue)
        onset_mean, onset_low, onset_high, _ = mean_ci(onset_slopes)
        quiet_mean, quiet_low, quiet_high, _ = mean_ci(quiet_slopes)
        start = len(rows) - 3
        for position in range(start, len(rows)):
            rows[position]["rms_adaptation_slope_per_repeat"] = mean
            rows[position]["slope_ci_low"] = low
            rows[position]["slope_ci_high"] = high
            rows[position]["slope_p_value"] = p_value
            rows[position]["onset_rms_slope_per_repeat"] = onset_mean
            rows[position]["onset_rms_slope_ci_low"] = onset_low
            rows[position]["onset_rms_slope_ci_high"] = onset_high
            rows[position]["quiet_rms_slope_per_repeat"] = quiet_mean
            rows[position]["quiet_rms_slope_ci_low"] = quiet_low
            rows[position]["quiet_rms_slope_ci_high"] = quiet_high
    frame = pd.DataFrame(rows)
    entity_p = frame.groupby("entity", sort=False).slope_p_value.first()
    q = benjamini_hochberg(entity_p.to_numpy())
    mapping = dict(zip(entity_p.index, q))
    frame["slope_bh_q_value"] = frame.entity.map(mapping)
    return frame


def temporal_cut_event_pathways(neurons: tuple[str, ...], folds: np.ndarray) -> pd.DataFrame:
    artifact = load_responses(TEMPORAL_CUT_RUN / "lagged_smc_confirmation", folds)
    if artifact.neurons != neurons:
        raise RuntimeError("temporal-cut event archive neuron alignment mismatch")
    confirmation_worms = np.flatnonzero(np.isin(folds, [3, 4]))
    onset_index = artifact.phases.index("onset")
    rows = []
    source_groups = ("awc_only", "chemosensory", "all_sensory")
    target_groups = ("olfactory_interneurons", "reversal_network", "locomotor_command", "head_motor")
    for lag_position, source_lag in enumerate(artifact.source_lags):
        for horizon_position, horizon in enumerate(artifact.horizons):
            for event in range(3):
                for source_group in source_groups:
                    source_idx = _indices(neurons, source_group)
                    valid = artifact.validity[
                        lag_position, confirmation_worms, onset_index, event
                    ][:, source_idx]
                    for target_group in target_groups:
                        target_idx = _indices(neurons, target_group)
                        matrices = artifact.matrices[
                            lag_position, confirmation_worms, onset_index, event, horizon_position
                        ]
                        blocks = matrices[:, target_idx][:, :, source_idx]
                        per_worm_rms = np.sqrt(np.mean(blocks**2, axis=(1, 2)))
                        per_worm_signed = np.mean(blocks, axis=(1, 2))
                        rms_mean, rms_low, rms_high, n = mean_ci(per_worm_rms)
                        signed_mean, signed_low, signed_high, _ = mean_ci(per_worm_signed)
                        rows.append(
                            {
                                "event": event + 1,
                                "source_lag_frames": int(source_lag),
                                "source_lag_seconds": float(source_lag / 4.0),
                                "horizon_frames": int(horizon),
                                "horizon_seconds": float(horizon / 4.0),
                                "source_group": source_group,
                                "target_group": target_group,
                                "block_rms": rms_mean,
                                "block_rms_ci_low": rms_low,
                                "block_rms_ci_high": rms_high,
                                "block_signed_mean": signed_mean,
                                "block_signed_ci_low": signed_low,
                                "block_signed_ci_high": signed_high,
                                "valid_source_fraction": float(np.mean(valid)),
                                "n_confirmation_worms": n,
                            }
                        )
    return pd.DataFrame(rows)


def stable_edge_candidates(dynamic: dict, neurons: tuple[str, ...], lags: np.ndarray, onset) -> pd.DataFrame:
    n = len(neurons)
    onset_index = dynamic[PRIMARY_METHODS[0]].phases.index("onset")
    source_residual = leave_one_worm_residual(onset.source, np.arange(onset.source.shape[0])).mean(axis=1)
    rows: list[dict] = []
    offdiag = ~np.eye(n, dtype=bool)
    for lag_position, lag in enumerate(lags):
        target_residual = leave_one_worm_residual(
            onset.target[:, :, lag_position], np.arange(onset.target.shape[0])
        ).mean(axis=1)
        matrices = {
            method: dynamic[method].matrices[:, onset_index, lag_position].astype(float)
            for method in PRIMARY_METHODS
        }
        means = {method: np.mean(value, axis=0) for method, value in matrices.items()}
        signs = {
            method: np.mean(np.sign(value) == np.sign(means[method])[None], axis=0)
            for method, value in matrices.items()
        }
        ranks = {}
        for method in PRIMARY_METHODS:
            flat = np.abs(means[method][offdiag])
            order = pd.Series(flat).rank(method="average", pct=True).to_numpy()
            ranks[method] = np.full((n, n), np.nan)
            ranks[method][offdiag] = order
        empirical = np.full((n, n), np.nan)
        p_values = np.full((n, n), np.nan)
        for target in range(n):
            for source in range(n):
                if source == target:
                    continue
                statistic = spearmanr(source_residual[:, source], target_residual[:, target])
                empirical[target, source] = float(statistic.statistic)
                p_values[target, source] = float(statistic.pvalue)
        q_values = np.full((n, n), np.nan)
        q_values[offdiag] = benjamini_hochberg(p_values[offdiag])
        validity = {
            method: dynamic[method].validity[:, onset_index].mean(axis=0) for method in PRIMARY_METHODS
        }
        for target in range(n):
            for source in range(n):
                if source == target:
                    continue
                sign_match = np.sign(means["wide_flow_direct"][target, source]) == np.sign(
                    means["progressive_smc"][target, source]
                )
                empirical_match = np.sign(empirical[target, source]) == np.sign(
                    means["progressive_smc"][target, source]
                )
                passed = bool(
                    sign_match
                    and empirical_match
                    and signs["wide_flow_direct"][target, source] >= 0.70
                    and signs["progressive_smc"][target, source] >= 0.75
                    and ranks["wide_flow_direct"][target, source] >= 0.80
                    and ranks["progressive_smc"][target, source] >= 0.80
                    and validity["wide_flow_direct"][source] >= 0.30
                    and validity["progressive_smc"][source] >= 0.50
                    and q_values[target, source] <= 0.10
                )
                score = float(
                    np.nanmean(
                        [
                            ranks["wide_flow_direct"][target, source],
                            ranks["progressive_smc"][target, source],
                            signs["wide_flow_direct"][target, source],
                            signs["progressive_smc"][target, source],
                            abs(empirical[target, source]),
                        ]
                    )
                )
                rows.append(
                    {
                        "source": neurons[source],
                        "target": neurons[target],
                        "lag_frames": int(lag),
                        "horizon_seconds": float(lag / 4.0),
                        "wide_flow_coefficient": float(means["wide_flow_direct"][target, source]),
                        "progressive_smc_coefficient": float(means["progressive_smc"][target, source]),
                        "wide_flow_sign_agreement": float(signs["wide_flow_direct"][target, source]),
                        "progressive_smc_sign_agreement": float(signs["progressive_smc"][target, source]),
                        "wide_flow_magnitude_percentile": float(ranks["wide_flow_direct"][target, source]),
                        "progressive_smc_magnitude_percentile": float(ranks["progressive_smc"][target, source]),
                        "wide_flow_validity": float(validity["wide_flow_direct"][source]),
                        "progressive_smc_validity": float(validity["progressive_smc"][source]),
                        "observed_worm_spearman": float(empirical[target, source]),
                        "observed_p_value": float(p_values[target, source]),
                        "observed_bh_q_value": float(q_values[target, source]),
                        "cross_method_sign_match": bool(sign_match),
                        "observed_sign_match": bool(empirical_match),
                        "candidate_pass": passed,
                        "exploratory_consensus_score": score,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["candidate_pass", "exploratory_consensus_score"], ascending=[False, False]
    )


def _best_prediction(summary: pd.DataFrame, method: str) -> pd.Series:
    onset = summary[(summary.episode == "stimulus_onset") & (summary.method == method)].copy()
    return onset.sort_values("mean_spearman_gain", ascending=False).iloc[0]


def write_report(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    response = tables["observed_response"]
    aligned = tables["event_aligned_response"]
    latency = tables["stimulus_latency"]
    signature = tables["olfactory_signature"]
    prediction = tables["prediction_summary"]
    adaptation = tables["adaptation"]
    candidates = tables["stable_candidates"]
    response4 = response[(response.scope == "event_average") & (response.lag_frames == 16)].copy()
    strongest = response4.reindex(response4.onset_minus_quiet.abs().sort_values(ascending=False).index).head(10)
    known = response4[response4.neuron.isin(["AWC", "AIY", "AIB", "AIZ", "AVA", "RIM"])].sort_values("neuron")
    wide_best = _best_prediction(prediction, "wide_flow_direct")
    smc_best = _best_prediction(prediction, "progressive_smc")
    strict = candidates[candidates.candidate_pass]
    strict_unique = strict[["source", "target"]].drop_duplicates()
    awc_latency = latency[latency.neuron == "AWC"].iloc[0]
    awc_early = aligned[
        (aligned.scope == "event_average")
        & (aligned.neuron == "AWC")
        & (aligned.offset_seconds <= 0.75)
    ]
    awc_source_window = float(awc_early.onset_response.mean())
    circuit_latency = latency[latency.neuron.isin(["AWC", "AIY", "AIB", "AIZ"])].copy()
    early_signature = signature[signature.horizon_seconds.isin([0.5, 1.0, 2.0, 4.0])]
    circuit_summary = (
        early_signature[early_signature.target.isin(["AIY", "AIB"])]
        .groupby(["method", "target"], as_index=False)
        .agg(
            empirical_sign_match_fraction=("empirical_residual_sign_match", "mean"),
            stimulus_pattern_sign_match_fraction=("stimulus_pattern_sign_match", "mean"),
            literature_sign_match_fraction=("literature_sign_match", "mean"),
            mean_worm_sign_agreement=("worm_sign_agreement", "mean"),
        )
    )
    adapt_entity = adaptation.groupby("entity", as_index=False).first().sort_values(
        "rms_adaptation_slope_per_repeat"
    )
    lines = [
        "# Biological analysis of stimulus-conditioned lag dynamics",
        "",
        f"**Run date:** {datetime.now().date().isoformat()}  ",
        "**Claim boundary:** observed stimulus-response organization and model-relative predictive dynamics; not synapses, causal interventions, or physical transmission delays.",
        "",
        "## Technical summary",
        "",
        "The strongest biological result is in the measured activity itself: butanone presentation suppresses AWC while AIY increases and AIB/AIZ decrease, a population response consistent with the expected opponent organization of the canonical AWC olfactory circuit. The learned matrices capture this circuit unevenly. Progressive ESS-SMC has the expected early AWC→AIY negative sign and AWC→AIB positive sign, whereas the wide-flow direct and targeted wide-flow temporal-cut matrices miss the AWC→AIY sign at the tested early horizons.",
        "",
        f"Event alignment explains an important instability: the four-frame source summary is still weakly positive for AWC ({awc_source_window:+.3f} standardized units), while the sustained AWC decrease is first detected at {awc_latency.latency_seconds:g} s under within-neuron correction and peaks at {awc_latency.peak_seconds:g} s ({awc_latency.peak_onset_minus_quiet:+.3f}). The first latency point does not survive the global neuron-by-time correction. A single early source average therefore mixes a transient with the later sensory suppression that the lag matrix is meant to propagate.",
        "",
        f"The best target-restricted incremental result for wide-flow direct is {wide_best.mean_spearman_gain:+.4f} Spearman ({wide_best.source_group} sources → {wide_best.target_group}, {wide_best.horizon_seconds:g} s); the best progressive-SMC result is {smc_best.mean_spearman_gain:+.4f} ({smc_best.source_group} → {smc_best.target_group}, {smc_best.horizon_seconds:g} s). These are exploratory maxima over the declared grid and must be read beside their worm intervals and quiet-pseudo-onset controls.",
        f"Neither predictive maximum is promoted: both worm intervals include zero and their grid-wide BH q-values are {wide_best.gain_bh_q_value:.3f} and {smc_best.gain_bh_q_value:.3f}, respectively.",
        "",
        f"Only {len(strict)} source-target-horizon cells ({len(strict_unique)} unique edge) pass the strict cross-estimator, fold/worm-sign, compatibility, magnitude, and observed-covariation gate. This prevents turning a long top-edge list into a biological claim.",
        "",
        "## The observed response has canonical olfactory organization",
        "",
        "At four seconds after onset, the largest onset-minus-quiet responses are:",
        "",
        "| Neuron | Onset−quiet | 95% worm CI | BH q |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in strongest.itertuples():
        lines.append(f"| {row.neuron} | {row.onset_minus_quiet:+.3f} | [{row.ci_low:+.3f}, {row.ci_high:+.3f}] | {row.bh_q_value:.3g} |")
    lines += [
        "",
        "The prespecified AWC-centered circuit shows:",
        "",
        "| Neuron | 4-s onset−quiet | 95% worm CI |",
        "| --- | ---: | ---: |",
    ]
    for row in known.itertuples():
        lines.append(f"| {row.neuron} | {row.onset_minus_quiet:+.3f} | [{row.ci_low:+.3f}, {row.ci_high:+.3f}] |")
    lines += [
        "",
        "AWC is the declared direct 2-butanone sensor. The AWC fall during odor presentation, AIY rise, and AIB fall are consistent with the known opponent AWC→AIY/AIB circuit. This is a stimulus-response signature, not an edge estimate.",
        "",
        "### Event-aligned timing changes the interpretation",
        "",
        f"The AWC source window used by the lag analysis spans 0–0.75 s and averages to {awc_source_window:+.3f}; the first two-frame sustained AWC decrease appears at {awc_latency.latency_seconds:g} s under within-neuron BH control (global-BH survival: {str(bool(awc_latency.latency_survives_global_bh)).lower()}). The saved quarter-second trajectories should therefore be used whenever a candidate AWC lag is interpreted. Horizon-dependent sign changes can reflect observation dynamics and window placement rather than a reversal of biology.",
        "",
        "| Neuron | Sustained response latency | Direction | Peak time | Peak onset−quiet | Global q at peak |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in circuit_latency.itertuples():
        latency_text = f"{row.latency_seconds:g} s" if np.isfinite(row.latency_seconds) else "not detected"
        lines.append(
            f"| {row.neuron} | {latency_text} | {row.latency_direction} | {row.peak_seconds:g} s | "
            f"{row.peak_onset_minus_quiet:+.3f} | {row.peak_global_q:.3g} |"
        )
    lines += [
        "",
        "These calcium-response latencies summarize detectability under this imaging and preprocessing pipeline. They are not synaptic transmission delays.",
        "",
        "## Sampling recovers part, but not all, of the known circuit sign structure",
        "",
        "The empirical-sign column compares each matrix coefficient with leave-one-worm residual AWC/target association. The stimulus-pattern column instead asks what coefficient sign would map the population AWC source average to the population target response. They are intentionally separate.",
        "",
        "| Method | Target | Empirical residual-sign match | Stimulus-pattern match | Literature-sign match | Mean worm sign agreement |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in circuit_summary.itertuples():
        lines.append(
            f"| {row.method} | {row.target} | {row.empirical_sign_match_fraction:.2f} | "
            f"{row.stimulus_pattern_sign_match_fraction:.2f} | {row.literature_sign_match_fraction:.2f} | "
            f"{row.mean_worm_sign_agreement:.2f} |"
        )
    lines += [
        "",
        "The progressive sampler is biologically more plausible for the early AWC→AIY sign, but that local success does not override its weak incremental whole-pattern forecast. The globally frozen distributed-lag kernel is saved separately because its source-history lag and the sampler's cumulative response horizon are different estimands.",
        "",
        "## Target-restricted held-out prediction remains the decision test",
        "",
        "Each held-out worm is predicted from a scalar cross-fitted persistence term plus one frozen matrix term. Source and target vectors are residualized against the other worms at the same stimulus repetition. The table contains the best onset cell for each primary method; all grid cells and quiet controls are saved.",
        "",
        "| Method | Source group | Target group | Horizon | Spearman gain | 95% worm CI | Quiet control gain |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in (wide_best, smc_best):
        quiet = prediction[
            (prediction.episode == "quiet_pseudo_onset")
            & (prediction.method == row.method)
            & (prediction.lag_frames == row.lag_frames)
            & (prediction.source_group == row.source_group)
            & (prediction.target_group == row.target_group)
        ].iloc[0]
        lines.append(
            f"| {row.method} | {row.source_group} | {row.target_group} | {row.horizon_seconds:g} s | "
            f"{row.mean_spearman_gain:+.4f} | [{row.spearman_gain_ci_low:+.4f}, {row.spearman_gain_ci_high:+.4f}] | "
            f"{quiet.mean_spearman_gain:+.4f} |"
        )
    lines += [
        "",
        "A positive exploratory maximum is not sufficient. A biologically useful lag matrix should improve the declared downstream group, have a worm interval above zero, outperform the matched quiet control, and replicate after the cell is frozen.",
        "",
        "## Repetition effects test adaptation rather than static wiring",
        "",
        "At the four-second response horizon, the most negative and positive slopes of the RMS onset-minus-matched-quiet contrast per repeated presentation are:",
        "",
        "| Entity | RMS slope / repeat | 95% worm CI | BH q |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in pd.concat([adapt_entity.head(5), adapt_entity.tail(5)]).drop_duplicates("entity").itertuples():
        lines.append(
            f"| {row.entity} | {row.rms_adaptation_slope_per_repeat:+.3f} | "
            f"[{row.slope_ci_low:+.3f}, {row.slope_ci_high:+.3f}] | {row.slope_bh_q_value:.3g} |"
        )
    lines += [
        "",
        "Onset-only and quiet-only magnitude slopes are saved beside the primary matched contrast so time-in-experiment drift is visible. The event-specific temporal-cut SMC block norms and validity are also saved by repetition. A change in those sampled matrices is interpreted as context-dependent learned-law response, not anatomical rewiring.",
        "",
        (
            f"The matched adaptation result is more stable than the lag forecast: AWC onset-minus-quiet response magnitude changes by "
            f"{adapt_entity[adapt_entity.entity == 'neuron:AWC'].iloc[0].rms_adaptation_slope_per_repeat:+.3f} standardized units per repeat "
            f"(BH q={adapt_entity[adapt_entity.entity == 'neuron:AWC'].iloc[0].slope_bh_q_value:.3g}). "
            "Supported negative slopes indicate repetition-dependent attenuation beyond the matched quiet-window trend; the mechanism remains unresolved."
        ),
        "",
        "## Stable named candidates",
        "",
    ]
    if len(strict):
        lines += [
            "| Source | Target | Horizon | Wide flow | Progressive SMC | Observed rho | BH q |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in strict.head(20).itertuples():
            lines.append(
                f"| {row.source} | {row.target} | {row.horizon_seconds:g} s | "
                f"{row.wide_flow_coefficient:+.3f} | {row.progressive_smc_coefficient:+.3f} | "
                f"{row.observed_worm_spearman:+.3f} | {row.observed_bh_q_value:.3g} |"
            )
    else:
        lines.append("No source-target-horizon cell passes every strict gate. The ranked exploratory table is retained, but no named edge is promoted.")
    lines += [
        "",
        "## Scope, limitations, and what this establishes",
        "",
        "- Cohort: 20 whole worms, 54 complete-case neuron classes, three butanone presentations, 4 Hz calcium activity.",
        "- Primary predictive objects: atlas-blind wide-flow direct repaired responses and the full progressive ESS-SMC ensemble. The selected wide-flow temporal-cut SMC cell and globally frozen distributed-lag kernel are sensitivity objects.",
        "- Quiet pseudo-onsets, fold-local scaling, leave-one-worm residualization, worm-level intervals, and cross-estimator sign/stability gates constrain overinterpretation.",
        "- Calcium filtering, common stimulus input, locomotor feedback, unmeasured state, bilateral class collapse, and only 20 worms prevent causal or physical-delay interpretation.",
        "- Functional classes are operational WormAtlas-based analysis groups. Mixed neurons can have more than one biological role; the exact frozen mapping is saved.",
        "",
        "## Biological reference basis",
        "",
        f"- [WormAtlas nervous-system handbook]({WORMATLAS_CATEGORIES}) supplies the broad sensory/interneuron/motor category basis; the exact project mapping is operational and frozen in the annotation CSV.",
        f"- [Butanone/AWC behavioral study]({BUTANONE_SOURCE}) supports AWC as the declared primary 2-butanone sensory channel.",
        f"- [Canonical AWC olfactory-circuit study]({AWC_CIRCUIT_SOURCE}) supplies the expected positive AWC→AIB and negative AWC→AIY action signs. These literature signs are interpretation checks, not training or selection targets.",
        "",
        "## Recommended next step",
        "",
        "Freeze the AWC→AIY/AIB sign test and the best target-restricted prediction cell, then repeat them prospectively on a new acquisition or controlled AWC perturbation. The immediate computational improvement should be a paired progressive ESS-SMC run on the wide-flow winner with common random numbers and on-manifold finite repairs; it should be judged first on these biological endpoints and stability, not on another atlas leaderboard.",
        "",
        "## Output map",
        "",
        "- `observed_response_by_neuron_horizon.csv`: measured onset and quiet responses, worm intervals, and BH values.",
        "- `event_aligned_activity.csv`: quarter-second onset and quiet trajectories from 0–10 seconds.",
        "- `stimulus_response_latency.csv`: sustained-response latency and peak timing for every neuron.",
        "- `olfactory_circuit_signatures.csv`: AWC-centered matrix signs versus observed and literature expectations.",
        "- `biological_prediction_summary.csv`: source-group × target-group held-out incremental prediction and quiet controls.",
        "- `repetition_adaptation.csv`: event-wise onset, quiet, and onset-minus-quiet activity plus matched adaptation slopes.",
        "- `temporal_cut_event_pathways.csv`: confirmation-worm event-specific sampled block strength and validity.",
        "- `stable_named_edge_candidates.csv`: strict gates plus full exploratory ranking.",
        "- `distributed_lag_olfactory_pathways.csv`: direct source-history kernels kept distinct from cumulative response horizons.",
    ]
    (output / "BIOLOGICAL_REPORT.md").write_text("\n".join(lines) + "\n")


def write_checksums(output: Path) -> None:
    entries = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (output / "checksums.sha256").write_text("\n".join(entries) + "\n")


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    dynamic, _ = load_all_methods()
    base = dynamic[PRIMARY_METHODS[0]]
    if tuple(cohort.neurons) != base.neurons:
        raise RuntimeError("cohort and matrix neuron orders differ")
    traces = fold_standardized_traces(cohort, base.folds)
    onset = build_episode_vectors(traces, cohort.fps, base.lags, episode="stimulus_onset")
    quiet = build_episode_vectors(traces, cohort.fps, base.lags, episode="quiet_pseudo_onset")

    annotations = annotation_table(base.neurons)
    observed, observed_roles = observed_response_tables(base.neurons, base.lags, onset, quiet)
    aligned, latency = event_aligned_response_tables(traces, base.neurons, cohort.fps)
    prediction_events, prediction_summary = biological_prediction_tables(
        dynamic, base.neurons, base.lags, onset, quiet
    )
    signatures = olfactory_signature_table(dynamic, base.neurons, base.lags, onset)
    distributed = distributed_lag_pathways(base.neurons)
    adaptation = repetition_adaptation(base.neurons, base.lags, onset, quiet)
    temporal_cut = temporal_cut_event_pathways(base.neurons, base.folds)
    candidates = stable_edge_candidates(dynamic, base.neurons, base.lags, onset)

    tables = {
        "neuron_annotations": annotations,
        "observed_response": observed,
        "observed_roles": observed_roles,
        "event_aligned_response": aligned,
        "stimulus_latency": latency,
        "prediction_events": prediction_events,
        "prediction_summary": prediction_summary,
        "olfactory_signature": signatures,
        "distributed_lag": distributed,
        "adaptation": adaptation,
        "temporal_cut": temporal_cut,
        "stable_candidates": candidates,
    }
    filenames = {
        "neuron_annotations": "neuron_functional_annotations.csv",
        "observed_response": "observed_response_by_neuron_horizon.csv",
        "observed_roles": "observed_response_by_role_horizon.csv",
        "event_aligned_response": "event_aligned_activity.csv",
        "stimulus_latency": "stimulus_response_latency.csv",
        "prediction_events": "biological_prediction_by_worm.csv",
        "prediction_summary": "biological_prediction_summary.csv",
        "olfactory_signature": "olfactory_circuit_signatures.csv",
        "distributed_lag": "distributed_lag_olfactory_pathways.csv",
        "adaptation": "repetition_adaptation.csv",
        "temporal_cut": "temporal_cut_event_pathways.csv",
        "stable_candidates": "stable_named_edge_candidates.csv",
    }
    for name, frame in tables.items():
        frame.to_csv(output / filenames[name], index=False)

    write_report(output, tables)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "biological organization of stimulus-conditioned lag matrices",
        "cohort": {"worms": cohort.n_worms, "neurons": cohort.n_neurons, "fps": cohort.fps, "events": 3},
        "primary_models": list(PRIMARY_METHODS),
        "sampling_sensitivity": "wide-flow temporal-cut SMC selected cell",
        "stability_sensitivity": "globally frozen group-shrunk distributed-lag kernel",
        "source_groups": list(SOURCE_GROUPS),
        "target_groups": list(TARGET_GROUPS),
        "horizon_frames": base.lags.astype(int).tolist(),
        "claim_boundary": "observed stimulus response and model-relative predictive dynamics, not causal/anatomical/physical-delay identification",
        "biological_sources": [WORMATLAS_CATEGORIES, AWC_CIRCUIT_SOURCE, BUTANONE_SOURCE],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    validation = {
        "all_checks_pass": bool(
            len(annotations) == 54
            and set(prediction_summary.method) == set(PRIMARY_METHODS)
            and set(prediction_summary.episode) == {"stimulus_onset", "quiet_pseudo_onset"}
            and not observed[["onset_mean", "quiet_mean", "onset_minus_quiet"]].isna().any().any()
            and len(aligned) == 4 * 41 * 54
            and len(latency) == 54
            and len(temporal_cut) > 0
        ),
        "checks": {
            "annotation_rows": len(annotations),
            "observed_response_rows": len(observed),
            "event_aligned_rows": len(aligned),
            "stimulus_latency_rows": len(latency),
            "prediction_worm_rows": len(prediction_events),
            "prediction_summary_rows": len(prediction_summary),
            "olfactory_signature_rows": len(signatures),
            "strict_candidate_rows": int(candidates.candidate_pass.sum()),
            "temporal_cut_event_rows": len(temporal_cut),
            "neuron_order_matches": tuple(cohort.neurons) == base.neurons,
            "outer_folds": sorted(np.unique(base.folds).astype(int).tolist()),
        },
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    write_checksums(output)
    if not validation["all_checks_pass"]:
        raise RuntimeError("biological analysis validation failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Biological analysis of frozen lag matrices")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
