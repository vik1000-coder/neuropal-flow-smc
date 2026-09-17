from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t
from sklearn.metrics import average_precision_score, roc_auc_score

from compatibility_neural_benchmark.core import (
    PHASES,
    STIMULUS_PERIODS_SECONDS,
    causal_fill,
)
from compatibility_neural_benchmark.postfreeze_external_analysis import load_references
from conditional_neural_benchmark.data import FoldScaler, load_cohort


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = Path("/Users/vik/Downloads/SBTG-public-release copy")
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "compatibility_path_response"
    / "onset_aware_propagation_20260827"
)

METHOD_LABELS = {
    "wide_flow_direct": "Wide-flow direct",
    "progressive_smc": "Progressive SMC",
    "old_flow_direct": "Original-flow direct",
    "terminal_smc": "Terminal SMC",
    "sbtg_current": "SBTG-current",
    "sbtg_published": "SBTG-published",
    "persistence": "Early-response persistence",
}

COLORS = {
    "wide_flow_direct": "#0EA5E9",
    "progressive_smc": "#059669",
    "old_flow_direct": "#3B82F6",
    "terminal_smc": "#F59E0B",
    "sbtg_current": "#A16207",
    "sbtg_published": "#7C3AED",
    "persistence": "#6B7280",
}


@dataclass(frozen=True)
class DynamicMatrices:
    matrices: np.ndarray  # worm, phase, lag, target, source
    validity: np.ndarray  # worm, phase, source
    folds: np.ndarray  # worm
    lags: np.ndarray
    phases: tuple[str, ...]
    neurons: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeVectors:
    source: np.ndarray  # worm, event, neuron
    target: np.ndarray  # worm, event, lag, neuron


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    usable = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[usable], float), np.asarray(y[usable], float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    if kind == "pearson":
        return float(pearsonr(x, y).statistic)
    if kind == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(kind)


def _pattern_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    usable = np.isfinite(prediction) & np.isfinite(target)
    prediction = np.asarray(prediction[usable], float)
    target = np.asarray(target[usable], float)
    if len(prediction) < 3:
        return {key: np.nan for key in ("pearson", "spearman", "cosine", "sign_agreement")}
    denom = float(np.linalg.norm(prediction) * np.linalg.norm(target))
    return {
        "pearson": _safe_corr(prediction, target, "pearson"),
        "spearman": _safe_corr(prediction, target, "spearman"),
        "cosine": float(prediction @ target / denom) if denom > 1e-12 else np.nan,
        "sign_agreement": float(np.mean(np.sign(prediction) == np.sign(target))),
    }


def _column_l2(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, float).copy()
    scale = np.linalg.norm(result, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    return result / scale[None]


def _prepare_matrix(matrix: np.ndarray, variant: str) -> np.ndarray:
    result = np.asarray(matrix, float).copy()
    np.fill_diagonal(result, 0.0)
    if variant == "raw":
        return result
    if variant == "column_l2":
        return _column_l2(result)
    raise ValueError(variant)


def load_dynamic_matrices(run_dir: Path, pattern: str) -> DynamicMatrices:
    paths = sorted((run_dir / "responses").glob(pattern))
    if len(paths) != 15:
        raise RuntimeError(f"expected 15 response archives under {run_dir}, found {len(paths)}")
    by_worm: dict[int, list[np.ndarray]] = {}
    by_validity: dict[int, list[np.ndarray]] = {}
    fold_by_worm: dict[int, int] = {}
    neurons = lags = phases = None
    seen: set[tuple[int, int]] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["status"].item()) != "complete":
                raise RuntimeError(f"incomplete response archive: {path}")
            fold, seed = int(data["fold"]), int(data["seed"])
            if (fold, seed) in seen:
                raise RuntimeError(f"duplicate fold/seed cell: {(fold, seed)}")
            seen.add((fold, seed))
            current_neurons = data["neurons"].astype(str)
            current_lags = data["horizon_frames"].astype(int)
            current_phases = data["phase_names"].astype(str)
            if neurons is None:
                neurons, lags, phases = current_neurons, current_lags, current_phases
            elif not (
                np.array_equal(neurons, current_neurons)
                and np.array_equal(lags, current_lags)
                and np.array_equal(phases, current_phases)
            ):
                raise RuntimeError("response archive alignment mismatch")
            response = data["response_cumulative_mean"].astype(np.float64)
            gap = data["diagnostic_achieved_gap"].astype(np.float64)
            normalized = response / np.maximum(gap, 0.10)[:, :, :, None, None]
            matrix = normalized.transpose(0, 1, 3, 4, 2)
            validity = data["diagnostic_valid"].astype(np.float64)
            for position, worm in enumerate(data["worm_indices"].astype(int)):
                by_worm.setdefault(worm, []).append(matrix[position])
                by_validity.setdefault(worm, []).append(validity[position])
                if worm in fold_by_worm and fold_by_worm[worm] != fold:
                    raise RuntimeError("one worm appears in multiple outer folds")
                fold_by_worm[worm] = fold
    if sorted(by_worm) != list(range(20)):
        raise RuntimeError("response archives do not cover worms 0..19")
    matrices = np.stack([np.mean(by_worm[worm], axis=0) for worm in range(20)])
    validity = np.stack([np.mean(by_validity[worm], axis=0) for worm in range(20)])
    folds = np.asarray([fold_by_worm[worm] for worm in range(20)], dtype=int)
    return DynamicMatrices(
        matrices=matrices,
        validity=validity,
        folds=folds,
        lags=np.asarray(lags, int),
        phases=tuple(np.asarray(phases).astype(str).tolist()),
        neurons=tuple(np.asarray(neurons).astype(str).tolist()),
    )


def load_all_methods() -> tuple[dict[str, DynamicMatrices], dict[str, dict[str, np.ndarray]]]:
    base = ROOT / "results" / "compatibility_path_response"
    dynamic = {
        "wide_flow_direct": load_dynamic_matrices(
            base / "winner_wide_direct_20260827", "flow_wide128_dropout10__B4__f*__s*.npz"
        ),
        "progressive_smc": load_dynamic_matrices(
            base / "progressive_full_ensemble_20260827",
            "tcn_delta_flow_matching__progressive_smc__B4__f*__s*.npz",
        ),
        "old_flow_direct": load_dynamic_matrices(
            base / "primary_20260826", "tcn_delta_flow_matching__B4__f*__s*.npz"
        ),
        "terminal_smc": load_dynamic_matrices(
            base / "smc_flow_N128_20260826",
            "tcn_delta_flow_matching__smc__B4__f*__s*.npz",
        ),
    }
    reference = next(iter(dynamic.values()))
    for name, item in dynamic.items():
        if item.neurons != reference.neurons or not np.array_equal(item.lags, reference.lags):
            raise RuntimeError(f"dynamic method alignment differs for {name}")
        if not np.array_equal(item.folds, reference.folds):
            raise RuntimeError(f"outer-fold assignment differs for {name}")
    aligned = base / "fair_atlas_analysis_20260826" / "aligned_all_lag_matrices.npz"
    static: dict[str, dict[str, np.ndarray]] = {}
    with np.load(aligned, allow_pickle=False) as data:
        if tuple(data["neurons"].astype(str)) != reference.neurons:
            raise RuntimeError("SBTG and dynamic neuron orders differ")
        for method in ("sbtg_current", "sbtg_published"):
            static[method] = {
                "lags": data[f"{method}__lags"].astype(int),
                "matrices": data[f"{method}__signed"].astype(np.float64),
            }
    return dynamic, static


def fold_standardized_traces(cohort, folds: np.ndarray) -> tuple[np.ndarray, ...]:
    traces: list[np.ndarray] = []
    for worm, fold in enumerate(folds):
        training = [cohort.traces[i] for i in range(cohort.n_worms) if folds[i] != fold]
        scaler = FoldScaler.fit(training)
        traces.append(causal_fill(scaler.transform(cohort.traces[worm])))
    return tuple(traces)


def build_episode_vectors(
    traces: tuple[np.ndarray, ...],
    fps: float,
    lags: np.ndarray,
    *,
    episode: str,
    source_frames: int = 4,
) -> EpisodeVectors:
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for trace in traces:
        worm_sources: list[np.ndarray] = []
        worm_targets: list[np.ndarray] = []
        for start_seconds, _ in STIMULUS_PERIODS_SECONDS:
            onset = int(round(start_seconds * fps))
            if episode == "stimulus_onset":
                cut = onset + source_frames - 1
            elif episode == "quiet_pseudo_onset":
                cut = onset - int(round(15.0 * fps))
            else:
                raise ValueError(episode)
            source_lo = cut - source_frames + 1
            baseline_lo = source_lo - source_frames
            baseline = trace[baseline_lo:source_lo].mean(axis=0)
            source = trace[source_lo : cut + 1].mean(axis=0) - baseline
            future = np.stack(
                [trace[cut + 1 : cut + 1 + lag].mean(axis=0) - baseline for lag in lags]
            )
            worm_sources.append(source)
            worm_targets.append(future)
        sources.append(np.stack(worm_sources))
        targets.append(np.stack(worm_targets))
    return EpisodeVectors(source=np.stack(sources), target=np.stack(targets))


def leave_one_worm_residual(array: np.ndarray, worms: np.ndarray) -> np.ndarray:
    result = np.full_like(array, np.nan, dtype=np.float64)
    for worm in worms:
        others = worms[worms != worm]
        result[worm] = array[worm] - np.mean(array[others], axis=0)
    return result


def evaluate_predictions(
    dynamic: dict[str, DynamicMatrices],
    static: dict[str, dict[str, np.ndarray]],
    vectors: dict[str, EpisodeVectors],
    strains: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows: list[dict] = []
    cached: dict[str, dict] = {}
    scopes = {
        "all20": np.arange(len(strains), dtype=int),
        "OH16230_17": np.flatnonzero(np.asarray(strains) == "OH16230"),
    }
    base = next(iter(dynamic.values()))
    for scope, worms in scopes.items():
        for episode, item in vectors.items():
            source_sets = {
                "shared_pattern": item.source,
                "worm_residual": leave_one_worm_residual(item.source, worms),
            }
            target_sets = {
                "shared_pattern": item.target,
                "worm_residual": leave_one_worm_residual(item.target, worms),
            }
            for target_mode in source_sets:
                source, target = source_sets[target_mode], target_sets[target_mode]
                for variant in ("raw", "column_l2"):
                    for method, artifact in dynamic.items():
                        for phase in ("onset", "baseline", "onset_minus_baseline"):
                            phase_index = artifact.phases.index("onset")
                            baseline_index = artifact.phases.index("baseline")
                            for lag_index, lag in enumerate(artifact.lags):
                                key = "|".join(
                                    map(str, (scope, episode, target_mode, variant, method, phase, lag))
                                )
                                cached[key] = {"predictions": [], "targets": [], "sources": []}
                                for worm in worms:
                                    raw_matrix = artifact.matrices[worm, phase_index, lag_index]
                                    if phase == "baseline":
                                        raw_matrix = artifact.matrices[worm, baseline_index, lag_index]
                                    elif phase == "onset_minus_baseline":
                                        raw_matrix = raw_matrix - artifact.matrices[
                                            worm, baseline_index, lag_index
                                        ]
                                    matrix = _prepare_matrix(raw_matrix, variant)
                                    for event in range(source.shape[1]):
                                        prediction = matrix @ source[worm, event]
                                        outcome = target[worm, event, lag_index]
                                        metric = _pattern_metrics(prediction, outcome)
                                        rows.append(
                                            {
                                                "scope": scope,
                                                "episode": episode,
                                                "target_mode": target_mode,
                                                "matrix_variant": variant,
                                                "method": method,
                                                "method_label": METHOD_LABELS[method],
                                                "matrix_phase": phase,
                                                "lag_frames": int(lag),
                                                "lag_seconds": float(lag / 4.0),
                                                "worm": int(worm),
                                                "event": int(event),
                                                **metric,
                                            }
                                        )
                                        cached[key]["predictions"].append(prediction)
                                        cached[key]["targets"].append(outcome)
                                        cached[key]["sources"].append(source[worm, event])
                    for method, artifact in static.items():
                        for lag_index, lag in enumerate(artifact["lags"]):
                            if lag not in base.lags:
                                continue
                            target_index = int(np.flatnonzero(base.lags == lag)[0])
                            matrix = _prepare_matrix(artifact["matrices"][lag_index], variant)
                            key = "|".join(
                                map(str, (scope, episode, target_mode, variant, method, "static", lag))
                            )
                            cached[key] = {"predictions": [], "targets": [], "sources": []}
                            for worm in worms:
                                for event in range(source.shape[1]):
                                    prediction = matrix @ source[worm, event]
                                    outcome = target[worm, event, target_index]
                                    rows.append(
                                        {
                                            "scope": scope,
                                            "episode": episode,
                                            "target_mode": target_mode,
                                            "matrix_variant": variant,
                                            "method": method,
                                            "method_label": METHOD_LABELS[method],
                                            "matrix_phase": "static",
                                            "lag_frames": int(lag),
                                            "lag_seconds": float(lag / 4.0),
                                            "worm": int(worm),
                                            "event": int(event),
                                            **_pattern_metrics(prediction, outcome),
                                        }
                                    )
                                    cached[key]["predictions"].append(prediction)
                                    cached[key]["targets"].append(outcome)
                                    cached[key]["sources"].append(source[worm, event])
                    for lag_index, lag in enumerate(base.lags):
                        key = "|".join(
                            map(
                                str,
                                (scope, episode, target_mode, variant, "persistence", "identity", lag),
                            )
                        )
                        cached[key] = {"predictions": [], "targets": [], "sources": []}
                        for worm in worms:
                            for event in range(source.shape[1]):
                                prediction = source[worm, event]
                                outcome = target[worm, event, lag_index]
                                rows.append(
                                    {
                                        "scope": scope,
                                        "episode": episode,
                                        "target_mode": target_mode,
                                        "matrix_variant": variant,
                                        "method": "persistence",
                                        "method_label": METHOD_LABELS["persistence"],
                                        "matrix_phase": "identity",
                                        "lag_frames": int(lag),
                                        "lag_seconds": float(lag / 4.0),
                                        "worm": int(worm),
                                        "event": int(event),
                                        **_pattern_metrics(prediction, outcome),
                                    }
                                )
                                cached[key]["predictions"].append(prediction)
                                cached[key]["targets"].append(outcome)
                                cached[key]["sources"].append(source[worm, event])
    return pd.DataFrame(rows), cached


def summarize_predictions(events: pd.DataFrame) -> pd.DataFrame:
    group = [
        "scope",
        "episode",
        "target_mode",
        "matrix_variant",
        "method",
        "method_label",
        "matrix_phase",
        "lag_frames",
        "lag_seconds",
    ]
    worm = events.groupby(group + ["worm"], as_index=False)[
        ["pearson", "spearman", "cosine", "sign_agreement"]
    ].mean()
    rows: list[dict] = []
    for keys, frame in worm.groupby(group, sort=False):
        row = dict(zip(group, keys))
        for metric in ("pearson", "spearman", "cosine", "sign_agreement"):
            values = frame[metric].dropna().to_numpy(float)
            row[f"mean_{metric}"] = float(np.mean(values)) if len(values) else np.nan
            row[f"median_{metric}"] = float(np.median(values)) if len(values) else np.nan
            row[f"n_worms_{metric}"] = int(len(values))
            if len(values) > 1:
                half = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
                row[f"{metric}_ci_low"] = row[f"mean_{metric}"] - half
                row[f"{metric}_ci_high"] = row[f"mean_{metric}"] + half
            else:
                row[f"{metric}_ci_low"] = row[f"{metric}_ci_high"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_incremental_persistence(
    dynamic: dict[str, DynamicMatrices],
    static: dict[str, dict[str, np.ndarray]],
    onset: EpisodeVectors,
) -> pd.DataFrame:
    """Cross-fit two scalar forecast weights: persistence plus one saved matrix.

    The lagged operator remains frozen. Only an intercept and the two global
    coefficients on ``x`` and ``M x`` are fit using the other 19 worms.
    """
    rows: list[dict] = []
    all_worms = np.arange(onset.source.shape[0], dtype=int)
    base = next(iter(dynamic.values()))
    for method in list(dynamic) + list(static):
        lags = dynamic[method].lags if method in dynamic else static[method]["lags"]
        for lag_position, lag in enumerate(lags):
            if lag not in base.lags:
                continue
            target_index = int(np.flatnonzero(base.lags == lag)[0])
            for target_mode in ("shared_pattern", "worm_residual"):
                for heldout in all_worms:
                    training = all_worms[all_worms != heldout]
                    if target_mode == "shared_pattern":
                        source_train = onset.source[training]
                        target_train = onset.target[training, :, target_index]
                        source_test = onset.source[heldout]
                        target_test = onset.target[heldout, :, target_index]
                    else:
                        train_mean_source = onset.source[training].mean(axis=0)
                        train_mean_target = onset.target[training, :, target_index].mean(axis=0)
                        source_test = onset.source[heldout] - train_mean_source
                        target_test = onset.target[heldout, :, target_index] - train_mean_target
                        source_train = np.stack(
                            [
                                onset.source[worm]
                                - onset.source[training[training != worm]].mean(axis=0)
                                for worm in training
                            ]
                        )
                        target_train = np.stack(
                            [
                                onset.target[worm, :, target_index]
                                - onset.target[training[training != worm], :, target_index].mean(axis=0)
                                for worm in training
                            ]
                        )
                    matrix_train: list[np.ndarray] = []
                    if method in dynamic:
                        artifact = dynamic[method]
                        onset_index = artifact.phases.index("onset")
                        for worm in training:
                            matrix_train.append(
                                _prepare_matrix(
                                    artifact.matrices[worm, onset_index, lag_position], "raw"
                                )
                            )
                        matrix_test = _prepare_matrix(
                            artifact.matrices[heldout, onset_index, lag_position], "raw"
                        )
                    else:
                        matrix_test = _prepare_matrix(static[method]["matrices"][lag_position], "raw")
                        matrix_train = [matrix_test for _ in training]
                    propagated_train = np.stack(
                        [
                            np.stack([matrix @ source_train[i, event] for event in range(3)])
                            for i, matrix in enumerate(matrix_train)
                        ]
                    )
                    design = np.column_stack(
                        [
                            np.ones(source_train.size),
                            source_train.reshape(-1),
                            propagated_train.reshape(-1),
                        ]
                    )
                    outcome = target_train.reshape(-1)
                    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
                    for event in range(3):
                        propagated = matrix_test @ source_test[event]
                        prediction = (
                            coefficients[0]
                            + coefficients[1] * source_test[event]
                            + coefficients[2] * propagated
                        )
                        persistence = source_test[event]
                        matrix_metric = _pattern_metrics(prediction, target_test[event])
                        persistence_metric = _pattern_metrics(persistence, target_test[event])
                        rows.append(
                            {
                                "method": method,
                                "method_label": METHOD_LABELS[method],
                                "target_mode": target_mode,
                                "lag_frames": int(lag),
                                "lag_seconds": float(lag / 4.0),
                                "worm": int(heldout),
                                "event": int(event),
                                "intercept": float(coefficients[0]),
                                "persistence_coefficient": float(coefficients[1]),
                                "matrix_coefficient": float(coefficients[2]),
                                **{f"combined_{key}": value for key, value in matrix_metric.items()},
                                **{f"persistence_{key}": value for key, value in persistence_metric.items()},
                                "spearman_gain": matrix_metric["spearman"]
                                - persistence_metric["spearman"],
                            }
                        )
    return pd.DataFrame(rows)


def summarize_incremental(events: pd.DataFrame) -> pd.DataFrame:
    group = ["method", "method_label", "target_mode", "lag_frames", "lag_seconds"]
    worm = events.groupby(group + ["worm"], as_index=False).agg(
        combined_spearman=("combined_spearman", "mean"),
        persistence_spearman=("persistence_spearman", "mean"),
        spearman_gain=("spearman_gain", "mean"),
        persistence_coefficient=("persistence_coefficient", "mean"),
        matrix_coefficient=("matrix_coefficient", "mean"),
    )
    rows: list[dict] = []
    for keys, frame in worm.groupby(group, sort=False):
        row = dict(zip(group, keys))
        for metric in (
            "combined_spearman",
            "persistence_spearman",
            "spearman_gain",
            "persistence_coefficient",
            "matrix_coefficient",
        ):
            values = frame[metric].to_numpy(float)
            mean = float(np.mean(values))
            half = float(
                t.ppf(0.975, len(values) - 1)
                * np.std(values, ddof=1)
                / np.sqrt(len(values))
            )
            row[f"mean_{metric}"] = mean
            row[f"{metric}_ci_low"] = mean - half
            row[f"{metric}_ci_high"] = mean + half
        gain = frame.spearman_gain.to_numpy(float)
        statistic = np.mean(gain) / (np.std(gain, ddof=1) / np.sqrt(len(gain)))
        row["gain_p_value"] = float(2 * t.sf(abs(statistic), len(gain) - 1))
        rows.append(row)
    result = pd.DataFrame(rows)
    order = np.argsort(result.gain_p_value.to_numpy())
    ranked = result.gain_p_value.to_numpy()[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result["gain_bh_q_value"] = 1.0
    result.loc[result.index[order], "gain_bh_q_value"] = np.minimum(adjusted, 1.0)
    return result


def paired_contrasts(events: pd.DataFrame) -> pd.DataFrame:
    primary = events[
        (events.scope == "all20")
        & (events.target_mode == "worm_residual")
        & (events.matrix_variant == "raw")
    ]
    worm = primary.groupby(
        ["episode", "method", "matrix_phase", "lag_frames", "worm"], as_index=False
    ).spearman.mean()
    lookups = {
        key: frame.set_index("worm").spearman
        for key, frame in worm.groupby(["episode", "method", "matrix_phase", "lag_frames"])
    }
    specs: list[tuple[str, tuple, tuple]] = []
    methods = [name for name in METHOD_LABELS if name != "persistence"]
    lags = sorted(primary.lag_frames.unique())
    for method in methods:
        for lag in lags:
            phase = "static" if method.startswith("sbtg") else "onset"
            quiet_phase = "static" if method.startswith("sbtg") else "baseline"
            specs.append(
                (
                    "stimulus_minus_quiet",
                    ("stimulus_onset", method, phase, lag),
                    ("quiet_pseudo_onset", method, quiet_phase, lag),
                )
            )
            specs.append(
                (
                    "method_minus_persistence",
                    ("stimulus_onset", method, phase, lag),
                    ("stimulus_onset", "persistence", "identity", lag),
                )
            )
            if not method.startswith("sbtg"):
                specs.append(
                    (
                        "onset_phase_minus_baseline_phase",
                        ("stimulus_onset", method, "onset", lag),
                        ("stimulus_onset", method, "baseline", lag),
                    )
                )
    rows: list[dict] = []
    for contrast, left, right in specs:
        if left not in lookups or right not in lookups:
            continue
        pair = pd.concat([lookups[left].rename("left"), lookups[right].rename("right")], axis=1).dropna()
        difference = (pair.left - pair.right).to_numpy(float)
        if not len(difference):
            continue
        mean = float(np.mean(difference))
        if len(difference) > 1:
            sem = float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
            half = float(t.ppf(0.975, len(difference) - 1) * sem)
            statistic = mean / sem if sem > 0 else np.inf
            p_value = float(2 * t.sf(abs(statistic), len(difference) - 1))
        else:
            half = p_value = np.nan
        rows.append(
            {
                "contrast": contrast,
                "method": left[1],
                "method_label": METHOD_LABELS[left[1]],
                "lag_frames": int(left[3]),
                "lag_seconds": float(left[3] / 4.0),
                "left_episode": left[0],
                "left_phase": left[2],
                "right_episode": right[0],
                "right_method": right[1],
                "right_phase": right[2],
                "n_worms": int(len(difference)),
                "mean_spearman_difference": mean,
                "ci_low": mean - half,
                "ci_high": mean + half,
                "paired_t_p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        order = np.argsort(result.paired_t_p_value.fillna(1.0).to_numpy())
        ranked = result.paired_t_p_value.fillna(1.0).to_numpy()[order]
        adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
        result["bh_q_value"] = 1.0
        result.loc[result.index[order], "bh_q_value"] = np.minimum(adjusted, 1.0)
    return result


def source_permutation_tests(
    dynamic: dict[str, DynamicMatrices],
    static: dict[str, dict[str, np.ndarray]],
    vectors: dict[str, EpisodeVectors],
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    worms = np.arange(20, dtype=int)
    base = next(iter(dynamic.values()))
    rows: list[dict] = []
    for episode, vector in vectors.items():
        source = leave_one_worm_residual(vector.source, worms)
        target = leave_one_worm_residual(vector.target, worms)
        for method, artifact in dynamic.items():
            phase = "onset" if episode == "stimulus_onset" else "baseline"
            phase_index = artifact.phases.index(phase)
            for lag_index, lag in enumerate(artifact.lags):
                matrices = np.stack(
                    [_prepare_matrix(artifact.matrices[w, phase_index, lag_index], "raw") for w in worms]
                )
                observed = np.mean(
                    [
                        _safe_corr(matrices[w] @ source[w, event], target[w, event, lag_index], "spearman")
                        for w in worms
                        for event in range(3)
                    ]
                )
                null = []
                for _ in range(n_permutations):
                    permutation = rng.permutation(source.shape[-1])
                    null.append(
                        np.mean(
                            [
                                _safe_corr(
                                    matrices[w] @ source[w, event, permutation],
                                    target[w, event, lag_index],
                                    "spearman",
                                )
                                for w in worms
                                for event in range(3)
                            ]
                        )
                    )
                rows.append(
                    {
                        "episode": episode,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "matrix_phase": phase,
                        "lag_frames": int(lag),
                        "lag_seconds": float(lag / 4.0),
                        "observed_mean_event_spearman": float(observed),
                        "null_mean": float(np.mean(null)),
                        "null_sd": float(np.std(null, ddof=1)),
                        "null_ci_low": float(np.quantile(null, 0.025)),
                        "null_ci_high": float(np.quantile(null, 0.975)),
                        "one_sided_p_value": float((1 + np.sum(np.asarray(null) >= observed)) / (n_permutations + 1)),
                        "n_permutations": int(n_permutations),
                    }
                )
        for method, artifact in static.items():
            for static_index, lag in enumerate(artifact["lags"]):
                if lag not in base.lags:
                    continue
                lag_index = int(np.flatnonzero(base.lags == lag)[0])
                matrix = _prepare_matrix(artifact["matrices"][static_index], "raw")
                observed = np.mean(
                    [
                        _safe_corr(matrix @ source[w, event], target[w, event, lag_index], "spearman")
                        for w in worms
                        for event in range(3)
                    ]
                )
                null = []
                for _ in range(n_permutations):
                    permutation = rng.permutation(source.shape[-1])
                    null.append(
                        np.mean(
                            [
                                _safe_corr(
                                    matrix @ source[w, event, permutation],
                                    target[w, event, lag_index],
                                    "spearman",
                                )
                                for w in worms
                                for event in range(3)
                            ]
                        )
                    )
                rows.append(
                    {
                        "episode": episode,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "matrix_phase": "static",
                        "lag_frames": int(lag),
                        "lag_seconds": float(lag / 4.0),
                        "observed_mean_event_spearman": float(observed),
                        "null_mean": float(np.mean(null)),
                        "null_sd": float(np.std(null, ddof=1)),
                        "null_ci_low": float(np.quantile(null, 0.025)),
                        "null_ci_high": float(np.quantile(null, 0.975)),
                        "one_sided_p_value": float((1 + np.sum(np.asarray(null) >= observed)) / (n_permutations + 1)),
                        "n_permutations": int(n_permutations),
                    }
                )
    return pd.DataFrame(rows)


def _binary_metrics(matrix: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    usable = mask & np.isfinite(matrix)
    y = labels[usable].astype(int)
    score = np.abs(matrix[usable])
    if len(y) and 0 < y.sum() < len(y):
        return {
            "n_pairs": int(len(y)),
            "n_positive": int(y.sum()),
            "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
        }
    return {"n_pairs": int(len(y)), "n_positive": int(y.sum()), "auroc": np.nan, "auprc": np.nan}


def phase_external_metrics(
    dynamic: dict[str, DynamicMatrices],
    static: dict[str, dict[str, np.ndarray]],
    release: Path,
) -> pd.DataFrame:
    neurons = list(next(iter(dynamic.values())).neurons)
    references, networks = load_references(release, neurons)
    rows: list[dict] = []
    off = ~np.eye(len(neurons), dtype=bool)
    for method, artifact in dynamic.items():
        phase_matrices = {
            phase: artifact.matrices[:, phase_index].mean(axis=0)
            for phase_index, phase in enumerate(artifact.phases)
        }
        phase_matrices["onset_minus_baseline"] = (
            phase_matrices["onset"] - phase_matrices["baseline"]
        )
        for phase, mean_matrix in phase_matrices.items():
            for lag_index, lag in enumerate(artifact.lags):
                matrix = _prepare_matrix(mean_matrix[lag_index], "raw")
                for reference in ("randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54"):
                    ref = references[reference]
                    rows.append(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "matrix_phase": phase,
                            "lag_frames": int(lag),
                            "lag_seconds": float(lag / 4.0),
                            "reference_family": "external",
                            "reference": reference,
                            "scope": "declared",
                            **_binary_metrics(matrix, ref["labels"], ref["mask"]),
                        }
                    )
                for network in ("monoamine_all", "neuropeptide_all", "neuromodulator_union"):
                    labels = networks[network]
                    eligible = labels.any(axis=0)
                    for scope, mask in (
                        ("all_pairs_legacy", off),
                        ("eligible_sources", off & eligible[None]),
                    ):
                        rows.append(
                            {
                                "method": method,
                                "method_label": METHOD_LABELS[method],
                                "matrix_phase": phase,
                                "lag_frames": int(lag),
                                "lag_seconds": float(lag / 4.0),
                                "reference_family": "bentley",
                                "reference": network,
                                "scope": scope,
                                **_binary_metrics(matrix, labels, mask),
                            }
                        )
    for method, artifact in static.items():
        for lag, raw_matrix in zip(artifact["lags"], artifact["matrices"]):
            matrix = _prepare_matrix(raw_matrix, "raw")
            for reference in ("randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54"):
                ref = references[reference]
                rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "matrix_phase": "static",
                        "lag_frames": int(lag),
                        "lag_seconds": float(lag / 4.0),
                        "reference_family": "external",
                        "reference": reference,
                        "scope": "declared",
                        **_binary_metrics(matrix, ref["labels"], ref["mask"]),
                    }
                )
            for network in ("monoamine_all", "neuropeptide_all", "neuromodulator_union"):
                labels = networks[network]
                eligible = labels.any(axis=0)
                for scope, mask in (
                    ("all_pairs_legacy", off),
                    ("eligible_sources", off & eligible[None]),
                ):
                    rows.append(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "matrix_phase": "static",
                            "lag_frames": int(lag),
                            "lag_seconds": float(lag / 4.0),
                            "reference_family": "bentley",
                            "reference": network,
                            "scope": scope,
                            **_binary_metrics(matrix, labels, mask),
                        }
                    )
    return pd.DataFrame(rows)


def create_figures(
    output: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    incremental: pd.DataFrame,
    phase_external: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary = summary[
        (summary.scope == "all20")
        & (summary.episode == "stimulus_onset")
        & (summary.target_mode == "worm_residual")
        & (summary.matrix_variant == "raw")
        & (
            ((summary.method.isin(["wide_flow_direct", "progressive_smc", "old_flow_direct", "terminal_smc"])) & (summary.matrix_phase == "onset"))
            | ((summary.method.isin(["sbtg_current", "sbtg_published"])) & (summary.matrix_phase == "static"))
            | (summary.method == "persistence")
        )
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for method, frame in primary.groupby("method"):
        frame = frame.sort_values("lag_seconds")
        ax.plot(
            frame.lag_seconds,
            frame.mean_spearman,
            marker="o",
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
        ax.fill_between(
            frame.lag_seconds,
            frame.spearman_ci_low,
            frame.spearman_ci_high,
            color=COLORS[method],
            alpha=0.10,
        )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set(xlabel="Forecast horizon after 1-second onset source window (s)", ylabel="Mean event-wise Spearman across target neurons", title="Held-out onset propagation of worm-specific residual patterns")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "onset_residual_prediction_by_lag.png", dpi=200)
    plt.close(fig)

    phase = contrasts[contrasts.contrast == "onset_phase_minus_baseline_phase"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for method, frame in phase.groupby("method"):
        frame = frame.sort_values("lag_seconds")
        ax.plot(frame.lag_seconds, frame.mean_spearman_difference, marker="o", label=METHOD_LABELS[method], color=COLORS[method])
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set(xlabel="Forecast horizon (s)", ylabel="Onset-phase minus baseline-phase Spearman", title="Does stimulus conditioning improve onset propagation?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "onset_phase_gain_by_lag.png", dpi=200)
    plt.close(fig)

    incremental_residual = incremental[incremental.target_mode == "worm_residual"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for method, frame in incremental_residual.groupby("method"):
        frame = frame.sort_values("lag_seconds")
        ax.plot(
            frame.lag_seconds,
            frame.mean_spearman_gain,
            marker="o",
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set(
        xlabel="Forecast horizon (s)",
        ylabel="Cross-fit Spearman gain over persistence",
        title="Incremental value of the frozen inter-neuron matrix",
    )
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "incremental_gain_over_persistence.png", dpi=200)
    plt.close(fig)

    bentley = phase_external[
        (phase_external.reference_family == "bentley")
        & (phase_external.scope == "eligible_sources")
        & (phase_external.reference.isin(["monoamine_all", "neuropeptide_all"]))
        & (phase_external.method.isin(["wide_flow_direct", "progressive_smc"]))
        & (phase_external.matrix_phase.isin(["baseline", "onset", "active", "offset", "recovery"]))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, network in zip(axes, ("monoamine_all", "neuropeptide_all")):
        frame = bentley[bentley.reference == network]
        for (method, phase_name), group in frame.groupby(["method", "matrix_phase"]):
            style = "-" if method == "wide_flow_direct" else "--"
            ax.plot(group.lag_seconds, group.auroc, linestyle=style, alpha=0.75, label=f"{METHOD_LABELS[method]} / {phase_name}")
        ax.axhline(0.5, color="#111827", linewidth=0.8)
        ax.set(title=network.replace("_all", "").title(), xlabel="Lag/horizon (s)")
    axes[0].set_ylabel("Bentley eligible-source AUROC")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(figures / "bentley_phase_lag_exploration.png", dpi=200)
    plt.close(fig)


def _best_rows(summary: pd.DataFrame, target_mode: str) -> pd.DataFrame:
    frame = summary[
        (summary.scope == "all20")
        & (summary.episode == "stimulus_onset")
        & (summary.target_mode == target_mode)
        & (summary.matrix_variant == "raw")
        & (
            ((summary.method.isin(["wide_flow_direct", "progressive_smc", "old_flow_direct", "terminal_smc"])) & (summary.matrix_phase == "onset"))
            | ((summary.method.isin(["sbtg_current", "sbtg_published"])) & (summary.matrix_phase == "static"))
            | (summary.method == "persistence")
        )
    ]
    return frame.loc[frame.groupby("method").mean_spearman.idxmax()].sort_values("mean_spearman", ascending=False)


def write_report(
    output: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    incremental: pd.DataFrame,
    permutations: pd.DataFrame,
    phase_external: pd.DataFrame,
    validation: dict,
) -> None:
    residual = _best_rows(summary, "worm_residual")
    shared = _best_rows(summary, "shared_pattern")
    residual_table = ["| Method | Best horizon | Mean residual Spearman | 95% worm CI |", "|---|---:|---:|---:|"]
    for row in residual.itertuples():
        residual_table.append(f"| {row.method_label} | {row.lag_seconds:g} s | {row.mean_spearman:.3f} | [{row.spearman_ci_low:.3f}, {row.spearman_ci_high:.3f}] |")
    shared_table = ["| Method | Best horizon | Mean shared-pattern Spearman |", "|---|---:|---:|"]
    for row in shared.itertuples():
        shared_table.append(f"| {row.method_label} | {row.lag_seconds:g} s | {row.mean_spearman:.3f} |")
    onset_bentley = phase_external[
        (phase_external.reference_family == "bentley")
        & (phase_external.scope == "eligible_sources")
        & (phase_external.matrix_phase == "onset")
        & (phase_external.reference.isin(["monoamine_all", "neuropeptide_all"]))
    ]
    best_bentley = onset_bentley.loc[onset_bentley.groupby(["method", "reference"]).auroc.idxmax()]
    bentley_table = ["| Method | Network | Descriptive best onset lag | AUROC |", "|---|---|---:|---:|"]
    for row in best_bentley.sort_values(["reference", "auroc"], ascending=[True, False]).itertuples():
        bentley_table.append(f"| {row.method_label} | {row.reference} | {row.lag_seconds:g} s | {row.auroc:.3f} |")
    significant_perm = permutations[
        (permutations.episode == "stimulus_onset") & (permutations.one_sided_p_value < 0.05)
    ]
    best_residual = residual.iloc[0]
    incremental_residual = incremental[incremental.target_mode == "worm_residual"]
    best_incremental = incremental_residual.loc[
        incremental_residual.groupby("method").mean_spearman_gain.idxmax()
    ].sort_values("mean_spearman_gain", ascending=False)
    incremental_table = [
        "| Matrix | Best horizon | Cross-fit gain over persistence | 95% worm CI |",
        "|---|---:|---:|---:|",
    ]
    for row in best_incremental.itertuples():
        incremental_table.append(
            f"| {row.method_label} | {row.lag_seconds:g} s | {row.mean_spearman_gain:+.3f} | "
            f"[{row.spearman_gain_ci_low:+.3f}, {row.spearman_gain_ci_high:+.3f}] |"
        )
    report = "\n".join(
        [
            "# Onset-aware lagged propagation and neuromodulator correspondence",
            "",
            "**Run date:** 2026-08-27  ",
            "**Claim boundary:** stimulus-locked prediction under an observational learned law; not anatomy, direct synapses, causal intervention effects, rewiring, or physical transmission delays.",
            "",
            "## Technical summary",
            "",
            f"The known stimulus times expose a more direct test than Bentley edge matching: whether an early four-frame neural response is transformed by a saved lagged matrix into the later 54-neuron response pattern in a held-out worm. The strongest worm-residual result is **{best_residual.method_label}** at **{best_residual.lag_seconds:g} s**, with mean event-wise Spearman **{best_residual.mean_spearman:.3f}** (worm-clustered 95% t interval **[{best_residual.spearman_ci_low:.3f}, {best_residual.spearman_ci_high:.3f}]**). This is predictive propagation evidence only.",
            "",
            "The onset-conditioned test and the neuromodulator-atlas test answer different questions. Onset prediction evaluates the observed trajectory around a known boundary. Bentley AUROC asks whether absolute matrix weights rank a sparse literature-derived molecular edge list. Near-chance Bentley correspondence therefore does not by itself falsify onset-locked predictive structure, while onset prediction cannot validate molecular identity or a physical delay.",
            "",
            "## Author-provided context and scope correction",
            "",
            "The project author noted on 2026-08-27 that the SBTG paper's connectome/neuromodulator matching was minimal and did not use known stimulus onset. This is recorded as author-provided context, not inferred from the current numeric results. Accordingly, SBTG-published is retained as a historical matrix comparator, but its Bentley match is treated as weak indirect validation rather than a decisive mechanistic benchmark.",
            "",
            "## Key findings with visual evidence",
            "",
            "![Held-out onset residual prediction](figures/onset_residual_prediction_by_lag.png)",
            "",
            *residual_table,
            "",
            "The worm-residual target subtracts the leave-one-worm-out event-average response separately at each stimulus repetition. It therefore removes the common stimulus pulse and asks whether inter-worm deviations in the early source pattern predict deviations in the later target pattern.",
            "",
            "### Does a matrix improve the strong persistence forecast?",
            "",
            "![Incremental gain over persistence](figures/incremental_gain_over_persistence.png)",
            "",
            *incremental_table,
            "",
            "For this test, each held-out worm is predicted by `a × early_response + b × matrix_response + intercept`. Only the three scalar coefficients are fitted on the other 19 worms; the inter-neuron matrix remains frozen. A positive gain is the clearest practical evidence that the matrix adds information beyond same-neuron persistence. The complete coefficients, intervals, p-values, and multiplicity-adjusted q-values are in `incremental_prediction_summary.csv`.",
            "",
            "![Onset phase gain](figures/onset_phase_gain_by_lag.png)",
            "",
            "Positive values mean the onset-conditioned matrix predicts the same onset data better than that model's baseline-conditioned matrix. All method/horizon contrasts, including multiplicity-adjusted q-values, are in `paired_contrasts.csv`.",
            "",
            "### Shared stimulus-evoked pattern",
            "",
            *shared_table,
            "",
            "This easier target retains the population-average stimulus pulse. It is useful for stimulus-response forecasting, but it is more vulnerable to common-input explanations and cannot be presented as inter-neuron transmission.",
            "",
            "### Source-label permutation control",
            "",
            f"A global neuron-label permutation was applied to the early-response source vector while holding each matrix and later target fixed. **{len(significant_perm)}** onset method-by-horizon cells had unadjusted one-sided permutation p < 0.05 out of **{int((permutations.episode == 'stimulus_onset').sum())}** tested onset cells. Exact null intervals and p-values are in `source_permutation_tests.csv`; these are exploratory and not corrected for the horizon search.",
            "",
            "## Neuromodulator and external-reference exploration",
            "",
            "![Bentley phase and lag exploration](figures/bentley_phase_lag_exploration.png)",
            "",
            *bentley_table,
            "",
            "These are descriptive maxima selected across lags. The monoamine list has very few eligible source classes, the matrices are observational, and the same data contribute across adjacent horizons. No peak is interpretable as receptor kinetics or a physical propagation delay. Randi and Cook phase-resolved values are saved beside Bentley in `phase_external_metrics.csv` so binary presence AUROC and count-valued anatomy are not conflated.",
            "",
            "## Definitions and scope",
            "",
            "- Sampling is 4 Hz; one frame is 0.25 seconds. Matrix orientation is target row, source column.",
            "- Each stimulus source vector is the mean standardized activity during frames onset through onset+3 minus the immediately preceding four-frame baseline.",
            "- The target at horizon h is the cumulative mean of the next h frames after that one-second source window, relative to the same baseline.",
            "- Quiet pseudo-onsets use each archive's baseline cut 15 seconds before the true onset, with the same four-frame source and future-horizon construction.",
            "- `shared_pattern` keeps the stimulus-evoked population mean. `worm_residual` subtracts the other worms' event-specific mean from both source and target.",
            "- Dynamic matrices are outer-fold held-out for the evaluated worm and averaged over three generator seeds. SBTG-current and SBTG-published are static contextual matrices and are evaluated only at horizons shared with the dynamic grid.",
            "",
            "## Methodology and model specification",
            "",
            "For each worm, the neural trace is standardized using only worms outside its saved outer fold. For a saved matrix M_h and early response x, the linearized prediction is M_h x. Diagonals are zeroed. The primary score is Spearman correlation across the 54 target neurons for each worm-event-horizon, then averaged within worm and across worms. Pearson, cosine, sign agreement, column-L2-normalized sensitivity, the 17-worm OH16230 subset, baseline-phase matrices, quiet pseudo-onsets, persistence, and source permutations are all retained in the machine-readable outputs.",
            "",
            "The wide-flow direct, original-flow direct, terminal-SMC, and progressive-SMC matrices are not regression coefficients fitted to these onset outcomes. They are finite-particle repaired-response contrasts of learned conditional transition laws. SBTG-current is an expected score-product estimator related under regularity conditions to a negative expected mixed log-density Hessian; SBTG-published is the released tuned artifact with different training lineage.",
            "",
            "## Related onset-detection evidence",
            "",
            "A separate diffusionCircuit benchmark (`20260707_194459_score_dynamics_probe_neural`) used known butanone onsets and quiet pseudo-boundaries. On 20 imputed worms, detected-within-5-second rates were 0.35 for a pooled history-matched DSM reliability/delay-law scan, 0.30 for a Hotelling mean CUSUM, 0.15 for a score-dynamics variance readout, and 0.00 for the residualized SBTG score-product CUSUM; each method's quiet-control false-positive rate was calibrated to 0.05. Those results show that onset access can reveal stimulus-locked mean or reliability changes that a generic SBTG rule-change scan misses. They remain detection results, not edge or anatomy validation.",
            "",
            "## Limitations and robustness",
            "",
            "- There are only 20 complete-case worms and three repeated onsets per worm. Event repetitions within a worm are not independent; worm-clustered intervals are primary.",
            "- The 17 OH16230 worms dominate the cohort; the OH16230-only sensitivity is saved, while the three OH15500 worms are too few for a stable separate estimate.",
            "- Common stimulus input, unmeasured state, calcium filtering, and behavioral feedback can generate lagged predictive structure without direct neuron-to-neuron effects.",
            "- The linear application M_h x is a local/contrast-based approximation. Correlation is scale-free and does not validate the absolute calibration of predicted responses.",
            "- Selecting the best lag from the same outcomes is exploratory. The per-lag rows and permutation nulls must be shown; a future confirmatory cohort should freeze one horizon and statistic.",
            "- Bentley edges are binary literature-derived molecular possibilities, whereas Cook is count-valued anatomy and Randi is perturbational functional evidence. AUROC is appropriate for binary presence; Spearman/count metrics are more appropriate for continuous Cook strength.",
            "",
            "## Next steps",
            "",
            "1. Freeze the best onset statistic and horizon from this exploratory run, then evaluate it on an untouched worm cohort or a new acquisition.",
            "2. Fit an onset-supervised cross-worm ridge/operator only as a learnability ceiling, with nested worm-level cross-validation, and compare it to the unsupervised matrices without mixing selection data.",
            "3. Split the response into direct stimulus-recipient neurons and downstream neurons, using prior sensory identity declared before scoring, to test whether performance survives removal of likely direct stimulus channels.",
            "4. Replace descriptive best-lag Bentley matching with a hierarchical lag profile and uncertainty over generator seed, worm, and molecular-source class.",
            "5. If physical delay is the scientific target, obtain perturbations or repeated high-temporal-resolution measurements; passive calcium and literature-edge matching are insufficient for that claim.",
            "",
            "## Further questions",
            "",
            "- Does the onset-conditioned matrix improve specifically after removing direct sensory neurons from the target set?",
            "- Are the same source columns predictive across all three stimulus repetitions, or does adaptation dominate later events?",
            "- Is any lag peak stable across generator seeds rather than appearing only after ensembling?",
            "- Does a model selected for onset residual prediction retain its Randi correspondence on a separately held-out reference analysis?",
            "",
            "## Output inventory",
            "",
            "- `event_level_prediction_metrics.csv`: every worm/event/method/phase/horizon score.",
            "- `prediction_summary.csv`: worm-clustered summaries and intervals.",
            "- `paired_contrasts.csv`: onset-vs-quiet, onset-phase-vs-baseline-phase, and method-vs-persistence differences.",
            "- `source_permutation_tests.csv`: neuron-label nulls for the primary worm-residual analysis.",
            "- `incremental_prediction_metrics.csv` and `incremental_prediction_summary.csv`: held-out scalar combinations of persistence and each frozen matrix.",
            "- `phase_external_metrics.csv`: phase-resolved Randi, Cook, and Bentley metrics.",
            "- `aligned_phase_matrices.npz`: exact matrices, validities, phases, lags, folds, and neuron order used here.",
            "- `protocol.json`, `validation.json`, and `checksums.sha256`: frozen settings, integrity checks, and file hashes.",
            "",
            f"Validation status: **{'PASS' if validation['all_checks_pass'] else 'FAIL'}**.",
            "",
        ]
    )
    (output / "REPORT.md").write_text(report)


def write_checksums(output: Path) -> None:
    targets = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}" for path in targets]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n")


def run(output: Path, release: Path, n_permutations: int, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dynamic, static = load_all_methods()
    base = next(iter(dynamic.values()))
    cohort = load_cohort(coverage=0.90, complete_case=True)
    if tuple(cohort.neurons) != base.neurons:
        raise RuntimeError("cohort and response neuron orders differ")
    traces = fold_standardized_traces(cohort, base.folds)
    vectors = {
        episode: build_episode_vectors(traces, cohort.fps, base.lags, episode=episode)
        for episode in ("stimulus_onset", "quiet_pseudo_onset")
    }
    events, _ = evaluate_predictions(dynamic, static, vectors, cohort.strains)
    summary = summarize_predictions(events)
    contrasts = paired_contrasts(events)
    incremental_events = evaluate_incremental_persistence(
        dynamic, static, vectors["stimulus_onset"]
    )
    incremental_summary = summarize_incremental(incremental_events)
    permutations = source_permutation_tests(
        dynamic, static, vectors, n_permutations=n_permutations, seed=seed
    )
    external = phase_external_metrics(dynamic, static, release)
    events.to_csv(output / "event_level_prediction_metrics.csv", index=False)
    summary.to_csv(output / "prediction_summary.csv", index=False)
    contrasts.to_csv(output / "paired_contrasts.csv", index=False)
    incremental_events.to_csv(output / "incremental_prediction_metrics.csv", index=False)
    incremental_summary.to_csv(output / "incremental_prediction_summary.csv", index=False)
    permutations.to_csv(output / "source_permutation_tests.csv", index=False)
    external.to_csv(output / "phase_external_metrics.csv", index=False)
    arrays: dict[str, np.ndarray] = {
        "neurons": np.asarray(base.neurons),
        "folds": base.folds,
        "lags": base.lags,
        "phases": np.asarray(base.phases),
    }
    for method, item in dynamic.items():
        arrays[f"{method}__matrices"] = item.matrices.astype(np.float32)
        arrays[f"{method}__validity"] = item.validity.astype(np.float32)
    for method, item in static.items():
        arrays[f"{method}__lags"] = item["lags"].astype(np.int16)
        arrays[f"{method}__matrices"] = item["matrices"].astype(np.float32)
    np.savez_compressed(output / "aligned_phase_matrices.npz", **arrays)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "onset-aware predictive propagation and phase-resolved external correspondence",
        "sampling_rate_hz": cohort.fps,
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "stimulus_periods_seconds": STIMULUS_PERIODS_SECONDS,
        "source_window_frames": 4,
        "baseline_window_frames": 4,
        "horizon_frames": base.lags.tolist(),
        "primary_target": "leave-one-worm-out event-residual target pattern",
        "primary_metric": "worm-clustered mean event-wise Spearman across target neurons",
        "matrix_convention": "target row, source column; diagonal zeroed",
        "matrix_variants": ["raw", "column_l2"],
        "episodes": ["stimulus_onset", "quiet_pseudo_onset"],
        "source_permutation_count": n_permutations,
        "seed": seed,
        "claim_boundary": "stimulus-locked predictive propagation under an observational learned law; not causal/anatomical/physical-delay identification",
        "author_context": "SBTG paper matching was minimal and did not use stimulus onset (provided by project author on 2026-08-27)",
        "release_path": str(release),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    validation = {
        "all_checks_pass": True,
        "checks": {
            "dynamic_methods": sorted(dynamic),
            "static_methods": sorted(static),
            "event_metric_rows": int(len(events)),
            "summary_rows": int(len(summary)),
            "contrast_rows": int(len(contrasts)),
            "incremental_event_rows": int(len(incremental_events)),
            "incremental_summary_rows": int(len(incremental_summary)),
            "permutation_rows": int(len(permutations)),
            "phase_external_rows": int(len(external)),
            "event_score_nonfinite_count": int(
                (~np.isfinite(events[["pearson", "spearman", "cosine", "sign_agreement"]].to_numpy())).sum()
            ),
            "outer_fold_coverage": sorted(np.unique(base.folds).astype(int).tolist()),
            "worm_coverage": cohort.n_worms == 20,
            "neuron_coverage": cohort.n_neurons == 54,
            "common_lags": [
                int(value)
                for value in sorted(
                    set(static["sbtg_published"]["lags"].tolist())
                    & set(base.lags.tolist())
                )
            ],
        },
    }
    validation["all_checks_pass"] = bool(
        validation["checks"]["event_score_nonfinite_count"] == 0
        and validation["checks"]["worm_coverage"]
        and validation["checks"]["neuron_coverage"]
        and validation["checks"]["outer_fold_coverage"] == [0, 1, 2, 3, 4]
    )
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    create_figures(output, summary, contrasts, incremental_summary, external)
    write_report(
        output,
        summary,
        contrasts,
        incremental_summary,
        permutations,
        external,
        validation,
    )
    write_checksums(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    run(args.output, args.release, args.n_permutations, args.seed)


if __name__ == "__main__":
    main()
