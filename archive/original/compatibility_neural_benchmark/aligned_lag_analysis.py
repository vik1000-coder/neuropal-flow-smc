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
from scipy.stats import pearsonr, spearmanr, ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
ALIGNED_ROOT = ROOT / "results/aligned_lag_response_20260828"
OUTPUT = ALIGNED_ROOT / "analysis"
STABILITY_RUN = (
    ROOT
    / "results/conditional_distribution_benchmark"
    / "rollout_stability_world_model_v2_20260828"
)
FULL_HISTORY_RUN = (
    ROOT
    / "results/conditional_distribution_benchmark"
    / "biological_world_model_20260828"
)
E22_RUN = ROOT / "results/biological_lag_analysis_20260828"

RUNS = {
    "regularized_direct": (
        ALIGNED_ROOT / "direct_regularized_N256",
        "tcn_flow128_dropout15_jitter_p01__direct__N256__f*__s*.npz",
    ),
    "regularized_progressive": (
        ALIGNED_ROOT / "progressive_regularized_N64",
        "tcn_flow128_dropout15_jitter_p01__progressive__N64__f*__s*.npz",
    ),
    "legacy_direct": (
        ALIGNED_ROOT / "direct_control_N256",
        "flow_wide128_dropout10__direct__N256__f*__s*.npz",
    ),
}

METHOD_LABELS = {
    "regularized_direct": "Regularized flow · direct N=256",
    "regularized_progressive": "Regularized flow · ESS-SMC N=64",
    "legacy_direct": "Legacy wide flow · direct N=256",
}

COLORS = {
    "regularized_direct": "#2563EB",
    "regularized_progressive": "#059669",
    "legacy_direct": "#94A3B8",
}

NAMED_EDGES = {
    ("AWC", "AIY"): -1,
    ("AWC", "AIB"): 1,
    ("RIB", "AVB"): 1,
}


@dataclass
class AlignedRun:
    name: str
    method: str
    model_id: str
    particles: int
    neurons: np.ndarray
    delays: np.ndarray
    delay_seconds: np.ndarray
    horizons: np.ndarray
    horizon_seconds: np.ndarray
    phases: np.ndarray
    coefficient_by_seed: dict[int, np.ndarray]
    raw_by_seed: dict[int, np.ndarray]
    validity_by_seed: dict[int, np.ndarray]
    diagnostics: pd.DataFrame
    files: list[Path]

    @property
    def coefficient(self) -> np.ndarray:
        return np.mean(np.stack(list(self.coefficient_by_seed.values())), axis=0)

    @property
    def raw(self) -> np.ndarray:
        return np.mean(np.stack(list(self.raw_by_seed.values())), axis=0)

    @property
    def validity(self) -> np.ndarray:
        return np.mean(np.stack(list(self.validity_by_seed.values())), axis=0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return result
    finite = values[mask]
    order = np.argsort(finite)
    ranked = finite[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    unsorted = np.empty_like(adjusted)
    unsorted[order] = adjusted
    result[mask] = unsorted
    return result


def safe_corr(left: np.ndarray, right: np.ndarray, kind: str) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        return np.nan
    fn = pearsonr if kind == "pearson" else spearmanr
    return float(fn(left[mask], right[mask]).statistic)


def off_diagonal(matrix: np.ndarray, source_mask: np.ndarray | None = None) -> np.ndarray:
    d = matrix.shape[-1]
    keep = ~np.eye(d, dtype=bool)
    if source_mask is not None:
        keep &= np.broadcast_to(np.asarray(source_mask, bool)[None, :], (d, d))
    return np.asarray(matrix)[keep]


def reindex_events_by_chemical(array: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Reorder the event axis (axis 2) into chemical-code order per worm."""
    array = np.asarray(array)
    codes = np.asarray(codes, dtype=int)
    if array.ndim < 3 or codes.shape != (array.shape[0], array.shape[2]):
        raise RuntimeError("chemical archive shape mismatch")
    if not np.all(np.sort(codes, axis=1) == np.asarray([1, 2, 3])):
        raise RuntimeError("chemical codes are not permutations")
    result = np.empty_like(array)
    for worm in range(len(codes)):
        for event, code in enumerate(codes[worm]):
            result[worm, :, code - 1] = array[worm, :, event]
    return result


def load_run(name: str, run_dir: Path, pattern: str) -> AlignedRun:
    paths = sorted((run_dir / "responses").glob(pattern))
    if not paths:
        raise RuntimeError(f"no archives found for {name}")
    coefficient: dict[int, list[np.ndarray | None]] = {}
    raw: dict[int, list[np.ndarray | None]] = {}
    validity: dict[int, list[np.ndarray | None]] = {}
    diagnostics: list[dict[str, object]] = []
    neurons = delays = delay_seconds = horizons = horizon_seconds = phases = None
    method = model_id = None
    particles = None
    seen: set[tuple[int, int]] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["status"].item()) != "complete":
                raise RuntimeError(f"incomplete aligned response: {path}")
            fold = int(data["fold"])
            seed = int(data["model_seed"])
            if (fold, seed) in seen:
                raise RuntimeError(f"duplicate fold/seed {fold}/{seed} in {name}")
            seen.add((fold, seed))
            current = (
                data["neurons"].astype(str),
                data["delay_frames"].astype(int),
                data["delay_seconds"].astype(float),
                data["horizon_frames"].astype(int),
                data["horizon_seconds"].astype(float),
                data["phase_names"].astype(str),
            )
            if neurons is None:
                neurons, delays, delay_seconds, horizons, horizon_seconds, phases = current
                method = str(data["method"].item())
                model_id = str(data["model_id"].item())
                particles = int(data["n_particles"])
            elif not all(np.array_equal(a, b) for a, b in zip(current, (
                neurons, delays, delay_seconds, horizons, horizon_seconds, phases
            ))):
                raise RuntimeError(f"alignment mismatch in {path}")
            response = data["response_cumulative_mean"].astype(np.float64)
            gap = data["diagnostic_achieved_gap"].astype(np.float64)
            target_gap = data["diagnostic_target_gap"].astype(np.float64)
            valid = data["diagnostic_valid"].astype(np.float64)
            if "chemical_code_by_worm_event" in data.files:
                codes = data["chemical_code_by_worm_event"].astype(int)
                response = reindex_events_by_chemical(response, codes)
                gap = reindex_events_by_chemical(gap, codes)
                target_gap = reindex_events_by_chemical(target_gap, codes)
                valid = reindex_events_by_chemical(valid, codes)
            # Archive: worm, phase, event, delay, source, horizon, target.
            normalized = response / np.maximum(gap, 0.10)[..., None, None]
            # Analysis: worm, phase, delay, horizon, target, source.
            coefficient_local = normalized.mean(axis=2).transpose(0, 1, 2, 4, 5, 3)
            raw_local = response.mean(axis=2).transpose(0, 1, 2, 4, 5, 3)
            validity_local = valid.mean(axis=2)
            worms = data["worm_indices"].astype(int)
            coefficient.setdefault(seed, [None] * 20)
            raw.setdefault(seed, [None] * 20)
            validity.setdefault(seed, [None] * 20)
            for position, worm in enumerate(worms):
                if coefficient[seed][worm] is not None:
                    raise RuntimeError(f"worm {worm} repeated for seed {seed} in {name}")
                coefficient[seed][worm] = coefficient_local[position]
                raw[seed][worm] = raw_local[position]
                validity[seed][worm] = validity_local[position]
            if method == "progressive":
                minimum_ess = np.minimum(
                    data["diagnostic_min_step_ess_low"].astype(float),
                    data["diagnostic_min_step_ess_high"].astype(float),
                )
            else:
                minimum_ess = np.minimum(
                    data["diagnostic_ess_low"].astype(float),
                    data["diagnostic_ess_high"].astype(float),
                )
            achieved_fraction = gap / np.maximum(target_gap, 1e-8)
            for worm_position, worm in enumerate(worms):
                for phase_position, phase in enumerate(phases):
                    for event in range(valid.shape[2]):
                        for delay_position, delay in enumerate(delays):
                            for source_position, source in enumerate(neurons):
                                diagnostics.append({
                                    "run": name,
                                    "method": method,
                                    "model_id": model_id,
                                    "particles": particles,
                                    "fold": fold,
                                    "seed": seed,
                                    "worm": int(worm),
                                    "phase": str(phase),
                                    "event": event + 1,
                                    "delay_frames": int(delay),
                                    "delay_seconds": float(delay_seconds[delay_position]),
                                    "source": str(source),
                                    "valid": float(valid[worm_position, phase_position, event, delay_position, source_position]),
                                    "achieved_fraction": float(achieved_fraction[worm_position, phase_position, event, delay_position, source_position]),
                                    "minimum_ess": float(minimum_ess[worm_position, phase_position, event, delay_position, source_position]),
                                })
    for seed in coefficient:
        if any(item is None for item in coefficient[seed]):
            raise RuntimeError(f"seed {seed} does not cover all 20 worms in {name}")
    return AlignedRun(
        name=name,
        method=str(method),
        model_id=str(model_id),
        particles=int(particles),
        neurons=np.asarray(neurons),
        delays=np.asarray(delays),
        delay_seconds=np.asarray(delay_seconds),
        horizons=np.asarray(horizons),
        horizon_seconds=np.asarray(horizon_seconds),
        phases=np.asarray(phases),
        coefficient_by_seed={key: np.stack(value) for key, value in coefficient.items()},
        raw_by_seed={key: np.stack(value) for key, value in raw.items()},
        validity_by_seed={key: np.stack(value) for key, value in validity.items()},
        diagnostics=pd.DataFrame(diagnostics),
        files=paths,
    )


def phase_matrix(array: np.ndarray, phase: str, delay_position: int, horizon_position: int) -> np.ndarray:
    if phase == "onset":
        return array[:, 1, delay_position, horizon_position]
    if phase == "quiet":
        return array[:, 0, delay_position, horizon_position]
    if phase == "onset_minus_quiet":
        return (
            array[:, 1, delay_position, horizon_position]
            - array[:, 0, delay_position, horizon_position]
        )
    raise ValueError(phase)


def delay_position(run: AlignedRun, frames: int) -> int:
    match = np.flatnonzero(run.delays == frames)
    if len(match) != 1:
        raise RuntimeError(f"delay {frames} is unavailable in {run.name}")
    return int(match[0])


def matrix_relationships(runs: dict[str, AlignedRun]) -> pd.DataFrame:
    comparisons: list[tuple[str, str, str, int, int]] = [
        ("sampler", "regularized_direct", "regularized_progressive", 7, 7),
        ("model", "regularized_direct", "legacy_direct", 7, 7),
    ]
    rows: list[dict[str, object]] = []
    for comparison, left_name, right_name, left_delay, right_delay in comparisons:
        left, right = runs[left_name], runs[right_name]
        li, ri = delay_position(left, left_delay), delay_position(right, right_delay)
        for quantity in ("coefficient", "raw"):
            left_array = getattr(left, quantity)
            right_array = getattr(right, quantity)
            for estimand in ("onset", "quiet", "onset_minus_quiet"):
                for horizon_position, seconds in enumerate(left.horizon_seconds):
                    lm = phase_matrix(left_array, estimand, li, horizon_position).mean(axis=0)
                    rm = phase_matrix(right_array, estimand, ri, horizon_position).mean(axis=0)
                    source_valid_left = left.validity[:, :, li].mean(axis=(0, 1)) >= 0.50
                    source_valid_right = right.validity[:, :, ri].mean(axis=(0, 1)) >= 0.50
                    for scope, mask in (
                        ("all_off_diagonal", None),
                        ("shared_valid_sources", source_valid_left & source_valid_right),
                    ):
                        lv, rv = off_diagonal(lm, mask), off_diagonal(rm, mask)
                        rows.append({
                            "comparison": comparison,
                            "left": left_name,
                            "right": right_name,
                            "quantity": quantity,
                            "estimand": estimand,
                            "horizon_seconds": float(seconds),
                            "scope": scope,
                            "n_edges": int(np.isfinite(lv * rv).sum()),
                            "pearson": safe_corr(lv, rv, "pearson"),
                            "spearman": safe_corr(lv, rv, "spearman"),
                        })
    regularized = runs["regularized_direct"]
    for quantity in ("coefficient", "raw"):
        seed_arrays = getattr(regularized, f"{quantity}_by_seed")
        seeds = sorted(seed_arrays)
        if len(seeds) >= 2:
            for estimand in ("onset", "quiet", "onset_minus_quiet"):
                for horizon_position, seconds in enumerate(regularized.horizon_seconds):
                    left = phase_matrix(seed_arrays[seeds[0]], estimand, delay_position(regularized, 7), horizon_position).mean(axis=0)
                    right = phase_matrix(seed_arrays[seeds[1]], estimand, delay_position(regularized, 7), horizon_position).mean(axis=0)
                    rows.append({
                        "comparison": "generator_seed",
                        "left": f"regularized_direct_seed_{seeds[0]}",
                        "right": f"regularized_direct_seed_{seeds[1]}",
                        "quantity": quantity,
                        "estimand": estimand,
                        "horizon_seconds": float(seconds),
                        "scope": "all_off_diagonal",
                        "n_edges": len(off_diagonal(left)),
                        "pearson": safe_corr(off_diagonal(left), off_diagonal(right), "pearson"),
                        "spearman": safe_corr(off_diagonal(left), off_diagonal(right), "spearman"),
                    })
        for first, second in ((5, 7), (7, 9)):
            for estimand in ("onset", "quiet", "onset_minus_quiet"):
                for horizon_position, seconds in enumerate(regularized.horizon_seconds):
                    left = phase_matrix(regularized.coefficient if quantity == "coefficient" else regularized.raw, estimand, delay_position(regularized, first), horizon_position).mean(axis=0)
                    right = phase_matrix(regularized.coefficient if quantity == "coefficient" else regularized.raw, estimand, delay_position(regularized, second), horizon_position).mean(axis=0)
                    rows.append({
                        "comparison": "timing",
                        "left": f"delay_{first}",
                        "right": f"delay_{second}",
                        "quantity": quantity,
                        "estimand": estimand,
                        "horizon_seconds": float(seconds),
                        "scope": "all_off_diagonal",
                        "n_edges": len(off_diagonal(left)),
                        "pearson": safe_corr(off_diagonal(left), off_diagonal(right), "pearson"),
                        "spearman": safe_corr(off_diagonal(left), off_diagonal(right), "spearman"),
                    })
    return pd.DataFrame(rows)


def animal_split_stability(runs: dict[str, AlignedRun], repeats: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(20260828)
    splits = [rng.permutation(20) for _ in range(repeats)]
    rows: list[dict[str, object]] = []
    for name, run in runs.items():
        di = delay_position(run, 7)
        for estimand in ("onset", "quiet", "onset_minus_quiet"):
            for horizon_position, seconds in enumerate(run.horizon_seconds):
                values = phase_matrix(run.coefficient, estimand, di, horizon_position)
                correlations = []
                for order in splits:
                    left, right = values[order[:10]].mean(axis=0), values[order[10:]].mean(axis=0)
                    correlations.append(safe_corr(off_diagonal(left), off_diagonal(right), "pearson"))
                finite = np.asarray(correlations)[np.isfinite(correlations)]
                rows.append({
                    "run": name,
                    "estimand": estimand,
                    "horizon_seconds": float(seconds),
                    "repeats": len(finite),
                    "median_split_half_pearson": float(np.median(finite)),
                    "p05_split_half_pearson": float(np.quantile(finite, 0.05)),
                    "p95_split_half_pearson": float(np.quantile(finite, 0.95)),
                })
    return pd.DataFrame(rows)


def sampler_diagnostics(runs: dict[str, AlignedRun]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, run in runs.items():
        frame = run.diagnostics
        for phase in ("all", "quiet", "onset_aligned"):
            phase_frame = frame if phase == "all" else frame[frame.phase == phase]
            for source in ("all", "AWC", "RIB"):
                subset = phase_frame if source == "all" else phase_frame[phase_frame.source == source]
                rows.append({
                    "run": name,
                    "method_label": METHOD_LABELS[name],
                    "phase": phase,
                    "source": source,
                    "particles": run.particles,
                    "cells": len(subset),
                    "validity_rate": float(subset.valid.mean()),
                    "achieved_fraction_median": float(subset.achieved_fraction.median()),
                    "achieved_fraction_p10": float(subset.achieved_fraction.quantile(0.10)),
                    "minimum_ess_median": float(subset.minimum_ess.median()),
                    "minimum_ess_p10": float(subset.minimum_ess.quantile(0.10)),
                    "minimum_ess_fraction_median": float(subset.minimum_ess.median() / run.particles),
                })
    return pd.DataFrame(rows)


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, repeats: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def named_edge_analysis(runs: dict[str, AlignedRun]) -> pd.DataFrame:
    rng = np.random.default_rng(713502)
    rows: list[dict[str, object]] = []
    for name, run in runs.items():
        di = delay_position(run, 7)
        positions = {neuron: i for i, neuron in enumerate(run.neurons)}
        source_validity = run.validity[:, :, di].mean(axis=(0, 1))
        for (source, target), expected_sign in NAMED_EDGES.items():
            source_position, target_position = positions[source], positions[target]
            for estimand in ("onset", "quiet", "onset_minus_quiet"):
                for horizon_position, seconds in enumerate(run.horizon_seconds):
                    values = phase_matrix(run.coefficient, estimand, di, horizon_position)[:, target_position, source_position]
                    low, high = bootstrap_interval(values, rng)
                    test = ttest_1samp(values, popmean=0.0)
                    mean = float(values.mean())
                    rows.append({
                        "run": name,
                        "method_label": METHOD_LABELS[name],
                        "source": source,
                        "target": target,
                        "expected_sign": expected_sign,
                        "estimand": estimand,
                        "horizon_seconds": float(seconds),
                        "mean_coefficient": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "worm_sign_agreement": float(np.mean(np.sign(values) == np.sign(mean))),
                        "p_value": float(test.pvalue),
                        "source_validity": float(source_validity[source_position]),
                        "expected_sign_match": bool(np.sign(mean) == expected_sign),
                        "n_worms": len(values),
                    })
    frame = pd.DataFrame(rows)
    frame["bh_q_value"] = np.nan
    for estimand in frame.estimand.unique():
        mask = frame.estimand == estimand
        frame.loc[mask, "bh_q_value"] = bh_adjust(frame.loc[mask, "p_value"].to_numpy())
    return frame


def exploratory_edges(runs: dict[str, AlignedRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = ("regularized_direct", "regularized_progressive")
    all_rows: list[dict[str, object]] = []
    per_method: dict[tuple[str, str, int, int], dict[str, object]] = {}
    for name in methods:
        run = runs[name]
        di = delay_position(run, 7)
        positions = {neuron: i for i, neuron in enumerate(run.neurons)}
        source_validity = run.validity[:, :, di].mean(axis=(0, 1))
        for estimand in ("onset", "onset_minus_quiet"):
            p_values: list[float] = []
            local_rows: list[dict[str, object]] = []
            for horizon_position, seconds in enumerate(run.horizon_seconds):
                values = phase_matrix(run.coefficient, estimand, di, horizon_position)
                test = ttest_1samp(values, popmean=0.0, axis=0)
                mean = values.mean(axis=0)
                agreement = np.mean(np.sign(values) == np.sign(mean)[None], axis=0)
                magnitudes = np.abs(mean[~np.eye(len(run.neurons), dtype=bool)])
                for source in run.neurons:
                    source_position = positions[str(source)]
                    for target in run.neurons:
                        target_position = positions[str(target)]
                        if source == target:
                            continue
                        magnitude_percentile = float(np.mean(magnitudes <= abs(mean[target_position, source_position])))
                        row = {
                            "run": name,
                            "estimand": estimand,
                            "source": str(source),
                            "target": str(target),
                            "horizon_seconds": float(seconds),
                            "mean_coefficient": float(mean[target_position, source_position]),
                            "worm_sign_agreement": float(agreement[target_position, source_position]),
                            "magnitude_percentile": magnitude_percentile,
                            "source_validity": float(source_validity[source_position]),
                            "p_value": float(test.pvalue[target_position, source_position]),
                        }
                        local_rows.append(row)
                        p_values.append(row["p_value"])
            q_values = bh_adjust(np.asarray(p_values))
            for row, q_value in zip(local_rows, q_values):
                row["bh_q_value"] = float(q_value)
                all_rows.append(row)
                horizon_position = int(np.flatnonzero(np.isclose(run.horizon_seconds, row["horizon_seconds"]))[0])
                per_method[(name, estimand, positions[row["source"]], positions[row["target"]], horizon_position)] = row
    edge_tests = pd.DataFrame(all_rows)
    regularized = runs["regularized_direct"]
    positions = {neuron: i for i, neuron in enumerate(regularized.neurons)}
    candidates: list[dict[str, object]] = []
    reg_seeds = sorted(regularized.coefficient_by_seed)
    for estimand in ("onset", "onset_minus_quiet"):
        for horizon_position, seconds in enumerate(regularized.horizon_seconds):
            timing_matrices = [
                phase_matrix(regularized.coefficient, estimand, delay_position(regularized, delay), horizon_position).mean(axis=0)
                for delay in (5, 7, 9)
            ]
            seed_matrices = [
                phase_matrix(regularized.coefficient_by_seed[seed], estimand, delay_position(regularized, 7), horizon_position).mean(axis=0)
                for seed in reg_seeds
            ]
            for source in regularized.neurons:
                source_position = positions[str(source)]
                for target in regularized.neurons:
                    target_position = positions[str(target)]
                    if source == target:
                        continue
                    direct = per_method[("regularized_direct", estimand, source_position, target_position, horizon_position)]
                    progressive = per_method[("regularized_progressive", estimand, source_position, target_position, horizon_position)]
                    means = np.asarray([direct["mean_coefficient"], progressive["mean_coefficient"]])
                    cross_method = np.sign(means[0]) == np.sign(means[1])
                    timing_sign = len(set(np.sign(matrix[target_position, source_position]) for matrix in timing_matrices)) == 1
                    seed_sign = len(set(np.sign(matrix[target_position, source_position]) for matrix in seed_matrices)) == 1
                    base_consensus = bool(
                        cross_method
                        and timing_sign
                        and seed_sign
                        and direct["bh_q_value"] <= 0.10
                        and progressive["bh_q_value"] <= 0.10
                        and direct["worm_sign_agreement"] >= 0.70
                        and progressive["worm_sign_agreement"] >= 0.70
                        and direct["source_validity"] >= 0.50
                        and progressive["source_validity"] >= 0.50
                    )
                    contextual_onset = bool(estimand == "onset" and base_consensus)
                    strict = bool(estimand == "onset_minus_quiet" and base_consensus)
                    score = float(np.mean([
                        direct["worm_sign_agreement"],
                        progressive["worm_sign_agreement"],
                        direct["magnitude_percentile"],
                        progressive["magnitude_percentile"],
                        float(cross_method), float(timing_sign), float(seed_sign),
                    ]))
                    candidates.append({
                        "estimand": estimand,
                        "source": str(source),
                        "target": str(target),
                        "horizon_seconds": float(seconds),
                        "direct_coefficient": direct["mean_coefficient"],
                        "progressive_coefficient": progressive["mean_coefficient"],
                        "direct_bh_q": direct["bh_q_value"],
                        "progressive_bh_q": progressive["bh_q_value"],
                        "direct_sign_agreement": direct["worm_sign_agreement"],
                        "progressive_sign_agreement": progressive["worm_sign_agreement"],
                        "direct_source_validity": direct["source_validity"],
                        "progressive_source_validity": progressive["source_validity"],
                        "cross_method_sign": bool(cross_method),
                        "timing_sign": bool(timing_sign),
                        "generator_seed_sign": bool(seed_sign),
                        "contextual_onset_pass": contextual_onset,
                        "strict_stimulus_specific_pass": strict,
                        "exploratory_score": score,
                    })
    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["strict_stimulus_specific_pass", "contextual_onset_pass", "exploratory_score"],
        ascending=[False, False, False],
    )
    return edge_tests, candidate_frame


def predictive_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    leaderboard = pd.read_csv(STABILITY_RUN / "leaderboard.csv")
    ids = [
        "control_tcn_flow128_natural",
        "tcn_flow128_dropout15_jitter_p01",
        "tcn_mdn4_jitter_p01",
    ]
    prediction = leaderboard[
        (leaderboard.phase == "biological_confirmation") & leaderboard.model_id.isin(ids)
    ].copy()
    prediction = prediction[[
        "rank", "model_id", "energy__mean", "energy__se",
        "energy__stim_balanced__mean", "energy__stim_balanced__se",
        "variogram__mean", "coverage90__mean", "sharpness90__mean", "n_trials",
    ]].sort_values("energy__mean")
    rollout = pd.read_csv(STABILITY_RUN / "rollout_confirmation/rollout_metrics_aggregate.csv")
    rollout = rollout[rollout.model_id.isin(ids)].copy()
    rollout = rollout.sort_values(["model_id", "rollout_horizon_seconds"])
    return prediction, rollout


def matrix_summary(runs: dict[str, AlignedRun]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, run in runs.items():
        for delay_pos, delay in enumerate(run.delays):
            for horizon_pos, seconds in enumerate(run.horizon_seconds):
                for estimand in ("quiet", "onset", "onset_minus_quiet"):
                    coefficient = phase_matrix(run.coefficient, estimand, delay_pos, horizon_pos)
                    raw = phase_matrix(run.raw, estimand, delay_pos, horizon_pos)
                    mean_coefficient = coefficient.mean(axis=0)
                    mean_raw = raw.mean(axis=0)
                    for target_pos, target in enumerate(run.neurons):
                        for source_pos, source in enumerate(run.neurons):
                            rows.append({
                                "run": name,
                                "estimand": estimand,
                                "delay_frames": int(delay),
                                "delay_seconds": float(run.delay_seconds[delay_pos]),
                                "horizon_seconds": float(seconds),
                                "target": str(target),
                                "source": str(source),
                                "coefficient_mean": float(mean_coefficient[target_pos, source_pos]),
                                "coefficient_sd_across_worms": float(coefficient[:, target_pos, source_pos].std(ddof=1)),
                                "raw_contrast_mean": float(mean_raw[target_pos, source_pos]),
                            })
    return pd.DataFrame(rows)


def figures(
    runs: dict[str, AlignedRun], prediction: pd.DataFrame, rollout: pd.DataFrame,
    diagnostics: pd.DataFrame, relationships: pd.DataFrame, named: pd.DataFrame,
    output: Path,
) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = {
        "control_tcn_flow128_natural": "Legacy control",
        "tcn_flow128_dropout15_jitter_p01": "Regularized flow",
        "tcn_mdn4_jitter_p01": "Regularized MDN",
    }
    colors = {
        "control_tcn_flow128_natural": "#94A3B8",
        "tcn_flow128_dropout15_jitter_p01": "#2563EB",
        "tcn_mdn4_jitter_p01": "#F59E0B",
    }
    order = prediction.model_id.tolist()
    axes[0].bar(
        [labels[item] for item in order], prediction.energy__mean,
        yerr=prediction.energy__se, color=[colors[item] for item in order], capsize=3,
    )
    axes[0].set_ylabel("One-step energy score (lower is better)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("Held-out one-step prediction")
    for model_id, group in rollout.groupby("model_id"):
        axes[1].plot(
            group.rollout_horizon_seconds, group.energy__stim_balanced__mean,
            marker="o", label=labels[model_id], color=colors[model_id],
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Rollout horizon (s)")
    axes[1].set_ylabel("Stimulus-balanced energy (lower is better)")
    axes[1].set_title("Ancestral rollout stability")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "predictive_and_rollout.png", dpi=190)
    plt.close(fig)

    overall = diagnostics[(diagnostics.phase == "all") & (diagnostics.source == "all")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for axis, metric, title, reference in (
        (axes[0], "validity_rate", "Compatibility-valid fraction", None),
        (axes[1], "achieved_fraction_median", "Median achieved / target gap", None),
        (axes[2], "minimum_ess_fraction_median", "Median minimum ESS / N", None),
    ):
        axis.bar(
            [METHOD_LABELS[item] for item in overall.run], overall[metric],
            color=[COLORS[item] for item in overall.run],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylim(bottom=0)
        if reference is not None:
            axis.axhline(reference, color="black", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(figure_dir / "sampler_diagnostics.png", dpi=190)
    plt.close(fig)

    selected = relationships[
        (relationships.quantity == "coefficient")
        & (relationships.scope == "all_off_diagonal")
        & relationships.comparison.isin(["sampler", "generator_seed", "timing"])
    ].copy()
    selected["series"] = selected.apply(
        lambda row: (
            "Direct vs ESS-SMC" if row.comparison == "sampler"
            else "Generator seeds" if row.comparison == "generator_seed"
            else f"Timing {row.left.replace('delay_', '')} vs {row.right.replace('delay_', '')} frames"
        ), axis=1,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, estimand, title in zip(
        axes, ("onset", "onset_minus_quiet"),
        ("Onset coefficient matrices", "Onset − matched quiet matrices"),
    ):
        for series, group in selected[selected.estimand == estimand].groupby("series"):
            axis.plot(group.horizon_seconds, group.pearson, marker="o", label=series)
        axis.axhline(0, color="black", lw=0.8)
        axis.set_xlabel("Response horizon (s)")
        axis.set_title(title)
    axes[0].set_ylabel("Off-diagonal Pearson correlation")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "matrix_stability.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharex=True)
    for axis, ((source, target), expected) in zip(axes, NAMED_EDGES.items()):
        subset = named[(named.source == source) & (named.target == target)]
        for name in runs:
            for estimand, linestyle in (("onset", "-"), ("quiet", "--")):
                group = subset[(subset.run == name) & (subset.estimand == estimand)]
                axis.plot(
                    group.horizon_seconds, group.mean_coefficient, marker="o", ls=linestyle,
                    color=COLORS[name], alpha=1 if estimand == "onset" else 0.65,
                    label=f"{METHOD_LABELS[name]} · {estimand}" if axis is axes[0] else None,
                )
        axis.axhline(0, color="black", lw=0.8)
        axis.set_title(f"{source} → {target} (prior sign {'+' if expected > 0 else '−'})")
        axis.set_xlabel("Response horizon (s)")
    axes[0].set_ylabel("Normalized response coefficient")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / "named_edges.png", dpi=190)
    plt.close(fig)

    regularized = runs["regularized_direct"]
    progressive = runs["regularized_progressive"]
    horizon_position = int(np.flatnonzero(np.isclose(regularized.horizon_seconds, 1.0))[0])
    matrices = []
    for run in (regularized, progressive):
        di = delay_position(run, 7)
        for estimand in ("onset", "onset_minus_quiet"):
            matrix = phase_matrix(run.coefficient, estimand, di, horizon_position).mean(axis=0).copy()
            np.fill_diagonal(matrix, 0.0)
            matrices.append(matrix)
    onset_scale = np.quantile(np.abs(np.concatenate([matrix.ravel() for matrix in matrices[::2]])), 0.99)
    difference_scale = np.quantile(np.abs(np.concatenate([matrix.ravel() for matrix in matrices[1::2]])), 0.99)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    titles = (
        "Direct · onset", "Direct · onset − quiet",
        "ESS-SMC · onset", "ESS-SMC · onset − quiet",
    )
    for axis, matrix, title, scale in zip(
        axes.ravel(), matrices, titles,
        (onset_scale, difference_scale, onset_scale, difference_scale),
    ):
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel("Source neuron index")
        axis.set_ylabel("Target neuron index")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Aligned 1-second inter-neuron coefficients (diagonal hidden; target row × source column)")
    fig.tight_layout()
    fig.savefig(figure_dir / "lag_matrix_heatmaps.png", dpi=190)
    plt.close(fig)


def write_protocol(runs: dict[str, AlignedRun], output: Path) -> None:
    protocol = {
        "created_utc": utc_now(),
        "cohort": {"worms": 20, "neurons": 54, "fps": 4.0, "events_per_worm": 3},
        "conditional_model": {
            "winner": "tcn_flow128_dropout15_jitter_p01",
            "selection": "held-out whole-worm one-step energy plus frozen multistep rollout confirmation",
            "regularization": "15% dropout plus 0.01 standardized input jitter; clean and noisy loss averaged in each optimizer update",
        },
        "sampling": {
            "direct": "N=256 compatibility importance weighting",
            "progressive": "N=64 progressive bridge SMC with ESS-adaptive tempering, branch factor 2, future branch factor 2",
            "coefficient": "cumulative high-minus-low response divided by max(achieved source gap, 0.10)",
        },
        "alignment": {
            "primary_delay_frames": 7,
            "primary_source_window_seconds": [1.0, 1.75],
            "timing_sensitivities_frames": [5, 9],
            "quiet_control": "same delay relative to a pseudo-cut 15 seconds before each true onset",
            "response_horizons_seconds": [0.5, 1.0, 2.0, 4.0],
        },
        "inference": {
            "unit": "worm; three events and available generator seeds averaged within worm",
            "named_edges": [f"{source}->{target}" for source, target in NAMED_EDGES],
            "multiplicity": "BH within estimand for named edges; global across all off-diagonal edge-horizon cells within method/estimand for discovery",
            "strict_candidate": "onset-minus-quiet q<=0.10 in both samplers, >=0.70 worm sign agreement, >=0.50 source validity, and sign agreement across samplers, direct generator seeds, and all three onset delays",
        },
        "claim_boundary": "observational, model-relative response coefficients; not identified synapses, causal interventions, receptor action, or physical transmission delays",
        "runs": {name: [str(path.relative_to(ROOT)) for path in run.files] for name, run in runs.items()},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def create_report(
    output: Path, prediction: pd.DataFrame, rollout: pd.DataFrame,
    diagnostics: pd.DataFrame, relationships: pd.DataFrame,
    animal_stability: pd.DataFrame, named: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    winner = prediction[prediction.model_id == "tcn_flow128_dropout15_jitter_p01"].iloc[0]
    control = prediction[prediction.model_id == "control_tcn_flow128_natural"].iloc[0]
    rollout_winner = rollout[
        (rollout.model_id == "tcn_flow128_dropout15_jitter_p01")
        & np.isclose(rollout.rollout_horizon_seconds, 10.0)
    ].iloc[0]
    rollout_control = rollout[
        (rollout.model_id == "control_tcn_flow128_natural")
        & np.isclose(rollout.rollout_horizon_seconds, 10.0)
    ].iloc[0]
    overall = diagnostics[(diagnostics.phase == "all") & (diagnostics.source == "all")].set_index("run")
    sampler = relationships[
        (relationships.comparison == "sampler")
        & (relationships.quantity == "coefficient")
        & (relationships.scope == "all_off_diagonal")
    ]
    seed = relationships[
        (relationships.comparison == "generator_seed")
        & (relationships.quantity == "coefficient")
        & (relationships.scope == "all_off_diagonal")
    ]
    timing = relationships[
        (relationships.comparison == "timing")
        & (relationships.quantity == "coefficient")
        & (relationships.scope == "all_off_diagonal")
    ]
    strict_count = int(candidates.strict_stimulus_specific_pass.sum())
    contextual_count = int(candidates.contextual_onset_pass.sum())
    onset_sampler = sampler[sampler.estimand == "onset"].sort_values("horizon_seconds")
    diff_sampler = sampler[sampler.estimand == "onset_minus_quiet"].sort_values("horizon_seconds")
    onset_seed = seed[seed.estimand == "onset"].sort_values("horizon_seconds")
    diff_seed = seed[seed.estimand == "onset_minus_quiet"].sort_values("horizon_seconds")
    timing_onset = timing[timing.estimand == "onset"]
    timing_diff = timing[timing.estimand == "onset_minus_quiet"]
    rib = named[
        (named.source == "RIB") & (named.target == "AVB")
        & (named.run.isin(["regularized_direct", "regularized_progressive"]))
    ]
    rib_diff_1s = rib[
        (rib.estimand == "onset_minus_quiet") & np.isclose(rib.horizon_seconds, 1.0)
    ].set_index("run")
    report = f"""# Aligned lag dynamics: predictive regularization and ESS-SMC

## Technical summary

The best result is an estimator and prediction improvement, not a new biological-delay claim. The dropout-plus-jitter conditional flow improves confirmation one-step energy from **{control.energy__mean:.6f}** to **{winner.energy__mean:.6f}** ({100*(control.energy__mean-winner.energy__mean)/control.energy__mean:.2f}% lower) and improves the 10-second stimulus-balanced rollout energy from **{rollout_control.energy__stim_balanced__mean:.6f}** to **{rollout_winner.energy__stim_balanced__mean:.6f}** ({100*(rollout_control.energy__stim_balanced__mean-rollout_winner.energy__stim_balanced__mean)/rollout_control.energy__stim_balanced__mean:.1f}% lower). Progressive ESS-SMC raises compatibility validity from **{overall.loc['regularized_direct','validity_rate']:.3f}** to **{overall.loc['regularized_progressive','validity_rate']:.3f}** while using 64 rather than 256 particles.

The onset-only normalized coefficient matrices have moderate direct-versus-ESS-SMC agreement (Pearson **{onset_sampler.pearson.min():.3f}–{onset_sampler.pearson.max():.3f}** across 0.5–4 seconds), but the biologically sharper onset-minus-matched-quiet matrices do not reproduce (Pearson **{diff_sampler.pearson.min():.3f}–{diff_sampler.pearson.max():.3f}**). Generator-seed agreement is likewise only **{onset_seed.pearson.min():.3f}–{onset_seed.pearson.max():.3f}** for onset and **{diff_seed.pearson.min():.3f}–{diff_seed.pearson.max():.3f}** for onset-minus-quiet. Although **{contextual_count}** onset-only edge-horizon cells pass the other consensus gates, no edge passes the prospective stimulus-specific rule (**{strict_count}** promoted cells).

## Key findings

1. **Regularization helps the learned transition law.** The gain is small at one step but grows under ancestral rollout, exactly where sampling-derived lag matrices are most vulnerable to off-manifold drift.
2. **ESS-SMC fixes much of the particle-compatibility failure.** Overall validity is **{overall.loc['regularized_progressive','validity_rate']:.3f}** versus **{overall.loc['regularized_direct','validity_rate']:.3f}** for direct importance weighting; median minimum ESS/N is **{overall.loc['regularized_progressive','minimum_ess_fraction_median']:.3f}** versus **{overall.loc['regularized_direct','minimum_ess_fraction_median']:.3f}**.
3. **Onset matrices contain a reproducible generic component, but stimulus recruitment does not.** Adjacent timing-window correlations for onset range from **{timing_onset.pearson.min():.3f}** to **{timing_onset.pearson.max():.3f}**; onset-minus-quiet timing correlations range from **{timing_diff.pearson.min():.3f}** to **{timing_diff.pearson.max():.3f}**.
4. **The earlier RIB→AVB candidate is a contextual activity relationship, not an onset-specific effect.** Its onset coefficient remains positive across the two regularized samplers, but the matched quiet coefficient is equal or larger in most cells. At 1 second the onset-minus-quiet estimate is **{rib_diff_1s.loc['regularized_direct','mean_coefficient']:+.3f}** for direct (BH q={rib_diff_1s.loc['regularized_direct','bh_q_value']:.3f}) and **{rib_diff_1s.loc['regularized_progressive','mean_coefficient']:+.3f}** for ESS-SMC (q={rib_diff_1s.loc['regularized_progressive','bh_q_value']:.3f}).
5. **The AWC circuit signs remain estimator-sensitive.** Progressive ESS-SMC gives a positive AWC→AIB onset coefficient, while direct-flow signs change with horizon; AWC→AIY reverses between samplers. Neither onset-minus-quiet contrast agrees robustly across samplers, seeds, and timing. Literature signs are useful priors, not a substitute for measured onset specificity.
6. **There is real generic coupling structure.** URA→URB, RIP→OLL, and related onset-only dependencies are strong across samplers, timing, seeds, and worms. Their quiet coefficients are also substantial and their onset-minus-quiet tests do not survive multiplicity, so they are best treated as state-dynamics relationships rather than stimulus-recruited edges.

![Predictive and rollout metrics](figures/predictive_and_rollout.png)

![Sampler diagnostics](figures/sampler_diagnostics.png)

## Scope and definitions

- The cohort is 20 held-out complete-case worms and 54 NeuroPAL neuron classes sampled at 4 Hz.
- The conditional density is the distribution of the next 54-neuron activity vector given 80 history frames (20 seconds) and an aligned binary stimulus history.
- The primary source window is 1.0–1.75 seconds after each known stimulus onset; 0.5–1.25 and 1.5–2.25 seconds are timing sensitivities. Each true onset is paired with a pseudo-cut 15 seconds earlier.
- A matrix entry is the cumulative target response to a source-compatible high-versus-low contrast divided by `max(achieved source gap, 0.10)`. Matrix orientation is target row by source column.
- A response horizon is cumulative future averaging time. It is not a receptor latency or transmission delay.

## Methodology

### Conditional-density search

The architecture search compared legacy and full-history TCNs, multiscale TCNs, GRUs, transformers, phase-aware encoders, Gaussian/low-rank/MDN/flow heads, stimulus balancing, dropout, transition weighting, and 0.005–0.05 standardized input jitter. Selection used whole-worm cross-validation and held-out proper scores; atlases and named biological edges were not selection inputs. The final regularized flow uses 15% dropout and 0.01 input jitter with clean and noisy losses averaged inside the same optimizer update.

### Aligned lag sampling

Direct importance weighting used 256 paths for two generator seeds and three onset delays. Progressive SMC used 64 retained particles, branch factor 2, future branch factor 2, and ESS-adaptive tempering for the primary delay. The legacy wide flow was resampled with the same aligned direct protocol as a model control. Three events and generator seeds were averaged within worm before inference.

### Biological decision rule

The primary estimand is onset minus its matched quiet pseudo-onset. A discovery edge must have BH q≤0.10 in both regularized direct and ESS-SMC analyses, at least 70% worm sign agreement, at least 50% source validity, and matching signs across sampler, the two direct-flow generator seeds, and all three onset delays. This deliberately asks more than a large raw onset coefficient.

![Matrix stability](figures/matrix_stability.png)

![Named edges](figures/named_edges.png)

![Lag matrix heatmaps](figures/lag_matrix_heatmaps.png)

## Limitations and robustness

- **Generator uncertainty remains substantial.** Two independently trained regularized-flow seeds yield only modest onset-matrix agreement and essentially no stable onset-minus-quiet matrix.
- **Stimulus subtraction increases variance.** The negative control is scientifically necessary, but it exposes that the apparent raw-onset structure is largely shared with nearby quiet dynamics.
- **Progressive SMC improves compatibility, not biological identification.** Higher ESS and validity reduce Monte Carlo failure; they cannot recover a stimulus-specific edge absent a stable learned-law contrast.
- **The old and new matrices are not interchangeable.** The earlier E22 candidate used a source window that began at onset and partly preceded sustained AWC suppression. This run corrects timing and treats the earlier result as a hypothesis, not confirmation.
- **The cohort is small.** Worm is the replication unit (n=20), adjacent horizons are dependent, and all edgewise discovery tests are exploratory even with multiplicity control.
- **Claim boundary:** these are observational, model-relative response coefficients. They do not identify synapses, causal interventions, neuromodulator receptor action, anatomical rewiring, or physical transmission delays.

The random 10-versus-10 animal split results are saved in `animal_split_stability.csv`; complete sampler/model/seed/timing comparisons are in `matrix_correlations.csv`. The exact worm-level matrices, raw contrasts, validity tensors, neuron order, delays, and horizons are in `aligned_matrices.npz`.

## Next steps

1. Freeze this corrected timing and ESS-SMC protocol before collecting or opening another cohort.
2. Use RIB→AVB and AWC→AIB only as declared prospective hypotheses; require onset-minus-quiet replication, not raw onset magnitude.
3. In new data, increase biological replication before particles. The current bottleneck is generator/animal stability, not ESS.
4. If perturbational data become available, evaluate predicted target distributions after source perturbation directly; do not use connectome or receptor presence as a surrogate for activity dynamics.
5. For variance/gain biology, design a separately powered conditional-scale experiment with repeated trials per worm. The present single-trace event structure is better suited to mean/path response than edgewise variance modulation.

## Further questions

- Does the generic RIB→AVB dependency replicate during stimulus-free periods in a new cohort?
- Are stimulus-specific effects more stable at the level of low-rank modules or functional groups than individual ordered edges?
- Can jointly trained deep ensembles or posterior weight sampling improve generator-seed stability without degrading rollout calibration?
- Would source-target perturbation experiments support distributional changes in gain or variance even when mean onset-minus-quiet effects are small?

## Saved artifacts

- `predictive_model_comparison.csv` and `rollout_comparison.csv`: held-out conditional-density results.
- `sampler_diagnostics.csv`: validity, achieved gap, and ESS summaries.
- `matrix_correlations.csv` and `animal_split_stability.csv`: sampler, model, seed, timing, and animal stability.
- `named_edges.csv`, `exploratory_edge_tests.csv`, and `candidate_edges.csv`: declared and discovery-level biological tests.
- `matrix_summary_long.csv` and `aligned_matrices.npz`: readable and exact lag matrices.
- `protocol.json`, `validation.json`, and `checksums.sha256`: frozen definitions, validation, and integrity ledger.
"""
    (output / "TECHNICAL_REPORT.md").write_text(report)


def write_artifact(
    output: Path, prediction: pd.DataFrame, rollout: pd.DataFrame,
    diagnostics: pd.DataFrame, relationships: pd.DataFrame, named: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    title = "Aligned lag dynamics: predictive regularization and ESS-SMC"
    generated = utc_now()
    overall = diagnostics[(diagnostics.phase == "all") & (diagnostics.source == "all")]
    stability = relationships[
        (relationships.quantity == "coefficient")
        & (relationships.scope == "all_off_diagonal")
        & relationships.comparison.isin(["sampler", "generator_seed"])
        & relationships.estimand.isin(["onset", "onset_minus_quiet"])
    ].copy()
    stability["series"] = stability.apply(
        lambda row: f"{row.comparison}: {row.estimand}", axis=1
    )
    named_plot = named[
        named.run.isin(["regularized_direct", "regularized_progressive"])
        & named.estimand.isin(["onset", "quiet"])
    ].copy()
    named_plot["edge"] = named_plot.source + "→" + named_plot.target
    named_plot["series"] = named_plot.run + ": " + named_plot.estimand
    source = {
        "id": "aligned_lag_source",
        "label": "Aligned held-out lag-response analysis",
        "path": "results/aligned_lag_response_20260828/analysis",
        "query": {
            "engine": "local Python",
            "language": "python",
            "sql": "SELECT * FROM read_csv_auto('results/aligned_lag_response_20260828/analysis/sampler_diagnostics.csv')",
            "description": "Worm-level aggregation of held-out aligned direct and progressive-SMC response archives.",
            "tables_used": [
                "results/aligned_lag_response_20260828/analysis/sampler_diagnostics.csv",
                "results/aligned_lag_response_20260828/analysis/matrix_correlations.csv",
                "results/aligned_lag_response_20260828/analysis/named_edges.csv",
            ],
            "executed_at": generated,
            "filters": ["20 worms", "54 neurons", "known onset and matched quiet cuts", "primary delay 7 frames"],
            "metric_definitions": [
                "Coefficient is cumulative high-minus-low response divided by max(achieved source gap, 0.10).",
                "Validity is the saved compatibility diagnostic; worm is the inferential unit.",
                "Onset-minus-quiet is the primary stimulus-specific estimand.",
            ],
        },
    }
    def widget_source(identifier: str, label: str, path: str, sql: str) -> dict[str, object]:
        return {
            "id": identifier,
            "label": label,
            "path": path,
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": sql,
                "tables_used": [path],
                "description": label,
                "executed_at": generated,
                "filters": ["20 held-out worms", "54 aligned neurons", "known onset and matched quiet controls"],
                "metric_definitions": source["query"]["metric_definitions"],
            },
        }
    rollout_source = widget_source(
        "rollout_source", "Held-out rollout metrics",
        "results/aligned_lag_response_20260828/analysis/rollout_comparison.csv",
        "SELECT * FROM read_csv_auto('results/aligned_lag_response_20260828/analysis/rollout_comparison.csv')",
    )
    diagnostics_source = widget_source(
        "diagnostics_source", "Aligned sampler diagnostics",
        "results/aligned_lag_response_20260828/analysis/sampler_diagnostics.csv",
        "SELECT * FROM read_csv_auto('results/aligned_lag_response_20260828/analysis/sampler_diagnostics.csv') WHERE phase = 'all' AND source = 'all'",
    )
    stability_source = widget_source(
        "stability_source", "Aligned lag-matrix correlations",
        "results/aligned_lag_response_20260828/analysis/matrix_correlations.csv",
        "SELECT * FROM read_csv_auto('results/aligned_lag_response_20260828/analysis/matrix_correlations.csv') WHERE quantity = 'coefficient' AND scope = 'all_off_diagonal'",
    )
    named_source = widget_source(
        "named_source", "Declared biological edge coefficients",
        "results/aligned_lag_response_20260828/analysis/named_edges.csv",
        "SELECT * FROM read_csv_auto('results/aligned_lag_response_20260828/analysis/named_edges.csv') WHERE run IN ('regularized_direct', 'regularized_progressive')",
    )
    candidate_source = widget_source(
        "candidate_source", "Exploratory onset-minus-quiet candidates",
        "results/aligned_lag_response_20260828/analysis/candidate_edges.csv",
        "SELECT * FROM read_csv_auto('results/aligned_lag_response_20260828/analysis/candidate_edges.csv') WHERE estimand = 'onset_minus_quiet' ORDER BY exploratory_score DESC LIMIT 25",
    )
    charts = [
        {
            "id": "rollout_chart", "title": "Stimulus-balanced rollout energy",
            "subtitle": "The regularized flow improves long-horizon rollout error; lower is better.",
            "type": "line", "intent": "trend", "dataset": "rollout", "sourceId": rollout_source["id"],
            "source": rollout_source,
            "encodings": {
                "x": {"field": "rollout_horizon_seconds", "type": "quantitative", "label": "Horizon (s)"},
                "y": {"field": "energy__stim_balanced__mean", "type": "quantitative", "label": "Energy"},
                "color": {"field": "model_label", "type": "nominal", "label": "Model"},
            },
        },
        {
            "id": "validity_chart", "title": "Compatibility-valid fraction",
            "subtitle": "ESS-SMC yields substantially more valid source contrasts with one quarter as many retained particles.",
            "type": "bar", "intent": "comparison", "dataset": "diagnostics", "sourceId": diagnostics_source["id"],
            "source": diagnostics_source,
            "encodings": {
                "x": {"field": "method_label", "type": "nominal", "label": "Method"},
                "y": {"field": "validity_rate", "type": "quantitative", "label": "Valid fraction"},
            },
        },
        {
            "id": "stability_chart", "title": "Off-diagonal lag-matrix agreement",
            "subtitle": "Raw onset structure is modestly reproducible; onset-minus-quiet structure is not.",
            "type": "line", "intent": "trend", "dataset": "stability", "sourceId": stability_source["id"],
            "source": stability_source,
            "encodings": {
                "x": {"field": "horizon_seconds", "type": "quantitative", "label": "Horizon (s)"},
                "y": {"field": "pearson", "type": "quantitative", "label": "Pearson r"},
                "color": {"field": "series", "type": "nominal", "label": "Comparison"},
            },
        },
        {
            "id": "named_chart", "title": "Declared edge coefficients",
            "subtitle": "Positive onset coefficients often persist in matched quiet windows.",
            "type": "line", "intent": "trend", "dataset": "named", "sourceId": named_source["id"],
            "source": named_source,
            "encodings": {
                "x": {"field": "horizon_seconds", "type": "quantitative", "label": "Horizon (s)"},
                "y": {"field": "mean_coefficient", "type": "quantitative", "label": "Coefficient"},
                "color": {"field": "series", "type": "nominal", "label": "Method and phase"},
            },
        },
    ]
    tables = [{
        "id": "candidate_table", "title": "Top onset-minus-quiet edge candidates",
        "dataset": "candidates", "sourceId": candidate_source["id"],
        "source": candidate_source,
        "columns": [
            {"field": "source", "label": "Source", "format": "text"},
            {"field": "target", "label": "Target", "format": "text"},
            {"field": "horizon_seconds", "label": "Horizon (s)", "format": "number"},
            {"field": "direct_coefficient", "label": "Direct", "format": "number"},
            {"field": "progressive_coefficient", "label": "ESS-SMC", "format": "number"},
            {"field": "exploratory_score", "label": "Stability score", "format": "number"},
            {"field": "strict_stimulus_specific_pass", "label": "Strict pass", "format": "boolean"},
        ],
        "defaultSort": {"field": "exploratory_score", "direction": "desc"},
    }]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}\n\nThe improved model and sampler make lag estimation numerically better, but no stimulus-specific edge survives the full stability rule."},
        {"id": "summary", "type": "markdown", "body": "## Technical summary\n\nDropout plus mild input jitter improves held-out rollout behavior. Progressive ESS-SMC markedly improves compatibility validity. Raw onset matrices show a modest shared component, whereas subtracting matched quiet dynamics removes cross-sampler and cross-seed agreement."},
        {"id": "rollout_block", "type": "chart", "chartId": "rollout_chart"},
        {"id": "sampling", "type": "markdown", "body": "## Sampling result\n\nESS-SMC maintains particle diversity while progressively imposing the source constraint. It improves the validity of the learned-law contrast, but that Monte Carlo gain does not create biological specificity."},
        {"id": "validity_block", "type": "chart", "chartId": "validity_chart"},
        {"id": "stability", "type": "markdown", "body": "## Lag-matrix stability\n\nThe primary biological estimand is true onset minus a matched quiet pseudo-onset. It is unstable across sampler and independently trained generator seeds, so no physical-delay or receptor-action claim is supported."},
        {"id": "stability_block", "type": "chart", "chartId": "stability_chart"},
        {"id": "biology", "type": "markdown", "body": "## Biological interpretation\n\nRIB→AVB remains positive as a generic activity dependency, but quiet coefficients are comparable or larger. AWC-centered signs are estimator-sensitive. These should remain prospective reduced-form hypotheses."},
        {"id": "named_block", "type": "chart", "chartId": "named_chart"},
        {"id": "candidate_block", "type": "table", "tableId": "candidate_table"},
        {"id": "limits", "type": "markdown", "body": "## Limitations and claim boundary\n\nTwenty worms limit edgewise power and adjacent horizons are dependent. Coefficients describe observational responses of a learned conditional law; they are not identified synapses, causal interventions, receptor actions, anatomical rewiring, or physical transmission delays."},
        {"id": "next", "type": "markdown", "body": "## Next steps\n\nFreeze this corrected protocol for a prospective cohort, increase biological replication before particles, and require onset-minus-quiet replication of declared RIB→AVB and AWC→AIB hypotheses."},
    ]
    model_labels = {
        "control_tcn_flow128_natural": "Legacy control",
        "tcn_flow128_dropout15_jitter_p01": "Regularized flow",
        "tcn_mdn4_jitter_p01": "Regularized MDN",
    }
    rollout_snapshot = rollout.copy()
    rollout_snapshot["model_label"] = rollout_snapshot.model_id.map(model_labels)
    snapshot = {
        "version": 1, "status": "ready", "generatedAt": generated,
        "datasets": {
            "prediction": json.loads(prediction.to_json(orient="records")),
            "rollout": json.loads(rollout_snapshot.to_json(orient="records")),
            "diagnostics": json.loads(overall.to_json(orient="records")),
            "stability": json.loads(stability.to_json(orient="records")),
            "named": json.loads(named_plot.to_json(orient="records")),
            "candidates": json.loads(
                candidates[candidates.estimand == "onset_minus_quiet"].head(25).to_json(orient="records")
            ),
        },
    }
    manifest = {
        "version": 1, "surface": "report", "title": title,
        "description": "Aligned predictive, sampler, stability, and biological analysis of lag-response matrices.",
        "generatedAt": generated,
        "sources": [source, rollout_source, diagnostics_source, stability_source, named_source, candidate_source],
        "charts": charts,
        "tables": tables, "blocks": blocks,
    }
    artifact = {
        "surface": "report", "manifest": manifest, "snapshot": snapshot,
        "sources": [source, rollout_source, diagnostics_source, stability_source, named_source, candidate_source],
    }
    (output / "artifact.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")


def write_validation(
    runs: dict[str, AlignedRun], output: Path, candidates: pd.DataFrame,
    relationships: pd.DataFrame, named: pd.DataFrame,
) -> None:
    checks = {
        "regularized_direct_archives": len(runs["regularized_direct"].files) == 10,
        "progressive_archives": len(runs["regularized_progressive"].files) == 5,
        "legacy_direct_archives": len(runs["legacy_direct"].files) == 5,
        "all_runs_cover_20_worms": all(run.coefficient.shape[0] == 20 for run in runs.values()),
        "all_runs_have_54_neurons": all(run.coefficient.shape[-2:] == (54, 54) for run in runs.values()),
        "all_matrices_finite": all(np.isfinite(run.coefficient).all() and np.isfinite(run.raw).all() for run in runs.values()),
        "all_validity_finite": all(np.isfinite(run.validity).all() for run in runs.values()),
        "primary_delay_available": all(7 in run.delays for run in runs.values()),
        "four_horizons_aligned": all(np.array_equal(run.horizons, np.array([2, 4, 8, 16])) for run in runs.values()),
        "matrix_relationships_nonempty": len(relationships) > 0,
        "named_edge_cells_complete": len(named) == 3 * 3 * 3 * 4,
        "strict_candidate_flag_present": "strict_stimulus_specific_pass" in candidates,
        "technical_report_exists": (output / "TECHNICAL_REPORT.md").exists(),
        "artifact_exists": (output / "artifact.json").exists(),
    }
    validation = {
        "created_utc": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {
            "matrix_relationship_rows": len(relationships),
            "named_edge_rows": len(named),
            "candidate_rows": len(candidates),
            "strict_stimulus_specific_passes": int(candidates.strict_stimulus_specific_pass.sum()),
            "contextual_onset_consensus_cells": int(candidates.contextual_onset_pass.sum()),
        },
        "claim_boundary": "observational, model-relative lag response; not causal, anatomical, receptor, or physical-delay identification",
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if validation["status"] != "passed":
        raise RuntimeError(f"analysis validation failed: {checks}")


def write_checksums(output: Path) -> None:
    targets = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    rows = []
    for path in targets:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(output)}")
    (output / "checksums.sha256").write_text("\n".join(rows) + "\n")


def write_tree_checksums(root: Path) -> None:
    targets = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    rows = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
        for path in targets
    ]
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n")


def write_parent_index() -> None:
    body = """# Corrected-onset regularized lag program

This directory contains the final aligned lag-response experiment registered as E23 / CPR-17.

## Reading order

1. `analysis/TECHNICAL_REPORT.md` — answer-first scientific report.
2. `analysis/protocol.json` — frozen timing, coefficient, sampler, inference, and claim definitions.
3. `analysis/aligned_matrices.npz` and `analysis/matrix_summary_long.csv` — exact and readable matrices.
4. `analysis/validation.json` and `analysis/checksums.sha256` — analysis validation and integrity.
5. `checksums.sha256` — recursive ledger for all aligned response archives and analysis artifacts.

## Run inventory

- `direct_regularized_N256/`: 10/10 held-out fold/seed archives; delays 5, 7, and 9 frames.
- `progressive_regularized_N64/`: 5/5 held-out fold archives; primary delay 7; ESS-adaptive progressive bridge SMC.
- `direct_control_N256/`: 5/5 aligned legacy-wide-flow control archives; primary delay 7.
- `analysis/`: worm-level sampler/model/seed/timing/animal stability, named-edge and all-edge inference, figures, and report artifact.

The selected conditional model is saved in `../conditional_distribution_benchmark/rollout_stability_world_model_v2_20260828/`; the full-history architecture tournament is in `../conditional_distribution_benchmark/biological_world_model_20260828/`. The earlier incorrectly implemented jitter run is explicitly marked invalid in `../conditional_distribution_benchmark/rollout_stability_world_model_20260828/INVALIDATED.md` and is not used here.

## Result boundary

Regularization improves prediction and ESS-SMC improves compatibility, but no onset-minus-matched-quiet edge survives the frozen cross-sampler, seed, timing, worm, and multiplicity rule. These matrices are observational responses of a learned conditional law, not synapses, causal interventions, receptor actions, or physical transmission delays.
"""
    (ALIGNED_ROOT / "README.md").write_text(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze aligned regularized-flow lag responses")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = {name: load_run(name, *spec) for name, spec in RUNS.items()}
    base_neurons = runs["regularized_direct"].neurons
    if not all(np.array_equal(run.neurons, base_neurons) for run in runs.values()):
        raise RuntimeError("neuron order differs across aligned runs")

    prediction, rollout = predictive_tables()
    diagnostics = sampler_diagnostics(runs)
    relationships = matrix_relationships(runs)
    animal_stability = animal_split_stability(runs)
    named = named_edge_analysis(runs)
    edge_tests, candidates = exploratory_edges(runs)
    summary = matrix_summary(runs)

    prediction.to_csv(output / "predictive_model_comparison.csv", index=False)
    rollout.to_csv(output / "rollout_comparison.csv", index=False)
    diagnostics.to_csv(output / "sampler_diagnostics.csv", index=False)
    relationships.to_csv(output / "matrix_correlations.csv", index=False)
    animal_stability.to_csv(output / "animal_split_stability.csv", index=False)
    named.to_csv(output / "named_edges.csv", index=False)
    edge_tests.to_csv(output / "exploratory_edge_tests.csv", index=False)
    candidates.to_csv(output / "candidate_edges.csv", index=False)
    summary.to_csv(output / "matrix_summary_long.csv", index=False)
    arrays: dict[str, np.ndarray] = {
        "neurons": base_neurons,
        "phase_names": runs["regularized_direct"].phases,
        "horizon_frames": runs["regularized_direct"].horizons,
        "horizon_seconds": runs["regularized_direct"].horizon_seconds,
    }
    for name, run in runs.items():
        arrays[f"{name}__coefficient"] = run.coefficient.astype(np.float32)
        arrays[f"{name}__raw_contrast"] = run.raw.astype(np.float32)
        arrays[f"{name}__validity"] = run.validity.astype(np.float32)
        arrays[f"{name}__delay_frames"] = run.delays.astype(np.int16)
        arrays[f"{name}__delay_seconds"] = run.delay_seconds.astype(np.float32)
        arrays[f"{name}__model_seeds"] = np.asarray(sorted(run.coefficient_by_seed), dtype=np.int32)
        for seed, value in run.coefficient_by_seed.items():
            arrays[f"{name}__seed_{seed}__coefficient"] = value.astype(np.float32)
    np.savez_compressed(output / "aligned_matrices.npz", **arrays)

    figures(runs, prediction, rollout, diagnostics, relationships, named, output)
    write_protocol(runs, output)
    create_report(output, prediction, rollout, diagnostics, relationships, animal_stability, named, candidates)
    write_artifact(output, prediction, rollout, diagnostics, relationships, named, candidates)
    write_validation(runs, output, candidates, relationships, named)
    write_checksums(output)
    write_parent_index()
    write_tree_checksums(ALIGNED_ROOT)
    write_tree_checksums(STABILITY_RUN)
    write_tree_checksums(FULL_HISTORY_RUN)
    print(json.dumps({
        "status": "complete", "output": str(output),
        "strict_candidates": int(candidates.strict_stimulus_specific_pass.sum()),
        "files": len(list(output.rglob("*"))),
    }, indent=2))


if __name__ == "__main__":
    main()
