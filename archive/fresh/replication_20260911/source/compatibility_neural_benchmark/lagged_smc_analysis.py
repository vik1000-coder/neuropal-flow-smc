from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, t

from compatibility_neural_benchmark.core import causal_fill, standardize_for_checkpoint
from conditional_neural_benchmark.data import load_cohort


SCREEN_FOLDS = (0, 1, 2)
CONFIRMATION_FOLDS = (3, 4)
DIRECT_SENSORY_TARGETS = ("AWC",)


@dataclass(frozen=True)
class LaggedResponses:
    matrices: np.ndarray  # lag, worm, phase, event, horizon, target, source
    validity: np.ndarray  # lag, worm, phase, event, source
    achieved_fraction: np.ndarray  # lag, worm, phase, event, source
    ess_min: np.ndarray  # lag, worm, phase, event, source
    max_weight: np.ndarray  # lag, worm, phase, event, source
    distinct_ancestors: np.ndarray  # lag, worm, phase, event, source
    folds: np.ndarray  # worm
    source_lags: np.ndarray
    horizons: np.ndarray
    phases: tuple[str, ...]
    neurons: tuple[str, ...]
    checkpoint_by_fold: dict[int, str]
    particles: int
    seeds: tuple[int, ...]


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    usable = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[usable], dtype=np.float64)
    y = np.asarray(y[usable], dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(spearmanr(x, y).statistic)


def _mean_ci(values: np.ndarray) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, 0
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, np.nan, np.nan, int(len(values))
    half = float(
        t.ppf(0.975, len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )
    return mean, mean - half, mean + half, int(len(values))


def _fold_assignments(path: Path, cohort) -> np.ndarray:
    table = pd.read_csv(path)
    mapping = dict(zip(table.worm_id.astype(str), table.outer_fold.astype(int)))
    result = np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)
    if set(result.tolist()) != set(range(5)):
        raise RuntimeError("expected immutable five-fold assignment")
    return result


def load_responses(run_dir: Path, folds: np.ndarray) -> LaggedResponses:
    paths = sorted((run_dir / "responses").glob("*__lagged_smc__ell*__N*__f*__s*.npz"))
    if not paths:
        raise FileNotFoundError(f"no lagged SMC archives under {run_dir}")

    records: dict[tuple[int, int, int], dict[str, np.ndarray | int | str]] = {}
    lags: set[int] = set()
    seeds: set[int] = set()
    horizons = phases = neurons = None
    particles: int | None = None
    checkpoint_by_fold: dict[int, str] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["status"].item()) != "complete":
                raise RuntimeError(f"incomplete archive: {path}")
            lag = int(data["source_lag_frames"])
            fold = int(data["fold"])
            seed = int(data["seed"])
            key = (lag, fold, seed)
            if key in records:
                raise RuntimeError(f"duplicate response cell {key}")
            current_horizons = data["horizon_frames"].astype(int)
            current_phases = data["phase_names"].astype(str)
            current_neurons = data["neurons"].astype(str)
            current_particles = int(data["n_particles"])
            if horizons is None:
                horizons = current_horizons
                phases = current_phases
                neurons = current_neurons
                particles = current_particles
            elif not (
                np.array_equal(horizons, current_horizons)
                and np.array_equal(phases, current_phases)
                and np.array_equal(neurons, current_neurons)
                and particles == current_particles
            ):
                raise RuntimeError("response archive schema mismatch")
            response = data["response_cumulative_mean"].astype(np.float64)
            achieved_gap = data["diagnostic_achieved_gap"].astype(np.float64)
            target_gap = data["diagnostic_target_gap"].astype(np.float64)
            normalized = response / np.maximum(achieved_gap[..., None, None], 0.10)
            # worm, phase, event, horizon, target, source
            matrix = normalized.transpose(0, 1, 2, 4, 5, 3)
            valid = data["diagnostic_valid"].astype(bool)
            # Invalid source columns are not silently treated as estimated effects.
            matrix = np.where(valid[..., None, None, :], matrix, 0.0)
            diagonal = np.arange(len(current_neurons))
            matrix[..., diagonal, diagonal] = 0.0
            record = {
                "worms": data["worm_indices"].astype(int),
                "matrix": matrix,
                "valid": valid.astype(np.float64),
                "achieved_fraction": achieved_gap / np.maximum(target_gap, 1e-8),
                "ess_min": np.minimum(
                    data["diagnostic_ess_low"].astype(np.float64),
                    data["diagnostic_ess_high"].astype(np.float64),
                ),
                "max_weight": np.maximum(
                    data["diagnostic_max_weight_low"].astype(np.float64),
                    data["diagnostic_max_weight_high"].astype(np.float64),
                ),
                "distinct_ancestors": np.minimum(
                    data["diagnostic_distinct_ancestors_low"].astype(np.float64),
                    data["diagnostic_distinct_ancestors_high"].astype(np.float64),
                ),
                "checkpoint": str(data["checkpoint"].item()),
            }
            records[key] = record
            lags.add(lag)
            seeds.add(seed)
            checkpoint_by_fold.setdefault(fold, str(record["checkpoint"]))

    source_lags = np.asarray(sorted(lags), dtype=int)
    seed_values = tuple(sorted(seeds))
    n_worms = len(folds)
    shape_matrix = (
        len(source_lags),
        n_worms,
        len(phases),
        3,
        len(horizons),
        len(neurons),
        len(neurons),
    )
    shape_diag = shape_matrix[:4] + (shape_matrix[-1],)
    stores = {
        "matrix": np.full(shape_matrix, np.nan, dtype=np.float64),
        "valid": np.full(shape_diag, np.nan, dtype=np.float64),
        "achieved_fraction": np.full(shape_diag, np.nan, dtype=np.float64),
        "ess_min": np.full(shape_diag, np.nan, dtype=np.float64),
        "max_weight": np.full(shape_diag, np.nan, dtype=np.float64),
        "distinct_ancestors": np.full(shape_diag, np.nan, dtype=np.float64),
    }
    for lag_position, lag in enumerate(source_lags):
        for fold in sorted(set(folds.tolist())):
            selected = [records[(lag, fold, seed)] for seed in seed_values if (lag, fold, seed) in records]
            if len(selected) != len(seed_values):
                raise RuntimeError(f"missing seed for lag={lag}, fold={fold}")
            worms = np.asarray(selected[0]["worms"], dtype=int)
            for record in selected[1:]:
                if not np.array_equal(worms, record["worms"]):
                    raise RuntimeError("seed archives disagree on held-out worms")
            for name in stores:
                array = np.mean(np.stack([np.asarray(record[name]) for record in selected]), axis=0)
                stores[name][lag_position, worms] = array
    for name, array in stores.items():
        if np.isnan(array).any():
            raise RuntimeError(f"incomplete worm coverage in {name}")
    return LaggedResponses(
        matrices=stores["matrix"],
        validity=stores["valid"],
        achieved_fraction=stores["achieved_fraction"],
        ess_min=stores["ess_min"],
        max_weight=stores["max_weight"],
        distinct_ancestors=stores["distinct_ancestors"],
        folds=np.asarray(folds, dtype=int),
        source_lags=source_lags,
        horizons=np.asarray(horizons, dtype=int),
        phases=tuple(np.asarray(phases).astype(str).tolist()),
        neurons=tuple(np.asarray(neurons).astype(str).tolist()),
        checkpoint_by_fold=checkpoint_by_fold,
        particles=int(particles),
        seeds=seed_values,
    )


def standardized_traces(cohort, artifact: LaggedResponses) -> tuple[np.ndarray, ...]:
    traces: list[np.ndarray] = []
    checkpoints: dict[int, dict] = {}
    for worm, fold in enumerate(artifact.folds):
        fold = int(fold)
        if fold not in checkpoints:
            checkpoints[fold] = torch.load(
                artifact.checkpoint_by_fold[fold], map_location="cpu", weights_only=False
            )
        traces.append(
            causal_fill(standardize_for_checkpoint(cohort.traces[worm], checkpoints[fold]))
        )
    return tuple(traces)


def build_vectors(
    traces: tuple[np.ndarray, ...],
    cut_times: np.ndarray,
    source_lag: int,
    horizons: np.ndarray,
    source_window: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source, current-state and future-response vectors.

    Each response is centered against a local, strictly pre-window baseline.
    Shapes are worm, phase, event, [horizon], neuron.
    """
    n_worms, n_phases, n_events = cut_times.shape
    d = traces[0].shape[1]
    source = np.empty((n_worms, n_phases, n_events, d), dtype=np.float64)
    current = np.empty_like(source)
    target = np.empty(
        (n_worms, n_phases, n_events, len(horizons), d), dtype=np.float64
    )
    for worm, trace in enumerate(traces):
        for phase in range(n_phases):
            for event in range(n_events):
                cut = int(cut_times[worm, phase, event])
                source_hi = cut - source_lag + 1
                source_lo = source_hi - source_window
                source_base_lo = source_lo - source_window
                current_hi = cut + 1
                current_lo = current_hi - source_window
                current_base_lo = current_lo - source_window
                if source_base_lo < 0 or cut + int(horizons.max()) >= len(trace):
                    raise RuntimeError("episode vector exceeds trace boundary")
                source_baseline = trace[source_base_lo:source_lo].mean(axis=0)
                current_baseline = trace[current_base_lo:current_lo].mean(axis=0)
                source[worm, phase, event] = (
                    trace[source_lo:source_hi].mean(axis=0) - source_baseline
                )
                current[worm, phase, event] = (
                    trace[current_lo:current_hi].mean(axis=0) - current_baseline
                )
                for horizon_position, horizon in enumerate(horizons):
                    target[worm, phase, event, horizon_position] = (
                        trace[cut + 1 : cut + 1 + int(horizon)].mean(axis=0)
                        - current_baseline
                    )
    return source, current, target


def _residualized(
    array: np.ndarray, evaluation_worm: int, training_worms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-worm residuals for train and one genuinely held-out worm."""
    heldout = array[evaluation_worm] - np.mean(array[training_worms], axis=0)
    training = []
    for worm in training_worms:
        others = training_worms[training_worms != worm]
        training.append(array[worm] - np.mean(array[others], axis=0))
    return np.stack(training), heldout


def _fit_scalar_forecast(
    source: np.ndarray,
    current: np.ndarray,
    propagated: np.ndarray,
    target: np.ndarray,
    training_worms: np.ndarray,
    evaluation_worm: int,
    target_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src_train, src_test = _residualized(source, evaluation_worm, training_worms)
    cur_train, cur_test = _residualized(current, evaluation_worm, training_worms)
    pro_train, pro_test = _residualized(propagated, evaluation_worm, training_worms)
    tar_train, tar_test = _residualized(target, evaluation_worm, training_worms)

    src_fit = src_train[..., target_mask].reshape(-1)
    cur_fit = cur_train[..., target_mask].reshape(-1)
    pro_fit = pro_train[..., target_mask].reshape(-1)
    tar_fit = tar_train[..., target_mask].reshape(-1)
    base_design = np.column_stack([np.ones(len(tar_fit)), cur_fit, src_fit])
    full_design = np.column_stack([base_design, pro_fit])
    beta_base, *_ = np.linalg.lstsq(base_design, tar_fit, rcond=None)
    beta_full, *_ = np.linalg.lstsq(full_design, tar_fit, rcond=None)

    base_test = (
        beta_base[0]
        + beta_base[1] * cur_test[..., target_mask]
        + beta_base[2] * src_test[..., target_mask]
    )
    full_test = (
        beta_full[0]
        + beta_full[1] * cur_test[..., target_mask]
        + beta_full[2] * src_test[..., target_mask]
        + beta_full[3] * pro_test[..., target_mask]
    )
    return base_test, full_test, tar_test[..., target_mask], beta_full


def evaluate_neural_prediction(
    artifact: LaggedResponses,
    vectors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    target_mask = ~np.isin(np.asarray(artifact.neurons), DIRECT_SENSORY_TARGETS)
    rows: list[dict] = []
    for split, allowed_folds in (
        ("screen", SCREEN_FOLDS),
        ("confirmation", CONFIRMATION_FOLDS),
    ):
        evaluation_worms = np.flatnonzero(np.isin(artifact.folds, allowed_folds))
        # Confirmation coefficients are frozen from screen worms. Screen uses LOO.
        fixed_training = np.flatnonzero(np.isin(artifact.folds, SCREEN_FOLDS))
        for lag_position, lag in enumerate(artifact.source_lags):
            source, current, target = vectors[int(lag)]
            for phase_position, phase in enumerate(artifact.phases):
                for horizon_position, horizon in enumerate(artifact.horizons):
                    matrices = artifact.matrices[
                        lag_position, :, phase_position, :, horizon_position
                    ]
                    propagated = np.stack(
                        [
                            np.stack(
                                [matrices[worm, event] @ source[worm, phase_position, event]
                                 for event in range(source.shape[2])]
                            )
                            for worm in range(len(artifact.folds))
                        ]
                    )
                    phase_source = source[:, phase_position]
                    phase_current = current[:, phase_position]
                    phase_target = target[:, phase_position, :, horizon_position]
                    for worm in evaluation_worms:
                        training = (
                            fixed_training[fixed_training != worm]
                            if split == "screen"
                            else fixed_training
                        )
                        base, full, outcome, coefficients = _fit_scalar_forecast(
                            phase_source,
                            phase_current,
                            propagated,
                            phase_target,
                            training,
                            int(worm),
                            target_mask,
                        )
                        for event in range(phase_source.shape[1]):
                            base_rho = _safe_spearman(base[event], outcome[event])
                            full_rho = _safe_spearman(full[event], outcome[event])
                            matrix_rho = _safe_spearman(
                                propagated[worm, event, target_mask], outcome[event]
                            )
                            rows.append(
                                {
                                    "split": split,
                                    "source_lag_frames": int(lag),
                                    "source_lag_seconds": float(lag / 4.0),
                                    "phase": phase,
                                    "horizon_frames": int(horizon),
                                    "horizon_seconds": float(horizon / 4.0),
                                    "worm": int(worm),
                                    "fold": int(artifact.folds[worm]),
                                    "event": int(event),
                                    "n_targets": int(target_mask.sum()),
                                    "excluded_targets": ";".join(DIRECT_SENSORY_TARGETS),
                                    "base_spearman": base_rho,
                                    "full_spearman": full_rho,
                                    "incremental_gain": full_rho - base_rho,
                                    "matrix_only_spearman": matrix_rho,
                                    "beta_intercept": float(coefficients[0]),
                                    "beta_current": float(coefficients[1]),
                                    "beta_source": float(coefficients[2]),
                                    "beta_matrix": float(coefficients[3]),
                                }
                            )
    return pd.DataFrame(rows)


def summarize_prediction(events: pd.DataFrame) -> pd.DataFrame:
    group = [
        "split",
        "source_lag_frames",
        "source_lag_seconds",
        "phase",
        "horizon_frames",
        "horizon_seconds",
    ]
    worm = events.groupby(group + ["worm"], as_index=False).agg(
        base_spearman=("base_spearman", "mean"),
        full_spearman=("full_spearman", "mean"),
        incremental_gain=("incremental_gain", "mean"),
        matrix_only_spearman=("matrix_only_spearman", "mean"),
        beta_matrix=("beta_matrix", "mean"),
    )
    rows: list[dict] = []
    for keys, frame in worm.groupby(group, sort=False):
        row = dict(zip(group, keys))
        for metric in (
            "base_spearman",
            "full_spearman",
            "incremental_gain",
            "matrix_only_spearman",
            "beta_matrix",
        ):
            mean, low, high, n = _mean_ci(frame[metric].to_numpy(float))
            row[f"mean_{metric}"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"n_worms_{metric}"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostic_summary(artifact: LaggedResponses) -> pd.DataFrame:
    rows: list[dict] = []
    for lag_position, lag in enumerate(artifact.source_lags):
        for phase_position, phase in enumerate(artifact.phases):
            for split, fold_set in (
                ("screen", SCREEN_FOLDS),
                ("confirmation", CONFIRMATION_FOLDS),
                ("all", tuple(range(5))),
            ):
                worms = np.flatnonzero(np.isin(artifact.folds, fold_set))
                selection = (lag_position, worms, phase_position)
                rows.append(
                    {
                        "split": split,
                        "source_lag_frames": int(lag),
                        "source_lag_seconds": float(lag / 4.0),
                        "phase": phase,
                        "n_particles": artifact.particles,
                        "n_seeds": len(artifact.seeds),
                        "valid_fraction": float(np.mean(artifact.validity[selection])),
                        "median_achieved_fraction": float(
                            np.median(artifact.achieved_fraction[selection])
                        ),
                        "median_min_ess": float(np.median(artifact.ess_min[selection])),
                        "p95_max_weight": float(
                            np.quantile(artifact.max_weight[selection], 0.95)
                        ),
                        "median_distinct_ancestors": float(
                            np.median(artifact.distinct_ancestors[selection])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def select_cell(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> dict:
    screen = summary[(summary.split == "screen") & (summary.phase == "onset")].copy()
    validity = diagnostics[
        (diagnostics.split == "screen") & (diagnostics.phase == "onset")
    ][["source_lag_frames", "valid_fraction"]]
    screen = screen.merge(validity, on="source_lag_frames", how="left")
    admissible = screen[screen.valid_fraction >= 0.50]
    pool = admissible if len(admissible) else screen
    winner = pool.sort_values(
        ["mean_incremental_gain", "mean_matrix_only_spearman"], ascending=False
    ).iloc[0]
    quiet = summary[
        (summary.split == "screen")
        & (summary.phase == "baseline")
        & (summary.source_lag_frames == winner.source_lag_frames)
        & (summary.horizon_frames == winner.horizon_frames)
    ].iloc[0]
    promotion = bool(
        winner.valid_fraction >= 0.50
        and winner.mean_incremental_gain > 0
        and winner.incremental_gain_ci_low > 0
        and winner.mean_incremental_gain > quiet.mean_incremental_gain
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "folds 0-2 only",
        "confirmation_split": "folds 3-4 untouched by selection",
        "source_lag_frames": int(winner.source_lag_frames),
        "source_lag_seconds": float(winner.source_lag_seconds),
        "horizon_frames": int(winner.horizon_frames),
        "horizon_seconds": float(winner.horizon_seconds),
        "screen_mean_incremental_spearman_gain": float(winner.mean_incremental_gain),
        "screen_gain_ci_low": float(winner.incremental_gain_ci_low),
        "screen_gain_ci_high": float(winner.incremental_gain_ci_high),
        "screen_matrix_only_spearman": float(winner.mean_matrix_only_spearman),
        "quiet_mean_incremental_spearman_gain": float(quiet.mean_incremental_gain),
        "screen_valid_fraction": float(winner.valid_fraction),
        "candidate_admissibility": "valid fraction >= 0.50",
        "strict_promotion_gate": (
            "valid fraction >= 0.50; onset incremental-gain 95% CI > 0; "
            "onset gain exceeds quiet pseudo-onset gain"
        ),
        "passes_strict_promotion_gate": promotion,
        "selection_metric": (
            "worm-residual, AWC-excluded incremental Spearman gain over a "
            "cross-fitted current-state + source-history forecast"
        ),
        "external_references_consulted": False,
    }


def load_cut_times(run_dir: Path, artifact: LaggedResponses) -> np.ndarray:
    paths = sorted((run_dir / "responses").glob("*__lagged_smc__ell*__N*__f*__s*.npz"))
    result = np.full((len(artifact.folds), len(artifact.phases), 3), -1, dtype=int)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            worms = data["worm_indices"].astype(int)
            current = data["cut_times"].astype(int)
            for position, worm in enumerate(worms):
                if np.any(result[worm] >= 0) and not np.array_equal(result[worm], current[position]):
                    raise RuntimeError("cut times differ across response archives")
                result[worm] = current[position]
    if np.any(result < 0):
        raise RuntimeError("cut times do not cover all worms")
    return result


def write_checksums(output: Path) -> None:
    paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in paths]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n")


def run(run_dir: Path, output: Path, fold_file: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    folds = _fold_assignments(fold_file, cohort)
    artifact = load_responses(run_dir, folds)
    if tuple(cohort.neurons) != artifact.neurons:
        raise RuntimeError("cohort and response neuron order differ")
    traces = standardized_traces(cohort, artifact)
    cut_times = load_cut_times(run_dir, artifact)
    vectors = {
        int(lag): build_vectors(traces, cut_times, int(lag), artifact.horizons)
        for lag in artifact.source_lags
    }
    events = evaluate_neural_prediction(artifact, vectors)
    summary = summarize_prediction(events)
    diagnostics = diagnostic_summary(artifact)
    selection = select_cell(summary, diagnostics)
    events.to_csv(output / "neural_prediction_events.csv", index=False)
    summary.to_csv(output / "neural_prediction_summary.csv", index=False)
    diagnostics.to_csv(output / "smc_diagnostics.csv", index=False)
    (output / "neural_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_run": str(run_dir.resolve()),
        "particles": artifact.particles,
        "seeds": list(artifact.seeds),
        "screen_folds": list(SCREEN_FOLDS),
        "confirmation_folds": list(CONFIRMATION_FOLDS),
        "direct_sensory_target_sensitivity": list(DIRECT_SENSORY_TARGETS),
        "selection_uses_external_atlases": False,
        "matrix_rule": "cumulative response divided by achieved source gap; invalid source columns zeroed; diagonal zeroed",
        "prediction_rule": "cross-fitted scalar current + source forecast versus same forecast plus M(source)",
        "residual_rule": "leave-one-worm residual within event and phase",
        "claim_boundary": "model-relative predictive dynamics, not anatomy, causality, or physical delay",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    validation = {
        "status": "pass",
        "n_event_rows": int(len(events)),
        "n_summary_rows": int(len(summary)),
        "n_diagnostic_rows": int(len(diagnostics)),
        "all_finite_gains": bool(np.isfinite(events.incremental_gain).all()),
        "selected_cell_present": bool(selection),
        "external_references_consulted": False,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    write_checksums(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=Path(
            "results/conditional_distribution_benchmark/atlas_blind_world_model_20260827/fold_assignments.csv"
        ),
    )
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.output_dir.resolve(), args.fold_file.resolve())


if __name__ == "__main__":
    main()
