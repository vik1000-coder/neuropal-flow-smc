from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from compatibility_neural_benchmark.core import causal_fill
from conditional_neural_benchmark.data import (
    Cohort,
    FoldScaler,
    _stimulus_features,
    _stimulus_mask,
    choose_evaluation_indices,
    load_cohort,
)
from conditional_neural_benchmark.distributed_lag_dynamics import (
    reconstruct_kernel,
    smooth_lag_basis,
)
from conditional_neural_benchmark.metrics import metric_rows
from conditional_neural_benchmark.runner import _split_indices


@dataclass(frozen=True)
class Design:
    neural: np.ndarray
    common: np.ndarray
    innovation_target: np.ndarray
    observed_target: np.ndarray
    observed_previous: np.ndarray
    worm: np.ndarray
    time: np.ndarray
    state: np.ndarray


@dataclass
class StructuredFit:
    base_models: list[Ridge]
    common_to_neural: Ridge
    cross_model: Ridge
    scale_base_models: list[Ridge]
    scale_cross_model: Ridge
    neural_mean: np.ndarray
    neural_scale: np.ndarray
    common_mean: np.ndarray
    common_scale: np.ndarray
    mean_kernel_basis: np.ndarray
    scale_kernel_basis: np.ndarray

def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_folds(path: Path, cohort: Cohort) -> np.ndarray:
    frame = pd.read_csv(path)
    mapping = dict(zip(frame.worm_id.astype(str), frame.outer_fold.astype(int)))
    missing = [worm for worm in cohort.worm_ids if worm not in mapping]
    if missing:
        raise RuntimeError(f"fold assignments missing worms: {missing}")
    return np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)


def fit_calcium_decay(
    traces: list[np.ndarray], schedules: list, *, minimum: float = 0.50, maximum: float = 0.995
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a quiet-period AR(1) observation decay independently per neuron."""
    previous: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for trace, schedule in zip(traces, schedules):
        active = _stimulus_mask(len(trace), schedule) > 0.5
        quiet_pair = ~(active[1:] | active[:-1])
        previous.append(trace[:-1][quiet_pair])
        current.append(trace[1:][quiet_pair])
    x = np.concatenate(previous).astype(np.float64)
    y = np.concatenate(current).astype(np.float64)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    centered_x = x - x_mean
    centered_y = y - y_mean
    denominator = np.square(centered_x).sum(axis=0)
    alpha = np.divide(
        (centered_x * centered_y).sum(axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-10,
    )
    alpha = np.clip(alpha, minimum, maximum)
    intercept = y_mean - alpha * x_mean
    return alpha.astype(np.float32), intercept.astype(np.float32)


def innovation_trace(
    observed: np.ndarray, alpha: np.ndarray, intercept: np.ndarray
) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float32)
    result = np.zeros_like(observed)
    result[1:] = observed[1:] - intercept - alpha * observed[:-1]
    return result


def state_for_frames(length: int, schedule) -> np.ndarray:
    result = np.full(length, "quiet", dtype="U8")
    fps = schedule.analysis_fps
    for start_s, end_s in schedule.event_intervals_seconds:
        start = int(round(start_s * fps))
        end = int(round(end_s * fps))
        result[start:min(end, length)] = "active"
        result[start:min(start + int(round(2.0 * fps)), end, length)] = "onset"
    return result


def fit_projection(traces: list[np.ndarray], components: int, seed: int) -> PCA:
    pooled = np.concatenate(traces, axis=0)
    return PCA(
        n_components=min(components, pooled.shape[1]),
        svd_solver="randomized",
        random_state=seed,
    ).fit(pooled)


def make_design(
    traces: list[np.ndarray],
    innovations: list[np.ndarray],
    schedules: list,
    worm_indices: np.ndarray,
    basis: np.ndarray,
    pca: PCA,
) -> Design:
    max_lag, n_basis = basis.shape
    chronological = basis[::-1]
    neural_rows: list[np.ndarray] = []
    common_rows: list[np.ndarray] = []
    innovation_targets: list[np.ndarray] = []
    observed_targets: list[np.ndarray] = []
    observed_previous: list[np.ndarray] = []
    worm_rows: list[np.ndarray] = []
    time_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    for trace, innovation, schedule, worm in zip(
        traces, innovations, schedules, worm_indices
    ):
        stimulus = _stimulus_features(
            len(trace), schedule, "binary_any_stimulus"
        )[:, 0]
        states = state_for_frames(len(trace), schedule)
        neural_windows = np.lib.stride_tricks.sliding_window_view(
            innovation, max_lag, axis=0
        )[:-1].transpose(0, 2, 1)
        observed_windows = np.lib.stride_tricks.sliding_window_view(
            trace, max_lag, axis=0
        )[:-1].transpose(0, 2, 1)
        stimulus_windows = np.lib.stride_tricks.sliding_window_view(
            stimulus, max_lag
        )[:-1]
        neural_basis = np.einsum(
            "nld,lq->ndq", neural_windows, chronological, optimize=True
        ).reshape(len(neural_windows), -1)
        global_history = observed_windows @ pca.components_.T.astype(np.float32)
        global_basis = np.einsum(
            "nlg,lq->ngq", global_history, chronological, optimize=True
        ).reshape(len(neural_windows), -1)
        stimulus_basis = stimulus_windows @ chronological
        stimulus_changes = np.abs(np.diff(stimulus_windows, axis=1)).sum(axis=1)
        common = np.concatenate(
            [
                global_basis,
                stimulus_basis,
                stimulus_windows[:, -1, None],
                stimulus_changes[:, None],
            ],
            axis=1,
        )
        target_times = np.arange(max_lag, len(trace), dtype=np.int32)
        neural_rows.append(neural_basis.astype(np.float32))
        common_rows.append(common.astype(np.float32))
        innovation_targets.append(innovation[max_lag:])
        observed_targets.append(trace[max_lag:])
        observed_previous.append(trace[max_lag - 1 : -1])
        worm_rows.append(np.full(len(target_times), int(worm), dtype=np.int16))
        time_rows.append(target_times)
        state_rows.append(states[max_lag:])
    return Design(
        neural=np.concatenate(neural_rows),
        common=np.concatenate(common_rows),
        innovation_target=np.concatenate(innovation_targets),
        observed_target=np.concatenate(observed_targets),
        observed_previous=np.concatenate(observed_previous),
        worm=np.concatenate(worm_rows),
        time=np.concatenate(time_rows),
        state=np.concatenate(state_rows),
    )


def standardize_design(
    train: Design, validation: Design, test: Design
) -> tuple[Design, Design, Design, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    neural_mean = train.neural.mean(axis=0, dtype=np.float64)
    neural_scale = train.neural.std(axis=0, dtype=np.float64)
    neural_scale = np.where(neural_scale > 1e-6, neural_scale, 1.0)
    common_mean = train.common.mean(axis=0, dtype=np.float64)
    common_scale = train.common.std(axis=0, dtype=np.float64)
    common_scale = np.where(common_scale > 1e-6, common_scale, 1.0)

    def transform(value: Design) -> Design:
        return Design(
            neural=((value.neural - neural_mean) / neural_scale).astype(np.float32),
            common=((value.common - common_mean) / common_scale).astype(np.float32),
            innovation_target=value.innovation_target,
            observed_target=value.observed_target,
            observed_previous=value.observed_previous,
            worm=value.worm,
            time=value.time,
            state=value.state,
        )

    return (
        transform(train),
        transform(validation),
        transform(test),
        neural_mean.astype(np.float32),
        neural_scale.astype(np.float32),
        common_mean.astype(np.float32),
        common_scale.astype(np.float32),
    )


def _ridge(alpha: float, *, intercept: bool = True) -> Ridge:
    return Ridge(
        alpha=float(alpha),
        fit_intercept=intercept,
        solver="lsqr",
        tol=1e-5,
        max_iter=4000,
    )


def base_features(design: Design, target: int, n_basis: int) -> np.ndarray:
    self_slice = slice(target * n_basis, (target + 1) * n_basis)
    return np.concatenate([design.common, design.neural[:, self_slice]], axis=1)


def fit_structured(
    train: Design,
    *,
    n_neurons: int,
    n_basis: int,
    base_alpha: float,
    cross_alpha: float,
) -> StructuredFit:
    base_models: list[Ridge] = []
    base_prediction = np.empty_like(train.innovation_target)
    for target in range(n_neurons):
        model = _ridge(base_alpha).fit(
            base_features(train, target, n_basis),
            train.innovation_target[:, target],
        )
        base_models.append(model)
        base_prediction[:, target] = model.predict(base_features(train, target, n_basis))
    common_to_neural = _ridge(base_alpha).fit(train.common, train.neural)
    neural_residual = train.neural - common_to_neural.predict(train.common)
    mean_residual = train.innovation_target - base_prediction
    cross_model = _ridge(cross_alpha, intercept=False).fit(neural_residual, mean_residual)
    mean_basis = cross_model.coef_.reshape(n_neurons, n_neurons, n_basis)
    for target in range(n_neurons):
        mean_basis[target, target] = 0.0
    full_mean = base_prediction + neural_residual @ mean_basis.reshape(n_neurons, -1).T

    squared = np.square(train.innovation_target - full_mean)
    floor = np.maximum(np.quantile(squared, 0.10, axis=0), 1e-4)
    log_squared = np.log(squared + floor)
    scale_base_models: list[Ridge] = []
    scale_base_prediction = np.empty_like(log_squared)
    for target in range(n_neurons):
        model = _ridge(base_alpha).fit(
            base_features(train, target, n_basis), log_squared[:, target]
        )
        scale_base_models.append(model)
        scale_base_prediction[:, target] = model.predict(
            base_features(train, target, n_basis)
        )
    scale_cross_model = _ridge(cross_alpha, intercept=False).fit(
        neural_residual, log_squared - scale_base_prediction
    )
    scale_basis = scale_cross_model.coef_.reshape(n_neurons, n_neurons, n_basis)
    for target in range(n_neurons):
        scale_basis[target, target] = 0.0
    return StructuredFit(
        base_models=base_models,
        common_to_neural=common_to_neural,
        cross_model=cross_model,
        scale_base_models=scale_base_models,
        scale_cross_model=scale_cross_model,
        neural_mean=np.empty(0),
        neural_scale=np.empty(0),
        common_mean=np.empty(0),
        common_scale=np.empty(0),
        mean_kernel_basis=mean_basis.astype(np.float32),
        scale_kernel_basis=scale_basis.astype(np.float32),
    )


def predict_structured(
    fitted: StructuredFit, design: Design, *, n_neurons: int, n_basis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_mean = np.column_stack(
        [
            fitted.base_models[target].predict(base_features(design, target, n_basis))
            for target in range(n_neurons)
        ]
    )
    neural_residual = design.neural - fitted.common_to_neural.predict(design.common)
    cross_mean = neural_residual @ fitted.mean_kernel_basis.reshape(n_neurons, -1).T
    full_mean = base_mean + cross_mean
    base_logvar = np.column_stack(
        [
            fitted.scale_base_models[target].predict(
                base_features(design, target, n_basis)
            )
            for target in range(n_neurons)
        ]
    )
    cross_logvar = neural_residual @ fitted.scale_kernel_basis.reshape(n_neurons, -1).T
    return base_mean, full_mean, base_logvar, base_logvar + cross_logvar


def calibrate_logvar(
    target: np.ndarray, mean: np.ndarray, logvar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    raw_variance = np.exp(np.clip(logvar, -10.0, 5.0))
    factor = np.mean(np.square(target - mean) / raw_variance, axis=0)
    offset = np.log(np.maximum(factor, 1e-4))
    return logvar + offset, offset.astype(np.float32)


def gaussian_nll(target: np.ndarray, mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    logvar = np.clip(logvar, -10.0, 5.0)
    return 0.5 * (
        np.log(2.0 * np.pi) + logvar + np.square(target - mean) / np.exp(logvar)
    )


def bootstrap_ci(values: np.ndarray, *, seed: int, draws: int = 4000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(sampled, [0.025, 0.975]))


def run_fold(
    cohort: Cohort,
    folds: np.ndarray,
    fold: int,
    basis: np.ndarray,
    *,
    base_alpha: float,
    cross_alpha: float,
    cross_alpha_grid: tuple[float, ...] | None,
    samples: int,
    eval_rows: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    train_idx, validation_idx, test_idx = _split_indices(folds, fold)
    scaler = FoldScaler.fit(cohort.traces[int(i)] for i in train_idx)
    standardized = [
        causal_fill(scaler.transform(cohort.traces[int(i)]))
        for i in range(cohort.n_worms)
    ]
    alpha, intercept = fit_calcium_decay(
        [standardized[int(i)] for i in train_idx],
        [cohort.stimulus_schedules[int(i)] for i in train_idx],
    )
    innovations = [innovation_trace(trace, alpha, intercept) for trace in standardized]
    pca = fit_projection([standardized[int(i)] for i in train_idx], components=4, seed=1701 + fold)

    def design(indices: np.ndarray) -> Design:
        return make_design(
            [standardized[int(i)] for i in indices],
            [innovations[int(i)] for i in indices],
            [cohort.stimulus_schedules[int(i)] for i in indices],
            indices,
            basis,
            pca,
        )

    train, validation, test = design(train_idx), design(validation_idx), design(test_idx)
    train, validation, test, neural_mean, neural_scale, common_mean, common_scale = standardize_design(
        train, validation, test
    )
    tuning_rows: list[dict[str, float]] = []
    fitted_by_alpha: dict[float, StructuredFit] = {}
    alpha_candidates = cross_alpha_grid or (float(cross_alpha),)
    for candidate_alpha in alpha_candidates:
        candidate = fit_structured(
            train,
            n_neurons=cohort.n_neurons,
            n_basis=basis.shape[1],
            base_alpha=base_alpha,
            cross_alpha=float(candidate_alpha),
        )
        val_candidate_base, val_candidate_full, _, val_candidate_lv = predict_structured(
            candidate,
            validation,
            n_neurons=cohort.n_neurons,
            n_basis=basis.shape[1],
        )
        calibrated_lv, _ = calibrate_logvar(
            validation.innovation_target, val_candidate_full, val_candidate_lv
        )
        tuning_rows.append(
            {
                "cross_alpha": float(candidate_alpha),
                "validation_nll": float(
                    gaussian_nll(
                        validation.innovation_target,
                        val_candidate_full,
                        calibrated_lv,
                    ).mean()
                ),
                "validation_rmse": float(
                    np.sqrt(
                        np.square(
                            validation.innovation_target - val_candidate_full
                        ).mean()
                    )
                ),
                "validation_base_rmse": float(
                    np.sqrt(
                        np.square(
                            validation.innovation_target - val_candidate_base
                        ).mean()
                    )
                ),
            }
        )
        fitted_by_alpha[float(candidate_alpha)] = candidate
    tuning_frame = pd.DataFrame(tuning_rows).sort_values(
        ["validation_nll", "cross_alpha"], ascending=[True, False]
    )
    selected_cross_alpha = float(tuning_frame.iloc[0].cross_alpha)
    fitted = fitted_by_alpha[selected_cross_alpha]
    fitted.neural_mean = neural_mean
    fitted.neural_scale = neural_scale
    fitted.common_mean = common_mean
    fitted.common_scale = common_scale
    val_base_mean, val_full_mean, val_base_lv, val_full_lv = predict_structured(
        fitted, validation, n_neurons=cohort.n_neurons, n_basis=basis.shape[1]
    )
    val_base_lv, base_offset = calibrate_logvar(
        validation.innovation_target, val_base_mean, val_base_lv
    )
    val_full_lv, full_offset = calibrate_logvar(
        validation.innovation_target, val_full_mean, val_full_lv
    )
    test_base_mean, test_full_mean, test_base_lv, test_full_lv = predict_structured(
        fitted, test, n_neurons=cohort.n_neurons, n_basis=basis.shape[1]
    )
    test_base_lv = test_base_lv + base_offset
    test_full_lv = test_full_lv + full_offset
    # Isolate scale usefulness while holding the full conditional mean fixed.
    full_mean_base_lv, full_mean_base_offset = calibrate_logvar(
        validation.innovation_target, val_full_mean, val_base_lv - base_offset
    )
    del full_mean_base_lv
    test_full_mean_base_lv = test_base_lv - base_offset + full_mean_base_offset

    observed_base_mean = intercept + alpha * test.observed_previous + test_base_mean
    observed_full_mean = intercept + alpha * test.observed_previous + test_full_mean
    base_nll = gaussian_nll(test.observed_target, observed_base_mean, test_base_lv)
    full_nll = gaussian_nll(test.observed_target, observed_full_mean, test_full_lv)
    full_mean_base_scale_nll = gaussian_nll(
        test.observed_target, observed_full_mean, test_full_mean_base_lv
    )
    rng = np.random.default_rng(20260828 + fold)
    index = choose_evaluation_indices(test.state, eval_rows, 20260828 + fold)
    base_draws = observed_base_mean[index, None] + np.exp(
        0.5 * np.clip(test_base_lv[index, None], -10.0, 5.0)
    ) * rng.normal(size=(len(index), samples, cohort.n_neurons))
    # Reuse the identical standard-normal draws for the base/full comparison.
    noise = (base_draws - observed_base_mean[index, None]) / np.exp(
        0.5 * np.clip(test_base_lv[index, None], -10.0, 5.0)
    )
    full_draws = observed_full_mean[index, None] + np.exp(
        0.5 * np.clip(test_full_lv[index, None], -10.0, 5.0)
    ) * noise
    base_metrics = metric_rows(base_draws, test.observed_target[index], 1701 + fold)
    full_metrics = metric_rows(full_draws, test.observed_target[index], 1701 + fold)

    worm_rows: list[dict[str, object]] = []
    for worm in test_idx:
        use = test.worm == int(worm)
        worm_rows.append(
            {
                "worm_id": cohort.worm_ids[int(worm)],
                "fold": fold,
                "nll_improvement": float(np.mean(base_nll[use] - full_nll[use])),
                "scale_nll_improvement": float(
                    np.mean(full_mean_base_scale_nll[use] - full_nll[use])
                ),
                "rmse_improvement": float(
                    np.sqrt(np.square(test.observed_target[use] - observed_base_mean[use]).mean())
                    - np.sqrt(np.square(test.observed_target[use] - observed_full_mean[use]).mean())
                ),
            }
        )
    eval_worm = test.worm[index]
    base_energy = base_metrics["energy"]
    full_energy = full_metrics["energy"]
    for row in worm_rows:
        worm_index = cohort.worm_ids.index(str(row["worm_id"]))
        use = eval_worm == worm_index
        row["energy_improvement"] = float(np.mean(base_energy[use] - full_energy[use]))

    mean_kernel = reconstruct_kernel(fitted.mean_kernel_basis, basis)
    scale_kernel = reconstruct_kernel(fitted.scale_kernel_basis, basis)
    arrays = {
        "alpha": alpha,
        "intercept": intercept,
        "mean_kernel": mean_kernel,
        "scale_kernel": scale_kernel,
        "mean_kernel_basis": fitted.mean_kernel_basis,
        "scale_kernel_basis": fitted.scale_kernel_basis,
        "base_offset": base_offset,
        "full_offset": full_offset,
        "mean_base_coef": np.stack([model.coef_ for model in fitted.base_models]).astype(np.float32),
        "mean_base_intercept": np.asarray([model.intercept_ for model in fitted.base_models], dtype=np.float32),
        "scale_base_coef": np.stack([model.coef_ for model in fitted.scale_base_models]).astype(np.float32),
        "scale_base_intercept": np.asarray([model.intercept_ for model in fitted.scale_base_models], dtype=np.float32),
        "common_to_neural_coef": np.asarray(fitted.common_to_neural.coef_, dtype=np.float32),
        "common_to_neural_intercept": np.asarray(fitted.common_to_neural.intercept_, dtype=np.float32),
        "neural_feature_mean": fitted.neural_mean,
        "neural_feature_scale": fitted.neural_scale,
        "common_feature_mean": fitted.common_mean,
        "common_feature_scale": fitted.common_scale,
        "pca_components": np.asarray(pca.components_, dtype=np.float32),
        "pca_mean": np.asarray(pca.mean_, dtype=np.float32),
        "observation_scaler_mean": scaler.mean,
        "observation_scaler_scale": scaler.scale,
    }
    metrics = {
        "fold": fold,
        "test_worms": [cohort.worm_ids[int(i)] for i in test_idx],
        "mean_nll_improvement": float(np.mean(base_nll - full_nll)),
        "mean_scale_nll_improvement": float(
            np.mean(full_mean_base_scale_nll - full_nll)
        ),
        "rmse_improvement": float(
            np.sqrt(np.square(test.observed_target - observed_base_mean).mean())
            - np.sqrt(np.square(test.observed_target - observed_full_mean).mean())
        ),
        "energy_improvement": float(np.mean(base_energy - full_energy)),
        "variogram_improvement": float(
            np.mean(base_metrics["variogram"] - full_metrics["variogram"])
        ),
        "decay_mean": float(alpha.mean()),
        "decay_min": float(alpha.min()),
        "decay_max": float(alpha.max()),
        "selected_cross_alpha": selected_cross_alpha,
        "tuning_rows": tuning_rows,
        "worm_rows": worm_rows,
    }
    return metrics, arrays


def kernel_reliability(kernels: list[np.ndarray]) -> dict[str, float]:
    correlations: list[float] = []
    signs: list[float] = []
    for first in range(len(kernels)):
        for second in range(first + 1, len(kernels)):
            a = kernels[first].reshape(-1)
            b = kernels[second].reshape(-1)
            correlations.append(float(spearmanr(a, b).statistic))
            nonzero = (np.abs(a) > 1e-12) | (np.abs(b) > 1e-12)
            signs.append(float(np.mean(np.sign(a[nonzero]) == np.sign(b[nonzero]))))
    return {
        "median_pairwise_spearman": float(np.median(correlations)),
        "mean_pairwise_spearman": float(np.mean(correlations)),
        "median_sign_agreement": float(np.median(signs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-evidence", type=Path, required=True)
    parser.add_argument("--cohort-mode", choices=("oh16230_head",), default="oh16230_head")
    parser.add_argument("--max-lag", type=int, default=32)
    parser.add_argument("--n-basis", type=int, default=8)
    parser.add_argument("--base-alpha", type=float, default=100.0)
    parser.add_argument("--cross-alpha", type=float, default=1000.0)
    parser.add_argument("--cross-alpha-grid", nargs="+", type=float)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--eval-rows", type=int, default=4000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cohort = load_cohort(cohort_mode=args.cohort_mode)
    folds = load_folds(args.fold_evidence.resolve(), cohort)
    basis = smooth_lag_basis(args.max_lag, args.n_basis)
    np.save(output / "lag_basis.npy", basis)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "protocol": "calcium-aware AR(1) innovation distributional lag model v1",
        "cohort_mode": cohort.cohort_mode,
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "neurons": list(cohort.neurons),
        "max_lag_frames": args.max_lag,
        "max_lag_seconds": args.max_lag / cohort.fps,
        "n_basis": args.n_basis,
        "base_alpha": args.base_alpha,
        "cross_alpha": args.cross_alpha,
        "cross_alpha_grid": args.cross_alpha_grid,
        "hyperparameter_provenance": (
            "whole-worm inner-validation selection by reconstructed held-out Gaussian NLL; ties prefer stronger regularization"
            if args.cross_alpha_grid
            else "inherited without retuning from the globally frozen E20 regularization scale"
        ),
        "observation_model": "per-neuron quiet-period AR(1) calcium decay; innovations drive a smooth multi-lag conditional mean and log variance",
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_schema": cohort.stimulus_schema_dict(),
        "selection_firewall": "no atlas/connectome/receptor/SBTG target used",
        "claim_boundary": "calcium-aware predictive innovation dynamics, not latent truth, causal edges, anatomy, or physical delay",
        "gate": {
            "nll": "worm-bootstrap 95% lower bound > 0",
            "energy": "worm-bootstrap 95% lower bound > 0",
            "mechanism": "either RMSE or isolated scale-NLL worm-bootstrap 95% lower bound > 0",
            "kernel_reliability": "median pairwise fold Spearman >= 0.30 for the passing mechanism",
        },
    }
    atomic_json(output / "manifest.json", manifest)
    fold_rows: list[dict] = []
    worm_rows: list[dict] = []
    mean_kernels: list[np.ndarray] = []
    scale_kernels: list[np.ndarray] = []
    for fold in range(5):
        metrics, arrays = run_fold(
            cohort,
            folds,
            fold,
            basis,
            base_alpha=args.base_alpha,
            cross_alpha=args.cross_alpha,
            cross_alpha_grid=(
                tuple(sorted(set(args.cross_alpha_grid)))
                if args.cross_alpha_grid
                else None
            ),
            samples=args.samples,
            eval_rows=args.eval_rows,
        )
        worm_rows.extend(metrics.pop("worm_rows"))
        tuning_rows = metrics.pop("tuning_rows")
        fold_rows.append(metrics)
        pd.DataFrame(tuning_rows).to_csv(output / f"fold_{fold}_tuning.csv", index=False)
        mean_kernels.append(arrays["mean_kernel"])
        scale_kernels.append(arrays["scale_kernel"])
        np.savez_compressed(
            output / f"fold_{fold}.npz",
            fold=np.asarray(fold),
            stimulus_schema_version=np.asarray(cohort.stimulus_schema_version),
            stimulus_schema_fingerprint=np.asarray(cohort.stimulus_schema_fingerprint),
            neurons=np.asarray(cohort.neurons),
            **arrays,
        )
        print(
            f"LATENT_FOLD fold={fold} nll={metrics['mean_nll_improvement']:+.6f} "
            f"energy={metrics['energy_improvement']:+.6f} "
            f"scale={metrics['mean_scale_nll_improvement']:+.6f}",
            flush=True,
        )
    fold_frame = pd.DataFrame(fold_rows)
    worm_frame = pd.DataFrame(worm_rows)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    worm_frame.to_csv(output / "worm_metrics.csv", index=False)
    mean_reliability = kernel_reliability(mean_kernels)
    scale_reliability = kernel_reliability(scale_kernels)
    intervals = {
        metric: bootstrap_ci(worm_frame[metric].to_numpy(), seed=20260828 + index)
        for index, metric in enumerate(
            ("nll_improvement", "energy_improvement", "rmse_improvement", "scale_nll_improvement")
        )
    }
    nll_pass = intervals["nll_improvement"][0] > 0
    energy_pass = intervals["energy_improvement"][0] > 0
    mean_pass = intervals["rmse_improvement"][0] > 0
    scale_pass = intervals["scale_nll_improvement"][0] > 0
    mechanism = "mean" if mean_pass else "scale" if scale_pass else None
    reliability = mean_reliability if mechanism == "mean" else scale_reliability
    reliability_pass = bool(
        mechanism is not None and reliability["median_pairwise_spearman"] >= 0.30
    )
    passed = bool(nll_pass and energy_pass and (mean_pass or scale_pass) and reliability_pass)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mean_metrics": {
            metric: float(worm_frame[metric].mean()) for metric in intervals
        },
        "worm_bootstrap_95_ci": {
            metric: list(value) for metric, value in intervals.items()
        },
        "mean_kernel_reliability": mean_reliability,
        "scale_kernel_reliability": scale_reliability,
        "nll_pass": bool(nll_pass),
        "energy_pass": bool(energy_pass),
        "mean_mechanism_pass": bool(mean_pass),
        "scale_mechanism_pass": bool(scale_pass),
        "selected_mechanism": mechanism,
        "kernel_reliability_pass": reliability_pass,
        "all_gates_pass": passed,
        "direct_distributional_sampling_authorized": passed,
        "temporal_cut_smc_authorized": False,
        "temporal_cut_rule": "requires a subsequent converged direct distributional audit even if this model passes",
        "selection_firewall": manifest["selection_firewall"],
        "claim_boundary": manifest["claim_boundary"],
    }
    atomic_json(output / "gate_decision.json", decision)
    manifest["status"] = "complete"
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "manifest.json", manifest)
    lines = [
        "# Calcium-aware latent-innovation lag model",
        "",
        "## Decision",
        "",
        (
            "**The frozen predictive and reliability gate passed.** A paired direct distributional audit is authorized; temporal-cut SMC still requires direct N=64/N=256 convergence."
            if passed
            else "**The frozen predictive/reliability gate failed.** No lag sampling is authorized from this model."
        ),
        "",
        "The model estimates a quiet-period AR(1) decay for every neuron, predicts the resulting innovation with smooth source-history lag bases, and reconstructs the observed next-frame Gaussian law. This is a calcium-aware approximation, not recovery of a known latent neural state.",
        "",
        "| Metric | Worm mean | 95% worm-bootstrap CI | Pass |",
        "| --- | ---: | ---: | :---: |",
    ]
    for metric, label in (
        ("nll_improvement", "Full vs base Gaussian NLL"),
        ("energy_improvement", "Full vs base energy"),
        ("rmse_improvement", "Full vs base RMSE"),
        ("scale_nll_improvement", "Cross-history scale NLL"),
    ):
        low, high = intervals[metric]
        metric_pass = low > 0
        lines.append(
            f"| {label} | {worm_frame[metric].mean():+.6f} | [{low:+.6f}, {high:+.6f}] | {'yes' if metric_pass else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Mean-kernel median fold Spearman: **{mean_reliability['median_pairwise_spearman']:.3f}**. Scale-kernel median fold Spearman: **{scale_reliability['median_pairwise_spearman']:.3f}**.",
            "",
            "No Cook, Randi, SBTG, receptor, transmitter, or neuromodulator map entered fitting, scoring, or selection.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
