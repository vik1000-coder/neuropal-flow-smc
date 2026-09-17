from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

from compatibility_neural_benchmark.postfreeze_external_analysis import load_references
from conditional_neural_benchmark.data import FoldScaler
from conditional_neural_benchmark.distributional_lag_tournament import (
    _calibrate_full_logvar,
    _cross_contribution,
    _fit_scale_baseline,
    _fit_selected_scale,
    _gaussian_nll_rows,
    _prepare_fold,
    _ridge,
    cohort_folds,
    load_cohorts,
)
from conditional_neural_benchmark.distributed_lag_dynamics import (
    _causal_fill,
    _standardize,
    reconstruct_kernel,
    smooth_lag_basis,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_gaussian_nll(y: np.ndarray, mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    safe = np.clip(logvar, -10.0, 5.0)
    return 0.5 * (math.log(2.0 * math.pi) + safe + np.square(y - mean) * np.exp(-safe))


def _selected_fold_details(cohort, folds, fold: int, selected: dict):
    mean_spec = selected["mean"]
    scale_spec = selected["scale"]
    bundle = _prepare_fold(cohort, folds, fold, int(mean_spec["lag"]))
    fitted = _fit_selected_scale(bundle, mean_spec, scale_spec)
    mean_coefficient, scale_coefficient = fitted[:2]
    train_mean, validation_mean, test_mean = fitted[2:5]
    train_logvar, validation_logvar, test_logvar = fitted[5:]
    baseline = _fit_scale_baseline(bundle, train_mean, validation_mean, test_mean)
    total_contribution = _cross_contribution(bundle.test, bundle.nuisance, scale_coefficient)
    offset = np.median(
        test_logvar - baseline.test_logvar - total_contribution, axis=0
    )
    return (
        bundle, mean_coefficient, scale_coefficient, test_mean,
        baseline.test_logvar, test_logvar, total_contribution, offset,
    )


def neural_controls(
    output: Path,
    structured: Path,
    evidence: Path,
    *,
    shifts: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    selections = json.loads((structured / "selection.json").read_text())
    cohorts = load_cohorts()
    actual_rows: list[dict] = []
    shift_rows: list[dict] = []
    ablations: dict[str, list[np.ndarray]] = {name: [] for name in cohorts}
    rng = np.random.default_rng(seed)
    for name, cohort in cohorts.items():
        folds = cohort_folds(name, cohort, evidence)
        for fold in (3, 4):
            (
                bundle, _, scale_coefficient, test_mean,
                baseline_logvar, full_logvar, contribution, offset,
            ) = _selected_fold_details(cohort, folds, fold, selections[name])
            target = bundle.test_features.target
            baseline_nll = _gaussian_nll_rows(target, test_mean, baseline_logvar)
            full_nll = _gaussian_nll_rows(target, test_mean, full_logvar)
            for worm in np.unique(bundle.test.worm):
                use = bundle.test.worm == worm
                for phase, phase_use in (
                    ("all", use),
                    ("onset", use & (bundle.test.stratum == "onset")),
                    ("quiet", use & (bundle.test.stratum == "off")),
                ):
                    if not np.any(phase_use):
                        continue
                    actual_rows.append({
                        "cohort": name, "fold": fold, "worm": int(worm),
                        "phase": phase, "rows": int(np.sum(phase_use)),
                        "scale_gaussian_nll_improvement": float(np.mean(
                            baseline_nll[phase_use] - full_nll[phase_use]
                        )),
                    })
            component_full = _component_gaussian_nll(target, test_mean, full_logvar)
            edge_delta = np.zeros((cohort.n_neurons, cohort.n_neurons), dtype=np.float64)
            q = bundle.nuisance.n_basis
            scale = bundle.nuisance.neural_scale.reshape(cohort.n_neurons, q)
            for target_index in range(cohort.n_neurons):
                sources = np.delete(np.arange(cohort.n_neurons), target_index)
                residual = bundle.test.residual_x[target_index]
                for source_position, source_index in enumerate(sources):
                    sl = slice(source_position * q, (source_position + 1) * q)
                    coefficient_std = (
                        scale_coefficient[target_index, source_index] * scale[source_index]
                    )
                    source_contribution = residual[:, sl] @ coefficient_std
                    ablated_logvar = full_logvar[:, target_index] - source_contribution
                    masked_nll = _component_gaussian_nll(
                        target[:, target_index], test_mean[:, target_index], ablated_logvar
                    )
                    edge_delta[target_index, source_index] = float(np.mean(
                        masked_nll - component_full[:, target_index]
                    ))
            ablations[name].append(edge_delta.astype(np.float32))
            for replicate in range(shifts):
                shifted = np.empty_like(contribution)
                for worm in np.unique(bundle.test.worm):
                    index = np.flatnonzero(bundle.test.worm == worm)
                    minimum = min(max(2, int(selections[name]["mean"]["lag"])), len(index) - 1)
                    if minimum >= len(index) - 1:
                        amount = 1
                    else:
                        amount = int(rng.integers(minimum, len(index) - minimum))
                    shifted[index] = np.roll(contribution[index], amount, axis=0)
                shifted_logvar = np.clip(
                    baseline_logvar + offset[None] + shifted, -10.0, 5.0
                )
                shifted_nll = _gaussian_nll_rows(target, test_mean, shifted_logvar)
                shift_rows.append({
                    "cohort": name, "fold": fold, "replicate": replicate,
                    "shifted_scale_nll_improvement": float(np.mean(baseline_nll - shifted_nll)),
                })
    actual = pd.DataFrame(actual_rows)
    shifted = pd.DataFrame(shift_rows)
    actual.to_csv(output / "actual_scale_controls.csv", index=False)
    shifted.to_csv(output / "circular_shift_scale_controls.csv", index=False)
    matrices = {
        name: np.mean(np.stack(values), axis=0).astype(np.float32)
        for name, values in ablations.items()
    }
    np.savez_compressed(
        output / "conditional_scale_ablation_matrices.npz", **matrices
    )
    summary = {}
    for name in cohorts:
        actual_value = float(actual[(actual.cohort == name) & (actual.phase == "all")]
                             .scale_gaussian_nll_improvement.mean())
        null = shifted[shifted.cohort == name].groupby("replicate")
        null_values = null.shifted_scale_nll_improvement.mean().to_numpy()
        summary[name] = {
            "actual_confirmation_scale_nll_improvement": actual_value,
            "shift_null_mean": float(np.mean(null_values)),
            "shift_null_95pct": float(np.quantile(null_values, 0.95)),
            "shift_p_value": float((1 + np.sum(null_values >= actual_value)) / (1 + len(null_values))),
            "replicates": int(len(null_values)),
        }
    _write_json(output / "neural_control_summary.json", summary)
    return actual, shifted, matrices


def kernel_stability(structured: Path, output: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for cohort in ("current54", "sbtg80", "sbtg_bridge54"):
        paths = [structured / "fold_models" / f"{cohort}__f{fold}.npz" for fold in range(5)]
        values = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                values.append({
                    "location": data["location_kernel"],
                    "logvariance": data["logvariance_kernel"],
                    "covariance_gain": data["covariance_gain_kernel"],
                })
        d = values[0]["location"].shape[1]
        off = ~np.eye(d, dtype=bool)
        for functional in values[0]:
            for left in range(5):
                for right in range(left + 1, 5):
                    a, b = values[left][functional], values[right][functional]
                    flat_a, flat_b = a[:, off].ravel(), b[:, off].ravel()
                    edge_a = np.max(np.abs(a), axis=0)[off]
                    edge_b = np.max(np.abs(b), axis=0)[off]
                    rows.append({
                        "cohort": cohort, "functional": functional,
                        "fold_left": left, "fold_right": right,
                        "kernel_pearson": float(np.corrcoef(flat_a, flat_b)[0, 1]),
                        "edge_rank_spearman": float(spearmanr(edge_a, edge_b).statistic),
                        "sign_agreement": float(np.mean(np.sign(flat_a) == np.sign(flat_b))),
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "kernel_stability.csv", index=False)
    return frame


def _variance_sequence(
    blocks: list[np.ndarray], *, seed: int, effect: float, smooth: bool, length: int = 4500
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, max_lag = min(12, blocks[0].shape[1]), 16
    truth = np.zeros((max_lag, d, d), dtype=np.float64)
    pairs = [(target, source) for target in range(d) for source in range(d) if target != source]
    chosen = rng.choice(len(pairs), 20, replace=False)
    delays = (1, 2, 4, 8, 16)
    for position, index in enumerate(chosen):
        target, source = pairs[int(index)]
        truth[delays[position % len(delays)] - 1, target, source] = effect
    noise = np.empty((length, d), dtype=np.float64)
    cursor = 0
    while cursor < length:
        block = np.asarray(blocks[int(rng.integers(len(blocks)))], dtype=np.float64)[:, :d]
        start = int(rng.integers(max(1, len(block) - 64)))
        take = min(64, length - cursor, len(block) - start)
        noise[cursor: cursor + take] = block[start: start + take]
        cursor += take
    noise -= noise.mean(axis=0)
    noise /= np.maximum(noise.std(axis=0), 1e-6)
    latent = np.zeros((length, d), dtype=np.float64)
    for time_index in range(max_lag, length):
        history = latent[time_index - max_lag: time_index][::-1]
        log_gain = np.einsum("ls,lts->t", history, truth, optimize=True)
        scale = np.exp(0.5 * np.clip(log_gain, -1.5, 1.5)) * 0.18
        latent[time_index] = 0.55 * latent[time_index - 1] + scale * noise[time_index]
    observed = latent.copy()
    if smooth:
        for index in range(1, length):
            observed[index] = 0.68 * observed[index - 1] + 0.32 * latent[index]
    return observed.astype(np.float32), truth.astype(np.float32)


def variance_semisynthetic(
    cohorts: dict,
    output: Path,
    seed: int,
    *,
    effects: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30),
    replicates: int = 6,
    filename: str = "variance_semisynthetic.csv",
) -> pd.DataFrame:
    rows: list[dict] = []
    basis = smooth_lag_basis(16)
    q = basis.shape[1]
    for cohort_name in ("current54", "sbtg80"):
        cohort = cohorts[cohort_name]
        scaler = FoldScaler.fit(cohort.traces)
        blocks = [
            np.diff(np.nan_to_num(_causal_fill(scaler.transform(trace))), axis=0)
            for trace in cohort.traces
        ]
        for smooth in (False, True):
            for effect in effects:
                for replicate in range(replicates):
                    observed, truth = _variance_sequence(
                        blocks, seed=seed + 1009 * replicate + int(effect * 10000) + 31 * smooth,
                        effect=effect, smooth=smooth,
                    )
                    history = np.lib.stride_tricks.sliding_window_view(
                        observed, window_shape=16, axis=0
                    )[:-1].transpose(0, 2, 1)
                    x = np.einsum("nld,lq->ndq", history, basis[::-1], optimize=True).reshape(len(history), -1)
                    y = observed[16:]
                    first, second = 2600, 3600
                    x_train, (x_validation, x_test), _, x_scale = _standardize(
                        x[:first], (x[first:second], x[second:])
                    )
                    mean_model = Ridge(alpha=100.0).fit(x_train, y[:first])
                    residual_train = y[:first] - mean_model.predict(x_train)
                    residual_validation = y[first:second] - mean_model.predict(x_validation)
                    floor = np.maximum(np.quantile(np.square(residual_train), 0.1, axis=0), 1e-5)
                    z_train = np.log(np.square(residual_train) + floor)
                    best = None
                    for alpha in (100.0, 1000.0, 10000.0):
                        model = Ridge(alpha=alpha).fit(x_train, z_train)
                        logvar = model.predict(x_validation)
                        offset = np.log(np.maximum(np.mean(
                            np.square(residual_train) / np.exp(model.predict(x_train)), axis=0
                        ), 1e-4))
                        nll = _component_gaussian_nll(
                            residual_validation, np.zeros_like(residual_validation),
                            np.clip(logvar + offset, -10, 5),
                        ).mean()
                        if best is None or nll < best[0]:
                            best = (float(nll), alpha)
                    x_fit = np.concatenate([x_train, x_validation])
                    y_fit = y[:second]
                    mean_fit = Ridge(alpha=100.0).fit(x_fit, y_fit)
                    residual_fit = y_fit - mean_fit.predict(x_fit)
                    model = Ridge(alpha=best[1]).fit(
                        x_fit, np.log(np.square(residual_fit) + floor)
                    )
                    raw = model.coef_.reshape(y.shape[1], y.shape[1], q) / x_scale.reshape(y.shape[1], q)[None]
                    raw[np.arange(y.shape[1]), np.arange(y.shape[1])] = 0.0
                    estimated = reconstruct_kernel(raw, basis)
                    d = y.shape[1]
                    off = ~np.eye(d, dtype=bool)
                    labels = (np.max(np.abs(truth), axis=0) > 0) & off
                    score = np.max(np.abs(estimated), axis=0)
                    true_lag = np.argmax(np.abs(truth), axis=0) + 1
                    estimated_lag = np.argmax(np.abs(estimated), axis=0) + 1
                    rows.append({
                        "noise_cohort": cohort_name,
                        "condition": "calcium_smoothed" if smooth else "latent",
                        "effect": effect, "replicate": replicate,
                        "selected_alpha": best[1],
                        "edge_auroc": float(roc_auc_score(labels[off], score[off])),
                        "lag_mae_frames": float(np.mean(np.abs(true_lag[labels] - estimated_lag[labels]))),
                        "lag_within_one_frame": float(np.mean(
                            np.abs(true_lag[labels] - estimated_lag[labels]) <= 1
                        )),
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / filename, index=False)
    return frame


def _signed_max(kernel: np.ndarray) -> np.ndarray:
    index = np.argmax(np.abs(kernel), axis=0)
    return np.take_along_axis(kernel, index[None], axis=0)[0]


def _target_covariates(cohort) -> np.ndarray:
    traces = np.concatenate([np.nan_to_num(value) for value in cohort.traces], axis=0)
    variance = np.var(traces, axis=0)
    derivative = np.std(np.diff(traces, axis=0), axis=0)
    amplitude = np.mean(np.abs(traces), axis=0)
    result = np.stack([np.log1p(variance), np.log1p(derivative), np.log1p(amplitude)], axis=1)
    scale = np.where(result.std(axis=0) > 1e-8, result.std(axis=0), 1.0)
    return (result - result.mean(axis=0)) / scale


def _matched_source_statistic(
    score: np.ndarray, adjacency: np.ndarray, covariates: np.ndarray, rng: np.random.Generator,
    permute: bool,
) -> tuple[float, int]:
    source_values = []
    for source in range(score.shape[1]):
        positives = np.flatnonzero(adjacency[:, source] > 0)
        positives = positives[positives != source]
        candidates = np.flatnonzero((adjacency[:, source] == 0) & (np.arange(len(score)) != source))
        if len(positives) == 0 or len(candidates) == 0:
            continue
        if permute:
            pool = np.concatenate([positives, candidates])
            positives = rng.choice(pool, size=len(positives), replace=False)
            candidates = np.setdiff1d(pool, positives)
        matched = []
        for target in positives:
            distance = np.sum(np.square(covariates[candidates] - covariates[target]), axis=1)
            matched.append(candidates[int(np.argmin(distance))])
        source_values.append(float(np.mean(score[positives, source]) - np.mean(score[matched, source])))
    return (float(np.mean(source_values)) if source_values else float("nan"), len(source_values))


def matched_enrichment(
    score: np.ndarray, adjacency: np.ndarray, covariates: np.ndarray,
    *, seed: int, permutations: int,
) -> dict:
    rng = np.random.default_rng(seed)
    observed, sources = _matched_source_statistic(score, adjacency, covariates, rng, False)
    null = np.asarray([
        _matched_source_statistic(score, adjacency, covariates, rng, True)[0]
        for _ in range(permutations)
    ])
    null = null[np.isfinite(null)]
    return {
        "matched_enrichment": observed,
        "eligible_sources": sources,
        "permutation_p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        "null_mean": float(np.mean(null)),
        "null_95pct": float(np.quantile(null, 0.95)),
    }


def _external_metrics(matrix: np.ndarray, references: dict) -> list[dict]:
    rows = []
    score = np.abs(matrix)
    for name in ("randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54"):
        ref = references[name]
        mask = np.asarray(ref["mask"], dtype=bool) & np.isfinite(score)
        labels = np.asarray(ref["labels"])[mask]
        values = score[mask]
        row = {
            "reference": name,
            "n_pairs": int(mask.sum()), "n_positive": int(labels.sum()),
            "auroc": float(roc_auc_score(labels, values)),
            "auprc": float(average_precision_score(labels, values)),
        }
        if name.startswith("randi"):
            dff = np.asarray(ref["signed_value"])
            use = mask & np.isfinite(dff)
            row["continuous_abs_spearman"] = float(spearmanr(score[use], np.abs(dff[use])).statistic)
            row["continuous_signed_spearman"] = float(spearmanr(matrix[use], dff[use]).statistic)
        else:
            weight = np.asarray(ref["weight"])
            row["count_spearman_all"] = float(spearmanr(score[mask], np.log1p(weight[mask])).statistic)
            positive = mask & (weight > 0)
            row["count_spearman_positive"] = float(
                spearmanr(score[positive], np.log1p(weight[positive])).statistic
            )
        rows.append(row)
    return rows


def _published_sbtg(neurons: tuple[str, ...], path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        names = data["neuron_names"].astype(str).tolist()
        lookup = {name: index for index, name in enumerate(names)}
        index = np.asarray([lookup[name] for name in neurons])
        lags = data["lags"].astype(int).tolist()
        matrices = np.stack([data[f"mu_hat_lag{lag}"] for lag in lags])
    return matrices[:, index][:, :, index]


def external_analysis(
    output: Path,
    structured: Path,
    release: Path,
    sbtg_result: Path,
    ablation: dict[str, np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohorts = load_cohorts()
    metric_rows: list[dict] = []
    enrichment_rows: list[dict] = []
    for cohort_name, cohort in cohorts.items():
        kernels = {key: [] for key in ("location", "logvariance", "covariance_gain")}
        for fold in (3, 4):
            with np.load(
                structured / "fold_models" / f"{cohort_name}__f{fold}.npz",
                allow_pickle=False,
            ) as data:
                kernels["location"].append(data["location_kernel"])
                kernels["logvariance"].append(data["logvariance_kernel"])
                kernels["covariance_gain"].append(data["covariance_gain_kernel"])
        methods = {
            key: _signed_max(np.mean(np.stack(value), axis=0))
            for key, value in kernels.items()
        }
        methods["conditional_scale_ablation"] = ablation[cohort_name]
        methods["sbtg_published"] = _signed_max(_published_sbtg(cohort.neurons, sbtg_result))
        if cohort_name != "sbtg80":
            # Contextual 54-neuron comparators are opened only after the neural
            # freeze.  Keep all estimators that have been discussed in the
            # project, rather than showing only the strongest prior flow.
            comparison_path = Path(
                "results/compatibility_path_response/full_progressive_analysis_20260827/"
                "aligned_full_ensemble_matrices.npz"
            )
            with np.load(comparison_path, allow_pickle=False) as comparison:
                comparison_neurons = comparison["neurons"].astype(str).tolist()
                if comparison_neurons == list(cohort.neurons):
                    for key, label in (
                        ("importance_weighting_full__signed", "importance_weighting_full"),
                        ("winner_wide_direct__signed", "wide_flow_direct"),
                        ("terminal_smc_full__signed", "terminal_smc_full"),
                        ("progressive_smc_full__signed", "progressive_smc_full"),
                        ("sbtg_current__signed", "sbtg_current"),
                    ):
                        methods[label] = _signed_max(comparison[key])
        references, networks = load_references(release, list(cohort.neurons))
        covariates = _target_covariates(cohort)
        for method, matrix in methods.items():
            for row in _external_metrics(matrix, references):
                metric_rows.append({"cohort": cohort_name, "method": method, **row})
            magnitude = np.abs(matrix)
            for network in (
                "neuromodulator_union", "monoamine_all", "neuropeptide_all",
                "monoamine_dopamine", "monoamine_serotonin",
            ):
                if network not in networks:
                    continue
                result = matched_enrichment(
                    magnitude, networks[network], covariates,
                    seed=seed + len(enrichment_rows), permutations=permutations,
                )
                enrichment_rows.append({
                    "cohort": cohort_name, "method": method,
                    "network": network, **result,
                })
    metrics = pd.DataFrame(metric_rows)
    enrichment = pd.DataFrame(enrichment_rows)
    metrics.to_csv(output / "postfreeze_randi_cook_metrics.csv", index=False)
    enrichment.to_csv(output / "postfreeze_receptor_matched_enrichment.csv", index=False)
    return metrics, enrichment


def _write_checksums(root: Path) -> None:
    paths = sorted(
        value for value in root.rglob("*")
        if value.is_file() and value.name != "checksums.sha256"
    )
    (root / "checksums.sha256").write_text(
        "\n".join(f"{_sha256(value)}  {value.relative_to(root)}" for value in paths) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured", type=Path, default=Path("results/higher_order_lag_20260828"))
    parser.add_argument("--output", type=Path, default=Path("results/higher_order_postfreeze_20260828"))
    parser.add_argument("--evidence", type=Path, default=Path("results/distributed_lag_dynamics_20260828"))
    parser.add_argument("--release", type=Path, default=Path("/Users/vik/Downloads/SBTG-public-release copy"))
    parser.add_argument("--sbtg-result", type=Path, default=Path("SBTG/merged_results/result_C_merged.npz"))
    parser.add_argument("--shifts", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--wait-hours", type=float, default=3.0)
    args = parser.parse_args()
    structured, output = args.structured.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_hours * 3600
    while not (structured / "validation.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("structured neural selection has not completed")
        time.sleep(10)
    actual, shifted, ablation = neural_controls(
        output, structured, args.evidence.resolve(), shifts=args.shifts, seed=args.seed
    )
    stability = kernel_stability(structured, output)
    cohorts = load_cohorts()
    semisynthetic = variance_semisynthetic(cohorts, output, args.seed)
    _write_json(output / "neural_freeze.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "structured_selection_sha256": _sha256(structured / "selection.json"),
        "actual_control_rows": int(len(actual)),
        "shift_control_rows": int(len(shifted)),
        "stability_rows": int(len(stability)),
        "variance_semisynthetic_rows": int(len(semisynthetic)),
        "external_references_consulted": False,
        "freeze_boundary": "external files are opened only after this record is written",
    })
    metrics, enrichment = external_analysis(
        output, structured, args.release.resolve(), args.sbtg_result.resolve(), ablation,
        permutations=args.permutations, seed=args.seed,
    )
    _write_json(output / "validation.json", {
        "status": "pass",
        "actual_control_rows": int(len(actual)),
        "shift_control_rows": int(len(shifted)),
        "stability_rows": int(len(stability)),
        "variance_semisynthetic_rows": int(len(semisynthetic)),
        "external_metric_rows": int(len(metrics)),
        "matched_enrichment_rows": int(len(enrichment)),
        "all_numeric_finite_or_missing": bool(
            np.isfinite(metrics.select_dtypes(include=[np.number]).fillna(0)).all().all()
            and np.isfinite(enrichment.select_dtypes(include=[np.number]).fillna(0)).all().all()
        ),
    })
    _write_checksums(output)
    print(f"HIGHER_ORDER_POSTFREEZE_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
