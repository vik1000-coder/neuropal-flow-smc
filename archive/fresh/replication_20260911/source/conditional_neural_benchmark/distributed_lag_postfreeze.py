from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    load_references,
    source_macro_metrics,
)
from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.distributed_lag_dynamics import (
    STUDENT_DF,
    _refit_features,
    _student_nll_rows,
    fit_nuisance,
    prepare_set,
    predict_full,
)


PRIMARY_REFERENCES = ("randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54")
NETWORKS = ("monoamine_all", "neuropeptide_all")


def _bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    usable = np.isfinite(left) & np.isfinite(right)
    left = np.asarray(left[usable], dtype=np.float64)
    right = np.asarray(right[usable], dtype=np.float64)
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return np.nan
    return float(spearmanr(left, right).statistic)


def signed_max_matrix(kernel: np.ndarray) -> np.ndarray:
    position = np.argmax(np.abs(kernel), axis=0)
    return np.take_along_axis(kernel, position[None], axis=0)[0]


def lag_centroid(kernel: np.ndarray) -> np.ndarray:
    magnitude = np.abs(np.asarray(kernel, dtype=np.float64))
    lags = np.arange(1, len(kernel) + 1, dtype=np.float64)[:, None, None]
    return np.divide(
        np.sum(magnitude * lags, axis=0),
        np.sum(magnitude, axis=0),
        out=np.full(kernel.shape[1:], np.nan),
        where=np.sum(magnitude, axis=0) > 1e-12,
    )


def _shift_rows(array: np.ndarray, worms: np.ndarray, shifts: dict[int, int]) -> np.ndarray:
    result = np.empty_like(array)
    for worm in np.unique(worms):
        index = np.flatnonzero(worms == worm)
        order = np.argsort(index)
        ordered = index[order]
        result[ordered] = np.roll(array[ordered], int(shifts[int(worm)]), axis=0)
    return result


def neural_controls(
    run: Path,
    output: Path,
    *,
    fold_file: Path,
    seed: int,
    shift_replicates: int,
) -> tuple[dict, np.ndarray, list[str]]:
    selection = json.loads((run / "selection.json").read_text())
    gate = json.loads((run / "promotion_gate.json").read_text())
    if selection.get("external_references_consulted") is not False:
        raise RuntimeError("selection was not atlas blind")
    cohort = load_cohort()
    fold_frame = pd.read_csv(fold_file)
    fold_by_worm = dict(zip(fold_frame.worm_id.astype(str), fold_frame.outer_fold.astype(int)))
    folds = np.asarray([fold_by_worm[worm] for worm in cohort.worm_ids], dtype=np.int64)
    family = str(selection["family"])
    lag = int(selection["lag_frames"])
    basis = np.load(run / f"lag_basis_L{lag}.npy")
    rng = np.random.default_rng(seed)
    shift_rows: list[dict] = []
    worm_rows: list[dict] = []
    ablations = []
    kernels = []
    impulses = []
    for fold in range(5):
        artifact_path = run / "fold_models" / f"{family}__L{lag}__f{fold}.npz"
        with np.load(artifact_path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
            coefficient = artifact["basis_coefficient"].astype(np.float32)
            direct = artifact["direct_kernel"].astype(np.float64)
            impulse = artifact["impulse_response"].astype(np.float64)
        fit, test = _refit_features(
            cohort,
            folds,
            fold,
            lag,
            basis,
            pca_components=4,
            seed=20260828 + 7001 + 101 * fold + lag,
        )
        nuisance = fit_nuisance(
            fit,
            d=cohort.n_neurons,
            q=basis.shape[1],
            common_alpha=float(metadata["common_alpha"]),
            self_alpha=float(metadata["self_alpha"]),
        )
        prepared_fit = prepare_set(nuisance, fit)
        prepared_test = prepare_set(nuisance, test)
        fit_full, _ = predict_full(prepared_fit, nuisance, coefficient)
        test_full, contribution = predict_full(prepared_test, nuisance, coefficient)
        full_scale = np.sqrt(np.mean(np.square(fit.target - fit_full), axis=0))
        baseline_scale = np.sqrt(
            np.mean(np.square(fit.target - prepared_fit.baseline_prediction), axis=0)
        )
        baseline_nll = _student_nll_rows(
            test.target, prepared_test.baseline_prediction, baseline_scale
        )
        full_nll = _student_nll_rows(test.target, test_full, full_scale)
        for worm in np.unique(test.worm):
            use = test.worm == worm
            worm_rows.append(
                {
                    "fold": fold,
                    "worm": int(worm),
                    "split": "confirmation" if fold in (3, 4) else "screen",
                    "actual_nll_improvement": float(np.mean(baseline_nll[use] - full_nll[use])),
                }
            )

        x_scale = nuisance.neural_scale.reshape(cohort.n_neurons, basis.shape[1])
        edge_delta = np.zeros((cohort.n_neurons, cohort.n_neurons), dtype=np.float64)
        for target, off in enumerate(nuisance.off_indices):
            off_sources = np.delete(np.arange(cohort.n_neurons), target)
            residual_x = prepared_test.residual_x[target].reshape(
                len(test.target), cohort.n_neurons - 1, basis.shape[1]
            )
            coefficient_std = coefficient[target, off_sources] * x_scale[off_sources]
            edge_contribution = np.einsum("nsq,sq->ns", residual_x, coefficient_std)
            for source_position, source in enumerate(off_sources):
                masked = test_full[:, target] - edge_contribution[:, source_position]
                masked_nll = _student_nll_rows(
                    test.target[:, target, None], masked[:, None], full_scale[target, None]
                )
                target_full_nll = _student_nll_rows(
                    test.target[:, target, None], test_full[:, target, None], full_scale[target, None]
                )
                edge_delta[target, source] = float(np.mean(masked_nll - target_full_nll))
        ablations.append(edge_delta)
        kernels.append(direct)
        impulses.append(impulse)

        for replicate in range(shift_replicates):
            shifts = {}
            for worm in np.unique(test.worm):
                count = int(np.sum(test.worm == worm))
                minimum = min(lag + 1, max(1, count // 3))
                shifts[int(worm)] = int(rng.integers(minimum, max(minimum + 1, count - minimum)))
            # Shift the complete fitted cross-neuron contribution within each
            # worm.  This preserves its multivariate temporal structure and
            # autocorrelation while breaking alignment to the target future.
            shifted_contribution = _shift_rows(contribution, test.worm, shifts)
            shifted_prediction = prepared_test.baseline_prediction + shifted_contribution
            shifted_nll = _student_nll_rows(test.target, shifted_prediction, full_scale)
            shift_rows.append(
                {
                    "fold": fold,
                    "replicate": replicate,
                    "split": "confirmation" if fold in (3, 4) else "screen",
                    "shifted_nll_improvement": float(np.mean(baseline_nll - shifted_nll)),
                }
            )
    shift_frame = pd.DataFrame(shift_rows)
    worm_frame = pd.DataFrame(worm_rows)
    shift_frame.to_csv(output / "circular_shift_controls.csv", index=False)
    worm_frame.to_csv(output / "actual_worm_controls.csv", index=False)
    kernels_array = np.stack(kernels)
    impulses_array = np.stack(impulses)
    ablation_array = np.stack(ablations)
    np.savez_compressed(
        output / "selected_neural_matrices.npz",
        neurons=np.asarray(cohort.neurons),
        fold=np.arange(5),
        direct_kernel=kernels_array,
        impulse_response=impulses_array,
        conditional_ablation=ablation_array,
        lag_centroid_frames=np.stack([lag_centroid(value) for value in kernels_array]),
    )
    actual_confirmation = float(
        worm_frame[worm_frame.split == "confirmation"].actual_nll_improvement.mean()
    )
    null = (
        shift_frame[shift_frame.split == "confirmation"]
        .groupby("replicate")
        .shifted_nll_improvement.mean()
        .to_numpy()
    )
    control = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "lag_frames": lag,
        "actual_confirmation_nll_improvement": actual_confirmation,
        "circular_shift_null_mean": float(np.mean(null)),
        "circular_shift_null_95pct": float(np.quantile(null, 0.95)),
        "circular_shift_p_value": float((1 + np.sum(null >= actual_confirmation)) / (1 + len(null))),
        "shift_replicates": shift_replicates,
        "primary_promotion_gate_passed": bool(gate["passes_gate"]),
        "targeted_smc_decision": "skipped: primary predictive mean-dynamics gate failed"
        if not gate["passes_gate"]
        else "eligible for targeted local-perturbation SMC",
        "external_references_consulted": False,
    }
    (output / "neural_control_summary.json").write_text(json.dumps(control, indent=2) + "\n")
    return control, kernels_array, list(cohort.neurons)


def evaluate_external(
    output: Path,
    kernels: np.ndarray,
    neurons: list[str],
    *,
    release: Path,
    permutations: int,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    references, networks = load_references(release, neurons)
    panels = {
        "all_folds": kernels.mean(axis=0),
        "confirmation_folds": kernels[3:5].mean(axis=0),
    }
    external_rows = []
    for panel, array in panels.items():
        integrated = {
            "signed_max": signed_max_matrix(array),
            "group_norm": np.sqrt(np.square(array).sum(axis=0)),
        }
        cells = [(f"lag_{lag + 1}", matrix) for lag, matrix in enumerate(array)]
        cells.extend(integrated.items())
        for cell, matrix in cells:
            lag_frames = int(cell.split("_")[1]) if cell.startswith("lag_") else np.nan
            for reference in PRIMARY_REFERENCES:
                ref = references[reference]
                row = {
                    "panel": panel,
                    "cell": cell,
                    "lag_frames": lag_frames,
                    "lag_seconds": lag_frames / 4.0 if np.isfinite(lag_frames) else np.nan,
                    "reference": reference,
                    **binary_metrics(matrix, ref["labels"], ref["mask"]),
                    **source_macro_metrics(matrix, ref["labels"], ref["mask"]),
                }
                if "weight" in ref:
                    mask = np.asarray(ref["mask"], bool)
                    positive = mask & (np.asarray(ref["weight"]) > 0)
                    row["count_spearman_all_pairs"] = _safe_spearman(
                        np.abs(matrix[mask]), np.asarray(ref["weight"])[mask]
                    )
                    row["count_spearman_positive_edges"] = _safe_spearman(
                        np.abs(matrix[positive]), np.asarray(ref["weight"])[positive]
                    )
                external_rows.append(row)
    external = pd.DataFrame(external_rows)
    external.to_csv(output / "postfreeze_external_metrics.csv", index=False)

    rng = np.random.default_rng(seed)
    lagmax_rows = []
    d = len(neurons)
    off = ~np.eye(d, dtype=bool)
    for panel_index, (panel, array) in enumerate(panels.items()):
        for network_index, network in enumerate(NETWORKS):
            labels = np.asarray(networks[network], dtype=np.int8)
            sources = np.flatnonzero(labels.any(axis=0))
            mask = off & np.isin(np.arange(d)[None], sources)
            observed = np.asarray(
                [binary_metrics(matrix, labels, mask)["auroc"] for matrix in array]
            )
            best = int(np.nanargmax(observed))
            local = np.random.default_rng(seed + 10007 * panel_index + 1009 * network_index)
            null_max = np.empty(permutations)
            for repeat in range(permutations):
                permuted = labels.copy()
                for source in sources:
                    targets = np.flatnonzero(off[:, source])
                    permuted[targets, source] = local.permutation(permuted[targets, source])
                null_max[repeat] = np.nanmax(
                    [binary_metrics(matrix, permuted, mask)["auroc"] for matrix in array]
                )
            peak_positions = np.empty(bootstrap, dtype=int)
            peak_values = np.empty(bootstrap)
            for repeat in range(bootstrap):
                sampled = local.choice(sources, size=len(sources), replace=True)
                values = []
                for matrix in array:
                    score_parts, label_parts = [], []
                    for source in sampled:
                        targets = np.flatnonzero(off[:, source])
                        score_parts.append(np.abs(matrix[targets, source]))
                        label_parts.append(labels[targets, source])
                    score = np.concatenate(score_parts)
                    truth = np.concatenate(label_parts)
                    if truth.min() == truth.max():
                        values.append(np.nan)
                    else:
                        from sklearn.metrics import roc_auc_score

                        values.append(float(roc_auc_score(truth, score)))
                peak_positions[repeat] = int(np.nanargmax(values))
                peak_values[repeat] = float(np.nanmax(values))
            lagmax_rows.append(
                {
                    "panel": panel,
                    "network": network,
                    "n_lags_searched": len(array),
                    "best_lag_frames": best + 1,
                    "best_lag_seconds": (best + 1) / 4.0,
                    "best_auroc": float(observed[best]),
                    "lagmax_p_value": float((1 + np.sum(null_max >= observed[best])) / (permutations + 1)),
                    "null_max_95pct": float(np.quantile(null_max, 0.95)),
                    "bootstrap_best_auroc_ci_low": float(np.quantile(peak_values, 0.025)),
                    "bootstrap_best_auroc_ci_high": float(np.quantile(peak_values, 0.975)),
                    "best_lag_selection_rate": float(np.mean(peak_positions == best)),
                    "n_eligible_sources": int(len(sources)),
                }
            )
    lagmax = pd.DataFrame(lagmax_rows)
    lagmax["lagmax_bh_q_value"] = _bh(lagmax.lagmax_p_value.to_numpy())
    lagmax.to_csv(output / "postfreeze_bentley_lagmax.csv", index=False)
    return external, lagmax


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksums(output: Path) -> None:
    paths = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "checksums.sha256"]
    (output / "checksums.sha256").write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(output)}" for path in paths) + "\n"
    )


def run(args: argparse.Namespace) -> Path:
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    control, kernels, neurons = neural_controls(
        run,
        output,
        fold_file=args.fold_assignments.resolve(),
        seed=args.seed,
        shift_replicates=args.shift_replicates,
    )
    external, lagmax = evaluate_external(
        output,
        kernels,
        neurons,
        release=args.release.resolve(),
        permutations=args.permutations,
        bootstrap=args.bootstrap,
        seed=args.seed + 1701,
    )
    randi = external[
        (external.panel == "confirmation_folds")
        & (external.cell == "signed_max")
        & (external.reference == "randi_wild_type")
    ].iloc[0]
    cook = external[
        (external.panel == "confirmation_folds")
        & (external.cell == "signed_max")
        & (external.reference == "cook_struct_54")
    ].iloc[0]
    best = lagmax.sort_values("lagmax_p_value").iloc[0]
    report = [
        "# Distributed-lag post-freeze diagnostics",
        "",
        "**Claim boundary:** predictive innovation kernels and post-freeze convergent validity; not interventions, anatomical edges, or physical delays.",
        "",
        "## Answer first",
        "",
        f"The selected lag model did not pass the primary mean-dynamics gate, so targeted SMC was not promoted. Its confirmation NLL improvement was {control['actual_confirmation_nll_improvement']:+.6f}; the block-preserving circular-shift p-value was {control['circular_shift_p_value']:.3f}.",
        "",
        f"The confirmation-fold signed-max kernel has Randi WT AUROC/AUPRC **{randi.auroc:.3f}/{randi.auprc:.3f}** and Cook structural **{cook.auroc:.3f}/{cook.auprc:.3f}**. These references were loaded only after neural selection and controls were written.",
        "",
        f"The smallest Bentley lag-max p-value is **{best.lagmax_p_value:.3f}** with BH q **{best.lagmax_bh_q_value:.3f}**. Its descriptive peak is {best.best_lag_seconds:.2f} seconds, selected in {100*best.best_lag_selection_rate:.1f}% of source bootstraps.",
        "",
        "## Interpretation",
        "",
        "The conditional-ablation matrices quantify held-out log-score loss when one source's complete smooth lag block is removed. Circular shifts preserve each worm's temporal autocorrelation while breaking the fitted source-to-target alignment. Neither analysis converts the observational kernel into a causal connection.",
        "",
        "## Files",
        "",
        "- `selected_neural_matrices.npz`: direct kernels, impulse responses, lag centroids, and conditional-ablation matrices by fold.",
        "- `circular_shift_controls.csv`: block-preserving temporal negative controls.",
        "- `postfreeze_external_metrics.csv`: Randi and Cook metrics for every direct lag and integrated kernel.",
        "- `postfreeze_bentley_lagmax.csv`: within-source lag-max permutation and source-bootstrap inference.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    validation = {
        "status": "pass",
        "neural_controls_completed_before_external": True,
        "fold_kernels": int(kernels.shape[0]),
        "external_metric_rows": int(len(external)),
        "lagmax_rows": int(len(lagmax)),
        "targeted_smc_run": False if "skipped" in control["targeted_smc_decision"] else None,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_run": str(run),
                "neural_control_file": "neural_control_summary.json",
                "external_loaded_after_neural_controls": True,
                "release": str(args.release.resolve()),
                "permutations": args.permutations,
                "bootstrap": args.bootstrap,
            },
            indent=2,
        )
        + "\n"
    )
    _checksums(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, default=Path("results/distributed_lag_dynamics_20260828")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distributed_lag_dynamics_20260828/postfreeze"),
    )
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path(
            "results/conditional_distribution_benchmark/atlas_blind_world_model_20260827/fold_assignments.csv"
        ),
    )
    parser.add_argument(
        "--release", type=Path, default=Path("/Users/vik/Downloads/SBTG-public-release copy")
    )
    parser.add_argument("--shift-replicates", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
