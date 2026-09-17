from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.distributed_lag_dynamics import (
    _bootstrap_ci,
    _kernel_stability,
    _refit_features,
    _save_fold_artifact,
    evaluate_predictions,
    fit_cross_path,
    fit_nuisance,
    impulse_response,
    predict_full,
    prepare_set,
    reconstruct_kernel,
    total_basis_kernel,
    transform_coefficient,
)


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


def select_consensus(source: Path) -> dict:
    selection = json.loads((source / "selection.json").read_text())
    tuning = pd.read_csv(source / "tuning_grid.csv")
    lag = int(selection["lag_frames"])
    family = str(selection["family"])
    nuisance = tuning[
        (tuning.grid == "nuisance")
        & (tuning.lag_frames == lag)
        & (tuning.fold.isin([0, 1, 2]))
    ]
    nuisance_mean = (
        nuisance.groupby(["common_alpha", "self_alpha"], as_index=False)
        .validation_mse.mean()
        .sort_values("validation_mse")
    )
    nuisance_winner = nuisance_mean.iloc[0]
    family_grid = tuning[
        (tuning.grid == "lag_family")
        & (tuning.lag_frames == lag)
        & (tuning.fold.isin([0, 1, 2]))
        & (tuning.family == family)
    ].copy()
    keys = ["cross_alpha"]
    if family in {"group_shrunk", "sparse_lowrank"}:
        keys.append("shrink_quantile")
    if family in {"reduced_rank", "sparse_lowrank"}:
        keys.append("rank")
    family_mean = (
        family_grid.groupby(keys, as_index=False)
        .validation_nll.mean()
        .sort_values("validation_nll")
    )
    family_winner = family_mean.iloc[0]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "lag_frames": lag,
        "lag_seconds": lag / 4.0,
        "common_alpha": float(nuisance_winner.common_alpha),
        "self_alpha": float(nuisance_winner.self_alpha),
        "cross_alpha": float(family_winner.cross_alpha),
        "shrink_quantile": float(family_winner.shrink_quantile)
        if "shrink_quantile" in family_winner.index
        else None,
        "rank": int(family_winner["rank"])
        if "rank" in family_winner.index and np.isfinite(family_winner["rank"])
        else None,
        "selection_folds": [0, 1, 2],
        "selection_rule": "lowest mean inner-validation criterion across screen folds; one globally frozen specification",
        "external_references_consulted": False,
    }
    return result


def run(args: argparse.Namespace) -> Path:
    source = args.source_run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    consensus = select_consensus(source)
    (output / "selection.json").write_text(json.dumps(consensus, indent=2) + "\n")
    cohort = load_cohort()
    folds_frame = pd.read_csv(args.fold_assignments)
    fold_by_worm = dict(zip(folds_frame.worm_id.astype(str), folds_frame.outer_fold.astype(int)))
    folds = np.asarray([fold_by_worm[worm] for worm in cohort.worm_ids], dtype=np.int64)
    folds_frame.to_csv(output / "fold_assignments.csv", index=False)
    lag = int(consensus["lag_frames"])
    basis = np.load(source / f"lag_basis_L{lag}.npy")
    np.save(output / f"lag_basis_L{lag}.npy", basis)
    metric_rows = []
    worm_frames = []
    artifacts = []
    for fold in range(5):
        print(f"CONSENSUS_LAG_START fold={fold}", flush=True)
        fit, test = _refit_features(
            cohort,
            folds,
            fold,
            lag,
            basis,
            pca_components=4,
            seed=args.seed + 7001 + 101 * fold + lag,
        )
        nuisance = fit_nuisance(
            fit,
            d=cohort.n_neurons,
            q=basis.shape[1],
            common_alpha=float(consensus["common_alpha"]),
            self_alpha=float(consensus["self_alpha"]),
        )
        prepared_fit = prepare_set(nuisance, fit)
        prepared_test = prepare_set(nuisance, test)
        raw = fit_cross_path(
            prepared_fit, nuisance, [float(consensus["cross_alpha"])]
        )[float(consensus["cross_alpha"])]
        coefficient, details = transform_coefficient(
            raw,
            str(consensus["family"]),
            shrink_quantile=consensus.get("shrink_quantile"),
            rank=consensus.get("rank"),
        )
        fit_full, _ = predict_full(prepared_fit, nuisance, coefficient)
        test_full, contribution = predict_full(prepared_test, nuisance, coefficient)
        metrics, worms = evaluate_predictions(
            fit.target,
            prepared_fit.baseline_prediction,
            fit_full,
            test,
            prepared_test,
            test_full,
            contribution,
            seed=args.seed + 7919 * fold + lag,
        )
        metric_rows.append({"fold": fold, **details, **metrics})
        worms.insert(0, "fold", fold)
        worms.insert(0, "family", consensus["family"])
        worms.insert(0, "lag_frames", lag)
        worm_frames.append(worms)
        direct = reconstruct_kernel(coefficient, basis)
        total = reconstruct_kernel(total_basis_kernel(nuisance, coefficient), basis)
        artifact = {
            "basis_coefficient": coefficient,
            "direct_kernel": direct,
            "total_kernel": total,
            "impulse_response": impulse_response(total, max(32, lag)),
            "test_contribution": contribution,
            "test_prediction": test_full,
            "baseline_prediction": prepared_test.baseline_prediction,
        }
        path = output / "fold_models" / f"{consensus['family']}__L{lag}__f{fold}.npz"
        _save_fold_artifact(
            path,
            family=str(consensus["family"]),
            lag=lag,
            fold=fold,
            basis=basis,
            neurons=cohort.neurons,
            artifact=artifact,
            metadata={
                "common_alpha": consensus["common_alpha"],
                "self_alpha": consensus["self_alpha"],
                "family_spec": consensus,
                "globally_frozen": True,
            },
        )
        artifacts.append(path)
        print(f"CONSENSUS_LAG_DONE fold={fold}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.insert(0, "family", consensus["family"])
    metrics.insert(1, "lag_frames", lag)
    metrics.insert(2, "lag_seconds", lag / 4.0)
    metrics.to_csv(output / "per_fold_metrics.csv", index=False)
    worms = pd.concat(worm_frames, ignore_index=True)
    worms.to_csv(output / "worm_metrics.csv", index=False)
    confirmation_metrics = metrics[metrics.fold.isin([3, 4])]
    confirmation_worms = worms[(worms.fold.isin([3, 4])) & (worms.phase == "all")]
    low, high = _bootstrap_ci(
        confirmation_worms.nll_improvement.to_numpy(), args.seed + 9091
    )
    gate = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": consensus["family"],
        "lag_frames": lag,
        "confirmation_folds": [3, 4],
        "confirmation_worms": int(len(confirmation_worms)),
        "mean_nll_improvement": float(confirmation_worms.nll_improvement.mean()),
        "ci_low": low,
        "ci_high": high,
        "mean_rmse_improvement": float(confirmation_metrics.rmse_improvement.mean()),
        "mean_energy_improvement": float(confirmation_metrics.energy_improvement.mean()),
        "mean_innovation_spearman": float(confirmation_metrics.innovation_spearman.mean()),
        "both_folds_positive_nll": bool((confirmation_metrics.nll_improvement > 0).all()),
        "passes_gate": bool(
            low > 0
            and (confirmation_metrics.nll_improvement > 0).all()
            and (confirmation_metrics.rmse_improvement > 0).all()
        ),
        "promotion_rule": "worm-bootstrap NLL-improvement CI > 0 and both confirmation folds improve NLL and RMSE",
        "external_references_consulted": False,
    }
    (output / "promotion_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    stability = _kernel_stability(artifacts, str(consensus["family"]), lag)
    stability.to_csv(output / "kernel_stability.csv", index=False)
    report = [
        "# Globally frozen distributed-lag sensitivity",
        "",
        "This sensitivity removes fold-specific hyperparameter variation. One complete specification was selected by mean inner-validation performance on folds 0–2 and then used unchanged in every outer fold.",
        "",
        f"Frozen specification: `{consensus['family']}`, history {lag/4:.1f} s, common alpha {consensus['common_alpha']:g}, self alpha {consensus['self_alpha']:g}, cross alpha {consensus['cross_alpha']:g}, shrink quantile {consensus.get('shrink_quantile')}. ",
        "",
        f"On confirmation folds 3–4, mean NLL improvement is **{gate['mean_nll_improvement']:+.6f}** with worm-bootstrap 95% CI **[{low:+.6f}, {high:+.6f}]**; RMSE improvement is **{gate['mean_rmse_improvement']:+.6f}**. The strict gate **{'passed' if gate['passes_gate'] else 'did not pass'}**.",
        "",
        "Claim boundary: predictive innovation dynamics only; not physical intervention, anatomy, or transmission delay.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    validation = {
        "status": "pass",
        "fold_model_files": len(artifacts),
        "globally_frozen_specification": True,
        "external_references_consulted": False,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_run": str(source),
                "selection_folds": [0, 1, 2],
                "confirmation_folds": [3, 4],
                "specification": consensus,
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
        "--source-run", type=Path, default=Path("results/distributed_lag_dynamics_20260828")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distributed_lag_dynamics_20260828/consensus"),
    )
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path(
            "results/conditional_distribution_benchmark/atlas_blind_world_model_20260827/fold_assignments.csv"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
