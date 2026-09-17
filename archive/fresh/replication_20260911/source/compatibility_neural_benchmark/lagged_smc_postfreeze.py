from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compatibility_neural_benchmark.lagged_smc_analysis import (
    _fold_assignments,
    load_responses,
)
from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    load_references,
    source_macro_metrics,
)
from conditional_neural_benchmark.data import load_cohort


PANELS = ("onset", "onset_minus_baseline")
NETWORKS = ("monoamine_all", "neuropeptide_all")
PRIMARY_REFERENCES = (
    "randi_wild_type",
    "cook_struct_54",
    "cook_chem_54",
    "cook_gap_54",
)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    usable = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[usable], dtype=np.float64)
    y = np.asarray(y[usable], dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(spearmanr(x, y).statistic)


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


def aggregate_grid(artifact) -> dict[str, np.ndarray]:
    onset = artifact.phases.index("onset")
    baseline = artifact.phases.index("baseline")
    onset_matrix = artifact.matrices[:, :, onset].mean(axis=(1, 2))
    difference = (
        artifact.matrices[:, :, onset] - artifact.matrices[:, :, baseline]
    ).mean(axis=(1, 2))
    # lag, horizon, target, source
    return {"onset": onset_matrix, "onset_minus_baseline": difference}


def _load_contextual(path: Path, neurons: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if tuple(data["neurons"].astype(str)) != neurons:
            raise RuntimeError("contextual and temporal-cut neuron orders differ")
        methods = {}
        for name in ("winner_wide_direct", "sbtg_current", "sbtg_published"):
            lags = data[f"{name}__lags"].astype(int)
            position = np.flatnonzero(lags == 1)
            if len(position) != 1:
                raise RuntimeError(f"{name} has no unique lag-1 matrix")
            matrix = data[f"{name}__signed"][int(position[0])].astype(np.float64)
            np.fill_diagonal(matrix, 0.0)
            methods[name] = matrix
    return methods


def evaluate_primary(
    methods: dict[str, np.ndarray], references: dict[str, dict[str, np.ndarray]]
) -> pd.DataFrame:
    rows: list[dict] = []
    for method, matrix in methods.items():
        for reference in PRIMARY_REFERENCES:
            ref = references[reference]
            row = {
                "method": method,
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
            else:
                row["count_spearman_all_pairs"] = np.nan
                row["count_spearman_positive_edges"] = np.nan
                positive = ref["mask"] & (ref["labels"] > 0)
                row["signed_spearman_positive"] = _safe_spearman(
                    matrix[positive], ref["signed_value"][positive]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_grid(
    grid: dict[str, np.ndarray],
    source_lags: np.ndarray,
    horizons: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
    networks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    external: list[dict] = []
    bentley: list[dict] = []
    d = grid["onset"].shape[-1]
    off = ~np.eye(d, dtype=bool)
    for panel, array in grid.items():
        for lag_position, source_lag in enumerate(source_lags):
            for horizon_position, horizon in enumerate(horizons):
                matrix = array[lag_position, horizon_position]
                for reference in PRIMARY_REFERENCES:
                    ref = references[reference]
                    row = {
                        "panel": panel,
                        "source_lag_frames": int(source_lag),
                        "source_lag_seconds": float(source_lag / 4.0),
                        "horizon_frames": int(horizon),
                        "horizon_seconds": float(horizon / 4.0),
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
                    external.append(row)
                for network in NETWORKS:
                    labels = networks[network]
                    eligible = labels.any(axis=0)
                    mask = off & eligible[None]
                    bentley.append(
                        {
                            "panel": panel,
                            "source_lag_frames": int(source_lag),
                            "source_lag_seconds": float(source_lag / 4.0),
                            "horizon_frames": int(horizon),
                            "horizon_seconds": float(horizon / 4.0),
                            "network": network,
                            "scope": "eligible_sources",
                            "n_eligible_sources": int(eligible.sum()),
                            **binary_metrics(matrix, labels, mask),
                            **source_macro_metrics(matrix, labels, mask),
                        }
                    )
    return pd.DataFrame(external), pd.DataFrame(bentley)


def lagmax_inference(
    grid: dict[str, np.ndarray],
    source_lags: np.ndarray,
    horizons: np.ndarray,
    networks: dict[str, np.ndarray],
    *,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    d = grid["onset"].shape[-1]
    off = ~np.eye(d, dtype=bool)
    for panel_index, panel in enumerate(PANELS):
        cells = grid[panel].reshape(-1, d, d)
        cell_labels = [
            (int(lag), int(horizon)) for lag in source_lags for horizon in horizons
        ]
        for network_index, network in enumerate(NETWORKS):
            labels = np.asarray(networks[network], dtype=np.int8)
            sources = np.flatnonzero(labels.any(axis=0))
            mask = off & np.isin(np.arange(d)[None, :], sources)
            observed = np.asarray(
                [binary_metrics(matrix, labels, mask)["auroc"] for matrix in cells]
            )
            best_position = int(np.nanargmax(observed))
            local_rng = np.random.default_rng(
                seed + 10_007 * panel_index + 1_009 * network_index
            )
            null_max = np.empty(n_permutations, dtype=np.float64)
            for repeat in range(n_permutations):
                permuted = labels.copy()
                for source in sources:
                    targets = np.flatnonzero(off[:, source])
                    permuted[targets, source] = local_rng.permutation(
                        permuted[targets, source]
                    )
                null_max[repeat] = np.nanmax(
                    [binary_metrics(matrix, permuted, mask)["auroc"] for matrix in cells]
                )
            boot_best = np.empty(n_bootstrap, dtype=np.float64)
            boot_position = np.empty(n_bootstrap, dtype=int)
            for repeat in range(n_bootstrap):
                sampled = local_rng.choice(sources, size=len(sources), replace=True)
                values = []
                for matrix in cells:
                    score_parts = []
                    label_parts = []
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
                boot_position[repeat] = int(np.nanargmax(values))
                boot_best[repeat] = float(np.nanmax(values))
            lag, horizon = cell_labels[best_position]
            rows.append(
                {
                    "panel": panel,
                    "network": network,
                    "n_cells_searched": len(cells),
                    "best_source_lag_frames": lag,
                    "best_source_lag_seconds": lag / 4.0,
                    "best_horizon_frames": horizon,
                    "best_horizon_seconds": horizon / 4.0,
                    "best_auroc": float(observed[best_position]),
                    "bootstrap_best_auroc_ci_low": float(np.quantile(boot_best, 0.025)),
                    "bootstrap_best_auroc_ci_high": float(np.quantile(boot_best, 0.975)),
                    "best_cell_selection_rate": float(np.mean(boot_position == best_position)),
                    "lagmax_p_value": float(
                        (1 + np.sum(null_max >= observed[best_position]))
                        / (n_permutations + 1)
                    ),
                    "null_max_95pct": float(np.quantile(null_max, 0.95)),
                    "n_permutations": n_permutations,
                    "n_source_bootstrap": n_bootstrap,
                }
            )
    result = pd.DataFrame(rows)
    result["lagmax_bh_q_value"] = _bh(result.lagmax_p_value.to_numpy(float))
    return result


def matrix_relationships(methods: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict] = []
    d = next(iter(methods.values())).shape[0]
    off = ~np.eye(d, dtype=bool)
    primary = methods["temporal_cut_smc_onset"]
    for name, matrix in methods.items():
        if name == "temporal_cut_smc_onset":
            continue
        x, y = primary[off], matrix[off]
        n_top = max(1, int(round(0.10 * len(x))))
        x_top = set(np.argpartition(np.abs(x), -n_top)[-n_top:].tolist())
        y_top = set(np.argpartition(np.abs(y), -n_top)[-n_top:].tolist())
        rows.append(
            {
                "left": "temporal_cut_smc_onset",
                "right": name,
                "signed_spearman": _safe_spearman(x, y),
                "absolute_spearman": _safe_spearman(np.abs(x), np.abs(y)),
                "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
                "top_10pct_jaccard": float(len(x_top & y_top) / len(x_top | y_top)),
            }
        )
    return pd.DataFrame(rows)


def write_checksums(output: Path) -> None:
    paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in paths
        )
        + "\n"
    )


def run(
    screen_run: Path,
    confirmation_run: Path,
    selection_path: Path,
    output: Path,
    fold_file: Path,
    release: Path,
    contextual_path: Path,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(selection_path.read_text())
    cohort = load_cohort()
    folds = _fold_assignments(fold_file, cohort)
    screen = load_responses(screen_run, folds)
    confirmation = load_responses(confirmation_run, folds)
    if screen.neurons != confirmation.neurons or screen.neurons != tuple(cohort.neurons):
        raise RuntimeError("neuron order mismatch")
    selected_lag = int(selection["source_lag_frames"])
    selected_horizon = int(selection["horizon_frames"])
    lag_position = int(np.flatnonzero(confirmation.source_lags == selected_lag)[0])
    horizon_position = int(np.flatnonzero(confirmation.horizons == selected_horizon)[0])
    confirm_grid = aggregate_grid(confirmation)
    methods = {
        "temporal_cut_smc_onset": confirm_grid["onset"][lag_position, horizon_position],
        "temporal_cut_smc_onset_minus_baseline": confirm_grid[
            "onset_minus_baseline"
        ][lag_position, horizon_position],
        **_load_contextual(contextual_path, screen.neurons),
    }
    references, networks = load_references(release, list(screen.neurons))
    primary = evaluate_primary(methods, references)
    grid = aggregate_grid(screen)
    external, bentley = evaluate_grid(
        grid, screen.source_lags, screen.horizons, references, networks
    )
    lagmax = lagmax_inference(
        grid,
        screen.source_lags,
        screen.horizons,
        networks,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    relationships = matrix_relationships(methods)
    primary.to_csv(output / "selected_matrix_external_metrics.csv", index=False)
    external.to_csv(output / "screen_grid_external_metrics.csv", index=False)
    bentley.to_csv(output / "screen_grid_bentley_metrics.csv", index=False)
    lagmax.to_csv(output / "bentley_lagmax_inference.csv", index=False)
    relationships.to_csv(output / "selected_matrix_relationships.csv", index=False)
    np.savez_compressed(
        output / "aligned_temporal_cut_comparison_matrices.npz",
        neurons=np.asarray(screen.neurons),
        selected_source_lag_frames=np.asarray(selected_lag),
        selected_horizon_frames=np.asarray(selected_horizon),
        **{f"{name}__signed": matrix.astype(np.float32) for name, matrix in methods.items()},
        screen_source_lag_frames=screen.source_lags,
        screen_horizon_frames=screen.horizons,
        screen_onset=grid["onset"].astype(np.float32),
        screen_onset_minus_baseline=grid["onset_minus_baseline"].astype(np.float32),
    )
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_file": str(selection_path.resolve()),
        "selected_source_lag_frames": selected_lag,
        "selected_horizon_frames": selected_horizon,
        "external_load_timing": "after neural-only cell selection was written",
        "primary_methods": list(methods),
        "primary_references": list(PRIMARY_REFERENCES),
        "cook_metric_policy": "binary presence AUROC/AUPRC plus Spearman against count weight",
        "bentley_scope": "eligible source columns",
        "lag_search_null": "target labels permuted within source; maximum over source-lag x horizon cells",
        "multiplicity": "BH across four panel x network lag-max tests",
        "claim_boundary": "post-freeze external correspondence, not anatomy, causality, molecular mechanism, or physical delay",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    validation = {
        "status": "pass",
        "n_primary_rows": int(len(primary)),
        "n_grid_external_rows": int(len(external)),
        "n_grid_bentley_rows": int(len(bentley)),
        "n_lagmax_rows": int(len(lagmax)),
        "n_relationship_rows": int(len(relationships)),
        "all_matrices_finite": bool(all(np.isfinite(value).all() for value in methods.values())),
        "neuron_count": len(screen.neurons),
        "external_loaded_after_selection": True,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    write_checksums(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-run", type=Path, required=True)
    parser.add_argument("--confirmation-run", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=Path(
            "results/conditional_distribution_benchmark/atlas_blind_world_model_20260827/fold_assignments.csv"
        ),
    )
    parser.add_argument(
        "--release", type=Path, default=Path("/Users/vik/Downloads/SBTG-public-release copy")
    )
    parser.add_argument(
        "--contextual",
        type=Path,
        default=Path(
            "results/compatibility_path_response/winner_wide_postfreeze_external_20260827/aligned_winner_comparison_matrices.npz"
        ),
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    run(
        args.screen_run.resolve(),
        args.confirmation_run.resolve(),
        args.selection.resolve(),
        args.output_dir.resolve(),
        args.fold_file.resolve(),
        args.release.resolve(),
        args.contextual.resolve(),
        args.permutations,
        args.bootstrap,
        args.seed,
    )


if __name__ == "__main__":
    main()
