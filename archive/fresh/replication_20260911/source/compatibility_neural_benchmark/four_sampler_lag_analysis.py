from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from compatibility_neural_benchmark.latent_distributional_audit import (
    benjamini_hochberg,
)
from compatibility_neural_benchmark.paired_lag_correspondence import (
    INFERENCE_NETWORKS,
    PRIMARY_NETWORKS,
    SPECIFIC_NETWORKS,
    permute_within_source,
)
from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    load_references,
    source_macro_metrics,
)


SAMPLERS = (
    "direct_importance",
    "terminal_smc",
    "progressive_bridge_smc",
    "temporal_cut_smc",
)
CHANNELS = (
    "endpoint_mean",
    "cumulative_mean",
    "peak_mean",
    "event_probability",
    "endpoint_sd",
)
PHASES = ("baseline", "onset", "active")
PANELS = PHASES + ("state_average", "onset_minus_baseline")
METHOD_LABELS = {
    "direct_importance": "Direct importance",
    "direct_importance_n256": "Direct importance N=256",
    "terminal_smc": "Terminal SMC",
    "progressive_bridge_smc": "Progressive bridge SMC",
    "temporal_cut_smc": "Temporal-cut SMC",
    "sbtg_current": "SBTG-current",
    "sbtg_published": "SBTG-published",
}
COLORS = {
    "direct_importance": "#2563EB",
    "direct_importance_n256": "#60A5FA",
    "terminal_smc": "#D97706",
    "progressive_bridge_smc": "#059669",
    "temporal_cut_smc": "#DC2626",
    "sbtg_current": "#7C3AED",
    "sbtg_published": "#111827",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def method_key(method: str, channel: str, panel: str) -> str:
    return f"{method}__{channel}__{panel}"


def normalize_and_orient(
    response: np.ndarray, achieved_gap: np.ndarray
) -> np.ndarray:
    """Return [phase,worm,horizon,target,source] from sampler tensors.

    ``response`` is [worm,phase,event,source,horizon,target] and
    ``achieved_gap`` is [worm,phase,event,source].  Normalization is performed
    before averaging events, then the source/target axes are transposed once.
    """
    if response.ndim != 6 or achieved_gap.shape != response.shape[:4]:
        raise ValueError("response and achieved-gap shapes are inconsistent")
    normalized = response / np.maximum(np.abs(achieved_gap), 0.10)[..., None, None]
    return normalized.mean(axis=2).transpose(1, 0, 3, 4, 2)


def load_sampler_matrices(
    run_dir: Path,
    *,
    particles: int,
    seeds: tuple[int, ...],
    samplers: tuple[str, ...] = SAMPLERS,
) -> tuple[dict[str, dict[str, object]], tuple[str, ...], pd.DataFrame, list[Path]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    expected_lags = tuple(int(value) for value in manifest["source_lag_frames"])
    if int(manifest["particles"]) != particles:
        raise RuntimeError("requested particle count disagrees with run manifest")
    if manifest["lag_definition"] != "source-window end to prediction cut":
        raise RuntimeError("run manifest has an incompatible lag definition")
    if int(manifest["n_neurons"]) != 54 or int(manifest["n_worms"]) != 17:
        raise RuntimeError("analysis requires the frozen 17-worm/54-neuron cohort")
    if float(manifest["fps"]) != 4.0:
        raise RuntimeError("analysis requires the native 4-Hz cohort clock")
    if tuple(int(value) for value in manifest["horizon_frames"]) != (1,):
        raise RuntimeError("primary analysis requires the one-frame forecast")
    methods: dict[str, dict[str, object]] = {}
    validity_rows: list[dict[str, object]] = []
    input_files: list[Path] = []
    neuron_order: tuple[str, ...] | None = None
    all_worm_ids: set[str] | None = None
    checkpoint_by_cell: dict[tuple[int, int], tuple[str, str]] = {}
    worm_fold: dict[str, int] = {}

    for sampler in samplers:
        lag_panel_channel: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
        for lag in expected_lags:
            files = sorted(
                path
                for seed in seeds
                for path in (run_dir / "responses" / sampler).glob(
                    f"*__{sampler}__ell{lag}__N{particles}__f*__s{seed}.npz"
                )
            )
            if len(files) != 5 * len(seeds):
                raise RuntimeError(
                    f"expected {5 * len(seeds)} fold-seed archives for {sampler}, "
                    f"lag {lag}; found {len(files)}"
                )
            input_files.extend(files)
            worm_repeats: dict[str, list[tuple[dict[str, np.ndarray], np.ndarray]]] = {}
            seen_cells: set[tuple[int, int]] = set()
            checkpoint_hashes: set[str] = set()
            for path in files:
                with np.load(path, allow_pickle=False) as data:
                    if str(data["status"].item()) != "complete":
                        raise RuntimeError(f"incomplete response archive {path}")
                    if str(data["method"].item()) != sampler:
                        raise RuntimeError(f"method label mismatch in {path}")
                    if int(data["source_lag_frames"]) != lag:
                        raise RuntimeError(f"source-lag label mismatch in {path}")
                    if str(data["lag_definition"].item()) != "source-window end to prediction cut":
                        raise RuntimeError(f"lag-definition mismatch in {path}")
                    fold = int(data["fold"])
                    model_seed = int(data["seed"])
                    cell = (fold, model_seed)
                    if cell in seen_cells:
                        raise RuntimeError(f"duplicate fold-seed {cell} for {sampler}, lag {lag}")
                    seen_cells.add(cell)
                    checkpoint_hash = str(data["checkpoint_sha256"].item())
                    checkpoint = Path(str(data["checkpoint"].item()))
                    checkpoint_hashes.add(checkpoint_hash)
                    declared_checkpoint = checkpoint_by_cell.get(cell)
                    current_checkpoint = (str(checkpoint.resolve()), checkpoint_hash)
                    if declared_checkpoint is None:
                        if not checkpoint.exists() or sha256(checkpoint) != checkpoint_hash:
                            raise RuntimeError(
                                f"checkpoint path/hash audit failed for {path}"
                            )
                        checkpoint_by_cell[cell] = current_checkpoint
                    elif declared_checkpoint != current_checkpoint:
                        raise RuntimeError(
                            f"checkpoint changed across method/lag for fold-seed {cell}"
                        )
                    if int(data["n_particles"]) != particles:
                        raise RuntimeError(f"particle-count mismatch in {path}")
                    if int(data["history_frames"]) != int(manifest["history_frames"]):
                        raise RuntimeError(f"history-length mismatch in {path}")
                    if int(data["source_window_frames"]) != int(
                        manifest["source_window_frames"]
                    ):
                        raise RuntimeError(f"source-window mismatch in {path}")
                    if int(data["repair_frames"]) != lag + int(
                        data["source_window_frames"]
                    ):
                        raise RuntimeError(f"repair-prefix length mismatch in {path}")
                    if tuple(int(value) for value in data["horizon_frames"]) != (1,):
                        raise RuntimeError(f"forecast-horizon mismatch in {path}")
                    neurons = tuple(data["neurons"].astype(str))
                    if neuron_order is None:
                        neuron_order = neurons
                    elif neuron_order != neurons:
                        raise RuntimeError("neuron order changed across response archives")
                    phases = tuple(data["phase_names"].astype(str))
                    if phases != PHASES:
                        raise RuntimeError(f"phase order mismatch in {path}: {phases}")
                    bounds = data["source_window_bounds"]
                    cuts = data["cut_times"]
                    if not np.all(bounds[..., 1] - 1 == cuts - lag):
                        raise RuntimeError(f"source-window end does not match lag in {path}")
                    if not np.all(bounds[..., 1] - bounds[..., 0] == int(data["source_window_frames"])):
                        raise RuntimeError(f"source-window width mismatch in {path}")
                    gaps = data["diagnostic_achieved_gap"]
                    valid = data["diagnostic_valid"] > 0.5
                    oriented = {
                        channel: normalize_and_orient(
                            data[f"response_{channel}"], gaps
                        ).astype(np.float32)
                        for channel in CHANNELS
                    }
                    for position, worm_id in enumerate(data["worm_ids"].astype(str)):
                        previous_fold = worm_fold.get(worm_id)
                        if previous_fold is None:
                            worm_fold[worm_id] = fold
                        elif previous_fold != fold:
                            raise RuntimeError(
                                f"worm {worm_id} appears in multiple held-out folds"
                            )
                        channel_values = {
                            channel: oriented[channel][:, position]
                            for channel in CHANNELS
                        }
                        worm_repeats.setdefault(worm_id, []).append(
                            (channel_values, valid[position])
                        )
            expected_cells = {(fold, seed) for fold in range(5) for seed in seeds}
            if seen_cells != expected_cells:
                raise RuntimeError(f"fold-seed coverage mismatch for {sampler}, lag {lag}")
            if len(checkpoint_hashes) != 5 * len(seeds):
                raise RuntimeError(
                    f"expected one distinct checkpoint per fold-seed for {sampler}, lag {lag}"
                )
            worm_records: list[tuple[str, dict[str, np.ndarray], np.ndarray]] = []
            for worm_id, repeats in sorted(worm_repeats.items()):
                if len(repeats) != len(seeds):
                    raise RuntimeError(
                        f"worm {worm_id} has {len(repeats)} rather than {len(seeds)} seeds"
                    )
                channel_values = {
                    channel: np.mean(
                        [repeat[0][channel] for repeat in repeats],
                        axis=0,
                    ).astype(np.float32)
                    for channel in CHANNELS
                }
                valid = np.mean([repeat[1] for repeat in repeats], axis=0)
                worm_records.append((worm_id, channel_values, valid))
            worm_ids = tuple(item[0] for item in worm_records)
            if len(worm_ids) != len(set(worm_ids)):
                raise RuntimeError(f"held-out worms duplicated for {sampler}, lag {lag}")
            if all_worm_ids is None:
                all_worm_ids = set(worm_ids)
            elif all_worm_ids != set(worm_ids):
                raise RuntimeError("held-out worm coverage differs across method/lag")
            for channel in CHANNELS:
                values = np.stack([item[1][channel] for item in worm_records])
                # [worm,phase,horizon,target,source] ->
                # [phase,worm,horizon,target,source].
                phase_worm = values.transpose(1, 0, 2, 3, 4)
                for phase_index, panel in enumerate(PHASES):
                    lag_panel_channel[(lag, panel, channel)] = (
                        phase_worm[phase_index, :, 0],
                        np.stack([item[2][phase_index].mean(axis=0) for item in worm_records]),
                    )
                state_average = phase_worm[:, :, 0].mean(axis=0)
                lag_panel_channel[(lag, "state_average", channel)] = (
                    state_average,
                    np.stack([item[2].mean(axis=(0, 1)) for item in worm_records]),
                )
                onset_minus = phase_worm[1, :, 0] - phase_worm[0, :, 0]
                lag_panel_channel[(lag, "onset_minus_baseline", channel)] = (
                    onset_minus,
                    np.stack(
                        [
                            np.minimum(
                                item[2][1].mean(axis=0), item[2][0].mean(axis=0)
                            )
                            for item in worm_records
                        ]
                    ),
                )

        for channel in CHANNELS:
            for panel in PANELS:
                worm_matrices = np.stack(
                    [lag_panel_channel[(lag, panel, channel)][0] for lag in expected_lags]
                )
                validity = np.stack(
                    [
                        lag_panel_channel[(lag, panel, channel)][1].mean(axis=0)
                        for lag in expected_lags
                    ]
                )
                key = method_key(sampler, channel, panel)
                methods[key] = {
                    "method": sampler,
                    "channel": channel,
                    "panel": panel,
                    "lags": np.asarray(expected_lags, dtype=np.int16),
                    "worm_matrices": worm_matrices,
                    "matrices": worm_matrices.mean(axis=1),
                    "validity": validity,
                    "worm_ids": np.asarray(sorted(all_worm_ids)),
                    "family": "flow_repaired",
                }
                for lag_position, lag in enumerate(expected_lags):
                    validity_rows.append(
                        {
                            "method": sampler,
                            "channel": channel,
                            "panel": panel,
                            "lag_frames": lag,
                            "lag_seconds": lag / float(manifest["fps"]),
                            "mean_source_validity": float(validity[lag_position].mean()),
                            "n_sources_valid_ge_50pct": int(
                                np.sum(validity[lag_position] >= 0.50)
                            ),
                            "n_sources_valid_ge_80pct": int(
                                np.sum(validity[lag_position] >= 0.80)
                            ),
                        }
                    )

    if neuron_order is None or all_worm_ids is None:
        raise RuntimeError("no sampler results were loaded")
    if len(all_worm_ids) != int(manifest["n_worms"]):
        raise RuntimeError("held-out worm union does not equal declared cohort")
    return methods, neuron_order, pd.DataFrame(validity_rows), input_files


def add_sbtg_comparators(
    methods: dict[str, dict[str, object]],
    archive: Path,
    neurons: tuple[str, ...],
) -> None:
    comparison_lags = methods[
        method_key("direct_importance", "endpoint_mean", "state_average")
    ]["lags"]
    with np.load(archive, allow_pickle=False) as data:
        if tuple(data["neurons"].astype(str)) != neurons:
            raise RuntimeError("SBTG and corrected flow neuron orders differ")
        for method in ("sbtg_current", "sbtg_published"):
            matrix_key = (
                f"{method}__matrices"
                if f"{method}__matrices" in data.files
                else f"{method}__signed"
            )
            available_lags = data[f"{method}__lags"].astype(np.int16)
            retained_lags = np.asarray(
                sorted(set(comparison_lags.tolist()) & set(available_lags.tolist())),
                dtype=np.int16,
            )
            if retained_lags.size < 2:
                raise RuntimeError(f"SBTG comparator has fewer than two comparable lags: {method}")
            positions = np.asarray(
                [int(np.flatnonzero(available_lags == lag)[0]) for lag in retained_lags]
            )
            methods[method] = {
                "method": method,
                "channel": "score_product",
                "panel": "all_windows",
                "lags": retained_lags,
                "matrices": data[matrix_key][positions].astype(np.float32),
                "validity": np.ones(
                    (len(retained_lags), len(neurons)), dtype=np.float32
                ),
                "family": "sbtg",
            }


def correlation_metrics(
    matrix: np.ndarray,
    reference: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
    use = mask & np.isfinite(matrix)
    output = {
        "absolute_spearman": np.nan,
        "signed_spearman": np.nan,
        "absolute_spearman_log1p_weight": np.nan,
        "positive_edge_absolute_spearman": np.nan,
        "within_source_weight_spearman": np.nan,
    }
    if use.sum() < 3:
        return output
    if "weight" in reference:
        weight = np.asarray(reference["weight"])[use]
        score = np.abs(matrix[use])
        if np.unique(weight).size >= 2 and np.unique(score).size >= 2:
            output["absolute_spearman"] = float(
                spearmanr(score, weight).statistic
            )
            output["absolute_spearman_log1p_weight"] = float(
                spearmanr(score, np.log1p(weight)).statistic
            )
        positive = use & (np.asarray(reference["weight"]) > 0)
        if positive.sum() >= 3:
            output["positive_edge_absolute_spearman"] = float(
                spearmanr(
                    np.abs(matrix[positive]),
                    np.asarray(reference["weight"])[positive],
                ).statistic
            )
        source_values: list[float] = []
        for source in range(matrix.shape[1]):
            column = use[:, source]
            if column.sum() < 3:
                continue
            column_weight = np.asarray(reference["weight"])[column, source]
            if np.unique(column_weight).size < 2:
                continue
            value = spearmanr(
                np.abs(matrix[column, source]), column_weight
            ).statistic
            if np.isfinite(value):
                source_values.append(float(value))
        if source_values:
            output["within_source_weight_spearman"] = float(np.mean(source_values))
    if "signed_value" in reference:
        value = np.asarray(reference["signed_value"])[use]
        output["absolute_spearman"] = float(
            spearmanr(np.abs(matrix[use]), np.abs(value)).statistic
        )
        output["signed_spearman"] = float(
            spearmanr(matrix[use], value).statistic
        )
    return output


def evaluate_references(
    methods: dict[str, dict[str, object]],
    references: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, item in methods.items():
        for lag_position, (lag, matrix) in enumerate(zip(item["lags"], item["matrices"])):
            support = np.asarray(item["validity"])[lag_position] >= 0.50
            for reference_name, reference in references.items():
                for scope, mask in (
                    ("all_estimated", reference["mask"]),
                    (
                        "method_support_qualified",
                        reference["mask"] & support[None, :],
                    ),
                ):
                    rows.append(
                        {
                            "method_key": key,
                            "method": item["method"],
                            "method_label": METHOD_LABELS[item["method"]],
                            "family": item["family"],
                            "channel": item["channel"],
                            "panel": item["panel"],
                            "lag_frames": int(lag),
                            "lag_seconds": float(lag) / 4.0,
                            "reference": reference_name,
                            "scope": scope,
                            "n_support_sources": int(support.sum()),
                            **binary_metrics(matrix, reference["labels"], mask),
                            **source_macro_metrics(matrix, reference["labels"], mask),
                            **correlation_metrics(matrix, reference, mask),
                        }
                    )
    return pd.DataFrame(rows)


def evaluate_neuromodulators(
    methods: dict[str, dict[str, object]], networks: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    d = next(iter(networks.values())).shape[0]
    off = ~np.eye(d, dtype=bool)
    for key, item in methods.items():
        for lag_position, (lag, matrix) in enumerate(zip(item["lags"], item["matrices"])):
            support = np.asarray(item["validity"])[lag_position] >= 0.50
            for network_name, labels in networks.items():
                eligible = labels.any(axis=0)
                for scope, mask in (
                    ("all_pairs_legacy", off),
                    ("eligible_sources", off & eligible[None, :]),
                    (
                        "eligible_support_qualified",
                        off & eligible[None, :] & support[None, :],
                    ),
                ):
                    rows.append(
                        {
                            "method_key": key,
                            "method": item["method"],
                            "method_label": METHOD_LABELS[item["method"]],
                            "family": item["family"],
                            "channel": item["channel"],
                            "panel": item["panel"],
                            "lag_frames": int(lag),
                            "lag_seconds": float(lag) / 4.0,
                            "network": network_name,
                            "scope": scope,
                            "n_eligible_sources": int(eligible.sum()),
                            "n_supported_eligible_sources": int(
                                np.sum(eligible & support)
                            ),
                            **binary_metrics(matrix, labels, mask),
                            **source_macro_metrics(matrix, labels, mask),
                        }
                    )
    return pd.DataFrame(rows)


def lagmax_inference(
    methods: dict[str, dict[str, object]],
    networks: dict[str, np.ndarray],
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    flow_keys = [
        method_key(method, "endpoint_mean", "state_average") for method in SAMPLERS
    ]
    sensitivity_key = method_key(
        "direct_importance_n256", "endpoint_mean", "state_average"
    )
    if sensitivity_key in methods:
        flow_keys.append(sensitivity_key)
    selected_keys = flow_keys + ["sbtg_current", "sbtg_published"]
    rows: list[dict[str, object]] = []
    common_lags = sorted(
        set.intersection(*(set(methods[key]["lags"].tolist()) for key in selected_keys))
    )
    grids = (
        (
            "native_four_lag_flow_only",
            flow_keys,
            None,
        ),
        ("common_two_lag_all_methods", selected_keys, common_lags),
    )
    for lag_grid, grid_keys, forced_lags in grids:
        for key in grid_keys:
            item = methods[key]
            use_positions = (
                np.arange(len(item["lags"]), dtype=np.int64)
                if forced_lags is None
                else np.asarray(
                    [int(np.flatnonzero(item["lags"] == lag)[0]) for lag in forced_lags]
                )
            )
            use_lags = np.asarray(item["lags"])[use_positions]
            use_matrices = np.asarray(item["matrices"])[use_positions]
            for network_name in INFERENCE_NETWORKS:
                labels = networks[network_name]
                eligible = labels.any(axis=0)
                mask = (~np.eye(len(labels), dtype=bool)) & eligible[None, :]
                observed = np.asarray(
                    [
                        roc_auc_score(labels[mask], np.abs(matrix[mask]))
                        for matrix in use_matrices
                    ]
                )
                best_position = int(np.argmax(observed))
                null_max = np.empty(permutations, dtype=np.float64)
                for repeat in range(permutations):
                    permuted = permute_within_source(labels, eligible, rng)
                    null_max[repeat] = max(
                        roc_auc_score(permuted[mask], np.abs(matrix[mask]))
                        for matrix in use_matrices
                    )
                bootstrap_low = np.nan
                bootstrap_high = np.nan
                selection_rate = np.nan
                if str(item["family"]).startswith("flow_repaired") and bootstraps > 0:
                    worm_matrices = np.asarray(item["worm_matrices"])[use_positions]
                    n_worms = worm_matrices.shape[1]
                    bootstrap_max = np.empty(bootstraps, dtype=np.float64)
                    bootstrap_position = np.empty(bootstraps, dtype=np.int16)
                    for repeat in range(bootstraps):
                        sampled = rng.integers(0, n_worms, size=n_worms)
                        values = np.asarray(
                            [
                                roc_auc_score(
                                    labels[mask],
                                    np.abs(
                                        worm_matrices[lag_index, sampled]
                                        .mean(axis=0)[mask]
                                    ),
                                )
                                for lag_index in range(len(use_lags))
                            ]
                        )
                        bootstrap_position[repeat] = int(np.argmax(values))
                        bootstrap_max[repeat] = float(values.max())
                    bootstrap_low, bootstrap_high = np.quantile(
                        bootstrap_max, [0.025, 0.975]
                    )
                    selection_rate = float(
                        np.mean(bootstrap_position == best_position)
                    )
                rows.append(
                    {
                        "lag_grid": lag_grid,
                        "method_key": key,
                        "method": item["method"],
                        "method_label": METHOD_LABELS[item["method"]],
                        "family": item["family"],
                        "channel": item["channel"],
                        "panel": item["panel"],
                        "network": network_name,
                        "n_lags": len(observed),
                        "n_eligible_sources": int(eligible.sum()),
                        "best_lag_frames": int(use_lags[best_position]),
                        "best_lag_seconds": float(use_lags[best_position]) / 4.0,
                        "best_auroc": float(observed[best_position]),
                        "lagmax_p_value": float(
                            (1 + np.sum(null_max >= observed[best_position]))
                            / (permutations + 1)
                        ),
                        "null_max_95pct": float(np.quantile(null_max, 0.95)),
                        "worm_bootstrap_supported": str(item["family"]).startswith(
                            "flow_repaired"
                        ),
                        "worm_bootstrap_best_auroc_ci_low": float(bootstrap_low),
                        "worm_bootstrap_best_auroc_ci_high": float(bootstrap_high),
                        "best_lag_selection_rate": selection_rate,
                    }
                )
    result = pd.DataFrame(rows)
    result["lagmax_bh_q_value"] = benjamini_hochberg(
        result.lagmax_p_value.to_numpy()
    )
    return result


def matrix_relationships(methods: dict[str, dict[str, object]]) -> pd.DataFrame:
    keys = [
        method_key(method, "endpoint_mean", "state_average") for method in SAMPLERS
    ]
    sensitivity_key = method_key(
        "direct_importance_n256", "endpoint_mean", "state_average"
    )
    if sensitivity_key in methods:
        keys.append(sensitivity_key)
    keys += ["sbtg_current", "sbtg_published"]
    rows: list[dict[str, object]] = []
    for left_position, left_key in enumerate(keys):
        for right_key in keys[left_position + 1 :]:
            left, right = methods[left_key], methods[right_key]
            common = sorted(set(left["lags"].tolist()) & set(right["lags"].tolist()))
            for lag in common:
                a = left["matrices"][np.flatnonzero(left["lags"] == lag)[0]]
                b = right["matrices"][np.flatnonzero(right["lags"] == lag)[0]]
                off = ~np.eye(len(a), dtype=bool)
                av, bv = a[off], b[off]
                top = max(1, int(round(0.10 * len(av))))
                a_top = set(np.argpartition(np.abs(av), -top)[-top:])
                b_top = set(np.argpartition(np.abs(bv), -top)[-top:])
                rows.append(
                    {
                        "left": left_key,
                        "right": right_key,
                        "lag_frames": lag,
                        "lag_seconds": lag / 4.0,
                        "signed_spearman": float(spearmanr(av, bv).statistic),
                        "absolute_spearman": float(
                            spearmanr(np.abs(av), np.abs(bv)).statistic
                        ),
                        "top_10pct_jaccard": len(a_top & b_top) / len(a_top | b_top),
                    }
                )
    return pd.DataFrame(rows)


def generator_seed_stability(
    run_dir: Path,
    *,
    particles: int,
    seeds: tuple[int, ...],
    samplers: tuple[str, ...],
    method_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compare frozen primary matrices across independently trained generators."""
    if len(seeds) < 2:
        return pd.DataFrame()
    seed_methods: dict[int, dict[str, dict[str, object]]] = {}
    for seed in seeds:
        loaded, _, _, _ = load_sampler_matrices(
            run_dir,
            particles=particles,
            seeds=(seed,),
            samplers=samplers,
        )
        seed_methods[seed] = loaded
    aliases = method_aliases or {}
    rows: list[dict[str, object]] = []
    for left_position, left_seed in enumerate(seeds):
        for right_seed in seeds[left_position + 1 :]:
            for sampler in samplers:
                key = method_key(sampler, "endpoint_mean", "state_average")
                left = seed_methods[left_seed][key]
                right = seed_methods[right_seed][key]
                for lag in left["lags"]:
                    lag = int(lag)
                    left_matrix = left["matrices"][
                        np.flatnonzero(left["lags"] == lag)[0]
                    ]
                    right_matrix = right["matrices"][
                        np.flatnonzero(right["lags"] == lag)[0]
                    ]
                    off = ~np.eye(len(left_matrix), dtype=bool)
                    left_values = left_matrix[off]
                    right_values = right_matrix[off]
                    top = max(1, int(round(0.10 * len(left_values))))
                    left_top = set(
                        np.argpartition(np.abs(left_values), -top)[-top:]
                    )
                    right_top = set(
                        np.argpartition(np.abs(right_values), -top)[-top:]
                    )
                    rows.append(
                        {
                            "method": aliases.get(sampler, sampler),
                            "n_particles": particles,
                            "left_generator_seed": left_seed,
                            "right_generator_seed": right_seed,
                            "lag_frames": lag,
                            "lag_seconds": lag / 4.0,
                            "signed_spearman": float(
                                spearmanr(left_values, right_values).statistic
                            ),
                            "absolute_spearman": float(
                                spearmanr(
                                    np.abs(left_values), np.abs(right_values)
                                ).statistic
                            ),
                            "top_10pct_jaccard": len(left_top & right_top)
                            / len(left_top | right_top),
                            "matrix_rmse": float(
                                np.sqrt(np.mean((left_values - right_values) ** 2))
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def response_archive_audit(
    files: list[Path], *, sensitivity_root: Path | None = None
) -> pd.DataFrame:
    """Retain archive-level runtime and support diagnostics for auditability."""
    rows: list[dict[str, object]] = []
    manifest_cache: dict[Path, dict[str, object]] = {}
    sensitivity_root = sensitivity_root.resolve() if sensitivity_root else None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            method = str(data["method"].item())
            if sensitivity_root is not None and path.resolve().is_relative_to(
                sensitivity_root
            ):
                method = "direct_importance_n256"
            valid = np.asarray(data["diagnostic_valid"], dtype=np.float64)
            achieved_gap = np.asarray(
                data["diagnostic_achieved_gap"], dtype=np.float64
            )
            n_particles = int(data["n_particles"])
            run_root = path.parents[2]
            if run_root not in manifest_cache:
                manifest_cache[run_root] = json.loads(
                    (run_root / "manifest.json").read_text()
                )
            configured_minimum_ess = float(
                manifest_cache[run_root]["minimum_effective_sample_size"]
            )
            common_relative_valid = (
                (np.asarray(data["diagnostic_ess_low"]) >= 0.1875 * n_particles)
                & (np.asarray(data["diagnostic_ess_high"]) >= 0.1875 * n_particles)
                & (np.asarray(data["diagnostic_max_weight_low"]) <= 0.20)
                & (np.asarray(data["diagnostic_max_weight_high"]) <= 0.20)
                & (
                    achieved_gap
                    >= 0.25 * np.asarray(data["diagnostic_target_gap"])
                )
            )
            rows.append(
                {
                    "archive": str(path.resolve()),
                    "method": method,
                    "n_particles": n_particles,
                    "configured_minimum_ess": configured_minimum_ess,
                    "lag_frames": int(data["source_lag_frames"]),
                    "lag_seconds": float(data["source_lag_seconds"]),
                    "fold": int(data["fold"]),
                    "generator_seed": int(data["seed"]),
                    "n_held_out_worms": len(data["worm_ids"]),
                    "wall_seconds": float(data["wall_seconds"]),
                    "mean_episode_source_validity": float(valid.mean()),
                    "common_relative_ess_18p75pct_validity": float(
                        common_relative_valid.mean()
                    ),
                    "mean_absolute_achieved_gap": float(
                        np.abs(achieved_gap).mean()
                    ),
                    "checkpoint_sha256": str(data["checkpoint_sha256"].item()),
                }
            )
    return pd.DataFrame(rows)


def save_archive(
    path: Path,
    methods: dict[str, dict[str, object]],
    neurons: tuple[str, ...],
) -> None:
    values: dict[str, np.ndarray] = {"neurons": np.asarray(neurons)}
    for key, item in methods.items():
        values[f"{key}__lags"] = np.asarray(item["lags"])
        values[f"{key}__matrices"] = np.asarray(item["matrices"], dtype=np.float32)
        values[f"{key}__validity"] = np.asarray(item["validity"], dtype=np.float32)
        if "worm_matrices" in item:
            values[f"{key}__worm_matrices"] = np.asarray(
                item["worm_matrices"], dtype=np.float32
            )
            values[f"{key}__worm_ids"] = np.asarray(item["worm_ids"])
    np.savez_compressed(path, **values)


def plot_profiles(neuromod: pd.DataFrame, output: Path) -> None:
    methods = list(SAMPLERS) + ["sbtg_current", "sbtg_published"]
    if "direct_importance_n256" in set(neuromod.method):
        methods.insert(1, "direct_importance_n256")
    subset = neuromod[
        (neuromod.channel.isin(["endpoint_mean", "score_product"]))
        & (neuromod.panel.isin(["state_average", "all_windows"]))
        & (neuromod.scope == "eligible_sources")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, network in zip(axes, PRIMARY_NETWORKS):
        for method in methods:
            row = subset[(subset.method == method) & (subset.network == network)].sort_values(
                "lag_seconds"
            )
            axis.plot(
                row.lag_seconds,
                row.auroc,
                marker="o",
                linewidth=2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.axhline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
        axis.set_title(network.replace("_", " "))
        axis.set_xlabel("Source-window-end lag (s)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Bentley edge AUROC")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(output / "primary_neuromodulator_lag_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for axis, network in zip(axes, SPECIFIC_NETWORKS):
        for method in methods:
            row = subset[(subset.method == method) & (subset.network == network)].sort_values(
                "lag_seconds"
            )
            axis.plot(
                row.lag_seconds,
                row.auroc,
                marker="o",
                linewidth=2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.axhline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
        axis.set_title(network.removeprefix("monoamine_"))
        axis.set_xlabel("Lag (s)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Bentley edge AUROC")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(output / "specific_transmitter_lag_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    distributional = neuromod[
        (neuromod.method.isin(SAMPLERS))
        & (neuromod.channel.isin(["endpoint_mean", "endpoint_sd", "event_probability"]))
        & (neuromod.panel == "state_average")
        & (neuromod.scope == "eligible_sources")
        & (neuromod.network == "neuromodulator_union")
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    styles = {
        "endpoint_mean": ("Mean", "-"),
        "endpoint_sd": ("SD", "--"),
        "event_probability": ("Tail probability", ":"),
    }
    for axis, method in zip(axes, SAMPLERS):
        for channel, (label, style) in styles.items():
            row = distributional[
                (distributional.method == method) & (distributional.channel == channel)
            ].sort_values("lag_seconds")
            axis.plot(row.lag_seconds, row.auroc, marker="o", linestyle=style, label=label)
        axis.axhline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Lag (s)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Neuromodulator-union AUROC")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(output / "mean_vs_distributional_channels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lag_matrices(
    methods: dict[str, dict[str, object]], output: Path, *, channel: str
) -> None:
    sampler_names = list(SAMPLERS)
    if method_key(
        "direct_importance_n256", channel, "state_average"
    ) in methods:
        sampler_names.insert(1, "direct_importance_n256")
    items = [methods[method_key(name, channel, "state_average")] for name in sampler_names]
    values = np.concatenate(
        [np.asarray(item["matrices"]).reshape(-1) for item in items]
    )
    limit = float(np.quantile(np.abs(values[np.isfinite(values)]), 0.99))
    limit = max(limit, 1e-6)
    lags = items[0]["lags"]
    fig, axes = plt.subplots(
        len(items), len(lags), figsize=(3.25 * len(lags), 3.05 * len(items))
    )
    axes = np.atleast_2d(axes)
    image = None
    for row, (name, item) in enumerate(zip(sampler_names, items)):
        for column, lag in enumerate(lags):
            matrix = np.asarray(item["matrices"])[column].copy()
            np.fill_diagonal(matrix, 0.0)
            image = axes[row, column].imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(f"{float(lag) / 4.0:g} s")
            if column == 0:
                axes[row, column].set_ylabel(METHOD_LABELS[name])
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.55, label="Effect per achieved source SD")
    fig.suptitle(channel.replace("_", " "))
    fig.savefig(
        output / f"{channel}_lag_matrices.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-release", type=Path, required=True)
    parser.add_argument("--sbtg-archive", type=Path, required=True)
    parser.add_argument("--direct-sensitivity-run-dir", type=Path)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903])
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--inference-seed", type=int, default=20260829)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    methods, neurons, validity, input_files = load_sampler_matrices(
        run_dir, particles=args.particles, seeds=tuple(args.seeds)
    )
    sensitivity_files: list[Path] = []
    if args.direct_sensitivity_run_dir is not None:
        sensitivity, sensitivity_neurons, sensitivity_validity, sensitivity_files = (
            load_sampler_matrices(
                args.direct_sensitivity_run_dir.resolve(),
                particles=256,
                seeds=tuple(args.seeds),
                samplers=("direct_importance",),
            )
        )
        if sensitivity_neurons != neurons:
            raise RuntimeError("direct N=256 sensitivity neuron order mismatch")
        for key, item in sensitivity.items():
            renamed = key.replace(
                "direct_importance__", "direct_importance_n256__", 1
            )
            updated = dict(item)
            updated["method"] = "direct_importance_n256"
            updated["family"] = "flow_repaired_particle_sensitivity"
            methods[renamed] = updated
        sensitivity_validity = sensitivity_validity.copy()
        sensitivity_validity["method"] = "direct_importance_n256"
        validity = pd.concat([validity, sensitivity_validity], ignore_index=True)
    add_sbtg_comparators(methods, args.sbtg_archive.resolve(), neurons)
    references, networks = load_references(args.reference_release.resolve(), list(neurons))

    expected_sources = {
        "monoamine_all": 5,
        "monoamine_dopamine": 2,
        "monoamine_serotonin": 1,
        "monoamine_tyramine": 1,
        "monoamine_octopamine": 1,
        "neuropeptide_all": 30,
        "neuromodulator_union": 31,
    }
    actual_sources = {name: int(value.any(axis=0).sum()) for name, value in networks.items()}
    if actual_sources != expected_sources:
        raise RuntimeError(
            f"neuromodulator source-count audit failed: {actual_sources}"
        )
    synthetic = np.zeros((2, 2), dtype=np.float32)
    synthetic[0, 1] = 7.0
    orientation_audit = bool(synthetic.T[1, 0] == 7.0 and synthetic.T[0, 1] == 0.0)
    if not orientation_audit:
        raise RuntimeError("orientation spot-check failed")

    external = evaluate_references(methods, references)
    neuromod = evaluate_neuromodulators(methods, networks)
    lagmax = lagmax_inference(
        methods,
        networks,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        seed=args.inference_seed,
    )
    relationships = matrix_relationships(methods)
    stability = generator_seed_stability(
        run_dir,
        particles=args.particles,
        seeds=tuple(args.seeds),
        samplers=SAMPLERS,
    )
    if args.direct_sensitivity_run_dir is not None:
        sensitivity_stability = generator_seed_stability(
            args.direct_sensitivity_run_dir.resolve(),
            particles=256,
            seeds=tuple(args.seeds),
            samplers=("direct_importance",),
            method_aliases={"direct_importance": "direct_importance_n256"},
        )
        stability = pd.concat(
            [stability, sensitivity_stability], ignore_index=True
        )
    archive_audit = response_archive_audit(
        input_files + sensitivity_files,
        sensitivity_root=(
            args.direct_sensitivity_run_dir.resolve()
            if args.direct_sensitivity_run_dir is not None
            else None
        ),
    )

    validity.to_csv(output / "sampler_support_by_lag.csv", index=False)
    archive_audit.to_csv(output / "response_archive_audit.csv", index=False)
    stability.to_csv(output / "generator_seed_matrix_stability.csv", index=False)
    external.to_csv(output / "randi_cook_metrics_by_lag.csv", index=False)
    neuromod.to_csv(output / "neuromodulator_metrics_by_lag.csv", index=False)
    lagmax.to_csv(output / "neuromodulator_lagmax_inference.csv", index=False)
    relationships.to_csv(output / "matrix_relationships_by_lag.csv", index=False)
    save_archive(output / "aligned_four_sampler_lag_matrices.npz", methods, neurons)
    plot_profiles(neuromod, output)
    plot_lag_matrices(methods, output, channel="endpoint_mean")
    plot_lag_matrices(methods, output, channel="endpoint_sd")

    checksums = {
        str(path.resolve()): sha256(path)
        for path in sorted(
            set(input_files + sensitivity_files + [args.sbtg_archive.resolve()])
        )
    }
    (output / "input_checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    )
    validation = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_response_archives": len(input_files),
        "expected_response_archives": len(SAMPLERS) * 4 * 5 * len(args.seeds),
        "direct_sensitivity_response_archives": len(sensitivity_files),
        "generator_seeds": args.seeds,
        "n_neurons": len(neurons),
        "n_worms": 17,
        "matrix_orientation": "target,source",
        "internal_sampler_orientation": "source,horizon,target",
        "transpose_count_at_analysis_boundary": 1,
        "orientation_spot_check": orientation_audit,
        "off_diagonal_only_for_external_metrics": True,
        "neuromodulator_eligible_source_counts": actual_sources,
        "reference_release": str(args.reference_release.resolve()),
        "sbtg_archive": str(args.sbtg_archive.resolve()),
        "normalization": "response divided by max(abs(achieved high-low source gap), 0.10), event-wise before averaging",
        "primary_effect": "endpoint mean at one forecast frame (0.25 s)",
        "lag_definition": "source-window end to prediction cut",
        "atlas_firewall": "references loaded only after sampler matrices were frozen",
        "claim_boundary": "observational model-relative lag correspondence; neither receptor edges nor effects identify physical delays",
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    main()
