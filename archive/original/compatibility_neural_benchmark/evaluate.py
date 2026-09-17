from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "SBTG") not in sys.path:
    sys.path.insert(0, str(ROOT / "SBTG"))

from sid_elegans.newlevers.funatlas import load_funatlas


@dataclass
class MethodEffects:
    name: str
    fold_scores: np.ndarray  # [fold, target, source]
    lag_scores: np.ndarray  # [lag, target, source]
    signed_lag: np.ndarray  # [lag, target, source]
    lag_frames: np.ndarray
    compatibility: np.ndarray | None = None  # [source]
    semantic_scores: dict[str, np.ndarray] | None = None
    semantic_validity: dict[str, np.ndarray] | None = None
    semantic_config: dict[str, tuple[float, float]] | None = None
    rollout: dict[str, float] | None = None
    rollout_by_horizon: dict[str, np.ndarray] | None = None

    @property
    def score(self) -> np.ndarray:
        return self.fold_scores.mean(axis=0)


def _read_folds(path: Path) -> dict[int, int]:
    with path.open(newline="") as handle:
        return {int(row["worm_index"]): int(row["outer_fold"]) for row in csv.DictReader(handle)}


def load_repaired(
    run_dir: Path,
    model_id: str,
    repair_frames: int = 4,
    seeds: set[int] | None = None,
) -> MethodEffects:
    paths = sorted(
        (run_dir / "responses").glob(f"{model_id}__B{repair_frames}__f*__s*.npz")
    )
    if seeds is not None:
        paths = [
            path
            for path in paths
            if int(path.stem.rsplit("__s", 1)[1]) in seeds
        ]
    if not paths:
        raise FileNotFoundError(f"no repaired responses for {model_id}")
    by_fold: dict[int, list[dict[str, np.ndarray]]] = {}
    semantic_names: list[str] | None = None
    semantic_validity_accumulator: dict[str, list[np.ndarray]] = {}
    semantic_config: dict[str, tuple[float, float]] = {}
    rollout_values = {"energy": [], "rmse": [], "coverage90": []}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            fold = int(data["fold"])
            response = data["response_cumulative_mean"].astype(np.float64)
            gap = data["diagnostic_achieved_gap"].astype(np.float64)
            denominator = np.maximum(gap, 0.10)[:, :, :, None, None]
            response_per_unit = response / denominator
            fold_item = {
                "response": response_per_unit,
                "valid": data["diagnostic_valid"].astype(np.float64),
                "semantic_score": data["semantic_score"].astype(np.float64),
            }
            by_fold.setdefault(fold, []).append(fold_item)
            if semantic_names is None:
                semantic_names = [str(x) for x in data["semantic_names"]]
                semantic_config = {
                    name: (
                        float(data["semantic_anchor_lambda"][index]),
                        float(data["semantic_epsilon_fraction"][index]),
                    )
                    for index, name in enumerate(semantic_names)
                }
                semantic_validity_accumulator = {name: [] for name in semantic_names}
            for semantic_index, name in enumerate(semantic_names):
                semantic_validity_accumulator[name].append(
                    data["semantic_valid_rate"][semantic_index].astype(np.float64)
                )
            rollout_values["energy"].append(float(data["rollout_energy"].mean()))
            rollout_values["rmse"].append(float(data["rollout_rmse"].mean()))
            rollout_values["coverage90"].append(float(data["rollout_coverage90"].mean()))
            for key, array_key in (
                ("energy_by_horizon", "rollout_energy"),
                ("rmse_by_horizon", "rollout_rmse"),
                ("coverage90_by_horizon", "rollout_coverage90"),
            ):
                rollout_values.setdefault(key, []).append(
                    data[array_key].astype(np.float64).mean(axis=(0, 1))
                )
            lag_frames = data["horizon_frames"].astype(np.int64)

    fold_scores = []
    fold_lag = []
    fold_signed = []
    fold_compatibility = []
    semantic_accumulator: dict[str, list[np.ndarray]] = {
        name: [] for name in (semantic_names or [])
    }
    for fold in sorted(by_fold):
        seed_responses = []
        seed_valid = []
        for item in by_fold[fold]:
            response = item["response"]  # [worm, phase, source, lag, target]
            seed_responses.append(response.mean(axis=(0, 1)))  # [source, lag, target]
            seed_valid.append(item["valid"].mean(axis=(0, 1)))
            for semantic_index, name in enumerate(semantic_names or []):
                semantic_accumulator[name].append(item["semantic_score"][semantic_index].T)
        signed_source_lag_target = np.mean(seed_responses, axis=0)
        signed = signed_source_lag_target.transpose(1, 2, 0)  # [lag,target,source]
        lag_abs = np.mean(
            [np.mean(np.abs(item["response"]), axis=(0, 1)).transpose(1, 2, 0) for item in by_fold[fold]],
            axis=0,
        )
        fold_lag.append(lag_abs)
        fold_signed.append(signed)
        fold_scores.append(lag_abs.max(axis=0))
        fold_compatibility.append(np.mean(seed_valid, axis=0))
    semantic_scores = {
        name: np.mean(values, axis=0) for name, values in semantic_accumulator.items()
    }
    return MethodEffects(
        name=model_id,
        fold_scores=np.asarray(fold_scores),
        lag_scores=np.mean(fold_lag, axis=0),
        signed_lag=np.mean(fold_signed, axis=0),
        lag_frames=lag_frames,
        compatibility=np.mean(fold_compatibility, axis=0),
        semantic_scores=semantic_scores,
        semantic_validity={
            name: np.mean(values, axis=0)
            for name, values in semantic_validity_accumulator.items()
        },
        semantic_config=semantic_config,
        rollout={
            key: float(np.mean(values))
            for key, values in rollout_values.items()
            if not key.endswith("_by_horizon")
        },
        rollout_by_horizon={
            key.removesuffix("_by_horizon"): np.mean(values, axis=0)
            for key, values in rollout_values.items()
            if key.endswith("_by_horizon")
        },
    )


def load_worm_integrated_effects(
    run_dir: Path, model_id: str, n_worms: int
) -> tuple[np.ndarray, np.ndarray]:
    by_worm_effect: dict[int, list[np.ndarray]] = {i: [] for i in range(n_worms)}
    by_worm_valid: dict[int, list[np.ndarray]] = {i: [] for i in range(n_worms)}
    paths = sorted((run_dir / "responses").glob(f"{model_id}__B4__f*__s*.npz"))
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            response = data["response_cumulative_mean"].astype(np.float64)
            gap = data["diagnostic_achieved_gap"].astype(np.float64)
            normalized = response / np.maximum(gap, 0.10)[:, :, :, None, None]
            # Primary pairwise statistic is the signed mean across prespecified
            # phases and horizons. No atlas labels or pair outcomes select a lag.
            integrated = normalized.mean(axis=(1, 3)).transpose(0, 2, 1)
            valid = data["diagnostic_valid"].astype(np.float64).mean(axis=1)
            for position, worm in enumerate(data["worm_indices"].astype(int)):
                by_worm_effect[worm].append(integrated[position])
                by_worm_valid[worm].append(valid[position])
    if any(not values for values in by_worm_effect.values()):
        missing = [worm for worm, values in by_worm_effect.items() if not values]
        raise RuntimeError(f"missing repaired effects for worms {missing}")
    effects = np.stack(
        [np.mean(by_worm_effect[worm], axis=0) for worm in range(n_worms)]
    )
    validity = np.stack(
        [np.mean(by_worm_valid[worm], axis=0) for worm in range(n_worms)]
    )
    return effects, validity


def load_worm_lag_effects(
    run_dir: Path, model_id: str, n_worms: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-animal signed effects as [worm, lag, target, source]."""
    by_worm_effect: dict[int, list[np.ndarray]] = {i: [] for i in range(n_worms)}
    by_worm_valid: dict[int, list[np.ndarray]] = {i: [] for i in range(n_worms)}
    paths = sorted((run_dir / "responses").glob(f"{model_id}__B4__f*__s*.npz"))
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            response = data["response_cumulative_mean"].astype(np.float64)
            gap = data["diagnostic_achieved_gap"].astype(np.float64)
            normalized = response / np.maximum(gap, 0.10)[:, :, :, None, None]
            # Average only over the prespecified phases here, leaving horizon
            # intact for the lag-specific family of tests.
            lagged = normalized.mean(axis=1).transpose(0, 2, 3, 1)
            valid = data["diagnostic_valid"].astype(np.float64).mean(axis=1)
            for position, worm in enumerate(data["worm_indices"].astype(int)):
                by_worm_effect[worm].append(lagged[position])
                by_worm_valid[worm].append(valid[position])
    if any(not values for values in by_worm_effect.values()):
        missing = [worm for worm, values in by_worm_effect.items() if not values]
        raise RuntimeError(f"missing lag-resolved repaired effects for worms {missing}")
    effects = np.stack(
        [np.mean(by_worm_effect[worm], axis=0) for worm in range(n_worms)]
    )
    validity = np.stack(
        [np.mean(by_worm_valid[worm], axis=0) for worm in range(n_worms)]
    )
    return effects, validity


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def pairwise_animal_signflip(
    effects: np.ndarray, *, n_perm: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Animal-block sign-flip test of the prespecified integrated pair effect."""
    effects = np.asarray(effects, dtype=np.float64)
    n_worms, d, _ = effects.shape
    mean = effects.mean(axis=0)
    se = effects.std(axis=0, ddof=1) / np.sqrt(n_worms)
    observed = np.abs(mean)
    exceed = np.zeros((d, d), dtype=np.int64)
    rng = np.random.default_rng(seed)
    remaining = int(n_perm)
    while remaining > 0:
        batch = min(250, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(batch, n_worms))
        null = np.einsum("bw,wts->bts", signs, effects, optimize=True) / n_worms
        exceed += np.sum(np.abs(null) >= observed[None], axis=0)
        remaining -= batch
    p = (exceed + 1.0) / (n_perm + 1.0)
    off = ~np.eye(d, dtype=bool)
    q = np.ones((d, d), dtype=np.float64)
    q[off] = _benjamini_hochberg(p[off])
    return mean, se, p, q, off


def lagged_animal_signflip(
    effects: np.ndarray, *, n_perm: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sign-flip all horizon-by-pair effects with one global BH family."""
    effects = np.asarray(effects, dtype=np.float64)
    n_worms, n_lags, d, _ = effects.shape
    mean = effects.mean(axis=0)
    se = effects.std(axis=0, ddof=1) / np.sqrt(n_worms)
    observed = np.abs(mean)
    exceed = np.zeros_like(mean, dtype=np.int64)
    rng = np.random.default_rng(seed)
    remaining = int(n_perm)
    while remaining > 0:
        batch = min(100, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(batch, n_worms))
        null = np.tensordot(signs, effects, axes=(1, 0)) / n_worms
        exceed += np.sum(np.abs(null) >= observed[None], axis=0)
        remaining -= batch
    p = (exceed + 1.0) / (n_perm + 1.0)
    test_mask = np.broadcast_to(~np.eye(d, dtype=bool), (n_lags, d, d))
    q = np.ones_like(p)
    q[test_mask] = _benjamini_hochberg(p[test_mask])
    return mean, se, p, q, test_mask


def load_sbtg(run_dir: Path, field: str, name: str) -> MethodEffects:
    paths = sorted((run_dir / "fold_lag").glob("sbtg_feature_bilinear__f*__lag*__s*.npz"))
    if not paths:
        raise FileNotFoundError("no SBTG outputs")
    by_fold: dict[int, dict[int, np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            by_fold.setdefault(int(data["fold"]), {})[int(data["lag"])] = data[field].astype(np.float64)
    lags = np.asarray(sorted(next(iter(by_fold.values()))), dtype=np.int64)
    fold_signed = []
    for fold in sorted(by_fold):
        if sorted(by_fold[fold]) != lags.tolist():
            raise RuntimeError(f"fold {fold} has an incomplete SBTG lag grid")
        fold_signed.append(np.stack([by_fold[fold][int(lag)] for lag in lags]))
    signed = np.asarray(fold_signed)
    lag_scores_by_fold = np.abs(signed)
    return MethodEffects(
        name=name,
        fold_scores=lag_scores_by_fold.max(axis=1),
        lag_scores=lag_scores_by_fold.mean(axis=0),
        signed_lag=signed.mean(axis=0),
        lag_frames=lags,
    )


def _preferred_signed(method: MethodEffects) -> tuple[np.ndarray, np.ndarray]:
    index = np.argmax(np.abs(method.signed_lag), axis=0)
    signed = np.take_along_axis(method.signed_lag, index[None], axis=0)[0]
    preferred = method.lag_frames[index]
    return signed, preferred


def evaluate_method(
    method: MethodEffects,
    atlas: dict,
    *,
    source_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    mask = np.asarray(atlas["eval_mask"], bool).copy()
    if source_mask is not None:
        mask &= np.asarray(source_mask, bool)[None]
    labels = atlas["positive"][mask] > 0
    scores = method.score[mask]
    signed, preferred = _preferred_signed(method)
    result: dict[str, float | int] = {
        "n_edges": int(mask.sum()),
        "n_positive": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else np.nan,
        "auprc": float(average_precision_score(labels, scores)) if labels.any() else np.nan,
        "prevalence": float(labels.mean()) if len(labels) else np.nan,
    }
    positive = mask & (atlas["positive"] > 0) & np.isfinite(atlas["dFF"])
    if positive.sum() >= 3:
        result["signed_spearman"] = float(
            spearmanr(signed[positive], atlas["dFF"][positive]).statistic
        )
        result["sign_agreement"] = float(
            np.mean(np.sign(signed[positive]) == np.sign(atlas["dFF"][positive]))
        )
    else:
        result["signed_spearman"] = np.nan
        result["sign_agreement"] = np.nan
    kinetic = mask & np.isfinite(atlas["timescale"])
    if kinetic.sum() >= 3 and np.unique(preferred[kinetic]).size > 1:
        result["timescale_spearman"] = float(
            spearmanr(preferred[kinetic] / 4.0, atlas["timescale"][kinetic]).statistic
        )
        result["n_kinetic"] = int(kinetic.sum())
    else:
        result["timescale_spearman"] = np.nan
        result["n_kinetic"] = int(kinetic.sum())
    top_precision = []
    for source in range(method.score.shape[1]):
        candidates = np.flatnonzero(mask[:, source])
        if len(candidates) == 0:
            continue
        take = candidates[np.argsort(method.score[candidates, source])[-min(10, len(candidates)) :]]
        top_precision.append(float(np.mean(atlas["positive"][take, source] > 0)))
    result["precision_at_10"] = float(np.mean(top_precision)) if top_precision else np.nan
    return result


def bootstrap_metrics(
    method: MethodEffects,
    atlas: dict,
    *,
    n_boot: int,
    seed: int,
    source_mask: np.ndarray | None = None,
) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    values = {"auroc": [], "auprc": []}
    n_folds = len(method.fold_scores)
    original = method.fold_scores
    for _ in range(n_boot):
        index = rng.integers(0, n_folds, size=n_folds)
        method.fold_scores = original[index]
        metric = evaluate_method(method, atlas, source_mask=source_mask)
        for key in values:
            values[key].append(float(metric[key]))
    method.fold_scores = original
    observed = evaluate_method(method, atlas, source_mask=source_mask)
    return {
        key: (
            float(observed[key]),
            float(np.nanquantile(array, 0.025)),
            float(np.nanquantile(array, 0.975)),
        )
        for key, array in ((key, np.asarray(value)) for key, value in values.items())
    }


def paired_bootstrap_difference(
    left: MethodEffects,
    right: MethodEffects,
    atlas: dict,
    *,
    n_boot: int,
    seed: int,
    source_mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    if len(left.fold_scores) != len(right.fold_scores):
        raise ValueError("paired methods need the same outer-fold count")
    rng = np.random.default_rng(seed)
    left_original, right_original = left.fold_scores, right.fold_scores
    n = len(left_original)
    values = {"auroc": [], "auprc": []}
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        left.fold_scores, right.fold_scores = left_original[index], right_original[index]
        lm = evaluate_method(left, atlas, source_mask=source_mask)
        rm = evaluate_method(right, atlas, source_mask=source_mask)
        for key in values:
            values[key].append(float(lm[key]) - float(rm[key]))
    left.fold_scores, right.fold_scores = left_original, right_original
    observed_left = evaluate_method(left, atlas, source_mask=source_mask)
    observed_right = evaluate_method(right, atlas, source_mask=source_mask)
    result = {}
    for key, samples in values.items():
        array = np.asarray(samples)
        result[key] = {
            "difference": float(observed_left[key]) - float(observed_right[key]),
            "ci_low": float(np.nanquantile(array, 0.025)),
            "ci_high": float(np.nanquantile(array, 0.975)),
            "p_left_not_better": float((1 + np.sum(array <= 0)) / (len(array) + 1)),
        }
    return result


def source_permutation_test(
    method: MethodEffects, atlas: dict, *, n_perm: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    observed = evaluate_method(method, atlas)["auroc"]
    original = method.fold_scores.copy()
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(original.shape[-1])
        method.fold_scores = original[:, :, perm]
        null.append(float(evaluate_method(method, atlas)["auroc"]))
    method.fold_scores = original
    null_array = np.asarray(null)
    return {
        "observed": float(observed),
        "null_mean": float(null_array.mean()),
        "null_sd": float(null_array.std(ddof=1)),
        "p_greater": float((1 + np.sum(null_array >= observed)) / (n_perm + 1)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def create_figures(methods: list[MethodEffects], metric_rows: list[dict], output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary = [row for row in metric_rows if row["scope"] == "all_confirmed"]
    names = [row["method"] for row in primary]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in zip(axes, ("auroc", "auprc"), ("Atlas AUROC", "Atlas AUPRC")):
        values = [row[metric] for row in primary]
        low = [row[f"{metric}_ci_low"] for row in primary]
        high = [row[f"{metric}_ci_high"] for row in primary]
        ax.errorbar(values, x, xerr=[np.asarray(values) - low, np.asarray(high) - values], fmt="o", capsize=3)
        ax.set_yticks(x, names)
        ax.axvline(primary[0]["prevalence"] if metric == "auprc" else 0.5, color="#999999", ls="--")
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "atlas_method_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in methods:
        lag_auc = []
        original = method.fold_scores
        for lag_score in method.lag_scores:
            method.fold_scores = np.repeat(lag_score[None], len(original), axis=0)
            lag_auc.append(evaluate_method(method, _ATLAS_FOR_FIGURE)["auroc"])
        method.fold_scores = original
        ax.plot(method.lag_frames / 4.0, lag_auc, marker="o", label=method.name)
    ax.axhline(0.5, color="#999999", ls="--")
    ax.set(xlabel="Future horizon / SBTG lag (seconds)", ylabel="Atlas AUROC")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "atlas_auroc_by_lag.png", dpi=180)
    plt.close(fig)

    neural = [method for method in methods if method.rollout_by_horizon is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for method in neural:
        seconds = method.lag_frames / 4.0
        axes[0].plot(
            seconds,
            method.rollout_by_horizon["energy"],
            marker="o",
            label=method.name,
        )
        axes[1].plot(
            seconds,
            method.rollout_by_horizon["coverage90"],
            marker="o",
            label=method.name,
        )
    axes[0].set(xlabel="Future horizon (seconds)", ylabel="Held-out rollout energy")
    axes[1].set(xlabel="Future horizon (seconds)", ylabel="Marginal 90% coverage")
    axes[1].axhline(0.90, color="#999999", ls="--")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "heldout_rollout_audit.png", dpi=180)
    plt.close(fig)

    if len(neural) >= 2:
        order = np.argsort(np.minimum(neural[0].compatibility, neural[1].compatibility))
        fig, ax = plt.subplots(figsize=(11, 4.8))
        x = np.arange(len(order))
        width = 0.42
        ax.bar(x - width / 2, neural[0].compatibility[order], width, label=neural[0].name)
        ax.bar(x + width / 2, neural[1].compatibility[order], width, label=neural[1].name)
        ax.axhline(0.50, color="#444444", ls="--", label="strict support threshold")
        ax.set(
            xticks=x,
            xticklabels=np.asarray(_ATLAS_FOR_FIGURE["neuron_names"])[order],
            ylabel="Compatibility-valid query rate",
            xlabel="Source neuron (sorted by joint support)",
            ylim=(0, 1),
        )
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.legend(frameon=False, ncol=3)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / "compatibility_by_source.png", dpi=180)
        plt.close(fig)


_ATLAS_FOR_FIGURE: dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-run", type=Path, required=True)
    parser.add_argument("--sbtg-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sensitivity-run-8", type=Path)
    parser.add_argument("--sensitivity-run-16", type=Path)
    parser.add_argument("--particle-sensitivity-run", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads((args.source_run / "manifest.json").read_text())
    neurons = source_manifest["neurons"]
    atlas = load_funatlas(neurons, verbose=True)
    if atlas is None:
        raise RuntimeError("Randi/Leifer atlas is unavailable")
    global _ATLAS_FOR_FIGURE
    _ATLAS_FOR_FIGURE = atlas

    flow = load_repaired(args.response_run, "tcn_delta_flow_matching")
    mdn = load_repaired(args.response_run, "tcn_delta_mdn4")
    sbtg = load_sbtg(args.sbtg_run, "mu_hat", "SBTG-FeatureBilinear")
    pearson = load_sbtg(args.sbtg_run, "pearson", "lagged Pearson")
    methods = [flow, mdn, sbtg, pearson]
    flow_compatible = flow.compatibility >= 0.50
    mdn_compatible = mdn.compatibility >= 0.50
    compatible = flow_compatible & mdn_compatible
    scopes = [
        ("all_confirmed", None),
        ("flow_compatible_sources", flow_compatible),
        ("mdn4_compatible_sources", mdn_compatible),
        ("common_compatible_sources", compatible),
    ]
    metric_rows: list[dict] = []
    for scope_name, source_mask in scopes:
        for method in methods:
            metric = evaluate_method(method, atlas, source_mask=source_mask)
            boot = bootstrap_metrics(
                method,
                atlas,
                n_boot=args.bootstrap,
                seed=args.seed + len(metric_rows),
                source_mask=source_mask,
            )
            metric_rows.append(
                {
                    "scope": scope_name,
                    "method": method.name,
                    **metric,
                    "auroc_ci_low": boot["auroc"][1],
                    "auroc_ci_high": boot["auroc"][2],
                    "auprc_ci_low": boot["auprc"][1],
                    "auprc_ci_high": boot["auprc"][2],
                    "n_compatible_sources": int(source_mask.sum())
                    if source_mask is not None
                    else len(neurons),
                }
            )
    _write_csv(output / "method_metrics.csv", metric_rows)

    lag_rows = []
    for method in methods:
        original = method.fold_scores
        for lag_index, lag in enumerate(method.lag_frames):
            method.fold_scores = np.repeat(method.lag_scores[lag_index][None], len(original), axis=0)
            metric = evaluate_method(method, atlas)
            lag_rows.append(
                {"method": method.name, "lag_frames": int(lag), "lag_seconds": lag / 4.0, **metric}
            )
        method.fold_scores = original
    _write_csv(output / "lag_metrics.csv", lag_rows)

    rollout_rows = []
    for method in (flow, mdn):
        for lag_index, lag in enumerate(method.lag_frames):
            rollout_rows.append(
                {
                    "method": method.name,
                    "horizon_frames": int(lag),
                    "horizon_seconds": float(lag / 4.0),
                    "energy": float(method.rollout_by_horizon["energy"][lag_index]),
                    "rmse": float(method.rollout_by_horizon["rmse"][lag_index]),
                    "coverage90": float(
                        method.rollout_by_horizon["coverage90"][lag_index]
                    ),
                }
            )
    _write_csv(output / "rollout_metrics_by_horizon.csv", rollout_rows)

    comparisons = {}
    for scope_index, (scope_name, source_mask) in enumerate(scopes):
        comparisons[scope_name] = {}
        for candidate in (flow, mdn, pearson):
            comparisons[scope_name][f"{candidate.name}_minus_SBTG"] = (
                paired_bootstrap_difference(
                    candidate,
                    sbtg,
                    atlas,
                    n_boot=args.bootstrap,
                    seed=args.seed + 100 + scope_index,
                    source_mask=source_mask,
                )
            )
    permutation = {
        method.name: source_permutation_test(
            method, atlas, n_perm=args.permutations, seed=args.seed + i
        )
        for i, method in enumerate(methods)
    }
    (output / "inference.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "paired_outer_fold_bootstrap_by_scope": comparisons,
                "source_label_permutation": permutation,
                "common_compatible_sources": np.asarray(neurons)[compatible].tolist(),
                "compatibility_threshold": 0.50,
            },
            indent=2,
            sort_keys=True,
        )
    )

    compatibility_rows = [
        {
            "neuron": neuron,
            "flow_valid_rate": float(flow.compatibility[i]),
            "mdn4_valid_rate": float(mdn.compatibility[i]),
            "common_compatible": bool(compatible[i]),
        }
        for i, neuron in enumerate(neurons)
    ]
    _write_csv(output / "compatibility_by_source.csv", compatibility_rows)

    semantic_rows = []
    offdiagonal = ~np.eye(len(neurons), dtype=bool)
    for method in (flow, mdn):
        primary_semantic = method.semantic_scores["primary"]
        for name, score in method.semantic_scores.items():
            anchor_lambda, epsilon_fraction = method.semantic_config[name]
            semantic_rows.append(
                {
                    "model": method.name,
                    "semantic": name,
                    "anchor_lambda": anchor_lambda,
                    "epsilon_iqr_fraction": epsilon_fraction,
                    "mean_valid_source_rate": float(
                        method.semantic_validity[name].mean()
                    ),
                    "median_valid_source_rate": float(
                        np.median(method.semantic_validity[name])
                    ),
                    "offdiagonal_score_spearman_vs_primary": 1.0
                    if name == "primary"
                    else float(
                        spearmanr(
                            primary_semantic[offdiagonal], score[offdiagonal]
                        ).statistic
                    ),
                    "median_absolute_score_ratio_vs_primary": float(
                        np.median(
                            np.abs(score[offdiagonal])
                            / np.maximum(np.abs(primary_semantic[offdiagonal]), 1e-8)
                        )
                    ),
                }
            )
    _write_csv(output / "repair_semantic_sensitivity.csv", semantic_rows)

    repair_rows = []
    for repair_frames, sensitivity_run in (
        (4, args.response_run),
        (8, args.sensitivity_run_8),
        (16, args.sensitivity_run_16),
    ):
        if sensitivity_run is None:
            continue
        for model_id in ("tcn_delta_flow_matching", "tcn_delta_mdn4"):
            sensitivity_method = load_repaired(
                sensitivity_run,
                model_id,
                repair_frames,
                seeds={1701} if repair_frames == 4 else None,
            )
            repair_rows.append(
                {
                    "model": model_id,
                    "repair_frames": repair_frames,
                    "repair_seconds": repair_frames / 4.0,
                    **evaluate_method(sensitivity_method, atlas),
                    "mean_source_valid_rate": float(sensitivity_method.compatibility.mean()),
                    "median_source_valid_rate": float(np.median(sensitivity_method.compatibility)),
                }
            )
    _write_csv(output / "repair_length_sensitivity.csv", repair_rows)

    particle_rows = []
    if args.particle_sensitivity_run is not None:
        primary_particle_count = json.loads(
            (args.response_run / "manifest.json").read_text()
        )["config"]["n_particles"]
        sensitivity_particle_count = json.loads(
            (args.particle_sensitivity_run / "manifest.json").read_text()
        )["config"]["n_particles"]
        for model_id in ("tcn_delta_flow_matching", "tcn_delta_mdn4"):
            base = load_repaired(args.response_run, model_id, 4, seeds={1701})
            high_particle = load_repaired(args.particle_sensitivity_run, model_id, 4)
            base_metric = evaluate_method(base, atlas)
            high_metric = evaluate_method(high_particle, atlas)
            off = ~np.eye(len(neurons), dtype=bool)
            particle_rows.append(
                {
                    "model": model_id,
                    "base_seed": 1701,
                    "base_particles": int(primary_particle_count),
                    "sensitivity_particles": int(sensitivity_particle_count),
                    "offdiagonal_score_spearman": float(
                        spearmanr(base.score[off], high_particle.score[off]).statistic
                    ),
                    "base_auroc": base_metric["auroc"],
                    "sensitivity_auroc": high_metric["auroc"],
                    "auroc_difference": high_metric["auroc"] - base_metric["auroc"],
                    "base_auprc": base_metric["auprc"],
                    "sensitivity_auprc": high_metric["auprc"],
                    "auprc_difference": high_metric["auprc"] - base_metric["auprc"],
                    "base_mean_valid_rate": float(base.compatibility.mean()),
                    "sensitivity_mean_valid_rate": float(
                        high_particle.compatibility.mean()
                    ),
                }
            )
    _write_csv(output / "particle_count_sensitivity.csv", particle_rows)

    pair_summary = {}
    top_pair_rows: list[dict] = []
    top_lag_rows: list[dict] = []
    top_atlas_positive_lag_rows: list[dict] = []
    for model_index, method in enumerate((flow, mdn)):
        worm_effects, worm_validity = load_worm_integrated_effects(
            args.response_run, method.name, int(source_manifest["n_worms"])
        )
        mean, se, p_value, q_value, off = pairwise_animal_signflip(
            worm_effects,
            n_perm=max(args.permutations, 2000),
            seed=args.seed + 500 + model_index,
        )
        np.savez_compressed(
            output / f"{method.name}__animal_pair_effects.npz",
            neurons=np.asarray(neurons),
            worm_effects=worm_effects.astype(np.float32),
            worm_source_validity=worm_validity.astype(np.float32),
            mean_effect=mean.astype(np.float32),
            standard_error=se.astype(np.float32),
            p_value=p_value.astype(np.float32),
            q_value=q_value.astype(np.float32),
        )
        pair_rows = []
        for target in range(len(neurons)):
            for source in range(len(neurons)):
                if target == source:
                    continue
                pair_rows.append(
                    {
                        "target": neurons[target],
                        "source": neurons[source],
                        "mean_effect_per_achieved_source_unit": float(mean[target, source]),
                        "animal_se": float(se[target, source]),
                        "ci_low_normal": float(mean[target, source] - 1.96 * se[target, source]),
                        "ci_high_normal": float(mean[target, source] + 1.96 * se[target, source]),
                        "signflip_p": float(p_value[target, source]),
                        "bh_q": float(q_value[target, source]),
                        "source_valid_rate": float(worm_validity[:, source].mean()),
                        "atlas_confirmed": bool(atlas["eval_mask"][target, source]),
                        "atlas_positive": bool(atlas["positive"][target, source] > 0),
                        "atlas_dff": float(atlas["dFF"][target, source]) if np.isfinite(atlas["dFF"][target, source]) else "",
                    }
                )
        _write_csv(output / f"{method.name}__pairwise_tests.csv", pair_rows)
        compatible_pairs = [row for row in pair_rows if row["source_valid_rate"] >= 0.50]
        compatible_pairs.sort(
            key=lambda row: (row["bh_q"], -abs(row["mean_effect_per_achieved_source_unit"]))
        )
        for rank, row in enumerate(compatible_pairs[:25], start=1):
            top_pair_rows.append({"model": method.name, "rank": rank, **row})

        worm_lag_effects, lag_validity = load_worm_lag_effects(
            args.response_run, method.name, int(source_manifest["n_worms"])
        )
        lag_mean, lag_se, lag_p, lag_q, lag_test_mask = lagged_animal_signflip(
            worm_lag_effects,
            n_perm=max(args.permutations, 2000),
            seed=args.seed + 700 + model_index,
        )
        np.savez_compressed(
            output / f"{method.name}__lagged_animal_pair_effects.npz",
            neurons=np.asarray(neurons),
            horizon_frames=method.lag_frames,
            horizon_seconds=method.lag_frames / 4.0,
            worm_lag_effects=worm_lag_effects.astype(np.float32),
            worm_source_validity=lag_validity.astype(np.float32),
            mean_effect=lag_mean.astype(np.float32),
            standard_error=lag_se.astype(np.float32),
            p_value=lag_p.astype(np.float32),
            global_bh_q_value=lag_q.astype(np.float32),
        )
        lag_pair_rows = []
        for lag_index, lag in enumerate(method.lag_frames):
            for target in range(len(neurons)):
                for source in range(len(neurons)):
                    if target == source:
                        continue
                    lag_pair_rows.append(
                        {
                            "target": neurons[target],
                            "source": neurons[source],
                            "horizon_frames": int(lag),
                            "horizon_seconds": float(lag / 4.0),
                            "mean_effect_per_achieved_source_unit": float(
                                lag_mean[lag_index, target, source]
                            ),
                            "animal_se": float(lag_se[lag_index, target, source]),
                            "signflip_p": float(lag_p[lag_index, target, source]),
                            "global_pair_by_horizon_bh_q": float(
                                lag_q[lag_index, target, source]
                            ),
                            "source_valid_rate": float(lag_validity[:, source].mean()),
                            "atlas_confirmed": bool(atlas["eval_mask"][target, source]),
                            "atlas_positive": bool(atlas["positive"][target, source] > 0),
                            "atlas_dff": float(atlas["dFF"][target, source])
                            if np.isfinite(atlas["dFF"][target, source])
                            else "",
                        }
                    )
        _write_csv(output / f"{method.name}__lagged_pairwise_tests.csv", lag_pair_rows)
        compatible_lag_pairs = [
            row for row in lag_pair_rows if row["source_valid_rate"] >= 0.50
        ]
        compatible_lag_pairs.sort(
            key=lambda row: (
                row["global_pair_by_horizon_bh_q"],
                -abs(row["mean_effect_per_achieved_source_unit"]),
            )
        )
        for rank, row in enumerate(compatible_lag_pairs[:25], start=1):
            top_lag_rows.append({"model": method.name, "rank": rank, **row})
        atlas_positive_lag_pairs = [
            row for row in compatible_lag_pairs if row["atlas_positive"]
        ]
        for rank, row in enumerate(atlas_positive_lag_pairs[:25], start=1):
            top_atlas_positive_lag_rows.append(
                {"model": method.name, "rank": rank, **row}
            )
        pair_summary[method.name] = {
            "offdiagonal_pairs": int(off.sum()),
            "bh_q_le_0.10": int(np.sum((q_value <= 0.10) & off)),
            "bh_q_le_0.05": int(np.sum((q_value <= 0.05) & off)),
            "compatible_bh_q_le_0.10": int(
                np.sum((q_value <= 0.10) & off & (worm_validity.mean(axis=0)[None] >= 0.50))
            ),
            "lagged_tests": int(lag_test_mask.sum()),
            "lagged_global_bh_q_le_0.10": int(
                np.sum((lag_q <= 0.10) & lag_test_mask)
            ),
            "compatible_lagged_global_bh_q_le_0.10": int(
                np.sum(
                    (lag_q <= 0.10)
                    & lag_test_mask
                    & (lag_validity.mean(axis=0)[None, None] >= 0.50)
                )
            ),
        }

    _write_csv(output / "top_compatibility_qualified_integrated_effects.csv", top_pair_rows)
    _write_csv(output / "top_compatibility_qualified_lagged_effects.csv", top_lag_rows)
    _write_csv(
        output / "top_compatibility_qualified_atlas_positive_lagged_effects.csv",
        top_atlas_positive_lag_rows,
    )

    np.savez_compressed(
        output / "lag_resolved_effect_matrices.npz",
        neurons=np.asarray(neurons),
        horizon_frames=flow.lag_frames,
        horizon_seconds=flow.lag_frames / 4.0,
        flow_signed=flow.signed_lag.astype(np.float32),
        flow_strength=flow.lag_scores.astype(np.float32),
        mdn4_signed=mdn.signed_lag.astype(np.float32),
        mdn4_strength=mdn.lag_scores.astype(np.float32),
        sbtg_signed=sbtg.signed_lag.astype(np.float32),
        sbtg_strength=sbtg.lag_scores.astype(np.float32),
        pearson_signed=pearson.signed_lag.astype(np.float32),
        pearson_strength=pearson.lag_scores.astype(np.float32),
    )

    create_figures(methods, metric_rows, output)
    all_rows = {row["method"]: row for row in metric_rows if row["scope"] == "all_confirmed"}
    common_rows = {
        row["method"]: row
        for row in metric_rows
        if row["scope"] == "common_compatible_sources"
    }
    flow_diff = comparisons["all_confirmed"][f"{flow.name}_minus_SBTG"]
    common_flow_diff = comparisons["common_compatible_sources"][
        f"{flow.name}_minus_SBTG"
    ]
    if (
        flow_diff["auroc"]["difference"] > 0
        and common_flow_diff["auroc"]["difference"] > 0
    ):
        verdict = (
            "The repaired flow response beats SBTG on the atlas AUROC point estimate "
            "both overall and in the strict common-compatible source scope."
        )
    elif flow_diff["auroc"]["difference"] > 0:
        verdict = (
            "The repaired flow response beats SBTG on the all-pair atlas AUROC point "
            "estimate but not in the strict common-compatible source scope; the "
            "advantage is therefore not compatibility-robust."
        )
    else:
        verdict = (
            "The repaired flow response does not beat SBTG on the all-pair atlas AUROC "
            "point estimate."
        )
    repair_table_lines = [
        "## Repair-length sensitivity",
        "",
        "All rows hold model seed 1701 fixed so the repair-prefix comparison is paired.",
        "",
        "| Model | Repair seconds | AUROC | AUPRC | Mean valid-source rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    repair_table_lines.extend(
        f"| {row['model']} | {row['repair_seconds']:.1f} | {row['auroc']:.3f} | "
        f"{row['auprc']:.3f} | {row['mean_source_valid_rate']:.3f} |"
        for row in repair_rows
    )
    repair_table_lines.append("")
    semantic_table_lines = [
        "## Clamp/anchor regularization sensitivity",
        "",
        "| Model | Setting | Anchor λ | Clamp IQR fraction | Mean valid rate | Score rho vs primary |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    semantic_table_lines.extend(
        f"| {row['model']} | {row['semantic']} | {row['anchor_lambda']:.2f} | "
        f"{row['epsilon_iqr_fraction']:.2f} | {row['mean_valid_source_rate']:.3f} | "
        f"{row['offdiagonal_score_spearman_vs_primary']:.3f} |"
        for row in semantic_rows
    )
    semantic_table_lines.append("")
    particle_table_lines = [
        "## Monte Carlo particle-count sensitivity",
        "",
    ]
    if particle_rows:
        particle_table_lines.extend(
            [
                "The comparison holds model seed 1701 fixed and changes only the natural-path bank size.",
                "",
                "| Model | Particles | Score rho vs base | AUROC change | AUPRC change | Mean valid rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        particle_table_lines.extend(
            f"| {row['model']} | {row['sensitivity_particles']} | {row['offdiagonal_score_spearman']:.3f} | "
            f"{row['auroc_difference']:+.3f} | {row['auprc_difference']:+.3f} | "
            f"{row['sensitivity_mean_valid_rate']:.3f} |"
            for row in particle_rows
        )
    else:
        particle_table_lines.append("No higher-particle sensitivity run was supplied.")
    particle_table_lines.append("")
    top_table_lines = [
        "## Strongest compatibility-qualified lag-specific effects",
        "",
        "These are ranked within each model by global pair-by-horizon BH q-value and then absolute effect size. The complete family, including unsupported queries, is retained in the CSV files.",
        "",
        "| Model | Source → target | Horizon (s) | Signed effect | Global BH q | Atlas positive |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for model in (flow.name, mdn.name):
        rows = [row for row in top_lag_rows if row["model"] == model][:5]
        top_table_lines.extend(
            f"| {model} | {row['source']} → {row['target']} | {row['horizon_seconds']:.2f} | "
            f"{row['mean_effect_per_achieved_source_unit']:+.3f} | "
            f"{row['global_pair_by_horizon_bh_q']:.3g} | {row['atlas_positive']} |"
            for row in rows
        )
    top_table_lines.append("")
    atlas_positive_table_lines = [
        "## Strongest compatibility-qualified effects on atlas-positive pairs",
        "",
        "This reporting-only view restricts the frozen lag-specific results to atlas-positive pairs; the atlas dFF column makes sign agreement or disagreement explicit. Atlas labels did not select models, lags, contrasts, or compatibility thresholds.",
        "",
        "| Model | Source → target | Horizon (s) | Signed effect | Global BH q | Atlas dFF |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in (flow.name, mdn.name):
        rows = [
            row for row in top_atlas_positive_lag_rows if row["model"] == model
        ][:5]
        atlas_positive_table_lines.extend(
            f"| {model} | {row['source']} → {row['target']} | {row['horizon_seconds']:.2f} | "
            f"{row['mean_effect_per_achieved_source_unit']:+.3f} | "
            f"{row['global_pair_by_horizon_bh_q']:.3g} | {float(row['atlas_dff']):+.3f} |"
            for row in rows
        )
    atlas_positive_table_lines.append("")
    lines = [
        "# Compatibility-aware NeuroPAL path-response benchmark",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Answer",
        "",
        verdict,
        "",
        "All effects below are observational, model-relative history-to-future responses under the fixed stimulus schedule. They are not physical interventions, direct synapses, or anatomical edges.",
        "",
        "## Primary external validation",
        "",
        "The primary target is the confirmed-edge evaluation mask from the Randi/Leifer whole-brain optogenetic signal-propagation atlas. Hyperparameters and compatibility rules were frozen without consulting atlas labels.",
        "",
        "| Method | AUROC (95% outer-fold bootstrap) | AUPRC (95%) | Signed rho | Sign agreement | Timescale rho |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        row = all_rows[method.name]
        lines.append(
            f"| {method.name} | {row['auroc']:.3f} [{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}] | "
            f"{row['auprc']:.3f} [{row['auprc_ci_low']:.3f}, {row['auprc_ci_high']:.3f}] | "
            f"{row['signed_spearman']:.3f} | {row['sign_agreement']:.3f} | {row['timescale_spearman']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strict common-compatible-source validation",
            "",
            f"This scope retains {int(compatible.sum())}/{len(neurons)} sources that pass the 50% validity threshold in both neural generators.",
            "",
            "| Method | AUROC (95% outer-fold bootstrap) | AUPRC (95%) | Positive prevalence |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method in methods:
        row = common_rows[method.name]
        lines.append(
            f"| {method.name} | {row['auroc']:.3f} [{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}] | "
            f"{row['auprc']:.3f} [{row['auprc_ci_low']:.3f}, {row['auprc_ci_high']:.3f}] | "
            f"{row['prevalence']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Direct comparison with SBTG",
            "",
            f"- Flow minus SBTG AUROC: {flow_diff['auroc']['difference']:+.3f} "
            f"[{flow_diff['auroc']['ci_low']:+.3f}, {flow_diff['auroc']['ci_high']:+.3f}], "
            f"one-sided bootstrap p={flow_diff['auroc']['p_left_not_better']:.4f}.",
            f"- Flow minus SBTG AUPRC: {flow_diff['auprc']['difference']:+.3f} "
            f"[{flow_diff['auprc']['ci_low']:+.3f}, {flow_diff['auprc']['ci_high']:+.3f}], "
            f"one-sided bootstrap p={flow_diff['auprc']['p_left_not_better']:.4f}.",
            f"- Strict common-compatible flow minus SBTG AUROC: {common_flow_diff['auroc']['difference']:+.3f} "
            f"[{common_flow_diff['auroc']['ci_low']:+.3f}, {common_flow_diff['auroc']['ci_high']:+.3f}], "
            f"one-sided bootstrap p={common_flow_diff['auroc']['p_left_not_better']:.4f}.",
            f"- Strict common-compatible flow minus SBTG AUPRC: {common_flow_diff['auprc']['difference']:+.3f} "
            f"[{common_flow_diff['auprc']['ci_low']:+.3f}, {common_flow_diff['auprc']['ci_high']:+.3f}], "
            f"one-sided bootstrap p={common_flow_diff['auprc']['p_left_not_better']:.4f}.",
            "- The bootstrap resamples the five matched outer animal folds. With only five clusters, intervals are necessarily coarse.",
            "",
            "## Compatibility and predictive audit",
            "",
            f"- Common high-compatibility sources (valid in at least 50% of queries for both neural generators): {int(compatible.sum())}/{len(neurons)}.",
            f"- Animal-block pair tests at BH q<=0.10: flow {pair_summary[flow.name]['bh_q_le_0.10']}, MDN-4 {pair_summary[mdn.name]['bh_q_le_0.10']} out of 2,862 ordered off-diagonal pairs; compatibility-qualified counts are {pair_summary[flow.name]['compatible_bh_q_le_0.10']} and {pair_summary[mdn.name]['compatible_bh_q_le_0.10']}.",
            f"- Lag-specific animal-block tests at global pair-by-horizon BH q<=0.10: flow {pair_summary[flow.name]['lagged_global_bh_q_le_0.10']}, MDN-4 {pair_summary[mdn.name]['lagged_global_bh_q_le_0.10']} out of {pair_summary[flow.name]['lagged_tests']:,} tests per model; compatibility-qualified counts are {pair_summary[flow.name]['compatible_lagged_global_bh_q_le_0.10']} and {pair_summary[mdn.name]['compatible_lagged_global_bh_q_le_0.10']}.",
            f"- Flow multi-step rollout energy averaged across phases/horizons: {flow.rollout['energy']:.3f}; RMSE {flow.rollout['rmse']:.3f}; marginal 90% coverage {flow.rollout['coverage90']:.3f}.",
            f"- MDN-4 multi-step rollout energy: {mdn.rollout['energy']:.3f}; RMSE {mdn.rollout['rmse']:.3f}; coverage {mdn.rollout['coverage90']:.3f}.",
            "- Unsupported low-compatibility queries remain in the raw files but are explicitly marked for abstention.",
            "",
            *repair_table_lines,
            *semantic_table_lines,
            *particle_table_lines,
            *top_table_lines,
            *atlas_positive_table_lines,
            "## Method-to-implementation mapping",
            "",
            "- The attached repaired-law construction is implemented as a potential-reweighted learned path law: a phase-specific source clamp times a population anchor, normalized by natural-path importance weights.",
            "- Low and high source laws are training-fold 25th and 75th percentiles of the one-second source statistic; the external atlas never selects these contrasts.",
            "- A shared bank of natural generated prefixes is reweighted separately for every source and contrast. This targets the same repaired law while avoiding 54 separate expensive rollout banks.",
            "- The designated source is omitted from the population anchor throughout its one-second statistic window. After the temporal cut, trajectories roll freely under the learned generator and fixed observed stimulus schedule.",
            "- Fifteen predeclared evaluation cuts per held-out worm cover baseline, onset, active, offset, and recovery phases for each of three stimulus events. The underlying conditional generators were trained on all eligible windows.",
            "- Compatibility diagnostics include source, repair, and total log mass; effective sample size; maximum normalized weight; entropy-equivalent ancestors; achieved contrast; and explicit validity/abstention.",
            "",
            "## Estimand and design",
            "",
            "- Models: fold-specific TCN residual flow-matching and MDN-4 checkpoints, five held-out-animal folds and three seeds.",
            "- Source: one-second activity average, contrasted between training-fold phase-specific 25th and 75th percentiles.",
            "- Primary repair: one second; observed population anchor with source omitted during the source window.",
            "- Horizons: 0.25, 0.5, 1, 2, 4, 6, 8, and 10 seconds.",
            "- Pair score: maximum absolute cumulative-mean response across horizons, averaged across stimulus phases and normalized by achieved source displacement.",
            "- Comparator: SBTG-FeatureBilinear rerun on the same 54 neurons and frozen outer animal folds, with two-fold whole-worm held-out score cross-fitting inside each outer-training set.",
            "",
            "## Artifacts",
            "",
            "- `method_metrics.csv`: primary and compatibility-restricted atlas results.",
            "- `lag_metrics.csv`: result curves by future horizon/SBTG lag.",
            "- `rollout_metrics_by_horizon.csv`: held-out-animal free-rollout energy, RMSE, and coverage at every future horizon.",
            "- `inference.json`: paired fold bootstrap and source-label negative controls.",
            "- `compatibility_by_source.csv`: source-level support diagnostics.",
            "- `repair_length_sensitivity.csv`: one-, two-, and four-second repair comparisons.",
            "- `repair_semantic_sensitivity.csv`: support and response-rank stability under clamp/anchor regularization changes fixed before atlas evaluation.",
            "- `particle_count_sensitivity.csv`: fixed-seed Monte Carlo stability when doubling the natural-path bank.",
            "- `lag_resolved_effect_matrices.npz`: aligned target-by-source matrices for every future horizon and comparator lag.",
            "- `*__pairwise_tests.csv`: animal-block integrated effect estimates, sign-flip p-values, and BH q-values.",
            "- `*__lagged_pairwise_tests.csv`: the full pair-by-horizon animal-block test family with a single global BH correction.",
            "- `top_compatibility_qualified_*_effects.csv`: ranked, support-qualified integrated and lag-specific effects for inspection.",
            "- `top_compatibility_qualified_atlas_positive_lagged_effects.csv`: a reporting-only ranked view of externally confirmed supported effects.",
            "- `figures/`: method comparison and lag curves.",
            "- Raw repaired response tensors and all compatibility diagnostics remain in the response run directory.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
