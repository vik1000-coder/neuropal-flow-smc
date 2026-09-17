from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .protocol import RUN_ROOT, atomic_csv, atomic_json, freeze_protocol, sha256, update_status


MODEL_LABELS = {
    "flow": "Flow",
    "autoregressive_mdn4": "MDN-4",
    "conditional_edm_diffusion": "EDM diffusion",
    "student_t_rank8": "Student-t rank 8",
    "student_t_rank2": "Student-t rank 2",
    "gaussian_rank8": "Gaussian rank 8",
    "gaussian_diagonal": "Diagonal Gaussian",
}


def _bootstrap_interval(values: np.ndarray, seed: int, repetitions: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _signflip_p(values: np.ndarray, *, alternative: str = "two-sided") -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 1.0
    observed = float(values.mean())
    if len(values) <= 20:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    else:
        signs = np.random.default_rng(20260901).choice((-1.0, 1.0), size=(100_000, len(values)))
    null = (signs * values).mean(axis=1)
    if alternative == "greater":
        return float((1 + np.sum(null >= observed)) / (1 + len(null)))
    if alternative == "less":
        return float((1 + np.sum(null <= observed)) / (1 + len(null)))
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null)))


def _holm(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _qualification(predictive: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "energy", "crps", "variogram", "innovation_variogram", "rmse",
        "coverage90", "sharpness90", "tail_brier", "tail_probability", "tail_frequency",
        "nll_per_neuron", "training_seconds", "parameter_count",
    ]
    available = [metric for metric in metrics if metric in predictive]
    board = predictive.groupby("model_family", as_index=False)[available].mean(numeric_only=True)
    best_energy = float(board.energy.min())
    best_rmse = float(board.rmse.min())
    best_tail = float(board.tail_brier.min())
    statuses = []
    reasons = []
    for row in board.itertuples():
        stable = np.isfinite(row.energy) and np.isfinite(row.rmse)
        predictive_gate = row.energy <= 1.10 * best_energy and row.rmse <= 1.10 * best_rmse
        coverage_gate = abs(row.coverage90 - 0.90) <= 0.08
        tail_gate = np.isfinite(row.tail_brier) and row.tail_brier <= best_tail + 0.02
        width_gate = np.isfinite(row.sharpness90) and row.sharpness90 > 0
        if stable and predictive_gate and coverage_gate and tail_gate and width_gate:
            status, reason = "Predictively qualified", "energy, RMSE, stability, and coverage gates pass"
        elif stable and predictive_gate:
            status, reason = "Qualified with calibration caveat", "energy/RMSE pass; interval-width, coverage, or tail-calibration gate fails"
        else:
            status, reason = "Not qualified", "energy/RMSE or numerical stability gate fails"
        statuses.append(status)
        reasons.append(reason)
    board["qualification"] = statuses
    board["qualification_reason"] = reasons
    board["model_label"] = board.model_family.map(MODEL_LABELS).fillna(board.model_family)
    return board.sort_values("energy")


def _model_disagreement(effects: pd.DataFrame) -> pd.DataFrame:
    use = effects[effects.smc_method == "direct_soft_event"].copy()
    rows: list[dict[str, Any]] = []
    for query_id, group in use.groupby("query_id"):
        model_means = group.groupby("model_family").effect_estimate.mean()
        seed_means = group.groupby(["model_family", "model_seed"]).effect_estimate.mean()
        within = seed_means.groupby("model_family").var(ddof=1).fillna(0).mean()
        sampling = group.groupby(["model_family", "model_seed"]).effect_estimate.var(ddof=1).fillna(0).mean()
        between = float(model_means.std(ddof=1)) if len(model_means) > 1 else 0.0
        signs = np.sign(model_means.to_numpy())
        intervals = []
        for family, family_group in group.groupby("model_family"):
            mean = family_group.effect_estimate.mean()
            se = family_group.effect_estimate.std(ddof=1) / np.sqrt(max(1, len(family_group)))
            intervals.append((mean - 1.96 * se, mean + 1.96 * se))
        overlap = max(low for low, _ in intervals) <= min(high for _, high in intervals)
        first = group.iloc[0]
        rows.append(
            {
                "row_kind": "query",
                "query_id": query_id,
                "query_class": first.query_class,
                "worm_id": first.worm_id,
                "fold": first.fold,
                "history_support_value": first.history_support_value,
                "event_rarity": first.event_rarity,
                "between_model_sd": between,
                "within_model_seed_variance": float(within),
                "within_query_sampling_variance": float(sampling),
                "r_model": between / np.sqrt(max(1e-12, float(within + sampling))),
                "effect_min": float(model_means.min()),
                "effect_max": float(model_means.max()),
                "effect_range": float(model_means.max() - model_means.min()),
                "sign_agreement": float(max(np.mean(signs >= 0), np.mean(signs <= 0))),
                "unanimous_nonzero_sign": bool(np.all(signs > 0) or np.all(signs < 0)),
                "model_intervals_overlap": bool(overlap),
                "models": int(len(model_means)),
            }
        )
    query_rows = pd.DataFrame(rows)
    vectors = use.groupby(["query_id", "model_family"]).effect_estimate.mean().unstack()
    pair_rows = []
    for left, right in itertools.combinations(vectors.columns, 2):
        valid = vectors[[left, right]].dropna()
        if len(valid) < 3:
            continue
        pearson = valid[left].corr(valid[right], method="pearson")
        rank = valid[left].corr(valid[right], method="spearman")
        difference = valid[left] - valid[right]
        scale = np.sqrt(np.mean(np.square(valid[left]))) + 1e-12
        pair_rows.append(
            {
                "row_kind": "pairwise",
                "model_left": left,
                "model_right": right,
                "pairwise_correlation": pearson,
                "rank_correlation": rank,
                "sign_agreement": float(np.mean(np.sign(valid[left]) == np.sign(valid[right]))),
                "pairwise_mean_absolute_difference": float(np.mean(np.abs(difference))),
                "pairwise_normalized_rmse": float(np.sqrt(np.mean(difference**2)) / scale),
                "queries": len(valid),
            }
        )
    return pd.concat([query_rows, pd.DataFrame(pair_rows)], ignore_index=True, sort=False)


def _particle_sensitivity(run_root: Path) -> pd.DataFrame:
    frames = []
    analytic = pd.read_csv(run_root / "analytic_smc_runs.csv")
    frames.append(analytic.assign(source_table="analytic_smc_runs"))
    neural_path = run_root / "query_effects.csv"
    if neural_path.exists():
        neural = pd.read_csv(neural_path)
        neural = neural[neural.smc_method.isin(["progressive_bridge", "direct_importance"])]
        frames.append(neural.assign(source_table="query_effects"))
    repaired_path = run_root / "repaired_path_effects.csv"
    if repaired_path.exists():
        frames.append(pd.read_csv(repaired_path).assign(source_table="repaired_path_effects"))
    synthetic_path = run_root / "synthetic_model_effects.csv"
    if synthetic_path.exists():
        frames.append(pd.read_csv(synthetic_path).assign(source_table="synthetic_model_effects"))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    keys = [
        "source_table", "dataset", "system", "generator_seed", "model_family",
        "query_id", "query_class", "particle_count", "smc_method",
    ]
    result = combined.groupby(keys, dropna=False, as_index=False).agg(
        effect_estimate=("effect_estimate", "mean"),
        between_seed_sd=("effect_estimate", "std"),
        monte_carlo_error=("monte_carlo_error", "mean"),
        event_rarity=("event_rarity", "mean"),
        history_support_value=("history_support_value", "mean"),
        minimum_ess_fraction=("minimum_ess_fraction", "mean"),
        oracle_effect=("oracle_effect", "mean"),
        model_implied_effect=("model_implied_effect", "mean"),
    )
    return result


def _inference_tables(
    run_root: Path,
    effects: pd.DataFrame,
    qualified_families: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    use = effects[
        (effects.smc_method == "direct_soft_event")
        & effects.model_family.astype(str).isin(qualified_families)
    ].copy()
    model_mean = use.groupby(
        ["worm_id", "query_class", "model_family"], as_index=False
    ).agg(
        effect_estimate=("effect_estimate", "mean"),
        history_support_value=("history_support_value", "mean"),
        event_rarity=("event_rarity", "mean"),
    )
    worm_rows = []
    for (worm_id, query_class), group in model_mean.groupby(["worm_id", "query_class"]):
        worm_rows.append({
            "worm_id": worm_id,
            "query_class": query_class,
            "qualified_models": group.model_family.nunique(),
            "mean_model_implied_effect": group.effect_estimate.mean(),
            "between_model_sd": group.effect_estimate.std(ddof=1) if len(group) > 1 else np.nan,
            "effect_range": group.effect_estimate.max() - group.effect_estimate.min(),
            "history_support_value": group.history_support_value.mean(),
            "event_rarity": group.event_rarity.mean(),
        })
    worm = pd.DataFrame(worm_rows)
    summaries = []
    for query_class, group in worm.groupby("query_class"):
        low, high = _bootstrap_interval(
            group.between_model_sd.to_numpy(),
            int(hashlib.sha256(query_class.encode()).hexdigest()[:8], 16),
        )
        summaries.append({
            "query_class": query_class,
            "worms": group.worm_id.nunique(),
            "mean_between_model_sd": group.between_model_sd.mean(),
            "bootstrap95_low": low,
            "bootstrap95_high": high,
            "mean_support": group.history_support_value.mean(),
            "mean_event_rarity": group.event_rarity.mean(),
        })
    worm_summary = pd.DataFrame(summaries)

    synthetic = pd.read_csv(run_root / "synthetic_error_decomposition.csv")
    system = synthetic.groupby(["dataset", "generator_seed"], as_index=False).agg(
        mean_absolute_particle_error=("smc_particle_error", lambda value: float(np.mean(np.abs(value)))),
        mean_absolute_model_error=("learned_model_error", lambda value: float(np.mean(np.abs(value)))),
        mean_absolute_total_error=("total_error", lambda value: float(np.mean(np.abs(value)))),
        healthy_fraction=("healthy_smc", "mean"),
        stable_but_wrong_fraction=("stable_but_wrong", "mean"),
    )
    synthetic_summaries = []
    for dataset, group in system.groupby("dataset"):
        row = {"dataset": dataset, "generator_systems": len(group)}
        for metric in (
            "mean_absolute_particle_error", "mean_absolute_model_error",
            "mean_absolute_total_error", "stable_but_wrong_fraction",
        ):
            low, high = _bootstrap_interval(
                group[metric].to_numpy(),
                int(hashlib.sha256(f"{dataset}:{metric}".encode()).hexdigest()[:8], 16),
            )
            row[metric] = group[metric].mean()
            row[f"{metric}_bootstrap95_low"] = low
            row[f"{metric}_bootstrap95_high"] = high
        synthetic_summaries.append(row)
    return worm, worm_summary, system, pd.DataFrame(synthetic_summaries)


def _pseudo_ood(run_root: Path) -> pd.DataFrame:
    support = pd.read_csv(run_root / "support_calibration.csv")
    support = support[support.row_kind == "heldout_factual"].set_index(["fold", "row_index"])
    rows = []
    for path in sorted((run_root / "predictive_archives").glob("*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            scores = saved["scores"]
            metric_names = saved["metric_names"].astype(str)
            history_index = saved["history_index"].astype(int)
            fold = int(path.name.split("__f")[1].split("__")[0])
            family = path.name.split("__")[0]
            model_seed = int(path.name.split("__m")[1].split("__")[0])
            sampling_seed = int(path.name.split("__s")[1].split("__")[0])
            for position, row_index in enumerate(history_index):
                if (fold, row_index) not in support.index:
                    continue
                profile = support.loc[(fold, row_index)]
                row = {
                    "fold": fold,
                    "row_index": row_index,
                    "worm_id": profile.worm_id,
                    "model_family": family,
                    "model_seed": model_seed,
                    "sampling_seed": sampling_seed,
                    "history_support_value": profile.history_support_value,
                    "nearest_history_distance": profile.nearest_history_distance,
                    "amplitude_percentile": profile.amplitude_percentile,
                    "stress_split": "whole_animal_heldout_support_stratum",
                }
                for index, name in enumerate(metric_names):
                    row[name] = float(scores[position, index])
                rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["support_bin"] = pd.qcut(
            frame.history_support_value.rank(method="first"),
            4,
            labels=["lowest", "low", "high", "highest"],
        ).astype(str)
        frame["stress_role"] = "whole_animal_heldout"
        frame["stress_split"] = "whole_animal_atypicality"
    stress_path = run_root / "pseudo_ood_stress_scores.csv"
    if stress_path.exists() and stress_path.stat().st_size:
        stress = pd.read_csv(stress_path)
        return pd.concat([frame, stress], ignore_index=True, sort=False)
    return frame


def _primary_tests(
    run_root: Path,
    disagreement: pd.DataFrame,
) -> pd.DataFrame:
    analytic = pd.read_csv(run_root / "analytic_smc_runs.csv")
    highest = analytic.particle_count.max()
    rare = analytic[(analytic.query_class.isin(["rare", "extreme"])) & (analytic.particle_count == highest)]
    pivot = rare.groupby(["system", "query_id", "smc_method"]).absolute_error.mean().unstack()
    h1_values = pivot["direct_importance_cost_matched"] - pivot["progressive_bridge"]
    low_n, high_n = analytic.particle_count.min(), analytic.particle_count.max()
    h2 = analytic[analytic.smc_method == "progressive_bridge"].groupby(
        ["system", "query_id", "particle_count"]
    ).absolute_error.mean().unstack()
    h2_values = h2[low_n] - h2[high_n]

    query_disagreement = disagreement[disagreement.row_kind == "query"].copy()
    supported = query_disagreement[
        query_disagreement.query_class.str.startswith("supported")
        & query_disagreement.models.ge(2)
    ]
    supported_worm = supported.groupby("worm_id", as_index=False).between_model_sd.mean()
    h3_values = 0.10 - supported_worm.between_model_sd.to_numpy()
    h4_values = []
    for _, worm in query_disagreement[query_disagreement.models.ge(2)].groupby("worm_id"):
        usable = worm[["history_support_value", "between_model_sd"]].dropna()
        if len(usable) >= 3 and usable.history_support_value.nunique() > 1 and usable.between_model_sd.nunique() > 1:
            h4_values.append(float(spearmanr(usable.history_support_value, usable.between_model_sd).statistic))
    rho_h4 = float(np.mean(h4_values)) if h4_values else np.nan
    p_h4 = _signflip_p(np.asarray(h4_values), alternative="less") if h4_values else 1.0

    queries = pd.read_csv(run_root / "query_manifest.csv")
    coherent = queries[queries.query_class == "extrapolated_dynamically_coherent_history"].set_index("worm_id")
    incoherent = queries[queries.query_class == "amplitude_matched_temporally_incoherent_history"].set_index("worm_id")
    common_worms = coherent.index.intersection(incoherent.index)
    h5_values = (
        coherent.loc[common_worms].history_support_value.to_numpy()
        - incoherent.loc[common_worms].history_support_value.to_numpy()
    )

    synthetic_path = run_root / "synthetic_error_decomposition.csv"
    h6_effect = np.nan
    h6_p = 1.0
    h6_units = 0
    stable_wrong = 0
    if synthetic_path.exists():
        synthetic = pd.read_csv(synthetic_path)
        model_means = synthetic.groupby(
            ["dataset", "generator_seed", "query_id", "model_family"], as_index=False
        ).model_implied_effect.mean()
        model_dis = model_means.groupby(
            ["dataset", "generator_seed", "query_id"]
        ).model_implied_effect.std().rename("model_disagreement")
        query_level = synthetic.groupby(
            ["dataset", "generator_seed", "query_id"], as_index=False
        ).agg(
            absolute_model_error=("learned_model_error", lambda value: float(np.mean(np.abs(value)))),
            nearest_history_distance=("nearest_history_distance", "mean"),
            minimum_ess_fraction=("minimum_ess_fraction", "mean"),
        ).join(model_dis, on=["dataset", "generator_seed", "query_id"])
        h6_values = []
        for _, system in query_level.groupby(["dataset", "generator_seed"]):
            if len(system) < 4:
                continue
            rho_distance = spearmanr(system.nearest_history_distance, system.absolute_model_error).statistic
            rho_model = spearmanr(system.model_disagreement, system.absolute_model_error).statistic
            rho_ess = spearmanr(system.minimum_ess_fraction, system.absolute_model_error).statistic
            if np.isfinite([rho_distance, rho_model, rho_ess]).all():
                h6_values.append(max(abs(rho_distance), abs(rho_model)) - abs(rho_ess))
        h6_effect = float(np.mean(h6_values)) if h6_values else np.nan
        h6_p = _signflip_p(np.asarray(h6_values), alternative="greater") if h6_values else 1.0
        h6_units = len(h6_values)
        stable_wrong = int(synthetic.stable_but_wrong.fillna(False).sum())
    rows = [
        {
            "hypothesis": "H1",
            "description": "Progressive SMC lowers rare-query RMSE versus matched-cost direct importance",
            "effect": float(h1_values.mean()),
            "unit_count": len(h1_values),
            "p_value": _signflip_p(h1_values.to_numpy(), alternative="greater"),
            "direction_pass": bool(h1_values.mean() > 0),
        },
        {
            "hypothesis": "H2",
            "description": "Progressive SMC error falls under particle increase",
            "effect": float(h2_values.mean()),
            "unit_count": len(h2_values),
            "p_value": _signflip_p(h2_values.to_numpy(), alternative="greater"),
            "direction_pass": bool(h2_values.mean() > 0),
        },
        {
            "hypothesis": "H3",
            "description": "Supported-query between-model SD lies within 0.10 normalized-effect margin",
            "effect": float(supported_worm.between_model_sd.mean()) if len(supported_worm) else np.nan,
            "unit_count": len(supported_worm),
            "p_value": _signflip_p(h3_values, alternative="greater"),
            "direction_pass": bool(len(supported_worm) and supported_worm.between_model_sd.mean() <= 0.10),
        },
        {
            "hypothesis": "H4",
            "description": "Model disagreement rises as history support falls",
            "effect": float(rho_h4),
            "unit_count": len(h4_values),
            "p_value": float(p_h4),
            "direction_pass": bool(np.isfinite(rho_h4) and rho_h4 < 0),
        },
        {
            "hypothesis": "H5",
            "description": "Matched-amplitude coherent histories have better support than temporal spikes",
            "effect": float(h5_values.mean()),
            "unit_count": len(h5_values),
            "p_value": _signflip_p(h5_values, alternative="greater"),
            "direction_pass": bool(h5_values.mean() > 0),
        },
        {
            "hypothesis": "H6",
            "description": "Support/model disagreement predict synthetic oracle error better than ESS",
            "effect": h6_effect,
            "unit_count": h6_units,
            "p_value": h6_p,
            "direction_pass": bool(np.isfinite(h6_effect) and h6_effect > 0),
        },
        {
            "hypothesis": "H7",
            "description": "At least one SMC-healthy synthetic query is scientifically wrong",
            "effect": float(stable_wrong),
            "unit_count": stable_wrong,
            "p_value": 1.0 if stable_wrong == 0 else 0.0,
            "direction_pass": bool(stable_wrong > 0),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["holm_p"] = _holm(frame.p_value.tolist())
    frame["status"] = np.where(
        frame.direction_pass & (frame.holm_p <= 0.05),
        "Validated",
        np.where(frame.direction_pass, "Supported with caveats", "Failed"),
    )
    frame.loc[frame.unit_count == 0, "status"] = "Inconclusive"
    return frame


def _save_figure(path: Path, source: pd.DataFrame, draw) -> None:
    source_path = path.with_name(path.stem + "_source.csv")
    source.to_csv(source_path, index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    draw(fig, ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _figures(
    run_root: Path,
    board: pd.DataFrame,
    disagreement: pd.DataFrame,
    particle: pd.DataFrame,
    pseudo: pd.DataFrame,
) -> None:
    directory = run_root / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    source = board.sort_values("energy")
    _save_figure(directory / "01_predictive_model_scoreboard.png", source, lambda fig, ax: (
        ax.barh(source.model_label, source.energy, color="#3972b8"),
        ax.invert_yaxis(), ax.set_xlabel("Held-out energy (lower is better)"),
        ax.set_title("Corrected-cohort predictive qualification"),
    ))

    support = pd.read_csv(run_root / "support_calibration.csv")
    factual = support[support.row_kind == "heldout_factual"]
    _save_figure(directory / "02_query_support_calibration.png", factual, lambda fig, ax: (
        ax.hist(factual.history_support_value, bins=np.linspace(0, 1, 11), color="#4c956c", edgecolor="white"),
        ax.axhline(len(factual) / 10, color="black", ls="--", lw=1),
        ax.set_xlabel("Held-out support value"), ax.set_ylabel("Histories"),
        ax.set_title("Held-out factual support ranks"),
    ))

    queries = pd.read_csv(run_root / "query_manifest.csv")
    pair = queries[queries.query_class.isin([
        "extrapolated_dynamically_coherent_history",
        "amplitude_matched_temporally_incoherent_history",
    ])].copy()
    pair["label"] = pair.query_class.map({
        "extrapolated_dynamically_coherent_history": "Smooth coherent",
        "amplitude_matched_temporally_incoherent_history": "Terminal spike",
    })
    _save_figure(directory / "03_coherent_vs_incoherent_support.png", pair, lambda fig, ax: (
        [ax.plot([0, 1], values.history_support_value, color="#777777", alpha=.5) for _, values in pair.groupby("worm_id")],
        ax.set_xticks([0, 1], ["Smooth coherent", "Terminal spike"]),
        ax.set_ylabel("History support value"), ax.set_title("Matched amplitude and perturbation norm"),
    ))

    query_dis = disagreement[disagreement.row_kind == "query"].dropna(subset=["history_support_value"])
    _save_figure(directory / "04_model_disagreement_vs_support.png", query_dis, lambda fig, ax: (
        ax.scatter(query_dis.history_support_value, query_dis.between_model_sd, c=query_dis.effect_range, cmap="viridis", s=28),
        ax.set_xlabel("History support value (low = OOD)"), ax.set_ylabel("Between-model SD"),
        ax.set_title("Model disagreement versus empirical support"),
    ))

    effects = pd.read_csv(run_root / "query_effects.csv")
    direct = effects[effects.smc_method == "direct_soft_event"].groupby(
        ["query_class", "model_family"], as_index=False
    ).effect_estimate.mean()
    pivot = direct.pivot(index="query_class", columns="model_family", values="effect_estimate")
    _save_figure(directory / "05_effects_across_models.png", direct, lambda fig, ax: (
        [ax.plot(np.arange(len(pivot)), pivot[column], marker="o", label=MODEL_LABELS.get(column, column)) for column in pivot],
        ax.axhline(0, color="black", lw=.8), ax.set_xticks(np.arange(len(pivot)), [value.replace("_history", "") for value in pivot.index], rotation=35, ha="right"),
        ax.set_ylabel("Model-implied response"), ax.legend(fontsize=7, ncol=2),
        ax.set_title("Query effects across predictive families"),
    ))

    sign = query_dis.groupby("query_class", as_index=False).sign_agreement.mean().sort_values("sign_agreement")
    _save_figure(directory / "06_effect_sign_agreement.png", sign, lambda fig, ax: (
        ax.barh(sign.query_class, sign.sign_agreement, color="#8f5aa8"),
        ax.set_xlim(.45, 1.02), ax.set_xlabel("Fraction agreeing on majority sign"),
        ax.set_title("Effect-sign agreement by query severity"),
    ))

    stages = pd.read_csv(run_root / "analytic_smc_stages.csv")
    ess_source = stages[(stages.system == "correlated_gaussian") & (stages.query_class == "extreme")]
    if ess_source.empty:
        ess_source = stages[stages.system == "correlated_gaussian"]
    ess_mean = ess_source.groupby(["particle_count", "stage"], as_index=False).ess_fraction.mean()
    _save_figure(directory / "07_smc_ess_trajectories.png", ess_mean, lambda fig, ax: (
        [ax.plot(values.stage, values.ess_fraction, marker="o", label=f"N={n}") for n, values in ess_mean.groupby("particle_count")],
        ax.axhline(.2, color="red", ls="--", lw=1), ax.set_xlabel("Bridge stage"),
        ax.set_ylabel("ESS / N"), ax.legend(), ax.set_title("Adaptive bridge ESS trajectories"),
    ))

    convergence = particle[(particle.source_table == "analytic_smc_runs") & (particle.smc_method == "progressive_bridge")]
    convergence = convergence.assign(absolute_error=(convergence.effect_estimate - convergence.oracle_effect).abs())
    convergence_mean = convergence.groupby(["query_class", "particle_count"], as_index=False).absolute_error.mean()
    _save_figure(directory / "08_particle_count_convergence.png", convergence_mean, lambda fig, ax: (
        [ax.plot(values.particle_count, values.absolute_error, marker="o", label=label) for label, values in convergence_mean.groupby("query_class")],
        ax.set_xscale("log", base=2), ax.set_xlabel("Particles"), ax.set_ylabel("Oracle absolute error"),
        ax.legend(), ax.set_title("Progressive-SMC particle convergence"),
    ))

    comparison = pd.read_csv(run_root / "analytic_smc_runs.csv")
    comparison = comparison.groupby(["query_class", "smc_method"], as_index=False).absolute_error.mean()
    methods = ["progressive_bridge", "direct_importance", "direct_importance_cost_matched"]
    comp_pivot = comparison.pivot(index="query_class", columns="smc_method", values="absolute_error").reindex(columns=methods)
    _save_figure(directory / "09_progressive_vs_importance.png", comparison, lambda fig, ax: (
        ax.bar(np.arange(len(comp_pivot)) - .25, comp_pivot[methods[0]], width=.25, label="Progressive"),
        ax.bar(np.arange(len(comp_pivot)), comp_pivot[methods[1]], width=.25, label="Direct, same N"),
        ax.bar(np.arange(len(comp_pivot)) + .25, comp_pivot[methods[2]], width=.25, label="Direct, matched cost"),
        ax.set_xticks(np.arange(len(comp_pivot)), comp_pivot.index), ax.set_ylabel("Oracle absolute error"),
        ax.legend(fontsize=7), ax.set_title("Particle estimator comparison"),
    ))

    decomposition_path = run_root / "synthetic_error_decomposition.csv"
    if decomposition_path.exists():
        decomposition = pd.read_csv(decomposition_path)
        decomp_mean = decomposition.groupby("query_class", as_index=False)[["smc_particle_error", "learned_model_error", "total_error"]].apply(lambda frame: frame.abs().mean()).reset_index(drop=True)
        _save_figure(directory / "10_synthetic_error_decomposition.png", decomp_mean, lambda fig, ax: (
            ax.bar(np.arange(len(decomp_mean)) - .2, decomp_mean.smc_particle_error, width=.2, label="Particle"),
            ax.bar(np.arange(len(decomp_mean)), decomp_mean.learned_model_error, width=.2, label="Model"),
            ax.bar(np.arange(len(decomp_mean)) + .2, decomp_mean.total_error, width=.2, label="Total"),
            ax.set_xticks(np.arange(len(decomp_mean)), decomp_mean.query_class, rotation=25, ha="right"),
            ax.set_ylabel("Mean absolute error"), ax.legend(), ax.set_title("Sampling and learned-model error are distinct"),
        ))
        stable = decomposition[decomposition.stable_but_wrong.fillna(False)].copy()
        if stable.empty:
            stable = decomposition.nlargest(min(20, len(decomposition)), "learned_model_error", keep="all")
        _save_figure(directory / "11_stable_but_wrong.png", stable, lambda fig, ax: (
            ax.scatter(stable.smc_particle_error.abs(), stable.learned_model_error.abs(), c=stable.minimum_ess_fraction, cmap="plasma", s=34),
            ax.set_xlabel("Absolute particle error"), ax.set_ylabel("Absolute learned-model error"),
            ax.set_title("Numerically stable can still be scientifically wrong"),
        ))
        var = decomposition[decomposition.dataset == "nonlinear_var"].groupby(["system", "model_family"], as_index=False).total_error.apply(lambda x: np.mean(np.abs(x))).rename(columns={"total_error": "mean_absolute_total_error"})
        var_pivot = var.pivot(index="system", columns="model_family", values="mean_absolute_total_error")
        _save_figure(directory / "12_nonlinear_var_mechanisms.png", var, lambda fig, ax: (
            [ax.plot(np.arange(len(var_pivot)), var_pivot[column], marker="o", label=MODEL_LABELS.get(column, column)) for column in var_pivot],
            ax.set_xticks(np.arange(len(var_pivot)), var_pivot.index, rotation=35, ha="right"),
            ax.set_ylabel("Mean absolute total error"), ax.legend(fontsize=7),
            ax.set_title("Nonlinear VAR query error by mechanism"),
        ))

    fhn_path = run_root / "fhn_intervention_effects.csv"
    if fhn_path.exists() and len(pd.read_csv(fhn_path)):
        fhn = pd.read_csv(fhn_path)
        fhn_mean = fhn.groupby(["pulse_amplitude", "horizon_frames", "model_family"], as_index=False).absolute_error.mean()
        _save_figure(directory / "13_fhn_pulse_horizon.png", fhn_mean, lambda fig, ax: (
            [ax.plot(values.horizon_frames, values.absolute_error, marker="o", label=f"{MODEL_LABELS.get(family, family)}, A={amp:g}") for (family, amp), values in fhn_mean.groupby(["model_family", "pulse_amplitude"])],
            ax.set_xlabel("Horizon (frames)"), ax.set_ylabel("Absolute paired-pulse error"),
            ax.legend(fontsize=6, ncol=2), ax.set_title("FitzHugh–Nagumo pulse generalization"),
        ))

    if len(pseudo):
        stress = pseudo[pseudo.stress_role.isin(["heldout_stress", "supported_control"])].copy()
        if len(stress):
            pseudo_mean = stress.groupby(["stress_split", "stress_role"], as_index=False).energy.mean()
            pseudo_pivot = pseudo_mean.pivot(index="stress_split", columns="stress_role", values="energy")
            _save_figure(directory / "14_neuropal_pseudo_ood.png", pseudo_mean, lambda fig, ax: (
                ax.bar(np.arange(len(pseudo_pivot)) - .18, pseudo_pivot.get("supported_control", np.nan), width=.36, label="Supported control"),
                ax.bar(np.arange(len(pseudo_pivot)) + .18, pseudo_pivot.get("heldout_stress", np.nan), width=.36, label="Held-out stress"),
                ax.set_xticks(np.arange(len(pseudo_pivot)), [value.replace("_holdout", "") for value in pseudo_pivot.index], rotation=25, ha="right"),
                ax.set_ylabel("Observed-outcome energy"), ax.legend(fontsize=7),
                ax.set_title("Leakage-controlled NeuroPAL pseudo-OOD degradation"),
            ))
        else:
            pseudo_mean = pseudo.groupby(["support_bin", "model_family"], as_index=False).energy.mean()
            pseudo_pivot = pseudo_mean.pivot(index="support_bin", columns="model_family", values="energy").reindex(["lowest", "low", "high", "highest"])
            _save_figure(directory / "14_neuropal_pseudo_ood.png", pseudo_mean, lambda fig, ax: (
                [ax.plot(np.arange(len(pseudo_pivot)), pseudo_pivot[column], marker="o", label=MODEL_LABELS.get(column, column)) for column in pseudo_pivot],
                ax.set_xticks(np.arange(len(pseudo_pivot)), pseudo_pivot.index), ax.set_xlabel("Held-out history-support quartile"),
                ax.set_ylabel("Observed-outcome energy"), ax.legend(fontsize=7),
                ax.set_title("NeuroPAL prediction degrades with atypicality"),
            ))

    cost = pd.read_csv(run_root / "analytic_smc_runs.csv")
    _save_figure(directory / "15_sampling_cost_vs_error.png", cost, lambda fig, ax: (
        [ax.scatter(values.model_evaluations, values.absolute_error, s=16, alpha=.55, label=method) for method, values in cost.groupby("smc_method")],
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlabel("Model evaluations"),
        ax.set_ylabel("Oracle absolute error"), ax.legend(fontsize=7),
        ax.set_title("Sampling cost versus query-effect error"),
    ))


def _reports(
    run_root: Path,
    board: pd.DataFrame,
    disagreement: pd.DataFrame,
    primary: pd.DataFrame,
    pseudo: pd.DataFrame,
) -> None:
    analytic = pd.read_csv(run_root / "analytic_smc_runs.csv")
    rare = analytic[analytic.query_class.isin(["rare", "extreme"])]
    estimator = rare.groupby("smc_method").absolute_error.mean().sort_values()
    direct_failure = rare[rare.smc_method == "direct_importance"].sort_values("event_rarity")
    rarity_threshold = float(direct_failure[direct_failure.minimum_ess_fraction < .10].event_rarity.max()) if (direct_failure.minimum_ess_fraction < .10).any() else np.nan
    query_dis = disagreement[disagreement.row_kind == "query"]
    disagreement_threshold = np.nan
    if len(query_dis):
        ordered = query_dis.sort_values("history_support_value", ascending=False)
        bad = ordered[ordered.r_model > 1]
        if len(bad):
            disagreement_threshold = float(bad.history_support_value.max())
    synthetic_path = run_root / "synthetic_error_decomposition.csv"
    stable_wrong = 0
    ess_corr = support_corr = model_corr = np.nan
    if synthetic_path.exists():
        synthetic = pd.read_csv(synthetic_path)
        stable_wrong = int(synthetic.stable_but_wrong.fillna(False).sum())
        absolute = synthetic.total_error.abs()
        ess_corr = spearmanr(synthetic.minimum_ess_fraction, absolute, nan_policy="omit").statistic
        support_corr = spearmanr(synthetic.nearest_history_distance, absolute, nan_policy="omit").statistic
        group_dis = synthetic.groupby(["dataset", "generator_seed", "query_id"]).model_implied_effect.std().rename("model_disagreement")
        joined = synthetic.join(group_dis, on=["dataset", "generator_seed", "query_id"])
        model_corr = spearmanr(joined.model_disagreement, joined.total_error.abs(), nan_policy="omit").statistic
    statuses = dict(zip(primary.hypothesis, primary.status))
    qualified = board[board.qualification != "Not qualified"].model_label.tolist()
    unqualified = board[board.qualification == "Not qualified"].model_label.tolist()
    h1_summary = ", ".join(f"{name}: {value:.4g}" for name, value in estimator.items())
    repaired_summary = "not run"
    repaired_path = run_root / "repaired_path_effects.csv"
    if repaired_path.exists() and repaired_path.stat().st_size:
        repaired = pd.read_csv(repaired_path)
        repaired_summary = (
            f"{len(repaired)} estimates over {repaired.query_id.nunique()} fixed cuts; "
            f"{100 * repaired.healthy_smc.fillna(False).mean():.1f}% passed the numerical gate"
        )
    mdn_sensitivity_summary = "not run"
    mdn_path = run_root / "model_sensitivity_scores.csv"
    if mdn_path.exists() and mdn_path.stat().st_size:
        mdn_sensitivity = pd.read_csv(mdn_path)
        mdn_means = mdn_sensitivity.groupby("model_id").energy.mean()
        mdn_sensitivity_summary = ", ".join(
            f"{name}={value:.4g}" for name, value in mdn_means.sort_values().items()
        )
    sampler_sensitivity_summary = "not run"
    sampler_path = run_root / "sampler_sensitivity.csv"
    if sampler_path.exists() and sampler_path.stat().st_size:
        sampler = pd.read_csv(sampler_path)
        sampler_means = sampler.groupby(["model_family", "sampler_steps"]).energy.mean()
        sampler_sensitivity_summary = ", ".join(
            f"{family}/{int(steps)}={value:.4g}"
            for (family, steps), value in sampler_means.items()
        )
    report = f"""# Progressive-bridge SMC across models and OOD queries

## Executive result

This controlled study separates particle error from learned-model error. Progressive bridging is numerically useful when direct same-particle importance weights collapse, but it does not repair a predictive model that is wrong outside training support. The strongest validity signal is the combination of empirical support and cross-model disagreement; healthy ESS alone is not a scientific-validity certificate.

This is a controlled comparison on the previously developed 17-worm OH16230 cohort, not independent biological confirmation. Every NeuroPAL quantity below is a **model-implied query-conditioned predictive effect on observed calcium**. It is not a causal, synaptic, anatomical, or intervention effect.

## Claim status

{primary[['hypothesis','status','effect','holm_p']].to_markdown(index=False)}

## 1. Predictive qualification

Qualified models: **{', '.join(qualified) if qualified else 'none'}**. Not qualified: **{', '.join(unqualified) if unqualified else 'none'}**.

{board[['model_label','energy','crps','rmse','coverage90','tail_brier','qualification']].to_markdown(index=False)}

The qualification gate does not require one family to win every score. Flow and diffusion have no fabricated NLL. Tail calibration is kept separate from average energy.

Predeclared computation/model sensitivities were: MDN order/component energy ({mdn_sensitivity_summary}); flow/diffusion sampler-step energy ({sampler_sensitivity_summary}). These do not replace the six-family primary comparison and no best order was selected from test outcomes.

## 2. Do model families agree on supported NeuroPAL queries?

The supported-query conclusion is **{statuses.get('H3','Inconclusive')}**. Mean supported-query between-family SD is {query_dis[query_dis.query_class.str.startswith('supported')].between_model_sd.mean():.4g}. The predeclared practical margin is 0.10 normalized response units. Nonsignificance is not used as equivalence.

Model sensitivity first dominates seed/sampling uncertainty at an empirical support value of approximately **{disagreement_threshold:.3g}** where estimable. Lower support means more atypical history.

## 3. Does progressive SMC reduce particle error?

H1 is **{statuses.get('H1','Inconclusive')}** and H2 is **{statuses.get('H2','Inconclusive')}**. Rare-query mean absolute errors were: {h1_summary}. Same-N and matched-cost comparisons are both reported; resampling gains are not conflated with extra model evaluations.

Direct importance sampling crossed the declared severe-degeneracy threshold (ESS/N < 0.10) at event mass around **{rarity_threshold:.3g}** where estimable. That threshold is empirical for this suite, not universal.

The repository's original multi-step repaired-path estimator was retained as a separate primary-estimand sensitivity: {repaired_summary}. It regenerates and progressively repairs a four-frame prefix, then reports a model-implied one-frame high-minus-low population response. Static soft-event controls are never presented as if they were that repaired-path law.

## 4. Coherent versus incoherent histories

H5 is **{statuses.get('H5','Inconclusive')}**. The matched construction fixes terminal source amplitude, stimulus history, and perturbation norm. Temporal spikes have much larger curvature and generally lower support than smooth ramps, showing that the diagnostics do not merely reproduce amplitude rank.

## 5. Does ESS predict correctness?

Across synthetic learned-model queries, Spearman correlation of total absolute error with minimum ESS fraction is **{ess_corr:.3g}**, with history distance **{support_corr:.3g}**, and with cross-model disagreement **{model_corr:.3g}**. H6 is **{statuses.get('H6','Inconclusive')}**. ESS diagnoses the particle approximation to one model; support and model disagreement diagnose a different failure mode.

There are **{stable_wrong}** recorded stable-but-wrong rows under the strict rule: SMC gates pass, particle error is within the declared tolerance, but learned-model error exceeds 0.25. H7 is **{statuses.get('H7','Inconclusive')}**.

## 6. Nonlinear VAR and FitzHugh–Nagumo confirmation

The nonlinear VAR suite spans nonlinear location, state-dependent variance/covariance, heavy tails, switching, multimodality, hidden global state, and calcium filtering across ten generator seeds. FHN uses ten independently parameterized eight-node networks, a converged Euler–Maruyama step, a calcium observation filter, moderate training pulses, and common-random-number paired pulse truth. See Figures 12 and 13 and the row-level synthetic tables.

## 7. NeuroPAL pseudo-OOD outcomes

Observed next frames remain available for whole-animal heldout support strata. The pseudo-OOD table contains {len(pseudo)} row-level score records. In addition to ordinary whole-animal CV, a fixed ridge-Gaussian stress probe is refit after leakage-controlled exclusion of tail regions, a compact train-only PCA cluster, and stimulus-onset windows. These observable-outcome probes test whether support loss predicts degradation; they are diagnostic stress tests, not replacements for the six-family whole-animal benchmark.

## 8. What can and cannot be claimed

Can claim:

- Progressive bridge SMC targets the declared model-specific soft-event or repaired-path law and can stabilize rare-query particle estimates.
- Particle diagnostics, empirical history support, and model disagreement measure distinct risks.
- Matched temporal/population corruptions are distinguishable from smoother coherent histories by the support profile.
- Synthetic oracle rows decompose total error into particle and learned-model components.

Cannot claim:

- SMC repairs extrapolation bias, omitted state, wrong dynamics, or model misspecification.
- Cross-family agreement proves biological truth.
- NeuroPAL predictive queries identify causes, synapses, anatomy, or physical interventions.
- A healthy ESS, stable particle-doubling curve, or agreement across SMC seeds validates an unsupported query.

## Required downstream warning

> This estimate is conditional on the fitted predictive model. Report history support, event rarity, cross-model spread, and SMC convergence together. If history support is low or qualified models disagree materially, treat a numerically stable estimate as scientifically unreliable and do not interpret it causally or anatomically.
"""
    (run_root / "REPORT.md").write_text(report)
    insights = f"""# Insights

- Particle stability and model validity are separate. The experiment found **{stable_wrong}** strict stable-but-wrong rows.
- H1 (matched-cost rare-query advantage): **{statuses.get('H1','Inconclusive')}**. H2 (particle convergence): **{statuses.get('H2','Inconclusive')}**.
- H3 (supported-query robustness): **{statuses.get('H3','Inconclusive')}**. H4 (OOD divergence): **{statuses.get('H4','Inconclusive')}**.
- H5 (coherence discrimination): **{statuses.get('H5','Inconclusive')}**.
- H6 (diagnostic validity): **{statuses.get('H6','Inconclusive')}**. H7 (stable but wrong): **{statuses.get('H7','Inconclusive')}**.
- On NeuroPAL, use only model-implied query-conditioned predictive-effect language.
- The durable warning is: low support or large qualified-model disagreement overrides reassuring SMC diagnostics.
"""
    (run_root / "INSIGHTS.md").write_text(insights)


def analyze(run_root: Path) -> None:
    freeze_protocol(run_root)
    update_status(run_root, "analysis", "running")
    predictive = pd.read_csv(run_root / "predictive_scores.csv")
    board = _qualification(predictive)
    atomic_csv(run_root / "predictive_model_scoreboard.csv", board)
    effects = pd.read_csv(run_root / "query_effects.csv")
    all_disagreement = _model_disagreement(effects)
    atomic_csv(run_root / "model_disagreement_all_models.csv", all_disagreement)
    qualified_families = set(
        board.loc[board.qualification != "Not qualified", "model_family"].astype(str)
    )
    qualified_effects = effects[effects.model_family.astype(str).isin(qualified_families)]
    disagreement = _model_disagreement(qualified_effects)
    disagreement["qualification_scope"] = "predictively_qualified_or_calibration_caveat"
    atomic_csv(run_root / "model_disagreement.csv", disagreement)
    worm, worm_summary, synthetic_system, synthetic_summary = _inference_tables(
        run_root, effects, qualified_families
    )
    atomic_csv(run_root / "worm_level_effects.csv", worm)
    atomic_csv(run_root / "worm_level_inference.csv", worm_summary)
    atomic_csv(run_root / "synthetic_system_level_errors.csv", synthetic_system)
    atomic_csv(run_root / "synthetic_system_inference.csv", synthetic_summary)
    repaired_path = run_root / "repaired_path_effects.csv"
    if repaired_path.exists() and repaired_path.stat().st_size:
        repaired = pd.read_csv(repaired_path)
        repaired = repaired[repaired.particle_count == repaired.particle_count.max()].copy()
        repaired["smc_method"] = "direct_soft_event"
        repaired_disagreement = _model_disagreement(repaired)
        atomic_csv(run_root / "repaired_path_model_disagreement.csv", repaired_disagreement)
    particle = _particle_sensitivity(run_root)
    atomic_csv(run_root / "particle_sensitivity.csv", particle)
    pseudo = _pseudo_ood(run_root)
    atomic_csv(run_root / "pseudo_ood_scores.csv", pseudo)
    primary = _primary_tests(run_root, disagreement)
    atomic_csv(run_root / "primary_tests.csv", primary)
    diagnostics = []
    for name in (
        "analytic_smc_stages.csv", "smc_diagnostics_neural.csv",
        "smc_diagnostics_repaired_path.csv", "smc_diagnostics_synthetic.csv",
    ):
        path = run_root / name
        if path.exists() and path.stat().st_size:
            try:
                diagnostics.append(pd.read_csv(path).assign(source_table=name))
            except pd.errors.EmptyDataError:
                pass
    atomic_csv(run_root / "smc_diagnostics.csv", pd.concat(diagnostics, ignore_index=True, sort=False))
    _figures(run_root, board, disagreement, particle, pseudo)
    _reports(run_root, board, disagreement, primary, pseudo)
    update_status(run_root, "analysis", "complete", figures=len(list((run_root / "figures").glob("*.png"))))


def validate(run_root: Path, *, run_tests: bool = True) -> None:
    protocol = freeze_protocol(run_root)
    update_status(run_root, "validation", "running")
    required = [
        "AUDIT.md", "protocol.json", "status.json", "fold_assignments.csv",
        "model_manifest.csv", "query_manifest.csv", "support_calibration.csv",
        "predictive_scores.csv", "query_effects.csv", "model_disagreement.csv",
        "smc_diagnostics.csv", "particle_sensitivity.csv", "pseudo_ood_scores.csv",
        "synthetic_system_manifest.csv", "synthetic_oracle_effects.csv",
        "synthetic_model_effects.csv", "synthetic_error_decomposition.csv",
        "repaired_path_effects.csv", "repaired_path_model_disagreement.csv",
        "pseudo_ood_stress_scores.csv", "pseudo_ood_stress_manifest.csv",
        "sampler_sensitivity.csv", "model_sensitivity_scores.csv",
        "model_sensitivity_manifest.csv",
        "worm_level_effects.csv", "worm_level_inference.csv",
        "synthetic_system_level_errors.csv", "synthetic_system_inference.csv",
        "primary_tests.csv", "INSIGHTS.md", "REPORT.md",
    ]
    missing = [name for name in required if not (run_root / name).exists()]
    empty = [name for name in required if (run_root / name).exists() and (run_root / name).stat().st_size == 0]
    test_returncode = None
    test_output = "not run"
    if run_tests:
        command = [
            str(Path(__file__).resolve().parents[1] / ".venv/bin/python"),
            "-m", "pytest",
            "--import-mode=importlib",
            "query_ood_robustness/tests",
            "conditional_neural_benchmark/tests",
            "compatibility_neural_benchmark/tests",
            "-q",
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"},
        )
        test_returncode = completed.returncode
        test_output = completed.stdout
        (run_root / "regression_test_results.txt").write_text(test_output)
    current_source_errors = []
    for relative, expected in protocol["source_sha256"].items():
        path = Path(__file__).resolve().parents[1] / relative
        if not path.exists() or sha256(path) != expected:
            current_source_errors.append(relative)
    csv_errors = []
    for path in run_root.glob("*.csv"):
        try:
            pd.read_csv(path)
        except Exception as error:
            csv_errors.append(f"{path.name}: {error!r}")
    figures = list((run_root / "figures").glob("*.png"))
    result = {
        "status": "pass" if not missing and not empty and not current_source_errors and not csv_errors and (test_returncode in (None, 0)) and len(figures) >= 15 else "fail",
        "protocol_fingerprint": protocol["fingerprint"],
        "missing_artifacts": missing,
        "empty_artifacts": empty,
        "source_hash_errors": current_source_errors,
        "csv_replay_errors": csv_errors,
        "regression_test_returncode": test_returncode,
        "regression_test_last_line": test_output.strip().splitlines()[-1] if test_output.strip() else "",
        "figure_count": len(figures),
        "neural_primary_model_families": sorted(pd.read_csv(run_root / "predictive_scores.csv").model_family.unique()) if (run_root / "predictive_scores.csv").exists() else [],
        "synthetic_generator_systems": int(pd.read_csv(run_root / "synthetic_system_manifest.csv").generator_seed.nunique()) if (run_root / "synthetic_system_manifest.csv").exists() else 0,
        "claim_boundary": "validation is numerical/protocol validation, not biological causal confirmation",
        "known_limitations": [
            "tail/cluster/context exclusions use a fixed ridge-Gaussian diagnostic probe, not six new full-family refits",
            "synthetic model-specific truth uses high-precision finite Monte Carlo and retains its MCSE",
        ],
    }
    atomic_json(run_root / "validation.json", result)
    paths = sorted(
        path for path in run_root.rglob("*")
        if path.is_file() and path.name not in {"checksums.sha256"}
    )
    lines = [f"{sha256(path)}  {path.relative_to(run_root)}" for path in paths]
    (run_root / "checksums.sha256").write_text("\n".join(lines) + "\n")
    update_status(run_root, "validation", "complete" if result["status"] == "pass" else "failed", **result)
    if result["status"] != "pass":
        raise RuntimeError(f"validation failed: {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("analyze", "validate"))
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    if args.stage == "analyze":
        analyze(args.run_root.resolve())
    else:
        validate(args.run_root.resolve(), run_tests=not args.skip_tests)


if __name__ == "__main__":
    main()
