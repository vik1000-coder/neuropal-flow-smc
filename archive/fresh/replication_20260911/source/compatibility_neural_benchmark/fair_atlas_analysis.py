from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


METHOD_LABELS = {
    "importance_weighting": "Flow importance weighting",
    "smc_ess": "Flow bootstrap SMC (ESS)",
    "sbtg_current": "SBTG-current",
    "sbtg_published": "SBTG-published",
}
COLORS = {
    "importance_weighting": "#2B6CB0",
    "smc_ess": "#2F855A",
    "sbtg_current": "#C05621",
    "sbtg_published": "#805AD5",
}


def align_square(
    matrix: np.ndarray,
    source_names: list[str],
    target_names: list[str],
    *,
    fill: float = np.nan,
) -> np.ndarray:
    output = np.full((len(target_names), len(target_names)), fill, dtype=np.float64)
    lookup = {str(name): i for i, name in enumerate(source_names)}
    present = [i for i, name in enumerate(target_names) if name in lookup]
    source_index = [lookup[target_names[i]] for i in present]
    output[np.ix_(present, present)] = np.asarray(matrix)[np.ix_(source_index, source_index)]
    return output


def load_smc(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted((run_dir / "responses").glob("*__smc__B4__f*__s*.npz"))
    if not paths:
        raise FileNotFoundError(f"no complete SMC outputs under {run_dir}")
    by_worm: dict[int, list[np.ndarray]] = {}
    by_worm_valid: dict[int, list[np.ndarray]] = {}
    diagnostic_rows = []
    neurons = None
    horizons = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["status"].item()) != "complete":
                continue
            current_neurons = data["neurons"].astype(str)
            current_horizons = data["horizon_frames"].astype(int)
            if neurons is None:
                neurons = current_neurons
                horizons = current_horizons
            elif not np.array_equal(neurons, current_neurons) or not np.array_equal(
                horizons, current_horizons
            ):
                raise RuntimeError("SMC output alignment differs across checkpoints")
            response = data["response_cumulative_mean"].astype(np.float64)
            gap = data["diagnostic_achieved_gap"].astype(np.float64)
            normalized = response / np.maximum(gap, 0.10)[:, :, :, None, None]
            lagged = normalized.mean(axis=1).transpose(0, 2, 3, 1)
            validity = data["diagnostic_valid"].astype(np.float64).mean(axis=1)
            for position, worm in enumerate(data["worm_indices"].astype(int)):
                by_worm.setdefault(worm, []).append(lagged[position])
                by_worm_valid.setdefault(worm, []).append(validity[position])
            diagnostic_rows.append(
                np.asarray(
                    [
                        data["diagnostic_ess_low"].mean(),
                        data["diagnostic_ess_high"].mean(),
                        data["diagnostic_min_step_ess_low"].mean(),
                        data["diagnostic_min_step_ess_high"].mean(),
                        data["diagnostic_distinct_ancestors_low"].mean(),
                        data["diagnostic_distinct_ancestors_high"].mean(),
                        data["diagnostic_valid"].mean(),
                        data["diagnostic_step_resampled_low"][..., :-1].mean(),
                        data["diagnostic_step_resampled_high"][..., :-1].mean(),
                    ],
                    dtype=np.float64,
                )
            )
    if neurons is None or horizons is None:
        raise RuntimeError("SMC outputs were present but none were complete")
    expected_worms = sorted(by_worm)
    if expected_worms != list(range(20)):
        raise RuntimeError(f"SMC outputs do not cover all 20 worms: {expected_worms}")
    animal_effects = np.stack(
        [np.mean(by_worm[worm], axis=0) for worm in expected_worms]
    )
    animal_validity = np.stack(
        [np.mean(by_worm_valid[worm], axis=0) for worm in expected_worms]
    )
    return (
        animal_effects.mean(axis=0),
        animal_validity.mean(axis=0),
        np.asarray(diagnostic_rows),
        np.asarray(neurons),
    )


def binary_metrics(
    score: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    usable = np.asarray(mask, bool) & np.isfinite(score) & np.isfinite(labels)
    y_true = np.asarray(labels[usable] > 0, dtype=np.int8)
    y_score = np.abs(np.asarray(score[usable], dtype=np.float64))
    n_positive = int(y_true.sum())
    n_total = int(len(y_true))
    result: dict[str, float | int] = {
        "n_edges": n_total,
        "n_positive": n_positive,
        "prevalence": n_positive / n_total if n_total else np.nan,
    }
    if n_positive and n_positive < n_total:
        result["auroc"] = float(roc_auc_score(y_true, y_score))
        result["auprc"] = float(average_precision_score(y_true, y_score))
        order = np.argsort(y_score, kind="stable")[::-1]
        predicted = np.zeros(n_total, dtype=bool)
        predicted[order[:n_positive]] = True
        true_positive = int(np.sum(predicted & (y_true > 0)))
        result["precision_at_reference_density"] = true_positive / n_positive
        result["recall_at_reference_density"] = true_positive / n_positive
        result["f1_at_reference_density"] = true_positive / n_positive
    else:
        result.update(
            auroc=np.nan,
            auprc=np.nan,
            precision_at_reference_density=np.nan,
            recall_at_reference_density=np.nan,
            f1_at_reference_density=np.nan,
        )
    return result


def source_bootstrap_ci(
    score: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d = score.shape[1]
    values = {"auroc": [], "auprc": []}
    for _ in range(n_boot):
        columns = rng.integers(0, d, size=d)
        metric = binary_metrics(score[:, columns], labels[:, columns], mask[:, columns])
        for key in values:
            values[key].append(float(metric[key]))
    return {
        f"{key}_{bound}": float(np.nanquantile(samples, quantile))
        for key, samples in values.items()
        for bound, quantile in (("ci_low", 0.025), ("ci_high", 0.975))
    }


def build_reference_adjacency(
    edge_file: Path,
    neurons: list[str],
    *,
    signal: str | None = None,
) -> tuple[np.ndarray, int, int]:
    frame = pd.read_csv(
        edge_file, header=None, names=["source", "target", "signal", "receptor"]
    )
    frame["source"] = frame["source"].astype(str).str.upper().str.strip()
    frame["target"] = frame["target"].astype(str).str.upper().str.strip()
    frame["signal"] = frame["signal"].astype(str).str.lower().str.strip()
    if signal is not None:
        frame = frame[frame["signal"] == signal.lower()]
    lookup = {name: i for i, name in enumerate(neurons)}
    adjacency = np.zeros((len(neurons), len(neurons)), dtype=np.int8)
    for row in frame.itertuples(index=False):
        if row.source in lookup and row.target in lookup and row.source != row.target:
            adjacency[lookup[row.target], lookup[row.source]] = 1
    return adjacency, int(len(frame)), int(adjacency.sum())


def lag1_reference_rows(
    methods: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
    *,
    n_boot: int,
    seed: int,
) -> list[dict]:
    rows = []
    for method_index, (method, artifact) in enumerate(methods.items()):
        lag_index = int(np.flatnonzero(artifact["lags"] == 1)[0])
        signed = artifact["signed"][lag_index]
        for reference_index, (reference, item) in enumerate(references.items()):
            metric = binary_metrics(signed, item["labels"], item["mask"])
            ci = source_bootstrap_ci(
                signed,
                item["labels"],
                item["mask"],
                n_boot=n_boot,
                seed=seed + 1009 * method_index + 97 * reference_index,
            )
            row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "reference": reference,
                **metric,
                **ci,
            }
            positive = item["mask"] & (item["labels"] > 0)
            if "signed_value" in item and positive.sum() >= 3:
                value = item["signed_value"]
                finite = positive & np.isfinite(value)
                row["signed_spearman_on_positive"] = float(
                    spearmanr(signed[finite], value[finite]).statistic
                )
                row["sign_agreement_on_positive"] = float(
                    np.mean(np.sign(signed[finite]) == np.sign(value[finite]))
                )
            else:
                row["signed_spearman_on_positive"] = np.nan
                row["sign_agreement_on_positive"] = np.nan
            if "weight" in item and positive.sum() >= 3:
                row["absolute_score_weight_spearman_on_positive"] = float(
                    spearmanr(
                        np.abs(signed[positive]), np.log1p(item["weight"][positive])
                    ).statistic
                )
            else:
                row["absolute_score_weight_spearman_on_positive"] = np.nan
            rows.append(row)
    return rows


def relationship_rows(
    methods: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    lag1 = {
        name: item["signed"][int(np.flatnonzero(item["lags"] == 1)[0])]
        for name, item in methods.items()
    }
    for scope, mask in masks.items():
        for left, right in itertools.combinations(methods, 2):
            usable = mask & np.isfinite(lag1[left]) & np.isfinite(lag1[right])
            x, y = lag1[left][usable], lag1[right][usable]
            n_top = max(1, int(round(0.10 * len(x))))
            left_top = set(np.argpartition(np.abs(x), -n_top)[-n_top:].tolist())
            right_top = set(np.argpartition(np.abs(y), -n_top)[-n_top:].tolist())
            rows.append(
                {
                    "scope": scope,
                    "left": left,
                    "right": right,
                    "n_edges": int(len(x)),
                    "signed_spearman": float(spearmanr(x, y).statistic),
                    "absolute_spearman": float(spearmanr(np.abs(x), np.abs(y)).statistic),
                    "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
                    "top_10pct_jaccard": len(left_top & right_top)
                    / len(left_top | right_top),
                }
            )
    return rows


def neuromod_rows(
    methods: dict[str, dict[str, np.ndarray]],
    networks: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    d = next(iter(networks.values())).shape[0]
    off = ~np.eye(d, dtype=bool)
    for method, item in methods.items():
        for lag, matrix in zip(item["lags"], item["signed"]):
            for network, adjacency in networks.items():
                eligible_sources = adjacency.any(axis=0)
                for scope, mask in (
                    ("all_pairs_legacy", off),
                    ("eligible_sources", off & eligible_sources[None]),
                ):
                    rows.append(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "lag_frames": int(lag),
                            "lag_seconds": float(lag / 4.0),
                            "network": network,
                            "scope": scope,
                            "n_eligible_sources": int(eligible_sources.sum()),
                            **binary_metrics(matrix, adjacency, mask),
                        }
                    )
    return rows


def save_matrix_csvs(
    output: Path,
    neurons: list[str],
    methods: dict[str, dict[str, np.ndarray]],
) -> None:
    matrix_dir = output / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for method, item in methods.items():
        for lag, matrix in zip(item["lags"], item["signed"]):
            frame = pd.DataFrame(matrix, index=neurons, columns=neurons)
            frame.index.name = "target_by_source"
            frame.to_csv(matrix_dir / f"{method}__lag{int(lag)}.csv")


def create_figures(
    output: Path,
    neurons: list[str],
    methods: dict[str, dict[str, np.ndarray]],
    lag1_metrics: pd.DataFrame,
    relationships: pd.DataFrame,
    neuromod: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for ax, (method, item) in zip(axes.flat, methods.items()):
        matrix = item["signed"][int(np.flatnonzero(item["lags"] == 1)[0])]
        off = matrix[~np.eye(len(neurons), dtype=bool)]
        scale = max(float(np.nanquantile(np.abs(off), 0.98)), 1e-8)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, interpolation="none")
        ax.set_title(METHOD_LABELS[method])
        ticks = np.arange(0, len(neurons), 4)
        ax.set_xticks(ticks, np.asarray(neurons)[ticks], rotation=90, fontsize=6)
        ax.set_yticks(ticks, np.asarray(neurons)[ticks], fontsize=6)
        ax.set_xlabel("source")
        ax.set_ylabel("target")
        fig.colorbar(image, ax=ax, shrink=0.72)
    fig.suptitle("Aligned lag-1 signed matrices (54 shared neurons; method-specific scales)")
    fig.savefig(figures / "lag1_aligned_matrices.png", dpi=180)
    plt.close(fig)

    selected_refs = [
        value
        for value in ("randi_wild_type", "cook_struct_54", "cook_struct_strict44")
        if value in set(lag1_metrics["reference"])
    ]
    fig, axes = plt.subplots(1, len(selected_refs), figsize=(5 * len(selected_refs), 4.8), squeeze=False)
    for ax, reference in zip(axes.flat, selected_refs):
        sub = lag1_metrics[lag1_metrics.reference == reference]
        x = np.arange(len(sub))
        ax.bar(x, sub.auroc, color=[COLORS[name] for name in sub.method])
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks(x, [METHOD_LABELS[name] for name in sub.method], rotation=35, ha="right")
        ax.set_ylim(0.35, max(0.75, float(sub.auroc.max()) + 0.05))
        ax.set_title(reference)
        ax.set_ylabel("lag-1 AUROC")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "lag1_reference_auroc.png", dpi=180)
    plt.close(fig)

    names = list(methods)
    similarity = np.eye(len(names))
    subset = relationships[relationships.scope == "shared_54_offdiagonal"]
    for row in subset.itertuples(index=False):
        i, j = names.index(row.left), names.index(row.right)
        similarity[i, j] = similarity[j, i] = row.absolute_spearman
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    image = ax.imshow(similarity, cmap="viridis", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(names)), [METHOD_LABELS[name] for name in names], rotation=35, ha="right")
    ax.set_yticks(range(len(names)), [METHOD_LABELS[name] for name in names])
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center", color="white" if similarity[i, j] < 0.45 else "black")
    fig.colorbar(image, ax=ax, label="absolute-score Spearman")
    fig.tight_layout()
    fig.savefig(figures / "lag1_method_similarity.png", dpi=180)
    plt.close(fig)

    for scope in ("all_pairs_legacy", "eligible_sources"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        for ax, network in zip(axes, ("monoamine_all", "neuropeptide_all")):
            sub = neuromod[(neuromod.scope == scope) & (neuromod.network == network)]
            for method in methods:
                method_rows = sub[sub.method == method].sort_values("lag_seconds")
                ax.plot(
                    method_rows.lag_seconds,
                    method_rows.auroc,
                    marker="o",
                    color=COLORS[method],
                    label=METHOD_LABELS[method],
                )
            ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
            ax.set_title(network)
            ax.set_xlabel("lag / future horizon (seconds)")
            ax.set_ylabel("AUROC")
            ax.grid(alpha=0.2)
        axes[1].legend(frameon=False, fontsize=8)
        fig.savefig(figures / f"neuromodulator_auroc_by_lag__{scope}.png", dpi=180)
        plt.close(fig)


def write_report(
    output: Path,
    neurons: list[str],
    lag1_metrics: pd.DataFrame,
    relationships: pd.DataFrame,
    neuromod: pd.DataFrame,
    smc_diagnostics: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
) -> None:
    primary_refs = ["randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54"]
    table = lag1_metrics[lag1_metrics.reference.isin(primary_refs)].copy()
    pivot_auc = table.pivot(index="method", columns="reference", values="auroc")
    pivot_ap = table.pivot(index="method", columns="reference", values="auprc")
    relation_primary = relationships[relationships.scope == "shared_54_offdiagonal"]
    best_rows = []
    common_neuromod = neuromod[neuromod.lag_frames.isin([1, 2, 8])]
    for (method, network, scope), frame in common_neuromod.groupby(
        ["method", "network", "scope"]
    ):
        usable = frame[np.isfinite(frame.auroc)]
        if len(usable):
            best_rows.append(usable.loc[usable.auroc.idxmax()])
    best = pd.DataFrame(best_rows)
    best = best[
        (best.scope == "eligible_sources")
        & best.network.isin(["monoamine_all", "neuropeptide_all"])
    ]
    native_best_rows = []
    for (method, network, scope), frame in neuromod.groupby(
        ["method", "network", "scope"]
    ):
        usable = frame[np.isfinite(frame.auroc)]
        if len(usable):
            native_best_rows.append(usable.loc[usable.auroc.idxmax()])
    native_best = pd.DataFrame(native_best_rows)
    native_best = native_best[
        (native_best.scope == "eligible_sources")
        & native_best.network.isin(
            ["monoamine_all", "neuropeptide_all", "neuromodulator_union"]
        )
    ]
    diagnostic_mean = smc_diagnostics.mean(axis=0)

    lines = [
        "# Fair shared-neuron lag, atlas, and neuromodulator comparison",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## What was compared",
        "",
        f"All four model-derived matrices were restricted to the same {len(neurons)} neurons shared by the current 54-neuron cohort and the published 80-neuron SBTG artifact. Nothing was retrained. Matrix orientation is target-by-source throughout, and lag 1 is one 4 Hz frame (0.25 seconds).",
        "",
        "- **Flow importance weighting:** the previously saved direct natural-path importance estimator.",
        "- **Flow bootstrap SMC (ESS):** a new literal bootstrap particle filter using incremental factual-anchor potentials, ESS-triggered systematic resampling, an unconditional terminal resample, and free unweighted future rollout.",
        "- **SBTG-current:** the newly cross-fitted FeatureBilinear SBTG estimator on the 54-neuron cohort.",
        "- **SBTG-published:** the released manuscript lag matrices, subset and reordered to the same 54 neurons. Its lag-1 matrix is the released regime-gated estimate; lags 2+ have the documented production multi-lag lineage.",
        "",
        "The shared-neuron restriction equalizes the evaluated node set, but it cannot equalize how already-frozen models were trained: SBTG-published used its released 80-neuron/imputed-cohort pipeline, whereas the current generators and SBTG-current use the present 54-neuron cohort and their documented split procedures. Keeping both SBTG versions makes that remaining lineage difference visible rather than silently treating them as identical fits.",
        "",
        "Lag 1 is time-matched but not estimand-identical: the flow matrices are repaired-path high-minus-low future responses evaluated at prespecified episode cuts, whereas SBTG matrices are lagged score-product effective-coupling summaries over their respective observational windows. Atlas AUROC compares their rankings, not equality of their numerical entries.",
        "",
        "These are observational effective-response or score-product matrices, not identified synapses or physical interventions.",
        "",
        "## Alignment denominators",
        "",
        f"- Model-to-model matrices: {len(neurons)} neurons, {len(neurons) * (len(neurons) - 1):,} ordered off-diagonal pairs.",
        f"- Randi wild-type: {int(np.sum(references['randi_wild_type']['mask'])):,} confirmed measured pairs on {int(references['randi_wild_type']['present_neurons'].sum())} available shared neurons; ambiguous measured pairs are excluded.",
        f"- Cook primary: all {len(neurons)} model-shared neurons. A strict Cook sensitivity uses the same {int(references['randi_wild_type']['present_neurons'].sum())} neurons available to Randi.",
        "",
        "## Lag-1 atlas results",
        "",
        "Scores are absolute matrix magnitudes. Randi positives are `q < 0.05`; confirmed negatives are `q_eq < 0.05`, with positives taking priority. Cook positives are nonzero structural counts and all off-diagonal nonedges are negatives.",
        "",
        "| Method | Randi WT AUROC / AUPRC | Cook structural | Cook chemical | Cook gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in lag1_metrics.method.drop_duplicates():
        def cell(reference: str) -> str:
            return f"{pivot_auc.loc[method, reference]:.3f} / {pivot_ap.loc[method, reference]:.3f}"

        lines.append(
            f"| {METHOD_LABELS[method]} | {cell('randi_wild_type')} | {cell('cook_struct_54')} | {cell('cook_chem_54')} | {cell('cook_gap_54')} |"
        )
    lines.extend(
        [
            "",
            "Source-column bootstrap intervals and the strict 44-neuron Cook sensitivity are in `lag1_atlas_metrics.csv`. The bootstrap describes source-level sampling variation in a frozen matrix; it is not a generator-refit or animal-refit interval for SBTG-published.",
            "",
            "## How the lag-1 matrices relate",
            "",
            "| Pair | Signed rho | Absolute rho | Sign agreement | Top-10% Jaccard |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in relation_primary.itertuples(index=False):
        lines.append(
            f"| {METHOD_LABELS[row.left]} vs {METHOD_LABELS[row.right]} | {row.signed_spearman:.3f} | {row.absolute_spearman:.3f} | {row.sign_agreement:.3f} | {row.top_10pct_jaccard:.3f} |"
        )
    lines.extend(
        [
            "",
            "## SMC numerical audit",
            "",
            f"Across checkpoint summaries, terminal ESS averaged {diagnostic_mean[0]:.1f} for low repairs and {diagnostic_mean[1]:.1f} for high repairs out of 128 particles. Terminal distinct-root ancestry averaged {diagnostic_mean[4]:.1f} and {diagnostic_mean[5]:.1f}. The mean compatibility-valid query rate was {diagnostic_mean[6]:.3f}.",
            "",
            f"Intermediate ESS-triggered resampling occurred in {diagnostic_mean[7]:.3%} of low and {diagnostic_mean[8]:.3%} of high source-by-step summaries before the mandatory terminal resample. Thus this run genuinely checked the sequential ESS rule; whether it activated was determined by the learned flow and repair potential.",
            "",
            "## Lag correspondence with neuromodulator networks",
            "",
            "Bentley class-level monoamine and neuropeptide edge lists were aligned to the same 54-neuron order. The legacy paper-style scope treats every off-diagonal pair as eligible. Because monoamine positives arise only from a few transmitter-producing sources, the additional eligible-source scope restricts negatives to columns containing at least one reference edge and is the less source-identity-confounded view.",
            "",
            "For a fair between-method best-lag comparison, the table below uses only the three lag frames shared by every artifact: 1, 2, and 8. Full native-grid curves and maxima remain in the CSV outputs.",
            "",
            "| Method | Network | Best common lag (s) | AUROC | AUPRC | Eligible sources |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best.sort_values(["network", "method"]).itertuples(index=False):
        lines.append(
            f"| {METHOD_LABELS[row.method]} | {row.network} | {row.lag_frames} ({row.lag_seconds:.2f}) | {row.auroc:.3f} | {row.auprc:.3f} | {row.n_eligible_sources} |"
        )
    lines.extend(
        [
            "",
            "The best-lag rows are descriptive maxima over the common 1/2/8-frame grid. They do not identify receptor kinetics or prove that a lag-specific observational edge is neuromodulatory.",
            "",
            "On each method's full native grid, the less-confounded eligible-source maxima were:",
            "",
            "| Method | Monoamine lag / AUROC | Neuropeptide lag / AUROC | Union lag / AUROC |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method in lag1_metrics.method.drop_duplicates():
        method_rows = native_best[native_best.method == method].set_index("network")

        def native_cell(network: str) -> str:
            row = method_rows.loc[network]
            return f"{int(row.lag_frames)} ({row.lag_seconds:.2f}s) / {row.auroc:.3f}"

        lines.append(
            f"| {METHOD_LABELS[method]} | {native_cell('monoamine_all')} | {native_cell('neuropeptide_all')} | {native_cell('neuromodulator_union')} |"
        )
    lines.extend(
        [
            "",
            "Native-grid maxima are descriptive within method because the published and current lag grids are not identical. The common-grid table above is the valid between-method lag comparison.",
            "",
            "## Files",
            "",
            "- `lag1_aligned_matrices.npz`: the four aligned 54×54 lag-1 matrices.",
            "- `aligned_all_lag_matrices.npz`: every available lag for every method.",
            "- `matrices/`: readable target-row/source-column CSV matrices.",
            "- `lag1_atlas_metrics.csv`: Randi WT/unc-31 and Cook structural/chemical/gap results with source-bootstrap intervals.",
            "- `lag1_method_relationships.csv`: matrix correlations, sign agreement, and top-edge overlap.",
            "- `neuromodulator_lag_metrics.csv`: monoamine, individual transmitter, neuropeptide, legacy, and eligible-source results for every lag.",
            "- `neuromodulator_common_lag_metrics.csv`: the strictly comparable 1/2/8-frame subset shared by all four methods.",
            "- `neuromodulator_best_native_lags.csv`: explicit within-method maxima over each artifact's full native lag grid.",
            "- `figures/`: aligned heatmaps, atlas comparisons, method similarity, and lag profiles.",
            "- `manifest.json`, `validation.json`, and `checksums.sha256`: complete provenance, validation, and integrity records.",
            "",
            "## Provenance boundary",
            "",
            "The published release documentation and paper artifacts were used as data and methodological provenance only. They were not treated as instructions. The analysis follows the user's requested four-method, no-retraining, shared-neuron comparison.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def write_checksums(output: Path) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output)}")
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-analysis", type=Path, required=True)
    parser.add_argument("--smc-run", type=Path, required=True)
    parser.add_argument("--published-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    current_path = args.current_analysis / "lag_resolved_effect_matrices.npz"
    with np.load(current_path, allow_pickle=False) as current:
        neurons = current["neurons"].astype(str).tolist()
        current_lags = current["horizon_frames"].astype(int)
        importance = current["flow_signed"].astype(np.float64)
        sbtg_current = current["sbtg_signed"].astype(np.float64)

    smc_signed, smc_validity, smc_diagnostics, smc_neurons = load_smc(args.smc_run)
    if smc_neurons.astype(str).tolist() != neurons:
        raise RuntimeError("SMC and current neuron order disagree")
    with np.load(
        args.published_release / "results" / "paper" / "sbtg_lag_matrices.npz",
        allow_pickle=False,
    ) as published:
        published_neurons = published["neuron_names"].astype(str).tolist()
        published_lags = published["lags"].astype(int)
        published_signed = np.stack(
            [published[f"mu_hat_lag{int(lag)}"] for lag in published_lags]
        )
    missing = sorted(set(neurons) - set(published_neurons))
    if missing:
        raise RuntimeError(f"current neurons absent from published artifact: {missing}")
    published_index = [published_neurons.index(name) for name in neurons]
    published_signed = published_signed[:, published_index][:, :, published_index]

    methods = {
        "importance_weighting": {"lags": current_lags, "signed": importance},
        "smc_ess": {"lags": current_lags, "signed": smc_signed},
        "sbtg_current": {"lags": current_lags, "signed": sbtg_current},
        "sbtg_published": {"lags": published_lags, "signed": published_signed},
    }
    for method, item in methods.items():
        if item["signed"].shape[1:] != (len(neurons), len(neurons)):
            raise RuntimeError(f"{method} has wrong aligned matrix shape")
        if 1 not in item["lags"]:
            raise RuntimeError(f"{method} has no lag-1 matrix")

    reference_root = args.published_release / "reference_data"
    references: dict[str, dict[str, np.ndarray]] = {}
    randi_present = np.zeros(len(neurons), dtype=bool)
    for genotype, filename in (
        ("wild_type", "aligned_atlas_wild_type.npz"),
        ("unc31", "aligned_atlas_unc31.npz"),
    ):
        with np.load(reference_root / "functional_atlas" / filename, allow_pickle=False) as atlas:
            names = atlas["neuron_order"].astype(str).tolist()
            q = align_square(atlas["q"], names, neurons)
            q_eq = align_square(atlas["q_eq"], names, neurons)
            dff = align_square(atlas["dff"], names, neurons)
        positive = q < 0.05
        negative = (q_eq < 0.05) & ~positive
        mask = (positive | negative) & ~np.eye(len(neurons), dtype=bool)
        present = np.asarray([name in names for name in neurons])
        if genotype == "wild_type":
            randi_present = present
        references[f"randi_{genotype}"] = {
            "labels": positive.astype(np.int8),
            "mask": mask,
            "signed_value": dff,
            "present_neurons": present,
        }

    cook_names = json.loads((reference_root / "connectome" / "nodes.json").read_text())
    strict44 = randi_present[:, None] & randi_present[None] & ~np.eye(len(neurons), dtype=bool)
    for kind, filename in (
        ("chem", "A_chem.npy"),
        ("gap", "A_gap.npy"),
        ("struct", "A_struct.npy"),
    ):
        weight = align_square(
            np.load(reference_root / "connectome" / filename), cook_names, neurons, fill=0.0
        )
        labels = weight > 0
        off = ~np.eye(len(neurons), dtype=bool)
        references[f"cook_{kind}_54"] = {
            "labels": labels.astype(np.int8),
            "mask": off,
            "weight": weight,
        }
        references[f"cook_{kind}_strict44"] = {
            "labels": labels.astype(np.int8),
            "mask": strict44,
            "weight": weight,
        }

    lag1_rows = lag1_reference_rows(
        methods, references, n_boot=args.bootstrap, seed=args.seed
    )
    lag1_metrics = pd.DataFrame(lag1_rows)
    lag1_metrics.to_csv(output / "lag1_atlas_metrics.csv", index=False)

    relation_masks = {
        "shared_54_offdiagonal": ~np.eye(len(neurons), dtype=bool),
        "randi_confirmed_pairs": references["randi_wild_type"]["mask"],
        "strict_44_offdiagonal": strict44,
    }
    relationships = pd.DataFrame(relationship_rows(methods, relation_masks))
    relationships.to_csv(output / "lag1_method_relationships.csv", index=False)

    edge_root = reference_root / "modulatory_atlas" / "edge_lists"
    networks: dict[str, np.ndarray] = {}
    for signal in (None, "dopamine", "serotonin", "tyramine", "octopamine"):
        adjacency, _, _ = build_reference_adjacency(
            edge_root / "edgelist_MA_classes.csv", neurons, signal=signal
        )
        networks["monoamine_all" if signal is None else f"monoamine_{signal}"] = adjacency
    neuropeptide, _, _ = build_reference_adjacency(
        edge_root / "edgelist_NP_classes.csv", neurons
    )
    networks["neuropeptide_all"] = neuropeptide
    networks["neuromodulator_union"] = (
        (networks["monoamine_all"] + networks["neuropeptide_all"]) > 0
    ).astype(np.int8)
    neuromod = pd.DataFrame(neuromod_rows(methods, networks))
    neuromod.to_csv(output / "neuromodulator_lag_metrics.csv", index=False)
    neuromod[neuromod.lag_frames.isin([1, 2, 8])].to_csv(
        output / "neuromodulator_common_lag_metrics.csv", index=False
    )
    native_best_rows = []
    for (_, _, _), frame in neuromod.groupby(["method", "network", "scope"]):
        usable = frame[np.isfinite(frame.auroc)]
        if len(usable):
            native_best_rows.append(usable.loc[usable.auroc.idxmax()])
    pd.DataFrame(native_best_rows).to_csv(
        output / "neuromodulator_best_native_lags.csv", index=False
    )

    lag1_arrays = {
        name: item["signed"][int(np.flatnonzero(item["lags"] == 1)[0])].astype(np.float32)
        for name, item in methods.items()
    }
    np.savez_compressed(
        output / "lag1_aligned_matrices.npz",
        neurons=np.asarray(neurons),
        lag_frames=np.asarray(1),
        lag_seconds=np.asarray(0.25),
        **lag1_arrays,
    )
    np.savez_compressed(
        output / "aligned_all_lag_matrices.npz",
        neurons=np.asarray(neurons),
        smc_validity=smc_validity.astype(np.float32),
        **{
            f"{name}__lags": item["lags"].astype(np.int16)
            for name, item in methods.items()
        },
        **{
            f"{name}__signed": item["signed"].astype(np.float32)
            for name, item in methods.items()
        },
    )
    long_rows = []
    for method, matrix in lag1_arrays.items():
        for target, target_name in enumerate(neurons):
            for source, source_name in enumerate(neurons):
                long_rows.append(
                    {
                        "method": method,
                        "target": target_name,
                        "source": source_name,
                        "signed_score": float(matrix[target, source]),
                        "is_diagonal": target == source,
                    }
                )
    pd.DataFrame(long_rows).to_csv(output / "lag1_matrices_long.csv", index=False)
    save_matrix_csvs(output, neurons, methods)
    pd.DataFrame(
        [
            {
                "neuron": neuron,
                "smc_valid_rate": float(smc_validity[index]),
                "randi_available": bool(randi_present[index]),
            }
            for index, neuron in enumerate(neurons)
        ]
    ).to_csv(output / "neuron_alignment_and_smc_validity.csv", index=False)

    create_figures(output, neurons, methods, lag1_metrics, relationships, neuromod)
    write_report(
        output,
        neurons,
        lag1_metrics,
        relationships,
        neuromod,
        smc_diagnostics,
        references,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "no-retraining fair shared-neuron lag/atlas/neuromodulator comparison",
        "matrix_orientation": "target row, source column",
        "fps": 4.0,
        "shared_method_neurons": neurons,
        "n_shared_method_neurons": len(neurons),
        "current_analysis": str(args.current_analysis.resolve()),
        "smc_run": str(args.smc_run.resolve()),
        "published_release": str(args.published_release.resolve()),
        "published_lag_lineage": "lag 1 regime-gated; lags 2+ released production multi-lag",
        "randi_label_policy": "positive q<0.05; confirmed negative q_eq<0.05; positive priority; ambiguous excluded",
        "cook_label_policy": "nonzero count is positive; all off-diagonal nonedges negative",
        "neuromodulator_scopes": ["all_pairs_legacy", "eligible_sources"],
        "bootstrap": {
            "unit": "source column",
            "replicates": args.bootstrap,
            "seed": args.seed,
        },
        "claim_boundary": "observational model-relative matrices; external correspondence is not causal or anatomical identification",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "checks": {
            "four_methods_present": set(methods) == set(METHOD_LABELS),
            "all_method_matrices_finite": all(
                np.isfinite(item["signed"]).all() for item in methods.values()
            ),
            "all_method_matrices_are_54_by_54": all(
                item["signed"].shape[1:] == (54, 54) for item in methods.values()
            ),
            "lag1_present_for_every_method": all(
                1 in item["lags"] for item in methods.values()
            ),
            "smc_checkpoint_count": len(
                list((args.smc_run / "responses").glob("*__smc__B4__f*__s*.npz"))
            ),
            "randi_wild_type_confirmed_pairs": int(
                references["randi_wild_type"]["mask"].sum()
            ),
            "randi_shared_neurons": int(randi_present.sum()),
            "lag1_metric_rows": int(len(lag1_metrics)),
            "neuromodulator_metric_rows": int(len(neuromod)),
        },
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True)
    )
    write_checksums(output)


if __name__ == "__main__":
    main()
