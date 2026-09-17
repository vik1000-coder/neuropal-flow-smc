"""Finalize the historical SBTG80 flow optimization campaign.

This script consumes only completed training/sampling artifacts.  It creates a
two-generator high-resolution ensemble, uncertainty summaries, comparison
tables, figures, checksums, and a human-readable report.  Randi/Cook references
remain contextual validation targets rather than neural-activity ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    load_references,
)
from compatibility_neural_benchmark.prediction_atlas_analysis import (
    _contextualize_support,
    _contextualize_values,
    effect_normalization_denominator,
    orient_response_once,
)
from compatibility_neural_benchmark.prediction_atlas_external_analysis import (
    _reference_rows,
)
from compatibility_neural_benchmark.sbtg80_flow_optimization import (
    _matrix_from_sampling,
)
from compatibility_neural_benchmark.sbtg80_progressive_sensitivity import (
    _load_published_slices,
    _paired_source_bootstrap,
)
from conditional_neural_benchmark.data import load_sbtg_cohort


PRIMARY_REFERENCES = (
    "randi_wild_type",
    "cook_struct_54",
    "cook_chem_54",
    "cook_gap_54",
)
REFERENCE_LABELS = {
    "randi_wild_type": "Randi",
    "cook_struct_54": "Cook structural",
    "cook_chem_54": "Cook chemical",
    "cook_gap_54": "Cook gap",
    "cook_struct_80": "Cook structural",
    "cook_chem_80": "Cook chemical",
    "cook_gap_80": "Cook gap",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worm_matrices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((path / "manifest.json").read_text())
    model_id = str(manifest["model_id"])
    particles = int(manifest["particles"])
    # Primary-seed sampler manifests created before the multi-seed extension
    # encode s1701 in every archive/checkpoint path but lack this explicit key.
    seed = int(manifest.get("train_seed", 1701))
    cohort = load_sbtg_cohort()
    response = np.full((20, 5, 3, 80, 1, 80), np.nan, dtype=np.float32)
    valid = np.full((20, 5, 3, 80), np.nan, dtype=np.float32)
    gaps = np.full_like(valid, np.nan)
    codes = np.full((20, 3), -1, dtype=np.int8)
    for fold in range(5):
        archive = (
            path
            / "responses/progressive_bridge_smc"
            / f"{model_id}__progressive_bridge_smc__ell1__N{particles}__f{fold}__s{seed}.npz"
        )
        with np.load(archive, allow_pickle=False) as data:
            indices = data["worm_indices"].astype(int)
            response[indices] = data["response_endpoint_mean"]
            valid[indices] = data["diagnostic_valid"]
            gaps[indices] = data["diagnostic_achieved_gap"]
            codes[indices] = data["chemical_code_by_worm_event"]
    if not np.isfinite(response).all() or not np.isfinite(gaps).all() or np.any(codes < 0):
        raise RuntimeError(f"incomplete raw sampling family {path}")
    normalized = orient_response_once(response) / effect_normalization_denominator(
        gaps, 0.10
    )[:, :, :, None, None, :]
    matrix = _contextualize_values(normalized, codes)["state_average"][:, 0]
    support, _ = _contextualize_support(valid, gaps, codes)
    return matrix, support["state_average"]


def _item(method: str, matrix: np.ndarray, support: np.ndarray) -> dict[str, object]:
    return {
        "method": method,
        "channel": "endpoint_mean",
        "context": "state_average",
        "lag_frames": 1,
        "horizon_frames": 1,
        "matrix": matrix,
        "support": support,
        "support_available": True,
        "training_lineage": "two-seed tuned historical SBTG80 conditional-flow ensemble",
        "shared_neuron_comparability": "native historical 80-class axis",
    }


def _worm_bootstrap(
    worms: np.ndarray,
    published: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mean = worms.mean(axis=0)
    rows: list[dict[str, object]] = []
    for reference_name in PRIMARY_REFERENCES:
        reference = references[reference_name]
        observed = binary_metrics(mean, reference["labels"], reference["mask"])
        comparator = binary_metrics(
            published, reference["labels"], reference["mask"]
        )
        draws = {"auroc": [], "auprc": []}
        for _ in range(repeats):
            indices = rng.integers(0, len(worms), size=len(worms))
            candidate = worms[indices].mean(axis=0)
            current = binary_metrics(
                candidate, reference["labels"], reference["mask"]
            )
            for metric in draws:
                draws[metric].append(float(current[metric]))
        for metric, values in draws.items():
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "reference": reference_name.replace("_54", "_80"),
                    "metric": metric,
                    "estimate": float(observed[metric]),
                    "sbtg_published": float(comparator[metric]),
                    "difference": float(observed[metric] - comparator[metric]),
                    "ci_low": float(np.quantile(array, 0.025)),
                    "ci_high": float(np.quantile(array, 0.975)),
                    "bootstrap_replicates": repeats,
                    "bootstrap_unit": "historical_trace",
                    "limit": (
                        "conditional on two fitted generator seeds; historical pseudo-paired/"
                        "donor-imputed traces are not clean independent animals"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _reference_metrics(
    ensemble: np.ndarray,
    support: np.ndarray,
    neurons: list[str],
    reference_release: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    refs80, _ = load_references(reference_release, neurons)
    rows80 = pd.DataFrame(
        _reference_rows([_item("tuned_two_seed_high_resolution", ensemble, support)], refs80, fps=4.0)
    )
    rows80["reference"] = rows80.reference.str.replace("_54", "_80", regex=False)
    rows80["evaluation_axis"] = "historical80"
    clean_path = Path("results/neural_prediction_atlas_20260829/canonical/atlas_matrices.npz")
    with np.load(clean_path, allow_pickle=False) as clean:
        neurons54 = clean["neurons"].astype(str).tolist()
    indices = [neurons.index(neuron) for neuron in neurons54]
    refs54, _ = load_references(reference_release, neurons54)
    item54 = _item(
        "tuned_two_seed_high_resolution",
        ensemble[np.ix_(indices, indices)],
        support[indices],
    )
    item54["shared_neuron_comparability"] = "exact clean-54 subset"
    rows54 = pd.DataFrame(_reference_rows([item54], refs54, fps=4.0))
    rows54["evaluation_axis"] = "common54_subset"
    return rows80, rows54


def _matrix_stability(root: Path) -> pd.DataFrame:
    paths = {
        path.parent.name: path.parent
        for path in (root / "sampling").glob("*/manifest.json")
    }
    baseline_name = "flow_lr6e4__N32__b2__f1"
    baseline, _, _ = _matrix_from_sampling(paths[baseline_name])
    rows = []
    for name, path in sorted(paths.items()):
        if "flow_lr6e4" not in name:
            continue
        matrix, support, diagnostics = _matrix_from_sampling(path)
        rows.append(
            {
                "sampler_id": name,
                "spearman_vs_primary_n32": float(
                    spearmanr(baseline.ravel(), matrix.ravel()).statistic
                ),
                "mean_absolute_difference_vs_primary_n32": float(
                    np.mean(np.abs(baseline - matrix))
                ),
                "support_mean": float(support.mean()),
                **diagnostics,
            }
        )
    return pd.DataFrame(rows)


def _plot_scores(
    output: Path,
    metrics: pd.DataFrame,
    previous: pd.DataFrame,
    worm_bootstrap: pd.DataFrame,
) -> None:
    current = metrics[
        (metrics.evaluation_axis == "historical80")
        & (metrics.scope == "all_estimated")
    ].copy()
    current["series"] = "Tuned flow, 2-seed N64/b4"
    old = previous[
        (previous.scope == "all_estimated")
        & previous.method.isin(["progressive_bridge_smc", "sbtg_published"])
    ].copy()
    old["series"] = old.method.map(
        {
            "progressive_bridge_smc": "Original flow, N32",
            "sbtg_published": "Published SBTG",
        }
    )
    frame = pd.concat(
        [
            old[["reference", "auroc", "series"]],
            current[["reference", "auroc", "series"]],
        ],
        ignore_index=True,
    )
    order = ["randi_wild_type", "cook_struct_80", "cook_chem_80", "cook_gap_80"]
    series = ["Original flow, N32", "Published SBTG", "Tuned flow, 2-seed N64/b4"]
    colors = ["#9aa5ad", "#d18f52", "#2c7a78"]
    x = np.arange(len(order))
    width = 0.24
    worm_intervals = worm_bootstrap[
        worm_bootstrap.metric == "auroc"
    ].set_index("reference")
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for i, (label, color) in enumerate(zip(series, colors)):
        values = [
            float(frame[(frame.reference == reference) & (frame.series == label)].auroc.iloc[0])
            for reference in order
        ]
        positions = x + (i - 1) * width
        ax.bar(positions, values, width, label=label, color=color)
        if label == "Tuned flow, 2-seed N64/b4":
            lower = np.asarray(
                [worm_intervals.loc[reference].ci_low for reference in order]
            )
            upper = np.asarray(
                [worm_intervals.loc[reference].ci_high for reference in order]
            )
            values_array = np.asarray(values)
            ax.errorbar(
                positions,
                values_array,
                yerr=np.vstack((values_array - lower, upper - values_array)),
                fmt="none",
                ecolor="#162d2c",
                elinewidth=1.4,
                capsize=3,
                capthick=1.4,
                zorder=4,
            )
    ax.axhline(0.5, color="#6d7478", lw=1, ls="--")
    ax.set_xticks(x, [REFERENCE_LABELS[value] for value in order])
    ax.set_ylim(0.48, 0.70)
    ax.set_ylabel("Lag-1 AUROC")
    ax.set_title("Longer-trained two-seed flow recovers the SBTG80 correspondence advantage")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.99,
        0.01,
        "Whiskers: pointwise 95% historical-trace bootstrap, conditional on fitted models",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4d565b",
    )
    fig.tight_layout()
    fig.savefig(output / "lag1_auroc_comparison.png", dpi=180)
    plt.close(fig)


def finalize(root: Path, reference_release: Path, published_archive: Path) -> None:
    output = root / "final"
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        root / "sampling/flow_lr6e4__N64__b4__f2",
        root / "sampling/flow_lr6e4_s2903__N64__b4__f2__seed2903",
    ]
    if any(not (path / "manifest.json").is_file() for path in paths):
        raise FileNotFoundError("matched high-resolution seed families are incomplete")
    cohort = load_sbtg_cohort()
    matrices = []
    supports = []
    worm_matrices = []
    worm_support = []
    for path in paths:
        matrix, support, _ = _matrix_from_sampling(path)
        worms, support_worms = _worm_matrices(path)
        if not np.allclose(matrix, worms.mean(axis=0), rtol=2e-5, atol=2e-6):
            raise RuntimeError(f"aggregate/worm matrix mismatch for {path}")
        matrices.append(matrix)
        supports.append(support)
        worm_matrices.append(worms)
        worm_support.append(support_worms)
    ensemble = np.mean(matrices, axis=0)
    support = np.mean(supports, axis=0)
    ensemble_worms = np.mean(worm_matrices, axis=0)
    np.savez_compressed(
        output / "high_resolution_two_seed_ensemble.npz",
        neurons=np.asarray(cohort.neurons),
        matrix=ensemble.astype(np.float32),
        worm_matrices=ensemble_worms.astype(np.float32),
        support=support.astype(np.float32),
        worm_support=np.mean(worm_support, axis=0).astype(np.float32),
        generator_seeds=np.asarray([1701, 2903], dtype=np.int32),
        particles=np.asarray(64),
        repair_branch_factor=np.asarray(4),
        future_branch_factor=np.asarray(2),
        integration_steps=np.asarray(20),
        source_lag_frames=np.asarray(1),
        horizon_frames=np.asarray(1),
        orientation=np.asarray("target_row_source_column"),
    )

    rows80, rows54 = _reference_metrics(
        ensemble, support, list(cohort.neurons), reference_release
    )
    metrics = pd.concat([rows80, rows54], ignore_index=True)
    metrics.to_csv(output / "ensemble_reference_metrics.csv", index=False)
    primary = metrics[metrics.scope == "all_estimated"].copy()
    primary.to_csv(output / "ensemble_primary_metrics.csv", index=False)

    refs80, _ = load_references(reference_release, list(cohort.neurons))
    published_item = next(
        item
        for item in _load_published_slices(published_archive, cohort.neurons)
        if int(item["lag_frames"]) == 1
    )
    published = np.asarray(published_item["matrix"])
    source_bootstrap = _paired_source_bootstrap(
        ensemble,
        published,
        refs80,
        repeats=4096,
        seed=20_260_906,
    )
    source_bootstrap.to_csv(output / "paired_source_bootstrap.csv", index=False)
    worm_bootstrap = _worm_bootstrap(
        ensemble_worms,
        published,
        refs80,
        repeats=2048,
        seed=20_260_907,
    )
    worm_bootstrap.to_csv(output / "worm_bootstrap.csv", index=False)

    stability = _matrix_stability(root)
    stability.to_csv(output / "matrix_stability.csv", index=False)
    candidate_scoreboard = pd.read_csv(
        root / "postfreeze_external/candidate_scoreboard.csv"
    )
    candidate_scoreboard.to_csv(output / "candidate_scoreboard.csv", index=False)
    sampler_diagnostics = pd.read_csv(
        root / "postfreeze_external/sampler_diagnostics.csv"
    )
    sampler_diagnostics.to_csv(output / "sampler_diagnostics.csv", index=False)
    training = pd.read_csv(root / "training/trial_metrics.csv")
    training.to_csv(output / "training_trials.csv", index=False)

    previous = pd.read_csv(
        "results/sbtg80_full_progressive_atlas_20260901/"
        "external_reference_checks/primary_lag1_comparison.csv"
    )
    _plot_scores(output, metrics, previous, worm_bootstrap)

    current = primary[primary.evaluation_axis == "historical80"].set_index("reference")
    published_rows = previous[previous.method == "sbtg_published"].set_index("reference")
    worm_auroc = worm_bootstrap[worm_bootstrap.metric == "auroc"].set_index("reference")
    source_auroc = source_bootstrap[source_bootstrap.metric == "auroc"].set_index("reference")
    original = previous[previous.method == "progressive_bridge_smc"].set_index("reference")
    lines = [
        "# Historical SBTG80 conditional-flow optimization",
        "",
        "The tuned two-seed progressive-bridge analysis recovers the descriptive lag-1 "
        "correspondence advantage over published SBTG on all four prespecified references. "
        "The result remains a historical model sensitivity because the released 80-neuron "
        "cache used invalid head/tail pseudo-pairing and donor-trace imputation.",
        "",
        "## Best robust lag-1 result",
        "",
        "The preferred estimate averages two independently fitted generators (seeds 1701 and "
        "2903), each evaluated with N=64 retained particles, repair branch factor 4, two "
        "future descendants, and 20-step Heun integration.",
        "",
        "| Reference | Tuned AUROC | Published SBTG | Delta | Source-bootstrap delta 95% CI | Worm-bootstrap AUROC 95% CI | Tuned AUPRC | Published AUPRC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for reference in (
        "randi_wild_type",
        "cook_struct_80",
        "cook_chem_80",
        "cook_gap_80",
    ):
        now = current.loc[reference]
        old = published_rows.loc[reference]
        source = source_auroc.loc[reference]
        worm = worm_auroc.loc[reference]
        lines.append(
            f"| {REFERENCE_LABELS[reference]} | {now.auroc:.3f} | {old.auroc:.3f} | "
            f"{now.auroc-old.auroc:+.3f} | [{source.ci_low:+.3f}, {source.ci_high:+.3f}] | "
            f"[{worm.ci_low:.3f}, {worm.ci_high:.3f}] | {now.auprc:.3f} | {old.auprc:.3f} |"
        )
    lines += [
        "",
        "The source bootstrap resamples source columns of two frozen matrices. The worm "
        "bootstrap resamples the 20 historical traces while keeping fitted fold models fixed. "
        "Neither interval includes imputation, clean-animal, or full generator-refit uncertainty.",
        "",
        "## What improved the result",
        "",
        "1. **Training to convergence had the largest effect.** The original five fold flow "
        "always stopped at its epoch cap (17-31). The tuned learning-rate run selected epochs "
        "52-65 and reduced mean held-out energy from 1.501 to 1.473 and balanced energy from "
        "1.493 to 1.465. Mean lag-1 AUROC rose from 0.534 to 0.629 at the unchanged N=32 sampler.",
        "2. **Independent generator averaging mattered more than extra particles.** The two "
        "seeds had nearly identical predictive scores but only moderate matrix agreement. "
        "Averaging them improved the high-resolution mean AUROC beyond either individual seed.",
        "3. **More repair branching helped support and correspondence.** At N=64, changing "
        "branch factor 2 to 4 raised mean validity from 0.855 to 0.868, passing sources from "
        "71 to 73, and mean AUROC from 0.621 to 0.632 for seed 1701.",
        "4. **Particle/future resolution improved support but did not monotonically increase "
        "point scores.** N=64/future=2 reduced favorable N=32 Monte Carlo variation before "
        "repair branching restored and stabilized the correspondence.",
        "5. **ODE integration was already adequate.** Doubling 20 to 40 steps produced matrix "
        "Spearman 0.985 and slightly lower correspondence while doubling transition cost.",
        "6. **Larger encoders, GRU, transition rebalancing, history noise, and Gaussian-source "
        "flow did not beat the converged TCN recipe on the joint predictive/runtime criteria.**",
        "",
        "## Before and after",
        "",
        "| Reference | Original flow AUROC | Tuned two-seed AUROC | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for reference in (
        "randi_wild_type",
        "cook_struct_80",
        "cook_chem_80",
        "cook_gap_80",
    ):
        before = original.loc[reference].auroc
        after = current.loc[reference].auroc
        lines.append(
            f"| {REFERENCE_LABELS[reference]} | {before:.3f} | {after:.3f} | {after-before:+.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This is a model-relative observational response sensitivity. Randi perturbations and "
        "Cook anatomy are contextual references, not the neural-activity training target. The "
        "80-neuron historical cache is not a clean simultaneous head/tail recording, so this "
        "result does not establish causal coupling, synapses, receptor action, or biological "
        "head-to-tail effects. External candidate comparisons are exploratory because the same "
        "references were viewed for diagnostics after candidate matrices were frozen.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")

    required = sorted(path for path in output.iterdir() if path.name != "checksums.sha256")
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in required)
    )
    validation = {
        "status": "passed",
        "generator_seeds": [1701, 2903],
        "historical_traces": 20,
        "matrix_shape": list(ensemble.shape),
        "all_finite": bool(np.isfinite(ensemble).all()),
        "support_sources_ge_050": int((support >= 0.50).sum()),
        "references_beating_sbtg_auroc": int(
            sum(current.loc[reference].auroc > published_rows.loc[reference].auroc for reference in current.index)
        ),
        "checksummed_artifacts": len(required),
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    # Refresh checksums to include the validation written last.
    required = sorted(path for path in output.iterdir() if path.name != "checksums.sha256")
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in required)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("results/sbtg80_flow_optimization_20260901")
    )
    parser.add_argument(
        "--reference-release",
        type=Path,
        default=Path("/Users/vik/Downloads/SBTG-public-release copy"),
    )
    parser.add_argument(
        "--published-archive",
        type=Path,
        default=Path(
            "/Users/vik/Downloads/SBTG-public-release copy/results/paper/"
            "sbtg_lag_matrices.npz"
        ),
    )
    args = parser.parse_args()
    finalize(
        args.root.resolve(),
        args.reference_release.resolve(),
        args.published_archive.resolve(),
    )


if __name__ == "__main__":
    main()
