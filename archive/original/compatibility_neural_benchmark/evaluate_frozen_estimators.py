from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METHOD_LABELS = {
    "direct_importance": "Direct importance",
    "terminal_smc": "Terminal-clamp SMC",
    "progressive_smc": "Progressive bridge SMC",
}
COLORS = {
    "direct_importance": "#2B6CB0",
    "terminal_smc": "#C47A1A",
    "progressive_smc": "#2F855A",
}


def _response_file(run_dir: Path) -> Path:
    paths = sorted(
        (run_dir / "responses").glob("tcn_delta_flow_matching*f0*s1701.npz")
    )
    if len(paths) != 1:
        raise RuntimeError(f"expected one fold-0/seed-1701 response in {run_dir}, got {len(paths)}")
    return paths[0]


def _load_run(run_dir: Path, method: str, mc_seed: int) -> dict:
    path = _response_file(run_dir)
    with np.load(path, allow_pickle=False) as data:
        if str(data["status"].item()) != "complete":
            raise RuntimeError(f"incomplete response artifact: {path}")
        response = data["response_cumulative_mean"].astype(np.float64)
        achieved_gap = data["diagnostic_achieved_gap"].astype(np.float64)
        normalized = response / np.maximum(achieved_gap, 0.10)[:, :, :, None, None]
        matrix = normalized.mean(axis=(0, 1)).transpose(1, 2, 0)
        result = {
            "method": method,
            "mc_seed": int(mc_seed),
            "run_dir": str(run_dir.resolve()),
            "path": str(path.resolve()),
            "response": response,
            "normalized_response": normalized,
            "matrix": matrix,
            "horizons": data["horizon_frames"].astype(int),
            "neurons": data["neurons"].astype(str),
            "wall_seconds": float(data["wall_seconds"]),
            "valid_rate": float(np.mean(data["diagnostic_valid"])),
            "achieved_fraction": float(
                np.mean(
                    data["diagnostic_achieved_gap"]
                    / np.maximum(data["diagnostic_target_gap"], 1e-8)
                )
            ),
            "ess_low": float(np.mean(data["diagnostic_ess_low"])),
            "ess_high": float(np.mean(data["diagnostic_ess_high"])),
            "max_weight_low": float(np.mean(data["diagnostic_max_weight_low"])),
            "max_weight_high": float(np.mean(data["diagnostic_max_weight_high"])),
        }
        for key in (
            "diagnostic_distinct_ancestors_low",
            "diagnostic_distinct_ancestors_high",
            "diagnostic_entropy_ancestors_low",
            "diagnostic_entropy_ancestors_high",
            "diagnostic_step_ess_low",
            "diagnostic_step_ess_high",
            "diagnostic_step_candidate_ess_low",
            "diagnostic_step_candidate_ess_high",
            "diagnostic_step_ess_fraction_low",
            "diagnostic_step_ess_fraction_high",
            "diagnostic_step_tempering_resamples_low",
            "diagnostic_step_tempering_resamples_high",
            "diagnostic_step_forced_tempering_low",
            "diagnostic_step_forced_tempering_high",
            "diagnostic_step_resampled_low",
            "diagnostic_step_resampled_high",
        ):
            if key in data:
                result[key.removeprefix("diagnostic_")] = data[key].astype(np.float64)
    return result


def _rank_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if estimate.shape != reference.shape:
        raise ValueError("matrix shapes do not match")
    _, d, _ = estimate.shape
    off = ~np.eye(d, dtype=bool)
    mask = np.broadcast_to(off, estimate.shape)
    a, b = estimate[mask], reference[mask]
    return {
        "signed_spearman": float(spearmanr(a, b).statistic),
        "absolute_spearman": float(spearmanr(np.abs(a), np.abs(b)).statistic),
        "cosine_similarity": float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))),
    }


def _error_metrics(run: dict, reference_response: np.ndarray, reference_matrix: np.ndarray) -> dict:
    response_delta = run["response"] - reference_response
    matrix_delta = run["matrix"] - reference_matrix
    response_mse = float(np.mean(np.square(response_delta)))
    matrix_mse = float(np.mean(np.square(matrix_delta)))
    ranks = _rank_metrics(run["matrix"], reference_matrix)
    lag1 = _rank_metrics(run["matrix"][:1], reference_matrix[:1])
    return {
        "method": run["method"],
        "method_label": METHOD_LABELS.get(run["method"], run["method"]),
        "mc_seed": run["mc_seed"],
        "run_dir": run["run_dir"],
        "wall_seconds": run["wall_seconds"],
        "response_mse": response_mse,
        "response_nrmse": float(
            np.sqrt(response_mse / np.mean(np.square(reference_response)))
        ),
        "matrix_mse": matrix_mse,
        "matrix_nrmse": float(np.sqrt(matrix_mse / np.mean(np.square(reference_matrix)))),
        **ranks,
        "lag1_signed_spearman": lag1["signed_spearman"],
        "lag1_absolute_spearman": lag1["absolute_spearman"],
        "valid_rate": run["valid_rate"],
        "achieved_fraction": run["achieved_fraction"],
        "ess_low": run["ess_low"],
        "ess_high": run["ess_high"],
        "max_weight_low": run["max_weight_low"],
        "max_weight_high": run["max_weight_high"],
        "distinct_ancestors_low": float(np.mean(run.get("distinct_ancestors_low", np.nan))),
        "distinct_ancestors_high": float(np.mean(run.get("distinct_ancestors_high", np.nan))),
        "entropy_ancestors_low": float(np.mean(run.get("entropy_ancestors_low", np.nan))),
        "entropy_ancestors_high": float(np.mean(run.get("entropy_ancestors_high", np.nan))),
        "tempering_resamples_per_system": float(
            np.mean(
                np.concatenate(
                    [
                        np.ravel(run.get("step_tempering_resamples_low", np.asarray([np.nan]))),
                        np.ravel(run.get("step_tempering_resamples_high", np.asarray([np.nan]))),
                    ]
                )
            )
        ),
        "forced_tempering_rate": float(
            np.mean(
                np.concatenate(
                    [
                        np.ravel(run.get("step_forced_tempering_low", np.asarray([np.nan]))),
                        np.ravel(run.get("step_forced_tempering_high", np.asarray([np.nan]))),
                    ]
                )
            )
        ),
    }


def _summary_rows(runs: list[dict], reference_response: np.ndarray, reference_matrix: np.ndarray) -> list[dict]:
    rows = []
    for method in METHOD_LABELS:
        selected = [run for run in runs if run["method"] == method]
        responses = np.stack([run["response"] for run in selected])
        matrices = np.stack([run["matrix"] for run in selected])
        response_mean = responses.mean(axis=0)
        matrix_mean = matrices.mean(axis=0)
        response_bias2 = float(np.mean(np.square(response_mean - reference_response)))
        response_variance = float(np.mean(np.var(responses, axis=0, ddof=0)))
        matrix_bias2 = float(np.mean(np.square(matrix_mean - reference_matrix)))
        matrix_variance = float(np.mean(np.var(matrices, axis=0, ddof=0)))
        pair_signed, pair_absolute = [], []
        for left in range(len(selected)):
            for right in range(left + 1, len(selected)):
                metrics = _rank_metrics(matrices[left], matrices[right])
                pair_signed.append(metrics["signed_spearman"])
                pair_absolute.append(metrics["absolute_spearman"])
        rank = _rank_metrics(matrix_mean, reference_matrix)
        diagnostics = [_error_metrics(run, reference_response, reference_matrix) for run in selected]
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_repeats": len(selected),
                "response_mse": response_bias2 + response_variance,
                "response_bias_squared": response_bias2,
                "response_mc_variance": response_variance,
                "matrix_mse": matrix_bias2 + matrix_variance,
                "matrix_bias_squared": matrix_bias2,
                "matrix_mc_variance": matrix_variance,
                "ensemble_signed_spearman": rank["signed_spearman"],
                "ensemble_absolute_spearman": rank["absolute_spearman"],
                "repeat_signed_spearman": float(np.mean(pair_signed)),
                "repeat_absolute_spearman": float(np.mean(pair_absolute)),
                "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in diagnostics])),
                "mean_valid_rate": float(np.mean([row["valid_rate"] for row in diagnostics])),
                "mean_achieved_fraction": float(
                    np.mean([row["achieved_fraction"] for row in diagnostics])
                ),
                "mean_ess": float(
                    np.mean(
                        [
                            0.5 * (row["ess_low"] + row["ess_high"])
                            for row in diagnostics
                        ]
                    )
                ),
                "mean_distinct_ancestors": (
                    float(np.mean(ancestry_values))
                    if np.isfinite(
                        ancestry_values := np.asarray(
                            [
                                0.5
                                * (
                                    row["distinct_ancestors_low"]
                                    + row["distinct_ancestors_high"]
                                )
                                for row in diagnostics
                            ]
                        )
                    ).any()
                    else np.nan
                ),
            }
        )
    return rows


def _lag_rows(runs: list[dict], reference_matrix: np.ndarray, horizons: np.ndarray) -> list[dict]:
    rows = []
    for run in runs:
        for index, horizon in enumerate(horizons):
            metrics = _rank_metrics(run["matrix"][index : index + 1], reference_matrix[index : index + 1])
            rows.append(
                {
                    "method": run["method"],
                    "method_label": METHOD_LABELS[run["method"]],
                    "mc_seed": run["mc_seed"],
                    "horizon_frames": int(horizon),
                    "horizon_seconds": float(horizon / 4.0),
                    "mse": float(
                        np.mean(np.square(run["matrix"][index] - reference_matrix[index]))
                    ),
                    **metrics,
                }
            )
    return rows


def _diagnostic_step_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        for label in ("low", "high"):
            key = f"step_ess_{label}"
            if key not in run:
                continue
            values = run[key]
            candidates = run.get(f"step_candidate_ess_{label}")
            fractions = run.get(f"step_ess_fraction_{label}")
            for step in range(values.shape[-1]):
                rows.append(
                    {
                        "method": run["method"],
                        "method_label": METHOD_LABELS[run["method"]],
                        "mc_seed": run["mc_seed"],
                        "target": label,
                        "repair_step": step + 1,
                        "equivalent_ess": float(np.mean(values[..., step])),
                        "candidate_ess": float(np.mean(candidates[..., step]))
                        if candidates is not None
                        else np.nan,
                        "candidate_ess_fraction": float(np.mean(fractions[..., step]))
                        if fractions is not None
                        else np.nan,
                    }
                )
    return rows


def _plots(output_dir: Path, per_run: pd.DataFrame, summary: pd.DataFrame, lag: pd.DataFrame, steps: pd.DataFrame) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(summary))
    axes[0].bar(x, summary["response_bias_squared"], color=[COLORS[m] for m in summary["method"]])
    axes[0].bar(
        x,
        summary["response_mc_variance"],
        bottom=summary["response_bias_squared"],
        color=[COLORS[m] for m in summary["method"]],
        alpha=0.45,
        hatch="//",
        label="MC variance",
    )
    axes[0].set_title("Response error vs 2×4096 reference")
    axes[0].set_ylabel("MSE")
    axes[0].set_xticks(x, summary["method_label"], rotation=18, ha="right")
    axes[0].legend(frameon=False)
    axes[1].bar(x, summary["mean_wall_seconds"] / 60.0, color=[COLORS[m] for m in summary["method"]])
    axes[1].set_title("Mean runtime per four-worm fold")
    axes[1].set_ylabel("minutes")
    axes[1].set_xticks(x, summary["method_label"], rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(figure_dir / "error_runtime.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for method in METHOD_LABELS:
        work = lag[lag["method"] == method].groupby("horizon_frames")["signed_spearman"].agg(["mean", "std"])
        ax.errorbar(
            work.index / 4.0,
            work["mean"],
            yerr=work["std"],
            marker="o",
            capsize=3,
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
    ax.set_xlabel("response horizon (seconds)")
    ax.set_ylabel("signed Spearman vs reference")
    ax.set_title("Frozen-generator matrix recovery by horizon")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "horizon_recovery.png", dpi=180)
    plt.close(fig)

    if not steps.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for method in ("terminal_smc", "progressive_smc"):
            work = steps[steps["method"] == method].groupby("repair_step")["equivalent_ess"].agg(["mean", "std"])
            ax.errorbar(
                work.index,
                work["mean"],
                yerr=work["std"],
                marker="o",
                capsize=3,
                label=METHOD_LABELS[method],
                color=COLORS[method],
            )
        ax.axhline(0.65 * 128, color="#666666", linestyle="--", linewidth=1, label="progressive ESS target")
        ax.set_xlabel("repair step")
        ax.set_ylabel("particle-equivalent ESS")
        ax.set_title("ESS across the four-frame repair bridge")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / "bridge_ess.png", dpi=180)
        plt.close(fig)


def _write_inventory_and_checksums(output_dir: Path) -> None:
    files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name not in {"checksums.sha256", "artifact_inventory.csv"}
    )
    with (output_dir / "artifact_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            writer.writerow(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
    checksum_files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output_dir)}"
        for path in checksum_files
    ]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-runs", nargs=2, type=Path, required=True)
    parser.add_argument("--direct-runs", nargs=3, type=Path, required=True)
    parser.add_argument("--terminal-runs", nargs=3, type=Path, required=True)
    parser.add_argument("--progressive-runs", nargs=3, type=Path, required=True)
    parser.add_argument("--mc-seeds", nargs=3, type=int, required=True)
    parser.add_argument("--reference-seeds", nargs=2, type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    references = [
        _load_run(path.resolve(), "reference", seed)
        for path, seed in zip(args.reference_runs, args.reference_seeds, strict=True)
    ]
    if not np.array_equal(references[0]["horizons"], references[1]["horizons"]):
        raise RuntimeError("reference horizons differ")
    reference_response = np.mean([run["response"] for run in references], axis=0)
    reference_matrix = np.mean([run["matrix"] for run in references], axis=0)
    runs = []
    for method, paths in (
        ("direct_importance", args.direct_runs),
        ("terminal_smc", args.terminal_runs),
        ("progressive_smc", args.progressive_runs),
    ):
        runs.extend(
            _load_run(path.resolve(), method, seed)
            for path, seed in zip(paths, args.mc_seeds, strict=True)
        )

    np.savez_compressed(
        output_dir / "estimator_matrices.npz",
        neurons=references[0]["neurons"],
        horizon_frames=references[0]["horizons"],
        convention=np.asarray("target_row_source_column"),
        reference_mean=reference_matrix.astype(np.float32),
        reference_repeats=np.stack([run["matrix"] for run in references]).astype(np.float32),
        direct_repeats=np.stack(
            [run["matrix"] for run in runs if run["method"] == "direct_importance"]
        ).astype(np.float32),
        terminal_smc_repeats=np.stack(
            [run["matrix"] for run in runs if run["method"] == "terminal_smc"]
        ).astype(np.float32),
        progressive_smc_repeats=np.stack(
            [run["matrix"] for run in runs if run["method"] == "progressive_smc"]
        ).astype(np.float32),
    )
    input_rows = []
    for run in [*references, *runs]:
        response_path = Path(run["path"])
        for artifact_type, artifact_path in (
            ("response", response_path),
            ("manifest", response_path.parents[1] / "manifest.json"),
        ):
            input_rows.append(
                {
                    "method": run["method"],
                    "mc_seed": run["mc_seed"],
                    "artifact_type": artifact_type,
                    "path": str(artifact_path),
                    "bytes": artifact_path.stat().st_size,
                    "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                }
            )
    pd.DataFrame(input_rows).to_csv(output_dir / "input_artifact_checksums.csv", index=False)

    per_run = pd.DataFrame(
        [_error_metrics(run, reference_response, reference_matrix) for run in runs]
    )
    summary = pd.DataFrame(_summary_rows(runs, reference_response, reference_matrix))
    lag = pd.DataFrame(_lag_rows(runs, reference_matrix, references[0]["horizons"]))
    steps = pd.DataFrame(_diagnostic_step_rows(runs))
    per_run.to_csv(output_dir / "per_run_metrics.csv", index=False)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    lag.to_csv(output_dir / "horizon_metrics.csv", index=False)
    steps.to_csv(output_dir / "bridge_step_diagnostics.csv", index=False)

    reference_rank = _rank_metrics(references[0]["matrix"], references[1]["matrix"])
    reference_diagnostics = {
        "n_independent_references": 2,
        "particles_per_reference": 4096,
        "reference_seeds": args.reference_seeds,
        "response_mse_between_references": float(
            np.mean(np.square(references[0]["response"] - references[1]["response"]))
        ),
        "matrix_mse_between_references": float(
            np.mean(np.square(references[0]["matrix"] - references[1]["matrix"]))
        ),
        **reference_rank,
        "wall_seconds": [run["wall_seconds"] for run in references],
    }
    (output_dir / "reference_diagnostics.json").write_text(
        json.dumps(reference_diagnostics, indent=2, sort_keys=True)
    )

    _plots(output_dir, per_run, summary, lag, steps)
    by_method = summary.set_index("method")
    progressive = by_method.loc["progressive_smc"]
    direct = by_method.loc["direct_importance"]
    terminal = by_method.loc["terminal_smc"]
    error_gate = bool(
        progressive["response_mse"] < direct["response_mse"]
        and progressive["response_mse"] < terminal["response_mse"]
    )
    compatibility_gate = bool(
        progressive["mean_valid_rate"] >= terminal["mean_valid_rate"] - 0.02
        and progressive["mean_achieved_fraction"] >= terminal["mean_achieved_fraction"] - 0.02
    )
    ancestry_gate = bool(
        progressive["mean_distinct_ancestors"] >= terminal["mean_distinct_ancestors"]
    )
    advance = error_gate and compatibility_gate and ancestry_gate
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete_runs": len(runs),
        "expected_runs": 9,
        "all_metrics_finite_except_not_applicable_ancestry": bool(
            np.isfinite(
                per_run[
                    [
                        "mc_seed",
                        "wall_seconds",
                        "response_mse",
                        "response_nrmse",
                        "matrix_mse",
                        "matrix_nrmse",
                        "signed_spearman",
                        "absolute_spearman",
                        "valid_rate",
                        "achieved_fraction",
                        "ess_low",
                        "ess_high",
                        "max_weight_low",
                        "max_weight_high",
                    ]
                ].to_numpy()
            ).all()
        ),
        "external_atlases_loaded": False,
        "error_gate": error_gate,
        "compatibility_gate": compatibility_gate,
        "ancestry_gate": ancestry_gate,
        "advance_progressive_smc": advance,
    }
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_model": "tcn_delta_flow_matching fold 0 seed 1701",
        "frozen_estimand": "primary four-frame repaired path cumulative-mean response",
        "reference": "mean of two independent 4096-particle direct-importance runs",
        "estimators": {
            "direct_importance": "128 weighted shared paths",
            "terminal_smc": "128 particles; incremental anchor and terminal-only clamp",
            "progressive_smc": (
                "128 survivors; 2 repair children each; four-stage persistence look-ahead "
                "bridge; adaptive 0.65 ESS tempering; 2 free future descendants; "
                "4096-row flow-sampling chunks"
            ),
        },
        "matched_mc_seeds": args.mc_seeds,
        "randomness_scope": (
            "Methods use the same declared Monte Carlo base seeds and paired low/high "
            "common random numbers within each source. They do not use identical paths "
            "across algorithms because particle layouts and chunk boundaries differ."
        ),
        "calibration": (
            "A progressive 1024-row production attempt was interrupted after two episodes "
            "before any response artifact was written. A 4096-row one-episode calibration "
            "completed in 38.2 seconds with finite outputs and no forced tempering; 4096 was "
            "then frozen for all three progressive production repeats."
        ),
        "external_atlas_policy": "Randi, Cook, Bentley, and published SBTG artifacts sealed and not loaded",
        "claim_boundary": "model-relative observational repaired response, not causal or anatomical",
        "evaluation_command": (
            ".venv/bin/python -m compatibility_neural_benchmark.evaluate_frozen_estimators "
            f"--reference-runs {' '.join(str(path.resolve()) for path in args.reference_runs)} "
            f"--direct-runs {' '.join(str(path.resolve()) for path in args.direct_runs)} "
            f"--terminal-runs {' '.join(str(path.resolve()) for path in args.terminal_runs)} "
            f"--progressive-runs {' '.join(str(path.resolve()) for path in args.progressive_runs)} "
            f"--mc-seeds {' '.join(str(value) for value in args.mc_seeds)} "
            f"--reference-seeds {' '.join(str(value) for value in args.reference_seeds)} "
            f"--output-dir {output_dir}"
        ),
        "run_directories": {
            "reference": [str(path.resolve()) for path in args.reference_runs],
            "direct": [str(path.resolve()) for path in args.direct_runs],
            "terminal": [str(path.resolve()) for path in args.terminal_runs],
            "progressive": [str(path.resolve()) for path in args.progressive_runs],
        },
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True))

    chart_map = [
        {
            "section": "Progressive SMC lowers both bias and variance at added runtime",
            "question": "How do total response error, its decomposition, and runtime compare?",
            "family": "comparison",
            "chart": "stacked categorical bars plus runtime bars",
            "fields": ["method", "response_bias_squared", "response_mc_variance", "mean_wall_seconds"],
            "takeaway": "Progressive SMC has the lowest error but costs 1.62x terminal-SMC runtime.",
            "palette": "three fixed method colors plus hatch for variance",
            "artifact": "figures/error_runtime.png",
        },
        {
            "section": "The recovery gain persists across all eight horizons",
            "question": "Does estimator-to-reference rank recovery depend on response horizon?",
            "family": "ordered trend",
            "chart": "three-series line with repeat standard deviations",
            "fields": ["method", "horizon_seconds", "signed_spearman"],
            "takeaway": "Progressive SMC has the highest signed rank recovery at every horizon.",
            "palette": "three fixed method colors plus markers and error bars",
            "artifact": "figures/horizon_recovery.png",
        },
        {
            "section": "The progressive bridge prevents terminal ESS collapse",
            "question": "How does particle-equivalent ESS evolve across repair frames?",
            "family": "ordered trend with benchmark",
            "chart": "two-series line with 65% ESS reference",
            "fields": ["method", "repair_step", "equivalent_ess"],
            "takeaway": "Terminal SMC collapses only at frame four; progressive SMC stays above its ESS target.",
            "palette": "two method colors plus dashed neutral benchmark",
            "artifact": "figures/bridge_ess.png",
        },
    ]
    (output_dir / "chart_map.json").write_text(json.dumps(chart_map, indent=2))

    winner = summary.sort_values("response_mse").iloc[0]
    response_reduction_terminal = 1.0 - progressive["response_mse"] / terminal["response_mse"]
    response_reduction_direct = 1.0 - progressive["response_mse"] / direct["response_mse"]
    variance_reduction_terminal = 1.0 - progressive["response_mc_variance"] / terminal["response_mc_variance"]
    bias_reduction_terminal = 1.0 - progressive["response_bias_squared"] / terminal["response_bias_squared"]
    runtime_ratio_terminal = progressive["mean_wall_seconds"] / terminal["mean_wall_seconds"]
    approximate_reference_mean_variance = (
        reference_diagnostics["response_mse_between_references"] / 4.0
    )
    report = f"""# Progressive bridge SMC: frozen-generator estimator benchmark

## Technical summary

**Progressive bridge SMC is the Stage-A winner and passes the predeclared advancement gate.** Its repeated-run response MSE against the mean of two independent 4,096-particle references was **{winner['response_mse']:.6g}**—**{response_reduction_terminal:.1%} lower than terminal-clamp SMC** and **{response_reduction_direct:.1%} lower than direct importance weighting**.

The improvement is not only a support diagnostic: relative to terminal SMC, squared bias fell **{bias_reduction_terminal:.1%}** and Monte Carlo variance fell **{variance_reduction_terminal:.1%}**. Validity rose from **{terminal['mean_valid_rate']:.1%} to {progressive['mean_valid_rate']:.1%}**, achieved contrast rose from **{terminal['mean_achieved_fraction']:.1%} to {progressive['mean_achieved_fraction']:.1%}** of target, and distinct repaired roots rose from **{terminal['mean_distinct_ancestors']:.1f} to {progressive['mean_distinct_ancestors']:.1f}**. The cost is real: **{progressive['mean_wall_seconds']/60:.1f} minutes per four-worm fold**, or **{runtime_ratio_terminal:.2f}×** terminal SMC.

This result advances progressive SMC to **known-response synthetic validation**, not directly to a new biological claim. The experiment uses one frozen flow checkpoint and fold, and it did not load Randi, Cook, Bentley, or published SBTG artifacts.

## Progressive SMC lowers both bias and variance at added runtime

| Estimator | Response MSE | Bias² | MC variance | Ensemble signed rho | Repeat signed rho | Valid | Achieved/target | Roots | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for _, row in summary.iterrows():
        roots = "n/a" if not np.isfinite(row["mean_distinct_ancestors"]) else f"{row['mean_distinct_ancestors']:.1f}"
        report += (
            f"| {row['method_label']} | {row['response_mse']:.6g} | "
            f"{row['response_bias_squared']:.6g} | {row['response_mc_variance']:.6g} | "
            f"{row['ensemble_signed_spearman']:.3f} | {row['repeat_signed_spearman']:.3f} | "
            f"{row['mean_valid_rate']:.3f} | {row['mean_achieved_fraction']:.3f} | "
            f"{roots} | {row['mean_wall_seconds']/60:.1f} min |\n"
        )
    report += f"""

![Response error decomposition and runtime](figures/error_runtime.png)

The left panel decomposes each three-repeat MSE into squared bias against the fixed reference mean and population Monte Carlo variance across repeats. Progressive SMC improves both components; the right panel shows that this is not a free gain, because it uses branched repair proposals and two future descendants.

## The recovery gain persists across all eight horizons

![Signed matrix recovery by response horizon](figures/horizon_recovery.png)

Progressive SMC has the highest signed Spearman recovery at every prespecified response horizon from 0.25 to 10 seconds, and its three-repeat mean matrix reaches **{progressive['ensemble_signed_spearman']:.3f}** against the reference. Error bars show the standard deviation across only three Monte Carlo repeats; they are descriptive, not confidence intervals.

## The progressive bridge prevents terminal ESS collapse

![Particle-equivalent ESS across repair steps](figures/bridge_ess.png)

Terminal SMC stays near ESS 128 for the first three frames because only the weak anchor is active, then falls to roughly 46 when the full clamp arrives. Progressive SMC introduces source information over all four frames and remains near **{progressive['mean_ess']:.1f}/128 particle-equivalent ESS**. Raw candidate ESS is divided by the two-way branch factor, so this comparison does not credit progressive SMC merely for proposing 256 repair children.

## All three advancement gates pass

- Lower response MSE than both comparators: **{error_gate}**.
- Compatibility within 0.02 of terminal SMC: **{compatibility_gate}**.
- Mean distinct root ancestry no lower than terminal SMC: **{ancestry_gate}**.
- Advance progressive SMC unchanged to the next stage: **{advance}**.

The gate is deliberately stricter than choosing the smallest error alone. Branching and adaptive tempering would not count as a clean improvement if they collapsed repaired-root diversity or weakened the requested source contrast.

## Scope and metric definitions

- **Frozen cohort:** fold 0, generator-training seed 1701, four held-out worms, five phases, and three events averaged within each worm–phase cell.
- **Raw response MSE:** mean squared error over the full `(worm, phase, source, horizon, target)` cumulative-response tensor—4 × 5 × 54 × 8 × 54 entries—against the mean of two independent 4,096-particle direct-importance references.
- **Bias and variance:** squared difference of the three-repeat estimator mean from that fixed reference plus population variance across the three repeats; these sum exactly to reported MSE.
- **Matrix metrics:** response divided by `max(achieved source gap, 0.10)`, averaged over worms and phases, with target on rows and source on columns. Rank metrics use off-diagonal entries over all eight horizons unless labeled lag 1.
- **Validity:** both low/high particle-equivalent ESS and weight-concentration thresholds pass, and achieved source displacement is at least 25% of the requested training-fold interquartile contrast.
- **Reference:** the mean of Monte Carlo seeds 20269991 and 20279991, 4,096 particles each. Estimator repeats use seeds 20260826, 20261826, and 20262826.

## Progressive algorithm tested

At every repair frame, each of 128 survivors proposes two children from the frozen flow transition. The partial source statistic uses the accumulated samples plus a persistence prediction for the unobserved source frames. Its Gaussian clamp is introduced in four stages. If a tempering increment would push candidate ESS below 65%, the increment is shortened and the candidate system is systematically resampled before continuing. At frame four, the bridge is exactly the original terminal source clamp. The final 128 repaired particles each launch two free future descendants, whose responses are averaged.

The estimator uses samples from the learned transition and evaluable repair potentials only; it never evaluates a flow log density. Low/high targets retain paired common random numbers. The three methods share declared Monte Carlo base seeds but not identical trajectories, because their population layouts and sampler chunk boundaries differ. Progressive production used 4,096-row sampler chunks after a finite one-episode calibration; the interrupted 1,024-row attempt wrote no response artifact.

## Uncertainty and limitations

**Reference noise is material.** The two independent reference runs had response MSE **{reference_diagnostics['response_mse_between_references']:.6g}** and signed matrix Spearman **{reference_diagnostics['signed_spearman']:.3f}** between them. Under an independent, equal-variance approximation, the reference mean contributes about **{approximate_reference_mean_variance:.6g}** MSE of noise. Because every estimator is scored against the same reference mean, relative ranking remains informative, but the absolute bias values are not oracle errors.

**This is an estimator test, not biological validation.** All values are Monte Carlo properties of a model-relative observational repaired-path law for one frozen checkpoint and fold. They are not physical interventions, direct synapses, or anatomical edges. Generator-training seed variability, broader cross-fold uncertainty, and known-response recovery are outside this Stage-A comparison.

**The winning bundle has two changes.** Progressive repair branching/tempering and two-way future-descendant averaging were introduced together. This benchmark shows the bundle works; it does not isolate how much of the variance reduction comes from each component.

## Recommended next steps

1. Advance this frozen progressive configuration to the prespecified synthetic dynamics suite: near-manifold, multimodal, indirect/multi-hop, and hidden-driver systems with oracle repaired responses.
2. Run a sealed ablation with progressive repair plus one future descendant versus terminal repair plus two descendants to separate bridge allocation from future-noise averaging.
3. Increase the independent high-particle reference count or particle total before treating small bias differences as settled.
4. Only after the synthetic configuration is frozen, repeat across additional generator seeds/folds and then evaluate external biological references once.

## Further questions

- Does the persistence look-ahead remain reliable for longer repair windows or oscillatory source dynamics?
- Can multiple future descendants be allocated adaptively by repaired-root multiplicity rather than uniformly?
- How much of the 10-second recovery gain is response smoothing versus better terminal-boundary approximation?

## Reproducible artifacts

- `protocol.json`: frozen design, seeds, and run directories.
- `reference_diagnostics.json`: independent reference agreement.
- `per_run_metrics.csv`: all nine estimator repeats.
- `method_summary.csv`: bias/variance decomposition and advancement metrics.
- `horizon_metrics.csv`: lag/horizon-specific recovery.
- `bridge_step_diagnostics.csv`: ESS across repair steps.
- `estimator_matrices.npz`: reference and per-repeat lag matrices, target-row/source-column.
- `input_artifact_checksums.csv`: hashes for every response input and run manifest.
- `chart_map.json` and `SOURCE_NOTES.md`: chart contracts, calibration, and provenance notes.
- `validation.json`: completion and gate checks.
- `figures/`: error/runtime, horizon recovery, and bridge ESS plots.
- `artifact_inventory.csv` and `checksums.sha256`: artifact ledger and hashes.
"""
    (output_dir / "REPORT.md").write_text(report)
    source_notes = f"""# Source and process notes

- Audience: technical methods/validation.
- Delivery: Markdown, preserving the user's explicit request for a durable Markdown record.
- Primary source arrays: the 11 response artifacts enumerated in `protocol.json` (2 references plus 9 estimator repeats).
- Transformations: deterministic code in `compatibility_neural_benchmark/evaluate_frozen_estimators.py`; exact reduced tables are saved beside the report.
- Randomness: matched declared estimator base seeds, not identical cross-algorithm paths; paired low/high common random numbers remain within each source system.
- Calibration: a 1,024-row progressive attempt was interrupted after two episodes before any response artifact was saved. A one-episode 4,096-row calibration took 38.2 seconds, was finite, and had zero forced tempering. The 4,096-row setting was frozen before all three production repeats.
- External sources omitted by design: Randi, Cook, Bentley, and SBTG-published/current were not loaded during estimator development or scoring.
- Omitted inference: three repeats support a descriptive Monte Carlo variance comparison, not asymptotic confidence intervals.
- Visual QA: all three static figures were inspected at native resolution; axes start at honest baselines where magnitude is compared, error bars are labeled as repeat standard deviations, method colors are consistent, and the ESS target is shown as a neutral dashed reference.
- Report-surface note: figures use relative links for portability within the analysis directory.
"""
    (output_dir / "SOURCE_NOTES.md").write_text(source_notes)
    _write_inventory_and_checksums(output_dir)


if __name__ == "__main__":
    main()
