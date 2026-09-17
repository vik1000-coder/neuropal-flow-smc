from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _markdown_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    result = [
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(rows[0])) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |",
    ]
    result.extend(
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows[1:]
    )
    return result


def archive_run(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    deployment = json.loads((run_dir / "deployment_manifest.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    board = _read_csv(run_dir / "leaderboard.csv")
    final = sorted(
        (row for row in board if row["phase"] == "final_confirmation"),
        key=lambda row: int(row["rank"]),
    )
    lag_rows = sorted(
        (
            row
            for row in board
            if row["phase"] == "lag_screen" and row["model_id"] == "ridge_full_gaussian"
        ),
        key=lambda row: int(row["lag"]),
    )

    inventory: list[dict[str, str | int]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {
            "checksums.sha256", "artifact_inventory.csv", "RESULTS_SUMMARY.md"
        }:
            continue
        inventory.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    with (run_dir / "artifact_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(inventory)
    (run_dir / "checksums.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in inventory)
    )

    winner = final[0]
    ridge = next(row for row in final if row["model_id"] == "ridge_full_gaussian")
    persistence = next(row for row in final if row["model_id"] == "persistence_full_gaussian")
    improvement = {
        name: 100.0 * (_float(ridge, name) - _float(winner, name)) / _float(ridge, name)
        for name in ("energy__mean", "energy__stim_balanced__mean", "variogram__mean")
    }
    pattern_metrics = _read_csv(run_dir / "trial_metrics.csv")
    pattern_trials = [
        row
        for row in pattern_metrics
        if row["phase"] == "final_confirmation" and row["model_id"] == winner["model_id"]
    ]
    pattern_names = ["off", "onset", "offset", "mixed"]
    pattern_means = {
        label: sum(float(row[f"energy__{label}"]) for row in pattern_trials) / len(pattern_trials)
        for label in pattern_names
    }

    leaderboard_table = [[
        "Rank", "Model", "Energy", "Balanced", "Variogram", "RMSE", "90% coverage", "Exact NLL/neuron"
    ]]
    for row in final:
        nll = row["nll_per_neuron__mean"]
        leaderboard_table.append(
            [
                row["rank"],
                row["model_id"],
                f"{_float(row, 'energy__mean'):.6f}",
                f"{_float(row, 'energy__stim_balanced__mean'):.6f}",
                f"{_float(row, 'variogram__mean'):.6f}",
                f"{_float(row, 'rmse__mean'):.6f}",
                f"{_float(row, 'coverage90__mean'):.4f}",
                f"{float(nll):.6f}" if nll else "implicit / unavailable",
            ]
        )

    lag_table = [["Lag (frames)", "Seconds", "Energy", "Balanced energy"]]
    for row in lag_rows:
        lag_table.append(
            [
                row["lag"],
                f"{int(row['lag']) / float(manifest['fps']):.2f}",
                f"{_float(row, 'energy__mean'):.6f}",
                f"{_float(row, 'energy__stim_balanced__mean'):.6f}",
            ]
        )

    lines = [
        "# Frozen conditional NeuroPAL distribution benchmark",
        "",
        f"Run ID: `{manifest['run_id']}`  ",
        f"Completed locally: `{manifest['finished_local']}`  ",
        f"Status: **{manifest['status']}** ({status['ok']} successful trials, {status['failed']} failures)  ",
        f"Integrity inventory: `{len(inventory)}` pre-existing run artifacts are recorded in `artifact_inventory.csv` and `checksums.sha256`; this generated summary is intentionally excluded to avoid a circular self-checksum.",
        "",
        "## Exact estimand",
        "",
        r"The benchmark learns $p(x_{t+1}\mid x_{t-L+1:t},s_{t-L+1:t})$, where $x$ is the observed 54-neuron calcium vector and $s$ is the exactly aligned binary stimulus history. The horizon is one frame (0.25 seconds). This is a predictive conditional law, not an anatomical or causal graph.",
        "",
        "## Data and leakage controls",
        "",
        f"- {manifest['n_worms']} worms ({sum(1 for x in _read_csv(run_dir / 'fold_assignments.csv') if x['strain'] == 'OH16230')} OH16230 and {sum(1 for x in _read_csv(run_dir / 'fold_assignments.csv') if x['strain'] == 'OH15500')} OH15500), {manifest['n_neurons']} retained neurons, {manifest['n_frames']:,} frames at {manifest['fps']:.1f} Hz.",
        f"- Quality-filtered coordinates: {', '.join(manifest['quality_dropped_neurons'])}.",
        "- Every split is by whole animal. Scaling, early stopping, and model selection are training-fold operations.",
        "- Isolated missing history values use causal carry-forward. Missing next-frame targets are excluded rather than imputed.",
        "- Fold membership is frozen in `fold_assignments.csv`.",
        "",
        "## Lag selection",
        "",
        f"The selected history length is **L={manifest['selected_lag']} frames ({manifest['selected_lag'] / manifest['fps']:.2f} seconds)**. Candidate lags were screened with a nested-alpha multiscale ridge distributional baseline; rare stimulus-transition windows were retained and natural-frequency scores were population-reweighted.",
        "",
        *_markdown_table(lag_table),
        "",
        "## Final whole-animal cross-validation leaderboard",
        "",
        "Five held-out-animal folds and three seeds were used for every neural finalist. Lower is better except coverage, whose nominal target is 0.90.",
        "",
        *_markdown_table(leaderboard_table),
        "",
        "## Selected model",
        "",
        f"The selected model is **{winner['model_id']}**: a causal TCN history encoder with a residual next-state target and a conditional flow-matching head.",
        "",
        f"- Natural multivariate energy: **{_float(winner, 'energy__mean'):.6f}** (reported trial-level SE {_float(winner, 'energy__se'):.6f}).",
        f"- Stimulus-balanced energy: **{_float(winner, 'energy__stim_balanced__mean'):.6f}**.",
        f"- Variogram score: **{_float(winner, 'variogram__mean'):.6f}**.",
        f"- RMSE: **{_float(winner, 'rmse__mean'):.6f}** fold-standardized units.",
        f"- Marginal 90% coverage: **{_float(winner, 'coverage90__mean'):.4f}**.",
        f"- Versus nested-alpha ridge: {improvement['energy__mean']:.1f}% lower energy, {improvement['energy__stim_balanced__mean']:.1f}% lower balanced energy, and {improvement['variogram__mean']:.1f}% lower variogram.",
        f"- Versus persistence: {100.0 * (_float(persistence, 'energy__mean') - _float(winner, 'energy__mean')) / _float(persistence, 'energy__mean'):.1f}% lower natural energy.",
        "",
        "All five neural finalists fall inside the leaderboard's one-standard-error band. The MDN-4 is therefore a required architecture-sensitivity model rather than a decisively inferior model. Fold heterogeneity dominates seed variation, so the trial-level SE must not be treated as 15 independent biological replicates.",
        "",
        "### Winner energy by stimulus-history pattern",
        "",
        *(f"- {label}: {pattern_means[label]:.6f}" for label in pattern_names),
        "",
        "## Deployment ensemble",
        "",
        f"Three all-worm `{deployment['model_id']}` fits use the cross-validated median duration of {deployment['epochs_transferred_from_cv_median']} epochs. They are for exploration/deployment only and have no unbiased test score.",
        "",
        *(f"- seed {member['seed']}: `{member['checkpoint']}`" for member in deployment["members"]),
        "",
        "## Reuse and interpretation",
        "",
        "- Use `conditional_neural_benchmark.inference.sample_next` for one-step conditional samples in the original neural coordinate system.",
        "- Use fold-specific `final_confirmation` checkpoints for scientific evaluation on held-out worms.",
        "- Flow matching provides samples but no exact normalized likelihood; MDN/Gaussian rows retain exact likelihood evaluation.",
        "- Results concern observed, stimulus-conditioned history-to-future activity. They do not identify latent anatomical rewiring, direct synapses, or physical interventions.",
        "",
        "## Artifact map",
        "",
        "- `leaderboard.csv`: aggregate phase-specific ranking.",
        "- `trial_metrics.csv`: every fold/seed trial and stimulus-pattern metric.",
        "- `fold_assignments.csv`: immutable animal-level folds.",
        "- `manifest.json`, `status.json`: provenance and completion state.",
        "- `deployment_manifest.json`: all-worm ensemble metadata and evaluation warning.",
        "- `checkpoints/`: architecture screen, final confirmation, lag sensitivity, and deployment weights.",
        "- `figures/`: lag, final-leaderboard, and fold-heterogeneity figures.",
        "- `artifact_inventory.csv`, `checksums.sha256`: durable file inventory and integrity hashes.",
        "",
    ]
    (run_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    archive_run(args.run_dir)


if __name__ == "__main__":
    main()
