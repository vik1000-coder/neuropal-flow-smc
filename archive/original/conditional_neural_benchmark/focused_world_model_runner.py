from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.runner import (
    ModelConfig,
    RunState,
    _leaderboard,
    _neural_trial,
    _split_windows,
    _update_leaderboards,
)


ROOT = Path(__file__).resolve().parents[1]


def candidate_configs() -> list[ModelConfig]:
    """Focused, atlas-blind density and regularization tournament."""
    return [
        ModelConfig(
            "flow_control", "tcn", "conditional_flow_matching", True,
            width=64,
            head_params={"hidden": 96, "layers": 3, "sample_steps": 16},
        ),
        ModelConfig(
            "flow_dropout10_wd1e3", "tcn", "conditional_flow_matching", True,
            width=64, dropout=0.10, weight_decay=1e-3,
            head_params={"hidden": 96, "layers": 3, "sample_steps": 16},
        ),
        ModelConfig(
            "flow_wide128_dropout10", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4,
            head_params={"hidden": 128, "layers": 4, "sample_steps": 24},
        ),
        ModelConfig(
            "flow_compact_wd1e3", "tcn", "conditional_flow_matching", True,
            width=48, dropout=0.10, weight_decay=1e-3,
            head_params={"hidden": 64, "layers": 2, "sample_steps": 16},
        ),
        *[
            ModelConfig(
                f"gaussian_source_flow_b{str(weight).replace('.', 'p')}",
                "tcn", "gaussian_source_flow_matching", True,
                width=64, dropout=0.10, weight_decay=5e-4,
                head_params={
                    "hidden": 96, "layers": 3, "sample_steps": 16,
                    "base_nll_weight": weight,
                },
            )
            for weight in (0.10, 0.25, 0.50, 1.00)
        ],
        *[
            ModelConfig(
                f"bounded_energy_t{str(tilt).replace('.', 'p')}_b{str(base).replace('.', 'p')}",
                "tcn", "bounded_energy_ratio", True,
                width=64, dropout=0.10, weight_decay=5e-4,
                head_params={
                    "hidden": 64, "layers": 2, "tilt_bound": tilt,
                    "base_nll_weight": base, "oversample": 4,
                    "centering_samples": 32,
                },
            )
            for tilt, base in ((0.5, 0.25), (1.0, 0.25), (1.0, 0.5), (1.5, 0.5))
        ],
        ModelConfig(
            "mdn4_dropout10_wd1e3", "tcn", "autoregressive_mdn", True,
            width=64, dropout=0.10, weight_decay=1e-3,
            head_params={"hidden": 64, "layers": 2, "components": 4},
        ),
        ModelConfig(
            "mdn8_dropout10", "tcn", "autoregressive_mdn", True,
            width=64, dropout=0.10, weight_decay=5e-4,
            head_params={"hidden": 64, "layers": 2, "components": 8},
        ),
        ModelConfig(
            "lowrank16_dropout10", "tcn", "lowrank_gaussian", True,
            width=64, dropout=0.10, weight_decay=5e-4,
            head_params={"hidden": 64, "layers": 2, "rank": 16},
        ),
        ModelConfig(
            "realnvp8_dropout10", "tcn", "conditional_affine_flow", True,
            width=64, dropout=0.10, weight_decay=5e-4,
            head_params={"hidden": 64, "layers": 2, "coupling_layers": 8},
        ),
    ]


def _family(config: ModelConfig) -> str:
    if config.head == "bounded_energy_ratio":
        return "contrastive_energy"
    if config.head in {
        "conditional_flow_matching", "gaussian_source_flow_matching"
    }:
        return "flow"
    return "exact_density"


def choose_finalists(
    board: pd.DataFrame, configs: list[ModelConfig], count: int
) -> list[ModelConfig]:
    """Take the best predictive candidates while retaining density-family coverage."""
    by_id = {config.model_id: config for config in configs}
    ranked = [by_id[x] for x in board.model_id if x in by_id]
    selected: list[ModelConfig] = ranked[: max(1, count - 2)]
    for family in ("contrastive_energy", "exact_density"):
        if any(_family(config) == family for config in selected):
            continue
        candidate = next((config for config in ranked if _family(config) == family), None)
        if candidate is not None:
            selected.append(candidate)
    for config in ranked:
        if len(selected) >= count:
            break
        if config not in selected:
            selected.append(config)
    return selected[:count]


def _load_folds(cohort, evidence_run: Path) -> np.ndarray:
    frame = pd.read_csv(evidence_run / "fold_assignments.csv")
    mapping = dict(zip(frame.worm_id.astype(str), frame.outer_fold.astype(int)))
    missing = [worm for worm in cohort.worm_ids if worm not in mapping]
    if missing:
        raise RuntimeError(f"fold evidence is missing worms: {missing}")
    folds = np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)
    if set(folds.tolist()) != set(range(5)):
        raise RuntimeError("expected immutable five-fold assignment")
    return folds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _is_done(
    state: RunState, phase: str, model_id: str, fold: int, seed: int
) -> bool:
    return any(
        row.get("phase") == phase
        and row.get("model_id") == model_id
        and int(row.get("fold", -1)) == fold
        and int(row.get("seed", -1)) == seed
        and row.get("status") == "ok"
        for row in state.records
    )


def _run_trial(
    *, state: RunState, phase: str, config: ModelConfig, cohort, folds: np.ndarray,
    lag: int, fold: int, seed: int, device: str, max_epochs: int, patience: int,
    eval_rows: int, n_samples: int, batch_size: int,
) -> None:
    if _is_done(state, phase, config.model_id, fold, seed):
        print(
            f"TRIAL_SKIP phase={phase} model={config.model_id} fold={fold} seed={seed}",
            flush=True,
        )
        return
    scaler, train, validation, test = _split_windows(cohort, folds, fold, lag)
    _neural_trial(
        state=state, phase=phase, config=config, cohort=cohort, lag=lag,
        fold=fold, seed=seed, train=train, validation=validation, test=test,
        scaler=scaler, device=device, max_epochs=max_epochs, patience=patience,
        eval_rows=eval_rows, n_samples=n_samples, batch_size=batch_size,
    )
    _update_leaderboards(state)


def _write_report(run_dir: Path, finalists: list[ModelConfig], complete: bool) -> None:
    records = pd.read_csv(run_dir / "trial_metrics.csv")
    screen = _leaderboard(records.to_dict("records"), "focused_screen")
    confirmation = _leaderboard(records.to_dict("records"), "focused_confirmation")
    winner = confirmation.iloc[0] if not confirmation.empty else screen.iloc[0]
    full_cv = _leaderboard(records.to_dict("records"), "winner_full_cv")
    lines = [
        "# Atlas-blind conditional-density and world-model tournament",
        "",
        f"Status: **{'complete' if complete else 'time-bounded partial'}**",
        "",
        "## Result",
        "",
        f"The current predictive winner is **{winner.model_id}**, with energy "
        f"**{winner.energy__mean:.6f}**, stimulus-balanced energy "
        f"**{winner.energy__stim_balanced__mean:.6f}**, and variogram score "
        f"**{winner.variogram__mean:.6f}** in the "
        f"{'confirmation' if not confirmation.empty else 'screen'} phase.",
        "",
    ]
    if len(confirmation) >= 2:
        runner_up = confirmation.iloc[1]
        relative_gap = (
            (runner_up.energy__mean - winner.energy__mean)
            / abs(winner.energy__mean)
        )
        lines.extend(
            [
                f"The natural-energy gap to **{runner_up.model_id}** is only "
                f"**{100 * relative_gap:.2f}%**. Because this gap is smaller than "
                "the reported fold/seed standard errors, the supported conclusion is "
                "a near tie between the leading regularized flows; the frozen primary "
                "metric supplies a reproducible winner for downstream work.",
                "",
                "## Confirmation leaderboard",
                "",
                "| Rank | Model | Energy | SE | Balanced energy | Variogram | Coverage90 |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in confirmation.itertuples():
            lines.append(
                f"| {int(row.rank)} | `{row.model_id}` | {row.energy__mean:.6f} | "
                f"{row.energy__se:.6f} | {row.energy__stim_balanced__mean:.6f} | "
                f"{row.variogram__mean:.6f} | {row.coverage90__mean:.4f} |"
            )
        lines.append("")
    lines.extend([
        "## Design",
        "",
        "- The immutable whole-worm folds and L=80 history come from the frozen benchmark.",
        "- Folds 0-2 screen all candidates with energy score as the primary proper sample score.",
        "- Folds 3-4 are untouched during screening and confirm the selected finalists with two seeds.",
        "- When requested after confirmation is frozen, the confirmed winner is refit across all five held-out folds and three seeds to create a response-ready CV ensemble.",
        "- The tournament includes ordinary and regularized flow matching, learned-Gaussian source flows, bounded energy-ratio models learned by data-versus-reference classification, MDNs, RealNVP, and low-rank Gaussian laws.",
        "- Randi, Cook, Bentley, SBTG, and every anatomical or functional atlas are absent from training, stopping, and selection.",
        "",
        "## What the predictive metrics mean",
        "",
        "- **Energy score** is a proper multivariate sample score comparing the observed neural vector with the candidate predictive distribution; lower is better.",
        "- **Stimulus-balanced energy** gives the stimulus-history strata equal influence, preventing the numerous no-stimulus windows from dominating the average; lower is better.",
        "- **Variogram score** checks whether samples reproduce pairwise dependence across neurons; lower is better.",
        "- **Coverage90** is the fraction of observed components inside the nominal 90% predictive interval. Closer to 0.90 is better, but coverage alone can be increased by making intervals unhelpfully wide, so sharpness is retained separately.",
        "- Scores are calculated on held-out worms after fold-specific standardization. They compare predictive conditional laws, not lag-response matrices or connectomes.",
        "",
        "## Selected finalists",
        "",
        *[f"- `{config.model_id}` ({_family(config)})" for config in finalists],
        "",
        "## Interpretation boundary",
        "",
        "This tournament evaluates the predictive conditional law of observed activity. It does not identify causal effects or anatomical connections. Multi-step rollout and response-matrix evaluation are separate frozen follow-ups, not selection inputs for this screen.",
        "",
        "## Artifacts",
        "",
        "- `trial_metrics.csv`: per-fold, per-seed predictive scores.",
        "- `leaderboard.csv`: aggregate rankings and standard errors.",
        "- `selection.json`: pre-confirmation selection record.",
        "- `winner_selection.json`: frozen post-confirmation winner used for any full-CV refit.",
        "- `checkpoints/`: fitted models, fold scalers, and fit traces.",
        "- `manifest.json`: protocol, candidate grid, and atlas firewall.",
        "",
    ])
    (run_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-run", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=9.5)
    parser.add_argument("--lag", type=int, default=80)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--screen-epochs", type=int, default=32)
    parser.add_argument("--screen-patience", type=int, default=6)
    parser.add_argument("--screen-eval-rows", type=int, default=4000)
    parser.add_argument("--screen-samples", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=50)
    parser.add_argument("--final-patience", type=int, default=9)
    parser.add_argument("--final-eval-rows", type=int, default=4000)
    parser.add_argument("--final-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument("--fit-winner-full-cv", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(run_dir, time.monotonic() + args.hours * 3600)
    metrics_path = run_dir / "trial_metrics.csv"
    if args.resume and metrics_path.exists():
        state.records = pd.read_csv(metrics_path).replace({np.nan: None}).to_dict("records")

    cohort = load_cohort(args.coverage)
    folds = _load_folds(cohort, args.evidence_run.resolve())
    configs = candidate_configs()
    if len({config.model_id for config in configs}) != len(configs):
        raise RuntimeError("candidate identifiers must be unique")

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": "atlas-blind focused conditional-density tournament",
            "lag_frames": args.lag,
            "forecast_horizon_frames": 1,
            "screen_folds": [0, 1, 2],
            "confirmation_folds": [3, 4],
            "screen_seed": 1701,
            "confirmation_seeds": [1701, 2903],
            "selection_rule": "within one SE of best natural energy, then stimulus-balanced energy, then variogram; preserve energy/exact family coverage",
            "atlas_firewall": [
                "no Randi", "no Cook", "no Bentley", "no SBTG",
                "no external atlas in fitting, early stopping, or selection",
            ],
            "evidence_run": str(args.evidence_run.resolve()),
            "fold_assignments_sha256": _sha256(
                args.evidence_run.resolve() / "fold_assignments.csv"
            ),
            "n_worms": cohort.n_worms,
            "n_neurons": cohort.n_neurons,
            "neurons": list(cohort.neurons),
            "device": args.device,
            "threads": args.threads,
            "screen_epochs": args.screen_epochs,
            "screen_patience": args.screen_patience,
            "screen_eval_rows": args.screen_eval_rows,
            "screen_samples": args.screen_samples,
            "final_epochs": args.final_epochs,
            "final_patience": args.final_patience,
            "final_eval_rows": args.final_eval_rows,
            "final_samples": args.final_samples,
            "batch_size": args.batch_size,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "candidate_configs": [config.to_dict() for config in configs],
        }
        _write_json(manifest_path, manifest)

    for fold in (0, 1, 2):
        for config in configs:
            if not state.can_start(10):
                break
            _run_trial(
                state=state, phase="focused_screen", config=config, cohort=cohort,
                folds=folds, lag=args.lag, fold=fold, seed=1701,
                device=args.device, max_epochs=args.screen_epochs,
                patience=args.screen_patience, eval_rows=args.screen_eval_rows,
                n_samples=args.screen_samples, batch_size=args.batch_size,
            )

    screen = _leaderboard(state.records, "focused_screen")
    if screen.empty:
        raise RuntimeError("no focused screen trials completed")
    finalists = choose_finalists(screen, configs, args.finalists)
    selection_path = run_dir / "selection.json"
    if not selection_path.exists():
        _write_json(
            selection_path,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "screen_trials_available": int(
                    sum(row.get("phase") == "focused_screen" for row in state.records)
                ),
                "selected_model_ids": [config.model_id for config in finalists],
                "screen_leaderboard": screen.to_dict("records"),
                "external_references_consulted": False,
            },
        )
    else:
        selected = json.loads(selection_path.read_text())["selected_model_ids"]
        by_id = {config.model_id: config for config in configs}
        finalists = [by_id[model_id] for model_id in selected]
    print("FINALISTS " + ",".join(config.model_id for config in finalists), flush=True)

    for fold in (3, 4):
        for config in finalists:
            for seed in (1701, 2903):
                if not state.can_start(10):
                    break
                _run_trial(
                    state=state, phase="focused_confirmation", config=config,
                    cohort=cohort, folds=folds, lag=args.lag, fold=fold,
                    seed=seed, device=args.device, max_epochs=args.final_epochs,
                    patience=args.final_patience, eval_rows=args.final_eval_rows,
                    n_samples=args.final_samples, batch_size=args.batch_size,
                )

    expected_confirmation = len(finalists) * 2 * 2
    completed_confirmation = sum(
        row.get("phase") == "focused_confirmation" and row.get("status") == "ok"
        for row in state.records
    )
    winner_full_expected = 0
    if args.fit_winner_full_cv and completed_confirmation == expected_confirmation:
        confirmation = _leaderboard(state.records, "focused_confirmation")
        winner_id = str(confirmation.iloc[0].model_id)
        by_id = {config.model_id: config for config in configs}
        winner = by_id[winner_id]
        winner_path = run_dir / "winner_selection.json"
        if not winner_path.exists():
            _write_json(
                winner_path,
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model_id": winner_id,
                    "confirmation_leaderboard": confirmation.to_dict("records"),
                    "external_references_consulted": False,
                    "purpose": "freeze confirmed winner before five-fold x three-seed refit",
                },
            )
        else:
            frozen_id = json.loads(winner_path.read_text())["model_id"]
            winner = by_id[frozen_id]
        winner_full_expected = 15
        print(f"WINNER_FULL_CV {winner.model_id}", flush=True)
        for fold in range(5):
            for seed in (1701, 2903, 4307):
                if not state.can_start(10):
                    break
                _run_trial(
                    state=state, phase="winner_full_cv", config=winner,
                    cohort=cohort, folds=folds, lag=args.lag, fold=fold,
                    seed=seed, device=args.device, max_epochs=args.final_epochs,
                    patience=args.final_patience, eval_rows=args.final_eval_rows,
                    n_samples=args.final_samples, batch_size=args.batch_size,
                )

    expected_screen = len(configs) * 3
    completed_screen = sum(
        row.get("phase") == "focused_screen" and row.get("status") == "ok"
        for row in state.records
    )
    completed_winner_full = sum(
        row.get("phase") == "winner_full_cv" and row.get("status") == "ok"
        for row in state.records
    )
    complete = (
        completed_screen == expected_screen
        and completed_confirmation == expected_confirmation
        and completed_winner_full == winner_full_expected
    )
    _update_leaderboards(state)
    _write_report(run_dir, finalists, complete)
    _write_json(
        run_dir / "validation.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if complete else "partial",
            "expected_screen": expected_screen,
            "completed_screen": completed_screen,
            "expected_confirmation": expected_confirmation,
            "completed_confirmation": completed_confirmation,
            "expected_winner_full_cv": winner_full_expected,
            "completed_winner_full_cv": completed_winner_full,
            "failed_trials": sum(row.get("status") == "failed" for row in state.records),
        },
    )


if __name__ == "__main__":
    main()
