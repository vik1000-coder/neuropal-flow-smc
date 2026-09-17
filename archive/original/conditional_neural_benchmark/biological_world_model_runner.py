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
from conditional_neural_benchmark.focused_world_model_runner import (
    _is_done,
    _load_folds,
    _run_trial,
)
from conditional_neural_benchmark.runner import (
    ModelConfig,
    RunState,
    _leaderboard,
    _update_leaderboards,
)


def candidate_configs() -> list[ModelConfig]:
    """Mechanism-focused grid for full-pulse history and rare-regime learning."""
    flow = {"hidden": 128, "layers": 4, "sample_steps": 24}
    compact_flow = {"hidden": 96, "layers": 3, "sample_steps": 20}
    return [
        ModelConfig(
            "control_tcn_flow128_natural", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
        ),
        ModelConfig(
            "full_tcn_flow128_natural", "full_history_tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
        ),
        ModelConfig(
            "full_tcn_flow128_balanced", "full_history_tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
            training_scheme="stratum_balanced",
        ),
        ModelConfig(
            "phase_tcn_flow128_natural", "stimulus_phase_tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
        ),
        ModelConfig(
            "phase_tcn_flow128_balanced", "stimulus_phase_tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
            training_scheme="stratum_balanced",
        ),
        ModelConfig(
            "multiscale_tcn_flow128_balanced", "multiscale_tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
            training_scheme="stratum_balanced",
        ),
        ModelConfig(
            "phase_tcn_flow96_balanced_wd1e3", "stimulus_phase_tcn", "conditional_flow_matching", True,
            width=96, dropout=0.15, weight_decay=1e-3, head_params=compact_flow,
            training_scheme="stratum_balanced",
        ),
        ModelConfig(
            "gru_flow96_balanced", "gru", "conditional_flow_matching", True,
            width=96, dropout=0.10, weight_decay=5e-4, head_params=compact_flow,
            training_scheme="stratum_balanced",
        ),
        ModelConfig(
            "transformer_flow96_balanced", "transformer", "conditional_flow_matching", True,
            width=96, dropout=0.10, weight_decay=5e-4, head_params=compact_flow,
            training_scheme="stratum_balanced",
        ),
        ModelConfig(
            "phase_gaussian_source96_balanced", "stimulus_phase_tcn", "gaussian_source_flow_matching", True,
            width=96, dropout=0.10, weight_decay=5e-4,
            head_params={**compact_flow, "base_nll_weight": 0.25},
            training_scheme="stratum_balanced",
        ),
        ModelConfig(
            "phase_mdn4_64_balanced", "stimulus_phase_tcn", "autoregressive_mdn", True,
            width=64, dropout=0.10, weight_decay=1e-3,
            head_params={"hidden": 64, "layers": 2, "components": 4},
            training_scheme="stratum_balanced",
        ),
        ModelConfig(
            "phase_lowrank16_96_balanced", "stimulus_phase_tcn", "lowrank_gaussian", True,
            width=96, dropout=0.10, weight_decay=5e-4,
            head_params={"hidden": 96, "layers": 2, "rank": 16},
            training_scheme="stratum_balanced",
        ),
    ]


def stability_candidate_configs() -> list[ModelConfig]:
    """Focused grid for smoother off-manifold rollout and perturbation response."""
    flow = {"hidden": 128, "layers": 4, "sample_steps": 24}
    configs = [
        ModelConfig(
            "control_tcn_flow128_natural", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
        ),
    ]
    for noise in (0.005, 0.01, 0.02, 0.03, 0.05):
        label = str(noise).replace("0.", "p")
        configs.append(ModelConfig(
            f"tcn_flow128_jitter_{label}", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
            history_noise_std=noise, history_noise_copies=1,
        ))
    configs.extend([
        ModelConfig(
            "tcn_flow128_jitter_p02_x2", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
            history_noise_std=0.02, history_noise_copies=2,
        ),
        ModelConfig(
            "tcn_flow128_transition40", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
            training_scheme="transition_moderate",
        ),
        ModelConfig(
            "tcn_flow128_transition40_jitter_p01", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
            training_scheme="transition_moderate",
            history_noise_std=0.01, history_noise_copies=1,
        ),
        ModelConfig(
            "tcn_flow128_transition40_jitter_p02", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4, head_params=flow,
            training_scheme="transition_moderate",
            history_noise_std=0.02, history_noise_copies=1,
        ),
        ModelConfig(
            "tcn_flow128_dropout05_jitter_p01", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.05, weight_decay=5e-4, head_params=flow,
            history_noise_std=0.01, history_noise_copies=1,
        ),
        ModelConfig(
            "tcn_flow128_dropout15_jitter_p01", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.15, weight_decay=7.5e-4, head_params=flow,
            history_noise_std=0.01, history_noise_copies=1,
        ),
        ModelConfig(
            "tcn_mdn4_jitter_p01", "tcn", "autoregressive_mdn", True,
            width=64, dropout=0.10, weight_decay=1e-3,
            head_params={"hidden": 64, "layers": 2, "components": 4},
            history_noise_std=0.01, history_noise_copies=1,
        ),
    ])
    return configs


def _family(config: ModelConfig) -> str:
    if config.head in {"autoregressive_mdn", "lowrank_gaussian"}:
        return "exact_density"
    if config.encoder == "stimulus_phase_tcn":
        return "phase_aware_flow"
    if config.encoder in {"full_history_tcn", "multiscale_tcn"}:
        return "full_history_flow"
    return "control_or_recurrent"


def choose_finalists(
    board: pd.DataFrame, configs: list[ModelConfig], count: int
) -> list[ModelConfig]:
    by_id = {config.model_id: config for config in configs}
    ranked = [by_id[str(model_id)] for model_id in board.model_id if str(model_id) in by_id]
    selected = ranked[: max(1, count - 1)]
    if not any(_family(config) == "exact_density" for config in selected):
        exact = next((config for config in ranked if _family(config) == "exact_density"), None)
        if exact is not None:
            selected.append(exact)
    for config in ranked:
        if len(selected) >= count:
            break
        if config not in selected:
            selected.append(config)
    return selected[:count]


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(run_dir: Path, status: str) -> None:
    metrics = pd.read_csv(run_dir / "trial_metrics.csv")
    lines = [
        "# Full-history biological world-model tournament",
        "",
        f"Status: **{status}**",
        "",
        "This run tests the specific hypothesis that the legacy L=80 TCN lost stimulus-phase information because its 31-frame receptive field covered only 7.75 of the declared 20 seconds. The new full-history encoders cover 127 frames and are crossed with natural versus equal-stratum training.",
        "",
        "External connectomes and SBTG response matrices are excluded from fitting, early stopping, and predictive model selection.",
        "",
    ]
    for phase in ("biological_screen", "biological_confirmation", "winner_full_cv"):
        board = _leaderboard(metrics.replace({np.nan: None}).to_dict("records"), phase)
        if board.empty:
            continue
        lines.extend([
            f"## {phase.replace('_', ' ').title()}", "",
            "| Rank | Model | Energy | Balanced energy | Variogram | Trials |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ])
        for row in board.itertuples():
            lines.append(
                f"| {int(row.rank)} | `{row.model_id}` | {row.energy__mean:.6f} | "
                f"{row.energy__stim_balanced__mean:.6f} | {row.variogram__mean:.6f} | "
                f"{int(row.n_trials)} |"
            )
        lines.append("")
    lines.extend([
        "## Claim boundary", "",
        "Energy and variogram scores evaluate held-out predictive activity distributions. They do not make a lag matrix causal or anatomical. Lag-response sampling, quiet cuts, pseudo-boundaries, and biological sign checks are separate downstream analyses.", "",
    ])
    (run_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-run", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=3.5)
    parser.add_argument("--lag", type=int, default=80)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--screen-epochs", type=int, default=30)
    parser.add_argument("--screen-patience", type=int, default=6)
    parser.add_argument("--screen-eval-rows", type=int, default=3200)
    parser.add_argument("--screen-samples", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=45)
    parser.add_argument("--final-patience", type=int, default=9)
    parser.add_argument("--final-eval-rows", type=int, default=4000)
    parser.add_argument("--final-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument("--winner-seeds", nargs="+", type=int, default=[1701, 2903])
    parser.add_argument("--profile", choices=("history", "stability"), default="history")
    parser.add_argument("--stop-after-confirmation", action="store_true")
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
    evidence = args.evidence_run.resolve()
    folds = _load_folds(cohort, evidence)
    configs = (
        candidate_configs() if args.profile == "history" else stability_candidate_configs()
    )
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        _write_json(manifest_path, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "profile": args.profile,
            "protocol": (
                "full-history and stimulus-regime conditional-density tournament"
                if args.profile == "history"
                else "history-jitter and moderate-transition rollout-stability tournament"
            ),
            "hypothesis": (
                "legacy L80 TCN receptive field was 31 frames; full-history variants cover 127 frames"
                if args.profile == "history"
                else "small neural-history jitter smooths off-manifold autoregressive rollout and perturbation response"
            ),
            "screen_folds": [0, 1, 2], "confirmation_folds": [3, 4],
            "screen_seed": 1701, "confirmation_seeds": [1701, 2903],
            "winner_full_cv_seeds": args.winner_seeds,
            "selection_rule": "natural energy one-SE set, then balanced energy and variogram; preserve one exact-density finalist",
            "atlas_firewall": "no Randi, Cook, Bentley, SBTG, or external atlas in fitting/stopping/selection",
            "lag_frames": args.lag, "fps": cohort.fps,
            "legacy_receptive_field_frames": 31, "new_receptive_field_frames": 127,
            "n_worms": cohort.n_worms, "n_neurons": cohort.n_neurons,
            "neurons": list(cohort.neurons), "device": args.device,
            "torch": torch.__version__, "python": platform.python_version(),
            "batch_size": args.batch_size, "screen_epochs": args.screen_epochs,
            "final_epochs": args.final_epochs,
            "fold_assignments_sha256": _sha256(evidence / "fold_assignments.csv"),
            "candidate_configs": [config.to_dict() for config in configs],
        })

    for fold in (0, 1, 2):
        for config in configs:
            if not state.can_start(12):
                break
            _run_trial(
                state=state, phase="biological_screen", config=config, cohort=cohort,
                folds=folds, lag=args.lag, fold=fold, seed=1701,
                device=args.device, max_epochs=args.screen_epochs,
                patience=args.screen_patience, eval_rows=args.screen_eval_rows,
                n_samples=args.screen_samples, batch_size=args.batch_size,
            )

    screen = _leaderboard(state.records, "biological_screen")
    if screen.empty:
        raise RuntimeError("no screen trials completed")
    selection_path = run_dir / "selection.json"
    if not selection_path.exists():
        finalists = choose_finalists(screen, configs, args.finalists)
        _write_json(selection_path, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selected_model_ids": [config.model_id for config in finalists],
            "screen_leaderboard": screen.to_dict("records"),
            "external_references_consulted": False,
        })
    selected_ids = json.loads(selection_path.read_text())["selected_model_ids"]
    lookup = {config.model_id: config for config in configs}
    finalists = [lookup[model_id] for model_id in selected_ids]
    print("FINALISTS " + ",".join(selected_ids), flush=True)

    for fold in (3, 4):
        for config in finalists:
            for seed in (1701, 2903):
                if not state.can_start(12):
                    break
                _run_trial(
                    state=state, phase="biological_confirmation", config=config,
                    cohort=cohort, folds=folds, lag=args.lag, fold=fold, seed=seed,
                    device=args.device, max_epochs=args.final_epochs,
                    patience=args.final_patience, eval_rows=args.final_eval_rows,
                    n_samples=args.final_samples, batch_size=args.batch_size,
                )

    expected_confirmation = 4 * len(finalists)
    completed_confirmation = sum(
        row.get("phase") == "biological_confirmation" and row.get("status") == "ok"
        for row in state.records
    )
    if completed_confirmation == expected_confirmation and not args.stop_after_confirmation:
        confirmation = _leaderboard(state.records, "biological_confirmation")
        winner_path = run_dir / "winner_selection.json"
        if not winner_path.exists():
            _write_json(winner_path, {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "model_id": str(confirmation.iloc[0].model_id),
                "confirmation_leaderboard": confirmation.to_dict("records"),
                "external_references_consulted": False,
            })
        winner_id = json.loads(winner_path.read_text())["model_id"]
        winner = lookup[winner_id]
        print(f"WINNER_FULL_CV {winner_id}", flush=True)
        for fold in range(5):
            for seed in args.winner_seeds:
                if not state.can_start(12):
                    break
                _run_trial(
                    state=state, phase="winner_full_cv", config=winner, cohort=cohort,
                    folds=folds, lag=args.lag, fold=fold, seed=seed,
                    device=args.device, max_epochs=args.final_epochs,
                    patience=args.final_patience, eval_rows=args.final_eval_rows,
                    n_samples=args.final_samples, batch_size=args.batch_size,
                )

    expected_screen = 3 * len(configs)
    completed_screen = sum(
        row.get("phase") == "biological_screen" and row.get("status") == "ok"
        for row in state.records
    )
    expected_full = (
        5 * len(args.winner_seeds)
        if completed_confirmation == expected_confirmation and not args.stop_after_confirmation
        else 0
    )
    completed_full = sum(
        row.get("phase") == "winner_full_cv" and row.get("status") == "ok"
        for row in state.records
    )
    complete = (
        completed_screen == expected_screen
        and completed_confirmation == expected_confirmation
        and completed_full == expected_full
    )
    _update_leaderboards(state)
    status = "complete" if complete else "time-bounded partial"
    _write_report(run_dir, status)
    _write_json(run_dir / "validation.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "partial",
        "expected_screen": expected_screen, "completed_screen": completed_screen,
        "expected_confirmation": expected_confirmation,
        "completed_confirmation": completed_confirmation,
        "expected_winner_full_cv": expected_full,
        "completed_winner_full_cv": completed_full,
        "failed_trials": sum(row.get("status") == "failed" for row in state.records),
    })


if __name__ == "__main__":
    main()
