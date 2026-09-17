from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.distributional_lag_tournament import (
    cohort_folds,
    load_cohorts,
)
from conditional_neural_benchmark.runner import (
    ModelConfig,
    RunState,
    _neural_trial,
    _split_windows,
    _update_leaderboards,
)


def candidate_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            "tcn_locscale_gaussian", "tcn", "heteroscedastic_gaussian", True,
            width=64, dropout=0.10, weight_decay=1e-3,
        ),
        ModelConfig(
            "tcn_locscale_lowrank4", "tcn", "lowrank_gaussian", True,
            width=64, dropout=0.10, weight_decay=5e-4,
            head_params={"hidden": 64, "layers": 2, "rank": 4},
        ),
        ModelConfig(
            "tcn_locscale_lowrank8", "tcn", "lowrank_gaussian", True,
            width=64, dropout=0.10, weight_decay=5e-4,
            head_params={"hidden": 64, "layers": 2, "rank": 8},
        ),
        ModelConfig(
            "tcn_mdn4", "tcn", "autoregressive_mdn", True,
            width=64, dropout=0.10, weight_decay=1e-3,
            head_params={"hidden": 64, "layers": 2, "components": 4},
        ),
        ModelConfig(
            "tcn_score_dsm", "tcn", "constrained_gaussian_dsm", True,
            width=64, dropout=0.10, weight_decay=1e-3,
            head_params={"sigma_min": 0.03, "sigma_max": 1.5},
        ),
        ModelConfig(
            "tcn_energy_ratio", "tcn", "bounded_energy_ratio", True,
            width=64, dropout=0.10, weight_decay=5e-4,
            head_params={
                "hidden": 64, "layers": 2, "tilt_bound": 1.0,
                "base_nll_weight": 0.5, "oversample": 4,
                "centering_samples": 24,
            },
        ),
        ModelConfig(
            "tcn_gaussian_source_flow", "tcn", "gaussian_source_flow_matching", True,
            width=64, dropout=0.10, weight_decay=5e-4,
            head_params={
                "hidden": 96, "layers": 3, "sample_steps": 16,
                "base_nll_weight": 0.5,
            },
        ),
        ModelConfig(
            "tcn_wide_flow", "tcn", "conditional_flow_matching", True,
            width=128, dropout=0.10, weight_decay=5e-4,
            head_params={"hidden": 128, "layers": 4, "sample_steps": 20},
        ),
    ]


def _family(config: ModelConfig) -> str:
    if config.head in {
        "heteroscedastic_gaussian", "lowrank_gaussian", "autoregressive_mdn",
        "constrained_gaussian_dsm",
    }:
        return "exact_or_score"
    if config.head == "bounded_energy_ratio":
        return "contrastive_energy"
    return "transport_flow"


def _write_json(path: Path, value: dict) -> None:
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(entry) for key, entry in item.items()}
        if isinstance(item, list):
            return [safe(entry) for entry in item]
        if isinstance(item, tuple):
            return [safe(entry) for entry in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _done(state: RunState, phase: str, model_id: str, fold: int, seed: int) -> bool:
    return any(
        row.get("status") == "ok"
        and row.get("phase") == phase
        and row.get("model_id") == model_id
        and int(row.get("fold", -1)) == fold
        and int(row.get("seed", -1)) == seed
        for row in state.records
    )


def _run_trial(
    *, state: RunState, phase: str, config: ModelConfig, cohort, folds: np.ndarray,
    lag: int, fold: int, seed: int, device: str, epochs: int, patience: int,
    eval_rows: int, samples: int, batch_size: int,
) -> None:
    if _done(state, phase, config.model_id, fold, seed):
        return
    scaler, train, validation, test = _split_windows(cohort, folds, fold, lag)
    _neural_trial(
        state=state, phase=phase, config=config, cohort=cohort,
        lag=lag, fold=fold, seed=seed, train=train, validation=validation,
        test=test, scaler=scaler, device=device, max_epochs=epochs,
        patience=patience, eval_rows=eval_rows, n_samples=samples,
        batch_size=batch_size,
    )
    _update_leaderboards(state)


def _rank(records: list[dict], phases: set[str]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame = frame[(frame.status == "ok") & frame.phase.isin(phases)].copy()
    if frame.empty:
        return frame
    result = frame.groupby(["model_id", "encoder", "head", "lag"], as_index=False).agg(
        energy=("energy", "mean"),
        energy_se=("energy", "sem"),
        balanced_energy=("energy__stim_balanced", "mean"),
        variogram=("variogram", "mean"),
        coverage90=("coverage90", "mean"),
        rmse=("rmse", "mean"),
        nll_per_neuron=("nll_per_neuron", "mean"),
        trials=("energy", "size"),
        wall_seconds=("wall_seconds", "sum"),
    )
    return result.sort_values(
        ["energy", "balanced_energy", "variogram"], ascending=True
    ).reset_index(drop=True)


def _unique(values: list[ModelConfig]) -> list[ModelConfig]:
    result: list[ModelConfig] = []
    seen: set[str] = set()
    for value in values:
        if value.model_id not in seen:
            result.append(value)
            seen.add(value.model_id)
    return result


def _screen2_candidates(board: pd.DataFrame, configs: dict[str, ModelConfig]) -> list[ModelConfig]:
    ranked = [configs[value] for value in board.model_id if value in configs]
    chosen = ranked[:1]
    for family in ("exact_or_score", "contrastive_energy", "transport_flow"):
        candidate = next((value for value in ranked if _family(value) == family), None)
        if candidate is not None:
            chosen.append(candidate)
    parametric = next(
        (
            value for value in ranked
            if value.head in {
                "heteroscedastic_gaussian", "lowrank_gaussian",
                "constrained_gaussian_dsm",
            }
        ),
        None,
    )
    if parametric is not None:
        chosen.append(parametric)
    for value in ranked:
        chosen.append(value)
    return _unique(chosen)[:4]


def _finalists(board: pd.DataFrame, configs: dict[str, ModelConfig]) -> list[ModelConfig]:
    ranked = [configs[value] for value in board.model_id if value in configs]
    chosen = ranked[:1]
    exact = next((value for value in ranked if _family(value) == "exact_or_score"), None)
    if exact is not None:
        chosen.append(exact)
    parametric = next(
        (
            value for value in ranked
            if value.head in {
                "heteroscedastic_gaussian", "lowrank_gaussian",
                "constrained_gaussian_dsm",
            }
        ),
        None,
    )
    if parametric is not None:
        chosen.append(parametric)
    for value in ranked:
        chosen.append(value)
    return _unique(chosen)[:3]


def run_one(
    root: Path,
    cohort_name: str,
    cohort,
    folds: np.ndarray,
    lag: int,
    *,
    hours: float,
    device: str,
    resume: bool,
) -> dict:
    run_dir = root / cohort_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(run_dir, time.monotonic() + hours * 3600)
    metrics = run_dir / "trial_metrics.csv"
    if resume and metrics.exists():
        state.records = pd.read_csv(metrics).replace({np.nan: None}).to_dict("records")
    configs_list = candidate_configs()
    configs = {value.model_id: value for value in configs_list}
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        _write_json(manifest, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cohort": cohort_name,
            "worms": cohort.n_worms,
            "neurons": cohort.n_neurons,
            "lag": lag,
            "screen_fold0_seed": 1701,
            "screen_folds12_seed": 1701,
            "confirmation_seed": 1701,
            "winner_sensitivity_seed": 2903,
            "selection": "energy, then balanced energy, then variogram; preserve exact/score family",
            "candidate_configs": [value.to_dict() for value in configs_list],
            "atlas_firewall": "no Randi, Cook, Bentley, receptor, SBTG result, or external atlas",
        })
    for config in configs_list:
        if state.can_start(8):
            _run_trial(
                state=state, phase="screen_fold0", config=config, cohort=cohort,
                folds=folds, lag=lag, fold=0, seed=1701, device=device,
                epochs=18, patience=4, eval_rows=2400, samples=12, batch_size=256,
            )
    first = _rank(state.records, {"screen_fold0"})
    if first.empty:
        raise RuntimeError(f"no neural screen completed for {cohort_name}")
    screen2 = _screen2_candidates(first, configs)
    screen_selection = run_dir / "screen_selection.json"
    if not screen_selection.exists():
        _write_json(screen_selection, {
            "selected_model_ids": [value.model_id for value in screen2],
            "fold0_leaderboard": first.to_dict("records"),
            "external_references_consulted": False,
        })
    else:
        frozen = json.loads(screen_selection.read_text())["selected_model_ids"]
        screen2 = [configs[value] for value in frozen]
    for config in screen2:
        for fold in (1, 2):
            if state.can_start(8):
                _run_trial(
                    state=state, phase="screen_folds12", config=config, cohort=cohort,
                    folds=folds, lag=lag, fold=fold, seed=1701, device=device,
                    epochs=24, patience=5, eval_rows=3000, samples=16, batch_size=256,
                )
    screen = _rank(state.records, {"screen_fold0", "screen_folds12"})
    finalists = _finalists(screen, configs)
    finalist_path = run_dir / "finalist_selection.json"
    if not finalist_path.exists():
        _write_json(finalist_path, {
            "selected_model_ids": [value.model_id for value in finalists],
            "screen_leaderboard": screen.to_dict("records"),
            "external_references_consulted": False,
        })
    else:
        frozen = json.loads(finalist_path.read_text())["selected_model_ids"]
        finalists = [configs[value] for value in frozen]
    for config in finalists:
        for fold in (3, 4):
            if state.can_start(10):
                _run_trial(
                    state=state, phase="confirmation", config=config, cohort=cohort,
                    folds=folds, lag=lag, fold=fold, seed=1701, device=device,
                    epochs=32, patience=7, eval_rows=4000, samples=24, batch_size=256,
                )
    confirmation = _rank(state.records, {"confirmation"})
    if confirmation.empty:
        raise RuntimeError(f"no neural confirmation completed for {cohort_name}")
    winner_path = run_dir / "winner_selection.json"
    if not winner_path.exists():
        winner_id = str(confirmation.iloc[0].model_id)
        _write_json(winner_path, {
            "model_id": winner_id,
            "confirmation_leaderboard_before_seed_sensitivity": confirmation.to_dict("records"),
            "external_references_consulted": False,
        })
    else:
        winner_id = json.loads(winner_path.read_text())["model_id"]
    winner = configs[winner_id]
    for fold in (3, 4):
        if state.can_start(10):
            _run_trial(
                state=state, phase="winner_seed_sensitivity", config=winner,
                cohort=cohort, folds=folds, lag=lag, fold=fold, seed=2903,
                device=device, epochs=32, patience=7, eval_rows=4000,
                samples=24, batch_size=256,
            )
    final_board = _rank(state.records, {"confirmation", "winner_seed_sensitivity"})
    first.to_csv(run_dir / "fold0_leaderboard.csv", index=False)
    screen.to_csv(run_dir / "screen_leaderboard.csv", index=False)
    confirmation.to_csv(run_dir / "confirmation_leaderboard.csv", index=False)
    final_board.to_csv(run_dir / "final_leaderboard.csv", index=False)
    failed = sum(value.get("status") == "failed" for value in state.records)
    result = {
        "cohort": cohort_name, "lag": lag, "winner": winner_id,
        "completed_trials": sum(value.get("status") == "ok" for value in state.records),
        "failed_trials": failed,
        "complete_confirmation": bool(
            sum(
                value.get("status") == "ok" and value.get("phase") == "confirmation"
                for value in state.records
            ) == len(finalists) * 2
        ),
        "external_references_consulted": False,
    }
    _write_json(run_dir / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/higher_order_neural_20260828"))
    parser.add_argument("--structured", type=Path, default=Path("results/higher_order_lag_20260828"))
    parser.add_argument("--evidence", type=Path, default=Path("results/distributed_lag_dynamics_20260828"))
    parser.add_argument("--hours", type=float, default=6.5)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    structured = args.structured.resolve()
    deadline = time.monotonic() + args.hours * 3600
    required_cohorts = {"current54", "sbtg80", "sbtg_bridge54"}
    selections = None
    while selections is None:
        if time.monotonic() > deadline:
            raise TimeoutError("structured selection did not become available")
        if (structured / "selection.json").exists():
            candidate = json.loads((structured / "selection.json").read_text())
            if required_cohorts.issubset(candidate):
                selections = candidate
                break
        time.sleep(10)
    assert selections is not None
    cohorts = load_cohorts()
    results = []
    for index, (name, cohort) in enumerate(cohorts.items()):
        remaining = max(0.3, (deadline - time.monotonic()) / 3600)
        cohorts_left = len(cohorts) - index
        budget = remaining / cohorts_left
        folds = cohort_folds(name, cohort, args.evidence.resolve())
        lag = int(selections[name]["mean"]["lag"])
        results.append(run_one(
            output, name, cohort, folds, lag,
            hours=budget, device=args.device, resume=args.resume,
        ))
    _write_json(output / "validation.json", {
        "status": "pass" if all(value["complete_confirmation"] for value in results) else "partial",
        "cohorts": results,
    })
    print(f"HIGHER_ORDER_NEURAL_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
