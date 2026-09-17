"""Reference-firewalled optimization of the historical SBTG80 flow and lag-1 sampler.

The historical 80-neuron cohort is retained only for comparability with the
released SBTG result.  It contains the known pseudo-pairing and donor-imputation
lineage and is not a simultaneous 80-neuron recording.  Generator candidates
are selected using held-out predictive scores before any external reference is
opened.  External Randi/Cook comparisons are a separate, explicitly exploratory
stage after candidate matrices have been frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from compatibility_neural_benchmark.prediction_atlas_analysis import (
    _contextualize_support,
    _contextualize_values,
    effect_normalization_denominator,
    orient_response_once,
)
from compatibility_neural_benchmark.prediction_atlas_external_analysis import (
    _reference_rows,
)
from compatibility_neural_benchmark.prediction_atlas_runner import run_one, sha256
from compatibility_neural_benchmark.postfreeze_external_analysis import load_references
from compatibility_neural_benchmark.sbtg80_progressive_sensitivity import (
    _load_published_slices,
    _paired_source_bootstrap,
)
from conditional_neural_benchmark.data import load_sbtg_cohort, make_fold_assignments
from conditional_neural_benchmark.runner import (
    ModelConfig,
    RunState,
    _neural_trial,
    _split_windows,
)


FOLD_SEED = 20_260_828
TRAIN_SEED = 1701
BASE_SEED = 20_260_904
SOURCE_LAG = 1
HORIZONS = (1,)
SOURCE_WINDOW_FRAMES = 4
DEFAULT_ROOT = Path("results/sbtg80_flow_optimization_20260901")
DEFAULT_REFERENCE_RELEASE = Path("/Users/vik/Downloads/SBTG-public-release copy")
DEFAULT_PUBLISHED_ARCHIVE = (
    DEFAULT_REFERENCE_RELEASE / "results/paper/sbtg_lag_matrices.npz"
)
def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def candidate_configs() -> list[ModelConfig]:
    """Prespecified flow-only screen spanning optimization and architecture."""
    common = dict(
        encoder="tcn",
        head="conditional_flow_matching",
        residual_target=True,
        width=128,
        dropout=0.10,
        weight_decay=5e-4,
        learning_rate=3e-4,
        head_params={"hidden": 128, "layers": 4, "sample_steps": 20},
    )
    return [
        ModelConfig("flow_long_base", **common),
        ModelConfig(
            "flow_lr1e4", **{**common, "learning_rate": 1e-4, "weight_decay": 1e-4}
        ),
        ModelConfig(
            "flow_lr6e4", **{**common, "learning_rate": 6e-4, "weight_decay": 1e-4}
        ),
        ModelConfig(
            "flow_dropout0", **{**common, "dropout": 0.0, "weight_decay": 1e-4}
        ),
        ModelConfig(
            "flow_w192",
            **{
                **common,
                "width": 192,
                "dropout": 0.05,
                "weight_decay": 1e-4,
                "head_params": {"hidden": 192, "layers": 4, "sample_steps": 24},
            },
        ),
        ModelConfig(
            "flow_h256_l5",
            **{
                **common,
                "dropout": 0.05,
                "weight_decay": 1e-4,
                "head_params": {"hidden": 256, "layers": 5, "sample_steps": 24},
            },
        ),
        ModelConfig(
            "flow_transition",
            **{**common, "training_scheme": "transition_moderate"},
        ),
        ModelConfig(
            "flow_balanced",
            **{**common, "training_scheme": "stratum_balanced"},
        ),
        ModelConfig(
            "flow_noise02",
            **{**common, "history_noise_std": 0.02, "history_noise_copies": 1},
        ),
        ModelConfig(
            "flow_multiscale",
            **{
                **common,
                "encoder": "multiscale_tcn",
                "dropout": 0.05,
                "weight_decay": 1e-4,
                "head_params": {"hidden": 192, "layers": 4, "sample_steps": 24},
            },
        ),
        ModelConfig(
            "flow_gru",
            **{
                **common,
                "encoder": "gru",
                "dropout": 0.05,
                "weight_decay": 1e-4,
                "head_params": {"hidden": 192, "layers": 4, "sample_steps": 24},
            },
        ),
        ModelConfig(
            "gaussian_source_long",
            encoder="tcn",
            head="gaussian_source_flow_matching",
            residual_target=True,
            width=128,
            dropout=0.05,
            weight_decay=1e-4,
            learning_rate=3e-4,
            head_params={
                "hidden": 192,
                "layers": 4,
                "sample_steps": 24,
                "base_nll_weight": 0.25,
            },
        ),
        ModelConfig(
            "gaussian_source_transition",
            encoder="tcn",
            head="gaussian_source_flow_matching",
            residual_target=True,
            width=128,
            dropout=0.05,
            weight_decay=1e-4,
            learning_rate=3e-4,
            training_scheme="transition_moderate",
            head_params={
                "hidden": 192,
                "layers": 4,
                "sample_steps": 24,
                "base_nll_weight": 0.25,
            },
        ),
    ]


def _checkpoint_phase(fold: int) -> str:
    return "screen_fold0" if fold == 0 else "screen_folds12" if fold < 3 else "confirmation"


def _checkpoint_path(
    train_dir: Path,
    config: ModelConfig,
    fold: int,
    train_seed: int = TRAIN_SEED,
) -> Path:
    return (
        train_dir
        / "checkpoints"
        / _checkpoint_phase(fold)
        / f"{config.model_id}__L8__f{fold}__s{train_seed}.pt"
    )


def _done(
    records: list[dict[str, Any]],
    phase: str,
    model_id: str,
    fold: int,
    train_seed: int,
) -> bool:
    return any(
        row.get("status") == "ok"
        and row.get("phase") == phase
        and row.get("model_id") == model_id
        and int(row.get("fold", -1)) == fold
        and int(row.get("seed", -1)) == train_seed
        for row in records
    )


def _run_training_trial(
    state: RunState,
    config: ModelConfig,
    cohort,
    folds: np.ndarray,
    fold: int,
    *,
    device: str,
    train_seed: int = TRAIN_SEED,
) -> None:
    phase = _checkpoint_phase(fold)
    if _done(state.records, phase, config.model_id, fold, train_seed):
        return
    scaler, train, validation, test = _split_windows(cohort, folds, fold, 8)
    _neural_trial(
        state=state,
        phase=phase,
        config=config,
        cohort=cohort,
        lag=8,
        fold=fold,
        seed=train_seed,
        train=train,
        validation=validation,
        test=test,
        scaler=scaler,
        device=device,
        max_epochs=160 if config.learning_rate <= 3e-4 else 120,
        patience=20,
        eval_rows=5000,
        n_samples=32,
        batch_size=256,
        trial_metadata={
            "optimization_run": "historical_sbtg80_flow_v2",
            "external_references_consulted": False,
            "cohort_mode": cohort.cohort_mode,
            "stimulus_encoding": "binary_any_stimulus",
        },
    )


def _selection_board(records: list[dict[str, Any]], folds: Iterable[int]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    fold_set = set(int(value) for value in folds)
    frame = frame[
        (frame.status == "ok")
        & frame.fold.astype(int).isin(fold_set)
    ].copy()
    if frame.empty:
        return frame
    result = frame.groupby(
        ["model_id", "encoder", "head", "training_scheme"], as_index=False
    ).agg(
        energy=("energy", "mean"),
        balanced_energy=("energy__stim_balanced", "mean"),
        variogram=("variogram", "mean"),
        rmse=("rmse", "mean"),
        coverage90=("coverage90", "mean"),
        best_epoch=("best_epoch", "mean"),
        stopped_epoch=("stopped_epoch", "mean"),
        trials=("energy", "size"),
        wall_seconds=("wall_seconds", "sum"),
    )
    result["calibration_error90"] = np.abs(result.coverage90 - 0.90)
    return result.sort_values(
        ["balanced_energy", "energy", "variogram", "calibration_error90"]
    ).reset_index(drop=True)


def _shortlist(board: pd.DataFrame, configs: dict[str, ModelConfig], n: int) -> list[str]:
    chosen = list(board.model_id.astype(str).head(max(1, n - 2)))
    # Preserve one Gaussian-source and one non-legacy encoder if their runs succeeded.
    for predicate in (
        lambda row: row["head"] == "gaussian_source_flow_matching",
        lambda row: row["encoder"] in {"multiscale_tcn", "gru"},
    ):
        match = next((str(row.model_id) for _, row in board.iterrows() if predicate(row)), None)
        if match is not None:
            chosen.append(match)
    result: list[str] = []
    for model_id in chosen:
        if model_id in configs and model_id not in result:
            result.append(model_id)
    for model_id in board.model_id.astype(str):
        if len(result) >= n:
            break
        if model_id in configs and model_id not in result:
            result.append(model_id)
    return result[:n]


def train(root: Path, *, device: str, hours: float, resume: bool) -> None:
    train_dir = root / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(train_dir, time.monotonic() + hours * 3600)
    metrics = train_dir / "trial_metrics.csv"
    if resume and metrics.exists():
        state.records = pd.read_csv(metrics).replace({np.nan: None}).to_dict("records")
    cohort = load_sbtg_cohort()
    folds = make_fold_assignments(cohort, n_folds=5, seed=FOLD_SEED)
    configs_list = candidate_configs()
    configs = {config.model_id: config for config in configs_list}
    manifest = {
        "created_utc": _utc(),
        "stage": "generator_training",
        "cohort_mode": cohort.cohort_mode,
        "lineage_warning": cohort.lineage_warning,
        "worms": cohort.n_worms,
        "neurons": cohort.n_neurons,
        "history_frames": 8,
        "fold_seed": FOLD_SEED,
        "train_seed": TRAIN_SEED,
        "candidate_configs": [config.to_dict() for config in configs_list],
        "selection": "held-out balanced energy, then energy, variogram, and calibration",
        "external_references_consulted": False,
    }
    if not (train_dir / "manifest.json").exists():
        _atomic_json(train_dir / "manifest.json", manifest)

    for config in configs_list:
        if state.can_start(4):
            _run_training_trial(state, config, cohort, folds, 0, device=device)
    fold0 = _selection_board(state.records, (0,))
    fold0.to_csv(train_dir / "fold0_leaderboard.csv", index=False)
    screen_ids = _shortlist(fold0, configs, 7)
    screen_path = train_dir / "screen_selection.json"
    if screen_path.exists():
        screen_ids = json.loads(screen_path.read_text())["selected_model_ids"]
    else:
        _atomic_json(
            screen_path,
            {
                "created_utc": _utc(),
                "selected_model_ids": screen_ids,
                "external_references_consulted": False,
            },
        )
    for model_id in screen_ids:
        for fold in (1, 2):
            if state.can_start(4):
                _run_training_trial(state, configs[model_id], cohort, folds, fold, device=device)
    screen = _selection_board(state.records, (0, 1, 2))
    screen.to_csv(train_dir / "screen_leaderboard.csv", index=False)
    finalist_ids = _shortlist(screen[screen.trials >= 3], configs, 4)
    finalist_path = train_dir / "finalist_selection.json"
    if finalist_path.exists():
        finalist_ids = json.loads(finalist_path.read_text())["selected_model_ids"]
    else:
        _atomic_json(
            finalist_path,
            {
                "created_utc": _utc(),
                "selected_model_ids": finalist_ids,
                "external_references_consulted": False,
            },
        )
    for model_id in finalist_ids:
        for fold in (3, 4):
            if state.can_start(4):
                _run_training_trial(state, configs[model_id], cohort, folds, fold, device=device)
    final = _selection_board(state.records, (0, 1, 2, 3, 4))
    final.to_csv(train_dir / "final_leaderboard.csv", index=False)
    complete = final[final.trials >= 5].copy()
    if complete.empty:
        raise RuntimeError("no candidate completed all five folds")
    winner = str(complete.iloc[0].model_id)
    _atomic_json(
        train_dir / "winner_selection.json",
        {
            "created_utc": _utc(),
            "model_id": winner,
            "complete_five_fold_leaderboard": complete.to_dict("records"),
            "external_references_consulted": False,
        },
    )
    _atomic_json(
        train_dir / "validation.json",
        {
            "created_utc": _utc(),
            "status": "passed",
            "completed_trials": int(sum(row.get("status") == "ok" for row in state.records)),
            "failed_trials": int(sum(row.get("status") == "failed" for row in state.records)),
            "complete_five_fold_candidates": complete.model_id.astype(str).tolist(),
            "winner": winner,
            "external_references_consulted": False,
        },
    )


def _candidate_lookup() -> dict[str, ModelConfig]:
    result = {config.model_id: config for config in candidate_configs()}
    base = result["flow_lr6e4"]
    result["flow_lr6e4_s2903"] = ModelConfig(
        **{**base.to_dict(), "model_id": "flow_lr6e4_s2903"}
    )
    return result


def train_secondary_seed(
    root: Path,
    *,
    train_seed: int,
    device: str,
    hours: float,
    resume: bool,
) -> None:
    if train_seed == TRAIN_SEED:
        raise ValueError("secondary seed must differ from the primary training seed")
    train_dir = root / f"training_seed{train_seed}"
    train_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(train_dir, time.monotonic() + hours * 3600)
    metrics = train_dir / "trial_metrics.csv"
    if resume and metrics.exists():
        state.records = pd.read_csv(metrics).replace({np.nan: None}).to_dict("records")
    config = _candidate_lookup()["flow_lr6e4_s2903"]
    if train_seed != 2903:
        config = ModelConfig(
            **{**config.to_dict(), "model_id": f"flow_lr6e4_s{train_seed}"}
        )
    cohort = load_sbtg_cohort()
    folds = make_fold_assignments(cohort, n_folds=5, seed=FOLD_SEED)
    _atomic_json(
        train_dir / "manifest.json",
        {
            "created_utc": _utc(),
            "stage": "independent_generator_seed_confirmation",
            "model_config": config.to_dict(),
            "train_seed": train_seed,
            "fold_seed": FOLD_SEED,
            "cohort_mode": cohort.cohort_mode,
            "lineage_warning": cohort.lineage_warning,
            "external_references_consulted": False,
        },
    )
    for fold in range(5):
        if state.can_start(4):
            _run_training_trial(
                state,
                config,
                cohort,
                folds,
                fold,
                device=device,
                train_seed=train_seed,
            )
    frame = pd.DataFrame(state.records)
    ok = frame[(frame.status == "ok") & (frame.seed.astype(int) == train_seed)]
    if len(ok) != 5 or set(ok.fold.astype(int)) != set(range(5)):
        raise RuntimeError("secondary generator seed did not complete all five folds")
    _atomic_json(
        train_dir / "validation.json",
        {
            "created_utc": _utc(),
            "status": "passed",
            "model_id": config.model_id,
            "train_seed": train_seed,
            "folds": sorted(ok.fold.astype(int).tolist()),
            "mean_energy": float(ok.energy.mean()),
            "mean_balanced_energy": float(ok["energy__stim_balanced"].mean()),
            "mean_variogram": float(ok.variogram.mean()),
            "external_references_consulted": False,
        },
    )


def _sampler_id(
    model_id: str,
    particles: int,
    branch: int,
    future: int,
    integration_steps: int | None,
    train_seed: int,
) -> str:
    suffix = "" if integration_steps is None else f"__steps{integration_steps}"
    if train_seed != TRAIN_SEED:
        suffix += f"__seed{train_seed}"
    return f"{model_id}__N{particles}__b{branch}__f{future}{suffix}"


def _inference_checkpoint(
    train_dir: Path,
    config: ModelConfig,
    fold: int,
    integration_steps: int | None,
    train_seed: int,
) -> Path:
    source = _checkpoint_path(train_dir, config, fold, train_seed)
    if integration_steps is None or int(config.head_params["sample_steps"]) == integration_steps:
        return source
    if integration_steps < 4:
        raise ValueError("integration steps must be at least four")
    target = (
        train_dir
        / "inference_checkpoints"
        / f"steps{integration_steps}"
        / source.name
    )
    if target.exists():
        return target
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["model_config"] = dict(checkpoint["model_config"])
    checkpoint["model_config"]["head_params"] = dict(
        checkpoint["model_config"]["head_params"]
    )
    checkpoint["model_config"]["head_params"]["sample_steps"] = int(integration_steps)
    checkpoint["inference_only_integration_steps"] = int(integration_steps)
    checkpoint["inference_parent_checkpoint"] = str(source.resolve())
    checkpoint["inference_parent_sha256"] = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, target)
    return target


def sample(
    root: Path,
    *,
    model_ids: list[str],
    particles: int,
    branch: int,
    future: int,
    integration_steps: int | None,
    train_seed: int,
    device: str,
) -> None:
    configs = _candidate_lookup()
    cohort = load_sbtg_cohort()
    folds = make_fold_assignments(cohort, n_folds=5, seed=FOLD_SEED)
    train_dir = (
        root / "training"
        if train_seed == TRAIN_SEED
        else root / f"training_seed{train_seed}"
    )
    for model_id in model_ids:
        config = configs[model_id]
        sampler_id = _sampler_id(
            model_id, particles, branch, future, integration_steps, train_seed
        )
        out = root / "sampling" / sampler_id
        out.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for fold in range(5):
            checkpoint = _inference_checkpoint(
                train_dir, config, fold, integration_steps, train_seed
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            result = run_one(
                cohort=cohort,
                folds=folds,
                source_run=train_dir,
                checkpoint_phase="optimization_fold_specific",
                output=out,
                method="progressive_bridge_smc",
                model_id=model_id,
                history_lag=8,
                fold=fold,
                seed=train_seed,
                source_lag=SOURCE_LAG,
                particles=particles,
                horizons=HORIZONS,
                source_window_frames=SOURCE_WINDOW_FRAMES,
                device=device,
                base_seed=BASE_SEED,
                min_ess=max(4.0, particles * 0.1875),
                progressive_branch_factor=branch,
                progressive_future_branch_factor=future,
                checkpoint_override=checkpoint,
                checkpoint_validation_profile="historical_sbtg80",
            )
            records.append({"fold": fold, **result})
            pd.DataFrame(records).to_csv(out / "run_status.csv", index=False)
        _atomic_json(
            out / "manifest.json",
            {
                "created_utc": _utc(),
                "status": "sampling_complete",
                "sampler_id": sampler_id,
                "model_id": model_id,
                "train_seed": train_seed,
                "particles": particles,
                "repair_branch_factor": branch,
                "future_branch_factor": future,
                "integration_steps": (
                    int(integration_steps)
                    if integration_steps is not None
                    else int(config.head_params["sample_steps"])
                ),
                "source_lag_frames": SOURCE_LAG,
                "horizon_frames": list(HORIZONS),
                "source_window_frames": SOURCE_WINDOW_FRAMES,
                "external_references_consulted": False,
                "checkpoint_sha256": {
                    str(fold): sha256(
                        _inference_checkpoint(
                            train_dir, config, fold, integration_steps, train_seed
                        )
                    )
                    for fold in range(5)
                },
            },
        )


def _sampling_archives(
    path: Path, model_id: str, particles: int, train_seed: int
) -> list[Path]:
    return [
        path
        / "responses"
        / "progressive_bridge_smc"
        / f"{model_id}__progressive_bridge_smc__ell1__N{particles}__f{fold}__s{train_seed}.npz"
        for fold in range(5)
    ]


def _matrix_from_sampling(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    manifest = json.loads((path / "manifest.json").read_text())
    model_id = str(manifest["model_id"])
    particles = int(manifest["particles"])
    train_seed = int(manifest.get("train_seed", TRAIN_SEED))
    cohort = load_sbtg_cohort()
    response = np.full((cohort.n_worms, 5, 3, 80, 1, 80), np.nan, dtype=np.float32)
    valid = np.full((cohort.n_worms, 5, 3, 80), np.nan, dtype=np.float32)
    gaps = np.full_like(valid, np.nan)
    codes = np.full((cohort.n_worms, 3), -1, dtype=np.int8)
    diagnostics: dict[str, list[float]] = {
        "wall_seconds": [],
        "ess_low": [],
        "ess_high": [],
        "max_weight_low": [],
        "max_weight_high": [],
        "distinct_ancestors_low": [],
        "distinct_ancestors_high": [],
    }
    for archive in _sampling_archives(path, model_id, particles, train_seed):
        with np.load(archive, allow_pickle=False) as data:
            idx = data["worm_indices"].astype(int)
            response[idx] = data["response_endpoint_mean"]
            valid[idx] = data["diagnostic_valid"]
            gaps[idx] = data["diagnostic_achieved_gap"]
            codes[idx] = data["chemical_code_by_worm_event"]
            diagnostics["wall_seconds"].append(float(data["wall_seconds"].item()))
            for name in diagnostics:
                if name == "wall_seconds":
                    continue
                diagnostics[name].append(float(np.mean(data[f"diagnostic_{name}"])))
    if not np.isfinite(response).all() or not np.isfinite(valid).all() or np.any(codes < 0):
        raise RuntimeError(f"incomplete sampling archive union: {path}")
    oriented = orient_response_once(response)
    denominator = effect_normalization_denominator(gaps, 0.10)
    normalized = oriented / denominator[:, :, :, None, None, :]
    contexts = _contextualize_values(normalized, codes)
    support_contexts, _ = _contextualize_support(valid, gaps, codes)
    matrix = contexts["state_average"].mean(axis=0)[0]
    support = support_contexts["state_average"].mean(axis=0)
    summary = {
        "sampler_wall_seconds": float(sum(diagnostics["wall_seconds"])),
        "valid_fraction_mean": float(valid.mean()),
        "valid_sources_ge_050": int((support >= 0.50).sum()),
    }
    for name, values in diagnostics.items():
        if name != "wall_seconds":
            summary[f"mean_{name}"] = float(np.mean(values))
    return matrix, support, summary


def evaluate(
    root: Path,
    *,
    reference_release: Path,
    published_archive: Path,
    bootstrap_repeats: int,
) -> None:
    evaluation = root / "postfreeze_external"
    evaluation.mkdir(parents=True, exist_ok=True)
    sampling_dirs = sorted((root / "sampling").glob("*/manifest.json"))
    if not sampling_dirs:
        raise RuntimeError("no frozen sampling candidates found")
    cohort = load_sbtg_cohort()
    refs80, _ = load_references(reference_release, list(cohort.neurons))
    clean54_path = Path("results/neural_prediction_atlas_20260829/canonical/atlas_matrices.npz")
    with np.load(clean54_path, allow_pickle=False) as clean:
        neurons54 = clean["neurons"].astype(str).tolist()
    index54 = [list(cohort.neurons).index(neuron) for neuron in neurons54]
    refs54, _ = load_references(reference_release, neurons54)
    published80 = _load_published_slices(published_archive, cohort.neurons)
    published_lag1 = next(item for item in published80 if int(item["lag_frames"]) == 1)

    rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    summaries: list[dict[str, Any]] = []
    for manifest_path in sampling_dirs:
        sample_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        sampler_id = str(manifest["sampler_id"])
        matrix, support, summary = _matrix_from_sampling(sample_dir)
        matrices[sampler_id] = matrix
        summaries.append({"sampler_id": sampler_id, **summary})
        item80 = {
            "method": sampler_id,
            "channel": "endpoint_mean",
            "context": "state_average",
            "lag_frames": 1,
            "horizon_frames": 1,
            "matrix": matrix,
            "support": support,
            "support_available": True,
            "training_lineage": "historical SBTG80 flow optimization",
            "shared_neuron_comparability": "native historical 80-class axis",
        }
        current80 = _reference_rows([item80], refs80, fps=cohort.fps)
        for row in current80:
            row["reference"] = str(row["reference"]).replace("_54", "_80")
            row["evaluation_axis"] = "historical80"
        rows.extend(current80)
        item54 = {
            **item80,
            "matrix": matrix[np.ix_(index54, index54)],
            "support": support[index54],
            "shared_neuron_comparability": "exact clean-54 subset of historical80 model",
        }
        current54 = _reference_rows([item54], refs54, fps=cohort.fps)
        for row in current54:
            row["evaluation_axis"] = "common54_subset"
        rows.extend(current54)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(evaluation / "candidate_reference_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(evaluation / "sampler_diagnostics.csv", index=False)

    primary = metrics[metrics.scope == "all_estimated"].copy()
    primary.to_csv(evaluation / "primary_all_estimated.csv", index=False)
    score = primary.pivot_table(
        index=["evaluation_axis", "method"],
        columns="reference",
        values=["auroc", "auprc", "macro_source_auroc"],
    )
    score.columns = [f"{metric}__{reference}" for metric, reference in score.columns]
    score = score.reset_index()
    score["mean_auroc"] = score.filter(regex=r"^auroc__").mean(axis=1)
    score["mean_auprc"] = score.filter(regex=r"^auprc__").mean(axis=1)
    score = score.sort_values(["evaluation_axis", "mean_auroc"], ascending=[True, False])
    score.to_csv(evaluation / "candidate_scoreboard.csv", index=False)

    best80 = score[score.evaluation_axis == "historical80"].iloc[0]
    best_id = str(best80.method)
    bootstrap = _paired_source_bootstrap(
        matrices[best_id],
        np.asarray(published_lag1["matrix"]),
        refs80,
        repeats=bootstrap_repeats,
        seed=20_260_905,
    )
    bootstrap.insert(0, "method", best_id)
    bootstrap.to_csv(evaluation / "best_paired_source_bootstrap.csv", index=False)
    _atomic_json(
        evaluation / "manifest.json",
        {
            "created_utc": _utc(),
            "stage": "postfreeze_external_exploration",
            "candidate_count": len(matrices),
            "best_historical80_mean_auroc_candidate": best_id,
            "selection_warning": (
                "candidate ranking by external correspondence is exploratory and cannot "
                "serve as an independent confirmation on these same references"
            ),
            "reference_release": str(reference_release.resolve()),
            "published_archive": str(published_archive.resolve()),
            "bootstrap_repeats": bootstrap_repeats,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("train", "train-seed", "sample", "evaluate")
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-ids", nargs="*")
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--branch", type=int, default=2)
    parser.add_argument("--future", type=int, default=1)
    parser.add_argument("--integration-steps", type=int)
    parser.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--reference-release", type=Path, default=DEFAULT_REFERENCE_RELEASE)
    parser.add_argument("--published-archive", type=Path, default=DEFAULT_PUBLISHED_ARCHIVE)
    parser.add_argument("--bootstrap-repeats", type=int, default=512)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.stage == "train":
        train(root, device=args.device, hours=args.hours, resume=args.resume)
    elif args.stage == "train-seed":
        train_secondary_seed(
            root,
            train_seed=args.train_seed,
            device=args.device,
            hours=args.hours,
            resume=args.resume,
        )
    elif args.stage == "sample":
        model_ids = args.model_ids
        if not model_ids:
            finalists = json.loads((root / "training/finalist_selection.json").read_text())
            model_ids = list(finalists["selected_model_ids"])
        sample(
            root,
            model_ids=model_ids,
            particles=args.particles,
            branch=args.branch,
            future=args.future,
            integration_steps=args.integration_steps,
            train_seed=args.train_seed,
            device=args.device,
        )
    else:
        evaluate(
            root,
            reference_release=args.reference_release.resolve(),
            published_archive=args.published_archive.resolve(),
            bootstrap_repeats=args.bootstrap_repeats,
        )


if __name__ == "__main__":
    main()
