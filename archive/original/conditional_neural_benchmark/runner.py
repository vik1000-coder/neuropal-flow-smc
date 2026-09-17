from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.baselines import GaussianBaseline
from conditional_neural_benchmark.data import (
    balance_window_strata,
    Cohort,
    FoldScaler,
    Windows,
    choose_evaluation_indices,
    load_cohort,
    make_fold_assignments,
    make_windows,
    multiscale_features,
    resample_window_strata,
)
from conditional_neural_benchmark.metrics import metric_rows, summarize_metrics
from conditional_neural_benchmark.models import build_encoded_model

from history_tangent_benchmark.models import fit_model, resolve_device


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "conditional_distribution_benchmark"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    encoder: str
    head: str
    residual_target: bool = False
    width: int = 64
    dropout: float = 0.0
    head_params: dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    training_scheme: str = "natural"
    history_noise_std: float = 0.0
    history_noise_copies: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_configs(profile: str) -> list[ModelConfig]:
    configs = [
        ModelConfig("flat_gaussian", "flat", "heteroscedastic_gaussian"),
        ModelConfig("gru_gaussian", "gru", "heteroscedastic_gaussian"),
        ModelConfig("tcn_gaussian", "tcn", "heteroscedastic_gaussian"),
        ModelConfig("transformer_gaussian", "transformer", "heteroscedastic_gaussian"),
        ModelConfig("flat_delta_gaussian", "flat", "heteroscedastic_gaussian", True),
        ModelConfig("gru_delta_gaussian", "gru", "heteroscedastic_gaussian", True),
        ModelConfig("tcn_delta_gaussian", "tcn", "heteroscedastic_gaussian", True),
        ModelConfig("transformer_delta_gaussian", "transformer", "heteroscedastic_gaussian", True),
    ]
    if profile == "smoke":
        return [configs[0], configs[4], configs[6]]
    configs.extend(
        [
            ModelConfig("tcn_delta_gaussian_dropout10", "tcn", "heteroscedastic_gaussian", True, dropout=0.10),
            ModelConfig("tcn_delta_gaussian_dropout25", "tcn", "heteroscedastic_gaussian", True, dropout=0.25),
            ModelConfig("tcn_delta_gaussian_wd1e3", "tcn", "heteroscedastic_gaussian", True, weight_decay=1e-3),
            ModelConfig("tcn_mean_residual", "tcn", "mean_mlp", True),
            ModelConfig(
                "tcn_delta_lowrank4", "tcn", "lowrank_gaussian", True,
                head_params={"rank": 4},
            ),
            ModelConfig(
                "tcn_delta_lowrank8", "tcn", "lowrank_gaussian", True,
                head_params={"rank": 8},
            ),
            ModelConfig(
                "tcn_delta_gaussian_dsm", "tcn", "constrained_gaussian_dsm", True,
                head_params={"sigma_min": 0.03, "sigma_max": 1.5},
            ),
            ModelConfig(
                "tcn_delta_realnvp4", "tcn", "conditional_affine_flow", True,
                head_params={"hidden": 64, "layers": 2, "coupling_layers": 4},
            ),
            ModelConfig(
                "gru_delta_realnvp8", "gru", "conditional_affine_flow", True,
                head_params={"hidden": 64, "layers": 2, "coupling_layers": 8},
            ),
            ModelConfig(
                "tcn_delta_mdn2", "tcn", "autoregressive_mdn", True,
                head_params={"hidden": 48, "layers": 1, "components": 2},
            ),
            ModelConfig(
                "tcn_delta_mdn4", "tcn", "autoregressive_mdn", True,
                head_params={"hidden": 48, "layers": 1, "components": 4},
            ),
            ModelConfig(
                "tcn_delta_output_transformer_mdn", "tcn", "autoregressive_transformer", True,
                head_params={
                    "d_model": 32, "nhead": 4, "transformer_layers": 2,
                    "feedforward": 96, "components": 4,
                },
            ),
            ModelConfig(
                "tcn_delta_edm", "tcn", "conditional_edm_diffusion", True,
                head_params={"hidden": 96, "layers": 3, "sample_steps": 16},
            ),
            ModelConfig(
                "tcn_delta_gaussian_anchored_edm", "tcn", "gaussian_anchored_edm", True,
                head_params={"hidden": 96, "layers": 3, "sample_steps": 16},
            ),
            ModelConfig(
                "tcn_delta_flow_matching", "tcn", "conditional_flow_matching", True,
                head_params={"hidden": 96, "layers": 3, "sample_steps": 16},
            ),
            ModelConfig(
                "tcn_delta_gaussian_source_flow", "tcn", "gaussian_source_flow_matching", True,
                head_params={"hidden": 96, "layers": 3, "sample_steps": 16},
            ),
            ModelConfig(
                "tcn_delta_bounded_energy", "tcn", "bounded_energy_ratio", True,
                head_params={
                    "hidden": 64, "layers": 2, "tilt_bound": 1.5,
                    "base_nll_weight": 0.5, "oversample": 4,
                },
            ),
            ModelConfig(
                "tcn_delta_stochastic_interpolant", "tcn",
                "conditional_point_source_stochastic_interpolant", True,
                head_params={"hidden": 96, "layers": 3, "sample_steps": 24},
            ),
        ]
    )
    return configs


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "conditional_neural_benchmark").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _device_name(requested: str) -> str:
    return str(resolve_device(requested))


def _fold_rows(cohort: Cohort, folds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "worm_index": np.arange(cohort.n_worms),
            "worm_id": cohort.worm_ids,
            "strain": cohort.strains,
            "outer_fold": folds,
        }
    )


def _split_indices(folds: np.ndarray, test_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_folds = int(folds.max()) + 1
    validation_fold = (test_fold + 1) % n_folds
    test = np.flatnonzero(folds == test_fold)
    validation = np.flatnonzero(folds == validation_fold)
    train = np.flatnonzero((folds != test_fold) & (folds != validation_fold))
    return train, validation, test


def _split_windows(
    cohort: Cohort,
    folds: np.ndarray,
    fold: int,
    lag: int,
    *,
    stimulus_encoding: str = "binary_any_stimulus",
    chemical_permutations: dict[int, tuple[int, int, int]] | None = None,
) -> tuple[FoldScaler, Windows, Windows, Windows]:
    train_idx, validation_idx, test_idx = _split_indices(folds, fold)
    scaler = FoldScaler.fit(cohort.traces[i] for i in train_idx)
    return (
        scaler,
        make_windows(
            cohort, train_idx, lag, scaler,
            stimulus_encoding=stimulus_encoding,
            chemical_permutations=chemical_permutations,
        ),
        make_windows(
            cohort, validation_idx, lag, scaler,
            stimulus_encoding=stimulus_encoding,
            chemical_permutations=chemical_permutations,
        ),
        make_windows(
            cohort, test_idx, lag, scaler,
            stimulus_encoding=stimulus_encoding,
            chemical_permutations=chemical_permutations,
        ),
    )


def _sample_neural(
    model: torch.nn.Module,
    flat_history: np.ndarray,
    target: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    device: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    samples: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []
    exact = bool(model.capabilities.normalized_density)
    target_device = torch.device(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat_history), chunk_size):
            h = torch.as_tensor(
                flat_history[start : start + chunk_size], dtype=torch.float32, device=target_device
            )
            y = torch.as_tensor(
                target[start : start + chunk_size], dtype=torch.float32, device=target_device
            )
            draw = model.sample(h, n_samples, seed=seed + start)
            samples.append(draw.detach().cpu().numpy())
            if exact:
                log_probs.append(model.log_prob(y, h).detach().cpu().numpy())
    return np.concatenate(samples), (np.concatenate(log_probs) if exact else None)


def _evaluate_draws(
    samples: np.ndarray,
    target: np.ndarray,
    strata: np.ndarray,
    log_prob: np.ndarray | None,
    seed: int,
    population_strata: np.ndarray | None = None,
) -> dict[str, float]:
    rows = metric_rows(samples, target, seed)
    return summarize_metrics(
        rows, strata, log_prob, target.shape[1], population_strata=population_strata
    )


def _worm_chemical_metrics(
    samples: np.ndarray,
    target: np.ndarray,
    worm: np.ndarray,
    chemical_code: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Equal-weight active-event proper scores across worm×chemical cells."""
    rows = metric_rows(samples, target, seed)
    active = np.asarray(chemical_code) > 0
    if not active.any():
        return {}
    result: dict[str, float] = {}
    names = {1: "butanone", 2: "pentanedione", 3: "nacl"}
    pairs = sorted(set(zip(np.asarray(worm)[active], np.asarray(chemical_code)[active])))
    for metric, values in rows.items():
        cell_values = []
        by_chemical: dict[int, list[float]] = {1: [], 2: [], 3: []}
        for worm_index, code in pairs:
            mask = (worm == worm_index) & (chemical_code == code)
            value = float(np.mean(values[mask]))
            cell_values.append(value)
            by_chemical[int(code)].append(value)
        result[f"{metric}__worm_chemical_balanced"] = float(np.mean(cell_values))
        for code, label in names.items():
            if by_chemical[code]:
                result[f"{metric}__chemical_{label}"] = float(
                    np.mean(by_chemical[code])
                )
    result["n_worm_chemical_cells"] = len(pairs)
    return result


class RunState:
    def __init__(self, run_dir: Path, deadline: float):
        self.run_dir = run_dir
        self.deadline = deadline
        self.records: list[dict[str, Any]] = []

    def time_left_hours(self) -> float:
        return max(0.0, (self.deadline - time.monotonic()) / 3600)

    def can_start(self, reserve_minutes: float = 5.0) -> bool:
        return time.monotonic() + reserve_minutes * 60 < self.deadline

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        pd.DataFrame(self.records).to_csv(self.run_dir / "trial_metrics.csv", index=False)
        _atomic_json(
            self.run_dir / "status.json",
            {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "trials": len(self.records),
                "ok": sum(r.get("status") == "ok" for r in self.records),
                "failed": sum(r.get("status") == "failed" for r in self.records),
                "hours_left": self.time_left_hours(),
            },
        )


def _save_baseline(path: Path, model: GaussianBaseline, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = model.state_dict()
    values["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **values)


def _baseline_trial(
    *,
    state: RunState,
    phase: str,
    kind: str,
    lag: int,
    fold: int,
    seed: int,
    train: Windows,
    validation: Windows,
    test: Windows,
    scaler: FoldScaler,
    eval_rows: int,
    n_samples: int,
) -> None:
    started = time.perf_counter()
    model_id = "persistence_full_gaussian" if kind == "persistence" else "ridge_full_gaussian"
    print(f"TRIAL_START phase={phase} model={model_id} lag={lag} fold={fold}", flush=True)
    try:
        if kind == "persistence":
            model = GaussianBaseline.persistence(train.history, train.target)
            test_features = None
            selected_alpha = None
        else:
            train_features = multiscale_features(train.history, train.target.shape[1])
            validation_features = multiscale_features(
                validation.history, validation.target.shape[1]
            )
            best: tuple[float, float] | None = None
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                candidate = GaussianBaseline.ridge(train_features, train.target, alpha)
                nll = -candidate.log_prob(validation.target, features=validation_features).mean()
                if best is None or nll < best[0]:
                    best = (float(nll), alpha)
            assert best is not None
            selected_alpha = best[1]
            # Refit on train+validation only after alpha selection.
            fit_features = np.concatenate([train_features, validation_features])
            fit_target = np.concatenate([train.target, validation.target])
            model = GaussianBaseline.ridge(fit_features, fit_target, selected_alpha)
            test_features = multiscale_features(test.history, test.target.shape[1])
        idx = choose_evaluation_indices(test.stratum, eval_rows, seed + fold)
        draw = model.sample(
            n_samples,
            seed + 500,
            history=test.history[idx] if kind == "persistence" else None,
            features=test_features[idx] if test_features is not None else None,
        )
        log_prob = model.log_prob(
            test.target[idx],
            history=test.history[idx] if kind == "persistence" else None,
            features=test_features[idx] if test_features is not None else None,
        )
        metrics = _evaluate_draws(
            draw, test.target[idx], test.stratum[idx], log_prob, seed,
            population_strata=test.stratum,
        )
        metrics.update(_worm_chemical_metrics(
            draw,
            test.target[idx],
            test.worm[idx],
            test.chemical_code[idx],
            seed + 2000,
        ))
        checkpoint = state.run_dir / "checkpoints" / phase / f"{model_id}__L{lag}__f{fold}.npz"
        _save_baseline(
            checkpoint,
            model,
            {"model_id": model_id, "lag": lag, "fold": fold, "alpha": selected_alpha, "scaler": scaler.to_dict()},
        )
        record = {
            "phase": phase, "model_id": model_id, "model_family": "baseline",
            "encoder": "multiscale_linear" if kind == "ridge" else "identity",
            "head": "full_shrinkage_gaussian", "lag": lag, "fold": fold,
            "seed": seed, "status": "ok", "selected_alpha": selected_alpha,
            "wall_seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint.relative_to(state.run_dir)), **metrics,
        }
        state.add(record)
        print(
            f"TRIAL_DONE model={model_id} energy={metrics['energy']:.5f} "
            f"balanced={metrics['energy__stim_balanced']:.5f} sec={record['wall_seconds']:.1f}",
            flush=True,
        )
    except Exception as error:
        state.add(
            {
                "phase": phase, "model_id": model_id, "model_family": "baseline",
                "lag": lag, "fold": fold, "seed": seed, "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=8),
                "wall_seconds": time.perf_counter() - started,
            }
        )
        print(f"TRIAL_FAILED model={model_id}: {type(error).__name__}: {error}", flush=True)


def _neural_trial(
    *,
    state: RunState,
    phase: str,
    config: ModelConfig,
    cohort: Cohort,
    lag: int,
    fold: int,
    seed: int,
    train: Windows,
    validation: Windows,
    test: Windows,
    scaler: FoldScaler,
    device: str,
    max_epochs: int,
    patience: int,
    eval_rows: int,
    n_samples: int,
    batch_size: int,
    trial_metadata: dict[str, Any] | None = None,
) -> None:
    started = time.perf_counter()
    print(
        f"TRIAL_START phase={phase} model={config.model_id} lag={lag} fold={fold} seed={seed}",
        flush=True,
    )
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        input_channels = int(train.history.shape[2])
        stimulus_channels = input_channels - cohort.n_neurons
        if stimulus_channels < 1:
            raise ValueError("at least one stimulus channel is required")
        if validation.history.shape[2] != input_channels or test.history.shape[2] != input_channels:
            raise ValueError("train/validation/test stimulus channel counts disagree")
        model = build_encoded_model(
            head_name=config.head,
            encoder_name=config.encoder,
            lag=lag,
            channels=input_channels,
            dy=cohort.n_neurons,
            width=config.width,
            dropout=config.dropout,
            head_params=config.head_params,
        )
        natural_train_counts = {
            str(label): int(count)
            for label, count in zip(*np.unique(train.stratum, return_counts=True))
        }
        if config.training_scheme == "stratum_balanced":
            fit_train = balance_window_strata(
                train, seed=seed + 31 * fold, total_rows=len(train.target), replace_rare=True
            )
            fit_validation = balance_window_strata(
                validation, seed=seed + 47 * fold, replace_rare=False
            )
        elif config.training_scheme == "transition_moderate":
            labels = sorted(set(np.asarray(train.stratum).astype(str).tolist()))
            quiet_weight = 0.40 if "off" in labels else 0.0
            transition_labels = [label for label in labels if label != "off"]
            remaining = 1.0 - quiet_weight
            proportions = {
                label: (
                    quiet_weight if label == "off" else remaining / len(transition_labels)
                )
                for label in labels
            }
            fit_train = resample_window_strata(
                train, proportions=proportions, seed=seed + 31 * fold,
                total_rows=len(train.target),
            )
            fit_validation = validation
        elif config.training_scheme == "natural":
            fit_train, fit_validation = train, validation
        else:
            raise ValueError(f"unknown training scheme {config.training_scheme}")
        fit_train_counts = {
            str(label): int(count)
            for label, count in zip(*np.unique(fit_train.stratum, return_counts=True))
        }
        train_h = fit_train.flat()
        validation_h = fit_validation.flat()
        train_target = (
            fit_train.target - fit_train.history[:, -1, :cohort.n_neurons]
            if config.residual_target else fit_train.target
        )
        validation_target = (
            fit_validation.target - fit_validation.history[:, -1, :cohort.n_neurons]
            if config.residual_target else fit_validation.target
        )
        if config.history_noise_std < 0 or config.history_noise_copies < 0:
            raise ValueError("history-noise settings must be nonnegative")
        history_noise_mask = np.tile(
            np.concatenate(
                [
                    np.ones(cohort.n_neurons, dtype=np.float32),
                    np.zeros(stimulus_channels, dtype=np.float32),
                ]
            ),
            lag,
        )
        trace = fit_model(
            model,
            train_h,
            train_target,
            validation_h,
            validation_target,
            seed=seed,
            device=device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            gradient_clip=1.0,
            input_noise_std=config.history_noise_std,
            input_noise_mask=history_noise_mask,
            input_noise_draws=config.history_noise_copies,
        )
        target_device = torch.device(device)
        if hasattr(model, "finalize"):
            with torch.no_grad():
                model.finalize(
                    torch.as_tensor(train_h, dtype=torch.float32, device=target_device),
                    torch.as_tensor(train_target, dtype=torch.float32, device=target_device),
                )
        idx = choose_evaluation_indices(test.stratum, eval_rows, seed + 13 * fold)
        test_h = test.flat()[idx]
        last_observed = test.history[idx, -1, :cohort.n_neurons]
        model_target = test.target[idx] - last_observed if config.residual_target else test.target[idx]
        draws, log_prob = _sample_neural(
            model,
            test_h,
            model_target,
            n_samples=n_samples,
            seed=seed + 1000,
            device=device,
            chunk_size=96 if config.head in {"autoregressive_mdn", "autoregressive_transformer"} else 192,
        )
        if config.residual_target:
            draws = draws + last_observed[:, None, :]
        metrics = _evaluate_draws(
            draws, test.target[idx], test.stratum[idx], log_prob, seed,
            population_strata=test.stratum,
        )
        metrics.update(_worm_chemical_metrics(
            draws,
            test.target[idx],
            test.worm[idx],
            test.chemical_code[idx],
            seed + 2000,
        ))
        checkpoint = (
            state.run_dir / "checkpoints" / phase /
            f"{config.model_id}__L{lag}__f{fold}__s{seed}.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_config": config.to_dict(),
                "lag": lag,
                "fold": fold,
                "seed": seed,
                "neurons": list(cohort.neurons),
                "input_channels": input_channels,
                "stimulus_channels": stimulus_channels,
                "scaler": scaler.to_dict(),
                "fit_trace": trace.to_dict(),
                "training_scheme": config.training_scheme,
                "natural_train_stratum_counts": natural_train_counts,
                "fit_train_stratum_counts": fit_train_counts,
                "history_noise_std": config.history_noise_std,
                "history_noise_copies": config.history_noise_copies,
                "trial_metadata": dict(trial_metadata or {}),
                "stimulus_schema_version": cohort.stimulus_schema_version,
                "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
                "stimulus_schema": cohort.stimulus_schema_dict(),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            },
            checkpoint,
        )
        record = {
            "phase": phase, "model_id": config.model_id, "model_family": "neural",
            "encoder": config.encoder, "head": config.head, "lag": lag, "fold": fold,
            "residual_target": config.residual_target,
            "training_scheme": config.training_scheme,
            "natural_train_rows": len(train.target),
            "fit_train_rows": len(fit_train.target),
            "seed": seed, "status": "ok", "wall_seconds": time.perf_counter() - started,
            "train_seconds": trace.wall_seconds, "best_epoch": trace.best_epoch,
            "stopped_epoch": trace.stopped_epoch, "parameter_count": trace.parameter_count,
            "device": trace.device, "checkpoint": str(checkpoint.relative_to(state.run_dir)),
            **dict(trial_metadata or {}),
            **metrics,
        }
        state.add(record)
        print(
            f"TRIAL_DONE model={config.model_id} energy={metrics['energy']:.5f} "
            f"balanced={metrics['energy__stim_balanced']:.5f} epoch={trace.best_epoch} "
            f"sec={record['wall_seconds']:.1f}", flush=True,
        )
        del model, train_h, validation_h, draws
        if device == "mps":
            torch.mps.empty_cache()
    except Exception as error:
        state.add(
            {
                "phase": phase, "model_id": config.model_id, "model_family": "neural",
                "encoder": config.encoder, "head": config.head, "lag": lag, "fold": fold,
                "seed": seed, "status": "failed", "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=12),
                "wall_seconds": time.perf_counter() - started,
            }
        )
        print(f"TRIAL_FAILED model={config.model_id}: {type(error).__name__}: {error}", flush=True)
        if device == "mps":
            torch.mps.empty_cache()


def _leaderboard(records: list[dict[str, Any]], phase: str) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty or "phase" not in frame or "status" not in frame:
        return pd.DataFrame()
    frame = frame[(frame.phase == phase) & (frame.status == "ok")].copy()
    if frame.empty:
        return pd.DataFrame()
    keys = ["model_id", "model_family", "encoder", "head", "lag"]
    metric_names = [
        "energy", "energy__stim_balanced", "variogram", "variogram__stim_balanced",
        "rmse", "coverage90", "sharpness90", "nll_per_neuron", "wall_seconds",
    ]
    available = [name for name in metric_names if name in frame]
    grouped = frame.groupby(keys, dropna=False)
    means = grouped[available].mean().add_suffix("__mean")
    counts = grouped.size().rename("n_trials")
    std = grouped[available].std(ddof=1)
    ses = std.div(np.sqrt(counts), axis=0).add_suffix("__se")
    board = pd.concat([means, ses, counts], axis=1).reset_index()
    best_energy = board["energy__mean"].min()
    best_row = board.loc[board["energy__mean"].idxmin()]
    tolerance = float(best_row.get("energy__se", 0.0))
    if not math.isfinite(tolerance):
        tolerance = 0.0
    board["within_1se_natural"] = board["energy__mean"] <= best_energy + tolerance
    board = board.sort_values(
        ["within_1se_natural", "energy__stim_balanced__mean", "variogram__mean", "energy__mean"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board


def _update_leaderboards(state: RunState) -> None:
    boards = []
    for phase in sorted({str(r.get("phase")) for r in state.records}):
        board = _leaderboard(state.records, phase)
        if not board.empty:
            board.insert(0, "phase", phase)
            boards.append(board)
    if boards:
        pd.concat(boards, ignore_index=True).to_csv(state.run_dir / "leaderboard.csv", index=False)


def _choose_lag(state: RunState, candidate_lags: list[int]) -> int:
    board = _leaderboard(state.records, "lag_screen")
    if board.empty or "model_id" not in board:
        return candidate_lags[0]
    ridge = board[board.model_id == "ridge_full_gaussian"].copy()
    if ridge.empty:
        return candidate_lags[0]
    ridge = ridge.sort_values(
        ["within_1se_natural", "energy__stim_balanced__mean", "energy__mean"],
        ascending=[False, True, True],
    )
    return int(ridge.iloc[0].lag)


def _choose_finalists(state: RunState, configs: list[ModelConfig], count: int) -> list[ModelConfig]:
    board = _leaderboard(state.records, "architecture_screen")
    neural = board[board.model_family == "neural"]
    ids = neural.model_id.head(count).tolist()
    # Keep at least one exact-density and one implicit-generator finalist when available.
    exact_heads = {
        "heteroscedastic_gaussian", "conditional_affine_flow", "autoregressive_mdn",
        "autoregressive_transformer",
    }
    implicit_heads = {
        "conditional_edm_diffusion", "gaussian_anchored_edm", "conditional_flow_matching",
        "gaussian_source_flow_matching", "conditional_point_source_stochastic_interpolant",
        "bounded_energy_ratio",
    }
    by_id = {c.model_id: c for c in configs}
    selected = [by_id[i] for i in ids if i in by_id]
    for family in (exact_heads, implicit_heads):
        if any(c.head in family for c in selected):
            continue
        for model_id in neural.model_id:
            candidate = by_id.get(model_id)
            if candidate is not None and candidate.head in family:
                if len(selected) >= count:
                    selected[-1] = candidate
                else:
                    selected.append(candidate)
                break
    unique: list[ModelConfig] = []
    seen: set[str] = set()
    for config in selected:
        if config.model_id not in seen:
            unique.append(config)
            seen.add(config.model_id)
    return unique


def _write_report(
    run_dir: Path,
    cohort: Cohort,
    best_lag: int,
    state: RunState,
    complete: bool,
) -> None:
    final_board = _leaderboard(state.records, "final_confirmation")
    if final_board.empty:
        final_board = _leaderboard(state.records, "architecture_screen")
    winner = final_board.iloc[0] if not final_board.empty else None
    records = pd.DataFrame(state.records)
    ridge = None
    if not final_board.empty and np.any(final_board.model_id == "ridge_full_gaussian"):
        ridge = final_board[final_board.model_id == "ridge_full_gaussian"].iloc[0]
    observed_labels: list[str] = []
    winner_trials = pd.DataFrame()
    if winner is not None and not records.empty:
        winner_trials = records[
            (records.phase == "final_confirmation")
            & (records.model_id == winner["model_id"])
            & (records.status == "ok")
        ]
        for column in records.columns:
            if not column.startswith("population_n__"):
                continue
            if column in winner_trials and winner_trials[column].notna().any():
                observed_labels.append(column.replace("population_n__", ""))
    deployment_path = run_dir / "deployment_manifest.json"
    deployment = json.loads(deployment_path.read_text()) if deployment_path.exists() else None
    lines = [
        "# Conditional NeuroPAL distribution benchmark",
        "",
        f"Run directory: `{run_dir}`",
        f"Status: {'complete' if complete else 'time-bounded partial'}",
        "",
        "## Estimand",
        "",
        r"The benchmark estimates $p(x_{t+1}\mid x_{t-L+1:t}, s_{t-L+1:t})$, where "
        r"$x_t$ is the observed NeuroPAL calcium-activity vector and $s_t$ is the binary "
        r"stimulus-presence indicator. The forecast horizon is one frame (0.25 s).",
        "",
        "## Data and validation",
        "",
        f"- Primary cohort: {cohort.n_worms} worms, {cohort.n_neurons} quality-filtered neurons, "
        f"{sum(map(len, cohort.traces)):,} frames at {cohort.fps:g} Hz.",
        f"- Coordinates removed for inadequate within-worm observation: {', '.join(cohort.quality_dropped_neurons) or 'none'}.",
        "- Isolated missing history samples use causal carry-forward; any window with an unobserved next-frame target is excluded rather than target-imputed.",
        "- Splits are by whole worm. The validation fold controls early stopping; test worms are never window-shuffled into training.",
        "- Neural scaling is fitted on training worms only for every fold.",
        f"- Selected history: L={best_lag} frames ({best_lag / cohort.fps:.2f} s).",
        f"- Primary proper score: multivariate energy score; stimulus-balanced energy averages the observed window patterns ({', '.join(observed_labels)}) equally.",
        "- Dependence tie-break: variogram score. Lower is better for energy, variogram, NLL, RMSE, and sharpness.",
        "",
        "## Result",
        "",
    ]
    if winner is not None:
        improvement_energy = (
            100.0 * (ridge["energy__mean"] - winner["energy__mean"]) / ridge["energy__mean"]
            if ridge is not None else float("nan")
        )
        improvement_balanced = (
            100.0
            * (ridge["energy__stim_balanced__mean"] - winner["energy__stim_balanced__mean"])
            / ridge["energy__stim_balanced__mean"]
            if ridge is not None else float("nan")
        )
        improvement_variogram = (
            100.0 * (ridge["variogram__mean"] - winner["variogram__mean"])
            / ridge["variogram__mean"]
            if ridge is not None else float("nan")
        )
        lines.extend(
            [
                f"The selected model is **{winner['model_id']}** (encoder `{winner['encoder']}`, head `{winner['head']}`).",
                "",
                f"- Cross-validated natural energy score: {winner['energy__mean']:.6f}",
                f"- Stimulus-balanced energy score: {winner['energy__stim_balanced__mean']:.6f}",
                f"- Variogram score: {winner['variogram__mean']:.6f}",
                f"- Marginal 90% interval coverage: {winner['coverage90__mean']:.4f}",
                f"- Trials aggregated: {int(winner.n_trials)}",
                f"- Improvement versus nested-alpha ridge: {improvement_energy:.1f}% energy, {improvement_balanced:.1f}% balanced energy, {improvement_variogram:.1f}% variogram.",
                "",
            ]
        )
        if not winner_trials.empty:
            lines.extend(["### Energy score by stimulus-history pattern", ""])
            for label in observed_labels:
                column = f"energy__{label}"
                if column in winner_trials:
                    lines.append(f"- {label}: {winner_trials[column].mean():.6f}")
            lines.append("")
    else:
        lines.extend(["No trial completed successfully.", ""])
    if deployment is not None:
        lines.extend(
            [
                "## All-data deployment ensemble",
                "",
                f"Three `{deployment['model_id']}` members were trained on all {deployment['n_worms']} worms and {deployment['n_training_windows']:,} usable windows for {deployment['epochs_transferred_from_cv_median']} epochs.",
                "The duration was transferred from the median cross-validated best epoch; deployment fits were not scored as if they were held out.",
                "",
            ]
        )
        for member in deployment["members"]:
            lines.append(f"- seed {member['seed']}: `{member['checkpoint']}`")
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This ranks predictive conditional laws for observed neural activity. It does not identify anatomical connectivity, latent causal edges, or stimulus-specific molecular mechanisms.",
            "Flow matching supplies conditional samples but not an exact normalized likelihood; its selection is based on multivariate proper sample scores. The exact-likelihood MDN and Gaussian finalists remain in the leaderboard.",
            "The sample contains 20 worms, so fold-to-fold biological heterogeneity remains an important uncertainty source.",
            "",
            "## Artifacts",
            "",
            "- `leaderboard.csv`: phase-specific aggregate ranking with standard errors.",
            "- `trial_metrics.csv`: every fold/seed result, including failures.",
            "- `fold_assignments.csv`: immutable whole-worm split assignments.",
            "- `checkpoints/`: fitted weights, scalers, model configurations, and fit traces.",
            "- `manifest.json`: provenance and resolved run settings.",
            "- `deployment_manifest.json`: all-worm ensemble metadata and the warning separating deployment fits from unbiased evaluation.",
            "- `figures/`: lag selection and final-model comparison plots.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    started_wall = datetime.now().astimezone()
    run_id = args.run_id or started_wall.strftime("%Y%m%d_%H%M%S") + f"_{args.profile}"
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.hours * 3600
    state = RunState(run_dir, deadline)

    cohort = load_cohort(args.coverage)
    folds = make_fold_assignments(cohort, n_folds=args.folds, seed=args.split_seed)
    _fold_rows(cohort, folds).to_csv(run_dir / "fold_assignments.csv", index=False)
    device = _device_name(args.device)
    configs = model_configs(args.profile)
    manifest = {
        "run_id": run_id,
        "started_local": started_wall.isoformat(),
        "profile": args.profile,
        "hours_budget": args.hours,
        "coverage": args.coverage,
        "complete_case": True,
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "n_frames": int(sum(map(len, cohort.traces))),
        "fps": cohort.fps,
        "neurons": list(cohort.neurons),
        "quality_dropped_neurons": list(cohort.quality_dropped_neurons),
        "raw_nonfinite_values_before_quality_filter": cohort.raw_nonfinite_values,
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "source_sha256": _source_hash(),
        "candidate_lags": args.lags,
        "forecast_horizon_frames": 1,
        "conditioning": "neural_history_plus_aligned_binary_stimulus_history",
        "cohort_mode": cohort.cohort_mode,
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_schema": cohort.stimulus_schema_dict(),
        "model_configs": [c.to_dict() for c in configs],
        "selection_rule": "within 1SE of best natural energy, then stimulus-balanced energy, then variogram",
        "fixed_lag": args.fixed_lag,
        "lag_evidence_run": args.lag_evidence_run,
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    print(f"RUN_DIR {run_dir}", flush=True)
    print(f"DEVICE {device} HOURS {args.hours:.2f}", flush=True)

    # Stage 1: cheap but joint full-covariance nested-alpha lag screen. A fixed
    # lag may reuse a completed prior screen after an implementation-only restart.
    if args.fixed_lag is not None:
        best_lag = int(args.fixed_lag)
        if args.lag_evidence_run:
            evidence_dir = Path(args.lag_evidence_run).resolve()
            evidence_csv = evidence_dir / "trial_metrics.csv"
            evidence = pd.read_csv(evidence_csv)
            evidence = evidence[(evidence.phase == "lag_screen") & (evidence.status == "ok")]
            for row in evidence.to_dict(orient="records"):
                checkpoint = str(row.get("checkpoint", ""))
                if checkpoint:
                    row["checkpoint"] = str(evidence_dir / checkpoint)
                row["evidence_source"] = str(evidence_dir)
                state.records.append(row)
            if state.records:
                pd.DataFrame(state.records).to_csv(run_dir / "trial_metrics.csv", index=False)
                _update_leaderboards(state)
    else:
        lag_folds = list(range(args.folds)) if args.profile != "smoke" else [0]
        for lag in args.lags:
            for fold in lag_folds:
                if not state.can_start():
                    break
                scaler, train, validation, test = _split_windows(cohort, folds, fold, lag)
                _baseline_trial(
                    state=state, phase="lag_screen", kind="ridge", lag=lag, fold=fold,
                    seed=args.seed, train=train, validation=validation, test=test, scaler=scaler,
                    eval_rows=args.screen_eval_rows, n_samples=args.screen_samples,
                )
            _update_leaderboards(state)
            if not state.can_start():
                break
        best_lag = _choose_lag(state, args.lags)
    print(f"SELECTED_LAG L={best_lag} seconds={best_lag / cohort.fps:.2f}", flush=True)

    # Stage 2: broad distribution/encoder/regularization architecture screen.
    screen_folds = [0] if args.profile == "smoke" else list(range(min(3, args.folds)))
    for fold in screen_folds:
        if not state.can_start():
            break
        scaler, train, validation, test = _split_windows(cohort, folds, fold, best_lag)
        for kind in ("persistence", "ridge"):
            if state.can_start():
                _baseline_trial(
                    state=state, phase="architecture_screen", kind=kind, lag=best_lag,
                    fold=fold, seed=args.seed, train=train, validation=validation, test=test,
                    scaler=scaler, eval_rows=args.screen_eval_rows,
                    n_samples=args.screen_samples,
                )
        for config in configs:
            if not state.can_start():
                break
            _neural_trial(
                state=state, phase="architecture_screen", config=config, cohort=cohort,
                lag=best_lag, fold=fold, seed=args.seed, train=train, validation=validation,
                test=test, scaler=scaler, device=device, max_epochs=args.screen_epochs,
                patience=args.screen_patience, eval_rows=args.screen_eval_rows,
                n_samples=args.screen_samples, batch_size=args.batch_size,
            )
        _update_leaderboards(state)

    finalists = _choose_finalists(state, configs, args.finalists)
    print("FINALISTS " + ",".join(c.model_id for c in finalists), flush=True)

    # Stage 3: all outer folds, longer early-stopped fits, independent seeds.
    final_seeds = [args.seed] if args.profile == "smoke" else args.final_seeds
    for fold in range(args.folds if args.profile != "smoke" else 1):
        if not state.can_start():
            break
        scaler, train, validation, test = _split_windows(cohort, folds, fold, best_lag)
        # Deterministic comparators need one fit, not fake seed replication.
        for kind in ("persistence", "ridge"):
            if state.can_start():
                _baseline_trial(
                    state=state, phase="final_confirmation", kind=kind, lag=best_lag,
                    fold=fold, seed=args.seed, train=train, validation=validation, test=test,
                    scaler=scaler, eval_rows=args.final_eval_rows,
                    n_samples=args.final_samples,
                )
        for config in finalists:
            for seed in final_seeds:
                if not state.can_start():
                    break
                _neural_trial(
                    state=state, phase="final_confirmation", config=config, cohort=cohort,
                    lag=best_lag, fold=fold, seed=seed, train=train, validation=validation,
                    test=test, scaler=scaler, device=device, max_epochs=args.final_epochs,
                    patience=args.final_patience, eval_rows=args.final_eval_rows,
                    n_samples=args.final_samples, batch_size=args.batch_size,
                )
        _update_leaderboards(state)

    # Stage 4: test lag stability for the provisional winner if time remains.
    final_board = _leaderboard(state.records, "final_confirmation")
    if not final_board.empty and state.can_start(20):
        winner_id = str(final_board[final_board.model_family == "neural"].iloc[0].model_id)
        winner_config = next((c for c in configs if c.model_id == winner_id), None)
        nearby = sorted(set([max(2, best_lag // 2), best_lag, min(80, best_lag * 2)]))
        if winner_config is not None:
            for lag in nearby:
                if lag == best_lag:
                    continue
                for fold in range(min(3, args.folds)):
                    if not state.can_start(10):
                        break
                    scaler, train, validation, test = _split_windows(cohort, folds, fold, lag)
                    _neural_trial(
                        state=state, phase="winner_lag_sensitivity", config=winner_config,
                        cohort=cohort, lag=lag, fold=fold, seed=args.seed, train=train,
                        validation=validation, test=test, scaler=scaler, device=device,
                        max_epochs=args.screen_epochs, patience=args.screen_patience,
                        eval_rows=args.screen_eval_rows, n_samples=args.screen_samples,
                        batch_size=args.batch_size,
                    )
                _update_leaderboards(state)

    complete = state.can_start(0.0)
    _update_leaderboards(state)
    _write_report(run_dir, cohort, best_lag, state, complete)
    manifest["finished_local"] = datetime.now().astimezone().isoformat()
    manifest["selected_lag"] = best_lag
    manifest["n_trials"] = len(state.records)
    manifest["status"] = "complete" if complete else "time_budget_exhausted"
    _atomic_json(run_dir / "manifest.json", manifest)
    print(f"RUN_FINISHED {run_dir} trials={len(state.records)}", flush=True)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "overnight"), default="overnight")
    parser.add_argument("--hours", type=float, default=9.5)
    parser.add_argument("--run-id")
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260825)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[1701, 2903, 4307])
    parser.add_argument("--lags", type=int, nargs="+", default=[2, 4, 8, 16, 32, 40, 64, 80])
    parser.add_argument("--fixed-lag", type=int)
    parser.add_argument("--lag-evidence-run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--screen-epochs", type=int, default=14)
    parser.add_argument("--screen-patience", type=int, default=4)
    parser.add_argument("--final-epochs", type=int, default=45)
    parser.add_argument("--final-patience", type=int, default=8)
    parser.add_argument("--screen-eval-rows", type=int, default=1400)
    parser.add_argument("--final-eval-rows", type=int, default=3000)
    parser.add_argument("--screen-samples", type=int, default=24)
    parser.add_argument("--final-samples", type=int, default=64)
    parser.add_argument("--finalists", type=int, default=5)
    args = parser.parse_args(argv)
    if args.hours <= 0:
        parser.error("--hours must be positive")
    if any(lag < 1 for lag in args.lags):
        parser.error("all lags must be positive")
    if args.fixed_lag is not None and args.fixed_lag < 1:
        parser.error("--fixed-lag must be positive")
    if args.profile == "smoke":
        args.screen_epochs = min(args.screen_epochs, 2)
        args.final_epochs = min(args.final_epochs, 2)
        args.screen_eval_rows = min(args.screen_eval_rows, 256)
        args.final_eval_rows = min(args.final_eval_rows, 256)
        args.screen_samples = min(args.screen_samples, 8)
        args.final_samples = min(args.final_samples, 8)
        args.finalists = min(args.finalists, 2)
        args.lags = args.lags[:2]
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
