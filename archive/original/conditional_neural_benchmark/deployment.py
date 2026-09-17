from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.data import FoldScaler, load_cohort, make_windows
from conditional_neural_benchmark.models import build_encoded_model
from history_tangent_benchmark.models import resolve_device


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def fit_deployment_ensemble(
    run_dir: Path,
    *,
    seeds: list[int],
    device: str = "auto",
    batch_size: int = 256,
) -> Path:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    board = pd.read_csv(run_dir / "leaderboard.csv")
    final = board[board.phase == "final_confirmation"].sort_values("rank")
    winner = final.iloc[0]
    model_id = str(winner["model_id"])
    config = next(c for c in manifest["model_configs"] if c["model_id"] == model_id)
    lag = int(winner["lag"])
    target_device = resolve_device(device)

    trials = pd.read_csv(run_dir / "trial_metrics.csv")
    winner_trials = trials[
        (trials.phase == "final_confirmation")
        & (trials.model_id == model_id)
        & (trials.status == "ok")
    ]
    # best_epoch is zero-indexed in the CV trace. Transfer the median selected
    # training duration to all-data fits without using the test folds again.
    epochs = int(np.median(winner_trials.best_epoch.dropna())) + 1

    cohort = load_cohort(float(manifest["coverage"]))
    scaler = FoldScaler.fit(cohort.traces)
    windows = make_windows(cohort, range(cohort.n_worms), lag, scaler)
    flat_history = windows.flat()
    target = windows.target
    if bool(config["residual_target"]):
        target = target - windows.history[:, -1, :cohort.n_neurons]

    h = torch.as_tensor(flat_history, dtype=torch.float32, device=target_device)
    y = torch.as_tensor(target, dtype=torch.float32, device=target_device)
    output_dir = run_dir / "checkpoints" / "deployment"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for seed in seeds:
        print(f"DEPLOY_START model={model_id} seed={seed} epochs={epochs}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_encoded_model(
            head_name=str(config["head"]),
            encoder_name=str(config["encoder"]),
            lag=lag,
            channels=windows.history.shape[2],
            dy=cohort.n_neurons,
            width=int(config["width"]),
            dropout=float(config["dropout"]),
            head_params=dict(config["head_params"]),
        ).to(target_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        rng = np.random.default_rng(seed)
        losses: list[float] = []
        started = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            permutation = rng.permutation(len(h))
            epoch_losses: list[float] = []
            for start in range(0, len(permutation), batch_size):
                index = torch.as_tensor(
                    permutation[start : start + batch_size],
                    dtype=torch.long,
                    device=target_device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = model.native_loss(h[index], y[index])
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite deployment loss at epoch {epoch}, seed {seed}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(epoch_losses)))
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                print(
                    f"DEPLOY_EPOCH seed={seed} epoch={epoch + 1}/{epochs} "
                    f"loss={losses[-1]:.6f}", flush=True,
                )
        wall = time.perf_counter() - started
        checkpoint = output_dir / f"{model_id}__L{lag}__all_worms__s{seed}.pt"
        torch.save(
            {
                "format_version": 1,
                "deployment": True,
                "trained_on_all_worms": True,
                "model_config": config,
                "lag": lag,
                "forecast_horizon_frames": 1,
                "seed": seed,
                "epochs": epochs,
                "neurons": list(cohort.neurons),
                "worm_ids": list(cohort.worm_ids),
                "input_channels": int(windows.history.shape[2]),
                "stimulus_channels": int(windows.history.shape[2] - cohort.n_neurons),
                "stimulus_schema_version": cohort.stimulus_schema_version,
                "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
                "stimulus_schema": cohort.stimulus_schema_dict(),
                "scaler": scaler.to_dict(),
                "train_loss": losses,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            },
            checkpoint,
        )
        summaries.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint.relative_to(run_dir)),
                "epochs": epochs,
                "final_train_loss": losses[-1],
                "wall_seconds": wall,
            }
        )
        print(f"DEPLOY_DONE seed={seed} seconds={wall:.1f} checkpoint={checkpoint}", flush=True)
        del model
        if str(target_device) == "mps":
            torch.mps.empty_cache()

    deployment_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "model_id": model_id,
        "model_config": config,
        "lag": lag,
        "lag_seconds": lag / float(manifest["fps"]),
        "forecast_horizon_frames": 1,
        "epochs_transferred_from_cv_median": epochs,
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "n_training_windows": len(windows.target),
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_schema": cohort.stimulus_schema_dict(),
        "n_dropped_windows": windows.n_dropped,
        "device": str(target_device),
        "members": summaries,
        "evaluation_warning": (
            "These all-worm checkpoints are deployment fits and have no unbiased test score. "
            "Use final_confirmation rows for performance claims."
        ),
    }
    destination = run_dir / "deployment_manifest.json"
    _atomic_json(destination, deployment_manifest)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1701, 2903, 4307])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = fit_deployment_ensemble(
        args.run_dir, seeds=args.seeds, device=args.device, batch_size=args.batch_size
    )
    print(f"DEPLOYMENT_MANIFEST {result}", flush=True)


if __name__ == "__main__":
    main()
