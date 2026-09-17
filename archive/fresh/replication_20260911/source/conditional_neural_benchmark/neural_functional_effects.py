from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.data import FoldScaler, make_windows
from conditional_neural_benchmark.distributional_lag_tournament import (
    cohort_folds,
    load_cohorts,
)
from conditional_neural_benchmark.models import build_encoded_model
from conditional_neural_benchmark.runner import _split_indices


PARAMETRIC_HEADS = {
    "heteroscedastic_gaussian", "lowrank_gaussian", "constrained_gaussian_dsm"
}


def _load_model(checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = build_encoded_model(
        head_name=config["head"], encoder_name=config["encoder"],
        lag=int(payload["lag"]), channels=len(payload["neurons"]) + 1,
        dy=len(payload["neurons"]), width=int(config["width"]),
        dropout=float(config["dropout"]), head_params=dict(config["head_params"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device)).eval()
    return payload, model


def _parameters(model, flat: torch.Tensor, lag: int, d: int):
    context = model.context(flat)
    head = model.head
    if head.name in {"heteroscedastic_gaussian", "constrained_gaussian_dsm"}:
        mean, log_std = head.parameters_at(context)
        total_logvar = 2.0 * log_std
        covariance_row_energy = total_logvar
    elif head.name == "lowrank_gaussian":
        mean, log_std, factor = head.parameters_at(context)
        variance = torch.exp(2.0 * log_std) + torch.square(factor).sum(dim=2)
        total_logvar = torch.log(variance.clamp_min(1e-8))
        covariance = torch.diag_embed(torch.exp(2.0 * log_std)) + factor @ factor.transpose(1, 2)
        covariance_row_energy = torch.log(torch.square(covariance).sum(dim=2).clamp_min(1e-8))
    else:
        raise ValueError(f"unsupported parametric head: {head.name}")
    # Every tournament model predicts the residual from the last observed frame.
    last = flat.reshape(len(flat), lag, d + 1)[:, -1, :d]
    return mean + last, total_logvar, covariance_row_energy


def finite_basis_effects(
    model,
    history: np.ndarray,
    *,
    lag_basis: np.ndarray,
    epsilon: float,
    device: str,
    chunk: int = 256,
) -> dict[str, np.ndarray]:
    lag, d = history.shape[1], history.shape[2] - 1
    chronological = lag_basis[::-1].astype(np.float32)
    q = chronological.shape[1]
    responses = {
        "mean": np.zeros((q, d, d), dtype=np.float64),
        "logvariance": np.zeros((q, d, d), dtype=np.float64),
        "covariance_row_energy": np.zeros((q, d, d), dtype=np.float64),
    }
    count = 0
    target_device = torch.device(device)
    with torch.no_grad():
        for start in range(0, len(history), chunk):
            block_np = history[start: start + chunk]
            block = torch.as_tensor(
                block_np.reshape(len(block_np), -1), dtype=torch.float32,
                device=target_device,
            )
            for source in range(d):
                for basis_index in range(q):
                    plus_np = block_np.copy()
                    minus_np = block_np.copy()
                    direction = epsilon * chronological[:, basis_index]
                    plus_np[:, :, source] += direction[None]
                    minus_np[:, :, source] -= direction[None]
                    plus = torch.as_tensor(
                        plus_np.reshape(len(plus_np), -1), dtype=torch.float32,
                        device=target_device,
                    )
                    minus = torch.as_tensor(
                        minus_np.reshape(len(minus_np), -1), dtype=torch.float32,
                        device=target_device,
                    )
                    plus_values = _parameters(model, plus, lag, d)
                    minus_values = _parameters(model, minus, lag, d)
                    for key, left, right in zip(
                        responses,
                        plus_values,
                        minus_values,
                    ):
                        derivative = (left - right) / (2.0 * epsilon)
                        responses[key][basis_index, :, source] += derivative.sum(dim=0).cpu().numpy()
            count += len(block_np)
    gram_inverse = np.linalg.pinv(chronological.T @ chronological)
    result = {}
    for key, response in responses.items():
        response /= max(count, 1)
        coefficient = np.einsum("pq,qts->pts", gram_inverse, response, optimize=True)
        chronological_kernel = np.einsum(
            "lp,pts->lts", chronological, coefficient, optimize=True
        )
        kernel = chronological_kernel[::-1]
        kernel[:, np.arange(d), np.arange(d)] = 0.0
        result[key] = kernel.astype(np.float32)
    return result


def _choose_parametric(run_dir: Path) -> str:
    board = pd.read_csv(run_dir / "confirmation_leaderboard.csv")
    usable = board[board["head"].isin(PARAMETRIC_HEADS)]
    if usable.empty:
        raise RuntimeError(f"no confirmed parametric finalist in {run_dir}")
    return str(usable.sort_values(["energy", "variogram"]).iloc[0].model_id)


def run(
    neural_root: Path,
    structured_root: Path,
    evidence: Path,
    output: Path,
    *,
    device: str,
    rows: int,
    epsilon: float,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    cohorts = load_cohorts()
    summaries = []
    for cohort_name, cohort in cohorts.items():
        run_dir = neural_root / cohort_name
        model_id = _choose_parametric(run_dir)
        folds = cohort_folds(cohort_name, cohort, evidence)
        fold_effects = []
        for fold in (3, 4):
            metrics = pd.read_csv(run_dir / "trial_metrics.csv")
            hit = metrics[
                (metrics.phase == "confirmation")
                & (metrics.model_id == model_id)
                & (metrics.fold == fold)
                & (metrics.seed == 1701)
                & (metrics.status == "ok")
            ]
            if len(hit) != 1:
                raise RuntimeError(f"missing frozen effect checkpoint {cohort_name}/{model_id}/f{fold}")
            checkpoint = run_dir / str(hit.iloc[0].checkpoint)
            payload, model = _load_model(checkpoint, device)
            scaler = FoldScaler(
                mean=np.asarray(payload["scaler"]["mean"], dtype=np.float32),
                scale=np.asarray(payload["scaler"]["scale"], dtype=np.float32),
            )
            _, _, test_index = _split_indices(folds, fold)
            windows = make_windows(cohort, test_index, int(payload["lag"]), scaler)
            rng = np.random.default_rng(20260828 + fold)
            index = np.arange(len(windows.target))
            if len(index) > rows:
                # Balance onset windows into the finite-difference subset.
                onset = np.flatnonzero(windows.stratum == "onset")
                remaining = np.setdiff1d(index, onset)
                take = max(0, rows - len(onset))
                index = np.sort(np.concatenate([
                    onset[:rows], rng.choice(remaining, size=min(take, len(remaining)), replace=False)
                ]))[:rows]
            with np.load(
                structured_root / "fold_models" / f"{cohort_name}__f{fold}.npz",
                allow_pickle=False,
            ) as structured:
                lag_basis = structured["lag_basis"]
            effects = finite_basis_effects(
                model, windows.history[index], lag_basis=lag_basis,
                epsilon=epsilon, device=device,
            )
            destination = output / f"{cohort_name}__{model_id}__f{fold}.npz"
            np.savez_compressed(
                destination,
                neurons=np.asarray(cohort.neurons), model_id=np.asarray(model_id),
                fold=np.asarray(fold), lag=np.asarray(int(payload["lag"])),
                rows=np.asarray(len(index)), epsilon=np.asarray(epsilon), **effects,
            )
            fold_effects.append(effects)
            summaries.append({
                "cohort": cohort_name, "model_id": model_id, "fold": fold,
                "lag": int(payload["lag"]), "rows": len(index),
                "effect_file": destination.name,
            })
            del model
            if device == "mps":
                torch.mps.empty_cache()
        averaged = {
            key: np.mean(np.stack([value[key] for value in fold_effects]), axis=0)
            for key in fold_effects[0]
        }
        np.savez_compressed(
            output / f"{cohort_name}__confirmation_mean.npz",
            neurons=np.asarray(cohort.neurons), model_id=np.asarray(model_id), **averaged,
        )
    pd.DataFrame(summaries).to_csv(output / "effect_manifest.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "method": "central finite differences along smooth lag-basis directions",
        "rows_per_fold": rows, "epsilon_standardized_activity": epsilon,
        "selection": "best confirmed parametric finalist by atlas-blind energy",
        "external_references_consulted": False,
        "claim_boundary": "local predictive distribution functional, not intervention",
    }, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural", type=Path, default=Path("results/higher_order_neural_20260828"))
    parser.add_argument("--structured", type=Path, default=Path("results/higher_order_lag_20260828"))
    parser.add_argument("--evidence", type=Path, default=Path("results/distributed_lag_dynamics_20260828"))
    parser.add_argument("--output", type=Path, default=Path("results/higher_order_neural_effects_20260828"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--wait-hours", type=float, default=7.0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_hours * 3600
    completion = args.neural / "validation.json"
    while not completion.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("neural confirmation did not complete before the effect deadline")
        time.sleep(20)
    result = run(
        args.neural.resolve(), args.structured.resolve(), args.evidence.resolve(),
        args.output.resolve(), device=args.device, rows=args.rows, epsilon=args.epsilon,
    )
    print(f"NEURAL_FUNCTIONAL_EFFECTS_COMPLETE {result}", flush=True)


if __name__ == "__main__":
    main()
