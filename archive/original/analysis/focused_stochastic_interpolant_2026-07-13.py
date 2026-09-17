"""Frozen-anchor stochastic-interpolant comparison on G1 and G8.

This is a focused, resumable experiment rather than another broad search.  It
uses the exact train/validation/test splits and frozen Gaussian anchors from the
preregistered core run.  The only experimental factor is beta(t)=t versus
beta(t)=t^2.  Epsilon is fixed at one and 128 Euler--Maruyama steps are primary;
64/256 are numerical-resolution checks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from history_tangent_benchmark.dgps import G1LocationGaussian, G8FunctionalShapeMixture, make_dgp
from history_tangent_benchmark.finite_contrast_runner import _load_model
from history_tangent_benchmark.g8_readouts import (
    g8_quartic_central_effect,
    g8_quartic_readout,
)
from history_tangent_benchmark.metrics import energy_score_fair, tangent_nrmse
from history_tangent_benchmark.models import fit_model
from history_tangent_benchmark.reporting import load_case_records
from history_tangent_benchmark.serialization import atomic_json
from history_tangent_benchmark.stochastic_interpolant import (
    ConditionalPointSourceStochasticInterpolant,
)


CORE = Path("history_tangent_benchmark/results/stable_sid_core_20260713")
OUTPUT = Path("analysis/focused_stochastic_interpolant_2026-07-13")
GENERATORS = {"g1_gaussian_sanity", "g8_functional_shape"}
SCHEDULES = ("linear", "squared")
N_TRAIN = 8_000
N_VALIDATION = 1_600
ENERGY_CASES = 96
ENERGY_DRAWS = 64
EFFECT_ANCHORS = 8
EFFECT_DRAWS = 512
PRIMARY_STEPS = 128
RESOLUTION_STEPS = (64, 128, 256)


def _seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _pairs(dgp, n: int, history_seed: int, response_seed: int) -> tuple[np.ndarray, np.ndarray]:
    history = dgp.sample_history(n, history_seed)
    response = dgp.sample_response(history, 1, response_seed)[:, 0]
    return history.numpy(), response.numpy()


def _sample_original(model, scaler, history: torch.Tensor, n: int, seed: int, steps: int):
    standardized_history = torch.as_tensor(
        np.asarray(scaler.transform_history(history), dtype=np.float32)
    )
    with torch.no_grad():
        standardized = model.sample(
            standardized_history, n, seed=seed, steps=steps
        ).cpu().numpy()
    return torch.as_tensor(
        np.asarray(scaler.inverse_response(standardized), dtype=np.float64)
    )


def _energy_metrics(model, scaler, dgp, data_seed: int) -> dict[str, float]:
    test_h, test_y = _pairs(
        dgp, 512, data_seed + 30_001, data_seed + 30_002
    )
    history = torch.as_tensor(test_h[:ENERGY_CASES], dtype=torch.float64)
    draws = _sample_original(
        model,
        scaler,
        history,
        ENERGY_DRAWS,
        _seed(dgp.seed, data_seed, "si_energy"),
        PRIMARY_STEPS,
    ).numpy()
    return {
        "energy_score_fair": float(
            energy_score_fair(test_y[:ENERGY_CASES], draws)
        )
    }


def _g1_effect_metrics(model, scaler, dgp: G1LocationGaussian, data_seed: int) -> dict[str, float]:
    history = dgp.sample_history(
        EFFECT_ANCHORS, _seed(dgp.seed, data_seed, "si_g1_anchors")
    )
    direction = next(iter(dgp.mechanism_directions().values()))
    direction = direction / torch.linalg.vector_norm(direction)
    delta = 0.12
    plus, minus = history + delta * direction, history - delta * direction
    truth = (dgp.mean(plus) - dgp.mean(minus)) / (2.0 * delta)
    result: dict[str, float] = {}
    for steps in RESOLUTION_STEPS:
        seed = _seed(dgp.seed, data_seed, "si_g1_effect", steps)
        # Same seed supplies common Brownian increments at plus/minus.
        yp = _sample_original(model, scaler, plus, EFFECT_DRAWS, seed, steps)
        ym = _sample_original(model, scaler, minus, EFFECT_DRAWS, seed, steps)
        estimate = (yp.mean(1) - ym.mean(1)) / (2.0 * delta)
        result[f"mean_effect_nrmse_steps_{steps}"] = tangent_nrmse(
            estimate.numpy(), truth.numpy()
        )
        result[f"mean_effect_slope_steps_{steps}"] = float(
            torch.sum(estimate * truth) / torch.sum(truth.square()).clamp_min(1e-12)
        )
    return result


def _g8_effect_metrics(model, scaler, dgp: G8FunctionalShapeMixture, data_seed: int) -> dict[str, float]:
    history = dgp.sample_history(
        EFFECT_ANCHORS, _seed(dgp.seed, data_seed, "si_g8_anchors")
    )
    history[:, -1] = 0.0
    direction = torch.zeros(dgp.q, dtype=torch.float64)
    direction[-1] = 1.0
    result: dict[str, float] = {}
    for delta in (0.12, 1.0):
        truth = g8_quartic_central_effect(dgp, history, direction, delta)
        plus, minus = history + delta * direction, history - delta * direction
        for steps in RESOLUTION_STEPS:
            seed = _seed(dgp.seed, data_seed, "si_g8_effect", delta, steps)
            yp = _sample_original(model, scaler, plus, EFFECT_DRAWS, seed, steps)
            ym = _sample_original(model, scaler, minus, EFFECT_DRAWS, seed, steps)
            hp = plus[:, None, :].expand(-1, EFFECT_DRAWS, -1)
            hm = minus[:, None, :].expand(-1, EFFECT_DRAWS, -1)
            estimate = (
                g8_quartic_readout(dgp, yp, hp).mean(1)
                - g8_quartic_readout(dgp, ym, hm).mean(1)
            ) / (2.0 * delta)
            label = f"quartic_effect_delta_{delta:g}_steps_{steps}"
            result[f"{label}_nrmse"] = tangent_nrmse(
                estimate.numpy(), truth.numpy()
            )
            result[f"{label}_slope"] = float(
                torch.sum(estimate * truth)
                / torch.sum(truth.square()).clamp_min(1e-12)
            )
    return result


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases_dir = OUTPUT / "cases"
    cases_dir.mkdir(exist_ok=True)
    records = [
        record
        for record in load_case_records(CORE)
        if record["status"] == "ok"
        and record["generator"]["id"] in GENERATORS
        and record["model"]["name"] == "gaussian_nll"
    ]
    records.sort(
        key=lambda r: (
            r["generator"]["id"],
            r["generator"]["seed"],
            r["data_seed"],
            r["model"]["seed"],
        )
    )
    expected = len(records) * len(SCHEDULES)
    for anchor_record in records:
        generator = anchor_record["generator"]
        dgp = make_dgp(
            generator["kind"], seed=int(generator["seed"]), **dict(generator["params"])
        )
        anchor, scaler = _load_model(CORE, anchor_record, dgp)
        data_seed = int(anchor_record["data_seed"])
        train_h, train_y = _pairs(
            dgp, N_TRAIN, data_seed + 10_001, data_seed + 10_002
        )
        validation_h, validation_y = _pairs(
            dgp, N_VALIDATION, data_seed + 20_001, data_seed + 20_002
        )
        train_hs = np.asarray(scaler.transform_history(train_h), dtype=np.float32)
        train_ys = np.asarray(scaler.transform_response(train_y), dtype=np.float32)
        validation_hs = np.asarray(scaler.transform_history(validation_h), dtype=np.float32)
        validation_ys = np.asarray(scaler.transform_response(validation_y), dtype=np.float32)
        for schedule in SCHEDULES:
            case_id = f"{anchor_record['case_id']}_{schedule}"
            path = cases_dir / f"{case_id}.json"
            if path.exists():
                print(f"skip {case_id}", flush=True)
                continue
            print(
                f"[{len(list(cases_dir.glob('*.json'))) + 1}/{expected}] "
                f"{generator['id']} g={generator['seed']} d={data_seed} "
                f"m={anchor_record['model']['seed']} beta={schedule}",
                flush=True,
            )
            started = time.perf_counter()
            try:
                model_seed = int(anchor_record["model"]["seed"])
                torch.manual_seed(model_seed)
                model = ConditionalPointSourceStochasticInterpolant(
                    q=dgp.q,
                    dy=dgp.dy,
                    hidden=64,
                    layers=2,
                    beta_schedule=schedule,
                    epsilon=1.0,
                    sample_steps=PRIMARY_STEPS,
                    anchor=anchor,
                    freeze_anchor=True,
                )
                trace = fit_model(
                    model,
                    train_hs,
                    train_ys,
                    validation_hs,
                    validation_ys,
                    seed=model_seed,
                    device="cpu",
                    learning_rate=1e-3,
                    weight_decay=1e-4,
                    batch_size=256,
                    max_epochs=60,
                    patience=10,
                    gradient_clip=1.0,
                )
                metrics = _energy_metrics(model, scaler, dgp, data_seed)
                if isinstance(dgp, G1LocationGaussian):
                    metrics.update(_g1_effect_metrics(model, scaler, dgp, data_seed))
                elif isinstance(dgp, G8FunctionalShapeMixture):
                    metrics.update(_g8_effect_metrics(model, scaler, dgp, data_seed))
                else:
                    raise TypeError("focused experiment only supports G1/G8")
                result = {
                    "status": "ok",
                    "case_id": case_id,
                    "generator_id": generator["id"],
                    "generator_seed": int(generator["seed"]),
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "beta_schedule": schedule,
                    "epsilon": 1.0,
                    "primary_steps": PRIMARY_STEPS,
                    "fit": trace.to_dict(),
                    "metrics": metrics,
                    "wall_seconds": time.perf_counter() - started,
                }
            except Exception as error:
                result = {
                    "status": "failed",
                    "case_id": case_id,
                    "generator_id": generator["id"],
                    "generator_seed": int(generator["seed"]),
                    "data_seed": data_seed,
                    "model_seed": int(anchor_record["model"]["seed"]),
                    "beta_schedule": schedule,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "wall_seconds": time.perf_counter() - started,
                }
            atomic_json(path, result)

    rows = [json.loads(path.read_text()) for path in sorted(cases_dir.glob("*.json"))]
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key not in {"fit", "metrics"}}
        flat.update(row.get("metrics", {}))
        if row.get("fit"):
            flat["best_epoch"] = row["fit"]["best_epoch"]
            flat["fit_wall_seconds"] = row["fit"]["wall_seconds"]
        flat_rows.append(flat)
    frame = pd.DataFrame(flat_rows)
    frame.to_csv(OUTPUT / "case_metrics.csv", index=False)
    ok = frame.loc[frame.status.eq("ok")]
    numeric = [
        column
        for column in ok.columns
        if column not in {"case_id", "status", "generator_id", "beta_schedule"}
        and pd.api.types.is_numeric_dtype(ok[column])
    ]
    summary = ok.groupby(["generator_id", "beta_schedule"])[numeric].median().reset_index()
    summary.to_csv(OUTPUT / "schedule_summary.csv", index=False)
    manifest = {
        "schema_version": "1",
        "status": "complete" if len(frame) == expected and frame.status.eq("ok").all() else "complete_with_failures",
        "expected": expected,
        "observed": len(frame),
        "successful": int(frame.status.eq("ok").sum()),
        "schedules": list(SCHEDULES),
        "epsilon": 1.0,
        "primary_steps": PRIMARY_STEPS,
        "resolution_steps": list(RESOLUTION_STEPS),
        "train_size": N_TRAIN,
        "validation_size": N_VALIDATION,
        "anchor": "prefit frozen core heteroscedastic Gaussian",
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
