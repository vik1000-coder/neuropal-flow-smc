"""Developmental G8 control-design factorial using the corrected quartic readout.

The original G8 training set has one response for each continuously varying
history.  This experiment compares two randomized binary-control designs:

* unpaired_binary: distinct dynamic histories at balanced control +/-1;
* paired_rep4: the same dynamic state appears at both control levels, with four
  independent outcomes per state/control cell.

All models are evaluated on the supported finite contrast -1 versus +1.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import time

import numpy as np
import pandas as pd
import torch

from history_tangent_benchmark.dgps import G8FunctionalShapeMixture
from history_tangent_benchmark.metrics import TrainScaler
from history_tangent_benchmark.models import build_model, fit_model
from history_tangent_benchmark.serialization import atomic_json


OUTPUT = Path("analysis/g8_control_design_factorial_2026-07-13")
READOUT = runpy.run_path("analysis/g8_corrected_readout_benchmark_2026-07-13.py")
evaluate_sampler = READOUT["_evaluate_sampler"]

GENERATOR_SEEDS = (73, 79)
MODEL_SEEDS = (8101, 8107)
DESIGNS = ("unpaired_binary", "paired_rep4")
N_TRAIN = 8_000
N_VALIDATION = 1_600
EVALUATION_ANCHORS = 12
EVALUATION_DRAWS = 512
DELTA = 1.0

MODELS = (
    ("gaussian_nll", "heteroscedastic_gaussian", {"hidden": 64, "layers": 2}),
    (
        "autoregressive_mdn",
        "autoregressive_mdn",
        {"hidden": 56, "layers": 2, "components": 8},
    ),
    (
        "autoregressive_transformer",
        "autoregressive_transformer",
        {
            "d_model": 48,
            "nhead": 4,
            "transformer_layers": 2,
            "feedforward": 128,
            "components": 8,
            "dropout": 0.0,
            "permutation_seed": 0,
        },
    ),
    (
        "affine_flow",
        "conditional_affine_flow",
        {"hidden": 64, "layers": 2, "coupling_layers": 6, "max_log_scale": 1.5},
    ),
    (
        "diffusion_edm",
        "conditional_edm_diffusion",
        {
            "hidden": 64,
            "layers": 2,
            "sigma_min": 0.01,
            "sigma_max": 4.0,
            "sigma_data": 0.5,
            "sample_steps": 18,
            "rho": 7.0,
        },
    ),
    (
        "flow_matching",
        "conditional_flow_matching",
        {"hidden": 64, "layers": 2, "sample_steps": 18},
    ),
)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _dgp(seed: int) -> G8FunctionalShapeMixture:
    return G8FunctionalShapeMixture(
        seed=seed,
        q=17,
        dy=32,
        n_channels=4,
        motif_rank=4,
        motif_scale=0.45,
        noise_sd=0.08,
        correlated_noise_scale=0.06,
        shape_sensitivity=1.2,
    )


def _dataset(
    dgp: G8FunctionalShapeMixture, n: int, seed: int, design: str
) -> tuple[np.ndarray, np.ndarray]:
    if design == "unpaired_binary":
        history = dgp.sample_history(n, seed)
        sign = torch.where(
            torch.arange(n) % 2 == 0,
            -torch.ones(n, dtype=history.dtype),
            torch.ones(n, dtype=history.dtype),
        )
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        permutation = torch.randperm(n, generator=generator)
        history[:, -1] = sign[permutation]
    elif design == "paired_rep4":
        if n % 8:
            raise ValueError("paired_rep4 size must be divisible by eight")
        states = dgp.sample_history(n // 8, seed)
        states[:, -1] = 0.0
        history = states[:, None, :].expand(len(states), 8, dgp.q).clone()
        history[:, :4, -1] = -1.0
        history[:, 4:, -1] = 1.0
        history = history.reshape(n, dgp.q)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        history = history[torch.randperm(n, generator=generator)]
    else:
        raise KeyError(design)
    response = dgp.sample_response(history, 1, seed + 2)[:, 0, :]
    return history.numpy(), response.numpy()


def _run_case(
    generator_seed: int,
    model_seed: int,
    design: str,
    model_name: str,
    model_kind: str,
    params: dict[str, object],
) -> dict[str, object]:
    dgp = _dgp(generator_seed)
    train_h, train_y = _dataset(
        dgp, N_TRAIN, _stable_seed(generator_seed, design, "train"), design
    )
    validation_h, validation_y = _dataset(
        dgp,
        N_VALIDATION,
        _stable_seed(generator_seed, design, "validation"),
        design,
    )
    scaler = TrainScaler.fit(train_h, train_y)
    model = build_model(model_kind, dgp.q, dgp.dy, params)
    trace = fit_model(
        model,
        np.asarray(scaler.transform_history(train_h), dtype=np.float32),
        np.asarray(scaler.transform_response(train_y), dtype=np.float32),
        np.asarray(scaler.transform_history(validation_h), dtype=np.float32),
        np.asarray(scaler.transform_response(validation_y), dtype=np.float32),
        seed=model_seed,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=256,
        max_epochs=60,
        patience=10,
        gradient_clip=1.0,
    )
    anchors = dgp.sample_history(
        EVALUATION_ANCHORS,
        _stable_seed(generator_seed, model_seed, design, "anchors"),
    )
    anchors[:, -1] = 0.0

    def sampler(h: torch.Tensor, n: int, sample_seed: int) -> torch.Tensor:
        h_standardized = torch.as_tensor(
            np.asarray(scaler.transform_history(h), dtype=np.float32)
        )
        with torch.no_grad():
            y_standardized = model.sample(h_standardized, n, seed=sample_seed)
        return torch.as_tensor(
            np.asarray(scaler.inverse_response(y_standardized), dtype=np.float64)
        )

    evaluation = evaluate_sampler(
        dgp,
        anchors,
        sampler,
        EVALUATION_DRAWS,
        _stable_seed(generator_seed, model_seed, design, model_name, "samples"),
        delta=DELTA,
    )
    truth = np.asarray(evaluation["truth_quartic"], dtype=float)
    estimate = np.asarray(evaluation["estimated_quartic"], dtype=float)
    occupancy_truth = np.asarray(evaluation["truth_occupancy"], dtype=float)
    occupancy_estimate = np.asarray(evaluation["estimated_occupancy"], dtype=float)
    return {
        "status": "ok",
        "generator_seed": generator_seed,
        "model_seed": model_seed,
        "design": design,
        "model_name": model_name,
        "model_kind": model_kind,
        "best_epoch": trace.best_epoch,
        "stopped_epoch": trace.stopped_epoch,
        "fit_seconds": trace.wall_seconds,
        "parameter_count": trace.parameter_count,
        "quartic_calibration_slope": float(
            np.dot(estimate, truth) / max(np.dot(truth, truth), 1e-12)
        ),
        "occupancy_calibration_slope": float(
            np.dot(occupancy_estimate, occupancy_truth)
            / max(np.dot(occupancy_truth, occupancy_truth), 1e-12)
        ),
        **evaluation,
    }


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case_dir = OUTPUT / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        (generator_seed, model_seed, design, *model)
        for generator_seed in GENERATOR_SEEDS
        for model_seed in MODEL_SEEDS
        for design in DESIGNS
        for model in MODELS
    ]
    for index, specification in enumerate(specifications, start=1):
        generator_seed, model_seed, design, model_name, model_kind, params = specification
        case_id = hashlib.sha256(
            "|".join(map(str, specification[:-1])).encode()
        ).hexdigest()[:20]
        path = case_dir / f"{case_id}.json"
        if path.exists():
            print(f"[{index}/{len(specifications)}] skip {design}/{model_name}", flush=True)
            continue
        print(f"[{index}/{len(specifications)}] {design}/{model_name}", flush=True)
        started = time.perf_counter()
        try:
            result = _run_case(*specification)
        except Exception as error:
            result = {
                "status": "failed",
                "generator_seed": generator_seed,
                "model_seed": model_seed,
                "design": design,
                "model_name": model_name,
                "model_kind": model_kind,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        result["wall_seconds"] = time.perf_counter() - started
        result["case_id"] = case_id
        atomic_json(path, result)

    rows = [json.loads(path.read_text()) for path in sorted(case_dir.glob("*.json"))]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "case_metrics.csv", index=False)
    ok = frame.loc[frame.status.eq("ok")].copy()
    summary = (
        ok.groupby(["design", "model_name"], as_index=False)
        .agg(
            cases=("case_id", "size"),
            quartic_nrmse_median=("quartic_nrmse", "median"),
            quartic_nrmse_min=("quartic_nrmse", "min"),
            quartic_nrmse_max=("quartic_nrmse", "max"),
            quartic_slope_median=("quartic_calibration_slope", "median"),
            occupancy_nrmse_median=("occupancy_nrmse", "median"),
            occupancy_slope_median=("occupancy_calibration_slope", "median"),
            fit_seconds_median=("fit_seconds", "median"),
            total_seconds_median=("wall_seconds", "median"),
        )
        .sort_values(["design", "quartic_nrmse_median"])
    )
    summary.to_csv(OUTPUT / "model_summary.csv", index=False)
    manifest = {
        "schema_version": "1",
        "status": "complete" if frame.status.eq("ok").all() else "complete_with_failures",
        "expected": len(specifications),
        "observed": len(frame),
        "successful": int(frame.status.eq("ok").sum()),
        "failed": int(frame.status.eq("failed").sum()),
        "generator_seeds": list(GENERATOR_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "designs": list(DESIGNS),
        "n_train": N_TRAIN,
        "n_validation": N_VALIDATION,
        "delta": DELTA,
        "evaluation_anchors": EVALUATION_ANCHORS,
        "evaluation_draws_per_side": EVALUATION_DRAWS,
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
