"""Reevaluate frozen bounded-energy checkpoints with exact iid rejection draws."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from history_tangent_benchmark.dgps import G8FunctionalShapeMixture, make_dgp
from history_tangent_benchmark.finite_contrast_runner import _load_model
from history_tangent_benchmark.g8_readouts import (
    g8_quartic_central_effect,
    g8_quartic_readout,
)
from history_tangent_benchmark.metrics import energy_score_fair, tangent_nrmse
from history_tangent_benchmark.models import BoundedEnergyRatio
from history_tangent_benchmark.reporting import load_case_records
from history_tangent_benchmark.serialization import atomic_json


CORE = Path("history_tangent_benchmark/results/stable_sid_core_20260713")
OUTPUT = Path("analysis/exact_bounded_energy_reevaluation_2026-07-13")
ENERGY_CASES = 96
ENERGY_DRAWS = 64
G8_ANCHORS = 8
G8_DRAWS = 512
G8_DELTA = 1.0


def _pairs(dgp, n, hs, ys):
    h = dgp.sample_history(n, hs)
    y = dgp.sample_response(h, 1, ys)[:, 0]
    return h, y


def _sample(model, scaler, history, n, seed):
    hs = torch.as_tensor(
        np.asarray(scaler.transform_history(history), dtype=np.float32)
    )
    started = time.perf_counter()
    with torch.no_grad():
        standardized = model.sample(hs, n, seed=seed)
    wall = time.perf_counter() - started
    draws = torch.as_tensor(
        np.asarray(scaler.inverse_response(standardized.numpy()), dtype=np.float64)
    )
    return draws, wall, model.sampling_diagnostics()


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = OUTPUT / "cases"
    cases.mkdir(exist_ok=True)
    records = [
        record
        for record in load_case_records(CORE)
        if record["status"] == "ok" and record["model"]["name"] == "bounded_energy"
    ]
    records.sort(
        key=lambda r: (
            r["generator"]["id"], r["generator"]["seed"], r["data_seed"], r["model"]["seed"]
        )
    )
    for index, record in enumerate(records, 1):
        path = cases / f"{record['case_id']}.json"
        if path.exists():
            continue
        print(f"[{index}/{len(records)}] {record['generator']['id']}", flush=True)
        started = time.perf_counter()
        generator = record["generator"]
        try:
            dgp = make_dgp(
                generator["kind"], seed=int(generator["seed"]), **dict(generator["params"])
            )
            model, scaler = _load_model(CORE, record, dgp)
            if not isinstance(model, BoundedEnergyRatio):
                raise TypeError("expected bounded energy checkpoint")
            data_seed = int(record["data_seed"])
            test_h, test_y = _pairs(dgp, 512, data_seed + 30_001, data_seed + 30_002)
            draws, sample_wall, diagnostic = _sample(
                model,
                scaler,
                test_h[:ENERGY_CASES],
                ENERGY_DRAWS,
                data_seed + int(record["model"]["seed"]) + 40_002,
            )
            result = {
                "status": "ok",
                "energy_score_fair_exact": float(
                    energy_score_fair(test_y[:ENERGY_CASES].numpy(), draws.numpy())
                ),
                "samples_per_second_exact": ENERGY_CASES * ENERGY_DRAWS / sample_wall,
                **{f"rejection_{key}": value for key, value in diagnostic.items()},
            }
            if isinstance(dgp, G8FunctionalShapeMixture):
                history = dgp.sample_history(G8_ANCHORS, data_seed + 90_001)
                history[:, -1] = 0.0
                direction = torch.zeros(dgp.q, dtype=torch.float64)
                direction[-1] = 1.0
                plus = history + G8_DELTA * direction
                minus = history - G8_DELTA * direction
                seed = data_seed + int(record["model"]["seed"]) + 90_002
                yp, _, plus_diag = _sample(model, scaler, plus, G8_DRAWS, seed)
                ym, _, minus_diag = _sample(model, scaler, minus, G8_DRAWS, seed + 1)
                hp = plus[:, None, :].expand(-1, G8_DRAWS, -1)
                hm = minus[:, None, :].expand(-1, G8_DRAWS, -1)
                estimate = (
                    g8_quartic_readout(dgp, yp, hp).mean(1)
                    - g8_quartic_readout(dgp, ym, hm).mean(1)
                ) / (2.0 * G8_DELTA)
                truth = g8_quartic_central_effect(dgp, history, direction, G8_DELTA)
                result.update(
                    {
                        "g8_quartic_nrmse": tangent_nrmse(estimate.numpy(), truth.numpy()),
                        "g8_quartic_slope": float(
                            torch.sum(estimate * truth)
                            / torch.sum(truth.square()).clamp_min(1e-12)
                        ),
                        "g8_plus_acceptance": plus_diag["acceptance_rate"],
                        "g8_minus_acceptance": minus_diag["acceptance_rate"],
                    }
                )
        except Exception as error:
            result = {
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        atomic_json(
            path,
            {
                "case_id": record["case_id"],
                "generator_id": generator["id"],
                "generator_seed": int(generator["seed"]),
                "data_seed": int(record["data_seed"]),
                "model_seed": int(record["model"]["seed"]),
                "wall_seconds": time.perf_counter() - started,
                **result,
            },
        )

    rows = [json.loads(path.read_text()) for path in sorted(cases.glob("*.json"))]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "case_metrics.csv", index=False)
    ok = frame[frame.status.eq("ok")]
    summary = (
        ok.groupby("generator_id", as_index=False)
        .agg(
            cases=("case_id", "size"),
            energy_score_fair_exact=("energy_score_fair_exact", "median"),
            acceptance_rate=("rejection_acceptance_rate", "median"),
            samples_per_second=("samples_per_second_exact", "median"),
            g8_quartic_nrmse=("g8_quartic_nrmse", "median"),
            g8_quartic_slope=("g8_quartic_slope", "median"),
        )
    )
    summary.to_csv(OUTPUT / "generator_summary.csv", index=False)
    manifest = {
        "schema_version": "1",
        "status": "complete" if len(frame) == len(records) and frame.status.eq("ok").all() else "complete_with_failures",
        "expected": len(records),
        "observed": len(frame),
        "successful": int(frame.status.eq("ok").sum()),
        "sampler": "exact iid rejection from bounded Gaussian reference tilt",
        "energy_cases": ENERGY_CASES,
        "energy_draws": ENERGY_DRAWS,
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
