"""Contrast-aligned endpoint experts for the G8 mixture-of-transports hypothesis.

Each endpoint receives half of a fixed 8,000-row budget. The declared control is
removed from expert inputs; at the supported endpoints a normalized gate selects
the corresponding expert. This isolates whether weak conditioning, rather than
within-endpoint generative fit, caused the attenuated G8 effect.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from history_tangent_benchmark.dgps import G8FunctionalShapeMixture
from history_tangent_benchmark.g8_readouts import g8_quartic_central_effect, g8_quartic_readout
from history_tangent_benchmark.metrics import TrainScaler, energy_score_fair, tangent_nrmse
from history_tangent_benchmark.models import build_model, fit_model
from history_tangent_benchmark.serialization import atomic_json


OUTPUT = Path("analysis/g8_endpoint_expert_decomposition_2026-07-13")
GENERATOR_SEEDS = (61, 67, 71, 73)
DATA_SEEDS = (601, 607)
MODEL_SEEDS = (6001, 6007)
MODEL_SPECS = {
    "autoregressive_mdn": (
        "autoregressive_mdn",
        {"hidden": 56, "layers": 2, "components": 8},
    ),
    "affine_flow": (
        "conditional_affine_flow",
        {"hidden": 64, "layers": 2, "coupling_layers": 6, "max_log_scale": 1.5},
    ),
    "diffusion_edm": (
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
    "flow_matching": (
        "conditional_flow_matching",
        {"hidden": 64, "layers": 2, "sample_steps": 18},
    ),
    "gaussian_nll": (
        "heteroscedastic_gaussian",
        {"hidden": 64, "layers": 2},
    ),
}
TRAIN_PER_SIDE = 4_000
VALIDATION_PER_SIDE = 1_000
ENERGY_CASES_PER_SIDE = 96
ENERGY_DRAWS = 64
EFFECT_ANCHORS = 8
EFFECT_DRAWS = 512
DELTA = 1.0


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def _endpoint_data(dgp, n, seed):
    h0 = dgp.sample_history(n, seed)
    h0[:, -1] = 0.0
    hp, hm = h0.clone(), h0.clone()
    hp[:, -1], hm[:, -1] = DELTA, -DELTA
    yp = dgp.sample_response(hp, 1, seed + 1)[:, 0].numpy()
    ym = dgp.sample_response(hm, 1, seed + 2)[:, 0].numpy()
    return h0.numpy(), hp, hm, yp, ym


def _sample(model, scaler, baseline_history, n, seed):
    hs = torch.as_tensor(
        np.asarray(scaler.transform_history(baseline_history[:, :-1]), dtype=np.float32)
    )
    with torch.no_grad():
        ys = model.sample(hs, n, seed=seed).numpy()
    return torch.as_tensor(np.asarray(scaler.inverse_response(ys), dtype=np.float64))


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = OUTPUT / "cases"
    cases.mkdir(exist_ok=True)
    expected = len(GENERATOR_SEEDS) * len(DATA_SEEDS) * len(MODEL_SEEDS) * len(MODEL_SPECS)
    for generator_seed in GENERATOR_SEEDS:
        dgp = G8FunctionalShapeMixture(
            seed=generator_seed,
            q=17,
            dy=32,
            n_channels=4,
            motif_rank=4,
            motif_scale=0.45,
            noise_sd=0.08,
            correlated_noise_scale=0.06,
            shape_sensitivity=1.2,
        )
        for data_seed in DATA_SEEDS:
            train = _endpoint_data(dgp, TRAIN_PER_SIDE, _seed(generator_seed, data_seed, "train"))
            validation = _endpoint_data(dgp, VALIDATION_PER_SIDE, _seed(generator_seed, data_seed, "validation"))
            test = _endpoint_data(dgp, ENERGY_CASES_PER_SIDE, _seed(generator_seed, data_seed, "test"))
            train_h, _, _, train_yp, train_ym = train
            val_h, _, _, val_yp, val_ym = validation
            scaler = TrainScaler.fit(
                np.concatenate([train_h[:, :-1], train_h[:, :-1]], axis=0),
                np.concatenate([train_yp, train_ym], axis=0),
            )
            train_hs = np.asarray(scaler.transform_history(train_h[:, :-1]), dtype=np.float32)
            val_hs = np.asarray(scaler.transform_history(val_h[:, :-1]), dtype=np.float32)
            for model_seed in MODEL_SEEDS:
                for model_name, (kind, params) in MODEL_SPECS.items():
                    case_id = f"g{generator_seed}_d{data_seed}_m{model_seed}_{model_name}"
                    path = cases / f"{case_id}.json"
                    if path.exists():
                        continue
                    print(f"[{len(list(cases.glob('*.json'))) + 1}/{expected}] {case_id}", flush=True)
                    started = time.perf_counter()
                    try:
                        experts = []
                        traces = []
                        for side, response in (("plus", train_yp), ("minus", train_ym)):
                            torch.manual_seed(model_seed)
                            model = build_model(kind, q=dgp.q - 1, dy=dgp.dy, params=params)
                            trace = fit_model(
                                model,
                                train_hs,
                                np.asarray(scaler.transform_response(response), dtype=np.float32),
                                val_hs,
                                np.asarray(
                                    scaler.transform_response(val_yp if side == "plus" else val_ym),
                                    dtype=np.float32,
                                ),
                                seed=model_seed,
                                device="cpu",
                                learning_rate=1e-3,
                                weight_decay=1e-4,
                                batch_size=256,
                                max_epochs=60,
                                patience=10,
                                gradient_clip=1.0,
                            )
                            experts.append(model)
                            traces.append(trace.to_dict())

                        test_h, test_hp, test_hm, test_yp, test_ym = test
                        sample_seed = _seed(case_id, "energy")
                        draw_p = _sample(experts[0], scaler, test_h, ENERGY_DRAWS, sample_seed)
                        draw_m = _sample(experts[1], scaler, test_h, ENERGY_DRAWS, sample_seed)
                        energy = energy_score_fair(
                            np.concatenate([test_yp, test_ym], axis=0),
                            np.concatenate([draw_p.numpy(), draw_m.numpy()], axis=0),
                        )

                        anchors = dgp.sample_history(EFFECT_ANCHORS, _seed(case_id, "anchors"))
                        anchors[:, -1] = 0.0
                        hp, hm = anchors.clone(), anchors.clone()
                        hp[:, -1], hm[:, -1] = DELTA, -DELTA
                        effect_seed = _seed(case_id, "effect")
                        yp = _sample(experts[0], scaler, anchors.numpy(), EFFECT_DRAWS, effect_seed)
                        ym = _sample(experts[1], scaler, anchors.numpy(), EFFECT_DRAWS, effect_seed)
                        hp_rep = hp[:, None, :].expand(-1, EFFECT_DRAWS, -1)
                        hm_rep = hm[:, None, :].expand(-1, EFFECT_DRAWS, -1)
                        estimate = (
                            g8_quartic_readout(dgp, yp, hp_rep).mean(1)
                            - g8_quartic_readout(dgp, ym, hm_rep).mean(1)
                        ) / (2.0 * DELTA)
                        direction = torch.zeros(dgp.q, dtype=torch.float64)
                        direction[-1] = 1.0
                        truth = g8_quartic_central_effect(dgp, anchors, direction, DELTA)
                        result = {
                            "status": "ok",
                            "energy_score_fair": float(energy),
                            "quartic_nrmse": tangent_nrmse(estimate.numpy(), truth.numpy()),
                            "quartic_slope": float(
                                torch.sum(estimate * truth)
                                / torch.sum(truth.square()).clamp_min(1e-12)
                            ),
                            "plus_best_epoch": traces[0]["best_epoch"],
                            "minus_best_epoch": traces[1]["best_epoch"],
                            "plus_fit_seconds": traces[0]["wall_seconds"],
                            "minus_fit_seconds": traces[1]["wall_seconds"],
                        }
                    except Exception as error:
                        result = {
                            "status": "failed",
                            "failure_type": type(error).__name__,
                            "failure_message": str(error),
                        }
                    atomic_json(
                        path,
                        {
                            "case_id": case_id,
                            "generator_seed": generator_seed,
                            "data_seed": data_seed,
                            "model_seed": model_seed,
                            "model_name": model_name,
                            "wall_seconds": time.perf_counter() - started,
                            **result,
                        },
                    )

    rows = [json.loads(path.read_text()) for path in sorted(cases.glob("*.json"))]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "case_metrics.csv", index=False)
    ok = frame[frame.status.eq("ok")]
    summary = (
        ok.groupby("model_name", as_index=False)
        .agg(
            cases=("case_id", "size"),
            energy_score_fair=("energy_score_fair", "median"),
            quartic_nrmse=("quartic_nrmse", "median"),
            quartic_nrmse_q25=("quartic_nrmse", lambda x: x.quantile(0.25)),
            quartic_nrmse_q75=("quartic_nrmse", lambda x: x.quantile(0.75)),
            quartic_slope=("quartic_slope", "median"),
            wall_seconds=("wall_seconds", "median"),
        )
        .sort_values("quartic_nrmse")
    )
    summary.to_csv(OUTPUT / "model_summary.csv", index=False)
    manifest = {
        "schema_version": "1",
        "status": "complete" if len(frame) == expected and frame.status.eq("ok").all() else "complete_with_failures",
        "expected": expected,
        "observed": len(frame),
        "successful": int(frame.status.eq("ok").sum()),
        "train_per_expert": TRAIN_PER_SIDE,
        "total_train_budget": 2 * TRAIN_PER_SIDE,
        "validation_per_expert": VALIDATION_PER_SIDE,
        "expert_input": "baseline history without declared control",
        "endpoint_gate": "deterministic normalized selection at supported controls -1/+1",
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
