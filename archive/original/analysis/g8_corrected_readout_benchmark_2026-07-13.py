"""Resumable G8 evaluation with the analytically correct fourth-order readout.

This does not refit any backend.  It loads the frozen core checkpoints, samples
the central h +/- delta contrast, and evaluates two G8-specific typed effects:

1. the noise-debiased quartic motif contrast with a closed-form truth;
2. the oracle Bayes frame-occupancy functional.

Linear and quadratic motif contrasts are retained as exact negative controls.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch

from history_tangent_benchmark.dgps import G8FunctionalShapeMixture, make_dgp
from history_tangent_benchmark.finite_contrast_runner import _load_model
from history_tangent_benchmark.reporting import load_case_records
from history_tangent_benchmark.serialization import atomic_json


ROOT = Path("history_tangent_benchmark/results/stable_sid_core_20260713")
OUTPUT = Path("analysis/g8_corrected_readout_benchmark_2026-07-13")
DELTA = 0.12
ANCHORS = 12
MODEL_DRAWS = 512
ORACLE_DRAWS = 2_048
EXCLUDED_MODELS = {"bounded_energy", "ratio_critic"}
LOG_2PI = math.log(2.0 * math.pi)


def _case_seed(record: dict, stream: int) -> int:
    payload = (
        f"{record['generator']['seed']}|{record['data_seed']}|"
        f"{record['model']['seed']}|{stream}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _h4(value: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    return value.pow(4) - 6.0 * variance * value.square() + 3.0 * variance.square()


def _motif_coordinates(
    dgp: G8FunctionalShapeMixture, y: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    return (y - dgp.path_mean(h)) @ dgp.motifs / dgp.motif_scale


def _quartic_feature(
    dgp: G8FunctionalShapeMixture, y: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    coordinate = _motif_coordinates(dgp, y, h)
    motif_noise = (
        dgp.motifs.T @ dgp.base_covariance @ dgp.motifs
    ) / dgp.motif_scale**2
    rotated = coordinate @ dgp.motif_rotation
    rotated_noise = dgp.motif_rotation.T @ motif_noise @ dgp.motif_rotation
    return (
        _h4(rotated, torch.diagonal(rotated_noise)).sum(dim=-1)
        - _h4(coordinate, torch.diagonal(motif_noise)).sum(dim=-1)
    )


def _frame_posterior(
    dgp: G8FunctionalShapeMixture, y: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    means = dgp.component_means(h)
    difference = y[:, None, :] - means
    whitened = torch.linalg.solve_triangular(
        dgp.base_cholesky, difference.unsqueeze(-1), upper=False
    ).squeeze(-1)
    logdet = 2.0 * torch.log(torch.diagonal(dgp.base_cholesky)).sum()
    component = -0.5 * (dgp.dy * LOG_2PI + logdet + whitened.square().sum(-1))
    posterior = torch.softmax(dgp.log_weights(h) + component, dim=-1)
    return posterior[:, dgp.components_per_frame :].sum(dim=-1)


def _nrmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(np.square(estimate - truth))
            / max(np.mean(np.square(truth)), 1e-12)
        )
    )


def _evaluate_sampler(
    dgp: G8FunctionalShapeMixture,
    anchors: torch.Tensor,
    sampler: Callable[[torch.Tensor, int, int], torch.Tensor],
    draws: int,
    seed: int,
    delta: float = DELTA,
) -> dict[str, object]:
    direction = dgp.mechanism_directions()["shape_only_control"]
    rotation_fourth_sum = float(dgp.motif_rotation.pow(4).sum())
    contrast_constant = float(
        dgp.motif_rank**2 - dgp.motif_rank * rotation_fourth_sum
    )
    truth_quartic: list[float] = []
    estimated_quartic: list[float] = []
    quartic_se: list[float] = []
    truth_occupancy: list[float] = []
    estimated_occupancy: list[float] = []
    linear_control: list[float] = []
    quadratic_control: list[float] = []
    started = time.perf_counter()
    for index, anchor in enumerate(anchors):
        h0 = anchor[None, :]
        h_plus = h0 + delta * direction
        h_minus = h0 - delta * direction
        side_seed = seed + 10_000 * (index + 1)
        y_plus = sampler(h_plus, draws, side_seed)[0].to(torch.float64)
        y_minus = sampler(h_minus, draws, side_seed)[0].to(torch.float64)
        hp = h_plus.expand(draws, -1)
        hm = h_minus.expand(draws, -1)

        phi_plus = _quartic_feature(dgp, y_plus, hp)
        phi_minus = _quartic_feature(dgp, y_minus, hm)
        estimated_quartic.append(
            float((phi_plus.mean() - phi_minus.mean()) / (2.0 * delta))
        )
        quartic_se.append(
            float(
                torch.sqrt(
                    phi_plus.var(unbiased=True) / draws
                    + phi_minus.var(unbiased=True) / draws
                )
                / (2.0 * delta)
            )
        )
        p_plus = float(dgp.shape_probability(h_plus))
        p_minus = float(dgp.shape_probability(h_minus))
        truth_quartic.append(contrast_constant * (p_plus - p_minus) / delta)

        post_plus = _frame_posterior(dgp, y_plus, hp)
        post_minus = _frame_posterior(dgp, y_minus, hm)
        estimated_occupancy.append(
            float((post_plus.mean() - post_minus.mean()) / (2.0 * delta))
        )
        truth_occupancy.append((p_plus - p_minus) / (2.0 * delta))

        u_plus = _motif_coordinates(dgp, y_plus, hp)
        u_minus = _motif_coordinates(dgp, y_minus, hm)
        linear_control.append(
            float((u_plus[:, 0].mean() - u_minus[:, 0].mean()) / (2.0 * delta))
        )
        quadratic_control.append(
            float(
                (
                    u_plus.square().sum(-1).mean()
                    - u_minus.square().sum(-1).mean()
                )
                / (2.0 * delta)
            )
        )

    truth_q = np.asarray(truth_quartic)
    estimate_q = np.asarray(estimated_quartic)
    truth_o = np.asarray(truth_occupancy)
    estimate_o = np.asarray(estimated_occupancy)
    return {
        "status": "ok",
        "delta": delta,
        "anchors": len(anchors),
        "draws_per_side": draws,
        "quartic_contrast_constant": contrast_constant,
        "quartic_nrmse": _nrmse(estimate_q, truth_q),
        "quartic_rmse": float(np.sqrt(np.mean(np.square(estimate_q - truth_q)))),
        "quartic_bias": float(np.mean(estimate_q - truth_q)),
        "quartic_truth_rms": float(np.sqrt(np.mean(np.square(truth_q)))),
        "quartic_mc_se_median": float(np.median(quartic_se)),
        "occupancy_nrmse": _nrmse(estimate_o, truth_o),
        "occupancy_rmse": float(np.sqrt(np.mean(np.square(estimate_o - truth_o)))),
        "occupancy_bias": float(np.mean(estimate_o - truth_o)),
        "linear_null_rms": float(np.sqrt(np.mean(np.square(linear_control)))),
        "quadratic_null_rms": float(np.sqrt(np.mean(np.square(quadratic_control)))),
        "seconds": time.perf_counter() - started,
        "truth_quartic": truth_quartic,
        "estimated_quartic": estimated_quartic,
        "quartic_mc_se": quartic_se,
        "truth_occupancy": truth_occupancy,
        "estimated_occupancy": estimated_occupancy,
        "linear_null": linear_control,
        "quadratic_null": quadratic_control,
    }


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case_dir = OUTPUT / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    records = [
        record
        for record in load_case_records(ROOT)
        if record["status"] == "ok"
        and record["generator"]["id"] == "g8_functional_shape"
        and record["model"]["name"] not in EXCLUDED_MODELS
        and record["model"]["capabilities"]["sampler"]
    ]
    records.sort(
        key=lambda record: (
            record["generator"]["seed"],
            record["data_seed"],
            record["model"]["seed"],
            record["model"]["name"],
        )
    )

    oracle_done: set[tuple[int, int, int]] = set()
    for index, record in enumerate(records, start=1):
        dgp = make_dgp(
            record["generator"]["kind"],
            seed=int(record["generator"]["seed"]),
            **dict(record["generator"]["params"]),
        )
        if not isinstance(dgp, G8FunctionalShapeMixture):
            raise TypeError("expected G8")
        anchor_seed = _case_seed(record, 1)
        anchors = dgp.sample_history(ANCHORS, anchor_seed)
        triple = (
            int(record["generator"]["seed"]),
            int(record["data_seed"]),
            int(record["model"]["seed"]),
        )
        if triple not in oracle_done:
            oracle_id = f"oracle_{triple[0]}_{triple[1]}_{triple[2]}"
            oracle_path = case_dir / f"{oracle_id}.json"
            if not oracle_path.exists():
                oracle = _evaluate_sampler(
                    dgp,
                    anchors,
                    lambda h, n, s: dgp.sample_response(h, n, s),
                    ORACLE_DRAWS,
                    _case_seed(record, 2),
                )
                atomic_json(
                    oracle_path,
                    {
                        "case_id": oracle_id,
                        "model_name": "oracle_predictive_law",
                        "generator_seed": triple[0],
                        "data_seed": triple[1],
                        "model_seed": triple[2],
                        **oracle,
                    },
                )
            oracle_done.add(triple)

        output_path = case_dir / f"{record['case_id']}.json"
        if output_path.exists():
            print(f"[{index}/{len(records)}] skip {record['model']['name']}", flush=True)
            continue
        model, scaler = _load_model(ROOT, record, dgp)

        def model_sampler(h: torch.Tensor, n: int, sample_seed: int) -> torch.Tensor:
            standardized_h = torch.as_tensor(
                np.asarray(scaler.transform_history(h), dtype=np.float32)
            )
            with torch.no_grad():
                standardized_y = model.sample(standardized_h, n, seed=sample_seed)
            return torch.as_tensor(
                np.asarray(scaler.inverse_response(standardized_y), dtype=np.float64)
            )

        print(f"[{index}/{len(records)}] {record['model']['name']}", flush=True)
        try:
            result = _evaluate_sampler(
                dgp, anchors, model_sampler, MODEL_DRAWS, _case_seed(record, 3)
            )
        except Exception as error:
            result = {
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        atomic_json(
            output_path,
            {
                "case_id": record["case_id"],
                "model_name": record["model"]["name"],
                "model_kind": record["model"]["kind"],
                "generator_seed": triple[0],
                "data_seed": triple[1],
                "model_seed": triple[2],
                "source_core_manifest": str(ROOT / "run_manifest.json"),
                **result,
            },
        )

    rows = [json.loads(path.read_text()) for path in sorted(case_dir.glob("*.json"))]
    for row in rows:
        if row.get("status") != "ok":
            continue
        truth_q = np.asarray(row["truth_quartic"], dtype=float)
        estimate_q = np.asarray(row["estimated_quartic"], dtype=float)
        truth_o = np.asarray(row["truth_occupancy"], dtype=float)
        estimate_o = np.asarray(row["estimated_occupancy"], dtype=float)
        row["quartic_calibration_slope"] = float(
            np.dot(estimate_q, truth_q) / max(np.dot(truth_q, truth_q), 1e-12)
        )
        row["occupancy_calibration_slope"] = float(
            np.dot(estimate_o, truth_o) / max(np.dot(truth_o, truth_o), 1e-12)
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "case_metrics.csv", index=False)
    ok = frame.loc[frame.status.eq("ok")].copy()
    summary = (
        ok.groupby("model_name", as_index=False)
        .agg(
            cases=("case_id", "size"),
            quartic_nrmse_median=("quartic_nrmse", "median"),
            quartic_nrmse_q25=("quartic_nrmse", lambda x: x.quantile(0.25)),
            quartic_nrmse_q75=("quartic_nrmse", lambda x: x.quantile(0.75)),
            occupancy_nrmse_median=("occupancy_nrmse", "median"),
            quartic_calibration_slope_median=("quartic_calibration_slope", "median"),
            occupancy_calibration_slope_median=("occupancy_calibration_slope", "median"),
            linear_null_rms_median=("linear_null_rms", "median"),
            quadratic_null_rms_median=("quadratic_null_rms", "median"),
            seconds_median=("seconds", "median"),
        )
        .sort_values("quartic_nrmse_median")
    )
    summary.to_csv(OUTPUT / "model_summary.csv", index=False)
    manifest = {
        "schema_version": "1",
        "status": "complete" if frame.status.eq("ok").all() else "complete_with_failures",
        "records": len(frame),
        "successful": int(frame.status.eq("ok").sum()),
        "failed": int(frame.status.eq("failed").sum()),
        "delta": DELTA,
        "anchors": ANCHORS,
        "model_draws_per_side": MODEL_DRAWS,
        "oracle_draws_per_side": ORACLE_DRAWS,
        "excluded_models": sorted(EXCLUDED_MODELS),
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
