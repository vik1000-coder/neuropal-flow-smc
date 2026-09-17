"""Measure how much frozen normalized G8 models actually use the shape control.

For held-out pairs, compare log p(y|h) with log p(y|h_perm), where only the
independent shape-control coordinate is permuted.  The oracle difference is the
available conditional information in nats; the learned difference measures how
much of that information the model uses despite fitting the marginal path law.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from history_tangent_benchmark.dgps import make_dgp
from history_tangent_benchmark.finite_contrast_runner import _load_model
from history_tangent_benchmark.reporting import load_case_records
from history_tangent_benchmark.serialization import atomic_json


ROOT = Path("history_tangent_benchmark/results/stable_sid_core_20260713")
OUTPUT = Path("analysis/g8_control_information_audit_2026-07-13")
TEST_CASES = 2_048


def _seed(record: dict, stream: str) -> int:
    text = (
        f"{record['generator']['seed']}|{record['data_seed']}|"
        f"{record['model']['seed']}|{stream}"
    ).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big")


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = [
        record
        for record in load_case_records(ROOT)
        if record["status"] == "ok"
        and record["generator"]["id"] == "g8_functional_shape"
        and record["model"]["capabilities"]["normalized_density"]
    ]
    records.sort(
        key=lambda record: (
            record["model"]["name"],
            record["generator"]["seed"],
            record["data_seed"],
            record["model"]["seed"],
        )
    )
    rows = []
    for index, record in enumerate(records, start=1):
        dgp = make_dgp(
            record["generator"]["kind"],
            seed=int(record["generator"]["seed"]),
            **dict(record["generator"]["params"]),
        )
        model, scaler = _load_model(ROOT, record, dgp)
        h = dgp.sample_history(TEST_CASES, _seed(record, "history"))
        y = dgp.sample_response(h, 1, _seed(record, "response"))[:, 0, :]
        generator = torch.Generator(device="cpu").manual_seed(
            _seed(record, "permutation")
        )
        permutation = torch.randperm(TEST_CASES, generator=generator)
        h_permuted = h.clone()
        h_permuted[:, -1] = h[permutation, -1]
        h_zero = h.clone()
        h_zero[:, -1] = 0.0

        with torch.no_grad():
            oracle_correct = dgp.log_prob(y, h)
            oracle_permuted = dgp.log_prob(y, h_permuted)
            oracle_zero = dgp.log_prob(y, h_zero)
            ys = torch.as_tensor(
                np.asarray(scaler.transform_response(y), dtype=np.float32)
            )
            hs = torch.as_tensor(
                np.asarray(scaler.transform_history(h), dtype=np.float32)
            )
            hp = torch.as_tensor(
                np.asarray(scaler.transform_history(h_permuted), dtype=np.float32)
            )
            hz = torch.as_tensor(
                np.asarray(scaler.transform_history(h_zero), dtype=np.float32)
            )
            model_correct = model.log_prob(ys, hs)
            model_permuted = model.log_prob(ys, hp)
            model_zero = model.log_prob(ys, hz)

        oracle_difference = (oracle_correct - oracle_permuted).numpy()
        model_difference = (model_correct - model_permuted).numpy()
        oracle_zero_difference = (oracle_correct - oracle_zero).numpy()
        model_zero_difference = (model_correct - model_zero).numpy()
        oracle_information = float(np.mean(oracle_difference))
        model_information = float(np.mean(model_difference))
        row = {
            "case_id": record["case_id"],
            "generator_seed": record["generator"]["seed"],
            "data_seed": record["data_seed"],
            "model_seed": record["model"]["seed"],
            "model_name": record["model"]["name"],
            "n": TEST_CASES,
            "oracle_permuted_control_information_nats": oracle_information,
            "model_permuted_control_information_nats": model_information,
            "permuted_information_recovery_fraction": (
                model_information / max(oracle_information, 1e-12)
            ),
            "oracle_zero_control_difference_nats": float(
                np.mean(oracle_zero_difference)
            ),
            "model_zero_control_difference_nats": float(
                np.mean(model_zero_difference)
            ),
            "model_permuted_difference_se": float(
                np.std(model_difference, ddof=1) / np.sqrt(TEST_CASES)
            ),
        }
        rows.append(row)
        print(
            f"[{index}/{len(records)}] {row['model_name']} "
            f"oracle={oracle_information:.4f} model={model_information:.4f}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "case_metrics.csv", index=False)
    summary = (
        frame.groupby("model_name", as_index=False)
        .agg(
            cases=("case_id", "size"),
            oracle_information_median=(
                "oracle_permuted_control_information_nats",
                "median",
            ),
            model_information_median=(
                "model_permuted_control_information_nats",
                "median",
            ),
            recovery_fraction_median=(
                "permuted_information_recovery_fraction",
                "median",
            ),
            recovery_fraction_min=(
                "permuted_information_recovery_fraction",
                "min",
            ),
            recovery_fraction_max=(
                "permuted_information_recovery_fraction",
                "max",
            ),
        )
        .sort_values("recovery_fraction_median", ascending=False)
    )
    summary.to_csv(OUTPUT / "model_summary.csv", index=False)
    manifest = {
        "schema_version": "1",
        "status": "complete",
        "records": len(frame),
        "test_cases_per_record": TEST_CASES,
        "estimand": "held-out log-density gain from the correct versus permuted shape control",
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
