#!/usr/bin/env python3
"""Run the frozen post-hoc fixed-amplitude learning-curve diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

import pandas as pd

from empirical_sid.dgps import make_static_dgp
from empirical_sid.runner import (
    atomic_csv,
    atomic_json,
    atomic_npz,
    load_config,
    run_static_case,
    sha256_file,
    source_tree_hash,
)


SEEDS = tuple(range(4001, 4031))
MECHANISMS = ("m3_bounded", "m4_quartic", "m5_occupancy")
SAMPLE_SIZES = (2000, 8000, 32000)
MIN_FREE_DISK_GB = 2.0
MAX_RSS_GB = 8.0
MAX_OUTPUT_GB = 0.12


def enforce(output: Path) -> None:
    free = shutil.disk_usage(output.parent).free / 1024**3
    if free < MIN_FREE_DISK_GB:
        raise RuntimeError(f"disk safety stop: {free:.3f} GiB")
    if output.exists():
        size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if size / 1024**3 > MAX_OUTPUT_GB:
            raise RuntimeError("fixed-amplitude output cap exceeded")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    if rss > MAX_RSS_GB:
        raise RuntimeError("fixed-amplitude RSS cap exceeded")


def calibration(mechanism: str, delta: float) -> dict[str, object]:
    n_reference = 8000
    target_lambda = 3.0
    pilot = make_static_dgp(mechanism, amplitude=0.5)
    coefficient = pilot.information(0.0) / 0.5**2
    target_information = target_lambda / (n_reference * delta**2)
    raw_amplitude = (target_information / max(coefficient, 1e-12)) ** 0.5
    amplitude = float(min(0.85, max(0.15, raw_amplitude)))
    dgp = make_static_dgp(mechanism, amplitude=amplitude)
    information = dgp.information(0.0)
    return {
        "mechanism": mechanism,
        "reference_n": n_reference,
        "reference_target_lambda": target_lambda,
        "raw_amplitude": raw_amplitude,
        "amplitude": amplitude,
        "amplitude_bound_active": bool(amplitude in {0.15, 0.85}),
        "information": information,
        "achieved_lambda_n_2000": 2000 * delta**2 * information,
        "achieved_lambda_n_8000": 8000 * delta**2 * information,
        "achieved_lambda_n_32000": 32000 * delta**2 * information,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--upstream-package", required=True, type=Path)
    args = parser.parse_args()
    config, _ = load_config(args.base_config)
    output = args.output.resolve()
    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    delta = float(config["registered_scales"]["primary_delta"])
    calibrated = {mechanism: calibration(mechanism, delta) for mechanism in MECHANISMS}
    atomic_csv(output / "calibration.csv", pd.DataFrame(calibrated.values()))
    manifest = {
        "study_id": "sid_tier2_fixed_amplitude_diagnostic_20260714",
        "status": "running",
        "classification": "posthoc_exploratory_diagnostic",
        "started_unix": time.time(),
        "seeds": list(SEEDS),
        "mechanisms": list(MECHANISMS),
        "sample_sizes": list(SAMPLE_SIZES),
        "expected_cases": len(SEEDS) * len(MECHANISMS) * len(SAMPLE_SIZES),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_tree_sha256": source_tree_hash(),
        "base_config_sha256": sha256_file(args.base_config.resolve()),
        "plan_sha256": sha256_file(args.plan.resolve()),
        "upstream_package_sha256": sha256_file(args.upstream_package.resolve()),
        "resource_limits": {
            "threads": 1,
            "max_rss_gb": MAX_RSS_GB,
            "max_output_gb": MAX_OUTPUT_GB,
            "min_free_disk_gb": MIN_FREE_DISK_GB,
        },
    }
    atomic_json(output / "manifest.json", manifest)
    completed = 0
    failures = 0
    for n_train in SAMPLE_SIZES:
        for mechanism in MECHANISMS:
            choice = calibrated[mechanism]
            cell = f"fixed_amplitude_n_{n_train}__{mechanism}"
            cell_config = dict(config)
            cell_config["study_id"] = manifest["study_id"]
            cell_config["data"] = dict(config["data"])
            cell_config["data"]["n_train"] = n_train
            achieved_lambda = n_train * delta**2 * float(choice["information"])
            for seed in SEEDS:
                enforce(output)
                csv_path = cases_dir / f"{cell}__seed_{seed}.csv"
                npz_path = cases_dir / f"{cell}__seed_{seed}.npz"
                if csv_path.exists() and npz_path.exists():
                    completed += 1
                    continue
                try:
                    frame, arrays = run_static_case(
                        cell_config,
                        "posthoc_diagnostic",
                        seed,
                        mechanism,
                        float(choice["amplitude"]),
                    )
                    atomic_npz(npz_path, arrays)
                    frame["cell_id"] = cell
                    frame["slice_name"] = f"fixed_amplitude_n_{n_train}"
                    frame["target_lambda"] = float("nan")
                    frame["achieved_lambda"] = achieved_lambda
                    frame["amplitude"] = choice["amplitude"]
                    frame["amplitude_bound_active"] = choice["amplitude_bound_active"]
                    frame["array_uri"] = str(npz_path.relative_to(output))
                    frame["array_sha256"] = sha256_file(npz_path)
                    atomic_csv(csv_path, frame)
                    completed += 1
                except Exception as exc:
                    failures += 1
                    atomic_csv(
                        csv_path,
                        pd.DataFrame(
                            [{
                                "study_id": manifest["study_id"],
                                "phase": "posthoc_diagnostic",
                                "cell_id": cell,
                                "dgp_family": mechanism,
                                "dgp_seed": seed,
                                "status": "failed",
                                "failure_code": type(exc).__name__,
                                "failure_message": str(exc),
                            }]
                        ),
                    )
                    if failures / (completed + failures) > 0.10:
                        raise
    manifest.update(
        {
            "status": "complete",
            "completed_cases": completed,
            "failed_cases": failures,
            "finished_unix": time.time(),
        }
    )
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    raise SystemExit(main())
