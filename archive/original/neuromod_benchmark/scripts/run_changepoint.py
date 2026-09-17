#!/usr/bin/env python3
"""Run the isolated matched-null changepoint benchmark from a JSON config.

Example
-------
python neuromod_benchmark/scripts/run_changepoint.py \
  --config neuromod_benchmark/configs/changepoint_medium.json \
  --output neuromod_benchmark/results/changepoint_medium.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


# Keep BLAS-backed ridge solves polite on a shared workstation.  The benchmark is
# parallel across top-level series in principle, but this reference runner deliberately
# uses one process so peak memory and thermals remain predictable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from neuromod_benchmark.changepoint import (  # noqa: E402
    PostChangeStrengths,
    run_changepoint_benchmark,
)
from neuromod_benchmark.mechanistic import MechanisticConfig  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched-null neuromodulator changepoint evaluation with independent "
            "calibration and final-null pools."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "changepoint_medium.json",
        help="JSON configuration (default: configs/changepoint_medium.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "results" / "changepoint_medium.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, help="Override benchmark seed")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-event rows while retaining calibration and aggregate summaries",
    )
    return parser.parse_args()


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _finite_json(value.item())
    return value


def _load_config(path: Path) -> tuple[MechanisticConfig, dict[str, Any], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"name", "mechanistic", "benchmark"}:
        raise ValueError("config requires exactly name, mechanistic, and benchmark sections")
    mechanism = MechanisticConfig(**raw["mechanistic"])
    benchmark = dict(raw["benchmark"])
    strengths = {
        scenario: PostChangeStrengths(**settings)
        for scenario, settings in benchmark.pop("post_change_strengths").items()
    }
    benchmark["post_change_strengths"] = strengths
    if "tail_thresholds" in benchmark:
        benchmark["tail_thresholds"] = tuple(benchmark["tail_thresholds"])
    if "methods" in benchmark:
        benchmark["methods"] = tuple(benchmark["methods"])
    return mechanism, benchmark, raw


def _report_payload(
    report: Any,
    *,
    raw_config: dict[str, Any],
    elapsed_seconds: float,
    summary_only: bool,
) -> dict[str, Any]:
    def calibration_payload(calibration: Any) -> dict[str, Any]:
        fpr_low, fpr_high = calibration.calibration_fpr_interval(confidence=0.95)
        return {
            "alpha": calibration.alpha,
            "threshold": calibration.threshold,
            "minimum_p_value": calibration.minimum_p_value,
            "attainable_null_call_rate": calibration.attainable_null_call_rate,
            "calibration_threshold_fpr_ci_low": fpr_low,
            "calibration_threshold_fpr_ci_high": fpr_high,
            "n_calibration": int(calibration.null_max_scores.size),
            "null_max_scores": calibration.null_max_scores.tolist(),
            "calibration_ids": calibration.calibration_ids,
            "calibration_seeds": calibration.calibration_seeds,
            "match_key": calibration.match_key,
            "scan_options": dict(calibration.scan_options),
        }

    calibrations = {
        method: calibration_payload(calibration)
        for method, calibration in report.calibrations.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": raw_config,
        "elapsed_seconds": elapsed_seconds,
        "metadata": dict(report.metadata),
        "calibrations": calibrations,
        "detection_summary": list(report.detection_summary),
        "attribution_summary": list(report.attribution_summary),
    }
    if not summary_only:
        payload["detections"] = [asdict(item) for item in report.detections]
        payload["attributions"] = [asdict(item) for item in report.attributions]
    return _finite_json(payload)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'scenario':<13} {'method':<25} {'call/FPR':>9} "
        f"{'95% CI':>15} {'hit':>7} {'det+hit':>8} {'med|delay|':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        interval = f"[{row['call_rate_ci_low']:.2f},{row['call_rate_ci_high']:.2f}]"
        delay = row["mean_absolute_localization_error"]
        delay_text = "-" if delay is None or not math.isfinite(delay) else f"{delay:.1f}"
        hit = row["hit_rate"]
        detected = row["detected_within_rate"]
        hit_text = "-" if hit is None or not math.isfinite(hit) else f"{hit:.2f}"
        detected_text = (
            "-" if detected is None or not math.isfinite(detected) else f"{detected:.2f}"
        )
        print(
            f"{row['scenario']:<13} {row['method']:<25} {row['call_rate']:>9.2f} "
            f"{interval:>15} {hit_text:>7} {detected_text:>8} {delay_text:>11}"
        )


def main() -> int:
    args = _arguments()
    mechanism, benchmark, raw = _load_config(args.config)
    if args.seed is not None:
        benchmark["seed"] = args.seed
        raw["benchmark"]["seed"] = args.seed
    start = time.perf_counter()
    report = run_changepoint_benchmark(mechanism, **benchmark)
    elapsed = time.perf_counter() - start
    payload = _report_payload(
        report,
        raw_config=raw,
        elapsed_seconds=elapsed,
        summary_only=args.summary_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    _print_summary(payload["detection_summary"])
    print(f"\nWrote {args.output} in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
