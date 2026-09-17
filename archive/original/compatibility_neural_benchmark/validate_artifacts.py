from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


RESPONSE_ARRAYS = (
    "response_endpoint_mean",
    "response_cumulative_mean",
    "response_peak_mean",
    "response_event_probability",
    "response_endpoint_sd",
    "diagnostic_valid",
    "diagnostic_ess_low",
    "diagnostic_ess_high",
    "diagnostic_max_weight_low",
    "diagnostic_max_weight_high",
    "diagnostic_achieved_gap",
    "rollout_energy",
    "rollout_rmse",
    "rollout_coverage90",
)
SBTG_ARRAYS = (
    "mu_hat",
    "p_mean",
    "significant_mean",
    "volatility_stat",
    "significant_volatility",
    "pearson",
)


def _validate_npz(path: Path, required: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["status"].item()) != "complete":
                errors.append(f"{path}: status is not complete")
            for key in required:
                if key not in data:
                    errors.append(f"{path}: missing {key}")
                    continue
                array = np.asarray(data[key])
                if not np.isfinite(array).all():
                    errors.append(f"{path}: {key} contains non-finite values")
    except Exception as error:  # A corrupt archive should be reported, not abort validation.
        errors.append(f"{path}: unreadable ({error!r})")
    return errors


def _manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_response_run(path: Path) -> tuple[dict, list[str]]:
    manifest = _manifest(path / "manifest.json")
    files = sorted((path / "responses").glob("*.npz"))
    expected = len(manifest["models"]) * len(manifest["folds"]) * len(manifest["seeds"])
    errors: list[str] = []
    if manifest.get("status") != "complete":
        errors.append(f"{path}: manifest status is {manifest.get('status')!r}")
    if len(files) != expected:
        errors.append(f"{path}: expected {expected} response files, found {len(files)}")
    for file in files:
        errors.extend(_validate_npz(file, RESPONSE_ARRAYS))
    return {
        "path": str(path.resolve()),
        "expected_files": expected,
        "observed_files": len(files),
        "models": manifest["models"],
        "folds": manifest["folds"],
        "seeds": manifest["seeds"],
        "repair_frames": manifest["config"]["repair_frames"],
    }, errors


def validate_sbtg_run(path: Path) -> tuple[dict, list[str]]:
    manifest = _manifest(path / "manifest.json")
    files = sorted((path / "fold_lag").glob("*.npz"))
    expected = len(manifest["folds"]) * len(manifest["lags"])
    errors: list[str] = []
    if manifest.get("status") != "complete":
        errors.append(f"{path}: manifest status is {manifest.get('status')!r}")
    if len(files) != expected:
        errors.append(f"{path}: expected {expected} SBTG files, found {len(files)}")
    for file in files:
        errors.extend(_validate_npz(file, SBTG_ARRAYS))
    return {
        "path": str(path.resolve()),
        "expected_files": expected,
        "observed_files": len(files),
        "folds": manifest["folds"],
        "lags": manifest["lags"],
        "epochs": manifest["epochs"],
    }, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(root: Path, excluded: set[Path]) -> tuple[Path, int]:
    output = root / "checksums.sha256"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.resolve() not in excluded
        and path.resolve() != output.resolve()
        and "pilot_" not in str(path.relative_to(root))
        and "smoke_" not in str(path.relative_to(root))
    )
    output.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in files)
    )
    return output, len(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-run", type=Path, required=True)
    parser.add_argument("--sbtg-run", type=Path, required=True)
    parser.add_argument("--sensitivity-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    primary, found = validate_response_run(args.response_run)
    errors.extend(found)
    sensitivities = []
    for run in args.sensitivity_runs:
        summary, found = validate_response_run(run)
        sensitivities.append(summary)
        errors.extend(found)
    sbtg, found = validate_sbtg_run(args.sbtg_run)
    errors.extend(found)
    required_analysis = (
        "REPORT.md",
        "method_metrics.csv",
        "lag_metrics.csv",
        "rollout_metrics_by_horizon.csv",
        "inference.json",
        "compatibility_by_source.csv",
        "repair_length_sensitivity.csv",
        "repair_semantic_sensitivity.csv",
        "particle_count_sensitivity.csv",
        "lag_resolved_effect_matrices.npz",
        "top_compatibility_qualified_atlas_positive_lagged_effects.csv",
    )
    missing_analysis = [name for name in required_analysis if not (args.analysis_dir / name).exists()]
    errors.extend(f"{args.analysis_dir}: missing {name}" for name in missing_analysis)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    validation_path = root / "validation.json"
    report = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "primary_response": primary,
        "sensitivity_responses": sensitivities,
        "sbtg": sbtg,
        "analysis_dir": str(args.analysis_dir.resolve()),
        "required_analysis_artifacts": list(required_analysis),
        "errors": errors,
    }
    validation_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    checksum_path, checksum_count = write_checksums(root, {validation_path.resolve()})
    report["checksum_manifest"] = str(checksum_path)
    report["checksummed_files"] = checksum_count
    validation_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
