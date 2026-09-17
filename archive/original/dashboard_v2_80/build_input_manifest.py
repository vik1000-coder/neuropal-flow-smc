"""Freeze the code and result receipts trusted by the live historical-80 viewer."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ATLAS = ROOT / "results/sbtg80_optimized_full_atlas_20260902"
CAMPAIGN = ROOT / "results/sbtg80_flow_optimization_20260901"

FILES = (
    HERE / "server.py",
    HERE / "index.html",
    HERE / "app.js",
    HERE / "style.css",
    HERE / "run.sh",
    HERE / "README.md",
    HERE / "PLAN.md",
    HERE / "VALIDATION.md",
    HERE / "build_portable.py",
    HERE / "build_results_report.py",
    HERE / "build_input_manifest.py",
    HERE / "test_dashboard.py",
    HERE / "__init__.py",
    ROOT / "compatibility_neural_benchmark/sbtg80_optimized_full_atlas.py",
    ROOT / "compatibility_neural_benchmark/sbtg80_full_progressive_atlas.py",
    ROOT / "compatibility_neural_benchmark/prediction_atlas_runner.py",
    ATLAS / "manifest.json",
    ATLAS / "raw_validation.json",
    ATLAS / "lag1_regression_validation.json",
    ATLAS / "seed_choice.json",
    ATLAS / "final_audit.json",
    ATLAS / "latex_report/main.tex",
    ATLAS / "latex_report/historical_80_neuron_atlas_report.pdf",
    ATLAS / "latex_report/bentley_positive_neuron_relationships.csv",
    ATLAS / "checksums.sha256",
    ATLAS / "atlas/checksums.sha256",
    ATLAS / "external_reference_checks/checksums.sha256",
    ATLAS / "external_reference_checks/bentley_metrics.csv",
    ATLAS / "external_reference_checks/bentley_lagmax_inference.csv",
    CAMPAIGN / "training/validation.json",
    CAMPAIGN / "training/winner_selection.json",
    CAMPAIGN / "training_seed2903/validation.json",
    CAMPAIGN / "final/validation.json",
    CAMPAIGN / "final/checksums.sha256",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    missing = [str(path) for path in FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze incomplete dashboard inputs: {missing}")
    payload = {
        "schema": "historical_80_neural_atlas_v2_input_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in FILES
        ],
    }
    path = HERE / "input_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
