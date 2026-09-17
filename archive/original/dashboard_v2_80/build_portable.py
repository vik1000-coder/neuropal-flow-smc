"""Build a self-contained, read-only historical-80 Neural Atlas directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))
from dashboard_v2_80.server import (
    AtlasData,
    ATLAS_ROOT,
    CANONICAL,
    EXTERNAL,
    ROOT,
    safe_json,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "dist/neural_atlas_v2_80_portable"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def build(output: Path, force: bool = False) -> Path:
    output = output.resolve()
    temporary = output.with_name(f".{output.name}.building")
    if output.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    if temporary.exists():
        shutil.rmtree(temporary)
    data_dir, source_dir = temporary / "data", temporary / "source"
    data_dir.mkdir(parents=True)

    # The viewer uses the exact final atlas files, with no numerical re-aggregation.
    for name in (
        "atlas_matrices.npz",
        "worm_matrices.npz",
        "support_cells.csv",
        "top_effects.csv",
        "manifest.json",
        "protocol.json",
        "validation.json",
        "checksums.sha256",
    ):
        copy(CANONICAL / name, data_dir / "atlas" / name)
    copy(ATLAS_ROOT / "seed_choice.json", data_dir / "seed_choice.json")

    dashboard_data = AtlasData()
    cohort = dashboard_data.cohort()
    lengths = np.asarray([len(trace) for trace in cohort.traces], dtype=np.int32)
    traces = np.full(
        (len(lengths), int(lengths.max()), len(cohort.neurons)),
        np.nan,
        dtype=np.float32,
    )
    for index, trace in enumerate(cohort.traces):
        traces[index, : len(trace)] = trace
    intervals = np.asarray(
        [schedule.event_intervals_seconds for schedule in cohort.stimulus_schedules],
        dtype=np.float32,
    )
    chemicals = np.asarray(
        [schedule.chemical_name_by_event for schedule in cohort.stimulus_schedules],
        dtype="U16",
    )
    np.savez_compressed(
        data_dir / "observed_cohort.npz",
        traces=traces,
        lengths=lengths,
        neurons=np.asarray(cohort.neurons),
        worm_ids=np.asarray(cohort.worm_ids),
        fps=np.asarray(cohort.fps),
        event_intervals_seconds=intervals,
        chemical_name_by_worm_event=chemicals,
    )
    write_text(
        data_dir / "external_references.json",
        json.dumps(
            safe_json(dashboard_data.references()),
            separators=(",", ":"),
            allow_nan=False,
        ),
    )

    report_root = ATLAS_ROOT / "latex_report"
    copy(report_root / "main.tex", temporary / "report" / "main.tex")
    copy(
        report_root / "historical_80_neuron_atlas_report.pdf",
        temporary / "report" / "historical_80_neuron_atlas_report.pdf",
    )
    copy(
        report_root / "bentley_positive_neuron_relationships.csv",
        temporary / "report" / "bentley_positive_neuron_relationships.csv",
    )
    for figure in sorted((report_root / "figures").glob("*.png")):
        copy(figure, temporary / "report" / "figures" / figure.name)

    for name in ("server.py", "index.html", "app.js", "style.css", "__init__.py"):
        copy(HERE / name, temporary / name)
    copy(HERE / "build_portable.py", source_dir / "build_portable.py")
    copy(HERE / "build_results_report.py", source_dir / "build_results_report.py")
    copy(HERE / "test_dashboard.py", source_dir / "test_dashboard.py")
    for relative in (
        "compatibility_neural_benchmark/sbtg80_optimized_full_atlas.py",
        "compatibility_neural_benchmark/sbtg80_full_progressive_atlas.py",
        "compatibility_neural_benchmark/prediction_atlas_runner.py",
        "compatibility_neural_benchmark/sbtg80_flow_optimization.py",
        "compatibility_neural_benchmark/sbtg80_flow_optimization_report.py",
        "compatibility_neural_benchmark/core.py",
        "compatibility_neural_benchmark/progressive_smc.py",
        "conditional_neural_benchmark/data.py",
        "SBTG/pipeline/01_prepare_data.py",
    ):
        copy(ROOT / relative, source_dir / "analysis_code" / relative)
    for name in (
        "manifest.json",
        "raw_validation.json",
        "lag1_regression_validation.json",
        "final_audit.json",
        "REPORT.md",
        "RUNBOOK.md",
        "fold_assignments.csv",
        "checksums.sha256",
    ):
        path = ATLAS_ROOT / name
        if path.exists():
            copy(path, temporary / "provenance" / "optimized_atlas" / name)
    shutil.copytree(
        EXTERNAL,
        temporary / "provenance" / "external_reference_checks",
    )

    write_text(temporary / "requirements.txt", "numpy>=1.26,<3\npandas>=2.1,<4\n")
    write_text(
        temporary / "start.py",
        """from pathlib import Path
import sys
try:
    import numpy, pandas  # noqa: F401
except ImportError:
    print("Install the viewer dependencies first: python -m pip install -r requirements.txt")
    raise SystemExit(2)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import main
main()
""",
    )
    write_text(
        temporary / "verify_bundle.py",
        """from pathlib import Path
from server import AtlasData, IS_PORTABLE
assert IS_PORTABLE
data = AtlasData()
meta = data.meta()
assert len(meta["neurons"]) == 80 and len(meta["worm_ids"]) == 20
assert meta["seed_choice"]["selected_generator_seed"] == 1701
edge = data.edge({"source": "FLP", "target": "ADE"})
assert edge["worms"].shape == (4, 20, 6)
signal = data.signals({"source": "FLP", "target": "ADE", "chemical": "nacl"})
assert len(signal["rows"]) == 20
methods = {row["method"] for row in data.references()["comparisons"]}
assert methods == {"progressive_bridge_smc", "sbtg_published", "two_seed_sensitivity"}
bentley = {"monoamine_all", "monoamine_dopamine", "monoamine_serotonin", "monoamine_tyramine", "monoamine_octopamine", "neuropeptide_all", "neuromodulator_union"}
assert bentley == set(data.references()["bentley_lagmax"])
assert (Path(__file__).parent / "report/historical_80_neuron_atlas_report.pdf").is_file()
assert (Path(__file__).parent / "provenance/external_reference_checks/bentley_metrics.csv").is_file()
print(f"Portable bundle OK: {sum(item['entries'] for item in data.verified)} checksums; 80 classes, 20 traces, uncertainty, support, Bentley neuromodulator checks and PDF report loaded")
""",
    )
    write_text(
        temporary / "run.sh",
        """#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$here"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python start.py "$@"
""",
    )
    (temporary / "run.sh").chmod(0o755)
    write_text(
        temporary / "run.bat",
        r"""@echo off
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
.venv\Scripts\python.exe start.py %*
""",
    )
    write_text(
        temporary / "README.md",
        """# Neural atlas v2 — historical 80-neuron portable viewer

This directory contains the read-only dashboard, the exact frozen atlas files, historical observed traces, reference-check data, a compiled LaTeX results report, and a source snapshot. It does not rerun a model when you use the dashboard.

## Start

Python 3.11 or newer is required. The first launcher run creates `.venv` and installs NumPy and Pandas; later runs can be offline.

- macOS/Linux: `./run.sh`
- Windows: double-click `run.bat` or run it in Command Prompt
- Existing environment: `python -m pip install -r requirements.txt`, then `python start.py`

Open http://127.0.0.1:18783 and stop with Ctrl-C. Pass `--port 18784` if the port is occupied. Run `python verify_bundle.py` for a checksum and data-loading smoke test.

The written report is `report/historical_80_neuron_atlas_report.pdf`; its editable source is `report/main.tex`. Detailed Bentley comparisons for dopamine, serotonin, tyramine, octopamine, all monoamines, all neuropeptides, and the neuromodulator union are under `provenance/external_reference_checks/` and appear in the dashboard's Reference Checks view.

## Scope

The primary atlas uses held-out-score-selected generator seed 1701 with 64 particles, repair branch factor 4, future branch factor 2, and 20 flow integration steps. The reference view also reports the stronger two-seed lag-1 sensitivity. Pointwise intervals resample 20 historical traces while holding the generator fixed.

The historical 80-class dataset pseudo-pairs head and tail recordings and donor-imputes missing traces. Effects are model-relative response sensitivities, not causal or anatomical connections. The source snapshot documents the generating code, but checkpoints, raw MAT files, and the complete training environment are not included; this package reproduces the viewer from frozen results, not model fitting.
""",
    )
    write_text(
        temporary / "REDISTRIBUTION_NOTE.md",
        """# Redistribution note

No repository-wide license or data redistribution grant was found while assembling this bundle. Before forwarding it outside the intended collaboration, confirm that the NeuroPAL-derived historical data, model outputs, and copied source files may be redistributed to the recipient.
""",
    )

    created = datetime.now(timezone.utc).isoformat()
    receipt = {
        "created_utc": created,
        "purpose": "portable historical-80 neural atlas viewer",
        "source_dashboard": str(HERE.relative_to(ROOT)),
        "source_atlas": str(ATLAS_ROOT.relative_to(ROOT)),
        "primary_generator_seed": 1701,
        "limits": "reproduces the dashboard from frozen results; does not reproduce training or resampling",
    }
    write_text(
        temporary / "provenance" / "portable_build_receipt.json",
        json.dumps(receipt, indent=2) + "\n",
    )
    files = []
    for path in sorted(temporary.rglob("*")):
        if path.is_file() and path != data_dir / "bundle_manifest.json":
            files.append(
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    manifest = {
        "schema": "neural_atlas_v2_80_portable_bundle_v1",
        "created_utc": created,
        "files": files,
    }
    write_text(data_dir / "bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    temporary.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(f"Portable dashboard: {build(args.output, args.force)}")


if __name__ == "__main__":
    main()
