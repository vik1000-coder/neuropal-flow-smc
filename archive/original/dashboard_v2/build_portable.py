"""Build a self-contained, presentation-only Neural Atlas v2 directory.

The bundle carries the exact arrays used by the dashboard and a source snapshot.
It deliberately does not carry training checkpoints or the full research repository.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))
from dashboard_v2.server import AtlasData, ATLAS, CANONICAL, EVIDENCE, NULLS, ROOT, SAMPLES


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "dist/neural_atlas_v2_portable"


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
    data_dir = temporary / "data"
    source_dir = temporary / "source"
    provenance = temporary / "provenance"
    data_dir.mkdir(parents=True)

    atlas_keys = ["neurons", "worm_ids", "source_lag_frames", "horizon_frames", "orientation"]
    for method in ("progressive_bridge_smc", "direct_importance"):
        for channel in ("endpoint_mean", "cumulative_mean", "peak_mean", "event_probability", "endpoint_sd", "endpoint_log_sd", "endpoint_wasserstein1"):
            for context in ("baseline", "onset", "active", "offset", "recovery", "state_average", "onset_minus_baseline", "butanone_onset", "pentanedione_onset", "nacl_onset", "butanone_onset_minus_baseline", "pentanedione_onset_minus_baseline", "nacl_onset_minus_baseline"):
                atlas_keys.extend(f"{prefix}__{method}__{channel}__{context}" for prefix in ("mean_normalized", "ci_low_normalized", "ci_high_normalized"))
        for context in ("baseline", "onset", "active", "offset", "recovery", "state_average", "onset_minus_baseline", "butanone_onset", "pentanedione_onset", "nacl_onset", "butanone_onset_minus_baseline", "pentanedione_onset_minus_baseline", "nacl_onset_minus_baseline"):
            atlas_keys.append(f"valid_fraction__{method}__{context}")
            if method == "progressive_bridge_smc":
                atlas_keys.append(f"genealogy_valid_fraction_0_10__{method}__{context}")
    with np.load(CANONICAL / "atlas_matrices.npz", allow_pickle=False) as archive:
        missing = sorted(set(atlas_keys) - set(archive.files))
        if missing:
            raise RuntimeError(f"Canonical atlas lacks portable keys: {missing[:3]}")
        np.savez_compressed(data_dir / "atlas_matrices.npz", **{key: archive[key] for key in atlas_keys})

    with np.load(CANONICAL / "worm_matrices.npz", allow_pickle=False) as archive:
        keys = sorted(key for key in archive.files if key.startswith("normalized__"))
        np.savez_compressed(data_dir / "worm_matrices.npz", **{key: archive[key] for key in keys})

    support_columns = ["method", "context", "source_index", "source_lag_frames", "support_qualified", "genealogy_strong_gate_pass", "valid_fraction", "genealogy_valid_fraction_0_10"]
    pd.read_parquet(CANONICAL / "support_cells.parquet", columns=support_columns).to_csv(data_dir / "support_cells.csv", index=False)
    cell_columns = ["source_index", "target_index", "channel", "context", "source_lag_frames", "horizon_frames", "support_eligible", "joint_primary_cell_max_t_p_value", "joint_primary_lag_contrast_max_t_p_value", "evidence_label"]
    pd.read_parquet(EVIDENCE / "cell_evidence.parquet", columns=cell_columns).to_csv(data_dir / "cell_evidence.csv", index=False)
    edge_columns = ["source_index", "target_index", "channel", "context", "support_eligible", "joint_primary_flat_lag_max_t_p_value", "evidence_label"]
    pd.read_csv(EVIDENCE / "edge_evidence.csv", usecols=edge_columns).to_csv(data_dir / "edge_evidence.csv", index=False)
    for name in ("sham_calibration.csv", "null_inference_arrays.npz"):
        copy(NULLS / name, data_dir / name)
    shutil.copytree(SAMPLES / "responses", data_dir / "responses")

    dashboard_data = AtlasData()
    cohort = dashboard_data.cohort()
    lengths = np.asarray([len(trace) for trace in cohort.traces], dtype=np.int32)
    traces = np.full((len(lengths), int(lengths.max()), len(cohort.neurons)), np.nan, dtype=np.float32)
    for index, trace in enumerate(cohort.traces):
        traces[index, :len(trace)] = trace
    intervals = np.asarray([schedule.event_intervals_seconds for schedule in cohort.stimulus_schedules], dtype=np.float32)
    chemicals = np.asarray([schedule.chemical_name_by_event for schedule in cohort.stimulus_schedules], dtype="U16")
    np.savez_compressed(data_dir / "observed_cohort.npz", traces=traces, lengths=lengths, neurons=np.asarray(cohort.neurons), worm_ids=np.asarray(cohort.worm_ids), fps=np.asarray(cohort.fps), event_intervals_seconds=intervals, chemical_name_by_worm_event=chemicals)
    scalers = [dashboard_data.scaler(fold) for fold in range(5)]
    np.savez_compressed(data_dir / "fold_scalers.npz", mean=np.stack([value.mean for value in scalers]), scale=np.stack([value.scale for value in scalers]))
    (data_dir / "external_references.json").write_text(json.dumps(dashboard_data.references(), separators=(",", ":"), allow_nan=False))

    for name in ("server.py", "index.html", "app.js", "style.css", "__init__.py"):
        copy(HERE / name, temporary / name)
    copy(HERE / "build_portable.py", source_dir / "build_portable.py")
    for relative in (
        "conditional_neural_benchmark/data.py",
        "compatibility_neural_benchmark/core.py",
        "compatibility_neural_benchmark/progressive_smc.py",
        "compatibility_neural_benchmark/prediction_atlas_runner.py",
        "compatibility_neural_benchmark/targeted_sampling_nulls.py",
        "compatibility_neural_benchmark/targeted_sampling_null_analysis.py",
        "compatibility_neural_benchmark/full_family_inference.py",
        "sid_elegans/combined_data.py",
        "SBTG/pipeline/01_prepare_data.py",
    ):
        copy(ROOT / relative, source_dir / "analysis_code" / relative)
    for label, directory in (("canonical", CANONICAL), ("complete_family_evidence", EVIDENCE), ("sampling_null_analysis", NULLS), ("sampling_null_raw", SAMPLES)):
        target = provenance / label
        for name in ("manifest.json", "protocol.json", "validation.json", "checksums.sha256"):
            if (directory / name).exists():
                copy(directory / name, target / name)

    write_text(temporary / "requirements.txt", "numpy>=1.26,<3\npandas>=2.1,<4\n")
    write_text(temporary / "start.py", """from pathlib import Path\nimport sys\ntry:\n    import numpy, pandas  # noqa: F401\nexcept ImportError:\n    print(\"Install the two viewer dependencies first: python -m pip install -r requirements.txt\")\n    raise SystemExit(2)\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nfrom server import main\nmain()\n""")
    write_text(temporary / "verify_bundle.py", """from server import AtlasData, IS_PORTABLE\nassert IS_PORTABLE\ndata = AtlasData()\nmeta = data.meta()\nassert len(meta[\"neurons\"]) == 54 and len(meta[\"worm_ids\"]) == 17\nedge = data.edge({\"source\": \"FLP\", \"target\": \"ADE\"})\nassert edge[\"worms\"].shape == (4, 17, 6)\nsignal = data.signals({\"source\": \"FLP\", \"target\": \"ADE\", \"chemical\": \"nacl\"})\nassert len(signal[\"rows\"]) == 17\nprediction = data.prediction({\"source\": \"FLP\", \"target\": \"ADE\", \"lag\": \"1\", \"horizon\": \"8\", \"worm\": \"0\", \"chemical\": \"butanone\"})\nassert prediction[\"available\"] and len(prediction[\"low\"]) == 256\nassert len(data.references()[\"comparisons\"]) == 210\nprint(f\"Portable bundle OK: {sum(item['entries'] for item in data.verified)} checksums, 54 neurons, 17 worms, signals, controls, samples and references loaded\")\n""")
    write_text(temporary / "run.sh", """#!/bin/sh\nset -eu\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\ncd \"$here\"\nif [ ! -x .venv/bin/python ]; then\n  python3 -m venv .venv\n  .venv/bin/python -m pip install -r requirements.txt\nfi\nexec .venv/bin/python start.py \"$@\"\n""")
    (temporary / "run.sh").chmod(0o755)
    write_text(temporary / "run.bat", """@echo off\ncd /d %~dp0\nif not exist .venv\\Scripts\\python.exe (\n  py -3 -m venv .venv\n  .venv\\Scripts\\python.exe -m pip install -r requirements.txt\n)\n.venv\\Scripts\\python.exe start.py %*\n""")
    write_text(temporary / "README.md", """# Neural atlas v2 — portable viewer

This directory contains the complete read-only dashboard, presentation-ready frozen data, and a source snapshot. It does not need the original repository and does not rerun a model.

## Start

Python 3.11 or newer is required. The first launcher run creates `.venv` and installs NumPy and Pandas from PyPI; later runs are offline.

- macOS/Linux: `./run.sh`
- Windows: double-click `run.bat` or run it in Command Prompt
- Existing compatible environment: `python -m pip install -r requirements.txt`, then `python start.py`

Open http://127.0.0.1:18781 and stop with Ctrl-C. Pass `--port 18782` if that port is occupied. The server binds only to the local machine.

Run `python verify_bundle.py` in the installed environment for a checksum and data-loading smoke test.

## What is included

- Exact dashboard HTML, CSS, JavaScript and Python server
- All 54 × 54 atlas estimates and saved pointwise intervals for both estimators, all outcomes, contexts, lags and horizons
- Progressive-SMC individual-worm arrays, exact support/evidence rows, N128 control arrays and all 70 saved response banks
- Preprocessed observed cohort, saved fold scalers, and reviewed external-reference comparisons
- `source/build_portable.py`, a snapshot of the analysis files named by the result provenance, and original result manifests/checksum ledgers
- `data/bundle_manifest.json`, verified automatically at startup

The source snapshot documents the code lineage used by this presentation. Training checkpoints, the full research repository and raw MAT recordings are not included, so this package reproduces the dashboard, not model training. See `provenance/` for the archived scientific scope and `REDISTRIBUTION_NOTE.md` before forwarding data outside the intended collaboration.
""")
    write_text(temporary / "REDISTRIBUTION_NOTE.md", """# Redistribution note

No repository-wide license or data redistribution grant was found while assembling this bundle. The package was created locally at the user's request. Before sending it outside the existing collaboration, confirm that the NeuroPAL-derived data, model outputs, and copied source files may be redistributed to the intended recipient. This is a rights/provenance warning, not a technical limitation.
""")
    write_text(source_dir / "README.md", """The live viewer code is at the package root. `build_portable.py` is the exact extraction script used to create this directory. `analysis_code/` contains the specific analysis files named by the saved result provenance plus the cohort preparation path needed during extraction. It is a source snapshot for inspection, not a standalone training package; checkpoints and the complete research environment are intentionally absent.\n""")

    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "portable presentation-only neural atlas viewer",
        "source_dashboard": str(HERE.relative_to(ROOT)),
        "source_atlas": str(ATLAS.relative_to(ROOT)),
        "limits": "reproduces the dashboard from frozen results; does not reproduce training or resampling",
    }
    write_text(provenance / "portable_build_receipt.json", json.dumps(receipt, indent=2) + "\n")
    files = []
    for path in sorted(temporary.rglob("*")):
        if path.is_file() and path != data_dir / "bundle_manifest.json":
            files.append({"path": path.relative_to(temporary).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"schema": "neural_atlas_v2_portable_bundle_v1", "created_utc": receipt["created_utc"], "files": files}
    write_text(data_dir / "bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    temporary.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = build(args.output, args.force)
    print(f"Portable dashboard: {output}")


if __name__ == "__main__":
    main()
