# Reproduction

Use Python 3.11 or newer. The publication audit records the tested package versions
in `requirements-tested.txt`; archived environment freezes document original runs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/fetch_data.py --group all --verify
python tools/audit_archive.py
python tools/run_checks.py
```

The full hash audit requires all collections restored. The test runner launches
three subprocesses so identically named old test modules cannot shadow each other.
Do not replace the current requirements with archived SBTG's old NumPy<2 constraint:
that requirements file describes a historical environment, not this assembled one.

The new figure workflow uses compact, checked inputs under `data/publication/`.
The build script regenerates these from the archived raw/aggregate results when
`--refresh-data` is requested; the default build works on a normal Git clone.
See the analysis guide for figure and notebook commands when the analysis branch
has been merged.

## Original execution records

Frozen source/data are immutable. Historical scripts may include workstation paths,
MPS defaults and guards tied to source hashes. They are preserved to describe the
actual experiments, not presented as universally portable one-command pipelines.
A new training run belongs in a new output directory with its own configuration and
manifest. Do not edit a frozen setting just to bypass a receipt mismatch.

The core modules support CPU; original fitting used MPS. Floating-point trajectories
and training outcomes need not be bit-identical across hardware. The notebook and
new publication analysis use CPU-compatible routines.

The original top-level `results/` was absent before packaging. Reports citing that
path have report-level evidence only; no restore command can reconstruct absent files.
The missing-results distinction remains visible in all original-study comparisons.
