"""Seal the completed audit only after training, scoring, and replay pass."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from conditional_neural_benchmark.chemical_encoding_runner import sha256
from conditional_neural_benchmark.distribution_structure_runner import DEFAULT_RUN, ROOT, atomic_json


def finalize(run_dir: Path):
    required = ["training_validation.json", "analysis/validation.json",
                "analysis/numerical_replay.json", "synthetic_controls/validation.json"]
    checks = {}
    for name in required:
        checks[name] = json.loads((run_dir / name).read_text())
        if checks[name]["status"] != "pass":
            raise RuntimeError(f"validation did not pass: {name}")
    if checks["training_validation.json"]["completed"] != 30:
        raise RuntimeError("expected 30 fitted alternatives")
    if checks["analysis/validation.json"]["archives"] != 80 or not checks["analysis/validation.json"]["particle_sensitivity_complete"]:
        raise RuntimeError("primary or particle-sensitivity grid incomplete")
    if checks["analysis/numerical_replay.json"]["score_replays"] != 40:
        raise RuntimeError("checkpoint replay incomplete")
    tests = (run_dir / "analysis/unit_test_results.txt").read_text()
    match = re.search(r"(\d+) passed, (\d+) warning.* in ([\d.]+)s", tests)
    if not match or int(match.group(1)) < 289 or re.search(r"\d+ (failed|errors?)", tests):
        raise RuntimeError("final 289-test regression evidence is missing or failed")

    protocol = json.loads((run_dir / "protocol.json").read_text())
    unhashed = {k: v for k, v in protocol.items() if k not in ("fingerprint", "frozen_utc")}
    if hashlib.sha256(json.dumps(unhashed, sort_keys=True).encode()).hexdigest() != protocol["fingerprint"]:
        raise RuntimeError("protocol fingerprint mismatch")
    expected_sources = dict(protocol["training_source_sha256"])
    for stage in ("evaluation", "particle_sensitivity"):
        manifest = json.loads((run_dir / stage / "manifest.json").read_text())
        settings = {k: v for k, v in manifest.items() if k != "fingerprint"}
        if hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest() != manifest["fingerprint"]:
            raise RuntimeError(f"manifest fingerprint mismatch: {stage}")
        for name, digest in manifest["source"].items():
            expected_sources[f"conditional_neural_benchmark/{name}"] = digest
    for name, digest in expected_sources.items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"frozen source changed: {name}")
    if sha256(Path(protocol["fold_file"])) != protocol["fold_sha256"]:
        raise RuntimeError("frozen fold file changed")
    checkpoints = list(protocol["frozen_flow_checkpoints"])
    checkpoints += [dict(item, path=str(run_dir / item["path"]))
                    for item in checks["training_validation.json"]["checkpoints"]]
    for item in checkpoints:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"checkpoint changed: {item['path']}")

    sources = set(expected_sources)
    sources.update(str(p.relative_to(ROOT)) for p in (ROOT / "conditional_neural_benchmark").glob("*.py"))
    sources.update(["conditional_neural_benchmark/tests/test_distribution_structure.py",
                    "sid_elegans/combined_data.py", "SBTG/pipeline/01_prepare_data.py"])
    source_records = []
    for name in sorted(sources):
        destination = run_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        source_records.append(dict(path=name, sha256=sha256(destination),
                                   frozen_training_source=name in protocol["training_source_sha256"],
                                   frozen_evaluation_source=name in expected_sources and name not in protocol["training_source_sha256"]))
    atomic_json(run_dir / "source_inventory.json", {"files": source_records,
                "meaning": "Core and audit source snapshot; rerunning also requires this repository and the original data."})
    input_files = ["SBTG/data/Head_Activity_OH16230.mat", "SBTG/data/Head_Activity_OH15500.mat",
                   "SBTG/results/intermediate/connectome/nodes.json"]
    atomic_json(run_dir / "input_inventory.json", {
        "recorded_at": "finalization; these extra raw-file hashes were not part of the original protocol freeze",
        "note": "The inherited cohort loader uses Cook node names for its neuron vocabulary, not atlas edge weights. OH15500 affects the inherited observation-quality roster; only OH16230 worms enter these fits and scores.",
        "files": [dict(path=name, sha256=sha256(ROOT / name)) for name in input_files],
        "checkpoints": checkpoints})
    atomic_json(run_dir / "VALIDATION.json", {
        "status": "pass", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_fingerprint": protocol["fingerprint"], "trained_checkpoints": 30,
        "reused_flow_checkpoints": 10, "primary_archives": 80, "particle_sensitivity_archives": 20,
        "regression_tests": int(match.group(1)), "known_law_controls": 4,
        "score_replays": 40, "independent_density_checks": 30,
        "max_score_replay_error": checks["analysis/numerical_replay.json"]["max_score_error"],
        "max_dense_logpdf_error": checks["analysis/numerical_replay.json"]["max_dense_logpdf_error"],
        "max_marginal_invariance_error": checks["analysis/validation.json"]["max_marginal_invariance_error"],
        "frozen_sources_and_inputs_unchanged": True,
        "scope": "Numerical and protocol validation, not confirmation of biological mechanisms."})
    # Seal all finished artifacts except the ledger itself. Smoke output remains
    # available but is explicitly outside the scientific artifact inventory.
    files = sorted(p for p in run_dir.rglob("*") if p.is_file()
                   and "evaluation_smoke" not in p.parts and p.name != "SHA256SUMS"
                   and "__pycache__" not in p.parts and p.suffix != ".tmp")
    ledger = "".join(f"{sha256(p)}  {p.relative_to(run_dir)}\n" for p in files)
    (run_dir / "SHA256SUMS").write_text(ledger)
    print(f"SEALED {len(files)} files; {match.group(1)} tests; 100 evaluation archives")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    finalize(args.run_dir.resolve())


if __name__ == "__main__":
    main()
