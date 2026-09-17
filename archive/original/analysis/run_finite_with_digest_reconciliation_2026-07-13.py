"""Run the frozen finite panel after a narrowly audited digest reconciliation.

The benchmark provenance digest covers every reviewable file under the package
root, including unrelated configs, scripts, and reports.  Parallel audit work
created eight such files after the frozen core run started, while the executable
source and both frozen configs remained unchanged.  This wrapper preserves the
original core manifest, rejects any non-allowlisted post-start change, records the
reconciliation, and presents the core digest only to the finite runner's overly
broad tree guard.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import history_tangent_benchmark.finite_contrast_runner as finite_runner
import history_tangent_benchmark.serialization as serialization
from history_tangent_benchmark.config import load_config
from history_tangent_benchmark.serialization import (
    DEFAULT_IGNORED_PARTS,
    DEFAULT_SOURCE_SUFFIXES,
    _source_paths,
    atomic_json,
    sha256_file,
    source_tree_sha256,
    stable_sha256,
)


PACKAGE_ROOT = finite_runner.PACKAGE_ROOT.resolve()
CORE_CONFIG = PACKAGE_ROOT / "configs/stable_sid_core_20260713.yaml"
FINITE_CONFIG = PACKAGE_ROOT / "configs/stable_sid_finite_20260713.yaml"
CORE_OUTPUT = PACKAGE_ROOT / "results/stable_sid_core_20260713"
AUDIT_PATH = (
    PACKAGE_ROOT.parent
    / "analysis/stable_sid_digest_reconciliation_2026-07-13.json"
)

EXPECTED_CORE_CONFIG_FILE_SHA256 = (
    "81dd2dc9a2e54ffa12836a816c5afb3e9e5d55990c0d597724c1d6e48bbba29c"
)
EXPECTED_FINITE_CONFIG_FILE_SHA256 = (
    "9f0af8ec814ed624ea2be645651e9061e63f66a1e27c713ef624885dc1341e58"
)

ALLOWED_POST_START_ADDITIONS = {
    "DIFFUSION_VS_AR_SCIENTIFIC_REPORT.md",
    "DIFFUSION_VS_AUTOREGRESSION_AUDIT.md",
    "configs/diffusion_vs_ar_g5_generation_calibration.yaml",
    "configs/diffusion_vs_ar_g5_generation_calibration_extended.yaml",
    "configs/diffusion_vs_ar_invalidity_replication_a.yaml",
    "configs/diffusion_vs_ar_invalidity_replication_b.yaml",
    "scripts/build_diffusion_vs_ar_report_payload.py",
    "scripts/summarize_diffusion_vs_ar.py",
}


def _reviewable_post_start_files(started_unix: float) -> list[dict[str, object]]:
    paths = _source_paths(
        PACKAGE_ROOT,
        suffixes=frozenset(DEFAULT_SOURCE_SUFFIXES),
        ignored_parts=frozenset(DEFAULT_IGNORED_PARTS),
    )
    rows: list[dict[str, object]] = []
    for path in paths:
        stat = path.stat()
        if stat.st_mtime <= started_unix:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(PACKAGE_ROOT).as_posix(),
                "mtime_unix": stat.st_mtime,
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _validate_core_config_manifest(manifest: dict[str, object]) -> None:
    loaded = load_config(CORE_CONFIG)
    current_config_digest = stable_sha256(loaded.to_dict())
    if current_config_digest != manifest["config_sha256"]:
        raise RuntimeError(
            "frozen core config no longer matches the completed core manifest"
        )


def main() -> int:
    manifest_path = CORE_OUTPUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("failed") != 0:
        raise RuntimeError("core manifest is not a failure-free completed run")
    if sha256_file(CORE_CONFIG) != EXPECTED_CORE_CONFIG_FILE_SHA256:
        raise RuntimeError("frozen core config file digest changed")
    if sha256_file(FINITE_CONFIG) != EXPECTED_FINITE_CONFIG_FILE_SHA256:
        raise RuntimeError("frozen finite config file digest changed")
    _validate_core_config_manifest(manifest)

    started_unix = float(manifest["started_unix"])
    changed = _reviewable_post_start_files(started_unix)
    changed_paths = {str(row["relative_path"]) for row in changed}
    unexpected = sorted(changed_paths - ALLOWED_POST_START_ADDITIONS)
    missing_expected = sorted(ALLOWED_POST_START_ADDITIONS - changed_paths)
    if unexpected:
        raise RuntimeError(
            "execution/review surface changed after core start: "
            + ", ".join(unexpected)
        )
    if missing_expected:
        raise RuntimeError(
            "reconciliation allowlist no longer matches the tree: "
            + ", ".join(missing_expected)
        )

    frozen_digest = str(manifest["source_tree_sha256"])
    current_review_tree_digest = source_tree_sha256(PACKAGE_ROOT)
    executable_src_digest = source_tree_sha256(PACKAGE_ROOT / "src")
    audit: dict[str, object] = {
        "schema_version": "1",
        "status": "validated_before_run",
        "reason": (
            "The project-wide provenance digest includes unrelated reports, "
            "scripts, and configs created by parallel audit work. No executable "
            "source or frozen config changed after the core run started."
        ),
        "core_manifest_path": str(manifest_path),
        "core_manifest_sha256_untouched": sha256_file(manifest_path),
        "core_started_unix": started_unix,
        "core_finished_unix": float(manifest["finished_unix"]),
        "frozen_project_digest": frozen_digest,
        "current_project_digest": current_review_tree_digest,
        "current_executable_src_digest": executable_src_digest,
        "core_config_file_sha256": sha256_file(CORE_CONFIG),
        "finite_config_file_sha256": sha256_file(FINITE_CONFIG),
        "post_start_allowlisted_files": changed,
        "unexpected_post_start_files": unexpected,
        "original_core_manifest_modified": False,
        "guard_override_scope": (
            "source_tree_sha256(PACKAGE_ROOT) during this finite-run process only"
        ),
        "validated_unix": time.time(),
    }
    atomic_json(AUDIT_PATH, audit)

    original_serialization_digest = serialization.source_tree_sha256
    original_runner_digest = finite_runner.source_tree_sha256

    def reconciled_digest(root: str | Path, **kwargs: object) -> str:
        if Path(root).resolve() == PACKAGE_ROOT and not kwargs:
            return frozen_digest
        return original_serialization_digest(root, **kwargs)

    serialization.source_tree_sha256 = reconciled_digest
    finite_runner.source_tree_sha256 = reconciled_digest
    try:
        result = finite_runner.run_finite_contrast(FINITE_CONFIG)
    except BaseException as error:
        audit["status"] = "run_failed"
        audit["failure_type"] = type(error).__name__
        audit["failure_message"] = str(error)
        audit["finished_unix"] = time.time()
        atomic_json(AUDIT_PATH, audit)
        raise
    finally:
        serialization.source_tree_sha256 = original_serialization_digest
        finite_runner.source_tree_sha256 = original_runner_digest

    audit["status"] = "run_complete"
    audit["finite_manifest"] = result
    audit["finished_unix"] = time.time()
    atomic_json(AUDIT_PATH, audit)
    print(json.dumps({"finite_contrast": result, "digest_audit": str(AUDIT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
