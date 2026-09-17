#!/usr/bin/env python3
"""Consolidate read-only frozen-v1 completion/provenance checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frozen_v1_lib import (
    DEFAULT_EXECUTION_SUMMARY_PATH,
    DEFAULT_PREFLIGHT_PATH,
    DEFAULT_RUN_MANIFEST_PATH,
    build_execution_summary,
)
from neuromod_benchmark.serialization import atomic_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify every frozen-v1 manifest/result count/failure/digest"
    )
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT_PATH)
    parser.add_argument("--state", type=Path, default=DEFAULT_RUN_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXECUTION_SUMMARY_PATH)
    parser.add_argument(
        "--skip-report-checks",
        action="store_true",
        help="Do not require main-suite REPORT.md and valid validation.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    summary = build_execution_summary(
        preflight_path=args.preflight.expanduser().resolve(),
        state_path=args.state.expanduser().resolve(),
        require_reports=not args.skip_report_checks,
    )
    output = args.output.expanduser().resolve()
    atomic_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

