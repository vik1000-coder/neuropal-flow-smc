#!/usr/bin/env python3
"""Run the successful frozen-v1 preflight sequentially and fail closed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frozen_v1_lib import (
    DEFAULT_PREFLIGHT_PATH,
    DEFAULT_RUN_MANIFEST_PATH,
    execute_all,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume the one-job-at-a-time frozen-v1 execution plan"
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=DEFAULT_PREFLIGHT_PATH,
        help=f"Successful preflight manifest (default: {DEFAULT_PREFLIGHT_PATH})",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_RUN_MANIFEST_PATH,
        help=f"Execution manifest (default: {DEFAULT_RUN_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="Safety/progress polling interval in [5,60] (default: 30)",
    )
    parser.add_argument(
        "--allow-draft-run",
        action="store_true",
        help="Explicitly allow draft configs (never appropriate for a frozen claim run)",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip main-suite reporting; execution summary must then use the same option",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    state = execute_all(
        preflight_path=args.preflight.expanduser().resolve(),
        state_path=args.state.expanduser().resolve(),
        heartbeat_seconds=args.heartbeat_seconds,
        allow_draft_run=args.allow_draft_run,
        generate_reports=not args.skip_reports,
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

