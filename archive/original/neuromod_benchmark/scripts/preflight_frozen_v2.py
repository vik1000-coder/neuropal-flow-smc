#!/usr/bin/env python3
"""Validate and enumerate frozen-v2 without fitting benchmark estimators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from frozen_v2_lib import (
    DEFAULT_PREFLIGHT_PATH,
    FrozenV2Error,
    preflight,
    write_preflight_manifest,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct every frozen DGP/method grid, count cases/fits, verify "
            "resource/output invariants, and run response-truth scientific preflight."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="Config directory; defaults to configs/frozen_v2, or draft fallback",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PREFLIGHT_PATH,
        help=f"Preflight manifest (default: {DEFAULT_PREFLIGHT_PATH})",
    )
    parser.add_argument(
        "--skip-response-truth",
        action="store_true",
        help="Development-only static check; resulting manifest cannot authorize a frozen run",
    )
    parser.add_argument(
        "--ignore-existing-outputs",
        action="store_true",
        help="Development-only: do not validate existing output provenance",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    output = args.output.expanduser().resolve()
    try:
        payload = preflight(
            requested_config_dir=args.config_dir,
            run_response_truth=not args.skip_response_truth,
            validate_existing_outputs=not args.ignore_existing_outputs,
            progress=lambda message: print(message, flush=True),
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "finished_unix": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=20),
        }
        write_preflight_manifest(output, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2
    write_preflight_manifest(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

