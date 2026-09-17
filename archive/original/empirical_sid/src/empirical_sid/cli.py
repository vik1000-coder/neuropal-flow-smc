from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .runner import (
    calibrate_information,
    finalize_manifest,
    initialize_manifest,
    load_config,
    resolve_output,
    run_dynamic_phase,
    run_stage0,
    run_static_phase,
    update_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gated empirical SID validation")
    parser.add_argument(
        "command",
        choices=["stage0", "calibrate", "development", "confirmatory", "dynamic", "static-all"],
    )
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, config_path = load_config(args.config)
    output = resolve_output(config)
    initialize_manifest(config, config_path, output)
    if args.command == "stage0":
        payload = run_stage0(config, output)
        update_manifest(output, "stage0", payload)
    elif args.command == "calibrate":
        payload = calibrate_information(config, output)
        update_manifest(output, "information_calibration", payload)
    elif args.command in {"development", "confirmatory"}:
        payload = run_static_phase(config, output, args.command)
        update_manifest(output, f"static_{args.command}", payload)
    elif args.command == "dynamic":
        payload = run_dynamic_phase(config, output)
        update_manifest(output, "dynamic_confirmatory", payload)
    else:
        stage0 = run_stage0(config, output)
        update_manifest(output, "stage0", stage0)
        if stage0["status"] != "pass":
            finalize_manifest(output, "gate0_failed")
            print(json.dumps(stage0, indent=2))
            return 2
        calibration = calibrate_information(config, output)
        update_manifest(output, "information_calibration", calibration)
        development = run_static_phase(config, output, "development")
        update_manifest(output, "static_development", development)
        confirmatory = run_static_phase(config, output, "confirmatory")
        update_manifest(output, "static_confirmatory", confirmatory)
        finalize_manifest(output, "static_complete")
        payload = {"stage0": stage0, "development": development, "confirmatory": confirmatory}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
