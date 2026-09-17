"""Command-line entry points for developmental benchmark execution and reporting."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

from .config import load_config
from .reporting import summarize_output


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="history-tangent-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run cases and write a developmental summary")
    run.add_argument("--config", required=True, type=Path, help="benchmark YAML path")
    run.add_argument(
        "--output",
        type=Path,
        help="optional output-directory override for the resolved frozen config",
    )
    run.add_argument("--max-cases", type=_positive_int, default=None)
    run.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="do not reuse already completed atomic case records",
    )
    run.set_defaults(resume=True, handler=_command_run)

    validate = subparsers.add_parser(
        "validate-oracles", help="run exact-oracle validation checks"
    )
    validate.add_argument("--config", type=Path, default=None)
    validate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON file or directory for oracle validation output",
    )
    validate.set_defaults(handler=_command_validate_oracles)

    summarize = subparsers.add_parser(
        "summarize", help="validate case records and rebuild tables/report"
    )
    summarize.add_argument("--output", required=True, type=Path)
    summarize.set_defaults(handler=_command_summarize)

    finite = subparsers.add_parser(
        "finite-contrast",
        help="run the revised-note central signed-contrast V1 comparison",
    )
    finite.add_argument("--config", required=True, type=Path)
    finite.set_defaults(handler=_command_finite_contrast)
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False))


def _write_json_result(destination: Path, value: Any) -> Path:
    path = destination if destination.suffix.lower() == ".json" else destination / "oracle_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _command_run(args: argparse.Namespace) -> int:
    from .runner import PACKAGE_ROOT, run_benchmark

    config = load_config(args.config)
    if args.output is not None:
        config = replace(config, output_dir=str(args.output))
        config.validate()
    run_result = run_benchmark(
        config,
        max_cases=args.max_cases,
        resume=args.resume,
    )
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = PACKAGE_ROOT / output_dir
    summary = summarize_output(output_dir)
    _print_json({"run": run_result, "summary": summary})
    return 0


def _command_validate_oracles(args: argparse.Namespace) -> int:
    from .runner import validate_oracles

    config_or_path = None if args.config is None else args.config
    result = validate_oracles(config_or_path)
    payload: dict[str, Any] = {"oracle_validation": result}
    if args.output is not None:
        path = _write_json_result(args.output, result)
        payload["output"] = str(path.resolve())
    _print_json(payload)
    return 0


def _command_summarize(args: argparse.Namespace) -> int:
    _print_json(summarize_output(args.output))
    return 0


def _command_finite_contrast(args: argparse.Namespace) -> int:
    from .finite_contrast_runner import run_finite_contrast

    result = run_finite_contrast(args.config)
    _print_json({"finite_contrast": result})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
