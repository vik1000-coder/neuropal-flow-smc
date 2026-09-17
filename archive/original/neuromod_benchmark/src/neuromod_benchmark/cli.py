from __future__ import annotations

import argparse
import json

from .runner import run_suite


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the neuromodulator recovery benchmark")
    parser.add_argument("suite", help="suite YAML")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    manifest = run_suite(args.suite, max_cases=args.max_cases, resume=not args.no_resume)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
