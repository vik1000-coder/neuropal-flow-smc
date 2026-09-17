#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from neuromod_benchmark.robustness import run_robustness_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(run_robustness_suite(args.config), indent=2))


if __name__ == "__main__":
    main()
