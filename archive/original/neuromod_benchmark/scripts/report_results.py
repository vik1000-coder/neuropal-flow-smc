#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from neuromod_benchmark.reporting import write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(write_report(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
