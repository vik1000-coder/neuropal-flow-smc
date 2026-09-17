#!/usr/bin/env python3
"""Summarize discrete peak lags in released evaluation curves.

Only lags that were actually evaluated are reported.  The script does not
interpolate between sampled time points or attach inferential confidence to a
post-hoc maximum.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "derived" / "evaluation"


def discrete_peak(
    frame: pd.DataFrame,
    label: str,
    *,
    method: str,
    metric: str,
) -> dict[str, object] | None:
    """Return the sampled point with the largest finite metric value."""
    subset = frame.loc[frame["method"] == method].dropna(subset=[metric, "time_s"])
    if subset.empty:
        return None
    row = subset.loc[subset[metric].idxmax()]
    return {
        "network": label,
        "method": method,
        "peak_lag": int(row["lag"]),
        "peak_time_s": float(row["time_s"]),
        f"peak_{metric}": float(row[metric]),
        "interpretation": "maximum among sampled lags",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", default="SBTG")
    parser.add_argument("--metric", default="f1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output or args.input_dir / "discrete_peak_lags.csv"
    rows: list[dict[str, object]] = []

    inputs = (
        (args.input_dir / "structural_reference_metrics.csv", "Structural reference"),
        (args.input_dir / "functional_reference_metrics.csv", "Functional reference"),
    )
    for path, label in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation CSV not found: {path}")
        result = discrete_peak(
            pd.read_csv(path), label, method=args.method, metric=args.metric
        )
        if result:
            rows.append(result)

    modulatory_path = args.input_dir / "modulatory_reference_metrics.csv"
    if not modulatory_path.is_file():
        raise FileNotFoundError(f"Required evaluation CSV not found: {modulatory_path}")
    modulatory = pd.read_csv(modulatory_path)
    for transmitter, group in modulatory.groupby("transmitter", sort=True):
        result = discrete_peak(
            group,
            f"Modulatory reference: {transmitter}",
            method=args.method,
            metric=args.metric,
        )
        if result:
            rows.append(result)

    if not rows:
        raise ValueError(f"No rows found for method {args.method!r}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} discrete peak summaries to {output}")


if __name__ == "__main__":
    main()
