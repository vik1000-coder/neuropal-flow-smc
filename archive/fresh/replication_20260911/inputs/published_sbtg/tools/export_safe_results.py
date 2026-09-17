#!/usr/bin/env python3
"""Convert trusted NumPy archives to portable, pickle-free release artifacts.

The source archives are historical pipeline products and may contain Python
objects.  This utility reads only expected fields and emits numeric or Unicode
arrays plus JSON sidecars.  It also preserves the hybrid lag-1 provenance of
the manuscript result without treating its continuous lag-1 q-like values as
a Boolean significance mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LAGS = (1, 2, 3, 5, 8, 10, 15, 20)


def decode_config(value: object) -> dict:
    """Decode a scalar dictionary or JSON string from a trusted source file."""
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a hyperparameter mapping, got {type(value)!r}")
    return dict(value)


def convert_sbtg(source: Path, output: Path, config_output: Path, *, hybrid: bool) -> None:
    arrays: dict[str, np.ndarray] = {}
    configs: dict[str, dict] = {}
    with np.load(source, allow_pickle=True) as data:
        arrays["neuron_names"] = np.asarray(data["neuron_names"], dtype=str)
        arrays["lags"] = np.asarray([lag for lag in LAGS if f"mu_hat_lag{lag}" in data], dtype=np.int64)
        for lag in arrays["lags"]:
            lag = int(lag)
            arrays[f"mu_hat_lag{lag}"] = np.asarray(data[f"mu_hat_lag{lag}"], dtype=np.float64)
            if f"pval_lag{lag}" in data:
                arrays[f"p_value_lag{lag}"] = np.asarray(data[f"pval_lag{lag}"], dtype=np.float64)
            if f"sig_lag{lag}" in data:
                raw = np.asarray(data[f"sig_lag{lag}"])
                if lag == 1 and hybrid and raw.dtype.kind == "f" and not np.all(np.isin(raw, (0.0, 1.0))):
                    arrays["q_value_lag1"] = raw.astype(np.float64)
                else:
                    if not np.all(np.isin(raw, (0.0, 1.0))):
                        raise ValueError(f"Lag-{lag} significance field is not binary")
                    arrays[f"significant_lag{lag}"] = raw.astype(bool)
            key = f"hp_config_lag{lag}"
            if key in data:
                configs[str(lag)] = decode_config(data[key])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(
        json.dumps(
            {
                "lags": configs,
                "result_lineage": (
                    "hybrid: lag 1 uses the later regime-gated estimate; lags 2 and above "
                    "use the production multi-lag estimate"
                    if hybrid
                    else "single production multi-lag estimator"
                ),
                "format": "All NPZ arrays are numeric, Boolean, or Unicode and load with allow_pickle=False.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def convert_atlas(source: Path, output: Path) -> None:
    with np.load(source, allow_pickle=True) as data:
        arrays = {
            # Missing q-values remain unevaluated by mapping them to 1.0 in
            # both tests; missing amplitudes map to 0.0. This preserves the
            # positive/confirmed-negative evaluation mask without NaNs.
            "q": np.nan_to_num(
                np.asarray(data["q"], dtype=np.float64),
                nan=1.0,
                posinf=1.0,
                neginf=1.0,
            ),
            "q_eq": np.nan_to_num(
                np.asarray(data["q_eq"], dtype=np.float64),
                nan=1.0,
                posinf=1.0,
                neginf=1.0,
            ),
            "dff": np.nan_to_num(
                np.asarray(data["dff"], dtype=np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "neuron_order": np.asarray(data["neuron_order"], dtype=str),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def convert_residual(source: Path, output: Path) -> None:
    with np.load(source, allow_pickle=True) as data:
        arrays = {
            "correlation": np.asarray(data["corr"], dtype=np.float64),
            "absolute_correlation": np.asarray(data["abs_corr"], dtype=np.float64),
            "neuron_names": np.asarray(data["neuron_names"], dtype=str),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def convert_traces(source: Path, output: Path) -> None:
    """Convert a variable-length object array to flat values plus offsets."""
    with Path(source).open("rb") as handle:
        raw = np.load(handle, allow_pickle=True)
        traces = [
            np.asarray(raw[index], dtype=np.float64)
            for index in range(len(raw))
        ]
    if not traces:
        raise ValueError("Trace collection is empty")
    n_features = traces[0].shape[1]
    if any(trace.ndim != 2 or trace.shape[1] != n_features for trace in traces):
        raise ValueError("Every trace must be two-dimensional with a shared feature count")
    offsets = np.zeros(len(traces) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(trace) for trace in traces])
    values = np.concatenate(traces, axis=0)
    missing = ~np.isfinite(values)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        values=values,
        missing=missing.astype(bool),
        offsets=offsets,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="kind", required=True)
    sbtg = subparsers.add_parser("sbtg")
    sbtg.add_argument("source", type=Path)
    sbtg.add_argument("output", type=Path)
    sbtg.add_argument("config_output", type=Path)
    sbtg.add_argument("--hybrid", action="store_true")
    atlas = subparsers.add_parser("atlas")
    atlas.add_argument("source", type=Path)
    atlas.add_argument("output", type=Path)
    residual = subparsers.add_parser("residual")
    residual.add_argument("source", type=Path)
    residual.add_argument("output", type=Path)
    traces = subparsers.add_parser("traces")
    traces.add_argument("source", type=Path)
    traces.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "sbtg":
        convert_sbtg(args.source, args.output, args.config_output, hybrid=args.hybrid)
    elif args.kind == "atlas":
        convert_atlas(args.source, args.output)
    elif args.kind == "residual":
        convert_residual(args.source, args.output)
    else:
        convert_traces(args.source, args.output)


if __name__ == "__main__":
    main()
