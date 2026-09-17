"""Subprocess worker for optional published causal/dynamics packages.

Executed by ``.causal_venv/bin/python`` so optional compiled dependencies never
alter the main benchmark environment. Arrays cross the boundary through NPZ files;
only small fitted readouts are returned.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np


def _package_version(name: str) -> str:
    """Return installed distribution metadata, not an optional module attribute."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _ordered_groups(arrays: Any) -> list[np.ndarray]:
    groups = np.asarray(arrays["groups"], dtype=int)
    times = np.asarray(arrays["times"], dtype=int)
    result: list[np.ndarray] = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        rows = rows[np.argsort(times[rows], kind="stable")]
        if rows.size > 1 and np.any(np.diff(times[rows]) != 1):
            raise ValueError(f"group {group} is not a contiguous episode")
        result.append(rows)
    return result


def _write(path: Path, metadata: dict[str, Any], **arrays: np.ndarray) -> None:
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)


def _pcmci(source: Path, target: Path, options: dict[str, Any]) -> None:
    import tigramite
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    with np.load(source, allow_pickle=False) as data:
        values = np.asarray(data["features"], dtype=float)
        groups = _ordered_groups(data)
    trajectories = {index: values[rows] for index, rows in enumerate(groups)}
    frame = pp.DataFrame(trajectories, analysis_mode="multiple")
    model = PCMCI(
        dataframe=frame,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    tau_max = int(options.get("tau_max", 1))
    result = model.run_pcmci(
        tau_min=1,
        tau_max=tau_max,
        pc_alpha=float(options.get("pc_alpha", 0.2)),
        alpha_level=float(options.get("alpha_level", 0.05)),
        fdr_method=str(options.get("fdr_method", "fdr_bh")),
    )
    # Tigramite stores [source, target, lag]; our contract is [target, source, lag].
    val = np.transpose(np.asarray(result["val_matrix"], dtype=float), (1, 0, 2))
    p_value = np.transpose(np.asarray(result["p_matrix"], dtype=float), (1, 0, 2))
    _write(
        target,
        {
            "package": "tigramite",
            "version": _package_version("tigramite"),
            "algorithm": "PCMCI-ParCorr",
            "orientation": "[target, source, lag]",
            "analysis_mode": "multiple independent episodes",
            "assumptions": [
                "stationary time-series graph",
                "ParCorr tests linear conditional dependence",
                "causal interpretation requires causal sufficiency and valid time order",
            ],
        },
        val_matrix=val,
        p_matrix=p_value,
    )


def _var_lingam(source: Path, target: Path, options: dict[str, Any]) -> None:
    import lingam
    from lingam import VARLiNGAM

    with np.load(source, allow_pickle=False) as data:
        values = np.asarray(data["features"], dtype=float)
        groups = _ordered_groups(data)
    lag = int(options.get("lags", 1))
    lagged: list[np.ndarray] = []
    instantaneous: list[np.ndarray] = []
    for index, rows in enumerate(groups):
        if rows.size <= max(20, 4 * values.shape[1] * lag):
            raise ValueError("each VAR-LiNGAM episode needs > max(20, 4*p*lags) rows")
        model = VARLiNGAM(
            lags=lag,
            criterion=str(options.get("criterion", "bic")),
            prune=bool(options.get("prune", False)),
            random_state=int(options.get("seed", 0)) + index,
        ).fit(values[rows])
        adjacency = np.asarray(model.adjacency_matrices_, dtype=float)
        if adjacency.shape != (lag + 1, values.shape[1], values.shape[1]):
            raise RuntimeError(f"unexpected VAR-LiNGAM adjacency shape {adjacency.shape}")
        instantaneous.append(adjacency[0])
        lagged.append(adjacency[1:])
    _write(
        target,
        {
            "package": "lingam",
            "version": _package_version("lingam"),
            "algorithm": "VAR-LiNGAM",
            "orientation": "[lag, target, source]",
            "episode_aggregation": "mean of independently fitted coefficients",
            "assumptions": [
                "linear structural VAR",
                "non-Gaussian mutually independent disturbances",
                "acyclic contemporaneous structure",
                "no latent confounding for causal interpretation",
            ],
        },
        lag_tensor=np.mean(np.stack(lagged), axis=0),
        instantaneous=np.mean(np.stack(instantaneous), axis=0),
    )


def _pysindy(source: Path, target: Path, options: dict[str, Any]) -> None:
    import pysindy as ps

    with np.load(source, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=float)
        targets = np.asarray(data["targets"], dtype=float)
        groups = _ordered_groups(data)
    x = [features[rows] for rows in groups]
    x_next = [targets[rows] for rows in groups]
    optimizer = ps.STLSQ(
        threshold=float(options.get("threshold", 0.02)),
        alpha=float(options.get("alpha", 0.01)),
        max_iter=int(options.get("max_iter", 10)),
        normalize_columns=bool(options.get("normalize_columns", False)),
        unbias=bool(options.get("unbias", True)),
    )
    library = ps.PolynomialLibrary(
        degree=int(options.get("degree", 2)),
        include_bias=True,
        include_interaction=True,
    )
    model = ps.DiscreteSINDy(optimizer=optimizer, feature_library=library)
    model.fit(x, t=1.0, x_next=x_next)
    prediction = np.concatenate(
        [np.asarray(model.predict(part), dtype=float) for part in x], axis=0
    )
    ordered_targets = np.concatenate(x_next, axis=0)
    residual_variance = np.maximum(np.var(ordered_targets - prediction, axis=0), 1e-6)
    _write(
        target,
        {
            "package": "pysindy",
            "version": _package_version("pysindy"),
            "algorithm": "DiscreteSINDy-STLSQ",
            "formulation": "explicit one-step map, never future state as x_dot",
            "episode_handling": "x/x_next lists; no boundary transitions",
            "density_wrapper": "post-fit diagonal Gaussian residual law",
        },
        coefficients=np.asarray(model.coefficients(), dtype=float),
        powers=np.asarray(model.feature_library.powers_, dtype=int),
        residual_variance=residual_variance,
    )


WORKERS = {"pcmci": _pcmci, "var_lingam": _var_lingam, "pysindy": _pysindy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--action", choices=tuple(WORKERS))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args()
    if args.check:
        import lingam
        import pysindy
        import tigramite

        print(
            json.dumps(
                {
                    "lingam": _package_version("lingam"),
                    "pysindy": _package_version("pysindy"),
                    "tigramite": _package_version("tigramite"),
                },
                sort_keys=True,
            )
        )
        return
    if args.action is None or args.input is None or args.output is None:
        parser.error("--action, --input, and --output are required unless --check is used")
    WORKERS[args.action](args.input, args.output, json.loads(args.options))


if __name__ == "__main__":
    main()
