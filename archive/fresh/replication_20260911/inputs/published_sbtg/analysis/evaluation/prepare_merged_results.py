#!/usr/bin/env python3
"""Evaluate released lag matrices against independent reference networks.

The default inputs are the pickle-free artifacts distributed with this
repository.  Optional baseline archives may be supplied explicitly or placed
in ``results/paper/baselines``.  Every matrix follows the convention
``matrix[target, source]``.

Examples
--------
Run the released SBTG evaluation::

    python analysis/evaluation/prepare_merged_results.py

Include a baseline archive::

    python analysis/evaluation/prepare_merged_results.py \
        --baseline "VAR=path/to/var_lag_matrices.npz"

Baseline archives can contain either ``mu_hat_lagN`` arrays (the method name
comes from ``--baseline`` or the file stem) or named arrays such as
``Pearson_lagN`` and ``VAR_lagN``.  Archives must contain only numeric,
Boolean, or Unicode arrays and load with ``allow_pickle=False``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SBTG = PROJECT_ROOT / "results" / "paper" / "sbtg_lag_matrices.npz"
DEFAULT_BASELINES = PROJECT_ROOT / "results" / "paper" / "baselines"
DEFAULT_CONNECTOME = PROJECT_ROOT / "reference_data" / "connectome"
DEFAULT_FUNCTIONAL_ATLAS = (
    PROJECT_ROOT
    / "reference_data"
    / "functional_atlas"
    / "aligned_atlas_wild_type.npz"
)
DEFAULT_MODULATORY_EDGES = (
    PROJECT_ROOT
    / "reference_data"
    / "modulatory_atlas"
    / "edge_lists"
    / "edgelist_MA_classes.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "derived" / "evaluation"


@dataclass(frozen=True)
class MatrixSeries:
    """Lag-indexed matrices with their neuron order."""

    method: str
    neuron_names: tuple[str, ...]
    matrices: dict[int, np.ndarray]
    significance: dict[int, np.ndarray]


def _require_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _validate_square(matrix: np.ndarray, n_nodes: int, label: str) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.shape != (n_nodes, n_nodes):
        raise ValueError(
            f"{label} has shape {matrix.shape}; expected {(n_nodes, n_nodes)}"
        )
    if matrix.dtype.kind not in "biufc":
        raise TypeError(f"{label} must be numeric or Boolean, not {matrix.dtype}")
    return matrix


def _node_order(archive: np.lib.npyio.NpzFile, fallback: Iterable[str] | None = None) -> tuple[str, ...]:
    for key in ("neuron_names", "neuron_order", "node_order", "neurons"):
        if key in archive:
            names = tuple(str(value) for value in np.asarray(archive[key]).tolist())
            if len(names) != len(set(names)):
                raise ValueError(f"{key} contains duplicate neuron names")
            return names
    if fallback is None:
        raise ValueError("Archive has no neuron_names/neuron_order/node_order array")
    return tuple(str(value) for value in fallback)


def load_sbtg_archive(path: Path, q_alpha: float = 0.2) -> MatrixSeries:
    """Load a release-format SBTG archive without enabling pickle."""
    path = _require_file(path, "SBTG archive")
    with np.load(path, allow_pickle=False) as archive:
        names = _node_order(archive)
        matrices: dict[int, np.ndarray] = {}
        significance: dict[int, np.ndarray] = {}
        for key in archive.files:
            match = re.fullmatch(r"mu_hat_lag(\d+)", key)
            if match:
                lag = int(match.group(1))
                matrices[lag] = _validate_square(archive[key], len(names), key).astype(float)

        if not matrices:
            raise ValueError(f"No mu_hat_lagN matrices found in {path}")

        for lag in matrices:
            # Prefer explicit binary masks.  Explicit q-value fields are
            # thresholded; arbitrary continuous arrays are never coerced to
            # Boolean by their nonzero status.
            for key in (f"significant_lag{lag}", f"sig_lag{lag}"):
                if key not in archive:
                    continue
                raw = _validate_square(archive[key], len(names), key)
                if raw.dtype != np.bool_ and not np.all(np.isin(raw, (0, 1))):
                    raise ValueError(f"{key} is not a binary significance mask")
                significance[lag] = raw.astype(bool)
                break
            else:
                q_key = f"q_value_lag{lag}"
                if q_key in archive:
                    q_values = _validate_square(archive[q_key], len(names), q_key).astype(float)
                    finite = q_values[np.isfinite(q_values)]
                    if finite.size and (finite.min() < 0 or finite.max() > 1):
                        raise ValueError(f"{q_key} contains values outside [0, 1]")
                    significance[lag] = np.isfinite(q_values) & (q_values < q_alpha)

    return MatrixSeries("SBTG", names, dict(sorted(matrices.items())), significance)


def _load_baseline_archive(
    path: Path,
    requested_name: str | None,
    fallback_names: Iterable[str],
) -> list[MatrixSeries]:
    """Load one or more baseline series from a safe NPZ archive."""
    path = _require_file(path, "baseline archive")
    grouped: dict[str, dict[int, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as archive:
        names = _node_order(archive, fallback_names)
        for key in archive.files:
            match = re.fullmatch(r"(.+)_lag(\d+)", key)
            if not match:
                continue
            prefix, lag_text = match.groups()
            if prefix.lower() in {
                "p_value", "pval", "q_value", "significant", "sig", "p"
            }:
                continue
            value = np.asarray(archive[key])
            if value.ndim != 2:
                continue
            method = requested_name if prefix in {"mu_hat", "matrix", "weights"} else prefix
            if requested_name and prefix not in {"mu_hat", "matrix", "weights"}:
                if prefix.casefold() != requested_name.casefold():
                    continue
                method = requested_name
            grouped.setdefault(method, {})[int(lag_text)] = _validate_square(
                value, len(names), key
            ).astype(float)

    if not grouped:
        raise ValueError(f"No baseline lag matrices found in {path}")
    return [
        MatrixSeries(method, names, dict(sorted(matrices.items())), {})
        for method, matrices in sorted(grouped.items())
    ]


def parse_baseline_spec(spec: str) -> tuple[str, Path]:
    """Parse ``METHOD=PATH`` without placing paths in result metadata."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError("baseline must use METHOD=PATH syntax")
    method, path_text = spec.split("=", 1)
    if not method.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("baseline must use METHOD=PATH syntax")
    return method.strip(), Path(path_text).expanduser()


def load_baselines(
    specs: Iterable[tuple[str, Path]],
    directory: Path | None,
    fallback_names: Iterable[str],
) -> list[MatrixSeries]:
    """Load explicitly named baselines and any NPZ files in a directory."""
    series: list[MatrixSeries] = []
    seen_paths: set[Path] = set()
    for method, path in specs:
        resolved = path.resolve()
        seen_paths.add(resolved)
        series.extend(_load_baseline_archive(path, method, fallback_names))

    if directory and directory.is_dir():
        for path in sorted(directory.glob("*.npz")):
            if path.resolve() in seen_paths:
                continue
            series.extend(_load_baseline_archive(path, None, fallback_names))

    merged: dict[str, MatrixSeries] = {}
    for item in series:
        key = item.method.casefold()
        if key not in merged:
            merged[key] = item
            continue
        previous = merged[key]
        if previous.neuron_names != item.neuron_names:
            raise ValueError(f"Baseline {item.method!r} uses inconsistent neuron orders")
        overlap = set(previous.matrices) & set(item.matrices)
        if overlap:
            raise ValueError(f"Baseline {item.method!r} repeats lags {sorted(overlap)}")
        merged[key] = MatrixSeries(
            previous.method,
            previous.neuron_names,
            dict(sorted({**previous.matrices, **item.matrices}.items())),
            {},
        )
    return sorted(merged.values(), key=lambda item: item.method.casefold())


def _normalized_index(names: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, name in enumerate(names):
        normalized = str(name).strip().upper()
        if normalized in result:
            raise ValueError(f"Duplicate normalized neuron name: {normalized}")
        result[normalized] = index
    return result


def align_pair(
    prediction: np.ndarray,
    prediction_names: Iterable[str],
    reference: np.ndarray,
    reference_names: Iterable[str],
    *extra_reference_arrays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], tuple[str, ...]]:
    """Align prediction and reference arrays to the same sorted node order."""
    pred_names = tuple(prediction_names)
    ref_names = tuple(reference_names)
    prediction = _validate_square(prediction, len(pred_names), "prediction")
    reference = _validate_square(reference, len(ref_names), "reference")
    pred_index = _normalized_index(pred_names)
    ref_index = _normalized_index(ref_names)
    common = tuple(sorted(set(pred_index) & set(ref_index)))
    if not common:
        raise ValueError("Prediction and reference have no common neuron names")
    pred_ix = [pred_index[name] for name in common]
    ref_ix = [ref_index[name] for name in common]
    pred_aligned = prediction[np.ix_(pred_ix, pred_ix)]
    ref_aligned = reference[np.ix_(ref_ix, ref_ix)]
    extras = []
    for position, array in enumerate(extra_reference_arrays, start=1):
        array = _validate_square(array, len(ref_names), f"reference array {position}")
        extras.append(array[np.ix_(ref_ix, ref_ix)])
    return pred_aligned, ref_aligned, extras, common


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the maximum F1 over sampled score thresholds."""
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(np.nanmax(values))


def evaluate_scores(
    scores: np.ndarray,
    reference: np.ndarray,
    *,
    evaluation_mask: np.ndarray | None = None,
    weight_reference: np.ndarray | None = None,
    significance_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute ranking metrics and a descriptive best-threshold F1 score."""
    if scores.shape != reference.shape or scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores and reference must be square matrices with equal shape")
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    if evaluation_mask is not None:
        if evaluation_mask.shape != scores.shape:
            raise ValueError("evaluation_mask shape does not match scores")
        mask &= evaluation_mask.astype(bool)
    mask &= np.isfinite(scores) & np.isfinite(reference)

    y_true = (reference[mask] != 0).astype(int)
    y_score = np.abs(scores[mask])
    finite = np.isfinite(y_score)
    y_true, y_score = y_true[finite], y_score[finite]
    if not len(y_true) or np.unique(y_true).size < 2:
        auroc = auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))

    target = reference if weight_reference is None else weight_reference
    weight_mask = mask & np.isfinite(target)
    if significance_mask is not None:
        if significance_mask.shape != scores.shape:
            raise ValueError("significance_mask shape does not match scores")
        weight_mask &= significance_mask.astype(bool)
    if np.count_nonzero(weight_mask) >= 3:
        rho = spearmanr(scores[weight_mask], target[weight_mask]).statistic
        spearman = float(rho) if np.isfinite(rho) else float("nan")
    else:
        spearman = float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "spearman": spearman,
        "f1": best_f1(y_true, y_score),
        "n_evaluated": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)) if len(y_true) else float("nan"),
    }


def _load_structural_references(
    directory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    directory = Path(directory)
    chemical = np.load(
        _require_file(directory / "A_chem.npy", "chemical matrix"),
        allow_pickle=False,
    )
    gap = np.load(
        _require_file(directory / "A_gap.npy", "gap-junction matrix"),
        allow_pickle=False,
    )
    with _require_file(directory / "nodes.json", "connectome node order").open(
        encoding="utf-8"
    ) as handle:
        names = tuple(str(value) for value in json.load(handle))
    chemical = _validate_square(chemical, len(names), "chemical reference")
    gap = _validate_square(gap, len(names), "gap-junction reference")
    combined = _validate_square(chemical + gap, len(names), "structural reference")
    return combined, chemical, gap, names


def _load_functional_reference(
    path: Path, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    path = _require_file(path, "functional atlas")
    with np.load(path, allow_pickle=False) as archive:
        names = _node_order(archive)
        q = _validate_square(archive["q"], len(names), "q")
        q_eq = _validate_square(archive["q_eq"], len(names), "q_eq")
        amplitude = _validate_square(archive["dff"], len(names), "dff")
    diagonal = np.eye(len(names), dtype=bool)
    positive = (q < alpha) & ~diagonal
    negative = (q_eq < alpha) & ~diagonal & ~positive
    return positive.astype(float), positive | negative, amplitude, names


def _reference_name(name: object) -> str:
    """Normalize an already class-level reference name without merging it."""
    return str(name).strip().upper()


def _load_modulatory_references(
    path: Path, neuron_names: Iterable[str]
) -> dict[str, np.ndarray]:
    path = _require_file(path, "modulatory edge list")
    frame = pd.read_csv(
        path,
        header=None,
        names=["source", "target", "transmitter", "receptor"],
        dtype=str,
    )
    names = tuple(str(value).strip().upper() for value in neuron_names)
    index = {name: position for position, name in enumerate(names)}
    transmitters = ("dopamine", "serotonin", "tyramine", "octopamine")
    references = {name: np.zeros((len(names), len(names)), dtype=float) for name in transmitters}
    for row in frame.itertuples(index=False):
        transmitter = str(row.transmitter).strip().lower()
        if transmitter not in references:
            continue
        source = _reference_name(row.source)
        target = _reference_name(row.target)
        if source in index and target in index and source != target:
            references[transmitter][index[target], index[source]] = 1.0
    return references


def _aligned_significance(
    significance: np.ndarray | None,
    source_names: tuple[str, ...],
    common_names: tuple[str, ...],
) -> np.ndarray | None:
    if significance is None:
        return None
    aligned, _, _, _ = align_pair(
        significance.astype(bool), source_names, np.zeros((len(common_names),) * 2), common_names
    )
    return aligned.astype(bool)


def _evaluate_series(
    series: MatrixSeries,
    reference: np.ndarray,
    reference_names: tuple[str, ...],
    sampling_rate: float,
    *,
    evaluation_mask: np.ndarray | None = None,
    weight_reference: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    extras = [array for array in (evaluation_mask, weight_reference) if array is not None]
    for lag, matrix in series.matrices.items():
        pred, ref, aligned_extras, common = align_pair(
            matrix, series.neuron_names, reference, reference_names, *extras
        )
        cursor = 0
        aligned_eval = None
        aligned_weights = None
        if evaluation_mask is not None:
            aligned_eval = aligned_extras[cursor].astype(bool)
            cursor += 1
        if weight_reference is not None:
            aligned_weights = aligned_extras[cursor]
        aligned_sig = _aligned_significance(
            series.significance.get(lag), series.neuron_names, common
        )
        metrics = evaluate_scores(
            pred,
            ref,
            evaluation_mask=aligned_eval,
            weight_reference=aligned_weights,
            significance_mask=aligned_sig,
        )
        rows.append(
            {
                "method": series.method,
                "lag": lag,
                "time_s": lag / sampling_rate,
                "n_neurons": len(common),
                **metrics,
            }
        )
    return rows


def run_evaluation(args: argparse.Namespace) -> dict[str, Path]:
    """Run all reference evaluations and return the written paths."""
    sbtg = load_sbtg_archive(args.sbtg, q_alpha=args.q_alpha)
    baselines = load_baselines(args.baseline, args.baselines_dir, sbtg.neuron_names)
    methods = [sbtg, *baselines]

    structural, chemical, gap, structural_names = _load_structural_references(
        args.connectome_dir
    )
    functional, functional_mask, functional_amplitude, functional_names = (
        _load_functional_reference(args.functional_atlas, args.alpha)
    )
    modulatory = _load_modulatory_references(args.modulatory_edges, sbtg.neuron_names)

    structural_rows: list[dict] = []
    chemical_gap_rows: list[dict] = []
    functional_rows: list[dict] = []
    modulatory_rows: list[dict] = []
    for method in methods:
        structural_rows.extend(
            _evaluate_series(
                method, structural, structural_names, args.sampling_rate
            )
        )
        chemical_rows = _evaluate_series(
            method, chemical, structural_names, args.sampling_rate
        )
        gap_rows = _evaluate_series(
            method, gap, structural_names, args.sampling_rate
        )
        if [row["lag"] for row in chemical_rows] != [row["lag"] for row in gap_rows]:
            raise RuntimeError("Chemical and gap evaluations produced different lag grids")
        for chemical_row, gap_row in zip(chemical_rows, gap_rows):
            chemical_gap_rows.append(
                {
                    "method": method.method,
                    "lag": chemical_row["lag"],
                    "time_s": chemical_row["time_s"],
                    "n_neurons": chemical_row["n_neurons"],
                    **{
                        f"{metric}_chem": chemical_row[metric]
                        for metric in (
                            "auroc", "auprc", "spearman", "f1",
                            "n_evaluated", "n_positive", "prevalence",
                        )
                    },
                    **{
                        f"{metric}_gap": gap_row[metric]
                        for metric in (
                            "auroc", "auprc", "spearman", "f1",
                            "n_evaluated", "n_positive", "prevalence",
                        )
                    },
                }
            )
        functional_rows.extend(
            _evaluate_series(
                method,
                functional,
                functional_names,
                args.sampling_rate,
                evaluation_mask=functional_mask,
                weight_reference=functional_amplitude,
            )
        )
        for lag, matrix in method.matrices.items():
            for transmitter, reference in modulatory.items():
                pred, ref, _, common_names = align_pair(
                    matrix, method.neuron_names, reference, sbtg.neuron_names
                )
                n_neurons = len(common_names)
                metrics = evaluate_scores(pred, ref)
                modulatory_rows.append(
                    {
                        "method": method.method,
                        "lag": lag,
                        "time_s": lag / args.sampling_rate,
                        "transmitter": transmitter,
                        "n_neurons": n_neurons,
                        **metrics,
                    }
                )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "structural": output / "structural_reference_metrics.csv",
        "chemical_gap": output / "chemical_gap_reference_metrics.csv",
        "functional": output / "functional_reference_metrics.csv",
        "modulatory": output / "modulatory_reference_metrics.csv",
        "metadata": output / "evaluation_metadata.json",
    }
    pd.DataFrame(structural_rows).sort_values(["method", "lag"]).to_csv(
        paths["structural"], index=False
    )
    pd.DataFrame(chemical_gap_rows).sort_values(["method", "lag"]).to_csv(
        paths["chemical_gap"], index=False
    )
    pd.DataFrame(functional_rows).sort_values(["method", "lag"]).to_csv(
        paths["functional"], index=False
    )
    pd.DataFrame(modulatory_rows).sort_values(
        ["transmitter", "method", "lag"]
    ).to_csv(paths["modulatory"], index=False)

    structural_prevalence = structural_rows[0]["prevalence"] if structural_rows else None
    functional_prevalence = functional_rows[0]["prevalence"] if functional_rows else None
    modulatory_prevalence = {
        transmitter: float(
            np.count_nonzero(matrix)
            / max(1, matrix.shape[0] * (matrix.shape[0] - 1))
        )
        for transmitter, matrix in modulatory.items()
    }
    metadata = {
        "direction_convention": "matrix[target, source]",
        "sampling_rate_hz": args.sampling_rate,
        "functional_label_policy": {
            "positive": f"q < {args.alpha}",
            "confirmed_negative": f"q_eq < {args.alpha} and not positive",
            "ambiguous": "excluded",
        },
        "sbtg_significance_policy": (
            f"explicit Boolean masks when available; otherwise q_value < {args.q_alpha}"
        ),
        "f1_definition": "maximum F1 over thresholds on the evaluated score vector",
        "methods": [method.method for method in methods],
        "inputs": {
            "sbtg": Path(args.sbtg).name,
            "structural_chemical": "A_chem.npy",
            "structural_gap": "A_gap.npy",
            "structural_node_order": "nodes.json",
            "functional_atlas": Path(args.functional_atlas).name,
            "modulatory_edges": Path(args.modulatory_edges).name,
        },
        "pi_cook": structural_prevalence,
        "pi_leifer": functional_prevalence,
        "mono_densities": modulatory_prevalence,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbtg", type=Path, default=DEFAULT_SBTG)
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        type=parse_baseline_spec,
        metavar="METHOD=PATH",
        help="add a baseline archive; may be repeated",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=DEFAULT_BASELINES,
        help="optional directory scanned for baseline NPZ archives",
    )
    parser.add_argument("--connectome-dir", type=Path, default=DEFAULT_CONNECTOME)
    parser.add_argument(
        "--functional-atlas", type=Path, default=DEFAULT_FUNCTIONAL_ATLAS
    )
    parser.add_argument(
        "--modulatory-edges", type=Path, default=DEFAULT_MODULATORY_EDGES
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sampling-rate", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--q-alpha",
        type=float,
        default=0.2,
        help="threshold for explicit q_value_lagN arrays used in weight correlations",
    )
    args = parser.parse_args(argv)
    if args.sampling_rate <= 0:
        parser.error("--sampling-rate must be positive")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must lie between 0 and 1")
    if not 0 < args.q_alpha < 1:
        parser.error("--q-alpha must lie between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = run_evaluation(args)
    print(f"Evaluated SBTG results. Outputs written to {Path(args.output_dir)}:")
    for label, path in paths.items():
        print(f"  {label}: {path.name}")


if __name__ == "__main__":
    main()
