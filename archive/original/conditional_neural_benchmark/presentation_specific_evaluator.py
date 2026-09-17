from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.data import STIMULUS_PERIODS_SECONDS, load_cohort
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.metrics import metric_rows
from conditional_neural_benchmark.presentation_encoding_runner import _permutations
from conditional_neural_benchmark.runner import _sample_neural, _split_windows
from conditional_neural_benchmark.inference import load_checkpoint


PATTERN = re.compile(
    r"(?P<model>.+)__L(?P<lag>\d+)__f(?P<fold>\d+)__s(?P<seed>\d+)\.pt$"
)


def main() -> None:
    raise RuntimeError(
        "quarantined: epoch position is not chemical identity; corrected evaluation "
        "must use worm-specific StimulusSchedule metadata"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-run", type=Path, required=True)
    parser.add_argument("--phase", default="presentation_confirmation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    folds = _load_folds(cohort, args.evidence_run.resolve())
    permutations = _permutations(cohort.n_worms)
    checkpoints = sorted((args.run_dir / "checkpoints" / args.phase).glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError("no presentation checkpoints")

    rows: list[dict[str, object]] = []
    window_cache: dict[tuple[int, str, bool], object] = {}
    for path in checkpoints:
        match = PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"unrecognized checkpoint: {path}")
        fold, seed, lag = (int(match.group(key)) for key in ("fold", "seed", "lag"))
        model, checkpoint, device = load_checkpoint(path, device=args.device)
        metadata = checkpoint.get("trial_metadata", {})
        encoding = str(metadata.get("stimulus_encoding", "binary"))
        shuffled = bool(metadata.get("presentation_labels_subject_shuffled", False))
        key = (fold, encoding, shuffled)
        if key not in window_cache:
            _, _, _, test = _split_windows(
                cohort,
                folds,
                fold,
                lag,
                stimulus_encoding=encoding,
                presentation_permutations=permutations if shuffled else None,
            )
            window_cache[key] = test
        test = window_cache[key]
        for event, (start, end) in enumerate(STIMULUS_PERIODS_SECONDS, 1):
            lo = int(np.floor(start * cohort.fps))
            hi = int(np.ceil(end * cohort.fps))
            # At target frame `lo`, the causal history ends immediately before
            # onset and cannot yet contain the active presentation label.
            index = np.flatnonzero((test.time >= lo + 1) & (test.time < hi))
            history = test.flat()[index]
            last = test.history[index, -1, :cohort.n_neurons]
            target = test.target[index] - last
            draws, _ = _sample_neural(
                model,
                history,
                target,
                n_samples=args.samples,
                seed=91_003 + 1009 * fold + 17 * seed + event,
                device=str(device),
                chunk_size=192,
            )
            draws = draws + last[:, None]
            metrics = metric_rows(draws, test.target[index], seed + event)
            row = {
                "checkpoint": str(path.resolve()),
                "model_id": match.group("model"),
                "fold": fold,
                "seed": seed,
                "stimulus_encoding": encoding,
                "subject_shuffled": shuffled,
                "presentation": event,
                "rows": len(index),
            }
            row.update({name: float(np.mean(value)) for name, value in metrics.items()})
            rows.append(row)
        print(f"PRESENTATION_EVAL_DONE {path.name}", flush=True)
        del model
        if args.device == "mps":
            torch.mps.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics_by_checkpoint_presentation.csv", index=False)
    aggregate = frame.groupby(
        ["model_id", "stimulus_encoding", "subject_shuffled", "presentation"],
        dropna=False,
    )[["energy", "variogram", "rmse"]].agg(["mean", "std", "count"])
    aggregate.columns = ["__".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(output / "metrics_aggregate_presentation.csv", index=False)
    protocol = {
        "run_dir": str(args.run_dir.resolve()),
        "phase": args.phase,
        "samples": args.samples,
        "evaluation_window": "active targets from one frame after onset through last active frame",
        "reason_for_excluding_first_active_target": "presentation label is not yet in causal history",
        "selection_input": False,
        "claim_boundary": "held-out predictive distribution only",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
