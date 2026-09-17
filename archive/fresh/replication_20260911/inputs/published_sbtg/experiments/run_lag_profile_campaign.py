#!/usr/bin/env python3
"""Run the lag-profile recording-row resampling campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
LAGS = (1, 2, 3, 5, 8, 10, 15, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results/derived/lag_profile_bootstrap",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--effective-batch-size", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=20260803)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help=(
            "Bootstrap replicate indices. Use four independent recording-row resamples for the "
            "default analysis."
        ),
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help=(
            "Run only the requested bootstrap chains and reuse external full-data "
            "controls during summarization. This avoids duplicate control fits."
        ),
    )
    parser.add_argument(
        "--full-control-root",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help=(
            "Pass an external full-control root to the summarizer. May be "
            "repeated; its manifest, hashes, and fold seeds are validated."
        ),
    )
    return parser.parse_args()


def label(sample_kind: str, replicate: int, model_seed: int) -> str:
    if sample_kind == "full":
        return f"full_seed_{model_seed:04d}"
    return f"bootstrap_{replicate:03d}_seed_{model_seed:04d}"


def result_complete(path: Path, expected: dict) -> bool:
    manifest_path = path.parent / "manifest.json"
    if not path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("status") == "complete" and all(
        manifest.get(key) == value for key, value in expected.items()
    )


def run_command(command: list[str], log_path: Path, env: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed; see {log_path}")


def run_chain(
    args: argparse.Namespace,
    sample_kind: str,
    replicate: int,
    model_seed: int,
    env: dict,
) -> None:
    chain_label = label(sample_kind, replicate, model_seed)
    started = time.monotonic()
    print(f"[{chain_label}] START", flush=True)
    for lag in LAGS:
        result_path = args.output_root / chain_label / f"lag_{lag:02d}" / "result.npz"
        expected = {
            "sample_kind": sample_kind,
            "replicate": replicate,
            "lag": lag,
            "model_seed": model_seed,
            "data_seed": args.data_seed,
            "device": args.device,
            "effective_batch_size": args.effective_batch_size,
        }
        if result_complete(result_path, expected):
            print(f"[{chain_label} lag={lag:02d}] SKIP complete", flush=True)
            continue
        command = [
            str(args.python),
            "-m", "experiments.lag_profile_fit",
            "--out-root", str(args.output_root),
            "--sample-kind", sample_kind,
            "--replicate", str(replicate),
            "--lag", str(lag),
            "--model-seed", str(model_seed),
            "--data-seed", str(args.data_seed),
            "--device", args.device,
            "--torch-threads", str(args.torch_threads),
            "--effective-batch-size", str(args.effective_batch_size),
        ]
        run_command(
            command,
            args.output_root / "logs" / f"{chain_label}_lag_{lag:02d}.log",
            env,
        )
        print(f"[{chain_label} lag={lag:02d}] COMPLETE", flush=True)
    print(
        f"[{chain_label}] COMPLETE in {(time.monotonic() - started) / 60:.1f} min",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be positive")
    replicates = sorted(set(args.bootstrap_replicates))
    if replicates not in ([0, 1, 2, 3], [0, 1, 2, 3, 4]):
        raise ValueError(
            "--bootstrap-replicates must be `0 1 2 3` or `0 1 2 3 4`"
        )
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    thread_count = str(args.torch_threads)
    env.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "VECLIB_MAXIMUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
            "PYTHONUNBUFFERED": "1",
        }
    )

    chains = []
    if not args.bootstrap_only:
        chains.extend(("full", -1, seed) for seed in (42, 43, 44))
    chains.extend(
        ("bootstrap", replicate, seed)
        for replicate in replicates
        for seed in (42, 43)
    )
    print(
        f"Running {len(chains)} eight-lag chains with at most "
        f"{args.max_parallel} concurrent processes; {args.torch_threads} Torch "
        "threads each.",
        flush=True,
    )
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {
            pool.submit(run_chain, args, kind, replicate, seed, env):
            label(kind, replicate, seed)
            for kind, replicate, seed in chains
        }
        for future in as_completed(futures):
            chain_label = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((chain_label, repr(exc)))
                print(f"[{chain_label}] FAILED: {exc}", flush=True)
    if failures:
        raise RuntimeError(f"Bootstrap chains failed: {failures}")

    summary_dir = args.output_root / "summary"
    summary_command = [
        str(args.python),
        "-m", "experiments.lag_profile_summary",
        "--input-root", str(args.output_root),
        "--out-dir", str(summary_dir),
        "--data-seed", str(args.data_seed),
        "--fit-device", args.device,
        "--effective-batch-size", str(args.effective_batch_size),
        "--require-complete",
    ]
    for value in args.full_control_root:
        summary_command.extend(("--full-control-root", value))
    summary_command.append("--bootstrap-replicates")
    summary_command.extend(str(replicate) for replicate in replicates)
    run_command(
        summary_command,
        args.output_root / "logs/lag_profile_bootstrap_summary.log",
        env,
    )
    print(f"LAG-PROFILE CAMPAIGN COMPLETE: {summary_dir}", flush=True)


if __name__ == "__main__":
    main()
