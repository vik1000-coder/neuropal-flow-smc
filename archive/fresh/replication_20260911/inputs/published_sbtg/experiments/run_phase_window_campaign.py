#!/usr/bin/env python3
"""Run the prespecified phase-duration alternative-window campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.phase_window_fit import PHASES, VARIANTS


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_OUT = PROJECT_ROOT / "results/derived/phase_window_sensitivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--matched-replicates", type=int, default=3)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--campaign-name", default="campaign")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def expected_manifest(out_root: Path, variant: str, phase: str, seed: int, replicate: int) -> Path:
    return (
        out_root
        / variant
        / phase
        / f"seed_{seed:04d}"
        / f"replicate_{replicate:03d}"
        / "manifest.json"
    )


def is_complete(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "complete"
        and payload.get("settings", {}).get("epochs") == epochs
    )


def run_task(task: dict, args: argparse.Namespace, log_dir: Path) -> dict:
    variant = task["variant"]
    phase = task["phase"]
    seed = task["model_seed"]
    replicate = task["replicate"]
    manifest = expected_manifest(args.out_root, variant, phase, seed, replicate)
    if not args.force and is_complete(manifest, args.epochs):
        return {**task, "status": "skipped_complete", "elapsed_seconds": 0.0}

    command = [
        sys.executable,
        "-m",
        "experiments.phase_window_fit",
        "--out-root",
        str(args.out_root),
        "--variant",
        variant,
        "--phase",
        phase,
        "--replicate",
        str(replicate),
        "--model-seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--torch-threads",
        str(args.torch_threads),
    ]
    log_path = log_dir / f"{variant}__{phase}__seed{seed:04d}__rep{replicate:03d}.log"
    started = time.monotonic()
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    status = "complete" if completed.returncode == 0 else "failed"
    return {
        **task,
        "status": status,
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "log": log_path.relative_to(args.out_root).as_posix(),
    }


def main() -> None:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for variant in args.variants:
        for phase in PHASES:
            replicates = (
                range(args.matched_replicates)
                if phase in {"baseline", "steady"}
                else range(1)
            )
            for seed in args.model_seeds:
                for replicate in replicates:
                    tasks.append(
                        {
                            "variant": variant,
                            "phase": phase,
                            "model_seed": seed,
                            "replicate": replicate,
                        }
                    )

    campaign_path = args.out_root / f"{args.campaign_name}.json"
    campaign = {
        "status": "running",
        "configuration": {
            "variants": args.variants,
            "matched_replicates": args.matched_replicates,
            "model_seeds": args.model_seeds,
            "epochs": args.epochs,
            "max_workers": args.max_workers,
            "device": args.device,
            "torch_threads": args.torch_threads,
        },
        "n_tasks": len(tasks),
        "results": [],
    }
    atomic_json(campaign_path, campaign)
    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {executor.submit(run_task, task, args, log_dir): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            campaign["results"].append(result)
            if result["status"] == "failed":
                failures.append(result)
            print(
                f"[{len(campaign['results'])}/{len(tasks)}] "
                f"{result['variant']} {result['phase']} seed={result['model_seed']} "
                f"rep={result['replicate']} {result['status']} "
                f"({result['elapsed_seconds'] / 60:.1f} min)",
                flush=True,
            )
            atomic_json(campaign_path, campaign)

    campaign.update(
        {
            "status": "failed" if failures else "complete",
            "elapsed_seconds": time.monotonic() - started,
            "n_failures": len(failures),
        }
    )
    atomic_json(campaign_path, campaign)
    if failures:
        raise SystemExit(f"{len(failures)} phase-duration jobs failed; inspect {log_dir}")
    print(f"PHASE-WINDOW CAMPAIGN COMPLETE: {len(tasks)} tasks", flush=True)


if __name__ == "__main__":
    main()
