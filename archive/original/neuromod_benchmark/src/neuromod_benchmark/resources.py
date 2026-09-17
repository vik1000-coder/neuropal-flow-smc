"""Local-machine safety gates and resource accounting."""
from __future__ import annotations

import os
import resource
import shutil
from contextlib import contextmanager
from pathlib import Path

from threadpoolctl import threadpool_limits

from .schema import ResourceBudget


THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_thread_environment(max_threads: int) -> None:
    for name in THREAD_ENV:
        os.environ[name] = str(max_threads if name != "OMP_NUM_THREADS" else 1)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def resource_snapshot() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes; Linux reports KiB. This project is currently run on macOS.
    maxrss = float(usage.ru_maxrss)
    if maxrss < 10_000_000:  # defensive Linux/small-process heuristic
        maxrss *= 1024.0
    return {
        "peak_rss_gb": maxrss / 1024**3,
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
    }


def enforce_storage_budget(output: Path, budget: ResourceBudget) -> None:
    output.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output).free / 1024**3
    if free_gb < budget.stop_free_disk_gb:
        raise RuntimeError(
            f"free disk {free_gb:.2f} GiB is below safety floor {budget.stop_free_disk_gb:.2f} GiB"
        )
    output_gb = directory_size(output) / 1024**3
    if output_gb > budget.max_output_gb:
        raise RuntimeError(
            f"output size {output_gb:.2f} GiB exceeds quota {budget.max_output_gb:.2f} GiB"
        )


@contextmanager
def limited_threads(budget: ResourceBudget):
    configure_thread_environment(budget.max_threads)
    with threadpool_limits(limits=budget.max_threads):
        yield
