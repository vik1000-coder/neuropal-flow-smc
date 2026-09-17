"""Deterministic serialization, provenance, and local disk-safety helpers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import yaml


DEFAULT_SOURCE_SUFFIXES = frozenset(
    {".py", ".yaml", ".yml", ".json", ".toml", ".lock", ".txt", ".md", ".sh"}
)
DEFAULT_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
        "__pycache__",
        "outputs",
        "results",
    }
)


def jsonable(value: Any) -> Any:
    """Convert scientific Python values to strict JSON-compatible objects.

    Non-finite numbers are rejected. Callers must encode an unavailable or failed
    metric with its explicit status instead of smuggling it through as NaN/null.
    """

    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return jsonable(value.to_dict())
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    try:
        import torch

        if torch.is_tensor(value):
            return jsonable(value.detach().cpu().numpy())
    except ImportError:
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats require an explicit failed metric status")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    """SHA-256 over canonical semantic JSON, independent of mapping order."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, block_bytes: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config_or_path: Any) -> str:
    """Semantic SHA-256 for a resolved config object or YAML/JSON file.

    Reordering YAML mapping keys does not change this digest. Use ``sha256_file``
    separately when the exact source bytes must also be frozen.
    """

    if isinstance(config_or_path, (str, Path)):
        source = Path(config_or_path)
        suffix = source.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(source.read_text(encoding="utf-8"))
        elif suffix == ".json":
            value = json.loads(source.read_text(encoding="utf-8"))
        else:
            raise ValueError("config files must be YAML or JSON")
        return stable_sha256(value)
    if hasattr(config_or_path, "to_dict") and callable(config_or_path.to_dict):
        config_or_path = config_or_path.to_dict()
    return stable_sha256(config_or_path)


def _source_paths(
    root: Path,
    *,
    suffixes: frozenset[str],
    ignored_parts: frozenset[str],
) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(root)
        if any(part in ignored_parts or part.startswith("._") for part in relative.parts):
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def source_tree_sha256(
    root: str | Path,
    *,
    suffixes: Sequence[str] = tuple(DEFAULT_SOURCE_SUFFIXES),
    ignored_parts: Sequence[str] = tuple(DEFAULT_IGNORED_PARTS),
) -> str:
    """Hash reviewable source, tests, configs, locks, scripts, and docs.

    Relative paths are included, so moving content between modules changes the
    identity even when file bytes happen to be identical.
    """

    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    normalized_suffixes = frozenset(
        value if str(value).startswith(".") else f".{value}" for value in suffixes
    )
    ignored = frozenset(str(value) for value in ignored_parts)
    digest = hashlib.sha256()
    for path in _source_paths(
        source_root, suffixes=normalized_suffixes, ignored_parts=ignored
    ):
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    """Durably replace a JSON artifact without exposing a partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                jsonable(value),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some network or non-POSIX filesystems do not support directory fsync.
            pass
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            # A concurrent atomic writer may have replaced its temporary file.
            continue
    return total


def ensure_within(root: str | Path, path: str | Path) -> Path:
    resolved_root = Path(root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"output path {resolved} escapes isolated root {resolved_root}")
    return resolved


def disk_snapshot(path: str | Path) -> dict[str, float]:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    return {
        "free_gb": usage.free / 1024**3,
        "total_gb": usage.total / 1024**3,
        "output_gb": directory_size_bytes(target) / 1024**3,
    }


def enforce_disk_safety(
    output_dir: str | Path,
    *,
    min_free_gb: float,
    max_output_gb: float,
) -> dict[str, float]:
    if min_free_gb <= 0 or max_output_gb <= 0:
        raise ValueError("disk safety limits must be positive")
    snapshot = disk_snapshot(output_dir)
    if snapshot["free_gb"] < min_free_gb:
        raise RuntimeError(
            f"free disk {snapshot['free_gb']:.2f} GiB is below safety floor "
            f"{min_free_gb:.2f} GiB"
        )
    if snapshot["output_gb"] > max_output_gb:
        raise RuntimeError(
            f"output size {snapshot['output_gb']:.2f} GiB exceeds quota "
            f"{max_output_gb:.2f} GiB"
        )
    return snapshot


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_manifest(
    source_root: str | Path | None = None,
    *,
    packages: Sequence[str] = (
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "torch",
        "pyyaml",
        "pyarrow",
    ),
) -> dict[str, Any]:
    """Describe the execution environment without requiring a Git repository."""

    package_versions = _package_versions(packages)
    source_digest = None
    if source_root is not None:
        source_digest = source_tree_sha256(source_root)
    identity = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions,
        "source_tree_sha256": source_digest,
    }
    manifest = dict(identity)
    try:
        import torch

        manifest["accelerators"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
    except ImportError:
        manifest["accelerators"] = {"cuda_available": False, "mps_available": False}
    identity["accelerators"] = manifest["accelerators"]
    manifest["environment_sha256"] = stable_sha256(identity)
    return manifest


__all__ = [
    "atomic_json",
    "canonical_json_bytes",
    "config_sha256",
    "directory_size_bytes",
    "disk_snapshot",
    "enforce_disk_safety",
    "ensure_within",
    "environment_manifest",
    "jsonable",
    "sha256_file",
    "source_tree_sha256",
    "stable_sha256",
]
