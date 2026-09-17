"""Atomic compact results and reproducibility manifests."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    encoded = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def source_tree_digest(root: Path) -> str | None:
    """Hash reviewable benchmark source/config files when no outer git repo exists."""

    try:
        digest = hashlib.blake2b(digest_size=20)
        paths = []
        for directory in (root / "src", root / "configs", root / "scripts", root / "docs"):
            if directory.exists():
                paths.extend(
                    path
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json", ".sh", ".md"}
                )
        paths.extend(path for path in (root / "pyproject.toml", root / "requirements-causal.txt") if path.exists())
        for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
            relative = str(path.relative_to(root)).encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(path.read_bytes())
        return digest.hexdigest()
    except Exception:
        return None


def dependency_source_digest(root: Path) -> str | None:
    """Hash executable source for a local, non-git Python dependency.

    The benchmark imports ``sid_neuromod`` directly from the neighboring
    workspace tree.  Its implementation therefore belongs to the frozen
    execution identity even though it is not nested under this package.
    """

    try:
        paths = []
        source = root / "src"
        if source.is_dir():
            paths.extend(path for path in source.rglob("*.py") if path.is_file())
        for name in ("pyproject.toml", "requirements.txt"):
            path = root / name
            if path.is_file():
                paths.append(path)
        if not paths:
            return None
        digest = hashlib.blake2b(digest_size=20)
        for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
            relative = str(path.relative_to(root)).encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(path.read_bytes())
        return digest.hexdigest()
    except Exception:
        return None


def _venv_packages(venv: Path) -> dict[str, str] | None:
    """Read all installed distribution versions without importing the venv."""

    try:
        site_packages = sorted((venv / "lib").glob("python*/site-packages"))
        if not site_packages:
            return None
        packages: dict[str, str] = {}
        for distribution in importlib.metadata.distributions(
            path=[str(path) for path in site_packages]
        ):
            name = distribution.metadata.get("Name")
            if name:
                packages[str(name).lower()] = str(distribution.version)
        return dict(sorted(packages.items()))
    except Exception:
        return None


def environment_manifest(workspace: Path) -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "pandas",
        "statsmodels",
        "pyyaml",
        "threadpoolctl",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    benchmark_digest = source_tree_digest(workspace / "neuromod_benchmark")
    sid_digest = dependency_source_digest(workspace / "sid_neuromod")
    causal_packages = _venv_packages(
        workspace / "neuromod_benchmark" / ".causal_venv"
    )
    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "causal_environment_packages": causal_packages,
        "sid_neuromod_git_head": _git(workspace / "sid_neuromod", "rev-parse", "HEAD"),
        "sid_neuromod_source_digest": sid_digest,
        "sbtg_git_head": _git(workspace / "SBTG", "rev-parse", "HEAD"),
        "sbtg_dirty": bool(_git(workspace / "SBTG", "status", "--porcelain")),
        "benchmark_source_digest": benchmark_digest,
    }
    manifest["execution_environment_digest"] = stable_hash(
        {
            "python": manifest["python"],
            "platform": manifest["platform"],
            "packages": packages,
            "causal_environment_packages": causal_packages,
            "sid_neuromod_source_digest": sid_digest,
            "benchmark_source_digest": benchmark_digest,
        }
    )
    return manifest
