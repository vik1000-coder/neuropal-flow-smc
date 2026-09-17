#!/usr/bin/env python3
"""Build deterministic manifest and SHA-256 indexes for a release directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple, Sequence


MANIFEST_NAME = "release_manifest.csv"
CHECKSUM_NAME = "checksums.sha256"
EXCLUDED_INDEXES = {MANIFEST_NAME, CHECKSUM_NAME}
MANIFEST_HEADER = ("path", "sha256", "size_bytes", "description")


class FileRecord(NamedTuple):
    """Immutable index data plus the source stat used for consistency checks."""

    path: str
    sha256: str
    size_bytes: int
    description: str
    mtime_ns: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(value: str) -> bool:
    """Return whether *value* is a normalized, portable POSIX relative path."""
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.as_posix() == value


def describe(relative: str) -> str:
    """Return a deterministic human-readable description based only on path."""
    path = PurePosixPath(relative)
    name = path.name
    stem = path.stem.replace("_", " ").replace("-", " ")
    suffix = path.suffix.lower()
    top = path.parts[0] if path.parts else ""

    exact = {
        ".gitignore": "Version-control exclusion rules",
        "LICENSE": "Code and data-use notice",
        "README.md": "Release overview and usage guide",
        "pyproject.toml": "Python package and test configuration",
        "requirements.txt": "Python dependency specification",
    }
    if relative in exact:
        return exact[relative]
    if name.lower().startswith("readme"):
        return f"Documentation for {path.parent.as_posix()}"
    if top == "tests":
        return f"Regression test for {stem}"
    if top == "tools":
        return f"Release utility for {stem}"
    if top == "pipeline" and suffix == ".py":
        return f"Pipeline source for {stem}"
    if top == "analysis" and suffix == ".py":
        return f"Analysis source for {stem}"
    if top == "experiments" and suffix == ".py":
        return f"Sensitivity-analysis source for {stem}"
    if top == "results":
        if suffix in {".csv", ".tsv"}:
            return f"Validated result table for {stem}"
        if suffix in {".npz", ".npy"}:
            return f"Portable numerical result archive for {stem}"
        if suffix == ".json":
            return f"Result metadata for {stem}"
    if top == "reference_data":
        if suffix in {".csv", ".tsv"}:
            return f"Reference data table for {stem}"
        if suffix in {".npz", ".npy"}:
            return f"Portable reference array for {stem}"
        if suffix == ".json":
            return f"Reference data metadata for {stem}"
    if top == "reference_snapshot":
        if suffix in {".npz", ".npy"}:
            return f"Portable reference-snapshot array for {stem}"
        if suffix in {".csv", ".tsv"}:
            return f"Reference-snapshot table for {stem}"
        if suffix == ".json":
            return f"Reference-snapshot metadata for {stem}"
        if suffix == ".py":
            return f"Reference-snapshot source for {stem}"
    if top == "docs":
        return f"Release documentation for {stem}"
    if suffix == ".py":
        return f"Python source for {stem}"
    if suffix in {".md", ".rst", ".txt"}:
        return f"Documentation for {stem}"
    if suffix in {".csv", ".tsv"}:
        return f"Tabular data for {stem}"
    if suffix in {".npz", ".npy"}:
        return f"Portable numerical array for {stem}"
    if suffix in {".json", ".toml", ".yaml", ".yml"}:
        return f"Structured metadata for {stem}"
    return f"Release file {relative}"


def inventory(root: Path) -> list[Path]:
    """Return every regular file except the two indexes, rejecting links."""
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"symbolic links are not allowed: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"non-regular filesystem entry is not allowed: {relative}")
        if relative in EXCLUDED_INDEXES:
            continue
        if not safe_relative_path(relative):
            raise ValueError(f"unsafe release path: {relative!r}")
        paths.append(path)
    return paths


def collect_records(root: Path) -> list[FileRecord]:
    """Hash a stable snapshot of every indexed file in sorted path order."""
    records: list[FileRecord] = []
    for path in inventory(root):
        relative = path.relative_to(root).as_posix()
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"file changed while it was being hashed: {relative}")
        records.append(
            FileRecord(
                path=relative,
                sha256=digest,
                size_bytes=after.st_size,
                description=describe(relative),
                mtime_ns=after.st_mtime_ns,
            )
        )
    return records


def render_manifest(records: Sequence[FileRecord]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MANIFEST_HEADER)
    for record in records:
        writer.writerow(
            (record.path, record.sha256, record.size_bytes, record.description)
        )
    return buffer.getvalue()


def render_checksums(records: Sequence[FileRecord]) -> str:
    return "".join(f"{record.sha256}  {record.path}\n" for record in records)


def assert_snapshot_unchanged(root: Path, records: Sequence[FileRecord]) -> None:
    """Abort before publication if paths, sizes, or mtimes changed mid-build."""
    current_paths = [path.relative_to(root).as_posix() for path in inventory(root)]
    expected_paths = [record.path for record in records]
    if current_paths != expected_paths:
        raise RuntimeError("release file inventory changed while indexes were built")
    for record in records:
        stat = (root / record.path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record.size_bytes, record.mtime_ns):
            raise RuntimeError(f"file changed while indexes were built: {record.path}")


def atomic_write(path: Path, content: str) -> None:
    """Replace one UTF-8 text file atomically on the local filesystem."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_indexes(root: Path) -> tuple[Path, Path, int]:
    """Build both root indexes and return their paths and indexed-file count."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"release root is not a directory: {root}")
    records = collect_records(root)
    manifest = render_manifest(records)
    checksums = render_checksums(records)
    assert_snapshot_unchanged(root, records)
    manifest_path = root / MANIFEST_NAME
    checksum_path = root / CHECKSUM_NAME
    atomic_write(manifest_path, manifest)
    atomic_write(checksum_path, checksums)
    return manifest_path, checksum_path, len(records)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="release root (defaults to the parent of tools/)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, checksums, count = build_indexes(args.root)
    print(f"Indexed {count} regular files.")
    print(f"Wrote {manifest}")
    print(f"Wrote {checksums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
