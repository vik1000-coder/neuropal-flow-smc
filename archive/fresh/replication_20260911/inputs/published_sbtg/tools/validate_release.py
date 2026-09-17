#!/usr/bin/env python3
"""Read-only validation gate for the public SBTG release.

The validator checks repository hygiene, portable NumPy serialization, result
schemas, selected scientific invariants, and the two release indexes.  It does
not create or modify files, so a successful run cannot make its own manifest
stale.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np


MANIFEST_NAME = "release_manifest.csv"
CHECKSUM_NAME = "checksums.sha256"
INDEX_FILES = {MANIFEST_NAME, CHECKSUM_NAME}
MANIFEST_HEADER = ("path", "sha256", "size_bytes", "description")
PAPER_LAGS = (1, 2, 3, 5, 8, 10, 15, 20)

TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {
    "LICENSE",
    "NOTICE",
    "AUTHORS",
    MANIFEST_NAME,
    CHECKSUM_NAME,
}
BLOCKED_COMPONENTS = {
    ".git",
    ".idea",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "env",
    "logs",
    "venv",
}
BLOCKED_FILENAMES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}
BLOCKED_SUFFIXES = {
    ".7z",
    ".bak",
    ".err",
    ".joblib",
    ".log",
    ".out",
    ".pickle",
    ".pkl",
    ".pyo",
    ".pyc",
    ".rar",
    ".swp",
    ".tar",
    ".tgz",
    ".tmp",
    ".zip",
}

REQUIRED_PATHS = (
    "README.md",
    "results/paper/sbtg_lag_matrices.npz",
    "results/paper/hyperparameters.json",
    "results/paper/evaluation/chemical_gap_reference_metrics.csv",
    "results/paper/evaluation/functional_reference_metrics.csv",
    "results/paper/evaluation/modulatory_reference_metrics.csv",
    "results/paper/evaluation/structural_reference_metrics.csv",
    "results/robustness/residual_dependence",
    "results/robustness/structured_noise",
    "results/robustness/lag_profiles",
    "results/robustness/lag_profiles/fits",
    "results/robustness/phase_duration",
    "results/robustness/method_agreement",
    "results/robustness/partial_observation",
    "tools/validate_release.py",
    "tests",
)

# These are required columns, not merely suggestions.  Additional diagnostic
# columns are allowed so reruns can retain useful information without weakening
# compatibility of the published fields.
REQUIRED_CSV_COLUMNS: Mapping[str, set[str]] = {
    "results/paper/evaluation/chemical_gap_reference_metrics.csv": {
        "method", "lag", "time_s", "n_neurons", "auroc_chem",
        "auprc_chem", "spearman_chem", "f1_chem", "n_evaluated_chem",
        "n_positive_chem", "prevalence_chem", "auroc_gap", "auprc_gap",
        "spearman_gap", "f1_gap", "n_evaluated_gap", "n_positive_gap",
        "prevalence_gap",
    },
    "results/paper/evaluation/functional_reference_metrics.csv": {
        "method", "lag", "time_s", "n_neurons", "auroc", "auprc",
        "spearman", "f1", "n_evaluated", "n_positive", "prevalence",
    },
    "results/paper/evaluation/modulatory_reference_metrics.csv": {
        "method", "lag", "time_s", "transmitter", "n_neurons", "auroc",
        "auprc", "spearman", "f1", "n_evaluated", "n_positive",
        "prevalence",
    },
    "results/paper/evaluation/structural_reference_metrics.csv": {
        "method", "lag", "time_s", "n_neurons", "auroc", "auprc",
        "spearman", "f1", "n_evaluated", "n_positive", "prevalence",
    },
    "results/robustness/residual_dependence/order_summary.csv": {
        "var_order", "n_recordings", "n_neurons", "n_residual_rows",
        "heldout_mse_mean", "constant_mean_mse_mean",
        "mean_abs_residual_corr", "median_abs_residual_corr",
        "p90_abs_residual_corr", "p95_abs_residual_corr",
        "max_abs_residual_corr",
    },
    "results/robustness/structured_noise/metrics.csv": {
        "seed", "covariance_kind", "target_mean_abs_corr", "lag",
        "lag_is_null", "significant_edges", "false_positive_edges",
        "precision_significant", "auroc", "auprc", "f1_topk",
    },
    "results/robustness/structured_noise/paired_summary.csv": {
        "covariance_kind", "target_mean_abs_corr", "lag", "lag_is_null",
        "delta_auroc_mean", "delta_significant_edges_mean",
        "delta_precision_significant_mean",
    },
    "results/robustness/method_agreement/edge_rank_agreement.csv": {
        "left", "right", "n_neurons", "spearman_abs_weight", "top_k",
        "shared_edges", "jaccard",
    },
    "results/robustness/partial_observation/paired_summary.csv": {
        "frac_hidden", "n_seeds", "auroc_delta_mean", "auroc_delta_sd",
        "auroc_delta_ci_low", "auroc_delta_ci_high",
    },
    "results/robustness/phase_duration/exploratory_endpoint/lag5_lag1_phase_summary.csv": {
        "phase", "n_model_seeds", "all_mean", "all_sd", "bh_mean", "bh_sd",
        "by_mean", "by_sd",
    },
    "results/robustness/lag_profiles/production_lineage_lag2plus/published_reference_metrics.csv": {
        "network", "n_profile_lags", "center_of_mass_lag", "weighted_sd_lag",
        "peak_lag", "peak_f1", "profile_moments_defined",
    },
}


@dataclass(frozen=True, order=True)
class ValidationIssue:
    """One deterministic validation finding."""

    category: str
    path: str
    message: str

    def render(self) -> str:
        location = f" ({self.path})" if self.path else ""
        return f"[{self.category}]{location} {self.message}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _strict_json(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


class ReleaseValidator:
    """Run all public-release checks and collect findings instead of failing fast."""

    def __init__(
        self,
        root: Path,
        *,
        require_layout: bool = True,
        require_indexes: bool = True,
        max_file_bytes: int = 25 * 1024 * 1024,
        max_total_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        self.root = Path(root).resolve()
        self.require_layout = require_layout
        self.require_indexes = require_indexes
        self.max_file_bytes = int(max_file_bytes)
        self.max_total_bytes = int(max_total_bytes)
        self.issues: list[ValidationIssue] = []
        self._csv_cache: dict[str, list[dict[str, str]]] = {}

        # Assemble sensitive markers so the rules do not flag their own source.
        self._internal_markers = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"\b" + "review" + r"er\b",
                r"\b" + "rebut" + r"tal\b",
                r"\b" + "sl" + r"urm\b",
                r"\b" + "s" + r"batch\b",
                r"\b" + "clu" + r"ster\b",
                r"\bw\d+h\d+\b",
                r"\ba7u4\b",
                r"\bqzaw\b",
            )
        )
        home_roots = ("Users", "home")
        machine_roots = ("private", "scratch", "gpfs", "lustre", "nfs")
        self._machine_path_patterns = (
            re.compile(r"/(?:" + "|".join(home_roots) + r")/[^\s/'\"<>]+/"),
            re.compile(r"/(?:" + "|".join(machine_roots) + r")/[^\s'\"<>]+"),
            re.compile(r"[A-Za-z]:\\(?:" + "|".join(home_roots) + r")\\[^\\\s]+", re.I),
            re.compile(r"\b[A-Za-z0-9][A-Za-z0-9._-]*\.local\b", re.I),
        )
        self._unsafe_python_calls = {
            "getpass.getuser",
            "os.getcwd",
            "Path.cwd",
            "Path.home",
            "pathlib.Path.cwd",
            "pathlib.Path.home",
            "platform.node",
            "socket.gethostname",
        }

    def add(self, category: str, path: Path | str, message: str) -> None:
        if isinstance(path, Path):
            try:
                rendered = path.relative_to(self.root).as_posix()
            except ValueError:
                rendered = path.as_posix()
        else:
            rendered = path
        issue = ValidationIssue(category, rendered, message)
        if issue not in self.issues:
            self.issues.append(issue)

    def run(self) -> list[ValidationIssue]:
        if not self.root.is_dir():
            self.add("layout", "", "release root does not exist or is not a directory")
            return self.issues

        if self.require_layout:
            self._validate_layout()
        regular_files = self._inventory()
        self._validate_text_and_source(regular_files)
        self._validate_serialized_arrays(regular_files)
        self._validate_json_files(regular_files)
        self._validate_csv_files(regular_files)
        if self.require_layout:
            self._validate_paper_artifact()
            self._validate_numerical_summaries()
        if self.require_indexes:
            self._validate_indexes(regular_files)
        self.issues.sort()
        return self.issues

    def _validate_layout(self) -> None:
        for relative in REQUIRED_PATHS:
            if not (self.root / relative).exists():
                self.add("layout", relative, "required release path is missing")
        if not any((self.root / name).is_file() for name in ("LICENSE", "LICENSE.txt", "LICENSE.md")):
            self.add("licensing", "", "a root LICENSE file is required")

    def _inventory(self) -> list[Path]:
        files: list[Path] = []
        total = 0
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            components = set(relative.parts)
            if path.is_symlink():
                self.add("portability", path, "symbolic links are not allowed in the release")
                continue
            if components & BLOCKED_COMPONENTS:
                self.add("junk", path, "cache, environment, log, or VCS path is not allowed")
            if not path.is_file():
                continue
            files.append(path)
            name_lower = path.name.lower()
            suffix_lower = path.suffix.lower()
            if path.name in BLOCKED_FILENAMES:
                self.add("junk", path, "operating-system metadata file is not allowed")
            if suffix_lower in BLOCKED_SUFFIXES or name_lower.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
                self.add("junk", path, "generated, serialized-object, log, or archive file is not allowed")
            if path.name.endswith("~"):
                self.add("junk", path, "editor backup file is not allowed")
            try:
                size = path.stat().st_size
            except OSError as exc:
                self.add("filesystem", path, f"could not stat file: {exc}")
                continue
            total += size
            if size > self.max_file_bytes:
                self.add(
                    "size",
                    path,
                    f"file is {size} bytes; limit is {self.max_file_bytes} bytes",
                )
        if total > self.max_total_bytes:
            self.add(
                "size",
                "",
                f"release is {total} bytes; limit is {self.max_total_bytes} bytes",
            )
        return files

    def _validate_text_and_source(self, files: Sequence[Path]) -> None:
        for path in files:
            if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_FILENAMES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                self.add("encoding", path, f"text file is not valid UTF-8: {exc}")
                continue
            except OSError as exc:
                self.add("filesystem", path, f"could not read text file: {exc}")
                continue
            self._validate_text_controls(path, text)
            self._scan_sensitive_text(path, text)
            if path.suffix.lower() == ".md":
                self._validate_markdown_tables(path, text)
            if path.suffix.lower() == ".py":
                self._validate_python(path, text)

    def _validate_text_controls(self, path: Path, text: str) -> None:
        """Reject invisible C0 controls except tab and conventional newlines."""
        allowed = {"\t", "\n", "\r"}
        controls = [
            (position, character)
            for position, character in enumerate(text)
            if ord(character) < 0x20 and character not in allowed
        ]
        if not controls:
            return
        position, character = controls[0]
        line = text.count("\n", 0, position) + 1
        previous_newline = text.rfind("\n", 0, position)
        column = position - previous_newline
        self.add(
            "encoding",
            path,
            f"contains {len(controls)} disallowed C0 control character(s); "
            f"first is U+{ord(character):04X} at line {line}, column {column}",
        )

    def _validate_markdown_tables(self, path: Path, text: str) -> None:
        """Validate GitHub-style pipe tables outside fenced code blocks."""
        lines = text.splitlines()
        visible = self._markdown_lines_outside_fences(lines)
        index = 0
        while index < len(lines):
            if not visible[index]:
                index += 1
                continue
            header = self._parse_markdown_pipe_row(lines[index])
            if header is None:
                index += 1
                continue

            following = (
                self._parse_markdown_pipe_row(lines[index + 1])
                if index + 1 < len(lines) and visible[index + 1]
                else None
            )
            has_boundary_pipe = self._has_boundary_pipe(lines[index])
            # A boundary-pipe row is unambiguously table-like.  Rows without
            # boundary pipes are tables only when followed by another pipe row;
            # this avoids treating ordinary prose containing "A | B" as a table.
            if not has_boundary_pipe and following is None:
                index += 1
                continue
            if len(header) < 1:
                index += 1
                continue

            block_end = index + 1
            while block_end < len(lines) and visible[block_end]:
                if self._parse_markdown_pipe_row(lines[block_end]) is None:
                    break
                block_end += 1

            blank_headers = [position + 1 for position, cell in enumerate(header) if not cell.strip()]
            if blank_headers:
                self.add(
                    "markdown-table",
                    path,
                    f"blank header cell(s) {blank_headers} at line {index + 1}",
                )

            if following is None or not self._is_markdown_separator(following):
                detail = "missing a valid separator row"
                if following is not None and self._resembles_markdown_separator(following):
                    detail = "has an invalid separator row"
                self.add(
                    "markdown-table",
                    path,
                    f"table header at line {index + 1} {detail} immediately below it",
                )
                expected_columns = len(header)
                for row_index in range(index + 1, block_end):
                    row = self._parse_markdown_pipe_row(lines[row_index])
                    if row is not None and len(row) != expected_columns:
                        self.add(
                            "markdown-table",
                            path,
                            f"line {row_index + 1} has {len(row)} cells; expected {expected_columns}",
                        )
                index = max(block_end, index + 1)
                continue

            expected_columns = len(header)
            for row_index in range(index + 1, block_end):
                row = self._parse_markdown_pipe_row(lines[row_index])
                if row is not None and len(row) != expected_columns:
                    self.add(
                        "markdown-table",
                        path,
                        f"line {row_index + 1} has {len(row)} cells; expected {expected_columns}",
                    )
            index = max(block_end, index + 2)

    @staticmethod
    def _markdown_lines_outside_fences(lines: Sequence[str]) -> list[bool]:
        """Mark lines that are outside backtick or tilde fenced code blocks."""
        visible: list[bool] = []
        fence_character: str | None = None
        fence_length = 0
        for line in lines:
            match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
            if fence_character is None:
                if match:
                    marker = match.group(1)
                    fence_character = marker[0]
                    fence_length = len(marker)
                    visible.append(False)
                else:
                    visible.append(True)
                continue
            visible.append(False)
            if match:
                marker = match.group(1)
                if marker[0] == fence_character and len(marker) >= fence_length:
                    fence_character = None
                    fence_length = 0
        return visible

    @staticmethod
    def _has_boundary_pipe(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        starts = stripped.startswith("|")
        if stripped.endswith("|"):
            backslashes = 0
            position = len(stripped) - 2
            while position >= 0 and stripped[position] == "\\":
                backslashes += 1
                position -= 1
            ends = backslashes % 2 == 0
        else:
            ends = False
        return starts or ends

    @staticmethod
    def _parse_markdown_pipe_row(line: str) -> list[str] | None:
        """Split one pipe row, respecting escapes and inline code spans."""
        if not line.strip() or line.startswith("    ") or line.startswith("\t"):
            return None
        cells: list[str] = []
        buffer: list[str] = []
        separators = 0
        code_delimiter = 0
        index = 0
        while index < len(line):
            character = line[index]
            if character == "`":
                end = index
                while end < len(line) and line[end] == "`":
                    end += 1
                run = end - index
                if code_delimiter == 0:
                    code_delimiter = run
                elif run == code_delimiter:
                    code_delimiter = 0
                buffer.append(line[index:end])
                index = end
                continue
            if character == "|" and code_delimiter == 0:
                backslashes = 0
                position = index - 1
                while position >= 0 and line[position] == "\\":
                    backslashes += 1
                    position -= 1
                if backslashes % 2 == 0:
                    cells.append("".join(buffer).strip())
                    buffer = []
                    separators += 1
                    index += 1
                    continue
            buffer.append(character)
            index += 1
        if separators == 0:
            return None
        cells.append("".join(buffer).strip())
        stripped = line.strip()
        if stripped.startswith("|") and cells and not cells[0]:
            cells.pop(0)
        if ReleaseValidator._has_boundary_pipe(line) and stripped.endswith("|") and cells and not cells[-1]:
            cells.pop()
        return cells

    @staticmethod
    def _is_markdown_separator(cells: Sequence[str]) -> bool:
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)

    @staticmethod
    def _resembles_markdown_separator(cells: Sequence[str]) -> bool:
        if not cells:
            return False
        allowed = [bool(re.fullmatch(r"[:\-\s]*", cell)) for cell in cells]
        return all(allowed) and any("-" in cell for cell in cells)

    def _scan_sensitive_text(self, path: Path, text: str) -> None:
        for pattern in self._machine_path_patterns:
            match = pattern.search(text)
            if match:
                self.add(
                    "identity",
                    path,
                    f"machine-specific path or host identifier found near {match.group(0)!r}",
                )
        for pattern in self._internal_markers:
            match = pattern.search(text)
            if match:
                self.add(
                    "internal-language",
                    path,
                    f"internal response or infrastructure marker {match.group(0)!r} is not public-facing",
                )

        # q-values must never be converted directly to a Boolean mask.  Limit
        # the window so unrelated, well-guarded conversions do not false-positive.
        compact = re.sub(r"\s+", " ", text)
        q_name = "q_value_" + "lag1"
        bool_cast = r"(?:dtype\s*=\s*(?:np\.)?bool|astype\s*\(\s*(?:np\.)?bool)"
        unsafe_forward = re.compile(re.escape(q_name) + r".{0,160}" + bool_cast)
        unsafe_reverse = re.compile(bool_cast + r".{0,160}" + re.escape(q_name))
        if unsafe_forward.search(compact) or unsafe_reverse.search(compact):
            self.add("lag1-semantics", path, "lag-1 q-values are cast to Boolean")

    def _validate_python(self, path: Path, text: str) -> None:
        try:
            tree = ast.parse(text, filename=str(path))
            compile(text, str(path), "exec")
        except (SyntaxError, ValueError) as exc:
            self.add("python", path, f"source does not compile: {exc}")
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in self._unsafe_python_calls:
                    self.add(
                        "identity",
                        path,
                        f"runtime machine-identity collection via {name}() is not allowed",
                    )

    def _validate_serialized_arrays(self, files: Sequence[Path]) -> None:
        for path in files:
            suffix = path.suffix.lower()
            if suffix == ".npz":
                self._validate_npz(path)
            elif suffix == ".npy":
                self._validate_npy(path)

    def _validate_npy(self, path: Path) -> None:
        try:
            value = np.load(path, allow_pickle=False)
        except Exception as exc:
            self.add("serialization", path, f"array does not load with allow_pickle=False: {exc}")
            return
        self._validate_array(path, path.stem, np.asarray(value))

    def _validate_npz(self, path: Path) -> None:
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = list(archive.files)
                if len(keys) != len(set(keys)):
                    self.add("serialization", path, "archive contains duplicate keys")
                for key in keys:
                    if re.fullmatch(r"(?:sig|pval)_lag\d+", key):
                        self.add("lag1-semantics", path, f"legacy ambiguous field {key!r} is not allowed")
                    try:
                        value = np.asarray(archive[key])
                    except Exception as exc:
                        self.add(
                            "serialization",
                            path,
                            f"field {key!r} requires pickle or cannot be read: {exc}",
                        )
                        continue
                    self._validate_array(path, key, value)
        except Exception as exc:
            self.add("serialization", path, f"archive does not load with allow_pickle=False: {exc}")

    def _validate_array(self, path: Path, key: str, value: np.ndarray) -> None:
        if value.dtype.hasobject:
            self.add("serialization", path, f"field {key!r} has object dtype")
            return
        # Missing values can be meaningful in reference atlases (for example,
        # neuron pairs that were not measured).  Probability arrays and all
        # public result arrays are held to stricter rules below; generic atlas
        # matrices may represent missingness with NaN, but never infinity.
        if value.dtype.kind in "fc" and np.any(np.isinf(value)):
            self.add("numerics", path, f"field {key!r} contains infinity")
        if value.dtype.kind in "fc" and (
            path.relative_to(self.root).parts[:1] == ("results",)
            or path.relative_to(self.root).parts[:1] == ("reference_snapshot",)
        ):
            if not np.all(np.isfinite(value)):
                self.add("numerics", path, f"field {key!r} contains NaN or infinity")
        if key.startswith(("p_value_", "q_value_")):
            if value.dtype.kind not in "fiu":
                self.add("numerics", path, f"field {key!r} must be numeric")
            elif value.size and (np.min(value) < 0.0 or np.max(value) > 1.0):
                self.add("numerics", path, f"field {key!r} must lie in [0, 1]")
        if key.startswith("significant_lag"):
            if value.dtype.kind != "b":
                self.add("lag1-semantics", path, f"field {key!r} must have Boolean dtype")
        if key.startswith("q_value_") and value.dtype.kind == "b":
            self.add("lag1-semantics", path, f"field {key!r} cannot have Boolean dtype")
        if value.dtype.kind in "US":
            for item in value.reshape(-1):
                self._scan_sensitive_text(path, str(item))

    def _validate_json_files(self, files: Sequence[Path]) -> None:
        for path in files:
            if path.suffix.lower() != ".json":
                continue
            try:
                _strict_json(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self.add("json", path, f"invalid strict JSON: {exc}")

    def _validate_csv_files(self, files: Sequence[Path]) -> None:
        for path in files:
            if path.suffix.lower() not in {".csv", ".tsv"}:
                continue
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            relative = path.relative_to(self.root).as_posix()
            try:
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    raw_rows = list(csv.reader(handle, delimiter=delimiter, strict=True))
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                self.add("table", path, f"could not parse delimited text: {exc}")
                continue
            if not raw_rows:
                self.add("table", path, "table is empty")
                continue
            header = raw_rows[0]
            headerless_modulatory = "reference_data/modulatory_atlas/edge_lists" in relative
            index_column = (
                relative.startswith("reference_snapshot/phase_analysis/comparison/")
                and header
                and not header[0].strip()
            )
            if not header or (any(not column.strip() for column in header) and not index_column):
                self.add("table", path, "header contains a blank column name")
                continue
            if len(header) != len(set(header)) and not headerless_modulatory:
                self.add("table", path, "header contains duplicate column names")
            bad_rows = [index for index, row in enumerate(raw_rows[1:], start=2) if len(row) != len(header)]
            if bad_rows:
                self.add("table", path, f"row widths differ from header at lines {bad_rows[:5]}")
                continue
            if not raw_rows[1:]:
                self.add("table", path, "table has a header but no data rows")
            if headerless_modulatory:
                # The upstream atlas edge lists intentionally have no header;
                # validate rectangularity above without treating row one as a
                # public result schema.
                continue
            rows = [dict(zip(header, row)) for row in raw_rows[1:]]
            self._csv_cache[relative] = rows
            required = REQUIRED_CSV_COLUMNS.get(relative)
            if required:
                missing = sorted(required - set(header))
                if missing:
                    self.add("schema", path, f"required columns are missing: {missing}")
            if any(re.fullmatch(r"(?:sig|pval)_lag\d+", column) for column in header):
                self.add("lag1-semantics", path, "table uses a legacy ambiguous lag field")
            for line_number, row in enumerate(rows, start=2):
                for column, raw in row.items():
                    value = raw.strip()
                    if not value:
                        continue
                    if value.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
                        self.add("numerics", path, f"non-finite token in {column!r} at line {line_number}")
                    elif self._looks_numeric_column(column):
                        try:
                            number = float(value)
                            if not math.isfinite(number):
                                raise ValueError
                        except ValueError:
                            self.add("schema", path, f"non-numeric value in {column!r} at line {line_number}")
                    elif self._looks_boolean_column(column) and value.lower() not in {"0", "1", "false", "true"}:
                        self.add("schema", path, f"non-Boolean value in {column!r} at line {line_number}")

    @staticmethod
    def _looks_boolean_column(column: str) -> bool:
        lower = column.lower()
        return (
            lower.startswith(("is_", "has_", "passes_", "recovers_"))
            or lower.endswith(("_defined", "_is_null", "_is_tied", "_comparable"))
        )

    @staticmethod
    def _looks_numeric_column(column: str) -> bool:
        lower = column.lower()
        if lower in {
            "lag", "seed", "replicate", "model_seed", "training_seed", "time_s",
            "f1", "auroc", "auprc", "spearman", "jaccard", "top_k", "count",
            "frequency", "minimum", "maximum", "mean", "sd", "q025", "q975",
        }:
            return True
        if lower in {"hidden_indices", "observed_indices", "observed_peak_lags", "modal_peak_lags"}:
            return False
        return lower.startswith(("n_", "delta_", "f1_", "auroc_", "auprc_")) or lower.endswith(
            (
                "_mean", "_median", "_sd", "_std", "_min", "_max", "_low", "_high",
                "_lag", "_s", "_f1", "_auroc", "_auprc", "_spearman", "_correlation",
                "_fraction", "_rate", "_count", "_edges", "_error", "_rmse", "_value",
            )
        )

    def _validate_paper_artifact(self) -> None:
        path = self.root / "results/paper/sbtg_lag_matrices.npz"
        if not path.is_file():
            return
        expected = {"neuron_names", "lags", "mu_hat_lag1", "p_value_lag1", "q_value_lag1"}
        for lag in PAPER_LAGS[1:]:
            expected.update({f"mu_hat_lag{lag}", f"p_value_lag{lag}", f"significant_lag{lag}"})
        try:
            with np.load(path, allow_pickle=False) as archive:
                actual = set(archive.files)
                if actual != expected:
                    missing = sorted(expected - actual)
                    extra = sorted(actual - expected)
                    self.add("paper-artifact", path, f"unexpected schema; missing={missing}, extra={extra}")
                    return
                lags = tuple(int(value) for value in np.asarray(archive["lags"]).tolist())
                if lags != PAPER_LAGS:
                    self.add("paper-artifact", path, f"lags must be {PAPER_LAGS}, found {lags}")
                names = np.asarray(archive["neuron_names"])
                if names.dtype.kind not in "US" or names.shape != (80,):
                    self.add("paper-artifact", path, "neuron_names must be a Unicode vector of length 80")
                elif len(set(str(value) for value in names)) != 80:
                    self.add("paper-artifact", path, "neuron_names must be unique")
                for lag in PAPER_LAGS:
                    mu = np.asarray(archive[f"mu_hat_lag{lag}"])
                    p_value = np.asarray(archive[f"p_value_lag{lag}"])
                    if mu.shape != (80, 80) or p_value.shape != mu.shape:
                        self.add("paper-artifact", path, f"lag {lag} matrices must have shape (80, 80)")
                    if lag > 1:
                        significant = np.asarray(archive[f"significant_lag{lag}"])
                        if significant.shape != mu.shape or significant.dtype.kind != "b":
                            self.add("paper-artifact", path, f"lag {lag} significance must be an 80x80 Boolean mask")
                q_value = np.asarray(archive["q_value_lag1"])
                if q_value.shape != (80, 80) or q_value.dtype.kind not in "fiu":
                    self.add("lag1-semantics", path, "q_value_lag1 must be a numeric 80x80 matrix")
                elif not np.all(np.isfinite(q_value)) or np.min(q_value) < 0.0 or np.max(q_value) > 1.0:
                    self.add("lag1-semantics", path, "q_value_lag1 must be finite and lie in [0, 1]")
        except Exception as exc:
            self.add("paper-artifact", path, f"could not validate main artifact: {exc}")

    def _validate_numerical_summaries(self) -> None:
        self._check_csv_anchor(
            "results/paper/evaluation/structural_reference_metrics.csv",
            {"method": "SBTG", "lag": "1"},
            {"auroc": 0.5806514091904424, "f1": 0.3375214163335237},
        )
        self._check_csv_anchor(
            "results/paper/evaluation/functional_reference_metrics.csv",
            {"method": "SBTG", "lag": "1"},
            {
                "n_neurons": 67.0,
                "auroc": 0.6302367212782362,
                "auprc": 0.30025857600226885,
                "spearman": 0.29855833587077785,
                "f1": 0.3438320209973753,
                "n_evaluated": 1646.0,
                "n_positive": 273.0,
                "prevalence": 0.16585662211421628,
            },
        )
        self._check_csv_anchor(
            "results/robustness/structured_noise/paired_summary.csv",
            {"covariance_kind": "ar1", "target_mean_abs_corr": "0.2", "lag": "1"},
            {
                "delta_significant_edges_mean": 8.0,
                "delta_precision_significant_mean": -0.03414820958398701,
            },
        )
        self._check_csv_anchor(
            "results/robustness/structured_noise/paired_summary.csv",
            {"covariance_kind": "graph_diffusion", "target_mean_abs_corr": "0.2", "lag": "2"},
            {"delta_significant_edges_mean": 6.5},
        )
        self._check_csv_anchor(
            "results/robustness/method_agreement/edge_rank_agreement.csv",
            {"left": "SBTG", "right": "zero-lag Pearson"},
            {
                "spearman_abs_weight": 0.1744147934660587,
                "shared_edges": 46.0,
                "jaccard": 0.2987012987012987,
            },
        )
        phase_anchors = {
            "Baseline": 1.1601896988507783,
            "Steady": 1.1908345357546732,
            "On": 1.2506081932692663,
            "Off": 1.240250922073417,
        }
        for phase, value in phase_anchors.items():
            self._check_csv_anchor(
                "results/robustness/phase_duration/exploratory_endpoint/lag5_lag1_phase_summary.csv",
                {"phase": phase},
                {"bh_mean": value},
            )
        profile_anchors = {
            "dopamine": (9.111560405459773, 3.0),
            "serotonin": (13.706638955403612, 20.0),
            "tyramine": (7.500140486165516, 2.0),
            "octopamine": (7.7152175014259345, 3.0),
        }
        for network, (center, peak) in profile_anchors.items():
            self._check_csv_anchor(
                "results/robustness/lag_profiles/production_lineage_lag2plus/published_reference_metrics.csv",
                {"network": network},
                {"center_of_mass_lag": center, "peak_lag": peak},
            )

        summary_path = self.root / "results/robustness/residual_dependence/summary.json"
        if summary_path.is_file():
            try:
                summary = _strict_json(summary_path)
                if not isinstance(summary, dict):
                    raise TypeError("summary must be a JSON object")
                expected = {
                    "n_composite_rows": 20.0,
                    "n_neurons": 80.0,
                    "n_residual_timepoints": 18166.0,
                    "var_order": 20.0,
                    "mean_abs_offdiagonal_correlation": 0.024710816832680806,
                }
                for key, target in expected.items():
                    self._check_number(summary_path, key, summary.get(key), target)
            except Exception as exc:
                self.add("summary", summary_path, f"could not validate numerical summary: {exc}")

        metrics = self._csv_cache.get("results/robustness/structured_noise/metrics.csv")
        if metrics is not None and len(metrics) != 800:
            self.add("summary", "results/robustness/structured_noise/metrics.csv", "expected 800 rows")

    def _check_csv_anchor(
        self,
        relative: str,
        selector: Mapping[str, str],
        expected: Mapping[str, float],
    ) -> None:
        rows = self._csv_cache.get(relative)
        if rows is None:
            return
        matches = [row for row in rows if all(row.get(key) == value for key, value in selector.items())]
        if len(matches) != 1:
            self.add("summary", relative, f"expected one anchor row for {dict(selector)}, found {len(matches)}")
            return
        for key, target in expected.items():
            self._check_number(self.root / relative, key, matches[0].get(key), target)

    def _check_number(self, path: Path, field: str, raw: object, target: float) -> None:
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            self.add("summary", path, f"anchor field {field!r} is not numeric")
            return
        tolerance = max(1e-12, abs(target) * 1e-10)
        if not math.isfinite(value) or not math.isclose(value, target, rel_tol=1e-10, abs_tol=tolerance):
            self.add("summary", path, f"anchor {field!r} changed: expected {target}, found {value}")

    def _validate_indexes(self, regular_files: Sequence[Path]) -> None:
        expected_paths = sorted(
            path.relative_to(self.root).as_posix()
            for path in regular_files
            if path.relative_to(self.root).as_posix() not in INDEX_FILES
        )
        manifest_hashes = self._validate_manifest(expected_paths)
        checksum_hashes = self._validate_checksums(expected_paths)
        if manifest_hashes is not None and checksum_hashes is not None and manifest_hashes != checksum_hashes:
            self.add("indexes", "", "manifest and checksum index disagree")

    def _validate_manifest(self, expected_paths: Sequence[str]) -> dict[str, str] | None:
        path = self.root / MANIFEST_NAME
        if not path.is_file():
            self.add("manifest", MANIFEST_NAME, "manifest is missing")
            return None
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != MANIFEST_HEADER:
                    self.add("manifest", path, f"header must be exactly {MANIFEST_HEADER}")
                    return None
                rows = list(reader)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            self.add("manifest", path, f"could not parse manifest: {exc}")
            return None
        listed = [row["path"] for row in rows]
        if listed != sorted(listed):
            self.add("manifest", path, "paths must be sorted")
        if len(listed) != len(set(listed)):
            self.add("manifest", path, "paths must be unique")
        if listed != list(expected_paths):
            missing = sorted(set(expected_paths) - set(listed))
            extra = sorted(set(listed) - set(expected_paths))
            self.add("manifest", path, f"coverage mismatch; missing={missing}, extra={extra}")
        hashes: dict[str, str] = {}
        for row in rows:
            relative = row["path"]
            if not self._safe_relative_path(relative):
                self.add("manifest", path, f"unsafe path entry {relative!r}")
                continue
            digest = row["sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                self.add("manifest", path, f"invalid SHA-256 for {relative!r}")
                continue
            if not row["description"].strip():
                self.add("manifest", path, f"blank description for {relative!r}")
            target = self.root / relative
            if not target.is_file():
                continue
            try:
                declared_size = int(row["size_bytes"])
            except ValueError:
                self.add("manifest", path, f"invalid size for {relative!r}")
                continue
            if declared_size != target.stat().st_size:
                self.add("manifest", path, f"size mismatch for {relative!r}")
            actual = sha256_file(target)
            if actual != digest:
                self.add("manifest", path, f"digest mismatch for {relative!r}")
            hashes[relative] = digest
        return hashes

    def _validate_checksums(self, expected_paths: Sequence[str]) -> dict[str, str] | None:
        path = self.root / CHECKSUM_NAME
        if not path.is_file():
            self.add("checksums", CHECKSUM_NAME, "checksum index is missing")
            return None
        hashes: dict[str, str] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            self.add("checksums", path, f"could not read checksum index: {exc}")
            return None
        for line_number, line in enumerate(lines, start=1):
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if not match:
                self.add("checksums", path, f"invalid format at line {line_number}")
                continue
            digest, relative = match.groups()
            if not self._safe_relative_path(relative):
                self.add("checksums", path, f"unsafe path at line {line_number}")
                continue
            if relative in hashes:
                self.add("checksums", path, f"duplicate path {relative!r}")
            hashes[relative] = digest
        listed = list(hashes)
        if listed != sorted(listed):
            self.add("checksums", path, "paths must be sorted")
        if listed != list(expected_paths):
            missing = sorted(set(expected_paths) - set(listed))
            extra = sorted(set(listed) - set(expected_paths))
            self.add("checksums", path, f"coverage mismatch; missing={missing}, extra={extra}")
        for relative, digest in hashes.items():
            target = self.root / relative
            if target.is_file() and sha256_file(target) != digest:
                self.add("checksums", path, f"digest mismatch for {relative!r}")
        return hashes

    @staticmethod
    def _safe_relative_path(value: str) -> bool:
        if not value or "\\" in value:
            return False
        path = PurePosixPath(value)
        return not path.is_absolute() and ".." not in path.parts and path.as_posix() == value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="release root (defaults to the parent of tools/)",
    )
    parser.add_argument(
        "--root",
        dest="root_option",
        type=Path,
        default=None,
        help="named form of the release-root argument",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="skip manifest/checksum validation while assembling a release",
    )
    parser.add_argument(
        "--skip-layout",
        action="store_true",
        help="run generic safety checks without requiring the SBTG directory layout",
    )
    parser.add_argument("--max-file-mib", type=float, default=25.0)
    parser.add_argument("--max-total-mib", type=float, default=100.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.root is not None and args.root_option is not None:
        raise SystemExit("provide the release root either positionally or with --root, not both")
    root = args.root_option or args.root or Path(__file__).resolve().parents[1]
    validator = ReleaseValidator(
        root,
        require_layout=not args.skip_layout,
        require_indexes=not args.skip_indexes,
        max_file_bytes=int(args.max_file_mib * 1024 * 1024),
        max_total_bytes=int(args.max_total_mib * 1024 * 1024),
    )
    issues = validator.run()
    if issues:
        print(f"Release validation failed with {len(issues)} issue(s):", file=sys.stderr)
        for issue in issues:
            print(f"  {issue.render()}", file=sys.stderr)
        return 1
    print("Release validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
