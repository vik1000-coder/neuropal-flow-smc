"""Focused regression tests for the read-only public-release validator."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = RELEASE_ROOT / "tools" / "validate_release.py"
SPEC = importlib.util.spec_from_file_location("release_validator", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import bootstrap guard
    raise ImportError(f"Could not load {VALIDATOR_PATH}")
validator_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator_module
SPEC.loader.exec_module(validator_module)
ReleaseValidator = validator_module.ReleaseValidator
sha256_file = validator_module.sha256_file


class ReleaseValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_generic(self, **kwargs):
        settings = {
            "require_layout": False,
            "require_indexes": False,
            "max_file_bytes": 1024 * 1024,
            "max_total_bytes": 2 * 1024 * 1024,
        }
        settings.update(kwargs)
        return ReleaseValidator(self.root, **settings).run()

    def categories(self, issues) -> set[str]:
        return {issue.category for issue in issues}

    def test_clean_portable_files_pass_generic_checks(self) -> None:
        (self.root / "script.py").write_text("value = 3\n", encoding="utf-8")
        np.save(self.root / "values.npy", np.arange(4, dtype=np.float64))
        self.assertEqual(self.run_generic(), [])

    def test_machine_path_and_internal_marker_are_rejected(self) -> None:
        machine_path = "/" + "Users" + "/example/work/run.py"
        marker = "review" + "er"
        (self.root / "notes.txt").write_text(
            f"input={machine_path}\nfor the {marker}\n",
            encoding="utf-8",
        )
        issues = self.run_generic()
        self.assertTrue({"identity", "internal-language"} <= self.categories(issues))

    def test_archive_and_large_file_are_rejected(self) -> None:
        (self.root / "bundle.zip").write_bytes(b"not an archive")
        issues = self.run_generic(max_file_bytes=4)
        self.assertTrue({"junk", "size"} <= self.categories(issues))

    def test_object_array_and_legacy_field_are_rejected(self) -> None:
        legacy_key = "sig" + "_lag1"
        np.savez(self.root / "unsafe.npz", **{legacy_key: np.array([{"x": 1}], dtype=object)})
        issues = self.run_generic()
        self.assertTrue({"serialization", "lag1-semantics"} <= self.categories(issues))

    def test_probability_range_and_boolean_significance_are_checked(self) -> None:
        np.savez(
            self.root / "invalid.npz",
            q_value_lag1=np.array([0.1, 1.1]),
            significant_lag2=np.array([0, 1], dtype=np.int64),
        )
        issues = self.run_generic()
        self.assertTrue({"numerics", "lag1-semantics"} <= self.categories(issues))

    def test_direct_q_value_boolean_cast_is_rejected(self) -> None:
        q_name = "q_value_" + "lag1"
        source = f"value = archive[{q_name!r}].astype(bool)\n"
        (self.root / "unsafe_cast.py").write_text(source, encoding="utf-8")
        issues = self.run_generic()
        self.assertIn("lag1-semantics", self.categories(issues))

    def test_invalid_python_and_malformed_csv_are_rejected(self) -> None:
        (self.root / "broken.py").write_text("def broken(:\n", encoding="utf-8")
        (self.root / "broken.csv").write_text("a,b\n1\n", encoding="utf-8")
        issues = self.run_generic()
        self.assertTrue({"python", "table"} <= self.categories(issues))

    def test_text_files_reject_invisible_c0_controls(self) -> None:
        (self.root / "invalid.tsv").write_text(
            "source\ttarget\nA\tB\x08\n",
            encoding="utf-8",
        )
        issues = self.run_generic()
        encoding_issues = [issue for issue in issues if issue.category == "encoding"]
        self.assertEqual(len(encoding_issues), 1)
        self.assertIn("U+0008", encoding_issues[0].message)

    def test_markdown_pipe_tables_require_compilable_structure(self) -> None:
        valid = self.root / "valid.md"
        valid.write_text(
            """# Tables

Name | Expression | Note
:--- | :---: | ---:
first | `a | b` | escaped \\| pipe

```text
| this | is not checked |
| because | it is fenced |
```
""",
            encoding="utf-8",
        )
        self.assertEqual(self.run_generic(), [])

        valid.unlink()
        (self.root / "invalid.md").write_text(
            """| Name |  |
| value | note |
| too | many | cells |
""",
            encoding="utf-8",
        )
        issues = self.run_generic()
        markdown_issues = [issue for issue in issues if issue.category == "markdown-table"]
        self.assertGreaterEqual(len(markdown_issues), 3)
        messages = "\n".join(issue.message for issue in markdown_issues)
        self.assertIn("blank header", messages)
        self.assertIn("separator row", messages)
        self.assertIn("expected 2", messages)

    def test_manifest_and_checksums_verify_exact_coverage(self) -> None:
        payload = self.root / "data.txt"
        payload.write_text("portable\n", encoding="utf-8")
        digest = sha256_file(payload)
        with (self.root / "release_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("path", "sha256", "size_bytes", "description"))
            writer.writerow(("data.txt", digest, payload.stat().st_size, "test payload"))
        (self.root / "checksums.sha256").write_text(
            f"{digest}  data.txt\n",
            encoding="utf-8",
        )
        issues = ReleaseValidator(
            self.root,
            require_layout=False,
            require_indexes=True,
            max_file_bytes=1024 * 1024,
            max_total_bytes=2 * 1024 * 1024,
        ).run()
        self.assertEqual(issues, [])

        payload.write_text("tampered\n", encoding="utf-8")
        issues = ReleaseValidator(
            self.root,
            require_layout=False,
            require_indexes=True,
            max_file_bytes=1024 * 1024,
            max_total_bytes=2 * 1024 * 1024,
        ).run()
        self.assertTrue({"manifest", "checksums"} <= self.categories(issues))

    def test_main_paper_artifact_has_unambiguous_lag1_schema(self) -> None:
        target = self.root / "results" / "paper"
        target.mkdir(parents=True)
        arrays: dict[str, np.ndarray] = {
            "neuron_names": np.asarray([f"N{i:02d}" for i in range(80)]),
            "lags": np.asarray((1, 2, 3, 5, 8, 10, 15, 20), dtype=np.int64),
            "mu_hat_lag1": np.zeros((80, 80)),
            "p_value_lag1": np.full((80, 80), 0.5),
            "q_value_lag1": np.full((80, 80), 0.5),
        }
        for lag in (2, 3, 5, 8, 10, 15, 20):
            arrays[f"mu_hat_lag{lag}"] = np.zeros((80, 80))
            arrays[f"p_value_lag{lag}"] = np.full((80, 80), 0.5)
            arrays[f"significant_lag{lag}"] = np.zeros((80, 80), dtype=bool)
        np.savez(target / "sbtg_lag_matrices.npz", **arrays)

        checker = ReleaseValidator(self.root, require_layout=False, require_indexes=False)
        checker._validate_paper_artifact()
        self.assertEqual(checker.issues, [])

        arrays["significant_lag1"] = np.zeros((80, 80), dtype=bool)
        np.savez(target / "sbtg_lag_matrices.npz", **arrays)
        checker = ReleaseValidator(self.root, require_layout=False, require_indexes=False)
        checker._validate_paper_artifact()
        self.assertIn("paper-artifact", self.categories(checker.issues))


if __name__ == "__main__":
    unittest.main()
