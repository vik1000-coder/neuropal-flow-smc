"""Tests for deterministic construction of the two release indexes."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = RELEASE_ROOT / "tools" / "build_release_indexes.py"
SPEC = importlib.util.spec_from_file_location("release_index_builder", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import bootstrap guard
    raise ImportError(f"Could not load {BUILDER_PATH}")
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class ReleaseIndexBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_indexes_are_deterministic_sorted_and_complete(self) -> None:
        (self.root / "nested").mkdir()
        (self.root / "nested" / "table.csv").write_text("x\n1\n", encoding="utf-8")
        (self.root / "data.txt").write_text("value\n", encoding="utf-8")

        manifest, checksums, count = builder.build_indexes(self.root)
        self.assertEqual(count, 2)
        first_manifest = manifest.read_bytes()
        first_checksums = checksums.read_bytes()
        builder.build_indexes(self.root)
        self.assertEqual(manifest.read_bytes(), first_manifest)
        self.assertEqual(checksums.read_bytes(), first_checksums)

        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["path"] for row in rows], ["data.txt", "nested/table.csv"])
        self.assertTrue(all(row["description"] for row in rows))
        self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))
        self.assertNotIn(builder.MANIFEST_NAME, {row["path"] for row in rows})
        self.assertNotIn(builder.CHECKSUM_NAME, {row["path"] for row in rows})
        self.assertEqual(len(checksums.read_text(encoding="utf-8").splitlines()), 2)

    def test_existing_indexes_are_replaced_after_payload_change(self) -> None:
        payload = self.root / "data.txt"
        payload.write_text("first\n", encoding="utf-8")
        manifest, _, _ = builder.build_indexes(self.root)
        first = manifest.read_text(encoding="utf-8")
        payload.write_text("second value\n", encoding="utf-8")
        builder.build_indexes(self.root)
        second = manifest.read_text(encoding="utf-8")
        self.assertNotEqual(first, second)

    def test_symbolic_links_fail_closed(self) -> None:
        target = self.root / "target.txt"
        target.write_text("value\n", encoding="utf-8")
        link = self.root / "alias.txt"
        try:
            link.symlink_to(target.name)
        except (OSError, NotImplementedError):
            self.skipTest("symbolic links are unavailable")
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            builder.build_indexes(self.root)

    def test_path_descriptions_are_stable(self) -> None:
        self.assertEqual(builder.describe("LICENSE"), "Code and data-use notice")
        self.assertEqual(
            builder.describe("results/paper/metrics.csv"),
            "Validated result table for metrics",
        )
        self.assertEqual(
            builder.describe("tests/test_example.py"),
            "Regression test for test example",
        )


if __name__ == "__main__":
    unittest.main()
