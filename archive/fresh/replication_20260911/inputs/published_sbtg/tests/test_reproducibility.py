"""Regression tests for portable provenance helpers."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.evaluation.prepare_merged_results import (
    DEFAULT_MODULATORY_EDGES,
    _load_modulatory_references,
    _reference_name,
)
from pipeline.utils.io import load_structural_connectome
from pipeline.utils.reproducibility import compute_data_hash


class ReproducibilityTests(unittest.TestCase):
    def test_modulatory_name_normalization_preserves_singletons(self) -> None:
        self.assertEqual(_reference_name("PQR"), "PQR")
        self.assertEqual(_reference_name("AQR"), "AQR")
        self.assertEqual(_reference_name(" rim "), "RIM")

    def test_modulatory_reference_uses_complete_class_level_labels(self) -> None:
        with np.load(
            Path(__file__).parents[1] / "results/paper/sbtg_lag_matrices.npz",
            allow_pickle=False,
        ) as archive:
            names = [str(value) for value in archive["neuron_names"]]
        references = _load_modulatory_references(DEFAULT_MODULATORY_EDGES, names)
        self.assertEqual(
            {name: int(np.count_nonzero(matrix)) for name, matrix in references.items()},
            {"dopamine": 46, "serotonin": 22, "tyramine": 33, "octopamine": 11},
        )

    def test_data_hash_uses_contents_not_mtime_or_absolute_parent(self) -> None:
        with tempfile.TemporaryDirectory() as first_temp, tempfile.TemporaryDirectory() as second_temp:
            first = Path(first_temp) / "input.bin"
            second = Path(second_temp) / "input.bin"
            first.write_bytes(b"same scientific input")
            second.write_bytes(first.read_bytes())

            original = compute_data_hash([first])
            os.utime(first, (1, 1))
            self.assertEqual(compute_data_hash([first]), original)
            self.assertEqual(compute_data_hash([second]), original)

            second.write_bytes(b"changed scientific input")
            self.assertNotEqual(compute_data_hash([second]), original)

    def test_structural_metadata_does_not_expose_absolute_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "connectome"
            directory.mkdir()
            np.save(directory / "A_struct.npy", np.array([[0, 1], [0, 0]]))
            (directory / "nodes.json").write_text(json.dumps(["A", "B"]))

            _, _, metadata = load_structural_connectome(directory)
            self.assertEqual(metadata["source_dir"], "connectome")
            self.assertFalse(Path(metadata["source_dir"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
