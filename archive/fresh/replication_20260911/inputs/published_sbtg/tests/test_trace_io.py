from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.utils.io import load_trace_collection, save_trace_collection


class TraceArchiveTests(unittest.TestCase):
    def test_round_trip_variable_lengths_and_missing_values(self) -> None:
        traces = [
            np.array([[1.0, np.nan], [2.0, 3.0]]),
            np.empty((0, 2)),
            np.array([[4.0, 5.0]]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "traces.npz"
            save_trace_collection(path, traces)

            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"values", "missing", "offsets"})
                self.assertFalse(any(archive[key].dtype.hasobject for key in archive.files))
                self.assertTrue(np.all(np.isfinite(archive["values"])))

            restored = load_trace_collection(path)

        self.assertEqual([trace.shape for trace in restored], [(2, 2), (0, 2), (1, 2)])
        np.testing.assert_allclose(restored[0], traces[0], equal_nan=True)
        np.testing.assert_allclose(restored[2], traces[2])

    def test_rejects_inconsistent_feature_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "traces.npz"
            with self.assertRaisesRegex(ValueError, "shared feature count"):
                save_trace_collection(
                    path,
                    [np.zeros((2, 2)), np.zeros((2, 3))],
                )


if __name__ == "__main__":
    unittest.main()
