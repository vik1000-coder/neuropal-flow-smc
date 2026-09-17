"""Regression tests for conversion to portable, pickle-free artifacts."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
EXPORT_PATH = RELEASE_ROOT / "tools" / "export_safe_results.py"
SPEC = importlib.util.spec_from_file_location("safe_result_export", EXPORT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import bootstrap guard
    raise ImportError(f"Could not load {EXPORT_PATH}")
export = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = export
SPEC.loader.exec_module(export)


class SafeExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_atlas_nonfinite_values_have_explicit_neutral_mapping(self) -> None:
        source = self.root / "atlas_source.npz"
        output = self.root / "nested" / "atlas.npz"
        np.savez(
            source,
            q=np.array([[np.nan, np.inf], [-np.inf, 0.2]]),
            q_eq=np.array([[np.inf, np.nan], [0.3, -np.inf]]),
            dff=np.array([[np.nan, np.inf], [-np.inf, -0.4]]),
            neuron_order=np.array(["A", "B"]),
        )
        export.convert_atlas(source, output)
        with np.load(output, allow_pickle=False) as result:
            np.testing.assert_array_equal(result["q"], [[1.0, 1.0], [1.0, 0.2]])
            np.testing.assert_array_equal(result["q_eq"], [[1.0, 1.0], [0.3, 1.0]])
            np.testing.assert_array_equal(result["dff"], [[0.0, 0.0], [0.0, -0.4]])

    def test_sbtg_conversion_creates_nested_config_directory(self) -> None:
        source = self.root / "source.npz"
        output = self.root / "arrays" / "result.npz"
        config = self.root / "metadata" / "nested" / "hyperparameters.json"
        np.savez(
            source,
            neuron_names=np.array(["A", "B"]),
            mu_hat_lag1=np.zeros((2, 2)),
            pval_lag1=np.ones((2, 2)),
            sig_lag1=np.full((2, 2), 0.5),
        )
        export.convert_sbtg(source, output, config, hybrid=True)
        self.assertTrue(config.is_file())
        with np.load(output, allow_pickle=False) as result:
            self.assertIn("q_value_lag1", result.files)
            self.assertNotIn("significant_lag1", result.files)

    def test_variable_length_traces_become_flat_values_and_offsets(self) -> None:
        source = self.root / "traces.npy"
        output = self.root / "portable" / "traces.npz"
        raw = np.empty(2, dtype=object)
        raw[0] = np.array([[1.0, np.nan], [2.0, 3.0]])
        raw[1] = np.array([[4.0, 5.0]])
        np.save(source, raw, allow_pickle=True)
        export.convert_traces(source, output)
        with np.load(output, allow_pickle=False) as result:
            np.testing.assert_array_equal(result["offsets"], [0, 2, 3])
            np.testing.assert_array_equal(result["values"], [[1.0, 0.0], [2.0, 3.0], [4.0, 5.0]])
            np.testing.assert_array_equal(
                result["missing"],
                [[False, True], [False, False], [False, False]],
            )


if __name__ == "__main__":
    unittest.main()
