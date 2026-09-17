"""Scientific invariants for the public stationary VAR(2) generator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from pipeline.SyntheticTestingUtils import (  # noqa: E402
    companion_spectral_radius,
    generate_var_data,
    stabilize_var,
)


class StationaryVarTests(unittest.TestCase):
    def test_joint_stabilization_preserves_support_and_sign(self) -> None:
        first = np.array([[1.20, -0.25], [0.40, 0.0]])
        second = np.array([[0.60, 0.0], [-0.10, 0.45]])
        self.assertGreater(companion_spectral_radius(first, second), 1.0)

        stable_first, stable_second = stabilize_var(first, second, target=0.9)
        self.assertLessEqual(companion_spectral_radius(stable_first, stable_second), 0.900001)
        np.testing.assert_array_equal(stable_first != 0.0, first != 0.0)
        np.testing.assert_array_equal(stable_second != 0.0, second != 0.0)
        np.testing.assert_array_equal(np.sign(stable_first), np.sign(first))
        np.testing.assert_array_equal(np.sign(stable_second), np.sign(second))

    def test_already_stable_coefficients_are_unchanged_copies(self) -> None:
        first = np.array([[0.1]])
        second = np.array([[0.05]])
        stable_first, stable_second = stabilize_var(first, second, target=0.9)
        np.testing.assert_array_equal(stable_first, first)
        np.testing.assert_array_equal(stable_second, second)
        self.assertIsNot(stable_first, first)
        self.assertIsNot(stable_second, second)

    def test_invalid_shapes_and_targets_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            companion_spectral_radius(np.zeros((2, 3)), np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            companion_spectral_radius(np.zeros((2, 2)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            stabilize_var(np.zeros((2, 2)), np.zeros((2, 2)), target=1.0)

    def test_known_seed_generates_finite_bounded_traces(self) -> None:
        traces, truth = generate_var_data(
            n=12,
            T=600,
            m_stim=2,
            noise_level="low",
            seed=0,
        )
        self.assertEqual(set(truth), {1, 2})
        self.assertEqual(truth[1].shape, (12, 12))
        self.assertEqual(truth[2].shape, (12, 12))
        self.assertEqual(len(traces), 2)
        for trace in traces:
            self.assertEqual(trace.shape, (600, 12))
            self.assertTrue(np.all(np.isfinite(trace)))
            self.assertLess(float(np.max(np.abs(trace))), 100.0)


if __name__ == "__main__":
    unittest.main()
