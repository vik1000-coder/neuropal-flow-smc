"""Unit tests for profile-stability and phase-summary pure helpers."""

from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from experiments.lag_profile_stability import (  # noqa: E402
    ALL8_DIAGNOSTIC_SCOPE,
    PRIMARY_SCOPE,
    bootstrap_seed_ensemble_curves,
    configure_analysis_scope,
    pearson_correlation,
    spearman_correlation,
    weighted_profile_moments,
)
from experiments.lag_profile_fit import (  # noqa: E402
    DEFAULT_DATA_SEED,
    fit_label,
)
from experiments.lag_profile_summary import (  # noqa: E402
    default_full_control_roots,
    load_external_full_control,
    load_fit,
)
from experiments.snapshot import (  # noqa: E402
    EXPECTED_MULTILAG_SHA256,
    derive_seed,
    sha256,
)
from experiments.common import LAGS  # noqa: E402
from experiments.phase_duration_summary import (  # noqa: E402
    compute_phase_indices,
    contrast_record,
)
from experiments.phase_endpoint_summary import calculate  # noqa: E402


class LagProfileHelperTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_analysis_scope(PRIMARY_SCOPE)

    def test_weighted_profile_moments(self) -> None:
        result = weighted_profile_moments([1, 2, 3], [0, 1, 1])
        self.assertTrue(result["moments_defined"])
        self.assertAlmostEqual(float(result["f1_sum"]), 2.0)
        self.assertAlmostEqual(float(result["center_of_mass_lag"]), 2.5)
        self.assertAlmostEqual(float(result["weighted_sd_lag"]), 0.5)

        empty = weighted_profile_moments([1, 2], [0, 0])
        self.assertFalse(empty["moments_defined"])
        self.assertTrue(math.isnan(float(empty["center_of_mass_lag"])))
        with self.assertRaises(ValueError):
            weighted_profile_moments([1, 2], [1, -1])

    def test_profile_correlations(self) -> None:
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(spearman_correlation([1, 1, 3], [4, 4, 9]), 1.0)
        self.assertTrue(math.isnan(pearson_correlation([1, 1], [2, 3])))

    def test_scope_validation(self) -> None:
        configure_analysis_scope(ALL8_DIAGNOSTIC_SCOPE)
        with self.assertRaises(ValueError):
            configure_analysis_scope("unknown")

    def test_bootstrap_seed_repeats_are_averaged_not_pooled(self) -> None:
        rows = []
        for seed, offset in ((42, 0.0), (43, 0.2)):
            for lag in (2, 3, 5, 8, 10, 15, 20):
                rows.append(
                    {
                        "sample_kind": "bootstrap",
                        "replicate": 0,
                        "model_seed": seed,
                        "network": "example",
                        "lag": lag,
                        "time_s": lag * 0.25,
                        "f1": lag / 100.0 + offset,
                        "best_threshold": 0.5 + offset,
                    }
                )
        ensemble = bootstrap_seed_ensemble_curves(pd.DataFrame(rows))
        self.assertEqual(len(ensemble), 7)
        self.assertTrue((ensemble["n_model_seeds_averaged"] == 2).all())
        lag_two = ensemble.loc[ensemble["lag"] == 2].iloc[0]
        self.assertAlmostEqual(float(lag_two["f1"]), 0.12)
        self.assertAlmostEqual(float(ensemble["profile_weight"].sum()), 1.0)

    def test_campaign_full_controls_match_summary_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "prepared_data"
            dataset_dir.mkdir()
            np.savez_compressed(
                dataset_dir / "traces.npz",
                values=np.arange(4, dtype=float).reshape(2, 2),
            )
            model_seed = 42
            n_rows = 2
            names = ["A", "B"]
            label = fit_label("full", -1, model_seed)
            control_root = root / label
            self.assertEqual(
                default_full_control_roots(root)[model_seed], control_root
            )

            for lag in LAGS:
                lag_dir = control_root / f"lag_{lag:02d}"
                lag_dir.mkdir(parents=True)
                fold_seeds = np.asarray(
                    [
                        derive_seed(model_seed, 20260123, lag, fold)
                        for fold in range(5)
                    ],
                    dtype=np.uint32,
                )
                np.savez_compressed(
                    lag_dir / "result.npz",
                    mu_hat=np.full((2, 2), lag, dtype=float),
                    neuron_names=np.asarray(names),
                    lag=lag,
                    sample_kind="full",
                    replicate=-1,
                    model_seed=model_seed,
                    data_seed=DEFAULT_DATA_SEED,
                    source_row_indices=np.arange(n_rows, dtype=np.int64),
                    source_row_multiplicities=np.ones(n_rows, dtype=np.int64),
                    fold_seeds=fold_seeds,
                )
                manifest = {
                    "status": "complete",
                    "fit_label": label,
                    "sample_kind": "full",
                    "replicate": -1,
                    "lag": lag,
                    "model_seed": model_seed,
                    "data_seed": DEFAULT_DATA_SEED,
                    "source_row_indices": list(range(n_rows)),
                    "source_row_multiplicities": [1] * n_rows,
                    "n_source_rows": n_rows,
                    "n_sampled_rows_with_multiplicity": n_rows,
                    "n_neurons": len(names),
                    "n_folds": 5,
                    "device": "cpu",
                    "effective_batch_size": 256,
                    "dataset_sha256": sha256(dataset_dir / "traces.npz"),
                    "model_source_sha256": EXPECTED_MULTILAG_SHA256,
                    "fold_seeds": {
                        str(fold): int(value)
                        for fold, value in enumerate(fold_seeds)
                    },
                }
                (lag_dir / "manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True)
                )

            external, missing = load_external_full_control(
                control_root,
                model_seed,
                dataset_dir,
                n_rows,
                names,
                DEFAULT_DATA_SEED,
                "cpu",
                256,
            )
            self.assertEqual(missing, [])
            self.assertEqual(set(external), set(LAGS))

            direct, missing = load_fit(
                root,
                "full",
                -1,
                model_seed,
                DEFAULT_DATA_SEED,
                dataset_dir,
                n_rows,
                names,
                "cpu",
                256,
            )
            self.assertEqual(missing, [])
            self.assertEqual(set(direct), set(LAGS))


class PhaseSummaryHelperTests(unittest.TestCase):
    def test_phase_indices_drop_missing_pair_ratios_explicitly(self) -> None:
        common = {
            "source_kind": "new",
            "dataset_mode": "matched",
            "phase_code": "on",
            "phase": "On",
            "replicate": 0,
        }
        ratios = pd.DataFrame(
            [
                {**common, "significant_long_short_ratio": 0.5, "all_edges_long_short_ratio": 1.0},
                {**common, "significant_long_short_ratio": 1.5, "all_edges_long_short_ratio": 2.0},
                {**common, "significant_long_short_ratio": np.nan, "all_edges_long_short_ratio": 3.0},
            ]
        )
        result = compute_phase_indices(ratios).iloc[0]
        self.assertAlmostEqual(result["significant_long_short_ratio_pair_mean"], 1.0)
        self.assertEqual(result["significant_long_short_ratio_n_pairs"], 2)
        self.assertAlmostEqual(result["significant_long_short_ratio_fraction_above_one"], 0.5)
        self.assertAlmostEqual(result["all_edges_long_short_ratio_pair_median"], 2.0)

    def test_contrast_record_filters_nonfinite_references(self) -> None:
        result = contrast_record(2.0, np.array([1.0, np.nan, 3.0]))
        self.assertEqual(result["n_reference_replicates"], 2)
        self.assertAlmostEqual(result["reference_mean"], 2.0)
        self.assertAlmostEqual(result["event_minus_reference_mean"], 0.0)
        self.assertTrue(result["event_within_reference_range"])
        self.assertAlmostEqual(result["event_percentile_among_reference_replicates"], 0.5)
        self.assertEqual(contrast_record(np.nan, np.array([1.0])), {})

    def test_endpoint_calculation_averages_sampling_replicates_within_seed(self) -> None:
        phase_ratios = {"Baseline": 1.0, "Steady": 1.2, "On": 1.5, "Off": 1.4}
        rows = []
        for phase, ratio in phase_ratios.items():
            replicates = (0, 1) if phase in {"Baseline", "Steady"} else (0,)
            for seed in (42, 43, 44):
                for replicate in replicates:
                    for pair_index in range(9):
                        for lag in (1, 2, 3, 5):
                            value = ratio if lag == 5 else 1.0
                            rows.append(
                                {
                                    "phase": phase,
                                    "model_seed": seed,
                                    "replicate": replicate,
                                    "pair": f"P{pair_index}",
                                    "lag": lag,
                                    "mean_abs_mu_all": value,
                                    "mean_abs_mu_bh": value,
                                    "mean_abs_mu_by": value,
                                }
                            )
        by_seed, summary, contrasts = calculate(pd.DataFrame(rows))
        self.assertEqual(len(by_seed), 12)
        self.assertEqual(len(summary), 4)
        self.assertEqual(len(contrasts), 12)
        on = summary.loc[summary["phase"] == "On"].iloc[0]
        self.assertAlmostEqual(float(on["bh_mean"]), 1.5)
        baseline = by_seed.loc[by_seed["phase"] == "Baseline"]
        self.assertTrue((baseline["n_sampling_replicates"] == 2).all())
        on_vs_baseline = contrasts.loc[
            (contrasts["method"] == "bh")
            & (contrasts["event_phase"] == "On")
            & (contrasts["reference_phase"] == "Baseline")
        ].iloc[0]
        self.assertAlmostEqual(float(on_vs_baseline["mean_delta"]), 0.5)
        self.assertEqual(int(on_vs_baseline["positive_seed_count"]), 3)


if __name__ == "__main__":
    unittest.main()
