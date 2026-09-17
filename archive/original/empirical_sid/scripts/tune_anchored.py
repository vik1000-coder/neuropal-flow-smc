from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from empirical_sid.dgps import make_static_dgp  # noqa: E402
from empirical_sid.estimators import PolynomialScoreModel  # noqa: E402
from empirical_sid.metrics import nrmse  # noqa: E402
from empirical_sid.runner import _score_typed_effect, atomic_csv, atomic_json, load_config, resolve_output  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    output = resolve_output(config)
    calibration = json.loads(
        (output / "development" / "information_calibration.json").read_text()
    )["mechanisms"]
    mechanisms = ["m1_mean", "m3_cubic", "m3_local", "m4_quartic"]
    schedules = {
        "multi_full": (0.03, 0.06, 0.12, 0.25),
        "multi_low": (0.03, 0.06, 0.12),
        "single_primary": (0.12,),
    }
    rows = []
    for seed in config["seeds"]["development"]:
        for mechanism in mechanisms:
            rng = np.random.default_rng(int(seed))
            dgp = make_static_dgp(mechanism, calibration[mechanism]["amplitude"])
            h = rng.uniform(-1.2, 1.2, int(config["data"]["n_train"]))
            y = dgp.sample(h, rng)[:, 0]
            eval_h = rng.uniform(-0.8, 0.8, 48)
            for ridge in [0.001, 0.01, 0.1, 1.0]:
                for schedule_name, schedule in schedules.items():
                    fit_rng = np.random.default_rng(int(seed) + 900_000)
                    model = PolynomialScoreModel("anchored", ridge=ridge).fit(
                        h, y, fit_rng, sigma_ladder=schedule
                    )
                    estimate, truth, residual = _score_typed_effect(model, dgp, eval_h, 0.12)
                    rows.append(
                        {
                            "dgp_seed": seed,
                            "dgp_family": mechanism,
                            "ridge": ridge,
                            "noise_schedule": schedule_name,
                            "typed_nrmse": nrmse(estimate, truth),
                            "hodge_residual": residual,
                        }
                    )
    frame = pd.DataFrame(rows)
    atomic_csv(output / "development" / "anchored_tuning_seed_level.csv", frame)
    summary = (
        frame.groupby(["ridge", "noise_schedule"], as_index=False)
        .agg(
            mean_typed_nrmse=("typed_nrmse", "mean"),
            median_typed_nrmse=("typed_nrmse", "median"),
            p90_typed_nrmse=("typed_nrmse", lambda values: float(np.quantile(values, 0.9))),
            mean_hodge_residual=("hodge_residual", "mean"),
        )
        .sort_values(["mean_typed_nrmse", "p90_typed_nrmse"])
    )
    atomic_csv(output / "development" / "anchored_tuning_summary.csv", summary)
    winner = summary.iloc[0].to_dict()
    selection = {
        "status": "frozen_before_confirmatory",
        "selection_rule": "minimum mean typed NRMSE across four mechanisms and eight development seeds; one global setting",
        "selected_ridge": float(winner["ridge"]),
        "selected_noise_schedule": str(winner["noise_schedule"]),
        "selected_summary": winner,
        "test_seeds_used": False,
    }
    atomic_json(output / "development" / "anchored_selection.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
