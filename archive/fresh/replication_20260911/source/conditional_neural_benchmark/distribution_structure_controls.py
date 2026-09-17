"""Known-law positive/negative controls for the diagnostic, without fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from conditional_neural_benchmark.distribution_structure_analysis import paired_interval
from conditional_neural_benchmark.distribution_structure_runner import DEFAULT_RUN, atomic_json
from conditional_neural_benchmark.distribution_structure_scoring import score_samples, shuffle_coordinates, marginal_invariance


def run():
    rng = np.random.default_rng(8312026)
    groups, histories, samples, dimensions = 17, 64, 128, 4
    size = groups * histories
    output = DEFAULT_RUN / "synthetic_controls"
    output.mkdir(exist_ok=True)
    rows = []
    for name in ("independent_gaussian", "correlated_gaussian", "changing_variance", "student_shape"):
        covariance = np.eye(dimensions)
        if name == "correlated_gaussian":
            covariance = 0.8 * np.ones((dimensions, dimensions)) + 0.2 * np.eye(dimensions)
        root = np.linalg.cholesky(covariance)
        x = rng.normal(size=(size, samples, dimensions)) @ root.T
        y = rng.normal(size=(size, dimensions)) @ root.T
        if name == "changing_variance":
            scale = np.where(np.arange(size) % 2, 1.6, 0.4)
            x *= scale[:, None, None]
            y *= scale[:, None]
            alternative = rng.normal(size=x.shape) * np.sqrt(np.mean(scale ** 2))
        elif name == "student_shape":
            nu = 4.0
            x *= np.sqrt((nu - 2) / rng.chisquare(nu, size=(size, samples, 1)))
            y *= np.sqrt((nu - 2) / rng.chisquare(nu, size=(size, 1)))
            alternative = rng.normal(size=x.shape)
        else:
            alternative = shuffle_coordinates(x, 771)
        original = score_samples(x, y)
        counterfactual = score_samples(alternative, y)
        if name.endswith("gaussian"):
            marginal_invariance(original, counterfactual)
        for metric in ("energy", "variogram", "crps"):
            delta = (counterfactual[metric] - original[metric]).reshape(groups, histories).mean(axis=1)
            mean, low, high = paired_interval(delta)
            rows.append(dict(control=name, metric=metric, delta=mean, ci_low=low, ci_high=high))
    result = pd.DataFrame(rows)
    result.to_csv(output / "scores.csv", index=False)
    indexed = result.set_index(["control", "metric"])
    gates = {
        "independent_energy_small": abs(indexed.loc[("independent_gaussian", "energy"), "delta"]) < 0.01,
        "correlated_joint_energy_positive": indexed.loc[("correlated_gaussian", "energy"), "ci_low"] > 0,
        "correlated_variogram_positive": indexed.loc[("correlated_gaussian", "variogram"), "ci_low"] > 0,
        "correlated_crps_invariant": abs(indexed.loc[("correlated_gaussian", "crps"), "delta"]) < 1e-10,
        "changing_variance_crps_positive": indexed.loc[("changing_variance", "crps"), "ci_low"] > 0,
        "matched_student_crps_positive": indexed.loc[("student_shape", "crps"), "ci_low"] > 0,
    }
    atomic_json(output / "validation.json", {"status": "pass" if all(gates.values()) else "fail",
                "gates": {k: bool(v) for k, v in gates.items()}, "seed": 8312026,
                "groups": groups, "histories_per_group": histories, "samples": samples,
                "dimensions": dimensions, "meaning": "known-law diagnostic validation; no learned models and no biological evidence"})
    print(result.to_string(index=False))
    if not all(gates.values()):
        raise RuntimeError("known-law control gate failed")


if __name__ == "__main__":
    run()
