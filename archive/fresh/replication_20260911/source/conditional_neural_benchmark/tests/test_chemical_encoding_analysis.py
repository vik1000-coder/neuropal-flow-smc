import numpy as np
import pandas as pd

from conditional_neural_benchmark.chemical_encoding_analysis import (
    CANDIDATES,
    CONTROLS,
    paired_comparisons,
    gate_summary,
)


def test_paired_gate_requires_mean_improvement_and_majority_wins():
    rows = []
    pairs = [(fold, seed) for fold in range(5) for seed in (1701, 2903)]
    for encoding in (*CONTROLS, *CANDIDATES):
        for position, (fold, seed) in enumerate(pairs):
            value = 1.0
            if encoding == "chemical_onehot":
                value = 0.9
            elif encoding == "chemical_plus_position_onehot":
                value = 0.9 if position < 5 else 1.1
            rows.append({
                "stimulus_encoding": encoding,
                "fold": fold,
                "seed": seed,
                "energy__worm_chemical_balanced": value,
            })
    frame = pd.DataFrame(rows)
    comparisons = paired_comparisons(
        frame, cohort_label="test", bootstrap_draws=100, seed=1
    )
    gates = gate_summary(comparisons, "test").set_index("candidate")
    assert bool(gates.loc["chemical_onehot", "eligible_for_lag_sampling"])
    assert not bool(
        gates.loc["chemical_plus_position_onehot", "eligible_for_lag_sampling"]
    )
    assert np.all(
        comparisons[comparisons.candidate == "chemical_onehot"].paired_wins == 10
    )
