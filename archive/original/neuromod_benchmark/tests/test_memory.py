from pathlib import Path
import json

import pandas as pd
import neuromod_benchmark.memory as memory

from neuromod_benchmark.config import MethodSpec, ScenarioSpec
from neuromod_benchmark.memory import MemorySuite, run_memory_suite
from neuromod_benchmark.schema import ResourceBudget


def test_memory_curve_uses_fixed_worm_splits_and_zero_reference_gap(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(memory, "PACKAGE_ROOT", tmp_path)
    suite = MemorySuite(
        name="tiny_memory",
        scenarios=(ScenarioSpec("mixed", "mechanistic", {
            "n_neurons": 3,
            "n_modulators": 1,
            "n_steps": 45,
            "burn_in": 65,
            "mechanism": "mixed",
            "ligand_pulse_rate_hz": .1,
        }, n_trajectories=5),),
        methods=(MethodSpec("constant_gaussian"), MethodSpec("ridge_var")),
        seeds=(0,),
        views=("latent",),
        max_lags=(1, 3),
        output_dir="outputs/tiny",
        resource=ResourceBudget(max_output_gb=.1, stop_free_disk_gb=4),
        n_predictive_samples=4,
    )
    manifest = run_memory_suite(suite)
    assert manifest["status"] == "complete"
    assert manifest["records"] == 4
    frame = pd.read_csv(tmp_path / "outputs" / "tiny" / "metrics_tidy.csv")
    gaps = frame[
        (frame.metric == "memory.finite_model_nll_gap_to_max_lag")
        & (frame.max_lag == 3)
    ]
    assert len(gaps) == 2
    assert (gaps.value.abs() < 1e-12).all()
    records = json.loads((tmp_path / "outputs" / "tiny" / "results.json").read_text())
    for method in {record["method"] for record in records}:
        lanes = [record for record in records if record["method"] == method]
        counts = [record["sample_counts"] for record in lanes]
        assert counts[0] == counts[1]
        assert all(record["common_target_start"] == 2 for record in lanes)
    resumed = run_memory_suite(suite)
    assert resumed["records"] == 4
    assert resumed["skipped"] == 4
