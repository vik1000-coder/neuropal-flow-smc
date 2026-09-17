from pathlib import Path
import json

import pandas as pd
import neuromod_benchmark.robustness as robustness

from neuromod_benchmark.config import MethodSpec, ScenarioSpec
from neuromod_benchmark.hierarchy import ObservationVariant
from neuromod_benchmark.robustness import RobustnessSuite, run_robustness_suite
from neuromod_benchmark.schema import ResourceBudget


def test_hierarchical_robustness_is_paired_and_writes_tidy_results(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(robustness, "PACKAGE_ROOT", tmp_path)
    suite = RobustnessSuite(
        name="tiny_robustness",
        scenarios=(ScenarioSpec("mixed", "mechanistic", {
            "n_neurons": 3,
            "n_modulators": 1,
            "n_steps": 42,
            "burn_in": 60,
            "mechanism": "mixed",
            "ligand_pulse_rate_hz": .1,
        }),),
        methods=(MethodSpec("constant_gaussian"), MethodSpec("ridge_var")),
        seeds=(0,),
        coefficient_cvs=(.15,),
        views=("calcium",),
        history_lags=(1,),
        horizon=1,
        hierarchy_counts={"train": 2, "validation": 1, "test": 1},
        observation_variants=(
            ObservationVariant("reference"),
            ObservationVariant("noisy", measurement_noise_multiplier=2),
        ),
        output_dir="outputs/tiny",
        resource=ResourceBudget(max_output_gb=.1, stop_free_disk_gb=4),
        n_predictive_samples=4,
    )
    manifest = run_robustness_suite(suite)
    assert manifest["status"] == "complete"
    assert manifest["records"] == 4
    frame = pd.read_csv(tmp_path / "outputs" / "tiny" / "metrics_tidy.csv")
    assert set(frame.observation_variant) == {"reference", "noisy"}
    assert "transfer.delta_predictive.nll" in set(frame.metric)
    resumed = run_robustness_suite(suite)
    assert resumed["records"] == 4
    assert resumed["skipped"] == 4


def test_hierarchical_latent_lane_scores_states_but_not_one_population_parameter(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(robustness, "PACKAGE_ROOT", tmp_path)
    suite = RobustnessSuite(
        name="tiny_latent_robustness",
        scenarios=(ScenarioSpec("mixed", "mechanistic", {
            "n_neurons": 3,
            "n_modulators": 1,
            "n_steps": 35,
            "burn_in": 45,
            "mechanism": "mixed",
        }),),
        methods=(MethodSpec("latent_neuromodulated_ssm", params={
            "n_modulators": 1,
            "max_epochs": 2,
            "patience": 1,
            "truncation": 16,
            "warmup": 1,
        }),),
        seeds=(0,),
        coefficient_cvs=(.15,),
        views=("calcium",),
        history_lags=(1,),
        horizon=1,
        hierarchy_counts={"train": 2, "validation": 1, "test": 1},
        observation_variants=(ObservationVariant("reference"),),
        output_dir="outputs/tiny_latent",
        resource=ResourceBudget(max_output_gb=.1, stop_free_disk_gb=4),
        n_predictive_samples=2,
    )
    manifest = run_robustness_suite(suite)
    assert manifest["status"] == "complete"
    records = json.loads(
        (tmp_path / "outputs" / "tiny_latent" / "results.json").read_text()
    )
    metrics = records[0]["metrics"]
    assert "latent.test_spearman" in metrics
    assert metrics["latent.parameter_recovery_omitted_hierarchical_truth"] == 1.0
    assert "latent.clearance_tau_mae_seconds" not in metrics
