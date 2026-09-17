from dataclasses import replace
import json
from pathlib import Path

from history_tangent_benchmark.config import load_config
from history_tangent_benchmark.runner import run_benchmark, validate_oracles


def test_default_oracle_validation_passes():
    result = validate_oracles()
    assert result["status"] == "passed"
    assert all(row["passed"] for row in result["rows"])


def test_resume_preserves_completed_case_count(tmp_path, monkeypatch):
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        """
schema_version: "1"
name: tiny_resume
tier: development
output_dir: results/tiny
generator_seeds: [1]
data_seeds: [2]
model_seeds: [3]
generators:
  - id: g1
    kind: g1
    params: {q: 2, dy: 1}
    models: [m1]
models:
  - name: m1
    kind: heteroscedastic_gaussian
    params: {hidden: 8, layers: 1}
split:
  validation_fraction: 0.25
  validation_min: 8
  validation_max: 8
  test_size: 16
  evaluation_histories: 2
  conditional_oracle_draws: 4
training:
  n_train: 32
  device: cpu
  learning_rate: 0.001
  batch_size: 16
  max_epochs: 1
  patience: 1
evaluation:
  energy_cases: 4
  energy_samples: 4
  tangent_pairs: 8
  centering_histories: 2
  centering_draws: 4
  finite_delta: 0.1
resource:
  max_threads: 1
  max_parallel_jobs: 1
  max_rss_gb: 2
  max_output_gb: 0.1
  min_free_disk_gb: 1
  case_timeout_minutes: 5
  checkpoint_minutes: 1
  nice: 10
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    monkeypatch.setattr("history_tangent_benchmark.runner.PACKAGE_ROOT", tmp_path)
    first = run_benchmark(config)
    second = run_benchmark(config, resume=True)
    assert first["completed"] == 1
    assert first["failed"] == 0
    assert second["completed"] == 1
    assert second["failed"] == 0
    assert second["skipped_existing"] == 1
    assert second["status"] == "complete"

    independent = replace(config, output_dir="results/tiny_independent")
    third = run_benchmark(independent)
    assert third["completed"] == 1
    first_record = json.loads(
        next((tmp_path / "results" / "tiny" / "results").glob("*.json")).read_text()
    )
    third_record = json.loads(
        next(
            (tmp_path / "results" / "tiny_independent" / "results").glob("*.json")
        ).read_text()
    )

    def deterministic_metrics(record):
        return {
            metric["metric_id"]: metric["value"]
            for metric in record["metrics"]
            if metric["status"] == "ok"
            and metric["metric_id"]
            in {"native_validation_loss", "nll_original", "tangent_nrmse"}
        }

    assert deterministic_metrics(first_record) == deterministic_metrics(third_record)
