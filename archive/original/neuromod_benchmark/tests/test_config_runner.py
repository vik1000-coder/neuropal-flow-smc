from pathlib import Path
from dataclasses import replace
import json
import pytest

from neuromod_benchmark.config import load_suite
import neuromod_benchmark.runner as runner
from neuromod_benchmark.runner import PACKAGE_ROOT, run_suite


def test_ci_suite_parses_and_runs_capability_aware_cases(tmp_path, monkeypatch):
    config = PACKAGE_ROOT / "configs" / "ci.yaml"
    suite = load_suite(config)
    assert suite.name == "ci"
    monkeypatch.setattr(runner, "PACKAGE_ROOT", tmp_path)
    suite = replace(suite, output_dir="outputs/ci")
    manifest = run_suite(suite, max_cases=2, resume=False)
    assert manifest["completed"] == 2
    assert manifest["failed"] == 0
    results = list((tmp_path / "outputs" / "ci" / "results").glob("*.json"))
    assert results


def test_runner_rejects_mixing_changed_frozen_configs(tmp_path, monkeypatch):
    config = PACKAGE_ROOT / "configs" / "ci.yaml"
    suite = replace(load_suite(config), output_dir="outputs/frozen")
    monkeypatch.setattr(runner, "PACKAGE_ROOT", tmp_path)
    run_suite(suite, max_cases=1, resume=False)
    changed = replace(suite, n_predictive_samples=4)
    try:
        run_suite(changed, max_cases=1, resume=True)
    except RuntimeError as error:
        assert "different frozen suite configuration" in str(error)
    else:
        raise AssertionError("changed suite was allowed to mix with frozen results")


def test_runner_rejects_mixing_changed_execution_environments(tmp_path, monkeypatch):
    config = PACKAGE_ROOT / "configs" / "ci.yaml"
    suite = replace(load_suite(config), output_dir="outputs/frozen_environment")
    monkeypatch.setattr(runner, "PACKAGE_ROOT", tmp_path)
    base = {
        "benchmark_source_digest": "benchmark",
        "sid_neuromod_source_digest": "sid",
        "execution_environment_digest": "environment-a",
    }
    monkeypatch.setattr(runner, "environment_manifest", lambda _workspace: dict(base))
    run_suite(suite, max_cases=1, resume=False)
    monkeypatch.setattr(
        runner,
        "environment_manifest",
        lambda _workspace: {**base, "execution_environment_digest": "environment-b"},
    )
    with pytest.raises(RuntimeError, match="different execution environment digest"):
        run_suite(suite, max_cases=1, resume=True)


def test_runner_uses_identical_worm_split_across_horizons(tmp_path, monkeypatch):
    config = PACKAGE_ROOT / "configs" / "ci.yaml"
    original = load_suite(config)
    suite = replace(
        original,
        methods=(original.methods[0],),
        horizons=(1, 2),
        output_dir="outputs/horizons",
    )
    monkeypatch.setattr(runner, "PACKAGE_ROOT", tmp_path)
    run_suite(suite, resume=False)
    records = [
        json.loads(path.read_text())
        for path in (tmp_path / "outputs" / "horizons" / "results").glob("*.json")
    ]
    assert len(records) == 2
    assert records[0]["split"] == records[1]["split"]


def test_response_panel_preflight_failure_is_persisted_in_manifest(
    tmp_path, monkeypatch
):
    config = PACKAGE_ROOT / "configs" / "ci.yaml"
    suite = replace(load_suite(config), output_dir="outputs/preflight")
    monkeypatch.setattr(runner, "PACKAGE_ROOT", tmp_path)

    def fail_preflight(*args, **kwargs):
        raise ValueError("invalid registered response panel")

    monkeypatch.setattr(runner, "prepare_response_panels", fail_preflight)
    with pytest.raises(ValueError, match="invalid registered response panel"):
        run_suite(suite, resume=False)
    manifest = json.loads(
        (tmp_path / "outputs" / "preflight" / "run_manifest.json").read_text()
    )
    assert manifest["status"] == "failed_scientific_preflight"
    assert manifest["preflight_failure"]["error_type"] == "ValueError"
