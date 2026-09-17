from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from history_tangent_benchmark.config import load_config
from history_tangent_benchmark.metrics import (
    MetricStatus,
    MetricValue,
    ModelCapabilities,
    TrainScaler,
    conditional_centering_error,
    energy_score_fair,
    energy_score_vstat,
    finite_ratio_nrmse,
    hierarchical_paired_bootstrap,
    metric_for_capability,
    tangent_cosine,
    tangent_nrmse,
)
from history_tangent_benchmark.serialization import (
    atomic_json,
    config_sha256,
    enforce_disk_safety,
    environment_manifest,
    source_tree_sha256,
    stable_sha256,
)


def _write_config(path: Path) -> None:
    path.write_text(
        """
schema_version: "1"
name: infrastructure_smoke
tier: smoke
output_dir: outputs/infrastructure_smoke
generator_seeds: [11, 12]
data_seeds: [21, 22]
model_seeds: [31, 32, 33]
split:
  validation_fraction: 0.2
  validation_min: 20
  validation_max: 200
  test_size: 100
  evaluation_histories: 16
  conditional_oracle_draws: 32
resource:
  max_threads: 1
  max_parallel_jobs: 1
  max_rss_gb: 4.0
  max_output_gb: 0.2
  min_free_disk_gb: 0.01
  case_timeout_minutes: 10
  checkpoint_minutes: 2
  nice: 19
generators:
  - id: g1
    kind: nonlinear_location
    params: {history_dim: 2}
    grid: {response_dim: [1, 4]}
models:
  - name: gaussian
    kind: heteroscedastic_gaussian
    capabilities:
      exact_log_prob: true
      conditional_samples: true
training: {max_epochs: 10}
evaluation: {energy_draws: 8}
""".lstrip(),
        encoding="utf-8",
    )


def test_config_preserves_three_independent_seed_axes_and_is_frozen(tmp_path: Path):
    path = tmp_path / "smoke.yaml"
    _write_config(path)
    config = load_config(path)

    assert config.generator_seeds == (11, 12)
    assert config.data_seeds == (21, 22)
    assert config.model_seeds == (31, 32, 33)
    assert config.n_seed_triples == 12
    assert list(config.iter_seed_triples())[:4] == [
        (11, 21, 31),
        (11, 21, 32),
        (11, 21, 33),
        (11, 22, 31),
    ]
    with pytest.raises(TypeError):
        config.generators[0].params["history_dim"] = 9
    assert config.to_dict()["data_seeds"] == [21, 22]
    assert config.config_sha256 == config_sha256(config)


def test_config_rejects_duplicate_seed_within_any_axis(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    _write_config(path)
    original = path.read_text(encoding="utf-8")
    path.write_text(
        original.replace("model_seeds: [31, 32, 33]", "model_seeds: [31, 31]"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model_seeds must contain unique"):
        load_config(path)


def _central_difference(function, point: np.ndarray, step: float = 1e-6) -> np.ndarray:
    result = np.empty_like(point, dtype=np.float64)
    for index in range(len(point)):
        offset = np.zeros_like(point, dtype=np.float64)
        offset[index] = step
        result[index] = (function(point + offset) - function(point - offset)) / (2 * step)
    return result


def test_train_scaler_exact_history_tangent_chain_rule_and_no_test_fit():
    history_train = np.asarray(
        [[-4.0, 10.0], [-2.0, 14.0], [2.0, 18.0], [8.0, 26.0]]
    )
    response_train = np.asarray(
        [[-3.0, 2.0], [-1.0, 4.0], [3.0, 8.0], [9.0, 16.0]]
    )
    scaler = TrainScaler.fit(history_train, response_train)
    shifted_test = np.full((5, 2), 10_000.0)

    np.testing.assert_allclose(scaler.history_mean, history_train.mean(axis=0))
    assert not np.allclose(scaler.transform_history(shifted_test).mean(axis=0), 0.0)

    coefficient = np.asarray([[0.7, -0.2], [0.1, 0.4]])
    history = np.asarray([1.5, 17.0])
    response = np.asarray([0.25, 7.0])

    def original_log_prob(h):
        residual = response - coefficient @ h
        return -0.5 * float(residual @ residual)

    history_standardized = scaler.transform_history(history)
    response_standardized = scaler.transform_response(response)

    def standardized_log_prob(h_standardized):
        h_original = scaler.inverse_history(h_standardized)
        y_original = scaler.inverse_response(response_standardized)
        residual = y_original - coefficient @ h_original
        original = -0.5 * float(residual @ residual)
        return original + scaler.response_log_abs_det

    tangent_original_fd = _central_difference(original_log_prob, history)
    tangent_standardized_fd = _central_difference(
        standardized_log_prob, history_standardized
    )
    np.testing.assert_allclose(
        scaler.history_tangent_to_original(tangent_standardized_fd),
        tangent_original_fd,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        scaler.log_prob_to_original(standardized_log_prob(history_standardized)),
        original_log_prob(history),
        rtol=0,
        atol=1e-12,
    )
    direction_original = np.asarray([0.3, -0.8])
    direction_standardized = scaler.direction_to_standardized(direction_original)
    np.testing.assert_allclose(
        direction_original @ tangent_original_fd,
        direction_standardized @ tangent_standardized_fd,
        rtol=2e-6,
        atol=2e-7,
    )


def test_atomic_json_and_sha256_are_stable_and_source_digest_tracks_tests(tmp_path: Path):
    assert stable_sha256({"b": [2, 3], "a": 1}) == stable_sha256(
        {"a": 1, "b": [2, 3]}
    )
    first_yaml = tmp_path / "first.yaml"
    second_yaml = tmp_path / "second.yaml"
    first_yaml.write_text("a: 1\nb: [2, 3]\n", encoding="utf-8")
    second_yaml.write_text("b: [2, 3]\na: 1\n", encoding="utf-8")
    assert config_sha256(first_yaml) == config_sha256(second_yaml)

    artifact = tmp_path / "nested" / "result.json"
    atomic_json(artifact, {"status": "ok", "value": np.float64(1.25)})
    assert json.loads(artifact.read_text(encoding="utf-8"))["value"] == 1.25
    atomic_json(artifact, {"status": "ok", "value": 2.5})
    assert json.loads(artifact.read_text(encoding="utf-8"))["value"] == 2.5
    assert not list(artifact.parent.glob("*.tmp"))
    with pytest.raises(ValueError, match="non-finite"):
        atomic_json(artifact, {"value": float("nan")})

    source = tmp_path / "package"
    (source / "src").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "results").mkdir()
    (source / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    test_file = source / "tests" / "test_module.py"
    test_file.write_text("def test_value(): pass\n", encoding="utf-8")
    ignored_result = source / "results" / "ignored.json"
    ignored_result.write_text("{}\n", encoding="utf-8")
    before = source_tree_sha256(source)
    ignored_result.write_text('{"changed": true}\n', encoding="utf-8")
    assert source_tree_sha256(source) == before
    test_file.write_text("def test_value(): assert True\n", encoding="utf-8")
    after = source_tree_sha256(source)
    assert before != after
    manifest = environment_manifest(source)
    assert manifest["source_tree_sha256"] == after
    assert len(manifest["environment_sha256"]) == 64


def test_disk_safety_returns_audit_snapshot(tmp_path: Path):
    snapshot = enforce_disk_safety(
        tmp_path, min_free_gb=1e-9, max_output_gb=0.01
    )
    assert snapshot["free_gb"] > 0
    assert snapshot["output_gb"] >= 0


def test_energy_score_formulas_and_chunked_parity():
    observed = np.asarray([[0.0]])
    samples = np.asarray([[[-1.0], [1.0]]])
    assert energy_score_vstat(observed, samples) == pytest.approx(0.5)
    assert energy_score_fair(observed, samples) == pytest.approx(0.0)

    rng = np.random.default_rng(4)
    observed = rng.normal(size=(7, 3))
    samples = rng.normal(size=(7, 5, 3))
    first = np.linalg.norm(samples - observed[:, None, :], axis=-1).mean(axis=1)
    pairwise = np.linalg.norm(
        samples[:, :, None, :] - samples[:, None, :, :], axis=-1
    )
    direct_v = np.mean(first - 0.5 * pairwise.mean(axis=(1, 2)))
    direct_fair = np.mean(
        first - 0.5 * pairwise.sum(axis=(1, 2)) / (samples.shape[1] * (samples.shape[1] - 1))
    )
    assert energy_score_vstat(
        observed, samples, observation_chunk_size=2, draw_chunk_size=2
    ) == pytest.approx(direct_v)
    assert energy_score_fair(
        observed, samples, observation_chunk_size=3, draw_chunk_size=1
    ) == pytest.approx(direct_fair)


def test_tangent_centering_and_ratio_metric_formulas():
    truth = np.asarray([[1.0, 2.0], [3.0, -4.0]])
    estimate = 2.0 * truth
    assert tangent_nrmse(estimate, truth) == pytest.approx(1.0)
    assert tangent_cosine(estimate, truth) == pytest.approx(1.0)
    assert finite_ratio_nrmse(estimate[:, 0], truth[:, 0]) == pytest.approx(1.0)

    conditional_draws = np.asarray(
        [
            [[1.0, -2.0], [-1.0, 2.0]],
            [[3.0, 4.0], [-3.0, -4.0]],
        ]
    )
    assert conditional_centering_error(
        conditional_draws, true_tangent_draws=conditional_draws
    ) == pytest.approx(0.0)


def test_metric_statuses_and_capability_gate_are_explicit():
    capabilities = ModelCapabilities(exact_log_prob=False, conditional_samples=True)
    unavailable = metric_for_capability(
        "law.nll", capabilities, "exact_log_prob", lambda: 0.0
    )
    assert unavailable.status is MetricStatus.NOT_APPLICABLE
    assert unavailable.value is None
    assert unavailable.required_capability == "exact_log_prob"
    computed = metric_for_capability(
        "law.energy_v", capabilities, "conditional_samples", lambda: 1.5, n=10
    )
    assert computed == MetricValue.ok("law.energy_v", 1.5, n=10)
    adapted = ModelCapabilities.from_model(
        SimpleNamespace(
            capabilities=SimpleNamespace(
                normalized_density=True,
                sampler=True,
                history_tangent=True,
                response_score=False,
            )
        )
    )
    assert adapted.exact_log_prob and adapted.conditional_samples
    assert adapted.finite_log_ratio


def test_hierarchical_bootstrap_resamples_generator_then_nested_model_seed():
    generators = [0, 0, 0, 1, 1, 1]
    data_seeds = [20, 20, 20, 21, 21, 21]
    models = [10, 11, 12, 10, 11, 12]
    method_b = np.asarray([2.0, 4.0, 8.0, 3.0, 6.0, 12.0])
    method_a = 0.5 * method_b
    result = hierarchical_paired_bootstrap(
        method_a,
        method_b,
        generators,
        models,
        data_seeds=data_seeds,
        n_bootstrap=250,
        seed=9,
        practical_improvement=0.2,
    )
    assert result.n_generator_seeds == 2
    assert result.n_data_seed_instances == 2
    assert result.n_paired_fits == 6
    assert result.ratio == pytest.approx(0.5)
    assert result.ci95_ratio == pytest.approx((0.5, 0.5))
    assert result.probability_practical_improvement == 1.0
