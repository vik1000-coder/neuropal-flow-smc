from pathlib import Path

from neuromod_benchmark.serialization import (
    dependency_source_digest,
    environment_manifest,
)


def test_local_dependency_digest_tracks_executable_source_and_metadata(tmp_path: Path):
    dependency = tmp_path / "sid_neuromod"
    source = dependency / "src" / "sid_neuromod"
    source.mkdir(parents=True)
    implementation = source / "model.py"
    implementation.write_text("VALUE = 1\n")
    (dependency / "pyproject.toml").write_text("[project]\nname='sid-neuromod'\n")

    first = dependency_source_digest(dependency)
    implementation.write_text("VALUE = 2\n")
    second = dependency_source_digest(dependency)

    assert first is not None
    assert second is not None
    assert first != second


def test_execution_environment_digest_binds_neighbor_sid_source(tmp_path: Path):
    benchmark_source = tmp_path / "neuromod_benchmark" / "src"
    benchmark_source.mkdir(parents=True)
    (benchmark_source / "benchmark.py").write_text("VALUE = 1\n")
    sid_source = tmp_path / "sid_neuromod" / "src" / "sid_neuromod"
    sid_source.mkdir(parents=True)
    implementation = sid_source / "model.py"
    implementation.write_text("VALUE = 1\n")

    first = environment_manifest(tmp_path)
    implementation.write_text("VALUE = 2\n")
    second = environment_manifest(tmp_path)

    assert first["sid_neuromod_source_digest"] != second["sid_neuromod_source_digest"]
    assert first["execution_environment_digest"] != second["execution_environment_digest"]
