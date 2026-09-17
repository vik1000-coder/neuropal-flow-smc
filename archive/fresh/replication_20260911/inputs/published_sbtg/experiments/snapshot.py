"""Repository-relative access to the immutable production reference snapshot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = REPOSITORY_ROOT / "reference_snapshot"
PRODUCTION_ROOT = SNAPSHOT_ROOT / "production"
PREPARED_DATA = SNAPSHOT_ROOT / "prepared_data" / "full_traces_imputed"
PRODUCTION_MODEL_SOURCE = PRODUCTION_ROOT / "multilag_sbtg.py"
PRODUCTION_SAFE_RESULT = PRODUCTION_ROOT / "sbtg_lag_matrices.npz"
PRODUCTION_HYPERPARAMETERS = PRODUCTION_ROOT / "hyperparameters.json"
PRODUCTION_CONFIG = PRODUCTION_ROOT / "config.json"
STIMULUS_SOURCE = SNAPSHOT_ROOT / "stimulus" / "stimulus_periods.py"
PHASE_ROOT = SNAPSHOT_ROOT / "phase_analysis"
BENCHMARK_ROOT = SNAPSHOT_ROOT / "benchmarks"
PHASE_CODES = {
    "baseline": "baseline",
    "steady": "steady",
    "on": "on",
    "off": "off",
}

EXPECTED_SHA256 = {
    PRODUCTION_MODEL_SOURCE: "f987cfd13748adf32106353fa6a13431c9bfb8482d223a4af268d583c15df5a7",
    PRODUCTION_SAFE_RESULT: "e5e810c8694a94a27dc414fc04b49c357ba0495198b6eacc07d2ab48f2cfbd4a",
    PRODUCTION_HYPERPARAMETERS: "c1ac27166c187572a38f6d326004ec892e6fc93f8b688b58adf3e4b60a8f0718",
    PRODUCTION_CONFIG: "5b4c62aa283d78defb37f5d0971783b3dcde1d02638584ea964698af2d8d2d21",
    PREPARED_DATA / "traces.npz": "acbe8bb1a555e5b7f20e2d52c5fbea6692af2fb3daa3cce0db8e9b659fee9c6c",
    PREPARED_DATA / "standardization.json": "cde4ddb4d9352179e2ba1aeb13365fbe22f3d07289ea71f1b606096f64455816",
    PREPARED_DATA / "segments.csv": "112c078a1c4e6073fdd46698091ee8a2943c5fa76bae275e58db7f5271a6aa5b",
    STIMULUS_SOURCE: "168d32743a71f954551be0794057c344587b089440bb737bc626a6b5591982af",
    PHASE_ROOT / "baseline" / "result.npz": "ed12b2256e643fe7707d2dba3fa959dabe67895cdc7c74285f48f492cf6ec644",
    PHASE_ROOT / "steady" / "result.npz": "f57588eaa28c21ec4e70658042e4def801973208ef1619f76a2b4d2db4081654",
    PHASE_ROOT / "on" / "result.npz": "2c4c402a59b324c56dd4133e3fa7aca97d6a4e6bd995ef16d9b2e230bc413b1d",
    PHASE_ROOT / "off" / "result.npz": "f0e2a11f57212e75d20d2440b1a09eefc831fd6a3d99c6e0a86245950cf7e81f",
    PHASE_ROOT / "comparison" / "celltype_by_lag_baseline.csv": "55c1f86baca80535b3f817c3d3dbaadc09f1502ffc3a0876092656a2bf8a93d9",
    PHASE_ROOT / "comparison" / "celltype_by_lag_steady.csv": "062b8f90a3f018b962c7b5bef679369234edb55f53bf960f15d81072a55a7fdb",
    PHASE_ROOT / "comparison" / "celltype_by_lag_on.csv": "2e9fb69475d8192c1d822c005c81ae8ca1fc593fc2286dad8611f0c2ee88d0a5",
    PHASE_ROOT / "comparison" / "celltype_by_lag_off.csv": "e429d4ba02446ae51398151f5fbf0400cf6de2c8a8cb1d3854e0da42f90ae8d8",
    BENCHMARK_ROOT / "granger.npz": "e6ac7531b376334780feab2bb80fe41f2c15dbe3586d3e0915831d99caa19e99",
    BENCHMARK_ROOT / "lagged_cross_correlation.npz": "7eaf2f6223d5b0b880474f36d59bf9d58c807849c61a515c81431672ddb3f17e",
}

# SHA-256 digests of the upstream archives before their safe schema conversion.
SOURCE_ARCHIVE_SHA256 = {
    "production_result": "341771375f8bad095baf7cfe0836dc76e1eaf56f3d92f903ad958848079730cb",
    "phase_baseline": "782b8762dc04357a50cb6525bc0eed5f67473423ac93406f49ae21fccb53a82f",
    "phase_steady": "99cdf38d8475a0e875c587ca98ec69c78e2330c9b21f677f1dcac15d54261d0a",
    "phase_on": "8048dfb98fb6249dfa9aafcdebbc60edf3bd7179734ebaa3198b54fae837529a",
    "phase_off": "31a83f28afe20abb9eda6c496c4495041a61d0114d28fe36e0cce539d7fef5f6",
}
EXPECTED_MULTILAG_SHA256 = EXPECTED_SHA256[PRODUCTION_MODEL_SOURCE]
EXPECTED_SAFE_RESULT_SHA256 = (
    "e5e810c8694a94a27dc414fc04b49c357ba0495198b6eacc07d2ab48f2cfbd4a"
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_relative(path: Path) -> str:
    """Return a portable repository-relative path for metadata."""
    return Path(path).resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def portable_path(path: Path) -> str:
    """Describe a path without recording machine-specific parent directories."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return "external"


def validate_snapshot(include_phase_anchors: bool = False) -> None:
    """Verify required files and byte-exact provenance anchors."""
    required = {
        path: expected
        for path, expected in EXPECTED_SHA256.items()
        if include_phase_anchors or PHASE_ROOT not in path.parents
    }
    missing = [repository_relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Reference snapshot is incomplete: {missing}")
    changed = {
        repository_relative(path): (expected, sha256(path))
        for path, expected in required.items()
        if expected is not None and sha256(path) != expected
    }
    if changed:
        raise RuntimeError(f"Reference snapshot integrity check failed: {changed}")


def load_module(name: str, path: Path):
    """Load one source file without modifying the package import path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_production_multilag():
    validate_snapshot()
    return load_module("sbtg_production_multilag", PRODUCTION_MODEL_SOURCE)


def load_stimulus_periods():
    validate_snapshot()
    return load_module("sbtg_production_stimulus_periods", STIMULUS_SOURCE)


def load_fixed_hyperparameters(path: Path = PRODUCTION_HYPERPARAMETERS) -> dict[int, dict]:
    """Load fixed per-lag hyperparameters from safe JSON."""
    with Path(path).open() as handle:
        values = json.load(handle)
    return {int(lag): dict(config) for lag, config in values.items()}


def derive_seed(base_seed: int, *components: int) -> int:
    sequence = np.random.SeedSequence(
        [int(base_seed), *(int(component) for component in components)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))


def deterministic_estimator_class(module):
    """Return the production estimator with explicit fold-local RNG resets."""

    class DeterministicProductionEstimator(module.MinimalMultiBlockEstimator):
        def __init__(self, *args, model_seed: int = 42, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_seed = int(model_seed)
            self.fold_seeds: dict[tuple[int, int], int] = {}

        def _cross_fit_lag(self, windows, stim_ids, n_neurons, lag, config):
            n_blocks = lag + 1
            fold_ids = module.create_fold_assignments(
                stim_ids, self.n_folds, self.random_state
            )
            scores_heldout = np.zeros_like(windows)

            for fold_index in range(self.n_folds):
                fold_seed = derive_seed(self.model_seed, 20260123, lag, fold_index)
                self.fold_seeds[(int(lag), int(fold_index))] = fold_seed
                seed_all(fold_seed)
                if self.verbose:
                    print(
                        f"    Fold {fold_index + 1}/{self.n_folds} "
                        f"(torch seed {fold_seed})...",
                        flush=True,
                    )

                train_mask = fold_ids != fold_index
                heldout_mask = fold_ids == fold_index
                windows_std, _, _ = module.standardize_windows(windows, train_mask)
                if n_blocks == 2:
                    model = module.TwoBlockStructuredScoreNet(
                        n_neurons=n_neurons,
                        hidden_dim=config.hidden_dim,
                        num_layers=config.num_layers,
                    )
                else:
                    model = module.MultiBlockStructuredScoreNet(
                        n_neurons=n_neurons,
                        p_max=lag,
                        hidden_dim=config.hidden_dim,
                        num_layers=config.num_layers,
                    )
                module.train_score_model(
                    model,
                    windows_std[train_mask],
                    config.noise_std,
                    config.lr,
                    config.epochs,
                    self.batch_size,
                    self.l1_lambda,
                    self.device,
                    verbose=False,
                )
                scores_heldout[heldout_mask] = module.compute_scores(
                    model, windows_std[heldout_mask], self.device
                )
            return scores_heldout

    return DeterministicProductionEstimator
