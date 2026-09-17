"""Isolated adapters for official PCMCI, VAR-LiNGAM, and PySINDy.

Optional packages are imported only by a benchmark-local subprocess. Assumptions
and source/target orientation remain attached to every fitted readout, and graph-only
methods explicitly refuse fake predictive distributions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
from scipy.special import ndtr

from .capabilities import Capabilities, ResourceProfile, UnsupportedCapability
from .metrics import gaussian_log_prob
from .methods.base import Estimator, Standardizer, aggregate_features
from .schema import Prediction, SupervisedData


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER_PYTHON = PACKAGE_ROOT / ".causal_venv" / "bin" / "python"


def _recent_state_columns(data: SupervisedData) -> np.ndarray:
    columns = []
    n_sources = int(np.max(data.source_index)) + 1
    if n_sources != data.targets.shape[1]:
        raise ValueError("source and target neuron counts differ")
    for source in range(n_sources):
        matches = np.flatnonzero(data.source_index == source)
        if matches.size != 1:
            raise UnsupportedCapability(
                "published_causal_adapter", "exactly_one_observed_lag_per_neuron"
            )
        columns.append(int(matches[0]))
    return np.asarray(columns, dtype=int)


def _worker_python() -> Path:
    configured = os.environ.get("NEUROMOD_CAUSAL_PYTHON")
    path = Path(configured).expanduser() if configured else DEFAULT_WORKER_PYTHON
    if not path.is_file():
        raise RuntimeError(
            f"causal worker Python not found at {path}; run scripts/bootstrap_causal_env.sh"
        )
    return path


def _run_worker(
    action: str,
    data: SupervisedData,
    options: dict[str, Any],
    *,
    features: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    timeout_seconds: int = 1_800,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scratch = PACKAGE_ROOT / "outputs" / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{action}-", dir=scratch) as directory:
        root = Path(directory)
        source, target = root / "input.npz", root / "output.npz"
        np.savez_compressed(
            source,
            features=data.features if features is None else features,
            targets=data.targets if targets is None else targets,
            groups=data.groups,
            times=data.times,
            source_index=data.source_index,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(PACKAGE_ROOT / "src"),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        command = [
            str(_worker_python()),
            "-m",
            "neuromod_benchmark.causal_worker",
            "--action",
            action,
            "--input",
            str(source),
            "--output",
            str(target),
            "--options",
            json.dumps(options, sort_keys=True),
        ]
        completed = subprocess.run(
            command,
            env=environment,
            cwd=PACKAGE_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0 or not target.exists():
            detail = (completed.stderr or completed.stdout).strip()[-4_000:]
            raise RuntimeError(f"{action} worker failed ({completed.returncode}): {detail}")
        with np.load(target, allow_pickle=False) as result:
            metadata = json.loads(str(result["metadata_json"].item()))
            arrays = {
                key: np.asarray(result[key]).copy()
                for key in result.files
                if key != "metadata_json"
            }
    return arrays, metadata


class PCMCIParCorr(Estimator):
    """Official Tigramite PCMCI with ParCorr across independent episodes."""

    name = "pcmci_parcorr"
    capabilities = Capabilities(
        effect_channels=("conditional_independence_strength",), graph_scores=True
    )
    resource_profile = ResourceProfile("medium_cpu", threads=1, estimated_peak_gb=1.0)

    def __init__(
        self,
        tau_max: int = 1,
        pc_alpha: float = 0.2,
        alpha_level: float = 0.05,
        fdr_method: str = "fdr_bh",
    ) -> None:
        self.options = {
            "tau_max": int(tau_max),
            "pc_alpha": float(pc_alpha),
            "alpha_level": float(alpha_level),
            "fdr_method": str(fdr_method),
        }

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.state_columns_ = _recent_state_columns(train)
        arrays, self.worker_metadata_ = _run_worker("pcmci", train, self.options)
        val, p_value = arrays["val_matrix"], arrays["p_matrix"]
        n = train.targets.shape[1]
        if val.shape[0] < n or val.shape[1] < n:
            raise RuntimeError("PCMCI returned fewer variables than neural targets")
        lagged = val[:n, :n, 1 : self.options["tau_max"] + 1]
        p_lagged = p_value[:n, :n, 1 : self.options["tau_max"] + 1]
        best = np.argmax(np.abs(lagged), axis=2)
        score = np.take_along_axis(lagged, best[:, :, None], axis=2)[:, :, 0]
        p_score = np.take_along_axis(p_lagged, best[:, :, None], axis=2)[:, :, 0]
        self.channels_ = {
            "conditional_independence_strength": score,
            "conditional_independence_p_value": p_score,
            "conditional_independence_lag_tensor": lagged,
        }
        return self

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        del data, n_samples
        raise UnsupportedCapability(self.name, "predictive_distribution")

    def metadata(self) -> dict[str, Any]:
        return {
            **self.options,
            **getattr(self, "worker_metadata_", {}),
            "claim_level_complete_state": "assumption-limited lagged conditional graph",
            "claim_level_hidden_modulator": "reduced-form conditional dependence only",
        }


class VARLiNGAMOfficial(Estimator):
    """Official VAR-LiNGAM, independently fit per episode and aggregated."""

    name = "var_lingam_official"
    capabilities = Capabilities(
        effect_channels=("linear_transition_coefficient",), graph_scores=True
    )
    resource_profile = ResourceProfile("medium_cpu", threads=1, estimated_peak_gb=1.25)

    def __init__(
        self,
        lags: int = 1,
        criterion: str = "bic",
        prune: bool = False,
        seed: int = 0,
    ) -> None:
        self.options = {
            "lags": int(lags),
            "criterion": str(criterion),
            "prune": bool(prune),
            "seed": int(seed),
        }

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.state_columns_ = _recent_state_columns(train)
        # VAR-LiNGAM is an endogenous-vector model, not a VARX implementation.
        # Passing stimulus or current-modulator columns silently changes the graph
        # and fails when a legitimate null scenario makes those columns constant.
        arrays, self.worker_metadata_ = _run_worker(
            "var_lingam",
            train,
            self.options,
            features=train.features[:, self.state_columns_],
        )
        lag_tensor = arrays["lag_tensor"]
        n = train.targets.shape[1]
        lag_neural = lag_tensor[:, :n, :n]
        weights = 1.0 / (1.0 + np.arange(lag_neural.shape[0]))
        score = np.einsum("ltj,l->tj", lag_neural, weights)
        self.channels_ = {
            "linear_transition_coefficient": score,
            "var_lingam_lag_tensor": lag_neural,
            "var_lingam_instantaneous": arrays["instantaneous"][:n, :n],
        }
        return self

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        del data, n_samples
        raise UnsupportedCapability(self.name, "predictive_distribution")

    def metadata(self) -> dict[str, Any]:
        return {
            **self.options,
            **getattr(self, "worker_metadata_", {}),
            "instantaneous_matrix_evaluated_as_neural_truth": False,
            "claim_level_hidden_modulator": "not causal; latent-confounding assumption fails",
        }


class PySINDyDiscreteOfficial(Estimator):
    """Official PySINDy STLSQ plus a declared diagonal-Gaussian residual law."""

    name = "pysindy_discrete_official"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        effect_channels=("conditional_mean_derivative",),
        graph_scores=True,
        equations=True,
    )
    resource_profile = ResourceProfile("medium_cpu", threads=1, estimated_peak_gb=1.25)

    def __init__(
        self,
        degree: int = 2,
        threshold: float = 0.02,
        alpha: float = 0.01,
        max_iter: int = 10,
        normalize_columns: bool = False,
        unbias: bool = True,
        seed: int = 0,
    ) -> None:
        del seed
        self.options = {
            "degree": int(degree),
            "threshold": float(threshold),
            "alpha": float(alpha),
            "max_iter": int(max_iter),
            "normalize_columns": bool(normalize_columns),
            "unbias": bool(unbias),
        }

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        del validation
        self.scaler_ = Standardizer.fit(train)
        x, y = self.scaler_.x(train.features), self.scaler_.y(train.targets)
        arrays, self.worker_metadata_ = _run_worker(
            "pysindy", train, self.options, features=x, targets=y
        )
        self.coefficients_ = arrays["coefficients"]
        self.powers_ = arrays["powers"].astype(int)
        self.residual_variance_ = arrays["residual_variance"]
        jacobian = self._average_jacobian(train.features)
        self.channels_ = {
            "conditional_mean_derivative_feature": jacobian,
            "conditional_mean_derivative": aggregate_features(
                jacobian, train.source_index, train.targets.shape[1], signed=True
            ),
            "conditional_mean_strength": aggregate_features(
                jacobian, train.source_index, train.targets.shape[1], signed=False
            ),
            "equation_coefficients": self.coefficients_,
        }
        return self

    def _library(self, standardized: np.ndarray) -> np.ndarray:
        return np.prod(standardized[:, None, :] ** self.powers_[None, :, :], axis=2)

    def _predict_mean(self, features: np.ndarray) -> np.ndarray:
        theta = self._library(self.scaler_.x(features))
        return self.scaler_.inverse_mean(theta @ self.coefficients_.T)

    def _average_jacobian(self, features: np.ndarray, max_rows: int = 256) -> np.ndarray:
        if len(features) > max_rows:
            features = features[np.linspace(0, len(features) - 1, max_rows, dtype=int)]
        jacobian = np.zeros((self.scaler_.y_mean.size, features.shape[1]))
        step = 1e-4 * np.maximum(self.scaler_.x_scale, 1.0)
        for column in range(features.shape[1]):
            plus, minus = features.copy(), features.copy()
            plus[:, column] += step[column]
            minus[:, column] -= step[column]
            jacobian[:, column] = np.mean(
                (self._predict_mean(plus) - self._predict_mean(minus)) / (2 * step[column]),
                axis=0,
            )
        return jacobian

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        mean = self._predict_mean(data.features)
        variance = np.broadcast_to(
            self.scaler_.inverse_variance(self.residual_variance_), mean.shape
        ).copy()
        samples = None
        if n_samples:
            rng = np.random.default_rng(48_271)
            samples = mean[:, None, :] + np.sqrt(variance[:, None, :]) * rng.standard_normal(
                (len(mean), n_samples, mean.shape[1])
            )
        return Prediction(
            mean=mean,
            variance=variance,
            log_prob=gaussian_log_prob(data.targets, mean, variance),
            cdf=ndtr((data.targets - mean) / np.sqrt(variance)),
            samples=samples,
            channels=self.channels_,
        )

    def metadata(self) -> dict[str, Any]:
        return {**self.options, **getattr(self, "worker_metadata_", {})}


__all__ = ["PCMCIParCorr", "PySINDyDiscreteOfficial", "VARLiNGAMOfficial"]
