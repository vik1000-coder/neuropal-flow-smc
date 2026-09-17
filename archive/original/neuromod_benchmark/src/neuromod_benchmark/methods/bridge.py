"""Small conditional Schrödinger-bridge endpoint/path reference.

For each history, the initial state is a point mass at the current neural state.
The fitted endpoint law is Gaussian. Relative to Brownian motion starting at that
point, tilting by the desired endpoint density gives the Schrödinger bridge; its
conditional paths are ordinary Brownian bridges after sampling the tilted endpoint.

This model is generative and useful for path/endpoint tests. It does not identify a
neuromodulator or a causal edge by itself.
"""
from __future__ import annotations

import numpy as np

from ..capabilities import Capabilities, ResourceProfile
from ..schema import Prediction, SupervisedData
from .base import Estimator
from .classical import HeteroskedasticRidge


class ConditionalBrownianBridge(Estimator):
    name = "conditional_brownian_schrodinger_bridge"
    capabilities = Capabilities(
        predictive_mean=True,
        predictive_distribution="normalized",
        recursive_path_samples=True,
        effect_channels=(
            "conditional_mean_derivative",
            "conditional_log_variance_derivative",
        ),
    )
    resource_profile = ResourceProfile("bridge_small", threads=1, estimated_peak_gb=1.5)

    def __init__(
        self,
        *,
        reference_diffusion: float = 0.25,
        mean_ridge: float = 1.0,
        variance_ridge: float = 10.0,
        seed: int = 0,
    ):
        if reference_diffusion <= 0:
            raise ValueError("reference_diffusion must be positive")
        self.reference_diffusion = float(reference_diffusion)
        self.mean_ridge = float(mean_ridge)
        self.variance_ridge = float(variance_ridge)
        self.seed = int(seed)

    def fit(self, train: SupervisedData, validation: SupervisedData | None = None):
        self.endpoint_model_ = HeteroskedasticRidge(
            mean_ridge=self.mean_ridge, variance_ridge=self.variance_ridge
        ).fit(train, validation)
        self.channels_ = self.endpoint_model_.channels_
        self.source_index_ = train.source_index.copy()
        self.n_targets_ = train.targets.shape[1]
        self.current_columns_ = np.asarray(
            [np.flatnonzero(train.source_index == source)[0] for source in range(self.n_targets_)]
        )
        return self

    def predict(self, data: SupervisedData, *, n_samples: int = 0) -> Prediction:
        return self.endpoint_model_.predict(data, n_samples=n_samples)

    def reference_endpoint(self, data: SupervisedData) -> tuple[np.ndarray, np.ndarray]:
        initial = data.features[:, self.current_columns_]
        horizon = float(data.horizon)
        variance = np.full_like(initial, self.reference_diffusion**2 * horizon)
        return initial, variance

    def path_relative_entropy(self, data: SupervisedData) -> np.ndarray:
        """Exact KL(Q_path || Brownian_reference) for the endpoint h-transform."""

        endpoint = self.predict(data)
        reference_mean, reference_variance = self.reference_endpoint(data)
        ratio = endpoint.variance / reference_variance
        squared_shift = (endpoint.mean - reference_mean) ** 2 / reference_variance
        return 0.5 * np.sum(ratio + squared_shift - 1.0 - np.log(ratio), axis=1)

    def sample_paths(
        self,
        data: SupervisedData,
        *,
        n_paths: int = 32,
        n_steps: int = 16,
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample [observation, path, bridge_time, target] trajectories."""

        if n_paths < 1 or n_steps < 1:
            raise ValueError("n_paths and n_steps must be positive")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        prediction = self.predict(data)
        initial = data.features[:, self.current_columns_]
        endpoint = prediction.mean[:, None, :] + np.sqrt(prediction.variance[:, None, :]) * (
            rng.standard_normal((len(initial), n_paths, self.n_targets_))
        )
        # Generate a Brownian path and remove its linear endpoint component. This
        # produces a zero-endpoint Brownian bridge independent of sampled endpoints.
        increments = rng.standard_normal((len(initial), n_paths, n_steps, self.n_targets_))
        # Brownian reference time is the forecast horizon, matching
        # ``reference_endpoint`` variance sigma^2 * horizon.
        increments *= self.reference_diffusion * np.sqrt(
            float(data.horizon) / n_steps
        )
        brownian = np.concatenate(
            [np.zeros((len(initial), n_paths, 1, self.n_targets_)), np.cumsum(increments, axis=2)],
            axis=2,
        )
        time = np.linspace(0.0, 1.0, n_steps + 1)
        bridge_noise = brownian - time[None, None, :, None] * brownian[:, :, -1:, :]
        mean_path = (
            (1.0 - time[None, None, :, None]) * initial[:, None, None, :]
            + time[None, None, :, None] * endpoint[:, :, None, :]
        )
        return mean_path + bridge_noise

    def metadata(self):
        return {
            "reference": "Brownian motion from observed current state",
            "reference_diffusion": self.reference_diffusion,
            "reference_diffusion_unit": "target activity per sqrt(simulation step)",
            "endpoint_model": self.endpoint_model_.name,
            "mean_ridge": self.mean_ridge,
            "variance_ridge": self.variance_ridge,
            "scope": "experimental low-dimensional path baseline",
            "causal_interpretation": "none without an explicit intervention model",
        }
