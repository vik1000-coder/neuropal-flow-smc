"""Method and DGP registries with honest aliases."""
from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

from .config import ScenarioSpec
from .dgp import simulate_dataset
from .mechanistic import MechanisticConfig
from .mechanistic_dataset import simulate_mechanistic_dataset
from .external_causal import PCMCIParCorr, PySINDyDiscreteOfficial, VARLiNGAMOfficial
from .methods.bridge import ConditionalBrownianBridge
from .methods.classical import ConstantGaussian, HeteroskedasticRidge, RidgeGaussian, StudentTRidge
from .methods.covariance import ConditionalCovarianceRidge, FullCovarianceRidge
from .methods.dynamics import ConditionalGrangerRidge, DiscreteSINDy, LaggedCorrelation, SparseTransition
from .methods.neural import ConditionalScoreMLP, GaussianMLP, MixtureDensityMLP
from .methods.mechanistic_latent import LatentNeuromodulatedSSM
from .methods.sbtg import SBTGJointScore
from .methods.sid import GaussianNLLAdapter, SIDQuadratic
from .noise_kernels import GaussianNoise, StudentTNoise
from .schema import DGPConfig


METHODS = {
    "constant_gaussian": (ConstantGaussian, {}),
    "ridge_var": (RidgeGaussian, {}),
    "heteroskedastic_ridge": (HeteroskedasticRidge, {}),
    "student_t_ridge": (StudentTRidge, {}),
    "ridge_full_covariance": (FullCovarianceRidge, {}),
    "conditional_covariance_ridge": (ConditionalCovarianceRidge, {}),
    "gaussian_nll": (GaussianNLLAdapter, {}),
    "sid_hyvarinen": (SIDQuadratic, {"sigma_fraction": 0.0}),
    "sid_dsm": (SIDQuadratic, {"sigma_fraction": 0.2}),
    "sid_dsm_s075": (SIDQuadratic, {"sigma_fraction": 0.75}),
    "sid_dsm_s100": (SIDQuadratic, {"sigma_fraction": 1.0}),
    "lagged_correlation": (LaggedCorrelation, {}),
    "conditional_granger_ridge": (ConditionalGrangerRidge, {}),
    "lagged_sparse_transition": (SparseTransition, {}),
    "sindy_discrete": (DiscreteSINDy, {}),
    "pysindy_discrete_official": (PySINDyDiscreteOfficial, {}),
    "pcmci_parcorr": (PCMCIParCorr, {}),
    "var_lingam_official": (VARLiNGAMOfficial, {}),
    "gaussian_mlp_nll": (GaussianMLP, {"objective": "nll"}),
    "gaussian_mlp_dsm": (GaussianMLP, {"objective": "dsm"}),
    "gaussian_mlp_dsm_s025": (
        GaussianMLP,
        {"objective": "dsm", "dsm_sigma": 0.25},
    ),
    "gaussian_mlp_dsm_s050": (
        GaussianMLP,
        {"objective": "dsm", "dsm_sigma": 0.5},
    ),
    "mdn": (MixtureDensityMLP, {}),
    "mdn_k3": (MixtureDensityMLP, {"components": 3}),
    "mdn_k7": (MixtureDensityMLP, {"components": 7}),
    "score_mlp_gaussian": (ConditionalScoreMLP, {"kernel": GaussianNoise()}),
    "score_mlp_student": (ConditionalScoreMLP, {"kernel": StudentTNoise(df=5)}),
    "score_mlp_gaussian_s025": (
        ConditionalScoreMLP,
        {"kernel": GaussianNoise(), "noise_scale": 0.25},
    ),
    "score_mlp_gaussian_s050": (
        ConditionalScoreMLP,
        {"kernel": GaussianNoise(), "noise_scale": 0.5},
    ),
    "score_mlp_student_s025": (
        ConditionalScoreMLP,
        {"kernel": StudentTNoise(df=5), "noise_scale": 0.25},
    ),
    "score_mlp_student_s050": (
        ConditionalScoreMLP,
        {"kernel": StudentTNoise(df=5), "noise_scale": 0.5},
    ),
    "score_mlp_gaussian_ladder": (
        ConditionalScoreMLP,
        {"kernel": GaussianNoise(), "noise_scales": (0.1, 0.25, 0.5)},
    ),
    "score_mlp_student_ladder": (
        ConditionalScoreMLP,
        {"kernel": StudentTNoise(df=5), "noise_scales": (0.1, 0.25, 0.5)},
    ),
    "sbtg_linear": (SBTGJointScore, {"model_type": "linear", "kernel": GaussianNoise()}),
    "sbtg_linear_s025": (
        SBTGJointScore,
        {"model_type": "linear", "kernel": GaussianNoise(), "noise_scale": 0.25},
    ),
    "sbtg_linear_s050": (
        SBTGJointScore,
        {"model_type": "linear", "kernel": GaussianNoise(), "noise_scale": 0.5},
    ),
    "sbtg_feature_bilinear": (
        SBTGJointScore,
        {"model_type": "feature_bilinear", "kernel": GaussianNoise()},
    ),
    "sbtg_feature_bilinear_s025": (
        SBTGJointScore,
        {
            "model_type": "feature_bilinear",
            "kernel": GaussianNoise(),
            "noise_scale": 0.25,
        },
    ),
    "sbtg_feature_bilinear_s050": (
        SBTGJointScore,
        {
            "model_type": "feature_bilinear",
            "kernel": GaussianNoise(),
            "noise_scale": 0.5,
        },
    ),
    "sbtg_feature_student": (
        SBTGJointScore,
        {"model_type": "feature_bilinear", "kernel": StudentTNoise(df=5)},
    ),
    "sbtg_feature_student_s025": (
        SBTGJointScore,
        {
            "model_type": "feature_bilinear",
            "kernel": StudentTNoise(df=5),
            "noise_scale": 0.25,
        },
    ),
    "sbtg_feature_student_s050": (
        SBTGJointScore,
        {
            "model_type": "feature_bilinear",
            "kernel": StudentTNoise(df=5),
            "noise_scale": 0.5,
        },
    ),
    "sbtg_feature_student_ladder": (
        SBTGJointScore,
        {
            "model_type": "feature_bilinear",
            "kernel": StudentTNoise(df=5),
            "noise_scales": (0.1, 0.25, 0.5),
        },
    ),
    "conditional_schrodinger_bridge": (ConditionalBrownianBridge, {}),
    "conditional_schrodinger_bridge_d015": (
        ConditionalBrownianBridge,
        {"reference_diffusion": 0.15},
    ),
    "conditional_schrodinger_bridge_d030": (
        ConditionalBrownianBridge,
        {"reference_diffusion": 0.30},
    ),
    "conditional_schrodinger_bridge_d060": (
        ConditionalBrownianBridge,
        {"reference_diffusion": 0.60},
    ),
    "latent_neuromodulated_ssm": (LatentNeuromodulatedSSM, {}),
    "latent_neuromodulated_mixture_ssm": (
        LatentNeuromodulatedSSM,
        {"emission": "matched_tail_mixture"},
    ),
}


def method_names() -> tuple[str, ...]:
    return tuple(METHODS)


def make_method(name: str, params: dict[str, Any], seed: int):
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; available: {', '.join(METHODS)}")
    cls, fixed = METHODS[name]
    kwargs = dict(fixed)
    kwargs.update(params)
    if "kernel_df" in kwargs:
        df = float(kwargs.pop("kernel_df"))
        existing = kwargs.get("kernel")
        if isinstance(existing, StudentTNoise):
            kwargs["kernel"] = StudentTNoise(df=df)
        else:
            raise ValueError("kernel_df only applies to a Student-t kernel alias")
    signature = inspect.signature(cls)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "seed" in signature.parameters or accepts_kwargs:
        kwargs.setdefault("seed", seed)
    model = cls(**kwargs)
    model.benchmark_id = name
    return model


def simulate_scenario(spec: ScenarioSpec, seed: int):
    params = dict(spec.params)
    params["seed"] = seed
    if spec.family == "mechanistic":
        return simulate_mechanistic_dataset(
            MechanisticConfig(**params), n_trajectories=spec.n_trajectories
        )
    config = DGPConfig(**params)
    if config.n_trajectories != spec.n_trajectories:
        config = replace(config, n_trajectories=spec.n_trajectories)
    return simulate_dataset(config)
