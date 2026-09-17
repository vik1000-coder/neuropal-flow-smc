import pytest

from neuromod_benchmark.noise_kernels import GaussianNoise, StudentTNoise
from neuromod_benchmark.registry import make_method


@pytest.mark.parametrize(
    ("name", "attribute", "expected"),
    [
        ("sid_dsm_s075", "sigma_fraction", 0.75),
        ("sid_dsm_s100", "sigma_fraction", 1.0),
        ("gaussian_mlp_dsm_s025", "dsm_sigma", 0.25),
        ("gaussian_mlp_dsm_s050", "dsm_sigma", 0.5),
        ("mdn_k3", "components", 3),
        ("mdn_k7", "components", 7),
    ],
)
def test_frozen_sensitivity_aliases_bind_their_declared_setting(
    name, attribute, expected
):
    model = make_method(name, {}, seed=3)
    assert getattr(model, attribute) == expected
    assert model.benchmark_id == name


@pytest.mark.parametrize(
    ("name", "scale", "kernel_type"),
    [
        ("score_mlp_gaussian_s025", 0.25, GaussianNoise),
        ("score_mlp_gaussian_s050", 0.5, GaussianNoise),
        ("score_mlp_student_s025", 0.25, StudentTNoise),
        ("score_mlp_student_s050", 0.5, StudentTNoise),
        ("sbtg_linear_s025", 0.25, GaussianNoise),
        ("sbtg_linear_s050", 0.5, GaussianNoise),
        ("sbtg_feature_bilinear_s025", 0.25, GaussianNoise),
        ("sbtg_feature_bilinear_s050", 0.5, GaussianNoise),
        ("sbtg_feature_student_s025", 0.25, StudentTNoise),
        ("sbtg_feature_student_s050", 0.5, StudentTNoise),
    ],
)
def test_score_aliases_bind_kernel_and_single_corruption_scale(
    name, scale, kernel_type
):
    model = make_method(name, {}, seed=3)
    assert model.noise_scales == (scale,)
    assert isinstance(model.kernel, kernel_type)

