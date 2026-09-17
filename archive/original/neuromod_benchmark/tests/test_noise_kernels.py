import numpy as np

from neuromod_benchmark.noise_kernels import (
    GaussianNoise,
    StudentTNoise,
    finite_difference_score,
)


def test_exact_corruption_scores_match_finite_differences():
    clean = np.array([[0.2, -0.4], [1.0, 0.5]])
    noisy = np.array([[0.8, -0.7], [0.6, 1.2]])
    for kernel in (GaussianNoise(), StudentTNoise(df=5)):
        analytic = kernel.conditional_score(noisy, clean, 0.7, {})
        numerical = finite_difference_score(kernel, noisy, clean, 0.7)
        np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-6)


def test_student_t_corruption_shape_and_reproducibility():
    clean = np.zeros((100, 3))
    kernel = StudentTNoise(df=7)
    noisy1, _ = kernel.corrupt(clean, 0.4, np.random.default_rng(3))
    noisy2, _ = kernel.corrupt(clean, 0.4, np.random.default_rng(3))
    np.testing.assert_array_equal(noisy1, noisy2)
    assert noisy1.shape == clean.shape
