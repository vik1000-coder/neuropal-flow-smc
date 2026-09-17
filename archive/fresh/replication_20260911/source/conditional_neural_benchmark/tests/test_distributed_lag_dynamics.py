import numpy as np

from conditional_neural_benchmark.distributed_lag_dynamics import (
    _causal_fill,
    group_shrink,
    impulse_response,
    reconstruct_kernel,
    smooth_lag_basis,
)


def test_smooth_lag_basis_is_partition_of_unity() -> None:
    basis = smooth_lag_basis(32, 8)
    assert basis.shape == (32, 8)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(basis >= 0)


def test_reconstruct_kernel_orientation() -> None:
    basis = np.eye(3, dtype=np.float32)
    coefficient = np.zeros((2, 2, 3), dtype=np.float32)
    coefficient[1, 0, 1] = 0.75
    kernel = reconstruct_kernel(coefficient, basis)
    assert kernel.shape == (3, 2, 2)
    assert kernel[1, 1, 0] == 0.75


def test_impulse_response_separates_direct_and_propagated_effects() -> None:
    kernel = np.zeros((2, 3, 3), dtype=np.float32)
    kernel[0, 1, 0] = 0.5
    kernel[0, 2, 1] = 0.4
    response = impulse_response(kernel, 3)
    assert response[1, 1, 0] == 0.5
    np.testing.assert_allclose(response[2, 2, 0], 0.2, atol=1e-7)
    assert kernel[0, 2, 0] == 0.0


def test_group_shrink_preserves_zero_diagonal_and_adds_sparsity() -> None:
    coefficient = np.ones((4, 4, 3), dtype=np.float32)
    coefficient[0, 1] = 0.01
    shrunk, threshold, density = group_shrink(coefficient, 0.75)
    assert threshold > 0
    assert density < 1
    np.testing.assert_allclose(shrunk[np.arange(4), np.arange(4)], 0.0)


def test_semisynthetic_leading_missing_values_can_be_zero_filled() -> None:
    trace = np.asarray([[np.nan, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    filled = np.nan_to_num(_causal_fill(trace), nan=0.0)
    assert np.isfinite(filled).all()
    assert filled[0, 0] == 0.0
