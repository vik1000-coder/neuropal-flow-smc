import numpy as np

from conditional_neural_benchmark.distributed_lag_postfreeze import (
    lag_centroid,
    signed_max_matrix,
)


def test_signed_max_matrix_keeps_sign_at_largest_lag_effect() -> None:
    kernel = np.asarray([[[0.1]], [[-0.4]], [[0.2]]])
    result = signed_max_matrix(kernel)
    assert result[0, 0] == -0.4


def test_lag_centroid_uses_absolute_kernel_mass() -> None:
    kernel = np.asarray([[[1.0]], [[0.0]], [[1.0]]])
    result = lag_centroid(kernel)
    assert result[0, 0] == 2.0
