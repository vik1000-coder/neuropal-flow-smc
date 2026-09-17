import numpy as np

from conditional_neural_benchmark.neural_history_ablation import _source_roll


def test_source_roll_preserves_other_channels_and_source_window_multiset():
    history = np.arange(12 * 3 * 4, dtype=np.float32).reshape(12, 3, 4)
    worm = np.repeat([0, 1], 6)
    result = _source_roll(history, worm, source=2, rng=np.random.default_rng(17))
    np.testing.assert_array_equal(result[:, :, [0, 1, 3]], history[:, :, [0, 1, 3]])
    for value in (0, 1):
        index = worm == value
        assert sorted(result[index, :, 2].ravel()) == sorted(history[index, :, 2].ravel())
        assert not np.array_equal(result[index, :, 2], history[index, :, 2])
