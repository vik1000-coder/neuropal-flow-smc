"""Section 17.7: schema tests."""
import numpy as np
import pytest

from sid_neuromod.data.schema import NeuralDataset, SchemaError, validate_core


def _good():
    T, N = 100, 3
    return (np.zeros((T, N)), np.arange(T, dtype=float),
            np.array([f"n{i}" for i in range(N)]))


def test_schema_accepts_valid():
    X, t, ids = _good()
    ds = NeuralDataset(X=X, timestamps_s=t, neuron_ids=ids)
    assert ds.T == 100 and ds.N == 3


def test_schema_rejects_nonmonotonic_timestamps():
    X, t, ids = _good()
    t2 = t.copy(); t2[50] = t2[49]  # duplicate -> not strictly increasing
    with pytest.raises(SchemaError):
        validate_core(X, t2, ids)
    t3 = t.copy(); t3[50] = t3[48] - 1  # decrease
    with pytest.raises(SchemaError):
        validate_core(X, t3, ids)


def test_schema_rejects_nan_timestamps():
    X, t, ids = _good()
    t2 = t.copy(); t2[10] = np.nan
    with pytest.raises(SchemaError):
        validate_core(X, t2, ids)


def test_schema_rejects_shape_mismatch():
    X, t, ids = _good()
    with pytest.raises(SchemaError):
        validate_core(X, t[:-1], ids)          # timestamps too short
    with pytest.raises(SchemaError):
        validate_core(X, t, ids[:-1])          # neuron_ids too short
    with pytest.raises(SchemaError):
        validate_core(X[:, 0], t, ids)         # X not 2D


def test_schema_rejects_bad_behavior_shape():
    X, t, ids = _good()
    with pytest.raises(SchemaError):
        NeuralDataset(X=X, timestamps_s=t, neuron_ids=ids,
                      behavior=np.zeros((50, 2)))  # wrong T


def test_schema_is_valid_shapes():
    X, t, ids = _good()
    NeuralDataset(X=X, timestamps_s=t, neuron_ids=ids, is_valid=np.ones(100, bool))
    NeuralDataset(X=X, timestamps_s=t, neuron_ids=ids, is_valid=np.ones((100, 3), bool))
    with pytest.raises(SchemaError):
        NeuralDataset(X=X, timestamps_s=t, neuron_ids=ids, is_valid=np.ones((100, 2), bool))
