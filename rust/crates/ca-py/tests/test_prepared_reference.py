"""Reference ownership and exact repeated parallel nearest-distance queries."""

from concurrent.futures import ThreadPoolExecutor
import gc

import cloudanalyzer_core as cc
import numpy as np
import pytest


def points(count, offset=500000.):
    i = np.arange(count, dtype=np.float64)
    return np.column_stack([offset + (i * 19 % 113) / 17,
                            offset + (i * 37 % 127) / 23, (i * 53 % 89) / 13])


def assert_bits(actual, expected):
    assert actual.dtype == np.float64 and actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("count", [0, 1, 4095, 4096, 4097, 65535, 65536, 65537])
def test_changed_batches_preserve_bits_and_capacity(count):
    reference = points(273)
    prepared = cc.PreparedReference(reference)
    capacity = prepared.retained_bytes
    for source in [points(count, 500000.125), points(count, 500001.)[::-1]]:
        assert_bits(prepared.distances(source), cc.nearest_distances(source, reference))
        assert prepared.retained_bytes == capacity


def test_owned_strided_reference_and_independent_outputs():
    reference = points(33)
    backing = np.zeros((33, 6))
    backing[:, ::2] = reference
    prepared = cc.PreparedReference(backing[:, ::2])
    backing[:] = -1e20
    del backing
    gc.collect()
    source = points(4097, 500000.125)[::-1]
    source.setflags(write=False)
    expected = cc.nearest_distances(source, reference)
    result = prepared.distances(source)
    assert_bits(result, expected)
    result[:] = 1234.
    assert_bits(prepared.distances(source), expected)
    with pytest.raises(AttributeError):
        prepared.retained_bytes = 0
    with pytest.raises(AttributeError):
        prepared.target = reference


@pytest.mark.parametrize("scale", [1e-160, 1., 1e150, 1e200])
def test_duplicates_signed_zero_ties_and_extreme_coordinates(scale):
    reference = np.array([[0., -0., 0.], [-1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]) * scale
    source = np.array([[0., 0., 0.], [-0., -0., -0.], [0., 1., 0.], [.25, .25, .25]]) * scale
    prepared = cc.PreparedReference(reference)
    assert_bits(prepared.distances(source), cc.nearest_distances(source, reference))


@pytest.mark.parametrize("reference", [np.array([[0., 0., 0.], [np.nan, 0., 0.], [np.inf]*3]),
                                      np.array([[np.nan]*3, [np.inf]*3])])
def test_nonfinite_and_overflow_queries_match_existing_api(reference):
    source = np.tile([[np.nan, 0., 0.], [np.inf]*3, [1e308]*3, [0., 0., 0.], [.25]*3], (900, 1))
    prepared = cc.PreparedReference(reference)
    for batch in [source, source[::-1]]:
        assert_bits(prepared.distances(batch), cc.nearest_distances(batch, reference))


def test_empty_invalid_dtype_and_shape_errors():
    with pytest.raises(ValueError, match="target is empty"):
        cc.PreparedReference(np.empty((0, 3)))
    with pytest.raises(TypeError):
        cc.PreparedReference(points(3).astype(np.float32))
    with pytest.raises(ValueError, match="expected an"):
        cc.PreparedReference(np.zeros((3, 2)))
    prepared = cc.PreparedReference(points(3))
    with pytest.raises(ValueError, match="expected an"):
        prepared.distances(np.zeros((3, 4)))
    with pytest.raises(TypeError):
        prepared.distances(np.zeros(3))
    with pytest.raises(TypeError):
        prepared.distances(points(3).astype(np.float32))


def test_concurrent_readers_keep_independent_query_state():
    reference = points(273)
    prepared = cc.PreparedReference(reference)
    capacity = prepared.retained_bytes
    batches = [points(count, 500000.125 + i) for i, count in enumerate([1, 4095, 4097, 8193])]
    expected = [cc.nearest_distances(batch, reference) for batch in batches]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(prepared.distances, batch) for batch in batches]
        for future, values in zip(futures, expected):
            assert_bits(future.result(), values)
    assert prepared.retained_bytes == capacity
