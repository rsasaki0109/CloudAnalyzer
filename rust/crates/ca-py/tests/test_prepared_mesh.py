"""Binding ownership, array layout and concurrent query regression tests."""

from concurrent.futures import ThreadPoolExecutor
import gc

import cloudanalyzer_core as cc
import numpy as np
import pytest


def geometry():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    triangles = np.array([[0, 1, 2]], dtype=np.uint32)
    return vertices, triangles


def assert_bits(actual, expected):
    assert actual.dtype == np.float64
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("count", [0, 1, 4095, 4096, 4097, 65535, 65536, 65537])
@pytest.mark.parametrize("signed", [False, True])
def test_query_boundaries_and_changed_batches(count, signed):
    vertices, triangles = geometry()
    prepared = cc.PreparedMesh(vertices, triangles)
    size = prepared.retained_bytes
    queries = np.tile([[.25, .25, -.5], [.25, .25, .5]], (count // 2 + 1, 1))[:count]
    for batch in (queries, queries[::-1]):
        expected = cc.cloud_to_mesh(batch, vertices, triangles, signed=signed)
        assert_bits(prepared.distances(batch, signed=signed), expected)
        assert prepared.retained_bytes == size


def test_owned_strided_geometry_and_query_arrays():
    vertices, triangles = geometry()
    vertex_storage = np.zeros((3, 6), dtype=np.float64)
    vertex_storage[:, ::2] = vertices
    triangle_storage = np.zeros((1, 6), dtype=np.uint32)
    triangle_storage[:, ::2] = triangles
    prepared = cc.PreparedMesh(vertex_storage[:, ::2], triangle_storage[:, ::2])
    vertex_storage[:] = 12345.
    triangle_storage[:] = np.iinfo(np.uint32).max
    del vertex_storage, triangle_storage
    gc.collect()
    queries = np.array([[.25, .25, -.5], [.25, .25, .5]])[::-1, ::-1][:, ::-1]
    queries.setflags(write=False)
    assert_bits(prepared.distances(queries), cc.cloud_to_mesh(queries, vertices, triangles))
    result = prepared.distances(queries)
    result[:] = 99.
    assert_bits(prepared.distances(queries), cc.cloud_to_mesh(queries, vertices, triangles))
    with pytest.raises(AttributeError):
        prepared.retained_bytes = 0
    with pytest.raises(AttributeError):
        prepared.vertices = vertices


@pytest.mark.parametrize("signed", [False, True])
def test_missing_hits_do_not_poison_later_queries(signed):
    vertices, triangles = geometry()
    prepared = cc.PreparedMesh(vertices, triangles)
    invalid = [np.array([0x7ff8000000000023], dtype=np.uint64).view(np.float64)[0],
               np.inf, -np.inf, 1e308]
    rows = [[.25, .25, -.5]]
    expected = [-.5 if signed else .5]
    for value in invalid:
        for axis in range(3):
            row = [.25, .25, .5]
            row[axis] = value
            rows.extend([row, [.25, .25, .5], [.25, .25, -.5]])
            expected.extend([np.inf, .5, -.5 if signed else .5])
    queries = np.tile(np.array(rows), (300, 1))
    expected = np.tile(np.array(expected), 300)
    assert_bits(prepared.distances(queries, signed=signed), expected)
    assert_bits(prepared.distances(queries[::-1], signed=signed), expected[::-1].copy())


def test_invalid_triangles_are_filtered_and_empty_geometry_rejected():
    vertices, triangles = geometry()
    faces = np.concatenate([triangles, np.array([[0, 1, 999]], dtype=np.uint32)])
    queries = np.array([[.25, .25, -.5]])
    assert_bits(cc.PreparedMesh(vertices, faces).distances(queries),
                cc.cloud_to_mesh(queries, vertices, triangles))
    for v, t in [(vertices, faces[1:]), (vertices, faces[:0]), (vertices[:0], triangles)]:
        with pytest.raises(ValueError, match="mesh has no triangles"):
            cc.PreparedMesh(v, t)


@pytest.mark.parametrize("offset", [0., 500000., 1e8])
def test_f64_geometry_degenerate_faces_and_opposite_signed_ties(offset):
    vertices = np.array([[offset, offset, 0.], [offset + 1., offset, .125],
                         [offset, offset + 1., -.25], [offset + 1., offset + 1., .5]])
    triangles = np.array([[0, 1, 2], [0, 2, 1], [1, 3, 2], [3, 3, 3]], dtype=np.uint32)
    queries = np.array([[offset + .25, offset + .25, z] for z in [-.5, -.125, 0., .125, .5]])
    prepared = cc.PreparedMesh(vertices, triangles)
    for signed in [False, True]:
        assert_bits(prepared.distances(queries, signed=signed),
                    cc.cloud_to_mesh(queries, vertices, triangles, signed=signed))


@pytest.mark.parametrize("bad_vertex", [np.nan, np.inf])
def test_mesh_with_no_computable_hit_returns_positive_infinity(bad_vertex):
    vertices, triangles = geometry()
    vertices[:] = bad_vertex
    prepared = cc.PreparedMesh(vertices, triangles)
    queries = np.array([[.25, .25, -.5], [.25, .25, .5]])
    for signed in [False, True]:
        expected = cc.cloud_to_mesh(queries, vertices, triangles, signed=signed)
        assert_bits(prepared.distances(queries, signed=signed), expected)
        assert_bits(expected, np.full(2, np.inf))


@pytest.mark.parametrize("bad", [np.zeros((3, 2)), np.zeros((3, 4))])
def test_shape_errors(bad):
    vertices, triangles = geometry()
    with pytest.raises(ValueError, match="expected an"):
        cc.PreparedMesh(bad, triangles)
    with pytest.raises(ValueError, match="triangles must be"):
        cc.PreparedMesh(vertices, bad.astype(np.uint32))
    with pytest.raises(ValueError, match="expected an"):
        cc.PreparedMesh(vertices, triangles).distances(bad)


def test_dtype_and_rank_errors():
    vertices, triangles = geometry()
    with pytest.raises(TypeError):
        cc.PreparedMesh(vertices.astype(np.float32), triangles)
    with pytest.raises(TypeError):
        cc.PreparedMesh(vertices, triangles.astype(np.int64))
    prepared = cc.PreparedMesh(vertices, triangles)
    with pytest.raises(TypeError):
        prepared.distances(vertices.astype(np.float32))
    with pytest.raises(TypeError):
        prepared.distances(np.zeros(3))


def test_concurrent_readers_share_an_immutable_mesh():
    vertices, triangles = geometry()
    prepared = cc.PreparedMesh(vertices, triangles)
    size = prepared.retained_bytes
    batches = [np.tile([[.25, .25, -.5], [.25, .25, .5]], (4097, 1))[::step]
               for step in [1, 2, 3, 4]]
    jobs = [(batch, signed) for signed in [False, True] for batch in batches]
    expected = [cc.cloud_to_mesh(batch, vertices, triangles, signed=signed)
                for batch, signed in jobs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(prepared.distances, batch, signed=signed) for batch, signed in jobs]
        for future, baseline in zip(futures, expected):
            assert_bits(future.result(), baseline)
    assert prepared.retained_bytes == size
