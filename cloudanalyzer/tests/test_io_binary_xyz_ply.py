"""Bulk XYZ loading preserves Open3D coordinates, ownership and fallback layouts."""

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

import ca.io as io


def write_xyz(path, points, *, newline=b'\n', comments=(), encoding=b'binary_little_endian', scalar=b'double'):
    lines = [b'ply', b'format ' + encoding + b' 1.0', *comments,
             b'element vertex ' + str(len(points)).encode(),
             *[b'property ' + scalar + b' ' + axis for axis in [b'x', b'y', b'z']], b'end_header']
    header = newline.join(lines) + newline
    path.write_bytes(header + np.asarray(points).tobytes())
    return header


def signature(cloud):
    return tuple((np.asarray(getattr(cloud, name)).shape, np.asarray(getattr(cloud, name)).tobytes())
                 for name in ['points', 'normals', 'colors', 'covariances'])


@pytest.mark.parametrize('newline', [b'\n', b'\r\n'])
@pytest.mark.parametrize('kind', ['finite', 'special', 'one', 'duplicates'])
def test_bulk_loaded_bits_and_fields_match_open3d(tmp_path, monkeypatch, newline, kind):
    if kind == 'special':
        bits = np.array([0, 0x8000000000000000, 1, 0x8000000000000001,
                         0x7ff0000000000000, 0xfff0000000000000,
                         0x7ff8000000000001, 0xfff8000000000042, 0x7ff0000000000001], dtype='<u8')
        points = bits.view('<f8').reshape(-1, 3)
    elif kind == 'one':
        points = np.array([[1., -2., 3.]], dtype='<f8')
    elif kind == 'duplicates':
        points = np.zeros((100, 3), dtype='<f8')
    else:
        points = np.random.default_rng(72).normal(size=(193, 3)).astype('<f8')
    path = tmp_path / '点群 XYZ.PLY'
    write_xyz(path, points, newline=newline, comments=[b'comment measured-point layout', b'comment end_header is text'])
    expected = o3d.io.read_point_cloud(str(path))
    def unexpected(*args, **kwargs):
        raise AssertionError('eligible layout must use bulk loading')
    monkeypatch.setattr(o3d.io, 'read_point_cloud', unexpected)
    observed = io.load_point_cloud(str(path))
    assert signature(observed) == signature(expected)
    assert np.asarray(observed.points).tobytes() == np.asarray(expected.points).tobytes()


def test_array_and_file_lifetime_are_independent_from_cloud(tmp_path, monkeypatch):
    path = tmp_path / 'cloud.ply'
    points = np.arange(300, dtype='<f8').reshape(-1, 3)
    header = write_xyz(path, points)
    fromfile, arrays = np.fromfile, []
    def tracked(*args, **kwargs):
        values = fromfile(*args, **kwargs)
        arrays.append(values)
        return values
    monkeypatch.setattr(np, 'fromfile', tracked)
    first, second = io.load_point_cloud(str(path)), io.load_point_cloud(str(path))
    assert not np.shares_memory(np.asarray(first.points), arrays[0])
    arrays[0][:] = 123.
    np.asarray(second.points)[:] = 456.
    path.write_bytes(header + np.ones_like(points).tobytes())
    assert np.asarray(first.points).tobytes() == points.tobytes()


@pytest.mark.parametrize('layout', ['ascii', 'float32', 'big_endian', 'normals', 'colors', 'both', 'faces', 'reordered'])
def test_other_valid_layouts_preserve_all_open3d_fields(tmp_path, simple_pcd, monkeypatch, layout):
    path = tmp_path / 'cloud.ply'
    points = np.asarray(simple_pcd.points)
    if layout in ['normals', 'both']:
        simple_pcd.normals = o3d.utility.Vector3dVector(np.tile([0., 0., 1.], (len(points), 1)))
    if layout in ['colors', 'both']:
        simple_pcd.colors = o3d.utility.Vector3dVector(np.tile([.2, .4, .6], (len(points), 1)))
    if layout == 'float32':
        write_xyz(path, points.astype('<f4'), scalar=b'float')
    elif layout == 'big_endian':
        write_xyz(path, points.astype('>f8'), encoding=b'binary_big_endian')
    elif layout == 'faces':
        mesh = o3d.geometry.TriangleMesh.create_box()
        assert o3d.io.write_triangle_mesh(str(path), mesh)
    elif layout == 'reordered':
        header = write_xyz(path, points[:, [1, 0, 2]].copy())
        path.write_bytes(path.read_bytes().replace(b'property double x\nproperty double y', b'property double y\nproperty double x'))
    else:
        assert o3d.io.write_point_cloud(str(path), simple_pcd, write_ascii=layout == 'ascii')
    native, calls = o3d.io.read_point_cloud, []
    expected = native(str(path))
    def tracked(*args, **kwargs):
        calls.append(args[0])
        return native(*args, **kwargs)
    monkeypatch.setattr(o3d.io, 'read_point_cloud', tracked)
    assert signature(io.load_point_cloud(str(path))) == signature(expected)
    assert calls == [str(path)]


@pytest.mark.parametrize('kind', ['magic', 'format_version', 'negative_count', 'huge_count', 'extra_property',
                                 'extra_element', 'missing_end', 'long_line', 'many_lines', 'header_bytes',
                                 'truncated', 'trailing', 'payload_limit', 'empty'])
def test_ineligible_or_malformed_headers_delegate_without_bulk_allocation(tmp_path, simple_pcd, monkeypatch, kind):
    path = tmp_path / 'cloud.ply'
    points = np.arange(9, dtype='<f8').reshape(-1, 3)
    header = write_xyz(path, points)
    raw = path.read_bytes()
    if kind == 'magic': raw = raw.replace(b'ply\n', b'bad\n', 1)
    elif kind == 'format_version': raw = raw.replace(b'endian 1.0', b'endian 2.0')
    elif kind == 'negative_count': raw = raw.replace(b'vertex 3', b'vertex -3')
    elif kind == 'huge_count': raw = raw.replace(b'vertex 3', b'vertex ' + b'9' * 200)
    elif kind == 'extra_property': raw = raw.replace(b'end_header', b'property double intensity\nend_header')
    elif kind == 'extra_element': raw = raw.replace(b'end_header', b'element edge 0\nend_header')
    elif kind == 'missing_end': raw = raw.replace(b'end_header\n', b'')
    elif kind == 'long_line': raw = raw.replace(b'element vertex', b'comment ' + b'x' * 4097 + b'\nelement vertex')
    elif kind == 'many_lines': raw = raw.replace(b'element vertex', b'comment x\n' * 130 + b'element vertex')
    elif kind == 'header_bytes': raw = raw.replace(b'element vertex', (b'comment ' + b'x' * 4000 + b'\n') * 17 + b'element vertex')
    elif kind == 'truncated': raw = raw[:-1]
    elif kind == 'trailing': raw += b'x'
    elif kind == 'payload_limit': monkeypatch.setattr(io, '_BINARY_XYZ_PLY_MAX_BYTES', 24)
    elif kind == 'empty': raw = header.replace(b'vertex 3', b'vertex 0')
    path.write_bytes(raw)
    calls = []
    def fallback(actual):
        calls.append(actual)
        return o3d.geometry.PointCloud(simple_pcd)
    def unexpected(*args, **kwargs):
        raise AssertionError('layout checks must precede bulk allocation')
    monkeypatch.setattr(o3d.io, 'read_point_cloud', fallback)
    monkeypatch.setattr(np, 'fromfile', unexpected)
    assert signature(io.load_point_cloud(str(path))) == signature(simple_pcd)
    assert calls == [str(path)]


def test_short_read_retries_with_open3d(tmp_path, simple_pcd, monkeypatch):
    path = tmp_path / 'short.ply'
    write_xyz(path, np.zeros((3, 3), dtype='<f8'))
    monkeypatch.setattr(np, 'fromfile', lambda *args, **kwargs: np.zeros(8))
    calls = []
    def fallback(actual):
        calls.append(actual)
        return o3d.geometry.PointCloud(simple_pcd)
    monkeypatch.setattr(o3d.io, 'read_point_cloud', fallback)
    assert signature(io.load_point_cloud(str(path))) == signature(simple_pcd)
    assert calls == [str(path)]


def test_open_error_keeps_open3d_error_handling(tmp_path, monkeypatch):
    path = tmp_path / 'unreadable.ply'
    path.write_bytes(b'ply\n')
    original, calls = Path.open, []
    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError('denied')
        return original(self, *args, **kwargs)
    def fallback(actual):
        calls.append(actual)
        return o3d.geometry.PointCloud()
    monkeypatch.setattr(Path, 'open', denied)
    monkeypatch.setattr(o3d.io, 'read_point_cloud', fallback)
    with pytest.raises(ValueError, match='Point cloud is empty'):
        io.load_point_cloud(str(path))
    assert calls == [str(path)]


def test_empty_layout_keeps_existing_error(tmp_path):
    path = tmp_path / 'empty.ply'
    write_xyz(path, np.empty((0, 3), dtype='<f8'))
    with pytest.raises(ValueError, match='Point cloud is empty'):
        io.load_point_cloud(str(path))
