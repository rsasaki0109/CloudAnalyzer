"""LAS/LAZ fallback coordinates retain scaling, order and owned point data."""

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import ca.io as io


def las_module(extension):
    laspy = pytest.importorskip("laspy")
    if extension == ".laz" and not any(backend.is_available() for backend in laspy.LazBackend):
        pytest.skip("No LAZ decoder installed")
    return laspy


@pytest.mark.parametrize("extension", [".las", ".laz"])
@pytest.mark.parametrize("point_format", [3, 7])
@pytest.mark.parametrize("scaling", ["survey", "binary"])
@pytest.mark.parametrize("size", [1, 19])
def test_scaled_xyz_bytes_and_attributes(tmp_path, monkeypatch, extension, point_format, scaling, size):
    laspy = las_module(extension)
    monkeypatch.setattr(io, "core", lambda: None)
    header = laspy.LasHeader(point_format=point_format, version="1.2" if point_format == 3 else "1.4")
    if scaling == "survey":
        header.scales = [0.00025, 0.01, 0.001]
        header.offsets = [-9290.61925125122, 851209.9, -33.56099872589111]
    else:
        header.scales = [2.0**-12, 2.0**-7, 2.0**-20]
        header.offsets = [0.0, -0.0, 1e12]
    raw = np.resize(np.array([
        [np.iinfo(np.int32).min, 0, np.iinfo(np.int32).max],
        [-17, 23, -31], [0, -1, 1], [123456, -789012, 345678],
    ], dtype=np.int32), (size, 3))
    las = laspy.LasData(header)
    las.X, las.Y, las.Z = raw[:, 0], raw[:, 1], raw[:, 2]
    las.intensity = np.arange(size, dtype=np.uint16)
    las.classification = np.arange(size, dtype=np.uint8) % 16
    las.red = np.full(size, 65535, dtype=np.uint16)
    las.gps_time = np.arange(size, dtype=float) + 1234.5
    path = tmp_path / ("測量座標" + extension.upper())
    las.write(path, do_compress=extension == ".laz")
    before = hashlib.sha256(path.read_bytes()).digest()
    expected = raw.astype(np.float64) * header.scales + header.offsets

    cloud = io.load_point_cloud(str(path))

    assert np.asarray(cloud.points).tobytes() == expected.tobytes()
    assert not cloud.has_colors() and not cloud.has_normals() and not cloud.has_covariances()
    assert hashlib.sha256(path.read_bytes()).digest() == before


@pytest.mark.parametrize("extension", [".las", ".laz"])
def test_returned_coordinates_own_their_storage(tmp_path, monkeypatch, extension):
    laspy = las_module(extension)
    monkeypatch.setattr(io, "core", lambda: None)
    las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    las.X, las.Y, las.Z = np.arange(7), np.arange(7) * -2, np.arange(7) * 3
    path = tmp_path / ("cloud" + extension)
    las.write(path)
    actual_read = laspy.read
    retained = []

    def capture_read(*args, **kwargs):
        result = actual_read(*args, **kwargs)
        retained.append(result)
        return result

    monkeypatch.setattr(laspy, "read", capture_read)
    cloud = io.load_point_cloud(str(path))
    expected = np.asarray(cloud.points).tobytes()
    retained[0].X[:] = 999
    retained[0].Y[:] = -999
    retained.clear()
    path.unlink()
    assert np.asarray(cloud.points).tobytes() == expected


@pytest.mark.parametrize("extension", [".las", ".laz"])
def test_empty_las_retains_error(tmp_path, monkeypatch, extension):
    laspy = las_module(extension)
    monkeypatch.setattr(io, "core", lambda: None)
    path = tmp_path / ("empty" + extension)
    laspy.LasData(laspy.LasHeader(point_format=3, version="1.2")).write(path)
    with pytest.raises(ValueError, match="Point cloud is empty"):
        io.load_point_cloud(str(path))


def test_native_reader_still_bypasses_laspy(tmp_path, monkeypatch):
    path = tmp_path / "native.las"
    path.touch()
    expected = np.array([[1.5, -2.0, 3.25], [-4.0, 5.5, 6.75]])
    calls = []

    def read(actual):
        calls.append(actual)
        return {"positions": expected}

    monkeypatch.setattr(io, "core", lambda: SimpleNamespace(read=read))
    monkeypatch.setitem(sys.modules, "laspy", None)
    cloud = io.load_point_cloud(str(path))
    assert calls == [str(path)]
    assert np.asarray(cloud.points).tobytes() == expected.tobytes()
