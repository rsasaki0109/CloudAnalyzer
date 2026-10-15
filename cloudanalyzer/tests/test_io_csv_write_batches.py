"""CSV export retains its exact text and a stable coordinate snapshot."""

import csv
import io

import numpy as np
import open3d as o3d
import pytest

import ca.io as point_io


def cloud(points):
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float).reshape(-1, 3))
    return result


def preceding_bytes(points):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["x", "y", "z"])
    writer.writerows(np.asarray(points, dtype=float).reshape(-1, 3).tolist())
    return stream.getvalue().encode("utf-8")


@pytest.mark.parametrize("batch_size", [1, 3, 65_536])
@pytest.mark.parametrize("points", [
    [],
    [[1.25, -2.5, 3.0]],
    [[-0.0, 0.0, np.nextafter(0.0, 1.0)], [np.finfo(float).max, -np.finfo(float).max, 1e-300],
     [float("nan"), float("inf"), -float("inf")], [1e300, 1e-5, -0.125]],
    np.arange(33, dtype=float).reshape(-1, 3),
])
def test_exact_csv_bytes(tmp_path, monkeypatch, batch_size, points):
    monkeypatch.setattr(point_io, "_CSV_WRITE_BATCH_SIZE", batch_size)
    output = tmp_path / "点群" / "export.CSV"
    pcd = cloud(points)
    before = np.asarray(pcd.points).tobytes()

    assert point_io.save_point_cloud(str(output), pcd) is None

    assert output.read_bytes() == preceding_bytes(points)
    assert np.asarray(pcd.points).tobytes() == before


@pytest.mark.parametrize("count", [65_535, 65_536, 65_537, 131_073])
def test_default_batch_boundaries(tmp_path, count):
    points = np.arange(count * 3, dtype=float).reshape(-1, 3) / 8
    output = tmp_path / "points.csv"
    point_io.save_point_cloud(str(output), cloud(points))
    assert output.read_bytes() == preceding_bytes(points)


def test_only_xyz_exported_and_attributes_preserved(tmp_path):
    pcd = cloud([[1, 2, 3], [4, 5, 6]])
    pcd.colors = o3d.utility.Vector3dVector([[0.25, 0.5, 1.0], [1.0, 0.0, 0.5]])
    pcd.normals = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0]])
    pcd.covariances = o3d.utility.Matrix3dVector([np.eye(3), 2 * np.eye(3)])
    before = [np.asarray(getattr(pcd, attr)).tobytes() for attr in ["points", "colors", "normals", "covariances"]]
    output = tmp_path / "points.csv"
    output.write_bytes(b"old contents that must be replaced")
    point_io.save_point_cloud(str(output), pcd)
    assert output.read_bytes() == b"x,y,z\r\n1.0,2.0,3.0\r\n4.0,5.0,6.0\r\n"
    assert before == [np.asarray(getattr(pcd, attr)).tobytes() for attr in ["points", "colors", "normals", "covariances"]]


def test_snapshot_survives_source_mutation_during_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(point_io, "_CSV_WRITE_BATCH_SIZE", 2)
    points = np.arange(21, dtype=float).reshape(-1, 3)
    pcd = cloud(points)
    original_writer = csv.writer

    class MutatingWriter:
        def __init__(self, stream):
            self.delegate = original_writer(stream)

        def writerow(self, row):
            return self.delegate.writerow(row)

        def writerows(self, rows):
            self.delegate.writerows(rows)
            np.asarray(pcd.points)[:] = -1000

    monkeypatch.setattr(point_io.csv, "writer", MutatingWriter)
    output = tmp_path / "points.csv"
    point_io.save_point_cloud(str(output), pcd)
    monkeypatch.setattr(point_io.csv, "writer", original_writer)
    assert output.read_bytes() == preceding_bytes(points)
    assert (np.asarray(pcd.points) == -1000).all()


def test_write_failure_closes_file_and_propagates(tmp_path, monkeypatch):
    opened = []

    class FailingWriter:
        def __init__(self, stream):
            opened.append(stream)

        def writerow(self, row):
            pass

        def writerows(self, rows):
            raise OSError("output unavailable")

    monkeypatch.setattr(point_io.csv, "writer", FailingWriter)
    with pytest.raises(OSError, match="output unavailable"):
        point_io.save_point_cloud(str(tmp_path / "points.csv"), cloud([[1, 2, 3]]))
    assert len(opened) == 1 and opened[0].closed
