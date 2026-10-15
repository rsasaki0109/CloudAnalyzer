"""Batch finite checks retain strict whole-file CSV parsing and point bits."""

import numpy as np
import open3d as o3d
import pytest

import ca.io as io


@pytest.mark.parametrize("batch_size", [1, 3, 65_536])
@pytest.mark.parametrize("content,expected", [
    ("x,y,z\n1,2,3\n4,5,6\n", [[1, 2, 3], [4, 5, 6]]),
    ("1,2\n1,2,3\n\n4,5,6,ignored\n", [[1, 2, 3], [4, 5, 6]]),
    ('" Z "," X "," Y ",label\n3,1,2,"two\nlines"\n6,4,5,ignored\n', [[1, 2, 3], [4, 5, 6]]),
    ("point_time_sec,z_m,y_m,x_m\n0,3,2,1\n1,6,5,4\n", [[1, 2, 3], [4, 5, 6]]),
    ("x,x,y,z\n100,1,2,3\n200,4,5,6\n", [[1, 2, 3], [4, 5, 6]]),
    ("x_m,y_m,z_m,X,Y,Z\n100,200,300,1,2,3\n", [[1, 2, 3]]),
    ("x,y,z\nnan,0,1\n1,2,3\n0,-inf,1\n4,5,6\n0,1,Infinity\n7,8,9\n", [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ("-0,5e-324,-5e-324\n1_000,2.5e100,-3.25e-100\n", [[-0.0, 5e-324, -5e-324], [1000, 2.5e100, -3.25e-100]]),
    ("x,y,z\n ١ , ٢ , ٣ \n", [[1, 2, 3]]),
])
def test_exact_coordinates_and_csv_grammar(tmp_path, monkeypatch, batch_size, content, expected):
    monkeypatch.setattr(io, "_CSV_POINT_BATCH_SIZE", batch_size)
    path = tmp_path / "点群.CSV"
    path.write_bytes(content.encode("utf-8"))
    cloud = io.load_point_cloud(str(path))
    assert np.asarray(cloud.points).tobytes() == np.asarray(expected, dtype=float).tobytes()
    assert not cloud.has_colors() and not cloud.has_normals() and not cloud.has_covariances()
    assert path.read_bytes() == content.encode("utf-8")


@pytest.mark.parametrize("content,error", [
    ("", ValueError),
    ("\n", ValueError),
    ("a,b,c\n1,2,3\n", ValueError),
    ('"1","2","3"\n4,5,6\n', ValueError),
    ("x,y,z\n1,2,3\n4,5,6\n7,8,9\n10,11,12\nno,5,6\n", ValueError),
    ("1,2,3\n4,5,6\n7,8,9\n10,11,12\nno,5,6\n", ValueError),
    ("x,y,z\n1,2,3\n4,5,6\n7,8,9\n1,2\n", TypeError),
])
def test_parse_errors_still_propagate_after_a_flush(tmp_path, monkeypatch, content, error):
    monkeypatch.setattr(io, "_CSV_POINT_BATCH_SIZE", 3)
    path = tmp_path / "invalid.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(error):
        io.load_point_cloud(str(path))


@pytest.mark.parametrize("content", ["x,y,z\n", "nan,0,1\n0,inf,1\n", "1,2\n3,4\n"])
def test_no_accepted_rows_retains_original_empty_conversion(tmp_path, monkeypatch, content):
    monkeypatch.setattr(io, "_CSV_POINT_BATCH_SIZE", 1)
    path = tmp_path / "no-points.csv"
    path.write_text(content, encoding="utf-8")
    # Open3D versions may reject the original 1-D empty input themselves.
    try:
        o3d.utility.Vector3dVector(np.asarray([], dtype=float))
    except RuntimeError:
        error = RuntimeError
    else:
        error = ValueError
    with pytest.raises(error):
        io.load_point_cloud(str(path))


def test_finite_filter_batches_are_bounded_and_preserve_order(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "_CSV_POINT_BATCH_SIZE", 3)
    path = tmp_path / "batches.csv"
    path.write_text("x,y,z\n" + "nan,0,0\n1,2,3\n4,5,6\n" * 7, encoding="utf-8")
    observed_sizes = []
    original = io._filter_chunk

    def observed(points, bounds):
        observed_sizes.append(len(points))
        return original(points, bounds)

    monkeypatch.setattr(io, "_filter_chunk", observed)
    cloud = io.load_point_cloud(str(path))
    assert max(observed_sizes) <= 3
    assert sum(observed_sizes) == 21
    assert np.asarray(cloud.points).tobytes() == np.tile([[1., 2., 3.], [4., 5., 6.]], (7, 1)).tobytes()


def test_multiple_loads_and_source_file_have_independent_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "_CSV_POINT_BATCH_SIZE", 3)
    path = tmp_path / "owned.csv"
    path.write_text("1,2,3\n4,5,6\n7,8,9\n10,11,12\n", encoding="utf-8")
    first = io.load_point_cloud(str(path))
    expected = np.asarray(first.points).tobytes()
    path.write_text("999,999,999\n", encoding="utf-8")
    second = io.load_point_cloud(str(path))
    np.asarray(second.points)[:] = -123
    path.unlink()
    assert np.asarray(first.points).tobytes() == expected


def test_csv_still_bypasses_optional_native_dispatch(tmp_path, monkeypatch):
    path = tmp_path / "plain.csv"
    path.write_text("1,2,3\n", encoding="utf-8")

    def unexpected():
        raise AssertionError("CSV must retain the standard parser")

    monkeypatch.setattr(io, "core", unexpected)
    assert np.asarray(io.load_point_cloud(str(path)).points).tobytes() == np.array([[1., 2., 3.]]).tobytes()
