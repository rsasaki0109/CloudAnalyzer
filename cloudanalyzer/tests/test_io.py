"""Tests for ca.io module."""

import numpy as np
import pytest
import open3d as o3d

from ca.io import (
    PointAccumulator,
    PointChunkReader,
    SUPPORTED_EXTENSIONS,
    iter_point_chunks,
    load_point_cloud,
    save_point_cloud,
)


class TestLoadPointCloud:
    def test_load_pcd(self, sample_pcd_file):
        pcd = load_point_cloud(sample_pcd_file)
        assert isinstance(pcd, o3d.geometry.PointCloud)
        assert len(pcd.points) == 100

    def test_load_ply(self, tmp_path, simple_pcd):
        path = tmp_path / "test.ply"
        o3d.io.write_point_cloud(str(path), simple_pcd)
        pcd = load_point_cloud(str(path))
        assert len(pcd.points) == 100

    def test_load_xyz_csv(self, tmp_path):
        path = tmp_path / "scan.csv"
        path.write_text("x,y,z\n1,2,3\n4,5,6\n", encoding="utf-8")

        pcd = load_point_cloud(str(path))

        np.testing.assert_allclose(np.asarray(pcd.points), [[1, 2, 3], [4, 5, 6]])

    def test_load_glim_scan_csv(self, tmp_path):
        path = tmp_path / "glim_scan.csv"
        path.write_text(
            "x_m,y_m,z_m,point_time_sec\n"
            "1.5,2.5,3.5,0.0\n"
            "4.5,5.5,6.5,0.01\n",
            encoding="utf-8",
        )

        pcd = load_point_cloud(str(path))

        np.testing.assert_allclose(
            np.asarray(pcd.points),
            [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
        )

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_point_cloud("/nonexistent/file.pcd")

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_point_cloud(str(bad_file))

    def test_empty_point_cloud(self, tmp_path):
        # Open3D can't write an empty PCD, so create a minimal valid file manually
        path = tmp_path / "empty.pcd"
        path.write_text(
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            "WIDTH 0\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            "POINTS 0\n"
            "DATA ascii\n"
        )
        with pytest.raises(ValueError, match="empty"):
            load_point_cloud(str(path))

    def test_supported_extensions(self):
        assert ".pcd" in SUPPORTED_EXTENSIONS
        assert ".ply" in SUPPORTED_EXTENSIONS
        assert ".las" in SUPPORTED_EXTENSIONS
        assert ".laz" in SUPPORTED_EXTENSIONS
        assert ".csv" in SUPPORTED_EXTENSIONS


class TestPointChunkReader:
    def test_csv_chunks_include_first_numeric_row(self, tmp_path):
        path = tmp_path / "points.csv"
        path.write_text("0,1,2\n3,4,5\n6,7,8\n", encoding="utf-8")

        chunks = list(iter_point_chunks(str(path), chunk_size=2))

        assert [len(chunk) for chunk in chunks] == [2, 1]
        np.testing.assert_allclose(np.vstack(chunks), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    def test_csv_chunks_support_bounds(self, tmp_path):
        path = tmp_path / "points.csv"
        path.write_text("x,y,z\n0,0,0\n1,2,3\n4,5,6\n", encoding="utf-8")

        chunks = list(
            PointChunkReader(
                str(path),
                chunk_size=10,
                bounds=(0, 0, 0, 2, 3, 3),
            )
        )

        np.testing.assert_allclose(np.vstack(chunks), [[0, 0, 0], [1, 2, 3]])

    def test_accumulator_matches_batch_moments(self):
        points = np.arange(30, dtype=float).reshape(10, 3)
        accumulator = PointAccumulator()
        accumulator.update(points[:3]).update(points[3:7]).update(points[7:])

        summary = accumulator.finalize()
        np.testing.assert_allclose(summary["mean"], np.mean(points, axis=0))
        np.testing.assert_allclose(summary["covariance"], np.cov(points, rowvar=False))
        np.testing.assert_allclose(summary["minimum"], points.min(axis=0))
        np.testing.assert_allclose(summary["maximum"], points.max(axis=0))

    def test_remote_copc_reports_optional_backend(self, monkeypatch):
        monkeypatch.setattr("ca.io.core", lambda: None)
        with pytest.raises(ValueError, match="updated Rust core"):
            list(iter_point_chunks("https://example.invalid/map.copc.laz"))


class TestSavePointCloud:
    def test_laz_roundtrip_preserves_coordinates(self, tmp_path, simple_pcd):
        laspy = pytest.importorskip("laspy")
        # LAZ support depends on optional compression backends (e.g., lazrs/laszip).
        if not any(backend.is_available() for backend in laspy.LazBackend):
            pytest.skip("No LAZ backend available for laspy; skipping .laz roundtrip test.")
        path = str(tmp_path / "test.laz")
        save_point_cloud(path, simple_pcd)
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(
            np.asarray(loaded.points),
            np.asarray(simple_pcd.points),
            atol=1e-5,
        )

    def test_las_roundtrip_preserves_coordinates(self, tmp_path, simple_pcd):
        path = str(tmp_path / "test.las")
        save_point_cloud(path, simple_pcd)
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(
            np.asarray(loaded.points),
            np.asarray(simple_pcd.points),
            atol=1e-5,
        )

    def test_unsupported_format_raises(self, tmp_path, simple_pcd):
        with pytest.raises(ValueError, match="Unsupported format"):
            save_point_cloud(str(tmp_path / "out.xyz"), simple_pcd)

    def test_csv_roundtrip_preserves_coordinates(self, tmp_path, simple_pcd):
        path = str(tmp_path / "test.csv")
        save_point_cloud(path, simple_pcd)
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(
            np.asarray(loaded.points),
            np.asarray(simple_pcd.points),
            atol=1e-12,
        )
