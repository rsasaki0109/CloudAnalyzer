"""Tests for ca.batch module."""

import open3d as o3d
import pytest

from ca.batch import batch_info


class TestBatchInfo:
    def test_finds_files(self, tmp_path, simple_pcd):
        for name in ["a.pcd", "b.pcd", "c.pcd"]:
            o3d.io.write_point_cloud(str(tmp_path / name), simple_pcd)
        results = batch_info(str(tmp_path))
        assert len(results) == 3

    def test_recursive(self, tmp_path, simple_pcd):
        sub = tmp_path / "sub"
        sub.mkdir()
        o3d.io.write_point_cloud(str(tmp_path / "a.pcd"), simple_pcd)
        o3d.io.write_point_cloud(str(sub / "b.pcd"), simple_pcd)
        results = batch_info(str(tmp_path), recursive=True)
        assert len(results) == 2

    def test_no_recursive(self, tmp_path, simple_pcd):
        sub = tmp_path / "sub"
        sub.mkdir()
        o3d.io.write_point_cloud(str(tmp_path / "a.pcd"), simple_pcd)
        o3d.io.write_point_cloud(str(sub / "b.pcd"), simple_pcd)
        results = batch_info(str(tmp_path), recursive=False)
        assert len(results) == 1

    def test_empty_directory(self, tmp_path):
        results = batch_info(str(tmp_path))
        assert results == []

    def test_dir_not_found(self):
        with pytest.raises(FileNotFoundError):
            batch_info("/no/such/dir")
