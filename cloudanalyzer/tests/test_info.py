"""Tests for ca.info module."""

from ca.info import get_info


class TestGetInfo:
    def test_basic_info(self, sample_pcd_file):
        info = get_info(sample_pcd_file)
        assert info["num_points"] == 100
        assert info["path"] == sample_pcd_file
        assert len(info["bbox_min"]) == 3
        assert len(info["bbox_max"]) == 3
        assert len(info["extent"]) == 3
        assert len(info["robust_bbox_min"]) == 3
        assert len(info["robust_bbox_max"]) == 3
        assert len(info["robust_extent"]) == 3
        assert len(info["centroid"]) == 3
        assert info["robust_percentiles"] == [1.0, 99.0]

    def test_extent_positive(self, sample_pcd_file):
        info = get_info(sample_pcd_file)
        for v in info["extent"]:
            assert v >= 0.0

    def test_bbox_order(self, sample_pcd_file):
        info = get_info(sample_pcd_file)
        for lo, hi in zip(info["bbox_min"], info["bbox_max"]):
            assert lo <= hi
        for lo, hi in zip(info["robust_bbox_min"], info["robust_bbox_max"]):
            assert lo <= hi

    def test_outside_robust_bbox(self, sample_pcd_file):
        info = get_info(sample_pcd_file)
        assert info["outside_robust_bbox_count"] >= 0
        assert 0.0 <= info["outside_robust_bbox_ratio"] <= 1.0

    def test_has_colors_normals(self, sample_pcd_file):
        info = get_info(sample_pcd_file)
        assert isinstance(info["has_colors"], bool)
        assert isinstance(info["has_normals"], bool)
