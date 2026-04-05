"""Tests for ground segmentation evaluation."""

import numpy as np
import open3d as o3d
import pytest

from ca.ground_evaluate import evaluate_ground_segmentation


def _write_pcd(path, points: list[list[float]]) -> str:
    """Write a minimal PCD file from point coordinates."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pcd)
    return str(path)


class TestGroundEvaluate:
    def test_perfect_segmentation(self, tmp_path):
        ground = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        nonground = [[0, 0, 2], [1, 0, 2], [2, 0, 2]]
        est_ground = _write_pcd(tmp_path / "est_ground.pcd", ground)
        est_nonground = _write_pcd(tmp_path / "est_nonground.pcd", nonground)
        ref_ground = _write_pcd(tmp_path / "ref_ground.pcd", ground)
        ref_nonground = _write_pcd(tmp_path / "ref_nonground.pcd", nonground)

        result = evaluate_ground_segmentation(
            est_ground, est_nonground, ref_ground, ref_nonground, voxel_size=0.5
        )

        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)
        assert result["iou"] == pytest.approx(1.0)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["confusion_matrix"]["fp"] == 0
        assert result["confusion_matrix"]["fn"] == 0

    def test_all_misclassified(self, tmp_path):
        ground = [[0, 0, 0], [1, 0, 0]]
        nonground = [[0, 0, 2], [1, 0, 2]]
        # Swap ground and nonground in estimate
        est_ground = _write_pcd(tmp_path / "est_ground.pcd", nonground)
        est_nonground = _write_pcd(tmp_path / "est_nonground.pcd", ground)
        ref_ground = _write_pcd(tmp_path / "ref_ground.pcd", ground)
        ref_nonground = _write_pcd(tmp_path / "ref_nonground.pcd", nonground)

        result = evaluate_ground_segmentation(
            est_ground, est_nonground, ref_ground, ref_nonground, voxel_size=0.5
        )

        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)
        assert result["iou"] == pytest.approx(0.0)

    def test_partial_overlap(self, tmp_path):
        # 2 ground voxels in reference; estimate catches 1 of them + 1 false positive
        ref_ground_pts = [[0, 0, 0], [1, 0, 0]]
        ref_nonground_pts = [[0, 0, 2], [1, 0, 2]]
        est_ground_pts = [[0, 0, 0], [0, 0, 2]]  # 1 TP + 1 FP
        est_nonground_pts = [[1, 0, 0], [1, 0, 2]]  # 1 FN + 1 TN

        est_ground = _write_pcd(tmp_path / "est_g.pcd", est_ground_pts)
        est_nonground = _write_pcd(tmp_path / "est_ng.pcd", est_nonground_pts)
        ref_ground = _write_pcd(tmp_path / "ref_g.pcd", ref_ground_pts)
        ref_nonground = _write_pcd(tmp_path / "ref_ng.pcd", ref_nonground_pts)

        result = evaluate_ground_segmentation(
            est_ground, est_nonground, ref_ground, ref_nonground, voxel_size=0.5
        )

        assert result["confusion_matrix"]["tp"] == 1
        assert result["confusion_matrix"]["fp"] == 1
        assert result["confusion_matrix"]["fn"] == 1
        assert result["confusion_matrix"]["tn"] == 1
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)

    def test_quality_gate_fails(self, tmp_path):
        ground = [[0, 0, 0], [1, 0, 0]]
        nonground = [[0, 0, 2], [1, 0, 2]]
        est_ground = _write_pcd(tmp_path / "est_g.pcd", ground)
        est_nonground = _write_pcd(tmp_path / "est_ng.pcd", nonground)
        ref_ground = _write_pcd(tmp_path / "ref_g.pcd", ground)
        ref_nonground = _write_pcd(tmp_path / "ref_ng.pcd", nonground)

        result = evaluate_ground_segmentation(
            est_ground, est_nonground, ref_ground, ref_nonground,
            voxel_size=0.5, min_iou=1.1,  # impossible threshold
        )

        assert result["quality_gate"] is not None
        assert result["quality_gate"]["passed"] is False
        assert any("IoU" in r for r in result["quality_gate"]["reasons"])

    def test_quality_gate_passes(self, tmp_path):
        ground = [[0, 0, 0], [1, 0, 0]]
        nonground = [[0, 0, 2], [1, 0, 2]]
        est_ground = _write_pcd(tmp_path / "est_g.pcd", ground)
        est_nonground = _write_pcd(tmp_path / "est_ng.pcd", nonground)
        ref_ground = _write_pcd(tmp_path / "ref_g.pcd", ground)
        ref_nonground = _write_pcd(tmp_path / "ref_ng.pcd", nonground)

        result = evaluate_ground_segmentation(
            est_ground, est_nonground, ref_ground, ref_nonground,
            voxel_size=0.5, min_precision=0.9, min_recall=0.9,
        )

        assert result["quality_gate"]["passed"] is True

    def test_no_gate_returns_none(self, tmp_path):
        ground = [[0, 0, 0]]
        nonground = [[0, 0, 2]]
        est_ground = _write_pcd(tmp_path / "est_g.pcd", ground)
        est_nonground = _write_pcd(tmp_path / "est_ng.pcd", nonground)
        ref_ground = _write_pcd(tmp_path / "ref_g.pcd", ground)
        ref_nonground = _write_pcd(tmp_path / "ref_ng.pcd", nonground)

        result = evaluate_ground_segmentation(
            est_ground, est_nonground, ref_ground, ref_nonground, voxel_size=0.5,
        )

        assert result["quality_gate"] is None

    def test_voxel_size_validation(self, tmp_path):
        ground = [[0, 0, 0]]
        nonground = [[0, 0, 2]]
        p = _write_pcd(tmp_path / "a.pcd", ground)
        q = _write_pcd(tmp_path / "b.pcd", nonground)

        with pytest.raises(ValueError, match="voxel_size"):
            evaluate_ground_segmentation(p, q, p, q, voxel_size=0)

    def test_result_contains_paths_and_counts(self, tmp_path):
        ground = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        nonground = [[0, 0, 2]]
        est_ground = _write_pcd(tmp_path / "est_g.pcd", ground)
        est_nonground = _write_pcd(tmp_path / "est_ng.pcd", nonground)
        ref_ground = _write_pcd(tmp_path / "ref_g.pcd", ground)
        ref_nonground = _write_pcd(tmp_path / "ref_ng.pcd", nonground)

        result = evaluate_ground_segmentation(
            est_ground, est_nonground, ref_ground, ref_nonground, voxel_size=0.5,
        )

        assert result["estimated_ground_path"] == est_ground
        assert result["counts"]["estimated_ground_points"] == 3
        assert result["counts"]["reference_nonground_points"] == 1
        assert result["voxel_size"] == 0.5
