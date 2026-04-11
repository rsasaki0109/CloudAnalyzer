"""Tests for oriented 3D box IoU computation."""

import math

import numpy as np
import pytest

from ca.object_eval import Box3D, box_iou_3d


def _make_box(
    center: list[float],
    size: list[float],
    yaw: float = 0.0,
    label: str = "car",
) -> Box3D:
    return Box3D(
        frame_id="0001",
        label=label,
        center=np.array(center, dtype=np.float64),
        size=np.array(size, dtype=np.float64),
        yaw=yaw,
        score=1.0,
        track_id=None,
        index=0,
    )


class TestAABBIoU:
    """Axis-aligned IoU (yaw=0) — regression tests for existing behavior."""

    def test_identical_boxes(self):
        a = _make_box([0, 0, 0], [2, 2, 2])
        assert box_iou_3d(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = _make_box([0, 0, 0], [1, 1, 1])
        b = _make_box([10, 10, 10], [1, 1, 1])
        assert box_iou_3d(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = _make_box([0, 0, 0], [2, 2, 2])
        b = _make_box([1, 0, 0], [2, 2, 2])
        # Overlap: 1x2x2 = 4, union: 8+8-4 = 12
        assert box_iou_3d(a, b) == pytest.approx(4.0 / 12.0)


class TestOrientedIoU:
    """Oriented IoU with non-zero yaw."""

    def test_identical_box_rotated(self):
        """Same box at yaw=pi/4 should have IoU=1.0 with itself."""
        a = _make_box([0, 0, 0], [2, 2, 2], yaw=math.pi / 4)
        assert box_iou_3d(a, a) == pytest.approx(1.0)

    def test_square_box_45_degrees(self):
        """2x2x2 box vs same box rotated 45 degrees.

        BEV: a 2x2 square vs the same square rotated 45 degrees.
        The intersection is a regular octagon with area = 2*(sqrt(2)-1)*s^2
        where s=2 (side length). Area = 2*(sqrt(2)-1)*4 = 8*(sqrt(2)-1) ≈ 3.314.
        Union = 4 + 4 - 3.314 = 4.686.
        IoU_2d = 3.314 / 4.686 ≈ 0.7071.
        Height overlap is full (2), so IoU_3d = IoU_2d.
        """
        a = _make_box([0, 0, 0], [2, 2, 2], yaw=0.0)
        b = _make_box([0, 0, 0], [2, 2, 2], yaw=math.pi / 4)
        intersection_area = 8.0 * (math.sqrt(2) - 1)
        union_area = 4.0 + 4.0 - intersection_area
        expected_iou = intersection_area / union_area
        assert box_iou_3d(a, b) == pytest.approx(expected_iou, abs=1e-4)

    def test_rectangular_box_90_degrees(self):
        """4x2x2 box vs same box rotated 90 degrees around Z.

        BEV: 4x2 rectangle vs 2x4 rectangle, both centered at origin.
        Intersection: 2x2 = 4.
        Union: 8 + 8 - 4 = 12.
        IoU_2d = 4/12 = 1/3.
        """
        a = _make_box([0, 0, 0], [4, 2, 2], yaw=0.0)
        b = _make_box([0, 0, 0], [4, 2, 2], yaw=math.pi / 2)
        assert box_iou_3d(a, b) == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_no_overlap_rotated(self):
        """Widely separated boxes with yaw should have IoU=0."""
        a = _make_box([0, 0, 0], [2, 2, 2], yaw=0.3)
        b = _make_box([20, 20, 20], [2, 2, 2], yaw=0.7)
        assert box_iou_3d(a, b) == pytest.approx(0.0)

    def test_height_mismatch(self):
        """Same BEV overlap but partial height overlap."""
        a = _make_box([0, 0, 0], [2, 2, 2], yaw=0.0)
        b = _make_box([0, 0, 1], [2, 2, 2], yaw=math.pi / 6)
        iou = box_iou_3d(a, b)
        # Z overlap is 1.0 (not full 2.0), so IoU should be less than BEV-only
        assert 0.0 < iou < 1.0

    def test_one_yaw_zero_one_rotated(self):
        """When one box has yaw=0, oriented path should still work correctly."""
        a = _make_box([0, 0, 0], [2, 2, 2], yaw=0.0)
        b = _make_box([0, 0, 0], [2, 2, 2], yaw=0.001)
        # Very small rotation, should be very close to 1.0
        assert box_iou_3d(a, b) == pytest.approx(1.0, abs=1e-3)

    def test_oriented_matches_aabb_when_yaw_zero(self):
        """Forcing oriented path with yaw=0 should give same result as AABB."""
        a_aabb = _make_box([1, 2, 0], [3, 4, 2], yaw=0.0)
        b_aabb = _make_box([2, 1, 0.5], [2, 3, 1.5], yaw=0.0)
        # Use tiny yaw to force oriented path
        a_obb = _make_box([1, 2, 0], [3, 4, 2], yaw=1e-15)
        b_obb = _make_box([2, 1, 0.5], [2, 3, 1.5], yaw=0.0)
        iou_aabb = box_iou_3d(a_aabb, b_aabb)
        iou_obb = box_iou_3d(a_obb, b_obb)
        assert iou_obb == pytest.approx(iou_aabb, abs=1e-6)
