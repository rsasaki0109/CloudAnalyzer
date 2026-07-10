"""Tests for experimental reference-free plane-consistency proxies."""

import numpy as np
import pytest
from pathlib import Path

from ca.core.plane_consistency import evaluate_plane_consistency_points
from ca.core.checks import load_check_suite
from ca.pr_comment import build_pr_comment


def _parallel_planes(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clouds = []
    for x0 in (0.0, 1.0):
        yz = rng.uniform(0.05, 2.95, size=(900, 2))
        x = np.full((900, 1), x0 + 0.25) + rng.normal(0, 0.002, size=(900, 1))
        clouds.append(np.column_stack((x, yz)))
    return np.vstack(clouds)


def test_plane_proxy_is_deterministic_and_finite_for_planar_map() -> None:
    points = _parallel_planes()
    first = evaluate_plane_consistency_points(points, voxel_size=1.0)
    second = evaluate_plane_consistency_points(points, voxel_size=1.0)
    assert first["plane_normal_dispersion"] == pytest.approx(second["plane_normal_dispersion"])
    assert first["coplanar_offset_rmse"] == pytest.approx(second["coplanar_offset_rmse"])
    assert first["num_plane_patches"] >= 2
    assert first["experimental_proxy"] is True


def test_normal_metric_is_sign_invariant() -> None:
    points = _parallel_planes()
    mirrored = points.copy()
    mirrored[:, 0] *= -1
    a = evaluate_plane_consistency_points(points, voxel_size=1.0)
    b = evaluate_plane_consistency_points(mirrored, voxel_size=1.0)
    assert a["plane_normal_dispersion"] == pytest.approx(b["plane_normal_dispersion"], abs=1e-3)


def test_clean_orthogonal_planes_are_not_treated_as_normal_dispersion() -> None:
    points = _parallel_planes()
    rotated = points[:, [1, 0, 2]]
    result = evaluate_plane_consistency_points(np.vstack((points, rotated)), voxel_size=1.0)
    assert result["plane_normal_dispersion"] < 0.02


def test_sparse_map_reports_unavailable_metrics() -> None:
    result = evaluate_plane_consistency_points(np.zeros((2, 3)), voxel_size=1.0)
    assert np.isnan(result["plane_normal_dispersion"])
    assert result["num_plane_patches"] == 0


def test_single_plane_patch_is_insufficient_support() -> None:
    rng = np.random.default_rng(5)
    yz = rng.uniform(0.1, 0.9, size=(100, 2))
    points = np.column_stack((np.full(100, 0.25), yz))
    result = evaluate_plane_consistency_points(points, voxel_size=1.0)
    assert result["num_plane_patches"] == 1
    assert np.isnan(result["plane_normal_dispersion"])
    assert np.isnan(result["coplanar_offset_rmse"])


def test_coplanar_offset_is_translation_and_old_grid_boundary_invariant() -> None:
    points = _parallel_planes()
    baseline = evaluate_plane_consistency_points(points, voxel_size=1.0)
    # The non-integral shift crosses the old world-grid quantization boundary.
    translated = evaluate_plane_consistency_points(
        points + np.array([10.137, -7.231, 3.419]), voxel_size=1.0
    )
    assert translated["coplanar_offset_rmse"] == pytest.approx(
        baseline["coplanar_offset_rmse"], abs=1e-8
    )


def test_global_axis_permutation_preserves_metrics() -> None:
    points = _parallel_planes()
    baseline = evaluate_plane_consistency_points(points, voxel_size=1.0)
    rotated = evaluate_plane_consistency_points(points[:, [1, 2, 0]], voxel_size=1.0)
    assert rotated["plane_normal_dispersion"] == pytest.approx(
        baseline["plane_normal_dispersion"], abs=1e-8
    )
    assert rotated["coplanar_offset_rmse"] == pytest.approx(
        baseline["coplanar_offset_rmse"], abs=1e-8
    )


def test_arbitrary_global_rigid_rotation_preserves_metrics() -> None:
    points = _parallel_planes()
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.713
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross
    baseline = evaluate_plane_consistency_points(points, voxel_size=1.0)
    transformed = evaluate_plane_consistency_points(points @ rotation.T + np.array([2.3, -4.1, 0.7]), voxel_size=1.0)
    assert transformed["plane_normal_dispersion"] == pytest.approx(baseline["plane_normal_dispersion"], abs=1e-8)
    assert transformed["coplanar_offset_rmse"] == pytest.approx(baseline["coplanar_offset_rmse"], abs=1e-8)


def test_linear_voxel_is_rejected_as_non_planar() -> None:
    x = np.linspace(0.05, 0.95, 100)
    line = np.column_stack((x, np.full(100, 0.2), np.full(100, 0.3)))
    result = evaluate_plane_consistency_points(line, voxel_size=1.0)
    assert result["num_plane_patches"] == 0


def test_structure_metrics_surface_in_pr_comment() -> None:
    check = {
        "id": "structure", "kind": "structure", "passed": False,
        "summary": {"plane_normal_dispersion": 0.2, "coplanar_offset_rmse": 0.03, "num_plane_patches": 8},
        "result": {"quality_gate": {"passed": False, "reasons": ["dispersion too high"]}},
    }
    payload = {"summary": {"passed": False, "total_checks": 1, "passed_checks": 0, "failed_checks": 1}, "checks": [check]}
    comment = build_pr_comment(payload)
    assert "Plane normal dispersion" in comment
    assert "Coplanar offset RMSE=0.0300 m" in comment


def test_structure_check_config_accepts_reference_free_gates(tmp_path: Path) -> None:
    config = tmp_path / "cloudanalyzer.yaml"
    config.write_text(
        """
version: 1
checks:
  - id: structure
    kind: structure
    source: map.pcd
    gate:
      voxel_size: 1.5
      max_plane_normal_dispersion: 0.2
      max_coplanar_offset_rmse: 0.1
""",
        encoding="utf-8",
    )
    check = load_check_suite(str(config)).checks[0]
    assert check.kind == "structure"
    assert "reference" not in check.inputs
    assert check.gate["max_plane_normal_dispersion"] == pytest.approx(0.2)
