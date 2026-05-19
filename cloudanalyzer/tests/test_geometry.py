"""Tests for ca.geometry (cross-representation geometry QA)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from typer.testing import CliRunner

from ca.geometry import (
    REPRESENTATIONS,
    detect_representation,
    evaluate_geometry,
    load_representation,
)
from cloudanalyzer_cli.main import app


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SUITE_DIR = REPO_ROOT / "benchmarks" / "3dgs" / "synthetic-room"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_synthetic_3dgs_demo import build as build_3dgs_demo  # noqa: E402


def _logit(alpha: float) -> float:
    alpha = float(np.clip(alpha, 1e-4, 1.0 - 1e-4))
    return math.log(alpha / (1.0 - alpha))


def _write_ascii_gaussian_ply(
    points: np.ndarray, opacity_logits: np.ndarray, path: Path
) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float opacity",
        "end_header",
    ]
    rows = [
        f"{x:.6f} {y:.6f} {z:.6f} {op:.6f}"
        for (x, y, z), op in zip(points, opacity_logits)
    ]
    path.write_text("\n".join(header + rows) + "\n", encoding="ascii")


def _write_binary_le_gaussian_ply(
    points: np.ndarray, opacity_logits: np.ndarray, path: Path
) -> None:
    import struct

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float opacity\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        for (x, y, z), op in zip(points, opacity_logits):
            f.write(struct.pack("<ffff", float(x), float(y), float(z), float(op)))


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Regenerate the 3DGS demo in a temp dir so tests don't depend on the
    checked-in copy being up to date."""
    out = tmp_path_factory.mktemp("synthetic-room-3dgs")
    build_3dgs_demo(out)
    return out


# ---------------------------------------------------------------- detection


def test_detection_gaussian_ply(demo_dir: Path) -> None:
    assert detect_representation(str(demo_dir / "gaussians.ply")) == "gaussian-points"


def test_detection_point_cloud(demo_dir: Path) -> None:
    assert detect_representation(str(demo_dir / "reference.pcd")) == "point-cloud"


def test_detection_plain_ply_without_opacity(tmp_path: Path) -> None:
    """A vanilla PLY (just xyz) must not be misclassified as gaussian-points."""
    out = tmp_path / "plain.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((4, 3)))
    o3d.io.write_point_cloud(str(out), pcd, write_ascii=True)
    assert detect_representation(str(out)) == "point-cloud"


# ---------------------------------------------------------------- adapter


def test_gaussian_points_opacity_filter(tmp_path: Path) -> None:
    points = np.arange(30, dtype=np.float64).reshape(10, 3)
    alphas = np.array([0.9] * 5 + [0.05] * 5)
    logits = np.array([_logit(a) for a in alphas])
    ply = tmp_path / "g.ply"
    _write_ascii_gaussian_ply(points, logits, ply)

    no_filter = load_representation(str(ply))
    assert no_filter.final_count == 10
    assert no_filter.applied_filters == []

    filtered = load_representation(str(ply), opacity_threshold=0.5)
    assert filtered.final_count == 5
    assert any("opacity>=" in f for f in filtered.applied_filters)


def test_gaussian_points_binary_format(tmp_path: Path) -> None:
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    logits = np.array([_logit(0.9), _logit(0.1), _logit(0.6)])
    ply = tmp_path / "g.ply"
    _write_binary_le_gaussian_ply(points, logits, ply)

    loaded = load_representation(str(ply), opacity_threshold=0.5)
    # Splat #1 (alpha=0.1) is filtered out; #0 (0.9) and #2 (0.6) remain.
    assert loaded.final_count == 2


def test_point_cloud_passthrough(demo_dir: Path) -> None:
    loaded = load_representation(str(demo_dir / "reference.pcd"))
    assert loaded.representation_detected == "point-cloud"
    assert loaded.final_count > 0
    assert loaded.applied_filters == []


def test_point_cloud_opacity_threshold_is_a_noop(demo_dir: Path) -> None:
    loaded = load_representation(
        str(demo_dir / "reference.pcd"), opacity_threshold=0.5
    )
    assert loaded.final_count > 0
    assert any("ignored" in f for f in loaded.applied_filters)


def test_explicit_gaussian_points_on_non_gaussian_ply_raises(tmp_path: Path) -> None:
    out = tmp_path / "plain.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((3, 3)))
    o3d.io.write_point_cloud(str(out), pcd, write_ascii=True)
    with pytest.raises(ValueError, match="opacity_threshold|missing"):
        load_representation(
            str(out),
            representation="gaussian-points",
            opacity_threshold=0.5,
        )


def test_voxel_downsample(tmp_path: Path) -> None:
    points = np.array(
        [[0, 0, 0], [0.001, 0, 0], [0.002, 0, 0], [1, 1, 1], [1.001, 1, 1]],
        dtype=np.float64,
    )
    logits = np.array([_logit(0.9)] * 5)
    ply = tmp_path / "g.ply"
    _write_ascii_gaussian_ply(points, logits, ply)
    loaded = load_representation(str(ply), voxel_size=0.5)
    # Two voxels around 0 and (1,1,1).
    assert loaded.final_count == 2
    assert any("voxel=" in f for f in loaded.applied_filters)


def test_unknown_representation_raises(demo_dir: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported representation"):
        load_representation(
            str(demo_dir / "reference.pcd"), representation="splatfacto"
        )


# ---------------------------------------------------------------- evaluate_geometry


def test_evaluate_geometry_filter_improves_score(demo_dir: Path) -> None:
    unfiltered = evaluate_geometry(
        str(demo_dir / "gaussians.ply"),
        str(demo_dir / "reference.pcd"),
    )
    filtered = evaluate_geometry(
        str(demo_dir / "gaussians.ply"),
        str(demo_dir / "reference.pcd"),
        opacity_threshold=0.5,
    )
    assert filtered["chamfer_distance"] < unfiltered["chamfer_distance"]
    # representation block records what the adapter did.
    assert filtered["representation"]["detected"] == "gaussian-points"
    assert filtered["representation"]["opacity_threshold"] == 0.5
    assert filtered["source_path"] == str(demo_dir / "gaussians.ply")


def test_evaluate_geometry_dense_close_to_reference(demo_dir: Path) -> None:
    """The dense variant (only high-alpha splats, no noise) should score
    close to the reference on the source-side direction."""
    result = evaluate_geometry(
        str(demo_dir / "gaussians_dense.ply"),
        str(demo_dir / "reference.pcd"),
    )
    # Source->target mean distance is the noise floor; well below the
    # noisy gaussians.ply unfiltered case.
    assert result["distance_stats"]["source_to_target"]["mean"] < 0.05


def test_evaluate_geometry_empty_after_filter_raises(tmp_path: Path) -> None:
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    logits = np.array([_logit(0.05), _logit(0.05)])
    ply = tmp_path / "all_low.ply"
    _write_ascii_gaussian_ply(points, logits, ply)
    ref = tmp_path / "ref.pcd"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((4, 3)))
    o3d.io.write_point_cloud(str(ref), pcd, write_ascii=False)
    with pytest.raises(ValueError, match="no points left"):
        evaluate_geometry(
            str(ply),
            str(ref),
            opacity_threshold=0.5,
        )


# ---------------------------------------------------------------- CLI


def test_cli_geometry_evaluate(demo_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "geometry-evaluate",
            str(demo_dir / "gaussians.ply"),
            str(demo_dir / "reference.pcd"),
            "--opacity-threshold",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "representation=gaussian-points" in result.output
    assert "Filters:" in result.output


def test_cli_geometry_evaluate_format_json(demo_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "geometry-evaluate",
            str(demo_dir / "reference.pcd"),
            str(demo_dir / "reference.pcd"),
            "--format-json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["representation"]["detected"] == "point-cloud"
    # Identical inputs -> chamfer ~= 0
    assert payload["chamfer_distance"] < 1e-6


def test_cli_rejects_unknown_representation(demo_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "geometry-evaluate",
            str(demo_dir / "gaussians.ply"),
            str(demo_dir / "reference.pcd"),
            "--representation",
            "ellipsoid",
        ],
    )
    assert result.exit_code == 1
    assert "--representation" in result.output or "--representation" in result.stderr


def test_representations_constant_contains_known() -> None:
    assert "auto" in REPRESENTATIONS
    assert "point-cloud" in REPRESENTATIONS
    assert "gaussian-points" in REPRESENTATIONS
