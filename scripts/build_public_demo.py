#!/usr/bin/env python3
"""Build a public static demo bundle for GitHub Pages."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ca.web import export_static_bundle

BUNNY_ARCHIVE_URL = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
BUNNY_SOURCE_PAGE = "https://graphics.stanford.edu/data/3Dscanrep/"


def _download_bunny_mesh() -> o3d.geometry.TriangleMesh:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        archive_path = root / "bunny.tar.gz"
        urllib.request.urlretrieve(BUNNY_ARCHIVE_URL, archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(root)
        mesh_path = root / "bunny" / "reconstruction" / "bun_zipper.ply"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(mesh.vertices) == 0:
            raise RuntimeError("failed to load Stanford Bunny mesh")
        return mesh


def _rotation_matrix(z_degrees: float, x_degrees: float) -> np.ndarray:
    z_radians = math.radians(z_degrees)
    x_radians = math.radians(x_degrees)
    cos_z = math.cos(z_radians)
    sin_z = math.sin(z_radians)
    cos_x = math.cos(x_radians)
    sin_x = math.sin(x_radians)
    rotation_z = np.array(
        [
            [cos_z, -sin_z, 0.0],
            [sin_z, cos_z, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_x, -sin_x],
            [0.0, sin_x, cos_x],
        ],
        dtype=float,
    )
    return np.asarray(rotation_z @ rotation_x, dtype=float)


def _make_trajectory_rows(
    center: np.ndarray,
    extent: np.ndarray,
    *,
    phase: float,
    radial_wobble: float,
    vertical_wobble: float,
    sample_count: int = 72,
) -> list[tuple[float, float, float, float]]:
    radius = float(max(extent[0], extent[1]) * 1.9)
    height = float(center[2] + extent[2] * 1.8)
    rows: list[tuple[float, float, float, float]] = []
    for index in range(sample_count):
        angle = ((2.0 * math.pi) * index / sample_count) + phase
        radius_scale = 1.0 + (radial_wobble * math.sin(angle * 3.0))
        x = center[0] + (radius * radius_scale * math.cos(angle))
        y = center[1] + (radius * radius_scale * math.sin(angle))
        z = height + (extent[2] * vertical_wobble * math.sin(angle * 2.0))
        rows.append((index * 0.1, float(x), float(y), float(z)))
    return rows


def _write_csv_trajectory(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    lines = ["timestamp,x,y,z"]
    lines.extend(f"{timestamp:.3f},{x:.6f},{y:.6f},{z:.6f}" for timestamp, x, y, z in rows)
    path.write_text("\n".join(lines) + "\n")


def build_public_demo(output_dir: Path, max_points: int) -> dict:
    mesh = _download_bunny_mesh()
    vertices = np.asarray(mesh.vertices, dtype=float)
    center = np.mean(vertices, axis=0)
    extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)

    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(vertices)

    centered = vertices - center[None, :]
    rotation = _rotation_matrix(z_degrees=4.5, x_degrees=-2.0)
    reference_points = centered @ rotation.T
    reference_points[:, 0] *= 1.008
    reference_points[:, 1] *= 0.992
    reference_points += center[None, :] + np.array(
        [extent[0] * 0.035, -extent[1] * 0.018, extent[2] * 0.02],
        dtype=float,
    )
    reference_cloud = o3d.geometry.PointCloud()
    reference_cloud.points = o3d.utility.Vector3dVector(reference_points)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        source_path = tmp_root / "stanford_bunny_source.pcd"
        reference_path = tmp_root / "stanford_bunny_reference.pcd"
        estimated_traj_path = tmp_root / "stanford_bunny_traj.csv"
        reference_traj_path = tmp_root / "stanford_bunny_traj_ref.csv"

        o3d.io.write_point_cloud(str(source_path), source_cloud)
        o3d.io.write_point_cloud(str(reference_path), reference_cloud)
        _write_csv_trajectory(
            estimated_traj_path,
            _make_trajectory_rows(
                center,
                extent,
                phase=0.12,
                radial_wobble=0.035,
                vertical_wobble=0.08,
            ),
        )
        _write_csv_trajectory(
            reference_traj_path,
            _make_trajectory_rows(
                center,
                extent,
                phase=0.0,
                radial_wobble=0.0,
                vertical_wobble=0.05,
            ),
        )

        result = export_static_bundle(
            [str(source_path), str(reference_path)],
            output_dir=str(output_dir),
            max_points=max_points,
            heatmap=True,
            trajectory_path=str(estimated_traj_path),
            trajectory_reference_path=str(reference_traj_path),
            trajectory_max_time_delta=0.2,
        )

    attribution = "\n".join(
        [
            "# Stanford Bunny Demo Attribution",
            "",
            "Source mesh: The Stanford Bunny from the Stanford 3D Scanning Repository.",
            f"Source page: {BUNNY_SOURCE_PAGE}",
            f"Archive: {BUNNY_ARCHIVE_URL}",
            "",
            "This demo derives its source point cloud from bun_zipper.ply.",
            "The reference point cloud and both trajectories are generated by CloudAnalyzer",
            "to demonstrate heatmap inspection, progressive loading, and trajectory overlays.",
            "",
            "The Stanford repository states that the models may be used for research,",
            "mirrored or redistributed for free with credit to the Stanford Computer Graphics",
            "Laboratory, and are not for commercial use without permission.",
        ]
    )
    (output_dir / "ATTRIBUTION.md").write_text(attribution + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public Stanford Bunny demo bundle.")
    parser.add_argument(
        "--output",
        default="docs/demo/stanford-bunny",
        help="Output directory for the static viewer bundle",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50_000,
        help="Viewer point budget passed to ca web-export",
    )
    args = parser.parse_args()

    result = build_public_demo(Path(args.output), max_points=args.max_points)
    print(f"Exported:     {result['output_dir']}")
    print(f"Viewer mode:  {result['viewer_mode']}")
    print(f"Data:         {result['data_json']}")
    print(f"Chunks:       {result['chunk_count']}")
    print(f"Display pts:  {result['display_points']}")


if __name__ == "__main__":
    main()
