"""Shared public benchmark/demo asset utilities."""

from __future__ import annotations

import math
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import open3d as o3d

BUNNY_ARCHIVE_URL = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
BUNNY_SOURCE_PAGE = "https://graphics.stanford.edu/data/3Dscanrep/"


def download_bunny_mesh() -> o3d.geometry.TriangleMesh:
    """Download and return the Stanford Bunny mesh."""
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


def rotation_matrix(z_degrees: float, x_degrees: float) -> np.ndarray:
    """Create a simple Z then X rotation matrix."""
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


def make_trajectory_rows(
    center: np.ndarray,
    extent: np.ndarray,
    *,
    phase: float,
    radial_wobble: float,
    vertical_wobble: float,
    sample_count: int = 72,
) -> list[tuple[float, float, float, float]]:
    """Generate a smooth orbit-like trajectory around a scene."""
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


def write_csv_trajectory(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    """Write CSV trajectory rows with a standard header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["timestamp,x,y,z"]
    lines.extend(f"{timestamp:.3f},{x:.6f},{y:.6f},{z:.6f}" for timestamp, x, y, z in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
