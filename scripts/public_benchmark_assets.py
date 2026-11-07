"""Shared public benchmark/demo asset utilities."""

from __future__ import annotations

import math
import hashlib
import http.client
import shutil
import tarfile
import tempfile
import time
import zipfile
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import open3d as o3d

BUNNY_ARCHIVE_URL = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
BUNNY_SOURCE_PAGE = "https://graphics.stanford.edu/data/3Dscanrep/"
BUNNY_MESH_MEMBER = "bunny/reconstruction/bun_zipper.ply"
MAX_BUNNY_MESH_BYTES = 64 * 1024**2
MAX_BUNNY_ARCHIVE_ENTRIES = 10_000
MAX_LOCAL_BUNNY_ARCHIVE_BYTES = 128 * 1024**2
MAX_BUNNY_DOWNLOAD_BYTES = 128 * 1024**2
HDL_LOCALIZATION_MAP_URL = (
    "https://raw.githubusercontent.com/koide3/hdl_localization/master/data/map.pcd"
)
HDL_LOCALIZATION_REPO_URL = "https://github.com/koide3/hdl_localization"
HDL_LOCALIZATION_MAP_PAGE = (
    "https://github.com/koide3/hdl_localization/blob/master/data/map.pcd"
)
RELLIS_LIDAR_EXAMPLE_URL = (
    "https://drive.google.com/uc?export=download&id=1QikPnpmxneyCuwefr6m50fBOSB2ny4LC"
)
RELLIS_REPO_URL = "https://github.com/unmannedlab/RELLIS-3D"
RELLIS_README_URL = "https://github.com/unmannedlab/RELLIS-3D/blob/main/README.md"
RELLIS_LABEL_CONFIG_URL = (
    "https://github.com/unmannedlab/RELLIS-3D/blob/main/benchmarks/SalsaNext/"
    "train/tasks/semantic/config/labels/rellis.yaml"
)
RELLIS_EXAMPLE_ARCHIVE_NAME = "rellis_lidar_example.zip"


def _download_bunny_archive(path: Path) -> None:
    """Retry transient transport failures without reusing partial archives."""
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(BUNNY_ARCHIVE_URL, timeout=30) as response:
                length = response.headers.get('Content-Length')
                expected = int(length) if length is not None else None
                if expected is not None and expected < 0:
                    raise ValueError('Stanford Bunny response has a negative Content-Length')
                if expected is not None and expected > MAX_BUNNY_DOWNLOAD_BYTES:
                    raise ValueError('Stanford Bunny download exceeds 128 MiB')
                with path.open('wb') as output:
                    received = 0
                    while True:
                        block = response.read(min(1024**2, MAX_BUNNY_DOWNLOAD_BYTES - received + 1))
                        if not block:
                            break
                        received += len(block)
                        if received > MAX_BUNNY_DOWNLOAD_BYTES:
                            raise ValueError('Stanford Bunny download exceeds 128 MiB')
                        output.write(block)
                    if expected is not None and received < expected:
                        raise http.client.IncompleteRead(b'', expected - received)
            return
        except ValueError:
            path.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as error:
            path.unlink(missing_ok=True)
            if isinstance(error, urllib.error.HTTPError):
                retryable = error.code in (408, 429) or 500 <= error.code < 600
                error.close()
                if not retryable:
                    raise
            if attempt == 3:
                raise RuntimeError(f'Failed to download Stanford Bunny after 3 attempts: {BUNNY_ARCHIVE_URL}') from error
            time.sleep(attempt)


def _extract_bunny_mesh(archive_path: Path, mesh_path: Path) -> None:
    """Copy only the expected regular mesh, never archive-controlled paths."""
    with tarfile.open(archive_path, 'r:gz') as archive:
        selected = None
        for count, member in enumerate(archive, start=1):
            if count > MAX_BUNNY_ARCHIVE_ENTRIES:
                raise RuntimeError('Stanford Bunny archive has too many entries')
            if member.name != BUNNY_MESH_MEMBER:
                continue
            if selected is not None:
                raise RuntimeError('Stanford Bunny archive contains duplicate mesh entries')
            if not member.isfile() or not 0 < member.size <= MAX_BUNNY_MESH_BYTES:
                raise RuntimeError('Stanford Bunny mesh must be a regular file of at most 64 MiB')
            selected = member
        if selected is None:
            raise RuntimeError('Stanford Bunny archive is missing the expected mesh')
        source = archive.extractfile(selected)
        if source is None:
            raise RuntimeError('Cannot read Stanford Bunny mesh')
        with source, mesh_path.open('xb') as output:
            shutil.copyfileobj(source, output)


def download_bunny_mesh() -> o3d.geometry.TriangleMesh:
    """Download and return the Stanford Bunny mesh."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        archive_path = root / "bunny.tar.gz"
        _download_bunny_archive(archive_path)
        mesh_path = root / "bun_zipper.ply"
        _extract_bunny_mesh(archive_path, mesh_path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(mesh.vertices) == 0:
            raise RuntimeError("failed to load Stanford Bunny mesh")
        return mesh


def load_local_bunny_mesh(path: Path, *, expected_sha256: str | None = None) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Load a bounded local archive snapshot and identify exactly those bytes."""
    if expected_sha256 is not None:
        if len(expected_sha256) != 64 or any(c not in '0123456789abcdefABCDEF' for c in expected_sha256):
            raise ValueError('Expected Bunny SHA-256 must contain exactly 64 hexadecimal characters')
        expected_sha256 = expected_sha256.lower()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_LOCAL_BUNNY_ARCHIVE_BYTES:
        raise ValueError('Local Bunny archive must be a regular file of at most 128 MiB')
    with path.open('rb') as source:
        raw = source.read(MAX_LOCAL_BUNNY_ARCHIVE_BYTES + 1)
    if len(raw) > MAX_LOCAL_BUNNY_ARCHIVE_BYTES:
        raise ValueError('Local Bunny archive exceeds 128 MiB')
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f'Bunny archive SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}')
    identity = {'sha256': actual_sha256, 'bytes': len(raw),
                'origin': 'local', 'authenticity_verified': False}
    if expected_sha256 is not None:
        identity['expected_sha256'] = expected_sha256
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        archive = root/'bunny.tar.gz'
        archive.write_bytes(raw)
        mesh_path = root/'bun_zipper.ply'
        _extract_bunny_mesh(archive, mesh_path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(mesh.vertices) == 0:
            raise RuntimeError('failed to load local Bunny mesh')
        return mesh, identity


def download_hdl_localization_map(download_dir: Path) -> Path:
    """Download the public hdl_localization sample map and return its path."""
    download_dir.mkdir(parents=True, exist_ok=True)
    map_path = download_dir / "hdl_localization_map.pcd"
    if not map_path.exists():
        urllib.request.urlretrieve(HDL_LOCALIZATION_MAP_URL, map_path)
    return map_path


def download_rellis_lidar_example(download_dir: Path) -> Path:
    """Download and extract the public RELLIS-3D LiDAR example bundle."""
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / RELLIS_EXAMPLE_ARCHIVE_NAME
    extract_root = download_dir / "rellis_lidar_example"
    example_root = extract_root / "Rellis_3D_lidar_example"

    if example_root.exists():
        return example_root

    if not archive_path.exists():
        urllib.request.urlretrieve(RELLIS_LIDAR_EXAMPLE_URL, archive_path)

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_root)

    if not example_root.exists():
        raise RuntimeError("failed to extract RELLIS-3D LiDAR example")
    return example_root


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
