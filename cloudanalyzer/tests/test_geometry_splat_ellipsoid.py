"""Tests for 3DGS splat-aware ellipsoid sampling in ca geometry-evaluate.

Phase 19 extends the ``gaussian-points`` adapter so it can surface-sample
each splat's anisotropic ellipsoid using the standard 3DGS PLY properties
``scale_0..2`` (log-σ) and ``rot_0..3`` (wxyz quaternion). The default
remains ``splat_method='centers'`` to keep cross-representation regression
metrics backward-compatible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ca.geometry import (
    DEFAULT_SPLAT_SAMPLES,
    SPLAT_METHODS,
    _fibonacci_sphere,
    _quaternions_to_rotmats,
    _sample_splat_ellipsoids,
    load_representation,
)


# ---------------------------------------------------------- PLY synthesis util


def _write_full_3dgs_ply(
    path: Path,
    centers: np.ndarray,
    scales_log: np.ndarray,
    quats_wxyz: np.ndarray,
    opacity_logits: np.ndarray,
) -> None:
    """Write a binary little-endian PLY with full 3DGS property layout."""
    n = centers.shape[0]
    assert scales_log.shape == (n, 3)
    assert quats_wxyz.shape == (n, 4)
    assert opacity_logits.shape == (n,)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    ).encode("ascii")

    payload = np.empty((n, 11), dtype="<f4")
    payload[:, 0:3] = centers.astype("<f4")
    payload[:, 3] = opacity_logits.astype("<f4")
    payload[:, 4:7] = scales_log.astype("<f4")
    payload[:, 7:11] = quats_wxyz.astype("<f4")
    with path.open("wb") as f:
        f.write(header)
        f.write(payload.tobytes())


# ----------------------------------------------------------- helper unit tests


def test_fibonacci_sphere_on_unit_sphere() -> None:
    pts = _fibonacci_sphere(64)
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_fibonacci_sphere_rejects_too_few() -> None:
    with pytest.raises(ValueError, match=">=2"):
        _fibonacci_sphere(1)


def test_quaternions_identity_is_identity_matrix() -> None:
    q = np.array([[1.0, 0.0, 0.0, 0.0]])  # wxyz
    mat = _quaternions_to_rotmats(q)
    np.testing.assert_allclose(mat[0], np.eye(3), atol=1e-10)


def test_quaternions_90deg_z_rotation() -> None:
    # 90° rotation around +Z axis: w=cos(45°), z=sin(45°)
    angle = np.pi / 2
    q = np.array([[np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]])
    mat = _quaternions_to_rotmats(q)
    expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(mat[0], expected, atol=1e-7)


def test_quaternions_normalize_unnormalized_input() -> None:
    # Scaled-up identity quaternion still becomes identity matrix.
    q = np.array([[7.5, 0.0, 0.0, 0.0]])
    mat = _quaternions_to_rotmats(q)
    np.testing.assert_allclose(mat[0], np.eye(3), atol=1e-10)


# --------------------------------------------------- _sample_splat_ellipsoids


def test_sample_splat_ellipsoids_unit_sphere() -> None:
    """Identity rotation + log-scale 0 (σ=1) ⇒ unit-sphere samples."""
    centers = np.zeros((1, 3))
    scales_log = np.zeros((1, 3))  # exp(0) = 1
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    samples = _sample_splat_ellipsoids(centers, scales_log, quats, samples_per_splat=16)
    assert samples.shape == (16, 3)
    np.testing.assert_allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1e-6)


def test_sample_splat_ellipsoids_anisotropic_extent() -> None:
    """Axis-aligned anisotropic scale ⇒ bounding box matches exp(scale)."""
    centers = np.array([[5.0, -2.0, 1.0]])
    scales_log = np.log(np.array([[0.5, 2.0, 0.25]]))
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])  # identity
    samples = _sample_splat_ellipsoids(centers, scales_log, quats, samples_per_splat=128)
    extent = np.ptp(samples, axis=0)
    # With 128 Fibonacci samples each axis should very nearly span ±scale.
    np.testing.assert_allclose(extent, [1.0, 4.0, 0.5], atol=5e-2)
    np.testing.assert_allclose(samples.mean(axis=0), [5.0, -2.0, 1.0], atol=1e-1)


def test_sample_splat_ellipsoids_z_rotation() -> None:
    """A 90° Z-rotation swaps X and Y extents of an anisotropic ellipsoid."""
    centers = np.zeros((1, 3))
    scales_log = np.log(np.array([[0.5, 3.0, 1.0]]))
    angle = np.pi / 2
    quats = np.array([[np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]])
    samples = _sample_splat_ellipsoids(centers, scales_log, quats, samples_per_splat=128)
    extent = np.ptp(samples, axis=0)
    # X took on the original Y scale (3.0 → diameter 6.0)
    # Y took on the original X scale (0.5 → diameter 1.0)
    np.testing.assert_allclose(extent[0], 6.0, atol=5e-2)
    np.testing.assert_allclose(extent[1], 1.0, atol=5e-2)
    np.testing.assert_allclose(extent[2], 2.0, atol=5e-2)


def test_sample_splat_ellipsoids_multiple_splats_grouped() -> None:
    """Samples are emitted in N×K row-major order so callers can reshape."""
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    scales_log = np.log(np.full((2, 3), 0.5))
    quats = np.tile([[1.0, 0.0, 0.0, 0.0]], (2, 1))
    samples = _sample_splat_ellipsoids(centers, scales_log, quats, samples_per_splat=4)
    assert samples.shape == (8, 3)
    grouped = samples.reshape(2, 4, 3)
    np.testing.assert_allclose(grouped[0].mean(axis=0), [0.0, 0.0, 0.0], atol=0.5)
    np.testing.assert_allclose(grouped[1].mean(axis=0), [10.0, 0.0, 0.0], atol=0.5)


def test_sample_splat_ellipsoids_empty_input() -> None:
    out = _sample_splat_ellipsoids(
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty((0, 4)),
        samples_per_splat=8,
    )
    assert out.shape == (0, 3)


def test_sample_splat_ellipsoids_rejects_samples_per_splat_lt_2() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        _sample_splat_ellipsoids(
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.array([[1.0, 0.0, 0.0, 0.0]]),
            samples_per_splat=1,
        )


# --------------------------------------------------------------- PLY adapter


def test_load_representation_centers_method_unchanged(tmp_path: Path) -> None:
    """Default ``splat_method='centers'`` keeps the original single-point-per-splat behavior."""
    centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    scales_log = np.log(np.full((3, 3), 0.1))
    quats = np.tile([[1.0, 0.0, 0.0, 0.0]], (3, 1))
    opacities = np.zeros(3)
    path = tmp_path / "splats.ply"
    _write_full_3dgs_ply(path, centers, scales_log, quats, opacities)

    loaded = load_representation(str(path), representation="gaussian-points")
    assert loaded.final_count == 3
    np.testing.assert_allclose(loaded.points, centers, atol=1e-5)


def test_load_representation_ellipsoid_expands_per_splat(tmp_path: Path) -> None:
    """``splat_method='ellipsoid'`` yields ``N * splat_samples`` points."""
    n = 5
    rng = np.random.default_rng(42)
    centers = rng.uniform(-3, 3, size=(n, 3))
    scales_log = np.log(rng.uniform(0.05, 0.2, size=(n, 3)))
    quats = np.tile([[1.0, 0.0, 0.0, 0.0]], (n, 1))
    opacities = np.zeros(n)
    path = tmp_path / "splats.ply"
    _write_full_3dgs_ply(path, centers, scales_log, quats, opacities)

    loaded = load_representation(
        str(path),
        representation="gaussian-points",
        splat_method="ellipsoid",
        splat_samples=12,
    )
    assert loaded.final_count == n * 12
    # Filter trail records the sampling parameters.
    assert any("splat_method=ellipsoid" in f for f in loaded.applied_filters)
    assert any("samples_per_splat=12" in f for f in loaded.applied_filters)


def test_load_representation_ellipsoid_missing_fields_errors(tmp_path: Path) -> None:
    """A PLY without scale_*/rot_* must raise a clear error in ellipsoid mode."""
    # Build a minimal 3DGS PLY missing scale/rot fields.
    centers = np.array([[0.0, 0.0, 0.0]])
    n = centers.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float opacity\nend_header\n"
    ).encode("ascii")
    payload = np.empty((n, 4), dtype="<f4")
    payload[:, :3] = centers.astype("<f4")
    payload[:, 3] = 0.0
    path = tmp_path / "minimal.ply"
    with path.open("wb") as f:
        f.write(header)
        f.write(payload.tobytes())

    with pytest.raises(ValueError, match="splat_method='ellipsoid'"):
        load_representation(
            str(path), representation="gaussian-points", splat_method="ellipsoid"
        )


def test_load_representation_ellipsoid_with_opacity_filter(tmp_path: Path) -> None:
    """Opacity filter drops splats before ellipsoid sampling — N*K reflects survivors."""
    centers = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    scales_log = np.log(np.full((3, 3), 0.5))
    quats = np.tile([[1.0, 0.0, 0.0, 0.0]], (3, 1))
    # logits: 5 → sigmoid≈0.99 (keep), -5 → ≈0.007 (drop), 0 → 0.5 (keep at threshold 0.4)
    opacities = np.array([5.0, -5.0, 0.0])
    path = tmp_path / "splats.ply"
    _write_full_3dgs_ply(path, centers, scales_log, quats, opacities)

    loaded = load_representation(
        str(path),
        representation="gaussian-points",
        opacity_threshold=0.4,
        splat_method="ellipsoid",
        splat_samples=6,
    )
    # 2 splats survive opacity filter → 2 * 6 = 12 samples
    assert loaded.final_count == 12
    assert any("opacity>=0.4 kept=2" in f for f in loaded.applied_filters)


def test_splat_method_constants_exposed() -> None:
    assert "centers" in SPLAT_METHODS
    assert "ellipsoid" in SPLAT_METHODS
    assert DEFAULT_SPLAT_SAMPLES >= 2
