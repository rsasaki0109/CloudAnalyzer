"""Keep the original color mapping and image bytes when packing RGB rows."""

import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np
import open3d as o3d
import pytest

from ca.visualization import colorize, save_snapshot


def original_colors(distances, cmap_name):
    """Original mapping, including the strided RGBA-to-RGB view."""
    minimum, maximum = distances.min(), distances.max()
    with np.errstate(invalid="ignore", over="ignore"):
        normalized = (
            np.zeros_like(distances)
            if maximum - minimum < 1e-12
            else (distances - minimum) / (maximum - minimum)
        )
    return matplotlib.colormaps[cmap_name](normalized)[:, :3]


def populated_cloud(size):
    cloud = o3d.geometry.PointCloud()
    points = np.arange(size * 3, dtype=float).reshape(size, 3) / 13
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(np.roll(points, 1, axis=1))
    cloud.colors = o3d.utility.Vector3dVector(np.full((size, 3), 0.25))
    cloud.covariances = o3d.utility.Matrix3dVector(
        np.tile(np.eye(3), (size, 1, 1))
    )
    return cloud


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed"])
@pytest.mark.parametrize("cmap_name", ["jet", "viridis"])
def test_exact_rgb_and_point_attributes(dtype, layout, cmap_name):
    values = np.array([-2, 0, 0, 1, 4, 5, 9, 12], dtype=dtype)
    if layout == "strided":
        storage = np.empty(len(values) * 2, dtype=dtype)
        storage[::2] = values
        values = storage[::2]
    elif layout == "reversed":
        values = values[::-1]
    cloud = populated_cloud(len(values))
    fields = {
        name: np.asarray(getattr(cloud, name)).tobytes()
        for name in ("points", "normals", "covariances")
    }
    expected = original_colors(values, cmap_name).tobytes()
    assert colorize(cloud, values, cmap_name) is cloud
    assert np.asarray(cloud.colors).tobytes() == expected
    for name, before in fields.items():
        assert np.asarray(getattr(cloud, name)).tobytes() == before
    # Both the input array and Matplotlib scratch may be released or changed.
    values[:] = 0
    assert np.asarray(cloud.colors).tobytes() == expected


@pytest.mark.parametrize("values", [
    [5.0, 5.0, 5.0],
    [0.0, 0.5e-12, 0.9e-12],
    [0.0, 0.5e-12, 1.0e-12],
    [np.nan, 0.0, 1.0],
    [-np.inf, 0.0, np.inf],
    [-1e308, 0.0, 1e308],
    [-0.0],
])
def test_edge_values_match_original_mapping(values):
    distances = np.array(values)
    cloud = populated_cloud(len(values))
    expected = original_colors(distances, "jet")
    with np.errstate(invalid="ignore", over="ignore"):
        colorize(cloud, distances)
    assert np.asarray(cloud.colors).tobytes() == expected.tobytes()


def test_custom_colormap_bad_colors():
    name = "cloudanalyzer_test_packed_colors"
    cmap = ListedColormap([[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]])
    cmap.set_bad([0.21, 0.32, 0.43, 0.54])
    matplotlib.colormaps.register(cmap, name=name)
    try:
        values = np.array([0.0, np.nan, 1.0])
        cloud = populated_cloud(len(values))
        colorize(cloud, values, name)
        assert np.asarray(cloud.colors).tobytes() == original_colors(values, name).tobytes()
    finally:
        matplotlib.colormaps.unregister(name)


def test_empty_keeps_existing_colors_and_does_not_resolve_colormap():
    cloud = populated_cloud(3)
    before = np.asarray(cloud.colors).tobytes()
    assert colorize(cloud, np.array([]), "missing-colormap") is cloud
    assert np.asarray(cloud.colors).tobytes() == before


@pytest.mark.parametrize("cmap_name", ["jet", "viridis"])
def test_snapshot_bytes_match_original_color_mapping(monkeypatch, tmp_path, cmap_name):
    monkeypatch.setenv("CLOUDANALYZER_RENDER_BACKEND", "matplotlib")
    values = np.linspace(-2.0, 9.0, 129)[::-1]
    original = populated_cloud(len(values))
    candidate = o3d.geometry.PointCloud(original)
    original.colors = o3d.utility.Vector3dVector(original_colors(values, cmap_name))
    colorize(candidate, values, cmap_name)
    old_path, new_path = tmp_path / "original.png", tmp_path / "packed.png"
    save_snapshot(original, str(old_path), width=320, height=240)
    save_snapshot(candidate, str(new_path), width=320, height=240)
    assert old_path.read_bytes() == new_path.read_bytes()
