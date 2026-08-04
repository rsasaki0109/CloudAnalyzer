"""Tests for ca.visualization module."""

import numpy as np
import open3d as o3d
import pytest

from ca.visualization import colorize, save_snapshot


class TestColorize:
    def test_colors_applied(self, simple_pcd):
        distances = np.random.default_rng(0).random(100)
        result = colorize(simple_pcd, distances)
        assert result is simple_pcd  # modified in place
        colors = np.asarray(result.colors)
        assert colors.shape == (100, 3)
        assert np.all(colors >= 0.0) and np.all(colors <= 1.0)

    def test_uniform_distances(self, simple_pcd):
        distances = np.ones(100) * 5.0
        colorize(simple_pcd, distances)
        colors = np.asarray(simple_pcd.colors)
        # All same distance → all same color
        np.testing.assert_array_equal(colors[0], colors[-1])

    def test_empty_distances(self, simple_pcd):
        result = colorize(simple_pcd, np.array([]))
        assert result is simple_pcd

    def test_gradient_ordering(self, simple_pcd):
        distances = np.linspace(0, 1, 100)
        colorize(simple_pcd, distances, cmap_name="jet")
        colors = np.asarray(simple_pcd.colors)
        # First point (min distance) and last point (max distance) should differ
        assert not np.allclose(colors[0], colors[-1])


class TestSaveSnapshot:
    def test_creates_png(self, tmp_path, simple_pcd):
        # Give the point cloud some colors so rendering works
        colorize(simple_pcd, np.random.default_rng(0).random(100))
        path = tmp_path / "snapshot.png"
        save_snapshot(simple_pcd, str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_matplotlib_backend(self, monkeypatch, tmp_path, simple_pcd):
        monkeypatch.setenv("CLOUDANALYZER_RENDER_BACKEND", "matplotlib")
        path = tmp_path / "matplotlib-snapshot.png"
        save_snapshot(simple_pcd, str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_invalid_backend(self, monkeypatch, tmp_path, simple_pcd):
        monkeypatch.setenv("CLOUDANALYZER_RENDER_BACKEND", "unknown")
        with pytest.raises(ValueError, match="CLOUDANALYZER_RENDER_BACKEND"):
            save_snapshot(simple_pcd, str(tmp_path / "snapshot.png"))
