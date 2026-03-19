"""Visualization module for point cloud coloring and snapshot."""

import numpy as np
import matplotlib
import open3d as o3d


def colorize(
    pcd: o3d.geometry.PointCloud,
    distances: np.ndarray,
    cmap_name: str = "jet",
) -> o3d.geometry.PointCloud:
    """Map distance values to colors (blue=near, red=far).

    Args:
        pcd: Point cloud to colorize.
        distances: Distance array (same length as pcd points).
        cmap_name: Matplotlib colormap name.

    Returns:
        Colorized point cloud (modified in place and returned).
    """
    if len(distances) == 0:
        return pcd

    d_min = distances.min()
    d_max = distances.max()

    if d_max - d_min < 1e-12:
        normalized = np.zeros_like(distances)
    else:
        normalized = (distances - d_min) / (d_max - d_min)

    cmap = matplotlib.colormaps[cmap_name]
    colors = cmap(normalized)[:, :3]  # RGB only, drop alpha
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def save_snapshot(
    pcd: o3d.geometry.PointCloud,
    path: str,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Save point cloud as an image using offscreen rendering.

    Args:
        pcd: Point cloud to render.
        path: Output image path (png).
        width: Image width.
        height: Image height.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    # Auto-set viewpoint
    vis.get_view_control().set_zoom(0.8)
    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(str(path), do_render=True)
    vis.destroy_window()
