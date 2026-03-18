"""Interactive 3D point cloud viewer."""

import open3d as o3d

from ca.io import load_point_cloud


def view(paths: list[str]) -> None:
    """Open an interactive 3D viewer for one or more point clouds.

    Args:
        paths: List of point cloud file paths to display.
    """
    pcds: list[o3d.geometry.PointCloud] = []
    for path in paths:
        pcd = load_point_cloud(path)
        if not pcd.has_colors():
            # Assign a distinct color per cloud
            import numpy as np
            palette = [
                [0.2, 0.6, 1.0],
                [1.0, 0.4, 0.2],
                [0.2, 0.9, 0.4],
                [0.9, 0.8, 0.1],
                [0.7, 0.3, 0.9],
            ]
            color = palette[len(pcds) % len(palette)]
            pcd.paint_uniform_color(color)
        pcds.append(pcd)

    o3d.visualization.draw_geometries(
        pcds,
        window_name="CloudAnalyzer Viewer",
        width=1280,
        height=720,
    )
