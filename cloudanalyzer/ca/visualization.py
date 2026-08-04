"""Visualization module for point cloud coloring and snapshot."""

import os

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
    """Save a point cloud snapshot.

    Open3D's window renderer is used when a display is available.  In
    headless environments, a deterministic Matplotlib projection is used
    instead.  This avoids making CI depend on an EGL/OSMesa installation;
    callers can force a backend with ``CLOUDANALYZER_RENDER_BACKEND`` set to
    ``open3d`` or ``matplotlib``.

    Args:
        pcd: Point cloud to render.
        path: Output image path (png).
        width: Image width.
        height: Image height.
    """
    backend = os.environ.get("CLOUDANALYZER_RENDER_BACKEND", "auto").lower()
    if backend not in {"auto", "open3d", "matplotlib"}:
        raise ValueError(
            "CLOUDANALYZER_RENDER_BACKEND must be one of: auto, open3d, matplotlib"
        )

    has_display = bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if backend == "matplotlib" or (backend == "auto" and not has_display):
        _save_snapshot_matplotlib(pcd, path, width=width, height=height)
        return

    if backend == "open3d":
        _save_snapshot_open3d(pcd, path, width=width, height=height)
        return

    try:
        _save_snapshot_open3d(pcd, path, width=width, height=height)
    except RuntimeError:
        _save_snapshot_matplotlib(pcd, path, width=width, height=height)


def _save_snapshot_open3d(
    pcd: o3d.geometry.PointCloud,
    path: str,
    *,
    width: int,
    height: int,
) -> None:
    """Save a snapshot through Open3D when a windowing backend is available."""
    vis = o3d.visualization.Visualizer()
    try:
        created = vis.create_window(visible=False, width=width, height=height)
        if not created:
            raise RuntimeError("Open3D could not create a rendering window")
        vis.add_geometry(pcd)

        # Auto-set viewpoint
        view_control = vis.get_view_control()
        if view_control is None:
            raise RuntimeError("Open3D did not provide a view control")
        view_control.set_zoom(0.8)
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(str(path), do_render=True)
    finally:
        vis.destroy_window()


def _save_snapshot_matplotlib(
    pcd: o3d.geometry.PointCloud,
    path: str,
    *,
    width: int,
    height: int,
    max_points: int = 200_000,
) -> None:
    """Save a deterministic, display-independent 3D point projection."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    points = np.asarray(pcd.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point cloud must contain an (N, 3) point array")

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = np.asarray(pcd.colors, dtype=float)
    if colors.shape != (len(finite), 3):
        colors = None
    elif len(colors):
        colors = colors[finite]

    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]

    figure = Figure(figsize=(width / 100, height / 100), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111, projection="3d")
    axis.set_axis_off()

    if len(points):
        if colors is None:
            colors = np.broadcast_to(np.array([[0.15, 0.45, 0.85]]), (len(points), 3))
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=np.clip(colors, 0.0, 1.0),
            s=2,
            depthshade=False,
            linewidths=0,
        )
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        center = (lower + upper) / 2.0
        radius = max(float(np.max(upper - lower)) / 2.0, 1e-9)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.view_init(elev=25, azim=-60)

    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    figure.savefig(path, dpi=100, facecolor="white", edgecolor="white")
