"""Density heatmap visualization module."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ca.io import load_point_cloud


def density_map(
    input_path: str,
    output_path: str,
    resolution: float = 0.5,
    axis: str = "z",
    width: int = 1920,
    height: int = 1080,
) -> dict:
    """Generate a 2D density heatmap of a point cloud projected onto a plane.

    Args:
        input_path: Input point cloud file path.
        output_path: Output image path (png).
        resolution: Grid cell size for density binning.
        axis: Projection axis ("x", "y", or "z"). Points are projected
              onto the plane perpendicular to this axis.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        Dict with grid info and stats.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis.lower() not in axis_map:
        raise ValueError(f"Invalid axis: '{axis}'. Must be 'x', 'y', or 'z'.")

    pcd = load_point_cloud(input_path)
    points = np.asarray(pcd.points)
    num_points = len(points)

    ax_idx = axis_map[axis.lower()]
    # Project onto the 2 remaining axes
    axes = [i for i in range(3) if i != ax_idx]
    proj = points[:, axes]

    # Build 2D histogram
    x_range = (proj[:, 0].min(), proj[:, 0].max())
    y_range = (proj[:, 1].min(), proj[:, 1].max())

    x_bins = max(1, int(np.ceil((x_range[1] - x_range[0]) / resolution)))
    y_bins = max(1, int(np.ceil((y_range[1] - y_range[0]) / resolution)))

    hist, xedges, yedges = np.histogram2d(
        proj[:, 0], proj[:, 1], bins=[x_bins, y_bins],
        range=[x_range, y_range],
    )

    # Plot
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    im = ax.imshow(
        hist.T, origin="lower", aspect="auto",
        extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
        cmap="hot", interpolation="nearest",
    )
    axis_labels = ["X", "Y", "Z"]
    remaining = [axis_labels[i] for i in axes]
    ax.set_xlabel(remaining[0])
    ax.set_ylabel(remaining[1])
    ax.set_title(f"Density Map (proj. along {axis.upper()}, res={resolution})")
    fig.colorbar(im, ax=ax, label="Point count")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "input": input_path,
        "output": output_path,
        "num_points": num_points,
        "projection_axis": axis.lower(),
        "resolution": resolution,
        "grid_size": [x_bins, y_bins],
        "max_density": int(hist.max()),
        "mean_density": float(hist.mean()),
    }
