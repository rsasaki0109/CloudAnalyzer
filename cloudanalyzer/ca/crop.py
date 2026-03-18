"""Crop point cloud by bounding box."""

import numpy as np
import open3d as o3d

from ca.io import load_point_cloud


def crop(
    input_path: str,
    output_path: str,
    x_min: float,
    y_min: float,
    z_min: float,
    x_max: float,
    y_max: float,
    z_max: float,
) -> dict:
    """Crop a point cloud to an axis-aligned bounding box.

    Args:
        input_path: Input point cloud file path.
        output_path: Output file path.
        x_min, y_min, z_min: Minimum corner of the bounding box.
        x_max, y_max, z_max: Maximum corner of the bounding box.

    Returns:
        Dict with original/cropped point counts.
    """
    pcd = load_point_cloud(input_path)
    original_count = len(pcd.points)

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([x_min, y_min, z_min]),
        max_bound=np.array([x_max, y_max, z_max]),
    )
    cropped = pcd.crop(bbox)
    cropped_count = len(cropped.points)

    o3d.io.write_point_cloud(output_path, cropped)

    return {
        "input": input_path,
        "output": output_path,
        "original_points": original_count,
        "cropped_points": cropped_count,
        "bbox_min": [x_min, y_min, z_min],
        "bbox_max": [x_max, y_max, z_max],
    }
