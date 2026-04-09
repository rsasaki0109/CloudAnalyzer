"""Normal estimation module."""

import open3d as o3d

from ca.io import load_point_cloud, save_point_cloud


def estimate_normals(
    input_path: str,
    output_path: str,
    radius: float = 0.5,
    max_nn: int = 30,
) -> dict:
    """Estimate normals for a point cloud and save.

    Args:
        input_path: Input point cloud file path.
        output_path: Output file path with normals.
        radius: Search radius for normal estimation.
        max_nn: Maximum number of neighbors.

    Returns:
        Dict with point count and estimation parameters.
    """
    pcd = load_point_cloud(input_path)
    num_points = len(pcd.points)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    save_point_cloud(output_path, pcd)

    return {
        "input": input_path,
        "output": output_path,
        "num_points": num_points,
        "radius": radius,
        "max_nn": max_nn,
    }
