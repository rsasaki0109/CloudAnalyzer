"""Statistical outlier removal module."""

import open3d as o3d

from ca.io import load_point_cloud


def filter_outliers(
    input_path: str,
    output_path: str,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> dict:
    """Remove statistical outliers from a point cloud.

    Args:
        input_path: Input point cloud file path.
        output_path: Output file path.
        nb_neighbors: Number of neighbors for mean distance estimation.
        std_ratio: Standard deviation multiplier threshold.

    Returns:
        Dict with original/filtered point counts.
    """
    pcd = load_point_cloud(input_path)
    original_count = len(pcd.points)

    filtered, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    filtered_count = len(filtered.points)
    removed_count = original_count - filtered_count

    o3d.io.write_point_cloud(output_path, filtered)

    return {
        "input": input_path,
        "output": output_path,
        "original_points": original_count,
        "filtered_points": filtered_count,
        "removed_points": removed_count,
        "nb_neighbors": nb_neighbors,
        "std_ratio": std_ratio,
    }
