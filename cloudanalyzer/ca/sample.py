"""Random point sampling module."""

import numpy as np
import open3d as o3d

from ca.io import load_point_cloud


def random_sample(input_path: str, output_path: str, num_points: int) -> dict:
    """Randomly sample a fixed number of points from a point cloud.

    Args:
        input_path: Input point cloud file path.
        output_path: Output file path.
        num_points: Number of points to keep.

    Returns:
        Dict with original/sampled point counts.

    Raises:
        ValueError: If num_points exceeds the original point count.
    """
    pcd = load_point_cloud(input_path)
    original_count = len(pcd.points)

    if num_points > original_count:
        raise ValueError(
            f"Requested {num_points} points but cloud only has {original_count}"
        )

    if num_points == original_count:
        sampled = pcd
    else:
        indices = np.random.default_rng().choice(original_count, size=num_points, replace=False)
        sampled = pcd.select_by_index(indices.tolist())

    o3d.io.write_point_cloud(output_path, sampled)

    return {
        "input": input_path,
        "output": output_path,
        "original_points": original_count,
        "sampled_points": len(sampled.points),
    }
