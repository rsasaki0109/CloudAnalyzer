"""Sequential registration and merge (align multiple scans)."""

import open3d as o3d

from ca.io import load_point_cloud
from ca.registration import register


def align(
    paths: list[str],
    output_path: str,
    method: str = "gicp",
    max_correspondence_distance: float = 1.0,
) -> dict:
    """Align multiple point clouds sequentially and merge.

    The first cloud is the reference. Each subsequent cloud is registered
    to the accumulated result, then merged.

    Args:
        paths: List of point cloud file paths (>= 2).
        output_path: Output file path for merged result.
        method: Registration method ("icp" or "gicp").
        max_correspondence_distance: Max correspondence distance.

    Returns:
        Dict with per-step registration results and total points.

    Raises:
        ValueError: If fewer than 2 paths are given.
    """
    if len(paths) < 2:
        raise ValueError("At least 2 point clouds are required for alignment")

    accumulated = load_point_cloud(paths[0])
    steps = []

    for i, path in enumerate(paths[1:], start=1):
        source = load_point_cloud(path)
        transformed, fitness, rmse = register(
            source, accumulated, method=method,
            max_correspondence_distance=max_correspondence_distance,
        )
        steps.append({
            "step": i,
            "path": path,
            "fitness": fitness,
            "rmse": rmse,
        })
        accumulated += transformed

    o3d.io.write_point_cloud(output_path, accumulated)

    return {
        "output": output_path,
        "total_points": len(accumulated.points),
        "num_inputs": len(paths),
        "method": method,
        "steps": steps,
    }
