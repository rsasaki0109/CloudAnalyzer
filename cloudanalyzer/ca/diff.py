"""Quick diff between two point clouds (no reports, just stats)."""

from ca.io import load_point_cloud
from ca.metrics import compute_nn_distance, summarize, threshold_stats


def run_diff(source_path: str, target_path: str, threshold: float | None = None) -> dict:
    """Compute distance stats between two point clouds without registration.

    Args:
        source_path: Path to source point cloud.
        target_path: Path to target point cloud.

    Returns:
        Dict with source/target point counts and distance stats.
    """
    source = load_point_cloud(source_path)
    target = load_point_cloud(target_path)

    distances = compute_nn_distance(source, target)
    stats = summarize(distances)

    result = {
        "source_points": len(source.points),
        "target_points": len(target.points),
        "distance_stats": stats,
    }
    if threshold is not None:
        result["threshold"] = threshold_stats(distances, threshold)
    return result
