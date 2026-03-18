"""Core compare pipeline."""

import numpy as np

from ca.io import load_point_cloud
from ca.registration import register
from ca.metrics import compute_nn_distance, summarize, threshold_stats
from ca.visualization import colorize, save_snapshot
from ca.report import make_json, save_json, make_markdown


def run_compare(
    source_path: str,
    target_path: str,
    method: str | None = "gicp",
    json_path: str | None = None,
    report_path: str | None = None,
    snapshot_path: str | None = None,
    threshold: float | None = None,
) -> dict:
    """Run the full compare pipeline.

    Args:
        source_path: Path to source point cloud.
        target_path: Path to target point cloud.
        method: Registration method ("icp", "gicp") or None to skip.
        json_path: Output path for JSON report.
        report_path: Output path for Markdown report.
        snapshot_path: Output path for snapshot image.

    Returns:
        Report dict.
    """
    # 1. Load
    print(f"Loading source: {source_path}")
    source = load_point_cloud(source_path)
    print(f"  -> {len(source.points)} points")

    print(f"Loading target: {target_path}")
    target = load_point_cloud(target_path)
    print(f"  -> {len(target.points)} points")

    # 2. Register (optional)
    fitness = None
    rmse = None

    if method:
        print(f"Registering with {method.upper()}...")
        source, fitness, rmse = register(source, target, method=method)
        print(f"  -> Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")

    # 3. Compute distances
    print("Computing nearest neighbor distances...")
    distances = compute_nn_distance(source, target)
    stats = summarize(distances)
    print(f"  -> Mean: {stats['mean']:.4f}, Max: {stats['max']:.4f}")

    # 3.5 Threshold check (optional)
    thresh_result = None
    if threshold is not None:
        thresh_result = threshold_stats(distances, threshold)
        print(f"  -> Threshold {threshold}: {thresh_result['exceed_count']}/{thresh_result['total']} ({thresh_result['exceed_ratio']:.1%}) exceed")

    # 4. Colorize
    colorize(source, distances)

    # 5. Save outputs
    data = make_json(
        source_points=len(source.points),
        target_points=len(target.points),
        fitness=fitness,
        rmse=rmse,
        distance_stats=stats,
    )
    if thresh_result:
        data["threshold"] = thresh_result

    if json_path:
        save_json(data, json_path)
        print(f"JSON saved: {json_path}")

    if report_path:
        make_markdown(data, report_path)
        print(f"Report saved: {report_path}")

    if snapshot_path:
        print(f"Saving snapshot: {snapshot_path}")
        save_snapshot(source, snapshot_path)

    print("Done.")
    return data
