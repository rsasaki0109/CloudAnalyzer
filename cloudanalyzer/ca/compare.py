"""Core compare pipeline."""

import numpy as np

from ca.io import load_point_cloud
from ca.log import logger
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
        threshold: Distance threshold to check.

    Returns:
        Report dict.
    """
    # 1. Load
    logger.info("Loading source: %s", source_path)
    source = load_point_cloud(source_path)
    logger.info("  -> %d points", len(source.points))

    logger.info("Loading target: %s", target_path)
    target = load_point_cloud(target_path)
    logger.info("  -> %d points", len(target.points))

    # 2. Register (optional)
    fitness = None
    rmse = None

    if method:
        logger.info("Registering with %s...", method.upper())
        source, fitness, rmse = register(source, target, method=method)
        logger.info("  -> Fitness: %.4f, RMSE: %.4f", fitness, rmse)

    # 3. Compute distances
    logger.info("Computing nearest neighbor distances...")
    distances = compute_nn_distance(source, target)
    stats = summarize(distances)
    logger.info("  -> Mean: %.4f, Max: %.4f", stats["mean"], stats["max"])

    # 3.5 Threshold check (optional)
    thresh_result = None
    if threshold is not None:
        thresh_result = threshold_stats(distances, threshold)
        logger.info(
            "  -> Threshold %s: %d/%d (%.1f%%) exceed",
            threshold, thresh_result["exceed_count"],
            thresh_result["total"], thresh_result["exceed_ratio"] * 100,
        )

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
        logger.info("JSON saved: %s", json_path)

    if report_path:
        make_markdown(data, report_path)
        logger.info("Report saved: %s", report_path)

    if snapshot_path:
        logger.info("Saving snapshot: %s", snapshot_path)
        save_snapshot(source, snapshot_path)

    logger.info("Done.")
    return data
