"""Pipeline: chain filter → downsample → evaluate in one call."""

from ca.io import load_point_cloud
from ca.filter import filter_outliers
from ca.downsample import downsample
from ca.evaluate import evaluate
from ca.log import logger


def run_pipeline(
    input_path: str,
    reference_path: str,
    output_path: str,
    voxel_size: float = 0.1,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    thresholds: list[float] | None = None,
) -> dict:
    """Run filter → downsample → evaluate pipeline.

    Args:
        input_path: Input point cloud to process.
        reference_path: Reference point cloud for evaluation.
        output_path: Final output file path.
        voxel_size: Voxel size for downsampling.
        nb_neighbors: Neighbors for outlier filter.
        std_ratio: Std ratio for outlier filter.
        thresholds: Distance thresholds for evaluation.

    Returns:
        Dict with step results and final evaluation.
    """
    import tempfile
    import os

    # Step 1: Filter
    logger.info("Step 1/3: Filtering outliers...")
    tmp_filtered = os.path.join(tempfile.gettempdir(), "ca_pipeline_filtered.pcd")
    filter_result = filter_outliers(input_path, tmp_filtered, nb_neighbors, std_ratio)
    logger.info("  Removed %d outliers", filter_result["removed_points"])

    # Step 2: Downsample
    logger.info("Step 2/3: Downsampling (voxel=%.3f)...", voxel_size)
    downsample_result = downsample(tmp_filtered, voxel_size, output_path)
    logger.info("  %d -> %d pts (%.1f%% reduction)",
                downsample_result["original_points"],
                downsample_result["downsampled_points"],
                downsample_result["reduction_ratio"] * 100)

    # Step 3: Evaluate against reference
    logger.info("Step 3/3: Evaluating against reference...")
    eval_result = evaluate(output_path, reference_path, thresholds=thresholds)
    logger.info("  Chamfer=%.4f  AUC=%.4f", eval_result["chamfer_distance"], eval_result["auc"])

    # Cleanup temp
    try:
        os.remove(tmp_filtered)
    except OSError:
        pass

    return {
        "input": input_path,
        "reference": reference_path,
        "output": output_path,
        "filter": {
            "original": filter_result["original_points"],
            "filtered": filter_result["filtered_points"],
            "removed": filter_result["removed_points"],
        },
        "downsample": {
            "input": downsample_result["original_points"],
            "output": downsample_result["downsampled_points"],
            "voxel_size": voxel_size,
            "reduction": downsample_result["reduction_ratio"],
        },
        "evaluation": {
            "chamfer": eval_result["chamfer_distance"],
            "hausdorff": eval_result["hausdorff_distance"],
            "auc": eval_result["auc"],
            "f1_scores": eval_result["f1_scores"],
        },
    }
