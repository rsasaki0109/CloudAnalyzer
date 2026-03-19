"""Point cloud registration module."""

import numpy as np
import open3d as o3d


SUPPORTED_METHODS = {"icp", "gicp"}


def register(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    method: str = "gicp",
    max_correspondence_distance: float = 1.0,
) -> tuple[o3d.geometry.PointCloud, float, float]:
    """Perform ICP or GICP registration.

    Args:
        source: Source point cloud.
        target: Target point cloud.
        method: Registration method ("icp" or "gicp").
        max_correspondence_distance: Maximum correspondence distance.

    Returns:
        Tuple of (transformed_source, fitness, rmse).

    Raises:
        ValueError: If method is not supported.
    """
    method = method.lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: '{method}'. Supported: {', '.join(sorted(SUPPORTED_METHODS))}"
        )

    init_transform = np.eye(4)

    # Estimate normals for GICP (required for point-to-plane)
    for pcd in [source, target]:
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
            )

    if method == "icp":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:  # gicp
        estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

    result = o3d.pipelines.registration.registration_generalized_icp(
        source,
        target,
        max_correspondence_distance,
        init_transform,
        estimation,
    ) if method == "gicp" else o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        init_transform,
        estimation,
    )

    transformed = source.transform(result.transformation)

    return transformed, result.fitness, result.inlier_rmse
