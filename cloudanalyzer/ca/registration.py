"""Point cloud registration module."""

from dataclasses import dataclass

import numpy as np
import open3d as o3d


SUPPORTED_METHODS = {"icp", "gicp"}


@dataclass(frozen=True)
class RegistrationResult:
    """Detailed point cloud registration result."""

    transformed: o3d.geometry.PointCloud
    fitness: float
    rmse: float
    transformation: np.ndarray


def register_detailed(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    method: str = "gicp",
    max_correspondence_distance: float = 1.0,
    init_transform: np.ndarray | None = None,
) -> RegistrationResult:
    """Perform ICP or GICP registration and return the transformation."""

    method = method.lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: '{method}'. Supported: {', '.join(sorted(SUPPORTED_METHODS))}"
        )

    init = np.eye(4) if init_transform is None else np.asarray(init_transform, dtype=float)
    if init.shape != (4, 4):
        raise ValueError("init_transform must be a 4x4 matrix")

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

    result = (
        o3d.pipelines.registration.registration_generalized_icp(
            source,
            target,
            max_correspondence_distance,
            init,
            estimation,
        )
        if method == "gicp"
        else o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance,
            init,
            estimation,
        )
    )

    transformed = o3d.geometry.PointCloud(source)
    transformed.transform(result.transformation)

    return RegistrationResult(
        transformed=transformed,
        fitness=float(result.fitness),
        rmse=float(result.inlier_rmse),
        transformation=np.asarray(result.transformation, dtype=float),
    )


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
    result = register_detailed(
        source,
        target,
        method=method,
        max_correspondence_distance=max_correspondence_distance,
    )
    return result.transformed, result.fitness, result.rmse
