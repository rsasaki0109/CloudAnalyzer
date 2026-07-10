"""Experimental, reference-free plane-consistency proxies for point-cloud maps."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _normal_components(
    patches: list[dict[str, Any]],
    *,
    cosine_threshold: float,
    plane_distance: float | None = None,
) -> list[list[int]]:
    """Deterministic projective-normal connected components."""
    normals = np.asarray([patch["normal"] for patch in patches])
    radius = float(np.sqrt(max(0.0, 2.0 - 2.0 * cosine_threshold))) + 1e-12
    tree = cKDTree(normals)
    opposite_tree = cKDTree(-normals)
    parent = list(range(len(patches)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left, normal in enumerate(normals):
        candidates = set(tree.query_ball_point(normal, radius))
        candidates.update(opposite_tree.query_ball_point(normal, radius))
        for right in sorted(index for index in candidates if index > left):
            dot = float(np.dot(normal, normals[right]))
            if abs(dot) < cosine_threshold:
                continue
            if plane_distance is not None:
                aligned = normals[right] if dot >= 0 else -normals[right]
                common = normal + aligned
                common /= np.linalg.norm(common)
                separation = abs(
                    float(np.dot(common, patches[left]["centroid"] - patches[right]["centroid"]))
                )
                if separation > plane_distance:
                    continue
            union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(patches)):
        grouped.setdefault(find(index), []).append(index)
    return [grouped[root] for root in sorted(grouped)]


def evaluate_plane_consistency_points(
    points: np.ndarray,
    *,
    voxel_size: float = 1.0,
    min_points: int = 12,
    plane_merge_distance: float = 0.2,
    normal_angle_degrees: float = 15.0,
) -> dict[str, Any]:
    """Measure repeatability of locally planar patches without ground truth.

    These are deterministic PNE/CPV-inspired proxies, not reproductions of
    the metrics from Ouyang et al.  Lower values indicate better consistency.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be shape (N, 3); got {pts.shape}")
    if voxel_size <= 0 or plane_merge_distance <= 0:
        raise ValueError("voxel_size and plane_merge_distance must be > 0")
    if min_points < 3:
        raise ValueError("min_points must be >= 3")
    if not 0 < normal_angle_degrees < 90:
        raise ValueError("normal_angle_degrees must be between 0 and 90")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must contain only finite values")

    # Build the grid in a deterministic PCA frame.  This removes dependence on
    # world translation and (for non-degenerate global spectra) rigid rotation.
    centered_cloud = pts - pts.mean(axis=0) if len(pts) else pts
    if len(pts) >= 3:
        _, frame = np.linalg.eigh(centered_cloud.T @ centered_cloud)
        canonical = centered_cloud @ frame
        for axis in range(3):
            third_moment = float(np.sum(canonical[:, axis] ** 3))
            if third_moment < 0:
                canonical[:, axis] *= -1
    else:
        canonical = centered_cloud
    voxel_origin = canonical.min(axis=0) if len(canonical) else np.zeros(3)
    keys = np.floor((canonical - voxel_origin) / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    sorted_points = pts[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_keys, axis=0), axis=1)) + 1]
    ends = np.r_[starts[1:], len(sorted_points)]
    patches: list[dict[str, Any]] = []
    for start, end in zip(starts, ends):
        key = sorted_keys[start]
        count = int(end - start)
        if count < min_points:
            continue
        group = sorted_points[start:end]
        centroid = group.mean(axis=0)
        covariance = np.cov(group, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        largest = float(eigenvalues[2])
        if largest <= np.finfo(float).eps:
            continue
        normal = eigenvectors[:, 0]
        # Standard PCA planarity: a line has lambda0 ~= lambda1 ~= 0 and is
        # rejected, while a plane has lambda0 << lambda1 ~= lambda2.
        planarity = float((eigenvalues[1] - eigenvalues[0]) / largest)
        if planarity < 0.5:
            continue
        patches.append(
            {
                "voxel": [int(v) for v in key],
                "centroid": centroid,
                "normal": normal,
                "weight": float(count) * planarity,
            }
        )
    if len(patches) < 2:
        return {
            "plane_normal_dispersion": float("nan"),
            "coplanar_offset_rmse": float("nan"),
            "num_plane_patches": len(patches),
            "num_normal_groups": 0,
            "num_coplanar_groups": 0,
            "voxel_size": float(voxel_size),
            "experimental_proxy": True,
        }

    cosine_threshold = float(np.cos(np.deg2rad(normal_angle_degrees)))
    normal_groups = _normal_components(patches, cosine_threshold=cosine_threshold)
    cluster_dispersions: list[float] = []
    cluster_weights: list[float] = []
    for group_indices in normal_groups:
        selected = [patches[index] for index in group_indices]
        axis_weights = np.asarray([patch["weight"] for patch in selected])
        axis_normals = np.asarray([patch["normal"] for patch in selected])
        scatter = np.einsum("n,ni,nj->ij", axis_weights, axis_normals, axis_normals)
        scatter /= axis_weights.sum()
        cluster_dispersions.append(float(1.0 - np.linalg.eigvalsh(scatter)[-1]))
        cluster_weights.append(float(axis_weights.sum()))
    plane_normal_dispersion = float(
        np.average(cluster_dispersions, weights=cluster_weights)
    )

    coplanar_groups = _normal_components(
        patches,
        cosine_threshold=cosine_threshold,
        plane_distance=plane_merge_distance,
    )
    group_squared_errors: list[float] = []
    group_weights: list[float] = []
    supported_groups = 0
    for group_indices in coplanar_groups:
        if len(group_indices) < 2:
            continue
        supported_groups += 1
        selected = [patches[index] for index in group_indices]
        group_patch_weights = np.asarray([patch["weight"] for patch in selected])
        reference = np.asarray(selected[0]["normal"])
        aligned = np.asarray([
            patch["normal"] if np.dot(reference, patch["normal"]) >= 0 else -patch["normal"]
            for patch in selected
        ])
        group_normal = np.average(aligned, axis=0, weights=group_patch_weights)
        group_normal /= np.linalg.norm(group_normal)
        centroids = np.asarray([patch["centroid"] for patch in selected])
        group_centroid = np.average(centroids, axis=0, weights=group_patch_weights)
        # Relative offsets are invariant to a common rigid translation.
        residuals = (centroids - group_centroid) @ group_normal
        group_squared_errors.append(float(np.average(residuals**2, weights=group_patch_weights)))
        group_weights.append(float(group_patch_weights.sum()))
    coplanar_offset_rmse = (
        float(np.sqrt(np.average(group_squared_errors, weights=group_weights)))
        if group_squared_errors
        else float("nan")
    )
    return {
        "plane_normal_dispersion": plane_normal_dispersion,
        "coplanar_offset_rmse": coplanar_offset_rmse,
        "num_plane_patches": len(patches),
        "num_normal_groups": len(normal_groups),
        "num_coplanar_groups": supported_groups,
        "voxel_size": float(voxel_size),
        "experimental_proxy": True,
        "patches": [
            {
                "voxel": patch["voxel"],
                "normal": np.asarray(patch["normal"]).tolist(),
            }
            for patch in patches
        ],
    }


def evaluate_plane_consistency(path: str, **kwargs: Any) -> dict[str, Any]:
    """Load one point cloud and evaluate the experimental proxies."""
    from ca.io import load_point_cloud

    points = np.asarray(load_point_cloud(path).points, dtype=np.float64)
    return {"path": path, **evaluate_plane_consistency_points(points, **kwargs)}


__all__ = ["evaluate_plane_consistency", "evaluate_plane_consistency_points"]
