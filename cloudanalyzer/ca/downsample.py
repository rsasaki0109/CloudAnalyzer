"""Voxel downsampling module."""

import open3d as o3d

from ca.io import load_point_cloud


def downsample(path: str, voxel_size: float, output: str) -> dict:
    """Downsample a point cloud using voxel grid filtering.

    Args:
        path: Input point cloud file path.
        voxel_size: Voxel size for downsampling.
        output: Output file path.

    Returns:
        Dict with original and downsampled point counts.
    """
    if voxel_size <= 0:
        raise ValueError(f"Voxel size must be positive, got {voxel_size}")

    pcd = load_point_cloud(path)
    original_count = len(pcd.points)

    down = pcd.voxel_down_sample(voxel_size)
    down_count = len(down.points)

    o3d.io.write_point_cloud(output, down)

    return {
        "original_points": original_count,
        "downsampled_points": down_count,
        "reduction_ratio": 1.0 - down_count / original_count if original_count > 0 else 0.0,
        "voxel_size": voxel_size,
        "output": output,
    }
