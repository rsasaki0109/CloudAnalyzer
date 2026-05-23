"""Split point cloud into grid tiles."""

from typing import cast

import numpy as np
import open3d as o3d
import yaml  # type: ignore[import-untyped]
from pathlib import Path

from ca.io import load_point_cloud, save_point_cloud
from ca.log import logger


def split(
    input_path: str,
    output_dir: str,
    grid_size: float,
    axis: str = "xy",
) -> dict:
    """Split a point cloud into grid tiles.

    Args:
        input_path: Input point cloud file path.
        output_dir: Output directory for tile files.
        grid_size: Size of each grid cell.
        axis: Split axes ("xy", "xz", or "yz").

    Returns:
        Dict with tile info and counts.
    """
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if axis not in axis_map:
        raise ValueError(f"Invalid axis: '{axis}'. Must be 'xy', 'xz', or 'yz'.")

    pcd = load_point_cloud(input_path)
    points = np.asarray(pcd.points)
    total = len(points)
    ax0, ax1 = axis_map[axis]

    # Compute grid indices
    origin: np.ndarray = points[:, [ax0, ax1]].min(axis=0)
    indices = ((points[:, [ax0, ax1]] - origin) / grid_size).astype(int)

    # Group by tile
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Bucket points into tiles in C: find each unique (i, j) cell, then sort
    # point indices by their tile id so a single contiguous slice gives each
    # tile's index list. Previously this was a per-point Python loop, which
    # made multi-million-point splits unusable.
    if len(points) > 0:
        unique_keys, inverse = np.unique(indices, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        order = np.argsort(inverse, kind="stable")
        sorted_inverse = inverse[order]
        boundaries = np.flatnonzero(np.diff(sorted_inverse)) + 1
        group_starts = np.concatenate([[0], boundaries])
        group_ends = np.concatenate([boundaries, [len(order)]])
        tiles: dict[tuple[int, int], np.ndarray] = {
            (int(unique_keys[k, 0]), int(unique_keys[k, 1])): order[start:end]
            for k, (start, end) in enumerate(zip(group_starts, group_ends))
        }
    else:
        tiles = {}

    tile_info = []
    ext = Path(input_path).suffix
    for (i, j), point_indices in sorted(tiles.items()):
        tile_pcd = pcd.select_by_index(point_indices.tolist())
        filename = f"tile_{i:04d}_{j:04d}{ext}"
        tile_path = str(out / filename)
        save_point_cloud(tile_path, tile_pcd)
        tile_info.append({
            "file": filename,
            "grid": [i, j],
            "points": len(point_indices),
        })
        logger.debug("  %s: %d pts", filename, len(point_indices))

    # Write metadata.yaml
    metadata = {
        "grid_size": grid_size,
        "axis": axis,
        "tiles": [
            {
                "file": info["file"],
                "grid": info["grid"],
                "origin": [
                    float(origin[0] + cast(list[int], info["grid"])[0] * grid_size),
                    float(origin[1] + cast(list[int], info["grid"])[1] * grid_size),
                ],
                "points": info["points"],
            }
            for info in tile_info
        ],
    }
    metadata_path = out / "metadata.yaml"
    metadata_path.write_text(
        yaml.dump(metadata, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    logger.debug("metadata: %s", metadata_path)

    return {
        "input": input_path,
        "output_dir": output_dir,
        "total_points": total,
        "grid_size": grid_size,
        "axis": axis,
        "num_tiles": len(tile_info),
        "tiles": tile_info,
        "metadata_path": str(metadata_path),
    }
