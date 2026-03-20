"""Split point cloud into grid tiles."""

import numpy as np
import open3d as o3d
from pathlib import Path

from ca.io import load_point_cloud
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
    origin = points[:, [ax0, ax1]].min(axis=0)
    indices = ((points[:, [ax0, ax1]] - origin) / grid_size).astype(int)

    # Group by tile
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tiles: dict[tuple[int, int], list[int]] = {}
    for idx in range(len(points)):
        key = (int(indices[idx, 0]), int(indices[idx, 1]))
        if key not in tiles:
            tiles[key] = []
        tiles[key].append(idx)

    tile_info = []
    ext = Path(input_path).suffix
    for (i, j), point_indices in sorted(tiles.items()):
        tile_pcd = pcd.select_by_index(point_indices)
        filename = f"tile_{i:04d}_{j:04d}{ext}"
        tile_path = str(out / filename)
        o3d.io.write_point_cloud(tile_path, tile_pcd)
        tile_info.append({
            "file": filename,
            "grid": [i, j],
            "points": len(point_indices),
        })
        logger.debug("  %s: %d pts", filename, len(point_indices))

    return {
        "input": input_path,
        "output_dir": output_dir,
        "total_points": total,
        "grid_size": grid_size,
        "axis": axis,
        "num_tiles": len(tile_info),
        "tiles": tile_info,
    }
