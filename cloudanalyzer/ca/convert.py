"""Point cloud format conversion module."""

import open3d as o3d

from ca.io import load_point_cloud, save_point_cloud, SUPPORTED_EXTENSIONS


def convert(input_path: str, output_path: str) -> dict:
    """Convert a point cloud file to another format.

    Args:
        input_path: Input point cloud file path.
        output_path: Output file path (format determined by extension).

    Returns:
        Dict with input/output info.

    Raises:
        ValueError: If output format is not supported.
    """
    from pathlib import Path

    out_ext = Path(output_path).suffix.lower()
    if out_ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported output format: '{out_ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    pcd = load_point_cloud(input_path)
    num_points = len(pcd.points)

    save_point_cloud(output_path, pcd)

    return {
        "input": input_path,
        "output": output_path,
        "num_points": num_points,
        "input_format": Path(input_path).suffix.lower(),
        "output_format": out_ext,
    }
