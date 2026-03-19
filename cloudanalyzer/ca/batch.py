"""Batch processing module."""

from pathlib import Path

from ca.io import SUPPORTED_EXTENSIONS, load_point_cloud
from ca.info import get_info
from ca.log import logger


def batch_info(directory: str, recursive: bool = False) -> list[dict]:
    """Run info on all point cloud files in a directory.

    Args:
        directory: Directory path to scan.
        recursive: If True, scan subdirectories too.

    Returns:
        List of info dicts for each file found.
    """
    dirpath = Path(directory)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in dirpath.glob(pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No point cloud files found in %s", directory)
        return []

    results = []
    for f in files:
        logger.info("Processing: %s", f)
        try:
            info = get_info(str(f))
            results.append(info)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipped %s: %s", f, e)

    return results
