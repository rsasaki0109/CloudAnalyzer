"""Fast point cloud I/O, distances, ICP and filters from the CloudAnalyzer Rust core.

Points are ``(N, 3)`` float64 NumPy arrays. Heavy functions release the GIL
and use all cores.
"""

from ._core import (
    __version__,
    cloud_to_mesh,
    ground_csf,
    icp,
    m3c2,
    nearest_distances,
    normals,
    profile,
    read,
    read_mesh,
    statistical_outliers,
    volume,
    voxel_subsample,
)

__all__ = [
    "__version__",
    "cloud_to_mesh",
    "ground_csf",
    "icp",
    "m3c2",
    "nearest_distances",
    "normals",
    "profile",
    "read",
    "read_mesh",
    "statistical_outliers",
    "volume",
    "voxel_subsample",
]
