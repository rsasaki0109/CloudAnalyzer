"""Fast point cloud I/O, distances, ICP and filters from the CloudAnalyzer Rust core.

Points are ``(N, 3)`` float64 NumPy arrays. Heavy functions release the GIL
and use all cores.
"""

from ._core import (
    BagMessages,
    BagReader,
    LidarOdometry,
    PoseGraph,
    Scan,
    __version__,
    build_vector_map,
    calibrate_ups,
    changed_objects,
    cloud_to_mesh,
    connect_vector_map_junctions,
    ground_csf,
    icp,
    m3c2,
    measure_vector_map_signal,
    nearest_distances,
    normals,
    profile,
    read,
    read_mesh,
    statistical_outliers,
    volume,
    voxel_subsample,
)
from ._copc import read_copc

__all__ = [
    "BagMessages",
    "BagReader",
    "LidarOdometry",
    "PoseGraph",
    "Scan",
    "__version__",
    "build_vector_map",
    "calibrate_ups",
    "changed_objects",
    "cloud_to_mesh",
    "connect_vector_map_junctions",
    "ground_csf",
    "icp",
    "m3c2",
    "measure_vector_map_signal",
    "nearest_distances",
    "normals",
    "profile",
    "read",
    "read_copc",
    "read_mesh",
    "statistical_outliers",
    "volume",
    "voxel_subsample",
]
