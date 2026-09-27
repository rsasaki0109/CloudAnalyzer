from typing import TypedDict

import numpy as np
import numpy.typing as npt

__version__: str

class PointData(TypedDict, total=False):
    positions: npt.NDArray[np.float64]
    colors: npt.NDArray[np.uint8]
    intensity: npt.NDArray[np.float32]
    classification: npt.NDArray[np.uint8]

class IcpResult(TypedDict):
    transformation: npt.NDArray[np.float64]
    rms_initial: float
    rms_final: float
    iterations: int
    converged: bool

def read(path: str, keep_every: int = 1) -> PointData: ...
def read_mesh(path: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]] | None: ...
def nearest_distances(
    source: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def cloud_to_mesh(
    points_: npt.NDArray[np.float64],
    vertices: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.uint32],
    signed: bool = True,
) -> npt.NDArray[np.float64]: ...
def icp(
    moving: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
    max_iterations: int = 50,
    overlap: float = 1.0,
    match_centroids: bool = False,
    point_to_plane: bool = True,
) -> IcpResult: ...
def voxel_subsample(points_: npt.NDArray[np.float64], voxel: float) -> npt.NDArray[np.int64]: ...
def statistical_outliers(
    points_: npt.NDArray[np.float64], k: int = 8, ratio: float = 1.0
) -> npt.NDArray[np.int64]: ...

Surface = float | npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]

class VolumeResult(TypedDict):
    added: float
    removed: float
    net: float
    added_area: float
    removed_area: float
    matched_cells: int
    total_cells: int
    cell: float
    grid_min: tuple[float, float]
    difference: npt.NDArray[np.float64]

def volume(
    before: Surface, after: Surface, cell: float, height: str = "mean", fill_empty: bool = False
) -> VolumeResult: ...
def ground_csf(
    points_: npt.NDArray[np.float64],
    cloth_resolution: float = 1.0,
    class_threshold: float = 0.5,
    rigidness: str = "relief",
    max_iterations: int = 500,
) -> npt.NDArray[np.bool_]: ...
