from typing import TypedDict

import numpy as np
import numpy.typing as npt

__version__: str

class CopcSpatialQuery:
    def __init__(
        self, head: bytes, file_size: int, bounds: tuple[float, float, float, float, float, float],
        page_bytes: int = 1048576, compressed_node_bytes: int = 16777216,
        raw_node_bytes: int = 33554432, pending_entries: int = 16384, page_depth: int = 32,
    ) -> None: ...
    @property
    def total_points(self) -> int: ...
    @property
    def record_length(self) -> int: ...
    def next_item(self) -> tuple[str, int, int, int] | None: ...
    def supply_page(self, offset: int, page: bytes) -> None: ...
    def decode_records(self, offset: int, compressed: bytes) -> bytes: ...
    def advance_node(self) -> None: ...

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
def m3c2(
    core: npt.NDArray[np.float64],
    cloud1: npt.NDArray[np.float64],
    cloud2: npt.NDArray[np.float64],
    normal_radius: float = 1.0,
    projection_radius: float = 0.5,
    max_depth: float = 2.0,
    min_points: int = 5,
    registration_error: float = 0.0,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float64],
]: ...
def profile(
    points_: npt.NDArray[np.float64],
    line: npt.NDArray[np.float64],
    half_width: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: ...
def normals(
    points_: npt.NDArray[np.float64],
    k: int = 12,
    orientation: str = "up",
) -> npt.NDArray[np.float32]: ...

def calibrate_ups(
    rotations: npt.NDArray[np.float64], ups: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def changed_objects(
    positions: npt.NDArray[np.float64],
    change: npt.NDArray[np.float64],
    significant: npt.NDArray[np.bool_],
    min_change: float = 0.3,
    link: float = 1.0,
    min_points: int = 8,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ...

class CopcReader:
    @staticmethod
    def is_copc(head: bytes) -> bool: ...
    @staticmethod
    def header_length(head: bytes) -> int | None: ...
    def __init__(self, head: bytes) -> None: ...
    @property
    def total_points(self) -> int: ...
    def pages_for(self, level: int) -> list[tuple[int, int]]: ...
    def add_page(self, offset: int, page: bytes) -> None: ...
    def level_points(self, level: int) -> int: ...
    def deeper_than(self, level: int) -> bool: ...
    def nodes_to(self, level: int) -> list[tuple[int, int, int]]: ...
    def decode(self, chunks: list[bytes], counts: list[int]) -> PointData: ...

class Scan:
    """A PointCloud2 decoded: its stamp, points, intensity and per-point time within the scan."""

    stamp: float
    def positions(self) -> npt.NDArray[np.float64]:
        """The finite points, (N, 3)."""
        ...
    def intensity(self) -> npt.NDArray[np.float32] | None: ...
    def time(self) -> npt.NDArray[np.float32] | None:
        """How far through the scan each point was taken, 0 to 1, when the cloud has a time field."""
        ...
    def __len__(self) -> int: ...

class BagMessages:
    """The messages of some topics of a bag, oldest first: a `Scan` for a PointCloud2,
    else (topic, stamp, encoding, bytes)."""

    def __iter__(self) -> BagMessages: ...
    def __next__(self) -> Scan | tuple[str, float, str, bytes]: ...
    def progress(self) -> float: ...

class BagReader:
    """A ROS 1 bag (.bag), an MCAP file, a rosbag2 .db3 file or a rosbag2 folder, read without a
    ROS install (lz4, zstd and bz2 chunks)."""

    def __init__(self, path: str) -> None: ...
    def topics(self) -> list[tuple[str, str, int]]:
        """Every topic: (name, type as ROS 1 names it, message count)."""
        ...
    def time_range(self) -> tuple[float, float] | None:
        """When the first and the last message were recorded (seconds), from the bag's index."""
        ...
    def messages(self, topics: list[str]) -> BagMessages: ...
    def scans(self, topic: str | None = None) -> BagMessages:
        """The PointCloud2 messages of a topic (the one such topic when None), decoded as Scans."""
        ...
    def imu(
        self, topic: str | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per Imu message: stamps (N,), up directions in the IMU's frame (N, 3; NaN without an
        orientation) and linear accelerations (N, 3)."""
        ...
    def poses(
        self, topic: str | None = None, frame: str | None = None
    ) -> tuple[
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]], str, str
    ]:
        """The poses of a trajectory topic (nav_msgs/Odometry, geometry_msgs/PoseStamped, or
        tf2_msgs/TFMessage with the child `frame`): (stamps (N,), positions (N, 3), orientations
        (N, 4)), the topic and its type."""
        ...

class LidarOdometry:
    """LiDAR odometry: scans registered one after another onto a local map of the ones before."""

    def __init__(
        self,
        min_range: float = 1.5,
        max_range: float = 80.0,
        map_voxel: float = 1.0,
        map_points: int = 20,
        deskew: bool = False,
    ) -> None: ...
    def register(
        self, positions: npt.NDArray[np.float64], times: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float64]:
        """Register the next scan (N, 3), in its sensor's frame: its pose, 4x4."""
        ...
    def poses(self) -> npt.NDArray[np.float64]:
        """Every pose so far, (K, 4, 4)."""
        ...

class PoseGraph:
    """A pose graph with one scan per node: loops by ICP, IMU gravity, dynamic points, the map."""

    @staticmethod
    def from_poses(
        poses: npt.NDArray[np.float64],
        sigma_t: float = 0.05,
        sigma_r_deg: float = 0.25,
        ids: list[int] | None = None,
    ) -> PoseGraph: ...
    @staticmethod
    def from_g2o(text: str) -> PoseGraph: ...
    @property
    def node_count(self) -> int: ...
    @property
    def node_ids(self) -> list[int]: ...
    @property
    def loop_count(self) -> int: ...
    @property
    def edge_count(self) -> int: ...
    def set_scan(
        self,
        index: int,
        positions: npt.NDArray[np.float64],
        intensity: npt.NDArray[np.float32] | None = None,
        voxel: float = 0.4,
    ) -> int: ...
    def find_loops(
        self,
        max_distance: float = 10.0,
        drift: float = 0.03,
        min_travel: float = 30.0,
        spacing: float = 5.0,
        min_fitness: float = 0.5,
        inlier_distance: float = 0.5,
        retry_headings: int = 8,
        max_iterations: int = 50,
        overlap: float = 0.8,
        sigma_t: float = 0.1,
        sigma_r_deg: float = 1.0,
    ) -> dict: ...
    def set_gravity(
        self, nodes: list[int], ups: npt.NDArray[np.float64], sigma_deg: float = 0.1
    ) -> int: ...
    def optimize(self, loop_kernel: float = 0.0) -> dict: ...
    def poses(self, initial: bool = False) -> npt.NDArray[np.float64]: ...
    def to_g2o(self) -> str: ...
    def to_kitti(self) -> str: ...
    def to_tum(self, timestamps: list[float]) -> str: ...
    def detect_dynamic(
        self, window: int = 10, margin: float = 0.5, votes: int = 3
    ) -> tuple[int, int]: ...
    def join(
        self,
        other: PoseGraph,
        here: int,
        there: int,
        yaw_steps: int = 8,
        max_iterations: int = 50,
        overlap: float = 0.8,
        inlier_distance: float = 0.5,
        min_fitness: float = 0.5,
        sigma_t: float = 0.1,
        sigma_r_deg: float = 1.0,
    ) -> dict: ...
    def map(
        self,
        voxel: float = 0.0,
        part: str = "all",
        initial: bool = False,
        correction: bool = False,
        nodes: list[int] | None = None,
    ) -> dict: ...
