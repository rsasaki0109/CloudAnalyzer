from typing import Literal, TypedDict

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
def discover_vector_map_features(cloud: str, vector_map: str | None = None, options: str = "{}", confirmations: str | None = None) -> str: ...
def measure_vector_map_crosswalk(
    cloud: str, vector_map: str, options: str, preview_only: bool = True
) -> str: ...
def measure_vector_map_signal(
    cloud: str, vector_map: str, options: str, preview_only: bool = True,
) -> str: ...
def measure_vector_map_signal_points(
    points: npt.NDArray[np.float64], vector_map: str, options: str,
    preview_only: bool = True,
) -> str: ...
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

class PreparedReference:
    def __init__(self, target: npt.NDArray[np.float64]) -> None: ...
    @property
    def retained_bytes(self) -> int: ...
    def distances(self, source: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class PreparedMesh:
    def __init__(
        self, vertices: npt.NDArray[np.float64], triangles: npt.NDArray[np.uint32]
    ) -> None: ...
    @property
    def retained_bytes(self) -> int: ...
    def distances(
        self, points_: npt.NDArray[np.float64], signed: bool = True
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
    """A PointCloud2 decoded, retaining its sensor header time and frame.
    stamp_ns is exact integer nanoseconds; stamp retains floating header seconds.
    These are distinct from the bag recording time. No frame transform is applied.
    """

    stamp: float
    @property
    def stamp_ns(self) -> int: ...
    @property
    def frame_id(self) -> str: ...
    def positions(self) -> npt.NDArray[np.float64]:
        """The finite points, (N, 3)."""
        ...
    def intensity(self) -> npt.NDArray[np.float32] | None: ...
    def time(self) -> npt.NDArray[np.float32] | None:
        """How far through the scan each point was taken, 0 to 1, when the cloud has a time field."""
        ...
    @property
    def raw_time_field(self) -> str | None: ...
    @property
    def raw_time_datatype(self) -> int | None:
        """ROS PointField datatype code (1..8), when explicitly requested."""
        ...
    def raw_time(self) -> npt.NDArray[np.float64] | None:
        """Copied source values aligned with finite points, including nonfinite values.
        No units/origin inferred. All PointField integers fit exactly in float64.
        None unless raw_time_field was explicitly requested on the iterator.
        """
        ...
    def point_times_ns(
        self, *, unit: Literal["s", "ms", "us", "ns"],
        origin: Literal["header", "absolute"],
    ) -> npt.NDArray[np.int64]:
        """Convert explicitly retained raw time to copied int64 nanoseconds.
        Round the exact binary value times the unit scale to nearest ns, ties to
        even, then add the integer header (header) or zero (absolute).
        Missing raw time, nonfinite values and final int64 overflow are errors.
        Point order is preserved; no clock calibration or unit/origin inference.
        """
        ...
    def __len__(self) -> int: ...

class BagMessages:
    """The messages of some topics of a bag, oldest first: a `Scan` for a PointCloud2,
    else (topic, stamp, encoding, bytes)."""

    def __iter__(self) -> BagMessages: ...
    def __next__(self) -> Scan | tuple[str, float, str, bytes]: ...
    def progress(self) -> float: ...

MeasurementStatus = Literal["missing", "unknown", "known", "invalid"]

class RawImuData(TypedDict):
    timestamps_ns: npt.NDArray[np.int64]
    timestamps_s: npt.NDArray[np.float64]
    frame_ids: list[str]
    orientation_xyzw: npt.NDArray[np.float64]
    orientation_covariance: npt.NDArray[np.float64]
    angular_velocity_rad_s: npt.NDArray[np.float64]
    angular_velocity_covariance: npt.NDArray[np.float64]
    linear_acceleration_m_s2: npt.NDArray[np.float64]
    linear_acceleration_covariance: npt.NDArray[np.float64]
    orientation_status: list[MeasurementStatus]
    angular_velocity_status: list[MeasurementStatus]
    linear_acceleration_status: list[MeasurementStatus]

class PreintegratedImu(TypedDict):
    start_ns: int
    end_ns: int
    delta_rotation: npt.NDArray[np.float64]
    delta_velocity: npt.NDArray[np.float64]
    delta_position: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    bias_jacobian: npt.NDArray[np.float64]
    linearization_bias: npt.NDArray[np.float64]
    delta_time_s: float
    integration_steps: int
    frame_id: str

class ImuPrediction(TypedDict):
    pose: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]

class ImuEvaluation(TypedDict):
    residual: npt.NDArray[np.float64]
    whitened_residual: npt.NDArray[np.float64]
    cost: float

class ImuLinearization(ImuEvaluation):
    jacobian_i: npt.NDArray[np.float64]
    jacobian_j: npt.NDArray[np.float64]
    jacobian_bias: npt.NDArray[np.float64]
    jacobian_gravity: npt.NDArray[np.float64]
    whitened_jacobian_i: npt.NDArray[np.float64]
    whitened_jacobian_j: npt.NDArray[np.float64]
    whitened_jacobian_bias: npt.NDArray[np.float64]
    whitened_jacobian_gravity: npt.NDArray[np.float64]

class ImuInitialization(TypedDict):
    timestamps_ns: npt.NDArray[np.int64]
    imu_poses: npt.NDArray[np.float64]
    velocities: npt.NDArray[np.float64]
    gravity: npt.NDArray[np.float64]
    bias: npt.NDArray[np.float64]
    gyro_iterations: int
    gyro_singular_ratio: float
    gravity_error: float
    rotation_rms: float
    velocity_rms: float
    frame_id: str
    acceleration_scale: float
    accel_bias_fixed: bool

class ImuSeries:
    """Immutable whole-topic snapshot from BagReader.imu_series(); no file access after creation."""

    @property
    def acceleration_scale(self) -> float:
        """Raw-acceleration multiplier applied before SI-unit bias; read-only."""
        ...
    def __len__(self) -> int: ...
    def timestamps_ns(self) -> npt.NDArray[np.int64]: ...
    def frame_ids(self) -> list[str]: ...
    def initialize_lidar(
        self, timestamps_ns: npt.NDArray[np.int64], lidar_poses: npt.NDArray[np.float64], *,
        imu_from_lidar: npt.NDArray[np.float64], gyro_bias: tuple[float, float, float],
        accel_bias: tuple[float, float, float], gravity_magnitude: float,
        max_gravity_error: float, max_rotation_rms: float, max_velocity_rms: float,
        max_gyro_bias_change: float, max_iterations: int = 15,
        max_gap_s: float = 0.05, min_singular_ratio: float = 1e-6, parallel: bool = False,
    ) -> ImuInitialization:
        """Batch metric startup: exact gyro-bias reintegration, fixed acceleration bias,
        gravity/velocities from displacement triples. Raises on failed acceptance limits.
        State is available after the final reference; arrays are independent copies.
        """
        ...
    def trajectory(
        self, start_ns: int, end_ns: int, pose_i: npt.NDArray[np.float64],
        velocity_i: npt.NDArray[np.float64], *, gravity: tuple[float, float, float],
        bias: tuple[float, float, float, float, float, float], max_gap_s: float = 0.05,
    ) -> ImuTrajectory: ...
    def preintegrate(
        self, start_ns: int, end_ns: int, *, gyro_noise_density: float,
        accel_noise_density: float, max_gap_s: float = 0.05,
        bias: tuple[float, float, float, float, float, float] | None = None,
    ) -> PreintegratedImu: ...
    def factor(
        self, start_ns: int, end_ns: int, *, gyro_noise_density: float,
        accel_noise_density: float, max_gap_s: float = 0.05,
        bias: tuple[float, float, float, float, float, float] | None = None,
    ) -> ImuFactor: ...

class ImuTrajectory:
    """Immutable fixed-anchor nominal motion, created by ImuSeries.trajectory().
    Requires declared initial IMU-to-world state, world gravity and IMU biases.
    Uses held measurements, exact integer timestamp differences and no covariance.
    Source IMU frame_id is independent of the explicitly supplied LiDAR frame.
    """
    @property
    def start_ns(self) -> int: ...
    @property
    def end_ns(self) -> int: ...
    @property
    def frame_id(self) -> str: ...
    @property
    def acceleration_scale(self) -> float: ...
    def __len__(self) -> int: ...
    def timestamps_ns(self) -> npt.NDArray[np.int64]: ...
    def state(self, stamp_ns: int) -> ImuPrediction: ...
    def deskew(
        self, points: npt.NDArray[np.float64], point_times_ns: npt.NDArray[np.int64],
        reference_ns: int, *, imu_from_lidar: npt.NDArray[np.float64], parallel: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Copied points in LiDAR coordinates at reference_ns, with original order.
        imu_from_lidar is a proper rigid (4,4) transform including the lever arm.
        All times must be covered. No extrapolation, filtering or sorting.
        parallel=True uses an indexed CPU map; inputs are copied before GIL release.
        Disable ordinary odometry deskew when supplying these compensated points.
        """
        ...

class ImuFactor:
    """Created by BagReader.imu_factor(); owns the measurement and covariance factorization.
    Poses are IMU-to-world (4,4); velocity and gravity use world axes.
    State Jacobian columns: right-local rotation, world velocity, world position.
    Bias order: gyro, accelerometer; None uses the saved linearization point.
    """

    def measurement(self) -> PreintegratedImu: ...
    def predict(
        self, pose_i: npt.NDArray[np.float64], velocity_i: npt.NDArray[np.float64], *,
        gravity: tuple[float, float, float],
        bias: tuple[float, float, float, float, float, float] | None = None,
    ) -> ImuPrediction: ...
    def evaluate(
        self, pose_i: npt.NDArray[np.float64], velocity_i: npt.NDArray[np.float64],
        pose_j: npt.NDArray[np.float64], velocity_j: npt.NDArray[np.float64], *,
        gravity: tuple[float, float, float],
        bias: tuple[float, float, float, float, float, float] | None = None,
    ) -> ImuEvaluation: ...
    def linearize(
        self, pose_i: npt.NDArray[np.float64], velocity_i: npt.NDArray[np.float64],
        pose_j: npt.NDArray[np.float64], velocity_j: npt.NDArray[np.float64], *,
        gravity: tuple[float, float, float],
        bias: tuple[float, float, float, float, float, float] | None = None,
    ) -> ImuLinearization: ...

class ImuWindowStates(TypedDict):
    timestamps_ns: npt.NDArray[np.int64]
    poses: npt.NDArray[np.float64]
    velocities: npt.NDArray[np.float64]
    biases: npt.NDArray[np.float64]

class ImuWindowResult(ImuWindowStates):
    initial_cost: float
    final_cost: float
    iterations: int
    accepted_steps: int
    rejected_steps: int
    converged: bool
    termination: Literal["gradient", "step", "cost", "max_iterations", "no_descent"]
    cost_history: npt.NDArray[np.float64]
    max_abs_bias_correction: npt.NDArray[np.float64]

class ImuMarginalPrior(TypedDict):
    timestamp_ns: int
    reference_pose: npt.NDArray[np.float64]
    reference_velocity: npt.NDArray[np.float64]
    reference_bias: npt.NDArray[np.float64]
    sqrt_information: npt.NDArray[np.float64]
    residual_offset: npt.NDArray[np.float64]

class ImuMarginalizeResult(TypedDict):
    dropped_timestamp_ns: int
    retained_timestamp_ns: int
    removed_priors: int
    replaced_marginal_prior: bool
    dropped_linearized_constant: float

class ImuWindow:
    """Bounded IMU chain (2..32 states) with explicit oldest-state marginalization.
    All poses refer to the IMU origin. Biases are gyro then accelerometer.
    Every adjacent edge and at least one full state prior are required.
    """

    def __init__(
        self, timestamps_ns: list[int] | npt.NDArray[np.int64],
        poses: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64],
        biases: npt.NDArray[np.float64], *, gravity: tuple[float, float, float],
    ) -> None: ...
    def add_imu_factor(
        self, index: int, factor: ImuFactor, *,
        gyro_bias_random_walk: float, accel_bias_random_walk: float,
    ) -> None: ...
    def append_state(
        self, timestamp_ns: int, pose: npt.NDArray[np.float64],
        velocity: npt.NDArray[np.float64], bias: npt.NDArray[np.float64], factor: ImuFactor, *,
        gyro_bias_random_walk: float, accel_bias_random_walk: float,
    ) -> None: ...
    def marginalize_oldest(self) -> ImuMarginalizeResult: ...
    def marginal_prior(self) -> ImuMarginalPrior | None: ...
    def add_state_prior(
        self, index: int, pose: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64], *, sigmas: npt.NDArray[np.float64],
    ) -> None: ...
    def add_pose_prior(
        self, index: int, pose: npt.NDArray[np.float64], *,
        rotation_sigma: float, position_sigma: float,
    ) -> None: ...
    def states(self) -> ImuWindowStates: ...
    def optimize(
        self, *, max_iterations: int = 20, initial_damping: float = 1e-3,
        gradient_tolerance: float = 1e-8, step_tolerance: float = 1e-9,
        cost_tolerance: float = 1e-10,
    ) -> ImuWindowResult: ...

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
    def messages(self, topics: list[str], *, raw_time_field: str | None = None) -> BagMessages:
        """Within ROS 1/MCAP chunks, equal recording times retain physical record order.
        Sensor headers are not sorted or changed.
        """
        ...
    def scans(self, topic: str | None = None, *, raw_time_field: str | None = None) -> BagMessages:
        """The PointCloud2 messages of a topic (the one such topic when None), decoded as Scans."""
        ...
    def imu(
        self, topic: str | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per Imu message: stamps (N,), up directions in the IMU's frame (N, 3; NaN without an
        orientation) and linear accelerations (N, 3)."""
        ...
    def imu_raw(self, topic: str | None = None) -> RawImuData:
        """Raw fields and statuses in header time, with (N,3,3) covariances.
        Zero covariance is unknown. timestamps_ns retains exact integer time.
        """
        ...
    def imu_series(self, topic: str | None = None, *, acceleration_scale: float = 1.0) -> ImuSeries:
        """Decode and validate ordering once; retain raw fields and the configured scale.
        Positive finite acceleration_scale converts acceleration to m/s^2 during
        integration, before SI bias subtraction. Noise densities also use SI units.
        The complete topic stays in memory. No unit inference or frame conversion.
        """
        ...
    def preintegrate_imu(
        self, start_ns: int, end_ns: int, *, gyro_noise_density: float,
        accel_noise_density: float, max_gap_s: float = 0.05,
        bias: tuple[float, float, float, float, float, float] | None = None,
        topic: str | None = None,
        acceleration_scale: float = 1.0,
    ) -> PreintegratedImu:
        """Held-measurement preintegration in the initial IMU frame, excluding gravity.
        Density units: rad/s/sqrt(Hz), m/s^2/sqrt(Hz). Bias order: gyro, accel.
        Covariance/Jacobian rows: right-local rotation, additive velocity, position.
        Missing/invalid measurements, time gaps and frame changes raise ValueError.
        Positive finite acceleration_scale is applied before SI-unit bias subtraction.
        Raw reported covariances determine availability; explicit SI densities define noise.
        """
        ...
    def imu_factor(
        self, start_ns: int, end_ns: int, *, gyro_noise_density: float,
        accel_noise_density: float, max_gap_s: float = 0.05,
        bias: tuple[float, float, float, float, float, float] | None = None,
        topic: str | None = None,
        acceleration_scale: float = 1.0,
    ) -> ImuFactor:
        """A reusable inertial factor; requires numerically positive definite covariance."""
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

class OdometryCorrespondences(TypedDict):
    matched_points: int
    matched_fraction: float
    rmse_m: float | None
    max_residual_m: float | None

class OdometryReport(TypedDict):
    termination: Literal[
        "empty_map", "insufficient_points", "converged", "iteration_limit",
        "singular_system", "non_finite_system", "non_finite_update", "unfinished",
    ]
    iterations: int
    input_points: int
    registration_points: int
    threshold_m: float
    correspondences: OdometryCorrespondences | None

class ImuLidarOdometry:
    """IMU deskew then ICP, with poses/map points in the first reference LiDAR frame."""

    def __init__(
        self, trajectory: ImuTrajectory, imu_from_lidar: npt.NDArray[np.float64], *,
        min_range: float = 1.5, max_range: float = 80.0,
        map_voxel: float = 1.0, map_points: int = 20,
        parallel: bool = False, diagnostics: bool = False,
    ) -> None: ...
    def register(
        self, points: npt.NDArray[np.float64], point_times_ns: npt.NDArray[np.int64],
        reference_ns: int,
    ) -> npt.NDArray[np.float64]:
        """Compensate/register at a strictly increasing explicit reference timestamp."""
        ...
    @property
    def world_from_first_lidar(self) -> npt.NDArray[np.float64] | None:
        """Copied local-to-nominal-world alignment; None before the first scan."""
        ...
    def reference_timestamps_ns(self) -> npt.NDArray[np.int64]: ...
    def poses(self) -> npt.NDArray[np.float64]: ...
    def placed(self) -> npt.NDArray[np.float64]:
        """Copied last-scan map points after compensation, filtering and thinning."""
        ...
    def last_prediction(self) -> npt.NDArray[np.float64] | None: ...
    def last_iterations(self) -> int: ...
    def last_report(self) -> OdometryReport | None: ...

class LidarOdometry:
    """LiDAR odometry: scans registered one after another onto a local map of the ones before."""

    def __init__(
        self,
        min_range: float = 1.5,
        max_range: float = 80.0,
        map_voxel: float = 1.0,
        map_points: int = 20,
        deskew: bool = False,
        *, diagnostics: bool = False,
    ) -> None: ...
    def register(
        self, positions: npt.NDArray[np.float64], times: npt.NDArray[np.float32] | None = None,
        *, relative_prediction: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Register a scan (N,3); optional 4x4 prediction maps it into the previous scan frame.
        The first scan anchors identity. Predictions affect ICP initialization, not deskew.
        """
        ...
    def last_iterations(self) -> int:
        """Gauss-Newton steps of the last registration, zero without alignment."""
        ...
    def last_report(self) -> OdometryReport | None:
        """Termination/counts of the last scan, None before the first. With diagnostics=True,
        includes final-pose residuals against the map before insertion; no quality gate.
        """
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
