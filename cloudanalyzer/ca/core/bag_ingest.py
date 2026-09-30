"""Stable ROS bag / MCAP ingest contract for CloudAnalyzer.

Adopted from ``ca.experiments.bag_ingest``: materialize PointCloud2 scans for
``ca slam-run``, decode Odometry / PoseStamped / TF trajectories for
``ca traj-evaluate``, and expose topic metadata for ``ca info``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import open3d as o3d

BAG_SUFFIXES = frozenset({".bag", ".mcap", ".db3"})

# The Rust core reads recordings on its own; rosbags is the fallback without it.
ROS_INSTALL_HINT = (
    "ROS bag input needs the Rust core, or rosbags.\n"
    'Install with: pip install "cloudanalyzer[fast]" (or "cloudanalyzer[ros]")'
)

POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"
TF_MESSAGE_TYPE = "tf2_msgs/msg/TFMessage"
DIRECT_TRAJECTORY_TYPES = frozenset(
    {
        "nav_msgs/msg/Odometry",
        "geometry_msgs/msg/PoseStamped",
    }
)
TRAJECTORY_TYPES = DIRECT_TRAJECTORY_TYPES | frozenset({TF_MESSAGE_TYPE})

_DATATYPE_TO_DTYPE = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def is_bag_path(path: str | Path) -> bool:
    """Return True when *path* looks like a ROS bag / MCAP / sqlite recording, or a rosbag2
    recording folder (one holding a ``metadata.yaml``)."""
    p = Path(path)
    return p.suffix.lower() in BAG_SUFFIXES or (p.is_dir() and (p / "metadata.yaml").is_file())


def core_reader(path: str | Path) -> Any | None:
    """The Rust core's reader for a recording (no ROS install needed), when the core is
    installed and the path is a ROS 1 bag, an MCAP file, a rosbag2 ``.db3`` file or a rosbag2
    folder that exists."""
    p = Path(path)
    if not (p.is_dir() and (p / "metadata.yaml").is_file()) and not (
        p.is_file() and p.suffix.lower() in BAG_SUFFIXES
    ):
        return None
    try:
        import cloudanalyzer_core
    except ImportError:
        return None
    return cloudanalyzer_core.BagReader(str(p))


def require_rosbags() -> Any:
    """Import rosbags or raise a user-facing install hint."""
    try:
        from rosbags.highlevel import AnyReader

        return AnyReader
    except ImportError as exc:
        raise ValueError(ROS_INSTALL_HINT) from exc


def _header_timestamp_sec(message: Any) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _connection_rows(reader: Any) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    for connection in reader.connections:
        topics.append(
            {
                "topic": connection.topic,
                "type": connection.msgtype,
                "count": int(connection.msgcount),
            }
        )
    topics.sort(key=lambda row: row["topic"])
    return topics


def _inspect_with_core(core: Any, bag_path: Path, decode_sample: bool) -> dict[str, Any]:
    """:func:`inspect_bag` through the Rust core's reader."""
    topics = [
        {"topic": name, "type": kind.replace("/", "/msg/", 1), "count": int(count)}
        for name, kind, count in core.topics()
    ]
    time_range = core.time_range()
    if time_range is None:
        # A recording without an index: the first and last message, reading it all.
        stamps = [m[1] if isinstance(m, tuple) else m.stamp for m in core.messages([t["topic"] for t in topics])]
        time_range = (min(stamps), max(stamps)) if stamps else (0.0, 0.0)
    start, end = time_range
    decoded: list[str] = []
    if decode_sample:
        for row in topics:
            if row["count"] > 0 and next(iter(core.messages([row["topic"]])), None) is not None:
                decoded.append(row["topic"])
    return {
        "path": str(bag_path),
        "kind": "rosbag",
        "duration_ns": int(round((end - start) * 1e9)),
        "start_time_ns": int(round(start * 1e9)),
        "end_time_ns": int(round(end * 1e9)),
        "message_count": int(sum(row["count"] for row in topics)),
        "topics": topics,
        "decoded_sample_topics": decoded,
    }


def inspect_bag(path: str, *, decode_sample: bool = False) -> dict[str, Any]:
    """Return topic metadata for a ROS bag-like recording."""
    bag_path = Path(path)
    core = core_reader(bag_path)
    if core is None:
        AnyReader = require_rosbags()
    if not bag_path.exists():
        raise FileNotFoundError(path)
    if core is not None:
        return _inspect_with_core(core, bag_path, decode_sample)

    with AnyReader([bag_path]) as reader:
        topics = _connection_rows(reader)
        decoded_topics: list[str] = []
        if decode_sample:
            for connection in reader.connections:
                if connection.msgcount <= 0:
                    continue
                for _connection, _timestamp, _rawdata in reader.messages(
                    connections=[connection]
                ):
                    decoded_topics.append(connection.topic)
                    break

        return {
            "path": str(bag_path),
            "kind": "rosbag",
            "duration_ns": int(reader.duration),
            "start_time_ns": int(reader.start_time),
            "end_time_ns": int(reader.end_time),
            "message_count": int(sum(row["count"] for row in topics)),
            "topics": topics,
            "decoded_sample_topics": decoded_topics,
        }


def _position_from_message(message: Any, msgtype: str) -> list[float]:
    if msgtype == "nav_msgs/msg/Odometry":
        position = message.pose.pose.position
    elif msgtype == "geometry_msgs/msg/PoseStamped":
        position = message.pose.position
    else:
        raise ValueError(f"Unsupported trajectory message type: {msgtype}")
    return [float(position.x), float(position.y), float(position.z)]


def _poses_from_tf_message(message: Any, frame: str) -> list[tuple[float, list[float]]]:
    poses: list[tuple[float, list[float]]] = []
    for transform in message.transforms:
        if transform.child_frame_id != frame:
            continue
        stamp = transform.header.stamp
        timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        translation = transform.transform.translation
        poses.append(
            (
                timestamp,
                [float(translation.x), float(translation.y), float(translation.z)],
            )
        )
    return poses


def _pick_trajectory_connection(
    connections: list[Any],
    topic: str | None,
    *,
    frame: str | None,
) -> Any:
    if topic is not None:
        matches = [conn for conn in connections if conn.topic == topic]
        if not matches:
            available = ", ".join(sorted(conn.topic for conn in connections)) or "(none)"
            raise ValueError(f"Trajectory topic not found: {topic}. Available topics: {available}")
        connection = matches[0]
        if connection.msgtype == TF_MESSAGE_TYPE and not frame:
            raise ValueError(
                f"Topic {topic} carries {TF_MESSAGE_TYPE}; pass --frame with the child frame id"
            )
        if connection.msgtype not in TRAJECTORY_TYPES:
            raise ValueError(
                f"Unsupported trajectory message type on {topic}: {connection.msgtype}"
            )
        return connection

    supported = [conn for conn in connections if conn.msgtype in DIRECT_TRAJECTORY_TYPES]
    if not supported:
        raise ValueError(
            "No supported trajectory topic found in bag. "
            f"Supported types: {', '.join(sorted(DIRECT_TRAJECTORY_TYPES))}. "
            "Use --topic to select a topic."
        )
    if len(supported) > 1:
        candidates = ", ".join(sorted(conn.topic for conn in supported))
        raise ValueError(
            "Multiple supported trajectory topics found in bag; use --topic. "
            f"Candidates: {candidates}"
        )
    return supported[0]


def load_trajectory_from_bag(
    path: str,
    *,
    topic: str | None = None,
    frame: str | None = None,
) -> dict:
    """Load timestamps and XYZ positions from a ROS bag-like recording."""
    bag_path = Path(path)
    core = core_reader(bag_path)
    if core is not None:
        (stamps, xyz, _), read_topic, msgtype = core.poses(topic, frame)
        return _trajectory(path, list(map(float, stamps)), xyz.tolist(), read_topic, msgtype, frame)
    AnyReader = require_rosbags()
    with AnyReader([bag_path]) as reader:
        connection = _pick_trajectory_connection(reader.connections, topic, frame=frame)
        timestamps: list[float] = []
        positions: list[list[float]] = []
        for _connection, _timestamp, rawdata in reader.messages(connections=[connection]):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.msgtype == TF_MESSAGE_TYPE:
                assert frame is not None
                for stamp, position in _poses_from_tf_message(message, frame):
                    timestamps.append(stamp)
                    positions.append(position)
            else:
                timestamps.append(_header_timestamp_sec(message))
                positions.append(_position_from_message(message, connection.msgtype))

    return _trajectory(path, timestamps, positions, connection.topic, connection.msgtype, frame)


def _trajectory(
    path: str,
    timestamps: list[float],
    positions: list[list[float]],
    topic: str,
    msgtype: str,
    frame: str | None,
) -> dict:
    if len(timestamps) < 2:
        raise ValueError("Trajectory bag must contain at least 2 poses on the selected topic")
    timestamp_array = np.asarray(timestamps, dtype=float)
    position_array = np.asarray(positions, dtype=float)
    if np.any(np.diff(timestamp_array) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing")

    result = {
        "path": path,
        "format": "rosbag",
        "topic": topic,
        "message_type": msgtype,
        "timestamps": timestamp_array,
        "positions": position_array,
        "num_poses": int(timestamp_array.size),
    }
    if frame is not None:
        result["frame"] = frame
    return result


IMU_TYPE = "sensor_msgs/msg/Imu"


def _quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def imu_ups(
    path: str | Path,
    frame_times: tuple[float, ...] | list[float],
    *,
    topic: str | None = None,
    window_s: float = 0.5,
) -> dict[int, np.ndarray]:
    """Per frame, the up direction in the IMU's frame at that frame's time, from the
    ``sensor_msgs/msg/Imu`` messages of a bag: from the orientation when the IMU gives one
    (world up seen in its frame), else from the mean acceleration over ``window_s`` either
    side (at rest an accelerometer reads straight up). Empty when the bag has no IMU."""
    core = core_reader(path)
    if core is not None:
        core_stamps, core_ups, core_accels = core.imu(topic)
        if len(core_stamps) == 0:
            return {}
        times = np.asarray(core_stamps, dtype=float)
        order = np.argsort(times)
        return _ups_at(
            times[order],
            [None if np.isnan(u[0]) else u for u in np.asarray(core_ups)[order]],
            np.asarray(core_accels)[order],
            frame_times,
            window_s,
        )
    AnyReader = require_rosbags()
    stamps: list[float] = []
    oriented: list[np.ndarray | None] = []
    accels: list[np.ndarray] = []
    with AnyReader([Path(path)]) as reader:
        imus = [c for c in reader.connections if c.msgtype == IMU_TYPE]
        if topic is not None:
            imus = [c for c in imus if c.topic == topic]
            if not imus:
                raise ValueError(f"Imu topic not found: {topic}")
        if not imus:
            return {}
        if len(imus) > 1:
            names = ", ".join(sorted(c.topic for c in imus))
            raise ValueError(f"Several Imu topics in the bag; pick one: {names}")
        for _connection, _t, raw in reader.messages(connections=imus):
            message = reader.deserialize(raw, IMU_TYPE)
            q = message.orientation
            usable = float(message.orientation_covariance[0]) != -1.0 and abs(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w - 1) < 0.1
            stamps.append(_header_timestamp_sec(message))
            oriented.append(_quaternion_matrix(q.x, q.y, q.z, q.w).T @ np.array([0.0, 0.0, 1.0]) if usable else None)
            a = message.linear_acceleration
            accels.append(np.array([a.x, a.y, a.z], dtype=float))
    if not stamps:
        return {}
    times = np.array(stamps)
    order = np.argsort(times)
    return _ups_at(times[order], [oriented[k] for k in order], np.array(accels)[order], frame_times, window_s)


def _ups_at(
    times: np.ndarray,
    oriented: list[np.ndarray | None],
    acc: np.ndarray,
    frame_times: tuple[float, ...] | list[float],
    window_s: float,
) -> dict[int, np.ndarray]:
    """Per frame time, the up direction from the IMU messages sorted by ``times`` (see :func:`imu_ups`)."""
    ups: dict[int, np.ndarray] = {}
    for index, t in enumerate(frame_times):
        k = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1)) if len(times) > 1 else 0
        if len(times) > 1 and abs(times[k - 1] - t) < abs(times[k] - t):
            k -= 1
        if abs(times[k] - t) > window_s:
            continue
        up = oriented[k]
        if up is None:
            near = np.abs(times - t) <= window_s
            mean = acc[near].mean(axis=0)
            if not np.isfinite(mean).all() or np.linalg.norm(mean) < 1e-6:
                continue
            up = mean
        ups[index] = up / np.linalg.norm(up)
    return ups


def pointcloud2_to_xyz(message: Any) -> np.ndarray:
    """Decode ``sensor_msgs/msg/PointCloud2`` into an ``(N, 3)`` float64 array."""
    field_map = {field.name: field for field in message.fields}
    for axis in ("x", "y", "z"):
        if axis not in field_map:
            raise ValueError(f"PointCloud2 message is missing '{axis}' field")

    point_count = int(message.height) * int(message.width)
    if point_count == 0:
        return np.empty((0, 3), dtype=np.float64)

    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if raw.size < point_count * int(message.point_step):
        raise ValueError("PointCloud2 data buffer is smaller than width * height * point_step")

    coords = np.empty((point_count, 3), dtype=np.float64)
    point_step = int(message.point_step)
    for axis_index, axis in enumerate(("x", "y", "z")):
        field = field_map[axis]
        dtype = _DATATYPE_TO_DTYPE.get(int(field.datatype))
        if dtype is None:
            raise ValueError(f"Unsupported PointCloud2 datatype for '{axis}': {field.datatype}")
        column = cast(
            np.ndarray,
            np.ndarray(
                shape=(point_count,),
                dtype=dtype,
                buffer=raw,
                offset=int(field.offset),
                strides=(point_step,),
            ),
        )
        coords[:, axis_index] = column.astype(np.float64, copy=False)

    finite = np.isfinite(coords).all(axis=1)
    return coords[finite]


def _xyz_with_mask(message: Any) -> tuple[np.ndarray, np.ndarray]:
    """``pointcloud2_to_xyz`` and which of the message's points it kept (the finite ones)."""
    field_map = {field.name: field for field in message.fields}
    count = int(message.height) * int(message.width)
    if count == 0:
        return np.empty((0, 3)), np.zeros(0, dtype=bool)
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    coords = np.empty((count, 3), dtype=np.float64)
    for axis_index, axis in enumerate(("x", "y", "z")):
        if axis not in field_map:
            raise ValueError(f"PointCloud2 message is missing '{axis}' field")
        field = field_map[axis]
        dtype = _DATATYPE_TO_DTYPE.get(int(field.datatype))
        if dtype is None:
            raise ValueError(f"Unsupported PointCloud2 datatype for '{axis}': {field.datatype}")
        coords[:, axis_index] = np.ndarray(
            shape=(count,), dtype=dtype, buffer=raw, offset=int(field.offset), strides=(int(message.point_step),)
        )
    keep: np.ndarray = np.asarray(np.isfinite(coords).all(axis=1))
    kept: np.ndarray = coords[keep]
    return kept, keep


INTENSITY_FIELDS = ("intensity", "reflectivity", "i")


def pointcloud2_intensity(message: Any, keep: np.ndarray) -> np.ndarray:
    """The intensity of a PointCloud2's points (its ``intensity`` or ``reflectivity``
    field) as float32, at the points ``keep`` selects; zeros when it has neither."""
    field_map = {field.name: field for field in message.fields}
    name = next((n for n in INTENSITY_FIELDS if n in field_map), None)
    if name is None:
        return np.zeros(int(keep.sum()), dtype=np.float32)
    field = field_map[name]
    dtype = _DATATYPE_TO_DTYPE.get(int(field.datatype))
    if dtype is None:
        return np.zeros(int(keep.sum()), dtype=np.float32)
    count = int(message.height) * int(message.width)
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    column = cast(
        np.ndarray,
        np.ndarray(shape=(count,), dtype=dtype, buffer=raw, offset=int(field.offset), strides=(int(message.point_step),)),
    )
    values: np.ndarray = column[keep].astype(np.float32)
    return values


def _pick_pointcloud_connection(connections: list[Any], topic: str | None) -> Any:
    pointclouds = [conn for conn in connections if conn.msgtype == POINTCLOUD2_TYPE]
    if topic is not None:
        matches = [conn for conn in pointclouds if conn.topic == topic]
        if not matches:
            available = ", ".join(sorted(conn.topic for conn in connections)) or "(none)"
            raise ValueError(
                f"PointCloud2 topic not found: {topic}. Available topics: {available}"
            )
        return matches[0]
    if not pointclouds:
        raise ValueError(
            "No sensor_msgs/msg/PointCloud2 topic found in bag. "
            "Use --pointcloud-topic to select a topic."
        )
    if len(pointclouds) > 1:
        candidates = ", ".join(sorted(conn.topic for conn in pointclouds))
        raise ValueError(
            "Multiple PointCloud2 topics found in bag; use --pointcloud-topic. "
            f"Candidates: {candidates}"
        )
    return pointclouds[0]


def materialize_pointcloud_bag(
    path: str | Path,
    output_dir: str | Path,
    *,
    topic: str | None = None,
    max_frames: int | None = None,
    kitti_bin: bool = False,
) -> tuple[list[Path], tuple[float, ...]]:
    """Write each PointCloud2 message to ``frame_XXXXXX.pcd`` under *output_dir*, or with
    ``kitti_bin`` to ``frame_XXXXXX.bin`` (float32 x, y, z, intensity: KITTI's Velodyne
    format, which keeps the intensity and writes fast)."""
    bag_path = Path(path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    timestamps: list[float] = []

    core = core_reader(bag_path)
    if core is not None:
        # The Rust core reads the bag: no ROS install, and several times faster.
        for scan in core.scans(topic):
            if max_frames is not None and len(frame_paths) >= max_frames:
                break
            if len(scan) == 0:
                continue
            xyz = scan.positions()
            index = len(frame_paths)
            if kitti_bin:
                frame_path = out_dir / f"frame_{index:06d}.bin"
                intensity = scan.intensity()
                if intensity is None:
                    intensity = np.zeros(len(scan), dtype=np.float32)
                np.hstack([xyz.astype(np.float32), intensity[:, None]]).astype(np.float32).tofile(frame_path)
            else:
                frame_path = out_dir / f"frame_{index:06d}.pcd"
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(xyz)
                if not o3d.io.write_point_cloud(str(frame_path), cloud):
                    raise ValueError(f"Failed to write extracted scan: {frame_path}")
            frame_paths.append(frame_path)
            timestamps.append(scan.stamp)
        if not frame_paths:
            raise ValueError(f"No non-empty PointCloud2 frames extracted from {bag_path}")
        return frame_paths, tuple(timestamps)

    AnyReader = require_rosbags()
    with AnyReader([bag_path]) as reader:
        connection = _pick_pointcloud_connection(reader.connections, topic)
        for _connection, _timestamp, rawdata in reader.messages(connections=[connection]):
            if max_frames is not None and len(frame_paths) >= max_frames:
                break
            message = reader.deserialize(rawdata, connection.msgtype)
            if kitti_bin:
                xyz, keep = _xyz_with_mask(message)
                if xyz.shape[0] == 0:
                    continue
                frame_path = out_dir / f"frame_{len(frame_paths):06d}.bin"
                intensity = pointcloud2_intensity(message, keep)
                np.hstack([xyz.astype(np.float32), intensity[:, None]]).astype(np.float32).tofile(frame_path)
                frame_paths.append(frame_path)
                timestamps.append(_header_timestamp_sec(message))
                continue
            points = pointcloud2_to_xyz(message)
            if points.shape[0] == 0:
                continue
            index = len(frame_paths)
            frame_path = out_dir / f"frame_{index:06d}.pcd"
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            if not o3d.io.write_point_cloud(str(frame_path), cloud):
                raise ValueError(f"Failed to write extracted scan: {frame_path}")
            frame_paths.append(frame_path)
            timestamps.append(_header_timestamp_sec(message))

    if not frame_paths:
        raise ValueError(
            f"No non-empty PointCloud2 frames extracted from {bag_path} on topic "
            f"{connection.topic}"
        )

    return frame_paths, tuple(timestamps)
