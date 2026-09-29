"""Up directions from a bag's sensor_msgs/Imu, and rosbag2 folders recognised as bags."""

import math

import numpy as np
import pytest

from ca.core.bag_ingest import imu_ups, is_bag_path

rosbags = pytest.importorskip("rosbags")


def _bag(path, *, oriented: bool):
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Quaternion = ts.types["geometry_msgs/msg/Quaternion"]
    Vector3 = ts.types["geometry_msgs/msg/Vector3"]
    # Pitched 0.2 rad about y: seen in the IMU frame, up leans towards -x.
    pitch = 0.2
    with Writer(path) as w:
        conn = w.add_connection("/imu", Imu.__msgtype__, typestore=ts)
        for k in range(50):
            t = 100.0 + 0.02 * k
            q = Quaternion(x=0.0, y=math.sin(pitch / 2), z=0.0, w=math.cos(pitch / 2))
            covariance = np.zeros(9) if oriented else np.array([-1.0] + [0.0] * 8)
            gravity_in_imu = 9.81 * np.array([-math.sin(pitch), 0.0, math.cos(pitch)])
            m = Imu(
                header=Header(seq=k, stamp=Time(sec=int(t), nanosec=int(round((t % 1) * 1e9))), frame_id="imu"),
                orientation=q if oriented else Quaternion(x=0.0, y=0.0, z=0.0, w=0.0),
                orientation_covariance=covariance,
                angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=Vector3(x=gravity_in_imu[0], y=0.0, z=gravity_in_imu[2]),
                linear_acceleration_covariance=np.zeros(9),
            )
            w.write(conn, int(t * 1e9), ts.serialize_ros1(m, Imu.__msgtype__))
    return np.array([-math.sin(pitch), 0.0, math.cos(pitch)])


@pytest.mark.parametrize("oriented", [True, False])
def test_up_per_frame_from_orientation_or_acceleration(tmp_path, oriented):
    path = tmp_path / "imu.bag"
    expected = _bag(path, oriented=oriented)
    ups = imu_ups(path, [100.1, 100.5, 105.0])
    # The third frame is seconds after the last IMU message: no up for it.
    assert sorted(ups) == [0, 1]
    for up in ups.values():
        assert np.allclose(up, expected, atol=1e-6)


def test_a_rosbag2_folder_is_a_bag(tmp_path):
    (tmp_path / "run").mkdir()
    assert not is_bag_path(tmp_path / "run")
    (tmp_path / "run" / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    assert is_bag_path(tmp_path / "run")
    assert is_bag_path(tmp_path / "x.mcap")


def test_scans_come_out_as_kitti_bin_with_their_intensity(tmp_path):
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    from ca.core.bag_ingest import materialize_pointcloud_bag

    ts = get_typestore(Stores.ROS1_NOETIC)
    PointCloud2 = ts.types["sensor_msgs/msg/PointCloud2"]
    PointField = ts.types["sensor_msgs/msg/PointField"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    fields = [PointField(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate(["x", "y", "z", "intensity"])]
    # The second point is not finite: it goes, and the intensities stay with their points.
    rows = np.array([[1, 2, 3, 0.5], [np.nan, 0, 0, 0.9], [4, 5, 6, 0.25]], dtype=np.float32)
    path = tmp_path / "scans.bag"
    with Writer(path) as w:
        conn = w.add_connection("/points", PointCloud2.__msgtype__, typestore=ts)
        for k in range(2):
            msg = PointCloud2(
                header=Header(seq=k, stamp=Time(sec=10 + k, nanosec=0), frame_id="lidar"),
                height=1, width=3, fields=fields, is_bigendian=False, point_step=16, row_step=48,
                data=np.frombuffer(rows.tobytes(), dtype=np.uint8), is_dense=False,
            )
            w.write(conn, (10 + k) * 10**9, ts.serialize_ros1(msg, PointCloud2.__msgtype__))
    frames, stamps = materialize_pointcloud_bag(path, tmp_path / "out", kitti_bin=True)
    assert [f.name for f in frames] == ["frame_000000.bin", "frame_000001.bin"] and stamps == (10.0, 11.0)
    points = np.fromfile(frames[0], dtype=np.float32).reshape(-1, 4)
    assert np.allclose(points, [[1, 2, 3, 0.5], [4, 5, 6, 0.25]])


def test_raw_scans_need_odometry_first(tmp_path):
    from ca.posegraph_fix import needs_odometry

    scans = tmp_path / "scans"
    scans.mkdir()
    (scans / "000000.bin").write_bytes(b"")
    assert needs_odometry(str(scans), None)
    assert not needs_odometry(str(scans), str(tmp_path / "trajectory.tum"))
    (scans / "poses.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")
    assert not needs_odometry(str(scans), None)
    assert needs_odometry(str(tmp_path / "drive.mcap"), None)
