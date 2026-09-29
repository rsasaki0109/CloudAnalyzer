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
