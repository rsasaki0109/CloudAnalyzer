"""Standard ROS2 schemas for the NCLT demo's hand-serialized CDR messages."""
from rosbags.typesys import Stores, get_typestore

TYPES = ('sensor_msgs/msg/PointCloud2', 'sensor_msgs/msg/Imu')


def schemas() -> dict[str, bytes]:
    store = get_typestore(Stores.ROS2_HUMBLE)
    return {name: store.generate_msgdef(name, ros_version=2)[0].encode('utf-8') for name in TYPES}
