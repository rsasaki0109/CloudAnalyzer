"""Pack part of an NCLT session as a small ROS 2 bag (MCAP) for the web app's demo: raw LiDAR scans and an
IMU without poses, which the app turns into a pose graph with odometry.

The University of Michigan North Campus Long-Term dataset (NCLT, Carlevaris-Bianco, Ushani and Eustice,
IJRR 2016, http://robots.engin.umich.edu/nclt/) is made available under the Open Database License, its
contents under the Database Contents License; the bag is a derived database under the same terms.

    python scripts/prepare_nclt.py 2012-04-29 nclt/2012-04-29      # the scans, sensors and keyframe times
    python scripts/make_nclt_bag.py nclt/2012-04-29 2520 2740 web/public/samples/nclt-2012-04-29.mcap
    python scripts/make_nclt_bag.py nclt/2012-06-15 2476 2700 web/public/samples/nclt-2012-06-15.mcap
    # pip install numpy mcap zstandard

takes the drive from keyframe ``first`` to keyframe ``end`` (``times.txt``): every ``--every``-th Velodyne
scan (5 Hz in ``velodyne_sync``) on ``/velodyne_points`` (sensor_msgs/PointCloud2), in the frame the
keyframes use (z up), thinned to one point per ``--voxel`` metres, with intensity as a byte; and the MS25
IMU's roll and pitch at 10 Hz on ``/imu`` (sensor_msgs/Imu) as an orientation in the scans' frame (its yaw
left out), compressed with zstd. ``--bins DIR`` also writes the scans as KITTI ``.bin`` files.
"""
import argparse
import os
import struct
import tarfile

import mcap.writer
import numpy as np
import zstandard
from mcap.writer import CompressionType, Writer

# The mcap writer compresses at zstd's default level; a demo downloaded by every visitor is worth level 19.
mcap.writer.zstandard.compress = lambda data: zstandard.ZstdCompressor(level=19).compress(data)  # type: ignore

# As in prepare_nclt.py: the Velodyne in the Segway's body frame, and z down to z up.
BODY_VEL = [0.002, -0.004, -0.957, 0.807, 0.166, 0.0]
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

parser = argparse.ArgumentParser()
parser.add_argument("session", help="a folder prepare_nclt.py wrote")
parser.add_argument("first", type=int, help="the keyframe the drive starts at")
parser.add_argument("end", type=int, help="the keyframe it ends at")
parser.add_argument("out")
parser.add_argument("--every", type=int, default=4)
parser.add_argument("--voxel", type=float, default=0.8)
parser.add_argument("--bins")
args = parser.parse_args()


def ssc_to_matrix(x, y, z, roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    m[:3, 3] = [x, y, z]
    return m


def read_scan(path):
    """velodyne_sync points: x y z as uint16 (5 mm steps, offset -100 m), intensity, laser."""
    raw = np.fromfile(path, dtype=np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("i", "u1"), ("l", "u1")]))
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1) * 0.005 - 100.0
    keep = np.linalg.norm(xyz, axis=1) > 1.5  # the Segway and its rider
    xyz = xyz[keep] * np.array([1.0, -1.0, -1.0])  # z up
    return np.hstack([xyz, raw["i"][keep, None] / 255.0]).astype(np.float32)


class Cdr:
    """ROS 2 serialisation (little-endian CDR), just what PointCloud2 and Imu need."""

    def __init__(self):
        self.b = bytearray(b"\x00\x01\x00\x00")

    def align(self, n):
        self.b += bytes(-(len(self.b) - 4) % n)

    def u8(self, v):
        self.b.append(v)
        return self

    def u32(self, v):
        self.align(4)
        self.b += struct.pack("<I", v)
        return self

    def f64(self, *values):
        for v in values:
            self.align(8)
            self.b += struct.pack("<d", v)
        return self

    def string(self, s):
        self.u32(len(s) + 1)
        self.b += s.encode() + b"\x00"
        return self

    def header(self, ns, frame):
        return self.u32(ns // 1_000_000_000).u32(ns % 1_000_000_000).string(frame)


def point_cloud(points, ns):
    data = np.zeros(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "u1")])
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["intensity"] = np.clip(np.round(points[:, 3] * 255), 0, 255)
    m = Cdr().header(ns, "velodyne").u32(1).u32(len(points)).u32(4)
    for k, (name, datatype) in enumerate((("x", 7), ("y", 7), ("z", 7), ("intensity", 2))):
        m.string(name).u32(4 * k).u8(datatype).u32(1)
    m.u8(0).u32(13).u32(13 * len(points)).u32(13 * len(points))
    m.b += data.tobytes()
    return bytes(m.u8(1).b)


def level(up):
    """The quaternion (x, y, z, w) of the smallest turn taking ``up`` to +z: an orientation whose
    matrix's last row, world up seen in the sensor's frame, is ``up``."""
    up = up / np.linalg.norm(up)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    s = np.linalg.norm(axis)
    angle = np.arctan2(s, up[2])
    axis = axis / s if s > 1e-12 else np.array([1.0, 0.0, 0.0])
    return (*(axis * np.sin(angle / 2)), np.cos(angle / 2))


def imu(up, ns):
    m = Cdr().header(ns, "velodyne").f64(*level(up))
    m.f64(0.0005, *[0.0] * 8)  # orientation covariance (radians²): the orientation is given
    m.f64(*[0.0] * 3, -1.0, *[0.0] * 8)  # no angular velocity
    m.f64(*(9.81 * up), -1.0, *[0.0] * 8)
    return bytes(m.b)


times = np.loadtxt(f"{args.session}/times.txt")
start, stop = times[args.first] * 1e6, times[args.end] * 1e6
raw = f"{args.session}/velodyne_sync"
stamps = sorted(s for s in (int(f[:-4]) for f in os.listdir(raw) if f.endswith(".bin")) if start <= s <= stop)
stamps = stamps[:: args.every]
date = os.path.basename(os.path.normpath(args.session))
with tarfile.open(f"{args.session}/{date}_sen.tar.gz", "r:gz") as tar:
    member = next(m for m in tar if m.name.endswith("ms25_euler.csv"))
    euler = np.loadtxt(tar.extractfile(member), delimiter=",")
euler = euler[(euler[:, 0] >= start) & (euler[:, 0] <= stop)]
body_vel = ssc_to_matrix(*BODY_VEL[:3], *np.radians(BODY_VEL[3:]))[:3, :3]
if args.bins:
    os.makedirs(args.bins, exist_ok=True)

messages = []
last = -np.inf
for stamp, roll, pitch in euler[:, :3]:
    if stamp - last < 100_000:  # 10 Hz
        continue
    last = stamp
    up = FLIP[:3, :3] @ body_vel.T @ ssc_to_matrix(0, 0, 0, roll, pitch, 0)[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    messages.append((int(stamp) * 1000, "imu", up))
for stamp in stamps:
    messages.append((stamp * 1000, "scan", None))
messages.sort(key=lambda m: m[0])

total = 0
with open(args.out, "wb") as f:
    w = Writer(f, compression=CompressionType.ZSTD, chunk_size=1 << 20)
    w.start(profile="ros2", library="CloudAnalyzer make_nclt_bag.py")
    cloud_schema = w.register_schema(name="sensor_msgs/msg/PointCloud2", encoding="ros2msg", data=b"")
    imu_schema = w.register_schema(name="sensor_msgs/msg/Imu", encoding="ros2msg", data=b"")
    clouds = w.register_channel(topic="/velodyne_points", message_encoding="cdr", schema_id=cloud_schema)
    imus = w.register_channel(topic="/imu", message_encoding="cdr", schema_id=imu_schema)
    for k, (ns, kind, up) in enumerate(messages):
        if kind == "imu":
            w.add_message(imus, log_time=ns, publish_time=ns, sequence=k, data=imu(up, ns))
            continue
        scan = read_scan(f"{raw}/{ns // 1000}.bin")
        _, keep = np.unique(np.floor(scan[:, :3] / args.voxel).astype(np.int64), axis=0, return_index=True)
        scan = scan[np.sort(keep)]
        total += len(scan)
        if args.bins:
            scan.tofile(f"{args.bins}/{ns // 1000}.bin")
        w.add_message(clouds, log_time=ns, publish_time=ns, sequence=k, data=point_cloud(scan, ns))
    w.finish()
print(f"{len(stamps)} scans ({total / len(stamps):.0f} points each), {len(messages) - len(stamps)} IMU messages")
