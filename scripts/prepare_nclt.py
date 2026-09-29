"""Turn an NCLT session into a folder the web app's pose graph panel opens.

The University of Michigan North Campus Long-Term dataset (NCLT, Carlevaris-Bianco, Ushani and Eustice,
IJRR 2016, http://robots.engin.umich.edu/nclt/) is made available under the Open Database License, its
contents under the Database Contents License. Its sessions drive a Segway round the same campus, coming
back to the same places: real loops for the pose graph.

    python scripts/prepare_nclt.py 2013-01-10 nclt/2013-01-10          # pip install numpy kiss-icp

downloads the session's Velodyne tarball and ground truth (2.9 GB for 2013-01-10) unless they are in the
output folder already, and writes

    nclt/2013-01-10/velodyne/000000.bin ...  a keyframe every metre: x y z intensity (float32), z up
    nclt/2013-01-10/velodyne/kiss_poses.txt  KISS-ICP odometry of every scan, at the keyframes (KITTI format)
    nclt/2013-01-10/gt_lidar.txt             the ground truth, in the same frame (KITTI format)
    nclt/2013-01-10/gravity/gravity.txt      the MS25 IMU's up direction per keyframe (for IMU gravity)

The README's loop and correction pictures are of session 2012-04-29 (8.7 GB; `NCLT_DIR=nclt/2012-04-29 npm run
media` in web/). `--gravity-only` as a third argument rewrites just the gravity file. The seasons GIF joins
2012-06-15 with keyframes 200-2199 of 2012-12-01, cut out by

    python scripts/prepare_nclt.py 2012-12-01 nclt/2012-12-01 --slice 200:2200 nclt/2012-12-01-part
"""
import csv
import os
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"
# Velodyne in the Segway's body frame: x y z (m), roll pitch yaw (degrees), from the NCLT devkit, but
# without its -90.7° yaw: the velodyne_sync scans are already turned by it (with it, the ground truth
# path comes out a quarter turn off the scans' odometry; without, the two agree).
BODY_VEL = [0.002, -0.004, -0.957, 0.807, 0.166, 0.0]
# NCLT's frames have z down; the app's have z up (a half turn about x).
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
KEYFRAME_SPACING = 1.0

date, out = sys.argv[1], sys.argv[2]
os.makedirs(f"{out}/velodyne", exist_ok=True)


def fetch(url, path, parts=16):
    """Download with parallel range requests (S3 is slow per connection)."""
    if os.path.exists(path):
        return
    size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    chunk = (size + parts - 1) // parts

    def get(k):
        lo, hi = k * chunk, min(size, (k + 1) * chunk) - 1
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
        with urllib.request.urlopen(req) as r, open(f"{path}.part{k}", "wb") as f:
            while b := r.read(1 << 20):
                f.write(b)

    with ThreadPoolExecutor(parts) as pool:
        list(pool.map(get, range(parts)))
    with open(path, "wb") as f:
        for k in range(parts):
            with open(f"{path}.part{k}", "rb") as p:
                while b := p.read(1 << 24):
                    f.write(b)
            os.remove(f"{path}.part{k}")


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


def read_scan(data):
    """velodyne_sync points: x y z as uint16 (5 mm steps, offset -100 m), intensity, laser."""
    raw = np.frombuffer(data, dtype=np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("i", "u1"), ("l", "u1")]))
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1) * 0.005 - 100.0
    keep = np.linalg.norm(xyz, axis=1) > 1.5  # the Segway and its rider
    xyz = xyz[keep] * np.array([1.0, -1.0, -1.0])  # z up
    return np.hstack([xyz, raw["i"][keep, None] / 255.0]).astype(np.float32)


def write_kitti(path, poses):
    with open(path, "w") as f:
        for p in poses:
            f.write(" ".join(f"{v:.9g}" for v in p[:3].reshape(-1)) + "\n")


def write_gravity(stamps):
    """gravity/gravity.txt: the MS25 IMU's up direction at each keyframe, in its scan's frame
    (`keyframe ux uy uz`, for the pose graph panel's IMU gravity)."""
    sensors = f"{out}/{date}_sen.tar.gz"
    fetch(f"{BASE}/sensor_data/{date}_sen.tar.gz", sensors, parts=1)
    with tarfile.open(sensors, "r:gz") as tar:
        member = next(m for m in tar if m.name.endswith("ms25_euler.csv"))
        euler = np.loadtxt(tar.extractfile(member), delimiter=",")
    os.makedirs(f"{out}/gravity", exist_ok=True)
    body_vel_rotation = body_vel[:3, :3]
    with open(f"{out}/gravity/gravity.txt", "w") as f:
        for k, stamp in enumerate(stamps):
            roll, pitch = euler[np.abs(euler[:, 0] - stamp).argmin(), 1:3]
            body = ssc_to_matrix(0, 0, 0, roll, pitch, 0)[:3, :3]
            up = FLIP[:3, :3] @ body_vel_rotation.T @ body.T @ np.array([0.0, 0.0, -1.0])
            f.write(f"{k} {up[0]:.9f} {up[1]:.9f} {up[2]:.9f}\n")


if len(sys.argv) > 4 and sys.argv[3] == "--slice":
    # Keyframes FIRST:END of a prepared session, renumbered from 0, into another folder
    # (a long session's stretch that passes where another drive went, small enough to join).
    first, end = (int(v) for v in sys.argv[4].split(":"))
    dest = sys.argv[5]
    os.makedirs(f"{dest}/velodyne", exist_ok=True)
    os.makedirs(f"{dest}/gravity", exist_ok=True)
    for k in range(first, end):
        with open(f"{out}/velodyne/{k:06d}.bin", "rb") as src, open(f"{dest}/velodyne/{k - first:06d}.bin", "wb") as dst:
            dst.write(src.read())
    lines = open(f"{out}/velodyne/kiss_poses.txt").read().splitlines()[first:end]
    open(f"{dest}/velodyne/kiss_poses.txt", "w").write("\n".join(lines) + "\n")
    up = open(f"{out}/gravity/gravity.txt").read().splitlines()[first:end]
    open(f"{dest}/gravity/gravity.txt", "w").write(
        "".join(f"{k} {' '.join(line.split()[1:])}\n" for k, line in enumerate(up))
    )
    sys.exit()

if len(sys.argv) > 3 and sys.argv[3] == "--gravity-only":
    body_vel = ssc_to_matrix(*BODY_VEL[:3], *np.radians(BODY_VEL[3:]))
    write_gravity(np.loadtxt(f"{out}/times.txt") * 1e6)
    sys.exit()

tarball = f"{out}/{date}_vel.tar.gz"
fetch(f"{BASE}/velodyne_data/{date}_vel.tar.gz", tarball)
gt_csv = f"{out}/groundtruth_{date}.csv"
fetch(f"{BASE}/ground_truth/groundtruth_{date}.csv", gt_csv, parts=1)

raw = f"{out}/velodyne_sync"
if not os.path.isdir(raw):
    print("extracting scans…")
    os.makedirs(raw)
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar:
            if "velodyne_sync/" in member.name and member.name.endswith(".bin"):
                with open(f"{raw}/{os.path.basename(member.name)}", "wb") as f:
                    f.write(tar.extractfile(member).read())
stamps = np.array(sorted(int(f[:-4]) for f in os.listdir(raw) if f.endswith(".bin")))
print(len(stamps), "scans")


def scan_at(stamp):
    with open(f"{raw}/{stamp}.bin", "rb") as f:
        return read_scan(f.read())

rows = np.array([[float(v) for v in r] for r in csv.reader(open(gt_csv))])
rows = rows[~np.isnan(rows[:, 1:]).any(axis=1)]
body_vel = ssc_to_matrix(*BODY_VEL[:3], *np.radians(BODY_VEL[3:]))


def truth(stamp):
    k = np.searchsorted(rows[:, 0], stamp)
    k = min(max(k, 1), len(rows) - 1)
    near = k if abs(rows[k, 0] - stamp) < abs(rows[k - 1, 0] - stamp) else k - 1
    if abs(rows[near, 0] - stamp) > 50_000:
        return None
    return FLIP @ ssc_to_matrix(*rows[near, 1:7]) @ body_vel @ FLIP


print("KISS-ICP odometry…")
from kiss_icp.config import load_config  # noqa: E402
from kiss_icp.kiss_icp import KissICP  # noqa: E402

config = load_config(None)
config.data.deskew = False
config.data.max_range = 80.0
config.data.min_range = 1.5
odometry = KissICP(config)
kept, kiss, gt = [], [], []
last = None
for n, stamp in enumerate(stamps):
    g = truth(stamp)
    scan = scan_at(stamp)
    odometry.register_frame(scan[:, :3].astype(np.float64), np.zeros(len(scan)))
    pose = odometry.last_pose.copy()
    if g is None:
        continue
    if last is None or np.linalg.norm(pose[:3, 3] - last) >= KEYFRAME_SPACING:
        last = pose[:3, 3]
        scan.tofile(f"{out}/velodyne/{len(kept):06d}.bin")
        kept.append(stamp)
        kiss.append(pose)
        gt.append(g)
    if n % 1000 == 0:
        print(f"  {n} of {len(stamps)} scans, {len(kept)} keyframes")
origin = np.linalg.inv(gt[0])
write_kitti(f"{out}/velodyne/kiss_poses.txt", kiss)
write_kitti(f"{out}/gt_lidar.txt", [origin @ g for g in gt])
np.savetxt(f"{out}/times.txt", np.array(kept) / 1e6, fmt="%.6f")
print(len(kept), "keyframes written to", out)
write_gravity(np.array(kept))
