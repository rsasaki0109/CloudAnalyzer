"""Fetch one PandaSet scene and write it as a folder the web app's pose graph panel opens.

PandaSet (Scale AI and Hesai, https://pandaset.org) is licensed CC BY 4.0. Only the scene's
files are read from the Hugging Face copy of the archive, by HTTP range requests
(`pip install remotezip numpy pandas`):

    python scripts/fetch_pandaset.py 019 pandaset/019

pandaset/019/velodyne/000000.bin ...  x y z intensity (float32): the 360° Pandar64 only, in its own frame
pandaset/019/velodyne/poses.txt       KITTI poses (sensor to world, 3x4 row-major)
pandaset/019/labels/000000.label      SemanticKITTI-style: 252 inside a moving cuboid, else 0
pandaset/019/labels_vehicle/...       the same, moving vehicles only

The README's odometry and dynamic-object GIFs are of scene 019 (`PANDASET_DIR=pandaset/019 npm run media`);
the labels score dynamic removal with `cargo run --release -p ca-core --example bench_dynamic`.
"""
import gzip
import io
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/georghess/pandaset/resolve/main/pandaset.zip"
seq, out = sys.argv[1], sys.argv[2]
os.makedirs(f"{out}/velodyne", exist_ok=True)
os.makedirs(f"{out}/labels", exist_ok=True)
os.makedirs(f"{out}/labels_vehicle", exist_ok=True)
PEOPLE = ("Pedestrian", "Pedestrian with Object")


def read(name):
    with RemoteZip(URL) as z:
        return z.read(f"pandaset/{seq}/{name}")


def quat_to_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


poses = json.loads(read("lidar/poses.json"))
rows = []
for p in poses:
    r = quat_to_matrix(p["heading"]["w"], p["heading"]["x"], p["heading"]["y"], p["heading"]["z"])
    t = np.array([p["position"]["x"], p["position"]["y"], p["position"]["z"]])
    rows.append((r, t))
with open(f"{out}/velodyne/poses.txt", "w") as f:
    for r, t in rows:
        m = np.hstack([r, t[:, None]]).ravel()
        f.write(" ".join(f"{v:.9g}" for v in m) + "\n")


def inside(points, cuboids):
    """Points (world) inside any of the cuboids (world position, dimensions, yaw about z)."""
    hit = np.zeros(len(points), bool)
    for _, c in cuboids.iterrows():
        cx, cy, cz = c["position.x"], c["position.y"], c["position.z"]
        dx, dy, dz = c["dimensions.x"], c["dimensions.y"], c["dimensions.z"]
        yaw = c["yaw"]
        d = points - np.array([cx, cy, cz])
        cs, sn = np.cos(-yaw), np.sin(-yaw)
        lx = cs * d[:, 0] - sn * d[:, 1]
        ly = sn * d[:, 0] + cs * d[:, 1]
        # A little room, as the labels are fitted tight.
        hit |= (np.abs(lx) <= dx / 2 + 0.1) & (np.abs(ly) <= dy / 2 + 0.1) & (np.abs(d[:, 2]) <= dz / 2 + 0.1)
    return hit


def frame(k):
    df = pickle.load(io.BytesIO(gzip.decompress(read(f"lidar/{k:02d}.pkl.gz"))))
    df = df[df["d"] == 0]
    world = df[["x", "y", "z"]].to_numpy()
    cub = pickle.load(io.BytesIO(gzip.decompress(read(f"annotations/cuboids/{k:02d}.pkl.gz"))))
    moving = cub[~cub["stationary"]]
    labels = np.where(inside(world, moving), 252, 0).astype(np.uint32)
    r, t = rows[k]
    local = (world - t) @ r
    data = np.hstack([local, df[["i"]].to_numpy() / 255.0]).astype(np.float32)
    data.tofile(f"{out}/velodyne/{k:06d}.bin")
    labels.tofile(f"{out}/labels/{k:06d}.label")
    vehicles = moving[~moving["label"].isin(PEOPLE)]
    np.where(inside(world, vehicles), 252, 0).astype(np.uint32).tofile(f"{out}/labels_vehicle/{k:06d}.label")
    return k, len(data), int((labels > 0).sum())


with ThreadPoolExecutor(8) as pool:
    for k, n, m in pool.map(frame, range(len(poses))):
        print(k, n, "points,", m, "moving")
