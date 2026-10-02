"""Prepare the vector-map README drive from PandaSet 019 (CC BY 4.0).

Requires numpy, pandas and remotezip. Keeps original world XYZ/intensity from
Pandar64 frames 0, 8, ..., 72 and all 80 recorded sensor positions. No map geometry
is supplied. The CSV uses recorded LiDAR timestamps relative to the first frame.
"""

import argparse
import gzip
import io
import json
import pickle
from pathlib import Path

import numpy as np
from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/georghess/pandaset/resolve/main/pandaset.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if any((args.out / name).exists() for name in ["map.pcd", "trajectory.csv"]):
        parser.error("choose a directory without map.pcd or trajectory.csv")
    with RemoteZip(URL) as archive:
        poses = json.loads(archive.read("pandaset/019/lidar/poses.json"))
        timestamps = json.loads(archive.read("pandaset/019/lidar/timestamps.json"))
        if len(poses) != 80 or len(timestamps) != len(poses):
            raise ValueError("expected 80 matching scene 019 poses and timestamps")
        frames = []
        for k in range(0, len(poses), 8):
            # These are the dataset's own pandas pickle files, like fetch_pandaset.py.
            table = pickle.load(
                io.BytesIO(
                    gzip.decompress(archive.read(f"pandaset/019/lidar/{k:02d}.pkl.gz"))
                )
            )
            frames.append(
                table[table.d == 0][["x", "y", "z", "i"]].to_numpy(dtype=np.float32)
            )
            print(f"frame {k}: {len(frames[-1])} points", flush=True)
    points = np.concatenate(frames)
    header = (
        "# PandaSet 019, Pandar64 frames 0:80:8, CC BY 4.0\nVERSION .7\n"
        "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "map.pcd").open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(points.tobytes())
    with (args.out / "trajectory.csv").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write("timestamp,x,y,z\n")
        for timestamp, pose in zip(timestamps, poses, strict=True):
            p = pose["position"]
            stream.write(
                f"{timestamp - timestamps[0]:.9f},{p['x']},{p['y']},{p['z']}\n"
            )
    print(
        f"Wrote {len(points)} original intensity points and {len(poses)} poses to {args.out}"
    )


if __name__ == "__main__":
    main()
