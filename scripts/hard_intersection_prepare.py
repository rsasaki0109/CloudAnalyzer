"""Prepare geometry-only evaluation inputs, without opening any reference labels/map.

Retains the FIRST original point per 10 cm voxel, including intensity/RGB.
Semantic fields are cleared. Memory is bounded by a bitset and a LAS chunk;
20 m ownership tiles have 12 m halos for the unchanged detector.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
from pathlib import Path

import laspy
import numpy as np

VOXEL = .10
CORE = 20.
HALO = 12.
CHUNK = 300_000


class FirstVoxel:
    def __init__(self, minimum, maximum, voxel=VOXEL, max_bytes=128 * 1024**2):
        self.minimum = np.asarray(minimum, dtype=float)
        maximum = np.asarray(maximum, dtype=float)
        if not np.isfinite([*self.minimum, *maximum, voxel]).all() or voxel <= 0 or np.any(maximum < self.minimum):
            raise ValueError("invalid voxel extent")
        self.voxel = voxel
        self.shape = np.floor((maximum - self.minimum) / voxel).astype(np.int64) + 1
        count = math.prod(map(int, self.shape))
        if (count + 7) // 8 > max_bytes:
            raise ValueError("voxel bitset exceeds memory budget; split the scene")
        self.bits = np.zeros((count + 7) // 8, dtype=np.uint8)

    def retain(self, xyz):
        if not np.isfinite(xyz).all():
            raise ValueError("nonfinite source point")
        grid = np.floor((xyz - self.minimum) / self.voxel).astype(np.int64)
        if np.any(grid < 0) or np.any(grid >= self.shape):
            raise ValueError("point outside declared voxel extent")
        key = (grid[:, 0] * self.shape[1] + grid[:, 1]) * self.shape[2] + grid[:, 2]
        unique, first = np.unique(key, return_index=True)
        byte, mask = unique >> 3, (1 << (unique & 7)).astype(np.uint8)
        unseen = (self.bits[byte] & mask) == 0
        np.bitwise_or.at(self.bits, byte, mask)
        return np.sort(first[unseen])


def trajectory_segments(path, minimum, maximum):
    """Split at source-extent exits; never connect a drive across missing geometry."""
    expected = ["Time[s]", "Easting[m]", "Northing[m]", "Height[m]", "Roll[deg]", "Pitch[deg]", "Yaw[deg]"]
    segments, current = [], []
    last_time = None
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if next(reader) != expected:
            raise ValueError("unexpected source trajectory header")
        for row in reader:
            time, x, y, z = map(float, row[:4])
            if not np.isfinite([time, x, y, z]).all() or (last_time is not None and time <= last_time):
                raise ValueError("nonfinite or nonincreasing recorded trajectory")
            last_time = time
            inside = minimum[0] <= x <= maximum[0] and minimum[1] <= y <= maximum[1]
            if not inside:
                if len(current) >= 2:
                    segments.append(current)
                current = []
            elif not current or math.hypot(x - current[-1][1], y - current[-1][2]) >= .25:
                current.append([time, x, y, z])
    if len(current) >= 2:
        segments.append(current)
    return segments


def prepare(source: Path, trajectories: Path, out: Path):
    if shutil.disk_usage(out.parent).free < 6 * 1024**3:
        raise ValueError("preparation requires 6 GiB free")
    out.mkdir(exist_ok=False)
    compact = out / "geometry.las"
    retained = 0
    with laspy.open(source) as reader:
        header = copy.deepcopy(reader.header)
        minimum, maximum = header.mins.copy(), header.maxs.copy()
        seen = FirstVoxel(minimum, maximum)
        source_count = header.point_count
        with laspy.open(compact, mode="w", header=header) as writer:
            for points in reader.chunk_iterator(CHUNK):
                xyz = np.column_stack([points.x, points.y, points.z])
                selected = points[seen.retain(xyz)].copy()
                selected.classification[:] = 0
                selected.user_data[:] = 0
                writer.write_points(selected)
                retained += len(selected)
    print(f"Retained {retained:,}/{source_count:,} original points; bitset {seen.bits.nbytes:,} bytes", flush=True)
    del seen
    tiles = []
    nx, ny = [max(1, int(math.ceil((maximum[i] - minimum[i]) / CORE))) for i in (0, 1)]
    for ix in range(nx):
        for iy in range(ny):
            lower = minimum[:2] + np.array([ix, iy]) * CORE
            upper = lower + CORE
            if ix == nx - 1:
                upper[0] = np.nextafter(maximum[0], math.inf)
            if iy == ny - 1:
                upper[1] = np.nextafter(maximum[1], math.inf)
            path = out / f"tile-{ix:02d}-{iy:02d}.las"
            count = 0
            with laspy.open(compact) as reader, laspy.open(path, mode="w", header=copy.deepcopy(reader.header)) as writer:
                for points in reader.chunk_iterator(CHUNK):
                    mask = ((points.x >= lower[0] - HALO) & (points.x < upper[0] + HALO)
                            & (points.y >= lower[1] - HALO) & (points.y < upper[1] + HALO))
                    count += int(mask.sum())
                    if count > 2_000_000:
                        raise ValueError("tile exceeds unchanged detector's 2M source-point cap")
                    writer.write_points(points[mask])
            tiles.append({"path": path.name, "points": count, "core_min": lower.tolist(), "core_max": upper.tolist()})
    drives = []
    for src in sorted(trajectories.glob("*.txt")):
        for i, segment in enumerate(trajectory_segments(src, minimum, maximum)):
            name = f"{src.stem}-{i:02d}.csv"
            with (out / name).open("w", encoding="utf-8", newline="") as f:
                csv.writer(f, lineterminator="\n").writerows([["timestamp", "x", "y", "z"], *segment])
            drives.append({"path": name, "poses": len(segment), "recording": src.name})
    report = {"crs": "EPSG:6677", "height": "source orthometric, unchanged", "source_points": source_count,
              "retained_points": retained, "voxel_m": VOXEL, "selection": "first original point, no averaging",
              "semantic_fields": "classification and UserData cleared", "reference_inputs": [],
              "core_m": CORE, "halo_m": HALO, "chunk_points": CHUNK, "tiles": tiles, "drives": drives,
              "bytes": sum(p.stat().st_size for p in out.iterdir() if p.is_file())}
    (out / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_las", type=Path)
    parser.add_argument("trajectory_directory", type=Path)
    parser.add_argument("output", type=Path, help="NEW directory")
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw_las, args.trajectory_directory, args.output), indent=2))


if __name__ == "__main__":
    main()
