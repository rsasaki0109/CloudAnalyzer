#!/usr/bin/env python3
"""Prepare ``kitti-mini`` scans + GT suite for the public SLAM leaderboard.

KITTI Odometry data cannot be redistributed with CloudAnalyzer, so this
script materializes a local-only leaderboard fixture:

1. Subsample Velodyne ``.bin`` frames into ``benchmarks/slam/kitti-mini/scans/``
2. Accumulate a coarse GT reference map from the subsampled scans + poses
3. Run ``scripts/prepare_kitti_mini.py`` to write ``suite.yaml``

Example::

    python scripts/prepare_leaderboard_kitti.py \\
      --velodyne-dir ~/datasets/kitti_seq00_velodyne \\
      --kitti-poses ~/datasets/kitti_odometry_training_subsets/seq00/poses_00.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
KITTI_MINI_DIR = REPO_ROOT / "benchmarks" / "slam" / "kitti-mini"
DEFAULT_MAX_FRAMES = 400
DEFAULT_STRIDE = 10
DEFAULT_VOXEL = 0.50


def _load_kitti_poses(path: Path) -> list[np.ndarray]:
    """Return 4x4 world-from-velodyne matrices for each KITTI pose line."""

    matrices: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = [float(part) for part in stripped.split()]
        if len(values) != 12:
            raise ValueError(f"{path}: expected 12 floats per pose, got {len(values)}")
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
        matrices.append(mat)
    if not matrices:
        raise ValueError(f"{path}: no KITTI poses found")
    return matrices


def _load_velodyne_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0 or raw.size % 4 != 0:
        raise ValueError(f"{path}: invalid KITTI .bin size {raw.size}")
    return raw.reshape(-1, 4)[:, :3].astype(np.float64, copy=False)


def _select_frame_indices(total_frames: int, *, stride: int, max_frames: int) -> list[int]:
    indices = list(range(0, total_frames, max(stride, 1)))
    if len(indices) > max_frames:
        step = max(1, len(indices) // max_frames)
        indices = indices[::step][:max_frames]
    return indices


def _materialize_scans(
    velodyne_dir: Path,
    scans_dir: Path,
    *,
    stride: int,
    max_frames: int,
) -> tuple[list[Path], list[int]]:
    bins = sorted(velodyne_dir.glob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"No .bin frames under {velodyne_dir}")

    indices = _select_frame_indices(len(bins), stride=stride, max_frames=max_frames)
    if scans_dir.exists():
        for child in scans_dir.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
    else:
        scans_dir.mkdir(parents=True, exist_ok=True)

    selected_bins: list[Path] = []
    for out_idx, src_idx in enumerate(indices):
        src = bins[src_idx]
        dst = scans_dir / f"{out_idx:06d}.bin"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        selected_bins.append(dst)
    return selected_bins, indices


def _build_reference_map(
    velodyne_dir: Path,
    poses_path: Path,
    frame_indices: list[int],
    *,
    voxel_size: float,
    map_path: Path,
) -> None:
    poses = _load_kitti_poses(poses_path)
    bins = sorted(velodyne_dir.glob("*.bin"))
    clouds: list[o3d.geometry.PointCloud] = []
    for idx in frame_indices:
        if idx >= len(bins) or idx >= len(poses):
            break
        points = _load_velodyne_bin(bins[idx])
        if points.size == 0:
            continue
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.transform(poses[idx])
        clouds.append(cloud)

    if not clouds:
        raise RuntimeError("No Velodyne points accumulated for the reference map")

    merged = clouds[0]
    for extra in clouds[1:]:
        merged += extra
    merged = merged.voxel_down_sample(float(voxel_size))
    map_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(map_path), merged, write_ascii=False):
        raise RuntimeError(f"Failed to write reference map to {map_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--velodyne-dir",
        type=Path,
        required=True,
        help="Directory of KITTI Velodyne .bin frames",
    )
    parser.add_argument(
        "--kitti-poses",
        type=Path,
        required=True,
        help="KITTI 12-float poses file (e.g. poses/00.txt)",
    )
    parser.add_argument(
        "--scans-dir",
        type=Path,
        default=KITTI_MINI_DIR / "scans",
        help="Output scans directory for ca slam-run (default: kitti-mini/scans)",
    )
    parser.add_argument(
        "--reference-map",
        type=Path,
        default=KITTI_MINI_DIR / "data" / "sequence_00" / "map.pcd",
        help="Accumulated GT reference map output path",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Take every Nth Velodyne frame (default: {DEFAULT_STRIDE})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Cap subsampled scan count (default: {DEFAULT_MAX_FRAMES})",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=DEFAULT_VOXEL,
        help=f"Voxel size for the accumulated GT map (default: {DEFAULT_VOXEL})",
    )
    args = parser.parse_args()

    if not args.velodyne_dir.is_dir():
        raise SystemExit(f"Velodyne directory not found: {args.velodyne_dir}")
    if not args.kitti_poses.is_file():
        raise SystemExit(f"KITTI poses file not found: {args.kitti_poses}")

    _, frame_indices = _materialize_scans(
        args.velodyne_dir,
        args.scans_dir,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    _build_reference_map(
        args.velodyne_dir,
        args.kitti_poses,
        frame_indices,
        voxel_size=args.voxel,
        map_path=args.reference_map,
    )

    prep_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "prepare_kitti_mini.py"),
        "--kitti-poses",
        str(args.kitti_poses),
        "--reference-map",
        str(args.reference_map),
        "--sequence",
        "sequence_00",
        "--voxel",
        str(args.voxel),
        "--max-poses",
        str(min(args.max_frames, 2000)),
    ]
    subprocess.run(prep_cmd, check=True)

    print(f"Prepared {len(frame_indices)} scan frames under {args.scans_dir}")
    print(f"Reference map: {args.reference_map}")
    print(f"Suite manifest: {KITTI_MINI_DIR / 'suite.yaml'}")


if __name__ == "__main__":
    main()
