#!/usr/bin/env python3
"""Prepare ``newer-college-mini`` scans + GT suite for the SLAM leaderboard.

Newer College data cannot be redistributed with CloudAnalyzer. Point this
script at a local scan directory (``.pcd`` / ``.ply`` frames) plus the
published GT map + trajectory, then it:

1. Symlinks / copies scans into ``benchmarks/slam/newer-college-mini/scans/``
2. Runs ``scripts/prepare_newer_college_mini.py`` for ``suite.yaml``

Example::

    python scripts/prepare_leaderboard_newer_college.py \\
      --scans-dir /data/newer-college/short_experiment/scans \\
      --reference-map /data/newer-college/short_experiment/gt_map.pcd \\
      --reference-trajectory /data/newer-college/short_experiment/gt_poses.tum
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NC_MINI_DIR = REPO_ROOT / "benchmarks" / "slam" / "newer-college-mini"
SCAN_SUFFIXES = (".pcd", ".ply", ".bin")


def _select_scan_paths(scans_dir: Path, *, max_frames: int, stride: int) -> list[Path]:
    scans = sorted(
        path
        for path in scans_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SCAN_SUFFIXES
    )
    if not scans:
        raise FileNotFoundError(f"No scan frames under {scans_dir}")
    selected = scans[:: max(stride, 1)]
    if len(selected) > max_frames:
        step = max(1, len(selected) // max_frames)
        selected = selected[::step][:max_frames]
    return selected


def _materialize_scans(source_dir: Path, dest_dir: Path, *, max_frames: int, stride: int) -> int:
    selected = _select_scan_paths(source_dir, max_frames=max_frames, stride=stride)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(selected):
        dst = dest_dir / f"{idx:06d}{src.suffix.lower()}"
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans-dir", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument("--reference-trajectory", type=Path, required=True)
    parser.add_argument(
        "--output-scans-dir",
        type=Path,
        default=NC_MINI_DIR / "scans",
    )
    parser.add_argument("--sequence", default="short_experiment")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--voxel", type=float, default=0.10)
    parser.add_argument("--max-poses", type=int, default=2000)
    args = parser.parse_args()

    for label, path in (
        ("scans-dir", args.scans_dir),
        ("reference-map", args.reference_map),
        ("reference-trajectory", args.reference_trajectory),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    count = _materialize_scans(
        args.scans_dir,
        args.output_scans_dir,
        max_frames=args.max_frames,
        stride=args.stride,
    )
    prep_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "prepare_newer_college_mini.py"),
        "--reference-map",
        str(args.reference_map),
        "--reference-trajectory",
        str(args.reference_trajectory),
        "--sequence",
        args.sequence,
        "--voxel",
        str(args.voxel),
        "--max-poses",
        str(args.max_poses),
    ]
    subprocess.run(prep_cmd, check=True)
    print(f"Prepared {count} scan frames under {args.output_scans_dir}")
    print(f"Suite manifest: {NC_MINI_DIR / 'suite.yaml'}")


if __name__ == "__main__":
    main()
