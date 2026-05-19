#!/usr/bin/env python3
"""Materialize the `kitti-mini` SLAM benchmark suite locally.

The KITTI Odometry benchmark (Geiger et al., CVPR 2012) ships
ground-truth poses for sequences 00-10 in a flat 12-float format
(one 3x4 row-major transformation matrix per line, no timestamps).
The dataset itself is too large to ship and CC BY-NC-SA 3.0 prevents
redistribution.

Users download the KITTI Odometry GT bundle locally, build a reference
map from the Velodyne scans (or use one they already trust), then run
this script once to turn it into a `ca benchmark eval`-ready suite
under `benchmarks/slam/kitti-mini/`.

This wraps `ca benchmark init` with KITTI-specific defaults baked in:
- KITTI 12-float pose conversion to TUM (auto, when `--kitti-poses` is given)
- 10 Hz timestamp synthesis (KITTI Velodyne rotation rate)
- Outdoor-scale voxel default (0.50 m, vs Newer College's 0.10 m)
- KITTI license string + a starter gate calibrated for ~500-2000 m sequences

See `benchmarks/slam/kitti-mini/README.md` for the full workflow.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ca.benchmark import GATE_KEYS, materialize_suite  # noqa: E402


SUITE_DIR = REPO_ROOT / "benchmarks" / "slam" / "kitti-mini"
DEFAULT_NAME = "kitti-mini"
DEFAULT_LICENSE = (
    "CC-BY-NC-SA 3.0 (KITTI Odometry Benchmark, Geiger et al., CVPR 2012)"
)
# Outdoor-scale starter gate. KITTI sequences span 500-2000 m, so absolute
# ATE / drift tolerances are larger than indoor datasets like Newer College.
DEFAULT_GATE: dict[str, float] = {
    "min_auc": 0.85,
    "max_chamfer": 0.50,
    "max_ate": 5.00,
    "max_rpe": 1.00,
    "max_drift": 5.00,
    "min_coverage": 0.80,
}
KITTI_FRAME_RATE_HZ = 10.0


def _rotation_matrix_to_quaternion(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to (qx, qy, qz, qw) using Shepperd's method."""
    m = np.asarray(rot, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def kitti_poses_to_tum(
    kitti_poses_path: Path,
    tum_out_path: Path,
    frame_rate_hz: float = KITTI_FRAME_RATE_HZ,
) -> int:
    """Convert KITTI 12-float pose file to TUM trajectory format.

    KITTI pose file: one 3x4 row-major transformation matrix per line
    (12 space-separated floats: ``r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz``).
    TUM trajectory: ``timestamp tx ty tz qx qy qz qw`` per line.

    Timestamps are synthesized at ``frame_rate_hz`` (KITTI Velodyne rate),
    starting from 0.0 — KITTI does not ship per-pose timestamps with the
    GT poses, and downstream tools (`ca run-evaluate`) only need
    monotonically increasing timestamps for pairing.

    Returns the number of poses written.
    """
    lines = [
        line.strip()
        for line in kitti_poses_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"{kitti_poses_path} contains no KITTI poses")

    dt = 1.0 / float(frame_rate_hz)
    out_lines: list[str] = []
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 12:
            raise ValueError(
                f"{kitti_poses_path}: line {idx + 1} has {len(parts)} fields, "
                "expected 12 (KITTI 3x4 pose)"
            )
        values = np.fromstring(line, sep=" ", dtype=np.float64)
        rot = values.reshape(3, 4)[:, :3]
        tx, ty, tz = values[3], values[7], values[11]
        qx, qy, qz, qw = _rotation_matrix_to_quaternion(rot)
        timestamp = idx * dt
        out_lines.append(
            f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
            f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
        )
    tum_out_path.parent.mkdir(parents=True, exist_ok=True)
    tum_out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(out_lines)


def _parse_gate_overrides(values: list[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if not values:
        return overrides
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"--gate expects key=value; got {raw!r}")
        key, _, value = raw.partition("=")
        key = key.strip()
        if key not in GATE_KEYS:
            raise SystemExit(
                f"unknown gate key {key!r}; allowed: {', '.join(GATE_KEYS)}"
            )
        try:
            overrides[key] = float(value)
        except ValueError as exc:
            raise SystemExit(f"--gate {key} must be numeric: {value!r}") from exc
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    pose_group = parser.add_mutually_exclusive_group(required=True)
    pose_group.add_argument(
        "--kitti-poses",
        type=Path,
        default=None,
        help="Path to KITTI ground-truth poses (12-float 3x4 matrix per line).",
    )
    pose_group.add_argument(
        "--reference-trajectory",
        type=Path,
        default=None,
        help="Path to a pre-converted TUM trajectory (skip the KITTI pose conversion).",
    )
    parser.add_argument(
        "--reference-map",
        required=True,
        type=Path,
        help=(
            "Path to a reference map (PCD/PLY). KITTI does not ship one — "
            "users typically build it by accumulating Velodyne scans using GT poses."
        ),
    )
    parser.add_argument(
        "--sequence",
        default="sequence_00",
        help="Sequence name written to the suite manifest (default: sequence_00)",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=KITTI_FRAME_RATE_HZ,
        help=f"Pose frame rate in Hz used to synthesize TUM timestamps "
        f"(default: {KITTI_FRAME_RATE_HZ}, KITTI Velodyne rotation rate)",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.50,
        help="Voxel size in meters for downsampling the reference map (default: 0.50)",
    )
    parser.add_argument(
        "--max-poses",
        type=int,
        default=2000,
        help="Keep at most this many evenly-spaced trajectory poses (default: 2000)",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Override the suite description (default: auto-generated from settings)",
    )
    parser.add_argument(
        "--license",
        default=DEFAULT_LICENSE,
        help=f"License string (default: {DEFAULT_LICENSE!r})",
    )
    parser.add_argument(
        "--gate",
        action="append",
        metavar="KEY=VALUE",
        help="Override a gate threshold (repeatable). Defaults: "
        + ", ".join(f"{k}={v}" for k, v in DEFAULT_GATE.items()),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SUITE_DIR,
        help=f"Output suite directory (default: {SUITE_DIR.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    if not args.reference_map.is_file():
        raise SystemExit(f"--reference-map not found: {args.reference_map}")

    # Resolve the reference trajectory: either user-provided TUM or
    # KITTI 12-float poses that we convert in a tempfile.
    with tempfile.TemporaryDirectory(prefix="kitti-mini-prep-") as tmpdir:
        if args.reference_trajectory is not None:
            if not args.reference_trajectory.is_file():
                raise SystemExit(
                    f"--reference-trajectory not found: {args.reference_trajectory}"
                )
            traj_path = args.reference_trajectory
            converted_poses: int | None = None
        else:
            assert args.kitti_poses is not None  # mutually-exclusive group guarantees this
            if not args.kitti_poses.is_file():
                raise SystemExit(f"--kitti-poses not found: {args.kitti_poses}")
            traj_path = Path(tmpdir) / "trajectory.tum"
            converted_poses = kitti_poses_to_tum(
                args.kitti_poses, traj_path, frame_rate_hz=args.frame_rate
            )

        gate = dict(DEFAULT_GATE)
        gate.update(_parse_gate_overrides(args.gate))

        description = args.description or (
            f"KITTI Odometry {args.sequence} sequence "
            f"(reference map downsampled to {args.voxel:g} m, "
            f"GT trajectory subsampled to {args.max_poses} poses)."
        )

        suite = materialize_suite(
            args.output,
            name=DEFAULT_NAME,
            description=description,
            reference_map=args.reference_map,
            reference_trajectory=traj_path,
            sequence_name=args.sequence,
            sequence_description=f"KITTI Odometry {args.sequence} GT bundle.",
            license=args.license,
            voxel_size=args.voxel,
            max_poses=args.max_poses,
            gate=gate,
        )

    print(f"Suite manifest: {suite.source_path}")
    print(f"Sequence:       {args.sequence}")
    if converted_poses is not None:
        print(f"KITTI poses:    converted {converted_poses} entries to TUM")
    seq = suite.resolve_sequence(args.sequence)
    print(f"Reference map:  {seq.reference_map_path}")
    print(f"Reference traj: {seq.reference_trajectory_path}")
    print("Gate:")
    for key, value in suite.gate.items():
        print(f"  {key}: {value}")
    print()
    print("Ready. Try:")
    try:
        suite_arg = suite.source_path.relative_to(REPO_ROOT)
    except ValueError:
        suite_arg = suite.source_path
    print(
        f"  ca benchmark eval {suite_arg} "
        f"--map <your_slam_map.pcd> --trajectory <your_slam_trajectory.tum>"
    )


if __name__ == "__main__":
    main()
