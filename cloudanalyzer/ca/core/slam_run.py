"""Stable contract for ``ca slam-run``: drive a LiDAR-odometry pipeline
end-to-end on a sequence of input scans and produce the artifacts the rest of
CloudAnalyzer's evaluation stack expects (estimated map PLY + TUM trajectory).

The slice has two implementations under ``ca.experiments.slam_run``:

- ``KissICPSlamDriver`` (adopted, also re-exported here): wraps the
  ``kiss-icp`` package — a small, well-known scan-to-map LiDAR-odometry
  registration pipeline. Picks up KITTI / Newer College sequences with
  ``deskew`` enabled.
- ``IdentityPassthroughSlamDriver``: a zero-motion sentinel that emits identity
  poses and concatenates input frames as the "map". It is intentionally
  bad — its only job is to prove the harness is wired and to provide a floor
  for the slice's evaluator.

The contract:

- :class:`SlamRunRequest` carries the inputs you actually have (frame paths,
  optional per-frame timestamps, driver knobs).
- :class:`SlamRunResult` carries everything needed to feed the existing
  ``ca run-evaluate`` pipeline plus enough metadata for ``ca check`` /
  ``ca report-pr-comment``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(slots=True)
class SlamRunRequest:
    """Inputs to a single SLAM driver run."""

    frame_paths: tuple[Path, ...]
    """Per-frame scan paths in temporal order (``.pcd`` / ``.ply`` / ``.bin``)."""

    timestamps_s: tuple[float, ...] | None = None
    """Optional per-frame timestamps in seconds (length must equal frame_paths).
    Used for the TUM trajectory output and for drivers that consume timing.
    If omitted, drivers fall back to ``index * frame_period_s``."""

    frame_period_s: float = 0.1
    """Fallback timestamp spacing used when ``timestamps_s`` is None."""

    max_range_m: float | None = None
    """Drop scan points farther than this from the sensor before registration.
    ``None`` keeps every point."""

    voxel_size_m: float | None = None
    """Driver-side voxel grid size for the local map. ``None`` uses the driver's
    own default."""

    deskew: bool = True
    """Whether the driver should motion-deskew input frames (KISS-ICP default)."""

    max_frames: int | None = None
    """Optional cap on the number of frames consumed. ``None`` runs everything."""


@dataclass(slots=True)
class SlamRunResult:
    """Outputs of a single SLAM driver run."""

    driver: str
    """Identifier of the driver that produced this result."""

    poses: np.ndarray
    """``(N, 4, 4)`` array of sensor-to-world poses, one per processed frame."""

    timestamps_s: np.ndarray
    """``(N,)`` array of pose timestamps in seconds."""

    map_points: np.ndarray
    """``(M, 3)`` accumulated map point cloud in the world frame."""

    runtime_s: float
    """Total wall-clock seconds spent inside the driver (frame iteration only)."""

    frames_processed: int
    """How many frames the driver consumed before producing the result."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Driver-specific knobs / config snapshot. Echoed into the summary JSON
    so a downstream baseline can prove two runs used the same settings."""


@runtime_checkable
class SlamRunDriver(Protocol):
    """A SLAM driver consumes a request and returns the artifacts contract."""

    name: str

    def run(self, request: SlamRunRequest) -> SlamRunResult:  # pragma: no cover - Protocol
        ...


# ---------------------------------------------------------------------------
# Shared helpers used by every driver (loading frames, writing artifacts).
# ---------------------------------------------------------------------------


def load_frame(path: Path) -> np.ndarray:
    """Load one LiDAR scan as an ``(N, 3)`` float64 ndarray.

    Supports:

    - ``.bin``: KITTI Velodyne format (float32 quadruples, XYZI; intensity dropped).
    - ``.pcd`` / ``.ply``: read through Open3D.
    - ``.csv``: x,y,z columns (optionally with a header), parsed by ``ca.io``.

    Empty scans are returned as ``(0, 3)`` so drivers can skip them without
    raising.
    """

    suffix = path.suffix.lower()
    if suffix == ".bin":
        raw = np.fromfile(str(path), dtype=np.float32)
        if raw.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if raw.size % 4 != 0:
            raise ValueError(
                f"KITTI .bin frame {path} has size {raw.size} not divisible by 4"
            )
        return raw.reshape(-1, 4)[:, :3].astype(np.float64, copy=False)
    if suffix in {".pcd", ".ply", ".csv"}:
        from ca.io import load_point_cloud

        pcd = load_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return pts
    raise ValueError(
        f"Unsupported frame format '{suffix}'. Expected one of .bin .pcd .ply .csv"
    )


def discover_frame_paths(input_path: Path) -> list[Path]:
    """Resolve ``--input`` to an ordered list of frame paths.

    Accepts either:

    - A directory: pick up ``*.bin``, ``*.pcd``, ``*.ply`` (in that priority
      order, picking whichever extension dominates) and sort lexicographically.
      KITTI Velodyne dumps use lex-sortable zero-padded filenames so this
      matches frame order.
    - A list file (``*.txt``): one path per line. Relative paths are resolved
      against the list file's parent directory.
    """

    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        out: list[Path] = []
        base = input_path.parent
        for raw in input_path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            candidate = Path(stripped)
            if not candidate.is_absolute():
                candidate = base / candidate
            out.append(candidate)
        if not out:
            raise ValueError(f"Frame list {input_path} is empty")
        return out

    if not input_path.is_dir():
        raise ValueError(
            f"--input must be a directory of scans or a .txt list; got {input_path}"
        )

    for ext in (".bin", ".pcd", ".ply"):
        hits = sorted(input_path.glob(f"*{ext}"))
        if hits:
            return hits
    raise ValueError(
        f"No .bin/.pcd/.ply scans found under {input_path}. "
        "Pass a directory containing LiDAR sweeps or a frames-list.txt."
    )


def write_tum_trajectory(
    path: Path, poses: np.ndarray, timestamps_s: np.ndarray
) -> None:
    """Dump poses to TUM format (``timestamp tx ty tz qx qy qz qw``).

    Matches the format ``ca traj-evaluate`` already consumes.
    """

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (N, 4, 4); got {poses.shape}")
    if timestamps_s.shape != (poses.shape[0],):
        raise ValueError(
            f"timestamps_s {timestamps_s.shape} must match poses {poses.shape[0]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    quats = _rotation_matrices_to_quaternions(poses[:, :3, :3])
    with path.open("w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, pose, q in zip(timestamps_s, poses, quats):
            tx, ty, tz = pose[:3, 3]
            qx, qy, qz, qw = q
            f.write(
                f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )


def write_map_ply(path: Path, map_points: np.ndarray) -> None:
    """Save the accumulated SLAM map as an Open3D-readable PLY."""

    if map_points.ndim != 2 or map_points.shape[1] != 3:
        raise ValueError(f"map_points must be (M, 3); got {map_points.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(map_points, dtype=np.float64)
    )
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _rotation_matrices_to_quaternions(rotmats: np.ndarray) -> np.ndarray:
    """Vectorized rotation-matrix → ``(qx, qy, qz, qw)`` quaternion.

    Uses the standard Shepperd-style branchless trace selection so it works
    cleanly even when most rotations are near-identity (the common case for
    indoor LiDAR sweeps where motion is small per frame).
    """

    if rotmats.ndim != 3 or rotmats.shape[1:] != (3, 3):
        raise ValueError(f"rotmats must be (N, 3, 3); got {rotmats.shape}")

    n = rotmats.shape[0]
    out = np.zeros((n, 4), dtype=np.float64)

    m00 = rotmats[:, 0, 0]
    m11 = rotmats[:, 1, 1]
    m22 = rotmats[:, 2, 2]
    trace = m00 + m11 + m22

    # Branch on the largest diagonal entry to avoid numerical blowups.
    case_w = trace > 0.0
    case_x = (~case_w) & (m00 >= m11) & (m00 >= m22)
    case_y = (~case_w) & (~case_x) & (m11 >= m22)
    case_z = (~case_w) & (~case_x) & (~case_y)

    if np.any(case_w):
        idx = np.where(case_w)[0]
        s = np.sqrt(trace[idx] + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotmats[idx, 2, 1] - rotmats[idx, 1, 2]) / s
        qy = (rotmats[idx, 0, 2] - rotmats[idx, 2, 0]) / s
        qz = (rotmats[idx, 1, 0] - rotmats[idx, 0, 1]) / s
        out[idx, 0] = qx
        out[idx, 1] = qy
        out[idx, 2] = qz
        out[idx, 3] = qw

    if np.any(case_x):
        idx = np.where(case_x)[0]
        s = np.sqrt(1.0 + m00[idx] - m11[idx] - m22[idx]) * 2.0
        out[idx, 0] = 0.25 * s
        out[idx, 1] = (rotmats[idx, 0, 1] + rotmats[idx, 1, 0]) / s
        out[idx, 2] = (rotmats[idx, 0, 2] + rotmats[idx, 2, 0]) / s
        out[idx, 3] = (rotmats[idx, 2, 1] - rotmats[idx, 1, 2]) / s

    if np.any(case_y):
        idx = np.where(case_y)[0]
        s = np.sqrt(1.0 + m11[idx] - m00[idx] - m22[idx]) * 2.0
        out[idx, 0] = (rotmats[idx, 0, 1] + rotmats[idx, 1, 0]) / s
        out[idx, 1] = 0.25 * s
        out[idx, 2] = (rotmats[idx, 1, 2] + rotmats[idx, 2, 1]) / s
        out[idx, 3] = (rotmats[idx, 0, 2] - rotmats[idx, 2, 0]) / s

    if np.any(case_z):
        idx = np.where(case_z)[0]
        s = np.sqrt(1.0 + m22[idx] - m00[idx] - m11[idx]) * 2.0
        out[idx, 0] = (rotmats[idx, 0, 2] + rotmats[idx, 2, 0]) / s
        out[idx, 1] = (rotmats[idx, 1, 2] + rotmats[idx, 2, 1]) / s
        out[idx, 2] = 0.25 * s
        out[idx, 3] = (rotmats[idx, 1, 0] - rotmats[idx, 0, 1]) / s

    return out


# ---------------------------------------------------------------------------
# Adopted driver re-export. Concrete impl lives under ca.experiments.slam_run
# so the slice's evaluator can still benchmark it against the sentinel; the
# CLI imports from here so ``ca.core`` stays the only stable surface.
# ---------------------------------------------------------------------------


def _import_kiss_icp_driver() -> type[SlamRunDriver]:
    from ca.experiments.slam_run.kiss_icp_driver import KissICPSlamDriver

    return KissICPSlamDriver


def make_default_driver() -> SlamRunDriver:
    """Return the currently adopted SLAM driver instance (KISS-ICP)."""

    cls = _import_kiss_icp_driver()
    return cls()


def run_slam(request: SlamRunRequest, driver: SlamRunDriver | None = None) -> SlamRunResult:
    """Convenience entry point used by the CLI; delegates to the supplied
    driver or to the adopted default."""

    drv = driver or make_default_driver()
    return drv.run(request)


__all__ = [
    "SlamRunRequest",
    "SlamRunResult",
    "SlamRunDriver",
    "load_frame",
    "discover_frame_paths",
    "write_tum_trajectory",
    "write_map_ply",
    "make_default_driver",
    "run_slam",
]
