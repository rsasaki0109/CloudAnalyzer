"""Open3D ICP scan-to-scan driver for ``ca slam-run``.

This is the canonical example of a third-party SLAM driver. It depends
only on what CloudAnalyzer already requires (``numpy``, ``open3d``) —
no ``kiss-icp`` / ``kiss-slam`` / ``small_gicp`` extras. It registers
itself under the ``example`` name via the
``cloudanalyzer.slam_run_drivers`` entry-point group declared in this
package's ``pyproject.toml``, so after ``pip install
cloudanalyzer-driver-example`` it shows up as
``ca slam-run --driver example`` automatically.

The registration loop is a deliberately plain point-to-point ICP between
consecutive scans, with global pose accumulated by composing the
per-frame relative transforms. The world-frame map is a scan-stitched +
voxel-downsampled concatenation of the input scans transformed by their
recovered poses (same shape as the built-in ``small_gicp`` driver's
map output).

This is *not* tuned to win the slam_run bake-off. It exists as a working
template for anyone shipping their own driver.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import open3d as o3d

from ca.core.slam_run import (
    SlamRunRequest,
    SlamRunResult,
    load_frame,
)


class Open3DICPSlamDriver:
    """Scan-to-scan point-to-point ICP via Open3D."""

    name: str = "open3d_icp"

    def run(self, request: SlamRunRequest) -> SlamRunResult:
        frame_paths = (
            request.frame_paths[: request.max_frames]
            if request.max_frames is not None
            else request.frame_paths
        )

        if request.timestamps_s is not None:
            base_ts = np.asarray(request.timestamps_s, dtype=np.float64)
            if base_ts.shape[0] < len(frame_paths):
                raise ValueError(
                    f"timestamps_s shorter than frame_paths "
                    f"({base_ts.shape[0]} < {len(frame_paths)})"
                )
            timestamps_s = base_ts[: len(frame_paths)]
        else:
            timestamps_s = np.arange(len(frame_paths), dtype=np.float64) * float(
                request.frame_period_s
            )

        # Knobs. voxel_size_m doubles as the input downsampling resolution
        # so dense LiDAR sweeps stay tractable; max_range_m caps the ICP
        # correspondence distance.
        voxel_size = (
            float(request.voxel_size_m) if request.voxel_size_m is not None else 0.5
        )
        max_corr = (
            float(request.max_range_m) if request.max_range_m is not None else 1.0
        )
        # Tight max-correspondence cap so cluttered scans don't get spurious
        # matches; 5 m is generous for indoor LiDAR.
        max_corr = min(max_corr, 5.0)

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=30,
        )

        poses_list: list[np.ndarray] = []
        raw_scans: list[np.ndarray] = []
        processed = 0
        prev_pose = np.eye(4, dtype=np.float64)
        prev_delta = np.eye(4, dtype=np.float64)
        prev_pcd: o3d.geometry.PointCloud | None = None

        t0 = time.perf_counter()
        for path in frame_paths:
            pts = load_frame(path)
            if pts.shape[0] == 0:
                continue
            pts = pts.astype(np.float64, copy=False)
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(pts)
            if voxel_size > 0:
                src_pcd = src_pcd.voxel_down_sample(voxel_size=voxel_size)

            if prev_pcd is None:
                pose = np.eye(4, dtype=np.float64)
            else:
                # Constant-velocity initial guess.
                init_guess = prev_delta
                result = o3d.pipelines.registration.registration_icp(
                    src_pcd,
                    prev_pcd,
                    max_corr,
                    init_guess,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria,
                )
                t_relative = np.asarray(result.transformation, dtype=np.float64)
                pose = prev_pose @ t_relative

            prev_delta = np.linalg.inv(prev_pose) @ pose
            prev_pose = pose
            prev_pcd = src_pcd
            poses_list.append(pose.copy())
            raw_scans.append(pts)
            processed += 1
        runtime_s = time.perf_counter() - t0

        if processed == 0:
            raise ValueError(
                f"open3d_icp processed 0 frames. Check that "
                f"{request.frame_paths[0]} contains non-empty scans."
            )

        poses_arr = np.stack(poses_list, axis=0).astype(np.float64, copy=False)
        timestamps_kept = timestamps_s[:processed]

        # World-frame map: transform each scan by its pose, concatenate,
        # voxel-downsample once at the end.
        world_chunks: list[np.ndarray] = []
        for pose, scan in zip(poses_arr, raw_scans):
            R = pose[:3, :3]
            t = pose[:3, 3]
            world_chunks.append(scan @ R.T + t)
        if world_chunks:
            stitched = np.vstack(world_chunks)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(stitched)
            if voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size / 2.0)
            map_world = np.asarray(pcd.points, dtype=np.float64)
        else:
            map_world = np.zeros((0, 3), dtype=np.float64)

        metadata: dict[str, Any] = {
            "open3d_icp": {
                "registration_type": "PointToPointICP",
                "voxel_size_m": float(voxel_size),
                "max_correspondence_distance_m": float(max_corr),
                "max_iterations": int(criteria.max_iteration),
            }
        }

        return SlamRunResult(
            driver=self.name,
            poses=poses_arr,
            timestamps_s=timestamps_kept.astype(np.float64, copy=False),
            map_points=map_world,
            runtime_s=runtime_s,
            frames_processed=processed,
            metadata=metadata,
        )


__all__ = ["Open3DICPSlamDriver"]
