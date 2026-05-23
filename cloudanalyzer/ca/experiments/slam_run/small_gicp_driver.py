"""small_gicp driver for ``ca slam-run``.

Wraps `small_gicp` (https://github.com/koide3/small_gicp), a fast,
parallelized C++ point-cloud registration library with a Python binding
on PyPI under MIT. The driver does **scan-to-scan** GICP registration —
no incremental local map — which is intentionally simpler than the
KISS-ICP scan-to-map approach and gives an honest second operating point
for the slice's bake-off: lower per-frame cost, higher drift over long
sequences.

The world-frame map is built by transforming each input scan with its
recovered pose and voxel-downsampling once at the end (mirrors what
``KissSLAMSlamDriver`` does).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ca.core.slam_run import (
    SlamRunRequest,
    SlamRunResult,
    load_frame,
)


class SmallGICPSlamDriver:
    """Experimental SLAM driver — scan-to-scan GICP via ``small_gicp``."""

    name: str = "small_gicp"

    def run(self, request: SlamRunRequest) -> SlamRunResult:
        try:
            import small_gicp
        except ImportError as exc:  # pragma: no cover - exercised via the CLI test
            raise ImportError(
                "small_gicp is required for SmallGICPSlamDriver. "
                "Install with: pip install 'cloudanalyzer[slam]' "
                "or pip install small_gicp"
            ) from exc

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

        # Knobs. max_range, voxel_size, deskew share names with the other
        # drivers; small_gicp doesn't support deskew so it is silently
        # ignored (with the request still recorded in the metadata block
        # so a misconfig is visible in the summary).
        downsampling = (
            float(request.voxel_size_m) if request.voxel_size_m is not None else 0.25
        )
        max_corr_dist = (
            float(request.max_range_m) if request.max_range_m is not None else 1.0
        )
        # max_correspondence_distance smaller than scan extent is correct.
        # Cap it at 5 m so short-range scans still match meaningfully even
        # if --max-range is 80.
        max_corr_dist = min(max_corr_dist, 5.0)

        poses_list: list[np.ndarray] = []
        scans: list[np.ndarray] = []
        processed = 0
        cumulative_pose = np.eye(4, dtype=np.float64)

        t0 = time.perf_counter()
        prev_pts: np.ndarray | None = None
        for path in frame_paths:
            pts = load_frame(path)
            if pts.shape[0] == 0:
                continue
            if prev_pts is None:
                poses_list.append(cumulative_pose.copy())
                scans.append(pts)
                prev_pts = pts
                processed += 1
                continue

            result = small_gicp.align(
                prev_pts.astype(np.float64),
                pts.astype(np.float64),
                registration_type="GICP",
                downsampling_resolution=downsampling,
                max_correspondence_distance=max_corr_dist,
                num_threads=1,
                max_iterations=30,
            )
            t_relative = np.asarray(result.T_target_source, dtype=np.float64)
            cumulative_pose = cumulative_pose @ t_relative
            poses_list.append(cumulative_pose.copy())
            scans.append(pts)
            prev_pts = pts
            processed += 1
        runtime_s = time.perf_counter() - t0

        if processed == 0:
            raise ValueError(
                f"small_gicp processed 0 frames. Check that {request.frame_paths[0]} "
                "contains non-empty scans."
            )

        poses_arr = np.stack(poses_list, axis=0).astype(np.float64, copy=False)
        timestamps_kept = timestamps_s[:processed]

        # Build the world-frame map by transforming each scan with its
        # recovered pose and voxel-downsampling once at the end.
        world_chunks: list[np.ndarray] = []
        for pose, scan in zip(poses_arr, scans):
            if scan.shape[0] == 0:
                continue
            R = pose[:3, :3]
            t = pose[:3, 3]
            world_chunks.append(scan @ R.T + t)
        if world_chunks:
            map_world = np.vstack(world_chunks)
            if downsampling > 0 and map_world.shape[0] > 0:
                import open3d as o3d

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(map_world)
                pcd = pcd.voxel_down_sample(voxel_size=downsampling)
                map_world = np.asarray(pcd.points, dtype=np.float64)
        else:
            map_world = np.zeros((0, 3), dtype=np.float64)

        metadata: dict[str, Any] = {
            "small_gicp": {
                "registration_type": "GICP",
                "downsampling_resolution_m": float(downsampling),
                "max_correspondence_distance_m": float(max_corr_dist),
                "scan_to_scan": True,
                "deskew_requested_but_unsupported": bool(request.deskew),
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


__all__ = ["SmallGICPSlamDriver"]
