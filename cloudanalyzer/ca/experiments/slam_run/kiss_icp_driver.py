"""KISS-ICP driver for ``ca slam-run``.

Wraps the upstream ``kiss-icp`` package (``pip install kiss-icp``) which
exposes a minimal Python class :class:`kiss_icp.kiss_icp.KissICP` with a
``register_frame(points, timestamps)`` API. We feed scans one at a time,
read back ``last_pose`` after each step to accumulate the trajectory, and
finally pull the densified local map via ``local_map.point_cloud()``.

The driver only depends on the SLAM run contract in :mod:`ca.core.slam_run`
plus ``kiss-icp`` itself. It does *not* pull in ROS, Open3D's visualizer,
or any heavy IO — those stay outside the slice.
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


class KissICPSlamDriver:
    """Adopted SLAM driver — KISS-ICP, scan-to-map registration only."""

    name: str = "kiss_icp"

    def run(self, request: SlamRunRequest) -> SlamRunResult:
        try:
            from kiss_icp.config import load_config
            from kiss_icp.kiss_icp import KissICP
        except ImportError as exc:  # pragma: no cover - exercised via the CLI test
            raise ImportError(
                "kiss-icp is required for KissICPSlamDriver. "
                "Install with: pip install 'cloudanalyzer[slam]' "
                "or pip install kiss-icp"
            ) from exc

        max_range = (
            float(request.max_range_m) if request.max_range_m is not None else 100.0
        )
        config = load_config(None)
        config.data.max_range = max_range
        config.data.deskew = bool(request.deskew)
        if request.voxel_size_m is not None:
            config.mapping.voxel_size = float(request.voxel_size_m)

        odom = KissICP(config)

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

        poses: list[np.ndarray] = []
        processed = 0
        t0 = time.perf_counter()
        for path in frame_paths:
            pts = load_frame(path)
            if pts.shape[0] == 0:
                continue
            # KISS-ICP wants per-point timestamps in [0, 1] across the sweep.
            # We don't know the actual sweep timing here, so feed zeros — this
            # disables deskew effectively for this point, which is what we want
            # for offline LiDAR dumps that have already been ego-motion compensated
            # or that we don't have per-point timing for.
            point_timestamps = np.zeros(pts.shape[0], dtype=np.float64)
            odom.register_frame(pts, point_timestamps)
            poses.append(odom.last_pose.copy())
            processed += 1
        runtime_s = time.perf_counter() - t0

        if not poses:
            raise ValueError(
                f"KISS-ICP processed 0 frames. Check that {request.frame_paths[0]} "
                "contains non-empty scans."
            )

        poses_arr = np.stack(poses, axis=0).astype(np.float64, copy=False)
        # The trajectory only has timestamps for the frames we actually processed,
        # which may be fewer than the input set if some scans were empty.
        timestamps_kept = timestamps_s[:processed]

        map_points = np.asarray(odom.local_map.point_cloud(), dtype=np.float64)
        if map_points.ndim != 2 or map_points.shape[1] != 3:
            map_points = map_points.reshape(-1, 3)

        metadata: dict[str, Any] = {
            "kiss_icp": {
                "max_range_m": max_range,
                "voxel_size_m": float(config.mapping.voxel_size),
                "deskew": bool(config.data.deskew),
                "max_points_per_voxel": int(config.mapping.max_points_per_voxel),
            }
        }

        return SlamRunResult(
            driver=self.name,
            poses=poses_arr,
            timestamps_s=timestamps_kept.astype(np.float64, copy=False),
            map_points=map_points,
            runtime_s=runtime_s,
            frames_processed=processed,
            metadata=metadata,
        )


__all__ = ["KissICPSlamDriver"]
