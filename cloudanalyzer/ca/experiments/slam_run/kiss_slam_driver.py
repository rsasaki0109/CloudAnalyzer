"""KISS-SLAM driver for ``ca slam-run``.

KISS-SLAM extends KISS-ICP with on-line local-map graph construction and
final pose-graph optimization across optional loop closures. The wrapper
mirrors :class:`KissICPSlamDriver`: lazy import, deskew disabled by
default (the upstream KISS-ICP inside KISS-SLAM crashes when fed
zero-timestamps with deskew enabled), and the world-frame map is built by
transforming each ingested scan with its final optimized pose.

On short bounded trajectories (e.g. the bundled synthetic-figure8 suite
whose sensor stays within ~3.5 m of origin), KISS-SLAM only produces a
single local map — no loop closure is fired and the pose graph collapses
to KISS-ICP's odometry chain plus one round of optimization. KISS-SLAM
mostly justifies itself on longer sequences that drift past the local-map
splitting distance (default 100 m).
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


class KissSLAMSlamDriver:
    """Experimental SLAM driver — KISS-SLAM (KISS-ICP + pose-graph + LC).

    Not yet adopted: on the bake-off cases the slice evaluator currently
    runs, KISS-SLAM does not consistently outperform plain KISS-ICP
    because the trajectories are too short to fire loop closures. Kept in
    ``ca.experiments`` so the comparison stays visible in
    ``docs/experiments.md``.
    """

    name: str = "kiss_slam"

    def run(self, request: SlamRunRequest) -> SlamRunResult:
        try:
            from kiss_slam.config import load_config
            from kiss_slam.slam import KissSLAM
        except ImportError as exc:  # pragma: no cover - exercised via the CLI test
            raise ImportError(
                "kiss-slam is required for KissSLAMSlamDriver. "
                "Install with: pip install 'cloudanalyzer[slam]' "
                "or pip install kiss-slam"
            ) from exc

        config = load_config(None)
        # KISS-SLAM wraps KISS-ICP and rebuilds the KISSConfig from
        # config.odometry on every kiss_icp_config() call, so mutate the
        # nested odometry fields rather than the materialized KISSConfig.
        if request.max_range_m is not None:
            config.odometry.preprocessing.max_range = float(request.max_range_m)
        config.odometry.preprocessing.deskew = bool(request.deskew)
        if request.voxel_size_m is not None:
            config.odometry.mapping.voxel_size = float(request.voxel_size_m)
            config.local_mapper.voxel_size = float(request.voxel_size_m)

        slam = KissSLAM(config)

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

        scans: list[np.ndarray] = []
        processed = 0
        t0 = time.perf_counter()
        for path in frame_paths:
            pts = load_frame(path)
            if pts.shape[0] == 0:
                continue
            point_timestamps = np.zeros(pts.shape[0], dtype=np.float64)
            slam.process_scan(pts, point_timestamps)
            scans.append(pts)
            processed += 1

        if processed == 0:
            raise ValueError(
                f"KISS-SLAM processed 0 frames. Check that {request.frame_paths[0]} "
                "contains non-empty scans."
            )

        # Force-finalize the in-flight local map and discard the empty
        # trailing node that ``generate_new_node`` creates, mirroring the
        # upstream pipeline runner.
        slam.generate_new_node()
        slam.local_map_graph.erase_last_local_map()
        optimized_poses, _ = slam.fine_grained_optimization()
        poses_arr = np.asarray(optimized_poses, dtype=np.float64)
        if poses_arr.ndim != 3 or poses_arr.shape[1:] != (4, 4):
            poses_arr = poses_arr.reshape(-1, 4, 4)

        # Truncate / pad so #poses == #scans processed. KISS-SLAM emits one
        # pose per ingested scan in practice, but be defensive.
        n = min(poses_arr.shape[0], processed)
        poses_arr = poses_arr[:n]
        timestamps_kept = timestamps_s[:n]
        scans = scans[:n]

        # Build the world-frame map by transforming each input scan with
        # its optimized pose and voxel-downsampling once at the end.
        world_chunks: list[np.ndarray] = []
        for pose, scan in zip(poses_arr, scans):
            if scan.shape[0] == 0:
                continue
            R = pose[:3, :3]
            t = pose[:3, 3]
            world_chunks.append(scan @ R.T + t)
        if world_chunks:
            map_world = np.vstack(world_chunks)
            voxel = float(config.local_mapper.voxel_size)
            if voxel > 0 and map_world.shape[0] > 0:
                import open3d as o3d

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(map_world)
                pcd = pcd.voxel_down_sample(voxel_size=voxel)
                map_world = np.asarray(pcd.points, dtype=np.float64)
        else:
            map_world = np.zeros((0, 3), dtype=np.float64)

        runtime_s = time.perf_counter() - t0

        metadata: dict[str, Any] = {
            "kiss_slam": {
                "max_range_m": float(config.odometry.preprocessing.max_range),
                "voxel_size_m": float(config.local_mapper.voxel_size),
                "deskew": bool(config.odometry.preprocessing.deskew),
                "local_map_splitting_distance_m": float(
                    config.local_mapper.splitting_distance
                ),
                "closures_detected": int(len(slam.get_closures())),
            }
        }

        return SlamRunResult(
            driver=self.name,
            poses=poses_arr,
            timestamps_s=timestamps_kept.astype(np.float64, copy=False),
            map_points=map_world,
            runtime_s=runtime_s,
            frames_processed=n,
            metadata=metadata,
        )


__all__ = ["KissSLAMSlamDriver"]
