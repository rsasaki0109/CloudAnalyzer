"""Sentinel SLAM driver: emit identity poses and concatenate input frames.

Intentionally bad — it ignores the registration problem entirely. Its only
purpose is to give the slam_run slice evaluator a floor: any real driver
should beat this on a curved trajectory by a wide margin. If a "real"
driver is no better than identity-passthrough, the experiment harness
flags the regression.
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


class IdentityPassthroughSlamDriver:
    """Zero-motion sentinel — every pose is identity, the map is just the
    union of input scans."""

    name: str = "identity_passthrough"

    def run(self, request: SlamRunRequest) -> SlamRunResult:
        frame_paths = (
            request.frame_paths[: request.max_frames]
            if request.max_frames is not None
            else request.frame_paths
        )

        if request.timestamps_s is not None:
            timestamps_s = np.asarray(request.timestamps_s, dtype=np.float64)[
                : len(frame_paths)
            ]
        else:
            timestamps_s = np.arange(len(frame_paths), dtype=np.float64) * float(
                request.frame_period_s
            )

        all_points: list[np.ndarray] = []
        processed = 0
        t0 = time.perf_counter()
        for path in frame_paths:
            pts = load_frame(path)
            if pts.shape[0] == 0:
                continue
            if request.max_range_m is not None:
                radii = np.linalg.norm(pts, axis=1)
                pts = pts[radii <= float(request.max_range_m)]
                if pts.shape[0] == 0:
                    continue
            all_points.append(pts)
            processed += 1
        runtime_s = time.perf_counter() - t0

        if not all_points:
            raise ValueError(
                f"identity_passthrough processed 0 frames. Check that "
                f"{request.frame_paths[0]} contains non-empty scans."
            )

        identity = np.eye(4, dtype=np.float64)
        poses = np.broadcast_to(identity, (processed, 4, 4)).copy()
        timestamps_kept = timestamps_s[:processed]
        map_points = np.concatenate(all_points, axis=0).astype(np.float64, copy=False)

        metadata: dict[str, Any] = {
            "identity_passthrough": {
                "note": "sentinel driver; identity poses, concatenated scans",
            }
        }

        return SlamRunResult(
            driver=self.name,
            poses=poses,
            timestamps_s=timestamps_kept.astype(np.float64, copy=False),
            map_points=map_points,
            runtime_s=runtime_s,
            frames_processed=processed,
            metadata=metadata,
        )


__all__ = ["IdentityPassthroughSlamDriver"]
