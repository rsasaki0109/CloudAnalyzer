"""Experimental SLAM drivers compared by the slam_run slice evaluator.

Four drivers participate:

- :class:`KissICPSlamDriver` — the adopted real driver, wrapping the
  ``kiss-icp`` package. Also re-exported from ``ca.core.slam_run`` so the
  CLI keeps depending only on core. Scan-to-map with constant-velocity
  prediction and adaptive max-correspondence-distance.
- :class:`KissSLAMSlamDriver` — wraps ``kiss-slam`` (KISS-ICP odometry +
  pose-graph optimization + MapClosures loop closure). Same scan-to-map
  inner loop as KISS-ICP with a PGO pass on top.
- :class:`SmallGICPSlamDriver` — wraps ``small_gicp`` (PyPI MIT).
  Scan-to-map VGICP using ``small_gicp.GaussianVoxelMap`` as the
  registration target; map output is scan-stitched + voxel-downsampled
  (the voxel map stores quantized centers and can't clear the Chamfer
  threshold against a dense reference on its own).
- :class:`IdentityPassthroughSlamDriver` — sentinel that returns identity
  poses and concatenates input frames as the "map". It is intentionally
  bad; its only job is to prove the harness runs and to set the floor for
  ``evaluate.py``.
"""

from ca.experiments.slam_run.identity_passthrough import IdentityPassthroughSlamDriver
from ca.experiments.slam_run.kiss_icp_driver import KissICPSlamDriver
from ca.experiments.slam_run.kiss_slam_driver import KissSLAMSlamDriver
from ca.experiments.slam_run.small_gicp_driver import SmallGICPSlamDriver


def get_slam_run_drivers() -> list:
    """Return the concrete drivers compared by the slice evaluator."""

    return [
        KissICPSlamDriver(),
        KissSLAMSlamDriver(),
        SmallGICPSlamDriver(),
        IdentityPassthroughSlamDriver(),
    ]


__all__ = [
    "IdentityPassthroughSlamDriver",
    "KissICPSlamDriver",
    "KissSLAMSlamDriver",
    "SmallGICPSlamDriver",
    "get_slam_run_drivers",
]
