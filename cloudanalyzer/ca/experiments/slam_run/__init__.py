"""Experimental SLAM drivers compared by the slam_run slice evaluator.

Two drivers participate:

- :class:`KissICPSlamDriver` — the adopted real driver, wrapping the
  ``kiss-icp`` package. Also re-exported from ``ca.core.slam_run`` so the
  CLI keeps depending only on core.
- :class:`IdentityPassthroughSlamDriver` — sentinel that returns identity
  poses and concatenates input frames as the "map". It is intentionally
  bad; its only job is to prove the harness runs and to set the floor for
  ``evaluate.py``.
"""

from ca.experiments.slam_run.identity_passthrough import IdentityPassthroughSlamDriver
from ca.experiments.slam_run.kiss_icp_driver import KissICPSlamDriver


def get_slam_run_drivers() -> list:
    """Return the concrete drivers compared by the slice evaluator."""

    return [KissICPSlamDriver(), IdentityPassthroughSlamDriver()]


__all__ = [
    "IdentityPassthroughSlamDriver",
    "KissICPSlamDriver",
    "get_slam_run_drivers",
]
