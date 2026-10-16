"""Frames for NCLT velodyne_sync data, from nclt.pdf Table 4 and section 7.3.

Synchronized points are motion compensated in the Segway body frame. The
existing point reader flips y/z to z up; no LiDAR mounting transform remains.
"""

import numpy as np

FRAME_ID = "nclt_body_z_up"
FLIP = np.diag([1., -1., -1., 1.])
# Native IMU axes coincide with the native body axes; its origin is offset.
BODY_FROM_IMU = np.eye(4)
BODY_FROM_IMU[:3, 3] = [-.11, -.18, -.71]
BODY_UP_FROM_IMU = FLIP @ BODY_FROM_IMU


def body_pose_z_up(position, rpy):
    """NCLT world/body xyz+rpy -> z-up world/body transform, at the axle."""
    position, rpy = np.asarray(position, dtype=np.float64), np.asarray(rpy, dtype=np.float64)
    if position.shape != (3,) or rpy.shape != (3,) or not np.isfinite([position, rpy]).all():
        raise ValueError("NCLT pose needs finite xyz and roll/pitch/yaw triples")
    sr, sp, sy = np.sin(rpy)
    cr, cp, cy = np.cos(rpy)
    pose = np.eye(4)
    pose[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                   [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                   [-sp, cp * sr, cp * cr]]
    pose[:3, 3] = position
    return FLIP @ pose @ FLIP


def imu_pose_from_body_z_up(body_pose):
    """Z-up world/body pose -> z-up world/native-IMU pose, including lever arm.

    The returned orientation keeps native IMU axes, matching ms25.csv vectors.
    """
    pose = np.asarray(body_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("NCLT body pose needs a finite 4x4 rigid transform")
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-12, rtol=0.)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.)
            or abs(np.linalg.det(rotation) - 1.) > 1e-6):
        raise ValueError("NCLT body pose must be a proper rigid transform")
    return pose @ BODY_UP_FROM_IMU


def body_up_from_rpy(roll, pitch):
    """World up in the z-up body scan frame; yaw does not affect this vector."""
    pose = body_pose_z_up([0., 0., 0.], [roll, pitch, 0.])
    return pose[:3, :3].T @ np.array([0., 0., 1.])
