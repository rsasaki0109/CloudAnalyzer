# NCLT synchronized scan and IMU frames

Use `scripts/prepare_nclt_body_reference.py` for new reference experiments on
`velodyne_sync`-derived scans. It keeps the reference pose at the Segway axle,
matching the synchronized points, and converts world/body axes to z up.

The [official NCLT documentation](https://robots.engin.umich.edu/nclt/nclt.pdf)
states that `velodyne_sync` contains a preceding 0.1-second revolution, already
motion compensated and expressed in the Segway **body frame** (section 7.3,
pages 15–16). The native body axes point forward, right and down (page 8).
These synchronized points must not receive another Velodyne mounting transform.

`nclt_body_frames.py` uses `F = diag(1,-1,-1,1)`:

```text
point_body_z_up = F.rotation * point_body_native
pose_world_z_up_body_z_up = F * pose_world_native_body_native * F
```

The helper's frame name is `nclt_body_z_up`. Historical bundled recordings carry
the legacy `velodyne` label despite using body coordinates with y/z flipped.
Check preparation provenance along with `Scan.frame_id`; a label alone does
not supply the calibration or coordinate history.

## Preparing a fresh reference

The input mapping job must use z-up body-coordinate synchronized points and a
hashed `source_motion` trajectory. The estimated trajectory selects reference
times; its estimated positions and orientations are not used to produce truth.

```powershell
python scripts/prepare_nclt_body_reference.py `
  --job path/to/body-pointcloud-job `
  --ground-truth path/to/2012-04-29.csv `
  --out path/to/new-body-reference `
  --source-url https://robots.engin.umich.edu/nclt/
```

The tool writes `reference.tum`, selected original CSV rows, provenance and a
preparation receipt. The receipt includes hashes of the job, motion trajectory,
original CSV, preparation script and frame helper. It refuses an existing output
directory and output within a mapping job. Input estimates, maps and original
reference files are not modified.

Historical preparation scripts and successful benchmark packets retain their
original conventions and hashes for reproducibility. Their references used an
additional mounting transform; their frozen accuracy numbers must not be used
as body-frame results. New body-frame comparisons need distinct cohorts and
receipts. This change does not rerun or replace those experiments and does not
establish improved odometry accuracy.

## Pose at the native IMU origin

For the Microstrain IMU, Table 4 gives body-relative position
`[-0.11,-0.18,-0.71]` metres and zero relative roll/pitch/yaw. The frame helper
retains native IMU axes to match the `ms25.csv` vectors:

```python
from nclt_body_frames import imu_pose_from_body_z_up

pose_world_imu = imu_pose_from_body_z_up(pose_world_body_z_up)
```

With the scripts directory on Python's import path, this computes
`pose_world_body_z_up * F * pose_body_native_imu_native`, including the lever arm.
It converts a pose; velocity initialization also needs the rotational lever-arm
term. Noise calibration, residual clock offsets, estimator initialization and
raw LiDAR/IMU fusion remain separate work.
