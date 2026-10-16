# Inertial predictions for LiDAR registration

`LidarOdometry.register(points, relative_prediction=delta)` accepts a finite
proper rigid 4x4 transform mapping the current scan into the previous scan's
frame. Its ICP initial pose is `last_frontend_pose @ delta`. The first scan
anchors identity and rejects an external prediction. Omitting the keyword
keeps the constant-motion behavior. Invalid predictions fail before poses,
map points or the adaptive correspondence threshold change.

The override changes ICP initialization only. Deskew retains its existing
constant-motion model; the caller must handle sweep timing separately.
`last_iterations()` reports the last registration's Gauss-Newton steps, with
zero when there was no alignment. It does not certify registration quality.

`BagReader.imu_series(topic=...)` decodes one complete IMU topic into an
immutable `ImuSeries`. It validates exact header-time ordering once. Its
`preintegrate(t0, t1, ...)` and `factor(t0, t1, ...)` methods search the endpoints
and integrate only overlapping held-sample intervals, without reopening the
recording. Measurements, covariance, bias Jacobians, frame validation and the
original gap policy match the existing reader methods. A clipped interval
cannot conceal a long original sampling gap. Timestamp and frame accessors
return copies. The whole topic stays in memory; this is intended for bounded
recordings and is not a streaming reader.

The opt-in diagnostic in
[`replay_nclt_imu_frontend.py`](../scripts/replay_nclt_imu_frontend.py) connects
these APIs to the [four-state inertial window](slam-imu-marginalization.md).
For each supported interval, it predicts from the previous optimized IMU
pose, world velocity and bias before registering the current points. It
converts both initial and predicted poses through the IMU mounting transform,
including its lever arm, then passes their relative body motion to ICP.
The first valid interval initializes velocity after LiDAR registration and
does not supply an IMU prediction. An original IMU gap clears the entire
window, including its marginal prior; the next valid interval bootstraps again.

The frontend pose returned by ICP builds the local map and is emitted once
per scan. Backend corrections are saved separately for diagnosis. This avoids
moving already inserted map points when an optimized pose changes. Both
comparison arms run the same window and fixed noise settings; the assisted
arm supplies its relative predictions while the baseline uses the ordinary
initialization. NCLT synchronized scans are already motion compensated, so
this experiment leaves deskew disabled.

```powershell
python scripts/replay_nclt_imu_frontend.py --lidar scans.mcap --imu raw-imu `
  --euler ms25_euler.csv --out new-frontend-replay
python scripts/evaluate_nclt_imu_frontend.py --replay new-frontend-replay `
  --ground-truth reference-raw-rows.csv --out new-frontend-score
python benchmarks/slam/nclt-imu-frontend/verify.py
```

Estimation does not read reference poses. Outputs and input hashes are frozen
before a separate full-trajectory evaluation, including startup and all IMU
fallback transitions. The samples are the existing NCLT April and June cohort,
so this experiment does not establish held-out accuracy. Noise densities and
pose-prior weights remain diagnostic; shared-map correlations, bias diffusion
within an integration interval and fixed-gravity uncertainty are not calibrated.
The default LiDAR driver remains unchanged.

The [frozen real-data comparison](../benchmarks/slam/nclt-imu-frontend/README.md)
reports fewer ICP steps (23.0% and 5.3%) with nearly unchanged trajectory
accuracy. Some worst-case errors increase, so frontend feedback remains
opt-in. Separately, indexing the real IMU recordings reduced factor generation
time by 74.4–111.3 times while retaining bit-identical measurements. Those
measurements cover factor construction rather than complete SLAM runtime.
