# NCLT LiDAR-pose and raw-IMU pilot

`scripts/pilot_nclt_imu_window.py` runs existing native LiDAR odometry on actual
synchronized scans, then combines its estimated poses with original raw IMU
channels in separate bounded windows. It accepts no reference trajectory.
`scripts/evaluate_nclt_imu_pilot.py` subsequently scores the frozen outputs.
These are offline experimental tools; production LiDAR odometry keeps its
existing behavior.

## Inputs and execution

Build the current `cloudanalyzer_core` Python extension and place it on the
Python path. NumPy, SciPy and the `ca` package are also required. Supply:

- `velodyne_sync`-derived body-z-up scans at the Segway axle, including exact
  sensor header times. The bundled NCLT clips satisfy this coordinate convention
  despite their legacy `velodyne` frame label. Do not deskew these scans again.
- A ROS recording with original `ms25.csv` acceleration in m/s² and angular
  velocity in rad/s, timestamped from integer UTIME microseconds. Vectors must
  remain in native IMU axes and use frame `nclt_ms25_native`. The default topic
  is `/ms25/raw`. Missing gyro/acceleration is rejected by the native factor.
  The bundled orientation-only `/imu` channel cannot supply these measurements.
- Original `ms25_euler.csv`, with integer UTIME microseconds and roll, pitch,
  heading in radians. Only the last roll/pitch at or before the first scan,
  at most 50 ms old, initializes gravity. Heading and ground truth are unused.

See [NCLT coordinate preparation](slam-nclt-body-frames.md) for the documented
body/IMU axes and mounting translation. The original NCLT data and derived
trajectories retain NCLT's ODbL/DBCL terms; cite the
[NCLT dataset](https://robots.engin.umich.edu/nclt/) when sharing results.

```powershell
$env:RAYON_NUM_THREADS = '1'
python scripts/pilot_nclt_imu_window.py `
  --lidar web/public/samples/nclt-2012-04-29.mcap `
  --imu path/to/original-ms25-recording `
  --euler path/to/ms25_euler.csv --out runs/new-april-pilot

python scripts/evaluate_nclt_imu_pilot.py `
  --pilot runs/new-april-pilot --ground-truth path/to/original-body-rows.csv `
  --out runs/new-april-evaluation
```

Each output directory must be new. The estimator hashes its input recording,
Euler CSV, controller, frame helper and native extension before and after the
run. NPZ arrays preserve integer nanoseconds; TUM timestamps are formatted
directly from those integers. The evaluator verifies output hashes, writes
elsewhere and never calls the estimator. Keep successful cohorts unchanged;
use fresh directories and receipts for another experiment.

## Fixed diagnostic protocol

LiDAR odometry uses min/max range 1.5/80 m, map voxel 1 m, 20 map points per
voxel and no deskew. All scans contribute to this baseline, including those
outside fusion coverage. Their estimated poses are converted to the native
IMU origin including its lever arm. Second-order finite differences of these
positions initialize velocity, including the rotational lever-arm term. This
offline initialization uses neighboring future LiDAR estimates. Bias starts
at zero; gravity has magnitude 9.81 m/s² in the first scan's body world frame.

The original IMU sample gap is checked before clipping at scan endpoints.
Edges crossing gaps over 50 ms are excluded. Every remaining consecutive run
is split greedily into disjoint chunks of at most 16 states; chunks with fewer
than four states are excluded. No result is spliced into a continuous fused
trajectory. Coverage records excluded scan nodes, gap edges, unused valid
edges at chunk boundaries or short tails, and covered duration.

The model is fixed in `settings()`, before scoring:

| Parameter | Diagnostic value |
| --- | --- |
| Gyro / accelerometer measurement density | 0.02 rad/s/√Hz; 0.15 m/s²/√Hz |
| Gyro / accelerometer bias random walk | 0.005 (rad/s)/√s; 0.05 (m/s²)/√s |
| LiDAR pose rotation / position sigma | 0.02 rad; 0.1 m |
| Initial state sigmas, three axes each | rotation 0.02 rad; velocity 0.5 m/s; position 0.05 m; gyro bias 0.02 rad/s; accel bias 0.5 m/s² |
| Optimizer | at most 80 iterations; initial damping 0.001 |

The first node receives a full-state prior; later nodes receive LiDAR-derived
pose priors. These weights are not calibrated sensor or registration
uncertainties. Shared-map LiDAR estimates, reused velocity initialization and
clipped IMU samples at adjoining boundaries introduce correlations absent from
this diagonal-prior model. Within-interval bias diffusion and residual clock
offsets are also unmodeled. Fixed gravity and first-order bias correction can
limit accuracy. No automatic bias reintegration is performed; every window
reports its correction magnitude, costs and convergence status.

## Evaluation and interpretation

Use original NCLT body ground-truth CSV rows, before any historical extra
LiDAR mounting transform. The evaluator applies only the z-up conversion,
interpolates position and uses SO(3) SLERP. Both bracketing reference rows
must lie within 50 ms of each scan; extrapolation is rejected. Reference
availability does not select fusion windows. NCLT ground truth uses SLAM/RTK
and odometry interpolation, so it is not statistically independent sensor truth.

The primary comparison pools consecutive SE(3) relative-pose errors over the
same within-window edges for LiDAR-only and fusion. It reports translation
RMSE in metres and rotation RMSE in degrees. Secondary ATE independently fits
a rigid position transform for each window, without scale; short windows can
hide drift under this fit. Window endpoint translation errors are also saved.
All selected windows, including nonconverged solves, remain in the comparison.
Original arrays and per-window errors remain available for inspection.

Lower optimizer cost alone does not establish better trajectory accuracy.
This pilot does not implement marginalization, continuous LIO, loop closure,
GPU registration or a GLIM comparison. Changes to model weights or coverage
need a new experiment and later independent validation before default adoption.
