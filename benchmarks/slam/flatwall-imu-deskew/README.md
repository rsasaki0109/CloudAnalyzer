# Fixed-anchor IMU deskew and indexed CPU mapping

`ImuSeries.trajectory()` constructs an immutable nominal motion snapshot from
an explicit initial IMU-to-world pose, world velocity/gravity and IMU biases.
`ImuTrajectory.deskew()` maps every LiDAR point into LiDAR coordinates at a
declared reference timestamp, including the calibrated rigid sensor lever arm.
The source series acceleration scale is applied before SI acceleration bias.
See the [API and coordinate contract](../../../docs/slam-imu-trajectory.md).

The trajectory retains covered IMU knots and evaluates each point independently.
It uses held gyro exponentials and quadratic translation from fixed segment
boundaries, matching the existing nominal preintegration model. Integer time
differences are converted to small floating durations. Initial world translation
is kept separate so it cancels before point arithmetic. No per-point pose
matrix or covariance cache is retained.

The serial path is the default. `parallel=True` uses an indexed CPU map with
the same scalar point arithmetic and output order. Python copies input buffers
before releasing the GIL and returns a new array. The API rejects uncovered
times, invalid states/calibration, nonfinite coordinates and count/shape errors;
it performs no sorting, extrapolation or silent point removal.

## Frozen real-input comparison

The baseline native build is from parent commit
`0c9eb931bbb8ec8f23c4eaeea1cf035fccd0d092`. The same audit controller runs
against old and new builds. It independently decodes the original ROS 1 IMU
messages and standard PointCloud2 fields, constructs SciPy/NumPy nominal motion,
and checks all covered points. NCLT MCAP controls use an explicitly supplied
standard ROS2_HUMBLE definition because their embedded schema text is empty.

The numerical policy is deliberately explicit: integer Header plus `UINT32 t`
nanosecond offsets; first IMU time as anchor; identity initial pose; zero velocity
and bias; world gravity `[0,0,-9.80665]`; acceleration scale `9.80665`; identity
IMU-from-LiDAR transform; scan maximum point time as reference. These declared
test inputs do not estimate physical initialization, bias or calibration.

| Check | Result |
| --- | --- |
| Source IMU messages | All 3,755 retain raw fields; source numeric fields independently decoded |
| Existing scan outputs | All 660 scans / 7,175,872 finite points retain old digests |
| Integer point timestamps | All 4,464,000 flat-wall values retain their old digest |
| Compensated points | All 185 covered scans / 4,440,000 points match independent motion within tolerance |
| Native vs independent coordinates | Maximum absolute error 8.03e-13 |
| Motion knots vs independent states | Maximum rotation 8.88e-15, velocity 5.40e-13, position 4.21e-12 |
| Serial vs four-thread output | Every output bit matches; source ordering and caller arrays retained |
| First scan | API rejects its uncovered point at index zero; audit excludes the entire scan |

The first scan has 632 points before the first IMU header. No extrapolation or
clock shift is fitted. Eleven decreasing source-time steps remain in physical
point order. Existing full preintegrations at fixed points in eight scans also
retain identical digests across builds. Independent and native deskew hashes
can differ because matrix operation ordering differs; their coordinate error
is checked numerically. The two native CPU paths match bit for bit.

Separate coordinate fixtures cover nonidentity initial rotations, nonzero biases,
sensor rotation/lever arm, clipped endpoints, large world origins, copied arrays
and lifetimes, invalid inputs, and four-worker deterministic error ordering.
They are numerical tests, not synthetic trajectory accuracy benchmarks.

## API timing on fixed real scans

The measurement uses scans `[1,2,9,17,49,93,139,185]`, containing 192,000 points
per eight-scan batch. It requests four Rayon workers, warms both paths for
three pairs, then measures seven pairs with alternating order. Timing includes
Python input/output copies and compensation; recording I/O and trajectory
construction occur beforehand. No source reads, hashing, builds or tests occur
inside timed calls. Input and output hashes are checked outside those calls.

| API path | Median per eight-scan batch | Range across seven samples |
| --- | ---: | ---: |
| Serial | 21.60 ms | 19.96–24.14 ms |
| Four requested CPU workers | 9.46 ms | 8.96–11.11 ms |

The ratio of medians is **2.28x** on this host and fixed workload. These are
prewarmed API measurements, with identical results. They do not measure complete
SLAM, trajectory accuracy, GPU execution or a universal optimal thread count.
The default remains serial so callers can choose workers alongside other work.
Exact samples, native/controller/source hashes and scope are in `performance/`.

## Reproduction

Use NumPy, SciPy, `rosbags`, and separate old/new native Python builds. Set
`RAYON_NUM_THREADS=4` before starting each fresh Python process, and keep the
controllers/helpers unchanged. Choose new output directories:

```bash
python scripts/audit_flatwall_imu_deskew.py \
  --bag /path/to/flatwall_01.bag --out /path/to/new/baseline --baseline \
  --control nclt-april web/public/samples/nclt-2012-04-29.mcap /velodyne_points \
  --control nclt-june web/public/samples/nclt-2012-06-15.mcap /velodyne_points
# Switch to the native extension built from this revision.
python scripts/audit_flatwall_imu_deskew.py \
  --bag /path/to/flatwall_01.bag --out /path/to/new/native \
  --comparison /path/to/new/baseline \
  --control nclt-april web/public/samples/nclt-2012-04-29.mcap /velodyne_points \
  --control nclt-june web/public/samples/nclt-2012-06-15.mcap /velodyne_points
# After functional verification, with other local work stopped:
python scripts/measure_flatwall_imu_deskew.py \
  --bag /path/to/flatwall_01.bag --audit /path/to/new/native \
  --out /path/to/new/timing
```

Frozen protocols and receipts are exact original byte copies. Source recordings
and prior results are not rewritten. No reference trajectory was parsed, no
LiDAR-inertial optimizer run, and no clock/calibration parameters fitted. When
connecting compensated points to odometry, disable ordinary constant-motion
deskew and align the estimated pose's timestamp with the explicit reference.

Dataset: Kenji Koide (AIST), [LiDAR Degeneration Test Dataset, version
2](https://zenodo.org/records/7641866), DOI
[10.5281/zenodo.7641866](https://doi.org/10.5281/zenodo.7641866),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
NCLT-derived inputs retain the University of Michigan
[attribution and ODbL/DBCL terms](../nclt-imu-frontend/README.md).
Original flat-wall recordings and reference poses are not redistributed here.
Software remains under the repository's software license.
