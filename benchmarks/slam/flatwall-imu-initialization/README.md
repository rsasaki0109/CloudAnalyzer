# Metric LiDAR/IMU startup: rejected real prefix

`ImuSeries.initialize_lidar()` estimates gyro bias, world gravity and velocities
from metric LiDAR poses and covered IMU measurements. Acceleration bias remains
an explicit fixed input. Independent interval preintegration can run on CPU
workers, with ordered accumulation and deterministic first-error selection.

The fixed real startup prefix below is **rejected** by both the independent
calculation and native implementation. No fitted state is adopted, and the
predeclared limits are unchanged. This is a consistency and rejection check;
it does not establish successful physical calibration or trajectory accuracy.

## Frozen experiment

| Input or check | Result |
| --- | --- |
| Recording | Original `flatwall_01.bag`, SHA-256 `4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6`. |
| Startup selection | First 64 fully covered scans, source indices 1 through 64. |
| Pose references | Maximum checked integer point timestamp of each scan. |
| Metric poses | Pure LiDAR ICP, ordinary deskew disabled, no IMU prediction; the new native reuses the exact saved baseline poses. |
| LiDAR settings | Range 1.5–80 m, map voxel 1 m, 20 map points per voxel, diagnostics enabled. |
| Declared IMU inputs | Source acceleration multiplier 9.80665, identity IMU-from-LiDAR transform, initial gyro bias zero and fixed acceleration bias zero. |
| Gravity norm error | 0.213961 m/s²; below the frozen 1.0 m/s² limit. |
| Rotation residual RMS | 0.0593397 rad; exceeds the frozen 0.05 rad limit. |
| Velocity equation RMS | 1.57616 m/s; exceeds the frozen 0.2 m/s limit. |
| Native result | Both serial and parallel modes reject with the same rotation/velocity limit errors; caller arrays remain unchanged. |

The other frozen settings are gravity magnitude 9.80665 m/s², maximum
componentwise gyro-bias change 0.2 rad/s, 15 iterations, maximum IMU gap 0.05 s,
and minimum gyro sensitivity singular ratio 1e-6. These are declared experiment
inputs, not fitted sensor calibration. The residuals do not identify the cause
of this prefix's inconsistency. Rejected candidate gravity, bias and velocity
values in the evidence must not be used as an accepted initialization.

The independent calculation decodes standard ROS IMU messages, fits gyro bias
using SciPy finite differences and exact held-measurement reintegration, then
uses NumPy displacement-triple equations for gravity and velocities. It uses no
reported IMU orientation or stationary-start assumption. The native method uses
analytical SO(3)/preintegration derivatives and exact reintegration at each
accepted bias update. Neither method normalizes gravity to force acceptance.
All fitting information ends at the last selected startup reference; the batch
cannot be used causally before that reference.

## Validation and existing behavior

Local release-native validation passes 447 Python tests, including 65 new cases
covering independent moving dynamics, full sensor extrinsics, source scale,
fixed acceleration bias, gyro reintegration, array ownership/strides, large map
origins and rejection contracts. Successful numerical fixtures are contract
checks, not real-world accuracy evidence. Focused core checks pass 6 default
and 7 parallel/sqlite tests, including an explicitly verified four-worker pool.
Formatting, all-target Clippy, WASM checking, stub mypy and targeted Ruff pass.

The unchanged component controller verifies all 4,440,000 covered compensated
points against the independent model and confirms serial/parallel bit equality.
The compensated-coordinate digest remains
`a31223af0ab27fe527203ffa1887310d022295e01e717eff6f4eb65e16f9b53b`.
All fields over the previous 660 scans / 7,175,872 points, raw IMU measurements,
integer point times and fixed old preintegrations retain their prior digests.

## Evidence and reproduction

The [preimplementation protocol](slice-protocol.json) freezes the startup and
acceptance choices. The baseline [protocol](before/protocol.json),
[receipt](before/receipt.json), [independent rejected fit](before/independent.json)
and [LiDAR diagnostics](before/lidar-diagnostics.json) record the fixed pose
source. The final-native [protocol](after/protocol.json) and
[receipt](after/receipt.json) reuse those same poses. Their compressed timestamp
and pose array file stays private and is identified by its exact output hash in
the baseline receipt. The [component regression](deskew-regression/receipt.json)
has its own [protocol](deskew-regression/protocol.json). Raw recordings are not
redistributed here.

Install NumPy, SciPy and rosbags in one environment and select old/new native
snapshots via `PYTHONPATH`. Use fresh processes with `RAYON_NUM_THREADS=4` and
`OPENBLAS_NUM_THREADS=1`. With the old snapshot, run:

```sh
python scripts/audit_flatwall_imu_initialization.py \
  --bag /path/to/flatwall_01.bag --out /path/to/fresh/before --baseline
```

With the new snapshot, run:

```sh
python scripts/audit_flatwall_imu_initialization.py \
  --bag /path/to/flatwall_01.bag --out /path/to/fresh/after \
  --comparison /path/to/fresh/before
```

Preserve successful controllers, protocols, snapshots and receipts byte for
byte. Reproduction on another host generates its own provenance and baseline;
do not substitute a newly computed ICP trajectory into this saved experiment.
Use the unchanged [deskew controller](../../../scripts/audit_flatwall_imu_deskew.py)
and both NCLT controls as described in its
[packet](../flatwall-imu-deskew/README.md) for the component regression.

This slice contains no timing run, GPU execution, reference-pose parsing or
coupled LiDAR/IMU optimization. Startup pose consistency, accelerometer
calibration, observability and later velocity/bias feedback remain work for the
estimator. See the [API contract](../../../docs/slam-imu-initialization.md).

Dataset: Kenji Koide (AIST), [LiDAR Degeneration Test Dataset, version
2](https://zenodo.org/records/7641866), DOI
[10.5281/zenodo.7641866](https://doi.org/10.5281/zenodo.7641866),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
NCLT-derived controls retain the University of Michigan
[attribution and ODbL/DBCL terms](../nclt-imu-frontend/README.md).
Software remains under the repository's software license.
