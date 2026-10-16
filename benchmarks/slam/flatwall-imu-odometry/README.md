# Nominal IMU deskew to ICP and map insertion

`ImuLidarOdometry` registers complete, IMU-compensated real scans with an explicit
integer pose reference timestamp. The same compensated coordinates feed ICP
and map insertion. It owns an odometry with ordinary constant-motion deskew
disabled, shares immutable nominal motion knots, and returns poses/insertion
points in the first accepted reference LiDAR frame. A copied alignment transform
makes the relation to the nominal IMU world explicit.

This packet checks integration and coordinates with declared numerical initial
state. It is not a physical trajectory-accuracy benchmark. No initial attitude,
velocity, gravity, biases or sensor calibration were fitted, and no reference
poses were parsed. ICP convergence counts below do not certify alignment.

## Recorded checks

| Check | Result |
| --- | --- |
| Input | Original `flatwall_01.bag`, 186 scans / 4,464,000 points. |
| Rejected | Scan 0, with 632 points before IMU coverage; no frontend state changed. |
| Accepted | All 185 covered scans / 4,440,000 points in source order. |
| Pose reference | Maximum checked integer point timestamp of every accepted scan; all emitted references match exactly. |
| IMU prediction | Every relative reference transform agrees with independent standard ROS decoding and SciPy/NumPy held-measurement dynamics; maximum absolute error 3.31e-13. |
| Compensation | All 4,440,000 coordinates checked independently; maximum absolute error 8.03e-13. Native serial/parallel output bits remain equal in the component regression. |
| Map insertion | All 50,975 insertion points match independent range filtering, ordered half-voxel thinning and the final ICP pose; maximum absolute error 1.43e-14. |
| World alignment | First reference LiDAR pose agrees with the independent nominal world pose; maximum absolute error 1.34e-15. |
| ICP termination | One initial empty map, 184 converged steps, 5,475 iterations total. |
| Legacy regression | All fields over 660 scans / 7,175,872 points, raw IMU fields, integer point times and fixed old preintegrations retain their prior digests. |

Initial numerical state is identity IMU pose and zero velocity/bias at the first
IMU header, gravity `[0,0,-9.80665]`, source acceleration scale `9.80665`, and
identity IMU-from-LiDAR transform. Settings are 1.5–80 m reference-frame range,
1 m map voxel, 20 points per local-map voxel, diagnostics enabled, parallel point
compensation and four requested CPU workers. These settings are frozen in the
protocol. They establish no physical calibration or initialization.

The map check first verifies compensated coordinates against the independent
model. It then independently selects insertion points from the native-compensated
coordinates, avoiding an ambiguous discrete voxel choice at floating-point cell
boundaries. These points are handed to the internal map before its capacity and
distance pruning. The saved output map accumulates those insertion chunks; it is
not the final retained contents of the internal local map.

The exact native compensated-point digest remains:

```text
a31223af0ab27fe527203ffa1887310d022295e01e717eff6f4eb65e16f9b53b
```

Local release-native validation passes all 382 Python tests, including 46 new
reference-coordinate, full sensor-transform, real ICP contract, state-ownership,
noncontiguous-buffer, large-origin, filter/thinning and atomic-error cases. The
contract fixtures are not trajectory benchmarks. Core trajectory tests pass
8 cases by default and 9 with parallel/sqlite, including two new full-extrinsic
relative-pose checks. Formatting, Clippy, WASM, stub mypy and targeted Ruff pass.

## Evidence and reproduction

The unchanged [`audit_flatwall_imu_deskew.py`](../../../scripts/audit_flatwall_imu_deskew.py)
controller supplies the [component regression](deskew-regression/receipt.json).
The new [`audit_flatwall_imu_odometry.py`](../../../scripts/audit_flatwall_imu_odometry.py)
supplies the [integration protocol](integration/protocol.json),
[receipt](integration/receipt.json) and [per-scan diagnostics](integration/per-scan.json).
The private complete pose/reference/prediction arrays and output map are identified
by their hashes in that receipt. Raw recordings and output map coordinates are
not redistributed here.

Build or snapshot the old and new native packages on the same host. Install
NumPy, SciPy and rosbags in a single environment. In a fresh process with
`RAYON_NUM_THREADS=4`, use the old snapshot to run the component controller with
`--baseline`, then the new snapshot with `--comparison` on that fresh baseline.
Use both NCLT controls, as in the
[component packet](../flatwall-imu-deskew/README.md). This preserves exact
controller, helper, library and recording provenance on the reproduction host.
With the new snapshot selected via `PYTHONPATH`:

```sh
python scripts/audit_flatwall_imu_odometry.py \
  --bag /path/to/flatwall_01.bag \
  --component-audit /path/to/fresh/deskew-regression \
  --out /path/to/fresh/odometry-integration
```

Outputs must be fresh; successful protocols, controllers, native snapshots and
receipts are preserved byte for byte. Protocols identify all numerical inputs,
the native extension, decoder/backend sources and controller/helper hashes.
This slice contains no timing run, GPU execution, coupled LiDAR/IMU optimization
or accuracy comparison. See the [API contract](../../../docs/slam-imu-odometry.md)
for pose time, frames, failures and remaining estimation scope.

Dataset: Kenji Koide (AIST), [LiDAR Degeneration Test Dataset, version
2](https://zenodo.org/records/7641866), DOI
[10.5281/zenodo.7641866](https://doi.org/10.5281/zenodo.7641866),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
NCLT-derived controls retain the University of Michigan
[attribution and ODbL/DBCL terms](../nclt-imu-frontend/README.md).
Software remains under the repository's software license.
