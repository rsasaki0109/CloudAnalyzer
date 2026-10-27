# mapping-check-glim

Inspect a ROS2 MCAP recording before attempting GLIM LiDAR/inertial processing.
This is a read-only diagnostic; it does not start Docker or SLAM.
Install the existing `cloudanalyzer[ros]` optional dependencies first.

```bash
ca mapping-check-glim recording.mcap \
  --points-topic /velodyne_points --imu-topic /imu > preflight.json
```

Topics are explicit and must match the intended GLIM configuration. The command
streams the selected messages through the `rosbags` decoder and reports:

- missing topics or incorrect ROS message types;
- IMU angular velocity and acceleration marked unavailable (covariance[0] = -1),
  and nonfinite measurement vectors;
- fewer than two messages, invalid nanoseconds or non-increasing header stamps;
- missing/changing frame IDs, malformed point layouts, missing XYZ fields or empty clouds;
- nonfinite XYZ counts, including dense-cloud contradictions and clouds with no finite points;
- whether the IMU header timestamp span covers the point cloud span;
- IMU intervals exceeding `--max-imu-gap` seconds (default 0.05);
- decoding errors and input changes detected by before/after SHA-256 inventories.

Exit code 0 means `basic_checks_passed`; exit code 1 means blocked or an input
error. A decoding failure leaves partial counts and blocks the result. A missing
optional dependency returns an installation hint. No output files are created
unless the caller redirects stdout. ROS2 bag directories are not yet supported
by this diagnostic, although the GLIM execution command accepts them.

Passing these checks does **not** verify units, clock alignment, sensor
extrinsics, per-point timestamps or GLIM settings.
Calibration remains unverified. `mapping-run-glim` invokes this diagnostic
before contacting Docker when both `--points-topic` and `--imu-topic` are supplied;
without them it explicitly records that preflight was not requested.

## PointCloud2 checks

Each decoded cloud is limited to 5,000,000 points, 256 MiB of point data and
256 fields. The diagnostic checks positive dimensions/strides, exact
`data` length (`row_step * height`), unique field names and field offsets/types/
counts within `point_step`. XYZ must be distinct, nonoverlapping scalar FLOAT32
or FLOAT64 fields. Other fields use standard ROS numeric types. Unsupported
layouts are blocked, not silently interpreted as another datatype.

XYZ inspection honors byte order, organized rows and row padding; it counts
records with any nonfinite coordinate once. A cloud with no finite XYZ records
is blocked. A cloud claiming `is_dense=true` with nonfinite XYZ is also blocked.
For `is_dense=false`, mixed valid/invalid records are permitted by the ROS
contract and produce a warning with counts. No filtering or repair is performed.
Inspect `point_checks` for checked-message, invalid-layout and point counts;
malformed layouts do not stop diagnosis of subsequent messages. Counts can be
partial if decoding itself fails, in which case the report is blocked.

These checks do not validate physical ranges, coordinate frames, timestamp
fields, sensor calibration or the downstream GLIM module's datatype support.

## IMU gap policy

Every consecutive positive IMU header interval is checked, including intervals
outside the point-cloud span. The default 50 ms limit follows the existing NCLT
pilot diagnostic policy; it is not a sensor calibration or a universal GLIM
requirement. An interval exactly equal to the limit passes. Choose an explicit
limit appropriate to the recording, for example `--max-imu-gap 0.02` for 20 ms.
The same option is accepted by `mapping-prepare-glim-recording`.

Limits must be finite, between 1 ns and 3600 seconds, and are rounded to integer
nanoseconds. The report's `imu_gaps.limit_ns` records the effective policy;
`max_observed_ns` is the largest positive interval (null when none was observed).
`count` includes every interval exceeding the limit, while `examples` retains
only the first 32 endpoint pairs and durations. Integer subtraction preserves
nanosecond precision at epoch timestamps. Repeated/backward timestamps remain
separate blocking errors; intervals are evaluated in recording log-time order.
On decoding failure these diagnostics can be partial and the report is blocked.

## Repository sample findings

Both `nclt-2012-04-29.mcap` and `nclt-2012-06-15.mcap` now decode with
`rosbags` 0.11.9. Their original empty ROS schema definitions caused a CDR
message-size mismatch. Complete standard ROS2 definitions were added without
changing any message payload, timestamp or sequence; the generator now embeds
these definitions too.

The preflight still correctly blocks both recordings: every IMU message marks
angular velocity and acceleration unavailable, and the IMU timestamp span does
not cover the complete point cloud span. These orientation-demo inputs must not
be treated as raw LiDAR/inertial recordings for a GLIM benchmark.

Tests include both repository MCAPs, serialized ROS2 IMU fixtures with provided
and unavailable angular velocity, and sensor/timing/CLI error cases. Real GLIM
estimation remains unverified; compatible raw sensor recordings and calibration
are still required, along with an installed runtime.

If points and raw inertial measurements are in separate MCAP files,
[`mapping-prepare-glim-recording`](mapping-prepare-glim-recording.md) copies the
selected channels into a new recording and retains this diagnostic in a receipt.
