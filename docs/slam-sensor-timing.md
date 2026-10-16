# LiDAR header time and frame identity

`BagReader.scans()` and PointCloud2 messages returned by `BagReader.messages()`
retain three read-only header properties:

| Property | Meaning |
| --- | --- |
| `stamp_ns` | Exact integer sensor header timestamp in nanoseconds. |
| `stamp` | Existing floating-point header timestamp in seconds. |
| `frame_id` | Header coordinate-frame name, including an empty name if supplied. |

These properties follow the ROS [PointCloud2 header](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/PointCloud2.msg)
and [two-integer Time definition](https://github.com/ros2/rcl_interfaces/blob/rolling/builtin_interfaces/msg/Time.msg).

```python
from cloudanalyzer_core import BagReader

reader = BagReader("recording.mcap")
for scan in reader.scans(topic="/points"):
    print(scan.stamp_ns, scan.frame_id, len(scan))
    points = scan.positions()
```

The same properties are available for ROS 1 bags, ROS 2 little- or big-endian
CDR in MCAP, rosbag2 SQLite files and recording folders. ROS 2 signed seconds
are preserved, including timestamps before the epoch; ROS 1 seconds are
unsigned. Nanoseconds outside `[0, 1_000_000_000)` are rejected.

`stamp_ns` is computed from integer header seconds and nanoseconds, without a
floating-point round trip. At large epoch timestamps, two scans one nanosecond
apart can have the same `stamp`. Recovering nanoseconds with
`int(scan.stamp * 1e9)` cannot restore the missing precision. The existing
`stamp` calculation and point/intensity/per-point-time arrays retain their
previous behavior. Header metadata remains available when all points are
filtered as nonfinite and the decoded cloud is empty.

The bag's recording timestamp is separate from the sensor header timestamp.
Iteration follows recording order; it does not reorder scans by their header
times. Callers must explicitly handle repeated or decreasing sensor times
before constructing a strictly increasing estimator window.

## Connecting to inertial intervals

The integer timestamps provide the boundaries needed by
[`BagReader.imu_factor()`](slam-imu-preintegration.md) and
[`ImuWindow`](slam-imu-window.md), whose factors require exact endpoint matches.
First establish the LiDAR/IMU clock relationship and the convention for the
LiDAR pose timestamp. The header alone does not say whether it represents the
start, middle or end of a sweep.

`frame_id` names the coordinates of the returned points. No coordinate transform
is applied, and the name supplies no extrinsic calibration. IMU poses and pose
observations supplied to `ImuWindow` must refer to the IMU origin, with the
calibrated LiDAR-to-IMU rotation and lever arm applied explicitly.

`Scan.time()` retains its existing normalized per-point sweep fractions in
`[0,1]`; it does not provide absolute point timestamps or a calibrated sweep
duration. This metadata addition supports subsequent timing/calibration work;
it does not by itself perform motion compensation or LiDAR-inertial fusion.

## Retaining a source point-time field

Request the exact field name when creating the iterator:

```python
for scan in reader.scans(topic="/livox/points", raw_time_field="t"):
    values = scan.raw_time()  # float64, aligned with scan.positions()
    print(scan.raw_time_field, scan.raw_time_datatype)  # "t", 6 (UINT32)
```

`BagReader.messages(topics, raw_time_field="t")` provides the same option for
its PointCloud2 messages. Other message types retain their existing behavior.
The default `raw_time_field=None` retains no raw-time array or metadata.

| Scan member | Meaning |
| --- | --- |
| `raw_time()` | A copied float64 array of the requested scalar source values, or `None` when not requested. |
| `raw_time_field` | Read-only exact source field name, or `None`. |
| `raw_time_datatype` | Read-only ROS PointField datatype code, or `None`. |

All standard PointField integer values (INT8 through UINT32) are exactly
representable in float64. FLOAT32 source values widen to float64, and FLOAT64
source values retain their precision. Signed zero, nonfinite values, constant
fields and the original point order are retained. Nonfinite XYZ points are
filtered as before, with the corresponding raw values removed. An empty
decoded scan retains the requested metadata and an empty array. Mutating a
returned array does not change the scan.

The field name and datatype do not establish a physical unit, a clock origin,
or whether a value is an offset or absolute time. Values are not resampled,
sorted, converted to nanoseconds or shifted by the header. A consumer must
declare these conventions and validate nonfinite values before constructing
inertial timestamps. Standard float64 timestamp precision cannot recover
information already lost by the original publisher.

An explicit request raises `ValueError` for a missing/duplicate field name,
unsupported datatype, nonscalar count, an out-of-point field extent, overlapping
rows or incomplete point records. Missing values are not replaced with zeros.
The existing `Scan.time()` continues to use its existing field selection and
normalization, independently of the requested raw field. Headers, XYZ and
intensity retain their existing behavior.

Retaining raw values extends the lifetime of a float64 buffer. If the requested
field shares the layout used for normalized time, the decoder moves its existing
temporary buffer into the scan. A different field requires a separate buffer.
Calling `raw_time()` copies the retained values into a NumPy array.

The [flat-wall raw-time audit](../benchmarks/slam/flatwall-raw-point-time/README.md)
contains frozen old/new native receipts and independent field comparisons.

## Declared integer point timestamps

After establishing the source field's units and clock origin, request a new
int64 array with both conventions explicitly supplied:

```python
scan = next(reader.scans("/livox/points", raw_time_field="t"))
# Only when the recording's convention is ns offsets from this integer Header:
point_times = scan.point_times_ns(unit="ns", origin="header")
```

`unit` must be `"s"`, `"ms"`, `"us"` or `"ns"`. `origin="header"` adds
`scan.stamp_ns`; `origin="absolute"` adds zero for values already expressed
from the intended clock origin. Both arguments are required keywords. Neither
the field name, datatype nor header establishes the correct convention.

Each exact represented binary source value is scaled to nanoseconds using
integer arithmetic, rounded to the nearest integer with ties to even, then
added to the integer origin. Rounding occurs before origin addition: for
example, a source offset of 0.5 ns rounds to zero even when the header is odd.
No intermediate float64 unit multiplication or epoch addition adds another
rounding. All standard PointField integers therefore convert exactly at these
unit scales. Float values retain the precision of their represented binary
value; this cannot recover precision lost by the publisher. A decimal value
near half a nanosecond can lie just above or below that boundary in binary.

`ValueError` identifies the finite-point index of a nonfinite source value or
a final timestamp outside signed int64. The range check follows integer origin
addition, so an offset outside int64 can still be accepted when its final
timestamp fits. A scan whose raw field was not explicitly retained also raises
`ValueError`; no timestamps are synthesized from normalized `time()` values.
An empty retained source returns an empty int64 array.

The array is newly allocated. Conversion preserves point order, including
repeated or decreasing times, and leaves the raw field, normalized fractions,
XYZ and header unchanged. No converted array is cached. Consumers must check
IMU coverage using the minimum and maximum point timestamps rather than
assuming that point times are sorted. This conversion does not calibrate
clocks, choose a scan pose timestamp or perform deskew.

The [flat-wall integer-time audit](../benchmarks/slam/flatwall-point-times-ns/README.md)
compares every point with independent integer arithmetic and a previous native
build under a declared ns-offset/header-origin policy.

With an explicitly initialized state and calibrated sensor transform, the
[nominal IMU trajectory](slam-imu-trajectory.md) consumes these timestamps for
order-preserving serial or CPU-parallel point compensation.

For the bundled NCLT lineage, see [synchronized body-frame conventions](slam-nclt-body-frames.md)
before interpreting the historical `velodyne` frame label or preparing new
reference trajectories.
