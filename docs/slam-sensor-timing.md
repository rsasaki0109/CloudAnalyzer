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
