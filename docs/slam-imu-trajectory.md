# Nominal IMU motion and point deskew

`ImuSeries.trajectory()` builds an immutable motion snapshot for a declared
interval. It integrates once at the covered IMU boundaries, then queries each
point independently. `ImuTrajectory.deskew()` returns points in the LiDAR frame
at an explicit reference timestamp, including rotation, translation and the
sensor lever arm. This component does not estimate its initial state or fuse
LiDAR observations. To connect compensation to ICP and map insertion, see
[IMU-compensated LiDAR registration](slam-imu-odometry.md).

The application supplies the initial IMU-to-world pose, world velocity, world
gravity and IMU-frame bias. Acceleration uses the scale configured when creating
the source series, applied before the SI acceleration bias is subtracted:

```python
import numpy as np
from cloudanalyzer_core import BagReader

reader = BagReader("recording.bag")
# Choose the source acceleration scale from the recording's sensor convention.
series = reader.imu_series("/imu", acceleration_scale=source_acceleration_scale)
motion = series.trajectory(
    start_ns, end_ns, initial_imu_to_world, initial_world_velocity,
    gravity=world_gravity, bias=imu_bias,
)

# After declaring the source field's units and clock origin:
scan = next(reader.scans("/points", raw_time_field="t"))
point_times = scan.point_times_ns(unit="ns", origin="header")
reference_ns = int(point_times.max())
points_at_reference = motion.deskew(
    scan.positions(), point_times, reference_ns,
    imu_from_lidar=calibrated_imu_from_lidar, parallel=True,
)
```

The symbols for initialization, scale and calibration in this example are
application inputs. The header/field name alone cannot establish units, clocks,
frame transforms, gravity, velocity or scan pose time. An empty scan needs an
explicit reference time rather than `point_times.max()`.

## Coordinates and input contract

| Input | Convention |
| --- | --- |
| `pose_i` | `(4,4)` IMU-to-world rigid pose at `start_ns`, translation in metres. |
| `velocity_i` | Three world-axis velocity components in m/s. |
| `gravity` | Required three world-axis components in m/s². |
| `bias` | Required six IMU-axis values: gyro xyz in rad/s, then acceleration xyz in m/s². |
| `imu_from_lidar` | Required `(4,4)` rigid transform taking LiDAR coordinates into IMU coordinates, including its translation in metres. |
| `points` | Finite `(N,3)` float64 LiDAR points at their individual acquisition times. |
| `point_times_ns` | `(N,)` int64 timestamps using the same clock as the trajectory. |
| `reference_ns` | Required integer timestamp for the output LiDAR frame. |

With `T_A_B` mapping coordinates from B to A, compensation is:

```text
T_world_lidar(t) = T_world_imu(t) * T_imu_lidar
p_lidar_reference = inverse(T_world_lidar(reference)) * T_world_lidar(t) * p_lidar
```

The output is in the LiDAR frame at the reference, rather than the IMU or world
frame. Rotations must be proper orthonormal matrices (the shared IMU validation
tolerance is 1e-6); Python homogeneous last rows must match `[0,0,0,1]` within
1e-12. Inputs are not normalized into valid poses automatically.

## Motion model and queries

The model matches the existing [held-measurement preintegration](slam-imu-preintegration.md):
gyro and specific force are held over `[t_k,t_{k+1})`. Each segment begins at
the fixed trajectory anchor or a covered source IMU boundary. Rotation advances
with the held gyro exponential. World acceleration uses the rotation at that
fixed segment start, plus the supplied world gravity. Velocity is linear and
position quadratic in the segment's elapsed time.

Timestamp subtraction occurs in integer nanoseconds before converting the small
duration to seconds. Partial queries use the same fixed left boundary; they do
not create a new integration partition per point. Directly integrating from
each point time to the reference can partition a partial IMU interval differently
and change the held-force approximation.

Positions are accumulated as world-axis displacements from the initial position.
Deskew uses their difference before adding rotated points, so the absolute world
translation cancels without adding a large map origin to every point. The
`state()` API adds that origin when returning an absolute pose.

| Member | Result |
| --- | --- |
| `start_ns`, `end_ns` | Read-only inclusive query coverage. Construction requires a positive interval fitting signed int64 differences. |
| `frame_id` | Read-only source IMU frame label; it does not define the caller's LiDAR frame. |
| `acceleration_scale` | Read-only scale inherited from the source series. |
| `len(motion)` | Number of retained motion knots. |
| `timestamps_ns()` | Copied int64 knot times, including clipped endpoints. |
| `state(stamp_ns)` | Copied `pose` and `velocity` arrays in the declared world frame. |

The snapshot owns the covered knots. It remains usable after releasing the
series or recording, without reopening a bag. It retains no per-point pose
matrix or covariance array.

## Coverage, validation and parallelism

Construction rejects unusable gyro/acceleration measurements, changing IMU
frames, excessive source sample gaps and nonfinite inputs/results in the
covered interval. Unknown reported covariance remains usable as in
preintegration. A missing final endpoint measurement can close the preceding
valid held sample. Intervals outside the requested range are not integrated.

Every point and the reference must lie within coverage. Deskew rejects a shape
or count mismatch, nonfinite coordinates, invalid rigid calibration or any
uncovered point. It performs no extrapolation, sorting or silent removal.
For unsorted scans, check minimum and maximum times rather than the first/last
point. An empty point array is accepted with a valid reference and calibration.

The serial path is the default. `parallel=True` uses an indexed Rayon map of
independent point calculations, preserving point order and scalar arithmetic.
Invalid inputs and nonfinite output errors report the first point index in
input order. Python snapshots input buffers before releasing the GIL and
returns a new float64 array; caller arrays remain unchanged. Core tests exercise
four worker threads and compare output bits with the serial path.

The optimal worker count depends on workload and surrounding CPU work. The
[flat-wall audit](../benchmarks/slam/flatwall-imu-deskew/README.md) compares both
paths with independent nominal dynamics on real measurements. Its declared
identity initial pose, zero velocity and gravity are numerical test inputs,
not a physical initialization or trajectory accuracy result.

When sending compensated points to `LidarOdometry`, disable its ordinary
constant-motion deskew and align the registration pose timestamp with the
chosen reference. `begin_with_prediction` changes ICP initialization only;
it does not change that ordinary deskew model. This component supplies no
LiDAR factors, coupled optimizer, clock calibration or GLIM accuracy claim.
