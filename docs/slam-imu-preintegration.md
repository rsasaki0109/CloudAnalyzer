# Raw IMU data and preintegration

`BagReader.imu_raw()` preserves the complete `sensor_msgs/Imu` fields from ROS 1
bags, ROS 2 MCAP and rosbag2 SQLite recordings. Header timestamps are returned as
integer nanoseconds, separately from floating-point seconds and recording times.
Quaternions and covariances are returned unmodified. This API complements the
existing `imu()` gravity-direction API.

```python
from cloudanalyzer_core import BagReader

reader = BagReader("recording.mcap")
raw = reader.imu_raw(topic="/imu/data")
print(raw["angular_velocity_status"][:5])
```

The three status lists distinguish:

| Status | Meaning |
| --- | --- |
| `missing` | Covariance entry 0 is -1: ROS declares this measurement unavailable. |
| `unknown` | Measurement is finite and all nine covariance entries are zero. |
| `known` | Measurement is finite; covariance is finite, symmetric and positive semidefinite within numerical tolerance. |
| `invalid` | Nonfinite measurement, invalid covariance or invalid orientation quaternion. |

Orientation validity also requires squared quaternion norm within 0.001 of one.
The missing sentinel takes precedence over the other checks. An unknown
covariance does not imply perfect certainty. These conventions follow the
[ROS Imu definition](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/Imu.msg).

## Integrating an interval

```python
# Use calibrated sensor noise densities; these names intentionally have no defaults.
delta = reader.preintegrate_imu(
    int(raw["timestamps_ns"][0]),
    int(raw["timestamps_ns"][10]),
    gyro_noise_density=gyro_density,        # rad/s/sqrt(Hz)
    accel_noise_density=accel_density,      # m/s^2/sqrt(Hz)
    bias=(bg_x, bg_y, bg_z, ba_x, ba_y, ba_z),
    max_gap_s=0.05,
    topic="/imu/data",
)
```

Measurements are held constant on `[t_k, t_{k+1})`. The requested interval must
be bracketed by samples; its endpoints clip the first and last intervals.
Integer timestamp differences are computed before converting to seconds.
Times must be strictly increasing. Missing or invalid angular velocity or
acceleration, changes in `frame_id`, and sample gaps above `max_gap_s` raise
`ValueError`. The gap limit applies to the original sampling interval, even if
only a small part overlaps the requested interval. The final endpoint supplies
time/frame metadata and need not contain an available measurement. Orientation
estimates are not required. Multiple IMU topics require explicit selection.

The mean follows the discrete manifold preintegration equations:

```text
a = measured_acceleration - accelerometer_bias
w = measured_angular_velocity - gyro_bias
delta_position += delta_velocity * dt + 0.5 * delta_rotation * a * dt^2
delta_velocity += delta_rotation * a * dt
delta_rotation  = delta_rotation * Exp(w * dt)
```

The result contains a 3x3 rotation, 3-vectors for velocity and position, elapsed
seconds, integration step count, frame ID, a 9x9 covariance, and a 9x6 bias
Jacobian. Covariance/Jacobian rows are **right-local rotation, additive velocity,
additive position**; bias columns are **gyro, accelerometer**. Velocity and
position use the initial IMU frame. These array orders must be converted when
interfacing with libraries that use different conventions.

Noise propagation uses independent, isotropic gyro/accelerometer densities:
per-interval measurement variance is `density^2 / dt`. The nominal dynamics and
their analytical Jacobians use the same held-measurement discretization.
Reported ROS covariance values determine availability; they do not replace the
explicit density model. Biases remain fixed at the supplied linearization point;
bias random walk and calibration uncertainty are not modeled.

This is an estimator building block based on
[manifold IMU preintegration](https://www.roboticsproceedings.org/rss11/p06.pdf),
not a complete inertial odometry system. It excludes gravity, initial velocity,
LiDAR extrinsics, clock calibration, initialization and optimization factors.
For a stationary, upright sensor reporting positive 9.81 m/s^2 on z,
`delta_velocity.z` is positive 9.81 times elapsed seconds: gravity must be added
by the future state predictor. A changing or empty frame name does not provide
an extrinsic calibration.

The existing LiDAR odometry and WASM gravity-direction buffer are unchanged.
No real-world accuracy or speed improvement is established by this addition.
The bundled NCLT demo bags declare their gyro and accelerometer measurements
unavailable and must not be treated as raw inertial recordings for this API.
Validation covers ROS serialization, exact timestamp handling, numerical bias
derivatives, and covariance checked against independent measurement derivatives.
The next estimator step needs raw LiDAR+IMU data, calibration, initialization,
an inertial factor and a trajectory accuracy evaluation.
