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

The result contains exact integer `start_ns` and `end_ns`, a 3x3 rotation,
3-vectors for velocity and position, elapsed seconds, integration step count,
frame ID, a 9x9 covariance, and a 9x6 bias
Jacobian, and the saved `linearization_bias` (gyro then accelerometer).
Covariance/Jacobian rows are **right-local rotation, additive velocity,
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
not a complete inertial odometry system. Preintegration excludes gravity and
initial velocity; the inertial factor below adds them explicitly. A calibrated
LiDAR-inertial estimator also needs extrinsics, clock calibration, initialization,
bias dynamics and a state optimizer.
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
The [bounded inertial window](slam-imu-window.md) supplies a state optimizer.
Calibration, initialization, LiDAR integration and trajectory accuracy evaluation
with raw LiDAR+IMU data are still required.

## Reusable inertial factors

`BagReader.imu_factor()` uses the same interval, noise and bias arguments as
`preintegrate_imu()`. It owns the measurement and factors its covariance once.
Repeated prediction, evaluation and linearization use that saved measurement.

```python
factor = reader.imu_factor(
    start_ns, end_ns,
    gyro_noise_density=gyro_density,
    accel_noise_density=accel_density,
    bias=(bg_x, bg_y, bg_z, ba_x, ba_y, ba_z),
    topic="/imu/data",
)
prediction = factor.predict(pose_i, velocity_i, gravity=(0.0, 0.0, -9.81))
evaluation = factor.evaluate(
    pose_i, velocity_i, pose_j, velocity_j, gravity=(0.0, 0.0, -9.81),
)
linear = factor.linearize(
    pose_i, velocity_i, pose_j, velocity_j, gravity=(0.0, 0.0, -9.81),
)
```

Each pose is a float64 `(4,4)` IMU-to-world transform at the **IMU origin**.
Velocity and gravity are world-frame vectors. The example assumes world z is
up; supply gravity for the chosen world frame. LiDAR poses require calibrated
conversion to the IMU frame, including the lever arm. Rotation matrices must be
proper orthonormal matrices, and homogeneous last rows must be `[0,0,0,1]`.
Inputs and results must be finite.

The optional `bias=` on these three methods is the current gyro/accelerometer
bias, in the same IMU axes as the measurement. Its default is the saved
linearization bias. A changed bias uses the first-order correction:

```text
db = bias - linearization_bias
correction = bias_jacobian * db
dR = delta_rotation * Exp(correction.rotation)
dv = delta_velocity + correction.velocity
dp = delta_position + correction.position

Rj = Ri * dR
vj = vi + gravity * dt + Ri * dv
pj = pi + vi * dt + 0.5 * gravity * dt^2 + Ri * dp

residual.rotation = Log(dR^T * Ri^T * Rj)
residual.velocity = Ri^T * (vj - vi - gravity * dt) - dv
residual.position = Ri^T * (pj - pi - vi * dt - 0.5 * gravity * dt^2) - dp
```

Bias correction and covariance use the saved linearization point. Reintegrate
when bias changes are too large for a first-order approximation; the factor does
not estimate a validity radius or update its covariance for the new bias. This
is the discrete manifold correction described in
[Forster et al.](https://www.roboticsproceedings.org/rss11/p06.pdf).

`evaluate()` returns the raw residual, `whitened_residual`, and
`cost = 0.5 * whitened_residual.dot(whitened_residual)`. Whitening solves
`L * whitened_residual = residual` for `covariance = L * L^T`.
The covariance must be numerically positive definite: correlation-matrix
Cholesky pivots must exceed `1e-12`. No diagonal noise or jitter is added.
Zero-noise and single held-measurement intervals can still be preintegrated,
but their singular covariance cannot construct a weighted factor.

`linearize()` additionally returns `jacobian_i`, `jacobian_j` (9x9),
`jacobian_bias` (9x6), `jacobian_gravity` (9x3), and their `whitened_` versions.
State columns use `[right-local rotation, world velocity, world position]`:
`R <- R * Exp(dtheta)`, `v <- v + dv_world`, `p <- p + dp_world`.
This differs from an SE(3) increment with body-local translation. Residual rows
retain `[rotation, velocity, position]`, and bias columns retain `[gyro, accel]`.
At a rotation residual within `1e-6` radians of pi, `linearize()` rejects the
SO(3) logarithm branch cut; the finite residual can still be evaluated.

The factor supplies the residual and derivative blocks for a navigation-state
optimizer. It does not include a prior, bias random-walk factor, marginalization
or a LiDAR-inertial initialization policy. Numerical optimizer checks establish
the interface and derivatives, without establishing fused SLAM accuracy.

`ImuWindow` combines these factors, bias random walks and explicit state/pose
priors in a bounded batch problem. See [inertial windows](slam-imu-window.md)
for its noise model, frame conventions and limitations.
