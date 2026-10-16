# Metric LiDAR/IMU startup alignment

`ImuSeries.initialize_lidar()` estimates gyro bias, a world gravity vector and
world-axis velocities from a batch of metric LiDAR poses and IMU samples. It
uses no reported IMU orientation and makes no stationary-start assumption.
Acceleration bias is a required, fixed calibration input. The method returns
states only when every caller-declared acceptance limit is satisfied.

```python
startup = series.initialize_lidar(
    reference_times_ns, metric_lidar_poses,
    imu_from_lidar=calibrated_imu_from_lidar,
    gyro_bias=initial_gyro_bias, accel_bias=fixed_accel_bias,
    gravity_magnitude=known_gravity_magnitude,
    max_gravity_error=gravity_consistency_limit,
    max_rotation_rms=rotation_consistency_limit,
    max_velocity_rms=velocity_equation_limit,
    max_gyro_bias_change=gyro_change_limit,
    parallel=True,
)

# The batch is available after its final reference, not at its first sample.
last_ns = int(startup["timestamps_ns"][-1])
motion = series.trajectory(
    last_ns, covered_future_end_ns,
    startup["imu_poses"][-1], startup["velocities"][-1],
    gravity=tuple(startup["gravity"]), bias=tuple(startup["bias"]),
)
```

All calibration and acceptance symbols in the example are application inputs.
Every subsequent point and reference time must be covered by the new motion
interval; overlapping earlier points require an explicit coverage policy. This
batch initializer does not resume an existing ICP map or implement later
per-scan velocity/bias feedback.

## Inputs and coordinates

| Input | Contract |
| --- | --- |
| `timestamps_ns` | `(N,)` int64, strictly increasing, with positive adjacent differences fitting int64; no float epoch conversion. |
| `lidar_poses` | `(N,4,4)` float64 LiDAR-to-world metric rigid poses at exactly those references; `3 <= N <= 4096`. |
| `imu_from_lidar` | Proper `(4,4)` transform from LiDAR into IMU coordinates, including the translational lever arm in metres. |
| `gyro_bias` | Required initial gyro xyz bias in IMU axes, rad/s. |
| `accel_bias` | Required fixed acceleration xyz bias in IMU axes, m/s² after the source acceleration scale. |
| `gravity_magnitude` | Required positive physical gravity norm in m/s². |

The supplied LiDAR poses establish the world frame. Returned `imu_poses` map IMU
coordinates into that same world; velocities and gravity use its world axes.
No absolute heading is inferred. The method removes the initial LiDAR map origin
before lever-arm and displacement arithmetic. It cannot recover precision already
lost in the caller's large absolute pose coordinates.

Python copies pose/time/calibration buffers before releasing the GIL. Strided
arrays, including ascending logical times stored with negative strides, work.
Returned timestamps, IMU poses, velocities, gravity and six-element bias are
independent writable copies. Bias order is gyro xyz followed by the fixed
acceleration xyz; `accel_bias_fixed` is always true. `frame_id` identifies the
source IMU sensor frame, not the pose world.

## Computation

First, three gyro-bias components minimize unweighted rotation residuals between
metric pose pairs and held-measurement IMU integration. Analytical SO(3) and
preintegration derivatives supply Gauss–Newton steps. Each accepted candidate
is exactly reintegrated at its new bias, rather than applying a large first-order
correction to one frozen preintegration. Backtracking enforces finite descent and
the bias-change bound. Numerical stationarity uses the full undamped step and
its predicted decrease; a small failed backtracking trial cannot establish
convergence.

Next, with acceleration bias fixed, velocities are eliminated from adjacent
displacement equations. For IMU-origin positions `p_i`, duration `dt_i`, rotation
`R_i` and preintegrated `delta_p_i`, `delta_v_i`:

```text
average_i = (p_(i+1) - p_i - R_i * delta_p_i) / dt_i
y_i = average_(i+1) - average_i - R_i * delta_v_i
alpha_i = (dt_i + dt_(i+1)) / 2

Fit y_i = alpha_i * gravity in world axes.
velocity_i = average_i - gravity * dt_i / 2
velocity_last = velocity_previous + gravity * dt_last + R_previous * delta_v_last
```

The fitted gravity is not normalized to force agreement with the requested norm.
The calculation is unweighted. Reported ROS covariance remains availability
metadata; it supplies no initialization noise calibration or uncertainty estimate.
Accelerometer bias and gravity cannot generally be separated without additional
observability or model constraints, so this API keeps the acceleration bias
explicitly fixed.

## Acceptance and diagnostics

The following limits are required and are checked before returning any state:

| Limit | Meaning |
| --- | --- |
| `max_gravity_error` | Maximum `abs(norm(fitted_gravity) - gravity_magnitude)`, m/s². |
| `max_rotation_rms` | RMS norm of per-edge rotation residuals, radians. |
| `max_velocity_rms` | RMS norm of `y_i - alpha_i * gravity`, m/s; this is an equation residual, not measured velocity accuracy. |
| `max_gyro_bias_change` | Maximum componentwise change from the supplied gyro bias, rad/s. |

The three residual/norm limits must be finite and nonnegative. The bias-change
limit must be finite and positive. Numerical settings are `max_iterations=15`
(`1..100`), `max_gap_s=0.05` (positive), and `min_singular_ratio=1e-6` (`0 < ratio <= 1`).
Gyro sensitivity is checked without damping or added jitter. Rank deficiency,
nonconvergence, bad frames/coverage/gaps, invalid rigid poses, nonfinite values
or failed physical limits raise an error. A failed call leaves the immutable
series and caller arrays unchanged and supplies no fallback initialization.

Three references provide the minimum algebraic solution, with no redundant
gravity equation. A small residual on that minimum batch cannot establish
physical reliability. Applications must choose a suitable startup span and pose
quality policy in addition to the numerical acceptance limits.

On success, `gyro_iterations`, `gyro_singular_ratio`, `gravity_error`,
`rotation_rms` and `velocity_rms` report the actual fit. Parallel execution is
off by default. With `parallel=True`, independent interval preintegrations use
indexed CPU work while ordered scalar accumulation preserves outputs and first
error ordering. Rust builds require the `parallel` feature for that mode.

## Remaining estimation work

The [fixed real-prefix audit](../benchmarks/slam/flatwall-imu-initialization/README.md)
uses LiDAR-only ICP poses for startup. Its acceptance limits are frozen before
running the experiment, including for rejection cases. No pose reference is
parsed and no failed fit is relabeled as calibration or trajectory accuracy.

LiDAR pose quality, remaining accelerometer calibration, physical initialization
observability and subsequent causal velocity/bias updates remain estimator work.
This initializer does not fit the LiDAR poses, optimize coupled LiDAR/IMU factors,
or establish complete SLAM accuracy or GPU acceleration.
