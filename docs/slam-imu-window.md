# Bounded inertial windows

`cloudanalyzer_core.ImuWindow` jointly optimizes pose, world velocity and IMU
bias at 2 to 32 explicitly initialized timestamps. Adjacent
[preintegrated IMU factors](slam-imu-preintegration.md), bias random walks,
full state priors and optional pose priors form a small batch problem.

This is an estimator building block. It has no old-state marginalization,
sliding-window lifecycle, automatic initialization, LiDAR frontend, clock or
extrinsic calibration. Numerical agreement with independent solvers does not
establish fused SLAM trajectory accuracy. Existing LiDAR odometry does not
automatically use this class.

## Constructing a problem

All poses are float64 IMU-to-world transforms at the **IMU origin**. Velocities
and gravity use the same world axes. Biases use IMU axes, ordered gyro (rad/s)
then accelerometer (m/s²). Supply calibrated initial states and noise parameters;
none of these are inferred from the recording.

```python
from cloudanalyzer_core import ImuWindow

# timestamps_ns: int64 (N,), strictly increasing exact IMU header times
# poses: float64 (N,4,4), velocities: float64 (N,3), biases: float64 (N,6)
window = ImuWindow(timestamps_ns, poses, velocities, biases,
                   gravity=(0.0, 0.0, -9.81))

for i in range(len(timestamps_ns) - 1):
    factor = reader.imu_factor(
        int(timestamps_ns[i]), int(timestamps_ns[i + 1]),
        gyro_noise_density=gyro_density,
        accel_noise_density=accel_density,
        bias=biases[i], topic=imu_topic,
    )
    window.add_imu_factor(
        i, factor,
        gyro_bias_random_walk=gyro_bias_walk,
        accel_bias_random_walk=accel_bias_walk,
    )

# A finite full-state prior is required; it may anchor any node.
# state_sigmas has 15 entries: rotation, world velocity, world position,
# gyro bias, accelerometer bias (three axes each).
window.add_state_prior(0, poses[0], velocities[0], biases[0], sigmas=state_sigmas)

# Optional independent pose observation, already calibrated to the IMU origin.
window.add_pose_prior(observed_index, observed_imu_pose,
                      rotation_sigma=rotation_std_rad,
                      position_sigma=position_std_m)

result = window.optimize(max_iterations=20)
print(result["initial_cost"], result["final_cost"], result["termination"])
poses, velocities, biases = result["poses"], result["velocities"], result["biases"]
```

The window copies its inputs, factors and priors. `states()` and `optimize()`
return independent copied arrays with `timestamps_ns`, `poses`, `velocities`
and `biases`. Noncontiguous NumPy inputs are supported. Solving releases the
Python GIL.

Each adjacent edge is required exactly once. Its integer `start_ns`/`end_ns`
must match the two state timestamps, including absolute epoch; equal duration
alone is insufficient. All factors must use the same IMU frame. The original
factor construction still rejects unavailable channels, excessive sampling
gaps and invalid measurements. Split the problem at missing intervals; the
window never fills a gap implicitly.

## Residuals and noise

Each node has 15 increments:

```text
[right-local rotation, world velocity, world position, gyro bias, accel bias]
R <- R Exp(dtheta), v <- v + dv, p <- p + dp, b <- b + db
```

An IMU edge uses the initial node's current bias to correct its saved
preintegration to first order, with the factor's original covariance.
Bias is constant within each integration interval. A separate six-dimensional
endpoint random walk has independent residuals:

```text
sigma_bg = gyro_bias_random_walk  * sqrt(delta_time_s)
sigma_ba = accel_bias_random_walk * sqrt(delta_time_s)
r_bias   = (bias_j - bias_i) / [sigma_bg (3 axes), sigma_ba (3 axes)]
```

The random-walk densities have units `(rad/s)/sqrt(s)` and `(m/s²)/sqrt(s)`.
They are distinct from the gyro/accelerometer measurement noise densities.
This model does not propagate within-interval bias diffusion into IMU
preintegration covariance or its cross-correlations.

For a full state prior the unwhitened residual is:

```text
[Log(R_prior^T R), v - v_prior, p - p_prior, b - b_prior]
```

Divide each of the 15 rows by its supplied standard deviation. A pose prior
uses only `Log(R_prior^T R) / rotation_sigma` and
`(p - p_prior) / position_sigma`, with isotropic rotation/position uncertainty.
These are diagonal, independent priors; correlated pose observations require a
different factor model. Rotation standard deviations are in radians. Position
differences use world axes, so this is not the translation part of an SE(3)
logarithm. Repeated priors contribute repeated independent measurements; callers
must avoid double counting correlated information.

All standard deviations and random-walk densities must be positive and finite,
with representable positive squared inverse weights. At least one complete
15D prior is required to explicitly constrain initial pose, velocity and bias.
Pose-only priors do not satisfy that requirement. Prior and IMU rotation
linearizations reject residuals within `1e-6` radians of the SO(3) branch cut.

## Solver and diagnostics

The objective is half the total squared whitened residual norm. Analytical
Jacobians form a block tridiagonal normal matrix. Per-coordinate diagonal
scaling and a Levenberg–Marquardt solve use 15×15 block Cholesky factors without
forming a dense global matrix or an explicit inverse. For fixed block size,
memory and solve work scale linearly with the number of states. This structural
property is not a measured speedup claim.

`initial_damping` defaults to `1e-3`; damping is applied to the scaled matrix,
equivalently `H + lambda * diag(H)` before scaling. A successful trial must
strictly decrease the actual cost. Damping decreases by 0.3 after acceptance
and increases by 10 after rejection; at most 12 trials are attempted per
iteration, with a maximum damping of `1e16`. `max_iterations` is 1 to 100.
The other parameters are nonnegative finite tolerances:

| Parameter | Default | Accepted stopping condition |
| --- | --- | --- |
| `gradient_tolerance` | `1e-8` | Maximum absolute unscaled gradient component is small. |
| `step_tolerance` | `1e-9` | Maximum absolute accepted increment component is small. |
| `cost_tolerance` | `1e-10` | Accepted cost decrease ≤ tolerance × (1 + previous cost). |

Increments have mixed physical units, so tolerances must suit the problem's
scale. A small rejected step caused by large damping does not imply convergence.
Results include initial/final cost, attempted iterations, accepted/rejected
trial counts, `converged`, and `termination` (`gradient`, `step`, `cost`,
`max_iterations` or `no_descent`). `cost_history` contains the initial cost and
strictly decreasing accepted costs. A nonconverged report retains accepted
progress; exceptions leave all states unchanged.

`max_abs_bias_correction` is the per-axis maximum absolute difference between
each edge's optimized initial bias and its saved integration bias. It is a
diagnostic, not a certified validity radius. The window does not reintegrate
measurements or update their covariance; large bias changes require caller-led
reintegration and reconstruction of the problem.
