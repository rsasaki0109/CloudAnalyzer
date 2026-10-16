# Explicit inertial-window marginalization

`ImuWindow.marginalize_oldest()` removes the oldest state while preserving its
linearized information as a full 15-dimensional Gaussian prior on the next
state. `append_state()` adds one new state and its adjacent IMU factor. Callers
can maintain bounded windows without replacing the old estimate with an
arbitrary independent diagonal prior.

These are explicit operations, not a complete SLAM lifecycle. Initialization,
gap handling, observation weighting, feedback to LiDAR registration and bias
reintegration remain caller policies. The
[original NCLT pilot](slam-nclt-imu-pilot.md) retains its frozen batch protocol.

## Updating a window

Start with 2–32 states, every adjacent factor and an explicit full-state prior.
To marginalize, the window must have at least three states and either a full
state prior at index zero or a prior created by an earlier marginalization.
An initial anchor at a later index is supported for batch optimization but
rejected for marginalization. At least two states always remain.

```python
# After inspecting the preceding optimize() report:
if len(window.states()["timestamps_ns"]) == desired_window_size:
    marginalization = window.marginalize_oldest()

window.append_state(next_ns, next_imu_pose, next_world_velocity, next_bias,
                    next_factor, gyro_bias_random_walk=gyro_walk,
                    accel_bias_random_walk=accel_walk)
last_index = len(window.states()["timestamps_ns"]) - 1
window.add_pose_prior(last_index, next_observed_imu_pose,
                      rotation_sigma=rotation_sigma,
                      position_sigma=position_sigma)
result = window.optimize()
```

`next_factor` must span the exact previous last timestamp to `next_ns`, use the
same IMU frame and satisfy the existing measurement/gap validation. Appending
to an incomplete or 32-state window fails. Marginalization changes neither the
values nor timestamps of retained states; prior indices and edges shift by one.
Both operations validate before mutation. They do not themselves optimize.

## Which information is retained

Only the first IMU factor, its endpoint bias random walk, priors on the dropped
state, and any existing marginal prior participate in elimination. Pose/state
priors on retained nodes and later IMU factors remain exactly once in the
active window. Eliminating the whole window and then reusing those retained
factors would count their information twice.

For oldest increment `dx0` and next-state increment `dx1`, let the local normal
matrix blocks be `H00`, `H01`, `H10`, `H11`, with gradients `g0`, `g1`.
The undamped reduced quadratic has:

```text
Hm = H11 - H10 solve(H00, H01)
gm = g1  - H10 solve(H00, g0)
```

Diagonal scaling and Cholesky triangular solves avoid an explicit inverse.
No LM damping, jitter, eigenvalue clipping or covariance regularization enters
the prior. A nonpositive pivot is an error and leaves the window unchanged.
The resulting square-root prior preserves all rotation, velocity, position and
bias cross terms; it is not reduced to diagonal standard deviations.

`marginal_prior()` returns independent copied arrays:

```text
timestamp_ns, reference_pose, reference_velocity, reference_bias,
sqrt_information A (15,15), residual_offset r0 (15,)

delta = [Log(R_ref^T R), v-v_ref, p-p_ref, bg-bg_ref, ba-ba_ref]
residual = A delta + r0
information = A^T A
```

The reference remains frozen after subsequent optimization. The rotation
coordinate derivative uses the right SO(3) log Jacobian; linearization near its
branch cut is rejected. This prior approximates the discarded nonlinear
factors at the marginalization point. Those factors can no longer be
relinearized or reintegrated after removal. Large changes can make that
approximation inadequate.

The report includes dropped/retained timestamps, removed diagonal-prior count,
whether an existing marginal prior was replaced, and
`dropped_linearized_constant`. Add that constant to the new prior's squared
cost to recover the reduced linearized cost. Optimization omits this
state-independent constant; it can have tiny negative rounding error. Costs
from different window contents are not directly comparable.

Independent tests compare the complete reduced normal system to dense
elimination and QR projection, check all prior retraction columns, and solve
the retained nonlinear problem with SciPy. These establish numerical
correctness, not calibrated sensor uncertainty or full SLAM accuracy.
See [GTSAM fixed-lag smoothing](https://borglab.github.io/gtsam/fixedlagsmoother/)
for the broader factor-graph approach.

## Ordered NCLT replay

`scripts/replay_nclt_imu_window.py` consumes an immutable LiDAR-only baseline
from the original pilot plus original raw IMU and Euler inputs. The replay
uses at most four states (about 2.4 seconds at the bundled scan cadence).
Each update uses the current and previous LiDAR IMU-origin positions for a
backward velocity difference; stored future-dependent pilot velocities are
not loaded. Bias starts at zero after each reset; new factors are integrated
at the last estimated bias. Old factors are not automatically reintegrated.

For a raw sample gap over 50 ms, it clears the inertial window and emits the
current LiDAR pose. The next valid edge starts a new two-state window. It does
not propagate inertial information through a missing interval. Every scan
receives exactly one latest-state output, including startup and gap fallbacks;
later smoothing never overwrites earlier published outputs. This is ordered
pose/IMU replay, with no feedback into the LiDAR frontend or real-time speed
claim. Measurement and observation weights remain diagnostic and uncalibrated.

```powershell
python scripts/replay_nclt_imu_window.py `
  --pilot path/to/frozen-april-estimation --imu path/to/raw-ms25-recording `
  --euler path/to/ms25_euler.csv --out runs/new-april-replay
python scripts/evaluate_nclt_imu_replay.py `
  --replay runs/new-april-replay --ground-truth path/to/original-body-rows.csv `
  --out runs/new-april-replay-evaluation
```

The separate evaluator includes every consecutive scan edge, including
fallback transitions. It also reports valid-IMU and gap edges separately and
uses one global SE(3) position fit per session for ATE, without scale.
Ground truth does not initialize or tune the replay. New directories and
receipts are required for every experiment; historical successful cohorts
and controllers remain unchanged.

The first [frozen full-session replay](../benchmarks/slam/nclt-imu-fixed-lag/README.md)
includes all 474 poses and gap transitions, independent score verification and
the recorded marginalization events.
