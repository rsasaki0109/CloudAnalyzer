# Final-pose point-to-point linearization

`LidarOdometry(..., capture_linearization=True)` enables an owned diagnostic
snapshot for each finished scan. The default is `False`. Python exposes a
read-only `capture_linearization` property and `last_linearization()`; the latter
returns `None` before the first scan or when capture was disabled, and fresh
NumPy arrays on every call. Rust exposes `set_capture_linearization(bool)`,
`capture_linearization()` and `last_linearization()`. The choice belongs to
`begin`, so changing the flag does not change a pending registration.

```python
odometry = LidarOdometry(diagnostics=True, capture_linearization=True)
pose = odometry.register(points)
snapshot = odometry.last_linearization()
H = snapshot["normal_matrix"]
b = snapshot["right_hand_side"]
```

Capture recomputes correspondences at the returned, orthonormalized pose against
the local map **before** inserting the current scan. It uses the existing
27-voxel nearest-point search, distance threshold and robust kernel. The snapshot
is separate from `last_report()` and from the last attempted solver step. It
does not alter the solver, its parallel reduction, threshold history, map policy,
initialization gates or defaults. Capturing and copying all points has an
additional memory and computation cost; use it when inspecting registrations.
The plain Python LiDAR class and Rust core expose this API; the IMU frontend and
browser interface do not expose it yet.

The snapshot contains:

| Fields | Meaning |
| --- | --- |
| `pose`, `center_world` | Returned sensor-to-world pose and fixed prediction center used for rotation increments |
| `source_sensor`, `source_world` | All selected registration points in source order, before and after the returned pose |
| `matched_source_indices`, `target_world` | Matched source rows and their actual map targets, in source order |
| `squared_distances`, `weights` | Squared residual distances and fixed robust weights |
| `normal_matrix`, `right_hand_side` | Serial diagnostic `J.T W J` and `-J.T W residual` |
| `weight_sum`, `weighted_rms_radius_m` | Total support and weighted RMS source distance about the prediction center |
| `threshold_sq`, `kernel` | Existing squared match threshold and raw legacy kernel parameter |
| `map_voxel`, `map_points`, `max_range` | Map voxel size, per-voxel capacity and pruning range |
| `inserted_world` | Canonical map insertion points before capacity and range pruning |
| `alignment_eligible`, `finite` | Begin-time map/source eligibility and numeric finiteness, independently of solve success |

Parameters are **translation xyz, then world rotation xyz about `center_world`**.
For `q = source_world - center_world`, the per-point Jacobian is `[I, -[q]x]`.
The preserved legacy weight is `kernel**2 / (kernel + distance_squared)**2`;
`kernel` is the raw solver parameter, not a new distance scale. Weights and
matches are held fixed when forming the matrix. This is an IRLS surrogate,
not the Hessian of the full robust objective including changing matches/weights.

An empty map or insufficient source points yields an owned snapshot with no
matches, zero matrix/vector/support and `None` radius, while preserving selected
source and insertion data. A singular solve can still have matches. `finite`
flags nonfinite diagnostic values; no repair, regularization or estimator
feedback is applied. A zero radius is retained as zero rather than replaced
with an epsilon.

For a supported finite matrix, the audit uses weighted RMS radius `r` and
`D = diag(1,1,1,1/r,1/r,1/r)` to compare eigenvalues of
`sym(D H D) / weight_sum`. The corresponding parameters `[translation, r*rotation]`
have metre units. No spectrum is used without positive support and radius.
Signed eigenvalues are retained; the declared weak-direction ratio is `1e-3`.

This **conditional, uncalibrated point-to-point matrix** can be full rank even
for matched points on a plane. It is not calibrated information, covariance,
point-to-plane/GICP geometry, physical observability, a pose-quality gate or an
explanation of initialization failure. It is never inverted or fed back into
the estimator by this feature.

The [real-recording audit packet](../benchmarks/slam/flatwall-odometry-linearization/README.md)
documents four unchanged 64-scan sequences, independent map/matching/matrix
checks and their limits. The recording and prior evidence are described in
[IMU rotation compensation](slam-imu-rotation.md) and
[startup residual decomposition](slam-startup-residuals.md).

The later [full-cohort constraint audit](slam-flatwall-frontend-constraints.md)
extends the independent replay to all four 185-scan controls, preserving their
prior pose bits and reports and recording prediction-only propagation explicitly.
