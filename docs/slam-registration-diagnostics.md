# LiDAR registration diagnostics

`LidarOdometry` records why the last registration stopped. Optional diagnostics
also measure correspondence support and residuals at the returned pose, against
the local map **before** the scan is inserted. These observations prepare future
registration changes; they do not reject scans or change the map acceptance
policy.

```python
import cloudanalyzer_core as core

odometry = core.LidarOdometry(diagnostics=True)
pose = odometry.register(points)  # float64 (N, 3)
report = odometry.last_report()
print(report["termination"], report["iterations"])
if report["correspondences"] is not None:
    print(report["correspondences"]["matched_fraction"])
    print(report["correspondences"]["rmse_m"])
```

`last_report()` is `None` before the first scan and returns an independent copy
afterward. A rejected relative prediction leaves the last report unchanged,
along with the poses and map. Without `diagnostics=True`, termination, update
count, input/registration point counts and correspondence threshold are still
recorded, but `correspondences` is `None`. Enabling residuals retains the sensor
points thinned for registration and adds one serial map-query pass per aligned
scan. It does not add queries to every ICP iteration.

## Termination

| Value | Meaning |
| --- | --- |
| `empty_map` | No local map points; uses the initial pose without alignment. |
| `insufficient_points` | Map exists, but fewer than ten registration points remain. |
| `converged` | Applied update is below both configured translation and rotation tolerances. |
| `iteration_limit` | Applied updates reached the configured maximum without meeting both tolerances. |
| `singular_system` | Partial-pivot solve encountered an absolute pivot below `1e-12`. Includes zero correspondences and some degenerate configurations. |
| `non_finite_system` | The supplied normal equations or their solved update contain non-finite values, including detected elimination overflow. |
| `non_finite_update` | The proposed rigid-transform composition contains non-finite values. |
| `unfinished` | Caller used the incremental API and finished before a step reported completion. |

`iterations` counts successfully applied Gauss–Newton updates. Failed solves or
non-finite proposed transforms do not increment it. `last_iterations()` exposes
the same count as before. A step after termination returns completion without
applying another update. Invalid systems retain the last finite correction.

`converged` describes a small update, not a correct registration. The fixed
absolute pivot test is not a calibrated degeneracy or condition-number test.
For example, collinear matches can have zero residual yet an unsolvable system.
The existing policy still inserts scans whose registration did not converge;
introducing acceptance thresholds requires separate calibration and evaluation.

## Correspondences and units

`input_points` is the original scan count. `registration_points` counts points
after range filtering, optional deskew and registration thinning, even when the
empty map or fewer than ten points prevents alignment. `threshold_m` is the
distance threshold used for this registration, before the adaptive motion-error
history is updated.

The final query uses the exact returned pose after rotation orthonormalization.
It uses the same nearest-point search in the containing voxel and its 26
neighbors as ICP, followed by the same distance cutoff. It does not search the
entire map. Points that have no local neighbor or exceed the cutoff remain in
the denominator of `matched_fraction`.

| Field | Meaning |
| --- | --- |
| `matched_points` | Number of registration points with an accepted correspondence. |
| `matched_fraction` | `matched_points / registration_points`. |
| `rmse_m` | Square root of mean squared Euclidean distance over accepted pairs, in metres. |
| `max_residual_m` | Maximum Euclidean distance over accepted pairs, in metres. |

Distances are unweighted; the robust kernel used to solve ICP does not weight
these statistics. With zero accepted pairs, the fraction is zero and both
distances are `None`. With no alignment attempt, the whole correspondence
object is `None`, including when diagnostics are enabled. Residuals are map
fit measurements, not ground-truth pose errors or covariance estimates.

## Rust and WebAssembly

Rust callers enable `OdometryParams { diagnostics: true, ..Default::default() }`
or call `set_diagnostics(true)` before `begin`. The flag is captured at `begin`;
changing it later applies to subsequent registrations. `last_report()` returns
an optional `RegistrationReport`. `Registration::termination()` is `None` while
steps are still required. `finish` also records reports when equations came
from external workers.

```javascript
odometry.setDiagnostics(true);
const pose = odometry.register(cloud);
const report = JSON.parse(odometry.lastReport());
```

WASM `lastReport()` returns a JSON string or `undefined` before the first scan.
Incremental `begin/equations/step/finish` uses the same reports. Normal equations
remain 42 numbers: the row-major 6×6 matrix followed by six right-hand sides.
Worker messages and reduction formats are unchanged. Diagnostics run on the
owner's map at finish; they do not extend the worker equation payload.

## Frozen real-data audit

[The NCLT report packet](../benchmarks/slam/nclt-registration-diagnostics/README.md)
contains every scan from both existing frontend arms, with all startup, gap and
bootstrap cases. The controller replays saved relative predictions from the
[previous frontend experiment](slam-imu-frontend.md). It requires each returned
pose bit and ICP update count to equal the frozen result. It does not rerun the
inertial optimizer, read reference poses, measure runtime, or retune thresholds.
The inputs were already examined during development.

```powershell
$env:RAYON_NUM_THREADS = '1'
python scripts/audit_nclt_registration.py `
  --lidar web/public/samples/nclt-2012-04-29.mcap `
  --frontend /path/to/frozen/frontend/estimation `
  --out /path/to/new/audit
```

The frontend directory must contain its original receipt and every hashed
output, including `trajectory.npz` with saved predictions. The audit refuses an
existing output directory, verifies inputs before and after registration, and
writes a new protocol, all per-scan reports, audited poses and a receipt. Use a
separate output directory for each session or new experiment.
