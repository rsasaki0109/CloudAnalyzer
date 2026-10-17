# Gyro-only scan rotation compensation

`ImuSeries.rotation_trajectory()` creates an immutable gyro-only snapshot for
point compensation before an initial velocity or gravity estimate is available.
It integrates usable body gyro minus an explicitly declared fixed bias. It
ignores acceleration, its scale and reported orientation, including missing or
invalid values in those channels.

This is a **fixed IMU-origin model within each scan**. Translation of that origin
remains unmodeled. It is not a full inertial trajectory, stationary detector,
calibration, gravity estimate or fallback accepted initialization state. Use
[`ImuTrajectory`](slam-imu-trajectory.md) when a complete initial state is declared.

```python
import numpy as np
import cloudanalyzer_core as core

reader = core.BagReader("recording.bag")
series = reader.imu_series("/imu")
times = series.timestamps_ns()
rotation = series.rotation_trajectory(
    int(times[0]), int(times[-1]),
    gyro_bias=(0.0, 0.0, 0.0),  # Declared assumption, not fitted calibration.
    max_gap_s=0.05,
)
odometry = core.LidarOdometry(deskew=False)
imu_from_lidar = np.eye(4)  # Replace with the measured full sensor transform.
for scan in reader.scans("/points", raw_time_field="t"):
    # Use these units/origin only when they match the recording's contract.
    point_ns = scan.point_times_ns(unit="ns", origin="header")
    if len(point_ns) == 0:
        continue
    if point_ns.min() < rotation.start_ns or point_ns.max() > rotation.end_ns:
        continue  # Explicit whole-scan coverage policy; no extrapolation.
    reference_ns = int(point_ns.max())
    points = rotation.deskew(
        scan.positions(), point_ns, reference_ns,
        imu_from_lidar=imu_from_lidar, parallel=True,
    )
    pose_at_reference = odometry.register(points)
```

Point times and the reference are signed integer nanoseconds on the IMU clock.
Units, origin, clock offset, sensor transform and gyro bias must come from the
recording contract. No synchronization or units are guessed. Disable ordinary
constant-motion deskew when passing already compensated points to odometry.
Range filtering, registration sampling and insertion follow compensation. The
ICP pose belongs to the explicit reference time; no gyro prediction is supplied.

## Integration and coordinates

The start rotation is the identity gauge, not a measured absolute attitude.
`rotation(stamp_ns)` returns the IMU-to-start-gauge 3 by 3 rotation. Each left
gyro sample is held on `[t_k, t_(k+1))`. Clipped steps and queries subtract
integer timestamps before converting durations to seconds. Every point searches
fixed knots independently, so decreasing/repeated times do not alter integration.

For the full IMU-from-LiDAR transform `(R_IL, t_IL)` and gyro rotation `R(t)`:

```text
p_ref = R_IL.T * (R(ref).T * R(t) * (R_IL * p + t_IL) - t_IL)
```

The lever arm moves the LiDAR origin under IMU rotation and must be included.
IMU-origin translation remains omitted. Points exactly at the reference are
copied bit for bit after validation. Output preserves input order and count.

## Ownership and failures

The snapshot owns knots, frame ID and copied gyro bias. Python exposes
`start_ns`, `end_ns`, `frame_id`, `gyro_bias`, `len()`, copied `timestamps_ns()`
and copied `rotation()`. It exposes no velocity, position or physical state.
Neither the series nor recording is accessed after construction. Python copies
strided point/time buffers before releasing the GIL; output is a new array.
`parallel=True` uses an indexed Rayon point map with identical scalar arithmetic;
the default is serial.

Construction rejects nonfinite bias, nonpositive/nonfinite gap limits, incomplete
bracketing, nonpositive/overflowing intervals, frame changes, unusable integrated
gyro and invalid rotation results. Gaps check complete source intervals even
when the requested interval is short. The final sample needs a matching frame;
its gyro is not required when it only closes a preceding held sample.

Deskew rejects mismatched counts, nonfinite points, uncovered point/reference
times, invalid full rigid extrinsics and nonfinite outputs. Python also rejects
incorrect shapes, timestamp dtype and nonhomogeneous transform last rows.
Serial and parallel modes report the same first input error index.

## Real-recording result

The [frozen flat-wall packet](../benchmarks/slam/flatwall-imu-rotation/README.md)
checks 185 covered scans and 4,440,000 points against independent ROS decoding
and SciPy rotations, including a numerical nonzero lever-arm control. Serial and
parallel bits match. The same 64 startup references are compared with the prior
raw-point pipeline under unchanged initialization limits.

Rotation compensation does **not** pass physical initialization on this prefix.
Standard sampling slightly reduces residuals; the fine-source case worsens.
The API is opt-in, defaults and acceptance limits are unchanged, and no rejected
state is adopted. This slice makes no timing, GPU or trajectory accuracy claim.
Translation, physical sensor/clock calibration and noise remain unresolved.

The later [full-cohort comparison](slam-flatwall-frontend-controls.md) retains
all 185 poses with raw and gyro-only controls at both source sampling settings.
Conditional trajectory errors improve with legacy sampling and worsen with
fine sampling. This interaction and the unverified reference body/clock mapping
preclude adopting gyro compensation or a different initialization gate here.
