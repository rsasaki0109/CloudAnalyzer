# Inspecting startup residuals before changing calibration

`scripts/audit_flatwall_startup_residuals.py` explains the existing metric
initializer's residuals on four frozen real-recording pose sequences. It also
probes fixed reference spacings and clock offsets. It changes no estimator,
acceptance limit, frontend default or calibration, and adopts no fitted state.

The [frozen packet](../benchmarks/slam/flatwall-startup-residuals/README.md)
shows why changing the clock alone is not currently a supported remedy. None of
the 52 predeclared cases passes every existing initialization limit.

## Separate the equation terms

The [initializer](slam-imu-initialization.md) eliminates velocities using
successive displacement equations. Write its gravity equation as

```text
A = diff(diff(p) / dt)
B = -diff(R * delta_p / dt) - (R * delta_v)[:-1]
alpha_i = (dt_i + dt_(i+1)) / 2
y = A + B
P = I - alpha * alpha.T / (alpha.T * alpha)

velocity_equation_residual = P * y = P * A + P * B
```

`P` removes the fitted constant world-gravity term. The audit evaluates the
original recurrence and checks the decomposition against it. It records RMS of
the projected pose and IMU terms, all residual vectors, and the signed energy
identity

```text
||residual||² = ||P*A||² + ||P*B||² + 2 * dot(P*A, P*B)
```

The cross term can be negative. These are not independent positive sensor-error
shares. A larger pose term is an algebraic observation under the declared model;
it does not prove that a particular sensor, calibration or registration caused
the physical discrepancy. The IMU term also depends on the supplied rotations
and fitted gyro bias. World-axis component values have the supplied pose gauge.

For every sequence, per-edge rotation and per-triple velocity-equation residuals
are preserved in private detailed JSON outputs identified by receipt hashes.
The public summaries include RMS, norm quantiles, largest-row indices and the
top one/five squared-energy shares. No equation or pose is removed or reweighted.

## Reference and clock probes

All inputs retain the same original 64 LiDAR reference poses, sensor settings
and first/last times. Strides 1, 3 and 7 select indices `0,stride,...,63`, giving
64, 22 and 10 references over the same span. They produce different equations:
lower coarse-batch RMS does not demonstrate better full-rate motion estimates,
validate omitted references or authorize a new startup policy. Longer rotation
intervals also have different residual semantics.

The raw-pose clock probe uses exactly 21 offsets from -50 to +50 ms in 5 ms
steps, with the explicit sign

```text
IMU integration timestamp = saved LiDAR reference timestamp + offset_ns
```

Addition and interval validation use integers. Clock offsets apply only to raw
poses with stride 1. Previously gyro-compensated poses were formed with zero
offset; they are evaluated only at zero, avoiding reuse as an inconsistent joint
calibration. A lowest sampled error at a grid boundary is not an identified
clock calibration, and the controller does not select or adopt an offset.

## Execution and reproducibility

Select the complete native Python package used by the prior sampling/rotation
audits. Use NumPy, SciPy and rosbags in the same environment. With
`RAYON_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=1` in a fresh process:

```sh
python scripts/audit_flatwall_startup_residuals.py \
  --bag /path/to/flatwall_01.bag \
  --raw /path/to/registration-sampling/after \
  --rotation /path/to/rotation-audit \
  --out /path/to/fresh/residual-audit --workers 4
```

The four processes evaluate independent cases in fixed output order. Each owns
one immutable IMU snapshot reused across its cases. Native interval integration
is serial within a worker. `--workers 1` is also supported. This is a scheduling
choice, with no runtime or universal thread-count claim in this study.

Protocols capture recording, prior poses/receipts, native/controller/helper,
ROS decoder and numerical implementation hashes before reading measurements.
All four zero-offset/full-rate fits reproduce the previous independent outputs;
all 52 accept/reject outcomes agree with the unchanged native initializer.
The public packet verifier checks published bytes, case scheduling, reported
energy identities and acceptance summaries. Recomputing the real fit requires
the recording, previous pose outputs and matching runtime described above.
