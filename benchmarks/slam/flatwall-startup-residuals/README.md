# Fixed-prefix startup residual inspection

**All 52 predeclared cases remain rejected by the existing initializer.** Within
the tested ±50 ms clock range, velocity-equation RMS changes by less than 0.1%
relative to the zero-offset baseline at its lowest sampled value. Coarser
references reduce that RMS but can increase rotation inconsistency. This study
does not select a new clock, reference policy, calibration or accepted state.

The [slice protocol](slice-protocol.json) predates implementation and results.
The [execution protocol](audit/protocol.json) binds the original recording,
four previously saved 64-pose sequences, native extension, numerical libraries
and controllers before reading measurements. The [receipt](audit/receipt.json)
identifies every detailed output by hash. No estimator is rerun to change those
input poses, and no equation, point or pose is deleted, smoothed or reweighted.

## What contributes algebraically to the residual

The existing gravity/velocity initializer differentiates pose displacement
rates. Projecting out fitted gravity separates its velocity-equation residual
into a pose term and an integrated-IMU term. The
[method contract](../../../docs/slam-startup-residuals.md) gives the exact formula.
The new controller checks the sum identity and keeps the signed cross energy.

| Frozen pose input, full rate | Projected pose term RMS (m/s) | Projected IMU term RMS (m/s) | Combined RMS (m/s) |
|---|---:|---:|---:|
| Raw, legacy sampling | 1.572615 | 0.065472 | 1.576159 |
| Raw, fine sampling | 0.686615 | 0.043115 | 0.687500 |
| Gyro compensated, legacy | 1.544881 | 0.073525 | 1.545273 |
| Gyro compensated, fine | 0.784215 | 0.047026 | 0.784635 |

The larger projected pose term is an algebraic observation under the declared
model, **not a causal sensor diagnosis or positive error-share allocation**.
The two terms are correlated; the IMU term also uses pose rotations and a fitted
gyro bias. No state from any rejected candidate is fed back into compensation.

The top five of 62 full-rate velocity equations contain 32.8% of the raw-fine
residual energy and 49.8% of the gyro-compensated fine energy. These reports
describe concentration without dropping equations. Exact quantiles, row indices,
all four baseline diagnostics and signed energies are in the
[baseline summary](audit/baseline-summary.json).

## Same-span reference spacing

Strides 1, 3 and 7 retain indices `0,stride,...,63`, giving 64, 22 and 10 poses
over the same approximately 6.3-second span, including both endpoints. The fine
raw pose sequence gives:

| Stride | References | Rotation RMS (rad) | Velocity-equation RMS (m/s) | Existing failed limits |
|---|---:|---:|---:|---|
| 1 | 64 | 0.027449 | 0.687500 | Velocity |
| 3 | 22 | 0.039593 | 0.254086 | Velocity |
| 7 | 10 | 0.069850 | 0.249763 | Rotation and velocity |

The thresholds remain 0.05 rad and 0.2 m/s, with all other declared limits
unchanged. Coarsening changes the equations and rotational spans; its lower
velocity RMS is not a full-rate accuracy improvement and does not validate the
omitted references. All four pose inputs fail at every tested stride.

## Fixed clock grid

Only the two raw pose sequences are probed at nonzero offsets, to avoid reusing
zero-offset compensated points as a joint clock/compensation calibration. Each
offset shifts exact LiDAR references into the IMU clock by checked integer
addition: `IMU reference = saved LiDAR reference + offset_ns`.

| Raw pose input | Lowest sampled rotation RMS (rad), offset | Lowest sampled velocity RMS (m/s), offset |
|---|---|---|
| Legacy | 0.059096 at -50 ms | 1.574988 at +50 ms |
| Fine | 0.026753 at -50 ms | 0.686854 at -50 ms |

Both lowest sampled rotation errors lie at the grid boundary. This is not an
identified clock optimum and cannot establish synchronization outside the tested
range. No grid case passes initialization. The velocity reduction at its lowest
sampled value is approximately 0.074% for legacy and 0.094% for fine sampling.
All 52 rows, including the twelve same-span stride cases, are in the
[sensitivity summary](audit/sensitivity.json).

These observations prioritize inspecting the supplied LiDAR pose sequence and
registration constraints before treating a small clock adjustment as a remedy.
That is a direction for the next estimator work, not proof of the physical cause.
Translation within scans, sensor calibration, noise, pose observability and
longer-span rotational inconsistency remain unresolved.

## Checks and reproduction

The four full-rate/zero-offset independent fits reproduce their prior saved
diagnostics, gravity, bias and velocity values within 1e-10. Every case's
accept/reject outcome matches the unchanged native initializer. Five closed-form
tests check unequal intervals, signed cancellation, exact clock arithmetic,
reference endpoints, distribution reporting and invalid inputs. Two packet
tests verify published summaries and reject changed evidence bytes.

Four CPU processes evaluate independent cases in fixed output order, each
reusing one immutable native IMU topic snapshot. This study contains no runtime,
GPU, physical accuracy, uncertainty or ground-truth trajectory result. The
[sensor summary](audit/sensor.json) describes the measured expanded prefix;
its acceleration norms and gyro variation are not stationary/noise calibration.

Reproduce the [prior source sampling](../flatwall-registration-sampling/README.md)
and [rotation](../flatwall-imu-rotation/README.md) audits first, keeping their
successful files and matching native/Python runtime fixed. Then follow the
[command and environment contract](../../../docs/slam-startup-residuals.md).
Outputs must be fresh. Public JSON files are exact byte copies; detailed fit and
per-interval JSON outputs remain private and are identified by receipt hashes.
Source recordings and pose arrays are not redistributed here.

For the public checks, without the recording or native extension:

```sh
python benchmarks/slam/flatwall-startup-residuals/verify.py
```

This verifies public bytes, controller, case scheduling, reported energy
identities and acceptance summaries. It does not independently recompute the
real IMU fits or validate unbundled detailed outputs.

Dataset: Kenji Koide (AIST), [LiDAR Degeneration Test Dataset, version
2](https://zenodo.org/records/7641866), DOI
[10.5281/zenodo.7641866](https://doi.org/10.5281/zenodo.7641866),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Software remains under the repository's software license.
