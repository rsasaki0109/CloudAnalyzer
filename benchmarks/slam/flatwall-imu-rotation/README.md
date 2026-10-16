# Gyro-only flat-wall compensation

This packet checks a separate gyro-only API against independently decoded real
measurements and tests startup consistency at the same 64 explicit reference
times as the [registration sampling study](../flatwall-registration-sampling/README.md).
The [slice protocol](slice-protocol.json) was frozen before implementation/results.
The [audit protocol](rotation/protocol.json) binds controller, native extension,
recording, numerical backends and prior outputs before execution.

## Motion assumption and component checks

The new model uses held body gyro minus a declared fixed bias. Its IMU origin
is fixed within each compensated scan; IMU-origin translation remains unmodeled.
The full IMU-from-LiDAR transform includes rotation-induced lever-arm motion.
It needs no acceleration or reported attitude, and supplies no physical velocity,
gravity or accepted initialization state. See the
[API contract](../../../docs/slam-imu-rotation.md).

The original recording contains 186 scans. The first has uncovered points and
is rejected whole. For each of the other **185 scans / 4,440,000 points**, the
audit checks independently decoded coordinates and integer point times, computes
SciPy held-gyro rotations and compares every output coordinate. It also runs a
numerical nonidentity sensor transform with a nonzero lever arm; that control
does not assert a different physical calibration for the real sensor.

| Component | Maximum absolute difference |
|---|---:|
| IMU rotation at all 3,755 sample times | 8.89e-15 |
| Identity real extrinsic, point coordinates | 7.47e-13 |
| Numerical full-extrinsic control, point coordinates | 8.39e-13 |

Every serial/parallel output bit matches with four requested Rayon workers.
Reference-time point bits and caller buffers are unchanged. All 11 decreasing
point-time steps are retained in source order. Exact errors, coordinate digests,
inputs and checks are in the [receipt](rotation/receipt.json).

## Same-prefix startup result

The point reference remains the maximum exact point timestamp in each of source
scans 1 through 64. Raw-point controls reproduce every prior ICP pose bit and
diagnostic on the new native. Compensated points enter `LidarOdometry(deskew=False)`
before filtering, source sampling and insertion, without an IMU pose prediction.
Map voxel remains 1 m, capacity 20 and range 1.5–80 m; the independent source
option changes only between the two previously declared cases.

| Source sampling | Rotation RMS, raw → compensated (rad) | Velocity-equation RMS, raw → compensated (m/s) | Frozen gate |
|---|---:|---:|---|
| Legacy, median 30 points | 0.059340 → 0.051605 | 1.576159 → 1.545273 | Reject rotation + velocity |
| Explicit 0.1875 m, median 1026.5 compensated points | 0.027449 → 0.031831 | 0.687500 → 0.784635 | Reject velocity |

The limits remain 0.05 rad rotation RMS and 0.2 m/s velocity-equation RMS, with
all other initialization settings unchanged. This velocity residual is a
displacement-triple equation consistency measure, not reference trajectory error.
Independent SciPy/NumPy and native serial/parallel runs reject the same limits.
All 63 noninitial ICP registrations converge in both cases; that does not certify
their physical poses. No rejected estimate is adopted and no default is enabled.
These results do not show that rotation-only compensation resolves startup.
Declared zero gyro bias, omitted translation, sensor/clock calibration and noise
remain limitations. No ground-truth poses are parsed, and this slice contains
no timing, GPU or trajectory accuracy result.

## Regression and reproduction

The unchanged full-motion [component regression](deskew-regression/receipt.json)
checks 660 scans / 7,175,872 points across the three prior recording controls.
Its full-motion compensation digest remains unchanged. The existing frontend
[integration receipt](integration/receipt.json) covers 185 scans; its complete
pose NPZ, per-scan JSON and map NPY files match the prior frontend byte for byte.
Local release-native tests pass 528 cases, including 47 new gyro-only cases.
Core gyro tests pass 6 cases by default and 7 with parallel/sqlite; formatting,
all-target Clippy, WASM compilation, stub mypy and Ruff pass.

Build separate native packages and keep NumPy, SciPy, rosbags and previous
controllers fixed. Reproduce the previous registration sampling audit on that
host, then select this revision's complete Python package via `PYTHONPATH`.
In a fresh process with `RAYON_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=1`:

```sh
python scripts/audit_flatwall_imu_rotation.py \
  --bag /path/to/flatwall_01.bag \
  --previous /path/to/registration-sampling/after \
  --out /path/to/fresh/rotation-audit
```

Outputs must be fresh. Protocols, controllers, native/source snapshots and receipts
are preserved verbatim; public JSON files are exact byte copies. Source recordings
and generated maps/pose arrays are not redistributed here.

Dataset: Kenji Koide (AIST), [LiDAR Degeneration Test Dataset, version
2](https://zenodo.org/records/7641866), DOI
[10.5281/zenodo.7641866](https://doi.org/10.5281/zenodo.7641866),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
NCLT-derived controls retain the University of Michigan
[attribution and ODbL/DBCL terms](../nclt-imu-frontend/README.md).
Software remains under the repository's software license.
