# Explicit IMU acceleration units

Raw Livox acceleration in the flat-wall recording uses vendor units. The
[published sensor configuration](https://staff.aist.go.jp/k.koide/projects/glim_params/config_flatwall.tar.gz)
specifies an acceleration multiplier of 9.80665. The native integration API now
accepts an explicit positive finite `acceleration_scale`, with default 1.0.
It converts the measured acceleration before subtracting an SI-unit bias.
The caller supplies noise densities in SI units as well; the multiplier does
not rescale those densities. It does not infer units or fit calibration.

```python
series = reader.imu_series("/livox/imu", acceleration_scale=9.80665)
delta = series.preintegrate(start_ns, end_ns,
                           gyro_noise_density=gyro_density,
                           accel_noise_density=accel_density)
```

`series.acceleration_scale` is read-only. Raw measurements and reported
covariances remain unmodified. Covariance availability, missing/invalid
sentinels, strict header ordering, original-gap checks and endpoint bracketing
retain their existing meaning. Reported covariance determines availability;
the explicit SI noise densities define the integration covariance. No extra
sample array is allocated during integration.

The audit freezes separate old-native and new-native runs with the same
controller. It covers every raw field and four predeclared intervals per
recording: the complete topic and first/middle/last 100 ms windows, including
clipped integer-nanosecond endpoints.

| Recording | Raw IMU samples | Interval outcomes identical to old native |
| --- | ---: | --- |
| Flat-wall | 3,755 | 4 accepted |
| NCLT April raw window | 9,043 | 3 accepted; complete topic rejected for a gap |
| NCLT June raw window | 8,839 | 3 accepted; complete topic rejected for a gap |

All 21,637 samples retain identical raw-field digests. All ten accepted
default-scale results retain identical hashes of every output field; both
gap errors remain unchanged. Explicit scale 1.0 also matches the default.

For all four flat-wall intervals, scaled series, direct preintegration and
reusable factors have identical output-field hashes. Independent SciPy
rotations and held-measurement dynamics agree with the scaled nominal motion:

| Maximum absolute difference | Value |
| --- | ---: |
| Rotation matrix elements | 6.89e-15 |
| Delta velocity | 5.41e-13 m/s |
| Delta position | 4.10e-12 m |

The validation uses zero bias and explicit densities 0.02 rad/s/sqrt(Hz) and
0.15 m/s^2/sqrt(Hz), with a 50 ms gap limit. These are numerical validation
settings, not calibrated sensor noise. Serialization regression tests also
cover nonzero SI biases, known/unknown reported covariances, ROS 1, both CDR
byte orders and SQLite. A separate closed-form covariance check verifies that
the SI noise density is not scaled again.

Build the native Python extension and install NumPy/SciPy, then run:

```bash
python scripts/audit_flatwall_imu_scale.py \
  --bag /path/to/flatwall_01.bag \
  --out /path/to/new/baseline --baseline
# Switch to the new native extension, keeping the same controller.
python scripts/audit_flatwall_imu_scale.py \
  --bag /path/to/flatwall_01.bag \
  --out /path/to/new/scaled-audit --comparison /path/to/new/baseline
```

To include the raw NCLT controls, add `--control LABEL /path/to/window.mcap
/ms25/raw` for each window in both invocations. The controller refuses existing
output directories, freezes input/native/controller hashes, validates the
baseline protocol hash, and checks the original flat-wall recording hash.
Frozen before/after protocols and receipts are retained here without rewriting.

This is an input/preintegration audit: no reference trajectory was parsed,
no trajectory estimated, and no deskew, accuracy score or runtime benchmark
was performed. Physical point-time handling, initialization, sensor calibration
and LiDAR factors remain prerequisites for LiDAR-inertial odometry.

Dataset: Kenji Koide (AIST), [LiDAR Degeneration Test Dataset, version
2](https://zenodo.org/records/7641866), DOI
[10.5281/zenodo.7641866](https://doi.org/10.5281/zenodo.7641866),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
NCLT-derived inputs retain the University of Michigan
[source attribution and ODbL/DBCL terms](../nclt-imu-frontend/README.md).
The original recordings and reference poses are not redistributed here.
Software remains under the repository's software license.
