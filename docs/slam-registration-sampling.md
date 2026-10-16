# Independent registration sampling

Python `LidarOdometry` and `ImuLidarOdometry` accept the keyword-only option
`registration_voxel`, in metres. A finite positive value samples the registration
source independently from the points retained for map insertion:

```python
odometry = cloudanalyzer_core.LidarOdometry(
    map_voxel=1.0, map_points=20,
    registration_voxel=0.375, diagnostics=True,
)
```

The same option is available on `ImuLidarOdometry(...)`. Its IMU compensation
still runs once before sampling, in the explicit pose reference frame. Python
exposes the selected option as a read-only `registration_voxel` property.

## Sampling contract

Range filtering and any configured deskew happen first. An explicit source voxel
then keeps the first encountered point in each cell, in original point order.
It samples this filtered/compensated point set directly, including points that
the separate insertion thinning would discard. Negative coordinates use floor
division, with the same origin and cell-key arithmetic as the existing sampler.

Omitting the option, or passing `None`, preserves the original two-stage path
exactly: first thin at `0.5 * map_voxel`, then thin that result at
`1.5 * map_voxel` for registration. An explicit value uses the independent-source
path; even `1.5 * map_voxel` is not a promise of bitwise legacy equivalence at
floating-point cell boundaries. Use `None` for the legacy contract.

Map insertion still uses `0.5 * map_voxel`, with the same map voxel size, point
capacity, range pruning and 27-cell nearest-neighbor stencil. Registration still
uses the existing robust kernel, adaptive distance threshold, prediction and
solver. A different source can produce a different fitted pose, which also
changes future world map coordinates. Holding map settings fixed does not hold
the complete fitted map fixed.

`last_report()["registration_points"]` counts the points selected for the source,
including on an initial empty map. This count may exceed the number of insertion
points. Correspondence support and residuals refer to that selected source and
the map before insertion. Neither convergence nor a small correspondence residual
certifies physical pose accuracy or geometric observability.

## Rust and WASM

The core keeps existing `OdometryParams` constructors compatible. Configure an
odometry with `set_registration_voxel(Some(voxel))`; `None` restores the legacy
path. The accessor `registration_voxel()` returns the explicit option. Invalid
zero, negative or nonfinite values return an error without changing state.

WASM exposes `setRegistrationVoxel(value)` and the `registrationVoxel` getter.
Pass `undefined` to restore legacy sampling. A rejected value throws an error
and retains the previous setting. Settings affect registrations begun afterward;
an existing `Registration` owns its original source/insertion points. Worker map
copies and the 42-number normal-equation wire format continue to work with either
source path.

## Real startup experiment

The [fixed-map real-prefix audit](../benchmarks/slam/flatwall-registration-sampling/README.md)
keeps the original recording, 64 source scans, integer reference times, map
settings and IMU initialization limits fixed while changing source sampling.
The omitted and `None` paths must reproduce the previous pose bits and reports.
Each explicit case has a separate immutable pose sequence and independent
startup consistency calculation. No failed fit is adopted into the frontend.

These comparisons diagnose settings sensitivity. They do not establish a
physical calibration, trajectory accuracy, speedup or GPU result. Per-point
motion/reference-time consistency and geometric degeneracy remain estimator work.
