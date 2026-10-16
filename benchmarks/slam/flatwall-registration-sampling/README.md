# Fixed-map registration sampling sensitivity

`registration_voxel` independently selects the ICP source from range-filtered,
compensated points. Map insertion retains its existing half-map-voxel thinning,
capacity, nearest-search stencil and pruning. Omitted/`None` settings preserve
the exact original two-stage source path. See the
[API contract](../../../docs/slam-registration-sampling.md).

This real-prefix experiment holds map settings fixed while changing source
sampling. **Every tested startup candidate is rejected** by the unchanged IMU
consistency limits. No fitted initialization is adopted. These residuals are
equation-consistency diagnostics, not measured trajectory accuracy.

## Frozen real-prefix comparison

Original `flatwall_01.bag` is used, with SHA-256
`4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6`.
The selected source scans are indices 1 through 64, at their exact saved maximum
integer point timestamps. ICP uses raw scans with ordinary deskew disabled and
no IMU prediction. Each explicit case has its own fitted pose sequence.

| Registration voxel | Median registration points | Rotation RMS, rad | Velocity equation RMS, m/s | Failed limits |
| --- | ---: | ---: | ---: | --- |
| Omitted / `None` | 30 | 0.059340 | 1.576159 | Rotation, velocity |
| Explicit 1.5 m | 30 | 0.059340 | 1.576159 | Rotation, velocity |
| Explicit 0.75 m | 104 | 0.037170 | 1.354351 | Velocity |
| Explicit 0.375 m | 324.5 | 0.039307 | 1.273330 | Velocity |
| Explicit 0.1875 m | 1023.5 | 0.027449 | 0.687500 | Velocity |

Map voxel remains 1 m, insertion thinning 0.5 m, map capacity 20 points per
voxel and range 1.5–80 m. The same adaptive threshold algorithm, robust kernel,
prediction and solver remain in use; their fitted numerical histories can change
with the source and resulting poses. Default internal controls are initial sigma
2 m, minimum counted motion 0.1 m, 500 iterations and translation/rotation stopping
tolerances 1e-4 m / 1e-5 rad. Fitted poses also change subsequent map coordinates,
so fixed map algorithms/settings do not imply an identical complete fitted map.

The unchanged startup inputs are identity IMU-from-LiDAR transform, source
acceleration multiplier 9.80665, initial gyro bias zero and fixed acceleration
bias zero. Acceptance limits remain gravity norm error 1.0 m/s², rotation RMS
0.05 rad, velocity-equation RMS 0.2 m/s and componentwise gyro-bias change
0.2 rad/s. The remaining settings are gravity magnitude 9.80665 m/s²,
15 iterations, maximum IMU gap 0.05 s and minimum gyro singular ratio 1e-6.
No limits or tested settings were changed after observing results.

The independent calculation uses standard ROS IMU decoding, SciPy finite
differences with exact gyro-bias reintegration and NumPy gravity/velocity
equations. Native serial and parallel initialization reject every case for the
same failed limits. These declared inputs are not accepted physical calibration.
The undeskewed ICP poses and chosen pose reference need further temporal and
geometric validation before any trajectory-accuracy claim.

## Default behavior and validation

Both omitted and explicit `None` settings reproduce all previous 64 pose bits,
integer reference timestamps and per-scan reports. The independent 1.5 m path
also happens to reproduce this prefix's pose digest; the API preserves generic
bitwise legacy behavior through omitted/`None`, including floating grid boundaries.

The unchanged nominal-IMU frontend audit covers all 185 bracketed scans and
4,440,000 points. Its complete trajectory/prediction archive, per-scan reports and
50,975 map-insertion points have exactly the same file hashes as the previous
[frontend packet](../flatwall-imu-odometry/README.md). The nominal initial state
is declared numerical input; this regression does not adopt any failed startup
fit. All 4,440,000 compensated point bits also retain digest
`a31223af0ab27fe527203ffa1887310d022295e01e717eff6f4eb65e16f9b53b`.
All earlier fields over 660 scans / 7,175,872 points, raw IMU data, integer point
times and fixed preintegrations retain their previous digests.

Local release-native validation passes 481 Python cases, including 34 new
sampling, insertion, final-residual, IMU-compensation, array-stride, default-bit
and invalid-option contracts. All 14 core odometry tests pass in default and
parallel/sqlite builds, including four new sampling checks. Rust formatting,
all-target Clippy, WASM checking, stub mypy, Ruff and the production Web build
pass. A new Chromium WASM contract passes without retries: independent
source density, atomic errors/reset, pending-registration ownership and the
unchanged 42-number worker equation format.

## Evidence and reproduction

The [preimplementation protocol](slice-protocol.json) fixes the choices.
The old-native [protocol](before/protocol.json) and [receipt](before/receipt.json)
recompute and check the saved default poses. The new-native
[protocol](after/protocol.json) and [receipt](after/receipt.json) record all six
cases, including separate omitted/`None` checks. Complete per-case pose arrays,
LiDAR diagnostics, independent rejected velocities and native rejection records
remain private and are identified by their exact hashes in the receipts.

The unchanged deskew [protocol](deskew-regression/protocol.json) and
[receipt](deskew-regression/receipt.json), plus the nominal frontend
[protocol](integration/protocol.json) and [receipt](integration/receipt.json),
bind the regression outputs. Raw recordings and map coordinates are not
redistributed here.

Install NumPy, SciPy and rosbags in one environment. Preserve a fresh previous
startup baseline using the
[initialization controller](../../../scripts/audit_flatwall_imu_initialization.py)
and its [instructions](../flatwall-imu-initialization/README.md). Run the new
[sampling controller](../../../scripts/audit_flatwall_registration_sampling.py)
in fresh processes with `RAYON_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1` and old/new
snapshots selected via `PYTHONPATH`:

```sh
# Old native; --previous contains the original 64 timestamps, poses and reports.
python scripts/audit_flatwall_registration_sampling.py \
  --bag /path/to/flatwall_01.bag --previous /path/to/previous/startup/before \
  --out /path/to/fresh/sampling/before --baseline

# New native, with the same previous startup and sampling baseline.
python scripts/audit_flatwall_registration_sampling.py \
  --bag /path/to/flatwall_01.bag --previous /path/to/previous/startup/before \
  --out /path/to/fresh/sampling/after --comparison /path/to/fresh/sampling/before
```

Use the unchanged component and frontend controllers from their linked packets
for the regression. Successful protocols, controllers, native/WASM snapshots and
receipts are preserved byte for byte. This slice has no timing run, GPU
execution, reference-pose parsing or physical accuracy result. Further work must
separate point-time consistency, map geometry and geometric degeneracy rather
than choosing new defaults from this rejected prefix.

Dataset: Kenji Koide (AIST), [LiDAR Degeneration Test Dataset, version
2](https://zenodo.org/records/7641866), DOI
[10.5281/zenodo.7641866](https://doi.org/10.5281/zenodo.7641866),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
NCLT-derived controls retain the University of Michigan
[attribution and ODbL/DBCL terms](../nclt-imu-frontend/README.md).
Software remains under the repository's software license.
