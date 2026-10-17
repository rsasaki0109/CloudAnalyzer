# Flatwall full-cohort frontend comparison

The complete 185-scan comparison shows that gyro-only compensation is sensitive
to source sampling. Conditional rigid position RMSE changes from 3.206 m to
2.289 m with legacy sampling, and from 0.716 m to 1.484 m with a 0.1875 m source
voxel. All four new controls are reported alongside the frozen nominal IMU
baseline. The [evidence packet](../benchmarks/slam/flatwall-frontend-controls/README.md)
contains the five trajectories, per-scan diagnostics, error rows and figure.

These are exploratory arithmetic results. The reference was already inspected,
and its archive-specific body axes/pose direction, text layout and numerical
clock/extrinsic mapping remain assumptions. The common policy covers 183 poses
and 182 adjacent edges, with two explicit reference-tail exclusions. Even the
raw fine-sampling trajectory has a covered path of 18.252 m against a conditional
reference path of 4.081 m. A smaller global fit error is insufficient to accept
the frontend or compare it with published GLIM results.

## Estimation controls

`scripts/audit_flatwall_frontend_controls.py` separates estimation and scoring.
The estimation stage imports the declared private native package in a fresh
process, verifies its frozen bytes, uses every original scan 1–185 and writes
all four complete trajectories before any scoring begins. It supplies no
external IMU pose prediction. The existing internal ICP prediction and map
settings are identical across the four arms, with independently evolving maps.

The two pre-existing source sampling policies are retained for both raw points
and gyro-only compensation. Gyro compensation uses the immutable rotation
trajectory, declared zero bias and identity extrinsic, a fixed IMU origin within
each scan, and no rejected initializer state. Both arms disable ordinary
constant-motion point deskew. Raw clouds are assigned the common scan-end
timestamp despite their acquisition interval; this approximation is explicit.

The native library and nominal estimate are unchanged. Every new 64-scan prefix
reproduces its prior pose bits and reports; the full gyro point digest also
matches the existing component audit. Legacy raw/gyro arms retain respectively
36/37 insufficient-point reports. Both fine arms report convergence for all
184 post-anchor scans, despite their different conditional errors. No stopping
label is treated as an accuracy gate.

## Running a separately frozen study

Prepare a new protocol and freeze with checkout-relative input/source paths and
SHA-256 hashes, following the [recorded protocol](../benchmarks/slam/flatwall-frontend-controls/protocol.json)
and [controller freeze](../benchmarks/slam/flatwall-frontend-controls/controller-freeze.json).
Their private input paths require the recording and exact native package;
the public packet verifier alone does not rerun this experiment.

Set `RAYON_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=1` before starting Python.
Use new output directories outside every frozen input. Do not reuse or replace
the original protocols, estimate closure, failed attempts or receipts.

```console
python scripts/audit_flatwall_frontend_controls.py estimate --protocol path/to/protocol.json --controller-freeze path/to/controller-freeze.json --out path/to/new-estimates
python scripts/audit_flatwall_frontend_controls.py score --protocol path/to/protocol.json --controller-freeze path/to/controller-freeze.json --estimates path/to/new-estimates --out path/to/new-results
```

Estimation failures preserve partial poses and scan reports once processing
begins. Scoring requires a complete hash closure for every control, the exact
original timestamps and scan identities, and the unchanged inherited reference
policy. It rejects partial cohorts and mismatched times before reading reference
poses. It also reproduces the original nominal metrics exactly. Independent evo
arithmetic checks and bit-identical reuse of the prior pairing are recorded
separately, with their narrower scope stated in the packet.

This comparison changes no estimator API, default, initialization gate or
accepted physical state. It measures no GPU or runtime performance. The useful
next question is why local registration and prediction produce large adjacent
errors under these declared models; sampling, compensation and calibration
must remain visible in that investigation.
