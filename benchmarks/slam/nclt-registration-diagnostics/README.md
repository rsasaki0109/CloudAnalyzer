# NCLT registration diagnostics

Every registration from both arms of the existing April and June frontend
experiment was replayed with final correspondence diagnostics enabled. The
saved relative predictions were reused; the inertial optimizer was not rerun.
All 948 returned pose matrices are bit-identical to the frozen originals, and
all ICP update counts agree. No reference trajectory was read or parameter
retuned. This is an audit of the previously examined cohort.

There are four startup registrations with an empty map. All other 944
registrations stopped because both update tolerances were met, with no
iteration-limit, singular-system or non-finite-system termination. This
distinguishes the earlier step counts from silent failed solves. Convergence
still does not certify a correct alignment.

| Session | Arm | Aligned scans | Minimum matched fraction | Median map-fit RMSE | 95th percentile RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 2012-04-29 | Baseline | 235 | 93.803% | 0.289494 m | 0.375162 m |
| 2012-04-29 | IMU prediction | 235 | 93.803% | 0.289175 m | 0.374669 m |
| 2012-06-15 | Baseline | 237 | 93.684% | 0.292370 m | 0.378290 m |
| 2012-06-15 | IMU prediction | 237 | 93.684% | 0.292189 m | 0.378212 m |

These are unweighted Euclidean correspondence distances at the returned pose,
after orthonormalization and **before** inserting the scan into the map. The
query uses ICP's 27-voxel stencil and distance threshold. The denominator
includes unmatched registration points. RMSE uses only accepted matches.
Quantiles describe the per-scan statistics, not pooled point residuals.
They are neither reference trajectory errors nor pose covariance estimates.
No quality threshold or scan rejection rule is selected from these results.

The CSVs contain both arms in chronological order, including every startup,
gap fallback and bootstrap case. `results.json` includes full-precision
aggregates for every frontend status, all input/output hashes, source and
binary provenance, and the original audit receipt hashes. The original
successful frontend experiment and its controllers, outputs and receipts were
preserved. The [previous packet](../nclt-imu-frontend/README.md) retains the
accuracy results, including cases where the IMU prediction worsened errors.

```bash
python benchmarks/slam/nclt-registration-diagnostics/verify.py
```

The independent verifier checks the report hashes, chronology, correspondence
ranges and every aggregate. It compares timestamps, fallback labels and ICP
counts with the previous public pose CSVs. It does not rerun ICP, reconstruct
the map, verify raw residuals, or independently establish pose-bit parity;
those checks were performed by the frozen native audit. Numerical core and
binding tests separately compare final residuals with a brute-force search.
Only LF/CRLF checkout differences are accepted when checking text hashes.

See [the API and audit instructions](../../../docs/slam-registration-diagnostics.md).
Residual measurement is opt-in and adds a final serial query pass. Runtime was
not measured here. Longer unseen sequences and calibration remain necessary
before using diagnostics to gate registrations or comparing a complete LIO
system. There is no GPU speedup claim in this audit.

Source: [University of Michigan North Campus Long-Term dataset](https://robots.engin.umich.edu/nclt/),
Carlevaris-Bianco, Ushani and Eustice, *University of Michigan North Campus
Long-Term Vision and LIDAR Dataset*, IJRR 2016. NCLT data and derived contents
retain [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) /
[DBCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) terms. Software remains
under the repository's software license.
