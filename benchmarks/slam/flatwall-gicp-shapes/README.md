# Real-recording native GICP surface shapes

The f64 Rust/Python shape-preparation API agrees with independent exhaustive
neighbor search and NumPy PCA on **72 point sets / 47,889 points** from real
fixed-map registration inputs. Serial/parallel shape and registration outputs
match every bit. All neighbor squared distances agree exactly in this run.

| Case | Shape points | Maximum independent shape error | Maximum old-pose entry difference |
| --- | ---: | ---: | ---: |
| raw-legacy | 8,259 | 1.27e-14 | 1.67e-15 |
| raw-fine | 15,626 | 3.16e-14 | 2.65e-15 |
| rotation-legacy | 8,410 | 1.18e-14 | 2.66e-15 |
| rotation-fine | 15,594 | 1.87e-14 | 3.83e-15 |

Maximum shape disagreement is below `3.17e-14`; the native/SciPy shape difference
is also below `3.17e-14`. The run has no unsupported points, equal-distance
boundary ties or ambiguous minimum eigenspaces. Separate closed-form unit
controls cover all three cases, including explicit zero shapes for unsupported
rows and repeated minimum eigenvalues without an extra support gate.

Each point uses exact native k-nearest neighbors (default 20, minimum five),
query-relative centered population scatter and a scaled f64 eigensolve.
Normalized shape eigenvalues are `[.001,1,1]`. Support requires positive largest
spread and middle spread above `1e-12` times largest. These are surface shapes,
not calibrated measurement covariance. Native traversal determines exact
distance ties; the reference verifies ranks/selected-neighbor PCA separately
from the old SciPy-neighborhood model. For near-repeated minimum eigenvalues,
the declared independent check uses eigenpair residual and normalized spectrum.
No silent identity fallback or point removal occurs in the preparation API.

With native-produced shapes, all **36 registrations / 374 outer iterations /
343 trial costs** agree with the independent iterative oracle. Outcomes remain
27 step-converged, five iteration-limited and four empty-map cases. Maximum
native/reference pose-entry difference is below `5.63e-15`; the new/old shape
registration pose difference is below `3.84e-15`. All original caller-shape
registration histories and outputs remain bit-identical on the new binary.

Rows `[0,1,2,9,17,33,49,60,63]` are fixed before implementation for each of four
raw/gyro-only source variants. Inputs, predictions, fixed centers, 2 m strict
matching threshold, unit weights, 30 iterations and `1e-4 m` / `1e-5 rad`
step tolerances are unchanged. New local trial poses never enter later maps.
This is a numerical fixed-map validation, **not a new SLAM trajectory or
physical accuracy result**. No reference trajectory parsing, calibration,
initializer acceptance/gate change or production frontend adoption occurs.
Prepared-target caching and online integration remain subsequent work.

The unchanged full-motion component regression covers 660 scans. The saved
diagnostics, trajectory and map from the 185-scan frontend remain byte-identical
to the previous native package. Local native validation has 649 tests,
including 28 new shape tests, plus five core controls. This packet makes no
timing, whole-SLAM speed or GPU claim.

Input provenance: [fixed-map registration packet](../flatwall-gicp-registration/README.md).
Kenji Koide/AIST, [flatwall recording](https://doi.org/10.5281/zenodo.7641866),
CC BY 4.0. `flatwall_01.bag`, 184,597,802 bytes,
SHA256 `4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6`.
Nine JSON files preserve scope, all 44 original input artifacts, source/native/
controller identities, summaries and regression receipts. Full neighborhood,
shape and iteration arrays remain outside Git. LF checkout attributes preserve
the public packet and controller bytes on Windows.

```powershell
python benchmarks/slam/flatwall-gicp-shapes/verify.py
$env:RAYON_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
python scripts/audit_flatwall_gicp_shapes.py `
  --input /path/to/frozen/gicp-registration/audit --out /path/to/fresh/audit `
  --slice-protocol benchmarks/slam/flatwall-gicp-shapes/slice-protocol.json `
  --build-inputs benchmarks/slam/flatwall-gicp-shapes/native-build-inputs.json
```

The verifier checks published identities and reported summaries; it does not
rerun the real inputs or independently certify numerical/physical accuracy.
Reproduction requires the original hashed arrays and a compatible native.
See [API semantics](../../../docs/slam-gicp-shapes.md).
