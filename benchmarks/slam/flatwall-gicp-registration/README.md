# Real-recording iterative GICP on fixed maps

The standalone Rust/Python registration API agrees with an independent
NumPy/SciPy iterative solver on **36 fixed-map registrations, 374 outer
iterations and 343 frozen-metric trial costs**. All serial/parallel results
match every bit, including optional histories. Outcomes are 27 step-converged,
five iteration-limited and four empty-initial-map cases. Every accepted real
trial uses the full scale; no iteration limit is relabeled as convergence.

| Case | Step converged | Iteration limit | Empty map | Outer iterations |
| --- | ---: | ---: | ---: | ---: |
| raw-legacy | 8 | 0 | 1 | 64 |
| raw-fine | 6 | 2 | 1 | 99 |
| rotation-legacy | 7 | 1 | 1 | 89 |
| rotation-fine | 6 | 2 | 1 | 122 |

Maximum native/reference pose-entry difference is below `4.90e-15`, matrix
difference below `2.01e-8`, and frozen trial-cost difference below `5.83e-13`.
Correspondences, weights, transported shapes, H/b, every attempted cost,
accepted scale, stopping reason and final refreshed matches/cost are checked.
The reference uses exhaustive distance search and independent NumPy solves;
native queries use the existing exact f64 k-d tree. Exact ties follow native
tree traversal, without a minimum input-index promise. The real run's match
indices also agree with the exhaustive reference.

Each original raw/gyro-only source variant has nine predeclared frame rows
`[0,1,2,9,17,33,49,60,63]`. Maps replay the original captured insertions,
capacity and pruning, with no new pose fed into later maps. Initial predictions
use the original constant relative motion. Global matching, a 2 m strict
threshold, unit weights, 30 iterations and step tolerances `1e-4 m` / `1e-5 rad`
were frozen before implementation/results. Global matching differs from the
odometry stencil: this is not a residual-only comparison to that frontend.

Caller-prepared source/target shapes use exact 20-neighbor centered local PCA,
minimum five neighbors and secondary spread greater than `1e-12` times largest
spread. Normalized eigenvalues are `[.001,1,1]`. Unsupported points are removed
explicitly; none are unsupported here. These shapes describe surface geometry
and are not calibrated covariance. Each outer iteration freezes matches,
weights and `C=Ct+R Cs R.T` through its candidate search. Costs between refreshed
outer surrogates need not decrease. Separate unit controls cover all six
backtracking scales, a shortened accepted step, explicit no-descent stopping,
numeric overflow, singular shapes and the representability boundary.

These are independent local trial poses on old fixed maps, **not a new SLAM
trajectory or a physical accuracy result**. No reference trajectory is parsed,
no calibration/IMU initialization state is fitted or adopted, and no production
frontend/default/gate changes. Native neighborhood preparation and online map
integration remain subsequent work. No timing or GPU claim is made.

The unchanged full-motion component regression covers 660 scans. All three
saved outputs of the 185-scan frontend remain byte-identical to the baseline.
Local native validation covers 621 tests, including 40 new API tests.

Input provenance: [captured map snapshots](../flatwall-odometry-linearization/README.md),
Kenji Koide/AIST, [flatwall recording](https://doi.org/10.5281/zenodo.7641866),
CC BY 4.0. `flatwall_01.bag`, 184,597,802 bytes,
SHA256 `4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6`.
The nine JSON files preserve the scope, controller/build/native identities,
all 272 input snapshot identities, iteration summaries and regression receipts.
Full arrays and histories remain outside Git. `verify.py` checks published
bytes and reported schedule/acceptance, without rerunning the real recording
or independently certifying native/reference agreement.

```powershell
python benchmarks/slam/flatwall-gicp-registration/verify.py
$env:RAYON_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
python scripts/audit_flatwall_gicp_registration.py `
  --input /path/to/frozen/odometry-linearization/audit --out /path/to/fresh/audit `
  --slice-protocol benchmarks/slam/flatwall-gicp-registration/slice-protocol.json `
  --build-inputs benchmarks/slam/flatwall-gicp-registration/native-build-inputs.json
```

A compatible native package and original hashed snapshots are required.
See [API semantics](../../../docs/slam-gicp-registration.md).
