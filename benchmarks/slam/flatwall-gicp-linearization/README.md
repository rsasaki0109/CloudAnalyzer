# Real-recording fixed-match GICP factor audit

The standalone Rust/Python GICP kernel agrees with independent NumPy matrix
algebra on **256 captured real frames / 118,040 actual matched pairs**. All pairs
retain valid declared shape neighborhoods in this run. Serial/parallel outputs
match every bit. Identity combined-metric controls reproduce the point-to-point
system on the same pairs and weights. Each initial frame has no map; 252 frames
have supported conditional spectra.

| Case | Retained pairs | Minimum GICP ratio | Minimum P2P ratio | GICP frames below `1e-3` |
| --- | ---: | ---: | ---: | ---: |
| raw-legacy | 1,698 | 0.00106031 | 0.01355356 | 0 |
| raw-fine | 57,340 | 0.00310160 | 0.02031256 | 0 |
| rotation-legacy | 1,692 | 0.00079348 | 0.01531217 | 1 |
| rotation-fine | 57,310 | 0.00251777 | 0.02031266 | 0 |

Maximum absolute matrix disagreement is `1.61e-8`; RHS disagreement is below
`6.05e-11` and cost disagreement below `1.09e-12`. Tolerances were frozen before
implementation/results. The shape model uses 20 nearest neighbors (minimum five),
centered PCA, secondary spread above `1e-12` times largest spread and normalized
eigenvalues `[0.001,1,1]`. Unsupported pairs would be removed explicitly from
both compared models. No planarity selection, identity fallback or gate tuning
is applied. These are declared surface shapes, **not calibrated covariance**.

Sources and source shapes use sensor axes; target points/shapes use the replayed
pre-insertion world map. `C=C_target+R C_source R.T` is held fixed with the
captured correspondences and Euclidean GMC weights during differentiation.
Parameters are translation xyz then world rotation xyz about the captured fixed
prediction center. Cholesky whitening builds the system without an explicit
inverse. The independent reference uses NumPy linear solves and contractions.

Spectra use the same retained-weight RMS arm radius for both models and
`eigvalsh(sym(D H D)/sum_w)`, `D=diag(1,1,1,1/r,1/r,1/r)`. Signed eigenvalues are
preserved. Shape anisotropy changes the metric and its spectrum; this is an
uncalibrated conditional comparison. A smaller/larger objective value or
eigenvalue ratio across models does **not** establish trajectory accuracy,
physical observability or an initialization-failure cause. No information
matrix inversion, optimizer, new trajectory or state adoption occurs here.

The kernel is a separate API. Existing odometry/ICP solvers, correspondence
searches, defaults and initialization gates are unchanged. Full iterative GICP
registration and native neighborhood preparation remain separate work. This
packet makes no timing or GPU claim. The unchanged full-motion regression covers
660 scans; the 185-scan frontend's three saved output files remain byte identical.

Input provenance is the [original captured-match packet](../flatwall-odometry-linearization/README.md).
Kenji Koide/AIST, [flatwall recording](https://doi.org/10.5281/zenodo.7641866),
CC BY 4.0. `flatwall_01.bag`, 184,597,802 bytes,
SHA256 `4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6`.
The same raw/gyro-only compensated source variants and 64 references are used.
No reference trajectory is parsed. No sensor calibration is fitted or adopted.

Nine JSON files preserve the frozen scope, controller/native/build identities,
captured input hashes, all frame summaries, private NPZ hashes and regression
receipts. Large correspondence/shape/system arrays remain outside Git.
`verify.py` checks published bytes and reported summaries; it does not rerun the
recording or independently certify shape estimation and native matrix agreement.

```powershell
python benchmarks/slam/flatwall-gicp-linearization/verify.py
$env:RAYON_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
python scripts/audit_flatwall_gicp_linearization.py `
  --input /path/to/frozen/odometry-linearization/audit --out /path/to/fresh/audit `
  --slice-protocol benchmarks/slam/flatwall-gicp-linearization/slice-protocol.json `
  --build-inputs benchmarks/slam/flatwall-gicp-linearization/native-build-inputs.json
```

A compatible native package and all original hashed capture files are required.
See [API semantics and derivative limits](../../../docs/slam-gicp-linearization.md).
