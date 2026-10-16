# Real-recording final-pose linearization audit

The opt-in Rust/Python snapshot reproduces independently reconstructed selected
source points, all matched map targets and squared distances, weights, normal
matrices, right-hand sides, support and radius on **4 × 64 real scans**. Both
capture modes reproduce each prior pose bit and report: eight 64-pose sequences.
The initial frame has no map; the other 252 frames have supported spectra.

| Case | Minimum eigenvalue ratio | Median ratio | Frames below `1e-3` |
| --- | ---: | ---: | ---: |
| raw-legacy | 0.01355356 | 0.02141333 | 0 |
| raw-fine | 0.02031256 | 0.02230164 | 0 |
| rotation-legacy | 0.01531217 | 0.02130487 | 0 |
| rotation-fine | 0.02031266 | 0.02211963 | 0 |

All independently reconstructed matrix/vector entries match the snapshots
exactly in this run. Source/world coordinate checks use a `2e-8 m` tolerance.
The independent map replay exhaustively searches all 27 cells, center first,
retains strict first-distance ties, replays per-cell capacity and prunes using
the first point of each cell. It uses the canonical native insertion points,
whose transforms are separately checked, to avoid changing cell-boundary bits.

Geometry is `eigvalsh(sym(D H D)/sum_w)`, with weighted RMS radius about the fixed
prediction center and `D=diag(1,1,1,1/r,1/r,1/r)`. Eigenvalues retain their sign.
These fixed-match, fixed-weight **point-to-point** systems are uncalibrated
conditional diagnostics. Full rank is possible on a physically degenerate plane.
They establish neither physical observability, covariance, GICP geometry,
initialization-failure cause nor trajectory accuracy. There is no matrix
inversion, state adoption, gate/default change, timing or GPU result here.

Raw and gyro-only compensated points use the same frozen 64 references as the
earlier [source-sampling](../flatwall-registration-sampling/README.md) and
[rotation](../flatwall-imu-rotation/README.md) packets. The fine source voxel is
`0.1875 m`; legacy sampling and the `1 m` map/20-point capacity are preserved.
Gyro bias zero, identity extrinsic and acceleration scale `9.80665` are declared
inputs, not calibrated states. The unchanged full-motion controller covers
660 scans; all three 185-scan frontend output files remain byte identical.

Input: Kenji Koide/AIST, [flatwall recording](https://doi.org/10.5281/zenodo.7641866),
CC BY 4.0. Original `flatwall_01.bag`, 184,597,802 bytes,
SHA256 `4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6`.
Scans 1 through 64 are selected in source order; references are maximum exact
integer point timestamps. No reference trajectory is parsed by this audit.

The nine JSON files preserve the predeclared slice, exact controller/native/data
identities, build inputs, per-frame geometry, private snapshot hashes and
regression receipts. Large point/matrix NPZ snapshots remain outside Git.
`verify.py` checks published bytes, case coverage and reported summaries; it
does **not** rerun real-data map matching or independently certify the numbers.

```powershell
python benchmarks/slam/flatwall-odometry-linearization/verify.py
$env:RAYON_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
python scripts/audit_flatwall_odometry_linearization.py `
  --bag /path/to/flatwall_01.bag --out /path/to/fresh/audit `
  --raw-previous /path/to/frozen/registration-sampling/after `
  --rotation-previous /path/to/frozen/rotation `
  --slice-protocol benchmarks/slam/flatwall-odometry-linearization/slice-protocol.json
```

A compatible native package and the original hashed prior receipts, reports and
pose arrays are required for a full rerun. See [API semantics](../../../docs/slam-odometry-linearization.md).
