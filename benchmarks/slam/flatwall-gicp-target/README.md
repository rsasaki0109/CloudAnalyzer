# Real-recording immutable GICP target reuse

Python `GicpTarget` and Rust `PreparedTarget` own validated target points,
explicit world-frame SPD shapes and an exact f64 search tree. Repeated calls
borrow these immutable arrays/tree. Both APIs use the unchanged registration
loop, with the standalone API retaining its validation order and behavior.

On **36 fixed real registration problems with two shape variants (72 runs)**,
all outputs and complete iteration/trial histories match the frozen old native
results and standalone API **bit for bit**. Fresh construction, repeated reuse,
serial/parallel calls and changing constructor arrays after construction all
preserve those bits. Every run also agrees with the independent NumPy/SciPy
iterative oracle under the original numerical tolerances.

| Shape family | Case | Outer iterations | Trial costs | Maximum reference pose-entry error |
| --- | --- | ---: | ---: | ---: |
| caller_shapes | raw-legacy | 64 | 55 | 8.33e-16 |
| caller_shapes | raw-fine | 99 | 92 | 2.48e-15 |
| caller_shapes | rotation-legacy | 89 | 81 | 1.37e-15 |
| caller_shapes | rotation-fine | 122 | 115 | 4.9e-15 |
| native_shapes | raw-legacy | 64 | 55 | 9.02e-16 |
| native_shapes | raw-fine | 99 | 92 | 1.94e-15 |
| native_shapes | rotation-legacy | 89 | 81 | 1.14e-15 |
| native_shapes | rotation-fine | 122 | 115 | 5.62e-15 |

The caller-shape family uses the original SciPy neighborhood/PCA shapes; the
native-shape family uses the frozen f64 shapes from the previous shape audit.
These are two variants of the same fixed problems, not 72 independent scans.
Rows `[0,1,2,9,17,33,49,60,63]` in each of four cases were selected before this
implementation. The strict 2 m matching limit, unit weights, 30 iterations,
fixed predictions/centers and `1e-4 m` / `1e-5 rad` full-step tolerances stay
unchanged. Each family retains 27 step-converged, five iteration-limited and
four empty-target outcomes. No iteration-limited result is relabeled.

Exact tree traversal ties and original target IDs are retained. The constructor
validates all rows, even unmatched ones, and copies original coefficients.
No shape estimation, support filtering, fallback shape or implicit map update
occurs. Frozen Python targets and `Send + Sync` core targets permit read-only
thread sharing; source inputs and output arrays own their storage.

The unchanged 660-scan motion regression passes. The three saved diagnostics,
trajectory and map files from the 185-scan frontend remain byte-identical to
the previous package. Local validation includes 686 native tests (37 new),
nine core registration tests (three new), Clippy and the WebAssembly build.

This is fixed-map numerical validation. No new poses enter later maps, no
reference trajectory is parsed, and no calibration/initializer gate, solver,
default or production frontend changes. These checks do not establish physical
accuracy or whole-SLAM speed. Timing is a separate private protocol and this
packet makes no performance or GPU claim.

Input provenance: [original registration](../flatwall-gicp-registration/README.md)
and [native shape audit](../flatwall-gicp-shapes/README.md).
Kenji Koide/AIST, [flatwall recording](https://doi.org/10.5281/zenodo.7641866),
CC BY 4.0. `flatwall_01.bag`, 184,597,802 bytes,
SHA256 `4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6`.
Nine JSON files preserve schedules, source/native/controller/input identities,
reported outputs and regression receipts. Complete arrays remain outside Git.
LF checkout attributes preserve packet and controller bytes on Windows.

```powershell
python benchmarks/slam/flatwall-gicp-target/verify.py
$env:RAYON_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
python scripts/audit_flatwall_gicp_target.py `
  --input /path/to/frozen/registration/audit `
  --shape-input /path/to/frozen/shapes/audit --out /path/to/fresh/audit `
  --slice-protocol benchmarks/slam/flatwall-gicp-target/slice-protocol.json `
  --build-inputs benchmarks/slam/flatwall-gicp-target/native-build-inputs.json
```

The verifier checks published bytes/identities and reported summaries. It does
not rerun recording inputs or independently certify numerical/physical accuracy.
Reproduction requires both original hashed input sets and a compatible native.
See [API and ownership semantics](../../../docs/slam-gicp-target.md).
