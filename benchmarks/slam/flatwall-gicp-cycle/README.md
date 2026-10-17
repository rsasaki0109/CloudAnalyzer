# Flatwall GICP bounded cycle diagnosis

This packet records an explicit opt-in recurrence guard on the same 36 fixed
real registration problems, with both caller-prepared and native-prepared
shapes (72 histories). The input points, shapes, initial poses, centers and
30-iteration limit come from the [immutable-target study](../flatwall-gicp-target/README.md).
The recorded Flatwall data are credited there; unit controls are separate.

The [slice protocol](slice-protocol.json) was fixed before implementation and
new classifications. An independent NumPy detector first read the saved
histories. A fresh native package, built from the
[frozen source hashes](native-build-inputs.json), then replayed every problem.
The native package identity is in [native-artifact.json](native-artifact.json).
Original controllers, successful and failed attempts, protocols, source/native
snapshots and raw arrays remain preserved in private study directories.

## Observed outcomes

| Termination | Original / guard disabled | Guard enabled |
| --- | ---: | ---: |
| `converged_step` | 54 | 54 |
| `max_iterations` | 10 | 6 |
| `insufficient_correspondences` | 8 | 8 |
| `cycle_detected` | 0 | 4 |

The guard witnesses two distinct fixed inputs in both shape variants:

| Case / row | Period | Detecting iteration | Accepted steps |
| --- | ---: | ---: | ---: |
| raw-fine / 60 | 3 | 29 | 28 |
| rotation-fine / 49 | 7 | 29 | 28 |

Each witness covers every phase over two complete returns. The other three
limited inputs in both shape variants retain `max_iterations`. The previously
observed approximate periods 2, 3 and 5 do not meet this stricter full-witness
contract within the unchanged 30-iteration limit. No bounds were relaxed to
classify them. See [all independent decisions](prototype/frames.json),
[native outcomes](audit/frames.json) and [aggregate checks](audit/summary.json).

All disabled outputs, including histories and floating-point bits, equal the
original frozen results. Enabled calls with no witness also retain every bit.
For detected cycles, the accepted prefix, last accepted pose and terminal
rematched IDs/weights/cost match the corresponding original baseline exactly.
Serial/parallel execution, standalone/prepared targets, copied constructor
ownership and history on/off agree. Thirty-two shared-target calls on the four
cycling histories agree, and mutating returned data does not affect reuse.

The [prototype receipt](prototype/receipt.json) and [native receipt](audit/receipt.json)
record input/controller identities and before/after checks. The native receipt
also hashes complete captured output arrays, which are not included here.
The same complete native package was used for the native unit suite and
unchanged real motion/frontend regression checks recorded privately.

## Interpretation and reproduction

This is a failure diagnosis, not convergence recovery. The guard returns
`converged=False` and the last accepted pose. It is disabled by default and is
not adopted by the odometry frontend. These fixed local problems do not measure
trajectory accuracy, physical observability, GPU speed or whole-SLAM speed.
The full [API contract](../../../docs/slam-gicp-cycles.md) specifies bounded
memory, literal IDs, frame-aware pose comparison, shape/weight bounds and
failure/convergence ordering.

`scripts/audit_flatwall_gicp_cycles.py` supplies the independent saved-history
prototype. `scripts/audit_flatwall_gicp_cycle_guard.py` replays the native API
and checks accepted prefixes. Both require the original private arrays and
fresh output paths; their command-line help lists arguments. Native replay
uses `RAYON_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, and a complete private
package on `PYTHONPATH`; it does not replace an installed extension.

Run `python benchmarks/slam/flatwall-gicp-cycle/verify.py` to verify the shipped
JSON bytes, reported witness bounds and status accounting. This lightweight
check does not independently rerun real data or the optimizer. Public files
and their two controllers use explicit LF checkout rules so recorded hashes
also survive a fresh Windows `core.autocrlf=true` checkout.
