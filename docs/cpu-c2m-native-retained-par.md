# Native parallel C2M with a retained BVH

`PreparedMesh::distances_par` adds an opt-in native parallel query API for
an owned immutable mesh/BVH. Repeated queries on the same measured-source
TIN shorten by 14.09–35.21% compared with rebuilding the BVH per call.
All six comparisons and all 60 timed pairs improve. Complete tested outputs
match an independent preceding production oracle.

This benefit requires retaining the same mesh. Initial construction and
additional persistent BVH storage are recorded separately. The stateless
`cloud_to_mesh_par` API keeps its existing behavior; Python/browser/WASM/GPU
API selection was unchanged for these native measurements; no Python extension
was rebuilt or installed at that stage. The subsequent
[Python prepared-mesh API](cpu-c2m-python-prepared-mesh.md) measures complete
NumPy binding calls separately, including its geometry-copy ownership costs.

## API and ownership

With the `parallel` feature enabled:

```rust
use ca_core::mesh::PreparedMesh;

// mesh is an owned TriangleMesh; construction validates its faces once.
let prepared = PreparedMesh::new(mesh).expect("nonempty validated mesh");
let first = prepared.distances_par(&points, false);
let next = prepared.distances_par(&other_points, true);
let retained_capacity_bytes = prepared.retained_bytes();
```

Construction takes ownership of the geometry and builds the existing BVH.
The geometry/BVH is private and immutable. Query calls borrow its storage,
without copying geometry or rebuilding the BVH. The object is `Send + Sync`
and may be shared through `Arc` by concurrent readers. Each caller owns its
result and call-local scratch. Dropping the object releases its owned vectors;
there is no new global cache, eviction policy or hidden retained object.
Construct another object when the reference mesh changes.

The fresh and retained APIs share the same copied-corner parallel query
kernel. Original serial Morton preparation/key-only unstable sorting,
4,096-point chunks, per-chunk previous-hit guesses/cached stacks, clamped
sibling-bound SIMD, scalar triangle projection, strict hit insertion,
sign/missing-hit arithmetic, compact f64 results and serial scatter stay
unchanged. Order, guesses and scratch reset on every call. Changed query
points, lengths or sign settings do not reuse an earlier query's state.

The existing constructor still drops invalid faces and returns `None` when
no valid triangles remain. Nonfinite/overflow queries preserve missing-hit
behavior. The optional rejected indexed-corner policy is not used by this
owned copied-corner API. Other rejected experiments stay disabled; preceding
adopted preparation, bounds and parallel-construction policies stay enabled.
No precision conversion, new projection algorithm or unsafe access is added.

## Measured scope

[Measured-source preparation](c2c-real-source.md) supplies full unchanged f64
XYZ and original source-index subsets. No generated/transformed coordinates,
displacement, noise or extra thinning is introduced. Tokyo retains its
preceding original-point preparation. These subsets are not independent
surveys or accuracy ground truth; native PLY order differs from browser LOD
order. Production Delaunay TINs keep automatic edge rejection and duplicate-XY
handling. Both contexts receive the same complete geometry bytes.

| Source | Reference/query points | TIN vertices | TIN triangles |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 501,097 | 844,324 |
| Autoware planning | 878,921 / 878,920 | 873,771 | 1,688,821 |
| Tokyo Takanawadai | 941,933 / 941,933 | 893,804 | 1,466,179 |

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run one frozen native
fresh/retained executable with fixed 12-thread pools, Rust 1.98.1/
x86_64-pc-windows-msvc, optimization level 3, LTO and one codegen unit.
An independent preceding archived production executable supplies complete
f64 distances, six statistics and scalar PLY oracles, then exits before
the current contexts start. Current production sources differ from that
oracle only in the two files adding the parallel retained API/shared kernel;
the other ten source/configuration fingerprints match.
Measured queries are unsigned; signed behavior is covered by the regressions.

Fresh and retained modes run in separate contexts using the same current
binary. Only one computes at a time. Retained setup moves the original TIN
vectors into `PreparedMesh` without cloning or reordering them. Setup time
and capacity bytes are recorded before query measurements. Fresh mode retains
the original geometry and builds/drops its BVH during each query. One excluded
first-observation pair and five alternating warm-up pairs precede ten
alternating timed pairs per source/pass. The first pass runs PandaSet →
Autoware → Tokyo; the second reverses that order with identical fingerprints.

Repeated-call clocks include pool installation, Morton work, parallel
queries, final allocation/scatter and call-local destruction. Fresh clocks
also include BVH construction/destruction. The initial retained constructor
and final retained-object destruction are outside these clocks. Loading,
meshing, pool creation, first/warm-up observations, statistics/serialization,
complete-output checks and cleanup are excluded. No builds, tests, scans or
edits run alongside timing.

| Source/pass | Fresh median, ms | Retained median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 355.16 | 256.35 | +28.47% | 10/10 |
| PandaSet final | 516.45 | 374.56 | +29.91% | 10/10 |
| Autoware initial | 678.76 | 458.57 | +33.96% | 10/10 |
| Autoware final | 696.09 | 445.21 | +35.21% | 10/10 |
| Tokyo initial | 1,979.71 | 1,729.60 | +14.71% | 10/10 |
| Tokyo final | 1,744.99 | 1,521.57 | +14.09% | 10/10 |

Shortening is `median(100 * (1 - retained_ms / fresh_ms))`, not a ratio of
marginal medians. All outliers remain, including final PandaSet's first three
pairs near 1,121.7 / 531.3, 1,109.8 / 557.4 and 926.7 / 557.9 ms, and initial
Autoware's seventh near 1,315.5 / 714.5 ms. Absolute times shift between passes.
Clocks, power, scheduling, other applications, code layout, allocator behavior
and retained pages are uncontrolled. These whole-call results do not isolate
construction savings from query/cache/ownership effects. Separate experiments'
percentages are not additive, and the preceding stage shares are historical
diagnostics rather than a prediction of this speedup.

## Initial cost and retained capacity

Setup clocks include face validation, existing BVH construction and pool
installation, with loading/TIN/pool creation already complete. They are one
observed constructor cost per context, not repeated constructor benchmarks.
The first-query totals below include that setup for the retained mode:

| Source/pass | Retained setup, ms | Fresh first query, ms | Setup + retained first query, ms |
| --- | ---: | ---: | ---: |
| PandaSet initial | 96.78 | 360.53 | 345.56 |
| PandaSet final | 102.47 | 376.75 | 378.18 |
| Autoware initial | 211.67 | 681.75 | 630.28 |
| Autoware final | 227.13 | 662.62 | 641.10 |
| Tokyo initial | 196.63 | 1,878.62 | 1,837.92 |
| Tokyo final | 185.83 | 1,544.56 | 1,552.90 |

`ceil(setup_ms / median_paired_saved_ms)` estimates one query to amortize setup
in all six reports. This is not a single-query guarantee: the first observed
PandaSet/Tokyo totals in the final pass are slightly slower. Execution order
is fixed for these observations, and allocator/page state is uncontrolled.
The estimate excludes loading, meshing, pool creation, conversion and final
object destruction. The supported improvement concerns repeated queries on
the same retained geometry; no cold-process or full lifecycle speedup is
established.

| Source | Original geometry capacity, bytes | Additional retained BVH, bytes | Total owned mesh/BVH, bytes |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 25,165,824 | 111,450,880 | 136,616,704 |
| Autoware planning | 50,331,648 | 222,924,428 | 273,256,076 |
| Tokyo Takanawadai | 50,331,648 | 193,535,684 | 243,867,332 |

These exact logical capacities match both passes. Additional BVH bytes are
node capacity times 56 plus triangle-order capacity times 4 plus copied-corner
capacity times 72. Independent audits derive the node growth capacity from
the original midpoint construction and verify every total. Both contexts
already own the geometry. Headers, allocator metadata, retained pages, input
clouds, query/results and temporary construction/query buffers are excluded.
This is not a process RSS or peak-memory comparison. Retaining approximately
111–223 MB of BVH payload per measured mesh is an explicit lifetime tradeoff.

## Validation and reproduction

All 96 first/warm-up/timed pairs match the complete f64 distances, six exact
statistics and scalar PLY bytes: 192 outputs and six excluded oracles.
Geometry, thread/query counts and preceding historical oracles match. Sources,
executables, compiler, example/regressions, controller, inputs/manifests and
original PCD fingerprints are checked before/after. Independent audits
recompute summaries, setup/amortization values, exact vector payloads and
pair ordering. No GPU model identifiers are collected.

419 native tests pass with the parallel feature, alongside native all-target/
WASM Clippy denying warnings, formatting and both release examples. The
independent original-chunk regression now checks fresh and retained APIs
against its separate preceding traversal for 216 finite signed/unsigned cases
across four scales and 1/2/4 threads through 65,537 points, plus eight mixed
missing-hit cases. Retained capacity stays constant. Existing missing-hit and
all-uncomputable-mesh regressions also exercise the new API.

The additional ownership test checks source mutation/drop independence,
invalid-face validation, changed/reversed batches, changing sign settings,
empty queries and 4,096-point chunk boundaries across 1/2/4 threads. The
additional shared-object test checks `Send + Sync` and eight concurrent reads across two rounds using
one four-thread Rayon pool, with complete bit comparisons and fixed capacity.
Generated fixtures are correctness tests only. Controller syntax and twelve
invalid CLI controls pass. The current classic stateless release also matches
the full original PandaSet geometry/distance/statistics/PLY oracle in a
correctness-only check.

The Native Rust API requires opting into `PreparedMesh::distances_par`.
Existing Python conversion, browser/WASM worker caching and GPU selection are
not switched by this change; these results establish no improvement for
those paths or for changing reference meshes.

Build release `bench_c2m_retained` and `bench_c2m_source` examples with
`--features parallel`. Archive the retained example as
`bench_c2m_retained.exe`, twelve source/configuration files, the new example
and two regression files under their repository paths, and `rustc -Vv` as
`compiler.txt`. Keep all adopted policies enabled and all rejected policies
disabled. Use a separate preceding archived `bench_c2m_source.exe` root as
the oracle; the controller permits only the API's two source files to differ.
Source guards cannot identify executable contents; matching frozen builds
matter. From the repository root:

```powershell
node web/scripts/benchmark-native-mesh-retained-par.mjs --oracle-root notes/native-mesh-projection-pair-baseline-20261014 --native-root notes/native-mesh-retained-par-20261014 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-retained-par.json --threads 12
```

`--threads` accepts 1..64 (default 12); `--repeats` accepts 1..100 (default
ten). Reports/run directories are not overwritten. Other checkouts need
matching archives and locally prepared inputs. Native artifacts remain under
ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-retained-par-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-retained-par-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-retained-par-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-retained-par-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-retained-par-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-retained-par-final.json)
- [Controller](../web/scripts/benchmark-native-mesh-retained-par.mjs), [native protocol example](../rust/crates/ca-core/examples/bench_c2m_retained.rs)
- [Owned API](../rust/crates/ca-core/src/mesh.rs), [shared parallel kernel](../rust/crates/ca-core/src/distance.rs)
- [Independent chunk regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs), [ownership/concurrency regressions](../rust/crates/ca-core/tests/mesh_preparation.rs)
