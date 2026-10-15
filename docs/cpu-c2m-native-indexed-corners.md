# Native C2M original-vertex leaf access experiment

`USE_PAR_MESH_INDEXED_CORNERS = false` remains the default. Reading original
mesh vertices instead of copying ordered triangle corners preserves tested
outputs and omits a large temporary array, but all six whole-call comparisons
regress by 5.05–18.47%. Only 4/60 timed pairs improve. Existing copied corners
and adopted native C2M accelerations remain enabled.

## Candidate and storage

A private native `IndexedMeshBvh` borrows the immutable original mesh and
builds the same nodes and triangle-order vector. Its corner vector has zero
length/capacity. Leaf tests fetch the triangle's original f64 vertices by
the same triangle ID, instead of reading a duplicated ordered corner array.
This type exposes only its indexed query method; ordinary `MeshBvh`,
prepared meshes, WASM/hinted APIs and GPU exports retain their copied corners.

Node bits, triangle order, near-first traversal, cached bounds, duplicate seed
leaf tests, strict hit insertion, closest-point/sign/missing-hit arithmetic,
original 4,096-point Morton chunks and per-chunk warm states stay unchanged.
Both modes share the original query/result policy: serial Morton preparation/
key-only unstable sorting and compact f64 result vectors with serial scatter.
No precision conversion, arithmetic regrouping, unsafe access or persistent
cache is introduced. Native C2C is unchanged. Candidate selection requires
cached stack bounds; disabling those retains the copied-corner path.

| Source | TIN triangles | Preceding copied-corner payload, bytes | Candidate payload, bytes |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 844,324 | 60,791,328 | 0 |
| Autoware planning | 1,688,821 | 121,595,112 | 0 |
| Tokyo Takanawadai | 1,466,179 | 105,564,888 | 0 |

Payload is `triangles * 72` bytes, derived from the recorded TIN counts and
the tested exact corner-vector capacities. Nodes/order capacities and final
output capacity remain unchanged. Geometry, query data, vector headers,
allocator metadata and retained pages are excluded. This logical buffer
reduction does not establish peak process RSS or allocator page reduction.
Extra original-vertex gathers and all construction/query/destruction costs
are included in whole timing. The storage benefit is insufficient to enable
the candidate under the current speed-focused default.

Both builds retain [compact results](cpu-c2m-native-results.md),
[parallel construction](cpu-c2m-native-builder.md),
[bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md) and
[cached bounds](cpu-c2m-native-bounds.md). Rejected
[duplicate seed skipping](cpu-c2m-native-seed.md),
[Morton preparation](cpu-c2m-native-morton.md),
[parallel restoration](cpu-c2m-native-parallel-results.md),
[push-time pruning](cpu-c2m-native-eager-prune.md),
[triangle-seed reuse](cpu-c2m-native-triangle-seed.md) and
[centroid bounds](cpu-c2m-native-centroid-bounds.md) stay disabled.

## Measured results

[Measured-source preparation](c2c-real-source.md) supplies full unchanged f64
XYZ and original source-index subsets. No generated coordinates, displacement,
noise or additional thinning is introduced. Tokyo retains its preceding
original-point preparation. These are not independent surveys or accuracy
ground truth. Native decoded PLY order differs from browser LOD order.
Production Delaunay TINs retain automatic edge rejection and duplicate-XY
handling. Complete geometry PLY bytes match across both builds and passes.
Reference/query counts are 531,908/531,908, 878,921/878,920 and 941,933/941,933;
TIN vertex counts are 501,097 / 873,771 / 893,804 respectively.

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve source/configuration
files match except for the indexed-corner flag in `distance.rs`.
An independent archived baseline supplies full f64 distances, six exact
statistics and scalar PLY oracles, then exits before timed contexts start.
Only one context computes at a time. One excluded first-use pair and five
alternating warm-up pairs precede ten alternating timed pairs per source/pass.
Initial order is PandaSet → Autoware → Tokyo; the second pass reverses it with
identical build/script fingerprints.

Whole API timing includes fresh BVH construction, original serial Morton
preparation/sorting, parallel queries, result allocation/scatter, destruction
and pool installation. Loading, TIN/pool creation, first/warm-up observations,
serialization, statistics/parity and cleanup are excluded. No builds, tests,
scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Candidate median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 355.05 | 380.40 | −5.09% | 2/10 |
| PandaSet final | 300.25 | 331.67 | −11.04% | 0/10 |
| Autoware initial | 640.48 | 693.70 | −8.67% | 1/10 |
| Autoware final | 545.01 | 567.69 | −5.05% | 1/10 |
| Tokyo initial | 1,828.14 | 2,127.02 | −12.90% | 0/10 |
| Tokyo final | 1,816.73 | 2,105.37 | −18.47% | 0/10 |

Shortening is `median(100 * (1 - indexed_ms / baseline_ms))`, not a ratio of
marginal medians. Every loss/spike is retained, including initial PandaSet's
tenth trial near 432.7 / 375.8 ms and initial Autoware's sixth trial near
786.7 / 731.3 ms. Clocks, power, scheduling, other applications, allocation,
code layout and retained pages are uncontrolled. The full-call results do not
isolate copy, vertex-gather, cache-miss or allocator costs. They support keeping
the preceding copied-corner default. No memory-constrained workload policy is
established. Percentages from separate experiments are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes, six
statistics values and full scalar PLY bytes: 192 outputs and six excluded
oracles. Geometry, thread counts and query counts match. Archived sources,
executables, compiler, script, inputs/manifests and original PCD are checked
before/after. An independent audit recomputes summaries and verifies pair
order, fingerprints and provenance. No GPU model identifiers are collected.
These measurements establish no retained-mesh query, Python conversion,
browser, WASM or GPU speedup. No rebuilt Python extension is installed.

## Validation and reproduction

With the candidate enabled, 412 native tests pass with the parallel feature,
alongside native/WASM Clippy denying warnings, formatting and both release
example builds. The new hit regression checks 22,272 cases: eight seeds,
six scales, 116 queries and four seed policies (none, previous hit, changing
arbitrary triangle and fixed triangle). Triangle IDs, closest-point bits,
squared-distance bits, signed-distance bits and stack capacities match.
Repeated/reversed and degenerate faces, nonfinite geometry, signed zero,
NaN payloads, infinities and finite-coordinate overflow are included.

The preparation regression checks 18 combinations of 1/2/4 threads,
65,535/65,536/65,537 triangles and finite/nonfinite geometry. Node bits,
triangle order and node/order capacities match. Every copied corner's bits
match the original vertex lookup. Copied corner capacity equals triangle
count; indexed capacity is zero, and the exact storage-byte difference is
72 bytes per triangle. Selected large-mesh hits and missing-hit resets match.

The independent original-chunk regression also checks 216 finite signed/
unsigned C2M comparisons across four scales and 1/2/4 threads through 65,536
points, plus eight mixed missing-hit cases. Generated fixtures are correctness
tests only. Benchmark syntax and twelve invalid CLI controls pass.

After restoring the flag to false, the native release is rebuilt; both new
regressions, the independent C2M regression, native Clippy and formatting pass
again. Current twelve source fingerprints match the archived baseline. A
separate correctness-only run checks rebuilt PandaSet geometry, complete f64
distances, statistics and scalar PLY against the archived oracle, recording
the current executable fingerprint. Rebuilt bytes need not equal the archive.

Build the release `bench_c2m_source` example with `--features parallel`, first
with `USE_PAR_MESH_INDEXED_CORNERS = false`, then true. Keep all adopted flags
listed above enabled, including cached stack bounds; keep all rejected flags
disabled. Archive each executable as `bench_c2m_source.exe`, its twelve source/
configuration files under repository paths and `rustc -Vv` as `compiler.txt`
in separate roots. Source guards cannot identify binary mode; matching builds
matter. Run from the repository root:

```powershell
node web/scripts/benchmark-native-mesh-indexed-corners.mjs --baseline-root notes/native-mesh-indexed-corners-baseline-20261013 --candidate-root notes/native-mesh-indexed-corners-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-indexed-corners.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts 1..100
(default ten). Reports/run directories are not overwritten. Other checkouts
need matching archived builds and locally prepared inputs. Native outputs
remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-indexed-corners-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-indexed-corners-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-indexed-corners-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-indexed-corners-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-indexed-corners-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-indexed-corners-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-indexed-corners.mjs)
- [Private BVH/storage policy and regressions](../rust/crates/ca-core/src/mesh.rs)
- [Native selection and shared query policy](../rust/crates/ca-core/src/distance.rs)
- [Independent original-chunk regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
