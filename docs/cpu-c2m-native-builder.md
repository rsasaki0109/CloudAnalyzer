# Native parallel C2M tree construction

Native BVH construction now joins independent child subtrees on the active
Rayon pool for large meshes. `USE_MESH_PAR_BUILD = true` is adopted.
Against the adopted bottom-up-bounds baseline, whole native C2M calls improve
by 14.49–33.45% in six measured-source comparisons, with 60/60 faster timed
pairs. Complete distances, statistics and scalar PLY outputs match the
independent preceding-policy oracle. Node ordering and retained capacities
also match in the native regressions.

This applies to native builds with the `parallel` feature, including the
Rust API used by Python when rebuilt. Measurements exclude Python/NumPy
conversion costs and installation of a rebuilt Python extension. Smaller
meshes and builds without the parallel feature retain the sequential builder;
no browser, WASM or GPU speedup is established by these observations.

## Computation and memory

The existing indexed preparation path admits native meshes with at least
65,536 triangles. The new builder pre-sizes the existing node vector to its
exact required length. Midpoint splits depend only on triangle count, so the
left subtree's node count and both children's preorder ranges can be known
before their geometry is processed. A constant-time count uses the last
complete tree level and the number of its subtrees that exceed the four-face
leaf limit. The initial vector capacity and its growth policy are preserved.
Every slot is initialized before being borrowed; no unsafe uninitialized
memory is introduced.

After the original serial centroid-bound scan, axis choice and unstable
midpoint partition, child item slices and child node slices are disjoint.
Nodes with at least 32,768 triangles use `rayon::join` to construct those
children concurrently. Smaller subtrees recurse sequentially within their
assigned node slice. Parent bounds are merged only after both children
finish. Completion order cannot change preorder node IDs, leaf triangle
offsets, ordered corner copies or search order. Each subtree performs the
same component arithmetic and partition operations as the preceding builder;
no floating-point reduction is parallelized.

Both measured builds retain adopted [bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed preparation](cpu-c2m-native-preparation.md), [stack reuse](cpu-c2m-native-stack.md),
[cached traversal bounds](cpu-c2m-native-bounds.md) and
[missing-hit recovery](cpu-c2m-missing-hit.md). Zero/nonfinite meshes keep the
original per-node bounds scans in their original item order. Query Morton
ordering, 4,096-point chunk boundaries, guesses, tie/sign arithmetic and
result joining are unchanged. Parallel work uses the active/default Rayon
pool; the benchmark installs a fixed 12-thread pool around each whole call.

The builder writes the existing node vector rather than allocating separate
subtree vectors. Existing node/item/order/corner capacities remain identical,
and no persistent geometry/query/result cache is added. Each measured call
still builds and destroys its BVH. Pre-sizing, node initialization, task
scheduling and all recursive work are included in timing. Retained capacities
do not bound peak process memory, task/stack overhead, allocator metadata or
retained pages.

## Measured results

[Measured-source preparation](c2c-real-source.md) supplies unchanged full
f64 XYZ and source-index subsets without generated coordinates, displacement,
noise or additional thinning. Tokyo retains its preceding original-point
preparation. These workloads are not independent surveys or accuracy ground
truth. Native PLY order differs from browser LOD order. Full-input production
2.5D Delaunay TINs use automatic edge rejection and recorded duplicate-XY
handling. Full geometry PLY bytes match across both native builds/passes;
equality with earlier browser mesh hashes is not assumed.

| Source | Reference / queries | TIN vertices / triangles | Original query chunks |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 501,097 / 844,324 | 130 |
| Autoware planning | 878,921 / 878,920 | 873,771 / 1,688,821 | 215 |
| Tokyo Takanawadai | 941,933 / 941,933 | 893,804 / 1,466,179 | 230 |

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve source/configuration
files match except for the parallel-builder flag in `mesh.rs`. Input/TIN and
idle pools remain resident in separate processes; only one computes at a time.

An independent archived baseline supplies full raw f64 distances, six exact
f64 summary values and scalar PLY oracles, then exits before timing. One
excluded first-use pair and five alternating warm-up pairs precede ten
alternating timed pairs per source/pass. Initial order is PandaSet → Autoware
→ Tokyo; the second pass reverses it with identical build/script fingerprints.
Whole API timing includes fresh BVH construction, Morton ordering, parallel
queries, joining, BVH destruction and Rayon pool installation. Loading,
TIN/pool creation, first/warm-up observations, serialization, statistics/parity
and cleanup are excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Sequential-builder baseline median, ms | Parallel-builder median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 421.16 | 303.75 | +26.21% | 10/10 |
| PandaSet final | 600.24 | 423.23 | +31.63% | 10/10 |
| Autoware initial | 781.83 | 527.06 | +32.93% | 10/10 |
| Autoware final | 932.74 | 622.92 | +33.45% | 10/10 |
| Tokyo initial | 1,854.12 | 1,601.99 | +14.49% | 10/10 |
| Tokyo final | 1,944.42 | 1,617.27 | +15.12% | 10/10 |

Shortening is `median(100 * (1 - parallel_ms / baseline_ms))`, not a ratio of
marginal medians. Every observation is retained, including final PandaSet's
candidate trial near 508.5 ms against 553.6 ms for the baseline. Absolute
times shift between passes. Clocks, power, scheduling, other applications,
allocation, code layout and retained pages are uncontrolled. Adoption is
based on consistent improvement across both passes of all three workloads
and unchanged outputs/capacities; it does not promise improvement on every
input or call. No isolated construction/traversal, other-platform, Python
conversion, browser, WASM, retained-mesh query or GPU speedup is claimed.
Percentages from different experiments are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes,
six f64 summary values and full scalar PLY bytes: 192 outputs and six excluded
oracles. Geometry, threads and query counts remain identical. Archived source/
executable sets, script, inputs, manifests and original PCD are checked
before/after. A separate audit recomputes summaries and verifies pair order,
build fingerprints and provenance. No GPU model identifiers are collected.

## Validation and reproduction

403 native tests pass with the parallel feature, together with native/WASM
Clippy denying warnings, formatting and both native release example builds.
The new node-count regression compares the formula against the preceding
recursive leaf policy for 8,199 triangle counts. The layout regression checks
108 combinations of 1/2/4 threads, triangle counts around leaf and
32,768/65,536 thresholds, ordinary map coordinates, signed zero and nonfinite
coordinates. Every node's f64 bound bits, preorder ranges, partitioned
triangle/centroid bits and node/item capacities match the frozen sequential
builder. Degenerate and reversed faces are included. Existing exact retained
capacity, missing-hit, signed-distance and original-query-chunk regressions
also pass. Generated fixtures are for correctness only. Benchmark syntax and
twelve invalid CLI controls pass.

The new builder and its node-count helpers are compiled only with the native
parallel feature. WASM compilation keeps the existing sequential path; this
experiment adds no browser/GPU execution tests or browser timing claims.

Build `bench_c2m_source` in release mode with `--features parallel`, first
with `USE_MESH_PAR_BUILD = false`, then true. Keep parallel preparation,
bottom-up bounds, native stack reuse and cached traversal bounds enabled in
both. Keep `USE_PAR_MESH_SKIP_GUESS = false` to reproduce this comparison
before the [seed-triangle experiment](cpu-c2m-native-seed.md). Also keep
`USE_PAR_MESH_MORTON_PREPARATION = false` before the later
[Morton experiment](cpu-c2m-native-morton.md). Keep `USE_PAR_MESH_COMPACT_RESULTS = false`
before [compact temporary results](cpu-c2m-native-results.md). Archive each
executable as `bench_c2m_source.exe`, its twelve source/
configuration files under repository paths and `rustc -Vv` as `compiler.txt`
in separate roots. Source guards cannot identify binary mode; matching builds
matter. Run from the repository root with unchanged measured inputs and a
new output:

```powershell
node web/scripts/benchmark-native-mesh-par-build.mjs --baseline-root notes/native-mesh-par-build-baseline-20261013 --candidate-root notes/native-mesh-par-build-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-par-build.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts
1..100 (default ten). Reports/run directories are not overwritten. Other
checkouts need equivalent archived builds and locally prepared inputs.
Native output files remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-par-build-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-par-build-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-par-build-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-par-build-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-par-build-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-par-build-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-par-build.mjs)
- [Builder and native regressions](../rust/crates/ca-core/src/mesh.rs)
