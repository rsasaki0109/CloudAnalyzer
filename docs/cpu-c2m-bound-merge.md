# Bottom-up C2M mesh bounds

BVH construction now calculates triangle bounds at leaves and merges child
minima/maxima into internal nodes. `USE_MESH_BOUND_MERGE = true` is adopted.
Against the adopted native parallel-preparation baseline, whole native C2M
calls improve by 26.07–45.36% in six measured-source comparisons, with
60/60 faster timed pairs. Complete distances, statistics and scalar PLY
outputs match the independent preceding-policy oracle.

This shared core change applies to native and WASM construction. Native
measurements include fresh BVH construction on every call; browser tests
establish compatibility, without establishing a browser or GPU speedup.
The native Rust API is used by Python when rebuilt, but Python/NumPy
conversion costs and installation of a rebuilt Python extension are outside
this experiment.

## Computation and memory

Previously, each recursive node scanned the vertices of every triangle in
its subtree to calculate its box. The candidate scans mesh coordinates once
to admit meshes whose every vertex component is finite and nonzero. This
includes unused vertices and conservatively rejects both positive and
negative zero, NaN and infinities. Rejected meshes use the original bounds
scan at every node. The three measured reference inputs contain no zero or
nonfinite coordinates, so their resulting TINs take the new path.

For admitted meshes, leaves retain the original ordered triangle-corner
scan. Each internal node receives the component-wise min/max of its two
children after both are built. Finite nonzero extrema select identical
floating-point values regardless of grouping. Zero and nonfinite fallback
preserves target-dependent tie/payload behavior of the preceding scans.

Centroid arithmetic, centroid bounds and axis selection, unstable
partitioning, preorder node IDs, compact triangle order and corner copies
remain unchanged. In these measured builds, recursive splitting stays sequential, and its centroid
scans/partition work remains; this does not make the entire builder linear.
The change removes repeated triangle-vertex scans at internal levels.
Native indexed centroid/corner preparation and parallel query traversal
remain enabled. No floating-point sums are regrouped or parallelized.

Both measured builds retain adopted [parallel preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md), [cached traversal bounds](cpu-c2m-native-bounds.md)
and [missing-hit recovery](cpu-c2m-missing-hit.md). Query Morton ordering,
4,096-point chunk boundaries, guesses, tie/sign arithmetic and result joining
are unchanged. Existing node/item/order/corner capacities remain identical.
The bounds are written into the existing node vector, without an additional
box buffer or persistent cache. Each measured native call still builds and
destroys its BVH. Retained capacities do not bound peak process memory,
allocator metadata or retained pages.

## Measured results

[Measured-source preparation](c2c-real-source.md) supplies unchanged full
f64 XYZ and source-index subsets without generated coordinates,
displacement, noise or additional thinning. Tokyo retains its preceding
original-point preparation. These workloads are not independent surveys or
accuracy ground truth. Native PLY order differs from browser LOD order.
Full-input production 2.5D Delaunay TINs use automatic edge rejection and
recorded duplicate-XY handling. Full geometry PLY bytes match across both
native builds and passes; equality with earlier browser hashes is not assumed.

| Source | Reference / queries | TIN vertices / triangles | Query chunks |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 501,097 / 844,324 | 130 |
| Autoware planning | 878,921 / 878,920 | 873,771 / 1,688,821 | 215 |
| Tokyo Takanawadai | 941,933 / 941,933 | 893,804 / 1,466,179 | 230 |

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve source/configuration
files match except for the bound-merge flag in `mesh.rs`. Input/TIN and idle
pools remain resident in separate processes; only one computes at a time.

An independent archived baseline supplies full raw f64 distances, six exact
f64 summary values and scalar PLY oracles, then exits before timing. One
excluded first-use pair and five alternating warm-up pairs precede ten
alternating timed pairs per source/pass. Initial order is PandaSet → Autoware
→ Tokyo; the second pass reverses it with identical build/script fingerprints.
Whole API timing includes fresh BVH construction, Morton ordering, parallel
queries, joining, BVH destruction and Rayon pool installation. Loading,
TIN/pool creation, first/warm-up observations, serialization, statistics/parity
and cleanup are excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Original bounds baseline median, ms | Merged bounds median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 951.36 | 584.32 | +38.93% | 10/10 |
| PandaSet final | 759.63 | 475.33 | +38.15% | 10/10 |
| Autoware initial | 1,871.76 | 1,037.29 | +44.89% | 10/10 |
| Autoware final | 2,114.70 | 1,164.53 | +45.36% | 10/10 |
| Tokyo initial | 2,971.31 | 2,175.83 | +26.07% | 10/10 |
| Tokyo final | 2,979.68 | 2,088.78 | +27.68% | 10/10 |

Shortening is `median(100 * (1 - merged_ms / baseline_ms))`, not a ratio of
marginal medians. All losses/spikes are retained, including initial Tokyo
observations near 5,036.7 / 3,118.3 ms and final Autoware observations near
4,577.0 / 2,182.9 ms. Clocks, power, scheduling, other applications,
allocation, code layout and retained pages are uncontrolled. Adoption is
based on consistent improvement across both passes of all three workloads
and unchanged outputs/capacities; it does not promise improvement on every
input or call. No isolated BVH construction, other-platform, Python conversion,
browser, retained-mesh query or GPU speedup is claimed. Percentages from
different experiments are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes,
six f64 summary values and full scalar PLY bytes: 192 outputs and six excluded
oracles. Geometry, threads and query counts remain identical. Archived source/
executable sets, script, inputs, manifests and original PCD are checked
before/after. A separate audit recomputes summaries and verifies pair order,
build fingerprints and provenance. No GPU model identifiers are collected.

## Validation and reproduction

401 native tests pass with the parallel feature, together with native/WASM
Clippy denying warnings, formatting, native release builds, the WASM release
build and TypeScript/Vite production build. The new native regression
compares every node's f64 bound bits, node IDs/leaf ranges, partitioned
triangle/centroid bits and capacities in 224 scale/count/seed cases. Separate
positive/negative-zero, NaN and infinity cases verify the original fallback.
The existing threshold, missing-hit and original-query-chunk regressions pass.

14 focused browser tests pass; three physical-GPU opt-in tests are skipped.
The new archived-production WASM comparison checks fresh/prepared signed and
unsigned f64 distances across 240 geometry/scale/query-count conditions,
repeated changed queries, retained mesh bytes and exported GPU node/corner
bytes/order. GPU buffer export is tested on the CPU; no hardware GPU execution
is established by this check. Generated fixtures are for correctness only.
Benchmark syntax and twelve invalid CLI controls pass.

Build `bench_c2m_source` in release mode with `--features parallel`, first
with `USE_MESH_BOUND_MERGE = false`, then true. Keep parallel preparation,
native stack reuse and cached traversal bounds enabled in both. With current
sources, keep `USE_MESH_PAR_BUILD = false` to reproduce this comparison
before [parallel tree construction](cpu-c2m-native-builder.md). Also keep
`USE_PAR_MESH_COMPACT_RESULTS = false` before
[compact temporary results](cpu-c2m-native-results.md). Archive each
executable as `bench_c2m_source.exe`, its twelve source/configuration files
under repository paths and `rustc -Vv` as `compiler.txt` in separate roots.
Source guards cannot identify binary mode; matching builds matter. Run from
the repository root with unchanged measured inputs and a new output:

```powershell
node web/scripts/benchmark-native-mesh-bound-merge.mjs --baseline-root notes/native-mesh-bound-merge-baseline-20261013 --candidate-root notes/native-mesh-bound-merge-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-bound-merge.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts
1..100 (default ten). Reports/run directories are not overwritten. Other
checkouts need equivalent archived builds and locally prepared inputs.
Native output files remain under ignored `notes/`.

Set `MESH_BOUND_MERGE_BASELINE` to an archived preceding production root
containing `dist/assets/ca_wasm*.wasm` and `web/src/wasm/ca_wasm.js`, build the
candidate, then run `e2e/mesh-bound-merge.spec.ts` from `web` to compare WASM
results and exported GPU buffers.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-bound-merge-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-bound-merge-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-bound-merge-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-bound-merge-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-bound-merge-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-bound-merge-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-bound-merge.mjs)
- [Builder and native regressions](../rust/crates/ca-core/src/mesh.rs)
- [Archived WASM regression](../web/e2e/mesh-bound-merge.spec.ts)
