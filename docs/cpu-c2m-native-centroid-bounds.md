# Native C2M parallel centroid bounds experiment

`USE_MESH_PAR_CENTROID_BOUNDS = false` remains the default. Parallel centroid
extrema scans preserve tested BVH bits, order, capacities and complete distance
outputs. Autoware improves by 2.58–3.86%, but PandaSet changes direction between
passes and Tokyo regresses in both. Only 35/60 timed pairs improve overall.
Existing adopted C2M accelerations remain enabled.

The candidate applies to the native parallel mesh builder shared by fresh and
prepared meshes. Measurements cover fresh native `cloud_to_mesh_par` calls;
they establish no retained-mesh query, Python conversion, C2C, browser, WASM or
GPU speedup. No rebuilt Python extension is installed.

## Candidate

Before each original midpoint partition, the builder scans triangle centroids
to choose the widest axis. The candidate parallelizes this scan only at nodes
with at least 262,144 items. It computes extrema and a finite/nonzero guard
in 16,384-item chunks, then reduces them on the existing Rayon pool without a
new heap vector. Eligible finite nonzero extrema have identical bits regardless
of grouping. If any centroid component is zero or nonfinite, it repeats the
complete original serial scan to preserve target-dependent zero/NaN tie bits.
Smaller nodes and sequential/WASM builders retain the preceding scan.

Centroid arithmetic `(a + b + c) / 3`, extent calculations, axis tie policy,
serial unstable midpoint partition, triangle order, preorder node IDs,
child-subtree scheduling and corner order remain unchanged. No floating-point
summation is regrouped. Node/corner/order vector capacities and downstream
query chunks, warm guesses, hit/sign arithmetic and compact result scatter
stay unchanged. No unsafe access or persistent cache is introduced. Rayon
reduction scratch does not establish a peak process memory improvement.

The measured meshes have 3 / 7 / 7 nodes meeting the size condition respectively;
the finite/nonzero guard can still require serial fallback. All parallel scan,
guard, reduction and fallback costs are included in whole timing.

Both builds retain [compact results](cpu-c2m-native-results.md),
[parallel BVH construction](cpu-c2m-native-builder.md),
[bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md) and
[cached bounds](cpu-c2m-native-bounds.md). Rejected
[duplicate seed skipping](cpu-c2m-native-seed.md),
[Morton preparation](cpu-c2m-native-morton.md),
[parallel restoration](cpu-c2m-native-parallel-results.md),
[push-time pruning](cpu-c2m-native-eager-prune.md) and
[triangle-seed reuse](cpu-c2m-native-triangle-seed.md) stay disabled.

## Measured results

[Measured-source preparation](c2c-real-source.md) supplies full unchanged f64
XYZ and original source-index subsets without generated coordinates,
displacement, noise or additional thinning. Tokyo retains its preceding
original-point preparation. These are not independent surveys or accuracy
ground truth. Native decoded PLY order differs from browser LOD order.
Production Delaunay TINs use the same automatic edge rejection and duplicate-XY
handling. Complete geometry PLY bytes match across builds/passes. Reference/
query counts are 531,908/531,908, 878,921/878,920 and 941,933/941,933; TIN vertex/
triangle counts are 501,097/844,324, 873,771/1,688,821 and 893,804/1,466,179.

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve source/configuration
files match except for the centroid-bound flag in `mesh.rs`.
An independent archived baseline supplies complete f64 distances, six exact
f64 statistics and scalar PLY oracles, then exits before timed contexts start.
Only one context computes at a time. One excluded first-use pair and five
alternating warm-up pairs precede ten alternating timed pairs per source/pass.
Initial order is PandaSet → Autoware → Tokyo; the second pass reverses it with
identical build/script fingerprints.

Whole API timing includes fresh BVH construction, serial Morton preparation/
sorting, parallel queries, result allocation/serial scatter, destruction and
pool installation. Loading, TIN/pool creation, first/warm-up observations,
serialization, statistics/parity and cleanup are excluded. No builds, tests,
scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Candidate median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 345.93 | 337.58 | +2.42% | 6/10 |
| PandaSet final | 351.06 | 351.34 | −1.75% | 5/10 |
| Autoware initial | 616.75 | 601.70 | +2.58% | 6/10 |
| Autoware final | 639.83 | 618.01 | +3.86% | 9/10 |
| Tokyo initial | 1,726.85 | 1,730.17 | −0.45% | 5/10 |
| Tokyo final | 1,605.79 | 1,642.13 | −1.36% | 4/10 |

Shortening is `median(100 * (1 - scanned_ms / baseline_ms))`, not a ratio of
marginal medians. Every loss/spike is retained, including initial Autoware's
first trial near 598.6 / 694.8 ms and initial Tokyo's sixth trial near
1,773.3 / 2,010.6 ms. Clocks, power, scheduling, other applications, allocation,
code layout and retained pages are uncontrolled. These whole-call measurements
do not isolate centroid scans, reduction or fallback costs. PandaSet's direction
change and Tokyo's repeated losses do not support enabling this default.
No automatic workload-selection rule is established. Percentages from separate
experiments are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes, six
statistics values and full scalar PLY bytes: 192 outputs and six excluded
oracles. Geometry, thread counts and query counts match. Archived sources,
executables, compiler, script, inputs/manifests and original PCD are checked
before/after. An independent audit recomputes summaries and verifies pair
order, fingerprints and provenance. No GPU model identifiers are collected.

## Validation and reproduction

With the candidate enabled, 410 native tests pass with the parallel feature,
alongside native/WASM Clippy denying warnings, formatting and both release
example builds. The extrema regression compares 198 combinations of 1/2/4
threads, eleven counts through 524,289 items and six coordinate patterns:
UTM, coincident, equal-axis extents, signed zero, nonfinite payloads and finite
overflow/subnormal extents. Min/max bits, selected axes and input capacity match.

The full-builder regression compares 24 combinations against the unchanged
serial Vec-push builder: 1/2/4 threads, 262,143/262,144/262,145/524,288 triangles
and two geometry patterns. The second uses finite nonzero vertices whose
centroid is zero, exercising guarded fallback while bottom-up bounds remain
eligible. Every node's min/max bits and range/child IDs, triangle partition,
centroid bits and vector capacities match. The 524,288-item case exercises
large child scans as well as the root.

The independent original-chunk regression checks 216 finite signed/unsigned
C2M comparisons across four scales and 1/2/4 threads through 65,536 points,
plus eight mixed missing-hit cases. Generated fixtures are correctness tests
only. Benchmark syntax and twelve invalid CLI controls pass.

After restoring the flag to false, the native release is rebuilt; both new
regressions, the independent C2M regression, native Clippy and formatting pass
again. Current twelve source fingerprints match the archived baseline. A
separate correctness-only run checks rebuilt PandaSet geometry, complete f64
distances, statistics and scalar PLY against the archived oracle, recording
the current executable fingerprint. Rebuilt bytes need not equal the archive.

Build the release `bench_c2m_source` example with `--features parallel`, first
with `USE_MESH_PAR_CENTROID_BOUNDS = false`, then true. Keep adopted flags
listed above enabled and all rejected flags disabled. Archive each executable
as `bench_c2m_source.exe`, its twelve source/configuration files under repository
paths and `rustc -Vv` as `compiler.txt` in separate roots. Source guards cannot
identify binary mode; matching builds matter. Run from the repository root:

With current sources, keep `USE_PAR_MESH_INDEXED_CORNERS = false` in both modes
to exclude the subsequent [original-vertex leaf experiment](cpu-c2m-native-indexed-corners.md).

```powershell
node web/scripts/benchmark-native-mesh-centroid-bounds.mjs --baseline-root notes/native-mesh-centroid-bounds-baseline-20261013 --candidate-root notes/native-mesh-centroid-bounds-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-centroid-bounds.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts 1..100
(default ten). Reports/run directories are not overwritten. Other checkouts
need matching archived builds and locally prepared inputs. Native outputs
remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-centroid-bounds-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-centroid-bounds-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-centroid-bounds-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-centroid-bounds-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-centroid-bounds-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-centroid-bounds-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-centroid-bounds.mjs)
- [Builder/scan flags and regressions](../rust/crates/ca-core/src/mesh.rs)
- [Independent original-chunk regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
