# Native C2M previous-hit triangle seed experiment

`USE_PAR_MESH_TRIANGLE_SEED = false` remains the default. Reusing the previous
hit's corners and edges preserves tested outputs, but only 24/60 timed pairs
improve. PandaSet and Tokyo change direction between passes; Autoware
regresses by 1.60–2.48% in both. Existing adopted C2M accelerations stay enabled.
These are native CPU observations, with no Python conversion, C2C, browser,
WASM or GPU speedup established and no rebuilt Python extension installed.

## Candidate

Each original 4,096-point Morton query chunk starts with an empty optional
triangle seed. After a successful query, the candidate stores the winning
triangle's original f64 corners and the two original edge subtractions
`b - a` and `c - a`. It replaces this seed only when the winning triangle
changes, and clears it when no triangle has a computable distance. The next
warm-start test uses the stored values. Every remaining closest-point and
distance expression retains the preceding evaluation order. The original
projection function remains available as the independent comparison.

Near-first traversal, cached bounds, leaf order, the duplicate seed leaf test,
strict insertion, signed-distance arithmetic and missing-hit recovery are
unchanged. Serial Morton preparation/key-only unstable sorting, equal-key
order, query chunk boundaries, compact f64 result vectors and serial scatter
also stay unchanged. Only native parallel C2M selects the candidate;
sequential/prepared/WASM/hinted and native C2C APIs retain their preceding policy.

No per-mesh edge array, new heap buffer, unsafe access, precision conversion or
floating-point regrouping is added. One optional seed enlarges chunk-local
state in both archived builds; the candidate populates and reuses it. Chunks
release their state at completion, with no seed retained across calls. Mesh/BVH
payload and tested traversal-stack/final-result capacities remain unchanged.
Replacement, copying and edge preparation, including preparation after a
chunk's final query that will not be reused, are included in timing.

Both builds retain [compact results](cpu-c2m-native-results.md),
[parallel BVH construction](cpu-c2m-native-builder.md),
[bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md) and
[cached bounds](cpu-c2m-native-bounds.md). Rejected
[duplicate seed skipping](cpu-c2m-native-seed.md),
[Morton preparation](cpu-c2m-native-morton.md),
[parallel restoration](cpu-c2m-native-parallel-results.md) and
[push-time pruning](cpu-c2m-native-eager-prune.md) stay disabled.

## Measured results

[Measured-source inputs](c2c-real-source.md) retain full unchanged f64 XYZ and
original source-index subsets, with no generated coordinates, displacement,
noise or additional thinning. Tokyo retains its preceding original-point
preparation. These are not independent surveys or accuracy ground truth.
Native decoded PLY order differs from browser LOD order. Production Delaunay
TINs use the same automatic edge rejection and duplicate-XY handling.
Complete geometry PLY bytes match across builds/passes. Reference/query counts
are 531,908/531,908, 878,921/878,920 and 941,933/941,933; TIN vertex/triangle
counts are 501,097/844,324, 873,771/1,688,821 and 893,804/1,466,179.

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve source/configuration
files match except for the triangle-seed flag in `distance.rs`.
An independent archived baseline supplies complete f64 distances, six exact
f64 statistics and scalar PLY oracles, then exits before timed contexts start.
Only one context computes at a time. One excluded first-use pair and five
alternating warm-up pairs precede ten alternating timed pairs per source/pass.
Initial order is PandaSet → Autoware → Tokyo; the second pass reverses it with
identical build/script fingerprints.

Whole API timing includes fresh BVH construction, serial Morton preparation/
sorting, parallel queries, seed preparation/replacement, result allocation/
serial scatter, destruction and pool installation. Loading, TIN/pool creation,
first/warm-up observations, serialization, statistics/parity and cleanup are
excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Candidate median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 372.51 | 388.39 | −0.78% | 4/10 |
| PandaSet final | 398.80 | 394.52 | +0.90% | 6/10 |
| Autoware initial | 672.08 | 674.46 | −2.48% | 4/10 |
| Autoware final | 701.11 | 718.39 | −1.60% | 2/10 |
| Tokyo initial | 2,115.01 | 2,040.52 | +1.01% | 5/10 |
| Tokyo final | 1,723.37 | 1,771.02 | −1.67% | 3/10 |

Shortening is `median(100 * (1 - seeded_ms / baseline_ms))`, not a ratio of
marginal medians. Every loss/spike is retained, including initial Autoware's
seventh trial near 651.8 / 814.2 ms and final PandaSet's second trial near
606.2 / 414.5 ms. Clocks, power, scheduling, other applications, allocation,
code layout and retained pages are uncontrolled. These measurements do not
isolate seed-hit frequency, copying, edge preparation or projection costs.
Direction changes and Autoware's repeated losses do not support enabling
this default. Percentages from separate experiments are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes, six
statistics values and full scalar PLY bytes: 192 outputs and six excluded
oracles. Geometry, thread counts and query counts match. Archived sources,
executables, compiler, script, inputs/manifests and original PCD are checked
before/after. An independent audit recomputes summaries and verifies pair
order, fingerprints and provenance. No GPU model identifiers are collected.

## Validation and reproduction

With the candidate enabled, 408 native tests pass with the parallel feature,
alongside native/WASM Clippy denying warnings, formatting and both release
example builds. The new regression compares 22,272 nearest-hit cases across
eight seeds, six scales, 116 queries and four seed policies: none, previous
hit, changing arbitrary triangle and fixed triangle. When a seed is present,
it also compares its corner/edge bits and full closest-point bits with the
original projection. Triangle IDs, closest-point bits, squared-distance bits,
signed-distance bits and stack capacities match. The seed matches the winning
triangle after success and is cleared after missing hits. Repeated/reversed
and degenerate faces, nonfinite geometry, signed zero, NaN payloads, infinities
and finite-coordinate overflow are included.

The independent original-chunk regression checks 216 finite signed/unsigned
C2M comparisons across four scales and 1/2/4 threads through 65,536 points,
plus eight mixed missing-hit cases. Generated fixtures are correctness tests
only. Benchmark syntax and twelve invalid CLI controls pass.

After restoring the flag to false, the native release is rebuilt; the new
and independent C2M regressions, native Clippy and formatting pass again.
Current twelve source fingerprints match the archived baseline. A separate
correctness-only run checks rebuilt PandaSet geometry, complete f64 distances,
statistics and scalar PLY against the archived oracle, recording the current
executable fingerprint. Rebuilt executable bytes need not equal the archive.

Build the release `bench_c2m_source` example with `--features parallel`, first
with `USE_PAR_MESH_TRIANGLE_SEED = false`, then true. Keep adopted flags
listed above enabled and all rejected flags disabled. Archive each executable
as `bench_c2m_source.exe`, its twelve source/configuration files under repository
paths and `rustc -Vv` as `compiler.txt` in separate roots. Source guards cannot
identify binary mode; matching builds matter. Run from the repository root:

With current sources, also keep `USE_MESH_PAR_CENTROID_BOUNDS = false` in both
modes to exclude the subsequent [centroid-bound experiment](cpu-c2m-native-centroid-bounds.md).

```powershell
node web/scripts/benchmark-native-mesh-triangle-seed.mjs --baseline-root notes/native-mesh-triangle-seed-baseline-20261013 --candidate-root notes/native-mesh-triangle-seed-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-triangle-seed.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts 1..100
(default ten). Reports/run directories are not overwritten. Other checkouts
need matching archived builds and locally prepared inputs. Native outputs
remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-triangle-seed-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-triangle-seed-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-triangle-seed-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-triangle-seed-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-triangle-seed-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-triangle-seed-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-triangle-seed.mjs)
- [Projection/seed implementation and regression](../rust/crates/ca-core/src/mesh.rs)
- [Native policy and chunk-local state](../rust/crates/ca-core/src/distance.rs)
- [Independent original-chunk regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
