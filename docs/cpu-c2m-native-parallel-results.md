# Native C2M parallel result restoration experiment

Parallel restoration of compact query results remains disabled:
`USE_PAR_MESH_PAR_RESULTS = false`. All six measured-source comparisons have
negative median paired shortening, from −0.10% to −5.06%, with only 18/60
faster timed pairs. Complete distance, statistics and scalar PLY outputs match.
The adopted [compact temporary results](cpu-c2m-native-results.md) remain
enabled, using the existing serial original-position scatter.

This experiment concerns native `cloud_to_mesh_par` with the parallel feature.
It does not establish Python conversion, native C2C, browser, WASM or GPU
performance. A rebuilt Python extension would use the restored native policy;
installation of a rebuilt extension is outside this experiment.

## Computation and storage

Both modes keep the same serial Morton bounds/key preparation, key-only
unstable sort, original equal-key order, indexed collection of 4,096-point
query chunks and per-chunk f64 result vectors. Each chunk creates the same
fresh default state and processes its queries sequentially in the same order.
Guessed hits, traversal scratch, signed-distance arithmetic and missing-hit
recovery are unchanged.

The candidate selects parallel restoration for 65,536..u32::MAX queries.
After querying, it builds a temporary u32 inverse Morton map serially:
each original point index maps to its position in the chunked results.
It releases the existing usize Morton-order vector before allocating final
output. Rayon then gathers the corresponding f64 values into disjoint
original-order output slices, with minimum task length 16,384. No unsafe
scatter, floating-point reduction, precision conversion or persistent cache
is introduced. Smaller inputs and inputs above the u32 limit use serial
scatter. Native C2C and sequential/prepared/WASM/hinted paths keep their
preceding policy.

Both archived builds retain compact results, [parallel BVH construction](cpu-c2m-native-builder.md),
[bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed mesh preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md) and
[cached traversal bounds](cpu-c2m-native-bounds.md). Rejected
[seed-triangle](cpu-c2m-native-seed.md) and
[Morton preparation](cpu-c2m-native-morton.md) experiments stay off.

| Source | Queries | Result chunks | Temporary inverse map, bytes |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 | 130 | 2,127,632 |
| Autoware planning | 878,920 | 215 | 3,515,680 |
| Tokyo Takanawadai | 941,933 | 230 | 3,767,732 |

The inverse map adds four bytes per query. Existing f64 result payload stays
at eight bytes per query. On the measured 64-bit native build, releasing the
eight-byte-per-query order before final output allocation changes buffer
lifetimes; it does not establish a reduction in peak process memory or
retained allocator pages. Vector headers, allocator metadata, source points,
mesh/BVH and other scratch are excluded from this table. The tested inverse
capacity equals its length, and final output length/capacity remain unchanged.

## Measured results

[Measured-source preparation](c2c-real-source.md) supplies unchanged full f64
XYZ and original source-index subsets without generated coordinates,
displacement, noise or additional thinning. Tokyo retains its preceding
original-point preparation. These are not independent surveys or accuracy
ground truth. Native decoded PLY order differs from browser LOD order.
Production full-input Delaunay TINs use automatic edge rejection and recorded
duplicate-XY handling. Complete geometry PLY bytes match across builds/passes.
Reference counts are 531,908 / 878,921 / 941,933; TIN vertex/triangle counts
are 501,097/844,324, 873,771/1,688,821 and 893,804/1,466,179 respectively.

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve source/configuration
files match except for the parallel-result flag in `distance.rs`. Separate
processes retain identical input/TIN and idle pools; only one computes at a time.

An independent archived baseline produces complete f64 distances, six exact
f64 summary values and scalar PLY oracles, then exits before timing. One
excluded first-use pair and five alternating warm-up pairs precede ten timed
alternating pairs per source/pass. Initial order is PandaSet → Autoware →
Tokyo; the second pass reverses it with identical build/script fingerprints.
Whole API timing includes fresh BVH construction, original serial Morton
preparation/sorting, parallel queries, result allocation/restoration, inverse
map construction, destruction and pool installation. Loading, TIN/pool
creation, first/warm-up observations, serialization, statistics/parity and
cleanup are excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Parallel restoration median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 326.23 | 333.09 | −1.32% | 3/10 |
| PandaSet final | 341.42 | 351.91 | −2.18% | 3/10 |
| Autoware initial | 585.07 | 587.72 | −0.10% | 5/10 |
| Autoware final | 592.74 | 609.88 | −2.10% | 2/10 |
| Tokyo initial | 1,537.00 | 1,599.00 | −5.06% | 3/10 |
| Tokyo final | 1,685.80 | 1,694.82 | −3.29% | 2/10 |

Shortening is `median(100 * (1 - gather_ms / baseline_ms))`, not a ratio of
marginal medians. Every loss/spike is retained, including initial Tokyo's
second trial near 1,526.8 / 1,977.1 ms and final PandaSet's fifth trial near
349.1 / 452.7 ms. Clocks, power, scheduling, other applications, allocation,
code layout and retained pages are uncontrolled. These whole-call measurements
do not isolate inverse-map construction, gather, scatter or allocator costs.
They provide no evidence to enable this candidate by default. Results from
different experiments are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes, six
summary values and full scalar PLY bytes: 192 outputs and six excluded oracles.
Geometry, threads and query counts remain identical. Archived sources,
executables, compiler, script, inputs, manifests and original PCD are checked
before/after. An independent audit recomputes summaries and verifies pair
order, fingerprints and provenance. No GPU model identifiers are collected.

## Validation and reproduction

With the candidate enabled, 406 native tests pass with the parallel feature,
together with native/WASM Clippy denying warnings, formatting and both native
release example builds. The expanded compact/restoration regression checks
144 combinations of 1/2/4 threads, twelve counts through 65,537 points and
four coordinate patterns, including restoration task and selection boundaries.
Its rolling coordinate-bit history and chunk-local counter return finite
values, signed zero, NaN payloads and infinities. It checks the preceding
pair-storage path, compact serial scatter, direct parallel gather and
threshold-selected gather for identical original-order f64 bits and final
capacity. Inverse permutation entries/capacity, result chunk counts and exact
inner f64 capacities also match their expected values.

The independent original-chunk regression checks 216 signed/unsigned finite
C2M comparisons across four scales and 1/2/4 threads through 65,536 points,
plus eight mixed missing-hit cases. Generated fixtures are for correctness
only. Benchmark syntax and twelve invalid CLI controls pass.

After restoring the flag to false, the native release is rebuilt and the
compact/restoration and independent C2M regressions, native Clippy and
formatting are checked again. Current twelve source fingerprints match the
archived baseline. A separate correctness-only run checks the rebuilt
release's complete PandaSet geometry, f64 distances, statistics and scalar
PLY against the archived oracle, recording its current executable fingerprint.
Rebuilding does not require the executable bytes to equal the archived binary.

Build `bench_c2m_source` in release mode with `--features parallel`, first
with `USE_PAR_MESH_PAR_RESULTS = false`, then true. Keep compact results,
native stack reuse, cached traversal bounds, indexed preparation, bottom-up
bounds and parallel construction enabled; keep seed deduplication and
parallel Morton preparation disabled in both. Archive each executable as
`bench_c2m_source.exe`, its twelve source/configuration files under repository
paths and `rustc -Vv` as `compiler.txt` in separate roots. Source guards cannot
identify binary mode; matching builds matter. Run from the repository root
with unchanged inputs and a new output:

With current sources, keep `USE_PAR_MESH_EAGER_PRUNE = false` in both modes
to exclude the subsequent [push-time pruning experiment](cpu-c2m-native-eager-prune.md).

Also keep `USE_PAR_MESH_TRIANGLE_SEED = false` in both modes to exclude the
subsequent [triangle-seed experiment](cpu-c2m-native-triangle-seed.md).

```powershell
node web/scripts/benchmark-native-mesh-parallel-results.mjs --baseline-root notes/native-mesh-parallel-results-baseline-20261013 --candidate-root notes/native-mesh-parallel-results-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-parallel-results.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts 1..100
(default ten). Reports/run directories are not overwritten. Other checkouts
need matching archived builds and locally prepared inputs. Native outputs
remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-parallel-results-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-parallel-results-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-parallel-results-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-parallel-results-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-parallel-results-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-parallel-results-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-parallel-results.mjs)
- [Restoration flags and stateful regression](../rust/crates/ca-core/src/distance.rs)
- [Independent original-chunk regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
