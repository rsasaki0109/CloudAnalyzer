# Native C2M compact temporary results

Native parallel C2M now stores only f64 distances in temporary chunk vectors
and reads their original indices from the existing Morton order during
scatter. `USE_PAR_MESH_COMPACT_RESULTS = true` is adopted. Whole native calls
improve by 1.68–4.86% in six measured-source comparisons, with 45/60 faster
timed pairs. Complete distance, statistics and scalar PLY outputs match.
On the measured 64-bit native build, temporary result payload is halved from
16 to 8 bytes per query, without changing final output capacity or precision.

This applies to native builds with the parallel feature and the Rust API
used by Python when rebuilt. Measurements exclude Python/NumPy conversion
costs and installation of a rebuilt extension. Native C2C and
sequential/prepared/WASM/hinted APIs keep their preceding policy; no speedup
for those paths or GPU execution is established by this experiment.

## Computation and memory

The original query order already holds each point's original index. The
preceding native batch also stored that index beside each f64 distance in
`Vec<Vec<(usize, f64)>>`. The candidate instead uses `Vec<Vec<f64>>`, retaining
the same indexed collection of original 4,096-point chunks. Each chunk
creates the same fresh default state and processes its points sequentially
in the same order. Guessed hits, traversal scratch, signed-distance arithmetic
and missing-hit recovery are unchanged.

The original serial scatter now pairs each stored chunk with the matching
slice of existing Morton indices. It writes each unchanged f64 value to the
same original output position. No inverse index map, unsafe scatter,
precision conversion or floating-point reduction is introduced. The chunk
vector count, final output length/capacity, original serial Morton preparation
and key-only unstable sort remain unchanged, including equal-key order.

Both measured builds retain [parallel BVH construction](cpu-c2m-native-builder.md),
[bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed mesh preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md) and
[cached traversal bounds](cpu-c2m-native-bounds.md). Rejected
[seed-triangle](cpu-c2m-native-seed.md) and [Morton preparation](cpu-c2m-native-morton.md)
experiments remain off.

| Source | Queries | Chunk vectors | Preceding result payload, bytes | Compact payload, bytes |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 | 130 | 8,510,528 | 4,255,264 |
| Autoware planning | 878,920 | 215 | 14,062,720 | 7,031,360 |
| Tokyo Takanawadai | 941,933 | 230 | 15,070,928 | 7,535,464 |

These are temporary vector payloads on the measured 64-bit native build,
excluding vector headers, Morton order, final output, mesh/BVH, source points,
allocator metadata and retained pages. Inner f64 capacities exactly equal
their chunk lengths in the regression. The outer vector may have spare
capacity for small jobs. The payload reduction does not guarantee a matching
drop in peak process memory or retained allocator pages. No persistent
geometry/query/result cache is added.

## Measured results

[Measured-source preparation](c2c-real-source.md) supplies unchanged full
f64 XYZ and source-index subsets without generated coordinates, displacement,
noise or additional thinning. Tokyo retains its preceding original-point
preparation. These are not independent surveys or accuracy ground truth.
Native PLY order differs from browser LOD order. Full-input production
Delaunay TINs use automatic edge rejection and recorded duplicate-XY handling.
Complete geometry PLY bytes match across both native builds and passes.
Reference counts are 531,908 / 878,921 / 941,933; TIN vertex/triangle counts
are 501,097/844,324, 873,771/1,688,821 and 893,804/1,466,179 respectively.

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve source/configuration
files match except for the compact-result flag in `distance.rs`. Separate
processes retain identical input/TIN and idle pools; only one computes at a time.

An independent archived baseline supplies complete f64 distances, six exact
f64 summary values and scalar PLY oracles, then exits before timing. One
excluded first-use pair and five alternating warm-up pairs precede ten timed
alternating pairs per source/pass. Initial order is PandaSet → Autoware →
Tokyo; the second pass reverses it with identical build/script fingerprints.
Whole API timing includes fresh BVH construction, original serial Morton
preparation/sorting, parallel queries, result allocation/joining, destruction
and pool installation. Loading, TIN/pool creation, first/warm-up observations,
serialization, statistics/parity and cleanup are excluded. No builds, tests,
scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Compact median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 352.97 | 343.77 | +3.66% | 8/10 |
| PandaSet final | 429.54 | 416.19 | +3.18% | 7/10 |
| Autoware initial | 634.80 | 620.66 | +1.68% | 8/10 |
| Autoware final | 704.08 | 677.23 | +4.86% | 8/10 |
| Tokyo initial | 1,868.29 | 1,814.79 | +2.93% | 6/10 |
| Tokyo final | 1,829.78 | 1,774.11 | +3.05% | 8/10 |

Shortening is `median(100 * (1 - compact_ms / baseline_ms))`, not a ratio of
marginal medians. Every loss/spike is retained, including final Tokyo's last
trial near 1,881.8 / 2,356.8 ms and final PandaSet's trial near 704.9 / 446.3 ms.
Clocks, power, scheduling, other applications, allocation, code layout and
retained pages are uncontrolled. Adoption is based on positive paired medians
in both passes of all three sources, 45/60 faster observations, exact tested
outputs and the direct temporary payload reduction. Some timing differences
are small and individual calls can regress. No isolated allocation/scatter,
other-platform or retained-mesh query speedup is claimed. Percentages from
different experiments are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes,
six summary values and full scalar PLY bytes: 192 outputs and six excluded
oracles. Geometry, threads and query counts remain identical. Archived source/
executable sets, script, inputs, manifests and original PCD are checked
before/after. An independent audit recomputes summaries and verifies pair
order, fingerprints and provenance. No GPU model identifiers are collected.

## Validation and reproduction

406 native tests pass with the parallel feature, together with native/WASM
Clippy denying warnings, formatting and both native release example builds.
The new packing regression compares 108 combinations of 1/2/4 threads, nine
counts through 65,537 points and four coordinate patterns. Its stateful query
uses a rolling coordinate-bit history and chunk-local counter, and returns
finite values, signed zero, NaN payloads and infinities. It verifies full
original-order output bits, final output capacity, chunk counts and exact
inner payload capacities against the preceding pair-storage path.

The independent original-chunk regression also checks 216 signed/unsigned
finite C2M comparisons across four scales and 1/2/4 threads through the
65,536-point boundary, plus eight mixed missing-hit cases. Existing mesh,
ordering and preparation regressions pass. Generated fixtures are for
correctness only. Benchmark syntax and twelve invalid CLI controls pass.

Build `bench_c2m_source` in release mode with `--features parallel`, first
with `USE_PAR_MESH_COMPACT_RESULTS = false`, then true. Keep native stack
reuse, cached traversal bounds, indexed preparation, bottom-up bounds and
parallel construction enabled; keep seed deduplication and parallel Morton
preparation disabled in both. Archive each executable as `bench_c2m_source.exe`,
its twelve source/configuration files under repository paths and `rustc -Vv`
as `compiler.txt` in separate roots. Source guards cannot identify binary mode;
matching builds matter. Run from the repository root with unchanged inputs
and a new output:

When rebuilding from current sources, also keep `USE_PAR_MESH_PAR_RESULTS = false`
in both modes. The subsequent [parallel restoration experiment](cpu-c2m-native-parallel-results.md)
is disabled after six negative comparisons; this compact-result adoption
continues to use serial scatter.

Also keep `USE_PAR_MESH_EAGER_PRUNE = false` in both modes when rebuilding
from current sources; the subsequent [push-time pruning experiment](cpu-c2m-native-eager-prune.md)
remains disabled after mixed measured results.

Keep `USE_PAR_MESH_TRIANGLE_SEED = false` in both modes as well; the subsequent
[triangle-seed experiment](cpu-c2m-native-triangle-seed.md) remains disabled.

Also keep `USE_MESH_PAR_CENTROID_BOUNDS = false` in both modes to exclude the
subsequent [centroid-bound experiment](cpu-c2m-native-centroid-bounds.md).

Keep `USE_PAR_MESH_INDEXED_CORNERS = false` in both modes as well, excluding
the subsequent [original-vertex leaf experiment](cpu-c2m-native-indexed-corners.md).

```powershell
node web/scripts/benchmark-native-mesh-compact-results.mjs --baseline-root notes/native-mesh-compact-results-baseline-20261013 --candidate-root notes/native-mesh-compact-results-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-compact-results.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts 1..100
(default ten). Reports/run directories are not overwritten. Other checkouts
need matching archived builds and locally prepared inputs. Native outputs
remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-compact-results-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-compact-results-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-compact-results-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-compact-results-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-compact-results-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-compact-results-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-compact-results.mjs)
- [Packing, flag and stateful regression](../rust/crates/ca-core/src/distance.rs)
- [Independent original-chunk regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
