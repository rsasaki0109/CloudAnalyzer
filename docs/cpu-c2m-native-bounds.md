# Native parallel C2M cached traversal bounds

`cloud_to_mesh_par` now stores child box distances already computed for
traversal ordering, avoiding duplicate computation when popped.
`USE_PAR_MESH_STACK_BOUNDS = true` is adopted alongside
[per-chunk stack reuse](cpu-c2m-native-stack.md). Six comparisons against
that adopted native baseline show positive median paired reductions of
0.58–13.43%, with 42/60 faster timed pairs and identical tested outputs.
Python uses this Rust API when rebuilt; Python/NumPy conversions are excluded.
Browser [cached bounds](cpu-c2m-stack-bounds.md) are separately adopted.
No additional browser or GPU speedup is inferred.
The subsequent [native preparation parallelism](cpu-c2m-native-preparation.md)
is also adopted; its comparison uses these cached bounds as its baseline.

## Computation

The candidate stack holds `(usize, f64)` instead of node indices alone.
Each child carries the same box distance computed to order traversal.
Pruning still occurs on pop against the current best hit; the root's distance
is computed once. Original Morton order, 4,096-point chunks, Rayon scheduling,
near-child order, strict triangle insertion, closest-point/sign arithmetic and
result joining remain unchanged. Guesses/scratch reset at chunk/call boundaries;
each point clears the stack. [Missing-hit recovery](cpu-c2m-missing-hit.md)
continues to return positive infinity and clear invalid guesses safely.

Native entries grow from eight to 16 bytes. Scratch is reused within a chunk
and drops on completion. No new persistent geometry/query/result cache is
retained; each call still builds and drops its BVH. Vec capacities, allocator
metadata and retained pages are not a fixed peak-memory guarantee.
Both measured builds have `USE_PAR_MESH_STACK_REUSE = true`.

## Measurements

[Measured-source preparation](c2c-real-source.md) supplies unchanged full
f64 XYZ and source-index subsets, without generated coordinates, added noise,
displacement or additional thinning. Tokyo retains its preceding original-point
preparation. These workloads are not independent surveys or accuracy ground truth.
Native PLY input order differs from browser LOD order. Full-input production
2.5D Delaunay TINs use automatic edge rejection and recorded duplicate-XY
handling. Full mesh PLY bytes match across both native builds and passes;
equality with earlier browser mesh hashes is not assumed.

| Source | Reference / queries | TIN vertices / triangles | Original chunks |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 501,097 / 844,324 | 130 |
| Autoware planning | 878,921 / 878,920 | 873,771 / 1,688,821 | 215 |
| Tokyo Takanawadai | 941,933 / 941,933 | 893,804 / 1,466,179 | 230 |

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run two archived release
executables with fixed 12-thread Rayon pools, Rust 1.98.1,
x86_64-pc-windows-msvc, optimization level 3, LTO and one codegen unit.
Twelve source/configuration files match except for the cached-bound flag.
Separate processes retain their inputs/TIN and idle pools; only one computes
at a time. An independent archived baseline supplies complete f64 distances,
exact f64 summary bits and scalar PLY oracles, then exits before timing.

One excluded first-use pair and five alternating warm-up pairs precede ten
alternating timed pairs per source/pass. Initial order is PandaSet → Autoware
→ Tokyo; the second pass reverses it with identical build/script fingerprints.
Whole API timing includes fresh BVH construction, Morton ordering, parallel
queries, joining, BVH destruction and Rayon pool installation. Loading, TIN
construction, pool creation, first/warm-up observations, serialization,
statistics/parity and cleanup are excluded. No builds, tests, scans or edits
run alongside timing.

| Source/pass | Reused-stack baseline median, ms | Cached-bound median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 837.45 | 828.95 | +1.47% | 6/10 |
| PandaSet final | 912.03 | 905.52 | +0.58% | 6/10 |
| Autoware initial | 1,628.27 | 1,595.58 | +1.42% | 6/10 |
| Autoware final | 1,581.49 | 1,475.33 | +13.43% | 8/10 |
| Tokyo initial | 2,377.51 | 2,177.12 | +7.23% | 10/10 |
| Tokyo final | 2,666.45 | 2,412.55 | +3.33% | 6/10 |

Shortening is `median(100 * (1 - bounded_ms / baseline_ms))`, not a ratio of
marginal medians. PandaSet and initial Autoware differences are small;
Autoware varies substantially between passes. All losses/spikes remain,
including final Autoware baseline trials near 2,592.8 and 3,174.6 ms.
Clocks, power, scheduling, other applications, allocation, code layout and
retained pages are uncontrolled. Both passes improve on every source,
supporting adoption on this native workload without guaranteeing every call
improves. No isolated traversal, other-platform, Python conversion, WASM or
GPU speedup is claimed. Percentages from different experiments are not additive.

All 96 first-use/warm-up/timed pairs match raw f64 distance bytes, six f64
summary values and full scalar PLY bytes: 192 outputs and six excluded oracles.
Geometry, threads and query counts stay identical. Archived executable/source
sets, script, inputs, manifests and original PCD are checked before/after.
A separate audit recomputes summaries and checks pair order/provenance.
No GPU model identifiers are collected.

## Validation and reproduction

398 native tests pass with the parallel feature, together with native/WASM
Clippy denying warnings, formatting and both native release example builds.
The independent original-chunk regression checks 144 finite comparisons across
four scales, counts 0/1/4095/4096/4097/8193, both signs and 1/2/4 threads.
Eight mixed missing-hit comparisons cover NaN, infinity and finite overflow
around chunk boundaries. Existing cached-bound tests compare exact BVH hits
and signed ties; geometry ownership and missing-hit recovery also pass.
Generated fixtures are for correctness only. Benchmark syntax and twelve
invalid CLI controls pass.

Build `bench_c2m_source` in release mode with `--features parallel`, first
with `USE_PAR_MESH_STACK_BOUNDS = false`, then true. With current sources,
keep `USE_MESH_PAR_PREPARATION = false` and `USE_MESH_BOUND_MERGE = false`
and `USE_MESH_PAR_BUILD = false` to reproduce this earlier traversal
comparison before [bottom-up bounds](cpu-c2m-bound-merge.md) and
[parallel construction](cpu-c2m-native-builder.md). Also keep
`USE_PAR_MESH_COMPACT_RESULTS = false` before
[compact temporary results](cpu-c2m-native-results.md). Keep stack reuse enabled
in both builds. Archive each executable as `bench_c2m_source.exe`, its twelve
source/configuration files under repository paths, and `rustc -Vv` as
`compiler.txt` in separate roots. Source guards cannot identify binary mode;
matching builds matter. Run from the repository root with unchanged inputs
and a new output:

```powershell
node web/scripts/benchmark-native-mesh-bounds.mjs --baseline-root notes/native-mesh-bounds-baseline-20261013 --candidate-root notes/native-mesh-bounds-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-bounds.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts
1..100 (default ten). Reports/run directories are not overwritten. Other
checkouts need equivalent archived builds and locally prepared inputs.
Native output files remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-bounds-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-bounds-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-bounds-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-bounds-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-bounds-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-bounds-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-bounds.mjs)
- [Native measured-source example](../rust/crates/ca-core/examples/bench_c2m_source.rs)
- [Exact chunk-policy regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
- [Parallel implementation](../rust/crates/ca-core/src/distance.rs)
- [Traversal implementation](../rust/crates/ca-core/src/mesh.rs)
