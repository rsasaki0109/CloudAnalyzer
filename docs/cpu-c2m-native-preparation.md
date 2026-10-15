# Native parallel C2M preparation

Native BVH preparation now calculates triangle centroids and copies ordered
triangle corners with indexed Rayon iterators for meshes with at least
65,536 triangles. `USE_MESH_PAR_PREPARATION = true` is adopted. PandaSet and
Autoware improve in both measured passes by 3.39–5.56% in whole native C2M
calls; Tokyo is mixed at −2.04% / +6.98%. Overall, 42/60 timed pairs improve.
Full distance, statistics and scalar PLY outputs match.

This applies to native builds with the `parallel` feature, including the
Rust API used by Python when rebuilt. Measurements exclude Python/NumPy
conversion costs and establish no browser or GPU speedup. Smaller meshes and
builds without the parallel feature retain the serial preparation loops.

## Computation and memory

Each centroid uses the preceding `(a + b + c) / 3` component arithmetic.
The indexed range preserves initial triangle-index order. In this experiment, recursive BVH
splitting, centroid bounds/axis selection, unstable partitioning, node bounds
and compact u32 order extraction remain sequential. An indexed ordered-corner
copy preserves the resulting triangle order. No floating-point reductions are
parallelized, and query scheduling, tie handling and sign arithmetic are unchanged.

Both measured builds retain adopted [stack reuse](cpu-c2m-native-stack.md),
[cached traversal bounds](cpu-c2m-native-bounds.md) and
[missing-hit recovery](cpu-c2m-missing-hit.md). Parallel work uses the active
Rayon pool; the measured `cloud_to_mesh_par` call installs a fixed 12-thread
pool around both preparation and queries. Other native construction calls
use their active/default Rayon pool. Existing retained vector capacities are
preserved. There is no new persistent geometry/query/result cache or GPU
dispatch; each measured call still builds and destroys its BVH.

Exact retained bytes do not bound peak process memory, temporary Rayon work,
allocator metadata or retained pages. This experiment parallelizes two
independent loops rather than the recursive tree builder.

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
files match except for the preparation flag in `mesh.rs`. Input/TIN and idle
pools remain resident in separate processes; only one computes at a time.

An independent archived baseline supplies full raw f64 distances, exact f64
summary bits and scalar PLY oracles, then exits before timing. One excluded
first-use pair and five alternating warm-up pairs precede ten alternating
timed pairs per source/pass. Initial order is PandaSet → Autoware → Tokyo;
the second pass reverses it with identical build/script fingerprints.
Whole API timing includes fresh BVH construction, Morton ordering, queries,
joining, BVH destruction and Rayon pool installation. Loading, TIN/pool
creation, first/warm-up observations, serialization, statistics/parity and
cleanup are excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Serial-preparation baseline median, ms | Parallel-preparation median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 818.64 | 793.91 | +3.54% | 7/10 |
| PandaSet final | 756.87 | 735.91 | +5.41% | 8/10 |
| Autoware initial | 1,855.32 | 1,742.05 | +5.56% | 8/10 |
| Autoware final | 1,319.76 | 1,276.62 | +3.39% | 8/10 |
| Tokyo initial | 2,533.02 | 2,581.92 | −2.04% | 4/10 |
| Tokyo final | 2,988.12 | 3,098.49 | +6.98% | 7/10 |

Shortening is `median(100 * (1 - parallel_ms / baseline_ms))`, not a ratio
of marginal medians. Tokyo's final marginal median is higher despite a
positive paired median and 7/10 faster pairs. Tokyo therefore provides weaker
evidence than the consistent improvements on the other two sources.
All losses/spikes remain, including final Tokyo candidate trials near
4,470.6 and 5,052.6 ms. Clocks, power, scheduling, other applications,
allocation, code layout and retained pages are uncontrolled. Adoption is based
on the two consistently improved sources, overall paired observations and
unchanged outputs/capacities; it does not promise improvement on every input
or call. No isolated preparation, other-platform, Python conversion, WASM or
GPU speedup is claimed. Percentages from different experiments are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes,
six f64 summary values and full scalar PLY bytes: 192 outputs and six excluded
oracles. Geometry, threads and query counts remain identical. Archived source/
executable sets, script, inputs, manifests and original PCD are checked
before/after. A separate audit recomputes all summaries and checks pair order
and provenance. No GPU model identifiers are collected.

## Validation and reproduction

399 native tests pass with the parallel feature, together with native/WASM
Clippy denying warnings, formatting and both native release example builds.
The new threshold regression exercises 65,535/65,536/65,537 triangles,
local/map coordinates and 1/2/4 threads: 18 preparation/layout cases. It
checks analytic signed/unsigned distances through prepared, hinted and
parallel fresh APIs, NaN/infinity/finite-overflow queries, and exact expected
retained vector capacities. The preceding independent original-chunk and
cached-bound regressions also pass. Generated fixtures are for correctness
only. Benchmark syntax and twelve invalid CLI controls pass.

Build `bench_c2m_source` in release mode with `--features parallel`, first
with `USE_MESH_PAR_PREPARATION = false`, then true. Keep native stack reuse
and cached bounds enabled in both. With current sources, keep
`USE_MESH_BOUND_MERGE = false` and `USE_MESH_PAR_BUILD = false` to reproduce
this earlier preparation comparison before [bottom-up bounds](cpu-c2m-bound-merge.md)
and [parallel construction](cpu-c2m-native-builder.md). Also keep
`USE_PAR_MESH_COMPACT_RESULTS = false` before
[compact temporary results](cpu-c2m-native-results.md). Archive each executable as
`bench_c2m_source.exe`, its twelve source/configuration files under repository
paths and `rustc -Vv` as `compiler.txt` in separate roots. Source guards cannot
identify binary mode; matching builds matter. Run from the repository root
with unchanged measured inputs and a new output:

```powershell
node web/scripts/benchmark-native-mesh-preparation.mjs --baseline-root notes/native-mesh-preparation-baseline-20261013 --candidate-root notes/native-mesh-preparation-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-preparation.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts
1..100 (default ten). Reports/run directories are not overwritten. Other
checkouts need equivalent archived builds and locally prepared inputs.
Native output files remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-preparation-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-preparation-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-preparation-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-preparation-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-preparation-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-preparation-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-preparation.mjs)
- [Threshold/capacity regression](../rust/crates/ca-core/tests/mesh_parallel_preparation.rs)
- [Preparation implementation](../rust/crates/ca-core/src/mesh.rs)
