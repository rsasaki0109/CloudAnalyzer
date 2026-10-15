# Native C2M Morton preparation experiment

Parallel query-bound scans and Morton-key generation preserve tested order
and output bits, but do not produce consistent whole-call improvement.
`USE_PAR_MESH_MORTON_PREPARATION = false` remains the default. PandaSet changes
by −2.71% / +0.71%, Autoware by +5.78% / +0.49%, and Tokyo by −2.67% /
−0.04% in two measured passes. Only 30/60 timed pairs improve.

The candidate remains internal. Adopted [parallel BVH construction](cpu-c2m-native-builder.md),
[bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed mesh preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md) and
[cached traversal bounds](cpu-c2m-native-bounds.md) stay enabled. The preceding
[seed-triangle experiment](cpu-c2m-native-seed.md) stays off in both builds.
These are native parallel C2M observations, with no Python conversion, C2C,
browser, WASM or GPU speedup claim.

## Candidate and equivalence

For native C2M queries containing at least 65,536 points, the candidate scans
16,384-point blocks on the active Rayon pool. Each block computes its coordinate
bounds and whether every component is finite and nonzero. Finite nonzero
extrema select identical f64 values regardless of grouping, so their block
summaries can be merged without changing bound bits. If any coordinate is
zero, signed zero, NaN or infinite, the candidate recomputes the full bounds
in the original serial source order, preserving target-dependent tie/NaN
behavior. Both scans and the fallback cost are included in timing.

The original extent/scale calculation, cell arithmetic, saturating f64-to-u64
casts and 21-bit encoding remain unchanged. Indexed Rayon collection produces
key/index pairs in original input order. The original serial key-only
`sort_unstable_by_key` and serial index extraction remain unchanged. Equal-key
order is preserved by retaining the same initial pairs and sorting policy;
no stable sort, parallel sort or alternate tie-breaker is introduced.

Original 4,096-point query chunks, warm starts, node/triangle visitation,
signed-distance arithmetic, missing-hit recovery and scatter order stay
unchanged. Only `cloud_to_mesh_par` selects the candidate; native C2C,
sequential/prepared/WASM/hinted APIs keep serial Morton preparation. Smaller
query sets use the preceding serial path.

Block summaries are transient and released before key allocation. The three
measured query sets require 33/54/58 summaries; on the measured native build,
their tuple payloads use 1,848/3,024/3,248 bytes, plus vector metadata. Existing
key/index capacities remain identical. No persistent geometry/query/result
cache is added. Retained capacities do not bound peak process memory, Rayon
task overhead, allocator metadata or retained pages. These observations do
not isolate the relative costs of scans, admission, collection or scheduling.

## Measured results

[Measured-source preparation](c2c-real-source.md) supplies unchanged full f64
XYZ and original source-index subsets without generated coordinates,
displacement, noise or additional thinning. Tokyo retains its preceding
original-point preparation. The datasets are not independent surveys or
accuracy ground truth. All three compared inputs contain no zero or nonfinite
coordinates and take the candidate's parallel-bound path. Query counts are
531,908 / 878,920 / 941,933; original query chunks are 130 / 215 / 230.
Native PLY order differs from browser LOD order. Full-input production
Delaunay TINs use automatic edge rejection and recorded duplicate-XY handling.
Complete geometry PLY bytes match across both native builds and passes.

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve source/configuration
files match except for the Morton-preparation flag in `distance.rs`. Separate
processes retain identical input/TIN and idle pools; only one computes at a time.

An independent archived baseline supplies complete f64 distances, six exact
f64 summary values and scalar PLY oracles, then exits before timing. One
excluded first-use pair and five alternating warm-up pairs precede ten timed
alternating pairs per source/pass. Initial order is PandaSet → Autoware →
Tokyo; the second pass reverses it with identical build/script fingerprints.
Whole API timing includes fresh BVH construction, Morton preparation/sorting,
parallel queries, joining, BVH destruction and pool installation. Loading,
TIN/pool creation, first/warm-up observations, serialization, statistics/parity
and cleanup are excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Candidate median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 314.37 | 323.64 | −2.71% | 3/10 |
| PandaSet final | 312.16 | 311.61 | +0.71% | 6/10 |
| Autoware initial | 692.43 | 633.36 | +5.78% | 8/10 |
| Autoware final | 589.78 | 575.28 | +0.49% | 6/10 |
| Tokyo initial | 1,582.83 | 1,627.53 | −2.67% | 2/10 |
| Tokyo final | 1,980.23 | 1,856.42 | −0.04% | 5/10 |

Shortening is `median(100 * (1 - prepared_ms / baseline_ms))`, not a ratio of
marginal medians. Tokyo's final marginal candidate median is lower despite a
slightly negative paired median. Every observation is retained, including
initial PandaSet trials near 306.7 / 366.7 ms and final Tokyo trials near
2,396.5 / 2,550.0 ms. Clocks, power, scheduling, other applications, allocation,
code layout and retained pages are uncontrolled. Mixed directions, only half
the timed pairs improving, and the much smaller second Autoware gain do not
justify changing the default. No isolated ordering, retained-mesh query or
other-platform speedup is established. Percentages from different experiments
are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes,
six summary values and full scalar PLY bytes: 192 outputs and six excluded
oracles. Geometry, threads and query counts remain identical. Archived source/
executable sets, script, inputs, manifests and original PCD are checked
before/after. An independent audit recomputes summaries and verifies pair
order, fingerprints and provenance. No GPU model identifiers are collected.

## Validation and reproduction

With the candidate enabled, 405 native tests pass with the parallel feature,
together with native/WASM Clippy denying warnings, formatting and both native
release example builds. The new independent bit-by-bit Morton reference
checks all indices, bound bits and index capacities in 198 combinations of
1/2/4 threads, eleven counts around 4,096/16,384/65,536 boundaries, and six
patterns: ordinary map coordinates, signed zero, NaN payloads/infinities,
coincident points, finite extent overflow and subnormal coordinates.

The original-chunk C2M regression now checks 216 finite signed/unsigned
comparisons through 65,535/65,536/65,537 points across four scales and 1/2/4
threads, plus eight mixed missing-hit comparisons. It freezes the original
key-only sort, original chunks and original nearest-hit policy independently.
Generated fixtures are for correctness only. Benchmark syntax and twelve
invalid CLI controls pass. The flag is restored to false after measurement;
native release, original-chunk regression and native Clippy/formatting are
checked again for the restored default.
The restored twelve source/configuration files match the archived baseline.
A separate correctness check of the current release executable also matches
complete PandaSet f64 distances, summary bits, scalar PLY and geometry against
the baseline oracle. This supplies no additional timing comparison.

Build `bench_c2m_source` in release mode with `--features parallel`, first
with `USE_PAR_MESH_MORTON_PREPARATION = false`, then true. Keep native stack
reuse, cached traversal bounds, indexed mesh preparation, bottom-up bounds
and parallel construction enabled, with seed deduplication disabled, in both.
Keep `USE_PAR_MESH_COMPACT_RESULTS = false` to reproduce this experiment
before [compact temporary results](cpu-c2m-native-results.md).
Archive each executable as `bench_c2m_source.exe`, its twelve source/configuration
files under repository paths and `rustc -Vv` as `compiler.txt` in separate roots.
Source guards cannot identify binary mode; matching builds matter. Run from
the repository root with unchanged measured inputs and a new output:

```powershell
node web/scripts/benchmark-native-mesh-morton-preparation.mjs --baseline-root notes/native-mesh-morton-preparation-baseline-20261013 --candidate-root notes/native-mesh-morton-preparation-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-morton-preparation.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts 1..100
(default ten). Reports/run directories are not overwritten. Other checkouts
need matching archived builds and locally prepared inputs. Native outputs
remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-morton-preparation-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-morton-preparation-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-morton-preparation-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-morton-preparation-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-morton-preparation-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-morton-preparation-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-morton-preparation.mjs)
- [Preparation and independent order regression](../rust/crates/ca-core/src/distance.rs)
- [Original-chunk signed-distance regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
