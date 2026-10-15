# Native C2M push-time pruning experiment

`USE_PAR_MESH_EAGER_PRUNE = false` remains the default. Omitting already-pruned
children from the cached traversal stack preserves tested outputs, but
PandaSet and Autoware regress in both measured passes. Tokyo improves by
1.41–2.20%; only 23/60 timed pairs improve overall. Adopted compact results,
cached bounds, stack reuse and parallel mesh construction stay enabled.

This experiment measures native parallel `cloud_to_mesh_par`. It establishes
no Python conversion, native C2C, browser, WASM or GPU speedup. No rebuilt
Python extension is installed as part of these measurements.

## Candidate and correctness

The preceding cached traversal computes both child box distances, queues
both children in the original near-first order, and rejects a child on pop
when its lower bound is greater than or equal to the current best distance.
The candidate additionally performs this same comparison before queuing.
Strict triangle insertion only decreases the best squared distance, so a
child rejected now would also be rejected when popped later. NaN lower bounds
keep the preceding comparison's false rejection and are queued. Pop-time
pruning remains for children made irrelevant by later hits.

Remaining children retain their original order and cached f64 bounds. Leaf
triangle order, the original duplicate seed test, strict insertion, closest
point calculations, signed-distance arithmetic and missing-hit recovery are
unchanged. Original serial Morton preparation/key-only unstable sorting,
equal-key order, 4,096-point chunks, fresh per-chunk state, compact f64 result
vectors and serial original-position scatter remain unchanged.

Only native parallel C2M selects the candidate. Ordinary sequential,
prepared, hinted and WASM queries retain their preceding traversal policy.
No new buffer, unsafe access, precision conversion, arithmetic regrouping or
persistent cache is added. A chunk's scratch stack can reserve less space;
retained mesh/BVH payload and final result capacity remain unchanged. This
does not establish a peak process memory reduction.

Both builds keep [compact results](cpu-c2m-native-results.md),
[parallel BVH construction](cpu-c2m-native-builder.md),
[bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md) and
[cached bounds](cpu-c2m-native-bounds.md) enabled. Rejected
[seed deduplication](cpu-c2m-native-seed.md),
[Morton preparation](cpu-c2m-native-morton.md) and
[parallel restoration](cpu-c2m-native-parallel-results.md) stay off.

## Measured results

[Measured-source preparation](c2c-real-source.md) supplies full unchanged f64
XYZ and original source-index subsets. No generated coordinates, displacement,
noise or additional thinning is introduced. Tokyo retains its preceding
original-point preparation. These are not independent surveys or accuracy
ground truth. Native decoded PLY order differs from browser LOD order.
Production Delaunay TINs use the same automatic edge rejection and duplicate-XY
handling. Complete geometry PLY bytes match across both builds and passes.

| Source | Reference / queries | TIN vertices / triangles |
| --- | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 501,097 / 844,324 |
| Autoware planning | 878,921 / 878,920 | 873,771 / 1,688,821 |
| Tokyo Takanawadai | 941,933 / 941,933 | 893,804 / 1,466,179 |

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve source/configuration
files match except for the eager-pruning flag in `distance.rs`. Separate
processes retain identical input/TIN and idle pools; only one computes at a time.

An independent archived baseline supplies complete f64 distances, six exact
f64 statistics and scalar PLY oracles, then exits before timing. One excluded
first-use pair and five alternating warm-up pairs precede ten alternating
timed pairs per source/pass. Initial order is PandaSet → Autoware → Tokyo;
the second pass reverses it with identical build/script fingerprints.
Whole API timing includes fresh BVH construction, serial Morton preparation/
sorting, parallel queries, all push/pop comparisons, compact result allocation/
serial scatter, destruction and pool installation. Loading, TIN/pool creation,
first/warm-up observations, serialization, statistics/parity and cleanup are
excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Candidate median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 278.26 | 287.69 | −4.26% | 2/10 |
| PandaSet final | 277.31 | 285.30 | −2.54% | 2/10 |
| Autoware initial | 509.36 | 529.66 | −3.91% | 3/10 |
| Autoware final | 480.34 | 497.94 | −2.33% | 2/10 |
| Tokyo initial | 1,346.99 | 1,325.81 | +2.20% | 7/10 |
| Tokyo final | 1,348.00 | 1,330.38 | +1.41% | 7/10 |

Shortening is `median(100 * (1 - pruned_ms / baseline_ms))`, not a ratio of
marginal medians. Every loss/spike is retained, including initial PandaSet's
ninth trial near 277.6 / 329.0 ms. Clocks, power, scheduling, other applications,
allocation, code layout and retained pages are uncontrolled. Whole-call
measurements do not isolate stack traffic, comparisons, allocator or search
costs. The two consistent regressions outweigh Tokyo's small gains for the
default policy. No automatic workload-selection rule is established.
Percentages from separate experiments are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes, six
statistics values and full scalar PLY bytes: 192 outputs and six excluded
oracles. Geometry, threads and query counts match. Archived sources,
executables, compiler, script, inputs/manifests and original PCD are checked
before/after. An independent audit recomputes summaries and verifies pair
order, fingerprints and provenance. No GPU model identifiers are collected.

## Validation and reproduction

With the candidate enabled, 407 native tests pass with the parallel feature,
alongside native/WASM Clippy denying warnings, formatting and both native
release example builds. The new regression compares 22,272 cached/pruned
nearest-hit results: eight seeds, six scales, 116 queries and four seed policies
(none, previous hit, changing arbitrary triangle and fixed triangle). It
includes repeated/reversed and degenerate faces, nonfinite geometry, signed
zero, NaN payloads, infinities and finite-coordinate overflow. Triangle IDs,
closest-point bits, squared-distance bits and signed-distance bits match.
Candidate scratch capacity never exceeds the preceding stack capacity.

The independent original-chunk regression checks 216 finite signed/unsigned
C2M comparisons across four scales and 1/2/4 threads through 65,536 points,
plus eight mixed missing-hit cases. Generated fixtures are correctness tests
only. Benchmark syntax and twelve invalid CLI controls pass.

After restoring the flag to false, the native release is rebuilt; the new
regression, independent C2M regression, native Clippy and formatting pass again.
Current twelve source fingerprints match the archived baseline. A separate
correctness-only run checks rebuilt PandaSet geometry, complete f64 distances,
statistics and scalar PLY against the archived oracle, recording the current
executable fingerprint. Rebuilt executable bytes need not equal the archive.

Build the release `bench_c2m_source` example with `--features parallel`, first
with `USE_PAR_MESH_EAGER_PRUNE = false`, then true. Keep all adopted flags
listed above enabled and all rejected flags disabled. Archive each executable
as `bench_c2m_source.exe`, its twelve source/configuration files under repository
paths and `rustc -Vv` as `compiler.txt` in separate roots. Source guards cannot
identify binary mode; matching builds matter. Run from the repository root:

With current sources, also keep `USE_PAR_MESH_TRIANGLE_SEED = false` in both
modes to exclude the subsequent [triangle-seed experiment](cpu-c2m-native-triangle-seed.md).

```powershell
node web/scripts/benchmark-native-mesh-eager-prune.mjs --baseline-root notes/native-mesh-eager-prune-baseline-20261013 --candidate-root notes/native-mesh-eager-prune-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-eager-prune.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts 1..100
(default ten). Reports/run directories are not overwritten. Other checkouts
need matching archived builds and locally prepared inputs. Native outputs
remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-eager-prune-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-eager-prune-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-eager-prune-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-eager-prune-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-eager-prune-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-eager-prune-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-eager-prune.mjs)
- [Traversal and new regression](../rust/crates/ca-core/src/mesh.rs)
- [Native policy flags](../rust/crates/ca-core/src/distance.rs)
- [Independent original-chunk regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
