# Native parallel C2M stack reuse

Native `cloud_to_mesh_par` now reuses its traversal stack within each original
4,096-point Morton chunk. The change is adopted with
`USE_PAR_MESH_STACK_REUSE = true`. Whole native C2M calls show positive median
paired reductions in all six measured comparisons, ranging from 1.48% to
9.88%, with 41/60 faster timed pairs. Full distance/statistics/PLY outputs match.
The later [native cached-bound traversal](cpu-c2m-native-bounds.md) is also
adopted; its comparisons use this stack-reuse policy as their baseline.

The Python extension already calls this API and uses the new policy when
rebuilt. These are native core measurements, excluding Python/NumPy conversion
cost. Browser workers use a separate implementation; no browser or GPU speedup
is inferred from these results.

## Allocation and computation

The preceding parallel closure used `MeshBvh::nearest`, creating a fresh
`Vec<usize>` traversal stack for every point. A chunk's state now holds both
the preceding guessed triangle and a stack, calling `nearest_with` to reuse
its capacity. The search clears the stack before every point. Chunk boundaries,
call boundaries and missing-hit recovery still reset guessed triangles.
Each chunk drops its scratch on completion; no stack, query/result or geometry
cache survives a call. Vec growth and the allocator's retained pages are not
a fixed peak-memory guarantee.

The existing Morton sort, original 4,096-point chunks, Rayon scheduling,
near-child order, pop pruning, strict triangle insertion, signed arithmetic
and result join remain unchanged. This experiment reuses the ordinary
`Vec<usize>` stack; it does not switch native parallel traversal to the later
cached-child-bound algorithm. Every call still builds and drops its BVH.
The [missing-hit correction](cpu-c2m-missing-hit.md) remains effective.
Browser [cached traversal bounds](cpu-c2m-stack-bounds.md) stay enabled,
while the query-order and borrowed-query experiments stay disabled.

## Measured comparison

Inputs follow [measured-source preparation](c2c-real-source.md). Full PLY
coordinates and source-index subsets are unchanged: no generated points,
displacement, noise or additional thinning. Tokyo retains its preceding
original-point preparation. These workload splits are not independent surveys
or accuracy/change ground truth.

Native PLY order is preserved, rather than passing through browser LOD
indexing. Each context builds the full-input production 2.5D Delaunay TIN with
automatic edge rejection and recorded duplicate-XY handling. Full binary
mesh PLY bytes match across both native builds and both passes; these native
mesh hashes are not asserted to equal earlier browser mesh hashes.

| Source | Reference / queries | TIN vertices / triangles | Original chunks |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 501,097 / 844,324 | 130 |
| Autoware planning | 878,921 / 878,920 | 873,771 / 1,688,821 | 215 |
| Tokyo Takanawadai | 941,933 / 941,933 | 893,804 / 1,466,179 | 230 |

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run two archived native
release executables with identical Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Each uses a fixed 12-thread
Rayon pool. Twelve relevant source/configuration files match except for the
stack-reuse flag in `distance.rs`; executable hashes and compiler descriptions
are recorded. Separate processes keep their input/TIN and idle thread pools
resident, but only one computes at a time.

An independent archived baseline process supplies complete f64 distances,
exact f64 statistics bits and full scalar PLY oracles, then exits before
either timed context starts. One excluded first-use pair and five alternating
warm-up pairs precede ten alternating timed pairs per source/pass. Initial
order is PandaSet → Autoware → Tokyo; the second pass reverses it with identical
build/script fingerprints. Whole API timing includes fresh BVH construction,
Morton ordering, parallel queries, result joining and BVH destruction, including
the Rayon pool installation call. Loading, TIN construction, pool creation,
first/warm-up observations, serialization, statistics/parity and cleanup are
excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Reuse median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 963.97 | 890.20 | +9.88% | 8/10 |
| PandaSet final | 867.03 | 835.09 | +5.11% | 7/10 |
| Autoware initial | 2,141.07 | 2,063.11 | +1.48% | 5/10 |
| Autoware final | 1,818.53 | 1,708.58 | +7.13% | 8/10 |
| Tokyo initial | 2,941.83 | 2,873.00 | +3.85% | 6/10 |
| Tokyo final | 3,107.01 | 3,137.38 | +4.69% | 7/10 |

Shortening is `median(100 * (1 - reused_ms / baseline_ms))`, not a ratio of
marginal medians. In the final Tokyo pass, the candidate's marginal median is
slightly higher despite a positive paired median and 7/10 faster pairs.
Autoware's initial difference is also small. All losses and spikes are retained.
Clocks, power, scheduling, other applications, allocation, code layout and
retained pages are uncontrolled. These observations support adopting the
allocation reuse on this native workload; they do not guarantee every call
improves or establish an isolated traversal, other-platform, Python conversion,
WASM or GPU speedup.

All 96 first-use/warm-up/timed pairs match raw f64 distance bytes, six f64
summary values and complete scalar PLY bytes: 192 outputs and six excluded
oracles. Mesh geometry, thread counts and query counts stay identical. Both
archived executable/source sets, script, inputs, manifests and original PCD
are fingerprinted before/after; a separate audit recomputes all summaries and
checks pair ordering/provenance. No GPU model identifiers are collected.

## Validation and reproduction

398 native tests pass with the parallel feature, together with native/WASM
Clippy denying warnings, formatting and both native release example builds.
The new regression independently reproduces the preceding per-query-stack
policy and original Morton keys/chunks, checking 144 finite comparisons across
four scales, counts 0/1/4095/4096/4097/8193, both signs and 1/2/4 threads.
Eight additional mixed missing-hit comparisons cover NaN, infinity and finite
overflow around chunk boundaries. Opposite-facing/degenerate triangles and
large map coordinates exercise ties and exact signs. Earlier missing-hit and
geometry-ownership regressions also pass. Generated data is for correctness
only. Benchmark syntax and twelve invalid CLI controls pass.

Build `bench_c2m_source` in release mode with `--features parallel`, first with
`USE_PAR_MESH_STACK_REUSE = false`, then true. When using current sources,
keep `USE_PAR_MESH_STACK_BOUNDS = false` and `USE_MESH_PAR_PREPARATION = false`
alongside `USE_MESH_BOUND_MERGE = false` and `USE_MESH_PAR_BUILD = false`
to reproduce this earlier stack-only comparison before
[bottom-up bounds](cpu-c2m-bound-merge.md) and
[parallel construction](cpu-c2m-native-builder.md). Also keep
`USE_PAR_MESH_COMPACT_RESULTS = false` before
[compact temporary results](cpu-c2m-native-results.md). Archive each matching executable
as `bench_c2m_source.exe`, its twelve source/configuration files under their
repository paths, and `rustc -Vv` as `compiler.txt` in separate snapshot roots.
Source guards cannot identify a binary's mode; matching builds matter. Then
run from the repository root using unchanged measured inputs and a new output:

```powershell
node web/scripts/benchmark-native-mesh-stack.mjs --baseline-root notes/native-mesh-stack-baseline-20261013 --candidate-root notes/native-mesh-stack-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-stack.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts
1..100 (default ten). Existing reports and native run directories are not
overwritten. Other checkouts need equivalent archived native builds and locally
prepared measured inputs. Native output files remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-stack-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-stack-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-stack-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-stack-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-stack-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-stack-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-stack.mjs)
- [Native measured-source example](../rust/crates/ca-core/examples/bench_c2m_source.rs)
- [Exact chunk-policy regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
- [Parallel implementation](../rust/crates/ca-core/src/distance.rs)
- [Python caller](../rust/crates/ca-py/src/lib.rs)
