# Native C2M sibling-bound SIMD

`USE_PAR_MESH_BOUND_PAIR_CLAMP = true` is adopted. Pairing the left/right
child bounds in two f64 SSE2 lanes, with NaN handling folded into clamping,
shortens whole native C2M calls by 3.81–6.46% across all six measured-source
comparisons. All 60 timed pairs improve. Complete tested outputs remain
identical to the preceding production oracle.

The preceding explicit-mask variant, `USE_PAR_MESH_BOUND_PAIR`, stays false.
It improves PandaSet in both passes but changes direction on Autoware, with
44/60 faster pairs. Both variants retain private helpers and regression
controls. Existing adopted construction, traversal and result policies remain
enabled.

## Candidate and scope

The preceding [stage profile](cpu-c2m-native-stages.md) places parallel query
work at 57.07–84.71% of full calls. This experiment changes the child-box
calculation inside that query work. It does not establish what fraction of
query time the box calculation alone occupies.

Each inner node needs two lower bounds to choose the next child. The original
code computes the two boxes separately. Both SIMD variants put the left box
in one f64 lane and the right box in the other. Each axis retains the original
differences, squaring and `0 + dx*dx + dy*dy + dz*dz` reduction order. There is
no FMA, horizontal reduction or precision conversion.

For each axis, the original difference is
`maxnum(maxnum(min - point, point - max), 0)`. SSE2 `MAXPD` selects its second
operand when either input is NaN. The initial variant uses an unordered mask
to retain a numeric first operand when the second is NaN, then clamps to zero.
The adopted variant first applies `MAXPD(difference, 0)` to each difference,
mapping NaN, negatives and zero to positive zero. A third `MAXPD` selects
between the two numeric nonnegative values. This yields the same squared
axis value: intermediate zero-sign differences disappear after squaring and
the initial zero addition. The separate unordered-mask operations are omitted.
This is an equivalence for the final bound, not for every intermediate max.

SSE2 is part of the native x86_64 baseline. Intrinsics use register operands
inside a documented unsafe block, with no pointers or unchecked memory access.
Other targets retain two scalar box calls. The measured benefit is established
on the recorded Windows/x86_64 host, not on other architectures.

Selection is native parallel C2M only. The root bound stays scalar. Nodes,
triangle order, copied corners, capacities, near-first ordering, pop-time
pruning, duplicate seed tests, strict hit insertion, closest-point/sign/
missing-hit arithmetic, original 4,096-point Morton chunks and chunk-local
warm states stay unchanged. Results retain compact f64 chunks and original
serial scatter. No buffer, persistent cache or new allocation is introduced.
Ordinary serial/prepared/WASM/hinted queries, GPU exports and native C2C retain
their preceding selected path.

Both builds retain [compact results](cpu-c2m-native-results.md),
[parallel construction](cpu-c2m-native-builder.md),
[bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md) and
[cached bounds](cpu-c2m-native-bounds.md). Rejected original-vertex access,
duplicate seed skipping, Morton preparation, parallel restoration, push-time
pruning, triangle-seed reuse and parallel centroid bounds remain disabled.

## Whole-call measurements

[Measured-source preparation](c2c-real-source.md) supplies full unchanged f64
XYZ and original source-index subsets. No generated coordinates, displacement,
noise or extra thinning is introduced. Tokyo retains its preceding original-
point preparation. These subsets are not independent surveys or accuracy
ground truth; native PLY order differs from browser LOD order. Production
Delaunay TINs keep automatic edge rejection and duplicate-XY handling.

| Source | Reference/query points | TIN vertices | TIN triangles |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 501,097 | 844,324 |
| Autoware planning | 878,921 / 878,920 | 873,771 | 1,688,821 |
| Tokyo Takanawadai | 941,933 / 941,933 | 893,804 | 1,466,179 |

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived release examples
with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc, optimization
level 3, LTO and one codegen unit. Within each variant's comparison, twelve
source/configuration files match except for its selection flag in `distance.rs`.
The masked and clamped experiments have separate source/build fingerprints.
Both compare against their frozen scalar sibling-bound baseline; the clamped
table is not a comparison against the preceding masked candidate.

An independent archived baseline supplies full f64 distances, six statistics
and scalar PLY oracles, then exits before timed contexts start. Baseline and
candidate retain identical geometry and pools in separate processes; only one
computes at a time. One excluded first-use pair and five alternating warm-up
pairs precede ten alternating timed pairs per source/pass. The first pass runs
PandaSet → Autoware → Tokyo; the second reverses that order with identical
variant-specific build/controller fingerprints.

Whole `cloud_to_mesh_par` timing includes fresh BVH construction, original
serial Morton preparation/sorting, parallel queries, final allocation/scatter,
destruction and pool installation. Loading, meshing, pool creation, first-use/
warm-up observations, serialization, statistics/parity and cleanup are
excluded. No builds, tests, scans or edits run alongside timing.

| Clamped variant source/pass | Scalar median, ms | SIMD median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 281.51 | 270.36 | +4.06% | 10/10 |
| PandaSet final | 337.12 | 321.98 | +4.78% | 10/10 |
| Autoware initial | 518.11 | 501.25 | +3.81% | 10/10 |
| Autoware final | 595.89 | 566.35 | +4.33% | 10/10 |
| Tokyo initial | 1,379.48 | 1,287.43 | +6.46% | 10/10 |
| Tokyo final | 1,365.99 | 1,286.70 | +6.11% | 10/10 |

The preceding masked variant remains off:

| Masked variant source/pass | Scalar median, ms | SIMD median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 306.71 | 293.69 | +3.33% | 9/10 |
| PandaSet final | 310.43 | 301.84 | +3.36% | 8/10 |
| Autoware initial | 543.92 | 550.18 | −0.38% | 5/10 |
| Autoware final | 561.70 | 544.19 | +1.77% | 9/10 |
| Tokyo initial | 1,384.20 | 1,386.54 | +0.36% | 6/10 |
| Tokyo final | 1,404.92 | 1,376.87 | +2.25% | 7/10 |

Shortening is `median(100 * (1 - candidate_ms / baseline_ms))`, not a ratio of
marginal medians. All outliers remain, including clamped final PandaSet's sixth
trial near 802.5 / 786.9 ms and seventh near 600.4 / 377.2 ms. Clocks, power,
scheduling, other applications, allocator behavior, code layout and retained
pages are uncontrolled. Absolute times shift between passes. Results support
the clamped default on these workloads; they do not isolate instruction,
cache-miss or primitive latency, or establish universal improvement. Separate
experiments' percentages are not additive.

Each variant's 96 first-use/warm-up/timed pairs matches complete f64 distances,
six statistics values and full scalar PLY bytes: 192 outputs and six excluded
oracle calls per variant. Complete geometry bytes and thread/query counts
match, including preceding historical production oracles. Source/executable/
compiler/controller/input/manifest/original PCD fingerprints are checked
before/after. Independent audits recompute summaries and verify order/provenance.
No GPU model identifiers are collected. These native timings establish no
browser, WASM, GPU, retained-mesh or Python-conversion speedup. No rebuilt
Python extension is installed.

## Validation and reproduction

With each candidate selected, 414 native tests pass with the parallel feature,
alongside native all-target/WASM Clippy with warnings denied, formatting and
the release example builds. The final clamped source/executable fingerprints
match the measured adopted candidate.

The boundary regression checks each SIMD variant over 3,162,112 paired-box
cases, with two exact scalar-bit comparisons per case: three axes times
16^5 combinations, plus 16,384 raw-bit randomized cases. Numeric/NaN operand
positions, NaN payloads, signed zeros, subnormals, extrema, infinities,
inverted bounds and finite overflow are included. Axes reset between sweeps
so overflow on an earlier axis cannot hide later differences.

The hit regression checks 22,272 cases per variant: eight seeds, six scales,
116 queries and four seed policies. Triangle IDs, closest-point bits,
squared-distance bits, signed-distance bits and reusable stack capacities
match scalar cached-bound traversal. Repeated/reversed and degenerate faces,
nonfinite geometry, missing-hit resets and extreme query values are covered.
The independent original-chunk regression checks 216 finite signed/unsigned
C2M comparisons across four scales and 1/2/4 threads through 65,536 points,
plus eight mixed missing-hit cases. Generated fixtures are correctness tests
only. Both controllers pass syntax and twelve invalid CLI controls each.

The stage profiling example follows the selected sibling policy by reading
its compiled-in source flags once outside command clocks. Its frozen earlier
profiles retain the preceding scalar policy and remain historical measurements.
A correctness-only full PandaSet smoke checks both current profiling modes:
14 outputs plus an excluded production oracle match complete distances,
statistics, scalar PLY and geometry. Its single timed pair is not used as
additional performance evidence.

Build `bench_c2m_source` in release mode with `--features parallel`, archiving
each executable as `bench_c2m_source.exe`, the controller's twelve source/
configuration files under repository paths, and `rustc -Vv` as `compiler.txt`
in separate roots. For the adopted clamped comparison, keep
`USE_PAR_MESH_BOUND_PAIR = false` in both builds and change only
`USE_PAR_MESH_BOUND_PAIR_CLAMP` from false to true. Keep other adopted policies
enabled and rejected policies disabled. For the masked control, use its
recorded earlier sources and change only `USE_PAR_MESH_BOUND_PAIR`.
Source guards cannot identify binary contents; matching archived builds matter.
Run from the repository root:

```powershell
node web/scripts/benchmark-native-mesh-bound-pair-clamp.mjs --baseline-root notes/native-mesh-bound-pair-clamp-baseline-20261014 --candidate-root notes/native-mesh-bound-pair-clamp-measured-20261014 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-bound-pair-clamp.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts
1..100 (default ten). Reports/run directories are not overwritten. Other
checkouts need matching archived builds and locally prepared inputs. Native
output artifacts remain under ignored `notes/`.

- Adopted PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-bound-pair-clamp-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-bound-pair-clamp-final.json)
- Adopted Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-bound-pair-clamp-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-bound-pair-clamp-final.json)
- Adopted Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-bound-pair-clamp-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-bound-pair-clamp-final.json)
- Masked PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-bound-pair-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-bound-pair-final.json)
- Masked Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-bound-pair-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-bound-pair-final.json)
- Masked Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-bound-pair-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-bound-pair-final.json)
- [Clamped controller](../web/scripts/benchmark-native-mesh-bound-pair-clamp.mjs), [masked controller](../web/scripts/benchmark-native-mesh-bound-pair.mjs)
- [Native selection](../rust/crates/ca-core/src/distance.rs), [BVH kernels and regressions](../rust/crates/ca-core/src/mesh.rs)
- [Independent original-chunk regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
