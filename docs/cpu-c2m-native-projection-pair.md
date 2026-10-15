# Native C2M paired projection dots

`USE_PAR_MESH_PAIRED_PROJECTION = false` remains the default. Pairing
independent triangle-projection dot products in f64 SSE2 lanes preserves
tested outputs, but whole native C2M calls regress on PandaSet by 6.78–12.24%
and Tokyo by 6.14–7.29% in both passes. Autoware changes direction
(−7.61% / +0.61% shortening); only 9/60 timed pairs improve.
The adopted [clamped sibling-bound SIMD](cpu-c2m-native-bound-pair.md) and
other preceding native accelerations remain enabled.

## Candidate

The original closest-point calculation evaluates two dot products at each
of three stages: `ab` and `ac` against `ap`, then `bp`, then `cp`, as needed
by the original region branches. The candidate puts the two independent
results in separate f64 lanes. Each lane keeps
`a[0]*v[0] + a[1]*v[1] + a[2]*v[2]` in the preceding left-to-right reduction
order. There is no initial extra addition, FMA, horizontal reduction or
precision conversion. Region comparisons, short-circuit branches and all
remaining closest-point expressions keep their preceding order.

The same private projection helper applies to the guessed triangle and all
visited leaf triangles. Subsequent offsets, squared distance and strict hit
insertion are unchanged. A compile-time query policy selects the candidate;
it requires the adopted clamped sibling-bound policy. Other cached traversal
variants retain their original projection. The public scalar closest-point
function stays as an independent control.

SSE2 is part of native x86_64. The documented unsafe block operates only on
register operands, without pointers or unchecked memory access. Other targets
retain two scalar dot calls. No heap buffer, persistent cache, triangle-edge
cache or allocation is introduced. Nodes, triangle order, copied corners,
capacities, scalar root bounds, adopted sibling bounds, near-first ordering,
pop-time pruning, duplicate seed tests, sign/missing-hit arithmetic, original
4,096-point Morton chunks and chunk-local warm states stay unchanged.
Results retain compact f64 chunks and original serial scatter.

Selection affects native parallel C2M only. Ordinary serial/prepared/WASM/
hinted queries, GPU exports and native C2C retain their preceding selected
path. This experiment differs from the rejected
[previous-hit triangle seed](cpu-c2m-native-triangle-seed.md): it keeps no
additional per-chunk corners/edges or state.

Both builds retain [compact results](cpu-c2m-native-results.md),
[parallel construction](cpu-c2m-native-builder.md),
[bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md),
[cached bounds](cpu-c2m-native-bounds.md) and clamped sibling SIMD. Rejected
original-vertex access, duplicate seed skipping, Morton preparation, parallel
restoration, push-time pruning, triangle-seed reuse, parallel centroid bounds
and masked sibling SIMD remain disabled.

## Whole-call measurements

[Measured-source preparation](c2c-real-source.md) supplies full unchanged f64
XYZ and original source-index subsets. No generated coordinates, displacement,
noise or additional thinning is introduced. Tokyo retains its preceding
original-point preparation. These subsets are not independent surveys or
accuracy ground truth; native PLY order differs from browser LOD order.
Production Delaunay TINs keep automatic edge rejection and duplicate-XY
handling. Complete geometry bytes match both builds, passes and preceding
production oracles.

| Source | Reference/query points | TIN vertices | TIN triangles |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 501,097 | 844,324 |
| Autoware planning | 878,921 / 878,920 | 873,771 | 1,688,821 |
| Tokyo Takanawadai | 941,933 / 941,933 | 893,804 | 1,466,179 |

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run frozen release examples
with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc, optimization
level 3, LTO and one codegen unit. Twelve source/configuration fingerprints
match except for the paired-projection flag in `distance.rs`.

An independent archived baseline supplies full f64 distances, six statistics
and scalar PLY oracles, then exits before timed contexts start. Baseline and
candidate retain identical TINs/pools in separate processes; only one computes
at a time. One excluded first-use pair and five alternating warm-up pairs
precede ten alternating timed pairs per source/pass. The first pass runs
PandaSet → Autoware → Tokyo; the second reverses that order with identical
build/controller fingerprints.

Whole `cloud_to_mesh_par` timing includes fresh BVH construction, original
serial Morton preparation/sorting, parallel queries, final allocation/scatter,
destruction and pool installation. Loading, meshing, pool creation, first-use/
warm-up observations, serialization, statistics/parity and cleanup are
excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Candidate median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 382.46 | 404.52 | −6.78% | 2/10 |
| PandaSet final | 352.35 | 402.29 | −12.24% | 1/10 |
| Autoware initial | 638.77 | 683.03 | −7.61% | 0/10 |
| Autoware final | 675.77 | 680.35 | +0.61% | 6/10 |
| Tokyo initial | 1,658.33 | 1,802.31 | −7.29% | 0/10 |
| Tokyo final | 1,676.32 | 1,776.73 | −6.14% | 0/10 |

Shortening is `median(100 * (1 - candidate_ms / baseline_ms))`, not a ratio of
marginal medians. All outliers remain, including final Autoware's sixth trial
near 805.0 / 684.7 ms. Its small positive paired median coexists with a higher
candidate marginal median. Clocks, power, scheduling, other applications,
allocator behavior, code layout and retained pages are uncontrolled. Whole
results do not isolate dot-product/projection latency, packing, register
pressure, branching or cache costs. They support retaining the preceding
scalar projection. Percentages from separate experiments are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distances, six exact
statistics and full scalar PLY bytes: 192 outputs and six excluded oracles.
Geometry, thread counts and query counts match. Sources, executables,
compiler, controller, inputs/manifests and original PCD fingerprints are
checked before/after. Independent audits recompute summaries and verify
ordering/provenance, including historical clamped-sibling production oracles.
No GPU model identifiers are collected. These measurements establish no
browser, WASM, GPU, retained-mesh or Python-conversion speedup. No rebuilt
Python extension is installed.

## Validation and reproduction

With the candidate selected, 417 native tests pass with the parallel feature,
alongside native all-target/WASM Clippy denying warnings, formatting and both
release example builds. Three new regressions retain independent scalar
controls:

- 28,672 paired dot cases: three axes times 16^3 combinations, plus 16,384
  raw-bit randomized cases. Both result bits match each scalar dot, including
  NaN operand positions/payloads, signed zeros, subnormals, extrema, infinities
  and overflow.
- 13,776 closest-point cases: eight seeds, six scales, seven random/reversed/
  degenerate faces and 41 queries. The existing canonical regression also
  compares exact point bits in all seven projection regions.
- 22,272 hit cases: eight seeds, six scales, 116 queries and four seed
  policies. Triangle IDs, closest-point bits, squared-distance bits, signed
  distance bits and reusable stack capacities match adopted clamped-bound
  traversal. Nonfinite geometry and mixed missing-hit resets are covered.

The independent original-chunk regression checks 216 finite signed/unsigned
C2M comparisons across four scales and 1/2/4 threads through 65,536 points,
plus eight mixed missing-hit cases. Generated fixtures are correctness tests
only. Controller syntax and twelve invalid CLI controls pass.

After restoring the flag to false, projection regressions, canonical regions,
the independent C2M regression, native/WASM Clippy and formatting pass again,
and the native release is rebuilt. Its twelve source fingerprints match the
measured baseline. A correctness-only full PandaSet run verifies the rebuilt
executable's geometry, complete distances, statistics and scalar PLY against
the archived oracle. Rebuilt executable bytes need not equal the archive.
The stage profiling example recognizes this optional query policy while the
standard scalar projection remains selected.

Build `bench_c2m_source` in release mode with `--features parallel`, first
with `USE_PAR_MESH_PAIRED_PROJECTION = false`, then true. Keep all adopted
policies enabled, including `USE_PAR_MESH_BOUND_PAIR_CLAMP = true`, and all
other rejected policies disabled. Archive each executable as
`bench_c2m_source.exe`, the controller's twelve source/configuration files
under repository paths, and `rustc -Vv` as `compiler.txt` in separate roots.
Source guards cannot identify binary contents; matching frozen builds matter.
Run from the repository root:

```powershell
node web/scripts/benchmark-native-mesh-projection-pair.mjs --baseline-root notes/native-mesh-projection-pair-baseline-20261014 --candidate-root notes/native-mesh-projection-pair-measured-20261014 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-projection-pair.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts
1..100 (default ten). Existing reports/run directories are not overwritten.
Other checkouts need matching archives and locally prepared inputs. Native
output artifacts remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-projection-pair-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-projection-pair-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-projection-pair-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-projection-pair-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-projection-pair-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-projection-pair-final.json)
- [Controller](../web/scripts/benchmark-native-mesh-projection-pair.mjs)
- [Native selection](../rust/crates/ca-core/src/distance.rs), [projection kernels and regressions](../rust/crates/ca-core/src/mesh.rs)
- [Independent original-chunk regression](../rust/crates/ca-core/tests/mesh_parallel_stack.rs)
