# C2M traversal with cached child bounds

Ordinary C2M batch queries now reuse child box distances already computed for
BVH traversal ordering. The adopted retained CPU baseline improves by
23.80–30.38% in whole unsigned C2M calls on three measured sources across two
passes, with 60/60 faster pairs and exact tested outputs.
`USE_MESH_STACK_BOUNDS = true` enables the batch driver by default.
The later [query-order reuse experiment](cpu-c2m-query-order.md) remains off
after repeated regressions on one source; cached traversal bounds stay enabled.
The later [borrowed-query experiment](cpu-c2m-query-borrow.md) also remains
off after both Autoware passes regress.

## Computation and storage

The preceding traversal computes both child box distances to decide which
child to visit first, pushes their indices, then computes each box distance
again when that child is popped. The new temporary stack stores `(index,
lower_distance_sq)` together, using that same f64 value on pop. Root distance
is computed once when pushed. Immutable geometry and the query remain the
same throughout a traversal.

Pruning still occurs on pop against the latest best triangle distance. The
nearer child is pushed last with the original `dl <= dr` comparison; the
leaf triangle order and strict distance insertion remain unchanged. Retaining
these choices preserves the guessed triangle and signed equal-distance ties.
Original-coordinate closest-point calculations, square root, normals and
signed-distance arithmetic are unchanged. Each batch retains its original
Morton order and starts with a fresh guess/stack, reusing scratch only between
queries in that batch. Original public nearest-query methods remain available;
hinted GPU verification retains its preceding traversal. No new public request
control or GPU dispatch is introduced.

The stack has an additional f64 value per live item plus layout padding. It
remains temporary query scratch, freed at the end of the batch. No query,
distance, additional geometry or cross-call bound cache is retained. Prepared
mesh vector capacities and the **1 GiB** aggregate geometry reservation policy
are identical to the preceding build. That allowance covers retained geometry,
not temporary stacks/copies, allocator overhead or process peak memory.

## Measured geometry and controls

Full unchanged inputs follow [measured-source preparation](c2c-real-source.md).
No generated coordinates, displacement, noise, input resampling or additional
thinning is introduced. Tokyo retains its earlier original-point preparation.
These source-index subsets are workload geometry, not independent surveys or
accuracy/change ground truth.

Each context produces a full-input 2.5D Delaunay TIN with automatic edge
rejection. Meshing and duplicate-XY handling retain their production policy;
no meshing voxel selection is used. Mesh descriptors, display-position/index
hashes and full binary PLY hashes match across contexts and passes. The mesh
PLY contains original f64 coordinates and all face indices, so this checks
more than shifted f32 display geometry.

| Source | Query points | TIN vertices | Retained triangles | BVHs / logical slices | Retained geometry bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 | 501,097 | 844,324 | 5 / 5 | 668,045,480 |
| Autoware planning | 878,920 | 873,771 | 1,688,821 | 4 / 8 | 1,056,643,136 |
| Tokyo Takanawadai | 941,933 | 893,804 | 1,466,179 | 4 / 8 | 930,324,512 |

Windows/i7-9750H, 12 logical CPUs, 32 GiB RAM and Chromium 153.0.8010.12
run archived/current production contexts separately. Both use the adopted
retained CPU mesh/BVH policy, identical worker counts, original logical query
slices, capacities and reservations. Every supplied query loads with
`keepEvery: 1`; GPU hints are explicitly disabled and all timed calls are
unsigned. The original logical slices are kept even when fewer BVHs serve
them, preserving per-slice guess/tie behavior.

An independent archived worker supplies f32 distance, exact f64 summary and
full scalar PLY oracles, then terminates before timing. Separate timed
contexts have one excluded cold first-use pair and five excluded alternating
warm-up pairs before ten alternating timed pairs. Warm calls must reuse
geometry without fallback. Initial order is PandaSet → Autoware → Tokyo;
the final pass reverses it with identical source/script/build fingerprints.
Both timed contexts coexist, with their retained meshes resident.

Whole RPC includes query copies/transfers, Morton ordering, traversal/triangle
tests, joins, summaries and returned arrays. Loading, production meshing,
pool startup, first observations, warmups, export/parity and cleanup are
excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Cached-bound median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 832.80 | 640.10 | +23.80% | 10/10 |
| PandaSet final | 751.65 | 555.40 | +25.05% | 10/10 |
| Autoware initial | 1,408.95 | 1,002.60 | +28.06% | 10/10 |
| Autoware final | 1,385.55 | 986.95 | +28.58% | 10/10 |
| Tokyo initial | 5,020.60 | 3,749.70 | +24.90% | 10/10 |
| Tokyo final | 4,975.95 | 3,560.20 | +30.38% | 10/10 |

Shortening is `median(100 * (1 - bounded_ms / baseline_ms))`, not the ratio
of marginal medians. All six medians improve and all 60 timed pairs are
faster, supporting adoption for these workloads. All observations remain,
including the initial 1,936.3/1,043.5 ms Autoware warm-up and variation across
Tokyo calls. Clocks, power, scheduling, GC, other applications, code layout
and retained pages are uncontrolled. This is a complete-call improvement,
with no isolated traversal, signed, native or GPU timing claim. First-use
BVH construction remains expensive; these percentages are warm retained-call
comparisons and should not be added to preceding cache-adoption percentages.

All first-use/warm-up/timed outputs match complete f32 distance bits, every
f64 summary value and scalar PLY bytes: 96 pairs, 192 outputs and six excluded
oracles. Retained geometry capacities/reservations remain identical in both
modes. Eleven relevant implementation files, both three-artifact sets,
script, full inputs, manifest and original PCD are hashed before/after.
A separate audit checks saved measured source/build fingerprints, all input
proofs, geometry/oracle consistency, cache/layout, pair order and summaries.
No GPU model identifiers are collected.

## Validation and reproduction

392 native tests pass with the parallel feature, along with native/WASM
Clippy denying warnings, formatting and release WASM/TypeScript/Vite builds.
The focused native test compares hit triangle, closest-point f64 bits and
squared-distance bits across scales/offsets, reversed/degenerate faces, cold/
guessed searches and nonfinite queries; valid hits also compare signed bits.

Nine browser tests pass. An independent archived WASM checks 240 count/
geometry/scale/sign conditions with raw f64 results, both fresh and prepared
meshes, repeated/changed query batches and identical retained bytes. Existing
CPU mesh tests cover signed ties, overlapping batches, budgets, source edits/
removal and failed-worker recovery. Two GPU lifecycle/quota mock tests pass;
three opt-in physical-GPU scenarios are skipped. This experiment adds no GPU
kernel, and no new physical-GPU result is claimed. Generated fixtures check
correctness only. Benchmark syntax and nine invalid-option/repeat/overwrite/
manifest controls pass.

Archive the preceding retained CPU production build first. Enable the source
flag and rebuild matching release WASM/app, then run from the repository root
with unchanged measured inputs and a new output:

```powershell
node web/scripts/benchmark-mesh-stack-bounds.mjs --baseline-dist notes/mesh-stack-bounds-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-mesh-stack-bounds.json
```

The benchmark requires an enabled source flag and distinct archived artifacts;
source inspection cannot identify binary mode, so matching rebuilds matter.
`--repeats` accepts 1..100, default ten; existing reports are not overwritten.
Other checkouts need equivalent archived builds and locally prepared measured
inputs. The raw browser regression needs optional `MESH_STACK_BOUNDS_BASELINE`
as an absolute snapshot root containing `dist` and matching
`web/src/wasm/ca_wasm.js`; it skips without that independent archive.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-stack-bounds-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-stack-bounds-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-stack-bounds-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-stack-bounds-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-stack-bounds-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-stack-bounds-final.json)
- [Benchmark](../web/scripts/benchmark-mesh-stack-bounds.mjs)
- [Archived-WASM regression](../web/e2e/mesh-stack-bounds.spec.ts)
- [Traversal](../rust/crates/ca-core/src/mesh.rs)
- [Batch driver](../rust/crates/ca-core/src/distance.rs)
