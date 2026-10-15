# C2M query-order reuse experiment

Repeated retained C2M calls still compute Morton query order for every logical
slice. A candidate retains coordinate-matched query orders under a separate
allowance. Tested outputs match, but PandaSet slows in both measured passes,
and only 38/60 timed pairs improve overall. It remains **off by default**:
`USE_MESH_QUERY_ORDER = false`. The adopted
[cached traversal bounds](cpu-c2m-stack-bounds.md) remain enabled.

## Matching, computation and lifetime

One `MeshQueryOrder` object on each pool worker retains at most eight query
sets in least-recently-used order. Each set owns exact original f64 XYZ and
the preceding `morton_order` index permutation, in boxed storage. Incoming
length and every coordinate's f64 bits must match. Signed zero, NaN payloads,
point edits, permutations and changed logical slices cannot silently share a
different order. The order depends on query coordinates rather than mesh or
signed mode, so matching queries can reuse it across those changes.

Hits still receive/copy incoming queries and execute all triangle searches,
with fresh guessed triangle and traversal scratch for every batch. Closest
point, strict insertion, cached child bounds, signed arithmetic, logical
slices, joins and summary reductions preserve their preceding policy. No
distance/result cache is retained. Misses compute the original unstable
Morton sort, then store its exact resulting permutation. Old payload is
evicted before admitting a replacement. Empty/oversized inputs, explicit
`meshOrderCache: false`, and hinted queries clear the payload and calculate
with the ordinary path.

Each worker's boxed payload is bounded at **8 MiB**, with at most eight
additional entry headers. The pool is capped at eight workers, so the payload
is at most **64 MiB** per analysis context across all live pool lanes. WASM
payload is 28 bytes per query: 24-byte XYZ plus a four-byte index; native
indices use their target width. This is separate from the existing 1 GiB
aggregate mesh/BVH reservation. Temporary input copies, sort scratch,
allocator/entry headers and retained WASM pages are not a peak-heap guarantee.

Source edits/removal queue release only on live generations, without starting
workers. Changed jobs also release after their responses drain, covering a
delayed old request that could otherwise repopulate a cleared cache. Failed
mesh runs drain, release order payload and retire geometry leases. The cache
is shared across mesh jobs on one worker, so these releases may also evict
unrelated idle query orders; output correctness does not depend on retention.

`meshQueryOrderHit` requires every logical slice to hit. Byte/entry diagnostics
sum the maximum observed value from each participating lane's responses; they
are not a complete snapshot of idle lanes or concurrent jobs. Source changes,
different jobs/layouts and quota eviction can cause misses. The default-off
build constructs no query-order object through ordinary requests and reports
false/zero. Direct experimental WASM use remains available for reproduction.
The internal control adds no product setting or GPU dispatch.

## Measured comparison

Inputs follow [measured-source preparation](c2c-real-source.md), unchanged and
loaded in full with `keepEvery: 1`. No generated coordinates, displacement,
noise or additional thinning is used; Tokyo retains its preceding original
point preparation. These source-index subsets are workload geometry, not
independent acquisition pairs or accuracy/change ground truth.

Each context creates a full-input production 2.5D Delaunay TIN with automatic
edge rejection and recorded duplicate-XY handling. Mesh descriptors, shifted
display/index hashes and full binary PLY geometry hashes match across all
contexts/passes. That PLY contains original f64 vertices and face indices.

| Source | Queries | TIN vertices / triangles | BVHs / logical slices | Extra query-order bytes |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 | 501,097 / 844,324 | 5 / 5 | 14,893,424 |
| Autoware planning | 878,920 | 873,771 / 1,688,821 | 4 / 8 | 24,609,760 |
| Tokyo Takanawadai | 941,933 | 893,804 / 1,466,179 | 4 / 8 | 26,374,124 |

Windows/i7-9750H, 12 logical CPUs, 32 GiB RAM and Chromium 153.0.8010.12
run separate archived/current production contexts. Both already use adopted
cached traversal bounds and retained CPU geometry under identical 1 GiB
reservations. Mesh payloads remain 668,045,480 / 1,056,643,136 / 930,324,512
bytes. Timings are unsigned, with GPU hints disabled, original query slices
and unchanged resident workers. Both timed contexts coexist with their
retained geometry resident; only the candidate retains query-order payload.

An independent archived worker supplies complete f32 distances, exact f64
summaries and scalar PLY oracles, then terminates before timing. One excluded
cold first-use pair and five alternating warm-up pairs precede ten alternating
timed pairs per source/pass. Initial order is PandaSet → Autoware → Tokyo;
the final pass reverses it with identical source/script/build fingerprints.
Candidate warm calls must hit both geometry and query-order reuse. Whole RPC
includes copies, full coordinate matching/order when needed, searches, joins,
summaries and returned arrays. Loading, meshing, pool startup, first/warm-up
observations, export/parity and cleanup are excluded. No builds, tests, scans
or edits run alongside timing.

| Source/pass | Baseline median, ms | Reused-order median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 505.70 | 512.70 | −2.57% | 3/10 |
| PandaSet final | 519.45 | 548.50 | −3.97% | 2/10 |
| Autoware initial | 1,032.25 | 991.85 | +2.20% | 7/10 |
| Autoware final | 1,111.30 | 1,014.15 | +6.80% | 8/10 |
| Tokyo initial | 3,954.75 | 3,781.20 | +2.02% | 8/10 |
| Tokyo final | 3,820.30 | 3,704.45 | +2.30% | 10/10 |

Shortening is `median(100 * (1 - ordered_ms / baseline_ms))`, not a ratio of
marginal medians. PandaSet regresses in both passes, with only 5/20 faster
pairs; Autoware/Tokyo gains are smaller than the preceding traversal change.
The total is 38/60 faster pairs. Additional coordinate storage/matching and
other runtime effects do not establish a broadly useful default tradeoff.
All observations remain, including the initial Tokyo 4,972.8/6,780.1 ms
warm-up pair. Clocks, power, scheduling, GC, other applications, code layout
and retained pages are uncontrolled. These are whole-call observations, with
no isolated sort, signed, native or GPU speedup claim. Candidate percentages
describe the separately archived enabled build, not a newly enabled feature
in the final default-off build.

All first-use/warm-up/timed outputs match full f32 distance bits, every f64
summary value and scalar PLY bytes: 96 pairs, 192 outputs and six excluded
oracles. Candidate payload is exactly 28 bytes per source query, warm order
hits are mandatory, and geometry capacities/reservations remain identical.
Eleven relevant implementation files, script, both three-artifact sets,
inputs, manifest and original PCD are hashed before/after and independently
audited against the saved enabled source/build. No GPU model identifiers
are collected.

## Validation and reproduction

394 native tests pass with the parallel feature, together with native/WASM
Clippy denying warnings, formatting and release WASM/TypeScript/Vite builds.
Focused native tests cover exact distances/signs, LRU/quota/eight-entry
eviction, release and signed-zero/NaN-payload key distinctions.

Eleven browser tests pass on the enabled candidate and again on the final
default-off app build. An independent archived WASM checks 60 count/scale/
sign conditions with raw f64 output, changed/permuted queries, mesh/sign
switches, clear and undersized allowances. The preceding traversal oracle
also passes. Production worker tests cover cold/warm/disabled diagnostics,
overlapping calls, source edits/removal, delayed obsolete responses and
failed-worker recovery. Release tests check that only live lanes are used.
Two GPU lifecycle/quota mocks pass; three opt-in physical-GPU scenarios skip.
Generated fixtures are for correctness only. Benchmark syntax and nine
invalid-option/repeat/overwrite/manifest controls pass. Default-off benchmark
runs reject before browser launch/report creation.

At the time of this experiment, ordinary WASM `MeshPrepared.distances` used
`from_flat`, allocating another XYZ vector from already imported flat queries.
The subsequent [borrowed-query experiment](cpu-c2m-query-borrow.md) tests
removing that copy. It also remains off by default after both Autoware passes
regress; ordinary queries continue to use owned triples.

Archive the adopted cached-bound CPU build first. Enable
`USE_MESH_QUERY_ORDER = true`, retain `USE_MESH_STACK_BOUNDS = true`, rebuild
matching release WASM/app, and run from the repository root with unchanged
measured inputs and a new output:

```powershell
node web/scripts/benchmark-mesh-query-order.mjs --baseline-dist notes/mesh-query-order-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-query-order.json
```

Source guards cannot identify binary mode; matching rebuilds matter. `--repeats`
accepts 1..100, default ten; existing reports are not overwritten. Other
checkouts need equivalent archived builds and locally prepared measured inputs.
The raw browser oracle needs optional `MESH_QUERY_ORDER_BASELINE` as an
absolute snapshot root containing `dist` and matching `web/src/wasm/ca_wasm.js`;
it skips without that independent archive.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-query-order-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-query-order-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-query-order-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-query-order-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-query-order-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-query-order-final.json)
- [Benchmark](../web/scripts/benchmark-mesh-query-order.mjs)
- [Archived-WASM regression](../web/e2e/mesh-query-order.spec.ts)
- [Cache implementation](../rust/crates/ca-core/src/mesh.rs)
- [Pool dispatch](../web/src/c2c-worker.ts)
