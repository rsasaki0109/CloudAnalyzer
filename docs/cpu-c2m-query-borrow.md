# C2M borrowed-query experiment

Borrowing imported WASM queries removes a redundant XYZ allocation, but does
not consistently improve whole C2M calls. PandaSet and Tokyo improve slightly;
Autoware regresses in both measured passes. Only 36/60 timed pairs improve.
The candidate remains **off by default**:
`USE_MESH_BORROWED_QUERIES = false`.
The adopted [cached traversal bounds](cpu-c2m-stack-bounds.md) remain enabled;
the [query-order cache](cpu-c2m-query-order.md) remains disabled.

## Ownership and unchanged computation

Wasm-bindgen already copies each JavaScript `Float64Array` into a flat WASM
input slice. The preceding `from_flat(queries).positions` then copies those
coordinates into another owned `Vec<[f64; 3]>` for a synchronous search.
The candidate `mesh_queries` helper validates the same multiple-of-three
length and uses safe `as_chunks::<3>()` to borrow that imported storage.
`Cow` selects borrowed storage when enabled and the preceding owned-triples
allocation when disabled. No unsafe conversion or input mutation is used.

This applies to `meshDistances`, `MeshPrepared.distances` and
`MeshPrepared.distancesHinted`. The borrow ends before the exported call
returns. No borrowed queries or computed results are retained across calls.
Owned mesh/BVH storage is unchanged. Experimental `MeshQueryOrder` still
requires owned query snapshots and is outside this change.

The enabled candidate avoids a 24-byte-per-point payload allocation and copy
in each query batch. This is an allocation calculation, not measured peak
process memory: imported inputs, outputs, sort/traversal scratch, allocator
metadata and retained WASM pages still exist. JS-to-WASM copying remains.
Morton ordering, guesses, traversal bounds, strict triangle insertion,
signed arithmetic, logical slices, joins and summary reductions are unchanged.
No new public request, UI setting or GPU dispatch is added.

## Measured results

Inputs follow [measured-source preparation](c2c-real-source.md), loaded in
full with `keepEvery: 1`. XYZ and original source indices are unchanged;
there are no generated coordinates, added displacement/noise or additional
thinning. Tokyo retains its preceding original-point preparation. These are
workload subsets, not independent surveys or accuracy/change ground truth.
Full-input production Delaunay TIN descriptors, shifted display/index hashes
and full binary PLY geometry hashes match across contexts and both passes.

| Source | Queries | TIN vertices / triangles | BVHs / logical slices | Avoided XYZ payload per call |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 | 501,097 / 844,324 | 5 / 5 | 12,765,792 bytes |
| Autoware planning | 878,920 | 873,771 / 1,688,821 | 4 / 8 | 21,094,080 bytes |
| Tokyo Takanawadai | 941,933 | 893,804 / 1,466,179 | 4 / 8 | 22,606,392 bytes |

Windows/i7-9750H, 12 logical CPUs, 32 GiB RAM and Chromium 153.0.8010.12
run separate archived/current production contexts. Both use adopted cached
traversal bounds, identical 1 GiB mesh/BVH policy, resident workers and
original slices. Retained mesh payloads are 668,045,480 / 1,056,643,136 /
930,324,512 bytes in both builds. Query-order diagnostics remain false/zero.
Only ordinary unsigned prepared CPU calls are timed; fresh, hinted and signed
calls receive correctness checks without a speedup claim.

An independent archived worker supplies full f32 distances, exact f64
summaries and scalar PLY oracles, then terminates before timing. One excluded
first-use pair and five alternating warm-up pairs precede ten alternating
timed pairs per source/pass. Initial order is PandaSet → Autoware → Tokyo;
the final pass reverses it with identical source/script/build fingerprints.
Whole RPC includes copies/transfers, ordering/search, joins, summaries and
returned arrays. Loading, production meshing, startup, first/warm-up calls,
export/parity and cleanup are excluded. No builds, tests, scans or edits run
alongside timing. Geometry misses, fallback, layout/budget mismatch,
query-order hits or output differences fail the measurement.

| Source/pass | Baseline median, ms | Borrowed median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 668.10 | 652.60 | +2.59% | 7/10 |
| PandaSet final | 494.50 | 468.45 | +3.81% | 8/10 |
| Autoware initial | 1,090.00 | 1,279.00 | −13.90% | 3/10 |
| Autoware final | 936.00 | 942.75 | −1.65% | 4/10 |
| Tokyo initial | 4,108.55 | 3,844.45 | +2.11% | 6/10 |
| Tokyo final | 3,802.05 | 3,765.70 | +3.65% | 8/10 |

Shortening is `median(100 * (1 - borrowed_ms / baseline_ms))`, not a ratio
of marginal medians. Autoware improves in only 7/20 pairs and regresses in
both passes, so the allocation reduction does not justify enabling this
candidate across ordinary C2M workloads. All observations and spikes remain.
Clocks, power, scheduling, GC, other applications, code layout and retained
pages are uncontrolled; the measurements do not identify the cause of the
regressions. No isolated copy/search, native, signed or GPU speedup is claimed.
The table describes the separately archived enabled candidate, not the final
default-off build or its binary identity with the preceding baseline.

All 96 first-use/warm-up/timed pairs match complete f32 distance bits, every
f64 summary value and full scalar PLY bytes: 192 outputs and six excluded
oracles. Eleven implementation files, script, both three-artifact sets,
inputs, manifests and original PCD are hashed before/after. A separate audit
checks the saved enabled source/build, geometry, budgets, pair order and all
summaries. No GPU model identifiers are collected.

## Validation and reproduction

395 native tests pass with the parallel feature. The focused ownership test
checks storage identity when borrowing is selected and preserves signed-zero,
NaN-payload and infinity bits. Native/WASM Clippy denying warnings, formatting,
release WASM and TypeScript/Vite builds pass.

Twelve browser checks cover the archived raw f64 oracle, earlier traversal and
order-cache oracles, production geometry leases, overlapping requests,
edits/removal, delayed obsolete responses and failed-worker recovery.
The new raw oracle exercises 240 geometry/count/scale/sign conditions,
repeated queries, hinted calls, subarray inputs, caller-storage preservation,
later mutations and malformed query/candidate lengths. Single unsigned
NaN/infinity queries also match. At the time of this experiment, kernels
could trap on some nonfinite batches or signed missing hits. The subsequent
[missing-hit correction](cpu-c2m-missing-hit.md) handles those cases without
enabling this borrowing experiment. Generated inputs are correctness fixtures only.
Two GPU lifecycle/quota mocks pass; three opt-in physical-GPU scenarios skip.
Benchmark syntax and nine invalid CLI controls pass. Default-off runs reject
before browser launch/report creation.

Archive the preceding adopted production build and matching WASM glue first.
Set `USE_MESH_BORROWED_QUERIES = true`, keep cached traversal bounds enabled
and query-order caching disabled, then rebuild matching release WASM/app.
Run from the repository root with unchanged measured inputs and a new output:

```powershell
node web/scripts/benchmark-mesh-query-borrow.mjs --baseline-dist notes/mesh-query-borrow-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-query-borrow.json
```

Source guards cannot identify binary mode; matching rebuilds matter.
`--repeats` accepts 1..100, default ten; existing reports are not overwritten.
Other checkouts need equivalent archived builds and locally prepared inputs.
The optional raw oracle requires `MESH_QUERY_BORROW_BASELINE` as an absolute
snapshot root containing `dist` and matching `web/src/wasm/ca_wasm.js`; it
skips without an independent archive.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-query-borrow-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-query-borrow-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-query-borrow-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-query-borrow-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-query-borrow-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-query-borrow-final.json)
- [Benchmark](../web/scripts/benchmark-mesh-query-borrow.mjs)
- [Archived-WASM regression](../web/e2e/mesh-query-borrow.spec.ts)
- [WASM bindings](../rust/crates/ca-wasm/src/lib.rs)
