# SOR distance-only query slots

The experiment leaves internal tree slots in SOR's reusable nearest-k scratch
instead of gathering original point indices after every query. SOR consumes
only squared distances. Outputs remain exact, but measured whole-call gains
are inconsistent. The standard therefore retains indexed queries:
`USE_SOR_TREE_SLOTS = false`. The adopted
[sorted-prefix join](cpu-sor-prefix.md) remains enabled.

## Implementation and validation

`KdTree::nearest_k_slots_into_with_options` performs the same clear/reserve,
root/near/far traversal, insertion, tie selection and optional SIMD/bounds
dispatch as the indexed API. It leaves `(tree_slot, distance_sq)` entries
instead of applying the final `self.indices[slot]` lookup. The indexed API
still applies that conversion for consumers that need original point IDs,
including normal fitting and ICP. No public WASM request fields are added.

The candidate routes SOR's retained-scratch single-threaded means, native
Rayon chunks, part-local queries and cross-part queries through this helper.
The allocating comparison path and optional GPU-hinted query paths retain
their indexed APIs. SOR's mean, radius, reach mask and returned candidate
arrays use only each tuple's distance. Morton/query order, square roots,
summation, global threshold, partitioning and kept-point order are unchanged.
The existing scratch and prepared vectors keep their capacities/lifetimes.

The measured build passes 382 native tests with the parallel feature,
native/WASM Clippy with warnings denied, formatting and release WASM/app
builds. A new tree regression maps internal slots back to original indices
only in the test, comparing the allocating indexed oracle's hit order and
every f64 distance bit. It covers multiple counts/k, scalar/SIMD and bounds
combinations, ties, georeferenced/tiny/huge coordinates and nonfinite queries.
Prepared SOR regressions additionally use allocating indexed queries as
their independent reference for means, open order, reach and candidate bits.

Fifteen targeted browser tests pass. The new archived-WASM comparison checks
288 count/geometry/k/bounds cases, comparing every local field, external-query
distance bit, both normal orientations and retained capacity with the preceding
build. It requires `SOR_SLOT_BASELINE` and skips when no independent snapshot
is supplied. Existing tests cover global SOR/PLY with normal attributes,
prepared/fresh jobs, budgets, edits, removal, worker loss/recovery and exact
joins. Generated fixtures are used for correctness only. The final default-off
build also passes the native checks and 15 targeted browser tests.

## Real-source comparison

The full unchanged `compared.ply` clouds from
[measured-source preparation](c2c-real-source.md) supply PandaSet 019
(531,908 points), Autoware planning (878,920) and Tokyo Takanawadai (941,933).
No generated performance inputs, displacement, noise or further thinning
are added. Source attribution, licence, original preparation and hashes are
embedded in each report. These subsets are not independent surveys or
filter-quality ground truth.

Measurements use the existing Windows/i7-9750H machine, 12 logical CPUs,
32 GiB RAM and Chromium 153.0.8010.12. Two independent production workers
compare the preceding indexed build with the enabled slot-only build, both
with k=24, ratio=1, `knnCache:true` and `knnBounds:false`. Both retain the
same adopted sorted-prefix join. Cache jobs remain stable, reuse counters
increment exactly and part counts/capacities match under the existing 128 MiB
idle payload budget. A third archived worker supplies an excluded complete
output oracle and terminates before timing.

One excluded first-use pair and five excluded alternating warm-up pairs
precede ten alternating timed pairs per source. Whole SOR RPC includes
local/cross-part searches, region checks, packing/messages, joining, threshold
and filtered-cloud indexing. Startup/load, cache diagnostics, validation,
export and removal are excluded. First use is not a cold median. Sources run
PandaSet → Autoware → Tokyo, without builds/tests/scans during timing.

| Source | Indexed median, ms | Slot-only median, ms | Median paired shortening | Slot-only wins |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 427.90 | 420.70 | +3.26% | 7/10 |
| Autoware planning | 545.25 | 562.50 | −3.71% | 3/10 |
| Tokyo Takanawadai | 657.40 | 638.55 | +0.21% | 5/10 |

Paired shortening is `median(100 * (1 - slots_ms / indexed_ms))`, not the ratio
of marginal medians. The candidate wins 15/30 timed pairs and makes Autoware
slower, so it is not selected as the standard. Tokyo's lower marginal median
accompanies almost zero paired change and only five wins. The comparison
does not isolate index gathering from compiler layout or other whole-call
costs. CPU clocks, power, scheduling, GC and other applications are uncontrolled;
these observations do not establish a universal speed prediction.

All 48 first-use/warm-up/timed pairs preserve the kept count and every PLY
byte: 96 paired outputs plus three excluded archived oracles. Kept counts
remain 498,305 / 822,878 / 874,565. Prepared payloads stay 35,138,912 /
58,041,728 / 62,200,604 bytes across five/eight/eight parts. No temporary
or retained buffers are added by slot-only queries.

Input, manifest, original measured source, ten implementation sources, script
and both worker/pool/WASM builds are hashed before/after. The measured
slot-enabled source, tests, script and build are saved and independently
hash-audited before restoring indexed SOR. **The table measures that candidate,
not the final default-off build.** No final-build speedup is claimed.

## Reports and reproduction

- [PandaSet report](../benchmarks/cpu/sor/real-source/windows-pandaset019-tree-slots-initial.json)
- [Autoware report](../benchmarks/cpu/sor/real-source/windows-autoware-planning-tree-slots-initial.json)
- [Tokyo report](../benchmarks/cpu/sor/real-source/windows-tokyo-takanawadai-tree-slots-initial.json)
- [Benchmark script](../web/scripts/benchmark-sor-slots.mjs)
- [Archived-WASM test](../web/e2e/sor-slots.spec.ts)

Archive the preceding indexed production build first. Set
`USE_SOR_TREE_SLOTS = true` in `filter.rs` and rebuild matching release
WASM/app, retaining the adopted prefix join. Run from the repository root:

```powershell
node web/scripts/benchmark-sor-slots.mjs --baseline-dist notes/sor-slot-baseline-20261013/dist --input notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-tree-slots.json
```

The source guard rejects the false flag; it cannot identify the binary mode,
so a matching rebuild is required. Full measured input/manifest and a new
output in an existing parent are mandatory. Reports cannot be overwritten.
`--repeats` accepts 1..100, default ten. Nine invalid-option/overwrite/manifest
cases and the disabled-source guard pass. For the browser oracle, supply
`SOR_SLOT_BASELINE` as an absolute snapshot root containing `dist` and its
matching `web/src/wasm/ca_wasm.js`. Restore the flag to false and rebuild
after comparison. No GPU adapter/model metadata is collected.
