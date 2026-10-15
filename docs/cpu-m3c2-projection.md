# Ordered M3C2 projection during radius traversal

Status: adopted automatically **when exact fitted normals are reused** on
the prepared browser CPU/WASM path. First calculations and fresh fitting
keep the existing hit-list projection by default. Internal
`fusedProjection: true` can also evaluate fresh fitting; `false` selects the
historical hit-list control, including with reused normals.

This follows [tree reuse](cpu-m3c2-preparation.md),
[normal reuse](cpu-m3c2-normals.md) and
[shared fresh reference searches](cpu-m3c2-shared-search.md).
Final paired whole-RPC reductions with reused normals were 13.03%, 2.13%
and 8.83%, with 24/30 faster pairs. Autoware's difference is small and noisy;
its independently calculated fused median is slower. The paired metric,
marginal medians and individual observations are all reported below.

## Implementation and output order

The historical projection first collects all spherical radius hits into a
temporary slot vector, then reads their coordinates again to calculate
deltas, cylinder membership and moments. The new
`KdTree.visit_within_with_bounds` visits those same hits in their original
near/far and leaf-slot order. It provides the already computed f64 delta
and squared distance to an ordered projection accumulator.

Sphere membership remains the existing inclusive `distance_sq <= reach²`,
with the same split-plane traversal and optional combined bounds. Projection
uses the same dot-product operation order, `distance_sq - along²`, inclusive
cylinder conditions, sequential `sum`/`sum2`, minimum count, sample variance,
LoD95 and significance arithmetic. Reusing the squared distance preserves
the original x+y+z addition order; moments are not regrouped or reduced in
parallel. Normal fitting, PCA, orientation and validity tags are unchanged.
Failed fits and fitted nonfinite normals retain their distinct cache records.
No new unsafe code or point resampling is introduced.

A normal-cache hit fuses both independent projection queries. During fresh
shared-reference fitting, the reference hit lists still supply normal fitting
and projection: the normal becomes available after that traversal. Explicit
fusion can then fuse only the compared projection. Fresh separate-search
mode can fit the normal first and fuse both subsequent projections.

Browser `M3c2Prepared.computeCached` resolves an omitted flag from the actual
normal-cache hit, not a predicted cache signature. Its uncached `compute`,
serial and copying exports default to hit lists. Each lane reports its actual
choice; result `fusedProjection` is true when every lane uses fusion. A mixed
warm/cold layout can use different choices while retaining identical results.
Native `M3c2.with_fused_projection` and `m3c2_with_projection` provide explicit
controls; ordinary fresh native serial/Rayon drivers keep hit lists.

The normal-cache key excludes this projection choice because fitted normals
are identical. Explicit switching therefore preserves valid cached normals.
The existing automatic fit policy leaves the first call uncached, captures
normals on a matching later calculation, then reuses them. With explicit
`cacheNormals: true`, capture starts on the first eligible call and later
hits automatically fuse projection. Changed fitting settings/cores and
oversized caches use the existing miss/capture behaviour.

Worker count, cyclic core scheduling, source snapshots, preparation lifetime,
failure drainage and output indexing retain their existing policies. There
is no new persistent cache. The existing 64 MiB normal allowance and 256 MiB
input-tree scheduling estimate remain in use. Warm fused projection leaves
its temporary hit vectors empty; retained WASM pages and overall peak memory
were not measured.

## Real-source comparison

The existing [source manifests](c2c-real-source.md) preserve measured f64
coordinates. All supplied points load with `keepEvery: 1`; cores are original
measured representatives selected by production voxel subsampling, not
generated coordinates. Even/odd source splits are workload geometry rather
than independent change-survey ground truth. Tokyo retains its earlier source
point selection. Parameters are normal radius 0.4 m, projection radius
0.25 m, half-depth 0.5 m and core spacing 0.25 m. Radius bounds stay disabled
and shared fresh reference searches stay enabled.

| Input | Reference / compared points | Cores / finite distances | CPU lanes | Normal-cache bytes |
|---|---:|---:|---:|---:|
| PandaSet 019 | 531,908 / 531,908 | 212,541 / 96,853 | 3 | 11,902,296 |
| Autoware planning | 878,921 / 878,920 | 696,739 / 160,487 | 2 | 39,017,384 |
| Tokyo Takanawadai | 941,933 / 941,933 | 299,755 / 281,763 | 2 | 16,786,280 |

The final reused-normal fused control **omits** the projection flag, exercising
the adopted automatic policy. Both controls require exact normal reuse.
Fresh-fitting fusion remains an explicit experiment. All times are whole
production-worker RPC milliseconds, not isolated kernel timings.

| Final reused-normal report | Hit-list median ms | Automatic fused median ms | Paired reduction | Faster pairs |
|---|---:|---:|---:|---:|
| [PandaSet](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-projection-fusion-final.json) | 295.60 | 278.90 | 13.03% | 8/10 |
| [Autoware](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-projection-fusion-final.json) | 891.75 | 922.60 | 2.13% | 7/10 |
| [Tokyo](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-projection-fusion-final.json) | 806.40 | 727.50 | 8.83% | 9/10 |

Paired reduction is `median(100 * (1 - fusedRpc / pairedHitListRpc))`, not
the ratio of marginal medians. Autoware fused observations range from
674.9 to 1,201.7 ms, versus 755.6 to 1,070.2 ms for hit lists. Its slower
marginal median and regressions remain in the report. These results support
the bounded repeated-fit policy on the tested workloads, with stronger
evidence for PandaSet/Tokyo; they do not guarantee every call/input improves.

| Final fresh-fitting experiment | Hit-list median ms | Explicit fused median ms | Paired reduction | Faster pairs |
|---|---:|---:|---:|---:|
| PandaSet | 346.00 | 344.25 | 2.74% | 6/10 |
| Autoware | 1,006.65 | 988.90 | 1.09% | 6/10 |
| Tokyo | 855.15 | 833.45 | 1.74% | 6/10 |

Fresh gains are small and inconsistent. They do not change the fresh-fitting
default. The earlier build used explicit projection flags in both cases:

| Initial report | Fresh paired reduction / faster pairs | Reused paired reduction / faster pairs |
|---|---:|---:|
| [PandaSet](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-projection-fusion.json) | 5.08% / 6/10 | 5.40% / 6/10 |
| [Autoware](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-projection-fusion.json) | −0.14% / 4/10 | 2.18% / 7/10 |
| [Tokyo](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-projection-fusion.json) | 6.06% / 7/10 | 9.94% / 9/10 |

Both rounds have positive reused-normal paired medians for every source;
fusion wins 46/60 reused-normal pairs in total. Each comparison is paired
within its own build/run. Differences between rounds do not isolate a policy
effect: clocks, power, other applications, scheduling, GC and retained WASM
pages are uncontrolled.

## Protocol and validation

Each source has separate sequential contexts for fresh fitting and reused
normals. Within a context, hit-list and fused controls share the same input
identities, retained exact trees, core layout, workers and normal policy.
One first observation per mode and five alternating warm-up pairs are
excluded; ten alternating timed pairs follow. First hit-list calls construct
trees and, where requested, capture normals. The following fused observation
already reuses them; these observations are not comparable cold timings.

Whole RPC includes core/attribute selection, bindings/transfers, search,
fitting/projection, joining, result attributes/LOD indexing and returned
arrays. Loading, pool startup, validation, exports and removal stay outside
the timer. Unexpected cache/layout state fails the run. Both rounds' 240
timed current outputs, 120 warm-up outputs and 24 first observations match
archived f32 distance/LoD95/significance bits and complete PLY bytes.
Raw f64 order/arithmetic checks are separate correctness fixtures.

Source, input, original source, manifest, script and selected production
worker/pool/WASM hashes are checked before/after. All final reports share
the same final source, artifacts and script. The initial source/script/build
is preserved in ignored `notes/m3c2-projection-initial-20261013`; the archived
pre-change oracle is in `notes/m3c2-projection-baseline-20261013`. Initial
reports are preserved. No builds/tests/scans ran alongside timing.

A full workspace run passed **375 Rust tests**. After the automatic policy
change, native M3C2/WASM checks, native and WASM Clippy with warnings denied,
rustfmt, production SIMD WASM and TypeScript/Vite builds passed again.
Native visitor checks cover negative/NaN/infinite radii, signed zero, boundary
adjacent floats, subnormal/overflow arithmetic and nonfinite points/queries.
The final **20 browser checks** cover raw f64 pairs, nonfinite/boundary
conditions, serial/copying/prepared/cached paths, first/capture/reuse
policy, explicit switching, overlapping modes, source edits/removal, worker
loss, failures and lane/normal-cache budgets. Seven invalid/overwrite CLI controls
were rejected without altering an existing report. Synthetic geometry is
used only for correctness/failure checks, not performance inputs.

## Reproduction

Build production SIMD WASM and the web app using the normal workflow. Save an
archived production `dist` before the change. Run from `web`, sequentially:

```powershell
npx playwright test e2e/m3c2-radius.spec.ts e2e/m3c2-projection.spec.ts e2e/m3c2-parallel.spec.ts e2e/m3c2-preparation.spec.ts --workers=2
node scripts/benchmark-m3c2-projection.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --baseline-dist ../notes/ARCHIVED-dist --output ../notes/NEW-m3c2-projection.json
```

Use the existing Autoware/Tokyo input directories for other sources. Output
parents must exist; reports cannot be overwritten. `--repeats` accepts
1..100, default 10. [Benchmark script](../web/scripts/benchmark-m3c2-projection.mjs).
