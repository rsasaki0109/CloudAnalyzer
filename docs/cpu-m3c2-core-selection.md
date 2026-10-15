# Retained M3C2 core selection

Repeated parallel M3C2 calls used to select the same voxel core points again,
even when prepared trees and fitted normals were already reused. The adopted
path retains one bounded list of original compared-source point indices.
Whole-call paired reductions are 2.29–12.00% on three measured sources across
two passes, with 100/120 faster pairs and exact tested outputs.
`USE_M3C2_CORE_SELECTION = true` enables the policy by default.

## Selection and lifetime

The first eligible call uses the original `voxel_subsample` operation: the
first original point in each occupied voxel, in source order. Its origin,
cell keys, membership, spacing semantics and selected coordinates are unchanged.
The resulting `Uint32Array` is keyed by compared `Cloud` identity and positive
finite spacing. Matching calls pass it to `M3c2Plan.fromIndices`, which checks
range and strict ascending order before selecting from the current cloud.

Every plan still takes a fresh owned snapshot of coordinates, colours and all
attributes synchronously before the first yield. Attributes are never cached;
in-flight plans survive source edits/removal and cache replacement. Existing
fit/projection arithmetic, paired eigensolvers, core partitioning, normal
capture/reuse, joins and result indexing are preserved. Radius, layout and
reference changes need no new compared-source selection, although existing
tree/normal invalidation still applies. Compared edits/removal explicitly
invalidate the list. Different compared identities or spacing replace it.

One index payload is retained on the analysis lane, bounded at **8 MiB**.
The conservative admission limit is 2,097,152 compared points, regardless of
the number that voxel selection might keep. Oversized inputs, zero/negative/
nonfinite spacing and `cacheCoreSelection: false` use the preceding plan
constructor and release the index list. The existing scheduling checks still
decide whether the job runs in parallel; selection can be prepared before the
selected-core-count check sends a heavily spaced job to the serial path.
Small jobs rejected before planning retain their preceding serial calculation.

The 8 MiB charge describes index payload, not a process peak-heap bound.
Selection misses and plan construction have temporary JS/WASM index copies;
each concurrent plan still owns its coordinate/attribute snapshot. Trees,
normal caches, core/result buffers and retained WASM pages have their existing
allowances. No selected coordinates/attributes, additional tree or pool index
cache is retained by this policy. The input-copy scheduling estimate and
64 MiB normal allowance are unchanged; the new index allowance is separate.

`coreSelectionCacheHit` and `coreSelectionCacheBytes` describe the selection
used by that parallel call, captured before yielding. Serial responses report
false/zero; an earlier idle list can still exist when scheduling returns before
planning. These diagnostics and the comparison control are internal protocol
fields; there is no new product setting or GPU dispatch.

## Measured comparison

The full unchanged even/odd source-index inputs follow
[measured-source preparation](c2c-real-source.md). No generated coordinates,
displacement, noise or additional thinning is introduced. Tokyo retains its
preceding original-point preparation. These are measured geometry subsets,
not independent acquisition pairs or change ground truth.

| Source | Reference / compared points | Original measured cores | Finite distances | Lanes | Index payload, bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 212,541 | 96,853 | 3 | 850,164 |
| Autoware planning | 878,921 / 878,920 | 696,739 | 160,487 | 2 | 2,786,956 |
| Tokyo Takanawadai | 941,933 / 941,933 | 299,755 | 281,763 | 2 | 1,199,020 |

The archived baseline already adopts paired fresh M3C2 fitting and disables
the rejected short-pair guard. The candidate changes selection preparation;
the preceding fitting/search/projection policies remain the comparison baseline.
Windows/i7-9750H, 12 logical CPUs, 32 GiB RAM and Chromium 153.0.8010.12 run
separate production contexts. Normal radius is 0.4 m, projection radius
0.25 m, half-depth 0.5 m and core spacing 0.25 m. All input points load with
`keepEvery: 1`; prepared trees, resident analysis lane, interleaving and shared
search are enabled, with radius bounds disabled.

Fresh-normal calls disable normal reuse and projection fusion. Reused-normal
calls capture exact normals and automatically fuse projection on warm hits.
Both cases reuse candidate core indices; this is additional work avoided even
when fitting was already bypassed. An independent archived worker provides
full-output/PLY oracles and terminates before timing. Two excluded first
observations and five alternating warm-up pairs precede ten alternating timed
pairs per source/case/pass. Initial order is PandaSet → Autoware → Tokyo;
the final pass reverses that order, using identical source/script/build hashes.

Whole RPC includes selection/plan snapshots, transfers, queries, fitting when
needed, projection, joins, attributes/LOD and returned arrays. Loading, startup,
first-use/warm-up observations, export, validation and removal are excluded.
No builds, tests, scans or edits run alongside timing.

| Source/pass | Case | Baseline median, ms | Retained median, ms | Median paired shortening | Faster pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| PandaSet initial | Fresh normals | 344.90 | 329.65 | +3.28% | 8/10 |
| PandaSet final | Fresh normals | 416.10 | 362.75 | +10.60% | 10/10 |
| Autoware initial | Fresh normals | 1,104.50 | 1,065.45 | +5.35% | 7/10 |
| Autoware final | Fresh normals | 1,186.65 | 1,033.65 | +8.57% | 8/10 |
| Tokyo initial | Fresh normals | 874.20 | 836.80 | +3.53% | 9/10 |
| Tokyo final | Fresh normals | 860.55 | 836.60 | +2.57% | 7/10 |
| PandaSet initial | Reused normals | 292.45 | 258.20 | +9.62% | 9/10 |
| PandaSet final | Reused normals | 259.85 | 236.95 | +12.00% | 9/10 |
| Autoware initial | Reused normals | 913.95 | 881.40 | +2.29% | 5/10 |
| Autoware final | Reused normals | 983.05 | 894.45 | +11.48% | 10/10 |
| Tokyo initial | Reused normals | 648.15 | 613.20 | +4.21% | 9/10 |
| Tokyo final | Reused normals | 651.60 | 602.30 | +7.49% | 9/10 |

Shortening is `median(100 * (1 - retained_ms / baseline_ms))`, not a ratio of
marginal medians. Fresh calls have 49/60 faster pairs and reused-normal calls
51/60. All twelve paired medians improve, supporting adoption for these
repeated-source workloads. Initial Autoware reused calls improve in only half
the pairs; all losses and spikes remain, including the final reused
1,882.1/1,548.0 ms Autoware pair. Clocks, power, scheduling, GC, other apps,
code layout and retained pages are uncontrolled. These are complete-call
comparisons, with no isolated selection, native or GPU speedup claim.

All first-use/warm-up/timed outputs match f32 distance/LoD95 bits,
significance values and full PLY bytes: 192 pairs, 384 outputs and six
excluded oracles. Candidate core hits are false initially and true when warm,
with exactly four retained bytes per selected core. Normal payloads remain
11,902,296 / 39,017,384 / 16,786,280 bytes in reused cases and zero in fresh
cases, within 64 MiB. Input/manifest/original PCD, seventeen implementation
files, script and both three-artifact sets are checked before/after, then
independently audited against the saved measured source/build. No GPU model
identifiers are collected.

## Validation and reproduction

391 native tests pass with the parallel feature, together with native/WASM
Clippy denying warnings, formatting and release WASM/TypeScript/Vite builds.
The new native test rebuilds selections across spacing controls, attribute
revisions and both core layouts. Thirty browser tests pass: index admission/
replacement/invalidation and owned snapshots, twelve complete PLY comparisons
through controls/layouts/radii/normal reuse/source transforms/concurrency/
removal/reload, and five malformed WASM index rejections. Existing independent
M3C2/normal/voxel oracles, worker recovery, layouts and budgets also pass.
Generated fixtures are for correctness only. Benchmark syntax and nine
invalid-option/repeat/overwrite/manifest checks pass.

Archive the preceding production build before enabling this policy. Rebuild
matching release WASM/app, then run from the repository root with unchanged
measured inputs and a new output:

```powershell
node web/scripts/benchmark-m3c2-core-selection.mjs --baseline-dist notes/m3c2-core-selection-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-core-selection.json
```

The script requires the enabled source flag and a distinct archived build;
source inspection cannot identify binary mode, so matching rebuilds matter.
`--repeats` accepts 1..100, default ten. Existing reports are not overwritten.
Another checkout needs equivalent archived artifacts and locally prepared
measured inputs. The three new browser checks need no private baseline;
existing independent-oracle checks retain their documented optional snapshots.

- PandaSet: [initial](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-core-selection-initial.json), [final](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-core-selection-final.json)
- Autoware: [initial](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-core-selection-initial.json), [final](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-core-selection-final.json)
- Tokyo: [initial](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-core-selection-initial.json), [final](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-core-selection-final.json)
- [Benchmark](../web/scripts/benchmark-m3c2-core-selection.mjs)
- [Browser regression](../web/e2e/m3c2-core-selection.spec.ts)
- [Selection policy](../web/src/m3c2-core-selection.ts)
- [WASM plan construction](../rust/crates/ca-wasm/src/m3c2.rs)
