# Retained voxel neighbours for Map quality

Owned reference voxel preparation now retains the 27 neighbouring reference
slot numbers for SCS. Repeated evaluation against that reference reads indexed
distances instead of rebuilding a distance HashMap and performing 27 coordinate
hash lookups per matched voxel. `USE_VOXEL_NEIGHBORS = true` enables this path.

Two passes on three unchanged measured sources reduce whole voxel-score RPC
time by **16.67–34.09%**, with **59/60 timed pairs** faster. Combined Map
evaluation improves **10.80–15.47%** on Autoware; PandaSet and Tokyo vary from
about −2.08% to +1.14%. The adopted benefit is strongest in voxel scoring.
The earlier [paired cross-matrix experiment](cpu-voxel-pair.md) stays off;
[paired normal fitting](cpu-normal-pair.md) stays enabled.

## Computation and exactness

The existing [reference Gaussian/root cache](cpu-voxel-reference.md) assigns
each retained Gaussian a stable slot. During preparation, a `u32[27]` row
records neighbouring reference slots in the preceding dx/dy/dz traversal
order. `u32::MAX` means no neighbour. Extreme i64 keys and references too large
for the slot representation retain the preceding hashed score path.

Every compared calculation still builds its own Gaussian statistics and
computes every individual Wasserstein distance with the preceding scalar
matrix/root arithmetic. Reference roots remain lazy and reuse counters retain
their meaning. The current call creates a fresh `Option<f64>` array indexed
by reference slots and a list of matched slots in compared traversal order.
An absent compared voxel stays `None`, so reference-only neighbours never
enter the SCS neighbourhood. No previous distances are retained or reused.

The neighbourhood's value sequence, mean and variance operation order remain
unchanged. AWD and global SCS sum order now follow the compared traversal
instead of the separately randomized distance HashMap. These aggregate scores
retain the existing `1e-12` relative/absolute reduction allowance. Individual
Gaussian/root/distance arithmetic is unchanged; no new approximation or f32
conversion is introduced. The per-call distance HashMap is removed only from
the indexed reference path. Ordinary uncached scores keep their prior path.

This is CPU/WASM lookup reuse on native and WASM targets. It adds no GPU
dispatch, worker or public request control. Existing source identity,
voxel/minimum occupancy/moment-SIMD keys, invalidation and final three-score
cache policy remain. A final-score cache hit still bypasses this computation.
Initial automatic calls remain ordinary; matching repeated reference settings
can prepare the owned reference as before.

## Memory and preparation cost

The retained allowance now includes the Gaussian slot field and twice the
neighbour-vector capacity as slack, in addition to twice the HashMap payload
capacity and a 4,096-byte header allowance. The web preflight increases from
400 to 640 bytes per possible eligible voxel, followed by the actual 128 MiB
retained check. This deliberately makes preflight more conservative: some
large/low-occupancy settings that previously qualified now use the ordinary
fallback. Scores remain available with the same numerical policy. Neither
the allowance nor preflight is a total peak process-memory guarantee.

| Source | Hashed reference allowance, bytes | Indexed allowance, bytes | Retained reference voxels | Matched voxels |
| --- | ---: | ---: | ---: | ---: |
| PandaSet | 11,472,896 | 15,896,112 | 18,354 | 15,650 |
| Autoware | 45,879,296 | 60,697,848 | 60,109 | 56,008 |
| Tokyo | 11,472,896 | 16,367,856 | 20,538 | 20,122 |

The graph is built once with 27 reference lookups per reference voxel. Both
the graph and charged capacity remain unchanged during reuse. A temporary
distance array and matched-slot list are freed after each calculation. They
are not counted as idle retained payload. The first-use observations include
graph construction and can be slower: for example, Autoware combined first
observations are 655.9/705.2 and 656.2/707.5 ms, hashed/indexed. These are
individual observations, not cold medians or a demonstrated first-use gain.

## Measurements

Inputs are full unchanged reference/compared source-index splits from
[measured-source preparation](c2c-real-source.md): PandaSet 019
(531,908/531,908 points), Autoware planning (878,921/878,920) and Tokyo
Takanawadai (941,933/941,933). All load with `keepEvery:1`; no generated
performance geometry, displacement, noise or additional thinning is used.
Tokyo retains its earlier original-point preparation. These subsets are not
independent surveys or map-accuracy ground truth.

Windows/i7-9750H, 12 logical CPUs, 32 GiB RAM and Chromium 153.0.8010.12 run
archived hashed and current indexed production workers separately. Both have
the same owned reference Gaussian/root cache enabled. Each calculation loads
a new compared identity from the original full file, with final score cache
disabled. This also prevents whole bidirectional nearest-pair reuse in combined
evaluation. Each warm call computes zero reference roots and reuses one per
matched voxel; indexed `neighborsUsed` equals the matched voxel count.

An independent archived ordinary worker supplies an excluded oracle and
terminates before timing. One excluded first-use pair prepares both references
and the indexed graph; five excluded alternating warm-up pairs precede ten
alternating timed pairs per source/method/pass. The first pass orders sources
PandaSet → Autoware → Tokyo; the second reverses that order. Both passes use
the same source/script/build fingerprints. Parameters are `voxel:1`,
`minPoints:5`, `voxelSimd:false`, with combined threshold 0.5.

Standalone timing includes compared Gaussian construction, scalar cross roots,
Wasserstein distances, indexed or hashed SCS and returned scores. Combined
timing additionally includes fresh exact bidirectional C2C preparation/search,
joins, summaries and arrays. Loading, startup, export, validation and removal
are excluded. No builds, tests or repository scans run alongside timing.

| Source/pass | Method | Hashed median, ms | Indexed median, ms | Median paired shortening | Indexed wins |
| --- | --- | ---: | ---: | ---: | ---: |
| PandaSet initial | Voxel scores | 70.70 | 56.00 | +20.93% | 10/10 |
| PandaSet final | Voxel scores | 60.30 | 49.60 | +19.56% | 10/10 |
| Autoware initial | Voxel scores | 176.00 | 123.05 | +29.22% | 10/10 |
| Autoware final | Voxel scores | 175.55 | 119.40 | +34.09% | 10/10 |
| Tokyo initial | Voxel scores | 98.00 | 83.90 | +16.67% | 9/10 |
| Tokyo final | Voxel scores | 108.30 | 75.80 | +23.40% | 10/10 |
| PandaSet initial | Combined Map | 306.95 | 304.45 | +1.02% | 6/10 |
| PandaSet final | Combined Map | 341.70 | 351.65 | −2.08% | 4/10 |
| Autoware initial | Combined Map | 461.05 | 409.25 | +10.80% | 10/10 |
| Autoware final | Combined Map | 478.80 | 408.20 | +15.47% | 10/10 |
| Tokyo initial | Combined Map | 469.65 | 465.25 | +1.14% | 6/10 |
| Tokyo final | Combined Map | 437.55 | 449.30 | −1.91% | 4/10 |

Shortening is `median(100 * (1 - indexed_ms / hashed_ms))`, not a ratio of
marginal medians. All losses and spikes remain. Six positive standalone
medians and 59/60 wins support adopting the voxel path. Combined has 40/60
wins, with gains concentrated in Autoware; these measurements do not establish
a whole-Map benefit for PandaSet or Tokyo. CPU clocks, power, scheduling, other
applications, GC, HashMap seeds and WASM pages are uncontrolled. Construction,
memory layout, traversal and overlap with nearest work can also contribute.
No isolated SCS-kernel or native timing claim is made.

All 192 first-use/warm-up/timed pairs preserve the voxel reduction allowance:
384 outputs plus 12 excluded archived oracles. Combined evaluation preserves
all distance f32 bits, f64 statistics, non-voxel metrics and full scalar PLY
bytes, with unchanged workers and C2C input reservations. Input, manifest,
original measured PCD, thirteen implementation files, script and both artifact
sets are checked before/after and independently audited against the saved
measured source/build. No GPU model metadata is collected.

## Validation and reproduction

All 388 native tests with the parallel feature and 27 browser tests pass,
including native/WASM Clippy with warnings denied, formatting and release
WASM/TypeScript/Vite builds. New native regressions compare every neighbourhood
value bit against coordinate lookups with changing matched masks, negative
keys, signed zero, tiny values, infinity and NaN. They check score parity,
changed compared geometry, empty maps, root reuse, capacities and extreme-key
fallback. The archived-WASM test checks 720 reference computations, including
adjacent voxels, missing compared cells, source ownership, scales, occupancy,
moment-SIMD settings and exact capacity charging. The adopted normal path also
passes its 504 archived-fit cases. Generated fixtures test correctness only.

Existing browser checks cover budgets, the stricter preflight boundary,
invalidation, edits/removal, final score cache, combined metrics, pool failures
and recovery. Script syntax and nine invalid-option/repeat/overwrite/manifest
controls pass. Archive a preceding hashed production build first, then build
matching release WASM/app with `USE_VOXEL_NEIGHBORS = true` and the paired
cross-matrix flag off. Run from the repository root with a new output:

```powershell
node web/scripts/benchmark-voxel-neighbors.mjs --baseline-dist notes/voxel-neighbors-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-voxel-neighbors.json
```

The source flag guard cannot identify binary mode; a matching rebuild is
required. `--repeats` accepts 1..100, default ten. Existing reports cannot be
overwritten. Another checkout needs equivalent archived builds and locally
prepared measured inputs. The raw browser oracle needs optional
`VOXEL_NEIGHBORS_BASELINE` as an absolute snapshot root containing `dist` and
matching `web/src/wasm/ca_wasm.js`; it skips without that independent archive.

- PandaSet: [initial](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-retained-neighbors-initial.json), [final](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-retained-neighbors-final.json)
- Autoware: [initial](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-retained-neighbors-initial.json), [final](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-retained-neighbors-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-retained-neighbors-initial.json), [final](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-retained-neighbors-final.json)
- [Benchmark script](../web/scripts/benchmark-voxel-neighbors.mjs)
- [Archived-WASM test](../web/e2e/voxel-neighbors.spec.ts)
- [Implementation](../rust/crates/ca-core/src/map_quality.rs)
