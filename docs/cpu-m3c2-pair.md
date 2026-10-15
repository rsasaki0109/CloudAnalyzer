# Paired fresh normal fits for prepared M3C2

Prepared M3C2 now fits two adjacent core normals with independent WASM
`f64x2` Jacobi lanes when fresh fitting is required. The default is enabled
with `USE_M3C2_PAIR_EIGEN = true`. Across two passes on three unchanged
measured sources, fresh-fitting whole-M3C2 paired reductions are
**1.42–10.84%**, with **46/60 timed pairs** faster and identical tested outputs.

The [exact normal cache](cpu-m3c2-normals.md) remains the main optimization
when fitting settings match: a cache hit uses its existing projection path.
The new pairing runs only on fresh prepared fitting on SIMD-enabled WASM.
Native serial/Rayon and ordinary borrowed/copying drivers retain their
individual fits. The adopted ordinary
[paired normals](cpu-normal-pair.md) and
[voxel-neighbour reuse](cpu-voxel-neighbors.md) remain enabled.

## Computation and exactness

The paired scratch keeps both cores' normal-hit lists and reference projection
lists from the existing [shared radius search](cpu-m3c2-shared-search.md).
Reference queries run for core one then core two, retaining each query's exact
hit sequence and inclusive boundaries. Centroid division/addition and covariance
accumulation retain the historical f64 operation order for each core.

The existing two-lane solver keeps each matrix's rotations, skip decisions,
independent convergence state and maximum 50 sweeps. Eigenvalue selection,
collinearity rejection and upward orientation retain their scalar arithmetic.
Too few hits in either core, nonfinite matrix entries or entries above `1e150`
fall back to the preceding independent fit for both cores. This can recompute
moments on a fallback. Odd tails use the individual search/fit/project path.
The subsequent [short-neighbourhood guard experiment](cpu-m3c2-short-pair.md)
avoids that staging when either core is short, but remains off after mixed
whole-call measurements.
The later [retained core selection](cpu-m3c2-core-selection.md) avoids repeated
voxel membership work and is adopted alongside these paired fits.
Failed fitting remains distinct from a fitted nonfinite normal: the optional
validity tag is preserved when capturing or reusing normals.

After fitting, projection proceeds core one then core two. Shared reference
projection reads its saved list; compared projection keeps the existing
list/fused control. Separate-search mode keeps its preceding projection calls.
Cylindrical filtering, along/radial arithmetic, sample variance, LoD95,
significance, output order and f32 result conversion remain unchanged.
The paired shared path also computes the invariant projection reach once per
pair instead of once per core. Grouping, allocation and scheduling can
contribute to timing differences alongside SIMD.

One additional normal-hit vector and one additional projection-hit vector are
transient per active prepared lane: four lists instead of two. They retain
capacity within a slice and are freed on return. WASM hit payload is four
bytes per slot, before vector capacity/allocator slack. No new persistent
cache or public request field is introduced. The 64 MiB pair-wide normal
allowance, exact normal/core keys, input-tree scheduling, lane count and core
partitioning keep their policies. The existing 256 MiB input-storage estimate
excludes these transient lists and other sources/results/WASM pages; it is
not a peak process-memory bound.

## Measurements

Inputs are full unchanged reference/compared source-index splits from
[measured-source preparation](c2c-real-source.md). No generated performance
geometry, transformed coordinates, noise or further input thinning is used.
Tokyo retains its earlier original-point preparation. These subsets are not
independent surveys or change ground truth. Full files load with `keepEvery:1`;
M3C2 selects original measured representatives at core spacing 0.25 m.

| Source | Reference / compared points | Cores | Finite distances | Lanes | Cached-normal payload, bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 212,541 | 96,853 | 3 | 11,902,296 |
| Autoware planning | 878,921 / 878,920 | 696,739 | 160,487 | 2 | 39,017,384 |
| Tokyo Takanawadai | 941,933 / 941,933 | 299,755 | 281,763 | 2 | 16,786,280 |

Windows/i7-9750H, 12 logical CPUs, 32 GiB RAM and Chromium 153.0.8010.12 run
archived individual and current paired production builds in separate workers.
Both use prepared trees, interleaved cores, the resident analysis lane, shared
reference searches and identical lane counts. Parameters are normal radius
0.4 m, projection radius 0.25 m, half-depth 0.5 m and radius bounds disabled.

Fresh-fitting measurements disable normal reuse and projection fusion in both
builds. A separate cached-normal case captures exact normals in each build
and automatically fuses both projections on warm cache hits. No paired fitting
runs in that cached case. An independent archived ordinary prepared worker
supplies an excluded full-array/PLY oracle, then terminates before timing.

One excluded first-use pair constructs each context's trees and optionally
captures normals. Five excluded alternating warm-up pairs precede ten
alternating timed pairs per source/case/pass. Both passes use identical
source/script/build fingerprints. Source order is PandaSet → Autoware → Tokyo,
then reversed. Every warm call requires tree reuse; normal hit/byte/fusion
states must match the selected case and each other.

Whole RPC includes core selection/planning, bindings/transfers, radius queries,
moments/eigen fitting when needed, projection, joining, attributes/LOD indexing
and returned arrays. Loading, startup, first observations, warmups, exports,
validation and removals are excluded. No builds, tests or repository scans run
alongside timing.

| Source/pass | Case | Individual median, ms | Paired-build median, ms | Median paired shortening | Faster pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| PandaSet initial | Fresh normals | 385.75 | 357.45 | +4.31% | 7/10 |
| PandaSet final | Fresh normals | 339.80 | 335.50 | +1.42% | 6/10 |
| Autoware initial | Fresh normals | 1,014.05 | 924.10 | +8.85% | 10/10 |
| Autoware final | Fresh normals | 1,014.20 | 980.90 | +3.84% | 6/10 |
| Tokyo initial | Fresh normals | 971.10 | 828.55 | +10.84% | 9/10 |
| Tokyo final | Fresh normals | 868.55 | 845.00 | +4.11% | 8/10 |
| PandaSet initial | Reused normals | 276.00 | 245.50 | +9.19% | 7/10 |
| PandaSet final | Reused normals | 260.15 | 257.30 | −1.47% | 4/10 |
| Autoware initial | Reused normals | 732.65 | 716.70 | +2.23% | 5/10 |
| Autoware final | Reused normals | 938.65 | 946.60 | +3.60% | 7/10 |
| Tokyo initial | Reused normals | 697.90 | 714.80 | +3.32% | 6/10 |
| Tokyo final | Reused normals | 679.45 | 656.50 | +5.77% | 9/10 |

Shortening is `median(100 * (1 - paired_build_ms / individual_ms))`, not a
ratio of marginal medians. All spikes and losses remain, including a
379.0/641.2 ms PandaSet fresh timed pair and a 2,544.7/1,135.5 ms Autoware
fresh warm-up pair. Clocks, power, other applications, scheduling, GC and
retained WASM pages are uncontrolled. Six positive fresh medians and 46/60
wins support enabling the prepared fresh path in these workloads, with the
smallest final gain modest relative to the variation.

The cached case has 38/60 faster pairs but **does not run the new paired
eigensolve**. Those differences can reflect build layout, scheduling,
allocation or uncontrolled state; they are not evidence of a SIMD fitting
gain. Neither case isolates eigen instruction time or establishes a native
speedup. This implementation accelerates CPU/WASM work and adds no GPU dispatch.

All 192 first-use/warm-up/timed pairs preserve every returned distance/LoD95
f32 bit, significance value and full PLY byte: 384 outputs plus six excluded
archived oracles. Normal-cache payloads match across builds and stay under
64 MiB; fresh cases retain none. Inputs, manifests, original measured PCDs,
fifteen implementation files, script and both production artifact sets are
checked before/after and independently audited against saved measured sources
and builds. No GPU model identifiers are collected.

## Validation and reproduction

All 389 native tests with the parallel feature and 27 browser tests pass,
including native/WASM Clippy with warnings denied, formatting and release
WASM/TypeScript/Vite builds. The new native regression compares complete f64
distance/LoD/optional-normal bits across 192 scale/geometry/bounds/shared/fused
conditions, including undefined and nonfinite cores, collinearity, duplicates
and unsafe-matrix fallback. The existing paired-solver regression covers all
eigenvalue/eigenvector bits and independent stopping.

The archived-WASM regression checks 420 core-count/geometry/radius/bounds/shared
conditions, each with individual/list or fused projections and normal capture/
reuse. It compares raw f64 result pairs and cache states, including empty/odd
core sets, georeferenced/tiny/large data and nonfinite queries. Existing tests
cover serial/copying/prepared paths, core tails and layouts, attributes,
budgets, source edits/removal, failures and concurrent leases. Adopted normal
and voxel-neighbour regressions also pass. Generated fixtures test correctness
only. Script syntax and nine invalid-option/repeat/overwrite/manifest controls
pass.

Archive a preceding individual production build, then build matching release
WASM/app with `USE_M3C2_PAIR_EIGEN = true`. Run from the repository root with
full measured inputs and a new output:

```powershell
node web/scripts/benchmark-m3c2-pair.mjs --baseline-dist notes/m3c2-pair-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-m3c2-pair.json
```

The source guard cannot identify binary mode; a matching rebuild is required.
`--repeats` accepts 1..100, default ten. Reports cannot be overwritten. Another
checkout needs equivalent archived artifacts and local measured preparation.
The raw browser oracle needs optional `M3C2_PAIR_BASELINE` as an absolute
snapshot root containing `dist` and matching `web/src/wasm/ca_wasm.js`; it skips
without that independent archive.

- PandaSet: [initial](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-paired-eigen-initial.json), [final](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-paired-eigen-final.json)
- Autoware: [initial](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-paired-eigen-initial.json), [final](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-paired-eigen-final.json)
- Tokyo: [initial](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-paired-eigen-initial.json), [final](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-paired-eigen-final.json)
- [Benchmark script](../web/scripts/benchmark-m3c2-pair.mjs)
- [Archived-WASM regression](../web/e2e/m3c2-pair.spec.ts)
- [Core paired fitting](../rust/crates/ca-core/src/m3c2.rs)
- [Prepared driver](../rust/crates/ca-wasm/src/m3c2.rs)
