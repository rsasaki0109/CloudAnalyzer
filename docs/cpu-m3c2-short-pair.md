# M3C2 short-neighbourhood guard experiment

The adopted [paired fresh M3C2 fitting](cpu-m3c2-pair.md) can stage a valid
partner's covariance twice when the other core has fewer than three hits.
A candidate checks both lengths first, avoiding that duplicate work. Tested
outputs match, but whole-call gains do not reproduce consistently. It remains
**off by default**, with `USE_M3C2_SHORT_PAIR_GUARD = false`.
The adopted paired eigensolver remains enabled.

## Computation

Previously, both covariance closures run before matching their optional
outputs. If one neighbourhood is short, the other can already have its
centroid/covariance computed; fallback then calls both individual fitting
kernels and computes the valid partner's moments again. The guard returns
the original individual fits directly if either length is below three.

Dense paired fits, unsafe-matrix fallback, odd tails, query/projection order,
normal validity tags, covariance arithmetic, eigen selection, orientation,
distance and LoD95 remain the same. The change adds no persistent memory,
worker, public request parameter or GPU dispatch. Prepared trees, core
partitioning, shared radius searches and the 64 MiB normal allowance keep
their policy. Matching normal-cache hits bypass fresh fitting and the guard.
All temporary buffers are those already used by the paired implementation.

## Measured comparison

Inputs are the full unchanged reference/compared files from
[measured-source preparation](c2c-real-source.md). No generated performance
coordinates, displacement, noise or additional input thinning is used. Tokyo
retains its preceding original-point preparation. These source-index subsets
are not independent surveys or change ground truth.

| Source | Reference / compared points | Original measured cores | Finite distances | Lanes |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 212,541 | 96,853 | 3 |
| Autoware planning | 878,921 / 878,920 | 696,739 | 160,487 | 2 |
| Tokyo Takanawadai | 941,933 / 941,933 | 299,755 | 281,763 | 2 |

Both builds already use the adopted paired WASM fits. Windows/i7-9750H,
12 logical CPUs, 32 GiB RAM and Chromium 153.0.8010.12 run archived unguarded
and current guarded production contexts separately. Normal radius is 0.4 m,
projection radius 0.25 m, half-depth 0.5 m and core spacing 0.25 m. All full
inputs load with `keepEvery:1`. Both builds use prepared trees, the resident
analysis lane, interleaved cores, shared searches and radius bounds disabled.

Fresh fitting disables normal reuse and projection fusion in both builds.
A separate cached-normal control captures exact normals and automatically
fuses projection on warm matching hits; it does not execute the guard.
An independent archived worker supplies a full-output/PLY oracle and
terminates before timing. Two excluded first observations construct each
context's trees and optionally capture normals. Five excluded alternating
warm-up pairs precede ten alternating timed pairs per source/case/pass.
The source order is PandaSet → Autoware → Tokyo initially, then reversed.
Both passes use the same source/script/build fingerprints.

Whole RPC includes core selection/planning, transfers, radius queries,
moments/eigen fitting when needed, projection, join, attributes/LOD indexing
and returned arrays. Loading, startup, first observations, warmups, export,
validation and removal are excluded. No builds, tests or repository scans
run alongside timing.

| Source/pass | Case | Unguarded median, ms | Guarded median, ms | Median paired shortening | Guarded wins |
| --- | --- | ---: | ---: | ---: | ---: |
| PandaSet initial | Fresh normals | 473.40 | 464.00 | +5.82% | 7/10 |
| PandaSet final | Fresh normals | 479.55 | 495.80 | −0.58% | 4/10 |
| Autoware initial | Fresh normals | 1,052.00 | 1,061.30 | +0.60% | 6/10 |
| Autoware final | Fresh normals | 1,342.35 | 1,363.85 | +1.04% | 6/10 |
| Tokyo initial | Fresh normals | 1,076.10 | 953.90 | +7.62% | 9/10 |
| Tokyo final | Fresh normals | 1,412.70 | 1,419.10 | −1.07% | 4/10 |
| PandaSet initial | Reused normals | 287.90 | 287.40 | +1.00% | 6/10 |
| PandaSet final | Reused normals | 346.25 | 333.95 | +1.73% | 7/10 |
| Autoware initial | Reused normals | 855.70 | 851.35 | +1.42% | 6/10 |
| Autoware final | Reused normals | 973.70 | 979.40 | −0.71% | 5/10 |
| Tokyo initial | Reused normals | 813.05 | 806.50 | +4.62% | 8/10 |
| Tokyo final | Reused normals | 953.20 | 1,009.70 | −4.65% | 1/10 |

Shortening is `median(100 * (1 - guarded_ms / unguarded_ms))`, not the ratio
of marginal medians. Fresh fitting has 36/60 wins: PandaSet and Tokyo gains
reverse in the second pass; Autoware's paired median changes remain small.
The cached control has 33/60 wins, although no guard runs there. Differences
can reflect code layout, scheduling, GC, clocks, power, other applications
or retained WASM pages. All spikes and losses remain, including the guarded
1,300.6 ms cached Tokyo observation. No isolated guard or native timing is
claimed. The measured repeatability does not support default adoption.

Every first-use/warm-up/timed pair matches all f32 distance/LoD95 bits,
significance values and full PLY bytes: 192 pairs, 384 outputs and six
excluded archived oracles. Normal payloads match at 11,902,296 / 39,017,384 /
16,786,280 bytes in cached cases and zero in fresh cases, within 64 MiB.
Input, manifest, original measured PCD, fifteen implementation files, script
and both production artifact sets are checked before/after and independently
audited against the saved measured source/build. No GPU model identifiers
are collected. The measured enabled build is archived separately from the
final default-off build; these candidate percentages do not describe a newly
enabled optimization in the final production build.

## Validation and reproduction

The candidate passes 390 native tests with the parallel feature, native/WASM
Clippy with warnings denied, formatting and release WASM/TypeScript/Vite builds.
The new focused regression compares 144 short/full neighbourhood combinations
in both orders across scales and offsets, checking optional-normal f64 bits.
Existing paired tests cover complete distances, LoD95, shared/bounds/fused
controls, undefined cores and validity tags. Generated fixtures test
correctness only.

Twenty-eight browser tests pass. An archived adopted-pair WASM oracle checks
420 count/geometry/radius/bounds/shared conditions with raw f64 results,
projection controls and normal capture/reuse. The preceding individual-fit
oracle also passes, alongside adopted normal and voxel-neighbour regressions,
budgets, layouts, source edits/removal and worker failure recovery. Script
syntax and nine invalid-option/repeat/overwrite/manifest controls pass.

After restoring the guard to `false`, 390 native tests, native/WASM Clippy,
formatting and release WASM/TypeScript/Vite builds pass again. Twenty-seven
browser regressions pass on that final build, including the independent
individual-fit M3C2, normal and voxel-neighbour oracles. The new guard-specific
browser regression above belongs to the separately archived enabled candidate.
The final default-off source rejects guard benchmark runs before starting a
browser or writing a report. The final artifacts are rebuilt and are not
byte-identical to the preceding adopted-pair archive; result parity is verified
by the browser oracles. All six candidate reports pass a separate fingerprint,
input, cache/layout and summary audit against the measured enabled snapshot.

Archive an unguarded adopted-pair production build first. Enable
`USE_M3C2_SHORT_PAIR_GUARD = true`, retain `USE_M3C2_PAIR_EIGEN = true`, and
rebuild matching release WASM/app. Run from the repository root with full
measured inputs and a new output:

```powershell
node web/scripts/benchmark-m3c2-short-pair.mjs --baseline-dist notes/m3c2-short-pair-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-m3c2-short-pair.json
```

The source guard cannot identify binary mode; a matching rebuild is required.
`--repeats` accepts 1..100, default ten. Existing reports cannot be overwritten.
Another checkout needs equivalent archived artifacts and locally prepared
measured inputs. The raw browser oracle needs optional
`M3C2_SHORT_PAIR_BASELINE` as an absolute snapshot root containing `dist` and
matching `web/src/wasm/ca_wasm.js`; it skips without that independent archive.
Use an enabled matching candidate build to exercise the guard.

- PandaSet: [initial](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-short-pair-initial.json), [final](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-short-pair-final.json)
- Autoware: [initial](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-short-pair-initial.json), [final](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-short-pair-final.json)
- Tokyo: [initial](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-short-pair-initial.json), [final](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-short-pair-final.json)
- [Benchmark script](../web/scripts/benchmark-m3c2-short-pair.mjs)
- [Archived-WASM regression](../web/e2e/m3c2-short-pair.spec.ts)
- [Implementation](../rust/crates/ca-core/src/m3c2.rs)
