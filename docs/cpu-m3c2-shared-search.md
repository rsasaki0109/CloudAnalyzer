# Shared reference radius searches for M3C2

The default browser CPU/WASM M3C2 path now shares the reference normal-radius
and reference projection-sphere searches. Serial, copying-worker and prepared
paths use it; the native serial and Rayon drivers do too. This reduces work
inside each fresh calculation, alongside the existing core-point parallelism
and [exact tree reuse](cpu-m3c2-preparation.md).

`sharedSearch: false` selects the historical separate-query control. The
request flag defaults to true and is reported as `M3c2Output.sharedSearch`.
The bounded [normal cache](cpu-m3c2-normals.md) now has an automatic repeated-fit
policy; the comparisons below explicitly disable it to measure fresh fitting.
A normal-cache hit skips fitting and uses the existing two projection queries;
a miss can use shared reference traversal while capturing fresh normals.

## Search order and temporary storage

`KdTree.within_two_slots_with_bounds` traverses the immutable tree once for
two independent radii. Each radius retains its own split-plane pruning. A
radius pruned from a subtree stays disabled there; the final implementation
then delegates to the existing single-radius traversal for the survivor.
Each visited leaf point's query-distance arithmetic runs once, and its slot
is appended separately to each accepting output. Near/far order, inclusive
boundary comparisons, and each output's unsorted hit sequence match two
independent searches. This also supports the optional combined-radius bounds.

M3C2 fits its normal from the first output and projects reference points from
the second. The compared-tree search follows as before. Centroid division,
covariance, PCA, projection order, sample variance and LoD95 arithmetic retain
the existing f64 operations and point order. Failed fits keep their validity
tag distinct from non-finite fitted normals. Shared/separate search and
projection parameters need no additional normal-cache key: tests confirm the
same fitted-normal bits across those controls.

One additional `Vec<usize>` holds temporary hit slots per active lane. Both
vectors reuse capacity within the slice and are dropped when it finishes;
there is no additional persistent cache. WASM slots use four bytes each, plus
vector capacity/allocator overhead. The existing 256 MiB input-tree scheduling
estimate excludes these temporary vectors, as it excluded the earlier scratch
vector, along with sources, cores, results, retained WASM pages and other
workflows. It is an input-storage estimate, not a total heap cap.

Sharing queries can inspect projection-sphere neighbours even when normal
fitting subsequently fails. Sparse inputs or very large projection ranges can
therefore incur extra work; the explicit separate-search control remains
available for comparisons. The measured results below determine the default
for the tested browser workloads, rather than guaranteeing every parameter set.

## Measured inputs and protocol

The reports use supplied measured point clouds, preserving original f64
coordinates and selecting original measured core representatives. No generated
coordinates, offsets or noise are added. The 2,000,000-point loading budget
loads every supplied input (`keepEvery: 1`). The full source provenance,
file hashes, original/loaded counts, and attribution are embedded in each
report. Even/odd source-index subsets are not independent surveys or change
measurement ground truth. Tokyo's supplied inputs retain original point
representatives from its earlier 0.1 m voxel selection.

| Measured input | Reference / compared points | Cores / finite distances | CPU lanes |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 212,541 / 96,853 | 3 |
| Autoware planning | 878,921 / 878,920 | 696,739 / 160,487 | 2 |
| Tokyo Takanawadai | 941,933 / 941,933 | 299,755 / 281,763 | 2 |

Parameters: normal radius 0.4 m, projection radius 0.25 m, half-depth 0.5 m,
core spacing 0.25 m. Radius bounds and normal reuse stay disabled. Each mode
uses a fresh reference identity for a cold prepared-tree call, immediately
followed by a warm-tree call that still fits fresh normals and recomputes both
projections. Separate/shared block order alternates; both use the same CPU
lane count and source/core selection. Five excluded warm-up trials precede
ten timed trials per report. Reports refuse to overwrite an existing file.

Full RPC timing includes core selection/attributes, tree construction/export/
validated import when cold, binding copies, fitting, projection, joining and
output attributes/LOD indexing. Loading, removal, worker-pool startup, serial
baseline, validation and exports are outside timers. Source, manifest, script
and selected production-artifact hashes are checked before and after timing.
No builds or tests ran alongside the benchmark. Other apps, CPU clocks, power,
GC and retained WASM memory are uncontrolled; every regression and outlier is
retained. Timings are local Windows/Chromium measurements.

The initial prototype carried both query controls throughout traversal; the
final build adds delegation after either query is pruned and enables sharing
by default. Each table comparison is paired within its own report/build.
Median reduction is `median(100 * (1 - shared / separate))`, rather than the
ratio of the two marginal medians. All times below are milliseconds.

## Cold calls, including tree construction and transfer

| Report | Separate median | Shared median | Paired median reduction | Faster trials |
| --- | ---: | ---: | ---: | ---: |
| [PandaSet 019 final](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-shared-search-final.json) | 709.50 | 656.20 | 8.54% | 9/10 |
| [PandaSet 019 initial](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-shared-search-initial.json) | 747.30 | 704.65 | 4.28% | 9/10 |
| [Autoware planning final](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-shared-search-final.json) | 1,641.20 | 1,555.45 | 6.02% | 8/10 |
| [Autoware planning initial](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-shared-search-initial.json) | 1,569.35 | 1,483.30 | 5.02% | 9/10 |
| [Tokyo Takanawadai final](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-shared-search-final.json) | 1,611.40 | 1,487.65 | 9.44% | 8/10 |
| [Tokyo Takanawadai initial](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-shared-search-initial.json) | 1,632.15 | 1,549.35 | 3.23% | 7/10 |

## Warm trees, with freshly fitted normals

| Report | Separate median | Shared median | Paired median reduction | Faster trials |
| --- | ---: | ---: | ---: | ---: |
| [PandaSet 019 final](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-shared-search-final.json) | 453.00 | 430.85 | 6.25% | 7/10 |
| [PandaSet 019 initial](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-shared-search-initial.json) | 493.65 | 469.60 | 4.03% | 7/10 |
| [Autoware planning final](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-shared-search-final.json) | 1,325.35 | 1,174.75 | 9.93% | 8/10 |
| [Autoware planning initial](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-shared-search-initial.json) | 1,275.95 | 1,109.55 | 10.22% | 9/10 |
| [Tokyo Takanawadai final](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-shared-search-final.json) | 1,194.35 | 1,011.00 | 9.99% | 10/10 |
| [Tokyo Takanawadai initial](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-shared-search-initial.json) | 1,214.90 | 1,094.55 | 9.03% | 10/10 |

Across the six reports, sharing is faster in 50/60 timed cold comparisons and
51/60 timed warm-tree comparisons. Both medians improve in every report.
The final build's paired median reductions are 6.02–9.44% cold and 6.25–9.99%
with warm trees. These are additional gains over the prepared-tree path with
fresh normals, rather than comparisons against rebuilding every worker tree.

## Validation

All 360 serial-baseline comparisons (four outputs per trial, including the
excluded warm-ups) match f32 distance, LoD95 and significance bits and the
complete exported PLY bytes. Real-source performance checks compare exported
results; raw f64 arithmetic/order checks are separate correctness fixtures.

Native dual-query tests compare both unsorted slot sequences with independent
queries across leaf sizes, duplicate/signed-zero coordinates, radius boundaries
and adjacent floats, negative/NaN/infinite radii, subnormal underflow, overflow,
and non-finite coordinates/queries. M3C2 tests compare fitted-normal validity
and every normal/distance/LoD95 f64 bit against the separate-search path,
including normal radii larger than, smaller than or equal to projection reach.

The full native run passed 319 core tests, nine integration checks and all 26
WASM-crate tests. After the final delegation/default changes, dual-query and
M3C2 checks, the M3C2 integration check, and all 26 WASM-crate tests passed again.
Native and WASM SIMD Clippy with warnings denied, Rustfmt, TypeScript/Vite,
and diff checks pass. Eight benchmark CLI guards reject invalid arguments or
existing outputs while preserving an existing report.

The final production build passed 25 browser checks: 19 M3C2 checks (including
30 raw-f64 cases, serial/copying/prepared/default/cached paths, radius changes,
failure recovery, concurrent requests, source edits/removal and lane budgets),
four C2C/Map-quality SIMD checks, and two physical-WebGPU regression checks.

## Reproduce

From `web/`, build and validate before timing. Run benchmarks sequentially,
with a new output path each time. Earlier M3C2 benchmark scripts explicitly
keep `sharedSearch: false` to isolate their historical comparisons.

```powershell
$env:PATH = (Join-Path $env:USERPROFILE '.cargo\bin') + ';' + $env:PATH
$env:RUSTFLAGS = '-C target-feature=+simd128'
npm run wasm
npm run build
$env:PW_PORT = '4193'
npx playwright test e2e/m3c2-radius.spec.ts e2e/m3c2-parallel.spec.ts e2e/m3c2-preparation.spec.ts e2e/c2c-simd.spec.ts e2e/map-quality-simd.spec.ts --workers=2
$env:CA_GPU_HARDWARE = '1'
npx playwright test e2e/gpu-c2c.spec.ts --grep 'WASM SIMD leaf verification|WebGPU plus CPU workers' --workers=1
node scripts/benchmark-m3c2-shared-search.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../benchmarks/cpu/m3c2/real-source/windows-pandaset019-shared-search-reproduce.json
node scripts/benchmark-m3c2-shared-search.mjs --reference ../notes/c2c-real-autoware/reference.ply --compared ../notes/c2c-real-autoware/compared.ply --input-manifest ../notes/c2c-real-autoware/inputs.json --output ../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-shared-search-reproduce.json
node scripts/benchmark-m3c2-shared-search.mjs --reference ../notes/c2c-real-tokyo/reference.ply --compared ../notes/c2c-real-tokyo/compared.ply --input-manifest ../notes/c2c-real-tokyo/inputs.json --output ../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-shared-search-reproduce.json
```

The script requires measured files; it has no synthetic-input fallback.

## Build fingerprints

[Benchmark script](../web/scripts/benchmark-m3c2-shared-search.mjs) SHA-256: `2823e7ac422fc7cbea91b8d67ef0425a1bd0bf27c3257bd59cc3c1600061785d`.

| Revision | Selected artifact | SHA-256 |
| --- | --- | --- |
| initial | `c2c-worker-BsEbNrtr.js` | `4d437a446e7be9ae8b086bb4a8cd9c85bc3ca823396579e62056265b7f83f1d0` |
| initial | `ca_wasm_bg-S_03JlXt.wasm` | `9c7fc039dc6a942278766162e1548307eba37d09138be658f2543ba09c20aa2a` |
| initial | `worker-BRg4LcvV.js` | `6a8dd51d4d66ea88aa448acf5ec6f88468c1b3c46114136b3373bf5566ea34d1` |
| final | `c2c-worker-rsPnHxAN.js` | `8908ab3e34f6e08b70a46b0a57a8ec19e474f37da99374cc9dcc7ff607f170fb` |
| final | `ca_wasm_bg-CA2cuJk2.wasm` | `1a7a9b902d3b2004670f45ee369704535528112f2d0d776b6fcba916cff3be7d` |
| final | `worker-kyASiyAd.js` | `370b2e117ff184fb022852f4cec097bf524ab61798a3cad41759d60a3b1523f1` |

| Report | SHA-256 |
| --- | --- |
| [windows-pandaset019-shared-search-final.json](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-shared-search-final.json) | `2369c992bb38e962e984862d23cb9dce14b36b112ea5e2ab0ee9ef661967d9f9` |
| [windows-pandaset019-shared-search-initial.json](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-shared-search-initial.json) | `aee5bb002a26a8ce046746052984632981272edf6492c3931e2bbeed23ec8142` |
| [windows-autoware-planning-shared-search-final.json](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-shared-search-final.json) | `cb6473dd03160fc2e02fd84a920b13094bd838dd555a125780cc5f11fdb0d7e9` |
| [windows-autoware-planning-shared-search-initial.json](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-shared-search-initial.json) | `0d361b8311e4ecd31c4c8f1407f4189f19f1504c27ef135a78b073161a204b06` |
| [windows-tokyo-takanawadai-shared-search-final.json](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-shared-search-final.json) | `6511aa34e486ee4cf975a969f9d86ee32fb1a0e8a98220e16e18b0aaf2e02983` |
| [windows-tokyo-takanawadai-shared-search-initial.json](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-shared-search-initial.json) | `6266066681a76fa3d5a0b2c360267404407cf20ac6d1a69e34ad09cd44d582c6` |
