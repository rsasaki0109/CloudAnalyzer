# Reusing bidirectional CPU Map quality preparation

Repeated Map quality evaluations now retain both nearest-distance directions'
exact query coordinates, reference trees, Morton order and output indices.
Every call still searches every point, computes voxel AWD/SCS, summarizes
forward distances, and recomputes f32 inlier metrics with the current threshold.
Changing the threshold, voxel size or minimum voxel occupancy reuses nearest
preparation and applies the new parameters to freshly computed results.

The later [cold CPU dispatch pipeline](cpu-preparation-pipeline.md) also
queues first-use searches before voxel computation, without waiting for
preparation acknowledgements. The measured cache comparisons below predate
that change; their warm paths and precision checks remain the same.

## Parallel work and ownership

The direction allocation, alternating lane order, scalar f64 search kernel,
100,000-query threshold, eight-lane cap and 256 MiB input/output copying
estimate are unchanged. Small directions remain serial. An eligible single
direction can use one pool lane; the reported count of one also covers the
fully serial path, so `cpuCached` identifies whether preparation is retained.

On first use, all coordinates and output indices are captured before the
evaluation yields. Each worker's existing `C2cPrepared` owns its queries and
tree. A lane searches as soon as its own preparation finishes. On reuse,
ready lanes receive searches synchronously before the main worker starts
the voxel kernel. This avoids a fulfilled-promise yield that would otherwise
postpone warm searches until after voxel computation. Distance searching and
voxel scoring can therefore overlap on repeat calls as well as first use.

`cpu-preparation.ts` manages both ordinary parallel C2C and Map quality.
Only one idle CPU preparation group is retained across these workflows.
Map quality keeps its two directions together; switching to C2C, reversing
the pair or changing the partition layout retires the group. Active retired
groups keep their independent snapshots until every reader has finished.
No source Cloud is read after the evaluation waits for worker results.

Coordinate changes and removals invalidate either source through the shared
CPU invalidation path. Generation checks detect worker loss while idle.
Preparation/search failures drain submitted lanes and suppress late output
writes, retire the affected group, and cause the next call to rebuild.
Cleanup releases only that group's state on live workers; it does not spawn
workers just to clean up state lost with a terminated instance. Collected
joining consumes indices inside the active lease, including when another
request has replaced or invalidated the group.

The internal `map-evaluate.cpuCache: false` control keeps rebuilding every
slice and does not evict the idle prepared group. `cpuCached`,
`cpuCacheReused` and `cpuCacheInputBytes` report retention, reuse and input
payload bytes. The historical `benchmark-map-quality.mjs` disables caching
for every workflow so its parallelization/summary controls remain independent.

## Retention cost

Input payload bytes count query/reference f64 XYZ and main-worker output
indices. The existing copying estimate also reserves 12 bytes per eligible
point for joined output. Neither value is a process-memory measurement or
a heap limit: derived tree nodes, reference slots, Morton ordering, spare
capacity, allocator overhead, source clouds, result arrays, GPU references
and overlapping retired groups are separate. Bidirectional preparation can
retain more data than a one-direction C2C group. Freed WASM allocations can
leave the instance's memory capacity at its high-water mark.

## Reproduce with measured-source inputs

Use the unchanged, disjoint even/odd source splits from
[real-source C2C preparation](c2c-real-source.md). They contain measured
coordinates but are subsets of one geometry, not independent surveys or
accuracy ground truth. No coordinate shifts, noise or synthetic points are
introduced, and the benchmark rejects thinning or manifest mismatches.

```powershell
cd web
npm run build
node scripts/benchmark-map-quality-cache.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --repeats 10 --output ../benchmarks/cpu/map-quality/real-source/windows-pandaset019-preparation-cache-new.json
```

Choose a fresh output filename for every run. Replace the source directory
with `c2c-real-autoware` or `c2c-real-tokyo` for the other measured geometries.
Only this production analysis worker and its existing pool run; there is no
viewer or second application pool. Do not run this project's builds/tests
concurrently with timing. Other applications and CPU clocks/power remain
uncontrolled.

Each report captures one fixed-order uncached/cached first-use pair, five
warm-up pairs and ten timed pairs with alternating call order. First-use
values include preparation and are not a paired performance claim. Warm
comparisons use identical partitions, worker counts, scalar searches,
progressive joining, adaptive summaries and voxel kernels; only CPU
preparation reuse differs. Wall times include RPCs, results, voxel work,
nearest searches and metrics, excluding file loading and parity validation.
The analysis worker's own duration is also recorded separately.

The prepared group remains resident during uncached control calls, so
allocation/locality effects are not isolated. Every pair checks f32 distance
bits, exact f64 summary fields and exact inlier/voxel counts. Existing voxel
HashMap reductions permit only `1e-12 * max(1, abs(a), abs(b))` roundoff for
AWD/SCS. Input and manifest hashes are checked before and after timing;
reports include production worker, pool worker, WASM and script hashes.

## Measured results

The final build used Chromium 153 and eight pool lanes on the same Windows
machine for all three full measured-source pairs. Each source was measured
twice, with five warm-up and ten alternating timed pairs per run. These are
within-run comparisons; absolute timings from separate runs should not be
subtracted to estimate the effect of an individual scheduling change.

| Measured source | Reference / compared points | Retained input payload bytes |
| --- | ---: | ---: |
| PandaSet scene 019 | 531,908 / 531,908 | 80,052,496 |
| Autoware planning survey | 878,921 / 878,920 | 103,744,548 |
| Tokyo MMS representative geometry | 941,933 / 941,933 | 110,153,944 |

Times below are medians of ten warm wall-time trials. Reduction is the
median of each trial's paired `1 - cachedMillis / uncachedMillis`.

| Source | Run/report | Uncached median | Cached median | Paired median reduction |
| --- | --- | ---: | ---: | ---: |
| PandaSet | [Final](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-preparation-cache-final.json) | 387.20 ms | 177.80 ms | 53.78% |
| PandaSet | [Repeat](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-preparation-cache-repeat.json) | 385.05 ms | 177.75 ms | 53.20% |
| Autoware | [Final](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-preparation-cache-final.json) | 648.85 ms | 383.15 ms | 41.24% |
| Autoware | [Repeat](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-preparation-cache-repeat.json) | 639.15 ms | 364.20 ms | 43.01% |
| Tokyo MMS | [Final](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-preparation-cache-final.json) | 562.20 ms | 290.35 ms | 48.26% |
| Tokyo MMS | [Repeat](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-preparation-cache-repeat.json) | 507.65 ms | 262.35 ms | 48.54% |

All 60 final timed pairs were faster with preparation reuse. They, the
30 warm-up pairs and six setup pairs preserved all distance bits, summary
fields, inlier metrics and voxel counts. Maximum timed AWD/SCS difference
was `1.3877787807814457e-15`, within the existing voxel reduction allowance.

First use still pays for planning, transfers and tree construction. The
fixed-order setup values were:

| Source/run | First uncached call | First cached call, including preparation |
| --- | ---: | ---: |
| PandaSet final | 464.40 ms | 441.30 ms |
| PandaSet repeat | 443.20 ms | 390.20 ms |
| Autoware final | 672.30 ms | 672.00 ms |
| Autoware repeat | 664.70 ms | 664.90 ms |
| Tokyo final | 581.60 ms | 566.80 ms |
| Tokyo repeat | 602.20 ms | 607.20 ms |

These are not alternating first-use comparisons and do not establish a
first-call speedup. The demonstrated benefit applies to repeated Map
quality evaluations, at the retention cost described above.

Two preliminary reports,
[PandaSet](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-preparation-cache.json)
and [Autoware](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-preparation-cache.json),
were captured before synchronous warm dispatch was added. They retain
their original worker hash and are excluded from the final table. Their
20 timed, ten warm-up and two setup pairs also passed parity. No
cross-run comparison of their absolute durations isolates the later
dispatch change; the scheduling test checks the intended overlap directly.

Final production artifacts for all six final/repeat reports:

- Main worker: `worker-BWgWiQHy.js`, SHA-256
  `2a97ea4b832114f7e779875d6a779daba28edd7a486a363ac36129a14810cee5`.
- Pool worker: `c2c-worker-D21THqES.js`, SHA-256
  `73124711329934dcf01fc659df1c490cba009c17e1cd1b8515883e4d69bb798c`.
- WASM: `ca_wasm_bg-jsjqg3N6.wasm`, SHA-256
  `b49e945097dd022c7a633af57118298b0c559fbcd29999101264e26b4012b479`.
- Benchmark script SHA-256:
  `4bdd69e9aef35d1ad842d3e35af7dceb2d9c54d817dc70cecaf159c0cbeb895b`.

## Validation

Six new browser tests exercise cold/warm reuse of both directions, unequal
input sizes, single-slice directions, collected/progressive joining,
changed threshold/voxel/minimum occupancy, shared C2C/Map quality eviction,
and uncached controls that preserve the idle group. They cover idle worker
loss, preparation/search/release failures, incomplete output, transforms,
held concurrent forward/reverse replies through source removal, and new
source identities. A controlled scheduling test verifies that warm searches
are posted before synchronous voxel work and that callbacks still await
their results. The existing copy-budget test exercises both cached and
uncached requests with modeled oversized buffers, without retaining trees
above the scheduling estimate.

Existing CPU C2C, median summaries, M3C2, bidirectional Map quality and
software/physical WebGPU suites check the shared pool and unchanged exact
arithmetic. The two Map quality/M3C2 UI checks exercise the production flow.
The Rust kernel and generated WASM do not change in this implementation.

The final build and 110 distinct browser checks passed: 36 CPU/Map quality
cases, 36 software WebGPU cases, 36 physical WebGPU cases and two UI cases.
The expanded default/uncached copy-budget case passed its final rerun.
Seven invalid benchmark CLI combinations and an overwrite attempt were
rejected without creating/changing reports. All eight reports' input and
manifest hashes remained unchanged; all six final reports matched the
final production artifacts. Script syntax, documentation links and public
filename/content checks passed.

```powershell
cd web
npm run build
$env:PW_PORT='4193'
npx playwright test e2e/map-quality-cache.spec.ts e2e/map-quality-parallel.spec.ts e2e/c2c-cache.spec.ts e2e/c2c-streaming.spec.ts e2e/c2c-simd.spec.ts e2e/m3c2-parallel.spec.ts e2e/distance-summary.spec.ts e2e/gpu-c2c.spec.ts --workers=1
npx playwright test e2e/app.spec.ts --grep 'map quality:|M3C2: a flat grid' --workers=1
$env:CA_GPU_HARDWARE='1'
npx playwright test e2e/gpu-c2c.spec.ts --workers=1
```
