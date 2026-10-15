# Queueing cold CPU searches behind preparation

Map quality now queues each cold nearest search immediately behind that
lane's preparation message. The worker can construct its exact tree and
search without waiting for another turn of the main analysis worker.
This lets first-use nearest searches overlap synchronous voxel AWD/SCS
computation, just as warm searches already did.

Ordinary CPU C2C keeps acknowledgement-gated cold dispatch by default.
Its main worker has no independent voxel phase to hide the wait, and the
measured-source comparisons did not show a consistent C2C benefit.
Both paths still retain exact preparation and reuse it on repeat calls.

The later [optional voxel SIMD evaluation](cpu-map-quality-simd.md) changes
independent moment channels only. Scalar moments remain the default; the
dispatch measurements below keep their original artifact hashes.

## Ordering, failures and memory

Both messages go to the same existing worker generation. Messages on that
worker are ordered, all requests await the same WASM initialization, and
`cloud-prepare` installs `C2cPrepared` synchronously before returning.
`cloud-prepared` therefore sees the preceding preparation even when both
messages arrive before WASM initialization completes. Queuing does not
merge requests or cache the initial distance result: every call executes
the existing fresh-seed f64 search.

`runOnGeneration` posts only to the exact live instance that received the
coordinates. A stopped worker cannot cause the search to create a replacement
that has never received its preparation. The next evaluation rebuilds the
group through the usual generation checks.

The job observes both preparation and search responses. Either failure
immediately closes result callbacks, and both submitted messages are drained
before the affected preparation is retired/released. An unsuccessful
preparation acknowledgement rejects the job even if the queued search has
already returned valid values. Short results also close callbacks immediately.
Other lanes cannot publish a partial result or write after failure.

The one-idle-group policy, active leases, source mutation/removal handling,
partition thresholds, eight-lane limit, exact arithmetic, summaries and
Map quality copy estimate are unchanged. There are no extra coordinate
transfers or retained result arrays. Per-call pending search results remain
temporary; the retained input payload is identical between dispatch modes.
Tree/order/allocator storage and WASM high-water capacity remain separate
from reported input bytes, as described in
[CPU preparation reuse](cpu-c2c-preparation.md) and
[bidirectional Map quality preparation](cpu-map-quality-preparation.md).

Internal controls and output metadata:

- `map-evaluate.cpuPipeline` defaults to true; false waits for a cold lane's
  acknowledgement before posting its search.
- Ordinary CPU `c2c.cpuPipeline` defaults to false; true selects queued cold
  dispatch for comparisons.
- `cpuPipelined` reports the dispatch policy only when CPU preparation is
  retained. Serial/uncached CPU computations report false.
- Warm preparations post immediately in either mode, so the comparison
  changes cold scheduling. SIMD/scalar and collected/progressive controls
  remain independent, with fresh output on every call.

## Measured-source comparisons

```powershell
cd web
npm run build
node scripts/benchmark-cpu-pipeline.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --repeats 10 --output ../benchmarks/cpu/map-quality/real-source/windows-pandaset019-cpu-preparation-pipeline-new.json
```

The default `--method both` compares C2C and Map quality separately;
`--method c2c` or `--method map-quality` limits the run. The other full
measured-source directories are `c2c-real-autoware` and `c2c-real-tokyo`.
Choose a new report filename: existing reports are rejected.

The source files/provenance are the unchanged
[real-source even/odd splits](c2c-real-source.md), with no point thinning,
noise, coordinate shifts or generated performance fixtures. These are
two subsets of one measured geometry, not independent surveys or accuracy
ground truth. Input and manifest hashes/counts are checked.

Each workflow warms five pairs, then measures ten sequential pairs with
waiting/pipelined order alternating. Before each mode, the previous reference
is removed and identical reference bytes are loaded under a fresh identity;
the compared cloud stays unchanged. Both timed controls therefore start with
no prepared pair and include the same planning, transfers, tree construction,
exact search, joins, summaries and metrics. Source removal/reloading and
old-state release submission occur before timing. Lane messages preserve
release/preparation/search order. Both controls use identical scalar leaves,
partition counts and progressive joining; only cold search dispatch differs.

The worker pool is initialized/warmed before measurements. These are cold
source-pair preparations in an already warmed pool, not browser/process
startup times. Each cold call is followed by a separate fresh-distance warm
call, whose timing and parity are recorded without claiming a warm scheduling
gain. Worker durations are recorded separately from RPC-inclusive wall times.

Every cold and warm comparison requires identical displayed f32 distance
bits and f64 summary fields. Inlier metrics and voxel counts match exactly;
AWD/SCS permit only the existing `1e-12 * max(1, abs(a), abs(b))` HashMap
reduction roundoff. Reports capture source/manifest and production worker,
pool worker, WASM and script hashes. This project's builds/tests do not run
concurrently with timing; other applications, clocks/power and GC remain
uncontrolled. Compare within-run paired reductions, not absolute durations
from separate reports.

The full reference/compared counts are 531,908/531,908 for PandaSet scene
019, 878,921/878,920 for Autoware Planning and 941,933/941,933 for Tokyo
Takanawadai. Map quality uses eight lanes for each source. Waiting and
queued controls retain the same input payload: 80,052,496, 103,744,548 and
110,153,944 bytes respectively, excluding derived tree/order storage.

Map quality wall-time comparisons:

| Source/run | Waiting median | Queued median | Median paired reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| [PandaSet preliminary](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-cpu-preparation-pipeline.json) | 407.50 ms | 387.05 ms | 5.01% | 9/10 |
| [PandaSet final](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-cpu-preparation-pipeline-final.json) | 474.40 ms | 465.55 ms | 8.35% | 6/10 |
| [Autoware preliminary](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-cpu-preparation-pipeline.json) | 680.70 ms | 648.80 ms | 5.91% | 9/10 |
| [Autoware final](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-cpu-preparation-pipeline-final.json) | 890.25 ms | 844.40 ms | 5.31% | 9/10 |
| [Tokyo preliminary](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-cpu-preparation-pipeline.json) | 559.45 ms | 515.30 ms | 5.90% | 9/10 |
| [Tokyo final](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-cpu-preparation-pipeline-final.json) | 858.60 ms | 806.30 ms | 10.68% | 8/10 |

The paired reduction is `median(1 - queued_i / waiting_i)` across the ten
alternating timed pairs. It is not computed from the two marginal medians.
For example, PandaSet final has a 1.87% difference between the marginal
medians, an 8.35% median paired reduction and only six faster pairs; its
variation remains material. Map quality is faster in 23/30 final pairs and
50/60 pairs across all six reports. All six within-run median paired
reductions are positive, supporting queued dispatch as the Map quality
default. This does not establish a gain for every call or another machine.

Ordinary C2C has no independent voxel computation to overlap:

| Source/run | Waiting median | Queued median | Median paired reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet preliminary | 222.60 ms | 222.05 ms | 0.89% | 5/10 |
| PandaSet final | 288.05 ms | 289.95 ms | -5.79% | 4/10 |
| Autoware preliminary | 323.95 ms | 326.15 ms | 1.96% | 6/10 |
| Autoware final | 348.40 ms | 352.25 ms | -2.54% | 3/10 |
| Tokyo preliminary | 285.75 ms | 293.20 ms | 2.65% | 6/10 |
| Tokyo final | 385.25 ms | 370.70 ms | 1.47% | 6/10 |

Those C2C cases are in the same linked reports. Only 30/60 C2C pairs are
faster, with two negative final paired medians. C2C therefore keeps its
existing waiting default; queued cold dispatch stays an internal comparison
option. Warm searches already dispatch immediately in both policies, so
their recorded times do not isolate a scheduling improvement.

All 120 timed cold and 120 timed warm pairs, plus 60 cold and 60 warm
warm-up pairs, preserve every displayed f32 distance bit and f64 summary
field. Map quality inlier metrics and voxel counts match exactly. Maximum
AWD/SCS difference across the six reports is `1.4155343563970746e-15`,
within the existing voxel reduction allowance.

Final production artifacts for the three final reports:

- Main worker: `worker-DMD03PLW.js`, SHA-256
  `561174195a7ad6504bb5590046059729436c1d59689daa3d26f1853cf47052c9`.
- Pool worker: `c2c-worker-D21THqES.js`, SHA-256
  `73124711329934dcf01fc659df1c490cba009c17e1cd1b8515883e4d69bb798c`.
- WASM: `ca_wasm_bg-jsjqg3N6.wasm`, SHA-256
  `b49e945097dd022c7a633af57118298b0c559fbcd29999101264e26b4012b479`.
- Benchmark script SHA-256:
  `842b00f0fff65f67160a76a3626f3b0057c42e3fd43cd6123bbd658db3fa17ea`.

The preliminary reports retain their original main worker SHA-256
`1b24b16c18db5deae660fd866a750d10382c1bce0e3db550708a5d7575a15ca3`
and script SHA-256
`03c4181792611b077603366383cef33d3a4a53e6d92df2198c3f6dd6427d665c`.
Both controls explicitly pass `cpuPipeline`, so the later C2C default-policy
change does not change the measured controls. The final script also
clarifies that old-state release submission is outside timing; release
acknowledgements are not awaited. Existing reports are kept as captured.

## Validation

The scheduling test checks both cold and warm queued searches before the
synchronous voxel phase, and confirms that an acknowledgement-gated cold
control dispatches afterwards. A direct production worker test posts
preparation, scalar/SIMD searches and release before WASM initialization,
then verifies ordered responses and exact distances. A separately rejected
preparation acknowledgement suppresses callbacks even when both queued
searches succeed; the group drains the held search before releasing state.

The existing cache tests exercise preparation/search/release failures,
incomplete results, idle worker loss, source edits/removal, active leases,
changed Map quality parameters and collected/progressive parity. CPU C2C,
Map quality, M3C2 and summary suites cover the shared pool; software and
physical WebGPU plus the Map quality/M3C2 UI checks cover the production
paths. The Rust kernel, generated WASM and pool worker are unchanged.

The final build and 112 distinct browser checks passed: 38 CPU/Map quality
cases, 36 software WebGPU cases, 36 physical WebGPU cases and two UI cases.
Four invalid CLI combinations and an overwrite attempt were rejected.
All six reports match the unchanged input/manifest hashes; all three final
reports match the final production artifacts. Script syntax, documentation
links and public filename/content checks passed.

```powershell
cd web
npm run build
$env:PW_PORT='4193'
npx playwright test e2e/map-quality-cache.spec.ts e2e/map-quality-parallel.spec.ts e2e/c2c-cache.spec.ts e2e/c2c-streaming.spec.ts e2e/c2c-simd.spec.ts e2e/m3c2-parallel.spec.ts e2e/distance-summary.spec.ts e2e/gpu-c2c.spec.ts --workers=1
npx playwright test e2e/app.spec.ts --grep 'map quality:|M3C2: a flat grid' --workers=1
$env:CA_GPU_HARDWARE='1'
npx playwright test e2e/gpu-c2c.spec.ts --workers=1
```
