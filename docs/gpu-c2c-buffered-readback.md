# Two reusable WebGPU C2C readback slots

The internal `gpuBufferedReadback: true` comparison permits the next GPU batch
submission while the preceding readback is still mapping. It requires
`gpuPrefetch` (enabled by default). The ordinary single-slot path remains the
default after variable complete-call measurements. `compute.bufferedReadback` reports the effective
option, which is false when prefetch is disabled.

## Ownership and scheduling

Each reference retains its existing point/node buffers and up to two independent
input/output/readback triples. Each slot has a FIFO lease that covers submission,
map, result copy and unmap. Resizing destroys only that idle slot's old buffers.
Partial allocation failure releases the new buffers before retry. Binding ranges
include only live queries, including short final batches.

Device error scopes remain serialized across references with at most three
active scopes. The buffered path starts mapping inside its submission scopes,
then closes those scopes before awaiting the readback. A second slot can submit
without waiting for that map. Scope errors still drain mapping before releasing
the lease. Map rejection is handled immediately. The ordinary path keeps its
scopes until readback completes, as before.

The C2C driver queues at most two batches for each job, in source order. After
joining one readback, it queues the subsequent batch before exact CPU
verification. GPU failure drains that job's queued batches before disposing its
search and completing the remaining distances on the CPU. Unexpected exits also
drain pending GPU work before releasing the reference lease. Other jobs retain
independent immutable query snapshots; coordinate edits and source removal
continue to retire references only after active leases finish.

This changes submission timing and buffer ownership. WGSL search, candidate
slot order, original-coordinate f64 verification, certificate counts and summary
arithmetic are unchanged. Verification-worker snapshots still reuse validated
import bounds; this option adds no CPU reference replica or staged-query cache.

One triple occupies 24 bytes per retained query capacity: 16 input, 4 output and
4 readback. Two triples at the usual 131,072 capacity occupy 6 MiB instead of
3 MiB; at the maximum 262,144 capacity they occupy 12 MiB instead of 6 MiB.
These are buffer capacities, not measurements of total process memory. Buffers
can retain the largest previously used capacity until the reference retires.

## Real-source comparison

`web/scripts/benchmark-gpu-buffered-readback.mjs` compares both modes with the
same warmed immutable reference and exact verification workers. Excluded setup
loads measured files, prepares the worker pool, main tree/device/reference
upload, exact worker snapshots and both buffer capacities. Five excluded
alternating warm-up pairs precede ten alternating timed pairs per report.
Both modes use four pool lanes, validated snapshot-bound reuse, f64 bounds/SIMD,
adaptive batching, 64 invocations per workgroup, prefetch, and no resident lane
or query staging.

The reported time covers the complete warm C2C RPC, including owned query
snapshots, GPU upload/search/readback, exact verification, joining, summaries
and return. Every result must match the excluded serial baseline and the other
mode's displayed f32 distance bits, exact reported f64 statistics and certificate
counts. Worker/GPU fallback, cache misses and mismatched controls abort the run.

`gpuWaitMillis` includes host await/queue delays and is not shader execution
time. `gpuMillis` sums dispatch-to-readback spans which can now overlap other GPU
batches as well as CPU verification. Phase intervals must not be added to each
other or subtracted from wall time.

The sources are [measured point-cloud files](c2c-real-source.md), split by original
source index; they are not independent surveys or distance ground truth. Input,
manifest, script and worker/WASM artifact hashes are verified before and after
timing. Adapter metadata contains only vendor/architecture/fallback status.
CPU/GPU clocks, power, garbage collection and other applications are uncontrolled.
No builds, tests or scans run concurrently with these measurements.

## Results and adoption decision

Six reports contain 60 timed pairs and 30 excluded warm-up pairs. Every report's
paired median reduction is positive, ranging from 1.13% to 8.38%, but only 38/60
timed pairs are faster. Readback-await medians are lower in each report by
0.85–1.75 ms. Complete-call timing varies substantially between runs and within
each run; scope closure and doubled retained capacity do not establish a
reliable speedup for every call. The comparison remains opt-in. The ordinary
path and UI defaults continue to use one slot.

| Source, run | Single / buffered wall median (ms) | Median paired reduction | Buffered faster | First mode faster | Single / buffered await median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [PandaSet 019, initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-buffered-readback-initial.json) | 98.10 / 97.20 | +8.38% | 6/10 | 3/10 | 11.15 / 10.30 |
| [PandaSet 019, final](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-buffered-readback-final.json) | 109.75 / 103.80 | +4.91% | 6/10 | 5/10 | 11.85 / 10.90 |
| [Autoware planning, initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-buffered-readback-initial.json) | 138.65 / 131.85 | +3.97% | 7/10 | 4/10 | 18.70 / 16.95 |
| [Autoware planning, final](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-buffered-readback-final.json) | 148.15 / 144.05 | +1.13% | 6/10 | 7/10 | 18.55 / 16.90 |
| [Tokyo Takanawadai, initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-buffered-readback-initial.json) | 128.40 / 124.90 | +1.75% | 7/10 | 6/10 | 10.65 / 9.30 |
| [Tokyo Takanawadai, final](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-buffered-readback-final.json) | 149.25 / 142.95 | +6.15% | 6/10 | 5/10 | 11.40 / 9.90 |

Paired reductions are calculated per pair before taking the median; they differ
from the ratio of the two marginal wall medians. The first-mode column records
execution-order wins, not optimized-mode wins. Each run has five pairs with
single first and five with buffered first. The final dataset order was the
reverse of the initial dataset order. No causal explanation is assigned to
timing fluctuations, and no cold-start speedup is claimed.

Each of the 90 warm-up/timed pairs checks both results against a serial baseline
and against each other: 270 exact result comparisons. Two additional setup
comparisons per report bring the total to 282. All distance bits, f64 statistics
and certificate counts match. Full-resolution point counts per reference /
compared cloud are PandaSet 531,908 / 531,908, Autoware 878,921 / 878,920 and
Tokyo 941,933 / 941,933. All runs use 131,072-query batches and four pool lanes.

## Reproduction and fingerprints

Build from `web/` with `npm run build` after the SIMD WASM build. Run from the
repository root, choosing a new report path:

```powershell
node web/scripts/benchmark-gpu-buffered-readback.mjs --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output REPORT.json
```

The script requires a physical adapter, defaults to five excluded warm-up pairs
and ten timed pairs, checks real-source manifests, rejects changed artifacts or
controls, and never overwrites an existing report. Eight CLI guard checks cover
missing/unknown/repeated options, invalid lane/point/repeat budgets and existing
report preservation. Adapter metadata omits device names and identifiers.

The measured production artifacts are the same in all six reports:

- `c2c-worker-Cax6DhSg.js`: `933d3a329bd754f713e323071351a303eec955b8483c0159204f9921fd3a575c`
- `ca_wasm_bg-B1Ifysx5.wasm`: `691397806eba68e134e2edb0a0881a8697d175e19afd9ef5025ab62bcdec0d66`
- `worker-Cu9todqF.js`: `aeeffda5952db5f7e642a6b70b58ebc61700876ce462ef274731c8e8670cf664`

Benchmark script SHA-256: `990b6ad51bbaf8d38273566446f4fcdafb6a7c354bcc55f7d2a4bcec93189a8b`.

Report SHA-256 values:

- `windows-webgpu-pandaset019-buffered-readback-initial.json`: `aa006999df124e75d4f006634319d27944445a26295da1c3493aaaed802ae4c6`
- `windows-webgpu-pandaset019-buffered-readback-final.json`: `bf6b4ed6fcbddd696c3c57998ae29efd31b94454ca8c5658474701aa5297f26c`
- `windows-webgpu-autoware-planning-buffered-readback-initial.json`: `ae1b5fa4fd1d8fc5b215d1221c7f3a16029bd467538c23ad03816a4ef193c46e`
- `windows-webgpu-autoware-planning-buffered-readback-final.json`: `cd7ebc63f6974170d6a2cc4624d13819e48f9c0e292db10466ec4f1f411c5837`
- `windows-webgpu-tokyo-takanawadai-buffered-readback-initial.json`: `c9cc7d95a012501c1ef0029ea7020de768fed0cc278147907bc683d46c794bfd`
- `windows-webgpu-tokyo-takanawadai-buffered-readback-final.json`: `9ff97993d186708ef9187ff813bed2cf0c596d1c8d7f7bd134ba24290452c679`

## Verification

The final TypeScript/Vite production build passes. All 56 physical-WebGPU C2C
browser tests pass, plus four ordinary CPU C2C/Map quality SIMD regressions
(60 distinct tests). Eleven selected GPU tests also pass with software WebGPU.
Rust/WASM kernels are unchanged by this comparison.

The new tests cover live partial ranges, buffer growth and hardware limits,
query staging, overlapping jobs across source edits/removal, map failure and
CPU export parity, partial allocation cleanup and retry, device loss with
outstanding submissions, and error-scope failure. A gated first mapping proves
that the second slot submits before the first readback promise completes, while
a third request for the first slot waits through copy/unmap even on scope
failure. Existing candidate, pipeline, snapshot, SIMD, cache and worker failure
checks continue to pass. Device-scope instrumentation stays at three active
scopes and all buffers are released when sources retire.

Run browser checks from `web/`, sequentially with other suites:

```powershell
$env:PW_PORT='4193'
$env:CA_GPU_HARDWARE='1'
npx playwright test e2e/gpu-c2c.spec.ts --workers=2
```

Set `CA_GPU_HARDWARE='0'` for the software-WebGPU comparison. Final report
fingerprints, manifest/input hashes, controls, documentation links and public
file naming were checked after measurement; builds/tests/scans were outside
all timing intervals.
