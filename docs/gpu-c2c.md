# Experimental WebGPU C2C

Public benchmark reports use generic adapter labels and omit GPU model names.
They retain vendor/architecture, applicable driver/memory information, raw
timings and artifact hashes. Recorded script hashes identify the versions
used to measure the reports, including versions before model omission.

The browser's **Distance → Nearest neighbour → Compute with → WebGPU
(experimental)** option runs nearest-neighbour candidate search on the GPU.
CPU workers remain the default. This is a hybrid with exact CPU verification;
performance depends on the data and whether the reference can be reused.

Open two point clouds, choose the compared and reference clouds, select
WebGPU, and compute. The status reports `with WebGPU` only after all GPU
batches complete. If WebGPU is unavailable, a device is lost, or a buffer or
pipeline fails, the complete result is computed on the CPU and the status
includes the fallback reason. C2M, M3C2, map quality, normals and filters use
their existing implementations.

Selecting WebGPU starts asynchronous device and pipeline preparation before
the compute button is pressed. Preparation errors leave the existing CPU
fallback available and can be retried on the next selection or computation.
The worker also starts adapter discovery before synchronous reference
preparation when no device is ready. Reference and query snapshots still
exist before the computation first yields to other worker requests.

## Precision and coverage

Rust builds a k-d tree from the reference's original `f64` coordinates. The
GPU receives shifted `f32` copies of its points and split planes and returns
candidate slots, not measured distances. For **every query**, Rust scans the
candidate's leaf using the original coordinates. A precomputed certificate
then tests whether every point outside that leaf is at least as far away.
Each outside point belongs to a sibling subtree along the leaf's ancestral
path. The certificate uses those siblings' exact coordinate extrema along
their split axes as lower bounds; it never relies on the GPU's rounded data.
All comparisons use squared `f64` distances with the same arithmetic as the
ordinary search. Entirely coincident leaves need only one distance evaluation.

The WASM build can scan two leaf points at a time with fixed-width `f64x2`
SIMD, for both the initial leaf and visited sibling leaves. Three
[unaligned vector loads](https://doc.rust-lang.org/core/arch/wasm32/fn.v128_load.html)
cover exactly the six coordinates of a pair, then shuffles put the points in
separate lanes. This uses the existing coordinate array without a second
coordinate copy. Separate multiply/add instructions preserve
`(dx*dx + dy*dy) + dz*dz`. A lane-wise minimum first checks whether any pair
has a distance below the current hit. If so, a scalar scan selects the slot
with the original point order and strict tie comparisons; otherwise the hit
is retained. Finite coordinates imply nonnegative squared distances (possibly
infinite), so this comparison introduces no NaN or negative-zero ambiguity.
A final odd point uses scalar arithmetic. Other compilation
targets use the scalar implementation. `compute.verifySimd` records whether
leaf SIMD is enabled in the compiled WASM and current job.

If an axis exclusion test fails, Rust walks the leaf's ancestors and verifies
each sibling subtree using its exact `f64` bounding box. Those disjoint
siblings cover every point outside the initial leaf. A single split-axis gap
first rejects distant siblings without projecting their full boxes.
For the remaining siblings, the query is projected
onto a box; its squared distance is a lower bound for all points inside,
computed with the same component order and arithmetic as point distance.
Boxes at least as far away as the current best hit are skipped. Other boxes
are searched recursively, narrowing the bound when a closer point is found.
The initial leaf is not searched a second time. Coincident sibling leaves
also need only one distance evaluation.

Invalid slots and nonfinite/overflow cases use the ordinary complete search.
A wrong or rounded candidate cannot omit a closer
original-coordinate point. Only the verified distance enters the existing
statistics and `f32` display/export arrays. The result's `compute.certified`
counts queries verified without evaluating points outside the candidate leaf,
including queries excluded by sibling boxes. `compute.verifyBounds` records
whether that verification path was enabled. The previous plane-only verifier
is retained for controlled benchmarks; it performs the ordinary complete
tree search when the axis certificate fails.

Queries and the reference tree are owned snapshots. One unchanged reference
tree and its GPU buffers are cached for subsequent jobs, including jobs with
a different compared cloud. Transform/ICP, source removal and reference
switches retire that entry. A retired entry is released after its active jobs
finish. The GPU device and pipeline are cached separately and survive
reference changes, removal and switches. Failed or lost devices are discarded;
the next preparation or computation can retry. `compute.deviceReused` records
whether a ready device and pipeline existed when the job started; this is
independent of `compute.referenceReused`.
Cloud edits or removal during asynchronous work cannot invalidate a job's
snapshots. Query snapshots are released after each job.

Query, output and readback GPU buffers are reused, growing only when a larger
batch needs them. Bindings and mapped ranges cover only the current batch,
including a final partial workgroup. Pipeline creation, reference uploads and
dispatch/readback are serialized per device, including requests for different
references. This also keeps the device's
[error scope stack](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/pushErrorScope)
from capturing errors from another reference's operation. Once a batch's
readback is unmapped, the next batch is submitted before
Rust verifies the current result. This overlaps GPU work and CPU verification
with at most one pending batch per job. The compute shader's stack, node
visits and coincident-point scans are bounded; unfinished searches still return
only hints and undergo the same authoritative verification.

This computes the loaded cloud's points. A display LOD remains a subsample;
choose a [full-density working box](large-point-clouds.md#web-full-density-working-box)
when the operation needs the original source density.

## Current limits

- Each cloud may contain at most 2,000,000 loaded points on the GPU path;
  larger inputs use the existing CPU implementation.
- The cache keeps at most one idle reference; retired references remain only
  for active snapshot jobs. Queries are processed in batches
  of at most 65,536 below 1,000,000 compared points and at most 131,072
  otherwise. Both caps are reduced to the device's storage, buffer and
  dispatch limits. Internal benchmark requests can choose a cap in 1..262,144.
- Query/output/readback buffers use 24 bytes per retained query slot: up to
  1.5 MiB for the smaller default cap, 3 MiB for the larger default cap,
  or 6 MiB for the largest benchmark cap, in addition to reference buffers.
  Capacity grows to a power of two (bounded by the device cap) and remains
  until the reference is released; smaller jobs bind only their live ranges.
- One healthy shared GPU device/pipeline remains until the worker/page closes
  or the device is lost, even when all reference buffers have been released.
- The first job includes device/pipeline setup unless it was prepared earlier.
  Jobs after reference changes still include tree preparation and upload;
  subsequent jobs against an unchanged reference reuse those resources. Readback and
  exact CPU verification still add costs; certificate success depends on the
  geometry and candidate quality.
- Subtree boxes and parent links remain in the CPU reference cache. These
  arrays add 52 bytes per tree node, about 6.5 MiB for the million-point grid
  used below; coincident clouds can have far fewer nodes.
- This is not GPU-only C2C, full-density out-of-core GPU processing, or a
  measured ten-billion-point implementation.

The buffer and dispatch checks follow the [WebGPU specification](https://www.w3.org/TR/webgpu/);
the compute shader uses [WGSL](https://www.w3.org/TR/WGSL/).

## Verification and measurement

Native tests check arbitrary and invalid candidates against brute force and
exact CPU distances, including certificate boundaries, micrometre offsets at
large coordinates, adjacent representable boundary values, duplicates,
overflow and shared snapshots after source changes. They also test diagonal
box exclusions and per-job verifier selection. Browser tests compare complete
PLY exports and exact statistics
against CPU output. They cover reference reuse/invalidation, overlapping jobs
with source edits/removal, multiple batches, missing WebGPU, device loss and
recovery, and a reduced buffer limit. Counters check device/pipeline reuse
across reference changes, buffer release, and non-overlapping device error
scopes. UI tests also exercise preparation before computation and retry after
a failed pipeline preparation. A separate browser test runs the actual WASM
SIMD verifier and its scalar control against brute force, comparing `f64`
distance bits and certificate counts. It covers one-point inputs, odd and
even leaves, duplicate ties, wrong/invalid seeds, UTM offsets and subnormal
through near-overflow squared distances. The standalone GPU host module checks
candidate slots against brute force across a split and partial workgroups,
and counts allocations through buffer reuse/growth and concurrent requests.
Batch-control tests grow through 65,536, 131,072 and 262,144 queries, then
shrink, preserving exact distances/statistics and the cached reference/device.
They also test successful clamping to each device limit, invalid controls
with complete CPU fallback, buffer release, and loss after a maximum-sized
batch followed by a one-query tail.

```sh
cd rust
cargo test --workspace
cd ../web
npm run wasm
npm run build
npx playwright test gpu-c2c.spec.ts --workers=1
```

The browser tests use SwiftShader for portable GPU API coverage. Set
`CA_GPU_HARDWARE=1` to use a physical GPU. GPU-dependent tests skip if no adapter
is available; the missing-WebGPU fallback test still runs. Give simultaneous
Playwright runs separate `--output` directories as well as separate ports.

Measure the production worker with the standalone benchmark:

```sh
node scripts/benchmark-gpu-c2c.mjs --hardware \
  --compare-simd \
  --sizes 10000,100000,1000000 --repeats 3 \
  --output ../notes/gpu-c2c-hardware.json
```

The report includes adapter information, browser version, worker/WASM hashes,
the first GPU job (`coldGpu`), all subsequent trial times, preparation/GPU/
verification costs, reference and device reuse, certificate counts, batches,
effective batch caps, CPU workers, bitwise distance equality and exact
statistic equality. `coldGpu`
uses a fresh reference, but the device may already exist after an earlier
size or preparation; inspect `deviceReused`. Ordinary trials alternate CPU/GPU
order with the reference already cached. `--compare-overlap` also measures
cached GPU execution with prefetch disabled, using the same production worker
and reference. It rotates CPU, serial GPU and overlapped GPU through all three
positions over three trials. `gpuMillis` includes setup/waits,
query packing, transfer, dispatch and readback; it is not a GPU timestamp.
GPU intervals can overlap CPU verification, so the phase times do not sum
to end-to-end `millis`. Run without concurrent builds or tests when comparing
times. Reports refuse overwrite. The fixture
is a synthetic UTM wavy grid, so its timings are not real-source throughput.
For a dirty working tree, `commit` identifies the checkout base; the worker
and WASM hashes identify the actual built artifacts measured.

Use `--compare-bounds` to compare the new sibling-box verifier with the
previous plane-only verifier in the same production worker. Both GPU modes
retain the same reference and overlap GPU batches with CPU verification.
CPU, plane-only GPU and bounded GPU rotate through all three trial positions.
Choose one comparison flag per run. Add `--fixture scatter` for a reproducible
uniform 3D cloud (seed `0x5eed`, 50 × 50 × 10 m, UTM coordinates) instead of
the wavy grid. Both fixtures compare against a copy lifted by 0.05 m and
remain synthetic, rather than measurements of real-source throughput.

Use `--compare-setup` to measure the first computation against a newly loaded
copy of the reference on every GPU trial. It rotates CPU, a new isolated
device, and the shared device through all three trial positions. Loading is
outside the measured computation; reference tree construction, upload,
candidate search, verification and summaries are included. Both GPU modes
use the same verifier and prefetch settings. The internal `gpuReuseDevice`
request flag selects the isolated control; ordinary UI requests share the
device. Select one comparison flag per run.

`--prewarm` prepares the device once after the first fixture and CPU warmup,
before the first GPU computation. `prewarmMillis` records that separate worker
request, including its round trip. This moves setup before the measured job;
it does not remove the setup work or measure a reduction in total elapsed
time including preparation.

Use `--compare-simd` to rotate CPU, scalar-leaf GPU and SIMD-leaf GPU through
three trial positions, retaining the same cached reference, device, GPU
shader, sibling bounds and prefetch settings. The internal `gpuVerifySimd`
flag disables only leaf SIMD in the control. Every result must match CPU
distance bits and exact statistics, and both GPU modes must make the same
certificate decisions. `scalarGpu` records the control's full computation
and phase times. Select one comparison flag per run.

Use `--compare-batches` to rotate CPU, the 65,536-query control and the
automatic production cap through three positions. An explicit `--batch-size
131072` or `--batch-size 262144` compares that fixed cap instead. Modes retain
the same reference/device, verifier and prefetch settings. Every GPU result
must match CPU distance bits and exact statistics, and both GPU modes must
make the same certificate decisions. `smallBatchGpu` records the control;
`compute.batchSize` records the effective cap and `batches` the successful
dispatch/readback count. Both modes retain the largest work-buffer allocation
after warm-up. This isolates dispatch grouping rather than allocation growth.

## Initial implementation, 2026-10-11 JST

The [recorded production-worker report](../benchmarks/gpu/c2c/windows-webgpu.json)
used Windows, an i7-9750H, an NVIDIA GPU (6 GB, driver 596.36) and Chromium 153.
The final run had no concurrent CloudAnalyzer builds or tests. These are
three warm trials per case on one machine, without controlling other apps,
power management or GPU clocks. Each cloud contained the listed point count.

| Points per cloud | CPU median ms | WebGPU hybrid median ms | CPU workers | GPU batches |
|---:|---:|---:|---:|---:|
| 10,000 | 7 | 72 | 1 | 1 |
| 100,000 | 120 | 182 | 1 | 2 |
| 1,000,000 | 361 | 851 | 8 | 16 |

Every trial had zero differing distance bits and identical `f64` statistics.
This initial version rebuilt the reference and GPU device on every job and
used complete tree verification for every query. It was slower on these
fixtures. The report is retained as the baseline for the caching and leaf
certificate changes. Synthetic timings do not establish real-cloud accuracy
or throughput beyond the comparisons performed.

## Reference reuse and overlapped batches, 2026-10-11 JST

The [comparison report](../benchmarks/gpu/c2c/windows-webgpu-overlap-comparison.json)
uses the same machine and fixture, with three warm trials per mode. Serial
GPU and overlapped GPU use the same cached reference, leaf certificates and
production worker. Only the internal `gpuPrefetch` request flag changes.
The first GPU job for each reference is recorded separately as a single
observation, including setup; it is not a median.

| Points per cloud | CPU median ms | Cached serial GPU median ms | Cached overlapped GPU median ms | First GPU job ms | Certified queries |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 6.9 | 6.5 | 5.1 | 184.0 | 58.1% |
| 100,000 | 87.4 | 46.3 | 43.6 | 146.5 | 55.1% |
| 1,000,000 | 246.1 | 291.6 | 228.2 | 481.1 | 61.6% |

Every recorded GPU result, including first jobs and serial mode, had zero
differing distance bits and identical `f64` statistics. For one million
points, overlapping reduced the warm GPU median by about 22% relative to
serial execution in this run. Reusing the reference removes repeated tree
construction and device/pipeline setup. Leaf certificates avoid complete tree
verification for about 55–62% of queries on these particular grids.

Timing varies substantially between runs. The [cache-only development run](../benchmarks/gpu/c2c/windows-webgpu-cached.json)
measured CPU/GPU medians of 224/266 ms at one million points. An
[earlier overlapped run](../benchmarks/gpu/c2c/windows-webgpu-pipelined.json)
measured 641/646 ms. These reports are retained; other apps, power management
and clocks were not controlled. The comparison above measures all modes
within one run to reduce that ambiguity, and does not establish a universal
GPU advantage. CPU remains the default, first-job setup is still costly,
and performance on real clouds with fewer successful certificates needs
separate measurement.

## Exact sibling-box verification, 2026-10-11 JST

The [grid comparison](../benchmarks/gpu/c2c/windows-webgpu-bounds-axis-grid.json)
and [3D scatter comparison](../benchmarks/gpu/c2c/windows-webgpu-bounds-axis-scatter.json)
measure the final verifier with a split-axis rejection before each sibling's
box projection. Each report uses the same machine, production worker and
cached reference for both GPU modes, with three warm trials per mode and
rotating execution order. Both modes retain batch prefetch. CPU remains the
default; sibling-box verification is enabled when WebGPU is selected.

| Fixture | Points per cloud | CPU median ms | Plane-only GPU median ms | Sibling-box GPU median ms | First sibling-box GPU job ms | Certified queries, planes → boxes |
|---|---:|---:|---:|---:|---:|---:|
| Grid | 10,000 | 7.5 | 7.4 | 6.8 | 179.4 | 58.1% → 76.8% |
| Grid | 100,000 | 91.9 | 54.4 | 41.7 | 165.7 | 55.1% → 71.6% |
| Grid | 1,000,000 | 292.2 | 267.5 | 223.6 | 544.4 | 61.6% → 77.2% |
| Scatter | 10,000 | 6.7 | 4.1 | 6.1 | 192.0 | 91.9% → 97.1% |
| Scatter | 100,000 | 200.6 | 117.8 | 74.9 | 281.3 | 81.7% → 91.8% |
| Scatter | 1,000,000 | 712.5 | 901.1 | 770.9 | 1,552.5 | 65.1% → 77.7% |

Every first job and warm trial had zero differing distance bits and identical
`f64` statistics. At one million points, the sibling-box warm median was
about 16% lower for the grid and 14% lower for scatter than plane-only GPU
within the respective run. Verification medians were 204.6 → 163.7 ms for
the grid and 673.8 → 663.5 ms for scatter; the end-to-end difference also
includes dispatch/readback waits and overlap. These observations do not
isolate the algorithm's effect from runtime or clock variation.

The GPU path was still slower than CPU on million-point scatter in this run,
and box verification cost more than plane-only GPU at 10,000 scatter points.
First-job setup and reference rebuilding remain substantial. Certificate
counts describe avoided outside-leaf distance evaluations, not elapsed time.

The earlier [grid experiment](../benchmarks/gpu/c2c/windows-webgpu-bounds-grid.json)
and [scatter experiment](../benchmarks/gpu/c2c/windows-webgpu-bounds-scatter.json)
projected every ancestral sibling's full box without the initial split-axis
rejection. They increased certificate counts but did not improve the warm
GPU median at one million points (grid 217.4 → 218.0 ms; scatter
381.9 → 389.5 ms). Those reports are retained to show the regression that led
to the cheaper rejection. Compare modes within each report: timings vary
substantially across reports and the fixtures are synthetic.

## Reusable devices and preparation on selection, 2026-10-11 JST

The [device-sharing comparison](../benchmarks/gpu/c2c/windows-webgpu-device-sharing.json)
loads a fresh reference before each GPU trial, so reference trees and point
buffers are not reused. It compares a new isolated device/pipeline with the
shared device/pipeline in the same production worker. Both modes use exact
sibling-box verification and overlapped batches. There are three trials per
mode, rotating CPU, isolated GPU and shared GPU through all three positions.

| Points per cloud | CPU median ms | Fresh reference/device GPU median ms | Fresh reference/shared-device GPU median ms |
|---:|---:|---:|---:|
| 10,000 | 6.2 | 52.4 | 10.1 |
| 100,000 | 76.3 | 122.2 | 69.4 |
| 1,000,000 | 229.5 | 409.2 | 362.5 |

Every GPU result, including the first job, matched CPU distance bits and `f64`
statistics exactly. All shared GPU trials reported `referenceReused: false`
and `deviceReused: true`; isolated trials reported both as false. At one
million points, shared-device execution reduced the median by about 11%
relative to the isolated control in this run. It still took longer than CPU
because reference construction, upload, readback and verification remain.
Loading each fresh reference is excluded from these computation times.

The [preparation report](../benchmarks/gpu/c2c/windows-webgpu-device-prewarmed.json)
recorded a separate 164.6 ms preparation request before the first GPU job.
That first job, on 10,000 points, then took 21.6 ms and reported device reuse
with a fresh reference. This demonstrates moving device/pipeline work before
computation, as selecting WebGPU now does; it does not eliminate the 164.6 ms
setup or demonstrate lower total time including it. The report also retains
all subsequent sizes and warm trials, including their timing variation.

The [pre-change baseline](../benchmarks/gpu/c2c/windows-webgpu-device-baseline.json)
is retained with its production artifact hashes. Use the within-worker
sharing comparison above to assess this change, rather than subtracting
measurements from separate runs. Other apps, power management and GPU clocks
were not controlled. All three reports use the synthetic UTM grid.

Browser tests count only one device and one pipeline across reference
transforms, switches and removal/reloading. They also verify that all buffers
are released after overlapping jobs finish on removed clouds, that error
scopes from different references do not overlap, and that a failed preparation
can be retried before computing. Both physical NVIDIA and software SwiftShader
tests retain exact CPU parity and device-loss fallback/recovery coverage.

## Pairwise f64 SIMD comparisons, 2026-10-11 JST

The [grid comparison](../benchmarks/gpu/c2c/windows-webgpu-simd-reduction-grid.json)
and [scatter comparison](../benchmarks/gpu/c2c/windows-webgpu-simd-reduction-scatter.json)
measure the final SIMD distance-reduction path against scalar leaf scans.
Both GPU modes use the same cached reference, shared device/pipeline,
candidate shader, exact sibling boxes and batch prefetch. Three trials
rotate CPU, scalar-leaf GPU and SIMD-leaf GPU through all three positions.

| Fixture | Points per cloud | Scalar-leaf GPU median ms | SIMD-leaf GPU median ms | Scalar verification median ms | SIMD verification median ms |
|---|---:|---:|---:|---:|---:|
| Grid | 10,000 | 7.1 | 7.0 | 2.4 | 2.3 |
| Grid | 100,000 | 75.1 | 68.7 | 48.0 | 55.8 |
| Grid | 1,000,000 | 289.1 | 268.9 | 205.1 | 203.7 |
| Scatter | 10,000 | 4.9 | 4.5 | 2.1 | 1.8 |
| Scatter | 100,000 | 57.6 | 61.7 | 45.2 | 48.2 |
| Scatter | 1,000,000 | 378.6 | 347.7 | 300.4 | 284.3 |

All first jobs and GPU trials had zero differing distance bits and identical
`f64` statistics. Both GPU modes made identical certificate decisions.
At one million points, SIMD lowered the overall median by about 7% for
the grid and 8% for scatter in these runs. The verification median improved
by only about 0.7% and 5.4%, respectively; overall time also includes
dispatch/readback waits, overlap and summaries. These observations do not
isolate the arithmetic kernel from runtime or clock variation.

The improvement is not uniform: SIMD verification was slower for both
100,000-point fixtures, and the overall scatter median was slower there too.
If a pair contains a point closer than the current hit, selecting its slot
requires a scalar rescan in addition to the SIMD comparisons. Candidate
quality therefore affects the cost. CPU remains the default. Initial jobs
still pay preparation costs; the first million-point jobs took 589.9 ms for
grid and 616.7 ms for scatter, with the device already retained from smaller
sizes. Real-cloud throughput needs separate measurement.

The earlier [grid experiment](../benchmarks/gpu/c2c/windows-webgpu-simd-grid.json)
and [scatter experiment](../benchmarks/gpu/c2c/windows-webgpu-simd-scatter.json)
extracted both distances and updated hit indices after every SIMD pair.
At one million points, their SIMD medians were slightly slower than scalar
(grid 243.5 → 245.4 ms; scatter 308.8 → 317.1 ms). Those reports are retained
as the regression that led to reducing distances before selecting a slot.
Compare modes within a report, rather than subtracting separate runs.

The direct WASM browser test checks 2,040 query distances against both scalar
verification and brute force, comparing all `f64` bits and certificate counts.
It includes odd tails, ties, exact hits, incorrect/invalid candidates, large
offsets, subnormal squared distances and overflow to infinity. The physical
and software GPU suites also cover all existing export, cache, asynchronous
snapshot, device-loss and fallback cases with SIMD enabled.

## Query batch comparisons, 2026-10-11 JST

Large jobs now use 131,072-query batches at 1,000,000 compared points or
above; smaller jobs retain 65,536. The candidate shader, exact verifier,
reference cache and one-batch prefetch are unchanged. The runtime supports
up to 262,144 queries for internal comparisons, with every cap clamped to
device dispatch, storage-binding and buffer limits. `gpuBatchSize` is an
internal request control; it does not add a UI option. Invalid controls use
the complete CPU fallback.

The [final grid report](../benchmarks/gpu/c2c/windows-webgpu-batches-final-grid.json)
and [final scatter report](../benchmarks/gpu/c2c/windows-webgpu-batches-final-scatter.json)
use the final production worker/WASM hashes. Six trials rotate CPU, the old
65,536-query control and automatic selection through all three positions
twice. No CloudAnalyzer builds or tests ran concurrently. Other applications
and CPU/GPU power/clocks remain uncontrolled. Hardware is the same
NVIDIA GPU/i7-9750H machine, using Chromium 153.0.8010.12.

| Fixture | Points per cloud | Old-cap median ms (range) | Automatic median ms (range) | Dispatch/readback count |
| --- | ---: | ---: | ---: | ---: |
| Grid | 1,000,000 | 177.4 (169.2–185.5) | 175.1 (164.7–181.3) | 16 → 8 |
| Grid | 2,000,000 | 378.4 (376.3–391.3) | 367.1 (357.5–384.9) | 31 → 16 |
| Scatter | 1,000,000 | 266.2 (244.0–342.6) | 247.9 (231.4–519.2) | 16 → 8 |
| Scatter | 2,000,000 | 616.3 (594.2–635.0) | 598.7 (577.1–621.1) | 31 → 16 |

At two million points, the end-to-end median improved by about 3% in both
fixtures. The million-point scatter trials varied substantially, including
slower individual large-batch trials. These results establish fewer
dispatches/readbacks and exact parity, not a guaranteed speedup. All first
GPU jobs and both GPU modes matched every CPU distance bit and exact f64
statistic; both modes also made identical certificate decisions.

The 100,000- and 999,999-point cases use the same 65,536 cap in both modes;
their differences are timing variation. The first fresh-reference
two-million-point jobs still took 799 ms for grid and 1,133 ms for scatter,
including tree preparation/upload and verification. CPU remains the default;
for example, the scatter CPU median was about 570 ms at two million points,
faster than the cached GPU hybrid in this run.

Earlier [262,144-cap grid](../benchmarks/gpu/c2c/windows-webgpu-batches-grid.json)
and [scatter](../benchmarks/gpu/c2c/windows-webgpu-batches-scatter.json)
experiments reduced million-point jobs to four dispatches, but used up to
6 MiB of query work buffers. The [131,072-cap grid](../benchmarks/gpu/c2c/windows-webgpu-batches-medium-grid.json)
and [scatter](../benchmarks/gpu/c2c/windows-webgpu-batches-medium-scatter.json)
experiments had comparable million-point improvements using up to 3 MiB.
This led to the smaller large-job cap. All these are fixed-cap comparisons;
reproduce them with an explicit `--batch-size`.

An earlier automatic threshold of 524,288 points was also tested on
[grid](../benchmarks/gpu/c2c/windows-webgpu-batches-adaptive-grid.json)
and [scatter](../benchmarks/gpu/c2c/windows-webgpu-batches-adaptive-scatter.json),
including one-point boundary changes. Some scatter cases around that
threshold were slower, so the final policy keeps smaller batches below one
million points. Compare modes within a report rather than subtracting
separate runs, which have different runtime/clock histories.

```sh
node scripts/benchmark-gpu-c2c.mjs --hardware --compare-batches \
  --sizes 100000,999999,1000000,2000000 --repeats 6 \
  --output ../notes/gpu-batches-grid.json
# Add --fixture scatter for the other fixture; use a new output path.
# Add --batch-size 131072 or 262144 for a fixed-cap comparison.
```

The browser suite covers full/partial large batches, growing then shrinking
caps, original-coordinate exact statistics, invalid controls, recovery and
reference/device reuse. Each device limit is reduced separately and must
still produce the complete GPU result; an insufficient reference-buffer
limit continues to use CPU fallback. Device-loss coverage now destroys the
device after a full 262,144-query batch and submission of its one-query tail,
then verifies exact CPU output and successful retry.

At the batch-sizing stage, the physical NVIDIA and software SwiftShader GPU
suites each passed all 11 tests. The five CPU-parallel M3C2 tests also passed
against that production worker. TypeScript checking and the production
build passed.

## Near-leaf GPU hint experiment, default off

An internal `gpuLeafOnly: true` control explores a cheaper GPU hint: descend
one near branch of the k-d tree and scan that leaf, without GPU backtracking.
Every query still receives the same authoritative original-coordinate f64
verification. An incorrect hint causes exact sibling searches; it never
becomes an approximate displayed distance. The shader bounds descent at 32
levels and coincident-leaf scans at 4,096 points, returning an invalid hint
for the verifier if descent does not finish.

The final experiment uses a separate `leafNearest` compute entry point so
the ordinary DFS stacks do not belong to the hint kernel. Its pipeline is
created only when explicitly requested and cached on the shared device.
Changing entry points rebuilds bind groups for that pipeline's layout while
reusing the same reference and batch buffers. A failed optional pipeline
can retry; CPU fallback still completes every snapshot query. Normal UI
jobs keep the original backtracking search and create only its one pipeline.

The [grid](../benchmarks/gpu/c2c/windows-webgpu-leaf-pipeline-grid.json)
and [scatter](../benchmarks/gpu/c2c/windows-webgpu-leaf-pipeline-scatter.json)
reports compare both hint methods with the same reference, device, batching,
prefetch and f64 verifier. Six warm trials rotate CPU, full-search hints and
near-leaf hints. Every trial matches CPU distance bits and exact f64
statistics. These GPU experiments preceded the final CPU Map quality copy
refinement; their worker hashes identify those measured builds.

| Fixture | Points per cloud | Full-search hints | Near-leaf hints |
| --- | ---: | ---: | ---: |
| Grid | 100,000 | 45.65 ms | 51.40 ms |
| Grid | 1,000,000 | 375.95 ms | 412.25 ms |
| Grid | 2,000,000 | 703.40 ms | 741.65 ms |
| Scatter | 100,000 | 60.40 ms | 59.70 ms |
| Scatter | 1,000,000 | 357.45 ms | 352.65 ms |
| Scatter | 2,000,000 | 913.55 ms | 863.05 ms |

The simpler GPU work did not reliably reduce hybrid wall time. At one
million grid points, certified queries fell from 771,505 to 690,205, and the
CPU verification median increased from 288.55 to 339.95 ms. More queries
therefore evaluated points outside the initial leaf. Near-leaf grid medians
were 5–13% slower across these sizes. Scatter medians improved by about
1–6%, with substantial timing variation and a smaller certificate drop.
CPU clocks, GPU clocks and other applications were uncontrolled. These
synthetic results support retaining the backtracking GPU search as the
default. The optional path remains for controlled experiments.

Earlier [grid](../benchmarks/gpu/c2c/windows-webgpu-leaf-initial-grid.json)
and [scatter](../benchmarks/gpu/c2c/windows-webgpu-leaf-initial-scatter.json)
reports used a shared entry point with a batch-uniform branch. That variant
also slowed grid workloads and was replaced by the separate entry point;
its raw measurements are retained.

```sh
node scripts/benchmark-gpu-c2c.mjs --hardware --compare-leaf \
  --sizes 100000,1000000,2000000 --repeats 6 \
  --output ../notes/gpu-leaf-grid.json
# Add --fixture scatter for the other fixture; use a new output path.
```

Three additional browser tests cover both hint modes over full and partial
batches, pipeline switching without additional batch buffers, concurrent
calls sharing query arrays, a near-leaf hint that differs from the true
nearest candidate, and optional pipeline failure followed by CPU completion
and successful retry. Together with the prior cases, the GPU suite has 14
tests.

At the near-leaf and Map quality streaming stage, the production build
passed the 28-case Map quality, M3C2 and
SwiftShader GPU regression run, both relevant UI tests, and all 14 physical
NVIDIA GPU cases. One NVIDIA case needed an isolated rerun after concurrent
Playwright invocations removed shared trace files during context teardown;
the isolated test passed. Run the suites sequentially or provide distinct
Playwright output directories as well as ports. TypeScript checking and the
production build passed, and final CPU benchmark artifact hashes match the
production files at that stage.

## GPU workgroup comparison

An internal `gpuWorkgroupSize` control selects 32, 64, 128 or 256 GPU
invocations per workgroup. The default remains 64. A WGSL override changes
only this execution size; each invocation still searches one query, and the
same original-coordinate f64 verifier completes every query.

Pipelines are prepared lazily and cached per shared device, hint entry point
and workgroup size. Ordinary jobs prepare only the default pipeline. A ready
variant submits synchronously when the device queue is idle, preserving
next-batch overlap with CPU verification. Switching variants rebuilds bind
groups for their layouts and retains the same five reference/batch buffers.
Failed optional preparation can retry. Device scopes and readback remain
serialized across reference snapshots.

The effective query cap also uses the selected workgroup size times
`maxComputeWorkgroupsPerDimension`, together with storage/buffer limits and
the existing 262,144-query ceiling. Invocation-count and X-dimension limits
reject an unsupported size and complete the owned snapshot on CPU. Invalid
control values use the ordinary complete CPU fallback before GPU setup.
`compute.workgroupSize` records the hybrid selection, or zero for an outer
CPU fallback. Partially filled workgroups bind only live query/output ranges.

```sh
node scripts/benchmark-gpu-c2c.mjs --hardware --compare-workgroups \
  --sizes 100000,1000000,2000000 --repeats 10 \
  --output ../notes/gpu-workgroups-grid.json
# Add --fixture scatter for the other fixture; use a new output path.
```

This comparison gives CPU and all four variants five unmeasured warm-up
rounds, then rotates CPU and the four GPU sizes. Ten repeats put each mode
in each ordering position twice. Each variant reuses the same reference,
device, retained buffers, batch caps and
exact verifier. Reports include end-to-end times, per-stage intervals,
ordering, artifact hashes and bit/statistic parity. `gpuMillis` is the
64-invocation control; `workgroups` contains all four timings.

The warmed [grid](../benchmarks/gpu/c2c/windows-webgpu-workgroups-warmed-grid.json)
and [scatter](../benchmarks/gpu/c2c/windows-webgpu-workgroups-warmed-scatter.json)
reports use the physical NVIDIA GPU. The table shows median warm worker
wall time, including preparation, upload/readback, exact CPU verification
and summaries. All 240 timed GPU variants match the CPU's distance bits and
f64 statistics, with identical certificate counts across workgroup sizes.

| Fixture | Points per cloud | 32 invocations | 64 (default) | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grid | 100,000 | 22.30 ms | 21.20 ms | 22.10 ms | 22.75 ms |
| Grid | 1,000,000 | 180.20 ms | 174.70 ms | 173.35 ms | 172.55 ms |
| Grid | 2,000,000 | 373.75 ms | 373.60 ms | 379.55 ms | 380.35 ms |
| Scatter | 100,000 | 22.55 ms | 24.05 ms | 22.30 ms | 23.75 ms |
| Scatter | 1,000,000 | 258.45 ms | 246.15 ms | 254.25 ms | 250.90 ms |
| Scatter | 2,000,000 | 643.35 ms | 628.40 ms | 640.35 ms | 637.45 ms |

No alternative consistently lowers hybrid wall time across these sizes and
fixtures, so the standard remains 64. Pairwise timing comparisons also vary;
the small median differences do not establish a portable GPU speedup.
CPU/GPU clocks, power and other applications were uncontrolled. These are
synthetic measurements on one adapter.

Initial [grid](../benchmarks/gpu/c2c/windows-webgpu-workgroups-grid.json)
and [scatter](../benchmarks/gpu/c2c/windows-webgpu-workgroups-scatter.json)
reports warmed each optional pipeline once. Their small-cloud trials had
large startup variation, and the scatter two-million-point trials also had
large CPU-verification fluctuations. The reruns above add five complete
warm-up rounds before timed comparisons. Both sets retain their measured
artifact hashes and raw timings.

Three additional GPU tests cover all four sizes with both hint entry points,
partial workgroups/batches, bounded pipeline caching, reuse of five buffers,
dispatch caps, separate invocation/X-dimension limits, invalid controls and
optional pipeline failure followed by complete CPU output and successful
retry. Concurrent hint calls now switch workgroup sizes while preserving the
shared query array, including its nonzero byte offset and NaN padding bits.

At this workgroup-comparison stage, all 17 SwiftShader GPU tests and all 17
physical NVIDIA GPU tests passed. The six M3C2, nine Map quality and two
related UI tests also passed: 51 browser cases in total. TypeScript checking
and the production build passed. The warmed reports' main-worker and WASM
hashes match this tested build. All benchmark runs and browser suites ran
sequentially, with no test/build workload during timing trials.

The subsequent [adaptive distance-summary change](distance-summary.md)
reduces the shared CPU/WASM median stage after GPU verification. Its
full-sort controls and newer benchmark reports isolate that summary change;
the workgroup reports above retain their earlier artifact hashes.

Later [measured-source C2C comparisons](c2c-real-source.md) cover three
actual source geometries, with parallel CPU/GPU comparisons and exact
distance/statistic checks. Backend performance varies by source; those
reports retain input-file hashes and explicit split provenance.

The optional **WebGPU + CPU workers** mode now adds
[parallel exact f64 verification](gpu-c2c-verification.md), with independent
cached worker trees and measured-source serial/parallel controls.
Subsequent [exact-tree preparation](gpu-c2c-verifier-setup.md) transfers the
built f64 tree instead of repeating median partitioning in every worker.
The later [parallel batch policy](gpu-c2c-parallel-batches.md) tunes batch
size for four or more verification lanes using measured-source comparisons.
The subsequent [exact-query preparation experiment](gpu-c2c-query-staging.md)
records per-job coordinate staging, recovery and real-source controls;
coordinate streaming remains the default.
