# Adaptive exact medians for distance summaries

CPU/WASM C2C workers and WebGPU candidate search both finish with a main
worker summary. That summary previously copied and fully sorted every f64
distance to obtain the median. It now selects the middle order statistic or
reads it directly when uniform or ordered input permits, reducing the work
in this serial stage after parallel distance calculation.
The same Rust `DistanceStats` implementation serves native code and browser
C2C/C2M, including Map quality's forward statistics.

The input reduction loop is unchanged: min, max, sum and squared sum visit
distances in the original order. Mean, RMS and standard deviation retain
the same arithmetic. Only the private median scratch copy is rearranged;
source distances and their f32 coloring copies retain their original order.

`select_nth_unstable_by(mid, f64::total_cmp)` places the upper middle value
in position. Odd counts return that value. For even counts, the maximum of
the lower partition in the same total order is the preceding value; the
median remains `0.5 * (lower + upper)`. This preserves signed zeros,
infinity and overflow behavior instead of introducing a different midpoint
formula. Total ordering also retains the placement of negative/positive
NaNs and their payloads. Arithmetic NaN payloads remain unspecified; NaN
results remain NaN. Empty summaries retain their existing error.

When the original reduction has equal nonzero extrema and a non-NaN sum,
every input distance is identical. This case needs no median scratch copy:
odd counts use the extremum, and even counts retain `0.5 * (min + max)`.
The sum guard excludes ignored NaNs; zero extrema skip this shortcut
to preserve mixed signed-zero semantics. Neither inference adds work to the
per-point reduction loop. The full-sort control always retains its copy and
sort. If endpoints match the extrema, a complete total-order scan checks
ascending or descending input. Confirmed order permits reading the original
middle values without a scratch copy, using lower-plus-upper operand order
even for descending input. Endpoint agreement alone is insufficient: a
failed order scan uses selection. This avoids the measured selection penalty
on already ascending data and also handles uniform signed zeros correctly.
Other inputs still need one scratch f64 per distance; this is not a total
peak-memory bound.

## Exact controls and validation

The core's `DistanceStats::from_distances_with_median_sort(distances, true)`
retains full sorting as a control. The browser WASM exports
`summarizeDistances(distances, sortMedian?)`,
`cloudToCloud(compared, reference, simd?, sortMedian?)` and
`cloudToMesh(compared, mesh, signed, sortMedian?)`; omission or `false`
uses the adaptive path. Existing calls can omit the new arguments.

An internal C2C/C2M request `summarySort: true` selects full sorting for
controlled comparisons. `C2cOutput.summarySort` records that choice for CPU,
GPU and CPU fallback outputs. Omission or `false` uses the adaptive path. This
control changes neither worker count nor GPU candidate/verification options.
Map quality's combined evaluation uses the adaptive path; its regression baseline
explicitly requests full-sort summaries in both former C2C directions.

Two native tests compare all non-NaN statistic bits and NaN semantics across
random raw f64 encodings, ordered/reversed inputs, duplicates, organ-pipe
inputs, tiny arrays and every combination of seven extreme values through
length six. Odd medians are also checked against independently sorted raw
bits, including NaN payloads. The original overflowing even-median formula
is checked explicitly.

Browser tests run the production SIMD WASM and compare it with full sorting
and an independent bit-order oracle. They check unchanged input bits,
including a nonzero byte offset, and unchanged f32 coloring. Production
worker tests switch the control across odd/even serial and pool C2C jobs,
signed C2M jobs crossing a plane, and four-batch GPU jobs with partial tails
and cached references. CPU SIMD, streaming C2C, Map quality, M3C2, GPU
fallbacks and the related UI workflows provide regression coverage.

Final-build validation passed 42 native tests (12 distance, 12 k-d-tree and
18 WASM-wrapper tests) and 64 browser cases: 23 CPU cases, five related UI
workflows, 18 SwiftShader GPU cases and 18 physical-GPU cases. Rust formatting,
native and wasm32 Clippy with warnings denied, TypeScript checking and the
production WASM/Vite builds also passed. Browser suites ran sequentially.

```powershell
cd web
npm run wasm
npm run build
npx playwright test e2e/distance-summary.spec.ts --workers=1
npx playwright test e2e/gpu-c2c.spec.ts --workers=1
```

## Reproducible end-to-end comparison

```powershell
cd web
node scripts/benchmark-c2c.mjs --compare-summary --fixture grid --sizes 10000,100000,1000000,2000000 --repeats 10 --output ../notes/c2c-summary-cpu-grid.json
node scripts/benchmark-c2c.mjs --compare-summary --backend webgpu --hardware --fixture grid --sizes 10000,100000,1000000,2000000 --repeats 10 --output ../notes/c2c-summary-gpu-grid.json
# Add --fixture independent and use new report names for varied distances.
```

Create the destination folder first. Existing reports are never overwritten.
Both controls run in the same production worker on a blank page, with the
same source clouds, distance kernels, worker counts and input snapshots.
The script awaits the already-started CPU pool preparation, warms both
controls five times, then alternates the order of ten timed pairs. CPU uses
scalar leaves and progressive joining. WebGPU uses its default prefetch,
bounds, SIMD verification, batch limits and 64-invocation workgroups, with
cached references and device reuse.

`sortedMillis` and `selectedMillis` include the complete distance request:
planning, snapshots, transfers, nearest search, joining, summaries and
coloring. File loading and rendering are excluded. Each pair must match all
f32 distance bits and exact f64 statistics. GPU controls must also retain
certificate counts, batch caps and the expected GPU backend. Reports record
machine/browser/adapter details and main-worker, pool-worker and WASM hashes.
The independent fixture uses different deterministic seeds for the reference
and compared clouds in the same 50×50×10 m box, while the grid uses paired
locations with a 0.05 m vertical shift. The independent fixture stresses
varied nearest distances instead of nearly uniform counterpart offsets.
GPU phase intervals exclude final summaries and can overlap; they should
not be added together to infer wall time.

For the summary stage alone:

```powershell
node scripts/benchmark-distance-summary.mjs --sizes 100000,100001,1000000,1000001,2000000,2000001 --repeats 10 --output ../notes/summary-stage.json
```

This uses the production WASM with its generated glue on a blank page,
without a pool or GPU. Constant nonzero, varied positive and ascending
varied distances cover different ordering and allocation behavior. Timings
include the f64 input copy, reductions, median and Rust f32 coloring
allocation, up to the WASM call's return. Reading statistic getters,
cloning f32 results to JS, parity checks and frees happen outside timing.
Five warm-up pairs precede ten alternating timed pairs. Odd/even counts
are measured separately. These stage results are not whole-C2C speedups.

## Measured-source follow-up

The subsequent [real-source comparisons](c2c-real-source.md) use measured
PandaSet, Autoware and Tokyo survey coordinates without generated offsets or
noise. They include source/split provenance, per-backend full-sort controls
and parallel CPU versus hybrid GPU comparisons. The tables below retain
the original synthetic runs and their measured artifact hashes.

## Final-build synthetic measurements

The final runs used an i7-9750H, 12 logical CPUs, approximately 32 GiB RAM,
Chromium 153 and a physical NVIDIA Turing adapter. Parallel CPU C2C cases
used eight workers; the hybrid GPU
path used one main worker. The tables show separate timing medians in ms
and the median of `(adaptive/sort - 1) * 100` across the ten paired trials.
Negative paired change means shorter. It is not a ratio of the two timing
medians.

Isolated production-WASM summary stage:

| Input | Distances | Full sort | Adaptive | Median paired change |
| --- | ---: | ---: | ---: | ---: |
| Constant | 2,000,000 | 20.25 | 14.50 | −30.28% |
| Constant | 2,000,001 | 22.00 | 15.55 | −29.57% |
| Varied | 2,000,000 | 146.45 | 23.55 | −83.57% |
| Varied | 2,000,001 | 129.30 | 18.60 | −85.32% |
| Ascending | 2,000,000 | 16.10 | 13.25 | −19.50% |
| Ascending | 2,000,001 | 14.90 | 12.25 | −14.02% |

The [summary-stage report](../benchmarks/cpu/distance-summary/windows-i7-9750h-adaptive-median.json)
also contains odd/even 100k and one-million-distance cases. All 180 timed
pairs matched exact f64 statistics and f32 coloring bits. The final ordered
path removed the regression seen in the earlier selection-only stage run.

Complete C2C request, including planning, distance calculation, joining and
the summary:

| Backend | Fixture | Points per cloud | Full sort | Adaptive | Median paired change |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | Grid | 1,000,000 | 267.80 | 268.10 | +0.12% |
| CPU | Grid | 2,000,000 | 592.60 | 620.10 | +0.14% |
| CPU | Independent | 1,000,000 | 420.05 | 361.75 | −11.77% |
| CPU | Independent | 2,000,000 | 785.30 | 667.05 | −15.98% |
| WebGPU | Grid | 1,000,000 | 210.35 | 213.75 | −2.87% |
| WebGPU | Grid | 2,000,000 | 761.95 | 696.90 | +2.90% |
| WebGPU | Independent | 1,000,000 | 666.40 | 623.75 | −6.84% |
| WebGPU | Independent | 2,000,000 | 1,915.20 | 1,785.15 | −14.06% |

Reports: CPU [grid](../benchmarks/cpu/c2c/windows-i7-9750h-adaptive-summary-grid.json)
and [independent](../benchmarks/cpu/c2c/windows-i7-9750h-adaptive-summary-independent.json);
WebGPU [grid](../benchmarks/gpu/c2c/windows-webgpu-adaptive-summary-grid.json)
and [independent](../benchmarks/gpu/c2c/windows-webgpu-adaptive-summary-independent.json).
They also contain 10k and 100k cases. All 160 end-to-end timed pairs matched
exact statistics and distance bits, retaining worker counts, GPU controls
and certificate counts. Together with the stage run, 340 final-build timed
pairs passed exact parity. Every report's relevant main-worker, pool-worker,
WASM and generated-glue hashes were verified against the tested production
build. The final assets are `worker-6iU97dFI.js`, `c2c-worker-Bm6KwllV.js`
and `ca_wasm_bg-B7alC_zx.wasm`.

All benchmarks ran sequentially after this project's tests and builds had
finished. CPU/GPU clocks, power, garbage collection and other applications
were uncontrolled; another project's CPU/GPU workloads were observed during
this session. The GPU grid's two-million-point full-sort trials ranged from
434 to 1,466 ms, illustrating substantial variation. Grid results do not
establish an end-to-end improvement. The independent fixture shows shorter
paired times in this run, with a much smaller whole-request benefit than the
isolated varied-input stage. These synthetic results are not portable speed
guarantees or measured peak-memory reductions.

## Earlier measurements

The first [selection-only C2C grid run](../benchmarks/cpu/c2c/windows-i7-9750h-summary-grid.json)
did not show a consistent end-to-end gain. A subsequent
[isolated summary run](../benchmarks/cpu/distance-summary/windows-i7-9750h-median-selection.json)
included the uniform-input shortcut but preceded ordered-input handling.
At two million distances, its median paired time change was −84.50% for
varied input and −29.86% for uniform input, but +44.60% for ascending input.
That ordered-input regression motivated the complete order scan and direct
middle-value access in the final implementation. Both earlier reports retain
their original measured artifact hashes; they are not final-build results.
