# CPU C2C result joining in the browser

Large ordinary C2C jobs use the existing spatial partitioner and CPU/WASM
pool. The main analysis worker now joins each completed distance slice while
other pool workers continue nearest-neighbour search. Previously it awaited
every slice, retained all results in an array, then scattered their distances
into the original point order.

The partitioning threshold remains 100,000 queries per prospective slice;
at least two slices are required. The pool uses `hardwareConcurrency - 1`,
capped at eight. Smaller jobs retain the serial kernel. WebGPU verification
has its separate batching path; CPU fallback uses this ordinary C2C path.

## Ownership and exact results

Query count, pruned reference coordinates, query coordinates and original
query indices are captured before yielding. Coordinate buffers belong to
the job and transfer to its pool workers. Moving or removing either source
while workers run does not change the captured job.

The joined f64 output is allocated on the first valid result, after planning
and dispatch. Each result must have exactly as many distances as its query
index slice. Distances are scattered to their original positions; the index
slice is released after consumption. There is no array retaining all result
slices. Those buffers become eligible for collection as callbacks finish;
this is not a guarantee of immediate collection or a total memory bound.

The Rust summary still runs once over the complete f64 array in the same
point order, followed by the same f32 coloring output. Arrival order cannot
change distance bits or statistics. A slice failure rejects the job without
publishing a partial result. A closed-job guard ignores late callbacks, and
the worker pool can serve subsequent jobs.

The internal request `cpuStream: false` retains the collect-all control;
omission or `true` enables progressive joining. `C2cOutput.cpuStreaming`
reports whether a pool job used it, and is false for serial jobs. Ordinary
CPU leaf SIMD is a separate [off-by-default experiment](cpu-c2c-simd.md).

The later [CPU preparation cache](cpu-c2c-preparation.md) retains one unchanged
parallel source pair's queries, worker trees and output indices for repeated
calculation. This joining-stage report records the earlier uncached policy;
its comparison controls explicitly disable preparation reuse.

## Validation

The three joining browser tests exercise the production worker and pool:

- Default progressive joining matches the collected control in every
  displayed f32 distance bit and all f64 statistics, with unequal inputs,
  duplicate coordinates and non-divisible slice lengths.
- A controlled worker response is held until another slice has already
  been joined. Both source clouds are then moved and removed before the
  held response is released; the complete result remains identical.
- A shortened first response rejects with an incomplete-result error. A
  delayed response causes zero output writes after rejection. The same
  pool then completes a new job with exact results.

CPU SIMD, Map quality, M3C2, GPU C2C (software and physical adapters) and
the related UI regressions exercise other users of the same kernels and
worker pool.

At the progressive-joining stage, validation passed 40 native tests (12 k-d-tree, 10 distance and
18 WASM-wrapper tests) and 56 browser tests: 20 CPU cases, 17 software
WebGPU cases, 17 NVIDIA adapter cases and two UI cases. The SIMD WASM
release build, TypeScript/Vite build, Rust formatting and Clippy with
warnings denied on native and wasm32 targets also passed.

```powershell
cd web
npm run wasm
npm run build
npx playwright test e2e/c2c-streaming.spec.ts e2e/c2c-simd.spec.ts e2e/map-quality-parallel.spec.ts e2e/m3c2-parallel.spec.ts --workers=1
```

## Reproducible comparison

```powershell
cd web
node scripts/benchmark-c2c.mjs --compare-join --fixture grid --sizes 10000,100000,1000000,2000000 --repeats 10 --output ../notes/c2c-joining-grid.json
node scripts/benchmark-c2c.mjs --compare-join --fixture scatter --sizes 10000,100000,1000000,2000000 --repeats 10 --output ../notes/c2c-joining-scatter.json
```

Create the destination folder first and use new report names: the script
refuses to overwrite reports. It runs the production analysis worker on a
blank page, awaits the already-started automatic pool warm-up and records
that preparation separately. Five unmeasured pairs warm both controls;
ten timed pairs alternate their order. Both controls use scalar leaves.

Times include planning, snapshots, transfers, reference-tree construction,
nearest search, joining and summaries. Loading and rendering are excluded.
Every pair must match distance bits, statistics and worker counts. Reports
record every trial, machine/browser details and main-worker, pool-worker
and WASM hashes.

On an i7-9750H with 12 logical CPUs, 32 GiB RAM and Chromium 153, the
parallel cases use eight workers. Medians in ms:

| Fixture | Points per cloud | Collected | Progressive | Median paired change |
| --- | ---: | ---: | ---: | ---: |
| Grid | 1,000,000 | 271.00 | 279.30 | +2.13% |
| Grid | 2,000,000 | 561.25 | 584.20 | +0.99% |
| Scatter | 1,000,000 | 375.55 | 400.05 | +1.22% |
| Scatter | 2,000,000 | 616.30 | 607.55 | −1.33% |

Paired change is the median of `(progressive/collected − 1) * 100` for
each pair; negative means shorter. It differs from a ratio of the separate
timing medians. The grid uses 0.1 m UTM spacing on a wavy surface; scatter
uses deterministic coordinates in a 50×50×10 m box. Compared inputs are
lifted by 0.05 m. The 10k and 100k cases stay serial and use identical joining
behavior, so their timing differences are variation rather than this change.

Reports: [grid](../benchmarks/cpu/c2c/windows-i7-9750h-streamed-joining-grid.json)
and [scatter](../benchmarks/cpu/c2c/windows-i7-9750h-streamed-joining-scatter.json).
All 80 timed pairs match distances and statistics exactly. Report hashes
match the production build tested at the progressive-joining stage, before
the separate [median-selection change](distance-summary.md). New comparisons
use the current summary implementation identically in both joining controls.

These runs show small paired differences in both directions, not a
consistent speedup. CPU clocks, power, other applications and garbage
collection remain uncontrolled. Progressive joining is enabled to overlap
the joining work and avoid explicitly retaining every slice result; the
measurements do not establish a wall-time improvement or a measured peak
memory reduction. Future timing comparisons should retain both controls.

The subsequent [measured-source comparisons](c2c-real-source.md) reuse
PandaSet, Autoware and Tokyo survey coordinates, with source-file hashes,
explicit split provenance and parallel CPU versus hybrid GPU controls.
