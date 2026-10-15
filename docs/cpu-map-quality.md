# CPU parallel Map quality in the browser

Map quality now evaluates both nearest-distance directions in a single worker
request. Independent C2C slices share the existing WASM worker pool, while
the main analysis worker computes voxel AWD/SCS. It needs no WebGPU,
SharedArrayBuffer or hosting headers.

Previously, the UI awaited map-to-truth C2C, truth-to-map C2C, then voxel
scores. Each large C2C direction already used the pool, but the directions
and voxel evaluation ran sequentially. The new request also keeps both
directions and voxel scores tied to the same source state.

The default now also [reuses bidirectional CPU preparation](cpu-map-quality-preparation.md)
for an unchanged source pair and partition layout. Warm searches are sent
before synchronous voxel computation so those phases overlap. Every
distance, statistic and voxel score is still computed afresh. C2C and Map
quality share one idle preparation group. The historical measurements below
predate reuse; the comparison script explicitly disables preparation caching
in all modes to keep those controls independent.

Fresh calculations on changing compared identities now also retain
[fixed reference voxel statistics and roots](cpu-voxel-reference.md) after
repeated use. The historical comparisons below predate that cache.

## Scheduling and precision

- A direction with at least 100,000 queries is eligible for the pool. Smaller
  directions use the native serial kernel before dispatch. With fewer than
  two available pool workers, both directions stay serial.
- Each eligible direction starts with one slice. Additional slices go to the
  direction with the most queries per prospective slice, with at least
  100,000 queries per slice. The two directions together use at most the
  available pool lanes: `hardwareConcurrency - 1`, capped at eight.
- Slices from the directions alternate in the dispatch queue. One slice
  receives full coordinates; multiple slices use the existing exact spatial
  partitioner and its conservative nearest-reference coverage.
- Original f64 coordinates, point indices, native results for small
  directions, and voxel scores are read or computed before the first yield.
  The source clouds are not accessed after waiting for the pool. Transforming
  or removing either source during dispatch therefore leaves the result
  intact. The UI captures the reference name before waiting, too.
- Returned f64 distances are joined in the original point order. Forward
  distances pass through the same Rust summary function. Reverse distances
  round directly to f32 for inlier arithmetic, without unused statistics.
  Inlier calculations retain the former UI's f32 distances, traversal order,
  threshold comparison and NaN
  semantics. A slice failure rejects the evaluation without publishing a
  partial result; temporary native results and plans are freed.

The reported worker count is the number of pool lanes used, or 1 for a
serial job. When the pool is used, it excludes the main worker that computes
voxel scores and joins the results.

## Copy estimate

The job has a 256 MiB scheduling budget for input coordinate/index snapshots
plus an estimate of 12 bytes per eligible point for joined output. Reference
bytes are counted after spatial pruning; assuming every slice needs the
entire reference would unnecessarily remove workers from large local maps.
An early guard of 40 bytes per point across both inputs avoids planning very
large pairs. If the prepared slices exceed the budget, they are discarded
and both directions run through the native serial kernels.

This is not a total memory limit. Planning can temporarily exceed the
estimate, and collection of discarded JS buffers is deferred. Source clouds,
per-slice results, worker tree construction and temporary allocations,
retained WASM pages, and overlapping requests also consume memory. Broad
reference coverage can consequently select serial computation even for a
large input.

## Validation

The nine-test browser suite covers unequal input sizes, one- and multi-slice
directions, a small reverse direction, small maps, a single available CPU,
empty inlier sets, moving/removing both sources, slice error recovery, and
copy-budget fallback with modeled buffer sizes. Its C2C regression also
confirms that the existing parallel C2C path captures the source
point count before its source is removed. The Map quality UI test retains
its known half-precise, half-complete result. M3C2 and software WebGPU C2C
tests check the other users of the unchanged worker pool.
Controlled completion ordering also verifies that forward statistics finish
while reverse slices are still pending, temporary results are freed, and
late slice callbacks cannot recreate results after a failed evaluation.

Distance coloring arrays are bit-identical to the previous workflow. Their
f64 statistics and all inlier metrics match exactly. AWD/SCS retain the
existing Rust voxel kernel; its HashMap reduction order can vary between
serial calls. Tests allow only `1e-12 * max(1, abs(a), abs(b))` reduction
roundoff for those two scores, with identical voxel counts.

```powershell
cd web
npm run build
npx playwright test e2e/map-quality-parallel.spec.ts e2e/m3c2-parallel.spec.ts e2e/gpu-c2c.spec.ts --workers=1
npx playwright test e2e/app.spec.ts --grep "map quality:|M3C2: a flat grid" --workers=1
```

## Reproducible benchmark

```powershell
cd web
node scripts/benchmark-map-quality.mjs --sizes 10000,99999,100000,200000,500000 --repeats 6 --output ../benchmarks/cpu/map-quality/windows-i7-9750h-final-small.json
node scripts/benchmark-map-quality.mjs --sizes 1000000,2000000 --repeats 6 --output ../benchmarks/cpu/map-quality/windows-i7-9750h-final-large.json
```

Use a new output filename for another run; existing reports are never
overwritten. The script launches only the production analysis worker, with
no viewer or second worker pool. Both workflows warm up once, then run
sequentially with their order alternating. The previous workflow retains
its original per-direction CPU parallelization. Every trial checks distance
bits, f64 statistics, inlier metrics and voxel-score roundoff. Reports record
production worker/WASM hashes, browser and machine details, parameters,
cold times and every individual trial.

Wall times include all RPCs, snapshots, reference tree construction, both
C2C directions, voxel evaluation, summaries and inlier arithmetic. File
loading and rendering are excluded. Overlapping phase times should not be
added to interpret wall time.

On an i7-9750H with Chromium 153, a synthetic UTM wavy grid at 0.1 m spacing
has half its points lifted by 0.07 m and half by 1.1 m. Medians of six warm
trials for the initial combined implementation were:

| Points per input | Previous workflow | Combined workflow | Reported workers |
| ---: | ---: | ---: | ---: |
| 10,000 | 61.85 ms | 64.70 ms | 1 (serial) |
| 99,999 | 305.65 ms | 310.00 ms | 1 (serial) |
| 100,000 | 346.35 ms | 209.15 ms | 2 |
| 200,000 | 726.55 ms | 420.95 ms | 4 |
| 500,000 | 1,974.50 ms | 1,593.30 ms | 8 |
| 1,000,000 | 1,818.55 ms | 1,526.95 ms | 8 |
| 2,000,000 | 6,362.75 ms | 5,425.40 ms | 8 |

These observations correspond to roughly 40–42% shorter median wall time at
100k–200k points and 15–19% at the larger sizes. Small inputs showed no
improvement. Other applications, CPU power and clocks were uncontrolled,
and scatter was substantial: the 2M case ranged from 3.248 to 7.572 s for
the previous workflow and 2.923 to 5.618 s for the combined workflow. The
500k and 1M rows come from separate runs and must not be interpreted as
monotonic scaling. This is one synthetic geometry on one machine, not a
general speed guarantee. The largest AWD/SCS difference in the final trials
was approximately `1.1e-15`.

Raw reports for that implementation: [small](../benchmarks/cpu/map-quality/windows-i7-9750h-final-small.json)
and [large](../benchmarks/cpu/map-quality/windows-i7-9750h-final-large.json).
The retained [initial experiment](../benchmarks/cpu/map-quality/windows-i7-9750h-initial.json)
used a worst-case full-reference copy estimate that cut the 2M job down to
two workers and slowed it down. The final policy counts pruned reference
copies and retained all eight pool lanes on that fixture.

## Streaming assembly and summaries

The default combined request now joins each slice as it finishes. When the
last forward slice arrives, it computes forward statistics and inlier
metrics immediately, while unfinished reverse slices continue on the pool.
Consumed query indices and completed joined arrays are released early.
Reverse distances only need f32 inlier arithmetic, so the parallel reverse
direction skips the unused Rust statistics and median sort. The forward
coloring array used for inlier arithmetic is also returned directly, saving
another native-to-JS copy. Native serial directions retain their existing
C2C kernel.

The internal `streamSummaries: false` request flag retains collect-all
assembly, both native distance summaries and the additional forward
coloring copy for comparison. Input snapshots, partitioning, worker counts
and voxel kernels are identical in the two combined modes. This comparison
measures the changes together; it does not isolate streaming from the
removed reverse summary or coloring copy.

```powershell
cd web
node scripts/benchmark-map-quality.mjs --compare-summaries --sizes 10000,100000,200000,500000,1000000,2000000 --repeats 6 --output ../benchmarks/cpu/map-quality/windows-i7-9750h-streamed-summaries-final.json
```

Use a fresh output filename for a repeat. All three modes warm up once and
then rotate their order over six sequential trials. On the same i7-9750H
and synthetic fixture, the final-build medians were:

| Points per input | Previous three-request workflow | Collect-all combined control | Streaming combined | Reduction vs control |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 56.15 ms | 60.55 ms | 58.15 ms | 4.0% |
| 100,000 | 312.55 ms | 174.10 ms | 173.30 ms | 0.5% |
| 200,000 | 630.35 ms | 367.80 ms | 360.55 ms | 2.0% |
| 500,000 | 786.80 ms | 657.25 ms | 647.65 ms | 1.5% |
| 1,000,000 | 1,331.65 ms | 1,106.40 ms | 1,092.95 ms | 1.2% |
| 2,000,000 | 2,839.80 ms | 2,341.85 ms | 2,154.00 ms | 8.0% |

The largest case reduced the median by about 188 ms versus the already
parallel control. Smaller differences are modest relative to timing noise;
the serial 10k case still offers no gain over the previous workflow. For
2M points, streaming trials ranged from 2.102 to 2.393 s and collect-all
trials from 2.138 to 2.445 s. CPU clocks, power and other applications were
uncontrolled. These runs are separate from the initial implementation's
measurements above, so timing changes between reports cannot be attributed
solely to code changes.

All 36 final trials have bit-identical coloring distances, exact f64 forward
statistics and exact inlier metrics against the previous workflow and the
collect-all control. The largest AWD/SCS reduction difference is
approximately `8.9e-16`, with identical voxel counts. Production worker,
pool worker and WASM hashes are recorded in the
[final report](../benchmarks/cpu/map-quality/windows-i7-9750h-streamed-summaries-final.json).
The retained [intermediate experiment](../benchmarks/cpu/map-quality/windows-i7-9750h-streamed-summaries.json)
still had the redundant coloring copy and used an element-wise reverse
conversion; it showed mixed timing changes. The final version uses typed
array conversion for the reverse distances and reuses forward coloring.
