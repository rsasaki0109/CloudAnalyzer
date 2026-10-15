# Parallel exact verification of GPU C2C candidates

Distance → Nearest neighbour → Compute with → **WebGPU + CPU workers
(experimental)** uses GPU candidate search followed by up to four independent
WASM workers for authoritative f64 verification. CPU workers remain the default
backend; the existing WebGPU option retains serial verification. The worker
protocol also accepts `gpuVerifyWorkers: 1..8`, capped by the available CPU pool.

Each verification worker now receives the complete immutable reference as a
validated exact tree snapshot, retaining f64 coordinate/split bits and the
same candidate-tree slots. The original-XYZ rebuilding control remains
available as `gpuVerifySnapshot: false`. See
[exact-tree preparation](gpu-c2c-verifier-setup.md) for the format, checks and
first-use comparisons. GPU-rounded coordinates never enter the verifier. Trees
are cached across jobs; increasing the lane count prepares only new lanes.
Each GPU batch is split into contiguous query ranges with transferred f64
queries and candidate slots. Returned f64 distances are joined in original
query order before the existing exact summary. Prefetch still submits the next
GPU batch while workers verify the current batch. SharedArrayBuffer and
cross-origin isolation are not required.
The [parallel batch policy](gpu-c2c-parallel-batches.md) uses larger batches
for four or more lanes at 524,288 queries or above, with real-source controls and
separate GPU wait measurements.
An optional [per-job query preparation control](gpu-c2c-query-staging.md)
keeps exact coordinates in worker WASM between batches. Streaming remains
the default after mixed real-source results.
An experimental [resident analysis lane](gpu-c2c-main-worker.md) can replace
one pool lane without another reference/query copy. It remains opt-in after
repeated real-source warm regressions; total verification lane counts stay
the same.

The reference owns these worker caches. Coordinate edits or source removal
retire it; simultaneous jobs retain their snapshots until all their batches
finish, then release worker trees. At most one idle reference is retained,
alongside any retired references still leased by running jobs. Each lane holds
an entire tree, so additional lanes increase memory and cold preparation costs.

Worker errors reject pending requests and allow later replacement workers.
Verification follow-ups use the exact prepared worker generation; losing that instance
rejects verification without spawning an unprepared replacement. Preparation
and verification drain all submitted lanes before recovery or release. If any
verification lane fails, the main job recomputes that entire batch and remaining
batches from its exact snapshot, overwriting any partial join. A later job can
prepare the failed lane again. Device loss uses invalid-candidate sentinels and
the same complete exact search. Invalid worker counts use the ordinary outer
CPU fallback; partial output is never returned.

`compute` adds `verifyWorkers`, `verifyPrepareMillis`,
`verifyReferenceReused` and an optional `verifyFallback`. `verifyMillis` includes
query extraction, transfers, exact worker computation, result transfer/join and
any serial recovery. `verifyPrepareMillis` includes reference extraction,
copies, transfer and worker tree import/construction; it can overlap GPU preparation
and the first dispatch. These intervals must not be summed to infer wall time.
`gpuWaitMillis` separately records host time awaiting GPU setup and readbacks,
excluding time spent verifying overlapped batches. It includes browser
scheduling and queue delays, so it is not a shader execution measurement.
After worker recovery, `verifyWorkers` reports the final serial path, with the
failure recorded separately. Displayed distance buffers remain f32; summaries
are computed from the joined f64 values.

## Reproduce measured-source comparisons

Use the unchanged prepared source pairs described in
[real-source C2C evaluation](c2c-real-source.md). From `web/`:

```powershell
node scripts/benchmark-c2c.mjs --compare-verify --verify-workers 4 --hardware --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../notes/gpu-verification-new.json
```

The harness prewarms the CPU pool, recording that excluded time. It then warms
only the main GPU reference/device and records a warm serial run alongside the
first parallel run, including the new workers' reference preparation. This
setup pair is not a comparison of two fully cold backends. Five warm-up pairs
precede ten alternating timed pairs with reference reuse. Every pair must match
all displayed f32 distance bits, all reported f64 statistics and certificate
counts. Timed GPU or verification-worker fallback is rejected. Input, manifest,
script and production-artifact hashes identify each run; existing reports are
never overwritten. Run benchmarks sequentially without test/build workloads.

## Measured results before exact-tree import

These reports measure the previous per-worker XYZ rebuilding implementation.
The i7-9750H / NVIDIA GPU machine, Chromium 153.0.8010.12,
processed every prepared point (`keepEvery: 1`). The four-lane results below
include the complete warmed C2C operation, not just the verification kernel.
Each row has ten alternating timed pairs. The paired reduction is the median
of `(serialMillis - parallelMillis) / serialMillis`, not a ratio of the two
separate medians.

| Source and report | Compared points | Serial median | Four-lane median | Paired reduction |
| --- | ---: | ---: | ---: | ---: |
| [PandaSet 019](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-verification-4.json) | 531,908 | 191.70 ms | 105.80 ms | 44.00% |
| [Autoware planning map](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-verification-4.json) | 878,920 | 282.25 ms | 159.65 ms | 43.84% |
| [Tokyo Takanawadai](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-verification-4.json) | 941,933 | 314.70 ms | 168.80 ms | 46.29% |

Corresponding serial/four-lane verification intervals were 155.55/73.80 ms,
223.10/106.15 ms and 258.05/122.10 ms. Every pair preserved certificate
counts (235,714, 421,096 and 481,955 respectively), every displayed distance
bit and every reported f64 statistic.

First-use reference preparation is substantial. With the main GPU state
already warmed, each setup pair recorded these individual runs:

| Source | Warm serial run | First four-lane run | Worker-reference preparation interval |
| --- | ---: | ---: | ---: |
| PandaSet | 219.10 ms | 313.10 ms | 215.10 ms |
| Autoware | 340.80 ms | 492.30 ms | 336.10 ms |
| Tokyo | 280.80 ms | 467.50 ms | 338.50 ms |

These are single setup observations, not cold-start medians. GPU setup and
CPU-pool startup are already excluded. The first parallel call is slower in
each setup pair. Reference changes require rebuilding the worker trees, so
the demonstrated benefit in these reports applies to reusing an unchanged reference. This is
why the new mode remains optional and the usual CPU backend is unchanged.

Autoware additionally compares two and eight lanes in separate runs:

| Lanes and report | Serial median | Parallel median | Paired reduction | First parallel run | Worker preparation |
| --- | ---: | ---: | ---: | ---: | ---: |
| [2](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-verification-2.json) | 304.40 ms | 218.40 ms | 26.59% | 455.20 ms | 245.20 ms |
| [4](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-verification-4.json) | 282.25 ms | 159.65 ms | 43.84% | 492.30 ms | 336.10 ms |
| [8](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-verification-8.json) | 292.20 ms | 130.05 ms | 55.71% | 675.40 ms | 510.40 ms |

Eight lanes lower warmed time further on this source, while doubling the
number of reference trees versus four and increasing first-use preparation.
The UI offers four as a compromise; the protocol and benchmark retain 2..8
for evaluation. These separate runs do not establish the best lane count on
other sources or machines.

All 50 timed pairs, five setup pairs and 25 warm-up pairs passed exact output
and certificate checks. The production hashes in every report match the
tested build: main worker `e8aecef0afc7e82e082743dbd00e178cefa3d93819372ac426e2b7dd758be770`,
pool worker `fff724d402d499c60aef72e36c79086ef73c869c800bd2d1988e5aa071d6aa75`,
WASM `f9b5abdfaa75fb29014af9929a71c186cb801169b82a975e476d6ee19bd4a8bc`.
The reports retain their measurement script hash,
`66eb435eed500cc8b4eb76b021822dfcab1521b567a8a88e71eb0daacba53a98`.
The subsequent output-only script change omits GPU model fields; measured
kernels are unchanged. Public report filenames use a generic WebGPU label,
and any earlier model metadata is generalized without changing measurements
or recorded source/artifact hashes.
Source provenance and prepared-file hashes remain the same as the earlier
[measured-source comparisons](c2c-real-source.md).

This project's test/build workloads finished before sequential timing runs.
Other workspace Rust builds were observed immediately before these runs;
other applications, CPU/GPU clocks and power were uncontrolled. The raw
timings and separate serial controls capture variation. These are measured
development workloads on one machine, not isolated hardware throughput or
portable speed guarantees.

## Verification before exact-tree import

Native checks passed 46 distinct cases across the WASM crate, k-d tree,
distance and associated distance-filtered tests. Browser checks passed 76
cases: 24 each on software and physical GPU, 23 CPU distance/M3C2/Map quality
cases and five related UI cases. Tests cover exact f64 bits and tree-slot
reconstruction, tiny/odd inputs and extreme magnitudes, partial GPU batches,
lane-cache growth, scalar/bounds/prefetch/hint controls, worker preparation
and mid-job crashes with whole-batch recovery and retry, concurrent snapshot
jobs with reference edits/removal, device loss, invalid controls and the new
UI/export path. Rust formatting, native/WASM Clippy with warnings denied,
TypeScript checking and the production WASM/Vite build passed.
