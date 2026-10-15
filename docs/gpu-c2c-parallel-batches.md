# Query batches for parallel exact WebGPU C2C verification

The experimental WebGPU + CPU workers option uses 131,072-query batches when
at least four verification lanes are available and the compared cloud contains
at least 524,288 points. This leaves at least four batches to overlap GPU
candidate search with exact CPU verification. Smaller jobs and one/two-lane
jobs keep 65,536-query batches below 1,000,000 points. All jobs retain the
existing 131,072 cap at 1,000,000 points or above.

Explicit `gpuBatchSize` overrides remain available in `1..262144`. Every cap
is clamped to the device's dispatch, storage-binding and buffer limits. A lane
failure continues to drain submitted work and recompute the entire affected
batch from the job's immutable original-coordinate snapshot.

Larger batches reduce GPU readback rounds and messages to each verification
lane. The reference trees, candidate slots, query order, f64 search arithmetic
and summary calculation remain the same. Only one next GPU batch is
prefetched. At 131,072 capacity, the three GPU query/output/readback buffers
occupy 3 MiB, up from 1.5 MiB at 65,536. This is query-buffer capacity, not a
measurement of total process memory or worker/WASM memory.

An internal [two-slot readback comparison](gpu-c2c-buffered-readback.md) can
queue a second batch before the preceding map finishes. It remains opt-in;
the default batching described here retains one query/output/readback triple.

## Measurement controls

`web/scripts/benchmark-c2c.mjs --compare-verify-batches` compares two explicit
batch sizes using the same warmed GPU reference, exact worker snapshots and
four verification lanes. The default control is 65,536 and the candidate is
131,072; `--batch-size 262144` and `--control-batch-size 131072` enable the
other comparisons. Five unmeasured warm-up pairs precede ten timed pairs in
alternating order. File loading, CPU pool prewarming and initial GPU/worker
reference setup are excluded. Both controls share the retained buffer capacity
after warm-up.

Each pair must preserve every displayed f32 distance bit, each reported f64
statistic and the exact certificate count. Worker/GPU fallback and clamping
that changes either requested size are rejected. Reports include input,
manifest, script and production artifact SHA-256 hashes.

`gpuWaitMillis` measures host time spent awaiting GPU setup and batch readbacks
after any overlapped verification. It includes browser scheduling and queue
delays. It does not measure shader execution. The existing `gpuMillis` spans
overlapped work and must not be added to `verifyMillis` to infer wall time.
`verifyMillis` includes f64 query extraction, worker messages/transfers,
verification and ordered joining.

Use the unchanged source pairs and provenance in
[measured-source C2C](c2c-real-source.md): PandaSet scene 019 (531,908 compared
points), Autoware planning (878,920) and Tokyo Takanawadai (941,933). These are
disjoint splits of measured source geometries. They are not independent
surveys or a ground-truth accuracy evaluation.

From `web/`, after the production build:

```sh
node scripts/benchmark-c2c.mjs --compare-verify-batches --verify-workers 4 --hardware \
  --reference ../notes/c2c-real-pandaset/reference.ply \
  --compared ../notes/c2c-real-pandaset/compared.ply \
  --input-manifest ../notes/c2c-real-pandaset/inputs.json \
  --output ../notes/parallel-batches-new.json

# Compare 262,144 directly against 131,072:
# add --control-batch-size 131072 --batch-size 262144 and choose a new output.
```

The generator refuses to overwrite existing reports. GPU model names are
omitted; public adapter metadata is limited to vendor, architecture and
fallback status.

## Initial comparisons

Each entry is the median of ten paired percentage time reductions,
`100 * (controlMillis - candidateMillis) / controlMillis`. Positive values
mean shorter full C2C wall time, including exact summaries.

| Source | 65,536 → 131,072 | 65,536 → 262,144 | 131,072 → 262,144 |
| --- | ---: | ---: | ---: |
| PandaSet | [6.31%](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-parallel-batches-131072.json) | [7.65%](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-parallel-batches-262144.json) | [0.36%](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-parallel-batches-131072-vs-262144.json) |
| Autoware | [12.82%](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-parallel-batches-131072.json) | [9.29%](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-parallel-batches-262144.json) | [−1.95%](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-parallel-batches-131072-vs-262144.json) |
| Tokyo | [4.87%](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-parallel-batches-131072.json) | [7.29%](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-parallel-batches-262144.json) | [2.44%](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-parallel-batches-131072-vs-262144.json) |

The direct comparison did not establish a consistent extra benefit from
262,144. It also doubled query-buffer capacity again and increased median
GPU wait in all three sources. The default therefore uses 131,072 for the
parallel workloads described above; 262,144 remains an explicit control.

Initial reports are under `benchmarks/gpu/c2c/real-source/`, with source
prefixes `windows-webgpu-pandaset019`, `windows-webgpu-autoware-planning` and
`windows-webgpu-tokyo-takanawadai`, and suffixes
`-parallel-batches-131072.json`, `-parallel-batches-262144.json` and
`-parallel-batches-131072-vs-262144.json`.

The 131,072 comparison did not reduce GPU wait on every source: Autoware's
median increased from 15.35 to 25.05 ms while its verification interval
decreased from 184.25 to 165.75 ms. These measurements support the combined
wall-time result, without attributing all savings to GPU waits or isolating
worker computation from message overhead.

## Final-build repetition

After changing the default policy, the same explicit 65,536/131,072 controls
were repeated with the tested final production build:

| Source / report | 65,536 wall median | 131,072 wall median | Paired reduction median | Faster pairs | GPU batches |
| --- | ---: | ---: | ---: | ---: | ---: |
| [PandaSet](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-parallel-batches-final.json) | 101.50 ms | 99.30 ms | 4.74% | 5/10 | 9 → 5 |
| [Autoware](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-parallel-batches-final.json) | 155.25 ms | 143.70 ms | 6.59% | 7/10 | 14 → 7 |
| [Tokyo](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-parallel-batches-final.json) | 139.50 ms | 132.10 ms | 4.59% | 7/10 | 15 → 8 |

The paired reduction median is not the percentage difference between the two
wall medians. Individual regressions remain; PandaSet was faster in only half
of its final pairs. The repeated medians support a modest improvement in
these workloads rather than a guaranteed benefit for every run.

Median verification intervals were 70.00/69.60, 107.25/95.60 and
96.45/89.50 ms for the three sources. GPU waits were 10.15/10.90,
15.95/16.80 and 9.55/10.20 ms. The final repetition therefore also does not
show shorter GPU waits. Certificate counts remained 235,714, 421,096 and
481,955 respectively.

The three final reports identify:

- Main worker: `67e75362e4e5de7c83960bdf6a8bbd3c5a74c745e3d6b88e7f2dfbd5d4269416`.
- Pool worker: `05c48e45dd14787e5de0b78202aef5ee39339378858f2eef2400bee66e862720`.
- WASM: `60be266c2b095017cc9ff6e0f388bda5561e06942155fda7e245e30c0bd90dac`.
- Measurement script: `73cd56aefadc3c2d7c6bfee0de0cf07ee65f3c8296fa20ec9c38e06a8bf82e08`.

The initial nine reports retain their own script and main-worker hashes.
The pool worker and WASM kernels did not change during this batch-tuning
stage. Timed comparisons use explicit sizes; browser tests exercise the
automatic defaults separately.

Builds and browser suites finish before sequential measurements. Other
applications, CPU/GPU clocks and power are uncontrolled. Absolute times vary
considerably between runs; comparisons use paired results within a run.
These measurements describe development workloads on one machine and do
not establish portable speed guarantees.

## Verification

The final production build passed 28 browser cases on each of software and
physical GPU (56 total). New cases cover clouds immediately below/above the
parallel threshold, automatic batch caps with one/two/four/eight lanes,
partial large batches, buffer growth/shrink, device dispatch caps, identical
distance/statistic/certificate output, reference reuse and buffer release.
Existing cases cover worker/GPU failures and retries, concurrent jobs,
reference edits/removal, malformed snapshots and identical UI exports.
TypeScript and the production Vite build passed; the WASM binary is unchanged.

All 120 timed pairs and 60 warm-up pairs in the twelve retained reports passed
exact output and certificate checks with no fallback. A real-input
`--compare-verify` smoke run also confirmed that its serial/parallel controls
both use the same 65,536 cap, keeping concurrency comparisons independent of
the new automatic policy. Ten invalid CLI combinations and the existing-report
overwrite guard were rejected without creating/modifying reports. Public
content and filename scans found no omitted GPU model name.
