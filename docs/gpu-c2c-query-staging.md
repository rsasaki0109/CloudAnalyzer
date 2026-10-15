# Exact query preparation for parallel WebGPU C2C

The internal `gpuStageQueries: true` control prepares a job's original f64
query coordinates in each verification worker before its first batch. Later
verification messages contain candidate slots and offsets into the owned WASM
query snapshot. The default remains per-batch coordinate streaming because
measured-source comparisons did not establish a consistent overall benefit.
The existing UI and parallel batch-size policy keep their defaults.

`GpuC2cJob.workerQueries(lane, workers, batchSize)` concatenates that lane's
ranges from all batches. It uses the effective device-clamped batch size and
the same contiguous within-batch partitions as streaming verification. Tiny
batches use only nonempty lanes, including tails smaller than the requested
worker count. Each query belongs to exactly one lane. Original f64 bits, query
order, tree-order candidate slots and certificate counts are preserved.

`GpuC2cVerifier` owns the received flat f64 vector in WASM and retains an `Rc`
to the immutable exact reference tree. Its `verify(first, candidates, bounds,
simd)` validates the range and uses the same verification helper as the
streaming reference method. No GPU-rounded coordinate enters this path.
Constructors enforce the existing 2,000,000-point limit and complete XYZ
triples. Invalid lane/batch/range controls are rejected.

Query snapshots are unique to a job and released after success or failure.
They are not reused across jobs or source edits. Concurrent jobs therefore
keep independent queries, even when a shared reference is edited or removed.
Preparation and verification drain all submitted lanes before serial recovery;
any failed batch is entirely recomputed from the main job's immutable snapshot.

Cleanup now targets existing pool workers. A terminated worker's WASM state
is already gone, so releasing reference/query state does not start a replacement
worker solely for cleanup. If a query-release request loses a worker, its cached
reference lane is invalidated. The next job imports that reference into the
replacement worker instead of encountering a missing reference during verification.

## Costs and measurements

The total coordinate transfer volume is unchanged: `24 * queryCount` bytes
per job. Staging moves these transfers and allocations before batch verification
and retains all lane queries until the job finishes. It adds preparation and
release messages, while reducing per-batch payloads and repeated WASM input
allocations. The worker query vectors collectively occupy 12,765,792,
21,094,080 and 22,606,392 bytes for the three measured-source workloads.
This is additional owned query storage, separate from cached reference trees
and the main job snapshot. WASM memory capacity can remain at its high-water
mark after the vectors are freed; these values are not process memory measurements.

`compute.verifyStagedQueries` reports successful staged parallel verification.
`verifyQueryPrepareMillis` includes lane packing, transfers and WASM ownership;
`verifyQueryBytes` records the aggregate prepared coordinate bytes. Preparation
starts after the first GPU dispatch and can overlap it. `verifyMillis` retains
batch verification/message/join timing. GPU spans overlap these intervals and
must not be summed to infer wall time.

The production benchmark's `--compare-verify-queries` holds reference snapshots,
worker count, batch size, candidate shader, prefetch, bounds, SIMD, workgroup and
adaptive-summary controls constant. Every timed call prepares and releases its
own staged queries; no cross-job query reuse is included. Five warm-up pairs
precede ten alternating timed pairs. Reference setup, automatic CPU pool
prewarming and file loading are excluded; every call's query preparation,
verification, joining, release submission and exact summaries are included.
Other comparison modes explicitly retain coordinate streaming to isolate
their existing controls.

The unchanged real input files and provenance are described in
[measured-source C2C](c2c-real-source.md). They are disjoint splits of measured
geometry, not independent surveys or ground-truth accuracy evaluations.

From `web/`, after building WASM and the production app:

```sh
node scripts/benchmark-c2c.mjs --compare-verify-queries --verify-workers 4 --hardware \
  --reference ../notes/c2c-real-pandaset/reference.ply \
  --compared ../notes/c2c-real-pandaset/compared.ply \
  --input-manifest ../notes/c2c-real-pandaset/inputs.json \
  --output ../notes/query-staging-new.json
```

Reports retain input/manifest/script/build hashes, all compared controls and
exact parity checks. GPU model names are omitted; the generator refuses to
overwrite existing reports.

## Initial and repeated results

Paired reduction is the median of `100 * (streamedMillis - stagedMillis) /
streamedMillis` across ten pairs. Negative values mean increased time. This
is distinct from the percentage difference between two wall-time medians.

| Source | Initial paired reduction | Repeated streaming median | Repeated staging median | Repeated paired reduction | Repeated faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet | [2.77%](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-query-staging-4.json) | 81.65 ms | 79.70 ms | [3.61%](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-query-staging-4-repeat.json) | 7/10 |
| Autoware | [2.12%](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-query-staging-4.json) | 125.65 ms | 125.85 ms | [−1.29%](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-query-staging-4-repeat.json) | 3/10 |
| Tokyo | [−2.25%](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-query-staging-4.json) | 115.00 ms | 123.00 ms | [−5.64%](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-query-staging-4-repeat.json) | 2/10 |

Repeated preparation medians were 4.85, 8.25 and 9.50 ms respectively. GPU
wait medians decreased from 10.25/16.05/9.60 to 5.55/7.60/1.15 ms, but lower
GPU wait did not imply lower wall time. Verification medians were
55.80/53.35, 82.15/80.30 and 77.30/78.90 ms (streamed/staged). These runs do
not isolate allocation/locality effects, result summaries or host workload
variation. The combined wall-time evidence supports retaining streaming as
the default and staging as an explicit comparison control.

## Final-build repetition

After the cleanup/recovery fix, the same comparisons were repeated against
the tested final production build. The verification kernels and partition
layout were unchanged from the initial six reports.

| Source / report | Streaming wall median | Staging wall median | Paired reduction median | Faster pairs | Preparation median |
| --- | ---: | ---: | ---: | ---: | ---: |
| [PandaSet](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-query-staging-final.json) | 93.95 ms | 86.65 ms | 2.53% | 6/10 | 5.65 ms |
| [Autoware](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-query-staging-final.json) | 131.90 ms | 128.35 ms | 3.53% | 6/10 | 8.75 ms |
| [Tokyo](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-query-staging-final.json) | 125.60 ms | 128.90 ms | 4.05% | 6/10 | 9.80 ms |

Tokyo's paired reduction is positive in this final run, while its staging
wall median is higher. The two statistics summarize different quantities;
neither removes the variation visible in individual pairs. Earlier Tokyo
runs regressed, and Autoware's repeated run did not improve. The final run
does not establish enough consistency to enable staging by default.

GPU wait medians were 10.85/5.50, 16.10/7.55 and 10.15/1.90 ms;
verification medians were 65.70/57.10, 88.45/83.65 and 86.15/87.15 ms
(streamed/staged). Certificate counts remained 235,714, 421,096 and 481,955.

The three final reports identify:

- Main worker: `a9c9ae0a9773e495df275de1661c1b2b33a3a59b42661b46b114dd7eabe075e7`.
- Pool worker: `d7c165769866e991cc9eef256f090e9419a5db78ef165b62cac0d1dd3c108763`.
- WASM: `e6c99c78eb3daf0a1f73d09da7fc1d8e60e8b421307ee4c8d15832945527bd4c`.
- Measurement script: `b2b7352da98695a401bd59515cb6dd6cdd90ad35f1e8c2bea514bedf4063101c`.

The initial six reports retain their earlier main-worker hash. Pool-worker,
WASM and measurement-script hashes match the final reports. Source files
and preparation manifests remained unchanged across all nine reports.

Builds/tests and measurements run separately; measurements run sequentially.
CPU/GPU clocks, power and other applications are uncontrolled. Absolute times
vary between runs, and these results do not establish a portable speed guarantee.

## Verification

The final build passed 22 native WASM-crate tests and 93 browser executions:
35 GPU cases on each of software and physical GPU, plus 23 CPU distance,
summary, M3C2 and Map quality cases. New coverage checks original f64 bits,
owned query/reference lifetimes, empty/malformed inputs and invalid ranges,
one/tiny/odd and partial batches, device caps, two/four/eight lanes,
scalar/bounds controls, independent concurrent jobs and reference edits/removal.
Injected query-preparation, staged-verification and release failures verify
whole-batch recovery and subsequent reference re-preparation.

Rust formatting, native and wasm32 Clippy with warnings denied, TypeScript
checking, and release WASM/Vite builds passed. All 90 timed pairs and 45
warm-up pairs in the nine retained reports passed exact f32 output, reported
f64 statistics and certificate checks without fallback. Final-report hashes
match the tested build. Invalid CLI comparison/worker/batch combinations
and overwriting an existing report were rejected. Public content and
filename scans found no omitted GPU model name.
