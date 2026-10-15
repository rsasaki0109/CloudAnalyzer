# GPU nearest-k candidate experiment

Status: implemented for comparison, **off by default**. On three measured
point clouds, the new normal/SOR path was about 1.9–2.4 times slower than
prepared CPU in paired trials. It won 0 of 60 comparisons. All 180 timed
outputs matched archived production output.

This follows the [block-hint experiment](gpu-knn-block-hints.md). Moving
actual candidate selection to the GPU reduced the CPU verification phase
relative to that experiment's block policy, but added much more GPU work
and readback data. The standard CPU implementation remains selected.

## Implementation and exactness

The opt-in `knnHint: "webgpu-knn"` path supports normal neighbourhoods up to
32 points and SOR up to k=31, whose local search needs k+1 points. Existing
`"cpu"` block hints and `"webgpu"` GPU block hints remain comparison controls.
Omitting the option retains prepared CPU.

Each immutable worker part provides its rounded, shifted f32 tree points,
split nodes and original-order queries. One physical WebGPU device owns a
cached pipeline. Each GPU invocation maintains up to 32 sorted candidates
and a 32-entry DFS stack, with at most 256 node visits. Workgroups have 64
invocations; batches are capped at 16,384 queries and device limits. A
bounded or rounded search can return imperfect candidates safely.

The result contains k tree-order u32 slots per query, never final distances.
The CPU checks that slots are distinct, in range and finite, with a finite
query and reference. Duplicate detection uses a small bit mask plus exact
comparison on collisions. It computes their maximum squared distance from
the original f64 coordinates. Any k distinct points supply an upper bound
on the true kth squared distance, even if they are poor candidates.

The CPU then runs the original near/far traversal and stable hit insertion.
It additionally excludes a far subtree only when its original f64 split
plane distance is strictly greater than this upper radius. Equality stays
eligible before the hit list fills; the original strict split test remains
after it fills. No GPU candidate is preinserted. Unlike block verification,
this traversal does not read a box for every node. Invalid, incomplete or
overflowing candidate rows use the original CPU query.

Final membership, distance bits and equal-distance ordering remain selected
on CPU. Normal PCA/orientation arithmetic and SOR reductions are unchanged.
Browser normals retain their existing part-local neighbourhoods; SOR still
completes cross-part queries on CPU.

`knn-hint-state` reports the selected backend, host phase timings, batches,
reference reuse, chosen GPU buffer bytes and `completeRows`. The last value
counts rows with k non-sentinel GPU slots; it is **not** a certificate that
those slots are exact neighbours or that every f64 validation succeeded.
The benchmark requires complete GPU rows for every query, rejects job-level
CPU fallback and checks final output independently.

Both GPU strategies retain their own immutable buffers under the same
leased CPU preparation. Source edits invalidate results; retired buffers
survive until users and pending operations drain. Partial creation/readback
failures drain before cleanup. A failed verification worker recovers through
the original CPU path, and a subsequent GPU job can retry. Pipeline/device
failures also recover on CPU. No new UI setting is exposed.

## Measured workloads and procedure

The inputs are unchanged measured compared clouds from
[real-source C2C preparation](c2c-real-source.md): PandaSet 019 (531,908
points), Autoware planning (878,920), and Tokyo Takanawadai (941,933). No
coordinates were added or newly thinned. Tokyo retains its previously
prepared original-point selection. The source-index splits are workload
geometry, not independent accuracy ground truth.

Each operation uses an archived production worker as a bitwise output
oracle, then a fresh current context sharing one immutable source between
prepared CPU, GPU blocks and GPU nearest-k candidates. Normals use k=24/up;
SOR uses k=24/ratio=1.0. There are six counterbalanced control orders, an
excluded first setup per control, five warm-up rounds and ten timed rounds.
Warm GPU reference reuse is required. Whole RPC timing includes submission,
readback, worker transfer, exact verification, joining and SOR indexing.
Normal-bit/complete-SOR-PLY comparison, export/removal and file/pool startup
are outside timing. The old block-hint reports are preserved separately.

There were 180 timed, 90 warm-up and 18 first-use current outputs, plus six
archived oracle outputs. All matched. Both GPU controls executed without
job-level fallback, and every candidate row was complete. Artifact, relevant
source, script, input, source-file and manifest hashes were checked before
and after each run. All three reports share the same source/build hashes.

No tests, builds or scans ran alongside timing. GPU vendor/architecture and
fallback status are the only GPU identifiers retained. This is one local
Windows/Chromium measurement with uncontrolled clocks, power, other
applications, scheduling and GC; it does not establish universal GPU limits.

## Results

Separate whole-RPC medians in milliseconds; the reduction is the median of
paired trial reductions, not a ratio of the separate medians. Negative means
slower.

| Source / operation | CPU | GPU blocks | GPU candidates | Candidate paired reduction | Faster |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet / normals | 437.10 | 597.75 | 861.20 | -91.80% | 0/10 |
| PandaSet / SOR | 379.00 | 499.75 | 789.05 | -104.96% | 0/10 |
| Autoware / normals | 483.80 | 697.80 | 1139.05 | -136.55% | 0/10 |
| Autoware / SOR | 507.95 | 714.35 | 1248.50 | -143.84% | 0/10 |
| Tokyo / normals | 511.00 | 709.95 | 1130.65 | -121.49% | 0/10 |
| Tokyo / SOR | 546.80 | 748.00 | 1278.05 | -132.11% | 0/10 |

Candidate versus block paired reductions were -42.39/-60.90%,
-55.79/-74.01%, and -59.24/-70.04% for normals/SOR respectively, also 0/60
wins. Neither GPU strategy beat CPU in these trials.

Host phase medians, measured separately:

| Source / operation | Block GPU phase | Candidate GPU phase | Block verification | Candidate verification |
| --- | ---: | ---: | ---: | ---: |
| PandaSet / normals | 38.45 | 421.65 | 555.45 | 468.50 |
| PandaSet / SOR | 40.60 | 418.30 | 412.85 | 323.60 |
| Autoware / normals | 68.20 | 635.60 | 619.95 | 478.90 |
| Autoware / SOR | 67.20 | 721.20 | 566.20 | 443.30 |
| Tokyo / normals | 62.80 | 626.10 | 642.50 | 501.55 |
| Tokyo / SOR | 64.45 | 724.85 | 605.05 | 487.95 |

GPU phase means host submission/search/readback elapsed time, including queue
and scheduling delays, not isolated shader execution time. Verification
includes candidate messages, exact search, normal PCA or SOR completion and
joining. Independently computed medians do not add to the whole-RPC median.

The normal result slots alone are 51.06/84.38/90.43 MB per job; SOR slots are
53.19/87.89/94.19 MB (decimal). This is 24 or 25 times the block-hint payload.
Candidate dispatch/readback totals are 35/57/60 batches versus 10/16/16 for
blocks, across 5/8/8 CPU parts. The new policy still repeats an authoritative
CPU k-neighbour traversal after the GPU search. Reducing its verification
time by roughly 87–141 ms relative to blocks did not offset the new GPU
phase's roughly 378–660 ms increase.

Chosen candidate GPU buffers retain 39,630,956 / 64,301,280 / 66,317,696
bytes, versus block buffers 12,770,348 / 20,878,432 / 21,886,640. Both sets
remain resident during this comparison. CPU retained preparation remains
39,070,832 / 64,332,800 / 68,491,676 bytes under the 128 MiB CPU budget.
These capacities exclude temporary candidate arrays, allocator overhead,
driver allocations and process/peak memory. They are not a memory saving.

Decision: keep this path opt-in. A useful next GPU design must eliminate
substantial CPU verification work or move more of the neighbourhood consumer
onto GPU with an explicit numerical contract. Merely improving a seed and
then repeating the full CPU consumer has not paid off in these workloads.

## Validation and reproduction

The native workspace's 371 tests pass, including candidate rows with exact,
reversed, arbitrary, duplicate, out-of-range and missing slots; k changes;
ties; large coordinates; signed zero; nonfinite inputs and overflow. Normal
bits and every local SOR field match the original. Native/wasm-target Clippy
with warnings denied, formatting and release WASM/TypeScript/Vite builds
pass. Twenty-two distinct browser regressions passed; the four GPU lease and
physical-adapter normal/SOR tests were rerun on the final radius traversal.
Six benchmark CLI guards and script syntax checks pass.

- [PandaSet report](../benchmarks/gpu/knn/real-source/windows-pandaset019-nearest-candidates.json)
- [Autoware report](../benchmarks/gpu/knn/real-source/windows-autoware-planning-nearest-candidates.json)
- [Tokyo report](../benchmarks/gpu/knn/real-source/windows-tokyo-takanawadai-nearest-candidates.json)
- [Benchmark script](../web/scripts/benchmark-gpu-knn-candidates.mjs)
- [GPU candidate implementation](../web/src/gpu-knn-candidates.ts)
- [Browser regressions](../web/e2e/gpu-knn.spec.ts)

Run from `web` with a separately preserved oracle distribution and the
measured inputs/manifests. The report path must be new:

```powershell
node scripts/benchmark-gpu-knn-candidates.mjs --baseline-dist ../notes/gpu-knn-baseline-20261012/dist --input ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../notes/NEW-pandaset-nearest-candidates.json --repeats 10

$env:CA_GPU_KNN_HARDWARE='1'
npx playwright test gpu-knn.spec.ts --workers=1
```

The local oracle distribution and measured inputs are ignored assets, not
committed binaries. The script validates matching provenance and refuses
existing report paths, incomplete GPU rows, changed artifacts and timed
job-level fallback.
