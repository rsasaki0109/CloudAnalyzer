# Exact-tree preparation for GPU C2C verification workers

The experimental **WebGPU + CPU workers** mode now transfers the already
built exact reference tree to its verification workers. Each worker imports
and validates it in O(n), then derives its own exact boxes, parents and leaf
certificates. This replaces repeating O(n log n) median partitioning in every
worker. GPU candidate slots, original f64 arithmetic, query order and the
existing reference/job lifetime rules are preserved.

Current imports also [retain validated finite subtree extrema](gpu-c2c-import-bounds.md)
for certificate construction and finiteness checks. This reduces worker
preparation without claiming a consistent full-call speedup. The measurements
below predate it; their benchmark explicitly uses `gpuReuseImportBounds: false`.

`gpuVerifySnapshot` defaults to `true`; `false` retains original-XYZ rebuilding
for comparisons. Each preparation mode owns a separate cache, so switching
controls cannot accidentally measure a tree prepared by the other method.
The UI still offers up to four verification workers. Ordinary CPU and serial
WebGPU remain available as before.

`GpuC2cReference.snapshot()` exports a portable little-endian byte sequence:
four-byte `CAKD` magic, u32 version 1, point count and node count, followed by
tree-ordered f64 XYZ triples, u32 original indices and 24-byte node records
(u32 start/end/right/axis and an f64 split). It never copies Rust struct
padding or GPU-rounded f32 values. Every coordinate and split bit is retained,
including signed zero and nonfinite payloads. Nonfinite data still takes the
existing complete-search path without certificates.

`fromSnapshot()` checks the version, exact payload length, the two-million
point limit, original-index permutation, point ranges, axes, balanced preorder
topology and split partitions before any search. Balanced half-range splits
bound validation recursion depth even for invalid input. Partition checks use
the same total order as tree construction. Derived bounds/certificates are
computed locally rather than trusted from the serialized input. A rejected
snapshot uses the existing whole-batch serial recovery; later jobs can retry.

The payload is `16 + 28 * points + 24 * nodes` bytes per prepared lane,
compared with `24 * points` for rebuilding. This exchanges a larger transfer
for less worker computation. Workers still own independent full trees;
resident reference memory is not shared. No SharedArrayBuffer or special
hosting headers are required. `compute.verifySnapshot` identifies the method
and `verifyReferenceBytes` records the payload size per lane.

## Reproduce first-use comparisons

Use the unchanged inputs and manifests from
[real-source C2C evaluation](c2c-real-source.md). From `web/`:

```powershell
node scripts/benchmark-c2c.mjs --compare-verify-setup --verify-workers 4 --hardware --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../notes/gpu-verifier-setup-new.json
```

The pool is prewarmed with its excluded startup time recorded. Every pair
reloads identical reference bytes under a fresh identity, releases the previous
worker caches, and warms only the main GPU reference/device with an excluded
serial run. Both timed preparations start with empty worker reference caches.
Five preparation warm-up pairs precede ten alternating timed pairs. Each pair
also records cache-reusing runs in reverse order. Both method caches coexist
until that pair's reference is retired.

Cold operation times include exporting, copying/transferring and
importing/rebuilding the worker trees, query snapshots, GPU readbacks, exact
verification, joining and summaries. They exclude loading, pool startup and
main GPU reference construction. These are first-use worker comparisons, not
fully cold application/backend measurements. `verifyPrepareMillis` isolates
worker preparation but overlaps GPU dispatch; phase intervals must not be
summed. Cold/warm pairs require identical displayed f32 distance bits, reported
f64 statistics and certificate counts; cold results must also match serial
verification. GPU and worker fallbacks are rejected. Reports record unchanged
source/manifest hashes plus script and production artifact hashes, omit GPU
model fields and refuse to overwrite existing files.

## Measured first-use results

All prepared points were used (`keepEvery: 1`) on the i7-9750H / NVIDIA GPU
machine with Chromium 153.0.8010.12. Each row below has ten alternating pairs
with four verification workers. Times include the complete first-use worker
operation after the main GPU reference/device and CPU pool are prepared.
The paired reduction is the median of `(rebuild - snapshot) / rebuild`, not
the ratio of the two independent medians.

| Source and report | Compared points | Rebuild median | Snapshot median | Paired reduction |
| --- | ---: | ---: | ---: | ---: |
| [PandaSet 019](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-verifier-setup-4.json) | 531,908 | 570.20 ms | 300.90 ms | 49.67% |
| [Autoware planning](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-verifier-setup-4.json) | 878,920 | 459.00 ms | 234.35 ms | 47.82% |
| [Tokyo Takanawadai](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-verifier-setup-4.json) | 941,933 | 516.40 ms | 267.55 ms | 49.62% |

The worker preparation intervals, which include exporting, transferring,
validating/importing or rebuilding the full reference, show the main benefit:

| Source | Rebuild preparation median | Snapshot preparation median | Paired reduction |
| --- | ---: | ---: | ---: |
| PandaSet | 373.30 ms | 86.20 ms | 77.48% |
| Autoware | 302.75 ms | 81.15 ms | 72.98% |
| Tokyo | 347.90 ms | 97.00 ms | 72.17% |

Payload sizes per worker increased from 12,765,792 to 16,832,040 bytes,
21,094,104 to 27,755,508 bytes and 22,606,392 to 29,519,844 bytes respectively.
This includes the topology and original-index permutation needed to preserve
candidate slots without sorting. Preparation time still decreased despite
those larger transfers.

## Warm execution and repeated measurement

The same pairs also measured both caches after preparation. This stage does
not demonstrate a warm execution improvement:

| Source | Warm rebuild median | Warm snapshot median | Paired time reduction |
| --- | ---: | ---: | ---: |
| PandaSet | 195.45 ms | 187.35 ms | 3.09% |
| Autoware | 150.10 ms | 158.35 ms | −4.38% |
| Tokyo | 153.05 ms | 160.85 ms | −6.06% |

Negative reductions mean increased time. An additional
[Autoware run](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-verifier-setup-4-repeat.json)
retained the cold benefit: 465.15/253.85 ms, a 47.30% paired reduction;
preparation was 308.30/85.20 ms, a 71.61% paired reduction. Its warm times were
158.05/164.20 ms, a −3.75% paired reduction. The warm penalty on Autoware thus
persisted in the repeated run. Trees, query splitting and exact search
arithmetic are identical; these runs do not separate allocation/cache effects
from host workload variation. The demonstrated improvement is first-use worker
preparation. The rebuilding control remains available for repeated-reference
evaluation.

All 40 timed cold pairs, 40 corresponding warm pairs and 20 preparation
warm-up cycles (each with a cold and a warm pair) passed displayed-distance bit, reported f64 statistic
and certificate checks. Each cold result also matched the excluded serial
verification call. Certificate counts stayed at 235,714, 421,096 and 481,955
for PandaSet, Autoware and Tokyo. No GPU or verification-worker fallback was
accepted. The inputs are disjoint splits of measured source geometries, not
independent surveys or ground-truth accuracy evaluations.

Builds and browser suites finished before sequential measurements. Another
workspace's Rust builds/GPU tests were observed immediately before timing;
other applications, clocks and power were uncontrolled. The two cache formats
coexist during each comparison. Results describe these development workloads
on one machine and are not isolated hardware throughput or portable speed
guarantees.

The three initial reports retain measurement script hash
`88209b84e2dabc497e7e3cac201d63cee231ec06361baaef891e0fee0b3a759c`.
Their `serialWarmMillis` field names the excluded serial preparation call,
including main GPU reference construction; it is not a warm serial control.
The repeated run and current generator use the clearer `serialPrepareMillis`
and record `serialCompute`, with script hash
`659d8a498b1171440c73b0451c8f8e76d8f9e3b16e640b6f2d94f6a20f9873c8`.
This output-only correction does not change the measured kernels or controls.
All four reports identify the same tested production artifacts:

- Main worker: `38a915e1d600f48fde8f7447fdbf0668dbd1f08e31cbb054e169b57ffc8ad8ad`.
- Pool worker: `05c48e45dd14787e5de0b78202aef5ee39339378858f2eef2400bee66e862720`.
- WASM: `60be266c2b095017cc9ff6e0f388bda5561e06942155fda7e245e30c0bd90dac`.

## Verification

Native checks passed 53 distinct cases across the WASM crate, k-d tree,
distance and associated integration tests. The final build passed 73 browser
cases: 25 each on software and physical GPU, plus 23 CPU distance/M3C2/Map
quality cases. New checks exercise exact snapshot round-trips, original origins
and GPU packing, tiny/odd/coincident inputs, signed zero and nonfinite payloads,
f64 distance bits/certificates, malformed headers/ranges/indices/topology/splits,
and worker rejection/recovery/retry. Existing checks cover lane growth,
simultaneous jobs with source edits/removal, partial batches, GPU loss and UI
exports. Rust formatting, native/WASM Clippy with warnings denied, TypeScript
and the production WASM/Vite builds passed. Invalid CLI combinations and the
existing-report overwrite guard were rejected without creating/modifying
reports. Public file-content and filename scans found no omitted GPU model.
