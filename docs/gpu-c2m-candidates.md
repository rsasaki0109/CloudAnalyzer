# GPU triangle candidates for retained C2M

Status: implemented as an internal, opt-in experiment; **off by default**.
It follows the adopted [CPU mesh/BVH cache](cpu-c2m-preparation.md).
Measured whole-RPC results on three existing real-source inputs did not
justify enabling GPU candidates: paired median reductions were −7.10%,
−10.56% and −4.34%, with only 8/30 faster pairs. All measured distance bits,
f64 summaries and scalar PLY bytes matched the archived retained-CPU build.

## Implementation

An eligible parallel C2M request can set internal `meshGpu: true`. The
ordinary UI keeps retained CPU C2M. The GPU path requires a physical WebGPU
adapter and the existing immutable CPU mesh/BVH preparation; its CPU cache
policy, resident workers and logical query slices remain unchanged.

One preparation lane exports the BVH and tree-order triangle corners.
Coordinates are shifted by one original f64 vertex before rounding to f32.
GPU nodes contain shifted bounds and child/leaf ranges; each triangle uses
three vec4 corners. A workgroup has 64 queries. Each query traverses the
rounded BVH, tests triangles and returns **one u32 triangle slot**, only
four readback bytes per point. Dispatches contain at most 65,536 queries,
including the final partial batch. A 32-entry stack and 4,096-node visit
limit bound individual searches. An unfinished or rounded search can still
supply a candidate; this is not a certificate of the nearest triangle.

Each CPU lane recomputes its candidate's closest point and squared distance
on original f64 geometry. This radius only adds strict `lower > upper`
exclusions to the original BVH traversal. The candidate is not inserted
as the final hit or warm guess. CPU retains the old Morton query ordering,
previous-triangle guess, child order and strictly closer triangle insertion.
Equal bounds stay eligible. Consequently the final selection, normal and
signed-distance arithmetic continue through the original CPU logic.
Invalid/nonfinite hints use the ordinary query; a bounded search whose
result exceeds its radius is repeated without the additional bound.
The measured and tested parity below establishes the checked cases rather
than a floating-point proof for arbitrary geometry.

CPU mesh/BVH reservations retain their aggregate **1 GiB** allowance.
Allocated GPU buffers have a separate **256 MiB aggregate** allowance across
active references, checked along with device/storage binding limits. Query,
output and readback buffers grow for reuse. Source geometry stays on the
device while the immutable mesh group lives. These allowances exclude
temporary exports, query snapshots, driver/allocator overhead and peak
process memory.

Whole finds on a shared mesh serialize, including failure cleanup. The
shared device also serializes error scopes and readback completion. Internal
leases keep retired references alive until pending finds drain. Failed GPU
preparation/readback destroys optional GPU buffers and retains the CPU cache
for fallback/retry. A failed verification lane drains the other slices and
falls back to fresh CPU C2M. Query edits/removal or mesh removal reject stale
results. Worker generation checks prevent use of lost native preparations.

Result `meshCompute` reports backend, fallback, host preparation/GPU/verify
durations, batches, GPU reuse, allocated bytes and `completeRows`. The latter
counts non-sentinel slots, not exact-nearest certificates. A CPU fallback is
not reported as successful GPU computation. Small serial jobs and requests
without retained preparation are outside this GPU experiment.

## Measured results

Full, unchanged input geometries use the existing
[measured-source manifests](c2c-real-source.md). Production meshing makes a
full-input 2.5D TIN, with its existing XY-duplicate/edge handling and no voxel
thinning. Even/odd source-index splits are workloads, not independent surveys
or accuracy ground truth. Query counts are 531,908 / 878,920 / 941,933;
triangle counts are 844,324 / 1,688,821 / 1,466,179.

| Source | Retained CPU RPC median ms | GPU + CPU RPC median ms | Paired reduction | GPU faster pairs |
|---|---:|---:|---:|---:|
| PandaSet scene 019 | 912.90 | 973.05 | −7.10% | 3/10 |
| Autoware planning map | 1,624.20 | 1,700.45 | −10.56% | 2/10 |
| Tokyo Takanawadai | 4,826.70 | 5,204.30 | −4.34% | 3/10 |

Reduction is `median(100 * (1 - pairedGpuRpc / pairedCpuRpc))`; it is not
the ratio of independently calculated medians. Negative means slower.
The measurements support leaving this experiment off on these inputs and
machine; they do not establish a universal backend ranking.

| Source | GPU host phase median ms | CPU verification phase median ms | Resident CPU BVHs / unchanged slices | GPU allocated bytes |
|---|---:|---:|---:|---:|
| PandaSet | 101.90 | 859.40 | 5 / 5 | 58,877,616 |
| Autoware | 249.35 | 1,408.75 | 4 / 8 | 116,190,688 |
| Tokyo | 475.40 | 4,496.05 | 4 / 8 | 105,503,872 |

GPU phase measures host packing, scopes, submission and readback, not device
timestamps. Verification includes worker transfers, ordering, native queries
and joining. Phase medians do not sum to the whole-RPC median. Tokyo GPU
phase ranged from 271.8 to 1,648.5 ms; CPU verification remained the largest
part of the operation. A smaller four-byte candidate result did not remove
enough end-to-end work to yield consistent gains.

Controls share one current worker context and the same retained CPU cache.
GPU reference buffers remain resident during CPU controls. Each source has
one first-use observation per mode, five excluded alternating warm-up
rounds, then ten alternating timed pairs. The first CPU observation builds
its native cache; the first GPU observation uploads geometry after that
cache exists. These observations are excluded and are not comparable cold
backend timings. Timed runs require warm CPU/GPU reuse, actual GPU dispatch,
one non-sentinel slot per query, unchanged slices/workers and valid budgets;
any job fallback fails the benchmark.

Whole RPC includes copies/transfers, GPU search/readback, CPU verification,
joining, summaries and returned arrays. Loading/meshing, startup, parity and
PLY exports are outside timing. All 60 timed current outputs, 30 warm-up
outputs and six first observations matched three archived output oracles.
Relevant source, production worker/pool/WASM, benchmark script, input,
source and manifest hashes are checked before/after; all three reports share
the same current build and script. No builds/tests/scans ran alongside
timing. Clocks, power, other applications, scheduling and GC were uncontrolled.
Adapter metadata records only vendor/architecture/fallback.

## Validation and reproduction

Rust workspace tests: **373 passed**, plus native/WASM Clippy with warnings
denied and rustfmt. New candidate tests cover random, georeferenced, tiny,
large, nearly collinear and oppositely oriented geometry, arbitrary/nearest/
sentinel slots, signed/unsigned results, export/no-export and malformed counts.
Production SIMD WASM and TypeScript/Vite builds pass.

Browser checks: **11 passed**, covering CPU cache behaviour and physical GPU output parity,
overlapping queries, signed ties, final batch tails, reuse, exact statistics
and PLY, injected readback/verification failures, edits/removal, retry and
aggregate GPU quota/partial-allocation rollback.
Synthetic geometry is used only for these correctness/failure checks.

- [PandaSet report](../benchmarks/gpu/c2m/real-source/windows-pandaset019-triangle-candidates.json)
- [Autoware report](../benchmarks/gpu/c2m/real-source/windows-autoware-planning-triangle-candidates.json)
- [Tokyo report](../benchmarks/gpu/c2m/real-source/windows-tokyo-takanawadai-triangle-candidates.json)
- [Benchmark script](../web/scripts/benchmark-gpu-mesh-candidates.mjs)

Build production SIMD WASM and the web app, save an archived retained-CPU
production `dist`, then run sequentially from `web` on physical WebGPU:

```powershell
$env:CA_GPU_MESH_HARDWARE = '1'
npx playwright test e2e/mesh-preparation.spec.ts e2e/gpu-mesh.spec.ts --workers=1
node scripts/benchmark-gpu-mesh-candidates.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --baseline-dist ../notes/ARCHIVED-retained-cpu-dist --output ../notes/NEW-c2m-triangle-candidates.json
```

Use the existing Autoware/Tokyo source preparation directories for other
runs. Output parents must exist and reports cannot be overwritten.
`--repeats` accepts 1..100 and defaults to 10. Do not run other builds,
tests or scans alongside the benchmark.
