# GPU-selected block hints for exact normals and SOR

An internal prototype now uses WebGPU to select small spatial blocks for the
existing prepared normal/SOR workers. It is **off by default**: measured whole
operations were slower than the current prepared CPU implementation on all
three source geometries. The CPU-only version of the same hint policy was
also slower, so neither policy is adopted as a production speedup.

The subsequent [nearest-k candidate experiment](gpu-knn-candidates.md) selects
actual point slots and uses a lighter CPU radius traversal. Its verification
phase improved relative to blocks, but GPU search/readback cost made whole
operations slower again; both paths remain opt-in.

This is actual GPU candidate-range selection, not GPU PCA, final nearest-k
membership, SOR thresholding or complete normal/SOR computation. It follows the
[GPU opportunity measurement](gpu-opportunities.md), which identified these
operations as substantial CPU work but did not establish a GPU speedup.

## Prototype and exact arithmetic

Each immutable prepared part optionally retains exact f64 subtree boxes. It
exports rounded, shifted f32 queries and five-word split nodes only when GPU
preparation is requested. The GPU walks the tree towards each query until a
child would have fewer than k points, and returns one u32 node number per
query. Queries and split buffers remain resident and are reused; output and
readback buffers grow only as needed. GPU work uses 64-invocation groups and
batches capped at 65,536 queries and device limits. No shared CPU memory is
required.

The CPU accepts a block only if it contains at least k distinct tree slots,
at most 64 points, k is in 1..32, and both reference/query coordinates and
candidate squared distances are finite. It computes the k-th squared distance
from the block's **original f64 coordinates**. This supplies an upper bound
on the true k-th distance even when the rounded GPU chooses the wrong block.
An invalid/oversized/insufficient/nonfinite hint uses the original complete
scalar search.

Accepted hints do not preinsert neighbours. The CPU still starts at the root
and uses the original near/far traversal and insertion order. Exact subtree
boxes can exclude a node only when its squared lower bound is strictly greater
than the safe upper radius or the current full list's worst distance. Equality
remains eligible; the historical strict split-plane predicate remains in place.
Final neighbour indices, tie order and squared-distance bits are preserved.

Normals retain the existing part-local neighbourhoods, original PCA accumulation
and orientation arithmetic. SOR retains its k+1 local hits, original open-point
order/reach masks and unchanged cross-part exact queries and merging. The
normal k=24 probe uses k=24 hints; SOR k=24 uses k=25 hints.

The optional CPU control performs the identical rounded block walk and exact
verification without GPU submission, readback or hint messages. It helps
distinguish the hint policy's CPU cost from GPU overhead.

## Cache, failures and scope

The existing immutable neighbourhood lease owns the optional bounds and GPU
parts. Source edits/removal and worker-generation changes retain the existing
invalidation rules. Retired groups keep their buffers while any calculation
uses them. Submitted native/GPU work is drained before failure recovery and
buffer release. Partial GPU preparation is also drained and disposed.
Verification-worker failure recovers through the original CPU path; a later
request can prepare GPU state again. Source changes reject the in-flight result
before normals or filtered output are committed.

Optional bounds are included in the existing 128 MiB retained-vector budget.
The measured retained CPU preparation totals are 39,070,832 / 64,332,800 /
68,491,676 bytes for PandaSet / Autoware / Tokyo. Warm GPU allocations total
12,770,348 / 20,878,432 / 21,886,640 bytes, including reusable output/readback
capacity. These are tracked vectors and GPU buffers, not process/peak-memory
measurements. Retired overlapping jobs, binding copies, allocator slack and
retained WASM pages remain relevant.

Internal requests accept `knnHint: "cpu" | "webgpu"` for normals or SOR;
omission retains the original prepared CPU search. `knn-hint-state` reports
the last requested hint operation's actual backend, fallback, k, worker count,
preparation/GPU-host/exact-verification intervals, reuse and buffer bytes.
The prototype supports normal k up to 32 and SOR k up to 31. Ineligible or
unavailable paths recover on CPU; benchmarks reject timed fallback. No new
GUI backend is enabled.

## Real-source comparison

The unchanged measured-source files and provenance are documented in
[source C2C](c2c-real-source.md). Compared clouds have 531,908 / 878,920 /
941,933 points and are fully loaded with `keepEvery: 1`. No generated points,
offsets or additional thinning are introduced. Tokyo retains its previous
0.1 m original-point preparation. These source-index splits are workloads,
not independent surveys or accuracy/change ground truth.

For each method, an immutable archived pre-prototype production worker supplies
the output oracle. All current modes must match every normal f32 bit or every
byte of the complete SOR PLY. A fresh current context then shares one prepared
source across the original CPU, CPU-hint and GPU-hint controls. All controls
retain the optional bounds/GPU buffers after warm-up; original CPU ignores
them. This is not a comparison of isolated memory footprints.

One first-use observation per mode is excluded, then five warm-up rounds
precede ten timed rounds using six alternating execution orders. Loading,
pool startup, validation/export/hashing and removal are outside timing.
Timings cover the complete worker RPC, including transfer/readback, exact CPU
verification and joining; SOR also includes filtered-cloud indexing/display
preparation. GPU buffers must be reused in timed trials and fallback fails
the run.

The same production SIMD WASM, worker and pool-worker artifacts are used in
all three reports. The machine is the existing Windows/i7-9750H environment
with Chromium 153 and a physical GPU. Public adapter metadata is limited to
vendor, architecture and fallback status. Builds/tests/scans finished before
timing; reports ran sequentially. Other applications, clocks, power, scheduling,
GC and retained heaps are uncontrolled. Large warm-up outliers and within-run
variation are preserved; absolute timings are not controls for historical runs.

Each table entry is a separate median in milliseconds. Paired reduction is
`median(100 * (1 - candidateMillis / originalMillis))`, so negative means slower;
it is not the percentage difference between the timing medians.

| Source / operation | Original CPU | CPU hints | GPU hints | GPU paired reduction | GPU faster |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet / normals | 457.75 | 583.85 | 603.20 | -33.70% | 0/10 |
| PandaSet / SOR | 527.75 | 648.75 | 663.25 | -26.41% | 0/10 |
| Autoware / normals | 629.25 | 844.85 | 850.10 | -37.16% | 0/10 |
| Autoware / SOR | 638.85 | 797.55 | 814.50 | -33.32% | 0/10 |
| Tokyo / normals | 575.70 | 756.10 | 798.20 | -43.76% | 0/10 |
| Tokyo / SOR | 561.00 | 693.80 | 765.65 | -36.95% | 0/10 |

CPU hints' paired reductions versus original CPU are -27.90/-24.59%,
-32.24/-27.45% and -32.33/-29.65% respectively. Only one of their 60 timed
comparisons was faster. Thus the new CPU hint/box policy already adds cost;
GPU transfer is not the only cause of the regression.

| Source / operation | GPU-host interval median | Exact verification/join interval median |
| --- | ---: | ---: |
| PandaSet / normals | 40.15 ms | 560.35 ms |
| PandaSet / SOR | 39.65 ms | 544.65 ms |
| Autoware / normals | 65.65 ms | 786.30 ms |
| Autoware / SOR | 62.60 ms | 646.15 ms |
| Tokyo / normals | 67.30 ms | 733.30 ms |
| Tokyo / SOR | 65.80 ms | 610.55 ms |

GPU-host time includes queue/submission, readback and browser scheduling; it
is not a device timestamp or isolated shader execution. Verification includes
worker messages, exact neighbourhood work and joining, plus PCA for normals
and cross-part work for SOR. SOR indexing follows this interval. Separate
medians do not reconstruct complete RPC wall time.

The current broad block hint requires scanning its original points to obtain
an upper radius, then performing exact root traversal with extra box tests.
The measurements show this policy does not reduce enough work to repay its
cost on these geometries. It stays a comparison-only prototype. A future GPU
experiment should try tighter actual nearest-k candidates that substantially
reduce exact CPU work, or another measured hotspot such as M3C2. Merely moving
this cheap block walk to the GPU is insufficient. These results do not rule
out different GPU nearest-k algorithms or other workloads/hardware.

## Validation and reproduction

All 180 timed RPCs, 90 warm-up RPCs and 18 setup RPCs match six archived output
oracles. All timed GPU calls use physical WebGPU without fallback and reuse
their reference buffers. Both builds, source code, input, source geometry,
manifest and script hashes were checked against the completed reports.

Native validation passes 371 tests, including exact nearest-k indices/distance
bits/tie order for every node seed and invalid seeds, normal/SOR parity across
ties, signed zeros, extreme magnitudes and nonfinite fallback. Twenty distinct
browser tests pass: GPU lease draining/retirement, actual physical-GPU normals
at k=3/24/32 in both orientations, SOR exports, verification-lane failure and
GPU retry, existing neighbourhood lease/edit/removal/generation controls,
joining/bounds, compute caches and physical C2C regression checks. Native and
SIMD-WASM Clippy with warnings denied, rustfmt, TypeScript/Vite and script
syntax checks pass. Six invalid-option/overwrite controls reject cleanly.

- [PandaSet report](../benchmarks/gpu/knn/real-source/windows-pandaset019-block-hints.json)
- [Autoware report](../benchmarks/gpu/knn/real-source/windows-autoware-planning-block-hints.json)
- [Tokyo report](../benchmarks/gpu/knn/real-source/windows-tokyo-takanawadai-block-hints.json)
- [Benchmark script](../web/scripts/benchmark-gpu-knn.mjs)

Run from `web`, with a current production SIMD build, a preserved pre-prototype
build and the existing measured inputs. Choose a new output path:

```powershell
node scripts/benchmark-gpu-knn.mjs --baseline-dist ../notes/gpu-knn-baseline-20261012/dist --input ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../notes/NEW-pandaset-block-hints.json

$env:CA_GPU_KNN_HARDWARE = '1'
npx playwright test gpu-knn.spec.ts --workers=1
```

`--repeats` accepts 1..100, default ten. Inputs must match measured manifests
and contain 200,000..1,000,000 points. A physical adapter is required. Existing
reports are never overwritten. Keep measurements sequential and separate from
builds/tests/scans.
