# Retained CPU mesh/BVH for repeated C2M

Status: adopted for eligible parallel C2M jobs. Repeating a calculation on an
unchanged mesh retains its exact f64 mesh/BVH and transfers only query points.
In the final three measured-source comparisons, paired whole-RPC time
reductions were 60.73%, 70.11% and 28.76%, with 30/30 faster pairs. Every
distance bit, f64 summary and scalar PLY matched archived production output.

This is CPU preparation reuse. It follows the
[C2M preparation probe](gpu-opportunities.md#c2m-preparation-probe), which
found substantial copying and BVH construction before distance queries.
The subsequent [GPU triangle-candidate experiment](gpu-c2m-candidates.md)
uses this retained CPU baseline. It remains off because measured whole-RPC
improvements were inconsistent.
The later [cached traversal bounds](cpu-c2m-stack-bounds.md) are adopted:
whole retained-CPU calls improve by 23.80–30.38% across two measured passes,
with 60/60 faster pairs and identical geometry capacities and outputs.

## Implementation and exactness

`PreparedMesh` owns the validated source mesh and immutable BVH storage.
The ordinary borrowed `MeshBvh` API remains available. Both paths use the
same construction, tree traversal, triangle tests, normals and signed
distance arithmetic. The prepared view borrows its owned storage without
self references or unsafe code. Every batch starts with a fresh Morton
order, guessed triangle and traversal stack, just like the old batch.

Triangle order now uses a separately allocated, exact-size u32 vector. It
does not retain the wider temporary centroid vector through in-place
iterator collection. Centroid scratch drops before corner allocation.
This allocation change also applies to the current fresh control; the
measurement below therefore compares reuse against the updated fresh path.
It does not isolate this allocation improvement against the archived build.

Pool workers retain `MeshPrepared` objects under a generation-bound job ID.
One idle mesh group can be reused across query clouds and signed/unsigned
switches. Queries themselves are not retained. A worker's lost state cannot
be reused on a replacement generation. Preparation and query failures drain
before retirement/release; unchanged sources then recover via fresh C2M.
Later requests can rebuild the cache.

The aggregate C2M reservation budget is **1 GiB**, including retired groups
still held by active leases. Preflight reserves, per worker,
`24 * vertices + 144 * triangles + 4096` bytes. The estimate covers owned
source coordinates/indices, exact order/corners and node capacity growth.
Actual retained vector capacities must fit the reservation; otherwise the
preparation retires and calculation falls back. Worker count is capped by
the available reservation and ordinary pool/query limits. A mesh exceeding
the allowance uses the fresh path. The budget covers this cache, not all
application memory, temporary copies or allocator overhead.

Crucially, reducing resident workers does **not** change logical query
slices. The original contiguous slices, their Morton orders and per-slice
guesses remain fixed, queued across fewer resident BVHs when necessary.
Changing slices could change which equally distant, oppositely oriented
triangle supplies a signed distance. Keeping them unchanged preserves the
existing tie/sign behaviour.

Query edits/removal reject active results. Removing a mesh retires all of
its leased groups, but immutable worker geometry survives until users drain.
Selecting a new mesh replaces the idle group. Query-only changes need not
invalidate an unchanged mesh's BVH. No additional UI control is exposed.
Internal `meshCache: false` requests use fresh calculation;
`mesh-cache-state` and result fields `meshCached`, `meshCacheReused`,
`meshCacheBytes`, `meshSlices` and `meshCacheFallback` support diagnostics.
Serial jobs below the existing parallel threshold retain the original path.

## Measured geometry and procedure

The unchanged inputs from [real-source preparation](c2c-real-source.md) are
used in full. Each reference is meshed by the production 2.5D Delaunay
operation, with automatic edge rejection and no voxel thinning. Meshing is
outside timing. Mesh topology and shifted display-coordinate hashes must
match the archived oracle. The TIN is derived from measured source vertices;
these source-index splits are workload geometry, not independent surveys.

| Source | Query points | Mesh vertices | Retained triangles |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 | 501,097 | 844,324 |
| Autoware planning | 878,920 | 873,771 | 1,688,821 |
| Tokyo Takanawadai | 941,933 | 893,804 | 1,466,179 |

An archived pre-change production worker provides unsigned f32 distance,
f64 summary and scalar PLY oracles. Fresh and retained controls then run in
separate, prewarmed current contexts on identical inputs/TINs, alternating
execution order. They have one excluded first-use observation each, five
warm-up pairs and ten timed pairs. Reuse is mandatory in timed retained
calls; fallback, incomplete results, changed slices or budget violations
fail the benchmark. Whole RPC includes copies/transfers, query ordering,
search, joining, summaries and returned arrays. Loading, meshing, pool
startup, parity/export checks and cleanup are excluded.

Both contexts coexist; cached BVHs remain resident during fresh-control
timing. Fewer resident workers are part of the bounded cache policy, so this
is not an isolated same-concurrency cache comparison. No tests, builds or
scans run alongside measurement. Clocks, power, other applications,
scheduling and GC remain uncontrolled. The result applies to these repeated,
unchanged-source workloads on this machine, not every C2M geometry.

## Final results

Separate whole-RPC medians in milliseconds. Paired reduction is calculated
per alternating pair, then summarized; it is not the ratio of these medians.

| Source | Fresh | Retained | Paired reduction | Faster | Resident workers / logical slices |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet | 3816.55 | 1340.05 | 60.73% | 10/10 | 5 / 5 |
| Autoware | 4400.40 | 1310.75 | 70.11% | 10/10 | 4 / 8 |
| Tokyo | 6978.15 | 5011.50 | 28.76% | 10/10 | 4 / 8 |

PandaSet has appreciable variability: fresh trials span 2367.10–4911.90 ms,
retained trials 928.90–1804.50 ms. Autoware ranges are 4286.10–4950.90 and
1189.00–1558.50 ms; Tokyo ranges are 6637.60–7695.10 and
4799.90–5489.50 ms. The paired wins are consistent despite that variability.

First-use observations (fresh/retained) were 2202.00/2295.20 ms,
4971.00/4591.00 ms and 8273.90/9444.90 ms. These are single excluded samples,
not cold-app medians. **The gain is established for reuse; initial BVH
preparation remains expensive and was slower in two of these observations.**

Retained source/BVH capacities are 668,045,480 / 1,056,643,136 / 930,324,512
bytes. Reservations are 668,065,400 / 1,056,659,296 / 930,340,672 bytes,
all under 1,073,741,824 bytes. These do not describe process or peak memory.
The memory allowance trades retained geometry against query parallelism.

## Rejected smaller allowance

The first implementation used a 512 MiB aggregate allowance, selecting
4/2/2 resident workers while preserving 5/8/8 logical slices. Paired reductions
were 47.97% (10/10 wins), 50.12% (10/10), and **-12.27% (1/10)** for Tokyo.
Its Tokyo query phase lost too much parallelism. That allowance was rejected;
the final policy permits 5/4/4 resident workers under a 1 GiB limit.

These are separate counterbalanced runs with different idle memory and
uncontrolled machine conditions; their absolute times should not be
subtracted to isolate the effect of raising the budget. Both initial and
final reports are preserved. The initial production build, script and budget
module were saved locally before changing the policy.

## Validation and reproduction

The final native workspace's 372 tests pass. New preparation tests cover
signed/unsigned switches, equal/degenerate triangles, large coordinates,
signed zero, changing/empty query batches, exact retained capacity bounds and
source destruction after snapshot creation. Prepared f64 distance bits match
fresh calculation. Native and wasm-target Clippy with warnings denied,
formatting and release WASM/TypeScript/Vite builds pass.

Six browser tests pass on the final policy: aggregate reservations including
overlap, drained failure/oversize/generation handling, concurrent signed C2M,
source edit, query removal, mesh removal and lane-failure recovery/retry.
The benchmark's six CLI guards and syntax check pass.

All 60 timed final current outputs, 30 warm-up outputs and six first-use
outputs matched three archived oracle outputs. Exact f32 distance bits, f64
summaries and complete scalar PLY bytes were checked outside timing. Every
timed retained call reused its geometry without fallback. Relevant source,
production artifacts, script, input/source and manifest hashes were checked
before/after; all final reports share the same source/build/script hashes.
The separate initial allowance runs also passed parity.

- Final: [PandaSet](../benchmarks/cpu/c2m/real-source/windows-pandaset019-mesh-preparation-final.json), [Autoware](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-mesh-preparation-final.json), [Tokyo](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-mesh-preparation-final.json)
- Initial 512 MiB: [PandaSet](../benchmarks/cpu/c2m/real-source/windows-pandaset019-mesh-preparation.json), [Autoware](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-mesh-preparation.json), [Tokyo](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-mesh-preparation.json)
- [Benchmark](../web/scripts/benchmark-mesh-preparation.mjs), [cache lifecycle](../web/src/mesh-preparation.ts), [browser tests](../web/e2e/mesh-preparation.spec.ts), [native tests](../rust/crates/ca-core/tests/mesh_preparation.rs)

Run from `web`, using separately preserved pre-change oracle artifacts and
matching full measured inputs/manifests. Existing report paths are refused:

```powershell
node scripts/benchmark-mesh-preparation.mjs --baseline-dist ../notes/c2m-preparation-baseline-20261012/dist --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../notes/NEW-pandaset-mesh-preparation.json --repeats 10
npx playwright test mesh-preparation.spec.ts --workers=1
```

Oracle/input files are ignored local assets, not committed binaries. The
initial 512 MiB report hashes describe that earlier policy and build, rather
than the final source tree. GPU model identifiers are not recorded.
