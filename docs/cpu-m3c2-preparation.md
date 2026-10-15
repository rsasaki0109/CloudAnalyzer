# Exact tree transfer and reuse for browser M3C2

The automatic M3C2 path now builds the reference and compared k-d trees
once on the analysis worker, transfers their exact snapshots to pool lanes,
and retains one immutable input pair for subsequent calculations. The
analysis worker computes one core lane alongside the pool. This is enabled
by default after measured-source comparisons of cold and warm calls.
Core selection, normals, distances, confidence bounds, result attributes
and output LOD indexing are recomputed for every request.

An optional [bounded normal cache](cpu-m3c2-normals.md) can retain the exact
fits for subsequent projection changes. It remains disabled by default;
the comparisons below use fresh fitting.

The current default also uses [shared reference searches](cpu-m3c2-shared-search.md)
when fitting fresh normals. The recorded measurements below use separate
searches; their benchmark explicitly selects `sharedSearch: false`.

## Numerical order and ownership

The standard `KdTree` now exposes the versioned, validated snapshot format
already used by GPU candidate verification. Snapshot bytes preserve all
coordinate/split f64 bits, point order, original indices and tree topology.
Import checks the exact length and point budget, index permutation, axes,
balanced preorder partitions and split ordering before traversal. Candidate
verification delegates to this same reader and rebuilds its certificates.

M3C2 radius queries can return tree slots in exactly the existing unsorted
hit order. Normal fitting and cylinder accumulation access the tree's own
coordinates in that order, so neither a second original-coordinate buffer
nor a rounded representation is needed. The arithmetic and PCA remain
unchanged. The prepared WASM helper collects only the distance/LoD95 pairs
required by the browser, avoiding an unused per-core normals/result array.

The core plan captures selected original positions, colors and attributes
before the first yield. A prepared pair owns both trees independently of
the source `Cloud` objects. Changing radii, core spacing, pruning or cyclic
versus contiguous core layout can reuse the pair when lane count remains
the same. The geometry, input direction and pool layout identify the cache.
Coordinate changes, source removal, a different pair/layout or a lost worker
retire it. Only one idle pair is retained; computations already in flight
hold leases until all dispatched operations have drained.

Each pool lane receives one transferred snapshot pair, exported immediately
before dispatch. No raw-position arrays or JS snapshot copies are retained
by the analysis cache. Cold searches are queued on the exact worker
generation before their preparation acknowledgement, followed by synchronous
main computation. The worker installs prepared state synchronously after
WASM initialization. Preparation and search responses are both awaited with
`Promise.allSettled`, as are all lanes. Import, search or main failure rejects
the entire operation and retires the preparation. A late failure or source
edit cannot free trees belonging to another active lease. No partial result
cloud is published; result lengths are checked before consuming the plan.

`preparedTrees: false` selects the preceding copying implementation, whose
resident lane is controlled by `mainWorker`. `mainWorker: false` also retains
the pool-only copying path. `parallel: false`, small core sets or an input
pair outside the scheduling estimate use the serial implementation.
`M3c2Output.preparedTrees`, `treeCacheHit`, `workers` and `mainWorker` expose
the actual choice for internal measurements. A hit includes reuse of an
existing in-flight preparation; its import responses still must succeed.
Experimental [radius pruning](cpu-m3c2-radius.md) remains false by default.

## Input-storage estimate

The same 256 MiB scheduling budget, minimum 10,000 cores per lane, 1,024-core
cyclic blocks and total cap `min(4, poolSize())` remain in effect. The old
control allows 128 bytes/input point/pool lane, or 80 for main construction.
Prepared jobs account for construction, snapshot export and concurrent
imports, with `n` the sum of both input point counts and `p` the pool lanes:

```text
estimate = n * max(80, 108 + (p - 1) * 104, 36 + p * 104)
```

Coordinates, indices and decoded nodes occupy at most about 34 bytes/point
at these sizes; 36 allows header and capacity slack for retained main trees.
The 104-byte import allowance includes JS and binding snapshot copies,
decoded trees and the validation bitmap. The 108-byte export allowance
includes the resident trees, snapshot buffers in WASM and their JS copies
while earlier lanes import. Both source trees are built before dispatch,
and each snapshot pair is transferred before exporting the next pair.

This admits three total lanes for the full PandaSet pair (259,571,104
estimated bytes), and two for Autoware (246,097,740) and Tokyo (263,741,240).
The prior control admitted two, one and one respectively. No input thinning
or extra CPU lanes beyond the existing cap are used to obtain these counts.

This estimate is not a total memory cap. Resident sources, core plans and
attributes, core bindings, output buffers, allocator/GC overhead, retained
WASM heap pages, caches belonging to other workflows and concurrent jobs
are excluded. The cold allowance is used even on warm cache hits.

## Measured results

Measurements use supplied real point clouds with unchanged coordinates:
[PandaSet, Autoware and Tokyo provenance](c2c-real-source.md). These even/odd
source subsets are not independent surveys or change-detection ground truth.
All cases use a 0.4 m normal radius, 0.25 m projection radius, 0.5 m half-depth
and 0.25 m core spacing. Core selection retains original measured points.
Full cases load every point. The separate smaller PandaSet case selects
every sixth original file point (88,652 per cloud), rather than generating
coordinates. It retains four lanes in both modes.

Each independent sequence excludes five warm-up trials and times ten trials.
A fresh reference identity before each trial forces cold preparation without
changing geometry. Control/prepared order alternates, and the warm prepared
call immediately follows cold. Complete RPC timings include core selection,
attributes, cold construction/export/import, fresh computation, joining and
output LOD indexing. Loading, pool startup, parity checks and exports are
excluded. Every control/cold/warm result must match an excluded serial result
in distance, LoD95 and significance f32 bits and complete exported PLY bytes.

| Input / sequence | Control / prepared lanes | Control, ms | Cold, ms | Warm, ms | Paired cold / warm reduction | Faster cold / warm trials |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Autoware repeat](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-exact-trees-final.json) | 1 / 2 | 2,377.95 | 1,644.15 | 1,280.15 | 32.32% / 46.85% | 10/10 / 10/10 |
| [Autoware initial](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-exact-trees-initial.json) | 1 / 2 | 2,907.70 | 1,770.80 | 1,380.30 | 35.18% / 49.64% | 9/10 / 10/10 |
| [PandaSet repeat](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-exact-trees-final.json) | 2 / 3 | 1,180.10 | 901.45 | 584.30 | 21.45% / 50.00% | 10/10 / 10/10 |
| [PandaSet initial](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-exact-trees-initial.json) | 2 / 3 | 868.30 | 668.00 | 447.00 | 17.59% / 43.70% | 9/10 / 10/10 |
| [PandaSet subset repeat](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-exact-trees-subset-final.json) | 4 / 4 | 81.55 | 74.65 | 42.75 | 8.80% / 45.76% | 8/10 / 10/10 |
| [PandaSet subset initial](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-exact-trees-subset-initial.json) | 4 / 4 | 84.40 | 76.05 | 42.35 | 10.15% / 50.26% | 9/10 / 10/10 |
| [Tokyo repeat](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-exact-trees-final.json) | 1 / 2 | 2,020.85 | 1,373.35 | 994.95 | 33.77% / 53.68% | 10/10 / 10/10 |
| [Tokyo initial](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-exact-trees-initial.json) | 1 / 2 | 1,967.70 | 1,338.15 | 1,011.35 | 31.67% / 48.91% | 10/10 / 10/10 |

The eight reports contain 80 timed and 40 warm-up trials. Cold was faster in 75/80 timed trials and warm in 80/80. All 360 control/cold/warm result and PLY comparisons matched the serial output.

Reductions are medians of per-trial paired reductions, not ratios of the
displayed marginal medians. CPU clocks, power, other applications and GC
remain uncontrolled. These measurements justify the default for the tested
workloads; they do not guarantee the same gain on every point distribution
or device. Warm calls cache only trees, never normals or final results.

## Verification and reproduction

Native tests check unsorted hit-coordinate order including nonfinite values,
signed zero, radius boundaries, malformed snapshots, exact f64 result pairs,
source lifetime and complete exports. Browser tests exercise WASM SIMD,
two/four-lane transfers, cyclic tails, contiguous layouts, changed parameters,
concurrent leases, source edits/removal, invalid input, failed imports/searches,
lost worker generations, cleanup and successful retry. Shared k-d tree changes
also receive CPU and GPU distance regression checks.

The selected SIMD build passed 317 core native tests and 25 WASM-crate
native tests, native/WASM Clippy with warnings denied, Rustfmt, TypeScript
and Vite. Browser regression runs passed 22 CPU/M3C2 checks, 47 software-GPU
checks and 47 physical-GPU checks (116 distinct browser checks). Three M3C2
recovery cases were rerun after strengthening finite-distance and changed
core-selection assertions. All eight report provenance, script/artifact
hashes, parity and point-selection controls were checked. Eight CLI guards
including preservation of an existing report passed; 50 local document
links and 1,999 public non-ignored file names/contents were checked.

`web/scripts/benchmark-m3c2-preparation.mjs` accepts supplied files only,
records original/loaded point counts and provenance, and checks source,
manifest, script and production artifact hashes before and after timing.
Existing reports cannot be overwritten. The older resident/synthetic scripts
explicitly set `preparedTrees: false` to isolate their historical controls;
their saved script/artifact hashes continue to describe those revisions.

```powershell
cd web
$env:PATH = "$env:USERPROFILE\.cargo\bin;" + $env:PATH
$env:RUSTFLAGS = '-C target-feature=+simd128'
npm run wasm
npm run build
node scripts/benchmark-m3c2-preparation.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../benchmarks/cpu/m3c2/real-source/windows-pandaset019-exact-trees-reproduce.json
```

The initial sequence used the disabled default with explicit true/false
controls; the final sequence used the adopted default. Both sequences use
the same compute paths and benchmark script. The final worker differs only
in the default switch; WASM and pool-worker artifacts are identical.

Benchmark script SHA-256: `5eff79eb674f43405de5687e021481118fc7e48b7cd6516e99011c14cd2fa787`.

| Artifact | SHA-256 |
| --- | --- |
| `c2c-worker-BvC061ie.js` | `052265de3175a9b664b909a5c5660e504464756abd2fbebb9bfc904c75d4ed60` |
| `ca_wasm_bg-DOqQY9PE.wasm` | `ccf52ac2d75b4d42963701c19686bdd378da36c6dba3034ddc289e3194e9e9e3` |
| `worker-43NThPx8.js` | `8ae4969bc1af906a78e55e4054fd42837a24dc0d9618718a4d82866483a58fe6` |
| `worker-FWw16PsD.js` | `21669a13a7f3bedcfd354f56da16d91f8678bd4127fd56eb65e42594cac21190` |

| Report | SHA-256 |
| --- | --- |
| [windows-autoware-planning-exact-trees-final.json](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-exact-trees-final.json) | `c6a3d5af315ddb146b5af35ead2d708693bca771f56d68f5e921b6600e4962e6` |
| [windows-autoware-planning-exact-trees-initial.json](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-exact-trees-initial.json) | `1d653419a0d0d368a0aa3673f2d54eb6279109098a65be1381e6d9cac5bca334` |
| [windows-pandaset019-exact-trees-final.json](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-exact-trees-final.json) | `0beabf2570206ed7cf39c85f948c0f1c61c12544a6382a48e59b84c2b200d48f` |
| [windows-pandaset019-exact-trees-initial.json](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-exact-trees-initial.json) | `e5ab43e5a965cb761c051ae93305061fa4d284a1770eb397e52f6c1882d9f4ea` |
| [windows-pandaset019-exact-trees-subset-final.json](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-exact-trees-subset-final.json) | `c5d5e417cee7b44e9d23ba5d769f2d490c7b674c98a265675e0bbc6f142ab492` |
| [windows-pandaset019-exact-trees-subset-initial.json](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-exact-trees-subset-initial.json) | `4b97724b44e4405e027852c9c9753238f90bfdf05a2280dadcebdf104cf326f2` |
| [windows-tokyo-takanawadai-exact-trees-final.json](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-exact-trees-final.json) | `83c9b57559ca06143c15e6361fc764405eb761e0feb88e273637413df70902f4` |
| [windows-tokyo-takanawadai-exact-trees-initial.json](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-exact-trees-initial.json) | `3677c2a5fd6639572d3fb733125b178e2b871bf54b3a3b47319c63ae7facec15` |
