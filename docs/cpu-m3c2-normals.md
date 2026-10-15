# Bounded exact-normal reuse for M3C2

M3C2 can reuse fitted reference normals while projection radius and depth
change. This extends the [exact tree preparation](cpu-m3c2-preparation.md):
every worker retains one normal set for its ordered cores, and still computes
both cylinders, distance, LoD95, significance and output indexing afresh.
No final distances or result clouds are cached.

The subsequent [paired prepared normal fits](cpu-m3c2-pair.md) accelerate fresh
SIMD-enabled WASM fitting. Cache hits keep the existing reused-normal path;
measurements below describe the earlier normal-cache implementation.

The [automatic repeated-fit policy](cpu-compute-scratch.md#automatic-m3c2-normal-reuse)
now leaves the first calculation uncached, captures normals after a successful
matching fitting configuration, and reuses matching records on later calls.
`cacheNormals: false` disables reuse; `true` requests explicit capture. Omission
selects the automatic policy. It requires the prepared parallel path.
The measurements below retain their original explicit false/true controls;
they predate the automatic policy and shared-search default.

The current default also uses [shared reference searches](cpu-m3c2-shared-search.md)
for fresh fitting. The measurements below retain `sharedSearch: false` to
isolate normal reuse. Shared/separate search gives the same fitted-normal bits
and can switch without invalidating an otherwise matching normal cache.

Matching normal-cache hits now also select
[ordered projection traversal](cpu-m3c2-projection.md) automatically. The
measurements below predate that traversal change and retain their original
controls. Fresh fitting keeps its existing projection default.
Current reruns of the historical normal-reuse benchmark explicitly disable
projection fusion to keep that comparison focused on normal reuse.

## Exact keys and lifetime

`M3c2.normal_at` fits the same reference neighbourhood as the original
`at` method. `at_with_normal` executes its existing projection arithmetic
and query order. The original `at` delegates to both steps, with no changes
to the fitted normal, covariance/PCA, sample order or confidence formula.

The prepared WASM object keeps original f64 core triples and tagged optional
f64 normals. It compares every core coordinate bit in order, normal-radius
bits and the radius-pruning flag before reuse. A numerical equality check,
core count alone or a hash would not suffice. Signed zero and NaN payloads
therefore participate in the exact key. Failed fits and `Some(NaN)` retain
their distinct validity tags. Changing radius, core positions/order or pruning
replaces the set; changing projection parameters can reuse it.

The immutable input trees identify the reference geometry and input direction.
Tree preparation already invalidates on source changes/removal, another pair
or lane count, and lost worker generations. Its active leases also own the
normal cache. Synchronous WASM calls do not hold a borrow across an await.
Main and pool metadata are captured immediately after each call, so another
concurrent radius request cannot overwrite an earlier result's reuse status.
Preparation/search failures still drain all dispatched work and reject the
whole result before releasing the preparation.

Disabling reuse skips the cache and reports `normalCacheHit: false`; an
already saved set can remain resident for a later enabled call. Invalid direct
WASM calls leave the last valid set intact. Enabled calls outside the capacity
limit discard it and compute without saving a replacement. Empty cores save
no set. `M3c2Output.normalCacheBytes` reports retained data on the participating
prepared lanes, and `normalCacheHit` requires a hit on every lane.

## Storage limit

Each saved core uses 24 bytes for coordinates and 32 for its optional normal
on the tested native/WASM targets. The browser allows **64 MiB of this data
per prepared pair**, with a fixed limit per lane:

```text
lane_capacity = floor(64 MiB / (56 * total_lanes))
```

If the longest selected core slice exceeds that capacity, every lane computes
without caching. The preparation helper also clamps its caller's allowance
to this limit. A lane releases its old set before allocating a replacement.
Fixed per-instance limits bound overlapping core layouts independently of
which lane completes first. The native binding additionally caps a directly
used object at 64 MiB. Only one idle tree pair/cache is retained, with older
active leases until their computations drain.

This is additional bounded intermediate data. The preceding 256 MiB estimate
for input-tree construction/transfer stays unchanged and does not include it.
Neither figure is a total application heap cap: resident sources, ordinary
core/result buffers, allocator overhead, retained WASM pages, other workflows
and multiple active preparations remain outside these estimates.

## Measured projection changes

Measurements use full supplied [real source point clouds](c2c-real-source.md)
without generated coordinates or additional thinning. These even/odd subsets
are not independent surveys or change-detection ground truth. Both modes use
the same prepared trees, lane counts and original measured core points.
Normal radius is 0.4 m and core spacing 0.25 m. Profile A uses a 0.25 m
projection radius and 0.5 m half-depth; B multiplies both by 1.4.

Each trial loads a fresh reference identity and prepares trees with normal
reuse disabled outside timers. Controls fit normals afresh for both profiles.
Candidate A captures normals; B reuses them with changed projection parameters.
Each sequence excludes five warm-up trials, then times ten trials with
alternating control/candidate block order. Complete RPC timing includes new
core selection/attributes, binding copies, normal fitting or exact-key checks,
fresh projection, joining and output indexing. Loading, tree preparation,
pool startup, validation and exports are excluded.

| Input / sequence | Lanes | Cold control / saved, ms | Cold paired reduction | Changed-projection control / reused, ms | Warm paired reduction | Faster cold / warm trials |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Autoware repeat](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-normal-reuse-final.json) | 2 | 1,117.40 / 1,074.30 | +2.13% | 1,489.30 / 1,123.70 | 25.94% | 8/10 / 9/10 |
| [Autoware initial](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-normal-reuse-initial.json) | 2 | 1,306.55 / 1,314.90 | -0.93% | 1,666.10 / 1,263.85 | 22.38% | 4/10 / 10/10 |
| [PandaSet repeat](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-normal-reuse-final.json) | 3 | 370.30 / 369.75 | +0.03% | 483.65 / 402.90 | 16.17% | 5/10 / 10/10 |
| [PandaSet initial](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-normal-reuse-initial.json) | 3 | 403.90 / 422.50 | -2.60% | 554.50 / 436.60 | 23.18% | 3/10 / 10/10 |
| [Tokyo repeat](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-normal-reuse-final.json) | 2 | 931.10 / 915.35 | +0.61% | 1,273.00 / 1,041.45 | 21.41% | 6/10 / 10/10 |
| [Tokyo initial](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-normal-reuse-initial.json) | 2 | 1,029.40 / 1,060.70 | -2.95% | 1,465.60 / 1,180.80 | 19.98% | 5/10 / 10/10 |

The six reports contain 60 timed and 30 warm-up trials. Cold was faster in 31/60 timed trials; changed-projection reuse was faster in 59/60. All 360 control/candidate result and PLY comparisons matched their serial profile. The slower warm trial is retained in the reports.

Paired reductions are medians of per-trial reductions, rather than ratios of
the marginal medians. A negative cold reduction means initially saving normals
was slower. All four outputs per trial must match the corresponding excluded
serial profile in distance/LoD95/significance f32 bits and complete PLY bytes.
CPU clocks, power, garbage collection and other applications are uncontrolled.
These results support an option for repeated projection refinement; they do
not justify replacing the default for one-off or changing-normal calculations.

## Validation and reproduction

Tests cover raw f64 pairs, source lifetime, undefined normals, changed
projections, radii and exact core bits/order, budget eviction, failed direct
calls, concurrent different radii, metadata capture, source changes/removal,
failed imports/searches, worker loss/retry and complete exports. A policy
test checks the per-lane capacity boundary without huge source allocations.
Synthetic data is used for correctness cases only.

The selected SIMD build passed seven native core M3C2 tests, all 26 native
WASM-crate tests, native/WASM Clippy with warnings denied, Rustfmt, TypeScript
and Vite. Nineteen browser M3C2 cases passed, with the raw-f64 case rechecked
after strengthening failed-call cache preservation assertions. Four CPU
C2C/Map compatibility cases and two physical-GPU exact-verification/export
cases also passed (25 distinct browser checks). All six measured report
source/manifest/script/artifact hashes, point-selection controls and parity
were verified. Eight CLI guards including existing-report preservation, 52
local document links and 2,007 public file names/contents were checked.

The measured-source-only script checks input/manifest, script and production
artifact hashes before and after timing and refuses existing output paths.
For the other sources, substitute the paths recorded in their manifests.

```powershell
cd web
node scripts/benchmark-m3c2-normals.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../benchmarks/cpu/m3c2/real-source/windows-pandaset019-normal-reuse-reproduce.json
```

Both sequences explicitly compare `cacheNormals: false` and true. The final
sequence follows the fixed per-lane capacity guard; all measured core sets fit
both versions. Compute paths, WASM and pool-worker artifacts are unchanged.

Benchmark script SHA-256: `0a1303ef22635182d7ad004a1821e9cb8a86f8d423e9f0d426cf272e0827eb64`.

| Artifact | SHA-256 |
| --- | --- |
| `c2c-worker-BodzXnvN.js` | `0ce1d3a5783c8dfc539b298432b2603019bb8959b8543247c8ef0377174b9fd6` |
| `ca_wasm_bg-DpLDQmEf.wasm` | `590b3ad7196918077e73de86449be1a2d62c530018f88ff768a205ef8130677d` |
| `worker-CCukDfe2.js` | `6abf26ecf58831251ee0c772982fa7331aecdd3cf1bf08733f56aa7a09dcd051` |
| `worker-CfGC6YMV.js` | `973d6c72b956f2d172133a8ed04ef3bc762a0b509cdd0a0bb7d243eea72dde28` |

| Report | SHA-256 |
| --- | --- |
| [windows-autoware-planning-normal-reuse-final.json](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-normal-reuse-final.json) | `f2f662cc6f8dc7cb8c4259e75c7b69a836ce91c37ac974706f39ab4e90fde6de` |
| [windows-autoware-planning-normal-reuse-initial.json](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-normal-reuse-initial.json) | `53ed968622a3e308d1b8fa93cf348be0604ad35ffe8c67040cb659c5f6397cde` |
| [windows-pandaset019-normal-reuse-final.json](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-normal-reuse-final.json) | `e5d28be2bb4e54f4216515caccfb6daf81ad6f6c378319a133a6e66a46922aa4` |
| [windows-pandaset019-normal-reuse-initial.json](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-normal-reuse-initial.json) | `20c90b3c04f3b05920925baf1a0a5a6bb2223c488a56d74d0d9e60b5d9082812` |
| [windows-tokyo-takanawadai-normal-reuse-final.json](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-normal-reuse-final.json) | `87c128249d06e3e7972f9375851cf220af5ef6c83b810cd281fca1ab334b2c4a` |
| [windows-tokyo-takanawadai-normal-reuse-initial.json](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-normal-reuse-initial.json) | `f42dc89d4dfc12bdfdeabe313bd66fbe25cdcfe0b12b7cb243257d3e2721723c` |
