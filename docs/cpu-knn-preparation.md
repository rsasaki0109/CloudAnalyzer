# Shared normal/SOR neighbourhood preparation

Parallel normal estimation and statistical outlier removal (SOR) now share one
immutable source cloud's partition indices, region descriptors, exact k-d trees
and Morton query orders. Preparation reuse is enabled by default for eligible
jobs. Normal/PCA fits, boundary queries, mean distances, thresholding and output
indexing are still recalculated; final normals or filtered results are not cached.

Two passes over three measured point clouds found warm whole-RPC paired median
reductions of **8.31–13.40% for normals** and **5.23–9.36% for SOR**, compared with
rebuilding on the same production build. Normals were faster in **57/60 timed
pairs**, SOR in **53/60**. These are reuse-eligible calculations on unchanged
geometry. They do not establish cold-start gains or gains for every cloud size.

This follows the allocation/merge improvements in
[`cpu-knn-join.md`](cpu-knn-join.md); the figures here measure the additional
preparation reuse, with those compact joins enabled in both timing controls.

## Ownership, limits and recovery

- `ca_core::filter::KnnPart::prepared` owns original f64 coordinates, an exact
  k-d tree and one Morton permutation. `KnnPart::normals` calls the same normal
  fitting routine as a fresh fit. Existing non-prepared APIs keep the fresh path.
  `retained_bytes` counts retained vector capacities, including the tree and
  permutation, without allocator overhead or temporary calculation scratch.
- WASM exposes `SorPart.prepared`, `normals` and `retainedBytes`. Pool workers
  hold these immutable parts by job ID, separate from transient SOR jobs.
- `web/src/knn-preparation.ts` captures every part, index array and region
  descriptor synchronously before yielding. Points transfer to the owning lane;
  no raw JavaScript coordinate copy remains in the preparation manager.
- One idle source/layout group is retained under a **128 MiB capacity budget**.
  Eligibility is checked before allocation using `100 * pointCount + 1 MiB`.
  After preparation, actual native capacities plus root index/region byte lengths
  are checked. An oversized group is retired and released after its active
  calculation, rather than retained for subsequent calls.
- Pool policy is unchanged: at most eight workers and at least 100,000 points
  per requested part. Jobs with fewer than two requested parts, or outside the
  conservative budget, use the existing path. k, orientation and SOR ratio do
  not affect immutable preparation and can change between calls.
- A lease protects every group still used by an in-flight calculation, including
  groups displaced by another source. Coordinate edits/removal invalidate all
  matching groups, including displaced ones. A prepared result cannot be applied
  to a changed/deleted source; the request fails with a coordinate-change error.
- Every preparation records exact worker generations. Searches do not create a
  replacement worker for lost state. Failed searches drain all submitted lanes,
  retire the preparation, and recover via the existing fresh path if the source
  is still valid. Cleanup never starts a worker and skips replacement generations.
- Retired groups free only after all leases end. Multiple active source groups,
  source clouds, other caches, scratch, JS objects, allocator overhead and WASM
  high-water marks are outside the idle capacity limit. **128 MiB is not a total
  process-memory cap.**

Normal neighbours at partition borders remain part-local, as in the prior pool
implementation. SOR continues querying every reached part and matches the global
filter. The default scalar k-neighbour and covariance kernels remain selected;
this change does not enable the optional SIMD comparisons.

`knnCache:false` on internal normal/filter requests forces the existing fresh
path for eligible jobs and retires the idle preparation. `knn-cache-state` is an
internal diagnostic request returning the current job, point/part counts,
retained bytes, pending state and reuse counter. No UI performance dial was added.

## Measured-source protocol

[`benchmark-knn-preparation.mjs`](../web/scripts/benchmark-knn-preparation.mjs)
uses the full measured `compared.ply` inputs documented in
[`c2c-real-source.md`](c2c-real-source.md): PandaSet scene 019 (531,908 points),
Autoware planning (878,920), and Tokyo Takanawadai (941,933). Files, decoded f64
coordinates and point counts are unchanged, with `keepEvery=1`. Each report embeds
input/source hashes, attribution and licenses. These are measured-source subsets,
not generated points, independent surveys or accuracy ground truth.

Each method gets two fresh current-build analysis workers with separate clouds
and pools. One sets `knnCache:false`, the other true. Thus the timed controls use
the same JS/WASM artifacts and compact joins. An archived pre-change production
worker computes one untimed complete output per method, then terminates before
timing. Every subsequent pair checks both current paths against that archived
oracle: all normal f32 bits or SOR kept count and byte-identical PLY.

Settings are k=24, upward normals, and SOR ratio=1. Pool startup/loading are
excluded. One excluded first-use pair precedes five excluded alternating warm-up
pairs and ten alternating timed pairs. Whole RPC timing includes search/PCA,
partition/tree/order construction when applicable, packing, messages, joins,
copies and SOR filtered-cloud indexing. Diagnostics, export, validation and result
removal are outside timing. The fresh path must retain no preparation; the warm
path must retain the same job with exact increasing reuse counters, no pending
preparation, and capacities below 128 MiB.

Initial source order was PandaSet → Autoware → Tokyo; the final pass reversed it.
No builds, tests or file scans ran during timing. CPU clocks, power, garbage
collection and other applications were uncontrolled. First-use values are single
observations; no cold-start median or first-use speedup is claimed. No GPU model
or adapter metadata is collected by the benchmark.

## Warm whole-RPC results

Times are separate medians in milliseconds. Reduction is the median of each
pair's `(fresh − prepared) / fresh`, not the ratio of the two separate medians.
Positive means faster. Individual slower pairs remain in the reports and medians.

| Source/pass | Normal medians, fresh / prepared | Normal paired reduction; faster | SOR medians, fresh / prepared | SOR paired reduction; faster |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 449.60 / 398.65 | +9.64%; 8/10 | 470.75 / 436.05 | +8.49%; 8/10 |
| PandaSet final | 512.80 / 441.15 | +13.40%; 10/10 | 526.35 / 472.15 | +7.95%; 9/10 |
| Autoware initial | 572.60 / 533.95 | +8.31%; 10/10 | 622.00 / 621.70 | +5.23%; 9/10 |
| Autoware final | 716.10 / 647.15 | +11.37%; 10/10 | 773.35 / 717.45 | +7.60%; 9/10 |
| Tokyo initial | 657.15 / 590.00 | +10.28%; 9/10 | 732.35 / 684.90 | +7.07%; 8/10 |
| Tokyo final | 734.70 / 655.20 | +9.10%; 10/10 | 848.60 / 763.20 | +9.36%; 10/10 |

Both methods and both passes retained the same capacities per source:

| Source | Parts | Retained bytes | MiB |
| --- | ---: | ---: | ---: |
| PandaSet | 5 | 35,138,912 | 33.51 |
| Autoware | 8 | 58,041,728 | 55.35 |
| Tokyo | 8 | 62,200,604 | 59.32 |

## Validation

- SIMD WASM production build, TypeScript/Vite build and Rust formatting passed.
- Native and wasm32 SIMD Clippy checks passed with warnings denied.
- **365 native tests** passed, including three new preparation tests. Prepared
  normals match fresh f32 bits over empty/tiny clouds, changed k/orientation,
  repeated points, large/UTM coordinates and nonfinite inputs. SOR open order,
  reach masks, f64 means/candidates and cross-query distances match fresh trees.
  Owned source coordinates and constant retained capacity across repeated fits
  are also tested.
- **19 Playwright tests** passed: three pure join/lifetime tests and sixteen
  browser tests, including normal/SOR sharing and parameter changes, concurrent
  fits, source replacement, source edit/removal during a job, worker-loss recovery,
  budgets/generations/drained failures, previous scratch/reuse controls, normal
  export/global SOR comparisons and 1.2M-point SOR. Four GPU-suite regressions
  ran with hardware opt-in, covering f64 verification, overlapping edits/removal,
  exact PLY export and split-boundary candidates.
- Six reports validated **180 warm-up/timed pairs plus 12 excluded first-use
  pairs**, all against both same-build controls and archived production output.
  No real-input precision tolerance is used in these comparisons.
- Nine CLI guards passed: missing/unknown/repeated options, missing values,
  invalid repeats, manifest mismatch, identical archive/current artifacts and
  attempted report overwrite. The existing report was preserved.
- The harness checks input, manifest, script, seven implementation source files
  and both builds' three artifacts before/after timing.

## Reproduction and fingerprints

Run from the repository root with original measured input/manifests and the
preserved pre-change production build from the compact-join step:

```powershell
node web/scripts/benchmark-knn-preparation.mjs --baseline-dist notes/knn-preparation-baseline-20261012/dist --input notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output NEWREPORT.json
```

Default is ten timed pairs; output must not exist. Local snapshots and input files
are ignored. A different checkout needs equivalent preserved artifacts and source
inputs. The archive supplies a precision oracle, not the timed fresh control.
The seven source-file fingerprints and per-trial cache states are embedded in
each report. Historical compact-join reports keep their original fingerprints.

| Artifact | SHA-256 |
| --- | --- |
| Archive `worker-C8RuKzXn.js` | `b50fe839cf6a9de7f7e1313d749da1c21849829886b3b67cc8acd700b2374827` |
| Archive `c2c-worker-Bygl9p01.js` | `74232abf342402a8d2c221a669333055bad0245b2f75f04befdeb34337d4163b` |
| Archive `ca_wasm_bg-B9WDv0qt.wasm` | `390ef24b4453c8179d8fbae994226eff475318de5360a1985fffc8a47542f345` |
| Both timed modes `worker-Bi2ftahU.js` | `e85af60d8505d3f217cfabfb642d88863043d7072ddd11b5d4526819d7fc882c` |
| Both timed modes `c2c-worker-Cm5O4znW.js` | `c2fd1e8a56fcc67a7cc6353fbee6d00d259b5409d5e66319ef26f3a20b8f43a0` |
| Both timed modes `ca_wasm_bg-dx7fFc55.wasm` | `e08cc5c9d9702c1fd09767f0adfd9c6d2c86f249075f99ead53feb13cd88a520` |
| Benchmark script | `291fce708ae975b3873d48c1749f67fb438a74fd9f341d9840ef091493d6ef6e` |

Raw reports and SHA-256 checksums follow.

| Report | SHA-256 |
| --- | --- |
| [windows-autoware-planning-knn-preparation-final.json](../benchmarks/cpu/knn-preparation/real-source/windows-autoware-planning-knn-preparation-final.json) | `302a85b32f7e73f3745563fb4a528a13a58f291b87cafb80f22946b17d2e37cb` |
| [windows-autoware-planning-knn-preparation-initial.json](../benchmarks/cpu/knn-preparation/real-source/windows-autoware-planning-knn-preparation-initial.json) | `cdc4a79a682295b21a350c723bfa164e3eca07582f77e5b2a42bc6ffd993ccf9` |
| [windows-pandaset019-knn-preparation-final.json](../benchmarks/cpu/knn-preparation/real-source/windows-pandaset019-knn-preparation-final.json) | `92407be3bc536f47557dc7114fa46ebc35a49a45cf2a465a2756052c6dd4e805` |
| [windows-pandaset019-knn-preparation-initial.json](../benchmarks/cpu/knn-preparation/real-source/windows-pandaset019-knn-preparation-initial.json) | `dd94de7773aca9628f94b6f53b03e42fe3fc0809f46b461f8944eaa03141813b` |
| [windows-tokyo-takanawadai-knn-preparation-final.json](../benchmarks/cpu/knn-preparation/real-source/windows-tokyo-takanawadai-knn-preparation-final.json) | `1492ca2dd24f915bf4ac92802209ef65d1b45a1daa0b36bd9795987e20bf8145` |
| [windows-tokyo-takanawadai-knn-preparation-initial.json](../benchmarks/cpu/knn-preparation/real-source/windows-tokyo-takanawadai-knn-preparation-initial.json) | `c248c3c6d18bbe177312500b8b9181eec9a1cb849f58039be3417d60af1b271e` |
