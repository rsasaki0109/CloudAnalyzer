# Reusable query scratch and invariant geometry across heavy calculations

This CPU/WASM optimization pass investigates M3C2, normal estimation, statistical
outlier removal (SOR), point-to-plane ICP and Map quality. It reduces redundant
allocation and invariant arithmetic while retaining the existing exact search
and accumulation order. It adds no GPU shader or coordinate approximation. The initial allocation-only
prototype was measured separately and did not establish a consistent whole-call
speedup; its reports are preserved alongside the larger changes below.

## Changes

- The M3C2 comparison computes the projection sphere's `hypot(radius, depth)` and squared
  cylinder radius once when constructing its per-calculation state, rather
  than for each core/projection. Parameters stay immutable; fresh calculations
  with changed projection settings construct fresh geometry. Normal-cache
  reuse still depends on normal fitting only.
- `KdTree.nearest_k_into` retains one caller-owned result buffer. It clears the
  live length, runs the same sorted insertion and strict tie comparisons, then
  converts tree slots to original indices in place. The original allocating
  `nearest_k` API remains available. Capacity is bounded by requested k/input
  length and retained only within the caller's calculation/chunk.
- Normal estimation and SOR reuse that buffer for each query. Native Rayon
  chunks have their own scratch; browser partitions and cross-part SOR queries
  do too. Morton scheduling, neighbourhood membership, tie order, PCA and SOR
  threshold arithmetic stay the same.
- Point-to-plane ICP reuses the k-neighbour result and short coordinate list
  when fitting a previously uncached reference normal. Its existing lazy normal
  cache, sequential correspondence guesses, overlap selection, fit/reduction
  order, transforms and convergence criteria stay the same. Point-to-point
  correspondence arithmetic is unchanged.
- Map quality reuses one 27-neighbour array for SCS instead of allocating one
  for every matched voxel. Gaussian moments, voxel keys, Wasserstein
  eigensolvers and each neighbourhood's enumeration/accumulation order stay
  the same. Gaussian/score maps are rebuilt as before.

These are transient buffers, not new source/reference caches. For k=24, a
neighbour list contains roughly 384 bytes (400 bytes for SOR's k+1); ICP's
12-point coordinate scratch contains 288 bytes. SCS retains capacity for
27 f64 values (216 bytes). Allocator slack and existing trees/heaps/results are
excluded from these capacities; no total process-memory reduction is claimed.

Core comparison APIs select allocating scratch for normals, SOR, ICP and
voxel scores. M3C2 `with_cached_geometry(false)` selects repeated geometry.
The browser's existing request and result protocols are unchanged.

## Larger changes after the allocation-only prototype

The optional WASM SIMD comparison calculates two adjacent k-neighbour leaf distances together, then
inserts them in the original scalar point order. Per-channel covariance updates
combine x/y columns with separate f64 multiply/add; point/accumulation order and
z arithmetic stay the same. Non-finite deviations use scalar covariance, and
NaN paired distances use scalar propagation. Native targets remain scalar. Whole-call measurements were mixed, so paired
neighbour/covariance SIMD and precomputed cylinder geometry stay off by default.
They remain available as explicit comparison kernels.

ICP now optionally retains one owned immutable reference/tree/lazy-normal cache,
with a conservative capacity estimate checked before construction and a 128 MiB
actual vector-capacity ceiling. It computes every new moving-cloud fit afresh;
only the unchanged reference preparation and previously fitted reference normals
are reused. Changing/removing the reference invalidates it; failed computation
releases the entry. `icpCache: false` selects the historical unretained path.

An optional `icpSimd` control also compares paired WASM correspondence leaves
while preserving this reference cache. It defaults off after inconsistent
whole-call measurements; see [ICP correspondence SIMD](cpu-icp-simd.md).
The independent [temporal-radius experiment](cpu-icp-temporal.md) also remains
off after comparison with the preceding production build.
`referenceReused`, `referenceCached` and `referenceBytes` describe this state.

Map quality retains only three scalar voxel scores for one immutable source
pair and voxel/minimum-occupancy/SIMD settings. Fresh nearest distances, exact
statistics and threshold-dependent inlier metrics still run for every combined
request. Coordinate edits/removal invalidate the entry. Scores are captured
synchronously before awaiting distance work; active jobs retain their own
scalar snapshots. `voxelCache: false` recomputes Gaussian scores, and
`voxelCacheHit` reports reuse. Historical voxel/kernel benchmarks explicitly
select false to preserve the computation they compare. Changing voxel settings
recomputes scores. No Gaussian maps or per-point outputs are retained by this
cache.

The subsequent [fixed voxel-reference cache](cpu-voxel-reference.md) provides
a separate bounded layer for Gaussian statistics and lazy covariance roots
when compared identities change. Final-score caching remains independent.

## Measurement protocol

`web/scripts/benchmark-compute-scratch.mjs` compares two immutable production
builds. The baseline is an archived build from immediately before this pass;
the optimized build contains these core changes. Each method gets its own two
analysis-worker contexts, existing CPU pools and identical full measured files.
Pool startup/loading and reusable-state setup are excluded. Five alternating
warm-up pairs precede ten alternating timed pairs per method.

M3C2 retains exact trees but fits fresh normals (0.4 m normal radius, 0.25 m
projection radius, 0.5 m half-depth, 0.25 m original-point core spacing).
Normals/SOR use k=24, with the existing browser partitioning. ICP reloads an
unchanged measured moving cloud before every timed request and uses up to ten
point-to-plane iterations at 0.8 overlap. Map quality keeps scalar moments,
1 m voxels, minimum occupancy 5, threshold 0.5 m and warm bidirectional exact
nearest preparation. No source coordinates are synthesized or thinned.

Wall timings cover the complete calculation and RPC, including output indexing
for SOR and applying the fitted transform for ICP. Loading, export/validation,
cleanup and setup are outside timing. M3C2/SOR/ICP PLY bytes must match; all
normal, distance, LoD95 and significance bits must match. ICP transforms, RMS,
iteration counts and convergence flags must match. Map quality requires exact
nearest-distance bits, f64 statistics and inlier metrics, exact voxel counts,
and the existing 1e-12 relative/absolute HashMap reduction tolerance for AWD/SCS.

Input/manifest, script and both builds' worker/WASM fingerprints are checked
before and after. No builds, tests or scans run concurrently with measurement.
Other applications, CPU clocks, power and garbage collection are uncontrolled.
Two contexts retain independent heaps/caches; existing per-context copy budgets
are not total-memory caps. Measured even/odd source splits are not independent
surveys or registration/accuracy ground truth.

## Allocation-only prototype results

These three reports use the first proposed build, before the SIMD and immutable
reference/score reuse additions. Five excluded warm-up pairs and ten timed
pairs per method matched every result. The median paired percentage reductions
below show why allocation-only changes were insufficient for this task.

| Source | M3C2 | Normals | SOR | ICP | Map quality |
| --- | ---: | ---: | ---: | ---: | ---: |
| [PandaSet 019](../benchmarks/cpu/compute-scratch/real-source/windows-pandaset019-compute-scratch-initial.json) | -1.54% | +0.23% | +0.24% | -1.71% | -1.43% |
| [Autoware planning](../benchmarks/cpu/compute-scratch/real-source/windows-autoware-planning-compute-scratch-initial.json) | -0.12% | +0.39% | +1.82% | -0.66% | +0.62% |
| [Tokyo Takanawadai](../benchmarks/cpu/compute-scratch/real-source/windows-tokyo-takanawadai-compute-scratch-initial.json) | +0.48% | +0.28% | -0.84% | +0.96% | +2.33% |

Positive values mean faster. Every method has ten alternating timed pairs per
source. These are complete RPC timings, not isolated allocator measurements.

## Automatic M3C2 normal reuse

The first prepared calculation with a set of fitting conditions does not allocate
normal-cache records. After a successful calculation, the prepared pair remembers
its normal radius, core spacing, pruning and core scheduling. A repeated fitting
configuration can capture normals within the existing 64 MiB combined/per-lane
budget; later matching calls reuse them. Projection radius/depth changes still
recompute both projection queries, distances and LoD95. Changed fitting settings
start with an uncached fit again. Native cache lookup additionally checks every
ordered core coordinate bit, normal-radius bits and pruning mode.

`cacheNormals: false` disables this policy. `true` retains the explicit capture
control. Omission selects the automatic repeated-fit policy. Failed calculations
do not update the fitting signature; source edits/removal retire the prepared
pair as before. No cache activates for serial/unprepared or oversized-core jobs.
All published earlier M3C2 comparison harnesses keep their explicit false flag.

Normal arrays add at most the existing 64 MiB budget, not a total process-memory
cap. Reference/tree caches and other workflow heaps are accounted separately.

## SIMD/reference-reuse comparison before adoption

The `final` reports combine optional SIMD/cached-geometry kernels with warmed
ICP reference and voxel-score reuse. M3C2 still fits fresh normals. Three rows
below give median paired complete-RPC percentage reductions. Kernel-only
changes remain mixed; the retained-reference/score paths are consistently faster.

| Source | M3C2 | Normals | SOR | ICP | Map quality |
| --- | ---: | ---: | ---: | ---: | ---: |
| [PandaSet 019](../benchmarks/cpu/compute-scratch/real-source/windows-pandaset019-compute-scratch-final.json) | -2.54% | +0.95% | +0.83% | +35.20% | +51.25% |
| [Autoware planning](../benchmarks/cpu/compute-scratch/real-source/windows-autoware-planning-compute-scratch-final.json) | +1.08% | +2.51% | +0.34% | +38.34% | +69.53% |
| [Tokyo Takanawadai](../benchmarks/cpu/compute-scratch/real-source/windows-tokyo-takanawadai-compute-scratch-final.json) | -1.72% | -0.67% | +1.82% | +61.59% | +45.78% |

## Adopted production behavior

The adopted build keeps paired-neighbour/covariance SIMD and invariant-cylinder
geometry comparisons off by default. Caller-owned scalar k-neighbour scratch
and SCS scratch remain available without new coordinate approximations or source
replicas. Whole-call normal/SOR timings vary; no general speedup is claimed for
these allocation-only changes.

The production M3C2 prepared path automatically captures normals only after a
successful matching fitting configuration, then reuses matching records under
64 MiB. ICP retains one immutable reference/tree/lazy-normal set under 128 MiB.
Map quality retains three scalar voxel scores for one immutable pair/settings.
Changing geometry/settings invalidates or misses the appropriate cache. These
measurements concern repeated calculations with unchanged eligible sources;
they do not establish cold-start, changed-source or changed-voxel speedups.

| Source / method | Baseline / adopted median (ms) | Median paired reduction | Adopted faster |
| --- | ---: | ---: | ---: |
| [PandaSet 019](../benchmarks/cpu/compute-scratch/real-source/windows-pandaset019-compute-scratch-adopted.json) / m3c2 | 500.90 / 410.75 | +18.12% | 10/10 |
| [PandaSet 019](../benchmarks/cpu/compute-scratch/real-source/windows-pandaset019-compute-scratch-adopted.json) / normals | 685.70 / 667.40 | +2.99% | 8/10 |
| [PandaSet 019](../benchmarks/cpu/compute-scratch/real-source/windows-pandaset019-compute-scratch-adopted.json) / sor | 720.75 / 720.10 | +0.22% | 5/10 |
| [PandaSet 019](../benchmarks/cpu/compute-scratch/real-source/windows-pandaset019-compute-scratch-adopted.json) / icp | 604.30 / 354.75 | +42.46% | 10/10 |
| [PandaSet 019](../benchmarks/cpu/compute-scratch/real-source/windows-pandaset019-compute-scratch-adopted.json) / map-evaluate | 219.10 / 98.10 | +55.99% | 10/10 |
| [Autoware planning](../benchmarks/cpu/compute-scratch/real-source/windows-autoware-planning-compute-scratch-adopted.json) / m3c2 | 1100.10 / 869.65 | +23.34% | 10/10 |
| [Autoware planning](../benchmarks/cpu/compute-scratch/real-source/windows-autoware-planning-compute-scratch-adopted.json) / normals | 612.70 / 620.15 | -1.06% | 2/10 |
| [Autoware planning](../benchmarks/cpu/compute-scratch/real-source/windows-autoware-planning-compute-scratch-adopted.json) / sor | 1055.20 / 1029.90 | +3.34% | 8/10 |
| [Autoware planning](../benchmarks/cpu/compute-scratch/real-source/windows-autoware-planning-compute-scratch-adopted.json) / icp | 869.15 / 496.10 | +42.80% | 10/10 |
| [Autoware planning](../benchmarks/cpu/compute-scratch/real-source/windows-autoware-planning-compute-scratch-adopted.json) / map-evaluate | 527.50 / 175.20 | +66.53% | 10/10 |
| [Tokyo Takanawadai](../benchmarks/cpu/compute-scratch/real-source/windows-tokyo-takanawadai-compute-scratch-adopted.json) / m3c2 | 863.90 / 670.05 | +22.03% | 10/10 |
| [Tokyo Takanawadai](../benchmarks/cpu/compute-scratch/real-source/windows-tokyo-takanawadai-compute-scratch-adopted.json) / normals | 705.10 / 686.45 | -1.16% | 4/10 |
| [Tokyo Takanawadai](../benchmarks/cpu/compute-scratch/real-source/windows-tokyo-takanawadai-compute-scratch-adopted.json) / sor | 812.45 / 801.50 | +2.30% | 7/10 |
| [Tokyo Takanawadai](../benchmarks/cpu/compute-scratch/real-source/windows-tokyo-takanawadai-compute-scratch-adopted.json) / icp | 438.70 / 149.25 | +64.36% | 10/10 |
| [Tokyo Takanawadai](../benchmarks/cpu/compute-scratch/real-source/windows-tokyo-takanawadai-compute-scratch-adopted.json) / map-evaluate | 285.55 / 151.45 | +46.49% | 10/10 |

M3C2 improves 18.12–23.34%, ICP 42.46–64.36%, and combined Map quality
46.49–66.53% in these adopted runs. Each of those three methods is faster in
30/30 timed pairs across the three inputs. Scalar normal/SOR scratch yields
mixed complete-call medians, with individual regressions; the optional SIMD
variants are not selected automatically. Paired reductions are computed per
pair before taking the median, rather than from the two marginal medians.

The `adopted` reports include excluded first-use observations. These are single
setup observations, not repeated cold-start medians. Timed M3C2 rows require
normal-cache hits; timed ICP rows require reference/tree reuse; timed Map quality
rows require matching voxel-score reuse and still compute fresh nearest
statistics and inlier metrics. Original-point core positions and complete
measured inputs remain unchanged.

All nine reports contain five methods, five excluded warm-up pairs and ten
timed pairs: 675 validated result pairs (450 timed, 225 excluded). M3C2 and
normal/SOR/ICP checks preserve their exact output/PLY contracts. Map voxel
reductions retain the existing 1e-12 tolerance, while nearest bits, f64
statistics, inlier metrics and voxel counts match exactly. The paired files
are source-index subsets, not independent surveys.

Automatic M3C2 normal-record capacities in adopted runs total 11,902,296 bytes
for PandaSet, 39,017,384 for Autoware and 16,786,280 for Tokyo. Retained ICP
vector capacities are respectively 47,871,744, 79,102,932 and 84,773,988 bytes.
These are separate workflow caches; neither budget includes source clouds,
results, other caches, allocator slack or retained WASM pages. Disabling normal
cache use can keep an earlier bounded record set until replacement/retirement;
`normalCacheBytes` reports retained records even on a disabled call.

## Reproduction

From the repository root, supply an immutable baseline production `dist` folder
and a new output path after building SIMD WASM and the current app in `web/`:

```powershell
node web/scripts/benchmark-compute-scratch.mjs --baseline-dist BASELINE_DIST --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output REPORT.json
```

`--methods m3c2,normals,sor,icp,map-evaluate` is the default. CLI guards reject
missing/unknown/repeated options, invalid/repeated methods, invalid repeats and
existing output. Reports are written with exclusive creation. The archived
baseline on the measurement host is `notes/performance-four-baseline-20261012/dist`;
its fingerprints below identify the exact pre-change build. It is ignored local
build output, not a public dependency. Reproduction needs that baseline build or
an equivalent build; current scripts alone do not recreate historical binaries.

## Build and report fingerprints

Earlier report schemas record the script/build versions actually used. Their
script hashes differ from the current adopted harness. All recorded builds,
inputs, manifests and scripts were checked before and after each timing run.
The baseline is identical across all nine reports; proposed/adopted builds differ.

### initial

Script SHA-256: `3136df70d43f3d54df67bcb618054d24d07c62edf7ddb564e59ae98a49ac890d`.

Baseline artifacts:

- `c2c-worker-Cax6DhSg.js`: `933d3a329bd754f713e323071351a303eec955b8483c0159204f9921fd3a575c`
- `ca_wasm_bg-B1Ifysx5.wasm`: `691397806eba68e134e2edb0a0881a8697d175e19afd9ef5025ab62bcdec0d66`
- `worker-Cu9todqF.js`: `aeeffda5952db5f7e642a6b70b58ebc61700876ce462ef274731c8e8670cf664`

Optimized artifacts:

- `c2c-worker-CCZzhqhN.js`: `bce64c7e34b2c0f69e64586f87d6c7b55a7862cd4e1c6298613f5142b9889039`
- `ca_wasm_bg-D2pMyBaP.wasm`: `7abae08e43622cbd6f3732a9c991f924800e9d2166ef18b0fb65869ba2003d95`
- `worker-DtidKgce.js`: `6192c634db09859fe70d1f887f48a9a8196cf3c1060538ae9683972ce2cd8353`

### final

Script SHA-256: `0b50a0315406d7fadee9c7d018e9866d915f00ac9de66a1276749613f3289b70`.

Optimized artifacts:

- `c2c-worker-Cax1TM1i.js`: `fcd439df0c975b43c69f19e66ddb6ee4996b1d693014d292c94a8ff3c0ebb5ce`
- `ca_wasm_bg-CBAyLAnH.wasm`: `0ed9cdf66dac44c8c896e554fe9d306b6417b1d85a02dcc816b56e7f9d41ebd8`
- `worker-B165W5Ci.js`: `8f43dd9a2ac19206f9b866069a25df556d2b335893801bec3ade7039dc9f6d48`

### adopted

Script SHA-256: `f957c77c8b522275ae4f6b36c39c55eebc83a361ebb871a8d390739498d3a15f`.

Optimized artifacts:

- `c2c-worker-Bygl9p01.js`: `74232abf342402a8d2c221a669333055bad0245b2f75f04befdeb34337d4163b`
- `ca_wasm_bg-B9WDv0qt.wasm`: `390ef24b4453c8179d8fbae994226eff475318de5360a1985fffc8a47542f345`
- `worker-C8ygN4MU.js`: `76d0ea3062843b6f4680820b2ed1c62a2234b491724caa1ccbd788c130b99fd0`

Report SHA-256 values:

- `windows-autoware-planning-compute-scratch-adopted.json`: `5a66f7e4ae99b124ceb5ff3132b7e46cf5102a4fecb3cc02f750e8f4ed3101ef`
- `windows-autoware-planning-compute-scratch-final.json`: `ba72da674e35d59502b8dbd9e5d0d2e025b5ab03141e6b3baa9f4cc7b2a0c6a7`
- `windows-autoware-planning-compute-scratch-initial.json`: `b445dcd6e596ae7dc626ae189990b518d0628df3d5c9bf82121da5a8a5f2b53e`
- `windows-pandaset019-compute-scratch-adopted.json`: `166ed60d64bd726815a7df5e3c45a31e622688203f1cc7d98797348baf7e3ac0`
- `windows-pandaset019-compute-scratch-final.json`: `fabdb9654f9a97364e22e2c1870da6b0f3d2803d753fd3f7154264c6b807ff74`
- `windows-pandaset019-compute-scratch-initial.json`: `c061f8cf64486a1c021b4f74d4358b80b82ee119f2695b1ffa8d95662439e6a4`
- `windows-tokyo-takanawadai-compute-scratch-adopted.json`: `db8094cf4c04e545185c0c3d18850be3843bb0f101a331e3d3c3fd4d5cc01371`
- `windows-tokyo-takanawadai-compute-scratch-final.json`: `6e6e43285d0b50b7312046f75feab17316a8212eb6a8debcc9e2e530e808a624`
- `windows-tokyo-takanawadai-compute-scratch-initial.json`: `0e397b9711e84422c1dd1a234f30d01a76d9db61ed18e5a8819d430782d522de`

## Final verification

The adopted production build passes TypeScript/Vite, native and SIMD-WASM
Clippy with warnings denied, and Rustfmt. The final native runs pass 320 core
unit tests, nine existing integration checks, seven new scratch/reference tests
and 26 WASM-crate tests (362 total). All 55 selected final browser checks pass:
44 M3C2/Map quality/CPU/new-kernel and cache checks, seven user-flow checks
including 1.2M-point SOR and pose-graph ICP, and four physical-WebGPU C2C
regressions. There are no skipped tests in these selected final runs.

The standalone WASM checks exercise 120 count/coordinate/k combinations,
comparing allocating scalar and explicit paired/covariance SIMD kernels on
odd tails, ties, tiny/large/UTM/coincident coordinates and non-finite values.
Two normal orientations and exact squared-neighbour distances are checked.
Reference/score/normal reuse tests cover changed settings, projection/inlier
threshold changes, source transforms/removal, cache controls, capacity bounds
and byte-identical ICP exports. Existing worker generation, partial failure,
concurrent source-edit/removal and drainage checks remain valid.

Nine final CLI guard checks preserve existing output. All nine report tables,
precision/parity flags, source/manifest hashes, adopted artifact/script hashes,
report hashes and documentation links are verified after timing. Public
tracked/nonignored filenames and contents omit GPU model names. All builds,
tests and scans are outside every recorded timing interval.
