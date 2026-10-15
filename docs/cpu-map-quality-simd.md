# Optional WASM SIMD for Map quality voxel moments

Map quality can accumulate independent Gaussian moment components in WASM
SIMD lanes. Measured-source comparisons showed mixed effects on the full
Map quality request, so the browser and WASM binding keep scalar moments
by default. `voxelSimd: true` is an internal comparison option on both
`map-quality` and `map-evaluate`; `mapQuality(..., true)` selects it directly
in the WASM API. It does not add a viewer setting.

Current reruns explicitly disable the subsequent
[reference Gaussian/root cache](cpu-voxel-reference.md) so that this
comparison continues to isolate moment instructions. Historical report
script/build fingerprints describe their recorded versions.

## Arithmetic and memory

Each voxel still accumulates its count, three coordinate sums and nine
outer-product sums in the original input-point order. The optional kernel
updates x/y channels together with Rust's
[WASM SIMD intrinsics](https://doc.rust-lang.org/core/arch/wasm32/): separate
`f64x2_mul` and `f64x2_add`, with scalar z updates. It performs no horizontal
sum, point-order change, fused multiply-add or f32 conversion. The scalar
control and native targets retain the original scalar accumulation. Voxel
keys, minimum occupancy, Gaussian covariance conversion, Wasserstein
eigensolvers, AWD/SCS reductions and empty-result NaNs use the same code.

The per-voxel sum layout, allocations and retained nearest preparation are
identical between controls. SIMD adds no coordinate/index transfers,
worker replicas, Gaussian cache or retained score arrays. The existing
Map quality copy budget and CPU source invalidation/active leases apply
unchanged. Both controls still compute fresh nearest distances and voxel
scores, with the same default
[cold/warm CPU dispatch](cpu-preparation-pipeline.md).

Existing `voxel_gaussians` and `voxel_scores` core APIs remain scalar.
The new `voxel_gaussians_with_simd` and `voxel_scores_with_simd` functions
select the optional kernel on wasm32 builds with `simd128`; other targets
fall back to scalar. Selection occurs outside the point loop through
constant-generic implementations.

## Reproducing measured-source comparisons

```powershell
cd web
npm run wasm
npm run build
node scripts/benchmark-map-quality-simd.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --repeats 10 --output ../benchmarks/cpu/map-quality/real-source/windows-pandaset019-voxel-simd-new.json
```

Choose a fresh report filename: overwrites are rejected. The default
`--method both` compares standalone voxel scores and combined Map quality;
`--method voxel-scores` or `--method map-quality` limits the run. Threshold,
voxel size and minimum occupancy are configurable. Arguments, manifests,
point counts and input hashes are checked; both inputs must be loaded in
full without thinning.

The unchanged [real-source even/odd splits](c2c-real-source.md) have
531,908/531,908 points for PandaSet scene 019, 878,921/878,920 for Autoware
Planning and 941,933/941,933 for Tokyo Takanawadai. They are disjoint
source-index subsets of one measured geometry, with possible geometric
duplicates, not independent surveys or accuracy ground truth. No offsets,
noise or generated performance fixtures are added.

One production analysis worker and its existing pool are used, without
rendering or a second pool. Every workflow warms five scalar/SIMD pairs,
then measures ten sequential pairs with alternating order. Standalone
voxel scores reconstruct both Gaussian maps and compute fresh AWD/SCS.
Combined Map quality reuses the same warm exact nearest preparation in
both modes, retaining scalar searches, eight lanes, queued dispatch,
progressive joins, adaptive summaries and original f32 inlier arithmetic.
There is no Gaussian or result cache. The comparison changes only the
moment accumulation instruction path.

The first combined warm-up pair includes nearest preparation in its first
mode; it is excluded from the timed comparisons. Final reports additionally
verify that production WASM `target_features` declares `simd128` before
starting the browser.

Wall times include computation and worker RPCs; combined worker durations
are recorded separately. Pool startup/warming, file loading, validation and
exports are excluded. Builds/tests do not overlap the timing trials. Other
applications, CPU power/clocks, GC and HashMap seeds remain uncontrolled;
comparisons apply within each run, not between absolute durations from
different runs. These measurements cover warm source pairs, not cold
browser/process startup.

Every pair requires the same voxel count and only the existing
`1e-12 * max(1, abs(a), abs(b))` HashMap reduction allowance for AWD/SCS.
Combined pairs additionally require identical displayed f32 distance bits,
f64 summary fields, inlier metrics, lane count and retained input bytes.
Reports capture the source/manifest, main worker, pool worker, WASM and
benchmark script hashes, browser and machine metadata.

## Measured results

Standalone voxel-score wall times:

| Source/run | Scalar median | SIMD median | Median paired reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| [PandaSet initial](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-voxel-simd.json) | 137.95 ms | 133.35 ms | 2.48% | 6/10 |
| [PandaSet final](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-voxel-simd-final.json) | 112.75 ms | 114.15 ms | 1.14% | 6/10 |
| [Autoware initial](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-voxel-simd.json) | 332.95 ms | 306.00 ms | 5.14% | 8/10 |
| [Autoware final](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-voxel-simd-final.json) | 346.25 ms | 336.05 ms | 9.22% | 6/10 |
| [Tokyo initial](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-voxel-simd.json) | 235.20 ms | 227.05 ms | 6.61% | 7/10 |
| [Tokyo final](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-voxel-simd-final.json) | 228.90 ms | 208.35 ms | -0.49% | 5/10 |

Combined warm Map quality wall times, from the same reports:

| Source/run | Scalar median | SIMD median | Median paired reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 213.05 ms | 245.85 ms | -5.51% | 1/10 |
| PandaSet final | 200.30 ms | 195.85 ms | 1.88% | 6/10 |
| Autoware initial | 506.60 ms | 475.80 ms | 3.01% | 6/10 |
| Autoware final | 455.45 ms | 447.55 ms | 2.05% | 8/10 |
| Tokyo initial | 373.00 ms | 349.50 ms | 7.82% | 9/10 |
| Tokyo final | 358.55 ms | 341.55 ms | 3.03% | 8/10 |

The paired column is `median(1 - simd_i / scalar_i)` over each run's ten
alternating timed pairs, not a ratio of the two marginal medians. SIMD is
not reliably faster for the full request: the PandaSet initial run is slower
in nine of ten combined pairs despite its modest standalone reduction.
The option remains disabled by default, without a point-count rule inferred
from these three geometries.

Standalone and combined workflows each have 38/60 faster pairs across both
runs. Final Tokyo standalone illustrates the remaining variation: its
marginal medians favor SIMD by 8.98%, yet its median paired reduction is
-0.49% and only half its pairs are faster. Final PandaSet standalone has
the opposite signs in its marginal and paired comparisons. These results
do not support a general speedup or enabling the option by default.

All 120 timed and 60 warm-up comparisons across the six reports match
voxel counts. The 60 timed and 30 warm-up combined comparisons preserve
all displayed distance bits, f64 statistics and inlier metrics exactly.
Maximum AWD/SCS difference is `9.992007221626409e-16`, within the existing
HashMap reduction allowance. Both combined controls retain identical
input payloads and use eight pool lanes.

Production artifacts used by the comparisons:

- Main worker: `worker-Cq-67UN_.js`, SHA-256
  `7790f70d6eeeee226017bed581fff8f89074ab20127a2f5cf0a067e04e5092be`.
- Pool worker: `c2c-worker-Cv--66xs.js`, SHA-256
  `378132d06c6c15fd4268fc553ab68f833b6da0e02296742b3f7d40be46144096`.
- WASM: `ca_wasm_bg-T5KOoA5N.wasm`, SHA-256
  `09a2e4ed10370ed258876ee2347a65c009064e6deddf8275d214c3095b26ebcd`.
- Initial benchmark script SHA-256:
  `1116693402a3ea3a1a2f875ff4d05d03fd81d28d434a775daccbb3b69bce0a63`.
- Final benchmark script SHA-256:
  `f64da2efa790bb6384f6c81020171cb6068ca2729f9579190ee4f669fb2b5d96`.

The script changes add the SIMD-build metadata preflight and clarify the
first untimed combined warm-up pair. They do not change the two timed
computation controls. Initial reports retain their captured script hash
and shorthand note; they used the same SIMD production WASM and excluded
all five warm-up pairs from the timed results.

## Validation

The native Gaussian test checks moment-derived mean/covariance bits across
UTM coordinates, occupancy controls, signed zero and nonfinite points;
native targets exercise scalar fallback. The direct production-WASM test
exercises the SIMD path across 144 combinations of small inputs, UTM and
extreme finite scales, voxel sizes and occupancy thresholds, including no
valid matching voxels. Empty input files are rejected by the parser;
empty-score behavior is tested through occupancy filtering.

The production worker test compares SIMD, scalar and omitted controls on
small serial and retained parallel jobs, changed thresholds/voxel sizes,
and an uncached collected control. Standalone voxel requests match the
combined request. Existing cache, source-lifetime, summary, M3C2 and
software/physical WebGPU tests cover the shared WASM and pool. The related
Map quality/M3C2 UI tests exercise the production flow.

The final production build, three native voxel-kernel tests and all 22
WASM-crate tests passed. Native/wasm32 Clippy and Rust formatting passed.
The final browser runs passed 114 distinct checks: 40 CPU/Map quality,
36 software WebGPU, 36 physical WebGPU and two UI cases. Four invalid CLI
combinations and an overwrite attempt were rejected. All six reports
match the unchanged input/manifest hashes and production artifacts; the
three final reports additionally verify `simd128` build metadata and the
final benchmark script hash. Script syntax, documentation links and
public filename/content checks passed.

```powershell
cd rust
cargo test -p ca-core map_quality::tests
cargo test -p ca-wasm --lib
cargo clippy -p ca-core -p ca-wasm --all-targets -- -D warnings
cargo clippy -p ca-core -p ca-wasm --target wasm32-unknown-unknown --lib -- -D warnings
cargo fmt --all -- --check
cd ../web
npm run build
$env:PW_PORT='4193'
npx playwright test e2e/map-quality-simd.spec.ts e2e/map-quality-cache.spec.ts e2e/map-quality-parallel.spec.ts e2e/c2c-cache.spec.ts e2e/c2c-streaming.spec.ts e2e/c2c-simd.spec.ts e2e/m3c2-parallel.spec.ts e2e/distance-summary.spec.ts e2e/gpu-c2c.spec.ts --workers=1
npx playwright test e2e/app.spec.ts --grep 'map quality:|M3C2: a flat grid' --workers=1
$env:CA_GPU_HARDWARE='1'
npx playwright test e2e/gpu-c2c.spec.ts --workers=1
```
