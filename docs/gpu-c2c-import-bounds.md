# Reusing validated subtree bounds in GPU verification imports

The experimental WebGPU C2C worker path now retains the finite subtree extrema
computed while validating its exact tree snapshot. It uses those ranges to
build verification certificates, avoiding a second point scan for subtree
boxes. It also determines finiteness from six root extrema instead of scanning
every imported coordinate again. This is enabled by default for snapshot
imports, alongside [exact tree transfer](gpu-c2c-verifier-setup.md).

The measured benefit is reduced verification-worker preparation time. Complete
C2C RPC timings vary strongly with execution order; these results do not
establish a consistent end-to-end or warm-call speedup. Preparation includes
snapshot export, copies/transfers, validated import and certificate creation,
as well as scheduling delays. It overlaps the first GPU batch and must not be
added to other phase intervals.

`gpuReuseImportBounds: false` retains the historical rebuilding control.
`compute.verifyReuseImportBounds` reports the enabled import option. It is
false when original-XYZ reconstruction is selected (`gpuVerifySnapshot: false`).
The verifier cache distinguishes snapshot/rebuild mode and bounds-reuse mode;
raw-XYZ mode normalizes reuse off. Switching experimental controls can retain
separate reference instances until that source retires. The usual default uses
one instance per lane. Active job leases and worker-generation checks retain
the existing source-edit/removal and failure-recovery rules.

## Validation and arithmetic

The CAKD version-1 snapshot format, point/split bits, original-index permutation,
node topology and partition validation remain unchanged. Bounds are calculated
from original f64 coordinates during validation, rather than trusted as extra
serialized data. The ordinary KdTree reader keeps its existing path without
allocating the optional bounds array.

Total-order extrema and numerical min/max have the same values on finite
coordinates, with possible signed-zero bit differences. Zero comparisons and
squared gaps preserve their numerical values; tests compare exact hit slots,
indices, distance bits and certificate decisions across both methods. Any NaN
or infinity lies at an axis extremum in total order. Thus checking both root
extrema on all three axes establishes the same finiteness condition as scanning
all coordinates. Non-finite trees retain the historical bounds calculation
and complete-search fallback, without certificates.

The final implementation bulk-initializes one bounds array and fills each
validated node's slot. Before allocating it, an impossible node-count header is
rejected using the already enforced balanced splits and minimum internal-range
size. A non-root leaf contains at least half the 16-point leaf size, giving a
conservative maximum of `2 * points / 8 + 1` nodes. Existing structural checks
still run; this additional allocation guard rejects no previously valid tree.

Both paths retain six f64 values (48 bytes) per subtree node, together with
their existing trees, parents, leaf mappings and certificates. No new bounds
cache or serialized payload is added. The allocation occurs earlier during
optimized import; no reduction in total process or peak heap memory is claimed.
Allocator slack, binding/snapshot copies and retained WASM pages remain relevant.

## Real-source measurement

[benchmark-gpu-import-bounds.mjs](../web/scripts/benchmark-gpu-import-bounds.mjs)
requires supplied measured files and a physical WebGPU adapter. It does not
generate coordinates. Source hashes/manifests, original and loaded counts,
thinning factors, attribution and coarse adapter metadata accompany every
report. The point budget loads each supplied input completely (`keepEvery: 1`).
Even/odd source-index subsets are not independent surveys or change ground truth.
Tokyo inputs retain original representatives from an earlier 0.1 m voxel
selection, with no averaged or generated coordinates.

| Input | Reference / compared points | Verification lanes |
| --- | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 4 |
| Autoware planning | 878,921 / 878,920 | 4 |
| Tokyo Takanawadai | 941,933 / 941,933 | 4 |

Each mode loads a fresh reference identity, then runs an excluded serial GPU
calculation to prepare the main tree, device, reference upload and shader.
The timed cold call prepares four pool verifier references; its warm call
reuses them. Both use identical snapshot bytes, query order, bounds/SIMD,
prefetch, adaptive batch size, workgroup size and summaries. Query-buffer growth
when moving from serial to parallel batching remains inside the cold timer.
Loading, removal, pool startup, main reference preparation and parity checks
are excluded. Full timed RPCs include job/query snapshots, required imports,
GPU dispatch/readback, f64 verification, ordered joining and exact summaries.

Five excluded warm-up trials precede ten alternating timed trials per report.
Every serial/cold, cold/warm and cross-mode comparison must match displayed
f32 distance bits, exact reported f64 statistics and certificate counts.
Fallback or scheduling mismatches reject the report. Input, manifest, script
and selected production-artifact hashes are checked before and after timing;
existing output files are never overwritten. No builds/tests/scans ran alongside
benchmark timing. Other applications, CPU/GPU clocks, power, GC and retained
heaps are uncontrolled. GPU model/description/device identifiers are omitted.

Initial reports use per-node bounds-vector growth and retain the original
finiteness scan. Improved reports use bulk initialization, the allocation
guard and root-extrema finiteness checks. Final reports use that same kernel
with the default enabled. Each comparison is paired within one report/build.
Reduction is `median(100 * (1 - retained / rebuilt))`, rather than the ratio of
marginal medians. Times below are milliseconds.

## Worker preparation interval

| Report | Rebuilt median | Retained median | Paired reduction | Faster trials |
| --- | ---: | ---: | ---: | ---: |
| [PandaSet 019 final](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-import-bounds-final.json) | 42.85 | 38.90 | 10.99% | 8/10 |
| [PandaSet 019 improved](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-import-bounds-improved.json) | 48.30 | 38.85 | 18.99% | 10/10 |
| [PandaSet 019 initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-import-bounds-initial.json) | 47.30 | 43.10 | 12.59% | 7/10 |
| [Autoware planning final](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-import-bounds-final.json) | 69.85 | 60.10 | 14.72% | 10/10 |
| [Autoware planning improved](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-import-bounds-improved.json) | 75.60 | 60.05 | 17.49% | 10/10 |
| [Autoware planning initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-import-bounds-initial.json) | 80.15 | 73.65 | 8.59% | 8/10 |
| [Tokyo Takanawadai final](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-import-bounds-final.json) | 79.40 | 65.45 | 18.21% | 10/10 |
| [Tokyo Takanawadai improved](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-import-bounds-improved.json) | 73.60 | 64.65 | 11.68% | 10/10 |
| [Tokyo Takanawadai initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-import-bounds-initial.json) | 80.90 | 71.25 | 14.51% | 10/10 |

The final build reduces preparation by paired medians of 10.99–18.21%, winning
28/30 timed comparisons. Improved and final builds together win 58/60 such
comparisons. Every report's preparation median improves. This preparation
benefit, unchanged retained structures and preserved exact verification justify
enabling the optimized import; whole-call timings are kept separately below.

## Complete cold and warm RPCs

| Report | Cold rebuilt / retained medians | Cold paired reduction | Cold faster | First mode faster | Warm rebuilt / retained medians | Warm paired reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [PandaSet 019 final](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-import-bounds-final.json) | 132.25 / 130.85 | -1.21% | 5/10 | 4/10 | 99.30 / 101.90 | -1.71% |
| [PandaSet 019 improved](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-import-bounds-improved.json) | 154.55 / 127.65 | +9.80% | 5/10 | 4/10 | 101.50 / 96.55 | +7.50% |
| [PandaSet 019 initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-import-bounds-initial.json) | 157.45 / 148.40 | +4.56% | 5/10 | 6/10 | 102.50 / 105.05 | -7.28% |
| [Autoware planning final](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-import-bounds-final.json) | 195.60 / 186.35 | +4.01% | 5/10 | 0/10 | 130.30 / 130.15 | +1.94% |
| [Autoware planning improved](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-import-bounds-improved.json) | 194.45 / 195.10 | -1.12% | 5/10 | 0/10 | 133.65 / 133.95 | +2.00% |
| [Autoware planning initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-import-bounds-initial.json) | 229.30 / 229.30 | +1.35% | 5/10 | 0/10 | 136.00 / 141.55 | -2.17% |
| [Tokyo Takanawadai final](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-import-bounds-final.json) | 203.40 / 197.95 | +5.12% | 5/10 | 0/10 | 125.25 / 119.10 | +2.28% |
| [Tokyo Takanawadai improved](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-import-bounds-improved.json) | 198.85 / 188.35 | +2.92% | 5/10 | 0/10 | 122.55 / 121.50 | -0.44% |
| [Tokyo Takanawadai initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-import-bounds-initial.json) | 227.40 / 209.95 | +9.46% | 7/10 | 2/10 | 132.35 / 133.35 | +3.63% |

Positive reductions mean shorter calls; negative values mean regressions.
All trials and outliers are retained. All three Autoware reports, and the
improved/final Tokyo reports, have faster cold calls in whichever mode ran
second on every timed trial. The report records that order explicitly; no
specific cause is assigned. Alternating order limits imbalance but does not
remove this effect. Full-call medians mix gains and regressions, and the warm
kernel is unchanged, so neither is presented as a stable speedup claim.
Main tree/device preparation is excluded here; these are verifier-preparation
comparisons rather than application startup measurements.

## Correctness checks

Across nine reports and 135 trials including warm-ups, all 675 comparison
edges match displayed distance bits, exact statistics and certificate counts.
Raw f64 checks are separate correctness fixtures rather than real-source
performance exports. Native tests compare original/restored/reused trees,
indices/slots, certificates and snapshot bytes, including signed zeros,
duplicates, subnormals, overflow, NaN/infinite coordinates or queries and
invalid seeds. Malformed headers, impossible padded node counts, index
permutations, topology and partitions reject under both import controls.

The final native run passed 320 core tests, nine integration checks and all
26 WASM-crate tests. Native/WASM SIMD Clippy with warnings denied, Rustfmt,
TypeScript/Vite and diff checks pass. Eight CLI guards reject invalid arguments
or an existing report without altering that report.

The final production build passed 70 browser checks: all 47 physical-WebGPU
checks, 19 M3C2 checks and four CPU C2C/Map-quality SIMD checks. The raw WASM
verification test imports both controls plus the default across 45 precision/
size cases, checks f64 bits and certificate counts with all bounds/SIMD
combinations, and checks owned snapshots/staged queries. Production tests
switch import modes, verify separate caches, run modes concurrently, invalidate
sources, lose workers/devices, reject corrupted snapshots and recover full
results. Export checks preserve the existing CPU PLY output.

## Reproduce

From `web/`, validate the production build before timing, and run reports
sequentially with new output paths. Earlier GPU benchmark scripts explicitly
keep `gpuReuseImportBounds: false` to isolate their recorded comparisons.

```powershell
$env:PATH = (Join-Path $env:USERPROFILE '.cargo\bin') + ';' + $env:PATH
$env:RUSTFLAGS = '-C target-feature=+simd128'
npm run wasm
npm run build
$env:PW_PORT = '4193'
$env:CA_GPU_HARDWARE = '1'
npx playwright test e2e/gpu-c2c.spec.ts --workers=2
npx playwright test e2e/m3c2-radius.spec.ts e2e/m3c2-parallel.spec.ts e2e/m3c2-preparation.spec.ts e2e/c2c-simd.spec.ts e2e/map-quality-simd.spec.ts --workers=2
node scripts/benchmark-gpu-import-bounds.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-import-bounds-reproduce.json
node scripts/benchmark-gpu-import-bounds.mjs --reference ../notes/c2c-real-autoware/reference.ply --compared ../notes/c2c-real-autoware/compared.ply --input-manifest ../notes/c2c-real-autoware/inputs.json --output ../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-import-bounds-reproduce.json
node scripts/benchmark-gpu-import-bounds.mjs --reference ../notes/c2c-real-tokyo/reference.ply --compared ../notes/c2c-real-tokyo/compared.ply --input-manifest ../notes/c2c-real-tokyo/inputs.json --output ../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-import-bounds-reproduce.json
```

## Fingerprints

Script SHA-256: `3f52470e9456bc969c55ec36725f0b7965727a3bd91a1e68a215891dac58006d`.

| Revision | Selected artifact | SHA-256 |
| --- | --- | --- |
| initial | `c2c-worker-Bzmb_ZAF.js` | `c18c8fde2681d32164c7cdb1a08bc7e067ef1a7ab35e88e88668c1c7f6b47491` |
| initial | `ca_wasm_bg-BSifW51k.wasm` | `881e4389a66ecb2ead4888b780f1e5f3cc1657486ec8bf87a9fcacc7a0019c87` |
| initial | `worker-G9bgQBCM.js` | `3b36e51cc409f53f235e9cf7703ba50856263299c4e077a052e946516b5cc7bc` |
| improved | `c2c-worker-DDNYp4tB.js` | `e715d6d25a40b73e44d10a9361712ca5dd97e0c51d39f3a77ee3109d07997f61` |
| improved | `ca_wasm_bg-BeknkLhT.wasm` | `e8e901e10eee478ee2e25f759cbf775a718380c58ec4492efb1bbbc4dcb05071` |
| improved | `worker-CbeWX9oT.js` | `e39ce0f27090202569b75055712e0bda17ce8c6196f89aa46590a17eb8d18924` |
| final | `c2c-worker-Cax6DhSg.js` | `933d3a329bd754f713e323071351a303eec955b8483c0159204f9921fd3a575c` |
| final | `ca_wasm_bg-B1Ifysx5.wasm` | `691397806eba68e134e2edb0a0881a8697d175e19afd9ef5025ab62bcdec0d66` |
| final | `worker-CfJfCpIF.js` | `5dc02b0224486b31e8bf4e8ea499eca4cc44df08a0f6b2930960818348d230de` |

| Report | SHA-256 |
| --- | --- |
| [windows-webgpu-pandaset019-import-bounds-final.json](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-import-bounds-final.json) | `f557c0e794c37c04cd0c1d2da841e36ecb59d1a73b8a856c2bcae71a061faafe` |
| [windows-webgpu-pandaset019-import-bounds-improved.json](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-import-bounds-improved.json) | `d2d39f391b6435b9c4bb64b3503484e9dae2a580b3d9c3372b523c0113954151` |
| [windows-webgpu-pandaset019-import-bounds-initial.json](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-import-bounds-initial.json) | `76049fe2a7442a5a4f641c320ffe49327c1ce4b8918ffd0b8de1f458f54167e4` |
| [windows-webgpu-autoware-planning-import-bounds-final.json](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-import-bounds-final.json) | `e7292e2d27d056a0d2ab62005d9aa8193467e0d23f038685d0a62adfcbf3eb7d` |
| [windows-webgpu-autoware-planning-import-bounds-improved.json](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-import-bounds-improved.json) | `36243d1871dcffcbc907a993cc06b633ad697c1e261ecfe3b23e7dc653a9de2c` |
| [windows-webgpu-autoware-planning-import-bounds-initial.json](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-import-bounds-initial.json) | `4aa201c6cd33ec8ceebade6bcbdd24a3661a8eca9155f975149ecf75f3cefb3f` |
| [windows-webgpu-tokyo-takanawadai-import-bounds-final.json](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-import-bounds-final.json) | `a0939ef10e3cdc4984962077eae3c2b785a09b16e0358e3a4a94bb54431b6820` |
| [windows-webgpu-tokyo-takanawadai-import-bounds-improved.json](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-import-bounds-improved.json) | `6ae78d0e0a3db0589ec08b3a26a8772c3d5b79440cfc68a4af8e7507d06108a3` |
| [windows-webgpu-tokyo-takanawadai-import-bounds-initial.json](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-import-bounds-initial.json) | `19ea3987b2f849f3dfc0aff0a67417d8c2b4fc0d4065f1cf056a3a7dcce26cdf` |
