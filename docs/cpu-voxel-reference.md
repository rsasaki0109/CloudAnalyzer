# Reusing a fixed Map quality voxel reference

Status: adopted with deferred automatic preparation. Fresh voxel calculations
that repeatedly use the same reference/settings retain its Gaussians and lazy
covariance square roots. Changing the compared cloud can reuse these records.
First-use calculation and the existing three-scalar final-score cache keep
their previous paths.

The subsequent [retained voxel neighbours](cpu-voxel-neighbors.md) also reuse
SCS connectivity for this owned reference. That follow-up documents the
current increased payload and stricter preflight; measurements below describe
the preceding Gaussian/root-only build.

On three measured workloads, standalone voxel-score paired RPC reductions
were **39–47%**, with 30/30 faster pairs. Complete Map quality, including fresh
bidirectional nearest preparation, improved by **9–17%**, with 27/30 faster
pairs. These workloads replace the compared Cloud identity using original
measured coordinates; they do not measure new acquisition geometry or an
unchanged pair served by the final-score cache.

## Preparation, precision and lifetime

`ca_core::map_quality::VoxelReference` owns reference Gaussians under the same
floor-key, occupancy, point-order and f64 moment/covariance rules. Each entry
also reserves an optional 3×3 covariance root. The existing `sqrt_psd` computes
it when a compared Gaussian first matches that key; later matches use the
stored f64 matrix. Compared Gaussians, the cross-matrix root, Wasserstein
distances and AWD/SCS are recomputed for every fresh request. No compared
coordinates, per-point distances or final scores are stored in this layer.

Scalar/SIMD moment controls retain their original arithmetic. Individual
Gaussian and Wasserstein calculations preserve tested f64 bits. Matrix
products, trace terms and clamps remain unchanged. Global AWD/SCS reductions
still iterate randomized HashMaps and use their existing
`1e-12 * max(1, abs(a), abs(b))` roundoff allowance. Nearest distances,
statistics/non-voxel metrics and PLY comparisons remain exact.

The browser retains one prepared reference keyed by source object identity
and exact voxel/minimum-occupancy/SIMD settings. Reference edits/removal free
it and clear the repeated-reference marker. Compared edits/removal invalidate
final scores while preserving an unchanged reference. Settings changes replace
the active preparation. Computation is synchronous on the analysis worker;
metadata is copied before awaiting nearest-worker results.

The automatic policy records the reference of a fresh calculation. Its first
occurrence uses the ordinary kernel; a matching later fresh calculation may
capture the reference, and later matches reuse it. A final three-score cache
hit skips native voxel work and allocates no reference preparation. Explicit
`voxelReferenceCache: true` starts capture immediately; `false` bypasses it
without destroying a valid idle reference. `voxelCache` separately controls
final-score reuse. No viewer setting is added.

The retained **payload allowance is 128 MiB**. Preflight uses
`ceil(referencePoints / max(2, minPoints)) * 400 + 4096`. After construction,
`2 * HashMap.capacity() * sizeof(key, Gaussian, optionalRoot) + 4096` must
also fit. The capacity factor/header allowance provide conservative slack;
they do not measure actual allocator usage, temporary allocations or peak
process memory. Roots fill entry storage without growing the map. Exceeded
allowances and preparation/compute errors release the optional preparation
and fall back to the ordinary kernel when it can complete.

Fresh responses include optional `voxelReference`: `cached`, `reused`, payload
allowance `bytes`, `rootsComputed`, `rootsReused` and `fallback`. A final-score
hit omits it because no reference calculation ran. Bypass diagnostics report
actual bypass work even while idle reference memory remains resident.

## Measured inputs and protocol

The [existing real-source manifests](c2c-real-source.md) retain measured f64
coordinates, hashes, attribution and licenses. Reference/compared counts are
531,908/531,908 for PandaSet 019, 878,921/878,920 for Autoware planning, and
941,933/941,933 for Tokyo Takanawadai. Every supplied point loads with
`keepEvery: 1`. No offsets, transformations, new points or further thinning
are added. Tokyo retains its earlier measured source-point selection.
Source-index splits are not independent surveys or accuracy ground truth.

Each calculation loads a new compared Cloud from the same measured file
outside timing and removes it afterward. The reference stays fixed. Both
controls recompute voxel scores with `voxelCache: false`, scalar moments,
1 m voxels and minimum occupancy 5. Fresh sets `voxelReferenceCache: false`;
reuse omits it, exercising automatic preparation. Retained reference memory
stays resident during bypass controls. Each method uses a separate sequential
current context.

| Source | Matched voxels / reused roots per warm call | Payload allowance bytes |
|---|---:|---:|
| PandaSet 019 | 15,650 | 11,472,896 |
| Autoware planning | 56,008 | 45,879,296 |
| Tokyo Takanawadai | 20,122 | 11,472,896 |

Standalone `map-quality`, whole-RPC milliseconds:

| Source | Fresh median | Reference-reuse median | Paired reduction | Faster pairs |
|---|---:|---:|---:|---:|
| [PandaSet](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-voxel-reference.json) | 145.45 | 81.95 | 47.08% | 10/10 |
| [Autoware](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-voxel-reference.json) | 381.85 | 239.90 | 38.66% | 10/10 |
| [Tokyo](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-voxel-reference.json) | 201.15 | 114.80 | 42.23% | 10/10 |

Complete `map-evaluate`, threshold 0.5 m:

| Source | Fresh median | Reference-reuse median | Paired reduction | Faster pairs |
|---|---:|---:|---:|---:|
| PandaSet | 429.75 | 388.85 | 9.06% | 9/10 |
| Autoware | 764.35 | 646.75 | 17.25% | 9/10 |
| Tokyo | 580.00 | 530.60 | 10.19% | 9/10 |

Reduction is `median(100 * (1 - reusedRpc / pairedFreshRpc))`, not the ratio
of marginal medians. Combined requests keep eight exact nearest lanes.
Changing compared identities prevents whole bidirectional pair reuse; both
controls prepare fresh nearest slices under identical input allowances of
80,052,496 / 103,744,548 / 110,153,944 bytes. Gains include this full operation.
Different compared geometry may match previously unseen reference voxels and
need new roots; that performance case is not measured here.

Each case excludes one first observation per mode and five alternating warm-up
pairs, then records ten alternating timed pairs. Warm automatic calls require
reference reuse, zero new roots, one reused root per matched voxel and no
fallback. Initial automatic calls build reference statistics/roots and are
excluded; single observations do not establish cold speedups. All regressions
remain recorded. Autoware combined reuse ranges from 566.3 to 783.9 ms, versus
634.0 to 836.9 ms fresh.

Whole RPC includes Gaussian construction, Wasserstein/SCS, combined nearest
preparation/search, joins/summaries and returned arrays. Loading, startup,
validation, exports and removal are outside timing. No builds/tests/scans ran
alongside timing. Clocks, power, other apps, scheduling, GC, HashMap seeds and
retained WASM pages are uncontrolled.

## Validation and reproduction

All **120 timed current outputs**, 60 warm-up outputs and 12 first observations
pass archived-production comparisons: exact voxel counts, AWD/SCS within the
existing allowance, and exact combined f32 distances, f64 summaries/non-voxel
metrics and full scalar PLY bytes. Six archived oracles use the same inputs.
Relevant source, worker/pool/WASM, input/original-source/manifest and script
hashes are checked before/after; all reports share one current build/script.

**377 Rust tests** passed, including Gaussian/pair bits, changing compared
geometry, input-buffer independence, empty/degenerate/nonfinite matrices,
scalar/SIMD, root counts and retained-size stability. Native/WASM Clippy with
warnings denied, rustfmt, SIMD WASM and TypeScript/Vite builds passed.
**25 distinct browser checks** passed, covering deferred capture, final-score
bypass, edits/removal, settings, scalar/SIMD, allocation preflight/postcheck,
compute-failure fallback and combined distance parity. The existing SIMD test
now distinguishes metrics from diagnostics; its benchmark disables reference
reuse to isolate moment instructions. Synthetic geometry is used only for
correctness checks. Seven invalid/overwrite CLI controls preserve old reports.

Build production WASM/web assets and preserve a pre-change `dist`. Run from
`web`, sequentially:

```powershell
npx playwright test e2e/voxel-reference.spec.ts e2e/compute-scratch.spec.ts e2e/map-quality-simd.spec.ts e2e/map-quality-parallel.spec.ts e2e/map-quality-cache.spec.ts --workers=2
node scripts/benchmark-voxel-reference.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --baseline-dist ../notes/ARCHIVED-dist --output ../notes/NEW-voxel-reference.json
```

Use the existing Autoware/Tokyo directories for other sources. Output parents
must exist; reports cannot be overwritten. `--repeats` accepts 1..100, default
10. [Benchmark script](../web/scripts/benchmark-voxel-reference.mjs).
The local oracle is in ignored `notes/voxel-reference-baseline-20261013`.
