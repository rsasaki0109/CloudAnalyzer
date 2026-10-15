# ICP correspondence leaf SIMD

ICP can optionally evaluate two reference-point distances together inside a
k-d-tree leaf using the existing WASM `f64x2` nearest-point kernel. This
experiment preserves the tested results, but whole-call improvements were
inconsistent across the three measured sources. `icpSimd` therefore defaults
to **false**. The retained reference/tree/lazy-normal cache remains enabled
under its existing 128 MiB limit.

## Implementation and exact results

`IcpPrepared::compute_with_simd` and `icp_with_simd` expose the core comparison.
The WASM `IcpPrepared.compute` and `registerIcp` exports accept an optional
SIMD boolean after the point-to-plane flag, defaulting to false. The production worker forwards
the internal request `icpSimd` through both cached and uncached registration.
`IcpOutput.correspondenceSimd` records whether paired leaves were requested
and the WASM build supports them. This is CPU SIMD; this change adds no GPU
dispatch.

The subsequent [temporal-radius experiment](cpu-icp-temporal.md) adds an
independent final optional boolean, also defaulting off. The SIMD benchmark
now pins `icpTemporal: false` to isolate leaf arithmetic; the existing reports
measure the earlier build before that independent control was introduced.

The correspondence loop still transforms samples in their original order,
seeds each query from the preceding sample's nearest hit, and starts the same
root traversal. The existing paired kernel keeps `(dx*dx + dy*dy) + dz*dz`
arithmetic, strict improvement comparisons, the earliest improving tree slot
on ties, and a tied initial guess. Odd leaf tails remain scalar. Native and
non-SIMD WASM targets use scalar arithmetic.

The sampled point set, overlap selection, pair order, RMS reduction, lazy
normal estimation, full point-to-plane matrix, solve and convergence criteria
are unchanged. The option is not part of the retained reference's key:
switching leaf arithmetic preserves reference/tree/normal reuse and its
capacity. It allocates no new per-point buffers.

## Measured decision

Measurements use unchanged full f64 XYZ inputs from the existing
[real-source preparation](c2c-real-source.md), split by original decoded
even/odd indices. There are no generated performance inputs, added offsets,
noise or further input thinning. ICP retains its existing stride sampling
limit of 50,000 moving points, overlap 0.8, maximum ten iterations,
point-to-plane metric and disabled initial centroid matching.

The machine has an i7-9750H, 12 logical CPUs and 32 GiB RAM; Chromium
153.0.8010.12 runs the production SIMD WASM. Current scalar and paired
controls use the same build in independent worker contexts, each retaining
one unchanged reference. Every call loads a fresh moving cloud from the
same measured file before timing. An archived production build supplies an
excluded exact scalar result and transformed-PLY oracle.

After one excluded first-use pair and five excluded alternating warm-up
pairs, ten timed pairs alternate execution order. The table reports whole
RPC wall time, including correspondence search, overlap selection, lazy
normal lookups, fitting, transform application and output indexing.
Load/pool startup, exports, validation and removal are excluded. No
builds, tests or repository scans run alongside timing.

| Source | Reference / moving points | Scalar median, ms | Paired median, ms | Median paired shortening | Paired wins | Iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 284.35 | 292.80 | +0.07% | 5/10 | 9 |
| Autoware planning | 878,921 / 878,920 | 411.15 | 426.85 | −1.58% | 4/10 | 10 |
| Tokyo Takanawadai | 941,933 / 941,933 | 151.70 | 143.35 | +7.24% | 7/10 | 1 |

Paired shortening is the median of `100 * (1 - paired_ms / scalar_ms)` for
each corresponding pair, not the ratio of the two marginal medians.
PandaSet's tiny positive paired median accompanies a slower paired marginal
median and only five wins. Across all sources there are 16 wins in 30 timed
pairs. Tokyo's one-iteration fit has a different cost mix from the other
sources; its result does not establish better scaling or a general SIMD gain.
These observations support retaining the scalar default.

Retained reference capacities are unchanged at 47,871,744 bytes for
PandaSet, 79,102,932 for Autoware and 84,773,988 for Tokyo. These are vector
payload capacities, excluding allocator overhead and temporary queries.
CPU clocks, power, GC and other applications are uncontrolled. The measured
source subsets are not independent surveys or registration ground truth.
This is one measured round with repeated timing samples, not a guarantee
for other CPUs, browsers, acquisitions or ICP settings.

## Validation and reproduction

All 48 current pairs (96 outputs, including first use and warm-ups) preserve
every transform-matrix value, initial/final RMS, iteration count, convergence
flag and transformed PLY byte. Three archived scalar oracles also match.
Each pair checks expected SIMD/cache flags, full input point counts and equal
bounded reference capacities. Input, manifest, original measured source,
seven implementation sources, script and current/archived build hashes are
checked before and after timing; the finished reports were audited again.

The native suite passes 378 tests. The new native test covers exact pair
membership/order and distance bits with duplicates, overlap trimming,
sampling, both metrics, centroid matching, changing moving geometry and
retained-reference mode changes. Actual SIMD WASM tests compare 96 parameter
and geometry cases (five fits each) across local/georeferenced coordinates, duplicate ties, tiny/overflow
scales, coincident and nonfinite geometry. Production-worker tests switch
SIMD controls with cached/uncached references, both metrics and exact PLY
outputs. The two new browser tests and four existing scratch/cache tests
pass. Native/WASM Clippy with warnings denied, formatting, TypeScript/Vite
build, script syntax and seven invalid-option/overwrite controls pass.

- [PandaSet report](../benchmarks/cpu/icp/real-source/windows-pandaset019-correspondence-simd.json)
- [Autoware report](../benchmarks/cpu/icp/real-source/windows-autoware-planning-correspondence-simd.json)
- [Tokyo report](../benchmarks/cpu/icp/real-source/windows-tokyo-takanawadai-correspondence-simd.json)
- [Benchmark script](../web/scripts/benchmark-icp-simd.mjs)

Archive the preceding scalar production `web/dist` before rebuilding, then
run from the repository root:

```powershell
node web/scripts/benchmark-icp-simd.mjs --baseline-dist notes/icp-correspondence-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-icp-simd.json
```

Use the corresponding Autoware/Tokyo preparation directory for the other
sources. Input/manifest and an archived scalar build are mandatory. The
output parent must exist, reports cannot be overwritten, and `--repeats`
accepts 1..100 (default ten). The benchmark explicitly selects both controls,
so it does not require changing the scalar production default.
